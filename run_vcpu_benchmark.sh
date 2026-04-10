#!/usr/bin/env bash
# run_vcpu_benchmark.sh — run inside the guest VM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"

_usage() {
    cat <<'EOF'
Usage: run_vcpu_benchmark.sh <config_name> [OPTIONS]

Arguments:
  config_name         Name for this run; results go to results/<config_name>/

Options:
  --tests LIST        Comma-separated list of tests to run (default: all).
                      Available tests:
                        cpu-single    sysbench single-thread CPU
                        cpu-all       sysbench multi-thread CPU (all vCPUs)
                        stream        STREAM memory bandwidth (+ per-NUMA if >1 node)
                        mem-latency   pointer-chase memory latency curve
                        gpu-compute   PyTorch matmul GPU compute benchmark
                        gpu-bandwidth GPU↔CPU memory bandwidth (bandwidthTest / HIP)
                        nccl          NCCL/RCCL all_reduce multi-GPU bandwidth
  --nccl-max-msg SIZE Max message size for NCCL all_reduce (default: 1G)
  --help              Show this help and exit

Examples:
  run_vcpu_benchmark.sh myconfig
  run_vcpu_benchmark.sh myconfig --tests cpu-single,cpu-all
  run_vcpu_benchmark.sh myconfig --tests gpu-compute,gpu-bandwidth,nccl --nccl-max-msg 512M
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    _usage; exit 0
fi

CONFIG_NAME="${1:?$(echo 'error: config_name is required'; echo; _usage)}"
NCCL_MAX_MSG="1G"
TESTS_FILTER=""

shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nccl-max-msg) NCCL_MAX_MSG="$2"; shift 2 ;;
        --tests)        TESTS_FILTER="$2"; shift 2 ;;
        --help|-h)      _usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Returns 0 (run) if TESTS_FILTER is empty or contains the given test name
_should_run() {
    [[ -z "$TESTS_FILTER" ]] || \
        echo ",$TESTS_FILTER," | grep -q ",${1},"
}

RESULTS_DIR="results/${CONFIG_NAME}"
mkdir -p "$RESULTS_DIR"

# ── Ensure CUDA/ROCm bin dirs are on PATH ─────────────────────────────────────
for _bin in /usr/local/cuda/bin /usr/local/cuda-*/bin /opt/rocm/bin /opt/rocm-*/bin; do
    [[ -d "$_bin" && ":$PATH:" != *":$_bin:"* ]] && export PATH="$_bin:$PATH"
done
unset _bin

# ── GPU vendor detection ──────────────────────────────────────────────────────
GPU_VENDOR="none"
if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null 2>&1; then
    GPU_VENDOR="nvidia"
    GPU_COUNT=$(nvidia-smi -L | wc -l)
elif command -v rocm-smi &>/dev/null && rocm-smi &>/dev/null 2>&1; then
    GPU_VENDOR="amd"
    GPU_COUNT=$(rocm-smi --showid 2>/dev/null | grep -oP 'GPU\[\K\d+(?=\])' | sort -un | wc -l)
fi

echo "=== Config: $CONFIG_NAME ==="
echo "vCPUs: $(nproc), GPU vendor: ${GPU_VENDOR}, GPU count: ${GPU_COUNT:-0}"

# ── System topology ───────────────────────────────────────────────────────────
numactl --hardware > "$RESULTS_DIR/numa_topology.txt"
lscpu             > "$RESULTS_DIR/cpu_topology.txt"
grep -i huge /proc/meminfo > "$RESULTS_DIR/hugepages.txt"

NUMA_NODES=$(numactl --hardware | awk '/^available:/{print $2}')

# ── metadata.json — machine-readable run context ──────────────────────────────
GPU_VENDOR="$GPU_VENDOR" GPU_COUNT="${GPU_COUNT:-0}" \
CONFIG_NAME="$CONFIG_NAME" NCCL_MAX_MSG="$NCCL_MAX_MSG" \
NUMA_NODES="${NUMA_NODES:-1}" RESULTS_DIR="$RESULTS_DIR" \
"$PYTHON" - <<'EOF'
import json, os, re, subprocess, datetime

vendor   = os.environ["GPU_VENDOR"]
count    = int(os.environ["GPU_COUNT"])
config   = os.environ["CONFIG_NAME"]
msg      = os.environ["NCCL_MAX_MSG"]
numa     = int(os.environ.get("NUMA_NODES") or 1)
rdir     = os.environ["RESULTS_DIR"]

devices = []
if vendor == "nvidia":
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True)
    devices = [l.strip() for l in r.stdout.splitlines() if l.strip()]
elif vendor == "amd":
    r = subprocess.run(["rocm-smi", "--showproductname"],
                       capture_output=True, text=True)
    devices = re.findall(r"Card series:\s+(.+)", r.stdout)

hugepages = 0
try:
    for line in open("/proc/meminfo"):
        if line.startswith("HugePages_Total:"):
            hugepages = int(line.split()[1])
except OSError:
    pass

meta = {
    "config":         config,
    "timestamp":      datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "vcpus":          os.cpu_count(),
    "numa_nodes":     numa,
    "gpu_vendor":     vendor,
    "gpu_count":      count,
    "gpu_devices":    devices,
    "hugepages_total": hugepages,
    "nccl_max_msg":   msg,
}
with open(f"{rdir}/metadata.json", "w") as f:
    json.dump(meta, f, indent=2)
print(f"  metadata.json written to {rdir}/")
EOF

# 1. CPU single-thread
if _should_run cpu-single; then
    sysbench cpu --threads=1 --time=30 run 2>&1 | tee "$RESULTS_DIR/cpu_single.txt"
fi

# 2. CPU all-threads
if _should_run cpu-all; then
    sysbench cpu --threads=$(nproc) --time=30 run 2>&1 | tee "$RESULTS_DIR/cpu_all.txt"
fi

# 3. Memory bandwidth
if _should_run stream; then
    OMP_NUM_THREADS=$(nproc) ./stream 2>&1 | tee "$RESULTS_DIR/stream.txt"

    # 3a. Per-NUMA-node memory bandwidth (only when >1 NUMA node)
    if [[ "${NUMA_NODES:-1}" -gt 1 ]]; then
        OMP_NUM_THREADS=$(nproc) numactl --membind=0 ./stream \
            2>&1 | tee "$RESULTS_DIR/stream_numa_local.txt"
        OMP_NUM_THREADS=$(nproc) numactl --membind=1 ./stream \
            2>&1 | tee "$RESULTS_DIR/stream_numa_remote.txt"
    fi
fi

# 4. Memory latency
if _should_run mem-latency && [[ -x ./mem_latency ]]; then
    ./mem_latency 2>&1 | tee "$RESULTS_DIR/mem_latency.txt"
fi

# 5. GPU compute (run before bandwidth to warm GPU to operating temperature)
if _should_run gpu-compute; then
    "$PYTHON" matmul_bench.py 2>&1 | tee "$RESULTS_DIR/gpu_compute.txt"
fi

# 6. GPU bandwidth — 5 iterations, all runs saved; report parser takes the best
# GPU is already warm from matmul; multiple runs guard against PCIe/boost variance.
GPU_BW_RUNS=5
if _should_run gpu-bandwidth && [[ "$GPU_VENDOR" == "nvidia" ]]; then
    {
        for _i in $(seq 1 $GPU_BW_RUNS); do
            echo "# run ${_i}"
            ./bandwidthTest --memory=pinned 2>&1
        done
    } | tee "$RESULTS_DIR/gpu_bandwidth.txt"
elif _should_run gpu-bandwidth && [[ "$GPU_VENDOR" == "amd" ]]; then
    # rocm-bandwidth-test / TransferBench crash on some AMD GPU generations (e.g. MI350X).
    # Fall back to gpu_bandwidth_bench.py which uses PyTorch/HIP directly.
    if command -v rocm-bandwidth-test &>/dev/null && \
       rocm-bandwidth-test 2>&1 | grep -q "GB/s"; then
        {
            for _i in $(seq 1 $GPU_BW_RUNS); do
                echo "# run ${_i}"
                rocm-bandwidth-test 2>&1
            done
        } | tee "$RESULTS_DIR/gpu_bandwidth.txt"
    else
        {
            for _i in $(seq 1 $GPU_BW_RUNS); do
                echo "# run ${_i}"
                "$PYTHON" "$SCRIPT_DIR/gpu_bandwidth_bench.py" 2>&1
            done
        } | tee "$RESULTS_DIR/gpu_bandwidth.txt"
    fi
fi

# 7. Multi-GPU collective bandwidth
# -w 10   : warmup iterations (covers NCCL connection setup + GPU clock ramp)
# -n 100  : timed iterations (reduces variance especially at small message sizes)
# -e 1G   : large max message ensures true PCIe/NVLink plateau is reached
# NCCL_ALGO/PROTO: pin to Ring+Simple for reproducible large-message results
if _should_run nccl && [[ "${GPU_COUNT:-0}" -gt 1 ]]; then
    # Collect unique NUMA nodes for all GPUs via PCIe sysfs.
    # nvidia-smi returns 00000000:XX:YY.Z; sysfs uses 0000:XX:YY.Z — strip leading 4 zeros.
    # Returns sorted unique node numbers, one per line.
    _gpu_numa_nodes() {
        local bdfs=()
        if [[ "$GPU_VENDOR" == "nvidia" ]]; then
            while IFS= read -r bdf; do
                bdfs+=("$(echo "$bdf" | tr '[:upper:]' '[:lower:]' | sed 's/^0000//')")
            done < <(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null)
        elif [[ "$GPU_VENDOR" == "amd" ]]; then
            while IFS= read -r bdf; do
                bdfs+=("$bdf")
            done < <(rocm-smi --showbus 2>/dev/null \
                     | grep -oP '[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]')
        fi
        local nodes=()
        for bdf in "${bdfs[@]}"; do
            local node
            node=$(cat "/sys/bus/pci/devices/${bdf}/numa_node" 2>/dev/null || echo -1)
            [[ "$node" -ge 0 ]] && nodes+=("$node")
        done
        if [[ ${#nodes[@]} -eq 0 ]]; then
            echo 0
        else
            printf '%s\n' "${nodes[@]}" | sort -un
        fi
    }

    mapfile -t GPU_NUMA_NODES < <(_gpu_numa_nodes)
    UNIQUE_GPU_NODES=${#GPU_NUMA_NODES[@]}

    if [[ "$UNIQUE_GPU_NODES" -eq 1 ]]; then
        # All GPUs on the same NUMA node — bind tightly for lowest latency
        NUMA_NODE="${GPU_NUMA_NODES[0]}"
        echo "=== NCCL/RCCL: all GPUs on NUMA node ${NUMA_NODE}, binding cpus+mem ==="
        NUMACTL_ARGS="--cpunodebind=${NUMA_NODE} --membind=${NUMA_NODE}"
    else
        # GPUs span multiple NUMA nodes — interleave memory across all GPU nodes
        NODES_CSV=$(IFS=,; echo "${GPU_NUMA_NODES[*]}")
        echo "=== NCCL/RCCL: GPUs span NUMA nodes [${NODES_CSV}], using --interleave ==="
        NUMACTL_ARGS="--interleave=${NODES_CSV}"
    fi

    NCCL_ALGO=Ring NCCL_PROTO=Simple \
    HSA_NO_SCRATCH_RECLAIM=1 \
    NCCL_P2P_DISABLE=1 \
    numactl $NUMACTL_ARGS \
        ./build/all_reduce_perf \
            -b 8 -e "$NCCL_MAX_MSG" -f 2 \
            -g "$GPU_COUNT" \
            -w 10 -n 100 \
        2>&1 | tee "$RESULTS_DIR/nccl_allreduce.txt"
fi

# ── summary.csv — machine-readable benchmark results ─────────────────────────
"$PYTHON" "$SCRIPT_DIR/generate_report.py" --write-summary "$RESULTS_DIR"
