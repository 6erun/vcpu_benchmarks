#!/usr/bin/env bash
# run_vcpu_benchmark.sh — run inside the guest VM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"

CONFIG_NAME="${1:?usage: $0 <config_name> [--nccl-max-msg SIZE]}"
NCCL_MAX_MSG="1G"

shift
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nccl-max-msg) NCCL_MAX_MSG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

RESULTS_DIR="results/${CONFIG_NAME}"
mkdir -p "$RESULTS_DIR"

# ── GPU vendor detection ──────────────────────────────────────────────────────
GPU_VENDOR="none"
if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null 2>&1; then
    GPU_VENDOR="nvidia"
    GPU_COUNT=$(nvidia-smi -L | wc -l)
elif command -v rocm-smi &>/dev/null && rocm-smi &>/dev/null 2>&1; then
    GPU_VENDOR="amd"
    GPU_COUNT=$(rocm-smi --showid 2>/dev/null | grep -c "GPU\[" || echo 1)
fi

echo "=== Config: $CONFIG_NAME ==="
echo "vCPUs: $(nproc), GPU vendor: ${GPU_VENDOR}, GPU count: ${GPU_COUNT:-0}"

# ── System topology ───────────────────────────────────────────────────────────
numactl --hardware > "$RESULTS_DIR/numa_topology.txt"
lscpu             > "$RESULTS_DIR/cpu_topology.txt"
grep -i huge /proc/meminfo > "$RESULTS_DIR/hugepages.txt"

# 1. CPU single-thread
sysbench cpu --threads=1 --time=30 run 2>&1 | tee "$RESULTS_DIR/cpu_single.txt"

# 2. CPU all-threads
sysbench cpu --threads=$(nproc) --time=30 run 2>&1 | tee "$RESULTS_DIR/cpu_all.txt"

# 3. Memory bandwidth
OMP_NUM_THREADS=$(nproc) ./stream 2>&1 | tee "$RESULTS_DIR/stream.txt"

# 3a. Per-NUMA-node memory bandwidth (only when >1 NUMA node)
NUMA_NODES=$(numactl --hardware | awk '/^available:/{print $2}')
if [[ "${NUMA_NODES:-1}" -gt 1 ]]; then
    OMP_NUM_THREADS=$(nproc) numactl --membind=0 ./stream \
        2>&1 | tee "$RESULTS_DIR/stream_numa_local.txt"
    OMP_NUM_THREADS=$(nproc) numactl --membind=1 ./stream \
        2>&1 | tee "$RESULTS_DIR/stream_numa_remote.txt"
fi

# 4. Memory latency
if [[ -x ./mem_latency ]]; then
    ./mem_latency 2>&1 | tee "$RESULTS_DIR/mem_latency.txt"
fi

# 5. GPU compute (run before bandwidth to warm GPU to operating temperature)
"$PYTHON" matmul_bench.py 2>&1 | tee "$RESULTS_DIR/gpu_compute.txt"

# 6. GPU bandwidth — 5 iterations, all runs saved; report parser takes the best
# GPU is already warm from matmul; multiple runs guard against PCIe/boost variance.
GPU_BW_RUNS=5
if [[ "$GPU_VENDOR" == "nvidia" ]]; then
    {
        for _i in $(seq 1 $GPU_BW_RUNS); do
            echo "# run ${_i}"
            ./bandwidthTest --memory=pinned 2>&1
        done
    } | tee "$RESULTS_DIR/gpu_bandwidth.txt"
elif [[ "$GPU_VENDOR" == "amd" ]]; then
    {
        for _i in $(seq 1 $GPU_BW_RUNS); do
            echo "# run ${_i}"
            rocm-bandwidth-test 2>&1
        done
    } | tee "$RESULTS_DIR/gpu_bandwidth.txt"
fi

# 7. Multi-GPU collective bandwidth
# -w 10   : warmup iterations (covers NCCL connection setup + GPU clock ramp)
# -n 100  : timed iterations (reduces variance especially at small message sizes)
# -e 1G   : large max message ensures true PCIe/NVLink plateau is reached
# NCCL_ALGO/PROTO: pin to Ring+Simple for reproducible large-message results
if [[ "${GPU_COUNT:-0}" -gt 1 ]]; then
    # Detect NUMA node of GPU 0 via its PCIe sysfs entry.
    # /sys/bus/pci/devices/<bdf>/numa_node is -1 on non-NUMA or single-node systems.
    _gpu_numa_node() {
        local bdf=""
        if [[ "$GPU_VENDOR" == "nvidia" ]]; then
            bdf=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader \
                  2>/dev/null | head -1 | tr '[:upper:]' '[:lower:]')
        elif [[ "$GPU_VENDOR" == "amd" ]]; then
            bdf=$(rocm-smi --showbus 2>/dev/null \
                  | grep -oP '[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]' | head -1)
        fi
        local node=-1
        if [[ -n "$bdf" ]]; then
            node=$(cat "/sys/bus/pci/devices/${bdf}/numa_node" 2>/dev/null || echo -1)
        fi
        # Fall back to 0 when NUMA info is unavailable or single-node system
        [[ "$node" -lt 0 ]] && node=0
        echo "$node"
    }
    NUMA_NODE=$(_gpu_numa_node)
    echo "=== NCCL/RCCL: binding to NUMA node ${NUMA_NODE} ==="

    NCCL_ALGO=Ring NCCL_PROTO=Simple \
    numactl --cpunodebind="$NUMA_NODE" --membind="$NUMA_NODE" \
        ./build/all_reduce_perf \
            -b 8 -e "$NCCL_MAX_MSG" -f 2 \
            -g "$GPU_COUNT" \
            -w 10 -n 100 \
        2>&1 | tee "$RESULTS_DIR/nccl_allreduce.txt"
fi
