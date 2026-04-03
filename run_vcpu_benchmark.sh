#!/usr/bin/env bash
# run_vcpu_benchmark.sh — run inside the guest VM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"

CONFIG_NAME="${1:?usage: $0 <config_name>}"
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

# 5. GPU bandwidth
if [[ "$GPU_VENDOR" == "nvidia" ]]; then
    ./bandwidthTest --memory=pinned 2>&1 | tee "$RESULTS_DIR/gpu_bandwidth.txt"
elif [[ "$GPU_VENDOR" == "amd" ]]; then
    rocm-bandwidth-test 2>&1 | tee "$RESULTS_DIR/gpu_bandwidth.txt"
fi

# 6. GPU compute
"$PYTHON" matmul_bench.py 2>&1 | tee "$RESULTS_DIR/gpu_compute.txt"

# 7. Multi-GPU collective bandwidth
if [[ "${GPU_COUNT:-0}" -gt 1 ]]; then
    ./build/all_reduce_perf -b 8 -e 256M -f 2 -g "$GPU_COUNT" \
        2>&1 | tee "$RESULTS_DIR/nccl_allreduce.txt"
fi
