#!/usr/bin/env bash
# run_vcpu_benchmark.sh — run inside the guest VM

CONFIG_NAME="${1:?usage: $0 <config_name>}"
RESULTS_DIR="results/${CONFIG_NAME}"
mkdir -p "$RESULTS_DIR"

echo "=== Config: $CONFIG_NAME ==="
echo "vCPUs: $(nproc)"
numactl --hardware > "$RESULTS_DIR/numa_topology.txt"

# 1. CPU single-thread
sysbench cpu --threads=1 --time=30 run 2>&1 | tee "$RESULTS_DIR/cpu_single.txt"

# 2. CPU all-threads
sysbench cpu --threads=$(nproc) --time=30 run 2>&1 | tee "$RESULTS_DIR/cpu_all.txt"

# 3. Memory bandwidth
OMP_NUM_THREADS=$(nproc) ./stream 2>&1 | tee "$RESULTS_DIR/stream.txt"

# 4. GPU bandwidth
./bandwidthTest --memory=pinned 2>&1 | tee "$RESULTS_DIR/gpu_bandwidth.txt"

# 5. GPU compute
python3 matmul_bench.py 2>&1 | tee "$RESULTS_DIR/gpu_compute.txt"

# 6. Multi-GPU (if applicable)
if [[ $(nvidia-smi -L | wc -l) -gt 1 ]]; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    ./build/all_reduce_perf -b 8 -e 256M -f 2 -g "$GPU_COUNT" \
        2>&1 | tee "$RESULTS_DIR/nccl_allreduce.txt"
fi