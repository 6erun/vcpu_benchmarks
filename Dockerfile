# Dockerfile — NVIDIA/CUDA variant
#
# Build (no GPU required):
#   docker build -t vcpu_benchmarks:nvidia .
#   docker build --build-arg CUDA_SMS="89 90 120" -t vcpu_benchmarks:nvidia .
#
# Run (GPU passthrough required):
#   docker run --gpus all --privileged \
#     -v $(pwd)/results:/vcpu_benchmarks/results \
#     vcpu_benchmarks:nvidia \
#     ./run_vcpu_benchmark.sh <config_name>
#
# CUDA_SMS controls which GPU architectures bandwidthTest and nccl-tests are
# compiled for. Common values:
#   89  — Ada Lovelace (RTX 4080/4090, L40S, RTX 6000 Ada)
#   90  — Hopper (H100, H200)
#   100 — Blackwell (H100 NVL / B200 server)
#   120 — Blackwell consumer (RTX 5080/5090)
# Default covers all four so one image works across all current NVIDIA GPUs.

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS builder

ARG CUDA_SMS="89 90 100 120"

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        gcc make git curl python3-venv python3-full \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# STREAM memory bandwidth benchmark
RUN curl -fsSL "https://www.cs.virginia.edu/stream/FTP/Code/stream.c" -o stream.c \
    && gcc -O3 -fopenmp -DSTREAM_ARRAY_SIZE=80000000 -o stream stream.c \
    && rm stream.c

# Memory latency benchmark
COPY mem_latency.c mem_latency.c
RUN gcc -O2 -o mem_latency mem_latency.c \
    && rm mem_latency.c

# GPU bandwidthTest — compiled for each SM in CUDA_SMS at build time so the
# binary runs at native speed without PTX JIT on any of the target GPUs.
RUN git clone --depth 1 https://github.com/NVIDIA/cuda-samples.git /tmp/cuda-samples \
    && make -C /tmp/cuda-samples/Samples/1_Utilities/bandwidthTest \
            -j"$(nproc)" SMS="${CUDA_SMS}" \
    && cp /tmp/cuda-samples/bin/x86_64/linux/release/bandwidthTest . \
    && rm -rf /tmp/cuda-samples

# nccl-tests (AllReduce collective benchmark)
RUN apt-get update -qq && apt-get install -y --no-install-recommends libnccl-dev \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git /tmp/nccl-tests \
    && make -C /tmp/nccl-tests -j"$(nproc)" CUDA_HOME=/usr/local/cuda \
    && mkdir -p build \
    && cp /tmp/nccl-tests/build/all_reduce_perf build/ \
    && rm -rf /tmp/nccl-tests

# Python venv + PyTorch (cu128 wheels support Blackwell sm_120 and require CUDA 12.8+)
COPY requirements.txt requirements.txt
RUN python3 -m venv .venv \
    && .venv/bin/pip install --quiet torch numpy \
        --index-url https://download.pytorch.org/whl/cu128 \
    && .venv/bin/pip install --quiet -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

# numactl     — NUMA pinning for STREAM and NCCL
# sysbench    — CPU benchmark
# libgomp1    — OpenMP runtime for STREAM
# python3     — needed to run the venv Python interpreter
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
        numactl sysbench libgomp1 python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /vcpu_benchmarks

# Compiled binaries from builder stage
COPY --from=builder /build/stream          ./
COPY --from=builder /build/mem_latency     ./
COPY --from=builder /build/bandwidthTest   ./
COPY --from=builder /build/build/all_reduce_perf ./build/

# Python venv with PyTorch and report dependencies
COPY --from=builder /build/.venv           ./.venv

# Benchmark scripts and Python sources
COPY run_vcpu_benchmark.sh  ./
COPY matmul_bench.py        ./
COPY gpu_bandwidth_bench.py ./
COPY generate_report.py     ./

# Results directory — mount a host volume here to retrieve outputs after the run
RUN mkdir -p results
VOLUME ["/vcpu_benchmarks/results"]
