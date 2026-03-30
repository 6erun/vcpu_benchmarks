#!/usr/bin/env bash
# setup_vcpu_benchmarks.sh — run once inside the guest VM to install dependencies
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure CUDA bin is on PATH if installed but not exported
for _cuda_bin in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
    [[ -d "$_cuda_bin" && ":$PATH:" != *":$_cuda_bin:"* ]] && export PATH="$_cuda_bin:$PATH"
done
unset _cuda_bin

# ── 1. System packages ────────────────────────────────────────────────────────
echo "=== Installing system packages ==="
apt-get update -qq
apt-get install -y --no-install-recommends \
    numactl \
    sysbench \
    gcc \
    make \
    git \
    python3-venv \
    python3-full \
    curl

# ── 2. STREAM memory bandwidth benchmark ─────────────────────────────────────
echo "=== Building STREAM ==="
if [[ ! -f stream ]]; then
    curl -fsSL \
        "https://www.cs.virginia.edu/stream/FTP/Code/stream.c" \
        -o stream.c
    gcc -O3 -fopenmp -DSTREAM_ARRAY_SIZE=80000000 -o stream stream.c
    echo "STREAM built OK"
else
    echo "STREAM already built, skipping"
fi

# ── 3. CUDA bandwidthTest ─────────────────────────────────────────────────────
echo "=== Building CUDA bandwidthTest ==="
if [[ ! -f bandwidthTest ]]; then
    if ! command -v nvcc &>/dev/null; then
        echo "WARNING: nvcc not found — skipping bandwidthTest build (no GPU or CUDA toolkit missing)"
    else
        apt-get install -y --no-install-recommends nvidia-cuda-samples
        # Find the samples root (contains both Samples/ and Common/)
        SAMPLES_ROOT=$(find /usr/share/doc/nvidia-cuda-toolkit -maxdepth 1 -name examples -type d 2>/dev/null | head -1)
        if [[ -z "$SAMPLES_ROOT" ]]; then
            echo "ERROR: cuda samples root not found after installing nvidia-cuda-samples" >&2
            exit 1
        fi
        cp -r "$SAMPLES_ROOT" /tmp/cuda-samples
        make -C /tmp/cuda-samples/Samples/1_Utilities/bandwidthTest -j"$(nproc)"
        cp /tmp/cuda-samples/bin/x86_64/linux/release/bandwidthTest .
        echo "bandwidthTest built OK"
    fi
else
    echo "bandwidthTest already present, skipping"
fi

# ── 4. NCCL tests (all_reduce_perf) ──────────────────────────────────────────
echo "=== Building nccl-tests ==="
if [[ ! -f build/all_reduce_perf ]]; then
    if ! command -v nvcc &>/dev/null; then
        echo "WARNING: nvcc not found — skipping nccl-tests build"
    else
        apt-get install -y --no-install-recommends libnccl-dev
        git clone --depth 1 https://github.com/NVIDIA/nccl-tests.git
        make -C nccl-tests -j"$(nproc)" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
        mkdir -p build
        cp nccl-tests/build/all_reduce_perf build/
        echo "nccl-tests built OK"
    fi
else
    echo "nccl-tests already present, skipping"
fi

# ── 5. Python venv + PyTorch with CUDA ───────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/.venv"
echo "=== Setting up Python venv ==="
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    echo "venv created at $VENV_DIR"
fi

echo "=== Installing PyTorch + numpy ==="
if ! "$VENV_DIR/bin/python" -c "import numpy; import torch; torch.zeros(1, device='cuda')" &>/dev/null 2>&1; then
    # cu128 wheels support Blackwell (sm_120) and require CUDA 12.8+ driver
    "$VENV_DIR/bin/pip" install --quiet torch numpy --index-url https://download.pytorch.org/whl/cu128
    echo "PyTorch + numpy installed OK"
else
    echo "PyTorch already installed, skipping"
fi

echo ""
echo "=== Setup complete. Run: bash run_vcpu_benchmark.sh <config_name> ==="
