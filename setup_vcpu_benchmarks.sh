#!/usr/bin/env bash
# setup_vcpu_benchmarks.sh — run once inside the guest VM to install dependencies
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── GPU vendor detection ──────────────────────────────────────────────────────
GPU_VENDOR="none"
if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null 2>&1; then
    GPU_VENDOR="nvidia"
elif command -v rocm-smi &>/dev/null && rocm-smi &>/dev/null 2>&1; then
    GPU_VENDOR="amd"
fi
echo "=== Detected GPU vendor: ${GPU_VENDOR} ==="

# Ensure CUDA/ROCm bin is on PATH if installed but not exported
for _bin in /usr/local/cuda/bin /usr/local/cuda-*/bin /opt/rocm/bin /opt/rocm-*/bin; do
    [[ -d "$_bin" && ":$PATH:" != *":$_bin:"* ]] && export PATH="$_bin:$PATH"
done
unset _bin

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

# ── 3. Memory latency benchmark ──────────────────────────────────────────────
echo "=== Building mem_latency ==="
if [[ ! -f mem_latency ]]; then
    gcc -O2 -o mem_latency mem_latency.c
    echo "mem_latency built OK"
else
    echo "mem_latency already built, skipping"
fi

# ── 4. GPU bandwidth test ─────────────────────────────────────────────────────
if [[ "$GPU_VENDOR" == "nvidia" ]]; then
    echo "=== Building CUDA bandwidthTest ==="
    if [[ ! -f bandwidthTest ]]; then
        if ! command -v nvcc &>/dev/null; then
            echo "WARNING: nvcc not found — skipping bandwidthTest build"
        else
            apt-get install -y --no-install-recommends nvidia-cuda-samples
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

elif [[ "$GPU_VENDOR" == "amd" ]]; then
    echo "=== Installing rocm-bandwidth-test + transferbench-dev ==="
    apt-get install -y --no-install-recommends rocm-bandwidth-test transferbench-dev 2>/dev/null || \
        apt-get install -y --no-install-recommends rocm-bandwidth-test
    echo "rocm-bandwidth-test installed OK"
fi

# ── 5. Collective bandwidth test (NCCL / RCCL) ───────────────────────────────
if [[ "$GPU_VENDOR" == "nvidia" ]]; then
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

elif [[ "$GPU_VENDOR" == "amd" ]]; then
    echo "=== Building rccl-tests ==="
    if [[ ! -f build/all_reduce_perf ]]; then
        if ! command -v hipcc &>/dev/null; then
            echo "WARNING: hipcc not found — skipping rccl-tests build"
        else
            apt-get install -y --no-install-recommends rccl-dev
            git clone --depth 1 https://github.com/ROCm/rccl-tests.git
            make -C rccl-tests -j"$(nproc)" \
                HIP_HOME="${HIP_HOME:-/opt/rocm}" \
                MPI=0
            mkdir -p build
            cp rccl-tests/build/all_reduce_perf build/
            echo "rccl-tests built OK"
        fi
    else
        echo "rccl-tests already present, skipping"
    fi
fi

# ── 6. Python venv + PyTorch ──────────────────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/.venv"
echo "=== Setting up Python venv ==="
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    echo "venv created at $VENV_DIR"
fi

echo "=== Installing PyTorch + numpy ==="
if ! "$VENV_DIR/bin/python" -c "import torch; torch.zeros(1, device='cuda')" &>/dev/null 2>&1; then
    if [[ "$GPU_VENDOR" == "nvidia" ]]; then
        # cu128 wheels support Blackwell (sm_120) and require CUDA 12.8+ driver
        "$VENV_DIR/bin/pip" install --quiet torch numpy \
            --index-url https://download.pytorch.org/whl/cu128
    elif [[ "$GPU_VENDOR" == "amd" ]]; then
        # ROCm PyTorch — maps device('cuda') to ROCm internally.
        # rocm7.0 nightly required for MI350X (gfx950) support; stable rocm6.x wheels
        # only cover up to gfx942 (MI300X).
        "$VENV_DIR/bin/pip" install --quiet --pre torch numpy \
            --index-url https://download.pytorch.org/whl/nightly/rocm7.0
    else
        "$VENV_DIR/bin/pip" install --quiet torch numpy
    fi
    echo "PyTorch + numpy installed OK"
else
    echo "PyTorch already installed, skipping"
fi

echo "=== Installing Python dependencies from requirements.txt ==="
"$VENV_DIR/bin/pip" install --quiet -r "$SCRIPT_DIR/requirements.txt"
echo "requirements.txt dependencies installed OK"

echo ""
echo "=== Setup complete. Run: bash run_vcpu_benchmark.sh <config_name> ==="
