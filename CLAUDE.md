# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`vcpu_benchmarks` is a project for benchmarking vCPU and GPU performance inside guest VMs.

## Scripts

- `setup_vcpu_benchmarks.sh` — one-time setup script; installs system packages, builds `stream` and `mem_latency`, and auto-detects GPU vendor (NVIDIA or AMD) to install the appropriate tools: `bandwidthTest` or `rocm-bandwidth-test`, nccl-tests or rccl-tests, and CUDA or ROCm PyTorch wheels. Run as root inside the guest VM.
- `run_vcpu_benchmark.sh <config_name>` — runs the full benchmark suite and writes results to `results/<config_name>/`. Requires setup to have been run first.
- `matmul_bench.py` — GPU matrix multiply benchmark using PyTorch (called by `run_vcpu_benchmark.sh`).
- `generate_report.py` — generates a self-contained HTML report from all `results/<gpu>/<config>/` directories. Requires `matplotlib` and `numpy`. Run with `python3 generate_report.py`; output defaults to `results/report.html`.

## Benchmark suite

| # | Tool | Measures |
| - | ------ | -------- |
| 1 | `sysbench cpu --threads=1` | Single-thread CPU performance |
| 2 | `sysbench cpu --threads=$(nproc)` | Multi-thread CPU performance |
| 3 | `stream` (OpenMP) | Memory bandwidth |
| 3a | `stream` via `numactl --membind` | Per-NUMA-node memory bandwidth (multi-NUMA only) |
| 4 | `mem_latency` (pointer chase) | Memory latency vs array size (L1/L2/L3/DRAM tiers) |
| 5 | `matmul_bench.py` (PyTorch) | GPU compute (matmul) — run first to warm GPU |
| 6 | `bandwidthTest` / `rocm-bandwidth-test` | GPU↔CPU memory bandwidth (5 runs, best kept) |
| 7 | `all_reduce_perf` (nccl-tests / rccl-tests) | Multi-GPU collective bandwidth |

## Results layout

```text
results/<config_name>/
  numa_topology.txt
  cpu_topology.txt          # lscpu — verifies socket/core/thread topology from domain XML
  hugepages.txt             # /proc/meminfo hugepages lines — verifies <memoryBacking>
  cpu_single.txt
  cpu_all.txt
  stream.txt
  stream_numa_local.txt     # only when >1 NUMA node
  stream_numa_remote.txt    # only when >1 NUMA node
  mem_latency.txt           # pointer-chase latency curve (4KB–512MB)
  gpu_bandwidth.txt
  gpu_compute.txt
  nccl_allreduce.txt        # only when >1 GPU
```
