# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`vcpu_benchmarks` is a project for benchmarking vCPU and GPU performance inside guest VMs.

## Scripts

- `setup_vcpu_benchmarks.sh` — one-time setup script; installs system packages (`numactl`, `sysbench`), builds `stream` and `bandwidthTest`, builds nccl-tests, installs PyTorch. Run as root inside the guest VM.
- `run_vcpu_benchmark.sh <config_name>` — runs the full benchmark suite and writes results to `results/<config_name>/`. Requires setup to have been run first.
- `matmul_bench.py` — GPU matrix multiply benchmark using PyTorch (called by `run_vcpu_benchmark.sh`).

## Benchmark suite

| # | Tool | Measures |
| - | ------ | -------- |
| 1 | `sysbench cpu --threads=1` | Single-thread CPU performance |
| 2 | `sysbench cpu --threads=$(nproc)` | Multi-thread CPU performance |
| 3 | `stream` (OpenMP) | Memory bandwidth |
| 4 | `bandwidthTest` (CUDA samples) | GPU↔CPU memory bandwidth |
| 5 | `matmul_bench.py` (PyTorch) | GPU compute (matmul) |
| 6 | `all_reduce_perf` (nccl-tests) | Multi-GPU collective bandwidth |

## Results layout

```text
results/<config_name>/
  numa_topology.txt
  cpu_single.txt
  cpu_all.txt
  stream.txt
  gpu_bandwidth.txt
  gpu_compute.txt
  nccl_allreduce.txt   # only when >1 GPU
```
