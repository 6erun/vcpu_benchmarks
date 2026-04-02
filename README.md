# vcpu_benchmarks

## Overview

Scripts to measure and compare performance of different libvirt configurations of vcpu.
The goal is to find out how much performance we can get from different vcpu pinning and topology configurations, and how it compares to the performance of the host.

## Usage

**1. Setup (once, inside the guest VM, as root):**

```bash
sudo ./setup_vcpu_benchmarks.sh
```

**2. Run benchmarks:**

```bash
./run_vcpu_benchmark.sh <config_name>
# Results are written to results/<config_name>/
```

Results are organized as `results/<gpu>/<config_name>/` — create subdirectories per GPU type to compare across hardware.

**3. Generate HTML report:**

```bash
python3 generate_report.py
# Output: results/report.html (self-contained, open in any browser)
```

Options:

```bash
python3 generate_report.py [results_dir] [-o output.html]
```

Dependencies: `pip install -r requirements.txt`

## NUMA nodes

Non-uniform memory access (NUMA) systems are server platforms with more than one system bus. These platforms can utilize multiple processors on a single motherboard, and all processors can access all the memory on the board. When a processor accesses memory that does not lie within its own node (remote memory), data must be transferred over the NUMA connection at a rate that is slower than it would be when accessing local memory. Thus, memory access times are not uniform and depend on the location (proximity) of the memory and the node from which it is accessed.

On servers with many CPU cores, they are clustered in NUMA (non-uniform memory access) nodes. It solves problem with UMA bottleneck (memory access through single system bus), but introduces latency when accessing other's node memory.

NUMA node is CPU + local RAM for low latency. To access other node's memory CPU has to ask another CPU and memory controller and that adds latency (The Remote Hop Cost).
You can pin a process to specific NUMA node using `numactl`.

NUMA nodes are not always equal to host sockets.