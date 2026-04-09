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
# Results are written to results/<gpu>/<config_name>/
```

Organise results as `results/<gpu>/<config_name>/` — one subdirectory per GPU type so the report can compare configurations across hardware.

Optional arguments:

```bash
./run_vcpu_benchmark.sh <config_name> [--nccl-max-msg SIZE]
```

- `--nccl-max-msg SIZE` — maximum message size for the NCCL/RCCL AllReduce sweep (default: `1G`).

**3. Generate HTML report:**

```bash
python3 generate_report.py
# Output: results/report.html (self-contained, open in any browser)
```

Options:

```bash
python3 generate_report.py [results_dir] [-o output.html|.pdf]
```

Dependencies: `pip install -r requirements.txt`

The report groups results by GPU type, with each group showing its own charts so
configurations are only compared against others on the same hardware. A summary table
at the top covers all configurations and supports click-to-sort on any column.

## NUMA nodes

Non-uniform memory access (NUMA) systems are server platforms with more than one system bus.
These platforms can utilize multiple processors on a single motherboard, and all processors can
access all the memory on the board. When a processor accesses memory that does not lie within
its own node (remote memory), data must be transferred over the NUMA connection at a rate that
is slower than it would be when accessing local memory. Thus, memory access times are not
uniform and depend on the location (proximity) of the memory and the node from which it is
accessed.

On servers with many CPU cores, they are clustered in NUMA nodes. This solves the UMA
bottleneck (memory access through a single system bus), but introduces latency when accessing
another node's memory.

A NUMA node is a CPU + its local RAM. To access another node's memory, the CPU has to
communicate across the interconnect, adding latency (the remote hop cost). You can pin a
process to a specific NUMA node using `numactl`.

NUMA nodes are not always equal to host sockets.

## Benchmark tests

### System topology capture

Before any benchmark runs, the script captures:

- **`numa_topology.txt`** (`numactl --hardware`) — shows how many NUMA nodes the guest sees,
  which vCPUs belong to each node, and the inter-node distances. Verifies that `<numatune>` in
  the domain XML is correctly reflected inside the guest.
- **`cpu_topology.txt`** (`lscpu`) — shows sockets, cores per socket, and threads per core
  from the guest's perspective. Verifies that `<topology sockets=".." cores=".." threads=".."/>`
  in the domain XML matches what the OS actually sees.
- **`hugepages.txt`** (`/proc/meminfo`) — shows whether huge pages are allocated.
  Huge pages (`<memoryBacking><hugepages/>` in domain XML) reduce TLB pressure and can
  significantly lower memory latency for large working sets. This confirms whether the setting
  took effect.

---

### 1. CPU single-thread — `sysbench cpu --threads=1`

Measures how many prime-number calculations a single vCPU can complete per second over 30
seconds.

**Why it matters:** Single-thread performance is governed by the host CPU's IPC and clock
frequency assigned to that vCPU. It is unaffected by how many vCPUs are configured or how they
are scheduled, so it acts as a baseline for raw per-core compute quality. If pinning improves
this number, it means unpinned vCPUs were suffering from scheduling noise — the hypervisor
migrating them between physical cores and evicting their caches. A drop after pinning means the
pinned cores are slower (e.g. efficiency cores, a heavily loaded NUMA node, or reduced turbo
headroom).

---

### 2. CPU multi-thread — `sysbench cpu --threads=$(nproc)`

Same prime calculation, scaled to all available vCPUs simultaneously.

**Why it matters:** This test stresses the scheduler and the vCPU-to-pCPU mapping. Without
pinning, the hypervisor migrates vCPUs freely, which introduces jitter and cache invalidation.
With `<vcpupin>`, each vCPU is locked to a specific physical CPU, reducing migrations. The key
signals are total throughput, per-thread throughput (total / thread count), and sysbench's
thread fairness stddev — a high stddev indicates scheduling imbalance from poor NUMA placement
or oversubscription.

---

### 3. Memory bandwidth — STREAM

Runs four memory kernels (Copy, Scale, Add, Triad) across arrays totalling ~1.8 GB using all
vCPUs via OpenMP. Reports the best bandwidth in MB/s for each kernel.

**Why it matters:** STREAM measures sustainable memory bandwidth — the ceiling for any
memory-bound workload (ML data loading, large matrix ops, I/O buffers). For vCPU tuning it is
sensitive to:

- **NUMA placement** — if vCPUs span two NUMA nodes but memory is allocated on one, half the
  threads pay the remote-hop penalty, cutting bandwidth significantly.
- **Memory interleaving** — `<numatune mode="interleave"/>` spreads allocations across nodes,
  which can raise aggregate bandwidth for multi-threaded workloads.
- **Hugepages** — reduce TLB misses during large sequential scans, lifting bandwidth by 5–15%
  in some configurations.

The Triad kernel (reads two arrays, writes one) is the most representative of real workloads.

---

### 3a. Per-NUMA-node STREAM — `numactl --membind`

Runs STREAM twice: once with memory bound to NUMA node 0 (`--membind=0`), once to node 1.
Only runs when the guest sees more than one NUMA node.

**Why it matters:** Isolates the bandwidth available from each NUMA node independently.
Comparing local vs. remote results directly quantifies the NUMA penalty: if remote bandwidth
is 30% lower, any workload that crosses NUMA nodes loses 30% of its memory throughput. This
guides decisions about `<numatune>` policy (`strict`, `preferred`, or `interleave`).

---

### 4. Memory latency — pointer chase (`mem_latency`)

Builds a random pointer cycle through an array of increasing size (4 KB to 512 MB) and
measures average nanoseconds per dereference. Each size is measured long enough for a stable
reading.

**Why it matters:** Latency — not bandwidth — is the bottleneck for workloads with irregular
memory access patterns (databases, graph traversals, sparse models). The output curve shows
distinct plateau regions that map to hardware:

| Array size vs. cache    | What you see      |
|-------------------------|-------------------|
| Fits in L1 (<=32 KB)    | ~4 ns             |
| Fits in L2 (<=512 KB)   | ~12 ns            |
| Fits in L3 (<= LLC size)| ~30-50 ns         |
| Exceeds L3 (DRAM)       | ~80-120 ns        |
| Remote NUMA DRAM        | 2-3x DRAM latency |

For vCPU tuning this directly reveals whether a vCPU's memory is landing on the correct NUMA
node (local DRAM latency) or a remote one (elevated plateau in the large-array region). It also
shows whether hugepages reduced TLB-miss overhead in the L3-to-DRAM transition zone.

---

### 5. GPU compute — `matmul_bench.py`

Runs 100 iterations of an 8192×8192 FP32 matrix multiply on the GPU using PyTorch, after a
10-iteration warmup. Reports average milliseconds per operation.

**Why it matters:** Matrix multiply is the core operation of deep learning training and
inference. It exercises the GPU's tensor cores and HBM at sustained throughput. This test is
largely independent of vCPU configuration (once the GPU is warmed up), so it serves as a
**control**: consistent numbers across configs confirm the GPU itself is not affected by vCPU
changes. Variation here suggests CPU-side interference — bottlenecked kernel launches,
PCIe contention, or power/thermal throttling triggered by the CPU benchmark tests run earlier.

This test runs **before** the bandwidth test so the GPU is already at operating temperature
when PCIe transfers are measured.

---

### 6. GPU ↔ CPU memory bandwidth — `bandwidthTest` (NVIDIA) / `gpu_bandwidth_bench.py` (AMD)

Tests pinned memory transfers in both directions (host→device, device→host) and device-to-device
bandwidth (GPU internal). Runs **5 times**; the report takes the best result across all runs to
guard against PCIe variance and GPU boost ramp-up on the first transfer.

For NVIDIA, `bandwidthTest` (from CUDA samples) is used. For AMD, `gpu_bandwidth_bench.py`
(a PyTorch/HIP script) is used — `rocm-bandwidth-test` in ROCm 7.x switched to a plugin
architecture that does not support all GPU generations (e.g. MI350X).

**Why it matters for vCPU configs:** Host↔device bandwidth is limited by PCIe (typically
25–60 GB/s on PCIe 4.0/5.0 ×16). Inside a VM the GPU is passed through via VFIO. If the vCPUs
are pinned to a NUMA node that is not the one closest to the GPU's PCIe root complex, the
CPU-side staging buffers used for DMA will be remote, degrading transfer bandwidth. Device-to-device
bandwidth is internal to the GPU and should be unaffected by vCPU configuration — a stable
value here confirms GPU passthrough is working correctly and acts as a control.

---

### 7. Multi-GPU collective bandwidth — `all_reduce_perf` (NCCL / RCCL)

Runs an AllReduce collective (float32 sum) across all GPUs in the guest, sweeping message
sizes from 8 B to 256 MB. Reports algorithmic and bus bandwidth at each size. Only runs when
more than one GPU is present. Uses NCCL on NVIDIA and RCCL on AMD — both produce identical
output format.

**Why it matters:** AllReduce is the dominant communication primitive in distributed deep
learning (gradient synchronization). The bandwidth at large message sizes (≥8 MB) reflects
effective inter-GPU bandwidth, limited by NVLink (if present) or PCIe topology. vCPU placement
matters because the CPU orchestrates NCCL/RCCL operations — if the vCPUs are on a different
NUMA node than the GPUs' PCIe attachment, coordination overhead increases latency at small
message sizes, raising the point at which bandwidth saturates. The average bus bandwidth across
all sizes is a useful single-number summary of collective performance.

---

## Troubleshooting: low PCIe H2D/D2H bandwidth on newer GPUs

### Symptom

A newer GPU (e.g. RTX 5090) shows H2D/D2H bandwidth of ~5 GB/s inside the VM while older GPUs
(e.g. RTX 4090/4080) on the same benchmark show ~22–26 GB/s. Device-to-device (internal memory)
bandwidth is unaffected and correctly higher on the newer card.

### Cause

The PCIe link trained at a degraded speed on the host — in the case observed, the GPU fell back
to **PCIe Gen 1 (2.5 GT/s)** despite the card being capable of Gen 5 (32 GT/s). PCIe Gen 1 ×16
gives ~4 GB/s bidirectional, which matches the symptom exactly. This is a host-side issue, not
a VM configuration issue.

This can happen with newer GPU architectures (e.g. Blackwell/RTX 50 series) on older server
platforms where the BIOS uses a conservative compatibility default for link speed training.

### Diagnosis

On the **host**, check the negotiated PCIe link speed for the GPU:

```bash
sudo lspci -vv -s <gpu_bdf> | grep -i "lnk"
```

Look for `LnkSta:`. A degraded link will show `(downgraded)`:

```text
LnkSta: Speed 2.5GT/s (downgraded), Width x16
```

Also note `LnkCtl2: Target Link Speed:` — if this is set to 2.5GT/s by the BIOS, the link
will train to Gen 1 on every boot.

### `setpci` register reference

PCIe link control lives inside the PCIe capability structure in config space. `setpci` uses the
symbolic alias `CAP_EXP` to locate it automatically. The syntax is:

```text
setpci -s <BDF> CAP_EXP+<offset>.<width>=<value>
```

| Part        | Meaning                                                                  |
|-------------|--------------------------------------------------------------------------|
| `CAP_EXP`   | Resolved by walking the capability list — no need to know the raw offset |
| `+<offset>` | Byte offset within the PCIe capability structure (see table below)       |
| `.<width>`  | `b` = byte (8-bit), `w` = word (16-bit), `l` = long (32-bit)             |
| `=<value>`  | Hex value to write                                                       |

**Relevant PCIe capability offsets:**

| Offset  | Register                   | Notes                                 |
|---------|----------------------------|---------------------------------------|
| `+0x0C` | Link Capabilities (LnkCap) | Max speed/width — read-only           |
| `+0x10` | Link Control (LnkCtl)      | Active control — bit 5 = Retrain Link |
| `+0x12` | Link Status (LnkSta)       | Current negotiated speed/width        |
| `+0x30` | Link Control 2 (LnkCtl2)   | Target speed for next retrain         |

**Link Control 2 (`+0x30`) bits [3:0] — target link speed:**

| Value  | Speed     | Generation  |
|--------|-----------|-------------|
| `0x1`  | 2.5 GT/s  | PCIe Gen 1  |
| `0x2`  | 5 GT/s    | PCIe Gen 2  |
| `0x3`  | 8 GT/s    | PCIe Gen 3  |
| `0x4`  | 16 GT/s   | PCIe Gen 4  |
| `0x5`  | 32 GT/s   | PCIe Gen 5  |

**Link Control (`+0x10`) bit 5 (`0x0020`) — Retrain Link:** self-clearing bit that drops the
link and re-negotiates at the target speed set in LnkCtl2. Some hardware auto-retrains when
LnkCtl2 changes; others require this bit to be set explicitly.

### Solution 1: force link retrain (immediate, temporary)

Set the target speed and trigger a retrain. The two-step sequence works on all hardware:

```bash
# Set target speed to Gen 5 (use 0x0004 for Gen 4, 0x0003 for Gen 3, etc.)
sudo setpci -s <gpu_bdf> CAP_EXP+0x30.w=0x0005
# Trigger the retrain
sudo setpci -s <gpu_bdf> CAP_EXP+0x10.w=0x0020
```

After retraining, verify the new speed with `lspci -vv` again. The link should train up to the
highest speed both the GPU and the upstream root port support.

> **Warning:** do not retrain while the GPU is in use by a VM. The link briefly drops during
> renegotiation, which will crash or hang the guest. Only retrain before the VM starts.

### Solution 2: make it persistent

Add the retrain command to `/etc/rc.local` or a systemd oneshot service on the host so it
runs on every boot before the VM starts:

```ini
# /etc/systemd/system/pcie-retrain-gpu.service
[Unit]
Description=Retrain PCIe link for GPU passthrough
Before=libvirtd.service

[Service]
Type=oneshot
ExecStart=/usr/sbin/setpci -s <gpu_bdf> CAP_EXP+0x30.w=0x0060
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now pcie-retrain-gpu.service
```

### Platform ceiling

If after retraining the link is still marked `(downgraded)`, the upstream PCIe root port may
not support the GPU's maximum speed. For example, AMD EPYC Rome (Starship/Matisse) platforms
support PCIe 4.0 (16 GT/s) only. An RTX 5090 (PCIe 5.0 capable) will therefore top out at
Gen 4 speeds (~22–26 GB/s H2D/D2H), matching a 4090 on the same platform.

To identify the upstream root port and its maximum link speed:

```bash
# Find bridges upstream of the GPU's bus
sudo lspci -D | grep "root port\|pci bridge"
# Then inspect the bridge whose secondary bus matches the GPU's bus number
sudo lspci -vv -s <bridge_bdf> | grep -i "lnkcap\|lnksta"
```

`LnkCap: Speed 16GT/s` on the root port confirms a PCIe 4.0 platform ceiling. Reaching Gen 5
requires a newer server CPU (e.g. EPYC Genoa or Intel Sapphire Rapids and later).

---

**AMD-specific notes:**

- `HSA_NO_SCRATCH_RECLAIM=1` is required on MI350X (and possibly other recent AMD GPUs) to
  prevent RCCL from failing with a fatal scratch reclaim error.
- `NCCL_P2P_DISABLE=1` is set automatically. On MI350X VF configurations, direct GPU↔GPU
  P2P/XGMI memory access hangs; routing through host shared memory (SHM transport) is required.
  This lowers peak bus bandwidth compared to bare-metal NVLink/XGMI but reflects the real
  achievable performance in the VM configuration.
- ROCm 7.0+ PyTorch wheels are required for MI350X (gfx950). Earlier rocm6.x wheels only
  support up to gfx942 (MI300X).
