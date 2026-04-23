#!/usr/bin/env python3
"""
Generate vCPU pinning XML fragments (or patch a full domain XML) for three configs:

  A  — vcpu_pin only       : static CPU pinning, no NUMA topology, no SMT
  B  — vcpu_pin + numa     : pinning + guest NUMA topology + memory binding, no SMT
  C  — vcpu_pin + numa + smt: same as B but includes SMT sibling threads

Supports any number of NUMA nodes (one --numa-first range per node).

Usage examples (4-NUMA, 7-GPU Xeon Platinum 8558, 94 vCPUs A/B, 188 vCPUs C):

  # Config A (pin only):
  ./gen_vcpu_pinning.py --input src.xml --output pin.xml --name pin \\
      --config A \\
      --numa-first 0-23 24-47 48-71 72-93 \\
      --emulator 94,95

  # Config B (pin + NUMA):
  ./gen_vcpu_pinning.py --input src.xml --output pin-numa.xml --name pin-numa \\
      --config B \\
      --numa-first 0-23 24-47 48-71 72-93 \\
      --emulator 94,95 \\
      --mem-mib 1638400

  # Config C (pin + NUMA + SMT):
  ./gen_vcpu_pinning.py --input src.xml --output pin-numa-smt.xml --name pin-numa-smt \\
      --config C \\
      --numa-first 0-23 24-47 48-71 72-93 \\
      --numa-smt 96-119 120-143 144-167 168-189 \\
      --emulator 94,95,190,191 \\
      --mem-mib 1638400

Legacy 2-NUMA usage (still works):

  ./gen_vcpu_pinning.py --input src.xml --output test-pin.xml --name test-pin \\
      --config A \\
      --numa-first 0-27 64-91 \\
      --emulator 28,29
"""

import argparse
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional


def parse_range(s: str) -> List[int]:
    """Parse a CPU range string like '0-27' or '28,29' into a sorted list of ints."""
    cpus = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return sorted(cpus)


def range_str(cpus: List[int]) -> str:
    """Convert a list of ints back to a compact range string like '0-27'."""
    if not cpus:
        return ""
    cpus = sorted(cpus)
    parts = []
    start = end = cpus[0]
    for c in cpus[1:]:
        if c == end + 1:
            end = c
        else:
            parts.append(f"{start}-{end}" if start != end else str(start))
            start = end = c
    parts.append(f"{start}-{end}" if start != end else str(start))
    return ",".join(parts)


def build_cputune(pins: List[Tuple[int, int]], emulator: str) -> str:
    lines = ["  <cputune>", f"    <emulatorpin cpuset='{emulator}'/>"]
    for vcpu, host_cpu in pins:
        lines.append(f"    <vcpupin vcpu='{vcpu}' cpuset='{host_cpu}'/>")
    lines.append("  </cputune>")
    return "\n".join(lines)


def build_cpu_plain() -> str:
    return "  <cpu mode='host-passthrough' check='none' migratable='on'/>"


def fit_topology(vcpu_count: int, n_nodes: int, threads: int) -> Tuple[int, int, int]:
    """Return (sockets, cores, threads) such that sockets*cores*threads == vcpu_count.

    Prefers n_nodes sockets; falls back to 1 socket if vcpu_count isn't divisible.
    """
    if vcpu_count % (n_nodes * threads) == 0:
        return n_nodes, vcpu_count // (n_nodes * threads), threads
    if vcpu_count % threads == 0:
        return 1, vcpu_count // threads, threads
    return 1, vcpu_count, 1


def build_cpu_numa(vcpu_count: int, n_nodes: int, threads: int,
                   cells: List[Tuple[int, str, int]], topoext: bool) -> str:
    sockets, cores, actual_threads = fit_topology(vcpu_count, n_nodes, threads)
    lines = [
        f"  <cpu mode='host-passthrough' check='none' migratable='on'>",
        f"    <topology sockets='{sockets}' cores='{cores}' threads='{actual_threads}'/>",
    ]
    if topoext:
        lines.append("    <feature policy='require' name='topoext'/>")
    lines.append("    <numa>")
    for cell_id, cpus, mem_mib in cells:
        lines.append(f"      <cell id='{cell_id}' cpus='{cpus}' memory='{mem_mib}' unit='MiB'/>")
    lines.append("    </numa>")
    lines.append("  </cpu>")
    return "\n".join(lines)


def build_numatune(n_nodes: int) -> str:
    nodeset = ",".join(str(i) for i in range(n_nodes))
    lines = [
        "  <numatune>",
        f"    <memory mode='strict' nodeset='{nodeset}'/>",
    ]
    for i in range(n_nodes):
        lines.append(f"    <memnode cellid='{i}' mode='strict' nodeset='{i}'/>")
    lines.append("  </numatune>")
    return "\n".join(lines)


def apply_to_xml(src_xml: str, name: str, vcpu_count: int,
                 cputune: str, cpu_block: str,
                 numatune: Optional[str], hugepage_n_nodes: Optional[int]) -> str:
    # Replace name
    old_name = re.search(r"<name>([^<]+)</name>", src_xml).group(1)
    xml = src_xml.replace(f"<name>{old_name}</name>", f"<name>{name}</name>")

    # New UUID
    new_uuid = subprocess.check_output(["uuidgen"]).decode().strip()
    xml = re.sub(r"<uuid>[^<]+</uuid>", f"<uuid>{new_uuid}</uuid>", xml)

    # NVRAM path
    xml = xml.replace(
        f"/var/lib/libvirt/qemu/nvram/{old_name}_VARS.fd",
        f"/var/lib/libvirt/qemu/nvram/{name}_VARS.fd",
    )

    # vcpu count + insert cputune after it
    xml = re.sub(
        r"  <vcpu placement='static'>\d+</vcpu>",
        f"  <vcpu placement='static'>{vcpu_count}</vcpu>\n{cputune}",
        xml,
    )

    # Replace <cpu .../> or <cpu ...>...</cpu>
    xml = re.sub(
        r"  <cpu mode='host-passthrough'[^/]*/>\n?|  <cpu mode='host-passthrough'.*?</cpu>\n?",
        cpu_block + "\n",
        xml,
        flags=re.DOTALL,
    )

    # Insert numatune before <clock
    if numatune:
        xml = xml.replace("  <clock", f"{numatune}\n  <clock", 1)

    # Hugepage nodeset
    if hugepage_n_nodes is not None:
        nodeset = ",".join(str(i) for i in range(hugepage_n_nodes))
        xml = xml.replace(
            "<page size='1048576' unit='KiB'/>",
            f"<page size='1048576' unit='KiB' nodeset='{nodeset}'/>",
        )

    return xml


def main():
    p = argparse.ArgumentParser(
        description="Generate vCPU-pinned libvirt domain XML variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", required=True, help="Source domain XML file")
    p.add_argument("--output", required=True, help="Output XML file")
    p.add_argument("--name", required=True, help="New domain name")
    p.add_argument("--config", required=True, choices=["A", "B", "C"],
                   help="A=pin only  B=pin+numa  C=pin+numa+smt")
    p.add_argument("--numa-first", required=True, nargs="+", metavar="RANGE",
                   help="First-thread host CPUs per NUMA node, one range per node "
                        "(e.g. 0-23 24-47 48-71 72-93)")
    p.add_argument("--numa-smt", nargs="+", metavar="RANGE",
                   help="SMT sibling host CPUs per NUMA node (config C only), "
                        "one range per node matching --numa-first order")
    p.add_argument("--emulator", required=True, metavar="CPUSET",
                   help="Emulator thread CPU affinity, e.g. 94,95 or 94,95,190,191")
    p.add_argument("--mem-mib", type=int, metavar="MiB",
                   help="Total VM memory in MiB (required for B/C; split equally across nodes)")
    args = p.parse_args()

    if args.config in ("B", "C") and not args.mem_mib:
        p.error("--mem-mib is required for config B and C")
    if args.config == "C" and not args.numa_smt:
        p.error("--numa-smt is required for config C")
    if args.config == "C" and len(args.numa_smt) != len(args.numa_first):
        p.error("--numa-smt must provide the same number of ranges as --numa-first")

    n_nodes = len(args.numa_first)
    first_cpus = [parse_range(r) for r in args.numa_first]

    with open(args.input) as f:
        src = f.read()

    if args.config == "A":
        pins = []
        vcpu = 0
        for node_cpus in first_cpus:
            for cpu in node_cpus:
                pins.append((vcpu, cpu))
                vcpu += 1
        vcpu_count = vcpu
        cputune = build_cputune(pins, args.emulator)
        cpu_block = build_cpu_plain()
        numatune = None
        hugepage_n_nodes = None

    elif args.config == "B":
        pins = []
        vcpu = 0
        node_vcpu_ranges: List[Tuple[int, int]] = []
        for node_cpus in first_cpus:
            start = vcpu
            for cpu in node_cpus:
                pins.append((vcpu, cpu))
                vcpu += 1
            node_vcpu_ranges.append((start, vcpu - 1))
        vcpu_count = vcpu
        mem_per_node = args.mem_mib // n_nodes
        cells = [
            (i, f"{s}-{e}", mem_per_node)
            for i, (s, e) in enumerate(node_vcpu_ranges)
        ]
        cpu_block = build_cpu_numa(
            vcpu_count=vcpu_count, n_nodes=n_nodes, threads=1,
            cells=cells,
            topoext=False,
        )
        cputune = build_cputune(pins, args.emulator)
        numatune = build_numatune(n_nodes)
        hugepage_n_nodes = n_nodes

    else:  # C
        smt_cpus = [parse_range(r) for r in args.numa_smt]
        pins = []
        vcpu = 0
        node_vcpu_ranges = []
        for node_first, node_smt in zip(first_cpus, smt_cpus):
            start = vcpu
            # Interleave: (first0, smt0), (first1, smt1), … so that each
            # consecutive guest vCPU pair maps to the same physical HT core.
            for f_cpu, s_cpu in zip(node_first, node_smt):
                pins.append((vcpu, f_cpu))
                vcpu += 1
                pins.append((vcpu, s_cpu))
                vcpu += 1
            node_vcpu_ranges.append((start, vcpu - 1))
        vcpu_count = vcpu
        mem_per_node = args.mem_mib // n_nodes
        cells = [
            (i, f"{s}-{e}", mem_per_node)
            for i, (s, e) in enumerate(node_vcpu_ranges)
        ]
        # topoext is AMD-only; on Intel, threads=2 in <topology> is sufficient
        cpu_block = build_cpu_numa(
            vcpu_count=vcpu_count, n_nodes=n_nodes, threads=2,
            cells=cells,
            topoext=False,
        )
        cputune = build_cputune(pins, args.emulator)
        numatune = build_numatune(n_nodes)
        hugepage_n_nodes = n_nodes

    result = apply_to_xml(src, args.name, vcpu_count, cputune, cpu_block,
                          numatune, hugepage_n_nodes)

    with open(args.output, "w") as f:
        f.write(result)

    print(f"Written {args.output}  (config {args.config}, {vcpu_count} vCPUs, {n_nodes} NUMA nodes)")


if __name__ == "__main__":
    main()
