#!/usr/bin/env python3
"""Generate an HTML benchmark report from results/<gpu>/<config>/ directories."""

import base64
import io
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Config:
    gpu: str
    name: str
    label: str = ""           # e.g. "rtx4080 / default"

    # NUMA
    vcpus: int = 0
    ram_gb: float = 0.0

    # sysbench
    cpu_single_evs: float = 0.0
    cpu_all_evs: float = 0.0
    cpu_all_threads: int = 0

    # STREAM (MB/s)
    stream_copy: float = 0.0
    stream_scale: float = 0.0
    stream_add: float = 0.0
    stream_triad: float = 0.0

    # GPU bandwidth (GB/s)
    gpu_h2d: float = 0.0
    gpu_d2h: float = 0.0
    gpu_d2d: float = 0.0
    gpu_device_name: str = ""

    # GPU compute (ms per matmul)
    gpu_matmul_ms: float = 0.0

    # NCCL (GB/s bus bandwidth at largest message)
    nccl_peak_busbw: float = 0.0
    nccl_avg_busbw: float = 0.0
    nccl_ngpus: int = 0
    nccl_rows: list = field(default_factory=list)  # (size_bytes, busbw_oop)

    # Memory latency (pointer chase)
    mem_latency_data: list = field(default_factory=list)  # (size_bytes, latency_ns)
    mem_latency_dram_ns: float = 0.0   # latency at largest tested size (DRAM proxy)

    # System info
    hugepages_total: int = 0

    @property
    def has_nccl(self) -> bool:
        return self.nccl_peak_busbw > 0


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _first(pattern, text, group=1, cast=float, flags=0):
    m = re.search(pattern, text, flags)
    return cast(m.group(group)) if m else None


def parse_numa(text: str, cfg: Config):
    m = re.search(r"node \d+ cpus:(.*)", text)
    if m:
        cfg.vcpus = len(m.group(1).split())
    m = re.search(r"node \d+ size:\s+(\d+) MB", text)
    if m:
        cfg.ram_gb = round(int(m.group(1)) / 1024, 1)


def parse_sysbench(text: str) -> tuple[float, int]:
    evs = _first(r"events per second:\s+([\d.]+)", text)
    threads = _first(r"Number of threads:\s+(\d+)", text, cast=int)
    return evs or 0.0, threads or 0


def parse_stream(text: str, cfg: Config):
    for func, attr in [("Copy", "stream_copy"), ("Scale", "stream_scale"),
                       ("Add", "stream_add"), ("Triad", "stream_triad")]:
        v = _first(rf"{func}:\s+([\d.]+)", text)
        if v is not None:
            setattr(cfg, attr, v)


def parse_gpu_bandwidth(text: str, cfg: Config):
    if "RocmBandwidthTest" in text or "Src Device Type" in text:
        _parse_gpu_bandwidth_rocm(text, cfg)
    elif "Host-to-Device:" in text:
        _parse_gpu_bandwidth_pytorch(text, cfg)
    else:
        _parse_gpu_bandwidth_cuda(text, cfg)


def _parse_gpu_bandwidth_pytorch(text: str, cfg: Config):
    """Parse output from gpu_bandwidth_bench.py (PyTorch/HIP).

    Format (repeated per GPU per run block):
        GPU 0 [AMD Instinct MI350X VF]
          Host-to-Device: 54.70 GB/s
          Device-to-Host: 57.04 GB/s
          Device-to-Device: 2660.62 GB/s
    Takes the best value across all GPUs and all run blocks.
    """
    m = re.search(r"GPU \d+ \[([^\]]+)\]", text)
    if m:
        cfg.gpu_device_name = m.group(1).strip()

    for h2d in re.findall(r"Host-to-Device:\s*([\d.]+)", text):
        cfg.gpu_h2d = max(cfg.gpu_h2d, float(h2d))
    for d2h in re.findall(r"Device-to-Host:\s*([\d.]+)", text):
        cfg.gpu_d2h = max(cfg.gpu_d2h, float(d2h))
    for d2d in re.findall(r"Device-to-Device:\s*([\d.]+)", text):
        cfg.gpu_d2d = max(cfg.gpu_d2d, float(d2d))


def _parse_gpu_bandwidth_cuda(text: str, cfg: Config):
    m = re.search(r"Device \d+: (.+)", text)
    if m:
        cfg.gpu_device_name = m.group(1).strip()
    # File may contain multiple runs (one per "# run N" block) — take best across all
    sections = re.split(r"(Host to Device|Device to Host|Device to Device)", text)
    mapping: dict[str, float] = {}
    for i in range(1, len(sections), 2):
        label = sections[i]
        body = sections[i + 1]
        bw_match = re.search(r"\d+\s+([\d.]+)\s*$", body.strip(), re.MULTILINE)
        if bw_match:
            bw = float(bw_match.group(1))
            mapping[label] = max(mapping.get(label, 0.0), bw)
    cfg.gpu_h2d = mapping.get("Host to Device", 0.0)
    cfg.gpu_d2h = mapping.get("Device to Host", 0.0)
    cfg.gpu_d2d = mapping.get("Device to Device", 0.0)


def _parse_gpu_bandwidth_rocm(text: str, cfg: Config):
    m = re.search(r"Device\s+\d+\s*[:\-]\s*(.+)", text)
    if m:
        cfg.gpu_device_name = m.group(1).strip()

    # File may contain multiple runs — take best across all blocks
    blocks = re.split(r"={4,}.*?Benchmark Result.*?={4,}", text)
    for block in blocks:
        src_m = re.search(r"Src Device Type:\s*(\w+)", block)
        dst_m = re.search(r"Dst Device Type:\s*(\w+)", block)
        if not src_m or not dst_m:
            continue
        src, dst = src_m.group(1).lower(), dst_m.group(1).lower()

        # Peak BW is the last column; grab the last (largest message size) row
        bw_vals = re.findall(r"[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)", block)
        if not bw_vals:
            continue
        peak_bw = float(bw_vals[-1])

        if src == "cpu" and dst == "gpu":
            cfg.gpu_h2d = max(cfg.gpu_h2d, peak_bw)
        elif src == "gpu" and dst == "cpu":
            cfg.gpu_d2h = max(cfg.gpu_d2h, peak_bw)
        elif src == "gpu" and dst == "gpu":
            cfg.gpu_d2d = max(cfg.gpu_d2d, peak_bw)


def parse_gpu_compute(text: str, cfg: Config):
    m = re.search(r"\((\d+(?:\.\d+)?)ms each\)", text)
    if m:
        cfg.gpu_matmul_ms = float(m.group(1))


def parse_nccl(text: str, cfg: Config):
    # Count GPUs
    ranks = re.findall(r"Rank\s+\d+", text)
    cfg.nccl_ngpus = len(ranks)

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        parts = line.split()
        if len(parts) >= 9:
            try:
                size = int(parts[0])
                busbw_oop = float(parts[6])
                rows.append((size, busbw_oop))
            except ValueError:
                continue
    cfg.nccl_rows = rows

    if rows:
        cfg.nccl_peak_busbw = max(bw for _, bw in rows)

    m = re.search(r"Avg bus bandwidth\s*:\s*([\d.]+)", text)
    if m:
        cfg.nccl_avg_busbw = float(m.group(1))


def parse_mem_latency(text: str, cfg: Config):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            try:
                rows.append((int(parts[0]), float(parts[1])))
            except ValueError:
                continue
    cfg.mem_latency_data = rows
    if rows:
        cfg.mem_latency_dram_ns = rows[-1][1]


def parse_hugepages(text: str, cfg: Config):
    m = re.search(r"HugePages_Total:\s+(\d+)", text)
    if m:
        cfg.hugepages_total = int(m.group(1))


_SUMMARY_FIELDS = [
    "cpu_single_evs", "cpu_all_evs", "cpu_all_threads",
    "stream_copy", "stream_scale", "stream_add", "stream_triad",
    "mem_latency_dram_ns",
    "gpu_h2d", "gpu_d2h", "gpu_d2d", "gpu_device_name", "gpu_matmul_ms",
    "nccl_peak_busbw", "nccl_avg_busbw", "nccl_ngpus",
]


def write_summary_csv(cfg: Config, path: Path):
    import csv
    row = {f: getattr(cfg, f) for f in _SUMMARY_FIELDS}
    with open(path / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SUMMARY_FIELDS)
        w.writeheader()
        w.writerow(row)


def _load_metadata(path: Path, cfg: Config):
    import json
    try:
        meta = json.loads((path / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError):
        return
    if "vcpus" in meta:
        cfg.vcpus = int(meta["vcpus"])
    if "hugepages_total" in meta:
        cfg.hugepages_total = int(meta["hugepages_total"])
    if meta.get("gpu_devices"):
        cfg.gpu_device_name = meta["gpu_devices"][0]


def _load_summary_csv(path: Path, cfg: Config):
    import csv
    try:
        with open(path / "summary.csv", newline="") as f:
            row = next(csv.DictReader(f))
    except (OSError, StopIteration):
        return
    for field in _SUMMARY_FIELDS:
        if field not in row or row[field] == "":
            continue
        val = row[field]
        attr_type = type(getattr(cfg, field))
        try:
            setattr(cfg, field, attr_type(val))
        except (ValueError, TypeError):
            setattr(cfg, field, val)


def load_config(path: Path) -> Config:
    gpu = path.parent.name
    name = path.name
    cfg = Config(gpu=gpu, name=name, label=f"{gpu} / {name}")

    def read(fname):
        p = path / fname
        return p.read_text() if p.exists() else ""

    # Load structured files first; fall back to text parsing for missing fields
    has_summary = (path / "summary.csv").exists()
    if has_summary:
        _load_summary_csv(path, cfg)
    if (path / "metadata.json").exists():
        _load_metadata(path, cfg)

    # Always parse from raw text files; summary.csv values take precedence only
    # when non-zero/non-empty (guards against stale summaries from failed runs).
    parse_numa(read("numa_topology.txt"), cfg)
    _cpu_single, _ = parse_sysbench(read("cpu_single.txt"))
    _cpu_all, _cpu_threads = parse_sysbench(read("cpu_all.txt"))
    if not has_summary or cfg.cpu_single_evs == 0:
        cfg.cpu_single_evs = _cpu_single
    if not has_summary or cfg.cpu_all_evs == 0:
        cfg.cpu_all_evs = _cpu_all
        cfg.cpu_all_threads = _cpu_threads
    _stream_cfg = Config(gpu=cfg.gpu, name=cfg.name, label=cfg.label)
    parse_stream(read("stream.txt"), _stream_cfg)
    if not has_summary or cfg.stream_copy == 0:
        cfg.stream_copy  = _stream_cfg.stream_copy
        cfg.stream_scale = _stream_cfg.stream_scale
        cfg.stream_add   = _stream_cfg.stream_add
        cfg.stream_triad = _stream_cfg.stream_triad
    _bw_cfg = Config(gpu=cfg.gpu, name=cfg.name, label=cfg.label)
    parse_gpu_bandwidth(read("gpu_bandwidth.txt"), _bw_cfg)
    if not has_summary or cfg.gpu_h2d == 0:
        cfg.gpu_h2d = _bw_cfg.gpu_h2d
        cfg.gpu_d2h = _bw_cfg.gpu_d2h
        cfg.gpu_d2d = _bw_cfg.gpu_d2d
    if not has_summary or not cfg.gpu_device_name:
        cfg.gpu_device_name = _bw_cfg.gpu_device_name
    _gcomp_cfg = Config(gpu=cfg.gpu, name=cfg.name, label=cfg.label)
    parse_gpu_compute(read("gpu_compute.txt"), _gcomp_cfg)
    if not has_summary or cfg.gpu_matmul_ms == 0:
        cfg.gpu_matmul_ms = _gcomp_cfg.gpu_matmul_ms
    _ml_cfg = Config(gpu=cfg.gpu, name=cfg.name, label=cfg.label)
    parse_mem_latency(read("mem_latency.txt"), _ml_cfg)
    if not has_summary or cfg.mem_latency_dram_ns == 0:
        cfg.mem_latency_dram_ns = _ml_cfg.mem_latency_dram_ns
    parse_hugepages(read("hugepages.txt"), cfg)

    # nccl_rows (full curve) always comes from the raw file — not stored in summary.csv
    if (path / "nccl_allreduce.txt").exists():
        parse_nccl(read("nccl_allreduce.txt"), cfg)

    return cfg


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

PALETTE = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2",
    "#59a14f", "#edc948", "#b07aa1", "#ff9da7",
]

def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def bar_chart(title, labels, values, unit, colors, highlight_max=True, highlight_min=False):
    fig, ax = plt.subplots(figsize=(max(5, len(labels) * 1.6), 3.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.55, zorder=3)

    if highlight_max and values:
        best = max(range(len(values)), key=lambda i: values[i])
        bars[best].set_edgecolor("#1a1a1a")
        bars[best].set_linewidth(2)

    if highlight_min and values:
        worst = min(range(len(values)), key=lambda i: values[i])
        bars[worst].set_edgecolor("#cc0000")
        bars[worst].set_linewidth(2)
        bars[worst].set_linestyle("--")

    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{val:,.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel(unit, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylim(0, max(max(values) * 1.2, 1) if values else 1)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig_to_b64(fig)


def stream_grouped_chart(configs, colors):
    metrics = ["Copy", "Scale", "Add", "Triad"]
    attrs = ["stream_copy", "stream_scale", "stream_add", "stream_triad"]
    n_configs = len(configs)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / n_configs

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (cfg, color) in enumerate(zip(configs, colors)):
        vals = [getattr(cfg, a) for a in attrs]
        offset = (i - n_configs / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9, label=cfg.label,
                      color=color, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                        f"{v/1000:.0f}k", ha="center", va="bottom", fontsize=7.5)

    ax.set_title("STREAM Memory Bandwidth", fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel("MB/s", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig_to_b64(fig)


def mem_latency_chart(configs: list, colors: list):
    data = [(c, col) for c, col in zip(configs, colors) if c.mem_latency_data]
    if not data:
        return None

    fig, ax = plt.subplots(figsize=(10, 4.5))
    xlabels = None
    for cfg, color in data:
        sizes, lats = zip(*cfg.mem_latency_data)
        ax.plot(range(len(sizes)), lats, marker="o", label=cfg.label,
                color=color, linewidth=2, markersize=5)
        if xlabels is None:
            xlabels = []
            for s in sizes:
                if s >= 1024 * 1024:
                    xlabels.append(f"{s // (1024 * 1024)}MB")
                elif s >= 1024:
                    xlabels.append(f"{s // 1024}KB")
                else:
                    xlabels.append(f"{s}B")

    ax.set_title("Memory Latency — Pointer Chase (lower is better)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel("Latency (ns)", fontsize=9)
    ax.set_xlabel("Array size", fontsize=9)
    if xlabels:
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig_to_b64(fig)


def nccl_chart(cfg: Config, color: str):
    if not cfg.nccl_rows:
        return None
    sizes, bws = zip(*cfg.nccl_rows)
    labels = []
    for s in sizes:
        if s >= 1024 * 1024:
            labels.append(f"{s // (1024*1024)}M")
        elif s >= 1024:
            labels.append(f"{s // 1024}K")
        else:
            labels.append(str(s))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(bws)), bws, marker="o", color=color, linewidth=2, markersize=5)
    ax.fill_between(range(len(bws)), bws, alpha=0.15, color=color)
    ax.set_title(f"NCCL AllReduce Bus Bandwidth — {cfg.label} ({cfg.nccl_ngpus} GPUs)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel("Bus BW (GB/s)", fontsize=9)
    ax.set_xlabel("Message size", fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.axhline(cfg.nccl_avg_busbw, linestyle="--", color="gray", linewidth=1,
               label=f"avg {cfg.nccl_avg_busbw:.2f} GB/s")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig_to_b64(fig)


def nccl_overlay_chart(configs: list, colors: list, title: str = "NCCL AllReduce Bus Bandwidth"):
    data = [(c, col) for c, col in zip(configs, colors) if c.nccl_rows]
    if not data:
        return None

    fig, ax = plt.subplots(figsize=(12, 5))
    xlabels = None
    for cfg, color in data:
        sizes, bws = zip(*cfg.nccl_rows)
        ax.plot(range(len(bws)), bws, marker="o", label=cfg.name,
                color=color, linewidth=2, markersize=5)
        ax.axhline(cfg.nccl_avg_busbw, linestyle="--", color=color,
                   linewidth=1, alpha=0.5)
        if xlabels is None:
            xlabels = []
            for s in sizes:
                if s >= 1024 * 1024:
                    xlabels.append(f"{s // (1024*1024)}M")
                elif s >= 1024:
                    xlabels.append(f"{s // 1024}K")
                else:
                    xlabels.append(str(s))

    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel("Bus BW (GB/s)", fontsize=9)
    ax.set_xlabel("Message size", fontsize=9)
    if xlabels:
        ax.set_xticks(range(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig_to_b64(fig)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #f5f6fa; color: #222; padding: 24px; }
h1 { font-size: 1.8rem; margin-bottom: 4px; }
.meta { color: #666; font-size: 0.9rem; margin-bottom: 16px; }
nav { background: #fff; border-radius: 8px; padding: 12px 20px; margin-bottom: 24px;
      box-shadow: 0 1px 4px rgba(0,0,0,.08); display: flex; flex-wrap: wrap; gap: 8px 16px;
      align-items: center; font-size: 0.88rem; }
nav a { color: #1a5296; text-decoration: none; font-weight: 500; }
nav a:hover { text-decoration: underline; }
nav .nav-sep { color: #ccc; }
h2 { font-size: 1.25rem; margin: 0 0 16px; }
h3 { font-size: 1rem; margin: 24px 0 8px; color: #333; border-bottom: 1px solid #eef; padding-bottom: 4px; }
.section { background: #fff; border-radius: 10px; padding: 24px; margin-bottom: 24px;
           box-shadow: 0 1px 4px rgba(0,0,0,.08); }
.gpu-section h2 { font-size: 1.2rem; background: #f0f2f8; margin: -24px -24px 20px;
                  padding: 14px 24px; border-radius: 10px 10px 0 0;
                  border-bottom: 2px solid #dde; }
.charts { display: flex; flex-wrap: wrap; gap: 20px; }
.charts img { border-radius: 6px; border: 1px solid #eee; max-width: 100%; }
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
th { background: #f0f2f8; text-align: left; padding: 8px 12px; border-bottom: 2px solid #ccd; }
th.sortable { cursor: pointer; user-select: none; white-space: nowrap; }
th.sortable:hover { background: #e0e4f0; }
th.sort-asc::after  { content: " ▲"; font-size: 0.7rem; }
th.sort-desc::after { content: " ▼"; font-size: 0.7rem; }
td { padding: 7px 12px; border-bottom: 1px solid #eee; }
tr:hover td { background: #fafbff; }
.best { font-weight: 700; color: #1a7340; }
.worst { color: #b03030; }
.note { font-size: 0.82rem; color: #666; margin-top: 8px; font-style: italic; }
.badge { display: inline-block; padding: 2px 7px; border-radius: 10px;
         font-size: 0.75rem; font-weight: 600; }
.badge-gpu  { background: #e8f0fe; color: #1a5296; }
.badge-nccl { background: #e6f4ea; color: #1a7340; }
details summary { cursor: pointer; font-weight: 600; color: #444; padding: 6px 0;
                  user-select: none; }
details summary:hover { color: #1a5296; }
h3 .desc { font-weight: 400; font-size: 0.8rem; color: #888; margin-left: 8px; }
@media print {
  body { background: #fff; padding: 12px; }
  nav { display: none; }
  .section { box-shadow: none; border: 1px solid #dde; page-break-inside: avoid; }
  .gpu-section { page-break-before: always; }
  .gpu-section:first-of-type { page-break-before: auto; }
  h3 { page-break-after: avoid; }
  .charts { display: block; }
  .charts img { max-width: 100%; width: 100%; margin-bottom: 12px; }
  details { display: block; }
  details > * { display: block; }
  details summary::marker { display: none; }
  table { page-break-inside: auto; font-size: 0.78rem; }
  tr { page-break-inside: avoid; }
  th, td { padding: 5px 8px; }
  th.sortable::after { display: none; }
}
"""

SORT_JS = r"""
<script>
document.querySelectorAll('table.sortable').forEach(function(table) {
  var tbody = table.querySelector('tbody');
  table.querySelectorAll('thead th').forEach(function(th, col) {
    th.classList.add('sortable');
    th.addEventListener('click', function() {
      var asc = !th.classList.contains('sort-asc');
      table.querySelectorAll('thead th').forEach(function(h) {
        h.classList.remove('sort-asc', 'sort-desc');
      });
      th.classList.add(asc ? 'sort-asc' : 'sort-desc');
      var rows = Array.from(tbody.querySelectorAll('tr'));
      rows.sort(function(a, b) {
        var av = a.cells[col] ? a.cells[col].textContent.trim() : '';
        var bv = b.cells[col] ? b.cells[col].textContent.trim() : '';
        var an = parseFloat(av.replace(/[^0-9.-]/g, ''));
        var bn = parseFloat(bv.replace(/[^0-9.-]/g, ''));
        if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
        return asc ? av.localeCompare(bv) : bv.localeCompare(av);
      });
      rows.forEach(function(r) { tbody.appendChild(r); });
    });
  });
});
</script>
"""


def img_tag(b64: str, alt: str = "") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}">'


def pct(val, ref):
    if ref == 0:
        return ""
    p = (val - ref) / ref * 100
    cls = "best" if p > 0 else "worst"
    sign = "+" if p > 0 else ""
    return f' <span class="{cls}">({sign}{p:.0f}%)</span>'


def summary_table(configs: list[Config]) -> str:
    # Metrics definition: (header, getter, unit, lower_is_better)
    metrics = [
        ("CPU 1T (ev/s)",      lambda c: c.cpu_single_evs,      "ev/s",  False),
        ("CPU MT (ev/s)",      lambda c: c.cpu_all_evs,          "ev/s",  False),
        ("Threads",            lambda c: float(c.cpu_all_threads),"",     False),
        ("Mem lat (ns)",       lambda c: c.mem_latency_dram_ns,  "ns",    True),
        ("STREAM Copy (MB/s)", lambda c: c.stream_copy,          "MB/s",  False),
        ("STREAM Triad (MB/s)",lambda c: c.stream_triad,         "MB/s",  False),
        ("H→D (GB/s)",         lambda c: c.gpu_h2d,              "GB/s",  False),
        ("D→H (GB/s)",         lambda c: c.gpu_d2h,              "GB/s",  False),
        ("D→D (GB/s)",         lambda c: c.gpu_d2d,              "GB/s",  False),
        ("MatMul (ms)",        lambda c: c.gpu_matmul_ms,        "ms",    True),
        ("NCCL peak (GB/s)",   lambda c: c.nccl_peak_busbw,      "GB/s",  False),
    ]

    # Per-GPU baseline: first config index for each gpu name
    gpu_baseline: dict[str, int] = {}
    for i, c in enumerate(configs):
        if c.gpu not in gpu_baseline:
            gpu_baseline[c.gpu] = i

    # Per-metric: find global best config index (across all configs)
    def best_index(getter, lower_is_better):
        vals = [getter(c) for c in configs]
        nonzero = [(v, i) for i, v in enumerate(vals) if v != 0]
        if not nonzero:
            return None
        return min(nonzero, key=lambda x: x[0])[1] if lower_is_better \
               else max(nonzero, key=lambda x: x[0])[1]

    headers = "".join(
        f"<th style='white-space:nowrap'>{h}</th>"
        for h, *_ in metrics)

    rows = ""
    for i, cfg in enumerate(configs):
        ref_i = gpu_baseline[cfg.gpu]
        cells = ""
        for _, getter, unit, lib in metrics:
            v = getter(cfg)
            bi = best_index(getter, lib)
            if v == 0:
                cells += "<td>—</td>"
                continue
            ref_v = getter(configs[ref_i])
            if i != ref_i and ref_v != 0:
                delta = pct(-v if lib else v, -ref_v if lib else ref_v)
            else:
                delta = ""
            cls = ' class="best"' if i == bi else ""
            fmt = f"{v:,.0f}" if unit == "" else f"{v:,.1f}"
            cells += f"<td{cls}>{fmt}{' ' + unit if unit else ''}{delta}</td>"
        rows += f"<tr><td><b>{cfg.label}</b></td>{cells}</tr>"

    return f"""
<table class="sortable">
<thead><tr><th>Config</th>{headers}</tr></thead>
<tbody>{rows}</tbody>
</table>
"""


def config_table(configs: list[Config]) -> str:
    rows = ""
    for c in configs:
        gpu_badge = f'<span class="badge badge-gpu">{c.gpu_device_name or c.gpu}</span>'
        nccl_badge = f'<span class="badge badge-nccl">{c.nccl_ngpus} GPUs (NCCL)</span>' \
                     if c.has_nccl else ""
        hp = f'<span class="badge badge-nccl">{c.hugepages_total} hugepages</span>' \
             if c.hugepages_total > 0 else "—"
        rows += f"<tr><td><b>{c.label}</b></td><td>{gpu_badge} {nccl_badge}</td>" \
                f"<td>{c.vcpus}</td><td>{c.ram_gb} GB</td><td>{hp}</td></tr>"
    return f"""
<table>
<thead><tr><th>Config</th><th>GPU</th><th>vCPUs</th><th>RAM</th><th>Hugepages</th></tr></thead>
<tbody>{rows}</tbody>
</table>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_configs(results_dir: Path) -> list[Config]:
    configs = []
    for gpu_dir in sorted(results_dir.iterdir()):
        if not gpu_dir.is_dir():
            continue
        for cfg_dir in sorted(gpu_dir.iterdir()):
            if cfg_dir.is_dir():
                cfg = load_config(cfg_dir)
                write_summary_csv(cfg, cfg_dir)
                configs.append(cfg)
    return configs


def _gpu_section_html(gpu_name: str, gpu_configs: list[Config]) -> str:
    """Build the HTML for one GPU group."""
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(gpu_configs))]
    labels = [c.name for c in gpu_configs]

    cpu_single_img = bar_chart(
        "CPU Single-thread (sysbench)", labels,
        [c.cpu_single_evs for c in gpu_configs], "events/sec", colors)
    cpu_all_img = bar_chart(
        "CPU Multi-thread (sysbench)", labels,
        [c.cpu_all_evs for c in gpu_configs], "events/sec", colors)
    cpu_per_thread_img = bar_chart(
        "CPU Throughput per Thread", labels,
        [c.cpu_all_evs / c.cpu_all_threads if c.cpu_all_threads else 0
         for c in gpu_configs],
        "events/sec/thread", colors)

    stream_img = stream_grouped_chart(gpu_configs, colors)
    mem_lat_img = mem_latency_chart(gpu_configs, colors)

    gpu_pcie_img = bar_chart(
        "GPU PCIe Bandwidth — Host↔Device", labels,
        [c.gpu_h2d for c in gpu_configs], "GB/s", colors)
    gpu_d2d_img = bar_chart(
        "GPU Device-to-Device Bandwidth", labels,
        [c.gpu_d2d for c in gpu_configs], "GB/s", colors)
    gpu_matmul_img = bar_chart(
        "GPU MatMul Latency (lower is better)", labels,
        [c.gpu_matmul_ms for c in gpu_configs], "ms/matmul", colors,
        highlight_max=False, highlight_min=True)

    # NCCL
    nccl_html = ""
    if any(c.has_nccl for c in gpu_configs):
        nccl_rows = ""
        for cfg in gpu_configs:
            if cfg.has_nccl:
                nccl_rows += (f"<tr><td><b>{cfg.name}</b></td>"
                              f"<td>{cfg.nccl_ngpus}</td>"
                              f"<td>{cfg.nccl_peak_busbw:.2f} GB/s</td>"
                              f"<td>{cfg.nccl_avg_busbw:.2f} GB/s</td></tr>")
            else:
                nccl_rows += (f"<tr><td><b>{cfg.name}</b></td>"
                              f"<td>—</td><td>—</td><td>—</td></tr>")

        overlay_img = nccl_overlay_chart(
            gpu_configs, colors,
            title=f"NCCL AllReduce Bus Bandwidth — {gpu_name}")
        overlay = img_tag(overlay_img, "NCCL overlay") if overlay_img else ""

        individual = []
        for cfg, color in zip(gpu_configs, colors):
            if cfg.has_nccl:
                img = nccl_chart(cfg, color)
                if img:
                    individual.append(img_tag(img, cfg.name))
        collapsible = ""
        if individual:
            charts_html = "".join(individual)
            collapsible = (f'<details style="margin-top:12px">'
                           f'<summary>Individual config charts ({len(individual)})</summary>'
                           f'<div class="charts" style="margin-top:12px">{charts_html}</div>'
                           f'</details>')

        nccl_html = f"""
<h3>7. NCCL / RCCL AllReduce <span class="desc">measures collective inter-GPU bandwidth (bus BW) — the bottleneck for distributed training across multiple GPUs</span></h3>
<table>
  <thead><tr><th>Config</th><th>GPUs</th><th>Peak Bus BW</th><th>Avg Bus BW</th></tr></thead>
  <tbody>{nccl_rows}</tbody>
</table>
<p class="note">float32 sum, out-of-place, 8B–1G. Single-GPU configs show —.</p>
<div class="charts" style="margin-top:12px">{overlay}</div>
{collapsible}"""

    n = len(gpu_configs)
    mem_lat_section = ""
    if mem_lat_img:
        mem_lat_section = f"""
<h3>2. Memory Latency — Pointer Chase <span class="desc">random pointer-chase reveals cache hierarchy (L1/L2/L3/DRAM) — lower is better</span></h3>
<div class="charts">{img_tag(mem_lat_img, "Memory latency")}</div>"""

    s = 3 if mem_lat_img else 2   # STREAM section number
    return f"""
<div class="section gpu-section" id="gpu-{gpu_name}">
  <h2>GPU: {gpu_name} <span style="font-size:0.85rem;font-weight:400;color:#666">({n} config{"s" if n != 1 else ""})</span></h2>

  <h3>1. CPU Performance (sysbench prime, 30 s) <span class="desc">integer prime-number generation — measures IPC, scheduler, and single-core clock</span></h3>
  <div class="charts">
    {img_tag(cpu_single_img, "CPU single-thread")}
    {img_tag(cpu_all_img, "CPU multi-thread")}
    {img_tag(cpu_per_thread_img, "CPU per-thread")}
  </div>

  {mem_lat_section}

  <h3>{s}. Memory Bandwidth — STREAM <span class="desc">four kernels (Copy/Scale/Add/Triad) stress the memory controller at different access patterns</span></h3>
  <div class="charts">{img_tag(stream_img, "STREAM")}</div>

  <h3>{s+1}. GPU ↔ CPU Memory Bandwidth <span class="desc">PCIe transfer speed between host RAM and GPU VRAM; bottlenecked by the interconnect, not the GPU itself</span></h3>
  <div class="charts">
    {img_tag(gpu_pcie_img, "GPU PCIe BW")}
    {img_tag(gpu_d2d_img, "GPU D2D BW")}
  </div>

  <h3>{s+2}. GPU Compute — MatMul (100 iterations) <span class="desc">FP16 4096³ matrix multiply repeated 100×; lower latency means more GPU FLOPS or better memory subsystem</span></h3>
  <div class="charts">{img_tag(gpu_matmul_img, "GPU matmul")}</div>

  {nccl_html}
</div>"""


_CHROMIUM_CANDIDATES = ["chromium", "chromium-browser", "google-chrome", "google-chrome-stable"]


def _write_pdf(html: str, output: Path):
    import shutil
    import subprocess
    import tempfile

    chrome = next((c for c in _CHROMIUM_CANDIDATES if shutil.which(c)), None)
    if chrome:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            f.write(html.encode())
            tmp_html = Path(f.name)
        try:
            subprocess.run(
                [chrome, "--headless", "--no-sandbox", "--disable-gpu",
                 f"--print-to-pdf={output.resolve()}",
                 "--print-to-pdf-no-header",
                 tmp_html.as_uri()],
                check=True, capture_output=True,
            )
            print(f"PDF report written to: {output.resolve()}  (via {chrome})")
        finally:
            tmp_html.unlink(missing_ok=True)
        return

    # Fallback: weasyprint
    try:
        import weasyprint
        weasyprint.HTML(string=html).write_pdf(output)
        print(f"PDF report written to: {output.resolve()}  (via weasyprint)")
    except ImportError:
        print(
            "ERROR: No Chromium/Chrome found and weasyprint is not installed.\n"
            "Install one of:\n"
            "  apt install chromium  /  apt install google-chrome-stable\n"
            "  pip install weasyprint",
            file=sys.stderr,
        )
        sys.exit(1)


def generate_report(results_dir: Path, output: Path):
    configs = discover_configs(results_dir)
    if not configs:
        print("No configs found.", file=sys.stderr)
        sys.exit(1)

    # Group by GPU, preserving discovery order
    gpu_groups: dict[str, list[Config]] = {}
    for cfg in configs:
        gpu_groups.setdefault(cfg.gpu, []).append(cfg)

    print(f"Found {len(configs)} config(s) across {len(gpu_groups)} GPU(s): "
          f"{', '.join(gpu_groups)}")

    from datetime import date
    today = date.today().isoformat()

    nav_links = " <span class='nav-sep'>|</span> ".join(
        f'<a href="#gpu-{g}">{g} ({len(cs)})</a>'
        for g, cs in gpu_groups.items())
    nav_links = (f'<a href="#summary">Summary</a>'
                 f' <span class="nav-sep">|</span> ') + nav_links

    gpu_sections = "".join(
        _gpu_section_html(gpu_name, gpu_configs)
        for gpu_name, gpu_configs in gpu_groups.items())

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>vCPU Benchmark Report</title>
<style>{CSS}</style>
</head>
<body>
<h1>vCPU Benchmark Report</h1>
<p class="meta">Generated: {today} &nbsp;|&nbsp; {len(configs)} config(s) &nbsp;|&nbsp; {len(gpu_groups)} GPU type(s)</p>
<nav>{nav_links}</nav>

<div class="section" id="summary">
  <h2>Summary — all configurations</h2>
  {summary_table(configs)}
  <p class="note">Bold green = best value. Percentages relative to the first config within
  the same GPU. Lower is better for latency columns.</p>
</div>

{gpu_sections}

{SORT_JS}
</body>
</html>
"""

    if output.suffix.lower() == ".pdf":
        _write_pdf(html, output)
    else:
        output.write_text(html)
        print(f"Report written to: {output.resolve()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate HTML benchmark report")
    parser.add_argument("results_dir", nargs="?", default="results",
                        help="Path to results directory (default: results)")
    parser.add_argument("-o", "--output", default="results/report.html",
                        help="Output HTML file (default: results/report.html)")
    parser.add_argument("--write-summary", metavar="DIR",
                        help="Parse one result directory and write summary.csv, then exit")
    args = parser.parse_args()

    if args.write_summary:
        p = Path(args.write_summary)
        # Always parse raw text files — do not let an existing summary.csv seed the values
        summary = p / "summary.csv"
        if summary.exists():
            summary.unlink()
        cfg = load_config(p)
        write_summary_csv(cfg, p)
        print(f"  summary.csv written to {p}/")
        sys.exit(0)

    generate_report(Path(args.results_dir), Path(args.output))
