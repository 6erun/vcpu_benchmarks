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
    m = re.search(r"Device \d+: (.+)", text)
    if m:
        cfg.gpu_device_name = m.group(1).strip()
    cfg.gpu_h2d = _first(r"Host to Device.*?(\d+\.\d+)\s*$", text, flags=re.S) or 0.0
    cfg.gpu_d2h = _first(r"Device to Host.*?(\d+\.\d+)\s*$", text, flags=re.S) or 0.0
    cfg.gpu_d2d = _first(r"Device to Device.*?(\d+\.\d+)\s*$", text, flags=re.S) or 0.0

    # More robust: grab all bandwidth values per section
    sections = re.split(r"(Host to Device|Device to Host|Device to Device)", text)
    mapping = {}
    for i in range(1, len(sections), 2):
        label = sections[i]
        body = sections[i + 1]
        bw_match = re.search(r"(\d+)\s+([\d.]+)\s*$", body.strip(), re.MULTILINE)
        if bw_match:
            mapping[label] = float(bw_match.group(2))
    cfg.gpu_h2d = mapping.get("Host to Device", cfg.gpu_h2d)
    cfg.gpu_d2h = mapping.get("Device to Host", cfg.gpu_d2h)
    cfg.gpu_d2d = mapping.get("Device to Device", cfg.gpu_d2d)


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


def load_config(path: Path) -> Config:
    gpu = path.parent.name
    name = path.name
    cfg = Config(gpu=gpu, name=name, label=f"{gpu} / {name}")

    def read(fname):
        p = path / fname
        return p.read_text() if p.exists() else ""

    parse_numa(read("numa_topology.txt"), cfg)
    cfg.cpu_single_evs, _ = parse_sysbench(read("cpu_single.txt"))
    cfg.cpu_all_evs, cfg.cpu_all_threads = parse_sysbench(read("cpu_all.txt"))
    parse_stream(read("stream.txt"), cfg)
    parse_gpu_bandwidth(read("gpu_bandwidth.txt"), cfg)
    parse_gpu_compute(read("gpu_compute.txt"), cfg)
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
    ax.set_ylim(0, max(values) * 1.2 if values else 1)
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


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #f5f6fa; color: #222; padding: 24px; }
h1 { font-size: 1.8rem; margin-bottom: 4px; }
.meta { color: #666; font-size: 0.9rem; margin-bottom: 32px; }
h2 { font-size: 1.2rem; margin: 32px 0 12px; border-bottom: 2px solid #dde; padding-bottom: 6px; }
h3 { font-size: 1rem; margin: 20px 0 8px; color: #444; }
.section { background: #fff; border-radius: 10px; padding: 24px; margin-bottom: 24px;
           box-shadow: 0 1px 4px rgba(0,0,0,.08); }
.charts { display: flex; flex-wrap: wrap; gap: 20px; }
.charts img { border-radius: 6px; border: 1px solid #eee; max-width: 100%; }
table { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
th { background: #f0f2f8; text-align: left; padding: 8px 12px; border-bottom: 2px solid #ccd; }
td { padding: 7px 12px; border-bottom: 1px solid #eee; }
tr:hover td { background: #fafbff; }
.best { font-weight: 700; color: #1a7340; }
.worst { color: #b03030; }
.note { font-size: 0.82rem; color: #666; margin-top: 8px; font-style: italic; }
.badge { display: inline-block; padding: 2px 7px; border-radius: 10px;
         font-size: 0.75rem; font-weight: 600; }
.badge-gpu  { background: #e8f0fe; color: #1a5296; }
.badge-nccl { background: #e6f4ea; color: #1a7340; }
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
    ref = configs[0]

    def row(label, getter, unit, lower_is_better=False):
        vals = [getter(c) for c in configs]
        best_i = min(range(len(vals)), key=lambda i: vals[i]) if lower_is_better \
                 else max(range(len(vals)), key=lambda i: vals[i])
        cells = ""
        for i, (c, v) in enumerate(zip(configs, vals)):
            if v == 0:
                cells += "<td>—</td>"
                continue
            ref_v = vals[0]
            delta = pct(v if not lower_is_better else -v,
                        ref_v if not lower_is_better else -ref_v) if i > 0 else ""
            cls = ' class="best"' if i == best_i else ""
            cells += f"<td{cls}>{v:,.1f} {unit}{delta}</td>"
        return f"<tr><td><b>{label}</b></td>{cells}</tr>"

    headers = "".join(f"<th>{c.label}</th>" for c in configs)
    return f"""
<table>
<thead><tr><th>Metric</th>{headers}</tr></thead>
<tbody>
{row("CPU single-thread (ev/s)", lambda c: c.cpu_single_evs, "ev/s")}
{row("CPU multi-thread (ev/s)", lambda c: c.cpu_all_evs, "ev/s")}
{row("CPU threads", lambda c: c.cpu_all_threads, "", False)}
{row("STREAM Copy (MB/s)", lambda c: c.stream_copy, "MB/s")}
{row("STREAM Triad (MB/s)", lambda c: c.stream_triad, "MB/s")}
{row("GPU H→D PCIe (GB/s)", lambda c: c.gpu_h2d, "GB/s")}
{row("GPU D→H PCIe (GB/s)", lambda c: c.gpu_d2h, "GB/s")}
{row("GPU D→D (GB/s)", lambda c: c.gpu_d2d, "GB/s")}
{row("GPU matmul latency (ms)", lambda c: c.gpu_matmul_ms, "ms", lower_is_better=True)}
{row("NCCL peak bus BW (GB/s)", lambda c: c.nccl_peak_busbw, "GB/s")}
</tbody>
</table>
"""


def config_table(configs: list[Config]) -> str:
    rows = ""
    for c in configs:
        gpu_badge = f'<span class="badge badge-gpu">{c.gpu_device_name or c.gpu}</span>'
        nccl_badge = f'<span class="badge badge-nccl">{c.nccl_ngpus} GPUs (NCCL)</span>' \
                     if c.has_nccl else ""
        rows += f"<tr><td><b>{c.label}</b></td><td>{gpu_badge} {nccl_badge}</td>" \
                f"<td>{c.vcpus}</td><td>{c.ram_gb} GB</td></tr>"
    return f"""
<table>
<thead><tr><th>Config</th><th>GPU</th><th>vCPUs</th><th>RAM</th></tr></thead>
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
                configs.append(load_config(cfg_dir))
    return configs


def generate_report(results_dir: Path, output: Path):
    configs = discover_configs(results_dir)
    if not configs:
        print("No configs found.", file=sys.stderr)
        sys.exit(1)

    colors = PALETTE[: len(configs)]
    labels = [c.label for c in configs]

    print(f"Found {len(configs)} config(s): {', '.join(labels)}")

    # --- build charts ---
    cpu_single_img = bar_chart(
        "CPU Single-thread (sysbench)", labels,
        [c.cpu_single_evs for c in configs], "events/sec", colors)

    cpu_all_img = bar_chart(
        "CPU Multi-thread (sysbench)", labels,
        [c.cpu_all_evs for c in configs], "events/sec", colors)

    cpu_per_thread_img = bar_chart(
        "CPU Throughput per Thread", labels,
        [c.cpu_all_evs / c.cpu_all_threads if c.cpu_all_threads else 0 for c in configs],
        "events/sec/thread", colors)

    stream_img = stream_grouped_chart(configs, colors)

    gpu_pcie_img = bar_chart(
        "GPU PCIe Bandwidth — Host↔Device", labels,
        [c.gpu_h2d for c in configs], "GB/s", colors)

    gpu_d2d_img = bar_chart(
        "GPU Device-to-Device Bandwidth", labels,
        [c.gpu_d2d for c in configs], "GB/s", colors)

    gpu_matmul_img = bar_chart(
        "GPU MatMul Latency (4096×4096 FP32, lower is better)", labels,
        [c.gpu_matmul_ms for c in configs], "ms/matmul", colors,
        highlight_max=False, highlight_min=True)

    nccl_imgs = []
    for cfg, color in zip(configs, colors):
        if cfg.has_nccl:
            img = nccl_chart(cfg, color)
            if img:
                nccl_imgs.append((cfg, img))

    # --- assemble HTML ---
    from datetime import date
    today = date.today().isoformat()

    nccl_section = ""
    if nccl_imgs:
        rows = ""
        for cfg, _ in nccl_imgs:
            rows += f"<tr><td><b>{cfg.label}</b></td>" \
                    f"<td>{cfg.nccl_ngpus}</td>" \
                    f"<td>{cfg.nccl_peak_busbw:.2f} GB/s</td>" \
                    f"<td>{cfg.nccl_avg_busbw:.2f} GB/s</td></tr>"
        charts = "".join(img_tag(img, cfg.label) for cfg, img in nccl_imgs)
        nccl_section = f"""
<div class="section">
  <h2>5. NCCL AllReduce</h2>
  <table>
    <thead><tr><th>Config</th><th>GPUs</th><th>Peak Bus BW</th><th>Avg Bus BW</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <p class="note">float32 sum, out-of-place, message sizes 8B–256MB</p>
  <div class="charts" style="margin-top:16px">{charts}</div>
</div>
"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>vCPU Benchmark Report</title>
<style>{CSS}</style>
</head>
<body>
<h1>vCPU Benchmark Report</h1>
<p class="meta">Generated: {today} &nbsp;|&nbsp; {len(configs)} configuration(s) &nbsp;|&nbsp;
Results directory: <code>{results_dir.resolve()}</code></p>

<div class="section">
  <h2>Configurations</h2>
  {config_table(configs)}
</div>

<div class="section">
  <h2>Summary</h2>
  {summary_table(configs)}
  <p class="note">Bold green = best value in row. Percentages relative to first config.
  Lower is better for matmul latency.</p>
</div>

<div class="section">
  <h2>1. CPU Performance (sysbench prime, 30 s)</h2>
  <div class="charts">
    {img_tag(cpu_single_img, "CPU single-thread")}
    {img_tag(cpu_all_img, "CPU multi-thread")}
    {img_tag(cpu_per_thread_img, "CPU per-thread")}
  </div>
</div>

<div class="section">
  <h2>2. Memory Bandwidth — STREAM</h2>
  <div class="charts">
    {img_tag(stream_img, "STREAM")}
  </div>
</div>

<div class="section">
  <h2>3. GPU ↔ CPU Memory Bandwidth (CUDA bandwidthTest)</h2>
  <div class="charts">
    {img_tag(gpu_pcie_img, "GPU PCIe BW")}
    {img_tag(gpu_d2d_img, "GPU D2D BW")}
  </div>
</div>

<div class="section">
  <h2>4. GPU Compute — MatMul (4096×4096 FP32, 100 iterations)</h2>
  <div class="charts">
    {img_tag(gpu_matmul_img, "GPU matmul")}
  </div>
</div>

{nccl_section}

</body>
</html>
"""

    output.write_text(html)
    print(f"Report written to: {output.resolve()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate HTML benchmark report")
    parser.add_argument("results_dir", nargs="?", default="results",
                        help="Path to results directory (default: results)")
    parser.add_argument("-o", "--output", default="results/report.html",
                        help="Output HTML file (default: results/report.html)")
    args = parser.parse_args()

    generate_report(Path(args.results_dir), Path(args.output))
