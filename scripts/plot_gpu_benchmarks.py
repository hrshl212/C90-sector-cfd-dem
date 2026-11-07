#!/usr/bin/env python3
"""Plot the public GPU benchmark tables in data/benchmarks/."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK = "#1f2933"
MUTED = "#6b7280"
GREEN = "#2f6f4e"
RUST = "#b5532c"
SAND = "#d9a441"
GRAY = "#9aa5b1"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=INK)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_capacity(data: Path, out: Path) -> None:
    table = rows(data / "single_gpu_particle_capacity.csv")
    labels, values, colors, notes = [], [], [], []
    for row in table:
        millions = int(row["particles"]) / 1e6
        cfg = row["configuration"]
        if row["status"] != "completed":
            labels.append(f"{millions:.0f}M\nuniform search")
            values.append(0.0)
            colors.append(GRAY)
            notes.append("crash: 31.2B neighbor\nentries > 2^32 CSR limit")
            continue
        if cfg.endswith("every_substep"):
            tag, color = "uniform, rebuild\nevery substep", GRAY
        elif cfg.endswith("once_per_cfd_step"):
            tag, color = "uniform, rebuild\nonce per step", SAND
        else:
            tag, color = f"size-aware bins\n{row['dem_substeps']} substeps", GREEN
        labels.append(f"{millions:.0f}M\n{tag}")
        values.append(float(row["mean_step_s"]))
        colors.append(color)
        notes.append(f"{float(row['mean_step_s']):.2f} s")

    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=150)
    bars = ax.bar(range(len(values)), values, color=colors, width=0.68)
    for bar, note, value in zip(bars, notes, values):
        y = value + 0.3 if value else 0.4
        ax.text(bar.get_x() + bar.get_width() / 2, y, note, ha="center",
                va="bottom", fontsize=8.5, color=RUST if not value else INK)
    ax.set_xticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylabel("Wall time per CFD step on one H100 (s)", color=INK)
    ax.set_title("Particle capacity on a single NVIDIA H100 (80 GB)",
                 color=INK, loc="left", fontsize=12)
    ax.set_ylim(0, 20)
    style(ax)
    fig.tight_layout()
    fig.savefig(out / "single_gpu_particle_capacity.png")
    plt.close(fig)


def plot_scaling(data: Path, out: Path) -> None:
    table = rows(data / "multi_gpu_strong_scaling.csv")
    gpus = [row["gpus"] for row in table]
    parts = (("fluid_s", "Fluid + pressure", GREEN),
             ("coupling_s", "Fluid-particle coupling", RUST),
             ("particles_s", "Particles (DEM)", SAND))
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150,
                                      gridspec_kw={"width_ratios": [1.3, 1]})
    bottom = [0.0] * len(table)
    for key, label, color in parts:
        values = [float(row[key]) for row in table]
        left.bar(gpus, values, bottom=bottom, color=color, label=label, width=0.6)
        bottom = [b + v for b, v in zip(bottom, values)]
    for x, row in zip(gpus, table):
        left.text(x, float(row["total_s"]) + 0.02, f"{float(row['total_s']):.3f} s",
                  ha="center", fontsize=8.5, color=INK)
    left.set_xlabel("H100 GPUs (one MPI rank per GPU)", color=INK)
    left.set_ylabel("Mean time per step (s)", color=INK)
    left.set_title("Strong scaling, 5-box mesh: slower beyond one GPU",
                   color=INK, loc="left", fontsize=11)
    left.legend(frameon=False, fontsize=8, loc="upper left")
    style(left)

    base = float(table[0]["particles_s"])
    speedup = [base / float(row["particles_s"]) for row in table]
    right.plot([int(g) for g in gpus], speedup, "o-", color=SAND, label="DEM stage")
    right.plot([int(g) for g in gpus], [int(g) for g in gpus], "--", color=GRAY,
               label="ideal")
    right.set_xscale("log", base=2)
    right.set_yscale("log", base=2)
    right.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    right.set_yticks([1, 2, 4, 8], ["1x", "2x", "4x", "8x"])
    right.set_xlabel("H100 GPUs", color=INK)
    right.set_title("Only DEM speeds up: 1.8x on 8 GPUs", color=INK, loc="left", fontsize=11)
    right.legend(frameon=False, fontsize=8)
    style(right)
    fig.tight_layout()
    fig.savefig(out / "multi_gpu_strong_scaling.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "benchmarks")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build" / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_capacity(args.data_dir, args.output_dir)
    plot_scaling(args.data_dir, args.output_dir)
    print(f"wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
