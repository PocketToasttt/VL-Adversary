#!/usr/bin/env python3
"""Plot a single-axis normalized trajectory-realism profile.

The script consumes ``trajectory_realism_results.json`` produced by
``trajectory_realism_evaluator.py``.  It creates one publication-style figure
with five kinematic metrics on the x-axis and normalized scene-level
Wasserstein distance (mean +/- standard deviation) on the y-axis.

No min-max normalization is used: with only two methods it would force each
metric to 0 or 1 and exaggerate small differences.  Instead, every WD is
divided by the corresponding benign P99 threshold, preserving the magnitude
of each distribution shift while making the five physical units comparable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    ("speed", "Speed"),
    ("longitudinal_acceleration", "Long. Acc."),
    ("lateral_acceleration", "Lat. Acc."),
    ("longitudinal_jerk", "Long. Jerk"),
    ("yaw_rate", "Yaw Rate"),
)

DEFAULT_STYLES = (
    {"color": "#0072B2", "marker": "o", "linestyle": "-"},
    {"color": "#D55E00", "marker": "s", "linestyle": "--"},
    {"color": "#009E73", "marker": "^", "linestyle": "-."},
    {"color": "#CC79A7", "marker": "D", "linestyle": ":"},
)


def load_results(path: Path) -> Mapping[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("thresholds"), dict):
        raise ValueError("Input JSON does not contain a thresholds object")
    if not isinstance(data.get("summaries"), list) or not data["summaries"]:
        raise ValueError("Input JSON does not contain method summaries")
    return data


def normalized_profile(
    summary: Mapping[str, object], thresholds: Mapping[str, object]
) -> Tuple[np.ndarray, np.ndarray]:
    means: List[float] = []
    stds: List[float] = []
    for key, _ in METRICS:
        scale = float(thresholds[key])
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid normalization threshold for {key}: {scale}")
        mean = float(summary[f"wd_{key}_scene_mean"])
        std = float(summary[f"wd_{key}_scene_std"])
        means.append(mean / scale)
        stds.append(std / scale)
    return np.asarray(means), np.asarray(stds)


def plot_profile(
    data: Mapping[str, object],
    output: Path,
    title: Optional[str],
    dpi: int,
) -> None:
    thresholds = data["thresholds"]
    summaries = data["summaries"]
    x = np.arange(len(METRICS), dtype=float)

    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for index, summary in enumerate(summaries):
        means, stds = normalized_profile(summary, thresholds)
        style: Dict[str, object] = dict(DEFAULT_STYLES[index % len(DEFAULT_STYLES)])
        axis.errorbar(
            x,
            means,
            yerr=stds,
            label=str(summary["method"]),
            linewidth=2.1,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.8,
            capsize=3.5,
            elinewidth=1.2,
            alpha=0.95,
            **style,
        )

    axis.axhline(
        0.0,
        color="#555555",
        linewidth=1.0,
        linestyle=":",
        label="Benign reference",
        zorder=0,
    )
    axis.set_xticks(x)
    axis.set_xticklabels([label for _, label in METRICS])
    axis.set_ylabel("Normalized Wasserstein Distance ↓")
    axis.set_xlabel("Attacker-trajectory kinematic metric")
    if title:
        axis.set_title(title)
    axis.set_xlim(-0.25, len(METRICS) - 0.75)
    axis.set_ylim(bottom=0.0)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    axis.grid(axis="x", visible=False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        frameon=False,
        ncol=min(3, len(summaries) + 1),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    if output.suffix.lower() != ".pdf":
        figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot all five trajectory-realism metrics in one figure."
    )
    parser.add_argument(
        "--input",
        type=lambda value: Path(value).expanduser().resolve(),
        default=Path(
            "evaluation_results/trajectory_realism/trajectory_realism_results.json"
        ).resolve(),
    )
    parser.add_argument(
        "--output",
        type=lambda value: Path(value).expanduser().resolve(),
        default=Path(
            "evaluation_results/trajectory_realism/trajectory_realism_profile.png"
        ).resolve(),
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")
    try:
        data = load_results(args.input)
        plot_profile(data, args.output, args.title, args.dpi)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(f"Saved: {args.output}")
    if args.output.suffix.lower() != ".pdf":
        print(f"Saved: {args.output.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
