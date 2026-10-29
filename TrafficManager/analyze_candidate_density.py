"""Read-only exploratory analysis of VLM-visible traffic density.

The script prints descriptive statistics and Spearman correlations. It does
not create figures or modify experiment results.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


SCENE_PATTERN = re.compile(
    r"scene_id=(?P<scene_id>[^,]+), "
    r"attack_type=(?P<attack_type>[^,]+), "
    r"collision=(?P<collision>[01]), "
    r"min_ttc=(?P<min_ttc>[^,]+), "
    r"danger_exposure=(?P<danger_exposure>[\d.]+), "
    r"dac=(?P<dac>[\d.]+), "
    r"comfortable=(?P<comfortable>[\d.]+)"
)

ALLOWED_ATTACKS_PATTERN = re.compile(r"allowed_attacks=\[([^\]]*)\]")

METRICS = (
    "collision",
    "min_ttc_capped",
    "danger_exposure",
    "dac",
    "comfortable",
)


def read_text_auto(path: Path) -> str:
    raw = path.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
    return raw.decode(encoding, errors="replace")


def parse_metrics(path: Path, ttc_cap: float):
    rows = {}
    for match in SCENE_PATTERN.finditer(read_text_auto(path)):
        values = match.groupdict()
        raw_ttc = values["min_ttc"]
        min_ttc = np.inf if raw_ttc == "inf" else float(raw_ttc)
        values.update(
            {
                "collision": float(values["collision"]),
                "min_ttc_capped": min(min_ttc, ttc_cap),
                "danger_exposure": float(values["danger_exposure"]),
                "dac": float(values["dac"]),
                "comfortable": float(values["comfortable"]),
            }
        )
        rows[values["scene_id"]] = values
    return rows


def parse_density_records(result_dir: Path):
    rows = {}
    for record_path in result_dir.glob("*/attack_record.json"):
        if not (record_path.parent / "drive_arena.pkl").exists():
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        scene_id = str(record.get("scene_name") or record_path.parent.name)
        matches = ALLOWED_ATTACKS_PATTERN.findall(
            str(record.get("structured_scene", ""))
        )
        rows[scene_id] = {
            "visible_background_count": len(matches),
            "eligible_candidate_count": sum(
                bool(value.strip()) for value in matches
            ),
        }
    return rows


def planned_attack_type(scene_id: str) -> str:
    return re.sub(r"_\d+$", "", scene_id)


def print_distribution(label, values):
    values = np.asarray(values, dtype=float)
    unique, counts = np.unique(values.astype(int), return_counts=True)
    distribution = ", ".join(
        f"{value}:{count}" for value, count in zip(unique, counts)
    )
    print(
        f"{label}: min={values.min():.0f}, max={values.max():.0f}, "
        f"mean={values.mean():.3f}, std={values.std():.3f}; "
        f"counts={{ {distribution} }}"
    )


def print_correlations(rows, density_field, title):
    print(f"\n{title}")
    density = np.asarray([row[density_field] for row in rows], dtype=float)
    if len(np.unique(density)) < 2:
        print("  No variation in density; correlations are undefined.")
        return
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        rho, p_value = spearmanr(density, values)
        print(f"  {metric:20s} rho={rho: .4f}, p={p_value:.4g}")


def analyze_condition(label, result_dir, metric_path, ttc_cap):
    metrics = parse_metrics(metric_path, ttc_cap)
    density = parse_density_records(result_dir)
    scene_ids = sorted(set(metrics).intersection(density))
    rows = []
    for scene_id in scene_ids:
        row = dict(metrics[scene_id])
        row.update(density[scene_id])
        row["planned_attack_type"] = planned_attack_type(scene_id)
        rows.append(row)

    print(f"\n{'=' * 72}\n{label}: matched scenes={len(rows)}\n{'=' * 72}")
    if not rows:
        return
    print_distribution(
        "VLM-visible background vehicles",
        [row["visible_background_count"] for row in rows],
    )
    print_distribution(
        "Eligible candidates (non-empty allowed_attacks)",
        [row["eligible_candidate_count"] for row in rows],
    )
    print("\nDescriptive means by planned scenario category")
    for attack_type in sorted({row["planned_attack_type"] for row in rows}):
        subset = [row for row in rows if row["planned_attack_type"] == attack_type]
        means = {
            field: float(np.mean([row[field] for row in subset]))
            for field in (
                "visible_background_count",
                "eligible_candidate_count",
                *METRICS,
            )
        }
        print(
            f"  {attack_type:30s} n={len(subset):2d}, "
            f"visible={means['visible_background_count']:.3f}, "
            f"eligible={means['eligible_candidate_count']:.3f}, "
            f"collision={means['collision']:.3f}, "
            f"min_ttc={means['min_ttc_capped']:.3f}, "
            f"danger={means['danger_exposure']:.3f}, "
            f"dac={means['dac']:.3f}, "
            f"comfort={means['comfortable']:.3f}"
        )
    print_correlations(
        rows,
        "visible_background_count",
        "Overall Spearman correlations: visible background count",
    )
    print_correlations(
        rows,
        "eligible_candidate_count",
        "Overall Spearman correlations: eligible candidate count",
    )

    for attack_type in sorted({row["planned_attack_type"] for row in rows}):
        subset = [row for row in rows if row["planned_attack_type"] == attack_type]
        print_correlations(
            subset,
            "eligible_candidate_count",
            f"Stratified by planned category: {attack_type} (n={len(subset)})",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttc-cap", type=float, default=10.0)
    args = parser.parse_args()

    analyze_condition(
        "Closed-loop attack",
        Path("results/batch_attack_closedloop"),
        Path("evaluation_results/attack_closedloop_metrics.txt"),
        args.ttc_cap,
    )
    analyze_condition(
        "Open-loop attack",
        Path("results/batch_attack_openloop"),
        Path("evaluation_results/attack_openloop_metrics.txt"),
        args.ttc_cap,
    )


if __name__ == "__main__":
    main()
