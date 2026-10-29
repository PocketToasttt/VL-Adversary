import argparse
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SUMMARY_PATTERN = re.compile(
    r"=+\s*(?P<title>(?:Overall Summary|Results|Attack Type:)[^=\n]*?)\s*=+\s*"
    r".*?Collision Rate\s*:\s*(?P<collision_rate>[\d.]+)"
    r"\s*[±+/\-]+\s*(?P<collision_rate_std>[\d.]+)\s*"
    r".*?Mean Min TTC\s*:\s*(?P<min_ttc>[\d.]+)"
    r"\s*[±+/\-]+\s*(?P<min_ttc_std>[\d.]+)\s*"
    r".*?Mean Danger Exposure Rate\s*:\s*(?P<danger_rate>[\d.]+)"
    r"\s*[±+/\-]+\s*(?P<danger_rate_std>[\d.]+)\s*"
    r".*?Mean DAC\s*:\s*(?P<dac>[\d.]+)"
    r"\s*[±+/\-]+\s*(?P<dac_std>[\d.]+)\s*"
    r".*?Mean Comfortable\s*:\s*(?P<comfortable>[\d.]+)"
    r"\s*[±+/\-]+\s*(?P<comfortable_std>[\d.]+)"
    r"(?=\s*=+)",
    re.DOTALL,
)

FLOAT_FIELDS = {
    "collision_rate",
    "collision_rate_std",
    "min_ttc",
    "min_ttc_std",
    "danger_rate",
    "danger_rate_std",
    "dac",
    "dac_std",
    "comfortable",
    "comfortable_std",
}

ATTACK_TYPES = [
    "brake_check",
    "cut_in",
    "opposite_direction_intrusion",
]
ATTACK_LABELS = ["Brake Check", "Cut-in", "Opposite Intrusion"]


def parse_input(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("Input must use LABEL=PATH")
    label, path = value.split("=", 1)
    return label.strip(), Path(path.strip())


def parse_summaries(path):
    raw_text = path.read_bytes()
    encoding = (
        "utf-16"
        if raw_text.startswith((b"\xff\xfe", b"\xfe\xff"))
        else "utf-8"
    )
    text = raw_text.decode(encoding, errors="replace")
    summaries = []
    for match in SUMMARY_PATTERN.finditer(text):
        record = match.groupdict()
        record["title"] = record["title"].strip()
        for field in FLOAT_FIELDS:
            record[field] = float(record[field])
        summaries.append(record)
    if not summaries:
        raise ValueError(f"No metric summary found in {path}")
    return summaries


def overall_summary(summaries):
    return next(
        (
            summary
            for summary in summaries
            if "Overall Summary" in summary["title"]
        ),
        summaries[0],
    )


def attack_type_summaries(summaries):
    records = {}
    for summary in summaries:
        match = re.search(r"Attack Type:\s*\[([^\]]+)\]", summary["title"])
        if match:
            records[match.group(1)] = summary
    return records


def save_csv(path, records):
    fields = [
        "Condition",
        "Collision Rate (mean±std)",
        "Mean Min TTC (mean±std, cap=10)",
        "Mean Danger Exposure Rate (mean±std)",
        "Mean DAC (mean±std)",
        "Mean Comfortable (mean±std)",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "Condition": record["group"],
                    "Collision Rate (mean±std)": (
                        f"{record['collision_rate']:.4f} ± "
                        f"{record['collision_rate_std']:.4f}"
                    ),
                    "Mean Min TTC (mean±std, cap=10)": (
                        f"{record['min_ttc']:.4f} ± "
                        f"{record['min_ttc_std']:.4f}"
                    ),
                    "Mean Danger Exposure Rate (mean±std)": (
                        f"{record['danger_rate']:.4f} ± "
                        f"{record['danger_rate_std']:.4f}"
                    ),
                    "Mean DAC (mean±std)": (
                        f"{record['dac']:.4f} ± {record['dac_std']:.4f}"
                    ),
                    "Mean Comfortable (mean±std)": (
                        f"{record['comfortable']:.4f} ± "
                        f"{record['comfortable_std']:.4f}"
                    ),
                }
            )


def plot_paper_curves(overall_records, grouped_records, output_dir):
    metrics = [
        ("collision_rate", "collision_rate_std", "Collision Rate", (0, 1.0)),
        ("min_ttc", "min_ttc_std", "Mean Min TTC (s)", None),
        (
            "danger_rate",
            "danger_rate_std",
            "Danger Exposure Rate",
            (0, 0.5),
        ),
        ("dac", "dac_std", "Drivable Area Compliance", (0.7, 1.02)),
        (
            "comfortable",
            "comfortable_std",
            "Comfort Score",
            (0.65, 1.02),
        ),
    ]
    colors = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728"]
    markers = ["o", "s", "^", "D"]
    x_values = np.arange(len(ATTACK_TYPES))
    overall_by_group = {
        record["group"]: record for record in overall_records
    }

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 7.2))

    for metric_index, (field, std_field, title, limits) in enumerate(metrics):
        axis = axes.flat[metric_index]
        for group_index, (group, type_records) in enumerate(
            grouped_records.items()
        ):
            color = colors[group_index % len(colors)]
            marker = markers[group_index % len(markers)]
            has_breakdown = all(
                attack_type in type_records for attack_type in ATTACK_TYPES
            )
            if has_breakdown:
                values = [
                    type_records[attack_type][field]
                    for attack_type in ATTACK_TYPES
                ]
                errors = (
                    [
                        type_records[attack_type][std_field]
                        for attack_type in ATTACK_TYPES
                    ]
                    if std_field
                    else None
                )
                axis.errorbar(
                    x_values,
                    values,
                    yerr=errors,
                    color=color,
                    marker=marker,
                    markersize=6,
                    linewidth=2,
                    capsize=3,
                    label=group,
                )
            else:
                axis.axhline(
                    overall_by_group[group][field],
                    color=color,
                    linestyle="--",
                    linewidth=1.8,
                    label=f"{group} (overall ref.)",
                )

        axis.set_title(f"({chr(97 + metric_index)}) {title}")
        axis.set_xticks(x_values)
        axis.set_xticklabels(ATTACK_LABELS, rotation=10)
        axis.set_ylabel(title)
        if limits:
            axis.set_ylim(*limits)
        axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    for unused_axis in axes.flat[len(metrics):]:
        unused_axis.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles))
    figure.legend(
        unique_legend.values(),
        unique_legend.keys(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.99,
        bottom=0.09,
        top=0.88,
        wspace=0.32,
        hspace=0.38,
    )
    output_path = output_dir / "adversarial_comparison_curves"
    figure.savefig(
        output_path.with_suffix(".png"),
        dpi=400,
        bbox_inches="tight",
    )
    figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        type=parse_input,
        required=True,
        metavar="LABEL=PATH",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation_results/figures"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    overall_records = []
    grouped_records = {}
    for label, path in args.input:
        summaries = parse_summaries(path)
        overall = dict(overall_summary(summaries))
        overall["group"] = label
        overall_records.append(overall)
        grouped_records[label] = attack_type_summaries(summaries)

    save_csv(args.output_dir / "overall_metrics.csv", overall_records)
    plot_paper_curves(overall_records, grouped_records, args.output_dir)
    print(f"Saved publication figures to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
