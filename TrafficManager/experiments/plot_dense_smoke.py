"""Summarize paired SUMO smoke runs and draw a local traffic preview."""
import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "TrafficManager/networkFiles/boston-seaport"


def road_shapes():
    shapes = []
    for _, element in ET.iterparse(MAP / "osm.net.xml", events=("end",)):
        if element.tag == "edge" and not element.get("id", "").startswith(":"):
            for lane in element.findall("lane"):
                raw = lane.get("shape", "")
                if raw:
                    shapes.append([tuple(map(float, item.split(","))) for item in raw.split()])
            element.clear()
    return shapes


def corrected_counts(data, planned):
    log = Path(data["log_path"]).read_text(encoding="utf-8")
    inserted = re.search(r"Inserted:\s*(\d+)", log)
    data["loaded"] = planned
    data["inserted"] = int(inserted.group(1)) if inserted else None
    data["ego_present_at_2s"] = "2.0" in data.get("snapshots", {})
    data["smoke_pass"] = (data["returncode"] == 0 and data["inserted"] == planned
                          and data["ego_present_at_2s"] and data["attacker_at_2s"]
                          and data["collision_count"] == 0)


def summarize(rows, index, output):
    by_id = {scene["scene_id"]: scene for scene in index["scenes"]}
    report_rows = []
    for row in rows:
        scene = by_id[row["scene_id"]]
        for label, planned in (("original", scene["vehicle_count"] - len(scene["added_local_vehicle_ids"])),
                               ("dense", scene["vehicle_count"])):
            # The source count is read from its route file because some pilot
            # layouts have fewer added cars than the nominal target.
            if label == "original":
                old = MAP / "scenarios_diverse_3types" / row["source_scene_id"] / (row["source_scene_id"] + ".rou.xml")
                planned = len(ET.parse(old).getroot().findall("vehicle"))
            corrected_counts(row[label], planned)
        old = row["original"]
        new = row["dense"]
        before = old.get("snapshots", {}).get("2.0", {}).get("nearby_60m")
        after = new.get("snapshots", {}).get("2.0", {}).get("nearby_60m")
        report_rows.append({"scene_id": row["scene_id"], "attack_type": row["attack_type"],
                            "original_vehicles": old["loaded"], "dense_vehicles": new["loaded"],
                            "original_inserted": old["inserted"], "dense_inserted": new["inserted"],
                            "nearby_at_2s_before": before, "nearby_at_2s_after": after,
                            "added_nearby_at_2s": new.get("snapshots", {}).get("2.0", {}).get("added_nearby_60m"),
                            "original_collisions": old["collision_count"], "dense_collisions": new["collision_count"],
                            "smoke_pass": new["smoke_pass"],
                            "failure_reasons": [name for name, test in (
                                ("SUMO return code", new["returncode"] != 0),
                                ("incomplete insertion", new["inserted"] != new["loaded"]),
                                ("ego absent at 2 s", not new["ego_present_at_2s"]),
                                ("attacker absent at 2 s", not new["attacker_at_2s"]),
                                ("SUMO collision", new["collision_count"] > 0),
                            ) if test]})
    (output / "smoke_report.json").write_text(json.dumps(report_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    valid = [row for row in report_rows if row["smoke_pass"]]
    mean_delta = sum(row["nearby_at_2s_after"] - row["nearby_at_2s_before"] for row in valid) / len(valid)
    lines = ["# Dense traffic SUMO smoke test", "",
             "Paired original and generated scenes; 10 s, 0.1 s SUMO steps, no attack or UniAD.",
             f"Passed: {len(valid)}/{len(report_rows)} generated scenes. Mean increase among passing scenes "
             f"within 60 m of ego at 2 s: {mean_delta:.2f} vehicles.", "",
             "| Scene | Inserted / loaded | Nearby 60 m at 2 s | SUMO collisions | Status |",
             "|---|---:|---:|---:|---|"]
    for row in report_rows:
        near = f"{row['nearby_at_2s_before']} → {row['nearby_at_2s_after']}"
        lines.append(f"| {row['scene_id']} | {row['dense_inserted']}/{row['dense_vehicles']} | "
                     f"{near} | {row['dense_collisions']} | "
                     f"{'PASS' if row['smoke_pass'] else ', '.join(row['failure_reasons'])} |")
    lines += ["", "This is plain SUMO traffic simulation. It does not validate UniAD handover or DreamForge rendering."]
    (output / "smoke_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_rows


def draw_preview(rows, index, output):
    lookup = {row["scene_id"]: row for row in rows}
    scenes = {scene["scene_id"]: scene for scene in index["scenes"]}
    picks = ["brake_check_dense_002", "cut_in_dense_002", "opposite_direction_intrusion_dense_002"]
    shapes = road_shapes()
    fig, axes = plt.subplots(3, 2, figsize=(12, 15), dpi=150)
    fig.patch.set_facecolor("#f8fafc")
    for row_idx, scene_id in enumerate(picks):
        row = lookup[scene_id]
        scene = scenes[scene_id]
        center_data = row["original"]["snapshots"]["2.0"]["ego"]
        cx, cy = float(center_data["x"]), float(center_data["y"])
        for col, label in enumerate(("original", "dense")):
            ax = axes[row_idx][col]
            ax.set_facecolor("#ffffff")
            for shape in shapes:
                xs = [p[0] for p in shape]
                ys = [p[1] for p in shape]
                if max(xs) < cx - 70 or min(xs) > cx + 70 or max(ys) < cy - 70 or min(ys) > cy + 70:
                    continue
                ax.plot(xs, ys, color="#d5dde5", linewidth=1.6, zorder=1)
            snapshot = row[label]["snapshots"]["2.0"]
            for vehicle in snapshot["vehicles"]:
                vid = vehicle["id"]
                if vid == str(scene["ego_id"]):
                    color, marker, size = "#e63946", "*", 180
                elif vid == str(scene["attacker_candidate_id"]):
                    color, marker, size = "#f59e0b", "D", 75
                elif label == "dense" and int(vid) in scene["added_local_vehicle_ids"]:
                    color, marker, size = "#2563eb", "o", 55
                else:
                    color, marker, size = "#64748b", "o", 43
                x, y = float(vehicle["x"]), float(vehicle["y"])
                ax.scatter(x, y, c=color, marker=marker, s=size, edgecolors="white", linewidths=.6, zorder=3)
            ax.set_xlim(cx - 60, cx + 60)
            ax.set_ylim(cy - 60, cy + 60)
            ax.set_aspect("equal")
            ax.grid(color="#edf2f7", linewidth=.5)
            ax.tick_params(labelsize=7)
            title = f"{'Original' if col == 0 else 'Denser'} · {snapshot['nearby_60m']} nearby / 60 m"
            ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
            if col == 0:
                ax.set_ylabel(scene["attack_type"].replace("_", " ") + "\nmap y (m)", fontsize=10)
            if row_idx == 2:
                ax.set_xlabel("map x (m)", fontsize=9)
    fig.suptitle("SUMO traffic at t = 2.0 s  |  red: ego   orange: planned attacker   blue: added local traffic",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(output / "traffic_preview_t2.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=MAP / "scenarios_dense_closedloop_pilot_v2/scene_index.json")
    args = parser.parse_args()
    rows = json.loads(args.summary.read_text(encoding="utf-8"))
    index = json.loads(args.index.read_text(encoding="utf-8"))
    output = args.summary.parent
    report = summarize(rows, index, output)
    draw_preview(rows, index, output)
    print("pass", sum(row["smoke_pass"] for row in report), "of", len(report))
    print(output / "traffic_preview_t2.png")


if __name__ == "__main__":
    main()
