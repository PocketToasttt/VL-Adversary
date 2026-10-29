"""Render the four passed road-shape SUMO FCD runs as a short GIF."""
import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import sumolib

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "TrafficManager/networkFiles/boston-seaport"
PICKS = ("straight_01", "curve_01", "intersection_02", "merge_01")
NAMES = ("Straight road", "Curved road", "Intersection", "Converging roads")


def read_fcd(path):
    frames = {}
    for _, step in ET.iterparse(path, events=("end",)):
        if step.tag != "timestep":
            continue
        at = round(float(step.get("time")), 1)
        if at <= 9.5:
            frames[at] = [(v.get("id"), float(v.get("x")), float(v.get("y")))
                          for v in step.findall("vehicle")]
        step.clear()
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    args = parser.parse_args()
    report = {r["scene_id"]: r for r in json.loads(
        (args.scenes / "smoke_report.json").read_text(encoding="utf-8"))}
    for scene_id in PICKS:
        if not report[scene_id]["pass"]:
            parser.error(f"{scene_id} did not pass SUMO smoke screening")
    net = sumolib.net.readNet(str(MAP / "osm.net.xml"))
    data = []
    for scene_id in PICKS:
        row = report[scene_id]
        ego = next(v for v in row["snapshot_t2"]["vehicles"] if v["id"] == "0")
        center = float(ego["x"]), float(ego["y"])
        fcd = args.scenes / scene_id / f"{scene_id}.fcd.xml"
        data.append((center, read_fcd(fcd)))
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), dpi=100)
    fig.patch.set_facecolor("#f8fafc")
    scatters = []
    for ax, ((cx, cy), _), name in zip(axes.flat, data, NAMES):
        for road in net.getEdges():
            for lane in road.getLanes():
                shape = lane.getShape()
                if (max(p[0] for p in shape) < cx - 93 or min(p[0] for p in shape) > cx + 93
                        or max(p[1] for p in shape) < cy - 93 or min(p[1] for p in shape) > cy + 93):
                    continue
                ax.plot([p[0] for p in shape], [p[1] for p in shape],
                        color="#cbd5e1", linewidth=1.1, zorder=1)
        npc = ax.scatter([], [], c="#2563eb", s=21, zorder=3)
        ego = ax.scatter([], [], c="#e63946", marker="*", s=120, zorder=4)
        scatters.append((npc, ego))
        ax.set_xlim(cx - 90, cx + 90)
        ax.set_ylim(cy - 90, cy + 90)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=6)
        ax.grid(color="#edf2f7", linewidth=.4)
        ax.set_title(name, loc="left", fontsize=10, fontweight="bold")

    def update(frame):
        at = round(frame * .5, 1)
        for ((_, frames), (npc, ego)) in zip(data, scatters):
            cars = frames.get(at, [])
            ego_car = next((v for v in cars if v[0] == "0"), None)
            if ego_car:
                ex, ey = ego_car[1:]
                ego.set_offsets([[ex, ey]])
                points = [(x, y) for vid, x, y in cars if vid != "0"
                          and math.hypot(x - ex, y - ey) <= 100]
            else:
                ego.set_offsets([[float("nan"), float("nan")]])
                points = []
            npc.set_offsets(points if points else [[float("nan"), float("nan")]])
        fig.suptitle(f"SUMO traffic  |  t = {at:.1f} s  |  red: ego",
                     fontsize=13, fontweight="bold")
        return [artist for pair in scatters for artist in pair]

    animation = FuncAnimation(fig, update, frames=20, interval=500, blit=False)
    fig.tight_layout(rect=(0, 0, 1, .96))
    out = args.scenes / ("road_shapes_10s.gif" if report[PICKS[0]]["density_profile"] == "balanced"
                         else "high_density_road_shapes_10s.gif")
    animation.save(str(out), writer=PillowWriter(fps=2))
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
