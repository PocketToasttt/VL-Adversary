"""Generate and SUMO-screen dense traffic previews across four road shapes.

These are traffic previews, not attack scenes or UniAD closed-loop results.
Requires sumolib, SUMO, and matplotlib in the active Python environment.
"""
import argparse
import json
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sumolib


ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "TrafficManager/networkFiles/boston-seaport"
NET = MAP / "osm.net.xml"
CANDIDATES = {
    "straight": ["-1120503844", "426488940#2"],
    "curve": ["8640122", "544506505#1"],
    "intersection": ["292967029#1", "426491386#0"],
    "merge": ["236579256", "8648213#0"],
}
TYPES = {
    "ego": {"color": "255,47,72", "maxSpeed": "11.0"},
    "traffic": {"color": "37,99,235", "maxSpeed": "11.0"},
}


def connected_next(edge):
    return [other for other in edge.getOutgoing()
            if other.allows("passenger") and other.getFunction() != "internal"]


def add_vehicle(root, vid, edge, lane, pos, speed, role, next_edge=None):
    vehicle = ET.SubElement(root, "vehicle", {
        "id": str(vid), "type": role, "depart": "0.00", "departLane": str(lane),
        "departPos": f"{pos:.1f}", "departSpeed": f"{speed:.1f}",
    })
    route = edge.getID() + (" " + next_edge.getID() if next_edge is not None else "")
    ET.SubElement(vehicle, "route", {"edges": route})
    return {"id": str(vid), "role": role, "start_edge": edge.getID(),
            "lane": lane, "position_m": round(pos, 1), "route": route}


def build_scene(net, output, morphology, number, edge_id, profile):
    edge = net.getEdge(edge_id)
    node = edge.getToNode()
    scene_id = f"{morphology}_{number:02d}"
    scene_dir = output / scene_id
    scene_dir.mkdir(parents=True, exist_ok=False)
    root = ET.Element("routes")
    for type_id, attrs in TYPES.items():
        ET.SubElement(root, "vType", dict(id=type_id, vClass="passenger", length="4.5",
                                          minGap="1.5", tau="1.0", accel="2.0", decel="4.5",
                                          sigma="0.0", guiShape="passenger", **attrs))
    successors = connected_next(edge)
    if morphology in ("intersection", "merge", "curve"):
        ego_pos = edge.getLength() - 35
    else:
        ego_pos = min(edge.getLength() - 48, max(25, edge.getLength() * .55))
    roster = [add_vehicle(root, 0, edge, 0, ego_pos, 8.0, "ego",
                          successors[0] if successors else None)]
    occupied = {(edge_id, 0): [ego_pos]}
    next_id = 1
    # Fill the ego road, approaches to the same node, and departing roads.
    incoming = [edge] + [other for other in node.getIncoming()
                         if other != edge and other.allows("passenger") and other.getLength() >= 45]
    outgoing = [other for other in node.getOutgoing()
                if other.allows("passenger") and other.getLength() >= 45]
    if profile == "balanced":
        # Distinct roles rather than a grid of cars on every lane. Each local
        # approach contributes only 2–3 vehicles, preserving open space.
        side_limit = 2 if morphology in ("intersection", "merge") else 1
        out_limit = 2 if morphology == "intersection" else 1
        selected = [(edge, "main")]
        selected += [(road, "approach") for road in incoming[1:1 + side_limit]]
        selected += [(road, "exit") for road in outgoing[:out_limit]]
        for road, side in selected:
            length = road.getLength()
            if side == "main":
                slots = [(0, ego_pos + 35), (0, ego_pos - 32)]
                if len(road.getLanes()) > 1:
                    slots += [(1, ego_pos + 47), (1, ego_pos + 8), (1, ego_pos - 30)]
                    if morphology == "straight":
                        slots.append((1, ego_pos - 48))
                    elif morphology == "curve":
                        slots += [(0, ego_pos - 55), (1, ego_pos - 55)]
            elif side == "approach":
                slots = [(0, length - 28), (0, length - 56)]
                if len(road.getLanes()) > 1:
                    slots.append((1, length - 42))
            else:
                slots = [(0, 45), (0, 15)]
                if len(road.getLanes()) > 1:
                    slots.append((1, 30))
            # Put leading vehicles first to avoid artificial depart queues.
            for lane, pos in sorted(slots, key=lambda item: -item[1]):
                if not 7 <= pos <= length - 9:
                    continue
                if road == edge and lane == 0 and -18 < pos - ego_pos < 22:
                    continue
                if any(abs(pos - prior) < 11.5 for prior in occupied.get((road.getID(), lane), [])):
                    continue
                roster.append(add_vehicle(root, next_id, road, lane, pos, 8.0,
                                          "traffic", (connected_next(road) or [None])[0]))
                occupied.setdefault((road.getID(), lane), []).append(pos)
                next_id += 1
    else:
        roads = [(other, "incoming") for other in incoming]
        roads += [(other, "outgoing") for other in outgoing]
        for road, side in roads:
            lane_count = min(len(road.getLanes()), 3)
            length = road.getLength()
            if side == "incoming":
                low, high = max(7, length - 90), length - 18
            else:
                low, high = 8, min(length - 12, 77)
            if high < low:
                continue
            for lane in range(lane_count):
                positions = []
                pos = low + lane * 3.0
                while pos <= high + 0.01:
                    positions.append(pos)
                    pos += 13.5
                for pos in reversed(positions):
                    if road == edge and lane == 0 and -18 < pos - ego_pos < 22:
                        continue
                    if all(abs(pos - prior) >= 11.5 for prior in occupied.get((road.getID(), lane), [])):
                        roster.append(add_vehicle(root, next_id, road, lane, pos, 8.0,
                                                  "traffic", (connected_next(road) or [None])[0]))
                        occupied.setdefault((road.getID(), lane), []).append(pos)
                        next_id += 1
    route_file = scene_dir / f"{scene_id}.rou.xml"
    ET.ElementTree(root).write(route_file, encoding="utf-8", xml_declaration=True)
    cfg = ET.Element("configuration")
    inputs = ET.SubElement(cfg, "input")
    ET.SubElement(inputs, "net-file", {"value": "../../osm.net.xml"})
    ET.SubElement(inputs, "route-files", {"value": route_file.name})
    times = ET.SubElement(cfg, "time")
    ET.SubElement(times, "begin", {"value": "0"})
    ET.SubElement(times, "end", {"value": "10"})
    processing = ET.SubElement(cfg, "processing")
    ET.SubElement(processing, "collision.action", {"value": "warn"})
    ET.SubElement(processing, "collision.check-junctions", {"value": "true"})
    cfg_file = scene_dir / f"{scene_id}.sumocfg"
    ET.ElementTree(cfg).write(cfg_file, encoding="utf-8", xml_declaration=True)
    metadata = {"scene_id": scene_id, "morphology": morphology, "ego_edge": edge_id,
                "junction_id": node.getID(), "road_length_m": round(edge.getLength(), 1),
                "road_lanes": len(edge.getLanes()), "planned_vehicles": len(roster),
                "vehicles": roster, "density_profile": profile,
                "purpose": "SUMO traffic preview only; no attack"}
    (scene_dir / f"{scene_id}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata, cfg_file


def fcd_snapshot(path, at_time=2.0):
    for _, timestep in ET.iterparse(path, events=("end",)):
        if timestep.tag != "timestep":
            continue
        if abs(float(timestep.get("time")) - at_time) < .001:
            cars = [{key: v.get(key) for key in ("id", "x", "y", "speed", "lane")}
                    for v in timestep.findall("vehicle")]
            ego = next((v for v in cars if v["id"] == "0"), None)
            if ego is None:
                return {"ego_present": False, "active_vehicles": len(cars)}
            ex, ey = float(ego["x"]), float(ego["y"])
            for v in cars:
                v["distance_to_ego_m"] = round(math.hypot(float(v["x"]) - ex,
                                                           float(v["y"]) - ey), 1)
            return {"ego_present": True, "active_vehicles": len(cars),
                    "nearby_60m": sum(v["id"] != "0" and v["distance_to_ego_m"] <= 60 for v in cars),
                    "nearby_80m": sum(v["id"] != "0" and v["distance_to_ego_m"] <= 80 for v in cars),
                    "vehicles": cars}
        timestep.clear()
    return {"ego_present": False, "active_vehicles": 0}


def smoke(metadata, cfg):
    scene_dir = cfg.parent
    scene_id = metadata["scene_id"]
    fcd = scene_dir / f"{scene_id}.fcd.xml"
    collision_file = scene_dir / f"{scene_id}.collisions.xml"
    command = ["sumo", "-c", str(cfg), "--step-length", "0.1",
               "--no-step-log", "true", "--duration-log.statistics", "true",
               "--device.fcd.period", "0.5", "--fcd-output", str(fcd),
               "--collision-output", str(collision_file)]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=60)
    log = process.stdout + process.stderr
    (scene_dir / f"{scene_id}.log").write_text(log, encoding="utf-8")
    match = re.search(r"Inserted:\s*(\d+)", log)
    inserted = int(match.group(1)) if match else None
    collisions = (len(ET.parse(collision_file).getroot().findall("collision"))
                  if collision_file.exists() else None)
    snapshot = fcd_snapshot(fcd) if process.returncode == 0 and fcd.exists() else {}
    passed = (process.returncode == 0 and inserted == metadata["planned_vehicles"]
              and collisions == 0 and snapshot.get("ego_present")
              and snapshot.get("nearby_80m", 0) >= (5 if metadata["density_profile"] == "balanced" else 10))
    return {**metadata, "sumo_returncode": process.returncode, "inserted": inserted,
            "collisions": collisions, "snapshot_t2": snapshot, "pass": passed}


def draw(net, rows, output):
    selected = []
    for morphology in CANDIDATES:
        options = [r for r in rows if r["morphology"] == morphology and r["pass"]]
        if options:
            # The first curved candidate has a visibly stronger bend.
            selected.append(options[0] if morphology == "curve" else
                            max(options, key=lambda r: r["snapshot_t2"]["nearby_80m"]))
    if len(selected) != 4:
        return None
    fig, axes = plt.subplots(2, 2, figsize=(13, 12), dpi=150)
    fig.patch.set_facecolor("#f8fafc")
    names = {"straight": "Straight arterial", "curve": "Curved road",
             "intersection": "Multi-arm intersection", "merge": "Converging approaches"}
    for ax, row in zip(axes.flat, selected):
        cars = row["snapshot_t2"]["vehicles"]
        ego = next(v for v in cars if v["id"] == "0")
        ex, ey = float(ego["x"]), float(ego["y"])
        for road in net.getEdges():
            for lane in road.getLanes():
                shape = lane.getShape()
                if (max(p[0] for p in shape) < ex - 105 or min(p[0] for p in shape) > ex + 105
                        or max(p[1] for p in shape) < ey - 105 or min(p[1] for p in shape) > ey + 105):
                    continue
                ax.plot([p[0] for p in shape], [p[1] for p in shape],
                        color="#ccd5df", linewidth=1.6, zorder=1)
        for v in cars:
            near = v["distance_to_ego_m"] <= 80
            ax.scatter(float(v["x"]), float(v["y"]),
                       c="#e63946" if v["id"] == "0" else ("#2563eb" if near else "#94a3b8"),
                       marker="*" if v["id"] == "0" else "o",
                       s=210 if v["id"] == "0" else 48, edgecolors="white", linewidths=.6, zorder=3)
        ax.set_xlim(ex - 95, ex + 95)
        ax.set_ylim(ey - 95, ey + 95)
        ax.set_aspect("equal")
        ax.grid(color="#edf2f7", linewidth=.5)
        ax.tick_params(labelsize=7)
        snap = row["snapshot_t2"]
        ax.set_title(f"{names[row['morphology']]}  |  {row['scene_id']}\n"
                     f"{snap['nearby_80m']} cars within 80 m, {snap['active_vehicles']} active at 2 s",
                     loc="left", fontsize=11, fontweight="bold")
    fig.suptitle("SUMO traffic at t = 2 s  |  red: ego  blue: within 80 m  gray: farther",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .96))
    path = output / ("road_shapes_t2.png" if rows[0]["density_profile"] == "balanced"
                     else "high_density_road_shapes_t2.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("high", "balanced"), default="high")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a fresh directory")
    args.output.mkdir(parents=True)
    net = sumolib.net.readNet(str(NET))
    rows = []
    for morphology, edges in CANDIDATES.items():
        for number, edge_id in enumerate(edges, start=1):
            metadata, cfg = build_scene(net, args.output, morphology, number, edge_id, args.profile)
            result = smoke(metadata, cfg)
            rows.append(result)
            snap = result["snapshot_t2"]
            print(f"{result['scene_id']}: {result['inserted']}/{result['planned_vehicles']} inserted, "
                  f"{snap.get('nearby_80m')} within 80 m, {result['collisions']} collisions, "
                  f"{'PASS' if result['pass'] else 'FAIL'}", flush=True)
    (args.output / "smoke_report.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = [{key: row[key] for key in ("scene_id", "morphology", "ego_edge", "planned_vehicles")}
              for row in rows if row["pass"]]
    (args.output / "passed_index.json").write_text(
        json.dumps({"purpose": "SUMO traffic preview only; no attack or closed-loop validation",
                    "scenes": passed}, ensure_ascii=False, indent=2), encoding="utf-8")
    preview = draw(net, rows, args.output)
    print("Passed:", len(passed), "/", len(rows), "Preview:", preview)


if __name__ == "__main__":
    main()
