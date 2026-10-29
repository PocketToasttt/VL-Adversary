"""Build a separate, statically screened closed-loop traffic pilot from the 105 source scenes.

This is a candidate generator, not a SUMO/UniAD validation. It keeps the three
attack geometries while widening the ego's same-lane handover gap and placing
additional traffic on the ego's actual road instead of random distant edges.
"""
import argparse
import copy
import json
import random
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAP = ROOT / "networkFiles" / "boston-seaport"
SOURCE = MAP / "scenarios_diverse_3types"
KINDS = ("brake_check", "cut_in", "opposite_direction_intrusion")


def edge_lanes(net_path):
    lengths = {}
    for _, element in ET.iterparse(net_path, events=("end",)):
        if element.tag == "edge" and not element.get("id", "").startswith(":"):
            lanes = element.findall("lane")
            if lanes:
                lengths[element.get("id")] = [float(lane.get("length")) for lane in lanes]
            element.clear()
    return lengths


def vehicles(route_tree):
    result = []
    for element in route_tree.getroot().findall("vehicle"):
        route = element.find("route")
        if route is None:
            continue
        result.append((element, route.get("edges", "").split()))
    return result


def initial_clearance(entries, ego):
    ego_edge = ego[1][0]
    lane = ego[0].get("departLane")
    pos = float(ego[0].get("departPos"))
    ahead, behind = [], []
    for vehicle, route in entries:
        if vehicle is ego[0] or not route or route[0] != ego_edge or vehicle.get("departLane") != lane:
            continue
        gap = float(vehicle.get("departPos")) - pos
        (ahead if gap > 0 else behind).append(abs(gap))
    return min(ahead, default=None), min(behind, default=None)


def widen_handover(entries, ego, lengths, front_gap=18.0, rear_gap=14.0):
    """Translate each same-lane group without changing its internal spacing."""
    edge = ego[1][0]
    lane = int(ego[0].get("departLane"))
    ego_pos = float(ego[0].get("departPos"))
    same = [vehicle for vehicle, route in entries if route and route[0] == edge
            and int(vehicle.get("departLane")) == lane and vehicle is not ego[0]]
    front = [v for v in same if float(v.get("departPos")) > ego_pos]
    rear = [v for v in same if float(v.get("departPos")) <= ego_pos]
    front_shift = max(0.0, front_gap - min((float(v.get("departPos")) - ego_pos for v in front), default=front_gap))
    rear_shift = max(0.0, rear_gap - min((ego_pos - float(v.get("departPos")) for v in rear), default=rear_gap))
    edge_limit = lengths[edge][lane] - 8.0
    if any(float(v.get("departPos")) + front_shift > edge_limit for v in front):
        return False
    if any(float(v.get("departPos")) - rear_shift < 5.0 for v in rear):
        return False
    for vehicle in front:
        vehicle.set("departPos", f"{float(vehicle.get('departPos')) + front_shift:.1f}")
    for vehicle in rear:
        vehicle.set("departPos", f"{float(vehicle.get('departPos')) - rear_shift:.1f}")
    return True


def add_local_traffic(tree, entries, ego, lengths, target, rng, scene_id):
    edge = ego[1][0]
    lanes = lengths[edge]
    ego_lane = int(ego[0].get("departLane"))
    ego_pos = float(ego[0].get("departPos"))
    ego_speed = float(ego[0].get("departSpeed"))
    route_text = " ".join(ego[1])
    occupied = {}
    for vehicle, route in entries:
        if route and route[0] == edge and float(vehicle.get("depart")) == 0:
            occupied.setdefault(int(vehicle.get("departLane")), []).append(float(vehicle.get("departPos")))
    existing_ids = {int(v.get("id")) for v, _ in entries}
    next_id = max(existing_ids) + 1
    new_ids = []
    # Mix cars behind and ahead. Adjacent lanes get priority so handover stays clear.
    lane_order = sorted(range(len(lanes)), key=lambda value: (value == ego_lane, abs(value - ego_lane)))
    offsets = [-36, 36, -27, 27, -48, 48, -19, 19, -60, 60]
    for lane in lane_order:
        for offset in offsets:
            if len(new_ids) >= target:
                break
            pos = ego_pos + offset + rng.uniform(-1.5, 1.5)
            if pos < 6 or pos > lanes[lane] - 9:
                continue
            if lane == ego_lane and abs(pos - ego_pos) < 20:
                continue
            gap = 11.0 if lane == ego_lane else 9.0
            if any(abs(pos - other) < gap for other in occupied.get(lane, [])):
                continue
            speed = max(2.0, min(12.5, ego_speed + rng.uniform(-1.2, 1.2)))
            if lane == ego_lane and pos < ego_pos:
                speed = min(speed, ego_speed)
            if lane == ego_lane and pos > ego_pos:
                speed = max(speed, ego_speed)
            vehicle = ET.SubElement(tree.getroot(), "vehicle", {
                "id": str(next_id), "type": "default_type", "depart": "0.00",
                "departLane": str(lane), "departPos": f"{pos:.1f}",
                "departSpeed": f"{speed:.1f}",
            })
            ET.SubElement(vehicle, "route", {"edges": route_text})
            for key, value in (("scene_id", scene_id), ("scenario_type", "local_context"),
                               ("role", "npc"), ("scene_ego_id", ego[0].get("id"))):
                ET.SubElement(vehicle, "param", {"key": key, "value": value})
            new_ids.append(next_id)
            next_id += 1
            occupied.setdefault(lane, []).append(pos)
        if len(new_ids) >= target:
            break
    return new_ids


def initial_layout_ok(route_tree, lengths, minimum_gap=6.0):
    occupied = {}
    for vehicle, route in vehicles(route_tree):
        if not route or route[0] not in lengths:
            return False
        lane = int(vehicle.get("departLane"))
        pos = float(vehicle.get("departPos"))
        if lane >= len(lengths[route[0]]) or pos < 0 or pos > lengths[route[0]][lane]:
            return False
        if float(vehicle.get("depart")) == 0:
            occupied.setdefault((route[0], lane), []).append(pos)
    for positions in occupied.values():
        positions.sort()
        if any(right - left < minimum_gap for left, right in zip(positions, positions[1:])):
            return False
    return True


def select_sources(index, kind, count, rng):
    groups = {}
    for scene in index["scenes"]:
        if scene["attack_type"] == kind:
            template = re.search(r"\[([^]]+)\]", scene.get("description", ""))
            groups.setdefault(template.group(1) if template else "other", []).append(scene)
    for group in groups.values():
        rng.shuffle(group)
    ordered = []
    while any(groups.values()):
        for key in sorted(groups):
            if groups[key]:
                ordered.append(groups[key].pop())
    return ordered[:count * 3]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MAP / "scenarios_dense_closedloop_pilot_v2")
    parser.add_argument("--per-type", type=int, default=10)
    parser.add_argument("--extra-local", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new directory. Existing scenes are never overwritten.")
    if args.per_type < 1 or args.extra_local < 1:
        parser.error("--per-type and --extra-local must be positive")
    rng = random.Random(args.seed)
    index = json.loads((SOURCE / "scene_index.json").read_text(encoding="utf-8"))
    lengths = edge_lanes(MAP / "osm.net.xml")
    generated = []
    rejected = []
    for kind in KINDS:
        selected = 0
        for source in select_sources(index, kind, args.per_type, rng):
            if selected >= args.per_type:
                break
            old_id = source["scene_id"]
            route_path = SOURCE / old_id / (old_id + ".rou.xml")
            route_tree = ET.parse(route_path)
            entries = vehicles(route_tree)
            ego = next(((v, route) for v, route in entries if v.get("type") == "ego_type"), None)
            if not ego or not ego[1] or ego[1][0] not in lengths:
                rejected.append({"source": old_id, "reason": "missing_ego_or_edge"})
                continue
            before = initial_clearance(entries, ego)
            if not widen_handover(entries, ego, lengths):
                rejected.append({"source": old_id, "reason": "insufficient_edge_length_for_handover_gap"})
                continue
            after = initial_clearance(entries, ego)
            if (after[0] is not None and after[0] < 17.9) or (after[1] is not None and after[1] < 13.9):
                rejected.append({"source": old_id, "reason": "handover_gap_check_failed"})
                continue
            new_id = f"{kind}_dense_{selected + 1:03d}"
            new_ids = add_local_traffic(route_tree, entries, ego, lengths, args.extra_local, rng, new_id)
            if len(new_ids) < 2:
                rejected.append({"source": old_id, "reason": "too_few_local_traffic_slots"})
                continue
            if not initial_layout_ok(route_tree, lengths):
                rejected.append({"source": old_id, "reason": "initial_spacing_or_lane_check_failed"})
                continue
            source_meta = json.loads((SOURCE / old_id / (old_id + ".json")).read_text(encoding="utf-8"))
            scene_dir = args.output / new_id
            scene_dir.mkdir(parents=True)
            for vehicle, _ in vehicles(route_tree):
                for param in vehicle.findall("param"):
                    if param.get("key") == "scene_id":
                        param.set("value", new_id)
            route_out = scene_dir / (new_id + ".rou.xml")
            ET.indent(route_tree, space="    ")
            route_tree.write(route_out, encoding="utf-8", xml_declaration=True)
            cfg_tree = ET.parse(SOURCE / old_id / (old_id + ".sumocfg"))
            cfg_tree.getroot().find("input/route-files").set("value", route_out.name)
            cfg_out = scene_dir / (new_id + ".sumocfg")
            ET.indent(cfg_tree, space="    ")
            cfg_tree.write(cfg_out, encoding="utf-8", xml_declaration=True)
            metadata = copy.deepcopy(source_meta)
            metadata["source_timestamp"] = metadata.pop("timestamp", None)
            metadata.update(scene_id=new_id, source_scene_id=old_id,
                            background_vehicle_ids=metadata.get("background_vehicle_ids", []) + new_ids,
                            vehicle_count=len(vehicles(route_tree)),
                            handover_clearance_m={"front_before": before[0], "rear_before": before[1],
                                                   "front_after": after[0], "rear_after": after[1]},
                            added_local_vehicle_ids=new_ids,
                            validation_status="static_only_requires_sumo_and_closedloop_pilot",
                            generated_at=datetime.now(timezone.utc).isoformat(),
                            generated_files={"route_file": route_out.relative_to(ROOT.parent).as_posix(),
                                             "config_file": cfg_out.relative_to(ROOT.parent).as_posix()})
            (scene_dir / (new_id + ".json")).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            generated.append(dict(scene_id=new_id, source_scene_id=old_id, scenario_type=source["scenario_type"],
                                  attack_type=kind, ego_id=source["ego_id"],
                                  attacker_candidate_id=source["attacker_candidate_id"],
                                  description=source["description"], vehicle_count=metadata["vehicle_count"],
                                  added_local_vehicle_ids=new_ids, handover_clearance_m=metadata["handover_clearance_m"],
                                  validation_status=metadata["validation_status"]))
            selected += 1
        if selected < args.per_type:
            raise RuntimeError(f"Only generated {selected}/{args.per_type} {kind} scenes; see output for partial set")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"metadata": {"source_index": str(SOURCE / "scene_index.json"), "seed": args.seed,
                              "extra_local_target": args.extra_local, "front_handover_gap_m": 18,
                              "rear_handover_gap_m": 14, "validation_status": "static_only",
                              "note": "Run SUMO no-attack preflight and UniAD closed-loop baseline before attack testing."},
                "statistics": {"total_scenes": len(generated), "scene_ids": [s["scene_id"] for s in generated]},
                "scenes": generated, "rejected_sources": rejected}
    (args.output / "scene_index.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {len(generated)} candidate scenes in {args.output}; rejected {len(rejected)} source layouts")


if __name__ == "__main__":
    main()
