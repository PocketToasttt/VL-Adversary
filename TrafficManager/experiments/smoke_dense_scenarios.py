"""Compare generated and source scenes in plain SUMO, without model services."""
import argparse
import json
import math
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "TrafficManager/networkFiles/boston-seaport"
SOURCE = MAP / "scenarios_diverse_3types"


def sample_fcd(path, ego_id, attacker_id, added_ids):
    selected = {}
    peak_nearby = 0
    ego_seen = False
    attacker_at_two = False
    for _, element in ET.iterparse(path, events=("end",)):
        if element.tag != "timestep":
            continue
        time = round(float(element.get("time")), 2)
        cars = {v.get("id"): v for v in element.findall("vehicle")}
        ego = cars.get(str(ego_id))
        if ego is not None:
            ego_seen = True
            ex, ey = float(ego.get("x")), float(ego.get("y"))
            nearby = [v for vid, v in cars.items() if vid != str(ego_id)
                      and math.hypot(float(v.get("x")) - ex, float(v.get("y")) - ey) <= 60]
            peak_nearby = max(peak_nearby, len(nearby))
            if time in (0.5, 2.0, 5.0, 9.5):
                selected[str(time)] = {
                    "active_vehicles": len(cars), "nearby_60m": len(nearby),
                    "added_nearby_60m": sum(v.get("id") in added_ids for v in nearby),
                    "ego": {key: ego.get(key) for key in ("x", "y", "speed", "lane")},
                    "vehicles": [{key: v.get(key) for key in ("id", "x", "y", "speed", "lane")}
                                 for v in cars.values()],
                }
        if time == 2.0:
            attacker_at_two = str(attacker_id) in cars
        element.clear()
    return {"snapshots": selected, "peak_nearby_60m": peak_nearby,
            "ego_seen": ego_seen, "attacker_at_2s": attacker_at_two}


def run_one(label, scene_id, cfg, output, ego_id, attacker_id, added_ids):
    target = output / label
    target.mkdir(exist_ok=True)
    fcd = target / (scene_id + ".fcd.xml")
    collisions = target / (scene_id + ".collisions.xml")
    command = ["sumo", "-c", str(cfg), "--step-length", "0.1", "--no-step-log", "true",
               "--duration-log.statistics", "true", "--device.fcd.period", "0.5",
               "--fcd-output", str(fcd), "--collision-output", str(collisions)]
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60)
    log = process.stdout + process.stderr
    (target / (scene_id + ".log")).write_text(log, encoding="utf-8")
    count = re.search(r"Inserted:\s*(\d+)(?:\s*\(Loaded:\s*(\d+)\))?", log)
    waiting = re.search(r"Waiting:\s*(\d+)", log)
    collision_rows = []
    if collisions.is_file():
        collision_rows = [dict(element.attrib) for element in ET.parse(collisions).getroot().findall("collision")]
    expected = len(ET.parse(cfg.with_suffix(".rou.xml")).getroot().findall("vehicle"))
    data = {"returncode": process.returncode, "inserted": int(count.group(1)) if count else None,
            "loaded": int(count.group(2)) if count and count.group(2) else expected,
            "waiting": int(waiting.group(1)) if waiting else None,
            "collision_count": len(collision_rows), "collisions": collision_rows,
            "fcd_path": str(fcd), "log_path": str(target / (scene_id + ".log"))}
    if process.returncode == 0 and fcd.is_file():
        data.update(sample_fcd(fcd, ego_id, attacker_id, {str(x) for x in added_ids}))
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=MAP / "scenarios_dense_closedloop_pilot_v2/scene_index.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    args.output.mkdir(parents=True)
    index = json.loads(args.index.read_text(encoding="utf-8"))
    rows = []
    for scene in index["scenes"]:
        dense_id, source_id = scene["scene_id"], scene["source_scene_id"]
        new_cfg = args.index.parent / dense_id / (dense_id + ".sumocfg")
        old_cfg = SOURCE / source_id / (source_id + ".sumocfg")
        old = run_one("original", source_id, old_cfg, args.output,
                      scene["ego_id"], scene["attacker_candidate_id"], [])
        new = run_one("dense", dense_id, new_cfg, args.output,
                      scene["ego_id"], scene["attacker_candidate_id"], scene["added_local_vehicle_ids"])
        row = {"scene_id": dense_id, "source_scene_id": source_id,
               "attack_type": scene["attack_type"], "original": old, "dense": new}
        rows.append(row)
        print(f"{dense_id}: loaded {old['loaded']}->{new['loaded']}, "
              f"inserted {old['inserted']}->{new['inserted']}, "
              f"nearby@2s {old.get('snapshots', {}).get('2.0', {}).get('nearby_60m')}"
              f"->{new.get('snapshots', {}).get('2.0', {}).get('nearby_60m')}, "
              f"collisions {old['collision_count']}->{new['collision_count']}", flush=True)
    (args.output / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Summary:", args.output / "summary.json")


if __name__ == "__main__":
    main()
