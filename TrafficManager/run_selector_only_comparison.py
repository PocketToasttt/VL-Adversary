"""Replay frozen pre-attack inputs: 15 scenes x 3 selection arms, no simulation.

Prepare: python TrafficManager/run_selector_only_comparison.py --output results/selector_only --prepare-only
Run:     python TrafficManager/run_selector_only_comparison.py --output results/selector_only --detach
Existing response files are never requested again. No trajectory generation is imported.
"""
import argparse
import base64
import csv
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import types
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from TrafficManager.experiments.sumo_selector_protocol import select_scene_ids, shared_rules

# Import only the client and selection definitions, without SUMO package initialization.
PACKAGE = "_selector_only_framework"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT / "TrafficManager/attack_framework")]
sys.modules[PACKAGE] = package
config_module = importlib.import_module(PACKAGE + ".config")
client_module = importlib.import_module(PACKAGE + ".llm_client")
selector_module = importlib.import_module(PACKAGE + ".vlm_selector")

ATTACKS = ("brake_check", "cut_in", "opposite_direction_intrusion")
ARMS = {
    "vlm_image": {"model": "qwen3-vl-plus", "image": True},
    "vlm_text": {"model": "qwen3-vl-plus", "image": False},
    "qwen_plus": {"model": "qwen-plus", "image": False},
}
RULES = shared_rules(selector_module.TEXT_SELECTION_RULES)
TEMPERATURE = 0.5


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def clean_description(description):
    # Preserve measured states/topology. Remove historical per-vehicle answer hints.
    description = re.sub(r",\s*allowed_attacks=\[[^\]\n]*\]", "", description)
    description = re.sub(r",\s*可用攻击:[^\n]*", "", description)
    for token in (*ATTACKS, "allowed_attacks", "suitable_attacks", "attacker_candidate_id"):
        if token in description:
            raise ValueError("Unexpected answer hint in scene description: " + token)
    return description


def parse_candidates(description):
    relations = dict(re.findall(r"vehicle=([^,\s]+),[^\n]*?relation=(\w+)", description))
    pattern = (r"- ID: ([^,]+), \[([^\]]+)\], 车道: ([^,]+), 位置: ([^,]+), "
               r"距离: ([\d.]+)m, 速度: ([\d.]+) m/s, 航向差: ([\d.]+)°")
    candidates = []
    for vid, direction, lane, position, distance, speed, heading in re.findall(pattern, description):
        candidates.append(dict(id=vid, direction=direction, lane_id=lane,
                               lane_relation=relations.get(vid, "unknown"),
                               relative_position=position, distance=float(distance),
                               speed=float(speed), heading_diff=float(heading)))
    if not candidates or len(candidates) != len(re.findall(r"^- ID:", description, re.M)):
        raise ValueError("Incomplete candidate parsing")
    return candidates


def matched_image(source, description, candidates):
    """Match recorded frame states; never guess the image number from attack time."""
    ego_speed = float(re.search(r"速度 ([\d.]+) m/s", description).group(1))
    ego_lane = re.search(r"主车当前车道ID: ([^\n]+)", description).group(1)
    matches = []
    for info_path in sorted((source / "infos").glob("*.json")):
        frame = read_json(info_path)[-1]
        ego = frame["ego"]
        vehicles = {str(v["VehID"]): v for v in frame["veh"]}
        if f"{ego['Speed']:.1f}" != f"{ego_speed:.1f}" or ego["LaneID"] != ego_lane:
            continue
        if set(vehicles) != {c["id"] for c in candidates}:
            continue
        if not all(vehicles[c["id"]]["LaneID"] == c["lane_id"] and
                   f"{vehicles[c['id']]['Speed']:.1f}" == f"{c['speed']:.1f}" and
                   f"{math.hypot(*vehicles[c['id']]['RelPos'][:2]):.1f}" == f"{c['distance']:.1f}"
                   for c in candidates):
            continue
        image = source / "imgs" / ("bev_" + info_path.stem + ".png")
        if image.is_file():
            matches.append((image, info_path))
    if len(matches) != 1:
        raise ValueError(f"Expected a unique state-matched BEV, found {len(matches)} in {source}")
    return matches[0]


def prepare(output, source_batch, per_type):
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["arms"] != ARMS or manifest["rules_sha256"] != digest(RULES.encode()):
            raise ValueError("Existing run uses a different protocol; use a new output folder")
        if manifest["per_type"] != per_type or manifest["source_batch"] != str(source_batch):
            raise ValueError("Existing run uses different inputs")
        return manifest
    index_path = ROOT / "TrafficManager/networkFiles/boston-seaport/scenarios_diverse_3types/scene_index.json"
    scene_ids = select_scene_ids(read_json(index_path), per_type)
    inputs = []
    for number, scene_id in enumerate(scene_ids, 1):
        # Choose the earliest recorded run deterministically, never based on its outcome.
        records = sorted(source_batch.glob(scene_id + "_*/attack_record.json"))
        if not records:
            raise FileNotFoundError("Missing recorded input for " + scene_id)
        record_path = records[0]
        record = read_json(record_path)
        if record["scene_name"] != scene_id or not (0 < record["trigger_time"] < 60):
            raise ValueError("Unexpected snapshot provenance")
        description = clean_description(record["structured_scene"])
        candidates = parse_candidates(description)
        image, info = matched_image(record_path.parent, description, candidates)
        sample = f"S{number:03d}"
        directory = output / "inputs" / sample
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, directory / "bev.png")
        snapshot = dict(sample=sample, scene_id=scene_id, category=scene_id.rsplit("_", 1)[0],
                        description=description, candidates=candidates,
                        topology=record["sumo_topology"], trigger_time=record["trigger_time"],
                        source_record=str(record_path), source_record_sha256=digest(record_path.read_bytes()),
                        source_image=str(image), source_frame_info=str(info),
                        text_sha256=digest(description.encode()), image_sha256=digest(image.read_bytes()))
        save_json(directory / "snapshot.json", snapshot)
        inputs.append(sample)
    output.mkdir(parents=True, exist_ok=True)
    (output / "prompt.txt").write_text(RULES, encoding="utf-8")
    shutil.copyfile(Path(__file__), output / "runner_snapshot.py")
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), arms=ARMS,
                    per_type=per_type, samples=inputs, expected_calls=len(inputs) * len(ARMS),
                    source_batch=str(source_batch), rules_sha256=digest(RULES.encode()),
                    temperature=TEMPERATURE, protocol="frozen-selection-only-v1",
                    labels_sent=False, per_vehicle_attack_hints_sent=False,
                    trajectory_generation=False, sumo_execution=False,
                    source_mode="previous closed-loop pre-attack snapshots",
                    image_alignment="unique matching ego/candidate speeds, lane IDs and relative distances in saved frame info",
                    topology_check="necessary lane/direction checks only; not full reachability ground truth")
    save_json(manifest_path, manifest)
    return manifest


def messages_for(snapshot, image_path, with_image):
    # No sample ID, filename, category, expected attacker or previous reasoning enters messages.
    text = RULES + "\n\n实时结构化场景描述：\n" + snapshot["description"]
    content = text
    if with_image:
        content = [{"type": "text", "text": text},
                   {"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                    base64.b64encode(Path(image_path).read_bytes()).decode("ascii")}}]
    return [{"role": "system", "content": RULES}, {"role": "user", "content": content}]


def validate(raw, candidates):
    result = dict(json_valid=False, schema_valid=False, candidate_valid=False,
                  topology_compatible=None, output_legal=False, vehicle_id=None,
                  attack_type=None, reasoning=None, errors=[])
    try:
        value = json.loads(raw)
        result["json_valid"] = True
    except (ValueError, TypeError):
        result["errors"].append("not_strict_json")
        return result
    if not isinstance(value, dict):
        result["errors"].append("not_json_object")
        return result
    vid, attack, reasoning = (value.get(k) for k in ("target_vehicle_id", "attack_type", "reasoning"))
    result.update(vehicle_id=vid, attack_type=attack, reasoning=reasoning)
    result["schema_valid"] = (isinstance(vid, str) and bool(vid.strip()) and
                              isinstance(attack, str) and attack in ATTACKS and
                              isinstance(reasoning, str) and bool(reasoning.strip()) and
                              isinstance(value.get("behavior_params"), dict))
    candidate = next((c for c in candidates if c["id"] == vid), None)
    result["candidate_valid"] = candidate is not None
    if not result["schema_valid"]:
        result["errors"].append("invalid_schema_or_attack_type")
    if candidate is None:
        result["errors"].append("unknown_vehicle_id")
    else:
        result["selected_road_context"] = candidate
        if attack == "brake_check":
            compatible = candidate["heading_diff"] <= 30 and candidate["lane_relation"] == "same_lane" and candidate["relative_position"] == "Front"
        elif attack == "cut_in":
            compatible = candidate["heading_diff"] <= 30 and candidate["lane_relation"] == "adjacent_lane" and candidate["relative_position"] != "Behind"
        elif attack == "opposite_direction_intrusion":
            # Opposite heading alone cannot prove lane intrusion reachability.
            compatible = None if candidate["heading_diff"] >= 150 else False
        else:
            compatible = False
        result["topology_compatible"] = compatible
    result["output_legal"] = result["schema_valid"] and result["candidate_valid"]
    return result


def service_config():
    import yaml
    cfg = yaml.safe_load((ROOT / "TrafficManager/config-attack.yaml").read_text(encoding="utf-8"))["llm"]
    endpoint = cfg["api_base"].rstrip("/")
    if endpoint != "https://dashscope.aliyuncs.com/compatible-mode/v1":
        raise ValueError("Only the authorized DashScope endpoint is allowed")
    key = os.environ.get("DASHSCOPE_API_KEY") or cfg.get("api_key")
    if not key:
        saved = yaml.safe_load((ROOT / "TrafficManager/temp_ego_317_mqbq0vhx.yaml").read_text(encoding="utf-8"))["llm"]
        if saved.get("api_base", "").rstrip("/") != endpoint:
            raise ValueError("Saved credential endpoint mismatch")
        key = saved.get("api_key")
    if not key:
        raise ValueError("Missing configured DashScope credential")
    return config_module.LLMServiceConfig(True, key, endpoint, "qwen3-vl-plus")


def export_report(output, manifest):
    rows = [read_json(p) for p in sorted((output / "responses").glob("*.json"))]
    columns = ["sample", "scene_id", "category", "arm", "model", "vehicle_id", "attack_type", "reasoning",
               "json_valid", "schema_valid", "candidate_valid", "output_legal", "topology_compatible",
               "elapsed_seconds", "api_error"]
    with (output / "selections.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    paired = []
    for sample in manifest["samples"]:
        group = {r["arm"]: r for r in rows if r["sample"] == sample}
        for left, right in itertools.combinations(ARMS, 2):
            if left not in group or right not in group:
                continue
            a, b = group[left], group[right]
            eligible = a["output_legal"] and b["output_legal"]
            paired.append(dict(sample=sample, scene_id=a["scene_id"], category=a["category"],
                               left=left, right=right, eligible=eligible,
                               same_vehicle=a["vehicle_id"] == b["vehicle_id"] if eligible else None,
                               same_attack=a["attack_type"] == b["attack_type"] if eligible else None,
                               left_context=a.get("selected_road_context"), right_context=b.get("selected_road_context")))
    save_json(output / "pairwise.json", paired)
    counts = {}
    for left, right in itertools.combinations(ARMS, 2):
        pairs = [p for p in paired if p["left"] == left and p["right"] == right and p["eligible"]]
        counts[left + " vs " + right] = dict(valid_pairs=len(pairs),
            different_vehicle=sum(not p["same_vehicle"] for p in pairs),
            different_attack=sum(not p["same_attack"] for p in pairs),
            different_either=sum(not(p["same_vehicle"] and p["same_attack"]) for p in pairs))
    summary = dict(completed=len(rows), expected=manifest["expected_calls"], pairwise=counts,
                   arms={arm: {"completed": sum(r["arm"] == arm for r in rows),
                               "legal": sum(r["arm"] == arm and r["output_legal"] for r in rows),
                               "api_errors": sum(r["arm"] == arm and bool(r.get("api_error")) for r in rows)} for arm in ARMS})
    save_json(output / "summary.json", summary)
    lines = ["# 仅选车实验", "", f"已完成 {len(rows)}/{manifest['expected_calls']} 次。", "",
             "三组共享同一攻击前快照与文本；仅有图组附带BEV。场景类别只用于统计。", "",
             "| 对比 | 双方输出合法的场景数 | 选车不同 | 类型不同 | 任一不同 |",
             "|---|---:|---:|---:|---:|"]
    for label, c in counts.items():
        lines.append(f"| {label} | {c['valid_pairs']} | {c['different_vehicle']} | {c['different_attack']} | {c['different_either']} |")
    lines += ["", "## 差异与道路关系（描述性核对）", ""]
    for p in paired:
        if p["eligible"] and not(p["same_vehicle"] and p["same_attack"]):
            contexts = []
            for side in ("left", "right"):
                c = p[side + "_context"]
                contexts.append(f"{p[side]} → ID {c['id']}，{c['lane_relation']}，{c['direction']}，{c['relative_position']}，{c['distance']}m")
            lines.append(f"- {p['scene_id']}: " + "；".join(contexts))
    lines += ["", "完整理由和合法性见 selections.csv；原始响应见 responses。",
              "合法性指严格JSON、字段类型、三种攻击枚举及车辆ID；道路必要条件另列，未知可达性不当作正确。",
              "差异与道路关系同时出现不等于因果证据；单次15场景筛查不证明视觉更优或信息衰减。",
              ("SUMO执行指标另存于同批次stage2_metrics，按场景ID和组别对应。" if manifest.get('combined_sumo_run') else
               "未运行SUMO，不生成碰撞率等指标。第二轮需另行决定。")]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run(output, manifest):
    responses = output / "responses"
    responses.mkdir(exist_ok=True)
    client = client_module.LLMClient(service_config())
    for number, sample in enumerate(manifest["samples"]):
        directory = output / "inputs" / sample
        snapshot = read_json(directory / "snapshot.json")
        if digest(snapshot["description"].encode()) != snapshot["text_sha256"] or digest((directory / "bev.png").read_bytes()) != snapshot["image_sha256"]:
            raise ValueError("Frozen input modified")
        # Rotate request order to avoid tying model arm to a fixed position in time.
        arms = list(ARMS)
        arms = arms[number % 3:] + arms[:number % 3]
        for arm in arms:
            path = responses / (sample + "_" + arm + ".json")
            if path.exists():
                continue
            spec = ARMS[arm]
            row = dict(sample=sample, scene_id=snapshot["scene_id"], category=snapshot["category"],
                       arm=arm, model=spec["model"], image_attached=spec["image"],
                       text_sha256=snapshot["text_sha256"], image_sha256=snapshot["image_sha256"] if spec["image"] else None,
                       raw_response=None, api_error=None)
            started = time.monotonic()
            print(f"SELECT {sample} {arm}", flush=True)
            try:
                # Single request: do not silently multiply the planned 45 API calls on retry.
                request = dict(url=client.config.api_base + "/chat/completions",
                               headers={"Authorization": "Bearer " + client.config.api_key, "Content-Type": "application/json"},
                               payload=dict(model=spec["model"], messages=messages_for(snapshot, directory / "bev.png", spec["image"]),
                                            temperature=TEMPERATURE), timeout=120)
                row["raw_response"] = client_module._bounded_completion_request(request, 120)
                row.update(validate(row["raw_response"], snapshot["candidates"]))
            except Exception as error:
                row.update(validate(None, snapshot["candidates"]))
                row["api_error"] = str(error).replace(client.config.api_key, "[REDACTED]")
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
            save_json(path, row)
            summary = export_report(output, manifest)
            save_json(output / "progress.json", dict(status="running", **summary))
            print(f"SAVED {sample} {arm} legal={row['output_legal']}", flush=True)
    summary = export_report(output, manifest)
    save_json(output / "progress.json", dict(status="complete", **summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-batch", default="results/batch_attack_closedloop")
    parser.add_argument("--per-type", type=int, default=5)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    source = (ROOT / args.source_batch).resolve()
    manifest = prepare(output, source, args.per_type)
    if args.prepare_only:
        print(json.dumps(dict(output=str(output), scenes=len(manifest["samples"]), calls=manifest["expected_calls"])))
        return
    # Fail before detaching if credentials are unavailable; never print their value.
    service_config()
    if args.detach:
        command = [sys.executable, str(Path(__file__).resolve()), "--output", str(output),
                   "--source-batch", str(source), "--per-type", str(args.per_type)]
        with (output / "run.log").open("a", encoding="utf-8") as log:
            kwargs = {"start_new_session": True} if os.name != "nt" else {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS}
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log,
                                       stdin=subprocess.DEVNULL, **kwargs)
        save_json(output / "launcher.json", dict(pid=process.pid, command=command))
        print(json.dumps(dict(started=True, pid=process.pid, output=str(output))))
        return
    lock = output / "running.lock"
    # Refuse a concurrent invocation; an interrupted run's lock needs explicit review.
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        run(output, manifest)
    except Exception as error:
        save_json(output / "progress.json", dict(status="failed", error_type=type(error).__name__))
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
