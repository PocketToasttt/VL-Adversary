import argparse
import glob
import json
import os
import re
from collections import Counter

from batch_test import (
    SCENE_INDEX_PATH,
    SCENE_PLAN,
    detect_completion_reason,
    get_all_scenes,
    select_scenes_by_plan,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)


def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            return file.read()
    except OSError:
        return ""


def find_last_exception(log_text: str):
    matches = re.findall(
        r"(?m)^([A-Za-z_][\w.]*(?:Error|Exception)):\s*(.+)$",
        log_text,
    )
    if not matches:
        return None
    exception_type, message = matches[-1]
    return f"{exception_type}: {message.strip()}"


def count_frames(log_text: str) -> int:
    return len(re.findall(r"(?m)^Frame \d+:", log_text))


def last_simulation_timestamp(log_text: str):
    timestamps = re.findall(r"Current timestamp:\s*([0-9.]+)", log_text)
    return float(timestamps[-1]) if timestamps else None


def find_scene_log(batch_dir: str, scene_id: str, ego_id: str):
    pattern = os.path.join(batch_dir, f"attack_*_{scene_id}_ego{ego_id}.txt")
    paths = [path for path in glob.glob(pattern) if not path.endswith("_TIMEOUT.txt")]
    return max(paths, key=os.path.getmtime) if paths else None


def has_timeout_record(batch_dir: str, scene_id: str, ego_id: str) -> bool:
    pattern = os.path.join(
        batch_dir,
        f"attack_*_{scene_id}_ego{ego_id}_TIMEOUT.txt",
    )
    return bool(glob.glob(pattern))


def audit_scene(batch_dir: str, scene: dict) -> dict:
    scene_id = scene["scene_id"]
    ego_id = str(scene["ego_id"])
    log_path = find_scene_log(batch_dir, scene_id, ego_id)
    log_text = read_text(log_path) if log_path else ""
    completion_reason = detect_completion_reason(log_text)
    exception = find_last_exception(log_text)

    if completion_reason is not None:
        status = "completed"
        reason = completion_reason
    elif has_timeout_record(batch_dir, scene_id, ego_id):
        status = "rerun"
        reason = "timeout"
    elif exception is not None:
        status = "rerun"
        reason = "exception"
    elif log_path is None:
        status = "rerun"
        reason = "missing"
    elif "return_code" not in log_text:
        status = "rerun"
        reason = "interrupted"
    else:
        status = "rerun"
        reason = "abnormal_exit"

    return {
        "scene_id": scene_id,
        "attack_type": scene["attack_type"],
        "ego_id": ego_id,
        "status": status,
        "reason": reason,
        "frames": count_frames(log_text),
        "last_simulation_timestamp": last_simulation_timestamp(log_text),
        "exception": exception,
        "log_path": log_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit a DriveArena batch and identify scenes that must rerun."
    )
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--scene-index", default=SCENE_INDEX_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    batch_dir = os.path.join(
        PROJECT_ROOT,
        "test_results",
        "attack",
        args.batch_id,
    )
    if not os.path.isdir(batch_dir):
        raise FileNotFoundError(f"Batch log directory does not exist: {batch_dir}")

    scenes = select_scenes_by_plan(
        get_all_scenes(args.scene_index),
        SCENE_PLAN,
        shuffle_within_type=False,
    )
    records = [audit_scene(batch_dir, scene) for scene in scenes]
    rerun_records = [record for record in records if record["status"] == "rerun"]
    summary = {
        "batch_id": args.batch_id,
        "total_expected": len(records),
        "completed": len(records) - len(rerun_records),
        "rerun": len(rerun_records),
        "counts_by_reason": dict(Counter(record["reason"] for record in records)),
    }

    report_path = os.path.join(batch_dir, "audit_report.json")
    rerun_path = os.path.join(batch_dir, "rerun_scenes.txt")
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            {"summary": summary, "scenes": records},
            file,
            ensure_ascii=False,
            indent=2,
        )
    with open(rerun_path, "w", encoding="utf-8") as file:
        for record in rerun_records:
            file.write(
                f"{record['scene_id']}\t{record['ego_id']}\t"
                f"{record['reason']}\tframes={record['frames']}\t"
                f"last_t={record['last_simulation_timestamp']}\n"
            )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Audit report: {report_path}")
    print(f"Rerun list  : {rerun_path}")


if __name__ == "__main__":
    main()
