# batch_test.py

import argparse
import json
import os
import queue
import signal
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict

import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from TrafficManager.selector_config import (
    add_selector_arguments, apply_selector_overrides,
    resolve_selector_config, selector_metadata, ensure_experiment_manifest,
)
SELECTOR_ARGS = None

CONFIG_PATH = os.path.join(SCRIPT_DIR, "config-attack.yaml")
SCENE_INDEX_PATH = os.path.join(
    SCRIPT_DIR,
    "networkFiles/boston-seaport/scenarios_diverse_3types/scene_index.json",
)
ATTACK_SCRIPT = os.path.join(SCRIPT_DIR, "sim_manager_t_attack.py")

# "attack": run adversarial attack; "baseline": run without attack for comparison.
RUN_MODE = "attack"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results", RUN_MODE)
BATCH_RUN_ID = ""


VALID_COMPLETION_REASONS = {"max_sim_time", "collision", "offroad"}


def detect_completion_reason(log_text: str):
    if "Simulation time end." in log_text:
        return "max_sim_time"
    if (
        "Collision recorded at SUMO=" in log_text
        or "CollisionException" in log_text
    ):
        return "collision"
    if "OffRoadException" in log_text or "body off road" in log_text:
        return "offroad"
    return None


def is_valid_completion_marker(marker_path: str, result_file: str) -> bool:
    try:
        with open(marker_path, "r", encoding="utf-8") as marker_file:
            marker = json.load(marker_file)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if marker.get("completion_reason") in VALID_COMPLETION_REASONS:
        return True
    try:
        with open(result_file, "r", encoding="utf-8", errors="ignore") as log_file:
            return detect_completion_reason(log_file.read()) is not None
    except OSError:
        return False

# Keep this batch runner aligned with sim_manager_test_attack.py.
SCENE_PLAN = {
    "brake_check": 35,
    "cut_in": 35,
    "opposite_direction_intrusion": 35,
}

ATTACK_TYPE_CN = {
    "brake_check": "前车急刹",
    "cut_in": "强行切入",
    "opposite_direction_intrusion": "对向车道入侵",
}

SHUFFLE_WITHIN_TYPE = False
RANDOM_SEED = 42
SCENE_TIMEOUT_SECONDS = 120


def load_scene_index(scene_index_path: str) -> dict:
    if not os.path.exists(scene_index_path):
        raise FileNotFoundError(f"scene_index.json 不存在: {scene_index_path}")
    with open(scene_index_path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_scene_file(scene_index_path: str, scene: dict, suffix: str) -> str:
    scene_id = scene.get("scene_id", "")
    return os.path.normpath(
        os.path.join(os.path.dirname(scene_index_path), scene_id, f"{scene_id}{suffix}")
    )


def load_scene_metadata(scene_index_path: str, indexed_scene: dict) -> dict:
    scene_id = indexed_scene.get("scene_id", "")
    scene_json_path = resolve_scene_file(scene_index_path, indexed_scene, ".json")
    if not os.path.isfile(scene_json_path):
        raise FileNotFoundError(f"场景元数据不存在: {scene_json_path}")

    with open(scene_json_path, "r", encoding="utf-8") as file:
        actual_scene = json.load(file)

    if actual_scene.get("scene_id") != scene_id:
        raise ValueError(
            f"场景目录与元数据不一致: index={scene_id}, "
            f"json={actual_scene.get('scene_id')}"
        )

    indexed_ego = str(indexed_scene.get("ego_id", ""))
    actual_ego = str(actual_scene.get("ego_id", ""))
    if indexed_ego and actual_ego and indexed_ego != actual_ego:
        print(
            f"⚠️  {scene_id} 总索引已过期: ego {indexed_ego} -> {actual_ego}，"
            "使用场景目录中的元数据"
        )

    return actual_scene


def get_all_scenes(scene_index_path: str) -> list:
    index = load_scene_index(scene_index_path)
    raw_scenes = index.get("scenes", [])
    allowed_types = set(SCENE_PLAN.keys())

    scenes = []
    for indexed_scene in raw_scenes:
        scene_id = indexed_scene.get("scene_id", "")
        if not scene_id:
            continue

        try:
            scene = load_scene_metadata(scene_index_path, indexed_scene)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"⚠️  跳过无效场景 {scene_id}: {exc}")
            continue

        attack_type = scene.get("attack_type", "")
        ego_id = str(scene.get("ego_id", ""))

        if not attack_type or not ego_id:
            continue
        if attack_type not in allowed_types:
            continue

        scenes.append(
            {
                "scene_id": scene_id,
                "attack_type": attack_type,
                "scenario_type": scene.get("scenario_type", ""),
                "type": attack_type,
                "ego_id": ego_id,
                "attacker_candidate_id": scene.get("attacker_candidate_id", None),
                "attack_time": scene.get("attack_time", 2.0),
                "score": scene.get("score", 0),
                "signature": scene.get("signature", ""),
                "description": scene.get("description", ""),
                "config_file": resolve_scene_file(
                    scene_index_path, scene, ".sumocfg"
                ),
                "route_file": resolve_scene_file(
                    scene_index_path, scene, ".rou.xml"
                ),
                "scene_dir": scene.get("scene_dir", ""),
            }
        )

    return scenes


def select_scenes_by_plan(
    all_scenes: list,
    scene_plan: dict,
    shuffle_within_type: bool = False,
) -> list:
    scenes_by_type = defaultdict(list)
    for scene in all_scenes:
        scenes_by_type[scene["attack_type"]].append(scene)

    for attack_type in scenes_by_type:
        scenes_by_type[attack_type].sort(key=lambda item: item["scene_id"])

    if shuffle_within_type:
        import random

        random.seed(RANDOM_SEED)
        for attack_type in scenes_by_type:
            random.shuffle(scenes_by_type[attack_type])

    print("可用场景统计（按 attack_type 分组）:")
    print(f"   {'attack_type':<32} {'数量':>4}  {'avg_score':>9}  scenario_type")
    print("   " + "-" * 76)
    for attack_type in sorted(scenes_by_type.keys()):
        group = scenes_by_type[attack_type]
        avg_score = sum(scene["score"] for scene in group) / len(group)
        scenario_types = sorted({scene["scenario_type"] for scene in group})
        print(
            f"   {attack_type:<32} {len(group):>4}  {avg_score:>8.1f}  "
            f"{', '.join(scenario_types)}"
        )

    print("\n场景选择计划:")
    print(f"   {'attack_type':<32} {'请求':>4}  {'可用':>4}  {'选中':>4}  中文名")
    print("   " + "-" * 76)

    selected = []
    for attack_type, limit in scene_plan.items():
        group = scenes_by_type.get(attack_type, [])
        available = len(group)
        picked = min(limit, available)
        cn_name = ATTACK_TYPE_CN.get(attack_type, attack_type)
        status = "" if picked == limit else "  不足"
        print(
            f"   {attack_type:<32} {limit:>4}  {available:>4}  "
            f"{picked:>4}  {cn_name}{status}"
        )
        selected.extend(group[:limit])

    return selected


def make_temp_config(
    base_config_path: str,
    ego_id: str,
    gui_display: bool = False,
    attacker_id: int = None,
    attack_type: str = None,
    attack_time: float = 2.0,
    duration: float = 10.0,
) -> str:
    with open(base_config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config = apply_selector_overrides(config, SELECTOR_ARGS)

    config["simulation"]["ego_id"] = str(ego_id)
    config["simulation"]["gui_display"] = gui_display
    config["simulation"]["max_sim_time"] = float(duration)

    if "use_worlddreamer" in config.get("simulation", {}):
        config["simulation"]["use_worlddreamer"] = False

    config.setdefault("attack", {})
    config["attack"]["enable"] = RUN_MODE == "attack"
    if attacker_id is not None:
        config["attack"]["attacker_candidate_id"] = int(attacker_id)
        config["attack"]["target_vehicle_id"] = str(attacker_id)
    if attack_type:
        config["attack"]["attack_type"] = attack_type
    config["attack"]["attack_time"] = float(attack_time)
    config["attack"]["trigger_time"] = float(attack_time)

    fd, temp_config_path = tempfile.mkstemp(
        prefix=f"temp_ego_{ego_id}_",
        suffix=".yaml",
        dir=SCRIPT_DIR,
    )
    os.close(fd)

    with open(temp_config_path, "w", encoding="utf-8") as file:
        yaml.dump(config, file, allow_unicode=True, sort_keys=False)

    return temp_config_path


def stop_scene_processes(process):
    """Reap the scene and its SUMO/image workers, including on timeout."""
    if process is None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    process.wait()


def run_scene(
    scene: dict, scene_index: int, total_scenes: int, duration: float
) -> bool:
    ego_id = scene["ego_id"]
    attack_type = scene["attack_type"]
    scene_id = scene["scene_id"]
    attacker_id = scene.get("attacker_candidate_id")
    attack_time = scene.get("attack_time", 2.0)
    score = scene.get("score", 0)
    cn_name = ATTACK_TYPE_CN.get(attack_type, attack_type)

    effective_scene_name = f"baseline_{scene_id}" if RUN_MODE == "baseline" else scene_id
    temp_config_path = None
    process = None
    completion_dir = os.path.join(RESULTS_DIR, "completed")
    completion_path = os.path.join(
        completion_dir,
        f"{scene_id}_ego{ego_id}.done.json",
    )
    result_file = os.path.join(
        RESULTS_DIR,
        f"{RUN_MODE}_{scene_index:03d}_{scene_id}_ego{ego_id}.txt",
    )
    if (
        os.path.isfile(completion_path)
        and is_valid_completion_marker(completion_path, result_file)
    ):
        print(
            f"\n[SKIP COMPLETED] [{scene_index + 1}/{total_scenes}] "
            f"{effective_scene_name}",
            flush=True,
        )
        return True

    try:
        temp_config_path = make_temp_config(
            CONFIG_PATH,
            ego_id,
            gui_display=False,
            attacker_id=attacker_id,
            attack_type=attack_type,
            attack_time=attack_time,
            duration=duration,
        )

        mode_tag = "BASELINE" if RUN_MODE == "baseline" else "ATTACK"
        print(
            f"\n▶ [{mode_tag}] 场景 [{scene_index + 1}/{total_scenes}]\n"
            f"   scene_name = {effective_scene_name}\n"
            f"   attack_type= {attack_type}（{cn_name}）\n"
            f"   ego={ego_id}  att={attacker_id}  "
            f"score={score}  t_attack={attack_time:.1f}s"
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
        env["PYTHONUNBUFFERED"] = "1"
        env["DRIVEARENA_BATCH_RUN_ID"] = BATCH_RUN_ID

        cmd = [
            sys.executable,
            "-u",
            ATTACK_SCRIPT,
            "--config",
            temp_config_path,
            "--scene_name",
            effective_scene_name,
        ]

        config_file = scene.get("config_file", "")
        route_file = scene.get("route_file", "")
        missing_files = [
            path for path in (config_file, route_file)
            if not path or not os.path.isfile(path)
        ]
        if missing_files:
            print(f"   失败: 场景文件不存在: {', '.join(missing_files)}")
            return False

        if config_file:
            cmd += ["--sumo_cfg", config_file]
        if route_file:
            cmd += ["--sumo_rou", route_file]
        if attacker_id is not None:
            cmd += ["--attacker_id", str(attacker_id)]
        if attack_type:
            cmd += ["--attack_type", attack_type]
        cmd += ["--attack_time", str(attack_time)]

        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(result_file, "w", encoding="utf-8") as file:
            file.write("=" * 60 + "\n")
            file.write(f"run_mode             : {RUN_MODE}\n")
            file.write(f"scene_id             : {scene_id}\n")
            file.write(f"effective_scene_name : {effective_scene_name}\n")
            file.write(f"attack_type          : {attack_type}（{cn_name}）\n")
            file.write(f"scenario_type        : {scene.get('scenario_type', '')}\n")
            file.write(f"ego_id               : {ego_id}\n")
            file.write(f"attacker_id          : {attacker_id}\n")
            file.write(f"attack_time          : {attack_time:.2f}s\n")
            file.write(f"score                : {score}\n")
            file.write(f"signature            : {scene.get('signature', '')}\n")
            file.write(f"description          : {scene.get('description', '')}\n")
            file.write(f"config_file          : {config_file}\n")
            file.write(f"route_file           : {route_file}\n")
            file.write("=" * 60 + "\n\n")
            file.flush()

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=PROJECT_ROOT,
                env=env,
                start_new_session=(os.name == "posix"),
            )
            output_queue = queue.Queue()

            def read_output():
                try:
                    for output_line in iter(process.stdout.readline, ""):
                        output_queue.put(output_line)
                finally:
                    process.stdout.close()
                    output_queue.put(None)

            reader = threading.Thread(target=read_output, daemon=True)
            reader.start()
            started_at = time.monotonic()

            while True:
                if (
                    time.monotonic() - started_at > SCENE_TIMEOUT_SECONDS
                ):
                    stop_scene_processes(process)
                    reader.join(timeout=1)
                    raise subprocess.TimeoutExpired(cmd, SCENE_TIMEOUT_SECONDS)

                try:
                    output_line = output_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if output_line is None:
                    break

                print(output_line, end="", flush=True)
                file.write(output_line)
                file.flush()

            reader.join(timeout=1)
            return_code = process.wait()
            file.write(f"\nreturn_code          : {return_code}\n")
            file.flush()

        with open(result_file, "r", encoding="utf-8", errors="ignore") as log_file:
            result_log = log_file.read()
        completion_reason = detect_completion_reason(result_log)
        if completion_reason is not None:
            os.makedirs(completion_dir, exist_ok=True)
            completion_record = {
                "batch_run_id": BATCH_RUN_ID,
                "run_mode": RUN_MODE,
                "scene_id": scene_id,
                "effective_scene_name": effective_scene_name,
                "ego_id": str(ego_id),
                "attack_type": attack_type,
                "return_code": int(return_code),
                "status": "completed",
                "completion_reason": completion_reason,
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            temporary_completion_path = completion_path + ".tmp"
            with open(
                temporary_completion_path,
                "w",
                encoding="utf-8",
            ) as completion_file:
                json.dump(
                    completion_record,
                    completion_file,
                    ensure_ascii=False,
                    indent=2,
                )
            os.replace(temporary_completion_path, completion_path)
        else:
            print(
                f"[INCOMPLETE] {effective_scene_name} exited with "
                f"returncode={return_code}; it will run again on resume.",
                flush=True,
            )

        if completion_reason is not None:
            print(
                f"[COMPLETED:{completion_reason}] {effective_scene_name}",
                flush=True,
            )
            return True

        print(
            f"[FAILED/INCOMPLETE] {effective_scene_name} "
            f"(returncode={return_code})",
            flush=True,
        )
        return False

    except subprocess.TimeoutExpired:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        result_file = os.path.join(
            RESULTS_DIR,
            f"{RUN_MODE}_{scene_index:03d}_{scene_id}_ego{ego_id}_TIMEOUT.txt",
        )
        with open(result_file, "w", encoding="utf-8") as file:
            file.write(f"run_mode    : {RUN_MODE}\n")
            file.write(f"scene_id    : {scene_id}\n")
            file.write(f"attack_type : {attack_type}（{cn_name}）\n")
            file.write(f"ego_id      : {ego_id}\n")
            file.write(f"attacker_id : {attacker_id}\n")
            file.write(f"attack_time : {attack_time:.2f}s\n")
            file.write(f"score       : {score}\n")
            file.write(f"result      : TIMEOUT（超过 {SCENE_TIMEOUT_SECONDS}s）\n")
        print(f"   超时: {effective_scene_name}")
        return False

    except KeyboardInterrupt:
        stop_scene_processes(process)
        print(
            f"\n[INTERRUPTED] {effective_scene_name} was not marked complete; "
            "it will run again on resume.",
            flush=True,
        )
        raise

    except Exception as exc:
        print(f"   异常: {effective_scene_name} -> {exc}")
        return False

    finally:
        stop_scene_processes(process)
        if temp_config_path and os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def print_scene_list(selected_scenes: list) -> None:
    mode_tag = "B" if RUN_MODE == "baseline" else "A"
    print(f"\n本次共运行 {len(selected_scenes)} 个场景\n")
    print(
        f"   {'':<3}  {'scene_id':<35} {'ego':>4} {'att':>4} "
        f"{'score':>5} {'t_att':>6}  attack_type（中文）"
    )
    print("   " + "-" * 100)
    for index, scene in enumerate(selected_scenes, 1):
        cn_name = ATTACK_TYPE_CN.get(scene["attack_type"], scene["attack_type"])
        print(
            f"   {mode_tag}[{index:02d}] {scene['scene_id']:<35} "
            f"{str(scene['ego_id']):>4} {str(scene['attacker_candidate_id']):>4} "
            f"{scene['score']:>5} {scene['attack_time']:>5.1f}s  "
            f"{scene['attack_type']}（{cn_name}）"
        )


def print_summary(selected_scenes: list, passed: int) -> None:
    total = len(selected_scenes)
    failed = total - passed
    total_by_type = defaultdict(int)
    for scene in selected_scenes:
        total_by_type[scene["attack_type"]] += 1

    print(f"\n{'=' * 60}")
    print(f"  运行模式 : {RUN_MODE.upper()}")
    print(f"  总计     : 通过 {passed}/{total}，失败 {failed}/{total}")
    print(f"  结果目录 : {RESULTS_DIR}")
    print("  类型统计 :")
    for attack_type, count in total_by_type.items():
        print(f"    {attack_type:<32} {count}")
    print(f"{'=' * 60}\n")


def parse_scene_plan(value: str) -> dict:
    plan = {}
    for item in value.split(","):
        attack_type, count = item.split("=", 1)
        plan[attack_type.strip()] = int(count)
    required = set(SCENE_PLAN)
    if set(plan) != required:
        raise argparse.ArgumentTypeError(
            "scene-plan必须同时包含brake_check、cut_in和"
            "opposite_direction_intrusion"
        )
    return plan


def parse_args():
    parser = argparse.ArgumentParser(description="DreamForge三类攻击批量测试")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument(
        "--scene-index", default=SCENE_INDEX_PATH,
        help="场景索引路径；默认使用原始105场景，可指定独立的新场景库",
    )
    add_selector_arguments(parser)
    parser.add_argument("--mode", choices=["attack", "baseline"], default="attack")
    parser.add_argument(
        "--scene-plan",
        type=parse_scene_plan,
        default=dict(SCENE_PLAN),
        help="例如 brake_check=35,cut_in=35,opposite_direction_intrusion=35",
    )
    parser.add_argument("--per-type-limit", type=int, default=None)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--attack-time", type=float, default=2.0)
    parser.add_argument("--scene-timeout", type=int, default=600)
    parser.add_argument(
        "--scene-ids",
        default=None,
        help=(
            "Comma-separated scene IDs to run, for example "
            "cut_in_012,cut_in_027,opposite_direction_intrusion_016"
        ),
    )
    parser.add_argument(
        "--resume-run",
        default=None,
        help="继续已有批次，例如 batch_attack_20260720_205838",
    )
    return parser.parse_args()


def main():
    global RUN_MODE, RESULTS_DIR, SCENE_TIMEOUT_SECONDS, BATCH_RUN_ID, CONFIG_PATH, SCENE_INDEX_PATH, SELECTOR_ARGS
    args = parse_args()
    RUN_MODE = args.mode
    CONFIG_PATH = os.path.abspath(args.config)
    SCENE_INDEX_PATH = os.path.abspath(args.scene_index)
    SELECTOR_ARGS = args
    with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
        base_config = apply_selector_overrides(yaml.safe_load(config_file) or {}, args)
    metadata = selector_metadata(base_config) if RUN_MODE == "attack" else {}
    if args.resume_run:
        if os.path.basename(args.resume_run) != args.resume_run:
            raise ValueError("--resume-run 必须是批次ID，不能包含目录路径")
        if not args.resume_run.startswith(f"batch_{RUN_MODE}_"):
            raise ValueError("--resume-run 与当前 --mode 不匹配")
        BATCH_RUN_ID = args.resume_run
    else:
        tag = ""
        if RUN_MODE == "attack":
            tag = "_" + re.sub(r"[^A-Za-z0-9_-]+", "-", metadata["selector_mode"] + "_" + metadata["selector_model"])
        BATCH_RUN_ID = f"batch_{RUN_MODE}{tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    RESULTS_DIR = os.path.join(
        PROJECT_ROOT, "test_results", RUN_MODE, BATCH_RUN_ID
    )
    SCENE_TIMEOUT_SECONDS = args.scene_timeout
    print(
        "  Batch data  : "
        + os.path.join(PROJECT_ROOT, "results", BATCH_RUN_ID),
        flush=True,
    )
    scene_plan = dict(args.scene_plan)
    if args.per_type_limit is not None:
        scene_plan = {
            attack_type: min(count, args.per_type_limit)
            for attack_type, count in scene_plan.items()
        }

    if RUN_MODE == "attack":
        _, selection_service = resolve_selector_config(base_config)
        for stage, service in (("trajectory", base_config.get("llm", {})), ("selector", selection_service)):
            key_env = str(service.get("api_key_env", "DEEPSEEK_API_KEY"))
            key = (os.environ.get(key_env) if key_env else None) or service.get("api_key")
            if not service.get("enable", False) or not key:
                raise RuntimeError(f"{stage} service is disabled or missing credentials (api_key_env={key_env!r})")
        print("  Selection   : " + metadata["selector_mode"] + " / " + metadata["selector_model"])
        print("  Trajectory  : " + metadata["trajectory_model"])
    print(f"\n{'=' * 65}")
    print("  批量攻击测试框架 | 对齐 sim_manager_test_attack.py | 三类场景")
    print(f"  运行模式 : {RUN_MODE.upper()}")
    print(f"  结果目录 : {RESULTS_DIR}")
    print(f"  scene_index: {SCENE_INDEX_PATH}")
    print(f"  sim 脚本   : {ATTACK_SCRIPT}")
    print(f"{'=' * 65}\n")

    all_scenes = get_all_scenes(SCENE_INDEX_PATH)
    if not all_scenes:
        print("scene_index.json 中没有三类可用场景，请先运行场景生成脚本。")
        return

    print(f"共加载 {len(all_scenes)} 个三类候选场景\n")
    selected_scenes = select_scenes_by_plan(
        all_scenes,
        scene_plan,
        shuffle_within_type=SHUFFLE_WITHIN_TYPE,
    )
    if args.scene_ids:
        requested_scene_ids = {
            scene_id.strip()
            for scene_id in args.scene_ids.split(",")
            if scene_id.strip()
        }
        available_scene_ids = {
            scene["scene_id"] for scene in selected_scenes
        }
        missing_scene_ids = requested_scene_ids - available_scene_ids
        if missing_scene_ids:
            raise ValueError(
                "Unknown or unselected scene IDs: "
                + ", ".join(sorted(missing_scene_ids))
            )
        selected_scenes = [
            scene
            for scene in selected_scenes
            if scene["scene_id"] in requested_scene_ids
        ]
    if not selected_scenes:
        print("根据当前 SCENE_PLAN 没有选中任何场景。")
        return

    if RUN_MODE == "attack":
        metadata.update({
            "scene_ids": [scene["scene_id"] for scene in selected_scenes],
            "scene_index": os.path.abspath(SCENE_INDEX_PATH),
            "duration": args.duration,
            "attack_time": args.attack_time,
        })
        ensure_experiment_manifest(RESULTS_DIR, metadata, resume=bool(args.resume_run))
    print_scene_list(selected_scenes)

    passed = 0
    for index, scene in enumerate(selected_scenes):
        scene["attack_time"] = args.attack_time
        if run_scene(scene, index, len(selected_scenes), args.duration):
            passed += 1

    print_summary(selected_scenes, passed)


if __name__ == "__main__":
    main()
