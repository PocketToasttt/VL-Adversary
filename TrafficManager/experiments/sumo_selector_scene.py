"""SUMO scene experiment using existing selection/generation/injection code.
No changes to production entry points; no DreamForge or UniAD requests.
"""
from pathlib import Path
import copy
import argparse
import hashlib
from datetime import datetime, timezone, timedelta
import json
import os
import sys
import time
import traceback
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TrafficManager"))
os.environ.setdefault("MPLBACKEND", "Agg")
parser = argparse.ArgumentParser()
parser.add_argument('--scene-id', required=True)
parser.add_argument('--run-id', required=True)
parser.add_argument('--variant', choices=('vlm_image', 'vlm_text', 'qwen_plus'), required=True)
parser.add_argument('--cache-dir', required=True)
parser.add_argument('--vision-model', default='qwen3-vl-plus')
parser.add_argument('--templates-dir')
parser.add_argument('--selection-dir')
args = parser.parse_args()
RUN_ID = args.run_id
RUN_ROOT = ROOT / 'results' / RUN_ID
STATUS_ROOT = RUN_ROOT / '_status' / args.scene_id
STATUS_ROOT.mkdir(parents=True, exist_ok=False)
print('RUN_ROOT:', RUN_ROOT, flush=True)
import yaml
import requests
from TrafficManager import sim_manager_t_attack as existing
from TrafficManager import sim_manager_only as original_open_loop
from TrafficManager.LimSim.simInfo.CustomExceptions import (
    CollisionException as OpenLoopCollisionException,
    OffRoadException as OpenLoopOffRoadException,
)
from TrafficManager.attack_framework import LLMServiceConfig, VehicleState
from LimSim.simInfo.CustomExceptions import OffRoadException
from TrafficManager.selector_config import selector_metadata
from TrafficManager.experiments.sumo_selector_protocol import (
    VARIANTS, PROMPT_VERSION, shared_rules, cached_planner_completion,
)
from TrafficManager.attack_framework import vlm_selector as selection_module
common_rules = shared_rules(selection_module.TEXT_SELECTION_RULES)
selection_module.TEXT_SELECTION_RULES = common_rules
selection_module.VISUAL_SELECTION_RULES = common_rules

INDEX = ROOT / "TrafficManager/networkFiles/boston-seaport/scenarios_diverse_3types/scene_index.json"
index = json.loads(INDEX.read_text(encoding="utf-8"))
assert index["statistics"]["total_scenes"] == 105
SCENE_ID = args.scene_id
assert SCENE_ID in index["statistics"]["scene_ids"]
SCENE_DIR = INDEX.parent / SCENE_ID
scene = json.loads((SCENE_DIR / (SCENE_ID + ".json")).read_text(encoding="utf-8"))
RUNTIME_CONFIG = yaml.safe_load((ROOT / "TrafficManager/config-attack.yaml").read_text(encoding="utf-8"))
RUNTIME_CONFIG["simulation"].update(ego_id=str(scene["ego_id"]), gui_display=False, use_agent_path=False, max_sim_time=10.0)
RUNTIME_CONFIG["attack"].update(enable=True, mode="auto", trigger_time=2.0)
# Keep trajectory updates fixed in the selector-only comparison protocol.
RUNTIME_CONFIG["attack"]["llm_replanning"] = False
RUNTIME_CONFIG["selector"] = dict(VARIANTS[args.variant])
if args.variant in ('vlm_image', 'vlm_text'):
    RUNTIME_CONFIG["selector"]["model"] = args.vision_model
RUNTIME_CONFIG["llm"]["api_key_env"] = "DASHSCOPE_API_KEY"
for preset in ("target_vehicle_id", "attacker_candidate_id", "attack_type"):
    RUNTIME_CONFIG["attack"].pop(preset, None)
if not os.environ.get("DASHSCOPE_API_KEY") and not RUNTIME_CONFIG["llm"].get("api_key"):
    prior_path = ROOT / "TrafficManager/temp_ego_317_mqbq0vhx.yaml"
    prior_service = (yaml.safe_load(prior_path.read_text(encoding="utf-8")) or {}).get("llm", {})
    if prior_service.get("api_base", "").rstrip("/") != RUNTIME_CONFIG["llm"]["api_base"].rstrip("/"):
        raise RuntimeError("Prior credential belongs to a different API endpoint")
    os.environ["DASHSCOPE_API_KEY"] = prior_service.get("api_key") or ""
    print("Using explicitly authorized credential from existing scene configuration (value hidden).", flush=True)
service = LLMServiceConfig.from_mapping(RUNTIME_CONFIG["llm"])
if not service.enabled:
    raise RuntimeError("No API credential available for the configured model service")
assert service.model == "qwen-coder-plus-1106"
os.environ["DRIVEARENA_BATCH_RUN_ID"] = RUN_ID

# The current client sends requests in a bounded worker process.
# Validate its destination before dispatch, and block service HTTP in this process.
from TrafficManager.attack_framework import llm_client
allowed_url = service.api_base.rstrip('/') + '/chat/completions'
http_calls = []
original_completion = llm_client._bounded_completion_request

def audited_completion(request, timeout):
    model = request['payload']['model']
    if request['url'] != allowed_url or model not in ('qwen-plus', args.vision_model, 'qwen-coder-plus-1106'):
        raise RuntimeError('Unexpected model endpoint blocked')
    call = {'model': model, 'path': urlsplit(request['url']).path}
    http_calls.append(call)
    started = time.monotonic()
    try:
        if model == service.model:
            result = cached_planner_completion(request, timeout, original_completion, args.cache_dir, call)
        else:
            result = original_completion(request, timeout)
        call['success'] = True
        return result
    finally:
        call['elapsed_seconds'] = round(time.monotonic() - started, 3)

llm_client._bounded_completion_request = audited_completion

def block_service_http(session, method, url, *args, **kwargs):
    raise RuntimeError('SUMO-only experiment blocked service HTTP: ' + urlsplit(url).path)

requests.sessions.Session.request = block_service_http


class FirstSceneExperiment(existing.SimulationManager):
    @staticmethod
    def load_config(config_path):
        return copy.deepcopy(RUNTIME_CONFIG)

    def initialize_simulation(self):
        # Reuse the repository's existing no-server initialization unchanged.
        original_open_loop.SimulationManager.initialize_simulation(self)
        self.completion_reason = None
        self.collision_time = None
        print("Using original sim_manager_only initialization and frame processing; no DreamForge/UniAD calls.", flush=True)

    def render_collision_bev(self):
        self.model.putRenderData()
        road, vehicles = self.model.renderQueue.get()
        path = str(Path(self.img_save_path) / "collision_bev.png")
        self.renderer.render(road, vehicles, path)
        return path

    def process_frame(self):
        previous_timestamp = self.timestamp
        simulation_time = self.model.timeStep * self.STEP_LENGTH
        try:
            # Reuse existing SUMO checks, planning, data preparation, clock and stop rule.
            keep_running = original_open_loop.SimulationManager.process_frame(self)
        except OpenLoopCollisionException as collision:
            self.scorer.record_frame(None, is_planning_frame=False)
            self.scorer.data["metas"].update(collision=True, collision_time=simulation_time, collision_detail=str(collision))
            self.completion_reason = "collision"
            self.collision_time = simulation_time
            path = self.render_collision_bev()
            print(f"COLLISION @ SUMO={simulation_time:.1f}s: {collision}; BEV={path}", flush=True)
            return False
        except OpenLoopOffRoadException as error:
            self.scorer.record_frame(None, is_planning_frame=False)
            self.scorer.data["metas"].update(offroad=True, offroad_time=simulation_time, offroad_detail=str(error))
            self.completion_reason = "offroad"
            print(f"OFFROAD @ SUMO={simulation_time:.1f}s: {error}", flush=True)
            return False
        self.scorer.data["type"] = "sumo-only"
        self.scorer.data["metas"].update(
            experiment_mode="sumo-only", dreamforge_enabled=False, uniad_enabled=False,
            **self.selection_metadata,
        )
        if not keep_running:
            self.completion_reason = "max_sim_time"
            return False
        if self.timestamp != previous_timestamp:
            self.model.putRenderData()
            road, vehicles = self.model.renderQueue.get()
            self.frame_queue.put([copy.deepcopy(road), copy.deepcopy(vehicles)])
            # Original recorder determines the filenames: bev_000, bev_001, ...
            bev = self.record_clip(self.frame_queue.data)
            print(f"FRAME SUMO={simulation_time:.1f}s timestamp={self.timestamp:.1f}s BEV={bev}", flush=True)
            if not self.attack_triggered and simulation_time >= self.attack_runtime.trigger_time:
                analysis = existing.analyze_scene(self.vehicles["egoCar"], self.vehicles.get("carInAoI", []), self.build_sumo_topology())
                print("SELECTION INPUT (shared text):\n" + analysis.description, flush=True)
                self.launch_attack(simulation_time)
            self.update_attack_trajectory(simulation_time)
            self.model.setTrajectories(self.attack_injector.merge({}, simulation_time))
        return True


metadata = {
    "scene_id": SCENE_ID, "scene_index": index["statistics"]["scene_ids"].index(SCENE_ID) + 1, "scene_count": 105,
    "ego_id": str(scene["ego_id"]), "duration": 10.0, "attack_time": 2.0,
    "experiment_mode": "sumo-only", "dreamforge_enabled": False, "uniad_enabled": False,
    **selector_metadata(RUNTIME_CONFIG),
    "variant": args.variant, "selector_prompt_version": PROMPT_VERSION,
    "prompt_sha256": hashlib.sha256(common_rules.encode()).hexdigest(),
    "ego_control": "SUMO nominal; no UniAD feedback",
    "trajectory_cache": "identical planner requests reuse first response across variants",
}
(STATUS_ROOT / "experiment_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
print("EXPERIMENT:", json.dumps(metadata, ensure_ascii=False), flush=True)
manager = FirstSceneExperiment(
    config_path="TrafficManager/config-attack.yaml", scene_name=SCENE_ID,
    sumo_cfg=str(SCENE_DIR / (SCENE_ID + ".sumocfg")),
    sumo_rou=str(SCENE_DIR / (SCENE_ID + ".rou.xml")),
)
manager.selection_metadata.update(selector_prompt_version=PROMPT_VERSION, variant=args.variant)
if args.templates_dir:
    from TrafficManager.experiments.reusable_trajectory_templates import use_templates
    use_templates(manager.trajectory_generator, args.templates_dir)
    manager.selection_metadata['trajectory_execution'] = 'shared frozen functions per attack type'
# Preserve the exact existing selector inputs/output for later independent review.
original_select = manager.vlm_selector.select

def record_selection(analysis, bev_image_path=None):
    snapshot = {
        'scene_id': SCENE_ID, 'description': analysis.description,
        'text_sha256': hashlib.sha256(analysis.description.encode('utf-8')).hexdigest(),
        'candidates': analysis.candidates, 'skipped': analysis.skipped,
        'image_attached': VARIANTS[args.variant]['mode'] == 'vlm',
        'prompt': common_rules, 'variant': args.variant,
        'bev_sha256': hashlib.sha256(Path(bev_image_path).read_bytes()).hexdigest() if bev_image_path else None,
    }
    path = Path(manager.result_path) / 'selection_audit.json'
    try:
        if args.selection_dir:
            from TrafficManager import run_selector_only_comparison as selection_only
            from TrafficManager.attack_framework.models import AttackDecision
            phase1 = Path(args.selection_dir)
            inputs = phase1 / 'inputs' / SCENE_ID
            inputs.mkdir(parents=True, exist_ok=True)
            description = selection_only.clean_description(analysis.description)
            parsed = selection_only.parse_candidates(description)
            selection_input = {'description': description}
            row = dict(sample=SCENE_ID, scene_id=SCENE_ID, category=SCENE_ID.rsplit('_', 1)[0],
                       arm=args.variant, model=RUNTIME_CONFIG['selector']['model'],
                       image_attached=bev_image_path is not None, raw_response=None, api_error=None,
                       text_sha256=snapshot['text_sha256'], image_sha256=snapshot['bev_sha256'])
            selection_only.save_json(inputs / (args.variant + '.json'), snapshot)
            if bev_image_path:
                (inputs / 'bev.png').write_bytes(Path(bev_image_path).read_bytes())
            tick = time.monotonic()
            try:
                row['raw_response'] = manager.vlm_selector.client.create_completion(
                    selection_only.messages_for(selection_input, bev_image_path, bev_image_path is not None),
                    model=row['model'], temperature=0.5, timeout=120)
                row.update(selection_only.validate(row['raw_response'], parsed))
            except Exception as error:
                row.update(selection_only.validate(None, parsed))
                row['api_error'] = str(error).replace(service.api_key, '[REDACTED]')
            row['elapsed_seconds'] = round(time.monotonic() - tick, 3)
            selection_only.save_json(phase1 / 'responses' / (SCENE_ID + '_' + args.variant + '.json'), row)
            snapshot['selection_details'] = dict(raw_response=row['raw_response'], topology_corrected=False,
                                                 output_legal=row['output_legal'])
            if not row['output_legal'] or row['topology_compatible'] is False:
                raise ValueError('Selection output invalid or inconsistent with necessary road conditions; retained in stage1')
            decision_data = json.loads(row['raw_response'])
            manager.vlm_selector.last_selection = dict(snapshot['selection_details'], raw_decision=decision_data)
            return AttackDecision.from_mapping(decision_data)
        decision = original_select(analysis, bev_image_path)
        snapshot['selection_details'] = manager.vlm_selector.last_selection
        return decision
    finally:
        path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2,
                                   default=str), encoding='utf-8')

manager.vlm_selector.select = record_selection
started = time.monotonic()
status = dict(metadata)
try:
    manager.run_simulation()
    reason = manager.completion_reason or ("route_end" if manager.model.tpEnd else "unknown")
    status.update(complete=reason in ("max_sim_time", "collision", "offroad", "route_end"),
                  completion_reason=reason, attack_triggered=manager.attack_triggered,
                  attacker_id=manager.active_attacker_id,
                  attack_type=manager.active_attack_type.value if manager.active_attack_type else None,
                  collision_time=manager.collision_time,
                  last_simulation_time=manager.model.timeStep * manager.STEP_LENGTH,
                  result_path=manager.result_path)
except Exception as error:
    status.update(complete=False, error_type=type(error).__name__, error=str(error))
    traceback.print_exc()
finally:
    status["wall_time_seconds"] = round(time.monotonic() - started, 3)
    status["http_calls"] = http_calls
    (STATUS_ROOT / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print("RUN STATUS:", json.dumps(status, ensure_ascii=False, indent=2), flush=True)
if not status.get("complete") or not status.get("attack_triggered"):
    sys.exit(1)
