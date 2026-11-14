import base64
from datetime import datetime
import io
import json
import os
import sys
import time
import math
from typing import Dict, List, Optional, Tuple
import dearpygui.dearpygui as dpg
from matplotlib import pyplot as plt
import requests
import numpy as np
import torch
import cv2
from PIL import Image
from io import BytesIO
import yaml
import copy
import argparse
from pathlib import Path
# Add LimSim to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "LimSim"))  # noqa
from TrafficManager.utils.sim_utils import limsim2diffusion, normalize_angle, transform_to_ego_frame, interpolate_traj
from TrafficManager.utils.map_utils import VectorizedLocalMap
from LimSim.utils.trajectory import Trajectory, State
from LimSim.trafficManager.traffic_manager import TrafficManager
from LimSim.simModel.MPGUI import GUI
from LimSim.simModel.Model import Model
from LimSim.simModel.DataQueue import CameraImages
from TrafficManager.utils.matplot_render import MatplotlibRenderer
from LimSim.simInfo.CustomExceptions import (
    CollisionChecker,
    CollisionException,
    OffRoadChecker,
)
from TrafficManager.utils.scorer import Scorer
from TrafficManager.attack_framework import (
    AttackRuntimeConfig,
    AttackTrajectoryInjector,
    AttackType,
    GeneratedAttackCodeExecutor,
    LLMClient,
    LLMServiceConfig,
    TwoStageTrajectoryGenerator,
    VLMTargetSelector,
    VehicleState,
    analyze_scene,
)


from TrafficManager.selector_config import (
    add_selector_arguments, apply_selector_overrides,
    resolve_selector_config, selector_metadata,
)


CAM_NAMES = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
    ]


class AutoQueue:
    def __init__(self, max_size):
        self.data = []
        self.max_size = max_size

    def put(self, item):
        if len(self.data) >= self.max_size:
            self.data.pop(0)
        self.data.append(item)

    def is_full(self):
        return len(self.data) == self.max_size


class SimulationManager:
    def __init__(self, config_path: str, scene_name: str = None,
                 sumo_cfg: str = None, sumo_rou: str = None, selector_args=None):
        self.config = apply_selector_overrides(self.load_config(config_path), selector_args)
        self.scene_name = scene_name or str(
            self.config.get("simulation", {}).get("ego_id", "attack")
        )
        self.sumo_cfg_override = sumo_cfg
        self.sumo_rou_override = sumo_rou
        self.setup_constants()
        self.setup_paths()
        self.model: Optional[Model] = None
        self.planner: Optional[TrafficManager] = None
        self.vectorized_map: Optional[VectorizedLocalMap] = None
        self.gui: Optional[GUI] = None
        self.renderer: Optional[MatplotlibRenderer] = None
        self.checkers: List = []
        self.scorer: Optional[Scorer] = None
        self.timestamp: float = -0.5
        self.data_template: Optional[torch.Tensor] = None
        self.last_pose: torch.Tensor = torch.eye(4)
        self.accel: List[float] = [0, 0, 9.80]
        self.rotation_rate: List[float] = [0, 0, 0]
        self.vel: List[float] = [0, 0, 0]
        self.agent_command: int = 2  # Defined by UniAD  0: Right 1:Left 2:Forward
        batch_run_id = os.environ.get("DRIVEARENA_BATCH_RUN_ID", "").strip()
        result_root = os.path.join(".", "results")
        if batch_run_id:
            result_root = os.path.join(result_root, batch_run_id)
        self.result_path = os.path.join(
            result_root,
            f"{self.scene_name}_{datetime.now().strftime('%m-%d-%H%M%S')}",
            "",
        )
        self.img_save_path = f"{self.result_path}imgs/"
        self.info_save_path = f"{self.result_path}infos/"
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.img_save_path, exist_ok=True)
        os.makedirs(self.info_save_path, exist_ok=True)

        # data queue
        self.data_queue = AutoQueue(max_size=7)
        self.info_queue = AutoQueue(max_size=5)
        self.frame_queue = AutoQueue(max_size=5)

        self.attack_runtime = AttackRuntimeConfig.from_mapping(
            self.config.get("attack", {})
        )
        self.llm_service = LLMServiceConfig.from_mapping(
            self.config.get("llm", {})
        )
        self.llm_client = LLMClient(self.llm_service)
        selector_mode, selector_service = resolve_selector_config(self.config)
        self.selector_service = LLMServiceConfig.from_mapping(selector_service)
        self.selection_metadata = selector_metadata(self.config)
        self.vlm_selector = VLMTargetSelector(
            LLMClient(self.selector_service),
            vision_model=self.selector_service.model,
            mode=selector_mode,
        )
        self.trajectory_generator = TwoStageTrajectoryGenerator(
            self.llm_client,
            GeneratedAttackCodeExecutor(),
        )
        self.attack_injector = AttackTrajectoryInjector(point_interval=0.1)
        self.attack_triggered = False
        self.active_attack_type = None
        self.active_attacker_id = None
        self.last_attack_replan_time = None
        self.attack_replan_completed = False
        self.attack_log_path = os.path.join(self.result_path, "attack_log.txt")

    @staticmethod
    def load_config(config_path: str) -> Dict:
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)

    def setup_constants(self):
        self.DIFFUSION_SERVER = self.config['servers']['diffusion']
        self.DRIVER_SERVER = self.config['servers']['driver']
        self.USE_AGENT_PATH =  self.config['simulation']['use_agent_path']
        self.STEP_LENGTH = self.config['simulation']['step_length']
        self.GUI_DISPLAY = self.config['simulation']['gui_display']
        self.MAX_SIM_TIME = self.config['simulation']['max_sim_time']
        self.EGO_ID = self.config['simulation']['ego_id']
        self.MAP_NAME = self.config['map']['name']
        self.GEN_PROMPT = self.config['map']['gen_description']
        self.IMAGE_SIZE = self.config['image']['size']
        self.TARGET_SIZE = tuple(self.config['image']['target_size'])
        self.AGENT_TYPE = self.config.get('agent', 'uniad')

    def setup_paths(self):
        data_root = os.path.dirname(os.path.abspath(__file__))
        sumo_cfg_file = self.sumo_cfg_override or self.config['map']['sumo_cfg_file']
        sumo_rou_file = self.sumo_rou_override or self.config['map']['sumo_rou_file']
        self.SUMO_CFG_FILE = self._resolve_path(data_root, sumo_cfg_file)
        self.SUMO_NET_FILE = os.path.join(
            data_root, self.config['map']['sumo_net_file'].format(map_name=self.MAP_NAME))
        self.SUMO_ROU_FILE = self._resolve_path(data_root, sumo_rou_file)
        self.DATA_TEMPLATE_PATH = os.path.join(
            data_root, self.config['data']['template_path'])
        self.NU_SCENES_DATA_ROOT = os.path.join(
            data_root, self.config['data']['nu_scenes_root'].format(map_name=self.MAP_NAME))

    def _resolve_path(self, data_root: str, path: str) -> str:
        resolved = path.format(
            map_name=self.MAP_NAME,
            ego_id=self.EGO_ID,
            scene_name=self.scene_name,
        )
        if os.path.isabs(resolved):
            return os.path.normpath(resolved)
        return os.path.normpath(os.path.join(data_root, resolved))

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def send_request_diffusion(self, data_list: List[Dict], output_filename: Optional[str] = None) -> Optional[np.ndarray]:
        # Response checks from sim_manager_t.py; layout decoding from sim_manager_t_ori.py.
        # Preserve the original hstack(first 3 / last 3), then vstack stitching.
        # Keep output_filename for the existing collision snapshot path.
        serialized_data = []
        for diffusion_data in data_list:
            serialized_data.append({
                k: v.numpy().tolist() if isinstance(v, torch.Tensor) else
                {k2: v2.numpy().tolist() if isinstance(v2, torch.Tensor) else v2 for k2, v2 in v.items()} if isinstance(v, dict) else
                v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in diffusion_data.items()
            })

        try:
            print(f"Sending data to WorldDreamer server...")
            response = requests.post(
                self.DIFFUSION_SERVER + "dreamer-api/", json=serialized_data,
                timeout=(10, 300))
            if response.status_code != 200:
                raise RuntimeError(
                    f"DreamForge request failed: HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )

            server_mode = response.headers.get("X-DreamForge-Mode")
            if server_mode != "online-conditioned":
                raise RuntimeError(
                    "DreamForge response is not confirmed as online-conditioned "
                    f"(mode={server_mode!r}). Update run_fastapi.py on the server "
                    "and restart it without --debug."
                )

            print(
                "DreamForge response: mode=online-conditioned, "
                f"server_timestamp={response.headers.get('X-DreamForge-Timestamp')}\n"
                "Current camera grid: "
                f"{response.headers.get('X-DreamForge-Camera-Order')}",
                flush=True,
            )
            if 'image' in response.headers.get('Content-Type', ''):
                image = Image.open(BytesIO(response.content)).convert('RGB')
                image_array = np.array(image)
                image_height, image_width = image_array.shape[:2]
                print(
                    f"DreamForge response image: {image_width}x{image_height}",
                    flush=True,
                )

                if image_height > image_width and image_height % 6 == 0:
                    images_array = np.stack(
                        np.split(image_array, 6, axis=0)
                    )
                    response_layout = "vertical-6"
                elif (
                    image_width > image_height
                    and image_height % 2 == 0
                    and image_width % 3 == 0
                ):
                    view_height = image_height // 2
                    view_width = image_width // 3
                    images_array = np.stack([
                        image_array[0:view_height, 0:view_width],
                        image_array[0:view_height, view_width:2 * view_width],
                        image_array[0:view_height, 2 * view_width:3 * view_width],
                        image_array[view_height:2 * view_height, 0:view_width],
                        image_array[view_height:2 * view_height, view_width:2 * view_width],
                        image_array[view_height:2 * view_height, 2 * view_width:3 * view_width],
                    ])
                    response_layout = "grid-2x3"
                else:
                    raise ValueError(
                        "Unsupported DreamForge response layout: "
                        f"{image_width}x{image_height}; expected vertical 6-view "
                        "or 2x3 grid"
                    )

                print(
                    f"DreamForge response layout: {response_layout}; "
                    "camera order preserved",
                    flush=True,
                )
                combined_image = np.vstack(
                    (np.hstack(images_array[:3]), np.hstack(images_array[3:])))
                frame_index = str(int(self.timestamp * 2)).zfill(3)
                diffusion_path = os.path.join(
                    self.img_save_path,
                    output_filename or f"diffusion_{frame_index}.jpg",
                )
                cv2.imwrite(
                    diffusion_path,
                    cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR),
                )
                print(
                    f"DreamForge frame saved: {diffusion_path} | "
                    "FL,F,FR / BR,B,BL",
                    flush=True,
                )
                return images_array
        except requests.exceptions.RequestException as e:
            print(f"Warning: Request failed due to {e}")
        return None

    def render_collision_diffusion(self) -> Optional[str]:
        try:
            drivable_mask = self.get_drivable_mask(self.model)
            diffusion_data, _ = limsim2diffusion(
                self.vehicles,
                self.data_template,
                self.vectorized_map,
                self.MAP_NAME,
                self.agent_command,
                self.last_pose,
                drivable_mask,
                self.accel,
                self.rotation_rate,
                self.vel,
                gen_location=self.MAP_NAME,
                gen_prompts=self.GEN_PROMPT,
                return_info=True,
            )
            self.data_queue.put(copy.deepcopy(diffusion_data))
            generated_images = self.send_request_diffusion(
                self.data_queue.data,
                output_filename="collision_diffusion.jpg",
            )
            if generated_images is None:
                return None
            return os.path.join(
                self.img_save_path,
                "collision_diffusion.jpg",
            )
        except Exception as error:
            print(
                f"WARNING: Collision diffusion rendering failed: {error}",
                flush=True,
            )
            return None

    def get_drivable_mask(self, model: Model) -> np.ndarray:
        img = np.zeros((self.IMAGE_SIZE, self.IMAGE_SIZE), dtype=np.uint8)
        roadgraphRenderData, VRDDict = model.renderQueue.get()
        egoVRD = VRDDict['egoCar'][0]
        ex, ey, ego_yaw = egoVRD.x, egoVRD.y, egoVRD.yaw

        OffRoadChecker().draw_roadgraph(img, roadgraphRenderData, ex, ey, ego_yaw)
        return img.astype(bool)

    def initialize_simulation(self):
        # Initialising models, planners, maps etc
        self.model = Model(
            egoID=self.EGO_ID, netFile=self.SUMO_NET_FILE, rouFile=self.SUMO_ROU_FILE,
            cfgFile=self.SUMO_CFG_FILE, dataBase=self.result_path+"limsim.db", SUMOGUI=False,
            CARLACosim=False,
        )
        self.model.start()
        self.planner = TrafficManager(
            self.model, config_file_path='./TrafficManager/LimSim/trafficManager/config.yaml')

        print(f"Testing connection to WorldDreamer & Driver servers...")
        requests.get(self.DIFFUSION_SERVER + "dreamer-clean/", timeout=(10, 15)).raise_for_status()
        requests.get(self.DRIVER_SERVER + "driver-clean/", timeout=(10, 15)).raise_for_status()

        self.data_template = torch.load(self.DATA_TEMPLATE_PATH)
        self.vectorized_map = VectorizedLocalMap(
            dataroot=self.NU_SCENES_DATA_ROOT, map_name=self.MAP_NAME, patch_size=[100, 100], fixed_ptsnum_per_line=-1)

        self.gui = GUI(self.model)
        if self.GUI_DISPLAY:
            self.gui.start()

        self.renderer = MatplotlibRenderer()
        self.collision_checker = CollisionChecker(
            contact_tolerance=0.05,
            near_miss_threshold=0.20,
        )
        self.checkers = [self.collision_checker, OffRoadChecker()]

    def record_clip(self, data):
        if not data:
            return None
        roadgraphRenderData, VRDDict = data[-1]
        frame_index = str(int(self.timestamp * 2)).zfill(3)
        bev_path = f'{self.img_save_path}bev_{frame_index}.png'
        self.renderer.render(roadgraphRenderData, VRDDict, bev_path)
        print(
            f'Frame {frame_index}: BEV={bev_path}, '
            f'Diffusion={self.img_save_path}diffusion_{frame_index}.jpg, '
            f'Agent={self.img_save_path}agent_{frame_index}.jpg',
            flush=True,
        )
        return bev_path

    def build_sumo_topology(self) -> Dict:
        topology = {
            "lanes": {},
            "junctions": {},
            "ego_lane_id": "unknown",
            "ego_edge_id": "unknown",
        }
        if getattr(self.model, "ego", None) is not None:
            topology["ego_lane_id"] = getattr(
                self.model.ego, "laneID", "unknown"
            )
            ego_lane_id = topology["ego_lane_id"]
            if ego_lane_id != "unknown" and "_" in ego_lane_id:
                topology["ego_edge_id"] = ego_lane_id.rsplit("_", 1)[0]
        if self.roadgraph:
            for lane_id, lane in getattr(self.roadgraph, "lanes", {}).items():
                edge_id = lane_id
                lane_index = None
                if "_" in lane_id:
                    edge_id, raw_lane_index = lane_id.rsplit("_", 1)
                    try:
                        lane_index = int(raw_lane_index)
                    except ValueError:
                        lane_index = None
                topology["lanes"][lane_id] = {
                    "id": lane_id,
                    "edge_id": edge_id,
                    "lane_index": lane_index,
                    "length": getattr(lane, "length", 100.0),
                    "width": getattr(lane, "width", 3.5),
                    "speed_limit": getattr(lane, "speed_limit", 13.89),
                    "is_ego_lane": lane_id == topology["ego_lane_id"],
                }
            for junction_id, junction in getattr(
                self.roadgraph, "junctions", {}
            ).items():
                topology["junctions"][junction_id] = {
                    "id": junction_id,
                    "x": getattr(junction, "x", 0),
                    "y": getattr(junction, "y", 0),
                }
        return topology

    def latest_bev_frame(self) -> Optional[str]:
        frames = list(Path(self.img_save_path).glob("bev_*.png"))
        if not frames:
            return None
        return str(max(frames, key=lambda path: path.stat().st_mtime))

    def launch_attack(self, simulation_time: float) -> None:
        if not self.attack_runtime.enabled or self.attack_triggered:
            return
        if not self.llm_service.enabled or not self.selector_service.enabled:
            raise RuntimeError("Attack requires both selector and trajectory model credentials")

        background_cars = self.vehicles.get("carInAoI", [])
        if not background_cars:
            print("Attack skipped: no interactive background vehicle", flush=True)
            return

        topology = self.build_sumo_topology()
        analysis = analyze_scene(
            self.vehicles["egoCar"], background_cars, topology
        )
        bev_image_path = self.latest_bev_frame() if self.vlm_selector.mode == "vlm" else None
        decision = self.vlm_selector.select(analysis, bev_image_path)
        target_car = next(
            car for car in background_cars
            if str(car.get("id", "")) == decision.target_vehicle_id
        )
        generated = self.trajectory_generator.generate(
            decision.attack_type,
            VehicleState.from_limsim(
                self.vehicles["egoCar"]
            ).to_prompt_dict(),
            VehicleState.from_limsim(target_car).to_prompt_dict(),
            topology,
        )
        self.attack_injector.add(
            decision.target_vehicle_id,
            generated.points,
            simulation_time,
        )
        self.attack_triggered = True
        self.active_attack_type = decision.attack_type
        self.active_attacker_id = decision.target_vehicle_id
        self.last_attack_replan_time = simulation_time
        if self.scorer is not None:
            self.scorer.data["metas"].update({
                "scene_name": self.scene_name,
                "attack_triggered": True,
                "attack_trigger_time": float(simulation_time),
                "attack_type": decision.attack_type.value,
                "attacker_id": decision.target_vehicle_id,
                "vlm_reasoning": decision.reasoning,  # Legacy reader compatibility.
                "selector_reasoning": decision.reasoning,
                **self.selection_metadata,
            })
        attack_record = {
            "scene_name": self.scene_name,
            "trigger_time": float(simulation_time),
            "attacker_id": decision.target_vehicle_id,
            "attack_type": decision.attack_type.value,
            "behavior_params": decision.behavior_params,
            "vlm_reasoning": decision.reasoning,
            "selector_reasoning": decision.reasoning,
            **self.selection_metadata,
            "selection_details": self.vlm_selector.last_selection,
            "structured_scene": analysis.description,
            "sumo_topology": topology,
            "goal_state": generated.goal_state,
            "trajectory": generated.points,
            "goal_code": generated.goal_code,
            "path_code": generated.path_code,
        }
        with open(
            os.path.join(self.result_path, "attack_record.json"),
            "w",
            encoding="utf-8",
        ) as attack_record_file:
            json.dump(
                attack_record,
                attack_record_file,
                ensure_ascii=False,
                indent=2,
                default=lambda value: (
                    value.item() if hasattr(value, "item") else str(value)
                ),
            )
        with open(self.attack_log_path, "a", encoding="utf-8") as attack_log:
            attack_log.write(
                f"timestamp={simulation_time:.1f}, "
                f"vehicle={decision.target_vehicle_id}, "
                f"type={decision.attack_type.value}, "
                f"reasoning={decision.reasoning}\n"
            )
        print(
            f"Attack generated: vehicle={decision.target_vehicle_id}, "
            f"type={decision.attack_type.value}, "
            f"points={len(generated.points)}",
            flush=True,
        )

    def update_attack_trajectory(self, simulation_time: float) -> None:
        attack_type = self.active_attack_type
        if (
            attack_type is None
            or self.attack_replan_completed
            or self.active_attacker_id is None
        ):
            return
        if not self.attack_runtime.llm_replanning and attack_type != AttackType.CUT_IN:
            return
        if (
            self.last_attack_replan_time is not None
            and simulation_time - self.last_attack_replan_time < 0.5 - 1e-6
        ):
            return

        background_cars = (
            self.vehicles.get("carInAoI", [])
            + self.vehicles.get("outOfAoI", [])
        )
        target_car = next(
            (
                car for car in background_cars
                if str(car.get("id", "")) == self.active_attacker_id
            ),
            None,
        )
        if target_car is None:
            print(
                f"{attack_type.value} replan skipped: vehicle {self.active_attacker_id} "
                "is outside the observable area",
                flush=True,
            )
            return

        ego_state = VehicleState.from_limsim(
            self.vehicles["egoCar"]
        ).to_prompt_dict()
        attacker_state = VehicleState.from_limsim(target_car).to_prompt_dict()
        completed = False
        if attack_type == AttackType.CUT_IN:
            completed = (
                ego_state["lane_id"] != "unknown"
                and ego_state["lane_id"] == attacker_state["lane_id"]
            )
        elif attack_type == AttackType.BRAKE_CHECK:
            completed = attacker_state["speed"] <= 0.01
            speeds = target_car.get("speedQ", ())
            if len(speeds) >= 2:
                attacker_state["acceleration"] = (speeds[-1] - speeds[-2]) / self.STEP_LENGTH
        elif attack_type == AttackType.OPPOSITE_DIRECTION_INTRUSION:
            heading = ego_state["heading"]
            # End updates once the oncoming vehicle has passed the ego longitudinally.
            relative_forward = (
                (attacker_state["x"] - ego_state["x"]) * math.cos(heading)
                + (attacker_state["y"] - ego_state["y"]) * math.sin(heading)
            )
            completed = relative_forward < 0.0
        if completed:
            self.attack_replan_completed = True
            with open(self.attack_log_path, "a", encoding="utf-8") as attack_log:
                attack_log.write(
                    f"{attack_type.value}_completed=true, timestamp={simulation_time:.1f}, "
                    f"vehicle={self.active_attacker_id}, "
                    f"lane={ego_state['lane_id']}\n"
                )
            print(
                f"{attack_type.value} completed at SUMO={simulation_time:.1f}s: "
                f"vehicle={self.active_attacker_id}, lane={ego_state['lane_id']}",
                flush=True,
            )
            return

        topology = self.build_sumo_topology()
        replan_source = "rule_based"
        points = None
        if self.attack_runtime.llm_replanning:
            try:
                generated = self.trajectory_generator.generate(
                    attack_type,
                    ego_state,
                    attacker_state,
                    topology,
                    runtime_replan=True,
                )
                points = generated.points
                replan_source = "llm"
            except Exception as error:
                print(
                    f"LLM {attack_type.value} replan failed ({type(error).__name__})",
                    flush=True,
                )
                if attack_type != AttackType.CUT_IN:
                    self.last_attack_replan_time = simulation_time
                    with open(self.attack_log_path, "a", encoding="utf-8") as attack_log:
                        attack_log.write(
                            f"{attack_type.value}_replan=false, source=previous_segment, "
                            f"timestamp={simulation_time:.1f}, vehicle={self.active_attacker_id}\n"
                        )
                    return
                replan_source = "rule_based_fallback"
        if points is None:
            ego_lane = topology.get("lanes", {}).get(topology.get("ego_lane_id"), {})
            points = GeneratedAttackCodeExecutor.replan_cut_in(
                ego_state,
                attacker_state,
                speed_limit=float(ego_lane.get("speed_limit", 13.89)),
            )
        self.attack_injector.add(
            self.active_attacker_id,
            points,
            simulation_time,
        )
        self.last_attack_replan_time = simulation_time
        with open(self.attack_log_path, "a", encoding="utf-8") as attack_log:
            attack_log.write(
                f"{attack_type.value}_replan=true, source={replan_source}, "
                f"timestamp={simulation_time:.1f}, "
                f"vehicle={self.active_attacker_id}, "
                f"ego=({ego_state['x']:.2f},{ego_state['y']:.2f}), "
                f"attacker=({attacker_state['x']:.2f},{attacker_state['y']:.2f}), "
                f"target_speed={points[-1]['v']:.2f}\n"
            )
        print(
            f"{attack_type.value} trajectory updated at SUMO={simulation_time:.1f}s: "
            f"vehicle={self.active_attacker_id}, "
            f"target_speed={points[-1]['v']:.2f}m/s",
            flush=True,
        )

    def process_frame(self):
        # Single frame processing logic
        if self.scorer is None:
            self.scorer = Scorer(self.model, map_name=self.MAP_NAME,
                                 save_file_path=self.result_path+"drive_arena.pkl")
            self.scorer.data["metas"].update({
                "scene_name": self.scene_name,
                "attack_triggered": False,
                "attack_trigger_time": float(self.attack_runtime.trigger_time),
                "attack_type": "unknown",
                "attacker_id": None,
                "collision": False,
                "near_collision": False,
            })
        try:
            for checker in self.checkers:
                checker.check(self.model)
            near_collision = self.collision_checker.last_near_collision
            if near_collision is not None:
                simulation_time = self.model.timeStep * self.STEP_LENGTH
                previous_separation = self.scorer.data["metas"].get(
                    "near_collision_min_separation"
                )
                if (
                    previous_separation is None
                    or near_collision["separation"] < previous_separation
                ):
                    self.scorer.data["metas"].update({
                        "near_collision": True,
                        "near_collision_time": float(simulation_time),
                        "near_collision_vehicle_id": near_collision["vehicle_id"],
                        "near_collision_min_separation": float(
                            near_collision["separation"]
                        ),
                    })
                    print(
                        "Near collision recorded at SUMO="
                        f"{simulation_time:.1f}s: vehicle="
                        f"{near_collision['vehicle_id']}, separation="
                        f"{near_collision['separation']:.3f}m",
                        flush=True,
                    )
                    with open(
                        self.attack_log_path,
                        "a",
                        encoding="utf-8",
                    ) as attack_log:
                        attack_log.write(
                            "near_collision=true, "
                            f"timestamp={simulation_time:.1f}, "
                            f"vehicle={near_collision['vehicle_id']}, "
                            "separation="
                            f"{near_collision['separation']:.3f}m\n"
                        )
        except CollisionException as collision:
            simulation_time = self.model.timeStep * self.STEP_LENGTH
            self.scorer.record_frame(
                None,
                is_planning_frame=False,
            )
            self.scorer.data["metas"].update({
                "collision": True,
                "collision_time": float(simulation_time),
                "collision_detail": str(collision),
            })
            self.model.putRenderData()
            roadgraph_render_data, vehicle_render_data = self.model.renderQueue.get()
            collision_bev_path = os.path.join(
                self.img_save_path, "collision_bev.png"
            )
            self.renderer.render(
                roadgraph_render_data,
                vehicle_render_data,
                collision_bev_path,
            )
            collision_diffusion_path = self.render_collision_diffusion()
            self.scorer.data["metas"].update({
                "collision_bev_path": collision_bev_path,
                "collision_diffusion_path": collision_diffusion_path,
            })
            with open(self.attack_log_path, "a", encoding="utf-8") as attack_log:
                attack_log.write(
                    f"collision=true, timestamp={simulation_time:.1f}, "
                    f"detail={collision}, "
                    f"collision_bev={collision_bev_path}, "
                    "collision_diffusion="
                    f"{collision_diffusion_path or 'generation_failed'}\n"
                )
            print(
                f"Collision recorded at SUMO={simulation_time:.1f}s: "
                f"{collision}\nCollision BEV: {collision_bev_path}\n"
                "Collision Diffusion: "
                f"{collision_diffusion_path or 'generation failed'}",
                flush=True,
            )
            return False
        except Exception as error:
            print(
                f"WARNING: Checker failed @ timestep {self.model.timeStep}. "
                f"{error}"
            )
            raise


        drivable_mask = self.get_drivable_mask(self.model)
        simulation_time = self.model.timeStep * self.STEP_LENGTH
        if self.model.timeStep % 5 == 0 and self.data_queue.is_full():
            self.timestamp += 0.5
            if self.timestamp >= self.MAX_SIM_TIME:
                print("Simulation time end.")
                return False

            limsim_trajectories = self.planner.plan(
                self.model.timeStep * 0.1, self.roadgraph, self.vehicles)
            if not limsim_trajectories[self.EGO_ID].states:
                return True

            traj_len = min(
                len(limsim_trajectories[self.EGO_ID].states) - 1, 25)
            local_x, local_y, local_yaw = transform_to_ego_frame(
                limsim_trajectories[self.EGO_ID].states[0], limsim_trajectories[self.EGO_ID].states[traj_len])
            self.agent_command = 2 if local_x <= 5.0 else (
                1 if local_y > 4.0 else 0 if local_y < -4.0 else 2)
            print("Agent command:", self.agent_command)

            diffusion_data, frame_infos = limsim2diffusion(
                self.vehicles, self.data_template, self.vectorized_map, self.MAP_NAME, self.agent_command, self.last_pose, drivable_mask,
                self.accel, self.rotation_rate, self.vel,
                gen_location=self.MAP_NAME,
                gen_prompts=self.GEN_PROMPT,
                return_info=True
            )
            self.data_queue.put(copy.deepcopy(diffusion_data))

            self.last_pose = diffusion_data['metas']['ego_pos']
            gen_images = self.send_request_diffusion(self.data_queue.data)

            if gen_images is not None:
                front_left_image, front_image, front_right_image = [
                    Image.fromarray(img).convert('RGBA') for img in gen_images[:3]]
            else:
                raise ValueError("No images generated!")

            new_width, new_height = self.TARGET_SIZE[0], int(
                (self.TARGET_SIZE[0] / front_image.width) * front_image.height)
            resized_images = [img.resize((new_width, new_height), Image.Resampling.LANCZOS) for img in [
                front_left_image, front_image, front_right_image]]

            ci = CameraImages()
            ci.CAM_FRONT_LEFT, ci.CAM_FRONT, ci.CAM_FRONT_RIGHT = [
                np.array(img) for img in resized_images]
            print("Current timestamp:", self.timestamp)

            driver_deadline = time.monotonic() + float(
                self.config.get('servers', {}).get('driver_response_timeout', 180))
            response = requests.get(self.DRIVER_SERVER + "driver-get/", timeout=(10, 15))
            while response.status_code != 200 or response.text == "false":
                if time.monotonic() >= driver_deadline:
                    raise TimeoutError(
                        f"Driver response not ready after "
                        f"{self.config.get('servers', {}).get('driver_response_timeout', 180)}s"
                    )
                # print("The Driver Agent not processing done, try again in 1s")
                time.sleep(0.5)
                response = requests.get(self.DRIVER_SERVER + "driver-get/", timeout=(10, 15))
                # print("Driver Agent", response.status_code)

            driver_output = json.loads(response.text)
            if self.AGENT_TYPE == 'uniad':
                path_points = driver_output["bbox_results"][0]["planning_traj"][0]
            elif self.AGENT_TYPE == 'vad':
                path_points = driver_output[0]["planning_traj"]
            print("Driver Agent's Path:", path_points)

            # add driver predict BEV
            if self.AGENT_TYPE == 'uniad':
                pred_bev_base64 = driver_output["bev_pred_img"]
            elif self.AGENT_TYPE == 'vad':
                pred_bev_base64 = driver_output[0]["bev_pred_img"]
            pred_bev_img = base64.b64decode(pred_bev_base64)
            pred_bev_img = Image.open(io.BytesIO(pred_bev_img))
            # save image
            pred_bev_img = pred_bev_img.convert('RGB')
            pred_bev_img.save(
                f"{self.img_save_path}agent_{str(int(self.timestamp*2)).zfill(3)}.jpg")
            pred_bev_img = pred_bev_img.convert('RGBA')
            pred_bev_img = pred_bev_img.resize(
                (800, 800), Image.Resampling.LANCZOS)
            ci.PRED_BEV = np.array(pred_bev_img, dtype=np.float32)

            self.model.imageQueue.put(ci)

            path_points.insert(0, [0.0, 0.0])
            ego_vehicle = self.vehicles['egoCar']
            ego_traj = interpolate_traj(ego_vehicle, path_points)

            if len(limsim_trajectories[self.EGO_ID].states) < 10:
                yaw_rate = 0
            else:
                yaw_rate = limsim_trajectories[self.EGO_ID].states[9].yaw - \
                    limsim_trajectories[self.EGO_ID].states[0].yaw
            vx_1, vx_2 = path_points[2][0] - \
                path_points[0][0], path_points[3][0] - path_points[1][0]
            vy_1, vy_2 = path_points[2][1] - \
                path_points[0][1], path_points[3][1] - path_points[1][1]
            ax, ay = (vx_2 - vx_1) / 0.5, (vy_2 - vy_1) / 0.5
            self.accel = [ax, ay, 9.80]
            self.rotation_rate = [0, 0, yaw_rate]
            self.vel = [limsim_trajectories[self.EGO_ID].states[0].vel, 0, 0]
            print("Accel:", self.accel, "\nRotation rate:",
                  self.rotation_rate, "\nVel:", self.vel)

            self.model.putRenderData()
            roadgraphRenderData, VRDDict = self.model.renderQueue.get()
            self.frame_queue.put([copy.deepcopy(roadgraphRenderData), copy.deepcopy(VRDDict)])
            self.record_clip(self.frame_queue.data)

            if (
                not self.attack_triggered
                and simulation_time >= self.attack_runtime.trigger_time
            ):
                self.launch_attack(simulation_time)

            self.scorer.record_frame(drivable_mask, is_planning_frame=True,
                                     planned_traj=ego_traj, ref_traj=limsim_trajectories[self.EGO_ID])

            limsim_trajectories = {}
            if self.USE_AGENT_PATH and self.timestamp > 1:
                ## Because first two frames, drive agents may not ready
                print(f"Use agent path to drive.")
                limsim_trajectories[self.EGO_ID] = ego_traj
            self.update_attack_trajectory(simulation_time)
            limsim_trajectories = self.attack_injector.merge(
                limsim_trajectories,
                simulation_time,
            )
            self.model.setTrajectories(limsim_trajectories)

            # Save frame info
            frame_infos['ego']['YawRate'] = yaw_rate
            frame_infos['ego']['Command'] = self.agent_command
            self.info_queue.put(copy.deepcopy(frame_infos))

            with open(f"{self.info_save_path}{str(int(self.timestamp*2)).zfill(3)}.json", 'w') as f:
                json.dump(self.info_queue.data, f, indent=4)

        else:
            diffusion_data, frame_infos = limsim2diffusion(
                self.vehicles, self.data_template, self.vectorized_map, self.MAP_NAME, self.agent_command, self.last_pose, drivable_mask,
                self.accel, self.rotation_rate, self.vel,
                gen_location=self.MAP_NAME,
                gen_prompts=self.GEN_PROMPT,
                return_info=True
            )
            self.last_pose = diffusion_data['metas']['ego_pos']
            self.data_queue.put(copy.deepcopy(diffusion_data))
            self.info_queue.put(copy.deepcopy(frame_infos))

            self.model.putRenderData()
            roadgraphRenderData, VRDDict = self.model.renderQueue.get()
            self.frame_queue.put([copy.deepcopy(roadgraphRenderData), copy.deepcopy(VRDDict)])


            if self.scorer is not None:
                self.scorer.record_frame(
                    drivable_mask, is_planning_frame=False)

        return True

    def run_simulation(self):
        self.initialize_simulation()
        try:
            while not self.model.tpEnd:
                self.model.moveStep()
                self.roadgraph, self.vehicles = self.model.exportSce()
                if self.vehicles and 'egoCar' in self.vehicles:
                    if not self.process_frame():
                        break
                self.model.updateVeh()
        finally:
            self.cleanup()

    def cleanup(self):
        print("Simulation ends")
        if self.scorer:
            self.scorer.save()
        if self.model:
            self.model.destroy()
        if self.gui is not None and getattr(self.gui, "_popen", None) is not None:
            if self.gui.is_alive():
                self.gui.terminate()
            self.gui.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(
        description="DreamForge closed-loop adversarial simulation"
    )
    parser.add_argument(
        "--config", default="./TrafficManager/config-attack.yaml"
    )
    parser.add_argument("--scene_name", default=None)
    parser.add_argument("--sumo_cfg", default=None)
    parser.add_argument("--sumo_rou", default=None)
    parser.add_argument("--attacker_id", default=None)
    parser.add_argument("--attack_type", default=None)
    parser.add_argument("--attack_time", type=float, default=None)
    add_selector_arguments(parser)
    args = parser.parse_args()

    sim_manager = SimulationManager(
        config_path=args.config,
        scene_name=args.scene_name,
        sumo_cfg=args.sumo_cfg,
        sumo_rou=args.sumo_rou,
        selector_args=args,
    )
    sim_manager.run_simulation()


if __name__ == '__main__':
    main()
