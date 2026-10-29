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
from pathlib import Path
import argparse
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
from LimSim.simInfo.CustomExceptions import CollisionChecker, OffRoadChecker
from TrafficManager.utils.scorer import Scorer
from TrafficManager.attack_framework import (
    AttackRuntimeConfig,
    AttackTrajectoryInjector,
    GeneratedAttackCodeExecutor,
    LLMClient,
    LLMServiceConfig,
    TwoStageTrajectoryGenerator,
    VLMTargetSelector,
    VehicleState,
    analyze_scene,
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
                 sumo_cfg: str = None, sumo_rou: str = None):
        self.config = self.load_config(config_path)
        self.scene_name = scene_name or str(
            self.config.get('simulation', {}).get('ego_id', 'dreamforge')
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
        timestamp = datetime.now().strftime('%m-%d-%H%M%S')
        self.result_path = f"./results/{self.scene_name}_{timestamp}/"
        self.img_save_path = f"{self.result_path}imgs/"
        self.info_save_path = f"{self.result_path}infos/"
        os.makedirs(self.result_path, exist_ok=True)
        os.makedirs(self.img_save_path, exist_ok=True)
        os.makedirs(self.info_save_path, exist_ok=True)

        # data queue
        self.data_queue = AutoQueue(max_size=9)
        self.info_queue = AutoQueue(max_size=5)
        self.frame_queue = AutoQueue(max_size=5)

        self.attack_runtime = AttackRuntimeConfig.from_mapping(
            self.config.get('attack', {})
        )
        self.llm_service = LLMServiceConfig.from_mapping(
            self.config.get('llm', {})
        )
        self.llm_client = LLMClient(self.llm_service)
        self.vlm_selector = VLMTargetSelector(self.llm_client)
        self.trajectory_generator = TwoStageTrajectoryGenerator(
            self.llm_client,
            GeneratedAttackCodeExecutor(),
        )
        self.attack_injector = AttackTrajectoryInjector(point_interval=0.1)
        self.attack_triggered = False
        self.attack_log_path = os.path.join(self.result_path, 'attack_log.txt')
        with open(self.attack_log_path, 'w', encoding='utf-8') as attack_log:
            attack_log.write('DreamForge 对抗性轨迹注入日志\n')
            attack_log.write('=' * 50 + '\n')

        print(
            f"攻击配置: enable={self.attack_runtime.enabled}, "
            f"trigger={self.attack_runtime.trigger_time}s, "
            f"llm={self.llm_service.enabled}"
        )

    @staticmethod
    def load_config(config_path: str) -> Dict:
        with open(config_path, 'r') as config_file:
            return yaml.safe_load(config_file)

    def setup_constants(self):
        self.DIFFUSION_SERVER = os.getenv(
            'DREAMFORGE_SERVER_URL', self.config['servers']['diffusion']
        ).rstrip('/') + '/'
        self.DRIVER_SERVER = os.getenv(
            'DRIVER_SERVER_URL', self.config['servers']['driver']
        ).rstrip('/') + '/'
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
        self.DRIVER_RESPONSE_TIMEOUT = self.config.get(
            'servers', {}
        ).get('driver_response_timeout', 180)

    def wait_for_driver_output(self) -> Dict:
        deadline = time.monotonic() + self.DRIVER_RESPONSE_TIMEOUT
        attempts = 0
        while time.monotonic() < deadline:
            attempts += 1
            response = requests.get(
                self.DRIVER_SERVER + 'driver-get/', timeout=10
            )
            if response.status_code == 200:
                return response.json()
            if response.status_code != 204:
                raise RuntimeError(
                    f'Driver 服务异常: HTTP {response.status_code}, '
                    f'{response.text[:200]}'
                )
            if attempts == 1 or attempts % 10 == 0:
                elapsed = attempts * 0.5
                print(
                    f'等待 {self.AGENT_TYPE.upper()} 推理结果: '
                    f'{elapsed:.1f}s（输出队列仍为空）',
                    flush=True,
                )
            time.sleep(0.5)
        raise TimeoutError(
            f'等待 {self.AGENT_TYPE.upper()} 结果超过 '
            f'{self.DRIVER_RESPONSE_TIMEOUT}s。请检查 DreamForge 服务端是否出现 '
            f'POST /driver-api/，以及驾驶模型推理是否报错。'
        )

    def setup_paths(self):
        data_root = os.path.dirname(os.path.abspath(__file__))
        self.SUMO_CFG_FILE = self._resolve_path(
            data_root, self.sumo_cfg_override or self.config['map']['sumo_cfg_file'])
        self.SUMO_NET_FILE = os.path.join(
            data_root, self.config['map']['sumo_net_file'].format(map_name=self.MAP_NAME))
        self.SUMO_ROU_FILE = self._resolve_path(
            data_root, self.sumo_rou_override or self.config['map']['sumo_rou_file'])
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

    def send_request_diffusion(self, data_list: List[Dict]) -> Optional[np.ndarray]:
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
                self.DIFFUSION_SERVER + "dreamer-api/", json=serialized_data)
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
                grid_image = Image.open(BytesIO(response.content)).convert('RGB')
                grid_array = np.array(grid_image)
                grid_height, grid_width = grid_array.shape[:2]
                if grid_height % 2 or grid_width % 3:
                    raise ValueError(
                        f"DreamForge响应尺寸不是2x3网格: {grid_width}x{grid_height}"
                    )
                view_height = grid_height // 2
                view_width = grid_width // 3
                images_array = np.stack([
                    grid_array[0:view_height, 0:view_width],
                    grid_array[0:view_height, view_width:2 * view_width],
                    grid_array[0:view_height, 2 * view_width:3 * view_width],
                    grid_array[view_height:2 * view_height, 0:view_width],
                    grid_array[view_height:2 * view_height, view_width:2 * view_width],
                    grid_array[view_height:2 * view_height, 2 * view_width:3 * view_width],
                ])
                frame_index = str(int(self.timestamp * 2)).zfill(3)
                diffusion_path = f"{self.img_save_path}diffusion_{frame_index}.jpg"
                cv2.imwrite(
                    diffusion_path,
                    cv2.cvtColor(grid_array, cv2.COLOR_RGB2BGR),
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
        requests.get(self.DIFFUSION_SERVER + "dreamer-clean/")
        requests.get(self.DRIVER_SERVER + "driver-clean/")

        self.data_template = torch.load(self.DATA_TEMPLATE_PATH)
        self.vectorized_map = VectorizedLocalMap(
            dataroot=self.NU_SCENES_DATA_ROOT, map_name=self.MAP_NAME, patch_size=[100, 100], fixed_ptsnum_per_line=-1)

        self.gui = GUI(self.model)
        if self.GUI_DISPLAY:
            self.gui.start()

        self.renderer = MatplotlibRenderer()
        self.checkers = [OffRoadChecker(), CollisionChecker()]

    def save_bev_frame(self, roadgraph_render_data, vehicle_render_data) -> str:
        frame_index = int(round(self.timestamp * 2))
        bev_path = os.path.join(
            self.img_save_path, f"bev_{str(frame_index).zfill(3)}.png"
        )
        self.renderer.render(
            roadgraph_render_data,
            vehicle_render_data,
            bev_path,
        )
        print(
            f"BEV frame saved: {bev_path} "
            f"(SUMO={self.model.timeStep * self.STEP_LENGTH:.1f}s, "
            f"DreamForge={self.timestamp:.1f}s)",
            flush=True,
        )
        return bev_path

    def save_frame_manifest(self, bev_path: str, agent_path: str) -> str:
        frame_index = str(int(self.timestamp * 2)).zfill(3)
        manifest_path = os.path.join(
            self.info_save_path, f"frame_{frame_index}.json"
        )
        manifest = {
            "frame_index": frame_index,
            "sumo_time": self.model.timeStep * self.STEP_LENGTH,
            "dreamforge_time": self.timestamp,
            "bev_image": bev_path,
            "diffusion_image": os.path.join(
                self.img_save_path, f"diffusion_{frame_index}.jpg"
            ),
            "agent_image": agent_path,
            "camera_grid": [
                ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
                ["CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"],
            ],
        }
        with open(manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, ensure_ascii=False, indent=2)
        print(
            f"Frame aligned: index={frame_index}, "
            f"SUMO={manifest['sumo_time']:.1f}s, "
            f"DreamForge={self.timestamp:.1f}s\n"
            f"  BEV       : {bev_path}\n"
            f"  Diffusion : {manifest['diffusion_image']}\n"
            f"  Agent     : {agent_path}",
            flush=True,
        )
        return manifest_path

    def build_sumo_topology(self) -> Dict:
        topology = {
            'lanes': {},
            'junctions': {},
            'ego_lane_id': 'unknown',
            'ego_edge_id': 'unknown',
        }
        try:
            if getattr(self.model, 'ego', None) is not None:
                topology['ego_lane_id'] = getattr(
                    self.model.ego, 'laneID', 'unknown'
                )
            if self.roadgraph:
                for lane_id, lane in getattr(self.roadgraph, 'lanes', {}).items():
                    topology['lanes'][lane_id] = {
                        'id': lane_id,
                        'length': getattr(lane, 'length', 100.0),
                        'width': getattr(lane, 'width', 3.5),
                        'speed_limit': getattr(lane, 'speed_limit', 13.89),
                    }
                for junction_id, junction in getattr(
                    self.roadgraph, 'junctions', {}
                ).items():
                    topology['junctions'][junction_id] = {
                        'id': junction_id,
                        'x': getattr(junction, 'x', 0),
                        'y': getattr(junction, 'y', 0),
                    }
        except Exception as exc:
            print(f"构建SUMO拓扑失败: {exc}")
        return topology

    def latest_bev_frame(self) -> Optional[str]:
        frames = list(Path(self.img_save_path).glob('bev_*.png'))
        if not frames:
            return None
        selected_frame = max(frames, key=lambda path: path.stat().st_mtime)
        print(f"VLM使用BEV图像: {selected_frame}", flush=True)
        return str(selected_frame)

    def launch_attack(
        self,
        attack_start_time: Optional[float] = None,
        bev_image_path: Optional[str] = None,
    ) -> None:
        if not self.attack_runtime.enabled or self.attack_triggered:
            return
        if not self.llm_service.enabled:
            print("攻击已启用，但VLM/LLM服务未启用或缺少API Key。")
            return
        if 'egoCar' not in self.vehicles:
            print("攻击中止: 当前场景没有Ego车辆。")
            return

        background_cars = self.vehicles.get('carInAoI', [])
        if not background_cars:
            print("攻击中止: Ego周围没有候选车辆。")
            return

        try:
            topology = self.build_sumo_topology()
            analysis = analyze_scene(
                self.vehicles['egoCar'], background_cars, topology
            )
            selected_bev_path = bev_image_path or self.latest_bev_frame()
            decision = self.vlm_selector.select(analysis, selected_bev_path)
            target_index, target_car = next(
                (index, car)
                for index, car in enumerate(background_cars)
                if str(car.get('id', '')) == decision.target_vehicle_id
            )
            ego_state = VehicleState.from_limsim(
                self.vehicles['egoCar']
            ).to_prompt_dict()
            attacker_state = VehicleState.from_limsim(
                target_car
            ).to_prompt_dict()
            generated = self.trajectory_generator.generate(
                decision.attack_type,
                ego_state,
                attacker_state,
                topology,
            )
            effective_start_time = (
                self.timestamp if attack_start_time is None else attack_start_time
            )
            self.attack_injector.add(
                decision.target_vehicle_id,
                generated.points,
                effective_start_time,
            )
            self.attack_triggered = True
            print(
                f"攻击已生成: vehicle={decision.target_vehicle_id}, "
                f"type={decision.attack_type.value}, points={len(generated.points)}, "
                f"SUMO触发时间={effective_start_time:.1f}s, "
                f"VLM图像={selected_bev_path}"
            )
            with open(self.attack_log_path, 'a', encoding='utf-8') as attack_log:
                attack_log.write(
                    f"timestamp={effective_start_time:.1f}, "
                    f"vehicle={decision.target_vehicle_id}, "
                    f"target_index={target_index}, "
                    f"type={decision.attack_type.value}, "
                    f"reasoning={decision.reasoning}\n"
                )
        except Exception as exc:
            print(f"攻击生成失败: {exc}")

    def process_frame(self):
        # Single frame processing logic
        if self.scorer is None:
            self.scorer = Scorer(self.model, map_name=self.MAP_NAME,
                                 save_file_path=self.result_path+"drive_arena.pkl")
        try:
            for checker in self.checkers:
                checker.check(self.model)
        except Exception as e:
            print(
                f"WARNING: Checker failed @ timestep {self.model.timeStep}. {e}")
            raise e
        

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

            driver_output = self.wait_for_driver_output()
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
            frame_index = str(int(self.timestamp * 2)).zfill(3)
            agent_path = f"{self.img_save_path}agent_{frame_index}.jpg"
            pred_bev_img.save(agent_path)
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
            current_bev_path = self.save_bev_frame(
                roadgraphRenderData, VRDDict
            )
            self.save_frame_manifest(current_bev_path, agent_path)

            if (
                self.attack_runtime.enabled
                and not self.attack_triggered
                and simulation_time >= self.attack_runtime.trigger_time
            ):
                self.launch_attack(simulation_time, current_bev_path)

            self.scorer.record_frame(drivable_mask, is_planning_frame=True,
                                     planned_traj=ego_traj, ref_traj=limsim_trajectories[self.EGO_ID])

            limsim_trajectories = {}
            if self.USE_AGENT_PATH and self.timestamp > 1:
                ## Because first two frames, drive agents may not ready
                print(f"Use agent path to drive.")
                limsim_trajectories[self.EGO_ID] = ego_traj
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

            if (
                self.attack_runtime.enabled
                and not self.attack_triggered
                and simulation_time >= self.attack_runtime.trigger_time
            ):
                current_bev_path = self.save_bev_frame(
                    roadgraphRenderData, VRDDict
                )
                self.launch_attack(simulation_time, current_bev_path)
                attack_trajectories = self.attack_injector.merge(
                    {}, simulation_time
                )
                if attack_trajectories:
                    self.model.setTrajectories(attack_trajectories)


            if self.scorer is not None:
                self.scorer.record_frame(
                    drivable_mask, is_planning_frame=False)

        return True

    def run_simulation(self):
        self.initialize_simulation()
        loop_elapsed_time = 0.0
        try:
            while not self.model.tpEnd:
                self.model.moveStep()
                loop_elapsed_time += self.STEP_LENGTH
                if loop_elapsed_time >= self.MAX_SIM_TIME:
                    print(f"Simulation time end: {loop_elapsed_time:.1f}s")
                    break
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
            try:
                self.scorer.save()
            except Exception as exc:
                print(f"保存评分结果失败: {exc}")
        if self.model:
            try:
                self.model.destroy()
            except Exception as exc:
                print(f"关闭仿真模型失败: {exc}")
        gui = getattr(self, 'gui', None)
        if gui is not None and getattr(gui, '_popen', None) is not None:
            try:
                if gui.is_alive():
                    gui.terminate()
                gui.join(timeout=5)
            except Exception as exc:
                print(f"关闭GUI失败: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description='DreamForge闭环对抗性场景仿真'
    )
    default_config = Path(__file__).resolve().parent / 'config-attack.yaml'
    parser.add_argument('--config', default=str(default_config))
    parser.add_argument('--scene_name', default=None)
    parser.add_argument('--sumo_cfg', default=None)
    parser.add_argument('--sumo_rou', default=None)
    parser.add_argument('--attacker_id', type=int, default=None)
    parser.add_argument('--attack_type', default=None)
    parser.add_argument('--attack_time', type=float, default=None)
    args = parser.parse_args()

    sim_manager = SimulationManager(
        config_path=args.config,
        scene_name=args.scene_name,
        sumo_cfg=args.sumo_cfg,
        sumo_rou=args.sumo_rou,
    )
    sim_manager.run_simulation()


if __name__ == '__main__':
    main()
