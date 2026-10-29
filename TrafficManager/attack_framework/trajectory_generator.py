"""Two-stage LLM pipeline for executable adversarial trajectories."""

from dataclasses import dataclass
import json
import math
from typing import Any, Dict, List, Mapping

from .code_executor import GeneratedAttackCodeExecutor
from .llm_client import LLMClient
from .models import AttackType
from .prompts import build_goal_function_prompt, build_path_planning_prompt


@dataclass(frozen=True)
class GeneratedAttackTrajectory:
    goal_code: str
    path_code: str
    goal_state: Dict[str, Any]
    points: List[Dict[str, Any]]


class TwoStageTrajectoryGenerator:
    """Generates an attack goal first, then an executable trajectory."""

    def __init__(
        self,
        llm_client: LLMClient,
        code_executor: GeneratedAttackCodeExecutor,
    ):
        self.llm_client = llm_client
        self.code_executor = code_executor

    def generate(
        self,
        attack_type: AttackType,
        ego_state: Mapping[str, Any],
        attacker_state: Mapping[str, Any],
        sumo_topology: Mapping[str, Any],
    ) -> GeneratedAttackTrajectory:
        print("LLM stage 1: generating attack goal", flush=True)
        goal_code = self.llm_client.create_code(
            build_goal_function_prompt(attack_type, sumo_topology),
            timeout=120,
        )
        print("\n" + "=" * 22 + " LLM STAGE 1 CODE " + "=" * 22)
        print(goal_code, flush=True)
        print("=" * 68 + "\n")
        print("\n" + "=" * 20 + " LLM阶段1返回代码 " + "=" * 20, flush=True)
        print(goal_code, flush=True)
        print("=" * 60 + "\n", flush=True)
        try:
            target_position = self.code_executor.calculate_goal(
                goal_code,
                ego_state,
                attacker_state,
                sumo_topology,
            )
        except Exception as error:
            print(
                f"LLM stage 1 execution failed, using constrained fallback: {error}",
                flush=True,
            )
            heading = float(
                attacker_state["heading"]
                if attack_type == AttackType.BRAKE_CHECK
                else ego_state["heading"]
            )
            if attack_type == AttackType.BRAKE_CHECK:
                stop_distance = float(attacker_state["speed"]) ** 2 / 14.0
                target_x = float(attacker_state["x"]) + stop_distance * math.cos(heading)
                target_y = float(attacker_state["y"]) + stop_distance * math.sin(heading)
                target_speed = 0.0
            else:
                forward_distance = (
                    2.0 if attack_type == AttackType.CUT_IN
                    else max(3.0, float(ego_state["speed"]) * 2.0)
                )
                target_x = float(ego_state["x"]) + forward_distance * math.cos(heading)
                target_y = float(ego_state["y"]) + forward_distance * math.sin(heading)
                target_speed = float(attacker_state["speed"])
            target_position = {
                "target_x": target_x,
                "target_y": target_y,
                "target_v": target_speed,
            }
        print("LLM stage 1: SUCCESS", flush=True)
        ego_heading = float(ego_state["heading"])
        ego_cos = math.cos(ego_heading)
        ego_sin = math.sin(ego_heading)
        target_dx = float(target_position["target_x"]) - float(ego_state["x"])
        target_dy = float(target_position["target_y"]) - float(ego_state["y"])
        target_forward = target_dx * ego_cos + target_dy * ego_sin
        target_lateral = -target_dx * ego_sin + target_dy * ego_cos
        ego_lane = sumo_topology.get("lanes", {}).get(
            sumo_topology.get("ego_lane_id"),
            {},
        )
        ego_lane_width = float(ego_lane.get("width", 3.5))
        if attack_type == AttackType.OPPOSITE_DIRECTION_INTRUSION:
            minimum_forward = max(3.0, float(ego_state["speed"]) * 0.8)
            maximum_forward = max(40.0, float(ego_state["speed"]) * 3.0)
            invalid_intercept = (
                target_forward < minimum_forward
                or target_forward > maximum_forward
                or abs(target_lateral) > max(1.5, ego_lane_width * 0.5)
            )
            if invalid_intercept:
                intercept_distance = max(
                    minimum_forward,
                    float(ego_state["speed"]) * 2.0,
                )
                target_position["target_x"] = (
                    float(ego_state["x"]) + intercept_distance * ego_cos
                )
                target_position["target_y"] = (
                    float(ego_state["y"]) + intercept_distance * ego_sin
                )
                print(
                    "LLM stage 1 intercept corrected: target was not "
                    "inside the ego lane ahead",
                    flush=True,
                )
        elif attack_type == AttackType.CUT_IN:
            invalid_cut_in_target = (
                target_forward < -1.0
                or target_forward > 8.0
                or abs(target_lateral) > max(1.5, ego_lane_width * 0.5)
            )
            if invalid_cut_in_target:
                target_position["target_x"] = float(ego_state["x"]) + 2.0 * ego_cos
                target_position["target_y"] = float(ego_state["y"]) + 2.0 * ego_sin
                print(
                    "LLM stage 1 cut-in target corrected to ego lane",
                    flush=True,
                )
        if attack_type == AttackType.BRAKE_CHECK:
            default_target_speed = 0.0
        elif attack_type == AttackType.CUT_IN:
            speed_limit = float(ego_lane.get("speed_limit", 13.89))
            relative_x = attacker_state["x"] - ego_state["x"]
            relative_y = attacker_state["y"] - ego_state["y"]
            longitudinal_gap = (
                relative_x * math.cos(ego_state["heading"])
                + relative_y * math.sin(ego_state["heading"])
            )
            default_target_speed = max(
                0.0,
                min(
                    speed_limit,
                    ego_state["speed"] + (2.0 - longitudinal_gap) / 2.1,
                ),
            )
        else:
            attacker_lane = sumo_topology.get("lanes", {}).get(
                attacker_state.get("lane_id"),
                {},
            )
            speed_limit = float(attacker_lane.get("speed_limit", 13.89))
            default_target_speed = min(
                speed_limit,
                max(
                    attacker_state["speed"],
                    min(11.0, ego_state["speed"]),
                ),
            )
        if attack_type in (
            AttackType.BRAKE_CHECK,
            AttackType.CUT_IN,
            AttackType.OPPOSITE_DIRECTION_INTRUSION,
        ):
            goal_speed = default_target_speed
        else:
            goal_speed = target_position.get("target_v", default_target_speed)
        goal_state = {
            "x": target_position["target_x"],
            "y": target_position["target_y"],
            "speed": goal_speed,
            "heading": target_position.get(
                "target_heading", attacker_state["heading"]
            ),
        }
        print(f"LLM stage 1 goal_state: {goal_state}", flush=True)
        print(f"LLM阶段1执行结果 goal_state={goal_state}", flush=True)

        print("LLM stage 2: generating executable trajectory", flush=True)
        path_code = self.llm_client.create_code(
            build_path_planning_prompt(
                attacker_state,
                goal_state,
                sumo_topology,
                attack_type,
            ),
            timeout=120,
        )
        print("\n" + "=" * 22 + " LLM STAGE 2 CODE " + "=" * 22)
        print(path_code, flush=True)
        print("=" * 68 + "\n")
        print("\n" + "=" * 20 + " LLM阶段2返回代码 " + "=" * 20, flush=True)
        print(path_code, flush=True)
        print("=" * 60 + "\n", flush=True)
        try:
            points = self.code_executor.plan_trajectory(
                path_code,
                attacker_state,
                goal_state,
                sumo_topology,
                attack_type.value,
            )
        except Exception as error:
            print(
                f"LLM stage 2 execution failed, using constrained fallback: {error}",
                flush=True,
            )
            points = self.code_executor.build_fallback_trajectory(
                attacker_state,
                goal_state,
                attack_type.value,
            )
        print(f"LLM stage 2: SUCCESS points={len(points)}", flush=True)
        print("LLM trajectory points:", flush=True)
        print(json.dumps(points, ensure_ascii=False, indent=2), flush=True)
        print("LLM阶段2轨迹点:", flush=True)
        for index, point in enumerate(points, 1):
            print(f"  [{index:02d}] {point}", flush=True)
        return GeneratedAttackTrajectory(
            goal_code=goal_code,
            path_code=path_code,
            goal_state=goal_state,
            points=points,
        )
