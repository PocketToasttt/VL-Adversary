"""Execution boundary for the two-stage LLM-generated attack code."""

import math
from numbers import Real
from typing import Any, Dict, List, Mapping, Optional

import numpy as np


class _GeneratedState(dict):
    _ALIASES = {
        "v": "speed",
        "target_v": "speed",
        "yaw": "heading",
        "target_heading": "heading",
        "target_x": "x",
        "start_x": "x",
        "target_y": "y",
        "start_y": "y",
    }

    def __missing__(self, key):
        canonical_key = self._ALIASES.get(key)
        if canonical_key is None or canonical_key not in self:
            raise KeyError(key)
        return dict.__getitem__(self, canonical_key)

    def __iter__(self):
        return iter(self.values())


class GeneratedAttackCodeExecutor:
    """Executes generated goal and trajectory functions with explicit inputs."""

    @staticmethod
    def _normalize_state(state: Mapping[str, Any]) -> Dict[str, Any]:
        normalized = dict(state)
        # 同时提供 speed 和 v 两种写法，防止 LLM 任意选一种
        if "speed" in normalized and "v" not in normalized:
            normalized["v"] = normalized["speed"]
        if "v" in normalized and "speed" not in normalized:
            normalized["speed"] = normalized["v"]
        # 同样处理其他常见别名
        if "velocity" in normalized and "speed" not in normalized:
            normalized["speed"] = normalized["velocity"]
        return normalized



    @staticmethod
    def calculate_goal(
        generated_code: str,
        ego_state: Mapping[str, Any],
        attacker_state: Mapping[str, Any],
        sumo_topology: Mapping[str, Any],
    ) -> Dict[str, Any]:
        normalized_ego = GeneratedAttackCodeExecutor._normalize_state(ego_state)
        normalized_attacker = GeneratedAttackCodeExecutor._normalize_state(attacker_state)
        topology = dict(sumo_topology)

        # 预注入常用派生变量，防止 LLM 生成代码直接引用时报 NameError
        execution_scope = {
            "math": math,
            # ego 相关
            "ego_x": normalized_ego.get("x"),
            "ego_y": normalized_ego.get("y"),
            "ego_heading": normalized_ego.get("heading"),
            "ego_speed": normalized_ego.get("speed"),
            "ego_lane_id": normalized_ego.get("lane_id"),
            # attacker 相关
            "attacker_x": normalized_attacker.get("x"),
            "attacker_y": normalized_attacker.get("y"),
            "attacker_heading": normalized_attacker.get("heading"),
            "attacker_speed": normalized_attacker.get("speed"),
            "attacker_lane_id": normalized_attacker.get("lane_id"),  # 修复当前 NameError
        }

        exec(generated_code, execution_scope)
        goal_function = execution_scope.get("calculate_goal_position")
        if not callable(goal_function):
            raise ValueError("Generated code does not define calculate_goal_position")

        result = goal_function(normalized_ego, normalized_attacker, topology)
        if not isinstance(result, dict):
            raise TypeError("calculate_goal_position must return a dict")
        if result.get("target_x") is None or result.get("target_y") is None:
            raise ValueError("Goal result must contain target_x and target_y")
        for key in ("target_x", "target_y"):
            value = result[key]
            if not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"Goal field {key} must be a finite number")
        return result

    @staticmethod
    def plan_trajectory(
        generated_code: str,
        start_state: Mapping[str, Any],
        end_state: Mapping[str, Any],
        sumo_topology: Mapping[str, Any],
        attack_type: str = "",
        *,
        cut_in_ego_state: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        execution_scope = {"np": np, "math": math}
        exec(generated_code, execution_scope)
        trajectory_function = execution_scope.get("plan_attack_trajectory")
        if not callable(trajectory_function):
            raise ValueError("Generated code does not define plan_attack_trajectory")

        trajectory = trajectory_function(
            GeneratedAttackCodeExecutor._normalize_state(start_state),
            GeneratedAttackCodeExecutor._normalize_state(end_state),
            dict(sumo_topology),
        )
        if not isinstance(trajectory, list):
            raise TypeError("plan_attack_trajectory must return a list")
        if not 10 <= len(trajectory) <= 100:
            raise ValueError(
                "plan_attack_trajectory must return between 10 and 100 points, "
                f"got {len(trajectory)}"
            )

        required_fields = ("x", "y", "v", "heading")
        for index, point in enumerate(trajectory):
            if not isinstance(point, dict):
                raise TypeError(f"Trajectory point {index} must be a dict")
            for field in required_fields:
                value = point.get(field)
                if not isinstance(value, Real) or not math.isfinite(value):
                    raise ValueError(
                        f"Trajectory point {index} field {field} must be a finite number"
                    )
        if cut_in_ego_state is not None:
            ego_lane = sumo_topology.get("lanes", {}).get(
                sumo_topology.get("ego_lane_id"), {},
            )
            return GeneratedAttackCodeExecutor.replan_cut_in(
                cut_in_ego_state,
                start_state,
                speed_limit=float(ego_lane.get("speed_limit", 13.89)),
                goal_state=end_state,
                speed_profile=[float(point["v"]) for point in trajectory],
            )
        return GeneratedAttackCodeExecutor._enforce_vehicle_dynamics(
            trajectory,
            start_state,
            end_state,
            attack_type,
        )

    @staticmethod
    def build_fallback_trajectory(
        start_state: Mapping[str, Any],
        end_state: Mapping[str, Any],
        attack_type: str,
        point_count: int = 30,
    ) -> List[Dict[str, Any]]:
        raw_trajectory = []
        for index in range(point_count):
            progress = index / max(1, point_count - 1)
            smooth = progress * progress * (3.0 - 2.0 * progress)
            raw_trajectory.append({
                "x": float(start_state["x"]) + (
                    float(end_state["x"]) - float(start_state["x"])
                ) * smooth,
                "y": float(start_state["y"]) + (
                    float(end_state["y"]) - float(start_state["y"])
                ) * smooth,
                "v": float(start_state["speed"]) + (
                    float(end_state["speed"]) - float(start_state["speed"])
                ) * smooth,
                "heading": float(start_state["heading"]),
            })
        return GeneratedAttackCodeExecutor._enforce_vehicle_dynamics(
            raw_trajectory,
            start_state,
            end_state,
            attack_type,
        )

    @staticmethod
    def replan_cut_in(
        ego_state: Mapping[str, Any],
        attacker_state: Mapping[str, Any],
        point_count: int = 30,
        dt: float = 0.1,
        speed_limit: float = 13.89,
        *,
        goal_state: Optional[Mapping[str, Any]] = None,
        speed_profile: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        ego_heading = float(ego_state["heading"])
        desired_longitudinal_gap = 2.0
        if goal_state is None:
            target_x = float(ego_state["x"]) + desired_longitudinal_gap * math.cos(ego_heading)
            target_y = float(ego_state["y"]) + desired_longitudinal_gap * math.sin(ego_heading)
        else:
            target_x, target_y = float(goal_state["x"]), float(goal_state["y"])
        attacker_x = float(attacker_state["x"])
        attacker_y = float(attacker_state["y"])
        attacker_speed = max(0.0, float(attacker_state["speed"]))
        attacker_heading = float(attacker_state["heading"])
        cos_heading = math.cos(ego_heading)
        sin_heading = math.sin(ego_heading)
        relative_x = attacker_x - float(ego_state["x"])
        relative_y = attacker_y - float(ego_state["y"])
        current_longitudinal_gap = (
            relative_x * cos_heading + relative_y * sin_heading
        )
        merge_duration = 2.1
        speed_limit = max(0.0, float(speed_limit))
        target_speed = float(np.clip(
            float(ego_state["speed"])
            + (
                desired_longitudinal_gap - current_longitudinal_gap
            ) / merge_duration,
            0.0,
            speed_limit,
        ))
        target_dx = target_x - attacker_x
        target_dy = target_y - attacker_y
        target_lateral = float(np.clip(
            -target_dx * sin_heading + target_dy * cos_heading,
            -3.6,
            3.6,
        ))
        heading_delta = math.atan2(
            math.sin(attacker_heading - ego_heading),
            math.cos(attacker_heading - ego_heading),
        )
        initial_lateral_speed = float(np.clip(
            attacker_speed * math.sin(heading_delta),
            -2.6,
            2.6,
        ))
        lateral_duration = merge_duration
        previous_speed = attacker_speed
        longitudinal = 0.0
        trajectory = []
        for index in range(point_count):
            elapsed = index * dt
            progress = min(1.0, elapsed / lateral_duration)
            h10 = progress ** 3 - 2.0 * progress ** 2 + progress
            h01 = -2.0 * progress ** 3 + 3.0 * progress ** 2
            lateral = (
                h10 * lateral_duration * initial_lateral_speed
                + h01 * target_lateral
            )
            if progress < 1.0:
                h10_derivative = (
                    3.0 * progress ** 2 - 4.0 * progress + 1.0
                ) / lateral_duration
                h01_derivative = (
                    -6.0 * progress ** 2 + 6.0 * progress
                ) / lateral_duration
                lateral_speed = (
                    h10_derivative
                    * lateral_duration
                    * initial_lateral_speed
                    + h01_derivative * target_lateral
                )
            else:
                lateral_speed = 0.0

            if index == 0:
                speed = attacker_speed
            else:
                # Keep the full merge horizon even if the model proposes fewer samples.
                reference_speed = target_speed
                if speed_profile:
                    reference_speed = float(np.clip(
                        speed_profile[min(index, len(speed_profile) - 1)],
                        0.0,
                        speed_limit,
                    ))
                speed = float(np.clip(
                    reference_speed,
                    previous_speed - 5.0 * dt,
                    previous_speed + 3.0 * dt,
                ))
                longitudinal += 0.5 * (previous_speed + speed) * dt
            heading = ego_heading + math.atan2(
                lateral_speed,
                max(speed, 0.5),
            )
            trajectory.append({
                "x": attacker_x + longitudinal * cos_heading - lateral * sin_heading,
                "y": attacker_y + longitudinal * sin_heading + lateral * cos_heading,
                "v": speed,
                "heading": heading,
            })
            previous_speed = speed
        return trajectory

    @staticmethod
    def _enforce_vehicle_dynamics(
        trajectory: List[Dict[str, Any]],
        start_state: Mapping[str, Any],
        end_state: Mapping[str, Any],
        attack_type: str,
        dt: float = 0.1,
    ) -> List[Dict[str, Any]]:
        start_x = float(start_state["x"])
        start_y = float(start_state["y"])
        start_speed = max(0.0, float(start_state["speed"]))
        start_heading = float(start_state["heading"])
        cos_heading = math.cos(start_heading)
        sin_heading = math.sin(start_heading)

        if attack_type == "brake_check":
            return GeneratedAttackCodeExecutor._plan_emergency_brake(
                trajectory,
                start_state,
                dt,
            )

        if attack_type in ("cut_in", "opposite_direction_intrusion"):
            target_end = end_state
        else:
            target_end = trajectory[-1]
        end_dx = float(target_end["x"]) - start_x
        end_dy = float(target_end["y"]) - start_y
        target_lateral = -end_dx * sin_heading + end_dy * cos_heading
        if attack_type == "brake_check":
            target_lateral = 0.0
        elif attack_type == "cut_in":
            target_lateral = float(np.clip(target_lateral, -3.6, 3.6))
        else:
            target_lateral = float(np.clip(target_lateral, -4.0, 4.0))

        duration = max(dt, (len(trajectory) - 1) * dt)
        opposite_target_speed = None
        if attack_type == "opposite_direction_intrusion":
            opposite_target_speed = max(
                start_speed,
                min(11.0, float(end_state.get("speed", start_speed))),
            )
        if attack_type == "cut_in":
            lateral_duration = min(duration, 2.1)
        elif attack_type == "opposite_direction_intrusion":
            target_longitudinal = max(
                0.0,
                end_dx * cos_heading + end_dy * sin_heading,
            )
            time_to_conflict = target_longitudinal / max(
                opposite_target_speed,
                0.5,
            )
            lateral_duration = float(np.clip(
                time_to_conflict,
                1.8,
                min(duration, 2.5),
            ))
        else:
            lateral_duration = duration
        previous_speed = start_speed
        longitudinal = 0.0
        result = []
        for index, raw_point in enumerate(trajectory):
            elapsed = index * dt
            progress = min(1.0, elapsed / lateral_duration)
            smooth = (
                10.0 * progress ** 3
                - 15.0 * progress ** 4
                + 6.0 * progress ** 5
            )
            smooth_derivative = (
                30.0 * progress ** 2
                - 60.0 * progress ** 3
                + 30.0 * progress ** 4
            ) / lateral_duration
            lateral = target_lateral * smooth
            lateral_speed = target_lateral * smooth_derivative

            if index == 0:
                speed = start_speed
            else:
                if attack_type == "brake_check":
                    desired_speed = 0.0
                elif attack_type == "opposite_direction_intrusion":
                    desired_speed = opposite_target_speed
                else:
                    desired_speed = max(0.0, float(raw_point["v"]))
                if attack_type == "opposite_direction_intrusion":
                    speed = float(np.clip(
                        desired_speed,
                        previous_speed,
                        previous_speed + 1.5 * dt,
                    ))
                else:
                    speed = float(np.clip(
                        desired_speed,
                        previous_speed - 5.0 * dt,
                        previous_speed + 3.0 * dt,
                    ))
                longitudinal += 0.5 * (previous_speed + speed) * dt

            heading = start_heading + math.atan2(
                lateral_speed,
                max(speed, 0.5),
            )
            x = start_x + longitudinal * cos_heading - lateral * sin_heading
            y = start_y + longitudinal * sin_heading + lateral * cos_heading
            result.append({
                "x": x,
                "y": y,
                "v": speed,
                "heading": heading,
            })
            previous_speed = speed

        return result

    @staticmethod
    def _plan_emergency_brake(
        trajectory: List[Dict[str, Any]],
        start_state: Mapping[str, Any],
        dt: float,
    ) -> List[Dict[str, Any]]:
        start_x = float(start_state["x"])
        start_y = float(start_state["y"])
        start_speed = max(0.0, float(start_state["speed"]))
        heading = float(start_state["heading"])
        cos_heading = math.cos(heading)
        sin_heading = math.sin(heading)

        raw_end = trajectory[-1]
        requested_distance = (
            (float(raw_end["x"]) - start_x) * cos_heading
            + (float(raw_end["y"]) - start_y) * sin_heading
        )
        nominal_deceleration = 7.0
        if requested_distance > 0.1 and start_speed > 0.0:
            required_deceleration = (
                start_speed ** 2 / (2.0 * requested_distance)
            )
        else:
            required_deceleration = nominal_deceleration
        deceleration = float(np.clip(required_deceleration, 5.0, 8.0))
        max_jerk = 8.0
        current_speed = start_speed
        current_deceleration = 0.0
        longitudinal = 0.0
        result = []
        for index in range(len(trajectory)):
            if index > 0 and current_speed > 0.0:
                next_deceleration = min(
                    deceleration,
                    current_deceleration + max_jerk * dt,
                )
                next_speed = max(
                    0.0,
                    current_speed
                    - 0.5 * (current_deceleration + next_deceleration) * dt,
                )
                longitudinal += 0.5 * (current_speed + next_speed) * dt
                current_speed = next_speed
                current_deceleration = next_deceleration
            result.append({
                "x": start_x + longitudinal * cos_heading,
                "y": start_y + longitudinal * sin_heading,
                "v": current_speed,
                "heading": heading,
            })
        return result
