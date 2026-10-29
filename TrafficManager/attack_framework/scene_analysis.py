"""Scene semantics used by the VLM pressure-test expert."""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from .models import VehicleState


@dataclass(frozen=True)
class SceneAnalysis:
    description: str
    candidates: List[Dict[str, Any]]
    skipped: List[Dict[str, Any]]


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def relative_position(
    car_x: float,
    car_y: float,
    ego_x: float,
    ego_y: float,
    ego_heading: float,
) -> str:
    angle_to_car = math.atan2(car_y - ego_y, car_x - ego_x)
    relative_angle = math.degrees(normalize_angle(angle_to_car - ego_heading))
    if -30 <= relative_angle <= 30:
        return "Front"
    if 30 < relative_angle < 150:
        return "Left"
    if -150 < relative_angle < -30:
        return "Right"
    return "Behind"


def classify_direction(heading_difference: float):
    absolute_difference = abs(heading_difference)
    if absolute_difference <= 30:
        return "同向", ["cut_in", "brake_check"], True
    if absolute_difference >= 150:
        return "对向", ["opposite_direction_intrusion"], True
    if 60 <= absolute_difference <= 120:
        return "横向(交叉口)", [], False
    return "斜向", [], False


def lane_relation(ego_lane_id: str, car_lane_id: str) -> str:
    if not ego_lane_id or not car_lane_id:
        return "unknown"
    if ego_lane_id == "unknown" or car_lane_id == "unknown":
        return "unknown"
    if ego_lane_id == car_lane_id:
        return "same_lane"
    try:
        ego_edge, ego_index = ego_lane_id.rsplit("_", 1)
        car_edge, car_index = car_lane_id.rsplit("_", 1)
        if ego_edge == car_edge and abs(int(ego_index) - int(car_index)) == 1:
            return "adjacent_lane"
        if ego_edge == car_edge:
            return "same_edge_non_adjacent"
    except (ValueError, AttributeError):
        pass
    return "different_edge"


def analyze_scene(
    ego_info: Mapping[str, Any],
    background_cars: Sequence[Mapping[str, Any]],
    sumo_topology: Mapping[str, Any],
    maximum_distance: float = 60.0,
) -> SceneAnalysis:
    ego = VehicleState.from_limsim(ego_info)
    lines = [
        f"主车 (Ego) 状态: 坐标 ({ego.x:.1f}, {ego.y:.1f}), "
        f"速度 {ego.speed:.1f} m/s, 航向角 {math.degrees(ego.heading):.1f}度.",
        f"主车当前车道ID: {sumo_topology.get('ego_lane_id', 'unknown')}",
        f"主车当前道路ID: {sumo_topology.get('ego_edge_id', 'unknown')}",
        "",
    ]

    lanes = sumo_topology.get("lanes") or {}
    if lanes:
        lines.append(f"周围车道信息 ({len(lanes)} 条):")
        for lane_id, lane_info in list(lanes.items())[:10]:
            lane_line = (
                f"  - Lane {lane_id}: 长度={lane_info.get('length', 0):.1f}m, "
                f"宽度={lane_info.get('width', 0):.1f}m"
            )
            if lane_info.get("is_ego_lane"):
                lane_line += " [主车所在车道]"
            lines.append(lane_line)

    lines.extend(("", "周围背景车信息:"))
    candidates = []
    skipped = []
    for car_info in background_cars:
        try:
            car = VehicleState.from_limsim(car_info)
        except ValueError:
            continue

        distance = math.hypot(car.x - ego.x, car.y - ego.y)
        position = relative_position(car.x, car.y, ego.x, ego.y, ego.heading)
        heading_difference = abs(
            math.degrees(normalize_angle(car.heading - ego.heading))
        )
        direction, suitable_attacks, valid = classify_direction(heading_difference)
        relation = lane_relation(ego.lane_id, car.lane_id)
        if heading_difference <= 30:
            if relation == "same_lane":
                suitable_attacks = ["brake_check"]
            elif relation == "adjacent_lane":
                suitable_attacks = ["cut_in"]
        vehicle_id = str(car_info.get("id", ""))
        if not valid:
            skipped.append(
                {
                    "id": vehicle_id,
                    "reason": (
                        f"{direction}(航向差{heading_difference:.1f}°)，"
                        "不具备持续交互条件"
                    ),
                }
            )

        candidate = {
            "id": vehicle_id,
            "x": car.x,
            "y": car.y,
            "speed": car.speed,
            "heading": car.heading,
            "heading_deg": math.degrees(car.heading),
            "distance_to_ego": distance,
            "relative_position": position,
            "direction_type": direction,
            "suitable_attacks": suitable_attacks,
            "lane_id": car.lane_id,
            "lane_relation": relation,
            "heading_diff_deg": heading_difference,
        }
        candidates.append(candidate)
        lines.append(
            f"- SUMO车道关系: vehicle={vehicle_id}, "
            f"ego_lane={ego.lane_id}, vehicle_lane={car.lane_id}, "
            f"relation={relation}"
        )
        lines.append(
            f"- ID: {vehicle_id}, [{direction}], 车道: {car.lane_id}, "
            f"位置: {position}, 距离: {distance:.1f}m, 速度: {car.speed:.1f} m/s, "
            f"航向差: {heading_difference:.1f}°"
        )

    return SceneAnalysis("\n".join(lines), candidates, skipped)
