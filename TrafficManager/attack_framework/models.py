"""Domain models shared by adversarial decision and execution stages."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping


class AttackType(str, Enum):
    BRAKE_CHECK = "brake_check"
    CUT_IN = "cut_in"
    OPPOSITE_DIRECTION_INTRUSION = "opposite_direction_intrusion"


@dataclass(frozen=True)
class AttackDecision:
    target_vehicle_id: str
    attack_type: AttackType
    behavior_params: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "AttackDecision":
        target_vehicle_id = str(values.get("target_vehicle_id", "")).strip()
        if not target_vehicle_id:
            raise ValueError("VLM decision is missing target_vehicle_id")

        raw_attack_type = str(values.get("attack_type", "")).strip().lower()
        try:
            attack_type = AttackType(raw_attack_type)
        except ValueError as exc:
            supported = ", ".join(item.value for item in AttackType)
            raise ValueError(
                f"Unsupported attack_type={raw_attack_type!r}; expected one of: {supported}"
            ) from exc

        behavior_params = values.get("behavior_params") or {}
        if not isinstance(behavior_params, dict):
            raise ValueError("behavior_params must be a JSON object")

        return cls(
            target_vehicle_id=target_vehicle_id,
            attack_type=attack_type,
            behavior_params=dict(behavior_params),
            reasoning=str(values.get("reasoning", "")),
        )


@dataclass(frozen=True)
class VehicleState:
    x: float
    y: float
    speed: float
    heading: float
    lane_id: str = "unknown"

    @classmethod
    def from_limsim(cls, vehicle: Mapping[str, Any]) -> "VehicleState":
        try:
            lane_id = vehicle.get("laneID")
            if not lane_id:
                lane_history = vehicle.get("laneIDQ")
                if lane_history:
                    lane_id = lane_history[-1]
            return cls(
                x=float(vehicle["xQ"][-1]),
                y=float(vehicle["yQ"][-1]),
                speed=float(vehicle["speedQ"][-1]),
                heading=float(vehicle["yawQ"][-1]),
                lane_id=str(lane_id or "unknown"),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("Incomplete LimSim vehicle state") from exc

    def to_prompt_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "speed": self.speed,
            "heading": self.heading,
            "lane_id": self.lane_id,
        }
