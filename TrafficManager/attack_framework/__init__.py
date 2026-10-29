"""Adversarial closed-loop scenario generation components."""

from .code_executor import GeneratedAttackCodeExecutor
from .collision import (
    predict_constant_velocity_trajectory,
    sat_collision,
    vehicle_collision,
    vehicle_corners,
)
from .config import AttackRuntimeConfig, LLMServiceConfig
from .llm_client import LLMClient
from .models import AttackDecision, AttackType, VehicleState
from .prompts import build_goal_function_prompt, build_path_planning_prompt
from .scene_analysis import SceneAnalysis, analyze_scene
from .trajectory_generator import GeneratedAttackTrajectory, TwoStageTrajectoryGenerator
from .trajectory_injector import AttackTrajectoryInjector
from .vlm_selector import VLMTargetSelector

__all__ = [
    "AttackDecision",
    "AttackRuntimeConfig",
    "AttackTrajectoryInjector",
    "AttackType",
    "GeneratedAttackCodeExecutor",
    "GeneratedAttackTrajectory",
    "LLMClient",
    "LLMServiceConfig",
    "SceneAnalysis",
    "TwoStageTrajectoryGenerator",
    "VLMTargetSelector",
    "VehicleState",
    "build_goal_function_prompt",
    "build_path_planning_prompt",
    "analyze_scene",
    "predict_constant_velocity_trajectory",
    "sat_collision",
    "vehicle_collision",
    "vehicle_corners",
]

