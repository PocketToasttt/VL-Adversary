"""Typed configuration for adversarial scenario generation."""

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class AttackRuntimeConfig:
    enabled: bool = False
    mode: str = "auto"
    target_vehicle_id: str = ""
    attack_type: str = ""
    trigger_time: float = 5.0
    check_safety: bool = True
    llm_replanning: bool = True

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]]) -> "AttackRuntimeConfig":
        config = values or {}
        return cls(
            enabled=bool(config.get("enable", False)),
            mode=str(config.get("mode", "auto")),
            target_vehicle_id=str(config.get("target_vehicle_id", "")),
            attack_type=str(config.get("attack_type", "")),
            trigger_time=float(config.get("trigger_time", 5.0)),
            check_safety=bool(config.get("check_safety", True)),
            llm_replanning=bool(
                config.get("llm_replanning", config.get("llm_cut_in_replanning", True))
            ),
        )


@dataclass(frozen=True)
class LLMServiceConfig:
    enabled: bool = False
    api_key: Optional[str] = None
    api_base: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-chat"

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]]) -> "LLMServiceConfig":
        config = values or {}
        key_env = str(config.get("api_key_env", "DEEPSEEK_API_KEY"))
        api_key = (os.environ.get(key_env) if key_env else None) or config.get("api_key")
        enabled = bool(config.get("enable", False))
        if enabled and not api_key:
            enabled = False

        return cls(
            enabled=enabled,
            api_key=api_key,
            api_base=str(config.get("api_base", "https://api.deepseek.com/v1")).rstrip("/"),
            model=str(config.get("model", "deepseek-chat")),
        )

