"""Lightweight selector settings shared by the simulator and batch runner."""

import copy
import json
from pathlib import Path


def add_selector_arguments(parser):
    parser.add_argument("--selector-mode", choices=("vlm", "llm"), default=None)
    parser.add_argument("--selector-model", default=None)
    parser.add_argument("--selector-api-base", default=None)
    parser.add_argument("--selector-api-key-env", default=None)


def apply_selector_overrides(config, args):
    result = copy.deepcopy(config)
    overrides = {
        key: getattr(args, "selector_" + key, None)
        for key in ("mode", "model", "api_base", "api_key_env")
    }
    overrides = {key: value for key, value in overrides.items() if value is not None}
    configured_mode = (result.get("selector") or {}).get("mode", "vlm")
    if overrides.get("mode", configured_mode) != configured_mode and "model" not in overrides:
        raise ValueError("When changing selector mode, also specify --selector-model")
    if overrides:
        result.setdefault("selector", {}).update(overrides)
    return result


def resolve_selector_config(config):
    selector = dict(config.get("selector") or {})
    mode = selector.pop("mode", "vlm")
    if mode not in ("vlm", "llm"):
        raise ValueError("selector.mode must be vlm or llm")
    if mode == "llm" and not selector.get("model"):
        raise ValueError("Text-only selection requires selector.model")
    trajectory = dict(config.get("llm") or {})
    service = dict(trajectory)
    # A different endpoint must never receive the trajectory provider's key.
    own_credentials = "api_key_env" in selector or "api_key" in selector
    changed_endpoint = (
        "api_base" in selector
        and selector["api_base"].rstrip("/")
        != trajectory.get("api_base", "https://api.deepseek.com/v1").rstrip("/")
    )
    if changed_endpoint and not own_credentials:
        raise ValueError("A different selector API requires selector.api_key_env or api_key")
    if own_credentials:
        service.pop("api_key", None)
        service.pop("api_key_env", None)
        service["api_key_env"] = selector.get("api_key_env", "")
    service.update(selector)
    service["model"] = selector.get("model", "qwen-vl-max")
    return mode, service


def selector_metadata(config):
    mode, service = resolve_selector_config(config)
    trajectory = config.get("llm") or {}
    return {
        "selector_mode": mode,
        "selector_model": service["model"],
        "selector_api_base": service.get("api_base", "https://api.deepseek.com/v1").rstrip("/"),
        "selector_prompt_version": "text-v1" if mode == "llm" else "legacy-visual-v1",
        "selector_temperature": 0.5,
        "scene_description_version": "state-only-v2",
        "trajectory_model": trajectory.get("model", "deepseek-chat"),
        "trajectory_api_base": trajectory.get("api_base", "https://api.deepseek.com/v1").rstrip("/"),
        "use_agent_path": config.get("simulation", {}).get("use_agent_path", True),
    }


def ensure_experiment_manifest(result_dir, metadata, resume=False):
    path = Path(result_dir) / "experiment_config.json"
    if resume:
        if not path.is_file():
            raise ValueError("Cannot verify this batch's model settings: experiment_config.json is missing; start a new batch")
        if json.loads(path.read_text(encoding="utf-8")) != metadata:
            raise ValueError("Resume configuration differs from the saved experiment; use the original settings or start a new batch")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as output:
            json.dump(metadata, output, ensure_ascii=False, indent=2)
