"""Shared, modality-controlled protocol for the SUMO selection pilot."""
import hashlib
import json
from pathlib import Path

VARIANTS = {
    "vlm_image": {"mode": "vlm", "model": "qwen-vl-max"},
    "vlm_text": {"mode": "llm", "model": "qwen-vl-max"},
    "qwen_plus": {"mode": "llm", "model": "qwen-plus"},
}
PROMPT_VERSION = "sumo-shared-input-v1"


def shared_rules(text_rules):
    """Same instructions in all arms; only attachment and model differ."""
    return text_rules.replace(
        "Use only the supplied real-time structured scene description.",
        "Use the supplied real-time structured scene description and the BEV image if attached. "
        "If no image is attached, use only the supplied structured description.",
    ).replace(
        "请仅根据当前实时结构化场景描述",
        "请根据当前实时结构化场景描述，以及实际附带的BEV图像（如有）",
    ).replace(
        "基于结构化场景描述给出比较和选择原因",
        "基于实际提供的场景信息给出比较和选择原因",
    )


def select_scene_ids(index, limit):
    if limit < 1:
        raise ValueError("per-type limit must be positive")
    ids = index["statistics"]["scene_ids"]
    selected = []
    for attack in ("brake_check", "cut_in", "opposite_direction_intrusion"):
        group = sorted(s for s in ids if s.startswith(attack + "_"))
        if len(group) < limit:
            raise ValueError("Insufficient scenes for " + attack)
        selected.extend(group[:limit])
    return selected


def planner_cache_key(request):
    # Endpoint/payload only. Credentials never enter a cache file or hash.
    content = {"url": request["url"], "payload": request["payload"]}
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def cached_planner_completion(request, timeout, original, cache_dir, audit):
    key = planner_cache_key(request)
    path = Path(cache_dir) / (key + ".json")
    audit["cache_key"] = key
    if path.is_file():
        audit["cache_hit"] = True
        return json.loads(path.read_text(encoding="utf-8"))["content"]
    audit["cache_hit"] = False
    result = original(request, timeout)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"content": result, "model": request["payload"]["model"]}, stream, ensure_ascii=False)
    return result
