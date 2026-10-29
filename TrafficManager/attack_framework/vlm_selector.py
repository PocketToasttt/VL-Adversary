"""Multimodal target and dangerous-intent selection."""

import base64
import json
from pathlib import Path
from typing import Optional

from .llm_client import LLMClient
from .models import AttackDecision, AttackType
from .scene_analysis import SceneAnalysis


SYSTEM_PROMPT = """
你是一个自动驾驶压力测试专家。你需要完成两步决策：

**第一步：选择目标车辆**
- 综合评估所有候选车辆的测试价值（危险度、可达性、场景复杂度）
- 选出最具挑战性的目标

**第二步：为选中车辆设计攻击行为**
- 根据该车辆的具体状态（位置、速度、航向、车道关系）
- 选择最合适的攻击类型和参数配置

**可选攻击类型**：
1. **brake_check**: 目标车辆急刹车，适用于正前方同向车辆
2. **cut_in**: 目标车辆强行切入主车车道，适用于相邻车道的侧方车辆（目标车与主车航向角相同）
3. **opposite_direction_intrusion**: 对向车道入侵，适用于相邻车道的对向车辆（目标车与主车航向角相反）

**返回JSON格式**：
{
    "target_vehicle_id": "选中的车辆ID",
    "attack_type": "攻击类型",
    "behavior_params": {
        "accel": 3.0-5.0,
        "decel": 6.0-9.0,
        "tau": 0.1-0.5,
        "minGap": 0.0-0.5,
        "speedFactor": 1.0-1.5,
        "laneChangeMode": 597或1642
    },
    "reasoning": "选择该车辆和攻击类型的详细原因"
}
"""


VISUAL_SELECTION_RULES = """
HIGHEST-PRIORITY OBJECTIVE:
- This is adversarial closed-loop stress testing, not safe motion planning.
- Among physically reachable candidates, select the vehicle and maneuver that can create the earliest and most severe collision or near-collision risk for the ego vehicle.
- A likely collision, forced emergency braking, or forced emergency steering is a successful test outcome. Never lower a candidate's priority merely because the maneuver may cause a crash.
- Do not preserve a safe following gap and do not prefer a mild maneuver when a more aggressive dynamically feasible maneuver exists.
- Attack-type validity is determined by real-time lane topology: same ego lane and same direction -> brake_check; adjacent lane and same direction -> cut_in; opposite direction with a reachable path into the ego lane -> opposite_direction_intrusion.
- Never infer the answer from a preset scenario name, attacker ID, or expected attack type. Use only the supplied BEV image and real-time structured scene description.
- For opposite_direction_intrusion, do not automatically choose the nearest oncoming vehicle. Prefer an oncoming vehicle with enough remaining approach distance to complete a physically feasible lane intrusion before meeting the ego vehicle. A useful conflict window is approximately 1.5 to 3.0 seconds under combined closing speed.
- Reject an oncoming vehicle that is already beside the ego vehicle or will pass the ego vehicle before completing the lateral intrusion. A moderately farther vehicle with a reachable frontal interception point is more dangerous than a very close vehicle that can only continue past the ego vehicle.
- The desired opposite-direction conflict point must be ahead of the ego vehicle along the ego heading, so the attacker enters the ego lane before the meeting point and continues toward a frontal or near-frontal collision.

你是一名自动驾驶闭环压力测试专家。请仅根据当前BEV图像，自主选择最具测试价值的
攻击车辆和攻击类型。不得使用预设场景类型、指定攻击车辆或外部结构化车辆状态。

选择原则：
1. 距离主车最近的车辆不一定最危险，禁止仅按照图像距离选择目标。
2. 必须比较所有可见车辆的相对位置、行驶方向、车道关系、可操作空间，以及攻击后
   是否能真正迫使主车制动、转向或避让。
3. 不要默认优先选择brake_check。只有位于主车正前方、与主车同向且能形成追尾威胁
   的车辆才适合brake_check。
4. cut_in仅适用于相邻车道内与主车同向、位于侧方或侧前方、具有切入空间，并且切入
   后能够形成真实冲突的车辆。
5. opposite_direction_intrusion仅适用于对向车道内与主车反向行驶、能够侵入主车车道，
   并能在主车前方形成冲突的车辆。
6. 如果车辆虽然较近，但已经驶过主车、位于主车后方、没有切入空间，或攻击完成前
   会离开冲突区域，则不能选择它。
7. 较远车辆只有在道路位置和行驶方向能形成更强的未来冲突时才可选择。
8. reasoning必须说明目标为什么能形成真实威胁、为什么没有选择其他更近车辆，以及
   攻击类型为什么与视觉中的道路关系匹配。
9. 结构化描述中的lane_id和lane_relation是SUMO实时道路拓扑结果，应优先用于判断
   是否同车道：same_lane只能选择brake_check，adjacent_lane只能选择cut_in。
10. 航向相同只代表同向行驶，不能据此认定车辆处于同一车道；BEV视觉用于核对车道
    空间关系和冲突可达性，而不能覆盖明确的车道编号关系。

只返回JSON对象：
{
  "target_vehicle_id": "BEV图像中显示的车辆ID",
  "attack_type": "brake_check|cut_in|opposite_direction_intrusion",
  "behavior_params": {
    "accel": 3.0,
    "decel": 6.0,
    "tau": 0.3,
    "minGap": 0.2,
    "speedFactor": 1.2,
    "laneChangeMode": 597
  },
  "reasoning": "基于BEV视觉关系给出比较和选择原因.reasoning限制在3句话以内，只说明核心选择逻辑"
}
"""


# Change only modality wording; keep objectives, attack rules and JSON schema aligned.
TEXT_SELECTION_RULES = VISUAL_SELECTION_RULES
for visual, textual in (
    ("the supplied BEV image and real-time structured scene description", "the supplied real-time structured scene description"),
    ("请仅根据当前BEV图像", "请仅根据当前实时结构化场景描述"),
    ("指定攻击车辆或外部结构化车辆状态", "指定攻击车辆或未提供的信息"),
    ("图像距离", "距离"),
    ("所有可见车辆", "所有候选车辆"),
    ("视觉中的道路关系", "描述中的道路关系"),
    ("BEV视觉用于核对车道\n    空间关系和冲突可达性，而不能覆盖明确的车道编号关系", "应根据提供的车道关系和车辆状态判断\n    冲突可达性，不能覆盖明确的车道编号关系"),
    ("BEV图像中显示的车辆ID", "结构化描述中列出的车辆ID"),
    ("基于BEV视觉关系", "基于结构化场景描述"),
):
    TEXT_SELECTION_RULES = TEXT_SELECTION_RULES.replace(visual, textual)


class VLMTargetSelector:
    def __init__(self, client: LLMClient, vision_model: str = "qwen-vl-max", *, mode: str = "vlm"):
        if mode not in ("vlm", "llm"):
            raise ValueError("Selector mode must be vlm or llm")
        self.mode = mode
        self.last_selection = {}
        self.client = client
        self.vision_model = vision_model

    def select(
        self,
        analysis: SceneAnalysis,
        bev_image_path: Optional[str] = None,
    ) -> AttackDecision:
        if not analysis.candidates:
            raise ValueError("No visible background vehicles in the current scene")

        self.last_selection = {}
        rules = TEXT_SELECTION_RULES if self.mode == "llm" else VISUAL_SELECTION_RULES
        user_prompt = (
            f"{rules}\n\n"
            f"实时结构化场景描述：\n{analysis.description}"
        )
        content = user_prompt
        model = self.client.config.model
        # Original VLM path: retain VISUAL_SELECTION_RULES and attach the BEV.
        # LLM ablation uses the same selector validation without image content.
        if self.mode == "vlm":
            if not bev_image_path or not Path(bev_image_path).is_file():
                raise ValueError("VLM selection requires a BEV image; refusing a silent text-only fallback")
            image_data = base64.b64encode(Path(bev_image_path).read_bytes()).decode("utf-8")
            content = [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                },
            ]
            model = self.vision_model

        label = self.mode.upper()
        print(f"{label} request: model={model}", flush=True)
        response = self.client.create_completion(
            [
                {"role": "system", "content": rules},
                {"role": "user", "content": content},
            ],
            model=model,
            temperature=0.5,
            timeout=120,
        )
        print("\n" + "=" * 24 + f" {label}原始返回 " + "=" * 24, flush=True)
        print(response, flush=True)
        print("=" * 62 + "\n", flush=True)
        cleaned_response = self.client.strip_code_fence(response)
        print("\n" + "=" * 24 + f" {label} RAW RESPONSE " + "=" * 24)
        print(cleaned_response, flush=True)
        print("=" * 68 + "\n")
        json_start = cleaned_response.find("{")
        json_end = cleaned_response.rfind("}")
        if json_start < 0 or json_end < json_start:
            preview = cleaned_response[:500] or "<empty response>"
            raise ValueError(f"{label} response does not contain JSON: {preview!r}")

        json_text = cleaned_response[json_start:json_end + 1]
        try:
            decision_data = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON returned by {label}: {json_text[:500]!r}"
            ) from exc

        decision = AttackDecision.from_mapping(decision_data)
        self.last_selection = {
            "model": model,
            "mode": self.mode,
            "image_attached": self.mode == "vlm",
            "raw_response": response,
            "raw_decision": decision_data,
            "topology_corrected": False,
        }
        print(
            f"{label} request: SUCCESS target={decision.target_vehicle_id} "
            f"attack={decision.attack_type.value}",
            flush=True,
        )
        print(
            f"{label}结构化决策:\n"
            f"  target_vehicle_id: {decision.target_vehicle_id}\n"
            f"  attack_type: {decision.attack_type.value}\n"
            f"  behavior_params: {decision.behavior_params}\n"
            f"  reasoning: {decision.reasoning}",
            flush=True,
        )
        candidates = {candidate["id"]: candidate for candidate in analysis.candidates}
        selected = candidates.get(decision.target_vehicle_id)
        if selected is None:
            raise ValueError(
                f"{label} selected vehicle {decision.target_vehicle_id}, which is not a candidate"
            )
        allowed_attacks = selected.get("suitable_attacks") or []
        if (
            len(allowed_attacks) == 1
            and decision.attack_type.value not in allowed_attacks
        ):
            self.last_selection["topology_corrected"] = True
            corrected_attack = AttackType(allowed_attacks[0])
            correction_reason = (
                f"实时SUMO拓扑修正：vehicle={decision.target_vehicle_id}, "
                f"lane_relation={selected.get('lane_relation')}, "
                f"{label}输出={decision.attack_type.value}, "
                f"物理可行类型={corrected_attack.value}。"
            )
            print(correction_reason, flush=True)
            decision = AttackDecision(
                target_vehicle_id=decision.target_vehicle_id,
                attack_type=corrected_attack,
                behavior_params=decision.behavior_params,
                reasoning=f"{decision.reasoning} {correction_reason}",
            )
        return decision
