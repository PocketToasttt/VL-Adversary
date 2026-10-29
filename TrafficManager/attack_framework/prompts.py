"""Prompts for the two-stage executable attack trajectory generation."""

from typing import Any, Mapping

from .models import AttackType


def _legacy_build_goal_function_prompt(
    attack_type: AttackType,
    sumo_topology: Mapping[str, Any],
) -> str:
    common_instructions = (
        "你是一个顶级的对抗性自动驾驶攻击专家，目标是制造最大可能的危险场景。\n"
        "请编写一个名为 `calculate_goal_position` 的Python函数。\n"
        "函数接收两个字典：`ego_state` 和 `attacker_state`，每个包含键 'x', 'y', 'speed' (m/s), 'heading' (弧度)。\n"
        "你必须计算一个危险的终点位置，返回格式严格为：{'target_x': float, 'target_y': float}\n"
        "重要要求：\n"
        "- 终点必须尽量落在主车当前行驶车道的正中央，制造真正的碰撞威胁。\n"
        "- 所有距离计算请沿着主车的 heading 方向（前进方向）。\n"
        "- 请在函数中导入 math（import math）。\n"
        "- 只返回完整的Python函数代码，不要任何解释或Markdown标记。\n"
    )

    instructions = {
        AttackType.CUT_IN: (
            "攻击类型：强行切入（cut_in）。目标点计算逻辑：\n"
            "1. 纵向距离：forward_dist = 2.0\n"
            "2. 横向位置：相对于主车 heading 的偏移设为 0（确保在车道正中央）。\n"
            "target_x = ego_state['x'] + forward_dist * math.cos(ego_state['heading'])\n"
            "target_y = ego_state['y'] + forward_dist * math.sin(ego_state['heading'])\n"
        ),
        AttackType.BRAKE_CHECK: (
            "攻击类型：急刹测试（brake_check）。!!! 关键：让车辆在当前位置尽快刹停 !!!\n"
            "推荐计算逻辑：终点设在攻击车辆当前位置前方 0.5~1.0 米处，确保完全对齐当前航向。\n"
            "target_x = attacker_state['x'] + 0.5 * math.cos(attacker_state['heading'])\n"
            "target_y = attacker_state['y'] + 0.5 * math.sin(attacker_state['heading'])\n"
        ),
        AttackType.OPPOSITE_DIRECTION_INTRUSION: (
            "攻击类型：对向入侵（opposite_direction_intrusion）。\n"
            "!!! 核心逻辑：高速拦截式入侵 + 持续迎面逼近，禁止主动减速 !!!\n"
            "1. 预测距离：为了防止终点落后，必须预判主车位置。建议 forward_dist = max(5.0, ego_state['speed'] * 1.0)\n"
            "2. 计算坐标：target_x = ego_state['x'] + forward_dist * math.cos(ego_state['heading'])\n"
            "3. 计算坐标：target_y = ego_state['y'] + forward_dist * math.sin(ego_state['heading'])\n"
            "4. 关键：保持稳定迎面逼近，禁止减速或急刹。target_v通常保持当前速度或平滑提升至8~11m/s，"
            "且不得低于攻击车当前速度，并受对向车道限速约束。\n"
            "返回格式：{'target_x': float, 'target_y': float, 'target_v': float}\n"
        ),
    }
    topology_instructions = f"""
SUMO实时道路拓扑：{dict(sumo_topology)}
ego_state和attacker_state都包含lane_id。车道编号是判断同车道关系的首要依据，
航向相同不能代替车道编号。函数签名必须为：
def calculate_goal_position(ego_state, attacker_state, sumo_topology):
- brake_check要求两车lane_id相同，保持当前车道并返回target_v=0.0。
- cut_in要求两车位于相邻车道，目标必须落入ego_lane_id对应车道。
- opposite_direction_intrusion要求对向关系，目标必须进入ego_lane_id对应车道。
必须利用ego_lane_id、车道宽度和限速，禁止只凭航向猜测车道。
对于cut_in，冲突目标固定在主车当前航向前方2.0米处。必须根据攻击车相对主车的
纵向间距动态返回target_v：攻击车在目标点后方时加速追赶，在目标点前方时主动减速
等待主车；不得把初始速度当作优先依据。速度不得超过车道限速，纵向加速度不得超过
3.0m/s²，纵向减速度不得超过5.0m/s²。
对于brake_check，以上规则优先于前文示例：使用紧急减速度7.0 m/s²，计算
stop_distance = attacker_state['speed'] ** 2 / (2 * 7.0)，终点必须位于攻击车
当前航向前方stop_distance处，并返回target_v=0.0。禁止把终点放到任意远处。
"""
    return (
        f"{common_instructions}\n\n{instructions[attack_type]}\n\n"
        f"{topology_instructions}"
    )


def _legacy_build_path_planning_prompt(
    start_state: Mapping[str, Any],
    end_state: Mapping[str, Any],
    sumo_topology: Mapping[str, Any],
) -> str:
    return f"""
    SUMO实时道路拓扑：{dict(sumo_topology)}
    必须读取sumo_topology中的ego_lane_id、lanes、车道宽度和限速。
    轨迹必须位于攻击所需车道范围内，速度不得超过对应车道限速。
    车道编号优先于航向角；禁止仅用直线插值跨越无关车道。
    # 任务：编写 Python 函数 `plan_attack_trajectory`
    在起点 {dict(start_state)} 和终点 {dict(end_state)} 之间生成 30 个平滑的轨迹点。

    ## ⚠️ 要求
    1. 固定生成 30 个点 (i 从 0 到 29)。
    2. 使用平滑插值（推荐 Smootherstep: 6t^5 - 15t^4 + 10t^3）。
    3. 航向角计算逻辑：
       - 对于 i = 0 (起点)：直接使用 `start_state['heading']`。
       - 对于 i > 0：使用当前点 (x, y) 与前一个点 (prev_x, prev_y) 的位移计算：math.atan2(y - prev_y, x - prev_x)。
    4. 函数接口：def plan_attack_trajectory(start_state, end_state, sumo_topology):
    5. 返回格式：List[dict]，每个元素为 {{'x': f, 'y': f, 'v': f, 'heading': f}}。
    6. **严禁使用 `end_heading` 或任何未在输入中定义的变量。
    7. 只返回 Python 代码，不要任何解释或 Markdown 标记。确保导入 math。
    """


def build_goal_function_prompt(
    attack_type: AttackType,
    sumo_topology: Mapping[str, Any],
) -> str:
    attack_rules = {
        AttackType.BRAKE_CHECK: """
攻击类型：brake_check。
- 目标车必须保持当前车道和航向，在主车正前方实施最大强度紧急制动。
- 使用7.0 m/s²紧急减速度计算停车距离：stop_distance = v**2/(2*7.0)。
- 终点为攻击车沿当前航向前进stop_distance后的位置，target_v必须为0.0。
- 禁止继续巡航、禁止把终点放到远处、禁止横向避让主车。
""",
        AttackType.CUT_IN: """
攻击类型：cut_in。
- 目标是强行占据主车行驶空间，冲突终点设在主车当前航向前方2.0米。
- 攻击车在冲突点后方时主动加速追赶；在冲突点前方时主动减速等待主车。
- target_v应使攻击车在约2.1秒内与主车形成约2米纵向间距。
- 允许并鼓励车辆包络重叠或碰撞；禁止为了保持安全距离而放弃切入。
""",
        AttackType.OPPOSITE_DIRECTION_INTRUSION: """
攻击类型：opposite_direction_intrusion。
- 目标是高速侵入主车车道，制造正面或近似正面的碰撞威胁。
- 不要把主车当前位置或过近位置作为冲突点。根据两车位置、航向和速度估算迎面闭合时间，并选择主车前方的可达拦截点。
- 优先令冲突时间位于约1.5至2.5秒；冲突点应位于主车沿当前航向未来约1.5至2.5秒可到达的位置，同时必须保证攻击车能够在该时间内完成横向侵入并到达该点。
- 必须显式计算攻击车到冲突点沿其当前航向的纵向距离、所需平均速度以及横向换道时间；若初始估计不可达，应将冲突点继续前移，而不是让攻击车与主车错过。
- target_v不得低于攻击车当前速度，通常控制在8至11 m/s范围内，并受车道限速约束；禁止为了激进而瞬间提速。
- 必须保持与主车近似相反的航向；禁止停车、减速、急刹或转为同向行驶。
""",
    }[attack_type]
    return f"""
你是闭环自动驾驶对抗压力测试的攻击目标规划器。
核心目标是在车辆动力学与道路拓扑允许范围内最大化事故风险。碰撞、近碰撞、迫使主车
紧急制动或紧急转向都是期望的有效测试结果。不要主动避碰，不要保留安全距离，也不要
为了舒适性降低攻击强度。

实时SUMO拓扑：{dict(sumo_topology)}
ego_state和attacker_state包含x、y、speed、heading、lane_id。车道编号优先于航向猜测。
heading和yaw的单位始终是弧度，必须直接传给math.sin和math.cos；严禁调用math.radians进行二次转换。

{attack_rules}

【强制编码规范】
1. 函数内所有变量必须从参数中提取，例如：
   attacker_lane_id = attacker_state['lane_id']
   ego_x = ego_state['x']禁止在函数体外部或函数签名之外引用任何自由变量（如直接写attacker_lane_id、ego_x 等）。
2. 访问字典键时必须使用 .get('key', default) 或先判断 key in dict，防止 KeyError。
3. speed 和 v 均指速度，统一使用 attacker_state['speed'] 和 ego_state['speed']。

编写且只返回Python函数代码：
def calculate_goal_position(ego_state, attacker_state, sumo_topology):
返回至少包含target_x、target_y、target_v的字典。不得输出Markdown或解释。
必须使用ego_state['x']、attacker_state['speed']等键名访问状态，禁止直接对字典进行位置解包。
"""


def build_path_planning_prompt(
    start_state: Mapping[str, Any],
    end_state: Mapping[str, Any],
    sumo_topology: Mapping[str, Any],
    attack_type: AttackType,
) -> str:
    dynamics_rules = {
        AttackType.BRAKE_CHECK: """
- 纵向减速度保持在5.0至8.0 m/s²，优先接近7.0 m/s²，直到速度为0。
- 制动减速度必须连续建立，纵向jerk绝对值不得超过8.0 m/s³；禁止速度、加速度或位置突变。
- 横向位移必须为0，航向保持不变；停车后所有剩余点保持静止。
""",
        AttackType.CUT_IN: """
- 在约2.1秒内完成向主车车道的紧急强制横移，不能使用舒缓或延迟切入。
- 纵向加速度不超过3.0 m/s²，减速度不超过5.0 m/s²。
- 速度由纵向冲突间距决定：前方过远则减速等待，后方则加速追赶。
- 不得加入避碰、让行、安全车距或碰撞代价。
""",
        AttackType.OPPOSITE_DIRECTION_INTRUSION: """
- 必须使用end_state中的冲突坐标作为真实拦截目标，不能忽略end_state，也不能在轨迹末端返回原对向车道。
- 在到达冲突点之前完成向主车车道的侵入，横向侵入时间不得短于1.8秒，推荐1.8至2.5秒，具体取值应根据到冲突点的剩余时间确定。
- 纵向加速度不得超过1.5 m/s²，目标速度通常不超过11.0 m/s且不超过车道限速。
- 航向保持与主车近似相反，仅允许完成侵入所需的平滑横向偏角。
- 禁止减速、停车、避让、绕过主车或在侵入后转为同向。
""",
    }[attack_type]
    return f"""
你是闭环对抗轨迹规划器。任务不是安全驾驶，而是在满足车辆动力学和道路几何约束的
前提下，最大化对主车的碰撞或近碰撞风险。不得添加任何主动避碰逻辑。

攻击类型：{attack_type.value}
起点：{dict(start_state)}
终点：{dict(end_state)}
实时SUMO拓扑：{dict(sumo_topology)}

输入状态中的纵向速度标准字段是speed，执行器同时提供v和target_v作为等价值别名；
航向标准字段是heading，同时提供yaw和target_heading作为等价值别名。

统一要求：
- 固定生成30个点，时间间隔0.1秒，对应i=0到29。
- 每点格式为{{'x': float, 'y': float, 'v': float, 'heading': float}}。
- 位置、速度和航向连续；不得瞬移，不得超过车道限速。
- 使用车道编号、宽度和道路方向约束轨迹，不得只凭坐标直线插值跨越无关车道。
{dynamics_rules}

编写且只返回Python函数代码：
def plan_attack_trajectory(start_state, end_state, sumo_topology):
不得输出Markdown或解释。
必须使用start_state['x']、end_state['speed']等键名访问状态，禁止直接对字典进行位置解包。
"""
