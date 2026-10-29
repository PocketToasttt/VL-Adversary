#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_attack_scenarios_backup.py

生成“单场景独立文件”的交互场景库：
- 每个 cluster = 一个独立场景
- 每个场景只指定一个 ego_id
- 默认所有车辆在 t=0 出现（可通过 --depart-window 设置微小抖动）
- following 场景支持密集跟车、多模板、多速度模式、邻车道上下文车
- 同类场景内部尽量不重复；跨类型允许复用道路，提高可生成场景数

输出结构:
    <output>/<scene_id>/<scene_id>.rou.xml
    <output>/<scene_id>/<scene_id>.sumocfg
    <output>/<scene_id>/<scene_id>.json
    <output>/scene_index.json

示例:
python3 generate_attack_scenarios_backup.py \
  --net networkFiles/boston-seaport/osm.net.xml \
  --output networkFiles/boston-seaport/scenarios_by_ego \
  --num-following 20 \
  --num-parallel 12 \
  --num-oncoming 8 \
  --num-junction 10 \
  --background-per-scene 2 \
  --depart-window 0.0 \
  --seed 42 \
  --clean-output
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime

# SUMO tools 路径
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
else:
    for p in [
        r"C:\Program Files (x86)\Eclipse\Sumo\tools",
        r"C:\Program Files\Eclipse\Sumo\tools",
        "/usr/share/sumo/tools",
        "/usr/local/share/sumo/tools",
        "/opt/homebrew/share/sumo/tools",
    ]:
        if os.path.exists(p):
            sys.path.append(p)
            break

try:
    import sumolib
except ImportError:
    print("❌ 无法导入 sumolib，请确认 SUMO_HOME 或 SUMO tools 路径。")
    raise


SCENARIO_ATTACK_TYPES = {
    "following": "brake_check",
    "parallel": "cut_in",
    "oncoming": "opposite_direction_intrusion",
}


ATTACK_TYPE_CN = {
    "brake_check": "前车急刹",
    "cut_in": "强行切入",
    "opposite_direction_intrusion": "对向车道入侵",
}


# ═══════════════════════════════════════════════════════════
# 通用工具
# ═══════════════════════════════════════════════════════════

def to_unix_path(path):
    return path.replace("\\", "/")


def relpath_for_cfg(target_path, base_dir):
    try:
        p = os.path.relpath(os.path.abspath(target_path), os.path.abspath(base_dir))
    except ValueError:
        p = os.path.abspath(target_path)
    return to_unix_path(p)


def clamp(x, low, high):
    return max(low, min(x, high))


def prioritized_randomized(items, topk=None):
    """
    前 topk 个优先候选先打乱，剩余部分也打乱，然后拼接。
    这样既保留“优先选择优质道路”的倾向，也避免总是同一条路。
    """
    xs = list(items)
    if not xs:
        return xs
    if topk is None or topk <= 0 or topk >= len(xs):
        random.shuffle(xs)
        return xs

    head = xs[:topk]
    tail = xs[topk:]
    random.shuffle(head)
    random.shuffle(tail)
    return head + tail


def pair_key(e1, e2):
    return tuple(sorted((e1.getID(), e2.getID())))


def route_to_str(route):
    return " ".join(e.getID() for e in route)


def iter_outgoing_edges(edge):
    """
    兼容 sumolib 中 Edge.getOutgoing() 的不同返回格式。
    """
    outgoing = edge.getOutgoing()
    if isinstance(outgoing, dict):
        return list(outgoing.keys())
    return list(outgoing)


def sample_depart(args):
    """
    统一控制车辆出发时间：
    - 默认 0.0：所有车在 t=0 出现
    - 如果想保留一点抖动，可设置 --depart-window 0.1 / 0.2
    """
    window = max(0.0, getattr(args, "depart_window", 0.0))
    if window <= 0.0:
        return 0.0
    return random.uniform(0.0, window)


def choose_adjacent_lane(main_lane, lane_num):
    cand = []
    if main_lane - 1 >= 0:
        cand.append(main_lane - 1)
    if main_lane + 1 < lane_num:
        cand.append(main_lane + 1)
    return random.choice(cand) if cand else None


def enforce_positions(raw_positions, min_spacing, max_pos):
    """
    对同车道初始位置做整理，避免 t=0 插入失败。
    """
    pos = sorted(raw_positions)
    out = []
    for i, p in enumerate(pos):
        if i == 0:
            out.append(clamp(p, 5, max_pos))
        else:
            out.append(max(out[-1] + min_spacing, p))

    if not out or out[-1] > max_pos:
        return None
    return out


# ═══════════════════════════════════════════════════════════
# 路网分析
# ═══════════════════════════════════════════════════════════

def analyze_network(net):
    all_edges = [
        e for e in net.getEdges()
        if not e.getID().startswith(':') and e.allows("passenger")
    ]

    long_edges = [e for e in all_edges if e.getLength() > 60]
    multilane = [e for e in all_edges if e.getLaneNumber() >= 2 and e.getLength() > 40]

    # 双向道路对：按 node 反向匹配
    reverse_lookup = {}
    for e in all_edges:
        key = (e.getFromNode().getID(), e.getToNode().getID())
        reverse_lookup.setdefault(key, []).append(e)

    bidir_pairs = []
    seen = set()
    for e in all_edges:
        reverse_key = (e.getToNode().getID(), e.getFromNode().getID())
        for re in reverse_lookup.get(reverse_key, []):
            if re.getID() == e.getID():
                continue
            k = pair_key(e, re)
            if k in seen:
                continue
            bidir_pairs.append((e, re))
            seen.add(k)
            break

    # 交叉口附近道路
    junction_edges = []
    junction_edge_ids = set()
    for node in net.getNodes():
        incoming = [
            e for e in node.getIncoming()
            if not e.getID().startswith(':') and e.allows("passenger")
        ]
        outgoing = [
            e for e in node.getOutgoing()
            if not e.getID().startswith(':') and e.allows("passenger")
        ]
        if len(incoming) + len(outgoing) >= 4:
            for e in incoming:
                if e.getLength() > 30 and e.getID() not in junction_edge_ids:
                    junction_edges.append(e)
                    junction_edge_ids.add(e.getID())

    print("=" * 60)
    print("📊 路网分析结果")
    print("=" * 60)
    print(f"  总 edge 数:           {len(all_edges)}")
    print(f"  长道路 (>60m):         {len(long_edges)}")
    print(f"  多车道 (≥2):          {len(multilane)}")
    print(f"  双向道路对数:          {len(bidir_pairs)}")
    print(f"  交叉口附近道路:        {len(junction_edges)}")
    print("=" * 60)

    return {
        "all": all_edges,
        "long": long_edges,
        "multilane": multilane,
        "bidirectional": bidir_pairs,
        "junction": junction_edges,
    }


def build_route(net, start_edge, min_edges=2, max_edges=6):
    """
    构建多样化路线：
    - 不再永远选最长后继
    - 从前几个较优后继里按长度加权随机
    """
    if max_edges < min_edges:
        max_edges = min_edges


    route = [start_edge]
    current = start_edge
    visited = {start_edge.getID()}
    target_len = random.randint(min_edges, max_edges)

    for _ in range(target_len - 1):
        successors = []
        for e in iter_outgoing_edges(current):
            if e.getID().startswith(':'):
                continue
            if e.getID() in visited:
                continue
            if not e.allows("passenger"):
                continue
            successors.append(e)

        if not successors:
            break

        successors.sort(key=lambda x: x.getLength(), reverse=True)
        top = successors[:min(3, len(successors))]
        weights = [max(1.0, e.getLength()) for e in top]
        current = random.choices(top, weights=weights, k=1)[0]

        route.append(current)
        visited.add(current.getID())

        if len(route) >= min_edges and len(route) < max_edges and random.random() < 0.20:
            break

    if len(route) >= min_edges:
        return route
    return None


# ═══════════════════════════════════════════════════════════
# 场景对象
# ═══════════════════════════════════════════════════════════

class VehicleCluster:
    """
    一个 cluster = 一个独立场景
    """

    def __init__(self, cluster_type, description, focus_key=None):
        self.cluster_type = cluster_type
        self.description = description
        self.focus_key = focus_key
        self.scene_id = None
        self.ego_id = None
        self.attack_type = SCENARIO_ATTACK_TYPES.get(cluster_type)
        self.attacker_candidate_id = None
        self.vehicles = []

    def add(self, vid, edge_route_str, lane, pos, speed, depart, edge_len=None, role="npc"):
        start_edge = edge_route_str.split()[0] if edge_route_str else ""
        self.vehicles.append({
            "id": int(vid),
            "route": edge_route_str,
            "start_edge": start_edge,
            "lane": int(lane),
            "pos": float(pos),
            "speed": float(speed),
            "depart": float(depart),
            "edge_len": float(edge_len) if edge_len is not None else None,
            "role": role,
        })

    def vehicle_ids(self):
        return [int(v["id"]) for v in self.vehicles]

    def depart_range(self):
        if not self.vehicles:
            return None
        deps = [v["depart"] for v in self.vehicles]
        return [round(min(deps), 2), round(max(deps), 2)]

    def choose_ego(self):
        if not self.vehicles:
            return

        for v in self.vehicles:
            if v["role"] == "ego":
                v["role"] = "npc"

        ego = None

        if self.cluster_type == "following":
            candidates = list(self.vehicles)

            if self.focus_key:
                parts = self.focus_key.split("|")
                main_edge = parts[0]
                main_lane = None

                for p in parts[1:]:
                    if p.startswith("lane="):
                        main_lane = int(p.split("=", 1)[1])

                edge_candidates = [v for v in candidates if v["start_edge"] == main_edge]
                if edge_candidates:
                    candidates = edge_candidates

                if main_lane is not None:
                    lane_candidates = [v for v in candidates if v["lane"] == main_lane]
                    if len(lane_candidates) >= 3:
                        candidates = lane_candidates

            ordered = sorted(candidates, key=lambda v: v["pos"])
            n = len(ordered)
            if n == 0:
                ego = random.choice(self.vehicles)
            elif n % 2 == 1:
                ego = ordered[n // 2]
            else:
                ego = random.choice([ordered[n // 2 - 1], ordered[n // 2]])

        elif self.cluster_type == "parallel":
            def score(v):
                s = 0.0
                for u in self.vehicles:
                    if u["id"] == v["id"]:
                        continue
                    lane_diff = abs(u["lane"] - v["lane"])
                    pos_diff = abs(u["pos"] - v["pos"])
                    if lane_diff == 1 and pos_diff < 15:
                        s += 3.0
                    elif lane_diff == 2 and pos_diff < 12:
                        s += 1.5
                    if lane_diff == 0 and pos_diff < 25:
                        s += 1.2
                return s

            ego = max(self.vehicles, key=score)

        elif self.cluster_type == "oncoming":
            primary_edge = self.focus_key.split("|")[0] if self.focus_key else None
            candidates = [v for v in self.vehicles if v["start_edge"] == primary_edge]
            if not candidates:
                candidates = self.vehicles
            center_pos = sum(v["pos"] for v in candidates) / len(candidates)
            ego = min(candidates, key=lambda v: abs(v["pos"] - center_pos))

        elif self.cluster_type == "junction":
            candidates = [v for v in self.vehicles if v["edge_len"] is not None]
            if candidates:
                ego = min(candidates, key=lambda v: max(0.0, v["edge_len"] - v["pos"]))
            else:
                ego = max(self.vehicles, key=lambda v: v["pos"])

        else:
            ego = random.choice(self.vehicles)

        ego["role"] = "ego"
        self.ego_id = int(ego["id"])

    def choose_attacker_candidate(self):
        if self.ego_id is None:
            self.choose_ego()
        if self.ego_id is None:
            return

        existing_attacker = next(
            (v for v in self.vehicles if v["role"] == "attacker_candidate"),
            None,
        )
        if existing_attacker is not None:
            self.attacker_candidate_id = int(existing_attacker["id"])
            return

        for v in self.vehicles:
            if v["role"] == "attacker_candidate":
                v["role"] = "npc"

        ego = next((v for v in self.vehicles if int(v["id"]) == int(self.ego_id)), None)
        if ego is None:
            return

        non_ego = [v for v in self.vehicles if int(v["id"]) != int(self.ego_id)]
        attacker = None

        if self.cluster_type == "following":
            same_lane_front = [
                v for v in non_ego
                if v["start_edge"] == ego["start_edge"]
                and v["lane"] == ego["lane"]
                and v["pos"] > ego["pos"]
            ]
            if same_lane_front:
                attacker = min(same_lane_front, key=lambda v: v["pos"] - ego["pos"])

        elif self.cluster_type == "parallel":
            adjacent_lane = [
                v for v in non_ego
                if v["start_edge"] == ego["start_edge"]
                and abs(v["lane"] - ego["lane"]) == 1
            ]
            if adjacent_lane:
                attacker = min(adjacent_lane, key=lambda v: abs(v["pos"] - ego["pos"]))

        elif self.cluster_type == "oncoming":
            opposite_edge = [v for v in non_ego if v["start_edge"] != ego["start_edge"]]
            if opposite_edge:
                attacker = min(opposite_edge, key=lambda v: abs(v["pos"] - ego["pos"]))

        if attacker is None and non_ego:
            attacker = min(non_ego, key=lambda v: abs(v["pos"] - ego["pos"]))

        if attacker is not None:
            attacker["role"] = "attacker_candidate"
            self.attacker_candidate_id = int(attacker["id"])


# ═══════════════════════════════════════════════════════════
# 场景生成器
# ═══════════════════════════════════════════════════════════

def generate_following_cluster(net, analysis, vid_start, used, args):
    """
    更丰富的 following 场景：
    - dense_low:      低速密集跟车
    - dense_medium:   中速密集跟车
    - closing_in:     后车更快，形成逼近
    - accordion:      速度波动，类似拥堵波
    - near_junction_queue: 路口前低速短距队列
    """
    candidate_edges = []
    seen_ids = set()

    junction_ids = {e.getID() for e in analysis["junction"] if e.getLength() > 45}

    # 开放道路 + 交叉口进口道混合，提高多样性
    for e in list(analysis["long"]) + [x for x in analysis["junction"] if x.getLength() > 45]:
        if e.getID() in used["following_edges"]:
            continue
        if e.getID() in seen_ids:
            continue
        candidate_edges.append(e)
        seen_ids.add(e.getID())

    candidate_edges.sort(key=lambda e: (-e.getLaneNumber(), -e.getLength()))

    templates = [
        {
            "name": "dense_low",
            "cars": (4, 6),
            "spacing": (8.0, 11.5),
            "speed": (3.0, 6.0),
            "mode": "flat",
            "near_end": False,
        },
        {
            "name": "dense_medium",
            "cars": (4, 6),
            "spacing": (10.0, 15.0),
            "speed": (6.0, 9.0),
            "mode": "flat",
            "near_end": False,

        },
        {
            "name": "closing_in",
            "cars": (4, 5),
            "spacing": (12.0, 18.0),
            "speed": (5.0, 7.5),
            "mode": "rear_fast",
            "near_end": False,
        },
        {
            "name": "accordion",
            "cars": (5, 6),
            "spacing": (8.5, 13.0),
            "speed": (2.5, 6.5),
            "mode": "wave",
            "near_end": False,
        },
        {
            "name": "near_junction_queue",
            "cars": (4, 5),
            "spacing": (7.5, 10.5),
            "speed": (0.5, 3.0),
            "mode": "queue",
            "near_end": True,
        },
    ]

    non_queue_templates = [t for t in templates if t["name"] != "near_junction_queue"]

    for edge in prioritized_randomized(candidate_edges, topk=60):
        route = build_route(net, edge, min_edges=3, max_edges=6)
        if route is None:
            continue

        if edge.getID() in junction_ids:
            template = random.choice(templates)
        else:
            template = random.choice(non_queue_templates)

        edge_len = edge.getLength()
        num_cars = random.randint(*template["cars"])
        nominal_spacing = random.uniform(*template["spacing"])
        min_spacing = 7.8 if template["name"] == "near_junction_queue" else 8.5
        total_span = nominal_spacing * (num_cars - 1)

        # 两种布置方式：
        # 1) 普通开放道路跟车
        # 2) 贴近路口末端排队
        if template["near_end"]:
            front_pos = edge_len - random.uniform(12, 24)
            rear_pos = front_pos - total_span
            if rear_pos < 6:
                continue
            raw_positions = [
                rear_pos + i * nominal_spacing + random.uniform(-1.0, 1.0)
                for i in range(num_cars)
            ]
        else:
            max_rear = min(edge_len * 0.40, edge_len - 15 - total_span)
            if max_rear < 8:
                continue
            rear_pos = random.uniform(8, max_rear)
            raw_positions = [
                rear_pos + i * nominal_spacing + random.uniform(-1.2, 1.2)
                for i in range(num_cars)
            ]

        positions = enforce_positions(
            raw_positions,
            min_spacing=min_spacing,
            max_pos=edge_len - 10
        )
        if positions is None:
            continue

        lane = random.randint(0, edge.getLaneNumber() - 1)
        route_str = route_to_str(route)

        cluster = VehicleCluster(
            "following",
            f"跟车群组[{template['name']}] @ {edge.getID()} ({edge_len:.0f}m)",
            focus_key=f"{edge.getID()}|lane={lane}|template={template['name']}"
        )

        base_speed = random.uniform(*template["speed"])
        step = random.uniform(0.8, 1.6)

        # 主跟车队列
        for i, pos in enumerate(positions):
            if template["mode"] == "flat":
                speed = base_speed + random.uniform(-0.8, 0.8)

            elif template["mode"] == "rear_fast":
                # 越靠后越快，形成逼近感
                speed = base_speed + (num_cars - 1 - i) * step + random.uniform(-0.3, 0.3)

            elif template["mode"] == "wave":
                # 拥堵波/速度波
                delta = random.uniform(0.8, 1.8)
                speed = base_speed + (delta if i % 2 == 0 else -delta) + random.uniform(-0.4, 0.4)

            elif template["mode"] == "queue":
                # 路口前低速排队，前方略快
                speed = base_speed + i * random.uniform(0.15, 0.60) + random.uniform(-0.15, 0.15)

            else:
                speed = base_speed

            cluster.add(
                vid=vid_start + len(cluster.vehicles),
                edge_route_str=route_str,
                lane=lane,
                pos=pos,
                speed=clamp(speed, 0.0, 12.5),
                depart=sample_depart(args),
                edge_len=edge_len,
            )

        # 邻车道上下文车辆：让场景更丰富
        if edge.getLaneNumber() >= 2 and random.random() < 0.70:
            side_lane = choose_adjacent_lane(lane, edge.getLaneNumber())
            if side_lane is not None:
                anchor = positions[len(positions) // 2]
                for _ in range(random.randint(1, 2)):
                    side_pos = clamp(anchor + random.uniform(-6, 6), 5, edge_len - 10)
                    side_speed = clamp(base_speed + random.uniform(-1.5, 1.5), 0.0, 12.5)
                    cluster.add(
                        vid=vid_start + len(cluster.vehicles),
                        edge_route_str=route_str,
                        lane=side_lane,
                        pos=side_pos,
                        speed=side_speed,
                        depart=sample_depart(args),
                        edge_len=edge_len,
                    )

        if len(cluster.vehicles) >= 4:
            cluster.choose_ego()
            used["following_edges"].add(edge.getID())
            return cluster, vid_start + len(cluster.vehicles)

    return None, vid_start


def generate_parallel_cluster(net, analysis, vid_start, used, args):
    candidate_edges = sorted(
        [e for e in analysis["multilane"] if e.getID() not in used["parallel_edges"]],
        key=lambda e: -e.getLength()
    )

    for edge in prioritized_randomized(candidate_edges, topk=40):
        route = build_route(net, edge, min_edges=3, max_edges=6)
        if route is None:
            continue

        edge_len = edge.getLength()
        num_lanes = edge.getLaneNumber()
        if num_lanes < 2:
            continue

        lane_window = min(num_lanes, 3)
        start_lane = 0 if num_lanes == lane_window else random.randint(0, num_lanes - lane_window)
        selected_lanes = list(range(start_lane, start_lane + lane_window))

        lane_counts = {lane_idx: 1 for lane_idx in selected_lanes}
        extra_cars = random.randint(1, lane_window)
        for _ in range(extra_cars):
            lane_counts[random.choice(selected_lanes)] += 1

        cluster = VehicleCluster(
            "parallel",
            f"并行群组 @ {edge.getID()} ({edge_len:.0f}m, {num_lanes}车道)",
            focus_key=edge.getID()
        )

        route_str = route_to_str(route)
        base_pos = random.uniform(10, min(45, edge_len * 0.35))
        base_speed = random.uniform(8, 12)

        vid = vid_start
        for lane_order, lane_idx in enumerate(selected_lanes):
            lane_spacing = random.uniform(20, 35)
            for j in range(lane_counts[lane_idx]):
                pos = base_pos + j * lane_spacing + lane_order * random.uniform(-4, 4)
                pos = clamp(pos, 5, edge_len - 10)
                cluster.add(
                    vid=vid,
                    edge_route_str=route_str,
                    lane=lane_idx,
                    pos=pos,
                    speed=base_speed + random.uniform(-2, 2),
                    depart=sample_depart(args),
                    edge_len=edge_len,
                )
                vid += 1

        if len(cluster.vehicles) >= 3:
            cluster.choose_ego()
            used["parallel_edges"].add(edge.getID())
            return cluster, vid

    return None, vid_start


def generate_diverse_parallel_cluster(net, analysis, vid_start, used, args):
    layouts = [
        ("left", "rear"),
        ("left", "side"),
        ("left", "front"),
        ("right", "rear"),
        ("right", "side"),
        ("right", "front"),
    ]
    layout_index = used.setdefault("parallel_layout_index", 0)
    desired_side, desired_zone = layouts[layout_index % len(layouts)]
    candidate_edges = sorted(
        [
            edge for edge in analysis["multilane"]
            if edge.getID() not in used["parallel_edges"]
        ],
        key=lambda edge: (-edge.getLaneNumber(), -edge.getLength()),
    )

    for edge in prioritized_randomized(candidate_edges, topk=60):
        route = build_route(net, edge, min_edges=3, max_edges=6)
        if route is None or edge.getLength() < 70:
            continue
        lane_count = edge.getLaneNumber()
        if lane_count < 2:
            continue

        if desired_side == "left":
            ego_lane = random.randint(0, lane_count - 2)
            attacker_lane = ego_lane + 1
        else:
            ego_lane = random.randint(1, lane_count - 1)
            attacker_lane = ego_lane - 1

        ego_speed = random.uniform(7.5, 11.0)
        if desired_zone == "rear":
            attacker_speed = min(13.0, ego_speed + random.uniform(1.8, 3.0))
            desired_delta_at_attack = random.uniform(-10.0, -6.0)
        elif desired_zone == "front":
            attacker_speed = min(13.0, ego_speed + random.uniform(0.5, 1.8))
            desired_delta_at_attack = random.uniform(6.0, 11.0)
        else:
            attacker_speed = min(13.0, ego_speed + random.uniform(0.8, 2.2))
            desired_delta_at_attack = random.uniform(-2.5, 2.5)

        attack_time = 2.0
        initial_delta = desired_delta_at_attack - (
            attacker_speed - ego_speed
        ) * attack_time
        edge_len = edge.getLength()
        margin = max(20.0, abs(initial_delta) + 10.0)
        lower = margin
        upper = edge_len - margin
        if upper <= lower:
            continue
        ego_pos = random.uniform(lower, upper)
        attacker_pos = ego_pos + initial_delta
        if not 5.0 <= attacker_pos <= edge_len - 10.0:
            continue

        route_str = route_to_str(route)
        cluster = VehicleCluster(
            "parallel",
            (
                f"多样化切入[{desired_side}_{desired_zone}] @ "
                f"{edge.getID()} ({edge_len:.0f}m, {lane_count}车道)"
            ),
            focus_key=(
                f"{edge.getID()}|side={desired_side}|zone={desired_zone}"
            ),
        )
        ego_id = vid_start
        attacker_id = vid_start + 1
        cluster.add(
            ego_id, route_str, ego_lane, ego_pos, ego_speed,
            sample_depart(args), edge_len=edge_len, role="ego",
        )
        cluster.add(
            attacker_id, route_str, attacker_lane, attacker_pos,
            attacker_speed, sample_depart(args), edge_len=edge_len,
            role="attacker_candidate",
        )

        context_specs = [
            (ego_lane, ego_pos - random.uniform(14.0, 22.0)),
            (ego_lane, ego_pos + random.uniform(18.0, 30.0)),
            (attacker_lane, attacker_pos + random.uniform(18.0, 30.0)),
        ]
        next_id = attacker_id + 1
        for lane, position in context_specs:
            position = clamp(position, 5.0, edge_len - 10.0)
            if any(
                vehicle["lane"] == lane
                and abs(vehicle["pos"] - position) < 8.0
                for vehicle in cluster.vehicles
            ):
                continue
            cluster.add(
                next_id,
                route_str,
                lane,
                position,
                clamp(ego_speed + random.uniform(-1.5, 1.5), 4.0, 13.0),
                sample_depart(args),
                edge_len=edge_len,
            )
            next_id += 1

        cluster.ego_id = ego_id
        cluster.attacker_candidate_id = attacker_id
        used["parallel_edges"].add(edge.getID())
        used["parallel_layout_index"] = layout_index + 1
        return cluster, next_id

    return None, vid_start


def generate_oncoming_cluster(net, analysis, vid_start, used, args):
    pair_usage = used.setdefault("oncoming_pair_usage", {})
    profiles = [
        {
            "name": "close_fast",
            "distance": (18.0, 26.0),
            "ego_speed": (6.0, 9.0),
            "oncoming_speed": (7.0, 10.0),
        },
        {
            "name": "medium_balanced",
            "distance": (27.0, 36.0),
            "ego_speed": (4.5, 7.0),
            "oncoming_speed": (4.5, 7.5),
        },
        {
            "name": "far_high_speed",
            "distance": (37.0, 48.0),
            "ego_speed": (7.0, 10.5),
            "oncoming_speed": (7.0, 10.5),
        },
    ]
    profile_index = used.setdefault("oncoming_profile_index", 0)
    profile = profiles[profile_index % len(profiles)]
    candidate_pairs = []
    for e1, e2 in analysis["bidirectional"]:
        if e1.getLaneNumber() < 2 <= e2.getLaneNumber():
            e1, e2 = e2, e1
        if e1.getLaneNumber() < 2:
            continue
        l1, l2 = e1.getLength(), e2.getLength()
        if l1 <= 40 or l2 <= 40:
            continue
        if max(l1, l2) / max(1.0, min(l1, l2)) > 1.8:
            continue
        key = pair_key(e1, e2)
        candidate_pairs.append((e1, e2))

    candidate_pairs.sort(
        key=lambda pair: (
            pair_usage.get(pair_key(pair[0], pair[1]), 0),
            -pair[0].getLaneNumber(),
            -(pair[0].getLength() + pair[1].getLength()),
        )
    )

    layout_candidates = candidate_pairs * 8
    for ego_edge, onc_edge in prioritized_randomized(layout_candidates):
        ego_route = build_route(net, ego_edge, min_edges=2, max_edges=4)
        onc_route = build_route(net, onc_edge, min_edges=2, max_edges=4)
        if ego_route is None or onc_route is None:
            continue

        ego_len = ego_edge.getLength()
        onc_len = onc_edge.getLength()

        center_ratio = random.uniform(0.20, 0.55)
        spacing = random.uniform(12, 24)
        ego_center = center_ratio * ego_len

        ego_positions = [
            clamp(ego_center - spacing / 2.0, 5, ego_len - 10),
            clamp(ego_center + spacing / 2.0, 5, ego_len - 10),
        ]
        if abs(ego_positions[1] - ego_positions[0]) < 8:
            continue

        ego_speeds = [
            random.uniform(*profile["ego_speed"])
            for _ in ego_positions
        ]
        onc_speeds = [
            random.uniform(*profile["oncoming_speed"])
            for _ in ego_positions
        ]
        desired_distance_at_attack = random.uniform(*profile["distance"])
        attack_trigger_time = 2.0
        onc_positions = []
        for index, p in enumerate(ego_positions):
            ratio = p / max(ego_len, 1.0)
            mirrored_position = (1.0 - ratio) * onc_len
            closing_distance = (
                ego_speeds[index] + onc_speeds[index]
            ) * attack_trigger_time
            initial_distance = desired_distance_at_attack + closing_distance
            q = mirrored_position - initial_distance + random.uniform(-1.5, 1.5)
            if q < 5 or q > onc_len - 10:
                onc_positions = []
                break
            onc_positions.append(q)

        if len(onc_positions) != len(ego_positions):
            continue

        cluster = VehicleCluster(
            "oncoming",
            f"对向群组 @ {ego_edge.getID()} / {onc_edge.getID()}",
            focus_key=f"{ego_edge.getID()}|{onc_edge.getID()}"
        )


        cluster.description = (
            f"对向交互[{profile['name']}] @ "
            f"{ego_edge.getID()} / {onc_edge.getID()}"
        )
        cluster.focus_key = (
            f"{ego_edge.getID()}|{onc_edge.getID()}|"
            f"profile={profile['name']}"
        )

        vid = vid_start
        if ego_edge.getLaneNumber() >= 3 and random.random() < 0.35:
            ego_lane = ego_edge.getLaneNumber() - 2
        else:
            ego_lane = ego_edge.getLaneNumber() - 1
        onc_lane = onc_edge.getLaneNumber() - 1

        ego_route_str = route_to_str(ego_route)
        onc_route_str = route_to_str(onc_route)

        for index, p in enumerate(ego_positions):
            cluster.add(
                vid=vid,
                edge_route_str=ego_route_str,
                lane=ego_lane,
                pos=p,
                speed=ego_speeds[index],
                depart=sample_depart(args),
                edge_len=ego_len,
            )
            vid += 1

        for index, q in enumerate(onc_positions):
            cluster.add(
                vid=vid,
                edge_route_str=onc_route_str,
                lane=onc_lane,
                pos=q,
                speed=onc_speeds[index],
                depart=sample_depart(args),
                edge_len=onc_len,
            )
            vid += 1

        if len(cluster.vehicles) >= 4:
            cluster.choose_ego()
            key = pair_key(ego_edge, onc_edge)
            pair_usage[key] = pair_usage.get(key, 0) + 1
            used["oncoming_profile_index"] = profile_index + 1
            return cluster, vid

    return None, vid_start


def generate_junction_cluster(net, analysis, vid_start, used, args):
    junction_edges = list(analysis["junction"])
    random.shuffle(junction_edges)

    node_edges = {}
    for e in junction_edges:
        nid = e.getToNode().getID()
        node_edges.setdefault(nid, []).append(e)

    node_items = []
    for nid, edges in node_edges.items():
        if nid in used["junction_nodes"]:
            continue
        if len(edges) >= 2:
            node_items.append((nid, edges))

    node_items.sort(key=lambda item: -len(item[1]))

    for nid, edges in prioritized_randomized(node_items, topk=40):
        cluster = VehicleCluster(
            "junction",
            f"交叉口群组 @ node {nid} ({len(edges)}条入边)",
            focus_key=nid
        )

        vid = vid_start
        used_approaches = set()

        candidate_edges = sorted(edges, key=lambda e: -e.getLength())
        candidate_edges = prioritized_randomized(candidate_edges, topk=6)
        selected = candidate_edges[:min(4, len(candidate_edges))]

        for edge in selected:
            route = build_route(net, edge, min_edges=2, max_edges=4)
            if route is None:
                continue

            edge_len = edge.getLength()
            route_str = route_to_str(route)
            lane = random.randint(0, edge.getLaneNumber() - 1)

            num_cars = random.randint(1, 2)
            spacing = random.uniform(10, 18)
            start_pos = edge_len - random.uniform(12, 28)

            for j in range(num_cars):
                pos = max(5, start_pos - j * spacing)
                cluster.add(
                    vid=vid,
                    edge_route_str=route_str,
                    lane=lane,
                    pos=pos,
                    speed=random.uniform(6, 10),
                    depart=sample_depart(args),
                    edge_len=edge_len,
                )
                vid += 1

            used_approaches.add(edge.getID())

        if len(cluster.vehicles) >= 3 and len(used_approaches) >= 2:
            cluster.choose_ego()
            used["junction_nodes"].add(nid)
            return cluster, vid

    return None, vid_start


def generate_background_vehicles(net, analysis, vid_start, args, count=0):
    """
    可选背景车，仅用于让单场景更自然。
    默认 count=0 时不生成。
    """
    if count <= 0:
        return None, vid_start

    cluster = VehicleCluster("background", f"背景车辆 x{count}")
    vid = vid_start
    attempts = 0
    max_attempts = max(20, count * 10)

    while len(cluster.vehicles) < count and attempts < max_attempts:
        attempts += 1
        edge = random.choice(analysis["all"])
        route = build_route(net, edge, min_edges=2, max_edges=4)
        if route is None:
            continue

        edge_len = edge.getLength()
        if edge_len <= 15:
            continue

        cluster.add(
            vid=vid,
            edge_route_str=route_to_str(route),
            lane=random.randint(0, edge.getLaneNumber() - 1),
            pos=random.uniform(5, max(6, edge_len - 10)),
            speed=random.uniform(5, 13),
            depart=sample_depart(args),
            edge_len=edge_len,
            role="npc",
        )
        vid += 1

    if cluster.vehicles:
        cluster.description = f"背景车辆 x{len(cluster.vehicles)}"
        return cluster, vid

    return None, vid_start


# ═══════════════════════════════════════════════════════════
# 输出
# ═══════════════════════════════════════════════════════════

def build_scene_xml(scene_cluster, background_cluster=None):
    clusters = [scene_cluster]
    if background_cluster is not None and background_cluster.vehicles:
        clusters.append(background_cluster)

    all_vehicles = []
    for c in clusters:
        for v in c.vehicles:
            vv = dict(v)
            vv["_cluster_type"] = c.cluster_type
            vv["_cluster_desc"] = c.description
            vv["_scene_id"] = scene_cluster.scene_id
            vv["_ego_id"] = scene_cluster.ego_id
            all_vehicles.append(vv)

    # 优化插入顺序：同一 edge/lane 上先插前车，再插后车
    all_vehicles.sort(
        key=lambda v: (
            float(v["depart"]),
            v["start_edge"],
            int(v["lane"]),
            -float(v["pos"]),
            int(v["id"]),
        )
    )

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
        '',
        '    <vType id="default_type" vClass="passenger" guiShape="passenger" length="4.5" minGap="1.5" tau="0.8" maxSpeed="13.89" accel="2.6" decel="4.5" sigma="0.1" color="255,255,255"/>',
        '    <vType id="ego_type" vClass="passenger" guiShape="passenger" length="4.5" minGap="1.5" tau="0.8" maxSpeed="13.89" accel="2.6" decel="4.5" sigma="0.1" color="255,0,0"/>',
        '    <vType id="attacker_type" vClass="passenger" guiShape="passenger" length="4.5" minGap="1.5" tau="0.8" maxSpeed="13.89" accel="2.8" decel="5.0" sigma="0.05" color="255,128,0"/>',
        '',
        f'    <!-- scene_id={scene_cluster.scene_id} | ego_id={scene_cluster.ego_id} | attacker_id={scene_cluster.attacker_candidate_id} | attack_type={scene_cluster.attack_type} | {scene_cluster.description} -->',
    ]

    for v in all_vehicles:
        if v["role"] == "ego":
            vtype = "ego_type"
        elif v["role"] == "attacker_candidate":
            vtype = "attacker_type"
        else:
            vtype = "default_type"

        lines.append(
            f'    <vehicle id="{v["id"]}" type="{vtype}" depart="{v["depart"]:.2f}" '
            f'departLane="{v["lane"]}" departPos="{v["pos"]:.1f}" departSpeed="{v["speed"]:.1f}">'
        )
        lines.append(f'        <route edges="{v["route"]}"/>')
        lines.append(f'        <param key="scene_id" value="{scene_cluster.scene_id}"/>')
        lines.append(f'        <param key="scenario_type" value="{scene_cluster.cluster_type}"/>')
        if scene_cluster.attack_type:
            lines.append(f'        <param key="attack_type" value="{scene_cluster.attack_type}"/>')
        if scene_cluster.attacker_candidate_id is not None:
            lines.append(f'        <param key="attacker_id" value="{scene_cluster.attacker_candidate_id}"/>')
        lines.append(f'        <param key="role" value="{v["role"]}"/>')
        lines.append(f'        <param key="scene_ego_id" value="{scene_cluster.ego_id}"/>')
        lines.append('    </vehicle>')

    lines.append('')
    lines.append('</routes>')
    return "\n".join(lines), all_vehicles


def generate_sumocfg(scene_dir, scene_name, net_path, duration=10.0):
    net_rel = relpath_for_cfg(net_path, scene_dir)
    route_rel = f"{scene_name}.rou.xml"

    net_dir = os.path.dirname(os.path.abspath(net_path))
    view_path = os.path.join(net_dir, "osm.view.xml")

    gui_block = ""
    if os.path.exists(view_path):
        view_rel = relpath_for_cfg(view_path, scene_dir)
        gui_block = (
            "    <gui_only>\n"
            f'        <gui-settings-file value="{view_rel}"/>\n'
            "    </gui_only>\n"
        )

    content = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{net_rel}"/>
        <route-files value="{route_rel}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{float(duration):.1f}"/>
    </time>
    <processing>
        <ignore-route-errors value="true"/>
        <collision.action value="warn"/>
        <collision.check-junctions value="true"/>
    </processing>

{gui_block}</configuration>'''

    cfg_path = os.path.join(scene_dir, f"{scene_name}.sumocfg")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(content)
    return cfg_path


def write_scene_files(
    output_root, net_path, scene_cluster, background_cluster=None, duration=10.0
):
    if scene_cluster.scene_id is None:
        raise ValueError("scene_id must be assigned before writing scene files")

    scene_name = str(scene_cluster.scene_id)
    scene_dir = os.path.join(output_root, scene_name)
    os.makedirs(scene_dir, exist_ok=True)

    xml_str, all_vehicles = build_scene_xml(scene_cluster, background_cluster)

    rou_path = os.path.join(scene_dir, f"{scene_name}.rou.xml")
    with open(rou_path, "w", encoding="utf-8") as f:
        f.write(xml_str)

    cfg_path = generate_sumocfg(scene_dir, scene_name, net_path, duration)

    scene_json = {
        "scene_id": scene_cluster.scene_id,
        "scenario_type": scene_cluster.cluster_type,
        "attack_type": scene_cluster.attack_type,
        "ego_id": scene_cluster.ego_id,
        "attacker_candidate_id": scene_cluster.attacker_candidate_id,
        "attack_type_cn": ATTACK_TYPE_CN.get(scene_cluster.attack_type, scene_cluster.attack_type),
        "description": scene_cluster.description,
        "focus_key": scene_cluster.focus_key,
        "scene_vehicle_ids": scene_cluster.vehicle_ids(),
        "background_vehicle_ids": background_cluster.vehicle_ids() if background_cluster else [],
        "vehicle_count": len(all_vehicles),
        "depart_range": scene_cluster.depart_range(),
        "generated_files": {
            "route_file": to_unix_path(rou_path),
            "config_file": to_unix_path(cfg_path),
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    json_path = os.path.join(scene_dir, f"{scene_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(scene_json, f, indent=2, ensure_ascii=False)

    return {
        "scene_id": scene_cluster.scene_id,
        "scenario_type": scene_cluster.cluster_type,
        "attack_type": scene_cluster.attack_type,
        "ego_id": scene_cluster.ego_id,
        "attacker_candidate_id": scene_cluster.attacker_candidate_id,
        "attack_type_cn": ATTACK_TYPE_CN.get(scene_cluster.attack_type, scene_cluster.attack_type),
        "description": scene_cluster.description,
        "focus_key": scene_cluster.focus_key,
        "vehicle_count": len(all_vehicles),
        "scene_vehicle_count": len(scene_cluster.vehicles),
        "background_vehicle_count": len(background_cluster.vehicles) if background_cluster else 0,
        "depart_range": scene_cluster.depart_range(),
        "scene_dir": to_unix_path(scene_dir),
        "route_file": to_unix_path(rou_path),
        "config_file": to_unix_path(cfg_path),
        "json_file": to_unix_path(json_path),
    }


def write_scene_index(output_root, args, analysis, scene_records):
    index = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "network_file": args.net,
            "output_dir": args.output,
            "seed": args.seed,
            "depart_window": args.depart_window,
            "background_per_scene": args.background_per_scene,
            "requested_scene_counts": {
                "following": args.num_following,
                "parallel": args.num_parallel,
                "oncoming": args.num_oncoming,
                "junction": args.num_junction,
            },
            "scenario_attack_mapping": SCENARIO_ATTACK_TYPES,
            "path_template_hint": {
                "sumo_cfg_file": "networkFiles/{map_name}/scenarios_by_ego/{scene_id}/{scene_id}.sumocfg",
                "sumo_rou_file": "networkFiles/{map_name}/scenarios_by_ego/{scene_id}/{scene_id}.rou.xml",
            }
        },
        "network_analysis": {
            "total_edges": len(analysis["all"]),
            "long_edges_gt60m": len(analysis["long"]),
            "multilane_edges": len(analysis["multilane"]),
            "bidirectional_pairs": len(analysis["bidirectional"]),
            "junction_edges": len(analysis["junction"]),
        },
        "statistics": {
            "total_scenes": len(scene_records),
            "scene_ids": [r["scene_id"] for r in scene_records],
            "ego_ids": [r["ego_id"] for r in scene_records],
            "attacker_candidate_ids": [r["attacker_candidate_id"] for r in scene_records],
            "count_by_type": {},
            "count_by_attack_type": {},
        },
        "scenes": scene_records,
        "scene_id_to_scene": {},
        "ego_to_scene": {},
    }

    count_by_type = {}
    count_by_attack_type = {}
    for r in scene_records:
        count_by_type[r["scenario_type"]] = count_by_type.get(r["scenario_type"], 0) + 1
        attack_type = r.get("attack_type")
        if attack_type:
            count_by_attack_type[attack_type] = count_by_attack_type.get(attack_type, 0) + 1
        index["scene_id_to_scene"][str(r["scene_id"])] = r
        index["ego_to_scene"][str(r["ego_id"])] = r

    index["statistics"]["count_by_type"] = count_by_type
    index["statistics"]["count_by_attack_type"] = count_by_attack_type

    index_path = os.path.join(output_root, "scene_index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    return index_path


def assign_scene_id(cluster, counters):
    scene_type = cluster.attack_type or cluster.cluster_type
    counters[scene_type] = counters.get(scene_type, 0) + 1
    cluster.scene_id = f"{scene_type}_{counters[scene_type]:03d}"


def print_scene_summary(scene_records):
    print("\n🧭 单场景摘要")
    print("=" * 100)
    for r in scene_records:
        dr = r["depart_range"]
        dr_text = f"{dr[0]:.2f}-{dr[1]:.2f}" if dr else "N/A"
        print(
            f"  {r['scene_id']:<32} ego={str(r['ego_id']):<4} "
            f"n={r['vehicle_count']:<2} "
            f"depart={dr_text:<12} "
            f"{r['description']}"
        )
    print("=" * 100)


# ═══════════════════════════════════════════════════════════
# 主逻辑
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="生成单场景独立文件的攻击/交互场景库")
    parser.add_argument("--net", required=True, help="路网文件")
    parser.add_argument("--output", default="scenarios_by_ego", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    parser.add_argument("--num-following", type=int, default=5, help="跟车交互场景数量，对应 brake_check")
    parser.add_argument("--num-parallel", type=int, default=5, help="并行/邻车道交互场景数量，对应 cut_in")
    parser.add_argument("--num-oncoming", type=int, default=5, help="对向交互场景数量，对应 opposite_direction_intrusion")
    parser.add_argument("--num-junction", type=int, default=0, help="交叉口交互场景数量；当前三类攻击默认不生成")

    parser.add_argument(
        "--depart-window",
        type=float,
        default=0.0,
        help="车辆出发时间窗口。默认 0.0，表示所有车辆在 t=0 出现"
    )
    parser.add_argument("--duration", type=float, default=10.0, help="场景仿真时长，单位秒")
    parser.add_argument("--background-per-scene", type=int, default=0, help="每个场景附加背景车数量，默认0")
    parser.add_argument("--clean-output", action="store_true", help="如果输出目录已存在则停止，并提示手动清理")

    args = parser.parse_args()

    if not os.path.exists(args.net):
        parser.error(f"路网文件不存在: {args.net}")

    if args.clean_output and os.path.exists(args.output):
        parser.error(
            f"输出目录已存在: {args.output}。根据项目规则不自动批量删除，请手动清理后重试。"
        )

    os.makedirs(args.output, exist_ok=True)
    random.seed(args.seed)

    net = sumolib.net.readNet(args.net)
    analysis = analyze_network(net)

    # 按场景类型分别去重，跨类型允许复用
    used = {
        "following_edges": set(),
        "parallel_edges": set(),
        "oncoming_pairs": set(),
        "junction_nodes": set(),
    }

    vid = 1
    scene_records = []
    scene_counters = {}

    plan = [
        ("following", generate_following_cluster, max(0, args.num_following)),
        ("parallel", generate_diverse_parallel_cluster, max(0, args.num_parallel)),
        ("oncoming", generate_oncoming_cluster, max(0, args.num_oncoming)),
        ("junction", generate_junction_cluster, max(0, args.num_junction)),
    ]

    for scene_type, generator, count in plan:
        for i in range(count):
            cluster, vid = generator(net, analysis, vid, used, args)
            if cluster is None:
                print(f"⚠️  未能生成 {scene_type} 场景 #{i + 1}，可能是该类独特道路资源不足。")
                continue

            assign_scene_id(cluster, scene_counters)
            cluster.choose_attacker_candidate()
            if cluster.attack_type and cluster.attacker_candidate_id is None:
                print(f"⚠️  未能为 {cluster.scene_id} 选择攻击候选车，跳过该场景。")
                continue

            bg_cluster = None
            if args.background_per_scene > 0:
                bg_cluster, vid = generate_background_vehicles(
                    net, analysis, vid, args, count=args.background_per_scene
                )

            rec = write_scene_files(
                args.output, args.net, cluster, bg_cluster, duration=args.duration
            )
            scene_records.append(rec)

    if not scene_records:
        print("❌ 没有生成任何场景，请减少数量或放宽约束。")
        sys.exit(1)

    index_path = write_scene_index(args.output, args, analysis, scene_records)
    print_scene_summary(scene_records)

    print("\n🎉 生成完毕！")
    print(f"   场景总数: {len(scene_records)}")
    print(f"   索引文件: {to_unix_path(index_path)}")
    print(f"   输出目录: {to_unix_path(args.output)}")
    print("\n可直接使用的 scene_id 列表:")
    print("  " + ", ".join(str(r["scene_id"]) for r in scene_records))
    print("\n对应 ego_id 列表:")
    print("  " + ", ".join(str(r["ego_id"]) for r in scene_records))


if __name__ == "__main__":
    main()
