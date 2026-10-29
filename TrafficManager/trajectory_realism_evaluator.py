#!/usr/bin/env python3
"""Offline realism evaluation for executed adversarial vehicle trajectories.

The evaluator reads the 10 Hz states stored in each result directory's
``limsim.db``.  It deliberately evaluates the trajectory that SUMO actually
executed, rather than the initially planned trajectory in ``attack_record``.

For every adversarial method, the corresponding vehicle in the benign result
for the same scene is used as the reference.  The program reports:

* one-dimensional Wasserstein distances for speed, longitudinal acceleration,
  lateral acceleration, longitudinal jerk, and yaw rate;
* data-driven kinematic violation rates, using a configurable percentile of
  the pooled benign reference distribution;
* an attacker lane-membership/off-road proxy;
* ego-attacker collision rate, collision-normal relative velocity, head-on
  rate, and severe head-on rate.

No figures are produced.  CSV, JSON, and plain-text reports are written to the
requested output directory.

Nominal methods can borrow vehicle selection metadata with --selection-source.
Use --thresholds-from to retain an existing benign calibration when extending
an earlier comparison with additional methods.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCENE_RE = re.compile(
    r"(?P<scene>(?:brake_check|cut_in|opposite_direction_intrusion)_\d{3})"
)
PROFILE_NAMES = (
    "speed",
    "longitudinal_acceleration",
    "lateral_acceleration",
    "longitudinal_jerk",
    "yaw_rate",
)
ABS_THRESHOLD_PROFILES = set(PROFILE_NAMES) - {"speed"}
ATTACKER_KEYS = ("attacker_id", "target_vehicle_id", "selected_candidate")
TRIGGER_KEYS = (
    "actual_injection_time",
    "attack_trigger_time",
    "trigger_time",
    "injection_time",
)
METADATA_FILENAMES = (
    "attack_record.json",
    "challenger_attack_record.json",
    "closed_loop_status.json",
    "scene_summary.json",
    "best_candidate.json",
)


@dataclass(frozen=True)
class VehicleState:
    frame: int
    x: float
    y: float
    yaw: float
    speed: float
    accel: float
    lane_id: str


@dataclass
class Trace:
    scene_id: str
    scene_dir: Path
    attacker_id: str
    ego_id: str
    trigger_time: Optional[float]
    states: List[VehicleState]
    ego_states: Dict[int, VehicleState]
    attacker_length: float
    attacker_width: float
    ego_length: float
    ego_width: float
    profiles: Dict[str, np.ndarray]


def canonical_scene_id(path_or_name: object) -> Optional[str]:
    match = SCENE_RE.search(str(path_or_name))
    return match.group("scene") if match else None


def parse_method_spec(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            "method must use LABEL=RESULT_DIRECTORY syntax"
        )
    label, raw_path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("method label cannot be empty")
    return label, Path(raw_path).expanduser().resolve()


def find_scene_directories(root: Path) -> Tuple[Dict[str, Path], List[str]]:
    """Map canonical scene IDs to directories containing limsim.db.

    If reruns exist, the most recently modified database is selected and a
    warning is returned.  No file is modified.
    """
    if not root.exists():
        raise FileNotFoundError(f"Result directory does not exist: {root}")

    candidates: Dict[str, List[Path]] = {}
    if root.is_file() and root.name == "limsim.db":
        databases = [root]
    else:
        databases = list(root.rglob("limsim.db"))

    for database in databases:
        scene_id = canonical_scene_id(database.parent)
        if scene_id:
            candidates.setdefault(scene_id, []).append(database.parent)

    selected: Dict[str, Path] = {}
    warnings: List[str] = []
    for scene_id, directories in sorted(candidates.items()):
        directories.sort(
            key=lambda directory: (directory / "limsim.db").stat().st_mtime,
            reverse=True,
        )
        selected[scene_id] = directories[0]
        if len(directories) > 1:
            warnings.append(
                f"{scene_id}: found {len(directories)} runs under {root}; "
                f"using newest {directories[0]}"
            )
    return selected, warnings


def iter_nested_items(value: object) -> Iterable[Tuple[str, object]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from iter_nested_items(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_nested_items(child)


def metadata_files(scene_dir: Path) -> List[Path]:
    found: List[Path] = []
    for filename in METADATA_FILENAMES:
        direct = scene_dir / filename
        if direct.is_file():
            found.append(direct)
        for path in scene_dir.glob(f"*/{filename}"):
            if path.is_file():
                found.append(path)
    return list(dict.fromkeys(found))


def read_metadata(scene_dir: Path) -> Tuple[List[object], List[str]]:
    documents: List[object] = []
    warnings: List[str] = []
    for path in metadata_files(scene_dir):
        try:
            documents.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            warnings.append(f"could not read metadata {path}: {exc}")
    return documents, warnings


def first_nested_scalar(documents: Sequence[object], keys: Sequence[str]) -> object:
    key_set = set(keys)
    for document in documents:
        for key, value in iter_nested_items(document):
            if key in key_set and isinstance(value, (str, int, float)):
                if str(value).strip() not in {"", "None", "null"}:
                    return value
    return None


def resolve_attacker_id(scene_dir: Path, documents: Sequence[object]) -> Optional[str]:
    value = first_nested_scalar(documents, ATTACKER_KEYS)
    return None if value is None else str(value)


def resolve_trigger_time(documents: Sequence[object]) -> Optional[float]:
    value = first_nested_scalar(documents, TRIGGER_KEYS)
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result >= 0.0 else None


def connect_read_only(database: Path) -> sqlite3.Connection:
    uri = database.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def database_ego_id(connection: sqlite3.Connection) -> Optional[str]:
    row = connection.execute("SELECT egoID FROM simINFO LIMIT 1").fetchone()
    if row and row[0] is not None:
        return str(row[0])
    row = connection.execute(
        "SELECT vid FROM frameINFO WHERE vtag = 'ego' LIMIT 1"
    ).fetchone()
    return str(row[0]) if row else None


def vehicle_dimensions(
    connection: sqlite3.Connection, vehicle_id: str
) -> Tuple[float, float]:
    row = connection.execute(
        "SELECT length, width FROM vehicleINFO WHERE vid = ? LIMIT 1",
        (vehicle_id,),
    ).fetchone()
    if not row:
        # Conservative passenger-car fallback, explicitly exposed in warnings
        # by the caller through the returned dimensions.
        return 4.8, 2.0
    return float(row[0]), float(row[1])


def load_states(
    connection: sqlite3.Connection,
    vehicle_id: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> List[VehicleState]:
    clauses = ["vid = ?"]
    parameters: List[object] = [vehicle_id]
    if start_frame is not None:
        clauses.append("frame >= ?")
        parameters.append(int(start_frame))
    if end_frame is not None:
        clauses.append("frame <= ?")
        parameters.append(int(end_frame))
    rows = connection.execute(
        "SELECT frame, x, y, yaw, speed, accel, laneID "
        "FROM frameINFO WHERE " + " AND ".join(clauses) + " ORDER BY frame",
        parameters,
    ).fetchall()
    return [
        VehicleState(
            frame=int(row[0]),
            x=float(row[1]),
            y=float(row[2]),
            yaw=float(row[3]),
            speed=float(row[4]),
            accel=float(row[5]),
            lane_id=str(row[6] or ""),
        )
        for row in rows
    ]


def consecutive_pairs(states: Sequence[VehicleState]) -> Iterable[Tuple[VehicleState, VehicleState]]:
    for previous, current in zip(states[:-1], states[1:]):
        if current.frame > previous.frame:
            yield previous, current


def contiguous_segments(states: Sequence[VehicleState]) -> List[List[VehicleState]]:
    """Split states so derivatives never bridge missing database frames."""
    if not states:
        return []
    segments: List[List[VehicleState]] = [[states[0]]]
    for state in states[1:]:
        if state.frame == segments[-1][-1].frame + 1:
            segments[-1].append(state)
        else:
            segments.append([state])
    return segments


def centered_moving_average(values: np.ndarray, requested_window: int) -> np.ndarray:
    """Edge-preserving centered smoothing with an odd window."""
    values = np.asarray(values, dtype=float)
    if len(values) < 3 or requested_window <= 1:
        return values.copy()
    window = min(requested_window, len(values))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def build_profiles(
    states: Sequence[VehicleState],
    step_time: float,
    smoothing_window: int,
) -> Dict[str, np.ndarray]:
    """Build smoothed kinematic profiles from executed positions and headings.

    LimSim's stored acceleration can contain one-step controller impulses.  To
    avoid reporting those numerical impulses as human motion, speed and
    unwrapped yaw are smoothed first.  Acceleration, jerk, and yaw rate are then
    derived from the smoothed signals.  A five-frame window corresponds to
    0.5 seconds at the project's 10 Hz simulation frequency.
    """
    collected: Dict[str, List[np.ndarray]] = {name: [] for name in PROFILE_NAMES}
    for segment in contiguous_segments(states):
        if len(segment) < 3:
            continue
        speed_raw = np.asarray([state.speed for state in segment], dtype=float)
        yaw_raw = np.unwrap(np.asarray([state.yaw for state in segment], dtype=float))
        speed = centered_moving_average(speed_raw, smoothing_window)
        yaw = centered_moving_average(yaw_raw, smoothing_window)

        edge_order = 2 if len(segment) >= 3 else 1
        longitudinal_acceleration = np.gradient(
            speed, step_time, edge_order=edge_order
        )
        # Smooth the first derivative once before differentiating again.  This
        # prevents a single controller step from dominating the jerk profile.
        acceleration_for_jerk = centered_moving_average(
            longitudinal_acceleration, smoothing_window
        )
        longitudinal_jerk = np.gradient(
            acceleration_for_jerk, step_time, edge_order=edge_order
        )
        yaw_rate = np.gradient(yaw, step_time, edge_order=edge_order)
        lateral_acceleration = speed * yaw_rate

        collected["speed"].append(speed)
        collected["longitudinal_acceleration"].append(longitudinal_acceleration)
        collected["lateral_acceleration"].append(lateral_acceleration)
        collected["longitudinal_jerk"].append(longitudinal_jerk)
        collected["yaw_rate"].append(yaw_rate)

    profiles = {
        name: (
            np.concatenate(arrays) if arrays else np.asarray([], dtype=float)
        )
        for name, arrays in collected.items()
    }
    return {
        name: values[np.isfinite(values)]
        for name, values in profiles.items()
    }


def load_trace(
    scene_id: str,
    scene_dir: Path,
    attacker_id: str,
    trigger_time: Optional[float],
    step_time: float,
    smoothing_window: int,
    end_frame: Optional[int] = None,
) -> Trace:
    database = scene_dir / "limsim.db"
    start_frame = (
        int(math.ceil(trigger_time / step_time - 1e-9))
        if trigger_time is not None
        else None
    )
    with connect_read_only(database) as connection:
        ego_id = database_ego_id(connection)
        if ego_id is None:
            raise ValueError(f"{database}: ego vehicle ID not found")
        states = load_states(connection, attacker_id, start_frame, end_frame)
        ego_states = {
            state.frame: state
            for state in load_states(connection, ego_id, start_frame, end_frame)
        }
        attacker_length, attacker_width = vehicle_dimensions(connection, attacker_id)
        ego_length, ego_width = vehicle_dimensions(connection, ego_id)

    return Trace(
        scene_id=scene_id,
        scene_dir=scene_dir,
        attacker_id=attacker_id,
        ego_id=ego_id,
        trigger_time=trigger_time,
        states=states,
        ego_states=ego_states,
        attacker_length=attacker_length,
        attacker_width=attacker_width,
        ego_length=ego_length,
        ego_width=ego_width,
        profiles=build_profiles(states, step_time, smoothing_window),
    )


def wasserstein_1d(first: np.ndarray, second: np.ndarray) -> float:
    """Exact empirical first Wasserstein distance without SciPy dependency."""
    first = np.sort(np.asarray(first, dtype=float))
    second = np.sort(np.asarray(second, dtype=float))
    if len(first) == 0 or len(second) == 0:
        return float("nan")
    support = np.sort(np.concatenate([first, second]))
    if len(support) < 2:
        return 0.0
    widths = np.diff(support)
    first_cdf = np.searchsorted(first, support[:-1], side="right") / len(first)
    second_cdf = np.searchsorted(second, support[:-1], side="right") / len(second)
    return float(np.sum(np.abs(first_cdf - second_cdf) * widths))


def rectangle_corners(state: VehicleState, length: float, width: float) -> np.ndarray:
    local = np.asarray(
        [
            [length / 2.0, width / 2.0],
            [length / 2.0, -width / 2.0],
            [-length / 2.0, -width / 2.0],
            [-length / 2.0, width / 2.0],
        ],
        dtype=float,
    )
    cosine, sine = math.cos(state.yaw), math.sin(state.yaw)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + np.asarray([state.x, state.y])


def polygons_overlap_sat(first: np.ndarray, second: np.ndarray) -> bool:
    for polygon in (first, second):
        edges = np.roll(polygon, -1, axis=0) - polygon
        for edge in edges:
            axis = np.asarray([-edge[1], edge[0]], dtype=float)
            norm = np.linalg.norm(axis)
            if norm <= 1e-12:
                continue
            axis /= norm
            first_projection = first @ axis
            second_projection = second @ axis
            if (
                first_projection.max() < second_projection.min()
                or second_projection.max() < first_projection.min()
            ):
                return False
    return True


def wrapped_heading_difference(first: float, second: float) -> float:
    return abs(math.atan2(math.sin(first - second), math.cos(first - second)))


def collision_plausibility(
    trace: Trace,
    head_on_angle_degrees: float,
    severe_relative_speed: float,
) -> Dict[str, object]:
    for attacker in trace.states:
        ego = trace.ego_states.get(attacker.frame)
        if ego is None:
            continue
        attacker_polygon = rectangle_corners(
            attacker, trace.attacker_length, trace.attacker_width
        )
        ego_polygon = rectangle_corners(ego, trace.ego_length, trace.ego_width)
        if not polygons_overlap_sat(attacker_polygon, ego_polygon):
            continue

        center_delta = np.asarray([attacker.x - ego.x, attacker.y - ego.y])
        center_norm = float(np.linalg.norm(center_delta))
        if center_norm <= 1e-9:
            normal = np.asarray([math.cos(ego.yaw), math.sin(ego.yaw)])
        else:
            normal = center_delta / center_norm
        attacker_velocity = attacker.speed * np.asarray(
            [math.cos(attacker.yaw), math.sin(attacker.yaw)]
        )
        ego_velocity = ego.speed * np.asarray([math.cos(ego.yaw), math.sin(ego.yaw)])
        relative_normal_speed = abs(
            float(np.dot(attacker_velocity - ego_velocity, normal))
        )
        heading_difference = wrapped_heading_difference(attacker.yaw, ego.yaw)
        head_on = heading_difference >= math.radians(head_on_angle_degrees)
        return {
            "attacker_collision": True,
            "collision_frame": attacker.frame,
            "relative_collision_speed": relative_normal_speed,
            "heading_difference_degrees": math.degrees(heading_difference),
            "head_on_collision": bool(head_on),
            "severe_head_on_collision": bool(
                head_on and relative_normal_speed > severe_relative_speed
            ),
        }
    return {
        "attacker_collision": False,
        "collision_frame": None,
        "relative_collision_speed": None,
        "heading_difference_degrees": None,
        "head_on_collision": False,
        "severe_head_on_collision": False,
    }


def finite_summary(values: Iterable[object]) -> Dict[str, object]:
    array = np.asarray(
        [float(value) for value in values if value is not None], dtype=float
    )
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"n": 0, "mean": None, "std": None}
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def compute_thresholds(
    references: Sequence[Trace], percentile: float
) -> Dict[str, float]:
    thresholds: Dict[str, float] = {}
    for profile_name in PROFILE_NAMES:
        arrays = [trace.profiles[profile_name] for trace in references]
        arrays = [array for array in arrays if len(array)]
        if not arrays:
            raise ValueError(f"No benign reference samples for {profile_name}")
        values = np.concatenate(arrays)
        if profile_name in ABS_THRESHOLD_PROFILES:
            values = np.abs(values)
        thresholds[profile_name] = float(np.percentile(values, percentile))
    return thresholds


def scene_metrics(
    trace: Trace,
    reference: Trace,
    thresholds: Mapping[str, float],
    head_on_angle_degrees: float,
    severe_relative_speed: float,
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "scene_id": trace.scene_id,
        "attacker_id": trace.attacker_id,
        "trigger_time": trace.trigger_time,
        "num_attacker_states": len(trace.states),
        "num_reference_states": len(reference.states),
        "attacker_offroad_frame_rate": (
            float(np.mean([not state.lane_id.strip() for state in trace.states]))
            if trace.states
            else None
        ),
    }

    total_profile_samples = 0
    total_profile_violations = 0
    any_violation = False
    for profile_name in PROFILE_NAMES:
        values = trace.profiles[profile_name]
        reference_values = reference.profiles[profile_name]
        result[f"wd_{profile_name}"] = wasserstein_1d(values, reference_values)

        comparison = (
            np.abs(values)
            if profile_name in ABS_THRESHOLD_PROFILES
            else values
        )
        violations = comparison > thresholds[profile_name]
        violation_count = int(np.sum(violations))
        result[f"violation_rate_{profile_name}"] = (
            float(np.mean(violations)) if len(violations) else None
        )
        total_profile_samples += len(violations)
        total_profile_violations += violation_count
        any_violation = any_violation or violation_count > 0

    result["kinematic_violation_rate"] = (
        total_profile_violations / total_profile_samples
        if total_profile_samples
        else None
    )
    result["scene_has_kinematic_violation"] = float(any_violation)
    result.update(
        collision_plausibility(
            trace,
            head_on_angle_degrees=head_on_angle_degrees,
            severe_relative_speed=severe_relative_speed,
        )
    )
    return result


def aggregate_method(
    label: str,
    rows: Sequence[Mapping[str, object]],
    traces: Sequence[Trace],
    references: Sequence[Trace],
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "method": label,
        "num_scenes": len(rows),
    }
    for profile_name in PROFILE_NAMES:
        per_scene = finite_summary(row[f"wd_{profile_name}"] for row in rows)
        method_values = [trace.profiles[profile_name] for trace in traces]
        reference_values = [trace.profiles[profile_name] for trace in references]
        method_values = [value for value in method_values if len(value)]
        reference_values = [value for value in reference_values if len(value)]
        summary[f"wd_{profile_name}_scene_mean"] = per_scene["mean"]
        summary[f"wd_{profile_name}_scene_std"] = per_scene["std"]
        summary[f"wd_{profile_name}_pooled"] = (
            wasserstein_1d(
                np.concatenate(method_values), np.concatenate(reference_values)
            )
            if method_values and reference_values
            else None
        )

    for metric_name in (
        "kinematic_violation_rate",
        "scene_has_kinematic_violation",
        "attacker_offroad_frame_rate",
        "attacker_collision",
        "head_on_collision",
        "severe_head_on_collision",
    ):
        stats = finite_summary(row.get(metric_name) for row in rows)
        summary[f"{metric_name}_mean"] = stats["mean"]
        summary[f"{metric_name}_std"] = stats["std"]

    collision_speeds = [
        row.get("relative_collision_speed")
        for row in rows
        if bool(row.get("attacker_collision"))
    ]
    collision_speed_stats = finite_summary(collision_speeds)
    summary["relative_collision_speed_mean"] = collision_speed_stats["mean"]
    summary["relative_collision_speed_std"] = collision_speed_stats["std"]

    collision_rows = [row for row in rows if bool(row.get("attacker_collision"))]
    summary["head_on_rate_among_attacker_collisions"] = (
        float(np.mean([bool(row["head_on_collision"]) for row in collision_rows]))
        if collision_rows
        else None
    )
    summary["severe_head_on_rate_among_attacker_collisions"] = (
        float(
            np.mean(
                [bool(row["severe_head_on_collision"]) for row in collision_rows]
            )
        )
        if collision_rows
        else None
    )
    return summary


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(json_safe(dict(row)))


def format_number(value: object) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def write_text_report(
    path: Path,
    summaries: Sequence[Mapping[str, object]],
    thresholds: Mapping[str, float],
    percentile: float,
    smoothing_window: int,
    step_time: float,
    warnings: Sequence[str],
    thresholds_from: Optional[Path] = None,
    selection_sources: Sequence[Tuple[str, Path]] = (),
    method_references: Sequence[Tuple[str, Path]] = (),
) -> None:
    lines = [
        "Trajectory Realism Evaluation",
        "=============================",
        f"Reference threshold percentile: {percentile:g}",
        "Kinematic source: executed attacker states in limsim.db",
        f"Smoothing window: {smoothing_window} frames "
        f"({smoothing_window * step_time:.3f} s)",
        "Derivatives: recomputed from smoothed speed and unwrapped yaw",
        "Kinematic violation rate: fraction of profile samples exceeding thresholds in each scene, then equal-weight scene mean",
        "WD summary: equal-weight mean of per-scene distances (pooled values are separate)",
        f"Threshold source: {thresholds_from or 'pooled benign references from this run'}",
    ]
    for label, root in selection_sources:
        lines.append(f"Vehicle/trigger metadata source for {label}: {root}")
    for label, root in method_references:
        lines.append(f"Benign trajectory reference for {label}: {root}")
    for name in PROFILE_NAMES:
        lines.append(f"threshold.{name}: {thresholds[name]:.6f}")
    for summary in summaries:
        lines.extend(["", f"Method: {summary['method']}", "-" * 60])
        lines.append(f"Valid scenes: {summary['num_scenes']}")
        for name in PROFILE_NAMES:
            mean = summary[f"wd_{name}_scene_mean"]
            std = summary[f"wd_{name}_scene_std"]
            pooled = summary[f"wd_{name}_pooled"]
            lines.append(
                f"WD {name}: {format_number(mean)} +/- {format_number(std)} "
                f"(pooled={format_number(pooled)})"
            )
        lines.append(
            "Kinematic violation rate: "
            f"{format_number(summary['kinematic_violation_rate_mean'])} +/- "
            f"{format_number(summary['kinematic_violation_rate_std'])}"
        )
        lines.append(
            "Scenes with any kinematic violation: "
            f"{format_number(summary['scene_has_kinematic_violation_mean'])}"
        )
        lines.append(
            "Attacker off-road frame rate (lane-membership proxy): "
            f"{format_number(summary['attacker_offroad_frame_rate_mean'])} +/- "
            f"{format_number(summary['attacker_offroad_frame_rate_std'])}"
        )
        lines.append(
            "Ego-attacker collision rate: "
            f"{format_number(summary['attacker_collision_mean'])}"
        )
        lines.append(
            "Relative collision speed: "
            f"{format_number(summary['relative_collision_speed_mean'])} +/- "
            f"{format_number(summary['relative_collision_speed_std'])} m/s"
        )
        lines.append(
            "Head-on rate among ego-attacker collisions: "
            f"{format_number(summary['head_on_rate_among_attacker_collisions'])}"
        )
        lines.append(
            "Severe head-on rate among ego-attacker collisions: "
            f"{format_number(summary['severe_head_on_rate_among_attacker_collisions'])}"
        )
    if warnings:
        lines.extend(["", "Warnings", "--------"])
        lines.extend(f"- {warning}" for warning in warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    reference_dirs, warnings = find_scene_directories(args.reference)
    if not reference_dirs:
        raise ValueError(f"No limsim.db scene results found under {args.reference}")

    reference_overrides = dict(args.method_reference)
    reference_scene_maps = {}
    for root in set(reference_overrides.values()):
        scene_map, reference_warnings = find_scene_directories(root)
        if not scene_map:
            raise ValueError(f"No limsim.db scene results found under {root}")
        reference_scene_maps[root] = scene_map
        warnings.extend(reference_warnings)

    method_scene_maps: Dict[str, Dict[str, Path]] = {}
    for label, root in args.method:
        scene_map, method_warnings = find_scene_directories(root)
        method_scene_maps[label] = scene_map
        warnings.extend(method_warnings)

    # Nominal runs have no attacker metadata. An explicit donor selects the
    # same background vehicle and start time without copying attack states.
    selection_scene_maps: Dict[str, Dict[str, Path]] = {}
    for label, root in args.selection_source:
        scene_map, source_warnings = find_scene_directories(root)
        selection_scene_maps[label] = scene_map
        warnings.extend(source_warnings)

    loaded: Dict[str, List[Tuple[Trace, Trace]]] = {}
    threshold_reference_traces: Dict[Tuple[str, str, str, int, Optional[int]], Trace] = {}

    for label, scene_map in method_scene_maps.items():
        loaded[label] = []
        matching_references = (
            reference_scene_maps[reference_overrides[label]]
            if label in reference_overrides else reference_dirs
        )
        for scene_id, scene_dir in sorted(scene_map.items()):
            reference_dir = matching_references.get(scene_id)
            if reference_dir is None:
                warnings.append(f"{label}/{scene_id}: matching benign scene not found")
                continue
            selection_dir = scene_dir
            if label in selection_scene_maps:
                selection_dir = selection_scene_maps[label].get(scene_id)
                if selection_dir is None:
                    warnings.append(f"{label}/{scene_id}: selection source scene not found; skipped")
                    continue
            documents, metadata_warnings = read_metadata(selection_dir)
            warnings.extend(f"{label}/{scene_id}: {item}" for item in metadata_warnings)
            attacker_id = resolve_attacker_id(selection_dir, documents)
            if attacker_id is None:
                warnings.append(f"{label}/{scene_id}: attacker ID not found; skipped")
                continue
            trigger_time = resolve_trigger_time(documents)
            if trigger_time is None:
                if args.require_trigger_time:
                    warnings.append(
                        f"{label}/{scene_id}: trigger time not found; skipped"
                    )
                    continue
                warnings.append(
                    f"{label}/{scene_id}: trigger time not found; evaluating full run"
                )
            try:
                trace = load_trace(
                    scene_id,
                    scene_dir,
                    attacker_id,
                    trigger_time,
                    args.step_time,
                    args.smoothing_window,
                )
                if len(trace.states) < args.min_states:
                    warnings.append(
                        f"{label}/{scene_id}: only {len(trace.states)} attacker states; skipped"
                    )
                    continue
                end_frame = trace.states[-1].frame
                reference = load_trace(
                    scene_id,
                    reference_dir,
                    attacker_id,
                    trigger_time,
                    args.step_time,
                    args.smoothing_window,
                    end_frame=end_frame,
                )
                if len(reference.states) < args.min_states:
                    warnings.append(
                        f"{label}/{scene_id}: only {len(reference.states)} matching "
                        "benign attacker states; skipped"
                    )
                    continue
            except (OSError, sqlite3.Error, ValueError) as exc:
                warnings.append(f"{label}/{scene_id}: {exc}; skipped")
                continue

            loaded[label].append((trace, reference))
            reference_key = (
                str(reference_dir.resolve()),
                scene_id,
                attacker_id,
                int(math.ceil((trigger_time or 0.0) / args.step_time)),
                trace.states[-1].frame,
            )
            threshold_reference_traces[reference_key] = reference

    if not threshold_reference_traces:
        raise ValueError("No valid method/reference trajectory pairs were found")
    if args.thresholds_from:
        threshold_payload = json.loads(args.thresholds_from.read_text(encoding="utf-8"))
        saved_config = threshold_payload["configuration"]
        for key, expected in (
            ("step_time", args.step_time),
            ("smoothing_window_frames", args.smoothing_window),
            ("reference_percentile", args.reference_percentile),
        ):
            if saved_config.get(key) != expected:
                raise ValueError(f"Threshold source configuration mismatch: {key}")
        thresholds = {name: float(threshold_payload["thresholds"][name]) for name in PROFILE_NAMES}
        if any(not math.isfinite(value) or value < 0 for value in thresholds.values()):
            raise ValueError("Saved thresholds must be finite and nonnegative")
    else:
        thresholds = compute_thresholds(
            list(threshold_reference_traces.values()), args.reference_percentile
        )

    all_rows: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    for label, pairs in loaded.items():
        method_rows: List[Dict[str, object]] = []
        method_traces: List[Trace] = []
        method_references: List[Trace] = []
        for trace, reference in pairs:
            row = scene_metrics(
                trace,
                reference,
                thresholds,
                args.head_on_angle,
                args.severe_relative_speed,
            )
            row = {"method": label, **row}
            row["scene_dir"] = str(trace.scene_dir)
            row["reference_dir"] = str(reference.scene_dir)
            row["selection_source_dir"] = str(
                selection_scene_maps.get(label, method_scene_maps[label])[trace.scene_id]
            )
            method_rows.append(row)
            all_rows.append(row)
            method_traces.append(trace)
            method_references.append(reference)
        if method_rows:
            summaries.append(
                aggregate_method(
                    label,
                    method_rows,
                    method_traces,
                    method_references,
                )
            )
        else:
            warnings.append(f"{label}: no valid scenes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "trajectory_realism_summary.csv", summaries)
    if args.save_per_scene:
        write_csv(args.output_dir / "trajectory_realism_per_scene.csv", all_rows)
    payload = {
        "configuration": {
            "reference": str(args.reference),
            "method_references": {
                label: str(reference_overrides.get(label, args.reference))
                for label, _ in args.method
            },
            "methods": {label: str(root) for label, root in args.method},
            "selection_sources": {label: str(root) for label, root in args.selection_source},
            "thresholds_from": str(args.thresholds_from) if args.thresholds_from else None,
            "kinematic_violation_definition": "Per-scene fraction of profile samples exceeding thresholds, averaged equally across scenes",
            "step_time": args.step_time,
            "smoothing_window_frames": args.smoothing_window,
            "smoothing_window_seconds": args.smoothing_window * args.step_time,
            "reference_percentile": args.reference_percentile,
            "head_on_angle_degrees": args.head_on_angle,
            "severe_relative_speed_mps": args.severe_relative_speed,
            "minimum_states": args.min_states,
            "require_trigger_time": args.require_trigger_time,
            "save_per_scene": args.save_per_scene,
        },
        "thresholds": thresholds,
        "summaries": summaries,
        "warnings": warnings,
    }
    if args.save_per_scene:
        payload["per_scene"] = all_rows
    (args.output_dir / "trajectory_realism_results.json").write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_text_report(
        args.output_dir / "trajectory_realism_report.txt",
        summaries,
        thresholds,
        args.reference_percentile,
        args.smoothing_window,
        args.step_time,
        warnings,
        args.thresholds_from,
        args.selection_source,
        [(label, reference_overrides.get(label, args.reference)) for label, _ in args.method],
    )

    print(f"Valid scene rows: {len(all_rows)}")
    print(f"Methods summarized: {len(summaries)}")
    print(f"Output directory: {args.output_dir}")
    if warnings:
        print(f"Warnings: {len(warnings)} (see trajectory_realism_report.txt)")
    return 0 if summaries else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate executed attacker-trajectory realism against matching "
            "benign LimSim trajectories. No plots are generated."
        )
    )
    parser.add_argument(
        "--reference",
        type=lambda value: Path(value).expanduser().resolve(),
        required=True,
        help="Benign result root containing per-scene limsim.db files.",
    )
    parser.add_argument(
        "--method",
        action="append",
        type=parse_method_spec,
        required=True,
        metavar="LABEL=DIR",
        help="Adversarial method result root; repeat for multiple methods.",
    )
    parser.add_argument(
        "--method-reference",
        action="append",
        type=parse_method_spec,
        default=[],
        metavar="LABEL=DIR",
        help="Override the benign trajectory reference for LABEL, e.g. match open-loop attacks to an open-loop baseline. Does not change frozen --thresholds-from calibration.",
    )
    parser.add_argument(
        "--selection-source",
        action="append",
        type=parse_method_spec,
        default=[],
        metavar="LABEL=DIR",
        help="Use attacker ID and trigger time from this result root for LABEL (e.g. a benign method). Trajectories still come from --method.",
    )
    parser.add_argument(
        "--thresholds-from",
        type=lambda value: Path(value).expanduser().resolve(),
        help="Reuse thresholds from an earlier evaluator JSON to keep the calibration fixed when adding methods.",
    )
    parser.add_argument(
        "--output-dir",
        type=lambda value: Path(value).expanduser().resolve(),
        default=Path("evaluation_results/trajectory_realism").resolve(),
    )
    parser.add_argument(
        "--step-time",
        type=float,
        default=0.1,
        help="Seconds per LimSim frame (default: 0.1).",
    )
    parser.add_argument(
        "--reference-percentile",
        type=float,
        default=99.0,
        help="Benign percentile used for data-driven violation thresholds.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Odd moving-average window in frames (default: 5 = 0.5 s at 10 Hz).",
    )
    parser.add_argument(
        "--min-states",
        type=int,
        default=5,
        help="Minimum attacker states required in method and reference.",
    )
    parser.add_argument(
        "--head-on-angle",
        type=float,
        default=150.0,
        help="Minimum absolute heading difference for head-on classification.",
    )
    parser.add_argument(
        "--severe-relative-speed",
        type=float,
        default=5.0,
        help="Severe head-on collision-normal relative speed in m/s.",
    )
    parser.add_argument(
        "--require-trigger-time",
        action="store_true",
        help="Skip scenes whose attack injection/trigger time is unavailable.",
    )
    parser.add_argument(
        "--save-per-scene",
        action="store_true",
        help="Also save per-scene CSV/JSON details (disabled by default).",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.step_time <= 0.0:
        parser.error("--step-time must be positive")
    if not 0.0 < args.reference_percentile < 100.0:
        parser.error("--reference-percentile must be between 0 and 100")
    if args.min_states < 2:
        parser.error("--min-states must be at least 2")
    if args.smoothing_window < 1 or args.smoothing_window % 2 == 0:
        parser.error("--smoothing-window must be a positive odd integer")
    if not 0.0 <= args.head_on_angle <= 180.0:
        parser.error("--head-on-angle must be between 0 and 180")
    if args.severe_relative_speed < 0.0:
        parser.error("--severe-relative-speed cannot be negative")
    labels = [label for label, _ in args.method]
    if len(labels) != len(set(labels)):
        parser.error("--method labels must be unique")
    selection_labels = [label for label, _ in args.selection_source]
    if len(selection_labels) != len(set(selection_labels)):
        parser.error("--selection-source labels must be unique")
    if not set(selection_labels).issubset(labels):
        parser.error("--selection-source labels must match --method labels")
    reference_labels = [label for label, _ in args.method_reference]
    if len(reference_labels) != len(set(reference_labels)):
        parser.error("--method-reference labels must be unique")
    if not set(reference_labels).issubset(labels):
        parser.error("--method-reference labels must match --method labels")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    try:
        return run(args)
    except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
