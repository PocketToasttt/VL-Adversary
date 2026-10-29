import os
import argparse
import pickle
from collections import defaultdict

import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon


# ==============================
# 舒适性阈值
# ==============================
BOUNDARIES = {
    "max_abs_lat_accel": 4.89,   # [m/s^2]
    "max_lon_accel": 2.40,       # [m/s^2]
    "min_lon_accel": -4.05,      # [m/s^2]
    "max_abs_yaw_accel": 1.93,   # [rad/s^2]
    "max_abs_lon_jerk": 8.37,    # [m/s^3]
    "max_abs_yaw_rate": 0.95,    # [rad/s]
}


# ==============================
# 基础工具函数
# ==============================
def load_pkl(path_str):
    with open(path_str, "rb") as f:
        return pickle.load(f)


def collect_pkl_files(path):
    """
    支持三种输入：
    1. 直接传某个 .pkl 文件
    2. 传某个单场景目录（里面有 drive_arena.pkl）
    3. 传上层目录，递归查找所有 drive_arena.pkl
    """
    path = os.path.abspath(path)

    if os.path.isfile(path):
        return [path]

    if not os.path.isdir(path):
        return []

    drivearena_files = []
    other_pkl_files = []

    for root, _, files in os.walk(path):
        if "drive_arena.pkl" in files:
            drivearena_files.append(os.path.join(root, "drive_arena.pkl"))

        for f in files:
            if f.endswith(".pkl") and f != "drive_arena.pkl":
                other_pkl_files.append(os.path.join(root, f))

    if len(drivearena_files) > 0:
        return sorted(list(dict.fromkeys(drivearena_files)))

    return sorted(list(dict.fromkeys(other_pkl_files)))


# ==================== 改动1：summarize_scene_results 新增 danger_exposure_rate ====================
def summarize_scene_results(scene_results, ttc_cap=10.0, title="Results", do_print=True):
    if len(scene_results) == 0:
        if do_print:
            print(f"{title}: no valid scenes.")
        return None

    collision_flags = [float(r["scene_collision_flag"]) for r in scene_results]

    min_ttc_values = []
    for r in scene_results:
        if "scene_min_ttc_for_avg" in r and np.isfinite(r["scene_min_ttc_for_avg"]):
            t = float(r["scene_min_ttc_for_avg"])
        else:
            raw_t = r.get("scene_min_ttc", np.inf)
            t = min(float(raw_t), ttc_cap) if np.isfinite(raw_t) else float(ttc_cap)
        min_ttc_values.append(t)

    # DAC and Comfort are intentionally retained in the source but excluded
    # from the primary safety-critical evaluation. They characterize road
    # compliance and ride quality rather than interaction criticality.
    # dac_values = [float(r["scene_dac"]) for r in scene_results]
    # comfortable_values = [float(r["scene_comfortable"]) for r in scene_results]
    # ★ 新增：danger_exposure_rate 汇总
    danger_exposure_values = [
        float(r["danger_exposure_rate"])
        for r in scene_results
        if "danger_exposure_rate" in r
    ]
    peak_required_deceleration_values = [
        float(r["peak_required_deceleration"])
        for r in scene_results
        if np.isfinite(r.get("peak_required_deceleration", np.nan))
    ]
    collision_normal_velocity_values = [
        float(r["collision_normal_relative_velocity"])
        for r in scene_results
        if np.isfinite(r.get("collision_normal_relative_velocity", np.nan))
    ]
    prd_caps = [
        float(r["prd_cap"])
        for r in scene_results
        if np.isfinite(r.get("prd_cap", np.nan))
    ]

    summary = {
        "num_scenes": int(len(scene_results)),
        "num_collision_scenes": int(np.sum(collision_flags)),
        "collision_rate": float(np.mean(collision_flags)),
        "collision_rate_std": float(np.std(collision_flags)),
        "mean_min_ttc": float(np.mean(min_ttc_values)),
        "mean_min_ttc_std": float(np.std(min_ttc_values)),
        # "mean_dac": float(np.mean(dac_values)),
        # "mean_dac_std": float(np.std(dac_values)),
        # "mean_comfortable": float(np.mean(comfortable_values)),
        # "mean_comfortable_std": float(np.std(comfortable_values)),
        # ★ 新增
        "mean_danger_exposure_rate": float(np.mean(danger_exposure_values)) if danger_exposure_values else 0.0,
        "mean_danger_exposure_rate_std": float(np.std(danger_exposure_values)) if danger_exposure_values else 0.0,
        "mean_peak_required_deceleration": float(np.mean(peak_required_deceleration_values)) if peak_required_deceleration_values else 0.0,
        "mean_peak_required_deceleration_std": float(np.std(peak_required_deceleration_values)) if peak_required_deceleration_values else 0.0,
        "mean_collision_normal_relative_velocity": float(np.mean(collision_normal_velocity_values)) if collision_normal_velocity_values else np.nan,
        "mean_collision_normal_relative_velocity_std": float(np.std(collision_normal_velocity_values)) if collision_normal_velocity_values else np.nan,
        "num_collision_velocity_scenes": int(len(collision_normal_velocity_values)),
        "prd_cap": float(prd_caps[0]) if prd_caps else np.nan,
    }

    if do_print:
        print(f"\n============================= {title} =============================")
        print(f"Collision Rate            : {summary['collision_rate']:.4f} +/- {summary['collision_rate_std']:.4f}")
        print(f"Mean Min TTC              : {summary['mean_min_ttc']:.4f} +/- {summary['mean_min_ttc_std']:.4f}   (ttc_cap={ttc_cap})")
        # ★ 新增打印
        print(f"Mean Danger Exposure Rate : {summary['mean_danger_exposure_rate']:.4f} +/- {summary['mean_danger_exposure_rate_std']:.4f}   (TTC < threshold)")
        print(
            f"Mean Peak Required Decel. : {summary['mean_peak_required_deceleration']:.4f} +/- "
            f"{summary['mean_peak_required_deceleration_std']:.4f} m/s^2 "
            f"(cap={summary['prd_cap']:.1f})"
        )
        if summary["num_collision_velocity_scenes"] > 0:
            print(
                "Mean Collision-normal RV  : "
                f"{summary['mean_collision_normal_relative_velocity']:.4f} +/- "
                f"{summary['mean_collision_normal_relative_velocity_std']:.4f} m/s "
                f"(collision scenes n={summary['num_collision_velocity_scenes']})"
            )
        else:
            print("Mean Collision-normal RV  : N/A (no resolved collision scenes)")
        # print(f"Mean DAC                  : {summary['mean_dac']:.4f} +/- {summary['mean_dac_std']:.4f}")
        # print(f"Mean Comfortable          : {summary['mean_comfortable']:.4f} +/- {summary['mean_comfortable_std']:.4f}")
        print("=================================================================\n")

    return summary
# ==================== 改动1 结束 ====================


# ==============================
# 评分器
# ==============================
class ScoreCalculator:
    def __init__(self, data, box_order="wlh"):
        self.data = data if isinstance(data, dict) else {}
        frames = self.data.get("frames", [])
        self.frames = sorted(frames, key=lambda x: x.get("time_stamp", 0.0)) if isinstance(frames, list) else []
        self.box_order = box_order

    def _get_scene_id(self):
        for key in ["scene_id", "scene_name"]:
            if key in self.data and self.data[key]:
                return str(self.data[key])

        metas = self.data.get("metas", {})
        if isinstance(metas, dict):
            for key in ["scene_id", "scene_name", "scene_token", "token"]:
                if key in metas and metas[key]:
                    return str(metas[key])

        return "unknown_scene"

    # ==================== 改动2：新增 _get_attack_type，从 metas 读取攻击类型 ====================
    def _get_attack_type(self):
        """从 metas 中读取攻击类型，若不存在则返回 'unknown'"""
        metas = self.data.get("metas", {})
        if isinstance(metas, dict):
            return str(metas.get("attack_type", "unknown"))
        return "unknown"

    def _get_meta(self, key, default=None):
        metas = self.data.get("metas", {})
        if not isinstance(metas, dict):
            return default
        return metas.get(key, default)
    # ==================== 改动2 结束 ====================

    def _is_vehicle_name(self, name):
        if name is None:
            return True

        if isinstance(name, bytes):
            try:
                name = name.decode("utf-8", errors="ignore")
            except Exception:
                name = str(name)

        name = str(name).lower()

        vehicle_tokens = [
            "car", "vehicle", "truck", "bus", "trailer", "construction_vehicle",
        ]
        return any(token in name for token in vehicle_tokens)

    def _get_box_width_length(self, box):
        if self.box_order == "wlh":
            width = float(box[3])
            length = float(box[4])
        elif self.box_order == "lwh":
            length = float(box[3])
            width = float(box[4])
        else:
            raise ValueError(f"Unsupported box_order: {self.box_order}")
        return width, length

    def get_vehicle_corners(self, x, y, yaw, length, width):
        c = np.cos(yaw)
        s = np.sin(yaw)

        front_left  = np.array([x + c * length / 2 - s * width / 2, y + s * length / 2 + c * width / 2])
        front_right = np.array([x + c * length / 2 + s * width / 2, y + s * length / 2 - c * width / 2])
        rear_right  = np.array([x - c * length / 2 + s * width / 2, y - s * length / 2 - c * width / 2])
        rear_left   = np.array([x - c * length / 2 - s * width / 2, y - s * length / 2 + c * width / 2])

        return np.array([front_left, front_right, rear_right, rear_left], dtype=float)

    def _create_rectangle(self, center_x, center_y, width, length, yaw):
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        x_offs = [length / 2, length / 2, -length / 2, -length / 2]
        y_offs = [width / 2, -width / 2, -width / 2, width / 2]

        x_pts = [center_x + x_off * cos_yaw - y_off * sin_yaw for x_off, y_off in zip(x_offs, y_offs)]
        y_pts = [center_y + x_off * sin_yaw + y_off * cos_yaw for x_off, y_off in zip(x_offs, y_offs)]

        return ShapelyPolygon(zip(x_pts, y_pts))

    def _polygons_collide(self, poly1, poly2, area_thr=0.0):
        if not poly1.intersects(poly2):
            return False
        if area_thr <= 0.0:
            return True
        try:
            inter_area = poly1.intersection(poly2).area
            return inter_area > area_thr
        except Exception:
            return True

    def _obb_contact_normal(self, ego_box, obj_box):
        """Return the minimum-overlap SAT axis for two contacting OBBs."""
        ego_x, ego_y, _, _, _, _, ego_yaw = ego_box
        obj_x, obj_y, _, _, _, _, obj_yaw = obj_box
        ego_w, ego_l = self._get_box_width_length(ego_box)
        obj_w, obj_l = self._get_box_width_length(obj_box)
        ego_corners = self.get_vehicle_corners(
            ego_x, ego_y, ego_yaw, ego_l, ego_w
        )
        obj_corners = self.get_vehicle_corners(
            obj_x, obj_y, obj_yaw, obj_l, obj_w
        )
        axes = (
            np.asarray([np.cos(ego_yaw), np.sin(ego_yaw)]),
            np.asarray([-np.sin(ego_yaw), np.cos(ego_yaw)]),
            np.asarray([np.cos(obj_yaw), np.sin(obj_yaw)]),
            np.asarray([-np.sin(obj_yaw), np.cos(obj_yaw)]),
        )
        minimum_overlap = np.inf
        contact_axis = None
        for axis in axes:
            ego_projection = ego_corners @ axis
            obj_projection = obj_corners @ axis
            overlap = min(ego_projection.max(), obj_projection.max()) - max(
                ego_projection.min(), obj_projection.min()
            )
            if overlap < -1e-8:
                return None
            if overlap < minimum_overlap:
                minimum_overlap = float(overlap)
                contact_axis = axis.astype(float)
        if contact_axis is None:
            return None
        center_delta = np.asarray([obj_x - ego_x, obj_y - ego_y], dtype=float)
        if float(np.dot(center_delta, contact_axis)) < 0.0:
            contact_axis = -contact_axis
        return contact_axis

    def _constant_velocity_box_conflict(
        self,
        ego_box,
        obj_box,
        ego_velocity,
        obj_velocity,
    ):
        """Continuous 2D OBB conflict under constant translational velocity.

        The swept separating-axis theorem returns both TTC and the relative
        closing speed on the critical entry axis.  The latter supports a
        two-dimensional extension of required deceleration:
        closing_speed / (2 * TTC), which reduces to conventional DRAC for a
        one-dimensional following conflict.
        """
        ego_x, ego_y, _, _, _, _, ego_yaw = ego_box
        obj_x, obj_y, _, _, _, _, obj_yaw = obj_box
        ego_w, ego_l = self._get_box_width_length(ego_box)
        obj_w, obj_l = self._get_box_width_length(obj_box)

        ego_heading = np.array([np.cos(ego_yaw), np.sin(ego_yaw)])
        ego_lateral = np.array([-np.sin(ego_yaw), np.cos(ego_yaw)])
        obj_heading = np.array([np.cos(obj_yaw), np.sin(obj_yaw)])
        obj_lateral = np.array([-np.sin(obj_yaw), np.cos(obj_yaw)])
        axes = (ego_heading, ego_lateral, obj_heading, obj_lateral)

        center_delta = np.array([obj_x - ego_x, obj_y - ego_y], dtype=float)
        relative_velocity = np.asarray(obj_velocity, dtype=float) - np.asarray(
            ego_velocity, dtype=float
        )
        entry_time = -np.inf
        exit_time = np.inf
        critical_closing_speed = 0.0

        for axis in axes:
            ego_radius = (
                ego_l * 0.5 * abs(float(np.dot(ego_heading, axis)))
                + ego_w * 0.5 * abs(float(np.dot(ego_lateral, axis)))
            )
            obj_radius = (
                obj_l * 0.5 * abs(float(np.dot(obj_heading, axis)))
                + obj_w * 0.5 * abs(float(np.dot(obj_lateral, axis)))
            )
            obj_center = float(np.dot(center_delta, axis))
            axis_speed = float(np.dot(relative_velocity, axis))

            ego_min, ego_max = -ego_radius, ego_radius
            obj_min, obj_max = obj_center - obj_radius, obj_center + obj_radius
            if abs(axis_speed) <= 1e-8:
                if obj_max < ego_min or obj_min > ego_max:
                    return {"ttc": np.inf, "closing_speed": 0.0}
                continue

            crossing_1 = (ego_min - obj_max) / axis_speed
            crossing_2 = (ego_max - obj_min) / axis_speed
            axis_entry = min(crossing_1, crossing_2)
            axis_exit = max(crossing_1, crossing_2)
            if axis_entry > entry_time:
                entry_time = axis_entry
                critical_closing_speed = abs(axis_speed)
            exit_time = min(exit_time, axis_exit)
            if entry_time > exit_time:
                return {"ttc": np.inf, "closing_speed": 0.0}

        if exit_time < 0.0:
            return {"ttc": np.inf, "closing_speed": 0.0}
        return {
            "ttc": float(max(0.0, entry_time)),
            "closing_speed": float(max(0.0, critical_closing_speed)),
        }

    def _constant_velocity_box_ttc(
        self,
        ego_box,
        obj_box,
        ego_velocity,
        obj_velocity,
    ):
        """Continuous 2D OBB TTC under constant translational velocity."""
        return self._constant_velocity_box_conflict(
            ego_box,
            obj_box,
            ego_velocity,
            obj_velocity,
        )["ttc"]

    def _build_tracking_samples(self):
        """Build identity-preserving ego/object tracks from evaluation frames."""
        samples = []
        for frame_idx, frame in enumerate(self.frames):
            ego_box = frame.get("ego_box")
            if ego_box is None or len(ego_box) < 7:
                continue

            obj_boxes = frame.get("obj_boxes", []) or []
            obj_names = frame.get("obj_names", []) or []
            obj_ids = frame.get("obj_ids", []) or []
            objects = {}
            object_indices = {}
            for obj_idx, obj_box in enumerate(obj_boxes):
                if obj_box is None or len(obj_box) < 7:
                    continue
                obj_name = obj_names[obj_idx] if obj_idx < len(obj_names) else "car"
                if not self._is_vehicle_name(obj_name):
                    continue
                object_id = (
                    str(obj_ids[obj_idx])
                    if obj_idx < len(obj_ids)
                    else f"frame_index_{obj_idx}"
                )
                objects[object_id] = obj_box
                object_indices[object_id] = obj_idx

            samples.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp": float(frame.get("time_stamp", frame_idx)),
                    "ego_box": ego_box,
                    "objects": objects,
                    "object_indices": object_indices,
                }
            )
        return samples

    def _estimate_track_velocity(self, samples, sample_idx, object_id=None):
        """Estimate center velocity without mixing different object tracks."""
        def get_box(sample):
            if object_id is None:
                return sample["ego_box"]
            return sample["objects"].get(object_id)

        current_box = get_box(samples[sample_idx])
        if current_box is None:
            return None

        previous_idx = None
        for idx in range(sample_idx - 1, -1, -1):
            if get_box(samples[idx]) is not None:
                previous_idx = idx
                break

        next_idx = None
        for idx in range(sample_idx + 1, len(samples)):
            if get_box(samples[idx]) is not None:
                next_idx = idx
                break

        current_time = samples[sample_idx]["timestamp"]
        max_track_gap = 0.5
        if (
            previous_idx is not None
            and current_time - samples[previous_idx]["timestamp"] <= max_track_gap
        ):
            start_idx, end_idx = previous_idx, sample_idx
        elif (
            next_idx is not None
            and samples[next_idx]["timestamp"] - current_time <= max_track_gap
        ):
            start_idx, end_idx = sample_idx, next_idx
        else:
            return np.zeros(2, dtype=float)

        start_box = get_box(samples[start_idx])
        end_box = get_box(samples[end_idx])
        delta_time = samples[end_idx]["timestamp"] - samples[start_idx]["timestamp"]
        if delta_time <= 1e-8:
            return np.zeros(2, dtype=float)
        return np.array(
            [end_box[0] - start_box[0], end_box[1] - start_box[1]],
            dtype=float,
        ) / delta_time

    def _calculate_collision_course_ttc(
        self,
        return_detail=False,
        collision_area_thr=0.0,
    ):
        """Calculate TTC from identity-preserving, swept oriented boxes."""
        samples = self._build_tracking_samples()

        if not samples:
            empty_result = {
                "scene_min_ttc": np.inf,
                "ttc_series": [],
            }
            return empty_result if return_detail else np.inf

        ttc_series = []
        for sample_idx, sample in enumerate(samples):
            ego_box = sample["ego_box"]
            ego_x, ego_y, _, _, _, _, ego_yaw = ego_box
            ego_w, ego_l = self._get_box_width_length(ego_box)
            ego_poly = self._create_rectangle(
                ego_x, ego_y, ego_w, ego_l, ego_yaw
            )
            ego_velocity = self._estimate_track_velocity(samples, sample_idx)
            frame_ttc = np.inf

            for object_id, obj_box in sample["objects"].items():
                obj_x, obj_y, _, _, _, _, obj_yaw = obj_box
                obj_w, obj_l = self._get_box_width_length(obj_box)
                obj_poly = self._create_rectangle(
                    obj_x, obj_y, obj_w, obj_l, obj_yaw
                )
                if self._polygons_collide(
                    ego_poly, obj_poly, area_thr=collision_area_thr
                ):
                    frame_ttc = 0.0
                    break

                obj_velocity = self._estimate_track_velocity(
                    samples,
                    sample_idx,
                    object_id=object_id,
                )
                if obj_velocity is None:
                    continue
                frame_ttc = min(
                    frame_ttc,
                    self._constant_velocity_box_ttc(
                        ego_box,
                        obj_box,
                        ego_velocity,
                        obj_velocity,
                    ),
                )

            ttc_series.append(float(frame_ttc))

        finite_ttc = [value for value in ttc_series if np.isfinite(value)]
        min_ttc = min(finite_ttc) if finite_ttc else np.inf
        detail = {
            "scene_min_ttc": float(min_ttc),
            "ttc_series": ttc_series,
        }
        return detail if return_detail else float(min_ttc)

    def _calculate_peak_required_deceleration(self, prd_cap=20.0):
        """Return scene-level peak required deceleration in m/s^2.

        For every ego-object pair on a constant-velocity OBB collision course,
        the critical-axis closing speed and TTC are obtained from swept SAT.
        ``closing_speed / (2 * TTC)`` is the 2D critical-axis counterpart of
        conventional DRAC. Numerically unbounded near-zero-TTC conflicts are
        clipped at ``prd_cap``. Existing contacts use the maximum finite
        pre-contact demand; the cap is used only if sparse frames contain no
        resolvable pre-contact sample.
        """
        if prd_cap <= 0.0:
            raise ValueError("prd_cap must be positive")

        samples = self._build_tracking_samples()
        peak_required_deceleration = 0.0
        contact_seen = False
        peak_detail = {
            "peak_required_deceleration": 0.0,
            "peak_prd_timestamp": None,
            "peak_prd_object_id": None,
            "peak_prd_ttc": np.inf,
            "peak_prd_closing_speed": 0.0,
        }
        for sample_idx, sample in enumerate(samples):
            ego_velocity = self._estimate_track_velocity(samples, sample_idx)
            for object_id, obj_box in sample["objects"].items():
                obj_velocity = self._estimate_track_velocity(
                    samples, sample_idx, object_id=object_id
                )
                if obj_velocity is None:
                    continue
                conflict = self._constant_velocity_box_conflict(
                    sample["ego_box"],
                    obj_box,
                    ego_velocity,
                    obj_velocity,
                )
                ttc = float(conflict["ttc"])
                closing_speed = float(conflict["closing_speed"])
                if not np.isfinite(ttc):
                    continue
                if ttc <= 1e-6:
                    # A realized contact has no finite avoidance deceleration.
                    # Prefer the last finite pre-contact demand so PRD does not
                    # collapse into a duplicate of Collision Rate.
                    contact_seen = True
                    continue
                elif closing_speed > 0.0:
                    required_deceleration = min(
                        float(prd_cap), closing_speed / (2.0 * ttc)
                    )
                else:
                    required_deceleration = 0.0

                if required_deceleration > peak_required_deceleration:
                    peak_required_deceleration = required_deceleration
                    peak_detail = {
                        "peak_required_deceleration": float(required_deceleration),
                        "peak_prd_timestamp": float(sample["timestamp"]),
                        "peak_prd_object_id": object_id,
                        "peak_prd_ttc": ttc,
                        "peak_prd_closing_speed": closing_speed,
                    }
        if contact_seen and peak_required_deceleration <= 0.0:
            # Sparse saved frames can occasionally expose contact without a
            # preceding finite collision-course sample. Mark that unresolved
            # demand with the documented cap rather than incorrectly using 0.
            peak_detail["peak_required_deceleration"] = float(prd_cap)
        return peak_detail

    def _calculate_collision_normal_relative_velocity(self, collision_info):
        """Estimate first-collision relative speed along the contact normal.

        The collision target is any traffic participant, matching the primary
        scene collision-rate definition.  If a simulator-recorded collision
        lies between saved frames, the closest object in the nearest frame is
        used.  Unresolved collisions return NaN and are excluded from the
        collision-conditional dataset mean rather than being incorrectly set
        to zero.
        """
        collision_timestamp = collision_info.get("first_collision_timestamp")
        if collision_timestamp is None or not np.isfinite(collision_timestamp):
            return {
                "collision_normal_relative_velocity": np.nan,
                "collision_velocity_object_id": None,
                "collision_velocity_timestamp": None,
                "collision_normal_source": None,
            }

        samples = self._build_tracking_samples()
        if not samples:
            return {
                "collision_normal_relative_velocity": np.nan,
                "collision_velocity_object_id": None,
                "collision_velocity_timestamp": None,
                "collision_normal_source": None,
            }
        sample_idx = min(
            range(len(samples)),
            key=lambda idx: abs(samples[idx]["timestamp"] - collision_timestamp),
        )
        sample = samples[sample_idx]
        object_id = collision_info.get("collision_object_id")
        if object_id not in sample["objects"]:
            ego_box = sample["ego_box"]
            ego_x, ego_y, _, _, _, _, ego_yaw = ego_box
            ego_w, ego_l = self._get_box_width_length(ego_box)
            ego_poly = self._create_rectangle(
                ego_x, ego_y, ego_w, ego_l, ego_yaw
            )
            nearest_distance = np.inf
            object_id = None
            for candidate_id, candidate_box in sample["objects"].items():
                obj_x, obj_y, _, _, _, _, obj_yaw = candidate_box
                obj_w, obj_l = self._get_box_width_length(candidate_box)
                obj_poly = self._create_rectangle(
                    obj_x, obj_y, obj_w, obj_l, obj_yaw
                )
                distance = float(ego_poly.distance(obj_poly))
                if distance < nearest_distance:
                    nearest_distance = distance
                    object_id = candidate_id

        if object_id is None or object_id not in sample["objects"]:
            return {
                "collision_normal_relative_velocity": np.nan,
                "collision_velocity_object_id": None,
                "collision_velocity_timestamp": float(sample["timestamp"]),
                "collision_normal_source": None,
            }

        ego_box = sample["ego_box"]
        obj_box = sample["objects"][object_id]
        ego_velocity = self._estimate_track_velocity(samples, sample_idx)
        obj_velocity = self._estimate_track_velocity(
            samples, sample_idx, object_id=object_id
        )
        if obj_velocity is None:
            return {
                "collision_normal_relative_velocity": np.nan,
                "collision_velocity_object_id": object_id,
                "collision_velocity_timestamp": float(sample["timestamp"]),
                "collision_normal_source": None,
            }

        normal = self._obb_contact_normal(ego_box, obj_box)
        normal_source = "sat_contact_axis"
        if normal is None:
            # Simulator-recorded collisions can lie between saved 2 Hz frames.
            # In that case, use the center line in the nearest saved frame and
            # expose the approximation source in the per-scene diagnostics.
            center_delta = np.asarray(
                [obj_box[0] - ego_box[0], obj_box[1] - ego_box[1]], dtype=float
            )
            center_distance = float(np.linalg.norm(center_delta))
            if center_distance <= 1e-8:
                normal = np.asarray(
                    [np.cos(float(ego_box[6])), np.sin(float(ego_box[6]))],
                    dtype=float,
                )
            else:
                normal = center_delta / center_distance
            normal_source = "nearest_frame_centerline_approximation"
        relative_velocity = np.asarray(obj_velocity) - np.asarray(ego_velocity)
        collision_normal_velocity = abs(float(np.dot(relative_velocity, normal)))
        return {
            "collision_normal_relative_velocity": collision_normal_velocity,
            "collision_velocity_object_id": object_id,
            "collision_velocity_timestamp": float(sample["timestamp"]),
            "collision_normal_source": normal_source,
        }

    def _extract_actual_ego_traj(self):
        traj = []
        timestamps = []
        valid_frames = []

        for frame_idx, frame in enumerate(self.frames):
            ego_box = frame.get("ego_box", None)
            if ego_box is None or len(ego_box) < 7:
                continue

            x, y, _, _, _, _, yaw = ego_box
            ts = float(frame.get("time_stamp", frame_idx))

            traj.append([float(x), float(y), float(yaw)])
            timestamps.append(ts)
            valid_frames.append((frame_idx, frame))

        if len(traj) == 0:
            return (
                np.zeros((0, 3), dtype=float),
                np.zeros((0,), dtype=float),
                []
            )

        return (
            np.asarray(traj, dtype=float),
            np.asarray(timestamps, dtype=float),
            valid_frames
        )

    def _estimate_timestep_from_timestamps(self, timestamps, default=0.1):
        timestamps = np.asarray(timestamps, dtype=float)
        if len(timestamps) < 2:
            return float(default)

        diffs = np.diff(timestamps)
        diffs = diffs[np.isfinite(diffs) & (diffs > 1e-6)]

        if len(diffs) == 0:
            return float(default)

        return float(np.median(diffs))

    def _calculate_actual_scene_collision_flag(self, return_detail=False, collision_area_thr=0.0):
        total_valid_frames = 0

        for frame_idx, frame in enumerate(self.frames):
            ego_box = frame.get("ego_box", None)
            if ego_box is None or len(ego_box) < 7:
                continue

            total_valid_frames += 1
            timestamp = float(frame.get("time_stamp", frame_idx))

            ego_x, ego_y, _, _, _, _, ego_yaw = ego_box
            ego_w, ego_l = self._get_box_width_length(ego_box)
            ego_poly = self._create_rectangle(ego_x, ego_y, ego_w, ego_l, ego_yaw)

            obj_boxes = frame.get("obj_boxes", []) or []
            obj_names = frame.get("obj_names", []) or []
            obj_ids = frame.get("obj_ids", []) or []

            for obj_idx, obj in enumerate(obj_boxes):
                if obj is None or len(obj) < 7:
                    continue

                obj_name = obj_names[obj_idx] if obj_idx < len(obj_names) else "car"
                if not self._is_vehicle_name(obj_name):
                    continue

                obj_x, obj_y, _, _, _, _, obj_yaw = obj
                obj_w, obj_l = self._get_box_width_length(obj)
                obj_poly = self._create_rectangle(obj_x, obj_y, obj_w, obj_l, obj_yaw)

                if self._polygons_collide(ego_poly, obj_poly, area_thr=collision_area_thr):
                    if return_detail:
                        return {
                            "scene_collision_flag": 1.0,
                            "first_collision_frame_idx": frame_idx,
                            "first_collision_timestamp": timestamp,
                            "collision_obj_idx": obj_idx,
                            "collision_object_id": (
                                str(obj_ids[obj_idx])
                                if obj_idx < len(obj_ids)
                                else f"frame_index_{obj_idx}"
                            ),
                            "num_valid_frames": total_valid_frames,
                        }
                    return 1.0

        if return_detail:
            return {
                "scene_collision_flag": 0.0,
                "first_collision_frame_idx": None,
                "first_collision_timestamp": None,
                "collision_obj_idx": None,
                "collision_object_id": None,
                "num_valid_frames": total_valid_frames,
            }
        return 0.0

    def _calculate_drivable_area_compliance(
        self, drivable_mask, transformed_traj, vehicle_width, vehicle_length, return_detail=False
    ):
        def meter_to_pixel(x, y, mask_shape, real_size=100.0):
            h, w = mask_shape[:2]
            px = int(round((x + real_size / 2.0) * (w / real_size)))
            py = int(round((real_size / 2.0 - y) * (h / real_size)))
            return px, py

        def is_drivable(mask, x, y):
            if mask is None:
                return True
            mask = np.asarray(mask)
            if mask.ndim == 0:
                return True
            if mask.ndim > 2:
                mask = mask[..., 0]
            h, w = mask.shape[:2]
            px, py = meter_to_pixel(x, y, mask.shape)
            if px < 0 or px >= w or py < 0 or py >= h:
                return False
            return bool(mask[py, px] > 0)

        if transformed_traj is None or len(transformed_traj) == 0:
            if return_detail:
                return {"dac_score": 1.0, "min_pose_ratio": 1.0, "offroad_flag": 0.0, "pose_ratios": []}
            return 1.0

        pose_ratios = []
        for x, y, yaw in transformed_traj:
            corners = self.get_vehicle_corners(x, y, yaw, vehicle_length, vehicle_width)
            center = np.array([x, y], dtype=float)
            mids = 0.5 * (corners + center[None, :])
            sample_points = [center] + [p for p in corners] + [p for p in mids]
            flags = [is_drivable(drivable_mask, p[0], p[1]) for p in sample_points]
            pose_ratios.append(float(np.mean(flags)) if flags else 1.0)

        dac_score = float(np.mean(pose_ratios)) if pose_ratios else 1.0
        min_pose_ratio = float(np.min(pose_ratios)) if pose_ratios else 1.0
        offroad_flag = 1.0 if min_pose_ratio < 0.2 else 0.0

        if return_detail:
            return {"dac_score": dac_score, "min_pose_ratio": min_pose_ratio, "offroad_flag": offroad_flag, "pose_ratios": pose_ratios}
        return dac_score

    def _calculate_actual_scene_dac(self, return_detail=False):
        frame_scores = []

        for frame_idx, frame in enumerate(self.frames):
            ego_box = frame.get("ego_box", None)
            if ego_box is None or len(ego_box) < 7:
                continue

            drivable_mask = frame.get("drivable_mask", None)
            ego_w, ego_l = self._get_box_width_length(ego_box)

            if drivable_mask is None:
                frame_scores.append(1.0)
                continue

            mask = np.asarray(drivable_mask).astype(np.uint8)
            local_pose = np.array([[0.0, 0.0, 0.0]], dtype=float)

            info = self._calculate_drivable_area_compliance(mask, local_pose, ego_w, ego_l, return_detail=True)
            frame_scores.append(float(info["dac_score"]))

        scene_dac = float(np.mean(frame_scores)) if frame_scores else 1.0

        if return_detail:
            return {"scene_dac": scene_dac, "frame_dac_scores": frame_scores, "num_valid_frames": len(frame_scores)}
        return scene_dac

    def _calculate_is_comfortable(self, traj, timestep, return_detail=False):
        if traj is None or len(traj) < 4:
            if return_detail:
                return {
                    "comfortable_score": 1.0,
                    "pass_rates": {k: 1.0 for k in ["lat_accel", "lon_accel_upper", "lon_accel_lower", "lon_jerk", "yaw_accel", "yaw_rate"]},
                    "max_values": {k: 0.0 for k in ["max_abs_lat_accel", "max_lon_accel", "min_lon_accel", "max_abs_lon_jerk", "max_abs_yaw_accel", "max_abs_yaw_rate"]},
                }
            return 1.0

        if timestep is None or timestep <= 0:
            timestep = 0.1

        traj = np.asarray(traj, dtype=float)
        x   = traj[:, 0]
        y   = traj[:, 1]
        yaw = np.unwrap(traj[:, 2])

        vx    = np.gradient(x, timestep)
        vy    = np.gradient(y, timestep)
        speed = np.sqrt(vx ** 2 + vy ** 2)

        yaw_rate  = np.gradient(yaw, timestep)
        yaw_accel = np.gradient(yaw_rate, timestep)
        lon_accel = np.gradient(speed, timestep)
        lat_accel = speed * yaw_rate
        lon_jerk  = np.gradient(lon_accel, timestep)

        pass_rates = {
            "lat_accel":       float(np.mean(np.abs(lat_accel) <= BOUNDARIES["max_abs_lat_accel"])),
            "lon_accel_upper": float(np.mean(lon_accel <= BOUNDARIES["max_lon_accel"])),
            "lon_accel_lower": float(np.mean(lon_accel >= BOUNDARIES["min_lon_accel"])),
            "lon_jerk":        float(np.mean(np.abs(lon_jerk) <= BOUNDARIES["max_abs_lon_jerk"])),
            "yaw_accel":       float(np.mean(np.abs(yaw_accel) <= BOUNDARIES["max_abs_yaw_accel"])),
            "yaw_rate":        float(np.mean(np.abs(yaw_rate) <= BOUNDARIES["max_abs_yaw_rate"])),
        }

        comfortable_score = float(np.mean(list(pass_rates.values())))

        if return_detail:
            return {
                "comfortable_score": comfortable_score,
                "pass_rates": pass_rates,
                "max_values": {
                    "max_abs_lat_accel": float(np.max(np.abs(lat_accel))) if len(lat_accel) > 0 else 0.0,
                    "max_lon_accel":     float(np.max(lon_accel))         if len(lon_accel) > 0 else 0.0,
                    "min_lon_accel":     float(np.min(lon_accel))         if len(lon_accel) > 0 else 0.0,
                    "max_abs_lon_jerk":  float(np.max(np.abs(lon_jerk)))  if len(lon_jerk) > 0 else 0.0,
                    "max_abs_yaw_accel": float(np.max(np.abs(yaw_accel))) if len(yaw_accel) > 0 else 0.0,
                    "max_abs_yaw_rate":  float(np.max(np.abs(yaw_rate)))  if len(yaw_rate) > 0 else 0.0,
                }
            }

        return comfortable_score

    # ==================== 改动4：calculate() 新增 danger_exposure_rate + attack_type ====================
    def calculate(
        self,
        ttc_cap=10.0,
        collision_area_thr=0.0,
        danger_ttc_threshold=3.0,
        prd_cap=20.0,
    ):
        """
        新增参数：
          danger_ttc_threshold: TTC 低于此值（秒）视为"危险帧"，默认 3.0s
        新增返回字段：
          attack_type           : 从 metas 读取的攻击类型
          danger_exposure_rate  : 危险帧（TTC < danger_ttc_threshold）占总帧比例
        """
        if len(self.frames) == 0:
            return None

        actual_traj, timestamps, valid_frames = self._extract_actual_ego_traj()
        if len(valid_frames) == 0:
            return None

        # ★ 读取攻击类型
        attack_type = self._get_attack_type()

        collision_info = self._calculate_actual_scene_collision_flag(
            return_detail=True,
            collision_area_thr=collision_area_thr
        )
        scene_collision_flag = float(collision_info["scene_collision_flag"])
        recorded_collision = bool(self._get_meta("collision", False))
        if recorded_collision:
            scene_collision_flag = 1.0
            if collision_info.get("first_collision_timestamp") is None:
                collision_info["first_collision_timestamp"] = self._get_meta(
                    "collision_time"
                )

        # ★ 获取 ttc_series 用于计算 danger_exposure_rate
        ttc_detail = self._calculate_collision_course_ttc(
            return_detail=True,
            collision_area_thr=collision_area_thr
        )
        scene_min_ttc = ttc_detail["scene_min_ttc"]
        ttc_series = ttc_detail.get("ttc_series", [])
        if scene_collision_flag > 0.5:
            scene_min_ttc = 0.0
            if not any(np.isfinite(value) and value <= 0.0 for value in ttc_series):
                ttc_series = list(ttc_series) + [0.0]

        if len(ttc_series) > 0:
            danger_frames = sum(
                1 for ttc_val in ttc_series
                if np.isfinite(ttc_val) and ttc_val < danger_ttc_threshold
            )
            danger_exposure_rate = danger_frames / len(ttc_series)
        else:
            danger_exposure_rate = 0.0

        prd_info = self._calculate_peak_required_deceleration(prd_cap=prd_cap)
        collision_velocity_info = (
            self._calculate_collision_normal_relative_velocity(collision_info)
            if scene_collision_flag > 0.5
            else {
                "collision_normal_relative_velocity": np.nan,
                "collision_velocity_object_id": None,
                "collision_velocity_timestamp": None,
                "collision_normal_source": None,
            }
        )

        # DAC and Comfort implementations are intentionally retained for
        # supplementary analysis, but disabled in the primary safety output.
        # dac_info = self._calculate_actual_scene_dac(return_detail=True)
        # timestep = self._estimate_timestep_from_timestamps(timestamps, default=0.1)
        # comfort_info = self._calculate_is_comfortable(
        #     actual_traj, timestep, return_detail=True
        # )

        if np.isfinite(scene_min_ttc):
            scene_min_ttc_for_avg = min(float(scene_min_ttc), float(ttc_cap))
        else:
            scene_min_ttc_for_avg = float(ttc_cap)

        result = {
            "scene_id":                   self._get_scene_id(),
            # ★ 新增
            "attack_type":                attack_type,
            "scene_collision_flag":       scene_collision_flag,
            "scene_min_ttc":              float(scene_min_ttc) if np.isfinite(scene_min_ttc) else np.inf,
            "scene_min_ttc_for_avg":      float(scene_min_ttc_for_avg),
            "danger_exposure_rate":       float(danger_exposure_rate),
            "peak_required_deceleration": float(prd_info["peak_required_deceleration"]),
            "prd_cap":                    float(prd_cap),
            "collision_normal_relative_velocity": float(collision_velocity_info["collision_normal_relative_velocity"]),
            # "scene_dac":                float(dac_info["scene_dac"]),
            # "scene_comfortable":        float(comfort_info["comfortable_score"]),
            "num_eval_frames":            int(len(valid_frames)),
            "first_collision_frame_idx":  collision_info.get("first_collision_frame_idx"),
            "first_collision_timestamp":  collision_info.get("first_collision_timestamp"),
            "peak_prd_timestamp":         prd_info.get("peak_prd_timestamp"),
            "peak_prd_object_id":         prd_info.get("peak_prd_object_id"),
            "peak_prd_ttc":               prd_info.get("peak_prd_ttc"),
            "peak_prd_closing_speed":     prd_info.get("peak_prd_closing_speed"),
            "collision_velocity_object_id": collision_velocity_info.get("collision_velocity_object_id"),
            "collision_velocity_timestamp": collision_velocity_info.get("collision_velocity_timestamp"),
            "collision_normal_source":      collision_velocity_info.get("collision_normal_source"),
        }

        return result
    # ==================== 改动4 结束 ====================


# ==============================
# 数据集评估
# ==============================
def evaluate_dataset(
    data,
    ttc_cap=10.0,
    source_name=None,
    print_summary=True,
    collision_area_thr=0.0,
    box_order="wlh",
    danger_ttc_threshold=3.0,   # ★ 新增参数
    prd_cap=20.0,
):
    print(f"this pkl file contains {len(data)} experiment records.")

    all_scene_results = []

    for exp_idx, exp_data in enumerate(data):
        print(f"\n========== Processing scene {exp_idx + 1}/{len(data)} ==========")

        score_calc = ScoreCalculator(exp_data, box_order=box_order)
        result = score_calc.calculate(
            ttc_cap=ttc_cap,
            collision_area_thr=collision_area_thr,
            danger_ttc_threshold=danger_ttc_threshold,   # ★ 传入
            prd_cap=prd_cap,
        )

        if result is None:
            print("invalid scene, skipped.")
            continue

        if result["scene_id"] == "unknown_scene" and source_name is not None:
            if len(data) == 1:
                result["scene_id"] = source_name
            else:
                result["scene_id"] = f"{source_name}_{exp_idx:03d}"

        scene_ttc_str  = f"{result['scene_min_ttc']:.4f}" if np.isfinite(result["scene_min_ttc"]) else "inf"
        extra_info = ""
        if result["scene_collision_flag"] > 0.5:
            coll_ts  = result.get("first_collision_timestamp")
            coll_idx = result.get("first_collision_frame_idx")
            if coll_ts is not None and np.isfinite(coll_ts):
                extra_info = f", first_collision_t={coll_ts:.4f}, first_collision_frame={coll_idx}"

        # ★ 打印新增字段
        print(
            f"scene_id={result['scene_id']}, "
            f"attack_type={result['attack_type']}, "
            f"collision={int(result['scene_collision_flag'])}, "
            f"min_ttc={scene_ttc_str}, "
            f"danger_exposure={result['danger_exposure_rate']:.4f}, "
            f"peak_required_deceleration={result['peak_required_deceleration']:.4f}, "
            f"collision_normal_relative_velocity="
            f"{result['collision_normal_relative_velocity']:.4f}"
            # f"dac={result['scene_dac']:.4f}, "
            # f"comfortable={result['scene_comfortable']:.4f}"
            f"{extra_info}"
        )

        all_scene_results.append(result)

    if len(all_scene_results) == 0:
        print("No valid scene results.")
        return None

    summary = summarize_scene_results(
        all_scene_results,
        ttc_cap=ttc_cap,
        title="Results",
        do_print=print_summary
    )

    summary["scene_results"] = all_scene_results
    return summary


# ==============================
# 主函数
# ==============================
def main():
    parser = argparse.ArgumentParser(description="实际执行版攻击指标评估器")
    parser.add_argument(
        "path",
        nargs="?",
        default="./results",
        help="单个场景目录 / 单个 pkl / 多场景上层目录"
    )
    parser.add_argument(
        "--ttc_cap",
        type=float,
        default=10.0,
        help="对 inf 的 scene_min_ttc 做均值统计时的截断上限"
    )
    parser.add_argument(
        "--collision_area_thr",
        type=float,
        default=0.0,
        help="碰撞判定最小交叠面积阈值；0 表示只要接触就算碰撞"
    )
    parser.add_argument(
        "--box_order",
        type=str,
        default="wlh",
        choices=["wlh", "lwh"],
        help="box 格式。wlh 表示 [x,y,z,w,l,h,yaw]；lwh 表示 [x,y,z,l,w,h,yaw]"
    )
    # ★ 新增参数
    parser.add_argument(
        "--danger_ttc_threshold",
        type=float,
        default=3.0,
        help="TTC 低于此值（秒）视为危险帧，用于计算 danger_exposure_rate，默认 3.0s"
    )
    parser.add_argument(
        "--prd_cap",
        type=float,
        default=20.0,
        help="Peak Required Deceleration 的场景级截断上限，单位 m/s^2，默认 20.0"
    )
    args = parser.parse_args()
    if args.prd_cap <= 0.0:
        parser.error("--prd_cap must be positive")

    pkl_files = collect_pkl_files(args.path)
    if len(pkl_files) == 0:
        raise FileNotFoundError(f"No pkl files found under: {args.path}")

    # 单文件模式
    if len(pkl_files) == 1:
        pkl_path = pkl_files[0]
        print(f"Using data file: {pkl_path}")

        data = load_pkl(pkl_path)
        if not isinstance(data, list):
            raise TypeError("drive_arena.pkl should contain a list of experiment records.")

        source_name = os.path.basename(os.path.dirname(pkl_path))
        result = evaluate_dataset(
            data,
            ttc_cap=args.ttc_cap,
            source_name=source_name,
            print_summary=True,
            collision_area_thr=args.collision_area_thr,
            box_order=args.box_order,
            danger_ttc_threshold=args.danger_ttc_threshold,  # ★ 传入
            prd_cap=args.prd_cap,
        )

        # ★ 单文件也做按攻击类型分组统计
        if result and result.get("scene_results"):
            _print_grouped_summary(result["scene_results"], args.ttc_cap)

        return

    # 多文件批量模式
    print(f"Found {len(pkl_files)} pkl files under: {os.path.abspath(args.path)}")

    overall_scene_results = []

    for file_idx, pkl_path in enumerate(pkl_files):
        print(f"\n################ File {file_idx + 1}/{len(pkl_files)} ################")
        print(f"Using data file: {pkl_path}")

        try:
            data = load_pkl(pkl_path)
        except Exception as e:
            print(f"Failed to load {pkl_path}: {e}")
            continue

        if not isinstance(data, list):
            print(f"Invalid pkl format, skipped: {pkl_path}")
            continue

        source_name = os.path.basename(os.path.dirname(pkl_path))

        file_result = evaluate_dataset(
            data,
            ttc_cap=args.ttc_cap,
            source_name=source_name,
            print_summary=False,
            collision_area_thr=args.collision_area_thr,
            box_order=args.box_order,
            danger_ttc_threshold=args.danger_ttc_threshold,  # ★ 传入
            prd_cap=args.prd_cap,
        )

        if file_result is None:
            continue

        overall_scene_results.extend(file_result["scene_results"])

    # 总体汇总
    summarize_scene_results(
        overall_scene_results,
        ttc_cap=args.ttc_cap,
        title="Overall Summary Across All Scene Folders",
        do_print=True
    )

    # ★ 新增：按攻击类型分组汇总
    _print_grouped_summary(overall_scene_results, args.ttc_cap)


# ==================== 改动5（新增函数）：按攻击类型分组汇总 ====================
def _print_grouped_summary(scene_results, ttc_cap=10.0):
    """
    按 attack_type 分组，分别打印汇总指标。
    方便对比 blind_spot_merge / unprotected_left_turn / intersection_tight_turn 等类型的效果。
    """
    by_type = defaultdict(list)
    for r in scene_results:
        atype = r.get("attack_type", "unknown")
        by_type[atype].append(r)

    if len(by_type) <= 1 and "unknown" in by_type:
        # 所有场景都没有记录攻击类型，跳过分组
        print("\n[提示] 未检测到有效的 attack_type 字段，跳过分组统计。")
        print("      请在仿真代码中添加: self.scorer.data['metas']['attack_type'] = attack_type\n")
        return

    print("\n" + "=" * 60)
    print("  按攻击类型分组汇总（Attack Type Breakdown）")
    print("=" * 60)

    for atype in sorted(by_type.keys()):
        results = by_type[atype]
        summarize_scene_results(
            results,
            ttc_cap=ttc_cap,
            title=f"Attack Type: [{atype}]  (n={len(results)})",
            do_print=True
        )
# ==================== 改动5 结束 ====================


if __name__ == "__main__":
    main()
