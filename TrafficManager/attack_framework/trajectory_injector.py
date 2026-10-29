"""Runtime injection of generated adversarial trajectories into LimSim."""

from typing import Any, Dict, Iterable, Mapping, MutableMapping

from LimSim.utils.trajectory import State, Trajectory


class AttackTrajectoryInjector:
    def __init__(self, point_interval: float = 0.1):
        self.point_interval = point_interval
        self.active: Dict[str, Dict[str, Any]] = {}

    def add(
        self,
        vehicle_id: str,
        points: Iterable[Mapping[str, Any]],
        start_time: float,
    ) -> None:
        trajectory = [dict(point) for point in points]
        if not trajectory:
            raise ValueError("Attack trajectory cannot be empty")
        self.active[str(vehicle_id)] = {
            "trajectory": trajectory,
            "start_time": float(start_time),
        }

    def merge(
        self,
        trajectories: MutableMapping[Any, Trajectory],
        timestamp: float,
    ) -> MutableMapping[Any, Trajectory]:
        completed = []
        for vehicle_id, attack in self.active.items():
            points = attack["trajectory"]
            elapsed = max(0.0, timestamp - attack["start_time"])
            start_index = int(elapsed / self.point_interval)
            if start_index >= len(points):
                completed.append(vehicle_id)
                continue

            attack_trajectory = self._to_limsim_trajectory(
                points[start_index:], timestamp
            )
            matching_key = next(
                (key for key in trajectories if str(key) == vehicle_id),
                vehicle_id,
            )
            trajectories[matching_key] = attack_trajectory

        for vehicle_id in completed:
            del self.active[vehicle_id]
        return trajectories

    def _to_limsim_trajectory(
        self,
        points: Iterable[Mapping[str, Any]],
        timestamp: float,
    ) -> Trajectory:
        trajectory = Trajectory()
        if not hasattr(trajectory, "states"):
            trajectory.states = []

        for index, point in enumerate(points):
            state = State()
            state.x = point["x"]
            state.y = point["y"]
            if hasattr(state, "v"):
                state.v = point["v"]
            elif hasattr(state, "vel"):
                state.vel = point["v"]
            if hasattr(state, "heading"):
                state.heading = point["heading"]
            elif hasattr(state, "yaw"):
                state.yaw = point["heading"]
            if hasattr(state, "t"):
                state.t = timestamp + index * self.point_interval
            elif hasattr(state, "time_stamp"):
                state.time_stamp = timestamp + index * self.point_interval
            trajectory.states.append(state)
        return trajectory

