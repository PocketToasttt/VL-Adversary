"""Vehicle geometry and collision checks used by attack trajectories."""

import math
from typing import Dict, List, Sequence, Tuple


Point = Tuple[float, float]


def vehicle_corners(
    x: float,
    y: float,
    heading: float,
    length: float,
    width: float,
) -> List[Point]:
    half_length = length / 2
    half_width = width / 2
    cos_heading = math.cos(heading)
    sin_heading = math.sin(heading)
    local_corners = (
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    )
    return [
        (
            x + local_x * cos_heading - local_y * sin_heading,
            y + local_x * sin_heading + local_y * cos_heading,
        )
        for local_x, local_y in local_corners
    ]


def sat_collision(poly1: Sequence[Point], poly2: Sequence[Point]) -> bool:
    def axes(polygon: Sequence[Point]) -> List[Point]:
        result = []
        for index, point in enumerate(polygon):
            next_point = polygon[(index + 1) % len(polygon)]
            edge_x = next_point[0] - point[0]
            edge_y = next_point[1] - point[1]
            normal_x, normal_y = -edge_y, edge_x
            magnitude = math.hypot(normal_x, normal_y)
            if magnitude:
                result.append((normal_x / magnitude, normal_y / magnitude))
        return result

    for axis_x, axis_y in axes(poly1) + axes(poly2):
        projection1 = [point[0] * axis_x + point[1] * axis_y for point in poly1]
        projection2 = [point[0] * axis_x + point[1] * axis_y for point in poly2]
        if max(projection1) < min(projection2) or max(projection2) < min(projection1):
            return False
    return True


def vehicle_collision(
    x1: float,
    y1: float,
    heading1: float,
    length1: float,
    width1: float,
    x2: float,
    y2: float,
    heading2: float,
    length2: float,
    width2: float,
) -> bool:
    corners1 = vehicle_corners(x1, y1, heading1, length1, width1)
    corners2 = vehicle_corners(x2, y2, heading2, length2, width2)
    if (
        max(point[0] for point in corners1) < min(point[0] for point in corners2)
        or max(point[0] for point in corners2) < min(point[0] for point in corners1)
        or max(point[1] for point in corners1) < min(point[1] for point in corners2)
        or max(point[1] for point in corners2) < min(point[1] for point in corners1)
    ):
        return False
    return sat_collision(corners1, corners2)


def predict_constant_velocity_trajectory(
    x: float,
    y: float,
    velocity: float,
    heading: float,
    steps: int,
    dt: float,
    length: float,
    width: float,
) -> List[Dict[str, object]]:
    trajectory = []
    for index in range(steps):
        elapsed = index * dt
        predicted_x = x + velocity * math.cos(heading) * elapsed
        predicted_y = y + velocity * math.sin(heading) * elapsed
        trajectory.append(
            {
                "x": predicted_x,
                "y": predicted_y,
                "heading": heading,
                "vel": velocity,
                "step": index,
                "corners": vehicle_corners(
                    predicted_x, predicted_y, heading, length, width
                ),
                "length": length,
                "width": width,
            }
        )
    return trajectory

