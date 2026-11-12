from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .pipeline import MeasurementFailure


@dataclass(frozen=True)
class TopImageMeasurement:
    shape: str
    dimensions: dict[str, float]
    mask: np.ndarray
    diagnostics: dict[str, Any]
    confidence: float


def transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    projected = homogeneous @ np.asarray(homography, dtype=np.float64).T
    denominator = projected[:, 2:3]
    if np.any(np.abs(denominator) < 1e-12):
        raise ValueError("A calibration point maps to infinity.")
    return projected[:, :2] / denominator


def _gray_image(image: np.ndarray) -> np.ndarray:
    gray = np.asarray(image)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def _sample_pixels(image: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    gray = _gray_image(image).astype(np.float32)
    points = np.asarray(pixels, dtype=np.float32)
    map_x = points[:, 0].reshape(-1, 1)
    map_y = points[:, 1].reshape(-1, 1)
    sampled = cv2.remap(
        gray,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return sampled.reshape(-1).astype(np.float64)


def _profile_crossing(
    values: np.ndarray,
    rising: bool | None,
) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    edge_count = max(3, min(8, values.size // 6))
    low_level = float(np.median(values[:edge_count]))
    high_level = float(np.median(values[-edge_count:]))
    contrast = abs(high_level - low_level)
    if contrast < 18.0:
        return None
    threshold = (low_level + high_level) / 2.0
    signed = values - threshold
    products = signed[:-1] * signed[1:]
    candidates = np.flatnonzero(products <= 0.0)
    if candidates.size == 0:
        return None
    candidates = [
        int(index)
        for index in candidates
        if rising is None
        or (rising and values[index] < threshold <= values[index + 1])
        or (not rising and values[index] >= threshold > values[index + 1])
    ]
    if not candidates:
        return None
    index = min(candidates, key=lambda value: abs(value - (values.size - 1) / 2.0))
    denominator = values[index + 1] - values[index]
    if abs(denominator) < 1e-9:
        return float(index)
    fraction = (threshold - values[index]) / denominator
    return float(index + np.clip(fraction, 0.0, 1.0))


def _refine_circle_edges(
    image: np.ndarray,
    homography_px_to_mm: np.ndarray,
    center_mm: np.ndarray,
    radius_mm: float,
    outer_boundary: bool,
    sample_count: int = 180,
) -> tuple[float, float, float, float]:
    homography_mm_to_px = np.linalg.inv(homography_px_to_mm)
    metric_scale = float(np.sqrt(abs(np.linalg.det(homography_px_to_mm[:2, :2]))))
    margin = max(0.75, 2.2 * metric_scale)
    radial = np.linspace(radius_mm - margin, radius_mm + margin, 49)
    angles = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    metric_points = (
        center_mm[None, None, :] + radial[None, :, None] * directions[:, None, :]
    )
    pixels = transform_points(
        metric_points.reshape(-1, 2),
        homography_mm_to_px,
    )
    profile_values = _sample_pixels(image, pixels).reshape(angles.size, radial.size)
    edge_points: list[np.ndarray] = []
    for direction, values in zip(directions, profile_values):
        crossing = _profile_crossing(values, rising=outer_boundary)
        if crossing is None:
            continue
        distance = float(np.interp(crossing, np.arange(radial.size), radial))
        edge_points.append(center_mm + distance * direction)
    if len(edge_points) < 30:
        return float(center_mm[0]), float(center_mm[1]), float(radius_mm), float("nan")
    cx, cy, refined_radius, rms = _robust_circle_fit(np.asarray(edge_points))
    return cx, cy, refined_radius, rms


def segment_backlit_object(
    image: np.ndarray,
    object_is_dark: bool = True,
    blur_sigma: float = 1.0,
) -> np.ndarray:
    """Segment one opaque target from a high-contrast back-lit image."""

    gray = np.asarray(image)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur_sigma)
    mode = cv2.THRESH_BINARY_INV if object_is_dark else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, 0, 255, mode + cv2.THRESH_OTSU)
    mask = cv2.medianBlur(mask, 3)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        raise MeasurementFailure("No foreground component was found.")
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(component_areas))
    if stats[largest, cv2.CC_STAT_AREA] < 200:
        raise MeasurementFailure("The segmented target is too small.")
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def _major_contours(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours or hierarchy is None:
        raise MeasurementFailure("No target contour was found.")
    outer_index = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    outer = contours[outer_index]
    outer_area = max(1.0, cv2.contourArea(outer))
    holes = [
        contour
        for index, contour in enumerate(contours)
        if index != outer_index
        and cv2.contourArea(contour) >= outer_area * 0.002
        and hierarchy[0, index, 3] == outer_index
    ]
    holes.sort(key=cv2.contourArea, reverse=True)
    return outer, holes


def _oriented_basis(
    contour_metric: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float], dict[str, Any]]:
    rect = cv2.minAreaRect(contour_metric.astype(np.float32))
    box = cv2.boxPoints(rect).astype(np.float64)
    edges = [box[(index + 1) % 4] - box[index] for index in range(4)]
    index0 = max(range(4), key=lambda i: np.linalg.norm(edges[i]))
    e0 = edges[index0]
    e1 = edges[(index0 + 1) % 4]
    e0 /= np.linalg.norm(e0)
    e1 /= np.linalg.norm(e1)
    # minAreaRect supplies orthogonal edges for a rigid planar target.
    e1 = e1 - e0 * np.dot(e0, e1)
    e1 /= np.linalg.norm(e1)
    projected0 = contour_metric @ e0
    projected1 = contour_metric @ e1
    origin = (
        float(projected0.min()),
        float(projected1.min()),
    )
    diagnostics = {
        "rect_center_mm": [float(rect[0][0]), float(rect[0][1])],
        "rect_size_mm": [float(rect[1][0]), float(rect[1][1])],
        "rect_angle_deg": float(rect[2]),
    }
    return e0, e1, origin, diagnostics


def _to_local(
    points: np.ndarray,
    e0: np.ndarray,
    e1: np.ndarray,
) -> np.ndarray:
    return np.column_stack((points @ e0, points @ e1))


def _robust_circle_fit(points: np.ndarray) -> tuple[float, float, float, float]:
    current = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if current.shape[0] < 20:
        raise MeasurementFailure("A fitted circle has too few contour points.")

    center_x = center_y = radius = 0.0
    rms = float("inf")
    for _ in range(6):
        x = current[:, 0]
        y = current[:, 1]
        matrix = np.column_stack((2.0 * x, 2.0 * y, np.ones_like(x)))
        target = x**2 + y**2
        solution, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
        center_x, center_y, offset = solution
        radius = float(np.sqrt(max(0.0, offset + center_x**2 + center_y**2)))
        residuals = np.abs(
            np.hypot(current[:, 0] - center_x, current[:, 1] - center_y) - radius
        )
        median = float(np.median(residuals))
        mad = 1.4826 * float(np.median(np.abs(residuals - median)))
        tolerance = max(median + 3.5 * mad, radius * 0.001)
        keep = residuals <= tolerance
        if np.count_nonzero(keep) < max(20, int(current.shape[0] * 0.7)):
            break
        updated = current[keep]
        if updated.shape[0] == current.shape[0]:
            current = updated
            break
        current = updated

    residuals = np.abs(
        np.hypot(current[:, 0] - center_x, current[:, 1] - center_y) - radius
    )
    rms = float(np.sqrt(np.mean(residuals**2)))
    return center_x, center_y, radius, rms


def _robust_coordinate(values: list[float]) -> tuple[float, float, int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan"), 0
    median = float(np.median(array))
    mad = 1.4826 * float(np.median(np.abs(array - median)))
    if mad > 0:
        keep = np.abs(array - median) <= 3.5 * mad
        if np.count_nonzero(keep) >= max(10, array.size // 3):
            array = array[keep]
            median = float(np.median(array))
            mad = 1.4826 * float(np.median(np.abs(array - median)))
    return median, mad, int(array.size)


def _refine_boundary_coordinate(
    image: np.ndarray,
    homography_px_to_mm: np.ndarray,
    e0: np.ndarray,
    e1: np.ndarray,
    axis: int,
    expected: float,
    tangent_min: float,
    tangent_max: float,
    rising: bool,
    sample_count: int = 90,
) -> tuple[float, float, int]:
    homography_mm_to_px = np.linalg.inv(homography_px_to_mm)
    metric_scale = float(np.sqrt(abs(np.linalg.det(homography_px_to_mm[:2, :2]))))
    margin = max(0.65, 2.0 * metric_scale)
    normal_values = np.linspace(expected - margin, expected + margin, 41)
    tangent_span = tangent_max - tangent_min
    tangent_values = np.linspace(
        tangent_min + 0.045 * tangent_span,
        tangent_max - 0.045 * tangent_span,
        sample_count,
    )
    if axis == 0:
        local = np.stack(
            (
                np.broadcast_to(
                    normal_values[None, :], (tangent_values.size, normal_values.size)
                ),
                np.broadcast_to(
                    tangent_values[:, None], (tangent_values.size, normal_values.size)
                ),
            ),
            axis=-1,
        )
    else:
        local = np.stack(
            (
                np.broadcast_to(
                    tangent_values[:, None], (tangent_values.size, normal_values.size)
                ),
                np.broadcast_to(
                    normal_values[None, :], (tangent_values.size, normal_values.size)
                ),
            ),
            axis=-1,
        )
    metric_points = (
        local[..., 0:1] * e0[None, None, :] + local[..., 1:2] * e1[None, None, :]
    )
    pixels = transform_points(
        metric_points.reshape(-1, 2),
        homography_mm_to_px,
    )
    profile_values = _sample_pixels(image, pixels).reshape(
        tangent_values.size,
        normal_values.size,
    )
    crossings: list[float] = []
    for values in profile_values:
        crossing = _profile_crossing(values, rising=rising)
        if crossing is not None:
            crossings.append(
                float(
                    np.interp(
                        crossing,
                        np.arange(normal_values.size),
                        normal_values,
                    )
                )
            )
    return _robust_coordinate(crossings)


def _fit_boundary_line(points: np.ndarray) -> tuple[float, float, float]:
    """Fit y = m*x + b with iterative trimming; return m, b, RMS."""

    current = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if current.shape[0] < 5:
        raise MeasurementFailure("A side boundary has too few contour points.")
    slope, intercept = 0.0, 0.0
    for _ in range(5):
        x = current[:, 0]
        y = current[:, 1]
        matrix = np.column_stack((x, np.ones_like(x)))
        solution, _, _, _ = np.linalg.lstsq(matrix, y, rcond=None)
        slope, intercept = solution
        residuals = np.abs(y - (slope * x + intercept))
        median = float(np.median(residuals))
        mad = 1.4826 * float(np.median(np.abs(residuals - median)))
        keep = residuals <= median + 3.0 * mad + 1e-4
        if np.count_nonzero(keep) < max(5, current.shape[0] // 2):
            break
        if np.count_nonzero(keep) == current.shape[0]:
            break
        current = current[keep]
    y = current[:, 1]
    x = current[:, 0]
    residual = y - (slope * x + intercept)
    return float(slope), float(intercept), float(np.sqrt(np.mean(residual**2)))


def _polygon_l_parameters(
    image: np.ndarray,
    outer_pixel: np.ndarray,
    homography: np.ndarray,
) -> tuple[dict[str, float], dict[str, Any]]:
    perimeter = cv2.arcLength(outer_pixel, True)
    candidates: list[tuple[np.ndarray, float, float]] = []
    for factor in np.linspace(0.003, 0.035, 33):
        epsilon = float(perimeter * factor)
        polygon = cv2.approxPolyDP(outer_pixel, epsilon, True).reshape(-1, 2)
        candidates.append((polygon, factor, epsilon))
    polygon_candidates = [
        candidate for candidate in candidates if 6 <= candidate[0].shape[0] <= 8
    ]
    if not polygon_candidates:
        raise MeasurementFailure("Could not recover the six-sided L-block contour.")
    polygon_pixel = min(
        polygon_candidates,
        key=lambda candidate: (abs(candidate[0].shape[0] - 6), candidate[1]),
    )[0]
    polygon = transform_points(polygon_pixel, homography)
    contour_metric = transform_points(outer_pixel, homography)
    e0, e1, _, rect_diagnostics = _oriented_basis(contour_metric)
    local_polygon = _to_local(polygon, e0, e1)
    local_contour = _to_local(contour_metric, e0, e1)
    minimum = local_contour.min(axis=0)
    maximum = local_contour.max(axis=0)
    a = float(maximum[0] - minimum[0])
    b = float(maximum[1] - minimum[1])

    signed_area = 0.5 * np.sum(
        local_polygon[:, 0] * np.roll(local_polygon[:, 1], -1)
        - np.roll(local_polygon[:, 0], -1) * local_polygon[:, 1]
    )
    orientation = 1.0 if signed_area > 0 else -1.0
    turns = []
    for index in range(local_polygon.shape[0]):
        previous = local_polygon[index - 1]
        current = local_polygon[index]
        following = local_polygon[(index + 1) % local_polygon.shape[0]]
        v0 = current - previous
        v1 = following - current
        turns.append(float(v0[0] * v1[1] - v0[1] * v1[0]))
    concave_index = int(np.argmin(np.asarray(turns) * orientation))
    if turns[concave_index] * orientation >= 0:
        raise MeasurementFailure("No concave corner was found for the L-block.")
    concave = local_polygon[concave_index]

    outer_corners = np.array(
        [
            [minimum[0], minimum[1]],
            [maximum[0], minimum[1]],
            [maximum[0], maximum[1]],
            [minimum[0], maximum[1]],
        ]
    )
    diagonal = float(np.hypot(a, b))
    corner_distances = [
        float(np.min(np.linalg.norm(local_polygon - corner, axis=1)) / diagonal)
        for corner in outer_corners
    ]
    missing_index = int(np.argmax(corner_distances))
    outer_counts = {
        "left": 0,
        "right": 0,
        "top": 0,
        "bottom": 0,
    }
    left, _, count = _refine_boundary_coordinate(
        image,
        homography,
        e0,
        e1,
        axis=0,
        expected=float(minimum[0]),
        tangent_min=float(minimum[1]),
        tangent_max=float(maximum[1]),
        rising=False,
    )
    if count >= 8 and abs(left - minimum[0]) < 1.5:
        minimum[0] = left
        outer_counts["left"] = count
    right, _, count = _refine_boundary_coordinate(
        image,
        homography,
        e0,
        e1,
        axis=0,
        expected=float(maximum[0]),
        tangent_min=float(minimum[1]),
        tangent_max=float(maximum[1]),
        rising=True,
    )
    if count >= 8 and abs(right - maximum[0]) < 1.5:
        maximum[0] = right
        outer_counts["right"] = count
    top, _, count = _refine_boundary_coordinate(
        image,
        homography,
        e0,
        e1,
        axis=1,
        expected=float(minimum[1]),
        tangent_min=float(minimum[0]),
        tangent_max=float(maximum[0]),
        rising=False,
    )
    if count >= 8 and abs(top - minimum[1]) < 1.5:
        minimum[1] = top
        outer_counts["top"] = count
    bottom, _, count = _refine_boundary_coordinate(
        image,
        homography,
        e0,
        e1,
        axis=1,
        expected=float(maximum[1]),
        tangent_min=float(minimum[0]),
        tangent_max=float(maximum[0]),
        rising=True,
    )
    if count >= 8 and abs(bottom - maximum[1]) < 1.5:
        maximum[1] = bottom
        outer_counts["bottom"] = count
    a = float(maximum[0] - minimum[0])
    b = float(maximum[1] - minimum[1])

    # 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left in local y-down.
    u = concave[0] - minimum[0] if missing_index in {0, 3} else maximum[0] - concave[0]
    v = concave[1] - minimum[1] if missing_index in {0, 1} else maximum[1] - concave[1]
    notch_x = minimum[0] + u if missing_index in {0, 3} else maximum[0] - u
    notch_y = minimum[1] + v if missing_index in {0, 1} else maximum[1] - v
    notch_vertical = _refine_boundary_coordinate(
        image,
        homography,
        e0,
        e1,
        axis=0,
        expected=float(notch_x),
        tangent_min=float(minimum[1] if missing_index in {0, 1} else notch_y),
        tangent_max=float(notch_y if missing_index in {0, 1} else maximum[1]),
        rising=missing_index in {1, 2},
    )
    if notch_vertical[2] >= 10 and abs(notch_vertical[0] - notch_x) < 1.5:
        notch_x = notch_vertical[0]
    notch_horizontal = _refine_boundary_coordinate(
        image,
        homography,
        e0,
        e1,
        axis=1,
        expected=float(notch_y),
        tangent_min=float(minimum[0] if missing_index in {0, 3} else notch_x),
        tangent_max=float(notch_x if missing_index in {0, 3} else maximum[0]),
        rising=missing_index in {2, 3},
    )
    if notch_horizontal[2] >= 10 and abs(notch_horizontal[0] - notch_y) < 1.5:
        notch_y = notch_horizontal[0]
    u = notch_x - minimum[0] if missing_index in {0, 3} else maximum[0] - notch_x
    v = notch_y - minimum[1] if missing_index in {0, 1} else maximum[1] - notch_y
    corners = {
        0: "top_left",
        1: "top_right",
        2: "bottom_right",
        3: "bottom_left",
    }
    diagnostics = {
        **rect_diagnostics,
        "polygon_vertices_px": int(polygon_pixel.shape[0]),
        "approx_factor": float(
            min(
                polygon_candidates,
                key=lambda candidate: (
                    abs(candidate[0].shape[0] - 6),
                    candidate[1],
                ),
            )[1]
        ),
        "missing_corner": corners[missing_index],
        "concave_vertex_mm": [float(concave[0]), float(concave[1])],
        "corner_distance_ratio": [round(value, 5) for value in corner_distances],
        "outer_extent_mm": [a, b],
        "subpixel_outer_edge_counts": outer_counts,
        "subpixel_notch_vertical_count": notch_vertical[2],
        "subpixel_notch_horizontal_count": notch_horizontal[2],
    }
    return {"a_mm": a, "b_mm": b, "u_mm": float(u), "v_mm": float(v)}, diagnostics


def _rectangle_slot_parameters(
    image: np.ndarray,
    homography: np.ndarray,
    outer_metric: np.ndarray,
    hole_metric: np.ndarray,
) -> tuple[dict[str, float], dict[str, Any]]:
    e0, e1, _, rect_diagnostics = _oriented_basis(outer_metric)
    outer_local = _to_local(outer_metric, e0, e1)
    hole_local = _to_local(hole_metric, e0, e1)
    outer_min = outer_local.min(axis=0)
    outer_max = outer_local.max(axis=0)
    hole_min = hole_local.min(axis=0)
    hole_max = hole_local.max(axis=0)

    def update(
        minimum: np.ndarray,
        maximum: np.ndarray,
        axis: int,
        side: str,
        rising: bool,
        is_hole: bool,
    ) -> int:
        expected_index = 0 if side in {"left", "top"} else 1
        target = minimum if expected_index == 0 else maximum
        expected = float(target[axis])
        tangent_axis = 1 - axis
        refined, _, count = _refine_boundary_coordinate(
            image,
            homography,
            e0,
            e1,
            axis=axis,
            expected=expected,
            tangent_min=float(minimum[tangent_axis]),
            tangent_max=float(maximum[tangent_axis]),
            rising=rising,
        )
        if count >= 8 and abs(refined - expected) < 1.5:
            target[axis] = refined
            return count
        return 0

    counts: dict[str, int] = {}
    counts["outer_left"] = update(outer_min, outer_max, 0, "left", False, False)
    counts["outer_right"] = update(outer_min, outer_max, 0, "right", True, False)
    counts["outer_top"] = update(outer_min, outer_max, 1, "top", False, False)
    counts["outer_bottom"] = update(outer_min, outer_max, 1, "bottom", True, False)
    counts["inner_left"] = update(hole_min, hole_max, 0, "left", True, True)
    counts["inner_right"] = update(hole_min, hole_max, 0, "right", False, True)
    counts["inner_top"] = update(hole_min, hole_max, 1, "top", True, True)
    counts["inner_bottom"] = update(hole_min, hole_max, 1, "bottom", False, True)
    dimensions = {
        "a_mm": float(outer_max[0] - outer_min[0]),
        "b_mm": float(outer_max[1] - outer_min[1]),
        "u_mm": float(hole_max[0] - hole_min[0]),
        "v_mm": float(hole_max[1] - hole_min[1]),
    }
    return dimensions, {
        **rect_diagnostics,
        "subpixel_edge_counts": counts,
    }


def measure_top_image(
    image: np.ndarray,
    homography_px_to_mm: np.ndarray,
    shape_mode: str = "auto",
    object_is_dark: bool = True,
    blur_sigma: float = 1.0,
) -> TopImageMeasurement:
    mask = segment_backlit_object(
        image, object_is_dark=object_is_dark, blur_sigma=blur_sigma
    )
    outer, holes = _major_contours(mask)
    outer_metric = transform_points(outer, homography_px_to_mm)
    outer_circle = _robust_circle_fit(outer_metric)
    outer_rect = cv2.minAreaRect(outer_metric.astype(np.float32))
    outer_rect_area = max(
        1e-9,
        float(outer_rect[1][0] * outer_rect[1][1]),
    )
    outer_rect_fill = float(
        abs(cv2.contourArea(outer_metric.astype(np.float32))) / outer_rect_area
    )
    hole_circle = None
    hole_metric = None
    hole_rect_fill = None
    if holes:
        hole_metric = transform_points(holes[0], homography_px_to_mm)
        hole_circle = _robust_circle_fit(hole_metric)
        hole_rect = cv2.minAreaRect(hole_metric.astype(np.float32))
        hole_rect_area = max(1e-9, float(hole_rect[1][0] * hole_rect[1][1]))
        hole_rect_fill = float(
            abs(cv2.contourArea(hole_metric.astype(np.float32))) / hole_rect_area
        )

    outer_circle_refined = _refine_circle_edges(
        image,
        homography_px_to_mm,
        np.array(outer_circle[:2], dtype=np.float64),
        outer_circle[2],
        outer_boundary=True,
    )
    if np.isfinite(outer_circle_refined[3]):
        outer_circle = outer_circle_refined
    if hole_circle is not None:
        hole_circle_refined = _refine_circle_edges(
            image,
            homography_px_to_mm,
            np.array(hole_circle[:2], dtype=np.float64),
            hole_circle[2],
            outer_boundary=False,
        )
        if np.isfinite(hole_circle_refined[3]):
            hole_circle = hole_circle_refined

    classified = "l_block"
    classification_confidence = 0.65
    if holes:
        outer_relative_rms = outer_circle[3] / outer_circle[2]
        inner_relative_rms = hole_circle[3] / hole_circle[2]  # type: ignore[index]
        rectangular_by_area = outer_rect_fill > 0.94
        round_by_rect = outer_rect_fill < 0.90 and (
            hole_rect_fill is None or hole_rect_fill < 0.90
        )
        round_by_fit = outer_relative_rms < 0.008 and inner_relative_rms < 0.008
        if not rectangular_by_area and (round_by_rect or round_by_fit):
            classified = "cylinder"
            classification_confidence = 0.98
        else:
            classified = "blind_slot"
            classification_confidence = 0.95
    else:
        classification_confidence = 0.94 if outer_rect_fill < 0.94 else 0.58

    if shape_mode == "auto":
        shape = classified
    elif shape_mode in {"l_block", "cylinder", "blind_slot"}:
        shape = shape_mode
    else:
        raise ValueError("Unsupported shape mode.")

    diagnostics: dict[str, Any] = {
        "classified_shape": classified,
        "classification_confidence": classification_confidence,
        "outer_contour_points": int(outer.shape[0]),
        "hole_contour_points": int(holes[0].shape[0]) if holes else 0,
        "mask_area_px": int(np.count_nonzero(mask)),
        "outer_rect_fill": outer_rect_fill,
        "hole_rect_fill": hole_rect_fill,
        "outer_circle_relative_rms": outer_circle[3] / outer_circle[2],
        "hole_circle_relative_rms": (
            hole_circle[3] / hole_circle[2] if hole_circle else None
        ),
    }

    if shape == "l_block":
        if holes:
            raise MeasurementFailure(
                "An enclosed hole was found in the selected L-block."
            )
        dimensions, shape_diagnostics = _polygon_l_parameters(
            image, outer, homography_px_to_mm
        )
        diagnostics.update(shape_diagnostics)
        confidence = classification_confidence

    elif shape == "cylinder":
        if not holes:
            raise MeasurementFailure(
                "The selected through-hole cylinder has no enclosed hole."
            )
        assert hole_circle is not None
        dimensions = {
            "D_mm": 2.0 * outer_circle[2],
            "d_mm": 2.0 * hole_circle[2],
        }
        diagnostics.update(
            {
                "outer_circle_center_mm": [
                    float(outer_circle[0]),
                    float(outer_circle[1]),
                ],
                "inner_circle_center_mm": [
                    float(hole_circle[0]),
                    float(hole_circle[1]),
                ],
                "outer_circle_rms_mm": outer_circle[3],
                "inner_circle_rms_mm": hole_circle[3],
                "center_offset_mm": float(
                    np.hypot(
                        outer_circle[0] - hole_circle[0],
                        outer_circle[1] - hole_circle[1],
                    )
                ),
            }
        )
        center_tolerance = max(0.3, 0.02 * dimensions["D_mm"])
        confidence = classification_confidence * float(
            np.clip(
                1.0 - diagnostics["center_offset_mm"] / center_tolerance,
                0.4,
                1.0,
            )
        )

    else:
        if not holes:
            raise MeasurementFailure(
                "The selected blind-slot block has no enclosed slot opening."
            )
        assert hole_metric is not None
        dimensions, rectangle_diagnostics = _rectangle_slot_parameters(
            image,
            homography_px_to_mm,
            outer_metric,
            hole_metric,
        )
        diagnostics.update(
            {
                **rectangle_diagnostics,
                "slot_circle_rms_mm": hole_circle[3] if hole_circle else None,
            }
        )
        confidence = classification_confidence

    return TopImageMeasurement(
        shape=shape,
        dimensions=dimensions,
        mask=mask,
        diagnostics=diagnostics,
        confidence=float(np.clip(confidence, 0.02, 0.995)),
    )


def measure_side_height_image(
    image: np.ndarray,
    homography_px_to_mm: np.ndarray,
    object_is_dark: bool = True,
    blur_sigma: float = 1.0,
) -> tuple[float, dict[str, Any]]:
    mask = segment_backlit_object(
        image, object_is_dark=object_is_dark, blur_sigma=blur_sigma
    )
    outer, _ = _major_contours(mask)
    points = transform_points(outer, homography_px_to_mm)
    minimum_y = float(points[:, 1].min())
    maximum_y = float(points[:, 1].max())
    span = maximum_y - minimum_y
    tolerance = max(span * 0.012, 0.02)
    top_points = points[points[:, 1] <= minimum_y + tolerance]
    bottom_points = points[points[:, 1] >= maximum_y - tolerance]
    top_slope, top_intercept, top_rms = _fit_boundary_line(top_points)
    bottom_slope, bottom_intercept, bottom_rms = _fit_boundary_line(bottom_points)
    center_x = float((points[:, 0].min() + points[:, 0].max()) / 2.0)
    top_at_center = top_slope * center_x + top_intercept
    bottom_at_center = bottom_slope * center_x + bottom_intercept
    height = float(bottom_at_center - top_at_center)

    homography_mm_to_px = np.linalg.inv(homography_px_to_mm)
    metric_scale = float(np.sqrt(abs(np.linalg.det(homography_px_to_mm[:2, :2]))))
    margin = max(0.7, 2.2 * metric_scale)
    center_min = float(points[:, 0].min())
    center_max = float(points[:, 0].max())
    horizontal_span = center_max - center_min
    sample_x = np.linspace(
        center_min + 0.08 * horizontal_span,
        center_max - 0.08 * horizontal_span,
        160,
    )
    top_samples: list[np.ndarray] = []
    bottom_samples: list[np.ndarray] = []
    for x_value in sample_x:
        top_profile_y = np.linspace(
            top_at_center - margin,
            top_at_center + margin,
            45,
        )
        top_metric = np.column_stack(
            (np.full_like(top_profile_y, x_value), top_profile_y)
        )
        top_values = _sample_pixels(
            image, transform_points(top_metric, homography_mm_to_px)
        )
        top_crossing = _profile_crossing(top_values, rising=False)
        if top_crossing is not None:
            top_y = float(
                np.interp(
                    top_crossing,
                    np.arange(top_profile_y.size),
                    top_profile_y,
                )
            )
            top_samples.append(np.array([x_value, top_y]))

        bottom_profile_y = np.linspace(
            bottom_at_center - margin,
            bottom_at_center + margin,
            45,
        )
        bottom_metric = np.column_stack(
            (np.full_like(bottom_profile_y, x_value), bottom_profile_y)
        )
        bottom_values = _sample_pixels(
            image, transform_points(bottom_metric, homography_mm_to_px)
        )
        bottom_crossing = _profile_crossing(bottom_values, rising=True)
        if bottom_crossing is not None:
            bottom_y = float(
                np.interp(
                    bottom_crossing,
                    np.arange(bottom_profile_y.size),
                    bottom_profile_y,
                )
            )
            bottom_samples.append(np.array([x_value, bottom_y]))

    subpixel = False
    if len(top_samples) >= 20 and len(bottom_samples) >= 20:
        refined_top_slope, refined_top_intercept, refined_top_rms = _fit_boundary_line(
            np.asarray(top_samples)
        )
        refined_bottom_slope, refined_bottom_intercept, refined_bottom_rms = (
            _fit_boundary_line(np.asarray(bottom_samples))
        )
        refined_top = refined_top_slope * center_x + refined_top_intercept
        refined_bottom = refined_bottom_slope * center_x + refined_bottom_intercept
        refined_height = float(refined_bottom - refined_top)
        if 0.85 * height < refined_height < 1.15 * height:
            height = refined_height
            top_slope = refined_top_slope
            bottom_slope = refined_bottom_slope
            top_rms = refined_top_rms
            bottom_rms = refined_bottom_rms
            subpixel = True

    return height, {
        "mask_area_px": int(np.count_nonzero(mask)),
        "contour_points": int(outer.shape[0]),
        "top_slope": top_slope,
        "bottom_slope": bottom_slope,
        "top_line_rms_mm": top_rms,
        "bottom_line_rms_mm": bottom_rms,
        "subpixel_edge_refinement": subpixel,
        "top_edge_samples": len(top_samples),
        "bottom_edge_samples": len(bottom_samples),
    }
