from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import (
    CameraIntrinsics,
    LaserPlane,
    pixel_rays_camera,
)


@dataclass(frozen=True)
class LaserLineConfig:
    axis: str = "vertical"
    peak_threshold: float = 12.0
    min_contrast: float = 20.0
    centroid_half_window: int = 2
    max_saturation: float = 256.0


@dataclass(frozen=True)
class LaserLineResult:
    pixels: np.ndarray
    response: np.ndarray
    mask: np.ndarray


def _gray(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim == 3:
        return cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)
    return value


def laser_response(
    image: np.ndarray,
    background: np.ndarray | None = None,
) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim == 3:
        b, g, r = cv2.split(value)
        chroma = r.astype(np.float32) - np.maximum(g, b).astype(np.float32)
        if float(chroma.max()) > 10.0:
            response = np.clip(chroma, 0, None)
        else:
            value = cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)
            response = value.astype(np.float32)
    else:
        response = value.astype(np.float32)
    if background is not None:
        base = _gray(background).astype(np.float32)
        foreground = _gray(image).astype(np.float32)
        differential = np.clip(foreground - base, 0, None)
        if response.std() < 1e-6 or response.max() < 10:
            response = differential
        else:
            response = np.maximum(response, differential)
    return response


def extract_laser_line(
    image: np.ndarray,
    config: LaserLineConfig | None = None,
    background: np.ndarray | None = None,
) -> LaserLineResult:
    cfg = config or LaserLineConfig()
    response = laser_response(image, background)
    mask = np.zeros(response.shape, dtype=np.uint8)
    points: list[tuple[float, float, float]] = []
    if cfg.axis == "vertical":
        outer_indices = range(response.shape[0])
    elif cfg.axis == "horizontal":
        outer_indices = range(response.shape[1])
    else:
        raise ValueError("Laser line axis must be vertical or horizontal.")

    half = max(1, cfg.centroid_half_window)
    for outer_index in outer_indices:
        profile = (
            response[outer_index, :]
            if cfg.axis == "vertical"
            else response[:, outer_index]
        )
        if profile.size == 0:
            continue
        peak_index = int(np.argmax(profile))
        peak = float(profile[peak_index])
        baseline = float(np.median(profile))
        if peak < cfg.peak_threshold or peak - baseline < cfg.min_contrast:
            continue
        if peak >= cfg.max_saturation:
            continue
        start = max(0, peak_index - half)
        end = min(profile.size, peak_index + half + 1)
        local = profile[start:end].astype(np.float64)
        local -= float(local.min())
        denominator = float(local.sum())
        if denominator <= 1e-9:
            continue
        center = start + float(np.dot(np.arange(local.size), local) / denominator)
        if cfg.axis == "vertical":
            u, v = center, float(outer_index)
            mask[outer_index, peak_index] = 255
        else:
            u, v = float(outer_index), center
            mask[peak_index, outer_index] = 255
        points.append((u, v, peak))

    if not points:
        raise ValueError("No usable laser-line points were detected.")
    return LaserLineResult(
        pixels=np.asarray([[point[0], point[1]] for point in points]),
        response=np.asarray([point[2] for point in points]),
        mask=mask,
    )


def triangulate_laser_line(
    line: LaserLineResult,
    intrinsics: CameraIntrinsics,
    laser_plane: LaserPlane,
) -> np.ndarray:
    rays = pixel_rays_camera(line.pixels, intrinsics)
    camera_origin = np.zeros(3, dtype=np.float64)
    return laser_plane.intersect_rays(camera_origin, rays)


def distance_to_plane(
    points: np.ndarray, normal: np.ndarray, offset: float
) -> np.ndarray:
    unit_normal = np.asarray(normal, dtype=np.float64)
    unit_normal /= np.linalg.norm(unit_normal)
    return np.asarray(points, dtype=np.float64) @ unit_normal + float(offset)
