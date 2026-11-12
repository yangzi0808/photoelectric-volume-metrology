from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .calibration import (
    CameraIntrinsics,
    LaserPlane,
    fit_laser_plane,
    plane_points_at_depth,
)
from .laser import DepthEstimate, estimate_two_level_depth
from .laser_scan import (
    LaserLineConfig,
    LaserLineResult,
    extract_laser_line,
    triangulate_laser_line,
)


@dataclass(frozen=True)
class StagePose:
    rotation_camera_from_stage: np.ndarray
    translation_camera_from_stage: np.ndarray

    def camera_from_stage(self, points_stage: np.ndarray) -> np.ndarray:
        points = np.asarray(points_stage, dtype=np.float64).reshape(-1, 3)
        return (
            points @ self.rotation_camera_from_stage.T
            + self.translation_camera_from_stage
        )

    def stage_from_camera(self, points_camera: np.ndarray) -> np.ndarray:
        points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
        centered = points - self.translation_camera_from_stage
        return centered @ self.rotation_camera_from_stage


def pose_from_plane_homography(
    homography_mm_to_px: np.ndarray,
    camera_matrix: np.ndarray,
) -> StagePose:
    """Recover camera pose for a metric board plane z=0."""

    matrix = np.linalg.inv(camera_matrix) @ np.asarray(
        homography_mm_to_px, dtype=np.float64
    )
    first = matrix[:, 0]
    second = matrix[:, 1]
    scale = 2.0 / (np.linalg.norm(first) + np.linalg.norm(second))
    first *= scale
    second *= scale
    third = np.cross(first, second)
    rotation_guess = np.column_stack((first, second, third))
    u, _, vt = np.linalg.svd(rotation_guess)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    translation = matrix[:, 2] * scale
    return StagePose(
        rotation_camera_from_stage=rotation,
        translation_camera_from_stage=translation,
    )


def stage_plane_in_camera(
    pose: StagePose,
    height_mm: float,
) -> tuple[np.ndarray, float]:
    normal = pose.rotation_camera_from_stage[:, 2]
    offset = float(height_mm - np.dot(normal, pose.translation_camera_from_stage))
    return normal, offset


def triangulate_lines_in_stage(
    lines: Iterable[LaserLineResult],
    intrinsics: CameraIntrinsics,
    laser_plane: LaserPlane,
    pose: StagePose,
    angles_deg: Iterable[float] | None = None,
) -> np.ndarray:
    line_values = list(lines)
    if angles_deg is None:
        angles = [0.0] * len(line_values)
    else:
        angles = list(angles_deg)
    if len(angles) != len(line_values):
        raise ValueError("Each laser line must have one turntable angle.")
    stage_cloud = []
    for line, angle in zip(line_values, angles):
        camera_points = triangulate_laser_line(line, intrinsics, laser_plane)
        stage_points = pose.stage_from_camera(camera_points)
        theta = np.deg2rad(angle)
        cosine, sine = np.cos(theta), np.sin(theta)
        unrotation = np.array(
            [
                [cosine, sine, 0.0],
                [-sine, cosine, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        stage_points = stage_points @ unrotation.T
        stage_cloud.append(stage_points)
    return np.vstack(stage_cloud)


def calibrate_laser_plane_from_known_heights(
    images: Iterable[np.ndarray],
    known_heights_mm: Iterable[float],
    intrinsics: CameraIntrinsics,
    pose: StagePose,
    line_config: LaserLineConfig | None = None,
    backgrounds: Iterable[np.ndarray | None] | None = None,
) -> LaserPlane:
    image_values = list(images)
    heights = list(known_heights_mm)
    if len(image_values) != len(heights):
        raise ValueError("Each laser image must have one known plane height.")
    if len({round(float(height), 6) for height in heights}) < 2:
        raise ValueError("Laser-plane calibration needs at least two distinct heights.")
    if backgrounds is None:
        background_values = [None] * len(image_values)
    else:
        background_values = list(backgrounds)
        if len(background_values) != len(image_values):
            raise ValueError("Laser images and backgrounds must have equal length.")
    camera_points = []
    for image, height, background in zip(image_values, heights, background_values):
        line = extract_laser_line(image, line_config, background)
        normal, offset = stage_plane_in_camera(pose, height)
        points = plane_points_at_depth(
            line.pixels,
            intrinsics,
            normal,
            offset,
        )
        camera_points.append(points[np.all(np.isfinite(points), axis=1)])
    combined = np.vstack(camera_points)
    return fit_laser_plane(combined)


def estimate_blind_slot_depth(
    stage_points: np.ndarray,
) -> DepthEstimate:
    points = np.asarray(stage_points, dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] < 20:
        raise ValueError("Too few triangulated laser points.")
    return estimate_two_level_depth(points[:, 2])
