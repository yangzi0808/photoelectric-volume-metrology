from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class CharucoSpec:
    squares_x: int = 7
    squares_y: int = 5
    square_length_mm: float = 20.0
    marker_length_mm: float = 15.0
    dictionary_name: str = "DICT_6X6_250"

    def board(self) -> tuple[cv2.aruco.CharucoBoard, cv2.aruco.Dictionary]:
        dictionary_id = getattr(cv2.aruco, self.dictionary_name)
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length_mm,
            self.marker_length_mm,
            dictionary,
        )
        return board, dictionary


@dataclass(frozen=True)
class CameraIntrinsics:
    camera_matrix: np.ndarray
    distortion: np.ndarray
    image_size: tuple[int, int]
    rms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion": self.distortion.ravel().tolist(),
            "image_size": list(self.image_size),
            "rms": self.rms,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CameraIntrinsics:
        return cls(
            camera_matrix=np.asarray(value["camera_matrix"], dtype=np.float64),
            distortion=np.asarray(value["distortion"], dtype=np.float64),
            image_size=(
                int(value["image_size"][0]),
                int(value["image_size"][1]),
            ),
            rms=float(value["rms"]),
        )


@dataclass(frozen=True)
class PlaneCalibration:
    homography_px_to_mm: np.ndarray
    rms_mm: float
    detection_count: int
    image_size: tuple[int, int]

    @property
    def mm_per_px(self) -> float:
        return float(np.sqrt(abs(np.linalg.det(self.homography_px_to_mm[:2, :2]))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "homography_px_to_mm": self.homography_px_to_mm.tolist(),
            "rms_mm": self.rms_mm,
            "detection_count": self.detection_count,
            "image_size": list(self.image_size),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PlaneCalibration:
        return cls(
            homography_px_to_mm=np.asarray(
                value["homography_px_to_mm"], dtype=np.float64
            ),
            rms_mm=float(value["rms_mm"]),
            detection_count=int(value["detection_count"]),
            image_size=(
                int(value["image_size"][0]),
                int(value["image_size"][1]),
            ),
        )


@dataclass(frozen=True)
class LaserPlane:
    normal: np.ndarray
    offset: float
    rms_mm: float

    def intersect_rays(
        self,
        camera_origin: np.ndarray,
        ray_directions: np.ndarray,
    ) -> np.ndarray:
        origin = np.asarray(camera_origin, dtype=np.float64).reshape(1, 3)
        directions = np.asarray(ray_directions, dtype=np.float64).reshape(-1, 3)
        normal_norm = np.linalg.norm(self.normal)
        if normal_norm <= 1e-15:
            raise ValueError("Laser plane normal must be non-zero.")
        normal = self.normal / normal_norm
        offset = self.offset / normal_norm
        denominator = directions @ normal
        valid = np.abs(denominator) > 1e-9
        distances = np.full(directions.shape[0], np.nan, dtype=np.float64)
        distances[valid] = (
            -(np.dot(normal, origin.ravel()) + offset) / denominator[valid]
        )
        points = origin + distances[:, None] * directions
        points[~valid] = np.nan
        return points

    def to_dict(self) -> dict[str, Any]:
        return {
            "normal": self.normal.tolist(),
            "offset": self.offset,
            "rms_mm": self.rms_mm,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LaserPlane:
        return cls(
            normal=np.asarray(value["normal"], dtype=np.float64),
            offset=float(value["offset"]),
            rms_mm=float(value["rms_mm"]),
        )


@dataclass(frozen=True)
class CalibrationBundle:
    top_intrinsics: CameraIntrinsics
    side_intrinsics: CameraIntrinsics
    top_plane: PlaneCalibration
    side_plane: PlaneCalibration
    laser_plane: LaserPlane | None = None
    version: str = "2.0"
    created_at: str = ""

    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self.top_intrinsics

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "created_at": self.created_at or datetime.now().astimezone().isoformat(),
            "top_intrinsics": self.top_intrinsics.to_dict(),
            "side_intrinsics": self.side_intrinsics.to_dict(),
            "top_plane": self.top_plane.to_dict(),
            "side_plane": self.side_plane.to_dict(),
            "laser_plane": (self.laser_plane.to_dict() if self.laser_plane else None),
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> CalibrationBundle:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        laser = value.get("laser_plane")
        legacy_intrinsics = value.get("intrinsics")
        top_intrinsics = value.get("top_intrinsics", legacy_intrinsics)
        side_intrinsics = value.get(
            "side_intrinsics",
            top_intrinsics,
        )
        if top_intrinsics is None or side_intrinsics is None:
            raise ValueError("Calibration file has no camera intrinsics.")
        return cls(
            top_intrinsics=CameraIntrinsics.from_dict(top_intrinsics),
            side_intrinsics=CameraIntrinsics.from_dict(side_intrinsics),
            top_plane=PlaneCalibration.from_dict(value["top_plane"]),
            side_plane=PlaneCalibration.from_dict(value["side_plane"]),
            laser_plane=LaserPlane.from_dict(laser) if laser else None,
            version=str(value.get("version", "2.0")),
            created_at=str(value.get("created_at", "")),
        )


def _as_gray(image: np.ndarray) -> np.ndarray:
    gray = np.asarray(image)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def detect_charuco(
    image: np.ndarray,
    spec: CharucoSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    board, _ = spec.board()
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(_as_gray(image))
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 8:
        raise ValueError("Fewer than 8 ChArUco corners were detected.")
    return (
        charuco_corners.reshape(-1, 2).astype(np.float64),
        charuco_ids.reshape(-1).astype(np.int32),
        board.getChessboardCorners(),
    )


def calibrate_intrinsics(
    images: Iterable[np.ndarray],
    spec: CharucoSpec,
    min_views: int = 8,
) -> CameraIntrinsics:
    _board, _ = spec.board()
    all_object_points: list[np.ndarray] = []
    all_image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    for image in images:
        gray = _as_gray(image)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        elif image_size != (gray.shape[1], gray.shape[0]):
            raise ValueError("All intrinsic-calibration images must share one size.")
        try:
            corners, ids, board_corners = detect_charuco(gray, spec)
        except ValueError:
            continue
        all_object_points.append(
            board_corners[ids].reshape(-1, 1, 3).astype(np.float32)
        )
        all_image_points.append(corners.reshape(-1, 1, 2).astype(np.float32))

    if len(all_object_points) < min_views:
        raise ValueError(
            f"Only {len(all_object_points)} usable views; "
            f"at least {min_views} are required."
        )
    assert image_size is not None
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        all_object_points,
        all_image_points,
        image_size,
        None,
        None,
        flags=cv2.CALIB_RATIONAL_MODEL,
    )
    return CameraIntrinsics(
        camera_matrix=matrix,
        distortion=distortion,
        image_size=image_size,
        rms=float(rms),
    )


def calibrate_plane_homography(
    image: np.ndarray,
    spec: CharucoSpec,
    intrinsics: CameraIntrinsics | None = None,
) -> PlaneCalibration:
    prepared = _as_gray(image)
    if intrinsics is not None:
        if (prepared.shape[1], prepared.shape[0]) != intrinsics.image_size:
            raise ValueError(
                "Plane-calibration image size does not match camera intrinsics."
            )
        prepared = cv2.undistort(
            prepared,
            intrinsics.camera_matrix,
            intrinsics.distortion,
        )
    corners, ids, board_corners = detect_charuco(prepared, spec)
    object_points = board_corners[ids][:, :2].reshape(-1, 2)
    homography, inlier_mask = cv2.findHomography(
        corners,
        object_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=1.0,
        maxIters=5000,
        confidence=0.999,
    )
    if homography is None or inlier_mask is None:
        raise ValueError("Plane homography estimation failed.")
    inliers = inlier_mask.ravel().astype(bool)
    homogeneous = np.column_stack((corners, np.ones(corners.shape[0])))
    projected = homogeneous @ homography.T
    projected = projected[:, :2] / projected[:, 2:3]
    residual = np.linalg.norm(projected - object_points, axis=1)
    if np.count_nonzero(inliers) < 8:
        raise ValueError("Too few inlier ChArUco corners.")
    return PlaneCalibration(
        homography_px_to_mm=homography,
        rms_mm=float(np.sqrt(np.mean(residual[inliers] ** 2))),
        detection_count=int(corners.shape[0]),
        image_size=(prepared.shape[1], prepared.shape[0]),
    )


def fit_laser_plane(points_camera: np.ndarray) -> LaserPlane:
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] < 20:
        raise ValueError("At least 20 laser points are required for plane fitting.")
    center = points.mean(axis=0)
    _, singular_values, vh = np.linalg.svd(
        points - center,
        full_matrices=False,
    )
    if singular_values[1] <= max(singular_values[0] * 1e-5, 1e-9):
        raise ValueError("Laser points are collinear; the plane is underdetermined.")
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(np.dot(normal, center))
    rms = float(np.sqrt(np.mean((points @ normal + offset) ** 2)))
    return LaserPlane(normal=normal, offset=offset, rms_mm=rms)


def pixel_rays_camera(
    pixels: np.ndarray,
    intrinsics: CameraIntrinsics,
    normalize: bool = True,
) -> np.ndarray:
    points = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    if np.any(np.abs(intrinsics.distortion) > 1e-12):
        points = cv2.undistortPoints(
            points.reshape(-1, 1, 2),
            intrinsics.camera_matrix,
            intrinsics.distortion,
        ).reshape(-1, 2)
        homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    directions = homogeneous @ np.linalg.inv(intrinsics.camera_matrix).T
    if normalize:
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    return directions


def plane_points_at_depth(
    pixels: np.ndarray,
    intrinsics: CameraIntrinsics,
    plane_normal: np.ndarray,
    plane_offset: float,
) -> np.ndarray:
    """Intersect camera rays with a known calibration plane in camera coordinates."""

    rays = pixel_rays_camera(pixels, intrinsics)
    normal = np.asarray(plane_normal, dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm <= 1e-15:
        raise ValueError("Plane normal must be non-zero.")
    normal /= norm
    normalized_offset = float(plane_offset) / norm
    denominator = rays @ normal
    valid = np.abs(denominator) > 1e-9
    distances = np.full(rays.shape[0], np.nan, dtype=np.float64)
    distances[valid] = -normalized_offset / denominator[valid]
    return distances[:, None] * rays


def plane_homography_at_height(
    homography_px_to_mm_at_zero: np.ndarray,
    camera_matrix: np.ndarray,
    height_mm: float,
) -> np.ndarray:
    """Move a calibrated top-plane mapping from z=0 to z=height.

    The returned homography still maps the physical x-y coordinates attached
    to the calibration board, but projects them from the top-face plane.
    """

    from .laser_measure import pose_from_plane_homography

    homography_mm_to_px = np.linalg.inv(homography_px_to_mm_at_zero)
    pose = pose_from_plane_homography(homography_mm_to_px, camera_matrix)
    rotation = pose.rotation_camera_from_stage
    translation = pose.translation_camera_from_stage
    elevated = np.column_stack(
        (
            rotation[:, 0],
            rotation[:, 1],
            translation - rotation[:, 2] * float(height_mm),
        )
    )
    homography_elevated_mm_to_px = np.asarray(camera_matrix) @ elevated
    return np.linalg.inv(homography_elevated_mm_to_px)


def make_charuco_image(
    spec: CharucoSpec,
    pixels_per_mm: float = 10.0,
    margin_mm: float = 20.0,
) -> np.ndarray:
    board, _ = spec.board()
    margin_px = round(margin_mm * pixels_per_mm)
    width = round(
        (spec.squares_x * spec.square_length_mm + 2.0 * margin_mm) * pixels_per_mm
    )
    height = round(
        (spec.squares_y * spec.square_length_mm + 2.0 * margin_mm) * pixels_per_mm
    )
    return board.generateImage(
        (width, height),
        marginSize=margin_px,
        borderBits=1,
    )
