from __future__ import annotations

from time import perf_counter

import cv2
import numpy as np

from .calibration import CameraIntrinsics, plane_homography_at_height
from .geometry import (
    MeasurementIssue,
    ShapeName,
    VolumeResult,
    validate_dimensions,
    volume_mm3,
)
from .image_measure import measure_side_height_image, measure_top_image
from .laser import estimate_two_level_depth
from .realistic import RealisticImageCase
from .uncertainty import volume_sigma_mm3


def _metric_scale(homography_px_to_mm: np.ndarray) -> float:
    determinant = float(np.linalg.det(homography_px_to_mm[:2, :2]))
    if abs(determinant) < 1e-15:
        raise ValueError("The calibration homography is singular.")
    return float(np.sqrt(abs(determinant)))


def _undistort_if_available(
    image: np.ndarray,
    intrinsics: CameraIntrinsics | None,
    name: str,
) -> np.ndarray:
    if intrinsics is None:
        return image
    height, width = image.shape[:2]
    if (width, height) != intrinsics.image_size:
        raise ValueError(
            f"{name} image size {width}x{height} does not match "
            f"calibration {intrinsics.image_size[0]}x{intrinsics.image_size[1]}."
        )
    return cv2.undistort(
        image,
        intrinsics.camera_matrix,
        intrinsics.distortion,
    )


def _crop_with_calibration(
    image: np.ndarray,
    homography_px_to_mm: np.ndarray,
    roi: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object] | None]:
    if roi is None:
        return image, homography_px_to_mm, None
    x0, y0, x1, y1 = roi
    height, width = image.shape[:2]
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0 or x1 > width or y1 > height:
        raise ValueError(f"ROI {(x0, y0, x1, y1)} is outside image {width}x{height}.")
    crop = image[y0:y1, x0:x1]
    shift = np.array(
        [[1.0, 0.0, x0], [0.0, 1.0, y0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return (
        crop,
        np.asarray(homography_px_to_mm) @ shift,
        {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
    )


def _length_sigma(
    length_mm: float,
    metric_scale_mm_per_px: float,
    edge_sigma_px: float = 0.22,
    calibration_relative_sigma: float = 0.0008,
) -> float:
    pixel_term = np.sqrt(2.0) * edge_sigma_px * metric_scale_mm_per_px
    calibration_term = calibration_relative_sigma * length_mm
    return float(np.hypot(pixel_term, calibration_term))


def measure_calibrated_images(
    top_image: np.ndarray,
    side_image: np.ndarray,
    top_homography_px_to_mm: np.ndarray,
    side_homography_px_to_mm: np.ndarray,
    depth_profile_mm: np.ndarray | None = None,
    shape_mode: str = "auto",
    camera_matrix: np.ndarray | None = None,
    compensate_top_plane_height: bool = True,
    top_intrinsics: CameraIntrinsics | None = None,
    side_intrinsics: CameraIntrinsics | None = None,
    top_roi: tuple[int, int, int, int] | None = None,
    side_roi: tuple[int, int, int, int] | None = None,
) -> VolumeResult:
    """Measure one calibrated top/side image pair using the image engine."""

    start = perf_counter()
    top_image = _undistort_if_available(top_image, top_intrinsics, "Top")
    side_image = _undistort_if_available(side_image, side_intrinsics, "Side")
    top_image, calibrated_top_homography, top_roi_diagnostics = _crop_with_calibration(
        top_image,
        top_homography_px_to_mm,
        top_roi,
    )
    side_image, calibrated_side_homography, side_roi_diagnostics = (
        _crop_with_calibration(
            side_image,
            side_homography_px_to_mm,
            side_roi,
        )
    )
    effective_camera_matrix = (
        camera_matrix
        if camera_matrix is not None
        else top_intrinsics.camera_matrix
        if top_intrinsics is not None
        else None
    )
    height, side_diagnostics = measure_side_height_image(
        side_image,
        calibrated_side_homography,
        object_is_dark=True,
    )
    effective_top_homography = calibrated_top_homography
    if effective_camera_matrix is not None and compensate_top_plane_height:
        effective_top_homography = plane_homography_at_height(
            calibrated_top_homography,
            effective_camera_matrix,
            height,
        )
    top = measure_top_image(
        top_image,
        effective_top_homography,
        shape_mode=shape_mode,
        object_is_dark=True,
    )
    shape: ShapeName = top.shape  # type: ignore[assignment]
    top_scale = _metric_scale(effective_top_homography)
    side_scale = _metric_scale(calibrated_side_homography)
    issues: list[MeasurementIssue] = []
    diagnostics: dict[str, object] = {
        "top": top.diagnostics,
        "side": side_diagnostics,
        "top_metric_mm_per_px": top_scale,
        "side_metric_mm_per_px": side_scale,
        "classifier_confidence": top.confidence,
        "height_compensated_top_plane": bool(
            effective_camera_matrix is not None and compensate_top_plane_height
        ),
        "effective_top_homography_px_to_mm": effective_top_homography.tolist(),
        "top_roi_px": top_roi_diagnostics,
        "side_roi_px": side_roi_diagnostics,
    }

    dimensions: dict[str, float]
    sigmas: dict[str, float]
    if shape == "l_block":
        dimensions = {
            "a_mm": top.dimensions["a_mm"],
            "b_mm": top.dimensions["b_mm"],
            "h_mm": height,
            "u_mm": top.dimensions["u_mm"],
            "v_mm": top.dimensions["v_mm"],
        }
        sigmas = {
            "a_mm": _length_sigma(dimensions["a_mm"], top_scale),
            "b_mm": _length_sigma(dimensions["b_mm"], top_scale),
            "h_mm": _length_sigma(height, side_scale),
            "u_mm": _length_sigma(dimensions["u_mm"], top_scale),
            "v_mm": _length_sigma(dimensions["v_mm"], top_scale),
        }

    elif shape == "cylinder":
        dimensions = {
            "D_mm": top.dimensions["D_mm"],
            "d_mm": top.dimensions["d_mm"],
            "H_mm": height,
        }
        sigmas = {
            "D_mm": _length_sigma(dimensions["D_mm"], top_scale),
            "d_mm": _length_sigma(dimensions["d_mm"], top_scale),
            "H_mm": _length_sigma(height, side_scale),
        }

    else:
        if depth_profile_mm is None:
            raise ValueError("Blind-slot image case has no laser depth profile.")
        depth = estimate_two_level_depth(depth_profile_mm)
        dimensions = {
            "a_mm": top.dimensions["a_mm"],
            "b_mm": top.dimensions["b_mm"],
            "h_mm": height,
            "u_mm": top.dimensions["u_mm"],
            "v_mm": top.dimensions["v_mm"],
            "t_mm": depth.depth_mm,
        }
        sigmas = {
            "a_mm": _length_sigma(dimensions["a_mm"], top_scale),
            "b_mm": _length_sigma(dimensions["b_mm"], top_scale),
            "h_mm": _length_sigma(height, side_scale),
            "u_mm": _length_sigma(dimensions["u_mm"], top_scale),
            "v_mm": _length_sigma(dimensions["v_mm"], top_scale),
            "t_mm": float(np.hypot(depth.sigma_mm, 0.03)),
        }
        diagnostics["laser_depth"] = {
            "top_level_mm": depth.top_level_mm,
            "floor_level_mm": depth.floor_level_mm,
            "sample_sigma_mm": depth.sigma_mm,
            "inlier_count": depth.inlier_count,
            "rejected_count": depth.rejected_count,
        }

    issues.extend(validate_dimensions(shape, dimensions))
    measured_volume = volume_mm3(shape, dimensions)
    sigma = volume_sigma_mm3(shape, dimensions, sigmas)
    relative_sigma = sigma / measured_volume if measured_volume > 0 else float("inf")
    error_count = sum(issue.severity == "error" for issue in issues)
    warning_count = sum(issue.severity == "warning" for issue in issues)
    confidence = (
        top.confidence
        * float(np.exp(-relative_sigma / 0.025))
        * (0.75**error_count)
        * (0.93**warning_count)
    )
    confidence = float(np.clip(confidence, 0.02, 0.995))
    diagnostics["dimension_sigma_mm"] = {
        key: round(value, 6) for key, value in sigmas.items()
    }
    return VolumeResult(
        shape=shape,
        dimensions={key: float(value) for key, value in dimensions.items()},
        volume_mm3=float(measured_volume),
        volume_cm3=float(measured_volume / 1000.0),
        sigma_mm3=float(sigma),
        relative_sigma=float(relative_sigma),
        confidence=confidence,
        issues=tuple(issues),
        elapsed_ms=(perf_counter() - start) * 1000.0,
        mode="image_auto" if shape_mode == "auto" else "image_explicit",
        diagnostics=diagnostics,
    )


def measure_realistic_case(
    case: RealisticImageCase,
    shape_mode: str = "auto",
) -> VolumeResult:
    return measure_calibrated_images(
        case.top_gray,
        case.side_gray,
        case.top_homography_px_to_mm,
        case.side_homography_px_to_mm,
        depth_profile_mm=case.depth_profile_mm,
        shape_mode=shape_mode,
    )
