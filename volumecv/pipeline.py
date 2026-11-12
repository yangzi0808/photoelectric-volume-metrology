from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .features import (
    BBox,
    classify_top_features,
    extract_top_features,
    measure_side_height_px,
)
from .geometry import (
    MeasurementIssue,
    ShapeName,
    VolumeResult,
    validate_dimensions,
    volume_mm3,
)
from .laser import estimate_two_level_depth
from .synthetic import SyntheticCase
from .uncertainty import volume_sigma_mm3


class MeasurementFailure(RuntimeError):
    """The input does not contain the geometry required by the selected shape."""


@dataclass(frozen=True)
class MeasurementConfig:
    edge_sigma_px: float = 0.35
    scale_relative_sigma: float = 0.0015
    depth_systematic_sigma_mm: float = 0.03


def _bbox_dict(bbox: BBox) -> dict[str, int]:
    return {
        "x0": bbox.x0,
        "y0": bbox.y0,
        "x1": bbox.x1,
        "y1": bbox.y1,
        "width": bbox.width,
        "height": bbox.height,
    }


def _edge_sigma(
    dimension_mm: float,
    scale_px_per_mm: float,
    config: MeasurementConfig,
) -> float:
    pixel_term = np.sqrt(2.0) * config.edge_sigma_px / scale_px_per_mm
    scale_term = config.scale_relative_sigma * dimension_mm
    return float(np.hypot(pixel_term, scale_term))


def _validate_scale(value: float, name: str) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return float(value)


def measure_case(
    case: SyntheticCase,
    shape_mode: str = "auto",
    config: MeasurementConfig | None = None,
    top_px_per_mm_x: float | None = None,
    top_px_per_mm_y: float | None = None,
    side_px_per_mm: float | None = None,
) -> VolumeResult:
    """Run one measurement from a segmented top/side view and laser samples."""

    start = perf_counter()
    cfg = config or MeasurementConfig()
    scale_x = _validate_scale(
        top_px_per_mm_x if top_px_per_mm_x is not None else case.px_per_mm_x,
        "top_px_per_mm_x",
    )
    scale_y = _validate_scale(
        top_px_per_mm_y if top_px_per_mm_y is not None else case.px_per_mm_y,
        "top_px_per_mm_y",
    )
    scale_h = _validate_scale(
        side_px_per_mm if side_px_per_mm is not None else case.side_px_per_mm,
        "side_px_per_mm",
    )

    features = extract_top_features(case.top_mask)
    classified_shape, classification_confidence = classify_top_features(features)
    if shape_mode == "auto":
        shape: ShapeName = classified_shape  # type: ignore[assignment]
        mode = "auto"
        mode_confidence = classification_confidence
    elif shape_mode in {"l_block", "cylinder", "blind_slot"}:
        shape = shape_mode  # type: ignore[assignment]
        mode = "explicit"
        mode_confidence = 0.985
    else:
        raise ValueError("shape_mode must be auto, l_block, cylinder, or blind_slot.")

    box = features.bbox
    a = box.width / scale_x
    b = box.height / scale_y
    height = measure_side_height_px(case.side_mask) / scale_h
    issues: list[MeasurementIssue] = []
    dimensions: dict[str, float]
    sigmas: dict[str, float]
    diagnostics: dict[str, object] = {
        "top_bbox": _bbox_dict(box),
        "top_fill_ratio": round(features.outer_fill_ratio, 6),
        "aspect_ratio": round(features.aspect_ratio, 6),
        "classified_shape": classified_shape,
        "classification_confidence": round(classification_confidence, 6),
        "top_px_per_mm": [scale_x, scale_y],
        "side_px_per_mm": scale_h,
    }

    if shape == "l_block":
        if features.notch is None:
            raise MeasurementFailure(
                "Selected L-block but no corner notch was found in the top view."
            )
        notch = features.notch
        u = notch.width_px / scale_x
        v = notch.height_px / scale_y
        dimensions = {
            "a_mm": float(a),
            "b_mm": float(b),
            "h_mm": float(height),
            "u_mm": float(u),
            "v_mm": float(v),
        }
        sigmas = {
            "a_mm": _edge_sigma(a, scale_x, cfg),
            "b_mm": _edge_sigma(b, scale_y, cfg),
            "h_mm": _edge_sigma(height, scale_h, cfg),
            "u_mm": _edge_sigma(u, scale_x, cfg),
            "v_mm": _edge_sigma(v, scale_y, cfg),
        }
        diagnostics["notch"] = {
            "corner": notch.corner,
            "width_px": notch.width_px,
            "height_px": notch.height_px,
            "pixel_area": notch.pixel_area,
        }

    elif shape == "cylinder":
        if features.hole is None:
            raise MeasurementFailure(
                "Selected through-hole cylinder but no enclosed top-view hole was found."
            )
        hole = features.hole
        outer = ((box.width / scale_x) + (box.height / scale_y)) / 2.0
        inner = ((hole.bbox.width / scale_x) + (hole.bbox.height / scale_y)) / 2.0
        dimensions = {
            "D_mm": float(outer),
            "d_mm": float(inner),
            "H_mm": float(height),
        }
        sigmas = {
            "D_mm": _edge_sigma(outer, (scale_x + scale_y) / 2.0, cfg),
            "d_mm": _edge_sigma(inner, (scale_x + scale_y) / 2.0, cfg),
            "H_mm": _edge_sigma(height, scale_h, cfg),
        }
        diagnostics["hole"] = {
            "bbox": _bbox_dict(hole.bbox),
            "fill_ratio": round(hole.fill_ratio, 6),
            "pixel_area": hole.pixel_area,
        }

    else:
        if features.hole is None:
            raise MeasurementFailure(
                "Selected blind-slot block but no enclosed slot opening was found."
            )
        if case.depth_profile_mm is None:
            raise MeasurementFailure(
                "Selected blind-slot block but no line-laser depth profile was supplied."
            )
        hole = features.hole
        u = hole.bbox.width / scale_x
        v = hole.bbox.height / scale_y
        try:
            depth = estimate_two_level_depth(case.depth_profile_mm)
        except ValueError as exc:
            raise MeasurementFailure(str(exc)) from exc
        dimensions = {
            "a_mm": float(a),
            "b_mm": float(b),
            "h_mm": float(height),
            "u_mm": float(u),
            "v_mm": float(v),
            "t_mm": float(depth.depth_mm),
        }
        sigmas = {
            "a_mm": _edge_sigma(a, scale_x, cfg),
            "b_mm": _edge_sigma(b, scale_y, cfg),
            "h_mm": _edge_sigma(height, scale_h, cfg),
            "u_mm": _edge_sigma(u, scale_x, cfg),
            "v_mm": _edge_sigma(v, scale_y, cfg),
            "t_mm": float(
                np.hypot(
                    depth.sigma_mm,
                    cfg.depth_systematic_sigma_mm,
                )
            ),
        }
        diagnostics["slot"] = {
            "bbox": _bbox_dict(hole.bbox),
            "fill_ratio": round(hole.fill_ratio, 6),
            "pixel_area": hole.pixel_area,
        }
        diagnostics["laser_depth"] = {
            "top_level_mm": round(depth.top_level_mm, 6),
            "floor_level_mm": round(depth.floor_level_mm, 6),
            "inlier_count": depth.inlier_count,
            "rejected_count": depth.rejected_count,
            "sample_sigma_mm": round(depth.sigma_mm, 6),
        }

    issues.extend(validate_dimensions(shape, dimensions))
    measured_volume = volume_mm3(shape, dimensions)
    sigma = volume_sigma_mm3(shape, dimensions, sigmas)
    relative_sigma = sigma / measured_volume if measured_volume > 0 else float("inf")
    error_count = sum(issue.severity == "error" for issue in issues)
    warning_count = sum(issue.severity == "warning" for issue in issues)
    confidence = (
        mode_confidence
        * np.exp(-relative_sigma / 0.03)
        * (0.78**error_count)
        * (0.94**warning_count)
    )
    confidence = float(np.clip(confidence, 0.02, 0.995))
    diagnostics["dimension_sigma_mm"] = {
        key: round(value, 6) for key, value in sigmas.items()
    }
    elapsed_ms = (perf_counter() - start) * 1000.0
    return VolumeResult(
        shape=shape,
        dimensions={key: float(value) for key, value in dimensions.items()},
        volume_mm3=float(measured_volume),
        volume_cm3=float(measured_volume / 1000.0),
        sigma_mm3=float(sigma),
        relative_sigma=float(relative_sigma),
        confidence=confidence,
        issues=tuple(issues),
        elapsed_ms=elapsed_ms,
        mode=mode,
        diagnostics=diagnostics,
    )
