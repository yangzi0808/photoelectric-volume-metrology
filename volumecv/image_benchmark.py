from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import cv2
import numpy as np

from .geometry import volume_mm3
from .image_pipeline import measure_calibrated_images
from .pipeline import MeasurementFailure
from .realistic import make_realistic_case, random_realistic_case


@dataclass(frozen=True)
class ImageBenchmarkRow:
    case_id: str
    shape: str
    repeat: int
    truth_cm3: float
    measured_cm3: float
    relative_error: float
    predicted_relative_sigma: float
    confidence: float
    correct_shape: bool
    within_2sigma: bool
    elapsed_ms: float

    def as_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "shape": self.shape,
            "repeat": self.repeat,
            "truth_cm3": round(self.truth_cm3, 6),
            "measured_cm3": round(self.measured_cm3, 6),
            "relative_error": round(self.relative_error, 8),
            "predicted_relative_sigma": round(self.predicted_relative_sigma, 8),
            "confidence": round(self.confidence, 6),
            "correct_shape": self.correct_shape,
            "within_2sigma": self.within_2sigma,
            "elapsed_ms": round(self.elapsed_ms, 6),
        }


@dataclass(frozen=True)
class ImageBenchmarkSummary:
    rows: tuple[ImageBenchmarkRow, ...]
    mean_abs_error: float
    p95_abs_error: float
    max_abs_error: float
    pass_5pct_rate: float
    shape_accuracy: float
    mean_repeat_difference: float
    p95_repeat_difference: float
    two_sigma_coverage: float
    mean_confidence: float
    failures: tuple[dict[str, str], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "sample_count": len(self.rows),
            "mean_abs_error": round(self.mean_abs_error, 8),
            "p95_abs_error": round(self.p95_abs_error, 8),
            "max_abs_error": round(self.max_abs_error, 8),
            "pass_5pct_rate": round(self.pass_5pct_rate, 8),
            "shape_accuracy": round(self.shape_accuracy, 8),
            "mean_repeat_difference": round(self.mean_repeat_difference, 8),
            "p95_repeat_difference": round(self.p95_repeat_difference, 8),
            "two_sigma_coverage": round(self.two_sigma_coverage, 8),
            "mean_confidence": round(self.mean_confidence, 8),
            "failure_count": len(self.failures),
            "failures": list(self.failures),
        }


def _perturb_plane(
    homography_px_to_mm: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    scale_x = 1.0 + float(rng.normal(0.0, 0.00065))
    scale_y = 1.0 + float(rng.normal(0.0, 0.00065))
    angle = np.deg2rad(float(rng.normal(0.0, 0.00035)))
    cosine, sine = np.cos(angle), np.sin(angle)
    perturbation = np.array(
        [
            [scale_x * cosine, -scale_y * sine, rng.normal(0.0, 0.018)],
            [scale_x * sine, scale_y * cosine, rng.normal(0.0, 0.018)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return perturbation @ homography_px_to_mm


def _summarize(
    rows: list[ImageBenchmarkRow],
    failures: list[dict[str, str]],
) -> ImageBenchmarkSummary:
    errors = np.asarray([row.relative_error for row in rows], dtype=np.float64)
    repeat_groups: dict[str, list[float]] = {}
    for row in rows:
        repeat_groups.setdefault(row.case_id, []).append(row.measured_cm3)
    repeat_differences = []
    for values in repeat_groups.values():
        if len(values) >= 2:
            repeat_differences.append(
                abs(values[0] - values[1]) / mean([values[0], values[1]])
            )
    repeat_array = np.asarray(repeat_differences, dtype=np.float64)
    return ImageBenchmarkSummary(
        rows=tuple(rows),
        mean_abs_error=float(errors.mean()),
        p95_abs_error=float(np.quantile(errors, 0.95)),
        max_abs_error=float(errors.max()),
        pass_5pct_rate=float(np.mean(errors <= 0.05)),
        shape_accuracy=float(np.mean([row.correct_shape for row in rows])),
        mean_repeat_difference=(
            float(repeat_array.mean()) if repeat_array.size else 0.0
        ),
        p95_repeat_difference=(
            float(np.quantile(repeat_array, 0.95)) if repeat_array.size else 0.0
        ),
        two_sigma_coverage=float(np.mean([row.within_2sigma for row in rows])),
        mean_confidence=float(np.mean([row.confidence for row in rows])),
        failures=tuple(failures),
    )


def run_image_benchmark(
    per_shape: int = 20,
    repeats: int = 2,
    seed: int = 20261110,
) -> ImageBenchmarkSummary:
    if per_shape <= 0 or repeats <= 0:
        raise ValueError("per_shape and repeats must be positive.")
    rng = np.random.default_rng(seed)
    rows: list[ImageBenchmarkRow] = []
    failures: list[dict[str, str]] = []
    for shape in ("l_block", "cylinder", "blind_slot"):
        for index in range(per_shape):
            case_id = f"{shape}-{index:04d}"
            reference_case = random_realistic_case(
                shape,
                rng,
                name=case_id,
            )
            truth_cm3 = (
                volume_mm3(shape, reference_case.truth_dimensions) / 1000.0  # type: ignore[arg-type]
            )
            for repeat in range(repeats):
                case = make_realistic_case(
                    shape,
                    reference_case.truth_dimensions,
                    rng,
                    name=f"{case_id}-r{repeat}",
                )
                estimated_top = _perturb_plane(case.top_homography_px_to_mm, rng)
                estimated_side = _perturb_plane(case.side_homography_px_to_mm, rng)
                try:
                    result = measure_calibrated_images(
                        case.top_gray,
                        case.side_gray,
                        estimated_top,
                        estimated_side,
                        depth_profile_mm=case.depth_profile_mm,
                        shape_mode="auto",
                    )
                except (
                    MeasurementFailure,
                    ValueError,
                    RuntimeError,
                    OSError,
                    cv2.error,
                ) as exc:
                    failures.append(
                        {
                            "case_id": case_id,
                            "repeat": str(repeat),
                            "shape": shape,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                relative_error = abs(result.volume_cm3 - truth_cm3) / truth_cm3
                rows.append(
                    ImageBenchmarkRow(
                        case_id=case_id,
                        shape=shape,
                        repeat=repeat,
                        truth_cm3=truth_cm3,
                        measured_cm3=result.volume_cm3,
                        relative_error=relative_error,
                        predicted_relative_sigma=result.relative_sigma,
                        confidence=result.confidence,
                        correct_shape=result.shape == shape,
                        within_2sigma=(
                            abs(result.volume_cm3 - truth_cm3)
                            <= 2.0 * result.sigma_mm3 / 1000.0
                        ),
                        elapsed_ms=result.elapsed_ms,
                    )
                )
    if not rows:
        raise RuntimeError("Every realistic image benchmark case failed.")
    return _summarize(rows, failures)


def write_image_benchmark(
    summary: ImageBenchmarkSummary,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "image_benchmark.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(summary.rows[0].as_dict().keys()),
        )
        writer.writeheader()
        for row in summary.rows:
            writer.writerow(row.as_dict())
    summary_path = output_dir / "image_benchmark_summary.json"
    summary_path.write_text(
        json.dumps(summary.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"csv": csv_path, "summary": summary_path}
