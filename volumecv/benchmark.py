from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import mean

import numpy as np

from .geometry import volume_mm3
from .pipeline import MeasurementFailure, measure_case
from .synthetic import (
    make_case,
    random_blind_slot_dimensions,
    random_cylinder_dimensions,
    random_l_dimensions,
)


@dataclass(frozen=True)
class BenchmarkRow:
    case_id: str
    shape: str
    repeat: int
    truth_cm3: float
    measured_cm3: float
    relative_error: float
    predicted_relative_sigma: float
    confidence: float
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
            "within_2sigma": self.within_2sigma,
            "elapsed_ms": round(self.elapsed_ms, 6),
        }


@dataclass(frozen=True)
class BenchmarkSummary:
    rows: tuple[BenchmarkRow, ...]
    mean_abs_error: float
    p95_abs_error: float
    max_abs_error: float
    pass_5pct_rate: float
    mean_repeat_difference: float
    p95_repeat_difference: float
    two_sigma_coverage: float
    failure_count: int
    failed_cases: tuple[dict[str, str], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "sample_count": len(self.rows),
            "mean_abs_error": round(self.mean_abs_error, 8),
            "p95_abs_error": round(self.p95_abs_error, 8),
            "max_abs_error": round(self.max_abs_error, 8),
            "pass_5pct_rate": round(self.pass_5pct_rate, 8),
            "mean_repeat_difference": round(self.mean_repeat_difference, 8),
            "p95_repeat_difference": round(self.p95_repeat_difference, 8),
            "two_sigma_coverage": round(self.two_sigma_coverage, 8),
            "failure_count": self.failure_count,
            "failed_cases": list(self.failed_cases),
        }


def _summary(
    rows: Iterable[BenchmarkRow],
    failures: list[dict[str, str]],
) -> BenchmarkSummary:
    data = tuple(rows)
    errors = np.array([row.relative_error for row in data], dtype=np.float64)
    by_case: dict[str, list[float]] = {}
    for row in data:
        by_case.setdefault(row.case_id, []).append(row.measured_cm3)
    repeat_differences = []
    for values in by_case.values():
        if len(values) >= 2:
            first, second = values[0], values[1]
            repeat_differences.append(abs(first - second) / mean([first, second]))
    differences = np.asarray(repeat_differences, dtype=np.float64)
    return BenchmarkSummary(
        rows=data,
        mean_abs_error=float(errors.mean()),
        p95_abs_error=float(np.quantile(errors, 0.95)),
        max_abs_error=float(errors.max()),
        pass_5pct_rate=float(np.mean(errors <= 0.05)),
        mean_repeat_difference=(float(differences.mean()) if differences.size else 0.0),
        p95_repeat_difference=(
            float(np.quantile(differences, 0.95)) if differences.size else 0.0
        ),
        two_sigma_coverage=float(np.mean([row.within_2sigma for row in data])),
        failure_count=len(failures),
        failed_cases=tuple(failures),
    )


def run_benchmark(
    per_shape: int = 100,
    repeats: int = 2,
    seed: int = 20261110,
) -> BenchmarkSummary:
    if per_shape <= 0:
        raise ValueError("per_shape must be positive.")
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    rng = np.random.default_rng(seed)
    rows: list[BenchmarkRow] = []
    failures: list[dict[str, str]] = []
    generators = {
        "l_block": random_l_dimensions,
        "cylinder": random_cylinder_dimensions,
        "blind_slot": random_blind_slot_dimensions,
    }

    for shape, generator in generators.items():
        for index in range(per_shape):
            dimensions = generator(rng)
            truth_mm3 = volume_mm3(shape, dimensions)  # type: ignore[arg-type]
            truth_cm3 = truth_mm3 / 1000.0
            case_id = f"{shape}-{index:04d}"
            for repeat in range(repeats):
                scale = float(rng.uniform(8.0, 12.0))
                scale_y = (
                    scale
                    if shape == "cylinder"
                    else scale * float(rng.uniform(0.997, 1.003))
                )
                side_scale = float(rng.uniform(8.0, 12.0))
                mask_operation = int(rng.choice([-1, 0, 0, 1]))
                case = make_case(
                    shape,
                    dimensions,
                    rng,
                    px_per_mm_x=scale,
                    px_per_mm_y=scale_y,
                    side_px_per_mm=side_scale,
                    mask_operation=mask_operation,
                    name=case_id,
                )
                calibration_bias_x = float(rng.normal(0.0, 0.0012))
                calibration_bias_y = float(rng.normal(0.0, 0.0012))
                calibration_bias_h = float(rng.normal(0.0, 0.0012))
                try:
                    result = measure_case(
                        case,
                        shape_mode="auto",
                        top_px_per_mm_x=scale * (1.0 + calibration_bias_x),
                        top_px_per_mm_y=scale_y * (1.0 + calibration_bias_y),
                        side_px_per_mm=side_scale * (1.0 + calibration_bias_h),
                    )
                except (MeasurementFailure, ValueError) as exc:
                    failures.append(
                        {
                            "case_id": case_id,
                            "repeat": str(repeat),
                            "shape": shape,
                            "error": str(exc),
                        }
                    )
                    continue
                relative_error = abs(result.volume_cm3 - truth_cm3) / truth_cm3
                rows.append(
                    BenchmarkRow(
                        case_id=case_id,
                        shape=shape,
                        repeat=repeat,
                        truth_cm3=truth_cm3,
                        measured_cm3=result.volume_cm3,
                        relative_error=relative_error,
                        predicted_relative_sigma=result.relative_sigma,
                        confidence=result.confidence,
                        within_2sigma=(
                            abs(result.volume_cm3 - truth_cm3)
                            <= 2.0 * result.sigma_mm3 / 1000.0
                        ),
                        elapsed_ms=result.elapsed_ms,
                    )
                )

    if not rows:
        raise RuntimeError("All synthetic benchmark cases failed.")
    return _summary(rows, failures)
