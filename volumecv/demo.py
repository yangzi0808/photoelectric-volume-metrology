from __future__ import annotations

from pathlib import Path

import numpy as np

from .benchmark import BenchmarkSummary, run_benchmark
from .geometry import VolumeResult
from .pipeline import measure_case
from .reporting import write_report
from .synthetic import SyntheticCase, make_demo_cases


def repeat_demo_cases(
    seed: int,
) -> list[tuple[SyntheticCase, list[VolumeResult]]]:
    rng = np.random.default_rng(seed)
    base_cases = make_demo_cases(seed)
    records: list[tuple[SyntheticCase, list[VolumeResult]]] = []
    for case in base_cases:
        results: list[VolumeResult] = []
        for repeat in range(2):
            repeat_case = SyntheticCase(
                name=f"{case.name} / run {repeat + 1}",
                shape=case.shape,
                top_mask=case.top_mask.copy(),
                side_mask=case.side_mask.copy(),
                depth_profile_mm=(
                    None
                    if case.depth_profile_mm is None
                    else np.r_[
                        rng.normal(0.0, 0.06, 90),
                        rng.normal(
                            -case.truth_dimensions["t_mm"],
                            0.06,
                            55,
                        ),
                    ]
                ),
                truth_dimensions=case.truth_dimensions,
                px_per_mm_x=case.px_per_mm_x,
                px_per_mm_y=case.px_per_mm_y,
                side_px_per_mm=case.side_px_per_mm,
            )
            bias_x = float(rng.normal(0.0, 0.0008))
            bias_y = float(rng.normal(0.0, 0.0008))
            bias_h = float(rng.normal(0.0, 0.0008))
            result = measure_case(
                repeat_case,
                shape_mode="auto",
                top_px_per_mm_x=repeat_case.px_per_mm_x * (1.0 + bias_x),
                top_px_per_mm_y=repeat_case.px_per_mm_y * (1.0 + bias_y),
                side_px_per_mm=repeat_case.side_px_per_mm * (1.0 + bias_h),
            )
            results.append(result)
        records.append((case, results))
    return records


def run_demo(
    output_dir: Path,
    seed: int = 20261110,
    benchmark_per_shape: int = 35,
) -> tuple[
    list[tuple[SyntheticCase, list[VolumeResult]]],
    BenchmarkSummary,
    dict[str, Path],
]:
    records = repeat_demo_cases(seed)
    benchmark = run_benchmark(
        per_shape=benchmark_per_shape,
        repeats=2,
        seed=seed + 1,
    )
    paths = write_report(records, benchmark, output_dir, seed)
    return records, benchmark, paths
