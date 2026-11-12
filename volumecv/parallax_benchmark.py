from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .geometry import volume_mm3
from .image_pipeline import measure_calibrated_images
from .realistic import make_parallax_case
from .synthetic import (
    random_blind_slot_dimensions,
    random_cylinder_dimensions,
    random_l_dimensions,
)


def run_parallax_benchmark(
    per_shape: int = 5,
    seed: int = 20261110,
) -> list[dict[str, object]]:
    if per_shape <= 0:
        raise ValueError("per_shape must be positive.")
    rng = np.random.default_rng(seed)
    generators = {
        "l_block": random_l_dimensions,
        "cylinder": random_cylinder_dimensions,
        "blind_slot": random_blind_slot_dimensions,
    }
    rows: list[dict[str, object]] = []
    for shape, generator in generators.items():
        for index in range(per_shape):
            dimensions = generator(rng)
            truth_cm3 = volume_mm3(shape, dimensions) / 1000.0  # type: ignore[arg-type]
            case, camera_matrix = make_parallax_case(
                shape,
                dimensions,
                rng,
            )
            baseline = measure_calibrated_images(
                case.top_gray,
                case.side_gray,
                case.top_homography_px_to_mm,
                case.side_homography_px_to_mm,
                depth_profile_mm=case.depth_profile_mm,
                shape_mode=shape,
                camera_matrix=camera_matrix,
                compensate_top_plane_height=False,
            )
            corrected = measure_calibrated_images(
                case.top_gray,
                case.side_gray,
                case.top_homography_px_to_mm,
                case.side_homography_px_to_mm,
                depth_profile_mm=case.depth_profile_mm,
                shape_mode=shape,
                camera_matrix=camera_matrix,
                compensate_top_plane_height=True,
            )
            rows.append(
                {
                    "case_id": f"{shape}-{index:02d}",
                    "shape": shape,
                    "truth_cm3": round(truth_cm3, 6),
                    "baseline_cm3": round(baseline.volume_cm3, 6),
                    "baseline_error": round(
                        abs(baseline.volume_cm3 - truth_cm3) / truth_cm3,
                        8,
                    ),
                    "corrected_cm3": round(corrected.volume_cm3, 6),
                    "corrected_error": round(
                        abs(corrected.volume_cm3 - truth_cm3) / truth_cm3,
                        8,
                    ),
                    "object_height_mm": case.render_metadata["object_height_mm"],
                }
            )
    return rows


def write_parallax_benchmark(
    rows: list[dict[str, object]],
    output_dir: Path,
) -> dict[str, Path]:
    if not rows:
        raise ValueError("No parallax benchmark rows were produced.")
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = np.asarray([float(row["baseline_error"]) for row in rows])
    corrected = np.asarray([float(row["corrected_error"]) for row in rows])
    summary = {
        "sample_count": len(rows),
        "baseline_mean_error": float(baseline.mean()),
        "baseline_p95_error": float(np.quantile(baseline, 0.95)),
        "baseline_max_error": float(baseline.max()),
        "corrected_mean_error": float(corrected.mean()),
        "corrected_p95_error": float(np.quantile(corrected, 0.95)),
        "corrected_max_error": float(corrected.max()),
        "corrected_pass_5pct_rate": float(np.mean(corrected <= 0.05)),
    }
    csv_path = output_dir / "parallax_benchmark.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "parallax_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"csv": csv_path, "summary": summary_path}
