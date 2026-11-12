from __future__ import annotations

import unittest

from volumecv.benchmark import run_benchmark
from volumecv.geometry import volume_mm3
from volumecv.pipeline import measure_case
from volumecv.synthetic import make_demo_cases


class PipelineTests(unittest.TestCase):
    def test_demo_cases_end_to_end(self) -> None:
        for case in make_demo_cases():
            result = measure_case(case, shape_mode="auto")
            truth = volume_mm3(case.shape, case.truth_dimensions) / 1000.0  # type: ignore[arg-type]
            self.assertLess(abs(result.volume_cm3 - truth) / truth, 0.01)
            self.assertFalse(result.has_errors)
            self.assertEqual(result.shape, case.shape)

    def test_known_calibration_bias_propagates(self) -> None:
        case = make_demo_cases()[0]
        result = measure_case(
            case,
            shape_mode="auto",
            top_px_per_mm_x=case.px_per_mm_x * 1.01,
            top_px_per_mm_y=case.px_per_mm_y * 1.01,
        )
        self.assertGreater(result.volume_cm3, 0.0)
        self.assertGreater(result.relative_sigma, 0.0)

    def test_small_random_benchmark(self) -> None:
        summary = run_benchmark(per_shape=3, repeats=2, seed=7)
        self.assertEqual(summary.failure_count, 0)
        self.assertGreaterEqual(summary.pass_5pct_rate, 0.9)


if __name__ == "__main__":
    unittest.main()
