from __future__ import annotations

import unittest

import numpy as np

from volumecv.laser import estimate_two_level_depth


class LaserTests(unittest.TestCase):
    def test_robust_depth_estimate(self) -> None:
        rng = np.random.default_rng(42)
        samples = np.r_[
            rng.normal(0.0, 0.04, 100),
            rng.normal(-18.0, 0.04, 70),
            np.array([6.0, -9.0, 15.0, -30.0]),
        ]
        estimate = estimate_two_level_depth(samples)
        self.assertAlmostEqual(estimate.depth_mm, 18.0, delta=0.08)
        self.assertGreater(estimate.rejected_count, 0)

    def test_single_level_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            estimate_two_level_depth(np.zeros(30))


if __name__ == "__main__":
    unittest.main()
