from __future__ import annotations

import math
import unittest

from volumecv.geometry import (
    BlindSlotDimensions,
    CylinderDimensions,
    LBlockDimensions,
    validate_dimensions,
)
from volumecv.uncertainty import volume_sigma_mm3


class GeometryTests(unittest.TestCase):
    def test_l_block_volume(self) -> None:
        dims = LBlockDimensions(a=80, b=70, h=50, u=30, v=20)
        self.assertAlmostEqual(dims.volume_mm3, (5600 - 600) * 50)

    def test_cylinder_volume(self) -> None:
        dims = CylinderDimensions(outer_diameter=80, inner_diameter=30, height=60)
        self.assertAlmostEqual(
            dims.volume_mm3,
            math.pi / 4 * (80**2 - 30**2) * 60,
        )

    def test_blind_slot_volume(self) -> None:
        dims = BlindSlotDimensions(a=90, b=80, h=60, u=50, v=40, depth=20)
        self.assertAlmostEqual(dims.volume_mm3, 90 * 80 * 60 - 50 * 40 * 20)

    def test_validation_rejects_impossible_cylinder(self) -> None:
        issues = validate_dimensions(
            "cylinder",
            {"D_mm": 80, "d_mm": 70, "H_mm": 60},
        )
        self.assertTrue(any(issue.code == "CYLINDER_WALL_TOO_THIN" for issue in issues))

    def test_uncertainty_is_positive(self) -> None:
        values = {
            "a_mm": 80,
            "b_mm": 70,
            "h_mm": 50,
            "u_mm": 30,
            "v_mm": 20,
        }
        sigmas = {key: 0.1 for key in values}
        self.assertGreater(volume_sigma_mm3("l_block", values, sigmas), 0.0)


if __name__ == "__main__":
    unittest.main()
