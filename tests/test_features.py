from __future__ import annotations

import unittest

import numpy as np

from volumecv.features import (
    classify_top_features,
    extract_top_features,
    find_interior_hole,
)
from volumecv.synthetic import make_blind_slot_mask, make_cylinder_mask, make_l_mask


class FeatureTests(unittest.TestCase):
    def test_l_block_corner_notch(self) -> None:
        mask = make_l_mask(80, 70, 30, 20, 10, 10)
        features = extract_top_features(mask)
        self.assertEqual(features.bbox.width, 800)
        self.assertEqual(features.bbox.height, 700)
        self.assertIsNotNone(features.notch)
        self.assertIsNone(features.hole)
        self.assertEqual(features.notch.width_px, 300)
        self.assertEqual(features.notch.height_px, 200)
        self.assertEqual(classify_top_features(features)[0], "l_block")

    def test_cylinder_has_enclosed_hole(self) -> None:
        mask = make_cylinder_mask(80, 30, 10)
        features = extract_top_features(mask)
        self.assertIsNotNone(features.hole)
        self.assertIsNone(features.notch)
        self.assertEqual(classify_top_features(features)[0], "cylinder")

    def test_rectangular_enclosed_hole_is_blind_slot(self) -> None:
        mask = make_blind_slot_mask(90, 80, 50, 40, 10, 10)
        features = extract_top_features(mask)
        self.assertIsNotNone(features.hole)
        self.assertEqual(features.hole.bbox.width, 500)
        self.assertEqual(features.hole.bbox.height, 400)
        self.assertEqual(classify_top_features(features)[0], "blind_slot")

    def test_empty_mask_fails_cleanly(self) -> None:
        with self.assertRaises(ValueError):
            find_interior_hole(np.zeros((20, 20), dtype=bool))


if __name__ == "__main__":
    unittest.main()
