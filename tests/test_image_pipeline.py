from __future__ import annotations

import unittest

import cv2
import numpy as np

from volumecv.features import tight_bbox
from volumecv.geometry import volume_mm3
from volumecv.image_measure import measure_top_image, segment_backlit_object
from volumecv.image_pipeline import measure_calibrated_images, measure_realistic_case
from volumecv.realistic import make_parallax_case, make_realistic_demo_cases
from volumecv.synthetic import make_blind_slot_mask


class ImagePipelineTests(unittest.TestCase):
    def test_realistic_image_demo(self) -> None:
        for case in make_realistic_demo_cases(seed=31):
            result = measure_realistic_case(case, shape_mode="auto")
            truth = (
                volume_mm3(case.shape, case.truth_dimensions) / 1000.0  # type: ignore[arg-type]
            )
            relative_error = abs(result.volume_cm3 - truth) / truth
            self.assertEqual(result.shape, case.shape)
            self.assertLess(relative_error, 0.01)

    def test_height_compensation_removes_parallax_bias(self) -> None:
        import numpy as np

        dimensions = {
            "a_mm": 82.0,
            "b_mm": 71.0,
            "h_mm": 58.0,
            "u_mm": 31.0,
            "v_mm": 26.0,
        }
        case, camera_matrix = make_parallax_case(
            "l_block",
            dimensions,
            np.random.default_rng(99),
        )
        truth = volume_mm3("l_block", dimensions) / 1000.0
        baseline = measure_calibrated_images(
            case.top_gray,
            case.side_gray,
            case.top_homography_px_to_mm,
            case.side_homography_px_to_mm,
            shape_mode="l_block",
            camera_matrix=camera_matrix,
            compensate_top_plane_height=False,
        )
        corrected = measure_calibrated_images(
            case.top_gray,
            case.side_gray,
            case.top_homography_px_to_mm,
            case.side_homography_px_to_mm,
            shape_mode="l_block",
            camera_matrix=camera_matrix,
            compensate_top_plane_height=True,
        )
        self.assertGreater(abs(baseline.volume_cm3 - truth) / truth, 0.2)
        self.assertLess(abs(corrected.volume_cm3 - truth) / truth, 0.02)

    def test_large_rectangular_blind_slot_is_not_a_cylinder(self) -> None:
        mask = make_blind_slot_mask(
            60.86,
            81.33,
            35.81,
            49.70,
            10.0,
            10.0,
        )
        image = np.where(mask, 70, 225).astype(np.uint8)
        image = cv2.GaussianBlur(image, (0, 0), 0.8)
        homography_px_to_mm = np.diag([0.1, 0.1, 1.0])
        measurement = measure_top_image(
            image,
            homography_px_to_mm,
            shape_mode="auto",
        )
        self.assertEqual(measurement.shape, "blind_slot")

    def test_calibrated_roi_preserves_dimensions(self) -> None:
        case = make_realistic_demo_cases(seed=17)[0]
        box = tight_bbox(segment_backlit_object(case.top_gray))
        roi = (
            max(0, box.x0 - 20),
            max(0, box.y0 - 20),
            min(case.top_gray.shape[1], box.x1 + 21),
            min(case.top_gray.shape[0], box.y1 + 21),
        )
        full = measure_calibrated_images(
            case.top_gray,
            case.side_gray,
            case.top_homography_px_to_mm,
            case.side_homography_px_to_mm,
            shape_mode="l_block",
        )
        cropped = measure_calibrated_images(
            case.top_gray,
            case.side_gray,
            case.top_homography_px_to_mm,
            case.side_homography_px_to_mm,
            shape_mode="l_block",
            top_roi=roi,
        )
        self.assertAlmostEqual(full.volume_cm3, cropped.volume_cm3, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
