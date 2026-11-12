from __future__ import annotations

import unittest

import cv2
import numpy as np

from volumecv.calibration import (
    CharucoSpec,
    calibrate_intrinsics,
    calibrate_plane_homography,
    detect_charuco,
    make_charuco_image,
    plane_homography_at_height,
)


class CalibrationTests(unittest.TestCase):
    def test_charuco_board_detection_and_plane_scale(self) -> None:
        spec = CharucoSpec()
        image = make_charuco_image(spec, pixels_per_mm=8.0)
        corners, ids, _ = detect_charuco(image, spec)
        self.assertGreaterEqual(len(ids), 20)
        plane = calibrate_plane_homography(image, spec)
        self.assertAlmostEqual(plane.mm_per_px, 1.0 / 8.0, places=5)
        self.assertLess(plane.rms_mm, 0.05)
        self.assertEqual(corners.shape[1], 2)

    def test_model_points_are_finite(self) -> None:
        spec = CharucoSpec()
        board, _ = spec.board()
        points = np.asarray(board.getChessboardCorners())
        self.assertTrue(np.all(np.isfinite(points)))
        self.assertEqual(points.shape[1], 3)

    def test_intrinsics_from_synthetic_views(self) -> None:
        spec = CharucoSpec()
        board_image = make_charuco_image(spec, pixels_per_mm=10.0)
        camera_matrix = np.array(
            [[900.0, 0.0, 640.0], [0.0, 900.0, 480.0], [0.0, 0.0, 1.0]]
        )
        board_width = spec.squares_x * spec.square_length_mm + 40.0
        board_height = spec.squares_y * spec.square_length_mm + 40.0
        object_corners = np.array(
            [
                [0.0, 0.0, 0.0],
                [board_width, 0.0, 0.0],
                [board_width, board_height, 0.0],
                [0.0, board_height, 0.0],
            ],
            dtype=np.float32,
        )
        destination_corners = np.array(
            [
                [0.0, 0.0],
                [board_image.shape[1] - 1.0, 0.0],
                [board_image.shape[1] - 1.0, board_image.shape[0] - 1.0],
                [0.0, board_image.shape[0] - 1.0],
            ],
            dtype=np.float32,
        )
        views = []
        for index in range(10):
            angle_x = np.deg2rad((index % 3 - 1) * 11.0)
            angle_y = np.deg2rad(((index // 3) % 3 - 1) * 14.0)
            angle_z = np.deg2rad((index - 4.5) * 3.0)
            rotation_x = cv2.Rodrigues(np.array([angle_x, 0.0, 0.0]))[0]
            rotation_y = cv2.Rodrigues(np.array([0.0, angle_y, 0.0]))[0]
            rotation_z = cv2.Rodrigues(np.array([0.0, 0.0, angle_z]))[0]
            rotation = rotation_z @ rotation_y @ rotation_x
            translation = np.array(
                [-board_width / 2.0, -board_height / 2.0, 330.0 + 4.0 * index]
            )
            camera_points = (rotation @ object_corners.T).T + translation
            projected = cv2.projectPoints(
                camera_points,
                np.zeros(3),
                np.zeros(3),
                camera_matrix,
                None,
            )[0].reshape(-1, 2)
            homography = cv2.getPerspectiveTransform(
                destination_corners,
                projected.astype(np.float32),
            )
            views.append(
                cv2.warpPerspective(
                    board_image,
                    homography,
                    (1280, 960),
                    borderValue=255,
                )
            )
        intrinsics = calibrate_intrinsics(views, spec, min_views=8)
        self.assertLess(intrinsics.rms, 0.6)
        self.assertAlmostEqual(
            intrinsics.camera_matrix[0, 0] / 900.0,
            1.0,
            delta=0.03,
        )

    def test_top_plane_height_compensation(self) -> None:
        camera_matrix = np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        )
        homography_at_zero = np.array(
            [[0.2, 0.0, -64.0], [0.0, 0.2, -48.0], [0.0, 0.0, 1.0]]
        )
        homography_at_20 = plane_homography_at_height(
            homography_at_zero,
            camera_matrix,
            20.0,
        )
        board_xy = np.array([[10.0, 5.0]], dtype=np.float64)
        homogeneous = np.column_stack((board_xy, np.ones(1)))
        elevated_pixels = homogeneous @ np.linalg.inv(homography_at_20).T
        elevated_pixels = elevated_pixels[:, :2] / elevated_pixels[:, 2:3]
        recovered = np.column_stack((elevated_pixels, np.ones(1))) @ homography_at_20.T
        recovered = recovered[:, :2] / recovered[:, 2:3]
        self.assertTrue(np.allclose(recovered, board_xy))


if __name__ == "__main__":
    unittest.main()
