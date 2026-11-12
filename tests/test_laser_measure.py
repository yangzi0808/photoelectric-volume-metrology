from __future__ import annotations

import unittest

import numpy as np

from volumecv.calibration import CameraIntrinsics, plane_points_at_depth
from volumecv.laser_measure import (
    StagePose,
    calibrate_laser_plane_from_known_heights,
    pose_from_plane_homography,
    stage_plane_in_camera,
)


class LaserMeasureTests(unittest.TestCase):
    def test_plane_pose_and_depth_intersection(self) -> None:
        camera_matrix = np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        )
        homography_mm_to_px = np.array(
            [[10.0, 0.0, 320.0], [0.0, 10.0, 240.0], [0.0, 0.0, 1.0]]
        )
        pose = pose_from_plane_homography(
            homography_mm_to_px,
            camera_matrix,
        )
        self.assertTrue(np.allclose(pose.rotation_camera_from_stage, np.eye(3)))
        self.assertTrue(
            np.allclose(pose.translation_camera_from_stage, [0.0, 0.0, 50.0])
        )
        intrinsics = CameraIntrinsics(
            camera_matrix=camera_matrix,
            distortion=np.zeros(5),
            image_size=(640, 480),
            rms=0.0,
        )
        normal, offset = stage_plane_in_camera(pose, 0.0)
        points = plane_points_at_depth(
            np.array([[320.0, 240.0]], dtype=np.float64),
            intrinsics,
            normal,
            offset,
        )
        self.assertTrue(np.allclose(points, [[0.0, 0.0, 50.0]], atol=1e-6))

    def test_laser_plane_calibration_from_known_heights(self) -> None:
        camera_matrix = np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        )
        intrinsics = CameraIntrinsics(
            camera_matrix=camera_matrix,
            distortion=np.zeros(5),
            image_size=(640, 480),
            rms=0.0,
        )
        pose = StagePose(
            rotation_camera_from_stage=np.eye(3),
            translation_camera_from_stage=np.array([0.0, 0.0, 300.0]),
        )
        images = []
        for _ in range(3):
            image = np.zeros((480, 640), dtype=np.uint8)
            image[80:400, 570] = 220
            images.append(image)
        plane = calibrate_laser_plane_from_known_heights(
            images,
            [0.0, 20.0, 40.0],
            intrinsics,
            pose,
        )
        expected = np.array([1.0, 0.0, -0.5])
        expected /= np.linalg.norm(expected)
        normal = plane.normal if plane.normal[0] > 0 else -plane.normal
        self.assertTrue(np.allclose(normal, expected, atol=1e-3))
        self.assertLess(plane.rms_mm, 0.01)


if __name__ == "__main__":
    unittest.main()
