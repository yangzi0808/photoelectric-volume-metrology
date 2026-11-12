from __future__ import annotations

import unittest

import cv2
import numpy as np

from volumecv.laser_scan import extract_laser_line


class LaserScanTests(unittest.TestCase):
    def test_vertical_laser_centroid(self) -> None:
        image = np.zeros((400, 640), dtype=np.uint8)
        for y in range(60, 340):
            center = 320 + 0.03 * (y - 200)
            image[y, int(center) - 1 : int(center) + 3] = np.array(
                [100, 255, 100, 20],
                dtype=np.uint8,
            )
        line = extract_laser_line(image)
        self.assertGreater(len(line.pixels), 250)
        expected = 320.0 + 0.03 * (line.pixels[:, 1] - 200.0)
        residual = line.pixels[:, 0] - np.floor(expected)
        self.assertLess(float(np.median(np.abs(residual))), 0.25)

    def test_horizontal_laser(self) -> None:
        image = np.zeros((400, 640), dtype=np.uint8)
        image[200:204, 50:600] = 210
        image = cv2.GaussianBlur(image, (0, 0), 0.8)
        from volumecv.laser_scan import LaserLineConfig

        line = extract_laser_line(
            image,
            LaserLineConfig(axis="horizontal", peak_threshold=8, min_contrast=15),
        )
        self.assertGreater(len(line.pixels), 450)


if __name__ == "__main__":
    unittest.main()
