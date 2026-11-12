from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .image_measure import transform_points
from .synthetic import (
    make_depth_profile,
    random_blind_slot_dimensions,
    random_cylinder_dimensions,
    random_l_dimensions,
)


@dataclass(frozen=True)
class RealisticImageCase:
    name: str
    shape: str
    top_gray: np.ndarray
    side_gray: np.ndarray
    top_homography_px_to_mm: np.ndarray
    side_homography_px_to_mm: np.ndarray
    depth_profile_mm: np.ndarray | None
    truth_dimensions: dict[str, float]
    render_metadata: dict[str, Any]


def _identity_grid(
    homography_px_to_mm: np.ndarray, width: int, height: int
) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width]
    pixels = np.column_stack((xs.ravel(), ys.ravel()))
    return transform_points(pixels, homography_px_to_mm).reshape(height, width, 2)


def _l_coverage(metric: np.ndarray, dimensions: dict[str, float]) -> np.ndarray:
    a = dimensions["a_mm"]
    b = dimensions["b_mm"]
    u = dimensions["u_mm"]
    v = dimensions["v_mm"]
    x = metric[:, :, 0]
    y = metric[:, :, 1]
    x0, x1 = -a / 2.0, a / 2.0
    y0, y1 = -b / 2.0, b / 2.0
    outer = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    notch = (x <= x0 + u) & (y >= y1 - v)
    return outer & ~notch


def _cylinder_coverage(metric: np.ndarray, dimensions: dict[str, float]) -> np.ndarray:
    radius2 = metric[:, :, 0] ** 2 + metric[:, :, 1] ** 2
    outer_radius = dimensions["D_mm"] / 2.0
    inner_radius = dimensions["d_mm"] / 2.0
    return (radius2 <= outer_radius**2) & (radius2 > inner_radius**2)


def _blind_slot_coverage(
    metric: np.ndarray, dimensions: dict[str, float]
) -> np.ndarray:
    a = dimensions["a_mm"]
    b = dimensions["b_mm"]
    u = dimensions["u_mm"]
    v = dimensions["v_mm"]
    x = metric[:, :, 0]
    y = metric[:, :, 1]
    outer = (x >= -a / 2.0) & (x <= a / 2.0) & (y >= -b / 2.0) & (y <= b / 2.0)
    slot = (x >= -u / 2.0) & (x <= u / 2.0) & (y >= -v / 2.0) & (y <= v / 2.0)
    return outer & ~slot


def _coverage(
    shape: str,
    metric: np.ndarray,
    dimensions: dict[str, float],
) -> np.ndarray:
    if shape == "l_block":
        return _l_coverage(metric, dimensions)
    if shape == "cylinder":
        return _cylinder_coverage(metric, dimensions)
    return _blind_slot_coverage(metric, dimensions)


def _render_bright_background(
    coverage: np.ndarray,
    rng: np.random.Generator,
    blur_sigma: float,
    noise_sigma: float,
    jpeg_quality: int,
) -> np.ndarray:
    height, width = coverage.shape
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx - width / 2.0) / max(width, 1)
    ny = (yy - height / 2.0) / max(height, 1)
    gradient = (
        1.0
        + rng.uniform(-0.045, 0.045) * nx
        + rng.uniform(-0.045, 0.045) * ny
        + rng.uniform(-0.018, 0.018) * nx * ny
    )
    vignette = 1.0 - rng.uniform(0.0, 0.035) * (nx**2 + ny**2)
    background = rng.uniform(218.0, 238.0)
    foreground = rng.uniform(45.0, 92.0)
    image = np.where(coverage, foreground, background) * gradient * vignette
    if blur_sigma > 0:
        image = cv2.GaussianBlur(image, (0, 0), blur_sigma)
    image += rng.normal(0.0, noise_sigma, image.shape)
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    if jpeg_quality < 100:
        ok, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
        )
        if ok:
            image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return image


def _make_homographies(
    width: int,
    height: int,
    rng: np.random.Generator,
    object_extent_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    fit_scale = (min(width, height) - 150.0) / max(object_extent_mm, 1.0)
    scale = min(float(rng.uniform(8.0, 10.8)), max(5.5, fit_scale))
    angle = np.deg2rad(float(rng.uniform(-1.2, 1.2)))
    anisotropy = float(rng.uniform(0.9985, 1.0015))
    cosine, sine = np.cos(angle), np.sin(angle)
    center_x = width / 2.0 + float(rng.uniform(-35.0, 35.0))
    center_y = height / 2.0 + float(rng.uniform(-35.0, 35.0))
    inverse = np.array(
        [
            [scale * cosine, -scale * anisotropy * sine, center_x],
            [scale * sine, scale * anisotropy * cosine, center_y],
            [
                float(rng.uniform(-8e-6, 8e-6)),
                float(rng.uniform(-8e-6, 8e-6)),
                1.0,
            ],
        ],
        dtype=np.float64,
    )
    return np.linalg.inv(inverse), inverse


def _render_case(
    name: str,
    shape: str,
    dimensions: dict[str, float],
    rng: np.random.Generator,
    width: int = 1280,
    height: int = 960,
    supersample: int = 2,
) -> RealisticImageCase:
    if shape == "cylinder":
        top_extent = dimensions["D_mm"]
    else:
        top_extent = float(np.hypot(dimensions["a_mm"], dimensions["b_mm"]))
    top_homography, _ = _make_homographies(
        width,
        height,
        rng,
        top_extent,
    )
    if supersample > 1:
        high_width = width * supersample
        high_height = height * supersample
        scale_pixels = np.array(
            [
                [1.0 / supersample, 0.0, 0.0],
                [0.0, 1.0 / supersample, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        high_homography = top_homography @ scale_pixels
        metric = _identity_grid(high_homography, high_width, high_height)
        coverage_high = _coverage(shape, metric, dimensions).astype(np.uint8)
        coverage = cv2.resize(
            coverage_high,
            (width, height),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float64)
    else:
        metric = _identity_grid(top_homography, width, height)
        coverage = _coverage(shape, metric, dimensions).astype(np.float64)

    blur_sigma = float(rng.uniform(0.55, 1.15))
    noise_sigma = float(rng.uniform(0.8, 3.2))
    jpeg_quality = int(rng.integers(78, 97))
    top_gray = _render_bright_background(
        coverage,
        rng,
        blur_sigma,
        noise_sigma,
        jpeg_quality,
    )

    side_height = dimensions["H_mm"] if shape == "cylinder" else dimensions["h_mm"]
    side_width = (
        dimensions["D_mm"]
        if shape == "cylinder"
        else max(dimensions["a_mm"], dimensions["b_mm"])
    )
    side_scale = min(
        float(rng.uniform(8.0, 11.0)),
        max(5.0, (height * 0.82) / max(side_height, 1.0)),
    )
    side_tx = width / 2.0
    side_baseline = height * 0.88
    side_h_mm_to_px = np.array(
        [
            [side_scale, 0.0, side_tx],
            [0.0, side_scale, side_baseline],
            [0.0, 0.0, 1.0],
        ]
    )
    side_homography = np.linalg.inv(side_h_mm_to_px)
    side_metric = _identity_grid(side_homography, width, height)
    side_x = side_metric[:, :, 0]
    side_y = side_metric[:, :, 1]
    side_coverage = (
        (side_x >= -side_width / 2.0)
        & (side_x <= side_width / 2.0)
        & (side_y <= 0.0)
        & (side_y >= -side_height)
    ).astype(np.float64)
    side_gray = _render_bright_background(
        side_coverage,
        rng,
        blur_sigma=float(rng.uniform(0.5, 1.0)),
        noise_sigma=float(rng.uniform(0.7, 2.5)),
        jpeg_quality=int(rng.integers(82, 98)),
    )

    depth = (
        make_depth_profile(dimensions["t_mm"], rng) if shape == "blind_slot" else None
    )
    metadata = {
        "render_width": width,
        "render_height": height,
        "supersample": supersample,
        "top_blur_sigma_px": blur_sigma,
        "top_sensor_noise_sigma": noise_sigma,
        "jpeg_quality": jpeg_quality,
    }
    return RealisticImageCase(
        name=name,
        shape=shape,
        top_gray=top_gray,
        side_gray=side_gray,
        top_homography_px_to_mm=top_homography,
        side_homography_px_to_mm=side_homography,
        depth_profile_mm=depth,
        truth_dimensions=dimensions,
        render_metadata=metadata,
    )


def make_realistic_case(
    shape: str,
    dimensions: dict[str, float],
    rng: np.random.Generator,
    name: str | None = None,
) -> RealisticImageCase:
    return _render_case(name or shape, shape, dimensions, rng)


def make_realistic_demo_cases(seed: int = 20261110) -> list[RealisticImageCase]:
    rng = np.random.default_rng(seed)
    specs: list[tuple[str, dict[str, float]]] = [
        (
            "Realistic L-block",
            {
                "a_mm": 82.0,
                "b_mm": 71.0,
                "h_mm": 58.0,
                "u_mm": 31.0,
                "v_mm": 26.0,
            },
        ),
        (
            "Realistic cylinder",
            {"D_mm": 84.0, "d_mm": 28.0, "H_mm": 72.0},
        ),
        (
            "Realistic blind slot",
            {
                "a_mm": 88.0,
                "b_mm": 76.0,
                "h_mm": 61.0,
                "u_mm": 46.0,
                "v_mm": 38.0,
                "t_mm": 19.0,
            },
        ),
    ]
    shapes = ["l_block", "cylinder", "blind_slot"]
    return [
        make_realistic_case(shape, dimensions, rng, name)
        for shape, (name, dimensions) in zip(shapes, specs)
    ]


def random_realistic_case(
    shape: str,
    rng: np.random.Generator,
    name: str | None = None,
) -> RealisticImageCase:
    if shape == "l_block":
        dimensions = random_l_dimensions(rng)
    elif shape == "cylinder":
        dimensions = random_cylinder_dimensions(rng)
    elif shape == "blind_slot":
        dimensions = random_blind_slot_dimensions(rng)
    else:
        raise ValueError(f"Unsupported shape: {shape}")
    return make_realistic_case(shape, dimensions, rng, name=name)


def make_parallax_case(
    shape: str,
    dimensions: dict[str, float],
    rng: np.random.Generator,
    width: int = 1280,
    height: int = 960,
    stage_depth_mm: float = 280.0,
    focal_length_px: float = 1200.0,
    supersample: int = 2,
) -> tuple[RealisticImageCase, np.ndarray]:
    """Render a target at its true top-face height under a pinhole camera.

    ``case.top_homography_px_to_mm`` is the calibration at the empty-stage
    plane z=0. The returned camera matrix allows the measurement pipeline to
    move that homography to the top-face height measured from the side view.
    """

    object_height = dimensions["H_mm"] if shape == "cylinder" else dimensions["h_mm"]
    camera_matrix = np.array(
        [
            [focal_length_px, 0.0, width / 2.0],
            [0.0, focal_length_px, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    stage_homography_mm_to_px = camera_matrix @ np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, stage_depth_mm],
        ]
    )
    top_homography_px_to_mm = np.linalg.inv(stage_homography_mm_to_px)
    top_face_homography_mm_to_px = camera_matrix @ np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, stage_depth_mm - object_height],
        ]
    )
    top_face_homography_px_to_mm = np.linalg.inv(top_face_homography_mm_to_px)
    if supersample > 1:
        downscale = np.array(
            [
                [1.0 / supersample, 0.0, 0.0],
                [0.0, 1.0 / supersample, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        metric = _identity_grid(
            top_face_homography_px_to_mm @ downscale,
            width * supersample,
            height * supersample,
        )
        coverage_high = _coverage(shape, metric, dimensions).astype(np.uint8)
        coverage = cv2.resize(
            coverage_high,
            (width, height),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float64)
    else:
        metric = _identity_grid(top_face_homography_px_to_mm, width, height)
        coverage = _coverage(shape, metric, dimensions).astype(np.float64)
    blur_sigma = float(rng.uniform(0.6, 1.1))
    top_gray = _render_bright_background(
        coverage,
        rng,
        blur_sigma,
        float(rng.uniform(0.8, 2.5)),
        int(rng.integers(82, 97)),
    )

    side_scale = 7.0
    side_width_mm = (
        dimensions["D_mm"]
        if shape == "cylinder"
        else max(dimensions["a_mm"], dimensions["b_mm"])
    )
    side_homography_mm_to_px = np.array(
        [
            [side_scale, 0.0, width / 2.0],
            [0.0, side_scale, height * 0.86],
            [0.0, 0.0, 1.0],
        ]
    )
    side_homography_px_to_mm = np.linalg.inv(side_homography_mm_to_px)
    side_metric = _identity_grid(
        side_homography_px_to_mm,
        width,
        height,
    )
    side_coverage = (
        (side_metric[:, :, 0] >= -side_width_mm / 2.0)
        & (side_metric[:, :, 0] <= side_width_mm / 2.0)
        & (side_metric[:, :, 1] <= 0.0)
        & (side_metric[:, :, 1] >= -object_height)
    ).astype(np.float64)
    side_gray = _render_bright_background(
        side_coverage,
        rng,
        blur_sigma=0.75,
        noise_sigma=1.2,
        jpeg_quality=92,
    )
    depth = (
        make_depth_profile(dimensions["t_mm"], rng) if shape == "blind_slot" else None
    )
    case = RealisticImageCase(
        name=f"Parallax {shape}",
        shape=shape,
        top_gray=top_gray,
        side_gray=side_gray,
        top_homography_px_to_mm=top_homography_px_to_mm,
        side_homography_px_to_mm=side_homography_px_to_mm,
        depth_profile_mm=depth,
        truth_dimensions=dimensions,
        render_metadata={
            "stage_depth_mm": stage_depth_mm,
            "focal_length_px": focal_length_px,
            "object_height_mm": object_height,
            "top_face_true_homography_px_to_mm": (
                top_face_homography_px_to_mm.tolist()
            ),
        },
    )
    return case, camera_matrix
