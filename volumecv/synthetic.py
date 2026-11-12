from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .features import Corner, as_binary_mask


@dataclass(frozen=True)
class SyntheticCase:
    name: str
    shape: str
    top_mask: np.ndarray
    side_mask: np.ndarray
    depth_profile_mm: np.ndarray | None
    truth_dimensions: dict[str, float]
    px_per_mm_x: float
    px_per_mm_y: float
    side_px_per_mm: float


def _shape_grid(width_px: int, height_px: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:height_px, 0:width_px]
    return xx, yy


def make_l_mask(
    a_mm: float,
    b_mm: float,
    u_mm: float,
    v_mm: float,
    px_per_mm_x: float,
    px_per_mm_y: float,
    margin_px: int = 30,
    notch_corner: Corner = "bottom_left",
) -> np.ndarray:
    width = round(a_mm * px_per_mm_x)
    height = round(b_mm * px_per_mm_y)
    mask = np.ones((height, width), dtype=bool)
    u_px = round(u_mm * px_per_mm_x)
    v_px = round(v_mm * px_per_mm_y)
    if notch_corner == "bottom_left":
        mask[-v_px:, :u_px] = False
    elif notch_corner == "bottom_right":
        mask[-v_px:, -u_px:] = False
    elif notch_corner == "top_left":
        mask[:v_px, :u_px] = False
    elif notch_corner == "top_right":
        mask[:v_px, -u_px:] = False
    else:
        raise ValueError(f"Unsupported corner: {notch_corner}")
    return np.pad(mask, margin_px, constant_values=False)


def make_cylinder_mask(
    outer_mm: float,
    inner_mm: float,
    px_per_mm: float,
    margin_px: int = 30,
) -> np.ndarray:
    outer_px = round(outer_mm * px_per_mm)
    inner_px = round(inner_mm * px_per_mm)
    xx, yy = _shape_grid(outer_px, outer_px)
    x = xx - (outer_px - 1) / 2.0
    y = yy - (outer_px - 1) / 2.0
    radius = outer_px / 2.0
    radius2 = x**2 + y**2
    mask = radius2 <= radius**2
    mask &= radius2 > (inner_px / 2.0) ** 2
    return np.pad(mask, margin_px, constant_values=False)


def make_blind_slot_mask(
    a_mm: float,
    b_mm: float,
    u_mm: float,
    v_mm: float,
    px_per_mm_x: float,
    px_per_mm_y: float,
    margin_px: int = 30,
    offset_px: tuple[int, int] = (0, 0),
) -> np.ndarray:
    width = round(a_mm * px_per_mm_x)
    height = round(b_mm * px_per_mm_y)
    slot_width = round(u_mm * px_per_mm_x)
    slot_height = round(v_mm * px_per_mm_y)
    x0 = (width - slot_width) // 2 + offset_px[0]
    y0 = (height - slot_height) // 2 + offset_px[1]
    x0 = int(np.clip(x0, 1, width - slot_width - 1))
    y0 = int(np.clip(y0, 1, height - slot_height - 1))
    mask = np.ones((height, width), dtype=bool)
    mask[y0 : y0 + slot_height, x0 : x0 + slot_width] = False
    return np.pad(mask, margin_px, constant_values=False)


def make_side_mask(height_mm: float, px_per_mm: float) -> np.ndarray:
    height_px = max(8, round(height_mm * px_per_mm))
    width_px = max(80, round(height_mm * 1.5 * px_per_mm))
    return np.ones((height_px, width_px), dtype=bool)


def make_depth_profile(
    slot_depth_mm: float,
    rng: np.random.Generator,
    top_points: int = 90,
    floor_points: int = 55,
    noise_mm: float = 0.055,
    outlier_fraction: float = 0.04,
) -> np.ndarray:
    top = rng.normal(0.0, noise_mm, top_points)
    floor = rng.normal(-slot_depth_mm, noise_mm, floor_points)
    values = np.r_[top, floor]
    outlier_count = round(values.size * outlier_fraction)
    if outlier_count:
        indices = rng.choice(values.size, outlier_count, replace=False)
        values[indices] += rng.normal(0.0, 5.0, outlier_count)
    return values


def _perturb_mask(mask: np.ndarray, operation: int) -> np.ndarray:
    binary = as_binary_mask(mask)
    if operation == 0:
        return binary
    padded = np.pad(binary, 1, constant_values=False)
    neighbors = np.stack(
        [
            padded[dy : dy + binary.shape[0], dx : dx + binary.shape[1]]
            for dy in range(3)
            for dx in range(3)
        ]
    )
    if operation > 0:
        return np.any(neighbors, axis=0)
    return np.all(neighbors, axis=0)


def random_l_dimensions(rng: np.random.Generator) -> dict[str, float]:
    while True:
        a = float(rng.uniform(60, 100))
        b = float(rng.uniform(60, 100))
        u = float(rng.uniform(20, min(40, a - 20)))
        v = float(rng.uniform(20, min(40, b - 20)))
        h = float(rng.uniform(40, 80))
        if a - u >= 20 and b - v >= 20:
            return {"a_mm": a, "b_mm": b, "h_mm": h, "u_mm": u, "v_mm": v}


def random_cylinder_dimensions(rng: np.random.Generator) -> dict[str, float]:
    while True:
        outer = float(rng.uniform(60, 100))
        inner = float(rng.uniform(20, 40))
        height = float(rng.uniform(50, 100))
        if (outer - inner) / 2.0 >= 15:
            return {"D_mm": outer, "d_mm": inner, "H_mm": height}


def random_blind_slot_dimensions(rng: np.random.Generator) -> dict[str, float]:
    while True:
        a = float(rng.uniform(60, 100))
        b = float(rng.uniform(60, 100))
        u = float(rng.uniform(30, min(60, a - 20)))
        v = float(rng.uniform(30, min(60, b - 20)))
        h = float(rng.uniform(40, 80))
        t = float(rng.uniform(10, min(30, h - 15, 0.5 * min(u, v))))
        if a - u >= 20 and b - v >= 20 and t >= 10:
            return {
                "a_mm": a,
                "b_mm": b,
                "h_mm": h,
                "u_mm": u,
                "v_mm": v,
                "t_mm": t,
            }


def make_case(
    shape: str,
    dimensions: dict[str, float],
    rng: np.random.Generator,
    px_per_mm_x: float = 10.0,
    px_per_mm_y: float = 10.0,
    side_px_per_mm: float = 10.0,
    mask_operation: int = 0,
    name: str | None = None,
) -> SyntheticCase:
    if shape == "l_block":
        top = make_l_mask(
            dimensions["a_mm"],
            dimensions["b_mm"],
            dimensions["u_mm"],
            dimensions["v_mm"],
            px_per_mm_x,
            px_per_mm_y,
        )
        depth = None
    elif shape == "cylinder":
        if abs(px_per_mm_x - px_per_mm_y) > 1e-9:
            raise ValueError("Cylinder generation requires equal x/y pixel scales.")
        top = make_cylinder_mask(dimensions["D_mm"], dimensions["d_mm"], px_per_mm_x)
        depth = None
    elif shape == "blind_slot":
        top = make_blind_slot_mask(
            dimensions["a_mm"],
            dimensions["b_mm"],
            dimensions["u_mm"],
            dimensions["v_mm"],
            px_per_mm_x,
            px_per_mm_y,
        )
        depth = make_depth_profile(dimensions["t_mm"], rng)
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    height_key = "H_mm" if shape == "cylinder" else "h_mm"
    side = make_side_mask(dimensions[height_key], side_px_per_mm)
    return SyntheticCase(
        name=name or shape,
        shape=shape,
        top_mask=_perturb_mask(top, mask_operation),
        side_mask=_perturb_mask(side, mask_operation),
        depth_profile_mm=depth,
        truth_dimensions=dimensions,
        px_per_mm_x=px_per_mm_x,
        px_per_mm_y=px_per_mm_y,
        side_px_per_mm=side_px_per_mm,
    )


def make_demo_cases(seed: int = 20261110) -> list[SyntheticCase]:
    rng = np.random.default_rng(seed)
    specs: list[tuple[str, dict[str, float]]] = [
        (
            "L-block 01",
            {
                "a_mm": 82.0,
                "b_mm": 71.0,
                "h_mm": 58.0,
                "u_mm": 31.0,
                "v_mm": 26.0,
            },
        ),
        (
            "Cylinder 01",
            {"D_mm": 84.0, "d_mm": 28.0, "H_mm": 72.0},
        ),
        (
            "Blind slot 01",
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
    cases: list[SyntheticCase] = []
    for index, (name, dimensions) in enumerate(specs):
        shape = "l_block" if index == 0 else "cylinder" if index == 1 else "blind_slot"
        cases.append(
            make_case(
                shape,
                dimensions,
                rng,
                px_per_mm_x=10.0,
                px_per_mm_y=10.0,
                side_px_per_mm=10.0,
                mask_operation=0,
                name=name,
            )
        )
    return cases


def case_as_dict(case: SyntheticCase) -> dict[str, Any]:
    return {
        "name": case.name,
        "shape": case.shape,
        "truth_dimensions": case.truth_dimensions,
        "px_per_mm_x": case.px_per_mm_x,
        "px_per_mm_y": case.px_per_mm_y,
        "side_px_per_mm": case.side_px_per_mm,
    }
