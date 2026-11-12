from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Corner = Literal["top_left", "top_right", "bottom_left", "bottom_right"]


@dataclass(frozen=True)
class BBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class CornerNotch:
    corner: Corner
    width_px: float
    height_px: float
    pixel_area: int


@dataclass(frozen=True)
class InteriorHole:
    bbox: BBox
    pixel_area: int
    fill_ratio: float


@dataclass(frozen=True)
class TopFeatures:
    bbox: BBox
    object_pixel_area: int
    outer_fill_ratio: float
    aspect_ratio: float
    notch: CornerNotch | None
    hole: InteriorHole | None


def as_binary_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {array.shape}.")
    return array > 0


def tight_bbox(mask: np.ndarray) -> BBox:
    binary = as_binary_mask(mask)
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        raise ValueError("The mask is empty.")
    return BBox(
        x0=int(xs.min()),
        y0=int(ys.min()),
        x1=int(xs.max()),
        y1=int(ys.max()),
    )


def _longest_interior_runs(row_background: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, row_background, False]
    transitions = np.flatnonzero(
        padded[1:].astype(np.int8) != padded[:-1].astype(np.int8)
    )
    runs: list[tuple[int, int]] = []
    for start, end in zip(transitions[0::2], transitions[1::2]):
        # Runs touching a bbox edge are the exterior, not an enclosed hole.
        if start > 0 and end < row_background.size:
            runs.append((int(start), int(end)))
    return runs


def find_interior_hole(
    mask: np.ndarray, bbox: BBox | None = None
) -> InteriorHole | None:
    """Find the dominant fully enclosed background region.

    The extraction is optimized for the competition's known shapes: one
    through-hole or one blind-slot opening. It deliberately avoids a heavy
    contour library and is fast enough for an embedded/contest controller.
    """

    binary = as_binary_mask(mask)
    box = bbox or tight_bbox(binary)
    crop = binary[box.y0 : box.y1 + 1, box.x0 : box.x1 + 1]
    width, height = box.width, box.height
    min_run = max(2, round(min(width, height) * 0.02))

    records: list[tuple[int, int, int]] = []
    for y in range(height):
        runs = [
            run
            for run in _longest_interior_runs(~crop[y])
            if run[1] - run[0] >= min_run
        ]
        if not runs:
            continue
        start, end = max(runs, key=lambda run: run[1] - run[0])
        records.append((y, start, end))

    if not records:
        return None

    # Group consecutive scan lines whose interior intervals overlap.
    groups: list[list[tuple[int, int, int]]] = []
    current: list[tuple[int, int, int]] = []
    previous_y = -2
    previous_run = (0, 0)
    for record in records:
        y, start, end = record
        overlap = max(0, min(previous_run[1], end) - max(previous_run[0], start))
        if y != previous_y + 1 or overlap == 0:
            if current:
                groups.append(current)
            current = [record]
        else:
            current.append(record)
        previous_y = y
        previous_run = (start, end)
    if current:
        groups.append(current)

    best = max(groups, key=lambda group: sum(run[2] - run[1] for run in group))
    ys = [record[0] for record in best]
    starts = [record[1] for record in best]
    ends = [record[2] for record in best]
    local = BBox(
        x0=min(starts),
        y0=min(ys),
        x1=max(ends) - 1,
        y1=max(ys),
    )
    absolute = BBox(
        x0=local.x0 + box.x0,
        y0=local.y0 + box.y0,
        x1=local.x1 + box.x0,
        y1=local.y1 + box.y0,
    )
    area = sum(run[2] - run[1] for run in best)
    return InteriorHole(
        bbox=absolute,
        pixel_area=area,
        fill_ratio=area / local.area,
    )


def _edge_run(values: np.ndarray, from_start: bool) -> int:
    count = 0
    iterator = values if from_start else values[::-1]
    for value in iterator:
        if value:
            break
        count += 1
    return count


def find_corner_notch(mask: np.ndarray, bbox: BBox | None = None) -> CornerNotch | None:
    binary = as_binary_mask(mask)
    box = bbox or tight_bbox(binary)
    crop = binary[box.y0 : box.y1 + 1, box.x0 : box.x1 + 1]
    height, width = crop.shape
    margin_y = max(3, round(height * 0.08))
    margin_x = max(3, round(width * 0.08))

    corners: dict[Corner, np.ndarray] = {
        "top_left": crop[:margin_y, :margin_x],
        "top_right": crop[:margin_y, -margin_x:],
        "bottom_left": crop[-margin_y:, :margin_x],
        "bottom_right": crop[-margin_y:, -margin_x:],
    }
    occupancy = {
        name: float(window.mean()) if window.size else 0.0
        for name, window in corners.items()
    }
    corner = min(occupancy, key=occupancy.get)
    if occupancy[corner] > 0.85:
        return None

    if corner in {"top_left", "top_right"}:
        y = 0
    else:
        y = height - 1
    if corner in {"top_left", "bottom_left"}:
        x = 0
    else:
        x = width - 1

    horizontal = _edge_run(
        crop[y, :],
        from_start=corner in {"top_left", "bottom_left"},
    )
    vertical = _edge_run(
        crop[:, x],
        from_start=corner in {"top_left", "top_right"},
    )
    if horizontal == 0 or vertical == 0:
        return None

    # The notch is a filled background rectangle in this competition.
    pixel_area = int(
        np.count_nonzero(
            ~crop[
                :vertical,
                :horizontal,
            ]
            if corner == "top_left"
            else ~crop[
                :vertical,
                -horizontal:,
            ]
            if corner == "top_right"
            else ~crop[
                -vertical:,
                :horizontal,
            ]
            if corner == "bottom_left"
            else ~crop[
                -vertical:,
                -horizontal:,
            ]
        )
    )
    return CornerNotch(
        corner=corner,
        width_px=float(horizontal),
        height_px=float(vertical),
        pixel_area=pixel_area,
    )


def extract_top_features(mask: np.ndarray) -> TopFeatures:
    binary = as_binary_mask(mask)
    bbox = tight_bbox(binary)
    crop = binary[bbox.y0 : bbox.y1 + 1, bbox.x0 : bbox.x1 + 1]
    object_pixels = int(np.count_nonzero(crop))
    hole = find_interior_hole(binary, bbox)
    notch = None if hole is not None else find_corner_notch(binary, bbox)
    outer_pixels = object_pixels + (hole.pixel_area if hole else 0)
    return TopFeatures(
        bbox=bbox,
        object_pixel_area=object_pixels,
        outer_fill_ratio=outer_pixels / bbox.area,
        aspect_ratio=bbox.width / bbox.height,
        notch=notch,
        hole=hole,
    )


def classify_top_features(features: TopFeatures) -> tuple[str, float]:
    if features.hole is None and features.notch is not None:
        return "l_block", 0.99

    if features.hole is not None:
        round_outer = (
            abs(np.log(max(features.aspect_ratio, 1e-9))) < 0.11
            and features.outer_fill_ratio < 0.88
        )
        if round_outer:
            return "cylinder", 0.96
        return "blind_slot", 0.94

    return "l_block", 0.55


def measure_side_height_px(side_mask: np.ndarray) -> float:
    return float(tight_bbox(side_mask).height)
