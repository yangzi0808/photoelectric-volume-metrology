from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthEstimate:
    depth_mm: float
    sigma_mm: float
    top_level_mm: float
    floor_level_mm: float
    inlier_count: int
    rejected_count: int


def estimate_two_level_depth(samples_mm: np.ndarray | list[float]) -> DepthEstimate:
    """Estimate a step height from line-laser samples.

    The profile is expected to contain top-face and slot-floor samples. A
    two-cluster robust k-means is used because the competition shapes contain
    exactly one planar floor level. Points farther than 3 MAD from either
    cluster centre are rejected.
    """

    values = np.asarray(samples_mm, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size < 8:
        raise ValueError("At least 8 finite laser-depth samples are required.")

    low_guard, high_guard = np.quantile(values, [0.25, 0.75])
    centers = np.array([low_guard, high_guard], dtype=np.float64)
    if abs(centers[1] - centers[0]) < 0.25:
        raise ValueError("The depth profile does not contain two separable levels.")

    for _ in range(30):
        labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        updated = centers.copy()
        for index in range(2):
            cluster = values[labels == index]
            if cluster.size:
                updated[index] = np.median(cluster)
        if np.allclose(updated, centers, atol=1e-4):
            centers = updated
            break
        centers = updated

    labels = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
    keep = np.zeros(values.size, dtype=bool)
    for index in range(2):
        cluster = values[labels == index]
        if cluster.size < 3:
            continue
        center = np.median(cluster)
        mad = 1.4826 * np.median(np.abs(cluster - center))
        tolerance = max(0.04, 3.5 * mad)
        keep |= (labels == index) & (np.abs(values - center) <= tolerance)

    if np.count_nonzero(keep) < 8:
        raise ValueError("Too few samples remain after laser-depth outlier rejection.")

    labels = labels[keep]
    retained = values[keep]
    top_cluster = retained[labels == np.argmax(centers)]
    floor_cluster = retained[labels == np.argmin(centers)]
    if top_cluster.size < 3 or floor_cluster.size < 3:
        raise ValueError("One laser-depth level has too few inliers.")

    top = float(np.median(top_cluster))
    floor = float(np.median(floor_cluster))
    depth = abs(top - floor)
    residual = np.r_[top_cluster - top, floor_cluster - floor]
    robust_sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
    level_sigma = (
        1.2533 * robust_sigma / np.sqrt(min(top_cluster.size, floor_cluster.size))
    )
    return DepthEstimate(
        depth_mm=depth,
        sigma_mm=max(1e-4, float(np.hypot(level_sigma, level_sigma))),
        top_level_mm=top,
        floor_level_mm=floor,
        inlier_count=int(retained.size),
        rejected_count=int(values.size - retained.size),
    )
