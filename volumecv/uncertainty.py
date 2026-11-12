from __future__ import annotations

from math import pi, sqrt

from .geometry import ShapeName


def volume_sigma_mm3(
    shape: ShapeName,
    dimensions: dict[str, float],
    sigmas: dict[str, float],
) -> float:
    """First-order independent-error propagation."""

    if shape == "l_block":
        a = dimensions["a_mm"]
        b = dimensions["b_mm"]
        h = dimensions["h_mm"]
        u = dimensions["u_mm"]
        v = dimensions["v_mm"]
        terms = [
            (b * h, sigmas["a_mm"]),
            (a * h, sigmas["b_mm"]),
            (a * b - u * v, sigmas["h_mm"]),
            (-v * h, sigmas["u_mm"]),
            (-u * h, sigmas["v_mm"]),
        ]

    elif shape == "cylinder":
        outer = dimensions["D_mm"]
        inner = dimensions["d_mm"]
        height = dimensions["H_mm"]
        terms = [
            (pi * outer * height / 2.0, sigmas["D_mm"]),
            (-pi * inner * height / 2.0, sigmas["d_mm"]),
            (
                pi * (outer**2 - inner**2) / 4.0,
                sigmas["H_mm"],
            ),
        ]

    elif shape == "blind_slot":
        a = dimensions["a_mm"]
        b = dimensions["b_mm"]
        h = dimensions["h_mm"]
        u = dimensions["u_mm"]
        v = dimensions["v_mm"]
        depth = dimensions["t_mm"]
        terms = [
            (b * h, sigmas["a_mm"]),
            (a * h, sigmas["b_mm"]),
            (a * b, sigmas["h_mm"]),
            (-v * depth, sigmas["u_mm"]),
            (-u * depth, sigmas["v_mm"]),
            (-u * v, sigmas["t_mm"]),
        ]

    else:
        raise ValueError(f"Unsupported shape: {shape}")

    return sqrt(sum((partial * sigma) ** 2 for partial, sigma in terms))
