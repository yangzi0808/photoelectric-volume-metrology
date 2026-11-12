from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from typing import Any, Literal

ShapeName = Literal["l_block", "cylinder", "blind_slot"]


@dataclass(frozen=True)
class MeasurementIssue:
    code: str
    severity: Literal["error", "warning", "info"]
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass(frozen=True)
class LBlockDimensions:
    a: float
    b: float
    h: float
    u: float
    v: float

    @property
    def volume_mm3(self) -> float:
        return (self.a * self.b - self.u * self.v) * self.h

    def as_dict(self) -> dict[str, float]:
        return {
            "a_mm": self.a,
            "b_mm": self.b,
            "h_mm": self.h,
            "u_mm": self.u,
            "v_mm": self.v,
        }


@dataclass(frozen=True)
class CylinderDimensions:
    outer_diameter: float
    inner_diameter: float
    height: float

    @property
    def volume_mm3(self) -> float:
        return (
            pi / 4.0 * (self.outer_diameter**2 - self.inner_diameter**2) * self.height
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "D_mm": self.outer_diameter,
            "d_mm": self.inner_diameter,
            "H_mm": self.height,
        }


@dataclass(frozen=True)
class BlindSlotDimensions:
    a: float
    b: float
    h: float
    u: float
    v: float
    depth: float

    @property
    def volume_mm3(self) -> float:
        return self.a * self.b * self.h - self.u * self.v * self.depth

    def as_dict(self) -> dict[str, float]:
        return {
            "a_mm": self.a,
            "b_mm": self.b,
            "h_mm": self.h,
            "u_mm": self.u,
            "v_mm": self.v,
            "t_mm": self.depth,
        }


Dimensions = LBlockDimensions | CylinderDimensions | BlindSlotDimensions


@dataclass(frozen=True)
class VolumeResult:
    shape: str
    dimensions: dict[str, float]
    volume_mm3: float
    volume_cm3: float
    sigma_mm3: float
    relative_sigma: float
    confidence: float
    issues: tuple[MeasurementIssue, ...] = field(default_factory=tuple)
    elapsed_ms: float = 0.0
    mode: str = "explicit"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "mode": self.mode,
            "dimensions": {
                key: round(value, 6) for key, value in self.dimensions.items()
            },
            "volume_mm3": round(self.volume_mm3, 6),
            "volume_cm3": round(self.volume_cm3, 6),
            "sigma_mm3": round(self.sigma_mm3, 6),
            "relative_sigma": round(self.relative_sigma, 8),
            "confidence": round(self.confidence, 6),
            "issues": [issue.to_dict() for issue in self.issues],
            "elapsed_ms": round(self.elapsed_ms, 6),
            "diagnostics": self.diagnostics,
        }


def volume_mm3(shape: ShapeName, dimensions: dict[str, float]) -> float:
    if shape == "l_block":
        return LBlockDimensions(
            dimensions["a_mm"],
            dimensions["b_mm"],
            dimensions["h_mm"],
            dimensions["u_mm"],
            dimensions["v_mm"],
        ).volume_mm3
    if shape == "cylinder":
        return CylinderDimensions(
            dimensions["D_mm"],
            dimensions["d_mm"],
            dimensions["H_mm"],
        ).volume_mm3
    if shape == "blind_slot":
        return BlindSlotDimensions(
            dimensions["a_mm"],
            dimensions["b_mm"],
            dimensions["h_mm"],
            dimensions["u_mm"],
            dimensions["v_mm"],
            dimensions["t_mm"],
        ).volume_mm3
    raise ValueError(f"Unsupported shape: {shape}")


def _inside_range(
    issues: list[MeasurementIssue],
    name: str,
    value: float,
    low: float,
    high: float,
) -> None:
    if not (low <= value <= high):
        issues.append(
            MeasurementIssue(
                "OUT_OF_RANGE",
                "warning",
                f"{name}={value:.3f} mm is outside the expected "
                f"[{low:.0f}, {high:.0f}] mm range.",
            )
        )


def validate_dimensions(
    shape: ShapeName, dimensions: dict[str, float]
) -> list[MeasurementIssue]:
    issues: list[MeasurementIssue] = []
    for name, value in dimensions.items():
        if not value > 0.0:
            issues.append(
                MeasurementIssue(
                    "NON_POSITIVE_DIMENSION",
                    "error",
                    f"{name} must be positive, got {value:.3f} mm.",
                )
            )

    if shape == "l_block":
        a = dimensions["a_mm"]
        b = dimensions["b_mm"]
        h = dimensions["h_mm"]
        u = dimensions["u_mm"]
        v = dimensions["v_mm"]
        _inside_range(issues, "a", a, 60, 100)
        _inside_range(issues, "b", b, 60, 100)
        _inside_range(issues, "h", h, 40, 80)
        _inside_range(issues, "u", u, 20, 40)
        _inside_range(issues, "v", v, 20, 40)
        if u >= a - 20:
            issues.append(
                MeasurementIssue(
                    "L_ARM_TOO_NARROW",
                    "error",
                    "The measured L-block leaves an arm narrower than 20 mm.",
                )
            )
        if v >= b - 20:
            issues.append(
                MeasurementIssue(
                    "L_ARM_TOO_NARROW",
                    "error",
                    "The measured L-block leaves an arm narrower than 20 mm.",
                )
            )

    elif shape == "cylinder":
        outer = dimensions["D_mm"]
        inner = dimensions["d_mm"]
        height = dimensions["H_mm"]
        _inside_range(issues, "D", outer, 60, 100)
        _inside_range(issues, "d", inner, 20, 40)
        _inside_range(issues, "H", height, 50, 100)
        if inner >= outer:
            issues.append(
                MeasurementIssue(
                    "INVALID_HOLE",
                    "error",
                    "The measured inner diameter is not smaller than the outer diameter.",
                )
            )
        elif (outer - inner) / 2.0 < 15.0:
            issues.append(
                MeasurementIssue(
                    "CYLINDER_WALL_TOO_THIN",
                    "warning",
                    "The measured cylindrical wall is below the expected 15 mm.",
                )
            )

    elif shape == "blind_slot":
        a = dimensions["a_mm"]
        b = dimensions["b_mm"]
        h = dimensions["h_mm"]
        u = dimensions["u_mm"]
        v = dimensions["v_mm"]
        t = dimensions["t_mm"]
        _inside_range(issues, "a", a, 60, 100)
        _inside_range(issues, "b", b, 60, 100)
        _inside_range(issues, "h", h, 40, 80)
        _inside_range(issues, "u", u, 30, 60)
        _inside_range(issues, "v", v, 30, 60)
        _inside_range(issues, "t", t, 10, 30)
        if a - u < 20 or b - v < 20:
            issues.append(
                MeasurementIssue(
                    "SLOT_WALL_TOO_THIN",
                    "warning",
                    "At least one measured slot wall is below 10 mm.",
                )
            )
        if h - t < 15:
            issues.append(
                MeasurementIssue(
                    "SLOT_FLOOR_TOO_THIN",
                    "warning",
                    "The measured material below the slot is below 15 mm.",
                )
            )
        if t > min(u, v) / 2.0:
            issues.append(
                MeasurementIssue(
                    "SLOT_DEPTH_GEOMETRY",
                    "warning",
                    "The measured slot depth exceeds half of the short opening side.",
                )
            )
    else:
        raise ValueError(f"Unsupported shape: {shape}")

    return issues


def dimensions_object(shape: ShapeName, values: dict[str, float]) -> Dimensions:
    if shape == "l_block":
        return LBlockDimensions(
            values["a_mm"],
            values["b_mm"],
            values["h_mm"],
            values["u_mm"],
            values["v_mm"],
        )
    if shape == "cylinder":
        return CylinderDimensions(values["D_mm"], values["d_mm"], values["H_mm"])
    if shape == "blind_slot":
        return BlindSlotDimensions(
            values["a_mm"],
            values["b_mm"],
            values["h_mm"],
            values["u_mm"],
            values["v_mm"],
            values["t_mm"],
        )
    raise ValueError(f"Unsupported shape: {shape}")
