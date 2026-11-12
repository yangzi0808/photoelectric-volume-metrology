"""Known-shape photoelectric volume measurement prototype."""

from .geometry import (
    BlindSlotDimensions,
    CylinderDimensions,
    LBlockDimensions,
    MeasurementIssue,
    VolumeResult,
)
from .pipeline import measure_case

__all__ = [
    "BlindSlotDimensions",
    "CylinderDimensions",
    "LBlockDimensions",
    "MeasurementIssue",
    "VolumeResult",
    "measure_case",
]

__version__ = "0.1.0"
