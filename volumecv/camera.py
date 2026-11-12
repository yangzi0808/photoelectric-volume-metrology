from __future__ import annotations

from dataclasses import dataclass
from time import monotonic, sleep
from typing import Any, Self

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 960
    fps: float = 30.0
    fourcc: str = "MJPG"
    exposure: float | None = None
    gain: float | None = None
    auto_white_balance: bool = False
    auto_focus: bool = False
    focus: float | None = None
    warmup_frames: int = 20
    backend: int = cv2.CAP_DSHOW


class OpenCVCamera:
    """Small deterministic wrapper around OpenCV camera capture."""

    def __init__(self, config: CameraConfig):
        self.config = config
        self.capture: cv2.VideoCapture | None = None
        self.warnings: list[str] = []

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def open(self) -> None:
        if self.capture is not None and self.capture.isOpened():
            return
        capture = cv2.VideoCapture(self.config.index, self.config.backend)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open camera index {self.config.index}.")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.config.fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.config.exposure is not None:
            # DirectShow uses 0.25 for manual mode; other backends may ignore it.
            if not capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25):
                self.warnings.append("Manual exposure mode could not be selected.")
            if not capture.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure):
                self.warnings.append("Exposure value could not be set.")
        if self.config.gain is not None and not capture.set(
            cv2.CAP_PROP_GAIN, self.config.gain
        ):
            self.warnings.append("Gain value could not be set.")
        if not self.config.auto_white_balance and not capture.set(
            cv2.CAP_PROP_AUTO_WB, 0
        ):
            self.warnings.append("Automatic white balance could not be disabled.")
        if not self.config.auto_focus and not capture.set(cv2.CAP_PROP_AUTOFOCUS, 0):
            self.warnings.append("Automatic focus could not be disabled.")
        if self.config.focus is not None and not capture.set(
            cv2.CAP_PROP_FOCUS, self.config.focus
        ):
            self.warnings.append("Focus value could not be set.")
        self.capture = capture
        try:
            for _ in range(max(0, self.config.warmup_frames)):
                capture.grab()
            first = self.read()
        except RuntimeError:
            capture.release()
            self.capture = None
            raise
        if first.shape[1] != self.config.width or first.shape[0] != self.config.height:
            self.warnings.append(
                "Camera returned "
                f"{first.shape[1]}x{first.shape[0]} instead of the requested "
                f"{self.config.width}x{self.config.height}."
            )

    def read(self, retries: int = 5) -> np.ndarray:
        if self.capture is None:
            raise RuntimeError("Camera is not open.")
        for _ in range(max(1, retries)):
            ok, frame = self.capture.read()
            if ok and frame is not None:
                return frame
            sleep(0.01)
        raise RuntimeError("Camera frame acquisition failed.")

    def capture_average(
        self,
        frame_count: int = 8,
        discard_frames: int = 2,
    ) -> np.ndarray:
        if frame_count <= 0:
            raise ValueError("frame_count must be positive.")
        for _ in range(max(0, discard_frames)):
            self.read()
        accumulator = None
        for _ in range(frame_count):
            frame = self.read().astype(np.float64)
            accumulator = frame if accumulator is None else accumulator + frame
        assert accumulator is not None
        return np.clip(accumulator / frame_count, 0, 255).astype(np.uint8)

    def properties(self) -> dict[str, Any]:
        if self.capture is None:
            return {}
        return {
            "width": int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(self.capture.get(cv2.CAP_PROP_FPS)),
            "exposure": float(self.capture.get(cv2.CAP_PROP_EXPOSURE)),
            "gain": float(self.capture.get(cv2.CAP_PROP_GAIN)),
            "focus": float(self.capture.get(cv2.CAP_PROP_FOCUS)),
            "autofocus": float(self.capture.get(cv2.CAP_PROP_AUTOFOCUS)),
            "warnings": list(self.warnings),
        }

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None


def list_available_cameras(max_index: int = 8) -> list[int]:
    available: list[int] = []
    for index in range(max_index + 1):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if capture.isOpened():
            available.append(index)
        capture.release()
    return available


@dataclass(frozen=True)
class TimedFrame:
    image: np.ndarray
    timestamp_s: float
    sequence: int


def capture_timed(capture: OpenCVCamera, sequence: int) -> TimedFrame:
    return TimedFrame(
        image=capture.read(),
        timestamp_s=monotonic(),
        sequence=sequence,
    )
