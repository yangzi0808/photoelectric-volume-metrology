from __future__ import annotations

import json
import threading
from pathlib import Path
from tkinter import BOTH, LEFT, StringVar, Tk, filedialog, messagebox, ttk

import cv2

from .calibration import CalibrationBundle
from .camera import CameraConfig, OpenCVCamera
from .geometry import VolumeResult
from .image_pipeline import measure_calibrated_images
from .pipeline import MeasurementFailure
from .session import CompetitionSession


class MeasurementApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("Photoelectric Volume Metrology")
        self.root.geometry("920x650")
        self.root.minsize(820, 600)
        self.last_result: VolumeResult | None = None
        self.last_images: tuple[cv2.typing.MatLike, cv2.typing.MatLike] | None = None
        self.session = CompetitionSession()
        self.calibration_path = StringVar(value="configs/calibration.json")
        self.shape = StringVar(value="auto")
        self.top_camera = StringVar(value="0")
        self.side_camera = StringVar(value="1")
        self.frames = StringVar(value="8")
        self.status = StringVar(value="Ready")
        self._build_ui()

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"))
        style.configure("Volume.TLabel", font=("Segoe UI", 34, "bold"))
        style.configure("Status.TLabel", font=("Segoe UI", 12))

        outer = ttk.Frame(self.root, padding=24)
        outer.pack(fill=BOTH, expand=True)
        ttk.Label(
            outer,
            text="Photoelectric Volume Metrology",
            style="Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            outer,
            text="Calibrated top + side acquisition, automatic volume and result lock",
        ).pack(anchor="w", pady=(2, 18))

        controls = ttk.LabelFrame(outer, text="Measurement setup", padding=14)
        controls.pack(fill="x")
        ttk.Label(controls, text="Calibration").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.calibration_path, width=54).grid(
            row=0, column=1, columnspan=3, sticky="ew", padx=8
        )
        ttk.Button(controls, text="Browse", command=self._choose_calibration).grid(
            row=0, column=4
        )
        ttk.Label(controls, text="Shape").grid(
            row=1, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Combobox(
            controls,
            textvariable=self.shape,
            values=("auto", "l_block", "cylinder"),
            state="readonly",
            width=16,
        ).grid(row=1, column=1, sticky="w", padx=8, pady=(10, 0))
        ttk.Label(controls, text="Top camera").grid(row=1, column=2, pady=(10, 0))
        ttk.Spinbox(
            controls,
            from_=0,
            to=9,
            textvariable=self.top_camera,
            width=5,
        ).grid(row=1, column=3, sticky="w", padx=(4, 16), pady=(10, 0))
        ttk.Label(controls, text="Side camera").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Spinbox(
            controls,
            from_=0,
            to=9,
            textvariable=self.side_camera,
            width=5,
        ).grid(row=2, column=1, sticky="w", padx=8, pady=(8, 0))
        ttk.Label(controls, text="Frames").grid(row=2, column=2, pady=(8, 0))
        ttk.Spinbox(
            controls,
            from_=1,
            to=30,
            textvariable=self.frames,
            width=5,
        ).grid(row=2, column=3, sticky="w", padx=(4, 16), pady=(8, 0))
        controls.columnconfigure(1, weight=1)

        self.measure_button = ttk.Button(
            outer,
            text="MEASURE",
            command=self._measure_async,
        )
        self.measure_button.pack(fill="x", pady=(16, 8), ipady=8)
        self.lock_button = ttk.Button(
            outer,
            text="LOCK RESULT",
            command=self._lock_result,
            state="disabled",
        )
        self.lock_button.pack(fill="x", pady=(0, 16), ipady=6)

        result_card = ttk.LabelFrame(outer, text="Locked output candidate", padding=18)
        result_card.pack(fill=BOTH, expand=True)
        self.volume_label = ttk.Label(
            result_card,
            text="--.--- cm3",
            style="Volume.TLabel",
        )
        self.volume_label.pack(anchor="w")
        self.detail = ttk.Label(
            result_card,
            text="No measurement yet.",
            justify=LEFT,
            anchor="nw",
        )
        self.detail.pack(fill=BOTH, expand=True, pady=(12, 0))
        ttk.Label(
            outer,
            textvariable=self.status,
            style="Status.TLabel",
        ).pack(fill="x", pady=(12, 0))

    def _choose_calibration(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select calibration JSON",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if selected:
            self.calibration_path.set(selected)

    def _measure_async(self) -> None:
        if not Path(self.calibration_path.get()).exists():
            messagebox.showerror(
                "Calibration missing", "Load a calibration JSON first."
            )
            return
        self.measure_button.configure(state="disabled")
        self.lock_button.configure(state="disabled")
        self.status.set("Capturing and measuring...")
        thread = threading.Thread(target=self._measure_worker, daemon=True)
        thread.start()

    def _measure_worker(self) -> None:
        try:
            calibration = CalibrationBundle.load(self.calibration_path.get())
            frame_count = int(self.frames.get())
            top_config = CameraConfig(
                index=int(self.top_camera.get()),
                width=calibration.top_intrinsics.image_size[0],
                height=calibration.top_intrinsics.image_size[1],
            )
            side_config = CameraConfig(
                index=int(self.side_camera.get()),
                width=calibration.side_intrinsics.image_size[0],
                height=calibration.side_intrinsics.image_size[1],
            )
            with OpenCVCamera(top_config) as camera:
                top = camera.capture_average(frame_count)
                top_warnings = list(camera.warnings)
            with OpenCVCamera(side_config) as camera:
                side = camera.capture_average(frame_count)
                side_warnings = list(camera.warnings)
            result = measure_calibrated_images(
                top,
                side,
                calibration.top_plane.homography_px_to_mm,
                calibration.side_plane.homography_px_to_mm,
                shape_mode=self.shape.get(),
                top_intrinsics=calibration.top_intrinsics,
                side_intrinsics=calibration.side_intrinsics,
            )
            self.root.after(
                0,
                self._measurement_finished,
                result,
                top,
                side,
                top_warnings + side_warnings,
            )
        except (
            MeasurementFailure,
            ValueError,
            RuntimeError,
            OSError,
            KeyError,
            cv2.error,
        ) as exc:
            self.root.after(0, self._measurement_failed, str(exc))

    def _measurement_finished(
        self,
        result: VolumeResult,
        top: cv2.typing.MatLike,
        side: cv2.typing.MatLike,
        warnings: list[str],
    ) -> None:
        self.last_result = result
        self.last_images = (top, side)
        self.volume_label.configure(text=f"{result.volume_cm3:.3f} cm3")
        dimensions = ", ".join(
            f"{key.replace('_mm', '')}={value:.2f}"
            for key, value in result.dimensions.items()
        )
        issues = "\n".join(issue.message for issue in result.issues) or "none"
        warning_text = "\n".join(warnings) or "none"
        self.detail.configure(
            text=(
                f"shape: {result.shape}\n"
                f"dimensions (mm): {dimensions}\n"
                f"predicted relative sigma: {result.relative_sigma * 100:.3f}%\n"
                f"confidence: {result.confidence * 100:.1f}%\n"
                f"fit time: {result.elapsed_ms:.1f} ms\n"
                f"quality issues: {issues}\n"
                f"camera warnings: {warning_text}"
            )
        )
        self.status.set("Result ready. Check it, then lock once.")
        self.measure_button.configure(state="normal")
        if (
            not result.has_errors
            and result.confidence >= self.session.minimum_confidence
        ):
            self.lock_button.configure(state="normal")

    def _measurement_failed(self, error: str) -> None:
        self.measure_button.configure(state="normal")
        self.status.set("Measurement failed.")
        messagebox.showerror("Measurement failed", error)

    def _lock_result(self) -> None:
        if self.last_result is None or self.last_images is None:
            return
        if self.session.started_at is None:
            self.session.start()
        target_key = self.last_result.shape
        try:
            self.session.record(target_key, self.last_result)
        except (RuntimeError, ValueError) as exc:
            messagebox.showerror("Cannot lock", str(exc))
            return
        output = Path("output/gui") / f"{len(self.session.records):02d}_{target_key}"
        output.mkdir(parents=True, exist_ok=True)
        for name, image in zip(("top", "side"), self.last_images):
            ok, encoded = cv2.imencode(".png", image)
            if ok:
                (output / f"{name}.png").write_bytes(encoded.tobytes())
        (output / "result.json").write_text(
            json.dumps(self.last_result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.lock_button.configure(state="disabled")
        self.status.set(
            f"Locked {len(self.session.records)} result(s). "
            f"Remaining {self.session.remaining_s():.1f} s."
        )


def run_gui() -> None:
    root = Tk()
    MeasurementApp(root)
    root.mainloop()
