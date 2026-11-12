from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import VolumeResult, volume_mm3
from .image_measure import _major_contours, segment_backlit_object
from .image_pipeline import measure_realistic_case
from .realistic import (
    RealisticImageCase,
    make_realistic_case,
    make_realistic_demo_cases,
)


@dataclass(frozen=True)
class ImageDemoRecord:
    case: RealisticImageCase
    results: tuple[VolumeResult, ...]


def _font(
    size: int, bold: bool = False
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf"]
        if bold
        else ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _save_cv_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Unable to encode image {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def _mask_preview(case: RealisticImageCase, width: int, height: int) -> Image.Image:
    gray = cv2.cvtColor(case.top_gray, cv2.COLOR_GRAY2BGR)
    mask = segment_backlit_object(case.top_gray, object_is_dark=True)
    outer, holes = _major_contours(mask)
    cv2.drawContours(gray, [outer], -1, (65, 210, 145), 3, cv2.LINE_AA)
    for hole in holes:
        cv2.drawContours(gray, [hole], -1, (30, 160, 255), 3, cv2.LINE_AA)
    scale = min(width / gray.shape[1], height / gray.shape[0])
    resized = cv2.resize(
        gray,
        (max(1, int(gray.shape[1] * scale)), max(1, int(gray.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tile = Image.new("RGB", (width, height), "#111820")
    image = Image.fromarray(rgb)
    tile.paste(
        image,
        ((width - image.width) // 2, (height - image.height) // 2),
    )
    return tile


def _side_preview(case: RealisticImageCase, width: int, height: int) -> Image.Image:
    gray = case.side_gray
    scale = min(width / gray.shape[1], height / gray.shape[0])
    resized = cv2.resize(
        gray,
        (max(1, int(gray.shape[1] * scale)), max(1, int(gray.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    tile = Image.new("L", (width, height), 18)
    image = Image.fromarray(resized)
    tile.paste(
        image,
        ((width - image.width) // 2, (height - image.height) // 2),
    )
    return tile.convert("RGB")


def render_image_dashboard(
    records: list[ImageDemoRecord],
    output_path: Path,
) -> None:
    width, height = 1840, 1050
    canvas = Image.new("RGB", (width, height), "#f2efe7")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(42, True)
    heading_font = _font(28, True)
    body_font = _font(21)
    small_font = _font(17)
    draw.text(
        (68, 38),
        "Calibrated raw-image measurement pipeline",
        font=title_font,
        fill="#17202a",
    )
    draw.text(
        (70, 94),
        "Perspective + rotation + blur + noise + JPEG stress render",
        font=body_font,
        fill="#5b6672",
    )
    draw.line((68, 136, width - 68, 136), fill="#abb4bf", width=2)

    card_width = 548
    gap = 32
    for index, record in enumerate(records):
        case = record.case
        result = record.results[0]
        truth_cm3 = (
            volume_mm3(case.shape, case.truth_dimensions) / 1000.0  # type: ignore[arg-type]
        )
        relative_error = abs(result.volume_cm3 - truth_cm3) / truth_cm3
        if len(record.results) > 1:
            repeat_difference = abs(
                record.results[0].volume_cm3 - record.results[1].volume_cm3
            ) / np.mean([record.results[0].volume_cm3, record.results[1].volume_cm3])
        else:
            repeat_difference = 0.0

        x = 68 + index * (card_width + gap)
        y = 168
        draw.rounded_rectangle(
            (x, y, x + card_width, y + 812),
            radius=20,
            fill="#ffffff",
            outline="#d4cec2",
            width=2,
        )
        draw.text((x + 24, y + 18), case.name, font=heading_font, fill="#17202a")
        top_preview = _mask_preview(case, 252, 230)
        side_preview = _side_preview(case, 252, 230)
        canvas.paste(top_preview, (x + 18, y + 62))
        canvas.paste(side_preview, (x + 278, y + 62))
        draw.text((x + 24, y + 72), "TOP", font=small_font, fill="#e7edf2")
        draw.text((x + 284, y + 72), "SIDE", font=small_font, fill="#382f20")

        metric_y = y + 330
        draw.text(
            (x + 24, metric_y),
            f"TRUTH     {truth_cm3:8.3f} cm3",
            font=body_font,
            fill="#5b6672",
        )
        draw.text(
            (x + 24, metric_y + 34),
            f"MEASURED  {result.volume_cm3:8.3f} cm3",
            font=heading_font,
            fill="#08705c",
        )
        rows = [
            ("relative error", f"{relative_error * 100:.3f}%"),
            ("repeat difference", f"{repeat_difference * 100:.3f}%"),
            ("predicted sigma", f"{result.relative_sigma * 100:.3f}%"),
            ("confidence", f"{result.confidence * 100:.1f}%"),
        ]
        for row_index, (label, value) in enumerate(rows):
            draw.text(
                (x + 24, metric_y + 90 + row_index * 34),
                label,
                font=body_font,
                fill="#5b6672",
            )
            draw.text(
                (x + 234, metric_y + 90 + row_index * 34),
                value,
                font=body_font,
                fill="#17202a",
            )
        values = " ".join(
            f"{key.replace('_mm', '')}={value:.2f}"
            for key, value in result.dimensions.items()
        )
        draw.text(
            (x + 24, metric_y + 242),
            f"dimensions (mm): {values}",
            font=small_font,
            fill="#52606d",
        )
        top_diag = result.diagnostics.get("top", {})
        classifier = top_diag.get("classified_shape", "n/a")
        draw.text(
            (x + 24, metric_y + 281),
            f"classifier: {classifier}",
            font=small_font,
            fill="#52606d",
        )
        draw.text(
            (x + 24, metric_y + 310),
            f"fit time: {result.elapsed_ms:.1f} ms",
            font=small_font,
            fill="#52606d",
        )

    footer = (
        "This is still a validation harness, not a claim of national-award accuracy. "
        "The remaining acceptance gate is physical calibration, laser triangulation and repeated measurements on the official objects."
    )
    draw.text(
        (68, height - 52),
        footer,
        font=small_font,
        fill="#6b7280",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, optimize=True)


def run_image_demo(
    output_dir: Path,
    seed: int = 20261110,
) -> tuple[list[ImageDemoRecord], dict[str, Path]]:
    rng = np.random.default_rng(seed)
    base_cases = make_realistic_demo_cases(seed)
    records: list[ImageDemoRecord] = []
    input_dir = output_dir / "input_images"
    for case in base_cases:
        repeated = []
        for repeat in range(2):
            repeat_case = make_realistic_case(
                case.shape,
                case.truth_dimensions,
                rng,
                name=f"{case.name} / repeat {repeat + 1}",
            )
            repeated.append(measure_realistic_case(repeat_case, shape_mode="auto"))
            _save_cv_image(
                input_dir / f"{case.shape}_r{repeat + 1}_top.png",
                repeat_case.top_gray,
            )
            _save_cv_image(
                input_dir / f"{case.shape}_r{repeat + 1}_side.png",
                repeat_case.side_gray,
            )
        records.append(
            ImageDemoRecord(
                case=case,
                results=tuple(repeated),
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard = output_dir / "dashboard.png"
    render_image_dashboard(records, dashboard)
    report_path = output_dir / "report.json"
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "calibrated_image_simulation",
        "hardware_validated": False,
        "measurements": [
            {
                "name": record.case.name,
                "shape": record.case.shape,
                "truth_dimensions": record.case.truth_dimensions,
                "truth_cm3": volume_mm3(
                    record.case.shape,
                    record.case.truth_dimensions,  # type: ignore[arg-type]
                )
                / 1000.0,
                "runs": [result.to_dict() for result in record.results],
            }
            for record in records
        ],
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return records, {"dashboard": dashboard, "report": report_path}
