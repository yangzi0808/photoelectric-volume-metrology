from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from textwrap import fill
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .benchmark import BenchmarkSummary
from .features import extract_top_features, tight_bbox
from .geometry import VolumeResult, volume_mm3
from .synthetic import SyntheticCase


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


def _mask_image(case: SyntheticCase) -> tuple[Image.Image, tuple[int, int, int, int]]:
    binary = np.asarray(case.top_mask) > 0
    bbox = tight_bbox(binary)
    pad = 24
    x0 = max(0, bbox.x0 - pad)
    y0 = max(0, bbox.y0 - pad)
    x1 = min(binary.shape[1] - 1, bbox.x1 + pad)
    y1 = min(binary.shape[0] - 1, bbox.y1 + pad)
    crop = binary[y0 : y1 + 1, x0 : x1 + 1]
    rgb = np.empty((*crop.shape, 3), dtype=np.uint8)
    rgb[crop] = (104, 174, 215)
    rgb[~crop] = (18, 23, 31)
    return Image.fromarray(rgb), (x0, y0, x1, y1)


def render_dashboard(
    records: list[tuple[SyntheticCase, list[VolumeResult]]],
    output_path: Path,
) -> None:
    width, height = 1760, 1120
    canvas = Image.new("RGB", (width, height), "#f4f1ea")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(42, bold=True)
    heading_font = _font(26, bold=True)
    body_font = _font(21)
    small_font = _font(17)
    tiny_font = _font(14)
    draw.text(
        (72, 45), "Known-shape volume measurement MVP", fill="#17202a", font=title_font
    )
    draw.text(
        (74, 102),
        "Back-lit silhouette + fixed side height + local laser depth  |  synthetic verification",
        fill="#52606d",
        font=body_font,
    )
    draw.line((72, 145, width - 72, 145), fill="#aeb7c2", width=2)

    card_width = 515
    gap = 40
    for index, (case, results) in enumerate(records):
        x = 72 + index * (card_width + gap)
        y = 182
        result = results[0]
        truth_mm3 = volume_mm3(case.shape, case.truth_dimensions)  # type: ignore[arg-type]
        truth_cm3 = truth_mm3 / 1000.0
        relative_error = abs(result.volume_cm3 - truth_cm3) / truth_cm3
        if len(results) >= 2:
            repeat_difference = abs(
                results[0].volume_cm3 - results[1].volume_cm3
            ) / np.mean([results[0].volume_cm3, results[1].volume_cm3])
        else:
            repeat_difference = 0.0

        draw.rounded_rectangle(
            (x, y, x + card_width, y + 870),
            radius=20,
            fill="#ffffff",
            outline="#d8d1c6",
            width=2,
        )
        draw.text((x + 28, y + 20), case.name, fill="#17202a", font=heading_font)

        mask_img, crop_box = _mask_image(case)
        display_width, display_height = 455, 330
        scale = min(display_width / mask_img.width, display_height / mask_img.height)
        display_size = (
            max(1, int(mask_img.width * scale)),
            max(1, int(mask_img.height * scale)),
        )
        resized = mask_img.resize(display_size, Image.Resampling.NEAREST)
        image_x = x + 30 + (display_width - display_size[0]) // 2
        image_y = y + 66
        canvas.paste(resized, (image_x, image_y))
        draw.rectangle(
            (
                image_x - 1,
                image_y - 1,
                image_x + display_size[0],
                image_y + display_size[1],
            ),
            outline="#c2cad3",
            width=1,
        )

        features = extract_top_features(case.top_mask)
        if features.hole is not None:
            hole = features.hole.bbox
            left = image_x + (hole.x0 - crop_box[0]) * scale
            top = image_y + (hole.y0 - crop_box[1]) * scale
            right = image_x + (hole.x1 - crop_box[0] + 1) * scale
            bottom = image_y + (hole.y1 - crop_box[1] + 1) * scale
            draw.rectangle(
                (left, top, right, bottom),
                outline="#f0a126",
                width=4,
            )
        elif features.notch is not None:
            draw.text(
                (image_x + 8, image_y + 8),
                f"notch: {features.notch.corner}",
                fill="#f6c453",
                font=tiny_font,
            )

        info_y = image_y + display_height + 18
        draw.text(
            (x + 30, info_y),
            f"TRUTH    {truth_cm3:8.3f} cm3",
            fill="#52606d",
            font=body_font,
        )
        draw.text(
            (x + 30, info_y + 34),
            f"MEASURED {result.volume_cm3:8.3f} cm3",
            fill="#0c6b58",
            font=heading_font,
        )
        draw.text(
            (x + 30, info_y + 78),
            f"relative error  {relative_error * 100:5.2f}%",
            fill="#17202a",
            font=body_font,
        )
        draw.text(
            (x + 30, info_y + 110),
            f"repeat diff     {repeat_difference * 100:5.2f}%",
            fill="#17202a",
            font=body_font,
        )
        draw.text(
            (x + 30, info_y + 142),
            f"pred. sigma     {result.relative_sigma * 100:5.2f}%",
            fill="#17202a",
            font=body_font,
        )
        draw.text(
            (x + 30, info_y + 174),
            f"confidence      {result.confidence * 100:5.1f}%",
            fill="#17202a",
            font=body_font,
        )

        dimensions = "  ".join(
            f"{key.replace('_mm', '')}={value:.1f}"
            for key, value in result.dimensions.items()
        )
        wrapped = fill(f"dimensions (mm): {dimensions}", width=47)
        draw.text(
            (x + 30, info_y + 220),
            wrapped,
            fill="#52606d",
            font=small_font,
            spacing=5,
        )
        issue_text = (
            "quality: OK"
            if not result.issues
            else f"quality: {len(result.issues)} issue(s)"
        )
        issue_color = "#0c6b58" if not result.issues else "#b65b20"
        draw.text(
            (x + 30, info_y + 310),
            issue_text,
            fill=issue_color,
            font=small_font,
        )

    footer = (
        "Prototype scope: segmented masks and calibrated metric scales. "
        "Camera capture, sub-pixel edge refinement, laser-plane calibration, "
        "thermal tests and real-target validation remain to be completed."
    )
    draw.text(
        (72, height - 70),
        fill(footer, width=145),
        fill="#6b7280",
        font=small_font,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def _write_csv(summary: BenchmarkSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summary.rows[0].as_dict().keys())
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary.rows:
            writer.writerow(row.as_dict())


def write_report(
    records: list[tuple[SyntheticCase, list[VolumeResult]]],
    benchmark: BenchmarkSummary,
    output_dir: Path,
    seed: int,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    measurements: list[dict[str, Any]] = []
    for case, results in records:
        truth_mm3 = volume_mm3(case.shape, case.truth_dimensions)  # type: ignore[arg-type]
        truth_cm3 = truth_mm3 / 1000.0
        measurements.append(
            {
                "name": case.name,
                "shape": case.shape,
                "truth_dimensions_mm": case.truth_dimensions,
                "truth_volume_cm3": round(truth_cm3, 6),
                "runs": [
                    {
                        **result.to_dict(),
                        "relative_error": round(
                            abs(result.volume_cm3 - truth_cm3) / truth_cm3,
                            8,
                        ),
                    }
                    for result in results
                ],
            }
        )
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "seed": seed,
        "status": "synthetic_algorithm_demo",
        "hardware_validated": False,
        "measurements": measurements,
        "benchmark": benchmark.as_dict(),
        "interpretation": (
            "Errors describe geometry and calibration propagation against "
            "synthetic masks. They do not include real camera blur, specular "
            "reflections, laser speckle, robot repeatability or target form error."
        ),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_path = output_dir / "benchmark.csv"
    _write_csv(benchmark, csv_path)
    dashboard_path = output_dir / "dashboard.png"
    render_dashboard(records, dashboard_path)

    demo_errors = [
        abs(result.volume_cm3 - volume_mm3(case.shape, case.truth_dimensions) / 1000.0)
        / (volume_mm3(case.shape, case.truth_dimensions) / 1000.0)
        for case, results in records
        for result in results
    ]
    summary_text = "\n".join(
        [
            "VOLUMECV DEMO SUMMARY",
            "=====================",
            "Status: synthetic algorithm prototype; hardware not yet validated.",
            f"Demo repeated measurements: {len(demo_errors)}",
            f"Demo max relative error: {max(demo_errors) * 100:.3f}%",
            f"Synthetic benchmark rows: {len(benchmark.rows)}",
            f"Benchmark mean absolute error: {benchmark.mean_abs_error * 100:.3f}%",
            f"Benchmark p95 absolute error: {benchmark.p95_abs_error * 100:.3f}%",
            f"Benchmark <=5% pass rate: {benchmark.pass_5pct_rate * 100:.2f}%",
            f"Mean repeat difference: {benchmark.mean_repeat_difference * 100:.3f}%",
            f"95% repeat-difference p95: {benchmark.p95_repeat_difference * 100:.3f}%",
            f"Predicted two-sigma coverage: {benchmark.two_sigma_coverage * 100:.2f}%",
            f"Failures: {benchmark.failure_count}",
            "",
            "Next gate: replace synthetic masks with real back-lit captures and",
            "validate calibration, edge extraction and blind-slot depth on printed proxies.",
        ]
    )
    summary_path = output_dir / "summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    return {
        "report": report_path,
        "benchmark_csv": csv_path,
        "dashboard": dashboard_path,
        "summary": summary_path,
    }
