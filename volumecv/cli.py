from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .benchmark import run_benchmark
from .calibration import (
    CalibrationBundle,
    CharucoSpec,
    calibrate_intrinsics,
    calibrate_plane_homography,
    fit_laser_plane,
    make_charuco_image,
)
from .camera import CameraConfig, OpenCVCamera, list_available_cameras
from .demo import run_demo
from .gui import run_gui
from .image_benchmark import run_image_benchmark, write_image_benchmark
from .image_demo import run_image_demo
from .image_pipeline import measure_calibrated_images
from .laser_measure import (
    calibrate_laser_plane_from_known_heights,
    estimate_blind_slot_depth,
    pose_from_plane_homography,
    triangulate_lines_in_stage,
)
from .laser_scan import LaserLineConfig, extract_laser_line
from .parallax_benchmark import run_parallax_benchmark, write_parallax_benchmark
from .pipeline import MeasurementFailure, measure_case
from .synthetic import SyntheticCase


def _load_mask(path: Path, threshold: int, invert: bool) -> np.ndarray:
    image = Image.open(path).convert("L")
    values = np.asarray(image)
    mask = values < threshold if invert else values >= threshold
    if not mask.any():
        raise ValueError("The loaded mask is empty. Check --threshold and --invert.")
    return mask


def _load_depth(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    values = []
    for token in text.replace(",", " ").split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    if len(values) < 8:
        raise ValueError("The depth file must contain at least 8 numeric samples.")
    return np.asarray(values)


def _load_image(path: Path, grayscale: bool = True) -> np.ndarray:
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imdecode(data, mode)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def _save_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"Could not encode image as {suffix}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def _charuco_spec(args: argparse.Namespace) -> CharucoSpec:
    return CharucoSpec(
        squares_x=args.board_squares_x,
        squares_y=args.board_squares_y,
        square_length_mm=args.square_length_mm,
        marker_length_mm=args.marker_length_mm,
        dictionary_name=args.dictionary,
    )


def _parse_roi(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    parts = [int(item.strip()) for item in value.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x0,y0,x1,y1.")
    return parts[0], parts[1], parts[2], parts[3]


def _cmd_demo(args: argparse.Namespace) -> int:
    records, benchmark, paths = run_demo(
        Path(args.out),
        seed=args.seed,
        benchmark_per_shape=args.benchmark_per_shape,
    )
    print("VOLUMECV synthetic demo")
    print("-----------------------")
    for case, results in records:
        first, second = results
        repeat_diff = abs(first.volume_cm3 - second.volume_cm3) / (
            (first.volume_cm3 + second.volume_cm3) / 2.0
        )
        print(
            f"{case.name:18s} {first.volume_cm3:9.3f} cm3  "
            f"sigma {first.relative_sigma * 100:5.2f}%  "
            f"repeat diff {repeat_diff * 100:5.2f}%  "
            f"confidence {first.confidence * 100:5.1f}%"
        )
    print(
        f"Benchmark: mean |e| {benchmark.mean_abs_error * 100:.3f}%, "
        f"p95 {benchmark.p95_abs_error * 100:.3f}%, "
        f"<=5% rate {benchmark.pass_5pct_rate * 100:.2f}%"
    )
    print("Outputs:")
    for name, path in paths.items():
        print(f"  {name:14s} {path.resolve()}")
    return 0


def _cmd_image_benchmark(args: argparse.Namespace) -> int:
    summary = run_image_benchmark(
        per_shape=args.per_shape,
        repeats=args.repeats,
        seed=args.seed,
    )
    paths = write_image_benchmark(summary, Path(args.out))
    print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")
    return 0


def _cmd_image_demo(args: argparse.Namespace) -> int:
    records, paths = run_image_demo(Path(args.out), seed=args.seed)
    print("VOLUMECV calibrated image demo")
    print("------------------------------")
    for record in records:
        first = record.results[0]
        truth = json.loads(json.dumps(record.case.truth_dimensions))
        print(
            f"{record.case.name:20s} {first.volume_cm3:9.3f} cm3 "
            f"sigma {first.relative_sigma * 100:5.3f}% "
            f"confidence {first.confidence * 100:5.1f}%"
        )
        print(f"  truth dimensions: {truth}")
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")
    return 0


def _cmd_parallax_benchmark(args: argparse.Namespace) -> int:
    rows = run_parallax_benchmark(per_shape=args.per_shape, seed=args.seed)
    paths = write_parallax_benchmark(rows, Path(args.out))
    baseline = np.asarray([float(row["baseline_error"]) for row in rows])
    corrected = np.asarray([float(row["corrected_error"]) for row in rows])
    print(
        json.dumps(
            {
                "sample_count": len(rows),
                "baseline_mean_error": float(baseline.mean()),
                "baseline_p95_error": float(np.quantile(baseline, 0.95)),
                "corrected_mean_error": float(corrected.mean()),
                "corrected_p95_error": float(np.quantile(corrected, 0.95)),
                "corrected_max_error": float(corrected.max()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    image_dir = Path(args.images)
    image_paths = sorted(
        path
        for path in image_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    )
    if not image_paths:
        raise ValueError("No calibration images were found.")
    images = [_load_image(path) for path in image_paths]
    spec = _charuco_spec(args)
    top_intrinsics = calibrate_intrinsics(images, spec, min_views=args.min_views)
    if args.side_images:
        side_dir = Path(args.side_images)
        side_paths = sorted(
            path
            for path in side_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        )
        side_intrinsics = calibrate_intrinsics(
            [_load_image(path) for path in side_paths],
            spec,
            min_views=args.min_views,
        )
    else:
        side_intrinsics = top_intrinsics
    top_plane = calibrate_plane_homography(
        _load_image(Path(args.top_image)),
        spec,
        intrinsics=top_intrinsics,
    )
    side_plane = calibrate_plane_homography(
        _load_image(Path(args.side_image)),
        spec,
        intrinsics=side_intrinsics,
    )
    laser_plane = None
    if args.laser_points:
        points = np.loadtxt(args.laser_points, delimiter=",")
        laser_plane = fit_laser_plane(points)
    bundle = CalibrationBundle(
        top_intrinsics=top_intrinsics,
        side_intrinsics=side_intrinsics,
        top_plane=top_plane,
        side_plane=side_plane,
        laser_plane=laser_plane,
    )
    output = bundle.save(args.out)
    print(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2))
    print(f"Calibration saved: {output.resolve()}")
    if top_intrinsics.rms > 0.5 or side_intrinsics.rms > 0.5:
        print(
            "WARNING: camera intrinsic RMS exceeds 0.5 px.",
            file=sys.stderr,
        )
    if top_plane.rms_mm > 0.2 or side_plane.rms_mm > 0.2:
        print(
            "WARNING: plane homography RMS exceeds 0.2 mm.",
            file=sys.stderr,
        )
    if laser_plane is not None and laser_plane.rms_mm > 0.2:
        print(
            f"WARNING: laser plane RMS {laser_plane.rms_mm:.3f} mm exceeds 0.2 mm.",
            file=sys.stderr,
        )
    return 0


def _cmd_measure_images(args: argparse.Namespace) -> int:
    calibration = CalibrationBundle.load(args.calibration)
    depth = _load_depth(Path(args.depth_file)) if args.depth_file else None
    result = measure_calibrated_images(
        _load_image(Path(args.top_image)),
        _load_image(Path(args.side_image)),
        calibration.top_plane.homography_px_to_mm,
        calibration.side_plane.homography_px_to_mm,
        depth_profile_mm=depth,
        shape_mode=args.shape,
        top_intrinsics=calibration.top_intrinsics,
        side_intrinsics=calibration.side_intrinsics,
        top_roi=_parse_roi(args.top_roi),
        side_roi=_parse_roi(args.side_roi),
    )
    payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload)
    return 1 if result.has_errors else 0


def _cmd_calibrate_laser(args: argparse.Namespace) -> int:
    bundle = CalibrationBundle.load(args.calibration)
    image_paths = [Path(value) for value in args.images.split(",") if value]
    heights = [float(value) for value in args.heights.split(",") if value]
    if len(image_paths) != len(heights):
        raise ValueError("--images and --heights must contain equal counts.")
    backgrounds = None
    if args.backgrounds:
        background_paths = [
            Path(value) for value in args.backgrounds.split(",") if value
        ]
        if len(background_paths) != len(image_paths):
            raise ValueError("--backgrounds must match the laser image count.")
        backgrounds = [_load_image(path) for path in background_paths]
    pose = pose_from_plane_homography(
        np.linalg.inv(bundle.top_plane.homography_px_to_mm),
        bundle.top_intrinsics.camera_matrix,
    )
    laser_plane = calibrate_laser_plane_from_known_heights(
        [_load_image(path, grayscale=False) for path in image_paths],
        heights,
        bundle.top_intrinsics,
        pose,
        LaserLineConfig(axis=args.axis),
        backgrounds,
    )
    updated = replace(bundle, laser_plane=laser_plane)
    output = updated.save(args.out)
    print(json.dumps(laser_plane.to_dict(), ensure_ascii=False, indent=2))
    print(f"Calibration updated: {output.resolve()}")
    if laser_plane.rms_mm > 0.2:
        print(
            f"WARNING: laser plane RMS {laser_plane.rms_mm:.3f} mm exceeds 0.2 mm.",
            file=sys.stderr,
        )
    return 0


def _cmd_measure_blind(args: argparse.Namespace) -> int:
    bundle = CalibrationBundle.load(args.calibration)
    if bundle.laser_plane is None:
        raise ValueError("Calibration has no laser plane. Run calibrate-laser first.")
    image_paths = [Path(value) for value in args.laser_images.split(",") if value]
    if not image_paths:
        raise ValueError("At least one laser image is required.")
    backgrounds = None
    if args.backgrounds:
        background_paths = [
            Path(value) for value in args.backgrounds.split(",") if value
        ]
        if len(background_paths) != len(image_paths):
            raise ValueError("--backgrounds must match the laser image count.")
        backgrounds = [_load_image(path, grayscale=False) for path in background_paths]
    else:
        backgrounds = [None] * len(image_paths)
    images = [_load_image(path, grayscale=False) for path in image_paths]
    config = LaserLineConfig(axis=args.axis)
    lines = [
        extract_laser_line(image, config, background)
        for image, background in zip(images, backgrounds)
    ]
    angles = (
        [float(value) for value in args.angles.split(",") if value]
        if args.angles
        else [0.0] * len(lines)
    )
    pose = pose_from_plane_homography(
        np.linalg.inv(bundle.top_plane.homography_px_to_mm),
        bundle.top_intrinsics.camera_matrix,
    )
    cloud = triangulate_lines_in_stage(
        lines,
        bundle.top_intrinsics,
        bundle.laser_plane,
        pose,
        angles,
    )
    depth = estimate_blind_slot_depth(cloud)
    result = measure_calibrated_images(
        _load_image(Path(args.top_image)),
        _load_image(Path(args.side_image)),
        bundle.top_plane.homography_px_to_mm,
        bundle.side_plane.homography_px_to_mm,
        depth_profile_mm=cloud[:, 2],
        shape_mode="blind_slot",
        top_intrinsics=bundle.top_intrinsics,
        side_intrinsics=bundle.side_intrinsics,
        top_roi=_parse_roi(args.top_roi),
        side_roi=_parse_roi(args.side_roi),
    )
    payload = {
        "laser_depth": {
            "depth_mm": depth.depth_mm,
            "sigma_mm": depth.sigma_mm,
            "top_level_mm": depth.top_level_mm,
            "floor_level_mm": depth.floor_level_mm,
            "inlier_count": depth.inlier_count,
            "rejected_count": depth.rejected_count,
            "stage_points": int(cloud.shape[0]),
        },
        "result": result.to_dict(),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved: {output.resolve()}")
    return 1 if result.has_errors else 0


def _cmd_generate_charuco(args: argparse.Namespace) -> int:
    image = make_charuco_image(
        _charuco_spec(args),
        pixels_per_mm=args.pixels_per_mm,
        margin_mm=args.margin_mm,
    )
    output = Path(args.out)
    _save_image(output, image)
    print(f"Generated ChArUco target: {output.resolve()}")
    return 0


def _cmd_camera_list(_: argparse.Namespace) -> int:
    indices = list_available_cameras()
    print(json.dumps({"camera_indices": indices}, ensure_ascii=False))
    return 0


def _cmd_gui(_: argparse.Namespace) -> int:
    run_gui()
    return 0


def _cmd_live_measure(args: argparse.Namespace) -> int:
    calibration = CalibrationBundle.load(args.calibration)
    top_config = CameraConfig(
        index=args.top_camera,
        width=calibration.intrinsics.image_size[0],
        height=calibration.intrinsics.image_size[1],
        exposure=args.exposure,
        gain=args.gain,
        focus=args.focus,
    )
    side_config = CameraConfig(
        index=args.side_camera,
        width=calibration.side_intrinsics.image_size[0],
        height=calibration.side_intrinsics.image_size[1],
        exposure=args.exposure,
        gain=args.gain,
        focus=args.focus,
    )
    with OpenCVCamera(top_config) as top_camera:
        top_image = top_camera.capture_average(args.frames)
        top_properties = top_camera.properties()
    with OpenCVCamera(side_config) as side_camera:
        side_image = side_camera.capture_average(args.frames)
        side_properties = side_camera.properties()
    result = measure_calibrated_images(
        top_image,
        side_image,
        calibration.top_plane.homography_px_to_mm,
        calibration.side_plane.homography_px_to_mm,
        depth_profile_mm=None,
        shape_mode=args.shape,
        top_intrinsics=calibration.top_intrinsics,
        side_intrinsics=calibration.side_intrinsics,
        top_roi=_parse_roi(args.top_roi),
        side_roi=_parse_roi(args.side_roi),
    )
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_image(output_dir / "top.png", top_image)
    _save_image(output_dir / "side.png", side_image)
    payload = {
        "camera_properties": {
            "top": top_properties,
            "side": side_properties,
        },
        "result": result.to_dict(),
    }
    result_path = output_dir / "measurement.json"
    result_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved: {result_path.resolve()}")
    return 1 if result.has_errors else 0


def _cmd_benchmark(args: argparse.Namespace) -> int:
    summary = run_benchmark(
        per_shape=args.per_shape,
        repeats=args.repeats,
        seed=args.seed,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "benchmark_summary.json"
    summary_path.write_text(
        json.dumps(summary.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))
    print(f"Summary: {summary_path.resolve()}")
    return 0


def _cmd_measure(args: argparse.Namespace) -> int:
    top_mask = _load_mask(Path(args.top_mask), args.threshold, args.invert)
    side_mask = _load_mask(Path(args.side_mask), args.threshold, args.invert)
    depth = _load_depth(Path(args.depth_file)) if args.depth_file else None
    case = SyntheticCase(
        name=Path(args.top_mask).stem,
        shape=args.shape if args.shape != "auto" else "l_block",
        top_mask=top_mask,
        side_mask=side_mask,
        depth_profile_mm=depth,
        truth_dimensions={},
        px_per_mm_x=args.top_px_per_mm_x,
        px_per_mm_y=args.top_px_per_mm_y,
        side_px_per_mm=args.side_px_per_mm,
    )
    result = measure_case(case, shape_mode=args.shape)
    payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    print(payload)
    return 1 if result.has_errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m volumecv",
        description="Known-shape photoelectric volume measurement prototype.",
    )
    subparsers = parser.add_subparsers(dest="command")

    demo = subparsers.add_parser("demo", help="Run the synthetic end-to-end demo.")
    demo.add_argument("--out", default="output/demo")
    demo.add_argument("--seed", type=int, default=20261110)
    demo.add_argument("--benchmark-per-shape", type=int, default=35)
    demo.set_defaults(func=_cmd_demo)

    benchmark = subparsers.add_parser(
        "benchmark", help="Run a larger synthetic accuracy benchmark."
    )
    benchmark.add_argument("--per-shape", type=int, default=100)
    benchmark.add_argument("--repeats", type=int, default=2)
    benchmark.add_argument("--seed", type=int, default=20261110)
    benchmark.add_argument("--out", default="output/benchmark")
    benchmark.set_defaults(func=_cmd_benchmark)

    image_benchmark = subparsers.add_parser(
        "image-benchmark",
        help="Benchmark the calibrated raw-image engine on realistic renders.",
    )
    image_benchmark.add_argument("--per-shape", type=int, default=20)
    image_benchmark.add_argument("--repeats", type=int, default=2)
    image_benchmark.add_argument("--seed", type=int, default=20261110)
    image_benchmark.add_argument("--out", default="output/image_benchmark")
    image_benchmark.set_defaults(func=_cmd_image_benchmark)

    image_demo = subparsers.add_parser(
        "image-demo",
        help="Render and measure three calibrated raw-image targets.",
    )
    image_demo.add_argument("--out", default="output/image_demo")
    image_demo.add_argument("--seed", type=int, default=20261110)
    image_demo.set_defaults(func=_cmd_image_demo)

    parallax = subparsers.add_parser(
        "parallax-benchmark",
        help="Measure targets with and without top-plane height compensation.",
    )
    parallax.add_argument("--per-shape", type=int, default=5)
    parallax.add_argument("--seed", type=int, default=20261110)
    parallax.add_argument("--out", default="output/parallax_benchmark")
    parallax.set_defaults(func=_cmd_parallax_benchmark)

    calibrate = subparsers.add_parser(
        "calibrate",
        help="Calibrate intrinsics, top plane, side plane and optional laser plane.",
    )
    calibrate.add_argument("--images", required=True)
    calibrate.add_argument("--side-images")
    calibrate.add_argument("--top-image", required=True)
    calibrate.add_argument("--side-image", required=True)
    calibrate.add_argument("--laser-points")
    calibrate.add_argument("--min-views", type=int, default=8)
    calibrate.add_argument("--out", default="configs/calibration.json")
    calibrate.add_argument("--board-squares-x", type=int, default=7)
    calibrate.add_argument("--board-squares-y", type=int, default=5)
    calibrate.add_argument("--square-length-mm", type=float, default=20.0)
    calibrate.add_argument("--marker-length-mm", type=float, default=15.0)
    calibrate.add_argument("--dictionary", default="DICT_6X6_250")
    calibrate.set_defaults(func=_cmd_calibrate)

    measure_images = subparsers.add_parser(
        "measure-images",
        help="Measure calibrated top/side images from a real camera.",
    )
    measure_images.add_argument("--calibration", required=True)
    measure_images.add_argument("--top-image", required=True)
    measure_images.add_argument("--side-image", required=True)
    measure_images.add_argument("--depth-file")
    measure_images.add_argument("--top-roi")
    measure_images.add_argument("--side-roi")
    measure_images.add_argument(
        "--shape",
        default="auto",
        choices=["auto", "l_block", "cylinder", "blind_slot"],
    )
    measure_images.add_argument("--out")
    measure_images.set_defaults(func=_cmd_measure_images)

    calibrate_laser = subparsers.add_parser(
        "calibrate-laser",
        help="Fit the laser plane using images on two or more known heights.",
    )
    calibrate_laser.add_argument("--calibration", required=True)
    calibrate_laser.add_argument(
        "--images",
        required=True,
        help="Comma-separated laser-on images.",
    )
    calibrate_laser.add_argument(
        "--heights",
        required=True,
        help="Comma-separated known stage heights in mm.",
    )
    calibrate_laser.add_argument(
        "--backgrounds",
        help="Optional comma-separated laser-off images.",
    )
    calibrate_laser.add_argument(
        "--axis",
        default="vertical",
        choices=["vertical", "horizontal"],
    )
    calibrate_laser.add_argument("--out", default="configs/calibration.json")
    calibrate_laser.set_defaults(func=_cmd_calibrate_laser)

    measure_blind = subparsers.add_parser(
        "measure-blind",
        help="Measure the blind-slot block with calibrated laser triangulation.",
    )
    measure_blind.add_argument("--calibration", required=True)
    measure_blind.add_argument("--top-image", required=True)
    measure_blind.add_argument("--side-image", required=True)
    measure_blind.add_argument(
        "--laser-images",
        required=True,
        help="Comma-separated laser-on images.",
    )
    measure_blind.add_argument("--backgrounds")
    measure_blind.add_argument("--top-roi")
    measure_blind.add_argument("--side-roi")
    measure_blind.add_argument(
        "--angles",
        help="Optional comma-separated turntable angles for each frame.",
    )
    measure_blind.add_argument(
        "--axis",
        default="vertical",
        choices=["vertical", "horizontal"],
    )
    measure_blind.add_argument(
        "--out",
        default="output/blind_measurement.json",
    )
    measure_blind.set_defaults(func=_cmd_measure_blind)

    generate_charuco = subparsers.add_parser(
        "generate-charuco",
        help="Generate a printable ChArUco calibration target.",
    )
    generate_charuco.add_argument("--out", default="configs/charuco.png")
    generate_charuco.add_argument("--pixels-per-mm", type=float, default=10.0)
    generate_charuco.add_argument("--margin-mm", type=float, default=20.0)
    generate_charuco.add_argument("--board-squares-x", type=int, default=7)
    generate_charuco.add_argument("--board-squares-y", type=int, default=5)
    generate_charuco.add_argument("--square-length-mm", type=float, default=20.0)
    generate_charuco.add_argument("--marker-length-mm", type=float, default=15.0)
    generate_charuco.add_argument("--dictionary", default="DICT_6X6_250")
    generate_charuco.set_defaults(func=_cmd_generate_charuco)

    camera_list = subparsers.add_parser(
        "camera-list",
        help="Probe Windows DirectShow camera indices.",
    )
    camera_list.set_defaults(func=_cmd_camera_list)

    gui = subparsers.add_parser(
        "gui",
        help="Open the two-camera desktop measurement interface.",
    )
    gui.set_defaults(func=_cmd_gui)

    live_measure = subparsers.add_parser(
        "live-measure",
        help="Capture from two cameras and measure a base target.",
    )
    live_measure.add_argument("--calibration", required=True)
    live_measure.add_argument("--top-camera", type=int, default=0)
    live_measure.add_argument("--side-camera", type=int, default=1)
    live_measure.add_argument("--frames", type=int, default=8)
    live_measure.add_argument("--exposure", type=float)
    live_measure.add_argument("--gain", type=float)
    live_measure.add_argument("--focus", type=float)
    live_measure.add_argument("--top-roi")
    live_measure.add_argument("--side-roi")
    live_measure.add_argument(
        "--shape",
        default="auto",
        choices=["auto", "l_block", "cylinder"],
    )
    live_measure.add_argument("--out", default="output/live_measure")
    live_measure.set_defaults(func=_cmd_live_measure)

    measure = subparsers.add_parser(
        "measure", help="Measure from segmented top/side masks."
    )
    measure.add_argument(
        "--shape",
        required=True,
        choices=["auto", "l_block", "cylinder", "blind_slot"],
    )
    measure.add_argument("--top-mask", required=True)
    measure.add_argument("--side-mask", required=True)
    measure.add_argument("--depth-file")
    measure.add_argument("--top-px-per-mm-x", type=float, required=True)
    measure.add_argument("--top-px-per-mm-y", type=float, required=True)
    measure.add_argument("--side-px-per-mm", type=float, required=True)
    measure.add_argument("--threshold", type=int, default=128)
    measure.add_argument("--invert", action="store_true")
    measure.add_argument("--out")
    measure.set_defaults(func=_cmd_measure)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        args = parser.parse_args(["image-demo"])
    try:
        return int(args.func(args))
    except (MeasurementFailure, ValueError, OSError, RuntimeError, cv2.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
