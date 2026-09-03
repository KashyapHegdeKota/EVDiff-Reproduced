#!/usr/bin/env python3
"""
Evaluate EvDiff, E2VID, and HyperE2VID on identical DSEC frames.

Supported prediction layouts:

Canonical:
    outputs/<method>/<sequence>/<index>.png

Native EVREAL:
    EVREAL/outputs/std/DSEC_subset/<sequence>/<method>/frame_<index>.png

Protocol:
    - Prediction index i is compared with images[i].
    - N reference images define N-1 event intervals.
    - Predictions and references are converted to grayscale.
    - MSE uses skimage.metrics.mean_squared_error.
    - SSIM uses EVREAL-compatible settings.
    - LPIPS uses PyIQA with grayscale repeated across three channels.

Outputs:
    results/comparison/per_frame.csv
    results/comparison/summary.csv
    results/comparison/comparison.csv
    results/comparison/comparison.md
    results/comparison/summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import mean_squared_error, structural_similarity
from tqdm import tqdm

try:
    import pyiqa
except ImportError:
    print(
        "ERROR: PyIQA is not installed.\n"
        "Install dependencies with:\n\n"
        "    pip install pyiqa scikit-image\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


DEFAULT_SEQUENCES = [
    "zurich_city_00_a",
    "zurich_city_02_a",
    "zurich_city_04_b",
]

METHODS = {
    "evdiff": {
        "display_name": "EvDiff",
        "canonical_dir": "evdiff",
        "evreal_dir": None,
    },
    "e2vid": {
        "display_name": "E2VID",
        "canonical_dir": "e2vid",
        "evreal_dir": "E2VID",
    },
    "hypere2vid": {
        "display_name": "HyperE2VID",
        "canonical_dir": "hypere2vid",
        "evreal_dir": "HyperE2VID",
    },
}

NUMERIC_FRAME_PATTERN = re.compile(r"^(\d+)$")
EVREAL_FRAME_PATTERN = re.compile(r"^frame_(\d+)$")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Evaluate reconstruction methods on a DSEC subset."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(METHODS),
        help="Methods: evdiff, e2vid, hypere2vid.",
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=workspace_root / "outputs",
        help="Canonical prediction root containing <method>/<sequence>.",
    )
    parser.add_argument(
        "--evreal-output-root",
        type=Path,
        default=(
            workspace_root
            / "EVREAL"
            / "outputs"
            / "std"
            / "DSEC_subset"
        ),
        help="Native EVREAL output root used as a fallback.",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=workspace_root / "data" / "DSEC_mem" / "train",
        help="Converted DSEC ground-truth root.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=workspace_root / "results" / "comparison",
        help="Combined evaluation output directory.",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=DEFAULT_SEQUENCES,
        help="DSEC sequences to evaluate.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="LPIPS device: auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--lpips-batch-size",
        type=int,
        default=4,
        help="Number of frames per LPIPS batch.",
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=0,
        help="Frames excluded from the start of each sequence.",
    )

    return parser.parse_args()


def normalize_method_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "")


def resolve_methods(requested: list[str]) -> list[str]:
    methods: list[str] = []

    for name in requested:
        key = normalize_method_name(name)

        if key not in METHODS:
            available = ", ".join(METHODS)
            raise ValueError(
                f"Unknown method '{name}'. Available methods: {available}"
            )

        if key not in methods:
            methods.append(key)

    return methods


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(requested)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"Device '{requested}' was requested, but CUDA is unavailable."
        )

    return device


def directory_has_pngs(path: Path) -> bool:
    return path.is_dir() and next(path.glob("*.png"), None) is not None


def resolve_prediction_directory(
    method: str,
    sequence: str,
    pred_root: Path,
    evreal_output_root: Path,
) -> Path:
    """
    Prefer the canonical output directory.

    If it does not exist, use the native EVREAL output directory.
    """
    spec = METHODS[method]

    canonical = (
        pred_root
        / str(spec["canonical_dir"])
        / sequence
    )

    if directory_has_pngs(canonical):
        return canonical

    candidates = [canonical]
    evreal_dir = spec["evreal_dir"]

    if evreal_dir is not None:
        native = (
            evreal_output_root
            / sequence
            / str(evreal_dir)
        )
        candidates.append(native)

        if directory_has_pngs(native):
            return native

    checked = "\n".join(f"  {path}" for path in candidates)

    raise FileNotFoundError(
        f"No predictions found for "
        f"{spec['display_name']} / {sequence}.\n"
        f"Checked:\n{checked}"
    )


def parse_frame_index(path: Path) -> int | None:
    """
    Accept both:
        0000.png
        frame_0000000000.png
    """
    for pattern in (
        NUMERIC_FRAME_PATTERN,
        EVREAL_FRAME_PATTERN,
    ):
        match = pattern.fullmatch(path.stem)

        if match:
            return int(match.group(1))

    return None


def find_predictions(sequence_dir: Path) -> dict[int, Path]:
    predictions: dict[int, Path] = {}
    ignored: list[Path] = []

    for path in sorted(sequence_dir.glob("*.png")):
        frame_index = parse_frame_index(path)

        if frame_index is None:
            ignored.append(path)
            continue

        if frame_index in predictions:
            raise ValueError(
                f"Duplicate prediction index {frame_index} "
                f"in {sequence_dir}"
            )

        predictions[frame_index] = path

    if ignored:
        print(
            f"Warning: ignored {len(ignored)} PNG file(s) "
            f"with unsupported names in {sequence_dir}"
        )

    return predictions


def validate_prediction_indices(
    method_name: str,
    sequence: str,
    predictions: dict[int, Path],
    expected_count: int,
) -> list[int]:
    expected = set(range(expected_count))
    available = set(predictions)

    missing = sorted(expected - available)
    unexpected = sorted(available - expected)

    if missing or unexpected:
        messages = [
            f"Prediction validation failed for "
            f"{method_name}/{sequence}."
        ]

        if missing:
            messages.append(
                f"Missing {len(missing)} frame(s): {missing[:10]}"
            )

        if unexpected:
            messages.append(
                f"Unexpected {len(unexpected)} frame(s): "
                f"{unexpected[:10]}"
            )

        raise ValueError("\n".join(messages))

    return sorted(expected)


def load_prediction(path: Path) -> np.ndarray:
    """Load RGB or grayscale prediction as RGB float32."""
    with Image.open(path) as image:
        rgb = np.asarray(
            image.convert("RGB"),
            dtype=np.float32,
        )

    return rgb / 255.0


def load_reference(
    images: np.ndarray,
    frame_index: int,
) -> np.ndarray:
    """Load a DSEC RGB frame as float32 in [0, 1]."""
    frame = np.asarray(images[frame_index])

    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(
            f"Reference frame {frame_index} has shape "
            f"{frame.shape}; expected H x W x 3."
        )

    if np.issubdtype(frame.dtype, np.integer):
        maximum = np.iinfo(frame.dtype).max
        return frame.astype(np.float32) / maximum

    frame = frame.astype(np.float32)

    if frame.max() > 1.0:
        frame /= 255.0

    return np.clip(frame, 0.0, 1.0)


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY,
    )

    return np.asarray(gray, dtype=np.float32)


def gray_batch_to_tensor(
    frames: list[np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    array = np.stack(frames, axis=0)

    tensor = (
        torch.from_numpy(array)
        .unsqueeze(1)
        .repeat(1, 3, 1, 1)
    )

    return tensor.to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )


def calculate_lpips(
    metric: torch.nn.Module,
    predictions: list[np.ndarray],
    references: list[np.ndarray],
    device: torch.device,
) -> list[float]:
    pred_tensor = gray_batch_to_tensor(
        predictions,
        device,
    )
    ref_tensor = gray_batch_to_tensor(
        references,
        device,
    )

    with torch.inference_mode():
        scores = metric(pred_tensor, ref_tensor)

    scores = (
        scores
        .detach()
        .float()
        .cpu()
        .reshape(-1)
    )

    if scores.numel() == len(predictions):
        return [float(value) for value in scores]

    # Some PyIQA versions reduce batches to one value.
    individual_scores: list[float] = []

    with torch.inference_mode():
        for index in range(len(predictions)):
            score = metric(
                pred_tensor[index : index + 1],
                ref_tensor[index : index + 1],
            )

            individual_scores.append(
                float(
                    score
                    .detach()
                    .float()
                    .cpu()
                    .reshape(-1)[0]
                )
            )

    return individual_scores


def evaluate_sequence(
    method: str,
    sequence: str,
    prediction_dir: Path,
    gt_root: Path,
    metric: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    skip_first: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    display_name = str(
        METHODS[method]["display_name"]
    )

    sequence_gt_dir = gt_root / sequence
    images_path = sequence_gt_dir / "images.npy"
    timestamps_path = sequence_gt_dir / "images_ts.npy"

    if not images_path.is_file():
        raise FileNotFoundError(
            f"Missing ground-truth array: {images_path}"
        )

    if not timestamps_path.is_file():
        raise FileNotFoundError(
            f"Missing timestamp array: {timestamps_path}"
        )

    images = np.load(
        images_path,
        mmap_mode="r",
    )
    timestamps = np.load(
        timestamps_path,
        mmap_mode="r",
    ).reshape(-1)

    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(
            f"{images_path} has shape {images.shape}; "
            "expected N x H x W x 3."
        )

    if len(timestamps) != len(images):
        raise ValueError(
            f"{sequence}: {len(images)} images but "
            f"{len(timestamps)} timestamps."
        )

    expected_count = len(images) - 1
    predictions = find_predictions(prediction_dir)

    frame_indices = validate_prediction_indices(
        display_name,
        sequence,
        predictions,
        expected_count,
    )

    if skip_first >= len(frame_indices):
        raise ValueError(
            f"{display_name}/{sequence}: "
            "--skip-first excludes every frame."
        )

    frame_indices = frame_indices[skip_first:]

    print()
    print(f"Method:               {display_name}")
    print(f"Sequence:             {sequence}")
    print(f"Prediction directory: {prediction_dir}")
    print(f"Expected predictions: {expected_count:,}")
    print(f"Evaluated frames:     {len(frame_indices):,}")

    records: list[dict[str, Any]] = []
    progress_name = f"{display_name}:{sequence}"

    for batch_start in tqdm(
        range(0, len(frame_indices), batch_size),
        desc=progress_name,
        unit="batch",
    ):
        batch_indices = frame_indices[
            batch_start : batch_start + batch_size
        ]

        pred_grays: list[np.ndarray] = []
        ref_grays: list[np.ndarray] = []
        batch_records: list[dict[str, Any]] = []

        for frame_index in batch_indices:
            prediction_path = predictions[frame_index]

            pred_rgb = load_prediction(
                prediction_path
            )
            ref_rgb = load_reference(
                images,
                frame_index,
            )

            if pred_rgb.shape != ref_rgb.shape:
                raise ValueError(
                    f"{display_name}/{sequence} "
                    f"frame {frame_index}: "
                    f"prediction {pred_rgb.shape} != "
                    f"reference {ref_rgb.shape}."
                )

            pred_gray = rgb_to_gray(pred_rgb)
            ref_gray = rgb_to_gray(ref_rgb)

            mse = mean_squared_error(
                ref_gray,
                pred_gray,
            )

            ssim = structural_similarity(
                ref_gray,
                pred_gray,
                gaussian_weights=True,
                sigma=1.5,
                use_sample_covariance=False,
                data_range=1.0,
            )

            pred_grays.append(pred_gray)
            ref_grays.append(ref_gray)

            batch_records.append(
                {
                    "method": display_name,
                    "sequence": sequence,
                    "frame_index": frame_index,
                    "timestamp_seconds": float(
                        timestamps[frame_index]
                    ),
                    "prediction": prediction_path.name,
                    "mse": float(mse),
                    "ssim": float(ssim),
                }
            )

        lpips_scores = calculate_lpips(
            metric,
            pred_grays,
            ref_grays,
            device,
        )

        for record, lpips_score in zip(
            batch_records,
            lpips_scores,
            strict=True,
        ):
            record["lpips"] = lpips_score
            records.append(record)

    summary = {
        "method": display_name,
        "sequence": sequence,
        "aggregation": "sequence",
        "frames": len(records),
        "mse": float(
            np.mean([row["mse"] for row in records])
        ),
        "ssim": float(
            np.mean([row["ssim"] for row in records])
        ),
        "lpips": float(
            np.mean([row["lpips"] for row in records])
        ),
    }

    return records, summary


def aggregate_method(
    method_name: str,
    records: list[dict[str, Any]],
    sequence_summaries: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    weighted = {
        "method": method_name,
        "sequence": "ALL_WEIGHTED",
        "aggregation": "frame_weighted",
        "frames": len(records),
        "mse": float(
            np.mean([row["mse"] for row in records])
        ),
        "ssim": float(
            np.mean([row["ssim"] for row in records])
        ),
        "lpips": float(
            np.mean([row["lpips"] for row in records])
        ),
    }

    macro = {
        "method": method_name,
        "sequence": "ALL_SEQUENCE_MEAN",
        "aggregation": "sequence_mean",
        "frames": len(records),
        "mse": float(
            np.mean(
                [row["mse"] for row in sequence_summaries]
            )
        ),
        "ssim": float(
            np.mean(
                [row["ssim"] for row in sequence_summaries]
            )
        ),
        "lpips": float(
            np.mean(
                [row["lpips"] for row in sequence_summaries]
            )
        ),
    }

    return weighted, macro


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    columns: list[str],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )
        writer.writeheader()

        for row in rows:
            output = dict(row)

            for key in (
                "timestamp_seconds",
                "mse",
                "ssim",
                "lpips",
            ):
                if key in output:
                    output[key] = (
                        f"{float(output[key]):.10f}"
                    )

            writer.writerow(
                {
                    column: output.get(column, "")
                    for column in columns
                }
            )


def write_comparison_markdown(
    path: Path,
    weighted_summaries: list[dict[str, Any]],
) -> None:
    lines = [
        "# DSEC Subset Comparison",
        "",
        "| Method | Frames | MSE ↓ | SSIM ↑ | LPIPS ↓ |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in weighted_summaries:
        lines.append(
            f"| {row['method']} | "
            f"{row['frames']:,} | "
            f"{row['mse']:.6f} | "
            f"{row['ssim']:.6f} | "
            f"{row['lpips']:.6f} |"
        )

    lines.extend(
        [
            "",
            (
                "All methods use identical frame indices "
                "and RGB-to-grayscale metrics."
            ),
        ]
    )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def print_comparison(
    weighted_summaries: list[dict[str, Any]],
) -> None:
    print()
    print("=" * 78)
    print("DSEC SUBSET COMPARISON")
    print("=" * 78)

    print(
        f"{'Method':<18}"
        f"{'Frames':>10}"
        f"{'MSE':>16}"
        f"{'SSIM':>16}"
        f"{'LPIPS':>16}"
    )

    print("-" * 78)

    for row in weighted_summaries:
        print(
            f"{row['method']:<18}"
            f"{row['frames']:>10,}"
            f"{row['mse']:>16.6f}"
            f"{row['ssim']:>16.6f}"
            f"{row['lpips']:>16.6f}"
        )

    print("=" * 78)


def main() -> int:
    args = parse_args()
    methods = resolve_methods(args.methods)

    if args.lpips_batch_size < 1:
        raise ValueError(
            "--lpips-batch-size must be at least 1."
        )

    if args.skip_first < 0:
        raise ValueError(
            "--skip-first cannot be negative."
        )

    pred_root = args.pred_root.expanduser().resolve()
    evreal_output_root = (
        args.evreal_output_root
        .expanduser()
        .resolve()
    )
    gt_root = args.gt_root.expanduser().resolve()
    results_dir = (
        args.results_dir
        .expanduser()
        .resolve()
    )

    if not gt_root.is_dir():
        raise FileNotFoundError(
            f"Ground-truth root does not exist: {gt_root}"
        )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = resolve_device(args.device)

    print("=" * 78)
    print("UNIFIED DSEC RECONSTRUCTION EVALUATION")
    print("=" * 78)

    print(
        "Methods:           "
        + ", ".join(
            str(METHODS[key]["display_name"])
            for key in methods
        )
    )

    print(f"Ground truth:      {gt_root}")
    print(f"Results:           {results_dir}")
    print(f"LPIPS device:      {device}")
    print(
        f"LPIPS batch size:  "
        f"{args.lpips_batch_size}"
    )
    print(f"Skipped frames:    {args.skip_first}")
    print("Color protocol:    RGB to grayscale")
    print(
        "Frame mapping:     prediction i -> images[i]"
    )

    print()
    print("Loading LPIPS metric...")

    lpips_metric = (
        pyiqa
        .create_metric("lpips")
        .to(device)
    )
    lpips_metric.eval()

    print("LPIPS metric loaded.")

    all_records: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    weighted_summaries: list[dict[str, Any]] = []

    prediction_sources: dict[
        str,
        dict[str, str],
    ] = {}

    for method in methods:
        display_name = str(
            METHODS[method]["display_name"]
        )

        method_records: list[
            dict[str, Any]
        ] = []

        method_sequence_summaries: list[
            dict[str, Any]
        ] = []

        prediction_sources[display_name] = {}

        for sequence in args.sequences:
            prediction_dir = (
                resolve_prediction_directory(
                    method,
                    sequence,
                    pred_root,
                    evreal_output_root,
                )
            )

            prediction_sources[
                display_name
            ][sequence] = str(prediction_dir)

            records, summary = evaluate_sequence(
                method=method,
                sequence=sequence,
                prediction_dir=prediction_dir,
                gt_root=gt_root,
                metric=lpips_metric,
                device=device,
                batch_size=args.lpips_batch_size,
                skip_first=args.skip_first,
            )

            method_records.extend(records)
            method_sequence_summaries.append(
                summary
            )

        weighted, macro = aggregate_method(
            display_name,
            method_records,
            method_sequence_summaries,
        )

        all_records.extend(method_records)

        all_summaries.extend(
            method_sequence_summaries
            + [weighted, macro]
        )

        weighted_summaries.append(weighted)

    per_frame_path = (
        results_dir / "per_frame.csv"
    )
    summary_path = (
        results_dir / "summary.csv"
    )
    comparison_path = (
        results_dir / "comparison.csv"
    )
    markdown_path = (
        results_dir / "comparison.md"
    )
    json_path = (
        results_dir / "summary.json"
    )

    write_csv(
        per_frame_path,
        all_records,
        [
            "method",
            "sequence",
            "frame_index",
            "timestamp_seconds",
            "prediction",
            "mse",
            "ssim",
            "lpips",
        ],
    )

    write_csv(
        summary_path,
        all_summaries,
        [
            "method",
            "sequence",
            "aggregation",
            "frames",
            "mse",
            "ssim",
            "lpips",
        ],
    )

    write_csv(
        comparison_path,
        weighted_summaries,
        [
            "method",
            "frames",
            "mse",
            "ssim",
            "lpips",
        ],
    )

    write_comparison_markdown(
        markdown_path,
        weighted_summaries,
    )

    report = {
        "created_at_utc": (
            datetime
            .now(timezone.utc)
            .isoformat()
        ),
        "protocol": {
            "dataset": "DSEC",
            "sequences": args.sequences,
            "frame_mapping": (
                "prediction index i maps to images[i]"
            ),
            "expected_predictions": (
                "number of images minus one"
            ),
            "color_mode": (
                "RGB converted to grayscale"
            ),
            "grayscale_lpips_channels": 3,
            "mse": (
                "skimage.metrics.mean_squared_error"
            ),
            "ssim": {
                "implementation": (
                    "skimage.metrics."
                    "structural_similarity"
                ),
                "gaussian_weights": True,
                "sigma": 1.5,
                "use_sample_covariance": False,
                "data_range": 1.0,
            },
            "lpips": (
                "pyiqa.create_metric('lpips')"
            ),
            "skip_first": args.skip_first,
            "lpips_batch_size": (
                args.lpips_batch_size
            ),
            "device": str(device),
        },
        "prediction_sources": prediction_sources,
        "sequence_results": [
            row
            for row in all_summaries
            if row["aggregation"] == "sequence"
        ],
        "aggregate_frame_weighted": (
            weighted_summaries
        ),
        "aggregate_sequence_mean": [
            row
            for row in all_summaries
            if row["aggregation"]
            == "sequence_mean"
        ],
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    print_comparison(weighted_summaries)

    print()
    print("Results written to:")

    for path in (
        per_frame_path,
        summary_path,
        comparison_path,
        markdown_path,
        json_path,
    ):
        print(f"  {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())