#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np


DEFAULT_SEQUENCES = [
    "zurich_city_00_a",
    "zurich_city_02_a",
    "zurich_city_04_b",
]

ARRAY_NAMES = [
    "events_ts",
    "events_xy",
    "events_p",
    "images",
    "images_ts",
    "image_event_indices",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect converted DSEC NumPy arrays."
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("./data/DSEC_mem/train"),
        help="Root directory containing converted DSEC sequences.",
    )

    parser.add_argument(
        "--sequences",
        nargs="+",
        default=DEFAULT_SEQUENCES,
        help="Converted DSEC sequences to inspect.",
    )

    return parser.parse_args()


def format_size(path: Path) -> str:
    size = path.stat().st_size
    value = float(size)

    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"

        value /= 1024

    return f"{size} B"


def non_decreasing(
    array: np.ndarray,
    chunk_size: int = 1_000_000,
) -> bool:
    """Check ordering without loading an entire large array into memory."""

    values = array.reshape(-1)

    if len(values) < 2:
        return True

    previous_value = values[0]

    for start in range(0, len(values), chunk_size):
        chunk = np.asarray(values[start : start + chunk_size])

        if len(chunk) == 0:
            continue

        if chunk[0] < previous_value:
            return False

        if len(chunk) > 1 and np.any(chunk[1:] < chunk[:-1]):
            return False

        previous_value = chunk[-1]

    return True


def load_arrays(sequence_root: Path):
    arrays = {}

    for name in ARRAY_NAMES:
        path = sequence_root / f"{name}.npy"

        if not path.is_file():
            raise FileNotFoundError(path)

        arrays[name] = np.load(path, mmap_mode="r")

    return arrays


def print_array_information(
    sequence_root: Path,
    arrays: dict,
):
    for name in ARRAY_NAMES:
        array = arrays[name]
        path = sequence_root / f"{name}.npy"

        print(
            f"{name:22} "
            f"shape={str(array.shape):22} "
            f"dtype={str(array.dtype):10} "
            f"size={format_size(path)}"
        )


def validate_array_shapes(arrays: dict) -> list[str]:
    errors = []

    events_ts = arrays["events_ts"]
    events_xy = arrays["events_xy"]
    events_p = arrays["events_p"]
    images = arrays["images"]
    images_ts = arrays["images_ts"]
    image_event_indices = arrays["image_event_indices"]

    if events_ts.ndim != 1:
        errors.append(
            f"events_ts must be one-dimensional, got {events_ts.shape}"
        )

    if events_xy.ndim != 2 or events_xy.shape[1] != 2:
        errors.append(
            f"events_xy must have shape [N, 2], got {events_xy.shape}"
        )

    if events_p.ndim != 1:
        errors.append(
            f"events_p must be one-dimensional, got {events_p.shape}"
        )

    if images.ndim != 4 or images.shape[-1] != 3:
        errors.append(
            f"images must have shape [N, H, W, 3], got {images.shape}"
        )

    if images_ts.ndim not in {1, 2}:
        errors.append(
            f"images_ts must have one or two dimensions, got "
            f"{images_ts.shape}"
        )

    if images_ts.ndim == 2 and images_ts.shape[1] != 1:
        errors.append(
            f"Two-dimensional images_ts must have shape [N, 1], "
            f"got {images_ts.shape}"
        )

    if image_event_indices.ndim not in {1, 2}:
        errors.append(
            "image_event_indices must have one or two dimensions, "
            f"got {image_event_indices.shape}"
        )

    if (
        image_event_indices.ndim == 2
        and image_event_indices.shape[1] != 1
    ):
        errors.append(
            "Two-dimensional image_event_indices must have shape "
            f"[N, 1], got {image_event_indices.shape}"
        )

    return errors


def validate_array_lengths(arrays: dict) -> list[str]:
    errors = []

    events_ts = arrays["events_ts"]
    events_xy = arrays["events_xy"]
    events_p = arrays["events_p"]
    images = arrays["images"]
    images_ts = arrays["images_ts"].reshape(-1)
    image_event_indices = arrays["image_event_indices"].reshape(-1)

    if len(events_ts) != len(events_xy):
        errors.append(
            "events_ts and events_xy lengths do not match"
        )

    if len(events_ts) != len(events_p):
        errors.append(
            "events_ts and events_p lengths do not match"
        )

    if len(images) != len(images_ts):
        errors.append(
            "images and images_ts lengths do not match"
        )

    if len(images) != len(image_event_indices):
        errors.append(
            "images and image_event_indices lengths do not match"
        )

    return errors


def validate_array_types(arrays: dict) -> list[str]:
    errors = []

    events_ts = arrays["events_ts"]
    events_xy = arrays["events_xy"]
    events_p = arrays["events_p"]
    images = arrays["images"]
    images_ts = arrays["images_ts"]
    image_event_indices = arrays["image_event_indices"]

    if not np.issubdtype(events_ts.dtype, np.floating):
        errors.append(
            f"events_ts must use a floating-point type, "
            f"got {events_ts.dtype}"
        )

    if not np.issubdtype(events_xy.dtype, np.integer):
        errors.append(
            f"events_xy must use an integer type, "
            f"got {events_xy.dtype}"
        )

    if not np.issubdtype(events_p.dtype, np.integer):
        errors.append(
            f"events_p must use an integer type, "
            f"got {events_p.dtype}"
        )

    if images.dtype != np.uint8:
        errors.append(
            f"images must use uint8, got {images.dtype}"
        )

    if not np.issubdtype(images_ts.dtype, np.floating):
        errors.append(
            f"images_ts must use a floating-point type, "
            f"got {images_ts.dtype}"
        )

    if not np.issubdtype(
        image_event_indices.dtype,
        np.integer,
    ):
        errors.append(
            "image_event_indices must use an integer type, "
            f"got {image_event_indices.dtype}"
        )

    return errors


def validate_timestamps(arrays: dict) -> list[str]:
    errors = []

    events_ts = arrays["events_ts"].reshape(-1)
    images_ts = arrays["images_ts"].reshape(-1)
    image_event_indices = arrays["image_event_indices"].reshape(-1)

    if not non_decreasing(events_ts):
        errors.append(
            "Event timestamps are not monotonically ordered"
        )

    if not non_decreasing(images_ts):
        errors.append(
            "Image timestamps are not monotonically ordered"
        )

    if not non_decreasing(image_event_indices):
        errors.append(
            "Image event indices are not monotonically ordered"
        )

    if len(events_ts) and not np.isclose(
        float(events_ts[0]),
        0.0,
        atol=1e-6,
    ):
        errors.append(
            f"First event timestamp should be near zero, "
            f"got {float(events_ts[0]):.9f}"
        )

    if len(image_event_indices):
        minimum_index = int(image_event_indices[0])
        maximum_index = int(image_event_indices[-1])

        if minimum_index < 0:
            errors.append(
                "Image event indices contain a negative value"
            )

        if maximum_index > len(events_ts):
            errors.append(
                "An image event index exceeds the number of events"
            )

    return errors


def validate_image_resolution(arrays: dict) -> list[str]:
    errors = []
    images = arrays["images"]

    if images.ndim != 4:
        return errors

    height = images.shape[1]
    width = images.shape[2]

    if height % 16 != 0:
        errors.append(
            f"Image height {height} is not divisible by 16"
        )

    if width % 16 != 0:
        errors.append(
            f"Image width {width} is not divisible by 16"
        )

    return errors


def validate_polarities(arrays: dict) -> list[str]:
    errors = []
    events_p = arrays["events_p"].reshape(-1)

    sample_size = min(len(events_p), 1_000_000)

    if sample_size == 0:
        errors.append("Event polarity array is empty")
        return errors

    polarity_values = np.unique(
        np.asarray(events_p[:sample_size])
    )

    if not set(polarity_values.tolist()).issubset({0, 1}):
        errors.append(
            f"Unexpected event polarity values: {polarity_values}"
        )

    return errors


def print_sequence_statistics(arrays: dict):
    events_ts = arrays["events_ts"].reshape(-1)
    events_p = arrays["events_p"].reshape(-1)
    images = arrays["images"]
    images_ts = arrays["images_ts"].reshape(-1)
    image_event_indices = arrays["image_event_indices"].reshape(-1)

    print()
    print(f"Events: {len(events_ts):,}")
    print(f"Images: {len(images):,}")

    if len(events_ts):
        print(
            f"First event timestamp: "
            f"{float(events_ts[0]):.6f}"
        )
        print(
            f"Last event timestamp:  "
            f"{float(events_ts[-1]):.6f}"
        )

    if len(images_ts):
        print(
            f"First image timestamp: "
            f"{float(images_ts[0]):.6f}"
        )
        print(
            f"Last image timestamp:  "
            f"{float(images_ts[-1]):.6f}"
        )

    if len(image_event_indices):
        print(
            f"First image event index: "
            f"{int(image_event_indices[0]):,}"
        )
        print(
            f"Last image event index:  "
            f"{int(image_event_indices[-1]):,}"
        )

    if images.ndim == 4:
        print(
            f"Image resolution: "
            f"{images.shape[1]}x{images.shape[2]}"
        )
        print(f"Image channels: {images.shape[3]}")

    polarity_sample_size = min(
        len(events_p),
        1_000_000,
    )

    if polarity_sample_size:
        polarity_values = np.unique(
            np.asarray(events_p[:polarity_sample_size])
        )
        print(f"Sampled polarity values: {polarity_values}")


def inspect_sequence(
    input_root: Path,
    sequence: str,
) -> bool:
    sequence_root = input_root / sequence

    print()
    print("=" * 70)
    print(sequence)
    print("=" * 70)

    if not sequence_root.is_dir():
        print(f"[MISSING] {sequence_root}")
        return False

    try:
        arrays = load_arrays(sequence_root)
    except FileNotFoundError as error:
        print(f"[MISSING] {error}")
        return False
    except (OSError, ValueError) as error:
        print(f"[FAIL] Could not load converted arrays: {error}")
        return False

    print_array_information(
        sequence_root=sequence_root,
        arrays=arrays,
    )

    errors = []

    errors.extend(validate_array_shapes(arrays))
    errors.extend(validate_array_lengths(arrays))
    errors.extend(validate_array_types(arrays))
    errors.extend(validate_timestamps(arrays))
    errors.extend(validate_image_resolution(arrays))
    errors.extend(validate_polarities(arrays))

    print_sequence_statistics(arrays)

    if errors:
        print()
        print("[FAIL]")

        for error in errors:
            print(f"  - {error}")

        return False

    print()
    print("[PASS]")
    return True


def main():
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()

    print(f"Converted DSEC root: {input_root}")

    results = [
        inspect_sequence(
            input_root=input_root,
            sequence=sequence,
        )
        for sequence in args.sequences
    ]

    passed = sum(results)
    total = len(results)

    print()
    print(f"Result: {passed}/{total} sequences passed.")

    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())