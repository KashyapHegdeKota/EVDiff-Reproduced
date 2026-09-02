#!/usr/bin/env python3

import argparse
from pathlib import Path


DEFAULT_SEQUENCES = [
    "zurich_city_00_a",
    "zurich_city_02_a",
    "zurich_city_04_b",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify raw DSEC files required by EvDiff."
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("./data/DSEC/train"),
    )

    parser.add_argument(
        "--sequences",
        nargs="+",
        default=DEFAULT_SEQUENCES,
    )

    return parser.parse_args()


def verify_sequence(raw_root: Path, sequence: str) -> bool:
    sequence_root = raw_root / sequence

    required_files = {
        "events.h5": (
            sequence_root / "events" / "left" / "events.h5"
        ),
        "rectify_map.h5": (
            sequence_root / "events" / "left" / "rectify_map.h5"
        ),
        "exposure_timestamps.txt": (
            sequence_root
            / "images"
            / "left"
            / "exposure_timestamps.txt"
        ),
        "cam_to_cam.yaml": (
            sequence_root / "calibration" / "cam_to_cam.yaml"
        ),
    }

    image_directory = (
        sequence_root / "images" / "left" / "rectified"
    )

    print(f"\nSequence: {sequence}")

    if not sequence_root.is_dir():
        print(f"  [MISSING] {sequence_root}")
        return False

    valid = True

    for filename, path in required_files.items():
        if path.is_file():
            size_mb = path.stat().st_size / (1024**2)
            print(f"  [OK] {filename}: {size_mb:.2f} MiB")
        else:
            print(f"  [MISSING] {path}")
            valid = False

    image_count = (
        sum(1 for _ in image_directory.glob("*.png"))
        if image_directory.is_dir()
        else 0
    )

    if image_count:
        print(f"  [OK] Rectified RGB images: {image_count}")
    else:
        print(f"  [MISSING] No PNG files in {image_directory}")
        valid = False

    print(f"  [{'PASS' if valid else 'FAIL'}] {sequence}")

    return valid


def main():
    args = parse_args()
    raw_root = args.raw_root.expanduser().resolve()

    print(f"Raw DSEC root: {raw_root}")

    results = [
        verify_sequence(raw_root, sequence)
        for sequence in args.sequences
    ]

    passed = sum(results)
    total = len(results)

    print(f"\nResult: {passed}/{total} sequences passed.")

    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())