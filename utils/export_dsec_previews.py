#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SEQUENCES = [
    "zurich_city_00_a",
    "zurich_city_02_a",
    "zurich_city_04_b",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export RGB previews from converted DSEC data."
    )

    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("./data/DSEC_mem/train"),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./conversion_preview"),
    )

    parser.add_argument(
        "--sequences",
        nargs="+",
        default=DEFAULT_SEQUENCES,
    )

    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Number of preview frames per sequence.",
    )

    return parser.parse_args()


def choose_indices(
    number_of_images: int,
    count: int,
) -> list[int]:
    if number_of_images <= 0:
        return []

    count = min(number_of_images, max(1, count))

    return sorted(
        set(
            np.linspace(
                0,
                number_of_images - 1,
                num=count,
                dtype=int,
            ).tolist()
        )
    )


def export_sequence(
    input_root: Path,
    output_root: Path,
    sequence: str,
    count: int,
) -> bool:
    images_path = input_root / sequence / "images.npy"

    print(f"\nSequence: {sequence}")

    if not images_path.is_file():
        print(f"  [MISSING] {images_path}")
        return False

    images = np.load(images_path, mmap_mode="r")

    if images.ndim != 4 or images.shape[-1] != 3:
        print(f"  [FAIL] Invalid image shape: {images.shape}")
        return False

    sequence_output = output_root / sequence
    sequence_output.mkdir(parents=True, exist_ok=True)

    indices = choose_indices(
        number_of_images=len(images),
        count=count,
    )

    if not indices:
        print("  [FAIL] Image array is empty.")
        return False

    for index in indices:
        rgb_image = np.asarray(images[index])

        if rgb_image.dtype != np.uint8:
            rgb_image = np.clip(
                rgb_image,
                0,
                255,
            ).astype(np.uint8)

        bgr_image = cv2.cvtColor(
            rgb_image,
            cv2.COLOR_RGB2BGR,
        )

        output_path = (
            sequence_output / f"frame_{index:06d}.png"
        )

        if not cv2.imwrite(str(output_path), bgr_image):
            print(f"  [FAIL] Could not save {output_path}")
            return False

        print(f"  [SAVED] {output_path}")

    return True


def main():
    args = parse_args()

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    output_root.mkdir(parents=True, exist_ok=True)

    results = [
        export_sequence(
            input_root=input_root,
            output_root=output_root,
            sequence=sequence,
            count=args.count,
        )
        for sequence in args.sequences
    ]

    passed = sum(results)
    total = len(results)

    print(f"\nResult: {passed}/{total} sequences exported.")
    print(f"Preview directory: {output_root}")

    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())