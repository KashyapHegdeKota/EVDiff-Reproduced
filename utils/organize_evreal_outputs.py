#!/usr/bin/env python3
"""
Organize EVREAL reconstructions into the common project output structure.

Source:
    EVREAL/outputs/std/DSEC_subset/<sequence>/<EVREAL method>/frame_*.png

Destination:
    outputs/<method>/<sequence>/<index>.png

Relative symbolic links are used, so image data is not duplicated.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


DEFAULT_SEQUENCES = [
    "zurich_city_00_a",
    "zurich_city_02_a",
    "zurich_city_04_b",
]

DEFAULT_METHODS = {
    "E2VID": "e2vid",
    "HyperE2VID": "hypere2vid",
}

FRAME_PATTERN = re.compile(r"^frame_(\d+)\.png$")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    workspace_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Organize EVREAL baseline reconstructions."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=(
            workspace_root
            / "EVREAL"
            / "outputs"
            / "std"
            / "DSEC_subset"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=workspace_root / "outputs",
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=workspace_root / "data" / "DSEC_mem" / "train",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=DEFAULT_SEQUENCES,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing destination files or symbolic links.",
    )
    return parser.parse_args()


def find_source_frames(source_dir: Path) -> dict[int, Path]:
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"EVREAL output directory does not exist: {source_dir}"
        )

    frames: dict[int, Path] = {}

    for path in source_dir.glob("frame_*.png"):
        match = FRAME_PATTERN.match(path.name)

        if match is None:
            continue

        frame_index = int(match.group(1))

        if frame_index in frames:
            raise ValueError(
                f"Duplicate frame index {frame_index} in {source_dir}"
            )

        frames[frame_index] = path.resolve()

    return frames


def expected_frame_count(gt_root: Path, sequence: str) -> int:
    import numpy as np

    images_path = gt_root / sequence / "images.npy"

    if not images_path.is_file():
        raise FileNotFoundError(
            f"Missing ground-truth image array: {images_path}"
        )

    images = np.load(images_path, mmap_mode="r")

    if len(images) < 2:
        raise ValueError(
            f"{sequence} must contain at least two reference images."
        )

    return len(images) - 1


def create_relative_symlink(
    source: Path,
    destination: Path,
    force: bool,
) -> str:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink():
            current_target = destination.resolve(strict=False)

            if current_target == source:
                return "skipped"

        if not force:
            raise FileExistsError(
                f"Destination already exists: {destination}\n"
                "Use --force to replace existing output links."
            )

        if destination.is_dir():
            raise IsADirectoryError(
                f"Refusing to replace directory: {destination}"
            )

        destination.unlink()

    relative_source = os.path.relpath(
        source,
        start=destination.parent,
    )
    destination.symlink_to(relative_source)

    return "created"


def organize_method(
    source_root: Path,
    output_root: Path,
    gt_root: Path,
    sequences: list[str],
    evreal_name: str,
    destination_name: str,
    force: bool,
) -> int:
    total_created = 0
    total_skipped = 0

    print()
    print("=" * 72)
    print(f"Method: {evreal_name} -> {destination_name}")
    print("=" * 72)

    for sequence in sequences:
        source_dir = source_root / sequence / evreal_name
        destination_dir = output_root / destination_name / sequence
        destination_dir.mkdir(parents=True, exist_ok=True)

        expected_count = expected_frame_count(gt_root, sequence)
        source_frames = find_source_frames(source_dir)

        expected_indices = set(range(expected_count))
        available_indices = set(source_frames)

        missing = sorted(expected_indices - available_indices)
        unexpected = sorted(available_indices - expected_indices)

        if missing or unexpected:
            raise ValueError(
                f"{evreal_name}/{sequence} failed frame validation.\n"
                f"Expected: {expected_count}\n"
                f"Found: {len(source_frames)}\n"
                f"Missing indices: {missing[:10]}\n"
                f"Unexpected indices: {unexpected[:10]}"
            )

        sequence_created = 0
        sequence_skipped = 0

        for frame_index in range(expected_count):
            source = source_frames[frame_index]
            destination = (
                destination_dir / f"{frame_index:04d}.png"
            )

            result = create_relative_symlink(
                source,
                destination,
                force,
            )

            if result == "created":
                sequence_created += 1
            else:
                sequence_skipped += 1

        total_created += sequence_created
        total_skipped += sequence_skipped

        print(
            f"[OK] {sequence}: "
            f"{expected_count} frames, "
            f"{sequence_created} created, "
            f"{sequence_skipped} already present"
        )

    print()
    print(
        f"{destination_name}: "
        f"{total_created} links created, "
        f"{total_skipped} skipped"
    )

    return total_created + total_skipped


def main() -> int:
    args = parse_args()

    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    gt_root = args.gt_root.expanduser().resolve()

    print("=" * 72)
    print("ORGANIZING EVREAL OUTPUTS")
    print("=" * 72)
    print(f"Source:       {source_root}")
    print(f"Destination:  {output_root}")
    print(f"Ground truth: {gt_root}")
    print("Storage mode: relative symbolic links")

    totals: dict[str, int] = {}

    for evreal_name, destination_name in DEFAULT_METHODS.items():
        totals[destination_name] = organize_method(
            source_root=source_root,
            output_root=output_root,
            gt_root=gt_root,
            sequences=args.sequences,
            evreal_name=evreal_name,
            destination_name=destination_name,
            force=args.force,
        )

    print()
    print("=" * 72)
    print("ORGANIZATION COMPLETE")
    print("=" * 72)

    for method, count in totals.items():
        print(f"{method}: {count} frames")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())