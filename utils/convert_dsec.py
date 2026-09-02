#!/usr/bin/env python3

import argparse
from pathlib import Path

from tools.dsec.convert_small_align_rgb import DSECToHQFConverter


EXPECTED_FILES = [
    "events_ts.npy",
    "events_xy.npy",
    "events_p.npy",
    "images.npy",
    "images_ts.npy",
    "image_event_indices.npy",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert raw DSEC sequences for EvDiff."
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("./data/DSEC/train"),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./data/DSEC_mem/train"),
    )

    parser.add_argument(
        "--sequences",
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--event-chunk-size",
        type=int,
        default=20_000_000,
    )

    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def conversion_complete(output_directory: Path) -> bool:
    return all(
        (output_directory / filename).is_file()
        for filename in EXPECTED_FILES
    )


def convert_sequence(
    raw_root: Path,
    output_root: Path,
    sequence: str,
    event_chunk_size: int,
    image_batch_size: int,
    force: bool,
):
    raw_directory = raw_root / sequence
    output_directory = output_root / sequence

    if not raw_directory.is_dir():
        raise FileNotFoundError(
            f"Raw sequence does not exist: {raw_directory}"
        )

    if conversion_complete(output_directory) and not force:
        print(f"[SKIP] {sequence} is already converted.")
        return

    output_directory.mkdir(parents=True, exist_ok=True)

    print(f"\n[CONVERT] {sequence}")
    print(f"Input:  {raw_directory}")
    print(f"Output: {output_directory}")

    converter = DSECToHQFConverter(
        dsec_path=str(raw_directory),
        output_path=str(output_directory),
        event_chunk_size=event_chunk_size,
        image_batch_size=image_batch_size,
    )

    converter.convert()

    missing_files = [
        filename
        for filename in EXPECTED_FILES
        if not (output_directory / filename).is_file()
    ]

    if missing_files:
        raise RuntimeError(
            f"{sequence} is missing converted files: "
            f"{', '.join(missing_files)}"
        )

    print(f"[DONE] {sequence}")


def main():
    args = parse_args()

    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    output_root.mkdir(parents=True, exist_ok=True)

    for sequence in args.sequences:
        convert_sequence(
            raw_root=raw_root,
            output_root=output_root,
            sequence=sequence,
            event_chunk_size=args.event_chunk_size,
            image_batch_size=args.image_batch_size,
            force=args.force,
        )

    print("\nConversion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())