#!/usr/bin/env python3
"""Download a small DSEC training subset for EvDiff evaluation.

Uses only the Python standard library. Downloads independent files in parallel,
resumes partial files when the server supports HTTP ranges, validates ZIP CRCs
during extraction, and creates the raw directory layout expected by EvDiff's
tools/dsec/convert_small_align_rgb.py converter.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


BASE_URL = "https://download.ifi.uzh.ch/rpg/DSEC/train"
DEFAULT_SEQUENCES = (
    "zurich_city_00_a",
    "zurich_city_02_a",
    "zurich_city_04_b",
)
USER_AGENT = "dsec-subset-downloader/1.0"
BUFFER_SIZE = 4 * 1024 * 1024
PRINT_LOCK = threading.Lock()


@dataclasses.dataclass(frozen=True)
class Download:
    sequence: str
    label: str
    url: str
    local_path: Path
    extract_to: Path | None = None


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def human_size(size: int | None) -> str:
    if size is None:
        return "unknown size"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def remote_size(url: str, timeout: int) -> int | None:
    request = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = response.headers.get("Content-Length")
        return int(value) if value is not None else None


def download_file(item: Download, retries: int, timeout: int) -> Path:
    destination = item.local_path
    partial = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            expected = remote_size(item.url, timeout)
            if destination.exists():
                actual = destination.stat().st_size
                if expected is None or actual == expected:
                    log(f"[skip] {item.sequence}/{item.label} ({human_size(actual)})")
                    return destination
                log(
                    f"[redo] {item.sequence}/{item.label}: existing size "
                    f"{human_size(actual)}, expected {human_size(expected)}"
                )
                destination.replace(partial)

            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": USER_AGENT}
            if offset:
                headers["Range"] = f"bytes={offset}-"

            request = urllib.request.Request(item.url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", response.getcode())
                resumed = offset > 0 and status == 206
                if offset and not resumed:
                    offset = 0
                mode = "ab" if resumed else "wb"
                action = "resume" if resumed else "download"
                log(
                    f"[{action}] {item.sequence}/{item.label} "
                    f"from {human_size(offset)} of {human_size(expected)}"
                )
                downloaded = offset
                last_report = time.monotonic()
                with partial.open(mode) as output:
                    while True:
                        chunk = response.read(BUFFER_SIZE)
                        if not chunk:
                            break
                        output.write(chunk)
                        downloaded += len(chunk)
                        now = time.monotonic()
                        if now - last_report >= 10:
                            pct = (
                                f"{downloaded / expected * 100:5.1f}%"
                                if expected
                                else human_size(downloaded)
                            )
                            log(f"[progress] {item.sequence}/{item.label}: {pct}")
                            last_report = now

            actual = partial.stat().st_size
            if expected is not None and actual != expected:
                raise IOError(
                    f"incomplete download: got {actual} bytes, expected {expected}"
                )
            partial.replace(destination)
            log(f"[done] {item.sequence}/{item.label} ({human_size(actual)})")
            return destination
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_error = exc
            if attempt < retries:
                delay = min(2 ** (attempt - 1), 20)
                log(
                    f"[retry {attempt}/{retries}] {item.sequence}/{item.label}: "
                    f"{exc}; waiting {delay}s"
                )
                time.sleep(delay)

    raise RuntimeError(
        f"failed {item.sequence}/{item.label} after {retries} attempts: {last_error}"
    )


def safe_extract(archive: Path, destination: Path) -> None:
    """Extract a ZIP while rejecting absolute paths and path traversal."""
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"unsafe ZIP member in {archive}: {member.filename}")
            target = (destination / Path(*member_path.parts)).resolve()
            if resolved_destination not in target.parents and target != resolved_destination:
                raise ValueError(f"unsafe ZIP member in {archive}: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=BUFFER_SIZE)


def extract_archive(item: Download, force: bool) -> None:
    if item.extract_to is None:
        return
    marker = item.extract_to / f".{item.label}.complete"
    if marker.exists() and not force:
        log(f"[skip extract] {item.sequence}/{item.label}")
        return
    log(f"[extract] {item.sequence}/{item.label}")
    safe_extract(item.local_path, item.extract_to)
    marker.touch()
    log(f"[extracted] {item.sequence}/{item.label}")


def build_downloads(root: Path, sequences: list[str]) -> list[Download]:
    archive_root = root / ".archives"
    downloads: list[Download] = []
    for sequence in sequences:
        prefix = f"{BASE_URL}/{sequence}/{sequence}"
        sequence_root = root / sequence
        sequence_archive_root = archive_root / sequence
        downloads.extend(
            (
                Download(
                    sequence,
                    "events_left",
                    f"{prefix}_events_left.zip",
                    sequence_archive_root / f"{sequence}_events_left.zip",
                    sequence_root / "events" / "left",
                ),
                Download(
                    sequence,
                    "images_rectified_left",
                    f"{prefix}_images_rectified_left.zip",
                    sequence_archive_root / f"{sequence}_images_rectified_left.zip",
                    sequence_root / "images" / "left" / "rectified",
                ),
                Download(
                    sequence,
                    "calibration",
                    f"{prefix}_calibration.zip",
                    sequence_archive_root / f"{sequence}_calibration.zip",
                    sequence_root / "calibration",
                ),
                Download(
                    sequence,
                    "exposure_timestamps",
                    f"{prefix}_image_exposure_timestamps_left.txt",
                    sequence_root / "images" / "left" / "exposure_timestamps.txt",
                ),
            )
        )
    return downloads


def validate_sequence(root: Path, sequence: str) -> list[str]:
    base = root / sequence
    required = (
        base / "events" / "left" / "events.h5",
        base / "events" / "left" / "rectify_map.h5",
        base / "images" / "left" / "exposure_timestamps.txt",
        base / "calibration" / "cam_to_cam.yaml",
    )
    problems = [f"missing {path}" for path in required if not path.is_file()]
    image_dir = base / "images" / "left" / "rectified"
    if not image_dir.is_dir() or not any(image_dir.glob("*.png")):
        problems.append(f"no PNG frames found in {image_dir}")
    return problems


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and arrange a small DSEC training subset for EvDiff."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/DSEC/train"),
        help="output dataset root (default: data/DSEC/train)",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=list(DEFAULT_SEQUENCES),
        help="training sequence names",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="parallel download workers (default: 6)",
    )
    parser.add_argument(
        "--extract-workers",
        type=int,
        default=3,
        help="parallel ZIP extraction workers (default: 3)",
    )
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--download-only", action="store_true", help="do not extract ZIP files"
    )
    parser.add_argument(
        "--force-extract", action="store_true", help="extract ZIP files again"
    )
    parser.add_argument(
        "--delete-archives",
        action="store_true",
        help="delete ZIPs after successful extraction and validation",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print URLs without downloading"
    )
    args = parser.parse_args()
    if args.workers < 1 or args.extract_workers < 1 or args.retries < 1:
        parser.error("worker and retry counts must be positive")
    return args


def main() -> int:
    args = parse_args()
    root = args.output.expanduser().resolve()
    downloads = build_downloads(root, args.sequences)

    log(f"DSEC output: {root}")
    log(f"Sequences: {', '.join(args.sequences)}")
    log(f"Files: {len(downloads)}, download workers: {args.workers}")

    if args.dry_run:
        for item in downloads:
            print(f"{item.sequence:24} {item.label:24} {item.url}")
        return 0

    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_item = {
            pool.submit(download_file, item, args.retries, args.timeout): item
            for item in downloads
        }
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(f"{item.sequence}/{item.label}: {exc}")

    if failures:
        log("\nDownload failures:")
        for failure in failures:
            log(f"  - {failure}")
        log("Run the same command again to resume partial downloads.")
        return 1

    if args.download_only:
        log("All files downloaded. Extraction skipped by request.")
        return 0

    archives = [item for item in downloads if item.extract_to is not None]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.extract_workers) as pool:
        future_to_item = {
            pool.submit(extract_archive, item, args.force_extract): item
            for item in archives
        }
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(f"extract {item.sequence}/{item.label}: {exc}")

    for sequence in args.sequences:
        failures.extend(validate_sequence(root, sequence))

    if failures:
        log("\nExtraction or validation failures:")
        for failure in failures:
            log(f"  - {failure}")
        return 1

    if args.delete_archives:
        for item in archives:
            item.local_path.unlink(missing_ok=True)
        archive_root = root / ".archives"
        if archive_root.exists():
            shutil.rmtree(archive_root)
        log("Downloaded ZIP archives deleted after successful validation.")

    log("\nDSEC subset is ready for EvDiff conversion.")
    for sequence in args.sequences:
        log(f"  {root / sequence}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("\nInterrupted. Run the same command again to resume.")
        raise SystemExit(130)
