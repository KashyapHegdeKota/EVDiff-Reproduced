# EvDiff Reproduction

This repository reproduces the evaluation of [EvDiff: High Quality Video with an Event Camera](https://arxiv.org/abs/2511.17492) using the authors' released pretrained model and a smaller subset of the DSEC driving dataset.

The objective is to reconstruct synchronized frames from event-camera streams and reproduce a reduced version of the paper's DSEC evaluation table using the same reconstruction metrics.

## Reproduction scope

- Dataset: DSEC training split
- Evaluation subset:
  - `zurich_city_00_a`
  - `zurich_city_02_a`
  - `zurich_city_04_b`
- Primary method: EvDiff pretrained checkpoint
- Planned baselines: E2VID and HyperE2VID
- Planned metrics: MSE, SSIM, and LPIPS

This is a subset reproduction, so the final metrics are not expected to exactly match the full DSEC results reported in the paper.

## Reproduction status

- [x] Select DSEC evaluation sequences
- [x] Create parallel DSEC downloader
- [x] Download and validate raw DSEC data
- [x] Convert DSEC to the EVREAL memory-mapped format
- [x] Download Stable Diffusion 3 Medium
- [x] Download the EvDiff checkpoint
- [x] Run EvDiff inference
- [ ] Inspect frame and timestamp alignment
- [ ] Compute MSE, SSIM, and LPIPS
- [ ] Run baseline models
- [ ] Generate the reduced reproduction table

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd EVDiff-Reproduced
```

### 2. Create the Python environment

Create and activate a virtual environment:

```bash
python3 -m venv myenv
source myenv/bin/activate
python -m pip install --upgrade pip
```

For Windows PowerShell:

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install EvDiff dependencies

Install PyTorch first using the version appropriate for the system's CUDA installation:

```bash
pip install torch torchvision
```

Install the remaining EvDiff dependencies:

```bash
pip install -r requirements.txt
```

Verify that PyTorch can access CUDA:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())

for index in range(torch.cuda.device_count()):
    print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
PY
```

### 4. Install the DSEC subset

The dataset downloader uses only the Python standard library. No additional downloader packages are required.

Confirm that the downloader is in the repository root:

```bash
ls -lh download_dsec.py
python download_dsec.py --help
```

Preview the downloads without downloading any files:

```bash
python download_dsec.py \
  --output ./data/DSEC/train \
  --workers 12 \
  --extract-workers 8 \
  --dry-run
```

Download and extract the selected DSEC sequences:

```bash
python download_dsec.py \
  --output ./data/DSEC/train \
  --workers 12 \
  --extract-workers 8
```

The downloader retrieves only the files required by the EvDiff DSEC converter:

- Left event-camera stream
- Rectified left RGB frames
- Left-frame exposure timestamps
- Camera calibration

The downloader:

- Downloads files concurrently
- Resumes partial downloads
- Extracts ZIP archives concurrently
- Validates the resulting directory structure
- Skips completed files when rerun

The selected subset is approximately 4.5 GB compressed. Additional storage is required for the extracted dataset and converted NumPy arrays.

#### Download one sequence first

For a smaller smoke test, download only `zurich_city_02_a`:

```bash
python download_dsec.py \
  --output ./data/DSEC/train \
  --sequences zurich_city_02_a \
  --workers 4 \
  --extract-workers 3
```

Run the full command afterward to download the remaining sequences. Previously completed files will be skipped.

#### Run the download in a persistent session

For remote systems, use `tmux` so the download continues if the SSH connection closes:

```bash
tmux new -s dsec-download
source myenv/bin/activate

python download_dsec.py \
  --output ./data/DSEC/train \
  --workers 12 \
  --extract-workers 8 2>&1 | tee dsec_download.log
```

Detach from `tmux` with `Ctrl+B`, followed by `D`.

Reconnect with:

```bash
tmux attach -t dsec-download
```

#### Resume an interrupted download

Run the same command again:

```bash
python download_dsec.py \
  --output ./data/DSEC/train \
  --workers 12 \
  --extract-workers 8
```

Partial downloads use a `.part` suffix and will be resumed when the server supports HTTP range requests.

#### Validate the dataset

Each sequence should have the following structure:

```text
data/DSEC/train/
└── zurich_city_02_a/
    ├── events/
    │   └── left/
    │       ├── events.h5
    │       └── rectify_map.h5
    ├── images/
    │   └── left/
    │       ├── exposure_timestamps.txt
    │       └── rectified/
    │           ├── 000000.png
    │           └── ...
    └── calibration/
        └── cam_to_cam.yaml
```

Verify the required files:

```bash
for sequence in zurich_city_00_a zurich_city_02_a zurich_city_04_b; do
  test -f "data/DSEC/train/$sequence/events/left/events.h5" \
    || echo "Missing events for $sequence"

  test -f "data/DSEC/train/$sequence/events/left/rectify_map.h5" \
    || echo "Missing rectify map for $sequence"

  test -f "data/DSEC/train/$sequence/images/left/exposure_timestamps.txt" \
    || echo "Missing timestamps for $sequence"

  test -f "data/DSEC/train/$sequence/calibration/cam_to_cam.yaml" \
    || echo "Missing calibration for $sequence"

  find "data/DSEC/train/$sequence/images/left/rectified" \
    -maxdepth 1 \
    -name '*.png' \
    -print \
    -quit | grep -q . || echo "Missing RGB frames for $sequence"
done
```

No output means that all required files were found.

Check dataset storage usage:

```bash
du -sh ./data/DSEC
df -h .
```

#### Remove downloaded archives

After successful extraction and validation, the retained ZIP archives can be removed:

```bash
python download_dsec.py \
  --output ./data/DSEC/train \
  --workers 12 \
  --extract-workers 8 \
  --delete-archives
```
### 5. Prepare DSEC for EvDiff

The raw DSEC sequences must be verified, converted to memory-mapped NumPy arrays, and inspected before inference.

The preparation pipeline uses four Python scripts:

| Script | Purpose |
|---|---|
| `verify_raw_dsec.py` | Verifies events, RGB frames, timestamps, and calibration |
| `convert_dsec.py` | Converts raw DSEC sequences to the EvDiff input format |
| `inspect_dsec_arrays.py` | Validates array shapes, types, timestamps, and polarity |
| `export_dsec_previews.py` | Exports converted RGB frames for visual inspection |

The complete pipeline is orchestrated by:

```text
run_dsec_preparation.sh
```

Make the runner executable:

```bash
chmod +x run_dsec_preparation.sh
```

Run the complete preparation pipeline:

```bash
./run_dsec_preparation.sh
```

The script performs the following steps:

1. Verifies all three raw DSEC sequences.
2. Converts `zurich_city_02_a` as a smoke test.
3. Validates the smoke-test arrays.
4. Exports smoke-test RGB preview frames.
5. Converts the remaining two sequences.
6. Validates all converted arrays.
7. Exports RGB previews from every sequence.

The pipeline stops immediately if verification, conversion, or inspection fails.

The complete terminal output is saved to:

```text
dsec_preparation.log
```

#### Generated dataset

The converted dataset is written to:

```text
data/DSEC_mem/train/
├── zurich_city_00_a/
├── zurich_city_02_a/
└── zurich_city_04_b/
```

Each converted sequence contains:

```text
events_ts.npy
events_xy.npy
events_p.npy
images.npy
images_ts.npy
image_event_indices.npy
```

The arrays follow the format expected by EvDiff:

| Array | Expected format |
|---|---|
| `events_ts.npy` | Event timestamps stored as `float64` |
| `events_xy.npy` | Event coordinates with shape `[N, 2]` |
| `events_p.npy` | Event polarity represented as `0` or `1` |
| `images.npy` | RGB frames stored as `uint8` |
| `images_ts.npy` | Frame timestamps |
| `image_event_indices.npy` | Event index associated with each frame |

#### RGB previews

Preview frames are written to:

```text
conversion_preview/
├── zurich_city_00_a/
├── zurich_city_02_a/
└── zurich_city_04_b/
```

Inspect these frames before running inference. A blank band near the bottom of an aligned DSEC frame can be expected because the event and RGB cameras require spatial alignment.

#### Rerun behavior

Completed conversions are skipped automatically:

```bash
./run_dsec_preparation.sh
```

Force every sequence to be converted again with:

```bash
./run_dsec_preparation.sh --force
```

Custom paths can be provided through environment variables:

```bash
RAW_ROOT=/path/to/raw/DSEC \
CONVERTED_ROOT=/path/to/DSEC_mem \
PREVIEW_ROOT=/path/to/previews \
./run_dsec_preparation.sh
```

### 6. Install pretrained model weights

To be added after dataset conversion is working.

Required model releases:

- Stable Diffusion 3 Medium
- EvDiff pretrained checkpoint

## Inference

To be added after the dataset and pretrained weights are installed.

## Evaluation

To be added after EvDiff inference has been validated.

EvDiff's RGB reconstructions will be converted to grayscale before computing MSE, SSIM, and LPIPS, following the comparison protocol described in the paper.

The reduced reproduction table will use this format:

| Method | MSE ↓ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| E2VID | TBD | TBD | TBD |
| HyperE2VID | TBD | TBD | TBD |
| EvDiff | TBD | TBD | TBD |

## References

- [EvDiff paper](https://arxiv.org/abs/2511.17492)
- [Official EvDiff repository](https://github.com/LiWeitu/EvDiff)
- [DSEC dataset](https://dsec.ifi.uzh.ch/)
- [DSEC downloads](https://dsec.ifi.uzh.ch/dsec-datasets/download/)
- [EVREAL benchmark](https://github.com/ercanburak/EVREAL)