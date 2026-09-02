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
- [ ] Download and validate raw DSEC data
- [ ] Convert DSEC to the EVREAL memory-mapped format
- [ ] Download Stable Diffusion 3 Medium
- [ ] Download the EvDiff checkpoint
- [ ] Run EvDiff inference
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

### 5. Convert DSEC to the EvDiff input format

To be added after the raw DSEC dataset has been validated.

Planned paths:

```text
Raw input:         ./data/DSEC/train
Converted output:  ./data/DSEC_mem/train
Converter:         tools/dsec/convert_small_align_rgb.py
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