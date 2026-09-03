# EvDiff Reproduction

Reduced reproduction of [EvDiff: Event-Based Video Reconstruction using One-Step Diffusion Models](https://arxiv.org/abs/2511.17492) using pretrained EvDiff weights and three DSEC driving sequences.

This reproduction evaluates:

- EvDiff
- E2VID
- HyperE2VID

Metrics:

- MSE, lower is better
- SSIM, higher is better
- LPIPS, lower is better

All full-reference metrics are calculated after converting the reconstructed and reference RGB frames to grayscale.

## Reproduction status

- [x] Download selected DSEC sequences
- [x] Verify raw DSEC files
- [x] Convert DSEC to EVREAL-compatible memmap format
- [x] Inspect converted arrays
- [x] Export RGB previews
- [x] Download EvDiff and Stable Diffusion 3 checkpoints
- [x] Run EvDiff inference
- [x] Evaluate EvDiff with MSE, SSIM, and LPIPS
- [x] Install EVREAL
- [x] Run E2VID
- [x] Run HyperE2VID
- [x] Generate final comparison table

## Project structure

```text
EVDiff-Reproduced/
├── EvDiff/
├── EVREAL/
├── data/
│   ├── DSEC/
│   │   └── train/
│   └── DSEC_mem/
│       └── train/
├── outputs/
│   └── evdiff/
├── results/
│   └── evdiff/
├── conversion_preview/
├── utils/
│   ├── verify_raw_dsec.py
│   ├── convert_dsec.py
│   ├── inspect_dsec_arrays.py
│   ├── export_dsec_previews.py
│   ├── run_dsec_preparation.sh
│   └── evaluate_evdiff.py
├── download_dsec.py
└── README.md
```

## Dataset subset

The following DSEC training sequences are used:

| Sequence | Reference images | Reconstructed frames |
|---|---:|---:|
| `zurich_city_00_a` | 939 | 938 |
| `zurich_city_02_a` | 235 | 234 |
| `zurich_city_04_b` | 269 | 268 |
| **Total** | **1,443** | **1,440** |

There is one fewer reconstructed frame than reference images because each reconstruction uses the events between two consecutive reference frames.

## Installation

### 1. Clone EvDiff

From the workspace root:

```bash
git clone https://github.com/LiWeitu/EvDiff.git
```

### 2. Create the EvDiff environment

```bash
conda create -n evdiff python=3.10 -y
conda activate evdiff

cd EvDiff
pip install -r requirements.txt
cd ..
```

### 3. Download the pretrained models

Accept the Stable Diffusion 3 Medium license on Hugging Face before downloading it.

```bash
conda activate evdiff
cd EvDiff

huggingface-cli download \
  stabilityai/stable-diffusion-3-medium-diffusers \
  --local-dir ./checkpoints/sd3-medium

huggingface-cli download \
  litutu135/EvDiff \
  --local-dir ./checkpoints/evdiff

cd ..
```

Expected checkpoint directories:

```text
EvDiff/checkpoints/
├── evdiff/
└── sd3-medium/
```

### 4. Verify CUDA

```bash
conda activate evdiff

python - <<'PY'
import torch

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY
```

## Download DSEC

Download the three selected sequences:

```bash
conda activate myenv

python download_dsec.py \
  --output ./data/DSEC/train \
  --workers 12 \
  --extract-workers 8
```

For a system with 20 CPU cores, 12 download workers and 8 extraction workers provide a reasonable starting configuration.

The downloader retrieves the event stream, rectified left RGB images, timestamps, rectification map, and calibration required by EvDiff.

## Prepare DSEC

Make the preparation script executable:

```bash
chmod +x utils/run_dsec_preparation.sh
```

Run the complete preparation pipeline from any directory:

```bash
conda activate myenv
./utils/run_dsec_preparation.sh
```

The script performs:

1. Raw dataset validation
2. Smoke-test conversion of `zurich_city_02_a`
3. Converted array validation
4. RGB preview export
5. Conversion of the remaining sequences
6. Final validation and preview export

Converted data is written to:

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

The converted RGB frames have shape:

```text
N x 464 x 640 x 3
```

The blank alignment region visible near an image boundary is an expected consequence of aligning the DSEC event and RGB cameras.

## Run EvDiff inference

Activate the EvDiff environment:

```bash
conda activate evdiff
cd EvDiff
```

Run all three sequences:

```bash
CUDA_VISIBLE_DEVICES=0 python test/infer_e2vid.py \
  --checkpoint ./checkpoints/evdiff/checkpoint \
  --sd3_path ./checkpoints/sd3-medium \
  --embedding_dir ./dataset/default \
  --input_dir ../data/DSEC_mem/train \
  --output_dir ../outputs \
  --model_name evdiff \
  --sequences zurich_city_00_a,zurich_city_02_a,zurich_city_04_b
```

Expected output:

```text
outputs/evdiff/
├── zurich_city_00_a/
├── zurich_city_02_a/
└── zurich_city_04_b/
```

Verify the frame counts:

```bash
for sequence in \
  zurich_city_00_a \
  zurich_city_02_a \
  zurich_city_04_b
do
  count=$(find "../outputs/evdiff/$sequence" \
    -maxdepth 1 \
    -name '*.png' \
    | wc -l)

  echo "$sequence: $count frames"
done
```

Expected:

```text
zurich_city_00_a: 938 frames
zurich_city_02_a: 234 frames
zurich_city_04_b: 268 frames
```

## Evaluate EvDiff

Install the metric dependencies:

```bash
conda activate evdiff
pip install pyiqa scikit-image
```

Run the evaluator from the workspace root:

```bash
python utils/evaluate_evdiff.py \
  --device cuda \
  --lpips-batch-size 4
```

The first invocation downloads the AlexNet and LPIPS pretrained weights.

Evaluation outputs:

```text
results/evdiff/
├── per_frame.csv
├── summary.csv
└── summary.json
```

### Evaluation protocol

- Prediction `i` is compared with `images[i]`.
- All available reconstructed frames are evaluated.
- RGB predictions and reference images are converted to grayscale.
- Pixel values are normalized to `[0, 1]`.
- MSE uses `skimage.metrics.mean_squared_error`.
- SSIM uses Gaussian weights, sigma `1.5`, population covariance, and data range `1.0`.
- LPIPS uses the PyIQA AlexNet implementation.
- The recurrent sequence state is reset between sequences.
- `ALL_WEIGHTED` is used as the primary aggregate.

## EvDiff results

| Sequence | Frames | MSE ↓ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| `zurich_city_00_a` | 938 | 0.036225 | 0.379505 | 0.431870 |
| `zurich_city_02_a` | 234 | 0.034425 | 0.346985 | 0.435160 |
| `zurich_city_04_b` | 268 | 0.036995 | 0.357781 | 0.321861 |
| **ALL_WEIGHTED** | **1,440** | **0.036076** | **0.370178** | **0.411931** |

For reference, the paper reports the following full-DSEC EvDiff result:

| Evaluation | MSE ↓ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| EvDiff paper, full DSEC split | 0.0476 | 0.3677 | 0.4226 |
| This three-sequence subset | 0.0361 | 0.3702 | 0.4119 |

These numbers are not expected to match exactly because this experiment evaluates three selected DSEC training sequences rather than the complete official evaluation split.

## Install EVREAL baselines

EVREAL provides a unified implementation of E2VID, FireNet, E2VID+, FireNet+, SPADE-E2VID, SSL-E2VID, ET-Net, and HyperE2VID.

### 1. Clone EVREAL

From the workspace root:

```bash
git clone https://github.com/ercanburak/EVREAL.git
cd EVREAL

git lfs install
git lfs pull
```

### 2. Create a separate environment

```bash
conda create -n evreal python=3.10 -y
conda activate evreal

conda install pytorch torchvision pytorch-cuda=12.1 \
  -c pytorch \
  -c nvidia \
  -y

pip install -r requirements.txt
```

### 3. Verify pretrained models

```bash
ls -lh \
  pretrained/E2VID/model.pth \
  pretrained/HyperE2VID/model.pth
```

If either checkpoint is a small Git LFS pointer, run:

```bash
git lfs pull
```

### 4. Connect the converted DSEC dataset

From inside the `EVREAL` directory:

```bash
mkdir -p data
ln -s ../../data/DSEC_mem/train data/DSEC_subset
```

Verify the link:

```bash
ls data/DSEC_subset/zurich_city_02_a
```

### 5. Create an E2VID smoke-test configuration

```bash
python - <<'PY'
import json
from pathlib import Path

config = {
    "root_path": "data/DSEC_subset",
    "sequences": {
        "zurich_city_02_a": {}
    }
}

path = Path("config/dataset/DSEC_smoke.json")
path.write_text(json.dumps(config, indent=2) + "\n")
print(f"Created {path}")
PY
```

### 6. Run the E2VID smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
  -m E2VID \
  -c std \
  -d DSEC_smoke \
  -qm mse ssim lpips
```

Expected reconstruction directory:

```text
EVREAL/outputs/std/DSEC_smoke/zurich_city_02_a/E2VID/
```

Verify that 234 reconstructed frames were created:

```bash
find outputs/std/DSEC_smoke/zurich_city_02_a/E2VID \
  -maxdepth 1 \
  -name 'frame_*.png' \
  | wc -l
```

Expected:

```text
234
```

Do not run the complete baseline evaluation until the smoke-test images have been visually inspected and the frame count has been confirmed.

## DSEC subset results

The released EvDiff model was evaluated against E2VID and HyperE2VID on three DSEC driving sequences:

- `zurich_city_00_a`
- `zurich_city_02_a`
- `zurich_city_04_b`

A total of 1,440 reconstructed frames were evaluated. Every method used the same frames, ground-truth images, frame mapping, grayscale conversion, and metric implementations.

### Evaluation protocol

- Prediction frame `i` is compared with `images[i]`.
- The final ground-truth image is unused because `N` images define `N - 1` event intervals.
- RGB predictions and references are converted to grayscale.
- MSE uses `skimage.metrics.mean_squared_error`.
- SSIM uses Gaussian weighting with `sigma=1.5`, sample covariance disabled, and `data_range=1.0`.
- LPIPS uses PyIQA with grayscale images repeated across three channels.
- Results are frame-weighted across all three sequences.

### Results

| Method | Frames | MSE ↓ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| **EvDiff** | 1,440 | **0.036076** | **0.370178** | **0.411931** |
| HyperE2VID | 1,440 | 0.047467 | 0.341466 | 0.488303 |
| E2VID | 1,440 | 0.083338 | 0.352847 | 0.517974 |

EvDiff obtained the best result for all three metrics. Relative to HyperE2VID, EvDiff reduced MSE by approximately 24.0% and LPIPS by 15.6%, while improving SSIM by 8.4%.

These results constitute a reduced DSEC reproduction. They should not be interpreted as an exact reproduction of the paper's full DSEC test-set results.

### Run the unified evaluation

From the workspace root:

```bash
conda activate evdiff

python utils/unified_evaluation.py \
    --methods evdiff e2vid hypere2vid \
    --device cuda \
    --lpips-batch-size 4
```

The evaluator creates:
```text 
results/comparison/
├── per_frame.csv
├── summary.csv
├── comparison.csv
├── comparison.md
└── summary.json
```

## Reproducibility notes

- Use the same converted DSEC arrays for every reconstruction method.
- Use five event voxel bins.
- Use events between consecutive reference frames.
- Preserve recurrent state within a sequence.
- Reset recurrent state between sequences.
- Evaluate the same 1,440 frame indices for every method.
- Apply each baseline’s official input and output normalization.
- Use one unified metric implementation for the final comparison.
- Report the subset and sequence names with every result.
- Do not describe the subset result as an exact reproduction of the full DSEC benchmark.

## References

- [EvDiff repository](https://github.com/LiWeitu/EvDiff)
- [EvDiff paper](https://arxiv.org/abs/2511.17492)
- [DSEC dataset](https://dsec.ifi.uzh.ch/)
- [EVREAL repository](https://github.com/ercanburak/EVREAL)
- [E2VID repository](https://github.com/uzh-rpg/rpg_e2vid)
- [HyperE2VID repository](https://github.com/ercanburak/HyperE2VID)