## AeroFuse

This repository provides the official implementation of **AeroFuse**, a multimodal fusion framework for hyperspectral image (HSI) and LiDAR-based remote sensing classification. It includes the model implementation, training and evaluation scripts, and related experimental configurations for reproducing the results reported in our manuscript. The repository is intended to support research transparency and reproducibility. Please refer to the provided scripts and configuration files for details on model training, evaluation, and experimental settings.
## Installation and Inference

### 1. Install Dependencies

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

### 2. Verify Pretrained Checkpoints

Verify the integrity of all pretrained checkpoints using CPU:

```bash
python infer.py --verify-all --device cpu
```

The verification process checks the checkpoint hashes recorded in the manifest and validates the model state dictionaries.

### 3. Evaluate a Dataset

Set `DATA_ROOT` to the parent directory containing the dataset folders `Houston`, `Trento`, `MUUFL`, `Houston2018`, and `Augsburg`. To evaluate MUUFL using a GPU, run:

```bash
python infer.py \
    --dataset MUUFL \
    --data-root DATA_ROOT \
    --device cuda:0 \
    --batch-size 256 \
    --output muufl_predictions.npz
```

Replace `DATA_ROOT` with the actual path to your dataset directory.

Predicted class labels are represented as zero-based indices. If `--output` is omitted, the script prints evaluation metrics without saving predictions.

For a quick input and pipeline check, use `--max-samples 8`. In this mode, subset evaluation metrics are intentionally suppressed.

