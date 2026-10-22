## AeroFuse

This repository provides the official implementation of **AeroFuse**, a multimodal fusion framework for hyperspectral image (HSI) and LiDAR-based remote sensing classification. It includes the model implementation, training and evaluation scripts, and related experimental configurations for reproducing the results reported in our manuscript. The repository is intended to support research transparency and reproducibility. Please refer to the provided scripts and configuration files for details on model training, evaluation, and experimental settings.
## Installation and Inference


Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

Install dependencies:

    python -m pip install -r requirements.txt

Check all weights:

    python infer.py --verify-all --device cpu

Evaluate one dataset, where DATA_ROOT contains dataset directories named Houston, Trento, MUUFL, Houston2018 and Augsburg:

    python infer.py --dataset MUUFL --data-root DATA_ROOT --device cuda:0 --batch-size 256 --output muufl_predictions.npz

The output file stores three 0-based prediction arrays named `run1`, `run2`,
and `run3`. Without `--output`, only metrics are printed. Use `--max-samples 8`
for a quick input check; subset metrics are intentionally suppressed.
Houston2018 MAT v7.3 data are streamed in batches; Augsburg HSI parts are read
one at a time in numeric order. The auxiliary MAT array is loaded into host
memory. `--report REPORT.json` writes per-run metrics without requiring an
archived results file.


