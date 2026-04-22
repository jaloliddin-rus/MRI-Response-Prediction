# MRI Response Prediction

Deep-learning prediction of diffusion-weighted MRI signals from 3D vascular
structures. Given a 9-channel volumetric description of a microvascular chunk
(binary mask, gradient, hematocrit, pO₂, sO₂, velocity magnitude, and
component velocities) plus the three acquisition parameters (b-value, small
δ, big Δ), each model predicts the 11 signal time-series corresponding to
Spin Echo + 10 diffusion-weighting gradient directions.

Five architectures are compared — `CustomRegressor`, `BasicUNet`,
`AutoEncoder`, `DenseNet169`, and `EfficientNetB4` — each trained with four
loss functions (`L1`, `MSE`, `HuberLoss`, and a composite `CustomL1Loss`) and
two optimisers (`Adam`, `AdamW`).

## Repository layout

```
MRI-Response-Prediction/
├── architectures/          # 5 PyTorch / MONAI model definitions
├── benchmarks/             # Physics vs. deep-learning timing comparison
├── src/                    # Shared TiffDataset + custom losses
│   ├── dataset.py
│   └── losses.py
├── scripts/
│   ├── train.py            # main training loop
│   ├── evaluate.py         # per-sample MSE / MAE / R² on the test split
│   ├── gen_results.py      # aggregate metrics across all runs
│   ├── analysis/           # paper-figure analyses
│   └── visualization/      # plots, 3D renders, TIFF inspection
├── splits/                 # frozen train / val / test indices (committed)
│   ├── data_list.pkl
│   ├── train_indices.pkl
│   ├── val_indices.pkl
│   └── test_indices.pkl
├── timing_results/         # signal-generation timing benchmarks (paper asset)
├── requirements.txt
├── environment.yml
└── LICENSE
```

Users who want to reproduce results should contact the authors for the
dataset (can be shared upon reasonable request) and pretrained weights (can be shared).

## Installation

**Requires**: Python 3.10, and (for training) an NVIDIA GPU with CUDA 12.6.

```bash
# 1. Create the environment
conda env create -f environment.yml
conda activate mri_journal

# 2. Install the CUDA-matched PyTorch build
pip install --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.8.0 torchvision==0.23.0

# 3. Install the remaining Python dependencies
pip install -r requirements.txt
```

CPU-only users: skip step 2, then run
`pip install torch==2.8.0 torchvision==0.23.0` from the default index before
step 3.

## Data layout

Once you obtain the dataset, the expected layout is:

```
data/
├── K3/
│   ├── chunk_0/
│   │   ├── binary.tiff
│   │   ├── gradient.tiff
│   │   ├── hct.tiff
│   │   ├── po2.tiff
│   │   ├── so2.tiff
│   │   ├── velocity.tiff
│   │   ├── vx.tiff
│   │   ├── vy.tiff
│   │   ├── vz.tiff
│   │   └── signals_journal.npy
│   ├── chunk_1/
│   └── ...
├── L6/
├── L9/
├── S4/
└── S5/
```

Each chunk is a 64³ volume. Signals are stored as a NumPy structured array
with fields `(signal, b, small_delta, big_delta, phi, theta)`.

## Quick-start

All commands assume the project root as the working directory.

### Train a model

```bash
python scripts/train.py \
    --architecture Regressor \
    --loss_fn CustomL1Loss \
    --optimizer Adam \
    --epochs 300
```

On the first run, `scripts/train.py` scans `data/` and writes
`splits/data_list.pkl` plus the train/val/test index files. Subsequent runs
reuse those files so the split is deterministic.

### Evaluate a trained model

```bash
python scripts/evaluate.py \
    --architecture Regressor \
    --loss_fn CustomL1Loss \
    --optimizer Adam
```

Looks for `models/best_{architecture}_{loss_fn}_{optimizer}.pth` and writes
per-sample metrics to `results/mse/`, `results/mae/`, and
`results/metrics/`.

### Aggregate metrics across runs

```bash
python scripts/gen_results.py
```

### Paper analyses

```bash
# Gradient-direction (φ, θ) performance breakdown
python scripts/analysis/gradient_direction_analysis.py \
    --architecture DenseNet169 --loss_fn HuberLoss --optimizer Adam

# Verify Spin Echo signals are physically invariant (not data leakage)
python scripts/analysis/verify_spin_echo_physics.py --data_dir data

# Cross-architecture Spin Echo consistency
python scripts/analysis/cross_architecture_validation.py

# Input-channel ablation
python scripts/analysis/channel_importance_analysis.py \
    --architecture DenseNet169 --loss_fn HuberLoss --optimizer Adam

# Structure-level difficulty analysis
python scripts/analysis/structure_performance_analyzer.py

# Statistical significance (Friedman + Nemenyi, bootstrap CIs)
python scripts/analysis/statistical_analysis.py
```

### Visualisations

```bash
# Per-sample prediction vs ground-truth signal plots
python scripts/visualization/plot.py

# Multi-model side-by-side comparison plots
python scripts/visualization/plot_combined.py

# 3D vascular-structure render (requires VTK)
python scripts/visualization/visualise_samples.py --test_idx 0

# MSE / MAE boxplots across architectures
python scripts/visualization/generate_mse_mae_boxplots.py

# Radar chart of overall performance
python scripts/visualization/overall_perf_vis.py
```

## Reproducibility notes

- The train/val/test split is frozen in `splits/*.pkl` (80 % / 10 % / 10 %,
  random seed 42). Delete those files to regenerate.
- `splits/data_list.pkl` stores paths relative to the project root (e.g.
  `data/K3/chunk_0/binary.tiff`), so it is portable — just drop your dataset
  into `data/` with the layout shown above.
- Signal-generation timings (physics simulator vs neural inference) live
  under `timing_results/` and were produced by
  `benchmarks/signal_gen_comp.py` + `benchmarks/deepL_comp.py`. The physics
  benchmark depends on the [VirtualMRI] package; see
  `benchmarks/signal_gen_comp.py` for its own environment requirements.

## Related publications

- **Conference paper (2024):** *Learning-Based MRI Response Predictions from OCT
  Microvascular Models to Replace Simulation-Based Frameworks.* DOI:
  [10.1007/978-3-031-66955-2_4](https://doi.org/10.1007/978-3-031-66955-2_4).
  The conference-paper code is preserved in [`conf-paper/`](conf-paper/).

- **Journal paper (in preparation):** all code at the repository root
  (`architectures/`, `src/`, `scripts/`, `benchmarks/`, `splits/`,
  `timing_results/`) corresponds to the forthcoming journal paper.

## Citation

```bibtex
@inproceedings{rustamov2024mri,
  title     = {Learning-Based MRI Response Predictions from OCT Microvascular
               Models to Replace Simulation-Based Frameworks},
  author    = {Rustamov, Jaloliddin and others},
  booktitle = {Lecture Notes in Computer Science},
  year      = {2024},
  doi       = {10.1007/978-3-031-66955-2_4}
}

@article{rustamov2026mri,
  title   = {...},
  author  = {Rustamov, Jaloliddin and others},
  journal = {...},
  year    = {2026},
  note    = {In preparation}
}
```

## License

MIT — see [LICENSE](LICENSE).
