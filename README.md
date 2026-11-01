# MWD Signal Decomposition: Two-Stage Funnel Framework

Companion code and data for the paper:

> **A consistent feature-mapping method for measurement-while-drilling data under inconsistent borehole-state conditions: enhancing data utilization and improving correlation analysis**  
> Jingqi Cui, Shunchuan Wu, Keqing Li  
> *[Journal name, Year]* · DOI: [pending]

## Overview

Measurement-while-drilling (MWD) signals are jointly modulated by rock properties and borehole state (device configuration, control mode, and drilling parameters). This makes cross-condition comparison and model transfer difficult. This repository provides the implementation of the **two-stage funnel-shaped decomposition framework** that separates MWD signals into:

- **Common Component (CC)**: parameter-insensitive part, converges across boreholes of the same lithology
- **Differential Component (DiC)**: parameter-specific part, diverges across boreholes

The decomposition enables cross-state feature alignment without requiring labeled rock data.

### Method Overview

![Overall method process](figs/Overall%20method%20process.png)

## Web Platform

### 🌐 <https://mwd.drcui.top/>

An interactive browser-based platform for MWD signal analysis, decomposition visualization, and lithology prediction without writing code.

![MWD Web Platform](figs/WEB.png)

Key features:

- Upload borehole `.xlsx` files and run the two-stage decomposition interactively
- Visualize CC / DiC separation per signal and per borehole pair
- Inspect amplitude retention rates and cross-hole coherence metrics
- Run lithology classification and view per-group prediction results

## Supplementary Appendices

Six supplementary appendices referenced in the paper are provided in [`appendices/`](appendices/). Each is a self-contained PDF with its own figures and tables (39 pages in total).

| # | Appendix | Contents |
|---|----------|----------|
| 1 | [Baseline comparison of WPD against EMD, EEMD and VMD](appendices/Appendix%20-%20Baseline%20comparison%20of%20WPD%20against%20EMD%2C%20EEMD%20and%20VMD.pdf) | The three alternatives substituted for WPD in the first stage, with everything else held fixed; single case and all six borehole pairs |
| 2 | [Sensitivity of WPD to the choice of wavelet basis](appendices/Appendix%20-%20Sensitivity%20of%20WPD%20to%20the%20choice%20of%20wavelet%20basis.pdf) | db4 against haar, db2, db6, db8 and sym4, from one to eight vanishing moments |
| 3 | [Robustness of the node threshold and the soft-mask parameter](appendices/Appendix%20-%20Robustness%20of%20the%20node%20threshold%20and%20the%20soft-mask%20parameter.pdf) | 1551 decompositions across nine datasets; five threshold rules, three mask modes, and σ from 0.5 to 2.0 |
| 4 | [Feature correlation and redundancy analysis](appendices/Appendix%20-%20Feature%20correlation%20and%20redundancy%20analysis.pdf) | Correlation matrix, VIF and condition number for the 10 features, with an ablation over three feature subsets |
| 5 | [Class-wise lithology-identification performance](appendices/Appendix%20-%20Class-wise%20lithology-identification%20performance.pdf) | Per-class precision, recall and F1, macro-averaged F1, balanced accuracy and confusion matrices |
| 6 | [User guide to the MWD-data analysis web platform](appendices/Appendix%20-%20User%20guide%20to%20the%20MWD-data%20analysis%20web%20platform.pdf) | Input format, the full workflow screen by screen, and a worked example that reproduces one row of Table 3 |

## Repository Structure

```
MWD_Signal_decomposition/
├── figs/
│   ├── Overall method process.png
│   └── WEB.png
├── appendices/                 # Six supplementary appendices (.pdf)
├── algorithm/
│   ├── config.py               # All parameters (input path, decomposition settings)
│   ├── step0_preprocess.py     # Signal segmentation and pair preparation
│   ├── step1_line1.py          # Two-stage decomposition (WPD + CWT-FQI)
│   └── main.py                 # Entry point: runs step0 → step1
│
├── lithology_prediction/
│   └── predict_lithology_5types.py   # 5-class lithology classification (6 classifiers)
│
└── data/
    ├── SandstoneA/             # TRD and RCT devices combined, N=34 stable segments
    │   └── curve_features_per_signal.csv
    ├── SandstoneB/             # RCT Sandstone B, N=6 stable segments
    │   └── curve_features_per_signal.csv
    ├── SandstoneC/             # RCT Sandstone C, N=12 stable segments
    │   └── curve_features_per_signal.csv
    ├── Granite/                # TRD Granite, N=16 stable segments
    │   └── curve_features_per_signal.csv
    └── Limestone/              # TRD Limestone, N=11 stable segments
        └── curve_features_per_signal.csv
```

## Data Description

Each `curve_features_per_signal.csv` contains per-signal statistical features extracted from decomposed MWD signals. No raw waveforms are included. Key columns:

| Column | Description |
|--------|-------------|
| `signal` | Signal type: `pressure` (kN, weight on bit) or `torque` (N·m) |
| `stage` | Decomposition stage: `original` (raw signal) or `final_comm` (CC) |
| `pair` | Borehole pair identifier (e.g., `Case-2RCT_1-1 vs Case-2RCT_1-2`) |
| `hole` | Borehole ID within the pair (`hole_A` / `hole_B` give the two members) |
| `mean`, `rms`, `std`, ... | Amplitude-domain features |
| `dominant_freq`, `spectral_entropy`, ... | Frequency-domain features |
| `sample_entropy` | Complexity feature |

The pairwise pairing strategy expands the usable sample count from **71** original-signal samples to **1,284** CC samples for the pressure signal (Table 8 of the paper). Pairs are formed within each lithology group, and each borehole of a pair contributes one sample.

## Requirements

```
python >= 3.9
numpy >= 1.21
pandas >= 1.3
scipy >= 1.7
PyWavelets >= 1.1
scikit-learn >= 1.0
xgboost >= 1.6
lightgbm >= 3.3
openpyxl >= 3.0      # for reading .xlsx input files
```

Install with:
```bash
pip install numpy pandas scipy PyWavelets scikit-learn xgboost lightgbm openpyxl
```

## Running the Decomposition

1. Set `INPUT_FOLDER` in `algorithm/config.py` to point to your data directory (folder of `.xlsx` files, one per borehole).
2. Each `.xlsx` file should contain MWD time series with columns including `pressure` and `torque`.
3. Run:

```bash
cd algorithm
python main.py
```

Output is written to a folder named `<data_folder>_output/` alongside the data.

## Running Lithology Classification

The feature CSVs in `data/` can be used directly to reproduce the classification results in Table 9 of the paper:

```bash
cd lithology_prediction
python predict_lithology_5types.py
```

This evaluates LDA, KNN, SVM-RBF, XGBoost, LightGBM, and RandomForest on `original` vs. `final_comm` features across the five lithology groups, using stratified 80/20 splits with 5-fold CV repeated 10 times.

**Expected results** (pressure + torque, best classifiers; mean accuracy averaged over 10 random 80/20 splits):
- Original signal → RandomForest: **94.6%**
- Common component → LightGBM: **99.5%**

## Key Configuration Parameters

In `algorithm/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INPUT_MODE` | `"mixed"` | `"folder"` / `"excel"` / `"mixed"` |
| `MIXED_MAX_SEG_LEN` | `400` | Max segment length (samples); longer segments are split |
| `WPD_WAVELET` | `"db4"` | Wavelet basis for WPD |
| `OTSU_MODE` | `"global"` | Otsu threshold applied globally or per-pair |
| `SIGMOID_K` | `1.0` | Sigmoid slope (soft-mask sharpness) |
| `COMM_BASELINE` | `True` | Subtract signal baseline before decomposition |

## Citation

If you use this code or data, please cite:

```bibtex
@article{cui2026mwd,
  title   = {A consistent feature-mapping method for measurement-while-drilling data under inconsistent borehole-state conditions: enhancing data utilization and improving correlation analysis},
  author  = {Cui, Jingqi and Wu, Shunchuan and Li, Keqing},
  journal = {[Journal name]},
  year    = {2026},
  doi     = {[pending]}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).  
When using the data, please cite the paper.
