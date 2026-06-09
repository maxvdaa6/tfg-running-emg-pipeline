# Pilot Study of Leg Muscle Activity During Treadmill Running
### TFG — Biomedical Engineering, Universitat de Barcelona
**Author:** Max van der Aa  
**Lab:** Biomechanical Engineering Laboratory, CREB, ETSEIB (UPC), Barcelona  
**Collaboration:** Myalbatross

---

## Overview

This repository contains the data processing and analysis pipeline for a pilot study on lower-limb muscle activity during treadmill running at different speeds and inclines. The study recorded simultaneous surface EMG (16 channels), full-body IMU kinematics (Xsens MVN Awinda), and running dynamics (Garmin Forerunner 255) across 10 treadmill conditions in 4 recreational runners.

**Conditions:** Walking baseline (5 km/h, 0%) + 9 running conditions (8, 10, 12 km/h × 0%, 5%, 10% incline)

---

## Pipeline

The pipeline runs as three independent tracks, one per recording system, followed by a statistics and figures stage.

| Script | Description |
|---|---|
| `00b_diagnostic_plots.py` | EMG signal quality check — generates diagnostic plots per channel and participant |
| `01_emg_pipeline.py` | EMG processing: spike removal, bandpass filter, RMS envelope, normalisation to walking baseline |
| `02_figures.py` | EMG figure generation: individual heatmaps and group speed/incline plots |
| `03_garmin_pipeline.py` | Garmin FIT file processing: extracts condition-level metrics using lap markers |
| `04_garmin_figures.py` | Garmin figure generation: heatmaps and group trend plots |
| `05_imu_convert.py` | Converts Xsens Excel exports to CSV format |
| `06_imu_pipeline.py` | IMU processing: joint ROM and symmetry index computation per condition |
| `07_imu_figures.py` | IMU figure generation: ROM heatmaps, SI heatmaps, group trend plots |
| `08_estadistica.py` | Statistical analysis: Friedman test, Kendall's W, cross-modal Spearman correlations |
| `dashboard_v2.py` | Generates the interactive HTML dashboard from processed CSV outputs |

---

## Interactive Dashboard

The file `athlete_dashboard_v2.html` is a self-contained interactive dashboard that displays EMG activation maps, joint ROM heatmaps, bilateral symmetry indices, and Garmin running metrics for all four participants. Open it in any browser — no server required.

---

## Requirements

```
pandas
numpy
scipy
matplotlib
seaborn
```

Install with:
```
pip install pandas numpy scipy matplotlib seaborn
```

---

## Data

Raw data files are not included in this repository due to participant privacy. The pipeline expects the following input files in the `results/` directory:

- `emg_normalized.csv` — output of `01_emg_pipeline.py`
- `garmin_metrics.csv` — output of `03_garmin_pipeline.py`
- `imu_metrics.csv` — output of `06_imu_pipeline.py`

---

## Citation

If you use this pipeline, please cite the associated TFG report:

> van der Aa, M. (2026). *Pilot Study of Leg Muscle Activity During Treadmill Running at Different Speeds and Inclines.* Final Degree Project, Biomedical Engineering, Universitat de Barcelona.
