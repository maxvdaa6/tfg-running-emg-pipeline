"""
06_imu_pipeline.py

Computes joint kinematics metrics from Xsens IMU data per condition.
Analysis window: frames 1200-3000 (seconds 20-50 at 60 Hz), matching
the EMG analysis window.

Joints analysed:
  - Hip flexion/extension and abduction/adduction (right and left)
  - Knee flexion/extension (right and left)
  - Ankle dorsiflexion/plantarflexion (right and left)

Metrics per signal: mean, SD, peak p2, peak min, ROM (p95-p5)

Symmetry index (SI) per joint:
  SI = (Right - Left) / (0.5 * (|Right| + |Left|)) * 100
  Positive SI = right dominant, negative = left dominant

Requires:
  config/garmin_annotations.csv
  imu_csv/<participant>/trial-XXX_joint_angles.csv

Output:
  results/imu_metrics.csv

Usage:
    python 06_imu_pipeline.py
"""

import os
import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════
FRAME_START = 1200   # 20 s × 60 Hz
FRAME_END   = 3000   # 50 s x 60 Hz  (1800 samples = 30 s usable)

PARTICIPANTS = ["p2", "p1", "p3", "p4"]

# Trials excluded from IMU analysis
# trial-001 (walking baseline 5km/h) excluded because:
#   - Walking is biomechanically different from running (non-periodic)
#   - Generates artificial asymmetries due to step variability at session start
#   - The research question focuses on running (trials 002-010)
IMU_EXCLUSIONS = {"001"}

# Right/left pairs per joint
JOINTS = {
    "hip_abd":    ("R_hip_abd",    "L_hip_abd"),
    "hip_flex":   ("R_hip_flex",   "L_hip_flex"),
    "knee_flex":  ("R_knee_flex",  "L_knee_flex"),
    "ankle_dors": ("R_ankle_dors", "L_ankle_dors"),
}

# ══════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
IMU_DIR   = os.path.join(BASE_DIR, "imu_csv")
CFG_DIR   = os.path.join(BASE_DIR, "config")
OUT_DIR   = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# DEMOGRAPHICS
# ══════════════════════════════════════════════════════════════════
demo = pd.read_csv(os.path.join(CFG_DIR, "participants_demographics.csv"))
demo = demo.set_index("participant")

# ══════════════════════════════════════════════════════════════════
# FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def compute_metrics(series: pd.Series) -> dict:
    """
    Calcula les mètriques biomecàniques d'una sèrie d'angles.

    ROM principal: p95 - p5 (robust a valors extrems puntuals i artefactes IMU).
    Els pics absoluts es guarden com a variables secundàries.
    """
    vals = series.dropna()
    if len(vals) == 0:
        return dict(mean=None, sd=None, peak_max=None, peak_min=None, ROM=None)
    p5  = float(vals.quantile(0.05))
    p95 = float(vals.quantile(0.95))
    return {
        "mean":     round(float(vals.mean()), 3),
        "sd":       round(float(vals.std()),  3),
        "peak_max": round(float(vals.max()),  3),   # absolute peak (secondary)
        "peak_min": round(float(vals.min()),  3),   # absolute peak (secondary)
        "ROM":      round(p95 - p5,           3),   # main ROM metric: p95-p5
    }

def symmetry_index(r_val, l_val) -> float | None:
    """
    Índex d'asimetria (SI) entre dreta i esquerra.
    Usa valors absoluts al denominador per evitar divisió per zero
    quan els dos valors tenen signes oposats.
    Retorna None si no es pot calcular.
    """
    denom = 0.5 * (abs(r_val) + abs(l_val))
    if denom < 1e-9:
        return None
    return round((r_val - l_val) / denom * 100, 2)

# ══════════════════════════════════════════════════════════════════
# Load config (trial -> condition)
# ══════════════════════════════════════════════════════════════════
ann = pd.read_csv(os.path.join(CFG_DIR, "garmin_annotations.csv"))
ann["trial"] = ann["trial"].astype(str).str.zfill(3)
ann["condition"] = ann["condition"].str.strip()

# ══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════
all_rows = []

for participant in PARTICIPANTS:
    p_dir = os.path.join(IMU_DIR, participant)

    if not os.path.isdir(p_dir):
        print(f"[SKIP] {participant}: carpeta imu_csv/{participant} no trobada")
        continue

    p_ann = ann[ann.participant == participant]
    if len(p_ann) == 0:
        print(f"[SKIP] {participant}: sense anotacions")
        continue

    print(f"\n{'='*55}")
    print(f"  {participant.upper()}")
    print(f"{'='*55}")

    for _, arow in p_ann.iterrows():
        trial     = arow["trial"]          # '001', '002', ...
        condition = arow["condition"]      # '5km/h_0%', etc.

        if trial in IMU_EXCLUSIONS:
            print(f"  [SKIP] trial-{trial} ({condition}) — exclòs de l'anàlisi IMU")
            continue

        csv_path = os.path.join(p_dir, f"trial-{trial}_joint_angles.csv")
        if not os.path.exists(csv_path):
            print(f"  [SKIP] trial-{trial}: CSV no trobat")
            continue

        df  = pd.read_csv(csv_path)
        win = df[(df["Frame"] >= FRAME_START) & (df["Frame"] < FRAME_END)]

        if len(win) < 100:
            print(f"  [WARN] trial-{trial}: finestra massa curta ({len(win)} frames)")
            continue

        # Participant demographic data
        d = demo.loc[participant] if participant in demo.index else {}
        row = {
            "participant":   participant,
            "trial":         trial,
            "condition":     condition,
            "n_frames":      len(win),
            "sex":           d.get("sex",          "?"),
            "age":           d.get("age",           None),
            "height_m":      d.get("height_m",      None),
            "weight_kg":     d.get("weight_kg",     None),
            "dominant_leg":  d.get("dominant_leg",  "unknown"),
            "age_group":     d.get("age_group",     "?"),
        }

        # Metrics per joint and side
        for joint, (r_col, l_col) in JOINTS.items():
            for side, col in [("R", r_col), ("L", l_col)]:
                m = compute_metrics(win[col])
                for metric_name, val in m.items():
                    row[f"{side}_{joint}_{metric_name}"] = val

            # Symmetry index Right vs Left
            for metric_name in ["mean", "sd", "peak_max", "peak_min", "ROM"]:
                r_val = row.get(f"R_{joint}_{metric_name}")
                l_val = row.get(f"L_{joint}_{metric_name}")
                if r_val is not None and l_val is not None:
                    row[f"SI_{joint}_{metric_name}"] = symmetry_index(r_val, l_val)
                else:
                    row[f"SI_{joint}_{metric_name}"] = None

            # Note: SI always computed as Right vs Left (standard)
            # Dominant leg stored in demographics for future reference

        all_rows.append(row)

        # Terminal summary (ROM and SI_ROM per joint)
        print(f"  trial-{trial} ({condition:<14})  "
              + "  ".join(
                  f"{j}: ROM D={row[f'R_{j}_ROM']:.1f}° "
                  f"E={row[f'L_{j}_ROM']:.1f}° "
                  f"SI={row[f'SI_{j}_ROM']:+.1f}%"
                  for j in JOINTS
              ))

# ══════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════
results  = pd.DataFrame(all_rows)
out_path = os.path.join(OUT_DIR, "imu_metrics.csv")
results.to_csv(out_path, index=False)

print(f"\n{'='*55}")
print(f"✓ Taula guardada: {out_path}")
print(f"  Files: {len(results)}  |  Columnes: {len(results.columns)}")
print(f"  Participants: {results['participant'].unique().tolist()}")
print(f"\nAra executa: python 07_imu_figures.py")
