"""
03_garmin_pipeline.py

Extracts Garmin running metrics per participant and condition.
Analysis window is seconds 10-60 of each trial (wider than EMG/IMU
to better capture heart rate, which peaks toward the end).

Metrics extracted:
  - fc_mean_bpm  : mean heart rate (bpm)
  - fc_max_bpm   : peak heart rate (bpm)
  - cadence_ppm  : cadence (steps/min, Garmin value x2)
  - gct_ms       : mean ground contact time (ms)
  - vert_osc_mm  : mean vertical oscillation (mm)

Requires:
  config/garmin_annotations.csv  (garmin_start_s per participant and trial)
  <Participant>_data/Garmin_<participant>/*_ACTIVITY_record.csv

Output:
  results/garmin_metrics.csv

Usage:
    python 03_garmin_pipeline.py
"""

import os
import glob
import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════
T_START_S  = 10    # s — start of analysis window (treadmill adaptation margin)
T_END_S    = 60    # s — end of analysis window (50s to capture peak heart rate)

PARTICIPANTS = ["p2", "p1", "p3", "p4"]

# Trials excluded due to measurement error (participant, trial_zfill3)
# -> trial is skipped and left as NaN in results CSV
EXCLUSIONS = {
    ("p3", "003"),   # Abnormally low HR — likely sensor contact error
}

# ══════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "..")          # TFG/Data/
CFG_DIR   = os.path.join(BASE_DIR, "config")
OUT_DIR   = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# FUNCTION: find record.csv for a participant
# ══════════════════════════════════════════════════════════════════
def find_record_csv(participant):
    """
    Busca *_ACTIVITY_record.csv dins de <Participant>_data/Garmin_<participant>/.
    Retorna el path si el troba, None si no.
    """
    folder = os.path.join(DATA_DIR,
                          f"{participant.capitalize()}_data",
                          f"Garmin_{participant}")
    matches = glob.glob(os.path.join(folder, "*_ACTIVITY_record.csv"))
    if matches:
        return matches[0]
    # Fallback: any CSV in the Garmin folder
    matches = glob.glob(os.path.join(folder, "*.csv"))
    if matches:
        return matches[0]
    return None

# ══════════════════════════════════════════════════════════════════
# LOAD CONFIG
# ══════════════════════════════════════════════════════════════════
annotations = pd.read_csv(os.path.join(CFG_DIR, "garmin_annotations.csv"))
annotations["trial"] = annotations["trial"].astype(str).str.zfill(3)

# ══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════
all_rows = []

for participant in PARTICIPANTS:
    record_path = find_record_csv(participant)

    if record_path is None:
        print(f"  [SKIP] {participant}: no s'ha trobat cap *_ACTIVITY_record.csv")
        continue

    print(f"\n{'='*55}")
    print(f"  {participant.upper()}  →  {os.path.basename(record_path)}")
    print(f"{'='*55}")

    garmin_df = pd.read_csv(record_path)
    ann       = annotations[annotations.participant == participant]

    if len(ann) == 0:
        print(f"  [SKIP] {participant}: sense anotacions a garmin_annotations.csv")
        continue

    for _, row in ann.iterrows():
        trial     = row["trial"]
        condition = row["condition"].strip()
        gs        = int(row["garmin_start_s"])
        w_start   = gs + T_START_S
        w_end     = gs + T_END_S

        # Exclusions due to measurement error
        if (participant, trial) in EXCLUSIONS:
            print(f"  [EXCL] trial-{trial} ({condition}) — exclòs per error de mesura")
            all_rows.append({
                "participant": participant, "trial": trial, "condition": condition,
                "fc_mean_bpm": None, "fc_max_bpm": None, "cadence_ppm": None,
                "gct_ms": None, "gct_n_valid": 0,
                "vert_osc_mm": None, "vert_osc_n_valid": 0, "n_samples": 0,
            })
            continue

        # Check that the window is within the file
        if w_end > len(garmin_df):
            print(f"  [WARN] trial-{trial}: finestra ({w_start}–{w_end}) "
                  f"fora del registre ({len(garmin_df)} files)")
            w_end = len(garmin_df)

        window = garmin_df.iloc[w_start:w_end]
        n      = len(window)

        # Heart rate
        fc_vals  = window["heart_rate"].dropna()
        fc_mean  = round(fc_vals.mean(), 1) if len(fc_vals) > 0 else None
        fc_p2   = int(fc_vals.max())        if len(fc_vals) > 0 else None

        # Cadence (x2: Garmin reports one foot, we want total steps/min)
        cad_vals = window["cadence"].dropna()
        cadence  = round(cad_vals.mean() * 2, 1) if len(cad_vals) > 0 else None

        # Ground Contact Time
        gct_vals    = window["stance_time"].dropna()
        gct_mean    = round(gct_vals.mean(), 1) if len(gct_vals) > 0 else None
        gct_n_valid = len(gct_vals)

        # Vertical oscillation
        vo_vals     = window["vertical_oscillation"].dropna()
        vo_mean     = round(vo_vals.mean(), 1) if len(vo_vals) > 0 else None
        vo_n_valid  = len(vo_vals)

        all_rows.append({
            "participant":      participant,
            "trial":            trial,
            "condition":        condition,
            "fc_mean_bpm":      fc_mean,
            "fc_max_bpm":       fc_p2,
            "cadence_ppm":      cadence,
            "gct_ms":           gct_mean,
            "gct_n_valid":      gct_n_valid,
            "vert_osc_mm":      vo_mean,
            "vert_osc_n_valid": vo_n_valid,
            "n_samples":        n,
        })

        print(f"  trial-{trial} ({condition:<14})  "
              f"FC={fc_mean:.1f}/{fc_p2} bpm  "
              f"Cad={cadence:.1f} ppm  "
              f"GCT={gct_mean if gct_mean else 'N/A'} ms  "
              f"VO={vo_mean if vo_mean else 'N/A'} mm")

# ══════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════
results   = pd.DataFrame(all_rows)
out_path  = os.path.join(OUT_DIR, "garmin_metrics.csv")
results.to_csv(out_path, index=False)

print(f"\n{'='*55}")
print(f"✓ Taula guardada: {out_path}")
print(f"  Files totals:   {len(results)}")
print(f"  Participants:   {results['participant'].unique().tolist()}")
print(f"\nAra executa: python 04_garmin_figures.py")
