"""
01_emg_pipeline.py

Main EMG processing pipeline. Takes raw EMG signals and outputs
normalised RMS activation per muscle, condition, and participant.

Processing steps per channel and trial:
  1. Spike removal (10x MAD threshold, 10ms buffer, linear interpolation)
  2. Bandpass filter (20-500 Hz, 4th order Butterworth, zero-phase)
  3. Post-filter clipping (4x MAD)
  4. Rectification and RMS envelope (100ms window)
  5. Normalisation to walking baseline (trial-001, 5 km/h 0%)

Requires:
  config/channel_map.csv
  config/garmin_annotations.csv
  config/trial_exclusions.csv

Output:
  results/emg_normalized.csv

Usage:
    python 01_emg_pipeline.py
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# ══════════════════════════════════════════════════════════════════
# PARAMETERS — edit here if needed
# ══════════════════════════════════════════════════════════════════
FS            = 2100          # Hz — EMG sampling frequency
T_START_S     = 20            # s  — start of analysis window
T_END_S       = 50            # s  — end of analysis window
RMS_WIN_MS    = 100           # ms — RMS envelope window
SPIKE_N_RAW   = 10            # MAD multiplier for spike removal (raw signal)
SPIKE_BUF_MS  = 10            # ms — buffer around removed spike
CLIP_N_FILT   = 4             # MAD multiplier for post-filter clipping
BP_LOW        = 20            # Hz — bandpass lower cutoff
BP_HIGH       = 500           # Hz — bandpass upper cutoff
BP_ORDER      = 4             # Butterworth filter order

PARTICIPANTS  = ["p2", "p1", "p3", "p4"]   # add when data available

# ══════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR  = os.path.join(BASE_DIR, "results", "emg_csv")
CFG_DIR  = os.path.join(BASE_DIR, "config")
OUT_DIR  = os.path.join(BASE_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)

# Samples in analysis window
S_START     = int(T_START_S * FS)
S_END       = int(T_END_S   * FS)
RMS_WIN_SAM = int(RMS_WIN_MS / 1000 * FS)
SPIKE_BUF   = int(SPIKE_BUF_MS / 1000 * FS)

# ══════════════════════════════════════════════════════════════════
# LOAD CONFIG
# ══════════════════════════════════════════════════════════════════
channel_map = pd.read_csv(os.path.join(CFG_DIR, "channel_map.csv"))
abbrev_map  = dict(zip(channel_map["channel"], channel_map["abbreviation"]))
channels    = channel_map["channel"].tolist()

garmin = pd.read_csv(os.path.join(CFG_DIR, "garmin_annotations.csv"))
garmin["trial"] = garmin["trial"].astype(str).str.zfill(3)
cond_map = {(r.participant, r.trial): r.condition.strip()
            for _, r in garmin.iterrows()}

excl_df  = pd.read_csv(os.path.join(CFG_DIR, "trial_exclusions.csv"))
excl_df["trial"] = excl_df["trial"].astype(str)
excl_set = set()
for _, r in excl_df.iterrows():
    excl_set.add((r.participant, r.channel, r.trial))

def is_excluded(participant, channel, trial):
    return ((participant, channel, "all") in excl_set or
            (participant, channel, trial) in excl_set)

# ══════════════════════════════════════════════════════════════════
# BANDPASS FILTER (precomputed once)
# ══════════════════════════════════════════════════════════════════
b_filt, a_filt = butter(BP_ORDER, [BP_LOW, BP_HIGH], btype="bandpass", fs=FS)

# ══════════════════════════════════════════════════════════════════
# PROCESSING FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def remove_spikes_raw(raw, n=SPIKE_N_RAW, buf=SPIKE_BUF):
    """
    Elimina pics de la senyal crua per interpolació lineal.
    Detecta mostres on |raw| > n × 1.4826 × MAD i les substitueix.
    """
    raw  = raw.copy()
    mad  = np.median(np.abs(raw - np.median(raw)))
    thresh = n * 1.4826 * mad
    bad  = np.abs(raw) > thresh
    # Expand bad zone with buffer
    for idx in np.where(bad)[0]:
        bad[max(0, idx - buf): min(len(raw), idx + buf + 1)] = True
    good = ~bad
    if good.sum() > 1:
        raw = np.interp(np.arange(len(raw)), np.where(good)[0], raw[good])
    return raw

def preprocess(raw):
    """
    Pipeline complet: spike removal → filtre → clipping → rectif. → RMS.
    Retorna l'envolupant RMS (en les mateixes unitats que l'entrada).
    """
    raw = np.nan_to_num(raw, nan=0.0)

    # 1. Spike removal on raw signal
    raw = remove_spikes_raw(raw)

    # 2. Bandpass filter 20-500 Hz
    filt = filtfilt(b_filt, a_filt, raw)

    # 3. Post-filter clipping (N=4 x MAD robust)
    mad_f = np.median(np.abs(filt - np.median(filt)))
    filt  = np.clip(filt, -CLIP_N_FILT * 1.4826 * mad_f,
                           CLIP_N_FILT * 1.4826 * mad_f)

    # 4. Rectification + RMS envelope
    env = np.sqrt(np.convolve(np.abs(filt) ** 2,
                               np.ones(RMS_WIN_SAM) / RMS_WIN_SAM,
                               mode="same"))
    return env

def extract_metrics(env):
    """Extreu mean RMS i p95 RMS de l'envolupant."""
    return float(np.mean(env)), float(np.percentile(env, 95))

# ══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════
all_rows = []

for participant in PARTICIPANTS:
    # Check if data exists for this participant
    base_path = os.path.join(CSV_DIR, f"{participant}_trial_001_emg.csv")
    if not os.path.exists(base_path):
        print(f"  [SKIP] {participant}: sense dades (executa 00_convert_to_csv.py primer)")
        continue

    print(f"\n{'='*55}")
    print(f"  {participant.upper()}")
    print(f"{'='*55}")

    # Baseline (trial-001)
    base_df  = pd.read_csv(base_path)
    baseline = {}
    for ch in channels:
        if is_excluded(participant, ch, "001") or ch not in base_df.columns:
            baseline[ch] = (None, None)
            continue
        raw = base_df[ch].values[S_START:S_END]
        if np.nanstd(raw) < 1e-8:
            baseline[ch] = (None, None)
            continue
        env = preprocess(raw)
        baseline[ch] = extract_metrics(env)
    print(f"  Baseline calculat")

    # All trials
    for trial_num in range(1, 11):
        trial_str = f"{trial_num:03d}"
        path = os.path.join(CSV_DIR, f"{participant}_trial_{trial_str}_emg.csv")
        if not os.path.exists(path):
            continue

        df        = pd.read_csv(path)
        condition = cond_map.get((participant, trial_str), "unknown")

        for ch in channels:
            ab    = abbrev_map[ch]
            bm, bp = baseline[ch]

            # Trial/channel excluded
            if is_excluded(participant, ch, trial_str):
                all_rows.append({
                    "participant": participant, "trial": trial_str,
                    "condition": condition, "muscle": ab,
                    "mean_rms_uV": None, "p95_rms_uV": None,
                    "norm_mean": None, "norm_p95": None,
                    "excluded": True,
                })
                continue

            if ch not in df.columns:
                all_rows.append({
                    "participant": participant, "trial": trial_str,
                    "condition": condition, "muscle": ab,
                    "mean_rms_uV": None, "p95_rms_uV": None,
                    "norm_mean": None, "norm_p95": None,
                    "excluded": False,
                })
                continue

            raw = df[ch].values[S_START:S_END]
            if np.nanstd(raw) < 1e-8:
                all_rows.append({
                    "participant": participant, "trial": trial_str,
                    "condition": condition, "muscle": ab,
                    "mean_rms_uV": None, "p95_rms_uV": None,
                    "norm_mean": None, "norm_p95": None,
                    "excluded": False,
                })
                continue

            env  = preprocess(raw)
            m, p = extract_metrics(env)

            all_rows.append({
                "participant":  participant,
                "trial":        trial_str,
                "condition":    condition,
                "muscle":       ab,
                "mean_rms_uV":  round(m * 1e6, 3),
                "p95_rms_uV":   round(p * 1e6, 3),
                "norm_mean":    round(m / bm * 100, 2) if bm else None,
                "norm_p95":     round(p / bp * 100, 2) if bp else None,
                "excluded":     False,
            })

        print(f"  trial-{trial_str} ({condition}) ✓")

# Save results
results = pd.DataFrame(all_rows)
out_path = os.path.join(OUT_DIR, "emg_normalized.csv")
results.to_csv(out_path, index=False)

print(f"\n{'='*55}")
print(f"✓ Taula guardada: {out_path}")
print(f"  Files totals:   {len(results)}")
print(f"  Participants:   {results['participant'].unique().tolist()}")
print(f"\nAra executa: python 02_figures.py")
