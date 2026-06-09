"""
00b_diagnostic_plots.py

Generates diagnostic plots to check EMG signal quality before running
the main pipeline. Run this first to spot bad channels and decide which
trials to exclude.

For each participant and muscle it outputs two figures:
  - Raw signal across all 10 trials (2x5 grid), with the spike threshold
    and analysis window p4ed.
  - Preprocessed signal (filtered, clipped, rectified, RMS envelope),
    so you can compare before and after and catch any artifacts.

Usage:
    python 00b_diagnostic_plots.py               # all participants
    python 00b_diagnostic_plots.py p3 p4     # specific participants
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ══════════════════════════════════════════════════════════════════
# PARAMETERS
# ══════════════════════════════════════════════════════════════════
FS           = 2100
T_START_S    = 20
T_END_S      = 50
S_START      = T_START_S * FS
S_END        = T_END_S   * FS

SPIKE_N_RAW  = 10
SPIKE_BUF    = int(0.010 * FS)   # 10 ms
CLIP_N_FILT  = 4
RMS_WIN_SAM  = int(0.100 * FS)   # 100 ms
BP_LOW       = 20
BP_HIGH      = 500
BP_ORDER     = 4

ALL_PARTICIPANTS = ["p2", "p1", "p3", "p4"]

# ══════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_DIR   = os.path.join(BASE_DIR, "results", "emg_csv")
CFG_DIR   = os.path.join(BASE_DIR, "config")
OUT_RAW   = os.path.join(BASE_DIR, "results", "figures", "diagnostic_channels")
OUT_PRE   = os.path.join(BASE_DIR, "results", "figures", "diagnostic_preprocessed")
os.makedirs(OUT_RAW, exist_ok=True)
os.makedirs(OUT_PRE, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
channel_map = pd.read_csv(os.path.join(CFG_DIR, "channel_map.csv"))
abbrev_map  = dict(zip(channel_map["channel"], channel_map["abbreviation"]))
ch_name_map = dict(zip(channel_map["channel"], channel_map["muscle"]))

garmin = pd.read_csv(os.path.join(CFG_DIR, "garmin_annotations.csv"))
garmin["trial"] = garmin["trial"].astype(str).str.zfill(3)
garmin["condition"] = garmin["condition"].str.strip()
cond_map = {(r.participant, r.trial): r.condition
            for _, r in garmin.iterrows()}

excl_df  = pd.read_csv(os.path.join(CFG_DIR, "trial_exclusions.csv"))
excl_df["trial"] = excl_df["trial"].astype(str)
excl_set = set()
for _, r in excl_df.iterrows():
    excl_set.add((r.participant, r.channel, r.trial))

def is_excluded(participant, channel, trial):
    return ((participant, channel, "all") in excl_set or
            (participant, channel, trial) in excl_set)

# Filter (precomputed)
b_filt, a_filt = butter(BP_ORDER, [BP_LOW, BP_HIGH], btype="bandpass", fs=FS)

# ══════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════
def remove_spikes_raw(raw):
    raw  = raw.copy()
    mad  = np.median(np.abs(raw - np.median(raw)))
    thresh = SPIKE_N_RAW * 1.4826 * mad
    bad  = np.abs(raw) > thresh
    for idx in np.where(bad)[0]:
        bad[max(0, idx - SPIKE_BUF): min(len(raw), idx + SPIKE_BUF + 1)] = True
    good = ~bad
    if good.sum() > 1:
        raw = np.interp(np.arange(len(raw)), np.where(good)[0], raw[good])
    return raw, thresh

def preprocess_full(raw):
    """Retorna (env, thresh_raw, thresh_clip) per visualitzar tots els passos."""
    raw   = np.nan_to_num(raw, nan=0.0)
    raw_clean, thresh_raw = remove_spikes_raw(raw)
    filt  = filtfilt(b_filt, a_filt, raw_clean)
    mad_f = np.median(np.abs(filt - np.median(filt)))
    thresh_clip = CLIP_N_FILT * 1.4826 * mad_f
    filt  = np.clip(filt, -thresh_clip, thresh_clip)
    env   = np.sqrt(np.convolve(filt**2,
                                 np.ones(RMS_WIN_SAM) / RMS_WIN_SAM,
                                 mode="same"))
    return env, thresh_raw, thresh_clip

# ══════════════════════════════════════════════════════════════════
# Main function: generates the 2 diagnostic plots for one (participant, channel) pair
# ══════════════════════════════════════════════════════════════════
t_axis = np.linspace(T_START_S, T_END_S, S_END - S_START)  # time axis (s)

def make_diagnostic(participant, ch):
    ab       = abbrev_map[ch]
    ch_label = ch_name_map[ch]
    excl_all = is_excluded(participant, ch, "all")

    # Read all available trials
    trials_data = []
    for t in range(1, 11):
        trial_str = f"{t:03d}"
        path = os.path.join(CSV_DIR, f"{participant}_trial_{trial_str}_emg.csv")
        if not os.path.exists(path):
            continue
        df  = pd.read_csv(path)
        cond = cond_map.get((participant, trial_str), "?")
        if ch not in df.columns:
            raw = np.zeros(S_END - S_START)
        else:
            vals = df[ch].values.astype(float)
            # Use available window (may be shorter than S_END)
            raw = vals[S_START:min(S_END, len(vals))]
        raw = np.nan_to_num(raw, nan=0.0)
        trials_data.append((trial_str, cond, raw))

    if not trials_data:
        print(f"  [SKIP] {participant} {ab}: sense CSVs")
        return

    n_trials = len(trials_data)
    ncols    = 5
    nrows    = int(np.ceil(n_trials / ncols))

    # Minimum length across trials to share time axis
    min_len = min(len(r) for _, _, r in trials_data)

    # A) RAW SIGNAL
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 4, nrows * 2.5),
                              sharey=False)
    fig.suptitle(f"{participant.upper()} — {ab}  ({ch} · {ch_label})\n"
                 f"Senyal cru  |  finestra {T_START_S}–{T_END_S} s",
                 fontsize=13, fontweight="bold")

    axes_flat = axes.flatten() if nrows > 1 else np.array(axes).flatten()

    for i, (trial_str, cond, raw) in enumerate(trials_data):
        ax = axes_flat[i]
        excl = is_excluded(participant, ch, trial_str)
        raw  = raw[:min_len]
        t_ax = np.linspace(T_START_S, T_START_S + min_len / FS, min_len)

        # Spike threshold
        mad    = np.median(np.abs(raw - np.median(raw)))
        thresh = SPIKE_N_RAW * 1.4826 * mad

        ax.plot(t_ax, raw * 1e3, lw=0.5,  # mV
                color="tab:blue" if not excl else "lightgray",
                alpha=0.8)
        ax.axhline( thresh * 1e3, color="red", lw=0.8, ls="--", alpha=0.7)
        ax.axhline(-thresh * 1e3, color="red", lw=0.8, ls="--", alpha=0.7)
        ax.axhline(0,             color="k",   lw=0.4, alpha=0.3)

        std_val = raw.std()
        title   = f"t{trial_str}  {cond}"
        if excl_all or excl:
            title += "  [EXCL]"
        ax.set_title(title, fontsize=8,
                     color="gray" if (excl_all or excl) else "black")
        ax.set_xlabel("Temps (s)", fontsize=7)
        ax.set_ylabel("mV", fontsize=7)
        ax.tick_params(labelsize=7)

        # Std annotation
        ax.text(0.98, 0.95, f"std={std_val*1e3:.2f} mV",
                transform=ax.transAxes, fontsize=7,
                ha="right", va="top", color="navy")

    # Hide empty axes
    for j in range(len(trials_data), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(OUT_RAW, f"det_{ab}_{participant}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # B) PREPROCESSED SIGNAL (RMS envelope)
    fig2, axes2 = plt.subplots(nrows, ncols,
                                figsize=(ncols * 4, nrows * 2.5),
                                sharey=False)
    fig2.suptitle(f"{participant.upper()} — {ab}  ({ch} · {ch_label})\n"
                  f"Senyal preprocessat (filtre BP → clipping → RMS)  "
                  f"|  finestra {T_START_S}–{T_END_S} s",
                  fontsize=13, fontweight="bold")

    axes2_flat = axes2.flatten() if nrows > 1 else np.array(axes2).flatten()

    for i, (trial_str, cond, raw) in enumerate(trials_data):
        ax   = axes2_flat[i]
        excl = is_excluded(participant, ch, trial_str)
        raw  = raw[:min_len]
        t_ax = np.linspace(T_START_S, T_START_S + min_len / FS, min_len)

        env, thresh_raw, thresh_clip = preprocess_full(raw)
        std_env = env.std()

        ax.plot(t_ax, env * 1e6,   # µV
                lw=0.8,
                color="tab:green" if not excl else "lightgray",
                alpha=0.85)
        ax.axhline(thresh_clip * 1e6, color="orange", lw=0.8, ls="--",
                   alpha=0.7, label=f"clip ±{thresh_clip*1e6:.0f} µV")

        title = f"t{trial_str}  {cond}"
        if excl_all or excl:
            title += "  [EXCL]"
        ax.set_title(title, fontsize=8,
                     color="gray" if (excl_all or excl) else "black")
        ax.set_xlabel("Temps (s)", fontsize=7)
        ax.set_ylabel("µV", fontsize=7)
        ax.tick_params(labelsize=7)

        ax.text(0.98, 0.95, f"std={std_env*1e6:.0f} µV",
                transform=ax.transAxes, fontsize=7,
                ha="right", va="top", color="darkgreen")

    for j in range(len(trials_data), len(axes2_flat)):
        axes2_flat[j].set_visible(False)

    plt.tight_layout()
    out_path2 = os.path.join(OUT_PRE, f"pre_{ab}_{participant}.png")
    fig2.savefig(out_path2, dpi=120, bbox_inches="tight")
    plt.close(fig2)

    print(f"  ✓ {ab:8s}  →  det + pre  {'[EXCL ALL]' if excl_all else ''}")

# ══════════════════════════════════════════════════════════════════
# EXECUTION
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    participants = sys.argv[1:] if len(sys.argv) > 1 else ALL_PARTICIPANTS

    for participant in participants:
        base_path = os.path.join(CSV_DIR, f"{participant}_trial_001_emg.csv")
        if not os.path.exists(base_path):
            print(f"[SKIP] {participant}: sense dades EMG CSV")
            continue

        print(f"\n{'='*55}")
        print(f"  {participant.upper()}")
        print(f"{'='*55}")

        for _, row in channel_map.iterrows():
            make_diagnostic(participant, row["channel"])

    print(f"\n✓ Gràfics guardats a:")
    print(f"  {OUT_RAW}")
    print(f"  {OUT_PRE}")
