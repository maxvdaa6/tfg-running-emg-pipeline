"""
02_figures.py

Generates all EMG figures from results/emg_normalized.csv.
Run after 01_emg_pipeline.py.

Outputs:
  - Individual heatmaps per participant (muscle x condition grid)
  - Group speed/incline trend plots (mean of individual regressions)
  - Speed and incline effect plots per muscle

Usage:
    python 02_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.signal import butter, filtfilt

# ══════════════════════════════════════════════════════════════════
# PARAMETERS (must match 01_emg_pipeline.py)
# ══════════════════════════════════════════════════════════════════
FS           = 2100
T_START_S    = 20
T_END_S      = 50
RMS_WIN_MS   = 100
SPIKE_N_RAW  = 10
SPIKE_BUF_MS = 10
CLIP_N_FILT  = 4
BP_LOW, BP_HIGH, BP_ORDER = 20, 500, 4

S_START     = int(T_START_S * FS)
S_END       = int(T_END_S   * FS)
RMS_WIN_SAM = int(RMS_WIN_MS / 1000 * FS)
SPIKE_BUF   = int(SPIKE_BUF_MS / 1000 * FS)
DS          = 10   # downsampling factor for visualisation

# ══════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_DIR    = os.path.join(BASE_DIR, "results", "emg_csv")
CFG_DIR    = os.path.join(BASE_DIR, "config")
FIG_DIR    = os.path.join(BASE_DIR, "results", "figures")
DIAG_DIR   = os.path.join(FIG_DIR, "diagnostic_preprocessed")
COMP_DIR   = os.path.join(FIG_DIR, "comparatives")
for d in [DIAG_DIR, COMP_DIR]:
    os.makedirs(d, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# MUSCLE AND CONDITION ORDER AND LABELS
# ══════════════════════════════════════════════════════════════════
MUSCLES_R = ["TA_r", "MG_r", "SO_r", "VL_r", "BF_r", "ST_r", "GM_r"]
MUSCLES_L = ["TA_l", "MG_l", "SO_l", "VL_l", "RF_l", "BF_l", "ST_l", "GM_l"]
MUSCLE_ORDER = MUSCLES_R + MUSCLES_L

MUSCLE_LABELS = {
    "TA_r": "Tib. Anterior R",  "MG_r": "Med. Gastrocn. R",
    "SO_r": "Soleus R",          "VL_r": "Vastus Lat. R",
    "BF_r": "Biceps Fem. R",     "ST_r": "Semitendinós R",
    "GM_r": "Glut. Medius R",
    "TA_l": "Tib. Anterior L",  "MG_l": "Med. Gastrocn. L",
    "SO_l": "Soleus L",          "VL_l": "Vastus Lat. L",
    "RF_l": "Rectus Fem. L",     "BF_l": "Biceps Fem. L",
    "ST_l": "Semitendinós L",    "GM_l": "Glut. Medius L",
}

COND_ORDER = [
    "5km/h_0%",
    "8km/h_0%",  "8km/h_5%",  "8km/h_10%",
    "10km/h_0%", "10km/h_5%", "10km/h_10%",
    "12km/h_0%", "12km/h_5%", "12km/h_10%",
]
COND_RUN = [c for c in COND_ORDER if c != "5km/h_0%"]

SPEEDS   = [8, 10, 12]
INCLINES = [0, 5, 10]
COLORS_INC = {0: "steelblue",  5: "darkorange", 10: "crimson"}
P4ERS_INC = {0: "o", 5: "s", 10: "^"}
COLORS_SPD  = {8: "steelblue", 10: "darkorange", 12: "crimson"}
P4ERS_SPD = {8: "o", 10: "s", 12: "^"}
P_STYLE = {"p2": ("-", "P2"), "p1": ("--", "P1"),
           "p3": ("-.", "P3"), "p4": (":", "P4")}

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
# PIPELINE (replicates 01 to generate diagnostic figures)
# ══════════════════════════════════════════════════════════════════
b_filt, a_filt = butter(BP_ORDER, [BP_LOW, BP_HIGH], btype="bandpass", fs=FS)

def preprocess(raw):
    raw = np.nan_to_num(raw, nan=0.0)
    mad_r  = np.median(np.abs(raw - np.median(raw)))
    thresh = SPIKE_N_RAW * 1.4826 * mad_r
    bad    = np.abs(raw) > thresh
    for idx in np.where(bad)[0]:
        bad[max(0, idx - SPIKE_BUF): min(len(raw), idx + SPIKE_BUF + 1)] = True
    good = ~bad
    if good.sum() > 1:
        raw = np.interp(np.arange(len(raw)), np.where(good)[0], raw[good])
    filt  = filtfilt(b_filt, a_filt, raw)
    mad_f = np.median(np.abs(filt - np.median(filt)))
    filt  = np.clip(filt, -CLIP_N_FILT * 1.4826 * mad_f,
                           CLIP_N_FILT * 1.4826 * mad_f)
    return np.sqrt(np.convolve(np.abs(filt) ** 2,
                                np.ones(RMS_WIN_SAM) / RMS_WIN_SAM,
                                mode="same"))

# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — PREPROCESSED DIAGNOSTIC (per channel and participant)
# ══════════════════════════════════════════════════════════════════
def fig_diagnostic():
    print("\n── Figures de diagnòstic preprocessat ──")

    # Detect participants with data
    participants_amb_dades = []
    for p in ["p2", "p1", "p3", "p4"]:
        if os.path.exists(os.path.join(CSV_DIR, f"{p}_trial_001_emg.csv")):
            participants_amb_dades.append(p)

    # Pre-load CSVs
    raw_data = {}
    for p in participants_amb_dades:
        raw_data[p] = {}
        for t in range(1, 11):
            t_str = f"{t:03d}"
            path = os.path.join(CSV_DIR, f"{p}_trial_{t_str}_emg.csv")
            if os.path.exists(path):
                raw_data[p][t_str] = pd.read_csv(path)

    t_ax = np.arange(S_END - S_START, step=DS) / FS

    for _, row in channel_map.iterrows():
        ch     = row["channel"]
        abbrev = row["abbreviation"]
        muscle = row["muscle"]

        for participant in participants_amb_dades:
            fig, axes = plt.subplots(5, 2, figsize=(12, 14), sharex=True)
            fig.suptitle(
                f"{ch} — {abbrev} ({muscle}) · {participant.upper()}\n"
                f"Spike removal {SPIKE_N_RAW}×MAD → filtre {BP_LOW}–{BP_HIGH}Hz "
                f"→ clipping {CLIP_N_FILT}×MAD → RMS {RMS_WIN_MS}ms",
                fontsize=10, fontweight="bold"
            )

            for t_idx, t_num in enumerate(range(1, 11)):
                t_str     = f"{t_num:03d}"
                ax        = axes[t_idx // 2][t_idx % 2]
                condition = cond_map.get((participant, t_str), "?")
                df        = raw_data[participant].get(t_str)

                if df is None:
                    ax.set_visible(False)
                    continue

                if is_excluded(participant, ch, t_str):
                    ax.set_facecolor("#f0f0f0")
                    ax.text(0.5, 0.5, "EXCLÒS", ha="center", va="center",
                            transform=ax.transAxes, color="gray",
                            fontsize=14, fontweight="bold")
                    ax.set_title(f"trial-{t_str} | {condition}",
                                 fontsize=8, color="gray")
                elif ch not in df.columns:
                    ax.text(0.5, 0.5, "sense dades", ha="center", va="center",
                            transform=ax.transAxes, color="gray", fontsize=9)
                    ax.set_title(f"trial-{t_str} | {condition}", fontsize=8)
                else:
                    raw  = df[ch].values[S_START:S_END]
                    env  = preprocess(raw)
                    env_ds   = env[::DS]
                    mean_val = np.mean(env) * 1e6
                    ax.fill_between(t_ax[:len(env_ds)], env_ds,
                                    alpha=0.3, color="forestgreen")
                    ax.plot(t_ax[:len(env_ds)], env_ds,
                            color="forestgreen", linewidth=0.8)
                    ax.axhline(np.mean(env), color="darkgreen",
                               linewidth=0.8, linestyle="--", alpha=0.7)
                    ax.set_title(f"trial-{t_str} | {condition} | "
                                 f"mean={mean_val:.0f}µV",
                                 fontsize=8)

                ax.grid(alpha=0.2)
                ax.tick_params(labelsize=7)
                if t_idx % 2 == 0:
                    ax.set_ylabel("RMS (mV)", fontsize=7)

            for ax in axes[-1]:
                ax.set_xlabel("Temps (s)", fontsize=8)

            plt.tight_layout()
            fname = os.path.join(DIAG_DIR, f"pre_{abbrev}_{participant}.png")
            plt.savefig(fname, dpi=110, bbox_inches="tight")
            plt.close()
            print(f"  ✓ {abbrev} — {participant}")


# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — HEATMAPS (per participant)
# ══════════════════════════════════════════════════════════════════
def fig_heatmaps(df):
    print("\n── Heatmaps ──")
    cmap = plt.cm.RdYlGn_r
    norm = mcolors.TwoSlopeNorm(vmin=50, vcenter=100, vp2=500)

    participants = df["participant"].unique()
    for participant in participants:
        sub   = df[(df.participant == participant) &
                   (df.condition.isin(COND_RUN))].copy()
        pivot = sub.pivot_table(index="muscle", columns="condition",
                                values="norm_mean", aggfunc="first")
        muscles_ok = [m for m in MUSCLE_ORDER if m in pivot.index]
        conds_ok   = [c for c in COND_RUN   if c in pivot.columns]
        pivot = pivot.loc[muscles_ok, conds_ok]

        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, norm=norm)

        ax.set_xticks(range(len(conds_ok)))
        ax.set_xticklabels(conds_ok, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(muscles_ok)))
        ax.set_yticklabels([MUSCLE_LABELS.get(m, m) for m in muscles_ok],
                           fontsize=9)
        ax.axhline(len(MUSCLES_R) - 0.5, color="black", linewidth=2)

        for i in range(len(muscles_ok)):
            for j in range(len(conds_ok)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    color = "white" if (val > 350 or val < 70) else "black"
                    ax.text(j, i, f"{val:.0f}%",
                            ha="center", va="center",
                            fontsize=7.5, color=color)

        plt.colorbar(im, ax=ax, label="% del baseline", shrink=0.8)
        ax.set_title(
            f"{participant.upper()} — Activació muscular normalitzada al baseline\n"
            f"(100% = caminar 5 km/h 0% | verd = menys | vermell = més)",
            fontsize=11, fontweight="bold"
        )
        plt.tight_layout()
        fname = os.path.join(COMP_DIR, f"heatmap_{participant}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ✓ heatmap_{participant}")


# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — SPEED EFFECT
# ══════════════════════════════════════════════════════════════════
def fig_efecte_velocitat(df):
    print("\n── Efecte velocitat ──")
    participants = df["participant"].unique()
    muscles_plot = [m for m in MUSCLE_ORDER if m != "RF_l"]

    n_rows = (len(muscles_plot) + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 3.5))
    fig.suptitle(
        "Efecte de la VELOCITAT per cada múscul\n"
        "(mean RMS normalitzat al baseline · línies = desnivell · — = P2 · -- = P1/P3/P4)",
        fontsize=12, fontweight="bold"
    )

    for idx, muscle in enumerate(muscles_plot):
        ax = axes[idx // 3][idx % 3] if n_rows > 1 else axes[idx % 3]
        for inc in INCLINES:
            for participant in participants:
                ls, lp = P_STYLE.get(participant, ("-", participant))
                vals = []
                for spd in SPEEDS:
                    cond_str = f"{spd}km/h_{inc}%"
                    row = df[(df.participant == participant) &
                             (df.muscle == muscle) &
                             (df.condition == cond_str)]
                    v = (row["norm_mean"].values[0]
                         if len(row) and not pd.isna(row["norm_mean"].values[0])
                         else np.nan)
                    vals.append(v)
                ax.plot(SPEEDS, vals, color=COLORS_INC[inc], linestyle=ls,
                        p4er=P4ERS_INC[inc], p4ersize=5, linewidth=1.5,
                        label=f"{inc}% - {lp}")

        ax.axhline(100, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(MUSCLE_LABELS.get(muscle, muscle), fontsize=9, fontweight="bold")
        ax.set_xlabel("Velocitat (km/h)", fontsize=8)
        ax.set_ylabel("% baseline", fontsize=8)
        ax.set_xticks(SPEEDS)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)

    # Hide empty subplots
    total = n_rows * 3
    for idx in range(len(muscles_plot), total):
        axes[idx // 3][idx % 3].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    seen = set(); h2 = []; l2 = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    fig.legend(h2, l2, loc="lower right", fontsize=8,
               title="Desnivell — Participant", ncol=3,
               bbox_to_anchor=(0.99, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fname = os.path.join(COMP_DIR, "efecte_velocitat.png")
    plt.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  ✓ efecte_velocitat")


# ══════════════════════════════════════════════════════════════════
# FIGURE 4 — INCLINE EFFECT
# ══════════════════════════════════════════════════════════════════
def fig_efecte_desnivell(df):
    print("\n── Efecte desnivell ──")
    participants = df["participant"].unique()
    muscles_plot = [m for m in MUSCLE_ORDER if m != "RF_l"]

    n_rows = (len(muscles_plot) + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 3.5))
    fig.suptitle(
        "Efecte del DESNIVELL per cada múscul\n"
        "(mean RMS normalitzat al baseline · línies = velocitat · — = P2 · -- = P1/P3/P4)",
        fontsize=12, fontweight="bold"
    )

    for idx, muscle in enumerate(muscles_plot):
        ax = axes[idx // 3][idx % 3] if n_rows > 1 else axes[idx % 3]
        for spd in SPEEDS:
            for participant in participants:
                ls, lp = P_STYLE.get(participant, ("-", participant))
                vals = []
                for inc in INCLINES:
                    cond_str = f"{spd}km/h_{inc}%"
                    row = df[(df.participant == participant) &
                             (df.muscle == muscle) &
                             (df.condition == cond_str)]
                    v = (row["norm_mean"].values[0]
                         if len(row) and not pd.isna(row["norm_mean"].values[0])
                         else np.nan)
                    vals.append(v)
                ax.plot(INCLINES, vals, color=COLORS_SPD[spd], linestyle=ls,
                        p4er=P4ERS_SPD[spd], p4ersize=5, linewidth=1.5,
                        label=f"{spd}km/h - {lp}")

        ax.axhline(100, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(MUSCLE_LABELS.get(muscle, muscle), fontsize=9, fontweight="bold")
        ax.set_xlabel("Desnivell (%)", fontsize=8)
        ax.set_ylabel("% baseline", fontsize=8)
        ax.set_xticks(INCLINES)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)

    total = n_rows * 3
    for idx in range(len(muscles_plot), total):
        axes[idx // 3][idx % 3].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    seen = set(); h2 = []; l2 = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    fig.legend(h2, l2, loc="lower right", fontsize=8,
               title="Velocitat — Participant", ncol=3,
               bbox_to_anchor=(0.99, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fname = os.path.join(COMP_DIR, "efecte_desnivell.png")
    plt.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  ✓ efecte_desnivell")


# ══════════════════════════════════════════════════════════════════
# FIGURE 5 — ABSOLUTE VALUES (diagnostic)
# ══════════════════════════════════════════════════════════════════
def fig_absoluts(df):
    print("\n── Valors absoluts ──")
    participants  = df["participant"].unique()
    muscles_plot  = [m for m in MUSCLE_ORDER
                     if m not in ("RF_l", "RF_r")]

    n_rows = (len(muscles_plot) + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 3.5))
    fig.suptitle(
        "Valors ABSOLUTS de RMS (µV) — sense normalització\n"
        "(línia grisa = baseline | útil per verificar el preprocessing)",
        fontsize=12, fontweight="bold"
    )

    for idx, muscle in enumerate(muscles_plot):
        ax = axes[idx // 3][idx % 3] if n_rows > 1 else axes[idx % 3]
        for inc in INCLINES:
            for participant in participants:
                ls, lp = P_STYLE.get(participant, ("-", participant))
                vals = []
                for spd in SPEEDS:
                    cond_str = f"{spd}km/h_{inc}%"
                    row = df[(df.participant == participant) &
                             (df.muscle == muscle) &
                             (df.condition == cond_str)]
                    v = (row["mean_rms_uV"].values[0]
                         if len(row) and not pd.isna(row["mean_rms_uV"].values[0])
                         else np.nan)
                    vals.append(v)
                ax.plot(SPEEDS, vals, color=COLORS_INC[inc], linestyle=ls,
                        p4er=P4ERS_INC[inc], p4ersize=5, linewidth=1.5,
                        label=f"{inc}% - {lp}")

        # Reference baseline
        for participant in participants:
            ls, lp = P_STYLE.get(participant, ("-", participant))
            row = df[(df.participant == participant) &
                     (df.muscle == muscle) &
                     (df.condition == "5km/h_0%")]
            if len(row) and not pd.isna(row["mean_rms_uV"].values[0]):
                ax.axhline(row["mean_rms_uV"].values[0], color="gray",
                           linewidth=1, linestyle=ls, alpha=0.5)

        ax.set_title(MUSCLE_LABELS.get(muscle, muscle), fontsize=9, fontweight="bold")
        ax.set_xlabel("Velocitat (km/h)", fontsize=8)
        ax.set_ylabel("RMS (µV)", fontsize=8)
        ax.set_xticks(SPEEDS)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)

    total = n_rows * 3
    for idx in range(len(muscles_plot), total):
        axes[idx // 3][idx % 3].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    seen = set(); h2 = []; l2 = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    fig.legend(h2, l2, loc="lower right", fontsize=8,
               title="Desnivell — Participant", ncol=3,
               bbox_to_anchor=(0.99, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fname = os.path.join(COMP_DIR, "absolut_velocitat.png")
    plt.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  ✓ absolut_velocitat")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    results_path = os.path.join(BASE_DIR, "results", "emg_normalized.csv")
    if not os.path.exists(results_path):
        print("ERROR: no es troba emg_normalized.csv")
        print("  Executa primer: python 01_emg_pipeline.py")
        exit(1)

    df = pd.read_csv(results_path)
    print(f"✓ Taula carregada: {len(df)} files, "
          f"participants: {df['participant'].unique().tolist()}")

    fig_diagnostic()
    fig_heatmaps(df)
    fig_efecte_velocitat(df)
    fig_efecte_desnivell(df)
    fig_absoluts(df)

    print(f"\n{'='*55}")
    print(f"✓ Totes les figures generades a results/figures/")
