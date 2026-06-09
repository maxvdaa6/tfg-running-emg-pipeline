"""
04_garmin_figures.py
--------------------
Generates all Garmin figures from results/garmin_metrics.csv.

Figures generated:
  figures/en/garmin/heatmap_<participant>.png
      → Heatmap metric × condition per participant

  figures/en/garmin/speed_effect.png
      → Metrics vs speed (lines per incline · all participants)

  figures/en/garmin/incline_effect.png
      → Metrics vs incline (lines per speed · all participants)

  figures/en/garmin/all_conditions_summary.png
      → All conditions in order, all participants overlaid

  figures/en/garmin/group_comparison.png   [if N≥3]
      → Group mean ± SD per condition and metric

Prerequisite: run 03_garmin_pipeline.py first

Run:
    python 04_garmin_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "results", "figures", "en", "garmin")
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════
SPEEDS    = [8, 10, 12]
INCLINES  = [0, 5, 10]
COND_RUN  = [f"{s}km/h_{i}%" for s in SPEEDS for i in INCLINES]
COND_ALL  = ["5km/h_0%"] + COND_RUN

COLORS_INC = {0: "steelblue",  5: "darkorange", 10: "crimson"}
COLORS_SPD = {8: "steelblue", 10: "darkorange", 12: "crimson"}

# Style per participant (up to 4)
P_STYLES = {
    "p2":   ("-",  "o", "tab:blue",   "P2"),
    "p1": ("--", "s", "tab:orange", "P1"),
    "p3":  ("-.", "^", "tab:green",  "P3"),
    "p4":  (":",  "D", "tab:red",    "P4"),
}

METRICS = [
    ("fc_mean_bpm",  "Mean Heart Rate (bpm)",       "crimson"),
    ("fc_max_bpm",   "P2 Heart Rate (bpm)",         "darkred"),
    ("cadence_ppm",  "Cadence (steps/min)",          "steelblue"),
    ("gct_ms",       "Ground Contact Time (ms)",     "darkorange"),
    ("vert_osc_mm",  "Vertical Oscillation (mm)",    "forestgreen"),
]

def get_val(df, participant, condition, col):
    row = df[(df.participant == participant) & (df.condition == condition)]
    if len(row) == 0:
        return np.nan
    v = row[col].values[0]
    return np.nan if pd.isna(v) else float(v)


# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — HEATMAP per participant
# ══════════════════════════════════════════════════════════════════
def fig_heatmaps(df):
    print("\n── Heatmaps per participant ──")
    for participant in df["participant"].unique():
        sub = df[(df.participant == participant) & (df.condition.isin(COND_RUN))]

        fig, axes = plt.subplots(1, len(METRICS), figsize=(18, 5))
        fig.suptitle(f"{participant.upper()} — Garmin Metrics per Condition",
                     fontsize=12, fontweight="bold")

        for ax, (col, label, _) in zip(axes, METRICS):
            vals = np.array([get_val(sub, participant, c, col) for c in COND_RUN])
            vmin = np.nanmin(vals) * 0.97
            vmax = np.nanmax(vals) * 1.03
            # GCT: lower = better -> green at low end
            cmap = "RdYlGn" if col == "gct_ms" else "RdYlGn_r"
            im = ax.imshow(vals.reshape(-1, 1), aspect="auto",
                           cmap=cmap, vmin=vmin, vp2=vmax)
            ax.set_xticks([])
            ax.set_yticks(range(len(COND_RUN)))
            ax.set_yticklabels(COND_RUN, fontsize=8)
            ax.set_title(label, fontsize=8, fontweight="bold")
            for i, v in enumerate(vals):
                if not np.isnan(v):
                    fmt = f"{v:.0f}" if v > 10 else f"{v:.1f}"
                    brightness = (v - vmin) / (vp2 - vmin + 1e-9)
                    txt_color = "white" if brightness > 0.65 or brightness < 0.35 else "black"
                    ax.text(0, i, fmt, ha="center", va="center",
                            fontsize=9, fontweight="bold", color=txt_color)
            plt.colorbar(im, ax=ax, shrink=0.7)

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"heatmap_{participant}.png"),
                    dpi=140, bbox_inches="tight")
        plt.close()
        print(f"  ✓ heatmap_{participant}")


# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — SPEED EFFECT
# ══════════════════════════════════════════════════════════════════
def fig_efecte_velocitat(df):
    print("\n── Speed effect ──")
    participants = df["participant"].unique()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Effect of SPEED on Garmin Metrics\n"
                 "(each line = one incline · line style = participant)",
                 fontsize=12, fontweight="bold")
    axes_flat = axes.flatten()

    for ax, (col, label, color) in zip(axes_flat, METRICS):
        for inc in INCLINES:
            for participant in participants:
                ls, mk, pc, lp = P_STYLES.get(participant,
                                               ("-","o","gray",participant))
                vals = [get_val(df, participant, f"{s}km/h_{inc}%", col)
                        for s in SPEEDS]
                ax.plot(SPEEDS, vals, color=COLORS_INC[inc], linestyle=ls,
                        p4er=mk, p4ersize=6, linewidth=1.8,
                        label=f"{inc}% — {lp}")
        # Baseline
        for participant in participants:
            ls, mk, pc, lp = P_STYLES.get(participant, ("-","o","gray",participant))
            v = get_val(df, participant, "5km/h_0%", col)
            if not np.isnan(v):
                ax.axhline(v, color="gray", linewidth=0.7,
                           linestyle=ls, alpha=0.45)
        ax.set_title(label, fontsize=10, fontweight="bold", color=color)
        ax.set_xlabel("Speed (km/h)", fontsize=9)
        ax.set_ylabel(label, fontsize=8)
        ax.set_xticks(SPEEDS)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)

    # Legend without duplicates
    handles, labels = axes_flat[0].get_legend_handles_labels()
    seen = set(); h2 = []; l2 = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    fig.legend(h2, l2, loc="lower right", fontsize=9, ncol=3,
               title="Incline — Participant", bbox_to_anchor=(0.99, 0.01))
    axes_flat[-1].set_visible(False)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(OUT_DIR, "speed_effect.png"),
                dpi=140, bbox_inches="tight")
    plt.close()
    print("  ✓ speed_effect")


# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — INCLINE EFFECT
# ══════════════════════════════════════════════════════════════════
def fig_efecte_desnivell(df):
    print("\n── Incline effect ──")
    participants = df["participant"].unique()

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Effect of INCLINE on Garmin Metrics\n"
                 "(each line = one speed · line style = participant)",
                 fontsize=12, fontweight="bold")
    axes_flat = axes.flatten()

    for ax, (col, label, color) in zip(axes_flat, METRICS):
        for spd in SPEEDS:
            for participant in participants:
                ls, mk, pc, lp = P_STYLES.get(participant,
                                               ("-","o","gray",participant))
                vals = [get_val(df, participant, f"{spd}km/h_{i}%", col)
                        for i in INCLINES]
                ax.plot(INCLINES, vals, color=COLORS_SPD[spd], linestyle=ls,
                        p4er=mk, p4ersize=6, linewidth=1.8,
                        label=f"{spd} km/h — {lp}")
        for participant in participants:
            ls, mk, pc, lp = P_STYLES.get(participant, ("-","o","gray",participant))
            v = get_val(df, participant, "5km/h_0%", col)
            if not np.isnan(v):
                ax.axhline(v, color="gray", linewidth=0.7,
                           linestyle=ls, alpha=0.45)
        ax.set_title(label, fontsize=10, fontweight="bold", color=color)
        ax.set_xlabel("Incline (%)", fontsize=9)
        ax.set_ylabel(label, fontsize=8)
        ax.set_xticks(INCLINES)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    seen = set(); h2 = []; l2 = []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    fig.legend(h2, l2, loc="lower right", fontsize=9, ncol=3,
               title="Speed — Participant", bbox_to_anchor=(0.99, 0.01))
    axes_flat[-1].set_visible(False)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(OUT_DIR, "incline_effect.png"),
                dpi=140, bbox_inches="tight")
    plt.close()
    print("  ✓ incline_effect")


# ══════════════════════════════════════════════════════════════════
# FIGURE 4 — SUMMARY ALL CONDITIONS
# ══════════════════════════════════════════════════════════════════
def fig_resum(df):
    print("\n── All conditions summary ──")
    participants = df["participant"].unique()
    x = range(len(COND_ALL))

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(14, 16), sharex=True)
    fig.suptitle("Garmin Summary — All Conditions\n"
                 "(each symbol = one participant)",
                 fontsize=12, fontweight="bold")

    for ax, (col, label, color) in zip(axes, METRICS):
        for participant in participants:
            ls, mk, pc, lp = P_STYLES.get(participant,
                                           ("-","o","gray",participant))
            vals = [get_val(df, participant, c, col) for c in COND_ALL]
            ax.scatter(x, vals, p4er=mk, s=65, color=pc,
                       alpha=0.9, label=lp, zorder=3)
            ax.plot(x, vals, color=pc, linewidth=0.9, alpha=0.5)
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.25)
        ax.axvline(0.5, color="gray", linewidth=1,
                   linestyle="--", alpha=0.4)
        ymin, yp2 = ax.get_ylim()
        ax.text(0.15, yp2 - (yp2-ymin)*0.05, "baseline (5km/h 0%)",
                fontsize=7, color="gray", va="top")

    axes[-1].set_xticks(range(len(COND_ALL)))
    axes[-1].set_xticklabels(COND_ALL, rotation=35, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "all_conditions_summary.png"),
                dpi=140, bbox_inches="tight")
    plt.close()
    print("  ✓ all_conditions_summary")


# ══════════════════════════════════════════════════════════════════
# FIGURE 5 — BETWEEN-SUBJECT COMPARISON (mean +/- SD) [N>=3]
# ══════════════════════════════════════════════════════════════════
def fig_comparacio_subjectes(df):
    participants = df["participant"].unique()
    if len(participants) < 3:
        print("\n── Group comparison: need N≥3, skipping ──")
        return

    print("\n── Group comparison (mean ± SD) ──")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Group Comparison — Mean ± SD per Condition\n"
                 f"(N={len(participants)} participants)",
                 fontsize=12, fontweight="bold")
    axes_flat = axes.flatten()

    x  = range(len(COND_RUN))

    for ax, (col, label, color) in zip(axes_flat, METRICS):
        means = []; stds = []
        for c in COND_RUN:
            vals = [get_val(df, p, c, col) for p in participants]
            vals = [v for v in vals if not np.isnan(v)]
            means.append(np.mean(vals) if vals else np.nan)
            stds.append(np.std(vals)   if vals else np.nan)

        means = np.array(means); stds = np.array(stds)
        ax.bar(x, means, yerr=stds, capsize=4, color=color,
               alpha=0.7, error_kw={"linewidth": 1.5})
        ax.set_title(label, fontsize=10, fontweight="bold", color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(COND_RUN, rotation=40, ha="right", fontsize=7.5)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.25, axis="y")
        ax.tick_params(labelsize=8)

    axes_flat[-1].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "group_comparison.png"),
                dpi=140, bbox_inches="tight")
    plt.close()
    print("  ✓ group_comparison")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    results_path = os.path.join(BASE_DIR, "results", "garmin_metrics.csv")
    if not os.path.exists(results_path):
        print("ERROR: no es troba garmin_metrics.csv")
        print("  Executa primer: python 03_garmin_pipeline.py")
        exit(1)

    df = pd.read_csv(results_path)
    print(f"✓ Table loaded: {len(df)} rows, "
          f"participants: {df['participant'].unique().tolist()}")

    fig_heatmaps(df)
    fig_efecte_velocitat(df)
    fig_efecte_desnivell(df)
    fig_resum(df)
    fig_comparacio_subjectes(df)

    print(f"\n{'='*55}")
    print(f"✓ All figures saved to results/figures/en/garmin/")
