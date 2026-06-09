"""
07_imu_figures.py
-----------------
Generates all IMU figures from results/imu_metrics.csv.

Figures generated:
  figures/en/imu/heatmap_rom_<participant>.png
      → ROM heatmap (p5-p95) per joint × condition, right and left

  figures/en/imu/heatmap_si_<participant>.png
      → Symmetry Index (SI) heatmap per joint × condition

  figures/en/imu/speed_effect_rom.png
      → ROM vs speed, per joint and incline (all participants)

  figures/en/imu/incline_effect_rom.png
      → ROM vs incline, per joint and speed (all participants)

  figures/en/imu/asymmetry_by_condition.png
      → SI per joint and condition, all participants overlaid

  figures/en/imu/demographic_comparison.png
      → Mean ROM and SI per participant, ordered by age

Prerequisite: run 06_imu_pipeline.py first

Run:
    python 07_imu_figures.py
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
OUT_DIR  = os.path.join(BASE_DIR, "results", "figures", "en", "imu")
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════
SPEEDS   = [8, 10, 12]
INCLINES = [0, 5, 10]
COND_RUN = [f"{s}km/h_{i}%" for s in SPEEDS for i in INCLINES]

JOINTS = {
    "hip_abd":    "Hip Abd/Add (°)",
    "hip_flex":   "Hip Flex/Ext (°)",
    "knee_flex":  "Knee Flex/Ext (°)",
    "ankle_dors": "Ankle Dors/Plant (°)",
}
JOINT_COLORS = {
    "hip_abd":    "steelblue",
    "hip_flex":   "darkorange",
    "knee_flex":  "crimson",
    "ankle_dors": "forestgreen",
}

COLORS_INC = {0: "steelblue", 5: "darkorange", 10: "crimson"}
COLORS_SPD = {8: "steelblue", 10: "darkorange", 12: "crimson"}

P_STYLES = {
    "p2":   ("-",  "o", "tab:blue",   "P2 (M, 21y)"),
    "p1": ("--", "s", "tab:orange", "P1 (F, 21y)"),
    "p3":  ("-.", "^", "tab:green",  "P3 (F, 52y)"),
    "p4":  (":",  "D", "tab:red",    "P4 (M, 54y)"),
}

def get_val(df, participant, condition, col):
    row = df[(df.participant == participant) & (df.condition == condition)]
    if len(row) == 0:
        return np.nan
    v = row[col].values[0]
    return np.nan if pd.isna(v) else float(v)


# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — ROM HEATMAP per participant
# ══════════════════════════════════════════════════════════════════
def fig_heatmap_rom(df):
    print("\n── ROM Heatmaps ──")
    for participant in df["participant"].unique():
        sub = df[df.participant == participant]

        # 4 joints × 2 sides = 8 columns
        fig, axes = plt.subplots(1, 8, figsize=(22, 5))
        fig.suptitle(f"{participant.upper()} — Joint ROM per Condition (p5-p95)",
                     fontsize=11, fontweight="bold")

        col_idx = 0
        for joint, label in JOINTS.items():
            for side, side_label in [("R", "Right"), ("L", "Left")]:
                ax  = axes[col_idx]
                col = f"{side}_{joint}_ROM"
                vals = np.array([get_val(sub, participant, c, col) for c in COND_RUN])

                vmin = np.nanmin(vals) * 0.95
                vmax = np.nanmax(vals) * 1.05
                im = ax.imshow(vals.reshape(-1, 1), aspect="auto",
                               cmap="YlOrRd", vmin=vmin, vp2=vmax)
                ax.set_xticks([])
                ax.set_yticks(range(len(COND_RUN)))
                ax.set_yticklabels(COND_RUN if col_idx == 0 else [], fontsize=7)
                short = label.split(" ")[0]
                ax.set_title(f"{short}\n{side_label}", fontsize=7, fontweight="bold")

                for i, v in enumerate(vals):
                    if not np.isnan(v):
                        ax.text(0, i, f"{v:.1f}°", ha="center", va="center",
                                fontsize=7.5, fontweight="bold",
                                color="white" if v > (vmin + 0.6*(vp2-vmin)) else "black")
                plt.colorbar(im, ax=ax, shrink=0.6)
                col_idx += 1

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"heatmap_rom_{participant}.png"),
                    dpi=140, bbox_inches="tight")
        plt.close()
        print(f"  ✓ heatmap_rom_{participant}")


# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — SYMMETRY INDEX (SI) HEATMAP per participant
# ══════════════════════════════════════════════════════════════════
def fig_heatmap_si(df):
    print("\n── Symmetry Index (SI) Heatmaps ──")
    for participant in df["participant"].unique():
        sub = df[df.participant == participant]

        fig, axes = plt.subplots(1, 4, figsize=(14, 5))
        fig.suptitle(
            f"{participant.upper()} — Bilateral Symmetry Index (SI ROM) per Condition\n"
            "SI > 0 → Right dominant  |  SI < 0 → Left dominant",
            fontsize=10, fontweight="bold")

        for ax, (joint, label) in zip(axes, JOINTS.items()):
            col  = f"SI_{joint}_ROM"
            vals = np.array([get_val(sub, participant, c, col) for c in COND_RUN])
            vlim = max(abs(np.nanmax(vals)), abs(np.nanmin(vals))) * 1.1
            vlim = max(vlim, 5)  # minimum ±5% scale

            im = ax.imshow(vals.reshape(-1, 1), aspect="auto",
                           cmap="RdBu_r", vmin=-vlim, vp2=vlim)
            ax.set_xticks([])
            ax.set_yticks(range(len(COND_RUN)))
            ax.set_yticklabels(COND_RUN if joint == "hip_abd" else [], fontsize=7)
            ax.set_title(label.replace(" (°)", ""), fontsize=8, fontweight="bold")

            for i, v in enumerate(vals):
                if not np.isnan(v):
                    ax.text(0, i, f"{v:+.1f}%", ha="center", va="center",
                            fontsize=8, fontweight="bold",
                            color="white" if abs(v) > vlim * 0.6 else "black")
            plt.colorbar(im, ax=ax, shrink=0.6, label="SI (%)")

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"heatmap_si_{participant}.png"),
                    dpi=140, bbox_inches="tight")
        plt.close()
        print(f"  ✓ heatmap_si_{participant}")


# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — SPEED EFFECT on ROM
# ══════════════════════════════════════════════════════════════════
def fig_efecte_velocitat(df):
    print("\n── Speed effect on ROM ──")
    participants = df["participant"].unique()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Effect of SPEED on Joint ROM (p5-p95)\n"
                 "(each line = one incline · line style = participant)",
                 fontsize=11, fontweight="bold")

    for ax, (joint, label) in zip(axes.flatten(), JOINTS.items()):
        for inc in INCLINES:
            for participant in participants:
                ls, mk, pc, lp = P_STYLES.get(participant, ("-","o","gray",participant))
                # Mean right+left ROM
                vals = []
                for s in SPEEDS:
                    cond = f"{s}km/h_{inc}%"
                    r = get_val(df, participant, cond, f"R_{joint}_ROM")
                    l = get_val(df, participant, cond, f"L_{joint}_ROM")
                    vals.append(np.nanmean([r, l]))
                ax.plot(SPEEDS, vals, color=COLORS_INC[inc], linestyle=ls,
                        p4er=mk, p4ersize=6, linewidth=1.8,
                        label=f"{inc}% — {lp}")

        ax.set_title(label, fontsize=10, fontweight="bold",
                     color=JOINT_COLORS[joint])
        ax.set_xlabel("Speed (km/h)", fontsize=9)
        ax.set_ylabel("Mean ROM R+L (°)", fontsize=8)
        ax.set_xticks(SPEEDS)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)

    # Legend without duplicates
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    seen = set(); h2 = []; l2 = []
    for h, lb in zip(handles, labels):
        if lb not in seen:
            seen.add(lb); h2.append(h); l2.append(lb)
    fig.legend(h2, l2, loc="lower center", fontsize=8, ncol=4,
               title="Incline — Participant", bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(OUT_DIR, "speed_effect_rom.png"),
                dpi=140, bbox_inches="tight")
    plt.close()
    print("  ✓ speed_effect_rom")


# ══════════════════════════════════════════════════════════════════
# FIGURE 4 — INCLINE EFFECT on ROM
# ══════════════════════════════════════════════════════════════════
def fig_efecte_desnivell(df):
    print("\n── Incline effect on ROM ──")
    participants = df["participant"].unique()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Effect of INCLINE on Joint ROM (p5-p95)\n"
                 "(each line = one speed · line style = participant)",
                 fontsize=11, fontweight="bold")

    for ax, (joint, label) in zip(axes.flatten(), JOINTS.items()):
        for spd in SPEEDS:
            for participant in participants:
                ls, mk, pc, lp = P_STYLES.get(participant, ("-","o","gray",participant))
                vals = []
                for inc in INCLINES:
                    cond = f"{spd}km/h_{inc}%"
                    r = get_val(df, participant, cond, f"R_{joint}_ROM")
                    l = get_val(df, participant, cond, f"L_{joint}_ROM")
                    vals.append(np.nanmean([r, l]))
                ax.plot(INCLINES, vals, color=COLORS_SPD[spd], linestyle=ls,
                        p4er=mk, p4ersize=6, linewidth=1.8,
                        label=f"{spd} km/h — {lp}")

        ax.set_title(label, fontsize=10, fontweight="bold",
                     color=JOINT_COLORS[joint])
        ax.set_xlabel("Incline (%)", fontsize=9)
        ax.set_ylabel("Mean ROM R+L (°)", fontsize=8)
        ax.set_xticks(INCLINES)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)

    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    seen = set(); h2 = []; l2 = []
    for h, lb in zip(handles, labels):
        if lb not in seen:
            seen.add(lb); h2.append(h); l2.append(lb)
    fig.legend(h2, l2, loc="lower center", fontsize=8, ncol=4,
               title="Speed — Participant", bbox_to_anchor=(0.5, 0.0))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(OUT_DIR, "incline_effect_rom.png"),
                dpi=140, bbox_inches="tight")
    plt.close()
    print("  ✓ incline_effect_rom")


# ══════════════════════════════════════════════════════════════════
# FIGURE 5 — ASYMMETRY (SI) by condition — all joints
# ══════════════════════════════════════════════════════════════════
def fig_asimetria(df):
    print("\n── Asymmetry by condition ──")
    participants = df["participant"].unique()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Bilateral Symmetry Index (SI ROM) by Condition\n"
                 "SI > 0 → Right dominant  |  SI < 0 → Left dominant  "
                 "|  dashed = ±10% (clinical reference threshold)",
                 fontsize=10, fontweight="bold")

    x = range(len(COND_RUN))

    for ax, (joint, label) in zip(axes.flatten(), JOINTS.items()):
        for participant in participants:
            ls, mk, pc, lp = P_STYLES.get(participant, ("-","o","gray",participant))
            vals = [get_val(df, participant, c, f"SI_{joint}_ROM") for c in COND_RUN]
            ax.plot(x, vals, color=pc, linestyle=ls, p4er=mk,
                    p4ersize=6, linewidth=1.6, label=lp, zorder=3)
            ax.scatter(x, vals, color=pc, s=40, zorder=4)

        # Zero line and ±10% thresholds
        ax.axhline(0,   color="black", linewidth=0.8, zorder=1)
        ax.axhline(+10, color="gray",  linewidth=0.8, linestyle="--",
                   alpha=0.6, zorder=1)
        ax.axhline(-10, color="gray",  linewidth=0.8, linestyle="--",
                   alpha=0.6, zorder=1)
        ax.fill_between(x, -10, 10, color="lightgray", alpha=0.15, zorder=0)

        ax.set_title(label, fontsize=10, fontweight="bold",
                     color=JOINT_COLORS[joint])
        ax.set_ylabel("SI ROM (%)", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(COND_RUN, rotation=35, ha="right", fontsize=7.5)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "asymmetry_by_condition.png"),
                dpi=140, bbox_inches="tight")
    plt.close()
    print("  ✓ asymmetry_by_condition")


# ══════════════════════════════════════════════════════════════════
# FIGURE 6 — DEMOGRAPHIC COMPARISON (sex and age)
# ══════════════════════════════════════════════════════════════════
def fig_demografia(df):
    print("\n── Demographic comparison ──")

    participants = df["participant"].unique()
    joints       = list(JOINTS.keys())

    # Mean ROM per participant (mean across all conditions and sides)
    summary = []
    for p in participants:
        sub = df[df.participant == p]
        row = {
            "participant": p,
            "sex":         sub["sex"].iloc[0],
            "age":         sub["age"].iloc[0],
            "height_m":    sub["height_m"].iloc[0],
            "age_group":   sub["age_group"].iloc[0],
        }
        for joint in joints:
            r_rom = sub[f"R_{joint}_ROM"].mean()
            l_rom = sub[f"L_{joint}_ROM"].mean()
            row[f"ROM_{joint}"] = (r_rom + l_rom) / 2
            row[f"SI_{joint}"]  = sub[f"SI_{joint}_ROM"].mean()
        summary.append(row)
    sdf = pd.DataFrame(summary)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        "Demographic Comparison — Mean ROM and Asymmetry per Participant\n"
        "(Ordered: young → older  |  M = Male, F = Female  |  "
        "Note: N=2 per group, descriptive results only)",
        fontsize=10, fontweight="bold"
    )

    order = sorted(participants,
                   key=lambda p: (sdf[sdf.participant==p]["age"].values[0]))

    def bcolor(p):
        s = sdf[sdf.participant == p]["sex"].values[0]
        return "steelblue" if s in ("M", "H") else "salmon"

    xlabels = [f"{p.capitalize()}\n({sdf[sdf.participant==p]['sex'].values[0]},"
               f" {int(sdf[sdf.participant==p]['age'].values[0])}y,"
               f" {sdf[sdf.participant==p]['height_m'].values[0]}m)"
               for p in order]

    # Top row: ROM per joint
    for ax, (joint, label) in zip(axes[0], JOINTS.items()):
        vals   = [sdf[sdf.participant == p][f"ROM_{joint}"].values[0] for p in order]
        colors = [bcolor(p) for p in order]
        bars   = ax.bar(range(len(order)), vals, color=colors, alpha=0.8, edgecolor="white")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_title(f"ROM — {label}", fontsize=9, fontweight="bold",
                     color=JOINT_COLORS[joint])
        ax.set_ylabel("Mean ROM (°)", fontsize=8)
        ax.grid(alpha=0.2, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.5,
                    f"{v:.1f}°", ha="center", va="bottom", fontsize=8)

    # Bottom row: SI per joint
    for ax, (joint, label) in zip(axes[1], JOINTS.items()):
        vals   = [sdf[sdf.participant == p][f"SI_{joint}"].values[0] for p in order]
        colors = ["crimson" if v > 0 else "steelblue" for v in vals]
        bars   = ax.bar(range(len(order)), vals, color=colors, alpha=0.75,
                        edgecolor="white")
        ax.axhline(0,   color="black", linewidth=0.8)
        ax.axhline(+10, color="gray",  linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axhline(-10, color="gray",  linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_title(f"SI ROM — {label}", fontsize=9, fontweight="bold",
                     color=JOINT_COLORS[joint])
        ax.set_ylabel("Mean SI (%)", fontsize=8)
        ax.grid(alpha=0.2, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + (0.5 if v >= 0 else -1.5),
                    f"{v:+.1f}%", ha="center", va="bottom", fontsize=8)

    # Sex legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.8, label="Male"),
        Patch(facecolor="salmon",    alpha=0.8, label="Female"),
    ]
    fig.legend(handles=legend_elements, loc="lower right",
               fontsize=9, bbox_to_anchor=(0.99, 0.01))

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "demographic_comparison.png"),
                dpi=140, bbox_inches="tight")
    plt.close()
    print("  ✓ demographic_comparison")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    results_path = os.path.join(BASE_DIR, "results", "imu_metrics.csv")
    if not os.path.exists(results_path):
        print("ERROR: imu_metrics.csv not found")
        print("  Run first: python 06_imu_pipeline.py")
        exit(1)

    df = pd.read_csv(results_path)
    df["condition"] = df["condition"].str.strip()
    print(f"✓ Table loaded: {len(df)} rows, "
          f"participants: {df['participant'].unique().tolist()}")

    fig_heatmap_rom(df)
    fig_heatmap_si(df)
    fig_efecte_velocitat(df)
    fig_efecte_desnivell(df)
    fig_asimetria(df)
    fig_demografia(df)

    print(f"\n{'='*55}")
    print(f"✓ All figures saved to results/figures/en/imu/")
