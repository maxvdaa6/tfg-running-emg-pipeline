"""
08_statistics.py

Statistical analysis for the EMG + IMU + Garmin pilot study (n=4).

ESTRUCTURA:
  Bloc 1 — Friedman test
      Detecta si hi ha diferències significatives entre les 9 condicions
      de cursa per a cada variable. Test principal de l'estudi.

  Bloc 2 — Wilcoxon post-hoc amb Bonferroni
      Compara parelles clau (velocitats a 0%, desnivells a 10 km/h).
      NOTA: amb n=4, el p_min del Wilcoxon és 0.125 (dos-cues), de manera
      que mai s'assolirà p < 0.05. S'inclou per il·lustrar la metodologia
      i reportar la mida de l'efecte (r).

  Bloc 3 — Correlació de Spearman entre modalitats
      Valida la Hipòtesi 2 (mecanisme dual velocitat/desnivell):
        · Velocitat → ROM maluc + genoll (+ cadència Garmin)
        · Desnivell → ROM turmell + EMG VL_l (+ FC Garmin)
      S'usen mitjanes de grup (n=9 condicions) per evitar dependència
      intra-participant.

  Bloc 4 — Taula resum i figura

LIMITACIÓ PRINCIPAL:
  n=4 implica potència estadística molt baixa. Els resultats s'han
  d'interpretar com a pilot descriptiu, no com a evidència confirmatòria.
  Resultats no significatius NO indiquen absència d'efecte.

Output:
  results/estadistica/friedman_results.csv
  results/estadistica/wilcoxon_posthoc.csv
  results/estadistica/spearman_correlations.csv
  results/figures/en/estadistica/statistical_summary.png

Execució:
    python 08_estadistica.py
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RES_DIR   = os.path.join(BASE_DIR, "results")
STAT_DIR  = os.path.join(RES_DIR, "estadistica")
FIG_DIR   = os.path.join(RES_DIR, "figures", "en", "estadistica")
for d in [STAT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
emg = pd.read_csv(os.path.join(RES_DIR, "emg_normalized.csv"))
emg = emg[emg.excluded == False].copy()
emg["condition"] = emg["condition"].str.strip()

imu = pd.read_csv(os.path.join(RES_DIR, "imu_metrics.csv"))
imu["condition"] = imu["condition"].str.strip()

gar = pd.read_csv(os.path.join(RES_DIR, "garmin_metrics.csv"))
gar["condition"] = gar["condition"].str.strip()

PARTICIPANTS = ["p2", "p1", "p3", "p4"]

# 9 running conditions (walking baseline excluded)
COND_RUN = [
    "8km/h_0%",  "8km/h_5%",  "8km/h_10%",
    "10km/h_0%", "10km/h_5%", "10km/h_10%",
    "12km/h_0%", "12km/h_5%", "12km/h_10%",
]

# ══════════════════════════════════════════════════════════════════
# VARIABLES TO ANALYSE
# ══════════════════════════════════════════════════════════════════
# EMG: reliable muscles (CV < 80% and no full-channel exclusions)
EMG_MUSCLES = {
    "TA_r":  "Tibialis Ant. D",
    "TA_l":  "Tibialis Ant. E",
    "BF_r":  "Bíceps Fem. D",
    "BF_l":  "Bíceps Fem. E",
    "VL_l":  "Vastus Lat. E",
    "SO_l":  "Sòleus E",
    "SO_r":  "Sòleus D",
    "MG_r":  "Med. Gastrocn. D",
    "ST_r":  "Semitendinos D",
    "GM_l":  "Glut. Medius E",
    "GM_r":  "Glut. Medius D",
}

# IMU: joint ROM (hip_abd excluded due to calibration artefact in P4)
IMU_VARS = {
    "R_hip_flex_ROM":   "Maluc Flex D ROM",
    "L_hip_flex_ROM":   "Maluc Flex E ROM",
    "R_knee_flex_ROM":  "Genoll Flex D ROM",
    "L_knee_flex_ROM":  "Genoll Flex E ROM",
    "R_ankle_dors_ROM": "Turmell Dors D ROM",
    "L_ankle_dors_ROM": "Turmell Dors E ROM",
}

# Garmin
GAR_VARS = {
    "fc_mean_bpm":  "FC mitjana (bpm)",
    "cadence_ppm":  "Cadència (ppm)",
    "gct_ms":       "GCT (ms)",
    "vert_osc_mm":  "Oscil·lació vert. (mm)",
}

# ══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def kendall_w(chi2, n, k):
    """
    Kendall's W: mida de l'efecte per al test de Friedman.
    W = chi2 / (n × (k-1))
    Interpretació: 0.1=petit, 0.3=mitjà, 0.5=gran
    """
    return round(chi2 / (n * (k - 1)), 3)

def interpret_w(w):
    if w >= 0.5:   return "gran"
    elif w >= 0.3: return "mitjà"
    elif w >= 0.1: return "petit"
    else:          return "negligible"

def wilcoxon_r(stat, n):
    """Mida de l'efecte per a Wilcoxon: r = Z / sqrt(n)."""
    # Z approximation from T statistic
    # Using scipy wilcoxon which returns W and p
    # r estimated as sqrt(stat/(n*(n+1)/4)) — approximation for small n
    return round(np.sqrt(stat / (n * (n + 1) / 4)), 3) if n > 0 else None

def run_friedman(pivot_df, conds):
    """
    Executa Friedman sobre un pivot DataFrame (participants × condicions).
    Retorna chi2, p, W, interpretació.
    """
    # Filter available conditions
    conds_ok = [c for c in conds if c in pivot_df.columns]
    data = pivot_df[conds_ok].dropna()
    n = len(data)
    k = len(conds_ok)
    if n < 3 or k < 3:
        return None, None, None, "insuficient"
    arrays = [data[c].values for c in conds_ok]
    try:
        chi2, p = friedmanchisquare(*arrays)
        w = kendall_w(chi2, n, k)
        return round(chi2, 3), round(p, 4), w, interpret_w(w)
    except Exception:
        return None, None, None, "error"

# ══════════════════════════════════════════════════════════════════
# BLOC 1 — FRIEDMAN TEST
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  BLOC 1 — FRIEDMAN TEST (efecte condició)")
print("═"*60)
print(f"  n={len(PARTICIPANTS)} participants, k=9 condicions de cursa")
print(f"  H0: no hi ha diferències entre condicions")
print(f"  α = 0.05  |  Mida efecte: Kendall's W")

friedman_rows = []

# ── EMG ─────────────────────────────────────────────────────────
print("\n  [EMG]")
for muscle, label in EMG_MUSCLES.items():
    sub = emg[emg.muscle == muscle][["participant", "condition", "norm_mean"]]
    sub = sub[sub.condition.isin(COND_RUN)]
    pivot = sub.pivot(index="participant", columns="condition", values="norm_mean")
    chi2, p, w, interp = run_friedman(pivot, COND_RUN)
    sig = "**" if p is not None and p < 0.05 else ("†" if p is not None and p < 0.10 else "")
    n_valid = pivot.dropna().shape[0] if chi2 is not None else 0
    friedman_rows.append({
        "modalitat": "EMG", "variable": muscle, "etiqueta": label,
        "n": n_valid, "chi2": chi2, "p": p, "W_kendall": w,
        "mida_efecte": interp, "sig": sig,
    })
    p_str = f"{p:.4f}" if p is not None else "—"
    print(f"    {muscle:8s}  χ²={chi2 if chi2 else '—':6}  p={p_str}  W={w}  {interp} {sig}")

# ── IMU ─────────────────────────────────────────────────────────
print("\n  [IMU]")
for col, label in IMU_VARS.items():
    sub = imu[imu.condition.isin(COND_RUN)][["participant", "condition", col]]
    pivot = sub.pivot(index="participant", columns="condition", values=col)
    chi2, p, w, interp = run_friedman(pivot, COND_RUN)
    sig = "**" if p is not None and p < 0.05 else ("†" if p is not None and p < 0.10 else "")
    n_valid = pivot.dropna().shape[0] if chi2 is not None else 0
    friedman_rows.append({
        "modalitat": "IMU", "variable": col, "etiqueta": label,
        "n": n_valid, "chi2": chi2, "p": p, "W_kendall": w,
        "mida_efecte": interp, "sig": sig,
    })
    p_str = f"{p:.4f}" if p is not None else "—"
    print(f"    {col:22s}  χ²={chi2 if chi2 else '—':6}  p={p_str}  W={w}  {interp} {sig}")

# ── GARMIN ───────────────────────────────────────────────────────
print("\n  [GARMIN]")
gar_run = gar[gar.condition.isin(COND_RUN)]
for col, label in GAR_VARS.items():
    if col not in gar.columns:
        continue
    sub = gar_run[["participant", "condition", col]].dropna(subset=[col])
    pivot = sub.pivot(index="participant", columns="condition", values=col)
    chi2, p, w, interp = run_friedman(pivot, COND_RUN)
    sig = "**" if p is not None and p < 0.05 else ("†" if p is not None and p < 0.10 else "")
    n_valid = pivot.dropna().shape[0] if chi2 is not None else 0
    friedman_rows.append({
        "modalitat": "Garmin", "variable": col, "etiqueta": label,
        "n": n_valid, "chi2": chi2, "p": p, "W_kendall": w,
        "mida_efecte": interp, "sig": sig,
    })
    p_str = f"{p:.4f}" if p is not None else "—"
    print(f"    {col:15s}  χ²={chi2 if chi2 else '—':6}  p={p_str}  W={w}  {interp} {sig}")

df_friedman = pd.DataFrame(friedman_rows)
df_friedman.to_csv(os.path.join(STAT_DIR, "friedman_results.csv"), index=False)
print(f"\n  ✓ Guardat: friedman_results.csv")

# ══════════════════════════════════════════════════════════════════
# BLOCK 2 — WILCOXON POST-HOC (key pairs)
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  BLOC 2 — WILCOXON POST-HOC amb Bonferroni")
print("═"*60)
print("  ⚠ NOTA: amb n=4, p_mín Wilcoxon (dos-cues) = 0.125")
print("     Cap comparació pot assolir p<0.05 amb n=4.")
print("     S'usa per estimar la mida de l'efecte (r).")

# Pairs of interest
PAIRS_SPEED  = [("8km/h_0%",  "10km/h_0%"),
                ("8km/h_0%",  "12km/h_0%"),
                ("10km/h_0%", "12km/h_0%")]
PAIRS_INCL   = [("10km/h_0%", "10km/h_5%"),
                ("10km/h_0%", "10km/h_10%"),
                ("10km/h_5%", "10km/h_10%")]
ALL_PAIRS    = PAIRS_SPEED + PAIRS_INCL
N_PAIRS      = len(ALL_PAIRS)
ALPHA_BONF   = round(0.05 / N_PAIRS, 4)  # 0.0083

print(f"  {N_PAIRS} comparacions → α Bonferroni = {ALPHA_BONF}")

# Priority variables for post-hoc
POSTHOC_VARS = [
    ("EMG",    "TA_r",            "norm_mean",        "Tibialis Ant. D"),
    ("EMG",    "TA_l",            "norm_mean",        "Tibialis Ant. E"),
    ("EMG",    "BF_r",            "norm_mean",        "Bíceps Fem. D"),
    ("EMG",    "VL_l",            "norm_mean",        "Vastus Lat. E"),
    ("IMU",    "R_knee_flex_ROM", "R_knee_flex_ROM",  "Genoll Flex D ROM"),
    ("IMU",    "L_hip_flex_ROM",  "L_hip_flex_ROM",   "Maluc Flex E ROM"),
    ("IMU",    "R_ankle_dors_ROM","R_ankle_dors_ROM", "Turmell Dors D ROM"),
    ("Garmin", "fc_mean_bpm",     "fc_mean_bpm",      "FC mitjana"),
    ("Garmin", "cadence_ppm",     "cadence_ppm",      "Cadència"),
]

wilcoxon_rows = []
print(f"\n  {'Variable':22s}  {'Parella':30s}  {'W':>5}  {'p':>6}  {'r':>5}  {'Bonf.sig':>8}")

for modality, var, col, label in POSTHOC_VARS:
    for (c1, c2) in ALL_PAIRS:
        pair_type = "velocitat" if (c1, c2) in PAIRS_SPEED else "desnivell"

        # Extract paired values
        if modality == "EMG":
            sub = emg[emg.muscle == var][["participant", "condition", col]]
        elif modality == "IMU":
            sub = imu[["participant", "condition", col]]
        else:
            sub = gar[["participant", "condition", col]].dropna(subset=[col])

        v1 = sub[sub.condition == c1].set_index("participant")[col]
        v2 = sub[sub.condition == c2].set_index("participant")[col]
        common = v1.index.intersection(v2.index)
        v1, v2 = v1[common].values, v2[common].values
        n = len(v1)

        if n < 4:
            continue

        diff = v1 - v2
        if np.all(diff == 0):
            continue

        try:
            stat, p = wilcoxon(v1, v2, alternative="two-sided", zero_method="wilcox")
            r = wilcoxon_r(stat, n)
            bonf_sig = "**" if p < ALPHA_BONF else ""
            wilcoxon_rows.append({
                "modalitat": modality, "variable": var, "etiqueta": label,
                "condicio_A": c1, "condicio_B": c2, "tipus": pair_type,
                "n": n, "W_stat": round(stat, 3), "p": round(p, 4),
                "r_efecte": r, "bonf_sig": bonf_sig,
            })
            if pair_type == "velocitat" and var in ["TA_r", "VL_l", "cadence_ppm", "R_knee_flex_ROM"]:
                print(f"  {label:22s}  {c1:13s}vs{c2:13s}  "
                      f"{stat:5.1f}  {p:6.4f}  {r:5.3f}  {bonf_sig:>8}")
        except Exception:
            pass

df_wilcoxon = pd.DataFrame(wilcoxon_rows)
df_wilcoxon.to_csv(os.path.join(STAT_DIR, "wilcoxon_posthoc.csv"), index=False)
print(f"\n  ✓ Guardat: wilcoxon_posthoc.csv  ({len(df_wilcoxon)} comparacions)")

# ══════════════════════════════════════════════════════════════════
# BLOCK 3 — SPEARMAN (cross-modal correlations)
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  BLOC 3 — CORRELACIÓ DE SPEARMAN entre modalitats")
print("═"*60)
print("  Usant mitjanes de grup (n=9 condicions) per evitar")
print("  dependència intra-participant.")
print("  Valida la H2: velocitat→ROM_sagital, desnivell→ROM_turmell")

# Build table of group means per condition
def group_means_emg(muscle, metric="norm_mean"):
    sub = emg[(emg.muscle == muscle) & (emg.condition.isin(COND_RUN))]
    return sub.groupby("condition")[metric].mean()

def group_means_imu(col):
    sub = imu[imu.condition.isin(COND_RUN)]
    return sub.groupby("condition")[col].mean()

def group_means_gar(col):
    sub = gar[gar.condition.isin(COND_RUN)]
    return sub.groupby("condition")[col].mean()

# Numeric speed and incline variables per condition
cond_speed   = pd.Series({c: int(c.split("km")[0])    for c in COND_RUN}, name="speed")
cond_incline = pd.Series({c: int(c.split("_")[1][:-1]) for c in COND_RUN}, name="incline")

spearman_rows = []

SPEARMAN_TESTS = [
    # (label_X, series_X, label_Y, series_Y, hypothesis)
    ("Velocitat",            cond_speed,
     "ROM Maluc Flex D",     group_means_imu("R_hip_flex_ROM"),
     "H2: velocitat→ROM maluc"),

    ("Velocitat",            cond_speed,
     "ROM Genoll Flex D",    group_means_imu("R_knee_flex_ROM"),
     "H2: velocitat→ROM genoll"),

    ("Velocitat",            cond_speed,
     "Cadència (ppm)",       group_means_gar("cadence_ppm"),
     "H2: velocitat→cadència"),

    ("Velocitat",            cond_speed,
     "ROM Turmell D",        group_means_imu("R_ankle_dors_ROM"),
     "H2: velocitat flat→turmell"),

    ("Desnivell",            cond_incline,
     "ROM Turmell D",        group_means_imu("R_ankle_dors_ROM"),
     "H2: desnivell→ROM turmell"),

    ("Desnivell",            cond_incline,
     "ROM Maluc Flex D",     group_means_imu("R_hip_flex_ROM"),
     "H2: desnivell→ROM maluc"),

    ("Desnivell",            cond_incline,
     "ROM Genoll Flex D",    group_means_imu("R_knee_flex_ROM"),
     "H2: desnivell flat→genoll"),

    ("Desnivell",            cond_incline,
     "EMG VL_l",             group_means_emg("VL_l"),
     "H2: desnivell→VL_l EMG"),

    ("Desnivell",            cond_incline,
     "EMG TA_r",             group_means_emg("TA_r"),
     "control: desnivell flat→TA"),

    ("EMG VL_l",             group_means_emg("VL_l"),
     "ROM Genoll Flex D",    group_means_imu("R_knee_flex_ROM"),
     "Dissociació: VL↑ però ROM genoll pla amb inclinació"),

    ("EMG TA_r",             group_means_emg("TA_r"),
     "Cadència",             group_means_gar("cadence_ppm"),
     "Integració: TA reflecteix cadència"),

    ("FC mitjana",           group_means_gar("fc_mean_bpm"),
     "EMG BF_r",             group_means_emg("BF_r"),
     "Integració: càrrega cardiovascular vs muscular"),
]

print(f"\n  {'X':20s}  {'Y':22s}  {'ρ':>6}  {'p':>6}  {'n':>3}  Hipòtesi")
for lx, sx, ly, sy, hip in SPEARMAN_TESTS:
    common = sx.index.intersection(sy.index)
    x, y = sx[common].values, sy[common].values
    if len(x) < 5:
        continue
    rho, p = spearmanr(x, y)
    sig = "**" if p < 0.05 else ("†" if p < 0.10 else "")
    spearman_rows.append({
        "X": lx, "Y": ly, "rho": round(rho, 3), "p": round(p, 4),
        "n": len(x), "sig": sig, "hipotesi": hip,
    })
    print(f"  {lx:20s}  {ly:22s}  {rho:+6.3f}  {p:6.4f}  {len(x):3d}  {hip[:45]} {sig}")

df_spearman = pd.DataFrame(spearman_rows)
df_spearman.to_csv(os.path.join(STAT_DIR, "spearman_correlations.csv"), index=False)
print(f"\n  ✓ Guardat: spearman_correlations.csv")

# ══════════════════════════════════════════════════════════════════
# BLOCK 4 — SUMMARY FIGURE
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  BLOC 4 — FIGURA RESUM")
print("═"*60)

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle("Resum estadístic — Pilot EMG + IMU + Garmin (n=4)\n"
             "Friedman (Kendall's W) | Spearman ρ | Wilcoxon r",
             fontsize=13, fontweight="bold")

# ── Panel 1: Friedman W per variable ─────────────────────────────
ax1 = axes[0]
df_f = df_friedman.dropna(subset=["W_kendall"]).sort_values("W_kendall", ascending=True)

colors_modal = {"EMG": "#2196F3", "IMU": "#4CAF50", "Garmin": "#FF9800"}
bar_colors = [colors_modal.get(m, "gray") for m in df_f["modalitat"]]

bars = ax1.barh(df_f["variable"], df_f["W_kendall"], color=bar_colors, edgecolor="white", height=0.7)

# P4 significant
for i, (_, row) in enumerate(df_f.iterrows()):
    if row["sig"]:
        ax1.text(row["W_kendall"] + 0.01, i, row["sig"], va="center", fontsize=9, color="red")

ax1.axvline(0.1, color="gray",  ls="--", lw=0.8, alpha=0.6, label="petit (0.1)")
ax1.axvline(0.3, color="orange",ls="--", lw=0.8, alpha=0.6, label="mitjà (0.3)")
ax1.axvline(0.5, color="red",   ls="--", lw=0.8, alpha=0.6, label="gran (0.5)")
ax1.set_xlabel("Kendall's W (mida efecte Friedman)", fontsize=9)
ax1.set_title("Friedman — efecte condició\n(** p<0.05, † p<0.10)", fontsize=10, fontweight="bold")
ax1.set_xlim(0, max(df_f["W_kendall"].max() + 0.08, 0.7))
ax1.tick_params(labelsize=7)
ax1.legend(fontsize=7, loc="lower right")

# Modality legend
from matplotlib.patches import Patch
legend_patches = [Patch(color=v, label=k) for k, v in colors_modal.items()]
ax1.legend(handles=legend_patches, fontsize=7, loc="lower right")

# ── Panel 2: Spearman ρ ───────────────────────────────────────────
ax2 = axes[1]
df_s = df_spearman.copy()
labels_s = [f"{r.X}\nvs {r.Y}" for _, r in df_s.iterrows()]
sp_colors = ["#E53935" if r.rho > 0 else "#1E88E5" for _, r in df_s.iterrows()]
bars2 = ax2.barh(range(len(df_s)), df_s["rho"], color=sp_colors, edgecolor="white", height=0.7)

for i, (_, row) in enumerate(df_s.iterrows()):
    if row["sig"]:
        x_pos = row["rho"] + 0.03 if row["rho"] >= 0 else row["rho"] - 0.03
        ax2.text(x_pos, i, row["sig"], va="center", fontsize=9,
                 color="black", ha="left" if row["rho"] >= 0 else "right")

ax2.set_yticks(range(len(df_s)))
ax2.set_yticklabels(labels_s, fontsize=6)
ax2.axvline(0, color="black", lw=0.8)
ax2.axvline( 0.6, color="green", ls="--", lw=0.8, alpha=0.5)
ax2.axvline(-0.6, color="green", ls="--", lw=0.8, alpha=0.5)
ax2.set_xlabel("Spearman ρ (** p<0.05, † p<0.10)", fontsize=9)
ax2.set_title("Spearman — correlacions entre modalitats\n(vermell=+, blau=−)", fontsize=10, fontweight="bold")
ax2.set_xlim(-1.1, 1.1)
ax2.tick_params(labelsize=7)

# Panel 3: Summary table of significant variables
ax3 = axes[2]
ax3.axis("off")

# Build summary table
sig_fried = df_friedman[df_friedman["sig"] != ""].copy()
sig_spear = df_spearman[df_spearman["sig"] != ""].copy()

table_data = []
table_data.append(["FRIEDMAN significatius (p<0.05)", "", ""])
table_data.append(["Variable", "W", "p"])
for _, r in sig_fried.iterrows():
    table_data.append([f"{r['etiqueta']}", f"{r['W_kendall']}", f"{r['p']}"])

table_data.append(["", "", ""])
table_data.append(["SPEARMAN significatius (p<0.05)", "", ""])
table_data.append(["X vs Y", "ρ", "p"])
for _, r in sig_spear.iterrows():
    label = f"{r['X'][:12]} / {r['Y'][:12]}"
    table_data.append([label, f"{r['rho']:+.3f}", f"{r['p']:.4f}"])

if not table_data:
    table_data.append(["Cap resultat significatiu", "", ""])
    table_data.append(["(potència baixa: n=4)", "", ""])

tbl = ax3.table(cellText=table_data, loc="center", cellLoc="left")
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1.2, 1.4)

# Header formatting
for (row, col), cell in tbl.get_celld().items():
    if row == 0 or table_data[row][0] in ["Variable", "X vs Y",
                                            "FRIEDMAN significatius (p<0.05)",
                                            "SPEARMAN significatius (p<0.05)"]:
        cell.set_facecolor("#E3F2FD")
        cell.set_text_props(fontweight="bold")
    cell.set_edgecolor("white")

ax3.set_title("Resum resultats significatius\n(α=0.05)", fontsize=10, fontweight="bold")

# Footnote
fig.text(0.5, 0.01,
         "⚠ Pilot n=4 · Potència estadística limitada · Wilcoxon p_mín=0.125 · "
         "Spearman calculat sobre mitjanes de grup (n=9 condicions)",
         ha="center", fontsize=7.5, color="gray", style="italic")

plt.tight_layout(rect=[0, 0.04, 1, 1])
fig_path = os.path.join(FIG_DIR, "statistical_summary.png")
fig.savefig(fig_path, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ Figure saved: statistical_summary.png")

# ══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*60)
print("  RESUM FINAL")
print("═"*60)

n_sig_fried = (df_friedman["sig"] == "**").sum()
n_tend_fried = (df_friedman["sig"] == "†").sum()
n_sig_spear = (df_spearman["sig"] == "**").sum()

print(f"\n  Friedman:  {n_sig_fried} variables significatives (p<0.05)")
print(f"             {n_tend_fried} tendències (0.05<p<0.10)")
print(f"  Spearman:  {n_sig_spear} correlacions significatives (p<0.05)")

print(f"\n  Fitxers guardats a: results/estadistica/")
print(f"    · friedman_results.csv")
print(f"    · wilcoxon_posthoc.csv")
print(f"    · spearman_correlations.csv")
print(f"  Figure: results/figures/en/estadistica/statistical_summary.png")

print(f"\n  INTERPRETACIÓ GLOBAL:")
print(f"  Els resultats estadístics s'han d'interpretar com a pilot")
print(f"  descriptiu. La baixa potència (n=4) fa que resultats no")
print(f"  significatius no impliquin absència d'efecte.")
print(f"  Prioritzar la mida de l'efecte (W i ρ) sobre el p-valor.")
