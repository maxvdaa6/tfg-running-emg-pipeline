"""
Myalbatross · Athlete Dashboard v2  — redesigned
Output: results/dashboards/athlete_dashboard_v2.html
"""
import pandas as pd, json, math, os

BASE    = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, "results")
OUT_DIR = os.path.join(RESULTS, "dashboards")
os.makedirs(OUT_DIR, exist_ok=True)

emg_df    = pd.read_csv(os.path.join(RESULTS, "emg_normalized.csv"))
garmin_df = pd.read_csv(os.path.join(RESULTS, "garmin_metrics.csv"))
imu_df    = pd.read_csv(os.path.join(RESULTS, "imu_metrics.csv"))

RUNNING_CONDITIONS = [
    "8km/h_0%","8km/h_5%","8km/h_10%",
    "10km/h_0%","10km/h_5%","10km/h_10%",
    "12km/h_0%","12km/h_5%","12km/h_10%",
]
MUSCLES_R     = ["TA_r","MG_r","SO_r","VL_r","RF_r","ST_r","BF_r","GM_r"]
MUSCLES_L     = ["TA_l","MG_l","SO_l","VL_l","RF_l","ST_l","BF_l","GM_l"]
MUSCLE_LABELS = ["TA","MG","SO","VL","RF","ST","BF","GM"]
MUSCLE_FULL   = {"TA":"Tibialis Anterior","MG":"Medial Gastrocnemius","SO":"Soleus",
                 "VL":"Vastus Lateralis","RF":"Rectus Femoris","ST":"Semitendinosus",
                 "BF":"Biceps Femoris","GM":"Gluteus Medius"}
JOINTS        = ["hip_flex","knee_flex","ankle_dors"]
JOINT_LABELS  = ["Hip Flexion","Knee Flexion","Ankle Dorsiflexion"]
ATHLETES = {
    "p2":   {"name":"P2","initials":"P2","sex":"Male",  "age":21,"height":"1.86 m","weight":"72 kg","color":"#00d48a"},
    "p1": {"name":"P1","initials":"P1","sex":"Female","age":21,"height":"1.68 m","weight":"58 kg","color":"#38bdf8"},
    "p3":  {"name":"P3","initials":"P3","sex":"Female","age":52,"height":"1.62 m","weight":"53 kg","color":"#a78bfa"},
    "p4":  {"name":"P4","initials":"P4","sex":"Male",  "age":54,"height":"1.81 m","weight":"70 kg","color":"#fb923c"},
}

def safe(v):
    try:
        f = float(v)
        return None if math.isnan(f) else round(f, 1)
    except: return None

def build(p):
    ep = emg_df[emg_df["participant"]==p]
    hm = {}
    for side, muscles in [("R",MUSCLES_R),("L",MUSCLES_L)]:
        hm[side] = {}
        for label, muscle in zip(MUSCLE_LABELS, muscles):
            row = []
            for cond in RUNNING_CONDITIONS:
                rec = ep[(ep["condition"]==cond)&(ep["muscle"]==muscle)]
                if len(rec)==0 or bool(rec.iloc[0]["excluded"]): row.append(None)
                else: row.append(safe(rec.iloc[0]["norm_mean"]))
            hm[side][label] = {"vals": row}

    gp = garmin_df[garmin_df["participant"]==p].copy()
    gp = gp[gp["condition"].isin(RUNNING_CONDITIONS)].set_index("condition").reindex(RUNNING_CONDITIONS)
    garmin = {
        "hr":      [safe(v) for v in gp["fc_mean_bpm"]],
        "cadence": [safe(v) for v in gp["cadence_ppm"]],
        "gct":     [safe(v) for v in gp["gct_ms"]],
        "vo":      [safe(v) for v in gp["vert_osc_mm"]],
    }

    ip = imu_df[imu_df["participant"]==p].copy()
    ip = ip[ip["condition"].isin(RUNNING_CONDITIONS)].set_index("condition").reindex(RUNNING_CONDITIONS)
    rom = {}
    for joint, label in zip(JOINTS, JOINT_LABELS):
        rom[label] = {
            "R": [safe(v) for v in ip[f"R_{joint}_ROM"]],
            "L": [safe(v) for v in ip[f"L_{joint}_ROM"]],
        }

    run = ep[ep["condition"].isin(RUNNING_CONDITIONS) & ~ep["excluded"]]
    peak_hr  = safe(gp["fc_max_bpm"].max()) if "fc_max_bpm" in gp.columns else None
    top_row  = run.loc[run["norm_mean"].idxmax()] if len(run) else None
    cond_means = {}
    for cond in RUNNING_CONDITIONS:
        vals = run[run["condition"]==cond]["norm_mean"].dropna()
        cond_means[cond] = float(vals.mean()) if len(vals) else 0.0
    peak_cond = max(cond_means, key=lambda k: cond_means[k])

    # Cadence at peak condition
    cadence_at_peak = safe(gp.loc[peak_cond, "cadence_ppm"]) if (peak_cond in gp.index and "cadence_ppm" in gp.columns) else None

    # Speed-activation correlation (Pearson r across 9 conditions)
    speed_map = {"8km/h_0%":8,"8km/h_5%":8,"8km/h_10%":8,
                 "10km/h_0%":10,"10km/h_5%":10,"10km/h_10%":10,
                 "12km/h_0%":12,"12km/h_5%":12,"12km/h_10%":12}
    spd_vals, act_vals = [], []
    for cond in RUNNING_CONDITIONS:
        cond_run = run[run["condition"]==cond]["norm_mean"].dropna()
        if len(cond_run):
            spd_vals.append(speed_map[cond])
            act_vals.append(float(cond_run.mean()))
    if len(spd_vals) >= 3:
        import numpy as np
        r_val = float(np.corrcoef(spd_vals, act_vals)[0,1])
        speed_act_corr = round(r_val, 2)
    else:
        speed_act_corr = None

    asym_vals = []
    for label, mr, ml in zip(MUSCLE_LABELS, MUSCLES_R, MUSCLES_L):
        for cond in RUNNING_CONDITIONS:
            rec_r = ep[(ep["condition"]==cond)&(ep["muscle"]==mr)]
            rec_l = ep[(ep["condition"]==cond)&(ep["muscle"]==ml)]
            vr = None if (len(rec_r)==0 or bool(rec_r.iloc[0]["excluded"])) else safe(rec_r.iloc[0]["norm_mean"])
            vl = None if (len(rec_l)==0 or bool(rec_l.iloc[0]["excluded"])) else safe(rec_l.iloc[0]["norm_mean"])
            if vr is not None and vl is not None and (vr + vl) > 0:
                asym_vals.append(abs(vr - vl) / ((vr + vl) / 2) * 100)
    mean_asym = round(sum(asym_vals)/len(asym_vals), 1) if asym_vals else None
    gct_at_peak = safe(gp.loc[peak_cond, "gct_ms"]) if (peak_cond in gp.index and "gct_ms" in gp.columns) else None

    return {
        "id": p, "info": ATHLETES[p], "heatmap": hm, "garmin": garmin, "rom": rom,
        "kpi": {
            "peak_hr":      peak_hr,
            "mean_asym":    mean_asym,
            "top_muscle":   top_row["muscle"][:2] if top_row is not None else "--",
            "top_val":      safe(top_row["norm_mean"]) if top_row is not None else None,
            "top_cond":     str(top_row["condition"]) if top_row is not None else "--",
            "peak_cond":      peak_cond,
            "gct_at_peak":    gct_at_peak,
            "cadence_at_peak": cadence_at_peak,
            "speed_act_corr":  speed_act_corr,
        }
    }

ALL  = {p: build(p) for p in ATHLETES}
DATA_JS = json.dumps(ALL, allow_nan=False)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>myalbatross · Athlete Dashboard</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
  --bg:       #07100d;
  --surface:  #0c1a14;
  --card:     rgba(255,255,255,0.035);
  --border:   rgba(0,212,138,0.12);
  --border2:  rgba(255,255,255,0.07);
  --teal:     #00d48a;
  --teal2:    #00f0a0;
  --tealDim:  rgba(0,212,138,0.15);
  --text:     #e8f5f0;
  --muted:    #4d7a63;
  --subtle:   #8ab09a;
  --blue:     #38bdf8;
  --purple:   #a78bfa;
  --orange:   #fb923c;
  --red:      #f87171;
}

* { box-sizing: border-box; margin: 0; padding: 0; }
html, body {
  height: 100%; overflow: hidden;
  font-family: 'Inter', -apple-system, sans-serif;
  background: var(--bg); color: var(--text);
}

/* BG pattern */
body::before {
  content: '';
  position: fixed; inset: 0; z-index: 0;
  background-image:
    linear-gradient(rgba(0,212,138,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,138,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
}

.app { display: flex; height: 100vh; position: relative; z-index: 1; }

/* ── SIDEBAR ── */
.sidebar {
  width: 220px; min-width: 220px;
  background: var(--surface);
  border-right: 1px solid var(--border);
  display: flex; flex-direction: column;
}
.brand {
  padding: 22px 18px 18px;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 10px;
}
.brand-icon {
  width: 36px; height: 36px;
  background: linear-gradient(135deg, #00d48a, #00a06a);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 18px; box-shadow: 0 0 16px rgba(0,212,138,0.4);
}
.brand-name { font-size: 14px; font-weight: 800; letter-spacing: -0.3px; }
.brand-name span { color: var(--teal); }
.brand-sub { font-size: 10px; color: var(--muted); margin-top: 1px; letter-spacing: .4px; }

.sec-label {
  padding: 16px 18px 6px;
  font-size: 10px; font-weight: 700;
  color: var(--muted); text-transform: uppercase; letter-spacing: 1px;
}
.acard {
  margin: 2px 8px; padding: 10px 12px;
  border-radius: 10px; cursor: pointer;
  display: flex; align-items: center; gap: 10px;
  border: 1px solid transparent; transition: all .2s;
}
.acard:hover { background: var(--card); border-color: var(--border2); }
.acard.active { background: var(--tealDim); border-color: var(--border); }
.avatar {
  width: 36px; height: 36px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 12px; font-weight: 800; color: #000; flex-shrink: 0;
}
.aname { font-size: 13px; font-weight: 600; }
.ameta { font-size: 11px; color: var(--muted); margin-top: 1px; }
.aviewing {
  font-size: 10px; color: var(--teal); margin-top: 3px;
  display: flex; align-items: center; gap: 4px;
}
.dot { width: 5px; height: 5px; border-radius: 50%; background: var(--teal); display: inline-block; }

.sfooter {
  margin-top: auto; padding: 14px 18px;
  border-top: 1px solid var(--border);
  font-size: 10px; color: var(--muted); line-height: 1.7;
}
.sfooter strong { color: var(--teal); }

/* ── MAIN ── */
.main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

.topbar {
  height: 60px; padding: 0 28px;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
  display: flex; align-items: center; gap: 14px; flex-shrink: 0;
}
.tb-av {
  width: 36px; height: 36px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 800; color: #000;
}
.tb-name { font-size: 16px; font-weight: 700; letter-spacing: -0.3px; }
.tb-det { font-size: 11px; color: var(--muted); margin-top: 1px; }
.tb-right { margin-left: auto; display: flex; align-items: center; gap: 10px; }
.legend { display: flex; gap: 14px; }
.leg { display: flex; align-items: center; gap: 5px; font-size: 11px; color: var(--subtle); }
.legdot { width: 8px; height: 8px; border-radius: 50%; }

.scroll {
  flex: 1; overflow-y: auto; padding: 22px 28px;
  display: flex; flex-direction: column; gap: 22px;
}
.scroll::-webkit-scrollbar { width: 4px; }
.scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

/* ── KPI ── */
.kpi-row { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; }
.kpi {
  background: var(--card); border: 1px solid var(--border2);
  border-radius: 14px; padding: 18px 20px;
  transition: border-color .2s, box-shadow .2s;
  position: relative; overflow: hidden;
}
.kpi::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--teal), transparent);
  opacity: 0.4;
}
.kpi:hover { border-color: var(--border); box-shadow: 0 0 20px rgba(0,212,138,0.06); }
.kpi-lbl { font-size: 10px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .8px; }
.kpi-val { font-size: 30px; font-weight: 900; margin: 6px 0 2px; letter-spacing: -2px; line-height: 1; }
.kpi-val span { font-size: 14px; font-weight: 600; letter-spacing: 0; opacity: .7; }
.kpi-sub { font-size: 11px; color: var(--subtle); }
.kpi-tag {
  display: inline-block; margin-top: 8px;
  font-size: 10px; font-weight: 600;
  padding: 3px 10px; border-radius: 20px;
}

/* ── SECTION HEADER ── */
.sec-hd { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
.sec-t { font-size: 14px; font-weight: 700; }
.sec-n { font-size: 11px; color: var(--muted); }
.sec-badge {
  margin-left: auto; font-size: 10px; font-weight: 600;
  color: var(--teal); background: var(--tealDim);
  padding: 3px 10px; border-radius: 20px;
  border: 1px solid var(--border); cursor: pointer;
}
.sec-badge:hover { background: rgba(0,212,138,0.25); }

/* ── EMG GRID ── */
.emg-wrap { background: var(--card); border: 1px solid var(--border2); border-radius: 14px; padding: 20px; }
.emg-tabs { display: flex; gap: 8px; margin-bottom: 18px; }
.emg-tab {
  padding: 6px 18px; border-radius: 8px; font-size: 12px; font-weight: 600;
  cursor: pointer; border: 1px solid var(--border2); color: var(--subtle);
  background: transparent; transition: all .2s;
}
.emg-tab.active {
  background: var(--tealDim); border-color: var(--border);
  color: var(--teal);
}
.emg-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; }

.muscle-card {
  background: rgba(255,255,255,0.025); border: 1px solid var(--border2);
  border-radius: 10px; padding: 12px;
  transition: border-color .2s, box-shadow .2s;
}
.muscle-card:hover { border-color: var(--border); box-shadow: 0 0 14px rgba(0,212,138,0.05); }
.muscle-name { font-size: 11px; font-weight: 700; color: var(--teal); margin-bottom: 2px; }
.muscle-full { font-size: 9px; color: var(--muted); margin-bottom: 10px; }

/* 3×3 grid */
.grid33 { display: grid; grid-template-columns: 18px repeat(3,1fr); gap: 2px; }
.g-corner { width: 18px; }
.g-speed {
  font-size: 8px; font-weight: 600; color: var(--muted);
  text-align: center; padding-bottom: 3px;
}
.g-incline {
  font-size: 8px; font-weight: 600; color: var(--muted);
  display: flex; align-items: center; justify-content: flex-end;
  padding-right: 4px;
}
.g-cell {
  height: 28px; border-radius: 4px;
  display: flex; align-items: center; justify-content: center;
  font-size: 9px; font-weight: 700;
  transition: transform .15s;
}
.g-cell:hover { transform: scale(1.08); }
.g-cell.excl { background: rgba(255,255,255,0.03); color: var(--muted); }

/* speed axis labels below */
.speed-axis { display: grid; grid-template-columns: 18px repeat(3,1fr); gap: 2px; margin-top: 3px; }
.speed-lbl { font-size: 8px; color: var(--muted); text-align: center; }

/* ── BAR CHARTS ── */
.charts-4 { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; }
.charts-3 { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; }
.ccrd {
  background: var(--card); border: 1px solid var(--border2);
  border-radius: 14px; padding: 16px 18px;
  transition: border-color .2s;
}
.ccrd:hover { border-color: var(--border); }
.ctit { font-size: 10px; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .7px; margin-bottom: 14px; }
.barchart {
  display: flex; align-items: flex-end; gap: 3px; height: 120px;
  border-bottom: 1px solid var(--border2); padding-bottom: 0;
}
.bar-col { display: flex; flex-direction: column; align-items: center; flex: 1; height: 100%; justify-content: flex-end; gap: 2px; }
.bar { width: 100%; border-radius: 3px 3px 0 0; min-height: 2px; transition: opacity .2s, box-shadow .2s; }
.bar:hover { opacity: .8; box-shadow: 0 0 8px currentColor; }
.barlbl { font-size: 7.5px; color: var(--muted); text-align: center; margin-top: 5px; line-height: 1.3; white-space: pre-line; }
.barval { font-size: 8px; color: var(--subtle); margin-bottom: 2px; }
.rom-bar:hover { opacity: 0.75; }
.rom-tooltip {
  position: fixed; z-index: 999;
  background: #0c1a14; border: 1px solid var(--border);
  color: var(--teal); font-size: 12px; font-weight: 700;
  padding: 5px 10px; border-radius: 8px;
  pointer-events: none; display: none;
  box-shadow: 0 4px 16px rgba(0,0,0,0.5);
}
</style>
</head>
<body>
<div class="rom-tooltip" id="rom-tooltip"></div>
<div class="app">

<aside class="sidebar">
  <div class="brand">
    <div class="brand-icon">🦅</div>
    <div>
      <div class="brand-name"><span>my</span>albatross</div>
      <div class="brand-sub">Athlete Analytics</div>
    </div>
  </div>
  <div class="sec-label">Athletes</div>
  <div id="roster"></div>
  <div class="sfooter">
    <strong>CREB · ETSEIB · UB</strong><br>
    Pilot study — 4 participants<br>
    10 conditions · 3 modalities
  </div>
</aside>

<main class="main">
  <div class="topbar">
    <div class="tb-av" id="tb-av"></div>
    <div>
      <div class="tb-name" id="tb-name"></div>
      <div class="tb-det" id="tb-det"></div>
    </div>
    <div class="tb-right">
      <div class="legend">
        <div class="leg"><div class="legdot" style="background:#007a52"></div>8 km/h</div>
        <div class="leg"><div class="legdot" style="background:#00d48a"></div>10 km/h</div>
        <div class="leg"><div class="legdot" style="background:#a8ffd8"></div>12 km/h</div>
      </div>
    </div>
  </div>

  <div class="scroll">

    <!-- KPIs -->
    <div class="kpi-row" id="kpi-row"></div>

    <!-- EMG -->
    <div>
      <div class="sec-hd">
        <span class="sec-t">EMG Muscle Activation</span>
        <span class="sec-n">% of walking baseline (5 km/h, 0%) · 3×3 grid: speed → incline ↑</span>
      </div>
      <div class="emg-wrap">
        <div class="emg-tabs">
          <div class="emg-tab active" onclick="setEMGSide('R',this)">Right Leg</div>
          <div class="emg-tab" onclick="setEMGSide('L',this)">Left Leg</div>
        </div>
        <div class="emg-grid" id="emg-grid"></div>
      </div>
    </div>

    <!-- Garmin -->
    <div>
      <div class="sec-hd">
        <span class="sec-t">Running Dynamics</span>
        <span class="sec-n">Garmin Forerunner 255 · 50 s steady-state window</span>
      </div>
      <div class="charts-4" id="garmin-grid"></div>
    </div>

    <!-- ROM -->
    <div>
      <div class="sec-hd">
        <span class="sec-t">Joint Range of Motion</span>
        <span class="sec-n">Xsens Awinda IMU · ROM = p95−p5 · solid = Right / transparent = Left</span>
      </div>
      <div class="charts-3" id="rom-grid"></div>
    </div>

  </div>
</main>
</div>

<script>
var DATA = """ + DATA_JS + r""";
var CONDS = ["8km/h_0%","8km/h_5%","8km/h_10%","10km/h_0%","10km/h_5%","10km/h_10%","12km/h_0%","12km/h_5%","12km/h_10%"];
var MLABELS = ["TA","MG","SO","VL","RF","ST","BF","GM"];
var MFULL = {"TA":"Tibialis Anterior","MG":"Med. Gastrocnemius","SO":"Soleus","VL":"Vastus Lateralis","RF":"Rectus Femoris","ST":"Semitendinosus","BF":"Biceps Femoris","GM":"Gluteus Medius"};
var JLABELS = ["Hip Flexion","Knee Flexion","Ankle Dorsiflexion"];
var ATHLETES = ["p2","p1","p3","p4"];
var active = "p2";
var emgSide = "R";

// Teal gradient: low = dim, high = bright teal
function heatBg(t) {
  // t=0: very dim dark teal, t=1: bright teal
  var r = Math.round(0   + t * 0);
  var g = Math.round(80  + t * (212-80));
  var b = Math.round(60  + t * (138-60));
  var a = (0.15 + t * 0.85).toFixed(2);
  return "rgba(" + r + "," + g + "," + b + "," + a + ")";
}
function heatTx(t) {
  return t > 0.5 ? "#000" : "#a0e8c8";
}
function rowMinMax(vals) {
  var nums = vals.filter(function(v){ return v !== null; });
  if (!nums.length) return {mn:0, mx:1};
  var mn = Math.min.apply(null,nums), mx = Math.max.apply(null,nums);
  if (mx === mn) mx = mn + 1;
  return {mn:mn, mx:mx};
}

function scol(cond) {
  if (cond.indexOf("8km") === 0)  return "#007a52";
  if (cond.indexOf("10km") === 0) return "#00d48a";
  return "#a8ffd8";
}
function scolA(cond, a) {
  var c = scol(cond);
  var r = parseInt(c.slice(1,3),16), g = parseInt(c.slice(3,5),16), b = parseInt(c.slice(5,7),16);
  return "rgba("+r+","+g+","+b+","+a+")";
}

function buildRoster() {
  var el = document.getElementById("roster"); el.innerHTML = "";
  ATHLETES.forEach(function(id) {
    var d = DATA[id]; var c = d.info.color;
    var div = document.createElement("div");
    div.className = "acard" + (id === active ? " active" : "");
    div.setAttribute("onclick", "sw('" + id + "')");
    div.innerHTML =
      '<div class="avatar" style="background:'+c+'">'+d.info.initials+'</div>' +
      '<div><div class="aname">'+d.info.name+'</div>' +
      '<div class="ameta">'+d.info.sex[0]+' · '+d.info.age+' y · '+d.info.weight+'</div>' +
      (id === active ? '<div class="aviewing"><span class="dot"></span> Viewing</div>' : '') +
      '</div>';
    el.appendChild(div);
  });
}

function buildTopbar(id) {
  var d = DATA[id];
  var av = document.getElementById("tb-av");
  av.style.background = d.info.color; av.textContent = d.info.initials;
  document.getElementById("tb-name").textContent = d.info.name;
  document.getElementById("tb-det").textContent =
    d.info.sex + " · " + d.info.age + " years · " + d.info.height + " · " + d.info.weight + " · 9 running conditions";
}

function buildKPIs(id) {
  var k = DATA[id].kpi; var c = DATA[id].info.color;
  var tc = k.top_cond.replace("km/h_"," km/h · ");
  var pc = k.peak_cond.replace("km/h_"," km/h · ");
  var ai = k.mean_asym;
  var aiCol = ai===null?"#4d7a63":ai<10?"#00d48a":ai<20?"#f59e0b":"#f87171";
  var aiTag = ai===null?"No data":ai<10?"Balanced":ai<20?"Monitor":"Asymmetric";
  var gct = k.gct_at_peak;
  var gctCol = gct===null?"#4d7a63":gct<230?"#00d48a":gct<270?"#f59e0b":"#f87171";
  var gctTag = gct===null?"No data":gct<230?"Efficient":gct<270?"Normal":"Heavy";

  // Speed-activation correlation colour
  var r = k.speed_act_corr;
  var rCol = r===null?"#4d7a63":r>=0.7?"#00d48a":r>=0.4?"#f59e0b":"#f87171";
  var rTag = r===null?"No data":r>=0.7?"Strong":"Moderate";

  // Cadence at peak
  var cad = k.cadence_at_peak;

  // Peak condition label
  var pc = k.peak_cond.replace("km/h_"," km/h · ");

  document.getElementById("kpi-row").innerHTML =
    '<div class="kpi">' +
      '<div class="kpi-lbl">Peak Heart Rate</div>' +
      '<div class="kpi-val" style="color:#f87171">'+(k.peak_hr||"--")+'<span>bpm</span></div>' +
      '<div class="kpi-sub">p2 recorded across all conditions</div>' +
      '<div class="kpi-tag" style="background:#f8717122;color:#f87171">Cardiovascular peak</div>' +
    '</div>' +
    '<div class="kpi">' +
      '<div class="kpi-lbl">Peak Condition</div>' +
      '<div class="kpi-val" style="color:'+c+';font-size:18px;margin-top:6px;letter-spacing:-0.5px">'+pc+'</div>' +
      '<div class="kpi-sub">highest mean activation</div>' +
      '<div class="kpi-tag" style="background:'+c+'22;color:'+c+'">Most demanding</div>' +
    '</div>' +
    '<div class="kpi">' +
      '<div class="kpi-lbl">Cadence at Peak</div>' +
      '<div class="kpi-val" style="color:#38bdf8">'+(cad||"--")+'<span>spm</span></div>' +
      '<div class="kpi-sub">steps per min · '+pc+'</div>' +
      '<div class="kpi-tag" style="background:#38bdf822;color:#38bdf8">Stride frequency</div>' +
    '</div>' +
    '<div class="kpi">' +
      '<div class="kpi-lbl">Speed–Activation Correlation</div>' +
      '<div class="kpi-val" style="color:'+rCol+'">'+(r!==null?r.toFixed(2):"--")+'</div>' +
      '<div class="kpi-sub">Pearson r · speed vs mean EMG</div>' +
      '<div class="kpi-tag" style="background:'+rCol+'22;color:'+rCol+'">'+rTag+' speed response</div>' +
    '</div>';
}

function buildEMG(id, side) {
  var data = DATA[id].heatmap[side];
  var el = document.getElementById("emg-grid"); el.innerHTML = "";

  // 3x3 layout: rows = incline (10%,5%,0% top to bottom), cols = speed (8,10,12)
  // CONDS order: [8_0, 8_5, 8_10, 10_0, 10_5, 10_10, 12_0, 12_5, 12_10]
  // Grid cell [inc_row][speed_col]:
  //   inc=10% (row0): indices 2,5,8
  //   inc=5%  (row1): indices 1,4,7
  //   inc=0%  (row2): indices 0,3,6
  var idxMap = [
    [2,5,8],  // 10%
    [1,4,7],  // 5%
    [0,3,6],  // 0%
  ];
  var incLabels = ["10%","5%","0%"];
  var speedLabels = ["8","10","12"];

  MLABELS.forEach(function(label) {
    var entry = data[label]; var vals = entry.vals;
    var mm = rowMinMax(vals);

    var card = document.createElement("div");
    card.className = "muscle-card";

    var html = '<div class="muscle-name">'+label+'</div>' +
               '<div class="muscle-full">'+MFULL[label]+'</div>';

    // grid header row (speed labels)
    html += '<div class="grid33">';
    html += '<div class="g-corner"></div>';
    speedLabels.forEach(function(s){ html += '<div class="g-speed">'+s+'</div>'; });

    // data rows
    for (var ri = 0; ri < 3; ri++) {
      html += '<div class="g-incline">'+incLabels[ri]+'</div>';
      for (var ci = 0; ci < 3; ci++) {
        var idx = idxMap[ri][ci];
        var v = vals[idx];
        if (v === null) {
          html += '<div class="g-cell excl">—</div>';
        } else {
          var t = (v - mm.mn) / (mm.mx - mm.mn);
          var bg = heatBg(t); var tx = heatTx(t);
          html += '<div class="g-cell" style="background:'+bg+';color:'+tx+'" title="'+Math.round(v)+'% baseline">'+Math.round(v)+'</div>';
        }
      }
    }
    html += '</div>';

    card.innerHTML = html;
    el.appendChild(card);
  });
}

function setEMGSide(side, tabEl) {
  emgSide = side;
  document.querySelectorAll(".emg-tab").forEach(function(t){ t.classList.remove("active"); });
  tabEl.classList.add("active");
  buildEMG(active, side);
}

function buildGarmin(id) {
  var g = DATA[id].garmin;
  var el = document.getElementById("garmin-grid"); el.innerHTML = "";
  var metrics = [
    {key:"hr",     title:"Heart Rate",           unit:"bpm"},
    {key:"cadence",title:"Cadence",              unit:"spm"},
    {key:"gct",    title:"Ground Contact Time",  unit:"ms"},
    {key:"vo",     title:"Vertical Oscillation", unit:"mm"},
  ];
  metrics.forEach(function(m) {
    var vals = g[m.key];
    var nums = vals.filter(function(v){ return v!==null; });
    var mx = nums.length ? Math.max.apply(null,nums)*1.12 : 100;
    var div = document.createElement("div"); div.className = "ccrd";
    var html = '<div class="ctit">'+m.title+' ('+m.unit+')</div><div class="barchart">';
    for (var i=0; i<CONDS.length; i++) {
      var v=vals[i]; var cond=CONDS[i];
      var h = v!==null ? Math.max(Math.round((v/mx)*100),2) : 0;
      var col = scol(cond); var colA = scolA(cond,0.75);
      var lbl = cond.replace("km/h_","\n");
      var valStr = v!==null ? v.toFixed(0) : "";
      html += '<div class="bar-col">' +
        '<div class="barval">'+valStr+'</div>' +
        '<div class="bar" title="'+cond+': '+(v||"N/A")+'" style="height:'+h+'px;background:'+colA+';border:1px solid '+col+'"></div>' +
        '<div class="barlbl">'+lbl+'</div>' +
        '</div>';
    }
    html += '</div>';
    div.innerHTML = html;
    el.appendChild(div);
  });
}

function buildROM(id) {
  var rom = DATA[id].rom;
  var el = document.getElementById("rom-grid"); el.innerHTML = "";
  JLABELS.forEach(function(label) {
    var r = rom[label];
    var allvals = r.R.concat(r.L).filter(function(v){ return v!==null&&v>0; });
    var mx = allvals.length ? Math.max.apply(null,allvals)*1.12 : 100;
    var div = document.createElement("div"); div.className = "ccrd";
    var html = '<div class="ctit">'+label+' ROM (°)</div><div class="barchart">';
    for (var i=0; i<CONDS.length; i++) {
      var vR=r.R[i], vL=r.L[i], cond=CONDS[i];
      var hR=vR!==null&&vR>0?Math.max(Math.round((vR/mx)*100),2):0;
      var hL=vL!==null&&vL>0?Math.max(Math.round((vL/mx)*100),2):0;
      var col=scol(cond); var lbl=cond.replace("km/h_","\n");
      var rLabel = vR!==null ? vR.toFixed(1)+'°' : 'N/A';
      var lLabel = vL!==null ? vL.toFixed(1)+'°' : 'N/A';
      html += '<div style="display:flex;flex-direction:column;align-items:center;flex:1;height:100%;justify-content:flex-end;gap:2px;position:relative" class="rom-col">' +
        '<div style="display:flex;align-items:flex-end;gap:1px;width:100%;justify-content:center">' +
        '<div class="rom-bar" data-val="R: '+rLabel+'" style="width:44%;height:'+hR+'px;background:'+scolA(cond,0.85)+';border-radius:3px 3px 0 0;border:1px solid '+col+';cursor:pointer;transition:opacity .15s;position:relative"></div>' +
        '<div class="rom-bar" data-val="L: '+lLabel+'" style="width:44%;height:'+hL+'px;background:'+scolA(cond,0.25)+';border-radius:3px 3px 0 0;border:1px solid '+scolA(cond,0.5)+';cursor:pointer;transition:opacity .15s;position:relative"></div>' +
        '</div><div class="barlbl">'+lbl+'</div></div>';
    }
    html += '</div>';
    div.innerHTML = html;
    el.appendChild(div);
  });
}

function sw(id) {
  active = id; emgSide = "R";
  document.querySelectorAll(".emg-tab").forEach(function(t,i){ t.classList.toggle("active",i===0); });
  buildRoster(); buildTopbar(id); buildKPIs(id); buildEMG(id, emgSide); buildGarmin(id); buildROM(id);
}

// ROM tooltips
var tip = document.getElementById("rom-tooltip");
document.addEventListener("mouseover", function(e) {
  if (e.target.classList.contains("rom-bar")) {
    tip.textContent = e.target.getAttribute("data-val");
    tip.style.display = "block";
  }
});
document.addEventListener("mousemove", function(e) {
  if (e.target.classList.contains("rom-bar")) {
    tip.style.left = (e.clientX + 12) + "px";
    tip.style.top  = (e.clientY - 28) + "px";
  }
});
document.addEventListener("mouseout", function(e) {
  if (e.target.classList.contains("rom-bar")) {
    tip.style.display = "none";
  }
});

sw("p2");
</script>
</body>
</html>"""

out = os.path.join(OUT_DIR, "athlete_dashboard_v2.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(HTML)
print("Done:", out)
