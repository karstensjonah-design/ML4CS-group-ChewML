#!/usr/bin/env python3
"""
Live-Essklassifikation — Hierarchisches 2-Stufen-Modell
========================================================

Stufe 1 : Still  vs.  Essen
Stufe 2 : Apfel / Kaugummi / Skyr / Essen (unbekannt)

Pipeline: 25-s-Fenster → Movement Exclusion (k=5) → Random Forest

Start: python classifier_app.py
Stopp: Ctrl+C
"""

# ── Stdlib ────────────────────────────────────────────────────────────────────
import csv
import json
import socket
import sys
import threading
import time
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── Scientific ─────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# ── Rich (required) ────────────────────────────────────────────────────────────
try:
    from rich import box
    from rich.columns import Columns
    from rich.console import Console
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("Bitte 'rich' installieren:  pip install rich")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════════════════
#  KONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR      = Path(__file__).parent.parent
DATA_DIR      = BASE_DIR / "data" / "raw"
PORT          = 8000

FS            = 50.0    # Abtastrate Hz
TRIM_SECS     = 2       # Sekunden am Anfang/Ende abschneiden (Training)
WINDOW_SECS   = 10.0    # Fensterlänge für Klassifikation
MIN_WINDOW    = 8.0     # Mindestpuffer vor erster Klassifikation
SLIDE_SECS    = 2.0     # Klassifikation alle N Sekunden
K_MOV         = 5       # Movement Exclusion: k × Median
CONF_S1_MIN   = 0.95    # Stufe 1: Mindest-Konfidenz für "Essen" (sonst → Still)
MOVEMENT_EXCL = True    # Movement Exclusion standardmäßig aktiv

CLASSES_RAW   = ["Apfel", "Kaugummi", "Skyr", "Still", "Essen"]
TO_COARSE     = {c: ("Still" if c == "Still" else "Essen") for c in CLASSES_RAW}

# Stage-1 Features (RFECV-selektiert, k=5 → 100 % LOO-Accuracy)
FEATURES_S1   = ["stillness_ratio", "magnitude_max", "lin_y_mean", "lin_y_std", "yaw_mean"]

# Pflichtfelder im eingehenden Sensor-Stream
REQUIRED_KEYS = [
    "accelerationX", "accelerationY", "accelerationZ",
    "rotationRateX", "rotationRateY", "rotationRateZ",
    "pitch", "roll", "yaw",
]

# Log-Datei mit Zeitstempel pro App-Start
_log_start = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE    = Path(__file__).parent / f"classification_log_{_log_start}.csv"
LOG_FIELDS  = ["time", "label", "stage", "confidence", "s1_conf",
               "samples_clean"] + FEATURES_S1

def _write_log(row: dict) -> None:
    exists = LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)

_HTML_PAGE = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Live-Essklassifikation</title>
<style>
:root {
  --still:    #2ecc71; --apfel:    #e74c3c;
  --kaugummi: #3498db; --skyr:     #f39c12; --essen: #9b59b6;
  --bg: #0d0d1a; --card: #16162a; --border: #2a2a42;
  --text: #e0e0f0; --dim: #5a5a7a;
}
* { margin:0; padding:0; box-sizing:border-box; }
body { background:var(--bg); color:var(--text);
  font-family:'Segoe UI',system-ui,sans-serif;
  min-height:100vh; padding:1.5rem; }
h1 { text-align:center; font-size:.85rem; color:var(--dim);
  letter-spacing:.25em; text-transform:uppercase; margin-bottom:1.5rem; }
.grid { max-width:960px; margin:0 auto;
  display:grid; grid-template-columns:1fr 1fr; gap:1rem; }
.card { background:var(--card); border:1px solid var(--border);
  border-radius:14px; padding:1.4rem; }
.card-title { font-size:.7rem; text-transform:uppercase;
  letter-spacing:.15em; color:var(--dim); margin-bottom:1rem; }
.full { grid-column:1/-1; }

/* ── Ergebnis ── */
#result-card { text-align:center; padding:2rem; transition:border-color .4s; }
#label { font-size:2.6rem; font-weight:700; letter-spacing:.04em;
  transition:color .3s; margin-bottom:.4rem; }
.bar-wrap { background:#2a2a42; border-radius:99px; height:6px;
  margin:.9rem auto; max-width:280px; overflow:hidden; }
.bar { height:100%; border-radius:99px; transition:width .4s,background .3s; }
#confidence { font-size:1rem; color:var(--dim); }
#meta { font-size:.82rem; color:var(--dim); margin-top:.5rem; }

/* ── Buffer ── */
.buf-bar-wrap { background:#2a2a42; border-radius:99px; height:10px;
  overflow:hidden; margin:.5rem 0 .6rem; }
.buf-bar { height:100%; border-radius:99px; background:var(--still);
  transition:width .3s; }
.buf-info { display:flex; justify-content:space-between;
  font-size:.82rem; color:var(--dim); }
.ok   { color:var(--still); }
.warn { color:var(--skyr); }
.err  { color:var(--apfel); font-weight:600; }

/* ── Stats ── */
.stats-grid { display:grid; grid-template-columns:1fr 1fr; gap:.8rem; }
.stat { text-align:center; }
.stat-val { font-size:1.55rem; font-weight:700; }
.stat-lbl { font-size:.72rem; color:var(--dim); margin-top:.2rem; }

/* ── History ── */
table { width:100%; border-collapse:collapse; font-size:.88rem; }
th { text-align:left; color:var(--dim); font-weight:500;
  padding:.35rem .7rem; border-bottom:1px solid var(--border);
  font-size:.7rem; text-transform:uppercase; letter-spacing:.1em; }
td { padding:.45rem .7rem; border-bottom:1px solid rgba(255,255,255,.04); }
tr:first-child td { font-weight:600; }
.dot { display:inline-block; width:9px; height:9px;
  border-radius:50%; margin-right:.45rem; vertical-align:middle; }

@keyframes pop {
  0%,100% { transform:scale(1); }
  50%      { transform:scale(1.06); }
}
.tab-btn { padding:.4rem 1.2rem;border-radius:99px;border:1px solid var(--border);
  background:var(--card);color:var(--dim);cursor:pointer;font-size:.85rem;transition:all .2s; }
.tab-btn.active { background:var(--text);color:var(--bg);border-color:var(--text);font-weight:600; }
.cm-table { border-collapse:collapse;font-size:.82rem;width:100%; }
.cm-table th,.cm-table td { padding:.35rem .5rem;text-align:center; }
.cm-table th { color:var(--dim);font-weight:500;font-size:.72rem; }
.cm-val { border-radius:5px;font-weight:700; }
.met-table { width:100%;border-collapse:collapse;font-size:.85rem; }
.met-table th { text-align:left;color:var(--dim);font-size:.72rem;padding:.3rem .5rem;
  border-bottom:1px solid var(--border);text-transform:uppercase;letter-spacing:.08em; }
.met-table td { padding:.4rem .5rem;border-bottom:1px solid rgba(255,255,255,.04); }
.pop { animation:pop .35s ease; }
</style>
</head>
<body>
<h1>&#127860; Live-Essklassifikation</h1>

<div style="display:flex;justify-content:center;gap:.5rem;margin-bottom:1.5rem">
  <button class="tab-btn active" id="btn-live"  onclick="showTab('live')">Live</button>
  <button class="tab-btn"        id="btn-model" onclick="showTab('model')">Modell &amp; Daten</button>
</div>

<div id="tab-live">
<div class="grid">

  <div class="card full" id="result-card">
    <div class="card-title">Aktuelle Klassifikation</div>
    <div id="label" style="color:var(--dim)">Warte auf Sensordaten …</div>
    <div class="bar-wrap"><div class="bar" id="conf-bar" style="width:0%"></div></div>
    <div id="confidence">—</div>
    <div id="meta">—</div>
  </div>

  <div class="card">
    <div class="card-title">Datenpuffer</div>
    <div class="buf-bar-wrap"><div class="buf-bar" id="buf-bar" style="width:0%"></div></div>
    <div class="buf-info">
      <span id="buf-text">0 / 10 s</span>
      <span id="signal" class="warn">Sammle …</span>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Statistik</div>
    <div class="stats-grid">
      <div class="stat"><div class="stat-val" id="s-total">0</div>
        <div class="stat-lbl">Samples gesamt</div></div>
      <div class="stat"><div class="stat-val" id="s-buf">0</div>
        <div class="stat-lbl">Im Buffer</div></div>
      <div class="stat"><div class="stat-val" id="s-clf">0</div>
        <div class="stat-lbl">Klassifikationen</div></div>
      <div class="stat"><div class="stat-val" id="s-hz">—</div>
        <div class="stat-lbl">Hz</div></div>
    </div>
    <div style="margin-top:1rem;padding-top:.9rem;border-top:1px solid var(--border);display:flex;align-items:center;gap:.75rem">
      <span style="font-size:.8rem;color:var(--dim)">Movement Exclusion</span>
      <button id="mov-btn" onclick="toggleMovement()"
        style="padding:.3rem .9rem;border-radius:99px;border:none;cursor:pointer;
               font-size:.8rem;font-weight:600;transition:background .2s">
      </button>
    </div>
  </div>

  <div class="card full" id="session-card">
    <div class="card-title">Mahlzeiten-Session</div>
    <div id="sess-status" style="font-size:1.4rem;font-weight:700;margin-bottom:.8rem;color:var(--dim)">Kein Essen erkannt</div>
    <div id="sess-votes" style="margin-bottom:.8rem"></div>
    <div id="sess-meta" style="font-size:.82rem;color:var(--dim)"></div>
  </div>

  <div class="card full">
    <div class="card-title">Abgeschlossene Sessions</div>
    <table><thead><tr>
      <th>Start</th><th>Erkannt als</th><th>Stimmen</th><th>Dauer</th>
    </tr></thead>
    <tbody id="sess-hist"></tbody></table>
  </div>

  <div class="card full">
    <div class="card-title">Klassifikations-Verlauf</div>
    <table><thead><tr>
      <th>Zeit</th><th>Klasse</th><th>Signal (Magnitude)</th><th>Konfidenz</th><th>ME</th><th>Stufe</th>
    </tr></thead>
    <tbody id="hist"></tbody></table>
  </div>

</div><!-- grid -->
</div><!-- tab-live -->

<div id="tab-model" style="display:none;max-width:960px;margin:0 auto">

  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem">
    <div id="stats-status" style="font-size:.85rem;color:var(--dim)">Berechne LOO-CV …</div>
    <button onclick="recompute()"
      style="padding:.3rem .9rem;border-radius:99px;border:1px solid var(--border);
             background:var(--card);color:var(--text);cursor:pointer;font-size:.8rem">
      &#8635; Neu berechnen
    </button>
  </div>

  <!-- Stichproben -->
  <div class="card" style="margin-bottom:1rem">
    <div class="card-title">Trainingsdaten</div>
    <div id="samples-row" style="display:flex;gap:1.5rem;flex-wrap:wrap"></div>
  </div>

  <!-- Konfusionsmatrizen -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem">
    <div class="card">
      <div class="card-title">Stufe 1 — Still vs. Essen <span id="acc1" style="color:var(--still)"></span></div>
      <div id="cm1"></div>
    </div>
    <div class="card">
      <div class="card-title">Stufe 2 — Feinklassifikation <span id="acc2" style="color:var(--still)"></span></div>
      <div id="cm2"></div>
    </div>
  </div>

  <!-- Metriken -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem">
    <div class="card">
      <div class="card-title">Metriken — Stufe 1</div>
      <div id="rep1"></div>
    </div>
    <div class="card">
      <div class="card-title">Metriken — Stufe 2</div>
      <div id="rep2"></div>
    </div>
  </div>

</div><!-- tab-model -->

<script>
const C = {Still:'#2ecc71',Apfel:'#e74c3c',Kaugummi:'#3498db',Skyr:'#f39c12',Essen:'#9b59b6'};
const L = {Still:'STILL',Apfel:'APFEL',Kaugummi:'KAUGUMMI',Skyr:'SKYR',Essen:'ESSEN (unbekannt)'};
let lastLabel=null, prevTotal=0, hzBuf=[], clfCount=0;
let activeTab='live';

function showTab(t){
  activeTab=t;
  document.getElementById('tab-live').style.display  = t==='live'  ? '' : 'none';
  document.getElementById('tab-model').style.display = t==='model' ? '' : 'none';
  document.getElementById('btn-live').classList.toggle('active',  t==='live');
  document.getElementById('btn-model').classList.toggle('active', t==='model');
  if(t==='model') loadModelStats();
}

function cmColor(v, max){
  if(max===0) return 'transparent';
  const a = Math.round(v/max*220);
  return `rgba(52,152,219,${(a/255).toFixed(2)})`;
}
function renderCM(id, cm, labels){
  const max = Math.max(...cm.flat());
  let html = '<table class="cm-table"><thead><tr><th></th>';
  labels.forEach(l=>html+=`<th>${l}</th>`);
  html+='</tr></thead><tbody>';
  cm.forEach((row,i)=>{
    html+=`<tr><th style="text-align:right;color:var(--dim)">${labels[i]}</th>`;
    row.forEach((v,j)=>{
      const bg = i===j ? cmColor(v,max) : (v>0?'rgba(231,76,60,0.3)':'transparent');
      html+=`<td><div class="cm-val" style="background:${bg};padding:.3rem .4rem">${v}</div></td>`;
    });
    html+='</tr>';
  });
  html+='</tbody></table>';
  document.getElementById(id).innerHTML=html;
}
function renderReport(id, report, labels){
  let html='<table class="met-table"><thead><tr>'
    +'<th>Klasse</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th>'
    +'</tr></thead><tbody>';
  labels.forEach(l=>{
    const r=report[l]||{};
    const f1=r['f1-score']||0;
    const col=f1>=0.9?'var(--still)':f1>=0.7?'var(--skyr)':'var(--apfel)';
    html+=`<tr>
      <td style="color:${C[l]||'var(--text)'};font-weight:600">${l}</td>
      <td>${((r.precision||0)*100).toFixed(1)}%</td>
      <td>${((r.recall||0)*100).toFixed(1)}%</td>
      <td style="color:${col};font-weight:700">${(f1*100).toFixed(1)}%</td>
      <td style="color:var(--dim)">${r.support||0}</td>
    </tr>`;
  });
  const ma=report['macro avg']||{};
  html+=`<tr style="border-top:1px solid var(--border);color:var(--dim)">
    <td>Macro Avg</td>
    <td>${((ma.precision||0)*100).toFixed(1)}%</td>
    <td>${((ma.recall||0)*100).toFixed(1)}%</td>
    <td style="font-weight:700">${((ma['f1-score']||0)*100).toFixed(1)}%</td>
    <td></td></tr>`;
  html+='</tbody></table>';
  document.getElementById(id).innerHTML=html;
}
async function loadModelStats(){
  const d = await fetch('/api/model_stats').then(r=>r.json()).catch(()=>null);
  if(!d) return;
  const el=document.getElementById('stats-status');
  if(d.status==='pending'||d.status==='running'){
    el.textContent='Berechne LOO-CV … (kann einige Minuten dauern)'; return;
  }
  if(d.status==='error'){
    el.textContent='Fehler: '+d.message; return;
  }
  el.textContent='Zuletzt berechnet: '+d.computed_at;

  // Samples
  const sr=document.getElementById('samples-row');
  sr.innerHTML=Object.entries(d.n_samples).map(([k,v])=>{
    const c=C[k]||'var(--text)';
    return `<div style="text-align:center">
      <div style="font-size:1.4rem;font-weight:700;color:${c}">${v}</div>
      <div style="font-size:.75rem;color:var(--dim)">${k}</div>
    </div>`;
  }).join('');

  // Stufe 1
  document.getElementById('acc1').textContent = '— '+(d.stage1.accuracy*100).toFixed(1)+'%';
  renderCM('cm1', d.stage1.cm, d.stage1.labels);
  renderReport('rep1', d.stage1.report, d.stage1.labels);

  // Stufe 2
  document.getElementById('acc2').textContent = '— '+(d.stage2.accuracy*100).toFixed(1)+'%';
  renderCM('cm2', d.stage2.cm, d.stage2.labels);
  renderReport('rep2', d.stage2.report, d.stage2.labels);
}
async function recompute(){
  document.getElementById('stats-status').textContent='Berechne …';
  await fetch('/api/recompute');
  setTimeout(()=>{ if(activeTab==='model') loadModelStats(); }, 3000);
}

function renderSparkline(mag, excl, meActive, W=180, H=36){
  if(!mag||mag.length<2) return '<span style="color:var(--dim);font-size:.75rem">—</span>';
  const n   = mag.length;
  const max = Math.max(...mag) || 1;
  const min = Math.min(...mag);
  const range = max - min || 1;
  const pad = 2;

  // Polyline-Punkte
  const pts = mag.map((v,i)=>{
    const x = pad + (i/(n-1))*(W-2*pad);
    const y = H - pad - ((v-min)/range)*(H-2*pad);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');

  // Herausgefilterte Segmente als rote Rechtecke
  let rects = '';
  if(meActive && excl && excl.length===n){
    let inExcl=false, startI=0;
    for(let i=0;i<=n;i++){
      const ex = i<n ? excl[i] : false;
      if(ex && !inExcl){ startI=i; inExcl=true; }
      else if(!ex && inExcl){
        const x1 = pad+(startI/(n-1))*(W-2*pad);
        const x2 = pad+((i-1)/(n-1))*(W-2*pad);
        rects += `<rect x="${x1.toFixed(1)}" y="0" width="${(x2-x1).toFixed(1)}" height="${H}"
          fill="rgba(231,76,60,0.35)" rx="1"/>`;
        inExcl=false;
      }
    }
  }

  return `<svg width="${W}" height="${H}" style="display:block;overflow:visible">
    <rect width="${W}" height="${H}" rx="4" fill="rgba(255,255,255,0.03)"/>
    ${rects}
    <polyline points="${pts}" fill="none" stroke="#4e79a7" stroke-width="1.3"
      stroke-linejoin="round"/>
  </svg>`;
}

function updateMovBtn(active){
  const b = document.getElementById('mov-btn');
  if(active){
    b.textContent='AN'; b.style.background='var(--still)'; b.style.color='#000';
  } else {
    b.textContent='AUS'; b.style.background='var(--border)'; b.style.color='var(--dim)';
  }
}
async function toggleMovement(){
  const r = await fetch('/api/toggle_movement');
  const d = await r.json();
  updateMovBtn(d.movement_excl);
}

async function poll(){
  try{
    const d = await fetch('/api/status').then(r=>r.json());

    // Buffer
    const pct = Math.min(d.buffered_secs/d.window_secs*100,100);
    document.getElementById('buf-bar').style.width = pct+'%';
    document.getElementById('buf-text').textContent =
      d.buffered_secs.toFixed(1)+' / '+d.window_secs+' s';
    const sig = document.getElementById('signal');
    if(d.stale_secs>5){
      sig.textContent='KEIN SIGNAL '+d.stale_secs.toFixed(0)+'s'; sig.className='err';
    } else if(pct>=80){
      sig.textContent='Signal OK'; sig.className='ok';
    } else {
      sig.textContent='Sammle …'; sig.className='warn';
    }

    // Movement-Exclusion-Button sync
    if(d.movement_excl !== undefined) updateMovBtn(d.movement_excl);

    // Stats
    document.getElementById('s-total').textContent = d.total_received.toLocaleString('de');
    document.getElementById('s-buf').textContent   = d.buffer_count;
    document.getElementById('s-clf').textContent   = d.history_count;
    const delta = d.total_received - prevTotal; prevTotal = d.total_received;
    hzBuf.push(delta); if(hzBuf.length>5) hzBuf.shift();
    const hz = hzBuf.reduce((a,b)=>a+b,0)/hzBuf.length;
    document.getElementById('s-hz').textContent = hz.toFixed(0);

    // Result
    if(!d.result) return;
    const res=d.result;
    const color=C[res.label]||'#e0e0f0';
    const lbl  =document.getElementById('label');
    lbl.textContent = '● '+(L[res.label]||res.label);
    lbl.style.color = color;
    if(res.label!==lastLabel){
      lbl.classList.remove('pop');
      void lbl.offsetWidth; // reflow
      lbl.classList.add('pop');
      lastLabel=res.label;
    }
    const cpct=Math.round(res.conf*100);
    document.getElementById('conf-bar').style.width    = cpct+'%';
    document.getElementById('conf-bar').style.background = color;
    document.getElementById('confidence').textContent  = 'Konfidenz: '+cpct+'%';
    document.getElementById('result-card').style.borderColor = color+'55';
    const meta=['Stufe '+res.stage, res.time];
    if(res.stage===2&&res.s1_conf) meta.push('S1: '+Math.round(res.s1_conf*100)+'%');
    document.getElementById('meta').textContent = meta.join('  ·  ');

    // History
    if(d.history&&d.history.length){
      document.getElementById('hist').innerHTML = d.history.map(r=>{
        const c   = C[r.label]||'#e0e0f0';
        const svg = renderSparkline(r.mag||[], r.excl||[], r.me);
        const me  = r.me
          ? '<span style="font-size:.7rem;padding:.15rem .4rem;border-radius:99px;background:rgba(46,204,113,.2);color:#2ecc71">AN</span>'
          : '<span style="font-size:.7rem;padding:.15rem .4rem;border-radius:99px;background:var(--border);color:var(--dim)">AUS</span>';
        return `<tr>
          <td style="color:var(--dim);white-space:nowrap">${r.time}</td>
          <td><span class="dot" style="background:${c}"></span>${L[r.label]||r.label}</td>
          <td>${svg}</td>
          <td>${Math.round(r.conf*100)}%</td>
          <td>${me}</td>
          <td style="color:var(--dim)">Stufe ${r.stage}</td></tr>`;
      }).join('');
    }

    // Session
    if(!d.session) return;
    const s = d.session;
    const sc = document.getElementById('session-card');
    const st = document.getElementById('sess-status');
    const sv = document.getElementById('sess-votes');
    const sm = document.getElementById('sess-meta');

    if(s.state === 'idle' && s.essen_streak === 0){
      st.textContent = 'Kein Essen erkannt';
      st.style.color = 'var(--dim)';
      sc.style.borderColor = 'var(--border)';
      sv.innerHTML = '';
      sm.textContent = '';
    } else if(s.state === 'idle' && s.essen_streak > 0){
      const pct = Math.round(s.essen_streak / s.start_needed * 100);
      st.innerHTML = `Erkenne Essen … <span style="color:var(--skyr)">${s.essen_streak}/${s.start_needed}</span>`;
      st.style.color = 'var(--skyr)';
      sc.style.borderColor = 'var(--skyr)55';
      sv.innerHTML = `<div class="bar-wrap" style="max-width:200px;margin:0">
        <div class="bar" style="width:${pct}%;background:var(--skyr)"></div></div>`;
      sm.textContent = '';
    } else if(s.state === 'active'){
      const best  = s.best_label;
      const color = C[best] || '#e0e0f0';
      st.innerHTML = `<span class="dot" style="background:${color};width:12px;height:12px"></span> ${L[best]||best}`;
      st.style.color = color;
      sc.style.borderColor = color+'66';
      const total = Object.values(s.votes).reduce((a,b)=>a+b,0)||1;
      const foods = ['Apfel','Kaugummi','Skyr','Essen'];
      sv.innerHTML = foods.map(f=>{
        const v   = s.votes[f]||0;
        const pct = Math.round(v/total*100);
        const c   = C[f]||'#aaa';
        return `<div style="display:flex;align-items:center;gap:.5rem;margin:.25rem 0;font-size:.85rem">
          <span style="width:70px;color:var(--dim)">${f}</span>
          <div style="flex:1;background:#2a2a42;border-radius:99px;height:7px;overflow:hidden">
            <div style="width:${pct}%;background:${c};height:100%;border-radius:99px;transition:width .4s"></div>
          </div>
          <span style="width:30px;text-align:right;color:${c};font-weight:600">${v}</span>
        </div>`;
      }).join('');
      sm.textContent = `Seit ${d.session.duration_s}s  ·  ${total} Stimmen  ·  ${Math.round(s.vote_conf*100)}% Mehrheit`;
    }

    // Abgeschlossene Sessions
    if(s.completed&&s.completed.length){
      document.getElementById('sess-hist').innerHTML = s.completed.map(r=>{
        const c  = C[r.label]||'#e0e0f0';
        const vt = Object.entries(r.votes||{}).map(([k,v])=>`${k}:${v}`).join(', ');
        return `<tr>
          <td style="color:var(--dim)">${r.start}</td>
          <td><span class="dot" style="background:${c}"></span>${L[r.label]||r.label}</td>
          <td style="color:var(--dim);font-size:.8rem">${vt}</td>
          <td style="color:var(--dim)">${r.duration_s}s</td></tr>`;
      }).join('');
    }
  }catch(e){}
  setTimeout(poll, 1000);
}
poll();
</script>
</body>
</html>"""

LABEL_STYLE = {
    "Still":    ("green",   "●  STILL"),
    "Apfel":    ("red",     "●  APFEL"),
    "Kaugummi": ("blue",    "●  KAUGUMMI"),
    "Skyr":     ("yellow",  "●  SKYR"),
    "Essen":    ("magenta", "●  ESSEN (unbekannt)"),
}

# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE-EXTRAKTION
# ══════════════════════════════════════════════════════════════════════════════

def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lin_x"]     = df["accelerationX"]
    df["lin_y"]     = df["accelerationY"]
    df["lin_z"]     = df["accelerationZ"]
    df["magnitude"] = np.sqrt(df["lin_x"]**2 + df["lin_y"]**2 + df["lin_z"]**2)
    return df


def _movement_exclusion(df: pd.DataFrame, k: float = K_MOV) -> pd.DataFrame:
    thr      = max(0.02, k * df["magnitude"].median())
    roll_max = df["magnitude"].rolling(50, center=True, min_periods=1).max()
    return df[roll_max <= thr].reset_index(drop=True)


def extract_features(df: pd.DataFrame) -> dict:
    f = {}
    for col in ["lin_x", "lin_y", "lin_z", "magnitude"]:
        f[f"{col}_mean"] = df[col].mean()
        f[f"{col}_std"]  = df[col].std()
        f[f"{col}_max"]  = df[col].abs().max()
    f["stillness_ratio"] = (df["magnitude"] < 0.02).mean()
    f["movement_events"] = int((df["magnitude"] > df["magnitude"].quantile(0.75)).sum())
    for col in ["rotationRateX", "rotationRateY", "rotationRateZ"]:
        f[f"{col}_mean"] = df[col].mean()
        f[f"{col}_std"]  = df[col].std()
        f[f"{col}_max"]  = df[col].abs().max()
    for col in ["pitch", "roll", "yaw"]:
        f[f"{col}_mean"]  = df[col].mean()
        f[f"{col}_std"]   = df[col].std()
        f[f"{col}_range"] = df[col].max() - df[col].min()
    nperseg         = min(256, len(df) // 2)
    freqs, psd      = welch(df["magnitude"].values, fs=FS, nperseg=nperseg)
    chew            = (freqs >= 0.5) & (freqs <= 4.0)
    cf, cp          = freqs[chew], psd[chew]
    f["total_power"]        = float(psd.sum())
    f["chew_band_power"]    = float(cp.sum())
    f["rhythmicity"]        = f["chew_band_power"] / f["total_power"] if f["total_power"] > 0 else 0.0
    f["dominant_chew_freq"] = float(cf[np.argmax(cp)]) if len(cp) > 0 else 0.0
    return f

# ══════════════════════════════════════════════════════════════════════════════
#  MODELL-TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_models(console: Console) -> tuple:
    """Lädt Rohdaten, extrahiert Features, trainiert Stufe-1 und Stufe-2 Modell."""
    _SKIP = {"Metadata.csv", "Annotation.csv"}

    with console.status("[bold cyan]Lade Sensordaten aus data/raw …[/bold cyan]"):
        sessions: dict[str, list] = {cls: [] for cls in CLASSES_RAW}
        for zf in sorted(DATA_DIR.glob("*.zip")):
            for cls in CLASSES_RAW:
                if zf.name.startswith(cls + "_"):
                    sessions[cls].append(zf)
                    break

        rows, labels = [], []
        for cls in CLASSES_RAW:
            for zf in sessions[cls]:
                with zipfile.ZipFile(zf) as z:
                    csv_name = next(
                        n for n in z.namelist()
                        if n.endswith(".csv") and n not in _SKIP
                    )
                    with z.open(csv_name) as fh:
                        df = pd.read_csv(fh)

                # Trim
                t  = df["seconds_elapsed"]
                df = df[
                    (t >= t.iloc[0] + TRIM_SECS) & (t <= t.iloc[-1] - TRIM_SECS)
                ].reset_index(drop=True)
                df = _add_derived(df)

                # 25-s-Fenster
                ts = df["seconds_elapsed"].values
                t_start = ts[0]
                while t_start + MIN_WINDOW <= ts[-1]:
                    t_stop = t_start + WINDOW_SECS
                    w = df[(ts >= t_start) & (ts < t_stop)].reset_index(drop=True)
                    if len(w) > 1:
                        dur = w["seconds_elapsed"].iloc[-1] - w["seconds_elapsed"].iloc[0]
                        if dur >= MIN_WINDOW:
                            clean = _movement_exclusion(w)
                            if len(clean) > 50:
                                rows.append(extract_features(clean))
                                labels.append(cls)
                    t_start = t_stop

    X   = pd.DataFrame(rows)
    y   = np.array(labels)
    yc  = np.array([TO_COARSE[c] for c in y])

    n_samples = {cls: (y == cls).sum() for cls in CLASSES_RAW}

    with console.status("[bold cyan]Trainiere Stufe-1-Modell (Still vs. Essen) …[/bold cyan]"):
        m1 = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        m1.fit(X[FEATURES_S1], yc)

    with console.status("[bold cyan]Trainiere Stufe-2-Modell (Feinklassifikation) …[/bold cyan]"):
        eat      = yc == "Essen"
        X_eat    = X[eat].reset_index(drop=True)
        y_fine   = y[eat]

        # Schädliche Features via Permutation Importance entfernen
        clf_pi = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        clf_pi.fit(X_eat, y_fine)
        pi      = permutation_importance(clf_pi, X_eat, y_fine,
                                          n_repeats=10, random_state=42, n_jobs=-1)
        pi_mean = pd.Series(pi.importances_mean, index=X_eat.columns)
        feat_s2 = [f for f in X_eat.columns if pi_mean[f] >= 0]

        m2 = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
        m2.fit(X_eat[feat_s2], y_fine)

    # Zusammenfassung
    grid = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    grid.add_column(style="dim")
    grid.add_column()
    grid.add_row("Trainings-Samples",
                 "  ".join(f"[bold]{cls}[/bold]: {n_samples[cls]}" for cls in CLASSES_RAW))
    grid.add_row("Stufe-1-Features", f"{len(FEATURES_S1)}  ({', '.join(FEATURES_S1)})")
    grid.add_row("Stufe-2-Features",
                 f"{len(feat_s2)} / {len(X_eat.columns)} (nach Permutation-Importance-Filter)")

    # Trainings-Mittelwerte für Stage-1-Features (für Debug-Anzeige)
    df_train = X[FEATURES_S1].copy()
    df_train["_class"] = yc
    train_stats = df_train.groupby("_class")[FEATURES_S1].mean()

    console.print(Panel(grid, title="[bold green]Modell trainiert[/bold green]",
                         border_style="green"))

    return m1, m2, feat_s2, train_stats, X, y, yc, X_eat, y_fine

# ══════════════════════════════════════════════════════════════════════════════
#  SENSOR-PUFFER
# ══════════════════════════════════════════════════════════════════════════════

class SensorBuffer:
    """Thread-sicherer Ringpuffer für eingehende Sensor-Samples."""

    MAX_SECS = 20.0  # Puffergröße

    def __init__(self) -> None:
        self._lock         = threading.Lock()
        self._samples: list[dict] = []   # jedes Element: {key: value, …, "_ts": float}
        self.total_received: int = 0     # Gesamt empfangene Samples (wächst immer)
        self._last_add: float = 0.0      # monotonic time des letzten add()

    def add(self, values: dict, time_ns: int) -> None:
        if not all(k in values for k in REQUIRED_KEYS):
            return
        ts = time_ns / 1e9
        row = {k: float(values[k]) for k in REQUIRED_KEYS}
        row["_ts"] = ts
        with self._lock:
            self._samples.append(row)
            self.total_received += 1
            self._last_add = time.monotonic()
            # Altes Material verwerfen
            cutoff = self._samples[-1]["_ts"] - self.MAX_SECS
            while self._samples and self._samples[0]["_ts"] < cutoff:
                self._samples.pop(0)

    def get_window(self, duration: float = WINDOW_SECS) -> "pd.DataFrame | None":
        with self._lock:
            if not self._samples:
                return None
            t_end   = self._samples[-1]["_ts"]
            t_start = t_end - duration
            window  = [s for s in self._samples if s["_ts"] >= t_start]
            if len(window) < 2:
                return None
            actual = window[-1]["_ts"] - window[0]["_ts"]
            if actual < MIN_WINDOW:
                return None
            df = pd.DataFrame(window)
            df["seconds_elapsed"] = df["_ts"] - df["_ts"].iloc[0]
            return df.drop(columns=["_ts"])

    def buffered_seconds(self) -> float:
        with self._lock:
            if len(self._samples) < 2:
                return 0.0
            return self._samples[-1]["_ts"] - self._samples[0]["_ts"]

    def sample_count(self) -> int:
        with self._lock:
            return len(self._samples)

    def seconds_since_last(self) -> float:
        """Sekunden seit dem letzten empfangenen Sample."""
        if self._last_add == 0.0:
            return float("inf")
        return time.monotonic() - self._last_add

# ══════════════════════════════════════════════════════════════════════════════
#  MAHLZEITEN-SESSION
# ══════════════════════════════════════════════════════════════════════════════

class MealSession:
    """
    State-Automat für eine Mahlzeiten-Session.

    idle  ──(5× Essen in Folge)──▶  active  ──(3× Still in Folge)──▶  idle
                                       │
                               jede Essen-Klassifikation
                               gibt eine Stimme ab → Mehrheitsvotum
    """

    START_STREAK = 5   # 5 × SLIDE_SECS = 10 s konsistentes Essen → Session startet
    END_STREAK   = 3   # 3 × SLIDE_SECS =  6 s Still → Session endet

    def __init__(self) -> None:
        self.state:        str               = "idle"   # idle | active
        self.votes:        dict[str, int]    = {}
        self.essen_streak: int               = 0
        self.still_streak: int               = 0
        self.start_time:   datetime | None   = None
        self.completed:    list[dict]        = []

    def update(self, result: dict) -> None:
        label    = result["label"]
        is_essen = label != "Still"

        if is_essen:
            self.essen_streak += 1
            self.still_streak  = 0
        else:
            self.still_streak += 1
            self.essen_streak  = 0

        # idle → active
        if self.state == "idle" and self.essen_streak >= self.START_STREAK:
            self.state      = "active"
            self.start_time = datetime.now()
            self.votes      = {}

        # active: Stimme abgeben + Abbruchbedingung
        if self.state == "active":
            if is_essen:
                self.votes[label] = self.votes.get(label, 0) + 1

            if self.still_streak >= self.END_STREAK:
                winner   = self._best_label()
                duration = int((datetime.now() - self.start_time).total_seconds())
                entry = {
                    "start":      self.start_time.strftime("%H:%M:%S"),
                    "label":      winner,
                    "conf":       round(self._vote_conf(), 2),
                    "votes":      dict(self.votes),
                    "duration_s": duration,
                }
                self.completed.append(entry)
                if len(self.completed) > 20:
                    self.completed.pop(0)
                # Reset
                self.state        = "idle"
                self.votes        = {}
                self.start_time   = None
                self.essen_streak = 0

    def _best_label(self) -> str:
        if not self.votes:
            return "Essen (unbekannt)"
        return max(self.votes, key=self.votes.get)

    def _vote_conf(self) -> float:
        total = sum(self.votes.values())
        return max(self.votes.values()) / total if total > 0 else 0.0

    def api_state(self) -> dict:
        return {
            "state":        self.state,
            "essen_streak": self.essen_streak,
            "still_streak": self.still_streak,
            "start_needed": self.START_STREAK,
            "end_needed":   self.END_STREAK,
            "votes":        dict(self.votes),
            "best_label":   self._best_label() if self.votes else None,
            "vote_conf":    round(self._vote_conf(), 3),
            "duration_s":   int((datetime.now() - self.start_time).total_seconds())
                            if self.start_time else 0,
            "completed":    list(reversed(self.completed[-8:])),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  LIVE-KLASSIFIKATOR
# ══════════════════════════════════════════════════════════════════════════════

class LiveClassifier:
    def __init__(self, m1, m2, feat_s2: list[str], train_stats: pd.DataFrame) -> None:
        self.m1          = m1
        self.m2          = m2
        self.feat_s2     = feat_s2
        self.train_stats = train_stats   # mean pro Klasse für FEATURES_S1
        self.result: dict | None = None
        self.history: list[dict] = []
        self.last_feats: dict | None = None
        self.session = MealSession()

    def classify(self, df: pd.DataFrame) -> dict:
        df = _add_derived(df)

        # Maske explizit berechnen — für Filterung UND Visualisierung
        if _movement_excl:
            thr      = max(0.02, K_MOV * df["magnitude"].median())
            roll_max = df["magnitude"].rolling(50, center=True, min_periods=1).max()
            mask     = roll_max <= thr
            clean    = df[mask].reset_index(drop=True)
            excl_arr = (~mask).values          # True = herausgefiltert
        else:
            clean    = df
            excl_arr = np.zeros(len(df), dtype=bool)

        # Magnitude + Ausschlussmaske für Visualisierung (auf 80 Punkte reduziert)
        n    = len(df)
        idxs = np.linspace(0, n - 1, min(n, 80), dtype=int)
        mag_viz  = [round(float(v), 4) for v in df["magnitude"].values[idxs]]
        excl_viz = [bool(v)            for v in excl_arr[idxs]]

        if len(clean) < 50:
            result = {"label": "Still", "conf": 0.0, "stage": 1,
                      "note": "wenig Daten nach Filterung"}
        else:
            feats = pd.DataFrame([extract_features(clean)])
            self.last_feats = {f: float(feats[f].iloc[0]) for f in FEATURES_S1}

            # Stufe 1
            p1    = self.m1.predict_proba(feats[FEATURES_S1])[0]
            pred1 = self.m1.classes_[np.argmax(p1)]
            conf1 = float(np.max(p1))

            # Schwellenwert: nur als Essen klassifizieren wenn Konfidenz >= CONF_S1_MIN
            if pred1 == "Essen" and conf1 < CONF_S1_MIN:
                pred1 = "Still"
                conf1 = 1.0 - conf1

            if pred1 == "Still":
                result = {"label": "Still", "conf": conf1, "stage": 1}
            else:
                # Stufe 2
                p2    = self.m2.predict_proba(feats[self.feat_s2])[0]
                pred2 = self.m2.classes_[np.argmax(p2)]
                conf2 = float(np.max(p2))
                result = {"label": pred2, "conf": conf2, "stage": 2,
                          "s1_conf": conf1}

        result["time"] = datetime.now().strftime("%H:%M:%S")
        result["me"]   = _movement_excl
        result["mag"]  = mag_viz
        result["excl"] = excl_viz
        self.result = result
        self.history.append(result)
        self.session.update(result)

        # Log schreiben
        log_row = {"time": result["time"], "label": result["label"],
                   "stage": result["stage"], "confidence": f"{result['conf']:.4f}",
                   "s1_conf": f"{result.get('s1_conf', result['conf']):.4f}",
                   "samples_clean": len(clean) if len(clean) >= 50 else 0}
        if self.last_feats:
            log_row.update({k: f"{v:.6f}" for k, v in self.last_feats.items()})
        _write_log(log_row)
        if len(self.history) > 30:
            self.history.pop(0)
        return result

# ══════════════════════════════════════════════════════════════════════════════
#  HTTP-SERVER
# ══════════════════════════════════════════════════════════════════════════════

_buffer: SensorBuffer
_clf:    "LiveClassifier"
_movement_excl: bool  = MOVEMENT_EXCL
_model_stats:   dict  = {"status": "pending"}
_loo_data:      tuple = ()

def _run_loo_eval(X, y_raw, yc, X_eat, y_fine, feat_s2) -> None:
    """Läuft im Hintergrund — berechnet LOO-Metriken für beide Stufen."""
    global _model_stats
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

    _model_stats = {"status": "running", "computed_at": None}
    try:
        clf = RandomForestClassifier(n_estimators=200, random_state=42,
                                      class_weight="balanced")
        loo = LeaveOneOut()

        # ── Stufe 1 ──────────────────────────────────────────────────────────
        yt1, yp1 = [], []
        for tr, te in loo.split(X):
            clf.fit(X[FEATURES_S1].iloc[tr], yc[tr])
            yp1.append(clf.predict(X[FEATURES_S1].iloc[[te[0]]])[0])
            yt1.append(yc[te[0]])

        labels1 = ["Still", "Essen"]
        cm1     = confusion_matrix(yt1, yp1, labels=labels1).tolist()
        rep1    = classification_report(yt1, yp1, labels=labels1,
                                         output_dict=True, zero_division=0)
        acc1    = accuracy_score(yt1, yp1)

        # ── Stufe 2 ──────────────────────────────────────────────────────────
        yt2, yp2 = [], []
        for tr, te in loo.split(X_eat):
            clf.fit(X_eat[feat_s2].iloc[tr], y_fine[tr])
            yp2.append(clf.predict(X_eat[feat_s2].iloc[[te[0]]])[0])
            yt2.append(y_fine[te[0]])

        labels2 = ["Apfel", "Kaugummi", "Skyr", "Essen"]
        cm2     = confusion_matrix(yt2, yp2, labels=labels2).tolist()
        rep2    = classification_report(yt2, yp2, labels=labels2,
                                         output_dict=True, zero_division=0)
        acc2    = accuracy_score(yt2, yp2)

        _model_stats = {
            "status":      "ready",
            "computed_at": datetime.now().strftime("%H:%M:%S"),
            "n_samples":   {cls: int((y_raw == cls).sum()) for cls in CLASSES_RAW},
            "stage1": {"cm": cm1, "labels": labels1, "report": rep1, "accuracy": round(acc1, 4)},
            "stage2": {"cm": cm2, "labels": labels2, "report": rep2, "accuracy": round(acc2, 4)},
        }
    except Exception as e:
        _model_stats = {"status": "error", "message": str(e)}

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *_): pass

    # ── GET ───────────────────────────────────────────────────────────────────
    def do_GET(self):
        if self.path == "/":
            body = _HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/api/model_stats":
            body = json.dumps(_model_stats).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/api/recompute":
            global _loo_data
            if _model_stats.get("status") != "running" and _loo_data:
                threading.Thread(target=_run_loo_eval, args=_loo_data, daemon=True).start()
            body = b'{"started": true}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/api/toggle_movement":
            global _movement_excl
            _movement_excl = not _movement_excl
            body = json.dumps({"movement_excl": _movement_excl}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/api/status":
            status = {
                "result":        _clf.result,
                "history":       list(reversed(_clf.history[-15:])),
                "history_count": len(_clf.history),
                "buffered_secs": round(_buffer.buffered_seconds(), 2),
                "buffer_count":  _buffer.sample_count(),
                "total_received":_buffer.total_received,
                "stale_secs":    round(_buffer.seconds_since_last(), 1),
                "window_secs":   WINDOW_SECS,
                "session":       _clf.session.api_state(),
                "movement_excl": _movement_excl,
            }
            body = json.dumps(status).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_response(404); self.end_headers()

    # ── POST ──────────────────────────────────────────────────────────────────
    def do_POST(self):
        if self.path != "/data":
            self.send_response(404); self.end_headers(); return

        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)
        self.send_response(200); self.end_headers(); self.wfile.write(b"OK")

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return

        payload = data.get("payload", data) if isinstance(data, dict) else data
        if not isinstance(payload, list):
            return

        for entry in payload:
            ts     = entry.get("time", 0)
            values = entry.get("values", {})
            if isinstance(values, dict) and ts:
                _buffer.add(values, ts)

# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY (Rich)
# ══════════════════════════════════════════════════════════════════════════════

def _result_panel(clf: LiveClassifier) -> Panel:
    res = clf.result
    if res is None:
        content = Text("\n  Warte auf Sensordaten …\n", style="bold yellow")
        return Panel(content, title="[bold]Aktuelle Klassifikation[/bold]",
                     border_style="yellow", height=7)

    label  = res["label"]
    color, icon = LABEL_STYLE.get(label, ("white", label))
    conf   = res["conf"]
    stage  = res["stage"]
    ts     = res.get("time", "")

    content = Text()
    content.append(f"\n  {icon}\n", style=f"bold {color}")
    content.append(f"\n  Konfidenz: {conf:.0%}", style="dim")
    if stage == 2:
        content.append(f"   (Stufe 1: {res.get('s1_conf', 0):.0%} → Essen)", style="dim")
    content.append(f"\n  Stufe {stage}  ·  {ts}\n", style="dim")

    note = res.get("note")
    if note:
        content.append(f"  ⚠ {note}\n", style="dim yellow")

    return Panel(content, title="[bold]Aktuelle Klassifikation[/bold]",
                 border_style=color, height=7)


def _buffer_panel(buf: SensorBuffer) -> Panel:
    buffered  = buf.buffered_seconds()
    ratio     = min(buffered / WINDOW_SECS, 1.0)
    filled    = int(ratio * 24)
    bar       = "█" * filled + "░" * (24 - filled)
    stale     = buf.seconds_since_last()

    if stale > 5:
        color  = "red"
        status = f"[bold red]KEIN SIGNAL seit {stale:.0f} s[/bold red]"
    elif buffered >= MIN_WINDOW:
        color  = "green"
        status = "[green]bereit[/green]"
    else:
        color  = "yellow"
        status = "[yellow]sammle …[/yellow]"

    t = Text()
    t.append(f"  [{bar}] ", style=color)
    t.append(f"{buffered:.1f} / {WINDOW_SECS:.0f} s", style=f"bold {color}")
    t.append(f"  ·  im Buffer: {buf.sample_count()}", style="dim")
    t.append(f"  ·  gesamt empfangen: {buf.total_received}", style="dim")

    return Panel(t, title=f"[bold]Datenpuffer[/bold]  {status}", border_style=color, height=4)


def _debug_panel(clf: LiveClassifier) -> Panel:
    """Zeigt Live-Feature-Werte vs. Trainings-Mittelwerte für Stufe-1-Diagnose."""
    lf = clf.last_feats
    ts = clf.train_stats

    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim",
                  padding=(0, 1))
    table.add_column("Feature",       width=18, style="dim")
    table.add_column("Live",          width=10, justify="right")
    table.add_column("Ø Still",       width=10, justify="right", style="green")
    table.add_column("Ø Essen",       width=10, justify="right", style="magenta")
    table.add_column("näher bei",     width=10, justify="center")

    if lf is None:
        table.add_row("—", "—", "—", "—", "—")
    else:
        for feat in FEATURES_S1:
            live_v  = lf[feat]
            still_v = float(ts.loc["Still",  feat]) if "Still"  in ts.index else float("nan")
            essen_v = float(ts.loc["Essen",  feat]) if "Essen"  in ts.index else float("nan")
            d_still = abs(live_v - still_v)
            d_essen = abs(live_v - essen_v)
            closer  = Text("Still",  style="bold green")  if d_still <= d_essen \
                      else Text("Essen", style="bold magenta")
            table.add_row(
                feat,
                f"{live_v:.4f}",
                f"{still_v:.4f}",
                f"{essen_v:.4f}",
                closer,
            )

    return Panel(table,
                 title=f"[bold]Debug — Stufe-1-Features  (Schwelle Essen ≥ {CONF_S1_MIN:.0%})[/bold]",
                 border_style="bright_black")


def _history_panel(clf: LiveClassifier) -> Panel:
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold dim",
                  padding=(0, 1))
    table.add_column("Zeit",      width=9,  style="dim")
    table.add_column("Klasse",    width=20)
    table.add_column("Konfidenz", width=9,  justify="right")
    table.add_column("Stufe",     width=5,  justify="center", style="dim")

    for r in reversed(clf.history[-12:]):
        label = r["label"]
        color, icon = LABEL_STYLE.get(label, ("white", label))
        table.add_row(
            r.get("time", ""),
            Text(icon, style=f"bold {color}"),
            f"{r['conf']:.0%}",
            str(r["stage"]),
        )

    return Panel(table, title="[bold]Verlauf[/bold]", border_style="bright_black")


def build_layout(clf: LiveClassifier, buf: SensorBuffer) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(_result_panel(clf),  name="result",  size=7),
        Layout(_buffer_panel(buf),  name="buffer",  size=4),
        Layout(name="bottom"),
    )
    layout["bottom"].split_row(
        Layout(_debug_panel(clf),   name="debug"),
        Layout(_history_panel(clf), name="history"),
    )
    return layout

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    global _buffer, _clf, _loo_data

    console = Console()
    console.rule("[bold blue]Live-Essklassifikation[/bold blue]")

    # ── Modell trainieren ────────────────────────────────────────────────────
    m1, m2, feat_s2, train_stats, X, y_raw, yc, X_eat, y_fine = train_models(console)
    _clf     = LiveClassifier(m1, m2, feat_s2, train_stats)
    _loo_data = (X, y_raw, yc, X_eat, y_fine, feat_s2)

    # LOO-Evaluation im Hintergrund starten
    threading.Thread(target=_run_loo_eval, args=_loo_data, daemon=True).start()
    console.print("[dim]LOO-Evaluation läuft im Hintergrund …[/dim]")
    _buffer = SensorBuffer()

    # ── HTTP-Server starten ──────────────────────────────────────────────────
    server = HTTPServer(("0.0.0.0", PORT), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = "localhost"

    console.print(Panel(
        f"[bold green]Server läuft[/bold green]\n\n"
        f"  Sensordaten:  POST [bold cyan]http://{ip}:{PORT}/data[/bold cyan]\n"
        f"  Web-UI:       GET  [bold cyan]http://{ip}:{PORT}/[/bold cyan]\n\n"
        f"  Fenster: [bold]{WINDOW_SECS:.0f} s[/bold]  ·  "
        f"Update: [bold]{SLIDE_SECS:.0f} s[/bold]  ·  "
        f"k = [bold]{K_MOV}[/bold]  ·  "
        f"Schwelle: [bold]{CONF_S1_MIN:.0%}[/bold]\n\n"
        f"  Log: [dim]{LOG_FILE.name}[/dim]",
        title="[bold]Server[/bold]", border_style="green",
    ))

    # ── Klassifikations-Loop ─────────────────────────────────────────────────
    last_classify = 0.0
    console.print("[dim]Klassifikations-Log (Strg+C zum Beenden):[/dim]\n")

    while True:
        now      = time.monotonic()
        buffered = _buffer.buffered_seconds()

        if buffered >= MIN_WINDOW and (now - last_classify) >= SLIDE_SECS:
            window_df = _buffer.get_window(WINDOW_SECS)
            if window_df is not None:
                prev_state = _clf.session.state
                res   = _clf.classify(window_df)
                label = res["label"]
                conf  = res["conf"]
                color, icon = LABEL_STYLE.get(label, ("white", label))
                console.print(
                    f"[dim]{res['time']}[/dim]  "
                    f"[bold {color}]{icon:<22}[/bold {color}]  "
                    f"[dim]{conf:.0%}  Stufe {res['stage']}[/dim]"
                )
                # Session-Events hervorheben
                sess = _clf.session
                if prev_state == "idle" and sess.state == "active":
                    console.print(f"[bold green]▶ SESSION GESTARTET[/bold green]")
                elif prev_state == "active" and sess.state == "idle" and sess.completed:
                    last = sess.completed[-1]
                    c, _ = LABEL_STYLE.get(last["label"], ("white", last["label"]))
                    total = sum(last["votes"].values())
                    console.print(
                        f"[bold {c}]■ SESSION ENDE → {last['label']}  "
                        f"({total} Stimmen, {last['duration_s']}s)[/bold {c}]"
                    )
                last_classify = now

        time.sleep(0.25)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGestoppt.")
