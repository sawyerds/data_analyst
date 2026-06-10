"""
qpf_quick.py — standalone QPF verification web app.

Runs as its own FastAPI process on port 8011, sitting next to your existing
WxAgent snowfall app on port 8010. Zero changes to your existing code.

Start:
    cd /Users/sawyersmith/Desktop/datasets/gridded_snowfall_analysis
    conda activate wxagent
    uvicorn qpf_quick:app --port 8011

Then open http://localhost:8011 in a browser.
"""

import base64
import json
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from qpf_verification import SCHEMA, run, build_synthesize_payload

app = FastAPI(title="WxAgent QPF Verification")


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WxAgent — QPF Verification</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #0b1220;
    --surface: #131c2e;
    --surface-2: #1a2540;
    --border: #2a3754;
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --teal: #14b8a6;
    --teal-dark: #0d9488;
    --danger: #ef4444;
    --success: #10b981;
  }
  * { box-sizing: border-box; }
  body {
    font-family: 'IBM Plex Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
    margin: 0;
    padding: 0;
    line-height: 1.55;
  }
  header {
    padding: 1.5rem 2rem;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }
  header h1 {
    margin: 0;
    font-size: 1.25rem;
    font-weight: 500;
    color: var(--teal);
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.02em;
  }
  header .sub {
    margin: 0.25rem 0 0;
    font-size: 0.85rem;
    color: var(--text-muted);
    font-family: 'IBM Plex Mono', monospace;
  }
  main {
    max-width: 1100px;
    margin: 0 auto;
    padding: 2rem;
  }
  .group {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
  }
  .group h2 {
    margin: 0 0 1rem;
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--teal);
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }
  .field { margin-bottom: 0.9rem; }
  .field:last-child { margin-bottom: 0; }
  .field label {
    display: block;
    font-size: 0.8rem;
    margin-bottom: 0.3rem;
    color: var(--text-muted);
    font-family: 'IBM Plex Mono', monospace;
  }
  .field input[type=text],
  .field input[type=number],
  .field select {
    width: 100%;
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 0.5rem 0.7rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    border-radius: 4px;
  }
  .field input:focus, .field select:focus {
    outline: none;
    border-color: var(--teal);
    box-shadow: 0 0 0 2px rgba(20, 184, 166, 0.2);
  }
  .checkbox-row { display: flex; align-items: center; gap: 0.5rem; }
  .checkbox-row input[type=checkbox] {
    width: auto;
    accent-color: var(--teal);
  }
  .help {
    font-size: 0.72rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
    font-style: italic;
  }
  button {
    background: var(--teal);
    color: var(--bg);
    border: none;
    padding: 0.75rem 1.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    font-size: 0.85rem;
    border-radius: 4px;
    cursor: pointer;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.5rem;
  }
  button:hover { background: var(--teal-dark); }
  button:disabled { opacity: 0.5; cursor: not-allowed; }

  #status {
    margin: 1.5rem 0;
  }
  .status {
    padding: 0.75rem 1rem;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
  }
  .status.running { background: rgba(20, 184, 166, 0.08); border: 1px solid var(--teal); color: var(--teal); }
  .status.error { background: rgba(239, 68, 68, 0.08); border: 1px solid var(--danger); color: #fca5a5; white-space: pre-wrap; }
  .status.success { background: rgba(16, 185, 129, 0.08); border: 1px solid var(--success); color: var(--success); }

  #results { margin-top: 2rem; }
  #results h3 {
    color: var(--teal);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 2rem 0 0.75rem;
  }
  #results img {
    max-width: 100%;
    border: 1px solid var(--border);
    border-radius: 6px;
    margin-bottom: 1rem;
    background: white;
  }
  #results pre {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 1rem;
    border-radius: 6px;
    overflow-x: auto;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--text-muted);
    line-height: 1.5;
  }
  .download-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 1rem;
  }
  .dl-link {
    display: inline-block;
    padding: 0.4rem 0.8rem;
    background: var(--surface-2);
    border: 1px solid var(--border);
    color: var(--teal);
    text-decoration: none;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
  }
  .dl-link:hover { background: var(--border); }
</style>
</head>
<body>
<header>
  <h1>WxAgent / QPF Verification</h1>
  <p class="sub">WRF QPF vs Stage IV QPE — gridded + watershed analysis</p>
</header>
<main>
  <form id="form"></form>
  <div id="status"></div>
  <div id="results"></div>
</main>

<script>
let SCHEMA = null;

async function loadSchema() {
  const r = await fetch('/schema');
  SCHEMA = await r.json();
  const form = document.getElementById('form');
  SCHEMA.groups.forEach((g, gi) => {
    const div = document.createElement('div');
    div.className = 'group';
    div.dataset.groupIndex = gi;
    if (g.show_when) div.dataset.showWhen = JSON.stringify(g.show_when);
    div.innerHTML = `<h2>${g.title}</h2>`;
    g.fields.forEach(f => {
      const fd = document.createElement('div');
      fd.className = 'field';
      let inputHtml = '';
      const defVal = f.default !== undefined ? f.default : '';

      if (f.type === 'enum') {
        inputHtml = `<select name="${f.name}">` +
          f.options.map(o => `<option value="${o}" ${o === defVal ? 'selected' : ''}>${o}</option>`).join('') +
          `</select>`;
      } else if (f.type === 'boolean') {
        inputHtml = `<div class="checkbox-row"><input type="checkbox" name="${f.name}" ${defVal ? 'checked' : ''}><span style="font-size:0.85rem;font-family:'IBM Plex Mono',monospace;">${f.label}</span></div>`;
        fd.innerHTML = `${inputHtml}${f.help ? '<div class="help">' + f.help + '</div>' : ''}`;
        div.appendChild(fd);
        return;
      } else if (f.type === 'number') {
        inputHtml = `<input type="number" name="${f.name}" value="${defVal}" ${f.min !== undefined ? 'min=' + f.min : ''} ${f.max !== undefined ? 'max=' + f.max : ''}>`;
      } else if (f.type === 'array') {
        const v = Array.isArray(defVal) ? defVal.join(', ') : '';
        inputHtml = `<input type="text" name="${f.name}" data-array="true" value="${v}" placeholder="comma-separated">`;
      } else {
        inputHtml = `<input type="text" name="${f.name}" value="${defVal}" placeholder="${f.placeholder || ''}">`;
      }
      fd.innerHTML = `<label>${f.label}${f.required ? ' *' : ''}</label>${inputHtml}${f.help ? '<div class="help">' + f.help + '</div>' : ''}`;
      div.appendChild(fd);
    });
    form.appendChild(div);
  });
  const btn = document.createElement('button');
  btn.type = 'submit';
  btn.textContent = 'Run Verification';
  form.appendChild(btn);
  form.addEventListener('submit', onSubmit);
  form.addEventListener('change', updateConditional);
  form.addEventListener('input', updateConditional);
  updateConditional();
}

function updateConditional() {
  document.querySelectorAll('.group[data-show-when]').forEach(g => {
    const cond = JSON.parse(g.dataset.showWhen);
    let show = true;
    for (const [k, expected] of Object.entries(cond)) {
      const el = document.querySelector(`[name="${k}"]`);
      if (!el) continue;
      const actual = el.type === 'checkbox' ? el.checked : el.value;
      if (String(actual) !== String(expected)) show = false;
    }
    g.style.display = show ? '' : 'none';
  });
}

async function onSubmit(e) {
  e.preventDefault();
  const form = e.target;
  const status = document.getElementById('status');
  const results = document.getElementById('results');
  results.innerHTML = '';
  status.innerHTML = '<div class="status running">Running verification — this can take 30–90 seconds.</div>';
  form.querySelector('button').disabled = true;

  const data = {};
  form.querySelectorAll('input, select').forEach(el => {
    if (el.type === 'checkbox') data[el.name] = el.checked;
    else if (el.dataset.array) {
      data[el.name] = el.value.split(',').map(s => s.trim()).filter(Boolean);
      data[el.name] = data[el.name].map(v => isFinite(v) && v !== '' ? Number(v) : v);
    } else if (el.type === 'number') {
      data[el.name] = el.value === '' ? null : parseFloat(el.value);
    } else if (el.value !== '') {
      data[el.name] = el.value;
    }
  });

  try {
    const r = await fetch('/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data),
    });
    if (!r.ok) {
      const t = await r.text();
      throw new Error(t);
    }
    const result = await r.json();
    status.innerHTML = '<div class="status success">Done.</div>';

    if (result.figures.length) {
      const figHeader = document.createElement('h3');
      figHeader.textContent = 'Figures';
      results.appendChild(figHeader);
      result.figures.forEach(f => {
        const img = document.createElement('img');
        img.src = 'data:' + f.mime + ';base64,' + f.b64;
        img.alt = f.filename;
        results.appendChild(img);
      });
    }

    if (result.csv_files.length) {
      const csvHeader = document.createElement('h3');
      csvHeader.textContent = 'Downloads';
      results.appendChild(csvHeader);
      const row = document.createElement('div');
      row.className = 'download-row';
      result.csv_files.forEach(f => {
        const a = document.createElement('a');
        a.className = 'dl-link';
        a.href = 'data:text/csv;base64,' + f.b64;
        a.download = f.filename;
        a.textContent = '↓ ' + f.filename;
        row.appendChild(a);
      });
      result.figures.forEach(f => {
        const a = document.createElement('a');
        a.className = 'dl-link';
        a.href = 'data:' + f.mime + ';base64,' + f.b64;
        a.download = f.filename;
        a.textContent = '↓ ' + f.filename;
        row.appendChild(a);
      });
      results.appendChild(row);
    }

    const findHeader = document.createElement('h3');
    findHeader.textContent = 'Findings (JSON)';
    results.appendChild(findHeader);
    const pre = document.createElement('pre');
    pre.textContent = JSON.stringify(result.findings, null, 2);
    results.appendChild(pre);

  } catch (err) {
    status.innerHTML = '<div class="status error">Error:\n' + err.message + '</div>';
  } finally {
    form.querySelector('button').disabled = false;
  }
}

loadSchema();
</script>
</body>
</html>
"""


@app.get('/', response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.get('/schema')
def schema():
    return SCHEMA


@app.post('/run')
def do_run(plan: dict):
    try:
        result = run(plan)
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
    findings = build_synthesize_payload(result)
    return {
        "figures": [
            {"filename": fn, "mime": mt, "b64": base64.b64encode(b).decode()}
            for fn, mt, b in result.figures
        ],
        "csv_files": [
            {"filename": fn, "b64": base64.b64encode(b).decode()}
            for fn, b in result.csv_files
        ],
        "findings": findings,
    }
