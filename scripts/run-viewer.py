#!/usr/bin/env python3
"""SteerLab run viewer — a loopback-only browser for a workspace's runs/.

Read-only. Serves an index of every run directory plus type-specific views:

  validate  concept accuracies with binomial p-values and the cross-concept
            cosine matrix as a heatmap
  sweep     the layer x alpha grid as a heatmap, constraint failures marked,
            and the recommendation WITH an explicit statement of whether a
            matched-norm random control was declared
  run       per-condition summary and a paired generation browser: the same
            prompt under every condition, side by side

Usage:
    python3 scripts/run-viewer.py [--root <workspace>] [--port 8765]

Binds 127.0.0.1 only, per the project's security posture: this is an
unauthenticated single-researcher instrument and must not be reachable
off-box. Nothing here writes to the workspace.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

ROOT = ""

# --- workspace reading -------------------------------------------------------

# Library subtrees that live under runs/ but are not runs.
NOT_RUNS = {"model-variants", "neutral-pcs", "multi-agent-scenarios", ".corpus-shards"}


def runs_dir() -> str:
    return os.path.join(ROOT, "runs")


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _read_text(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except Exception:
        return None


def list_runs():
    out = []
    base = runs_dir()
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base), reverse=True):
        d = os.path.join(base, name)
        if not os.path.isdir(d) or name in NOT_RUNS:
            continue
        cfg = _read_json(os.path.join(d, "config.json")) or {}
        task = _read_text(os.path.join(d, "task.txt")) or cfg.get("runType") or "?"
        failed = os.path.exists(os.path.join(d, "FAILED.md"))
        status = _read_json(os.path.join(d, "run-status.json")) or {}
        out.append({
            "id": name,
            "task": task,
            "experiment": cfg.get("experiment") or _experiment_from_name(name),
            "createdAt": cfg.get("createdAt"),
            "modelID": cfg.get("modelID"),
            "dtype": cfg.get("dtype"),
            "revision": (cfg.get("revision") or "")[:12],
            "substrate": cfg.get("substrate"),
            "jobId": cfg.get("jobId"),
            "failed": failed,
            "error": status.get("error"),
            "has": {
                "validation": os.path.exists(os.path.join(d, "validation-report.json")),
                "cosine": os.path.exists(os.path.join(d, "cosine-matrix.csv")),
                "sweep": os.path.exists(os.path.join(d, "sweep.csv")),
                "generations": os.path.exists(os.path.join(d, "generations.jsonl")),
                "report": os.path.exists(os.path.join(d, "report.json")),
            },
        })
    return out


def _experiment_from_name(name):
    m = re.match(r"^\d+T?[\dZ]*-exp-(.+?)-(extract|validate|sweep|run|pipeline|"
                 r"evaluate|analyze|evaluate-judgment)$", name)
    return m.group(1) if m else name


def binom_p(k: int, n: int, p: float = 0.5) -> float:
    """One-tailed P(X >= k) under the null — the honest read of a balanced
    held-out accuracy."""
    if n <= 0 or k > n:
        return float("nan")
    return sum(math.comb(n, i) for i in range(k, n + 1)) * (p ** n)


def read_sweep(d):
    path = os.path.join(d, "sweep.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    def num(v):
        try:
            return float(v)
        except Exception:
            return None
    cells = []
    for r in rows:
        cells.append({
            "concept": r.get("concept"),
            "layer": int(float(r["layer"])) if r.get("layer") else None,
            "alpha": num(r.get("alpha")),
            "objective": num(r.get("objective")),
            "distinct2": num(r.get("distinct2")),
            "words": num(r.get("words")),
            "battery": num(r.get("batteryAccuracy")),
            "markerDensity": num(r.get("markerDensity")),
        })
    return cells


def read_cosine(d):
    path = os.path.join(d, "cosine-matrix.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return None
    header, body = rows[0], rows[1:]
    names = header[1:]
    matrix = []
    for r in body:
        vals = []
        for v in r[1:]:
            try:
                vals.append(float(v))
            except Exception:
                vals.append(None)
        matrix.append({"concept": r[0], "values": vals})
    return {"names": names, "rows": matrix}


def run_detail(run_id):
    d = os.path.join(runs_dir(), run_id)
    if not os.path.isdir(d):
        return None
    cfg = _read_json(os.path.join(d, "config.json")) or {}
    manifest = _read_json(os.path.join(d, "experiment.json")) or {}
    detail = {
        "id": run_id,
        "task": _read_text(os.path.join(d, "task.txt")),
        "config": cfg,
        "report": _read_json(os.path.join(d, "report.json")),
        "recommendations": _read_json(os.path.join(d, "recommendations.json")),
        "sweep": read_sweep(d),
        "cosine": read_cosine(d),
        "failure": _read_text(os.path.join(d, "FAILED.md")),
        "manifest": {
            "name": manifest.get("name"),
            "status": manifest.get("status"),
            "dtype": manifest.get("dtype"),
            "phase": manifest.get("phase"),
            "concepts": [c.get("name") for c in manifest.get("concepts", [])],
            "taskPromptsFile": manifest.get("taskPromptsFile"),
            "outcomeInstruments": manifest.get("outcomeInstruments"),
            "temperature": manifest.get("temperature"),
            "samplesPerItem": manifest.get("samplesPerItem"),
            "sweep": manifest.get("sweep"),
            "variantConditions": [v.get("name") for v in manifest.get("variantConditions", [])],
        },
    }

    vr = _read_json(os.path.join(d, "validation-report.json"))
    if vr:
        layer_count = None
        for c in detail["manifest"]["concepts"] or []:
            side = _read_json(os.path.join(d, f"{c}.json"))
            if side:
                layer_count = side.get("layerCount")
                break
        concepts = []
        for name, v in (vr.get("concepts") or {}).items():
            if not isinstance(v, dict):
                concepts.append({"concept": name, "note": str(v)})
                continue
            acc, n = v.get("scenarioAccuracy"), v.get("scenarioCount")
            row = {"concept": name, "accuracy": acc, "n": n,
                   "layer": v.get("layer"), "layerCount": layer_count,
                   "labeled": v.get("labeled")}
            if acc is not None and n:
                row["k"] = round(acc * n)
                row["p"] = binom_p(row["k"], n)
            concepts.append(row)
        concepts.sort(key=lambda r: -(r.get("accuracy") or 0))
        detail["validation"] = concepts
    return detail


def read_generations(run_id, limit=200, offset=0, condition=None, prompt_id=None):
    path = os.path.join(runs_dir(), run_id, "generations.jsonl")
    if not os.path.exists(path):
        return {"total": 0, "records": []}
    keep, total = [], 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if condition and r.get("condition") != condition:
                continue
            if prompt_id and r.get("promptID") != prompt_id:
                continue
            total += 1
            if total > offset and len(keep) < limit:
                keep.append(_slim(r))
    return {"total": total, "records": keep}


def _slim(r):
    """The fields the browser shows — both record shapes (sampled + logprob)."""
    out = {k: r.get(k) for k in
           ("promptID", "condition", "sampleIndex", "arm", "caseID", "target",
            "parsedChoice", "selected", "wordCount", "distinct2", "seed",
            "margin", "instrument", "interventionState")}
    out["output"] = r.get("output")
    out["prompt"] = r.get("prompt")
    for k in ("choiceProbability", "logOdds", "optionLogprobs"):
        if k in r:
            out[k] = r[k]
    return out


def generation_pairs(run_id):
    """Group records by promptID so conditions sit side by side — the
    compare-agents view. Also reports the baseline choice-margin distribution,
    because a saturated instrument makes a null uninterpretable."""
    path = os.path.join(runs_dir(), run_id, "generations.jsonl")
    if not os.path.exists(path):
        return {"prompts": [], "conditions": []}
    by_prompt, conditions, margins = {}, [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            cond = r.get("condition") or "?"
            if cond not in conditions:
                conditions.append(cond)
            pid = r.get("promptID") or "?"
            slot = by_prompt.setdefault(pid, {
                "promptID": pid, "arm": r.get("arm"), "caseID": r.get("caseID"),
                "target": r.get("target"), "prompt": r.get("prompt"),
                "byCondition": {}})
            slot["byCondition"].setdefault(cond, []).append(_slim(r))
            if cond == "baseline" and r.get("margin") is not None:
                margins.append(abs(r["margin"]))
    saturation = None
    if margins:
        margins.sort()
        mid = margins[len(margins) // 2]
        saturation = {
            "n": len(margins), "median": mid,
            "min": margins[0], "max": margins[-1],
            "belowFive": sum(1 for m in margins if m < 5),
        }
    return {"conditions": conditions,
            "prompts": sorted(by_prompt.values(), key=lambda p: p["promptID"]),
            "saturation": saturation}


# --- HTTP --------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def _send(self, body, ctype="application/json", code=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = u.path
        try:
            if p in ("/", "/index.html"):
                return self._send(PAGE, "text/html; charset=utf-8")
            if p == "/api/runs":
                return self._send({"root": ROOT, "runs": list_runs()})
            if p.startswith("/api/run/"):
                rest = p[len("/api/run/"):]
                if rest.endswith("/generations"):
                    rid = unquote(rest[: -len("/generations")])
                    return self._send(read_generations(
                        rid,
                        limit=int(q.get("limit", ["200"])[0]),
                        offset=int(q.get("offset", ["0"])[0]),
                        condition=(q.get("condition") or [None])[0],
                        prompt_id=(q.get("promptID") or [None])[0]))
                if rest.endswith("/pairs"):
                    return self._send(generation_pairs(unquote(rest[: -len("/pairs")])))
                d = run_detail(unquote(rest))
                if d is None:
                    return self._send({"error": "no such run"}, code=404)
                return self._send(d)
            self._send({"error": "not found"}, code=404)
        except Exception as exc:  # keep the viewer alive on bad data
            self._send({"error": f"{type(exc).__name__}: {exc}"}, code=500)

    def log_message(self, *_):  # quiet
        pass


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>SteerLab runs</title>
<style>
:root{--bg:#0f1115;--panel:#171a21;--line:#272c36;--fg:#e6e9ef;--dim:#9aa4b2;
      --pos:#3fb950;--neg:#f85149;--warn:#d29922;--acc:#58a6ff}
@media (prefers-color-scheme: light){:root{--bg:#f7f8fa;--panel:#fff;--line:#e3e6ea;
      --fg:#1b1f24;--dim:#616b76;--pos:#1a7f37;--neg:#cf222e;--warn:#9a6700;--acc:#0969da}}
*{box-sizing:border-box}
body{margin:0;font:13px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     background:var(--bg);color:var(--fg);display:flex;height:100vh;overflow:hidden}
#side{width:330px;flex:none;border-right:1px solid var(--line);overflow-y:auto;background:var(--panel)}
#main{flex:1;overflow-y:auto;padding:20px 26px}
h1{font-size:13px;margin:0;padding:14px 16px;border-bottom:1px solid var(--line);
   letter-spacing:.08em;text-transform:uppercase;color:var(--dim)}
h2{font-size:17px;margin:0 0 4px}
h3{font-size:13px;margin:26px 0 8px;color:var(--dim);text-transform:uppercase;letter-spacing:.06em}
.exp{padding:8px 16px 4px;font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.06em}
.run{padding:7px 16px;border-bottom:1px solid var(--line);cursor:pointer;font-size:12px}
.run:hover{background:rgba(127,127,127,.09)}
.run.sel{background:rgba(88,166,255,.14);box-shadow:inset 3px 0 var(--acc)}
.run .id{color:var(--dim);font-size:10.5px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.badge{display:inline-block;padding:1px 6px;border-radius:9px;font-size:10px;
       border:1px solid var(--line);margin-right:5px;color:var(--dim)}
.badge.run_{color:var(--acc);border-color:var(--acc)}
.badge.sweep{color:var(--warn);border-color:var(--warn)}
.badge.validate{color:var(--pos);border-color:var(--pos)}
.badge.failed{color:var(--neg);border-color:var(--neg)}
table{border-collapse:collapse;width:100%;margin:6px 0 14px}
th,td{text-align:left;padding:5px 9px;border-bottom:1px solid var(--line);font-size:12px}
th{color:var(--dim);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
td.num,th.num{text-align:right;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.sig{color:var(--pos);font-weight:600}.ns{color:var(--dim)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:10px 0}
.note{border-left:3px solid var(--warn);padding-left:10px;color:var(--warn);margin:10px 0;font-size:12px}
.bad{border-left-color:var(--neg);color:var(--neg)}
.ok{border-left-color:var(--pos);color:var(--pos)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 14px;font-size:12px}
.kv b{color:var(--dim);font-weight:500}
.cell{padding:4px 7px;text-align:right;font-family:ui-monospace,Menlo,monospace;font-size:11px;
      border-radius:3px;min-width:62px;display:inline-block}
.pair{display:grid;gap:12px;margin:10px 0;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.gen{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:10px 12px}
.gen pre{white-space:pre-wrap;word-break:break-word;margin:6px 0 0;font-size:12px;
         font-family:ui-monospace,Menlo,monospace;max-height:340px;overflow:auto;color:var(--fg)}
.pmt{color:var(--dim);font-size:11.5px;max-height:120px;overflow:auto;white-space:pre-wrap;
     border-left:2px solid var(--line);padding-left:9px;margin:6px 0}
button{background:transparent;border:1px solid var(--line);color:var(--fg);border-radius:6px;
       padding:4px 10px;font-size:12px;cursor:pointer;margin-right:6px}
button.on{border-color:var(--acc);color:var(--acc)}
input[type=search]{background:var(--bg);border:1px solid var(--line);color:var(--fg);
       border-radius:6px;padding:5px 9px;font-size:12px;width:230px}
.dim{color:var(--dim)}.mono{font-family:ui-monospace,Menlo,monospace}
</style></head><body>
<div id="side"><h1>Runs</h1><div id="list"></div></div>
<div id="main"><p class="dim">Select a run.</p></div>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s??"").replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const f=(v,d=3)=>v===null||v===undefined||Number.isNaN(v)?"—":(+v).toFixed(d);
let RUNS=[],CUR=null;

function heat(v,lo,hi){ // signed: red↔green through neutral
  if(v===null||v===undefined) return "transparent";
  const t=Math.max(-1,Math.min(1, v>=0 ? v/(hi||1) : -v/Math.abs(lo||1)));
  const a=0.13+0.5*Math.abs(t);
  return v>=0?`rgba(63,185,80,${a})`:`rgba(248,81,73,${a})`;
}

async function load(){
  const r=await (await fetch('/api/runs')).json();
  RUNS=r.runs; const byExp={};
  RUNS.forEach(x=>(byExp[x.experiment]=byExp[x.experiment]||[]).push(x));
  $('#list').innerHTML=Object.entries(byExp).map(([e,rs])=>
    `<div class="exp">${esc(e)}</div>`+rs.map(x=>`
      <div class="run" data-id="${esc(x.id)}">
        <span class="badge ${x.failed?'failed':(x.task==='run'?'run_':esc(x.task))}">${x.failed?'failed':esc(x.task)}</span>
        ${x.dtype?`<span class="dim">${esc(x.dtype)}</span>`:''}
        <div class="id">${esc(x.id.slice(0,17))}</div>
      </div>`).join('')).join('');
  document.querySelectorAll('.run').forEach(el=>el.onclick=()=>open_(el.dataset.id,el));
}

async function open_(id,el){
  document.querySelectorAll('.run').forEach(e=>e.classList.remove('sel'));
  el&&el.classList.add('sel');
  const d=await (await fetch('/api/run/'+encodeURIComponent(id))).json();
  CUR=d; const m=d.manifest||{},c=d.config||{};
  let h=`<h2>${esc(m.name||d.id)}</h2><div class="dim mono" style="font-size:11px">${esc(d.id)}</div>`;
  h+=`<div class="card"><div class="kv">
      <b>task</b><span>${esc(d.task)}</span>
      <b>model</b><span>${esc(c.modelID)} <span class="dim">@${esc((c.revision||'').slice(0,12))}</span></span>
      <b>precision</b><span>${c.dtype?esc(c.dtype):'<span class="dim">not pinned</span>'}</span>
      <b>substrate</b><span>${esc(c.substrate)} <span class="dim">${esc(c.platform||'')}</span></span>
      <b>slurm job</b><span class="mono">${esc(c.jobId||'—')}</span>
      ${m.concepts&&m.concepts.length?`<b>concepts</b><span>${m.concepts.map(esc).join(', ')}</span>`:''}
      ${m.outcomeInstruments?`<b>instruments</b><span>${m.outcomeInstruments.map(esc).join(', ')}</span>`:''}
      ${m.samplesPerItem?`<b>sampling</b><span>${esc(m.samplesPerItem)} × t=${esc(m.temperature)}</span>`:''}
      </div></div>`;
  if(d.failure) h+=`<div class="note bad"><b>FAILED</b><pre style="white-space:pre-wrap">${esc(d.failure.slice(0,1200))}</pre></div>`;
  if(d.validation) h+=validation(d);
  if(d.cosine) h+=cosine(d.cosine);
  if(d.sweep) h+=sweep(d);
  if(d.report) h+=report(d);
  $('#main').innerHTML=h;
  if(d.config&&d.report&&d.report.conditions) loadPairs(d.id);
}

function validation(d){
  let h=`<h3>Convergent validity</h3><table><tr><th>concept</th><th class="num">accuracy</th>
    <th class="num">k / n</th><th class="num">p</th><th class="num">layer</th></tr>`;
  d.validation.forEach(r=>{
    if(r.note) return h+=`<tr><td>${esc(r.concept)}</td><td colspan="4" class="dim">${esc(r.note)}</td></tr>`;
    const sig=r.p!==undefined&&r.p<0.05;
    const depth=r.layerCount?` <span class="dim">of ${r.layerCount} (${(r.layer/r.layerCount).toFixed(2)} depth)</span>`:'';
    h+=`<tr><td>${esc(r.concept)}</td>
      <td class="num" style="background:${heat((r.accuracy-0.5)*2,-1,1)}">${f(r.accuracy)}</td>
      <td class="num dim">${r.k} / ${r.n}</td>
      <td class="num ${sig?'sig':'ns'}">${r.p<0.0001?r.p.toExponential(1):f(r.p,4)}</td>
      <td class="num">${r.layer}${depth}</td></tr>`;
  });
  h+='</table>';
  const n=d.validation.filter(r=>r.p!==undefined).length;
  const passing=d.validation.filter(r=>r.p!==undefined&&r.p<0.05).length;
  if(n>1){
    const ps=d.validation.filter(r=>r.p!==undefined).map(r=>r.p).sort((a,b)=>a-b);
    let surv=0; ps.forEach((p,i)=>{ if(p<=((i+1)/n)*0.05) surv=i+1; });
    h+=`<div class="note ${surv?'ok':''}">${passing} of ${n} beat chance uncorrected;
        <b>${surv} survive BH-FDR at 0.05</b> across ${n} concepts.</div>`;
  }
  return h;
}

function cosine(cm){
  let h=`<h3>Cross-concept cosine</h3><table><tr><th></th>`+
    cm.names.map(n=>`<th class="num">${esc(n.slice(0,7))}</th>`).join('')+`</tr>`;
  let off=[];
  cm.rows.forEach((r,i)=>{
    h+=`<tr><td>${esc(r.concept)}</td>`+r.values.map((v,j)=>{
      if(i!==j&&v!==null) off.push(v);
      return `<td class="num" style="background:${i===j?'transparent':heat(v,-1,1)}">${i===j?'<span class="dim">1</span>':f(v,2)}</td>`;
    }).join('')+`</tr>`;
  });
  h+='</table>';
  if(off.length){
    const mean=off.reduce((a,b)=>a+b,0)/off.length, n=cm.names.length, base=-1/(n-1);
    h+=`<div class="note">mean off-diagonal <b>${f(mean,4)}</b>.
      For n=${n} grand-mean vectors the centering floor is ${f(base,4)} —
      excess <b>${f(mean-base,4)}</b>. (CAA vectors have no such floor.)</div>`;
  }
  return h;
}

function sweep(d){
  const cells=d.sweep.filter(c=>c.layer!==null&&c.layer>=0);
  const layers=[...new Set(cells.map(c=>c.layer))].sort((a,b)=>a-b);
  const alphas=[...new Set(cells.map(c=>c.alpha))].sort((a,b)=>a-b);
  const objs=cells.map(c=>c.objective).filter(v=>v!==null);
  const hi=Math.max(...objs,0), lo=Math.min(...objs,0);
  const rec=d.recommendations||{};
  let h='<h3>Sweep grid — objective</h3>';
  Object.entries(rec).forEach(([concept,r])=>{
    const ctl=r.control;
    h+=`<div class="card"><div class="kv">
      <b>winner</b><span>layer <b>${esc(r.winningCell&&r.winningCell.layer)}</b>,
         α <b>${esc(r.winningCell&&r.winningCell.alpha)}</b> —
         objective <b>${f(r.metrics&&r.metrics.logprobShift ?? r.metrics&&r.metrics.markerDensity)}</b></span>
      <b>criterion</b><span>${esc(r.criterion&&r.criterion.objective&&r.criterion.objective.metric)}
         <span class="dim">${esc((r.criterion&&r.criterion.objective&&r.criterion.objective.choicePromptsFile)||'')}</span></span>
      <b>battery</b><span>${f(r.metrics&&r.metrics.batteryAccuracy,2)}
         <span class="dim">baseline ${f(r.metrics&&r.metrics.baselineBatteryAccuracy,2)}</span></span>
      <b>distinct-2</b><span>${f(r.metrics&&r.metrics.distinct2,3)}</span>
      </div>`;
    h+= ctl
      ? `<div class="note ok">matched-norm random control: <b>${f(ctl.metricValue)}</b>,
          margin ${f(ctl.margin,2)} — winner beat it by ${f((r.metrics.logprobShift??0)-ctl.metricValue)}</div>`
      : `<div class="note bad"><b>No control was declared.</b> This winner was never required to
          beat a norm-matched random direction, so a small objective cannot be distinguished
          from noise. Set <span class="mono">matchedNormRandomMargin</span> before relying on it.</div>`;
    h+='</div>';
  });
  h+=`<table><tr><th>layer</th>`+alphas.map(a=>`<th class="num">α ${a}</th>`).join('')+`</tr>`;
  layers.forEach(L=>{
    h+=`<tr><td class="mono">${L}</td>`+alphas.map(a=>{
      const c=cells.find(x=>x.layer===L&&x.alpha===a);
      if(!c) return '<td></td>';
      const dead = c.battery!==null&&c.battery<=0.5 || c.distinct2!==null&&c.distinct2<0.45;
      return `<td class="num" title="distinct2 ${f(c.distinct2,3)} · battery ${f(c.battery,2)} · ${f(c.words,0)}w"
        style="background:${dead?'transparent':heat(c.objective,lo,hi)};${dead?'opacity:.4;text-decoration:line-through':''}">
        ${f(c.objective,2)}</td>`;
    }).join('')+`</tr>`;
  });
  h+='</table><div class="note">Struck-through cells failed a constraint (distinct-2 &lt; 0.45 or battery collapse) and were excluded from selection.</div>';
  return h;
}

function report(d){
  const r=d.report; if(!r.conditions) return '';
  let h=`<h3>Conditions</h3><table><tr><th>condition</th><th class="num">generations</th>
     <th class="num">choice rate</th><th class="num">mean words</th><th class="num">distinct-2</th>
     <th class="num">agreement</th></tr>`;
  Object.entries(r.conditions).forEach(([k,v])=>{
    h+=`<tr><td>${esc(k)}</td><td class="num">${v.generations??v.choiceReadouts??'—'}</td>
      <td class="num">${f(v.choiceRate)}</td><td class="num">${f(v.meanWordCount,1)}</td>
      <td class="num">${f(v.meanDistinct2,3)}</td>
      <td class="num">${v.agreementWithBaseline?f(v.agreementWithBaseline.agreement,3):'—'}</td></tr>`;
  });
  return h+'</table><div id="pairs"></div>';
}

async function loadPairs(id){
  const p=await (await fetch('/api/run/'+encodeURIComponent(id)+'/pairs')).json();
  if(!p.prompts||!p.prompts.length) return;
  const el=document.getElementById('pairs'); if(!el) return;
  let h='<h3>Generations — paired by prompt</h3>';
  if(p.saturation){
    const s=p.saturation, sat=s.median>10;
    h+=`<div class="note ${sat?'bad':''}">baseline |log-odds|: median <b>${f(s.median,2)}</b>,
      range ${f(s.min,2)}–${f(s.max,2)}; <b>${s.belowFive} of ${s.n}</b> items below 5.
      ${sat?'<b>The choice instrument is saturated</b> — a null here is not interpretable.':''}</div>`;
  }
  h+=`<div style="margin:8px 0"><input type="search" id="q" placeholder="filter by promptID / arm…">
      <span class="dim" style="margin-left:8px">${p.prompts.length} prompts × ${p.conditions.length} conditions</span></div>
      <div id="plist"></div>`;
  el.innerHTML=h;
  const render=(filter='')=>{
    const rows=p.prompts.filter(x=>!filter||
      (x.promptID+' '+(x.arm||'')).toLowerCase().includes(filter.toLowerCase()));
    document.getElementById('plist').innerHTML=rows.slice(0,60).map(x=>{
      const cards=p.conditions.map(c=>{
        const recs=x.byCondition[c]||[];
        if(!recs.length) return '';
        const r0=recs[0];
        const choice=r0.parsedChoice??r0.selected;
        const hit=choice&&x.target?(choice===x.target?'sig':'ns'):'';
        return `<div class="gen"><b>${esc(c)}</b>
          <span class="badge ${hit}">${esc(choice??'—')}${x.target?` / target ${esc(x.target)}`:''}</span>
          ${r0.margin!==undefined&&r0.margin!==null?`<span class="dim mono">margin ${f(r0.margin,2)}</span>`:''}
          ${recs.length>1?`<span class="dim">${recs.length} samples</span>`:''}
          ${r0.wordCount?`<span class="dim">${r0.wordCount}w</span>`:''}
          ${r0.output?`<pre>${esc(r0.output)}</pre>`:''}</div>`;
      }).join('');
      return `<div class="card"><b class="mono">${esc(x.promptID)}</b>
        <span class="dim">${esc(x.arm||'')}</span>
        <div class="pmt">${esc((x.prompt||'').slice(0,700))}</div>
        <div class="pair">${cards}</div></div>`;
    }).join('') || '<p class="dim">no matches</p>';
  };
  render();
  document.getElementById('q').oninput=e=>render(e.target.value);
}
load();
</script></body></html>
"""


def main():
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("STEERLAB_WORKSPACE", os.getcwd()),
                    help="workspace root (default: $STEERLAB_WORKSPACE or cwd)")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    ROOT = os.path.abspath(os.path.expanduser(args.root))
    if not os.path.isdir(os.path.join(ROOT, "runs")):
        raise SystemExit(f"no runs/ directory under {ROOT} — pass --root <workspace>")
    n = len(list_runs())
    print(f"SteerLab run viewer — {n} runs under {ROOT}")
    print(f"  http://127.0.0.1:{args.port}   (loopback only; ^C to stop)")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
