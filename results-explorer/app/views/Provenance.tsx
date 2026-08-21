"use client";

import { useState } from "react";
import { Badge, NoRunSelected } from "../components/ui";
import { demoPreviewEnabled } from "../lib/demo";
import { textValue } from "../lib/discovery";
import type { RunFile, WorkspaceRun } from "../lib/types";

export function LocalProvenanceView({ run, onOpenFile }: { run: WorkspaceRun; onOpenFile: (file: RunFile) => void }) {
  const [fileSearch, setFileSearch] = useState("");
  const visibleFiles = run.files.filter((file) => file.path.toLowerCase().includes(fileSearch.toLowerCase()));
  const entries = [
    ["Experiment", run.experiment],
    ["Model", run.model],
    ["Run directory", run.path],
    ["Status", run.status],
    ["Prompt count", run.promptCount || "Not stamped"],
    ["Condition count", run.conditionCount || "Not stamped"],
    ["Experiment hash", textValue(run.report, "experimentHash") || textValue(run.config, "experimentHash") || "Not stamped"],
    ["Model revision", textValue(run.config, "modelRevision", "revision") || "Not stamped"],
  ];
  return (
    <div className="view-enter inner-view provenance-view">
      <header className="page-title"><div><span className="section-number">LOCAL WORKSPACE · READ ONLY</span><h1>Provenance & files</h1><p>Metadata shown exactly as stored in the selected run. The browser does not infer verification status from missing stamps.</p></div></header>
      <section className="audit-banner local-audit"><span className="audit-mark">⌂</span><div><strong>Reading {run.name}</strong><p>Permission is scoped to the workspace folder selected for this local session.</p></div><Badge tone="blue">Local</Badge></section>
      <section className="provenance-grid">
        <div className="card config-card">
          <header className="section-header"><div><span className="section-number">RUN IDENTITY</span><h2>Stored metadata</h2></div></header>
          <dl>{entries.map(([label, value]) => <div key={String(label)}><dt>{label}</dt><dd>{String(value)}</dd></div>)}</dl>
        </div>
        <div className="card artifact-card local-artifact-list">
          <header className="section-header"><div><span className="section-number">DIRECTORY CONTENTS</span><h2>{run.files.length} artifacts</h2></div><span className="muted">Click any file to preview</span></header>
          <label className="file-search"><span>⌕</span><input value={fileSearch} onChange={(event) => setFileSearch(event.target.value)} placeholder="Filter files or nested paths" /></label>
          <div className="file-browser-list">{visibleFiles.map((file) => <button className="artifact" key={file.path} onClick={() => onOpenFile(file)}><span className="file-icon">↳</span><div><strong>{file.name}</strong><span>{file.path}{file.name === "generations.jsonl" ? ` · ${run.generationRows.length} records loaded` : file.name === "effect-sizes.csv" ? ` · ${run.effectRows.length} effect rows` : ""}</span></div><small>{file.size < 1024 ? `${file.size} B` : file.size < 1024 * 1024 ? `${(file.size / 1024).toFixed(1)} KB` : `${(file.size / 1024 / 1024).toFixed(1)} MB`}</small><b>→</b></button>)}</div>
          {!visibleFiles.length && <div className="empty-state">No files match that filter.</div>}
        </div>
      </section>
      <section className="card raw-stamps-card"><span className="section-number">PRESENTATION RULE</span><h2>Absence is visible.</h2><p>If a revision, hash, correction, or validation artifact is absent from the saved files, the explorer says “Not stamped.” It does not manufacture a green gate from a directory name or a completed status.</p></section>
    </div>
  );
}

export function ProvenanceView({ run, onOpenFile }: { run: WorkspaceRun | null; onOpenFile: (file: RunFile) => void }) {
  if (run) return <LocalProvenanceView run={run} onOpenFile={onOpenFile} />;
  if (!demoPreviewEnabled()) return <NoRunSelected title="Run files & provenance" />;
  const artifacts = [
    ["report.json", "Run summary + condition diagnostics", "12.8 KB"],
    ["generations.jsonl", "384 complete generation records", "1.8 MB"],
    ["effect-sizes.csv", "Paired estimates + inferential tests", "3.4 KB"],
    ["alien-residuals.csv", "Human-anchored residual classifications", "1.1 KB"],
    ["exclusions.json", "Declared exclusions and counts", "0.7 KB"],
    ["battery.jsonl", "Per-condition capability checks", "42.2 KB"],
  ];
  return (
    <div className="view-enter inner-view provenance-view">
      <header className="page-title">
        <div><span className="section-number">IMMUTABLE RUN · SCHEMA 2</span><h1>Provenance & validity</h1><p>Everything needed to audit what ran, what was measured, and what this result is allowed to claim.</p></div>
        <button className="primary">Export audit bundle <span>↓</span></button>
      </header>
      <section className="audit-banner"><span className="audit-mark">✓</span><div><strong>Run epoch verified</strong><p>The experiment hash matches the frozen manifest. Measurement inputs and model revision are pinned.</p></div><Badge tone="good">5 / 6 gates closed</Badge></section>
      <section className="provenance-grid">
        <div className="card config-card">
          <header className="section-header"><div><span className="section-number">RUN IDENTITY</span><h2>Configuration</h2></div><Badge tone="blue">Frozen</Badge></header>
          <dl>
            <div><dt>Experiment</dt><dd>alien-stance-emotion-confirm-01</dd></div>
            <div><dt>Model</dt><dd>google/gemma-3-27b-it</dd></div>
            <div><dt>Revision</dt><dd><code>2f3c8a7…91d4</code></dd></div>
            <div><dt>Engine</dt><dd>Python · PyTorch/HF · CUDA</dd></div>
            <div><dt>Intervention</dt><dd>Vector injection · layer 38 · +1.0σ</dd></div>
            <div><dt>Sampling</dt><dd>temperature 0.7 · 5 samples/item</dd></div>
            <div><dt>Seed policy</dt><dd>Per-record deterministic isolation</dd></div>
            <div><dt>Unit of analysis</dt><dd>Paired prompt item</dd></div>
          </dl>
        </div>
        <div className="card gates-card">
          <header className="section-header"><div><span className="section-number">CIRCULARITY FIREWALL</span><h2>Evidence gates</h2></div></header>
          <div className="gate"><i>✓</i><div><strong>Experiment frozen</strong><span>One-way freeze · no forced gates</span></div><Badge tone="good">Closed</Badge></div>
          <div className="gate"><i>✓</i><div><strong>Validation evidence</strong><span>Held-out probe movement + independence screen</span></div><Badge tone="good">Closed</Badge></div>
          <div className="gate"><i>✓</i><div><strong>Measurement inputs</strong><span>Prompts, markers, rubric, neutral corpus pinned</span></div><Badge tone="good">Closed</Badge></div>
          <div className="gate"><i>✓</i><div><strong>Capability retention</strong><span>Every condition within preregistered threshold</span></div><Badge tone="good">Closed</Badge></div>
          <div className="gate"><i>✓</i><div><strong>Multiple comparisons</strong><span>Holm correction on confirmatory family</span></div><Badge tone="good">Closed</Badge></div>
          <div className="gate gate-open"><i>!</i><div><strong>Human baseline source</strong><span>Demonstration placeholder; not hash-pinned</span></div><Badge tone="warn">Open</Badge></div>
        </div>
      </section>
      <section className="section-grid provenance-lower">
        <div className="card artifact-card">
          <header className="section-header"><div><span className="section-number">ARTIFACTS</span><h2>Files in this run</h2></div><span className="muted">All checksums valid</span></header>
          {artifacts.map(([name, description, size]) => <button className="artifact" key={name}><span className="file-icon">↳</span><div><strong>{name}</strong><span>{description}</span></div><small>{size}</small><b>→</b></button>)}
        </div>
        <div className="card methods-card">
          <span className="section-number">STATISTICAL NOTE</span>
          <h2>What the interval means</h2>
          <p>The 95% interval is a percentile bootstrap over <strong>same-item intervention-minus-baseline differences</strong>. It quantifies uncertainty across prompt items, not token-level or generation-level variability.</p>
          <div className="method-item"><span>10,000</span><small>bootstrap resamples</small></div>
          <div className="method-item"><span>0</span><small>deterministic analysis seed</small></div>
          <div className="method-item"><span>Holm</span><small>confirmatory family correction</small></div>
          <footer>Wilcoxon signed-rank is reported as a nonparametric robustness companion; the paired mean and bootstrap interval remain the primary estimand.</footer>
        </div>
      </section>
    </div>
  );
}
