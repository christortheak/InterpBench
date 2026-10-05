"use client";

import { useState } from "react";
import { Badge, NoRunSelected } from "../components/ui";
import { DemoProvenance, demoPreviewEnabled } from "../demo";
import { textValue } from "../lib/discovery";
import { freezeLabel, freezeOf, gateLabel } from "../lib/freeze";
import type { RunFile, WorkspaceRun } from "../lib/types";

export function LocalProvenanceView({ run, onOpenFile }: { run: WorkspaceRun; onOpenFile: (file: RunFile) => void }) {
  const [fileSearch, setFileSearch] = useState("");
  const visibleFiles = run.files.filter((file) => file.path.toLowerCase().includes(fileSearch.toLowerCase()));
  // The study's freeze state, from the manifest snapshot this run carries
  // (lib/freeze.ts). A forced freeze names the checks it skipped; a plain
  // one says none were.
  const freeze = freezeOf(run);
  const entries = [
    ["Experiment", run.experiment],
    ["Model", run.model],
    ["Run directory", run.path],
    ["Status", run.status],
    ["Prompt count", run.promptCount || "Not stamped"],
    ["Condition count", run.conditionCount || "Not stamped"],
    ["Experiment hash", textValue(run.report, "experimentHash") || textValue(run.config, "experimentHash") || "Not stamped"],
    ["Model revision", textValue(run.config, "modelRevision", "revision") || "Not stamped"],
    ["Study state", freezeLabel(freeze)],
    ...(freeze.frozen ? [
      ["Frozen at", freeze.frozenAt || "Not stamped"],
      ["Checks skipped by force", freeze.forced ? freeze.forcedGates.map(gateLabel).join("; ") || "Forced, but none listed" : "None"],
    ] : []),
    ...(freeze.present ? [
      ["Capability check not applied to", freeze.batteryNotApplied.map((entry) => `${entry.condition}${entry.reason ? ` (${entry.reason})` : ""}`).join("; ") || "None recorded"],
    ] : []),
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
          <div className="file-browser-list">{visibleFiles.map((file) => <button className="artifact" key={file.path} onClick={() => onOpenFile(file)}><span className="file-icon">↳</span><div><strong>{file.name}</strong><span>{file.path}{file.name === "generations.jsonl" ? ` · ${run.generationRows.length} records loaded${run.skippedGenerationLines ? ` · ${run.skippedGenerationLines} line${run.skippedGenerationLines === 1 ? "" : "s"} skipped` : ""}` : file.name === "effect-sizes.csv" ? ` · ${run.effectRows.length} effect rows${run.skippedEffectRows ? ` · ${run.skippedEffectRows} row${run.skippedEffectRows === 1 ? "" : "s"} skipped` : ""}` : ""}</span></div><small>{file.size < 1024 ? `${file.size} B` : file.size < 1024 * 1024 ? `${(file.size / 1024).toFixed(1)} KB` : `${(file.size / 1024 / 1024).toFixed(1)} MB`}</small><b>→</b></button>)}</div>
          {!visibleFiles.length && <div className="empty-state">No files match that filter.</div>}
        </div>
      </section>
      <section className="card raw-stamps-card"><span className="section-number">PRESENTATION RULE</span><h2>Absence is visible.</h2><p>If a revision, hash, correction, or validation artifact is absent from the saved files, the explorer says “Not stamped.” It does not manufacture a green gate from a directory name or a completed status.</p></section>
    </div>
  );
}

export function ProvenanceView({ run, onOpenFile }: { run: WorkspaceRun | null; onOpenFile: (file: RunFile) => void }) {
  if (run) return <LocalProvenanceView run={run} onOpenFile={onOpenFile} />;
  // The layout preview's page lives in app/demo/, which the embedded app
  // does not contain. It used to be written out here, with a "Run epoch
  // verified" banner over invented checks.
  if (!demoPreviewEnabled()) return <NoRunSelected title="Run files & provenance" />;
  return <DemoProvenance />;
}
