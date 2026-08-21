"use client";

import { useState } from "react";
import { Badge, ExportButton, ForestRow, NoRunSelected } from "../components/ui";
import { demoPreviewEnabled, effects } from "../lib/demo";
import { findFile } from "../lib/discovery";
import { effectConditions, effectEndpoints, estimandLabel, groupEffects, isDiagnostic, pairedCountLabel, stratumLabel } from "../lib/effects";
import { csvFilename, type ExportColumn } from "../lib/export";
import { fmt } from "../lib/format";
import type { Effect, RunFile, WorkspaceRun } from "../lib/types";

const ALL_CONDITIONS = "All conditions";
const ALL_ENDPOINTS = "All endpoints";

// Every column is read straight out of effect-sizes.csv; the viewer computes
// no statistic here and the export says so column by column. `condition`,
// `stratifyBy`/`stratum` and the estimand pair travel with the numbers —
// without them an exported row cannot be told from another condition's row
// for the same endpoint, or from a within-item diagnostic.
const columns: ExportColumn<Effect>[] = [
  { header: "condition", kind: "stored", value: (row) => row.condition },
  { header: "endpoint", kind: "stored", value: (row) => row.endpoint },
  { header: "stratifyBy", kind: "stored", value: (row) => row.stratifyBy },
  { header: "stratum", kind: "stored", value: (row) => row.stratum },
  { header: "pairedUnit", kind: "stored", value: (row) => row.pairedUnit },
  { header: "estimand", kind: "stored", value: (row) => row.estimand },
  { header: "inference", kind: "stored", value: (row) => row.inference },
  // Absent stays absent on the way out too: a blank `n` is written blank, not
  // as 0 (lib/export.ts writes null as an empty cell).
  { header: "n", kind: "stored", value: (row) => row.n },
  { header: "estimate", kind: "stored", value: (row) => row.estimate },
  { header: "ciLower", kind: "stored", value: (row) => row.low },
  { header: "ciUpper", kind: "stored", value: (row) => row.high },
  { header: "wilcoxonP", kind: "stored", value: (row) => row.p },
  { header: "adjustedP", kind: "stored", value: (row) => row.q },
  { header: "correction", kind: "stored", value: (row) => row.correction },
  { header: "unit", kind: "derived", value: (row) => row.unit },
];

/// The p-value pair for one row. A DIAGNOSTIC row (a single item's own
/// samples) is never shown a corrected p — the engine writes none, and this
/// says why rather than printing a bare dash next to confirmatory rows.
function PValues({ effect }: { effect: Effect }) {
  const diagnostic = isDiagnostic(effect);
  return (
    <>
      <div className="numeric"><strong>{effect.p == null ? "—" : effect.p.toFixed(3)}</strong><span>{effect.p == null ? "not reported" : diagnostic ? "Wilcoxon · locator" : "Wilcoxon"}</span></div>
      <div className="numeric"><strong>{diagnostic || effect.q == null ? "—" : effect.q.toFixed(3)}</strong><span>{diagnostic ? "not corrected" : effect.q == null ? "not reported" : effect.q < .05 ? "survives" : "n.s."}</span></div>
    </>
  );
}

function VerdictBadge({ effect }: { effect: Effect }) {
  if (isDiagnostic(effect)) return <Badge tone="warn">Diagnostic</Badge>;
  return <Badge tone={effect.q != null && effect.q < .05 ? "blue" : "neutral"}>{effect.q == null ? "Not tested" : effect.q < .05 ? "Moves" : "Uncertain"}</Badge>;
}

export function EffectsView({ run, onOpenFile }: { run: WorkspaceRun | null; onOpenFile: (file: RunFile) => void }) {
  const [condition, setCondition] = useState(ALL_CONDITIONS);
  const [endpoint, setEndpoint] = useState(ALL_ENDPOINTS);
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({});
  if (!run && !demoPreviewEnabled()) return <NoRunSelected title="Effects & robustness" />;
  const availableEffects = run ? run.effectRows : effects;
  // Filtering is by CONDITION (the agent) and endpoint — the two halves of a
  // row's identity. Filtering by endpoint alone left every condition's row
  // for that endpoint on screen at once, indistinguishable from each other.
  const visible = availableEffects.filter((effect) =>
    (condition === ALL_CONDITIONS || effect.condition === condition)
    && (endpoint === ALL_ENDPOINTS || effect.short === endpoint));
  const groups = groupEffects(visible);
  const conditions = effectConditions(availableEffects);
  const pooledCount = availableEffects.filter((effect) => effect.stratifyBy === "pooled").length;
  const stratifiedCount = availableEffects.length - pooledCount;
  // The correction family is the table's own stamp. Where the rows disagree
  // (or none stamps one) the column stays generic rather than claiming Holm.
  // Diagnostic rows are excluded — they carry no correction by construction,
  // and an empty stamp from them must not blank the label for the rest.
  const corrections = [...new Set(availableEffects.filter((effect) => !isDiagnostic(effect)).map((effect) => effect.correction).filter(Boolean))];
  const correctionLabel = corrections.length === 1 ? `${corrections[0]} p` : "Adjusted p";
  return (
    <div className="view-enter inner-view">
      <header className="page-title">
        <div><span className="section-number">{run ? `${pooledCount} POOLED ROW${pooledCount === 1 ? "" : "S"}${stratifiedCount ? ` · ${stratifiedCount} STRATIFIED` : ""} · LOCAL RUN` : "CONFIRMATORY FAMILY · 5 ENDPOINTS"}</span><h1>Effects &amp; robustness</h1><p>Paired intervention-minus-baseline estimates, one row per condition × endpoint. The item—not the generation—is the unit of analysis, except where a stratum says otherwise.</p></div>
        <div className="title-actions"><button className="secondary" onClick={() => document.querySelector(run ? ".local-method-note" : ".table-note")?.scrollIntoView({ behavior: "smooth" })}>Method notes</button><ExportButton filename={csvFilename(run?.name ?? "synthetic-preview", "effect-sizes")} columns={columns} rows={visible} /><button className="primary" disabled={!run || !findFile(run.files, "effect-sizes.csv")} onClick={() => { const file = run && findFile(run.files, "effect-sizes.csv"); if (file) onOpenFile(file); }}>{run ? "Open table" : "Preview table"} <span>→</span></button></div>
      </header>
      <section className="filterbar" aria-label="Effect filters">
        <label>Source<select disabled><option>{run ? run.name : "Synthetic preview"}</option></select></label>
        <label>Experiment<select disabled><option>{run ? run.experiment : "Alien stance"}</option></select></label>
        {conditions.length > 0 && <label>Condition<select value={condition} onChange={(event) => setCondition(event.target.value)}><option>{ALL_CONDITIONS}</option>{conditions.map((name) => <option key={name}>{name}</option>)}</select></label>}
        <label>Endpoint<select value={endpoint} onChange={(event) => setEndpoint(event.target.value)}><option>{ALL_ENDPOINTS}</option>{effectEndpoints(availableEffects).map((name) => <option key={name}>{name}</option>)}</select></label>
        <div className="filter-summary"><span>Correction</span><strong>{corrections.length === 1 ? corrections[0] : corrections.length ? corrections.join(" / ") : "Not stamped in this table"}</strong></div>
      </section>
      <section className="card effect-table-card">
        <div className="effect-table-head"><span>Condition · endpoint</span><span>Effect with 95% CI</span><span>Estimate</span><span>Raw p</span><span>{correctionLabel}</span><span>Read</span></div>
        {groups.map((group) => {
          const parent = group.pooled ?? group.strata[0];
          const nested = group.pooled ? group.strata : group.strata.slice(1);
          const open = openGroups[group.key] ?? false;
          return (
            <div className="effect-group" key={group.key}>
              <div className="effect-table-row" key={parent.key}>
                <div>
                  <strong>{parent.endpoint}</strong>
                  <span>{parent.condition ? `${parent.condition} · ` : ""}{parent.unit} · {pairedCountLabel(parent)}</span>
                  {/* Collapsed by default — a promptID family is one row per
                      item per endpoint — but the count is stated, and so is
                      how many of them are within-item diagnostics, so nothing
                      is hidden behind an unlabelled toggle. Both numbers are
                      counts of stamped rows, not a derived claim. */}
                  {nested.length > 0 && <button className="quiet-link strata-toggle" aria-expanded={open} onClick={() => setOpenGroups((state) => ({ ...state, [group.key]: !open }))}>{open ? "Hide" : "Show"} {nested.length} stratified row{nested.length === 1 ? "" : "s"}{nested.filter(isDiagnostic).length ? ` · ${nested.filter(isDiagnostic).length} diagnostic` : ""}</button>}
                </div>
                <ForestRow effect={parent} compact />
                <div className="numeric"><strong>{fmt(parent.estimate, parent.unit === "months" ? 1 : 2)}</strong><span>[{fmt(parent.low)}, {fmt(parent.high)}]</span></div>
                {/* The raw p is the table's stamped `wilcoxonP`. This column used
                    to print a hardcoded five-value array for the synthetic
                    preview and a bare dash for every real run — a fabricated
                    statistic beside real ones. */}
                <PValues effect={parent} />
                <VerdictBadge effect={parent} />
              </div>
              {open && nested.map((stratum) => (
                <div className="effect-table-row effect-stratum-row" key={stratum.key}>
                  <div>
                    <strong>{stratumLabel(stratum)}</strong>
                    <span>{pairedCountLabel(stratum)}{estimandLabel(stratum) ? ` · ${estimandLabel(stratum)}` : ""}</span>
                  </div>
                  <ForestRow effect={stratum} compact />
                  <div className="numeric"><strong>{fmt(stratum.estimate, stratum.unit === "months" ? 1 : 2)}</strong><span>[{fmt(stratum.low)}, {fmt(stratum.high)}]</span></div>
                  <PValues effect={stratum} />
                  <VerdictBadge effect={stratum} />
                </div>
              ))}
            </div>
          );
        })}
        {groups.length === 0 && <div className="artifact-empty"><span>∅</span><p>No readable effect rows were found for this run.</p></div>}
        <footer className="table-note"><strong>Interpretation.</strong> {run ? "Values are read directly from effect-sizes.csv; absent fields remain absent. Stratified rows are the engine’s per-cell companions to the pooled row above them: an “itemLevel” stratum is the pooled estimate restricted to that cell, while a “withinItemSamples” stratum compares one prompt’s own generations — a prompt-specific quantity that supports no cross-prompt claim, so the engine leaves it out of every correction family and it is shown here as a diagnostic locator only." : "CIs are percentile bootstrap intervals over paired item-level differences (10,000 resamples; seed 0). Two-sided Wilcoxon signed-rank p-values are a robustness companion, adjusted over the five-endpoint confirmatory family with Holm’s method."}</footer>
      </section>

      {!run && <section className="section-grid effects-lower">
        <div className="card residual-card">
          <header className="section-header"><div><span className="section-number">HUMAN-ANCHORED RESIDUAL</span><h2>Model movement exceeds the provisional human estimate.</h2></div><Badge tone="warn">Unverified baseline</Badge></header>
          <div className="equation"><span>R</span><small>=</small><strong>Δ<sub>model</sub></strong><small>−</small><strong>Δ<sub>human</sub></strong></div>
          <div className="residual-values"><div><span>Model</span><strong>+7.8</strong><small>months</small></div><div><span>Human</span><strong>+2.1</strong><small>months</small></div><div className="accent"><span>Residual R</span><strong>+5.7</strong><small>“hyper-human”</small></div></div>
          <p>Classification is shown to demonstrate the analysis surface only. It cannot support a human-comparison claim until the source table, page, extraction notes, and file hash are verified.</p>
        </div>
        <div className="card sensitivity-card">
          <header className="section-header"><div><span className="section-number">SENSITIVITY</span><h2>The primary estimate is stable.</h2></div></header>
          <div className="sensitivity-row"><span>All paired items</span><div><i style={{width:"65%"}}/></div><strong>+7.8</strong></div>
          <div className="sensitivity-row"><span>Exclude parse failures</span><div><i style={{width:"68%"}}/></div><strong>+8.1</strong></div>
          <div className="sensitivity-row"><span>Median per item</span><div><i style={{width:"58%"}}/></div><strong>+7.0</strong></div>
          <div className="sensitivity-row"><span>Winsorize 5%</span><div><i style={{width:"62%"}}/></div><strong>+7.4</strong></div>
          <footer>All analyses use the same frozen item set. No post-hoc endpoint exclusions.</footer>
        </div>
      </section>}
      {run && <section className="card local-method-note"><span className="section-number">LOCAL ARTIFACT CONTRACT</span><h2>No statistics are recomputed in the browser.</h2><p>The explorer presents the run’s saved estimates and intervals. It does not silently derive missing tests, correction families, or human residuals from partial files — and it does not pool a stratified row back into its parent, which would be a hierarchical model no artifact declared.</p></section>}
    </div>
  );
}
