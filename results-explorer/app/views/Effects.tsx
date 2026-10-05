"use client";

import { useState } from "react";
import { AnalysisStampsCard, IntervalNote } from "../components/stamps";
import { Badge, ExportButton, ForestRow, NoRunSelected } from "../components/ui";
import { analysisStampsOf } from "../lib/analysisStamps";
import { skippedLinesNote } from "../lib/csv";
import { demoCopy, demoEffects, DemoBanner, DemoEffectsLower, demoPreviewEnabled } from "../demo";
import { findFile } from "../lib/discovery";
import { effectConditions, effectEndpoints, estimandLabel, groupEffects, isDiagnostic, pairedCountLabel, stratumLabel } from "../lib/effects";
import { effectsUnitSummary, unitCaveat, unitOf } from "../lib/effectUnits";
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
//
// The descriptions travel in the "Column notes" file (lib/export.ts). This is
// the table most likely to be opened in statistics software, so every column
// says what it holds.
const columns: ExportColumn<Effect>[] = [
  { header: "condition", kind: "stored", value: (row) => row.condition, description: "The condition being compared with the baseline." },
  { header: "endpoint", kind: "stored", value: (row) => row.endpoint, description: "The outcome being compared." },
  { header: "stratifyBy", kind: "stored", value: (row) => row.stratifyBy, description: "\"pooled\" for a row over all items. Otherwise, the grouping this row is restricted to." },
  { header: "stratum", kind: "stored", value: (row) => row.stratum, description: "The group within stratifyBy. Empty on pooled rows." },
  { header: "pairedUnit", kind: "stored", value: (row) => row.pairedUnit, description: "On a stratified row, what one paired difference is: item or sample (the file's \"unit\" column). Empty on pooled rows, where it is the run's unit of analysis." },
  { header: "estimand", kind: "stored", value: (row) => row.estimand, description: "On a stratified row, itemLevel or withinItemSamples. A withinItemSamples row describes one prompt's own generations and supports no claim about other prompts." },
  { header: "inference", kind: "stored", value: (row) => row.inference, description: "On a stratified row, corrected or diagnostic. A diagnostic row is a locator, not a test, and has no adjusted p." },
  // Absent stays absent on the way out too: a blank `n` is written blank, not
  // as 0 (lib/export.ts writes null as an empty cell).
  { header: "n", kind: "stored", value: (row) => row.n, description: "The number of paired differences behind the estimate. Empty when the file gave none." },
  { header: "estimate", kind: "stored", value: (row) => row.estimate, description: "The mean paired difference, condition minus baseline (the file's deltaMean)." },
  { header: "ciLower", kind: "stored", value: (row) => row.low, description: "The lower end of the 95% interval the engine stored." },
  { header: "ciUpper", kind: "stored", value: (row) => row.high, description: "The upper end of the 95% interval the engine stored." },
  { header: "wilcoxonP", kind: "stored", value: (row) => row.p, description: "The unadjusted Wilcoxon signed-rank p-value. Empty when the file gave none." },
  { header: "adjustedP", kind: "stored", value: (row) => row.q, description: "The p-value after the correction named in the correction column. Empty when the file gave none." },
  { header: "correction", kind: "stored", value: (row) => row.correction, description: "The multiple-comparison correction the engine applied, as the file names it." },
  { header: "unit", kind: "derived", value: (row) => row.unit, description: "A display unit the explorer chose from the endpoint's name. It is a label for reading, not a measured unit." },
  // The run's own stamp, repeated on every row so the table still says what
  // `n` counts once it has left the run directory.
  { header: "unitOfAnalysis", kind: "stored", value: (row) => row.analysisUnit ?? "", description: "The run's unit of analysis as the run stamps it, such as transcript: what n counts on a pooled row. Empty when the run stamps none; unitResolved then says what the run's records show." },
  // The unit each row is read in, settled the way `results export` settles
  // it (lib/effectUnits.ts), so the exported table names the same unit.
  { header: "unitResolved", kind: "derived", value: (row) => unitOf(row).unit, description: "What one paired difference is: an item (its samples averaged), a transcript (one play-through of a multi-agent conversation), a sample, a response (one response paired with the baseline response to the same item and seed; responses to the same item are not independent), or unknown." },
  { header: "unitSource", kind: "derived", value: (row) => unitOf(row).source, description: "Where the unit comes from: recorded when the row or the analysis stamped it; engine_default when it did not and the run's records agree with the engines' rule, the item; inferred_from_records when the row counts more pairs than the items paired in the run, so it paired responses; and not_established when the records cannot tell." },
  { header: "pairedItems", kind: "derived", value: (row) => unitOf(row).pairedItems, description: "How many distinct items the run's records answer under both this row's condition and the baseline, counted by the explorer from generations.jsonl. Empty when the records were not read or hold none for the condition." },
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
  // With no run, the only rows are the layout preview's invented ones, and
  // only in a browser that asked for the preview: app/demo/ is not in the
  // embedded app, where this list is always empty.
  const availableEffects = run ? run.effectRows : demoEffects;
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
      {!run && <DemoBanner />}
      <header className="page-title">
        {/* What one paired difference IS comes from the run's own stamp. A
            multi-agent run pairs whole transcripts, and saying "the item"
            there described a different analysis than the one on screen. */}
        <div><span className="section-number">{run ? `${pooledCount} POOLED ROW${pooledCount === 1 ? "" : "S"}${stratifiedCount ? ` · ${stratifiedCount} STRATIFIED` : ""} · LOCAL RUN` : demoCopy.effectsEyebrow}</span><h1>Effects &amp; robustness</h1><p>Paired intervention-minus-baseline estimates, one row per condition × endpoint. {run ? effectsUnitSummary(run.effectRows, analysisStampsOf(run).unit?.unit ?? "") : "The item—not the generation—is the unit of analysis, except where a stratum says otherwise."}</p></div>
        <div className="title-actions"><button className="secondary" onClick={() => document.querySelector(run && run.effectRows.length ? ".interval-note" : run ? ".local-method-note" : ".table-note")?.scrollIntoView({ behavior: "smooth" })}>Method notes</button><ExportButton filename={csvFilename(run?.name ?? "invented-demo-data", "effect-sizes")} columns={columns} rows={visible} /><button className="primary" disabled={!run || !findFile(run.files, "effect-sizes.csv")} onClick={() => { const file = run && findFile(run.files, "effect-sizes.csv"); if (file) onOpenFile(file); }}>{run ? "Open table" : "Preview table"} <span>→</span></button></div>
      </header>
      <section className="filterbar" aria-label="Effect filters">
        <label>Source<select disabled><option>{run ? run.name : demoCopy.effectsSource}</option></select></label>
        <label>Experiment<select disabled><option>{run ? run.experiment : demoCopy.effectsExperiment}</option></select></label>
        {conditions.length > 0 && <label>Condition<select value={condition} onChange={(event) => setCondition(event.target.value)}><option>{ALL_CONDITIONS}</option>{conditions.map((name) => <option key={name}>{name}</option>)}</select></label>}
        <label>Endpoint<select value={endpoint} onChange={(event) => setEndpoint(event.target.value)}><option>{ALL_ENDPOINTS}</option>{effectEndpoints(availableEffects).map((name) => <option key={name}>{name}</option>)}</select></label>
        <div className="filter-summary"><span>Correction</span><strong>{corrections.length === 1 ? corrections[0] : corrections.length ? corrections.join(" / ") : "Not stamped in this table"}</strong></div>
      </section>
      {/* The stamps that say what the estimates below were measured on:
          the unit of analysis, exclusions, endpoint rescue, adjudication. */}
      {run && <AnalysisStampsCard run={run} onOpenFile={onOpenFile} />}
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
                  {/* A row of paired responses, or one whose unit nothing
                      settles, says so beside its count. */}
                  {run && unitCaveat(parent) && <small className="unit-caveat">{unitCaveat(parent)}</small>}
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
        {/* A line with no readable estimate or interval is left out, never
            patched up — and the table says it is short. */}
        {run && (run.skippedEffectRows ?? 0) > 0 && <div className="preview-warning skipped-note" role="status">{skippedLinesNote(run.skippedEffectRows ?? 0, "effect-sizes.csv", "effect rows (a row needs an endpoint, an estimate, and both ends of its interval)")}</div>}
        <footer className="table-note"><strong>Interpretation.</strong> {run ? "Values are read directly from effect-sizes.csv; absent fields remain absent. Stratified rows are the engine’s per-cell companions to the pooled row above them: an “itemLevel” stratum is the pooled estimate restricted to that cell, while a “withinItemSamples” stratum compares one prompt’s own generations — a prompt-specific quantity that supports no cross-prompt claim, so the engine leaves it out of every correction family and it is shown here as a diagnostic locator only." : demoCopy.effectsNote}</footer>
      </section>

      {/* The plain-language reading of an interval. It used to be shown only
          beside the invented demo rows; a reader needs it beside real ones. */}
      {run && run.effectRows.length > 0 && <IntervalNote unit={analysisStampsOf(run).unit} corrections={corrections} rows={run.effectRows} records={run.effectUnitRecords ?? null} />}
      {!run && <DemoEffectsLower />}
      {run && <section className="card local-method-note"><span className="section-number">LOCAL ARTIFACT CONTRACT</span><h2>No statistics are recomputed in the browser.</h2><p>The explorer presents the run’s saved estimates and intervals. It does not silently derive missing tests, correction families, or human residuals from partial files — and it does not pool a stratified row back into its parent, which would be a hierarchical model no artifact declared.</p></section>}
    </div>
  );
}
