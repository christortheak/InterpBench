"use client";

import { FreezeNotice, TruncationCard } from "../components/stamps";
import { Badge, ForestRow, NoRunSelected } from "../components/ui";
import { DemoOverview, demoPreviewEnabled } from "../demo";
import { runKindOf, runStatusOf } from "../lib/discovery";
import { fmt } from "../lib/format";
import { freezeLabel, freezeOf } from "../lib/freeze";
import { conditionErrors } from "../lib/runReport";
import { runKindLabel } from "../lib/runKind";
import { statusLabel, statusTone } from "../lib/status";
import type { View, WorkspaceRun } from "../lib/types";

const record = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
const numberOf = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

/// The per-condition block of report.json, rendered as the table it always
/// was (upgrade plan Phase 0). Every value here is STORED — read verbatim
/// from the run's own report — so nothing carries a derived badge. A column
/// whose values are absent for every condition is OMITTED rather than filled
/// with dashes: the engines write these keys only when the corresponding
/// instrument ran, and an empty column would read as a measurement that
/// returned nothing.
type ConditionRow = {
  name: string;
  generations: number | null;
  meanWordCount: number | null;
  meanDistinct2: number | null;
  choiceRate: number | null;
  choiceReadouts: number | null;
  agreement: number | null;
  agreementN: number | null;
  batteryAccuracy: number | null;
  /// The condition's `error`, when the engine recorded one in place of its
  /// numbers; "" otherwise.
  error: string;
};

type NumericColumn = Exclude<keyof ConditionRow, "name" | "error">;

function ConditionsTable({ run }: { run: WorkspaceRun }) {
  const conditions = record(run.report.conditions);
  const names = Object.keys(conditions).sort();
  if (!names.length) return null;
  // A condition that FAILED carries an `error` and no generations. It used
  // to show as a row of dashes beside the conditions that ran, which reads
  // as "measured, nothing found" rather than "this arm never ran".
  const errors = conditionErrors(run.report);
  const errorFor = (name: string) => errors.find((entry) => entry.condition === name)?.error ?? "";
  const rows: ConditionRow[] = names.map((name) => {
    const row = record(conditions[name]);
    const agreement = record(row.agreementWithBaseline);
    const battery = record(row.capabilityBattery);
    return {
      name,
      error: errorFor(name),
      generations: numberOf(row.generations),
      meanWordCount: numberOf(row.meanWordCount),
      meanDistinct2: numberOf(row.meanDistinct2),
      choiceRate: numberOf(row.choiceRate),
      choiceReadouts: numberOf(row.choiceReadouts),
      agreement: numberOf(agreement.agreement),
      agreementN: numberOf(agreement.n),
      batteryAccuracy: numberOf(battery.accuracy) ?? numberOf(row.capabilityAccuracy),
    };
  });
  const present = (key: NumericColumn) => rows.some((row) => row[key] !== null);
  const allColumns: Array<{ key: NumericColumn; label: string; render: (row: ConditionRow) => string }> = [
    { key: "generations", label: "Generations", render: (row) => row.generations === null ? "—" : String(row.generations) },
    { key: "meanWordCount", label: "Mean words", render: (row) => row.meanWordCount === null ? "—" : row.meanWordCount.toFixed(1) },
    { key: "meanDistinct2", label: "Mean distinct-2", render: (row) => row.meanDistinct2 === null ? "—" : row.meanDistinct2.toFixed(3) },
    { key: "choiceRate", label: "Choice rate", render: (row) => row.choiceRate === null ? "—" : row.choiceRate.toFixed(3) },
    { key: "choiceReadouts", label: "Choice readouts", render: (row) => row.choiceReadouts === null ? "—" : String(row.choiceReadouts) },
    { key: "agreement", label: "Agreement w/ baseline", render: (row) => row.agreement === null ? "—" : `${(row.agreement * 100).toFixed(1)}%${row.agreementN === null ? "" : ` (n = ${row.agreementN})`}` },
    { key: "batteryAccuracy", label: "Battery accuracy", render: (row) => row.batteryAccuracy === null ? "—" : `${(row.batteryAccuracy * 100).toFixed(1)}%` },
  ];
  const columns = allColumns.filter((column) => present(column.key));
  return (
    <section className="card" aria-label="Per-condition summary">
      <header className="section-header">
        <div><span className="section-number">PER-CONDITION SUMMARY</span><h2>{names.length} condition{names.length === 1 ? "" : "s"} as reported</h2></div>
        {errors.length > 0 && <Badge tone="warn">{errors.length} failed</Badge>}
      </header>
      {errors.length > 0 && (
        <div className="notice condition-errors" role="alert">
          <span className="notice-icon">!</span>
          <div>
            <p><strong>{errors.length === 1 ? "1 condition failed" : `${errors.length} conditions failed`} and produced no generations.</strong> There is nothing to compare with the baseline for {errors.length === 1 ? "it" : "them"}, so no effect is reported for {errors.length === 1 ? "it" : "them"}. The engine recorded:</p>
            <ul>{errors.map((entry) => <li key={entry.condition}><strong>{entry.condition}</strong>: <code>{entry.error}</code></li>)}</ul>
          </div>
        </div>
      )}
      <div className="raw-table-scroll">
        <table className="raw-table">
          <thead><tr><th>Condition</th>{columns.map((column) => <th key={String(column.key)}>{column.label}</th>)}</tr></thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.name} className={row.error ? "condition-failed" : ""}>
                <td>{row.name}{row.error ? " · failed" : ""}</td>
                {columns.map((column) => <td key={String(column.key)}>{column.render(row)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="table-note">Stored values, read from <code>report.json</code>. Columns the run did not stamp are not shown.</p>
    </section>
  );
}

export function LocalOverview({ run, onNavigate }: { run: WorkspaceRun; onNavigate: (view: View) => void }) {
  // The overview headline is the POOLED rows only. The stratified companion
  // rows (2026-08-06) belong under their parent in the Effects view, where
  // their estimand and diagnostic status are visible; a top-five slice that
  // silently mixed a within-item diagnostic in beside pooled estimates would
  // read as five comparable findings.
  const pooledEffects = run.effectRows.filter((effect) => effect.stratifyBy === "pooled");
  const shownEffects = pooledEffects.slice(0, 5);
  const primary = shownEffects[0];
  const hasConceptEvidence = run.validationConcepts.length > 0 || run.cosineMatrices.length > 0;
  const hasOptimization = run.sweepRows.length > 0;
  const hasPanel = run.panelEffects.length > 0 || run.generationRows.some((record) => record.speakerName || record.turnTitle);
  const primaryView: View = hasOptimization ? "optimization" : hasPanel ? "panels" : hasConceptEvidence ? "concepts" : run.effectRows.length ? "effects" : "generations";
  const primaryLabel = hasOptimization ? "Inspect optimization" : hasPanel ? "Inspect panel dynamics" : hasConceptEvidence ? "Inspect concept evidence" : run.effectRows.length ? "Inspect effects" : "Inspect generations";
  const conditionCount = run.conditionCount || (run.report.conditions && typeof run.report.conditions === "object" ? Object.keys(run.report.conditions as object).length : 0);
  // Status truth, not the old report.status literal. A failure record must
  // never read as a sparse success, so it says so before anything else on
  // the page (upgrade plan Phase 0).
  const status = runStatusOf(run);
  const kind = runKindOf(run);
  const unfinished = status.state === "failed" || status.state === "cancelled" || status.state === "partial" || status.state === "inProgress";
  return (
    <div className="view-enter">
      {unfinished && (
        <div className="notice" role="alert">
          <span className="notice-icon">!</span>
          <p>
            <strong>This run {status.state === "failed" ? "FAILED" : status.state === "cancelled" ? "was CANCELLED" : status.state === "inProgress" ? "is still IN PROGRESS" : "is PARTIAL"}.</strong>{" "}
            {status.error ? <>The stage recorded: <code>{status.error}</code>{" "}</> : null}
            Whatever is below was produced before it stopped and is a retention record, not a result.
            {status.itemsWritten !== null ? ` ${status.itemsWritten} ${status.itemLabel || "item"}${status.itemsWritten === 1 ? "" : "s"} written.` : ""}
            {status.pendingUnits.length ? ` Did not run: ${status.pendingUnits.join(", ")}.` : ""}
          </p>
        </div>
      )}
      <section className="local-run-hero">
        <div>
          <div className="kicker">
            <span>{runKindLabel(kind)}</span><span>·</span><span>{statusLabel(status)}</span>
            {status.stage ? <><span>·</span><span>stage {status.stage}</span></> : null}
            <span>·</span><span>{freezeLabel(freezeOf(run))}</span>
          </div>
          <h1>{run.experiment}</h1>
          <p>Loaded directly from <code>{run.path}</code>. Its artifacts remain on this device and are read-only in the explorer.</p>
          <div className="hero-actions">
            <button className="primary" onClick={() => onNavigate(primaryView)}>{primaryLabel} <span>→</span></button>
            <button className="text-button" onClick={() => onNavigate("provenance")}>Review run files</button>
          </div>
        </div>
        <div className="local-run-identity">
          <span className="hero-stat-label">Selected run</span>
          <strong>{run.name}</strong>
          <p>{run.dateLabel}</p>
          <footer><span>{run.model}</span><Badge tone={statusTone(status.state)}>{statusLabel(status)}</Badge></footer>
        </div>
      </section>

      <FreezeNotice run={run} />

      <div className="notice local-notice" role="note">
        <span className="notice-icon">✓</span>
        <p><strong>Local artifacts loaded.</strong> Summary counts, effect rows, configuration, and the bounded generation preview below come from this run. Missing artifacts are shown as absent rather than inferred.</p>
      </div>

      <section className="metric-strip" aria-label="Selected run summary">
        <div><span>Prompt items</span><strong>{run.promptCount || "—"}</strong><small>report.json</small></div>
        <div><span>Conditions</span><strong>{conditionCount || "—"}</strong><small>declared run arms</small></div>
        {/* The report's whole-run count and the viewer's count of loaded
            preview rows are different numbers; the caption used to say
            "complete local preview" under either one. */}
        {run.generationCount
          ? <div><span>Generations</span><strong>{run.generationCount}</strong><small>report.json{run.previewTruncated ? ` · ${run.generationRows.length} loaded in preview` : ""}</small></div>
          : <div><span>Generations</span><strong>{run.generationRows.length || "—"}</strong><small>{run.generationRows.length ? `counted from the ${run.previewTruncated ? "bounded" : "loaded"} preview — report.json stamped none` : "not stamped"}</small></div>}
        <div><span>Artifacts</span><strong>{run.artifacts.length}</strong><small>files in run directory</small></div>
      </section>

      <ConditionsTable run={run} />

      <TruncationCard run={run} />

      <section className="section-grid main-evidence">
        <div className="card evidence-card">
          <header className="section-header">
            <div><span className="section-number">{hasConceptEvidence ? "CONCEPT VALIDATION" : "PAIRED EFFECTS"}</span><h2>{hasConceptEvidence ? `${run.validationConcepts.length} validation rows · ${run.cosineMatrices.length} cosine matrices` : shownEffects.length ? "Reported estimates" : "No effect-size table found"}</h2></div>
            {hasConceptEvidence && <button className="quiet-link" onClick={() => onNavigate("concepts")}>Open evidence →</button>}
            {shownEffects.length > 0 && <button className="quiet-link" onClick={() => onNavigate("effects")}>Full table →</button>}
          </header>
          {hasConceptEvidence ? <div className="concept-preview-grid">{run.validationConcepts.slice(0, 6).map((row) => <div key={`${row.name}-${row.layer}`}><span>{row.layer == null ? "Layer —" : `Layer ${row.layer}`}</span><strong>{row.name}</strong><b>{row.calibratedAccuracy != null ? `${(row.calibratedAccuracy * 100).toFixed(0)}%` : row.accuracy != null ? `${(row.accuracy * 100).toFixed(0)}%` : "not run"}</b><small>{row.calibratedAccuracy != null ? "calibrated accuracy" : "transfer accuracy"}</small></div>)}</div> : shownEffects.length > 0 ? <>
            <div className="axis-hint"><span>Negative</span><span>No difference</span><span>Positive</span></div>
            <div className="forest">{shownEffects.map((effect) => <ForestRow key={effect.key} effect={effect} compact />)}</div>
            <div className="legend"><span><i className="legend-dot" /> Estimate</span><span><i className="legend-line" /> Reported 95% CI</span><span>● adjusted p &lt; .05</span></div>
          </> : <div className="artifact-empty"><span>∅</span><p>This run has no readable <code>effect-sizes.csv</code>. Generation and provenance views are still available.</p></div>}
        </div>
        <aside className="card local-contents-card">
          <span className="section-number">RUN CONTENTS</span>
          <h2>Available locally</h2>
          <div><i className={run.artifacts.includes("report.json") ? "available" : ""}>✓</i><span><strong>Run report</strong><small>conditions and summary counts</small></span></div>
          <div><i className={run.artifacts.includes("generations.jsonl") ? "available" : ""}>✓</i><span><strong>Generations</strong><small>{run.generationRows.length} readable preview records</small></span></div>
          <div><i className={run.artifacts.includes("effect-sizes.csv") ? "available" : ""}>✓</i><span><strong>Effect sizes</strong><small>{pooledEffects.length} pooled row{pooledEffects.length === 1 ? "" : "s"}{run.effectRows.length > pooledEffects.length ? ` · ${run.effectRows.length - pooledEffects.length} stratified` : ""}</small></span></div>
          <div><i className={run.cosineMatrices.length ? "available" : ""}>✓</i><span><strong>Concept geometry</strong><small>{run.cosineMatrices.length} cosine matrices</small></span></div>
          <div><i className={hasOptimization ? "available" : ""}>✓</i><span><strong>Optimization grid</strong><small>{run.sweepRows.length} layer × strength cells</small></span></div>
          <div><i className={hasPanel ? "available" : ""}>✓</i><span><strong>Panel dynamics</strong><small>{run.panelEffects.length} decomposed endpoints</small></span></div>
          <div><i className={run.artifacts.includes("config.json") ? "available" : ""}>✓</i><span><strong>Configuration</strong><small>model and run provenance</small></span></div>
          <button onClick={() => onNavigate("provenance")}>Open provenance <b>→</b></button>
        </aside>
      </section>

      <section className="section-grid local-lower">
        <div className="card local-path-card"><span className="section-number">READ BOUNDARY</span><h2>Explicit folder permission only</h2><p>The app can see this workspace because you chose it through the browser. It does not retain access after the local session ends and never writes to the run.</p></div>
        <div className="card local-primary-card"><span className="section-number">PRIMARY REPORTED ROW</span>{primary ? <><h2>{primary.short}</h2><strong>{fmt(primary.estimate, primary.unit === "months" ? 1 : 2)} <small>{primary.unit}</small></strong><p>95% CI {fmt(primary.low)} to {fmt(primary.high)} · n = {primary.n} · adjusted p {primary.q == null ? "not reported" : primary.q.toPrecision(2)}</p></> : <><h2>Not available</h2><p>Select Generations or Provenance to inspect the artifacts this run does contain.</p></>}</div>
      </section>
    </div>
  );
}

export function Overview({ onNavigate, run }: { onNavigate: (view: View) => void; run: WorkspaceRun | null }) {
  if (run) return <LocalOverview run={run} onNavigate={onNavigate} />;
  // With no run there is nothing to show. The one exception is the layout
  // preview a developer asks for by name in a browser; its page lives in
  // app/demo/, which the embedded app does not contain.
  if (!demoPreviewEnabled()) return <NoRunSelected title="Study overview" />;
  return <DemoOverview onNavigate={onNavigate} />;
}
