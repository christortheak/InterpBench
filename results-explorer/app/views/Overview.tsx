"use client";

import { Badge, ForestRow, NoRunSelected } from "../components/ui";
import { demoPreviewEnabled, effects } from "../lib/demo";
import { runKindOf, runStatusOf } from "../lib/discovery";
import { fmt } from "../lib/format";
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
};

function ConditionsTable({ run }: { run: WorkspaceRun }) {
  const conditions = record(run.report.conditions);
  const names = Object.keys(conditions).sort();
  if (!names.length) return null;
  const rows: ConditionRow[] = names.map((name) => {
    const row = record(conditions[name]);
    const agreement = record(row.agreementWithBaseline);
    const battery = record(row.capabilityBattery);
    return {
      name,
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
  const present = (key: keyof ConditionRow) => rows.some((row) => row[key] !== null);
  const allColumns: Array<{ key: keyof ConditionRow; label: string; render: (row: ConditionRow) => string }> = [
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
      </header>
      <div className="raw-table-scroll">
        <table className="raw-table">
          <thead><tr><th>Condition</th>{columns.map((column) => <th key={String(column.key)}>{column.label}</th>)}</tr></thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.name}>
                <td>{row.name}</td>
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
  if (!demoPreviewEnabled()) return <NoRunSelected title="Study overview" />;
  return (
    <div className="view-enter">
      <section className="hero-grid">
        <div className="hero-copy">
          <div className="kicker"><span>Confirmatory readout</span><span>·</span><span>Synthetic preview</span></div>
          <h1>Anger shifts decisions toward <em>harsher, more formalistic</em> outcomes.</h1>
          <p className="dek">Across matched judicial prompts, activation steering increased sentence severity and rule adherence without measurable capability loss. The human-anchored residual remains provisional.</p>
          <div className="hero-actions">
            <button className="primary" onClick={() => onNavigate("effects")}>Inspect the evidence <span>→</span></button>
            <button className="text-button" onClick={() => onNavigate("generations")}>Read all 384 generations</button>
          </div>
        </div>
        <div className="hero-stat" aria-label="Primary effect estimate">
          <span className="hero-stat-label">Paired mean shift</span>
          <div><strong>+7.8</strong><span>months</span></div>
          <p>95% bootstrap CI <b>+3.1 to +12.4</b></p>
          <div className="mini-scale"><i /><b /><em /></div>
          <footer><span>Holm-adjusted p</span><strong>0.012</strong></footer>
        </div>
      </section>

      <div className="notice" role="note">
        <span className="notice-icon">i</span>
        <p><strong>Demonstration data.</strong> Values are realistic but synthetic, designed to show how a completed frozen run will read. Nothing on this page is a research finding.</p>
      </div>

      <section className="metric-strip" aria-label="Study summary">
        <div><span>Paired items</span><strong>64</strong><small>5 samples / condition</small></div>
        <div><span>Conditions</span><strong>6</strong><small>baseline + interventions</small></div>
        <div><span>Parse success</span><strong>98.7%</strong><small>379 / 384 outputs</small></div>
        <div><span>Capability retained</span><strong>99.1%</strong><small>−0.4 pp vs baseline</small></div>
      </section>

      <section className="section-grid main-evidence">
        <div className="card evidence-card">
          <header className="section-header">
            <div><span className="section-number">01 / EFFECTS</span><h2>What moved?</h2></div>
            <button className="quiet-link" onClick={() => onNavigate("effects")}>Full analysis →</button>
          </header>
          <div className="axis-hint"><span>Favors less / lower</span><span>No difference</span><span>Favors more / higher</span></div>
          <div className="forest">
            {effects.map((effect) => <ForestRow key={effect.key} effect={effect} compact />)}
          </div>
          <div className="legend"><span><i className="legend-dot" /> Estimate</span><span><i className="legend-line" /> 95% paired bootstrap CI</span><span>● Holm-adjusted p &lt; .05</span></div>
        </div>

        <aside className="card claim-card">
          <span className="section-number">CLAIM STATUS</span>
          <h2>Evidence is coherent, not yet citable.</h2>
          <div className="claim-step active">
            <i>1</i><div><strong>Model-internal effect</strong><span>Supported in preview</span></div><Badge tone="good">4 / 5 gates</Badge>
          </div>
          <div className="claim-step">
            <i>2</i><div><strong>Human-anchored residual</strong><span>Baseline transcription pending</span></div><Badge tone="warn">Provisional</Badge>
          </div>
          <div className="claim-step">
            <i>3</i><div><strong>Panel propagation</strong><span>Not included in this run</span></div><Badge>Not run</Badge>
          </div>
          <button className="claim-foot" onClick={() => onNavigate("provenance")}><span>Why this is not publication-ready</span><b>→</b></button>
        </aside>
      </section>

      <section className="section-grid lower-grid">
        <div className="card dose-card">
          <header className="section-header"><div><span className="section-number">02 / DOSE RESPONSE</span><h2>The effect rises with intervention strength.</h2></div><Badge tone="good">ρ = .98</Badge></header>
          <div className="dose-chart" aria-label="Sentence severity by steering dose">
            <div className="y-labels"><span>+12 mo</span><span>+6 mo</span><span>0</span><span>−6 mo</span></div>
            <div className="dose-plot">
              <span className="gridline g1"/><span className="gridline g2"/><span className="gridline g3"/><span className="gridline g4"/>
              <span className="dose-segment s1"/><span className="dose-segment s2"/><span className="dose-segment s3"/><span className="dose-segment s4"/>
              <i className="dose-dot d1"/><i className="dose-dot d2"/><i className="dose-dot d3"/><i className="dose-dot d4"/><i className="dose-dot d5"/>
              <div className="x-labels"><span>−1.0σ</span><span>−0.5σ</span><span>0</span><span>+0.5σ</span><span>+1.0σ</span></div>
            </div>
          </div>
          <p className="annotation"><span>↗</span> Monotone across the preregistered grid; negative-dose reversal is visible and the random-vector control stays near zero.</p>
        </div>
        <div className="card controls-card">
          <header className="section-header"><div><span className="section-number">03 / VALIDITY CHECKS</span><h2>Alternative explanations</h2></div></header>
          <div className="control-row"><span className="status-check">✓</span><div><strong>Random direction floor</strong><span>CI crosses zero on all primary endpoints</span></div><Badge tone="good">Pass</Badge></div>
          <div className="control-row"><span className="status-check">✓</span><div><strong>Negative-dose reversal</strong><span>Expected sign on 3 / 3 primary endpoints</span></div><Badge tone="good">Pass</Badge></div>
          <div className="control-row"><span className="status-check">✓</span><div><strong>Capability battery</strong><span>Change −0.4 pp; threshold −3.0 pp</span></div><Badge tone="good">Pass</Badge></div>
          <div className="control-row"><span className="status-warn">!</span><div><strong>Human baseline pin</strong><span>Placeholder source; transcription unverified</span></div><Badge tone="warn">Open</Badge></div>
        </div>
      </section>
    </div>
  );
}
