"use client";

import { useMemo, useState } from "react";
import { DerivedBadge, ProvenanceLegend } from "../components/provenance";
import { Badge } from "../components/ui";
import { findFile } from "../lib/discovery";
import { fmt, metricLabel, shortHash, sweepMetricValue } from "../lib/format";
import type { RunFile, SweepRow, WorkspaceRun } from "../lib/types";

// The gate verdicts on this page ("Passes gates", "Eligible cells N / M")
// are the VIEWER applying the manifest's declared constraints to the stored
// grid — the engine stamps only the winning cell. They carry a derived badge
// like every other viewer-computed reading (wave-1 convention; this view
// predates it).
const GATE_FORMULA = "the run's declared coherence floor and capability tolerance applied to each stored grid cell; the engine stamps only the winning cell";

/// A metric a cell did not record must read as absent. `(value ?? 0)` here
/// printed "0.000" — an unrecorded objective became the worst possible one.
const metricText = (value: number | null | undefined, digits = 3) => value == null ? "—" : value.toFixed(digits);

export function OptimizationView({ run, onOpenFile }: { run: WorkspaceRun | null; onOpenFile: (file: RunFile) => void }) {
  const concepts = useMemo(() => run ? [...new Set(run.sweepRows.map((row) => row.concept))] : [], [run]);
  // Concept + displayed metric are stored WITH the run they were chosen in.
  // Selecting another run therefore falls back to that run's first concept
  // and ITS declared objective without an effect — and without the one render
  // in which the grid was filtered by the previous run's concept (or, on the
  // first render of all, by the empty-string concept, which matched no rows).
  const [choice, setChoice] = useState<{ runKey: string; concept: string; metric: string } | null>(null);
  const declaredMetricFor = (name: string) => {
    const declared = run?.sweepRecommendations.find((item) => item.concept === name)?.metric;
    return declared && declared !== "markerDensity" ? "objective" : "markerDensity";
  };
  const active = choice?.runKey === (run?.key ?? "") ? choice : null;
  const concept = active?.concept ?? concepts[0] ?? "";
  const metric = active?.metric ?? declaredMetricFor(concepts[0] ?? "");
  if (!run) return <div className="view-enter inner-view"><header className="page-title"><div><span className="section-number">OPTIMIZATION STUDIES</span><h1>Optimization & sweeps</h1><p>Select a local sweep run to inspect its full layer × intervention-strength surface.</p></div></header><div className="card no-run-card"><span>⌂</span><h2>No local run selected</h2><p>Choose a workspace and sweep run from the sidebar.</p></div></div>;
  const rows = run.sweepRows.filter((row) => row.concept === concept);
  const recommendation = run.sweepRecommendations.find((item) => item.concept === concept) ?? null;
  const objectiveMetric = recommendation?.metric ?? (rows.some((row) => row.objective != null) ? "objective" : "markerDensity");
  const baseline = rows.find((row) => row.layer < 0) ?? null;
  const cells = rows.filter((row) => row.layer >= 0);
  const layers = [...new Set(cells.map((row) => row.layer))].sort((a, b) => a - b);
  const alphas = [...new Set(cells.map((row) => row.alpha))].sort((a, b) => a - b);
  const values = cells.flatMap((row) => { const value = sweepMetricValue(row, metric); return value == null ? [] : [value]; });
  const minimum = values.length ? Math.min(...values) : 0;
  const maximum = values.length ? Math.max(...values) : 1;
  const tolerance = recommendation?.capabilityTolerance ?? .15;
  const coherenceFloor = recommendation?.coherenceFloor ?? .45;
  const gateState = (row: SweepRow): "pass" | "fail" | "unknown" => {
    if (row.distinct2 < coherenceFloor) return "fail";
    if (baseline?.batteryAccuracy == null || row.batteryAccuracy == null) return "unknown";
    return row.batteryAccuracy >= baseline.batteryAccuracy - tolerance ? "pass" : "fail";
  };
  const eligibleCount = cells.filter((row) => gateState(row) === "pass").length;
  const winner = cells.find((row) => recommendation?.layer === row.layer && recommendation?.alpha === row.alpha) ?? null;
  const sweepFile = findFile(run.files, "sweep.csv");
  const recommendationFile = findFile(run.files, "recommendations.json");
  const availableMetrics = [objectiveMetric === "markerDensity" ? "markerDensity" : "objective", "markerDensity", "distinct2", ...(rows.some((row) => row.batteryAccuracy != null) ? ["batteryAccuracy"] : [])].filter((item, index, all) => all.indexOf(item) === index);
  return <div className="view-enter inner-view optimization-view">
    <header className="page-title"><div><span className="section-number">{cells.length} CELLS · {layers.length} LAYERS · {alphas.length} STRENGTHS</span><h1>Optimization & sweeps</h1><p>Selection lives on a development split. Read the whole surface: expression should rise before coherence or capability collapses.</p></div><div className="title-actions">{recommendationFile && <button className="secondary" onClick={() => onOpenFile(recommendationFile)}>Recommendation JSON</button>}{sweepFile && <button className="primary" onClick={() => onOpenFile(sweepFile)}>Open full grid <span>→</span></button>}</div></header>
    <ProvenanceLegend />
    {!rows.length ? <div className="card no-run-card"><span>∅</span><h2>No readable sweep grid</h2><p>This run does not contain a schema-compatible sweep.csv.</p></div> : <>
      <section className="sweep-controls" aria-label="Sweep filters"><label><span>Concept</span><select value={concept} onChange={(event) => setChoice({ runKey: run.key, concept: event.target.value, metric: declaredMetricFor(event.target.value) })}>{concepts.map((name) => <option key={name}>{name}</option>)}</select></label><label><span>Displayed metric</span><select value={metric} onChange={(event) => setChoice({ runKey: run.key, concept, metric: event.target.value })}>{availableMetrics.map((name) => <option value={name} key={name}>{name === "objective" ? `${metricLabel(objectiveMetric)} (declared objective)` : metricLabel(name)}</option>)}</select></label><div><span>Selection criterion</span><strong>{metricLabel(objectiveMetric)}</strong></div><div><span>Eligible cells <DerivedBadge formula={GATE_FORMULA} /></span><strong>{eligibleCount} / {cells.length}</strong></div></section>
      <section className="section-grid sweep-layout">
        <div className="card sweep-grid-card">
          <header className="section-header"><div><span className="section-number">LAYER × α SURFACE</span><h2>{metric === "objective" ? metricLabel(objectiveMetric) : metricLabel(metric)}</h2></div><div className="heat-legend"><span>Low</span><i /><span>High</span></div></header>
          <div className="sweep-grid-scroll"><div className="sweep-grid" style={{ gridTemplateColumns: `70px repeat(${alphas.length}, minmax(92px, 1fr))` }}><div className="sweep-corner">Layer / α</div>{alphas.map((alpha) => <div className="sweep-alpha" key={alpha}>{fmt(alpha, 2)}σ</div>)}{layers.flatMap((layer) => [<div className="sweep-layer" key={`layer-${layer}`}>L{layer}</div>, ...alphas.map((alpha) => {
            const row = cells.find((candidate) => candidate.layer === layer && candidate.alpha === alpha);
            if (!row) return <div className="sweep-cell sweep-missing" key={`${layer}-${alpha}`}>—</div>;
            const value = sweepMetricValue(row, metric);
            const intensity = value == null || maximum === minimum ? .45 : .12 + .76 * ((value - minimum) / (maximum - minimum));
            const isWinner = recommendation?.layer === layer && recommendation?.alpha === alpha;
            const state = gateState(row);
            return <div className={`sweep-cell ${isWinner ? "winner" : ""} ${state === "pass" ? "eligible" : state === "fail" ? "failed" : "unknown"}`} key={`${layer}-${alpha}`} style={{ backgroundColor: `rgba(31, 88, 111, ${intensity})` }} title={`Layer ${layer}, alpha ${alpha}: ${value == null ? "not recorded" : value}`}><strong>{value == null ? "—" : value.toFixed(3)}</strong><span>{isWinner ? "Selected" : state === "pass" ? "Passes gates" : state === "fail" ? "Fails gate" : "Gate unknown"}</span></div>;
          })])}</div></div>
          <footer className="sweep-grid-note"><span><i className="winner-key" />Stamped winner</span><span><i className="pass-key" />Eligible <DerivedBadge formula={GATE_FORMULA} /></span><span><i className="unknown-key" />Capability not recorded</span><span><i className="fail-key" />Fails a declared constraint</span></footer>
        </div>
        <aside className="card recommendation-card">
          <span className="section-number">STAMPED SELECTION</span>
          {recommendation?.failure ? <><h2>No recommendation</h2><Badge tone="warn">Selection failed</Badge><p>{recommendation.failure}</p></> : winner ? <><h2>{concept} · L{winner.layer} · {fmt(winner.alpha)}σ</h2><Badge tone="good">Selected on dev data</Badge><div className="winner-readout"><strong>{metricText(sweepMetricValue(winner, objectiveMetric === "markerDensity" ? "markerDensity" : "objective"))}</strong><span>{metricLabel(objectiveMetric)}</span></div><dl><div><dt>Coherence</dt><dd>{winner.distinct2.toFixed(3)}</dd></div><div><dt>Capability</dt><dd>{winner.batteryAccuracy == null ? "Not measured" : `${(winner.batteryAccuracy * 100).toFixed(1)}%`}</dd></div><div><dt>Dev prompts</dt><dd title={recommendation?.devPromptsHash}>{shortHash(recommendation?.devPromptsHash ?? "")}</dd></div><div><dt>Sweep run</dt><dd>{recommendation?.sweepRun || run.name}</dd></div></dl></> : <><h2>Recommendation absent</h2><Badge tone="warn">Not stamped</Badge><p>The grid is readable, but no winning cell was found in recommendations.json. The explorer does not infer one from the maximum.</p></>}
        </aside>
      </section>
      <section className="section-grid sweep-diagnostics">
        <div className="card constraint-card"><header className="section-header"><div><span className="section-number">SELECTION GATES</span><h2>Capability and coherence</h2></div></header><div className="constraint-row"><div><strong>Distinct-2 floor</strong><span>Absolute coherence threshold</span></div><b>≥ {coherenceFloor.toFixed(2)}</b></div><div className="constraint-row"><div><strong>Capability tolerance</strong><span>Relative to the matching baseline</span></div><b>≤ −{(tolerance * 100).toFixed(0)} pp</b></div><div className="constraint-row"><div><strong>Matched-norm random margin</strong><span>Control must not explain the objective</span></div><b>{recommendation?.matchedNormRandomMargin == null ? "Not declared" : recommendation.matchedNormRandomMargin.toFixed(3)}</b></div>{baseline && <div className="baseline-strip"><span>Baseline</span><strong>{metricLabel(metric === "objective" ? objectiveMetric : metric)} {metricText(sweepMetricValue(baseline, metric))}</strong><strong>Distinct-2 {baseline.distinct2.toFixed(3)}</strong><strong>Battery {baseline.batteryAccuracy == null ? "—" : `${(baseline.batteryAccuracy * 100).toFixed(1)}%`}</strong></div>}</div>
        <div className="card sweep-audit-card"><span className="section-number">SCIENTIFIC READ</span><h2>A winner is not a dose-response.</h2><p>The highlighted cell is the result of the declared rule, not proof that it is a stable setting. Check neighboring layers and strengths for a smooth response and keep the chosen α comfortably before the coherence cliff.</p><div><Badge tone={cells.some((row) => row.batteryAccuracy != null) ? "good" : "warn"}>{cells.some((row) => row.batteryAccuracy != null) ? "Battery recorded" : "Battery absent"}</Badge><Badge tone={recommendation?.devPromptsHash ? "good" : "warn"}>{recommendation?.devPromptsHash ? "Dev split stamped" : "Dev hash absent"}</Badge></div></div>
      </section>
    </>}
  </div>;
}
