"use client";

// The choice-instrument view (upgrade plan Phase 3): the compare endpoint.
//
// Answer-token readouts ride inside generations.jsonl, so this view loads
// that file itself (bounded read, same 32 MB discipline as the generations
// loader) rather than waiting for a shape in the shared run object. Stored
// values render plain; the baseline join, Δ target log-odds, flips and
// counts carry a `derived` badge; counterbalance pairing and the saturation
// call carry a `heuristic` badge naming the assumption.

import { useEffect, useMemo, useState } from "react";
import { DerivedBadge, HeuristicBadge, ProvenanceLegend } from "../components/provenance";
import { Badge, ExportButton } from "../components/ui";
import { csvFilename, type ExportColumn } from "../lib/export";
import { fmt } from "../lib/format";
import {
  findChoiceDeltaRuns,
  loadEngineChoiceDeltas,
  DEFAULT_SATURATION_THRESHOLD,
  ORDER_CONSISTENCY_ASSUMPTION,
  PAIRING_ASSUMPTION,
  SATURATION_ASSUMPTION,
  buildInstrumentTable,
  loadInstrumentRecords,
  readConditionRollups,
  requestGenerationRecord,
  shortConditionLabels,
} from "../lib/instruments";
import type { EngineChoiceDeltas, InstrumentCell, InstrumentItem, InstrumentLoad } from "../lib/instruments";
import type { RunFile, View, WorkspaceRun } from "../lib/types";
import "./choice.css";

const DELTA_FORMULA = "Δ = this record's target log-odds − the same-item baseline record's target log-odds, joined on promptID";
const FLIP_FORMULA = "flip = this record's `selected` differs from the same-item baseline record's `selected`";

// The per-item table exports LONG (one row per item × condition cell) so the
// column set does not change with the run's condition list, and so each
// column has exactly one provenance kind — which is what the stamp line
// claims.
type ChoiceExportRow = { item: InstrumentItem; cell: InstrumentCell };

const choiceColumns: ExportColumn<ChoiceExportRow>[] = [
  { header: "promptID", kind: "stored", value: ({ item }) => item.promptID },
  { header: "counterbalanceOrder", kind: "heuristic", value: ({ item }) => item.order ?? "" },
  { header: "caseID", kind: "stored", value: ({ item }) => item.caseID },
  { header: "arm", kind: "stored", value: ({ item }) => item.arm },
  { header: "target", kind: "stored", value: ({ cell }) => cell.record.target },
  { header: "condition", kind: "stored", value: ({ cell }) => cell.condition },
  { header: "isBaseline", kind: "stored", value: ({ cell }) => cell.isBaseline },
  { header: "targetLogOdds", kind: "stored", value: ({ cell }) => cell.targetLogOdds },
  { header: "selected", kind: "stored", value: ({ cell }) => cell.selected },
  { header: "margin", kind: "stored", value: ({ cell }) => cell.record.margin },
  { header: "choiceProbabilityOfTarget", kind: "stored", value: ({ cell }) => cell.record.choiceProbability[cell.record.target] ?? null },
  { header: "baselineTargetLogOdds", kind: "stored", value: ({ item }) => item.baselineTargetLogOdds },
  { header: "baselineSelected", kind: "stored", value: ({ item }) => item.baselineSelected },
  { header: "deltaTargetLogOdds", kind: "derived", value: ({ cell }) => cell.delta },
  { header: "flippedVsBaseline", kind: "derived", value: ({ cell }) => cell.flipped },
  { header: "targetDiffersFromBaseline", kind: "derived", value: ({ cell }) => cell.targetMismatch },
  { header: "atCeiling", kind: "heuristic", value: ({ item }) => item.saturated },
  { header: "instrument", kind: "stored", value: ({ cell }) => cell.record.instrument },
  { header: "generations.jsonl line", kind: "stored", value: ({ cell }) => cell.record.line },
];

const logOdds = (value: number | null) => value == null ? "—" : `${value > 0 ? "+" : ""}${value.toFixed(2)}`;
const percent = (value: number | null) => value == null ? "—" : `${(value * 100).toFixed(1)}%`;

export function ChoiceInstrumentView(props: {
  run: WorkspaceRun | null;
  workspaceRuns: WorkspaceRun[];
  onActivateRun: (run: WorkspaceRun) => void;
  onNavigate: (view: View) => void;
  onOpenFile: (file: RunFile) => void;
}) {
  const { run, workspaceRuns, onActivateRun, onNavigate, onOpenFile } = props;
  const [load, setLoad] = useState<InstrumentLoad | null>(null);
  const [engineDeltas, setEngineDeltas] = useState<EngineChoiceDeltas | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [threshold, setThreshold] = useState(DEFAULT_SATURATION_THRESHOLD);

  useEffect(() => {
    let cancelled = false;
    setLoad(null);
    setError("");
    setThreshold(DEFAULT_SATURATION_THRESHOLD);
    if (!run) { setLoading(false); return () => { cancelled = true; }; }
    setLoading(true);
    loadInstrumentRecords(run)
      .then((result) => { if (!cancelled) { setLoad(result); setLoading(false); } })
      .catch((cause: unknown) => { if (!cancelled) { setError(cause instanceof Error ? cause.message : String(cause)); setLoading(false); } });
    // The engine's own per-item deltas, when an analyze run consumed this
    // run: newest artifact wins; absence simply leaves the derived numbers
    // (already badged) as the only reading.
    setEngineDeltas(null);
    const candidates = findChoiceDeltaRuns(run.name, workspaceRuns);
    if (candidates.length) {
      const full = workspaceRuns.find((entry) => entry.name === candidates[0].name);
      if (full) {
        void loadEngineChoiceDeltas(full).then((result) => { if (!cancelled) setEngineDeltas(result); });
      }
    }
    return () => { cancelled = true; };
  }, [run?.key]);

  const table = useMemo(() => buildInstrumentTable(load?.records ?? [], { saturationThreshold: threshold }), [load, threshold]);
  const rollups = useMemo(() => readConditionRollups(run?.report ?? {}), [run?.report]);
  const labels = useMemo(() => shortConditionLabels(table.conditions, table.baselineCondition), [table]);

  const header = (eyebrow: string, note: string) => (
    <header className="page-title">
      <div><span className="section-number">{eyebrow}</span><h1>Choice instrument</h1><p>{note}</p></div>
      {load?.file && <div className="title-actions"><button className="secondary" onClick={() => load.file && onOpenFile(load.file)}>generations.jsonl</button></div>}
    </header>
  );

  if (!run) return <div className="view-enter inner-view choice-view">{header("NO RUN SELECTED", "Choose a run from the sidebar. Answer-token readouts are read from that run's generations.jsonl; nothing is inferred or substituted.")}<div className="card no-run-card"><span>⌂</span><h2>No local run selected</h2><p>Pick a run to load its instrument records read-only.</p></div></div>;
  if (loading) return <div className="view-enter inner-view choice-view">{header("READING", `Reading ${run.name}/generations.jsonl.`)}<div className="card no-run-card"><span>…</span><h2>Reading generations.jsonl</h2><p>Answer-token records ride inside the generations file.</p></div></div>;
  if (error) return <div className="view-enter inner-view choice-view">{header("READ FAILED", `generations.jsonl could not be read from ${run.name}.`)}<div className="card no-run-card"><span>!</span><h2>generations.jsonl could not be read</h2><p>{error}</p></div></div>;
  if (!load?.present) return <div className="view-enter inner-view choice-view">{header("NO GENERATIONS FILE", `${run.name} carries no generations.jsonl.`)}<div className="card no-run-card"><span>∅</span><h2>No generations.jsonl in this run</h2><p>Answer-token instrument records are written inside generations.jsonl. This run directory has no such file, so there is nothing to read.</p></div></div>;
  if (!load.records.length) return <div className="view-enter inner-view choice-view">{header(`${load.sampledRecords} SAMPLED RECORDS · 0 READOUTS`, `Read from ${run.name}/generations.jsonl.`)}<div className="card no-run-card"><span>∅</span><h2>This run carries no answer-token instrument readouts</h2><p>generations.jsonl holds {load.sampledRecords} sampled-text record{load.sampledRecords === 1 ? "" : "s"} and no record carrying an instrument marker{load.skipped ? `, plus ${load.skipped} line${load.skipped === 1 ? "" : "s"} that did not parse` : ""}.</p></div></div>;

  const nonBaseline = table.conditions.filter((condition) => condition !== table.baselineCondition);
  const unsaturated = table.items.filter((item) => !item.saturated);
  const deltas = nonBaseline.map((condition) => ({
    condition,
    points: unsaturated.flatMap((item) => {
      const cell = item.cellByCondition[condition];
      return cell && cell.delta != null ? [{ promptID: item.promptID, delta: cell.delta, flipped: cell.flipped }] : [];
    }),
  }));
  // The table as shown, flattened: item order, condition order, every cell
  // the run actually wrote (a missing cell is absent from the export just as
  // it renders "—" on screen).
  const exportRows: ChoiceExportRow[] = table.items.flatMap((item) => item.cells.map((cell) => ({ item, cell })));
  const bound = Math.max(0.5, ...deltas.flatMap((row) => row.points.map((point) => Math.abs(point.delta)))) * 1.2;
  const position = (value: number) => Math.max(1.5, Math.min(98.5, ((value + bound) / (2 * bound)) * 100));

  return (
    <div className="view-enter inner-view choice-view">
      {header(
        `${table.instrumentName.toUpperCase() || "INSTRUMENT"} · ${table.items.length} ITEMS × ${table.conditions.length} CONDITIONS`,
        `${load.records.length} answer-token readouts read from ${run.name}/generations.jsonl${load.sampledRecords ? `, beside ${load.sampledRecords} sampled-text record${load.sampledRecords === 1 ? "" : "s"}` : ""}. Deterministic scoring: no sampling, no judge.`,
      )}
      <ProvenanceLegend />
      {(load.truncated || load.skipped > 0 || table.instrumentNames.length > 1 || !table.baselineCondition) && (
        <p className="choice-note">
          {load.truncated && <span>Bounded read: only the first 32 MB of generations.jsonl was parsed.</span>}
          {load.skipped > 0 && <span>{load.skipped} instrument line{load.skipped === 1 ? "" : "s"} did not parse and {load.skipped === 1 ? "is" : "are"} excluded.</span>}
          {table.instrumentNames.length > 1 && <span>This run mixes instruments ({table.instrumentNames.join(", ")}); showing {table.instrumentName}.</span>}
          {!table.baselineCondition && <span>No condition is named <b>baseline</b>, so no Δ, flip, or ceiling call is made — stored log-odds only.</span>}
        </p>
      )}

      {rollups.length > 0 && (
        <div className="choice-rollups">
          {rollups.map((rollup) => (
            <div className={`choice-rollup ${rollup.condition === table.baselineCondition ? "is-baseline" : ""}`} key={rollup.condition}>
              <span title={rollup.condition}>{labels[rollup.condition] ?? rollup.condition}</span>
              <strong>{percent(rollup.choiceRate)}<small>choice rate</small></strong>
              <dl>
                <dt>Choice readouts</dt><dd>{rollup.choiceReadouts ?? "—"}</dd>
                <dt>Generations</dt><dd>{rollup.generations ?? "—"}</dd>
                <dt>Agreement with baseline</dt><dd>{rollup.agreementWithBaseline ? `${percent(rollup.agreementWithBaseline.agreement)} · n ${rollup.agreementWithBaseline.n ?? "—"}` : "—"}</dd>
              </dl>
            </div>
          ))}
        </div>
      )}
      {engineDeltas && (
        <section className="card engine-deltas-card">
          <header className="section-header">
            <div>
              <span className="section-number">ENGINE ANALYSIS · STORED</span>
              <h2>Per-condition Δ target log-odds, computed by analyze</h2>
            </div>
            {(() => {
              const analyzeRun = workspaceRuns.find((entry) => entry.name === engineDeltas.analyzeRun);
              return analyzeRun
                ? <button className="quiet-link" onClick={() => onActivateRun(analyzeRun)}>Open {engineDeltas.analyzeRun} →</button>
                : <span className="muted">{engineDeltas.analyzeRun}</span>;
            })()}
          </header>
          <div className="engine-deltas-grid">
            {engineDeltas.summaries.map((summary) => (
              <div key={summary.condition}>
                <span>{labels[summary.condition] ?? summary.condition}</span>
                <strong>{logOdds(summary.mean)}</strong>
                <small>
                  {summary.ciLower != null && summary.ciUpper != null
                    ? `95% CI ${logOdds(summary.ciLower)} to ${logOdds(summary.ciUpper)}`
                    : "CI not stamped"}
                  {summary.n != null ? ` · n = ${summary.n}` : ""}
                  {summary.flipped != null ? ` · flips ${summary.flipped}` : ""}
                </small>
              </div>
            ))}
          </div>
          <footer className="engine-deltas-note">
            choice-deltas.json · paired bootstrap
            {engineDeltas.summaries[0]?.replicates != null ? ` (${engineDeltas.summaries[0].replicates} resamples, seed ${engineDeltas.summaries[0]?.seed ?? "—"})` : ""}
            {(engineDeltas.skippedNoBaseline ?? 0) > 0 ? ` · ${engineDeltas.skippedNoBaseline} readout(s) skipped without a baseline partner` : ""}
            {(engineDeltas.skippedNoTargetValue ?? 0) > 0 ? ` · ${engineDeltas.skippedNoTargetValue} skipped without a target value` : ""}
            {" — engine artifact; the per-item Δ below remains the viewer's derivation."}
          </footer>
        </section>
      )}

      <section className="card choice-table-card">
        <header className="section-header">
          <div>
            <span className="section-number">PER-ITEM READOUT · TARGET LOG-ODDS</span>
            <h2>Every scoped item, every condition</h2>
          </div>
          <div className="choice-controls">
            <label>Ceiling threshold |log-odds| ≥
              <input type="number" min={0} step={0.5} value={threshold} onChange={(event) => setThreshold(Number.isFinite(event.target.valueAsNumber) ? event.target.valueAsNumber : DEFAULT_SATURATION_THRESHOLD)} aria-label="Saturation threshold in log-odds" />
            </label>
            <HeuristicBadge assumption={SATURATION_ASSUMPTION} />
            <ExportButton filename={csvFilename(run.name, "choice-items")} columns={choiceColumns} rows={exportRows} />
          </div>
        </header>
        <p className="choice-note">
          <span><b>{table.saturatedCount} of {table.items.length}</b> items are at ceiling: |baseline target log-odds| ≥ {threshold} — little headroom for an intervention to move them. Those rows are muted; they are still real readings.</span>
          <HeuristicBadge assumption={SATURATION_ASSUMPTION} />
        </p>
        <div className="choice-table-scroll">
          <table className="choice-table">
            <thead>
              <tr>
                <th className="choice-item-head">Item</th>
                {table.conditions.map((condition) => (
                  <th key={condition}>
                    <span className="choice-condition-name" title={condition}>{labels[condition] ?? condition}</span>
                    {condition === table.baselineCondition
                      ? <small>target log-odds</small>
                      : <small>log-odds · Δ <DerivedBadge formula={DELTA_FORMULA} /></small>}
                  </th>
                ))}
              </tr>
            </thead>
            {table.groups.map((group) => (
              <tbody className="choice-group" key={group.key}>
                <tr className="choice-group-row">
                  <th colSpan={table.conditions.length + 1} scope="colgroup">
                    <span>{group.stem}</span>
                    {group.pair && <HeuristicBadge assumption={PAIRING_ASSUMPTION} />}
                    {group.pair && group.pair.possibleOrderArtifact.length > 0 && <Badge tone="warn">possible order artifact</Badge>}
                    {!group.pair && group.items.some((item) => item.order) && <Badge tone="neutral">no mirrored order</Badge>}
                  </th>
                </tr>
                {group.items.map((item) => (
                  <tr className={`choice-row ${item.saturated ? "is-saturated" : item.baselineTargetLogOdds != null ? "is-boundary" : ""}`} key={item.promptID}>
                    <th className="choice-item-cell" scope="row">
                      <strong>{item.order && <i className="choice-order">{item.order}</i>}{item.promptID}</strong>
                      <span>{[item.arm, item.caseID && `case ${item.caseID}`, item.target && `target ${item.target}`, item.saturated ? "at ceiling" : "has headroom"].filter(Boolean).join(" · ")}</span>
                    </th>
                    {table.conditions.map((condition) => {
                      const cell = item.cellByCondition[condition];
                      if (!cell) return <td className="choice-cell choice-missing" key={condition}>—</td>;
                      return (
                        <td className={`choice-cell ${cell.flipped ? "is-flip" : ""}`} key={condition} title={`${condition} · selected ${cell.selected} · target ${cell.record.target} · margin ${cell.record.margin ?? "—"}`}>
                          <strong>{logOdds(cell.targetLogOdds)}</strong>
                          {cell.isBaseline
                            ? <span className="choice-selected">selected {cell.selected}</span>
                            : <span className={`choice-delta ${cell.delta === 0 ? "is-flat" : ""}`}>{cell.targetMismatch ? "target differs" : cell.delta == null ? "Δ —" : `Δ ${fmt(cell.delta, 2)}`}</span>}
                          {cell.flipped && <em className="choice-flip-mark">{cell.baselineSelected} → {cell.selected}</em>}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            ))}
          </table>
        </div>
      </section>

      {nonBaseline.length > 0 && (
        <section className="card choice-section">
          <header className="section-header">
            <div>
              <span className="section-number">Δ TARGET LOG-ODDS · ITEMS WITH HEADROOM</span>
              <h2>Movement where movement was possible</h2>
              <p>One dot per item with headroom, per condition, on a shared scale. Ceiling items are excluded — an unmoved item at |log-odds| {threshold}+ measures the ceiling, not the intervention.</p>
            </div>
            <div className="choice-controls"><DerivedBadge formula={DELTA_FORMULA} /><HeuristicBadge assumption={SATURATION_ASSUMPTION} /></div>
          </header>
          <div className="choice-strips">
            {deltas.map((row) => (
              <div className="choice-strip" key={row.condition}>
                <div className="choice-strip-label"><strong title={row.condition}>{labels[row.condition] ?? row.condition}</strong><span>{row.points.length} of {unsaturated.length} item{unsaturated.length === 1 ? "" : "s"} with headroom</span></div>
                <div className="choice-strip-track">
                  <span className="choice-strip-zero" style={{ left: `${position(0)}%` }} />
                  {row.points.map((point) => (
                    <span className={`choice-dot ${point.flipped ? "is-flip" : ""}`} key={point.promptID} style={{ left: `${position(point.delta)}%` }} title={`${point.promptID}: Δ ${fmt(point.delta, 2)}${point.flipped ? " · selection flipped" : ""}`} />
                  ))}
                </div>
                <div className="choice-strip-value">{row.points.length ? `${fmt(Math.min(...row.points.map((point) => point.delta)), 1)} … ${fmt(Math.max(...row.points.map((point) => point.delta)), 1)}` : "no Δ"}</div>
              </div>
            ))}
          </div>
          <div className="choice-scale"><span>{fmt(-bound, 1)}</span><span>0</span><span>{fmt(bound, 1)}</span></div>
          {unsaturated.length === 0 && <div className="empty-state">Every item is at ceiling under the current threshold — raise it to see movement.</div>}
        </section>
      )}

      {(table.pairs.length > 0 || table.unpairedOrderedItems.length > 0) && (
        <section className="card choice-section">
          <header className="section-header">
            <div>
              <span className="section-number">COUNTERBALANCE</span>
              <h2>Do the two orders agree?</h2>
              <p>Consistent orders select <b>opposite letters</b>: the pair swaps the options, so the same substantive choice reads as A in one order and B in the other. Two orders selecting the same letter disagree substantively.</p>
            </div>
            <div className="choice-controls"><HeuristicBadge assumption={`${PAIRING_ASSUMPTION}; ${ORDER_CONSISTENCY_ASSUMPTION}`} /></div>
          </header>
          <div className="choice-pairs">
            {table.pairs.map((pair) => (
              <article className={`choice-pair ${pair.possibleOrderArtifact.length ? "is-artifact" : ""}`} key={pair.stem}>
                <header>
                  <strong>{pair.stem}</strong>
                  {pair.possibleOrderArtifact.length ? <Badge tone="warn">possible order artifact</Badge> : <Badge tone="good">orders agree</Badge>}
                </header>
                <table>
                  <thead><tr><th>Condition</th><th>-ab</th><th>-ba</th><th>Reads as</th></tr></thead>
                  <tbody>
                    {pair.consistency.map((entry) => (
                      <tr key={entry.condition}>
                        <td title={entry.condition}>{labels[entry.condition] ?? entry.condition}</td>
                        <td>{entry.abSelected || "—"}</td>
                        <td>{entry.baSelected || "—"}</td>
                        <td className={entry.comparable && !entry.consistent ? "is-inconsistent" : ""}>{!entry.comparable ? "—" : entry.consistent ? "same option" : "order artifact?"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </article>
            ))}
          </div>
          {table.unpairedOrderedItems.length > 0 && (
            <p className="choice-note"><span>{table.unpairedOrderedItems.length} item{table.unpairedOrderedItems.length === 1 ? " carries" : "s carry"} an order suffix with no mirrored order in this run and {table.unpairedOrderedItems.length === 1 ? "is" : "are"} paired with nothing: {table.unpairedOrderedItems.map((item) => item.promptID).join(", ")}.</span></p>
          )}
        </section>
      )}

      <section className="card choice-section">
        <header className="section-header">
          <div>
            <span className="section-number">SELECTION FLIPS</span>
            <h2>Where the chosen option changed</h2>
            <p>A flip is the coarsest possible endpoint: the intervention moved the selected option, not merely its log-odds.</p>
          </div>
          <div className="choice-controls"><DerivedBadge formula={FLIP_FORMULA} /></div>
        </header>
        {table.flips.length === 0
          ? <div className="empty-state">No condition selected a different option than its same-item baseline record{table.baselineCondition ? "" : " — no baseline condition to compare against"}.</div>
          : <div className="choice-flips">
              {table.flips.map((flip) => (
                <div className="choice-flip-row" key={`${flip.promptID}-${flip.condition}`}>
                  <div>
                    <strong>{flip.promptID}</strong>
                    <p>
                      <span title={flip.condition}>{labels[flip.condition] ?? flip.condition}</span> selected <b>{flip.to}</b> where baseline selected <b>{flip.from}</b>
                      {" · "}target log-odds {logOdds(flip.baselineTargetLogOdds)} → {logOdds(flip.targetLogOdds)}
                      {flip.delta != null && ` (Δ ${fmt(flip.delta, 2)})`}
                      {flip.saturated && " · this item is at ceiling"}
                    </p>
                  </div>
                  <button onClick={() => { requestGenerationRecord({ promptID: flip.promptID, condition: flip.condition }); onNavigate("generations"); }}>Open record →</button>
                </div>
              ))}
            </div>}
      </section>
    </div>
  );
}
