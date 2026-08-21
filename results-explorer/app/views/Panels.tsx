"use client";

import { useEffect, useMemo, useState } from "react";
import { Badge, ExportButton } from "../components/ui";
import { DerivedBadge, HeuristicBadge, ProvenanceLegend } from "../components/provenance";
import { embeddedRunsDirectory, isEmbedded } from "../embedded-workspace";
import { findFile, recordValue, textValue } from "../lib/discovery";
import { csvFilename, type ExportColumn } from "../lib/export";
import { fmt, shortHash } from "../lib/format";
import {
  loadScenario,
  resolveSeats,
  routingByTurnTitle,
  routingLabel,
  seatTable,
  speakerAudit,
  speakerNames,
  transcripts,
  turnTitles,
  variantHeadline,
  type PanelSeat,
  type ScenarioLoad,
  type SeatConditionCell,
  type SeatTable,
} from "../lib/panels";
import type { Generation, LocalDirectoryHandle, RunFile, WorkspaceRun } from "../lib/types";
import "./panels.css";

export type { PanelTranscript } from "../lib/panels";

const ALL_TURNS = "All turns";
const ALL_SPEAKERS = "All seats";

const meanText = (value: number | null, digits = 1) => value == null ? "—" : value.toFixed(digits);
const deltaText = (value: number | null, digits = 1) => value == null ? "—" : fmt(value, digits);
const deltaTone = (value: number | null) => value == null ? "" : value < 0 ? "delta-down" : value > 0 ? "delta-up" : "delta-flat";

/// The seat-aggregate export. `turns` is a count of stored records; every
/// mean and Δ is computed by the viewer over stored per-record fields — no
/// engine artifact carries this cut, which is exactly why the column stamps
/// matter here.
type SeatExportRow = { turnTitle: string; speaker: string; cell: SeatConditionCell };

const seatColumns: ExportColumn<SeatExportRow>[] = [
  { header: "turnTitle", kind: "stored", value: (row) => row.turnTitle },
  { header: "seat", kind: "stored", value: (row) => row.speaker },
  { header: "condition", kind: "stored", value: (row) => row.cell.condition },
  { header: "isBaselineColumn", kind: "derived", value: (row) => row.cell.isBaseline },
  { header: "turns", kind: "derived", value: (row) => row.cell.turns },
  { header: "meanWordCount", kind: "derived", value: (row) => row.cell.meanWordCount },
  { header: "deltaWordCount", kind: "derived", value: (row) => row.cell.deltaWordCount },
  { header: "percentWordCount", kind: "derived", value: (row) => row.cell.percentWordCount },
  { header: "meanDistinct2", kind: "derived", value: (row) => row.cell.meanDistinct2 },
  { header: "deltaDistinct2", kind: "derived", value: (row) => row.cell.deltaDistinct2 },
];

const seatExportRows = (table: SeatTable, turnTitle: string): SeatExportRow[] =>
  table.seats.flatMap((seat) => seat.cells.map((cell) => ({ turnTitle, speaker: seat.speaker, cell })));

/// The stored facts about one seat: who sat there, on what base model, and
/// whether a variant artifact was in the chair. Nothing here is computed.
function SeatCard({ seat }: { seat: PanelSeat }) {
  const variant = seat.variant;
  return (
    <article className="seat-card">
      <header>
        <div><strong>{seat.name}</strong><span>{seat.agentID || "no agent id"}</span></div>
        {variant.state === "stock"
          ? <Badge tone="neutral">stock</Badge>
          : variant.state === "loaded"
            ? <Badge tone="blue">{variant.artifact.injections.length ? "steered" : "variant"}</Badge>
            : variant.state === "drifted"
              ? <Badge tone="warn">steered · artifact drifted</Badge>
              : <Badge tone="warn">variant not read</Badge>}
      </header>
      <dl>
        <div><dt>Base model</dt><dd>{seat.baseModelID || "not stamped"}</dd></div>
        {variant.state === "stock" && <div><dt>Agent</dt><dd>Stock base model — the scenario declares no variant artifact for this seat.</dd></div>}
        {(variant.state === "loaded" || variant.state === "drifted") && <>
          <div><dt>Agent</dt><dd className="seat-agent-name">{variant.artifact.name || "unnamed variant"}</dd></div>
          <div><dt>Intervention</dt><dd className="seat-injections">{variantHeadline(variant.artifact)}</dd></div>
          {variant.artifact.alphaInNormUnits !== null && <div><dt>Strength units</dt><dd>{variant.artifact.alphaInNormUnits ? "residual-norm units" : "raw activation units"}{variant.artifact.bandWidth != null ? ` · band ${variant.artifact.bandWidth}` : ""}</dd></div>}
          {variant.artifact.promotion
            ? <div><dt>Promotion</dt><dd>
              {variant.artifact.promotion.promotedBy || "not stamped"}
              {variant.artifact.promotion.winningLayer != null && variant.artifact.promotion.winningAlpha != null
                ? ` · winning cell L${variant.artifact.promotion.winningLayer}, α ${variant.artifact.promotion.winningAlpha}`
                : ""}
              {variant.artifact.promotion.sweepRun ? <small>{variant.artifact.promotion.sweepRun}</small> : null}
              {variant.artifact.promotion.overrideReason ? <small>override: {variant.artifact.promotion.overrideReason}</small> : null}
              {variant.artifact.promotion.selectionOutcome ? <small>selection: {variant.artifact.promotion.selectionOutcome}</small> : null}
            </dd></div>
            : <div><dt>Promotion</dt><dd>No promotion certificate — hand-created variant (legal, but not sweep-selected).</dd></div>}
          {variant.artifact.baseRevision && <div><dt>Base revision</dt><dd className="mono-cell">{shortHash(variant.artifact.baseRevision)}</dd></div>}
        </>}
        {variant.state === "drifted" && <div><dt>Drift</dt><dd className="seat-unreachable">{variant.note}</dd></div>}
        {(variant.state === "unreachable" || variant.state === "unreadable") && <div><dt>Agent</dt><dd className="seat-unreachable">{variant.note}</dd></div>}
        {seat.variantPath && <div><dt>Artifact</dt><dd className="mono-cell">{seat.variantPath}</dd></div>}
        {seat.variantHash && <div><dt>Artifact hash</dt><dd className="mono-cell">{shortHash(seat.variantHash)}</dd></div>}
      </dl>
    </article>
  );
}

export function PanelView({ run, onOpenFile }: { run: WorkspaceRun | null; onOpenFile: (file: RunFile) => void }) {
  const rows = useMemo<Generation[]>(() => run?.generationRows ?? [], [run]);
  const arms = useMemo(() => transcripts(rows), [rows]);
  const replicates = useMemo(() => [...new Set(arms.map((item) => item.replicate))].sort((a, b) => a - b), [arms]);
  const speakers = useMemo(() => speakerNames(rows), [rows]);
  const titles = useMemo(() => turnTitles(rows), [rows]);
  const [replicate, setReplicate] = useState(0);
  const [endpoint, setEndpoint] = useState("");
  const [aggregateTurn, setAggregateTurn] = useState(ALL_TURNS);
  const [mode, setMode] = useState<"replicate" | "speaker">("replicate");
  const [transcriptSpeaker, setTranscriptSpeaker] = useState(ALL_SPEAKERS);
  const [transcriptTurn, setTranscriptTurn] = useState(ALL_TURNS);
  const [auditSpeaker, setAuditSpeaker] = useState("");
  const [scenarioLoad, setScenarioLoad] = useState<ScenarioLoad | null>(null);
  const [seats, setSeats] = useState<PanelSeat[]>([]);
  const [resolvingSeats, setResolvingSeats] = useState(false);

  // Self-load: the scenario snapshot lives in the run directory, the variant
  // artifacts it names live elsewhere under runs/ — neither is hydrated onto
  // the run object, so this view reads them itself, keyed on the run.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- clears the previous run's selections before this effect re-reads the scenario snapshot and its variant artifacts from disk
    setReplicate(replicates[0] ?? 0);
    setEndpoint(run?.panelEffects[0]?.endpoint ?? "");
    setAggregateTurn(ALL_TURNS);
    setMode("replicate");
    setTranscriptSpeaker(ALL_SPEAKERS);
    setTranscriptTurn(ALL_TURNS);
    setAuditSpeaker(speakers[0] ?? "");
    setScenarioLoad(null);
    setSeats([]);
    if (!run) return;
    let cancelled = false;
    setResolvingSeats(true);
    void (async () => {
      const load = await loadScenario(run);
      if (cancelled) return;
      setScenarioLoad(load);
      if (load.scenario) {
        // Only the embedded bridge serves the whole runs/ tree; a browser
        // session holds a handle to what the researcher picked, and this
        // view is not given the workspace root — so the honest answer there
        // is "not reachable", never a guess.
        const runsDirectory = isEmbedded() ? (embeddedRunsDirectory() as unknown as LocalDirectoryHandle) : null;
        const resolved = await resolveSeats(load.scenario, runsDirectory);
        if (!cancelled) setSeats(resolved);
      }
      if (!cancelled) setResolvingSeats(false);
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- reload per run
  }, [run?.key]);

  const table = useMemo(() => seatTable(rows, aggregateTurn === ALL_TURNS ? null : aggregateTurn), [rows, aggregateTurn]);
  // The motivating signal: which seat × turn moved most in word count.
  // Derived (means over stored wordCount), ranked by |Δ| against the same
  // seat's baseline cell.
  const shifts = useMemo(() => titles.flatMap((title) => seatTable(rows, title).seats.flatMap((seat) =>
    seat.cells.flatMap((cell) => cell.deltaWordCount == null ? [] : [{
      title, speaker: seat.speaker, condition: cell.condition,
      delta: cell.deltaWordCount, percent: cell.percentWordCount,
      from: seat.cells.find((candidate) => candidate.isBaseline)?.meanWordCount ?? null,
      to: cell.meanWordCount,
    }])))
    .sort((left, right) => Math.abs(right.delta) - Math.abs(left.delta))
    .slice(0, 6), [rows, titles]);
  const audit = useMemo(() => auditSpeaker
    ? speakerAudit(rows, auditSpeaker, transcriptTurn === ALL_TURNS ? null : transcriptTurn)
    : null, [rows, auditSpeaker, transcriptTurn]);
  const routing = useMemo(() => routingByTurnTitle(scenarioLoad?.scenario ?? null), [scenarioLoad]);
  const panelRowCount = useMemo(() => rows.filter((row) => row.speakerName || row.turnTitle).length, [rows]);
  const unstampedWordCounts = useMemo(
    () => rows.filter((row) => (row.speakerName || row.turnTitle) && !row.wordCountStored).length, [rows]);

  if (!run) return <div className="view-enter inner-view"><header className="page-title"><div><span className="section-number">MULTI-AGENT STUDIES</span><h1>Panel dynamics</h1><p>Select a local multi-agent run to inspect seat attribution, per-seat aggregates, and reconstructed transcripts.</p></div></header><div className="card no-run-card"><span>⌂</span><h2>No local run selected</h2><p>Choose a workspace and panel run from the sidebar.</p></div></div>;

  const visibleTranscripts = arms.filter((item) => item.replicate === replicate).map((item) => ({
    ...item,
    turns: item.turns.filter((turn) =>
      (transcriptSpeaker === ALL_SPEAKERS || (turn.speakerName || "Unstamped speaker") === transcriptSpeaker) &&
      (transcriptTurn === ALL_TURNS || (turn.turnTitle || "Untitled turn") === transcriptTurn)),
  }));
  const selectedEffect = run.panelEffects.find((item) => item.endpoint === endpoint) ?? run.panelEffects[0] ?? null;
  const modelBySeat = recordValue(run.report.modelBySeat);
  const modelsUsed = Array.isArray(run.report.modelsUsed) ? run.report.modelsUsed.filter((item): item is string => typeof item === "string") : [];
  const panelFile = findFile(run.files, "panel-effects.csv");
  const generationsFile = findFile(run.files, "generations.jsonl");
  const scenarioFile = findFile(run.files, "scenario.json");
  const point = (value: number | null) => value == null ? "—" : fmt(value, 3);
  const steeredSeats = seats.filter((seat) => (seat.variant.state === "loaded" || seat.variant.state === "drifted") && seat.variant.artifact.injections.length);

  return <div className="view-enter inner-view panel-view">
    <header className="page-title"><div><span className="section-number">{arms.length} TRANSCRIPTS · {replicates.length} REPLICATES · {run.panelEffects.length} ENDPOINTS</span><h1>Multi-agent panels</h1><p>Read who was steered, how each seat behaved, and the canonical turns that produced the panel outcome.</p></div><div className="title-actions">{scenarioFile && <button className="secondary" onClick={() => onOpenFile(scenarioFile)}>Scenario</button>}{generationsFile && <button className="secondary" onClick={() => onOpenFile(generationsFile)}>Turn records</button>}{panelFile && <button className="primary" onClick={() => onOpenFile(panelFile)}>Effect table <span>→</span></button>}</div></header>
    <ProvenanceLegend />
    {!arms.length && !run.panelEffects.length ? <div className="card no-run-card"><span>∅</span><h2>No panel artifacts found</h2><p>This run has neither multi-agent turn fields nor a readable panel-effects.csv.</p></div> : <>

      <section className="card seat-attribution-card">
        <header className="section-header">
          <div><span className="section-number">SEAT ATTRIBUTION</span><h2>Who sat in each chair</h2><p>Read from the run&apos;s scenario snapshot and the variant artifacts it names — stored values, not inferred from output.</p></div>
          {scenarioLoad?.scenario && <Badge tone={steeredSeats.length ? "blue" : "neutral"}>{steeredSeats.length} steered / {seats.length} seats</Badge>}
        </header>
        {!scenarioLoad ? <p className="seat-status">Reading scenario.json…</p> : scenarioLoad.scenario ? <>
          <div className="scenario-line">
            <div><span>Scenario</span><strong>{scenarioLoad.scenario.name || "unnamed scenario"}</strong></div>
            <div><span>Base model</span><strong>{scenarioLoad.scenario.baseModelID || "not stamped"}</strong></div>
            <div><span>Turns declared</span><strong>{scenarioLoad.scenario.turns.length || "—"}</strong></div>
            <div><span>Temperature</span><strong>{scenarioLoad.scenario.temperature == null ? "—" : scenarioLoad.scenario.temperature.toFixed(2)}</strong></div>
          </div>
          <div className="seat-grid">{seats.map((seat) => <SeatCard seat={seat} key={seat.agentID || seat.name} />)}</div>
          {resolvingSeats && <p className="seat-status">Resolving variant artifacts…</p>}
          {!isEmbedded() && seats.some((seat) => seat.variant.state === "unreachable") && <p className="table-note"><strong>Browser session.</strong> Variant artifacts live outside the selected run directory, so only the scenario&apos;s declared path and hash can be shown here. The app&apos;s embedded explorer resolves them from the workspace&apos;s runs/ tree.</p>}
        </> : <>
          <p className="seat-absent">{scenarioLoad.note}</p>
          {Object.keys(modelBySeat).length > 0 && <div className="seat-fallback"><span>Model by seat <small>report.json</small></span><div>{Object.entries(modelBySeat).map(([seat, model]) => <div key={seat}><strong>{seat}</strong><span>{String(model)}</span></div>)}</div></div>}
          {speakers.length > 0 && <p className="seat-status">Speakers observed in the turn records: {speakers.join(", ")} <DerivedBadge formula="distinct speakerName values over this run's generation records" /> — which of them carried a steered variant is not recoverable from this run directory.</p>}
        </>}
      </section>

      <section className="panel-model-strip"><div><span>Unit of analysis</span><strong>{textValue(run.report, "unitOfAnalysis") || "Transcript"}</strong></div><div><span>Replicates / condition</span><strong>{typeof run.report.transcriptsPerCondition === "number" ? run.report.transcriptsPerCondition : replicates.length || "—"}</strong></div><div><span>Models used</span><strong>{modelsUsed.length || (run.model ? 1 : "—")}</strong><small>{modelsUsed.join(" · ") || run.model}</small></div><div><span>Seats attributed</span><strong>{seats.length || Object.keys(modelBySeat).length || speakers.length || "—"}</strong><small>{scenarioLoad?.scenario ? "scenario.json" : Object.keys(modelBySeat).length ? "report.json modelBySeat" : "turn records"}</small></div></section>

      {run.panelEffects.length > 0 && <section className="card panel-effects-card"><header className="section-header"><div><span className="section-number">PROPAGATION DECOMPOSITION</span><h2>Direct, spillover, and group movement</h2></div><select value={selectedEffect?.endpoint ?? ""} onChange={(event) => setEndpoint(event.target.value)}>{run.panelEffects.map((row) => <option key={row.endpoint}>{row.endpoint}</option>)}</select></header>{selectedEffect && <><div className="panel-estimates"><div className="direct"><span>Direct effect</span><strong>{point(selectedEffect.direct)}</strong><small>Treated seats · n = {selectedEffect.directN}</small></div><div className="spill"><span>Spillover effect</span><strong>{point(selectedEffect.spillover)}</strong><small>Untreated seats · n = {selectedEffect.spilloverN}</small></div><div className="group"><span>Group outcome</span><strong>{point(selectedEffect.group)}</strong><small>Designated panel turn · n = {selectedEffect.groupN}</small></div><div><span>Transmission ratio</span><strong>{point(selectedEffect.transmissionRatio)}</strong><small>Spillover ÷ direct</small></div><div><span>Amplification</span><strong>{point(selectedEffect.amplification)}</strong><small>Group ÷ direct</small></div></div><div className="ratio-warning"><span>i</span><p><strong>Ratios are descriptive.</strong> Transmission and amplification divide by the direct effect and become unstable as that denominator approaches zero; interpret them alongside the component estimates. {selectedEffect.droppedTurns} turn{selectedEffect.droppedTurns === 1 ? " was" : "s were"} dropped for missing pairs or endpoint parses.</p></div></>}</section>}

      {table.seats.length > 0 && <section className="card seat-aggregates-card">
        <header className="section-header">
          <div><span className="section-number">PER-SEAT AGGREGATES</span><h2>Turns, words, and distinct-2 by seat <DerivedBadge formula="means over this run's stored per-record wordCount and distinct2; Δ = condition mean − the same seat's baseline-condition mean" /></h2><p>Descriptive only. Votes, dispositions, and dissents come from an engine turn-endpoint parse, never from reading turn text here.</p></div>
          <div className="seat-aggregate-actions">
            <label className="turn-filter"><span>Turn</span><select value={aggregateTurn} onChange={(event) => setAggregateTurn(event.target.value)}><option>{ALL_TURNS}</option>{titles.map((title) => <option key={title}>{title}</option>)}</select></label>
            <ExportButton filename={csvFilename(run.name, "seat-aggregates")} columns={seatColumns} rows={seatExportRows(table, aggregateTurn)} />
          </div>
        </header>
        {!table.baselineDeclared && table.baseline && <p className="seat-status">No condition is named <code>baseline</code>; deltas compare against <strong>{table.baseline}</strong>, the first condition. <HeuristicBadge assumption="the first condition in sort order is the comparison arm when none is named baseline" /></p>}
        {/* The means average `wordCount`; when a record did not stamp one,
            the loader counted the text itself. Which rows those are is a
            fact about the run, so it is stated rather than absorbed. */}
        {unstampedWordCounts > 0 && <p className="seat-status">{unstampedWordCounts} of {panelRowCount} turn record{panelRowCount === 1 ? "" : "s"} carried no stamped <code>wordCount</code>; the viewer counted those outputs itself, and they are inside these means. <DerivedBadge formula="whitespace-token count of the output text, used only where the record stamped no wordCount" /></p>}
        <div className="seat-table-scroll">
          <table className="seat-table">
            <thead><tr><th>Seat</th><th>Condition</th><th className="numeric-cell">Turns</th><th className="numeric-cell">Mean words</th><th className="numeric-cell">Δ words</th><th className="numeric-cell">Δ %</th><th className="numeric-cell">Mean distinct-2</th><th className="numeric-cell">Δ distinct-2</th></tr></thead>
            <tbody>{table.seats.flatMap((seat) => seat.cells.map((cell, index) => <tr key={`${seat.speaker}-${cell.condition}`} className={index === 0 ? "seat-first-row" : ""}>
              <th scope="row">{index === 0 ? seat.speaker : ""}</th>
              <td>{cell.condition}{cell.isBaseline ? <small> baseline</small> : null}</td>
              <td className="numeric-cell">{cell.turns}</td>
              <td className="numeric-cell">{meanText(cell.meanWordCount)}</td>
              <td className={`numeric-cell ${deltaTone(cell.deltaWordCount)}`}>{deltaText(cell.deltaWordCount)}</td>
              <td className={`numeric-cell ${deltaTone(cell.percentWordCount)}`}>{cell.percentWordCount == null ? "—" : `${fmt(cell.percentWordCount, 1)}%`}</td>
              <td className="numeric-cell">{meanText(cell.meanDistinct2, 3)}</td>
              <td className={`numeric-cell ${deltaTone(cell.deltaDistinct2)}`}>{deltaText(cell.deltaDistinct2, 3)}</td>
            </tr>))}</tbody>
          </table>
        </div>
        <p className="table-note">{table.rowsUsed} turn record{table.rowsUsed === 1 ? "" : "s"} in this cut{aggregateTurn === ALL_TURNS ? " (every turn title)" : ` (${aggregateTurn})`}. Cell means are unweighted over the records present; a seat with fewer turns in one condition is visible in the Turns column.</p>
        {shifts.length > 0 && <div className="shift-list">
          <h3>Largest word-count shifts by seat × turn <DerivedBadge formula="per turn title, condition mean wordCount − the same seat's baseline mean; ranked by |Δ|" /></h3>
          {shifts.map((shift) => <button key={`${shift.title}-${shift.speaker}-${shift.condition}`} onClick={() => setAggregateTurn(shift.title)}>
            <div><strong>{shift.speaker}</strong><span>{shift.title}</span></div>
            <div className="shift-numbers"><em className={deltaTone(shift.delta)}>{meanText(shift.from, 0)} → {meanText(shift.to, 0)} words</em><span>{shift.condition} · {deltaText(shift.delta)} · {shift.percent == null ? "—" : `${fmt(shift.percent, 0)}%`}</span></div>
          </button>)}
        </div>}
      </section>}

      <section className="transcript-section">
        <header>
          <div><span className="section-number">CANONICAL TURN RECORDS</span><h2>{mode === "replicate" ? "Transcript comparison" : "Cross-replicate seat audit"}</h2><p>{mode === "replicate" ? "One arm per condition for the same independent play-through." : "One seat's turns across every replicate, conditions side by side."}</p></div>
          <div className="transcript-controls">
            <div className="mode-toggle" role="group" aria-label="Transcript view mode">
              <button className={mode === "replicate" ? "active" : ""} onClick={() => setMode("replicate")}>By replicate</button>
              <button className={mode === "speaker" ? "active" : ""} onClick={() => setMode("speaker")} disabled={!speakers.length}>By seat</button>
            </div>
            {mode === "replicate"
              ? <label><span>Replicate</span><select value={replicate} onChange={(event) => setReplicate(Number(event.target.value))}>{replicates.map((value) => <option value={value} key={value}>Replicate {value}</option>)}</select></label>
              : <label><span>Seat</span><select value={auditSpeaker} onChange={(event) => setAuditSpeaker(event.target.value)}>{speakers.map((name) => <option key={name}>{name}</option>)}</select></label>}
            {mode === "replicate" && <label><span>Seat</span><select value={transcriptSpeaker} onChange={(event) => setTranscriptSpeaker(event.target.value)}><option>{ALL_SPEAKERS}</option>{speakers.map((name) => <option key={name}>{name}</option>)}</select></label>}
            <label><span>Turn</span><select value={transcriptTurn} onChange={(event) => setTranscriptTurn(event.target.value)}><option>{ALL_TURNS}</option>{titles.map((title) => <option key={title}>{title}</option>)}</select></label>
          </div>
        </header>

        {mode === "replicate" ? <>
          <div className="transcript-grid" style={{ gridTemplateColumns: `repeat(${Math.max(1, visibleTranscripts.length)}, minmax(300px, 1fr))` }}>{visibleTranscripts.map((transcript) => <article className="transcript-arm" key={`${transcript.condition}-${transcript.replicate}`}>
            <header><div><span>Condition</span><strong>{transcript.condition}</strong></div><Badge tone={transcript.condition === "baseline" ? "neutral" : "blue"}>{transcript.turns.length} turns</Badge></header>
            <div>{transcript.turns.map((turn, index) => <PanelTurn turn={turn} index={index} routing={routing} key={`${turn.id}-${index}`} />)}</div>
          </article>)}</div>
          {!visibleTranscripts.some((item) => item.turns.length) && <div className="card empty-state">No turn records match this replicate and filter combination.</div>}
        </> : audit && audit.replicates.length ? <>
          <div className="audit-list">{audit.replicates.map((entry) => <article className="audit-row" key={entry.replicate}>
            <div className="audit-rail"><span>Replicate</span><strong>{entry.replicate}</strong></div>
            <div className="audit-columns" style={{ gridTemplateColumns: `repeat(${Math.max(1, audit.conditions.length)}, minmax(280px, 1fr))` }}>{entry.columns.map((column) => <div className="audit-column" key={column.condition}>
              <header><strong>{column.condition}</strong><Badge tone={column.condition === "baseline" ? "neutral" : "blue"}>{column.turns.length} turn{column.turns.length === 1 ? "" : "s"}</Badge></header>
              {column.turns.map((turn, index) => <PanelTurn turn={turn} index={index} routing={routing} key={`${turn.id}-${index}`} />)}
              {!column.turns.length && <p className="audit-empty">No turn recorded in this condition.</p>}
            </div>)}</div>
          </article>)}</div>
          <p className="table-note">{audit.turnsShown} turn record{audit.turnsShown === 1 ? "" : "s"} for {audit.speaker}{audit.turnTitle ? ` · ${audit.turnTitle}` : " · every turn title"}, across {audit.replicates.length} replicate{audit.replicates.length === 1 ? "" : "s"}.</p>
        </> : <div className="card empty-state">No turn records for this seat and turn filter.</div>}
      </section>

      {Object.keys(modelBySeat).length > 0 && <section className="card seat-model-card"><header className="section-header"><div><span className="section-number">MODEL ATTRIBUTION</span><h2>Model by seat</h2></div></header><div>{Object.entries(modelBySeat).map(([seat, model]) => <div key={seat}><strong>{seat}</strong><span>{String(model)}</span></div>)}</div></section>}
    </>}
  </div>;
}

function PanelTurn({ turn, index, routing }: { turn: Generation; index: number; routing: Map<string, string> }) {
  const declared = turn.turnTitle ? routing.get(turn.turnTitle) : undefined;
  const isPrivate = turn.routedAgentIDs === null || turn.routedAgentIDs?.length === 0 || declared === "speakerOnly";
  return <section className="panel-turn">
    <div className="turn-rail"><i>{index + 1}</i><span /></div>
    <div className="turn-body">
      <header><div><strong>{turn.speakerName || "Unstamped speaker"}</strong><span>{turn.turnTitle || turn.caseName}</span></div>{turn.modelID && <small>{turn.modelID}</small>}</header>
      <p>{turn.output}</p>
      <footer>
        <span title={turn.wordCountStored ? "wordCount stamped on the turn record" : "the record stamped no wordCount; counted here from the output text"}>{turn.words} words{turn.wordCountStored ? "" : " (counted here)"}</span>
        <span className={isPrivate ? "private-route" : ""}>{routingLabel(turn)}{declared ? ` · ${declared}` : ""}</span>
      </footer>
    </div>
  </section>;
}
