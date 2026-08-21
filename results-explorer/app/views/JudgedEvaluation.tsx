"use client";

// The judged-evaluation reader (Phase 2, docs/RESULTS-EXPLORER-UPGRADE-PLAN.md).
// The evaluate artifacts already existed and were invisible: tallies, judge
// agreement, and — the flagship affordance — the cells where the judges
// SPLIT, with both unblinded responses side by side.
//
// This view self-loads its own artifacts (judge-report.json,
// judgments.jsonl, judging-context.json, run-status.json) rather than
// extending the shared hydrateRun, under the same bounded-read discipline
// as lib/loaders.ts. Parsing lives in lib/judged.ts; this file renders.
// Stored numbers render plain; anything the viewer counted or selected
// carries a `derived` badge.

import { useEffect, useMemo, useState } from "react";
import { DerivedBadge, ProvenanceLegend } from "../components/provenance";
import { Badge, CopyLinkButton, ExportButton } from "../components/ui";
import { responseRecordKey, splitRecordKey, takePendingRecord, updateDeepLink } from "../lib/deeplink";
import { findFile } from "../lib/discovery";
import { csvFilename, type ExportColumn } from "../lib/export";
import { shortHash } from "../lib/format";
import {
  cellKey,
  confidenceHistogram,
  disagreementCells,
  findSourceRun,
  judgeTallies,
  loadJudgments,
  loadSourceResponses,
  parseJudgeReport,
  parseJudgingContext,
  parseRunStatus,
  readJSONArtifact,
  type ConditionTally,
  type JudgeReport,
  type JudgeTally,
  type JudgingContext,
  type JudgmentCell,
  type JudgmentOutcome,
  type JudgmentRow,
  type RunStatus,
  type SourceResponse,
} from "../lib/judged";
import type { RunFile, View, WorkspaceRun } from "../lib/types";
import "./judged.css";

type JudgedData = {
  report: JudgeReport;
  context: JudgingContext;
  status: RunStatus;
  rows: JudgmentRow[];
  skipped: number;
  truncated: boolean;
  judgmentsPresent: boolean;
};

const outcomeLabel: Record<JudgmentOutcome, string> = { variant: "Variant", baseline: "Baseline", tie: "Tie", unknown: "Not stamped" };
const percent = (value: number | null, digits = 0) => value == null ? "—" : `${(value * 100).toFixed(digits)}%`;
const count = (value: number | null) => value == null ? "—" : String(value);
const decimal = (value: number | null, digits = 2) => value == null ? "—" : value.toFixed(digits);

/// The letter a judge saw a given arm under. `baselineWas` is the blinding
/// the engine stamped on the pair; the variant is necessarily the other
/// letter. Empty when the row carried no blinding stamp — never guessed.
const otherLetter = (letter: string) => letter === "A" ? "B" : letter === "B" ? "A" : "";

// --- exports ---------------------------------------------------------------
// The per-condition tallies are judge-report.json's own numbers; the split
// browser's rows are judgments.jsonl's. The one column either table computes
// is marked as such, and the per-judge table's kind is decided ONCE by
// whether the report carried per-judge blocks — a file may not mix kinds
// inside one column.

const tallyColumns: ExportColumn<ConditionTally>[] = [
  { header: "condition", kind: "stored", value: (row) => row.condition },
  { header: "pairs", kind: "stored", value: (row) => row.pairs },
  { header: "baselineWins", kind: "stored", value: (row) => row.baselineWins },
  { header: "ties", kind: "stored", value: (row) => row.ties },
  { header: "variantWins", kind: "stored", value: (row) => row.variantWins },
  { header: "meanConfidence", kind: "stored", value: (row) => row.meanConfidence },
];

/// Nullable throughout, because the STORED half comes from a report block
/// that may omit any of these keys — and a missing tally must leave the cell
/// blank, never write a 0 that reads as "the judge chose it zero times".
type PerJudgeExportRow = {
  condition: string; judge: string;
  n: number | null; baselineWins: number | null; ties: number | null;
  variantWins: number | null; unknown: number | null; meanConfidence: number | null;
};

const judgeTallyColumns = (stored: boolean): ExportColumn<PerJudgeExportRow>[] => {
  const kind = stored ? "stored" as const : "derived" as const;
  return [
    { header: "condition", kind: "stored", value: (row) => row.condition },
    { header: "judge", kind: "stored", value: (row) => row.judge },
    { header: "n", kind, value: (row) => row.n },
    { header: "baselineWins", kind, value: (row) => row.baselineWins },
    { header: "ties", kind, value: (row) => row.ties },
    { header: "variantWins", kind, value: (row) => row.variantWins },
    { header: "notStamped", kind, value: (row) => row.unknown },
    { header: "meanConfidence", kind, value: (row) => row.meanConfidence },
  ];
};

const perJudgeRows = (report: JudgeReport, derived: JudgeTally[], stored: boolean): PerJudgeExportRow[] =>
  stored
    ? report.judges.flatMap((judge) => judge.conditions.map((entry) => ({
      condition: entry.condition, judge: judge.name,
      n: entry.pairs, baselineWins: entry.baselineWins, ties: entry.ties,
      variantWins: entry.variantWins, unknown: null, meanConfidence: entry.meanConfidence,
    })))
    : derived.map((tally) => ({
      condition: tally.condition, judge: tally.judge,
      n: tally.n, baselineWins: tally.baselineWins, ties: tally.ties,
      variantWins: tally.variantWins, unknown: tally.unknown, meanConfidence: tally.meanConfidence,
    }));

type SplitExportRow = { cell: JudgmentCell; row: JudgmentRow };

const splitColumns: ExportColumn<SplitExportRow>[] = [
  { header: "condition", kind: "stored", value: ({ cell }) => cell.condition },
  { header: "promptID", kind: "stored", value: ({ cell }) => cell.promptID },
  { header: "sampleIndex", kind: "stored", value: ({ cell }) => cell.sampleIndex },
  { header: "judge", kind: "stored", value: ({ row }) => row.judge },
  { header: "judgeModel", kind: "stored", value: ({ row }) => row.judgeModel },
  { header: "outcome", kind: "stored", value: ({ row }) => row.outcome },
  { header: "winner", kind: "stored", value: ({ row }) => row.winner },
  { header: "baselineWas", kind: "stored", value: ({ row }) => row.baselineWas },
  { header: "confidence", kind: "stored", value: ({ row }) => row.confidence },
  { header: "briefReason", kind: "stored", value: ({ row }) => row.briefReason },
  { header: "reasoningTruncated", kind: "stored", value: ({ row }) => row.reasoningTruncated },
  { header: "judgesInCell", kind: "derived", value: ({ cell }) => cell.verdicts.length },
  { header: "cellDisagrees", kind: "derived", value: ({ cell }) => cell.disagrees },
];

export function JudgedEvaluationView({ run, workspaceRuns, onActivateRun, onNavigate, onOpenFile }: {
  run: WorkspaceRun | null;
  workspaceRuns: WorkspaceRun[];
  onActivateRun: (run: WorkspaceRun) => void;
  onNavigate: (view: View) => void;
  onOpenFile: (file: RunFile) => void;
}) {
  const [data, setData] = useState<JudgedData | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [sources, setSources] = useState<{ responses: Map<string, SourceResponse>; truncated: boolean; present: boolean } | null>(null);
  const [sourceLoading, setSourceLoading] = useState(false);
  const [selectedKey, setSelectedKey] = useState("");
  const [conditionFilter, setConditionFilter] = useState("All conditions");
  const [judgeFilter, setJudgeFilter] = useState("All judges");

  useEffect(() => {
    let cancelled = false;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- clears the previous run's artifacts before this effect re-reads them from disk; the alternative (remount via key) would discard the pane's scroll and re-run every load
    setData(null); setSources(null); setLoadError(""); setSelectedKey(""); setConditionFilter("All conditions"); setJudgeFilter("All judges");
    if (!run) return;
    setLoading(true);
    void (async () => {
      try {
        const [reportJSON, contextJSON, statusJSON, judgments] = await Promise.all([
          readJSONArtifact(run, "judge-report.json"),
          readJSONArtifact(run, "judging-context.json"),
          readJSONArtifact(run, "run-status.json"),
          loadJudgments(run),
        ]);
        if (cancelled) return;
        setData({
          report: parseJudgeReport(reportJSON),
          context: parseJudgingContext(contextJSON),
          status: parseRunStatus(statusJSON),
          rows: judgments.rows,
          skipped: judgments.skipped,
          truncated: judgments.truncated,
          judgmentsPresent: judgments.present,
        });
      } catch (error) {
        if (!cancelled) setLoadError(error instanceof Error ? error.message : "The judged-evaluation artifacts could not be read from this run.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [run?.key]);

  // Stable identity: the memos below (cells, tallies, histograms) must not
  // recompute on every render of a large judgment file.
  const rows = useMemo(() => data?.rows ?? [], [data]);
  const report = data?.report;
  const context = data?.context;
  const sourceRunName = report?.sourceRun || context?.sourceRun || data?.status.sourceRun || "";
  const sourceRun = useMemo(() => findSourceRun(workspaceRuns, sourceRunName), [workspaceRuns, sourceRunName]);
  const derivedTallies = useMemo(() => judgeTallies(rows), [rows]);
  const cells = useMemo(() => disagreementCells(rows), [rows]);
  const conditions = useMemo(() => [...new Set(rows.map((row) => row.condition))].sort(), [rows]);
  const judgeNames = useMemo(() => [...new Set(rows.map((row) => row.judge))].sort(), [rows]);
  const visibleCells = useMemo(() => cells.filter((cell) => (conditionFilter === "All conditions" || cell.condition === conditionFilter) && (judgeFilter === "All judges" || cell.rows.some((row) => row.judge === judgeFilter))), [cells, conditionFilter, judgeFilter]);
  const selected = visibleCells.find((cell) => cell.key === selectedKey) ?? visibleCells[0] ?? null;

  // The side-by-side responses live in the SOURCE run, not this one. Load
  // them once, lazily, the first time a disagreement cell is inspected.
  useEffect(() => {
    if (!selected || !sourceRun || sources || sourceLoading) return;
    let cancelled = false;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- in-flight flag for the lazy read of the SOURCE run's responses (an external system), guarding the re-entry above
    setSourceLoading(true);
    void (async () => {
      try {
        const loaded = await loadSourceResponses(sourceRun);
        if (!cancelled) setSources(loaded);
      } catch {
        if (!cancelled) setSources({ responses: new Map(), truncated: false, present: false });
      } finally {
        if (!cancelled) setSourceLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [selected?.key, sourceRun?.key]);

  // A `?record=` permalink names a judged cell by the same (condition,
  // promptID, sampleIndex) triple the engines join on. Consumed once the
  // cells exist — a key matching nothing simply leaves the first split cell
  // selected rather than emptying the pane.
  useEffect(() => {
    if (!cells.length) return;
    const linked = takePendingRecord("judged");
    if (!linked) return;
    const [condition, promptID, sample] = splitRecordKey(linked);
    const match = cells.find((cell) => cell.condition === condition && cell.promptID === promptID && String(cell.sampleIndex) === sample);
    // eslint-disable-next-line react-hooks/set-state-in-effect -- applies a one-shot permalink taken from the deep-link queue; takePendingRecord CONSUMES, so it must not run during render
    if (match) { setSelectedKey(match.key); setConditionFilter("All conditions"); setJudgeFilter("All judges"); }
  }, [cells]);

  useEffect(() => {
    if (selected) updateDeepLink({ record: responseRecordKey(selected.condition, selected.promptID, selected.sampleIndex) });
  }, [selected?.key]);

  if (!run) return <div className="view-enter inner-view"><header className="page-title"><div><span className="section-number">NO RUN SELECTED</span><h1>Judged evaluation</h1><p>Select an evaluate run to read its judge report, tallies, agreement, and the cells where the judges split.</p></div></header><div className="card no-run-card"><span>⌂</span><h2>No local run selected</h2><p>Choose a workspace and an evaluate run in the sidebar. Nothing here is inferred.</p></div></div>;

  if (loading || !data) return <div className="view-enter inner-view"><header className="page-title"><div><span className="section-number">JUDGED EVALUATION</span><h1>Judged evaluation</h1><p>Reading judge artifacts from {run.name}.</p></div></header><div className="card artifact-empty"><span>◴</span><p>{loadError || "Reading judge-report.json, judgments.jsonl, judging-context.json…"}</p></div></div>;

  const nothingPresent = !report!.present && !context!.present && !data.judgmentsPresent && !data.status.present;
  if (nothingPresent) return <div className="view-enter inner-view"><header className="page-title"><div><span className="section-number">JUDGED EVALUATION</span><h1>Judged evaluation</h1><p>Loaded from {run.name}, read-only.</p></div></header><div className="card artifact-empty"><span>∅</span><p>This run carries no judged-evaluation artifacts. A paired evaluate writes <code>judge-report.json</code>, <code>judgments.jsonl</code>, and <code>judging-context.json</code>; none of them is present here, and nothing is substituted in their place.</p></div></div>;

  const judgeReport = report!;
  const judgingContext = context!;
  const failedFile = findFile(run.files, "FAILED.md");
  const reportFile = findFile(run.files, "judge-report.json");
  const judgmentsFile = findFile(run.files, "judgments.jsonl");
  const failed = data.status.status === "failed";
  const storedPerJudge = judgeReport.judges.some((judge) => judge.conditions.length > 0);
  const tallyRows = judgeReport.conditions;
  const perJudgeExportRows = perJudgeRows(judgeReport, derivedTallies, storedPerJudge);
  // One row per judgment inside a split cell, in the order shown, honouring
  // the condition/judge filters above the browser.
  const splitExportRows: SplitExportRow[] = visibleCells.flatMap((cell) => cell.rows.map((row) => ({ cell, row })));

  // Judge identity is a pin, assembled from the two artifacts that stamp it
  // (judging-context.json before judging, judge-report.json after). No
  // number is involved — this is provenance, not a derived quantity.
  const judgeCards = [...new Set([...judgeReport.judges.map((judge) => judge.name), ...judgingContext.judges.map((judge) => judge.name), ...judgeNames])].map((name) => {
    const detail = judgeReport.judges.find((judge) => judge.name === name);
    const pinned = judgingContext.judges.find((judge) => judge.name === name);
    const row = rows.find((item) => item.judge === name);
    return {
      name,
      kind: detail?.kind || pinned?.kind || row?.judgeKind || "",
      requested: detail?.requestedModel || pinned?.model || row?.judgeModel || "",
      actual: detail?.actualModel || "",
      provider: detail?.provider || pinned?.provider || row?.judgeProvider || "",
      revision: detail?.revision || pinned?.revision || row?.judgeRevision || "",
      dtype: detail?.dtype || pinned?.dtype || "",
      ran: rows.some((item) => item.judge === name),
    };
  });

  const openSourceRun = (view: View | null) => {
    if (!sourceRun) return;
    onActivateRun(sourceRun);
    if (view) onNavigate(view);
  };

  return (
    <div className="view-enter inner-view judged-view">
      <header className="page-title">
        <div>
          <span className="section-number">{rows.length} JUDGMENT ROWS · {judgeCards.length} JUDGE{judgeCards.length === 1 ? "" : "S"} · {cells.length} SPLIT CELL{cells.length === 1 ? "" : "S"}{data.truncated ? " · BOUNDED PREVIEW" : ""}</span>
          <h1>Judged evaluation</h1>
          <p>Which arm each judge preferred, how far the judges agreed, and every pair they read differently — with the two responses unblinded side by side.</p>
        </div>
        <div className="title-actions">
          {judgmentsFile && <button className="secondary" onClick={() => onOpenFile(judgmentsFile)}>judgments.jsonl</button>}
          {reportFile && <button className="primary" onClick={() => onOpenFile(reportFile)}>judge-report.json <span>→</span></button>}
        </div>
      </header>

      <ProvenanceLegend />

      {failed && <div className="card judged-alert alert-failed">
        <span>!</span>
        <div>
          <strong>This evaluate FAILED — it is a failure record, not a result.</strong>
          <p>{data.status.error || "run-status.json stamps status: failed with no error message."}{data.status.errorType ? ` (${data.status.errorType})` : ""}</p>
          <p>{count(data.status.itemsWritten)} {data.status.itemLabel}{data.status.itemsWritten === 1 ? "" : "s"} written before the failure.{data.status.pendingUnits.length ? ` Never ran / incomplete: ${data.status.pendingUnits.join(", ")}.` : ""}</p>
          {failedFile && <button className="text-button" onClick={() => onOpenFile(failedFile)}>Read FAILED.md</button>}
        </div>
      </div>}

      {judgeReport.epochUnverified && <div className="card judged-alert alert-warn"><span>⚠</span><div><strong>Epoch unverified.</strong><p>The source run carried no experiment-hash stamp and was judged under allow-unverified-epoch. The report is stamped <code>epochUnverified</code>; these tallies are not epoch-checked evidence.</p></div></div>}
      {judgeReport.measurementDrift && <div className="card judged-alert alert-warn"><span>⚠</span><div><strong>Measurement drift tolerated.</strong><p>The live manifest differed from the source run&rsquo;s epoch in measurement-side fields: <code>{judgeReport.measurementDrift}</code>. Judging proceeded under the live settings.</p></div></div>}
      {judgeReport.evaluationSource === "pinnedRubric" && <div className="card judged-alert alert-note"><span>i</span><div><strong>Evaluation spec synthesized from pins.</strong><p>No explicit evaluation block: the panel and rubric came from the manifest&rsquo;s pinned judges + rubric file (<code>evaluationSource: pinnedRubric</code>).</p></div></div>}
      {judgeReport.exclusions && <div className="card judged-alert alert-note"><span>i</span><div><strong>Declared exclusions applied before judging.</strong><p>{Object.entries(judgeReport.exclusions).map(([key, value]) => `${key}: ${typeof value === "object" ? JSON.stringify(value) : String(value)}`).join(" · ")}</p></div></div>}

      <section className="card judged-pins">
        <header className="section-header"><div><span className="section-number">MEASUREMENT PINS</span><h2>What judged what</h2></div>{judgeReport.judgedOn && <Badge tone="neutral">judged on {judgeReport.judgedOn}</Badge>}</header>
        <dl className="pin-grid">
          <div><dt>Rubric file</dt><dd>{judgeReport.rubricFile || judgingContext.rubricFile || "Not stamped"}</dd></div>
          <div><dt>Rubric hash</dt><dd><code>{shortHash(judgeReport.rubricHash || judgingContext.rubricHash)}</code></dd></div>
          <div><dt>Experiment</dt><dd>{judgeReport.experiment || judgingContext.experiment || run.experiment}</dd></div>
          <div><dt>Experiment hash</dt><dd><code>{shortHash(judgeReport.experimentHash || judgingContext.experimentHash)}</code></dd></div>
          <div><dt>Source generations</dt><dd><code>{shortHash(judgingContext.sourceGenerationsSha256)}</code></dd></div>
          <div><dt>Structured prompt</dt><dd><code>{judgingContext.structuredPromptSha256 ? shortHash(judgingContext.structuredPromptSha256) : "None"}</code></dd></div>
          <div><dt>Evaluation source</dt><dd>{judgeReport.evaluationSource || "Not stamped"}</dd></div>
          <div><dt>Judgment provenance</dt><dd>{judgeReport.reusedJudgments == null && judgeReport.freshJudgments == null ? "Not stamped" : `${count(judgeReport.freshJudgments)} fresh · ${count(judgeReport.reusedJudgments)} reused`}</dd></div>
        </dl>
        <div className="source-run-row">
          <div><span>Source run</span><strong>{sourceRunName || "Not stamped"}</strong></div>
          {sourceRun
            ? <div className="source-actions"><button className="secondary" onClick={() => openSourceRun(null)}>Activate run</button><button className="secondary" onClick={() => openSourceRun("generations")}>Open its generations</button></div>
            : <p className="muted">{sourceRunName ? "That run directory is not in the loaded workspace, so the judged responses cannot be joined here." : "This evaluate stamped no source run."}</p>}
        </div>
        <div className="judge-cards">
          {judgeCards.map((judge) => <article key={judge.name} className="judge-card">
            <header><strong>{judge.name}</strong><Badge tone={judge.ran ? "blue" : "warn"}>{judge.ran ? judge.kind || "judge" : "no rows"}</Badge></header>
            <dl>
              <div><dt>Requested model</dt><dd>{judge.requested || "Not stamped"}</dd></div>
              {judge.actual && judge.actual !== judge.requested && <div><dt>Actual model</dt><dd>{judge.actual}</dd></div>}
              {judge.provider && <div><dt>Provider</dt><dd>{judge.provider}</dd></div>}
              {judge.revision && <div><dt>Revision</dt><dd><code>{shortHash(judge.revision)}</code></dd></div>}
              {judge.dtype && <div><dt>dtype</dt><dd>{judge.dtype}</dd></div>}
            </dl>
          </article>)}
          {!judgeCards.length && <p className="muted">No judge panel is stamped in this run&rsquo;s artifacts.</p>}
        </div>
      </section>

      <section className="card">
        <header className="section-header">
          <div><span className="section-number">PER-CONDITION TALLIES</span><h2>Which arm the panel preferred</h2></div>
          <div className="section-actions">
            {!judgeReport.present && <DerivedBadge formula="counted from judgments.jsonl rows because judge-report.json is absent" />}
            {tallyRows.length > 0 && <ExportButton filename={csvFilename(run.name, "judge-tallies")} columns={tallyColumns} rows={tallyRows} label="Export tallies" />}
            {derivedTallies.length > 0 && <ExportButton filename={csvFilename(run.name, "judge-tallies-by-judge")} columns={judgeTallyColumns(storedPerJudge)} rows={perJudgeExportRows} label="Export per-judge splits" />}
          </div>
        </header>
        {tallyRows.length === 0 && rows.length === 0 && <div className="artifact-empty"><span>∅</span><p>No tallies: this run has neither a <code>conditions</code> block in judge-report.json nor readable judgment rows.</p></div>}
        {(tallyRows.length ? tallyRows : []).map((condition) => {
          const total = (condition.baselineWins ?? 0) + (condition.variantWins ?? 0) + (condition.ties ?? 0);
          const share = (value: number | null) => total ? `${((value ?? 0) / total) * 100}%` : "0%";
          const perJudge = storedPerJudge
            ? judgeReport.judges.flatMap((judge) => judge.conditions.filter((entry) => entry.condition === condition.condition).map((entry) => ({ judge: judge.name, baselineWins: entry.baselineWins, variantWins: entry.variantWins, ties: entry.ties, n: entry.pairs, meanConfidence: entry.meanConfidence })))
            : derivedTallies.filter((tally) => tally.condition === condition.condition).map((tally) => ({ judge: tally.judge, baselineWins: tally.baselineWins, variantWins: tally.variantWins, ties: tally.ties, n: tally.n, meanConfidence: tally.meanConfidence }));
          return <div className="tally-block" key={condition.condition}>
            <header><div><strong>{condition.condition}</strong>{condition.condition === "baseline" && <Badge tone="neutral">anchor</Badge>}</div><span>{count(condition.pairs)} pairs</span></header>
            <div className="tally-bar" role="img" aria-label={`${condition.baselineWins ?? 0} baseline wins, ${condition.variantWins ?? 0} variant wins, ${condition.ties ?? 0} ties`}>
              <i className="seg-baseline" style={{ width: share(condition.baselineWins) }} />
              <i className="seg-tie" style={{ width: share(condition.ties) }} />
              <i className="seg-variant" style={{ width: share(condition.variantWins) }} />
            </div>
            <div className="tally-numbers">
              <div><span>Baseline wins</span><strong>{count(condition.baselineWins)}</strong></div>
              <div><span>Ties</span><strong>{count(condition.ties)}</strong></div>
              <div><span>Variant wins</span><strong>{count(condition.variantWins)}</strong></div>
              <div><span>Mean confidence</span><strong>{decimal(condition.meanConfidence)}</strong></div>
            </div>
            {perJudge.length > 0 && <table className="jv-table">
              <thead><tr><th>Judge {!storedPerJudge && <DerivedBadge formula="per-judge outcomes counted from judgments.jsonl rows for this condition" />}</th><th>n</th><th>Baseline</th><th>Ties</th><th>Variant</th><th>Mean confidence</th></tr></thead>
              <tbody>{perJudge.map((entry) => <tr key={entry.judge}><th scope="row">{entry.judge}</th><td>{count(entry.n)}</td><td>{count(entry.baselineWins)}</td><td>{count(entry.ties)}</td><td>{count(entry.variantWins)}</td><td>{decimal(entry.meanConfidence)}</td></tr>)}</tbody>
            </table>}
            {condition.structuredSummaries && <details className="structured-summaries"><summary>Structured field summaries</summary><pre>{JSON.stringify(condition.structuredSummaries, null, 2)}</pre></details>}
          </div>;
        })}
        {!tallyRows.length && rows.length > 0 && <table className="jv-table">
          <thead><tr><th>Condition · judge <DerivedBadge formula="outcomes counted from judgments.jsonl rows; judge-report.json carried no conditions block" /></th><th>n</th><th>Baseline</th><th>Ties</th><th>Variant</th><th>Mean confidence</th></tr></thead>
          <tbody>{derivedTallies.map((tally) => <tr key={`${tally.condition}-${tally.judge}`}><th scope="row">{tally.condition} · {tally.judge}</th><td>{tally.n}</td><td>{tally.baselineWins}</td><td>{tally.ties}</td><td>{tally.variantWins}</td><td>{decimal(tally.meanConfidence)}</td></tr>)}</tbody>
        </table>}
        <footer className="table-note"><strong>Stored:</strong> judge-report.json <code>conditions</code>{storedPerJudge ? " and its per-judge blocks" : ""}. The engine&rsquo;s vocabulary differs by substrate (Swift writes <code>conditionWins</code>, the server <code>variantWins</code>); both are shown here as <em>variant</em>.</footer>
      </section>

      <section className="card">
        <header className="section-header"><div><span className="section-number">INTER-JUDGE AGREEMENT</span><h2>How far the panel agreed</h2></div></header>
        {judgeReport.agreement.length ? <table className="jv-table">
          <thead><tr><th>Judge pair</th><th>Items</th><th>Percent agreement</th><th>Cohen&rsquo;s κ</th></tr></thead>
          <tbody>{judgeReport.agreement.map((entry) => <tr key={`${entry.judgeA}-${entry.judgeB}`}><th scope="row">{entry.judgeA} ↔ {entry.judgeB}</th><td>{count(entry.n)}</td><td>{percent(entry.percentAgreement, 1)}</td><td>{entry.kappa == null ? "undefined" : entry.kappa.toFixed(3)}</td></tr>)}</tbody>
        </table> : <div className="artifact-empty"><span>∅</span><p>No agreement block in judge-report.json. Percent agreement and κ are engine statistics; the viewer never computes them.</p></div>}
        {judgeReport.humanAgreement.length > 0 && <>
          <h3 className="sub-head">Judge vs. the pinned human subset</h3>
          <table className="jv-table">
            <thead><tr><th>Judge</th><th>Items</th><th>Percent agreement</th><th>Cohen&rsquo;s κ</th></tr></thead>
            <tbody>{judgeReport.humanAgreement.map((entry) => <tr key={entry.judge}><th scope="row">{entry.judge}</th><td>{count(entry.n)}</td><td>{percent(entry.percentAgreement, 1)}</td><td>{entry.kappa == null ? "undefined" : entry.kappa.toFixed(3)}</td></tr>)}</tbody>
          </table>
        </>}
        <footer className="table-note"><strong>Stored:</strong> every value in this section is read from judge-report.json.</footer>
      </section>

      <section className="card disagreement-card">
        <header className="section-header">
          <div><span className="section-number">DISAGREEMENT BROWSER <DerivedBadge formula="cells where two judges recorded different outcomes for the same (condition, promptID, sampleIndex); selection computed in the viewer from judgments.jsonl" /></span><h2>Where the judges split</h2><p>{cells.length} of {new Set(rows.map((row) => cellKey(row.condition, row.promptID, row.sampleIndex))).size} judged cells drew different verdicts. Ties count as a verdict.</p></div>
          {rows.length > 0 && <div className="disagree-filters">
            <select value={conditionFilter} onChange={(event) => { setConditionFilter(event.target.value); setSelectedKey(""); }} aria-label="Filter split cells by condition"><option>All conditions</option>{conditions.map((name) => <option key={name}>{name}</option>)}</select>
            <select value={judgeFilter} onChange={(event) => { setJudgeFilter(event.target.value); setSelectedKey(""); }} aria-label="Filter split cells by judge"><option>All judges</option>{judgeNames.map((name) => <option key={name}>{name}</option>)}</select>
            <ExportButton filename={csvFilename(run.name, "judge-disagreements")} columns={splitColumns} rows={splitExportRows} label="Export" />
          </div>}
        </header>
        {!data.judgmentsPresent ? <div className="artifact-empty"><span>∅</span><p>No <code>judgments.jsonl</code> in this run: there are no per-judge rows to compare. {failed ? "This evaluate failed before writing any." : ""}</p></div>
          : !rows.length ? <div className="artifact-empty"><span>∅</span><p><code>judgments.jsonl</code> is present but empty{data.skipped ? ` (${data.skipped} unparseable line${data.skipped === 1 ? "" : "s"} skipped)` : ""}. {failed ? "This evaluate failed before any judge produced a verdict." : "Nothing is substituted in its place."}</p></div>
          : !visibleCells.length ? <div className="artifact-empty"><span>✓</span><p>{cells.length ? "No split cells match those filters." : "The judges agreed on every cell they both read — no disagreements to browse."}</p></div>
          : <div className="reader-shell">
            <aside className="record-list">
              <div className="record-count"><span>{visibleCells.length} split cell{visibleCells.length === 1 ? "" : "s"}</span><span>{data.truncated ? "First 32 MB" : "Complete file"}</span></div>
              <div className="records">
                {visibleCells.slice(0, 500).map((cell) => <button key={cell.key} onClick={() => setSelectedKey(cell.key)} className={selected?.key === cell.key ? "selected" : ""}>
                  <div><strong>{cell.promptID}</strong><Badge tone={cell.condition === "baseline" ? "neutral" : "blue"}>{cell.condition}</Badge></div>
                  <p>{cell.verdicts.map((verdict) => `${verdict.judge} → ${outcomeLabel[verdict.outcome].toLowerCase()}`).join(" · ")}</p>
                  <footer><span>sample {cell.sampleIndex}</span><span>{cell.rows.length} judgment{cell.rows.length === 1 ? "" : "s"}</span></footer>
                </button>)}
              </div>
              {visibleCells.length > 500 && <div className="record-pagination"><span>Showing the first 500 of {visibleCells.length}</span></div>}
            </aside>
            {selected && <DisagreementDetail cell={selected} sources={sources} sourceLoading={sourceLoading} sourceRun={sourceRun} sourceRunName={sourceRunName} />}
          </div>}
      </section>

      <section className="card">
        <header className="section-header"><div><span className="section-number">CONFIDENCE <DerivedBadge formula="counts of judgments.jsonl confidence values into ten fixed 0.1-wide bins, per condition" /></span><h2>How confident the verdicts were</h2></div></header>
        {rows.length ? <div className="confidence-grid">
          {conditions.map((condition) => {
            const histogram = confidenceHistogram(rows.filter((row) => row.condition === condition));
            const peak = Math.max(1, ...histogram.bins.map((bin) => bin.count));
            return <article key={condition}>
              <header><strong>{condition}</strong><span>{histogram.counted} stamped{histogram.missing ? ` · ${histogram.missing} without a confidence` : ""}</span></header>
              <div className="histogram">{histogram.bins.map((bin) => <div key={bin.from} title={`${bin.from.toFixed(1)}–${bin.to.toFixed(1)}: ${bin.count}`}><i style={{ height: `${(bin.count / peak) * 100}%` }} /><span>{bin.from.toFixed(1)}</span></div>)}</div>
            </article>;
          })}
        </div> : <div className="artifact-empty"><span>∅</span><p>No judgment rows to summarize.</p></div>}
      </section>

      <section className="card">
        <header className="section-header"><div><span className="section-number">RUN STATUS</span><h2>Invalid responses and unfinished judges</h2></div>{data.status.present && <Badge tone={failed ? "warn" : "neutral"}>{data.status.status}</Badge>}</header>
        {data.status.present ? <>
          <dl className="pin-grid">
            <div><dt>Invalid judge responses</dt><dd>{count(data.status.invalidResponses)}</dd></div>
            <div><dt>{data.status.itemLabel === "item" ? "Judgments" : data.status.itemLabel} written</dt><dd>{count(data.status.itemsWritten)}</dd></div>
            <div><dt>Judges expected</dt><dd>{data.status.expectedUnits.join(", ") || "Not stamped"}</dd></div>
            <div><dt>Judges completed</dt><dd>{data.status.completedUnits.join(", ") || "None"}</dd></div>
            <div><dt>Never ran / incomplete</dt><dd>{data.status.pendingUnits.join(", ") || "None"}</dd></div>
            <div><dt>Evidence complete</dt><dd>{data.status.evidenceComplete == null ? "Not stamped" : data.status.evidenceComplete ? "Yes" : "No"}</dd></div>
          </dl>
          <p className="table-note">An invalid response is a judge reply that failed validation and was retried or refused; the count is the engine&rsquo;s, from run-status.json.{failedFile ? " The full failure story, including the traceback, is in FAILED.md." : ""}</p>
          {failedFile && <button className="secondary" onClick={() => onOpenFile(failedFile)}>Open FAILED.md</button>}
        </> : <div className="artifact-empty"><span>∅</span><p>No <code>run-status.json</code> in this run, so the invalid-response count and per-judge completion are unknown — not zero.</p></div>}
      </section>
    </div>
  );
}

/// The flagship pane: one split cell, every judge's verdict, and the two
/// responses unblinded through `baselineWas`.
function DisagreementDetail({ cell, sources, sourceLoading, sourceRun, sourceRunName }: {
  cell: JudgmentCell;
  sources: { responses: Map<string, SourceResponse>; truncated: boolean; present: boolean } | null;
  sourceLoading: boolean;
  sourceRun: WorkspaceRun | null;
  sourceRunName: string;
}) {
  const blinding = cell.rows.find((row) => row.baselineWas)?.baselineWas ?? "";
  const mixedBlinding = new Set(cell.rows.map((row) => row.baselineWas).filter(Boolean)).size > 1;
  const baseline = sources?.responses.get(cellKey("baseline", cell.promptID, cell.sampleIndex)) ?? null;
  const variant = sources?.responses.get(cellKey(cell.condition, cell.promptID, cell.sampleIndex)) ?? null;
  const prompt = cell.rows.find((row) => row.prompt)?.prompt || baseline?.prompt || variant?.prompt || "";
  return (
    <article className="record-detail">
      <header>
        <div>
          <span className="section-number">{cell.condition.toUpperCase()} · SAMPLE {cell.sampleIndex}</span>
          <h2>{cell.promptID}</h2>
          <CopyLinkButton view="judged" record={responseRecordKey(cell.condition, cell.promptID, cell.sampleIndex)} label="Copy link to this cell" />
        </div>
      </header>
      <div className="record-meta">
        <Badge tone="blue">{cell.condition}</Badge>
        <span>{cell.rows.length} judgments</span>
        <span>{blinding ? `Baseline shown as Response ${blinding}` : "Blinding not stamped"}</span>
        {mixedBlinding && <span className="warn-text">Blinding differs between rows</span>}
      </div>

      <section className="verdicts">
        {cell.rows.map((row, index) => <div className={`verdict verdict-${row.outcome}`} key={`${row.judge}-${index}`}>
          <header>
            <div><strong>{row.judge}</strong><span>{row.judgeModel || row.judgeKind || "model not stamped"}{row.judgeProvider ? ` · ${row.judgeProvider}` : ""}</span></div>
            <div className="verdict-outcome"><strong>{outcomeLabel[row.outcome]}</strong><span>{row.winner ? `chose Response ${row.winner}` : "no winner stamped"}{row.baselineWas ? ` · baseline was ${row.baselineWas}` : ""}</span></div>
          </header>
          <p>{row.briefReason || "No reason recorded."}{row.reasoningTruncated && <em> — the judge&rsquo;s JSON never closed; this reason is only the salvageable prefix.</em>}</p>
          <footer>
            <span>Confidence {decimal(row.confidence)}</span>
            {Object.keys(row.aScores).length > 0 && <span>A: {Object.entries(row.aScores).map(([key, value]) => `${key} ${value}`).join(", ")}</span>}
            {Object.keys(row.bScores).length > 0 && <span>B: {Object.entries(row.bScores).map(([key, value]) => `${key} ${value}`).join(", ")}</span>}
          </footer>
          {row.structuredFields && <details><summary>Structured fields</summary><pre>{JSON.stringify(row.structuredFields, null, 2)}</pre></details>}
        </div>)}
      </section>

      {prompt && <section className="text-block prompt-block"><span>TASK PROMPT</span><p>{prompt}</p></section>}

      <section className="side-by-side">
        <header><span className="section-number">THE TWO RESPONSES, UNBLINDED</span>{sources?.truncated && <Badge tone="warn">source read bounded at 32 MB</Badge>}</header>
        {!sourceRun ? <div className="artifact-empty"><span>∅</span><p>The source run <code>{sourceRunName || "(not stamped)"}</code> is not in the loaded workspace, so the judged text cannot be shown. Load that workspace to read the responses beside the verdicts.</p></div>
          : sourceLoading ? <div className="empty-state">Reading {sourceRunName}/generations.jsonl…</div>
          : !sources?.present ? <div className="artifact-empty"><span>∅</span><p>The source run carries no <code>generations.jsonl</code>.</p></div>
          : !baseline && !variant ? <div className="artifact-empty"><span>∅</span><p>No generation record in {sourceRunName} matches (<code>{cell.condition}</code>, <code>{cell.promptID}</code>, sample {cell.sampleIndex}) — the pair may lie beyond the bounded read.</p></div>
          : <div className="response-pair">
            <article className="response-baseline">
              <header><div><span>BASELINE</span><strong>{blinding ? `Shown as Response ${blinding}` : "Blinding not stamped"}</strong></div>{baseline?.wordCount != null && <small>{baseline.wordCount} words</small>}</header>
              <p>{baseline ? baseline.output : "No baseline record matched this item in the source run."}</p>
              {baseline?.seed && <footer>seed {baseline.seed}</footer>}
            </article>
            <article className="response-variant">
              <header><div><span>{cell.condition.toUpperCase()}</span><strong>{blinding ? `Shown as Response ${otherLetter(blinding)}` : "Blinding not stamped"}</strong></div>{variant?.wordCount != null && <small>{variant.wordCount} words</small>}</header>
              <p>{variant ? variant.output : "No record for this condition matched this item in the source run."}</p>
              {variant?.seed && <footer>seed {variant.seed}</footer>}
            </article>
          </div>}
      </section>
      <footer className="record-path">judgments.jsonl · {cell.rows.length} complete row{cell.rows.length === 1 ? "" : "s"}{sourceRun ? ` · responses joined from ${sourceRunName}/generations.jsonl` : ""}</footer>
    </article>
  );
}
