"use client";

// The per-response coding reader (Phase 2, docs/RESULTS-EXPLORER-UPGRADE-PLAN.md)
// for the 2026-08-04 instrument: each response is coded individually and
// blinded against a rubric's declared fields — no pair, no winner. Until
// now the instrument had no view on any surface.
//
// Self-loads coding-report.json + codings.jsonl (bounded read, 32 MB) and,
// lazily, the source run's generations so a coding can be read beside the
// text it describes. Aggregates, per-field agreement (percent / κ / mean
// absolute difference) and mean word counts are the ENGINE's; the viewer
// only filters, joins, counts and selects — all badged derived.

import { useEffect, useMemo, useState } from "react";
import { DerivedBadge, ProvenanceLegend } from "../components/provenance";
import { Badge, CopyLinkButton, ExportButton } from "../components/ui";
import { codingRecordKey, splitRecordKey, takePendingRecord, updateDeepLink } from "../lib/deeplink";
import { findFile } from "../lib/discovery";
import { csvFilename, type ExportColumn } from "../lib/export";
import { shortHash } from "../lib/format";
import {
  codingDisagreements,
  fieldsFromRows,
  formatCode,
  loadCodings,
  parseCodingReport,
  wordCountProfiles,
  type CodingConditionRow,
  type CodingFieldSpec,
  type CodingReport,
  type CodingRow,
} from "../lib/codingdata";
import {
  cellKey,
  findSourceRun,
  loadSourceResponses,
  parseRunStatus,
  readJSONArtifact,
  type RunStatus,
  type SourceResponse,
} from "../lib/judged";
import type { RunFile, View, WorkspaceRun } from "../lib/types";
import "./judged.css";
import "./coding.css";

type CodingData = { report: CodingReport; status: RunStatus; rows: CodingRow[]; skipped: number; truncated: boolean; codingsPresent: boolean };

const count = (value: number | null) => value == null ? "—" : String(value);
const decimal = (value: number | null, digits = 2) => value == null ? "—" : value.toFixed(digits);
const percent = (value: number | null, digits = 0) => value == null ? "—" : `${(value * 100).toFixed(digits)}%`;

// --- exports ---------------------------------------------------------------
// The aggregate matrix is coding-report.json's; the reader's rows are
// codings.jsonl's. Both are entirely stored — the viewer's only contribution
// to either export is the shape (one row per field × condition, one row per
// coding), and the matrix EXPORTS LONG so the columns do not change with the
// run's condition list.

type MatrixExportRow = { field: CodingFieldSpec; condition: CodingConditionRow };

const matrixColumns: ExportColumn<MatrixExportRow>[] = [
  { header: "field", kind: "stored", value: ({ field }) => field.name },
  { header: "fieldType", kind: "stored", value: ({ field }) => field.type },
  { header: "optional", kind: "stored", value: ({ field }) => field.optional },
  { header: "condition", kind: "stored", value: ({ condition }) => condition.condition },
  { header: "codedResponses", kind: "stored", value: ({ condition }) => condition.codedResponses },
  { header: "codings", kind: "stored", value: ({ condition }) => condition.codings },
  { header: "meanWordCount", kind: "stored", value: ({ condition }) => condition.meanWordCount },
  { header: "n", kind: "stored", value: ({ field, condition }) => condition.fields[field.name]?.n ?? null },
  { header: "nulls", kind: "stored", value: ({ field, condition }) => condition.fields[field.name]?.nulls ?? null },
  { header: "trueCount", kind: "stored", value: ({ field, condition }) => condition.fields[field.name]?.trueCount ?? null },
  { header: "trueShare", kind: "stored", value: ({ field, condition }) => condition.fields[field.name]?.trueShare ?? null },
  { header: "mean", kind: "stored", value: ({ field, condition }) => condition.fields[field.name]?.mean ?? null },
  // A JSON blob, because an enum's category counts have no fixed columns.
  { header: "counts", kind: "stored", value: ({ field, condition }) => { const counts = condition.fields[field.name]?.counts; return counts ? JSON.stringify(counts) : null; } },
];

const codingRowColumns = (fields: CodingFieldSpec[]): ExportColumn<CodingRow>[] => [
  { header: "condition", kind: "stored", value: (row) => row.condition },
  { header: "promptID", kind: "stored", value: (row) => row.promptID },
  { header: "sampleIndex", kind: "stored", value: (row) => row.sampleIndex },
  { header: "coder", kind: "stored", value: (row) => row.judge },
  { header: "coderModel", kind: "stored", value: (row) => row.judgeModel },
  { header: "wordCount", kind: "stored", value: (row) => row.wordCount },
  { header: "seed", kind: "stored", value: (row) => row.seed },
  // One column per declared field. A field the row never carried is BLANK
  // (not coded); a field coded null exports the literal "null" — the two
  // are different facts and stay different in the file.
  ...fields.map((field): ExportColumn<CodingRow> => ({
    header: field.name,
    kind: "stored",
    value: (row) => field.name in row.codes ? formatCode(row.codes[field.name]) : null,
  })),
  { header: "briefReason", kind: "stored", value: (row) => row.briefReason },
];

export function CodingView({ run, workspaceRuns, onActivateRun, onNavigate, onOpenFile }: {
  run: WorkspaceRun | null;
  workspaceRuns: WorkspaceRun[];
  onActivateRun: (run: WorkspaceRun) => void;
  onNavigate: (view: View) => void;
  onOpenFile: (file: RunFile) => void;
}) {
  const [data, setData] = useState<CodingData | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [sources, setSources] = useState<{ responses: Map<string, SourceResponse>; truncated: boolean; present: boolean } | null>(null);
  const [sourceLoading, setSourceLoading] = useState(false);
  const [conditionFilter, setConditionFilter] = useState("All conditions");
  const [judgeFilter, setJudgeFilter] = useState("All coders");
  const [fieldFilter, setFieldFilter] = useState("Any field");
  const [valueFilter, setValueFilter] = useState("Any value");
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [disagreementKey, setDisagreementKey] = useState("");

  useEffect(() => {
    let cancelled = false;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- clears the previous run's artifacts before this effect re-reads them from disk; the alternative (remount via key) would discard the pane's scroll and re-run every load
    setData(null); setSources(null); setLoadError(""); setSelectedIndex(0); setDisagreementKey("");
    setConditionFilter("All conditions"); setJudgeFilter("All coders"); setFieldFilter("Any field"); setValueFilter("Any value"); setQuery("");
    if (!run) return;
    setLoading(true);
    void (async () => {
      try {
        const [reportJSON, statusJSON, codings] = await Promise.all([
          readJSONArtifact(run, "coding-report.json"),
          readJSONArtifact(run, "run-status.json"),
          loadCodings(run),
        ]);
        if (cancelled) return;
        setData({ report: parseCodingReport(reportJSON), status: parseRunStatus(statusJSON), rows: codings.rows, skipped: codings.skipped, truncated: codings.truncated, codingsPresent: codings.present });
      } catch (error) {
        if (!cancelled) setLoadError(error instanceof Error ? error.message : "The coding artifacts could not be read from this run.");
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [run?.key]);

  const rows = useMemo(() => data?.rows ?? [], [data]);
  const report = data?.report;
  const fields = useMemo<CodingFieldSpec[]>(() => report?.fields.length ? report.fields : fieldsFromRows(rows), [report, rows]);
  const sourceRunName = report?.sourceRun || data?.status.sourceRun || "";
  const sourceRun = useMemo(() => findSourceRun(workspaceRuns, sourceRunName), [workspaceRuns, sourceRunName]);
  const conditions = useMemo(() => [...new Set(rows.map((row) => row.condition))].sort((left, right) => left === "baseline" ? -1 : right === "baseline" ? 1 : left.localeCompare(right)), [rows]);
  const coders = useMemo(() => [...new Set(rows.map((row) => row.judge))].sort(), [rows]);
  const fieldValues = useMemo(() => fieldFilter === "Any field" ? [] : [...new Set(rows.filter((row) => fieldFilter in row.codes).map((row) => formatCode(row.codes[fieldFilter])))].sort(), [rows, fieldFilter]);
  const filtered = useMemo(() => rows.filter((row) => {
    if (conditionFilter !== "All conditions" && row.condition !== conditionFilter) return false;
    if (judgeFilter !== "All coders" && row.judge !== judgeFilter) return false;
    // A row that never carried the field is "not coded", which is not the
    // same fact as a coded null — it never matches a value filter.
    if (fieldFilter !== "Any field" && valueFilter !== "Any value" && (!(fieldFilter in row.codes) || formatCode(row.codes[fieldFilter]) !== valueFilter)) return false;
    if (!query.trim()) return true;
    const haystack = `${row.promptID} ${row.condition} ${row.judge} ${row.briefReason} ${Object.entries(row.codes).map(([field, value]) => `${field} ${formatCode(value)}`).join(" ")}`.toLowerCase();
    return haystack.includes(query.trim().toLowerCase());
  }), [rows, conditionFilter, judgeFilter, fieldFilter, valueFilter, query]);
  const selected = filtered[Math.min(selectedIndex, Math.max(0, filtered.length - 1))] ?? null;
  const disagreements = useMemo(() => codingDisagreements(rows, fields), [rows, fields]);
  const selectedDisagreement = disagreements.find((item) => item.key === disagreementKey) ?? disagreements[0] ?? null;
  const wordCounts = useMemo(() => wordCountProfiles(rows), [rows]);

  // The coded text lives in the SOURCE run; read it once, lazily.
  useEffect(() => {
    if (!sourceRun || sources || sourceLoading || !rows.length) return;
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
  }, [sourceRun?.key, rows.length]);

  // A `?record=` permalink names one coding by (condition, promptID,
  // sampleIndex, coder) — the coder included, because one response carries
  // one row per coder. Filters are cleared so the named row is reachable
  // whatever was selected before; a key matching nothing changes nothing.
  useEffect(() => {
    if (!rows.length) return;
    const linked = takePendingRecord("coding");
    if (!linked) return;
    const [condition, promptID, sample, coder] = splitRecordKey(linked);
    const index = rows.findIndex((row) => row.condition === condition && row.promptID === promptID
      && String(row.sampleIndex) === sample && (!coder || row.judge === coder));
    if (index < 0) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- applies a one-shot permalink taken from the deep-link queue; takePendingRecord CONSUMES, so it must not run during render
    setConditionFilter("All conditions"); setJudgeFilter("All coders");
    setFieldFilter("Any field"); setValueFilter("Any value"); setQuery("");
    setSelectedIndex(index);
  }, [rows]);

  useEffect(() => {
    if (selected) updateDeepLink({ record: codingRecordKey(selected.condition, selected.promptID, selected.sampleIndex, selected.judge) });
  }, [selected]);

  if (!run) return <div className="view-enter inner-view"><header className="page-title"><div><span className="section-number">NO RUN SELECTED</span><h1>Response coding</h1><p>Select a coding evaluate run to read its field aggregates, coder agreement, and every coded response.</p></div></header><div className="card no-run-card"><span>⌂</span><h2>No local run selected</h2><p>Choose a workspace and a coding run in the sidebar. Nothing here is inferred.</p></div></div>;

  if (loading || !data) return <div className="view-enter inner-view"><header className="page-title"><div><span className="section-number">RESPONSE CODING</span><h1>Response coding</h1><p>Reading coding artifacts from {run.name}.</p></div></header><div className="card artifact-empty"><span>◴</span><p>{loadError || "Reading coding-report.json and codings.jsonl…"}</p></div></div>;

  if (!data.report.present && !data.codingsPresent) return <div className="view-enter inner-view"><header className="page-title"><div><span className="section-number">RESPONSE CODING</span><h1>Response coding</h1><p>Loaded from {run.name}, read-only.</p></div></header><div className="card artifact-empty"><span>∅</span><p>This run carries no per-response coding artifacts. A coding evaluate writes <code>coding-report.json</code> and <code>codings.jsonl</code>; neither is present here{data.report.error ? ` (${data.report.error})` : ""}, and nothing is substituted in their place.</p></div></div>;

  const codingReport = data.report;
  const reportFile = findFile(run.files, "coding-report.json");
  const codingsFile = findFile(run.files, "codings.jsonl");
  const failedFile = findFile(run.files, "FAILED.md");
  const failed = data.status.status === "failed";
  const conditionRows = codingReport.conditions;
  const coderCards = [...new Set([...codingReport.judgeDetails.map((detail) => detail.name), ...codingReport.judges, ...coders])].map((name) => {
    const detail = codingReport.judgeDetails.find((entry) => entry.name === name);
    const row = rows.find((entry) => entry.judge === name);
    return { name, kind: detail?.kind || row?.judgeKind || "", requested: detail?.requestedModel || row?.judgeModel || "", actual: detail?.actualModel || "", revision: detail?.revision || row?.judgeRevision || "", provider: row?.judgeProvider || "", coded: rows.some((entry) => entry.judge === name) };
  });

  const openSourceRun = (view: View | null) => { if (!sourceRun) return; onActivateRun(sourceRun); if (view) onNavigate(view); };

  return (
    <div className="view-enter inner-view coding-view">
      <header className="page-title">
        <div>
          <span className="section-number">{count(codingReport.codings ?? (rows.length || null))} CODINGS · {fields.length} FIELD{fields.length === 1 ? "" : "S"} · {coderCards.length} CODER{coderCards.length === 1 ? "" : "S"}{data.truncated ? " · BOUNDED PREVIEW" : ""}</span>
          <h1>Response coding</h1>
          <p>What each response contained, field by field — the blinded per-response instrument, with every coding readable beside the text it describes.</p>
        </div>
        <div className="title-actions">
          {codingsFile && <button className="secondary" onClick={() => onOpenFile(codingsFile)}>codings.jsonl</button>}
          {reportFile && <button className="primary" onClick={() => onOpenFile(reportFile)}>coding-report.json <span>→</span></button>}
        </div>
      </header>

      <ProvenanceLegend />

      {failed && <div className="card judged-alert alert-failed"><span>!</span><div><strong>This coding evaluate FAILED — it is a failure record, not a result.</strong><p>{data.status.error || "run-status.json stamps status: failed with no error message."}{data.status.errorType ? ` (${data.status.errorType})` : ""}</p><p>{count(data.status.itemsWritten)} {data.status.itemLabel}{data.status.itemsWritten === 1 ? "" : "s"} written before the failure.{data.status.pendingUnits.length ? ` Never ran / incomplete: ${data.status.pendingUnits.join(", ")}.` : ""} No coding report is written for a partial panel.</p>{failedFile && <button className="text-button" onClick={() => onOpenFile(failedFile)}>Read FAILED.md</button>}</div></div>}
      {codingReport.epochUnverified && <div className="card judged-alert alert-warn"><span>⚠</span><div><strong>Epoch unverified.</strong><p>The source run carried no experiment-hash stamp and was coded under allow-unverified-epoch (<code>epochUnverified</code>).</p></div></div>}
      {codingReport.measurementDrift && <div className="card judged-alert alert-warn"><span>⚠</span><div><strong>Measurement drift tolerated.</strong><p>Measurement-side fields differed from the source run&rsquo;s epoch: <code>{codingReport.measurementDrift}</code>.</p></div></div>}
      {codingReport.exclusions && <div className="card judged-alert alert-note"><span>i</span><div><strong>Declared exclusions applied before coding.</strong><p>{Object.entries(codingReport.exclusions).map(([key, value]) => `${key}: ${typeof value === "object" ? JSON.stringify(value) : String(value)}`).join(" · ")}</p></div></div>}
      {codingReport.present && codingReport.mode !== "perResponseCoding" && <div className="card judged-alert alert-warn"><span>⚠</span><div><strong>Unexpected report mode.</strong><p>coding-report.json stamps <code>mode: {codingReport.mode || "(absent)"}</code>; this reader renders the per-response coding contract.</p></div></div>}

      <section className="card judged-pins">
        <header className="section-header"><div><span className="section-number">MEASUREMENT PINS</span><h2>What coded what</h2></div>{codingReport.mode && <Badge tone="blue">{codingReport.mode}</Badge>}</header>
        <dl className="pin-grid">
          <div><dt>Rubric file</dt><dd>{codingReport.rubricFile || "Not stamped"}</dd></div>
          <div><dt>Rubric hash</dt><dd><code>{shortHash(codingReport.rubricHash)}</code></dd></div>
          <div><dt>Experiment</dt><dd>{codingReport.experiment || run.experiment}</dd></div>
          <div><dt>Experiment hash</dt><dd><code>{shortHash(codingReport.experimentHash)}</code></dd></div>
          <div><dt>Evaluation source</dt><dd>{codingReport.evaluationSource || "Not stamped"}</dd></div>
          <div><dt>Codings</dt><dd>{count(codingReport.codings)}</dd></div>
        </dl>
        <div className="source-run-row">
          <div><span>Source run</span><strong>{sourceRunName || "Not stamped"}</strong></div>
          {sourceRun
            ? <div className="source-actions"><button className="secondary" onClick={() => openSourceRun(null)}>Activate run</button><button className="secondary" onClick={() => openSourceRun("generations")}>Open its generations</button></div>
            : <p className="muted">{sourceRunName ? "That run directory is not in the loaded workspace, so the coded responses cannot be joined here." : "This coding run stamped no source run."}</p>}
        </div>
        <div className="judge-cards">
          {coderCards.map((coder) => <article key={coder.name} className="judge-card">
            <header><strong>{coder.name}</strong><Badge tone={coder.coded ? "blue" : "warn"}>{coder.coded ? coder.kind || "coder" : "no rows"}</Badge></header>
            <dl>
              <div><dt>Requested model</dt><dd>{coder.requested || "Not stamped"}</dd></div>
              {coder.actual && coder.actual !== coder.requested && <div><dt>Actual model</dt><dd>{coder.actual}</dd></div>}
              {coder.provider && <div><dt>Provider</dt><dd>{coder.provider}</dd></div>}
              {coder.revision && <div><dt>Revision</dt><dd><code>{shortHash(coder.revision)}</code></dd></div>}
            </dl>
          </article>)}
          {!coderCards.length && <p className="muted">No coder panel is stamped in this run&rsquo;s artifacts.</p>}
        </div>
        {fields.length > 0 && <div className="field-schema">
          <span className="section-number">DECLARED SCHEMA</span>
          <ul>{fields.map((field) => <li key={field.name}><strong>{field.name}</strong><span>{field.type || "type not stamped"}{field.optional ? " · optional" : ""}{field.values.length ? ` · ${field.values.join(" | ")}` : ""}</span></li>)}</ul>
        </div>}
      </section>

      <section className="card">
        <header className="section-header">
          <div><span className="section-number">FIELD AGGREGATES</span><h2>Every field, condition by condition</h2></div>
          {conditionRows.length > 0 && fields.length > 0 && <ExportButton
            filename={csvFilename(run.name, "coding-aggregates")}
            columns={matrixColumns}
            rows={fields.flatMap((field) => conditionRows.map((condition) => ({ field, condition })))}
          />}
        </header>
        {conditionRows.length && fields.length ? <div className="jv-table-scroll"><table className="jv-table coding-matrix">
          <thead><tr><th>Field</th>{conditionRows.map((condition) => <th key={condition.condition} className={condition.condition === "baseline" ? "anchor-column" : ""}>{condition.condition}{condition.condition === "baseline" ? " · anchor" : ""}<br /><small>{count(condition.codedResponses)} responses · {count(condition.codings)} codings</small></th>)}</tr></thead>
          <tbody>
            {fields.map((field) => <tr key={field.name}>
              <th scope="row"><strong>{field.name}</strong><small>{field.type || "type not stamped"}{field.optional ? " · optional" : ""}</small></th>
              {conditionRows.map((condition) => {
                const aggregate = condition.fields[field.name];
                return <td key={condition.condition} className={condition.condition === "baseline" ? "anchor-column" : ""}>
                  {!aggregate ? <span className="muted">—</span>
                    : field.type === "boolean" || aggregate.trueShare != null || aggregate.trueCount != null
                      ? <><strong>{percent(aggregate.trueShare, 1)}</strong><small>{count(aggregate.trueCount)}/{count(aggregate.n)} true{aggregate.nulls ? ` · ${aggregate.nulls} null` : ""}</small></>
                      : aggregate.mean != null || field.type === "integer" || field.type === "number"
                        ? <><strong>{decimal(aggregate.mean)}</strong><small>mean · n {count(aggregate.n)}{aggregate.nulls ? ` · ${aggregate.nulls} null` : ""}</small></>
                        : aggregate.counts
                          ? <><span className="count-list">{Object.entries(aggregate.counts).map(([label, value]) => <em key={label}>{label} <b>{value}</b></em>)}</span><small>n {count(aggregate.n)}{aggregate.nulls ? ` · ${aggregate.nulls} null` : ""}</small></>
                          : <><strong>—</strong><small>n {count(aggregate.n)}{aggregate.nulls ? ` · ${aggregate.nulls} null` : ""}</small></>}
                </td>;
              })}
            </tr>)}
            <tr className="wordcount-row">
              <th scope="row"><strong>Mean word count</strong><small>engine-computed</small></th>
              {conditionRows.map((condition) => <td key={condition.condition} className={condition.condition === "baseline" ? "anchor-column" : ""}><strong>{decimal(condition.meanWordCount, 1)}</strong><small>words</small></td>)}
            </tr>
          </tbody>
        </table></div> : <div className="artifact-empty"><span>∅</span><p>coding-report.json carried no <code>conditions</code> aggregate block{fields.length ? "" : " and no field schema"}. Aggregates are engine output; the viewer does not recompute them from the rows.</p></div>}
        <footer className="table-note"><strong>Stored:</strong> every number in this table is read from coding-report.json — <code>trueShare</code>/<code>trueCount</code> for booleans, <code>mean</code> for numerics, <code>counts</code> for enums and strings, with nulls reported and never imputed.</footer>
      </section>

      <section className="card">
        <header className="section-header"><div><span className="section-number">INTER-CODER AGREEMENT</span><h2>Field by field, coder pair by coder pair</h2></div></header>
        {codingReport.fieldAgreement.length ? <table className="jv-table">
          <thead><tr><th>Field</th><th>Coder pair</th><th>n</th><th>Percent agreement</th><th>Cohen&rsquo;s κ</th><th>Mean abs. difference</th></tr></thead>
          <tbody>{codingReport.fieldAgreement.map((entry, index) => <tr key={`${entry.field}-${entry.judgeA}-${entry.judgeB}-${index}`}>
            <th scope="row">{entry.field}</th>
            <td className="pair-cell">{entry.judgeA} ↔ {entry.judgeB}</td>
            <td>{count(entry.n)}</td>
            <td>{entry.percentAgreement == null ? "n/a" : percent(entry.percentAgreement, 1)}</td>
            <td>{entry.kappa == null ? (entry.percentAgreement == null ? "n/a" : "undefined") : entry.kappa.toFixed(3)}</td>
            <td>{entry.meanAbsoluteDifference == null ? "n/a" : entry.meanAbsoluteDifference.toFixed(3)}</td>
          </tr>)}</tbody>
        </table> : <div className="artifact-empty"><span>∅</span><p>No <code>fieldAgreement</code> block in coding-report.json (a single-coder panel writes none). Percent agreement, κ and mean absolute difference are engine statistics; the viewer never computes them.</p></div>}
        <footer className="table-note">Categorical fields report percent agreement + κ; numeric fields report mean absolute difference — κ over continuous codes would be meaningless precision.</footer>
      </section>

      <section className="card coding-reader-card">
        <header className="section-header">
          <div><span className="section-number">CODED RESPONSES</span><h2>Every coding, beside the text it describes</h2><p>{filtered.length} of {rows.length} coding rows{data.truncated ? " (first 32 MB of codings.jsonl)" : ""}{data.skipped ? ` · ${data.skipped} unparseable line${data.skipped === 1 ? "" : "s"} skipped` : ""}.</p></div>
          <ExportButton filename={csvFilename(run.name, "codings")} columns={codingRowColumns(fields)} rows={filtered} label="Export filtered rows" />
        </header>
        {!rows.length ? <div className="artifact-empty"><span>∅</span><p>{data.codingsPresent ? "codings.jsonl is present but carried no readable rows." : "This run has no codings.jsonl."} Nothing is substituted in its place.</p></div>
          : <div className="reader-shell">
            <aside className="record-list">
              <div className="reader-filters">
                <label className="search"><span>⌕</span><input value={query} onChange={(event) => { setQuery(event.target.value); setSelectedIndex(0); }} placeholder="Search codes, reasons, prompt ids" aria-label="Search codings" /></label>
                <select value={conditionFilter} onChange={(event) => { setConditionFilter(event.target.value); setSelectedIndex(0); }} aria-label="Filter by condition"><option>All conditions</option>{conditions.map((name) => <option key={name}>{name}</option>)}</select>
                <select value={judgeFilter} onChange={(event) => { setJudgeFilter(event.target.value); setSelectedIndex(0); }} aria-label="Filter by coder"><option>All coders</option>{coders.map((name) => <option key={name}>{name}</option>)}</select>
                <div className="value-filter">
                  <select value={fieldFilter} onChange={(event) => { setFieldFilter(event.target.value); setValueFilter("Any value"); setSelectedIndex(0); }} aria-label="Filter by field"><option>Any field</option>{fields.map((field) => <option key={field.name}>{field.name}</option>)}</select>
                  <select value={valueFilter} disabled={fieldFilter === "Any field"} onChange={(event) => { setValueFilter(event.target.value); setSelectedIndex(0); }} aria-label="Filter by field value"><option>Any value</option>{fieldValues.map((value) => <option key={value}>{value}</option>)}</select>
                </div>
              </div>
              <div className="record-count"><span>{filtered.length} coding rows</span><span>{data.truncated ? "First 32 MB" : "Complete file"}</span></div>
              <div className="records">
                {filtered.slice(0, 500).map((row, index) => <button key={`${row.judge}-${row.condition}-${row.promptID}-${row.sampleIndex}-${index}`} onClick={() => setSelectedIndex(index)} className={selected === row ? "selected" : ""}>
                  <div><strong>{row.promptID}</strong><Badge tone={row.condition === "baseline" ? "neutral" : "blue"}>{row.condition}</Badge></div>
                  <p>{Object.entries(row.codes).map(([field, value]) => `${field} = ${formatCode(value)}`).join(" · ") || "No codes recorded."}</p>
                  <footer><span>{row.judge}</span><span>{row.wordCount == null ? "words not stamped" : `${row.wordCount} words`}</span></footer>
                </button>)}
                {!filtered.length && <div className="empty-state">No coding rows match those filters.</div>}
              </div>
              {filtered.length > 500 && <div className="record-pagination"><span>Showing the first 500 of {filtered.length}</span></div>}
            </aside>
            {selected && <article className="record-detail">
              <header><div><span className="section-number">{selected.condition.toUpperCase()} · SAMPLE {selected.sampleIndex}</span><h2>{selected.promptID}</h2><CopyLinkButton view="coding" record={codingRecordKey(selected.condition, selected.promptID, selected.sampleIndex, selected.judge)} label="Copy link to this coding" /></div></header>
              <div className="record-meta">
                <Badge tone="blue">{selected.condition}</Badge>
                <span>Coder {selected.judge}</span>
                <span>{selected.judgeModel || selected.judgeKind || "model not stamped"}</span>
                <span>Seed {selected.seed || "not stamped"}</span>
                <span>{selected.wordCount == null ? "Word count not stamped" : `${selected.wordCount} words`}</span>
              </div>
              <section className="codes-block">
                <span className="section-number">CODES</span>
                <div className="code-chips">
                  {fields.map((field) => field.name in selected.codes ? <span key={field.name} className={`chip chip-${selected.codes[field.name] === null ? "null" : typeof selected.codes[field.name]}`}><b>{field.name}</b>{formatCode(selected.codes[field.name])}</span> : <span key={field.name} className="chip chip-missing"><b>{field.name}</b>not coded</span>)}
                </div>
                <p className="brief-reason">{selected.briefReason || "No brief reason recorded."}</p>
              </section>
              <section className="text-block output-block">
                <span>CODED RESPONSE</span>
                {(() => {
                  const source = sources?.responses.get(cellKey(selected.condition, selected.promptID, selected.sampleIndex)) ?? null;
                  if (!sourceRun) return <p className="muted">The source run <code>{sourceRunName || "(not stamped)"}</code> is not in the loaded workspace, so the coded text cannot be shown here.</p>;
                  if (sourceLoading) return <p className="muted">Reading {sourceRunName}/generations.jsonl…</p>;
                  if (!source) return <p className="muted">No generation record in {sourceRunName} matches ({selected.condition}, {selected.promptID}, sample {selected.sampleIndex}){sources?.truncated ? " — it may lie beyond the bounded 32 MB read" : ""}.</p>;
                  return <p>{source.output}</p>;
                })()}
              </section>
              <footer className="record-path">codings.jsonl · complete row{sourceRun ? ` · response joined from ${sourceRunName}/generations.jsonl` : ""}</footer>
            </article>}
          </div>}
      </section>

      <section className="card">
        <header className="section-header"><div><span className="section-number">CODER SPLITS <DerivedBadge formula="cells where two coders recorded different values for the same field of the same (condition, promptID, sampleIndex); selection computed in the viewer from codings.jsonl" /></span><h2>Where the coders differed</h2><p>{disagreements.length} field-level split{disagreements.length === 1 ? "" : "s"} across {rows.length} coding rows.</p></div></header>
        {!disagreements.length ? <div className="artifact-empty"><span>{coders.length > 1 ? "✓" : "∅"}</span><p>{coders.length > 1 ? "The coders recorded identical values on every field of every response they both coded." : "Only one coder is present in codings.jsonl, so there is nothing to compare."}</p></div>
          : <div className="reader-shell">
            <aside className="record-list">
              <div className="record-count"><span>{disagreements.length} split{disagreements.length === 1 ? "" : "s"}</span><span>by field</span></div>
              <div className="records">
                {disagreements.slice(0, 500).map((item) => <button key={item.key} onClick={() => setDisagreementKey(item.key)} className={selectedDisagreement?.key === item.key ? "selected" : ""}>
                  <div><strong>{item.field}</strong><Badge tone={item.condition === "baseline" ? "neutral" : "blue"}>{item.condition}</Badge></div>
                  <p>{item.codings.map((coding) => `${coding.judge} = ${formatCode(coding.value)}`).join(" · ")}</p>
                  <footer><span>{item.promptID}</span><span>sample {item.sampleIndex}</span></footer>
                </button>)}
              </div>
            </aside>
            {selectedDisagreement && <article className="record-detail">
              <header><div><span className="section-number">{selectedDisagreement.condition.toUpperCase()} · SAMPLE {selectedDisagreement.sampleIndex}</span><h2>{selectedDisagreement.field} — {selectedDisagreement.promptID}</h2></div></header>
              <div className="split-codings">
                {selectedDisagreement.codings.map((coding) => <div key={coding.judge} className="split-coding">
                  <header><strong>{coding.judge}</strong><span className={`chip chip-${coding.value === null ? "null" : typeof coding.value}`}>{formatCode(coding.value)}</span></header>
                  <p>{coding.briefReason || "No brief reason recorded."}</p>
                </div>)}
              </div>
              <section className="text-block output-block">
                <span>CODED RESPONSE</span>
                {(() => {
                  const source = sources?.responses.get(cellKey(selectedDisagreement.condition, selectedDisagreement.promptID, selectedDisagreement.sampleIndex)) ?? null;
                  if (!sourceRun) return <p className="muted">The source run <code>{sourceRunName || "(not stamped)"}</code> is not in the loaded workspace.</p>;
                  if (sourceLoading) return <p className="muted">Reading {sourceRunName}/generations.jsonl…</p>;
                  if (!source) return <p className="muted">No matching generation record in {sourceRunName}.</p>;
                  return <p>{source.output}</p>;
                })()}
              </section>
              <footer className="record-path">codings.jsonl · {selectedDisagreement.codings.length} coder rows for this cell</footer>
            </article>}
          </div>}
      </section>

      <section className="card">
        <header className="section-header"><div><span className="section-number">RESPONSE LENGTH</span><h2>Word count by condition</h2></div><DerivedBadge formula="distribution counted in the viewer over DISTINCT responses (deduplicated across coders) from codings.jsonl wordCount; the per-condition mean beside it is the engine's" /></header>
        {wordCounts.profiles.length ? <div className="wordcount-grid">
          {wordCounts.profiles.map((profile) => {
            const stored = conditionRows.find((condition) => condition.condition === profile.condition);
            const peak = Math.max(1, ...profile.bins.map((bin) => bin.count));
            return <article key={profile.condition}>
              <header>
                <div><strong>{profile.condition}</strong><span>{profile.responses} distinct responses</span></div>
                <div className="wordcount-means">
                  <div><span>Mean (report)</span><strong>{decimal(stored?.meanWordCount ?? null, 1)}</strong></div>
                  <div><span>Range <DerivedBadge formula="min and max wordCount over distinct responses in this condition" /></span><strong>{profile.min}–{profile.max}</strong></div>
                </div>
              </header>
              <div className="histogram">{profile.bins.map((bin) => <div key={bin.from} title={`${bin.from}–${bin.to} words: ${bin.count}`}><i style={{ height: `${(bin.count / peak) * 100}%` }} /><span>{bin.from}</span></div>)}</div>
            </article>;
          })}
        </div> : <div className="artifact-empty"><span>∅</span><p>No stamped word counts in codings.jsonl{wordCounts.missing ? ` (${wordCounts.missing} row${wordCounts.missing === 1 ? "" : "s"} without one)` : ""}. The engine computes wordCount at coding time; the viewer never counts words itself.</p></div>}
        {wordCounts.profiles.length > 0 && wordCounts.missing > 0 && <p className="table-note">{wordCounts.missing} coding row{wordCounts.missing === 1 ? "" : "s"} carried no wordCount and {wordCounts.missing === 1 ? "is" : "are"} excluded from these distributions.</p>}
      </section>
    </div>
  );
}
