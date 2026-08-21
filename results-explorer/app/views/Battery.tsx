"use client";

// The capability-battery view (upgrade plan Phase 5: port the native-only
// sections). A study run scores the manifest's pinned battery under EVERY
// condition — the control that says whether a steered arm is still able to
// do simple things, and therefore whether its behavioural shift is a stance
// change or a competence loss.
//
// The per-condition accuracy is the ENGINE's, read from report.json's
// `conditions[…].capabilityBattery`. The viewer's own counts over
// battery.jsonl are shown beside it, badged, and any disagreement between
// the two is surfaced rather than reconciled — a mismatch means the row
// file and the report describe different work.

import { useEffect, useMemo, useState } from "react";
import { DerivedBadge, ProvenanceLegend } from "../components/provenance";
import { Badge, ExportButton } from "../components/ui";
import {
  batteryCounts,
  loadBatteryRecords,
  readBatterySummaries,
  type BatteryLoad,
  type BatteryRecord,
} from "../lib/battery";
import type { ExportColumn } from "../lib/export";
import { csvFilename } from "../lib/export";
import { shortHash } from "../lib/format";
import type { RunFile, WorkspaceRun } from "../lib/types";
import "./judged.css";
import "./analysis.css";

const ALL_CONDITIONS = "All conditions";
type Grade = "all" | "correct" | "incorrect" | "ungraded";

const percent = (value: number | null) => value == null ? "—" : `${(value * 100).toFixed(1)}%`;

const columns: ExportColumn<BatteryRecord>[] = [
  { header: "condition", kind: "stored", value: (row) => row.condition },
  { header: "promptID", kind: "stored", value: (row) => row.promptID },
  { header: "promptIndex", kind: "stored", value: (row) => row.promptIndex },
  { header: "sampleIndex", kind: "stored", value: (row) => row.sampleIndex },
  { header: "prompt", kind: "stored", value: (row) => row.prompt },
  { header: "expected", kind: "stored", value: (row) => row.expected },
  { header: "output", kind: "stored", value: (row) => row.output },
  { header: "correct", kind: "stored", value: (row) => row.correct },
  { header: "batteryHash", kind: "stored", value: (row) => row.batteryHash },
  { header: "battery.jsonl line", kind: "stored", value: (row) => row.line },
];

export function BatteryView({ run, onOpenFile }: { run: WorkspaceRun | null; onOpenFile: (file: RunFile) => void }) {
  const [load, setLoad] = useState<BatteryLoad | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [condition, setCondition] = useState(ALL_CONDITIONS);
  const [grade, setGrade] = useState<Grade>("all");
  const [selectedLine, setSelectedLine] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setLoad(null); setError(""); setCondition(ALL_CONDITIONS); setGrade("all"); setSelectedLine(0);
    if (!run) { setLoading(false); return () => { cancelled = true; }; }
    setLoading(true);
    loadBatteryRecords(run)
      .then((result) => { if (!cancelled) { setLoad(result); setLoading(false); } })
      .catch((cause: unknown) => { if (!cancelled) { setError(cause instanceof Error ? cause.message : String(cause)); setLoading(false); } });
    return () => { cancelled = true; };
  }, [run?.key]);

  const records = useMemo(() => load?.records ?? [], [load]);
  const summaries = useMemo(() => readBatterySummaries(run?.report ?? {}), [run?.report]);
  const counts = useMemo(() => batteryCounts(records), [records]);
  const conditions = useMemo(
    () => [...new Set([...summaries.map((row) => row.condition), ...counts.map((row) => row.condition)])],
    [summaries, counts]);
  const filtered = useMemo(() => records.filter((row) =>
    (condition === ALL_CONDITIONS || row.condition === condition)
    && (grade === "all"
      || (grade === "correct" && row.correct === true)
      || (grade === "incorrect" && row.correct === false)
      || (grade === "ungraded" && row.correct === null))), [records, condition, grade]);
  const selected = filtered.find((row) => row.line === selectedLine) ?? filtered[0] ?? null;

  const header = (eyebrow: string, note: string) => (
    <header className="page-title">
      <div><span className="section-number">{eyebrow}</span><h1>Capability battery</h1><p>{note}</p></div>
      <div className="title-actions">
        {load?.file && <button className="secondary" onClick={() => load.file && onOpenFile(load.file)}>battery.jsonl</button>}
        {filtered.length > 0 && <ExportButton filename={csvFilename(run?.name, "battery-items")} columns={columns} rows={filtered} />}
      </div>
    </header>
  );

  if (!run) return <div className="view-enter inner-view analysis-view">{header("NO RUN SELECTED", "Choose a run. The battery is read from that run's own battery.jsonl and report.json; nothing is inferred.")}<div className="card no-run-card"><span>⌂</span><h2>No local run selected</h2><p>Pick a run to read its capability-battery evidence.</p></div></div>;
  if (loading) return <div className="view-enter inner-view analysis-view">{header("READING", `Reading ${run.name}/battery.jsonl.`)}<div className="card no-run-card"><span>…</span><h2>Reading battery.jsonl</h2><p>Battery generations stream to their own file, never into generations.jsonl.</p></div></div>;
  if (error) return <div className="view-enter inner-view analysis-view">{header("READ FAILED", `battery.jsonl could not be read from ${run.name}.`)}<div className="card no-run-card"><span>!</span><h2>battery.jsonl could not be read</h2><p>{error}</p></div></div>;

  if (!summaries.length && !records.length) return <div className="view-enter inner-view analysis-view">
    {header("NO BATTERY EVIDENCE", `${run.name} carries no capability-battery artifacts.`)}
    <div className="card no-run-card"><span>∅</span><h2>This run scored no capability battery</h2><p>A run scores the battery only when its manifest pins one. There is no <code>battery.jsonl</code>{load?.present ? " with readable rows" : ""} and no <code>capabilityBattery</code> block in report.json — so the capability control was not run here, which is different from having been run and passed.</p></div>
  </div>;

  // The two sources are independent facts about the same work. Where a
  // condition's stored accuracy and the viewer's count of its graded rows
  // disagree, the run is describing itself two ways; say so.
  const mismatches = summaries.flatMap((summary) => {
    const counted = counts.find((row) => row.condition === summary.condition);
    if (!counted || summary.accuracy == null || counted.records === 0) return [];
    const graded = counted.correct + counted.incorrect;
    if (!graded) return [];
    const derived = counted.correct / graded;
    return Math.abs(derived - summary.accuracy) > 0.005 ? [{ condition: summary.condition, stored: summary.accuracy, derived }] : [];
  });
  const hashes = [...new Set([...summaries.map((row) => row.batteryHash), ...records.map((row) => row.batteryHash)].filter(Boolean))];

  return (
    <div className="view-enter inner-view analysis-view">
      {header(
        `${summaries.length || conditions.length} CONDITION${(summaries.length || conditions.length) === 1 ? "" : "S"} · ${records.length} BATTERY RECORD${records.length === 1 ? "" : "S"}${load?.truncated ? " · BOUNDED PREVIEW" : ""}`,
        `The pinned battery scored under every arm of ${run.name}. Battery items are capability CONTROLS, not study outputs — they never enter outcome analysis.`,
      )}
      <ProvenanceLegend />

      {(load?.truncated || (load?.skipped ?? 0) > 0 || hashes.length > 1) && <p className="analysis-note">
        {load?.truncated && <span>Bounded read: only the first 32 MB of battery.jsonl was parsed.</span>}
        {(load?.skipped ?? 0) > 0 && <span>{load?.skipped} line{load?.skipped === 1 ? "" : "s"} did not parse into a battery record and {load?.skipped === 1 ? "is" : "are"} excluded.</span>}
        {hashes.length > 1 && <span>This run&apos;s records name more than one battery hash ({hashes.map((hash) => shortHash(hash)).join(", ")}) — the arms were not all scored against the same battery.</span>}
      </p>}

      <section className="card">
        <header className="section-header">
          <div><span className="section-number">PER-CONDITION ACCURACY</span><h2>Did the intervention cost the model its competence?</h2><p>Accuracy is read from report.json. The counted split beside it is the viewer&apos;s own tally over battery.jsonl.</p></div>
          {hashes.length === 1 && <Badge tone="neutral">battery {shortHash(hashes[0])}</Badge>}
        </header>
        {summaries.length === 0 && <p className="analysis-note"><span>report.json carries no <code>capabilityBattery</code> block, so no stored accuracy is available — only the viewer&rsquo;s counts below.</span></p>}
        <div className="battery-strip">
          {conditions.map((name) => {
            const summary = summaries.find((row) => row.condition === name) ?? null;
            const counted = counts.find((row) => row.condition === name) ?? null;
            const graded = counted ? counted.correct + counted.incorrect : 0;
            return (
              <article className={`battery-cell ${name === "baseline" ? "is-baseline" : ""}`} key={name}>
                <header><strong title={name}>{name}</strong>{name === "baseline" && <Badge tone="neutral">anchor</Badge>}</header>
                <b>{percent(summary?.accuracy ?? null)}</b>
                <span className="battery-cell-source">report.json accuracy{summary?.itemCount == null ? "" : ` · ${summary.itemCount} items`}</span>
                <div className="battery-bar" role="img" aria-label={counted ? `${counted.correct} correct of ${graded} graded` : "no rows"}>
                  <i style={{ width: graded ? `${(counted!.correct / graded) * 100}%` : "0%" }} />
                </div>
                <dl>
                  <dt>Correct <DerivedBadge formula="battery.jsonl rows with correct: true, counted in the viewer" /></dt>
                  <dd>{counted ? counted.correct : "—"}</dd>
                  <dt>Incorrect</dt><dd>{counted ? counted.incorrect : "—"}</dd>
                  <dt>Ungraded</dt><dd>{counted ? counted.ungraded : "—"}</dd>
                  <dt>Rows</dt><dd>{counted ? counted.records : "—"}</dd>
                </dl>
              </article>
            );
          })}
        </div>
        {mismatches.length > 0 && <p className="analysis-warn">
          <strong>Stored accuracy and the counted rows disagree</strong> for {mismatches.map((row) => `${row.condition} (report ${percent(row.stored)}, rows ${percent(row.derived)})`).join("; ")}. Both readings are shown as they are; neither is corrected into the other.
        </p>}
        <footer className="table-note"><strong>Stored:</strong> the large percentage and item count are report.json&rsquo;s <code>capabilityBattery</code> block. The correct/incorrect/ungraded split is counted by the viewer from battery.jsonl — an ungraded row carries no <code>correct</code> field and is neither right nor wrong.</footer>
      </section>

      <section className="card">
        <header className="section-header">
          <div><span className="section-number">ITEM BROWSER</span><h2>Every battery item, as answered</h2><p>{filtered.length} of {records.length} record{records.length === 1 ? "" : "s"} in this cut.</p></div>
          <div className="analysis-filters">
            <select value={condition} onChange={(event) => { setCondition(event.target.value); setSelectedLine(0); }} aria-label="Filter battery items by condition">
              <option>{ALL_CONDITIONS}</option>{conditions.map((name) => <option key={name}>{name}</option>)}
            </select>
            <select value={grade} onChange={(event) => { setGrade(event.target.value as Grade); setSelectedLine(0); }} aria-label="Filter battery items by grade">
              <option value="all">Correct and incorrect</option>
              <option value="correct">Correct only</option>
              <option value="incorrect">Incorrect only</option>
              <option value="ungraded">Ungraded only</option>
            </select>
          </div>
        </header>
        {!records.length ? <div className="artifact-empty"><span>∅</span><p>{load?.present ? "battery.jsonl is present but carried no readable records." : "This run has no battery.jsonl — only the report's per-condition rollup above."} Nothing is substituted in its place.</p></div>
          : !filtered.length ? <div className="artifact-empty"><span>∅</span><p>No battery record matches those filters.</p></div>
          : <div className="reader-shell">
            <aside className="record-list">
              <div className="record-count"><span>{filtered.length} item{filtered.length === 1 ? "" : "s"}</span><span>{load?.truncated ? "First 32 MB" : "Complete file"}</span></div>
              <div className="records">
                {filtered.slice(0, 500).map((row) => (
                  <button key={row.line} onClick={() => setSelectedLine(row.line)} className={selected?.line === row.line ? "selected" : ""}>
                    <div><strong>{row.promptID || `line ${row.line}`}</strong><Badge tone={row.correct === true ? "good" : row.correct === false ? "warn" : "neutral"}>{row.correct === true ? "correct" : row.correct === false ? "incorrect" : "not graded"}</Badge></div>
                    <p>{row.prompt || "Prompt text not stored on this record."}</p>
                    <footer><span>{row.condition}</span><span>{row.output.trim().split(/\s+/)[0] || "—"}</span></footer>
                  </button>
                ))}
              </div>
              {filtered.length > 500 && <div className="record-pagination"><span>Showing the first 500 of {filtered.length}</span></div>}
            </aside>
            {selected && <article className="record-detail">
              <header>
                <div><span className="section-number">{selected.condition.toUpperCase()}</span><h2>{selected.promptID || `battery.jsonl line ${selected.line}`}</h2></div>
                <Badge tone={selected.correct === true ? "good" : selected.correct === false ? "warn" : "neutral"}>{selected.correct === true ? "correct" : selected.correct === false ? "incorrect" : "no grade stamped"}</Badge>
              </header>
              <div className="record-meta">
                <Badge tone="blue">{selected.condition}</Badge>
                <span>{selected.promptIndex == null ? "Item index not stamped" : `Item ${selected.promptIndex}`}</span>
                <span>Sample {selected.sampleIndex}</span>
                <span>{selected.batteryHash ? `battery ${shortHash(selected.batteryHash)}` : "battery hash not stamped"}</span>
              </div>
              <section className="text-block prompt-block"><span>BATTERY ITEM</span><p>{selected.prompt || "Prompt text was not stored on this record."}</p></section>
              <section className="text-block output-block"><span>MODEL OUTPUT</span><p>{selected.output}</p></section>
              <section className="derived-block">
                <header><span>GRADING</span><Badge tone={selected.correct === true ? "good" : selected.correct === false ? "warn" : "neutral"}>{selected.correct === null ? "Not graded" : "Engine-graded"}</Badge></header>
                <div><span>Expected answer</span><strong>{selected.expected || "Not stamped"}</strong></div>
                <div><span>Graded</span><strong>{selected.correct === null ? "This record carries no correct field" : selected.correct ? "Correct" : "Incorrect"}</strong></div>
              </section>
              <footer className="record-path">battery.jsonl · line {selected.line} · read locally</footer>
            </article>}
          </div>}
        <footer className="table-note">Grading is the engine&rsquo;s (<code>scoring.is_correct</code> / Swift&rsquo;s inferred grading mode) and is read verbatim. The viewer never re-grades an answer.</footer>
      </section>
    </div>
  );
}
