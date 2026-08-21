"use client";

import { useEffect, useMemo, useState } from "react";
import { DerivedBadge } from "../components/provenance";
import { Badge, CopyLinkButton, NoRunSelected } from "../components/ui";
import { responseRecordKey, splitRecordKey, takePendingRecord, updateDeepLink } from "../lib/deeplink";
import { demoPreviewEnabled, generations } from "../lib/demo";
import {
  instrumentRecordFor,
  loadInstrumentRecords,
  promptIDFromGenerationID,
  takeGenerationRecord,
} from "../lib/instruments";
import type { InstrumentRecord } from "../lib/instruments";
import type { Generation, WorkspaceRun } from "../lib/types";
// The instrument detail block and the juxtapose layout are styled with the
// choice view's sheet — same records, same vocabulary, two surfaces.
import "./choice.css";

const isPanelRow = (row: Generation) => Boolean(row.speakerName || row.turnTitle);

/// Panel runs stamp a UUID promptID and no caseID, so the loader's caseName
/// is an unreadable identifier. The legible label is already on the row:
/// replicate + turn title + speaker.
const rowTitle = (row: Generation) => isPanelRow(row) ? row.turnTitle || row.speakerName || row.caseName : row.caseName;
const rowSubtitle = (row: Generation) => isPanelRow(row)
  ? [row.replicateIndex != null ? `Replicate ${row.replicateIndex}` : null, row.speakerName].filter(Boolean).join(" · ") || row.id
  : row.id;
const rowEyebrow = (row: Generation) => isPanelRow(row)
  ? [row.replicateIndex != null ? `REPLICATE ${row.replicateIndex}` : null, (row.speakerName || "").toUpperCase()].filter(Boolean).join(" · ") || row.family.toUpperCase()
  : `${row.family.toUpperCase()} · ${row.id}`;

const signed = (value: number | null | undefined, digits = 2) => value == null ? "—" : `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;

/// The engines stamp `wordCount` and `distinct2` on a sampled generation.
/// When they are absent the loader still fills the fields (a word count it
/// made itself; a zero placeholder), so both facts must be said out loud
/// here rather than printed as though the run had reported them.
const distinct2Text = (row: Generation) => row.distinct2Stored ? row.distinct2.toFixed(2) : "not stamped";
const wordText = (row: Generation) => `${row.words} words${row.wordCountStored ? "" : " (counted here)"}`;

/// The permalink key for one generation record: the same (condition,
/// promptID, sampleIndex) triple the engines join on.
const generationKey = (row: Generation) => responseRecordKey(row.condition, promptIDFromGenerationID(row.id), row.sample);

const matchesRecordKey = (row: Generation, key: string) => {
  const [condition, promptID, sample] = splitRecordKey(key);
  return row.condition === condition
    && promptIDFromGenerationID(row.id) === promptID
    && (sample === undefined || sample === "" || String(row.sample) === sample);
};

/// Every field below is STORED on the instrument record — this block only
/// lays them out. (Before this, the reader collapsed the whole readout to
/// "Selected option: B".)
function InstrumentDetailBlock({ record }: { record: InstrumentRecord }) {
  const state = record.interventionState;
  return (
    <section className="instrument-detail">
      <header>
        <span>{record.instrument} · answer-token readout</span>
        <Badge tone={record.selected === record.target ? "good" : "warn"}>selected {record.selected} · target {record.target || "—"}</Badge>
      </header>
      <div className="instrument-option-scroll">
        <table className="instrument-option-table">
          <thead>
            <tr><th>Option</th><th>Log-odds</th><th>Choice p</th><th>Logprob</th><th>Mean token logprob</th><th>Tokens</th><th>Token ids</th></tr>
          </thead>
          <tbody>
            {record.options.map((option) => (
              <tr key={option} className={option === record.selected ? "is-selected" : ""}>
                <td>
                  {option}
                  <span className="option-flags">
                    {option === record.selected && <i>selected</i>}
                    {option === record.target && <i className="is-target">target</i>}
                  </span>
                </td>
                <td>{signed(record.logOdds[option] ?? null)}</td>
                <td>{record.choiceProbability[option] == null ? "—" : record.choiceProbability[option].toPrecision(4)}</td>
                <td>{record.optionLogprobs[option] == null ? "—" : record.optionLogprobs[option].toFixed(2)}</td>
                <td>{record.optionMeanTokenLogprobs[option] == null ? "—" : record.optionMeanTokenLogprobs[option].toFixed(2)}</td>
                <td>{record.optionTokenCounts[option] ?? "—"}{record.optionTokenLogprobs[option]?.length ? ` (${record.optionTokenLogprobs[option].map((value) => value.toFixed(1)).join(", ")})` : ""}</td>
                <td>{record.optionTokenIDs[option]?.join(", ") || "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="instrument-facts">
        <div><span>Margin</span><strong>{record.margin == null ? "—" : record.margin.toFixed(2)}</strong></div>
        <div><span>Option length ratio</span><strong>{record.optionLengthRatio == null ? "—" : record.optionLengthRatio.toFixed(2)}</strong></div>
        <div><span>Arm</span><strong>{record.arm || "—"}</strong></div>
        <div><span>Case</span><strong>{record.caseID || "—"}</strong></div>
        <div><span>Intervention</span><strong>{!state ? "not stamped" : state.slots.length ? state.slots.map((slot) => `${slot.concept} L${slot.layer ?? "—"} α ${slot.alpha ?? "—"}`).join(" + ") : "none (unsteered)"}</strong></div>
        <div><span>Variant / control</span><strong>{state?.variant || "—"}{state?.controlType ? ` · ${state.controlType}` : ""}</strong></div>
        <div><span>α units</span><strong>{state?.alphaInNormUnits == null ? "—" : state.alphaInNormUnits ? "residual-norm units" : "raw"}{state?.bandWidth != null ? ` · band ${state.bandWidth}` : ""}</strong></div>
        <div><span>Decoding</span><strong>{record.doSample === null ? "—" : record.doSample ? "sampled" : "deterministic"}{record.temperature != null ? ` · T ${record.temperature}` : ""}{record.topP != null ? ` · top-p ${record.topP}` : ""}{record.topK != null ? ` · top-k ${record.topK}` : ""}</strong></div>
        <div><span>Model</span><strong>{record.modelID || "not stamped"}{record.modelRevision ? ` @ ${record.modelRevision.slice(0, 10)}` : ""}</strong></div>
        <div><span>Record</span><strong>generations.jsonl line {record.line}</strong></div>
      </div>
    </section>
  );
}

export function GenerationsView({ run }: { run: WorkspaceRun | null }) {
  const [query, setQuery] = useState("");
  const [condition, setCondition] = useState("All conditions");
  const [page, setPage] = useState(0);
  const [compare, setCompare] = useState(false);
  // Keyed by run so a pending read can never paint the previous run's
  // readouts onto this one.
  const [instrument, setInstrument] = useState<{ key: string; records: InstrumentRecord[] }>({ key: "", records: [] });
  const records = run ? run.generationRows : generations;
  const [selected, setSelectedRecord] = useState(records[0] ?? generations[0]);
  const setSelected = (row: Generation) => { setSelectedRecord(row); setCompare(false); };
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- resets the filters, then applies the one-shot deep-link / choice-view hand-offs below; both takePendingRecord and takeGenerationRecord CONSUME, so they must not run during render
    setCondition("All conditions");
    setQuery("");
    setPage(0);
    setCompare(false);
    // Two one-shot hand-offs, in priority order: a `?record=` permalink the
    // shell parsed (lib/deeplink), then the choice view's in-app request
    // (lib/instruments). Neither invents a record: a key that matches
    // nothing leaves the reader on the first row and says nothing false.
    const linked = takePendingRecord("generations");
    const linkedRow = linked ? records.find((row) => matchesRecordKey(row, linked)) : null;
    const request = takeGenerationRecord();
    const requested = request ? records.find((row) => row.isInstrument && promptIDFromGenerationID(row.id) === request.promptID && row.condition === request.condition) : null;
    if (linkedRow) { setSelected(linkedRow); setQuery(promptIDFromGenerationID(linkedRow.id)); }
    else if (request && requested) { setSelected(requested); setQuery(request.promptID); }
    else setSelected(records[0] ?? generations[0]);
  }, [run?.key]);
  useEffect(() => {
    let cancelled = false;
    if (!run) return () => { cancelled = true; };
    // Answer-token readouts ride inside generations.jsonl; the shared
    // generation rows keep only the selected option, so the full readout is
    // read here (bounded, read-only).
    const key = run.key;
    loadInstrumentRecords(run).then((load) => { if (!cancelled) setInstrument({ key, records: load.records }); }).catch(() => { if (!cancelled) setInstrument({ key, records: [] }); });
    return () => { cancelled = true; };
  }, [run?.key]);
  // Keep the permalink in step with the selection (embedded only; no-op in
  // the browser build, where a session-scoped folder grant has no address).
  useEffect(() => {
    if (run && selected) updateDeepLink({ record: generationKey(selected) });
  }, [run?.key, selected]);
  const instrumentRecords = instrument.key === (run?.key ?? "") ? instrument.records : [];
  const conditions = useMemo(() => [...new Set(records.map((record) => record.condition))].sort(), [records]);
  const filtered = useMemo(() => records.filter((generation) => {
    const matchesQuery = `${generation.id} ${generation.caseName} ${generation.family} ${generation.turnTitle ?? ""} ${generation.speakerName ?? ""} ${generation.output}`.toLowerCase().includes(query.toLowerCase());
    return matchesQuery && (condition === "All conditions" || generation.condition === condition);
  }), [query, condition, records]);
  // Changing a filter returns to page 1. Done in the two handlers that can
  // change one rather than in an effect watching the values: the run-change
  // reset above already sets the page itself, so these are the only edges.
  const changeQuery = (value: string) => { setQuery(value); setPage(0); };
  const changeCondition = (value: string) => { setCondition(value); setPage(0); };
  const pageSize = 200;
  const pageCount = Math.max(1, Math.ceil(filtered.length / pageSize));
  const pageRecords = filtered.slice(page * pageSize, (page + 1) * pageSize);
  const selectedIndex = records.indexOf(selected);
  const moveSelected = (delta: number) => {
    const next = records[selectedIndex + delta];
    if (next) setSelected(next);
  };
  if (!run && !demoPreviewEnabled()) return <NoRunSelected title="Generations" />;
  if (run && run.generationRows.length === 0) {
    return (
      <div className="view-enter inner-view">
        <header className="page-title"><div><span className="section-number">GENERATIONS</span><h1>Generations</h1><p>Loaded from {run.name}, read-only.</p></div></header>
        <div className="card no-run-card"><span>∅</span><h2>No readable generation records</h2><p>This run&apos;s generations.jsonl is absent, empty, or carried no parseable records{run.skippedGenerationLines ? ` (${run.skippedGenerationLines} unparseable line${run.skippedGenerationLines === 1 ? "" : "s"} skipped)` : ""}. Nothing is substituted in its place.</p></div>
      </div>
    );
  }
  // Prefer a partner of the same kind (instrument ↔ instrument), and among
  // those prefer the baseline condition — the comparison that matters.
  const partnerPool = records.filter((record) => record.id === selected.id && record.condition !== selected.condition);
  const sameKind = partnerPool.filter((record) => record.isInstrument === selected.isInstrument);
  const pool = sameKind.length ? sameKind : partnerPool;
  const pairedRecord = pool.find((record) => record.condition.toLowerCase() === "baseline") ?? pool[0] ?? null;
  const recordFor = (row: Generation) => row.isInstrument ? instrumentRecordFor(instrumentRecords, promptIDFromGenerationID(row.id), row.condition) : null;
  const selectedInstrument = recordFor(selected);
  const downloadGenerations = async () => {
    if (!run?.generationFile) return;
    const file = await run.generationFile.getFile();
    const url = URL.createObjectURL(file);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "generations.jsonl";
    anchor.click();
    URL.revokeObjectURL(url);
  };
  return (
    <div className="view-enter inner-view generation-view">
      <header className="page-title compact-title">
        <div><span className="section-number">{run ? `${records.length} LOADED RECORDS${run.previewTruncated ? " · BOUNDED PREVIEW" : ""}${instrumentRecords.length ? ` · ${instrumentRecords.length} INSTRUMENT READOUTS` : ""}` : "384 RECORDS · 0 DECODE ERRORS"}</span><h1>Generation reader</h1><p>Inspect outputs, paired conditions, parser results, and record-level provenance without leaving the study.</p></div>
        <button className="primary" onClick={downloadGenerations} disabled={!run?.generationFile}>{run ? "Download JSONL" : "Preview only"} <span>↓</span></button>
      </header>
      <div className="reader-shell">
        <aside className="record-list">
          <div className="reader-filters">
            <label className="search"><span>⌕</span><input value={query} onChange={(event) => changeQuery(event.target.value)} placeholder="Search cases, speakers, or output" aria-label="Search generations" /></label>
            <select value={condition} onChange={(event) => changeCondition(event.target.value)} aria-label="Filter by condition"><option>All conditions</option>{conditions.map((name) => <option key={name}>{name}</option>)}</select>
          </div>
          <div className="record-count"><span>{filtered.length} {run ? "local" : "preview"} records</span><span>{run?.previewTruncated ? "First 32 MB" : "Complete preview"}</span></div>
          <div className="records">
            {pageRecords.map((generation, index) => (
              <button key={`${generation.id}-${generation.condition}-${index}`} onClick={() => setSelected(generation)} className={selected === generation ? "selected" : ""}>
                <div><strong>{rowTitle(generation)}</strong><Badge tone={generation.condition === "baseline" ? "neutral" : generation.condition.includes("random") ? "warn" : "blue"}>{generation.condition}</Badge></div>
                <p>{generation.output}</p>
                <footer><span>{rowSubtitle(generation)}</span><span>{generation.decision}</span></footer>
              </button>
            ))}
            {filtered.length === 0 && <div className="empty-state">No preview records match those filters.</div>}
          </div>
          {filtered.length > pageSize && <div className="record-pagination"><button onClick={() => setPage((value) => Math.max(0, value - 1))} disabled={page === 0}>← Previous</button><span>Page {page + 1} of {pageCount}</span><button onClick={() => setPage((value) => Math.min(pageCount - 1, value + 1))} disabled={page >= pageCount - 1}>Next →</button></div>}
        </aside>
        <article className="record-detail">
          <header>
            <div><span className="section-number">{rowEyebrow(selected)}</span><h2>{rowTitle(selected)}</h2>{run && <CopyLinkButton view="generations" record={generationKey(selected)} label="Copy link to this record" />}</div>
            <div className="pager"><button aria-label="Previous record" onClick={() => moveSelected(-1)} disabled={selectedIndex <= 0}>←</button><span>{Math.max(1, selectedIndex + 1)} of {records.length || 1}</span><button aria-label="Next record" onClick={() => moveSelected(1)} disabled={selectedIndex < 0 || selectedIndex >= records.length - 1}>→</button></div>
          </header>
          <div className="record-meta"><Badge tone="blue">{selected.condition}</Badge>{isPanelRow(selected) ? <span>{selected.turnTitle ?? "Turn not titled"}</span> : <span>Sample {selected.sample}</span>}<span>Seed {selected.seed || "not stamped"}</span><span>{selectedInstrument?.modelID ?? selected.modelID ?? run?.model ?? "Gemma 3 · 27B"}</span></div>
          <section className="text-block prompt-block"><span>PROMPT</span><p>{selected.prompt}</p></section>
          {compare && pairedRecord ? (
            <div className="juxtapose">
              <div className="juxtapose-head"><span>Same item · two conditions · prompt shown once above</span><button onClick={() => setCompare(false)}>Close side-by-side</button></div>
              <div className="juxtapose-grid">
                {[pairedRecord.condition.toLowerCase() === "baseline" ? pairedRecord : selected, pairedRecord.condition.toLowerCase() === "baseline" ? selected : pairedRecord].map((column) => {
                  const columnInstrument = recordFor(column);
                  return (
                    <article className="juxtapose-column" key={column.condition}>
                      <header><Badge tone={column.condition === "baseline" ? "neutral" : "blue"}>{column.condition}</Badge><strong>{column.decision}</strong></header>
                      {columnInstrument ? <InstrumentDetailBlock record={columnInstrument} /> : <p>{column.output}</p>}
                      <footer>{wordText(column)} · distinct-2 {distinct2Text(column)} · seed {column.seed || "not stamped"}</footer>
                    </article>
                  );
                })}
              </div>
            </div>
          ) : selectedInstrument ? (
            <InstrumentDetailBlock record={selectedInstrument} />
          ) : (
            <>
              <section className="text-block output-block"><span>MODEL OUTPUT</span><p>{selected.output}</p></section>
              <section className="derived-block">
                <header><span>DERIVED ENDPOINTS</span><Badge tone="good">Parsed</Badge></header>
                <div><span>Decision</span><strong>{selected.decision}</strong></div>
                <div><span>Numeric parse</span><strong>{selected.months === null ? "Not applicable" : `${selected.months} months`}</strong></div>
                <div><span>Parser read</span><strong>{selected.parsed}</strong></div>
                <div>
                  <span>Surface diagnostics{selected.wordCountStored ? "" : " "}{selected.wordCountStored ? null : <DerivedBadge formula="the record stamped no wordCount, so the viewer counted whitespace-separated tokens of the output text" />}</span>
                  <strong>{wordText(selected)} · distinct-2 {distinct2Text(selected)}</strong>
                </div>
              </section>
            </>
          )}
          {selected.isInstrument && !selectedInstrument && <div className="empty-state">The full answer-token readout for this record was not found in generations.jsonl{run?.previewTruncated ? " within the bounded 32 MB read" : ""}. Only the selected option is shown.</div>}
          <div className={`pair-callout ${pairedRecord ? "" : "pair-unavailable"}`}>
            <span>⇄</span>
            <div>
              <strong>{pairedRecord ? `Same-item pair available · ${pairedRecord.condition}` : "No paired record in loaded preview"}</strong>
              <p>{pairedRecord ? `Open the matching ${pairedRecord.condition} record for this prompt, or read both side by side.` : "The pair may be outside the bounded preview or this run may not use a paired design."}</p>
            </div>
            <div className="pair-actions">
              <button onClick={() => setCompare((value) => !value)} disabled={!pairedRecord}>{compare ? "Hide side-by-side" : "Compare side by side"}</button>
              <button onClick={() => pairedRecord && setSelected(pairedRecord)} disabled={!pairedRecord}>Open pair</button>
            </div>
          </div>
          <footer className="record-path">generations.jsonl{selectedInstrument ? ` · line ${selectedInstrument.line}` : ""} · complete line{run ? " · read locally" : " · SHA-256 verified"}</footer>
        </article>
      </div>
    </div>
  );
}
