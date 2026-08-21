"use client";

// Analyze-verb outputs the native Results sections rendered and the explorer
// did not (upgrade plan Phase 5): the alien-stance residual table and the
// screen funnel's promoted-movers record.
//
// Everything on this page is STORED. The residual table is rendered from its
// own header row — columns keep the file's labels and order — and every
// numeric cell goes through the strict reader, so a blank adjusted p stays
// missing instead of becoming a significant zero. R itself is the engine's
// number: the viewer never subtracts the human baseline itself.

import { useEffect, useMemo, useState } from "react";
import { ProvenanceLegend } from "../components/provenance";
import { Badge, ExportButton } from "../components/ui";
import {
  loadAlienResiduals,
  parsePromotedMovers,
  residualColumnIndex,
  type PromotedMover,
  type PromotedMoversFile,
  type ResidualRow,
  type ResidualTable,
} from "../lib/analyze";
import { csvFilename, type ExportColumn } from "../lib/export";
import { shortHash } from "../lib/format";
import { readJSONArtifact } from "../lib/judged";
import type { RunFile, WorkspaceRun } from "../lib/types";
import "./judged.css";
import "./analysis.css";

const REGION_NOTE: Record<string, string> = {
  alien: "the model moves where the human baseline does not (or the reverse) — no human-shaped counterpart",
  humanAligned: "the model's movement is statistically indistinguishable from the measured human effect",
  hyperHuman: "same direction as the human effect, larger",
  hypoHuman: "same direction as the human effect, muted",
  inverted: "opposite direction to the measured human effect",
  inertBoth: "neither the model nor the humans moved",
};

const regionTone = (region: string): "good" | "warn" | "neutral" | "blue" =>
  region === "humanAligned" ? "good"
    : region === "alien" || region === "inverted" ? "warn"
      : region === "hyperHuman" || region === "hypoHuman" ? "blue" : "neutral";

const number = (value: number | null) => value == null ? "—" : `${value > 0 ? "+" : ""}${value.toPrecision(4)}`;

export function AnalysisOutputsView({ run, onOpenFile }: { run: WorkspaceRun | null; onOpenFile: (file: RunFile) => void }) {
  const [residuals, setResiduals] = useState<ResidualTable | null>(null);
  const [movers, setMovers] = useState<PromotedMoversFile | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [regionFilter, setRegionFilter] = useState("All regions");
  const [moverFilter, setMoverFilter] = useState<"all" | "promoted" | "rejected">("all");

  useEffect(() => {
    let cancelled = false;
    setResiduals(null); setMovers(null); setError(""); setRegionFilter("All regions"); setMoverFilter("all");
    if (!run) { setLoading(false); return () => { cancelled = true; }; }
    setLoading(true);
    void (async () => {
      try {
        const [table, moversJSON] = await Promise.all([
          loadAlienResiduals(run),
          readJSONArtifact(run, "promoted-movers.json"),
        ]);
        if (cancelled) return;
        setResiduals(table);
        setMovers(parsePromotedMovers(moversJSON));
      } catch (cause) {
        if (!cancelled) setError(cause instanceof Error ? cause.message : String(cause));
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [run?.key]);

  const columns = residuals?.columns ?? [];
  const regionIndex = residualColumnIndex(columns, "region");
  const rIndex = residualColumnIndex(columns, "r");
  const regions = useMemo(() => regionIndex < 0 ? [] : [...new Set((residuals?.rows ?? []).map((row) => row.cells[regionIndex].text).filter(Boolean))].sort(), [residuals, regionIndex]);
  const visibleRows = useMemo(() => (residuals?.rows ?? []).filter((row) =>
    regionFilter === "All regions" || (regionIndex >= 0 && row.cells[regionIndex].text === regionFilter)), [residuals, regionFilter, regionIndex]);

  // Header-driven export: the file's own column names, all stored.
  const residualColumns: ExportColumn<ResidualRow>[] = columns.map((column, index) => ({
    header: column.header,
    kind: "stored" as const,
    value: (row: ResidualRow) => row.cells[index]?.text ?? "",
  }));

  const moverColumns: ExportColumn<PromotedMover>[] = [
    { header: "concept", kind: "stored", value: (row) => row.concept },
    { header: "promoted", kind: "stored", value: (row) => row.promoted },
    { header: "condition", kind: "stored", value: (row) => row.condition },
    { header: "endpoint", kind: "stored", value: (row) => row.endpoint },
    { header: "effectEstimate", kind: "stored", value: (row) => row.effectEstimate },
    { header: "effectCILower", kind: "stored", value: (row) => row.effectCILower },
    { header: "effectCIUpper", kind: "stored", value: (row) => row.effectCIUpper },
    { header: "wilcoxonP", kind: "stored", value: (row) => row.wilcoxonP },
    { header: "adjustedP", kind: "stored", value: (row) => row.adjustedP },
    { header: "correction", kind: "stored", value: (row) => row.correction },
    { header: "doseMonotone", kind: "stored", value: (row) => row.doseMonotone },
    { header: "capabilityPassed", kind: "stored", value: (row) => row.capabilityPassed },
    { header: "randomFloorEffect", kind: "stored", value: (row) => row.randomFloorEffect },
    { header: "reasons", kind: "stored", value: (row) => row.reasons.join("; ") },
  ];

  const header = (eyebrow: string, note: string) => (
    <header className="page-title">
      <div><span className="section-number">{eyebrow}</span><h1>Analysis outputs</h1><p>{note}</p></div>
      <div className="title-actions">
        {residuals?.file && <button className="secondary" onClick={() => residuals.file && onOpenFile(residuals.file)}>alien-residuals.csv</button>}
        {movers?.file && <button className="secondary" onClick={() => movers.file && onOpenFile(movers.file)}>promoted-movers.json</button>}
      </div>
    </header>
  );

  if (!run) return <div className="view-enter inner-view analysis-view">{header("NO RUN SELECTED", "Choose an analyze run to read its human-anchored residuals and screen-funnel decisions.")}<div className="card no-run-card"><span>⌂</span><h2>No local run selected</h2><p>Pick a run to load its analyze artifacts read-only.</p></div></div>;
  if (loading) return <div className="view-enter inner-view analysis-view">{header("READING", `Reading analyze artifacts from ${run.name}.`)}<div className="card no-run-card"><span>…</span><h2>Reading alien-residuals.csv and promoted-movers.json</h2><p>Both are small, engine-written tables.</p></div></div>;
  if (error) return <div className="view-enter inner-view analysis-view">{header("READ FAILED", `The analyze artifacts could not be read from ${run.name}.`)}<div className="card no-run-card"><span>!</span><h2>Analyze artifacts could not be read</h2><p>{error}</p></div></div>;

  const hasResiduals = Boolean(residuals?.present && residuals.rows.length);
  const hasMovers = Boolean(movers?.present && (movers.promoted.length || movers.rejected.length));

  if (!hasResiduals && !hasMovers) return <div className="view-enter inner-view analysis-view">
    {header("NO ANALYZE OUTPUTS", `${run.name} carries neither analyze artifact.`)}
    <div className="card no-run-card"><span>∅</span><h2>No residuals and no funnel record</h2><p>
      <code>alien-residuals.csv</code> is written only when the manifest pins a HUMAN BASELINE, and <code>promoted-movers.json</code> only for a <code>screen</code>-phase manifest with a promotion rule. Neither is present here{residuals?.present ? " with readable rows" : ""}{movers?.error ? ` (${movers.error})` : ""} — this run measured no human comparison and made no promotion decisions, which is different from having measured one that came out null.
    </p></div>
  </div>;

  const visibleMovers = moverFilter === "promoted" ? (movers?.promoted ?? [])
    : moverFilter === "rejected" ? (movers?.rejected ?? [])
      : [...(movers?.promoted ?? []), ...(movers?.rejected ?? [])];

  return (
    <div className="view-enter inner-view analysis-view">
      {header(
        `${hasResiduals ? `${residuals!.rows.length} RESIDUAL ROW${residuals!.rows.length === 1 ? "" : "S"}` : "NO RESIDUALS"} · ${hasMovers ? `${(movers?.promoted.length ?? 0)}/${visibleMoversTotal(movers)} PROMOTED` : "NO FUNNEL RECORD"}`,
        `Human-anchored residuals and screen-funnel decisions, as ${run.name} wrote them.`,
      )}
      <ProvenanceLegend />

      {hasResiduals && <section className="card">
        <header className="section-header">
          <div>
            <span className="section-number">ALIEN-STANCE RESIDUALS</span>
            <h2>R = Δ<sub>model</sub> − Δ<sub>human</sub></h2>
            <p>How far the model&rsquo;s response to the intervention sits from the measured human effect: human-shaped, muted, amplified, inverted, or alien. Every value — including R, its interval, and the region call — is the engine&rsquo;s.</p>
          </div>
          <div className="analysis-filters">
            {regions.length > 1 && <select value={regionFilter} onChange={(event) => setRegionFilter(event.target.value)} aria-label="Filter residual rows by region"><option>All regions</option>{regions.map((name) => <option key={name}>{name}</option>)}</select>}
            <ExportButton filename={csvFilename(run.name, "alien-residuals")} columns={residualColumns} rows={visibleRows} />
          </div>
        </header>
        {residuals!.skipped > 0 && <p className="analysis-note"><span>{residuals!.skipped} row{residuals!.skipped === 1 ? "" : "s"} did not match the header&rsquo;s column count and {residuals!.skipped === 1 ? "is" : "are"} excluded rather than padded.</span></p>}
        <div className="jv-table-scroll">
          <table className="jv-table residual-table">
            <thead><tr>{columns.map((column, index) => <th key={`${column.header}-${index}`} className={index === rIndex ? "is-headline" : ""}>{column.header}</th>)}</tr></thead>
            <tbody>
              {visibleRows.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {row.cells.map((cell, index) => {
                    const column = columns[index];
                    if (index === regionIndex) return <td key={index} className="region-cell">{cell.text ? <Badge tone={regionTone(cell.text)}>{cell.text}</Badge> : "—"}<small>{REGION_NOTE[cell.text] ?? ""}</small></td>;
                    if (column?.numeric) return <td key={index} className={`numeric-cell ${index === rIndex ? "is-headline" : ""}`}>{cell.value == null ? "—" : number(cell.value)}</td>;
                    return <td key={index}>{cell.text || "—"}</td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!visibleRows.length && <div className="empty-state">No residual row is in that region.</div>}
        <footer className="table-note"><strong>Stored:</strong> every column above is read from <code>alien-residuals.csv</code> with its own header label, in the file&rsquo;s own order. A blank cell stays blank — never zero. The human side of R is a PINNED baseline table; its provenance is the manifest&rsquo;s <code>humanBaseline</code> pin, not this run.</footer>
      </section>}

      {!hasResiduals && <section className="card"><header className="section-header"><div><span className="section-number">ALIEN-STANCE RESIDUALS</span><h2>Not measured in this run</h2></div></header><div className="artifact-empty"><span>∅</span><p>No <code>alien-residuals.csv</code>: this analyze ran against a manifest with no pinned human baseline, so R was never computed. Nothing is substituted for it.</p></div></section>}

      {hasMovers && <section className="card">
        <header className="section-header">
          <div>
            <span className="section-number">SCREEN FUNNEL · PROMOTED MOVERS</span>
            <h2>Which concepts enter the confirm phase, and why the others do not</h2>
            <p>{movers!.promoted.length} promoted · {movers!.rejected.length} rejected{movers!.experiment ? ` · ${movers!.experiment}` : ""}{movers!.experimentHash ? ` @ ${shortHash(movers!.experimentHash)}` : ""}.</p>
          </div>
          <div className="analysis-filters">
            <select value={moverFilter} onChange={(event) => setMoverFilter(event.target.value as "all" | "promoted" | "rejected")} aria-label="Filter funnel decisions">
              <option value="all">All decisions</option>
              <option value="promoted">Promoted only</option>
              <option value="rejected">Rejected only</option>
            </select>
            <ExportButton filename={csvFilename(run.name, "promoted-movers")} columns={moverColumns} rows={visibleMovers} />
          </div>
        </header>
        {movers!.rule && <dl className="pin-grid">
          {Object.entries(movers!.rule).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{value === null ? "not set" : typeof value === "object" ? JSON.stringify(value) : String(value)}</dd></div>)}
        </dl>}
        <div className="mover-list">
          {visibleMovers.map((entry) => (
            <article className={`mover ${entry.promoted ? "is-promoted" : "is-rejected"}`} key={`${entry.concept}-${entry.condition}-${entry.endpoint}`}>
              <header>
                <div><strong>{entry.concept}</strong><span>{[entry.condition, entry.endpoint].filter(Boolean).join(" · ") || "condition and endpoint not stamped"}</span></div>
                <Badge tone={entry.promoted ? "good" : "neutral"}>{entry.promoted ? "promoted" : "rejected"}</Badge>
              </header>
              <dl>
                <div><dt>Effect</dt><dd>{number(entry.effectEstimate)}{entry.effectCILower != null && entry.effectCIUpper != null ? ` [${entry.effectCILower.toPrecision(4)}, ${entry.effectCIUpper.toPrecision(4)}]` : ""}</dd></div>
                <div><dt>Adjusted p</dt><dd>{entry.adjustedP == null ? "not reported" : entry.adjustedP.toPrecision(3)}{entry.correction ? ` · ${entry.correction}` : ""}</dd></div>
                <div><dt>Raw p</dt><dd>{entry.wilcoxonP == null ? "not reported" : entry.wilcoxonP.toPrecision(3)}</dd></div>
                <div><dt>Dose monotone</dt><dd>{entry.doseMonotone == null ? "not evaluated" : entry.doseMonotone ? "yes" : "no"}</dd></div>
                <div><dt>Capability gate</dt><dd>{entry.capabilityPassed == null ? "not evaluated" : entry.capabilityPassed ? "passed" : "failed"}</dd></div>
                <div><dt>Random floor</dt><dd>{entry.randomFloorEffect == null ? "not measured" : entry.randomFloorEffect.toPrecision(4)}</dd></div>
              </dl>
              <p className="mover-reasons">{entry.reasons.length ? entry.reasons.join(" · ") : "Every declared criterion was satisfied."}</p>
            </article>
          ))}
        </div>
        <footer className="table-note"><strong>Stored:</strong> read verbatim from <code>promoted-movers.json</code>, rule block included. Rejection reasons are the engine&rsquo;s own words — the funnel is only defensible if rejections are as documented as promotions, so none is summarized away here.</footer>
      </section>}
    </div>
  );
}

const visibleMoversTotal = (movers: PromotedMoversFile | null) =>
  (movers?.promoted.length ?? 0) + (movers?.rejected.length ?? 0);
