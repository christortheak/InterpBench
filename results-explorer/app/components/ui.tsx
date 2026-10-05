"use client";

// Shared presentational pieces. These render values the loaders already
// parsed; they never read a file or derive a statistic themselves.

import { useState } from "react";
import { deepLinkHref, deepLinksAvailable } from "../lib/deeplink";
import { splitCSV } from "../lib/csv";
import { pairedCountLabel } from "../lib/effects";
import { exportCSV, type ExportColumn } from "../lib/export";
import { fmt } from "../lib/format";
import { saveRunFile, saveStatusText, type SaveOutcome } from "../lib/save";
import type { Effect, FilePreview, View } from "../lib/types";

export function Mark() {
  return (
    <span className="mark" aria-hidden="true">
      <i />
      <i />
      <i />
    </span>
  );
}

export function Badge({ children, tone = "neutral" }: { children: React.ReactNode; tone?: "neutral" | "good" | "warn" | "blue" }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

export function ForestRow({ effect, compact = false }: { effect: Effect; compact?: boolean }) {
  const bound = Math.max(Math.abs(effect.low), Math.abs(effect.high), Math.abs(effect.estimate), 0.01) * 1.25;
  const adjustedMin = -bound;
  const max = bound;
  const position = (value: number) => Math.max(1, Math.min(99, ((value - adjustedMin) / (max - adjustedMin)) * 100));
  const zero = position(0);
  return (
    <div className={`forest-row ${compact ? "forest-compact" : ""}`}>
      <div className="forest-label">
        <strong>{effect.short}</strong>
        <span>{pairedCountLabel(effect)}</span>
      </div>
      <div className="forest-track" aria-label={`${effect.endpoint}: ${fmt(effect.estimate)} ${effect.unit}, 95% CI ${fmt(effect.low)} to ${fmt(effect.high)}`}>
        <span className="zero-line" style={{ left: `${zero}%` }} />
        <span className="ci-line" style={{ left: `${position(effect.low)}%`, width: `${position(effect.high) - position(effect.low)}%` }} />
        <span className={`estimate-dot ${effect.q != null && effect.q < 0.05 ? "is-sig" : ""}`} style={{ left: `${position(effect.estimate)}%` }} />
      </div>
      <div className="forest-value">
        <strong>{fmt(effect.estimate, effect.unit === "months" ? 1 : 2)}</strong>
        <span>{effect.unit}</span>
      </div>
    </div>
  );
}

/// What a save control is doing or last did: nothing yet, in progress, or
/// one of lib/save.ts's outcomes.
export type SaveState = SaveOutcome | { state: "saving" } | null;

/// The state every export and download control shares. `run` starts a save
/// and records how it ended, so the control can say "Saved as …" or why
/// nothing was saved. It never throws: a failure is an outcome to show.
export function useSave() {
  const [status, setStatus] = useState<SaveState>(null);
  const run = async (action: () => Promise<SaveOutcome>) => {
    setStatus({ state: "saving" });
    try {
      setStatus(await action());
    } catch (error) {
      setStatus({ state: "failed", message: error instanceof Error && error.message ? error.message : "The file could not be saved." });
    }
  };
  return { status, run };
}

/// The line beside a save control. A cancelled save shows nothing; a
/// failure stays until the next attempt, because the reader needs time to
/// read why.
export function SaveStatus({ status }: { status: SaveState }) {
  const text = saveStatusText(status);
  if (!text || !status) return null;
  return <span className={`save-status save-${status.state}`} role="status">{text}</span>;
}

/// Export the rows a table is CURRENTLY showing, with each column's
/// provenance kind stamped in the file's first line (lib/export.ts). Given
/// no rows it is disabled rather than writing a header-only file that would
/// read as "the table was empty" when the truth is "nothing matched the
/// filters".
export function ExportButton<Row>({ filename, columns, rows, label = "Export CSV", className = "secondary" }: {
  filename: string;
  columns: ExportColumn<Row>[];
  rows: Row[];
  label?: string;
  className?: string;
}) {
  const save = useSave();
  return (
    <span className="export-group">
      <button
        className={className}
        disabled={!rows.length}
        title={rows.length
          ? `Save these ${rows.length} row${rows.length === 1 ? "" : "s"} as CSV. The first line stamps each column as stored, derived, or heuristic.`
          : "Nothing to export in the current filter."}
        onClick={() => save.run(() => exportCSV(filename, columns, rows))}
      >{label} <span>↓</span></button>
      <SaveStatus status={save.status} />
    </span>
  );
}

/// "Copy link to this record" (upgrade plan Phase 5). Rendered ONLY in the
/// embedded app, where the page has a stable address for one known
/// workspace; in the browser build the workspace is a session-scoped folder
/// grant with no URL identity, so there is nothing honest to copy and the
/// button is absent rather than disabled.
export function CopyLinkButton({ view, record, label = "Copy link" }: { view: View; record: string; label?: string }) {
  const [state, setState] = useState<"idle" | "copied" | "failed">("idle");
  if (!deepLinksAvailable()) return null;
  const copy = async () => {
    const href = deepLinkHref({ view, record });
    try {
      await navigator.clipboard.writeText(href);
      setState("copied");
    } catch { setState("failed"); }
    window.setTimeout(() => setState("idle"), 2400);
  };
  return (
    <button className="copy-link-button" onClick={() => void copy()} title={deepLinkHref({ view, record })}>
      {state === "copied" ? "Link copied" : state === "failed" ? "Copy blocked — link is in the tooltip" : label}
    </button>
  );
}

export const NoRunSelected = ({ title }: { title: string }) => (
  <div className="view-enter inner-view">
    <header className="page-title"><div><span className="section-number">NO RUN SELECTED</span><h1>{title}</h1><p>Choose a workspace and run from the sidebar. Nothing shown here is ever inferred or invented.</p></div></header>
    <div className="card no-run-card"><span>⌂</span><h2>No local run selected</h2><p>Pick a run to load its artifacts read-only.</p></div>
  </div>
);

export function FilePreviewModal({ preview, onClose }: { preview: FilePreview; onClose: () => void }) {
  const extension = preview.file.name.split(".").pop()?.toLowerCase() ?? "";
  const isCSV = extension === "csv";
  const csvRows = isCSV && preview.text ? preview.text.split(/\r?\n/).filter(Boolean).slice(0, 250).map(splitCSV) : [];
  const save = useSave();
  // A copy of the file itself, byte for byte. In the app the native host
  // copies it to the place the reader picks; the bytes never pass through
  // this page (lib/save.ts).
  const download = () => save.run(() => saveRunFile(preview.file.name, preview.file.handle));
  return <div className="modal-backdrop file-modal-backdrop" role="presentation" onMouseDown={onClose}>
    <section className="file-preview-modal" role="dialog" aria-modal="true" aria-labelledby="file-preview-title" onMouseDown={(event) => event.stopPropagation()}>
      <header><div><span className="section-number">RUN FILE · READ ONLY</span><h2 id="file-preview-title">{preview.file.name}</h2><p>{preview.file.path} · {preview.file.size < 1024 * 1024 ? `${(preview.file.size / 1024).toFixed(1)} KB` : `${(preview.file.size / 1024 / 1024).toFixed(1)} MB`}</p></div><div><SaveStatus status={save.status} /><button className="secondary" onClick={download}>Download</button><button className="close-file" onClick={onClose} aria-label="Close file preview">×</button></div></header>
      {preview.truncated && <div className="preview-warning">Showing the first 1 MB. Download the file to inspect every byte.</div>}
      <div className="file-preview-body">
        {preview.loading ? <div className="empty-state">Reading local file…</div> : preview.error ? <div className="empty-state">{preview.error}</div> : isCSV && csvRows.length ? <div className="raw-table-scroll"><table className="raw-table"><thead><tr>{csvRows[0].map((cell, index) => <th key={index}>{cell}</th>)}</tr></thead><tbody>{csvRows.slice(1).map((row, rowIndex) => <tr key={rowIndex}>{row.map((cell, cellIndex) => <td key={cellIndex}>{cell}</td>)}</tr>)}</tbody></table></div> : <pre>{preview.text || "This file is binary or has no text preview. Use Download to open it in its native application."}</pre>}
      </div>
    </section>
  </div>;
}
