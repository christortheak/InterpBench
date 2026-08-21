// Analyze-verb artifacts the native Results sections rendered and the
// explorer did not (upgrade plan Phase 5): `alien-residuals.csv` and
// `promoted-movers.json`.
//
// ARTIFACT CONTRACTS.
//
// alien-residuals.csv — written by the server's analyze when the manifest
// pins a human baseline (`experiment/residuals.py`). Its header is
// `condition,endpoint,deltaModel,ciModelLower,ciModelUpper,deltaHuman,
// ciHumanLower,ciHumanUpper,R,ciRLower,ciRUpper,region`, and the headline
// quantity is the project's R = delta_model − delta_human: how far the
// model's response to the intervention sits from the measured human effect.
// The `region` cell is the engine's own classification (alien /
// humanAligned / hyperHuman / hypoHuman / inverted / inertBoth).
//
// This reader is HEADER-DRIVEN on purpose. Every column is labelled from the
// file's own header row and rendered in the file's own order, so a future
// column appears rather than being dropped, and no column is renamed into a
// meaning the file did not claim. Cells are read with `strictNumber`: a
// blank stays NULL — never 0, and never a q of 0 that would read as
// significant.
//
// promoted-movers.json — the screen-funnel decision record
// (`experiment/promotion.py`). Its point is that REJECTIONS are as
// documented as promotions: every decision carries the reasons it failed.
// Nothing here re-decides anything; the artifact is rendered as written.

import { splitCSV, strictNumber } from "./csv";
import { findFile } from "./discovery";
import type { LoadedJSON } from "./judged";
import type { RunFile, WorkspaceRun } from "./types";

// ---------------------------------------------------------------------------
// alien-residuals.csv
// ---------------------------------------------------------------------------

export type ResidualColumn = {
  /// The header cell verbatim — the column's only label.
  header: string;
  /// Lower-cased header, for the few semantic touches the view makes
  /// (highlighting R, colouring `region`).
  key: string;
  /// Every non-blank cell in this column parsed as a finite number, and at
  /// least one did. Numeric columns render right-aligned and monospaced;
  /// nothing is coerced.
  numeric: boolean;
};

export type ResidualCell = { text: string; value: number | null };
export type ResidualRow = { cells: ResidualCell[] };

export type ResidualTable = {
  present: boolean;
  file: RunFile | null;
  columns: ResidualColumn[];
  rows: ResidualRow[];
  /// Lines whose cell count did not match the header — reported, not padded.
  skipped: number;
};

export const parseResidualTable = (text: string): Pick<ResidualTable, "columns" | "rows" | "skipped"> => {
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  if (!lines.length) return { columns: [], rows: [], skipped: 0 };
  const headers = splitCSV(lines[0]);
  let skipped = 0;
  const parsed = lines.slice(1).flatMap((line): string[][] => {
    const cells = splitCSV(line);
    if (cells.length !== headers.length) { skipped += 1; return []; }
    return [cells];
  });
  const columns: ResidualColumn[] = headers.map((header, index) => {
    const values = parsed.map((cells) => cells[index]).filter((cell) => cell.trim() !== "");
    return {
      header,
      key: header.trim().toLowerCase(),
      numeric: values.length > 0 && values.every((cell) => strictNumber(cell) !== null),
    };
  });
  const rows: ResidualRow[] = parsed.map((cells) => ({
    cells: cells.map((cell) => ({ text: cell, value: strictNumber(cell) })),
  }));
  return { columns, rows, skipped };
};

export const loadAlienResiduals = async (run: Pick<WorkspaceRun, "files">): Promise<ResidualTable> => {
  const file = findFile(run.files, "alien-residuals.csv");
  if (!file) return { present: false, file: null, columns: [], rows: [], skipped: 0 };
  const text = await (await file.handle.getFile()).text();
  return { present: true, file, ...parseResidualTable(text) };
};

/// The column index a header names, or -1. Used only for presentation
/// (which column to emphasise), never to reinterpret a cell.
export const residualColumnIndex = (columns: ResidualColumn[], key: string) =>
  columns.findIndex((column) => column.key === key);

// ---------------------------------------------------------------------------
// promoted-movers.json
// ---------------------------------------------------------------------------

export type PromotedMover = {
  concept: string;
  condition: string;
  endpoint: string;
  effectEstimate: number | null;
  effectCILower: number | null;
  effectCIUpper: number | null;
  wilcoxonP: number | null;
  adjustedP: number | null;
  correction: string;
  /// null = the decision record stamped no verdict on this criterion, which
  /// is not the same as failing it.
  doseMonotone: boolean | null;
  capabilityPassed: boolean | null;
  randomFloorEffect: number | null;
  promoted: boolean;
  /// Why the concept did NOT pass, in the engine's words. Empty for a
  /// promotion (nothing failed).
  reasons: string[];
};

export type PromotedMoversFile = {
  present: boolean;
  error: string;
  file: RunFile | null;
  experiment: string;
  experimentHash: string;
  /// The pinned promotion rule as stamped, rendered verbatim.
  rule: Record<string, unknown> | null;
  promoted: PromotedMover[];
  rejected: PromotedMover[];
};

const record = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
const str = (value: unknown) => typeof value === "string" ? value : "";
const finite = (value: unknown) => typeof value === "number" && Number.isFinite(value) ? value : null;
const bool = (value: unknown) => typeof value === "boolean" ? value : null;

const mover = (value: unknown, fallbackPromoted: boolean): PromotedMover | null => {
  const raw = record(value);
  const concept = str(raw.concept);
  if (!concept) return null;
  return {
    concept,
    condition: str(raw.condition),
    endpoint: str(raw.endpoint),
    effectEstimate: finite(raw.effectEstimate),
    effectCILower: finite(raw.effectCILower),
    effectCIUpper: finite(raw.effectCIUpper),
    wilcoxonP: finite(raw.wilcoxonP),
    adjustedP: finite(raw.adjustedP),
    correction: str(raw.correction),
    doseMonotone: bool(raw.doseMonotone),
    capabilityPassed: bool(raw.capabilityPassed),
    randomFloorEffect: finite(raw.randomFloorEffect),
    // The list a decision was written under is the fact; the per-entry flag
    // agrees with it in every engine-written file, and backs it up here.
    promoted: typeof raw.promoted === "boolean" ? raw.promoted : fallbackPromoted,
    reasons: Array.isArray(raw.reasons) ? raw.reasons.filter((item): item is string => typeof item === "string") : [],
  };
};

export const parsePromotedMovers = (loaded: LoadedJSON): PromotedMoversFile => {
  const raw = loaded.raw;
  const list = (key: string, fallbackPromoted: boolean) =>
    (Array.isArray(raw[key]) ? raw[key] as unknown[] : [])
      .flatMap((item) => { const parsed = mover(item, fallbackPromoted); return parsed ? [parsed] : []; });
  return {
    present: loaded.present && !loaded.error,
    error: loaded.error,
    file: loaded.file,
    experiment: str(raw.experiment),
    experimentHash: str(raw.experimentHash),
    rule: Object.keys(record(raw.promotionRule)).length ? record(raw.promotionRule) : null,
    promoted: list("promoted", true),
    rejected: list("rejected", false),
  };
};
