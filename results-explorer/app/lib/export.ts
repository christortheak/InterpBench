// CSV export of whatever slice a table is CURRENTLY showing (upgrade plan
// Phase 5, "Export").
//
// TWO FILES, each plain.
//
// 1. The table: a header row, then one line per row. Nothing comes before
//    the header. An earlier version put a `# columns: …` comment on the first
//    line; R, Stata, and SPSS do not skip a comment line unless told to, so
//    they read the comment as the header and the real header as data.
//
// 2. The column notes: a second, small CSV with one row per column of the
//    table. A CSV that leaves the viewer loses the badges the screen carried,
//    so the badge vocabulary travels here instead — every column declared
//    STORED (read from a run artifact), DERIVED (computed by the viewer from
//    stored records), or HEURISTIC (derived AND resting on a convention the
//    data does not declare), with the meaning of that word written out.
//
// The exporters themselves live with their views: a view knows its filters,
// and only the filtered rows may be written — an export that silently
// widened to the unfiltered table would be a different claim than the one on
// screen.

import { saveText, type SaveOutcome } from "./save";

export type ColumnKind = "stored" | "derived" | "heuristic";

/// What a cell may hold before formatting. `null`/`undefined` are ABSENT and
/// are written as an empty cell — never as 0, "" -> 0, or "n/a" (the strict
/// reader on the way back in treats a blank as missing; see lib/csv.ts).
export type ExportValue = string | number | boolean | null | undefined;

export type ExportColumn<Row> = {
  header: string;
  kind: ColumnKind;
  value: (row: Row) => ExportValue;
  /// What the column holds, in a sentence, for the column notes file.
  /// Optional: a column without one still gets its kind and that kind's
  /// meaning.
  description?: string;
};

/// RFC-4180 quoting: a cell is quoted when it holds a comma, a quote, a
/// newline, or leading/trailing whitespace a naive reader would trim; inner
/// quotes are doubled. Round-trips through lib/csv.ts `splitCSV`.
export const csvCell = (value: ExportValue): string => {
  if (value === null || value === undefined) return "";
  const text = typeof value === "boolean" ? (value ? "true" : "false") : String(value);
  if (!text) return "";
  const needsQuotes = /[",\r\n]/.test(text) || text !== text.trim();
  return needsQuotes ? `"${text.replaceAll('"', '""')}"` : text;
};

export const csvRow = (cells: ExportValue[]) => cells.map(csvCell).join(",");

/// The header row, then one line per row — and nothing before the header, so
/// statistics software reads the first line as the column names. Rows are
/// written in the order given — the order on screen.
export const buildCSV = <Row,>(columns: ExportColumn<Row>[], rows: Row[]): string => {
  const lines = [
    csvRow(columns.map((column) => column.header)),
    ...rows.map((row) => csvRow(columns.map((column) => column.value(row)))),
  ];
  return `${lines.join("\n")}\n`;
};

/// What each kind means, in words a reader of the notes file needs no
/// glossary for.
export const KIND_MEANING: Record<ColumnKind, string> = {
  stored: "Read as written from the run's saved files.",
  derived: "Worked out by the explorer from the run's saved records, such as a count, a difference, or a label. The engine did not write this value.",
  heuristic: "Worked out by the explorer using a rule of thumb the data does not state. Use it to find things, not as evidence.",
};

/// The column notes: one row per column of the table, in the table's order.
/// `column` matches the table's header exactly, so the two files join on it.
export const buildColumnNotes = <Row,>(columns: ExportColumn<Row>[]): string => {
  const lines = [
    csvRow(["column", "kind", "kindMeaning", "description"]),
    ...columns.map((column) => csvRow([column.header, column.kind, KIND_MEANING[column.kind], column.description ?? ""])),
  ];
  return `${lines.join("\n")}\n`;
};

/// The notes file's name, beside its table: `run-effect-sizes.csv` →
/// `run-effect-sizes-columns.csv`.
export const columnNotesFilename = (tableFilename: string) =>
  `${tableFilename.replace(/\.csv$/i, "")}-columns.csv`;

/// A filesystem-safe name built from the run and table names, ending in the
/// given extension.
export const exportFilename = (extension: string, ...parts: (string | null | undefined)[]) => {
  const slug = parts
    .filter((part): part is string => Boolean(part && part.trim()))
    .map((part) => part.trim().replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, ""))
    .filter(Boolean)
    .join("-");
  return `${slug || "export"}.${extension}`;
};

/// The CSV form, e.g.
/// `20260805T004016927-exp-test-compare-2-2-evaluate-judge-tallies.csv`.
export const csvFilename = (...parts: (string | null | undefined)[]) => exportFilename("csv", ...parts);

/// The one call a view makes: build the CSV for the rows it is showing and
/// hand it to the reader — through the browser's downloads in the
/// standalone build, through the app's save panel when embedded
/// (lib/save.ts). Resolves with what happened, so the control can say so.
export const exportCSV = <Row,>(
  filename: string,
  columns: ExportColumn<Row>[],
  rows: Row[],
): Promise<SaveOutcome> => saveText(filename, buildCSV(columns, rows));

/// The second download: the notes for the same table's columns.
export const exportColumnNotes = <Row,>(
  tableFilename: string,
  columns: ExportColumn<Row>[],
): Promise<SaveOutcome> => saveText(columnNotesFilename(tableFilename), buildColumnNotes(columns));
