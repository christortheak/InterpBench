// CSV export of whatever slice a table is CURRENTLY showing (upgrade plan
// Phase 5, "Export").
//
// THE PROVENANCE RULE, in file form. The plan's cross-cutting contract ends
// with: "export files stamp each column's kind in a header comment". A CSV
// that leaves the viewer loses the badges the screen carried, so the badge
// vocabulary travels as the first line instead — every column declared
// STORED (read from a run artifact), DERIVED (computed by the viewer from
// stored records) or HEURISTIC (derived AND resting on a convention the data
// does not declare). The stamp is a `#` comment, so the rest of the file is
// still an ordinary CSV that any reader parses.
//
// The exporters themselves live with their views: a view knows its filters,
// and only the filtered rows may be written — an export that silently
// widened to the unfiltered table would be a different claim than the one on
// screen.

export type ColumnKind = "stored" | "derived" | "heuristic";

/// What a cell may hold before formatting. `null`/`undefined` are ABSENT and
/// are written as an empty cell — never as 0, "" -> 0, or "n/a" (the strict
/// reader on the way back in treats a blank as missing; see lib/csv.ts).
export type ExportValue = string | number | boolean | null | undefined;

export type ExportColumn<Row> = {
  header: string;
  kind: ColumnKind;
  value: (row: Row) => ExportValue;
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

/// The header comment: `# columns: name (stored), other (derived), …`.
/// A header carrying a comma would make the stamp ambiguous, so it is
/// quoted there exactly as it is in the header row.
export const columnStampLine = <Row,>(columns: ExportColumn<Row>[]) =>
  `# columns: ${columns.map((column) => `${csvCell(column.header)} (${column.kind})`).join(", ")}`;

/// Stamp line, header row, then one line per row. Rows are written in the
/// order given — the order on screen.
export const buildCSV = <Row,>(columns: ExportColumn<Row>[], rows: Row[]): string => {
  const lines = [
    columnStampLine(columns),
    csvRow(columns.map((column) => column.header)),
    ...rows.map((row) => csvRow(columns.map((column) => column.value(row)))),
  ];
  return `${lines.join("\n")}\n`;
};

/// A filesystem-safe name built from the run and table names, e.g.
/// `20260805T004016927-exp-test-compare-2-2-evaluate-judge-tallies.csv`.
export const csvFilename = (...parts: (string | null | undefined)[]) => {
  const slug = parts
    .filter((part): part is string => Boolean(part && part.trim()))
    .map((part) => part.trim().replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, ""))
    .filter(Boolean)
    .join("-");
  return `${slug || "export"}.csv`;
};

/// Write the text to the user's downloads through an object URL. Browser
/// only — a no-op where there is no document (the unit suite).
export const downloadCSV = (filename: string, text: string) => {
  if (typeof document === "undefined") return;
  const url = URL.createObjectURL(new Blob([text], { type: "text/csv;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
};

/// The one call a view makes: build the stamped CSV for the rows it is
/// showing and hand it to the browser.
export const exportCSV = <Row,>(
  filename: string,
  columns: ExportColumn<Row>[],
  rows: Row[],
) => downloadCSV(filename, buildCSV(columns, rows));
