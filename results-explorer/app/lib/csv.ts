// CSV cell splitting and the strict numeric reader every artifact parser
// shares.

export const splitCSV = (line: string) => {
  const values: string[] = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"' && line[index + 1] === '"' && quoted) { value += '"'; index += 1; }
    else if (character === '"') quoted = !quoted;
    else if (character === "," && !quoted) { values.push(value); value = ""; }
    else value += character;
  }
  values.push(value);
  return values;
};

/// How many lines of a CSV the file preview lays out as a table, header
/// included. A bound, so a table with a hundred thousand rows does not
/// freeze the page.
export const CSV_PREVIEW_LINES = 250;

export type CSVPreview = {
  /// The lines laid out, split into cells; the first is the header.
  lines: string[][];
  /// Data rows laid out (the header is not a row).
  shownRows: number;
  /// Data rows in the text that was read.
  readRows: number;
  /// Rows were left out of the table by the bound above.
  cut: boolean;
  /// The text itself is only the start of the file, so `readRows` is a
  /// floor on the file's real row count, not the count.
  partial: boolean;
};

/// Lay out the head of a CSV. `partial` says the text is only the first part
/// of a larger file; its last line is then likely half a row and is dropped.
export const csvPreview = (text: string, partial = false): CSVPreview => {
  const all = text.split(/\r?\n/);
  if (partial) all.pop();
  const present = all.filter(Boolean);
  const lines = present.slice(0, CSV_PREVIEW_LINES).map(splitCSV);
  return {
    lines,
    shownRows: Math.max(0, lines.length - 1),
    readRows: Math.max(0, present.length - 1),
    cut: present.length > CSV_PREVIEW_LINES,
    partial,
  };
};

/// What to tell the reader when the table is not the whole file. "" when it
/// is. The preview used to stop at 250 lines without a word, so a
/// 3,000-row table looked like a 249-row one.
export const csvPreviewNotice = (preview: CSVPreview): string => {
  if (!preview.cut) return "";
  return preview.partial
    ? `Showing the first ${preview.shownRows} rows. The file is larger than the part read for this preview, which alone holds ${preview.readRows} rows. Download the file to see them all.`
    : `Showing the first ${preview.shownRows} of ${preview.readRows} rows. Download the file to see them all.`;
};

/// One sentence for lines of a file that could not be read as records and
/// were left out. "" when none were. Said whenever ANY line is skipped — it
/// used to be said only when every line was, so a file that lost three
/// records out of four hundred looked complete.
export const skippedLinesNote = (skipped: number, file: string, what = "records"): string =>
  skipped > 0
    ? `${skipped} line${skipped === 1 ? "" : "s"} in ${file} could not be read as ${what} and ${skipped === 1 ? "is" : "are"} not shown here. ${skipped === 1 ? "It is" : "They are"} still in the file.`
    : "";

// STRICT numeric parsing (review 2026-08-03, P0): `Number("") === 0`, so a
// blank CSV cell silently became a substantive zero — a missing adjusted
// p-value even read as q = 0, falsely significant. A cell is a number only
// when non-empty and finite; missing stays null.
export const strictNumber = (raw: unknown): number | null => {
  const text = String(raw ?? "").trim();
  if (!text) return null;
  const value = Number(text);
  return Number.isFinite(value) ? value : null;
};
