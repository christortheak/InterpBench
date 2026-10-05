import { describe, expect, it } from "vitest";
import { splitCSV, strictNumber } from "../app/lib/csv";
import { buildColumnNotes, buildCSV, columnNotesFilename, csvCell, csvFilename, csvRow, exportFilename, KIND_MEANING, type ExportColumn } from "../app/lib/export";

// The export contract: a table whose FIRST line is the header row, so R,
// Stata, SPSS, and spreadsheets read it without being told to skip anything,
// and a second small file that says what kind of value each column holds.
// These pin both shapes, the quoting, and — the reason the strict reader
// exists — that an absent value leaves an EMPTY cell that reads back as
// missing, never as zero.

type Row = { name: string; stored: number | null; derived: number | null; flag: boolean };

const columns: ExportColumn<Row>[] = [
  { header: "item", kind: "stored", value: (row) => row.name, description: "The prompt item's identifier." },
  { header: "target log-odds", kind: "stored", value: (row) => row.stored },
  { header: "Δ vs baseline", kind: "derived", value: (row) => row.derived },
  { header: "at ceiling", kind: "heuristic", value: (row) => row.flag },
];

const rows: Row[] = [
  { name: "loan-legal-ab", stored: 12.5, derived: null, flag: true },
  { name: "loan-legal-ba", stored: 0, derived: -1.25, flag: false },
];

describe("buildColumnNotes", () => {
  const lines = buildColumnNotes(columns).split("\n");

  it("is itself a plain table: a header row, then one row per column, in order", () => {
    expect(splitCSV(lines[0])).toEqual(["column", "kind", "kindMeaning", "description"]);
    expect(lines.slice(1, 5).map((line) => splitCSV(line)[0])).toEqual(["item", "target log-odds", "Δ vs baseline", "at ceiling"]);
    expect(lines.slice(1, 5).map((line) => splitCSV(line)[1])).toEqual(["stored", "stored", "derived", "heuristic"]);
    expect(lines.filter((line) => line !== "")).toHaveLength(5);
  });

  it("writes out what each kind means, and the column's own description when it has one", () => {
    expect(splitCSV(lines[1])).toEqual(["item", "stored", KIND_MEANING.stored, "The prompt item's identifier."]);
    expect(splitCSV(lines[3])).toEqual(["Δ vs baseline", "derived", KIND_MEANING.derived, ""]);
    expect(splitCSV(lines[4])[2]).toBe(KIND_MEANING.heuristic);
  });

  it("names a column exactly as the table's header does, so the two files join", () => {
    const awkward: ExportColumn<Row>[] = [{ header: "a,b", kind: "stored", value: () => "" }];
    expect(splitCSV(buildColumnNotes(awkward).split("\n")[1])[0]).toBe("a,b");
    expect(splitCSV(buildCSV(awkward, []).split("\n")[0])).toEqual(["a,b"]);
  });

  it("is saved beside its table, under a name that says what it is", () => {
    expect(columnNotesFilename("run-effect-sizes.csv")).toBe("run-effect-sizes-columns.csv");
    expect(columnNotesFilename("export.CSV")).toBe("export-columns.csv");
  });
});

describe("csvCell", () => {
  it("writes absent values as an EMPTY cell, which reads back as missing", () => {
    expect(csvCell(null)).toBe("");
    expect(csvCell(undefined)).toBe("");
    // The round trip that matters: blank in, null out — never 0.
    expect(strictNumber(splitCSV("a,,c")[1])).toBeNull();
  });

  it("keeps an explicit zero as a zero", () => {
    expect(csvCell(0)).toBe("0");
    expect(strictNumber(splitCSV(csvRow([0]))[0])).toBe(0);
  });

  it("writes booleans as true/false", () => {
    expect(csvCell(true)).toBe("true");
    expect(csvCell(false)).toBe("false");
  });

  it("quotes commas, quotes, and newlines, doubling inner quotes", () => {
    expect(csvCell("a,b")).toBe('"a,b"');
    expect(csvCell('he said "yes"')).toBe('"he said ""yes"""');
    expect(csvCell("line1\nline2")).toBe('"line1\nline2"');
    expect(csvCell("carriage\rreturn")).toBe('"carriage\rreturn"');
  });

  it("quotes cells whose leading or trailing whitespace a reader would trim", () => {
    expect(csvCell(" padded ")).toBe('" padded "');
  });

  it("leaves an ordinary cell unquoted", () => {
    expect(csvCell("loan-legal-ab")).toBe("loan-legal-ab");
    expect(csvCell(-1.25)).toBe("-1.25");
  });
});

describe("buildCSV", () => {
  const text = buildCSV(columns, rows);
  const lines = text.split("\n");

  it("starts with the header row — nothing before it", () => {
    expect(splitCSV(lines[0])).toEqual(["item", "target log-odds", "Δ vs baseline", "at ceiling"]);
  });

  it("carries no comment line anywhere: statistics software would read one as the header", () => {
    // R's read.csv, Stata's import delimited, and SPSS all take the first
    // line as the column names. The old `# columns: …` first line made them
    // name the columns after the comment and read the real header as a row.
    expect(lines.some((line) => line.startsWith("#"))).toBe(false);
    expect(text).not.toContain("# columns");
  });

  it("gives every line the same number of cells as the header", () => {
    const width = splitCSV(lines[0]).length;
    for (const line of lines.filter((candidate) => candidate !== "")) expect(splitCSV(line)).toHaveLength(width);
  });

  it("writes one line per row, in the order given", () => {
    expect(splitCSV(lines[1])).toEqual(["loan-legal-ab", "12.5", "", "true"]);
    expect(splitCSV(lines[2])).toEqual(["loan-legal-ba", "0", "-1.25", "false"]);
  });

  it("round-trips every cell through the shared CSV reader", () => {
    const quoted = buildCSV(
      [{ header: "reason", kind: "stored", value: (row: { reason: string }) => row.reason }],
      [{ reason: 'the judge said "affirm", then hedged' }],
    );
    expect(splitCSV(quoted.split("\n")[1])).toEqual(['the judge said "affirm", then hedged']);
  });

  it("ends with a trailing newline and no extra blank row", () => {
    expect(text.endsWith("\n")).toBe(true);
    expect(lines.filter((line) => line !== "")).toHaveLength(3);
  });

  it("writes a header-only file for an empty slice rather than inventing rows", () => {
    expect(buildCSV(columns, []).split("\n").filter(Boolean)).toHaveLength(1);
  });
});

describe("csvFilename", () => {
  it("joins the parts and keeps a run directory name readable", () => {
    expect(csvFilename("20260805T004016927-exp-test-compare-2-2-evaluate", "judge-tallies"))
      .toBe("20260805T004016927-exp-test-compare-2-2-evaluate-judge-tallies.csv");
  });

  it("drops empty parts and replaces path-unsafe characters", () => {
    expect(csvFilename("", null, undefined, "a b/c:d")).toBe("a-b-c-d.csv");
  });

  it("never produces a bare extension", () => {
    expect(csvFilename()).toBe("export.csv");
    expect(csvFilename("///")).toBe("export.csv");
  });

  it("builds other extensions the same way", () => {
    expect(exportFilename("jsonl", "20260805T004016927-exp-test-run", "generations"))
      .toBe("20260805T004016927-exp-test-run-generations.jsonl");
    expect(exportFilename("jsonl", undefined, "generations")).toBe("generations.jsonl");
  });
});
