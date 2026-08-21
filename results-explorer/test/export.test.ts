import { describe, expect, it } from "vitest";
import { splitCSV, strictNumber } from "../app/lib/csv";
import { buildCSV, columnStampLine, csvCell, csvFilename, csvRow, type ExportColumn } from "../app/lib/export";

// The export contract (docs/RESULTS-EXPLORER-UPGRADE-PLAN.md): "export files
// stamp each column's kind in a header comment". These pin the stamp's shape,
// the quoting, and — the reason the strict reader exists — that an absent
// value leaves an EMPTY cell that reads back as missing, never as zero.

type Row = { name: string; stored: number | null; derived: number | null; flag: boolean };

const columns: ExportColumn<Row>[] = [
  { header: "item", kind: "stored", value: (row) => row.name },
  { header: "target log-odds", kind: "stored", value: (row) => row.stored },
  { header: "Δ vs baseline", kind: "derived", value: (row) => row.derived },
  { header: "at ceiling", kind: "heuristic", value: (row) => row.flag },
];

const rows: Row[] = [
  { name: "loan-legal-ab", stored: 12.5, derived: null, flag: true },
  { name: "loan-legal-ba", stored: 0, derived: -1.25, flag: false },
];

describe("columnStampLine", () => {
  it("names every column with its provenance kind, in order", () => {
    expect(columnStampLine(columns)).toBe(
      "# columns: item (stored), target log-odds (stored), Δ vs baseline (derived), at ceiling (heuristic)",
    );
  });

  it("quotes a header carrying a comma so the stamp stays unambiguous", () => {
    expect(columnStampLine([{ header: "a,b", kind: "stored", value: () => "" }])).toBe('# columns: "a,b" (stored)');
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

  it("puts the column stamp first, as a comment, then the header row", () => {
    expect(lines[0].startsWith("# columns: ")).toBe(true);
    expect(splitCSV(lines[1])).toEqual(["item", "target log-odds", "Δ vs baseline", "at ceiling"]);
  });

  it("writes one line per row, in the order given", () => {
    expect(splitCSV(lines[2])).toEqual(["loan-legal-ab", "12.5", "", "true"]);
    expect(splitCSV(lines[3])).toEqual(["loan-legal-ba", "0", "-1.25", "false"]);
  });

  it("round-trips every cell through the shared CSV reader", () => {
    const quoted = buildCSV(
      [{ header: "reason", kind: "stored", value: (row: { reason: string }) => row.reason }],
      [{ reason: 'the judge said "affirm", then hedged' }],
    );
    expect(splitCSV(quoted.split("\n")[2])).toEqual(['the judge said "affirm", then hedged']);
  });

  it("ends with a trailing newline and no extra blank row", () => {
    expect(text.endsWith("\n")).toBe(true);
    expect(lines.filter((line) => line !== "")).toHaveLength(4);
  });

  it("writes a header-only file for an empty slice rather than inventing rows", () => {
    expect(buildCSV(columns, []).split("\n").filter(Boolean)).toHaveLength(2);
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
});
