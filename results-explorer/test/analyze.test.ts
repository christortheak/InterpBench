import { describe, expect, it } from "vitest";
import {
  loadAlienResiduals,
  parsePromotedMovers,
  parseResidualTable,
  residualColumnIndex,
} from "../app/lib/analyze";
import type { LoadedJSON } from "../app/lib/judged";
import type { LocalFileHandle, RunFile } from "../app/lib/types";

const fakeFile = (name: string, text: string): File => {
  const bytes = new TextEncoder().encode(text);
  return {
    name, size: bytes.byteLength, lastModified: 0, type: "",
    text: async () => text,
    arrayBuffer: async () => bytes.buffer,
    slice: (start?: number, end?: number) => ({ text: async () => text.slice(start ?? 0, end) }),
  } as unknown as File;
};

const fakeRunFile = (name: string, text: string): RunFile => ({
  name, path: name, size: new TextEncoder().encode(text).byteLength, modified: 0,
  handle: { kind: "file", name, getFile: async () => fakeFile(name, text) } as LocalFileHandle,
});

const files = (contents: Record<string, string>) => ({
  files: Object.entries(contents).map(([name, text]) => fakeRunFile(name, text)),
});

// The server's ALIEN_RESIDUALS_HEADER, verbatim (experiment/residuals.py).
const residualCSV = [
  "condition,endpoint,deltaModel,ciModelLower,ciModelUpper,deltaHuman,ciHumanLower,ciHumanUpper,R,ciRLower,ciRUpper,region",
  "fear-agent,sentence_months,7.8,3.1,12.4,2.1,0.4,3.8,5.7,1.2,10.3,hyperHuman",
  "fear-agent,rule_adherent_rate,0.14,0.04,0.23,0,,,0.14,,,alien",
].join("\n");

describe("parseResidualTable", () => {
  const table = parseResidualTable(residualCSV);

  it("labels every column from the file's own header, in the file's order", () => {
    expect(table.columns.map((column) => column.header)).toEqual([
      "condition", "endpoint", "deltaModel", "ciModelLower", "ciModelUpper",
      "deltaHuman", "ciHumanLower", "ciHumanUpper", "R", "ciRLower", "ciRUpper", "region",
    ]);
    expect(residualColumnIndex(table.columns, "r")).toBe(8);
    expect(residualColumnIndex(table.columns, "region")).toBe(11);
    expect(residualColumnIndex(table.columns, "notacolumn")).toBe(-1);
  });

  it("reads numeric cells strictly and leaves a BLANK interval null, never zero", () => {
    const row = table.rows[1];
    expect(row.cells[8].value).toBe(0.14);
    // ciRLower / ciRUpper are blank on that row: missing, not 0.
    expect(row.cells[9].value).toBeNull();
    expect(row.cells[10].value).toBeNull();
    // An explicit zero stays a zero.
    expect(row.cells[5].value).toBe(0);
  });

  it("marks a column numeric only when every non-blank cell is a number", () => {
    const byHeader = Object.fromEntries(table.columns.map((column) => [column.header, column.numeric]));
    expect(byHeader.deltaModel).toBe(true);
    expect(byHeader.R).toBe(true);
    // Present on one row, blank on the other — still numeric.
    expect(byHeader.ciRLower).toBe(true);
    expect(byHeader.condition).toBe(false);
    expect(byHeader.region).toBe(false);
  });

  it("keeps the region text verbatim for the engine's own classification", () => {
    expect(table.rows[0].cells[11].text).toBe("hyperHuman");
    expect(table.rows[1].cells[11].text).toBe("alien");
  });

  it("excludes a row whose cell count does not match the header rather than padding it", () => {
    const ragged = parseResidualTable(`${residualCSV}\nfear-agent,short-row,1`);
    expect(ragged.rows).toHaveLength(2);
    expect(ragged.skipped).toBe(1);
  });

  it("reads an empty file as an empty table", () => {
    expect(parseResidualTable("")).toEqual({ columns: [], rows: [], skipped: 0 });
  });

  it("survives a future column it has never seen", () => {
    const widened = parseResidualTable("condition,endpoint,R,region,newColumn\na,b,1.5,alien,note");
    expect(widened.columns.map((column) => column.header)).toContain("newColumn");
    expect(widened.rows[0].cells[4].text).toBe("note");
  });
});

describe("loadAlienResiduals", () => {
  it("reads the run's table", async () => {
    const table = await loadAlienResiduals(files({ "alien-residuals.csv": residualCSV }));
    expect(table.present).toBe(true);
    expect(table.rows).toHaveLength(2);
    expect(table.file?.name).toBe("alien-residuals.csv");
  });

  it("says a run has no residual table rather than returning an empty one", async () => {
    const table = await loadAlienResiduals(files({}));
    expect(table.present).toBe(false);
    expect(table.columns).toEqual([]);
    expect(table.rows).toEqual([]);
  });
});

const loaded = (raw: Record<string, unknown>): LoadedJSON => ({ present: true, raw, error: "", file: null });

describe("parsePromotedMovers", () => {
  // The shape of PromotionDecision.as_json() (experiment/promotion.py).
  const payload = {
    experiment: "screen-2026-08",
    experimentHash: "abc123",
    promotionRule: { fdrThreshold: 0.1, doseMonotone: true, exceedsRandomFloor: true, capabilityGate: null },
    promoted: [{
      concept: "fear", condition: "fear-agent", endpoint: "sentence_months",
      effectEstimate: 7.8, effectCILower: 3.1, effectCIUpper: 12.4,
      wilcoxonP: 0.003, adjustedP: 0.012, correction: "bh",
      doseMonotone: true, doseSpearmanRho: 0.98, randomFloorEffect: 0.4,
      capabilityPassed: true, promoted: true, reasons: [],
    }],
    rejected: [{
      concept: "sympathy", condition: "sympathy-agent", endpoint: "sentence_months",
      effectEstimate: 0.2, effectCILower: -1.1, effectCIUpper: 1.5,
      wilcoxonP: null, adjustedP: null, correction: "bh",
      doseMonotone: null, randomFloorEffect: null, capabilityPassed: null,
      promoted: false,
      reasons: ["no adjusted p-value (screen analysis incomplete)", "no matched-norm random floor measured"],
    }],
  };

  it("reads both lists with their provenance", () => {
    const movers = parsePromotedMovers(loaded(payload));
    expect(movers.present).toBe(true);
    expect(movers.experiment).toBe("screen-2026-08");
    expect(movers.experimentHash).toBe("abc123");
    expect(movers.rule).toEqual(payload.promotionRule);
    expect(movers.promoted.map((row) => row.concept)).toEqual(["fear"]);
    expect(movers.rejected.map((row) => row.concept)).toEqual(["sympathy"]);
  });

  it("keeps every rejection reason — the funnel is only defensible if they survive", () => {
    const rejected = parsePromotedMovers(loaded(payload)).rejected[0];
    expect(rejected.reasons).toHaveLength(2);
    expect(rejected.reasons[0]).toContain("no adjusted p-value");
    expect(rejected.promoted).toBe(false);
  });

  it("keeps an unevaluated criterion NULL rather than reading it as a failure", () => {
    const rejected = parsePromotedMovers(loaded(payload)).rejected[0];
    expect(rejected.doseMonotone).toBeNull();
    expect(rejected.capabilityPassed).toBeNull();
    expect(rejected.randomFloorEffect).toBeNull();
    expect(rejected.adjustedP).toBeNull();
    expect(rejected.wilcoxonP).toBeNull();
  });

  it("falls back to the list a decision was written under when the flag is absent", () => {
    const movers = parsePromotedMovers(loaded({ promoted: [{ concept: "anger" }], rejected: [{ concept: "calm" }] }));
    expect(movers.promoted[0].promoted).toBe(true);
    expect(movers.rejected[0].promoted).toBe(false);
  });

  it("drops an entry with no concept and reports an absent file as absent", () => {
    expect(parsePromotedMovers(loaded({ promoted: [{ endpoint: "x" }], rejected: [] })).promoted).toEqual([]);
    const missing = parsePromotedMovers({ present: false, raw: {}, error: "", file: null });
    expect(missing.present).toBe(false);
    expect(missing.rule).toBeNull();
  });

  it("carries a parse error through instead of showing an empty funnel", () => {
    const broken = parsePromotedMovers({ present: true, raw: {}, error: "promoted-movers.json is present but could not be parsed as JSON.", file: null });
    expect(broken.present).toBe(false);
    expect(broken.error).toContain("could not be parsed");
  });
});
