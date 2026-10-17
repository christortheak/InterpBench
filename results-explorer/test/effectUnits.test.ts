import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { strictNumber } from "../app/lib/csv";
import { effectKey, pairedCountLabel } from "../app/lib/effects";
import {
  countPairedItems, countPairedRecords, countPhrase, effectsUnitSummary, effectUnit, enginePairing, MINIMUM_PAIRS,
  pairedItemsFromText, pairedRecordsFromText, RESPONSE_CAVEAT, RESPONSE_UNIT_EXPLANATION, resolveUnits,
  UNKNOWN_UNIT_NOTE, UNSTAMPED_UNIT_LINES, unitCaveat, unitLabel, unitLines,
  type EffectUnit, type EnginePairing, type SamplesPerItem, type UnitSource,
} from "../app/lib/effectUnits";
import { fmt } from "../app/lib/format";
import { hasTooFewPairs, intervalLine } from "../app/lib/headline";
import type { Effect, RunFile } from "../app/lib/types";

// What one paired difference of a stored effect row is. Before SteerLab
// 0.9.7 the Mac engine paired every response of a multi-sample run with the
// baseline response to the same item and seed, counted responses in n, and
// stamped no unit, so five seeds of one prompt read as "5 paired items" with
// an interval; and fewer pairs than items (after an exclusion, missing
// values, or in a subgroup) does not show a row counts items either. The
// cases are the SHARED fixture the Python reader
// (Server/tests/test_results_export.py) answers and the Mac app
// (EffectUnitTests.swift) reads too, so the three copies of the rule cannot
// drift apart unnoticed. The hand-written answers that prove the rule right
// are in Server/tests/test_effect_unit_interpretation.py.

const checkout = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");

type StoredRow = { condition: string; endpoint: string; n: string; unit: string; stratifyBy: string; stratum: string };
type ExpectedRow = {
  condition: string; endpoint: string; stratifyBy: string; stratum: string;
  unit: string; unitSource: UnitSource; pairedItems: number | null; count: string | null; tooFew: boolean;
};
type Case = {
  label: string; generations: string[]; config: Record<string, unknown> | null; stampedUnit: string | null;
  effectRows: StoredRow[];
  expected: {
    enginePairing: EnginePairing | null; pairedItems: Record<string, number>;
    samplesPerItem: Record<string, SamplesPerItem>; rows: ExpectedRow[];
  };
};

const fixture = JSON.parse(readFileSync(resolve(checkout, "Tests", "Fixtures", "cross-engine", "effect-units.json"), "utf8")) as {
  minimumPairs: number;
  wording: { responseCaveat: string; unknownNote: string; responseUnitExplanation: string; unitLines: Record<string, string> };
  cases: Case[];
};

const caseNamed = (label: string) => {
  const found = fixture.cases.find((entry) => entry.label === label);
  if (!found) throw new Error(`no fixture case "${label}"`);
  return found;
};

/// A stored row as the loader builds it from effect-sizes.csv.
const effectRow = (stored: StoredRow): Effect => {
  const base = {
    condition: stored.condition, endpoint: stored.endpoint, short: stored.endpoint, estimate: 1, low: 0.5, high: 1.5,
    unit: "Δ units", n: strictNumber(stored.n), q: null, p: null, correction: "", direction: "positive" as const,
    stratifyBy: stored.stratifyBy, stratum: stored.stratum, pairedUnit: stored.unit, estimand: "", inference: "",
  };
  return { ...base, key: effectKey(base) };
};

/// A generations file as the browser hands it over.
const fileOf = (text: string): RunFile => {
  const blob = new File([text], "generations.jsonl");
  return { name: "generations.jsonl", path: "generations.jsonl", size: blob.size, modified: 0, handle: { getFile: async () => blob } } as unknown as RunFile;
};

const contextOf = (entry: Case, stored: StoredRow) =>
  ({ stratifyBy: stored.stratifyBy, metric: stored.endpoint, pairing: enginePairing(entry.config) });

const settled = (label: string, index = 0): Effect => {
  const entry = caseNamed(label);
  const stored = entry.effectRows[index];
  const row = effectRow(stored);
  const records = pairedRecordsFromText(entry.generations.join("\n"), true);
  return { ...row, effectUnit: effectUnit(row.condition, row.n, row.pairedUnit, entry.stampedUnit ?? "", records, contextOf(entry, stored)) };
};

describe("the shared effect-unit fixture", () => {
  it("carries the cases the rule is defined by", () => {
    const labels = fixture.cases.map((entry) => entry.label);
    for (const label of [
      "one prompt, five seeds", "four items, two seeds", "a current item-level analysis of four items, two seeds",
      "the same item-level rows with no engine record",
      "the reviewer's exclusion case, made by the Mac engine before 0.9.7",
      "the reviewer's exclusion case, with no engine version recorded", "a subgroup row within the run's items",
      "an ordinal readout row made by the Mac engine before 0.9.7",
      "a condition the records cannot place", "records a reader leaves out", "a unit the analysis stamped",
    ]) expect(labels).toContain(label);
  });

  for (const entry of fixture.cases) {
    it(`settles as the Python reader does: ${entry.label}`, async () => {
      const text = `${entry.generations.join("\n")}\n`;
      const counts = pairedItemsFromText(text);
      expect(counts).toEqual(entry.expected.pairedItems);
      const records = pairedRecordsFromText(text, true);
      expect(records.samplesPerItem).toEqual(entry.expected.samplesPerItem);
      // The whole-file reader, with windows small enough to split lines.
      expect(await countPairedItems(fileOf(text), 7)).toEqual(entry.expected.pairedItems);
      expect(await countPairedRecords(fileOf(text), 7)).toEqual(records);
      expect(enginePairing(entry.config)).toBe(entry.expected.enginePairing);
      expect(entry.effectRows).toHaveLength(entry.expected.rows.length);
      entry.effectRows.forEach((stored, index) => {
        const expected = entry.expected.rows[index];
        const row = effectRow(stored);
        const unit = effectUnit(row.condition, row.n, row.pairedUnit, entry.stampedUnit ?? "", records, contextOf(entry, stored));
        const where = `${entry.label}: ${stored.stratifyBy} ${stored.stratum} ${stored.endpoint}`;
        expect([where, unit]).toEqual([where, { unit: expected.unit, source: expected.unitSource, pairedItems: expected.pairedItems }]);
        const withUnit = { ...row, effectUnit: unit };
        expect([where, countPhrase(withUnit)]).toEqual([where, expected.count]);
        expect([where, hasTooFewPairs(withUnit)]).toEqual([where, expected.tooFew]);
      });
    });
  }

  it("uses the Python reader's words", () => {
    expect(MINIMUM_PAIRS).toBe(fixture.minimumPairs);
    expect(RESPONSE_CAVEAT).toBe(fixture.wording.responseCaveat);
    expect(UNKNOWN_UNIT_NOTE).toBe(fixture.wording.unknownNote);
    expect(RESPONSE_UNIT_EXPLANATION).toBe(fixture.wording.responseUnitExplanation);
    const records = { counts: { formal: 4 }, complete: true };
    for (const [source, unit, key] of [
      ["one_sample_per_item", "item", "one_sample_per_item"], ["analysis_engine", "item", "analysis_engine_item"],
      ["analysis_engine", "response", "analysis_engine_response"], ["inferred_from_records", "response", "inferred_from_records"],
      ["not_established", "unknown", "not_established"],
    ] as const) {
      const effect: EffectUnit = { unit, source, pairedItems: 4 };
      expect(UNSTAMPED_UNIT_LINES[key]).toBe(fixture.wording.unitLines[key]);
      for (const settledAgainst of [records, null]) {
        expect(unitLines([{ effectUnit: effect }], settledAgainst)).toEqual([`Unit of analysis: ${fixture.wording.unitLines[key]}`]);
      }
    }
    // One Mac analysis from before 0.9.7 holds item rows (its ordinal
    // readouts) and response rows: one line for each.
    expect(unitLines([
      { effectUnit: { unit: "item", source: "analysis_engine", pairedItems: 4 } },
      { effectUnit: { unit: "response", source: "analysis_engine", pairedItems: 4 } },
    ], records)).toEqual([
      `Unit of analysis for the rows marked item: ${fixture.wording.unitLines.analysis_engine_item}`,
      `Unit of analysis for the rows marked response: ${fixture.wording.unitLines.analysis_engine_response}`,
    ]);
    // A recorded unit is defined, not only named.
    for (const unit of ["item", "transcript"]) {
      const effect: EffectUnit = { unit, source: "recorded", pairedItems: null };
      expect(unitLines([{ effectUnit: effect }], records)).toEqual([`Unit of analysis: ${fixture.wording.unitLines[`recorded_${unit}`]}`]);
    }
  });
});

describe("a row's count, in its own unit", () => {
  it("reads five seeds of one prompt as responses from one item, too few for an interval", () => {
    const row = settled("one prompt, five seeds");
    expect(pairedCountLabel(row)).toBe("n = 5 paired responses from 1 item");
    expect(pairedCountLabel(row)).not.toMatch(/paired items?\b/);
    expect(hasTooFewPairs(row)).toBe(true);
    expect(intervalLine(row, fmt)).toBe("too few items for a confidence interval (n = 5 paired responses from 1 item; at least 3 are needed)");
    expect(unitCaveat(row)).toBe(`${RESPONSE_CAVEAT}.`);
    expect(unitLabel(row)).toBe("response (from the records)");
  });

  it("reads four items with two seeds each, n = 8, as 8 paired responses from at most 4 items", () => {
    const row = settled("four items, two seeds");
    expect(pairedCountLabel(row)).toBe("n = 8 paired responses from at most 4 items");
    expect(hasTooFewPairs(row)).toBe(false);
    expect(intervalLine(row, fmt)).toBe("95% CI +0.50 to +1.50 · n = 8 paired responses from at most 4 items");
  });

  it("reads a row the Mac engine 0.9.7 made as items, because its config.json says so", () => {
    const row = settled("a current item-level analysis of four items, two seeds");
    expect(pairedCountLabel(row)).toBe("n = 4 paired items");
    expect(unitCaveat(row)).toBe("");
    expect(unitLabel(row)).toBe("item (from the engine record)");
    expect(effectsUnitSummary([row], "")).toBe("The item—not the generation—is the unit of analysis, except where a stratum says otherwise.");
  });

  it("does not call fewer pairs than items items when nothing records which", () => {
    const row = settled("the same item-level rows with no engine record");
    expect(pairedCountLabel(row)).toBe("n = 4 pairs");
    expect(unitCaveat(row)).toBe("The analysis that made this row did not record whether its pairs count items or responses.");
    expect(unitLabel(row)).toBe("unknown");
    expect(effectsUnitSummary([row], "")).toBe("The analysis that made these rows did not record whether their pairs count items or responses. The numbers are shown as stored; the unit of analysis below says more.");
  });

  it("reads the external reviewer's exclusion case as responses when the Mac engine before 0.9.7 made it", () => {
    const row = settled("the reviewer's exclusion case, made by the Mac engine before 0.9.7");
    expect(pairedCountLabel(row)).toBe("n = 5 paired responses from at most 10 items");
    expect(unitLabel(row)).toBe("response (from the engine record)");
    expect(unitCaveat(row)).toBe(`${RESPONSE_CAVEAT}.`);
    expect(pairedCountLabel(settled("the reviewer's exclusion case, with no engine version recorded"))).toBe("n = 5 pairs");
  });

  it("says plain pairs when the records cannot place a condition", () => {
    const row = settled("a condition the records cannot place", 1);
    expect(pairedCountLabel(row)).toBe("n = 3 pairs");
    expect(unitCaveat(row)).toBe(`${UNKNOWN_UNIT_NOTE}.`);
  });
});

describe("which engine made a table", () => {
  it("is read from config.json's substrate and appVersion", () => {
    const cases: [Record<string, unknown> | null, EnginePairing | null][] = [
      [{ substrate: "python-hf-transformers" }, "items"],
      [{ substrate: "python-hf-transformers", appVersion: "steerlab-server 0.1.0" }, "items"],
      [{ substrate: "swift-mlx", appVersion: "swift-app 0.9.7" }, "items"],
      [{ substrate: "swift-mlx", appVersion: "swift-app 0.9.8+0a1b2c3d" }, "items"],
      [{ substrate: "swift-mlx", appVersion: "swift-app 1.0.0-beta+0a1b2c3d" }, "items"],
      [{ substrate: "swift-mlx", appVersion: "swift-app 0.10.0" }, "items"],
      [{ substrate: "swift-mlx", appVersion: "swift-app 0.9.6" }, "responses"],
      [{ substrate: "swift-mlx", appVersion: "swift-app 0.9.0-dev+abc12345" }, "responses"],
      [{ substrate: "swift-mlx", appVersion: "swift-app 0.9.7-dev+abc12345" }, null],
      [{ substrate: "swift-mlx", appVersion: "swift-app (fixture)" }, null],
      [{ substrate: "swift-mlx", appVersion: 97 }, null],
      [{ substrate: "swift-mlx" }, null],
      [{ substrate: "something-else", appVersion: "swift-app 0.9.8" }, null],
      [{}, null],
      [null, null],
    ];
    for (const [config, pairing] of cases) expect([config, enginePairing(config)]).toEqual([config, pairing]);
  });
});

describe("settling a run's rows", () => {
  const rows = caseNamed("four items, two seeds").effectRows.map(effectRow);

  it("asks for the records only when a row's unit is not stamped", async () => {
    let asked = 0;
    const records = async () => { asked += 1; return { counts: { formal: 4 }, complete: true }; };
    const stamped = await resolveUnits(rows, "transcript", records);
    expect(asked).toBe(0);
    expect(stamped.rows[0].effectUnit).toEqual({ unit: "transcript", source: "recorded", pairedItems: null });
    expect(stamped.pairedItems).toBeNull();
    const read = await resolveUnits(rows, "", records);
    expect(asked).toBe(1);
    expect(read.rows[0].effectUnit?.unit).toBe("response");
  });

  it("passes the engine's pairing through to every row", async () => {
    const records = async () => pairedRecordsFromText(caseNamed("four items, two seeds").generations.join("\n"), true);
    const settledRows = await resolveUnits(rows, "", records, "responses");
    expect(settledRows.rows[0].effectUnit).toEqual({ unit: "response", source: "analysis_engine", pairedItems: 4 });
  });

  it("settles from part of the records only what part of them can", () => {
    // A head: two items under each arm, each answered twice. Its counts are
    // lower bounds, so it settles nothing alone; an engine record does.
    const partial = { counts: { formal: 2 }, complete: false, samplesPerItem: { baseline: { responses: 2, readouts: 0 }, formal: { responses: 2, readouts: 0 } } };
    expect(effectUnit("formal", 2, "", "", partial)).toEqual({ unit: "unknown", source: "not_established", pairedItems: 2 });
    expect(effectUnit("formal", 4, "", "", partial)).toEqual({ unit: "unknown", source: "not_established", pairedItems: 2 });
    expect(effectUnit("formal", 2, "", "", partial, { pairing: "items" })).toEqual({ unit: "item", source: "analysis_engine", pairedItems: 2 });
    // Several responses already seen are several responses; how many items
    // they come from, a head cannot say.
    expect(effectUnit("formal", 4, "", "", partial, { pairing: "responses" })).toEqual({ unit: "response", source: "analysis_engine", pairedItems: null });
  });

  it("claims no unit for a row nothing settled", () => {
    expect(pairedCountLabel(rows[0])).toBe("n = 8 pairs");
    expect(unitLines([rows[0]], null)).toEqual([`Unit of analysis: ${fixture.wording.unitLines.not_established}`]);
  });

  it("decodes a character split between two reads", async () => {
    const lines = ["baseline", "formal"].map((condition) => JSON.stringify({ condition, promptID: "élan-ö", output: "Ça va." }));
    expect(await countPairedItems(fileOf(`${lines.join("\n")}\n`), 3)).toEqual({ formal: 1 });
  });
});
