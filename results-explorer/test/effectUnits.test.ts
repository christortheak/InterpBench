import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { strictNumber } from "../app/lib/csv";
import { effectKey, pairedCountLabel } from "../app/lib/effects";
import {
  countPairedItems, countPhrase, effectsUnitSummary, effectUnit, MINIMUM_PAIRS, pairedItemsFromText,
  RESPONSE_CAVEAT, RESPONSE_UNIT_EXPLANATION, resolveUnits, UNKNOWN_UNIT_NOTE, unitCaveat, unitLabel, unitLines,
  type EffectUnit, type UnitSource,
} from "../app/lib/effectUnits";
import { fmt } from "../app/lib/format";
import { hasTooFewPairs, intervalLine } from "../app/lib/headline";
import type { Effect, RunFile } from "../app/lib/types";

// What one paired difference of a stored effect row is. Before SteerLab
// 0.9.7 the Mac engine paired every response of a multi-sample run with the
// baseline response to the same item and seed, counted responses in n, and
// stamped no unit, so five seeds of one prompt read as "5 paired items" with
// an interval. The cases are the SHARED fixture the Python reader
// (Server/tests/test_results_export.py) answers and the Mac app
// (EffectUnitTests.swift) reads too, so the three copies of the rule cannot
// drift apart unnoticed.

const checkout = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");

type StoredRow = { condition: string; endpoint: string; n: string; unit: string; stratifyBy: string; stratum: string };
type ExpectedRow = {
  condition: string; endpoint: string; stratifyBy: string; stratum: string;
  unit: string; unitSource: UnitSource; pairedItems: number | null; count: string | null; tooFew: boolean;
};
type Case = {
  label: string; generations: string[]; stampedUnit: string | null; effectRows: StoredRow[];
  expected: { pairedItems: Record<string, number>; rows: ExpectedRow[] };
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

const settled = (label: string, index = 0): Effect => {
  const entry = caseNamed(label);
  const row = effectRow(entry.effectRows[index]);
  const counts = pairedItemsFromText(entry.generations.join("\n"));
  return { ...row, effectUnit: effectUnit(row.condition, row.n, row.pairedUnit, entry.stampedUnit ?? "", { counts, complete: true }) };
};

describe("the shared effect-unit fixture", () => {
  it("carries the cases the rule is defined by", () => {
    const labels = fixture.cases.map((entry) => entry.label);
    for (const label of [
      "one prompt, five seeds", "four items, two seeds", "a current item-level analysis of four items, two seeds",
      "a condition the records cannot place", "records a reader leaves out", "a unit the analysis stamped",
    ]) expect(labels).toContain(label);
  });

  for (const entry of fixture.cases) {
    it(`settles as the Python reader does: ${entry.label}`, async () => {
      const text = `${entry.generations.join("\n")}\n`;
      const counts = pairedItemsFromText(text);
      expect(counts).toEqual(entry.expected.pairedItems);
      // The whole-file reader, with windows small enough to split lines.
      expect(await countPairedItems(fileOf(text), 7)).toEqual(entry.expected.pairedItems);
      expect(entry.effectRows).toHaveLength(entry.expected.rows.length);
      entry.effectRows.forEach((stored, index) => {
        const expected = entry.expected.rows[index];
        const row = effectRow(stored);
        const unit = effectUnit(row.condition, row.n, row.pairedUnit, entry.stampedUnit ?? "", { counts, complete: true });
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
    for (const [source, unit] of [["engine_default", "item"], ["inferred_from_records", "response"], ["not_established", "unknown"]] as const) {
      const effect: EffectUnit = { unit, source, pairedItems: 4 };
      expect(unitLines([{ effectUnit: effect }], records)).toEqual([`Unit of analysis: ${fixture.wording.unitLines[source]}`]);
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

  it("reads four items with two seeds each, n = 8, as 8 paired responses from 4 items", () => {
    const row = settled("four items, two seeds");
    expect(pairedCountLabel(row)).toBe("n = 8 paired responses from 4 items");
    expect(hasTooFewPairs(row)).toBe(false);
    expect(intervalLine(row, fmt)).toBe("95% CI +0.50 to +1.50 · n = 8 paired responses from 4 items");
  });

  it("leaves a current item-level row reading as items", () => {
    const row = settled("a current item-level analysis of four items, two seeds");
    expect(pairedCountLabel(row)).toBe("n = 4 paired items");
    expect(unitCaveat(row)).toBe("");
    expect(unitLabel(row)).toBe("item (default)");
    expect(effectsUnitSummary([row], "")).toBe("The item—not the generation—is the unit of analysis, except where a stratum says otherwise.");
  });

  it("says plain pairs when the records cannot place a condition", () => {
    const row = settled("a condition the records cannot place", 1);
    expect(pairedCountLabel(row)).toBe("n = 3 pairs");
    expect(unitCaveat(row)).toBe(`${UNKNOWN_UNIT_NOTE}.`);
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

  it("does not settle a row beyond a partial count", () => {
    const partial = { counts: { formal: 2 }, complete: false };
    expect(effectUnit("formal", 2, "", "", partial)).toEqual({ unit: "item", source: "engine_default", pairedItems: 2 });
    expect(effectUnit("formal", 4, "", "", partial)).toEqual({ unit: "unknown", source: "not_established", pairedItems: 2 });
    expect(unitLines([{ effectUnit: effectUnit("formal", 4, "", "", partial) }], partial)[0]).toContain("only the first part of the run's records");
  });

  it("claims no unit for a row nothing settled", () => {
    expect(pairedCountLabel(rows[0])).toBe("n = 8 pairs");
    expect(unitLines([rows[0]], null)).toEqual(["Unit of analysis: not established. The analysis did not stamp it, and the run's records are not available here."]);
  });

  it("decodes a character split between two reads", async () => {
    const lines = ["baseline", "formal"].map((condition) => JSON.stringify({ condition, promptID: "élan-ö", output: "Ça va." }));
    expect(await countPairedItems(fileOf(`${lines.join("\n")}\n`), 3)).toEqual({ formal: 1 });
  });
});
