import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { effectKey } from "../app/lib/effects";
import { fmt } from "../app/lib/format";
import {
  correctionFamilySize, hasTooFewPairs, headlineRows, intervalLine, outcomeTitle, pLine, plainPhrase,
  selectHeadline, tierOf,
} from "../app/lib/headline";
import { headlineOutcomeData } from "../app/lib/headlineOutcomes.generated";
import type { Effect } from "../app/lib/types";

// The headline outcome: which outcome leads the overview's headline card, and
// which rule chose it. The cases are the SHARED fixture the Python engine
// (Server/tests/test_headline_outcome.py) and the Mac engine
// (HeadlineOutcomeTests.swift) read too, so a rule that moves in one codebase
// and not the others fails where it did not move.

const checkout = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const readJSON = (...parts: string[]) => JSON.parse(readFileSync(resolve(checkout, ...parts), "utf8"));

type SelectionCase = {
  label: string;
  declared: string | null;
  analysisOutcomes: string[];
  evaluationOutcomes: string[];
  expected: { outcome: string | null; tier: string | null; rule: string | null; source: string | null; declaredAbsent: boolean; chosenBy: string };
};

const fixture = readJSON("Tests", "Fixtures", "cross-engine", "headline-outcome.json") as {
  selection: SelectionCase[];
  tiers: Array<{ name: string; tier: string }>;
  phrases: Array<{ name: string; plain: string | null }>;
};

describe("the shared headline fixture", () => {
  it("covers the cases the rule is defined by", () => {
    const labels = fixture.selection.map((entry) => entry.label);
    for (const label of [
      "declared by the researcher", "judged present", "choice outcome present", "numeric outcome present",
      "only marker density", "only surface measures", "declared outcome absent from the run",
    ]) expect(labels).toContain(label);
  });

  for (const entry of fixture.selection) {
    it(`selects: ${entry.label}`, () => {
      const headline = selectHeadline(entry.declared, entry.analysisOutcomes, entry.evaluationOutcomes);
      expect(headline.outcome).toBe(entry.expected.outcome);
      expect(headline.tier).toBe(entry.expected.tier);
      expect(headline.rule).toBe(entry.expected.rule);
      expect(headline.source).toBe(entry.expected.source);
      expect(headline.declaredAbsent).toBe(entry.expected.declaredAbsent);
      expect(headline.chosenBy).toBe(entry.expected.chosenBy);
    });
  }

  it("puts every outcome name in its tier", () => {
    for (const entry of fixture.tiers) expect([entry.name, tierOf(entry.name)]).toEqual([entry.name, entry.tier]);
  });

  it("uses the shared plain phrases", () => {
    for (const entry of fixture.phrases) expect([entry.name, plainPhrase(entry.name)]).toEqual([entry.name, entry.plain]);
  });
});

describe("the generated mapping", () => {
  it("is the file both engines read", () => {
    // scripts/ci/check-headline-outcomes.py checks the same thing; this is
    // the check that runs with the suite.
    const shipped = readJSON("Server", "steerlab_server", "client", "resources", "headline-outcomes.json");
    expect(headlineOutcomeData).toEqual(shipped);
  });
});

describe("a declaration of the wrong type", () => {
  it("reads as no declaration", () => {
    for (const value of [undefined, null, 7, "", "   ", ["choiceRate"], { outcome: "choiceRate" }]) {
      const headline = selectHeadline(value, ["wordCount", "choiceRate"]);
      expect(headline.outcome).toBe("choiceRate");
      expect(headline.rule).toBe("defaultOrder");
      expect(headline.declaredAbsent).toBe(false);
    }
  });
});

// --- what the card prints ------------------------------------------------------

const effect = (overrides: Partial<Effect> & { condition: string; endpoint: string }): Effect => {
  const base = {
    short: overrides.endpoint, estimate: 0.31, low: 0.12, high: 0.48,
    unit: "Δ probability", n: 24 as number | null, q: 0.012 as number | null,
    p: 0.012 as number | null, correction: "bh",
    direction: "positive" as const, stratifyBy: "pooled", stratum: "",
    pairedUnit: "", estimand: "", inference: "",
    ...overrides,
  };
  return { ...base, key: effectKey(base) };
};

describe("the headline card", () => {
  const wordCount = effect({ condition: "steered", endpoint: "wordCount" });
  const choice = effect({ condition: "steered", endpoint: "choiceLogOdds" });
  const choiceB = effect({ condition: "steered-strong", endpoint: "choiceLogOdds" });

  it("no longer leads with the first row of the table", () => {
    // The engine writes word count first; the card leads with the choice.
    const pooled = [wordCount, choice];
    const headline = selectHeadline(null, pooled.map((row) => row.endpoint));
    expect(headline.outcome).toBe("choiceLogOdds");
    expect(headline.chosenBy).toBe("chosen by default order");
    expect(headlineRows(headline, pooled)).toEqual([choice]);
    expect(outcomeTitle("choiceLogOdds")).toBe("the target option's log odds (choiceLogOdds)");
    expect(outcomeTitle("aNewOutcome")).toBe("aNewOutcome");
  });

  it("shows one row per condition for the headline outcome, in file order", () => {
    const pooled = [wordCount, choice, choiceB];
    const headline = selectHeadline(null, pooled.map((row) => row.endpoint));
    expect(headlineRows(headline, pooled)).toEqual([choice, choiceB]);
  });

  it("has no table row for a judged headline", () => {
    const headline = selectHeadline(null, ["wordCount"], ["judged"]);
    expect(headline.source).toBe("evaluationReport");
    expect(headlineRows(headline, [wordCount])).toEqual([]);
  });

  it("says adjusted only when the correction covered more than one comparison", () => {
    // One treatment condition: the family has one member, so the adjusted p
    // equals the raw one and nothing was corrected.
    expect(correctionFamilySize(choice, [wordCount, choice])).toBe(1);
    expect(pLine(choice, [wordCount, choice])).toBe("p 0.012 (one comparison, so no correction applies)");
    // Two conditions: a real correction, named.
    expect(correctionFamilySize(choice, [wordCount, choice, choiceB])).toBe(2);
    expect(pLine(choice, [wordCount, choice, choiceB])).toBe("adjusted p 0.012 (bh, 2 comparisons)");
    // No adjusted p at all: never presented as adjusted.
    const raw = effect({ condition: "steered", endpoint: "choiceRate", q: null, p: 0.2 });
    expect(pLine(raw, [raw])).toBe("uncorrected p 0.20");
    const none = effect({ condition: "steered", endpoint: "choiceRate", q: null, p: null });
    expect(pLine(none, [none])).toBe("p not reported");
  });

  it("prints no interval for fewer than three pairs", () => {
    for (const n of [1, 2]) {
      const tiny = effect({ condition: "steered", endpoint: "choiceLogOdds", n });
      expect(hasTooFewPairs(tiny)).toBe(true);
      expect(intervalLine(tiny, fmt)).toBe(`too few pairs for a confidence interval (n = ${n}; at least 3 are needed)`);
      expect(intervalLine(tiny, fmt)).not.toContain("95% CI");
      expect(pLine(tiny, [tiny])).toBe("");
    }
    const three = effect({ condition: "steered", endpoint: "choiceLogOdds", n: 3 });
    expect(hasTooFewPairs(three)).toBe(false);
    expect(intervalLine(three, fmt)).toBe("95% CI +0.12 to +0.48 · n = 3");
    // A table that carried no n says so; it is not "too few pairs".
    const unstamped = effect({ condition: "steered", endpoint: "choiceLogOdds", n: null });
    expect(hasTooFewPairs(unstamped)).toBe(false);
    expect(intervalLine(unstamped, fmt)).toBe("95% CI +0.12 to +0.48 · n not reported");
  });
});
