import { describe, expect, it } from "vitest";
import {
  effectConditions, effectEndpoints, effectKey, estimandLabel, groupEffects,
  isDiagnostic, pairedCountLabel, stratumLabel,
} from "../app/lib/effects";
import type { Effect } from "../app/lib/types";

const effect = (overrides: Partial<Effect> & { condition: string; endpoint: string }): Effect => {
  const base = {
    short: overrides.endpoint, estimate: 0.1, low: 0, high: 0.2,
    unit: "Δ probability", n: 12 as number | null, q: null as number | null,
    p: null as number | null, correction: "",
    direction: "positive" as const, stratifyBy: "pooled", stratum: "",
    pairedUnit: "", estimand: "", inference: "",
    ...overrides,
  };
  return { ...base, key: effectKey(base) };
};

describe("effectKey", () => {
  it("separates two conditions' rows for the same endpoint", () => {
    const a = effect({ condition: "agent-a", endpoint: "choiceRate" });
    const b = effect({ condition: "agent-b", endpoint: "choiceRate" });
    expect(a.key).not.toBe(b.key);
  });

  it("separates a stratified row from its pooled parent", () => {
    const pooled = effect({ condition: "steered", endpoint: "choiceRate" });
    const stratum = effect({ condition: "steered", endpoint: "choiceRate", stratifyBy: "promptID", stratum: "loan" });
    expect(pooled.key).not.toBe(stratum.key);
  });

  it("cannot be spoofed by a field that reads like a joined key", () => {
    // The separator is a control character no CSV field the engines write can
    // contain, so no two distinct rows collapse onto one key.
    const a = effect({ condition: "a", endpoint: "b-c" });
    const b = effect({ condition: "a-b", endpoint: "c" });
    expect(a.key).not.toBe(b.key);
  });
});

describe("groupEffects", () => {
  const pooledA = effect({ condition: "agent-a", endpoint: "choiceRate" });
  const stratumA = effect({ condition: "agent-a", endpoint: "choiceRate", stratifyBy: "promptID", stratum: "loan", pairedUnit: "sample", estimand: "withinItemSamples", inference: "diagnostic" });
  const pooledB = effect({ condition: "agent-b", endpoint: "choiceRate" });

  it("nests stratified rows under the pooled row of their own condition", () => {
    const groups = groupEffects([pooledA, stratumA, pooledB]);
    expect(groups).toHaveLength(2);
    expect(groups[0].condition).toBe("agent-a");
    expect(groups[0].pooled).toBe(pooledA);
    expect(groups[0].strata).toEqual([stratumA]);
    // agent-b's group must not adopt agent-a's stratum.
    expect(groups[1].pooled).toBe(pooledB);
    expect(groups[1].strata).toEqual([]);
  });

  it("preserves file order of both groups and strata", () => {
    const second = effect({ condition: "agent-a", endpoint: "choiceRate", stratifyBy: "arm", stratum: "notLegal" });
    const groups = groupEffects([pooledA, stratumA, second]);
    expect(groups[0].strata.map((row) => row.stratum)).toEqual(["loan", "notLegal"]);
  });

  it("still shows strata whose pooled parent is missing from the table", () => {
    // Nothing is dropped: a filtered or malformed table must not silently
    // lose rows the file contains.
    const groups = groupEffects([stratumA]);
    expect(groups).toHaveLength(1);
    expect(groups[0].pooled).toBeNull();
    expect(groups[0].strata).toEqual([stratumA]);
  });

  it("keeps a duplicate pooled row rather than replacing the first", () => {
    const duplicate = effect({ condition: "agent-a", endpoint: "choiceRate", estimate: 0.9 });
    const groups = groupEffects([pooledA, duplicate]);
    expect(groups[0].pooled).toBe(pooledA);
    expect(groups[0].strata).toEqual([duplicate]);
  });
});

describe("filter options", () => {
  it("lists conditions in file order, without blanks", () => {
    const rows = [
      effect({ condition: "agent-b", endpoint: "x" }),
      effect({ condition: "agent-a", endpoint: "x" }),
      effect({ condition: "agent-b", endpoint: "y" }),
      effect({ condition: "", endpoint: "y" }),
    ];
    expect(effectConditions(rows)).toEqual(["agent-b", "agent-a"]);
  });

  it("de-duplicates endpoints across conditions", () => {
    const rows = [
      effect({ condition: "agent-a", endpoint: "choiceRate" }),
      effect({ condition: "agent-b", endpoint: "choiceRate" }),
    ];
    expect(effectEndpoints(rows)).toEqual(["choiceRate"]);
  });
});

describe("estimand labelling", () => {
  it("marks a within-item row diagnostic from any of the three stamps", () => {
    expect(isDiagnostic(effect({ condition: "c", endpoint: "e", inference: "diagnostic" }))).toBe(true);
    expect(isDiagnostic(effect({ condition: "c", endpoint: "e", estimand: "withinItemSamples" }))).toBe(true);
    // An engine that has not grown the estimand columns still states the same
    // fact in `unit`; its rows must be demoted too, not shown as corrected
    // cross-item findings.
    expect(isDiagnostic(effect({ condition: "c", endpoint: "e", stratifyBy: "promptID", stratum: "p1", pairedUnit: "sample" }))).toBe(true);
    expect(isDiagnostic(effect({ condition: "c", endpoint: "e", stratifyBy: "promptID", stratum: "p1", pairedUnit: "item" }))).toBe(false);
    expect(isDiagnostic(effect({ condition: "c", endpoint: "e", estimand: "itemLevel", inference: "corrected" }))).toBe(false);
  });

  it("says nothing about the estimand of a table that stamped none", () => {
    // A pre-2026-08-06 file carries no estimand column; the view must not
    // guess which one a stratified row meant.
    expect(estimandLabel(effect({ condition: "c", endpoint: "e", stratifyBy: "promptID", stratum: "p1" }))).toBe("");
    expect(estimandLabel(effect({ condition: "c", endpoint: "e", estimand: "itemLevel" }))).toBe("per-item differences");
    expect(estimandLabel(effect({ condition: "c", endpoint: "e", estimand: "withinItemSamples" }))).toContain("within one item");
  });

  it("names a stratum by its family and cell", () => {
    expect(stratumLabel(effect({ condition: "c", endpoint: "e", stratifyBy: "arm×caseID", stratum: "notLegal×loan" }))).toBe("arm×caseID · notLegal×loan");
    expect(stratumLabel(effect({ condition: "c", endpoint: "e" }))).toBe("pooled");
  });
});

describe("pairedCountLabel", () => {
  it("reads an absent n as not reported, never as zero", () => {
    expect(pairedCountLabel(effect({ condition: "c", endpoint: "e", n: null }))).toBe("n not reported");
  });

  it("names the paired unit the row's own column declares", () => {
    expect(pairedCountLabel(effect({ condition: "c", endpoint: "e", n: 12 }))).toBe("n = 12 paired items");
    expect(pairedCountLabel(effect({ condition: "c", endpoint: "e", n: 50, pairedUnit: "sample" }))).toBe("n = 50 paired samples");
  });

  it("keeps a genuine zero distinguishable from a missing count", () => {
    expect(pairedCountLabel(effect({ condition: "c", endpoint: "e", n: 0 }))).toBe("n = 0 paired items");
  });
});
