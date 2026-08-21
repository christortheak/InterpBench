import { describe, expect, it } from "vitest";
import {
  buildInstrumentTable,
  instrumentRecordFor,
  loadEngineChoiceDeltas,
  findChoiceDeltaRuns,
  loadInstrumentRecords,
  promptIDFromGenerationID,
  readConditionRollups,
  requestGenerationRecord,
  resolveBaselineCondition,
  shortConditionLabels,
  splitOrderSuffix,
  takeGenerationRecord,
} from "../app/lib/instruments";
import type { InstrumentRecord } from "../app/lib/instruments";
import type { LocalFileHandle, RunFile } from "../app/lib/types";

// In-memory stand-ins for the structural handle surface (File System Access
// in the browser, the native bridge when embedded), matching
// test/loaders.test.ts. NOTHING here reads a real workspace: every byte
// below is declared in this file. The shapes mirror the engine's
// answer-token instrument contract, but the text is invented.

const fakeFile = (name: string, body: string): File => {
  const bytes = new TextEncoder().encode(body);
  const like = {
    name,
    size: bytes.byteLength,
    lastModified: 0,
    type: "",
    text: async () => body,
    arrayBuffer: async () => bytes.buffer,
    slice: (start?: number, end?: number) => ({ text: async () => body.slice(start ?? 0, end) }),
  };
  return like as unknown as File;
};

const fakeFileHandle = (name: string, body: string): LocalFileHandle => ({ kind: "file", name, getFile: async () => fakeFile(name, body) });

const fakeRunFile = (name: string, body: string): RunFile => ({
  name,
  path: name,
  size: new TextEncoder().encode(body).byteLength,
  modified: 0,
  handle: fakeFileHandle(name, body),
});

const fakeRun = (contents: Record<string, string>) => ({ files: Object.entries(contents).map(([name, body]) => fakeRunFile(name, body)) });

// One instrument record. `logOdds` carries the whole readout the joins use;
// the rest is contract padding so the parser sees a realistic row.
const instrumentLine = (values: {
  condition: string;
  promptID: string;
  promptIndex?: number;
  target?: string;
  selected: string;
  logOdds: Record<string, number>;
  instrument?: string;
}) => JSON.stringify({
  instrument: values.instrument ?? "answerTokenLogprob",
  condition: values.condition,
  promptID: values.promptID,
  promptIndex: values.promptIndex ?? 0,
  caseID: values.promptID.split("-")[0],
  arm: "legal+noReasons",
  prompt: "A declared prompt.",
  options: ["A", "B"],
  optionLogprobs: { A: values.logOdds.A ?? 0, B: values.logOdds.B ?? 0 },
  optionMeanTokenLogprobs: { A: -1, B: -2 },
  optionTokenCounts: { A: 1, B: 1 },
  optionTokenIDs: { A: [1], B: [2] },
  optionTokenLogprobs: { A: [-1], B: [-2] },
  choiceProbability: { A: 0.4, B: 0.6 },
  logOdds: values.logOdds,
  margin: 1.5,
  selected: values.selected,
  target: values.target ?? "B",
  optionLengthRatio: 1,
  interventionState: { slots: [{ concept: "fear", layer: 31, alpha: 0.2 }], variant: values.condition, controlType: null, bandWidth: 1, alphaInNormUnits: true, adapters: [] },
  modelID: "declared/model",
  modelRevision: "abc123",
  temperature: 0,
  doSample: false,
  experiment: "fixture",
  experimentHash: "hash",
});

const sampledLine = (condition: string, promptID: string) => JSON.stringify({ condition, promptID, sampleIndex: 0, output: "Sampled prose.", wordCount: 2 });

// The motivating shape (2026-08-05): a pair at ceiling in both orders, a
// near-boundary pair whose -ab order flips under one condition, and one
// item that carries an order suffix with no mirror.
const jsonl = [
  sampledLine("baseline", "loan-legal-noreasons-ab"),
  instrumentLine({ condition: "baseline", promptID: "jewels-legal-noreasons-ab", promptIndex: 0, selected: "B", logOdds: { A: -27.6, B: 27.6 } }),
  instrumentLine({ condition: "baseline", promptID: "jewels-legal-noreasons-ba", promptIndex: 1, target: "A", selected: "A", logOdds: { A: 27.6, B: -27.6 } }),
  instrumentLine({ condition: "baseline", promptID: "loan-legal-noreasons-ab", promptIndex: 2, selected: "B", logOdds: { A: -2.5, B: 2.5 } }),
  instrumentLine({ condition: "baseline", promptID: "loan-legal-noreasons-ba", promptIndex: 3, target: "A", selected: "A", logOdds: { A: 15.5, B: -15.5 } }),
  instrumentLine({ condition: "baseline", promptID: "orphan-legal-noreasons-ab", promptIndex: 4, selected: "B", logOdds: { A: -4, B: 4 } }),
  instrumentLine({ condition: "fear-agent", promptID: "jewels-legal-noreasons-ab", promptIndex: 0, selected: "B", logOdds: { A: -26.6, B: 26.6 } }),
  instrumentLine({ condition: "fear-agent", promptID: "jewels-legal-noreasons-ba", promptIndex: 1, target: "A", selected: "A", logOdds: { A: 27.6, B: -27.6 } }),
  instrumentLine({ condition: "fear-agent", promptID: "loan-legal-noreasons-ab", promptIndex: 2, selected: "A", logOdds: { A: 8, B: -8 } }),
  instrumentLine({ condition: "fear-agent", promptID: "loan-legal-noreasons-ba", promptIndex: 3, target: "A", selected: "A", logOdds: { A: 13.25, B: -13.25 } }),
  instrumentLine({ condition: "fear-agent", promptID: "orphan-legal-noreasons-ab", promptIndex: 4, selected: "B", logOdds: { A: -4, B: 4 } }),
].join("\n");

const loadFixture = () => loadInstrumentRecords(fakeRun({ "generations.jsonl": jsonl }));

describe("loadInstrumentRecords", () => {
  it("reads only the instrument records out of generations.jsonl and counts the rest", async () => {
    const load = await loadFixture();
    expect(load.present).toBe(true);
    expect(load.records).toHaveLength(10);
    expect(load.sampledRecords).toBe(1);
    expect(load.skipped).toBe(0);
    expect(load.truncated).toBe(false);
    expect(load.file?.name).toBe("generations.jsonl");
  });

  it("keeps the whole stored readout, not just the selection", async () => {
    const row = (await loadFixture()).records[0];
    expect(row.instrument).toBe("answerTokenLogprob");
    expect(row.options).toEqual(["A", "B"]);
    expect(row.logOdds).toEqual({ A: -27.6, B: 27.6 });
    expect(row.choiceProbability.B).toBe(0.6);
    expect(row.optionTokenLogprobs.A).toEqual([-1]);
    expect(row.margin).toBe(1.5);
    expect(row.target).toBe("B");
    expect(row.interventionState?.slots).toEqual([{ concept: "fear", layer: 31, alpha: 0.2 }]);
    expect(row.line).toBe(2);
  });

  it("skips malformed instrument rows instead of crashing, and reports the count", async () => {
    const body = [
      "{not json at all",
      JSON.stringify({ instrument: "answerTokenLogprob", condition: "baseline" }), // no promptID/selected/options
      JSON.stringify({ instrument: "answerTokenLogprob", promptID: "x", selected: "A", options: [], condition: "baseline" }), // empty options
      instrumentLine({ condition: "baseline", promptID: "good-ab", selected: "B", logOdds: { A: -1, B: 1 } }),
      "",
    ].join("\n");
    const load = await loadInstrumentRecords(fakeRun({ "generations.jsonl": body }));
    expect(load.records.map((row) => row.promptID)).toEqual(["good-ab"]);
    expect(load.skipped).toBe(3);
  });

  it("tolerates a record whose numeric maps are the wrong shape", async () => {
    const body = JSON.stringify({
      instrument: "answerTokenLogprob", condition: "baseline", promptID: "odd-ab", selected: "A", target: "A",
      options: ["A", "B"], logOdds: "not a map", choiceProbability: { A: "high" }, margin: null, optionTokenIDs: { A: "x" },
    });
    const load = await loadInstrumentRecords(fakeRun({ "generations.jsonl": body }));
    expect(load.records).toHaveLength(1);
    expect(load.records[0].logOdds).toEqual({});
    expect(load.records[0].choiceProbability).toEqual({});
    expect(load.records[0].optionTokenIDs).toEqual({});
    expect(load.records[0].margin).toBeNull();
    // A record with no usable log-odds still builds a table — as blanks.
    const table = buildInstrumentTable(load.records);
    expect(table.items[0].baselineTargetLogOdds).toBeNull();
    expect(table.items[0].saturated).toBe(false);
  });

  it("reports an absent generations.jsonl as absent rather than empty", async () => {
    const load = await loadInstrumentRecords(fakeRun({}));
    expect(load.present).toBe(false);
    expect(load.records).toEqual([]);
    expect(load.file).toBeNull();
  });
});

describe("buildInstrumentTable — baseline join and Δ", () => {
  it("puts the baseline condition first and keeps items in prompt order", async () => {
    const table = buildInstrumentTable((await loadFixture()).records);
    expect(table.instrumentName).toBe("answerTokenLogprob");
    expect(table.baselineCondition).toBe("baseline");
    expect(table.conditions).toEqual(["baseline", "fear-agent"]);
    expect(table.items.map((item) => item.promptID)).toEqual([
      "jewels-legal-noreasons-ab", "jewels-legal-noreasons-ba",
      "loan-legal-noreasons-ab", "loan-legal-noreasons-ba",
      "orphan-legal-noreasons-ab",
    ]);
  });

  it("computes Δ as condition target log-odds minus the same-item baseline", async () => {
    const table = buildInstrumentTable((await loadFixture()).records);
    const boundary = table.items.find((item) => item.promptID === "loan-legal-noreasons-ab");
    expect(boundary?.baselineTargetLogOdds).toBe(2.5);
    expect(boundary?.cellByCondition["baseline"].targetLogOdds).toBe(2.5);
    // Target is B; the fear condition's B log-odds is −8 → Δ = −10.5.
    expect(boundary?.cellByCondition["fear-agent"].targetLogOdds).toBe(-8);
    expect(boundary?.cellByCondition["fear-agent"].delta).toBeCloseTo(-10.5, 10);
    // The baseline column never carries a Δ against itself.
    expect(boundary?.cellByCondition["baseline"].delta).toBeNull();
  });

  it("reads the target log-odds of each record's OWN declared target", async () => {
    const table = buildInstrumentTable((await loadFixture()).records);
    const mirrored = table.items.find((item) => item.promptID === "loan-legal-noreasons-ba");
    expect(mirrored?.target).toBe("A");
    expect(mirrored?.baselineTargetLogOdds).toBe(15.5);
    expect(mirrored?.cellByCondition["fear-agent"].delta).toBeCloseTo(-2.25, 10);
  });

  it("suppresses Δ when a condition record declares a different target than the baseline", () => {
    const records = [
      JSON.parse(instrumentLine({ condition: "baseline", promptID: "x-ab", selected: "B", target: "B", logOdds: { A: -1, B: 1 } })),
      JSON.parse(instrumentLine({ condition: "variant", promptID: "x-ab", selected: "A", target: "A", logOdds: { A: 3, B: -3 } })),
    ];
    const table = buildInstrumentTable(records.map((row, index) => ({ ...row, line: index + 1 })) as InstrumentRecord[]);
    const cell = table.items[0].cellByCondition["variant"];
    expect(cell.targetMismatch).toBe(true);
    expect(cell.delta).toBeNull();
  });

  it("leaves every Δ null when no condition is named baseline", () => {
    const records = [
      JSON.parse(instrumentLine({ condition: "arm-one", promptID: "x-ab", selected: "B", logOdds: { A: -1, B: 1 } })),
      JSON.parse(instrumentLine({ condition: "arm-two", promptID: "x-ab", selected: "A", logOdds: { A: 3, B: -3 } })),
    ].map((row, index) => ({ ...row, line: index + 1 })) as InstrumentRecord[];
    const table = buildInstrumentTable(records);
    expect(table.baselineCondition).toBe("");
    expect(table.items[0].cells.every((cell) => cell.delta === null)).toBe(true);
    expect(table.flips).toEqual([]);
    expect(table.items[0].saturated).toBe(false);
  });

  it("resolves the baseline condition by name, never by position", () => {
    expect(resolveBaselineCondition(["fear-agent", "baseline"])).toBe("baseline");
    expect(resolveBaselineCondition(["fear-agent", "study-baseline-arm"])).toBe("study-baseline-arm");
    expect(resolveBaselineCondition(["arm-one", "arm-two"])).toBe("");
  });
});

describe("buildInstrumentTable — flips", () => {
  it("flags the condition whose selection differs from its baseline record", async () => {
    const table = buildInstrumentTable((await loadFixture()).records);
    expect(table.flips).toHaveLength(1);
    expect(table.flips[0]).toMatchObject({ promptID: "loan-legal-noreasons-ab", condition: "fear-agent", from: "B", to: "A", saturated: false });
    expect(table.flips[0].delta).toBeCloseTo(-10.5, 10);
  });

  it("does not read an unchanged selection as a flip", async () => {
    const table = buildInstrumentTable((await loadFixture()).records);
    const held = table.items.find((item) => item.promptID === "jewels-legal-noreasons-ab");
    expect(held?.cellByCondition["fear-agent"].flipped).toBe(false);
  });
});

describe("buildInstrumentTable — counterbalance pairing (heuristic)", () => {
  it("splits the -ab / -ba suffix off a prompt id and leaves other ids alone", () => {
    expect(splitOrderSuffix("loan-legal-noreasons-ab")).toEqual({ stem: "loan-legal-noreasons", order: "ab" });
    expect(splitOrderSuffix("loan-legal-noreasons-ba")).toEqual({ stem: "loan-legal-noreasons", order: "ba" });
    expect(splitOrderSuffix("loan-legal-noreasons")).toEqual({ stem: "loan-legal-noreasons", order: null });
    // "ab" alone is a stem, not a suffix with an empty stem.
    expect(splitOrderSuffix("ab")).toEqual({ stem: "ab", order: null });
  });

  it("pairs the two orders of a stem and groups them together", async () => {
    const table = buildInstrumentTable((await loadFixture()).records);
    expect(table.pairs.map((pair) => pair.stem)).toEqual(["jewels-legal-noreasons", "loan-legal-noreasons"]);
    expect(table.groups.map((group) => group.items.map((item) => item.order))).toEqual([["ab", "ba"], ["ab", "ba"], ["ab"]]);
  });

  it("never invents a pair for a stem carrying only one order", async () => {
    const table = buildInstrumentTable((await loadFixture()).records);
    expect(table.pairs.some((pair) => pair.stem === "orphan-legal-noreasons")).toBe(false);
    expect(table.unpairedOrderedItems.map((item) => item.promptID)).toEqual(["orphan-legal-noreasons-ab"]);
    expect(table.groups.find((group) => group.stem === "orphan-legal-noreasons")?.pair).toBeNull();
  });

  it("reads opposite letters across the two orders as substantive consistency", async () => {
    const table = buildInstrumentTable((await loadFixture()).records);
    const jewels = table.pairs.find((pair) => pair.stem === "jewels-legal-noreasons");
    expect(jewels?.consistency).toEqual([
      { condition: "baseline", abSelected: "B", baSelected: "A", comparable: true, consistent: true },
      { condition: "fear-agent", abSelected: "B", baSelected: "A", comparable: true, consistent: true },
    ]);
    expect(jewels?.possibleOrderArtifact).toEqual([]);
  });

  it("flags the same letter in both orders as a possible order artifact", async () => {
    const table = buildInstrumentTable((await loadFixture()).records);
    const loan = table.pairs.find((pair) => pair.stem === "loan-legal-noreasons");
    // Baseline picks B then A (consistent); under fear both orders pick A.
    expect(loan?.consistency.find((entry) => entry.condition === "baseline")?.consistent).toBe(true);
    expect(loan?.consistency.find((entry) => entry.condition === "fear-agent")?.consistent).toBe(false);
    expect(loan?.possibleOrderArtifact).toEqual(["fear-agent"]);
  });
});

describe("buildInstrumentTable — saturation (heuristic)", () => {
  it("classifies items by |baseline target log-odds| against the threshold", async () => {
    const table = buildInstrumentTable((await loadFixture()).records);
    expect(table.saturationThreshold).toBe(10);
    expect(table.items.filter((item) => item.saturated).map((item) => item.promptID)).toEqual([
      "jewels-legal-noreasons-ab", "jewels-legal-noreasons-ba", "loan-legal-noreasons-ba",
    ]);
    expect(table.saturatedCount).toBe(3);
  });

  it("treats the threshold as inclusive at the boundary and honours a caller's value", async () => {
    const records = (await loadFixture()).records;
    // The mirrored loan item sits at exactly 15.5.
    const at = buildInstrumentTable(records, { saturationThreshold: 15.5 });
    expect(at.items.find((item) => item.promptID === "loan-legal-noreasons-ba")?.saturated).toBe(true);
    const above = buildInstrumentTable(records, { saturationThreshold: 15.51 });
    expect(above.items.find((item) => item.promptID === "loan-legal-noreasons-ba")?.saturated).toBe(false);
    expect(above.saturatedCount).toBe(2);
  });
});

describe("readConditionRollups", () => {
  it("reads the stored per-condition block with baseline first and missing keys null", () => {
    const rows = readConditionRollups({
      conditions: {
        "fear-agent": { choiceRate: 0.479, choiceReadouts: 12, generations: 48, agreementWithBaseline: { agreement: 0.9166, n: 60 } },
        baseline: { choiceRate: 0.52, choiceReadouts: 12, generations: 48 },
      },
    });
    expect(rows.map((row) => row.condition)).toEqual(["baseline", "fear-agent"]);
    expect(rows[0].agreementWithBaseline).toBeNull();
    expect(rows[1].agreementWithBaseline).toEqual({ agreement: 0.9166, n: 60 });
    expect(rows[1].choiceRate).toBe(0.479);
  });

  it("returns nothing when the report carries no conditions block", () => {
    expect(readConditionRollups({})).toEqual([]);
    expect(readConditionRollups({ conditions: ["not", "an", "object"] })).toEqual([]);
  });
});

describe("presentation helpers", () => {
  it("shortens condition labels by the prefix every variant shares", () => {
    const labels = shortConditionLabels(["baseline", "optimize-x-2026-08-03-fear-agent", "optimize-x-2026-08-03-efficiency-agent"], "baseline");
    expect(labels["baseline"]).toBe("baseline");
    expect(labels["optimize-x-2026-08-03-fear-agent"]).toBe("fear-agent");
    expect(labels["optimize-x-2026-08-03-efficiency-agent"]).toBe("efficiency-agent");
  });

  it("leaves a single variant's name intact rather than shortening it to nothing", () => {
    const labels = shortConditionLabels(["baseline", "fear-agent"], "baseline");
    expect(labels["fear-agent"]).toBe("fear-agent");
  });

  it("recovers a generation row's prompt id from the reader's composite id", () => {
    expect(promptIDFromGenerationID("loan-legal-noreasons-ab · S00")).toBe("loan-legal-noreasons-ab");
  });

  it("finds the one record for a (promptID, condition) pair", async () => {
    const records = (await loadFixture()).records;
    expect(instrumentRecordFor(records, "loan-legal-noreasons-ab", "fear-agent")?.selected).toBe("A");
    expect(instrumentRecordFor(records, "loan-legal-noreasons-ab", "absent-condition")).toBeNull();
  });

  it("hands a record request to the generations reader exactly once", () => {
    expect(takeGenerationRecord()).toBeNull();
    requestGenerationRecord({ promptID: "loan-legal-noreasons-ab", condition: "fear-agent" });
    expect(takeGenerationRecord()).toEqual({ promptID: "loan-legal-noreasons-ab", condition: "fear-agent" });
    expect(takeGenerationRecord()).toBeNull();
  });
});

describe("engine choice-deltas (analyze artifact preference)", () => {
  const sidecar = JSON.stringify({
    records: 36, skippedNoBaseline: 0, skippedNoTargetValue: 0,
    conditions: {
      "fear-agent": { n: 12, deltaTargetLogOddsMean: -0.85, ciLower: -2.1, ciUpper: 0.2, replicates: 10000, seed: 0, flipped: 1, skippedNoBaseline: 0, skippedNoTargetValue: 0 },
      "efficiency-agent": { n: 12, deltaTargetLogOddsMean: 0.71, ciLower: -0.4, ciUpper: 1.9, replicates: 10000, seed: 0, flipped: 0, skippedNoBaseline: 0, skippedNoTargetValue: 0 },
    },
  });

  it("finds analyze runs by sourceRun + artifact, newest first", () => {
    const found = findChoiceDeltaRuns("20260805T001709118-exp-run", [
      { name: "20260806T010000000-exp-analyze", sourceRun: "20260805T001709118-exp-run", artifacts: ["choice-deltas.json", "choice-deltas.csv"] },
      { name: "20260806T020000000-exp-analyze", sourceRun: "20260805T001709118-exp-run", artifacts: ["choice-deltas.json"] },
      { name: "20260806T030000000-exp-analyze", sourceRun: "some-other-run", artifacts: ["choice-deltas.json"] },
      { name: "20260806T040000000-exp-analyze", sourceRun: "20260805T001709118-exp-run", artifacts: ["effect-sizes.csv"] },
    ]);
    expect(found.map((entry) => entry.name)).toEqual([
      "20260806T020000000-exp-analyze", "20260806T010000000-exp-analyze",
    ]);
  });

  it("parses the sidecar's stored summaries, sorted by condition", async () => {
    const run = { name: "a", files: [fakeRunFile("choice-deltas.json", sidecar)] };
    const engine = await loadEngineChoiceDeltas(run);
    expect(engine?.summaries.map((s) => s.condition)).toEqual(["efficiency-agent", "fear-agent"]);
    expect(engine?.summaries[1].mean).toBe(-0.85);
    expect(engine?.summaries[1].ciLower).toBe(-2.1);
    expect(engine?.records).toBe(36);
  });

  it("refuses a malformed or absent sidecar with null, never a partial reading", async () => {
    expect(await loadEngineChoiceDeltas({ name: "a", files: [fakeRunFile("choice-deltas.json", "{not json")] })).toBeNull();
    expect(await loadEngineChoiceDeltas({ name: "a", files: [] })).toBeNull();
    expect(await loadEngineChoiceDeltas({ name: "a", files: [fakeRunFile("choice-deltas.json", JSON.stringify({ conditions: {} }))] })).toBeNull();
  });
});
