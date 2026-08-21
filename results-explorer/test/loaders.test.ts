import { describe, expect, it } from "vitest";
import { loadEffects, loadGenerations, loadSweepRecommendations, validationRows } from "../app/lib/loaders";
import type { LocalDirectoryHandle, LocalFileHandle, RunFile, WorkspaceRun } from "../app/lib/types";

// In-memory stand-ins for the structural handle surface the loaders read
// (File System Access in the browser, the native bridge when embedded).
// NOTHING here touches a real workspace: every byte is declared in the test.

const fakeFile = (name: string, text: string): File => {
  const bytes = new TextEncoder().encode(text);
  const like = {
    name,
    size: bytes.byteLength,
    lastModified: 0,
    type: "",
    text: async () => text,
    arrayBuffer: async () => bytes.buffer,
    slice: (start?: number, end?: number) => ({ text: async () => text.slice(start ?? 0, end) }),
  };
  return like as unknown as File;
};

const fakeFileHandle = (name: string, text: string): LocalFileHandle => ({
  kind: "file",
  name,
  getFile: async () => fakeFile(name, text),
});

const fakeRunFile = (name: string, text: string): RunFile => ({
  name,
  path: name,
  size: new TextEncoder().encode(text).byteLength,
  modified: 0,
  handle: fakeFileHandle(name, text),
});

const fakeDirectoryHandle: LocalDirectoryHandle = {
  kind: "directory",
  name: "run",
  values: () => (async function* () {})(),
  getDirectoryHandle: async () => { throw new Error("not used in this test"); },
  getFileHandle: async () => { throw new Error("not used in this test"); },
};

const fakeRun = (contents: Record<string, string>): WorkspaceRun => ({
  key: "run",
  name: "run",
  path: "run",
  experiment: "test",
  status: "not stamped",
  model: "Model not stamped",
  dateLabel: "run",
  promptCount: 0,
  conditionCount: 0,
  generationCount: 0,
  report: {},
  config: {},
  artifacts: Object.keys(contents),
  files: Object.entries(contents).map(([name, text]) => fakeRunFile(name, text)),
  handle: fakeDirectoryHandle,
  effectRows: [],
  generationRows: [],
  generationFile: null,
  previewTruncated: false,
  skippedGenerationLines: 0,
  cosineMatrices: [],
  validationConcepts: [],
  validationReport: {},
  sweepRows: [],
  sweepRecommendations: [],
  panelEffects: [],
});

describe("loadSweepRecommendations", () => {
  const recommendations = {
    // A concept whose selection FAILED: the server writes a bare string.
    sympathy: "no cell satisfied the coherence floor",
    // A concept with a stamped winning cell.
    anger: {
      winningCell: { layer: 38, alpha: 1 },
      criterion: {
        objective: { metric: "judgeScore" },
        constraints: { capabilityTolerance: 0.1, coherenceFloor: 0.6 },
        controls: { matchedNormRandomMargin: 0.05 },
      },
      metrics: { judgeScore: 0.71, distinct2: 0.93, bogus: "not a number" },
      devPromptsHash: "abc123def456ghi789",
      batteryHash: "battery-hash",
      sweepRun: "20260803T101010-sweep",
    },
  };

  it("reads a failure-string entry as a failure with no cell", async () => {
    const rows = await loadSweepRecommendations(fakeRun({ "recommendations.json": JSON.stringify(recommendations) }));
    const sympathy = rows.find((row) => row.concept === "sympathy");
    expect(sympathy).toBeDefined();
    expect(sympathy?.failure).toBe("no cell satisfied the coherence floor");
    expect(sympathy?.layer).toBeNull();
    expect(sympathy?.alpha).toBeNull();
    // Failure entries fall back to the historical defaults.
    expect(sympathy?.metric).toBe("markerDensity");
    expect(sympathy?.capabilityTolerance).toBe(0.15);
    expect(sympathy?.coherenceFloor).toBe(0.45);
    expect(sympathy?.matchedNormRandomMargin).toBeNull();
    expect(sympathy?.sweepRun).toBe("");
  });

  it("reads a selected entry with its declared criterion and provenance", async () => {
    const rows = await loadSweepRecommendations(fakeRun({ "recommendations.json": JSON.stringify(recommendations) }));
    const anger = rows.find((row) => row.concept === "anger");
    expect(anger).toBeDefined();
    expect(anger?.failure).toBe("");
    expect(anger?.layer).toBe(38);
    expect(anger?.alpha).toBe(1);
    expect(anger?.metric).toBe("judgeScore");
    expect(anger?.capabilityTolerance).toBe(0.1);
    expect(anger?.coherenceFloor).toBe(0.6);
    expect(anger?.matchedNormRandomMargin).toBe(0.05);
    expect(anger?.devPromptsHash).toBe("abc123def456ghi789");
    expect(anger?.batteryHash).toBe("battery-hash");
    expect(anger?.sweepRun).toBe("20260803T101010-sweep");
    // Non-numeric metric values are dropped rather than coerced.
    expect(anger?.metrics).toEqual({ judgeScore: 0.71, distinct2: 0.93 });
  });

  it("returns nothing when the run has no recommendations file", async () => {
    expect(await loadSweepRecommendations(fakeRun({}))).toEqual([]);
  });

  it("returns nothing for unparseable JSON rather than inventing rows", async () => {
    expect(await loadSweepRecommendations(fakeRun({ "recommendations.json": "{not json" }))).toEqual([]);
  });
});

describe("validationRows", () => {
  const report = {
    validation: {
      sympathy: {
        layer: 24,
        scenarios: 40,
        accuracy: 0.72,
        diagnostics: {
          auc: 0.81,
          oneSidedPredictions: false,
          heldOutCalibration: { accuracy: 0.68 },
        },
      },
      anger: {
        layer: 38,
        scenarioCount: 30,
        scenarioAccuracy: 0.5,
        note: "at chance on the held-out split",
      },
    },
    logitLens: {
      anger: [
        { layer: 38, topPositive: [{ token: "furious" }, { token: "outrage" }], topNegative: [{ token: "calm" }] },
      ],
    },
  };

  it("flattens the validation block, sorted by concept name", () => {
    const rows = validationRows(report);
    expect(rows.map((row) => row.name)).toEqual(["anger", "sympathy"]);
  });

  it("reads accuracy, calibration, and diagnostics where present", () => {
    const sympathy = validationRows(report)[1];
    expect(sympathy.layer).toBe(24);
    expect(sympathy.scenarios).toBe(40);
    expect(sympathy.accuracy).toBe(0.72);
    expect(sympathy.calibratedAccuracy).toBe(0.68);
    expect(sympathy.auc).toBe(0.81);
    expect(sympathy.oneSided).toBe(false);
    expect(sympathy.note).toBe("");
    expect(sympathy.positiveTokens).toEqual([]);
  });

  it("accepts the alternate scenarioCount / scenarioAccuracy keys and leaves absent fields null", () => {
    const anger = validationRows(report)[0];
    expect(anger.scenarios).toBe(30);
    expect(anger.accuracy).toBe(0.5);
    expect(anger.calibratedAccuracy).toBeNull();
    expect(anger.auc).toBeNull();
    expect(anger.oneSided).toBeNull();
    expect(anger.note).toBe("at chance on the held-out split");
  });

  it("attaches the layer-matched logit-lens tokens", () => {
    const anger = validationRows(report)[0];
    expect(anger.positiveTokens).toEqual(["furious", "outrage"]);
    expect(anger.negativeTokens).toEqual(["calm"]);
  });

  it("returns nothing for a report with no validation or concepts block", () => {
    expect(validationRows({})).toEqual([]);
    expect(validationRows({ concepts: ["not", "an", "object"] })).toEqual([]);
  });
});

describe("loadEffects", () => {
  // The strict-number contract seen through a real loader: a blank
  // adjustedP cell must arrive as q === null (missing), never 0.
  const csv = [
    "endpoint,deltaMean,ciLower,ciUpper,n,wilcoxonP,adjustedP,correction",
    "sentence_months,7.8,3.1,12.4,48,0.003,0.012,holm",
    "rule_adherent_rate,0.14,0.04,0.23,64,,,",
    "broken_row,,0.1,0.2,10,0.4,0.5,holm",
  ].join("\n");

  it("parses reported rows and derives the display unit from the endpoint name", async () => {
    const rows = await loadEffects(fakeRun({ "effect-sizes.csv": csv }));
    expect(rows).toHaveLength(2);
    expect(rows[0].endpoint).toBe("sentence_months");
    expect(rows[0].short).toBe("sentence months");
    expect(rows[0].unit).toBe("months");
    expect(rows[0].q).toBe(0.012);
    expect(rows[0].direction).toBe("positive");
    expect(rows[1].unit).toBe("Δ probability");
  });

  it("keeps a blank adjusted p missing rather than significant", async () => {
    const rows = await loadEffects(fakeRun({ "effect-sizes.csv": csv }));
    expect(rows[1].q).toBeNull();
  });

  it("drops a row whose estimate is blank rather than reading it as zero", async () => {
    const rows = await loadEffects(fakeRun({ "effect-sizes.csv": csv }));
    expect(rows.some((row) => row.endpoint === "broken_row")).toBe(false);
  });

  it("reads the raw Wilcoxon p and the correction family the table stamps", async () => {
    const rows = await loadEffects(fakeRun({ "effect-sizes.csv": csv }));
    expect(rows[0].p).toBe(0.003);
    expect(rows[0].correction).toBe("holm");
  });

  it("keeps a blank raw p and an unstamped correction absent", async () => {
    // The effects view used to print a hardcoded five-value array in this
    // column for every run — a fabricated statistic beside real ones.
    const rows = await loadEffects(fakeRun({ "effect-sizes.csv": csv }));
    expect(rows[1].p).toBeNull();
    expect(rows[1].correction).toBe("");
  });

  it("keeps a missing n absent rather than reading it as zero", async () => {
    // `?? 0` here turned an unstamped count into the substantive claim
    // "n = 0 items" — the same absent-remains-absent rule the strict number
    // reader exists for.
    const blankN = [
      "condition,endpoint,deltaMean,ciLower,ciUpper,n,wilcoxonP",
      "steered,choiceRate,0.14,0.04,0.23,,0.01",
    ].join("\n");
    const [row] = await loadEffects(fakeRun({ "effect-sizes.csv": blankN }));
    expect(row.n).toBeNull();
  });

  it("preserves the condition so two conditions' rows for one endpoint stay distinct", async () => {
    // The engines emit one row per condition × endpoint. Dropping the
    // condition made a multi-agent comparison render as duplicates keyed
    // against each other.
    const multi = [
      "condition,endpoint,deltaMean,ciLower,ciUpper,n,wilcoxonP,adjustedP,correction",
      "agent-a,choiceRate,0.14,0.04,0.23,64,0.009,0.028,holm",
      "agent-b,choiceRate,-0.02,-0.11,0.07,64,0.61,0.61,holm",
    ].join("\n");
    const rows = await loadEffects(fakeRun({ "effect-sizes.csv": multi }));
    expect(rows.map((row) => row.condition)).toEqual(["agent-a", "agent-b"]);
    expect(new Set(rows.map((row) => row.key)).size).toBe(2);
  });

  it("reads stratified companion rows with their family, estimand and inference class", async () => {
    // Both engines append per-stratum rows (stratifyBy ≠ "pooled",
    // 2026-08-06). They used to be discarded outright, so the browser could
    // not show the saturated-cell masking the strata exist to expose.
    const stratified = [
      "condition,endpoint,deltaMean,ciLower,ciUpper,n,wilcoxonP,adjustedP,correction,stratifyBy,stratum,unit,estimand,inference",
      "steered,choiceRate,0.047,0.01,0.09,12,1,1,bh,pooled,,,,",
      "steered,choiceRate,0.3,0.17,0.42,50,0.0001,,,promptID,loan-notLegal,sample,withinItemSamples,diagnostic",
      "steered,choiceRate,0.075,0.01,0.14,4,0.2,0.4,bh,arm,notLegal,item,itemLevel,corrected",
    ].join("\n");
    const rows = await loadEffects(fakeRun({ "effect-sizes.csv": stratified }));
    expect(rows).toHaveLength(3);
    expect(rows[0].stratifyBy).toBe("pooled");
    expect(rows[0].estimand).toBe("");
    expect(rows[1].stratum).toBe("loan-notLegal");
    expect(rows[1].pairedUnit).toBe("sample");
    expect(rows[1].estimand).toBe("withinItemSamples");
    expect(rows[1].inference).toBe("diagnostic");
    // The diagnostic row carries no adjusted p and no correction family.
    expect(rows[1].q).toBeNull();
    expect(rows[1].correction).toBe("");
    expect(rows[2].estimand).toBe("itemLevel");
    expect(rows[2].inference).toBe("corrected");
    expect(new Set(rows.map((row) => row.key)).size).toBe(3);
  });

  it("treats a table with no stratifyBy column as all pooled", async () => {
    const rows = await loadEffects(fakeRun({ "effect-sizes.csv": csv }));
    expect(rows.every((row) => row.stratifyBy === "pooled")).toBe(true);
    expect(rows.every((row) => row.estimand === "")).toBe(true);
  });

  it("returns nothing when the run has no effect-sizes table", async () => {
    expect(await loadEffects(fakeRun({}))).toEqual([]);
  });
});

describe("loadGenerations", () => {
  // The instrument marker, the stamped wordCount, and the stamped distinct2
  // are three separate facts about a record; the reader shows each one only
  // where the record actually carried it.
  const jsonl = [
    JSON.stringify({ condition: "baseline", promptID: "item-1", sampleIndex: 0, output: "The court affirms.", wordCount: 3, distinct2: 0.9, prompt: "p" }),
    JSON.stringify({ condition: "fear-agent", promptID: "item-1", sampleIndex: 0, instrument: "answerTokenLogprob", selected: "B", target: "A", options: ["A", "B"] }),
    JSON.stringify({ condition: "baseline", promptID: "item-2", sampleIndex: 0, output: "one two three four" }),
    // Text that merely BEGINS like the reader's own fallback string is not
    // an instrument record — the old prefix test called it one.
    JSON.stringify({ condition: "baseline", promptID: "item-3", sampleIndex: 0, output: "Selected option: B, the court holds." }),
  ].join("\n");

  it("flags a record carrying an instrument marker, and only that record", async () => {
    const { rows } = await loadGenerations(fakeRun({ "generations.jsonl": jsonl }));
    expect(rows.map((row) => row.isInstrument)).toEqual([false, true, false, false]);
  });

  it("still synthesises the display string for an instrument record with no text", async () => {
    const { rows } = await loadGenerations(fakeRun({ "generations.jsonl": jsonl }));
    expect(rows[1].output).toBe("Selected option: B");
  });

  it("says whether a word count is the engine's or the viewer's", async () => {
    const { rows } = await loadGenerations(fakeRun({ "generations.jsonl": jsonl }));
    expect(rows[0].wordCountStored).toBe(true);
    expect(rows[0].words).toBe(3);
    expect(rows[2].wordCountStored).toBe(false);
    expect(rows[2].words).toBe(4);
  });

  it("marks an unstamped distinct-2 as unstamped rather than letting 0 read as a measurement", async () => {
    const { rows } = await loadGenerations(fakeRun({ "generations.jsonl": jsonl }));
    expect(rows[0].distinct2Stored).toBe(true);
    expect(rows[2].distinct2Stored).toBe(false);
    expect(rows[2].distinct2).toBe(0);
  });
});
