import { describe, expect, it } from "vitest";
import {
  cellKey,
  confidenceHistogram,
  disagreementCells,
  findSourceRun,
  judgeTallies,
  judgmentCells,
  loadJudgments,
  normalizeOutcome,
  parseJudgeReport,
  parseJudgingContext,
  parseJudgmentRows,
  parseRunStatus,
  parseSourceResponses,
  readJSONArtifact,
} from "../app/lib/judged";
import type { LocalDirectoryHandle, LocalFileHandle, RunFile, WorkspaceRun } from "../app/lib/types";

// In-memory stand-ins for the handle surface (same pattern as
// test/loaders.test.ts). Every byte here is declared in the test: no
// workspace file is read, and no text was copied out of a real run.

const fakeFile = (name: string, text: string): File => {
  const bytes = new TextEncoder().encode(text);
  return {
    name,
    size: bytes.byteLength,
    lastModified: 0,
    type: "",
    text: async () => text,
    arrayBuffer: async () => bytes.buffer,
    slice: (start?: number, end?: number) => ({ text: async () => text.slice(start ?? 0, end) }),
  } as unknown as File;
};

const fakeRunFile = (name: string, text: string): RunFile => ({
  name, path: name, size: new TextEncoder().encode(text).byteLength, modified: 0,
  handle: { kind: "file", name, getFile: async () => fakeFile(name, text) } satisfies LocalFileHandle,
});

const fakeDirectory: LocalDirectoryHandle = {
  kind: "directory", name: "run",
  values: () => (async function* () {})(),
  getDirectoryHandle: async () => { throw new Error("not used in this test"); },
  getFileHandle: async () => { throw new Error("not used in this test"); },
};

const fakeRun = (contents: Record<string, string>, name = "run"): WorkspaceRun => ({
  key: name, name, path: name, experiment: "test", status: "not stamped", model: "Model not stamped",
  dateLabel: name, promptCount: 0, conditionCount: 0, generationCount: 0, report: {}, config: {},
  artifacts: Object.keys(contents), files: Object.entries(contents).map(([file, text]) => fakeRunFile(file, text)),
  handle: fakeDirectory, effectRows: [], generationRows: [], generationFile: null, previewTruncated: false,
  skippedGenerationLines: 0, cosineMatrices: [], validationConcepts: [], validationReport: {},
  sweepRows: [], sweepRecommendations: [], panelEffects: [],
});

const jsonl = (rows: unknown[]) => rows.map((row) => JSON.stringify(row)).join("\n") + "\n";

// One pair, judged by two judges, expressed in each engine's own keys.
const serverRow = (judge: string, outcome: string, confidence: number) => ({
  promptID: "prompt-1", sampleIndex: 0, condition: "fear agent", baselineSeed: 11, variantSeed: 22,
  baselineWas: "B", outcome, confidence,
  judgment: { winner: outcome === "tie" ? "tie" : outcome === "variant" ? "A" : "B", confidence, brief_reason: `${judge} reasoning`, a_scores: { clarity: 7 }, b_scores: { clarity: 5 } },
  judge, judgeKind: "openrouter", judgeModel: "vendor/model-x", judgeProvider: "provider-y",
});

const swiftRow = (judge: string, conditionResult: string, confidence: number) => ({
  experiment: "study", experimentHash: "hash", sourceRunDirectory: "20260101T000000000-exp-study-run",
  promptID: "prompt-1", sampleIndex: 0, condition: "fear agent", baselineSeed: 11, variantSeed: 22,
  baselineWas: "B", conditionWas: "A", conditionResult, prompt: "the task prompt",
  judgePrompt: "…", judgeRubricFile: "prompts/rubrics/r.md", judgeRubricHash: "abc", structuredPrompt: null,
  judgment: { winner: conditionResult === "tie" ? "tie" : conditionResult === "condition" ? "A" : "B", confidence, brief_reason: `${judge} reasoning`, a_scores: { clarity: 7 }, b_scores: { clarity: 5 } },
  judge, judgeKind: "local", judgeModel: "google/gemma-3-27b-it",
});

describe("normalizeOutcome", () => {
  it("maps Swift's 'condition' and the server's 'variant' onto one word", () => {
    expect(normalizeOutcome("condition")).toBe("variant");
    expect(normalizeOutcome("variant")).toBe("variant");
    expect(normalizeOutcome("baseline")).toBe("baseline");
    expect(normalizeOutcome("tie")).toBe("tie");
  });

  it("refuses to guess an outcome it was never given", () => {
    expect(normalizeOutcome(undefined)).toBe("unknown");
    expect(normalizeOutcome("winner")).toBe("unknown");
  });
});

describe("parseJudgmentRows", () => {
  it("normalizes both engine dialects to the same row", () => {
    const server = parseJudgmentRows(jsonl([serverRow("judge-1", "variant", 0.8)])).rows[0];
    const swift = parseJudgmentRows(jsonl([swiftRow("judge-1", "condition", 0.8)])).rows[0];
    for (const key of ["judge", "condition", "promptID", "sampleIndex", "outcome", "winner", "baselineWas", "confidence", "briefReason"] as const) {
      expect(server[key]).toEqual(swift[key]);
    }
    expect(server.outcome).toBe("variant");
    expect(server.dialect).toBe("server");
    expect(swift.dialect).toBe("swift");
  });

  it("reads the confidence from either the row or the nested judgment", () => {
    const nested = parseJudgmentRows(jsonl([{ ...serverRow("judge-1", "variant", 0.7), confidence: undefined }])).rows[0];
    expect(nested.confidence).toBe(0.7);
  });

  it("keeps UInt64 seeds exact instead of letting JSON.parse round them", () => {
    const line = JSON.stringify({ ...serverRow("judge-1", "variant", 0.8), baselineSeed: 0, variantSeed: 0 })
      .replace('"baselineSeed":0', '"baselineSeed":5230848306049226115');
    const row = parseJudgmentRows(line).rows[0];
    expect(row.baselineSeed).toBe("5230848306049226115");
    // What the naive path would have shown: JSON.parse rounds the UInt64.
    expect(String((JSON.parse(line) as { baselineSeed: number }).baselineSeed)).not.toBe(row.baselineSeed);
  });

  it("skips unparseable and unjoinable lines rather than counting them", () => {
    const { rows, skipped } = parseJudgmentRows(["{not json", JSON.stringify({ judge: "judge-1" }), JSON.stringify(serverRow("judge-1", "tie", 0.5))].join("\n"));
    expect(rows).toHaveLength(1);
    expect(skipped).toBe(2);
  });

  it("drops the final partial line of a truncated bounded read", () => {
    const text = jsonl([serverRow("judge-1", "variant", 0.8)]) + '{"promptID": "prompt-2", "cond';
    expect(parseJudgmentRows(text, true).rows).toHaveLength(1);
    expect(parseJudgmentRows(text, false).skipped).toBe(1);
  });

  it("records a missing outcome as unknown rather than inferring it from the winner", () => {
    const row = parseJudgmentRows(jsonl([{ promptID: "p", condition: "c", judge: "judge-1", judgment: { winner: "A", confidence: 0.9, brief_reason: "" } }])).rows[0];
    expect(row.outcome).toBe("unknown");
    expect(row.winner).toBe("A");
  });
});

describe("disagreement selection", () => {
  const rows = parseJudgmentRows(jsonl([
    serverRow("judge-1", "variant", 0.8),
    serverRow("judge-2", "baseline", 0.6),
    { ...serverRow("judge-1", "variant", 0.9), promptID: "prompt-2" },
    { ...serverRow("judge-2", "variant", 0.7), promptID: "prompt-2" },
    { ...serverRow("judge-1", "tie", 0.5), promptID: "prompt-3" },
    { ...serverRow("judge-2", "variant", 0.5), promptID: "prompt-3" },
    { ...serverRow("judge-1", "variant", 0.5), promptID: "prompt-4" },
  ])).rows;

  it("groups rows into (condition, promptID, sampleIndex) cells", () => {
    expect(judgmentCells(rows)).toHaveLength(4);
    expect(judgmentCells(rows)[0].key).toBe(cellKey("fear agent", "prompt-1", 0));
  });

  it("finds the cells where two judges split, including a tie against a win", () => {
    expect(disagreementCells(rows).map((cell) => cell.promptID)).toEqual(["prompt-1", "prompt-3"]);
  });

  it("never calls a single judge's cell a disagreement", () => {
    expect(disagreementCells(rows).some((cell) => cell.promptID === "prompt-4")).toBe(false);
  });

  it("cannot manufacture a disagreement out of an unstamped outcome", () => {
    const mixed = parseJudgmentRows(jsonl([
      serverRow("judge-1", "variant", 0.8),
      { ...serverRow("judge-2", "variant", 0.8), outcome: undefined },
    ])).rows;
    expect(disagreementCells(mixed)).toHaveLength(0);
    expect(judgmentCells(mixed)[0].verdicts.map((verdict) => verdict.outcome)).toEqual(["variant", "unknown"]);
  });

  it("counts per-judge splits and mean confidence from the rows", () => {
    const tallies = judgeTallies(rows);
    const judgeOne = tallies.find((tally) => tally.judge === "judge-1");
    expect(judgeOne?.variantWins).toBe(3);
    expect(judgeOne?.ties).toBe(1);
    expect(judgeOne?.n).toBe(4);
    expect(judgeOne?.meanConfidence).toBeCloseTo((0.8 + 0.9 + 0.5 + 0.5) / 4, 10);
  });
});

describe("confidenceHistogram", () => {
  it("bins stamped confidences and reports the unstamped ones separately", () => {
    const rows = parseJudgmentRows(jsonl([
      serverRow("judge-1", "variant", 0.85),
      serverRow("judge-2", "variant", 0.8),
      { ...serverRow("judge-3", "tie", 0), confidence: undefined, judgment: { winner: "tie" } },
    ])).rows;
    const histogram = confidenceHistogram(rows);
    expect(histogram.counted).toBe(2);
    expect(histogram.missing).toBe(1);
    expect(histogram.bins[8].count).toBe(2); // [0.8, 0.9)
    expect(histogram.bins.reduce((sum, bin) => sum + bin.count, 0)).toBe(2);
  });
});

describe("parseJudgeReport", () => {
  const serverReport = {
    conditions: { "fear agent": { baselineWins: 5, variantWins: 25, ties: 0, n: 30 } },
    pairs: 60, judgeModel: "google/gemma-3-4b-it", judgedOn: "client", evaluationSource: "manifest",
    judges: [
      { name: "judge-1", kind: "local", requestedModel: "m-a", actualModel: "m-a", pairs: 30, conditions: { "fear agent": { baselineWins: 5, variantWins: 25, ties: 0, n: 30 } } },
      { name: "judge-2", kind: "local", requestedModel: "m-b", actualModel: "m-b", pairs: 30 },
    ],
    agreement: [{ judges: ["judge-1", "judge-2"], kappa: 0.239, n: 60, percentAgreement: 0.4166 }],
    rubricFile: "prompts/rubrics/default-paired-v1.md", rubricHash: "ea7344", sourceRun: "20260101T000000000-exp-s-run",
  };

  const swiftReport = {
    conditions: { "fear agent": { pairs: 30, conditionWins: 25, baselineWins: 5, ties: 0, meanConfidence: 0.78, structuredSummaries: {} } },
    judges: ["judge-1", "judge-2"], judgeModel: "m-a, m-b",
    judgeRubricFile: "prompts/rubrics/default-paired-v1.md", judgeRubricHash: "ea7344",
    judgeAgreement: [{ judgeA: "judge-1", judgeB: "judge-2", items: 60, percentAgreement: 0.4166, kappa: 0.239 }],
    humanAgreement: [{ judge: "judge-1", items: 12, percentAgreement: 0.75, kappa: 0.5 }],
    sourceRunDirectory: "20260101T000000000-exp-s-run", epochUnverified: true,
  };

  it("normalizes the server dialect", () => {
    const report = parseJudgeReport({ present: true, raw: serverReport, error: "", file: null });
    expect(report.dialect).toBe("server");
    expect(report.conditions[0]).toMatchObject({ condition: "fear agent", pairs: 30, variantWins: 25, baselineWins: 5, ties: 0 });
    expect(report.judges[0].conditions[0].variantWins).toBe(25);
    expect(report.agreement[0]).toMatchObject({ judgeA: "judge-1", judgeB: "judge-2", n: 60, kappa: 0.239 });
    expect(report.sourceRun).toBe("20260101T000000000-exp-s-run");
    expect(report.rubricFile).toBe("prompts/rubrics/default-paired-v1.md");
  });

  it("normalizes the Swift dialect to the same shape", () => {
    const server = parseJudgeReport({ present: true, raw: serverReport, error: "", file: null });
    const swift = parseJudgeReport({ present: true, raw: swiftReport, error: "", file: null });
    expect(swift.dialect).toBe("swift");
    expect(swift.conditions[0].variantWins).toBe(server.conditions[0].variantWins);
    expect(swift.conditions[0].pairs).toBe(server.conditions[0].pairs);
    expect(swift.agreement[0]).toEqual(server.agreement[0]);
    expect(swift.sourceRun).toBe(server.sourceRun);
    expect(swift.rubricFile).toBe(server.rubricFile);
    expect(swift.judges.map((judge) => judge.name)).toEqual(["judge-1", "judge-2"]);
    expect(swift.conditions[0].meanConfidence).toBe(0.78);
    expect(swift.epochUnverified).toBe(true);
    expect(swift.humanAgreement[0]).toMatchObject({ judge: "judge-1", n: 12 });
  });

  it("leaves absent tallies null rather than zero", () => {
    const report = parseJudgeReport({ present: true, raw: { conditions: { solo: {} } }, error: "", file: null });
    expect(report.conditions[0]).toMatchObject({ pairs: null, variantWins: null, baselineWins: null, ties: null });
    expect(report.present).toBe(true);
  });

  it("reports an absent report as absent", () => {
    const report = parseJudgeReport({ present: false, raw: {}, error: "", file: null });
    expect(report.present).toBe(false);
    expect(report.conditions).toEqual([]);
    expect(report.agreement).toEqual([]);
    expect(report.dialect).toBe("unknown");
  });

  it("sorts baseline first among conditions", () => {
    const report = parseJudgeReport({ present: true, raw: { conditions: { "zeta agent": { n: 1 }, baseline: { n: 2 }, "alpha agent": { n: 3 } } }, error: "", file: null });
    expect(report.conditions.map((condition) => condition.condition)).toEqual(["baseline", "alpha agent", "zeta agent"]);
  });
});

describe("parseJudgingContext and parseRunStatus", () => {
  it("reads the pins a failed evaluate still leaves behind", () => {
    const context = parseJudgingContext({
      present: true, error: "", file: null,
      raw: {
        experiment: "study", experimentHash: "3f17c7", rubricFile: "prompts/rubrics/r.md", rubricHash: "30b239",
        sourceRun: "20260101T000000000-exp-s-run", sourceGenerationsSha256: "73bc59", structuredPromptSha256: null,
        judges: [{ name: "judge-ds", kind: "openrouter", model: "vendor/model", provider: "some-provider", revision: null, dtype: null }, { notAJudge: true }],
      },
    });
    expect(context.judges).toHaveLength(1);
    expect(context.judges[0]).toMatchObject({ name: "judge-ds", kind: "openrouter", model: "vendor/model", provider: "some-provider", revision: "" });
    expect(context.structuredPromptSha256).toBe("");
  });

  it("reads status truth, including which judges never ran", () => {
    const status = parseRunStatus({
      present: true, error: "", file: null,
      raw: { status: "failed", stage: "evaluate", itemsWritten: 0, invalidResponses: 0, itemLabel: "item", expectedUnits: ["judge-ds", "judge-tm"], completedUnits: [], pendingUnits: ["judge-ds", "judge-tm"], evidenceComplete: false, error: "judge response carried no content", errorType: "RuntimeError" },
    });
    expect(status.status).toBe("failed");
    expect(status.pendingUnits).toEqual(["judge-ds", "judge-tm"]);
    expect(status.invalidResponses).toBe(0);
    expect(status.evidenceComplete).toBe(false);
  });

  it("never defaults an unstamped status to success", () => {
    expect(parseRunStatus({ present: true, raw: {}, error: "", file: null }).status).toBe("not stamped");
    expect(parseRunStatus({ present: false, raw: {}, error: "", file: null }).status).toBe("");
    expect(parseRunStatus({ present: true, raw: {}, error: "", file: null }).invalidResponses).toBeNull();
  });
});

describe("artifact loading from a run", () => {
  it("loads judgments from a run directory", async () => {
    const run = fakeRun({ "judgments.jsonl": jsonl([serverRow("judge-1", "variant", 0.8), serverRow("judge-2", "baseline", 0.6)]) });
    const loaded = await loadJudgments(run);
    expect(loaded.present).toBe(true);
    expect(loaded.rows).toHaveLength(2);
    expect(loaded.truncated).toBe(false);
  });

  it("reports a missing judgments file as absent, not empty", async () => {
    const loaded = await loadJudgments(fakeRun({}));
    expect(loaded.present).toBe(false);
    expect(loaded.rows).toEqual([]);
    expect(loaded.file).toBeNull();
  });

  it("distinguishes an absent JSON artifact from an unparseable one", async () => {
    expect(await readJSONArtifact(fakeRun({}), "judge-report.json")).toMatchObject({ present: false, error: "" });
    const broken = await readJSONArtifact(fakeRun({ "judge-report.json": "{not json" }), "judge-report.json");
    expect(broken.present).toBe(true);
    expect(broken.error).toContain("could not be parsed");
    expect(parseJudgeReport(broken).present).toBe(false);
  });
});

describe("the source-run join", () => {
  const generations = jsonl([
    { condition: "baseline", promptID: "prompt-1", sampleIndex: 0, prompt: "task", output: "baseline text", wordCount: 2, seed: 1 },
    { condition: "fear agent", promptID: "prompt-1", sampleIndex: 0, prompt: "task", output: "variant text", wordCount: 2, seed: 2 },
    { condition: "fear agent", promptID: "prompt-1", output: "no sample index", wordCount: 3 },
    { condition: "fear agent", promptID: "prompt-9", instrument: "logprob" },
  ]);

  it("indexes responses by (condition, promptID, sampleIndex)", () => {
    const responses = parseSourceResponses(generations);
    expect(responses.get(cellKey("baseline", "prompt-1", 0))?.output).toBe("baseline text");
    expect(responses.get(cellKey("fear agent", "prompt-1", 0))?.output).toBe("no sample index"); // absent sampleIndex is 0
    expect(responses.has(cellKey("fear agent", "prompt-9", 0))).toBe(false); // instrument readouts carry no output
  });

  it("finds the source run by directory name and reports its absence honestly", () => {
    const runs = [fakeRun({}, "20260101T000000000-exp-s-run")];
    expect(findSourceRun(runs, "20260101T000000000-exp-s-run")?.name).toBe("20260101T000000000-exp-s-run");
    expect(findSourceRun(runs, "/somewhere/else/20260101T000000000-exp-s-run")?.name).toBe("20260101T000000000-exp-s-run");
    expect(findSourceRun(runs, "20260202T000000000-exp-other-run")).toBeNull();
    expect(findSourceRun(runs, "")).toBeNull();
  });
});
