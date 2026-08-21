import { describe, expect, it } from "vitest";
import {
  codingDisagreements,
  fieldsFromRows,
  formatCode,
  loadCodings,
  parseCodingReport,
  parseCodingRows,
  wordCountProfiles,
  type CodingFieldSpec,
} from "../app/lib/codingdata";
import type { LocalDirectoryHandle, LocalFileHandle, RunFile, WorkspaceRun } from "../app/lib/types";

// Fixtures are written from the ENGINE CONTRACTS (response_coding.py +
// tasks._evaluate_response_coding and their Swift twins), not copied from a
// run: no coding run existed when this reader was built.

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
  handle: { kind: "file", name, getFile: async () => fakeFile(name, text) } satisfies LocalFileHandle,
});

const fakeDirectory: LocalDirectoryHandle = {
  kind: "directory", name: "run",
  values: () => (async function* () {})(),
  getDirectoryHandle: async () => { throw new Error("not used in this test"); },
  getFileHandle: async () => { throw new Error("not used in this test"); },
};

const fakeRun = (contents: Record<string, string>): WorkspaceRun => ({
  key: "run", name: "run", path: "run", experiment: "test", status: "not stamped", model: "Model not stamped",
  dateLabel: "run", promptCount: 0, conditionCount: 0, generationCount: 0, report: {}, config: {},
  artifacts: Object.keys(contents), files: Object.entries(contents).map(([file, text]) => fakeRunFile(file, text)),
  handle: fakeDirectory, effectRows: [], generationRows: [], generationFile: null, previewTruncated: false,
  skippedGenerationLines: 0, cosineMatrices: [], validationConcepts: [], validationReport: {},
  sweepRows: [], sweepRecommendations: [], panelEffects: [],
});

const jsonl = (rows: unknown[]) => rows.map((row) => JSON.stringify(row)).join("\n") + "\n";

const codingRow = (judge: string, condition: string, promptID: string, codes: Record<string, unknown>, wordCount = 120) => ({
  experiment: "study", condition, promptID, sampleIndex: 0, seed: 7, wordCount, codes,
  briefReason: `${judge} on ${promptID}`, judge, judgeKind: "openrouter", judgeModel: "vendor/model", judgeProvider: "provider",
});

const fields: CodingFieldSpec[] = [
  { name: "mentionsLegalRule", type: "boolean", optional: false, values: [] },
  { name: "severity", type: "integer", optional: true, values: [] },
  { name: "stance", type: "enum", optional: false, values: ["legalist", "equitable", "mixed"] },
];

describe("parseCodingRows", () => {
  it("reads a coding row, keeping the engine's word count and the exact seed", () => {
    const line = JSON.stringify({ ...codingRow("coder-1", "baseline", "prompt-1", { mentionsLegalRule: true, severity: 4, stance: "legalist" }), seed: 0 })
      .replace('"seed":0', '"seed":9007199254740993');
    const row = parseCodingRows(line).rows[0];
    expect(row.codes).toEqual({ mentionsLegalRule: true, severity: 4, stance: "legalist" });
    expect(row.wordCount).toBe(120);
    expect(row.seed).toBe("9007199254740993");
    expect(row.judge).toBe("coder-1");
  });

  it("keeps an optional field's null as null instead of dropping it", () => {
    const row = parseCodingRows(jsonl([codingRow("coder-1", "baseline", "prompt-1", { severity: null })])).rows[0];
    expect("severity" in row.codes).toBe(true);
    expect(row.codes.severity).toBeNull();
    expect(formatCode(row.codes.severity)).toBe("null");
  });

  it("leaves an unstamped word count null rather than counting words itself", () => {
    const row = parseCodingRows(jsonl([{ condition: "baseline", promptID: "p", codes: {}, judge: "coder-1" }])).rows[0];
    expect(row.wordCount).toBeNull();
  });

  it("skips unparseable and unjoinable lines and drops a truncated tail", () => {
    const { rows, skipped } = parseCodingRows(["{not json", JSON.stringify({ codes: {} }), JSON.stringify(codingRow("coder-1", "baseline", "prompt-1", {}))].join("\n"));
    expect(rows).toHaveLength(1);
    expect(skipped).toBe(2);
    const truncated = parseCodingRows(jsonl([codingRow("coder-1", "baseline", "prompt-1", {})]) + '{"condition": "baseline", "prom', true);
    expect(truncated.rows).toHaveLength(1);
    expect(truncated.skipped).toBe(0);
  });

  it("loads from a run and reports a missing codings file as absent", async () => {
    const loaded = await loadCodings(fakeRun({ "codings.jsonl": jsonl([codingRow("coder-1", "baseline", "prompt-1", { stance: "mixed" })]) }));
    expect(loaded.present).toBe(true);
    expect(loaded.rows).toHaveLength(1);
    const missing = await loadCodings(fakeRun({}));
    expect(missing.present).toBe(false);
    expect(missing.rows).toEqual([]);
  });
});

describe("parseCodingReport", () => {
  const raw = {
    mode: "perResponseCoding", experiment: "study", experimentHash: "abc123", sourceRun: "20260101T000000000-exp-study-run",
    judges: ["coder-1", "coder-2"], judgeModel: "vendor/a, vendor/b",
    judgeDetails: [{ name: "coder-1", kind: "openrouter", requestedModel: "vendor/a", actualModel: "vendor/a" }, { name: "coder-2", kind: "local", requestedModel: "vendor/b", actualModel: "vendor/b", revision: "deadbeef" }],
    judgeRubricFile: "prompts/rubrics/coding-v1.md", judgeRubricHash: "30b239",
    fields: [
      { name: "mentionsLegalRule", type: "boolean", optional: false },
      { name: "severity", type: "integer", optional: true },
      { name: "stance", type: "enum", optional: false, values: ["legalist", "equitable", "mixed"] },
    ],
    codings: 24,
    conditions: {
      "fear agent": {
        codedResponses: 6, codings: 12, meanWordCount: 143.5,
        fields: {
          mentionsLegalRule: { n: 11, nulls: 1, trueCount: 4, trueShare: 4 / 11 },
          severity: { n: 10, nulls: 2, mean: 3.4 },
          stance: { n: 12, nulls: 0, counts: { equitable: 7, legalist: 5 } },
        },
      },
      baseline: {
        codedResponses: 6, codings: 12, meanWordCount: 120,
        fields: {
          mentionsLegalRule: { n: 12, nulls: 0, trueCount: 9, trueShare: 0.75 },
          severity: { n: 12, nulls: 0, mean: 2.5 },
          stance: { n: 12, nulls: 0, counts: { legalist: 10, mixed: 2 } },
        },
      },
    },
    fieldAgreement: [
      { field: "mentionsLegalRule", judgeA: "coder-1", judgeB: "coder-2", n: 12, percentAgreement: 0.91, kappa: 0.8 },
      { field: "severity", judgeA: "coder-1", judgeB: "coder-2", n: 11, meanAbsoluteDifference: 0.45 },
      { field: "stance", judgeA: "coder-1", judgeB: "coder-2", n: 12, percentAgreement: 1, kappa: null },
    ],
    evaluationSource: "pinnedRubric",
  };

  const report = parseCodingReport({ present: true, raw, error: "", file: null });

  it("reads the declared schema, including enum vocabularies and optionality", () => {
    expect(report.fields.map((field) => field.name)).toEqual(["mentionsLegalRule", "severity", "stance"]);
    expect(report.fields[1].optional).toBe(true);
    expect(report.fields[2].values).toEqual(["legalist", "equitable", "mixed"]);
  });

  it("makes baseline the anchor column by sorting it first", () => {
    expect(report.conditions.map((condition) => condition.condition)).toEqual(["baseline", "fear agent"]);
  });

  it("renders boolean, numeric and enum aggregates without inventing the missing ones", () => {
    const fear = report.conditions[1];
    expect(fear.fields.mentionsLegalRule).toMatchObject({ n: 11, nulls: 1, trueCount: 4, mean: null, counts: null });
    expect(fear.fields.mentionsLegalRule.trueShare).toBeCloseTo(4 / 11, 10);
    expect(fear.fields.severity).toMatchObject({ mean: 3.4, trueShare: null, counts: null });
    expect(fear.fields.stance.counts).toEqual({ equitable: 7, legalist: 5 });
    expect(fear.meanWordCount).toBe(143.5);
  });

  it("keeps a null kappa distinguishable from an absent statistic", () => {
    expect(report.fieldAgreement[1].percentAgreement).toBeNull();
    expect(report.fieldAgreement[1].meanAbsoluteDifference).toBe(0.45);
    expect(report.fieldAgreement[2].kappa).toBeNull();
    expect(report.fieldAgreement[2].percentAgreement).toBe(1);
  });

  it("carries the pins and the evaluation source", () => {
    expect(report.rubricFile).toBe("prompts/rubrics/coding-v1.md");
    expect(report.rubricHash).toBe("30b239");
    expect(report.sourceRun).toBe("20260101T000000000-exp-study-run");
    expect(report.evaluationSource).toBe("pinnedRubric");
    expect(report.judgeDetails[1].revision).toBe("deadbeef");
    expect(report.mode).toBe("perResponseCoding");
  });

  it("reports an absent report as absent rather than as an empty study", () => {
    const absent = parseCodingReport({ present: false, raw: {}, error: "", file: null });
    expect(absent.present).toBe(false);
    expect(absent.conditions).toEqual([]);
    expect(absent.fields).toEqual([]);
    expect(absent.codings).toBeNull();
  });

  it("falls back to the fields the rows actually coded when no schema is stamped", () => {
    const rows = parseCodingRows(jsonl([codingRow("coder-1", "baseline", "prompt-1", { stance: "mixed", severity: 2 })])).rows;
    expect(fieldsFromRows(rows).map((field) => field.name)).toEqual(["stance", "severity"]);
    expect(fieldsFromRows(rows)[0].type).toBe("");
  });
});

describe("codingDisagreements", () => {
  const rows = parseCodingRows(jsonl([
    codingRow("coder-1", "baseline", "prompt-1", { mentionsLegalRule: true, severity: 3, stance: "legalist" }),
    codingRow("coder-2", "baseline", "prompt-1", { mentionsLegalRule: false, severity: 3, stance: "legalist" }),
    codingRow("coder-1", "fear agent", "prompt-2", { mentionsLegalRule: true, severity: 2, stance: "mixed" }),
    codingRow("coder-2", "fear agent", "prompt-2", { mentionsLegalRule: true, severity: 5, stance: "mixed" }),
    codingRow("coder-1", "fear agent", "prompt-3", { mentionsLegalRule: true, severity: null, stance: "mixed" }),
    codingRow("coder-2", "fear agent", "prompt-3", { mentionsLegalRule: true, severity: 4, stance: "mixed" }),
    codingRow("coder-1", "fear agent", "prompt-4", { mentionsLegalRule: false, severity: 1, stance: "equitable" }),
  ])).rows;

  const splits = codingDisagreements(rows, fields);

  it("finds exactly the field cells two coders read differently", () => {
    expect(splits.map((split) => `${split.field}/${split.promptID}`)).toEqual([
      "mentionsLegalRule/prompt-1", "severity/prompt-2", "severity/prompt-3",
    ]);
  });

  it("counts a null against a coded value as a coding difference", () => {
    const nullSplit = splits.find((split) => split.promptID === "prompt-3");
    expect(nullSplit?.codings.map((coding) => formatCode(coding.value))).toEqual(["null", "4"]);
  });

  it("shows both codings with their reasons", () => {
    const split = splits[0];
    expect(split.codings).toHaveLength(2);
    expect(split.codings[0]).toMatchObject({ judge: "coder-1", value: true });
    expect(split.codings[1]).toMatchObject({ judge: "coder-2", value: false, briefReason: "coder-2 on prompt-1" });
  });

  it("never reports a split for a response only one coder saw", () => {
    expect(splits.some((split) => split.promptID === "prompt-4")).toBe(false);
    expect(codingDisagreements(rows.filter((row) => row.judge === "coder-1"), fields)).toEqual([]);
  });
});

describe("wordCountProfiles", () => {
  const rows = parseCodingRows(jsonl([
    codingRow("coder-1", "baseline", "prompt-1", {}, 100),
    codingRow("coder-2", "baseline", "prompt-1", {}, 100),
    codingRow("coder-1", "baseline", "prompt-2", {}, 200),
    codingRow("coder-1", "fear agent", "prompt-1", {}, 50),
    { ...codingRow("coder-2", "fear agent", "prompt-1", {}), wordCount: null },
  ])).rows;

  it("deduplicates responses across coders instead of counting a response once per coder", () => {
    const { profiles } = wordCountProfiles(rows);
    const baseline = profiles.find((profile) => profile.condition === "baseline");
    expect(baseline?.responses).toBe(2);
    expect(baseline?.mean).toBe(150);
    expect(baseline?.min).toBe(100);
    expect(baseline?.max).toBe(200);
  });

  it("puts baseline first and shares one axis across conditions", () => {
    const { profiles, low, high, missing } = wordCountProfiles(rows);
    expect(profiles.map((profile) => profile.condition)).toEqual(["baseline", "fear agent"]);
    expect(low).toBe(50);
    expect(high).toBe(200);
    expect(missing).toBe(1);
    expect(profiles[0].bins).toHaveLength(10);
    expect(profiles[0].bins.reduce((sum, bin) => sum + bin.count, 0)).toBe(2);
  });

  it("returns nothing to plot when no row carries a word count", () => {
    const empty = wordCountProfiles(parseCodingRows(jsonl([{ condition: "baseline", promptID: "p", codes: {}, judge: "coder-1" }])).rows);
    expect(empty.profiles).toEqual([]);
    expect(empty.missing).toBe(1);
  });
});
