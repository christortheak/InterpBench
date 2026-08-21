import { describe, expect, it } from "vitest";
import { detectRunKind, runKindLabel } from "../app/lib/runKind";

// Fixtures are SYNTHESIZED here — minimal objects shaped like the engines'
// config.json / report.json. No workspace bytes are copied into the repo.

const config = (runType?: string) => runType === undefined ? {} : { runType, schemaVersion: 3 };

describe("detectRunKind — the config.json stamp is authoritative", () => {
  it("reads each stamped run type", () => {
    const cases: Array<[string, string]> = [
      ["run", "study-run"],
      ["sweep", "sweep"],
      ["validate", "validate"],
      ["analyze", "analyze"],
      ["extract", "extract"],
      ["rescore-style", "rescore-style"],
      ["pipeline", "pipeline"],
      ["multi-agent", "multi-agent"],
    ];
    for (const [runType, kind] of cases) {
      const detected = detectRunKind(config(runType), {}, []);
      expect(detected.kind, runType).toBe(kind);
      expect(detected.source).toBe("runType");
      expect(detected.stampedRunType).toBe(runType);
    }
  });

  it("prefers the stamp over an artifact that would suggest another kind", () => {
    // A validate directory that also happens to hold a sweep grid: the stamp
    // wins, and the artifact fallback is never consulted.
    const detected = detectRunKind(config("validate"), {}, ["sweep.csv", "validation-report.json"]);
    expect(detected.kind).toBe("validate");
    expect(detected.source).toBe("runType");
  });

  it("resolves an unrecognized stamp by artifacts rather than guessing", () => {
    const detected = detectRunKind(config("neutral-pcs"), {}, ["neutral-pcs.json"]);
    expect(detected.kind).toBe("unknown");
    expect(detected.source).toBe("none");
    // The stamp is still reported verbatim — it is the citable fact.
    expect(detected.stampedRunType).toBe("neutral-pcs");
  });
});

describe("detectRunKind — evaluate splits into paired vs coding", () => {
  it("reads codings.jsonl / coding-report.json as per-response coding", () => {
    expect(detectRunKind(config("evaluate"), {}, ["codings.jsonl", "judging-context.json"]).kind)
      .toBe("evaluate-coding");
    expect(detectRunKind(config("evaluate"), {}, ["coding-report.json", "judging-context.json"]).kind)
      .toBe("evaluate-coding");
  });

  it("reads judgments.jsonl / judge-report.json as paired judging", () => {
    expect(detectRunKind(config("evaluate"), {}, ["judgments.jsonl", "judging-context.json"]).kind)
      .toBe("evaluate-paired");
    expect(detectRunKind(config("evaluate-judgment"), {}, ["judge-report.json"]).kind)
      .toBe("evaluate-paired");
  });

  it("does not split on judging-context.json, which both flavours write", () => {
    // Nothing but the shared file: the flavour is undecidable, and the
    // manifest default (paired) is used.
    expect(detectRunKind(config("evaluate"), {}, ["judging-context.json"]).kind)
      .toBe("evaluate-paired");
  });
});

describe("detectRunKind — a multi-agent study is stamped `run`", () => {
  it("refines a `run` stamp to multi-agent on stored seat evidence", () => {
    const report = { modelBySeat: { "Judge A": "google/gemma-3-27b-it" }, conditions: {} };
    const detected = detectRunKind(config("run"), report, ["report.json", "generations.jsonl"]);
    expect(detected.kind).toBe("multi-agent");
    expect(detected.stampedRunType).toBe("run");
  });

  it("refines a `run` stamp to multi-agent when panel-effects.csv is present", () => {
    expect(detectRunKind(config("run"), {}, ["panel-effects.csv"]).kind).toBe("multi-agent");
  });

  it("leaves an ordinary study run alone", () => {
    const report = { conditions: { baseline: { generations: 48 } } };
    expect(detectRunKind(config("run"), report, ["report.json", "generations.jsonl"]).kind)
      .toBe("study-run");
  });
});

describe("detectRunKind — artifact fallback for unstamped directories", () => {
  it("recognizes a pipeline ledger, a sweep, and a validation", () => {
    expect(detectRunKind({}, {}, ["pipeline.json"]).kind).toBe("pipeline");
    expect(detectRunKind({}, {}, ["pipeline-portable.json"]).kind).toBe("pipeline");
    expect(detectRunKind({}, {}, ["sweep.csv", "recommendations.json"]).kind).toBe("sweep");
    expect(detectRunKind({}, {}, ["validation-report.json"]).kind).toBe("validate");
  });

  it("separates an analyze directory from a study run by its source stamp", () => {
    expect(detectRunKind({}, {}, ["effect-sizes.csv", "source-run.txt"]).kind).toBe("analyze");
    expect(detectRunKind({}, {}, ["report.json", "generations.jsonl", "effect-sizes.csv"]).kind)
      .toBe("study-run");
  });

  it("matches on the basename, so nested paths still resolve", () => {
    const detected = detectRunKind({}, {}, ["shards/shard0/sweep.csv"]);
    expect(detected.kind).toBe("sweep");
    expect(detected.source).toBe("artifacts");
  });

  it("calls a directory it cannot recognize unknown rather than a study run", () => {
    const detected = detectRunKind({}, {}, ["notes.txt", "anger.safetensors"]);
    expect(detected.kind).toBe("unknown");
    expect(detected.source).toBe("none");
    expect(detected.stampedRunType).toBe("");
  });
});

describe("runKindLabel", () => {
  it("names every kind", () => {
    expect(runKindLabel("evaluate-coding")).toBe("Response coding");
    expect(runKindLabel("unknown")).toBe("Unrecognized run");
  });
});
