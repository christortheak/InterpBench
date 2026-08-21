// Run-kind detection — ONE module, read by the nav, the run picker, triage,
// and the overview (docs/RESULTS-EXPLORER-UPGRADE-PLAN.md, Phase 0).
//
// Ground truth is the engines' canonical `config.json` `runType` stamp
// (schema 2/3): the Python writer is
// `Server/steerlab_server/experiment/run_config.py::write_run_config`
// (called with the stage's task name from `tasks.py::_write_config_snapshot`),
// and the Swift twin is `Sources/ExperimentKit/RunMetadata.swift`. Stamped
// values seen in both engines: extract | validate | sweep | run | evaluate |
// evaluate-judgment | evaluate-awaiting | analyze | pipeline | rescore-style |
// multi-agent | variant-save | lora-train | neutral-pcs | probe-train |
// reader-fit | norm-backfill | sae-feature-import | gemmascope-analysis |
// jlens-support | variant-robustness | derive-reader-vector. It is an OPEN
// set, so an unrecognized stamp resolves to "unknown" rather than being
// guessed at.
//
// Two refinements the stamp alone cannot make, both resolved by artifacts:
//
// 1. `evaluate` covers two different instruments. Paired judging writes
//    `judgments.jsonl` + `judge-report.json`; per-response coding writes
//    `codings.jsonl` + `coding-report.json`. `judging-context.json` exists on
//    BOTH and therefore splits nothing.
// 2. A multi-agent STUDY is stamped `run`, not `multi-agent` — the manifest
//    path passes task "run" for panels too (tasks.py: "incl. the multi-agent
//    study path, which passes task 'run'"), and only the scenario API route
//    stamps "multi-agent". So a `run` whose report carries seat evidence
//    (`modelBySeat`) or which has `panel-effects.csv` is reported as
//    multi-agent. This is the one place kind detection reads past the stamp,
//    and it reads STORED evidence, never prose.
//
// Absence stays absence: a directory with no stamp and no recognizable
// artifact is "unknown", never optimistically a study run.

export type RunKind =
  | "study-run"
  | "multi-agent"
  | "sweep"
  | "evaluate-paired"
  | "evaluate-coding"
  | "validate"
  | "analyze"
  | "extract"
  | "rescore-style"
  | "pipeline"
  | "unknown";

/// How the kind was established — surfaced so a viewer can say "stamped" vs
/// "inferred from the files present" rather than implying the engine
/// declared something it did not.
export type RunKindSource = "runType" | "artifacts" | "none";

export type RunKindInfo = {
  kind: RunKind;
  source: RunKindSource;
  /// The raw `config.json` stamp, verbatim, or "" when the run carries none.
  /// Kept even when it did not decide the kind (e.g. `run` → multi-agent),
  /// because the stamp is the citable fact.
  stampedRunType: string;
};

export const runKindLabel = (kind: RunKind): string => ({
  "study-run": "Study run",
  "multi-agent": "Multi-agent panel",
  sweep: "Sweep",
  "evaluate-paired": "Paired judging",
  "evaluate-coding": "Response coding",
  validate: "Validation",
  analyze: "Analysis",
  extract: "Extraction",
  "rescore-style": "Reasoning-style rescore",
  pipeline: "Pipeline ledger",
  unknown: "Unrecognized run",
}[kind]);

const basenames = (artifacts: string[]) =>
  new Set(artifacts.map((path) => path.split("/").pop() ?? path));

const record = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};

/// Seat-bearing evidence: the multi-agent report's per-seat model map, or the
/// panel decomposition artifact. Both are engine-written.
const looksMultiAgent = (report: Record<string, unknown>, names: Set<string>) =>
  names.has("panel-effects.csv")
  || Object.keys(record(report.modelBySeat)).length > 0
  || names.has("scenario.json");

const evaluateFlavour = (names: Set<string>): RunKind => {
  if (names.has("coding-report.json") || names.has("codings.jsonl")) return "evaluate-coding";
  if (names.has("judge-report.json") || names.has("judgments.jsonl")) return "evaluate-paired";
  // An evaluate that failed before writing either row file (the 2026-08-05
  // OpenRouter failure wrote judgments.jsonl, but an earlier failure need
  // not). Paired judging is the manifest default; the flavour is a guess and
  // the nav's escape hatch exists for exactly this.
  return "evaluate-paired";
};

const fromRunType = (
  runType: string,
  report: Record<string, unknown>,
  names: Set<string>,
): RunKind | null => {
  switch (runType) {
    case "run": return looksMultiAgent(report, names) ? "multi-agent" : "study-run";
    case "multi-agent": return "multi-agent";
    case "sweep": return "sweep";
    case "evaluate":
    case "evaluate-judgment":
    case "evaluate-awaiting": return evaluateFlavour(names);
    case "validate": return "validate";
    case "analyze": return "analyze";
    case "extract": return "extract";
    case "rescore-style": return "rescore-style";
    case "pipeline": return "pipeline";
    default: return null;
  }
};

/// Artifact-presence fallback for a run with no (or an unrecognized) stamp.
/// Ordered most-specific first: a pipeline ledger and a sweep grid are
/// unambiguous, a bare `generations.jsonl` is not.
const fromArtifacts = (report: Record<string, unknown>, names: Set<string>): RunKind | null => {
  if (names.has("pipeline.json") || names.has("pipeline-portable.json")) return "pipeline";
  if (names.has("sweep.csv")) return "sweep";
  if (names.has("validation-report.json")) return "validate";
  if (names.has("coding-report.json") || names.has("codings.jsonl")) return "evaluate-coding";
  if (names.has("judge-report.json") || names.has("judgments.jsonl")) return "evaluate-paired";
  if (looksMultiAgent(report, names)) return "multi-agent";
  if (names.has("analysis.json")) return "analyze";
  // effect-sizes.csv beside a source-run stamp is an analyze directory; the
  // same file inside a run directory is that run's own table.
  if (names.has("effect-sizes.csv") && names.has("source-run.txt")) return "analyze";
  if (names.has("generations.jsonl") || names.has("report.json")) return "study-run";
  return null;
};

/// The one entry point. `artifacts` is the run's file list (paths or bare
/// names — both are accepted; only the basename is matched).
export const detectRunKind = (
  config: Record<string, unknown>,
  report: Record<string, unknown>,
  artifacts: string[],
): RunKindInfo => {
  const names = basenames(artifacts);
  const stamped = typeof config.runType === "string" ? config.runType.trim() : "";
  if (stamped) {
    const kind = fromRunType(stamped, report, names);
    if (kind) return { kind, source: "runType", stampedRunType: stamped };
  }
  const inferred = fromArtifacts(report, names);
  if (inferred) return { kind: inferred, source: "artifacts", stampedRunType: stamped };
  return { kind: "unknown", source: "none", stampedRunType: stamped };
};
