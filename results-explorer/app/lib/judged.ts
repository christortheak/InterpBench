// Judged-evaluation artifact readers (Phase 2 of
// docs/RESULTS-EXPLORER-UPGRADE-PLAN.md): judge-report.json,
// judgments.jsonl, judging-context.json, run-status.json, FAILED.md — plus
// the source-run generation join both Phase-2 readers use.
//
// Two engines write these files and their key vocabularies differ (the
// server writes `outcome: "variant"` and `sourceRun`; Swift writes
// `conditionResult: "condition"` and `sourceRunDirectory`). Everything is
// normalized HERE, defensively, so a view never asks which engine wrote a
// run. Absent keys stay absent: a missing tally is null, never zero, and a
// missing outcome is "unknown", never guessed from the winner letter.
//
// Nothing in this module computes an inferential statistic. Percent
// agreement and Cohen's kappa are read from the engine's report and never
// recomputed; the only viewer-computed quantities exported here are counts,
// means, distributions and disagreement selections, and every caller badges
// them as derived.

import { findFile } from "./discovery";
import type { RunFile, WorkspaceRun } from "./types";

/// Same bounded-read discipline as lib/loaders.ts: read at most 32 MB and
/// report the truncation honestly rather than pretending to hold the file.
export const READ_LIMIT = 32 * 1024 * 1024;

export const readBoundedText = async (file: RunFile): Promise<{ text: string; truncated: boolean }> => {
  const blob = await file.handle.getFile();
  const truncated = blob.size > READ_LIMIT;
  const text = await blob.slice(0, READ_LIMIT).text();
  return { text, truncated };
};

export type LoadedJSON = { present: boolean; raw: Record<string, unknown>; error: string; file: RunFile | null };

export const readJSONArtifact = async (run: WorkspaceRun, name: string): Promise<LoadedJSON> => {
  const file = findFile(run.files, name);
  if (!file) return { present: false, raw: {}, error: "", file: null };
  try {
    const { text } = await readBoundedText(file);
    const parsed: unknown = JSON.parse(text);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return { present: true, raw: {}, error: `${name} is present but is not a JSON object.`, file };
    return { present: true, raw: parsed as Record<string, unknown>, error: "", file };
  } catch {
    return { present: true, raw: {}, error: `${name} is present but could not be parsed as JSON.`, file };
  }
};

/// A bounded read cuts the final record mid-line; that partial line is
/// dropped rather than parsed into a half record.
export const jsonlLines = (text: string, truncated: boolean): string[] => {
  const lines = text.split(/\r?\n/);
  if (truncated) lines.pop();
  return lines.filter((line) => line.trim());
};

const str = (value: unknown): string => typeof value === "string" ? value : "";
const num = (value: unknown): number | null => typeof value === "number" && Number.isFinite(value) ? value : null;
const record = (value: unknown): Record<string, unknown> => value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
const list = (value: unknown): unknown[] => Array.isArray(value) ? value : [];
const first = <T,>(...values: (T | null)[]): T | null => values.find((value) => value != null) ?? null;

/// Seeds are UInt64 on both engines and JSON.parse silently rounds them
/// (5230848306049226115 becomes …226000). The exact digits are lifted out
/// of the raw line text so a displayed seed is the seed that was stamped.
export const rawIntegerField = (line: string, key: string): string => {
  const match = new RegExp(`"${key}"\\s*:\\s*(-?\\d+)`).exec(line);
  return match ? match[1] : "";
};

export type JudgmentOutcome = "baseline" | "variant" | "tie" | "unknown";

/// Swift's vocabulary calls the non-baseline arm "condition"; the server
/// calls it "variant". One word wins in the viewer: variant.
export const normalizeOutcome = (raw: unknown): JudgmentOutcome => {
  const value = str(raw).trim().toLowerCase();
  if (value === "variant" || value === "condition") return "variant";
  if (value === "baseline") return "baseline";
  if (value === "tie") return "tie";
  return "unknown";
};

export type JudgmentRow = {
  judge: string;
  judgeKind: string;
  judgeModel: string;
  judgeProvider: string;
  judgeRevision: string;
  condition: string;
  promptID: string;
  sampleIndex: number;
  outcome: JudgmentOutcome;
  winner: string;
  baselineWas: string;
  conditionWas: string;
  confidence: number | null;
  briefReason: string;
  reasoningTruncated: boolean;
  aScores: Record<string, number>;
  bScores: Record<string, number>;
  structuredFields: Record<string, unknown> | null;
  prompt: string;
  baselineSeed: string;
  variantSeed: string;
  dialect: "server" | "swift";
};

const scoreMap = (value: unknown): Record<string, number> => Object.fromEntries(Object.entries(record(value)).flatMap(([key, entry]) => typeof entry === "number" && Number.isFinite(entry) ? [[key, entry] as [string, number]] : []));

export const parseJudgmentRows = (text: string, truncated = false): { rows: JudgmentRow[]; skipped: number } => {
  let skipped = 0;
  const rows = jsonlLines(text, truncated).flatMap((line): JudgmentRow[] => {
    let parsed: unknown;
    try { parsed = JSON.parse(line); } catch { skipped += 1; return []; }
    const raw = record(parsed);
    if (!Object.keys(raw).length) { skipped += 1; return []; }
    const judgment = record(raw.judgment);
    const condition = str(raw.condition);
    const promptID = str(raw.promptID);
    // A row with no condition and no prompt id cannot be joined to
    // anything; counting it would inflate every tally on screen.
    if (!condition && !promptID) { skipped += 1; return []; }
    const swift = "conditionResult" in raw || "sourceRunDirectory" in raw;
    return [{
      judge: str(raw.judge) || "Unnamed judge",
      judgeKind: str(raw.judgeKind),
      judgeModel: str(raw.judgeModel),
      judgeProvider: str(raw.judgeProvider) || str(judgment.provider),
      judgeRevision: str(raw.judgeRevision),
      condition,
      promptID,
      sampleIndex: num(raw.sampleIndex) ?? 0,
      outcome: normalizeOutcome("outcome" in raw ? raw.outcome : raw.conditionResult),
      winner: str(judgment.winner),
      baselineWas: str(raw.baselineWas),
      conditionWas: str(raw.conditionWas),
      confidence: first(num(raw.confidence), num(judgment.confidence)),
      briefReason: str(judgment.brief_reason) || str(judgment.briefReason),
      reasoningTruncated: judgment.reasoningTruncated === true,
      aScores: scoreMap(judgment.a_scores ?? judgment.aScores),
      bScores: scoreMap(judgment.b_scores ?? judgment.bScores),
      structuredFields: judgment.structured_fields || judgment.structuredFields ? record(judgment.structured_fields ?? judgment.structuredFields) : null,
      prompt: str(raw.prompt),
      baselineSeed: rawIntegerField(line, "baselineSeed"),
      variantSeed: rawIntegerField(line, "variantSeed"),
      dialect: swift ? "swift" : "server",
    }];
  });
  return { rows, skipped };
};

export const loadJudgments = async (run: WorkspaceRun) => {
  const file = findFile(run.files, "judgments.jsonl");
  if (!file) return { rows: [] as JudgmentRow[], skipped: 0, truncated: false, file: null, present: false };
  const { text, truncated } = await readBoundedText(file);
  const { rows, skipped } = parseJudgmentRows(text, truncated);
  return { rows, skipped, truncated, file, present: true };
};

// ---------------------------------------------------------------------------
// judge-report.json — two dialects, one normalized shape.
// ---------------------------------------------------------------------------

export type ConditionTally = {
  condition: string;
  pairs: number | null;
  variantWins: number | null;
  baselineWins: number | null;
  ties: number | null;
  meanConfidence: number | null;
  structuredSummaries: Record<string, unknown> | null;
};

export type JudgeDetail = {
  name: string;
  kind: string;
  requestedModel: string;
  actualModel: string;
  revision: string;
  provider: string;
  dtype: string;
  pairs: number | null;
  conditions: ConditionTally[];
};

export type AgreementRow = { judgeA: string; judgeB: string; n: number | null; percentAgreement: number | null; kappa: number | null };
export type HumanAgreementRow = { judge: string; n: number | null; percentAgreement: number | null; kappa: number | null };

export type JudgeReport = {
  present: boolean;
  error: string;
  dialect: "server" | "swift" | "unknown";
  experiment: string;
  experimentHash: string;
  sourceRun: string;
  judgeModel: string;
  pairs: number | null;
  judges: JudgeDetail[];
  conditions: ConditionTally[];
  agreement: AgreementRow[];
  humanAgreement: HumanAgreementRow[];
  rubricFile: string;
  rubricHash: string;
  evaluationSource: string;
  epochUnverified: boolean;
  measurementDrift: string;
  exclusions: Record<string, unknown> | null;
  reusedJudgments: number | null;
  freshJudgments: number | null;
  judgedOn: string;
};

const conditionTally = (condition: string, raw: unknown): ConditionTally => {
  const entry = record(raw);
  const structured = entry.structuredSummaries;
  return {
    condition,
    // server: n · swift: pairs
    pairs: first(num(entry.n), num(entry.pairs)),
    // server: variantWins · swift: conditionWins
    variantWins: first(num(entry.variantWins), num(entry.conditionWins)),
    baselineWins: num(entry.baselineWins),
    ties: num(entry.ties),
    meanConfidence: num(entry.meanConfidence),
    structuredSummaries: structured && typeof structured === "object" && Object.keys(record(structured)).length ? record(structured) : null,
  };
};

const conditionTallies = (raw: unknown): ConditionTally[] => Object.entries(record(raw)).map(([condition, entry]) => conditionTally(condition, entry)).sort((left, right) => left.condition === "baseline" ? -1 : right.condition === "baseline" ? 1 : left.condition.localeCompare(right.condition));

const judgeDetails = (raw: unknown): JudgeDetail[] => list(raw).flatMap((item): JudgeDetail[] => {
  // Swift stamps `judges` as bare names; the server stamps objects.
  if (typeof item === "string") return [{ name: item, kind: "", requestedModel: "", actualModel: "", revision: "", provider: "", dtype: "", pairs: null, conditions: [] }];
  const entry = record(item);
  const name = str(entry.name);
  if (!name) return [];
  return [{
    name,
    kind: str(entry.kind),
    requestedModel: str(entry.requestedModel) || str(entry.model),
    actualModel: str(entry.actualModel),
    revision: str(entry.revision),
    provider: str(entry.provider),
    dtype: str(entry.actualDtype) || str(entry.dtype),
    pairs: num(entry.pairs),
    conditions: conditionTallies(entry.conditions),
  }];
});

const agreementRows = (raw: unknown): AgreementRow[] => list(raw).flatMap((item): AgreementRow[] => {
  const entry = record(item);
  // Server rows name the pair as a two-element `judges` array; Swift rows
  // (and some server versions) use judgeA/judgeB.
  const pair = list(entry.judges).filter((name): name is string => typeof name === "string");
  const judgeA = str(entry.judgeA) || pair[0] || "";
  const judgeB = str(entry.judgeB) || pair[1] || "";
  if (!judgeA || !judgeB) return [];
  return [{ judgeA, judgeB, n: first(num(entry.n), num(entry.items), num(entry.pairs)), percentAgreement: num(entry.percentAgreement), kappa: num(entry.kappa) }];
});

const humanAgreementRows = (raw: unknown): HumanAgreementRow[] => list(raw).flatMap((item): HumanAgreementRow[] => {
  const entry = record(item);
  const judge = str(entry.judge);
  if (!judge) return [];
  return [{ judge, n: first(num(entry.n), num(entry.items)), percentAgreement: num(entry.percentAgreement), kappa: num(entry.kappa) }];
});

export const parseJudgeReport = (loaded: LoadedJSON): JudgeReport => {
  const raw = loaded.raw;
  const swiftShaped = "judgeAgreement" in raw || "sourceRunDirectory" in raw;
  const serverShaped = "agreement" in raw || "sourceRun" in raw || "judgedOn" in raw;
  return {
    present: loaded.present && !loaded.error,
    error: loaded.error,
    dialect: swiftShaped ? "swift" : serverShaped ? "server" : "unknown",
    experiment: str(raw.experiment),
    experimentHash: str(raw.experimentHash),
    sourceRun: str(raw.sourceRun) || str(raw.sourceRunDirectory),
    judgeModel: str(raw.judgeModel),
    pairs: num(raw.pairs),
    judges: judgeDetails(raw.judges),
    conditions: conditionTallies(raw.conditions),
    agreement: agreementRows(raw.agreement ?? raw.judgeAgreement),
    humanAgreement: humanAgreementRows(raw.humanAgreement),
    rubricFile: str(raw.rubricFile) || str(raw.judgeRubricFile),
    rubricHash: str(raw.rubricHash) || str(raw.judgeRubricHash),
    evaluationSource: str(raw.evaluationSource),
    epochUnverified: raw.epochUnverified === true,
    measurementDrift: str(raw.measurementDrift),
    exclusions: Object.keys(record(raw.exclusions)).length ? record(raw.exclusions) : null,
    reusedJudgments: num(raw.reusedJudgments),
    freshJudgments: num(raw.freshJudgments),
    judgedOn: str(raw.judgedOn),
  };
};

// ---------------------------------------------------------------------------
// judging-context.json — the measurement pins, written before judging so a
// FAILED evaluate still says exactly what it was about to do.
// ---------------------------------------------------------------------------

export type ContextJudge = { name: string; kind: string; model: string; provider: string; revision: string; dtype: string };

export type JudgingContext = {
  present: boolean;
  error: string;
  experiment: string;
  experimentHash: string;
  judges: ContextJudge[];
  rubricFile: string;
  rubricHash: string;
  sourceRun: string;
  sourceGenerationsSha256: string;
  structuredPromptSha256: string;
};

export const parseJudgingContext = (loaded: LoadedJSON): JudgingContext => {
  const raw = loaded.raw;
  return {
    present: loaded.present && !loaded.error,
    error: loaded.error,
    experiment: str(raw.experiment),
    experimentHash: str(raw.experimentHash),
    judges: list(raw.judges).flatMap((item): ContextJudge[] => {
      const entry = record(item);
      const name = str(entry.name);
      if (!name) return [];
      return [{ name, kind: str(entry.kind), model: str(entry.model), provider: str(entry.provider), revision: str(entry.revision), dtype: str(entry.dtype) }];
    }),
    rubricFile: str(raw.rubricFile),
    rubricHash: str(raw.rubricHash),
    sourceRun: str(raw.sourceRun),
    sourceGenerationsSha256: str(raw.sourceGenerationsSha256),
    structuredPromptSha256: str(raw.structuredPromptSha256),
  };
};

// ---------------------------------------------------------------------------
// run-status.json — status truth, including the invalid-response count and
// which judges never ran.
// ---------------------------------------------------------------------------

export type RunStatus = {
  present: boolean;
  status: string;
  stage: string;
  experiment: string;
  sourceRun: string;
  itemsWritten: number | null;
  invalidResponses: number | null;
  itemLabel: string;
  expectedUnits: string[];
  completedUnits: string[];
  pendingUnits: string[];
  evidenceComplete: boolean | null;
  error: string;
  errorType: string;
};

const stringList = (value: unknown): string[] => list(value).filter((item): item is string => typeof item === "string");

export const parseRunStatus = (loaded: LoadedJSON): RunStatus => {
  const raw = loaded.raw;
  return {
    present: loaded.present && !loaded.error,
    // Never default an unstamped run to a happy status.
    status: str(raw.status) || (loaded.present ? "not stamped" : ""),
    stage: str(raw.stage),
    experiment: str(raw.experiment),
    sourceRun: str(raw.sourceRun),
    itemsWritten: num(raw.itemsWritten),
    invalidResponses: num(raw.invalidResponses),
    itemLabel: str(raw.itemLabel) || "item",
    expectedUnits: stringList(raw.expectedUnits),
    completedUnits: stringList(raw.completedUnits),
    pendingUnits: stringList(raw.pendingUnits),
    evidenceComplete: typeof raw.evidenceComplete === "boolean" ? raw.evidenceComplete : null,
    error: str(raw.error),
    errorType: str(raw.errorType),
  };
};

// ---------------------------------------------------------------------------
// Viewer-derived selections and rollups. Every function below is badged
// `derived` wherever its output reaches the screen.
// ---------------------------------------------------------------------------

/// The cross-engine join key for one sampled response: (condition,
/// promptID, sampleIndex) — never a seed, which differs between the two
/// sides of a pair under derived seeding. Unit-separated so no two distinct
/// triples can collide.
export const cellKey = (condition: string, promptID: string, sampleIndex: number) => `${condition}${promptID}${sampleIndex}`;

export type JudgmentCell = {
  key: string;
  condition: string;
  promptID: string;
  sampleIndex: number;
  rows: JudgmentRow[];
  /// One outcome per judge (a judge's last row wins), in judge order.
  verdicts: { judge: string; outcome: JudgmentOutcome }[];
  disagrees: boolean;
};

/// Group judgment rows into (condition, promptID, sampleIndex) cells and
/// mark the cells where the judges split. A cell disagrees when two
/// DIFFERENT judges recorded two different known outcomes — so tie-vs-win
/// counts, and a single judge's repeated rows never do. Rows whose outcome
/// is unknown (neither engine key present) are shown but cannot create a
/// disagreement.
export const judgmentCells = (rows: JudgmentRow[]): JudgmentCell[] => {
  const cells = new Map<string, JudgmentCell>();
  for (const row of rows) {
    const key = cellKey(row.condition, row.promptID, row.sampleIndex);
    const cell = cells.get(key) ?? { key, condition: row.condition, promptID: row.promptID, sampleIndex: row.sampleIndex, rows: [], verdicts: [], disagrees: false };
    cell.rows.push(row);
    cells.set(key, cell);
  }
  for (const cell of cells.values()) {
    const byJudge = new Map<string, JudgmentOutcome>();
    for (const row of cell.rows) byJudge.set(row.judge, row.outcome);
    cell.verdicts = [...byJudge.entries()].map(([judge, outcome]) => ({ judge, outcome }));
    const known = cell.verdicts.filter((verdict) => verdict.outcome !== "unknown");
    cell.disagrees = known.length > 1 && new Set(known.map((verdict) => verdict.outcome)).size > 1;
  }
  return [...cells.values()].sort((left, right) => left.condition.localeCompare(right.condition) || left.promptID.localeCompare(right.promptID) || left.sampleIndex - right.sampleIndex);
};

export const disagreementCells = (rows: JudgmentRow[]): JudgmentCell[] => judgmentCells(rows).filter((cell) => cell.disagrees);

export type JudgeTally = { judge: string; condition: string; variantWins: number; baselineWins: number; ties: number; unknown: number; n: number; meanConfidence: number | null };

/// Per-judge × condition splits counted from the judgment rows — the split
/// the server's report carries per judge and Swift's does not.
export const judgeTallies = (rows: JudgmentRow[]): JudgeTally[] => {
  const tallies = new Map<string, JudgeTally & { confidenceSum: number; confidenceN: number }>();
  for (const row of rows) {
    const key = `${row.judge}${row.condition}`;
    const tally = tallies.get(key) ?? { judge: row.judge, condition: row.condition, variantWins: 0, baselineWins: 0, ties: 0, unknown: 0, n: 0, meanConfidence: null, confidenceSum: 0, confidenceN: 0 };
    if (row.outcome === "variant") tally.variantWins += 1;
    else if (row.outcome === "baseline") tally.baselineWins += 1;
    else if (row.outcome === "tie") tally.ties += 1;
    else tally.unknown += 1;
    tally.n += 1;
    if (row.confidence != null) { tally.confidenceSum += row.confidence; tally.confidenceN += 1; }
    tallies.set(key, tally);
  }
  return [...tallies.values()].map(({ confidenceSum, confidenceN, ...tally }) => ({ ...tally, meanConfidence: confidenceN ? confidenceSum / confidenceN : null })).sort((left, right) => left.condition.localeCompare(right.condition) || left.judge.localeCompare(right.judge));
};

export type ConfidenceBin = { from: number; to: number; count: number };

/// Ten fixed 0.1-wide bins over [0, 1]. Rows with no stamped confidence are
/// excluded and reported separately by the caller.
export const confidenceHistogram = (rows: JudgmentRow[]): { bins: ConfidenceBin[]; counted: number; missing: number } => {
  const bins: ConfidenceBin[] = Array.from({ length: 10 }, (_, index) => ({ from: index / 10, to: (index + 1) / 10, count: 0 }));
  let counted = 0;
  let missing = 0;
  for (const row of rows) {
    if (row.confidence == null) { missing += 1; continue; }
    const clamped = Math.min(0.999999, Math.max(0, row.confidence));
    bins[Math.floor(clamped * 10)].count += 1;
    counted += 1;
  }
  return { bins, counted, missing };
};

// ---------------------------------------------------------------------------
// The source-run join (shared with the coding reader): a judgment or coding
// row names (condition, promptID, sampleIndex); the source run's
// generations.jsonl holds the text that was judged.
// ---------------------------------------------------------------------------

export type SourceResponse = { condition: string; promptID: string; sampleIndex: number; prompt: string; output: string; wordCount: number | null; seed: string; modelID: string };

export const parseSourceResponses = (text: string, truncated = false): Map<string, SourceResponse> => {
  const responses = new Map<string, SourceResponse>();
  for (const line of jsonlLines(text, truncated)) {
    let parsed: unknown;
    try { parsed = JSON.parse(line); } catch { continue; }
    const raw = record(parsed);
    const condition = str(raw.condition);
    const promptID = str(raw.promptID);
    if (!condition || !promptID || typeof raw.output !== "string") continue;
    const sampleIndex = num(raw.sampleIndex) ?? 0;
    responses.set(cellKey(condition, promptID, sampleIndex), {
      condition, promptID, sampleIndex,
      prompt: str(raw.prompt),
      output: raw.output,
      wordCount: num(raw.wordCount),
      seed: rawIntegerField(line, "seed"),
      modelID: str(raw.modelID),
    });
  }
  return responses;
};

export const findSourceRun = (runs: WorkspaceRun[], sourceRun: string): WorkspaceRun | null => {
  if (!sourceRun) return null;
  const wanted = sourceRun.replace(/\/+$/, "").split("/").pop() ?? sourceRun;
  return runs.find((run) => run.name === wanted) ?? runs.find((run) => run.path === sourceRun) ?? null;
};

export const loadSourceResponses = async (run: WorkspaceRun) => {
  const file = findFile(run.files, "generations.jsonl");
  if (!file) return { responses: new Map<string, SourceResponse>(), truncated: false, present: false };
  const { text, truncated } = await readBoundedText(file);
  return { responses: parseSourceResponses(text, truncated), truncated, present: true };
};
