// Per-response coding artifact readers (the 2026-08-04 instrument):
// coding-report.json + codings.jsonl. Ground truth for the shapes is
// Server/steerlab_server/experiment/response_coding.py +
// tasks._evaluate_response_coding and their Swift twins
// (ExperimentTasks.CodingRecord / CodingReport) — the two engines write the
// SAME keys here, so this module normalizes types and absence rather than
// dialects.
//
// The instrument codes each response individually and blinded: there is no
// pair and no winner anywhere on this path. Aggregates, per-field agreement
// (percent / kappa / mean absolute difference) and word-count means are
// read from the report; the viewer only counts, filters, joins, and selects
// disagreements — all badged derived where they reach the screen.

import { findFile } from "./discovery";
import { cellKey, jsonlLines, rawIntegerField, readBoundedText, type LoadedJSON } from "./judged";
import type { WorkspaceRun } from "./types";

const str = (value: unknown): string => typeof value === "string" ? value : "";
const num = (value: unknown): number | null => typeof value === "number" && Number.isFinite(value) ? value : null;
const record = (value: unknown): Record<string, unknown> => value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
const list = (value: unknown): unknown[] => Array.isArray(value) ? value : [];

export type CodeValue = boolean | number | string | null;

/// A code is one of the declared JSON scalar types, or null for an optional
/// field the coder left empty. Anything else (an object, an array) is a
/// contract violation and is surfaced as null with the raw text preserved
/// by `formatCode`.
export const codeValue = (value: unknown): CodeValue => typeof value === "boolean" || typeof value === "string" ? value : typeof value === "number" && Number.isFinite(value) ? value : null;

export const formatCode = (value: CodeValue): string => value === null ? "null" : typeof value === "boolean" ? (value ? "true" : "false") : String(value);

export type CodingRow = {
  experiment: string;
  condition: string;
  promptID: string;
  sampleIndex: number;
  seed: string;
  wordCount: number | null;
  codes: Record<string, CodeValue>;
  briefReason: string;
  judge: string;
  judgeKind: string;
  judgeModel: string;
  judgeProvider: string;
  judgeRevision: string;
};

export const parseCodingRows = (text: string, truncated = false): { rows: CodingRow[]; skipped: number } => {
  let skipped = 0;
  const rows = jsonlLines(text, truncated).flatMap((line): CodingRow[] => {
    let parsed: unknown;
    try { parsed = JSON.parse(line); } catch { skipped += 1; return []; }
    const raw = record(parsed);
    const condition = str(raw.condition);
    const promptID = str(raw.promptID);
    if (!condition && !promptID) { skipped += 1; return []; }
    return [{
      experiment: str(raw.experiment),
      condition,
      promptID,
      sampleIndex: num(raw.sampleIndex) ?? 0,
      // UInt64 seeds lose digits through JSON.parse; read the stamped ones.
      seed: rawIntegerField(line, "seed"),
      // wordCount is ENGINE-computed — never recounted here.
      wordCount: num(raw.wordCount),
      codes: Object.fromEntries(Object.entries(record(raw.codes)).map(([field, value]) => [field, codeValue(value)])),
      briefReason: str(raw.briefReason),
      judge: str(raw.judge) || "Unnamed judge",
      judgeKind: str(raw.judgeKind),
      judgeModel: str(raw.judgeModel),
      judgeProvider: str(raw.judgeProvider),
      judgeRevision: str(raw.judgeRevision),
    }];
  });
  return { rows, skipped };
};

export const loadCodings = async (run: WorkspaceRun) => {
  const file = findFile(run.files, "codings.jsonl");
  if (!file) return { rows: [] as CodingRow[], skipped: 0, truncated: false, file: null, present: false };
  const { text, truncated } = await readBoundedText(file);
  const { rows, skipped } = parseCodingRows(text, truncated);
  return { rows, skipped, truncated, file, present: true };
};

// ---------------------------------------------------------------------------
// coding-report.json
// ---------------------------------------------------------------------------

export type CodingFieldSpec = { name: string; type: string; optional: boolean; values: string[] };

export type FieldAggregate = {
  n: number | null;
  nulls: number | null;
  trueCount: number | null;
  trueShare: number | null;
  mean: number | null;
  counts: Record<string, number> | null;
};

export type CodingConditionRow = {
  condition: string;
  codedResponses: number | null;
  codings: number | null;
  meanWordCount: number | null;
  fields: Record<string, FieldAggregate>;
};

export type CodingAgreementRow = {
  field: string;
  judgeA: string;
  judgeB: string;
  n: number | null;
  percentAgreement: number | null;
  kappa: number | null;
  meanAbsoluteDifference: number | null;
};

export type CodingJudgeDetail = { name: string; kind: string; requestedModel: string; actualModel: string; revision: string };

export type CodingReport = {
  present: boolean;
  error: string;
  mode: string;
  experiment: string;
  experimentHash: string;
  sourceRun: string;
  judges: string[];
  judgeModel: string;
  judgeDetails: CodingJudgeDetail[];
  rubricFile: string;
  rubricHash: string;
  fields: CodingFieldSpec[];
  codings: number | null;
  conditions: CodingConditionRow[];
  fieldAgreement: CodingAgreementRow[];
  evaluationSource: string;
  epochUnverified: boolean;
  measurementDrift: string;
  exclusions: Record<string, unknown> | null;
};

const fieldAggregate = (raw: unknown): FieldAggregate => {
  const entry = record(raw);
  const counts = record(entry.counts);
  return {
    n: num(entry.n),
    nulls: num(entry.nulls),
    trueCount: num(entry.trueCount),
    trueShare: num(entry.trueShare),
    mean: num(entry.mean),
    counts: "counts" in entry ? Object.fromEntries(Object.entries(counts).flatMap(([label, value]) => num(value) != null ? [[label, value as number] as [string, number]] : [])) : null,
  };
};

export const parseCodingReport = (loaded: LoadedJSON): CodingReport => {
  const raw = loaded.raw;
  return {
    present: loaded.present && !loaded.error,
    error: loaded.error,
    mode: str(raw.mode),
    experiment: str(raw.experiment),
    experimentHash: str(raw.experimentHash),
    sourceRun: str(raw.sourceRun) || str(raw.sourceRunDirectory),
    judges: list(raw.judges).flatMap((item) => typeof item === "string" ? [item] : typeof item === "object" && item ? [str(record(item).name)].filter(Boolean) : []),
    judgeModel: str(raw.judgeModel),
    judgeDetails: list(raw.judgeDetails).flatMap((item): CodingJudgeDetail[] => {
      const entry = record(item);
      const name = str(entry.name);
      if (!name) return [];
      return [{ name, kind: str(entry.kind), requestedModel: str(entry.requestedModel), actualModel: str(entry.actualModel), revision: str(entry.revision) }];
    }),
    rubricFile: str(raw.judgeRubricFile) || str(raw.rubricFile),
    rubricHash: str(raw.judgeRubricHash) || str(raw.rubricHash),
    fields: list(raw.fields).flatMap((item): CodingFieldSpec[] => {
      const entry = record(item);
      const name = str(entry.name);
      if (!name) return [];
      return [{ name, type: str(entry.type) || "string", optional: entry.optional === true, values: list(entry.values).filter((value): value is string => typeof value === "string") }];
    }),
    codings: num(raw.codings),
    conditions: Object.entries(record(raw.conditions)).map(([condition, value]): CodingConditionRow => {
      const entry = record(value);
      return {
        condition,
        codedResponses: num(entry.codedResponses),
        codings: num(entry.codings),
        meanWordCount: num(entry.meanWordCount),
        fields: Object.fromEntries(Object.entries(record(entry.fields)).map(([field, aggregate]) => [field, fieldAggregate(aggregate)])),
      };
    }).sort((left, right) => left.condition === "baseline" ? -1 : right.condition === "baseline" ? 1 : left.condition.localeCompare(right.condition)),
    fieldAgreement: list(raw.fieldAgreement).flatMap((item): CodingAgreementRow[] => {
      const entry = record(item);
      const field = str(entry.field);
      if (!field) return [];
      return [{ field, judgeA: str(entry.judgeA), judgeB: str(entry.judgeB), n: num(entry.n), percentAgreement: num(entry.percentAgreement), kappa: num(entry.kappa), meanAbsoluteDifference: num(entry.meanAbsoluteDifference) }];
    }),
    evaluationSource: str(raw.evaluationSource),
    epochUnverified: raw.epochUnverified === true,
    measurementDrift: str(raw.measurementDrift),
    exclusions: Object.keys(record(raw.exclusions)).length ? record(raw.exclusions) : null,
  };
};

/// Field names to show when a report is absent or carried no `fields`
/// block: the union of the keys the rows actually coded, in first-seen
/// order. Types are unknown in that case — the reader renders raw values.
export const fieldsFromRows = (rows: CodingRow[]): CodingFieldSpec[] => {
  const names: string[] = [];
  for (const row of rows) for (const name of Object.keys(row.codes)) if (!names.includes(name)) names.push(name);
  return names.map((name) => ({ name, type: "", optional: false, values: [] }));
};

// ---------------------------------------------------------------------------
// Viewer-derived selections (badged where rendered).
// ---------------------------------------------------------------------------

export type CodingDisagreement = {
  key: string;
  field: string;
  condition: string;
  promptID: string;
  sampleIndex: number;
  codings: { judge: string; value: CodeValue; briefReason: string }[];
};

/// Cells where two judges coded the same field of the same response
/// differently. Numeric fields count as a disagreement on any difference —
/// the magnitude is what the report's mean absolute difference summarizes,
/// and the reader shows both values so the size is visible. A field a judge
/// left null disagrees with a coded value (that IS a coding difference),
/// but a cell only one judge coded is never a disagreement.
export const codingDisagreements = (rows: CodingRow[], fields: CodingFieldSpec[]): CodingDisagreement[] => {
  const cells = new Map<string, CodingRow[]>();
  for (const row of rows) {
    const key = cellKey(row.condition, row.promptID, row.sampleIndex);
    cells.set(key, [...(cells.get(key) ?? []), row]);
  }
  const found: CodingDisagreement[] = [];
  for (const [key, group] of cells) {
    const byJudge = new Map<string, CodingRow>();
    for (const row of group) byJudge.set(row.judge, row);
    if (byJudge.size < 2) continue;
    const judges = [...byJudge.values()];
    for (const field of fields) {
      const coded = judges.filter((row) => field.name in row.codes);
      if (coded.length < 2) continue;
      const values = new Set(coded.map((row) => formatCode(row.codes[field.name])));
      if (values.size < 2) continue;
      found.push({
        key: `${key}${field.name}`,
        field: field.name,
        condition: coded[0].condition,
        promptID: coded[0].promptID,
        sampleIndex: coded[0].sampleIndex,
        codings: coded.map((row) => ({ judge: row.judge, value: row.codes[field.name], briefReason: row.briefReason })),
      });
    }
  }
  return found.sort((left, right) => left.field.localeCompare(right.field) || left.condition.localeCompare(right.condition) || left.promptID.localeCompare(right.promptID) || left.sampleIndex - right.sampleIndex);
};

export type WordCountBin = { from: number; to: number; count: number };
export type WordCountProfile = { condition: string; responses: number; mean: number | null; min: number; max: number; bins: WordCountBin[] };

/// Word-count distribution over DISTINCT responses (coding rows repeat one
/// response once per judge, and a word count is a property of the response,
/// not of the coding). Ten equal bins spanning the whole run's range, so
/// every condition's histogram is on the same axis.
export const wordCountProfiles = (rows: CodingRow[]): { profiles: WordCountProfile[]; low: number; high: number; missing: number } => {
  const seen = new Map<string, { condition: string; wordCount: number }>();
  let missing = 0;
  for (const row of rows) {
    if (row.wordCount == null) { missing += 1; continue; }
    seen.set(cellKey(row.condition, row.promptID, row.sampleIndex), { condition: row.condition, wordCount: row.wordCount });
  }
  const values = [...seen.values()];
  if (!values.length) return { profiles: [], low: 0, high: 0, missing };
  const low = Math.min(...values.map((item) => item.wordCount));
  const high = Math.max(...values.map((item) => item.wordCount));
  const span = high - low || 1;
  const byCondition = new Map<string, number[]>();
  for (const item of values) byCondition.set(item.condition, [...(byCondition.get(item.condition) ?? []), item.wordCount]);
  const profiles = [...byCondition.entries()].map(([condition, counts]): WordCountProfile => {
    const bins: WordCountBin[] = Array.from({ length: 10 }, (_, index) => ({ from: Math.round(low + (span * index) / 10), to: Math.round(low + (span * (index + 1)) / 10), count: 0 }));
    for (const count of counts) bins[Math.min(9, Math.floor(((count - low) / span) * 10))].count += 1;
    return { condition, responses: counts.length, mean: counts.reduce((sum, value) => sum + value, 0) / counts.length, min: Math.min(...counts), max: Math.max(...counts), bins };
  });
  return { profiles: profiles.sort((left, right) => left.condition === "baseline" ? -1 : right.condition === "baseline" ? 1 : left.condition.localeCompare(right.condition)), low, high, missing };
};
