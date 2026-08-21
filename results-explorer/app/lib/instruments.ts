// Answer-token choice-instrument records: reading them, typing them, and
// building the joins the choice view stands on.
//
// ARTIFACT CONTRACT. The engines write one instrument record per
// (condition, scoped item) INSIDE generations.jsonl, interleaved with the
// sampled-text records; an instrument record is identified by its
// `instrument` marker (e.g. "answerTokenLogprob") and carries the stored
// readout: per-option logprobs, mean token logprobs, token counts/ids,
// choice probabilities, log-odds, the margin, the DECLARED `target`
// option, and the `selected` option.
//
// PROVENANCE DISCIPLINE (docs/RESULTS-EXPLORER-UPGRADE-PLAN.md). Everything
// parsed here is STORED and renders plain. Everything *computed* here is
// either DERIVED (the baseline join, Δ target log-odds, flips, counts) or
// HEURISTIC (counterbalance pairing by `-ab`/`-ba` id suffix, the
// saturation threshold) — the view badges each accordingly. This module
// never computes a CI, a p-value, or a correction; those are engine-only.

import { findFile } from "./discovery";
import type { RunFile, WorkspaceRun } from "./types";

/// Same bounded-slice discipline as loaders.loadGenerations: a run's
/// generations.jsonl can be arbitrarily large, so the viewer reads a
/// bounded prefix and says so rather than hanging.
export const INSTRUMENT_READ_LIMIT = 32 * 1024 * 1024;

/// HEURISTIC default (upgrade plan, Phase 3 open question 1): an item whose
/// baseline |target log-odds| is at or above this has essentially no
/// headroom left for an intervention to move it. Caller-adjustable.
export const DEFAULT_SATURATION_THRESHOLD = 10;

export const SATURATION_ASSUMPTION = "an item whose baseline |target log-odds| ≥ the threshold is at ceiling — the run declares no such threshold";
export const PAIRING_ASSUMPTION = "prompt ids ending -ab / -ba are the two orders of one counterbalanced item; the run declares no pairing";
export const ORDER_CONSISTENCY_ASSUMPTION = "a counterbalanced pair swaps the option letters, so consistent orders select OPPOSITE letters";

/// The loaders build a generation row's `output` from `selected` when a
/// record stored no text, so the reader has something to show. It is a
/// DISPLAY string only: whether a row is an instrument record is decided by
/// the record's own `instrument` marker (`Generation.isInstrument`), never
/// by this prefix.
export const GENERATION_INSTRUMENT_OUTPUT_PREFIX = "Selected option: ";

export type CounterbalanceOrder = "ab" | "ba";

export type InstrumentRecord = {
  /// Source line index in generations.jsonl (1-based) — provenance for the
  /// record detail, never an identifier the engine declares.
  line: number;
  instrument: string;
  condition: string;
  promptID: string;
  promptIndex: number | null;
  caseID: string;
  arm: string;
  prompt: string;
  options: string[];
  optionLogprobs: Record<string, number>;
  optionMeanTokenLogprobs: Record<string, number>;
  optionTokenCounts: Record<string, number>;
  optionTokenIDs: Record<string, number[]>;
  optionTokenLogprobs: Record<string, number[]>;
  choiceProbability: Record<string, number>;
  logOdds: Record<string, number>;
  margin: number | null;
  selected: string;
  target: string;
  optionLengthRatio: number | null;
  interventionState: InterventionState | null;
  modelID: string;
  modelRevision: string;
  temperature: number | null;
  topK: number | null;
  topP: number | null;
  doSample: boolean | null;
  experiment: string;
  experimentHash: string;
  systemPromptHash: string;
};

export type InterventionSlot = { concept: string; layer: number | null; alpha: number | null };

export type InterventionState = {
  slots: InterventionSlot[];
  variant: string;
  controlType: string;
  bandWidth: number | null;
  alphaInNormUnits: boolean | null;
  adapters: string[];
};

export type InstrumentLoad = {
  /// Whether generations.jsonl exists at all — an absent file and a file
  /// with no instrument records are different empty states.
  present: boolean;
  file: RunFile | null;
  records: InstrumentRecord[];
  /// Non-instrument (sampled-text) records seen in the same file.
  sampledRecords: number;
  /// Lines carrying an `instrument` marker that did not parse into a usable
  /// record — reported, never silently dropped.
  skipped: number;
  truncated: boolean;
};

const record = (value: unknown): Record<string, unknown> => value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
const text = (raw: Record<string, unknown>, key: string) => typeof raw[key] === "string" ? raw[key] as string : "";
const finite = (value: unknown) => typeof value === "number" && Number.isFinite(value) ? value : null;
const numberMap = (value: unknown): Record<string, number> => Object.fromEntries(Object.entries(record(value)).flatMap(([key, item]) => finite(item) == null ? [] : [[key, item as number]]));
const numberArrayMap = (value: unknown): Record<string, number[]> => Object.fromEntries(Object.entries(record(value)).flatMap(([key, item]) => Array.isArray(item) ? [[key, item.flatMap((entry) => finite(entry) == null ? [] : [entry as number])]] : []));
const stringArray = (value: unknown): string[] => Array.isArray(value) ? value.flatMap((item) => typeof item === "string" ? [item] : typeof item === "object" && item !== null ? [text(record(item), "name") || JSON.stringify(item)] : []) : [];

const parseInterventionState = (value: unknown): InterventionState | null => {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const raw = record(value);
  const slots = Array.isArray(raw.slots) ? raw.slots.flatMap((item): InterventionSlot[] => {
    const slot = record(item);
    const concept = text(slot, "concept");
    if (!concept) return [];
    return [{ concept, layer: finite(slot.layer), alpha: finite(slot.alpha) }];
  }) : [];
  return {
    slots,
    variant: text(raw, "variant"),
    controlType: text(raw, "controlType"),
    bandWidth: finite(raw.bandWidth),
    alphaInNormUnits: typeof raw.alphaInNormUnits === "boolean" ? raw.alphaInNormUnits : null,
    adapters: stringArray(raw.adapters),
  };
};

/// Parse one JSONL line's object into an instrument record. Returns null for
/// anything that is not a usable instrument record — a sampled-text record
/// (no `instrument` marker) and a malformed instrument row are both null;
/// the caller distinguishes them by the marker's presence.
export const parseInstrumentRecord = (value: unknown, line: number): InstrumentRecord | null => {
  const raw = record(value);
  const instrument = text(raw, "instrument");
  const condition = text(raw, "condition");
  const promptID = text(raw, "promptID");
  const selected = text(raw, "selected");
  const options = Array.isArray(raw.options) ? raw.options.flatMap((item) => typeof item === "string" ? [item] : []) : [];
  if (!instrument || !condition || !promptID || !selected || options.length === 0) return null;
  return {
    line,
    instrument,
    condition,
    promptID,
    promptIndex: finite(raw.promptIndex),
    caseID: text(raw, "caseID"),
    arm: text(raw, "arm"),
    prompt: text(raw, "prompt"),
    options,
    optionLogprobs: numberMap(raw.optionLogprobs),
    optionMeanTokenLogprobs: numberMap(raw.optionMeanTokenLogprobs),
    optionTokenCounts: numberMap(raw.optionTokenCounts),
    optionTokenIDs: numberArrayMap(raw.optionTokenIDs),
    optionTokenLogprobs: numberArrayMap(raw.optionTokenLogprobs),
    choiceProbability: numberMap(raw.choiceProbability),
    logOdds: numberMap(raw.logOdds),
    margin: finite(raw.margin),
    selected,
    target: text(raw, "target"),
    optionLengthRatio: finite(raw.optionLengthRatio),
    interventionState: parseInterventionState(raw.interventionState),
    modelID: text(raw, "modelID"),
    modelRevision: text(raw, "modelRevision"),
    temperature: finite(raw.temperature),
    topK: finite(raw.topK),
    topP: finite(raw.topP),
    doSample: typeof raw.doSample === "boolean" ? raw.doSample : null,
    experiment: text(raw, "experiment"),
    experimentHash: text(raw, "experimentHash"),
    systemPromptHash: text(raw, "systemPromptHash"),
  };
};

/// Read the instrument records out of a run's generations.jsonl. Same
/// bounded read as the generations loader: the final (possibly partial)
/// line of a truncated read is discarded rather than parsed.
export const loadInstrumentRecords = async (run: Pick<WorkspaceRun, "files">): Promise<InstrumentLoad> => {
  const runFile = findFile(run.files, "generations.jsonl");
  if (!runFile) return { present: false, file: null, records: [], sampledRecords: 0, skipped: 0, truncated: false };
  const file = await runFile.handle.getFile();
  const truncated = file.size > INSTRUMENT_READ_LIMIT;
  const body = await file.slice(0, INSTRUMENT_READ_LIMIT).text();
  const lines = body.split(/\r?\n/);
  if (truncated) lines.pop();
  const records: InstrumentRecord[] = [];
  let sampledRecords = 0;
  let skipped = 0;
  lines.forEach((line, index) => {
    if (!line.trim()) return;
    let value: unknown;
    try { value = JSON.parse(line); } catch { skipped += 1; return; }
    const marked = typeof record(value).instrument === "string" && record(value).instrument !== "";
    if (!marked) { sampledRecords += 1; return; }
    const parsed = parseInstrumentRecord(value, index + 1);
    if (parsed) records.push(parsed); else skipped += 1;
  });
  return { present: true, file: runFile, records, sampledRecords, skipped, truncated };
};

// ---------------------------------------------------------------------------
// Joins and derived quantities
// ---------------------------------------------------------------------------

/// HEURISTIC (PAIRING_ASSUMPTION): split a counterbalance order suffix off a
/// prompt id. Ids without the suffix are their own stem with a null order.
export const splitOrderSuffix = (promptID: string): { stem: string; order: CounterbalanceOrder | null } => {
  const match = /^(.+)-(ab|ba)$/i.exec(promptID);
  if (!match) return { stem: promptID, order: null };
  return { stem: match[1], order: match[2].toLowerCase() as CounterbalanceOrder };
};

/// The baseline condition a Δ is measured against: the condition literally
/// named "baseline", else one whose name contains it, else none (in which
/// case no Δ, flip, or saturation call is made — absence is reported, not
/// guessed around).
export const resolveBaselineCondition = (conditions: string[]): string =>
  conditions.find((name) => name.toLowerCase() === "baseline")
  ?? conditions.find((name) => name.toLowerCase().includes("baseline"))
  ?? "";

export type InstrumentCell = {
  condition: string;
  record: InstrumentRecord;
  /// STORED: logOdds[target] on this record.
  targetLogOdds: number | null;
  selected: string;
  isBaseline: boolean;
  /// DERIVED: this record's target log-odds − the same-item baseline
  /// record's target log-odds.
  delta: number | null;
  /// DERIVED: this record selected a different option than the same-item
  /// baseline record.
  flipped: boolean;
  baselineSelected: string;
  /// The condition record declares a different target than the baseline
  /// record — a Δ across different targets would be meaningless, so it is
  /// suppressed and surfaced instead.
  targetMismatch: boolean;
};

export type InstrumentItem = {
  promptID: string;
  stem: string;
  order: CounterbalanceOrder | null;
  caseID: string;
  arm: string;
  target: string;
  promptIndex: number | null;
  baselineTargetLogOdds: number | null;
  baselineSelected: string;
  /// HEURISTIC (SATURATION_ASSUMPTION).
  saturated: boolean;
  cells: InstrumentCell[];
  cellByCondition: Record<string, InstrumentCell>;
};

export type OrderConsistency = {
  condition: string;
  abSelected: string;
  baSelected: string;
  comparable: boolean;
  /// HEURISTIC (ORDER_CONSISTENCY_ASSUMPTION): opposite letters across the
  /// two orders = the same substantive option chosen twice.
  consistent: boolean;
};

export type CounterbalancePair = {
  stem: string;
  ab: InstrumentItem;
  ba: InstrumentItem;
  consistency: OrderConsistency[];
  /// Conditions whose two orders selected the SAME letter — the two orders
  /// disagree substantively, which reads as a possible order artifact.
  possibleOrderArtifact: string[];
};

export type InstrumentGroup = {
  key: string;
  stem: string;
  items: InstrumentItem[];
  pair: CounterbalancePair | null;
};

export type InstrumentFlip = {
  promptID: string;
  condition: string;
  from: string;
  to: string;
  delta: number | null;
  targetLogOdds: number | null;
  baselineTargetLogOdds: number | null;
  saturated: boolean;
};

export type InstrumentTable = {
  instrumentNames: string[];
  instrumentName: string;
  conditions: string[];
  baselineCondition: string;
  items: InstrumentItem[];
  groups: InstrumentGroup[];
  pairs: CounterbalancePair[];
  /// Items carrying an order suffix whose mirror order is absent — never
  /// paired with anything.
  unpairedOrderedItems: InstrumentItem[];
  flips: InstrumentFlip[];
  saturationThreshold: number;
  saturatedCount: number;
};

const orderRank = (order: CounterbalanceOrder | null) => order === "ab" ? 0 : order === "ba" ? 1 : 2;

/// Build the per-item × per-condition table: the baseline join, Δ target
/// log-odds, flips, counterbalance pairs, and saturation calls. Records
/// carrying different `instrument` markers are kept apart — the caller
/// selects one marker; by default the most common one is used.
export const buildInstrumentTable = (
  records: InstrumentRecord[],
  options: { saturationThreshold?: number; instrument?: string; baselineCondition?: string } = {},
): InstrumentTable => {
  const saturationThreshold = options.saturationThreshold ?? DEFAULT_SATURATION_THRESHOLD;
  const instrumentNames = [...new Set(records.map((item) => item.instrument))];
  const counts = new Map<string, number>();
  records.forEach((item) => counts.set(item.instrument, (counts.get(item.instrument) ?? 0) + 1));
  const instrumentName = options.instrument ?? [...counts.entries()].sort((left, right) => right[1] - left[1])[0]?.[0] ?? "";
  const scoped = records.filter((item) => item.instrument === instrumentName);

  const conditions = [...new Set(scoped.map((item) => item.condition))];
  const baselineCondition = options.baselineCondition ?? resolveBaselineCondition(conditions);
  // Baseline first, then by name: the column order must not depend on which
  // condition the engine happened to write first, and it must match the
  // stored rollup strip beside it.
  const orderedConditions = [...conditions].sort((left, right) => (left === baselineCondition ? -1 : 0) - (right === baselineCondition ? -1 : 0) || left.localeCompare(right));

  const byPrompt = new Map<string, InstrumentRecord[]>();
  scoped.forEach((item) => byPrompt.set(item.promptID, [...(byPrompt.get(item.promptID) ?? []), item]));

  const appearance = [...byPrompt.keys()];
  const items: InstrumentItem[] = appearance.map((promptID) => {
    const rows = byPrompt.get(promptID) ?? [];
    const baselineRow = rows.find((row) => row.condition === baselineCondition) ?? null;
    const baselineTargetLogOdds = baselineRow ? baselineRow.logOdds[baselineRow.target] ?? null : null;
    const baselineSelected = baselineRow?.selected ?? "";
    const anchor = baselineRow ?? rows[0];
    const { stem, order } = splitOrderSuffix(promptID);
    const cells = orderedConditions.flatMap((condition): InstrumentCell[] => {
      const row = rows.find((entry) => entry.condition === condition);
      if (!row) return [];
      const targetLogOdds = row.logOdds[row.target] ?? null;
      const isBaseline = condition === baselineCondition && baselineRow !== null;
      const targetMismatch = baselineRow !== null && !isBaseline && baselineRow.target !== row.target;
      return [{
        condition,
        record: row,
        targetLogOdds,
        selected: row.selected,
        isBaseline,
        delta: isBaseline || targetMismatch || targetLogOdds == null || baselineTargetLogOdds == null ? null : targetLogOdds - baselineTargetLogOdds,
        flipped: !isBaseline && baselineSelected !== "" && row.selected !== baselineSelected,
        baselineSelected,
        targetMismatch,
      }];
    });
    return {
      promptID,
      stem,
      order,
      caseID: anchor?.caseID ?? "",
      arm: anchor?.arm ?? "",
      target: anchor?.target ?? "",
      promptIndex: anchor?.promptIndex ?? null,
      baselineTargetLogOdds,
      baselineSelected,
      saturated: baselineTargetLogOdds != null && Math.abs(baselineTargetLogOdds) >= saturationThreshold,
      cells,
      cellByCondition: Object.fromEntries(cells.map((cell) => [cell.condition, cell])),
    };
  }).sort((left, right) => {
    const leftIndex = left.promptIndex ?? Number.MAX_SAFE_INTEGER;
    const rightIndex = right.promptIndex ?? Number.MAX_SAFE_INTEGER;
    return leftIndex - rightIndex || left.promptID.localeCompare(right.promptID);
  });

  const groupKeys: string[] = [];
  const byStem = new Map<string, InstrumentItem[]>();
  items.forEach((item) => {
    if (!byStem.has(item.stem)) { byStem.set(item.stem, []); groupKeys.push(item.stem); }
    byStem.get(item.stem)?.push(item);
  });

  const pairs: CounterbalancePair[] = [];
  const unpairedOrderedItems: InstrumentItem[] = [];
  const groups: InstrumentGroup[] = groupKeys.map((stem) => {
    const groupItems = [...(byStem.get(stem) ?? [])].sort((left, right) => orderRank(left.order) - orderRank(right.order));
    const ab = groupItems.find((item) => item.order === "ab");
    const ba = groupItems.find((item) => item.order === "ba");
    let pair: CounterbalancePair | null = null;
    if (ab && ba) {
      const consistency = orderedConditions.flatMap((condition): OrderConsistency[] => {
        const abCell = ab.cellByCondition[condition];
        const baCell = ba.cellByCondition[condition];
        if (!abCell || !baCell) return [];
        const comparable = abCell.selected !== "" && baCell.selected !== "";
        return [{ condition, abSelected: abCell.selected, baSelected: baCell.selected, comparable, consistent: comparable && abCell.selected !== baCell.selected }];
      });
      pair = { stem, ab, ba, consistency, possibleOrderArtifact: consistency.flatMap((entry) => entry.comparable && !entry.consistent ? [entry.condition] : []) };
      pairs.push(pair);
    } else {
      groupItems.forEach((item) => { if (item.order) unpairedOrderedItems.push(item); });
    }
    return { key: stem, stem, items: groupItems, pair };
  });

  const flips: InstrumentFlip[] = items.flatMap((item) => item.cells.flatMap((cell) => cell.flipped ? [{
    promptID: item.promptID,
    condition: cell.condition,
    from: cell.baselineSelected,
    to: cell.selected,
    delta: cell.delta,
    targetLogOdds: cell.targetLogOdds,
    baselineTargetLogOdds: item.baselineTargetLogOdds,
    saturated: item.saturated,
  }] : []));

  return {
    instrumentNames,
    instrumentName,
    conditions: orderedConditions,
    baselineCondition,
    items,
    groups,
    pairs,
    unpairedOrderedItems,
    flips,
    saturationThreshold,
    saturatedCount: items.filter((item) => item.saturated).length,
  };
};

// ---------------------------------------------------------------------------
// Stored per-condition rollups (report.json)
// ---------------------------------------------------------------------------

export type ConditionRollup = {
  condition: string;
  choiceRate: number | null;
  choiceReadouts: number | null;
  generations: number | null;
  agreementWithBaseline: { n: number | null; agreement: number | null } | null;
};

/// Read report.json's `conditions` block. Every value is STORED; a missing
/// key stays null rather than becoming a zero.
export const readConditionRollups = (report: Record<string, unknown>): ConditionRollup[] => {
  const raw = report.conditions;
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return [];
  return Object.entries(raw as Record<string, unknown>).map(([condition, value]) => {
    const block = record(value);
    const agreement = block.agreementWithBaseline && typeof block.agreementWithBaseline === "object" ? record(block.agreementWithBaseline) : null;
    return {
      condition,
      choiceRate: finite(block.choiceRate),
      choiceReadouts: finite(block.choiceReadouts),
      generations: finite(block.generations),
      agreementWithBaseline: agreement ? { n: finite(agreement.n), agreement: finite(agreement.agreement) } : null,
    };
  }).sort((left, right) => (left.condition === "baseline" ? -1 : 0) - (right.condition === "baseline" ? -1 : 0) || left.condition.localeCompare(right.condition));
};

/// Condition names in these studies share a long variant prefix
/// ("optimize-conscientiousness-2026-08-03-fear-agent"). Strip the longest
/// hyphen-aligned prefix shared by every non-baseline condition so column
/// headers stay readable; the full name is always available as a tooltip.
export const shortConditionLabels = (conditions: string[], baselineCondition: string): Record<string, string> => {
  const others = conditions.filter((name) => name !== baselineCondition);
  let prefix = "";
  if (others.length > 1) {
    const segments = others[0].split("-");
    for (let count = segments.length - 1; count > 0; count -= 1) {
      const candidate = `${segments.slice(0, count).join("-")}-`;
      if (others.every((name) => name.startsWith(candidate) && name.length > candidate.length)) { prefix = candidate; break; }
    }
  }
  return Object.fromEntries(conditions.map((name) => [name, name === baselineCondition || !prefix || !name.startsWith(prefix) ? name : name.slice(prefix.length)]));
};

// ---------------------------------------------------------------------------
// Choice view → generations reader handoff
// ---------------------------------------------------------------------------

export type GenerationSelectionRequest = { promptID: string; condition: string };

// A one-shot module-level hand-off: the choice view names a record and
// navigates; the generations reader consumes the request when it mounts.
// Deliberately not a store — nothing subscribes, nothing persists.
let pendingSelection: GenerationSelectionRequest | null = null;

export const requestGenerationRecord = (request: GenerationSelectionRequest) => { pendingSelection = request; };

export const takeGenerationRecord = (): GenerationSelectionRequest | null => {
  const request = pendingSelection;
  pendingSelection = null;
  return request;
};

/// The generations loader builds a row id as `${promptID} · S${sample}`.
export const promptIDFromGenerationID = (id: string) => id.split(" · ")[0];

/// Instrument records are unique per (condition, scoped item) by contract.
export const instrumentRecordFor = (records: InstrumentRecord[], promptID: string, condition: string) =>
  records.find((item) => item.promptID === promptID && item.condition === condition) ?? null;

/* ------------------------------------------------------------------ */
/* Engine-computed deltas (choice-deltas.json, written by `analyze`)    */
/* ------------------------------------------------------------------ */

/// Per-condition summary from the analyze sidecar — STORED engine output
/// (paired bootstrap CI included, which the viewer must never compute).
export type EngineChoiceDeltaSummary = {
  condition: string;
  n: number | null;
  mean: number | null;
  ciLower: number | null;
  ciUpper: number | null;
  replicates: number | null;
  seed: number | null;
  flipped: number | null;
  skippedNoBaseline: number | null;
  skippedNoTargetValue: number | null;
};

export type EngineChoiceDeltas = {
  /// The analyze run directory name the artifact came from.
  analyzeRun: string;
  summaries: EngineChoiceDeltaSummary[];
  records: number | null;
  skippedNoBaseline: number | null;
  skippedNoTargetValue: number | null;
};

/// Analyze runs that consumed `runName` and carry the choice-deltas
/// artifact, newest first (directory-name timestamps sort lexically).
export const findChoiceDeltaRuns = (
  runName: string, workspaceRuns: Array<Pick<WorkspaceRun, "name" | "sourceRun" | "artifacts">>,
): Array<Pick<WorkspaceRun, "name" | "sourceRun" | "artifacts">> =>
  workspaceRuns
    .filter((candidate) => (candidate.sourceRun ?? "") === runName
      && candidate.artifacts.some((path) => path.endsWith("choice-deltas.json")))
    .sort((left, right) => right.name.localeCompare(left.name));

const summaryNumber = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

/// Read the sidecar from an analyze run. `null` on absence or a shape this
/// viewer cannot vouch for — never a partial reading presented as whole.
export const loadEngineChoiceDeltas = async (
  analyzeRun: Pick<WorkspaceRun, "name" | "files">,
): Promise<EngineChoiceDeltas | null> => {
  const file = analyzeRun.files.find((entry) => entry.name === "choice-deltas.json");
  if (!file) return null;
  let root: Record<string, unknown>;
  try {
    const parsed: unknown = JSON.parse(await (await file.handle.getFile()).text());
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
    root = parsed as Record<string, unknown>;
  } catch { return null; }
  const conditions = root.conditions;
  if (!conditions || typeof conditions !== "object" || Array.isArray(conditions)) return null;
  const summaries = Object.entries(conditions as Record<string, unknown>).flatMap(
    ([condition, raw]): EngineChoiceDeltaSummary[] => {
      if (!raw || typeof raw !== "object" || Array.isArray(raw)) return [];
      const entry = raw as Record<string, unknown>;
      return [{
        condition,
        n: summaryNumber(entry.n),
        mean: summaryNumber(entry.deltaTargetLogOddsMean),
        ciLower: summaryNumber(entry.ciLower),
        ciUpper: summaryNumber(entry.ciUpper),
        replicates: summaryNumber(entry.replicates),
        seed: summaryNumber(entry.seed),
        flipped: summaryNumber(entry.flipped),
        skippedNoBaseline: summaryNumber(entry.skippedNoBaseline),
        skippedNoTargetValue: summaryNumber(entry.skippedNoTargetValue),
      }];
    });
  if (!summaries.length) return null;
  summaries.sort((left, right) => left.condition.localeCompare(right.condition));
  return {
    analyzeRun: analyzeRun.name,
    summaries,
    records: summaryNumber(root.records),
    skippedNoBaseline: summaryNumber(root.skippedNoBaseline),
    skippedNoTargetValue: summaryNumber(root.skippedNoTargetValue),
  };
};
