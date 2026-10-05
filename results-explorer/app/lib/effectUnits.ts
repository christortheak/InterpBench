// What one paired difference of a stored effect row is, settled from the row,
// the analysis's stamp, and the run's records — never assumed.
//
// Before SteerLab 0.9.7 the Mac engine paired every response of a
// multi-sample run with the baseline response to the same item and seed,
// counted responses in `n`, and stamped no unit. Since 0.9.7 it averages an
// item's samples within each condition and still stamps no unit on pooled
// rows, so an old response-level row and a new item-level row look the same
// in the file. The run's records tell them apart: an item-level row counts at
// most the items the records pair with the baseline.
//
// This is the Python reader's rule (`results_export._paired_items` and
// `resolve_units`, used by the export, its methods summary, and the results
// page), and the Mac app's (`EffectUnits.swift`). A recorded unit is used;
// otherwise a row counting no more pairs than the paired items is item-level,
// a row counting more is response-level, and a condition with no paired items
// is unknown. Tests/Fixtures/cross-engine/effect-units.json holds the Python
// reader's answers, and test/effectUnits.test.ts holds this copy to them.
// Nothing stored is changed: this decides how a stored count is NAMED.

import type { Effect, RunFile } from "./types";

/// Where a row's unit comes from, in the Python reader's words (the export's
/// `unit_of_analysis_source`).
export type UnitSource = "recorded" | "engine_default" | "inferred_from_records" | "not_established";

export type EffectUnit = {
  /// "item", "transcript", "sample", "response", or "unknown" — or a unit the
  /// row or analysis stamped, as stamped.
  unit: string;
  source: UnitSource;
  /// Distinct items the run's records answer under both the row's condition
  /// and the baseline. null when the records hold none for that condition,
  /// or were not read because a stamp already settled the unit.
  pairedItems: number | null;
};

/// Per condition, the items the run's records pair with the baseline.
export type PairedItems = {
  counts: Record<string, number>;
  /// Every record was read. The explorer always reads the whole file, so
  /// this is false only for a caller that hands in part of one.
  complete: boolean;
};

/// A row nothing has settled: it claims no unit.
export const UNRESOLVED: EffectUnit = { unit: "unknown", source: "not_established", pairedItems: null };

export const unitOf = (row: Pick<Effect, "effectUnit">): EffectUnit => row.effectUnit ?? UNRESOLVED;

export const isResponses = (row: Pick<Effect, "effectUnit">) => unitOf(row).unit === "response";

// --- the records ---------------------------------------------------------------

/// Python's reader accepts NaN and Infinity; JSON.parse does not. Outside
/// strings they become null, which changes nothing the count reads.
const withoutNonFinite = (line: string): string => {
  let out = "";
  let inString = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (inString) {
      out += character;
      if (character === "\\") { out += line[index + 1] ?? ""; index += 1; } else if (character === "\"") inString = false;
      continue;
    }
    if (character === "\"") { inString = true; out += character; continue; }
    const token = ["-Infinity", "Infinity", "NaN"].find((word) => line.startsWith(word, index));
    if (token) { out += "null"; index += token.length - 1; continue; }
    out += character;
  }
  return out;
};

const parseLine = (line: string): unknown => {
  try { return JSON.parse(line); } catch { /* the lenient reading below */ }
  try { return JSON.parse(withoutNonFinite(line)); } catch { return null; }
};

/// A record value as the Python reader keys it (`str(value)`). Only
/// distinctness matters, so a number keeps its JSON spelling.
const pythonText = (record: Record<string, unknown>, key: string): string => {
  if (!(key in record)) return "";
  const value = record[key];
  if (typeof value === "string") return value;
  if (value === null) return "None";
  if (typeof value === "boolean") return value ? "True" : "False";
  if (typeof value === "number") return String(value);
  return JSON.stringify(value);
};

/// Counts paired items one generations line at a time. A line counts as the
/// Python reader counts it (`results_export.paired_items`): a JSON object that
/// carries no `error` and is either a generated response (an `output`) or an
/// instrument readout (an `instrument`), since a deterministic choice study
/// records readouts alone. Its condition and item are the record's
/// `condition` and `promptID`, a missing one read as empty.
export class PairedItemCounter {
  private items = new Map<string, Set<string>>();

  add(raw: string) {
    const line = raw.trim();
    if (!line) return;
    const parsed = parseLine(line);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return;
    const record = parsed as Record<string, unknown>;
    if ("error" in record || !("instrument" in record || "output" in record)) return;
    const condition = pythonText(record, "condition");
    if (!this.items.has(condition)) this.items.set(condition, new Set());
    this.items.get(condition)!.add(pythonText(record, "promptID"));
  }

  counts(): Record<string, number> {
    const baseline = this.items.get("baseline") ?? new Set<string>();
    const counts: Record<string, number> = {};
    for (const [condition, ids] of this.items) {
      if (condition !== "baseline") counts[condition] = [...ids].filter((id) => baseline.has(id)).length;
    }
    return counts;
  }
}

export const pairedItemsFromText = (text: string): Record<string, number> => {
  const counter = new PairedItemCounter();
  for (const line of text.split("\n")) counter.add(line);
  return counter.counts();
};

/// Bytes per read of a generations file. The count reads the WHOLE file, as
/// the Python reader does, a window at a time, so a run larger than the
/// generations preview is still counted in full.
export const COUNT_WINDOW = 32 * 1024 * 1024;

/// The paired items of a generations file, or null when it cannot be read.
export const countPairedItems = async (file: RunFile, window = COUNT_WINDOW): Promise<Record<string, number> | null> => {
  try {
    const blob = await file.handle.getFile();
    const counter = new PairedItemCounter();
    // One streaming decoder across windows, so a character split between
    // two reads is decoded whole.
    const decoder = new TextDecoder();
    let pending = "";
    for (let offset = 0; offset < blob.size; offset += window) {
      const end = Math.min(blob.size, offset + window);
      pending += decoder.decode(await blob.slice(offset, end).arrayBuffer(), { stream: end < blob.size });
      const lines = pending.split("\n");
      pending = lines.pop() ?? "";
      for (const line of lines) counter.add(line);
    }
    counter.add(pending + decoder.decode());
    return counter.counts();
  } catch { return null; }
};

// --- the rule ------------------------------------------------------------------

/// The unit of one row. `n` is the row's stored count (null when the file
/// gave none); `pairedItems` is null when no records could be read.
///
/// With every record read this is exactly the Python reader's rule. With
/// only part of the records (`complete` false) a count is a lower bound, so
/// a row within it is still item-level, but a row beyond it cannot be
/// settled and is not established.
export const effectUnit = (
  condition: string, n: number | null, recordedUnit: string, stampedUnit: string, pairedItems: PairedItems | null,
): EffectUnit => {
  const items = pairedItems ? pairedItems.counts[condition] ?? null : null;
  const unit = recordedUnit || stampedUnit;
  if (unit) return { unit, source: "recorded", pairedItems: items };
  if (!pairedItems || !items) return { unit: "unknown", source: "not_established", pairedItems: items };
  if (n == null || n <= items) return { unit: "item", source: "engine_default", pairedItems: items };
  if (!pairedItems.complete) return { unit: "unknown", source: "not_established", pairedItems: items };
  return { unit: "response", source: "inferred_from_records", pairedItems: items };
};

/// Settle every row's unit, once, for every view. `records` is asked for
/// only when some row's unit is not already stamped, so a stamped analysis
/// never reads its run's records.
export const resolveUnits = async (
  rows: Effect[], stampedUnit: string, records: () => Promise<PairedItems | null>,
): Promise<{ rows: Effect[]; pairedItems: PairedItems | null }> => {
  const needed = !stampedUnit && rows.some((row) => !row.pairedUnit);
  const pairedItems = needed ? await records() : null;
  return {
    rows: rows.map((row) => ({ ...row, effectUnit: effectUnit(row.condition, row.n, row.pairedUnit, stampedUnit, pairedItems) })),
    pairedItems,
  };
};

// --- the words -----------------------------------------------------------------

/// Fewer independent pairs than this cannot carry an interval. Paired
/// responses are not independent of each other, so a response row counts
/// its items.
export const MINIMUM_PAIRS = 3;

/// What a row that paired responses is, and is not. The Python reader's
/// results page and the Mac app say the same (effect-units.json holds it).
export const RESPONSE_CAVEAT = "These are responses, not items: the analysis paired each response with the baseline response to the same item and seed, so its interval and test treat responses to the same item as independent and are not findings about items";

/// What a row whose unit nothing settles says about its pairs.
export const UNKNOWN_UNIT_NOTE = "The unit of these pairs is not established";

/// Why a row is read as paired responses (the Python reader's
/// `RESPONSE_UNIT_EXPLANATION`, word for word).
export const RESPONSE_UNIT_EXPLANATION = "These rows count more pairs than the run has paired items, so the analysis paired responses, not items: each response with the baseline response to the same item and seed. That is how the Mac engine paired a study with several samples per item before SteerLab 0.9.7, and it stamped no unit. Responses to the same item are not independent, so these intervals and tests are likely to understate the uncertainty, and they are not findings about items. The stored numbers are copied unchanged; analyzing the run again computes item-level rows.";

/// The nouns a count takes in each unit (the Python reader's `_UNITS`).
export const UNIT_NOUNS: Record<string, [string, string]> = {
  item: ["paired item", "paired items"],
  transcript: ["paired transcript", "paired transcripts"],
  sample: ["paired sample", "paired samples"],
  response: ["paired response", "paired responses"],
  unknown: ["pair", "pairs"],
};

const plural = (count: number, one: string, many: string) => `${count} ${count === 1 ? one : many}`;

/// The row's count in its own unit, as the Python reader's results page
/// states it: "24 paired items", "4 paired transcripts", "3 pairs", or "8
/// paired responses from 4 items". null when the file gave no count, or a
/// unit outside the shared nouns (the label then names it).
export const countPhrase = (row: Pick<Effect, "n" | "effectUnit">): string | null => {
  if (row.n == null || row.n < 1) return null;
  const unit = unitOf(row);
  const nouns = UNIT_NOUNS[unit.unit];
  if (!nouns) return null;
  const text = plural(row.n, ...nouns);
  return unit.unit === "response" && unit.pairedItems != null ? `${text} from ${plural(unit.pairedItems, "item", "items")}` : text;
};

/// Fewer than the minimum independent pairs: for paired responses, fewer
/// than the minimum items. A row that did not report its count is left
/// alone.
export const hasTooFewIndependentPairs = (row: Pick<Effect, "n" | "effectUnit">): boolean => {
  const unit = unitOf(row);
  const independent = unit.unit === "response" ? unit.pairedItems : row.n;
  return independent != null && independent >= 1 && independent < MINIMUM_PAIRS;
};

/// What the minimum counted: "items" for paired responses, "pairs" otherwise.
export const tooFewNoun = (row: Pick<Effect, "effectUnit">) => isResponses(row) ? "items" : "pairs";

/// The sentence that follows a row that is not about items, or "".
export const unitCaveat = (row: Pick<Effect, "effectUnit">): string => {
  const unit = unitOf(row).unit;
  if (unit === "response") return `${RESPONSE_CAVEAT}.`;
  if (unit === "unknown") return `${UNKNOWN_UNIT_NOTE}.`;
  return "";
};

/// The unit cell: the settled unit, marked when the analysis did not record
/// it itself (the Python reader's page marks its table the same way).
export const unitLabel = (row: Pick<Effect, "effectUnit">): string => {
  const unit = unitOf(row);
  if (unit.source === "engine_default") return `${unit.unit} (default)`;
  if (unit.source === "inferred_from_records") return `${unit.unit} (from the records)`;
  return unit.unit;
};

const joined = (items: string[]) =>
  items.length <= 1 ? items.join("") : items.length === 2 ? `${items[0]} and ${items[1]}` : `${items.slice(0, -1).join(", ")}, and ${items[items.length - 1]}`;

export type UnitExplanation = { source: UnitSource; units: string[]; said: string };

/// One explanation per way the rows' units are known, in the order they
/// first appear: the Python reader's methods summary (`unit_lines`).
/// `records` is what the units were settled against, which says why a unit
/// is not established.
export const unitExplanations = (rows: Pick<Effect, "effectUnit">[], records: PairedItems | null): UnitExplanation[] => {
  const groups: { source: UnitSource; units: string[] }[] = [];
  for (const row of rows) {
    const unit = unitOf(row);
    const group = groups.find((entry) => entry.source === unit.source);
    if (!group) groups.push({ source: unit.source, units: [unit.unit] });
    else if (!group.units.includes(unit.unit)) group.units.push(unit.unit);
  }
  return groups.map(({ source, units }) => ({
    source, units,
    said: source === "recorded" ? `${joined(units)}, as the analysis recorded.`
      : source === "engine_default" ? "the item, with an item's samples averaged within each condition. This is the engines' documented default; the analysis did not stamp the unit itself. The run's records are consistent with it: no such row counts more pairs than the items paired in the run."
      : source === "inferred_from_records" ? `the response, not the item. ${RESPONSE_UNIT_EXPLANATION}`
      : records === null ? "not established. The analysis did not stamp it, and the run's records are not available here."
      : !records.complete ? "not established. The analysis did not stamp it, and only the first part of the run's records was read here, which does not settle it."
      : "not established. The analysis did not stamp it, and the run's records have no items paired with the baseline for these conditions.",
  }));
};

/// The rows a unit-of-analysis summary describes: the pooled rows, as the
/// Python reader's summary does, or every row when there are none.
const summarized = <Row extends Pick<Effect, "effectUnit" | "stratifyBy">>(rows: Row[]): Row[] => {
  const pooled = rows.filter((row) => row.stratifyBy === "" || row.stratifyBy === "pooled");
  return pooled.length ? pooled : rows;
};

/// The unit of analysis of a run that stamps none, in sentences, for the
/// stamps card and the interval note: what the run's records settle.
export const unsettledUnitSentence = (rows: Pick<Effect, "effectUnit" | "stratifyBy">[], records: PairedItems | null): string => {
  const explained = unitExplanations(summarized(rows), records);
  if (!explained.length) return "Not stamped in this run.";
  return explained.map(({ units, said }) =>
    explained.length > 1 ? `For the rows marked ${joined(units)}: ${said}` : `${said.charAt(0).toUpperCase()}${said.slice(1)}`).join(" ");
};

/// The Effects page's one line on what n counts, from the pooled rows'
/// settled units.
export const effectsUnitSummary = (rows: Pick<Effect, "effectUnit" | "stratifyBy">[], stampedUnit: string): string => {
  if (stampedUnit === "transcript") return "Each transcript—not each turn—is the unit of analysis.";
  const units = summarized(rows).map((row) => unitOf(row).unit);
  if (units.includes("response")) {
    return `${units.every((unit) => unit === "response") ? "These rows" : "Some rows"} paired responses, not items: responses to the same item are not independent, so their intervals and tests are not findings about items. The unit of analysis below says how this was read from the run’s records.`;
  }
  if (units.includes("unknown")) return "The unit of analysis of some rows is not established. The unit of analysis below says why.";
  return "The item—not the generation—is the unit of analysis, except where a stratum says otherwise.";
};

/// The methods summary's lines: "Unit of analysis[ for the rows marked …]: …".
export const unitLines = (rows: Pick<Effect, "effectUnit">[], records: PairedItems | null): string[] => {
  const explained = unitExplanations(rows, records);
  return explained.map(({ units, said }) => `Unit of analysis${explained.length > 1 ? ` for the rows marked ${joined(units)}` : ""}: ${said}`);
};
