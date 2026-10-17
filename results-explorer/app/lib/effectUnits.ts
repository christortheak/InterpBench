// What one paired difference of a stored effect row is, settled from the row,
// the analysis's stamp, the engine its config.json names, and the run's
// records — never assumed.
//
// Before SteerLab 0.9.7 the Mac engine paired every response of a
// multi-sample run with the baseline response to the same item and seed,
// counted responses in `n`, and stamped no unit. Since 0.9.7 it averages an
// item's samples within each condition, as the Python engine always has; both
// stamp the unit since 0.9.8. A row counting fewer pairs than the run's items
// does not show it counts items: declared exclusions, missing values, or a
// subgroup leave five responses of one item looking like five items.
//
// This is the Python reader's rule (`results_export.resolve_units`, used by
// the export, its methods summary, and the results page), and the Mac app's
// (`EffectUnits.swift`). The first that applies settles a row: a recorded
// unit; item when the records answer each item only once under the
// row's condition and the baseline (the units coincide); for a pooled row, how
// the engine named by config.json paired (`enginePairing`); response when the
// row counts more pairs than the paired items; otherwise unknown.
// Tests/Fixtures/cross-engine/effect-units.json holds the Python reader's
// answers, and test/effectUnits.test.ts holds this copy to them. Nothing
// stored is changed: this decides how a stored count is NAMED.

import type { Effect, RunFile } from "./types";

/// Where a row's unit comes from, in the Python reader's words (the export's
/// `unit_of_analysis_source`).
export type UnitSource = "recorded" | "one_sample_per_item" | "analysis_engine" | "inferred_from_records" | "not_established";

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

/// The most generated responses, and the most instrument readouts, that any
/// one item has under one condition.
export type SamplesPerItem = { responses: number; readouts: number };

/// Per condition, the items the run's records pair with the baseline.
export type PairedItems = {
  counts: Record<string, number>;
  /// Every record was read. The explorer always reads the whole file, so
  /// this is false only for a caller that hands in part of one.
  complete: boolean;
  /// Per condition (the baseline too), the most responses and readouts any
  /// one item has: the Python reader's `samples_per_item`. Absent when the
  /// caller counted none.
  samplesPerItem?: Record<string, SamplesPerItem>;
};

/// How the engine that made an effect table paired its pooled rows, as the
/// directory's config.json names that engine (the Python reader's
/// `engine_pairing`): "items" (an item's samples averaged within each
/// condition: the Python engine always, the Mac engine since 0.9.7) or
/// "responses" (the Mac engine before 0.9.7).
export type EnginePairing = "items" | "responses";

/// The first Mac engine release that averages an item's samples.
export const MAC_ITEM_PAIRING_VERSION: [number, number, number] = [0, 9, 7];

/// The one outcome the Mac engine before 0.9.7 read from instrument readouts,
/// one per item and condition, and so paired by item even then.
const MAC_READOUT_OUTCOMES = new Set(["ordinalPosition"]);

const compareVersions = (left: number[], right: number[]) => {
  for (let index = 0; index < 3; index += 1) if (left[index] !== right[index]) return left[index] - right[index];
  return 0;
};

/// How the engine named by a directory's config.json paired, or null when
/// the file does not establish it: a missing or unreadable stamp, another
/// engine, or a pre-release of 0.9.7 itself (either side of the change).
/// One known limit, as in the Python reader: a development build made in the
/// hours between the item-pairing change and the version reaching 0.9.7
/// stamped 0.9.6 and already paired items; it reads as responses.
export const enginePairing = (config: Record<string, unknown> | null | undefined): EnginePairing | null => {
  if (!config || typeof config !== "object" || Array.isArray(config)) return null;
  if (config.substrate === "python-hf-transformers") return "items";
  if (config.substrate !== "swift-mlx" || typeof config.appVersion !== "string") return null;
  const match = /^swift-app (\d+)\.(\d+)\.(\d+)(-[^+\s]+)?(\+\S*)?$/u.exec(config.appVersion.trim());
  if (!match) return null;
  const version = [Number(match[1]), Number(match[2]), Number(match[3])];
  const order = compareVersions(version, MAC_ITEM_PAIRING_VERSION);
  if (match[4] && order === 0) return null;
  return order >= 0 ? "items" : "responses";
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
  /// condition → item → responses and readouts. A record carrying an
  /// `instrument` is a readout, as the Python reader sorts it.
  private perItem = new Map<string, Map<string, SamplesPerItem>>();

  add(raw: string) {
    const line = raw.trim();
    if (!line) return;
    const parsed = parseLine(line);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return;
    const record = parsed as Record<string, unknown>;
    if ("error" in record || !("instrument" in record || "output" in record)) return;
    const condition = pythonText(record, "condition");
    const item = pythonText(record, "promptID");
    if (!this.items.has(condition)) this.items.set(condition, new Set());
    this.items.get(condition)!.add(item);
    if (!this.perItem.has(condition)) this.perItem.set(condition, new Map());
    const byItem = this.perItem.get(condition)!;
    const tally = byItem.get(item) ?? { responses: 0, readouts: 0 };
    if ("instrument" in record) tally.readouts += 1; else tally.responses += 1;
    byItem.set(item, tally);
  }

  counts(): Record<string, number> {
    const baseline = this.items.get("baseline") ?? new Set<string>();
    const counts: Record<string, number> = {};
    for (const [condition, ids] of this.items) {
      if (condition !== "baseline") counts[condition] = [...ids].filter((id) => baseline.has(id)).length;
    }
    return counts;
  }

  samplesPerItem(): Record<string, SamplesPerItem> {
    const most: Record<string, SamplesPerItem> = {};
    for (const [condition, byItem] of this.perItem) {
      let responses = 0, readouts = 0;
      for (const tally of byItem.values()) {
        responses = Math.max(responses, tally.responses);
        readouts = Math.max(readouts, tally.readouts);
      }
      most[condition] = { responses, readouts };
    }
    return most;
  }

  /// Everything counted, as the unit rule reads it.
  records(complete: boolean): PairedItems {
    return { counts: this.counts(), complete, samplesPerItem: this.samplesPerItem() };
  }
}

export const pairedItemsFromText = (text: string): Record<string, number> => pairedRecordsFromText(text, true).counts;

/// The paired items and samples per item of a generations text in memory.
export const pairedRecordsFromText = (text: string, complete: boolean): PairedItems => {
  const counter = new PairedItemCounter();
  for (const line of text.split("\n")) counter.add(line);
  return counter.records(complete);
};

/// Bytes per read of a generations file. The count reads the WHOLE file, as
/// the Python reader does, a window at a time, so a run larger than the
/// generations preview is still counted in full.
export const COUNT_WINDOW = 32 * 1024 * 1024;

/// The paired items and samples per item of a generations file, read whole,
/// or null when it cannot be read.
export const countPairedRecords = async (file: RunFile, window = COUNT_WINDOW): Promise<PairedItems | null> => {
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
    return counter.records(true);
  } catch { return null; }
};

/// The paired items of a generations file, or null when it cannot be read.
export const countPairedItems = async (file: RunFile, window = COUNT_WINDOW): Promise<Record<string, number> | null> =>
  (await countPairedRecords(file, window))?.counts ?? null;

// --- the rule ------------------------------------------------------------------

/// What else a row's unit is settled by: its subgroup (`stratifyBy`), its
/// outcome (`metric`), and how the engine named by the effect table's
/// config.json paired (`pairing`).
export type UnitContext = { stratifyBy?: string; metric?: string; pairing?: EnginePairing | null };

/// The unit of one row. `n` is the row's stored count (null when the file
/// gave none); `pairedItems` is null when no records could be read. The
/// first that applies:
///
/// 1. a unit the row or its analysis recorded;
/// 2. item, when the records hold at most one response and one readout per
///    item under the row's condition and the baseline, and the row counts no
///    more pairs than the paired items (the units coincide);
/// 3. for a pooled row, how the engine named by config.json paired: items →
///    item; the Mac engine before 0.9.7 → item for its ordinal readouts,
///    response when the records hold several responses per item under both
///    the row's condition and the baseline;
/// 4. response, when the row counts more pairs than the paired items;
/// 5. otherwise unknown: fewer pairs than items settles nothing.
///
/// With every record read this is exactly the Python reader's rule. With
/// only part of the records (`complete` false) the counts are lower bounds,
/// so rules 2 and 4 do not apply, and a response row names no item count.
export const effectUnit = (
  condition: string, n: number | null, recordedUnit: string, stampedUnit: string, pairedItems: PairedItems | null,
  context: UnitContext = {},
): EffectUnit => {
  const items = pairedItems ? pairedItems.counts[condition] ?? null : null;
  const unit = recordedUnit || stampedUnit;
  if (unit) return { unit, source: "recorded", pairedItems: items };
  const samples = pairedItems?.samplesPerItem ?? {};
  const here = samples[condition], baseline = samples.baseline;
  const complete = pairedItems?.complete === true;
  const single = (side: SamplesPerItem | undefined) => (side?.responses ?? 0) <= 1 && (side?.readouts ?? 0) <= 1;
  if (complete && items && (n == null || n <= items) && Object.keys(samples).length > 0 && single(here) && single(baseline)) {
    return { unit: "item", source: "one_sample_per_item", pairedItems: items };
  }
  const pooled = !context.stratifyBy || context.stratifyBy === "pooled";
  if (pooled && context.pairing === "items") return { unit: "item", source: "analysis_engine", pairedItems: items };
  if (pooled && context.pairing === "responses") {
    if (MAC_READOUT_OUTCOMES.has(context.metric ?? "")) return { unit: "item", source: "analysis_engine", pairedItems: items };
    if (items && (here?.responses ?? 0) > 1 && (baseline?.responses ?? 0) > 1) {
      return { unit: "response", source: "analysis_engine", pairedItems: complete ? items : null };
    }
  }
  if (complete && items && n != null && n > items) return { unit: "response", source: "inferred_from_records", pairedItems: items };
  return { unit: "unknown", source: "not_established", pairedItems: items };
};

/// Settle every row's unit, once, for every view. `records` is asked for
/// only when some row's unit is not already stamped, so a stamped analysis
/// never reads its run's records. `pairing` is how the engine named by the
/// effect table's config.json paired (`enginePairing`).
export const resolveUnits = async (
  rows: Effect[], stampedUnit: string, records: () => Promise<PairedItems | null>, pairing: EnginePairing | null = null,
): Promise<{ rows: Effect[]; pairedItems: PairedItems | null }> => {
  const needed = !stampedUnit && rows.some((row) => !row.pairedUnit);
  const pairedItems = needed ? await records() : null;
  return {
    rows: rows.map((row) => ({
      ...row,
      effectUnit: effectUnit(row.condition, row.n, row.pairedUnit, stampedUnit, pairedItems, { stratifyBy: row.stratifyBy, metric: row.endpoint, pairing }),
    })),
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

/// What a row whose unit nothing settles says about its pairs (the Python
/// reader's `UNKNOWN_UNIT_NOTE`).
export const UNKNOWN_UNIT_NOTE = "The analysis that made this row did not record whether its pairs count items or responses";

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
/// paired responses from at most 4 items". null when the file gave no count,
/// or a unit outside the shared nouns (the label then names it). The items a
/// response row draws on are the run's paired items at most (exclusions or
/// missing values can leave fewer): the Python reader's `response_items`.
export const countPhrase = (row: Pick<Effect, "n" | "effectUnit">): string | null => {
  if (row.n == null || row.n < 1) return null;
  const unit = unitOf(row);
  const nouns = UNIT_NOUNS[unit.unit];
  if (!nouns) return null;
  // The whole number of pairs, read toward zero, as the Python page and the
  // Mac app state a stored count.
  const text = plural(Math.trunc(row.n), ...nouns);
  if (unit.unit !== "response" || unit.pairedItems == null) return text;
  return `${text} ${unit.pairedItems === 1 ? "from 1 item" : `from at most ${unit.pairedItems} items`}`;
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
  if (unit.source === "one_sample_per_item") return `${unit.unit} (answered once)`;
  if (unit.source === "analysis_engine") return `${unit.unit} (from the engine record)`;
  if (unit.source === "inferred_from_records") return `${unit.unit} (from the records)`;
  return unit.unit;
};

const joined = (items: string[]) =>
  items.length <= 1 ? items.join("") : items.length === 2 ? `${items[0]} and ${items[1]}` : `${items.slice(0, -1).join(", ")}, and ${items[items.length - 1]}`;

export type UnitExplanation = { source: UnitSource; units: string[]; said: string };

/// What the not-established line adds when this reading could not check the
/// run's records: none were available, or only part of them was handed in.
/// The Python reader's `UNIT_RECORDS_UNAVAILABLE` and `UNIT_RECORDS_PARTIAL`.
export const UNIT_RECORDS_UNAVAILABLE = "The run's records are not available here, so they could not be checked.";
export const UNIT_RECORDS_PARTIAL = "Only the first part of the run's records was read here; reading the whole file may settle it.";

/// What a unit the analysis did not record is, and how it is known: the
/// Python reader's `UNSTAMPED_UNIT_LINES`, word for word (pinned in
/// effect-units.json). Keyed by source, and for a unit the engine settles, by
/// unit too.
export const UNSTAMPED_UNIT_LINES: Record<string, string> = {
  one_sample_per_item: "the item. The analysis did not record its unit, but the run's records answer each item only once in each condition compared, so here an item and a response are the same thing.",
  analysis_engine_item: "the item, with an item's samples averaged within each condition. The analysis did not record its unit, but its config.json names the engine that made it, and that engine paired these rows by item.",
  analysis_engine_response: "the response, not the item. The analysis did not record its unit, but its config.json names the Mac engine from before SteerLab 0.9.7, which paired each response with the baseline response to the same item and seed when a study had several responses per item, as this run does. Responses to the same item are not independent, so these intervals and tests are likely to understate the uncertainty, and they are not findings about items. The stored numbers are copied unchanged; analyzing the run again computes item-level rows.",
  inferred_from_records: `the response, not the item. ${RESPONSE_UNIT_EXPLANATION}`,
  not_established: "not established. The analysis that made these rows did not record whether their pairs count items or responses, and neither its config.json nor the run's records settle it: fewer pairs than items can mean items, or responses from fewer items. The numbers are shown as stored; analyzing the run again with this version records the unit.",
};

/// What a recorded unit means, in the Python reader's words
/// (`results_export.RECORDED_UNIT_LINES`; pinned in effect-units.json).
export const RECORDED_UNIT_LINES: Record<string, string> = {
  item: "the item, with an item's samples averaged within each condition, as the analysis recorded.",
  transcript: 'the transcript, as the analysis recorded: one play-through of a multi-agent conversation, with its turns reduced to their average difference before any test, because turns in the same conversation depend on each other.',
};

/// Whether every row a unit summary describes recorded its own unit.
export const unitsRecorded = (rows: Pick<Effect, "effectUnit" | "stratifyBy">[]): boolean =>
  rows.length > 0 && summarized(rows).every((row) => unitOf(row).source === "recorded");

/// One explanation per way the rows' units are known, in the order they
/// first appear: the Python reader's methods summary (`unit_lines`). A unit
/// the analysis's engine settles is grouped by unit too: one Mac analysis
/// from before 0.9.7 can hold item rows (its ordinal readouts) and response
/// rows. `records` is what the units were settled against: when this reading
/// could not check them all, the not-established line says so.
export const unitExplanations = (rows: Pick<Effect, "effectUnit">[], records: PairedItems | null): UnitExplanation[] => {
  const groups: { key: string; source: UnitSource; units: string[] }[] = [];
  for (const row of rows) {
    const unit = unitOf(row);
    const key = unit.source === "analysis_engine" ? `${unit.source}_${unit.unit}` : unit.source;
    const group = groups.find((entry) => entry.key === key);
    if (!group) groups.push({ key, source: unit.source, units: [unit.unit] });
    else if (!group.units.includes(unit.unit)) group.units.push(unit.unit);
  }
  return groups.map(({ key, source, units }) => ({
    source, units,
    said: source === "recorded" ? (units.length === 1 && RECORDED_UNIT_LINES[units[0]]) || `${joined(units)}, as the analysis recorded.`
      : key !== "not_established" && UNSTAMPED_UNIT_LINES[key] ? UNSTAMPED_UNIT_LINES[key]
      // The line speaks of the files; what this reading saw of the run's
      // records is added when it could not check them all.
      : `${UNSTAMPED_UNIT_LINES.not_established}${records === null ? ` ${UNIT_RECORDS_UNAVAILABLE}` : !records.complete ? ` ${UNIT_RECORDS_PARTIAL}` : ""}`,
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
  const unknown = units.includes("unknown")
    ? ` The analysis that made ${units.every((unit) => unit === "unknown") ? "these rows" : "some rows"} did not record whether their pairs count items or responses.` : "";
  if (units.includes("response")) {
    return `${units.every((unit) => unit === "response") ? "These rows" : "Some rows"} paired responses, not items: responses to the same item are not independent, so their intervals and tests are not findings about items.${unknown} The unit of analysis below says how this was read.`;
  }
  if (unknown) return `${unknown.trim()} The numbers are shown as stored; the unit of analysis below says more.`;
  return "The item—not the generation—is the unit of analysis, except where a stratum says otherwise.";
};

/// The methods summary's lines: "Unit of analysis[ for the rows marked …]: …".
export const unitLines = (rows: Pick<Effect, "effectUnit">[], records: PairedItems | null): string[] => {
  const explained = unitExplanations(rows, records);
  return explained.map(({ units, said }) => `Unit of analysis${explained.length > 1 ? ` for the rows marked ${joined(units)}` : ""}: ${said}`);
};
