// Capability-battery artifacts (upgrade plan Phase 5: "port remaining
// native-only sections … battery view (per-condition accuracy + item
// browser over battery.jsonl)").
//
// ARTIFACT CONTRACT. A study run scores the manifest's PINNED capability
// battery under every condition — baseline included, with that condition's
// full intervention applied — and streams one record per (condition, item)
// to `battery.jsonl`, NEVER into generations.jsonl: battery items are
// capability controls, not study outputs, and must not enter outcome
// analysis (Server `experiment/tasks.py::_run_capability_battery`, Swift
// `ExperimentTasks.BatteryGenerationRecord`). The per-condition rollup
// {accuracy, itemCount, batteryHash} is stamped into report.json under
// `conditions[<name>].capabilityBattery`.
//
// Dialects: the server writes the expected answer as `answer` and stamps
// `promptID` / `sampleIndex` / `batteryHash` on every record; Swift writes
// `expected` and omits those. Both are read here; a key neither engine
// wrote stays absent rather than becoming a default.
//
// Everything parsed here is STORED. The only computed things this module
// exports are per-condition record COUNTS (`batteryCounts`), which the view
// badges derived — the accuracy it shows large is the engine's.

import { findFile } from "./discovery";
import type { RunFile, WorkspaceRun } from "./types";

/// Same bounded-read discipline as every other record reader.
export const BATTERY_READ_LIMIT = 32 * 1024 * 1024;

export type BatteryRecord = {
  /// 1-based source line in battery.jsonl — provenance for the item detail.
  line: number;
  condition: string;
  promptID: string;
  promptIndex: number | null;
  sampleIndex: number;
  prompt: string;
  /// The pinned answer key (`answer` on the server, `expected` on Swift).
  expected: string;
  output: string;
  /// null when the record stamped no grade — an ungraded item is not a
  /// wrong one.
  correct: boolean | null;
  batteryHash: string;
};

export type BatteryLoad = {
  /// Whether battery.jsonl exists at all: an absent file and a file with no
  /// readable rows are different empty states.
  present: boolean;
  file: RunFile | null;
  records: BatteryRecord[];
  /// Lines that did not parse into a usable record — reported, never
  /// silently dropped.
  skipped: number;
  truncated: boolean;
};

const record = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
const str = (raw: Record<string, unknown>, ...keys: string[]) => {
  for (const key of keys) if (typeof raw[key] === "string") return raw[key] as string;
  return "";
};
const finite = (value: unknown) => typeof value === "number" && Number.isFinite(value) ? value : null;

export const parseBatteryRecord = (value: unknown, line: number): BatteryRecord | null => {
  const raw = record(value);
  const condition = str(raw, "condition");
  // A battery row with no condition cannot be attributed to an arm, and a
  // row with no output was never scored — counting either would move an
  // accuracy strip that is meant to be the engine's.
  if (!condition || typeof raw.output !== "string") return null;
  const promptIndex = finite(raw.promptIndex);
  return {
    line,
    condition,
    promptID: str(raw, "promptID") || (promptIndex === null ? "" : `battery-${promptIndex}`),
    promptIndex,
    sampleIndex: finite(raw.sampleIndex) ?? 0,
    prompt: str(raw, "prompt"),
    expected: str(raw, "answer", "expected"),
    output: raw.output,
    correct: typeof raw.correct === "boolean" ? raw.correct : null,
    batteryHash: str(raw, "batteryHash"),
  };
};

export const loadBatteryRecords = async (run: Pick<WorkspaceRun, "files">): Promise<BatteryLoad> => {
  const runFile = findFile(run.files, "battery.jsonl");
  if (!runFile) return { present: false, file: null, records: [], skipped: 0, truncated: false };
  const file = await runFile.handle.getFile();
  const truncated = file.size > BATTERY_READ_LIMIT;
  const lines = (await file.slice(0, BATTERY_READ_LIMIT).text()).split(/\r?\n/);
  // A bounded read cuts the last record mid-line; drop it rather than parse
  // half a record.
  if (truncated) lines.pop();
  const records: BatteryRecord[] = [];
  let skipped = 0;
  lines.forEach((line, index) => {
    if (!line.trim()) return;
    let value: unknown;
    try { value = JSON.parse(line); } catch { skipped += 1; return; }
    const parsed = parseBatteryRecord(value, index + 1);
    if (parsed) records.push(parsed); else skipped += 1;
  });
  return { present: true, file: runFile, records, skipped, truncated };
};

// ---------------------------------------------------------------------------
// The stored per-condition rollup (report.json)
// ---------------------------------------------------------------------------

export type BatterySummary = {
  condition: string;
  /// STORED: `conditions[<name>].capabilityBattery.accuracy`. null when the
  /// condition carries no battery block — not zero.
  accuracy: number | null;
  itemCount: number | null;
  batteryHash: string;
};

export const readBatterySummaries = (report: Record<string, unknown>): BatterySummary[] => {
  const conditions = record(report.conditions);
  return Object.entries(conditions).flatMap(([condition, value]): BatterySummary[] => {
    const block = record(record(value).capabilityBattery);
    if (!Object.keys(block).length) return [];
    return [{
      condition,
      accuracy: finite(block.accuracy),
      itemCount: finite(block.itemCount),
      batteryHash: typeof block.batteryHash === "string" ? block.batteryHash : "",
    }];
  }).sort((left, right) =>
    (left.condition === "baseline" ? -1 : 0) - (right.condition === "baseline" ? -1 : 0)
    || left.condition.localeCompare(right.condition));
};

// ---------------------------------------------------------------------------
// DERIVED: counts over the loaded records (every caller badges these)
// ---------------------------------------------------------------------------

export type BatteryConditionCount = {
  condition: string;
  records: number;
  correct: number;
  incorrect: number;
  /// Records carrying no `correct` field: neither right nor wrong, counted
  /// apart so they cannot depress an accuracy read.
  ungraded: number;
};

export const batteryCounts = (records: BatteryRecord[]): BatteryConditionCount[] => {
  const counts = new Map<string, BatteryConditionCount>();
  for (const item of records) {
    const row = counts.get(item.condition)
      ?? { condition: item.condition, records: 0, correct: 0, incorrect: 0, ungraded: 0 };
    row.records += 1;
    if (item.correct === true) row.correct += 1;
    else if (item.correct === false) row.incorrect += 1;
    else row.ungraded += 1;
    counts.set(item.condition, row);
  }
  return [...counts.values()].sort((left, right) =>
    (left.condition === "baseline" ? -1 : 0) - (right.condition === "baseline" ? -1 : 0)
    || left.condition.localeCompare(right.condition));
};
