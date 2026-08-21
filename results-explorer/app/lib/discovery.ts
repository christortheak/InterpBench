// Run discovery: walking a workspace's runs/ tree over the structural
// directory-handle surface (File System Access in the browser, the native
// bridge when embedded) and reading the small metadata files that identify
// a run.

import { detectRunKind, type RunKind } from "./runKind";
import { deriveStatus, emptyStatusInfo, type StatusInfo } from "./status";
import type { LocalDirectoryHandle, RunFile, WorkspaceRun } from "./types";

export const optionalFile = async (directory: LocalDirectoryHandle, name: string) => {
  try { return await directory.getFileHandle(name); } catch { return null; }
};

export const readJSON = async (directory: LocalDirectoryHandle, name: string): Promise<Record<string, unknown>> => {
  const handle = await optionalFile(directory, name);
  if (!handle) return {};
  try { return JSON.parse(await (await handle.getFile()).text()) as Record<string, unknown>; } catch { return {}; }
};

export const textValue = (object: Record<string, unknown>, ...keys: string[]) => {
  for (const key of keys) {
    const value = object[key];
    if (typeof value === "string" && value.trim()) return value;
  }
  return "";
};

export const numberValue = (object: Record<string, unknown>, ...keys: string[]) => {
  for (const key of keys) {
    const value = object[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return 0;
};

export const collectFiles = async (directory: LocalDirectoryHandle, prefix = "", depth = 0): Promise<RunFile[]> => {
  const files: RunFile[] = [];
  for await (const entry of directory.values()) {
    if (entry.name.startsWith(".")) continue;
    const path = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.kind === "file") {
      const file = await entry.getFile();
      files.push({ name: entry.name, path, size: file.size, modified: file.lastModified, handle: entry });
    } else if (depth < 4) {
      files.push(...await collectFiles(entry, path, depth + 1));
    }
  }
  return files.sort((left, right) => left.path.localeCompare(right.path));
};

export const findFile = (files: RunFile[], name: string) => files.find((file) => file.name === name) ?? null;

export const readText = async (directory: LocalDirectoryHandle, name: string): Promise<string | null> => {
  const handle = await optionalFile(directory, name);
  if (!handle) return null;
  try { return await (await handle.getFile()).text(); } catch { return null; }
};

/// The sortable part of a run directory name: the engines' `YYYYMMDDTHHMMSS`
/// (+ optional milliseconds) prefix, e.g.
/// `20260805T004016927-exp-test-compare-2-2-evaluate`. Returned as the raw
/// digit string because it is already lexicographically ordered — and
/// because a Date parse would invent a timezone the name does not carry.
/// "" for a directory whose name has no such prefix.
export const runTimestampKey = (name: string) => /^(\d{8}T\d{6}\d*)/.exec(name)?.[1] ?? "";

/// Newest first, by the DIRECTORY NAME's timestamp — never by the formatted
/// date label. `localeCompare` on "Aug 4, 2026, 11:58 PM" orders months
/// alphabetically ("Apr" before "Aug" before "Dec"), which silently
/// mis-ordered every workspace spanning more than one month. Names carrying
/// no timestamp sort last, in stable name order, rather than to the top.
export const sortRunsByTimestamp = <T extends { name: string; timestampKey?: string }>(runs: T[]): T[] =>
  [...runs].sort((left, right) => {
    const leftKey = left.timestampKey ?? runTimestampKey(left.name);
    const rightKey = right.timestampKey ?? runTimestampKey(right.name);
    if (leftKey && rightKey && leftKey !== rightKey) return leftKey < rightKey ? 1 : -1;
    if (leftKey && !rightKey) return -1;
    if (!leftKey && rightKey) return 1;
    return left.name.localeCompare(right.name);
  });

/// Total readings of the appended run fields, so no consumer has to spell out
/// the fallback (a hand-built fixture may omit them; see types.ts).
export const runKindOf = (run: WorkspaceRun): RunKind => run.kind ?? "unknown";
export const runStatusOf = (run: WorkspaceRun): StatusInfo => run.statusInfo ?? emptyStatusInfo();

export const runDate = (name: string, report: Record<string, unknown>, config: Record<string, unknown>) => {
  const raw = textValue(report, "completedAt", "createdAt", "startedAt") || textValue(config, "createdAt", "startedAt") || name;
  const parsed = new Date(raw.replace(/_/g, ":"));
  if (!Number.isNaN(parsed.getTime())) return parsed.toLocaleString([], { month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit" });
  return name;
};

export const conditionTotal = (report: Record<string, unknown>) => {
  const raw = report.conditions;
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return 0;
  return Object.values(raw).reduce((sum: number, entry) => {
    if (!entry || typeof entry !== "object") return sum;
    const value = (entry as Record<string, unknown>).generations;
    return sum + (typeof value === "number" ? value : 0);
  }, 0);
};

export const discoverRuns = async (directory: LocalDirectoryHandle, path = "", depth = 0): Promise<WorkspaceRun[]> => {
  const rootNames: string[] = [];
  for await (const entry of directory.values()) if (entry.kind === "file") rootNames.push(entry.name);
  const names = rootNames.sort();
  const isRun = names.some((name) => [
    "report.json", "config.json", "generations.jsonl", "validation-report.json",
    "cosine-matrix.csv", "sweep.csv", "effect-sizes.csv", "analysis.json",
    "judge-report.json", "panel-effects.csv",
  ].includes(name));
  if (isRun) {
    const files = await collectFiles(directory);
    const ordinaryReport = await readJSON(directory, "report.json");
    const validationReport = await readJSON(directory, "validation-report.json");
    const analysisReport = await readJSON(directory, "analysis.json");
    const report = Object.keys(ordinaryReport).length ? ordinaryReport : Object.keys(validationReport).length ? validationReport : analysisReport;
    const config = await readJSON(directory, "config.json");
    const experiment = textValue(report, "experiment") || textValue(config, "experiment", "experimentName") || path.split("/").slice(-2, -1)[0] || "Unlabeled study";
    const model = textValue(config, "modelID", "modelId", "model") || textValue(report, "modelID", "modelId", "model") || "Model not stamped";
    const artifacts = files.map((file) => file.path);
    // The honesty record, read once here so every surface shows the same
    // status (upgrade plan Phase 0). Only files the directory actually
    // carries are opened.
    const has = (name: string) => names.includes(name);
    const statusInfo = deriveStatus({
      statusText: has("run-status.json") ? await readText(directory, "run-status.json") : null,
      failedText: has("FAILED.md") ? await readText(directory, "FAILED.md") : null,
      cancelledText: has("cancelled.txt") ? await readText(directory, "cancelled.txt") : null,
      artifacts,
    });
    const kindInfo = detectRunKind(config, report, artifacts);
    // Chain edge. Each engine stamps the consumed run in the place that suits
    // its stage; all four spellings mean the same thing, so the viewer reads
    // whichever is present rather than requiring one.
    const judgingContext = has("judging-context.json") ? await readJSON(directory, "judging-context.json") : {};
    const sourceRunFile = has("source-run.txt") ? (await readText(directory, "source-run.txt")) ?? "" : "";
    const sourceRun = statusInfo.sourceRun
      || sourceRunFile.trim()
      || textValue(judgingContext, "sourceRun")
      || textValue(report, "sourceRun");
    const pipeline = has("pipeline.json") ? await readJSON(directory, "pipeline.json")
      : has("pipeline-portable.json") ? await readJSON(directory, "pipeline-portable.json") : {};
    return [{
      key: path || directory.name,
      name: directory.name,
      path: path || directory.name,
      experiment,
      // NEVER default an unstamped run to "complete" (review 2026-08-03):
      // absence of a status is a fact, not success.
      status: textValue(report, "status") || "not stamped",
      model,
      dateLabel: runDate(directory.name, report, config),
      promptCount: numberValue(report, "promptCount"),
      conditionCount: numberValue(report, "conditionCount") || (report.conditions && typeof report.conditions === "object" ? Object.keys(report.conditions as object).length : 0),
      generationCount: conditionTotal(report),
      report,
      config,
      artifacts,
      files,
      handle: directory,
      effectRows: [],
      generationRows: [],
      generationFile: null,
      previewTruncated: false,
      skippedGenerationLines: 0,
      cosineMatrices: [],
      validationConcepts: [],
      validationReport,
      sweepRows: [],
      sweepRecommendations: [],
      panelEffects: [],
      kind: kindInfo.kind,
      kindSource: kindInfo.source,
      runTypeStamp: kindInfo.stampedRunType,
      statusInfo,
      sourceRun,
      pipeline,
      timestampKey: runTimestampKey(directory.name),
    }];
  }
  if (depth >= 4) return [];
  const found: WorkspaceRun[] = [];
  for await (const entry of directory.values()) {
    if (entry.kind !== "directory" || entry.name.startsWith(".")) continue;
    found.push(...await discoverRuns(entry, path ? `${path}/${entry.name}` : entry.name, depth + 1));
  }
  return found;
};

export const recordValue = (value: unknown): Record<string, unknown> => value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
