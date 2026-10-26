// Which records of its source run an analysis read (source-snapshot.json).
//
// A run parked at a checkpoint gains records when it is continued, in the
// same directory, and an analysis names its run by directory alone. So an
// analysis made while the run was paused (6 over two pairs) could sit under
// the completed run (3 over four pairs) with nothing to say so (review,
// October 23, 2026). Both engines now stamp the snapshot an analysis read:
// the run's status then, its record count, and the size and SHA-256 of the
// bytes parsed. This file holds the engines' comparison and sentences
// (source_snapshot.py, SourceSnapshot.swift), checked against the same
// hand-authored fixture, Tests/Fixtures/cross-engine/source-snapshot.json.
//
// The explorer compares SIZES rather than hashing the run's records: a run's
// generations.jsonl is only ever appended to, so a size that moved means
// records were added, and an unchanged size needs nothing read.

import { recordValue, runKindOf } from "./discovery";
import type { WorkspaceRun } from "./types";

export const STAMP_FILE = "source-snapshot.json";
const COMPLETED = "completed";
const PARKED = ["checkpointed", "inProgress"];

export type SnapshotStamp = {
  schemaVersion: number;
  sourceRun: string;
  status: string;
  records: number;
  generationsBytes: number;
  generationsSHA256: string;
};

export type SnapshotDescription = { records: number; generationsBytes: number; generationsSHA256: string };

export type SnapshotState = "matches" | "notRecorded" | "unfinishedSource" | "changed" | "notRecordedRunContinued";

export type SnapshotVerdict = {
  state: SnapshotState;
  analyzedRecords?: number;
  analyzedStatus?: string;
  runRecords?: number;
};

const integer = (value: unknown): value is number => typeof value === "number" && Number.isInteger(value);

/// The stamp, as strict as the engines read it: every key, of its type.
export const parseSnapshotStamp = (value: unknown): SnapshotStamp | null => {
  const raw = recordValue(value);
  if (!integer(raw.schemaVersion) || typeof raw.sourceRun !== "string" || typeof raw.status !== "string"
    || !integer(raw.records) || !integer(raw.generationsBytes) || typeof raw.generationsSHA256 !== "string") return null;
  return raw as unknown as SnapshotStamp;
};

/// Non-blank lines, universal newlines: the count a snapshot states.
export const countRecords = (text: string): number =>
  text.replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n").filter((line) => line.trim()).length;

/// The engines' comparison, pure, so the fixture's cases run here too.
export const snapshotVerdict = (stamp: SnapshotStamp | null, now: SnapshotDescription | null, continued: boolean): SnapshotVerdict => {
  if (!stamp) return { state: continued ? "notRecordedRunContinued" : "notRecorded" };
  const found: SnapshotVerdict = { state: "matches", analyzedRecords: stamp.records, analyzedStatus: stamp.status };
  if (now) found.runRecords = now.records;
  if (stamp.status !== COMPLETED) return { ...found, state: "unfinishedSource" };
  if (now && now.generationsSHA256 !== stamp.generationsSHA256) return { ...found, state: "changed" };
  return found;
};

const statusWords = (state: string): string => ({
  checkpointed: "paused at a checkpoint",
  inProgress: "still running",
  failed: "marked failed",
  unreadable: "in an unknown state (its status file could not be read)",
  noCompletionReport: "without a completion report",
} as Record<string, string>)[state] ?? `not complete (status: ${state})`;

/// What a reader is told, or null when there is nothing to say.
export const snapshotSentence = (found: SnapshotVerdict, run: string): string | null => {
  const analyzed = found.analyzedRecords ?? "None";
  if (found.state === "unfinishedSource") {
    const held = found.runRecords !== undefined && found.runRecords !== found.analyzedRecords ? `, and the run now holds ${found.runRecords}` : "";
    return `This analysis read run ${run} while it was ${statusWords(found.analyzedStatus ?? "None")}, with ${analyzed} records${held}. Re-analyze the run for effects over all its records.`;
  }
  if (found.state === "changed") {
    return `This analysis read ${analyzed} records of run ${run}, and the run's records have changed since (it now holds ${found.runRecords ?? "None"}). Re-analyze the run for effects over its current records.`;
  }
  if (found.state === "notRecordedRunContinued") {
    return `This analysis does not record which of run ${run}'s records it read, and the run was paused and continued, so it may predate the continuation. Re-analyze the run to be sure its effects cover every record.`;
  }
  return null;
};

/// An analysis's verdict against the run it read, as that run stands now.
/// `source` is that run when it is in the workspace (`findSourceRun`). Null
/// for any run that is not an analysis: an evaluation or a style rescore
/// names a source run too, but this stamp and its sentences are about what an
/// ANALYSIS read.
export type SourceSnapshotReading = { verdict: SnapshotVerdict; sentence: string | null };

const readJSON = async (run: WorkspaceRun, name: string): Promise<Record<string, unknown>> => {
  const file = run.files.find((entry) => entry.path === name);
  if (!file) return {};
  try { return recordValue(JSON.parse(await (await file.handle.getFile()).text())); } catch { return {}; }
};

export const loadSourceSnapshot = async (run: WorkspaceRun, source: WorkspaceRun | null): Promise<SourceSnapshotReading | null> => {
  if (runKindOf(run) !== "analyze") return null;
  const stampFile = run.files.find((entry) => entry.path === STAMP_FILE);
  const stamp = stampFile ? parseSnapshotStamp(await readJSON(run, STAMP_FILE)) : null;
  const runName = stamp?.sourceRun || source?.name || "";
  if (!stamp && !source) return null;
  let now: SnapshotDescription | null = null;
  let continued = false;
  if (source) {
    const generations = source.files.find((entry) => entry.path === "generations.jsonl");
    if (generations && stamp) {
      now = generations.size === stamp.generationsBytes
        ? { records: stamp.records, generationsBytes: stamp.generationsBytes, generationsSHA256: stamp.generationsSHA256 }
        : { records: countRecords(await (await generations.handle.getFile()).text()), generationsBytes: generations.size, generationsSHA256: `size:${generations.size}` };
    }
    const status = await readJSON(source, "run-status.json");
    const parked = recordValue(status.resumedFrom);
    continued = Object.keys(parked).length > 0
      || (typeof status.status === "string" && PARKED.includes(status.status) && source.files.some((entry) => entry.path === "report.json"));
  }
  const verdict = snapshotVerdict(stamp, now, continued);
  return { verdict, sentence: snapshotSentence(verdict, runName) };
};
