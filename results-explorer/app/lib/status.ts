// Status truth — the ONE reader of a run directory's honesty record
// (docs/RESULTS-EXPLORER-UPGRADE-PLAN.md, Phase 0). Everywhere the explorer
// shows a status it shows THIS, never the old `report.status === "complete"`
// literal.
//
// Engine contract (Swift `Sources/ExperimentKit/RunStatusFile.swift`, Python
// twin `Server/steerlab_server/experiment/run_status.py`):
//
// - `run-status.json` — schemaVersion, stage, status, startedAt, finishedAt,
//   evidenceComplete, itemLabel, itemsWritten, invalidResponses, experiment,
//   sourceRun, expectedUnits, completedUnits, pendingUnits, error, errorType.
//   `status` is one of `inProgress | completed | failed | checkpointed`.
// - `FAILED.md` — the human-readable account written beside a failure. Its
//   error line is `- **Error:** \`Type: message\``.
// - `cancelled.txt` — Swift's cooperative-cancellation note
//   (`ExperimentTasks.cancellationNoteFileName`). A cancelled task is ALSO
//   stamped `failed` with errorType `Cancelled` and gets no FAILED.md,
//   because a deliberate stop is not a failure.
//
// Three readings the engines are explicit about, and this module preserves:
//
// 1. ABSENT is not incomplete. Runs written before the contract existed
//    carry no status file; calling those partial would retroactively
//    invalidate real results. They read "not stamped".
// 2. UNREADABLE is partial, failing closed. A torn status file is evidence a
//    writer died mid-write; the safe reading of "I cannot tell" is "not
//    citable".
// 3. There is no "mostly done" that reads as done.

export type StatusState =
  | "completed"
  | "failed"
  | "cancelled"
  | "inProgress"
  | "partial"
  | "not stamped";

export type StatusInfo = {
  state: StatusState;
  /// The stage that wrote the record ("run", "evaluate", "pipeline", …), or
  /// "" when unstamped.
  stage: string;
  /// Best available failure text: the status file's message, else the
  /// FAILED.md error line. "" when the run did not fail.
  error: string;
  errorType: string;
  /// What this stage counts one of ("item", "judgment", "record", "cell").
  itemLabel: string;
  /// null = the run stamped no count, which is different from zero.
  itemsWritten: number | null;
  invalidResponses: number | null;
  pendingUnits: string[];
  completedUnits: string[];
  expectedUnits: string[];
  /// The run this one consumed (evaluate/analyze/rescore-style), from the
  /// status file. Discovery widens this with source-run.txt /
  /// judging-context.json.
  sourceRun: string;
  experiment: string;
  /// null when the field is absent — never assumed true.
  evidenceComplete: boolean | null;
  startedAt: number | null;
  finishedAt: number | null;
  /// Whether a run-status.json was present at all (an unreadable one counts
  /// as present: something wrote it and did not finish).
  stamped: boolean;
  /// Where a "completed" reading came from: the status file itself, or —
  /// for the many server stages that write no status file — the presence
  /// of the stage's summary artifact, which the engines write exclusively
  /// on success ("a partial panel is never summarized as a report").
  completionSource: "statusFile" | "summaryArtifact" | null;
};

/// Raw bytes as the discovery layer read them. `null` distinguishes "no such
/// file" from "an empty file".
export type StatusSources = {
  statusText: string | null;
  failedText: string | null;
  cancelledText: string | null;
  /// The run's file list (paths or bare names), used only for the
  /// no-status-file rows-without-a-summary reading below.
  artifacts: string[];
};

export const emptyStatusInfo = (): StatusInfo => ({
  state: "not stamped",
  stage: "",
  error: "",
  errorType: "",
  itemLabel: "",
  itemsWritten: null,
  invalidResponses: null,
  pendingUnits: [],
  completedUnits: [],
  expectedUnits: [],
  sourceRun: "",
  experiment: "",
  evidenceComplete: null,
  startedAt: null,
  finishedAt: null,
  stamped: false,
  completionSource: null,
});

/// The `- **Error:** \`RuntimeError: …\`` line of a FAILED.md, unwrapped from
/// its backticks. "" when the note carries no such line — a note we cannot
/// parse must not become an invented error string.
export const failureNoteError = (text: string | null): string => {
  if (!text) return "";
  for (const line of text.split(/\r?\n/)) {
    const match = /^\s*[-*]\s*\*\*Error:\*\*\s*(.*)$/.exec(line);
    if (!match) continue;
    return match[1].trim().replace(/^`+/, "").replace(/`+$/, "").trim();
  }
  return "";
};

const stringList = (value: unknown): string[] =>
  Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string") : [];

const finiteNumber = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

const text = (value: unknown): string => typeof value === "string" ? value.trim() : "";

const basenames = (artifacts: string[]) =>
  new Set(artifacts.map((path) => path.split("/").pop() ?? path));

/// Rows on disk with no completion artifact beside them. Applied ONLY when
/// the run carries no status file: the engines write the summary artifact
/// exclusively on success, so rows-without-summary is real evidence of an
/// interrupted stage — but a stamped run's own status always outranks it.
const rowsWithoutSummary = (names: Set<string>) =>
  (names.has("generations.jsonl") && !names.has("report.json"))
  || (names.has("judgments.jsonl") && !names.has("judge-report.json"))
  || (names.has("codings.jsonl") && !names.has("coding-report.json"));

/// The converse, same engine rule read forward (merge review 2026-08-05):
/// each stage's summary artifact is written exclusively after the last unit
/// finishes — so its PRESENCE, with no status file and no failure or
/// cancellation marker, is the engines' own completion record. Without this
/// reading, the many server stages that write no run-status.json (run,
/// sweep, validate, extract — 59 of the reference workspace's 70 runs) all
/// drown triage as "not stamped". The reading is attributed
/// (`completionSource: "summaryArtifact"`, surfaced in `statusLabel`) —
/// derived from a stored engine fact, never assumed from a directory name.
const SUMMARY_ARTIFACTS = [
  "report.json", "judge-report.json", "coding-report.json",
  "validation-report.json", "analysis.json", "recommendations.json",
];
const summaryArtifactPresent = (names: Set<string>) =>
  SUMMARY_ARTIFACTS.some((name) => names.has(name));

/// Pure core: everything the viewer knows about a run's completion, derived
/// from bytes already read. Tested directly; `loadStatusInfo` is the I/O
/// wrapper discovery uses.
export const deriveStatus = (sources: StatusSources): StatusInfo => {
  const info = emptyStatusInfo();
  const names = basenames(sources.artifacts);
  const cancelledNote = sources.cancelledText !== null || names.has("cancelled.txt");
  info.error = failureNoteError(sources.failedText);

  if (sources.statusText === null) {
    // No record at all. Absence is a fact, not success — and not failure.
    if (cancelledNote) {
      info.state = "cancelled";
      info.error = info.error || text(sources.cancelledText) || "cancelled by user";
      return info;
    }
    if (sources.failedText !== null) {
      // A FAILED.md normally travels with a status file; alone, it is
      // still a failure record, never "not stamped".
      info.state = "failed";
      return info;
    }
    if (rowsWithoutSummary(names)) { info.state = "partial"; return info; }
    if (summaryArtifactPresent(names)) {
      info.state = "completed";
      info.completionSource = "summaryArtifact";
      return info;
    }
    info.state = "not stamped";
    return info;
  }

  info.stamped = true;
  let parsed: Record<string, unknown> | null = null;
  try {
    const value: unknown = JSON.parse(sources.statusText);
    if (value && typeof value === "object" && !Array.isArray(value)) parsed = value as Record<string, unknown>;
  } catch { parsed = null; }

  if (!parsed) {
    // Fails CLOSED: a torn status file is evidence a writer died.
    info.state = cancelledNote ? "cancelled" : "partial";
    return info;
  }

  info.stage = text(parsed.stage);
  info.experiment = text(parsed.experiment);
  info.sourceRun = text(parsed.sourceRun);
  info.itemLabel = text(parsed.itemLabel);
  info.itemsWritten = finiteNumber(parsed.itemsWritten);
  info.invalidResponses = finiteNumber(parsed.invalidResponses);
  info.pendingUnits = stringList(parsed.pendingUnits);
  info.completedUnits = stringList(parsed.completedUnits);
  info.expectedUnits = stringList(parsed.expectedUnits);
  info.startedAt = finiteNumber(parsed.startedAt);
  info.finishedAt = finiteNumber(parsed.finishedAt);
  info.evidenceComplete = typeof parsed.evidenceComplete === "boolean" ? parsed.evidenceComplete : null;
  info.errorType = text(parsed.errorType);
  info.error = text(parsed.error) || info.error;

  const status = text(parsed.status);
  if (cancelledNote || info.errorType === "Cancelled") {
    info.state = "cancelled";
    info.error = info.error || text(sources.cancelledText) || "cancelled by user";
    return info;
  }
  if (status === "completed") {
    // `evidenceComplete: false` beside `completed` should not happen — the
    // writers set them together — so when it does, the conservative reading
    // wins rather than the cheerful one.
    info.state = info.evidenceComplete === false ? "partial" : "completed";
    if (info.state === "completed") info.completionSource = "statusFile";
    return info;
  }
  if (status === "failed") { info.state = "failed"; return info; }
  if (status === "inProgress") { info.state = "inProgress"; return info; }
  // `checkpointed` (a durably parked, resumable stage) and any status the
  // viewer does not recognize: real, incomplete, not a failure.
  info.state = "partial";
  return info;
};

export const statusTone = (state: StatusState): "good" | "warn" | "neutral" | "blue" =>
  state === "completed" ? "good"
    : state === "failed" ? "warn"
      : state === "cancelled" || state === "partial" ? "warn"
        : state === "inProgress" ? "blue"
          : "neutral";

/// A one-line human reading, used in the picker, topbar, and triage rows.
export const statusLabel = (info: StatusInfo): string => {
  switch (info.state) {
    case "completed":
      return info.completionSource === "summaryArtifact"
        ? "completed (summary artifact)" : "completed";
    case "failed": return "failed";
    case "cancelled": return "cancelled";
    case "inProgress": return "in progress";
    case "partial": return info.stamped ? "partial" : "partial (no summary artifact)";
    default: return "not stamped";
  }
};
