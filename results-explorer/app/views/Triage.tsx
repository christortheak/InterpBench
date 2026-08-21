"use client";

// Workspace triage — the morning view (docs/RESULTS-EXPLORER-UPGRADE-PLAN.md,
// Phase 1). It answers one question: *what happened since I last looked?*
//
// Everything here is STORED data: run-status.json, FAILED.md, report.json,
// recommendations.json, judge-report.json, coding-report.json, pipeline.json.
// The only viewer-computed things are counts of those rows and the
// "since last visit" marker, which is presentation state in localStorage and
// never a claim about a run. No number on this page is inferred from prose,
// and a run with nothing readable says so.

import { useEffect, useMemo, useRef, useState } from "react";
import { ExportButton } from "../components/ui";
import { findFile, runKindOf, runStatusOf, sortRunsByTimestamp } from "../lib/discovery";
import { csvFilename, type ExportColumn } from "../lib/export";
import { runKindLabel, type RunKind } from "../lib/runKind";
import { statusLabel, type StatusState } from "../lib/status";
import type { RunFile, View, WorkspaceRun } from "../lib/types";
import "./triage.css";

/// Presentation-only marker. Comparable with a run directory's timestamp
/// prefix because both are UTC `YYYYMMDDTHHMMSSmmm`.
const LAST_VISIT_KEY = "steerlab.resultsExplorer.triageLastVisit";
const nowKey = () => new Date().toISOString().replace(/[-:]/g, "").replace(".", "").replace("Z", "");

export const stateSlug = (state: StatusState) => state.replace(/\s+/g, "-");

/// How many runs get their extra summary artifact read. Triage must stay
/// cheap: the artifacts below are small, but a workspace can hold hundreds of
/// runs and the bridge reads are real I/O.
const HEADLINE_BUDGET = 24;

const ATTENTION: StatusState[] = ["failed", "cancelled", "inProgress", "partial"];
const needsAttention = (run: WorkspaceRun) => ATTENTION.includes(runStatusOf(run).state);

/// "2 failed · 1 partial" — which kinds of attention, not merely how much.
/// A group header saying "3 needing attention" hides whether the morning
/// starts with three crashes or three jobs still running.
const attentionBreakdown = (runs: WorkspaceRun[]) => {
  const order: StatusState[] = ["failed", "cancelled", "inProgress", "partial"];
  return order.flatMap((state) => {
    const count = runs.filter((run) => runStatusOf(run).state === state).length;
    return count ? [`${count} ${state === "inProgress" ? "in progress" : state}`] : [];
  });
};

/// The exported run list is what triage shows: run-status.json, FAILED.md,
/// and the run directory's own name. The one viewer-made column is the run
/// kind where the engine stamped none.
const runListColumns: ExportColumn<WorkspaceRun>[] = [
  { header: "run", kind: "stored", value: (run) => run.name },
  { header: "experiment", kind: "stored", value: (run) => run.experiment },
  { header: "dateLabel", kind: "stored", value: (run) => run.dateLabel },
  { header: "timestampKey", kind: "stored", value: (run) => run.timestampKey ?? "" },
  { header: "runTypeStamp", kind: "stored", value: (run) => run.runTypeStamp ?? "" },
  { header: "kind", kind: "derived", value: (run) => runKindLabel(runKindOf(run)) },
  { header: "kindSource", kind: "stored", value: (run) => run.kindSource ?? "" },
  { header: "status", kind: "stored", value: (run) => runStatusOf(run).state },
  { header: "statusReading", kind: "derived", value: (run) => statusLabel(runStatusOf(run)) },
  { header: "completionSource", kind: "stored", value: (run) => runStatusOf(run).completionSource ?? "" },
  { header: "stage", kind: "stored", value: (run) => runStatusOf(run).stage },
  { header: "itemsWritten", kind: "stored", value: (run) => runStatusOf(run).itemsWritten },
  { header: "invalidResponses", kind: "stored", value: (run) => runStatusOf(run).invalidResponses },
  { header: "pendingUnits", kind: "stored", value: (run) => runStatusOf(run).pendingUnits.join("; ") },
  { header: "error", kind: "stored", value: (run) => runStatusOf(run).error },
  { header: "sourceRun", kind: "stored", value: (run) => run.sourceRun ?? "" },
  { header: "model", kind: "stored", value: (run) => run.model },
  { header: "path", kind: "stored", value: (run) => run.path },
];

const record = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
const numberOf = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

const readJSONArtifact = async (run: WorkspaceRun, name: string): Promise<Record<string, unknown> | null> => {
  const file = findFile(run.files, name);
  if (!file) return null;
  try {
    const parsed: unknown = JSON.parse(await (await file.handle.getFile()).text());
    return record(parsed);
  } catch { return null; }
};

// --- kind-specific headlines ---------------------------------------------
// Each reads ONLY stored fields. When the artifact a headline needs is
// absent, the headline is absent too — never a placeholder number.

const studyRunHeadline = (run: WorkspaceRun): string => {
  const conditions = record(run.report.conditions);
  const names = Object.keys(conditions);
  if (!names.length) return "";
  const generations = names.reduce((sum, name) => sum + (numberOf(record(conditions[name]).generations) ?? 0), 0);
  const rates = names
    .map((name) => numberOf(record(conditions[name]).choiceRate))
    .filter((value): value is number => value !== null);
  const parts = [`${names.length} condition${names.length === 1 ? "" : "s"} × ${generations} generations`];
  if (rates.length) {
    const low = Math.min(...rates);
    const high = Math.max(...rates);
    parts.push(low === high ? `choice rate ${low.toFixed(3)}` : `choice rate ${low.toFixed(3)}–${high.toFixed(3)}`);
  }
  return parts.join(" · ");
};

const multiAgentHeadline = (run: WorkspaceRun): string => {
  const conditions = record(run.report.conditions);
  const seats = Object.keys(record(run.report.modelBySeat));
  const generations = Object.values(conditions)
    .reduce((sum: number, entry) => sum + (numberOf(record(entry).generations) ?? 0), 0);
  const replicates = numberOf(run.config.samplesPerItem);
  const parts: string[] = [];
  if (seats.length) parts.push(`${seats.length} seats`);
  if (replicates !== null) parts.push(`${replicates} replicates`);
  if (Object.keys(conditions).length) parts.push(`${Object.keys(conditions).length} conditions`);
  if (generations) parts.push(`${generations} turn records`);
  return parts.join(" · ");
};

const sweepHeadline = (recommendations: Record<string, unknown>): string => {
  const entries = Object.entries(recommendations);
  if (!entries.length) return "";
  // The engines write a bare STRING for a concept whose selection failed and
  // an object carrying `winningCell` for one that succeeded.
  const selected = entries.filter(([, value]) => Object.keys(record(value)).length > 0).length;
  const failed = entries.length - selected;
  return `${selected} concept${selected === 1 ? "" : "s"} selected · ${failed} with no qualifying cell`;
};

const pairedHeadline = (report: Record<string, unknown>): string => {
  const conditions = record(report.conditions);
  const tally = Object.values(conditions).reduce<{ variant: number; baseline: number; ties: number }>((totals, entry) => {
    const row = record(entry);
    return {
      variant: totals.variant + (numberOf(row.variantWins) ?? 0),
      baseline: totals.baseline + (numberOf(row.baselineWins) ?? 0),
      ties: totals.ties + (numberOf(row.ties) ?? 0),
    };
  }, { variant: 0, baseline: 0, ties: 0 });
  const pairs = numberOf(report.pairs);
  const judges = Array.isArray(report.judges) ? report.judges.length : 0;
  const parts: string[] = [];
  if (pairs !== null) parts.push(`${pairs} pairs`);
  if (judges) parts.push(`${judges} judges`);
  if (tally.variant || tally.baseline || tally.ties) {
    parts.push(`${tally.variant} variant / ${tally.baseline} baseline / ${tally.ties} ties`);
  }
  return parts.join(" · ");
};

const codingHeadline = (report: Record<string, unknown>): string => {
  const codings = numberOf(report.codings);
  const fields = Array.isArray(report.fields) ? report.fields.length : 0;
  const parts: string[] = [];
  if (codings !== null) parts.push(`${codings} codings`);
  if (fields) parts.push(`${fields} coded fields`);
  const conditions = Object.keys(record(report.conditions)).length;
  if (conditions) parts.push(`${conditions} conditions`);
  return parts.join(" · ");
};

/// Which extra artifact a kind's headline needs. Runs whose headline comes
/// straight from report.json (already in memory) read nothing.
const headlineArtifact = (kind: RunKind): string | null =>
  kind === "sweep" ? "recommendations.json"
    : kind === "evaluate-paired" ? "judge-report.json"
      : kind === "evaluate-coding" ? "coding-report.json"
        : null;

// --- chains ---------------------------------------------------------------

type PipelineStage = { stage: string; status: string; runName: string };

const pipelineStages = (run: WorkspaceRun): PipelineStage[] => {
  const ledger = record(run.pipeline);
  const declared = Array.isArray(ledger.stages) ? ledger.stages.filter((s): s is string => typeof s === "string") : [];
  const results = record(ledger.stageResults);
  const names = declared.length ? declared : Object.keys(results);
  return names.map((stage) => {
    const result = record(results[stage]);
    const directory = typeof result.runDirectory === "string" ? result.runDirectory : "";
    return {
      stage,
      status: typeof result.status === "string" ? result.status : "",
      runName: directory ? directory.split("/").filter(Boolean).pop() ?? "" : "",
    };
  });
};

const railDotClass = (status: string) =>
  status === "completed" ? "state-completed"
    : status === "failed" ? "state-failed"
      : status === "inProgress" ? "state-inProgress" : "";

// --- the view -------------------------------------------------------------

export function TriageView({ run, workspaceRuns, onActivateRun, onNavigate, onOpenFile }: {
  run: WorkspaceRun | null;
  workspaceRuns: WorkspaceRun[];
  onActivateRun: (run: WorkspaceRun) => void;
  onNavigate: (view: View) => void;
  onOpenFile: (file: RunFile) => void;
}) {
  const [headlines, setHeadlines] = useState<Record<string, string>>({});
  const [lastVisit, setLastVisit] = useState("");
  const visitMarked = useRef(false);

  // "Since last visit" is a reading aid, nothing more: on the FIRST visit the
  // marker is empty and nothing is flagged new, rather than every run in the
  // workspace lighting up.
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- reads the last-visit marker from localStorage (an external system) on mount; the cleanup below writes the new one
    try { setLastVisit(window.localStorage.getItem(LAST_VISIT_KEY) ?? ""); } catch { /* private mode */ }
    return () => {
      if (visitMarked.current) return;
      visitMarked.current = true;
      try { window.localStorage.setItem(LAST_VISIT_KEY, nowKey()); } catch { /* private mode */ }
    };
  }, []);

  const ordered = useMemo(() => sortRunsByTimestamp(workspaceRuns), [workspaceRuns]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      const wanted = ordered
        .filter((candidate) => headlineArtifact(runKindOf(candidate)))
        .slice(0, HEADLINE_BUDGET);
      const found: Record<string, string> = {};
      for (const candidate of wanted) {
        const kind = runKindOf(candidate);
        const name = headlineArtifact(kind);
        if (!name) continue;
        const report = await readJSONArtifact(candidate, name);
        if (!report) continue;
        const text = kind === "sweep" ? sweepHeadline(report)
          : kind === "evaluate-paired" ? pairedHeadline(report)
            : kind === "evaluate-coding" ? codingHeadline(report) : "";
        if (text) found[candidate.key] = text;
      }
      if (!cancelled) setHeadlines(found);
    };
    void load();
    return () => { cancelled = true; };
  }, [ordered]);

  const headlineFor = (candidate: WorkspaceRun): string => {
    const kind = runKindOf(candidate);
    if (kind === "study-run") return studyRunHeadline(candidate);
    if (kind === "multi-agent") return multiAgentHeadline(candidate);
    return headlines[candidate.key] ?? "";
  };

  const byName = useMemo(() => {
    const map = new Map<string, WorkspaceRun>();
    for (const candidate of ordered) map.set(candidate.name, candidate);
    return map;
  }, [ordered]);

  const groups = useMemo(() => {
    const collected = new Map<string, WorkspaceRun[]>();
    for (const candidate of ordered) {
      const list = collected.get(candidate.experiment) ?? [];
      list.push(candidate);
      collected.set(candidate.experiment, list);
    }
    return [...collected.entries()].map(([experiment, runs]) => {
      // A run whose source run is ALSO in this workspace is rendered under
      // it, not beside it — the chain is the unit a researcher reads.
      const isChild = (candidate: WorkspaceRun) => {
        const source = candidate.sourceRun ?? "";
        return Boolean(source) && byName.get(source)?.experiment === experiment;
      };
      const children = new Map<string, WorkspaceRun[]>();
      for (const candidate of runs) {
        if (!isChild(candidate)) continue;
        const source = candidate.sourceRun ?? "";
        children.set(source, [...(children.get(source) ?? []), candidate]);
      }
      const roots = runs.filter((candidate) => !isChild(candidate));
      // Failures, cancellations and unfinished work first; then newest.
      const sortedRoots = [
        ...roots.filter(needsAttention),
        ...roots.filter((candidate) => !needsAttention(candidate)),
      ];
      return { experiment, runs, roots: sortedRoots, children };
    });
  }, [ordered, byName]);

  const attention = ordered.filter(needsAttention);
  // How many "completed" readings rest on the summary-artifact rule rather
  // than a stamped run-status.json — the attribution is on every row, and
  // this is its total.
  const summaryAttributed = ordered.filter((candidate) => runStatusOf(candidate).completionSource === "summaryArtifact").length;
  const freshCount = lastVisit ? ordered.filter((candidate) => (candidate.timestampKey ?? "") > lastVisit).length : 0;

  const open = (candidate: WorkspaceRun) => { onActivateRun(candidate); onNavigate("overview"); };

  if (!workspaceRuns.length) {
    return (
      <div className="view-enter inner-view">
        <header className="page-title">
          <div>
            <span className="section-number">WORKSPACE TRIAGE</span>
            <h1>No runs discovered</h1>
            <p>Choose a workspace in the sidebar. Triage reads only what the runs directory actually contains.</p>
          </div>
        </header>
        <div className="card no-run-card"><span>⌂</span><h2>Nothing to triage</h2><p>This view lists every run directory in the workspace, grouped by experiment.</p></div>
      </div>
    );
  }

  const renderRun = (candidate: WorkspaceRun, group: { children: Map<string, WorkspaceRun[]> }, depth: number) => {
    const status = runStatusOf(candidate);
    const kind = runKindOf(candidate);
    const headline = headlineFor(candidate);
    const failedNote = findFile(candidate.files, "FAILED.md");
    const source = candidate.sourceRun ?? "";
    const sourceRun = source ? byName.get(source) ?? null : null;
    const stages = kind === "pipeline" ? pipelineStages(candidate) : [];
    const isNew = Boolean(lastVisit) && (candidate.timestampKey ?? "") > lastVisit;
    const children = depth < 3 ? group.children.get(candidate.name) ?? [] : [];
    return (
      <div key={candidate.key}>
        <article className={`triage-run ${depth ? "is-child" : ""} ${needsAttention(candidate) ? "needs-attention" : ""} ${run?.key === candidate.key ? "selected" : ""}`}>
          <div className="triage-run-head">
            {/* The dot is coloured by STATE, so a run whose completion was
                read from its summary artifact gets the same green as one
                that stamped a status file; the difference is in the label
                ("completed (summary artifact)") and the tooltip, not in a
                second shade that would read as a lesser kind of done. */}
            <span
              className={`run-status-dot state-${stateSlug(status.state)}`}
              aria-hidden="true"
              title={status.completionSource === "summaryArtifact"
                ? "completed — read from the stage's summary artifact; this run wrote no run-status.json"
                : status.completionSource === "statusFile" ? "completed — stamped in run-status.json" : status.state}
            />
            <span className="triage-state">{statusLabel(status)}</span>
            {status.completionSource === "summaryArtifact" && <span
              className="triage-attribution"
              title="No run-status.json. The engines write a stage's summary artifact only after the last unit finishes, so its presence is the completion record."
            >from summary artifact</span>}
            <button className="triage-open" onClick={() => open(candidate)}>{candidate.name}</button>
            <span className="triage-kind">
              {runKindLabel(kind)}
              {candidate.kindSource === "artifacts" ? " (inferred from files)" : ""}
            </span>
            {isNew && <span className="triage-new" title="New since your last visit" />}
            <span className="triage-when">{candidate.dateLabel}</span>
          </div>

          {headline
            ? <p className="triage-headline">{headline}</p>
            : status.state === "completed"
              ? <p className="triage-headline"><em>No summary artifact readable for this run kind.</em></p>
              : null}

          {(status.state === "failed" || status.state === "cancelled") && status.error && (
            <p className="triage-error" title={status.error}>{status.error}</p>
          )}

          {(status.state === "failed" || status.state === "cancelled" || status.state === "inProgress" || status.state === "partial") && (
            <div className="triage-retry">
              <span>
                {status.itemsWritten === null
                  ? "items written: not stamped"
                  : `${status.itemsWritten} ${status.itemLabel || "item"}${status.itemsWritten === 1 ? "" : "s"} written`}
              </span>
              {status.invalidResponses !== null && status.invalidResponses > 0 && <span>{status.invalidResponses} invalid responses</span>}
              {status.pendingUnits.length > 0 && <span>did not run: {status.pendingUnits.join(", ")}</span>}
              {failedNote && <button onClick={() => onOpenFile(failedNote)}>open FAILED.md</button>}
            </div>
          )}

          {sourceRun && (
            <p className="triage-chain">
              consumed <button onClick={() => open(sourceRun)}>{sourceRun.name}</button>
            </p>
          )}
          {source && !sourceRun && (
            <p className="triage-chain">consumed <code>{source}</code> — not present in this workspace</p>
          )}

          {stages.length > 0 && (
            <div className="triage-rail" aria-label="Pipeline stages">
              {stages.map((stage, index) => {
                const target = stage.runName ? byName.get(stage.runName) ?? null : null;
                return (
                  <span key={`${stage.stage}-${index}`} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                    {index > 0 && <i>→</i>}
                    {target
                      ? <button onClick={() => open(target)}>
                          <i className={`rail-dot ${railDotClass(stage.status)}`} />{stage.stage}
                        </button>
                      : <span className="rail-stage">
                          <i className={`rail-dot ${railDotClass(stage.status)}`} />{stage.stage}{stage.status ? ` · ${stage.status}` : " · not run"}
                        </span>}
                  </span>
                );
              })}
            </div>
          )}
        </article>
        {children.map((child) => renderRun(child, group, depth + 1))}
      </div>
    );
  };

  return (
    <div className="view-enter inner-view">
      <header className="page-title">
        <div>
          <span className="section-number">WORKSPACE TRIAGE</span>
          <h1>What happened since you last looked</h1>
          <p>Every run directory in this workspace, grouped by experiment and newest first. Failures, cancellations and unfinished work sort to the top of each group; a run that stamped no status says “not stamped” rather than passing as complete.</p>
        </div>
      </header>

      <div className="triage-summary">
        <span><strong>{ordered.length}</strong> runs</span>
        <span><strong>{groups.length}</strong> experiments</span>
        <span><strong>{attention.length}</strong> need attention{attention.length ? ` · ${attentionBreakdown(attention).join(" · ")}` : ""}</span>
        <span><strong>{summaryAttributed}</strong> completed via summary artifact</span>
        <span>{lastVisit ? <><strong>{freshCount}</strong> new since your last visit</> : "first visit — nothing marked new"}</span>
        <ExportButton filename={csvFilename("workspace", "runs")} columns={runListColumns} rows={ordered} label="Export run list" className="triage-export" />
      </div>

      {groups.map((group) => (
        <section className="triage-group" key={group.experiment}>
          <header>
            <h2>{group.experiment}</h2>
            {/* Failures sort to the top of the group, so the header names
                what is up there rather than only how many rows follow. */}
            <span>
              {group.runs.length} run{group.runs.length === 1 ? "" : "s"}
              {group.runs.filter(needsAttention).length
                ? ` · ${group.runs.filter(needsAttention).length} needing attention (${attentionBreakdown(group.runs.filter(needsAttention)).join(", ")})`
                : ""}
            </span>
          </header>
          {group.roots.map((candidate) => renderRun(candidate, group, 0))}
        </section>
      ))}
    </div>
  );
}
