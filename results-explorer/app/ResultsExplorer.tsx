"use client";

import { Fragment, useEffect, useMemo, useState } from "react";
import {
  embeddedRecordParam,
  embeddedRunName,
  embeddedRunsDirectory,
  embeddedViewParam,
  embeddedWorkspaceName,
  isEmbedded,
} from "./embedded-workspace";
import { Badge, FilePreviewModal, Mark } from "./components/ui";
import { asView, setPendingRecord, updateDeepLink } from "./lib/deeplink";
import { demoPreviewEnabled } from "./lib/demo";
import { discoverRuns, runKindOf, runStatusOf, sortRunsByTimestamp } from "./lib/discovery";
import { hydrateRun } from "./lib/loaders";
import { runKindLabel, type RunKind } from "./lib/runKind";
import { statusLabel, statusTone } from "./lib/status";
import type {
  FilePreview,
  LocalDirectoryHandle,
  PickerWindow,
  RunFile,
  View,
  WorkspaceRun,
} from "./lib/types";
import { AnalysisOutputsView } from "./views/Analysis";
import { BatteryView } from "./views/Battery";
import { ChoiceInstrumentView } from "./views/ChoiceInstrument";
import { CodingView } from "./views/Coding";
import { ConceptEvidenceView } from "./views/Concepts";
import { EffectsView } from "./views/Effects";
import { GenerationsView } from "./views/Generations";
import { JudgedEvaluationView } from "./views/JudgedEvaluation";
import { OptimizationView } from "./views/Optimization";
import { Overview } from "./views/Overview";
import { PanelView } from "./views/Panels";
import { ProvenanceView } from "./views/Provenance";
import { TriageView, stateSlug } from "./views/Triage";

const nav: { id: View; label: string; eyebrow: string }[] = [
  { id: "triage", label: "Workspace triage", eyebrow: "00" },
  { id: "overview", label: "Study overview", eyebrow: "01" },
  { id: "concepts", label: "Concept evidence", eyebrow: "02" },
  { id: "optimization", label: "Optimization & sweeps", eyebrow: "03" },
  { id: "effects", label: "Effects & robustness", eyebrow: "04" },
  { id: "residuals", label: "Analysis outputs", eyebrow: "05" },
  { id: "battery", label: "Capability battery", eyebrow: "06" },
  { id: "choice", label: "Choice instrument", eyebrow: "07" },
  { id: "judged", label: "Judged evaluation", eyebrow: "08" },
  { id: "coding", label: "Response coding", eyebrow: "09" },
  { id: "panels", label: "Multi-agent panels", eyebrow: "10" },
  { id: "generations", label: "Generations", eyebrow: "11" },
  { id: "provenance", label: "Run files", eyebrow: "12" },
];

// Kind-aware navigation (upgrade plan Phase 0/5). A section that cannot
// possibly apply to the selected run is HIDDEN rather than shown empty — an
// empty "Judged evaluation" beside a sweep reads as a missing artifact when
// it is really a category error. Triage, Generations and Run files always
// show (every run has files; most have records), Study overview shows
// whenever a run is selected, and the toggle below restores everything.
const ALWAYS_VISIBLE: View[] = ["triage", "generations", "provenance"];

const SECTIONS_BY_KIND: Record<RunKind, View[]> = {
  // The capability battery is scored INSIDE a run (battery.jsonl beside
  // generations.jsonl), so it belongs with the study-run sections; the
  // analyze-output sections (alien residuals, promoted movers) belong with
  // the verb that writes them.
  "study-run": ["overview", "choice", "effects", "battery"],
  "multi-agent": ["overview", "panels", "battery"],
  sweep: ["overview", "optimization"],
  "evaluate-paired": ["overview", "judged"],
  "evaluate-coding": ["overview", "coding"],
  validate: ["overview", "concepts"],
  extract: ["overview", "concepts"],
  analyze: ["overview", "effects", "residuals"],
  "rescore-style": ["overview"],
  pipeline: ["overview"],
  // A run we could not identify hides nothing: the researcher, not the
  // viewer, decides what is worth opening.
  unknown: nav.map((item) => item.id),
};

export default function Home() {
  const [view, setView] = useState<View>("triage");
  const [showAllSections, setShowAllSections] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [workspaceName, setWorkspaceName] = useState("");
  const [workspaceRuns, setWorkspaceRuns] = useState<WorkspaceRun[]>([]);
  const [selectedRun, setSelectedRun] = useState<WorkspaceRun | null>(null);
  const [scanning, setScanning] = useState(false);
  const [loadingRun, setLoadingRun] = useState(false);
  const [folderError, setFolderError] = useState("");
  const [runPickerOpen, setRunPickerOpen] = useState(false);
  const [runSearch, setRunSearch] = useState("");
  const [filePreview, setFilePreview] = useState<FilePreview | null>(null);
  // Navigating updates the permalink so the address bar always names what is
  // on screen (embedded only — see lib/deeplink.ts). The record param is
  // cleared here and re-set by the view that owns the new section, so a
  // record key can never outlive the view that could interpret it.
  const go = (next: View) => {
    setView(next);
    setMenuOpen(false);
    updateDeepLink({ view: next, record: null });
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  const chooseWorkspace = async () => {
    const picker = (window as PickerWindow).showDirectoryPicker;
    if (!picker) {
      setFolderError("This browser does not support local folder selection. Open the app in a current Chromium-based browser.");
      return;
    }
    setFolderError("");
    setScanning(true);
    try {
      const workspace = await picker({ mode: "read" });
      const runsDirectory = workspace.name === "runs" ? workspace : await workspace.getDirectoryHandle("runs");
      // Newest first BY DIRECTORY TIMESTAMP. The previous sort compared the
      // formatted date label with localeCompare, which orders month names
      // alphabetically and silently mis-ordered any workspace spanning more
      // than one month.
      const discovered = sortRunsByTimestamp(await discoverRuns(runsDirectory));
      setWorkspaceName(workspace.name);
      setWorkspaceRuns(discovered);
      setSelectedRun(null);
      setRunPickerOpen(discovered.length > 0);
      if (!discovered.length) setFolderError("The runs folder was found, but it contains no readable run artifacts.");
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setFolderError("Choose a SteerLab workspace containing a runs folder, or choose the runs folder itself.");
    } finally { setScanning(false); }
  };
  // `landOn` is the section a deep link asked for; without one a newly
  // activated run opens on its overview, as it always has.
  const activateRun = async (run: WorkspaceRun, landOn?: View) => {
    setLoadingRun(true);
    try {
      const hydrated = await hydrateRun(run);
      setWorkspaceRuns((current) => current.map((candidate) => candidate.key === hydrated.key ? hydrated : candidate));
      setSelectedRun(hydrated);
      setRunPickerOpen(false);
      setRunSearch("");
      updateDeepLink({ run: hydrated.name });
      go(landOn ?? "overview");
    } catch {
      setFolderError(`The run “${run.name}” could not be read completely.`);
    } finally { setLoadingRun(false); }
  };
  // Embedded (native SteerLab app) mode: the WKWebView host serves the
  // workspace's runs/ over the page's own origin, so there is no directory
  // picker — the workspace loads itself, and a `?run=&view=&record=` deep
  // link activates that run, opens that section, and selects that record.
  // Every parser downstream of discovery is the SAME code the browser path
  // runs.
  const embedded = isEmbedded();
  const loadEmbeddedWorkspace = async () => {
    setFolderError("");
    setScanning(true);
    try {
      const runsDirectory = embeddedRunsDirectory() as unknown as LocalDirectoryHandle;
      // Newest first BY DIRECTORY TIMESTAMP. The previous sort compared the
      // formatted date label with localeCompare, which orders month names
      // alphabetically and silently mis-ordered any workspace spanning more
      // than one month.
      const discovered = sortRunsByTimestamp(await discoverRuns(runsDirectory));
      setWorkspaceName(embeddedWorkspaceName());
      setWorkspaceRuns(discovered);
      const wanted = embeddedRunName();
      const target = wanted ? discovered.find((run) => run.name === wanted) ?? null : null;
      if (target) {
        // A `view=` naming no known section is ignored rather than guessed
        // at: the link still lands on the run it named.
        const linkedView = asView(embeddedViewParam());
        const linkedRecord = embeddedRecordParam() ?? "";
        if (linkedView && linkedRecord) setPendingRecord(linkedView, linkedRecord);
        await activateRun(target, linkedView ?? undefined);
      } else {
        // No ?run= deep link: land on WORKSPACE TRIAGE, not on a modal over
        // an empty overview. Triage is the answer to "what happened since I
        // last looked", which is why the pane was opened.
        setSelectedRun(null);
        setRunPickerOpen(false);
        go("triage");
      }
      if (!discovered.length) setFolderError("The workspace's runs folder contains no readable run artifacts.");
    } catch {
      setFolderError("The embedded workspace bridge did not answer — close and reopen the explorer pane.");
    } finally { setScanning(false); }
  };
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- mount-only read of the embedded workspace bridge (an external system); its state lands from the async callback
    if (embedded) void loadEmbeddedWorkspace();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- mount-only load
  }, []);
  const openRunFile = async (file: RunFile) => {
    const shell: FilePreview = { file, text: "", truncated: false, loading: true, error: "" };
    setFilePreview(shell);
    try {
      const localFile = await file.handle.getFile();
      const limit = 1024 * 1024;
      const textExtensions = new Set(["json", "jsonl", "csv", "md", "txt", "log", "yaml", "yml"]);
      const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
      if (!textExtensions.has(extension)) {
        setFilePreview({ ...shell, loading: false, text: "", truncated: false });
        return;
      }
      let text = await localFile.slice(0, limit).text();
      if (extension === "json" && localFile.size <= limit) {
        try { text = JSON.stringify(JSON.parse(text), null, 2); } catch { /* show raw text */ }
      }
      setFilePreview({ ...shell, text, truncated: localFile.size > limit, loading: false });
    } catch { setFilePreview({ ...shell, loading: false, error: "This file could not be read." }); }
  };
  const visibleRuns = workspaceRuns.filter((run) => `${run.name} ${run.experiment} ${run.model} ${run.path}`.toLowerCase().includes(runSearch.toLowerCase()));

  // --- kind-aware nav ---------------------------------------------------
  const selectedKind = selectedRun ? runKindOf(selectedRun) : null;
  const allowedSections = selectedKind && !showAllSections
    ? new Set<View>([...ALWAYS_VISIBLE, ...SECTIONS_BY_KIND[selectedKind]])
    : null;
  const visibleNav = allowedSections ? nav.filter((item) => allowedSections.has(item.id)) : nav;
  const hiddenSectionCount = nav.length - visibleNav.length;
  // Selecting a run must never strand the reader on a section that just
  // became inapplicable. DERIVED rather than corrected in an effect: the
  // requested view is remembered, so turning the escape hatch on returns to
  // it instead of having been overwritten.
  const shownView: View = allowedSections && !allowedSections.has(view)
    ? (selectedRun ? "overview" : "triage")
    : view;

  // --- run picker: grouped by experiment, chronological, chain-indented --
  const pickerGroups = useMemo(() => {
    const shown = sortRunsByTimestamp(visibleRuns);
    const grouped = new Map<string, WorkspaceRun[]>();
    for (const run of shown) grouped.set(run.experiment, [...(grouped.get(run.experiment) ?? []), run]);
    return [...grouped.entries()].map(([experiment, runs]) => {
      // A run whose source run is also on screen (and in the same
      // experiment) is indented directly beneath it: the chain
      // run → evaluate → analyze is the unit a researcher reads, and
      // strict newest-first would print it backwards.
      const names = new Set(runs.map((run) => run.name));
      const parentOf = (run: WorkspaceRun) => {
        const source = run.sourceRun ?? "";
        return source && names.has(source) ? source : "";
      };
      const children = new Map<string, WorkspaceRun[]>();
      for (const run of runs) {
        const parent = parentOf(run);
        if (parent) children.set(parent, [...(children.get(parent) ?? []), run]);
      }
      const rows: { run: WorkspaceRun; child: boolean }[] = [];
      const emit = (run: WorkspaceRun, child: boolean, depth: number) => {
        rows.push({ run, child });
        if (depth >= 3) return;
        for (const next of children.get(run.name) ?? []) emit(next, true, depth + 1);
      };
      for (const run of runs) if (!parentOf(run)) emit(run, false, 0);
      return { experiment, rows };
    });
  }, [visibleRuns]);

  return (
    <div className="app-shell">
      <aside className={`sidebar ${menuOpen ? "open" : ""}`}>
        <div className="brand"><Mark /><div><strong>SteerLab</strong><span>Research workbench</span></div></div>
        <nav aria-label="Results sections">
          <span className="nav-label">RESULTS</span>
          {visibleNav.map((item) => <button key={item.id} onClick={() => go(item.id)} className={shownView === item.id ? "active" : ""}><span>{item.eyebrow}</span><strong>{item.label}</strong></button>)}
          {selectedRun && (hiddenSectionCount > 0 || showAllSections) && (
            <button className="nav-show-all" onClick={() => setShowAllSections(!showAllSections)}>
              <span>{showAllSections ? "◂" : "▸"}</span>
              <strong>{showAllSections ? "Hide inapplicable sections" : `Show all sections (${hiddenSectionCount} hidden)`}</strong>
            </button>
          )}
        </nav>
        <section className="workspace-widget" aria-label="Local workspace and run selection">
          <span className="nav-label">LOCAL DATA</span>
          <button className={`workspace-folder ${workspaceName ? "connected" : ""}`} onClick={embedded ? loadEmbeddedWorkspace : chooseWorkspace} disabled={scanning}>
            <span className="folder-glyph">⌂</span>
            <div><small>Workspace</small><strong>{scanning ? "Scanning…" : workspaceName || "Choose folder"}</strong></div>
            <b>{workspaceName ? "↻" : "+"}</b>
          </button>
          <button className="run-select-button" onClick={() => setRunPickerOpen(true)} disabled={!workspaceRuns.length || scanning}>
            <div><small>Run</small><strong>{selectedRun?.name || (workspaceRuns.length ? `Select from ${workspaceRuns.length} runs` : "No workspace selected")}</strong></div>
            <span>⌄</span>
          </button>
          {folderError && <p className="workspace-error">{folderError}</p>}
          {selectedRun && <p className="local-read-status"><i />Read only · files stay local</p>}
        </section>
        <div className="sidebar-foot">
          <div className="study-chip"><span>{selectedRun ? "LR" : "—"}</span><div><strong>{selectedRun?.experiment || "No run selected"}</strong><small>{selectedRun?.model || (demoPreviewEnabled() ? "Synthetic preview" : "choose a run above")}</small></div><button aria-label="Change run" onClick={() => workspaceRuns.length && setRunPickerOpen(true)}>⌄</button></div>
          <p><span className={selectedRun ? "local-dot" : "live-dot"} />{selectedRun ? "Local run selected" : demoPreviewEnabled() ? "Synthetic preview data" : "No data loaded"}</p>
        </div>
      </aside>
      <main>
        <header className="topbar">
          <button className="menu-button" onClick={() => setMenuOpen(!menuOpen)} aria-expanded={menuOpen} aria-label="Toggle navigation">☰</button>
          <div className="breadcrumb"><button onClick={() => workspaceRuns.length && setRunPickerOpen(true)}>Runs</button><b>/</b><strong>{selectedRun?.dateLabel || "no run selected"}</strong>
            {/* Status truth, not `status === "complete"`: run-status.json +
                FAILED.md + cancelled.txt, with "not stamped" as the honest
                fallback (upgrade plan Phase 0). */}
            <Badge tone={selectedRun ? statusTone(runStatusOf(selectedRun).state) : "neutral"}>{selectedRun ? statusLabel(runStatusOf(selectedRun)) : "—"}</Badge></div>
          <div className="top-actions"><span className={selectedRun ? "local-epoch" : "epoch"}>{selectedRun ? "Local read" : demoPreviewEnabled() ? "Synthetic preview" : "No run selected"}</span>{workspaceRuns.length > 0 && <button className="icon-button" aria-label="Change selected run" onClick={() => setRunPickerOpen(true)}>⌄</button>}</div>
        </header>
        <div className="content">
          {shownView === "triage" && <TriageView run={selectedRun} workspaceRuns={workspaceRuns} onActivateRun={(run) => void activateRun(run)} onNavigate={go} onOpenFile={openRunFile} />}
          {shownView === "overview" && <Overview onNavigate={go} run={selectedRun} />}
          {shownView === "concepts" && <ConceptEvidenceView run={selectedRun} onOpenFile={openRunFile} />}
          {shownView === "optimization" && <OptimizationView run={selectedRun} onOpenFile={openRunFile} />}
          {shownView === "effects" && <EffectsView run={selectedRun} onOpenFile={openRunFile} />}
          {shownView === "residuals" && <AnalysisOutputsView run={selectedRun} onOpenFile={openRunFile} />}
          {shownView === "battery" && <BatteryView run={selectedRun} onOpenFile={openRunFile} />}
          {shownView === "panels" && <PanelView run={selectedRun} onOpenFile={openRunFile} />}
          {shownView === "choice" && <ChoiceInstrumentView run={selectedRun} workspaceRuns={workspaceRuns} onActivateRun={(run) => void activateRun(run)} onNavigate={go} onOpenFile={openRunFile} />}
          {shownView === "judged" && <JudgedEvaluationView run={selectedRun} workspaceRuns={workspaceRuns} onActivateRun={(run) => void activateRun(run)} onNavigate={go} onOpenFile={openRunFile} />}
          {shownView === "coding" && <CodingView run={selectedRun} workspaceRuns={workspaceRuns} onActivateRun={(run) => void activateRun(run)} onNavigate={go} onOpenFile={openRunFile} />}
          {shownView === "generations" && <GenerationsView run={selectedRun} />}
          {shownView === "provenance" && <ProvenanceView run={selectedRun} onOpenFile={openRunFile} />}
        </div>
      </main>
      {runPickerOpen && <div className="modal-backdrop" role="presentation" onMouseDown={() => !loadingRun && setRunPickerOpen(false)}>
        <section className="run-picker" role="dialog" aria-modal="true" aria-labelledby="run-picker-title" onMouseDown={(event) => event.stopPropagation()}>
          <header><div><span className="section-number">{workspaceName.toUpperCase()} / RUNS</span><h2 id="run-picker-title">Select a run</h2><p>{workspaceRuns.length} readable run {workspaceRuns.length === 1 ? "directory" : "directories"} discovered locally.</p></div><button onClick={() => setRunPickerOpen(false)} aria-label="Close run picker">×</button></header>
          <label className="run-search"><span>⌕</span><input autoFocus value={runSearch} onChange={(event) => setRunSearch(event.target.value)} placeholder="Search experiment, run, path, or model" /></label>
          <div className="run-picker-list">
            {/* Fragment, not a wrapper element: `.run-picker-list > button`
                in globals.css is a direct-child selector. */}
            {pickerGroups.map((group) => <Fragment key={group.experiment}>
              <span className="run-picker-group-label">{group.experiment}</span>
              {group.rows.map(({ run, child }) => {
                const status = runStatusOf(run);
                return <button key={run.key} className={`${selectedRun?.key === run.key ? "selected" : ""} ${child ? "chain-child" : ""}`} onClick={() => activateRun(run)} disabled={loadingRun}>
                  {/* Same dot colour for either completion source — a run
                      whose completion was read from its summary artifact is
                      completed, and the attribution rides in the label
                      ("completed (summary artifact)") and this tooltip. */}
                  <span
                    className={`run-status-dot state-${stateSlug(status.state)}`}
                    title={status.completionSource === "summaryArtifact"
                      ? "completed — read from the stage's summary artifact; this run wrote no run-status.json"
                      : status.completionSource === "statusFile" ? "completed — stamped in run-status.json" : status.state}
                  />
                  <div><strong>{run.name}</strong><span>{runKindLabel(runKindOf(run))} · {statusLabel(status)}</span><small>{run.path}</small></div>
                  <div className="run-picker-meta"><strong>{run.dateLabel}</strong><span>{run.model}</span><small>{run.promptCount || "—"} prompts · {run.conditionCount || "—"} conditions</small></div>
                  <b>{loadingRun ? "…" : "→"}</b>
                </button>;
              })}
            </Fragment>)}
            {!visibleRuns.length && <div className="empty-state">No runs match that search.</div>}
          </div>
          <footer><span><i />{embedded ? "Served read-only from the active SteerLab workspace." : "Folder access is read-only and lasts for this browser session."}</span>{embedded ? <button onClick={loadEmbeddedWorkspace}>Rescan workspace</button> : <button onClick={chooseWorkspace}>Choose another workspace</button>}</footer>
        </section>
      </div>}
      {filePreview && <FilePreviewModal preview={filePreview} onClose={() => setFilePreview(null)} />}
    </div>
  );
}
