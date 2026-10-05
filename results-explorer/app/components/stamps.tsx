"use client";

// The stamps a reader needs in order to judge a result, laid out. Every word
// here comes from lib/ (which reads the run's files); these components only
// arrange it.

import { adjudicationSentence, analysisStampsOf, exclusionsSentence, rescueSentence, unitSentence } from "../lib/analysisStamps";
import { freezeDetails, freezeLabel, freezeOf, freezeTone, runsBeforeFreeze } from "../lib/freeze";
import type { RunFile, WorkspaceRun } from "../lib/types";
import { Badge } from "./ui";

/// The study's freeze state as one badge, for the run's header. The tooltip
/// carries the same sentences `FreezeNotice` prints.
export function FreezeBadge({ run }: { run: WorkspaceRun }) {
  const stamp = freezeOf(run);
  const before = runsBeforeFreeze(run);
  const details = freezeDetails(stamp, before);
  return (
    <span className="freeze-badge" title={details.join(" ") || (stamp.frozenAt ? `The study's settings were locked on ${stamp.frozenAt}, before this run.` : "The study's settings were locked before this run.")}>
      <Badge tone={freezeTone(stamp, before)}>{freezeLabel(stamp)}</Badge>
    </span>
  );
}

/// The same facts in full sentences, for the overview. Renders nothing for
/// a plainly frozen study with no check skipped and no exemption: the badge
/// already says "Frozen", and a notice with nothing to notice trains the
/// reader to skip notices.
export function FreezeNotice({ run }: { run: WorkspaceRun }) {
  const stamp = freezeOf(run);
  const before = runsBeforeFreeze(run);
  const details = freezeDetails(stamp, before);
  if (!details.length) return null;
  const caution = freezeTone(stamp, before) === "warn" || stamp.batteryNotApplied.length > 0;
  return (
    <div className={`notice freeze-notice ${caution ? "" : "local-notice"}`} role="note">
      <span className="notice-icon">{caution ? "!" : "i"}</span>
      <p>
        <strong>{freezeLabel(stamp)}.</strong>{" "}
        {details.join(" ")}
      </p>
    </div>
  );
}

/// What the analysis did to the records before it estimated anything, shown
/// BESIDE the effects it produced: the unit of analysis, declared exclusions,
/// endpoint rescue, and adjudication. Each row is one stamp file from the
/// run directory; a stamp the run does not carry says so in words rather
/// than disappearing, so "none" is a statement and not a gap.
export function AnalysisStampsCard({ run, onOpenFile }: { run: WorkspaceRun; onOpenFile: (file: RunFile) => void }) {
  const stamps = analysisStampsOf(run);
  const file = (name: string) => run.files.find((candidate) => candidate.path === name) ?? null;
  const open = (name: string) => {
    const found = file(name);
    return found ? <button className="quiet-link" onClick={() => onOpenFile(found)}>{name}</button> : null;
  };
  const exclusions = stamps.exclusions;
  const conditions = exclusions ? Object.keys(exclusions.consideredN).sort() : [];
  return (
    <section className="card analysis-stamps-card" aria-label="What this analysis did to the records">
      <header className="section-header">
        <div><span className="section-number">BEFORE THE ESTIMATES</span><h2>What this analysis did to the records</h2></div>
      </header>
      <dl className="analysis-stamps">
        <div>
          <dt>Unit of analysis</dt>
          <dd>
            <p>{unitSentence(stamps.unit)}</p>
            {stamps.unit && <small>Stamped in {stamps.unit.source}. {open(stamps.unit.source)}</small>}
          </dd>
        </div>
        <div>
          <dt>Exclusions</dt>
          <dd>
            <p>{exclusionsSentence(exclusions)}</p>
            {exclusions && exclusions.rules.length > 0 && (
              <ul>{exclusions.rules.map((rule) => <li key={rule.rule}><strong>{rule.rule}</strong>{rule.description ? `: ${rule.description}` : ""}</li>)}</ul>
            )}
            {conditions.length > 0 && (
              <p className="analysis-stamps-counts">Records left for each condition: {conditions.map((condition) => `${condition} ${exclusions!.survivingN[condition] ?? "—"} of ${exclusions!.consideredN[condition]}`).join(", ")}.</p>
            )}
            {exclusions && <small>Stamped in {exclusions.source}. {open(exclusions.source)}</small>}
          </dd>
        </div>
        <div>
          <dt>Endpoint rescue</dt>
          <dd>
            <p>{rescueSentence(stamps.rescue)}</p>
            {stamps.rescue && <small>Stamped in endpoint-reparse.json. {open("endpoint-reparse.json")}</small>}
          </dd>
        </div>
        <div>
          <dt>Adjudication</dt>
          <dd>
            <p>{adjudicationSentence(stamps.adjudication)}</p>
            {stamps.adjudication && <small>Stamped in {stamps.adjudication.source}. {open(stamps.adjudication.source)} {open("adjudication-divergence.csv")}</small>}
          </dd>
        </div>
        {stamps.epochUnverified && (
          <div>
            <dt>Study fingerprint</dt>
            <dd><p>Not verified. The run this analysis read carried no study fingerprint, so the analysis could not confirm that it belongs to this version of the study, and it went ahead anyway.</p></dd>
          </div>
        )}
        {stamps.measurementDrift && (
          <div>
            <dt>Measurement settings</dt>
            <dd><p>These settings differed from the ones the run was made under, and the analysis used the current ones: <code>{stamps.measurementDrift}</code></p></dd>
          </div>
        )}
      </dl>
    </section>
  );
}
