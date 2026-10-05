"use client";

// The stamps a reader needs in order to judge a result, laid out. Every word
// here comes from lib/ (which reads the run's files); these components only
// arrange it.

import { freezeDetails, freezeLabel, freezeOf, freezeTone, runsBeforeFreeze } from "../lib/freeze";
import type { WorkspaceRun } from "../lib/types";
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
