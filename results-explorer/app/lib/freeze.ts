// The study's freeze state, read from the manifest snapshot a run carries.
//
// Every stage writes the study's manifest into its run directory as
// `experiment.json`. That snapshot says whether the study's settings were
// locked ("frozen") when the run was made, whether the freeze was forced past
// checks that had not passed, and whether the capability check was left off
// any condition. A reader needs all three to judge a result, and the explorer
// used to read none of them.
//
// Nothing here is inferred. A run without a snapshot says its freeze state is
// not recorded; it is never assumed frozen, and never assumed not.

import { recordValue, runKindOf, textValue } from "./discovery";
import type { WorkspaceRun } from "./types";

export type BatteryNotApplied = { condition: string; reason: string };

export type FreezeStamp = {
  /// Whether the run carried a readable manifest snapshot at all.
  present: boolean;
  /// The manifest's `status`, verbatim ("frozen", "draft"); "" when absent.
  status: string;
  frozen: boolean;
  frozenAt: string;
  freezeHash: string;
  /// The freeze was forced: `freezeForced` is true, or checks are listed as
  /// skipped.
  forced: boolean;
  /// The checks a forced freeze skipped, by the engines' ids, in the order
  /// stamped.
  forcedGates: string[];
  /// Conditions the capability check was not applied to, and why. Read from
  /// the manifest's freeze stamp and from the run's own report; one entry
  /// per condition.
  batteryNotApplied: BatteryNotApplied[];
};

export const emptyFreezeStamp = (): FreezeStamp => ({
  present: false, status: "", frozen: false, frozenAt: "", freezeHash: "",
  forced: false, forcedGates: [], batteryNotApplied: [],
});

const notApplied = (raw: unknown): BatteryNotApplied[] =>
  (Array.isArray(raw) ? raw : []).flatMap((item): BatteryNotApplied[] => {
    const entry = recordValue(item);
    const condition = textValue(entry, "condition");
    return condition ? [{ condition, reason: textValue(entry, "reason") }] : [];
  });

/// `manifest` is the run's `experiment.json`; `report` is its `report.json`,
/// which repeats the capability-check exemption for the conditions the run
/// actually had. Either may be `{}`.
export const parseFreezeStamp = (manifest: Record<string, unknown>, report: Record<string, unknown> = {}): FreezeStamp => {
  const present = Object.keys(manifest).length > 0;
  const status = textValue(manifest, "status");
  const forcedGates = (Array.isArray(manifest.forcedGatesSkipped) ? manifest.forcedGatesSkipped : [])
    .filter((gate): gate is string => typeof gate === "string" && gate !== "");
  const seen = new Set<string>();
  const batteryNotApplied = [...notApplied(manifest.capabilityBatteryNotApplied), ...notApplied(report.capabilityBatteryNotApplied)]
    .filter((entry) => !seen.has(entry.condition) && Boolean(seen.add(entry.condition)));
  return {
    present,
    status,
    frozen: status === "frozen",
    frozenAt: textValue(manifest, "frozenAt"),
    freezeHash: textValue(manifest, "freezeHash"),
    forced: manifest.freezeForced === true || forcedGates.length > 0,
    forcedGates,
    batteryNotApplied,
  };
};

/// The run's freeze stamp, or the "not recorded" one for a run that has not
/// been activated or was built by hand in a test.
export const freezeOf = (run: WorkspaceRun): FreezeStamp => run.freeze ?? emptyFreezeStamp();

/// Kinds of run that come BEFORE the freeze in a study's life: extracting a
/// concept, validating it, and sweeping for a setting all prepare the study
/// that is then frozen.
export const runsBeforeFreeze = (run: WorkspaceRun): boolean =>
  ["extract", "validate", "sweep"].includes(runKindOf(run));

/// What each check a forced freeze can skip is, in plain words. Keyed by the
/// engines' gate ids; an id not listed here is shown as it is.
const GATE_MEANING: Record<string, string> = {
  revision: "the model is pinned to one exact version",
  validateEvidence: "the concepts have matching validation evidence",
  batteryEvidence: "each condition has capability-check evidence",
  judgeValidity: "the judges and the rubric are valid",
  variantValidity: "each agent's saved files are complete",
  gitClean: "the pinned inputs are committed",
  measurementPins: "the measurement settings can be loaded",
};

export const gateLabel = (gate: string) => GATE_MEANING[gate] ? `${GATE_MEANING[gate]} (${gate})` : gate;

/// The short label for the run's header.
export const freezeLabel = (stamp: FreezeStamp): string => {
  if (!stamp.present) return "Freeze state not recorded";
  if (!stamp.frozen) return stamp.status ? `Not frozen (${stamp.status})` : "Not frozen";
  return stamp.forced ? "Frozen, with checks skipped" : "Frozen";
};

/// `beforeFreeze`: the run is of a kind that comes before the freeze in the
/// study's life (extracting, validating, or sweeping). Not being frozen is
/// then the normal state, not a caution.
export const freezeTone = (stamp: FreezeStamp, beforeFreeze = false): "neutral" | "good" | "warn" => {
  if (!stamp.present) return "neutral";
  if (!stamp.frozen) return beforeFreeze ? "neutral" : "warn";
  return stamp.forced ? "warn" : "good";
};

/// The sentence for one capability-check exemption. For the one reason the
/// engines record today this is their own sentence, word for word
/// (`freeze_policy.battery_not_applied_sentence` and its Swift twin).
export const batteryNotAppliedSentence = ({ condition, reason }: BatteryNotApplied): string =>
  reason === "interventionPolicy"
    ? `The capability battery was not applied to ${condition}, because its agent uses an intervention policy, which the battery cannot run. This study has no capability control for that agent.`
    : `The capability battery was not applied to ${condition}${reason ? ` (recorded reason: ${reason})` : ""}. This study has no capability control for that agent.`;

/// Everything a reader should be told beyond the short label, one sentence
/// per fact. Empty for a plainly frozen study with nothing skipped.
export const freezeDetails = (stamp: FreezeStamp, beforeFreeze = false): string[] => {
  if (!stamp.present) return ["This run has no experiment.json, so the explorer cannot say whether the study was frozen when the run was made."];
  const details: string[] = [];
  if (!stamp.frozen) {
    details.push(beforeFreeze
      ? "This run was made before the study was frozen. That is the usual order: this kind of run prepares the study, and the freeze comes after it."
      : "This run was made while the study was not frozen, so its settings could still change. A frozen study is one whose settings were locked before the run.");
  } else if (stamp.forced) {
    details.push(stamp.forcedGates.length
      ? `The study was frozen with force, which skipped ${stamp.forcedGates.length === 1 ? "a check" : `${stamp.forcedGates.length} checks`} that had not passed: ${stamp.forcedGates.map(gateLabel).join("; ")}.`
      : "The study was frozen with force. The manifest does not list which checks were skipped.");
  }
  for (const entry of stamp.batteryNotApplied) details.push(batteryNotAppliedSentence(entry));
  return details;
};
