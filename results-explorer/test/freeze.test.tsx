import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("react/jsx-dev-runtime", async (original) => (await import("./support/capture")).wrapJSXRuntime(await original()));
vi.mock("react/jsx-runtime", async (original) => (await import("./support/capture")).wrapJSXRuntime(await original()));

import { FreezeBadge, FreezeNotice } from "../app/components/stamps";
import { embeddedRunsDirectory } from "../app/embedded-workspace";
import { discoverRuns } from "../app/lib/discovery";
import { batteryNotAppliedSentence, emptyFreezeStamp, freezeDetails, freezeLabel, freezeOf, freezeTone, gateLabel, parseFreezeStamp } from "../app/lib/freeze";
import { hydrateRun } from "../app/lib/loaders";
import type { LocalDirectoryHandle, WorkspaceRun } from "../app/lib/types";
import { LocalOverview } from "../app/views/Overview";
import { LocalProvenanceView } from "../app/views/Provenance";
import { render } from "./support/capture";
import { enterEmbedded, leaveHost, serveRuns } from "./support/host";

// The explorer never read the study's manifest, so a forced freeze and a
// capability-check exemption were invisible. Each run carries the manifest
// as `experiment.json`; these pin what is read from it and where it shows.

afterEach(leaveHost);

const frozen = { name: "study", status: "frozen", frozenAt: "2026-08-01T10:00:00Z", freezeHash: "abc123" };
const forced = { ...frozen, freezeForced: true, forcedGatesSkipped: ["validateEvidence", "gitClean"] };
const exempt = { ...frozen, capabilityBatteryNotApplied: [{ condition: "policy-agent", reason: "interventionPolicy" }] };
const draft = { name: "study", status: "draft" };

describe("parseFreezeStamp", () => {
  it("reads a plain freeze: frozen, nothing skipped, nothing exempt", () => {
    const stamp = parseFreezeStamp(frozen);
    expect(stamp).toMatchObject({ present: true, frozen: true, forced: false, forcedGates: [], batteryNotApplied: [], frozenAt: "2026-08-01T10:00:00Z", freezeHash: "abc123" });
    expect(freezeLabel(stamp)).toBe("Frozen");
    expect(freezeTone(stamp)).toBe("good");
    expect(freezeDetails(stamp)).toEqual([]);
  });

  it("reads a forced freeze and names every check it skipped", () => {
    const stamp = parseFreezeStamp(forced);
    expect(stamp.forced).toBe(true);
    expect(stamp.forcedGates).toEqual(["validateEvidence", "gitClean"]);
    expect(freezeLabel(stamp)).toBe("Frozen, with checks skipped");
    expect(freezeTone(stamp)).toBe("warn");
    const [sentence] = freezeDetails(stamp);
    expect(sentence).toContain("skipped 2 checks");
    expect(sentence).toContain("the concepts have matching validation evidence (validateEvidence)");
    expect(sentence).toContain("the pinned inputs are committed (gitClean)");
  });

  it("treats listed skipped checks as a forced freeze even without the flag, and the flag alone as forced", () => {
    expect(parseFreezeStamp({ status: "frozen", forcedGatesSkipped: ["revision"] }).forced).toBe(true);
    const flagOnly = parseFreezeStamp({ status: "frozen", freezeForced: true });
    expect(flagOnly.forced).toBe(true);
    expect(freezeDetails(flagOnly)[0]).toContain("does not list which checks were skipped");
  });

  it("reads a capability-check exemption, which is not a forced freeze", () => {
    const stamp = parseFreezeStamp(exempt);
    expect(stamp.forced).toBe(false);
    expect(freezeLabel(stamp)).toBe("Frozen");
    expect(stamp.batteryNotApplied).toEqual([{ condition: "policy-agent", reason: "interventionPolicy" }]);
    // The engines' own sentence for this reason, word for word.
    expect(freezeDetails(stamp)).toEqual(["The capability battery was not applied to policy-agent, because its agent uses an intervention policy, which the battery cannot run. This study has no capability control for that agent."]);
    expect(batteryNotAppliedSentence({ condition: "c", reason: "somethingNew" })).toBe("The capability battery was not applied to c (recorded reason: somethingNew). This study has no capability control for that agent.");
  });

  it("also reads the exemption the run's own report repeats, once per condition", () => {
    const stamp = parseFreezeStamp(exempt, { capabilityBatteryNotApplied: [{ condition: "policy-agent", reason: "interventionPolicy" }, { condition: "other", reason: "interventionPolicy" }] });
    expect(stamp.batteryNotApplied.map((entry) => entry.condition)).toEqual(["policy-agent", "other"]);
  });

  it("says a draft is not frozen — as a caution for a study run, as the normal order for a preparing run", () => {
    const stamp = parseFreezeStamp(draft);
    expect(stamp.frozen).toBe(false);
    expect(freezeLabel(stamp)).toBe("Not frozen (draft)");
    expect(freezeTone(stamp)).toBe("warn");
    expect(freezeDetails(stamp)[0]).toContain("its settings could still change");
    expect(freezeTone(stamp, true)).toBe("neutral");
    expect(freezeDetails(stamp, true)[0]).toContain("That is the usual order");
  });

  it("never assumes: a run without a snapshot says its freeze state is not recorded", () => {
    const stamp = parseFreezeStamp({});
    expect(stamp).toEqual(emptyFreezeStamp());
    expect(freezeLabel(stamp)).toBe("Freeze state not recorded");
    expect(freezeTone(stamp)).toBe("neutral");
    expect(freezeDetails(stamp)[0]).toContain("no experiment.json");
  });

  it("shows a check id it does not know as it is", () => {
    expect(gateLabel("somethingNew")).toBe("somethingNew");
  });
});

const RUN = "20260803T101500000-exp-study-run";

const files = (manifest: Record<string, unknown> | null, report: Record<string, unknown> = {}) => ({
  [`${RUN}/report.json`]: JSON.stringify({ experiment: "study", conditions: { baseline: { generations: 2 }, "policy-agent": { generations: 2 } }, ...report }),
  [`${RUN}/config.json`]: JSON.stringify({ modelID: "org/model", runType: "run" }),
  ...(manifest ? { [`${RUN}/experiment.json`]: JSON.stringify(manifest) } : {}),
  // A nested file of the same name says nothing about THIS run.
  [`${RUN}/nested/experiment.json`]: JSON.stringify({ status: "frozen", freezeForced: true, forcedGatesSkipped: ["revision"] }),
});

const loadRun = async (served: Record<string, string>): Promise<WorkspaceRun> => {
  enterEmbedded();
  serveRuns(served);
  const [run] = await discoverRuns(embeddedRunsDirectory() as unknown as LocalDirectoryHandle);
  return hydrateRun(run);
};

describe("the freeze state reaches the run's header", () => {
  it("is read from the run's own experiment.json when the run is activated", async () => {
    const run = await loadRun(files(forced));
    expect(freezeOf(run).forcedGates).toEqual(["validateEvidence", "gitClean"]);
  });

  it("is 'not recorded' for a run with no snapshot at its top level", async () => {
    const run = await loadRun(files(null));
    expect(freezeOf(run).present).toBe(false);
  });

  it("badge: a forced freeze is labelled in the header, with the skipped checks in its tooltip", async () => {
    const run = await loadRun(files(forced));
    const page = render(<FreezeBadge run={run} />);
    expect(page.text).toBe("Frozen, with checks skipped");
    expect(page.html).toContain("badge-warn");
    expect(page.html).toContain("validateEvidence");
  });

  it("badge: a plain freeze reads Frozen, and a missing snapshot reads as not recorded", async () => {
    expect(render(<FreezeBadge run={await loadRun(files(frozen))} />).text).toBe("Frozen");
    leaveHost();
    expect(render(<FreezeBadge run={await loadRun(files(null))} />).text).toBe("Freeze state not recorded");
  });

  it("notice: says nothing for a plain freeze, and everything for a forced one with an exemption", async () => {
    expect(render(<FreezeNotice run={await loadRun(files(frozen))} />).html).toBe("");
    leaveHost();
    const page = render(<FreezeNotice run={await loadRun(files({ ...forced, ...exempt, freezeForced: true, forcedGatesSkipped: ["batteryEvidence"] }))} />);
    expect(page.text).toContain("Frozen, with checks skipped.");
    expect(page.text).toContain("each condition has capability-check evidence (batteryEvidence)");
    expect(page.text).toContain("The capability battery was not applied to policy-agent");
  });

  it("overview: the run's header carries the freeze state and its notice", async () => {
    const run = await loadRun(files(exempt));
    const page = render(<LocalOverview run={run} onNavigate={() => {}} />);
    expect(page.text).toContain("Frozen");
    expect(page.text).toContain("The capability battery was not applied to policy-agent, because its agent uses an intervention policy");
  });

  it("overview: a draft study run is told apart from a frozen one", async () => {
    const page = render(<LocalOverview run={await loadRun(files(draft))} onNavigate={() => {}} />);
    expect(page.text).toContain("Not frozen (draft)");
    expect(page.text).toContain("its settings could still change");
  });

  it("run files: the stored metadata lists the state, the skipped checks, and the exemption", async () => {
    const run = await loadRun(files({ ...forced, capabilityBatteryNotApplied: exempt.capabilityBatteryNotApplied }));
    const page = render(<LocalProvenanceView run={run} onOpenFile={() => {}} />);
    expect(page.text).toContain("Study state Frozen, with checks skipped");
    expect(page.text).toContain("Frozen at 2026-08-01T10:00:00Z");
    expect(page.text).toContain("Checks skipped by force the concepts have matching validation evidence (validateEvidence); the pinned inputs are committed (gitClean)");
    expect(page.text).toContain("Capability check not applied to policy-agent (interventionPolicy)");
  });

  it("run files: a plain freeze says no check was skipped, rather than saying nothing", async () => {
    const page = render(<LocalProvenanceView run={await loadRun(files(frozen))} onOpenFile={() => {}} />);
    expect(page.text).toContain("Checks skipped by force None");
    expect(page.text).toContain("Capability check not applied to None recorded");
  });
});
