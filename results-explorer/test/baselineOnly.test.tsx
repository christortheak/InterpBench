import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("react/jsx-dev-runtime", async (original) => (await import("./support/capture")).wrapJSXRuntime(await original()));
vi.mock("react/jsx-runtime", async (original) => (await import("./support/capture")).wrapJSXRuntime(await original()));

import { embeddedRunsDirectory } from "../app/embedded-workspace";
import { BASELINE_ONLY_EFFECTS, isBaselineOnlyStudy } from "../app/lib/baselineOnly";
import { discoverRuns } from "../app/lib/discovery";
import { hydrateRun } from "../app/lib/loaders";
import type { LocalDirectoryHandle } from "../app/lib/types";
import { EffectsView } from "../app/views/Effects";
import { LocalOverview } from "../app/views/Overview";
import { render } from "./support/capture";
import { enterEmbedded, leaveHost, serveRuns } from "./support/host";

// A study that measures the baseline agent alone (public issue #4): a
// declared agentComparison with no agent, injection, or SAE latent arm. Its
// analysis has no effects by design. The cases are the SHARED fixture the
// Python engine writes (scripts/regenerate-cross-engine-fixtures.py,
// baseline_only_study) and the Mac engine reads too
// (BaselineOnlyStudyTests.swift).

afterEach(leaveHost);

const checkout = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const fixture = JSON.parse(readFileSync(resolve(checkout, "Tests", "Fixtures", "cross-engine", "baseline-only-study.json"), "utf8")) as {
  effectsSentence: string;
  cases: Array<{ label: string; manifest: Record<string, unknown>; baselineOnly: boolean }>;
};

describe("the shared baseline-only fixture", () => {
  it("is this reader's answer for every shape", () => {
    expect(fixture.cases.length).toBeGreaterThan(5);
    for (const entry of fixture.cases) expect(isBaselineOnlyStudy(entry.manifest), entry.label).toBe(entry.baselineOnly);
  });

  it("says no effects in the results page's own words", () => {
    expect(BASELINE_ONLY_EFFECTS).toBe(fixture.effectsSentence);
  });
});

const RUN = "20261031T120000000-exp-replication-analyze";
const header = "condition,endpoint,n,deltaMean,ciLower,ciUpper,wilcoxonW,wilcoxonP,adjustedP,correction,modality,stratifyBy,stratum,unit,estimand,inference\n";
const manifestOf = (label: string) => fixture.cases.find((entry) => entry.label === label)!.manifest;

const load = async (manifest: Record<string, unknown>) => {
  enterEmbedded();
  serveRuns({
    [`${RUN}/report.json`]: JSON.stringify({ experiment: "replication", conditions: { baseline: { generations: 6 } } }),
    [`${RUN}/effect-sizes.csv`]: header,
    [`${RUN}/experiment.json`]: JSON.stringify(manifest),
  });
  const [run] = await discoverRuns(embeddedRunsDirectory() as unknown as LocalDirectoryHandle);
  return hydrateRun(run);
};

describe("an analysis of a baseline-only study", () => {
  it("says baseline only on the overview and the effects page, never a missing table", async () => {
    const run = await load(manifestOf("declared-nothing-attached"));
    expect(run.baselineOnly).toBe(true);
    const overview = render(<LocalOverview run={run} onNavigate={() => {}} />);
    expect(overview.text).toContain("Baseline only");
    expect(overview.text).toContain(BASELINE_ONLY_EFFECTS);
    expect(overview.text).not.toContain("has no readable");
    expect(overview.text).not.toContain("Not available");
    const effects = render(<EffectsView run={run} onOpenFile={() => {}} />);
    expect(effects.text).toContain(BASELINE_ONLY_EFFECTS);
    expect(effects.text).not.toContain("No readable effect rows were found");
  });

  it("leaves an empty analysis of any other study as it was", async () => {
    const run = await load(manifestOf("undeclared-nothing-attached"));
    expect(run.baselineOnly).toBe(false);
    expect(render(<EffectsView run={run} onOpenFile={() => {}} />).text).toContain("No readable effect rows were found");
  });
});
