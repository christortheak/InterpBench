import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("react/jsx-dev-runtime", async (original) => (await import("./support/capture")).wrapJSXRuntime(await original()));
vi.mock("react/jsx-runtime", async (original) => (await import("./support/capture")).wrapJSXRuntime(await original()));

import { TruncationCard } from "../app/components/stamps";
import { embeddedRunsDirectory } from "../app/embedded-workspace";
import { discoverRuns } from "../app/lib/discovery";
import { hydrateRun } from "../app/lib/loaders";
import { cellsOverThreshold, conditionErrors, cutOffCells, parseTruncation, percentText, thresholdSentence, truncationSentence } from "../app/lib/runReport";
import type { LocalDirectoryHandle, WorkspaceRun } from "../app/lib/types";
import { LocalOverview } from "../app/views/Overview";
import { render } from "./support/capture";
import { enterEmbedded, leaveHost, serveRuns } from "./support/host";

// Two things report.json says that the overview left out: a condition that
// failed, and how many generations were cut off at the length limit. The
// shapes are the engines' own (run_reporting.py writes `conditions[c].error`
// for a condition that produced no generations; truncation_gate.report writes
// the `truncation` block for every run).

afterEach(leaveHost);

const truncation = {
  threshold: 0.25,
  classified: 40,
  lengthStopped: 9,
  lengthStoppedInReasoning: 2,
  lengthStoppedFraction: 0.225,
  cells: [
    { condition: "baseline", promptID: "item-1", classified: 10, lengthStopped: 0, lengthStoppedInReasoning: 0, lengthStoppedFraction: 0 },
    { condition: "baseline", promptID: "item-2", classified: 10, lengthStopped: 1, lengthStoppedInReasoning: 0, lengthStoppedFraction: 0.1 },
    { condition: "steered", promptID: "item-1", classified: 10, lengthStopped: 6, lengthStoppedInReasoning: 2, lengthStoppedFraction: 0.6 },
    { condition: "steered", promptID: "item-2", classified: 10, lengthStopped: 2, lengthStoppedInReasoning: 0, lengthStoppedFraction: 0.2 },
  ],
};

describe("conditionErrors", () => {
  it("finds a condition that carries an error instead of numbers", () => {
    const report = { conditions: { baseline: { generations: 8 }, steered: { generations: 0, error: "adapter weights not found for agent 'steered'" } } };
    expect(conditionErrors(report)).toEqual([{ condition: "steered", error: "adapter weights not found for agent 'steered'" }]);
  });

  it("finds none in a clean report, and none in a report without conditions", () => {
    expect(conditionErrors({ conditions: { baseline: { generations: 8 } } })).toEqual([]);
    expect(conditionErrors({})).toEqual([]);
  });

  it("keeps an error that is not a string, rather than dropping it", () => {
    expect(conditionErrors({ conditions: { a: { error: { type: "OOM", detail: "out of memory" } } } })).toEqual([{ condition: "a", error: "{\"type\":\"OOM\",\"detail\":\"out of memory\"}" }]);
  });
});

describe("parseTruncation", () => {
  const block = parseTruncation({ truncation })!;

  it("reads the run-wide counts and every cell", () => {
    expect(block).toMatchObject({ threshold: 0.25, classified: 40, lengthStopped: 9, lengthStoppedInReasoning: 2, lengthStoppedFraction: 0.225 });
    expect(block.cells).toHaveLength(4);
  });

  it("is null for a report that has no truncation block", () => {
    expect(parseTruncation({})).toBeNull();
    expect(parseTruncation({ truncation: {} })).toBeNull();
  });

  it("lists the cells with a cut-off generation, worst first", () => {
    expect(cutOffCells(block).map((cell) => `${cell.condition}/${cell.promptID}`)).toEqual(["steered/item-1", "steered/item-2", "baseline/item-2"]);
  });

  it("marks a cell over the study's limit only when it is strictly over, as the engines do", () => {
    expect(cellsOverThreshold(block).map((cell) => `${cell.condition}/${cell.promptID}`)).toEqual(["steered/item-1"]);
    const atLimit = parseTruncation({ truncation: { ...truncation, threshold: 0.6 } })!;
    expect(cellsOverThreshold(atLimit)).toEqual([]);
    const noLimit = parseTruncation({ truncation: { ...truncation, threshold: null } })!;
    expect(cellsOverThreshold(noLimit)).toEqual([]);
  });

  it("says in plain words how many were cut off, and how many never reached an answer", () => {
    expect(truncationSentence(block)).toBe("9 of 40 generations (23%) were cut off at the length limit, so their text stops early. 2 of those were cut off while the model was still reasoning, before it gave any answer.");
    expect(thresholdSentence(block)).toBe("The study allowed at most 25% for any one condition on any one item.");
    expect(thresholdSentence({ ...block, threshold: null })).toBe("The study set no limit on how many may be cut off.");
  });

  it("says so when nothing was cut off, and when the run recorded no endings at all", () => {
    expect(truncationSentence({ ...block, lengthStopped: 0, lengthStoppedInReasoning: 0, lengthStoppedFraction: 0 })).toBe("None of the 40 generations that recorded how they ended was cut off at the length limit.");
    expect(truncationSentence({ ...block, classified: 0, lengthStopped: 0 })).toContain("cannot say whether any was cut off");
  });

  it("formats small shares without rounding them to zero", () => {
    expect(percentText(0.004)).toBe("0.4%");
    expect(percentText(0.6)).toBe("60%");
    expect(percentText(0)).toBe("0%");
    expect(percentText(null)).toBe("—");
  });
});

const RUN = "20260803T101500000-exp-study-run";

const load = async (report: Record<string, unknown>): Promise<WorkspaceRun> => {
  enterEmbedded();
  serveRuns({
    [`${RUN}/report.json`]: JSON.stringify({ experiment: "study", ...report }),
    [`${RUN}/config.json`]: JSON.stringify({ modelID: "org/model", runType: "run" }),
  });
  const [run] = await discoverRuns(embeddedRunsDirectory() as unknown as LocalDirectoryHandle);
  return hydrateRun(run);
};

describe("the overview shows them", () => {
  it("a failed condition is named, with the engine's error, instead of a row of dashes", async () => {
    const run = await load({ conditions: { baseline: { generations: 8, meanWordCount: 120 }, steered: { generations: 0, error: "adapter weights not found for agent 'steered'" } } });
    const page = render(<LocalOverview run={run} onNavigate={() => {}} />);
    expect(page.text).toContain("1 condition failed and produced no generations.");
    expect(page.text).toContain("steered: adapter weights not found for agent 'steered'");
    expect(page.text).toContain("steered · failed");
    expect(page.html).toContain("condition-failed");
  });

  it("a clean run shows no failure notice", async () => {
    const run = await load({ conditions: { baseline: { generations: 8 }, steered: { generations: 8 } } });
    const page = render(<LocalOverview run={run} onNavigate={() => {}} />);
    expect(page.text).not.toContain("failed");
  });

  it("the truncation block is rendered, worst cells first, with the study's limit", async () => {
    const run = await load({ conditions: { baseline: { generations: 20 }, steered: { generations: 20 } }, truncation });
    const page = render(<LocalOverview run={run} onNavigate={() => {}} />);
    expect(page.text).toContain("Some generations stop early");
    expect(page.text).toContain("9 of 40 generations (23%) were cut off at the length limit");
    expect(page.text).toContain("The study allowed at most 25% for any one condition on any one item.");
    expect(page.text).toContain("1 over the study’s limit");
    expect(page.text).toContain("steered item-1 6 10 60% 2 Over");
    expect(page.text).toContain("steered item-2 2 10 20% 0 Within");
    // A cell with nothing cut off is not listed.
    expect(page.text).not.toContain("baseline item-1 0 10");
  });

  it("a run with nothing cut off says so plainly", async () => {
    const run = await load({ conditions: {}, truncation: { threshold: null, classified: 12, lengthStopped: 0, lengthStoppedInReasoning: 0, lengthStoppedFraction: 0, cells: [] } });
    const page = render(<TruncationCard run={run} />);
    expect(page.text).toContain("No generation was cut off");
    expect(page.text).toContain("None of the 12 generations that recorded how they ended was cut off at the length limit. The study set no limit on how many may be cut off.");
  });

  it("a report from before the block existed shows no card, rather than an empty one", async () => {
    const run = await load({ conditions: { baseline: { generations: 8 } } });
    expect(render(<TruncationCard run={run} />).html).toBe("");
  });

  it("says how many affected cells are not shown when there are more than twelve", async () => {
    const cells = Array.from({ length: 15 }, (_, index) => ({ condition: "steered", promptID: `item-${String(index).padStart(2, "0")}`, classified: 4, lengthStopped: 1, lengthStoppedInReasoning: 0, lengthStoppedFraction: 0.25 }));
    const run = await load({ conditions: {}, truncation: { threshold: null, classified: 60, lengthStopped: 15, lengthStoppedInReasoning: 0, lengthStoppedFraction: 0.25, cells } });
    expect(render(<TruncationCard run={run} />).text).toContain("Showing the 12 most affected of 15 condition-and-item pairs");
  });
});
