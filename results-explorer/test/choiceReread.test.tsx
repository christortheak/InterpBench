import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("react/jsx-dev-runtime", async (original) => (await import("./support/capture")).wrapJSXRuntime(await original()));
vi.mock("react/jsx-runtime", async (original) => (await import("./support/capture")).wrapJSXRuntime(await original()));

import { embeddedRunsDirectory } from "../app/embedded-workspace";
import { analysisStampsOf, choiceRereadSentence, hasAnalysisStamps, parseAnalysisStamps } from "../app/lib/analysisStamps";
import { discoverRuns } from "../app/lib/discovery";
import { hydrateRun } from "../app/lib/loaders";
import type { LocalDirectoryHandle } from "../app/lib/types";
import { EffectsView } from "../app/views/Effects";
import { render } from "./support/capture";
import { enterEmbedded, leaveHost, serveRuns } from "./support/host";

// `analyze --reread-choices` writes choice-reread.json beside the effects. The
// explorer shows its sentence, built from the stamped numbers alone, in the
// engines' own words: the hand-written cross-engine fixture holds the
// sentence both engines build for its counts.

afterEach(leaveHost);

const checkout = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const fixture = JSON.parse(readFileSync(resolve(checkout, "Tests", "Fixtures", "cross-engine", "choice-reread.json"), "utf8")) as {
  parserVersion: string;
  expected: { total: Record<string, number>; conditions: Record<string, Record<string, number>>; sentence: string; note: string };
};

const stampFile = {
  parserVersion: fixture.parserVersion, sourceRun: "20261017T000000000-exp-reread-run", runUnchanged: true,
  taskPrompts: { file: "prompts/tasks/reread.jsonl", sha256: "a".repeat(64), readFrom: "prompts/tasks/reread.jsonl", location: "workspace" },
  total: fixture.expected.total, conditions: fixture.expected.conditions, examples: {}, note: fixture.expected.note,
};

describe("the choice re-read stamp", () => {
  it("reads the stamp and says the engines' sentence from its numbers", () => {
    const stamps = parseAnalysisStamps({ choiceReread: stampFile });
    expect(hasAnalysisStamps(stamps)).toBe(true);
    expect(choiceRereadSentence(stamps.choiceReread)).toBe(fixture.expected.sentence);
  });

  it("says none was recorded for an analysis that did not re-read", () => {
    expect(parseAnalysisStamps({}).choiceReread).toBeNull();
    expect(choiceRereadSentence(null)).toBe("None recorded. Every choice in this analysis is the one the run stored when it was made.");
  });

  it("shows the sentence beside the effects of a re-read analysis", async () => {
    const RUN = "20261017T000100000-exp-reread-analyze";
    enterEmbedded();
    serveRuns({
      [`${RUN}/analysis.json`]: JSON.stringify({ experiment: "reread", sourceRun: stampFile.sourceRun }),
      [`${RUN}/effect-sizes.csv`]: "condition,endpoint,n,deltaMean,ciLower,ciUpper,wilcoxonW,wilcoxonP,adjustedP,correction,modality,stratifyBy,stratum,unit,estimand,inference\nsteered,choiceRate,5,0.2,0.0,0.6,0,1.0,1.0,bh,injection,pooled,,item,,\n",
      [`${RUN}/choice-reread.json`]: JSON.stringify(stampFile),
    });
    const [run] = await discoverRuns(embeddedRunsDirectory() as unknown as LocalDirectoryHandle);
    const hydrated = await hydrateRun(run);
    expect(analysisStampsOf(hydrated).choiceReread?.changed).toBe(5);
    const page = render(<EffectsView run={hydrated} onOpenFile={() => {}} />);
    expect(page.text).toContain("Choices re-read");
    expect(page.text).toContain(fixture.expected.sentence);
  });
});
