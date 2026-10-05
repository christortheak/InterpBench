import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("react/jsx-dev-runtime", async (original) => (await import("./support/capture")).wrapJSXRuntime(await original()));
vi.mock("react/jsx-runtime", async (original) => (await import("./support/capture")).wrapJSXRuntime(await original()));

import { FilePreviewModal } from "../app/components/ui";
import { embeddedRunsDirectory } from "../app/embedded-workspace";
import { CSV_PREVIEW_LINES, csvPreview, csvPreviewNotice, skippedLinesNote } from "../app/lib/csv";
import { discoverRuns } from "../app/lib/discovery";
import { hydrateRun, loadEffectTable } from "../app/lib/loaders";
import type { LocalDirectoryHandle, RunFile, WorkspaceRun } from "../app/lib/types";
import { EffectsView } from "../app/views/Effects";
import { GenerationsView } from "../app/views/Generations";
import { LocalProvenanceView } from "../app/views/Provenance";
import { render } from "./support/capture";
import { enterEmbedded, leaveHost, serveRuns } from "./support/host";

// Two places the explorer showed less than the file and did not say so: the
// CSV preview stopped at 250 lines without a word, and skipped lines were
// mentioned only when EVERY line had been skipped.

afterEach(leaveHost);

const table = (rows: number) => ["id,value", ...Array.from({ length: rows }, (_, index) => `${index + 1},${index * 2}`)].join("\n") + "\n";

const csvFile = (name = "summaries.csv"): RunFile => ({ name, path: name, size: 10, modified: 0, handle: { kind: "file", name, getFile: async () => new File([""], name) } });

describe("csvPreview", () => {
  it("lays out a small table whole, and says nothing", () => {
    const preview = csvPreview(table(12));
    expect(preview).toMatchObject({ shownRows: 12, readRows: 12, cut: false, partial: false });
    expect(preview.lines[0]).toEqual(["id", "value"]);
    expect(csvPreviewNotice(preview)).toBe("");
  });

  it("a table of exactly the bound is not called cut", () => {
    const preview = csvPreview(table(CSV_PREVIEW_LINES - 1));
    expect(preview.cut).toBe(false);
    expect(csvPreviewNotice(preview)).toBe("");
  });

  it("says how many rows are shown and how many there are when the table is cut", () => {
    const preview = csvPreview(table(1204));
    expect(preview).toMatchObject({ shownRows: 249, readRows: 1204, cut: true });
    expect(preview.lines).toHaveLength(CSV_PREVIEW_LINES);
    expect(csvPreviewNotice(preview)).toBe("Showing the first 249 of 1204 rows. Download the file to see them all.");
  });

  it("does not claim a total it cannot know when only the start of the file was read", () => {
    // The text is the first part of a larger file: its last line is half a
    // row and is dropped, and the count is a floor.
    const text = `${table(3000)}3001,60`;
    const preview = csvPreview(text, true);
    expect(preview).toMatchObject({ shownRows: 249, readRows: 3000, cut: true, partial: true });
    expect(csvPreviewNotice(preview)).toBe("Showing the first 249 rows. The file is larger than the part read for this preview, which alone holds 3000 rows. Download the file to see them all.");
  });

  it("ignores blank lines and tolerates an empty file", () => {
    expect(csvPreview("a,b\n\n1,2\n\n").readRows).toBe(1);
    expect(csvPreview("")).toMatchObject({ shownRows: 0, readRows: 0, cut: false });
  });
});

describe("the file preview", () => {
  it("tells the reader the table is cut, with the total", () => {
    const page = render(<FilePreviewModal preview={{ file: csvFile(), text: table(1204), truncated: false, loading: false, error: "" }} onClose={() => {}} />);
    expect(page.text).toContain("Showing the first 249 of 1204 rows. Download the file to see them all.");
  });

  it("says nothing extra for a table shown whole", () => {
    const page = render(<FilePreviewModal preview={{ file: csvFile(), text: table(12), truncated: false, loading: false, error: "" }} onClose={() => {}} />);
    expect(page.text).not.toContain("Showing the first");
  });

  it("keeps the two notices apart when the file is also larger than the read", () => {
    const page = render(<FilePreviewModal preview={{ file: csvFile(), text: `${table(3000)}3001,6`, truncated: true, loading: false, error: "" }} onClose={() => {}} />);
    expect(page.text).toContain("Showing the first 1 MB.");
    expect(page.text).toContain("which alone holds 3000 rows");
  });
});

describe("skippedLinesNote", () => {
  it("is empty when nothing was skipped", () => {
    expect(skippedLinesNote(0, "generations.jsonl")).toBe("");
  });

  it("counts, names the file, and says the lines are still there", () => {
    expect(skippedLinesNote(1, "generations.jsonl", "generation records")).toBe("1 line in generations.jsonl could not be read as generation records and is not shown here. It is still in the file.");
    expect(skippedLinesNote(3, "judgments.jsonl", "judgment rows")).toBe("3 lines in judgments.jsonl could not be read as judgment rows and are not shown here. They are still in the file.");
  });
});

const RUN = "20260803T101500000-exp-study-run";
const good = (condition: string, promptID: string) => JSON.stringify({ condition, promptID, sampleIndex: 0, output: "An answer.", prompt: "A question." });

const load = async (files: Record<string, string>): Promise<WorkspaceRun> => {
  enterEmbedded();
  serveRuns({
    [`${RUN}/report.json`]: JSON.stringify({ experiment: "study", conditions: { baseline: { generations: 2 } } }),
    ...Object.fromEntries(Object.entries(files).map(([name, text]) => [`${RUN}/${name}`, text])),
  });
  const [run] = await discoverRuns(embeddedRunsDirectory() as unknown as LocalDirectoryHandle);
  return hydrateRun(run);
};

describe("skipped lines are reported whenever any are skipped", () => {
  it("generations: some rows load, some lines do not, and the reader is told", async () => {
    const run = await load({ "generations.jsonl": [good("baseline", "item-1"), "{not json", good("steered", "item-1"), JSON.stringify({ note: "no condition" })].join("\n") + "\n" });
    expect(run.generationRows).toHaveLength(2);
    expect(run.skippedGenerationLines).toBe(2);
    const page = render(<GenerationsView run={run} />);
    expect(page.text).toContain("2 lines in generations.jsonl could not be read as generation records and are not shown here. They are still in the file.");
  });

  it("generations: a clean file says nothing about skipped lines", async () => {
    const run = await load({ "generations.jsonl": `${good("baseline", "item-1")}\n` });
    expect(render(<GenerationsView run={run} />).text).not.toContain("could not be read");
  });

  it("generations: the zero-rows case still says how many lines were skipped", async () => {
    const run = await load({ "generations.jsonl": "{not json\n{also not\n" });
    expect(render(<GenerationsView run={run} />).text).toContain("2 unparseable lines skipped");
  });

  it("effects: a line with no readable estimate is counted, not silently dropped", async () => {
    const csv = "condition,endpoint,n,deltaMean,ciLower,ciUpper,wilcoxonP,adjustedP,correction\nsteered,choiceRate,12,0.25,0.05,0.45,0.02,0.04,holm\nsteered,wordCount,12,,,,,,holm\n";
    const run = await load({ "effect-sizes.csv": csv });
    expect(await loadEffectTable(run)).toMatchObject({ skipped: 1 });
    expect(run.effectRows).toHaveLength(1);
    expect(run.skippedEffectRows).toBe(1);
    const page = render(<EffectsView run={run} onOpenFile={() => {}} />);
    expect(page.text).toContain("1 line in effect-sizes.csv could not be read as effect rows");
    expect(page.text).toContain("It is still in the file.");
  });

  it("effects: a table read whole says nothing about skipped lines", async () => {
    const run = await load({ "effect-sizes.csv": "condition,endpoint,n,deltaMean,ciLower,ciUpper\nsteered,choiceRate,12,0.25,0.05,0.45\n" });
    expect(run.skippedEffectRows).toBe(0);
    expect(render(<EffectsView run={run} onOpenFile={() => {}} />).text).not.toContain("could not be read");
  });

  it("run files: the file list carries the same counts", async () => {
    const run = await load({
      "generations.jsonl": [good("baseline", "item-1"), "{not json"].join("\n") + "\n",
      "effect-sizes.csv": "condition,endpoint,n,deltaMean,ciLower,ciUpper\nsteered,choiceRate,12,0.25,0.05,0.45\nsteered,wordCount,12,,,\n",
    });
    const page = render(<LocalProvenanceView run={run} onOpenFile={() => {}} />);
    expect(page.text).toContain("1 records loaded · 1 line skipped");
    expect(page.text).toContain("1 effect rows · 1 row skipped");
  });
});
