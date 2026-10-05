import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("react/jsx-dev-runtime", async (original) => (await import("./support/capture")).wrapJSXRuntime(await original()));
vi.mock("react/jsx-runtime", async (original) => (await import("./support/capture")).wrapJSXRuntime(await original()));

import { ExportButton, FilePreviewModal, SaveStatus } from "../app/components/ui";
import { embeddedRunsDirectory } from "../app/embedded-workspace";
import { splitCSV } from "../app/lib/csv";
import { discoverRuns } from "../app/lib/discovery";
import type { ExportColumn } from "../app/lib/export";
import { hydrateRun } from "../app/lib/loaders";
import { readSaveReply, saveRunFile, saveStatusText, saveText, SAVE_PATH_UNKNOWN, SAVE_UNAVAILABLE } from "../app/lib/save";
import type { LocalDirectoryHandle, LocalFileHandle, WorkspaceRun } from "../app/lib/types";
import { EffectsView } from "../app/views/Effects";
import { GenerationsView } from "../app/views/Generations";
import { TriageView } from "../app/views/Triage";
import { render } from "./support/capture";
import { enterBrowser, enterEmbedded, leaveHost, serveRuns } from "./support/host";

// Every export and download control, pressed in EMBEDDED mode (the Mac
// app's web view) and in the standalone browser build.
//
// What "pressed" means here is spelled out in support/capture.ts: the real
// component is rendered by React's server renderer in Node and its click
// handler is called. There is no WebKit and no save panel in this suite, so
// these tests pin what each control SENDS to the native host. What the host
// does with it is ResultsExplorerBridgeTests on the Swift side.

afterEach(leaveHost);

const RUN = "20260803T101500000-exp-demo-run";

const generationLine = (condition: string, promptID: string, output: string) =>
  JSON.stringify({ condition, promptID, sampleIndex: 0, output, prompt: "Decide the case.", wordCount: 3, distinct2: 0.9, seed: 7 });

const runFiles = {
  [`${RUN}/report.json`]: JSON.stringify({ experiment: "demo", conditions: { baseline: { generations: 1 }, steered: { generations: 1 } } }),
  [`${RUN}/config.json`]: JSON.stringify({ modelID: "org/model", runType: "run" }),
  [`${RUN}/generations.jsonl`]: `${generationLine("baseline", "item-1", "Rule applies here.")}\n${generationLine("steered", "item-1", "Equity applies here.")}\n`,
  [`${RUN}/effect-sizes.csv`]: "condition,endpoint,n,deltaMean,ciLower,ciUpper,wilcoxonW,wilcoxonP,adjustedP,correction,modality,stratifyBy,stratum,unit,estimand,inference\nsteered,choiceRate,12,0.25,0.05,0.45,3,0.02,0.04,holm,injection,pooled,,,,\n",
};

/// A run read the way the app reads it: through the embedded adapter, over
/// the (stubbed) native bridge, then through the shared loaders.
const embeddedRun = async (): Promise<WorkspaceRun> => {
  const [run] = await discoverRuns(embeddedRunsDirectory() as unknown as LocalDirectoryHandle);
  return hydrateRun(run);
};

type Row = { item: string; value: number | null };
const columns: ExportColumn<Row>[] = [
  { header: "item", kind: "stored", value: (row) => row.item },
  { header: "value", kind: "derived", value: (row) => row.value },
];
const rows: Row[] = [{ item: "a", value: 1 }, { item: "b", value: null }];

describe("saving inside the app (embedded)", () => {
  it("sends a table to the native host as text, and never makes an object URL", async () => {
    const host = enterEmbedded();
    const outcome = await saveText("table.csv", "a,b\n1,2\n");
    expect(host.posted).toEqual([{ kind: "text", filename: "table.csv", text: "a,b\n1,2\n" }]);
    expect(outcome).toEqual({ state: "saved", name: "table.csv", via: "panel" });
    expect(host.createObjectURL).not.toHaveBeenCalled();
  });

  it("sends a run file as its PATH, so the host copies the bytes and the page fetches none", async () => {
    const host = enterEmbedded();
    const served = serveRuns(runFiles);
    const run = await embeddedRun();
    served.requested.length = 0;

    const outcome = await saveRunFile("generations.jsonl", run.generationFile);
    expect(host.posted).toEqual([{ kind: "runFile", filename: "generations.jsonl", path: `${RUN}/generations.jsonl` }]);
    expect(outcome.state).toBe("saved");
    // Not one byte of the file was requested in order to save it.
    expect(served.requested).toEqual([]);
    expect(host.createObjectURL).not.toHaveBeenCalled();
  });

  it("would throw if an embedded file were handed to an object URL — the defect this replaces", async () => {
    enterEmbedded();
    serveRuns(runFiles);
    const run = await embeddedRun();
    const standIn = await run.generationFile!.getFile();
    expect(standIn instanceof Blob).toBe(false);
    expect(() => URL.createObjectURL(standIn)).toThrow();
  });

  it("reports a cancelled panel, a refusal, and a silent host as three different things", async () => {
    enterEmbedded({ reply: () => ({ state: "cancelled" }) });
    expect(await saveText("a.csv", "x")).toEqual({ state: "cancelled" });
    leaveHost();

    enterEmbedded({ reply: () => ({ state: "failed", message: "The explorer does not save into the workspace's runs folder." }) });
    expect(await saveText("a.csv", "x")).toEqual({ state: "failed", message: "The explorer does not save into the workspace's runs folder." });
    leaveHost();

    enterEmbedded({ reply: () => { throw new Error("the handler went away"); } });
    expect(await saveText("a.csv", "x")).toEqual({ state: "failed", message: "the handler went away" });
    leaveHost();

    // A reply that is not one of the three known states is never read as
    // success.
    enterEmbedded({ reply: () => undefined });
    expect((await saveText("a.csv", "x")).state).toBe("failed");
    expect(readSaveReply({ state: "saved" }, "a.csv")).toEqual({ state: "saved", name: "a.csv", via: "panel" });
    expect(readSaveReply({ state: "failed" }, "a.csv").state).toBe("failed");
    expect(readSaveReply("saved", "a.csv").state).toBe("failed");
  });

  it("says so plainly when the app has no save handler, or the file's path is unknown", async () => {
    const host = enterEmbedded({ withoutSaveHandler: true });
    expect(await saveText("a.csv", "x")).toEqual({ state: "failed", message: SAVE_UNAVAILABLE });
    expect(host.createObjectURL).not.toHaveBeenCalled();
    leaveHost();

    const embedded = enterEmbedded();
    const bare: LocalFileHandle = { kind: "file", name: "x.json", getFile: async () => new File(["{}"], "x.json") };
    expect(await saveRunFile("x.json", bare)).toEqual({ state: "failed", message: SAVE_PATH_UNKNOWN });
    expect(await saveRunFile("x.json", null)).toMatchObject({ state: "failed" });
    expect(embedded.posted).toEqual([]);
  });
});

describe("saving in the standalone browser build", () => {
  it("downloads a table as a CSV blob", async () => {
    const browser = enterBrowser();
    const outcome = await saveText("table.csv", "a,b\n1,2\n");
    expect(outcome).toEqual({ state: "saved", name: "table.csv", via: "browser" });
    expect(browser.downloads).toHaveLength(1);
    expect(browser.downloads[0].filename).toBe("table.csv");
    expect(await browser.downloads[0].blob.text()).toBe("a,b\n1,2\n");
    expect(browser.downloads[0].blob.type).toBe("text/csv;charset=utf-8");
    expect(browser.revoked).toHaveLength(1);
  });

  it("downloads a run file's own bytes", async () => {
    const browser = enterBrowser();
    const handle: LocalFileHandle = { kind: "file", name: "generations.jsonl", getFile: async () => new File(["{\"a\":1}\n"], "generations.jsonl") };
    const outcome = await saveRunFile("generations.jsonl", handle);
    expect(outcome.state).toBe("saved");
    expect(await browser.downloads[0].blob.text()).toBe("{\"a\":1}\n");
  });

  it("fails with a message, rather than throwing, when a file cannot be read", async () => {
    enterBrowser();
    const broken: LocalFileHandle = { kind: "file", name: "x", getFile: async () => { throw new Error("permission was withdrawn"); } };
    expect(await saveRunFile("x", broken)).toEqual({ state: "failed", message: "permission was withdrawn" });
  });
});

describe("the Export CSV control", () => {
  it("embedded: pressing it sends the visible rows to the native host", async () => {
    const host = enterEmbedded();
    const page = render(<ExportButton filename="run-table.csv" columns={columns} rows={rows} />);
    await page.click("Export CSV");
    expect(host.posted).toHaveLength(1);
    // The file's first line is the header row: nothing comes before it.
    expect(host.posted[0]).toEqual({ kind: "text", filename: "run-table.csv", text: "item,value\na,1\nb,\n" });
    expect(host.createObjectURL).not.toHaveBeenCalled();
  });

  it("standalone: pressing it starts a browser download of the same file", async () => {
    const browser = enterBrowser();
    const page = render(<ExportButton filename="run-table.csv" columns={columns} rows={rows} />);
    await page.click("Export CSV");
    expect(browser.downloads.map((download) => download.filename)).toEqual(["run-table.csv"]);
    expect(await browser.downloads[0].blob.text()).toBe("item,value\na,1\nb,\n");
  });

  it("is disabled, not silently empty, when no rows match", () => {
    enterEmbedded();
    const page = render(<ExportButton filename="run-table.csv" columns={columns} rows={[]} />);
    expect(page.button("Export CSV").disabled).toBe(true);
  });
});

describe("the Column notes control", () => {
  it("embedded: pressing it sends the notes for the same table as a second file", async () => {
    const host = enterEmbedded();
    const page = render(<ExportButton filename="run-table.csv" columns={columns} rows={rows} />);
    await page.click("Column notes");
    expect(host.posted).toHaveLength(1);
    expect(host.posted[0]).toMatchObject({ kind: "text", filename: "run-table-columns.csv" });
    const lines = (host.posted[0] as { text: string }).text.split("\n");
    expect(lines[0]).toBe("column,kind,kindMeaning,description");
    expect(lines[1].startsWith("item,stored,")).toBe(true);
    expect(lines[2].startsWith("value,derived,")).toBe(true);
    expect(host.createObjectURL).not.toHaveBeenCalled();
  });

  it("standalone: pressing it downloads the notes file", async () => {
    const browser = enterBrowser();
    const page = render(<ExportButton filename="run-table.csv" columns={columns} rows={rows} />);
    await page.click("Column notes");
    expect(browser.downloads.map((download) => download.filename)).toEqual(["run-table-columns.csv"]);
  });

  it("stays available when no rows match: it describes columns, not rows", () => {
    enterEmbedded();
    const page = render(<ExportButton filename="run-table.csv" columns={columns} rows={[]} />);
    expect(page.button("Column notes").disabled).toBe(false);
  });
});

describe("export controls inside their views (embedded)", () => {
  it("Effects: Export CSV sends the effect table with its columns", async () => {
    const host = enterEmbedded();
    serveRuns(runFiles);
    const run = await embeddedRun();
    const page = render(<EffectsView run={run} onOpenFile={() => {}} />);
    await page.click("Export CSV");
    expect(host.posted).toHaveLength(1);
    expect(host.posted[0]).toMatchObject({ kind: "text", filename: `${RUN}-effect-sizes.csv` });
    const lines = (host.posted[0] as { text: string }).text.split("\n");
    // A plain header row first, then the row on screen.
    expect(lines[0]).toBe("condition,endpoint,stratifyBy,stratum,pairedUnit,estimand,inference,n,estimate,ciLower,ciUpper,wilcoxonP,adjustedP,correction,unit");
    expect(lines[1].startsWith("steered,choiceRate,pooled,,,,,12,0.25,0.05,0.45,0.02,0.04,holm,")).toBe(true);
    expect(lines.some((line) => line.startsWith("#"))).toBe(false);
  });

  it("Effects: Column notes describes every column of that table", async () => {
    const host = enterEmbedded();
    serveRuns(runFiles);
    const run = await embeddedRun();
    const page = render(<EffectsView run={run} onOpenFile={() => {}} />);
    await page.click("Column notes");
    expect(host.posted[0]).toMatchObject({ kind: "text", filename: `${RUN}-effect-sizes-columns.csv` });
    const notes = (host.posted[0] as { text: string }).text.split("\n").filter(Boolean).map(splitCSV);
    expect(notes[0]).toEqual(["column", "kind", "kindMeaning", "description"]);
    expect(notes.slice(1).map((row) => row[0])).toEqual(["condition", "endpoint", "stratifyBy", "stratum", "pairedUnit", "estimand", "inference", "n", "estimate", "ciLower", "ciUpper", "wilcoxonP", "adjustedP", "correction", "unit"]);
    // Every column of the table a methodologist opens says what it holds.
    for (const row of notes.slice(1)) expect(row[3]).not.toBe("");
  });

  it("Workspace triage: Export run list sends one row per run", async () => {
    const host = enterEmbedded();
    serveRuns(runFiles);
    const run = await embeddedRun();
    const page = render(<TriageView run={null} workspaceRuns={[run]} onActivateRun={() => {}} onNavigate={() => {}} onOpenFile={() => {}} />);
    await page.click("Export run list");
    expect(host.posted[0]).toMatchObject({ kind: "text", filename: "workspace-runs.csv" });
    expect((host.posted[0] as { text: string }).text).toContain(RUN);
  });
});

describe("the Download JSONL control (Generations)", () => {
  it("embedded: pressing it asks the host to copy the run's generations file", async () => {
    const host = enterEmbedded();
    const served = serveRuns(runFiles);
    const run = await embeddedRun();
    served.requested.length = 0;
    const page = render(<GenerationsView run={run} />);
    await page.click("Download JSONL");
    expect(host.posted).toEqual([{ kind: "runFile", filename: `${RUN}-generations.jsonl`, path: `${RUN}/generations.jsonl` }]);
    expect(served.requested).toEqual([]);
    expect(host.createObjectURL).not.toHaveBeenCalled();
  });

  it("standalone: pressing it downloads the file the browser handed over", async () => {
    // The rows are loaded once through the shared loaders; the file handle
    // is then the kind a browser's folder picker gives: a real File.
    enterEmbedded();
    serveRuns(runFiles);
    const loaded = await embeddedRun();
    leaveHost();
    const browser = enterBrowser();
    const text = runFiles[`${RUN}/generations.jsonl`];
    const file: LocalFileHandle = { kind: "file", name: "generations.jsonl", getFile: async () => new File([text], "generations.jsonl") };
    const page = render(<GenerationsView run={{ ...loaded, generationFile: file }} />);
    await page.click("Download JSONL");
    expect(browser.downloads.map((download) => download.filename)).toEqual([`${RUN}-generations.jsonl`]);
    expect(await browser.downloads[0].blob.text()).toBe(text);
  });
});

describe("the file preview's Download control", () => {
  it("embedded: pressing it asks the host to copy that file", async () => {
    const host = enterEmbedded();
    const served = serveRuns(runFiles);
    const run = await embeddedRun();
    const file = run.files.find((candidate) => candidate.name === "report.json")!;
    served.requested.length = 0;
    const page = render(<FilePreviewModal preview={{ file, text: "{}", truncated: false, loading: false, error: "" }} onClose={() => {}} />);
    await page.click("Download");
    expect(host.posted).toEqual([{ kind: "runFile", filename: "report.json", path: `${RUN}/report.json` }]);
    expect(served.requested).toEqual([]);
    expect(host.createObjectURL).not.toHaveBeenCalled();
  });

  it("standalone: pressing it downloads the file", async () => {
    const browser = enterBrowser();
    const handle: LocalFileHandle = { kind: "file", name: "report.json", getFile: async () => new File(["{}"], "report.json") };
    const page = render(<FilePreviewModal preview={{ file: { name: "report.json", path: "report.json", size: 2, modified: 0, handle }, text: "{}", truncated: false, loading: false, error: "" }} onClose={() => {}} />);
    await page.click("Download");
    expect(browser.downloads.map((download) => download.filename)).toEqual(["report.json"]);
  });
});

describe("what a control says afterwards", () => {
  it("names the saved file, explains a failure, and says nothing about a cancel", () => {
    expect(saveStatusText({ state: "saved", name: "a.csv", via: "panel" })).toBe("Saved as a.csv");
    expect(saveStatusText({ state: "saved", name: "a.csv", via: "browser" })).toBe("Sent to your browser's downloads as a.csv");
    expect(saveStatusText({ state: "failed", message: "Choose a location outside the runs folder." })).toBe("Not saved. Choose a location outside the runs folder.");
    expect(saveStatusText({ state: "cancelled" })).toBe("");
    expect(saveStatusText(null)).toBe("");
    expect(render(<SaveStatus status={{ state: "failed", message: "No room on the disk." }} />).text).toBe("Not saved. No room on the disk.");
    expect(render(<SaveStatus status={{ state: "cancelled" }} />).html).toBe("");
  });
});

describe("one save path", () => {
  // The defect was two controls each rolling their own download. Every
  // control now goes through lib/save.ts; this holds the line.
  const appRoot = fileURLToPath(new URL("../app", import.meta.url));
  const sources = (directory: string): string[] => readdirSync(directory).flatMap((name) => {
    const path = join(directory, name);
    return statSync(path).isDirectory() ? sources(path) : /\.tsx?$/.test(name) ? [path] : [];
  });

  it("no module but lib/save.ts makes an object URL or a download anchor", () => {
    const offenders = sources(appRoot)
      .filter((path) => relative(appRoot, path) !== join("lib", "save.ts"))
      .filter((path) => {
        const code = readFileSync(path, "utf8").split("\n").filter((line) => !line.trim().startsWith("//")).join("\n");
        return /createObjectURL|\.download\s*=/.test(code);
      })
      .map((path) => relative(appRoot, path));
    expect(offenders).toEqual([]);
  });
});
