import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  countRecords, loadSourceSnapshot, snapshotSentence, snapshotVerdict,
  type SnapshotDescription, type SnapshotStamp,
} from "../app/lib/sourceSnapshot";
import type { RunFile, WorkspaceRun } from "../app/lib/types";

// Which records of its run an analysis read. The cases are the SHARED
// fixture the Python engine (Server/tests/test_source_snapshot.py) and the
// Mac engine (SourceSnapshotTests.swift) read too.

const checkout = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..");
const fixture = JSON.parse(readFileSync(resolve(checkout, "Tests", "Fixtures", "cross-engine", "source-snapshot.json"), "utf8")) as {
  describe: { label: string; generations: string; expected: SnapshotDescription }[];
  verdicts: { label: string; stamp: SnapshotStamp | null; now: SnapshotDescription | null; continued: boolean; expected: { state: string; sentence: string | null } }[];
};

const file = (path: string, text: string): RunFile => {
  const blob = new Blob([text]);
  return { name: path, path, size: blob.size, modified: 0, handle: { getFile: async () => blob } } as unknown as RunFile;
};

const run = (name: string, files: RunFile[]): WorkspaceRun => ({ name, files } as unknown as WorkspaceRun);

const RUN = "20261004T000000000-exp-tone-run";
const lines = (count: number) => Array.from({ length: count }, (_, index) => `{"condition":"c${index % 2}","promptID":"item-${Math.floor(index / 2) + 1}"}`).join("\n") + "\n";

describe("the shared contract", () => {
  it.each(fixture.verdicts)("$label", (item) => {
    const found = snapshotVerdict(item.stamp, item.now, item.continued);
    expect(found.state).toBe(item.expected.state);
    expect(snapshotSentence(found, "R")).toBe(item.expected.sentence);
  });

  it.each(fixture.describe)("counts records: $label", (item) => {
    expect(countRecords(item.generations)).toBe(item.expected.records);
  });
});

describe("an analysis beside its run", () => {
  const checkpointStamp = (records: number, bytes: number) => JSON.stringify({
    schemaVersion: 1, sourceRun: RUN, status: "checkpointed", records, generationsBytes: bytes, generationsSHA256: "a".repeat(64),
  });

  it("says a checkpoint's analysis is not the completed run's", async () => {
    const early = lines(4);
    const source = run(RUN, [file("generations.jsonl", lines(8)), file("report.json", "{}"),
      file("run-status.json", JSON.stringify({ status: "completed", resumedFrom: { status: "checkpointed" } }))]);
    const analysis = run("20261004T010000000-exp-tone-analyze", [file("source-snapshot.json", checkpointStamp(4, new Blob([early]).size))]);
    const reading = await loadSourceSnapshot(analysis, source);
    expect(reading?.verdict.state).toBe("unfinishedSource");
    expect(reading?.sentence).toBe(`This analysis read run ${RUN} while it was paused at a checkpoint, with 4 records, and the run now holds 8. Re-analyze the run for effects over all its records.`);
  });

  it("says nothing for an analysis of the run as it stands", async () => {
    const records = lines(8);
    const source = run(RUN, [file("generations.jsonl", records), file("report.json", "{}")]);
    const stamp = JSON.stringify({ schemaVersion: 1, sourceRun: RUN, status: "completed", records: 8, generationsBytes: new Blob([records]).size, generationsSHA256: "b".repeat(64) });
    const reading = await loadSourceSnapshot(run("a", [file("source-snapshot.json", stamp)]), source);
    expect(reading?.verdict.state).toBe("matches");
    expect(reading?.sentence).toBeNull();
  });

  it("flags a completed run whose records grew after the analysis", async () => {
    const source = run(RUN, [file("generations.jsonl", lines(9)), file("report.json", "{}")]);
    const stamp = JSON.stringify({ schemaVersion: 1, sourceRun: RUN, status: "completed", records: 8, generationsBytes: new Blob([lines(8)]).size, generationsSHA256: "b".repeat(64) });
    const reading = await loadSourceSnapshot(run("a", [file("source-snapshot.json", stamp)]), source);
    expect(reading?.verdict.state).toBe("changed");
    expect(reading?.sentence).toContain("it now holds 9");
  });

  it("flags a legacy analysis only when its run was paused and continued", async () => {
    const continued = run(RUN, [file("generations.jsonl", lines(8)), file("report.json", "{}"),
      file("run-status.json", JSON.stringify({ status: "checkpointed" }))]);
    expect((await loadSourceSnapshot(run("a", []), continued))?.verdict.state).toBe("notRecordedRunContinued");
    const plain = run(RUN, [file("generations.jsonl", lines(8)), file("report.json", "{}")]);
    expect((await loadSourceSnapshot(run("a", []), plain))?.sentence).toBeNull();
  });

  it("speaks from the stamp alone when the run is not in the workspace", async () => {
    const reading = await loadSourceSnapshot(run("a", [file("source-snapshot.json", checkpointStamp(4, 200))]), null);
    expect(reading?.sentence).toBe(`This analysis read run ${RUN} while it was paused at a checkpoint, with 4 records. Re-analyze the run for effects over all its records.`);
  });

  it("has nothing to say for a run that is not an analysis", async () => {
    expect(await loadSourceSnapshot(run(RUN, [file("generations.jsonl", lines(2))]), null)).toBeNull();
  });
});
