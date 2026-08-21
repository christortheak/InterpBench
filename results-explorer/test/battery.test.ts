import { describe, expect, it } from "vitest";
import {
  batteryCounts,
  loadBatteryRecords,
  parseBatteryRecord,
  readBatterySummaries,
} from "../app/lib/battery";
import type { LocalFileHandle, RunFile } from "../app/lib/types";

// In-memory stand-ins for the handle surface (same pattern as
// test/loaders.test.ts): nothing here touches a real workspace. The record
// shapes mirror both engines — the server's `answer` / `promptID` /
// `batteryHash` dialect and Swift's `expected`.

const fakeFile = (name: string, text: string): File => {
  const bytes = new TextEncoder().encode(text);
  return {
    name, size: bytes.byteLength, lastModified: 0, type: "",
    text: async () => text,
    arrayBuffer: async () => bytes.buffer,
    slice: (start?: number, end?: number) => ({ text: async () => text.slice(start ?? 0, end) }),
  } as unknown as File;
};

const fakeFileHandle = (name: string, text: string): LocalFileHandle => ({
  kind: "file", name, getFile: async () => fakeFile(name, text),
});

const fakeRunFile = (name: string, text: string): RunFile => ({
  name, path: name, size: new TextEncoder().encode(text).byteLength, modified: 0, handle: fakeFileHandle(name, text),
});

const files = (contents: Record<string, string>) => ({
  files: Object.entries(contents).map(([name, text]) => fakeRunFile(name, text)),
});

const serverRow = (fields: Record<string, unknown>) => JSON.stringify({
  condition: "baseline", promptIndex: 0, promptID: "battery-0", sampleIndex: 0,
  prompt: "What is 17 + 26? Answer with just the number.", answer: "43",
  output: "43\n", batteryHash: "cb065012a3ace949", correct: true, ...fields,
});

describe("parseBatteryRecord", () => {
  it("reads the server's dialect", () => {
    const row = parseBatteryRecord(JSON.parse(serverRow({})), 1);
    expect(row).not.toBeNull();
    expect(row?.condition).toBe("baseline");
    expect(row?.promptID).toBe("battery-0");
    expect(row?.promptIndex).toBe(0);
    expect(row?.expected).toBe("43");
    expect(row?.correct).toBe(true);
    expect(row?.batteryHash).toBe("cb065012a3ace949");
    expect(row?.line).toBe(1);
  });

  it("reads Swift's `expected` and synthesises the promptID Swift omits", () => {
    const row = parseBatteryRecord({ condition: "fear-agent", promptIndex: 4, prompt: "p", expected: "0.7", output: "0.7", correct: false }, 9);
    expect(row?.expected).toBe("0.7");
    expect(row?.promptID).toBe("battery-4");
    expect(row?.correct).toBe(false);
    expect(row?.batteryHash).toBe("");
  });

  it("keeps an unstamped grade NULL — an ungraded item is not a wrong one", () => {
    const row = parseBatteryRecord({ condition: "baseline", promptIndex: 1, prompt: "p", output: "o" }, 2);
    expect(row?.correct).toBeNull();
  });

  it("refuses a row with no condition or no output rather than counting it", () => {
    expect(parseBatteryRecord({ promptIndex: 0, prompt: "p", output: "o" }, 1)).toBeNull();
    expect(parseBatteryRecord({ condition: "baseline", prompt: "p" }, 1)).toBeNull();
    expect(parseBatteryRecord("not an object", 1)).toBeNull();
  });
});

describe("loadBatteryRecords", () => {
  const jsonl = [
    serverRow({}),
    serverRow({ promptIndex: 1, promptID: "battery-1", answer: "paris", output: "Paris", correct: true }),
    "{not json",
    serverRow({ condition: "fear-agent", promptIndex: 0, output: "44", correct: false }),
    "",
  ].join("\n");

  it("reads every parseable record and counts the rest", async () => {
    const load = await loadBatteryRecords(files({ "battery.jsonl": jsonl }));
    expect(load.present).toBe(true);
    expect(load.records).toHaveLength(3);
    expect(load.skipped).toBe(1);
    expect(load.truncated).toBe(false);
    expect(load.records.map((row) => row.line)).toEqual([1, 2, 4]);
  });

  it("says so when the run has no battery.jsonl, rather than returning an empty run", async () => {
    const load = await loadBatteryRecords(files({}));
    expect(load.present).toBe(false);
    expect(load.file).toBeNull();
    expect(load.records).toEqual([]);
  });
});

describe("readBatterySummaries", () => {
  const report = {
    conditions: {
      "fear agent 7-21": { generations: 30, capabilityBattery: { accuracy: 0.9, itemCount: 10, batteryHash: "cb06" } },
      baseline: { generations: 30, capabilityBattery: { accuracy: 1, itemCount: 10, batteryHash: "cb06" } },
      "no-battery-arm": { generations: 30 },
    },
  };

  it("reads the stored per-condition rollup, baseline first", () => {
    const rows = readBatterySummaries(report);
    expect(rows.map((row) => row.condition)).toEqual(["baseline", "fear agent 7-21"]);
    expect(rows[0].accuracy).toBe(1);
    expect(rows[1].accuracy).toBe(0.9);
    expect(rows[1].itemCount).toBe(10);
  });

  it("omits a condition with no capabilityBattery block instead of giving it a zero", () => {
    expect(readBatterySummaries(report).some((row) => row.condition === "no-battery-arm")).toBe(false);
  });

  it("keeps a present-but-unstamped accuracy null", () => {
    const rows = readBatterySummaries({ conditions: { baseline: { capabilityBattery: { itemCount: 10 } } } });
    expect(rows[0].accuracy).toBeNull();
    expect(rows[0].itemCount).toBe(10);
  });

  it("returns nothing for a report with no conditions block", () => {
    expect(readBatterySummaries({})).toEqual([]);
    expect(readBatterySummaries({ conditions: "not an object" })).toEqual([]);
  });
});

describe("batteryCounts", () => {
  it("splits correct, incorrect and ungraded, baseline first", () => {
    const records = [
      parseBatteryRecord({ condition: "fear-agent", promptIndex: 0, output: "x", correct: false }, 1)!,
      parseBatteryRecord({ condition: "baseline", promptIndex: 0, output: "x", correct: true }, 2)!,
      parseBatteryRecord({ condition: "baseline", promptIndex: 1, output: "x" }, 3)!,
      parseBatteryRecord({ condition: "baseline", promptIndex: 2, output: "x", correct: true }, 4)!,
    ];
    const counts = batteryCounts(records);
    expect(counts.map((row) => row.condition)).toEqual(["baseline", "fear-agent"]);
    expect(counts[0]).toMatchObject({ records: 3, correct: 2, incorrect: 0, ungraded: 1 });
    expect(counts[1]).toMatchObject({ records: 1, correct: 0, incorrect: 1, ungraded: 0 });
  });

  it("counts nothing from nothing", () => {
    expect(batteryCounts([])).toEqual([]);
  });
});
