import { describe, expect, it } from "vitest";
import { deriveStatus, failureNoteError, statusLabel } from "../app/lib/status";
import { runTimestampKey, sortRunsByTimestamp } from "../app/lib/discovery";

// Every fixture below is SYNTHESIZED to the engines' documented shapes
// (run_status.py / RunStatusFile.swift). No workspace bytes are copied here.

const sources = (over: Partial<Parameters<typeof deriveStatus>[0]> = {}) => ({
  statusText: null,
  failedText: null,
  cancelledText: null,
  artifacts: [] as string[],
  ...over,
});

describe("deriveStatus — completed", () => {
  it("reads a completed stage and its counts", () => {
    const info = deriveStatus(sources({
      statusText: JSON.stringify({
        schemaVersion: 1, stage: "run", status: "completed", startedAt: 100, finishedAt: 200,
        evidenceComplete: true, itemLabel: "record", itemsWritten: 192, invalidResponses: 0,
        experiment: "test-compare",
      }),
      artifacts: ["report.json", "generations.jsonl", "run-status.json"],
    }));
    expect(info.state).toBe("completed");
    expect(info.stage).toBe("run");
    expect(info.itemsWritten).toBe(192);
    expect(info.itemLabel).toBe("record");
    expect(info.invalidResponses).toBe(0);
    expect(info.experiment).toBe("test-compare");
    expect(info.stamped).toBe(true);
    expect(info.error).toBe("");
  });

  it("refuses to call a completed stage complete when evidenceComplete is false", () => {
    const info = deriveStatus(sources({
      statusText: JSON.stringify({ status: "completed", evidenceComplete: false }),
      artifacts: ["run-status.json"],
    }));
    expect(info.state).toBe("partial");
  });
});

describe("deriveStatus — failed", () => {
  const statusText = JSON.stringify({
    schemaVersion: 1, stage: "evaluate", status: "failed",
    error: "OpenRouter judge response carried no content", errorType: "RuntimeError",
    evidenceComplete: false, itemLabel: "item", itemsWritten: 0, invalidResponses: 0,
    experiment: "test-compare-2-2", sourceRun: "20260805T001709118-exp-test-compare-2-2-run",
    expectedUnits: ["judge-ds", "judge-tm"], completedUnits: [],
    pendingUnits: ["judge-ds", "judge-tm"], startedAt: 1, finishedAt: 2,
  });

  it("reads the failure, what did not run, and the chain edge", () => {
    const info = deriveStatus(sources({ statusText, artifacts: ["run-status.json", "FAILED.md", "judgments.jsonl"] }));
    expect(info.state).toBe("failed");
    expect(info.error).toBe("OpenRouter judge response carried no content");
    expect(info.errorType).toBe("RuntimeError");
    expect(info.pendingUnits).toEqual(["judge-ds", "judge-tm"]);
    expect(info.completedUnits).toEqual([]);
    expect(info.expectedUnits).toEqual(["judge-ds", "judge-tm"]);
    expect(info.sourceRun).toBe("20260805T001709118-exp-test-compare-2-2-run");
    expect(info.itemsWritten).toBe(0);
    expect(statusLabel(info)).toBe("failed");
  });

  it("falls back to the FAILED.md error line when the status file carries none", () => {
    const failedText = [
      "# evaluate FAILED",
      "",
      "- **Stage:** evaluate",
      "- **Error:** `RuntimeError: judge response carried no content`",
      "- **Items written before the failure:** 0",
    ].join("\n");
    const info = deriveStatus(sources({
      statusText: JSON.stringify({ status: "failed" }),
      failedText,
      artifacts: ["run-status.json", "FAILED.md"],
    }));
    expect(info.state).toBe("failed");
    expect(info.error).toBe("RuntimeError: judge response carried no content");
  });
});

describe("failureNoteError", () => {
  it("unwraps the backticked error line", () => {
    expect(failureNoteError("- **Error:** `ValueError: bad pin`")).toBe("ValueError: bad pin");
  });

  it("accepts an unbackticked line", () => {
    expect(failureNoteError("* **Error:** plain message")).toBe("plain message");
  });

  it("invents nothing when the note has no error line", () => {
    expect(failureNoteError("# run FAILED\n\nsomething went wrong")).toBe("");
    expect(failureNoteError(null)).toBe("");
  });
});

describe("deriveStatus — cancelled", () => {
  it("reads Swift's cancelled.txt marker even though the status says failed", () => {
    const info = deriveStatus(sources({
      statusText: JSON.stringify({ status: "failed", error: "cancelled by user — this run stopped", errorType: "Cancelled" }),
      artifacts: ["run-status.json", "cancelled.txt"],
      cancelledText: "cancelled by user — this run stopped between units of work",
    }));
    expect(info.state).toBe("cancelled");
    expect(statusLabel(info)).toBe("cancelled");
  });

  it("reads a cancellation note left beside no status file at all", () => {
    const info = deriveStatus(sources({
      cancelledText: "cancelled by user — this sweep stopped between units of work",
      artifacts: ["cancelled.txt", "sweep-progress.jsonl"],
    }));
    expect(info.state).toBe("cancelled");
    expect(info.error).toContain("cancelled by user");
    expect(info.stamped).toBe(false);
  });

  it("reads the errorType alone as a cancellation (the server writes no note)", () => {
    const info = deriveStatus(sources({
      statusText: JSON.stringify({ status: "failed", errorType: "Cancelled", error: "stopped" }),
      artifacts: ["run-status.json"],
    }));
    expect(info.state).toBe("cancelled");
  });
});

describe("deriveStatus — in progress, checkpointed, torn", () => {
  it("reads inProgress", () => {
    const info = deriveStatus(sources({
      statusText: JSON.stringify({ status: "inProgress", itemsWritten: 12, itemLabel: "judgment" }),
      artifacts: ["run-status.json"],
    }));
    expect(info.state).toBe("inProgress");
    expect(statusLabel(info)).toBe("in progress");
    expect(info.itemsWritten).toBe(12);
  });

  it("reads a checkpointed stage as partial, not failed", () => {
    const info = deriveStatus(sources({
      statusText: JSON.stringify({ status: "checkpointed", errorType: "CheckpointRequested", error: "requeued" }),
      artifacts: ["run-status.json"],
    }));
    expect(info.state).toBe("partial");
  });

  it("fails closed on a torn status file", () => {
    const info = deriveStatus(sources({ statusText: "{\"status\": \"comp", artifacts: ["run-status.json"] }));
    expect(info.state).toBe("partial");
    expect(info.stamped).toBe(true);
  });
});

describe("deriveStatus — absence", () => {
  it("reads a legacy directory's summary artifact as the engines' own completion record", () => {
    // Merge review 2026-08-05: report.json is written exclusively after the
    // last unit finishes ("a partial panel is never summarized as a
    // report"), so its presence IS a stored completion fact — attributed in
    // the label so it is never mistaken for a stamped status file.
    const info = deriveStatus(sources({ artifacts: ["report.json", "generations.jsonl", "config.json"] }));
    expect(info.state).toBe("completed");
    expect(info.stamped).toBe(false);
    expect(info.completionSource).toBe("summaryArtifact");
    expect(statusLabel(info)).toBe("completed (summary artifact)");
  });

  it("reads rows with no completion artifact as partial", () => {
    expect(deriveStatus(sources({ artifacts: ["generations.jsonl", "config.json"] })).state).toBe("partial");
    expect(deriveStatus(sources({ artifacts: ["judgments.jsonl"] })).state).toBe("partial");
    expect(deriveStatus(sources({ artifacts: ["codings.jsonl"] })).state).toBe("partial");
  });

  it("stamps nothing about counts it was not given", () => {
    const info = deriveStatus(sources({ artifacts: ["report.json"] }));
    expect(info.itemsWritten).toBeNull();
    expect(info.invalidResponses).toBeNull();
    expect(info.evidenceComplete).toBeNull();
  });
});

describe("run ordering by directory-name timestamp", () => {
  it("extracts the YYYYMMDDTHHMMSS prefix, with or without milliseconds", () => {
    expect(runTimestampKey("20260805T004016927-exp-test-evaluate")).toBe("20260805T004016927");
    expect(runTimestampKey("20260722T191052-exp-test-validate")).toBe("20260722T191052");
    expect(runTimestampKey("model-variants")).toBe("");
  });

  it("orders newest first ACROSS A MONTH BOUNDARY, where the date label sorted alphabetically", () => {
    // "Aug 5" vs "Dec 1" vs "Apr 2": localeCompare on the formatted label put
    // December before August before April. The timestamp prefix does not.
    const runs = [
      { name: "20260402T090000000-exp-a-run", dateLabel: "Apr 2, 2026, 9:00 AM" },
      { name: "20261201T090000000-exp-c-run", dateLabel: "Dec 1, 2026, 9:00 AM" },
      { name: "20260805T090000000-exp-b-run", dateLabel: "Aug 5, 2026, 9:00 AM" },
    ];
    expect(sortRunsByTimestamp(runs).map((run) => run.name)).toEqual([
      "20261201T090000000-exp-c-run",
      "20260805T090000000-exp-b-run",
      "20260402T090000000-exp-a-run",
    ]);
  });

  it("orders same-day runs by their millisecond stamp", () => {
    const runs = [
      { name: "20260805T001709362-exp-x-pipeline" },
      { name: "20260805T004016927-exp-x-evaluate" },
      { name: "20260805T001709118-exp-x-run" },
    ];
    expect(sortRunsByTimestamp(runs).map((run) => run.name)).toEqual([
      "20260805T004016927-exp-x-evaluate",
      "20260805T001709362-exp-x-pipeline",
      "20260805T001709118-exp-x-run",
    ]);
  });

  it("sorts un-timestamped directories last rather than to the top", () => {
    const runs = [{ name: "neutral-pcs" }, { name: "20260805T001709118-exp-x-run" }];
    expect(sortRunsByTimestamp(runs).map((run) => run.name)).toEqual([
      "20260805T001709118-exp-x-run",
      "neutral-pcs",
    ]);
  });

  it("prefers a precomputed timestampKey when discovery attached one", () => {
    const runs = [
      { name: "b", timestampKey: "20260101T000000000" },
      { name: "a", timestampKey: "20260901T000000000" },
    ];
    expect(sortRunsByTimestamp(runs).map((run) => run.name)).toEqual(["a", "b"]);
  });
});

describe("summary-artifact completion (merge review 2026-08-05)", () => {
  const base = { statusText: null, failedText: null, cancelledText: null };

  it("a summary artifact with no status file reads completed, attributed", () => {
    const info = deriveStatus({ ...base, artifacts: ["report.json", "generations.jsonl", "config.json"] });
    expect(info.state).toBe("completed");
    expect(info.completionSource).toBe("summaryArtifact");
    expect(statusLabel(info)).toBe("completed (summary artifact)");
  });

  it("rows without their summary still outrank the reading", () => {
    const info = deriveStatus({ ...base, artifacts: ["generations.jsonl", "config.json"] });
    expect(info.state).toBe("partial");
  });

  it("a lone FAILED.md is a failure record, never not-stamped", () => {
    const info = deriveStatus({
      ...base,
      failedText: "# run FAILED\n\n- **Error:** `RuntimeError: boom`\n",
      artifacts: ["report.json"],
    });
    expect(info.state).toBe("failed");
    expect(info.error).toBe("RuntimeError: boom");
  });

  it("no artifacts at all stays not stamped", () => {
    const info = deriveStatus({ ...base, artifacts: ["config.json"] });
    expect(info.state).toBe("not stamped");
  });

  it("a status file's completed reading is attributed to the file", () => {
    const info = deriveStatus({
      ...base,
      statusText: JSON.stringify({ stage: "evaluate", status: "completed", evidenceComplete: true }),
      artifacts: ["judge-report.json"],
    });
    expect(info.completionSource).toBe("statusFile");
    expect(statusLabel(info)).toBe("completed");
  });
});
