import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("react/jsx-dev-runtime", async (original) => (await import("./support/capture")).wrapJSXRuntime(await original()));
vi.mock("react/jsx-runtime", async (original) => (await import("./support/capture")).wrapJSXRuntime(await original()));

import { embeddedRunsDirectory } from "../app/embedded-workspace";
import {
  adjudicationSentence, analysisStampsOf, emptyAnalysisStamps, exclusionsSentence, hasAnalysisStamps,
  parseAnalysisStamps, rescueSentence, unitSentence,
} from "../app/lib/analysisStamps";
import { splitCSV } from "../app/lib/csv";
import { discoverRuns } from "../app/lib/discovery";
import { pairedCountLabel } from "../app/lib/effects";
import { effectUnit } from "../app/lib/effectUnits";
import { findSourceRun } from "../app/lib/judged";
import { hydrateRun } from "../app/lib/loaders";
import type { LocalDirectoryHandle, WorkspaceRun } from "../app/lib/types";
import { EffectsView } from "../app/views/Effects";
import { LocalOverview } from "../app/views/Overview";
import { render } from "./support/capture";
import { enterEmbedded, leaveHost, serveRuns } from "./support/host";

// The stamps an analysis writes beside its effect table, and the one label
// they correct: a pooled row over TRANSCRIPTS was captioned "paired items".
// The shapes below are the engines' own (analysis_workflow.py writes each
// file; the Mac engine stamps the unit in report.json and embeds exclusions
// in analysis.json).

afterEach(leaveHost);

const exclusions = {
  rules: [
    { rule: "failedAttentionCheck", checkedItems: 2, description: "Drop a record whose attention-check item was answered wrongly." },
    { rule: "unparseableEndpoint", endpoint: "parsedMonths", description: "Drop a record whose numeric answer could not be read." },
  ],
  consideredN: { baseline: 64, steered: 64 },
  excludedByRule: { baseline: { failedAttentionCheck: 3, unparseableEndpoint: 1 }, steered: { failedAttentionCheck: 20, unparseableEndpoint: 4 } },
  excludedRecords: 28,
  survivingN: { baseline: 60, steered: 40 },
  pairwiseDeletion: true,
  scope: "allRecordTypes",
  note: "Excluded records are dropped from the paired statistics only.",
};
const reparse = { endpoint: "parsedMonths", parser: { name: "builtin:sentencing", kind: "durationMonths" }, unparsedRecords: 9, rescuedRecords: 6, stillUnparsed: 3, note: "Null-only endpoint rescue." };
const adjudication = {
  endpoint: "parsedMonths", sourceRun: "20260801T000000000-exp-s-run", fileSha256: "f".repeat(64),
  counts: { agree: 40, differ: 5, rescuedFromNull: 2, nulledFromValue: 1, unadjudicatable: 0, total: 48 },
  meanAbsDiff: 3.5, maxAbsDiff: 12, byCondition: {}, note: "The five divergence classes partition the adjudicated rows exactly.",
};
const transcriptUnit = { unitOfAnalysis: "transcript", reason: "turns within a transcript are dependent; each transcript is reduced to its mean paired difference before testing", skippedForSingleTranscript: true };

describe("parseAnalysisStamps", () => {
  it("records nothing for a run that carries no stamp files", () => {
    const stamps = parseAnalysisStamps({});
    expect(stamps).toEqual(emptyAnalysisStamps());
    expect(hasAnalysisStamps(stamps)).toBe(false);
  });

  it("reads exclusions: rules, how many records were dropped, and how many are left per condition", () => {
    const stamp = parseAnalysisStamps({ exclusions }).exclusions!;
    expect(stamp.rules.map((rule) => rule.rule)).toEqual(["failedAttentionCheck", "unparseableEndpoint"]);
    expect(stamp.excludedRecords).toBe(28);
    expect(stamp.survivingN).toEqual({ baseline: 60, steered: 40 });
    expect(stamp.consideredN).toEqual({ baseline: 64, steered: 64 });
    expect(stamp.source).toBe("exclusions.json");
    expect(exclusionsSentence(stamp)).toBe("28 records were left out of the paired statistics by 2 declared rules. They are still in generations.jsonl.");
  });

  it("reads the exclusions the Mac engine embeds in analysis.json when there is no separate file", () => {
    const stamp = parseAnalysisStamps({ analysis: { exclusions } }).exclusions!;
    expect(stamp.source).toBe("analysis.json");
    expect(stamp.excludedRecords).toBe(28);
  });

  it("reads endpoint rescue: how many answers were read again, rescued, and left unread", () => {
    const stamp = parseAnalysisStamps({ reparse }).rescue!;
    expect(stamp).toMatchObject({ endpoint: "parsedMonths", parser: "builtin:sentencing", unparsedRecords: 9, rescuedRecords: 6, stillUnparsed: 3 });
    expect(rescueSentence(stamp)).toBe("9 answers that the run could not read were read again with builtin:sentencing: 6 now have a value, and 3 still do not. Answers the run had already read were left as they were.");
    // The engine writes the stamp even when there was nothing to rescue.
    expect(rescueSentence({ ...stamp, unparsedRecords: 0, rescuedRecords: 0, stillUnparsed: 0 })).toContain("Nothing to rescue");
  });

  it("reads adjudication from its own file, and from config.json's note when only that is present", () => {
    const full = parseAnalysisStamps({ adjudication }).adjudication!;
    expect(full.counts.total).toBe(48);
    expect(full.source).toBe("adjudicated-endpoint.json");
    expect(adjudicationSentence(full)).toBe("48 answers were replaced by values from an outside adjudication of parsedMonths: 40 agreed with the run's value, 5 differed from it, 2 gave a value where the run had none, 1 removed a value the run had, 0 had no value before or after. The effects below use the adjudicated values.");

    const noted = parseAnalysisStamps({ config: { notes: { adjudicatedEndpoint: { fileSha256: "a".repeat(64), divergence: { endpoint: "parsedMonths", counts: { total: 3, differ: 1 }, meanAbsDiff: 2, maxAbsDiff: 2 } } } } }).adjudication!;
    expect(noted.source).toBe("config.json");
    expect(noted.counts).toEqual({ total: 3, differ: 1 });
  });

  it("reads the unit of analysis from the server's file and from the Mac engine's report", () => {
    const server = parseAnalysisStamps({ unit: transcriptUnit }).unit!;
    expect(server).toMatchObject({ unit: "transcript", skippedForSingleTranscript: true, source: "unit-of-analysis.json" });
    const mac = parseAnalysisStamps({ report: { unitOfAnalysis: "transcript", transcriptsPerCondition: 4 } }).unit!;
    expect(mac).toMatchObject({ unit: "transcript", transcriptsPerCondition: 4, source: "report.json" });
    expect(unitSentence(mac)).toContain("n counts transcripts, not turns");
    expect(unitSentence(mac)).toContain("This run has 4 transcripts for each condition.");
    expect(unitSentence(server)).toContain("only one transcript");
  });

  it("says in words what an absent stamp means, rather than leaving a blank", () => {
    expect(unitSentence(null)).toBe("Not stamped in this run.");
    expect(exclusionsSentence(null)).toContain("None recorded");
    expect(rescueSentence(null)).toContain("None recorded");
    expect(adjudicationSentence(null)).toContain("None recorded");
  });

  it("reads an unverified study fingerprint and tolerated measurement drift", () => {
    const stamps = parseAnalysisStamps({ epochUnverified: { epochUnverified: true }, measurementDrift: { measurementDrift: { maxTokens: [256, 512] } } });
    expect(stamps.epochUnverified).toBe(true);
    expect(stamps.measurementDrift).toBe("{\"maxTokens\":[256,512]}");
    expect(parseAnalysisStamps({ config: { notes: { epochUnverified: true } } }).epochUnverified).toBe(true);
  });
});

describe("pairedCountLabel", () => {
  const row = { condition: "steered", endpoint: "e", short: "e", estimate: 1, low: 0, high: 2, unit: "Δ units", n: 4, q: null, p: null, correction: "", direction: "positive" as const, stratifyBy: "pooled", stratum: "", pairedUnit: "", estimand: "", inference: "", key: "k" };

  // The row's unit as the loader settles it (lib/effectUnits.ts): the row's
  // own column first, then the run's stamp, then the run's records.
  const settledAs = (pairedUnit: string, analysisUnit: string, items: number | null = 4) =>
    ({ ...row, pairedUnit, analysisUnit, effectUnit: effectUnit(row.condition, row.n, pairedUnit, analysisUnit, items === null ? null : { counts: { steered: items }, complete: true }) });

  it("counts transcripts as transcripts, never as paired items", () => {
    expect(pairedCountLabel(settledAs("", "transcript"))).toBe("n = 4 paired transcripts");
    expect(pairedCountLabel({ ...settledAs("", "transcript"), n: 1 })).toBe("n = 1 paired transcript");
  });

  it("keeps paired items for an item-level run, stamped or settled from its records", () => {
    expect(pairedCountLabel(settledAs("", ""))).toBe("n = 4 paired items");
    expect(pairedCountLabel(settledAs("", "item"))).toBe("n = 4 paired items");
  });

  it("lets a stratified row's own unit win, and names a unit it does not know", () => {
    expect(pairedCountLabel(settledAs("sample", "transcript"))).toBe("n = 4 paired samples");
    expect(pairedCountLabel(settledAs("", "session"))).toBe("n = 4 (unit of analysis: session)");
    expect(pairedCountLabel({ ...settledAs("", "transcript"), n: null })).toBe("n not reported");
  });
});

const RUN = "20260803T101500000-exp-panel-analyze";
const effectCSV = "condition,endpoint,n,deltaMean,ciLower,ciUpper,wilcoxonW,wilcoxonP,adjustedP,correction,modality,stratifyBy,stratum,unit,estimand,inference\nsteered,wordCount,4,12.5,2.0,23.0,0,0.125,0.125,bh,injection,pooled,,,,\n";

/// A run's records: `items` items, each answered `seeds` times under the
/// baseline and the steered condition.
const recordsOf = (items: number, seeds: number) => ["baseline", "steered"].flatMap((condition) =>
  Array.from({ length: items }, (_, item) => Array.from({ length: seeds }, (_, seed) =>
    JSON.stringify({ condition, promptID: `item-${item + 1}`, seed, output: `An answer to item ${item + 1}.` }))).flat()).join("\n") + "\n";

const load = async (extra: Record<string, unknown>, report: Record<string, unknown> = {}, generations?: string): Promise<WorkspaceRun> => {
  enterEmbedded();
  serveRuns({
    [`${RUN}/report.json`]: JSON.stringify({ experiment: "panel", conditions: { baseline: { generations: 8 }, steered: { generations: 8 } }, ...report }),
    [`${RUN}/config.json`]: JSON.stringify({ modelID: "org/model", runType: "analyze" }),
    [`${RUN}/effect-sizes.csv`]: effectCSV,
    ...(generations === undefined ? {} : { [`${RUN}/generations.jsonl`]: generations }),
    ...Object.fromEntries(Object.entries(extra).map(([name, value]) => [`${RUN}/${name}`, JSON.stringify(value)])),
  });
  const [run] = await discoverRuns(embeddedRunsDirectory() as unknown as LocalDirectoryHandle);
  return hydrateRun(run);
};

describe("the stamps beside the effects", () => {
  it("a transcript-unit run labels its count in transcripts, on the effects page and the overview", async () => {
    const run = await load({ "unit-of-analysis.json": transcriptUnit });
    expect(analysisStampsOf(run).unit?.unit).toBe("transcript");
    expect(run.effectRows[0].analysisUnit).toBe("transcript");

    const effects = render(<EffectsView run={run} onOpenFile={() => {}} />);
    expect(effects.text).toContain("n = 4 paired transcripts");
    expect(effects.text).not.toContain("paired items");
    expect(effects.text).toContain("Each transcript—not each turn—is the unit of analysis.");

    const overview = render(<LocalOverview run={run} onNavigate={() => {}} />);
    expect(overview.text).toContain("n = 4 paired transcripts");
    expect(overview.text).not.toContain("n = 4 paired items");
  });

  it("the Mac engine's stamp in report.json has the same effect", async () => {
    const run = await load({}, { unitOfAnalysis: "transcript", transcriptsPerCondition: 4 });
    expect(render(<EffectsView run={run} onOpenFile={() => {}} />).text).toContain("n = 4 paired transcripts");
  });

  it("a run that stamps no unit keeps 'paired items' when its records agree, and says the unit is not stamped", async () => {
    const run = await load({}, {}, recordsOf(4, 1));
    const page = render(<EffectsView run={run} onOpenFile={() => {}} />);
    expect(page.text).toContain("n = 4 paired items");
    expect(page.text).toContain("Unit of analysis Not stamped in this run. The item, with an item's samples averaged within each condition.");
    expect(page.text).toContain("The item—not the generation—is the unit of analysis");
    expect(page.text).not.toContain("These are responses");
  });

  it("a run that stamps no unit and has no records claims no unit", async () => {
    const page = render(<EffectsView run={await load({})} onOpenFile={() => {}} />);
    expect(page.text).toContain("n = 4 pairs");
    expect(page.text).not.toContain("paired items");
    expect(page.text).toContain("Not established. The analysis did not stamp it, and the run's records are not available here.");
  });

  it("an analysis from before 0.9.7 that paired every response reads as responses from its items", async () => {
    // One item, four seeds, n = 4: four responses from one item.
    const run = await load({}, {}, recordsOf(1, 4));
    expect(run.effectRows[0].effectUnit).toEqual({ unit: "response", source: "inferred_from_records", pairedItems: 1 });
    const page = render(<EffectsView run={run} onOpenFile={() => {}} />);
    expect(page.text).toContain("n = 4 paired responses from 1 item");
    expect(page.text).toContain("These are responses, not items: the analysis paired each response with the baseline response to the same item and seed");
    expect(page.text).toContain("The response, not the item.");
    expect(page.text).toContain("Rows of paired responses are not read this way.");
    expect(page.text).not.toMatch(/n = 4 paired items?\b/);
    const overview = render(<LocalOverview run={run} onNavigate={() => {}} />);
    expect(overview.text).toContain("too few items for a confidence interval (n = 4 paired responses from 1 item; at least 3 are needed)");
    expect(overview.text).toContain("These are responses, not items");
  });

  it("an analysis is settled against the run it analyzed", async () => {
    enterEmbedded();
    const source = "20260803T100000000-exp-panel-run";
    serveRuns({
      [`${source}/report.json`]: JSON.stringify({ experiment: "panel", conditions: {} }),
      [`${source}/generations.jsonl`]: recordsOf(2, 4),
      [`${RUN}/analysis.json`]: JSON.stringify({ experiment: "panel", sourceRun: source }),
      [`${RUN}/effect-sizes.csv`]: effectCSV,
    });
    const runs = await discoverRuns(embeddedRunsDirectory() as unknown as LocalDirectoryHandle);
    const analysis = runs.find((run) => run.name === RUN)!;
    expect(analysis.sourceRun).toBe(source);
    const settled = await hydrateRun(analysis, findSourceRun(runs, analysis.sourceRun ?? ""));
    expect(settled.effectRows[0].effectUnit).toEqual({ unit: "response", source: "inferred_from_records", pairedItems: 2 });
    expect(settled.effectUnitRecords).toEqual({ counts: { steered: 2 }, complete: true });
    // Without the run it analyzed, nothing settles the unit.
    expect((await hydrateRun(analysis)).effectRows[0].effectUnit?.unit).toBe("unknown");
  });

  it("shows exclusions, endpoint rescue, and adjudication above the effect table", async () => {
    const run = await load({ "exclusions.json": exclusions, "endpoint-reparse.json": reparse, "adjudicated-endpoint.json": adjudication });
    const page = render(<EffectsView run={run} onOpenFile={() => {}} />);
    expect(page.text).toContain("28 records were left out of the paired statistics by 2 declared rules.");
    expect(page.text).toContain("failedAttentionCheck: Drop a record whose attention-check item was answered wrongly.");
    expect(page.text).toContain("Records left for each condition: baseline 60 of 64, steered 40 of 64.");
    expect(page.text).toContain("9 answers that the run could not read were read again with builtin:sentencing");
    expect(page.text).toContain("48 answers were replaced by values from an outside adjudication of parsedMonths");
    // Above the table, not after it: the reader meets them before the numbers.
    expect(page.html.indexOf("What this analysis did to the records")).toBeLessThan(page.html.indexOf("effect-table-card"));
  });

  it("opens the stamp's own file from the card", async () => {
    const run = await load({ "exclusions.json": exclusions });
    const opened: string[] = [];
    const page = render(<EffectsView run={run} onOpenFile={(file) => opened.push(file.path)} />);
    await page.click("exclusions.json");
    expect(opened).toEqual(["exclusions.json"]);
  });

  it("says 'none recorded' for each stamp a run does not carry", async () => {
    const page = render(<EffectsView run={await load({})} onOpenFile={() => {}} />);
    expect(page.text).toContain("Exclusions None recorded.");
    expect(page.text).toContain("Endpoint rescue None recorded.");
    expect(page.text).toContain("Adjudication None recorded.");
  });

  it("explains what the interval means for a real run, in the run's own unit and correction", async () => {
    const itemRun = await load({});
    const items = render(<EffectsView run={itemRun} onOpenFile={() => {}} />);
    expect(items.text).toContain("What the interval means");
    expect(items.text).toContain("on the same prompt items");
    expect(items.text).toContain("after the bh correction");
    leaveHost();
    const transcripts = render(<EffectsView run={await load({ "unit-of-analysis.json": transcriptUnit })} onOpenFile={() => {}} />);
    expect(transcripts.text).toContain("on the same transcripts");
    expect(transcripts.text).toContain("One transcript, as this run stamps it (unit-of-analysis.json).");
  });

  it("shows no interval card for a run with no effect rows", async () => {
    enterEmbedded();
    serveRuns({ [`${RUN}/report.json`]: JSON.stringify({ experiment: "panel", conditions: {} }) });
    const [found] = await discoverRuns(embeddedRunsDirectory() as unknown as LocalDirectoryHandle);
    expect(render(<EffectsView run={await hydrateRun(found)} onOpenFile={() => {}} />).text).not.toContain("What the interval means");
  });

  it("the exported effect table carries the unit of analysis on every row", async () => {
    const host = enterEmbedded();
    serveRuns({
      [`${RUN}/report.json`]: JSON.stringify({ experiment: "panel", conditions: {} }),
      [`${RUN}/effect-sizes.csv`]: effectCSV,
      [`${RUN}/unit-of-analysis.json`]: JSON.stringify(transcriptUnit),
    });
    const [found] = await discoverRuns(embeddedRunsDirectory() as unknown as LocalDirectoryHandle);
    const run = await hydrateRun(found);
    const page = render(<EffectsView run={run} onOpenFile={() => {}} />);
    await page.click("Export CSV");
    const [header, row] = (host.posted[0] as { text: string }).text.split("\n").map(splitCSV);
    expect(row[header.indexOf("unitOfAnalysis")]).toBe("transcript");
    expect(row[header.indexOf("n")]).toBe("4");
  });
});
