// What an analysis did to the records before it estimated anything.
//
// The effect table says how far a condition moved. Four stamps beside it say
// what that movement was measured ON, and each is a separate small file the
// engines write into the same run directory:
//
//   exclusions.json            declared rules that dropped records
//   endpoint-reparse.json      unparsed answers the analysis read again
//   adjudicated-endpoint.json  answers replaced by an outside adjudication
//   unit-of-analysis.json      what one paired difference is (the Mac engine
//                              stamps the same fact in report.json)
//
// The explorer used to show none of them, so an estimate over 40 surviving
// records looked exactly like one over all 64. Everything here is read as
// written. An absent stamp stays absent: the engines write each file only
// when the thing happened, so "no file" means "this run records none".

import { recordValue, textValue } from "./discovery";
import { unsettledUnitSentence, type PairedItems } from "./effectUnits";
import type { Effect, WorkspaceRun } from "./types";

const count = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

const counts = (value: unknown): Record<string, number> =>
  Object.fromEntries(Object.entries(recordValue(value)).flatMap(([key, entry]) => count(entry) === null ? [] : [[key, entry as number]]));

export type ExclusionRule = { rule: string; description: string };

export type ExclusionsStamp = {
  rules: ExclusionRule[];
  /// Records dropped, over every condition. null when the stamp gives none.
  excludedRecords: number | null;
  /// Per condition: how many records the rules looked at, and how many were
  /// left afterwards.
  consideredN: Record<string, number>;
  survivingN: Record<string, number>;
  note: string;
  /// The file it was read from.
  source: string;
};

export type EndpointRescueStamp = {
  endpoint: string;
  parser: string;
  unparsedRecords: number | null;
  rescuedRecords: number | null;
  stillUnparsed: number | null;
  note: string;
};

export type AdjudicationStamp = {
  endpoint: string;
  /// The engine's five classes plus `total`, as stamped.
  counts: Record<string, number>;
  meanAbsDiff: number | null;
  maxAbsDiff: number | null;
  fileSha256: string;
  note: string;
  source: string;
};

export type UnitOfAnalysis = {
  /// "transcript", "item", … exactly as stamped.
  unit: string;
  reason: string;
  /// Independent play-throughs per condition, when the run stamps it.
  transcriptsPerCondition: number | null;
  /// Some endpoints were left out because a condition had one transcript.
  skippedForSingleTranscript: boolean;
  source: string;
};

export type AnalysisStamps = {
  exclusions: ExclusionsStamp | null;
  rescue: EndpointRescueStamp | null;
  adjudication: AdjudicationStamp | null;
  unit: UnitOfAnalysis | null;
  /// The source run carried no study fingerprint, and the analysis went
  /// ahead anyway.
  epochUnverified: boolean;
  /// Measurement settings that differed from the source run's, verbatim.
  measurementDrift: string;
};

export const emptyAnalysisStamps = (): AnalysisStamps => ({
  exclusions: null, rescue: null, adjudication: null, unit: null, epochUnverified: false, measurementDrift: "",
});

/// The run's stamps, or none for a run that has not been activated or was
/// built by hand in a test.
export const analysisStampsOf = (run: WorkspaceRun): AnalysisStamps => run.analysisStamps ?? emptyAnalysisStamps();

/// The parsed JSON of each stamp file the run carries; `{}` for one it does
/// not. `report`, `analysis`, and `config` are the run's report.json,
/// analysis.json, and config.json, which repeat some of the same stamps.
export type AnalysisStampFiles = {
  exclusions?: Record<string, unknown>;
  reparse?: Record<string, unknown>;
  adjudication?: Record<string, unknown>;
  unit?: Record<string, unknown>;
  epochUnverified?: Record<string, unknown>;
  measurementDrift?: Record<string, unknown>;
  report?: Record<string, unknown>;
  analysis?: Record<string, unknown>;
  config?: Record<string, unknown>;
};

const has = (object: Record<string, unknown> | undefined): object is Record<string, unknown> =>
  Boolean(object && Object.keys(object).length);

const exclusionsFrom = (raw: Record<string, unknown>, source: string): ExclusionsStamp => ({
  rules: (Array.isArray(raw.rules) ? raw.rules : []).flatMap((item): ExclusionRule[] => {
    const entry = recordValue(item);
    const rule = textValue(entry, "rule");
    return rule ? [{ rule, description: textValue(entry, "description") }] : [];
  }),
  excludedRecords: count(raw.excludedRecords),
  consideredN: counts(raw.consideredN),
  survivingN: counts(raw.survivingN),
  note: textValue(raw, "note"),
  source,
});

const driftText = (value: unknown): string =>
  typeof value === "string" ? value : value && typeof value === "object" ? JSON.stringify(value) : "";

export const parseAnalysisStamps = (files: AnalysisStampFiles): AnalysisStamps => {
  const report = files.report ?? {};
  const analysis = files.analysis ?? {};
  const notes = recordValue((files.config ?? {}).notes);

  // The Mac engine embeds the exclusion stamp in analysis.json AND writes
  // the file; the file wins when both are present (they are the same
  // object).
  const embeddedExclusions = recordValue(analysis.exclusions);
  const exclusions = has(files.exclusions) ? exclusionsFrom(files.exclusions, "exclusions.json")
    : has(embeddedExclusions) ? exclusionsFrom(embeddedExclusions, "analysis.json")
    : null;

  const reparse = files.reparse;
  const rescue: EndpointRescueStamp | null = has(reparse) ? {
    endpoint: textValue(reparse, "endpoint"),
    parser: textValue(recordValue(reparse.parser), "name"),
    unparsedRecords: count(reparse.unparsedRecords),
    rescuedRecords: count(reparse.rescuedRecords),
    stillUnparsed: count(reparse.stillUnparsed),
    note: textValue(reparse, "note"),
  } : null;

  // The full stamp file, or the summary every adjudicated analysis repeats
  // in config.json's notes so that a reader of that file alone cannot miss
  // the substitution.
  const noted = recordValue(notes.adjudicatedEndpoint);
  const notedDivergence = recordValue(noted.divergence);
  const adjudicated = files.adjudication;
  const adjudication: AdjudicationStamp | null = has(adjudicated) ? {
    endpoint: textValue(adjudicated, "endpoint"),
    counts: counts(adjudicated.counts),
    meanAbsDiff: count(adjudicated.meanAbsDiff),
    maxAbsDiff: count(adjudicated.maxAbsDiff),
    fileSha256: textValue(adjudicated, "fileSha256"),
    note: textValue(adjudicated, "note"),
    source: "adjudicated-endpoint.json",
  } : has(noted) ? {
    endpoint: textValue(notedDivergence, "endpoint"),
    counts: counts(notedDivergence.counts),
    meanAbsDiff: count(notedDivergence.meanAbsDiff),
    maxAbsDiff: count(notedDivergence.maxAbsDiff),
    fileSha256: textValue(noted, "fileSha256"),
    note: "",
    source: "config.json",
  } : null;

  const unitFile = files.unit ?? {};
  const unitSource = textValue(unitFile, "unitOfAnalysis") ? { raw: unitFile, source: "unit-of-analysis.json" }
    : textValue(report, "unitOfAnalysis") ? { raw: report, source: "report.json" }
    : textValue(analysis, "unitOfAnalysis") ? { raw: analysis, source: "analysis.json" }
    : null;
  const unit: UnitOfAnalysis | null = unitSource ? {
    unit: textValue(unitSource.raw, "unitOfAnalysis"),
    reason: textValue(unitSource.raw, "reason"),
    transcriptsPerCondition: count(unitSource.raw.transcriptsPerCondition) ?? count(report.transcriptsPerCondition),
    skippedForSingleTranscript: unitSource.raw.skippedForSingleTranscript === true,
    source: unitSource.source,
  } : null;

  return {
    exclusions,
    rescue,
    adjudication,
    unit,
    epochUnverified: (files.epochUnverified ?? {}).epochUnverified === true || analysis.epochUnverified === true || notes.epochUnverified === true,
    measurementDrift: driftText((files.measurementDrift ?? {}).measurementDrift) || driftText(analysis.measurementDrift) || driftText(notes.measurementDrift),
  };
};

/// Whether the run records any of the stamps at all.
export const hasAnalysisStamps = (stamps: AnalysisStamps) =>
  Boolean(stamps.exclusions || stamps.rescue || stamps.adjudication || stamps.unit || stamps.epochUnverified || stamps.measurementDrift);

// --- sentences ---------------------------------------------------------------
// One plain sentence per stamp, built only from the stamped numbers.

const plural = (value: number, noun: string) => `${value} ${noun}${value === 1 ? "" : "s"}`;

/// A run that stamps no unit is read from its records, never assumed to pair
/// items: `rows` are its effect rows with their settled units, and `records`
/// what they were settled against (lib/effectUnits.ts).
export const unitSentence = (unit: UnitOfAnalysis | null, rows: Pick<Effect, "effectUnit" | "stratifyBy">[] = [], records: PairedItems | null = null): string => {
  if (!unit) return rows.length ? `Not stamped in this run. ${unsettledUnitSentence(rows, records)}` : "Not stamped in this run.";
  if (unit.unit === "transcript") {
    return `One transcript. Turns in the same conversation depend on each other, so each transcript is reduced to its own average difference before any test, and n counts transcripts, not turns.${unit.transcriptsPerCondition !== null ? ` This run has ${plural(unit.transcriptsPerCondition, "transcript")} for each condition.` : ""}${unit.skippedForSingleTranscript ? " Some outcomes were left out because a condition had only one transcript, which gives an estimate but no interval." : ""}`;
  }
  if (unit.unit === "item") return "One prompt item. Each condition is compared with the baseline on the same item, and n counts items.";
  return `"${unit.unit}", as the run stamps it.${unit.reason ? ` ${unit.reason}` : ""}`;
};

export const exclusionsSentence = (stamp: ExclusionsStamp | null): string => {
  if (!stamp) return "None recorded. This run has no exclusions stamp, which the engines write whenever a study declares exclusion rules.";
  const dropped = stamp.excludedRecords === null ? "An unstated number of records were" : stamp.excludedRecords === 1 ? "1 record was" : `${stamp.excludedRecords} records were`;
  return `${dropped} left out of the paired statistics by ${plural(stamp.rules.length, "declared rule")}. They are still in generations.jsonl.`;
};

export const rescueSentence = (stamp: EndpointRescueStamp | null): string => {
  if (!stamp) return "None recorded. No answer was read a second time for this analysis.";
  if (stamp.unparsedRecords === null) return `Answers the run could not read were read again${stamp.parser ? ` with ${stamp.parser}` : ""}. The stamp gives no counts.`;
  if (stamp.unparsedRecords === 0) return "Nothing to rescue. Every answer had already been read when the run was made.";
  return `${plural(stamp.unparsedRecords, "answer")} that the run could not read ${stamp.unparsedRecords === 1 ? "was" : "were"} read again${stamp.parser ? ` with ${stamp.parser}` : ""}: ${stamp.rescuedRecords ?? "an unstated number"} now ${stamp.rescuedRecords === 1 ? "has" : "have"} a value, and ${stamp.stillUnparsed ?? "an unstated number"} still ${stamp.stillUnparsed === 1 ? "does" : "do"} not. Answers the run had already read were left as they were.`;
};

export const adjudicationSentence = (stamp: AdjudicationStamp | null): string => {
  if (!stamp) return "None recorded. Every value in this analysis is the one the run read, or a rescued one.";
  const total = stamp.counts.total;
  const part = (key: string, label: string) => stamp.counts[key] === undefined ? [] : [`${stamp.counts[key]} ${label}`];
  const classes = [
    ...part("agree", "agreed with the run's value"),
    ...part("differ", "differed from it"),
    ...part("rescuedFromNull", "gave a value where the run had none"),
    ...part("nulledFromValue", "removed a value the run had"),
    ...part("unadjudicatable", "had no value before or after"),
  ];
  return `${total === undefined ? "Answers were" : `${plural(total, "answer")} ${total === 1 ? "was" : "were"}`} replaced by values from an outside adjudication${stamp.endpoint ? ` of ${stamp.endpoint}` : ""}${classes.length ? `: ${classes.join(", ")}` : ""}. The effects below use the adjudicated values.`;
};
