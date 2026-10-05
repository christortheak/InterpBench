// Which outcome leads the overview's headline card, and which rule chose it.
//
// A study's summary leads with the outcome the study is about. The rule, in
// order: the outcome the researcher declared (manifest key `primaryOutcome`);
// else a judged outcome; else a declared choice or numeric outcome; else a
// reader or probe score; else a reasoning-style feature; else marker density;
// else a surface measure such as word count. The card always says which rule
// chose its headline.
//
// The mapping from outcome names to tiers is DATA, generated into
// `headlineOutcomes.generated.ts` from the one source both engines read
// (`Server/steerlab_server/client/resources/headline-outcomes.json`, by
// `scripts/ci/check-headline-outcomes.py`). The selection below is the same
// function the Python and Mac engines implement, held to the same cases:
// `Tests/Fixtures/cross-engine/headline-outcome.json`.
//
// Nothing here computes a statistic, and nothing here reorders the table:
// selection is arrangement. Every number the card shows comes from the file.

import { headlineOutcomeData } from "./headlineOutcomes.generated";
import type { Effect } from "./types";

type MappingEntry = { name: string; match: string; tier: string; source: string; plain: string };
type Mapping = {
  manifestKey: string;
  rules: { declared: string; defaultOrder: string };
  tiers: Array<{ id: string; label: string }>;
  unlistedTier: string;
  outcomes: MappingEntry[];
};

const mapping: Mapping = headlineOutcomeData;

/// The manifest key a study declares its primary outcome under.
export const PRIMARY_OUTCOME_KEY = mapping.manifestKey;

/// The one outcome that comes from the evaluation report.
export const JUDGED = "judged";

/// The files a judged outcome is read from, in a run directory.
export const EVALUATION_REPORT_FILES = ["judge-report.json", "coding-report.json"];

export type HeadlineRule = "declared" | "defaultOrder";
export type HeadlineSource = "analysisRows" | "evaluationReport";

export type Headline = {
  /// null when the run has nothing to lead with.
  outcome: string | null;
  tier: string | null;
  rule: HeadlineRule | null;
  source: HeadlineSource | null;
  declaredOutcome: string | null;
  /// The study declared a primary outcome this run does not have, so the
  /// summary fell back; `chosenBy` says so in words.
  declaredAbsent: boolean;
  chosenBy: string;
};

/// The first mapping entry that names `name`, or -1. A prefix or suffix entry
/// needs something left over: `rs_` alone is not a reasoning-style feature,
/// and `MarkerDensity` alone names no concept.
const entryIndex = (name: string): number => mapping.outcomes.findIndex((entry) => {
  if (entry.match === "exact") return name === entry.name;
  if (entry.match === "prefix") return name.startsWith(entry.name) && name.length > entry.name.length;
  if (entry.match === "suffix") return name.endsWith(entry.name) && name.length > entry.name.length;
  return false;
});

/// The tier id of an outcome name. Names the mapping does not list fall in
/// the last tier, after the surface measures.
export const tierOf = (name: string): string => {
  const index = entryIndex(name);
  return index < 0 ? mapping.unlistedTier : mapping.outcomes[index].tier;
};

/// Plain words for an outcome name, or null when the mapping does not list
/// it. `{part}` in a pattern entry is the rest of the name.
export const plainPhrase = (name: string): string | null => {
  const index = entryIndex(name);
  if (index < 0) return null;
  const entry = mapping.outcomes[index];
  const part = entry.match === "prefix" ? name.slice(entry.name.length)
    : entry.match === "suffix" ? name.slice(0, name.length - entry.name.length) : "";
  return entry.plain.replace("{part}", part);
};

/// Code-point comparison (not UTF-16 code units), so the tie-break agrees
/// with the Python and Swift implementations for every name.
const compareByCodePoint = (left: string, right: string): number => {
  const a = Array.from(left, (character) => character.codePointAt(0) ?? 0);
  const b = Array.from(right, (character) => character.codePointAt(0) ?? 0);
  for (let index = 0; index < Math.min(a.length, b.length); index += 1) {
    if (a[index] !== b[index]) return a[index] - b[index];
  }
  return a.length - b.length;
};

/// Default-order comparison: tier, then entry order, then the name itself.
const compareByDefaultOrder = (left: string, right: string): number => {
  const tiers = mapping.tiers.map((tier) => tier.id);
  const key = (name: string): [number, number] => {
    const index = entryIndex(name);
    return [tiers.indexOf(tierOf(name)), index < 0 ? mapping.outcomes.length : index];
  };
  const [a, b] = [key(left), key(right)];
  return a[0] - b[0] || a[1] - b[1] || compareByCodePoint(left, right);
};

/// A declared outcome as stored: a non-empty string, or nothing. Any other
/// value reads as no declaration.
const clean = (declared: unknown): string | null => {
  if (typeof declared !== "string") return null;
  return declared.trim() || null;
};

const unique = (names: string[]): string[] => [...new Set(names.filter(Boolean))];

/// Choose the headline from what a run HAS. `analysisOutcomes` are the
/// outcome names of the pooled effect rows; `evaluationOutcomes` are the
/// names an evaluation report supplies (today only `judged`). A judged
/// outcome is never an analysis row, which is why the two arrive separately.
export const selectHeadline = (declaredValue: unknown, analysisOutcomes: string[], evaluationOutcomes: string[] = []): Headline => {
  const declared = clean(declaredValue);
  const evaluation = unique(evaluationOutcomes);
  const available = [...evaluation, ...unique(analysisOutcomes).filter((name) => !evaluation.includes(name))];
  const sourceOf = (name: string): HeadlineSource => evaluation.includes(name) ? "evaluationReport" : "analysisRows";
  if (declared !== null && available.includes(declared)) {
    return {
      outcome: declared, tier: tierOf(declared), rule: "declared", source: sourceOf(declared),
      declaredOutcome: declared, declaredAbsent: false, chosenBy: mapping.rules.declared,
    };
  }
  const absent = declared !== null ? `; the declared primary outcome '${declared}' is not in this run` : "";
  if (!available.length) {
    return {
      outcome: null, tier: null, rule: null, source: null,
      declaredOutcome: declared, declaredAbsent: declared !== null, chosenBy: `no outcome to lead with${absent}`,
    };
  }
  const first = [...available].sort(compareByDefaultOrder)[0];
  return {
    outcome: first, tier: tierOf(first), rule: "defaultOrder", source: sourceOf(first),
    declaredOutcome: declared, declaredAbsent: declared !== null, chosenBy: `${mapping.rules.defaultOrder}${absent}`,
  };
};

// --- what the card prints ------------------------------------------------------

/// Fewer paired items than this cannot carry an interval, so the card says
/// there are too few pairs instead of printing one.
export const MINIMUM_PAIRS_FOR_INTERVAL = 3;

/// One or two paired items. A row whose `n` is null did not report its
/// count, which is a different fact and is left alone.
export const hasTooFewPairs = (row: Effect): boolean =>
  row.n != null && row.n >= 1 && row.n < MINIMUM_PAIRS_FOR_INTERVAL;

/// How many comparisons the row's multiple-comparison correction covered.
/// Both engines correct one outcome at a time, across the conditions that
/// have a defined test for it, so the family is the pooled rows for the same
/// outcome that carry an adjusted p.
export const correctionFamilySize = (row: Effect, pooled: Effect[]): number =>
  pooled.filter((other) => other.endpoint === row.endpoint && other.q != null).length;

/// The interval and count line. An absent `n` reads "n not reported", never
/// "n = 0"; one or two pairs read as too few for an interval.
export const intervalLine = (row: Effect, format: (value: number) => string): string => {
  if (hasTooFewPairs(row)) {
    return `too few pairs for a confidence interval (n = ${row.n}; at least ${MINIMUM_PAIRS_FOR_INTERVAL} are needed)`;
  }
  return `95% CI ${format(row.low)} to ${format(row.high)} · ${row.n == null ? "n not reported" : `n = ${row.n}`}`;
};

/// The p-value line, or "" when the row prints no test. A correction over ONE
/// comparison changes nothing — the adjusted p equals the raw one — so the
/// card says "adjusted" only when the family has more than one member, and
/// otherwise says what was tested.
export const pLine = (row: Effect, pooled: Effect[]): string => {
  if (hasTooFewPairs(row)) return "";
  if (row.q == null) return row.p == null ? "p not reported" : `uncorrected p ${row.p.toPrecision(2)}`;
  const family = correctionFamilySize(row, pooled);
  if (family > 1) {
    return `adjusted p ${row.q.toPrecision(2)} (${row.correction ? `${row.correction}, ` : ""}${family} comparisons)`;
  }
  return `p ${row.q.toPrecision(2)} (one comparison, so no correction applies)`;
};

/// The pooled rows of the headline outcome, in file order: one per condition.
export const headlineRows = (headline: Headline, pooled: Effect[]): Effect[] =>
  headline.source === "analysisRows" && headline.outcome !== null
    ? pooled.filter((row) => row.endpoint === headline.outcome)
    : [];

/// The card's title for an outcome: plain words first, the engine's name
/// after it. An outcome the mapping does not list shows its own name.
export const outcomeTitle = (name: string): string => {
  const plain = plainPhrase(name);
  return plain === null ? name : `${plain} (${name})`;
};
