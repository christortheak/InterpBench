// Identity and grouping for effect-sizes.csv rows.
//
// Two facts about that table drive everything here.
//
// 1. A row's identity is CONDITION + ENDPOINT. The engines emit one row per
//    condition × endpoint, so "endpoint" alone names several rows the moment
//    a study compares more than one agent — which is the normal shape of a
//    multi-agent comparison. Keying or filtering by endpoint alone made those
//    rows indistinguishable on screen.
//
// 2. Since 2026-08-06 each of those rows may be followed by STRATIFIED
//    companion rows (stratifyBy ≠ "pooled") that re-run the same endpoint
//    within one prompt, one factor level, or one crossed cell. Those belong
//    UNDER their pooled parent, not beside it and not hidden: the strata
//    exist because pooling has both hidden a real single-cell effect behind
//    saturated cells and manufactured a pooled effect out of one cell's parse
//    garbage, and a viewer that shows only the pooled row cannot show either.
//
// Nothing here computes a statistic. Grouping is arrangement; every number
// still comes from the file.

import type { Effect } from "./types";

// ASCII unit separator: it cannot occur in a CSV field either engine writes,
// so no two distinct rows can join to the same key.
const SEPARATOR = "\u001f";

/// Stable identity for a row: condition + endpoint, plus the family and
/// stratum for stratified rows. Used as the React key and as the selection
/// identity, so two conditions' rows for one endpoint never collide.
export const effectKey = (row: { condition: string; endpoint: string; stratifyBy: string; stratum: string }) =>
  [row.condition, row.endpoint, row.stratifyBy, row.stratum].join(SEPARATOR);

/// The (condition, endpoint) a row belongs to — a stratified row shares this
/// with its pooled parent.
const groupKey = (row: Effect) => [row.condition, row.endpoint].join(SEPARATOR);

export type EffectGroup = {
  key: string;
  condition: string;
  endpoint: string;
  /// The all-items row. `null` only for a table that stamped strata with no
  /// pooled parent — the strata are still shown rather than dropped.
  pooled: Effect | null;
  /// The stratified companion rows, in file order.
  strata: Effect[];
};

/// Group the table into one entry per (condition, endpoint), with the
/// stratified rows attached to their pooled parent. File order is preserved:
/// groups appear in the order their first row appears.
export const groupEffects = (rows: Effect[]): EffectGroup[] => {
  const groups = new Map<string, EffectGroup>();
  for (const row of rows) {
    const key = groupKey(row);
    let group = groups.get(key);
    if (!group) {
      group = { key, condition: row.condition, endpoint: row.endpoint, pooled: null, strata: [] };
      groups.set(key, group);
    }
    // A duplicate pooled row (malformed table) is kept as a stratum rather
    // than silently replacing the first — nothing is dropped here.
    if (row.stratifyBy === "pooled" && !group.pooled) group.pooled = row;
    else group.strata.push(row);
  }
  return [...groups.values()];
};

/// The conditions present, in file order — the option list for the condition
/// (agent) filter. A table that stamps no condition contributes nothing, so
/// the filter simply does not appear for it.
export const effectConditions = (rows: Effect[]): string[] =>
  [...new Set(rows.map((row) => row.condition).filter(Boolean))];

/// The endpoints present, in file order, de-duplicated across conditions.
export const effectEndpoints = (rows: Effect[]): string[] =>
  [...new Set(rows.map((row) => row.short))];

/// A stratified row whose estimand is one prompt's own samples. Its p-values
/// are a locator, never a cross-prompt test — the view badges it and shows no
/// corrected p for it.
///
/// Three stamps, any of which settles it. `inference` and `estimand` are the
/// engine's explicit declaration; `unit === "sample"` is the same fact stated
/// by the older column, and it is honoured so that a table from an engine
/// that has not yet grown the estimand columns is demoted here too rather
/// than rendering its corrected p as a cross-item finding. This declines to
/// DISPLAY a number; it does not compute one.
export const isDiagnostic = (row: Effect) =>
  row.inference === "diagnostic" || row.estimand === "withinItemSamples"
  || (row.stratifyBy !== "pooled" && row.pairedUnit === "sample");

/// The paired-count sentence: "n = 48 paired items", or "paired samples"
/// where the row's own `unit` column says a difference is one sample. An
/// absent `n` reads "n not reported" — never "n = 0 items", which is a
/// substantive claim the file did not make.
export const pairedCountLabel = (row: Effect) => {
  if (row.n == null) return "n not reported";
  return `n = ${row.n} paired ${row.pairedUnit === "sample" ? "samples" : "items"}`;
};

/// How to name a row's stratum on screen: "promptID · loan-notLegal".
export const stratumLabel = (row: Effect) =>
  row.stratum ? `${row.stratifyBy} · ${row.stratum}` : row.stratifyBy;

/// Plain-language gloss of the stamped estimand. Returns "" when the table
/// stamped none (a pre-2026-08-06 file) — the view then says nothing rather
/// than guessing which estimand a stratified row carried.
export const estimandLabel = (row: Effect) => {
  if (row.estimand === "itemLevel") return "per-item differences";
  if (row.estimand === "withinItemSamples") return "per-sample differences within one item";
  return "";
};
