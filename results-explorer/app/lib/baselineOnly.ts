// A study that measures the baseline agent alone (public issue #4).
//
// Researchers replicating a published study on an unmodified model want a
// study whose only subject is the baseline agent: the pinned base model, with
// no vectors or adapters, at the study's sampling settings. The sanctioned
// spelling is a declared `studyType: agentComparison` with no agent,
// injection, or SAE latent arm. Its run executes one arm, `baseline`, and its
// analysis has no effects by design, so the explorer says that instead of
// reporting an empty or missing effect table.
//
// The rule is the engines' (`study_shape.baseline_only_study`,
// `ExperimentStore.isBaselineOnlyStudy`), read from the manifest snapshot the
// run carries; test/baselineOnly.test.ts holds it to the shared fixture.

/// How the explorer describes the analysis of such a study. The same
/// sentence the results page and the export use.
export const BASELINE_ONLY_EFFECTS =
  "Baseline only: this study measures the baseline agent alone, so it has no comparison arm and its analysis has no effects.";

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

const nonEmpty = (value: unknown): boolean =>
  Array.isArray(value) ? value.length > 0 : Boolean(value);

/// Whether a run's manifest snapshot (`experiment.json`) is a study that
/// measures the baseline agent alone. With no agent conditions, the only
/// declaration that keeps the concept machinery inert is `agentComparison`,
/// so the rule reads the declaration directly.
export const isBaselineOnlyStudy = (manifest: Record<string, unknown>): boolean => {
  if ((manifest.studyKind || "modelOutput") !== "modelOutput") return false;
  if (manifest.studyType !== "agentComparison") return false;
  if (nonEmpty(manifest.variantConditions) || nonEmpty(manifest.saeLatentConditions)) return false;
  const conditions = Array.isArray(manifest.conditions) ? manifest.conditions : [];
  return conditions.every((condition) => isRecord(condition) && condition.name === "baseline" && !nonEmpty(condition.slots));
};
