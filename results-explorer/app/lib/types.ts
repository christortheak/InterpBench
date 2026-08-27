// Shared types for the Results Explorer. Every view, loader, and the app
// shell import from here so no module re-declares a shape.

import type { RunKind, RunKindSource } from "./runKind";
import type { StatusInfo } from "./status";

export type View ="triage" | "overview" | "concepts" | "optimization" | "effects" | "battery" | "panels" | "generations" | "provenance" | "judged" | "coding" | "choice" | "residuals";

export type Effect = {
  /// The CONDITION the row belongs to (the agent / arm the engine stamped).
  /// The loader used to discard this column, so a multi-condition table
  /// collapsed into rows that were indistinguishable on screen and keyed
  /// against each other in React. A row's identity is condition + endpoint
  /// (+ its stratum, when it is a stratified row) — see `key`.
  condition: string;
  endpoint: string;
  short: string;
  estimate: number;
  low: number;
  high: number;
  unit: string;
  /// null = the table carried no `n` — shown as missing, NEVER as 0. A run
  /// whose n column is blank must not read as "n = 0 items".
  n: number | null;
  /// null = the table carried no adjusted p — shown as missing, NEVER as
  /// significant.
  q: number | null;
  /// The UNADJUSTED Wilcoxon p as stamped in the table (`wilcoxonP`); null
  /// when the column is absent or blank. The effects view used to print a
  /// hardcoded demo array here for every run — a fabricated statistic in the
  /// evidence browser.
  p: number | null;
  /// The correction family the engine stamped ("holm", "bh", …); "" when the
  /// table names none, in which case the column is labelled "adjusted p" and
  /// not attributed to a method the file never declared.
  correction: string;
  direction: "positive" | "negative";
  /// Stratification family the engine stamped: "pooled" for the all-items
  /// row, otherwise "promptID", a factor key, or "×"-joined crossed keys.
  /// Legacy tables without the column read as "pooled".
  stratifyBy: string;
  /// The cell label within that family ("" on pooled rows).
  stratum: string;
  /// The engine's `unit` column — what ONE paired difference is: "item",
  /// "sample", or "" on pooled rows. NOT the display unit above.
  pairedUnit: string;
  /// What the row estimates, as stamped: "itemLevel" (generalizes over the
  /// items in the stratum) or "withinItemSamples" (one prompt's own
  /// generations — prompt-specific, supports no cross-prompt claim). Empty
  /// on pooled rows and on tables written before the column existed.
  estimand: string;
  /// What the row's p-values are licensed for, as stamped: "corrected" (an
  /// adjustedP from a within-family correction) or "diagnostic" (held out of
  /// every correction family; raw p is a locator, not a test).
  inference: string;
  /// Stable row identity for keying and selection: condition, endpoint, and
  /// — for stratified rows — the family and stratum.
  key: string;
};

export type Generation = {
  id: string;
  caseName: string;
  family: string;
  condition: string;
  alpha: string;
  sample: number;
  decision: string;
  months: number | null;
  prompt: string;
  output: string;
  parsed: string;
  words: number;
  distinct2: number;
  seed: number;
  /// The record carried an `instrument` marker — i.e. it is an answer-token
  /// choice readout, not a sampled-text generation. Set by the loader from
  /// the record itself. It replaces the old test of whether `output` began
  /// with the loader's own "Selected option: " synthesis, which could only
  /// ever recognise instrument records that stored NO text, and which read a
  /// model that genuinely began its answer that way as an instrument record.
  isInstrument: boolean;
  /// Whether `words` is the engine's stamped `wordCount` (true) or a count
  /// the VIEWER made over the output text because the record carried none
  /// (false). A viewer-counted word count is a derived number and says so
  /// wherever it is shown.
  wordCountStored: boolean;
  /// Whether `distinct2` is stamped on the record. When false the field is 0
  /// as a placeholder and must render as "not stamped", never as 0.00 —
  /// a real distinct-2 of zero and an absent one are different facts.
  distinct2Stored: boolean;
  promptIndex?: number;
  speakerName?: string;
  turnTitle?: string;
  routedAgentIDs?: string[] | null;
  replicateIndex?: number | null;
  modelID?: string;
};

export type SweepRow = {
  concept: string;
  layer: number;
  alpha: number;
  markerDensity: number;
  distinct2: number;
  batteryAccuracy: number | null;
  objective: number | null;
  distinct2Ratio: number | null;
  words: number | null;
  lengthInflated: boolean;
};

export type SweepRecommendation = {
  concept: string;
  failure: string;
  layer: number | null;
  alpha: number | null;
  metric: string;
  metrics: Record<string, number>;
  capabilityTolerance: number;
  /** The absolute distinct-2 floor — the backstop under the baseline-relative rule. */
  coherenceFloor: number;
  /** Non-null = the baseline-relative rule at this multiple of the baseline's distinct-2. */
  coherenceRatioToBaseline: number | null;
  matchedNormRandomMargin: number | null;
  devPromptsHash: string;
  batteryHash: string;
  sweepRun: string;
};

export type PanelEffect = {
  endpoint: string;
  direct: number | null;
  directN: number;
  spillover: number | null;
  spilloverN: number;
  group: number | null;
  groupN: number;
  transmissionRatio: number | null;
  amplification: number | null;
  droppedTurns: number;
};

export type LocalFileHandle = {
  kind: "file";
  name: string;
  getFile: () => Promise<File>;
};

export type LocalDirectoryHandle = {
  kind: "directory";
  name: string;
  values: () => AsyncIterableIterator<LocalFileHandle | LocalDirectoryHandle>;
  getDirectoryHandle: (name: string) => Promise<LocalDirectoryHandle>;
  getFileHandle: (name: string) => Promise<LocalFileHandle>;
};

export type RunFile = {
  name: string;
  path: string;
  size: number;
  modified: number;
  handle: LocalFileHandle;
};

export type CosineMatrix = {
  file: string;
  concepts: string[];
  values: Array<Array<number | null>>;
  layer: number | null;
  mixedLayers: boolean;
};

export type ValidationConcept = {
  name: string;
  layer: number | null;
  scenarios: number | null;
  accuracy: number | null;
  calibratedAccuracy: number | null;
  auc: number | null;
  oneSided: boolean | null;
  note: string;
  positiveTokens: string[];
  negativeTokens: string[];
};

export type WorkspaceRun = {
  key: string;
  name: string;
  path: string;
  experiment: string;
  status: string;
  model: string;
  dateLabel: string;
  promptCount: number;
  conditionCount: number;
  generationCount: number;
  report: Record<string, unknown>;
  config: Record<string, unknown>;
  artifacts: string[];
  files: RunFile[];
  handle: LocalDirectoryHandle;
  effectRows: Effect[];
  generationRows: Generation[];
  generationFile: LocalFileHandle | null;
  previewTruncated: boolean;
  skippedGenerationLines: number;
  cosineMatrices: CosineMatrix[];
  validationConcepts: ValidationConcept[];
  validationReport: Record<string, unknown>;
  sweepRows: SweepRow[];
  sweepRecommendations: SweepRecommendation[];
  panelEffects: PanelEffect[];
  // --- appended 2026-08-05 (upgrade plan Phase 0). Append-only: other
  // modules read this type, so fields are added at the end, never reordered
  // or renamed. They are OPTIONAL because `WorkspaceRun` is also built by
  // hand in the unit suite's fixtures; `discoverRuns` always populates them,
  // and `runKindOf` / `runStatusOf` in lib/discovery.ts give every consumer a
  // total reading (kind "unknown", state "not stamped") without a `?.` chain.
  /// What kind of run this is — config.json `runType` when stamped, artifact
  /// presence otherwise. See lib/runKind.ts.
  kind?: RunKind;
  /// Whether `kind` came from the engine stamp or was inferred by the viewer.
  kindSource?: RunKindSource;
  /// The raw `config.json` runType stamp, verbatim; "" when absent.
  runTypeStamp?: string;
  /// run-status.json + FAILED.md + cancelled.txt, read at discovery. Replaces
  /// every `status === "complete"` reading. See lib/status.ts.
  statusInfo?: StatusInfo;
  /// The run this one consumed, resolved from (in order) run-status.json,
  /// source-run.txt, judging-context.json, and report.json's `sourceRun`.
  /// "" when this run is a chain root.
  sourceRun?: string;
  /// `pipeline.json` (or `pipeline-portable.json`) for a pipeline ledger
  /// directory; {} for every other run.
  pipeline?: Record<string, unknown>;
  /// Sortable directory-name timestamp prefix (`YYYYMMDDTHHMMSSmmm`), or ""
  /// when the name carries none.
  timestampKey?: string;
};

export type FilePreview = {
  file: RunFile;
  text: string;
  truncated: boolean;
  loading: boolean;
  error: string;
};

export type PickerWindow = Window & {
  showDirectoryPicker?: (options?: { mode?: "read" | "readwrite" }) => Promise<LocalDirectoryHandle>;
};
