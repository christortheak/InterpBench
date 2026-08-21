// Multi-agent panel readers: the run's scenario snapshot (engine contract
// landed 2026-08-05), the model-variant artifacts its seats point at, and
// the per-seat descriptive aggregates the viewer derives from turn records.
//
// Two hard rules this module exists to keep:
//
// 1. NOTHING here parses turn TEXT. Votes, dispositions, and dissents are an
//    engine instrument (the declared turn-endpoint contract, Phase 4 of
//    docs/RESULTS-EXPLORER-UPGRADE-PLAN.md); free-text endpoint mining never
//    ships in the viewer. Every aggregate below is a turn/word/distinct-2
//    statistic over stored record fields.
// 2. Absence is reported, never inferred. A run with no scenario.json is a
//    run whose seat attribution was not recorded — that fact is the answer,
//    not a prompt to guess from speaker names.

import { findFile } from "./discovery";
import type { Generation, LocalDirectoryHandle, WorkspaceRun } from "./types";

const str = (value: unknown) => (typeof value === "string" ? value : "");
const num = (value: unknown) => (typeof value === "number" && Number.isFinite(value) ? value : null);
const record = (value: unknown): Record<string, unknown> => (value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {});

/* ------------------------------------------------------------------ */
/* Scenario snapshot                                                    */
/* ------------------------------------------------------------------ */

export type ScenarioAgent = {
  id: string;
  name: string;
  baseModelID: string;
  systemPrompt: string;
  /// Workspace-relative path ("runs/model-variants/…/model-variant.json")
  /// or "" for a stock seat.
  variantArtifactPath: string;
  variantArtifactHash: string;
};

export type ScenarioTurn = {
  id: string;
  title: string;
  speakerAgentID: string;
  outputLabel: string;
  routing: string;
  routedAgentIDs: string[] | null;
};

export type PanelScenario = {
  name: string;
  description: string;
  baseModelID: string;
  schemaVersion: number | null;
  temperature: number | null;
  maxTokens: number | null;
  sharedMaterials: string;
  agents: ScenarioAgent[];
  turns: ScenarioTurn[];
};

export const SCENARIO_ABSENT_NOTE =
  "This run predates the scenario snapshot (2026-08-05) — seat attribution is not recorded in the run directory.";
export const SCENARIO_UNREADABLE_NOTE =
  "A scenario.json is present but could not be read as a scenario. Seat attribution is withheld rather than guessed.";
export const VARIANT_UNREACHABLE_NOTE =
  "Variant artifact not reachable from this session's folder permission — the scenario names it, but only the selected run's files were granted. Choose the workspace (or its runs folder) to resolve it, or open this run in the app's embedded explorer.";

/** Parse a scenario snapshot. Returns null when the JSON is not a scenario. */
export const parseScenario = (raw: unknown): PanelScenario | null => {
  const root = record(raw);
  if (!Array.isArray(root.agents)) return null;
  const agents = root.agents.flatMap((entry): ScenarioAgent[] => {
    const agent = record(entry);
    if (!str(agent.id) && !str(agent.name)) return [];
    return [{
      id: str(agent.id),
      name: str(agent.name) || str(agent.id),
      baseModelID: str(agent.baseModelID) || str(root.baseModelID),
      systemPrompt: str(agent.systemPrompt),
      variantArtifactPath: str(agent.variantArtifactPath),
      variantArtifactHash: str(agent.variantArtifactHash),
    }];
  });
  if (!agents.length) return null;
  const turns = (Array.isArray(root.turns) ? root.turns : []).flatMap((entry): ScenarioTurn[] => {
    const turn = record(entry);
    if (!str(turn.title) && !str(turn.id)) return [];
    return [{
      id: str(turn.id),
      title: str(turn.title),
      speakerAgentID: str(turn.speakerAgentID),
      outputLabel: str(turn.outputLabel),
      routing: str(turn.routing),
      routedAgentIDs: Array.isArray(turn.routedAgentIDs)
        ? turn.routedAgentIDs.filter((item): item is string => typeof item === "string")
        : null,
    }];
  });
  return {
    name: str(root.name),
    description: str(root.description),
    baseModelID: str(root.baseModelID),
    schemaVersion: num(root.schemaVersion),
    temperature: num(root.temperature),
    maxTokens: num(root.maxTokens),
    sharedMaterials: str(root.sharedMaterials),
    agents,
    turns,
  };
};

export type ScenarioLoad = {
  scenario: PanelScenario | null;
  /// "loaded" | "absent" (no snapshot in the run) | "unreadable".
  state: "loaded" | "absent" | "unreadable";
  note: string;
};

/** Read `<run>/scenario.json`. Absence is a stated fact, not an error. */
export const loadScenario = async (run: WorkspaceRun): Promise<ScenarioLoad> => {
  const runFile = findFile(run.files, "scenario.json");
  if (!runFile) return { scenario: null, state: "absent", note: SCENARIO_ABSENT_NOTE };
  try {
    const scenario = parseScenario(JSON.parse(await (await runFile.handle.getFile()).text()));
    if (!scenario) return { scenario: null, state: "unreadable", note: SCENARIO_UNREADABLE_NOTE };
    return { scenario, state: "loaded", note: "" };
  } catch {
    return { scenario: null, state: "unreadable", note: SCENARIO_UNREADABLE_NOTE };
  }
};

/* ------------------------------------------------------------------ */
/* Model-variant artifacts (the steered seats)                          */
/* ------------------------------------------------------------------ */

export type VariantInjection = {
  concept: string;
  layer: number | null;
  alpha: number | null;
  /// "add" (steer) or "ablate"; the artifact omits the key when it is `add`.
  mode: string;
  vectorArtifactID: string;
};

export type VariantPromotion = {
  promotedBy: string;
  sweepRun: string;
  experiment: string;
  promotedAt: string;
  overrideReason: string;
  selectionOutcome: string;
  winningLayer: number | null;
  winningAlpha: number | null;
};

export type VariantArtifact = {
  name: string;
  baseModelID: string;
  baseRevision: string;
  alphaInNormUnits: boolean | null;
  bandWidth: number | null;
  injections: VariantInjection[];
  adapters: string[];
  promotion: VariantPromotion | null;
};

/** Parse a `model-variant.json`. Returns null when it is not one. */
export const parseVariantArtifact = (raw: unknown): VariantArtifact | null => {
  const root = record(raw);
  if (!Array.isArray(root.injections) || !str(root.baseModelID)) return null;
  const promotionRaw = root.promotion;
  const promotion = promotionRaw && typeof promotionRaw === "object" && !Array.isArray(promotionRaw)
    ? record(promotionRaw)
    : null;
  const cell = promotion ? record(promotion.winningCell) : {};
  return {
    name: str(root.name),
    baseModelID: str(root.baseModelID),
    baseRevision: str(root.baseRevision),
    alphaInNormUnits: typeof root.alphaInNormUnits === "boolean" ? root.alphaInNormUnits : null,
    bandWidth: num(root.bandWidth),
    injections: root.injections.flatMap((entry): VariantInjection[] => {
      const injection = record(entry);
      if (!str(injection.concept)) return [];
      return [{
        concept: str(injection.concept),
        layer: num(injection.layer),
        alpha: num(injection.alpha),
        // Absent means `add`: the engine omits the key for steering agents
        // so their bytes stay hash-stable (ModelVariantStore.swift).
        mode: str(injection.mode) || "add",
        vectorArtifactID: str(injection.vectorArtifactID),
      }];
    }),
    adapters: (Array.isArray(root.adapters) ? root.adapters : []).map((entry) => {
      const adapter = record(entry);
      return str(adapter.name) || str(adapter.adapterDirectory) || "unnamed adapter";
    }),
    promotion: promotion
      ? {
        promotedBy: str(promotion.promotedBy),
        sweepRun: str(promotion.sweepRun),
        experiment: str(promotion.experiment),
        promotedAt: str(promotion.promotedAt),
        overrideReason: str(promotion.overrideReason),
        selectionOutcome: str(promotion.selectionOutcome),
        winningLayer: num(cell.layer),
        winningAlpha: num(cell.alpha),
      }
      : null,
  };
};

/**
 * Path segments of a workspace-relative artifact path, relative to the
 * workspace's `runs/` directory — the only tree either ingestion path can
 * reach. `null` when the path leaves that tree (absolute, `..`, or rooted
 * anywhere but `runs/`), which is reported as unreachable rather than
 * hunted for.
 */
export const runsRelativeSegments = (path: string): string[] | null => {
  const trimmed = path.trim().replace(/^\.\//, "");
  if (!trimmed || trimmed.startsWith("/")) return null;
  const segments = trimmed.split("/").filter((segment) => segment && segment !== ".");
  if (segments.some((segment) => segment === "..")) return null;
  if (segments[0] !== "runs") return null;
  const rest = segments.slice(1);
  return rest.length ? rest : null;
};

export type SeatVariant =
  | { state: "stock" }
  | { state: "loaded"; artifact: VariantArtifact }
  /** The library artifact parsed, but its bytes no longer hash to the
   * scenario's pin. `runs/model-variants/` is a MUTABLE library subtree by
   * design, so this is a real, expected state — the fields shown are the
   * CURRENT artifact, not necessarily what ran. */
  | { state: "drifted"; artifact: VariantArtifact; note: string }
  | { state: "unreachable"; note: string }
  | { state: "unreadable"; note: string };

/** SHA-256 hex of a UTF-8 string (Web Crypto — available in the embedded
 * WKWebView, browsers, and the Node test runner). */
export const sha256Hex = async (text: string): Promise<string> => {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
};

export type PanelSeat = {
  agentID: string;
  name: string;
  baseModelID: string;
  systemPrompt: string;
  variantPath: string;
  variantHash: string;
  variant: SeatVariant;
};

/**
 * Read one variant artifact through the runs/ directory handle. `null`
 * handle = no reachable runs tree for this session (the browser path when
 * the researcher granted only a single run directory).
 */
export const loadVariantArtifact = async (
  runsDirectory: LocalDirectoryHandle | null,
  path: string,
  declaredHash = "",
): Promise<SeatVariant> => {
  if (!path) return { state: "stock" };
  const segments = runsRelativeSegments(path);
  if (!segments) {
    return { state: "unreachable", note: `${VARIANT_UNREACHABLE_NOTE} Declared path: ${path}` };
  }
  if (!runsDirectory) return { state: "unreachable", note: VARIANT_UNREACHABLE_NOTE };
  // Reaching the bytes and understanding them are separate failures: a file
  // the session cannot open is a permission fact, a file it opens and
  // cannot parse is a data fact. They read very differently to a researcher.
  let text: string;
  try {
    let directory = runsDirectory;
    for (const segment of segments.slice(0, -1)) directory = await directory.getDirectoryHandle(segment);
    const handle = await directory.getFileHandle(segments[segments.length - 1]);
    text = await (await handle.getFile()).text();
  } catch {
    return { state: "unreachable", note: `${VARIANT_UNREACHABLE_NOTE} Declared path: ${path}` };
  }
  try {
    const artifact = parseVariantArtifact(JSON.parse(text));
    if (!artifact) throw new Error("not a model-variant artifact");
    // The library subtree is mutable BY DESIGN (a promoted agent's artifact
    // is editable in place), so the loaded bytes are only "what ran" if
    // they still hash to the scenario's pin. Drift is shown, never hidden —
    // but labeled as the current artifact, not the run's.
    if (declaredHash && (await sha256Hex(text)) !== declaredHash) {
      return {
        state: "drifted", artifact,
        note: "This artifact has CHANGED since the scenario pinned it "
          + "(bytes no longer match variantArtifactHash). Fields shown are "
          + "the current library artifact, not necessarily what ran.",
      };
    }
    return { state: "loaded", artifact };
  } catch {
    return { state: "unreadable", note: `The file at ${path} could not be read as a model-variant artifact. Its seat is shown unattributed rather than guessed.` };
  }
};

/** Every scenario seat with its variant artifact resolved (or its absence explained). */
export const resolveSeats = async (
  scenario: PanelScenario,
  runsDirectory: LocalDirectoryHandle | null,
): Promise<PanelSeat[]> =>
  await Promise.all(scenario.agents.map(async (agent) => ({
    agentID: agent.id,
    name: agent.name,
    baseModelID: agent.baseModelID,
    systemPrompt: agent.systemPrompt,
    variantPath: agent.variantArtifactPath,
    variantHash: agent.variantArtifactHash,
    variant: await loadVariantArtifact(
      runsDirectory, agent.variantArtifactPath, agent.variantArtifactHash),
  })));

/**
 * The at-a-glance seat label: "fear (L31, α 0.20)". Stored values only —
 * concept, layer, and alpha are read from the artifact, never computed.
 */
export const injectionLabel = (injection: VariantInjection) => {
  const layer = injection.layer == null ? "layer not stamped" : `L${injection.layer}`;
  const strength = injection.alpha == null
    ? "strength not stamped"
    : `${injection.mode === "ablate" ? "λ" : "α"} ${injection.alpha.toFixed(2)}`;
  return `${injection.concept} (${layer}, ${strength})`;
};

export const variantHeadline = (artifact: VariantArtifact) =>
  artifact.injections.length
    ? artifact.injections.map(injectionLabel).join(" · ")
    : artifact.adapters.length
      ? `adapters: ${artifact.adapters.join(", ")}`
      : "no injections or adapters stamped";

/* ------------------------------------------------------------------ */
/* Per-seat aggregates (DERIVED — every caller badges these)            */
/* ------------------------------------------------------------------ */

/** The multi-agent turn records: rows carrying a speaker or a turn title. */
export const panelRows = (rows: Generation[]) => rows.filter((row) => row.speakerName || row.turnTitle);

export const speakerNames = (rows: Generation[]) =>
  [...new Set(panelRows(rows).map((row) => row.speakerName || "Unstamped speaker"))].sort();

/**
 * Turn titles in the scenario's own turn order — ordered by each title's
 * smallest promptIndex, which the engine stamps as global turn order.
 */
export const turnTitles = (rows: Generation[]) => {
  const first = new Map<string, number>();
  panelRows(rows).forEach((row) => {
    const title = row.turnTitle || "Untitled turn";
    const index = row.promptIndex ?? 0;
    const seen = first.get(title);
    if (seen === undefined || index < seen) first.set(title, index);
  });
  return [...first.entries()].sort((left, right) => left[1] - right[1] || left[0].localeCompare(right[0])).map(([title]) => title);
};

const mean = (values: number[]) => (values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null);

export type SeatConditionCell = {
  condition: string;
  isBaseline: boolean;
  turns: number;
  meanWordCount: number | null;
  meanDistinct2: number | null;
  /// Δ against the SAME seat's baseline-condition cell; null when there is
  /// no baseline column or the cell is the baseline itself.
  deltaWordCount: number | null;
  deltaDistinct2: number | null;
  percentWordCount: number | null;
};

export type SeatRow = { speaker: string; cells: SeatConditionCell[] };

export type SeatTable = {
  turnTitle: string | null;
  conditions: string[];
  baseline: string | null;
  /// False when no condition is literally named "baseline" and the first
  /// condition was taken as the comparison column — a viewer convention the
  /// data does not declare (badge it heuristic).
  baselineDeclared: boolean;
  seats: SeatRow[];
  rowsUsed: number;
};

/**
 * Per-seat × condition turn/word/distinct-2 means, with deltas against the
 * seat's own baseline column. `turnTitle` null aggregates every turn.
 *
 * Derived in the viewer: means over the run's stored per-record `wordCount`
 * / `distinct2`; no engine artifact carries this cut.
 */
export const seatTable = (rows: Generation[], turnTitle: string | null = null): SeatTable => {
  const scoped = panelRows(rows).filter((row) => turnTitle == null || (row.turnTitle || "Untitled turn") === turnTitle);
  const conditions = [...new Set(scoped.map((row) => row.condition))].sort((left, right) =>
    left === "baseline" ? -1 : right === "baseline" ? 1 : left.localeCompare(right));
  const baselineDeclared = conditions.includes("baseline");
  const baseline = baselineDeclared ? "baseline" : conditions[0] ?? null;
  const speakers = [...new Set(scoped.map((row) => row.speakerName || "Unstamped speaker"))].sort();
  const seats = speakers.map((speaker): SeatRow => {
    const cellFor = (condition: string) => {
      const cellRows = scoped.filter((row) => (row.speakerName || "Unstamped speaker") === speaker && row.condition === condition);
      return {
        turns: cellRows.length,
        meanWordCount: mean(cellRows.map((row) => row.words)),
        meanDistinct2: mean(cellRows.map((row) => row.distinct2)),
      };
    };
    const baselineCell = baseline ? cellFor(baseline) : null;
    return {
      speaker,
      cells: conditions.map((condition): SeatConditionCell => {
        const cell = condition === baseline && baselineCell ? baselineCell : cellFor(condition);
        const isBaseline = condition === baseline;
        const deltaWordCount = isBaseline || !baselineCell || cell.meanWordCount == null || baselineCell.meanWordCount == null
          ? null
          : cell.meanWordCount - baselineCell.meanWordCount;
        const deltaDistinct2 = isBaseline || !baselineCell || cell.meanDistinct2 == null || baselineCell.meanDistinct2 == null
          ? null
          : cell.meanDistinct2 - baselineCell.meanDistinct2;
        return {
          condition,
          isBaseline,
          ...cell,
          deltaWordCount,
          deltaDistinct2,
          percentWordCount: deltaWordCount == null || !baselineCell?.meanWordCount
            ? null
            : (deltaWordCount / baselineCell.meanWordCount) * 100,
        };
      }),
    };
  });
  return { turnTitle, conditions, baseline, baselineDeclared, seats, rowsUsed: scoped.length };
};

/* ------------------------------------------------------------------ */
/* Transcript grouping                                                  */
/* ------------------------------------------------------------------ */

export type PanelTranscript = { condition: string; replicate: number; turns: Generation[] };

const byCondition = (left: string, right: string) =>
  left === "baseline" ? -1 : right === "baseline" ? 1 : left.localeCompare(right);

const byTurnOrder = (left: Generation, right: Generation) => (left.promptIndex ?? 0) - (right.promptIndex ?? 0);

/** One arm per (condition, replicate), turns in the scenario's turn order. */
export const transcripts = (rows: Generation[]): PanelTranscript[] => {
  const grouped = new Map<string, Generation[]>();
  panelRows(rows).forEach((row) => {
    const key = `${row.condition}\u001f${row.replicateIndex ?? 0}`;
    grouped.set(key, [...(grouped.get(key) ?? []), row]);
  });
  return [...grouped.entries()]
    .map(([key, turns]) => {
      const [condition, replicate] = key.split("\u001f");
      return { condition, replicate: Number(replicate), turns: [...turns].sort(byTurnOrder) };
    })
    .sort((left, right) => byCondition(left.condition, right.condition) || left.replicate - right.replicate);
};

export type SpeakerReplicate = {
  replicate: number;
  columns: Array<{ condition: string; turns: Generation[] }>;
};

export type SpeakerAudit = {
  speaker: string;
  turnTitle: string | null;
  conditions: string[];
  replicates: SpeakerReplicate[];
  turnsShown: number;
};

/**
 * One speaker's turns across EVERY replicate, conditions side by side —
 * how a per-seat behavioural audit actually reads ("all Judge C final
 * positions, both conditions"). Grouping only; no text is interpreted.
 */
export const speakerAudit = (rows: Generation[], speaker: string, turnTitle: string | null = null): SpeakerAudit => {
  const scoped = panelRows(rows).filter((row) =>
    (row.speakerName || "Unstamped speaker") === speaker &&
    (turnTitle == null || (row.turnTitle || "Untitled turn") === turnTitle));
  const conditions = [...new Set(scoped.map((row) => row.condition))].sort(byCondition);
  const replicateNumbers = [...new Set(scoped.map((row) => row.replicateIndex ?? 0))].sort((left, right) => left - right);
  return {
    speaker,
    turnTitle,
    conditions,
    replicates: replicateNumbers.map((replicate) => ({
      replicate,
      columns: conditions.map((condition) => ({
        condition,
        turns: scoped
          .filter((row) => (row.replicateIndex ?? 0) === replicate && row.condition === condition)
          .sort(byTurnOrder),
      })),
    })),
    turnsShown: scoped.length,
  };
};

/**
 * The scenario's declared routing per turn title (stored) — the turn
 * record's `routedAgentIDs` says who received it, the scenario says what
 * rule produced that. Empty map when no scenario snapshot was read.
 */
export const routingByTurnTitle = (scenario: PanelScenario | null) =>
  new Map((scenario?.turns ?? []).flatMap((turn) => (turn.title && turn.routing ? [[turn.title, turn.routing] as const] : [])));

/** Stored visibility label for a turn record's routing stamp. */
export const routingLabel = (turn: Generation) => {
  if (turn.routedAgentIDs === undefined) return "Visibility not stamped";
  if (turn.routedAgentIDs === null || turn.routedAgentIDs.length === 0) return "Private turn";
  return `Routed to ${turn.routedAgentIDs.length} agent${turn.routedAgentIDs.length === 1 ? "" : "s"}`;
};
