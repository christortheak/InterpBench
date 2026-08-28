// Artifact loaders: every readable file in a discovered run directory is
// parsed here, and only here. Missing artifacts return empty results —
// nothing is inferred or substituted.

import { splitCSV, strictNumber } from "./csv";
import { findFile, recordValue } from "./discovery";
import { effectKey } from "./effects";
import { GENERATION_INSTRUMENT_OUTPUT_PREFIX } from "./instruments";
import type {
  CosineMatrix,
  Effect,
  Generation,
  PanelEffect,
  SweepRecommendation,
  SweepRow,
  ValidationConcept,
  WorkspaceRun,
} from "./types";

export const loadEffects = async (run: WorkspaceRun): Promise<Effect[]> => {
  const runFile = findFile(run.files, "effect-sizes.csv");
  if (!runFile) return [];
  const lines = (await (await runFile.handle.getFile()).text()).split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return [];
  const headers = splitCSV(lines[0]).map((value) => value.toLowerCase());
  const cell = (row: string[], ...keys: string[]) => row[headers.findIndex((header) => keys.includes(header))] ?? "";
  return lines.slice(1).map(splitCSV).flatMap((row) => {
    // Both engines append per-stratum companion rows (stratifyBy ≠ "pooled",
    // 2026-08-06). ALL of them are read here — the view nests them under
    // their pooled parent rather than hiding them, so a saturated-cell
    // masking (the failure the strata exist to expose) is visible in the
    // browser and not only in the raw CSV. Legacy files without the column
    // are all pooled.
    const stratifyBy = cell(row, "stratifyby") || "pooled";
    const stratum = cell(row, "stratum");
    const endpoint = cell(row, "endpoint", "metric");
    const estimate = strictNumber(cell(row, "deltamean", "meandiff"));
    const low = strictNumber(cell(row, "cilower"));
    const high = strictNumber(cell(row, "ciupper"));
    if (!endpoint || estimate == null || low == null || high == null) return [];
    const normalized = endpoint.toLowerCase();
    const unit = normalized.includes("month") ? "months" : normalized.includes("rate") || normalized.includes("choice") || normalized.includes("prob") ? "Δ probability" : "Δ units";
    // The condition is part of the row's IDENTITY. Dropping it made every
    // condition's row for one endpoint look (and key) alike — a multi-agent
    // comparison rendered as a stack of duplicates.
    const condition = cell(row, "condition");
    return [{
      condition, endpoint, short: endpoint.replaceAll("_", " "), estimate, low, high, unit,
      // Missing stays missing: `strictNumber` returns null for a blank cell
      // and the view prints "n not reported". This used to `?? 0`, so an
      // absent count rendered as the substantive claim "n = 0 items".
      n: strictNumber(cell(row, "n")),
      q: strictNumber(cell(row, "adjustedp")),
      // Both engines stamp the raw Wilcoxon p and the correction family
      // beside the adjusted p; the effects view showed neither, printing a
      // hardcoded demo array in the "Raw p" column instead.
      p: strictNumber(cell(row, "wilcoxonp")),
      correction: cell(row, "correction"),
      direction: estimate < 0 ? "negative" as const : "positive" as const,
      stratifyBy, stratum,
      pairedUnit: cell(row, "unit"),
      estimand: cell(row, "estimand"),
      inference: cell(row, "inference"),
      key: effectKey({ condition, endpoint, stratifyBy, stratum }),
    }];
  });
};

export const loadGenerations = async (run: WorkspaceRun) => {
  const runFile = findFile(run.files, "generations.jsonl");
  if (!runFile) return { rows: [] as Generation[], handle: null, truncated: false, skipped: 0 };
  const file = await runFile.handle.getFile();
  const limit = 32 * 1024 * 1024;
  const truncated = file.size > limit;
  const text = await file.slice(0, limit).text();
  const lines = text.split(/\r?\n/);
  if (truncated) lines.pop();
  let skipped = 0;
  const rows = lines.flatMap((line, index): Generation[] => {
    if (!line.trim()) return [];
    try {
      const record = JSON.parse(line) as Record<string, unknown>;
      if (typeof record.condition !== "string" || (typeof record.output !== "string" && typeof record.selected !== "string")) { skipped += 1; return []; }
      const parsedMonths = typeof record.parsedMonths === "number" ? record.parsedMonths : null;
      const parsedChoice = typeof record.parsedChoice === "string" ? record.parsedChoice : typeof record.selected === "string" ? record.selected : "Not parsed";
      // The record's OWN marker, read exactly as lib/instruments.ts reads it,
      // so a row and its full readout can never disagree about what kind of
      // record they are.
      const isInstrument = typeof record.instrument === "string" && record.instrument !== "";
      const wordCountStored = typeof record.wordCount === "number";
      const distinct2Stored = typeof record.distinct2 === "number";
      return [{
        id: `${String(record.promptID ?? index + 1)} · S${String(record.sampleIndex ?? 0).padStart(2, "0")}`,
        caseName: String(record.caseID ?? record.promptID ?? `Record ${index + 1}`),
        family: String(record.arm ?? record.caseID ?? "Study generation"),
        condition: record.condition,
        alpha: String(record.alpha ?? "—"),
        sample: typeof record.sampleIndex === "number" ? record.sampleIndex : 0,
        decision: parsedMonths !== null ? `${parsedMonths} months` : parsedChoice,
        months: parsedMonths,
        prompt: typeof record.prompt === "string" ? record.prompt : "Prompt text was not stored in this record.",
        // Display-only fallback for a record that stored no text (see
        // lib/instruments.ts): what KIND of record this is comes from
        // `isInstrument` above, never from reading this string back.
        output: typeof record.output === "string" ? record.output : `${GENERATION_INSTRUMENT_OUTPUT_PREFIX}${String(record.selected)}`,
        parsed: parsedChoice,
        words: wordCountStored ? record.wordCount as number : typeof record.output === "string" ? record.output.trim().split(/\s+/).length : 0,
        distinct2: distinct2Stored ? record.distinct2 as number : 0,
        seed: typeof record.seed === "number" ? record.seed : 0,
        isInstrument,
        wordCountStored,
        distinct2Stored,
        promptIndex: typeof record.promptIndex === "number" ? record.promptIndex : index,
        speakerName: typeof record.speakerName === "string" ? record.speakerName : undefined,
        turnTitle: typeof record.turnTitle === "string" ? record.turnTitle : undefined,
        routedAgentIDs: Array.isArray(record.routedAgentIDs) ? record.routedAgentIDs.filter((item): item is string => typeof item === "string") : record.routedAgentIDs === null ? null : undefined,
        replicateIndex: typeof record.replicateIndex === "number" ? record.replicateIndex : undefined,
        modelID: typeof record.modelID === "string" ? record.modelID : undefined,
      }];
    } catch { skipped += 1; return []; }
  });
  return { rows, handle: runFile.handle, truncated, skipped };
};

export const loadCosineMatrices = async (run: WorkspaceRun): Promise<CosineMatrix[]> => {
  const matrices: CosineMatrix[] = [];
  for (const runFile of run.files.filter((file) => /^cosine-matrix(?:-L\d+)?\.csv$/i.test(file.name))) {
    const lines = (await (await runFile.handle.getFile()).text()).split(/\r?\n/).filter(Boolean).map(splitCSV);
    if (lines.length < 2 || lines[0].length < 2) continue;
    const hasLayer = lines[0][1]?.toLowerCase() === "layer";
    const start = hasLayer ? 2 : 1;
    const concepts = lines[0].slice(start);
    const rowNames: string[] = [];
    const layers = new Set<number>();
    const values: Array<Array<number | null>> = [];
    for (const row of lines.slice(1)) {
      if (row.length !== concepts.length + start) continue;
      rowNames.push(row[0]);
      const layerValue = hasLayer ? strictNumber(row[1]) : null;
      if (layerValue != null) layers.add(layerValue);
      values.push(row.slice(start).map((cell) => strictNumber(cell)));
    }
    if (rowNames.join("\u001f") !== concepts.join("\u001f") || values.length !== concepts.length) continue;
    matrices.push({ file: runFile.path, concepts, values, layer: layers.size === 1 ? [...layers][0] : null, mixedLayers: layers.size > 1 });
  }
  return matrices;
};

export const validationRows = (report: Record<string, unknown>): ValidationConcept[] => {
  const raw = (report.validation && typeof report.validation === "object" ? report.validation : report.concepts) as Record<string, unknown> | undefined;
  if (!raw || Array.isArray(raw)) return [];
  const lens = report.logitLens && typeof report.logitLens === "object" ? report.logitLens as Record<string, unknown> : {};
  const rows: ValidationConcept[] = [];
  const tokenNames = (block: Record<string, unknown>, key: string) => Array.isArray(block[key]) ? (block[key] as unknown[]).flatMap((item) => item && typeof item === "object" && typeof (item as Record<string, unknown>).token === "string" ? [(item as Record<string, unknown>).token as string] : []) : [];
  for (const name of Object.keys(raw).sort()) {
    const root = raw[name];
    const blocks = root && typeof root === "object" && Array.isArray((root as Record<string, unknown>).depths) && ((root as Record<string, unknown>).depths as unknown[]).length > 1 ? ((root as Record<string, unknown>).depths as unknown[]) : [root];
    for (const item of blocks) {
      const block = item && typeof item === "object" ? item as Record<string, unknown> : {};
      const diagnostics = block.diagnostics && typeof block.diagnostics === "object" ? block.diagnostics as Record<string, unknown> : {};
      const calibration = diagnostics.heldOutCalibration && typeof diagnostics.heldOutCalibration === "object" ? diagnostics.heldOutCalibration as Record<string, unknown> : {};
      const lensRaw = lens[name];
      const lensBlocks = Array.isArray(lensRaw) ? lensRaw : [lensRaw];
      const matchingLens = lensBlocks.find((candidate) => candidate && typeof candidate === "object" && (block.layer == null || (candidate as Record<string, unknown>).layer === block.layer));
      const lensBlock = matchingLens && typeof matchingLens === "object" ? matchingLens as Record<string, unknown> : {};
      rows.push({
        name,
        layer: typeof block.layer === "number" ? block.layer : null,
        scenarios: typeof block.scenarios === "number" ? block.scenarios : typeof block.scenarioCount === "number" ? block.scenarioCount : null,
        accuracy: typeof block.accuracy === "number" ? block.accuracy : typeof block.scenarioAccuracy === "number" ? block.scenarioAccuracy : null,
        calibratedAccuracy: typeof calibration.accuracy === "number" ? calibration.accuracy : null,
        auc: typeof diagnostics.auc === "number" ? diagnostics.auc : null,
        oneSided: typeof diagnostics.oneSidedPredictions === "boolean" ? diagnostics.oneSidedPredictions : null,
        note: typeof item === "string" ? item : typeof block.note === "string" ? block.note : "",
        positiveTokens: tokenNames(lensBlock, "topPositive"),
        negativeTokens: tokenNames(lensBlock, "topNegative"),
      });
    }
  }
  return rows;
};

export const loadSweepRows = async (run: WorkspaceRun): Promise<SweepRow[]> => {
  const runFile = findFile(run.files, "sweep.csv");
  if (!runFile) return [];
  const lines = (await (await runFile.handle.getFile()).text()).split(/\r?\n/).filter((line) => line.trim()).map(splitCSV);
  if (lines.length < 2) return [];
  const headers = lines[0].map((value) => value.trim().toLowerCase());
  const at = (row: string[], key: string) => row[headers.indexOf(key)] ?? "";
  if (!["concept", "layer", "alpha", "markerdensity", "distinct2"].every((key) => headers.includes(key))) return [];
  return lines.slice(1).flatMap((row) => {
    const layer = strictNumber(at(row, "layer"));
    const alpha = strictNumber(at(row, "alpha"));
    const markerDensity = strictNumber(at(row, "markerdensity"));
    const distinct2 = strictNumber(at(row, "distinct2"));
    if (layer == null || alpha == null || markerDensity == null || distinct2 == null || !at(row, "concept")) return [];
    return [{
      concept: at(row, "concept"), layer, alpha, markerDensity, distinct2,
      batteryAccuracy: strictNumber(at(row, "batteryaccuracy")),
      objective: strictNumber(at(row, "objective")),
      // The header row is lowercased above, so the LOOKUPS are lowercase —
      // the engines write these two columns camelCase (`distinct2Ratio`,
      // `lengthInflated`), and asking for that spelling here found nothing:
      // `indexOf` returned -1, `row[-1]` was undefined, and every sweep row
      // in the browser reported a null ratio and an un-inflated length while
      // the CSV beside it said otherwise (review round 9, finding 3).
      distinct2Ratio: strictNumber(at(row, "distinct2ratio")),
      words: strictNumber(at(row, "words")),
      lengthInflated: String(at(row, "lengthinflated") ?? "").toLowerCase() === "true",
    }];
  });
};

export const loadSweepRecommendations = async (run: WorkspaceRun): Promise<SweepRecommendation[]> => {
  const runFile = findFile(run.files, "recommendations.json");
  if (!runFile) return [];
  try {
    const root = recordValue(JSON.parse(await (await runFile.handle.getFile()).text()));
    return Object.entries(root).map(([concept, raw]) => {
      if (typeof raw === "string") return { concept, failure: raw, layer: null, alpha: null, metric: "markerDensity", metrics: {}, capabilityTolerance: .15, coherenceFloor: .45, coherenceRatioToBaseline: null, matchedNormRandomMargin: null, devPromptsHash: "", batteryHash: "", sweepRun: "" };
      const item = recordValue(raw);
      const cell = recordValue(item.winningCell);
      const criterion = recordValue(item.criterion);
      const objective = recordValue(criterion.objective);
      const constraints = recordValue(criterion.constraints);
      const controls = recordValue(criterion.controls);
      const metrics = Object.fromEntries(Object.entries(recordValue(item.metrics)).flatMap(([key, value]) => typeof value === "number" && Number.isFinite(value) ? [[key, value]] : []));
      return {
        concept, failure: "",
        layer: typeof cell.layer === "number" ? cell.layer : null,
        alpha: typeof cell.alpha === "number" ? cell.alpha : null,
        metric: typeof objective.metric === "string" ? objective.metric : "markerDensity",
        metrics,
        capabilityTolerance: typeof constraints.capabilityTolerance === "number" ? constraints.capabilityTolerance : .15,
        // Which coherence rule the criterion declared is decided by the PRESENCE of
        // the relative fields — never their values — exactly as both engines decide it.
        // Absent = the legacy absolute rule at coherenceFloor, forever.
        coherenceFloor: typeof constraints.coherenceAbsoluteBackstop === "number" ? constraints.coherenceAbsoluteBackstop : typeof constraints.coherenceFloor === "number" ? constraints.coherenceFloor : .45,
        coherenceRatioToBaseline: typeof constraints.coherenceRatioToBaseline === "number" ? constraints.coherenceRatioToBaseline : typeof constraints.coherenceAbsoluteBackstop === "number" ? .85 : null,
        matchedNormRandomMargin: typeof controls.matchedNormRandomMargin === "number" ? controls.matchedNormRandomMargin : null,
        devPromptsHash: typeof item.devPromptsHash === "string" ? item.devPromptsHash : "",
        batteryHash: typeof item.batteryHash === "string" ? item.batteryHash : "",
        sweepRun: typeof item.sweepRun === "string" ? item.sweepRun : "",
      };
    });
  } catch { return []; }
};

export const loadPanelEffects = async (run: WorkspaceRun): Promise<PanelEffect[]> => {
  const runFile = findFile(run.files, "panel-effects.csv");
  if (!runFile) return [];
  const lines = (await (await runFile.handle.getFile()).text()).split(/\r?\n/).filter((line) => line.trim()).map(splitCSV);
  if (lines.length < 2) return [];
  const headers = lines[0].map((value) => value.trim().toLowerCase());
  const at = (row: string[], key: string) => row[headers.indexOf(key)] ?? "";
  const maybeNumber = (row: string[], key: string) => at(row, key) !== "" && Number.isFinite(Number(at(row, key))) ? Number(at(row, key)) : null;
  if (!headers.includes("endpoint")) return [];
  return lines.slice(1).flatMap((row) => at(row, "endpoint") ? [{
    endpoint: at(row, "endpoint"),
    direct: maybeNumber(row, "direct"), directN: maybeNumber(row, "directn") ?? 0,
    spillover: maybeNumber(row, "spillover"), spilloverN: maybeNumber(row, "spillovern") ?? 0,
    group: maybeNumber(row, "group"), groupN: maybeNumber(row, "groupn") ?? 0,
    transmissionRatio: maybeNumber(row, "transmissionratio"), amplification: maybeNumber(row, "amplification"),
    droppedTurns: maybeNumber(row, "droppedturns") ?? 0,
  }] : []);
};

export const hydrateRun = async (run: WorkspaceRun): Promise<WorkspaceRun> => {
  const [effectRows, generationData, cosineMatrices, sweepRows, sweepRecommendations, panelEffects] = await Promise.all([loadEffects(run), loadGenerations(run), loadCosineMatrices(run), loadSweepRows(run), loadSweepRecommendations(run), loadPanelEffects(run)]);
  return { ...run, effectRows, generationRows: generationData.rows, generationFile: generationData.handle, previewTruncated: generationData.truncated, skippedGenerationLines: generationData.skipped, cosineMatrices, validationConcepts: validationRows(run.validationReport), sweepRows, sweepRecommendations, panelEffects };
};
