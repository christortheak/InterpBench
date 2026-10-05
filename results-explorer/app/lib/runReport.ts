// Two things report.json says about a run that the overview used to leave
// out: a condition that FAILED, and how many generations were CUT OFF at the
// length limit. Both are read as written; nothing is counted here.

import { recordValue, textValue } from "./discovery";

const count = (value: unknown): number | null =>
  typeof value === "number" && Number.isFinite(value) ? value : null;

export type ConditionError = { condition: string; error: string };

/// Conditions whose block carries an `error`: the condition produced no
/// generations, and the engine recorded why instead of a row of numbers.
export const conditionErrors = (report: Record<string, unknown>): ConditionError[] =>
  Object.entries(recordValue(report.conditions))
    .flatMap(([condition, block]): ConditionError[] => {
      const error = recordValue(block).error;
      const text = typeof error === "string" ? error : error === undefined || error === null ? "" : JSON.stringify(error);
      return text ? [{ condition, error: text }] : [];
    })
    .sort((left, right) => left.condition.localeCompare(right.condition));

export type TruncationCell = {
  condition: string;
  promptID: string;
  classified: number | null;
  lengthStopped: number | null;
  lengthStoppedInReasoning: number | null;
  lengthStoppedFraction: number | null;
};

export type TruncationBlock = {
  /// The ceiling the study declared for one condition on one item, as a
  /// fraction; null when it declared none.
  threshold: number | null;
  /// Generations that recorded how they ended.
  classified: number | null;
  /// Of those, how many stopped because they hit the length limit.
  lengthStopped: number | null;
  /// Of the cut-off ones, how many were still in the model's reasoning, so
  /// they never reached an answer.
  lengthStoppedInReasoning: number | null;
  lengthStoppedFraction: number | null;
  cells: TruncationCell[];
};

/// The report's `truncation` block, or null when the run wrote none (a run
/// from before the block existed).
export const parseTruncation = (report: Record<string, unknown>): TruncationBlock | null => {
  const raw = recordValue(report.truncation);
  if (!Object.keys(raw).length) return null;
  return {
    threshold: count(raw.threshold),
    classified: count(raw.classified),
    lengthStopped: count(raw.lengthStopped),
    lengthStoppedInReasoning: count(raw.lengthStoppedInReasoning),
    lengthStoppedFraction: count(raw.lengthStoppedFraction),
    cells: (Array.isArray(raw.cells) ? raw.cells : []).flatMap((item): TruncationCell[] => {
      const cell = recordValue(item);
      const condition = textValue(cell, "condition");
      const promptID = typeof cell.promptID === "string" ? cell.promptID : cell.promptID === undefined || cell.promptID === null ? "" : String(cell.promptID);
      if (!condition && !promptID) return [];
      return [{
        condition, promptID,
        classified: count(cell.classified),
        lengthStopped: count(cell.lengthStopped),
        lengthStoppedInReasoning: count(cell.lengthStoppedInReasoning),
        lengthStoppedFraction: count(cell.lengthStoppedFraction),
      }];
    }),
  };
};

/// The cells where at least one generation was cut off, worst first. Order
/// only; every number is the stored one.
export const cutOffCells = (block: TruncationBlock): TruncationCell[] =>
  block.cells
    .filter((cell) => (cell.lengthStopped ?? 0) > 0)
    .sort((left, right) => (right.lengthStoppedFraction ?? 0) - (left.lengthStoppedFraction ?? 0)
      || (right.lengthStopped ?? 0) - (left.lengthStopped ?? 0)
      || left.condition.localeCompare(right.condition) || left.promptID.localeCompare(right.promptID));

/// Cells over the study's declared ceiling. Empty when none was declared.
export const cellsOverThreshold = (block: TruncationBlock): TruncationCell[] =>
  block.threshold === null ? [] : cutOffCells(block).filter((cell) => (cell.lengthStoppedFraction ?? 0) > block.threshold!);

export const percentText = (fraction: number | null) =>
  fraction === null ? "—" : `${(fraction * 100).toFixed(fraction > 0 && fraction < 0.1 ? 1 : 0)}%`;

/// The headline sentence for the block.
export const truncationSentence = (block: TruncationBlock): string => {
  if (!block.classified) return "No generation in this run recorded how it ended, so the explorer cannot say whether any was cut off.";
  if (block.lengthStopped === null) return `${block.classified} generations recorded how they ended. The report does not say how many were cut off.`;
  if (block.lengthStopped === 0) return `None of the ${block.classified} generations that recorded how they ended was cut off at the length limit.`;
  const reasoning = block.lengthStoppedInReasoning
    ? ` ${block.lengthStoppedInReasoning === 1 ? "1 of those was" : `${block.lengthStoppedInReasoning} of those were`} cut off while the model was still reasoning, before it gave any answer.`
    : "";
  return `${block.lengthStopped} of ${block.classified} generations (${percentText(block.lengthStoppedFraction)}) were cut off at the length limit, so their text stops early.${reasoning}`;
};

export const thresholdSentence = (block: TruncationBlock): string =>
  block.threshold === null
    ? "The study set no limit on how many may be cut off."
    : `The study allowed at most ${percentText(block.threshold)} for any one condition on any one item.`;
