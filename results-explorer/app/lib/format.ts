// Display formatting shared across views. Nothing here reads a file or
// derives a statistic — it only renders numbers already parsed elsewhere.

import type { SweepRow } from "./types";

export const fmt = (value: number, digits = 2) => `${value > 0 ? "+" : ""}${value.toFixed(digits)}`;

export const sweepMetricValue = (row: SweepRow, metric: string) => metric === "distinct2" ? row.distinct2 : metric === "batteryAccuracy" ? row.batteryAccuracy : metric === "markerDensity" ? row.markerDensity : row.objective;
export const metricLabel = (metric: string) => ({ markerDensity: "Marker density", judgeScore: "Judge score", logprobShift: "Logprob shift", distinct2: "Distinct-2", batteryAccuracy: "Battery accuracy" }[metric] ?? metric.replaceAll("_", " "));
export const shortHash = (hash: string) => hash ? `${hash.slice(0, 10)}…${hash.slice(-6)}` : "Not stamped";
