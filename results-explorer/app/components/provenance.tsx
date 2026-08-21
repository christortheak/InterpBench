"use client";

/// The derived-data vocabulary (docs/RESULTS-EXPLORER-UPGRADE-PLAN.md).
/// Every number the explorer shows is one of three kinds, visually distinct
/// everywhere:
/// - STORED: read from a run artifact. Plain rendering; provenance = the
///   file name. No badge.
/// - DERIVED (viewer): computed by the explorer from stored records (joins,
///   deltas, counts, distributions). `DerivedBadge` — the tooltip states
///   the formula and inputs.
/// - HEURISTIC (viewer): derived AND resting on a convention the data does
///   not declare (id-suffix counterbalance pairing, saturation thresholds).
///   `HeuristicBadge` — the tooltip names the assumption.
/// Hard rules: the viewer never computes CIs, p-values, kappa, or
/// corrections (those come from engine reports only); a derived number
/// never appears unbadged beside a stored one; exports stamp each column's
/// kind.

export function DerivedBadge({ formula }: { formula: string }) {
  return (
    <span
      className="badge badge-derived"
      title={`Derived in the viewer: ${formula}. Not an engine artifact.`}
    >
      derived
    </span>
  );
}

export function HeuristicBadge({ assumption }: { assumption: string }) {
  return (
    <span
      className="badge badge-heuristic"
      title={`Viewer heuristic — assumes: ${assumption}. Not declared by the data.`}
    >
      heuristic
    </span>
  );
}

/// One legend per view that mixes kinds. Render it once, near the top.
export function ProvenanceLegend() {
  return (
    <p className="provenance-legend">
      <span>Unmarked values are read from run artifacts.</span>
      <span className="badge badge-derived">derived</span>
      <span>= computed by the viewer from stored records.</span>
      <span className="badge badge-heuristic">heuristic</span>
      <span>= viewer convention the data does not declare.</span>
    </p>
  );
}
