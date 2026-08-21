"use client";

import { useState } from "react";
import { Badge } from "../components/ui";
import { findFile } from "../lib/discovery";
import type { RunFile, WorkspaceRun } from "../lib/types";

export function ConceptEvidenceView({ run, onOpenFile }: { run: WorkspaceRun | null; onOpenFile: (file: RunFile) => void }) {
  // The reader's matrix pick is stored WITH the run it was made in, so
  // selecting another run falls back to that run's first matrix without an
  // effect — and without the one render in which the previous run's choice
  // was still live over the new run's data.
  const [matrixChoice, setMatrixChoice] = useState<{ runKey: string; file: string } | null>(null);
  const matrixFile = matrixChoice?.runKey === (run?.key ?? "") ? matrixChoice.file : "";
  const matrix = run?.cosineMatrices.find((candidate) => candidate.file === matrixFile) ?? run?.cosineMatrices[0] ?? null;
  if (!run) return <div className="view-enter inner-view"><header className="page-title"><div><span className="section-number">CONCEPT STUDIES</span><h1>Concept evidence</h1><p>Select a local validate run to inspect convergent accuracy, logit-lens tokens, and cross-concept geometry.</p></div></header><div className="card no-run-card"><span>⌂</span><h2>No local run selected</h2><p>Choose a workspace and run from the sidebar.</p></div></div>;
  const validationFile = findFile(run.files, "validation-report.json");
  return (
    <div className="view-enter inner-view concept-view">
      <header className="page-title">
        <div><span className="section-number">{run.validationConcepts.length} VALIDATION ROWS · {run.cosineMatrices.length} MATRICES</span><h1>Concept evidence</h1><p>Convergent validity and discriminant geometry are kept separate: a direction should transfer on held-out scenarios without collapsing into another concept.</p></div>
        {validationFile && <button className="secondary" onClick={() => onOpenFile(validationFile)}>Open validation report</button>}
      </header>

      <section className="card validation-card">
        <header className="section-header"><div><span className="section-number">CONVERGENT VALIDITY</span><h2>Held-out concept performance</h2></div><span className="muted">Chance line: 50%</span></header>
        {run.validationConcepts.length ? <div className="validation-table">
          <div className="validation-head"><span>Concept</span><span>Layer</span><span>Scenarios</span><span>Transfer</span><span>Calibrated</span><span>AUC</span><span>Read</span></div>
          {run.validationConcepts.map((row) => {
            const headline = row.calibratedAccuracy ?? row.accuracy;
            return <div className="validation-row" key={`${row.name}-${row.layer}`}>
              <div><strong>{row.name}</strong>{row.note && <small>{row.note}</small>}</div>
              <span>{row.layer ?? "—"}</span><span>{row.scenarios ?? "—"}</span>
              <strong className={row.accuracy != null && row.accuracy > .5 ? "valid-good" : "valid-warn"}>{row.accuracy == null ? "—" : `${(row.accuracy * 100).toFixed(0)}%`}{row.oneSided ? " ⚠" : ""}</strong>
              <strong className={row.calibratedAccuracy != null && row.calibratedAccuracy > .5 ? "valid-good" : ""}>{row.calibratedAccuracy == null ? "—" : `${(row.calibratedAccuracy * 100).toFixed(0)}%`}</strong>
              <span>{row.auc == null ? "—" : row.auc.toFixed(2)}</span>
              <Badge tone={headline != null && headline > .5 ? "good" : headline == null ? "warn" : "neutral"}>{headline == null ? "Not run" : headline > .5 ? "Above chance" : "At chance"}</Badge>
            </div>;
          })}
        </div> : <div className="artifact-empty"><span>∅</span><p>No structured validation concept block was found. Open the raw report or select a validate run.</p></div>}
      </section>

      <section className="card cosine-card">
        <header className="section-header">
          <div><span className="section-number">DISCRIMINANT VALIDITY</span><h2>Cross-concept cosine similarity</h2></div>
          {run.cosineMatrices.length > 1 && <select value={matrix?.file ?? ""} onChange={(event) => setMatrixChoice({ runKey: run.key, file: event.target.value })}>{run.cosineMatrices.map((item) => <option value={item.file} key={item.file}>{item.layer == null ? item.file : `Layer ${item.layer} · ${item.file}`}</option>)}</select>}
        </header>
        {matrix ? <>
          <div className={`matrix-note ${matrix.mixedLayers ? "matrix-warning" : ""}`}><span>{matrix.mixedLayers ? "!" : "L"}</span><p>{matrix.mixedLayers ? "Rows record different layers; this matrix is asymmetric and has no defined scientific reading." : matrix.layer == null ? "Layer was not recorded. The matrix predates depth stamping, so cross-run comparison is unsafe." : `Every cell was measured at layer ${matrix.layer}. Off-diagonal |cosine| > 0.50 is flagged.`}</p></div>
          <div className="cosine-scroll"><table className="cosine-table"><thead><tr><th>Concept</th>{matrix.concepts.map((concept) => <th key={concept}>{concept}</th>)}</tr></thead><tbody>{matrix.concepts.map((concept, row) => <tr key={concept}><th>{concept}</th>{matrix.values[row].map((value, column) => {
            const magnitude = value == null ? 0 : Math.min(1, Math.abs(value));
            const flagged = row !== column && value != null && Math.abs(value) > .5;
            const background = row === column ? "#e8e9e3" : value == null ? "transparent" : value >= 0 ? `rgba(46,91,255,${0.07 + magnitude * .48})` : `rgba(214,95,56,${0.07 + magnitude * .48})`;
            return <td key={`${row}-${column}`} className={flagged ? "cosine-flagged" : ""} style={{ background }} title={`${concept} × ${matrix.concepts[column]}: ${value == null ? "nan" : value.toFixed(4)}`}>{value == null ? "nan" : value.toFixed(2)}</td>;
          })}</tr>)}</tbody></table></div>
          <div className="matrix-legend"><span><i className="negative" />Negative</span><span><i />Near zero</span><span><i className="positive" />Positive</span><span><b />Flagged |cos| &gt; .50</span></div>
        </> : <div className="artifact-empty"><span>∅</span><p>No readable cosine-matrix CSV is present in this run.</p></div>}
      </section>

      {run.validationConcepts.some((row) => row.positiveTokens.length || row.negativeTokens.length) && <section className="card lens-card"><header className="section-header"><div><span className="section-number">LOGIT LENS DIAGNOSTIC</span><h2>Tokens favored by each direction</h2></div></header><div className="lens-grid">{run.validationConcepts.filter((row) => row.positiveTokens.length || row.negativeTokens.length).map((row) => <div key={`${row.name}-${row.layer}`}><header><strong>{row.name}</strong><span>{row.layer == null ? "" : `L${row.layer}`}</span></header><p><b>+</b>{row.positiveTokens.join("  ") || "—"}</p><p><b>−</b>{row.negativeTokens.join("  ") || "—"}</p></div>)}</div></section>}
    </div>
  );
}
