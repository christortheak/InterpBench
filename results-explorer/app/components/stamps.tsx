"use client";

// The stamps a reader needs in order to judge a result, laid out. Every word
// here comes from lib/ (which reads the run's files); these components only
// arrange it.

import { adjudicationSentence, analysisStampsOf, exclusionsSentence, rescueSentence, unitSentence, type UnitOfAnalysis } from "../lib/analysisStamps";
import { freezeDetails, freezeLabel, freezeOf, freezeTone, runsBeforeFreeze } from "../lib/freeze";
import { cellsOverThreshold, cutOffCells, parseTruncation, percentText, thresholdSentence, truncationSentence } from "../lib/runReport";
import { RESPONSE_UNIT_EXPLANATION, unitOf, unsettledUnitSentence, type PairedItems } from "../lib/effectUnits";
import type { Effect, RunFile, WorkspaceRun } from "../lib/types";
import { Badge } from "./ui";

/// The study's freeze state as one badge, for the run's header. The tooltip
/// carries the same sentences `FreezeNotice` prints.
export function FreezeBadge({ run }: { run: WorkspaceRun }) {
  const stamp = freezeOf(run);
  const before = runsBeforeFreeze(run);
  const details = freezeDetails(stamp, before);
  return (
    <span className="freeze-badge" title={details.join(" ") || (stamp.frozenAt ? `The study's settings were locked on ${stamp.frozenAt}, before this run.` : "The study's settings were locked before this run.")}>
      <Badge tone={freezeTone(stamp, before)}>{freezeLabel(stamp)}</Badge>
    </span>
  );
}

/// The same facts in full sentences, for the overview. Renders nothing for
/// a plainly frozen study with no check skipped and no exemption: the badge
/// already says "Frozen", and a notice with nothing to notice trains the
/// reader to skip notices.
export function FreezeNotice({ run }: { run: WorkspaceRun }) {
  const stamp = freezeOf(run);
  const before = runsBeforeFreeze(run);
  const details = freezeDetails(stamp, before);
  if (!details.length) return null;
  const caution = freezeTone(stamp, before) === "warn" || stamp.batteryNotApplied.length > 0;
  return (
    <div className={`notice freeze-notice ${caution ? "" : "local-notice"}`} role="note">
      <span className="notice-icon">{caution ? "!" : "i"}</span>
      <p>
        <strong>{freezeLabel(stamp)}.</strong>{" "}
        {details.join(" ")}
      </p>
    </div>
  );
}

/// "What the interval means", in plain words, beside a real run's effects.
///
/// This card used to exist only on the invented demo page, where it quoted
/// a resample count and a seed no run file states. Here it says only what
/// holds for every effect table both engines write (the estimate is the mean
/// of paired differences; the interval is a percentile bootstrap over them;
/// the raw p is a Wilcoxon signed-rank test) and takes the two things that
/// DO vary from the run itself: its unit of analysis, and the correction its
/// table names. A run that stamps no unit is read from its records
/// (`rows` carry their settled units): rows that paired responses are said
/// to be what they are.
export function IntervalNote({ unit, corrections, rows = [], records = null }: { unit: UnitOfAnalysis | null; corrections: string[]; rows?: Effect[]; records?: PairedItems | null }) {
  const one = unit?.unit === "transcript" ? "transcript" : "prompt item";
  const many = `${one}s`;
  const pooled = rows.filter((row) => row.stratifyBy === "pooled");
  const responses = pooled.some((row) => unitOf(row).unit === "response");
  const unknown = pooled.some((row) => unitOf(row).unit === "unknown");
  return (
    <section className="card methods-card interval-note" aria-label="What the interval means">
      <span className="section-number">HOW TO READ THE NUMBERS</span>
      <h2>What the interval means</h2>
      <p>Each pooled row compares one condition with the baseline on the <strong>same {many}</strong>. For every {one}, the engine takes the condition&rsquo;s value minus the baseline&rsquo;s value. The estimate is the average of those differences.</p>
      <p>The 95% interval is the range the engine found for that average by drawing the {many} again, many times over, from the ones in the study. It shows how much the average depends on which {many} the study happened to use. It does not show how much one answer differs from the next.</p>
      {responses && <p><strong>Rows of paired responses are not read this way.</strong> {RESPONSE_UNIT_EXPLANATION}</p>}
      {unknown && <p>For some rows, the unit of the pairs is not established, so this reading may not hold for them.</p>}
      <div className="method-item"><span>Unit</span><small>{unit ? `One ${unit.unit}, as this run stamps it (${unit.source}).` : unsettledUnitSentence(rows, records)}</small></div>
      <div className="method-item"><span>Raw p</span><small>A Wilcoxon signed-rank test on the same differences: a second check that does not assume they follow a bell curve.</small></div>
      <div className="method-item"><span>Adjusted p</span><small>{corrections.length ? `The raw p after the ${corrections.join(" / ")} correction, which allows for several outcomes being tested at once.` : "This table names no correction, so the explorer does not say which one was used."}</small></div>
      <footer>An interval that includes zero means the data are consistent with no difference. The explorer shows every number here as the engine stored it and works out none of them.</footer>
    </section>
  );
}

/// How many generations were cut off at the length limit, from report.json's
/// `truncation` block. A cut-off generation is not a short answer: its text
/// stops before the model finished, so anything read from it (a choice, a
/// number, a judge's verdict) may be read from half an answer. The engines
/// write this block for every run, whether or not the study set a limit;
/// the explorer used to show none of it.
///
/// Renders nothing for a run whose report has no such block.
export function TruncationCard({ run }: { run: WorkspaceRun }) {
  const block = parseTruncation(run.report);
  if (!block) return null;
  const cells = cutOffCells(block);
  const over = new Set(cellsOverThreshold(block).map((cell) => `${cell.condition}\u001f${cell.promptID}`));
  const shown = cells.slice(0, 12);
  const anyCut = (block.lengthStopped ?? 0) > 0;
  return (
    <section className="card truncation-card" aria-label="Generations cut off at the length limit">
      <header className="section-header">
        <div><span className="section-number">CUT-OFF GENERATIONS</span><h2>{anyCut ? "Some generations stop early" : "No generation was cut off"}</h2></div>
        {over.size > 0 ? <Badge tone="warn">{over.size} over the study&rsquo;s limit</Badge> : anyCut ? <Badge tone="warn">Read with care</Badge> : <Badge tone="good">None</Badge>}
      </header>
      <p className="truncation-summary">{truncationSentence(block)} {thresholdSentence(block)}</p>
      {shown.length > 0 && (
        <div className="raw-table-scroll">
          <table className="raw-table">
            <thead><tr><th>Condition</th><th>Item</th><th>Cut off</th><th>Of</th><th>Share</th><th>Cut off while reasoning</th>{block.threshold !== null && <th>Against the limit</th>}</tr></thead>
            <tbody>
              {shown.map((cell) => (
                <tr key={`${cell.condition}-${cell.promptID}`}>
                  <td>{cell.condition}</td>
                  <td>{cell.promptID}</td>
                  <td>{cell.lengthStopped ?? "—"}</td>
                  <td>{cell.classified ?? "—"}</td>
                  <td>{percentText(cell.lengthStoppedFraction)}</td>
                  <td>{cell.lengthStoppedInReasoning ?? "—"}</td>
                  {block.threshold !== null && <td>{over.has(`${cell.condition}\u001f${cell.promptID}`) ? "Over" : "Within"}</td>}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="table-note">
        Stored values, read from <code>report.json</code>.
        {cells.length > shown.length ? ` Showing the ${shown.length} most affected of ${cells.length} condition-and-item pairs with a cut-off generation; the rest are in the report.` : ""}
        {anyCut ? " Each generation's own ending is in generations.jsonl, under finishReason." : ""}
      </p>
    </section>
  );
}

export type NoncompliantItem ={ judge: string; condition: string; promptID: string; sampleIndex: number; reason: string };

/// Rows where a judge (or coder) ANSWERED but gave nothing the engine could
/// use. The engines keep each one as a row, with the judge's words, and
/// leave it out of every tally. The explorer used to show such a row as
/// "Not stamped" (judging) or "No codes recorded" (coding), which read as a
/// gap in the file rather than as something that happened.
///
/// `stamped` is the report's own count when it gives one; `items` are the
/// rows found in the loaded file. Renders nothing when both are empty.
export function NoncompliantNotice({ items, stamped, kind }: { items: NoncompliantItem[]; stamped: number | null; kind: "judgment" | "coding" }) {
  const found = items.length;
  const total = stamped ?? found;
  if (!total) return null;
  const judging = kind === "judgment";
  const headline = judging
    ? `${total} judgment${total === 1 ? " has" : "s have"} no verdict.`
    : `${total} coding${total === 1 ? " has" : "s have"} no codes.`;
  return (
    <div className="card judged-alert alert-warn noncompliant-notice">
      <span>!</span>
      <div>
        <strong>{headline}</strong>
        <p>
          {judging
            ? "The judge answered, but not with a verdict the engine could use. Each of these is kept as a row for review, with what the judge said, and is left out of every tally and agreement figure."
            : "The coder answered, but not with codes the engine could use. Each of these is kept as a row for review, with what the coder said, and is left out of every aggregate and agreement figure."}
          {stamped !== null && stamped !== found ? ` The report counts ${stamped}; ${found} ${found === 1 ? "is" : "are"} among the rows loaded here.` : ""}
        </p>
        {found > 0 && (
          <details>
            <summary>Show {found === 1 ? "it" : `the ${found} rows`}</summary>
            <ul>
              {items.slice(0, 200).map((item, index) => (
                <li key={`${item.judge}-${item.condition}-${item.promptID}-${item.sampleIndex}-${index}`}>
                  <strong>{item.judge}</strong> · {item.condition} · {item.promptID} · sample {item.sampleIndex}
                  {item.reason ? <>: <q>{item.reason}</q></> : ": no reason was recorded."}
                </li>
              ))}
            </ul>
            {found > 200 && <p>Showing the first 200 of {found}. The rest are in the run&rsquo;s file.</p>}
          </details>
        )}
      </div>
    </div>
  );
}

/// What the analysis did to the records before it estimated anything, shown
/// BESIDE the effects it produced: the unit of analysis, declared exclusions,
/// endpoint rescue, and adjudication. Each row is one stamp file from the
/// run directory; a stamp the run does not carry says so in words rather
/// than disappearing, so "none" is a statement and not a gap.
export function AnalysisStampsCard({ run, onOpenFile }: { run: WorkspaceRun; onOpenFile: (file: RunFile) => void }) {
  const stamps = analysisStampsOf(run);
  const file = (name: string) => run.files.find((candidate) => candidate.path === name) ?? null;
  const open = (name: string) => {
    const found = file(name);
    return found ? <button className="quiet-link" onClick={() => onOpenFile(found)}>{name}</button> : null;
  };
  const exclusions = stamps.exclusions;
  const conditions = exclusions ? Object.keys(exclusions.consideredN).sort() : [];
  return (
    <section className="card analysis-stamps-card" aria-label="What this analysis did to the records">
      <header className="section-header">
        <div><span className="section-number">BEFORE THE ESTIMATES</span><h2>What this analysis did to the records</h2></div>
      </header>
      <dl className="analysis-stamps">
        <div>
          <dt>Unit of analysis</dt>
          <dd>
            <p>{unitSentence(stamps.unit, run.effectRows, run.effectUnitRecords ?? null)}</p>
            {stamps.unit && <small>Stamped in {stamps.unit.source}. {open(stamps.unit.source)}</small>}
          </dd>
        </div>
        <div>
          <dt>Exclusions</dt>
          <dd>
            <p>{exclusionsSentence(exclusions)}</p>
            {exclusions && exclusions.rules.length > 0 && (
              <ul>{exclusions.rules.map((rule) => <li key={rule.rule}><strong>{rule.rule}</strong>{rule.description ? `: ${rule.description}` : ""}</li>)}</ul>
            )}
            {conditions.length > 0 && (
              <p className="analysis-stamps-counts">Records left for each condition: {conditions.map((condition) => `${condition} ${exclusions!.survivingN[condition] ?? "—"} of ${exclusions!.consideredN[condition]}`).join(", ")}.</p>
            )}
            {exclusions && <small>Stamped in {exclusions.source}. {open(exclusions.source)}</small>}
          </dd>
        </div>
        <div>
          <dt>Endpoint rescue</dt>
          <dd>
            <p>{rescueSentence(stamps.rescue)}</p>
            {stamps.rescue && <small>Stamped in endpoint-reparse.json. {open("endpoint-reparse.json")}</small>}
          </dd>
        </div>
        <div>
          <dt>Adjudication</dt>
          <dd>
            <p>{adjudicationSentence(stamps.adjudication)}</p>
            {stamps.adjudication && <small>Stamped in {stamps.adjudication.source}. {open(stamps.adjudication.source)} {open("adjudication-divergence.csv")}</small>}
          </dd>
        </div>
        {stamps.epochUnverified && (
          <div>
            <dt>Study fingerprint</dt>
            <dd><p>Not verified. The run this analysis read carried no study fingerprint, so the analysis could not confirm that it belongs to this version of the study, and it went ahead anyway.</p></dd>
          </div>
        )}
        {stamps.measurementDrift && (
          <div>
            <dt>Measurement settings</dt>
            <dd><p>These settings differed from the ones the run was made under, and the analysis used the current ones: <code>{stamps.measurementDrift}</code></p></dd>
          </div>
        )}
      </dl>
    </section>
  );
}
