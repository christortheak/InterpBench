import { describe, expect, it, vi } from "vitest";

vi.mock("react/jsx-dev-runtime", async (original) => (await import("./support/capture")).wrapJSXRuntime(await original()));
vi.mock("react/jsx-runtime", async (original) => (await import("./support/capture")).wrapJSXRuntime(await original()));

import { NoncompliantNotice } from "../app/components/stamps";
import { codesSummary, codingDisagreements, parseCodingReport, parseCodingRows } from "../app/lib/codingdata";
import {
  confidenceHistogram, disagreementCells, judgeTallies, judgmentCells, noncompliantRows, normalizeOutcome,
  OUTCOME_LABEL, parseJudgeReport, parseJudgmentRows,
} from "../app/lib/judged";
import { DisagreementDetail } from "../app/views/JudgedEvaluation";
import { render } from "./support/capture";

// A judge that ANSWERS but gives no usable verdict does not fail the run:
// both engines write the pair as a row with `outcome: null`,
// `noncompliant: true`, and the judge's words in `noncomplianceReason`, and
// leave it out of every tally. The explorer showed that row as "Not
// stamped". These rows are the engines' own shape (paired_judge.py and the
// Mac engine's NoncompliantJudgmentRecord write the same keys).

const verdict = (judge: string, outcome: string, promptID = "item-1") => JSON.stringify({
  promptID, sampleIndex: 0, condition: "steered", baselineSeed: 1, variantSeed: 2, baselineWas: "A",
  outcome, confidence: 0.8, judge, judgment: { winner: outcome === "tie" ? "tie" : "A", brief_reason: "Clearer reasoning." },
});
const noVerdict = (judge: string, promptID = "item-1") => JSON.stringify({
  promptID, sampleIndex: 0, condition: "steered", baselineSeed: 1, variantSeed: 2, baselineWas: "A",
  outcome: null, noncompliant: true, noncomplianceReason: "I can't compare these two responses.", judgment: null, judge,
});

describe("a judgment row with no verdict", () => {
  const { rows, skipped } = parseJudgmentRows([verdict("judge-a", "variant"), noVerdict("judge-b"), verdict("judge-a", "baseline", "item-2"), verdict("judge-b", "variant", "item-2")].join("\n"));

  it("is read as what it is, with the judge's own words kept", () => {
    expect(skipped).toBe(0);
    const row = rows[1];
    expect(row.outcome).toBe("noncompliant");
    expect(row.noncompliant).toBe(true);
    expect(row.noncomplianceReason).toBe("I can't compare these two responses.");
    expect(noncompliantRows(rows)).toEqual([row]);
  });

  it("is labelled 'No verdict' — not 'Not stamped', which stays for a row with no outcome at all", () => {
    expect(OUTCOME_LABEL.noncompliant).toBe("No verdict");
    expect(OUTCOME_LABEL.unknown).toBe("Not stamped");
    expect(normalizeOutcome(null)).toBe("unknown");
    expect(normalizeOutcome("noncompliant")).toBe("noncompliant");
    // The flag decides, not the null outcome beside it.
    expect(parseJudgmentRows(JSON.stringify({ promptID: "p", condition: "c", outcome: null, judge: "j" })).rows[0].outcome).toBe("unknown");
  });

  it("is counted beside the tallies, not inside them", () => {
    const tallies = judgeTallies(rows);
    const b = tallies.find((tally) => tally.judge === "judge-b")!;
    expect(b).toMatchObject({ variantWins: 1, baselineWins: 0, ties: 0, unknown: 0, noncompliant: 1, n: 1 });
    const a = tallies.find((tally) => tally.judge === "judge-a")!;
    expect(a).toMatchObject({ variantWins: 1, baselineWins: 1, noncompliant: 0, n: 2 });
  });

  it("cannot create a disagreement: giving no verdict is not disagreeing", () => {
    const cells = judgmentCells(rows);
    const first = cells.find((cell) => cell.promptID === "item-1")!;
    expect(first.verdicts).toEqual([{ judge: "judge-a", outcome: "variant" }, { judge: "judge-b", outcome: "noncompliant" }]);
    expect(first.disagrees).toBe(false);
    // The real split, on item-2, is still found.
    expect(disagreementCells(rows).map((cell) => cell.promptID)).toEqual(["item-2"]);
  });

  it("has no confidence to be missing", () => {
    expect(confidenceHistogram(rows)).toMatchObject({ counted: 3, missing: 0 });
  });

  it("is read from the report's own count when the report gives one", () => {
    expect(parseJudgeReport({ present: true, raw: { noncompliantJudgments: 4 }, error: "", file: null }).noncompliantJudgments).toBe(4);
    expect(parseJudgeReport({ present: true, raw: {}, error: "", file: null }).noncompliantJudgments).toBeNull();
  });
});

describe("the judged view's wording", () => {
  const { rows } = parseJudgmentRows([verdict("judge-a", "variant"), verdict("judge-b", "baseline"), noVerdict("judge-c")].join("\n"));
  const [cell] = judgmentCells(rows);

  it("the verdict pane says 'No verdict' and quotes the judge, where it used to say 'Not stamped'", () => {
    const page = render(<DisagreementDetail cell={cell} sources={null} sourceLoading={false} sourceRun={null} sourceRunName="" />);
    expect(page.text).toContain("No verdict");
    expect(page.text).toContain("the judge gave no usable verdict");
    expect(page.text).toContain("What the judge said: I can't compare these two responses.");
    expect(page.text).not.toContain("Not stamped");
    expect(page.text).not.toContain("no winner stamped");
    expect(page.html).toContain("verdict-noncompliant");
  });

  it("the notice counts them, says what they are, and lists each with its reason", () => {
    const items = noncompliantRows(rows).map((row) => ({ judge: row.judge, condition: row.condition, promptID: row.promptID, sampleIndex: row.sampleIndex, reason: row.noncomplianceReason }));
    const page = render(<NoncompliantNotice items={items} stamped={null} kind="judgment" />);
    expect(page.text).toContain("1 judgment has no verdict.");
    expect(page.text).toContain("left out of every tally and agreement figure");
    expect(page.text).toContain("judge-c · steered · item-1 · sample 0: I can't compare these two responses.");
  });

  it("the notice says so when the report's count and the loaded rows differ", () => {
    const page = render(<NoncompliantNotice items={[]} stamped={3} kind="judgment" />);
    expect(page.text).toContain("3 judgments have no verdict.");
    expect(page.text).toContain("The report counts 3; 0 are among the rows loaded here.");
  });

  it("the notice is absent when there is nothing to report", () => {
    expect(render(<NoncompliantNotice items={[]} stamped={null} kind="judgment" />).html).toBe("");
  });
});

describe("a coding row with no codes", () => {
  const coded = JSON.stringify({ experiment: "e", condition: "steered", promptID: "item-1", sampleIndex: 0, seed: 5, wordCount: 40, codes: { stance: "mixed" }, briefReason: "Both sides.", judge: "coder-a", judgeKind: "local", judgeModel: "m" });
  const refused = JSON.stringify({ experiment: "e", condition: "steered", promptID: "item-1", sampleIndex: 0, seed: 5, codes: null, noncompliant: true, noncomplianceReason: "The response is empty.", judge: "coder-b", judgeKind: "local", judgeModel: "m" });
  const { rows } = parseCodingRows(`${coded}\n${refused}\n`);

  it("is read as noncompliant, with the coder's words, and no codes", () => {
    expect(rows[1]).toMatchObject({ noncompliant: true, noncomplianceReason: "The response is empty.", codes: {} });
    expect(rows[0].noncompliant).toBe(false);
  });

  it("says so in the row list, where it used to say 'No codes recorded.'", () => {
    expect(codesSummary(rows[1])).toBe("No codes: the coder's answer could not be used.");
    expect(codesSummary(rows[0])).toBe("stance = mixed");
    expect(codesSummary({ ...rows[0], codes: {} })).toBe("No codes recorded.");
  });

  it("is not a coding disagreement", () => {
    expect(codingDisagreements(rows, [{ name: "stance", type: "string", optional: false, values: [] }])).toEqual([]);
  });

  it("is counted from the report, and the notice words it for coding", () => {
    expect(parseCodingReport({ present: true, raw: { noncompliantCodings: 2 }, error: "", file: null }).noncompliantCodings).toBe(2);
    const page = render(<NoncompliantNotice items={[{ judge: "coder-b", condition: "steered", promptID: "item-1", sampleIndex: 0, reason: "The response is empty." }]} stamped={2} kind="coding" />);
    expect(page.text).toContain("2 codings have no codes.");
    expect(page.text).toContain("left out of every aggregate and agreement figure");
    expect(page.text).toContain("The report counts 2; 1 is among the rows loaded here.");
    expect(page.text).toContain("coder-b · steered · item-1 · sample 0: The response is empty.");
  });
});
