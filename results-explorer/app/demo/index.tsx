"use client";

// THE DEMO — every invented number, name, and check in the explorer lives in
// this one file, and nowhere else.
//
// It exists so that someone working on the explorer in a browser
// (`npm run dev`, then `?demo=preview`) can see the layout with no workspace
// to hand. It is NOT part of the app a researcher runs:
//
// - The EMBEDDED build (`npm run build:embed`, what the Mac app ships) does
//   not contain this file. Its build config swaps in `./stub.tsx`, which
//   exports the same names with nothing behind them, and then fails the
//   build if any module from this folder other than the stub, or any of the
//   strings in DEMO_SENTINELS, reached the bundle (vite.embed.config.ts).
//   A researcher can never be shown an invented result as their own, because
//   the invented results are not in the app.
//
// - In the STANDALONE build every page drawn from this file carries
//   `DemoBanner`, and the content itself is plainly made up: outcomes are
//   "Example outcome A", items are "Example item 1". Nothing here borrows a
//   real study's vocabulary, and no badge claims that anything was verified.
//
// Rules for editing: keep all demo content in this file; keep `./stub.tsx`
// exporting the same names (test/demo.test.tsx compares them); and keep every
// distinctive string a demo page shows listed in DEMO_SENTINELS.

import { Badge, ForestRow } from "../components/ui";
import { isEmbedded } from "../embedded-workspace";
import { effectKey } from "../lib/effects";
import type { Effect, Generation, View } from "../lib/types";

/// Stamped on every demo page's wrapper, and the first thing the embedded
/// build's guard looks for.
export const DEMO_MARKER = "steerlab-invented-demo-content";

/// Strings that appear only in demo content. The embedded build fails if its
/// output contains any of them; a unit test fails if any of them appears in
/// the app's source outside this folder.
export const DEMO_SENTINELS = [
  DEMO_MARKER,
  "DEMONSTRATION ONLY",
  "Example outcome A",
  "Example item 1",
  "example-study-01",
  "An invented headline",
  "Invented demo data",
];

/// Demo mode is on only in a browser, only when asked for by name, and never
/// in the embedded app (which does not ship this file in any case).
export const demoPreviewEnabled = () =>
  typeof window !== "undefined" &&
  !isEmbedded() &&
  new URLSearchParams(window.location.search).get("demo") === "preview";

/// The shell's short label for "what is on screen is the demo"; "" otherwise.
export const demoLabel = () => demoPreviewEnabled() ? "Invented demo data" : "";

// The rows carry the same shape a real table does: one condition, pooled rows
// only (a preview must not illustrate a stratified finding that no file ever
// stated).
const demoRow = (row: Omit<Effect, "condition" | "stratifyBy" | "stratum" | "pairedUnit" | "estimand" | "inference" | "key">): Effect => ({
  ...row, condition: "example-condition", stratifyBy: "pooled", stratum: "",
  pairedUnit: "", estimand: "", inference: "",
  key: effectKey({ condition: "example-condition", endpoint: row.endpoint, stratifyBy: "pooled", stratum: "" }),
});

export const demoEffects: Effect[] = [
  demoRow({ endpoint: "Example outcome A", short: "Example outcome A", estimate: 7.8, low: 3.1, high: 12.4, unit: "example units", n: 48, q: 0.012, p: 0.003, correction: "holm", direction: "positive" }),
  demoRow({ endpoint: "Example outcome B", short: "Example outcome B", estimate: 0.14, low: 0.04, high: 0.23, unit: "Δ probability", n: 64, q: 0.028, p: 0.009, correction: "holm", direction: "positive" }),
  demoRow({ endpoint: "Example outcome C", short: "Example outcome C", estimate: -0.09, low: -0.17, high: -0.01, unit: "Δ probability", n: 64, q: 0.041, p: 0.021, correction: "holm", direction: "negative" }),
  demoRow({ endpoint: "Example outcome D", short: "Example outcome D", estimate: -0.06, low: -0.15, high: 0.02, unit: "score points", n: 48, q: 0.184, p: 0.061, correction: "holm", direction: "negative" }),
  demoRow({ endpoint: "Example outcome E", short: "Example outcome E", estimate: 0.01, low: -0.02, high: 0.04, unit: "Δ rate", n: 64, q: 0.742, p: 0.511, correction: "holm", direction: "positive" }),
];

const demoGeneration = (item: number, sample: number, condition: string, decision: string): Generation => ({
  id: `EX-${String(item).padStart(2, "0")} · S${String(sample).padStart(2, "0")}`,
  caseName: `Example item ${item}`,
  family: "Example group",
  condition,
  alpha: condition === "baseline" ? "0" : "+1.0",
  sample,
  decision,
  months: null,
  prompt: `Invented prompt ${item}. It stands in for the question a study would put to the model.`,
  output: `Invented answer for example item ${item} under ${condition}. It stands in for the text a model would write, so that this page has something to lay out.`,
  parsed: `${decision.toLowerCase()} · invented`,
  words: 28,
  distinct2: 0.9,
  seed: 1000 + item * 10 + sample,
  isInstrument: false,
  wordCountStored: true,
  distinct2Stored: true,
});

export const demoGenerations: Generation[] = [
  demoGeneration(1, 1, "example-condition", "Option A"),
  demoGeneration(2, 1, "example-condition", "Option B"),
  demoGeneration(3, 1, "baseline", "Option A"),
  demoGeneration(3, 1, "example-condition", "Option B"),
  demoGeneration(4, 2, "example-control", "Option A"),
  demoGeneration(5, 3, "example-condition", "Option B"),
];

/// The handful of strings the shared views show in place of a run's own.
export const demoCopy = {
  effectsEyebrow: "INVENTED DEMO DATA · 5 EXAMPLE OUTCOMES",
  effectsSource: "Invented demo data",
  effectsExperiment: "example-study-01",
  effectsNote: "These rows are invented to show the layout. In a real run, every value in this table is read from effect-sizes.csv.",
  generationsEyebrow: "INVENTED DEMO DATA · 6 EXAMPLE RECORDS",
  generationsModel: "example-model",
  generationsFooter: " · invented example",
  downloadLabel: "Demo only",
};

/// The strip every demo page starts with. It says what the page is before
/// the page says anything else.
export function DemoBanner() {
  return (
    <div className="demo-banner" role="alert" data-demo={DEMO_MARKER}>
      <strong>DEMONSTRATION ONLY.</strong>{" "}
      Every name, number, and check on this page is invented to show the layout. None of it comes from a run, and none of it is a result.
    </div>
  );
}

export function DemoOverview({ onNavigate }: { onNavigate: (view: View) => void }) {
  return (
    <div className="view-enter" data-demo={DEMO_MARKER}>
      <DemoBanner />
      <section className="hero-grid">
        <div className="hero-copy">
          <div className="kicker"><span>Layout preview</span><span>·</span><span>Invented demo data</span></div>
          <h1>An invented headline shows where <em>a study&rsquo;s main finding</em> would be read.</h1>
          <p className="dek">This paragraph stands in for one or two sentences describing a completed study. Open a workspace to see a real run here instead.</p>
          <div className="hero-actions">
            <button className="primary" onClick={() => onNavigate("effects")}>See the example effects <span>→</span></button>
            <button className="text-button" onClick={() => onNavigate("generations")}>Read the example generations</button>
          </div>
        </div>
        <div className="hero-stat" aria-label="Example effect estimate (invented)">
          <span className="hero-stat-label">Example estimate</span>
          <div><strong>+7.8</strong><span>example units</span></div>
          <p>Example 95% interval <b>+3.1 to +12.4</b></p>
          <div className="mini-scale"><i /><b /><em /></div>
          <footer><span>Example adjusted p</span><strong>0.012</strong></footer>
        </div>
      </section>

      <section className="metric-strip" aria-label="Example summary (invented)">
        <div><span>Example items</span><strong>64</strong><small>invented</small></div>
        <div><span>Example conditions</span><strong>6</strong><small>invented</small></div>
        <div><span>Example parse rate</span><strong>98.7%</strong><small>invented</small></div>
        <div><span>Example capability</span><strong>99.1%</strong><small>invented</small></div>
      </section>

      <section className="section-grid main-evidence">
        <div className="card evidence-card">
          <header className="section-header">
            <div><span className="section-number">EXAMPLE EFFECTS</span><h2>Where a study&rsquo;s effects are charted</h2></div>
            <button className="quiet-link" onClick={() => onNavigate("effects")}>Example table →</button>
          </header>
          <div className="axis-hint"><span>Lower</span><span>No difference</span><span>Higher</span></div>
          <div className="forest">
            {demoEffects.map((effect) => <ForestRow key={effect.key} effect={effect} compact />)}
          </div>
          <div className="legend"><span><i className="legend-dot" /> Example estimate</span><span><i className="legend-line" /> Example interval</span></div>
        </div>

        <aside className="card claim-card">
          <span className="section-number">EXAMPLE STATUS</span>
          <h2>Where a study&rsquo;s standing is summarised.</h2>
          <div className="claim-step active">
            <i>1</i><div><strong>Example step one</strong><span>Invented for the layout</span></div><Badge>Example</Badge>
          </div>
          <div className="claim-step">
            <i>2</i><div><strong>Example step two</strong><span>Invented for the layout</span></div><Badge>Example</Badge>
          </div>
          <button className="claim-foot" onClick={() => onNavigate("provenance")}><span>See the example run files page</span><b>→</b></button>
        </aside>
      </section>
    </div>
  );
}

export function DemoProvenance() {
  const artifacts = [
    ["report.json", "Run summary and per-condition counts"],
    ["generations.jsonl", "One record for each generation"],
    ["effect-sizes.csv", "Paired estimates and tests"],
    ["exclusions.json", "Declared exclusions and counts"],
    ["battery.jsonl", "Capability checks for each condition"],
  ];
  return (
    <div className="view-enter inner-view provenance-view" data-demo={DEMO_MARKER}>
      <DemoBanner />
      <header className="page-title">
        <div><span className="section-number">LAYOUT PREVIEW · INVENTED DEMO DATA</span><h1>Run files</h1><p>Where a run&rsquo;s stored facts and files are listed. Nothing on this page was read from a run, and nothing on it was checked.</p></div>
      </header>
      <section className="provenance-grid">
        <div className="card config-card">
          <header className="section-header"><div><span className="section-number">EXAMPLE RUN</span><h2>Stored metadata</h2></div><Badge>Example</Badge></header>
          <dl>
            <div><dt>Experiment</dt><dd>example-study-01</dd></div>
            <div><dt>Model</dt><dd>example-model</dd></div>
            <div><dt>Study state</dt><dd>Example only</dd></div>
            <div><dt>Unit of analysis</dt><dd>Example only</dd></div>
          </dl>
        </div>
        <div className="card artifact-card">
          <header className="section-header"><div><span className="section-number">EXAMPLE FILES</span><h2>What a run directory holds</h2></div><span className="muted">Names only</span></header>
          {artifacts.map(([name, description]) => <div className="artifact" key={name}><span className="file-icon">↳</span><div><strong>{name}</strong><span>{description}</span></div><small>example</small><b /></div>)}
        </div>
      </section>
    </div>
  );
}

/// The two cards under the demo effect table.
export function DemoEffectsLower() {
  return (
    <section className="section-grid effects-lower" data-demo={DEMO_MARKER}>
      <div className="card sensitivity-card">
        <header className="section-header"><div><span className="section-number">EXAMPLE CHECKS</span><h2>Where a study&rsquo;s robustness checks would sit.</h2></div><Badge>Example</Badge></header>
        <div className="sensitivity-row"><span>Example check one</span><div><i style={{ width: "65%" }} /></div><strong>+7.8</strong></div>
        <div className="sensitivity-row"><span>Example check two</span><div><i style={{ width: "58%" }} /></div><strong>+7.0</strong></div>
        <footer>Invented numbers. The explorer shows no such card for a real run unless the run&rsquo;s files carry the values.</footer>
      </div>
    </section>
  );
}
