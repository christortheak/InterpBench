import { describe, expect, it } from "vitest";
import {
  SCENARIO_ABSENT_NOTE,
  injectionLabel,
  loadScenario,
  loadVariantArtifact, sha256Hex,
  parseScenario,
  parseVariantArtifact,
  resolveSeats,
  routingByTurnTitle,
  runsRelativeSegments,
  seatTable,
  speakerAudit,
  speakerNames,
  transcripts,
  turnTitles,
  variantHeadline,
} from "../app/lib/panels";
import type { Generation, LocalDirectoryHandle, LocalFileHandle, RunFile, WorkspaceRun } from "../app/lib/types";

// In-memory stand-ins for the structural handle surface (File System Access
// in the browser, the native bridge when embedded). NOTHING here touches a
// real workspace: every byte is declared in the test.

const fakeFile = (name: string, text: string): File => {
  const bytes = new TextEncoder().encode(text);
  const like = {
    name,
    size: bytes.byteLength,
    lastModified: 0,
    type: "",
    text: async () => text,
    arrayBuffer: async () => bytes.buffer,
    slice: (start?: number, end?: number) => ({ text: async () => text.slice(start ?? 0, end) }),
  };
  return like as unknown as File;
};

const fakeFileHandle = (name: string, text: string): LocalFileHandle => ({
  kind: "file",
  name,
  getFile: async () => fakeFile(name, text),
});

const fakeRunFile = (name: string, text: string): RunFile => ({
  name,
  path: name,
  size: new TextEncoder().encode(text).byteLength,
  modified: 0,
  handle: fakeFileHandle(name, text),
});

/// A directory tree from a flat path → contents map ("a/b/file.json").
const fakeTree = (name: string, contents: Record<string, string>): LocalDirectoryHandle => {
  const directory = (prefix: string, directoryName: string): LocalDirectoryHandle => {
    const under = Object.keys(contents).filter((path) => path.startsWith(prefix));
    const childNames = new Set(under.map((path) => path.slice(prefix.length).split("/")[0]));
    return {
      kind: "directory",
      name: directoryName,
      values: () => (async function* () {
        for (const child of childNames) {
          const path = `${prefix}${child}`;
          yield contents[path] === undefined ? directory(`${path}/`, child) : fakeFileHandle(child, contents[path]);
        }
      })(),
      getDirectoryHandle: async (child: string) => {
        if (![...childNames].includes(child) || contents[`${prefix}${child}`] !== undefined) {
          throw new Error(`no directory '${child}'`);
        }
        return directory(`${prefix}${child}/`, child);
      },
      getFileHandle: async (child: string) => {
        const text = contents[`${prefix}${child}`];
        if (text === undefined) throw new Error(`no file '${child}'`);
        return fakeFileHandle(child, text);
      },
    };
  };
  return directory("", name);
};

const emptyDirectory: LocalDirectoryHandle = {
  kind: "directory",
  name: "run",
  values: () => (async function* () {})(),
  getDirectoryHandle: async () => { throw new Error("not used in this test"); },
  getFileHandle: async () => { throw new Error("not used in this test"); },
};

const fakeRun = (contents: Record<string, string>, generationRows: Generation[] = []): WorkspaceRun => ({
  key: "run", name: "run", path: "run", experiment: "test", status: "not stamped",
  model: "Model not stamped", dateLabel: "run", promptCount: 0, conditionCount: 0, generationCount: 0,
  report: {}, config: {}, artifacts: Object.keys(contents),
  files: Object.entries(contents).map(([name, text]) => fakeRunFile(name, text)),
  handle: emptyDirectory,
  effectRows: [], generationRows, generationFile: null, previewTruncated: false, skippedGenerationLines: 0,
  cosineMatrices: [], validationConcepts: [], validationReport: {}, sweepRows: [], sweepRecommendations: [],
  panelEffects: [],
});

// A turn record shaped like the engine's multi-agent generations.jsonl rows.
let turnCounter = 0;
const turn = (fields: Partial<Generation>): Generation => ({
  id: `turn-${(turnCounter += 1)}`,
  caseName: "case", family: "panel", condition: "baseline", alpha: "—", sample: 0,
  decision: "Not parsed", months: null, prompt: "prompt", output: "output", parsed: "Not parsed",
  words: 100, distinct2: 0.9, seed: 1, promptIndex: 0, replicateIndex: 0,
  isInstrument: false, wordCountStored: true, distinct2Stored: true,
  speakerName: "Judge A", turnTitle: "Private notes — Judge A", routedAgentIDs: ["judge-a"],
  modelID: "google/gemma-3-27b-it",
  ...fields,
});

const scenarioJSON = JSON.stringify({
  schemaVersion: 1,
  name: "KZ test multi",
  baseModelID: "google/gemma-3-27b-it",
  description: "",
  sharedMaterials: "the case",
  maxTokens: 2048,
  temperature: 0.7,
  agents: [
    { id: "judge-a", name: "Judge A", baseModelID: "google/gemma-3-27b-it", systemPrompt: "You are Judge A." },
    {
      id: "judge-b", name: "Judge B", baseModelID: "google/gemma-3-27b-it", systemPrompt: "You are Judge B.",
      variantArtifactPath: "runs/model-variants/2026-08-04-fear-agent/model-variant.json",
      variantArtifactHash: "80b1cc35371195cbe01ada99edd19c878649d4146c2fddc813d38cee0e6fc474",
    },
  ],
  turns: [
    { id: "t1", title: "Private notes — Judge A", speakerAgentID: "judge-a", outputLabel: "judge_a_private_notes", routing: "speakerOnly", routedAgentIDs: [], includeScenarioMaterials: true },
    { id: "t2", title: "Panel memo — Judge B", speakerAgentID: "judge-b", outputLabel: "judge_b_memo", routing: "allAgents", routedAgentIDs: ["judge-a", "judge-b"] },
  ],
});

const variantJSON = JSON.stringify({
  schemaVersion: 1,
  name: "optimize-conscientiousness-2026-08-03-fear-agent",
  baseModelID: "google/gemma-3-27b-it",
  baseRevision: "005ad3404e59d6023443cb575daa05336842228a",
  alphaInNormUnits: true,
  bandWidth: 1,
  adapters: [],
  injections: [{ concept: "fear", layer: 31, alpha: 0.2, vectorArtifactID: "runs/20260803-sweep/fear" }],
  promotion: {
    promotedBy: "criterion",
    sweepRun: "20260803T225631751-exp-optimize-conscientiousness-2026-08-03-sweep",
    experiment: "optimize-conscientiousness-2026-08-03",
    promotedAt: "2026-08-04T21:57:48Z",
    winningCell: { layer: 31, alpha: 0.2 },
  },
});

describe("parseScenario", () => {
  it("reads agents, seat variant pins, and declared turns", () => {
    const scenario = parseScenario(JSON.parse(scenarioJSON));
    expect(scenario).not.toBeNull();
    expect(scenario?.name).toBe("KZ test multi");
    expect(scenario?.agents.map((agent) => agent.name)).toEqual(["Judge A", "Judge B"]);
    expect(scenario?.agents[0].variantArtifactPath).toBe("");
    expect(scenario?.agents[1].variantArtifactPath).toBe("runs/model-variants/2026-08-04-fear-agent/model-variant.json");
    expect(scenario?.agents[1].variantArtifactHash).toHaveLength(64);
    expect(scenario?.turns.map((item) => item.routing)).toEqual(["speakerOnly", "allAgents"]);
    expect(scenario?.turns[0].routedAgentIDs).toEqual([]);
  });

  it("inherits the scenario base model for an agent that does not declare one", () => {
    const scenario = parseScenario({ name: "s", baseModelID: "m", agents: [{ id: "a", name: "A" }] });
    expect(scenario?.agents[0].baseModelID).toBe("m");
  });

  it("returns null for JSON that is not a scenario", () => {
    expect(parseScenario({ conditions: {} })).toBeNull();
    expect(parseScenario({ agents: [] })).toBeNull();
    expect(parseScenario("not an object")).toBeNull();
  });
});

describe("loadScenario", () => {
  it("reads a snapshotted scenario from the run directory", async () => {
    const load = await loadScenario(fakeRun({ "scenario.json": scenarioJSON }));
    expect(load.state).toBe("loaded");
    expect(load.scenario?.agents).toHaveLength(2);
    expect(load.note).toBe("");
  });

  it("reports absence as the dated engine-contract fact, not an error", async () => {
    const load = await loadScenario(fakeRun({ "generations.jsonl": "" }));
    expect(load.state).toBe("absent");
    expect(load.scenario).toBeNull();
    expect(load.note).toBe(SCENARIO_ABSENT_NOTE);
    expect(load.note).toContain("2026-08-05");
  });

  it("withholds attribution when the snapshot is unreadable rather than guessing", async () => {
    const broken = await loadScenario(fakeRun({ "scenario.json": "{not json" }));
    expect(broken.state).toBe("unreadable");
    const wrongShape = await loadScenario(fakeRun({ "scenario.json": JSON.stringify({ turns: [] }) }));
    expect(wrongShape.state).toBe("unreadable");
    expect(wrongShape.scenario).toBeNull();
  });
});

describe("parseVariantArtifact", () => {
  it("reads injections and the promotion birth certificate", () => {
    const artifact = parseVariantArtifact(JSON.parse(variantJSON));
    expect(artifact?.name).toBe("optimize-conscientiousness-2026-08-03-fear-agent");
    expect(artifact?.injections).toEqual([{ concept: "fear", layer: 31, alpha: 0.2, mode: "add", vectorArtifactID: "runs/20260803-sweep/fear" }]);
    expect(artifact?.promotion?.promotedBy).toBe("criterion");
    expect(artifact?.promotion?.winningLayer).toBe(31);
    expect(artifact?.promotion?.winningAlpha).toBe(0.2);
  });

  it("reads an absent mode as `add` — the engine omits the key for steering agents", () => {
    const artifact = parseVariantArtifact({ baseModelID: "m", injections: [{ concept: "c", layer: 3, alpha: 1 }] });
    expect(artifact?.injections[0].mode).toBe("add");
    const ablation = parseVariantArtifact({ baseModelID: "m", injections: [{ concept: "c", layer: 3, alpha: 1, mode: "ablate" }] });
    expect(ablation?.injections[0].mode).toBe("ablate");
  });

  it("keeps a hand-created variant distinguishable: no promotion block, no invention", () => {
    const artifact = parseVariantArtifact({ baseModelID: "m", name: "Hand golden gate", injections: [{ concept: "golden-gate-bridge", layer: 45, alpha: 0.19 }] });
    expect(artifact?.promotion).toBeNull();
  });

  it("returns null for JSON that is not a variant artifact", () => {
    expect(parseVariantArtifact({ name: "no injections" })).toBeNull();
    expect(parseVariantArtifact(null)).toBeNull();
  });

  it("labels an injection with its stored concept, layer, and strength", () => {
    const artifact = parseVariantArtifact(JSON.parse(variantJSON));
    expect(injectionLabel(artifact!.injections[0])).toBe("fear (L31, α 0.20)");
    expect(variantHeadline(artifact!)).toBe("fear (L31, α 0.20)");
    expect(variantHeadline({ ...artifact!, injections: [], adapters: [] })).toContain("no injections");
  });
});

describe("runsRelativeSegments", () => {
  it("resolves a workspace-relative variant path against the runs/ tree", () => {
    expect(runsRelativeSegments("runs/model-variants/x/model-variant.json")).toEqual(["model-variants", "x", "model-variant.json"]);
    expect(runsRelativeSegments("./runs/model-variants/x/model-variant.json")).toEqual(["model-variants", "x", "model-variant.json"]);
  });

  it("refuses paths that leave the served tree", () => {
    expect(runsRelativeSegments("/Users/ct/workspace/runs/x/model-variant.json")).toBeNull();
    expect(runsRelativeSegments("runs/../secrets.json")).toBeNull();
    expect(runsRelativeSegments("prompts/panels/kz.json")).toBeNull();
    expect(runsRelativeSegments("runs")).toBeNull();
    expect(runsRelativeSegments("")).toBeNull();
  });
});

describe("loadVariantArtifact / resolveSeats", () => {
  const runsTree = fakeTree("runs", {
    "model-variants/2026-08-04-fear-agent/model-variant.json": variantJSON,
    "model-variants/2026-08-04-broken/model-variant.json": "{not json",
  });

  it("reads the artifact through the runs/ handle (embedded mode)", async () => {
    const variant = await loadVariantArtifact(runsTree, "runs/model-variants/2026-08-04-fear-agent/model-variant.json");
    expect(variant.state).toBe("loaded");
    if (variant.state === "loaded") expect(variant.artifact.injections[0].concept).toBe("fear");
  });

  it("reports unreachable — never a guess — when no runs tree is granted (browser mode)", async () => {
    const variant = await loadVariantArtifact(null, "runs/model-variants/2026-08-04-fear-agent/model-variant.json");
    expect(variant.state).toBe("unreachable");
    if (variant.state === "unreachable") expect(variant.note).toContain("not reachable");
  });

  it("reports unreachable for a path the tree does not contain", async () => {
    const variant = await loadVariantArtifact(runsTree, "runs/model-variants/absent/model-variant.json");
    expect(variant.state).toBe("unreachable");
  });

  it("distinguishes a file it cannot READ from one it cannot REACH", async () => {
    const variant = await loadVariantArtifact(runsTree, "runs/model-variants/2026-08-04-broken/model-variant.json");
    expect(variant.state).toBe("unreadable");
    if (variant.state === "unreadable") expect(variant.note).toContain("rather than guessed");
  });

  it("maps every scenario seat to stock or to its resolved variant", async () => {
    const scenario = parseScenario(JSON.parse(scenarioJSON))!;
    const seats = await resolveSeats(scenario, runsTree);
    expect(seats.map((seat) => seat.name)).toEqual(["Judge A", "Judge B"]);
    expect(seats[0].variant.state).toBe("stock");
    // The shared fixture pins a DUMMY hash, so verification against the
    // mutable library correctly reports drift — and still shows the
    // artifact, labeled as current rather than as what ran.
    expect(seats[1].variant.state).toBe("drifted");
    if (seats[1].variant.state === "drifted") {
      expect(seats[1].variant.artifact.promotion?.sweepRun).toContain("sweep");
    }
    expect(seats[1].variantHash).toHaveLength(64);
  });

  it("keeps the declared path and hash visible even when the artifact is unreachable", async () => {
    const scenario = parseScenario(JSON.parse(scenarioJSON))!;
    const seats = await resolveSeats(scenario, null);
    expect(seats[1].variant.state).toBe("unreachable");
    expect(seats[1].variantPath).toContain("model-variant.json");
    expect(seats[1].variantHash).toHaveLength(64);
  });
});

describe("seatTable", () => {
  // Two seats × two conditions × two turn titles. Judge B is the steered
  // seat: its private notes drop 114 → 72 words, its memo is unchanged.
  const rows: Generation[] = [
    turn({ speakerName: "Judge A", turnTitle: "Private notes", promptIndex: 0, condition: "baseline", words: 100, distinct2: 0.9, replicateIndex: 0 }),
    turn({ speakerName: "Judge A", turnTitle: "Private notes", promptIndex: 0, condition: "baseline", words: 110, distinct2: 0.92, replicateIndex: 1 }),
    turn({ speakerName: "Judge A", turnTitle: "Private notes", promptIndex: 0, condition: "configured", words: 104, distinct2: 0.9, replicateIndex: 0 }),
    turn({ speakerName: "Judge A", turnTitle: "Private notes", promptIndex: 0, condition: "configured", words: 106, distinct2: 0.9, replicateIndex: 1 }),
    turn({ speakerName: "Judge B", turnTitle: "Private notes", promptIndex: 1, condition: "baseline", words: 110, distinct2: 0.96, replicateIndex: 0 }),
    turn({ speakerName: "Judge B", turnTitle: "Private notes", promptIndex: 1, condition: "baseline", words: 118, distinct2: 0.94, replicateIndex: 1 }),
    turn({ speakerName: "Judge B", turnTitle: "Private notes", promptIndex: 1, condition: "configured", words: 70, distinct2: 0.9, replicateIndex: 0 }),
    turn({ speakerName: "Judge B", turnTitle: "Private notes", promptIndex: 1, condition: "configured", words: 74, distinct2: 0.9, replicateIndex: 1 }),
    turn({ speakerName: "Judge B", turnTitle: "Panel memo", promptIndex: 2, condition: "baseline", words: 200, distinct2: 0.8, replicateIndex: 0 }),
    turn({ speakerName: "Judge B", turnTitle: "Panel memo", promptIndex: 2, condition: "configured", words: 200, distinct2: 0.8, replicateIndex: 0 }),
  ];

  it("means turns, words, and distinct-2 per seat × condition", () => {
    const table = seatTable(rows);
    expect(table.conditions).toEqual(["baseline", "configured"]);
    expect(table.baseline).toBe("baseline");
    expect(table.baselineDeclared).toBe(true);
    const judgeA = table.seats.find((seat) => seat.speaker === "Judge A")!;
    expect(judgeA.cells[0].turns).toBe(2);
    expect(judgeA.cells[0].meanWordCount).toBe(105);
    expect(judgeA.cells[1].meanWordCount).toBe(105);
    expect(judgeA.cells[1].deltaWordCount).toBe(0);
  });

  it("deltas each seat against its OWN baseline cell", () => {
    const judgeB = seatTable(rows).seats.find((seat) => seat.speaker === "Judge B")!;
    const configured = judgeB.cells.find((cell) => cell.condition === "configured")!;
    // baseline mean over 3 turns = (110 + 118 + 200) / 3; configured = (70 + 74 + 200) / 3.
    expect(configured.deltaWordCount).toBeCloseTo((70 + 74 + 200) / 3 - (110 + 118 + 200) / 3, 10);
    expect(judgeB.cells.find((cell) => cell.condition === "baseline")!.deltaWordCount).toBeNull();
  });

  it("cuts to a single turn title — the private-notes drop in one glance", () => {
    const table = seatTable(rows, "Private notes");
    const judgeB = table.seats.find((seat) => seat.speaker === "Judge B")!;
    const baseline = judgeB.cells.find((cell) => cell.condition === "baseline")!;
    const configured = judgeB.cells.find((cell) => cell.condition === "configured")!;
    expect(baseline.meanWordCount).toBe(114);
    expect(configured.meanWordCount).toBe(72);
    expect(configured.deltaWordCount).toBe(-42);
    expect(configured.percentWordCount).toBeCloseTo((-42 / 114) * 100, 10);
    expect(configured.deltaDistinct2).toBeCloseTo(0.9 - 0.95, 10);
    expect(table.rowsUsed).toBe(8);
  });

  it("marks an undeclared baseline so the view can badge the assumption", () => {
    const table = seatTable(rows.map((row) => ({ ...row, condition: row.condition === "baseline" ? "control" : row.condition })));
    expect(table.baselineDeclared).toBe(false);
    expect(table.baseline).toBe("configured");
  });

  it("reports empty cells as absent rather than zero", () => {
    // Judge C spoke only in the baseline arm; its configured cell has no
    // records at all and must read as missing, not as a zero-word mean.
    const table = seatTable([
      turn({ speakerName: "Judge C", condition: "baseline", words: 10 }),
      turn({ speakerName: "Judge D", condition: "baseline", words: 20 }),
      turn({ speakerName: "Judge D", condition: "configured", words: 30 }),
    ]);
    const cells = table.seats.find((seat) => seat.speaker === "Judge C")!.cells;
    expect(cells.find((cell) => cell.condition === "configured")?.turns).toBe(0);
    expect(cells.find((cell) => cell.condition === "configured")?.meanWordCount).toBeNull();
    expect(cells.find((cell) => cell.condition === "configured")?.deltaWordCount).toBeNull();
  });

  it("ignores non-panel rows (no speaker and no turn title)", () => {
    const table = seatTable([...rows, turn({ speakerName: undefined, turnTitle: undefined, words: 9999 })]);
    expect(table.seats.map((seat) => seat.speaker)).toEqual(["Judge A", "Judge B"]);
  });
});

describe("turnTitles / speakerNames", () => {
  it("orders turn titles by the engine's global turn order, not alphabetically", () => {
    const rows = [
      turn({ turnTitle: "Zebra turn", promptIndex: 0 }),
      turn({ turnTitle: "Alpha turn", promptIndex: 1 }),
      turn({ turnTitle: "Zebra turn", promptIndex: 3 }),
    ];
    expect(turnTitles(rows)).toEqual(["Zebra turn", "Alpha turn"]);
  });

  it("lists distinct speakers", () => {
    expect(speakerNames([turn({ speakerName: "Judge C" }), turn({ speakerName: "Judge A" }), turn({ speakerName: "Judge C" })]))
      .toEqual(["Judge A", "Judge C"]);
  });
});

describe("transcripts", () => {
  const rows = [
    turn({ condition: "configured", replicateIndex: 1, promptIndex: 1 }),
    turn({ condition: "configured", replicateIndex: 1, promptIndex: 0 }),
    turn({ condition: "baseline", replicateIndex: 1, promptIndex: 0 }),
    turn({ condition: "baseline", replicateIndex: 0, promptIndex: 0 }),
  ];

  it("groups by condition × replicate with baseline first and turns in order", () => {
    const arms = transcripts(rows);
    expect(arms.map((arm) => `${arm.condition}/${arm.replicate}`)).toEqual(["baseline/0", "baseline/1", "configured/1"]);
    expect(arms[2].turns.map((item) => item.promptIndex)).toEqual([0, 1]);
  });
});

describe("speakerAudit", () => {
  const rows = [
    turn({ speakerName: "Judge C", turnTitle: "Final position", condition: "baseline", replicateIndex: 0, promptIndex: 5, output: "join" }),
    turn({ speakerName: "Judge C", turnTitle: "Final position", condition: "configured", replicateIndex: 0, promptIndex: 5, output: "dissent" }),
    turn({ speakerName: "Judge C", turnTitle: "Final position", condition: "baseline", replicateIndex: 1, promptIndex: 5, output: "join" }),
    turn({ speakerName: "Judge C", turnTitle: "Private notes", condition: "baseline", replicateIndex: 0, promptIndex: 1 }),
    turn({ speakerName: "Judge A", turnTitle: "Final position", condition: "baseline", replicateIndex: 0, promptIndex: 4 }),
  ];

  it("gathers one seat's turns across replicates, conditions side by side", () => {
    const audit = speakerAudit(rows, "Judge C");
    expect(audit.conditions).toEqual(["baseline", "configured"]);
    expect(audit.replicates.map((entry) => entry.replicate)).toEqual([0, 1]);
    expect(audit.replicates[0].columns.map((column) => column.turns.length)).toEqual([2, 1]);
    expect(audit.turnsShown).toBe(4);
  });

  it("filters to one turn title and leaves missing condition columns empty", () => {
    const audit = speakerAudit(rows, "Judge C", "Final position");
    expect(audit.turnsShown).toBe(3);
    expect(audit.replicates[1].columns.map((column) => column.turns.length)).toEqual([1, 0]);
    expect(audit.replicates[0].columns[1].turns[0].output).toBe("dissent");
  });

  it("returns an empty audit for a speaker with no records", () => {
    const audit = speakerAudit(rows, "Judge Z");
    expect(audit.replicates).toEqual([]);
    expect(audit.turnsShown).toBe(0);
  });
});

describe("routingByTurnTitle", () => {
  it("carries the scenario's declared routing per turn title", () => {
    const routing = routingByTurnTitle(parseScenario(JSON.parse(scenarioJSON)));
    expect(routing.get("Private notes — Judge A")).toBe("speakerOnly");
    expect(routing.get("Panel memo — Judge B")).toBe("allAgents");
  });

  it("is empty when no scenario snapshot was read", () => {
    expect(routingByTurnTitle(null).size).toBe(0);
  });
});

describe("variant artifact hash verification (mutable library subtree)", () => {
  const artifactJSON = JSON.stringify({
    name: "fear-agent", baseModelID: "m",
    injections: [{ concept: "fear", layer: 31, alpha: 0.2 }],
  });
  const runsWith = (text: string) =>
    fakeTree("runs", { "model-variants/v/model-variant.json": text });
  const path = "runs/model-variants/v/model-variant.json";

  it("matching bytes load clean", async () => {
    const digest = await sha256Hex(artifactJSON);
    const variant = await loadVariantArtifact(runsWith(artifactJSON), path, digest);
    expect(variant.state).toBe("loaded");
  });

  it("drifted bytes are shown but labeled as the CURRENT artifact", async () => {
    const variant = await loadVariantArtifact(runsWith(artifactJSON), path, "0".repeat(64));
    expect(variant.state).toBe("drifted");
    if (variant.state === "drifted") {
      expect(variant.artifact.injections[0].concept).toBe("fear");
      expect(variant.note).toContain("CHANGED since the scenario pinned it");
    }
  });

  it("no declared hash means no drift claim", async () => {
    const variant = await loadVariantArtifact(runsWith(artifactJSON), path, "");
    expect(variant.state).toBe("loaded");
  });
});
