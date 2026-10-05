import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("react/jsx-dev-runtime", async (original) => (await import("./support/capture")).wrapJSXRuntime(await original()));
vi.mock("react/jsx-runtime", async (original) => (await import("./support/capture")).wrapJSXRuntime(await original()));

import * as demo from "../app/demo";
import * as stub from "../app/demo/stub";
import { EffectsView } from "../app/views/Effects";
import { GenerationsView } from "../app/views/Generations";
import { Overview } from "../app/views/Overview";
import { ProvenanceView } from "../app/views/Provenance";
import { render } from "./support/capture";
import { enterBrowser, enterEmbedded, leaveHost } from "./support/host";

// Invented results must never be shown to a researcher as theirs. All demo
// content lives in app/demo/index.tsx; the embedded build swaps in
// app/demo/stub.tsx and fails if any demo string survives
// (vite.embed.config.ts; tests/embedded-bundle.test.mjs checks the built
// bundle). These tests hold the boundary in source.

afterEach(leaveHost);

const appRoot = fileURLToPath(new URL("../app", import.meta.url));
const sources = (directory: string): string[] => readdirSync(directory).flatMap((name) => {
  const path = join(directory, name);
  return statSync(path).isDirectory() ? sources(path) : /\.(tsx?|css)$/.test(name) ? [path] : [];
});

describe("the demo boundary", () => {
  it("the stub exports exactly the demo's names, so the embedded swap cannot break a view", () => {
    expect(Object.keys(stub).sort()).toEqual(Object.keys(demo).sort());
  });

  it("the stub carries nothing: no rows, no copy, demo mode off for good", () => {
    expect(stub.demoPreviewEnabled()).toBe(false);
    expect(stub.demoEffects).toEqual([]);
    expect(stub.demoGenerations).toEqual([]);
    expect(Object.values(stub.demoCopy).every((value) => value === "")).toBe(true);
    expect(stub.DEMO_SENTINELS).toEqual([]);
  });

  it("no demo string appears in the app's source outside app/demo/", () => {
    const offenders = sources(appRoot)
      .filter((path) => !relative(appRoot, path).startsWith(`demo${"/"}`))
      .flatMap((path) => demo.DEMO_SENTINELS.filter((sentinel) => readFileSync(path, "utf8").includes(sentinel)).map((sentinel) => `${relative(appRoot, path)}: ${sentinel}`));
    expect(offenders).toEqual([]);
  });

});

describe("with no run selected", () => {
  it("embedded: every section says no run is selected, even if the address asks for the demo", () => {
    enterEmbedded({ search: "?embedded=steerlab&demo=preview" });
    expect(demo.demoPreviewEnabled()).toBe(false);
    for (const page of [
      render(<Overview run={null} onNavigate={() => {}} />),
      render(<EffectsView run={null} onOpenFile={() => {}} />),
      render(<GenerationsView run={null} />),
      render(<ProvenanceView run={null} onOpenFile={() => {}} />),
    ]) {
      expect(page.text).toContain("NO RUN SELECTED");
      for (const sentinel of demo.DEMO_SENTINELS) expect(page.html).not.toContain(sentinel);
    }
  });

  it("standalone, without asking: the same", () => {
    enterBrowser();
    expect(render(<Overview run={null} onNavigate={() => {}} />).text).toContain("NO RUN SELECTED");
  });

  it("standalone, asked for by name: every demo page opens with the demonstration banner", () => {
    enterBrowser({ search: "?demo=preview" });
    for (const page of [
      render(<Overview run={null} onNavigate={() => {}} />),
      render(<EffectsView run={null} onOpenFile={() => {}} />),
      render(<GenerationsView run={null} />),
      render(<ProvenanceView run={null} onOpenFile={() => {}} />),
    ]) {
      expect(page.text.startsWith("DEMONSTRATION ONLY.")).toBe(true);
      expect(page.html).toContain(demo.DEMO_MARKER);
      // And no demo page claims that anything was checked or verified.
      for (const claim of ["verified", "checksums valid", "gates closed"]) expect(page.text.toLowerCase()).not.toContain(claim);
    }
  });
});
