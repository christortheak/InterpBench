import react from "@vitejs/plugin-react";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig, type Plugin } from "vite";

// The EMBEDDED build (`npm run build:embed`): a plain static SPA — no
// worker, no RSC, no server — written to the repo root's
// `web/results-explorer/`, which the native macOS app ships as a code
// resource (CodeResources webAssets family) and serves to its WKWebView
// over a custom URL scheme. `base: "./"` keeps every asset reference
// relative so the custom scheme resolves them without a host.
//
// That directory is build output and is not tracked. The app build
// (`scripts/build-app.sh`, through `scripts/build-results-explorer.sh`) sets
// STEERLAB_EMBED_OUT_DIR to write the bundle straight into the app it is
// assembling, so what ships is always built from this source.
const outDir =
  process.env.STEERLAB_EMBED_OUT_DIR ||
  fileURLToPath(new URL("../web/results-explorer", import.meta.url));

// Writes `bundled-packages.json` into the bundle: the npm packages whose
// modules this build actually included, read from the bundler's own module
// list rather than from package.json (which also names packages only the
// standalone build uses). The app build turns that list into license notices
// (`scripts/generate-third-party-notices.py`), so a newly bundled dependency
// is covered without anyone remembering to add it.
function bundledPackages(): Plugin {
  return {
    name: "steerlab-bundled-packages",
    generateBundle(_options, bundle) {
      const names = new Set<string>();
      for (const output of Object.values(bundle)) {
        if (output.type !== "chunk") continue;
        for (const id of output.moduleIds) {
          const parts = id.split("\\").join("/").split("/node_modules/");
          if (parts.length < 2) continue;
          const [scopeOrName, name] = parts[parts.length - 1].split("/");
          names.add(scopeOrName.startsWith("@") ? `${scopeOrName}/${name}` : scopeOrName);
        }
      }
      this.emitFile({
        type: "asset",
        fileName: "bundled-packages.json",
        source: JSON.stringify([...names].sort(), null, 2) + "\n",
      });
    },
  };
}

// The demo (app/demo/index.tsx) holds every invented number in the explorer:
// a layout preview for someone working on the explorer in a browser. The app
// a researcher runs must never contain it, so this build swaps in
// app/demo/stub.tsx for every import of the demo module — same names,
// nothing behind them — and then FAILS if any demo module other than the
// stub, or any string the demo lists in DEMO_SENTINELS, reached the bundle.
// An invented result cannot be shown as a researcher's own if it is not in
// the app.
const norm = (path: string) => path.split("\\").join("/");
const demoDirectory = norm(fileURLToPath(new URL("./app/demo/", import.meta.url)));
const demoStub = norm(fileURLToPath(new URL("./app/demo/stub.tsx", import.meta.url)));
const isDemoModule = (id: string) => norm(id).startsWith(demoDirectory) && norm(id) !== demoStub;

// The sentinels, read from the demo's own source so the two lists can never
// drift. A plain read: the file is TSX, and only these literals are wanted.
const demoSentinels = (): string[] => {
  const source = readFileSync(fileURLToPath(new URL("./app/demo/index.tsx", import.meta.url)), "utf8");
  const marker = /export const DEMO_MARKER = "([^"]+)"/.exec(source)?.[1];
  const block = /export const DEMO_SENTINELS = \[([\s\S]*?)\];/.exec(source)?.[1] ?? "";
  const listed = [...block.matchAll(/"([^"]+)"/g)].map((match) => match[1]);
  const sentinels = [...(marker ? [marker] : []), ...listed];
  if (sentinels.length < 2) throw new Error("vite.embed.config.ts: could not read DEMO_SENTINELS from app/demo/index.tsx");
  return sentinels;
};

function withoutDemo(): Plugin {
  const sentinels = demoSentinels();
  return {
    name: "steerlab-without-demo",
    enforce: "pre",
    async resolveId(source, importer, options) {
      if (!importer) return null;
      const resolved = await this.resolve(source, importer, { ...options, skipSelf: true });
      if (!resolved) return null;
      return isDemoModule(resolved.id) ? demoStub : null;
    },
    generateBundle(_options, bundle) {
      for (const output of Object.values(bundle)) {
        const ids = output.type === "chunk" ? output.moduleIds : [];
        const leaked = ids.find(isDemoModule);
        if (leaked) this.error(`the embedded build contains demo content (${leaked}). It must import app/demo only through the stub.`);
        const text = output.type === "chunk" ? output.code : typeof output.source === "string" ? output.source : "";
        const found = sentinels.find((sentinel) => text.includes(sentinel));
        if (found) this.error(`the embedded build contains demo content: "${found}" in ${output.fileName}.`);
      }
    },
  };
}

export default defineConfig({
  root: fileURLToPath(new URL("./embed", import.meta.url)),
  base: "./",
  plugins: [withoutDemo(), react(), bundledPackages()],
  build: {
    outDir,
    emptyOutDir: true,
  },
});
