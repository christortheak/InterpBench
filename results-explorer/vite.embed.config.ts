import react from "@vitejs/plugin-react";
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

export default defineConfig({
  root: fileURLToPath(new URL("./embed", import.meta.url)),
  base: "./",
  plugins: [react(), bundledPackages()],
  build: {
    outDir,
    emptyOutDir: true,
  },
});
