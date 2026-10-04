import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

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

export default defineConfig({
  root: fileURLToPath(new URL("./embed", import.meta.url)),
  base: "./",
  plugins: [react()],
  build: {
    outDir,
    emptyOutDir: true,
  },
});
