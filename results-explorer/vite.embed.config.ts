import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// The EMBEDDED build (`npm run build:embed`): a plain static SPA — no
// worker, no RSC, no server — written to the repo root's
// `web/results-explorer/`, which the native macOS app ships as a code
// resource (CodeResources webAssets family) and serves to its WKWebView
// over a custom URL scheme. `base: "./"` keeps every asset reference
// relative so the custom scheme resolves them without a host.
export default defineConfig({
  root: fileURLToPath(new URL("./embed", import.meta.url)),
  base: "./",
  plugins: [react()],
  build: {
    outDir: fileURLToPath(
      new URL("../web/results-explorer", import.meta.url),
    ),
    emptyOutDir: true,
  },
});
