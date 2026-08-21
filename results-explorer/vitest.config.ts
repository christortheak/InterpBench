import { defineConfig } from "vitest/config";

// A dedicated Vitest config so the unit suite does NOT inherit
// `vite.config.ts` — that config boots the Cloudflare/Wrangler plugin and a
// Miniflare worker environment, which the pure-parser tests neither need nor
// should depend on. These tests exercise the artifact parsers in
// `app/lib/` against in-memory fixtures; no workspace data is read.
export default defineConfig({
  test: {
    include: ["test/**/*.test.ts"],
    environment: "node",
  },
});
