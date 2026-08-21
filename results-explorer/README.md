# Results Explorer

The canonical surface for reading SteerLab run results. One codebase, two
builds:

- **Embedded** (`npm run build:embed`) — a plain static SPA written to the
  repo root's `web/results-explorer/`, which the macOS app ships as a code
  resource and serves to its WKWebView over a custom URL scheme. This is the
  build that matters for the instrument; it is what "open the Results
  Explorer" in the app renders.
- **Standalone** (`npm run dev` / `npm run build`) — the same UI as a
  [vinext](https://github.com/cloudflare/vinext) app with a worker, useful for
  iterating on the explorer in a browser without launching the Mac app.

The explorer READS artifacts; it never computes evidence. κ, confidence
intervals, and p-values render only from engine-produced artifacts, and
anything the viewer derives itself carries a badge saying so.

## Prerequisites

- Node.js `>=22.13.0`

## Commands

```bash
npm ci                  # install from the committed lockfile
npm test                # vitest: the artifact parsers in app/lib/ against
                        # in-memory fixtures (no workspace data is read)
npm run build:embed     # the embedded SPA → ../web/results-explorer/
npm run dev             # standalone dev server
npm run build           # standalone production build
npm run lint            # eslint (not yet clean; not gated in CI)
```

CI (release repository) runs `npm ci`, `npm test`, and `npm run build:embed`,
so a cold clone is proved able to produce the bundle the app embeds.

## Shape

- site code lives under `app/`; `app/lib/` holds the artifact parsers and is
  where the unit suite points
- `embed/` is the entry point for the embedded SPA build
  (`vite.embed.config.ts`, `base: "./"` so the custom scheme resolves assets
  without a host)
- `vite.config.ts` configures the standalone build. It reads an optional
  `.openai/hosting.json` for D1/R2 bindings and defaults to no bindings when
  that file is absent — it is hosting scaffolding and is not part of a
  release checkout.
- `db/schema.ts` is intentionally empty; `examples/d1/` and
  `drizzle.config.ts` are optional D1 scaffolding the explorer does not
  require
