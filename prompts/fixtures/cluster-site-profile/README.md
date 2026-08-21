# Cluster-site profile: cross-engine render goldens

Committed byte-goldens for the WP5 cluster-site profile — the JSON both
engines decode, and the text both engines render from it. This directory *is*
the lockstep mechanism the site-profile schema requires: two schema
implementations agree only if they render the same profile to the same bytes,
so the two engines are pinned to one committed byte-truth rather than to each
other's source.

## Files

Four profiles, each with three goldens:

| Profile | What it exercises |
|---|---|
| `v1-maximal.json` | a **v1-stamped** profile: only v1 keys, so every schema-2 fact falls back to the renderer's *legacy* (`bootstrap.sh` / `executors.py`) constants. The compatibility half of §2.0 rule 3 — materializing an existing site must change nothing. It states no `account`, so the account NOTE branch renders. |
| `v2-maximal.json` | every schema-2 key stated, so a decoder that drops or renames one fails the key-list assertion. Also exercises tilde→`$HOME` normalization, single-quoted regexes and node templates, the per-partition QOS override, and per-job-class resource blocks. |
| `v2-neutral.json` | a **v2-stamped** profile that declares almost nothing. The declare-or-omit half of the rule: no legacy constant may leak in through a default. |
| `example-slurm-site.json` | the **worked example** (WP5 Step 12): a fictional but COHERENT production Slurm site stating what a real profile would state and nothing it would not. `v2-maximal` is a decoder torture test, not a site anyone would run; this is the one to copy. It is also what §4.2's "ship a fully populated but fictional example" means in practice, now that the real preset lives outside the repo. |

- `<profile>.env.golden.txt` — the complete rendered cluster env file (G1/G2/G4).
- `<profile>.headers.golden.txt` — the `#SBATCH` block for every job class, in
  the order `study, controller, setup, gpuSession`. The `# --- <class> ---`
  section marker is *fixture format*, not renderer output: each renderer
  returns a list of lines per class and both test harnesses join them the same
  way.
- `<profile>.unresolved.golden.txt` — the preview's "unresolved facts" pane,
  one `key<TAB>detail` per line (TAB because details contain colons). An empty
  pane is a file containing just a newline.

## Renderers pinned to these bytes

- Swift: `ClusterEnvironmentRenderer`
  (`Sources/ExperimentKit/ClusterEnvironmentRenderer.swift`), tested by
  `Tests/ExperimentKitTests/ClusterEnvironmentRendererTests.swift`.
- Python: `steerlab_server.api.site_environment` over
  `steerlab_server.api.site_profile`, tested by
  `Server/tests/test_site_profile.py`.

The wire key names are additionally pinned by the key-list assertions in
`Tests/ExperimentKitTests/ClusterSiteProfileTests.swift` and its Python twin
`test_maximal_fixture_carries_every_declared_key`.

## Regenerating

**Generate from Swift, never from Python.** The Swift renderer landed first
(commit a9b99e0) and its hand-transcribed compatibility proof against the
shipped `bootstrap.sh` / `executors.py` constants is the authority; Python is
held to it.

```sh
DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer \
TEST_RUNNER_STEERLAB_WRITE_CLUSTER_GOLDENS=1 \
  xcodebuild test -skipMacroValidation -scheme SteerLab-Package \
  -destination 'platform=macOS' -parallel-testing-enabled NO \
  -only-testing:ExperimentKitTests/ClusterEnvironmentRendererTests
cd Server && .venv.nosync/bin/python -m pytest -q tests/test_site_profile.py
```

Change the renderer only deliberately, on both engines at once, and re-run
both suites in the same sitting. Drift on either engine is a loud test
failure, never a skip.

## No institutional identifiers

DECIDED 2026-08-17 (audit §4.2): no real site configuration ships in this
repo, and since Step 12 no real site ships as a PRESET either —
`ClusterSiteProfile.presets` is two neutral Slurm templates (one conda-shaped,
one module-shaped) plus the workstation. Every value here is fictional —
`slurm.example.edu`, `bastion.example.edu`, `example-lab`,
`/project/examplelab` — and a Swift test asserts the fixtures name no
institution. `v1-maximal.json` and `example-slurm-site.json` are *shaped* like
production Slurm sites (partition caps, a GPU inventory, a scratch purge
window, a login-node guard) precisely so the public tree keeps a maximal worked
example under each default set while a real site stays private, imported as
JSON through `steerlab-cli cluster sites import`. The rendered goldens carry a
token PATH indirection (`$(cat "$HOME/.steerlab-token")`) and never a token
value.
