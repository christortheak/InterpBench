# SteerLab Server — PyTorch/HF parallel to the MLX steering engine

This directory is a **second, independent implementation** of SteerLab's
activation-steering compute engine, built on **PyTorch + HuggingFace
Transformers** so the experiments can run on a **remote, non-macOS GPU cluster**.

The macOS app's engine (`SteeringKit` + the MLX-bound parts of `ExperimentKit`)
is Apple-Silicon/Metal only and cannot run off-Apple. This package re-implements
that surface while **reading and writing the same on-disk artifact formats**, so
vectors, manifests, and run directories interoperate with the Swift side.

> Scope note: the server is one of two co-equal compute substrates, and the
> SwiftUI app is a full client of it — the app treats the substrate as a scope
> (remote workspaces, the SSH tunnel + cluster setup wizard, the unified Run
> substrate picker, and remote chat all drive this server from the Mac). For
> orientation start at the root [`README.md`](../README.md) and
> [`docs/ONBOARDING.md`](../docs/ONBOARDING.md); the CLI contract is
> [`docs/CLI-REFERENCE.md`](../docs/CLI-REFERENCE.md).

## What maps to what

| Swift (MLX) | Python parallel |
|---|---|
| vendored `SteeredQwen3` / `SteeredGemma3Text` + `LayerIntervention` loop | `steering/hooks.py` — `register_forward_hook` on every decoder block (vendored models disappear) |
| `VectorInjector` | `steering/injector.py` (per-decode-step, prompt-end gated) |
| `ActivationRecorder` / `ActivationBankRecorder` / `HookFireCounter` | `steering/recorder.py` |
| `SteeringVectorMath` | `steering/vector_math.py` (NumPy; deterministic Gram power-iteration) |
| `SteeringVectorStore` / `SteeringVectorSidecar` | `steering/vector_store.py` |
| `ConceptExtractor` | `steering/extractor.py` |
| `StimulusSet` (SHA-256 pinning) | `steering/stimulus_set.py` |
| `SteeredContainerLoader` / `SteeredModels` | `steering/model_loader.py` |
| `ExperimentTasks.generate` / `ChatService` | `experiment/generate.py` + `experiment/prompt_render.py` |
| `ExperimentTasks` extract/validate/sweep/run | `experiment/tasks.py` over `experiment/manifest.py` |
| `SmokeTest` / `ToyConceptRun` | `experiment/smoke_test.py` / `experiment/toy_concept.py` |
| `Scoring` / `FrenchMarkers` | `experiment/scoring.py` |
| `FineTuneTrainer` (MLX LoRATrain + PDFKit) | `experiment/lora_train.py` (HF PEFT + pypdf) |
| `GemmaScopeAnalysis` + `scripts/gemmascope_analyze.py` | `experiment/gemma_scope.py` |
| `WebServer` | `api/app.py` + `api/routes.py` + `api/jobs.py` + `api/dto.py` |
| `steerlab-cli` | `cli.py` |

## The cross-engine artifact contract (why this interoperates)

- **Steering vector** = `<name>.safetensors` with float32 tensors keyed
  `layer_0 … layer_N`, plus a `<name>.json` sidecar (`schemaVersion: 2`). The
  Swift loader only JSON-*decodes* the sidecar, so schema compatibility (not
  byte-identical formatting) is required.
- **Stimulus / corpus / manifest hashes** are plain **SHA-256 over raw file
  bytes** (positive then negative). Identical bytes → identical hash on both
  engines, so a frozen manifest verifies regardless of which engine reads it.
- **Runs** are immutable `runs/<timestamp>-<slug>/` directories with artifacts
  such as `generations.jsonl`, `report.json`, `metrics.csv`,
  `cosine-matrix.csv` when relevant, and an `experiment-hash.txt` stamp.

**Scientific caveat:** bf16/fp16 activations on CUDA will **not** byte-match
MLX-8bit-on-Metal. Vectors are model+revision-specific by design, so the cluster
is a **fresh extraction substrate** — re-extract and re-validate here; do not
transplant Mac vectors as scientific artifacts.

## Install roles

One distribution, two console scripts, two install sizes:

| you want | install | you get |
|---|---|---|
| the **client** (`steerlab`) — author a workspace on any platform | `pip install -e .` | manifests, pins, conditions, concepts, bundles. numpy + safetensors, no torch |
| the **engine** (`steerlab-server`) — execute and serve | `pip install -e ".[runner]"` | + torch, transformers, accelerate, huggingface_hub, fastapi, uvicorn, pydantic |
| the full workbench | `pip install -e ".[all]"` | + peft/pypdf (LoRA), sae-lens (Gemma Scope), pytest |

The bare dependencies are the **client's**, not the engine's: everything only
execution needs lives behind the `runner` extra (see the comments in
`pyproject.toml`). `all` still contains `runner`'s whole contents, so
`bootstrap.sh`, `docs/ONBOARDING.md`, the committed locks
(`update-locks.sh` compiles with `--extra all`) and the app's Local Engine
flow resolve exactly the package set they always did.

The client covers the **whole authoring lifecycle** on that bare install —
create, attach, declare-condition, set-protocol, list, duplicate, `verify`,
`freeze` and `bundle package` — including studies that declare an SAE latent
condition they never execute. `[runner]` is needed to *execute*, not to
author or to check. (`verify` / `freeze` used to need it, because
`Manifest.verify` reached `experiment.sae_latent` and through it the
torch-bound injector stack; the SAE latent declared surface now lives in the
torch-free `steering.sae_latent_schema`.) The boundary is measured and pinned
(`tests/test_client_cli.py::
test_the_whole_authoring_lifecycle_stays_light_including_verify`) and recorded
as the now-closed gap **G7** in `docs/PORTABILITY-CONTRACTS.md`.

```bash
cd Server
python -m venv .venv && source .venv/bin/activate
pip install -e .                # the client only
pip install -e ".[runner]"      # + the engine's model/serve stack
pip install -e ".[runner,lora]" # + PEFT/pypdf for LoRA training
pip install -e ".[test]"        # + pytest
```

The package's heavy deps (torch, transformers) are typically already present on
a cluster.

On a machine that already has torch in another venv, you can bridge it instead
of reinstalling (what this repo's dev setup does): create the venv and drop a
`.pth` pointing at the existing site-packages. The `.venv.nosync` name keeps it
out of iCloud sync.

## Dependency locks

`pyproject.toml` declares **floors** (`torch>=2.2`). That is the right contract
for a library and the wrong one for an instrument: two sites can satisfy the
same floors with different torch/transformers and produce different numbers —
on the substrate where the reproducibility claims actually live. The
exact-version contract is two committed platform locks:

| lock | platform | role |
|---|---|---|
| `requirements-macos-arm64.lock` | macOS arm64, CPython 3.12 | Mac parity/dev work (a **testing** substrate) |
| `requirements-linux-x86_64.lock` | Linux x86_64 + CUDA, CPython 3.12 | **the science substrate** |

Install from the lock for reproducible local work:

```bash
cd Server
pip install -r requirements-macos-arm64.lock
pip install -e ".[all]"        # the package itself; deps already pinned above
```

Regenerate both (never hand-edit — the header of each file says the same):

```bash
pip install -e "Server[dev]"   # or: pip install uv
Server/scripts/update-locks.sh
```

which runs, from `Server/`:

```bash
MACOSX_DEPLOYMENT_TARGET=14.0 uv pip compile pyproject.toml --extra all \
    --python-version 3.12 --python-platform aarch64-apple-darwin \
    --output-file requirements-macos-arm64.lock
uv pip compile pyproject.toml --extra all \
    --python-version 3.12 --python-platform x86_64-unknown-linux-gnu \
    --output-file requirements-linux-x86_64.lock
```

Four decisions worth knowing:

- **`MACOSX_DEPLOYMENT_TARGET=14.0` is load-bearing.** torch ≥ 2.12 publishes
  Apple-silicon wheels tagged `macosx_14_0_arm64`; without it uv silently pins
  torch back to 2.11.0.
- **torch is site-owned on the cluster.** `scripts/bootstrap.sh` installs torch
  from `--torch-index` (default cu128) and then applies the lock with
  `torch`, `triton`, and `nvidia-*` filtered out, so a site keeps its own CUDA
  or ROCm build. Everything else is pinned exactly. `--no-lock` opts out of the
  lock entirely and says so loudly.
- **No `--generate-hashes` by default.** Hashes would be a half-guarantee here:
  torch — the largest and most substrate-defining wheel — is deliberately
  outside the hash-enforced set, `--require-hashes` is brittle against the site
  wheel mirrors and offline caches clusters use, and payload integrity is
  already covered by `deployment-manifest.json`'s SHA-256 verification. Pass
  `Server/scripts/update-locks.sh --generate-hashes` if a site wants them.
- **`jlens` is not in the locks**, matching `pyproject.toml`'s reason for
  keeping it out of `all`: it installs from a git URL, so a lock-based install
  would require github.com egress on every node. (Verified 2026-08-18: the two
  groups *do* co-resolve — `--extra jlens` changes no version in either lock,
  it only appends the git pin — so this is policy, not a conflict. jlens's own
  identity is already a commit sha in `pyproject.toml`.)

The lock is the *intended* resolution. The *achieved* one is stamped into every
run: `config.json`'s `pythonEnvironment` (schema 4) records the interpreter and
the resolved versions of the science-relevant packages the run actually
imported, including torch's local version segment (`2.13.0+cu128`). At
`experiment run` start the server also logs a **non-blocking** advisory — and
appends it to the run's `advisories.txt` — when installed torch/transformers
differ from the lock's pins. It is never a refusal: a queued cluster job must
not die because PyPI moved, and the stamp is the durable half either way.

Set `HF_TOKEN` (a Hugging Face read token) in the serving environment, or run
`huggingface-cli login` once: unauthenticated Hub requests are rate-limited,
and gated repos (e.g. `google/gemma-3-*`) will not download without a token.
`serve` logs a reminder at startup when no token is configured.

## Qualifying a site

```bash
steerlab-server site qualify                       # human: report on stdout, one line per check on stderr
steerlab-server site qualify --json                # one agent envelope, report inside result.report
steerlab-server site qualify --skip-model-fixtures # fast path: skip the checks that need the model cache
```

**When to run it:** right after `bootstrap.sh` on a new node, after any deploy
that changes the engine, and before the first study on a site you have not run
one on. It is cheap, it needs no GPU, and it runs *on the node being
qualified* — a login node or a workstation shell.

**What it answers:** whether this machine reproduces the contracts a result
depends on, or merely runs. Nine checks — build identity, the measurement-stack
fingerprint, dependency-lock agreement, the stimulus SHA-256 convention, the
committed prompt-render and tokenization goldens, the vector-parity arithmetic,
this deployment's own profile validation, and GPU visibility against the site's
declared GPU vocabulary. None aborts the others, so a cold node always gets a
complete report, and every row states what it compared and why an instrument
needs it.

Exit **0** clean or with warnings only; **70** if any check failed, naming the
failing ids and carrying that check's own remedy. **Skipped checks never change
the exit code but are always counted** — `3 passed, 0 warnings, 0 failed, 6
skipped of 9 checks` is a node that is mostly *unverified*, not a node that
passed. Read the summary line, not the exit code alone.

Two limits it states about itself rather than hiding: the parity fixtures are
same-engine synthetic, so that check qualifies the parity arithmetic on the
node and not cross-substrate agreement (vectors must still be re-extracted and
re-validated on whichever substrate a study runs on); and the render/token
goldens need each fixture's tokenizer in the **local** HF cache, so a partial
sweep reports as a warning rather than a pass.

Full flag and verdict reference: `docs/CLI-REFERENCE.md` §6.13.

## Run

```bash
# Pure-CPU tests (no GPU, no model): vector math, store round-trip, injection
# fires-per-token, hook offsets, stimulus-hash parity, prompt-render branching,
# API wiring.
pytest -q

# GPU smoke test on one model — asserts steered≠baseline, hook fires on every
# decode step, alpha 0 reproduces baseline exactly.
python -m steerlab_server.cli --config configs/smoke-test.json

# Toy "speak French" concept: extract → persist → inject → sweep → assert the
# concept vector moves output and beats a matched-norm random control.
python -m steerlab_server.cli --config configs/toy-concept.json

# Experiment verbs over a pinned manifest (STEERLAB_ROOT points at the data tree).
STEERLAB_ROOT=/path/to/project python -m steerlab_server.cli experiment verify <name>
STEERLAB_ROOT=/path/to/project python -m steerlab_server.cli experiment extract <name>
STEERLAB_ROOT=/path/to/project python -m steerlab_server.cli experiment run <name>

# HTTP service (localhost; reach a remote box over `ssh -L 8080:localhost:8080`).
python -m steerlab_server.cli serve --port 8080
#   GET  /healthz
#   GET  /api/state
#   POST /api/load        {"model": "...", "dtype": "bfloat16"}
#   POST /api/generate    {"text": "...", "injections": [{"layer": 20, "alpha": 4, "vector": [...]}]}
#   POST /api/generate/stream   (SSE token stream)
#   POST /api/extract     {"conceptDirectory": "prompts/concepts/french"}  → job
#   POST /api/experiment/{name}/{extract|validate|sweep|run}               → job
#   GET  /api/jobs/{id} ·  GET /api/jobs/{id}/stream ·  POST /api/jobs/{id}/cancel
```

`STEERLAB_ROOT` is the **runtime-injectable** project root (the cluster owns the
canonical `prompts/`, `experiments/`, `runs/` tree) — replacing the Swift
`#filePath`-baked root.

## Web workbench (browser UI)

`serve` also hosts a **self-contained browser workbench** at `/` — a thin client
over the JSON+SSE API, so the same UI works from any Mac/browser over an SSH
tunnel with zero install. It lives entirely under `Server/` (no changes to the
Swift app or its own browser client), so it does not touch or fork the local MLX
workbench. Open `http://localhost:8080` after `serve`. Tabs:

- **Model** — load a model (id · dtype · device), see device/layers/hidden/ctx,
  list cached HF repos.
- **Chat / Steer** — system prompt, prompt mode, temperature, max tokens, a
  steering **mixer** (`h + Σ αᵢ·vᵢ`: pick saved vectors, set layer/alpha, toggle),
  and chat with **live SSE token streaming** + stop.
- **Concept Lab** — full concept authoring (parity with the SwiftUI Concepts panel):
  create/rename/delete concepts; edit/import contrastive pairs (CAA) or **RepE/LAT**;
  the **grand-mean (emotion multi-concept)** family with a story corpus editor,
  include-vs-build concept selection, and per-concept extraction; reading-position
  (last token / pool-from-token); all seven generation-prompt helpers (incl.
  Anthropic-style dialogue); **validation stats** (held-out accuracy, split-half,
  norm-by-layer, outliers, control cosines); **scalar probe** training; **neutral
  corpus** import + **neutral-PC basis** build; Claude-assisted proposals
  (needs `ANTHROPIC_API_KEY`); and an in-lab vector library.
- **Geometry** — pairwise cosine matrix across vectors (discriminant validity).
- **Gemma Scope** — SAE feature cross-check (Gemma 3) + import a feature as a vector.
- **Variants / LoRA** — save/apply model variants; train LoRA adapters (HF PEFT).
- **Studies** — full authoring (create / attach / add conditions / **freeze** with
  the circularity-firewall gating / duplicate), **verify**, and launch
  **extract / validate / sweep / run / evaluate** (paired judge: Claude or local
  model) as jobs.
- **Multi-Agent** — author scenarios (agents + turns + routing + per-agent
  variants) and run them (configured or steering-stripped baseline).
- **Runs** — browse immutable run directories and view `report.json`,
  `generations.jsonl` (as a table), `cosine-matrix.csv`, transcripts.
- **Jobs** — live job table with status, and **streaming logs** per job.

Authoring note: the server is now a firewall **writer** too — frozen manifests
are stamped `frozenBy: "server"` + a full-manifest content hash and are read-only
(duplicate to iterate). **Honest cross-engine caveat:** the server's `freezeHash`
is *not* byte-identical to Swift's `manifestHash`, so each engine performs
same-author freeze-hash drift checks only for manifests it wrote. Swift accepts
`frozenBy: "server"` by relying on the mutually identical **pinned-input
SHA-256s** (stimuli, neutral corpus, task prompts, scenario, variant artifacts).

The UI is plain HTML/JS (no build step) served from
`steerlab_server/api/static/index.html`; it reads `STEERLAB_ROOT`'s data tree
through the API and resolves steering vectors straight out of `runs/`.

## Running on a Mac GPU (Apple Silicon / MPS)

The cluster target is CUDA, but the engine also runs **locally on a MacBook
Pro's GPU** via PyTorch's **MPS** (Metal) backend — handy for developing/testing
the server without a cluster. Device is auto-detected **CUDA → MPS → CPU**;
override with the `STEERLAB_DEVICE` env, a `--device` flag, a `device` field in a
config, or `"device"` in the `/api/load` body.

```bash
# Auto-detect (uses MPS on your Mac):
python -m steerlab_server.cli --config configs/smoke-test.json
# Force the Apple GPU, with a CPU fallback for any op MPS hasn't implemented:
PYTORCH_ENABLE_MPS_FALLBACK=1 STEERLAB_DEVICE=mps \
  python -m steerlab_server.cli --config configs/smoke-test.json
```

Caveats specific to MPS:
- **Use a full-precision HF repo, not an MLX/`-MLX-4bit` repo** — HF Transformers
  can't load MLX checkpoints, and 8-bit quant (bitsandbytes) is CUDA-only. The
  example configs already use `Qwen/Qwen3-4B`.
- Default dtype on MPS is **fp16** (`auto` picks bf16 on CUDA, fp16 on MPS, fp32
  on CPU); set `dtype` explicitly to override.
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` so a missing Metal kernel falls back to CPU
  instead of erroring.
- This is *not* the same path as the native MLX app — it's the PyTorch engine
  happening to use Metal. For the real M-series-optimized experience on the Mac,
  the MLX app remains the intended tool; MPS here is for local server testing.

## Config files

`configs/smoke-test.json` and `configs/toy-concept.json` are small examples;
edit `models` to a model your cluster has cached. Both accept the same fields as
the Swift smoke-test/toy-concept configs (`models`, `prompt`, `maxTokens`,
`alpha`/`alphas`, `seed`, plus `conceptDirectory` for the toy).

## Load-bearing invariants (kept identical to the Swift engine)

1. Injection fires on the prompt end **and every decode step** — never on a
   mid-prompt prefill chunk (`VectorInjector.should_inject`).
2. Injectors compose additively: `h + Σ αᵢ·vᵢ`.
3. Norm-unit alpha folds out the vector norm: `α_eff = α·residual/‖v‖`.
4. Extraction is RNG-free and deterministic (Gram power-iteration with two fixed
   starts) so LAT/PCA reproduce.

## Known gaps / deferred

- **Cross-engine numerics re-validation (WS7)** is the remaining scientific step:
  re-extract + re-validate on real cluster hardware and confirm the *structure*
  reproduces (the *numbers* differ from MLX by design). Treat the cluster model as
  a separately-frozen variant.
- **Authoring authority:** the server now writes/freezes manifests (stamped
  `frozenBy: "server"`), so two engines can author. Frozen manifests are
  read-only (duplicate to iterate). The `freezeHash` is **not** byte-identical to
  Swift's; both engines verify each other's pinned-input SHA-256s, while
  same-author freeze-hash drift checks remain engine-local.
- **Reproducibility / sampling:** the server permits classic runs at
  `temperature > 0` (it seeds PyTorch per generation, so they reproduce); Swift
  currently rejects non-zero temperature because mlx-swift-lm can't pin a per-run
  seed. A deliberate, documented divergence — not a parity bug.
- **Resident model registry:** multi-agent agents and local paired judges may
  request different base models. On CUDA, the server keeps one resident model per
  CUDA device by default (`cuda:0`, `cuda:1`, ...); on MPS/CPU it defaults to one
  resident slot and evicts/reloads least-recently-used models as needed. Override
  with `STEERLAB_MAX_LOADED_MODELS`.
- **Local judge model selection** now uses the requested local model through the
  registry when possible, and stamps both requested/actual judge model in the
  judge report.
- **Gemma Scope** full SAE run needs a full-precision Gemma loaded (catalog +
  import + cosine ranking are built and tested; no MLX-free Gemma is cached here).
- TLS and multi-user isolation are not implemented (single-researcher
  assumption). Bearer-token auth is **the default**: `steerlab-server serve`
  resolves token mode on every platform, hydrating/creating
  `~/.steerlab-token` (`STEERLAB_AUTH_TOKEN_FILE`) and printing the path, and
  every mutating `/api` route is gated. `serve --dev-open-loopback` opts into
  the single-user open tier and refuses to start on a non-loopback bind or
  with `STEERLAB_EXECUTOR=slurm`. See the root `SECURITY.md` for the full
  posture, threat model, and residuals. On
  multi-GPU: **slot-based multi-model
  residency exists** (the model registry above — one resident slot per CUDA
  device, `STEERLAB_MAX_LOADED_MODELS` override); **memory-aware multi-GPU
  scheduling does not** — nothing places or balances jobs across devices by
  free VRAM, and a single job never spans devices.
- LoRA adapters use PEFT's format, which is not MLX-loadable (separate substrate);
  adapter apply/remove is verified live to change and revert generation.
- Claude-API features (AI stimulus proposals, the Claude paired judge) need
  `ANTHROPIC_API_KEY`; they degrade gracefully without it.

The remote-backend client work has landed: the SwiftUI app pairs with this
server for workspaces, runs, jobs, and chat. For the method and what each
result layer may claim, see [`docs/METHODS.md`](../docs/METHODS.md) and
[`docs/RESULTS-ARCHITECTURE.md`](../docs/RESULTS-ARCHITECTURE.md).
