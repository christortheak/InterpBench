# Handoff: per-request GPU type for managed scientific execution

- Date: 2026-09-12
- From: the maintainer's integration agent, on main `568b678`
- For: the refactor agents
- Priority: high; it blocks the H100 leg of the dimension-batch benchmark
  and, later, mixed-type fitting rounds.
- **Status 2026-09-12 (later the same day): the core slice landed on main
  as `c9f4f08`** — `gpuType` on plan and submit (HTTP, `--gpu-type` on both
  CLIs), vocabulary validation, plan-hash binding, guide text, tests, CLI
  reference — and was accepted live: the same staged bundle planned by
  default (identical to the earlier plan apart from controller identity)
  and with `--gpu-type H100` (`gpu:H100:1`, new hash), then submitted to an
  H100. **Still open for the agents: §2's round per-shard `gpuType`, the
  Mac execution-sheet picker, and recording the runtime GPU name in
  telemetry.** §3 and the identity rule stand.

## 1. The problem, observed live

Every managed scientific submission (`jlens-fit`, `jlens-fit-benchmark`,
`jlens-fit-round` shards, `jlens-fit-assess`, battery, stability) takes its
Slurm resources from `SlurmResources.from_env()` inside
`scientific_execution.plan`. The GPU type therefore comes from the
controller's environment (`STEERLAB_SLURM_GRES`, rendered from the site
profile) and from nothing else. A controller can submit to exactly one GPU
type for its whole lifetime, and changing it means editing the site's env
file and restarting the controller.

Study experiment submits do not have this limitation: the experiment submit
route accepts `resources.gres` in the body and validates it against the
site's declared GPU vocabulary. The science route never grew the same
field.

Consequences a researcher hits:

- Running the same benchmark on two GPU types, the handoff's A100 and H100
  legs, needs a controller restart between them.
- A fitting round cannot place shards on more than one GPU type, so it
  cannot take advantage of whichever pool has free single-GPU slots. The
  checkpoint identity already permits merging across GPU models (it binds
  dtype, attention, kernel policy, TF32 flags, torch and transformers
  versions, and the device class, not the GPU name), so the only obstacle
  is submission.

## 2. What to build

**A reviewed, per-request GPU type on managed scientific plan and submit.**
Placement is execution shape, not scientific content, so it does not go
into the published `request.json`. It goes where walltime already goes on
resubmit: the plan and submit call, recorded in the plan and bound into the
plan hash.

Contract:

- `POST /api/science/plan` and `POST /api/science/submit` accept an optional
  `gpuType` (string) beside the request document. Absent means the
  controller's default, exactly today's behaviour.
- Validation is the site's declare-or-refuse rule: the type must be in the
  declared vocabulary (`STEERLAB_SLURM_GPU_TYPES`), and the resulting gres
  is `gpu:<type>:<gpus>` through `SlurmResources.normalized_gres()`. An
  unknown type is a typed refusal naming the vocabulary. No untyped
  `gpu:N` request is ever produced; that gate exists because an untyped
  request once landed on an unsupported card.
- The chosen type flows into `result['resources']`, so it is inside the
  existing `planSHA256` and inside the scheduler preview. Submit with a
  plan hash reviewed for one type cannot silently run on another.
- The memory-fit and cost reviews that read `gpu_vram_gb` use the chosen
  type's capacity, not the default's.
- The run record and telemetry record the GPU actually seen at runtime
  (`torch.cuda.get_device_name`, compute capability) beside the requested
  type. Do **not** add the GPU model to the checkpoint identity in this
  change; whether mixed-type merges should be refused is a numerical
  question the A100 versus H100 benchmark answers first. Record, do not
  bind.

**Surfaces:**

- Mac: `remote science-plan --gpu-type <type>` and
  `remote science-submit --gpu-type <type>`; the same flag on the
  Research-methods execution sheet as a picker over the site's declared
  types, defaulting to the site default.
- Python client: `runner science-plan --gpu-type` and
  `runner science-submit --gpu-type`.
- Rounds: `jlens-fit-round` plan and submit actions accept `gpuType`
  (one type for the round) and, if it is cheap, a per-shard map so a
  researcher can direct waiting shards to another pool. Each child plan
  records its type; the round status lists it per shard.
- Guide text (`WorkspaceSeed/prompts/method-guides/jlens.md` and the
  cluster guide): state that the GPU type is chosen at plan time, that the
  site default applies when omitted, and that the identity does not bind
  the GPU model, so mixed-type merges are allowed and should be compared
  numerically before being relied upon.
- CLI reference regeneration and the generated-declaration gates.

**Tests:** Python for the refusal (unknown type), the hash change (same
request, two types, two hashes), the preview line, the default path being
byte-identical to today's plan, and round child plans carrying the type;
Swift for the flag parsing and the request body.

## 3. What not to do

- Do not change the site env rendering or the controller's default.
- Do not add an "any GPU" option.
- Do not put placement into the scientific `request.json`; a published
  request must stay valid on a site with different hardware.
- Do not touch the checkpoint identity.

## 4. Verification I will run before landing

Both suites on the tip; gates and audits; a live plan on the first site
with the default (hash unchanged from today's A100 plan for the same
staged bundle, `440f3aee…`) and with `--gpu-type H100` (a new hash whose
preview says `gpu:H100:1`), then a live submit of the benchmark bundle
already staged as `f22bf898…` to an H100.
