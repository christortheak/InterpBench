# Backend qualification protocol and evidence

This is a scoped measurement program, not a backend allowlist. Supported
operations remain available for exploration. CUDA execution is delegated by the
researcher to the cluster coding agents; no local result claims a CUDA comparison.

## Matrix declared before measurement

| Path | Required measurement | Initial local scope |
| --- | --- | --- |
| MLX sampling | Same record, replayed record, unrelated interleaving, reasoning budget | Controlled logits and cached model |
| Python RNG | MPS state restored on success/failure; concurrent scoped records | Actual MPS draws plus CPU fixtures |
| Extraction | Captured residuals; mean difference, LAT, designated reference directions | Small pinned HF model, CPU/MPS comparison |
| Injection / ablation | Actual edited residuals and resulting logits | Same model and layer, explicit dose |
| Readers | Train/selection/final role fixtures and model readout | Role fixtures; model qualification remains separate |
| Battery | End-to-end declared endpoints under intervention | Pending cluster/local matching study |
| Stability | Shared draw identity and resample summaries | Existing numerical fixtures; model study pending |
| OptVec training/evaluation | Owner loss, gradient, dose, selected artifact and held-out evaluation | Two-step training and separate evaluation measured; broader objectives pending |
| J-lens | Reference agreement, precision promotion, readout | Pending matched imported lens |
| J-space | Valid imported lens/artifact, projections and null comparisons | Pending matched imported lens/artifact |
| SAE | Decoder/import identity and actual latent intervention | Metadata fixtures only; device execution pending |
| Fine-tuning | Objective, updates, adapter scale, held-out loss | Pending matched recipe |
| Multi-agent | Turn seeds and repeated full transcript | Seed fixtures; live transcript qualification pending |
| Long context | Actual memory and device placement above chunking threshold | Explicit diagnostic; no context-wide certificate |

The optional `--optvec` journey is declared before its measurements: two fixed
training steps on two synthetic A/B choice rows, a separate two-row evaluation
file, float32, CPU-seeded initial direction, absolute dose 1, learning rate
0.01, seed 17, no anchor/capability loss and no gradient checkpointing. It runs
the production training and evaluation owners, saves the curve and selected
direction, and uses the same prospective float32/angle thresholds below for
comparison. It tests execution, not the usefulness of the learned direction.
The fixture source hash travels in the protocol. Longer training and preservation
objectives still require their own measurements.

The initial model probe uses fixed raw token IDs and float32 weights on the
same immutable revision. It captures residuals at a named block, logits at the
last position, and additive/ablation effects through production hooks. Token IDs
isolate numerical execution from rendering; this does **not** qualify chat
templates. Capture-to-CPU and CPU vector math are intentional and are reported
separately from model execution. MPS fallback is disabled before torch imports.

Initial diagnostic tolerances (chosen before running, subject to independent
review): float32 residual/logit absolute tolerance 0.005, relative tolerance
0.005; direction cosine at least 0.999. These are investigation thresholds,
not universal scientific equivalence criteria. Report maximum absolute error,
RMS error, cosine and decision agreement, including failures. Do not widen the
threshold after seeing results. Compare loss/gradient and technique-specific
readouts only under additional, prospectively declared protocols.

## Reproduce

Use `scripts/qualification/backend_probe.py` with the test-capable Python,
explicit device, cached model revision and output directory outside the checkout.
It installs nothing, uses offline loading, writes a fresh evidence directory,
records source/input/software identities, and produces portable tensor evidence.
Run the same command with `--device cuda` on the cluster, keeping model, revision,
dtype, tokens and layer identical. Use the script's comparison mode to compare
those evidence directories. An error remains a failed measurement with its
diagnostic; it is never relabeled CPU success or evidence for an unrun path.

The full product suites, controlled numerical fixtures, model probes and
end-to-end researcher journeys are different evidence layers. Neither a passing
suite nor a small model probe qualifies an entire backend or model family.

## Local observations, 2026-09-07

The committed producer was `ccf5045`, based on main `d84968c`. CPU and MPS
reports record a clean tracked diff, complete Python source hash and probe hash.
The subsequent comparator adds checks of the already-recorded OptVec evaluation
metrics; it does not change the measurement protocol or tolerances.
Small records are retained in [qualification/local-2026-09-07](qualification/local-2026-09-07/).
Raw tensors and immutable diagnostic runs remain outside the checkout.

Both Python runs used Qwen/Qwen3-0.6B at
`c1899de289a04d12100db370d81485cdf75e47ca`, float32, block 3, Python 3.12.13,
torch 2.13.0 and transformers 5.15.1 on Apple M5 Pro/macOS 27 beta. CPU used
SDPA; MPS used eager attention with fallback disabled and parameters on `mps:0`.
This compares the supported execution paths, including their attention difference.

| Measurement | CPU/MPS observation |
| --- | --- |
| Captured and edited residuals | Maximum absolute difference 2.63e-5 |
| Baseline, additive and ablation logits | Maximum absolute difference 2.04e-4; all six final-position argmax decisions agree |
| Three extraction recipes | Direction cosine above 0.9999999998; recipe arithmetic itself ran on CPU |
| Two-step OptVec owner training | Loss difference at most 1.51e-5; gradient-norm difference 1.05e-6; selected-direction cosine approximately 1 |
| OptVec separate evaluation | Four records per backend; log-odds-movement difference 1.10e-5; KL difference 6.24e-8 |
| Scoped RNG | Replayed draws agree; MPS global state restored |
| Long forward | 4,097 tokens completed with finite logits on both devices |

Every compared tensor and evaluation metric met the prospective diagnostic
tolerance. The two evaluation items already selected the target at baseline;
there were no flippable items. This is execution evidence, not steering efficacy.
The long forward is not the production generation chunking path. MPS reported
2,388,540,416 current allocated bytes and 10,902,945,792 driver bytes at the end;
these are not peak memory measurements.

The separate MLX probe used Qwen/Qwen3-4B-MLX-4bit at
`52a5ab34fa604bc8af6d3ce0cac0cab10b7eb495`, temperature 0.7, ten list-continuation
prompts and a 16-token answer cap. All 10 ordinary replays and all 10 replays with
an 8-token reasoning cap matched text and finish reason. An unrelated seed was
interleaved before each ordinary replay. Many answers are very short, so this
does not establish long-generation repeatability. Dependency, dtype and
quantization details are in the retained provenance. This is a different model
and precision from the Python probe; no MLX/PyTorch output comparison was made.

The opt-in MLX test reported four passing tests and Xcode reported success, but
the beta Metal runtime emitted `Completed handler provided after commit call`
after the tests completed. Preserve that diagnostic for review; a successful
result file does not establish clean process teardown on this beta toolchain.

## Cluster-agent handoff

Use the reviewed branch tip and an existing allocated CUDA runner. The researcher
has delegated cluster execution to the coding agents. Cache the exact model
revision through the normal approved model-preparation route; the probe does
not download dependencies or weights. From the checkout, with its runner Python:

```sh
PYTHONPATH=Server python scripts/qualification/backend_probe.py run \
  --device cuda --model Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --layer 3 --context-tokens 4097 --optvec --output /scratch/qualification-cuda
python scripts/qualification/backend_probe.py compare \
  /scratch/qualification-cpu /scratch/qualification-cuda
python scripts/qualification/backend_probe.py compare \
  /scratch/qualification-mps /scratch/qualification-cuda
```

Use policy-compliant scratch paths and fresh output directories. Transfer the
local CPU/MPS directories including `report.json` and `tensors.npz`; the small
committed JSON summaries alone cannot run the tensor comparison. On this build
host they are `/private/tmp/interpbench-cpu-probe-02` and
`/private/tmp/interpbench-mps-probe-02`. Alternatively rerun the same command with
`--device cpu` or `--device mps` and otherwise identical arguments. Verify source
hashes and disclose software/attention differences. Save failures without changing
tolerances, inspect the first differing tensors, and return the complete reports.

After this comparison, work through the still-open matrix rows with matched
assets and prospectively agreed path-specific criteria. J-lens and J-space need
their own lens/artifact inputs; neither is OptVec. Qualify full saved-agent and
multi-agent records, generation chunking, training objectives and researcher
journeys separately. Do not upgrade whole capability profiles based on this probe.
