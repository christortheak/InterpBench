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
| OptVec training/evaluation | Owner loss, gradient, dose, selected artifact and held-out evaluation | Pending model study |
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
