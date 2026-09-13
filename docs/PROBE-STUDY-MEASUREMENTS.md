# Study probe measurements (P3)

Implementation on `codex/probe-study-measurements`, from main `b1190f7`.
The capture/report clarifications are the preceding commit `f855904`.

## Researcher experience and scope

A researcher fits a probe in Probes, then selects it in a study's Measurements
area. Several probes can read selected conditions or panel seats, during prompt
processing, response generation, or both. The agent remains the base model plus
its vectors and adapters. These study settings measure that agent; they do not
alter its behavior. The shared agent guide explains this distinction, the input
format, the review/save commands, and the interpretation limits.

App, native CLI, portable CLI, and workbench HTTP author through
`experiment/probe_measurements.py`. Both CLIs expose `science measurements-review`
and `science measurements-save`. The app uses the same owners through its packaged
Python adapter. These are authoring operations under the existing workbench route,
not new runner authority. Execution uses the existing study run and bundle routes.

The execution backend is **PyTorch**, locally or on a remote Python engine. Native
MLX study measurement execution is not implemented. The app explains this in the
Measurements area; a direct native run gives a Python Compute repair before model
execution. This is surface access to one numerical owner, not an assertion of
cross-backend activation equivalence. Live app and large-model qualification stay
pending until the larger program is complete, per the researcher’s instruction.

## Declaration and identity

`probeMeasurements` is an optional study field. Absence leaves historical manifest
encoding unchanged. New settings participate in the ordinary study identity and
freeze mechanism. Adding measurements changes study identity; where sampling seeds
are derived from that identity, a newly frozen study can therefore get different
seeds. Observer noninterference is tested with the same actual seed, rather than
claiming that different study identities draw the same samples. A review precondition remains outside the manifest's hashed
bytes. Save checks the current study, settings, and probe byte hashes again under
the existing manifest transaction. Frozen studies must be duplicated before edits.

Each selection names an ID, `{path, sha256}` probe reference, condition/agent
filters, prefill/decode stages, and preAction/postAction reading stage. The artifact
itself declares layer, block input/output, prompt-position selection, model,
revision, coordinate convention, precision, and tokenizer/template identity.
Empty filters select every condition/agent. Ordinary agent instances use condition
names; panel instances use seat IDs, with replicate/turn context retained.

Both pin-surface enumerators carry every selected probe. Freeze relocates probes
under `runs/` into `experiments/<name>/pinned/probe-<sha256>.probe.json`, without
changing source or copied bytes. Frozen verification rechecks pins, and remote
bundles include them. Library discovery still scans original run artifacts; a
frozen study resolves its direct pinned references without needing an index.

## Runtime meaning

`probe_observation.py` installs response-scoped hooks inside the generation's
existing model-hook session. The decoder mapping is explicit, and PEFT wrappers
are unwrapped to their base decoder before resolving that mapping. There is no
largest-module guessing and no global observer state. Hooks are removed on normal
completion or failure. No forward pass is added and no sampling RNG is consumed.

A residualPre hook reads block inputs. A residualPost preAction hook precedes the
existing steering hook; postAction follows it. Existing steering arithmetic is
unchanged. PreAction does not mean globally unsteered: earlier layers or tokens
may already have changed. P4 can use the same site and recording-stage contract;
this slice does not implement new action providers or control flow.

Observed tensors are detached and transferred to CPU float64 for the same explicit
standardization/affine/ReLU arithmetic as fitting. JSON conversion follows scoring.
This incurs transfer and synchronization overhead, which remains unmeasured on
research-scale models. Model binding and tensor width/precision are checked, not
inferred from names. Scores retain the artifact's fixed threshold and score kind;
they are not calibrated probabilities.

Offsets advance across chunked prefill and ordinary decode. Prefill observations
cover the final or every prompt token as declared by the probe; decode observes
each naturally consumed generated token. Input position t predicts t+1. The final
sampled token is not observed without a subsequent pass; no pass is added to fill
that gap. Saved token IDs, absolute positions, and generated-token indices identify
alignment without guessing character offsets. Training population and observed
population are reported separately; population transfer needs its own evaluation.

A direct-logprob-only study does not execute this response observer. It must select
sampled text, or remove these measurements; the engine never silently manufactures
generation for a new measurement. This remains separate from the legacy reader
score over replayed output text and from J-lens readout.

## Evidence and resources

Successful response readings are embedded in `generations.jsonl`; panel turns also
carry them in `turns.jsonl`, and flattening preserves the same measurement document.
The existing response writer therefore supplies durability, resume keys, shard
identity, and bundle collection. Completed responses are not remeasured on resume.
A failure saves a unique `probe-failure-*.json` beside partial evidence, not a
completed response. Existing files are never overwritten by this failure writer.

The declaration bounds reading count across selected probes per response. Later
scheduled readings are counted as omitted. Full activation retention is opt-in,
with a separate bound on encoded activation-array bytes; it is not a total process
memory bound or total evidence-size bound. Scores remain when the activation
budget is exhausted. Shared limits can favor earlier hook execution; size the
budget to the desired schedule and inspect omissions before comparing probes.

Non-finite or failed score calculations either produce explicit missing readings
or stop, according to `onError`. Incompatible model/tokenizer bindings and invalid tensor rank/batch shapes
stop because the declared observation cannot execute. Observed width/precision
mismatches follow the selected missing-score policy. Unexecuted expected readings
are counted separately from budget omissions. Missing readings never become zero
scores or negative labels.

Results → Generations presents scores, stages, token positions/IDs, and missing
reasons next to the response, with a measurement-ID filter. It retains the existing
bounded generation-preview scope (first 80 responses); full files remain evidence.
A dedicated aggregate probe-analysis dashboard is not introduced here.

## Verification and remaining program

Acceptance covers independent arithmetic at both block sites, action ordering,
chunked positions, scope filters, budgets, errors/hook teardown, a real tiny
Transformers sampled generation with identical output/token/RNG state, shared
review/save and frozen pins, and Swift/Python manifest round trips. Further journey
checks exercise actual condition writing/resume, panel seats, and freeze relocation.
These fixtures establish the implemented contract; they do not qualify CUDA/MPS
performance, arbitrary model architectures, or live app interaction.

The historical lazy-import AST audit still proves its original migration at the
landed `b1190f7` code, checks the current lazy wrapper and every other body, and
excludes only the deliberately extended `run_scenario` from an unchanged-body
claim. Its P3 behavior is covered by runtime tests. No prior baseline was changed.

P4 remains the intervention-runtime adapters and ordering work, P5 conditional
policies, P6 broader decision evidence and remote parity, and P7 live qualification.
