# Which engine runs what

**Sampling on either engine.** Local Swift/MLX and Python measured runs use
record-local seeded streams. Positive-temperature runs with `samplesPerItem > 1`
derive seeds from study hash, condition, prompt ID and sample index; otherwise
ordinary runs enumerate the declared seeds. Multi-agent turns deliberately use
an empty condition in that derivation, sharing streams across conditions.
Greedy generation makes no RNG draws and stamps its seed inert; repeated greedy
seeds are not independent observations. Equal seeds across backends do not
promise equal tokens, and GPU repeatability depends on the exact model and
runtime configuration. Preserve the sampling provenance. Measured cross-backend
scope is recorded in the project's technique-parity qualification record, which
lives in the source repository and ships with neither client; without it, treat
repeatability across backends as unqualified.
For categorical outcomes, answer-token/logprob instruments remain temperature-free.

Activations do not transfer between engines. Vectors must be **re-extracted and
re-validated on whichever engine a study runs on** — the parity claim is
structural, not byte identity. Stimulus and corpus SHA-256 hashes *are*
identical across engines, which is what makes cross-engine comparison possible
at all. The freeze evidence gates enforce this: evidence from the other engine
will not satisfy them.

**Authoring is CLIENT-authority by design.** The authoring client's
workspace is the source of truth; the engine on compute hardware never
authors, and its workspace is a cache. The boundary is client versus running
hardware — not macOS versus everything else. `create`, `attach`, the
`pin-*`/`declare-*`/`set-*` verbs, `panel compile`, and `freeze` run on the
local CLI;
execution and analysis (`run`, `evaluate`, `analyze`, `sweep`) answer
identically-shaped envelopes on either engine, and the epoch guard keeps
`analyze` on the engine that produced the run. A server-side refusal whose
repair is an authoring act names the local verb on purpose — go author
there, then submit. Some verbs exist on one engine only; asking the server
for one of them is refused with `error.code: "macAuthorityVerb"` (no
`error.gate` — it describes the engine, not the study) and an
`error.repairAction` spelling the local command. **That code's name is
historical**: it is a stable machine code agents switch on, and it means
"this engine executes, it does not author", never "author on a Mac". Do not
emulate the verb; run the repair where it belongs.

**One verb runs on the compute engine and has no authoring-client verb of
its own.** `steerlab-server battery run <battery-file> --agent <ref>…` reads a
capability battery against one or more agents — `baseline`, a condition spec
`<concept>:<layer>:<alpha>`, or a promoted agent artifact — and writes its own
pinned run directory holding `battery.jsonl` and `battery-report.json`. It
loads models, so it is execution and cannot live on an authoring client, and
it is not manifest-shaped, so no study bundle carries it. Reach it from this
client through the reviewed managed route — `runner science-plan`, then
`runner science-submit`, with a request from the `batteries` method guide
(`steerlab workspace guide methods`) — or run it directly where the engine and
the models are. Either way, cite the report by its pins.

This is the **floor battery**, and it is a different artifact from the battery
a study pins. The pinned one (`batteryEvidence`, `workspace guide freeze`) is a per-condition control
inside a study's own run matrix. The floor one precedes any study: it asks
whether an agent is a working model at all, under a charter that is ex ante,
study-blind and fixed. Two consequences you will meet:

- **A floor battery declares `batteryFormat: 3` and CANNOT be pinned.** It
  carries a second operating regime — long-form generation at a positive
  temperature, several samples per item, read for generation health rather
  than graded — and scored per condition inside a run matrix that would be a
  second outcome measure wearing a control's name. Pin a `batteryFormat: 2`
  battery; run a `batteryFormat: 3` one.
- **Take a floor reading before you compare arms of different kinds.** Before
  claiming a prompted persona and an injected agent differ in behaviour, show
  they are capability-equivalent — otherwise the difference you measured is
  competence, and no analysis afterwards can separate the two. One
  `battery run` naming each arm as an `--agent` does it, and the report is
  keyed by pins so a later study can cite the same reading.

`steerlab-server battery generation-prompt` states the charter in full and is
what you hand an author who is drafting one.
