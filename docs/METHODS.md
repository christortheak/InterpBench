# SteerLab Methods

Status: living document, started 2026-06-10; refreshed 2026-07-12 (with
2026-07-20 status updates in §Behavioral design and §Layer/alpha selection);
amended 2026-07-29 (§Reference and centering policy, two same-day
amendments).

This document records *how* the steering methodology works and where it sits
relative to its published sources — the material a methods section is written
from. It is deliberately concept-agnostic: it describes the apparatus, not any
particular study run on it. For a first pass through the instrument see
[ONBOARDING.md](ONBOARDING.md); for the order to run a study in, see
[CONDUCTING-A-STUDY.md](CONDUCTING-A-STUDY.md); for every verb and flag, see
[CLI-REFERENCE.md](CLI-REFERENCE.md).

## Source lineage

Two papers anchor the approach:

- **Zou et al., "Representation Engineering: A Top-Down Approach to AI
  Transparency" (RepE)** — the program: read and control concept-level
  representations directly from the residual stream via stimulus sets and
  linear methods, skipping circuit-level analysis. Signature direction-finding
  method: LAT (PCA over difference vectors). Reads at the last token of a
  stimulus.
- **Sofroniew, Kauvar, et al., "Emotion Concepts and their Function in a
  Large Language Model" (Anthropic interpretability, Transformer Circuits,
  April 2026)** — a deep single-domain instance of the program on Claude
  Sonnet 4.5. Headline math is **difference-of-means** (not PCA); PCA appears
  only as nuisance removal (projecting out top PCs of neutral-corpus
  activations). Steering = activation addition at mid-late layers, strength
  reported as a fraction of residual-stream norm, swept ±. Key result for our
  purposes: +desperation raised reward-hacking ~14× *with no visible
  emotional text* — behavior moved while prose did not.
- Also in the family: **Rimsky et al., Contrastive Activation Addition
  (CAA)** — paired contrastive prompts, mean difference, the data format we
  use (`positive.jsonl`/`negative.jsonl` pairs).

The emotion paper's invisible-influence result — behavior moving while prose
did not — sets the measurement problem this instrument is built around.
A steering intervention can move what a model *decides* without moving how it
*talks*, so a study has to be able to separate the two: an outcome endpoint,
a reasoning-style endpoint, and a surface-vocabulary endpoint, measured on the
same generations. Every measurement facility below — outcome instruments,
pinned reasoning-style taxonomies, marker density as a diagnostic rather than
an objective — exists to keep that decomposition available to whatever concept
and task a workspace declares.

## Current method (implemented, Phase 0)

### Extraction

| Step | Choice | Source |
|---|---|---|
| Stimuli | Hand-written contrastive pairs, content-matched line-by-line; versioned JSONL, SHA-256 hashed into all artifacts | CAA format |
| Forward pass | Raw text, prefill only — no chat template, no sampling | (clean-room choice) |
| Reading position | Last token of each stimulus, every layer | RepE |
| Direction | mean(positive) − mean(negative), per layer | Emotion paper / CAA |
| Persistence | `.safetensors` + JSON sidecar (model id, stimulus hash, per-layer norms, date) | reproducibility spec |

Note: with equal pair counts, "average of per-pair differences" and
"difference of class means" are algebraically identical for mean-difference;
the distinction only matters for PCA-based extraction (see options below).

### Injection

- Activation addition `α·v` into the float residual stream after a chosen
  block, at the last position, **on every forward pass** (prefill and each
  decode step). The per-token guarantee is unit-tested on tiny in-memory
  models of both families and asserted live by the smoke test — steering only
  during prefill silently produces near-null results and is the classic
  implementation bug.
- Quantized weights are never touched; the residual stream stays bf16/fp16.
- Steering strength is reported in units of the layer's typical residual
  norm (the emotion paper's convention). Empirical motivation from the toy
  run: Qwen3-4B mid-layer difference-vector norm ≈ 10, Gemma-3-4B ≈ 1576 —
  raw α is meaningless across families.

### Ablation (directional removal) and neutral-mean centering

Ablation removes a direction by projection — `h′ = h − λ·P h`, where `P`
projects onto the span of the ablated directions ("the model cannot represent
this here"), applied at **every position and, by default, every layer**
(a single-layer removal is usually rewritten by the layers above it). λ = 1 is
full removal, λ = 2 a norm-preserving reflection. The projection math
(modified Gram-Schmidt orthonormalization, `h′·v̂ = 0` at λ = 1, exact norm
accounting, one ablator per chain running first) is unit-tested on both
engines — see `SteeringKit/Injection/SubspaceAblator.swift` /
`steerlab_server/steering/ablator.py`.

**The carrier problem (measured 2026-08-06).** Raw extracted concept vectors
must NOT be ablated at λ = 1: they routinely share a large component with the
**neutral-corpus residual mean** (the stream's "carrier" direction), and
differences of means do not cancel it. The effect is large enough to see
immediately: on a real set of extracted `gemma-3` vectors, cross-concept
cosines averaged ≈ 0.5 (individual pairs up to 0.98) where distinct concepts
should be near-orthogonal, with roughly three quarters of each vector's norm
lying along the single shared direction. On the 4B
testing tier, ablating a raw concept vector at λ = 1 — full band, middle
third, or even one middle layer — collapses generation into single-token
repetition; ablating the neutral-mean direction itself does the same, and a
matched-norm random direction does not. Qwen3 vectors from the same stimuli
have low mean-alignment (|cos| ≤ 0.28) and do not collapse: the collapse
tracks mean-alignment, not ablation per se.

**Mean-centering (the fix, declared and never silent).** The ablation
direction is centered against the neutral mean per layer —
`v ← v − (v·m̂)m̂` — which removes exactly the shared carrier component,
leaves the concept-specific part, and empirically restores full coherence
under λ = 1 band ablation. Mechanism (identical cross-engine semantics):

- **Artifacts carry the mean.** Extraction with a neutral corpus now stores
  the per-layer neutral mean in the vector artifact
  (`neutral_mean_layer_<i>` tensors + sidecar `neutralMeanSource`,
  additive keys; the corpus is already pinned by `neutralCorpusHash`).
  Legacy artifacts have no mean — re-extract to enable centering.
- **Centering is declared, not implied.** Playground: a visible toggle
  ("Center ablations on the neutral mean", default ON — the uncentered
  default was a reliable dead end), applied only when the artifact stores a
  mean, advisory shown when it cannot be. Variants/agents and API cells: an
  explicit `"centering": "neutralMean"` key on the injection (absent =
  none; the default is never encoded, so existing artifact bytes and hashes
  are untouched); declaring it against an artifact with no stored mean
  refuses loudly. Study-manifest slots do not yet take a centering key —
  study ablations get the preflight diagnostic below, and manifest-declared
  centering is a documented follow-up.
- **Preflight diagnostic.** Every ablation-cell builder (study conditions,
  variants, API, Playground) computes the per-layer |cos(direction, m̂)|
  when a mean is available and warns loudly — naming the worst layer — when
  it exceeds **0.35** (`ablationMeanAlignmentWarnThreshold`, calibrated:
  coherent at ≤ 0.28, collapse observed from ≈ 0.45 mean alignment). No
  stored mean → the warning says the check is IMPOSSIBLE, never "safe".
  Frozen-study semantics are never changed by the diagnostic (post-submit
  drift policy: continue loudly). The `randomDirectionAblation` control is
  exempt — an arbitrary direction is its point.

Interpretation note: centered ablation removes the concept's mean-free
component. The carrier component is shared across concepts (that is the
measurement above), so what centering discards is precisely what could not
have distinguished one concept from another. Cross-engine fixture tests pin
the centering math (`test_ablation_mean_centering.py` /
`AblationMeanCenteringTests.swift` — same numbers on both engines).

### Validation instruments (concepts panel)

- **Held-out accuracy** (headline): a deterministic ~20% of stimuli are
  excluded from the direction and classified by projection against the
  training-means midpoint. Out-of-sample by construction.
- **Split-half cosine**: directions from disjoint stimulus halves;
  measures whether the stimuli agree on what the concept is.
- **Rebuild stability**: cosine vs the previous rebuild; convergence signal.
- **Norm-by-layer curve**: where in the network the contrast lives.
- **Per-stimulus margins / outliers**: distance from the decision midpoint
  along the direction; rewrite/prune candidates.
- **Discriminant cosines** against permanent control concepts
  (`negative-valence`, `arousal`, committed as data). Responds directly to
  the emotion paper's geometry finding (PC1 ≈ valence, PC2 ≈ arousal
  dominate affect space): a "fear" vector must be shown to be more than a
  valence/arousal blend before it supports any claim about that concept
  specifically.
- **Matched-norm random vector** (coherence/placebo control), exercised live
  in the Phase 0 toy run: French concept vector at (L14, α4) on Qwen / (L20,
  α2) on Gemma moved generation into French (32 / 29 marker counts vs 0
  baseline) while the random vector at identical layer/norm/α left output
  unchanged (0 markers).

Methodological stance: panel stats are stimulus-*design* aids (reliability),
not construct validation. Iterating stimuli against in-panel stats is
legitimate measurement refinement; the gates before experimental use are
(a) held-out scenario validation where the concept is evoked *without being
named* (the emotion paper enforces the never-named property in validation
sets, not training stories), (b) discriminant separation from nuisance
directions, (c) on the Gemma track, comparison against Gemma Scope 2 SAE
features, and (d) freezing (committing) stimulus files before behavioral
runs — steering manipulations must remain independent of the outcomes they
will be tested on (the circularity constraint the freeze lifecycle enforces).

### Behavioral design (instruments landed 2026-07-01)

Inherited from the emotion paper's blackmail/reward-hacking design: paired
baseline-vs-steered generations per item, signed strength sweeps, a
positive-control concept that *should* move the declared outcome, a
matched-norm random vector, and a capability battery under every steering
condition (their "+anger disrupts planning" non-monotonicity is the cautionary
case — extreme steering breaks the task rather than biasing it). Measures, in
order: the outcome endpoint (flip rate, or choice probability); magnitude on a
predefined ordinal scale; reasoning-style features; concept vocabulary (the
visible-vs-invisible contrast).

2026-08-10, a parser lesson worth generalizing: the built-in duration grammar
was extended to accept number words — one through twelve wherever a digit
stands in a compound or single term ("ten years and six months" = 126 months;
ranges stay digit-only) — identically in the built-in parser and the parser
registry's `durationMonths` kind on both engines. The old digit-only grammar
left 58 of 1320 records unparsed **dose-dependently** (steering pushed the
generation toward the formal register, where number words are more common),
which biases treated conditions relative to baseline. The general rule: an
endpoint parser that fails at a rate correlated with the intervention is a
confound, not a nuisance, and must be checked as one. Server `analyze`
additionally rescues null run-time parses under the pinned grammar (null-only
— stored parses are never overwritten; stamped in `endpoint-reparse.json`), so
finished runs get such a fix without regeneration.

Status 2026-07-13: the answer-token logprob / choice instrument, the
categorical-choice and duration parsers, the derived outcome endpoints
(outcome rate, between-arm gap, rule-vs-standard interaction, anchor slope,
proportionality),
matched-norm random as a first-class `controlType` with control-matrix
templates, and the paired statistics (bootstrap CIs, Wilcoxon, BH-FDR/Holm)
exist in code on both engines — `analyze` is a CLI verb on both, local run
`report.json` carries `effectSizes` (+ `effect-sizes.csv`), and the
capability battery runs per-condition inside `run` on both engines
(`conditions[].capabilityBattery`, `battery.jsonl`). Reasoning-style
features landed 2026-07-13 on both engines as **taxonomy-as-pinned-data**: a
versioned JSON feature file (`prompts/taxonomies/<name>.json`; `wordList`
whole-word/phrase features and portable-subset `regex` features — no
lookbehind, rejected at load; normalization `perSentence` | `per1kWords` |
`rawCount`) is hash-pinned into the manifest (`reasoningStyleTaxonomyPath` +
`reasoningStyleTaxonomyHash` — drift is a verify violation; no pin = no
scoring). Scoring is deterministic text math at metrics/report time (no
model, no judge): per-generation `rs_<featureID>` columns in metrics.csv,
per-condition `reasoningStyle` means in report.json, and `rs_<featureID>`
endpoints in the paired effect-size machinery on both engines — so "the
decision moved without the style moving" is a measured contrast, not an
impression. Post-hoc recompute: `experiment rescore-style` (both CLIs;
epoch-guarded; writes `reasoning-style.csv` + `reasoning-style.json` into a
NEW run directory). Cross-engine parity is fixture-locked to 1e-9
(`reasoning-style-parity.json`, committed to both test trees). Like marker
density, these are surface features — style endpoints beside, never instead
of, outcome endpoints.

## Method options (flexibility matrix)

The steering core must support these as configuration, not code changes —
both to run the planned method comparisons and to keep artifacts portable to
larger hardware:

| Axis | Options | Status |
|---|---|---|
| Direction finding | mean-difference \| LAT (RepE C.1: per-pair differences **L2-normalized before PCA** so high-norm pairs cannot dominate PC1; fed to PCA in alternating ± orientation, reproducing the paper's unsupervised random pairing deterministically; sign fixed by the directionality of per-pair scores, stable exactly where PC1 diverges from the mean difference; then norm-matched to the mean difference so α stays comparable — our deliberate addition) \| emotion grand-mean (concept mean minus population mean over a multi-concept story corpus) | paired methods implemented in the extractor (`ExtractionMethod`; Gram-matrix PCA); current LAT-vs-CAA is a direction-only comparison because LAT is norm-matched to the CAA vector; emotion grand-mean math, schema, UI orchestration, and server extraction are implemented |
| Reading position | last token \| mean over tokens from index k (emotion paper uses k=50 on ~paragraph stories) | **both implemented** (`ReadingPosition`; activation caches key on it) |
| Data contract | CAA paired positive/negative files \| RepE paired reader prompts \| multi-concept emotion stories \| neutral corpus \| held-out probe validation items | **schema implemented** (`StimulusSet.loadPairs`, `loadMultiConceptTexts`, `VectorExtractionRecipe`, `ReadingProbeArtifact`); prompt templates live in `prompts/generation/` |
| Scalar estimation | projection onto calibrated reading direction; sign-oriented so higher scores mean more concept-positive; centered at class midpoint and scaled by projection variance | implemented as math/artifact type and exposed through probe data import, probe training, and chat highlighting; treat the workbench probe as diagnostic unless trained on genuinely independent probe data |
| Confound removal | none \| project out neutral-corpus PCs from a token-position activation bank | neutral corpora and PC bases are implemented for steering-time and variant-time removal; scientific quality depends on corpus design, token-position coverage, and substrate-specific validation |
| Injection layers | single layer \| contiguous band (vector per layer, centered, odd width) | **both implemented** (interventions are a list — a band is N injectors) |
| Strength units | raw α \| fraction of typical residual-stream norm at the layer | scale conversion is implemented — the raw scalar is α·r/‖v‖, so the injected perturbation's L2 norm is exactly α·r for whatever denominator `r` is supplied. The current denominator is recorded in the sidecar (`residualNormSource`) and is provisional when measured from one pooled activation per neutral text. The token-position neutral bank now exists and supplies a second, token-weighted denominator (`residualNormSource: "neutral-token-bank"`); the pooled and bank estimators weight texts differently, so α is comparable only within one `residualNormSource` — nothing warns on mixing yet (2026-07-12), so keep one convention per study. |
| Strength resolution | fine-grained α | **implemented**: 0.1 slider steps raw / 0.01 in norm units, plus numeric entry; sweep grids provide the dose-response curves and the statistics layer computes dose-monotonicity |
| Injection composition | single vector \| linear mix h + Σ αᵢ·vᵢ (per-vector α and band center; shared band width and units) | **both implemented** — injectors compose additively, so a mix is several injectors. Uses: concept composition (fear+authority), counter-vector pairs (the emotion paper's desperate-vs-calm), and empirical purification (negative α on a control concept cancels its component). Behavioral compositionality of steering vectors is largely untested in the literature — interaction conditions are a candidate contribution, but mixes multiply the condition matrix; treat each mix as a first-class hashed condition in run configs. Note: subtracting α·v_control is not geometric projection; a true orthogonalized "derived vector" operation (v ⊥ v_control, persisted with its own provenance) is a planned follow-up. |

## Dataset recipes

Vector generation methods now have separate data contracts instead of one
universal "concept prompt" folder:

- **CAA:** `prompts/concepts/<concept>/{positive,negative}.jsonl`, paired
  or at least class-matched sentence stimuli.
- **LAT paired direction (RepE-inspired):** `prompts/repe/<concept>/pairs.jsonl`, one
  `{"positive": "...", "negative": "...", "split": "train|test"}` object
  per line, with optional answer scaffolds/templates. The same family also
  produces calibrated scalar probes (`ReadingProbeArtifact`) from held-out
  labeled activations.
- **Emotion grand-mean:** `prompts/emotions/<dataset>/stories.jsonl`, one
  `{"concept": "...", "topic": "...", "text": "...", "split": "..."}` per
  line. Vectors are concept mean minus the grand mean over the whole
  multi-concept corpus.
- **Neutral/background:** `prompts/neutral/corpus.jsonl`, ordinary text for
  residual-norm calibration and, after the activation-bank build, neutral
  PCA. This is not the negative class.
- **Probe validation:** `prompts/probes/<probe-name>/items.jsonl`, held-out
  labeled examples for diagnostic scoring, never vector extraction.

LLM-drafting prompts for each dataset family live in
`prompts/generation/`. Generated data is always draft material until
reviewed, edited, committed, and pinned by hash.

## Reference and centering policy for grand-mean extraction

**Amendment 2026-07-29 (i).** (An earlier commit of this section was
misdated 2026-07-28; both amendments in this section were made 2026-07-29.)
Context: adopted during a corpus-authoring push, before any study was frozen,
following a recipe-selection design review. The trigger was
an asymmetry with the source: the emotion paper's grand mean is defensible
because 100+ roughly balanced sibling concepts in one format make the
centroid ≈ "concept-general narrative," so centering removes shared
register and leaves each concept's distinctive part. A hand-authored family is
typically far smaller (≈9–12 story concepts), and at that size the same
construction has two failure modes this policy exists to avoid.

**The two failure modes.**

1. *The centroid is not neutral.* The grand mean over a small, unbalanced
   multi-concept story corpus is itself a loaded direction (for a family of
   character-trait concepts, something like "admirable character acting
   deliberately"). Centering on it yields "concept X relative to
   the-family-in-general" — sometimes wanted, but it must be chosen, not
   inherited from corpus arithmetic.
2. *Small-N degeneracy.* Centering each concept on a centroid that
   includes its own corpus forces the centered vectors to sum to zero:
   they are linearly dependent by construction, pairwise cosines carry a
   built-in ≈ −1/(N−1) bias, and any family composite formed as the mean
   of centered vectors is identically the zero vector. Negligible at
   N > 100; structural at N = 9.

**Policy.**

1. **Recipe selection.** CAA (paired mean-difference) is the recipe for
   concepts with a minimal counterfactual twin or where the contrast class
   *is* the concept (states and presences: fear, hungry, tired, the
   placebos; axis/register concepts: formality, assistant-persona — the
   negative class defines the direction's semantics and is chosen
   deliberately). Grand-mean is the recipe for diffuse whole-passage
   concepts with no non-trivializing minimal pair (character traits,
   dispositions, stances, story-format emotions). RepE/LAT is never a
   primary extraction recipe: it serves triangulation (convergence of PC1
   with the mean difference is evidence the direction is geometry, not
   recipe artifact; divergence is a confound alarm) and the reading/probe
   side.
2. **Reference corpus.** Grand-mean concepts are referenced against a
   **matched-neutral-stories corpus** — authored in the workspace at
   `prompts/emotions/neutral-stories/stories.jsonl`: plain, concept-free
   tellings authored on the same topic grid, length band, and register as
   the concept story corpora. Semantics: "this concept, relative to an
   ordinary telling of the same scenes." The procedural neutral corpus
   (`prompts/neutral/corpus.jsonl`) is NOT a semantic reference for story
   concepts — story-mean minus procedural-mean carries a large
   narrative-vs-procedure register component — and remains reserved for
   residual-norm calibration and neutral-PC work. The multi-concept
   centroid is not the default reference (failure modes above).
3. **Family centering.** Where family-relative directions are wanted (the
   distinctive component of each member), center **leave-one-out** (each
   concept against the centroid of its siblings only), applied *after*
   matched-neutral referencing, and report the two-stage decomposition:
   the family common component (the "halo") and each concept's residual.
   A concept's own corpus never appears in its reference. Family composite
   agents steer the common component — never the mean of centered vectors
   (identically zero, per failure mode 2).
4. **Standing diagnostics.** Every grand-mean family report includes: the
   cross-concept cosine matrix; each vector's cosine to the family
   centroid; the PCA scree over the family's vectors (effective
   dimensionality — "nine concepts or one halo" is a reportable finding);
   and, transitionally, each concept's cosine between its
   matched-neutral-referenced and procedural-neutral-referenced versions,
   which quantifies the register confound the old default would have
   introduced.
5. **Negative-dose semantics.** Under CAA, −α steers toward the defined
   contrast class; under grand-mean, −α steers toward the reference —
   i.e., toward matched-neutral ordinariness under this policy. Reversal
   claims in reports must be phrased accordingly.

**Implementation note.** The flexibility-matrix row "emotion grand-mean
(concept mean minus population mean over a multi-concept story corpus)"
describes the implemented default, which this policy supersedes for study
use; the reference-corpus selector LANDED 2026-07-31 as the first-class
`designatedReference` method on both engines (attach pins the reference
{name, hash}; pooled reading from token 50 is set by attach as method
policy; the reference enters canonical recipe identity via
methodParameters). The pre-landing workaround remains valid history: the
matched-neutral-referenced vector was computable through
the mean-difference pathway by loading the concept stories as the positive
class and neutral-stories as the negative class (class-mean difference —
identical math, no pairing required); such vectors should be stamped with
the reference corpus hash like any other stimulus input.

**Amendment 2026-07-29 (ii): terminology, register matching, and building a
family deliberately.** Context: adopted the same day as (i), after a design
review of a lone abstract concept concluded that (a) no corpus family existed
of which it was naturally a member in the emotion paper's sense, and (b) the
right response is to author one deliberately rather than to leave the concept
as a lone vector with no discriminant context.

1. **Terminology.** What rules 2–3 of amendment (i) describe is a
   **designated-reference mean difference** — concept mean minus the mean
   of a deliberately authored reference class — and should be called that.
   The name "grand mean" is reserved for true family-centroid
   constructions, which are not usable as zeros at hand-authored family
   sizes. (The `prompts/emotions/` directory layout and story schema are
   retained for designated-reference concepts; the label describes the
   math, not the file location.)
2. **Register matching.** The designated reference must share the concept
   corpus's *register*, not just its topics: narrative concepts (traits,
   dispositions, story-format emotions) reference a neutral-stories corpus;
   expository concepts reference a plain-exposition corpus built on the same
   subject grid. Subtracting a narrative reference from an expository corpus
   injects an essay-vs-story register component into the vector.
3. **Authoring a family.** Where a concept only becomes interpretable
   against siblings, author the whole family: equally elaborated corpora on
   one shared subject grid, each written to the same counts, length band,
   and register mix, all referenced against the same designated reference.
   Equal elaboration is a validity requirement — unequal energy or
   abstractness across members turns the family geometry into a measure of
   *authorship* rather than of the construct, so a family-level blind
   confound check (valence and abstractness flat across members) is a gate.
   The family centroid is **named, not inert**: state in words what the
   members share, and never use it as a zero. The two-stage decomposition
   then applies — designated reference first, then leave-one-out family
   centering for the common component and each member's distinctive
   residual, both steerable objects. Author at least one **out-of-family
   confound corpus** (something that shares the family's surface energy
   without its defining move); it never enters the centroid. A member with
   no endpoint in the planned study may be authored for geometry only, and
   should be labeled as such.
4. **Diagnostics extended.** Family reports add: the family cosine matrix
   (including the confound corpus and the designated reference as impostor
   rows); a blind identification confusion matrix over sampled passages
   (discriminability at the text level, before any vector is computed);
   and the family PCA scree.

## Artifact-pinned concepts (landed 2026-08-10, server engine)

Everything above pins a **recipe**: stimuli by hash plus extraction options,
re-derived deterministically at every run. That is the firewall's normal
shape, and it is the right shape whenever a direction *is* a function of
pinned data.

Some legitimate directions are not. Family centering (policy rule 3 above) is
computed **from other vectors**, not from stimuli: the centred direction for a
family member is its raw vector minus the family mean of raw vectors. Nothing
in the concept's stimulus set produces it, and re-running extraction on those
stimuli reproduces the *un*-centred vector. The same is true of any post-hoc
re-referencing, of any composite built by arithmetic over existing artifacts,
and of an imported direction whose derivation happened outside the engine.

For these, the manifest pins the **bytes**. A concept may attach with

```json
{
  "name": "formality-dr",
  "stimulusSetHash": "<the SOURCE stimuli's hash, copied from the sidecar>",
  "options": {"method": "pinnedArtifact",
              "readingPosition": {"meanFromToken": {"_0": 50}}},
  "validationHash": "<the source concept's held-out validation.jsonl>",
  "designatedReference": {"name": "plain-exposition", "hash": "…"},
  "vectorArtifact": {
    "path": "runs/20260810T045146213-derived/formality-dr",
    "sha256TensorHash": "…",
    "sha256SidecarHash": "…",
    "sourceMethod": "designatedReference",
    "sourceConcept": "formality",
    "residualNormSource": "neutral-corpus",
    "normCorpusHash": "…"
  }
}
```

`path` is the extension-less artifact locator (`<path>.safetensors` +
`<path>.json`), and **both** files are hashed. The firewall property is
unchanged in kind: what the manifest pins is what the run reads, and drift
refuses loudly rather than changing silently. Both hashes are re-checked at
every `verify()`, at freeze, and again at the moment of use; a change to
either — including a sidecar edit that leaves the tensors alone — is a
violation exactly like stimulus drift. The pinned pair also enters the
validation **scope** hash, so evidence minted against one set of bytes cannot
certify another.

**Where the provenance lives.** In the artifact's own sidecar, which is what
the sidecar hash pins: the derivation formula, the family and its members, the
per-layer norm retained, the base concept, the date. The manifest deliberately
does *not* restate the derivation — it pins the sidecar that carries it, so
there is one copy of the claim and one hash over it. Attach copies only what
the lifecycle must act on without opening the artifact: the reading position,
the source method, the source stimulus hash, the reference or population when
the source recipe had one, and the residual-norm provenance.

**Extraction becomes materialization.** `extract` verifies both hashes and
writes the vectors into the run directory as an ordinary
`<concept>.safetensors` + sidecar pair, stamped `extractionMethod:
"pinnedArtifact"` and carrying a `pinnedFrom` block naming the source path and
both hashes. Everything downstream — validate, the layer × alpha sweep, the
run loop, promotion's recipe matching — sees a normal vector artifact and has
no special case. The materialized copy carries the source's per-layer residual
norms *and* the corpus they were measured on, because α in norm units is only
interpretable against the denominator that was actually measured.

### The residual-norm denominator convention (2026-08-20)

α is reported in units of the residual-stream norm at the injection layer, so
the DENOMINATOR is a measurement with a convention, and the convention has to
be identical on both engines or the same numeral is two different doses. It was
not: the token-bank path on the Python server averaged the residual norm over
BANKED positions only — the deterministic row-cap draw — while Swift averaged
over every corpus position. On a downsampled neutral corpus those differ.

**The convention is the WHOLE-CORPUS average**: every measured position counts
toward the denominator, banked or not, because the denominator describes the
corpus and not the draw. The server was corrected to match, and both engines
now accumulate through one shared tested unit
(`ResidualNormConvention` / `steering.residual_norm_convention`).

Fresh measurements stamp the sidecar `residualNormConvention:
"wholeCorpusMean-v1"` — same key and value on both engines, absent when
unknown. The stamp is **never retro-applied**: an artifact without it is
legacy, is read exactly as it always was, and is displayed as
"convention: legacy (pre-stamp)" rather than being credited with today's rule.
`vectors backfill-norms` re-measures under the current convention and stamps
it, which is the opt-in way to move a specific artifact forward. `α` remains
comparable only within one `residualNormSource` AND one
`residualNormConvention`.

**Validation is unchanged.** The held-out probe runs exactly as it does for a
derived vector: the same class means, the same scoring, the same cosine
matrix — read at the artifact's own reading position (a manifest that declares
a different one is refused, not quietly measured at the wrong depth), over the
**source** concept's data. Post-hoc directions are normally renamed
(`formality` → `formality-dr`) while keeping the base concept's stimuli and
held-out set;
`sourceConcept` records that, and the source data stays under the same hash
pins it would have had as a recipe concept.

**When this is legitimate, and when it is not.** It is legitimate when the
direction genuinely has no stimulus recipe — derived by arithmetic over other
artifacts, or imported — and its derivation is recorded in the artifact it
pins. It is *not* a shortcut for a direction that does have a recipe: pinning
bytes where a recipe exists throws away deterministic re-derivability and the
cross-substrate re-extraction path for nothing. Two consequences follow from
the artifact being substrate-stamped: a pinned artifact never transfers
between engines (materialization refuses a foreign substrate, as every load
path does), and a study built on one is therefore bound to the engine that
produced the bytes. Validation obligations do not relax at all — a pinned
direction earns its place in a study by moving a held-out probe and by not
collapsing into its neighbours, on the same evidence any extracted vector must
produce.

## Neutral calibration bank

The neutral corpus is the keystone for three things: norm-unit alphas,
cross-layer layer sweeps, and nuisance projection. A corpus of short neutral
texts read only at the last token is not enough. The needed artifact is a
bank of neutral activations sampled across many token positions in many
documents/transcripts, with per-layer residual-norm distributions and PCA
components plus explained-variance curves. The bank build path exists on both
engines (2026-07-06: token-position capture, corpus-hash-seeded deterministic
downsampling, persisted PCA bases via `NeutralPCStore`), with two caveats
(2026-07-13): the capture stage accumulates all configured layers × tokens in
CPU float32 *before* the row cap (restrict layers; keep bank corpora modest),
and a manifest cannot yet pin a token-bank projection into
`ExtractionOptions` — projected grand-mean extraction is exploratory-only
(the basis *bytes* are now verified: `neutralPCBasisHash` is enforced against
the file wherever pinned, 2026-07-13 — the open gap is representability, not
the hash). Fixed-k `--project-neutral` remains legacy
draft-only machinery: verification and freeze reject manifests that use it.
A fragile projection is worse than no projection because it can remove
genuine concept signal.

## Layer/alpha selection (sweep)

`steerlab-cli experiment sweep <name>`: per pinned concept, a layer ×
alpha grid (alphas in residual-norm units; greedy decoding, so cells are
deterministic) over neutral dev prompts (`prompts/dev/dev-prompts.jsonl`),
scoring three things per cell — concept expression (data-driven marker
rubric, `markers.json` beside each concept's stimuli; keyword first,
model-graded second, never model-graded only), degeneration
(distinct-bigram ratio; repetition collapse → 0), and the capability
battery (`prompts/batteries/basic.jsonl`; if accuracy drops under
steering, a "bias" finding is confounded by degradation). The recommended
cell — highest expression with battery within 0.15 of baseline and
distinct-2 ≥ 0.45 — is written into the manifest as a
`<concept>-recommended` condition **only while the manifest is a draft**:
settings are chosen on the dev split, then frozen, then behavior is
measured.

**Selection is manifest data (2026-07-08).** `sweep.selection` declares the
objective and constraints; absent, it resolves to the historical rule above
(markerDensity, battery tolerance 0.15, distinct-2 floor 0.45). Three
objectives are implemented on both engines: `markerDensity`; `judgeScore` —
paired judging of each cell against the baseline via the manifest's pinned
rubric + judges, baseline pinned at the 0.5 tie; and `logprobShift` — mean
Δ log-probability of a declared target option over a hashed choice-prompts
file, baseline 0. The resolved criterion, dev-split hash, winning cell,
metrics, and matched-norm-control outcome are stamped as a `selection`
provenance block on `<concept>-recommended`, and `promote` mints the agent
from that cell with a `promotion` birth certificate. Marker density is a
diagnostic/manipulation check — never the promotion objective wherever the
study's claim is about outcomes rather than prose, because selecting on
concept vocabulary selects for exactly the surface confound such a study has
to rule out; those screens select on
`judgeScore`/`logprobShift`. The 2026-07-12 caveat (sweep inputs
hashed only ex post) is closed as of 2026-07-20: `sweep.devPromptsHash` and
`sweep.batteryHash` are manifest pins on both engines — freeze pins them,
drifted pins refuse sweep start, and verify() reports drift as a violation.

**The matched-norm control is a screening rule, not specificity evidence
(2026-08-03).** `sweep.selection.controls` declares how the margin is
applied: `applyTo: "winner"` (default, historical) controls the argmax cell
alone; `applyTo: "topK"` with `topK: K` walks the top K promotable cells in
objective order (ties break by declared grid order — a cross-engine
contract) and promotes the first that beats its *own* matched-norm random
control, stamping every evaluated control into the recommendation's
`controlsEvaluated` provenance. Both modes compare each candidate against a
*single* deterministic random draw, and topK tries up to K such
comparisons, so the chance that some candidate passes a noisy control grows
with K — keep K small (~3) and read a pass as "survived the screen," never
as evidence the *direction* (rather than any vector of that norm) moves the
endpoint. Direction specificity is established downstream: the confirm
stage runs its declared matched-norm-random control conditions on held-out
data with paired statistics. A statistically stronger screen (several
random draws per candidate, compared against their distribution) is a
recognized future upgrade for both modes.

## Per-response coding instrument (2026-08-04, both engines)

The paired judge answers "which response is preferred?"; the coding
instrument answers "what does each response contain?" — the shape used in
the vignette-study literature's coding appendices (a per-response LLM coder,
validated against blind human coders, κ ≈ 0.8). A rubric file under
`prompts/rubrics/` opts in with a strict frontmatter block (`mode: perResponseCoding` + one `field: <name> <type>
[optional]` line per declared field; types boolean / integer / number /
string / enum(a|b|…)); the schema rides inside the rubric file, so the
existing `judgeRubricFile` + `judgeRubricHash` pin covers it — freeze,
drift refusal, and provenance come for free, and a rubric with no
frontmatter is a paired rubric, byte-for-byte unchanged.

`evaluate` detects the mode from the pinned rubric and forks after the
shared prelude (epoch guard, exclusions, judge roster, provider
preflight): every sampled-text record — **baseline included, no pairing,
no winner** — goes to each judge individually and blinded (task prompt +
one response; never the condition, never a second response). Codes are
validated against the declared schema with the same
retry-once-then-refuse closure as paired verdicts (invented data is never
recorded), streamed row-by-row to `codings.jsonl`, and summarized in
`coding-report.json`: per-condition per-field aggregates (trueShare /
mean / counts, nulls reported never imputed), engine-computed word counts
(`wordCount` per row — the reasoning-length measure is never
judge-estimated), and per-field inter-judge agreement (percent + Cohen's
κ for categorical fields, mean absolute difference for numeric ones — κ
being the paper's own coder-validation statistic). The coding prompt
wrapper is byte-identical across engines, pinned by committed goldens in
`prompts/fixtures/coding-judge/`.

Paired-only machinery refuses a coding rubric loudly rather than
mis-running it: the sweep's `judgeScore` objective (a coding rubric
declares no preference), structured comparison prompts, `humanValidation`
(paired-shape labels), deferred judging packets, and evaluate resume.
A per-response coding rubric is an ordinary versioned file under
`prompts/rubrics/`: a handful of declared fields (say, two booleans recording
whether a given kind of justification appears) coded per free-text response.

## LoRA adapters — fine-tuning as a contrast intervention

Implemented on the local MLX path and the server PyTorch/HF path, with an
important substrate distinction. The local app trains MLX adapters
(`adapters.safetensors`) over the loaded MLX model. The server trains HF PEFT
adapters (`adapter_model.safetensors` plus config) over full-precision HF
models. These artifacts are not interchangeable; each belongs to its own model
substrate and must be reported that way.

**Scientific role.** A second intervention type to contrast with steering:
a model given a disposition by *weight-level training on documents* versus
the same disposition installed by *activation-level injection*. Questions:
do the two interventions move the study's outcome endpoint the same way?
Does a concept vector extracted FROM the fine-tuned model strengthen, or
rotate? Is the trained disposition visible in prose where injection famously
is not? This adds the persistent-vs-transient and learned-vs-injected axes to
whatever contrast a study is already running.

**Artifact discipline (mirrors vectors).** Documents → chunked, hashed
dataset (versioned like stimulus sets); adapter = `.safetensors` + sidecar
(base model id + revision, dataset hash, rank/learning-rate/iterations,
final train/val loss, date); immutable run dirs; any vector extracted while
an adapter is active records the adapter id in its sidecar; run configs pin
adapter identity. The capability battery applies to fine-tuned models exactly
as to steered ones.

**Open design choices for study use:** continued pretraining on documents vs
instruction-style masked completion; whether a study compares adapter-only,
vector-only, and adapter-plus-vector variants; whether vectors are extracted
from the base model or the adapted model; and how much held-out validation is
needed before an adapter is evidence-grade.

## Portability stance

There are now two active substrates. The local app uses MLX-Swift on Apple
silicon. The server uses PyTorch/HF on Linux/CUDA or Mac MPS for development.
The shared contract is the artifact layer: JSONL stimuli, `.safetensors`
vectors, JSON sidecars/manifests, CSV/JSONL run outputs, and pinned hashes.
The substrates should not be expected to produce byte-identical activations.
Cluster vectors and adapters must be re-extracted or re-trained on the server
substrate and validated as separate artifacts.

## Known divergences from the sources (to revisit)

(Resolved former entries: neutral-corpus confound projection is implemented;
the norm-unit denominator now follows the paper's fixed-dataset convention;
LAT now normalizes pair differences per RepE C.1 — all 2026-06-12.)

**Labeling note (2026-07-03).** The
current LAT pathway is a *paired-direction* method faithful to RepE's
Appendix C.1 direction math, but NOT the paper's full template-mediated
reader pipeline (task-template rendering, LAT token position bound to the
scaffold, PCA fit parameters persisted for exact inference). It is labeled
"LAT paired direction (RepE-inspired)" everywhere. The faithful
`RepE reader LAT` recipe landed 2026-07-03 on both engines (hashed template
registry in `prompts/templates/`, fitted reader artifacts with persisted PCA
parameters, exact same-template inference, `repeReaderScore` outcome
instrument, derive-steering conversion with provenance); reader artifacts
are substrate-specific by rule.

1. **Reading template.** RepE reads the last token of an instruction
   template ("Consider the amount of `<concept>` in …; the amount of
   `<concept>` is") — the template focuses the representation. The **CAA and
   LAT paired-direction pathways** read raw stimulus text (last token or
   pooled), a clean-room choice that avoids naming the concept anywhere near
   extraction; this divergence is scoped to those pathways only. The
   `RepE reader LAT` recipe implements the paper's template-mediated reading
   faithfully (registry templates in `prompts/templates/`, including an
   unnamed clean-room scaffold stamped as a divergence). Revisit the
   raw-stimulus choice if paired-direction probe accuracy is weak at small
   scale.
2. **Stimulus genre.** Hand-written sentences vs the emotion paper's
   model-generated paragraph *stories depicting* the concept at scale
   (100 topics × 12 per concept), never-named enforced in validation sets
   (Phase 1 decision; the panel's Claude generation path exists for this).
3. **Steering positions.** We inject at the final prompt token and every
   generated token; the papers also steer prompt spans (the emotion paper:
   "the token positions of the steered activities", assistant-turn spans)
   and multi-layer bands (we have bands in the chat UI; the sweep
   recommends single layers; RepE's control uses strided bands,
   `range(8, 32, 3)`). **Consequence, and it is a real one:** the model
   reads the task input unsteered — the intervention modulates generation,
   never comprehension of the prompt. That is the cleaner covert-injection
   design, but it narrows what a null result can mean: "steering did not move
   the endpoint" is then evidence about the generation phase only. A
   prompt-span steering arm is the natural robustness check.
4. **Validation scoring.** Never-named scenarios are classified by
   projection against the midpoint of the training-class mean projections
   (CAA-style). RepE normalizes test activations with the PCA model's
   parameters before projecting; the emotion paper uses logistic probes.
   Upgrade if midpoint accuracy looks unstable.
5. **Neutral-corpus design remains experimental.** Neutral corpora and PC
   bases can now be built and selected, but the corpus design is load-bearing:
   domain-neutral text, concept-contrast-neutral dialogue, token-position
   coverage, and target-model substrate all affect what gets removed.
6. Scale humility: source findings are on a frontier model; linear concept
   geometry at 4B–32B may be weaker. Mitigations: validation gates,
   cross-family replication (Qwen3 + Gemma 3), SAE cross-check on Gemma.

## Gemma Scope import convention (decided 2026-07-12, both engines)

An SAE feature imported as a steering vector uses the
**`analyzed-vector-norm-match`** convention: the raw decoder row `W_dec[f]`
is rescaled once, at import, to the L2 norm of the analyzed concept vector
at the report layer —

```text
v_imported = W_dec[f] × ( ‖v_analyzed[L]‖ / ‖W_dec[f]‖ )        (float32)
```

— then placed at layer L of a full-model-depth zero artifact carrying the
analyzed artifact's model identity and residual-norm calibration. This is
the same norm-matching used for LAT directions: at a given α, the SAE
feature perturbs with the same magnitude as the concept vector it was
ranked against, so α stays comparable across origins. Degenerate guard: a
zero-norm row or non-positive target is kept raw (the sidecar's
`rawDecoderNorm` records why). Sidecars stamp `gemmascopeConvention`,
`rawDecoderNorm`, and `gemmascopeTargetNorm`; artifacts lacking the stamp
(pre-convention server imports stored raw decoder rows) warn
"re-import before evidence use" at load. Cross-engine agreement is held by
the `vectors compare` verb (both CLIs, identical JSON) over committed
parity fixtures.

## Reading-position diagnostic (standing, per pooled extraction)

Decided 2026-07-31; scope narrowed same day (external review, finding 6).
Any PAIRED or DESIGNATED-REFERENCE extraction whose reading position is
pooled (`mean from token k`) also reads **last token** from the *same
forward passes* — a second recorder in the hook session, zero extra
compute — and reports the per-layer cosine between the two conventions'
pre-projection mean-difference vectors. Grand-mean extraction does NOT yet
produce the diagnostic: its capture retains the whole corpus's activation
bank, so a second reading position doubles a cost that is already the
path's memory ceiling — implementing it there is deliberate future work,
not an oversight. The server writes
`reading-position-diagnostics.json` beside the vectors in every extract
run directory and logs the min/median/max per concept.

Why it is standing rather than a one-off: the pooled-reading rule for
story-format stimuli needs a justification a reviewer can check, and the
measured gap is that justification. First measurement (gemma-3-4b-it, a
designated-reference story concept, 25 stories/class, mean-from-50 vs
last-token): **median cosine 0.29, min −0.48, max 0.80 across 34 layers** —
the two conventions extract substantially different directions from identical
stimuli, i.e. a last-token read on paragraph stories captures
closing-sentence content, not the concept. "We measured a 0.29 median
cosine between the conventions" is the claim; "the emotion paper pooled
from token 50" is merely its provenance. Because it is re-measured for free
on every pooled extraction, per concept, a study reports its *own* numbers
from its own extract runs on its own substrate rather than citing these.

The diagnostic is a run output, never part of the vector sidecar (the
sidecar is a cross-engine artifact contract), and it compares
pre-projection directions so the confound projection cannot blur the
reading-position question. Server engine only for now; the Swift engine's
extraction remains diagnostic-free.

## Where this implementation goes beyond its sources

Cross-family replication on open models (two vendored families, one
concept-agnostic core); CAA-vs-SAE cross-validation via Gemma Scope 2; a
fully reproducible open pipeline (versioned stimuli, hashed configs,
immutable runs, headless CLI, a freeze lifecycle that fixes settings before
behavior is measured); psychometric design-stats with an out-of-sample
headline during stimulus construction; and a three-layer endpoint
decomposition — outcome, reasoning style, surface vocabulary — measured on
the same generations, so "the decision moved and the prose did not" is a
recorded contrast rather than an impression.

## References

- Zou et al., *Representation Engineering: A Top-Down Approach to AI
  Transparency*, arXiv:2310.01405.
- Sofroniew, Kauvar, Saunders, Chen, Henighan, et al., *Emotion Concepts and
  their Function in a Large Language Model*, Transformer Circuits Thread,
  April 2026.
- Rimsky et al., *Steering Llama 2 via Contrastive Activation Addition*,
  arXiv:2312.06681.
- Gemma Scope 2 (SAEs for every layer/size of Gemma 3):
  `google/gemma-scope-2-<size>-<pt|it>` on Hugging Face.
