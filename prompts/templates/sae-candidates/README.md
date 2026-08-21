# sae-candidates-template.json — the SAE candidate roster

The record of *which SAE features a study may seat, in what role, on what
discovery evidence, and what became of each one*.

This directory is **seed/template data**. A real roster lives in the study
WORKSPACE (e.g. `prompts/sae/candidates.json`) and is pinned into one
experiment manifest by hash.

## Shape

```json
{
  "schemaVersion": 1,
  "name": "<roster id>",
  "description": "<free text>",
  "candidates": [ { … } ],
  "pendingConstructs": [ { "constructLabel": …, "role": …, "notes": … } ]
}
```

Every key is CLOSED — an unknown key is a load error, so a typo can never be
silently ignored. Each candidate entry:

| key | required | meaning |
|---|---|---|
| `constructLabel` | yes | the researcher's name for the construct (data, never code) |
| `role` | yes | `focal`, `affectControl`, `embodiedControl`, `domainControl`, `discriminantControl`, `unrelatedTopicControl`, `positiveControl` |
| `model` | yes | base model id, e.g. `google/gemma-3-27b-it` |
| `source` | yes | dictionary id, e.g. `gemmascope-2-res-65k` |
| `layer` | yes | integer layer index |
| `featureId` | yes | integer feature index in that layer's dictionary |
| `neuronpediaUrl` | yes | discovery provenance (http/https) |
| `gemmaScope` | recommended | `{"release": "gemma-scope-2-27b-it-res", "saeID": "layer_40_width_65k_l0_medium"}` — the dictionary in **Gemma Scope's own vocabulary**, i.e. exactly what an imported artifact records in `gemmascopeSource`. Both fields are required once the block is present |
| `discovery` | focal roles only | snapshot: `explanationText`, `accessDate` (ISO), optional `topPositiveLogits` / `topNegativeLogits` / `exampleActivations` |
| `verification` | no | `{"status": "unverified" \| "verifiedOnNeuronpedia", "date": "YYYY-MM-DD"}` — a date is required whenever the status is not `unverified`; an absent block means unverified |
| `qualificationArtifact` | no | workspace-relative path to `sae-feature-qualification.json`, or `null` |
| `status` | yes | `candidate` \| `qualified` \| `rejected` \| `seated` |
| `notes` | no | free text |

`(model, source, layer, featureId)` — plus `gemmaScope` where an entry
declares it — must be unique across the roster: a feature exists only in its
own layer's dictionary, so there is no "same feature at another layer", and
two entries claiming one feature under different construct labels is a
nomination collision to resolve before anything is imported.

## Why `gemmaScope` matters (fill it in at nomination time)

`source` is the **discovery surface's** dictionary name
(`gemmascope-2-res-65k`); an imported artifact and an `saeLatentConditions`
entry record **Gemma Scope's** (`release`, `saeID`). Nothing maps one string
to the other, so a roster entry without `gemmaScope` can only be matched on
`(model, layer, featureId)` — which cannot tell feature 40802 of the 65k
dictionary from feature 40802 of the 262k dictionary at the same layer. With
the block present, the seating guard matches the dictionary exactly and a
mismatch is a verify violation naming both sides.

Fill it from the Neuronpedia source id you nominated from: the Gemma Scope 2
mapping is `<layer>-gemmascope-2-res-<width>` →
`release: gemma-scope-2-<size>-<tuning>-res`,
`saeID: layer_<layer>_width_<width>_l0_<l0>`. Confirm the L0 variant
(`small`/`medium`/`big`) against the SAE you will actually import rather than
assuming `medium`; the entries in this template assume it and are marked for
re-capture.

The block is optional and additive: rosters written before it stay valid and
keep the dictionary-blind fallback, which remains a human check.

`pendingConstructs` records a control SLOT with no feature nominated yet
(fear, hunger, unrelated topic). It carries no feature identity, so an
unfilled control is visible in the study record instead of silently absent.

## Three distinctions the schema keeps apart

- **Discovery** — what the upstream surface said, captured ONCE with an
  access date. Auto-interp labels get regenerated upstream, so the snapshot
  is the only citable record; Neuronpedia must never be a runtime dependency
  for evidence. Required for `focal` entries: a focal feature with no
  recorded discovery evidence is a number, not a candidate.
- **Verification** — a human re-opened the feature page and confirmed it.
  Nothing more.
- **Qualification** — the durable `sae-feature-qualification.json` (held-out
  probe movement, leakage, discriminant results, dose/sign response). It is
  evidence the promotion chain CITES; it is not a seating mechanism. Seating
  is still sweep → promote, like every other vector family.

## Pinning it

```bash
steerlab-server sae candidates check prompts/sae/candidates.json      # exit 2 on any schema violation
steerlab-server sae candidates pin <experiment> prompts/sae/candidates.json
```

`pin` validates that the file loads, then stamps
`saeCandidates: {path, hash}` (SHA-256 of the file BYTES) into the DRAFT
manifest. From that moment the roster is on the `verify()` pin surface: an
edit, a move, or a deletion is a verification violation exactly like stimulus
or markers drift, `freeze` refuses on it (force included — `verify()` is
never skippable), the file is git-gated at freeze, snapshotted into
`experiments/<name>/pinned/`, and packed into evidence bundles. Frozen
manifests are immutable: duplicate the experiment and re-pin.

The block is optional and additive — a study with no SAE arm never gains the
key, and every manifest frozen before this existed verifies unchanged.

## Filling in this template

The six entries here are the proposal's original roster. Their discovery
snapshots were **transcribed from the proposal document**, not captured from
the live feature pages: re-capture explanation text, top positive/negative
logits, example activations, and a fresh `accessDate` for each before any
qualification run, and set `verification` from your own inspection. Two
entries (F10346, F2214) were still unverified as of 2026-08-13.
