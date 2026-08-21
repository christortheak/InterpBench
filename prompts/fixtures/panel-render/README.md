# panel-render — cross-engine panel prompt fixture

Both engines render panel turn prompts, and a prompt that differs between them
is a study that cannot be pooled. This fixture pins the rendered bytes: it is
the executable half of the panel turn-prompt contract, and the engines'
renderers are held to these bytes on both sides.

- `scenario.json` — the panel. Written in the Python engine's canonical dialect
  (sorted keys, two-space indent, `\uXXXX`-escaped non-ASCII, one trailing
  newline); the Python round-trip is asserted byte for byte. This file is
  INPUT to the Swift twin — parse it and compare `expected/`, do not assert
  your own serialiser reproduces these bytes.
- `expected/<label>.txt` — the exact prompt each turn renders, given the stub
  outputs below. **No trailing newline**: a rendered prompt ends at its last
  character, and adding one would make the fixture disagree with what the
  runner actually sends. Read these as raw bytes and compare with `==`.

## Stub outputs

Replay `scenario.json` turn by turn. Instead of generating, take each turn's
output from this table, then route it into the listening agents' context and
the output-label table exactly as `run_scenario` / `MultiAgentRunner` does.
The labels are the turns' `outputLabel` values, in turn order.

| output label | stub output |
|---|---|
| `t1_ava` | `AVA DRAFT.` |
| `t1_ben` | `BEN DRAFT.` |
| `t2_ava` | `AVA RESPONSE.` |
| `t3_ava` | `AVA RECAP.` |
| `t4_ben` | `Verdict: yes` |
| `t5_cal` | `CAL NOTE.` |

Each output is a single line with no trailing newline.

## What each turn is here to pin

| turn | pins |
|---|---|
| `t1_ava` | a template turn positioning its own `{{scenario.materials}}`; empty context adds nothing |
| `t1_ben` | the fallback **prepend** (spec §3.2) — neither placeholder present, so materials then context land BEFORE the template body |
| `t2_ava` | the spec §2 **worked example, verbatim**. If an engine disagrees here, the engine is wrong, not the spec |
| `t3_ava` | reader-aware context entries (spec §3.1) — Ava's own turns render as `— your own earlier output (Ava)`, Ben's as `— Ben` |
| `t4_ben` | a contract turn WITH the routed transcript block, one other-authored input, an empty `stage`, no `role`, and a declared endpoint |
| `t5_cal` | `ownVoice: false` (no block 6), an overridden `materialsTitle`, `{{scenario.name}}` substitution in `task`, no inputs, empty `format` (no block 7) |

The panel is deliberately clean **on both engines**: `validate()` passes and
`advisories()` is empty, so a difference in either is a regression rather than
a property of the fixture. That is why all three seats carry
`"baseModelID": "fixture/model"` — both engines refuse an agent with no base
model, since a blank seat otherwise inherits whatever model happens to be
loaded. The model id is never resolved here (nothing generates), so it affects
nothing in `expected/`.

Python replay: `Server/tests/test_panel_contracts.py`.
