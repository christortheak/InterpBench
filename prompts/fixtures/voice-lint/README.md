# Voice-lint golden cases

`cases.jsonl` is the cross-engine fixture for the multi-agent voice lint
(`Server/steerlab_server/experiment/voice_lint.py` and its Swift twin
`Sources/ExperimentKit/MultiAgent/VoiceLint.swift`, §5 of the panel
turn-prompt contract). Both engines' tests replay every line and
assert the exact stamp, so a divergence fails on both sides.

One JSON object per line:

| key | meaning |
|---|---|
| `name` | case id, unique |
| `speakerName` | the agent whose turn this is |
| `otherNames` | the OTHER agents of the scenario, in scenario order |
| `output` | verbatim model text (an excerpt — see below) |
| `expected` | the exact `voiceLint` stamp, including key presence |
| `source` | provenance: which run, record and line range this came from |

`expected` is the stamp byte for byte: `otherSpeakerLines` is ABSENT when no
line signs as a colleague (additive-key discipline, same as the `endpoint`
stamp), so a test that only compares present keys is not testing the contract.

## Where the text came from

Every excerpt is a verbatim, contiguous line range of one real generation, from
the workspace runs named in the spec. Nothing is synthesised, edited, or
re-cased: the point of the fixture is that the rules were calibrated against
what the models actually wrote, not against what a rule author imagined.

Records are identified by `(promptIndex, sampleIndex, condition)` in that run's
`generations.jsonl`. `sampleIndex` is the replicate.

| case | run | record | what it pins |
|---|---|---|---|
| `colleague-signature-blocks` | `…-prec-delib-c0-run` | 10 / 0 / configured (Round 4 — opinion — Judge Marsden), last 12 lines | The documented Marsden-writes-Whitfield-and-Calloway example. Two colleague signature blocks (`**WHITFIELD, Judge,** concurring.`, `**CALLOWAY, Judge,** dissenting.`) and one third-person self ("I join Judge Marsden's opinion in full" — inside the concurrence Marsden wrote for Whitfield). |
| `own-byline-and-panel-caption` | `…-prec-delib-c0-run` | 10 / 0 / configured, first 16 lines | The head of that same opinion: a panel caption (`Before: MARSDEN, Judge; WHITFIELD, Judge; and CALLOWAY, Judge.`) and the speaker's OWN byline. Neither is a violation — the caption line does not start with a colleague's name, and a seat signing its own name is not speaking for anyone. |
| `line-initial-address-not-signature` | `…-prec-delib-c0-run` | 3 / 0 / configured (Round 2 — response to colleagues), first 12 lines | The discrimination the whole rule set turns on: `Judge Marsden, I wholeheartedly agree…` and `Judge Calloway, while I understand…` open lines with a colleague's FULL name and a comma, and are legitimate address. Round-2 memoranda are documented as the high-quality turn and must lint clean. |
| `blind-round-memo-clean` | `…-prec-delib-c0-run` | 0 / 0 / configured (Round 1 — initial memo), first 8 lines | A blind-round memo with a `From: Judge Whitfield` header. The own-name header is a field label, not third-person prose (60 such occurrences in the corpus). |
| `third-person-self-in-disposition` | `…-prec-delib-c0-run` | 8 / 0 / configured (Round 3 — disposition — Judge Calloway), whole output | Failure (b) from the work order, quoted there verbatim: Calloway crediting "Judge Calloway's compelling equitable arguments". |
| `unbolded-lowercase-signature-list` | `…-prec-delib-a01-run` | 9 / 3 / baseline (Round 4 — opinion — Judge Whitfield), last 9 lines | Three bare signature lines, unbolded and mixed-case. Two are colleagues; the third is the speaker's own and is not counted. This is the shape a bold-and-uppercase pattern misses. |
| `joins-its-own-opinion` | `…-prec-delib-c0-run` | 11 / 0 / configured (Round 4 — opinion — Judge Calloway), last 16 lines | Both failures at once: Calloway files signature blocks for Whitfield and Marsden, and has each of them say "I join the opinion of Judge Calloway" — third-person self, twice. |
| `screenplay-speaker-labels` | `…-consciousness-a-run` | 22 / 2 / configured (Dissent draft if needed — Judge C), first 6 lines | The documented screenplay turn: Judge C writes the whole conference as `**(Judge A, as presiding judge):**` / `**(Judge B):**` stage directions, and addresses itself in the third person. |
| `third-person-self-conference-summary` | `…-consciousness-a-run` | 9 / 2 / configured (Conference summary), first 8 lines | The documented "given Judge A's scale position" self-reference, written by Judge A. `Proposed majority author: Judge B or Judge C` is a field label naming colleagues mid-line — not a voice violation. |
| `roll-call-list-is-not-a-voice` | `…-consciousness-a-run` | 23 / 0 / baseline (Final disposition package), first 20 lines | `*   **Judge B:** Affirm` reports a colleague's vote in a Markdown list. Without the list-item exemption this legitimate turn type flagged 10 of 10 times. |
| `first-person-self-identification` | `…-consciousness-a-run` | 22 / 2 / baseline (Dissent draft if needed — Judge C), first 5 lines | The compliant twin of `screenplay-speaker-labels`: same seat, same turn, baseline arm. "Okay, I am Judge C." is first-person framing, and "Judge A's draft" is discussion of a colleague, not a signature. Lints clean. |

The runs live in the study workspace, not in this repo (workspace rule): the
`c0`, `a01` and `a012`/`a008` runs of `2026-08-11` and the
`hard-consciousness-a` run of `2026-08-08`.

## Calibration summary (why these rules, and what they cost)

Measured over the four `multiagent-kz-s3-prec-delib-*` runs (600 turn records)
and `hard-consciousness-a` (240 records):

- **speaksForOthers** fires on 101 of the 120 round-4 opinions (32/40, 34/40,
  35/40 for the three seats; 204 colleague-signed lines in all) and on 0 of the
  480 records of rounds 1, 2, 3 and 5. The work order's hand count was 81/120;
  every additional detection was inspected and is a real colleague signature
  block that a bold-plus-uppercase pattern misses (`Marsden, J., concurring.`,
  `WHITFIELD, Circuit Judge, concurring in the judgment.`).
- In the consciousness run it fires on 4 of 240 records — the screenplay turns
  and one stage direction — and on none of the ten disposition packages.
- **thirdPersonSelf** totals 46 occurrences over 38 records in the prec-delib
  corpus, all in rounds 3 and 4, and 35 over 17 records in the consciousness
  corpus. Every one was read; none is a header, a self-signature, or a
  first-person construction.
