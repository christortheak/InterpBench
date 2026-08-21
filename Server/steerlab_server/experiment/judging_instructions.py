"""The agent-facing judging instructions artifact (Cowork judging pipeline,
2026-08-11).

When ``evaluate`` takes the keyless custody fork it emits blinded packets
(``judging-packets.jsonl``) plus the identity map the judging client must
never read (``judging-map.json``). Until now the framing an agent
orchestrator needed — rubric, output schema, custody rules — was hand-written
per campaign (the ``cowork/<campaign>-reading/JOB.md`` pattern), which meant
two campaigns could judge identical packets under different framing and
nobody could prove it afterwards. This module renders that framing as a
canonical ENGINE artifact next to the packets: self-contained,
orchestrator-agnostic (Claude, Codex, anything that can spawn independent
agent contexts), hashed into the emission record, and verified back at
``complete-judgment`` intake via the ``instructionsSha256`` the judgments
claim.

Custody invariant: nothing rendered here may unblind. The instructions see
the rubric, the pinned judge panel, and counts — never the map's contents
(condition names, orientation, seeds, promptIDs).
"""

from __future__ import annotations

#: Canonical filename, next to the packets in the awaiting run directory.
INSTRUCTIONS_FILENAME = "judging-instructions.md"


def render(*, experiment: str, evaluate_run: str, packets_file: str,
           packet_count: int, rubric: str,
           structured_prompt: str | None,
           judges: list[dict]) -> str:
    """The canonical judging-instructions text for one awaiting evaluate
    run. Deterministic in its inputs (no timestamps) so the emission hash
    is reproducible from the manifest + run identity alone."""
    external = [j for j in judges
                if (j.get("kind") or "claude") in ("claude", "openrouter")]
    panel_rows = "\n".join(
        f"| `{j.get('name')}` | {j.get('kind') or 'claude'} "
        f"| `{j.get('model')}` "
        f"| {('`' + str(j['provider']) + '`') if j.get('provider') else '—'} |"
        for j in (external or judges))
    structured_section = ""
    if structured_prompt:
        structured_section = f"""
## Structured comparison prompt (pinned)

The evaluation additionally pins this structured comparison prompt — it is
part of the criterion, applied together with the rubric:

```text
{structured_prompt}
```
"""
    return f"""# Judging instructions — experiment `{experiment}`, awaiting run `{evaluate_run}`

This file was rendered by the SteerLab engine next to the blinded judging
packets it governs, and its SHA-256 is stamped in this run's
`judging-manifest.json` (`instructionsSha256`). It is self-contained and
orchestrator-agnostic: paste it into (or point at it) any agent
orchestrator — Claude, Codex, or another family — and the judging campaign
it defines is identical. All paths below are relative to the directory this
file lives in.

- **Packets:** `{packets_file}` — {packet_count} blinded packet(s), one JSON
  object per line: `{{"packetID", "prompt", "responseA", "responseB"}}`.
- **Verdicts go back to the engine** via
  `steerlab-server experiment complete-judgment` (see the last section).

## Custody requirements (binding — violating any of them voids the evidence)

1. **Never open `judging-map.json`.** It lives in this same directory and
   maps each packet back to arm identity and response orientation. The only
   legitimate reader is the engine's `complete-judgment` intake, which
   unblinds AFTER every verdict is fixed. An orchestrating context that
   reads it has unblinded the study, and every verdict produced under that
   orchestrator is void.
2. **One independent agent context per packet.** Spawn a fresh judging
   agent — no shared conversation, no accumulated transcript — for each
   packet, and give it only this file's rubric/schema plus that one
   packet's contents. Judging packets serially in one context lets earlier
   verdicts, drifting style expectations, and context fatigue condition the
   later ones: the errors become correlated across packets, which silently
   voids the independence assumption behind the inter-judge agreement
   statistics (Cohen's kappa). The numbers still compute; they just stop
   meaning what they claim.
3. **Judge only from the packet.** The pinned rubric below plus the
   packet's own `prompt`, `responseA`, and `responseB` are the entire
   evidence base. No other packets, no outside knowledge of the study, no
   repository spelunking.
4. **Do not infer arm identity.** Response order is randomized per packet
   at emission. Do not reason about which side "must be" which arm, and do
   not carry beliefs about sides from one packet to another.

## Rubric (pinned — judge with exactly this text)

```text
{rubric}
```
{structured_section}
## Judge panel (pinned at emission)

Every verdict row speaks for exactly one pinned judge and must echo that
judge's `name` and `model` verbatim from this table — intake refuses
off-pin rows:

| name | kind | model | provider |
|---|---|---|---|
{panel_rows}

## Output: one JSONL row per (packet × judge)

```json
{{"packetID": "<from the packet>", "judge": "<pinned judge name>",
 "winner": "A" | "B" | "tie",
 "model": "<that judge's pinned model, verbatim from the table>",
 "annotatorModel": "<the model the judging agent itself ran on>",
 "confidence": 0.9,
 "judgment": {{"winner": "<same as the row's winner>", "confidence": 0.9,
              "brief_reason": "<one sentence>"}}}}
```

- `packetID`, `judge`, `winner`, and `model` are **required**. `judge` and
  `model` must match the pinned panel; intake refuses unknown judges and
  off-pin models. Full coverage is required: every (packet × judge) cell
  exactly once.
- `annotatorModel` records the model the judging agent actually ran on
  (e.g. its own model id/version string) — distinct from the pinned
  `model`, which names the judge the row speaks for. Intake records it per
  judgment so cross-model annotation agreement (e.g. Claude-orchestrated
  vs Codex-orchestrated campaigns over identical packets) stays computable.
- `provider` is required on rows for `openrouter`-kind judges (echo the
  pinned provider) and forbidden on all other rows.
- `confidence` (0–1) and the full `judgment` payload are optional; when
  the payload is present its own `winner` must agree with the row's
  `winner`, or intake refuses the row as inconsistent.
- Write rows incrementally to disk as agents return (a single inline reply
  dies at output-token caps).

## Handing the verdicts back

Concatenate every agent's rows and wrap them, claiming this file's hash so
intake can verify the campaign judged under these exact instructions:

```json
{{"instructionsSha256": "<sha256 of this file, as received>",
 "judgments": [ …rows… ]}}
```

Compute the hash of the file you actually read (`shasum -a 256
{INSTRUCTIONS_FILENAME}`), then:

```bash
steerlab-server experiment complete-judgment {experiment} \\
    --awaiting-run {evaluate_run} --judgments <your-file.json>
```

Intake verifies the packet/map/rubric/epoch pins and full coverage as
always, and verifies the claimed `instructionsSha256` against the emission
stamp. A mismatched or missing claim completes LOUDLY — it is stamped into
the judgment run's report, not refused (post-submit drift policy) — so do
not skip the claim: an unverified campaign is weaker evidence.
"""
