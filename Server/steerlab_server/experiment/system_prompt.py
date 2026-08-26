"""How an arm's system prompt is COMPOSED, and how that composition is stamped.

Swift twin: ``Sources/ExperimentKit/SystemPromptComposition.swift`` — same
joiner, same degradation rules, same stamp key spellings. Both engines must
compose byte-identically or two runs of the same study on two substrates are
two different studies.

Two levels of system prompt exist, and they used to REPLACE one another:

* the **agent's** ``systemPrompt`` — who the model is (a persona carried on a
  saved agent artifact), and
* the **study's** ``manifest.systemPrompt`` — the deployment frame every arm
  of the study is read under (response format, task framing).

Replacement made those two mutually exclusive: a variant/agent arm ran under
its persona with the study's frame simply not applied, while baseline and
steering arms ran under the frame with no persona. The arms were then not
comparable — the frame was part of the contrast rather than held constant.

**The rule (maintainer ruling, 2026-08-24).** The effective system prompt of
an arm is the AGENT's text first, then the STUDY's frame, joined by one blank
line::

    <agent persona>

    <study frame>

Persona first because identity precedes instruction: the frame's "answer in
this format" is an instruction TO whoever the model is being.

**Degradation is graceful, and byte-exact.** An empty agent yields the frame
itself — the SAME object, not a re-joined copy — so every historical
empty-persona run stamps and renders exactly the bytes it always did. An empty
frame yields the persona alone. Both empty yields nothing. Emptiness is
whitespace-insensitive (a frame of ``"   "`` is not a persona-suppressing
frame), but a non-empty value is never trimmed: what the researcher wrote is
what the model is armed with.

Batteries compose the same way with a different second term — the battery's
own declared arming text, never the study frame (see :mod:`.battery`).

**Panels compose the same way too** (maintainer ruling, 2026-08-24). A panel
seat carries a CAST ENTRY prompt — the role, "you represent Team South" — and
the agent artifact cast into that seat carries the persona. Casting used to
REPLACE, exactly as the study levels did (``multi_agent._runtime_settings``),
so a cast agent lost its role or its persona depending on which one was
non-empty. The order is the SAME as the study rule and holds for the same
reason: identity precedes instruction. A cast role is situational instruction
TO whoever the agent is, just as the study frame is::

    compose(<agent artifact persona>, <cast entry role>)

One uniform rule everywhere, so there is deliberately no second entry point —
the panel path calls :func:`compose` with the cast text as the second term and
stamps with ``frame_key="cast"``.
"""

from __future__ import annotations

import hashlib

#: The separator between the two levels. One blank line, which every chat
#: template renders as a paragraph break inside a single system turn.
JOINER = "\n\n"


def _empty(text: str | None) -> bool:
    """True when ``text`` carries no system content at all."""
    return not (text or "").strip()


def compose(agent: str | None, frame: str | None) -> str | None:
    """The effective system prompt: ``agent`` first, then ``frame``.

    Returns the surviving side UNCHANGED when the other is empty — identity,
    not reconstruction, which is what makes the empty-persona case byte-exact
    against every run recorded before this rule existed.
    """
    if _empty(agent):
        return frame
    if _empty(frame):
        return agent
    return agent + JOINER + frame


def text_hash(text: str | None) -> str | None:
    """SHA-256 of a system prompt, or ``None`` when there is nothing to hash.

    The same rule the per-condition ``systemPromptHash`` stamp has always
    used (``tasks._sha256_text``), so the composition stamp's hashes and the
    effective hash beside them are comparable without a second convention.
    """
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stamp_hash(text: str | None) -> str | None:
    """The hash a COMPOSITION STAMP records for one level.

    :func:`text_hash`'s own rule is truthiness, shared on purpose with the
    per-condition ``systemPromptHash`` (``tasks._sha256_text``) so the two are
    comparable without a second convention — and it stays exactly that. But a
    stamp answers a different question: *what did this level contribute to the
    effective prompt?* :func:`compose` drops a whitespace-only level entirely,
    so a persona of ``"   "`` contributes nothing at all while hashing to a
    perfectly good digest. Stamping it claimed a contribution the effective
    bytes do not contain (review 2026-08-26). The composition's own emptiness
    rule is therefore applied HERE, at the stamp, and nowhere else. Swift twin:
    ``SystemPromptComposition.stampHash``.
    """
    return None if _empty(text) else text_hash(text)


def composition(agent: str | None, frame: str | None, *,
                frame_key: str = "study") -> dict:
    """The additive provenance block beside an effective ``systemPromptHash``.

    ``{"agent": <hash|null>, "study": <hash|null>}`` on a study record;
    ``frame_key="battery"`` spells the second term for a battery record, whose
    second term is the battery file's declared arming rather than the study
    frame; ``frame_key="cast"`` spells it for a PANEL TURN record, whose second
    term is the seat's cast-entry role text. Keys are ALWAYS present (``null``
    when that level contributed nothing) — an absent key would read as "this
    engine does not stamp composition", a different claim from "this level was
    empty". The ``agent`` key is always first, whatever the second term is, so
    the three stamp shapes read alike.

    "Contributed nothing" is :func:`compose`'s OWN rule, whitespace included
    (see :func:`_stamp_hash`): a level the composition drops stamps ``null``,
    so the stamp can never claim a contribution the effective prompt does not
    carry.
    """
    return {"agent": _stamp_hash(agent), frame_key: _stamp_hash(frame)}


def divergence_advisory(arms) -> str | None:
    """The loud, non-blocking run-start advisory for arms that are armed with
    DIFFERENT effective system content.

    ``arms`` is an ordered sequence of ``(arm name, effective system prompt)``
    in the run's own emission order — EFFECTIVE, meaning each text has already
    been through :func:`compose`, which is why this hashes with
    :func:`text_hash` rather than :func:`_stamp_hash`: there is no dropped
    level left to misrepresent, and two arms whose effective bytes differ by
    whitespace really are armed differently. Returns ``None`` when every arm
    shares one effective content — the universal case, which must stay silent — and
    otherwise a single line naming every arm and the hash it runs under.

    Why an advisory and not a refusal: a deliberately persona-varying design
    IS a legitimate study (that is the point of promoting agents), so this
    cannot gate. What it must not do is let the difference pass unremarked,
    because a contrast between two arms armed differently mixes identity and
    framing into whatever the intervention did.
    """
    listed = [(str(name), text_hash(text)) for name, text in arms]
    if len({digest for _name, digest in listed}) < 2:
        return None
    parts = ", ".join(
        f"{name} ({digest[:12] + '…' if digest else 'none'})"
        for name, digest in listed)
    return ("arms in this run are armed with DIFFERENT effective system "
            f"prompts — {parts}. A contrast between two such arms mixes "
            "identity and framing with the intervention; hold the system "
            "prompt constant across arms, or report the difference as part "
            "of the design. Not a refusal.")
