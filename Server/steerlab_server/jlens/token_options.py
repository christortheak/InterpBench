"""Resolve a typed word to EXACT token identities the researcher chooses from.

A J-lens direction is indexed by one vocabulary token, so the whole instrument
hangs on which token id was meant. The failure this service exists to prevent is
silent: a researcher types "courage", the tokenizer splits it, something takes
piece [0], and every downstream artifact is a direction for a word-fragment
while every label still says "courage".

So the rule is that the interface never chooses. It shows what each query form
produces — including whether that form is a single token — and the researcher
selects an id. The selected **token id**, not the displayed word, is the durable
identity that derivation and readout use (plan §5).

Case variants are shown only when the caller asks for them, and are labeled as
suggestions: " Courage" and " courage" are different directions, and quietly
folding them together would be the same class of error one level up.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from .schemas import JLensError

#: Query forms, in the order they are offered. ``exact`` is what was typed;
#: ``leadingSpace`` matters because most mid-sentence words tokenize with one
#: and the two are DIFFERENT tokens; ``decodedVocabularyMatch`` is a bounded
#: scan for vocabulary entries whose decoded text equals the query.
FORM_EXACT = "exact"
FORM_LEADING_SPACE = "leadingSpace"
FORM_VOCAB_MATCH = "decodedVocabularyMatch"

DEFAULT_MAX_MATCHES = 20


@dataclass
class TokenCandidate:
    tokenID: int
    piece: str
    decoded: str | None
    decodedBytes: str
    form: str
    singleToken: bool
    sequence: list[int]
    note: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _decode_bytes(tokenizer, token_id: int) -> tuple[str | None, str]:
    """``(utf8-or-None, unambiguous-hex)`` for one token id.

    A vocabulary entry need not be valid UTF-8 on its own — byte-level pieces
    and partial multi-byte characters are ordinary. The hex form is always
    present so a candidate is identifiable even when its text is not printable
    or renders identically to a different token.
    """
    try:
        raw = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens([token_id]))
    except Exception:  # noqa: BLE001 — tokenizers vary; fall back to decode
        raw = None
    if raw is None:
        try:
            raw = tokenizer.decode([token_id])
        except Exception:  # noqa: BLE001
            raw = None
    data = raw.encode("utf-8") if isinstance(raw, str) else b""
    return (raw if isinstance(raw, str) else None), data.hex()


def _candidate(tokenizer, ids: list[int], form: str, *, index: int = 0,
               note: str | None = None) -> TokenCandidate:
    token_id = int(ids[index])
    pieces = tokenizer.convert_ids_to_tokens([token_id])
    decoded, hexed = _decode_bytes(tokenizer, token_id)
    return TokenCandidate(
        tokenID=token_id,
        piece=pieces[0] if pieces else "",
        decoded=decoded,
        decodedBytes=hexed,
        form=form,
        singleToken=len(ids) == 1,
        sequence=[int(i) for i in ids],
        note=note)


def options_for(tokenizer, text: str, *, include_case_variants: bool = False,
                max_matches: int = DEFAULT_MAX_MATCHES) -> dict:
    """Bounded, deterministic candidates for a typed string.

    Multi-token forms are REPRESENTED, not resolved: every component is listed
    with its own id and the full sequence attached, so choosing one is an
    explicit act. Nothing here returns a "best" candidate.
    """
    if not isinstance(text, str) or not text:
        raise JLensError("token-options requires a non-empty string")
    if max_matches < 0:
        raise JLensError("max_matches must be non-negative")

    candidates: list[TokenCandidate] = []
    seen: set[tuple[int, str]] = set()

    def add(cand: TokenCandidate) -> None:
        key = (cand.tokenID, cand.form)
        if key not in seen:
            seen.add(key)
            candidates.append(cand)

    for form, query in ((FORM_EXACT, text), (FORM_LEADING_SPACE, " " + text)):
        if form is FORM_LEADING_SPACE and text.startswith(" "):
            continue          # already the leading-space form
        try:
            ids = tokenizer.encode(query, add_special_tokens=False)
        except Exception as exc:  # noqa: BLE001
            raise JLensError(f"tokenizer could not encode {query!r}: {exc}") from exc
        if not ids:
            continue
        if len(ids) == 1:
            add(_candidate(tokenizer, ids, form))
        else:
            # Every component, each flagged as part of a decomposition. The
            # researcher picks one deliberately or picks none.
            for i in range(len(ids)):
                add(_candidate(
                    tokenizer, ids, form, index=i,
                    note=(f"component {i + 1} of {len(ids)} — {query!r} is not a "
                          f"single token; choosing this derives a direction for "
                          f"the COMPONENT, not the word")))

    if include_case_variants:
        for variant in _case_variants(text):
            for query in (variant, " " + variant):
                try:
                    ids = tokenizer.encode(query, add_special_tokens=False)
                except Exception:  # noqa: BLE001
                    continue
                if len(ids) == 1:
                    add(_candidate(
                        tokenizer, ids, FORM_EXACT if query == variant
                        else FORM_LEADING_SPACE,
                        note=f"case variant of {text!r} — a DIFFERENT direction"))

    matches = _vocabulary_matches(tokenizer, text, max_matches, seen)
    for token_id in matches:
        add(_candidate(tokenizer, [token_id], FORM_VOCAB_MATCH))

    return {
        "query": text,
        "candidates": [c.to_dict() for c in candidates],
        # Said explicitly so a client cannot present a default: there is no
        # recommended candidate, and the selected id is the durable identity.
        "selection": "explicit",
        "truncated": len(matches) >= max_matches if max_matches else False,
    }


def _case_variants(text: str) -> list[str]:
    out = []
    for v in (text.lower(), text.upper(), text.capitalize()):
        if v != text and v not in out:
            out.append(v)
    return out


def _vocabulary_matches(tokenizer, text: str, limit: int,
                        seen: set[tuple[int, str]]) -> list[int]:
    """Vocabulary entries whose decoded text equals the query, bounded.

    Exact equality only — a substring scan over a 262k vocabulary would return
    noise and take long enough to feel broken. Sorted by id so repeated calls
    agree.
    """
    if limit <= 0:
        return []
    try:
        vocab = tokenizer.get_vocab()
    except Exception:  # noqa: BLE001
        return []
    already = {tid for tid, _ in seen}
    out: list[int] = []
    for want in (text, " " + text):
        for piece, token_id in vocab.items():
            if len(out) >= limit:
                break
            if token_id in already or token_id in out:
                continue
            try:
                decoded = tokenizer.convert_tokens_to_string([piece])
            except Exception:  # noqa: BLE001
                continue
            if decoded == want:
                out.append(int(token_id))
    return sorted(out)
