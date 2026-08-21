"""Claude-assisted stimulus proposals (parallel to Swift
``ClaudeStimulusGenerator``).

Generates contrastive pairs for a concept by calling the Claude API. Requires
``ANTHROPIC_API_KEY`` in the environment; without it the feature degrades
gracefully (the UI still has the copy-the-prompt helpers). The model returns
JSONL ``{"positive","negative"}`` rows which we parse into reviewable proposals.
"""

from __future__ import annotations

import json
import os
import re

from . import authoring

DEFAULT_MODEL = os.environ.get("STEERLAB_PROPOSAL_MODEL", "claude-opus-4-8")


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def generate_pairs(concept: str, count: int = 10, guidance: str = "",
                   examples_positive: list[str] | None = None,
                   examples_negative: list[str] | None = None) -> list[dict]:
    """Call Claude to draft ``count`` contrastive pairs. Raises if no API key."""
    if not available():
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set — use the 'Copy LLM prompt' helper instead, "
            "or set the key to enable in-app generation.")
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError("in-app proposals need the 'anthropic' package: pip install anthropic") from exc

    prompt = authoring.template_prompt("caa", concept=concept, count=count, guidance=guidance)
    if examples_positive or examples_negative:
        prompt += "\n\nExisting examples to match in style:\n"
        for p, n in zip(examples_positive or [], examples_negative or []):
            prompt += f'{{"positive": {json.dumps(p)}, "negative": {json.dumps(n)}}}\n'

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=DEFAULT_MODEL, max_tokens=4096,
        messages=[{"role": "user", "content": prompt}])
    text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
    return parse_pairs(text)


def parse_pairs(text: str) -> list[dict]:
    """Extract ``{"positive","negative"}`` pairs from a model response (tolerant
    of code fences and surrounding prose)."""
    pairs: list[dict] = []
    for match in re.finditer(r"\{[^{}]*\"positive\"[^{}]*\}", text):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if "positive" in obj and "negative" in obj:
            pairs.append({"positive": str(obj["positive"]), "negative": str(obj["negative"])})
    return pairs
