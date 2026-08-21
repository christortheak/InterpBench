# OpenRouter provider identity

`providers.json` is the authoritative display-name → routing-slug table for
OpenRouter serving providers, fetched from the public (unauthenticated)
`https://openrouter.ai/api/v1/providers` endpoint. Refresh it with:

```bash
python3 prompts/fixtures/openrouter/refresh.py
```

Both engines canonicalize against this one file — `OpenRouterProviderIdentity`
(Swift) and `paired_judge.canonical_openrouter_provider` (Python) — so there is
no second table to drift.

## Why this is measurement-path data, not config

An `openrouter` judge pins a model slug **and** a serving provider, with
`allow_fallbacks: false`. The pin is scientific, not cosmetic: the same model
slug served by two backends can run different quantizations and produce
different verdicts, so an unpinned or off-pin judgment is not the declared
judge. Every judgment's recorded provider is verified against the pin, and a
mismatch refuses.

That verification is a **string comparison**, which makes this table part of
the measurement path. If it says the wrong thing, correct judgments get
refused and correct refusals get accepted.

## Why it is fetched rather than written by hand

The hand-maintained alias list this replaced carried four guessed entries and
was wrong about one of them:

- OpenRouter's display name for Vertex is **`Google`**, not `Google Vertex`.
  A judgment correctly served by a pinned `google-vertex` therefore reported
  `"Google"`, canonicalized to `"google"`, and was refused as off-pin — a
  correct run failing on a spelling.
- `Together AI` was an alias for a display name that does not exist (it is
  `Together`).
- 86 of the ~96 providers were absent entirely.

Ten providers need a mapping that no slugify rule produces — `Moonshot AI` →
`moonshotai`, `Z.AI` → `z-ai`, `Mancer 2` → `mancer`, `InferenceNet` →
`inference-net`, `AionLabs` → `aion-labs`, `AtlasCloud` → `atlas-cloud`,
`Google` → `google-vertex`, `OpenInference` → `open-inference`, `Sakana AI` →
`sakana`, `FakeProvider` → `fake-provider` — so guessing was never going to
converge on the real table.

## Contract

- Canonicalization is: trim, lowercase; a known **slug** maps to itself; a
  known **display name** maps to its slug; anything else maps to the lowercased
  input and therefore still **fails closed** against a pin it does not match.
- An unknown provider is not silently accepted. If a judgment refuses with an
  off-pin error naming a provider this fixture does not know, re-run
  `refresh.py` and commit the result — do not hand-edit the JSON.
- `refresh.py` refuses to write a table in which one provider's display name
  lowercases onto a *different* provider's slug, since that would make
  canonicalization ambiguous and could silently reroute a pinned judge.

## Related

- `prompts/fixtures/paired-judge/` — the golden judge-prompt wrapper.
