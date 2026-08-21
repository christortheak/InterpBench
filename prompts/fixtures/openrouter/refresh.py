#!/usr/bin/env python3
"""Refresh the committed OpenRouter provider-identity fixture.

    python3 prompts/fixtures/openrouter/refresh.py

Fetches https://openrouter.ai/api/v1/providers (public, no auth) and writes
``providers.json`` — the authoritative display-name -> routing-slug table both
engines canonicalize against.

Why a fetched fixture rather than a hand-written alias list: the list this
replaced held four guessed aliases and was wrong about one of them.
OpenRouter's display name for Vertex is "Google", not "Google Vertex", so a
judgment correctly served by a pinned ``google-vertex`` reported ``"Google"``,
canonicalized to ``"google"``, and was REFUSED as off-pin. Ten of the ~96
providers need a mapping no slugify rule produces (``Moonshot AI`` ->
``moonshotai``, ``Z.AI`` -> ``z-ai``, ``Sakana AI`` -> ``sakana``, ...), so
guessing was never going to converge.

Re-run when a judgment refuses with an off-pin error naming a provider the
fixture does not know. Commit the result: it is measurement-path data, and a
silent change to it would change which judgments are accepted.
"""

from __future__ import annotations

import datetime
import json
import os
import urllib.request

SOURCE = "https://openrouter.ai/api/v1/providers"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "providers.json")


def main() -> None:
    with urllib.request.urlopen(SOURCE, timeout=30) as response:
        payload = json.load(response)
    rows = payload.get("data")
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"unexpected payload from {SOURCE}: no data array")

    providers = []
    for row in rows:
        name = str(row.get("name") or "").strip()
        slug = str(row.get("slug") or "").strip()
        if not name or not slug:
            raise SystemExit(f"provider row missing name/slug: {row!r}")
        providers.append({"name": name, "slug": slug})
    providers.sort(key=lambda p: p["slug"])

    # A display name that lowercases onto a DIFFERENT provider's slug would
    # make canonicalization ambiguous — and silently reroute a pinned judge.
    # Refuse to write rather than ship an ambiguous table.
    by_slug = {p["slug"] for p in providers}
    for entry in providers:
        lowered = entry["name"].lower()
        if lowered in by_slug and lowered != entry["slug"]:
            raise SystemExit(
                f"ambiguous provider identity: display name {entry['name']!r} "
                f"lowercases to {lowered!r}, which is another provider's slug")

    document = {
        "schemaVersion": 1,
        "source": SOURCE,
        "fetchedOn": datetime.date.today().isoformat(),
        "providerCount": len(providers),
        "providers": providers,
    }
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {OUT} ({len(providers)} providers)")


if __name__ == "__main__":
    main()
