"""Resolve a Neuronpedia feature using installed SAELens metadata, never URL fetches."""
from __future__ import annotations

from urllib.parse import unquote, urlsplit


def resolve_feature_url(url: str, *, directory=None) -> dict:
    parsed = urlsplit(url.strip())
    if parsed.scheme != "https" or parsed.hostname not in {"neuronpedia.org", "www.neuronpedia.org"} or parsed.username or parsed.password:
        raise ValueError("Use an HTTPS Neuronpedia feature link, or enter release, dictionary and feature identifiers directly.")
    parts = [unquote(p) for p in parsed.path.strip("/").split("/")]
    if len(parts) != 3 or not parts[2].isascii() or not parts[2].isdigit():
        raise ValueError("Expected a feature link ending in /model/dictionary/feature-number. Search and collection links need an individual feature selection.")
    key = "/".join(parts[:2])
    if directory is None:
        try:
            from sae_lens.loading.pretrained_saes_directory import get_pretrained_saes_directory
        except ImportError:
            try:
                from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
            except ImportError as exc:
                raise ValueError("Feature lookup needs the engine's SAELens package. You can enter identifiers directly; importing weights still needs its supported loader.") from exc
        directory = get_pretrained_saes_directory()
    matches = []
    for release, record in directory.items():
        for sae_id, neuronpedia_id in (record.neuronpedia_id or {}).items():
            if neuronpedia_id == key:
                matches.append({"release": release, "saeID": sae_id,
                                "model": record.model, "feature": int(parts[2]),
                                "neuronpediaURL": url.strip()})
    if len(matches) != 1:
        raise ValueError("This link does not have one unique mapping in the installed SAE directory. Enter the exact release and dictionary identifiers; no mapping was guessed.")
    return matches[0]
