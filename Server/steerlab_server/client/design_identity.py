"""Versioned authoring identity; never replaces an existing freeze/design hash.

portable-v1 uses a length-framed JSON tree with exact integers and IEEE doubles.
Defaults are the Mac manifest's decode defaults, not Python execution defaults.
The companion Swift owner keeps legacy (unstamped) lineage on its old algorithm.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import math
import struct

ALGORITHM = "portable-v1"
LIFECYCLE = ("frozenAt", "freezeHash", "frozenBy", "gitCommit", "appVersion",
             "freezeForced", "forcedGatesSkipped", "preregistrationHash", "preregistrationGeneratedHash")
REMOVED = LIFECYCLE + ("multiAgentScenarioPath", "multiAgentScenarioHash",
                      "multiAgentSemanticScenarioPath", "multiAgentSemanticScenarioHash", "templateProvenance")
DEFAULTS = {"studyKind": "modelOutput", "multiAgentIncludeBaseline": True,
            "concepts": [], "conditions": [], "variantConditions": [], "recordTokenIDs": False, "seeds": [20260610], "temperature": 0, "maxTokens": 2048}
# These are deliberately untyped JSON payloads in the Mac manifest. Null is data.
OPAQUE = {"pipeline", "jlensReadout", "saeCandidates", "maxSAEMixtureFeatures", "saeLatentConditions"}


def stripped(study: dict) -> dict:
    body = deepcopy(study)
    for key in REMOVED:
        body.pop(key, None)
    body.update(name="", createdAt="", status="draft", conditions=[], variantConditions=[])
    return body


def normalized(study: dict) -> dict:
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if v is not None or k == "validationHash"}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    body = {k: (deepcopy(v) if k in OPAQUE else clean(v)) for k, v in study.items() if v is not None and k not in LIFECYCLE}
    body.update(createdAt="", status="draft")
    for k, v in DEFAULTS.items():
        body.setdefault(k, deepcopy(v))
    return body


def framed(value) -> bytes:
    if value is None:
        return b"n"
    if isinstance(value, bool):
        return b"t" if value else b"f"
    if isinstance(value, str):
        data = value.encode("utf-8")
        return b"s" + str(len(data)).encode() + b":" + data
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Design settings must contain finite JSON numbers.")
        if int(value) == value:
            return b"i" + str(int(value)).encode() + b";"
        return b"d" + struct.pack(">d", value).hex().encode() + b";"
    if isinstance(value, list):
        return b"a" + str(len(value)).encode() + b":" + b"".join(framed(v) for v in value)
    if isinstance(value, dict):
        keys = sorted(value, key=lambda k: k.encode("utf-8"))
        return b"o" + str(len(keys)).encode() + b":" + b"".join(framed(k) + framed(value[k]) for k in keys)
    raise ValueError("Design settings must be JSON values.")


def content_hash(template: dict) -> str:
    material = {"schemaVersion": template.get("schemaVersion", 1), "study": normalized(template["study"]),
                "semanticScenario": template.get("semanticScenario")}
    return hashlib.sha256(ALGORITHM.encode() + b"\x00" + framed(material)).hexdigest()
