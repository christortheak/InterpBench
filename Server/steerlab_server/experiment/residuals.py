"""Alien-stance residuals: R = delta_model − delta_human.

The headline quantity for any study that anchors its intervention against a
measured HUMAN effect: how far the model's response to a concept intervention
sits from the human one. ``delta_model`` comes from the study's paired effect
estimates (study_stats); ``delta_human`` comes from a pinned, hash-verified
human-baseline CSV that the researcher transcribes from whatever published
literature measured the same manipulation on people (in the judicial-decision
study: published rates for the categorical items, and the 123-judge
anchoring literature for the sentencing items). One row per
(condition, endpoint); the region classification follows this decision flow:

- ``humanAligned``   — model and human effects both real, R's CI covers 0
- ``hyperHuman``     — same direction as humans, credibly larger
- ``hypoHuman``      — same direction as humans, credibly smaller (muted)
- ``alien``          — humans don't move, the model credibly does
- ``inertBoth``      — neither moves
- ``inverted``       — model credibly moves against the human direction

The R interval is conservative: [Δm_lo − Δh_hi, Δm_hi − Δh_lo] (exact when the
two estimates are independent, which they are — different populations).
"""

from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass

ALIEN_RESIDUALS_HEADER = [
    "condition", "endpoint", "deltaModel", "ciModelLower", "ciModelUpper",
    "deltaHuman", "ciHumanLower", "ciHumanUpper",
    "R", "ciRLower", "ciRUpper", "region",
]

HUMAN_BASELINE_FIELDS = ("endpoint", "deltaHuman", "ciLower", "ciUpper")


@dataclass
class HumanEffect:
    endpoint: str
    delta: float
    ci_lower: float
    ci_upper: float


def load_human_baseline(path: str, expected_hash: str | None = None) -> dict[str, HumanEffect]:
    """Load a pinned human-baseline table, verifying its SHA-256 first.

    CSV columns: endpoint, deltaHuman, ciLower, ciUpper (one row per endpoint).
    The hash check is not optional in study paths — an unpinned baseline would
    let the comparison target drift after freeze.
    """
    with open(path, "rb") as handle:
        blob = handle.read()
    if expected_hash is not None:
        live = hashlib.sha256(blob).hexdigest()
        if live != expected_hash:
            raise ValueError(
                f"human baseline {os.path.basename(path)} drifted: "
                f"have {live[:12]}…, pinned {expected_hash[:12]}…")
    effects: dict[str, HumanEffect] = {}
    reader = csv.DictReader(blob.decode("utf-8").splitlines())
    found = list(reader.fieldnames or [])
    missing = [f for f in HUMAN_BASELINE_FIELDS if f not in found]
    if missing:
        raise ValueError(
            f"human baseline {os.path.basename(path)} missing columns: "
            f"{', '.join(missing)}. Required columns: "
            f"{', '.join(HUMAN_BASELINE_FIELDS)}. Found columns: "
            f"{', '.join(found) if found else '(no header row)'}. "
            "Fix: rename the CSV's header to the required column names "
            "(one row per endpoint) and re-pin its hash in the manifest.")
    for row in reader:
        endpoint = row["endpoint"].strip()
        effects[endpoint] = HumanEffect(
            endpoint=endpoint, delta=float(row["deltaHuman"]),
            ci_lower=float(row["ciLower"]), ci_upper=float(row["ciUpper"]))
    return effects


@dataclass
class ResidualRow:
    condition: str
    endpoint: str
    delta_model: float
    ci_model: tuple[float, float]
    delta_human: float
    ci_human: tuple[float, float]

    @property
    def residual(self) -> float:
        return self.delta_model - self.delta_human

    @property
    def ci_residual(self) -> tuple[float, float]:
        return (self.ci_model[0] - self.ci_human[1],
                self.ci_model[1] - self.ci_human[0])

    @property
    def region(self) -> str:
        model_moves = not _covers_zero(self.ci_model)
        human_moves = not _covers_zero(self.ci_human)
        residual_zero = _covers_zero(self.ci_residual)
        if not model_moves and not human_moves:
            return "inertBoth"
        if model_moves and not human_moves:
            return "alien"
        same_sign = self.delta_model * self.delta_human > 0
        if model_moves and human_moves and not same_sign:
            return "inverted"
        if residual_zero:
            return "humanAligned"
        if abs(self.delta_model) > abs(self.delta_human):
            return "hyperHuman"
        return "hypoHuman"

    def as_csv_row(self) -> list[str]:
        return [self.condition, self.endpoint,
                f"{self.delta_model:.6g}", f"{self.ci_model[0]:.6g}",
                f"{self.ci_model[1]:.6g}",
                f"{self.delta_human:.6g}", f"{self.ci_human[0]:.6g}",
                f"{self.ci_human[1]:.6g}",
                f"{self.residual:.6g}", f"{self.ci_residual[0]:.6g}",
                f"{self.ci_residual[1]:.6g}", self.region]


def _covers_zero(ci: tuple[float, float]) -> bool:
    return ci[0] <= 0.0 <= ci[1]


def residual_rows(model_effects: list, human_effects: dict[str, HumanEffect]) -> list[ResidualRow]:
    """Join the study's effect rows (study_stats.EffectRow) against the human
    table by endpoint. Endpoints with no human row are skipped — they simply
    have no residual, which the report should say rather than fake."""
    rows: list[ResidualRow] = []
    for effect in model_effects:
        human = human_effects.get(effect.endpoint)
        if human is None:
            continue
        rows.append(ResidualRow(
            condition=effect.condition, endpoint=effect.endpoint,
            delta_model=effect.ci.mean,
            ci_model=(effect.ci.ci_lower, effect.ci.ci_upper),
            delta_human=human.delta,
            ci_human=(human.ci_lower, human.ci_upper)))
    return rows


def write_alien_residuals_csv(path: str, rows: list[ResidualRow]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(ALIEN_RESIDUALS_HEADER)
        for row in rows:
            writer.writerow(row.as_csv_row())
