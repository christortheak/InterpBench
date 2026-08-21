"""Concept-agnostic steering core (parallel to Swift ``SteeringKit``).

No UI, no experiment logic, no concept-specific assumptions. Forward hooks on
HF decoder blocks replace the vendored MLX model implementations entirely.
"""
