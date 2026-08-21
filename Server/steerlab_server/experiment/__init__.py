"""Experiment definitions, run configs, generation, and metrics (parallel to
the MLX-bound parts of Swift ``ExperimentKit``).

Unlike :mod:`steerlab_server.steering`, this layer may know about prompts,
conditions, scoring, and the shape of a particular study — but it drives the
concept-agnostic core, never the reverse.
"""
