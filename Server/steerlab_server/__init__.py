"""SteerLab server — a PyTorch + HuggingFace parallel to the MLX steering engine.

The Swift app's compute engine (``SteeringKit`` + the MLX-bound parts of
``ExperimentKit``) is Apple-Silicon/Metal only. This package re-implements that
surface on PyTorch + HF Transformers so the same activation-steering science can
run on a non-macOS GPU cluster, while reading and writing the **same on-disk
artifact formats** (``.safetensors`` vectors + JSON sidecars, experiment
manifests, immutable ``runs/`` directories).

Design rule carried over from the Swift side: everything under
``steerlab_server.steering`` is **concept-agnostic** — no assumptions about
any particular concept or study domain. If a change there would not work
equally for an arbitrary concept, it is wrong.
"""

# One release version across both engines as of the public flip
# (v0.9.0, 2026-08-20); the Swift SteerLabVersion.version matches.
__version__ = "0.9.2"
