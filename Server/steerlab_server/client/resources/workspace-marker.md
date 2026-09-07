# SteerLab Workspace

Created by SteerLab {{version}} on {{createdAt}}.

This is a data workspace. Inputs and study manifests live in `prompts/` and
`experiments/`; immutable outputs live in `runs/`; adapter data lives in
`adapters/`. Read `AGENTS.md` before authoring or executing a study.

Git tracks authored inputs when available. Frozen input snapshots remain the
reproducibility record even without Git. Existing runs are never rewritten.
