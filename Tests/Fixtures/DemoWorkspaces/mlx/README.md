# Placeholder Demo Workspace

This is a stand-in used by SteerLab's own tests. It has the shape of a Demo
Workspace so that shipping, copying, and verifying one can be tested before a
real one exists. It is not an example to learn from, and it does not ship.

Everything in it is synthetic:

- **The concept** (`prompts/concepts/courtesy/`) is six invented sentence
  pairs and four invented validation scenarios.
- **The task prompts** (`prompts/tasks/example-task-prompts.jsonl`) are the
  five example prompts that new workspaces used to be seeded with.
- **The frozen study** (`experiments/placeholder-study/`) names a model that
  does not exist. It was frozen with `--force`, and its manifest records that
  the validation gate was skipped.
- **The run** (`runs/2026-01-01T000000000Z-exp-placeholder-study-run/`) was
  written by hand. No model produced its outputs, and nothing in it is a
  measurement.
- **The draft** (`experiments/placeholder-study-draft/`) is a copy of the
  study before it was frozen.

A real Demo Workspace lives under `DemoWorkspaces/<backend>/` at the top of
the repository. Its README says what the study asks, what you will see, and
the steps from opening the workspace to an exported result.
