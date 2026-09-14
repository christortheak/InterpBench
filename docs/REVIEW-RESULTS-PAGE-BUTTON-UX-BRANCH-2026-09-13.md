# Review: claude/results-page-button-ux-94d33e (Server Jobs actions, guided diagnostic form, wrapping evidence sheet)

Date: 2026-09-13. Reviewed at branch head 0521032 (one commit over main
0208750, fast-forwardable). Origin: researcher UI complaints of 2026-09-13
about truncated Server Jobs buttons, a blank-field diagnostic form, and an
evidence sheet whose rows ran off the window.

## 1. What the branch does

- **Server Jobs toolbar.** Refresh and Import runs stay on the toolbar; Run
  scientific diagnostic…, Stage inputs, collect evidence, clean up…, and
  Reconcile job records move behind an Actions menu with full labels and
  longer help. In the log header, Recovery review… becomes Controller
  recovery… and the Stream/Stop stream pair becomes one Follow log / Stop
  following toggle (selecting a job already starts the stream, so the
  resting state no longer shows a live Stop beside a disabled Stream).
- **Run scientific diagnostic form.** Battery chosen by name from this
  workspace's `prompts/batteries/` (a Finder chooser constrained to the
  workspace and a typed server-relative path behind a disclosure); model
  chosen from the server's installed inventory (`/api/state`) with a typed
  identifier behind a disclosure; pinned revision filled from the server's
  model cache through the existing load preflight, never guessed; agent
  lines trimmed and blank lines dropped; a caption states, per the typed
  agents, whether a model pin is required, using the engine's agent grammar.
  The rules live in a new tested ExperimentKit type,
  `ScientificDiagnosticInputs` (battery listing, workspace-relative path with
  symlinked-root handling, agent-line cleaning, agent kind, model caption).
- **Stage inputs / evidence / cleanup sheet.** Relayout into a grouped,
  wrapping form with a 860 × 660 minimum (was 1050 × 740) and the campaign
  and fitting-round groups collapsed until opened. Actions and their gating
  are unchanged: I diffed the removed and added `Button`/`Toggle`/`.disabled`
  lines and they are the same set with the same conditions.
- Renamed controls are reflected in the shipped agent guide (one line), the
  bootstrap text, and four docs; the changelog entry describes the change.

## 2. What I checked

- Read the full diff: 14 files, +786/−223 (about 300 lines of it generated
  bootstrap text and identity). Swift app views, the new ExperimentKit type
  and its 6 tests, one line of the shipped Python client resource, docs.
- The diagnostic request body: `agents` now comes from the cleaned lines, so
  a trailing newline no longer reaches the server as an empty agent (the
  refusal the researcher hit). `batteryFile` is a workspace-relative path
  whichever route filled it; an absolute path outside the workspace is
  refused client-side with an explanation. `revision` is set only from the
  server's preflight when it returns a 40-character commit, with an honest
  note when the weights are not cached there.
- `workspaceRelativePath` handles `/var` → `/private/var` on either side,
  refuses the root itself and a sibling whose name merely extends the root's.
- Gates on the branch tree: `check-generated.py --audits` PASS, `public_scan`
  clean, `git diff --check` clean, identifying-vocabulary grep clean.
- Suites on the branch tree: see §4.

## 3. Findings

No landing fix. Notes:

- **N1 Live UI check owed.** The relayout and the new pickers were not
  exercised by me in the running app in this session; the P7-D surface pass
  is the natural place (the diagnostic form and the evidence sheet are both
  on its matrix).
- **N2 Battery listing is local.** The picker lists this Mac's
  `prompts/batteries/`; the server reads the same relative path in its own
  workspace, so a Mac-authored battery still needs staging. The header text
  says so; a server-side listing would be the exact version.
- **N3 Model picker uses `/api/state` models.** That is the installed
  inventory the study judge picker uses; a model prepared but not yet listed
  there needs the "Use another model" disclosure.

## 4. Suite results on the branch tree

Python (`Server/`, main venv, `HF_HUB_OFFLINE=1`): 6,588 passed, 9 skipped, 8 warnings
in 230 s (unchanged count; the only Python change is one guide line).

Swift (Xcode beta, Metal toolchain 32023.920.1, serial, coverage mapping off):
TEST SUCCEEDED, 4,951 tests, 4,946 passed, 5 skipped, 0 failed (six new
`ScientificDiagnosticInputsTests`).

## 5. Landing

Fast-forward of main to 0521032, this review committed on top, app rebuilt
and installed (Swift and a shipped Python resource changed), engine payload
pushed (the running controller keeps its loaded resources until its next
start; only a static guide line changed on the Python side).
