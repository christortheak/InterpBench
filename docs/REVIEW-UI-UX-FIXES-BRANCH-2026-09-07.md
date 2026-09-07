# Review: `ui-ux-fixes` @ `d75adea` merged onto main `11a4cd2`

Reviewer: the maintainer's integration agent, 2026-09-07. The branch was
cut from `6a94a8f` and carries 29 commits (the 2026-09-06 read-only audit,
twelve cluster fix branches merged one by one, an integration pass and two
polish commits): 134 files, +13,335/−2,852, almost all under
`Sources/SteerLabApp`. Main has moved 13 commits since, including UI work of
its own (Research Setup, the science guides' authoring and action sheets,
the jobs panel's custody sheet). This review covers both the branch and the
merge onto today's main.

## 1. Verdict

**Mergeable, as a merge commit, once the Swift suite on the merged tree is
green (row below).** It is not a fast-forward: three files conflicted where
the branch and main both edited the same view, and I resolved all three by
keeping both sides. No scientific owner, engine code or Python is touched;
the branch's changes outside the app are small, honest support changes in
ExperimentKit (Keychain writes now report failure, the connection state is
derived from facts instead of substring-matching a status line, labels
shared so no screen prints a raw enum case, a study created or renamed by
the app begins its own review so a fresh draft no longer warns that it
changed). Six product decisions the audit left to you are still open (§4);
none blocks the merge, and I give a recommendation for each.

## 2. Verified independently

| Check | Result |
|---|---|
| Ancestry and overlap | merge-base `6a94a8f`; four files changed on both sides, three of which conflicted textually |
| Conflicts resolved | `ScienceGuidesView` (main's authoring/action/custody sheets and `workflows` state kept beside the branch's `didLoad` guard, single catalog read and cancel-role Done button; the new custody button gains a help string), `ServerJobsPanelView` (branch's guarded reconcile button with help, plus main's custody button and sheet, plus the branch's cancel confirmation dialog, all kept), `WorkspaceControls` (branch's folder-name section header with main's Research Setup entry, which gains a help string) |
| Branch hygiene | `git diff --check` clean on the branch and on the staged merge; vocabulary scan of the branch diff and all 29 commit messages clean (the audit docs were scrubbed before their commit) |
| ExperimentKit and test diffs (21 files, +395/−72) | read in full; behaviour changes are the four listed in §1 plus two renamed advisories and their tests |
| Full Xcode beta suite on the merged tree (`TEST_RUNNER_STEERLAB_TEST_PYTHON` in the shell environment before `xcodebuild`) | `TEST SUCCEEDED`: 277 SteeringKit + 4,602 ExperimentKit; the app target compiled, including the three resolved files |
| Python suite | not rerun: the branch touches no Python, seed or client resource, and the identity gate is unaffected |

## 3. What the branch does

The consolidated audit found roughly 480 controls without help text, a
handful of live defects and a long tail of design inconsistencies. The
fixes, by cluster:

- **Shell and Home.** The window floor is now enforced through
  `.windowResizability(.contentMinSize)`; the controls pane keeps its floor
  when a persisted divider disagrees; old section names ("Steering",
  "Concept Lab", "Variants", "Geometry") are gone from user-visible copy;
  duplicated Home buttons collapsed.
- **Compute and jobs.** Cancel is parked behind a confirmation that says
  what is lost; reconcile has a busy state and a re-entry guard; one
  connection status per column; connection colour follows the derived
  phase rather than the status text.
- **Cluster connection, wizard, site editor.** Topology shown by its human
  label; Keychain refusals surface instead of reading as "no token".
- **Playground.** Return no longer clears the composer before a refusal can
  fire; transcript header renamed.
- **Concepts, vectors, data, adapters, OptVec, J-space.** Busy states and
  re-entry guards on every server-route button; raw error dumps replaced by
  the error's sentence; empty and loading states added.
- **Agents, templates, studies, evaluation, lifecycle.** The fresh-draft
  "saved study changed" warning fixed at its cause in the controller;
  create and rename errors typed; grid validated before an optimize draft
  is declared, so no orphan drafts.
- **Results and analysis.** Previewable files classified cheaply before any
  parse; unavailable previews carry their reason.
- **Everywhere.** Help strings on the missing controls, a shared
  `CopyButton` with visible "Copied" feedback, `role: .cancel` and
  `.destructive` where they belong, keyboard shortcuts on sheet primaries.

## 4. The six decisions, with recommendations (all six accepted and applied at landing, see §5)

1. **`.id(section.minimumContentWidth)` on the detail split** re-mounts the
   split when a section switch crosses a width floor, which resets a
   dragged divider and a pinned viewer. It exists because a column whose
   minimum changes mid-cycle is fatal on this macOS beta. Recommendation:
   keep it until a live pass shows the crash class does not reach this
   split; a reset divider is annoying, a crash is not.
2. **The 1240 window floor is now really enforced.** Recommendation: accept;
   the Results header wrap the audit saw live was caused by the floor not
   holding.
3. **`.keyboardShortcut(.defaultAction)` on "Add Condition"** sits in a
   Form, not a sheet, so Return anywhere in that window can add a
   condition. The fixer flagged it against its own brief. Recommendation:
   drop it.
4. **The workspace-mismatch banner has no reconnect action** because adding
   one changes a signature used by three clusters. Recommendation: leave
   for a follow-up chip; the banner is correct, only less convenient.
5. **Duplicated recipe pickers and optimization grids** on one screen: which
   survives is a product call. Recommendation: keep the one inside the
   composer, since that is where the value is consumed.
6. **Picking the first scenario on a fresh Multi-Agent tab raises the
   discard dialog** because a pristine template counts as unsaved.
   Recommendation: treat an unedited template as clean; small fix.

Items 1, 2 and the fresh-draft warning were verified in code only; the
live re-check the audit wanted needs a person at the keyboard, and the
rebuilt app is the moment to do it.

## 5. Landing shape and what the landing commit did

Landed as a merge commit from the resolved worktree, followed by one fix
commit applying the six decisions as recommended:

1. and 2. unchanged (the split re-mount and the enforced window floor stay).
3. `.keyboardShortcut(.defaultAction)` removed from "Add Condition".
4. The workspace-mismatch banner gains a "Reconnect" button that calls the
   store's own `connect()`, which refreshes capabilities and the pairing
   verdict; it is disabled while a connect or a repoint is in flight. No
   signature changed: the banner already held the store, so the three call
   sites are untouched.
5. One recipe control: the Concept Vector Builder's second picker is now a
   read-only "Recipe" echo pointing at the Dataset Builder. One grid: the
   "Measured grid" section and its legend are removed from Optimizations,
   and `SweepGridView.swift` with them (no other caller); the clickable
   grid, which is where a cell is chosen and consumed, already carries the
   legend, the α units and the struck-through failure state.
   `SweepGridPresentation` in ExperimentKit stays, with its tests.
6. A never-saved Multi-Agent draft now records the template it was seeded
   from and counts as dirty only when the editor differs from it, so an
   untouched fresh tab no longer raises the discard dialog.

The owed app rebuild covers this branch too, and the first live pass should
walk the three headline defects plus decisions 3, 4 and 6.
