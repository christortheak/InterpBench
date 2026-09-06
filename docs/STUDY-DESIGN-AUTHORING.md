# Inspect and describe a study design

A design holds the settings shared by a family of studies. Instantiating it
creates an ordinary study with its own casting and provenance. A description
explains what the design is for; changing that note does not change its scientific
content hash or the studies already created from it.

The Mac CLI, Swift workbench HTTP service and Templates view share the reviewed
description command. They require the file version that supplied the editor.
No file-version field is added to the stored template or study manifest.

## Agent workflow

```sh
steerlab-cli design list --json
steerlab-cli design inspect <name> --json
steerlab-cli design describe <name> --description <text> --file-sha256 <digest> --json
```

List returns `result.catalog.entries` and `issues` for designs it could not read.
Inspection returns the complete `result.document`, its `designFileSHA256`, and
its scientific `contentHash`. Use **designFileSHA256**, after reviewing that
inspection, as the write precondition. A content hash cannot substitute for the
file digest because metadata edits do not change scientific identity.

Description edits return the saved document and new file digest. A stale digest
returns exit 65 and `designChanged`; malformed names/digests return 64, and a
missing design returns 66. Refusals include a repair action. Inspect and review
intervening edits before reconstructing the intended change. Never fetch a new
digest merely to retry an old edit silently.

## App workflow

Select the design in Templates and edit its description. Return submits through
the shared command. An unsuccessful save retains the typed text and shows a
notice. **Discard description edits and reload** explicitly reads a new version.
Refreshing the library does not retag the text already in the editor. Changing
workspace or design resets the editor to that context.

## Workbench HTTP

These Swift workbench operations require an explicit absolute `workspaceRoot`:

| Operation | Additional body fields | Success result |
|---|---|---|
| `POST /api/design/list` | None | `ok`, `catalog` |
| `POST /api/design/inspect` | `name` | `ok`, `changed: false`, `design` |
| `POST /api/design/describe` | `name`, `description`, `designFileSHA256` | `ok`, `changed`, `design` |

The `design` object contains the same fields as CLI inspection. Unknown fields
refuse with 400; a missing write precondition returns 428; a stale one returns
412 `designChanged`; a different served workspace returns 409; a missing design
returns 404. The routes do not select a design or connect compute. Work executes
against the workspace captured when the request arrived.

## Scope and remaining migration

These operations inspect designs and edit descriptive metadata. They do not yet
create, instantiate, rename, delete or revise the scientific settings of a design
through these public adapters. The Python client/engine has no equivalent design
family yet. Those are separate tracked operations, not additional undocumented
verbs. Existing design creation, save-back, rename and instantiation writers still
need migration to reviewed owners and the complete shared locking protocol.
Interactive Templates qualification also remains outstanding.
