# Inspect a design and create reviewed studies

A design holds the settings shared by a family of studies. Instantiating it
creates an ordinary study with its own casting and provenance. A description
explains what the design is for; changing that note does not change its scientific
content hash or the studies already created from it.

The Mac CLI, Swift workbench HTTP service and app share reviewed design
description, instantiation and saving commands. They require the file version that supplied the editor.
No file-version field is added to the stored template or study manifest.

## Agent workflow

```sh
steerlab-cli design list --json
steerlab-cli design inspect <name> --json
steerlab-cli design describe <name> --description <text> --file-sha256 <digest> --json
```

List returns `result.catalog.entries` and `issues` for designs it could not read.
Inspection returns the complete `result.document`, its `designFileSHA256`, and
its scientific `contentHash`. Panel designs also return ordered `seatIDs`. If the pinned panel cannot be read,
inspection still returns the design with `advisories`; casting remains blocked. Use **designFileSHA256**, after reviewing that
inspection, as the write precondition. A content hash cannot substitute for the
file digest because metadata edits do not change scientific identity.

Description edits return the saved document and new file digest. A stale digest
returns exit 65 and `designChanged`; malformed names/digests return 64, and a
missing design returns 66. Refusals include a repair action. Inspect and review
intervening edits before reconstructing the intended change. Never fetch a new
digest merely to retry an old edit silently.

## Create a study

```sh
steerlab-cli design instantiate <name> --casting <file.json> --file-sha256 <digest> --study-name <new-name> --json
```

A comparison casting uses `{"agents":[]}` for baseline, or puts reviewed agent
references inside that array. Each reference has exactly `artifactPath` and
`artifactFileSHA256`, taken from `agent inspect`. A panel casting instead uses
`{"seats":{"speaker":null,"reviewer":{"artifactPath":"runs/model-variants/agent/model-variant.json","artifactFileSHA256":"<reviewed-digest>"}}}`.
Use the exact seat IDs returned by design inspection; null explicitly means the
unsteered base model. Extra or missing seats and incompatible agent models refuse.

The command checks the retained design file digest, task-prompt pins and agent
file versions, derives outcome scope, and uses the existing arm/seat compilation
owners before publishing one draft. It returns `name`, `workspaceRoot`,
`manifestFileSHA256` and the saved `document`. Names follow the app's convention:
an occupied name receives a suffix; **use the returned name**. Publication checks
that the destination is still unoccupied and refuses redirects through links.
Design provenance uses the existing scientific content hash. No revision field is
added, no source design or prior study is rewritten, and nothing runs or submits.

## Save a study as a reusable design

```sh
steerlab-cli experiment manifest <study> --json
steerlab-cli design save <study> --manifest-sha256 <source-digest> --name <design-name> --json
steerlab-cli design update <design> --study <study> --manifest-sha256 <source-digest> --file-sha256 <design-digest> --json
```

`save` creates a separate design for a source with no lineage or changed settings.
An unchanged instance reuses its existing design. The optional `--name` and
`--description` apply when a new design is created. The result says whether a
design was created, changed or reused, and reports its actual name. This avoids
filling the library with copies of the same unchanged instance.

`update` replaces the scientific settings of the design named by the study's
lineage. Inspect and review **both** the source manifest and destination design;
supply their external file digests. It preserves the design's name, description,
creation time and parent relationship. Earlier studies retain their own settings,
inputs and lineage stamps. Frozen source studies are legitimate read-only inputs.
No source study or run is rewritten, and no schema gains a revision field.

These operations use saved study settings. In the app, save Study Setup first if
unsaved fields should become part of the design. A source panel must still match
its pin and reviewed bytes. Hoisting removes the casting through the shared panel
owner; only an equal semantic file can be reused. Otherwise the service publishes
content-addressed semantic bytes in the panel library without overwriting a prior
input. Review derivation warnings, especially those describing mixed models or a
changed panel, before instantiating or submitting. The CLI also surfaces these
under the shared `designDerivationWarning` advisory code.

The shared result includes `sourceStudy`, `sourceManifestFileSHA256`, `created`,
`changed`, optional `hashBefore`, `warnings`, and the complete inspected `design`.
A stale source or destination refuses; obtaining fresh digests is not permission
to retry a save without reviewing the intervening changes.

## App workflow

Select the design in Templates and edit its description. Return submits through
the shared command. An unsuccessful save retains the typed text and shows a
notice. **Discard description edits and reload** explicitly reads a new version.
Refreshing the library does not retag the text already in the editor. Changing
workspace or design resets the editor to that context.

The new-studies sheet retains the design and agent files that supplied its
casting table. A changed file refuses minting while keeping the rows available
to inspect. **Discard casting edits and reload design** starts a fresh review and
clears the old rows. A workspace change refuses minting and stops subsequent
submissions; returning callbacks cannot select a study in another workspace.
A batch reports every row's result and preserves successful earlier rows if a
later row refuses. It is not an all-or-nothing transaction.

Save-back confirmation retains both the source and destination reviews and names
the captured source. Catalog refresh cannot advance that confirmation's authority.
The two design-saving panel forwarding methods are retired; the app calls the
management owner with explicit reviews, and it calls the shared saving command.

## Workbench HTTP

These Swift workbench operations require an explicit absolute `workspaceRoot`:

| Operation | Additional body fields | Success result |
|---|---|---|
| `POST /api/design/list` | None | `ok`, `catalog` |
| `POST /api/design/inspect` | `name` | `ok`, `changed: false`, `design` |
| `POST /api/design/describe` | `name`, `description`, `designFileSHA256` | `ok`, `changed`, `design` |
| `POST /api/design/instantiate` | `name`, `designFileSHA256`, `casting`; optional `studyName` | `ok`, saved study `name`, `workspaceRoot`, `manifestFileSHA256`, `document` |
| `POST /api/design/save` | `sourceStudy`, `manifestFileSHA256`; optional `name`, `description` | Shared design-saving result |
| `POST /api/design/update` | `name`, `sourceStudy`, `manifestFileSHA256`, `designFileSHA256` | Shared design-saving result |

The `design` object contains the same fields as CLI inspection. Unknown fields
refuse with 400; a missing write precondition returns 428; a stale one returns
412 `designChanged`; a different served workspace returns 409; a missing design
returns 404. The routes do not select a design or connect compute. Work executes
against the workspace captured when the request arrived.

## Scope and remaining migration

Single-study instantiation is callable through the Mac CLI and Swift workbench;
the app's batch uses the same command with per-row outcomes. Design save/update
are callable through both adapters and the app. Public batch/expansion adapters,
rename and deletion remain unfinished. The Python client/engine has no equivalent design family yet.
Existing seat assignments may carry absolute artifact pins. Creating siblings
normalizes them only when the canonical path belongs to the captured workspace,
then checks the same runs-file and byte-digest rules. The source study is unchanged.
Legacy design writers still need the complete shared locking protocol. The review
lock serializes participating operations; it cannot constrain external editors
or writers that have not migrated. Agent discovery retains its documented store
scan limits. Compiled panel files can remain unreferenced if final study publication
fails after compilation; this is not crash-atomic multi-file publication.
Interactive Templates qualification remains outstanding.
