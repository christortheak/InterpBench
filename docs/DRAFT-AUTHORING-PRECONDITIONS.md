# Draft authoring file preconditions

This is the concurrency contract for the researcher workflow implementation.
A file SHA-256 is an external precondition for a particular read of a particular
manifest path. It is not the study's content hash, freeze hash, validation scope,
or a field in the manifest JSON. Existing frozen files require no migration.

## Read, review, apply

1. Capture the workspace, study name, and exact manifest bytes together.
2. Present or reason over that document. Preserve its external file digest.
3. Submit the intended edit to that same destination with that digest.
4. The owner acquires the shared manifest lock, compares the digest, applies
   lifecycle admission, and publishes atomically before releasing the lock.
5. A stale precondition refuses the edit. Read the new document, review the
   intervening change, and reconstruct the intended edit. Do not fetch a new
   digest and silently retry the old whole-document replacement.

The lock lives under `.steerlab/manifest-locks/`, keyed by the SHA-256 of the
canonical absolute manifest path. Python and Swift use the same `flock` protocol
and a stable lock inode. A local ignore rule keeps these runtime files out of
workspace commits. Neither lock creation nor read/check rewrites a manifest.
The mechanism coordinates participating writers; an arbitrary external editor
that ignores the protocol can still race a write.

## Python and engine HTTP

`experiment_store.load_raw` returns a `manifest_files.Document`: a dictionary
with `source_path` and `source_digest` attributes outside its JSON keys. Its
`copy()` preserves the read precondition. `dict(document)` deliberately produces
an ordinary dictionary without overwrite authority. `save_raw` accepts an
explicit `expected_file_sha256` for an ordinary dictionary; omitting that
precondition is create-only. Retain the document returned by creation or save
when continuing an authoring transaction.

`GET /api/experiment/{name}/manifest` returns the exact stored document with a
strong `ETag` containing its SHA-256. The JSON shape is unchanged.

`PUT /api/experiment/{name}/manifest` requires one of:

- `If-Match: "<etag digest>"` for the previously reviewed file.
- `If-None-Match: *` to create a draft only if it is absent.

Missing preconditions return HTTP 428 with `manifest_precondition_required`;
malformed or conflicting preconditions return 400 with
`invalid_manifest_precondition`; stale preconditions return 412 with
`staleManifest`. Each has a repair. The existing draft/frozen and arm-preservation
rules still apply after the precondition succeeds. Runtime service authority
still determines whether the deployment can author at all.

## Swift owners and adapters

`steerlab-cli experiment manifest <name> --json` exposes `result.document` and
`result.manifestFileSHA256`. Reading is side-effect free. This command does not
add a manifest revision field or invent an edit operation.

`DraftAuthoringSnapshot` binds a decoded manifest to its exact file snapshot and
workspace. `DraftAuthoringTransaction.replace` admits a replacement against that
snapshot under the common lock. The panel's whole-document persistence uses the
snapshot that supplied the displayed document. Fresh field setters hold the
lock for their entire load–modify–save operation.

`StudyProtocolAuthoring.save` accepts a reviewed snapshot, complete
`StudyProtocolFields` values and an optional `StudyProtocolScenario`. It checks
the manifest precondition and draft admission before pinning inputs or compiling
seats, retaining the common lock through publication. The app captures its field
values through `StudyDraftState`; the service retains no panel or observable
editor state. Rubric and prompt pins, scenario reads and compiled output use the
snapshot's explicit workspace. A scenario selection carries decoded content and
the hash of the same bytes. Recompilation preserves that provenance rather than
stamping a later source-file version onto earlier content.

This command returns the saved snapshot, seat-edit reset status and advisories,
or a request to select a scenario. It throws typed admission/field errors without
publishing the manifest. These are complete setup values, not a sparse patch:
adapters must deliberately populate the fields they intend to retain. Public
CLI/HTTP request-schema migration remains part of WP-2/WP-3; the existing app
HTTP setup adapter currently reaches the command through the panel.

Server draft sync carries the digest read during its earlier identity check;
it refuses a missing reviewed version instead of inventing one at push time.
Its transport binds that review to the connection and workspace. Model revision
adoption from the response uses the local document captured for the push, so a
concurrent change to the local model cannot acquire the old model's revision.

## Remaining migration gate

This is an implemented foundation, not a declaration that WP-2 is complete.
The legacy Swift `ExperimentStore.save` still allows existing internal callers
without an explicit precondition. They now serialize publication, but a caller
holding an old whole document still needs conversion to a reviewed snapshot or
a narrowly scoped transaction. Finish the remaining task, pipeline, optimization,
creation/rename and adapter audit; remove selection-dependent authoring and the
three remaining compatibility bridges. Do not treat a lock around publication alone as
protection against a stale read. The complete acceptance gate remains in the
[implementation plan](RESEARCHER-WORKFLOW-IMPLEMENTATION-PLAN.md).
