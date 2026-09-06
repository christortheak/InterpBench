# Local evidence custody

A successful Swift evidence import now retains the exact archive, verifies the
published run files, and writes a separate receipt. This applies to the shared
importer used by the app, auto-import and CLI. Runs and study manifests do not
receive custody or concurrency fields.

The receipt records the archive SHA-256, metadata SHA-256, imported run IDs,
expanded-file hashes, local workspace, and remote origin when the caller knows
it. Scientific completeness is optional: an absent metadata field stays unknown.
Failure evidence is retained and identified separately from complete evidence.
Logs or other members not expanded into a run remain in the retained archive.

## Public verification

`steerlab-cli remote import` returns `custodyReceipt`, `custodyReceiptSHA256` and
`archiveSHA256` in its JSON result. Keep the receipt digest from that result.
For later discovery, including receipts created by a chain import, use the run ID:

```sh
steerlab-cli data custody <run-id> --json
```

`result.inventory.entries` lists matching receipts, including archives carrying
that run as a pipeline stage. `issues` reports unreadable/corrupt receipts instead
of silently hiding them. An empty inventory does not establish that evidence is
missing: local runs and older imports may have no receipt. Listing validates the
receipt identity but does not read and verify all evidence bytes.

Then, with the originating workspace selected:

```sh
steerlab-cli data verify-custody <receipt-sha256> --json
```

This is a local, read-only operation. It checks the receipt's content address,
workspace identity, retained archive and every recorded expanded file. Success
returns `result.verified: true` and the receipt. Failure returns exit 65,
`custodyUnverified`, and a repair action. Missing arguments return a usage error.
The shared Swift API is `EvidenceCustodyStore.loadVerified(receiptSHA256:workspaceRoot:)`.

Reverification proves current local possession of the recorded bytes, not their
scientific validity, all possible study outputs, or permission to remove a remote
copy. A partial archive can have valid custody. Unknown remote origin cannot
establish the source identity required for cleanup. Use the digest from the import
result or import ledger; a hand-authored JSON document is not an import receipt.

## App and workbench HTTP

In the study's local Runs & Results, expand **Evidence retained locally** for the
selected run, then choose **Verify retained evidence**. The view uses the local
workspace from that displayed run's path, independent of compute selection.
Discovery and hashing run off the UI actor; a replaced view does not publish a
late verification result into a different run. Refresh clears the previous check.

The Swift workbench exposes the same read-only operations:

- `POST /api/evidence/custody/list`, body `workspaceRoot` (absolute path) and `runID`.
  Returns `ok` and `inventory`, with the same inventory as the CLI.
- `POST /api/evidence/custody/verify`, body `workspaceRoot` and `receiptSHA256`.
  Returns `ok`, `verified`, `receiptSHA256` and `receipt` on success.

Both require exactly the named fields. An invalid request returns 400
`invalidCustodyRequest`; a different workspace returns 409
`custodyWorkspaceChanged`; failed verification returns 409 `custodyUnverified`.
Each refusal includes a repair action. These routes never select a study, connect
compute or delete files. They capture the served workspace before background work.

## Storage and failure behavior

- Exact archives live under `.steerlab/evidence-archives/<sha256>.tar.gz`.
  Identical archive bytes share one retained copy. Archive hashing streams bytes.
- Receipts live under `.steerlab/evidence-custody/<sha256>.json`. Their filenames
  identify their exact JSON bytes. Verification never updates or repairs them.
- Auto-import's external ledger records the receipt and archive hashes. Explicit
  manual re-import can refresh a legacy entry by verifying existing evidence;
  polling still deduplicates a known origin and current bundle version.
- Receipt publication is part of successful import: failure rolls back newly
  published runs, preserving previously verified runs. A failed import may leave
  an unreferenced retained archive. Existing transport downloads are also retained.
- Archive links and special files refuse before run publication. Verification
  refuses symlinked custody storage or expanded evidence. A copied workspace has
  a different identity and does not inherit cleanup authority from these receipts.

## Remaining adapters and lifecycle work

Chain import summaries still report run outcomes rather than receipt digests;
`data custody` discovers their receipts afterward. Some older panel paths cannot
yet supply complete remote origins. Python receipt parity, managed cleanup
plan/apply, dependency checks, interactive UI qualification and the complete
remote acceptance journey remain implementation work. No cleanup
operation is enabled by this change. Never improvise deletion inside immutable
`runs/` directories.
