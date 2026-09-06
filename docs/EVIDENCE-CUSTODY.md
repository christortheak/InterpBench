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

Chain imports persist receipts through the shared importer, but their summaries
still report run outcomes rather than receipt digests. Some older panel paths
cannot yet supply complete remote origins. Dedicated HTTP and SwiftUI custody
inspection, Python receipt parity, managed cleanup plan/apply, dependency checks,
and the complete remote acceptance journey remain implementation work. No cleanup
operation is enabled by this change. Never improvise deletion inside immutable
`runs/` directories.
