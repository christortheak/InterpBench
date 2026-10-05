# Evidence custody

<!-- client: all -->

Custody proves bytes: that the evidence in this workspace is the evidence that
was imported. It says nothing about scientific quality, and it never grants
permission to delete anything.

<!-- client: mac -->

**`data custody <run-id>`** discovers receipts for an imported run, including
receipts for a pipeline archive that carried that run. Discovery lists recorded
imports; it does not verify current evidence bytes. Use `data verify-custody`
on a returned digest for that check. Both operations work offline.

**`data verify-custody <receipt-sha256>`** rechecks a local evidence archive
and its imported files without contacting a server. Use the digest returned by
`remote import --json`, in the originating workspace. Missing or changed bytes
refuse with `custodyUnverified` and a repair action (exit 65). A verified receipt
proves local possession of those bytes, not scientific validity or permission
to delete remote evidence.

<!-- client: python -->

**`science custody`** re-verifies and lists this workspace's diagnostic
evidence receipts. **`science verify-custody <receipt-sha256>`** re-reads the
retained archive and every expanded local output before reporting custody.
Both work offline. Receipts are written by `steerlab runner science-fetch` and
`steerlab science import`. Missing or changed bytes refuse with a repair
action.

Study-run evidence comes home through `steerlab run`, or by hand with
`steerlab runner evidence <job-id> --out <file.tar.gz> --runner <url>` then
`steerlab bundle import <file.tar.gz> --sha256 <digest>`. Always pass the
digest: it is the only check that catches a wholly substituted archive.
`steerlab bundle inspect <file.tar.gz>` reads a bundle's metadata and
recomputes its outer digest without importing. `run` also stamps
`runs/<runID>/remote-execution.json` with the runner's URL and engine version,
the bundle digest, the job id, the submitted verb and executor, and the
outcome; the token never appears in it.

A verified receipt or digest proves local possession of those bytes, not
scientific validity or permission to delete remote evidence.

<!-- client: all -->

Custody and bounded cleanup of diagnostic outputs (batteries, stability, and
other managed operations) are covered in `{{cli}} workspace guide methods`.
