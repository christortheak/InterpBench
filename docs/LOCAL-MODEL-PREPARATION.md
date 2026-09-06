# Local model preparation

The Mac CLI and local workbench API drive the same installer used by the app.
Installation populates the local Hugging Face cache without loading weights.
It does not modify a study, choose a scientific model revision for a frozen
study, submit a job, or qualify a model's numerical behavior.

## Researcher and agent workflow

```sh
steerlab-cli model plan <owner/repo> --revision <commit-or-ref> --json
steerlab-cli model install <owner/repo> --revision <commit-or-ref> --json
steerlab-cli model plan <owner/repo> --revision <commit-or-ref> --json
```

Omitting `--revision` requests `main`. Use an immutable commit when the intended
study requires a specific version; a moving branch or tag is not an immutable
scientific pin. The plan reports the resolved cached revision when present.

Planning reads the cache's file-presence rules and does not download, load
weights or query credentials. `cacheFileSetPresent` requires the requested ref
and the loader's required file set, including weight shards. It is not a checksum
verification, a load test, a memory-fit decision or scientific qualification.
`memoryFit` and `credentials` explicitly say `notChecked`; download size is not
estimated. A missing snapshot or partial file set answers false.

Install explicitly starts the existing local fetch. The CLI waits in the
foreground and emits one final JSON envelope; diagnostics go to stderr. The
app and HTTP service show the installer's progress. Failure and cancellation
keep their reasons; partial cache files may remain and can be reused by a later
install. Install operations can mutate the cache even if they fail. Inspect the
result and cache plan before deciding whether to retry or load weights.

Local installation has no durable scheduler job and does not share an in-memory
request with a separate CLI process. Restarting the workbench loses its status
record; inspect the cache and explicitly reconstruct an install as needed.
Neither a completed download nor a cached file set proves that a model fits
available memory or passes scientific tests.

## Swift workbench API

These routes explicitly target the Mac hosting the service, even when the app
is displaying a remote compute connection:

| Operation | Request | Result |
|---|---|---|
| `POST /api/local-model/plan` | `modelID`; optional `revision` | Read-only cache plan |
| `POST /api/local-model/install` | `modelID`; optional `revision` | 202 with the accepted installation request and status |
| `GET /api/local-model/status` | None | Current local installer state and request |
| `POST /api/local-model/cancel` | `requestID` from observed status | Status after cancellation; 428 if missing, 409 if another request is current |

Unknown fields and malformed model/revision inputs refuse with 400. A second
install while one is active returns 409 and preserves the active request. Each
accepted installation gets a fresh ID, including a restart of the same model.
A delayed cancellation cannot cancel that successor. The app's installer and
these routes use the same state; a CLI invocation owns its own installer.

## Remaining work and verification boundary

The Python engine already has privileged model installation as a durable job.
Its CLI adapters, explicit preparation plan and server policy qualification are
separate outstanding work; a local plan says nothing about that server's cache,
disk policy, networking or credentials. Native revision-entry controls and
interactive UI qualification also remain. HTTP-started revision requests are
identified in the shared installer state.

Tests use synthetic cache files and injected fetches. They cover requested
revision forwarding, incomplete-cache observation, failures, busy admission,
status and stale cancellation. No real download, weight load or remote model
installation is authorized by this implementation work itself.
