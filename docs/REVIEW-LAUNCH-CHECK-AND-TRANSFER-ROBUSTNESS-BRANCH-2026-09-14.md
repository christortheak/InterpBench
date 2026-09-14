# Review: claude/launch-check-and-transfer-robustness

Date: 2026-09-14. Reviewed at branch head 14bd097 (four commits over main
42be802, fast-forwardable). Origin: the second incident of 2026-09-13
evening, in which the build script's eight-second launch check connected a
freshly built app to a live controller, its evidence auto-import began
fetching every succeeded job's bundle it had never seen and a 12 GB science
export was produced, the script killed the app mid-transfer, and the
controller stopped answering for over half an hour.

## 1. What the branch does

- **Offline launch check.** `STEERLAB_LAUNCH_CHECK=1` (any value but
  `0`/`false`/`no`) puts the app in a no-network mode read once from the
  environment: the connection store's `client` is nil and `connect()`
  refuses; every `ClusterClient`, however constructed, is built on a
  URLSession whose only protocol refuses and records the request; the tunnel
  runs no `ssh` at all (a refusing process runner at construction, `open()`
  refuses); `ChatService.connectCluster()` refuses before the tunnel step;
  the auto-import service neither polls nor transfers; the automatic update
  check is skipped. The app reads an isolated, wiped defaults suite and an
  empty temporary site registry, prints an "armed" line at once and a
  verdict line four seconds later: the offline line, or a violation naming
  the refused attempts. `scripts/build-app.sh` delegates the check to the
  new `scripts/app-bundle/launch-check.sh`, which runs the app with the
  switch and `STEERLAB_WORKSPACE` pinned to a scratch workspace the bundled
  CLI bootstraps, for ten seconds, and fails the build (exit 6) on a
  violation, a missing armed line, a missing verdict, or an early exit. The
  check's own behaviour is covered by `scripts/tests/launch-check-test.sh`
  (six cases against fake executables; never launches a GUI app).
- **Bounded auto-import.** First automatic pass no earlier than 60 s after
  polling starts (Import now is immediate); at most five new bundles per
  automatic pass, the rest recorded as deferred; every bundle download goes
  through a process-wide `EvidenceTransferGate` (two concurrent downloads,
  one export), shared by auto-import, job rows, and chain import; every
  decision is a `deferred` event in the feed. The agent verified from code
  that the auto-import service has no path to `POST /api/science/jobs/{id}/export`
  and excludes `science:*` job kinds; exports are reached only by explicit
  fetches, which now serialize through the gate.
- **Controller robustness.** `Server/steerlab_server/api/transfer_limits.py`:
  a per-`ServiceState` `TransferLimits` with a dedicated download executor
  (`STEERLAB_DOWNLOAD_CONCURRENCY`, default 8) and a single-worker export
  executor. `GET /api/bundles/download` is now `async`, answers 503 with
  `Retry-After: 5` when every slot is busy, and streams through
  `BoundedFileResponse`: FileResponse-identical headers and bytes in 64 KiB
  chunks, each read on the download executor, a disconnect check between
  chunks, the slot released on every exit. `POST /api/science/jobs/{id}/export`
  runs its tar and hash on the export worker, holding no request-threadpool
  token. Route paths, success bodies, the refusal shape, and the
  `require_http_transfer` gate order are unchanged.

## 2. What I checked

- Read the full diff: 23 files, +1841/−49. Engine module and the two route
  changes; the launch-check mode, transfer gate, and auto-import changes;
  the connection store, tunnel, client, chat service, app entry, update
  signpost, and panel changes; both scripts and the script test; docs.
- The wedge mechanism the agent confirmed from the stack: sync `def` routes
  (capabilities included) and FileResponse's chunk reads all draw from
  anyio's single 40-token default limiter; the export held a token for its
  whole duration; uvicorn's `send()` returns silently after a disconnect, so
  abandoned downloads kept reading whole files. The fix removes transfers
  from that limiter and stops them on disconnect. Exact token arithmetic for
  the observed 30 minutes is not provable from code; the design does not
  depend on it.
- Edge behaviour of `BoundedFileResponse`: empty file, exact multiple of the
  chunk size, HEAD, Range answered with a full 200 body (no SteerLab client
  sends ranges), slot release on resolve failure (400/404) and on any exit.
- Launch-check semantics: the app honours `STEERLAB_WORKSPACE` (WorkspaceStore
  precedence 1), so the scratch workspace pin is real; the verdict window
  (4 s) is inside the script's window (10 s); the isolated defaults suite is
  wiped on each launch; `--no-verify` unchanged.
- Gates on the branch tree: `check-generated.py --audits` PASS,
  `public_scan.py` clean, `git diff --check` clean, identifying-vocabulary
  grep clean; `zsh scripts/tests/launch-check-test.sh`: all six cases pass
  (it is a zsh script by shebang; under bash it fails on zsh syntax, which is
  expected and is how I first ran it by mistake).
- Suites on the branch tree: see §4.
- Live acceptance of the launch check: the app rebuild after landing runs the
  new check for real; recorded in §5.

## 3. Findings

No landing fix. Notes:

- **N1 Science stage still on the request threadpool.** `POST /api/science/stage`
  re-hashes staged archives (10 minutes for 12 GB) inside a sync handler;
  the agent left it out of scope. Same executor treatment is the natural
  follow-up.
- **N2 Export not cancelled on disconnect.** By design: the archive is keyed
  by job context so a retry finds it ready; a client that disconnects leaves
  the export running to completion on the single worker.
- **N3 Ledger policy unchanged.** A cold ledger against a controller with
  months of succeeded jobs still imports everything, five per pass; the
  branch bounds and surfaces this rather than filtering by age.
- **N4 Settle delay applies per `startPolling()`.** A workspace switch also
  restarts the 60-second window; acceptable, and the feed says so.

## 4. Suite results on the branch tree

Python (`Server/`, main venv, `HF_HUB_OFFLINE=1`): 6,612 passed, 9 skipped,
8 warnings in 258 s (8 new transfer-limit tests).

Swift (Xcode beta, Metal toolchain 32023.920.1, serial, coverage mapping off,
test Python = main venv): TEST SUCCEEDED, 4,959 tests, 4,954 passed, 5 skipped,
0 failed (8 new launch-check and transfer-bound tests).

## 5. Landing

Fast-forward of main to 14bd097, this review committed on top, app rebuilt
and installed through the new offline launch check (LAUNCH_RESULT), engine
pushed when the cluster session is next available.
