# Merge input bounds — in-place inventories are not transport archives (2026-09-15)

## The defect

A J-lens fitting round of eight shards completed. The round's `merge-plan`
action builds a `jlens-fit-merge` request whose `fits` lists the eight
completed shard run directories and plans it through
`scientific_execution.plan` → `managed_inputs.plan` →
`managed_inputs.inventory`. The `fits` key has the `trees` input role, so the
inventory adds every file under each shard directory (13 files a shard:
`jacobians.safetensors` and `checkpoint/sums.safetensors` at 6.6 GB each,
plus small JSON and JSONL) and then finished with
`diagnostic_archives.snapshot`, which hashed all of it (about five minutes)
and refused:

    409 diagnosticTransportRefused: "The diagnostic archive is empty or exceeds transport bounds."

`snapshot` is bounded by `MAX_BYTES = 16 GiB`, and eight 27B shards are about
106 GB.

## Why that bound did not belong there

The 16 GiB bound exists for diagnostic *transport* archives: staging an input
bundle to a runner, exporting evidence, fetching it home, verifying custody,
and cleaning up. A merge planned on the controller that already holds the
fits moves none of those bytes. The inputs are pinned by content hash where
they lie and the merge child reads them from the same filesystem. Applying
the transport bound to that inventory made every full-scale round — the
purpose of sharded fitting — unmergeable. The merge *output* (about 13.2 GB of
merged jacobians and checkpoint sums) is legitimately transported later by
export and fetch, and the transport bound has to stay exactly as it is for
that.

## The fix

`diagnostic_archives` now has two inventories with one entry shape:

- `snapshot(root, paths)` — unchanged contract: members that travel, or
  travelled, in an archive; bounded by `MAX_FILES` and `MAX_BYTES` (16 GiB);
  same refusal text.
- `pin(root, paths)` — members an operation reads in place; bounded by
  `MAX_FILES` and a separate, generous `MAX_PINNED_BYTES` (1 TiB) whose note
  explains why it differs. Its refusal reads "The in-place input inventory is
  empty or exceeds the pinning bound." The bound is a sanity check against
  pinning a whole workspace by accident, not a transport promise.

Both refuse on declared sizes *before* hashing, so an over-bound inventory no
longer spends minutes reading bytes it will never accept; the hashed entries
are bounded again afterwards, so a member that grows during review still
refuses.

### The decision per caller of `snapshot`

| Caller | Decision | Reason |
|---|---|---|
| `managed_inputs.inventory` | **`pin`** | The managed-input closure is read in place by the child; when it must travel, `diagnostic_inputs.package` applies `snapshot` to the same closure. |
| `diagnostic_inputs.plan` (battery, stability) | `snapshot`, unchanged | Those closures are only ever planned to travel (package, stage, verify a staged copy). |
| `diagnostic_archives.package` | `snapshot`, unchanged | Packaging is transport. This is what keeps a pinned closure refusing when it is asked to travel. |
| `diagnostic_transport.verify_inputs` | `snapshot`, unchanged | Re-verifies a staged copy that travelled inside a bounded archive. |
| `diagnostic_transport.output` (export) | `snapshot`, unchanged | Export is transport; the merge output travels through here. |
| `diagnostic_cleanup.apply` | `snapshot`, unchanged | Compares against custody entries that travelled. |
| `diagnostic_archives.import_evidence`, `verify_document` | `snapshot`, unchanged | Local expanded evidence is compared with its retained archive. |
| `managed_campaign_engine` (four sites) | `snapshot`, unchanged | Static campaign files are small and are exported later inside the campaign directory; the scientific inputs are pinned separately through `managed_inputs.plan`. |

## The invariant

Transport is bounded by `MAX_BYTES`, exactly as before, everywhere bytes are
packaged, staged, exported, imported, verified, or removed. In-place
inventories are bounded differently (`MAX_PINNED_BYTES`) and never decide
what may travel. A pinned closure that is asked to travel is refused at
`package` with the transport refusal.

## Tests (`Server/tests/test_managed_input_bounds.py`)

Constants are monkeypatched to small values over small fixture files; nothing
writes gigabytes.

- A `jlens-fit-merge` inventory over `trees` whose bytes exceed the transport
  bound plans through `managed_inputs.plan` and `diagnostic_inputs.plan`, with
  every file hashed, and `snapshot` over the same paths refuses.
- Packaging that pinned closure for transport refuses with the exact transport
  text and writes no archive.
- `pin` keeps `MAX_FILES`, refuses on `MAX_PINNED_BYTES`, refuses an empty
  inventory, and equals `snapshot` on a small set.
- Both inventories refuse over-bound sets before reading any bytes.
- A real completed fit (the tiny fixture model) merges through
  `scientific_execution.plan` with the transport bound set below the fit's
  size; the plan still binds input bytes.
- Export re-inventories its output under the transport bound and refuses with
  the exact text; a staged copy is re-verified under it; the battery plan
  refuses under it.

The generated `PythonClientIdentity.swift` moved with the Python sources and
was regenerated with `scripts/ci/check-generated.py --write`; both Mac
targets need a rebuild before the compiled client's identity check passes.

## Round trip timing, noted and left alone

For an eight-shard 27B merge, the bytes are hashed several times, about five
minutes each on the controller's filesystem:

1. `merge-plan` hashes the closure once (one `input_hashes` session per
   `scientific_execution.plan`).
2. `merge-submit` plans again inside the round action and then
   `scientific_execution.submit` re-plans — two hashings in one request,
   because `jlens_rounds.action` is not itself an `input_hashes` session.
   Wrapping the action in `input_hashes.session()` would let the second plan
   reuse the first's digests within one process while every fingerprint is
   still rechecked at use and at scope exit. This is proposed, not done: no
   caching that outlives a review, and nothing that weakens content pinning.
3. The child re-verifies the closure (`input_plan` in its own session), and
   `jlens_merge.source` hashes the two tensors of each fit again.

The fitting-round route is a synchronous FastAPI handler, so the whole
`merge-plan` hashing runs on the request threadpool and holds the submission
lock for those minutes; a client timeout can abandon a plan that then
completes unobserved. Known follow-up, not addressed here.
