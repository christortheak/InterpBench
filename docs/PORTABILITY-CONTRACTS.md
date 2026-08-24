# Portability Contracts

**Phase-0 deliverable of the portability program.** Phase 0 changes no
production behaviour. Its entire output is this page plus the golden tests it
indexes: a record of what the two engines currently promise each other, so a
later phase that breaks one of those promises fails a test instead of failing
in somebody's workspace.

The program's target is a cross-platform Python client that will have to do
three things the Swift and Python engines already do to each other:

1. **author a workspace the Swift engine reads** — manifests, pins, conditions;
2. **submit a hash-pinned run bundle either engine produced**;
3. **import an evidence bundle with verification** — and refuse a bad one.

Each contract below names what it guarantees and the test that pins it. Where a
contract *should* exist but cannot be pinned today, it is listed in
[§5 Gaps](#5-gaps-for-later-phases) rather than quietly omitted.

Two files carry the new goldens, one per engine, each the other's twin:

- `Tests/ExperimentKitTests/PortabilityContractTests.swift`
- `Server/tests/test_portability_contracts.py`

They reuse the suite's existing idioms rather than inventing new ones:
**producer-generated fixtures** (the engine that really emits the bytes writes
the fixture; the other engine's test consumes it — the discipline
`scripts/regenerate-cross-engine-fixtures.py` and `test_fixture_staleness.py`
established), **write-if-missing goldens** (`ExperimentCLIEnvelopeTests`,
`test_cli_envelope.py::_check`), and **twin literals** (`CLIEnvelopeParityTests`
↔ `test_cli_envelope.py`).

---

## 1. Manifest interop

| Contract | Guarantees | Pinned by |
|---|---|---|
| **manifest-interop** | A manifest the Python engine authors (create → attach → pin) decodes in the Swift engine with identity, pin surface, arms, and sampling policy intact. | fixture `Tests/Fixtures/cross-engine/manifest-interop.json` (written by `test_a_python_authored_manifest_fixture_is_current`); consumed by `PortabilityContractTests.aPythonAuthoredManifestReadsWithEveryFieldIntact` |
| **manifest-interop-lossless** | Decoding a Python-authored manifest and re-encoding it in Swift drops **no** key. `Codable` discards what it does not declare, so a server key Swift has not learned about would silently vanish the first time the Mac saved the study. | `PortabilityContractTests.aPythonAuthoredManifestSurvivesTheSwiftModelWithoutLosingKeys` |
| **manifest-interop-reverse** | A manifest the Swift engine authors loads in the Python engine with every cross-engine field intact, and its raw document survives key-for-key. | fixture `Tests/Fixtures/cross-engine/swift-authored-manifest.json` (written by `PortabilityContractTests.theSwiftAuthoredManifestFixtureIsCurrent`); consumed by `test_a_swift_authored_manifest_loads_with_every_field_intact` |
| **manifest-canonicalization** | The identity of a study is what it *measures*: the nine volatile freeze stamps (`status`, `frozenAt`, `freezeHash`, `gitCommit`, `frozenBy`, `createdAt`, `appVersion`, `freezeForced`, `forcedGatesSkipped`) are outside the content hash on **both** engines, and measured surface is inside it on both. | `PortabilityContractTests.volatileFreezeStampsAreOutsideTheContentHash` ↔ `test_volatile_freeze_stamps_are_outside_the_content_hash` |
| **freeze-hash-is-the-content-hash** | On the Python engine the freeze hash and the frozen document's content hash are the same value reached by two code paths (`Manifest.content_hash`, `_write_freeze_canonical`), so a client may recompute either. | `test_a_python_authored_manifest_fixture_is_current` |
| **server-freeze-canonical** | A study the Python engine froze verifies on the Swift engine against the server's *exact* `freeze-canonical.json` bytes: `sha256(bytes) == freezeHash`, and the parsed content deep-equals Swift's re-encoding minus the volatile stamps. | `PortabilityContractTests.aServerFrozenManifestVerifiesAgainstItsOwnCanonicalBytes` |
| **condition-globals-are-required** | The Swift decoder has **no default** for a condition's `bandWidth` / `alphaInNormUnits`; the Python store stamps both in `_condition_entry`. A client must route condition authoring through that API (or write both keys). | `PortabilityContractTests.aConditionWithoutItsGlobalsIsUnreadableHere` |

**Not a contract, deliberately:** the two engines' manifest content hashes are
**not** equal for the same study, and never were. `Manifest.content_hash` says
so in its docstring, and `CrossEngineLifecycleTests` documents why comparing
them across substrates is not a strict check but a check that *cannot pass* —
it refused legitimate cluster evidence in a workspace where every sweep is
foreign by design. Cross-engine safety rests on the stimulus SHA-256s and on
the run's own manifest snapshot, both of which each engine can hash itself.
A future client must adopt the same rule: **never compare a hash produced by
one engine with a hash produced by the other.** The bytes-level contract that
*does* cross engines is `server-freeze-canonical`.

## 2. Run bundles

| Contract | Guarantees | Pinned by |
|---|---|---|
| **run-bundle-metadata** | The `steerlab-bundle.json` header is the same nine keys on both engines (`createdAt`, `entries`, `experiment`, `experimentContentHash`, `kind`, `rootRelative`, `schemaVersion`, `validationScopeHash`, `verificationViolations`), and each entry is exactly `{path, sha256, bytes}`. This agreement is what makes a bundle from either engine submittable to either engine. | fixture `Tests/Fixtures/cross-engine/run-bundle-metadata.json` (written by `test_a_run_bundle_metadata_fixture_is_current`); Swift half `PortabilityContractTests.theRunBundleMetadataShapeMatchesTheServerLiteral` |
| **run-bundle-swift-to-python** | A bundle whose header is the Swift packager's metadata literal is inspectable and importable by the Python engine, with the Swift-only keys (`rootRelative`, `validationScopeHash`, `verificationViolations`) surviving the reader rather than being dropped or refused, and the outer digest recomputed from the file. Twin-literal, not a live archive: the Swift half asserts `RunBundlePackager` really writes that header. | `test_a_swift_packaged_run_bundle_is_inspectable_and_importable` ↔ `PortabilityContractTests.theRunBundleMetadataShapeMatchesTheServerLiteral` |
| **run-bundle-pin-closure** *(pre-existing)* | Every pinned input the manifest declares is packed, derived mechanically from `pinnedInputEntries` rather than a hand-written list. | `RunBundleClosureTests.runBundlePacksTheEntirePinSurface` ↔ `test_bundles.py::test_run_bundle_packs_the_entire_pin_surface` |

## 3. Evidence verification

| Contract | Guarantees | Pinned by |
|---|---|---|
| **evidence-member-integrity** | One flipped member byte is refused, and the refusal **names the member** — the only part a remote client can act on. | `test_a_flipped_member_byte_is_refused_and_the_refusal_names_the_member` (naming); `test_bundles.py::test_tampered_member_refuses_before_touching_the_workspace` (pre-existing: refusal happens before any disk write, no debris) |
| **evidence-path-containment** | A tar member whose path escapes the target root is refused by name, and nothing is written anywhere. | `test_a_member_escaping_the_target_root_is_refused` |
| **evidence-import-idempotence** | Re-importing identical evidence is not a way to quietly rewrite a workspace: it refuses, the existing files are byte- and mtime-identical afterwards, no second copy appears, and no staging debris is left. With the explicit override it is content-stable. | `test_re_importing_identical_evidence_neither_duplicates_nor_rewrites` (extends the pre-existing `test_bundles.py::test_import_rejects_overwrite`, which pinned only the refusal) |
| **evidence-outer-hash** | The archive digest is an **out-of-band** pin the recipient checks: it is never stamped inside the archive it protects, `inspect_bundle` recomputes it from the file, and one flipped member byte moves it. On the Mac the check runs before extraction. | `test_the_outer_bundle_hash_travels_out_of_band_and_moves_with_one_member`; Swift half `CrossEngineLifecycleTests.aBundleWhoseHashDoesNotMatchIsRefused` (pre-existing) |
| **evidence-wire** *(pre-existing)* | A real `package_evidence` archive imports on the Mac, its agent becomes discoverable, its vector resolves, and its declared sibling runs come home. | `CrossEngineLifecycleTests` over `Tests/Fixtures/cross-engine/server-evidence-bundle.tar.gz` |
| **member caps and streaming** *(pre-existing)* | Declared-oversize members are refused before expansion; a lying tar header is caught mid-stream; metadata has its own smaller bound. | `test_bundles.py::test_a_member_over_the_uncompressed_cap_is_refused` and siblings |

## 4. Envelope

| Contract | Guarantees | Pinned by |
|---|---|---|
| **envelope-vocabularies** *(pre-existing)* | Header keys, optional keys, schema version, engine stamp, the twelve states and their exit codes, both gate vocabularies, and the advisory codes are identical literals on both engines. | `CLIEnvelopeParityTests` ↔ `test_cli_envelope.py` twin-literal tests |
| **envelope-goldens** *(pre-existing)* | One committed document per agent-path verb, produced under a pinned clock, byte-compared. | `Tests/Fixtures/cli-envelopes/` + `Server/tests/fixtures/cli-envelopes/` |
| **envelope-dispatch-fields** | In **every** committed golden: `state` is in the closed vocabulary and resolves to an exit code; `workspace`, when present, is a non-empty locator; and every refusal carries a non-empty `error.code` **and** an actionable `error.repairAction`. These are the four fields a cross-platform client branches on, and the previous structural checks allowed them without requiring them. | `PortabilityContractTests.everyGoldenCarriesTheFieldsAClientDispatchesOn` ↔ `test_every_committed_golden_carries_the_fields_a_client_dispatches_on` |

---

## 5. Gaps for later phases

Contracts that *should* exist and cannot be pinned today. Each is recorded
here because a gap nobody wrote down is a gap the next phase rediscovers by
breaking something.

**G1 — A server-authored frozen study reads as drifted on the Mac.**
The Swift encoder always writes `multiAgentIncludeBaseline` and
`recordTokenIDs`; the Python engine omits them when they hold their defaults.
The defaults *agree* on both engines, so the two documents describe the same
study — but `serverFreezeCanonicalViolations` compares parsed documents, and a
key present on one side only is a difference. A study authored entirely on the
server therefore fails post-freeze verification with the generic "manifest
content changed after freeze (hash mismatch)", naming no field. It has stayed
invisible because today's server-frozen studies were authored on the Mac first
and so already carry both keys — but *authoring entirely on the server is
precisely what the future client does.*
Pinned as-is (behaviour unchanged) by
`PortabilityContractTests.aServerAuthoredFrozenManifestReadsAsDriftedHere`,
with the working half beside it for contrast.
*Phase 1: make the comparison default-aware, or make the refusal name the
differing keys. Either is a behaviour change and out of scope here.*

**G2 — The Swift engine has no run-bundle reader.**
`RunBundlePackager` writes `steerlab-bundle.json`; nothing on the Swift side
reads one back. `EvidenceBundleImporter` reads *evidence* bundles only. So
"a bundle either engine produced is inspectable by either engine" is currently
true in one direction only, and the Python→Swift half is pinned as a metadata
**shape** fixture rather than a round trip.
*Phase 1: either add a Swift inspector, or state that run-bundle reading is a
server-side capability by design and let the client rely on that.*

**G3 — The Python importer has no `expected_sha256` parameter.**
The outer archive digest is carried out of band and checked by the consumer;
on the Mac that is `EvidenceBundleImporter.importEvidenceBundle(_:expectedSHA256:)`,
which verifies *before* extracting. `bundles.import_bundle` offers no
equivalent, so a Python client must perform the check itself before calling in.
*Phase 1: give the Python importer the same pre-extraction check, so the
verification story is one story rather than one per engine.*

**G4 — A hand-built condition is unreadable on the Mac, with no typed refusal.**
See **condition-globals-are-required** above: the obvious client-side shape
`{"name": …, "slots": […]}` produces a manifest the Swift decoder rejects with
a `keyNotFound` deep inside `conditions[0]` rather than a typed refusal with a
`repairAction`. This is the same failure shape as open-issues #11 (the adapter
entry with no `name`), one level up.
*Phase 1: default the two globals on decode, or refuse with a named gate.*

**G5 — No cross-engine `push_manifest` canonical-body agreement.**
`experiment_store.push_manifest` returns a `canonicalBodyHash` over the whole
merged document; `ExperimentStore.canonicalManifestBodyHash` hashes the
document *minus* the volatile keys and with nulls stripped. The two are
therefore not comparable, and the server's docstring already says the value is
informational (the app re-fetches and compares documents itself). Nothing
depends on them agreeing today, and nothing pins that they do not.
*Phase 1: either align them or rename one, so a client author cannot reasonably
read the two identically-shaped fields as the same quantity.*

**G6 — `alphaInNormUnits` defaults differ, and it shows in the fixtures.**
Declaring a condition without naming the key yields `false` on the Python
engine (`experiment_store._condition_entry`) and `true` on the Swift engine
(`ExperimentManifest.Condition.init`). Nothing is broken in flight — both
engines always *write* the key, so a condition that has crossed the wire is
unambiguous — but the same client call produces a different study depending on
which engine served it, and α units are not a cosmetic setting (see the
alpha-in-norm-units convention). The divergence is visible in the two
committed manifests: `manifest-interop.json`'s conditions carry `false`,
`swift-authored-manifest.json`'s carry `true`, from the same two-condition
declaration.
Pinned on both sides by
`test_the_condition_alpha_unit_default_diverges_from_swift` ↔
`PortabilityContractTests.theConditionAlphaUnitDefaultDivergesFromTheServer`.
*Phase 1: pick one default and state it in both places — a client cannot be
asked to know which engine answered.*

---

## 6. Regenerating the fixtures

The Phase-0 fixtures live beside the existing cross-engine ones in
`Tests/Fixtures/cross-engine/`, and each is regenerated by **its producer**:

| Fixture | Producer | Regenerate |
|---|---|---|
| `manifest-interop.json` | Python engine | delete the file, re-run `Server/tests/test_portability_contracts.py` |
| `run-bundle-metadata.json` | Python engine | same |
| `swift-authored-manifest.json` | Swift engine | delete the file, re-run `PortabilityContractTests` |

`test_fixture_staleness.py::test_every_committed_fixture_has_a_staleness_test`
holds the register: a new fixture with no staleness check fails it.

A fixture diff in review means one engine changed its contract. That is the
point — decide deliberately whether the other should follow.
