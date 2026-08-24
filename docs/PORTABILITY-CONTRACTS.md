# Portability Contracts

**Phase-0 deliverable of the portability program**, extended by **Phase 1a**
and **Phase 1b** (§7).
Phase 0 changed no production behaviour: its entire output was this page plus
the golden tests it indexes — a record of what the two engines promise each
other, so a later phase that breaks one of those promises fails a test instead
of failing in somebody's workspace. Phase 1a closed four of the recorded gaps —
G1, G3, G6, and G4 alongside G6 — and moved their pins from "pinned as broken"
to "pinned as fixed"; §5 says what was decided in each and why.

The program's target is a cross-platform Python client that will have to do
three things the Swift and Python engines already do to each other:

1. **author a workspace the Swift engine reads** — manifests, pins, conditions;
2. **submit a hash-pinned run bundle either engine produced**;
3. **import an evidence bundle with verification** — and refuse a bad one.

Each contract below names what it guarantees and the test that pins it. Where a
contract *should* exist and still cannot be pinned, it is listed in
[§5 Gaps](#5-gaps) rather than quietly omitted.

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
| **server-freeze-canonical** | A study the Python engine froze verifies on the Swift engine against the server's *exact* `freeze-canonical.json` bytes: `sha256(bytes) == freezeHash`, and the parsed content deep-equals Swift's re-encoding minus the volatile stamps **and minus the keys the server elides at their default** (`ExperimentStore.defaultElidedFreezeKeys`). | `PortabilityContractTests.aServerFrozenManifestVerifiesAgainstItsOwnCanonicalBytes` |
| **server-authored-freeze** *(Phase 1a, was G1)* | A study authored **entirely** on the Python engine — the future client's path — passes the Swift post-freeze check; and a real edit is still refused, now with the differing field **named**. | `PortabilityContractTests.aServerAuthoredFrozenManifestVerifiesHere`, `.aRealEditToAServerFrozenStudyIsStillRefusedAndNamed`, `.theRepairIsInvisibleToEveryHashAndFingerprint` |
| **condition-globals-are-required** | A condition document with no `bandWidth` / `alphaInNormUnits` is refused by the Swift decoder — as a **typed refusal naming the arm, the key and the repair** since Phase 1a, not a raw `keyNotFound` (was G4). | `PortabilityContractTests.aConditionWithoutItsGlobalsIsRefusedByName` |
| **condition-alpha-units-explicit** *(Phase 1a, was G6)* | A **new** condition declaration that does not name `alphaInNormUnits` is refused on **both** engines, with the same repair in both spellings (manifest key and `--alpha-units`). An **existing** key-less condition keeps the reading its engine always gave it (`False` on the server; unreadable on the Mac), and the server advises at freeze rather than converting. | `PortabilityContractTests.theConditionAlphaUnitDeclarationIsExplicitOnBothEngines` ↔ `test_a_new_condition_that_declares_no_alpha_units_is_refused`, `test_an_existing_keyless_condition_keeps_its_original_reading`; CLI half `HeadlessAuthoringTests.declaringAnArmWithoutItsAlphaUnitsIsRefused` |

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
| **evidence-outer-hash** | The archive digest is an **out-of-band** pin the recipient checks: it is never stamped inside the archive it protects, `inspect_bundle` recomputes it from the file, and one flipped member byte moves it. | `test_the_outer_bundle_hash_travels_out_of_band_and_moves_with_one_member`; Swift half `CrossEngineLifecycleTests.aBundleWhoseHashDoesNotMatchIsRefused` (pre-existing) |
| **evidence-outer-hash-pre-extraction** *(Phase 1a, was G3)* | **Both** engines can verify that digest *before* extracting: `bundles.import_bundle(expected_sha256=…)` (and `POST /api/bundles/import` with `expectedSha256`, and `bundle import --sha256`) refuses a mismatch with **both hashes named**, having written nothing and opened nothing. The parameter is optional; absent, behaviour is exactly what it was. | `test_a_mismatched_expected_sha256_refuses_before_anything_is_extracted`, `test_a_matching_expected_sha256_imports_exactly_as_before`, `test_omitting_expected_sha256_is_unchanged_behaviour`; Mac twin `EvidenceBundleImporter.importEvidenceBundle(_:expectedSHA256:)` |
| **evidence-wire** *(pre-existing)* | A real `package_evidence` archive imports on the Mac, its agent becomes discoverable, its vector resolves, and its declared sibling runs come home. | `CrossEngineLifecycleTests` over `Tests/Fixtures/cross-engine/server-evidence-bundle.tar.gz` |
| **member caps and streaming** *(pre-existing)* | Declared-oversize members are refused before expansion; a lying tar header is caught mid-stream; metadata has its own smaller bound. | `test_bundles.py::test_a_member_over_the_uncompressed_cap_is_refused` and siblings |

## 4. Envelope

| Contract | Guarantees | Pinned by |
|---|---|---|
| **envelope-vocabularies** *(pre-existing)* | Header keys, optional keys, schema version, engine stamp, the twelve states and their exit codes, both gate vocabularies, and the advisory codes are identical literals on both engines. | `CLIEnvelopeParityTests` ↔ `test_cli_envelope.py` twin-literal tests |
| **envelope-goldens** *(pre-existing)* | One committed document per agent-path verb, produced under a pinned clock, byte-compared. | `Tests/Fixtures/cli-envelopes/` + `Server/tests/fixtures/cli-envelopes/` |
| **envelope-dispatch-fields** | In **every** committed golden: `state` is in the closed vocabulary and resolves to an exit code; `workspace`, when present, is a non-empty locator; and every refusal carries a non-empty `error.code` **and** an actionable `error.repairAction`. These are the four fields a cross-platform client branches on, and the previous structural checks allowed them without requiring them. | `PortabilityContractTests.everyGoldenCarriesTheFieldsAClientDispatchesOn` ↔ `test_every_committed_golden_carries_the_fields_a_client_dispatches_on` |

---

## 5. Gaps

Contracts that *should* exist and could not be pinned in Phase 0. Each is
recorded here because a gap nobody wrote down is a gap the next phase
rediscovers by breaking something. **G1, G3, G4 and G6 were closed in Phase
1a** and are kept below with what was decided and why; G2 and G5 remain open,
and **G7 was found in Phase 1b** while building the client that G1/G4/G6 made
possible.

**G1 — A server-authored frozen study read as drifted on the Mac.
CLOSED (Phase 1a).**
The Swift encoder always writes `multiAgentIncludeBaseline` and
`recordTokenIDs`; the Python engine omits them when they hold their defaults.
The defaults *agree* on both engines, so the two documents describe the same
study — but `serverFreezeCanonicalViolations` compared parsed documents, and a
key present on one side only is a difference. A study authored entirely on the
server therefore failed post-freeze verification with the generic "manifest
content changed after freeze (hash mismatch)", naming no field. It stayed
invisible because today's server-frozen studies were authored on the Mac first
and so already carry both keys — but *authoring entirely on the server is
precisely what the future client does.*

*The repair:* a **default-insensitive comparison**, not an emission change.
`ExperimentStore.defaultElidedFreezeKeys` enumerates the keys with the default
both engines hold, and `serverFreezeCanonicalViolations` drops each from **both**
sides when it is present *and* at that default; a present non-default value
still differs from an absence, because the server omits the key only at its
default. The refusal now also names the differing top-level fields (reusing
`manifestFieldMismatches`, the remote-freeze identity check's renderer).

*Why not emission alignment (making the Python engine write both keys).* It
would have repaired only *newly authored* server manifests — every study
already frozen on the server would still fail — while moving the content hash
of every new one, for a difference that was never about content. Both were
weighed; the comparison change is the one that fixes the studies that exist.

*What the repair deliberately did not touch.* The normalization lives in the
freeze comparison alone, **not** in `comparableFreezeObject`, which also backs
`canonicalManifestBodyHash` (a fingerprint researchers read off the screen) and
`compareManifestDocuments` (the remote-freeze identity check). Nothing hashed,
stamped or displayed moved — pinned by
`PortabilityContractTests.theRepairIsInvisibleToEveryHashAndFingerprint`, and
no manifest content hash on either engine changed at all.
Pinned as FIXED by `.aServerAuthoredFrozenManifestVerifiesHere`, with the
firewall half `.aRealEditToAServerFrozenStudyIsStillRefusedAndNamed` beside it
and the Mac-authored half (`.aServerFrozenManifestVerifiesAgainstItsOwnCanonicalBytes`)
passing unchanged.

**G2 — The Swift engine has no run-bundle reader.**
`RunBundlePackager` writes `steerlab-bundle.json`; nothing on the Swift side
reads one back. `EvidenceBundleImporter` reads *evidence* bundles only. So
"a bundle either engine produced is inspectable by either engine" is currently
true in one direction only, and the Python→Swift half is pinned as a metadata
**shape** fixture rather than a round trip.
*Still open after Phase 1a: either add a Swift inspector, or state that
run-bundle reading is a server-side capability by design and let the client
rely on that.*

**G3 — The Python importer had no `expected_sha256` parameter.
CLOSED (Phase 1a).**
The outer archive digest is carried out of band and checked by the consumer;
on the Mac that is `EvidenceBundleImporter.importEvidenceBundle(_:expectedSHA256:)`,
which verifies *before* extracting. `bundles.import_bundle` offered no
equivalent, so a Python client had to perform the check itself before calling
in — and the verification story was one story per engine.

*The repair:* an **optional** `expected_sha256` on `bundles.import_bundle`,
verified by `_refuse_outer_hash_mismatch` before `inspect_bundle` opens the
archive at all. A mismatch is a `BundleError` naming **both** hashes, with
nothing written and no staging debris. Fronted additively by
`POST /api/bundles/import` (`expectedSha256`) and `bundle import --sha256`.
Absent parameter = today's behaviour exactly, which is what keeps every
existing call site and every deployed cluster script working; the future client
always passes it. What this catches and the per-member checks cannot is
**substitution**: a wholesale swapped archive is internally consistent.

**G4 — A hand-built condition was unreadable on the Mac, with no typed refusal.
CLOSED (Phase 1a, with G6).**
The obvious client-side shape `{"name": …, "slots": […]}` produced a manifest
the Swift decoder rejected with a `keyNotFound` deep inside `conditions[0]`
rather than a typed refusal with a `repairAction` — the same failure shape as
open-issues #11 (the adapter entry with no `name`), one level up.

*The repair:* `ExperimentManifest.init(from:)` wraps the `conditions` decode
and rethrows through `conditionDecodeRefusal`, which re-reads the array
opaquely to name the arm (by `name`, falling back to `conditions[i]`), names
the missing key, and carries a repair — `ExperimentError.malformed`, so the
envelope answers `blocked`/64 with a `repairAction` rather than `verbFailed`/70.
The **answer** is unchanged: the document is still refused. Defaulting the two
globals on decode was the other option and was rejected — for `alphaInNormUnits`
it would mean inventing a dose unit, which is exactly what G6 forbids.

**G7 — The client's light import graph ends at `experiment verify`.
OPEN (found in Phase 1b).**
Importing `steerlab_server.client_cli` and running `steerlab --help` pulls
*nothing* third-party at all, and `create` / `attach` / `declare-condition` /
`set-protocol` / `duplicate` / `list` / `concept import` pull only what the
client declares (numpy, safetensors). `experiment verify` and `experiment
freeze` — and `bundle package`, which calls `verify` — pull **torch**, through
`Manifest.verify` → `experiment.sae_latent` → `steering.sae_latent` →
`steering.injector` → `import torch`.

*Why it was not repaired in 1b.* `experiment/sae_latent.py`'s own docstring
says "everything here validates OFFLINE: no SAE weights, no HuggingFace, no
network" — and that is true of what it *does*; it is false of what it
*imports*. Four names cross the line (`CLAMP`, `MODES`, `SAELatentEdit`,
`SAELatentFeature`), and the module they come from sits on the injector stack,
so no single lazy import reaches it. The repair is to split the SAE latent
**validation** surface from the **execution** surface — an engine refactor with
cross-engine twin-literal consequences (the mode vocabulary is one of the
closed key sets), not a client change. Doing it inside a client phase, to
satisfy a client guard, is exactly the kind of change that breaks an engine
nobody was looking at.

*Consequence, stated plainly so nobody discovers it at install time:* a bare
`pip install steerlab-server` gives a client that authors, declares and
packages-by-name, but cannot `verify` or `freeze` without `[runner]`. Pinned
as-is (both halves) by `test_client_cli.py::test_the_authoring_verbs_stay_light_and_verify_is_where_that_ends`,
which fails if either half moves — including if somebody closes it.

**G5 — No cross-engine `push_manifest` canonical-body agreement.**
`experiment_store.push_manifest` returns a `canonicalBodyHash` over the whole
merged document; `ExperimentStore.canonicalManifestBodyHash` hashes the
document *minus* the volatile keys and with nulls stripped. The two are
therefore not comparable, and the server's docstring already says the value is
informational (the app re-fetches and compares documents itself). Nothing
depends on them agreeing today, and nothing pins that they do not.
*Still open after Phase 1a: either align them or rename one, so a client author
cannot reasonably read the two identically-shaped fields as the same
quantity.*

**G6 — `alphaInNormUnits` defaults differed, and it showed in the fixtures.
CLOSED (Phase 1a).**
Declaring a condition without naming the key yielded `false` on the Python
engine (`experiment_store._condition_entry`) and `true` on the Swift engine
(`ExperimentManifest.Condition.init`). Nothing was broken in flight — both
engines always *write* the key, so a condition that has crossed the wire is
unambiguous — but the same client call produced a different study depending on
which engine served it, and α units are not a cosmetic setting (see the
alpha-in-norm-units convention). The divergence was visible in the two
committed manifests: `manifest-interop.json`'s conditions carried `false`,
`swift-authored-manifest.json`'s carried `true`, from the same two-condition
declaration.

*The repair — explicitness, not a chosen default.* "Pick one default" was the
Phase-0 suggestion and it is the one thing that could not be done: whichever
default was picked, every existing artifact authored under the other would have
been silently reinterpreted, and a reinterpreted α is a study that measured
something other than what it now says it measured. So **neither engine
defaults, at any surface a client reaches**:

- **New declarations are refused, typed, on both engines.**
  `experiment_store._condition_entry` raises `ExperimentStoreError` carrying
  `ALPHA_UNITS_REPAIR`; the Swift decoder refuses a condition document with no
  key through `conditionDecodeRefusal` (G4's machinery), carrying
  `ExperimentManifest.alphaUnitsRepairAction`. The two repair strings are
  **independent twin literals** and are asserted equal across the engines
  through the fixture — neither can reword the instruction the other gives.
  Both name **both spellings** of the fix: the manifest key
  (`"alphaInNormUnits": true|false`) and the CLI flag (`--alpha-units
  norm|raw`).
- **`steerlab-cli experiment declare-condition --alpha-units` is now required**,
  baselines included. A slot-less arm carries no α, but it still *stamps* the
  key the other engine reads, and two engines stamping different values for the
  same call is the gap itself.
- **Existing artifacts keep their reading.** `Manifest.from_dict` still reads a
  key-less condition as `False` — the reading every study frozen under it was
  measured with. The Swift engine cannot read such a manifest at all, so the
  reading is engine-dependent; `freeze_advisories` therefore **surfaces an
  advisory** naming the arms, saying which reading this engine gives them and
  that the Mac cannot open the document, rather than converting anything.
- **Not** removed: Swift's in-process `Condition.init(alphaInNormUnits: true)`
  default. It is not a client-facing declaration — a Swift caller states the
  unit in code, and the UI binds it to a control — and the two boundaries a
  client actually reaches both refuse silence.

Both committed fixtures now declare the key explicitly (`true` on each side),
and `manifest-interop.json` additionally carries the server's own refusal text,
its repair, and the preserved legacy reading (`conditionAlphaUnits`).
Pinned by `test_a_new_condition_that_declares_no_alpha_units_is_refused` +
`test_an_existing_keyless_condition_keeps_its_original_reading` ↔
`PortabilityContractTests.theConditionAlphaUnitDeclarationIsExplicitOnBothEngines`,
with the CLI half in
`HeadlessAuthoringTests.declaringAnArmWithoutItsAlphaUnitsIsRefused`.

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

---

## 7. Phase 1b — the `steerlab` client

Phase 0 wrote the contracts; Phase 1a closed the four gaps that stopped a
client-authored study from verifying on the Mac; **Phase 1b is the client
itself** — `Server/steerlab_server/client_cli.py`, console script `steerlab`,
declared beside the untouched `steerlab-server`.

### What it is

The **client** authors the LOCAL workspace it is pointed at. The **engine**
(`steerlab-server`) executes. They are complements: every verb the engine
redirects to the Mac as `macAuthorityVerb` is one the client can now perform
against a local workspace on any platform — and the engine's redirects are
unchanged, because on a cluster node the workspace really is a cache.

The workspace comes from `--root <dir>` or `$STEERLAB_WORKSPACE`, and there is
**no default**. The engine's `paths.project_root()` falls back to the current
directory, which is right for a node started inside its cache and wrong here:
the commonest client mistake is authoring into the source checkout, and a cwd
fallback makes that mistake silent and successful.

**Authoring verbs take no server URL, structurally.** There is no flag on any
of them that could hold one — pinned by iterating the declared verb table
(`test_client_cli.py::test_no_authoring_verb_accepts_a_server_locator`), so a
verb added without a test is still covered. Submitting to a remote runner is
**Phase 2** and is deliberately absent.

### Verb surface (v0)

| family | verbs |
|---|---|
| `experiment` | `create`, `attach`, `declare-condition`, `remove-condition`, `set-protocol`, `pin-revision`, `set-style-taxonomy`, `pin-sae-candidates`, `duplicate`, `verify`, `freeze`, `list` |
| `concept` | `import` |
| `bundle` | `package`, `inspect`, `import` |

Plus `steerlab --version` (package version + the `client` role — one
distribution ships two console scripts, and a caller that got the wrong one has
no other way to tell). Each verb is a thin wrapper over the module that already
holds the guarantee: the `experiment` verbs over `experiment_store` and
`manifest`, `concept import` over `authoring`, the `bundle` verbs over
`bundles`. Every call signature is transcribed from the HTTP route that already
fronts it in `api/routes.py`; every payload key twins the Mac verb's.

Two Phase-1a requirements surface here by construction rather than by
re-implementation:

- `declare-condition --alpha-units norm|raw` is **required**, baselines
  included (G6). When it is absent the client passes the key through as
  *absent* rather than defaulting, so the refusal is
  `experiment_store._condition_entry`'s own and its `repairAction` is the twin
  literal of `ExperimentManifest.alphaUnitsRepairAction`. A client that
  invented its own sentence would be a third independent spelling of a rule
  whose whole value is that there are exactly two, kept equal by test.
- `bundle import --sha256` carries the out-of-band outer pin (G3), verified
  before the archive is opened.

Two setters `experiment_store` exposes are deliberately **not** client verbs:
`replace_draft_manifest` (the server's draft-sync remedy — it installs a
document *into a server's* copy, which is the opposite of what this client
does) and `attach_artifact` (reachable as `attach --artifact`, which is the
spelling the store itself dispatches). The Mac's `pin-prompts`, `pin-rubric`,
`set-instruments` and `set-sweep-selection` are protocol *fields* here,
reachable through `set-protocol --set <key>=<json>`, because that is the shape
`set_protocol` actually has.

### The envelope

The client emits the **same document** the engine does: `cli_envelope` is the
one implementation on this engine, so §4's contracts — the closed header, the
state vocabulary and its exit codes, the advisory vocabulary, the four dispatch
fields — hold on both surfaces without a second literal to keep in sync. Two
client goldens sit beside the engine's in
`Server/tests/fixtures/cli-envelopes/`: `client-experiment-create.json` (ready)
and `client-declare-condition-no-alpha-units.json` (refused — chosen because
its repair is the G6 twin literal, so the committed bytes are a standing
comparison against the Mac's wording). Both are checked by
`test_every_committed_golden_carries_the_fields_a_client_dispatches_on`.

One deliberate difference: the client's exit code is derived from `state` in
**both** output modes. The engine holds its human-mode codes byte-stable
(a refusing `experiment verify` is 1 in human mode, 65 under `--json`) because
`set -e` wrappers depend on them; the client was born speaking the vocabulary
and has nothing to hold still.

### The light-install guarantee

`pip install` of the package **without extras** yields the client: numpy and
safetensors, no torch, no transformers, no FastAPI. The engine's stack moved
behind a new `runner` extra; `all` still contains it verbatim, so
`bootstrap.sh`, `ONBOARDING`, `README`, the committed platform locks
(`update-locks.sh` compiles `--extra all`) and the app's Local Engine flow
(which installs `-r <lock>` then `--no-deps -e Server`) resolve the same
package set they always did.

Guarded out of process, because an in-process assertion about `sys.modules`
passes or fails on test ORDER once another module has imported torch:
`test_importing_the_client_pulls_no_heavy_dependency` asserts that importing
`client_cli` and running `--help` imports nothing third-party at all.
`test_the_authoring_verbs_stay_light_and_verify_is_where_that_ends` extends
that to the verbs that really write — and pins **G7**, the one place the
guarantee currently stops.

### Phase 2

The runner HTTP adapter: submitting a hash-pinned bundle to an engine and
importing the evidence back. Nothing in this module anticipates it — the
authoring verbs have no locator flag to grow into one, and adding a `submit`
family is additive to a table, not a change to these verbs.
