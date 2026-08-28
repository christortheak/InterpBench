# Portability Contracts

**Phase-0 deliverable of the portability program**, extended by **Phase 1a**,
**Phase 1b** (§7), **Phase 2** (§8), **Phase 3** (§9) and **Phase 5** (§10 —
the composite `steerlab run`, which completes the eleven-step round trip this
document opened with), and by the **route-ownership census** (§11 — step 1 of
runner-profile narrowing, which restricts nothing).
Phase 0 changed no production behaviour: its entire output was this page plus
the golden tests it indexes — a record of what the two engines promise each
other, so a later phase that breaks one of those promises fails a test instead
of failing in somebody's workspace. Phase 1a closed four of the recorded gaps —
G1, G3, G6, and G4 alongside G6 — and moved their pins from "pinned as broken"
to "pinned as fixed"; G7, which Phase 1b found and Phase 2 left alone, has
since been closed as engine work. §5 says what was decided in each and why.

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
1a** and are kept below with what was decided and why; **G7 — found in Phase
1b, deferred through Phase 2 — is now closed too**; G2 and G5 remain open.

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

**G7 — The client's light import graph ended at `experiment verify`.
CLOSED.**
Importing `steerlab_server.client_cli` and running `steerlab --help` pulls
*nothing* third-party at all, and `create` / `attach` / `declare-condition` /
`set-protocol` / `duplicate` / `list` / `concept import` pull only what the
client declares (numpy, safetensors). `experiment verify` and `experiment
freeze` — and `bundle package`, which calls `verify` — used to pull **torch**,
through `Manifest.verify` → `experiment.sae_latent` → `steering.sae_latent` →
`steering.injector` → `import torch`. A pure metadata check paid for the whole
execution stack, and it did so for **every** study, including the ones with no
SAE latent condition anywhere in them.

*Why it was not repaired in 1b or 2.* `experiment/sae_latent.py`'s own
docstring says "everything here validates OFFLINE: no SAE weights, no
HuggingFace, no network" — and that was true of what it *does* and false of
what it *imports*. Four names crossed the line (`CLAMP`, `MODES`,
`SAELatentEdit`, `SAELatentFeature`), and the module they came from sits on the
injector stack, so no single lazy import reached it. The repair had to be an
engine refactor — splitting the SAE latent **declared** surface from the
**execution** surface — with cross-engine consequences (the mode vocabulary is
one of the closed key sets). Doing that inside a client phase, to satisfy a
client guard, is exactly the kind of change that breaks an engine nobody was
looking at, so it waited until it could be done as engine work.

*The repair — a schema seam, not a lazy import.* A new module
`steerlab_server/steering/sae_latent_schema.py` holds the SAE latent **declared
surface** and imports `math` and `dataclasses` and nothing else: the mode
vocabulary (`ADD` / `CLAMP` / `MODES`) and the two frozen dataclasses
(`SAELatentFeature`, `SAELatentEdit`) with their range checks. Everything moved
byte-for-byte; no constant, field, docstring or refusal string was reworded on
the way. `steering/sae_latent.py` keeps the execution half — the torch
arithmetic, `SAELatentIntervention`, `group_edits` — imports the schema module,
and **re-exports all five names explicitly** (`__all__`), so every existing
`from .sae_latent import SAELatentEdit` (`steering.plan`, `tasks`, the tests)
resolves to the same objects it always did and no call site changed.
`experiment/sae_latent.py` — the validation module `Manifest.verify` reaches —
now imports `..steering.sae_latent_schema` instead, and is torch-free at import
time end to end; its one execution-path function, `materialize`, already
imported `gemma_scope` and `vector_store` *inside* itself and still does.

*The dependency arrow points execution → schema and never back.* The schema
module imports nothing from the execution module, which is what makes the seam
hold: a future edit that reaches the other way would re-open the gap, and the
guard below would fail out of process rather than quietly.

*Why not lazify `Manifest.verify`'s import instead.* It is already lazy — the
`from . import sae_latent` inside `verify()` was never the problem. The cost
was one module further down, and the only way to reach it without moving code
was to make `condition_violations` stop using the mode vocabulary it validates
against, i.e. to duplicate a closed key set. Two copies of `MODES` is precisely
the failure the vocabulary exists to prevent.

*Consequence, restated:* a bare `pip install steerlab-server` now gives a
client that authors, declares, verifies, freezes and packages — **the whole
authoring lifecycle** — with numpy and safetensors and nothing else. `[runner]`
is needed to *execute*, which is what it was always named for.
Pinned as FIXED by `test_client_cli.py::test_the_whole_authoring_lifecycle_stays_light_including_verify`,
which runs the lifecycle out of process against three fixture studies — one
with no latent conditions, one that **declares an SAE latent condition it never
executes** (the case that used to cost an execution stack), and one whose
latent condition is malformed — and asserts that no step pulls
torch/transformers/fastapi/uvicorn/peft/sae_lens, *and* that the malformed one
is still refused in the same words. A validator that got lighter by getting
weaker would pass an import-set assertion alone.

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
redirects as `macAuthorityVerb` in the `experiment` family is one the client
can perform against a local workspace on any platform — as its own verb, or as
the `set-protocol` field it is a field assignment of. The comparison is
exhaustive and asserted from both tables
(`test_client_cli.py::test_the_client_declares_the_authoring_verbs_the_engine_refuses`),
with each field-reachable verb named and its route checked against the
vocabulary, so a verb added to either table without a decision fails. The one
redirected verb with no client counterpart is `panel compile`, which compiles a
scenario rather than writing through the store.
The engine's redirects are unchanged, because on a cluster node the workspace
really is a cache; what changed is the sentence they carry, which now names the
client's spelling as well as the Mac's (a Linux caller cannot run
`steerlab-cli`, and a repair they cannot run is not one).

**The boundary is the client and the running hardware, not macOS and
everything else.** That is the ruling this section was rewritten under (review
round 11): an engine on compute hardware never authors — its workspace is a
cache — but a client authoring a *local* workspace is as legitimate on Linux
or Windows as on a Mac. `macAuthorityVerb` keeps its name for compatibility
(it is a stable machine code, and agents switch on it); what it means is "this
engine executes, it does not author", not "author on a Mac".

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
| `experiment` | `create`, `attach`, `declare-condition`, `remove-condition`, `set-protocol`, `pin-revision`, `set-style-taxonomy`, `pin-sae-candidates`, `duplicate`, `verify`, `freeze`, `list` (since v0: `detach`, `set-sweep-grid`, `set-parser`, `set-instrument-scope`, `set-system-prompt`) |
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
reachable through `set-protocol --set <key>=<json>` — `taskPromptsFile` +
`taskPromptsHash`, `judgeRubricFile` + `judgeRubricHash`,
`outcomeInstruments`, and `sweep` respectively — because that is the shape
`set_protocol` actually has. The same shape covers the Mac's `set-sampling`
and `set-exclusions`: the generation protocol (`temperature`, `maxTokens`,
`promptMode`, `samplesPerItem`, `seedPolicy`) and the declared
`exclusionRules` are protocol fields here, so a stochastic replication arm
(`--set samplesPerItem=25 --set temperature=0.7 --set maxTokens=1024 --set
seedPolicy=derivedSHA256`) is authorable from any machine. The field
vocabulary is closed
(`experiment_store.PROTOCOL_FIELDS`) and a key outside it **refuses at 64**,
naming the key and listing the vocabulary, with nothing written — never a
silent drop reported as success. An `outcomeInstruments` value outside the
instrument vocabulary refuses the same way the Mac's `set-instruments` does,
and the sampling/exclusion fields carry the same declaration-time gates as
their Mac verbs, with the same sentences (`unknown seedPolicy …`, `unknown
promptMode …`, `samplesPerItem must be ≥ 1 …`, and the exclusion engine's
own violation wording): an out-of-vocabulary value would otherwise be read
by equality tests downstream and silently behave as the default, and a
non-numeric `temperature`/`maxTokens`/`samplesPerItem` would fail the next
manifest decode — bricking every later verb — so both engines refuse it at
the write.

A fourth family, `authoring`, joined for a different reason again. Its one
verb — `authoring prompt <kind>` — renders a generation prompt from the
`prompts/authoring-prompts/` registry and prints it. It is deliberately **not**
in `AUTHORING_FAMILIES` despite the name: an authoring family WRITES into the
workspace, and this one reads a template registry and emits text, because the
emitter of a generation prompt must never be the acceptor of its output. The
rendered bytes are identical to the Mac verb's for the same registry and
arguments (checked by rendering both), and the one deliberate divergence is the
flag that names the output FILE: the Mac verb owns `--out`, this client cannot,
because `--out` is lifted here before the family is chosen, so it is
`--out-file`. Repairs name the binary the caller actually ran — the repair
builders take a program, and both engines' spellings are pinned by test.

**The two measurement declarations are client verbs, and their pins are always
computed.** `set-parser` (the manifest's `numericParser` +
`parserRegistryHash`) and `set-instrument-scope` (its
`outcomeInstrumentScope`) are neither protocol fields nor Mac-only. They were
Mac-only until review round 11, on the reading that authoring is
"Mac-authority"; the ruling above replaced that reading, and these two joined
`attach`, which has always derived `stimulusSetHash` from workspace bytes on
any platform.

Neither is a field assignment, which is why their keys stay **out** of
`PROTOCOL_FIELDS` and `--set numericParser=…` / `--set
outcomeInstrumentScope='{…}'` still refuse at 64 (naming the key, listing the
vocabulary, nothing written). Each *derives* its pin from a workspace file at
the moment of declaration — the parser registry's SHA-256 for the first, the
selected item ids (`itemCount` + `itemIDsHash`) for the second — and both pins
are preregistration facts: *which parser version measured*, and *which rows
were measured*. **No surface anywhere accepts one as input.** There is no
`--registry-hash` and no `--item-ids-hash`, on either engine, and there will
not be: a caller-supplied pin would let a study claim provenance nothing
computed. That guarantee is what the Mac-only carve-out was actually
protecting, and it is untouched — only the *location* of the authoring machine
changed.

`experiment_store.set_numeric_parser` and
`experiment_store.declare_outcome_instrument_scope` mirror
`ExperimentStore.setNumericParser` and
`ExperimentStore.declareOutcomeInstrumentScope` sentence for sentence: the
undefined-parser and malformed-entry refusals are `parser_registry`'s own, the
unknown-`responseFormat` and zero-selection refusals are the Swift twins'
literals, `""` clears both declarations (and, for the scope, clears *before*
the prompts-file guard, so a stale scope is removable in exactly the states —
dropped pin, moved file, drifted bytes — that make clearing necessary), and an
out-of-vocabulary value is **malformed at 64**, never a refusal at 65, because
the caller typed a value the field cannot hold. Repairs name the binary the
caller actually ran, like every other repair builder here, and rendered with
the Mac's program they are the Swift literals byte for byte
(`test_measurement_declarations.py`; Swift side,
`MeasurementDeclarationVerbTests`).

**The study's system prompt is a client verb for a different reason: the
field was reachable and the writer was not findable.** `set-system-prompt`
(the manifest's `systemPrompt` — the deployment frame every arm is read under)
joined both surfaces on 2026-08-28. It is not a derived pin and not a
Mac-authority verb: it is a plain field assignment, `systemPrompt` **stays** in
`PROTOCOL_FIELDS`, and `set-protocol --set systemPrompt=…` keeps working
exactly as it did. What the field discovery showed is that reachability is not
the same as authorability — a persona-carrying replication stalled because
nothing on either surface *named* the thing being written, and running the
study without the persona would have been a different study. So both engines
grew the verb, spelled identically, echoing the same flat `result` keys
(`systemPrompt`, the derived `studyFrameHash`, `delivery`,
`itemsWithOwnSystemTurn`), and the engine's redirect names both spellings like
its siblings'.

The echo carries `delivery` because **what the model receives is
capability-dependent and identical on the two engines**:
`PromptRendering.hasSystemRole` ↔ `prompt_render.has_system_role` decide it,
a genuine system turn where the family's chat template has a system role and
the same text prepended to the first user turn where it does not. The one path
on which a declared frame does not reach the model — a pinned item whose
transcript opens with its own `system` turn, which replaces it for that item —
raises the `systemPromptNotApplied` advisory, whose detail sentence is a twin
literal like every other cross-engine sentence here.

The scope's pin needs the study's task prompts, and the loader of record
(`tasks._load_prompts`) imports torch. `experiment_store.scope_items` is the
torch-free reader that answers the same question — same blank-line handling,
same `prompt-<ordinal>` id fallback, same closed `responseFormat` vocabulary —
held to `tasks._load_prompts` by test on the same file, because a pin computed
under different rules would name a row set the run never selects.

The engine's own *reading* of these fields is unchanged and fully mirrored:
`parser_registry.py` and `response_format.py` carry the twin grammars,
vocabularies (`KNOWN_KINDS`, `KNOWN_RESPONSE_FORMATS`) and refusal sentences,
so a workspace authored anywhere runs identically here. The engine still
redirects both verbs — it executes, it does not author — and the redirect now
names both spellings.

Since v0 the table has gained `detach`, `set-sweep-grid` and the two
measurement declarations above. All four are verbs rather than protocol fields
for the same reason: none is a field assignment.
`detach` audits the whole manifest for declarations that still name a concept
before it removes a pin, and `set-sweep-grid` resolves absolute layers against
the pinned model's depth and refuses a grid no engine could sweep — rules that
live in `experiment_store` and answer identically on the Mac verb, this client,
and the HTTP route (`POST /api/authoring/{name}/sweep-grid`). The sweep block's
*other* half, `sweep.selection`, stays Mac-authority: the criterion is a
preregistration decision and `set-sweep-selection` is where it is made.

**Sharding is execution, so it splits the other way.** Nothing about the
multi-GPU fan-out is authoring — `parallelJobs` never enters the manifest or
its content hash, which is why a sharded run and a single-job run of the same
frozen study are the same measurement — so it belongs to whichever surface
submits, and every surface reaches the same server field. The Mac spells it
`steerlab-cli remote submit-bundle --parallel <n>` (encoded only when `n > 1`,
the executor is `slurm`, and the verb shards, with the envelope echoing
`parallelJobsRequested` / `parallelJobsEncoded` /
`parallelJobsSuppressedBecause`); the Python engine spells it `study submit
--parallel N`, which **refuses** rather than degrades on a non-shardable
request; the HTTP body spells it `parallelJobs`; and this client passes
`--parallel` through on `runner submit` and the composite `run` exactly as the
runner protocol exposes it, inventing no sharding surface of its own (§10.9).
One resolver rules on all of them, so the *rules* cannot drift — only the
answer to a non-shardable request differs, and deliberately. The merge is the
running server's reconciler on every path, and on every path a fan-out can
partially fail while the submit exits `0`, so the shard jobs are verified at
the scheduler, never inferred from an exit code.

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
`test_the_whole_authoring_lifecycle_stays_light_including_verify` extends that
to the verbs that really write.

**The guarantee covers the full authoring lifecycle** — create, attach,
declare-condition, set-protocol, list, duplicate, **verify, freeze, and bundle
package** — for studies with no SAE latent condition *and* for studies that
declare one they never execute. It stops where execution begins, which is what
`[runner]` is for. G7 was the one place it used to stop earlier than that; §5
records how it was closed (a torch-free `steering.sae_latent_schema` under the
validation path) and the guard that keeps it closed.

### Phase 2

The runner HTTP adapter: submitting a hash-pinned bundle to an engine and
bringing the evidence back. Nothing in this module anticipated it — the
authoring verbs had no locator flag to grow into one, and adding a family was
additive to a table, not a change to these verbs. **Delivered: §8.**

---

## 8. Phase 2 — the runner adapter

Phase 1b's client authors a workspace and talks to nothing. Phase 2 is the
other half of the three things the program's target must do (§0, items 2 and
3): **submit a hash-pinned run bundle either engine produced**, and **import an
evidence bundle with verification — and refuse a bad one.**

Two files, one new family:

- `Server/steerlab_server/client/runner.py` — the adapter (a new `client/`
  package; nothing was moved into it).
- `Server/steerlab_server/client_cli.py` — the `runner` verb family, added to
  the declared table beside `experiment` / `concept` / `bundle`.
- `Server/tests/test_client_runner.py` — the pins below.

### 8.1 The adapter surface, and the routes it speaks

Every route already existed. Nothing in this phase asked the engine for a new
endpoint, a `v2` of an old one, or a shape it does not already return: the
payload keys were transcribed out of `api/routes.py` and `api/submissions.py`.
That is Phase 1b's discipline (“every call signature is transcribed from the
HTTP route that already fronts it”) applied one layer out — an adapter that
invents a wire format makes the engine's tests stop being the client's tests.

| adapter method | route | what it sends / gets back |
|---|---|---|
| `info()` | `GET /api/info` | → `service`, `engineVersion`, `root`, `rootLooksLikeSourceCheckout`, `devices`, `loadedModels`, `capabilities`, `controllerChain` |
| `capabilities()` | `GET /api/capabilities` | → the capability snapshot `info()` also embeds |
| `upload_run_bundle(path)` | `POST /api/bundles/upload` | raw octet-stream body + `X-SteerLab-Filename`; → `path`, `filename`, `sha256`, `bytes`, `bundle` (inspected), `executable`, `stagingDirectory` |
| `inspect_remote_bundle(p)` | `POST /api/bundles/inspect` | `{bundlePath}` → the bundle metadata + recomputed `bundleSha256` |
| `submit_uploaded_bundle(…)` | `POST /api/studies/submit-bundle` | `{bundlePath, verb, executor, dryRun, targetRoot, dtype, device, promptsPath, sourcePath, packageEvidence, parallelJobs, force, resources, env}` → `StudySubmission.to_dict()` (`jobId`, `experiment`, `verb`, `executor`, `dryRun`, `runBundle`, `slurmBundle`, `slurmJobID`, `command`, `recordsDirectory`, `submissionDirectory`, `preflight`, `shardJobIDs`) |
| `job(id)` | `GET /api/jobs/{id}` | → `Job.to_dict()` (incl. `status`, `result`, `logTail`) |
| `jobs()` | `GET /api/jobs` | → `{jobs: […]}`, each with a tail |
| `cancel_job(id)` | `POST /api/jobs/{id}/cancel` | → `{ok: true}`; a 502 means `scancel` did not confirm and is **not** swallowed |
| `job_logs(id)` | `GET /api/jobs/{id}` | the `logTail` — bounded, returns at once |
| `job_logs(id, follow=True)` | `GET /api/jobs/{id}/stream` | SSE, consumed until the runner closes it (terminal status, `prepared` included) |
| `download_bundle(…)` | `GET /api/bundles/download?path=…` | streamed to a caller-supplied temp path, hashed as it arrives |

Explicit constructor inputs, and no ambient configuration anywhere in the
class: `base_url`, `token`, `timeout`, `verify` (TLS policy — `True`, a CA
bundle path, or an `SSLContext`), plus an optional `http_client` seam and a
`max_download_bytes` cap. A class that read the environment itself would make
“which credential did this use?” unanswerable at the call site, which is the
wrong property for the object that carries a bearer token.

### 8.2 The two integrity checks

These are why the adapter exists rather than a `curl` in a shell script.

| Contract | Guarantees | Pinned by |
|---|---|---|
| **upload-digest-agreement** | The client hashes the file it sends; the route hashes the bytes it received. A disagreement is refused **before the staged path can be submitted**, naming both digests — a truncated or mangled upload must never become a submitted study, because everything downstream (the run's manifest snapshot, its evidence, its citation) inherits that bundle's identity. | `test_client_runner.py::test_an_upload_the_runner_hashes_differently_is_refused` |
| **submit-identity-precheck** | `submit --bundle-sha` is checked against the runner's own `POST /api/bundles/inspect` of that path **before** anything is submitted; a mismatch creates no job. The extra round trip is deliberate: `inspect` is idempotent and free, `submit` is neither. | `test_submit_refuses_a_staged_path_that_is_not_the_pinned_bundle` (asserts the job count is unchanged) |
| **evidence-outer-hash-pre-download** *(the client half of §3's `evidence-outer-hash-pre-extraction`)* | Downloads stream to a **caller-supplied temp path**, hashing and size-capping as the chunks arrive; the archive moves to its destination only after the digest matches what the runner reported out of band. A mismatch deletes the temp file and leaves the destination non-existent. Refuses rather than overwrites an existing destination. | `test_a_download_whose_digest_disagrees_writes_nothing_to_the_destination`, `test_a_download_over_the_size_limit_stops_mid_stream`, `test_a_download_refuses_to_overwrite_an_existing_destination`, `test_a_download_with_no_expected_digest_is_refused_outright` |
| **download-is-not-import** | The adapter verifies and stops. `runner evidence` prints — and puts in `result.importCommand` and `nextAction` — the exact `steerlab bundle import <path> --sha256 <digest>`, and never runs it. Extraction writes into a workspace; that is a separate, named act. | `test_runner_evidence_verifies_downloads_and_names_the_import_command` |

What the outer digest catches and the per-member checks cannot is
**substitution**: a wholesale swapped archive is internally consistent. That is
the same sentence G3 carries, now true on the client's side of the wire too.

### 8.3 Retries and idempotency

**No retry policy in this phase beyond idempotent GETs, and no automatic retry
at all** — the adapter makes one attempt and reports what happened. What a
caller may safely retry is stated so nobody has to guess:

| call | retry? |
|---|---|
| `info`, `capabilities`, `job`, `jobs`, `job_logs`, `inspect` | plain reads — freely |
| `upload_run_bundle` | **yes.** Each upload lands in its own server-minted staging directory, so a retry costs disk and produces a second path, never a second effect |
| `download_bundle` | **yes.** A GET, and the write is temp → verify → move |
| `cancel_job` | idempotent in effect; a 502 is a real state to report, not one to retry blindly |
| `submit_uploaded_bundle` | **NO. Never automatically.** It creates a job and may spend a scheduler allocation. A timeout on submit is genuinely ambiguous — the job may exist — and the repair is to LOOK (`runner jobs`), not to submit again |

### 8.4 Token discipline

A runner in token mode wants a bearer token (`api/app.py`'s `auth_middleware`;
`/api/bundles/upload`, `/api/bundles/download`, `/api/studies/submit-bundle`
and every mutating route are in the privileged set). The client's rules:

- **Two sources, no third:** `$STEERLAB_RUNNER_TOKEN` or `--token-file <path>`.
- **There is no `--token` flag and there will not be one.** argv is readable by
  every process on a shared login node (`ps`), lands in shell history, and gets
  copied into job records by well-meaning wrappers. Pinned structurally by
  `test_there_is_no_token_flag_on_any_runner_verb`, which iterates the declared
  table: the only spelling containing “token” is `--token-file`, and it holds a
  path.
- **A client variable, not the engine's.** `STEERLAB_RUNNER_TOKEN`, never
  `STEERLAB_AUTH_TOKEN`: a machine that runs both must be able to serve a local
  engine with one secret and reach a remote runner with another, and an adapter
  that borrowed the local server's token would send it to a host nobody
  authorized.
- **Never in the workspace, never in a document.** The only thing any envelope
  says about the credential is the presence boolean `tokenPresent`. The token
  rides an `Authorization` header, never a URL (`test_the_token_travels_only_as_an_authorization_header`),
  and `RunnerClient.__repr__` renders `<present>`/`<absent>`.
- **`scrub` on every raised sentence**, so an upstream string that happened to
  contain the token comes out redacted — belt and braces for the failure nobody
  predicted.
- A world- or group-readable `--token-file` is **announced on stderr and still
  used**, the same call `api/posture.hydrate_token` makes at serve time.

Pinned end to end by
`test_the_token_never_reaches_any_envelope_log_line_or_exception`, which drives
one distinctive fake token through a success envelope, human stdout/stderr, a
refusal document, a real download refusal with a payload, an HTTP 401, an
unreachable runner, and the object's `repr` — and asserts the value appears in
none of them.

### 8.5 The wire, once, for real

`test_the_adapter_works_against_a_real_server_over_tcp_in_token_mode` launches
the genuine `steerlab-server serve` as a subprocess on a loopback ephemeral
port with `STEERLAB_AUTH_MODE=token` and a minted `STEERLAB_AUTH_TOKEN_FILE`,
then runs the adapter against it: info, capabilities, a **401 without the
token**, a chunked upload with the digest agreed across the socket, a dry-run
submit, job get/list, the log tail and the SSE follow (which terminates on its
own, because `prepared` is terminal), and a streamed download verified byte for
byte. Readiness is a polled deadline on `/api/info` with the subprocess watched
for early exit — never a sleep — and teardown terminates then kills in a
`finally`.

Every other test in the file shares a process with the app, which is the right
trade for the logic and cannot answer “does the token really travel as a
header, and does a real HTTP parser accept the chunked body?” This one can.

### 8.6 The light install, extended

`httpx` joined `[project] dependencies` (the client set) and is imported
**lazily, inside the runner verbs** — importing `client_cli` and running
`--help` still pulls nothing third-party at all. It costs the resolution
nothing: httpx and its whole required closure (`httpcore`, `h11`, `anyio`,
`certifi`, `idna`) are already pinned in **both committed platform locks**,
pulled in by `huggingface_hub`, so `uv pip compile --extra all` resolves the
same package set and the locks did not move — asserted directly by
`test_client_cli.py::test_the_new_client_dependency_was_already_in_the_locks`.

`test_the_runner_family_stays_within_the_clients_light_import_set` runs
`runner capabilities` against a dead port out of process and pins the import
set. One wrinkle it records rather than hides: `httpx/__init__.py` reaches for
its optional CLI dependencies if they happen to be installed
(`try: from ._main import main`), so in a dev venv `import httpx` also shows
click/rich/pygments. Those are the `httpx[cli]` extra, which a bare client
install does not resolve — `test_httpx_needs_none_of_its_optional_cli_dependencies`
blocks them on the meta-path and imports httpx anyway rather than taking that
on trust.

**G7 was untouched by this phase.** The runner family does not reach
`Manifest.verify`, so it never pulled torch either way. `experiment verify` /
`freeze` still did at the time, for the reason §5 gives; they no longer do —
G7 was closed afterwards, as engine work, and §5 records it.

### 8.7 What Phase 2 deliberately did NOT do

- **No composite `run --wait`.** Upload → submit → poll → download → import is
  six explicit acts here, and each one is separately refusable. Orchestrating
  them into one verb — with the waiting policy, the resume-after-disconnect
  question, and the decision about when a client may import on its own — is
  **Phase 5**, and it wants the pieces below it to be boring first.
- **No runner of its own.** Phase 2's client can talk to a runner and cannot
  *be* one; every test that needed a live engine launched `steerlab-server`
  itself. **Delivered: §9.**
- **No auto-import.** See `download-is-not-import` above.
- **No new or versioned endpoints, and no server behaviour change.** The engine
  and `steerlab-server` are byte-identical after this phase.
- **No authoring verb learned a locator.** `runner` is a separate family
  precisely so §7's structural contract stays a line in a table rather than a
  judgement call about a flag name — `runner submit` legitimately declares
  `--executor`, and no authoring verb ever may
  (`test_no_authoring_verb_accepts_a_server_locator`, now iterated over
  `client_cli.AUTHORING_FAMILIES`).

### 8.8 One repair made in passing

`--out` was declared by `bundle package` **and** intercepted unconditionally as
the envelope's destination by `client_cli.parse`, so `bundle package --out
foo.tar.gz` silently wrote the *document* to that path and packaged the archive
to its default location. Phase 2 needed `runner evidence --out <file>` to mean
the file, so the parse now lets a verb's **declaration win** over the global
flag. Those two verbs' envelopes still travel on stdout under `--json`; they
simply have no second spelling for “write the document to a file”. Pinned by
`test_client_runner.py::test_out_belongs_to_the_verb_that_declares_it`.

---

## 9. Phase 3 — localhost as a MANAGED runner

Phase 2 gave the client a runner to talk to and no way to have one. Phase 3 is
the runner a person has on the machine in front of them:

```bash
steerlab runner serve                    # loopback, token mode, a root of its own
```

One new verb, one new test file, and **no change to the engine**: what
`runner serve` starts is `python -m steerlab_server.cli serve` — the same entry
point an operator types, with no argument that did not already exist.

- `Server/steerlab_server/client_cli.py` — the `runner serve` verb and its
  helpers (`default_runner_root`, `_runner_environment`, `_mint_runner_token`,
  `_claim_port`, `_await_engine`).
- `Server/tests/test_local_runner.py` — the acceptance test below.

### 9.1 The ruling: two service roles

**The bundle protocol binds BATCH execution.** Local and remote runners are
reached the IDENTICAL way — upload → submit → evidence → import, every hop
hash-pinned — and there is **no privileged localhost path into the client
workspace**. A managed runner therefore gets a **runner-owned root**, and
`STEERLAB_ROOT` is never allowed to name the workspace the client authors.

The app's local **workbench** — interactive serving of a live workspace, which
is what the Mac app has always done — is the *other* service role, and this
phase does not touch it. It serves the workspace on purpose; it is not a
runner, and nothing about it is batch execution.

The reason the rule is worth a refusal rather than a convention: a runner
rooted in the workspace could read and write that tree directly, which no
remote runner can do. A study that "worked" against such a runner would prove
nothing about one that has to travel — and the round trip everything else in
this document pins would become optional in practice, which is how it stops
being tested.

| Contract | Guarantees | Pinned by |
|---|---|---|
| **runner-root-is-not-the-workspace** | `runner serve --runner-root <the workspace>` is a typed refusal (`runnerRootIsWorkspace`, 65) naming the rule and the repair — for the workspace named by `--root` **or** `$STEERLAB_WORKSPACE`, and for nesting in **either** direction. Containment, not string prefix: `/tmp/ws-runner` is a legal root beside `/tmp/ws`. Nothing is created before it refuses. | `test_local_runner.py::test_serving_the_client_workspace_as_the_runner_root_is_refused`, `::test_a_sibling_of_the_workspace_is_a_legal_runner_root` |
| **the-engine-root-is-the-runner-root** | The engine's own `GET /api/info` reports the runner root, asked over the wire rather than assumed. | `::test_the_whole_round_trip_runs_against_a_managed_local_runner` (assertion (c)) |
| **the-runner-root-is-disposable** | After the round trip, **deleting the entire runner root removes nothing the client workspace holds** — the imported run's bytes are re-hashed with the runner root gone. What the runner keeps is a cache; what the workspace keeps came home through `bundle import`. | same test, assertion (b) |
| **no-ambient-redirection** | The child engine's environment is BUILT, not inherited: `STEERLAB_ROOT`, `STEERLAB_RUN_ROOT`, `STEERLAB_METADATA_ROOT` and `STEERLAB_JOBS_DB` are set to runner-root paths, `STEERLAB_AUTH_MODE=token` + `STEERLAB_AUTH_TOKEN_FILE` are declared, `STEERLAB_EXECUTOR=local`, `STEERLAB_BIND=127.0.0.1`, and `STEERLAB_AUTH_TOKEN` / `STEERLAB_DEV_OPEN_LOOPBACK` are **removed**. A shell that exports `STEERLAB_ROOT` — the commonest thing a SteerLab user has in theirs — cannot reach through the verb and put the runner's staging in the workspace it names. | `::test_the_engine_environment_is_built_rather_than_inherited` |

### 9.2 The runner root, per platform

Derived by a small internal helper (`client_cli.default_runner_root`), not a
new dependency: the whole need is three `os.path.join` calls, and
`platformdirs` in the **client's** dependency set would have to be justified to
every packager for the rest of the project's life.

| platform | default runner root |
|---|---|
| macOS | `~/Library/Application Support/SteerLab/local-runner` |
| Linux / BSD | `$XDG_DATA_HOME/steerlab/local-runner` (default `~/.local/share/steerlab/local-runner`) |
| Windows | `%LOCALAPPDATA%\SteerLab\local-runner` |

**Platform ruling: Windows is client-only.** Authoring, freezing, packaging,
remote submission and evidence import are supported there; **local execution is
not**, and `runner serve` refuses on Windows rather than starting an engine on
an unsupported execution path. The Windows row above is therefore what
`default_runner_root` computes, not a supported configuration — a Windows
author points `--runner <url>` at a runner on macOS, Linux, or a cluster, and
every hash-pinned hop in this document is unchanged.

`--runner-root <dir>` overrides it. What is deliberately **not** in that table:
the current directory, `$STEERLAB_WORKSPACE`, and `$STEERLAB_ROOT` — the value
of the default is that a person who types `steerlab runner serve` while
standing in their workspace does not thereby serve it.

Inside, all of it runner-owned and all of it disposable:

```
<runner-root>/
  runner.token        # 0600, minted here — never the engine's ~/.steerlab-token
  prompts/  experiments/  runs/     # the artifact tree (STEERLAB_ROOT)
  .steerlab/                        # metadata root: the durable job database
```

`runs/` is where uploads stage (`runs/<stamp>-uploaded-bundle/`), where
submissions record themselves (`runs/<stamp>-submit-bundle-…/`), and where
evidence is packaged. `prompts/` and `experiments/` exist so the engine's
serve-time "artifact root has no prompts or experiments" warning does not fire
on a root that is legitimately empty.

**Token discipline, from the other side of the wire.** Token mode, always, and
the token is *minted* rather than presented: a 0600 file under the runner root,
created `O_EXCL` at that mode rather than chmod'ed afterwards, reused across
restarts (restarting a runner must not invalidate the credential already in
someone's script), and announced-but-still-used when its mode is loose — the
same call `api/posture.hydrate_token` makes. The **value is never printed**,
not on stdout, not on stderr, not in the envelope; the *path* is, together with
the exact `--token-file` invocation that uses it. `runner serve` declares no
token flag of any spelling, and §8.4's reason has not changed because the
secret is now ours: argv is public. Pinned by
`::test_a_reused_token_file_is_kept_and_a_loose_one_is_announced` and by the
secret-absence assertions in the round-trip test.

### 9.3 Subprocess, not an in-process serve

The mechanics decision, recorded because the alternative looks cheaper:

1. The engine's `cli._serve` resolves the WP-S posture by **mutating the
   process environment** — it exports `STEERLAB_AUTH_MODE` and hydrates
   `STEERLAB_AUTH_TOKEN` — and then reads its artifact root from
   `STEERLAB_ROOT`. In the client process `STEERLAB_ROOT` may already name the
   **workspace** (`resolve_workspace` exports it), so an in-process serve would
   have to unset and re-set the very variable the two-roles rule turns on. A
   child gets an environment built for it, and the rule holds by construction
   rather than by discipline.
2. An in-process serve would hydrate a **bearer token into the client's own
   environment**, where every later verb in that shell inherits it.
3. `uvicorn.run` installs signal handlers and does not return — there would be
   no process of our own left to stop cleanly.
4. It would import fastapi/uvicorn into the client process, and the
   light-import contract (§7, §8.6) is measured on exactly that.

So: `subprocess.Popen([sys.executable, "-m", "steerlab_server.cli", "serve",
"--host", "127.0.0.1", "--port", str(port)])`, with the engine's output pumped
line-by-line onto **stderr** by a thread. (A pipe and a pump rather than
handing the child our stderr directly: under `--json` the client's `sys.stdout`
is a Python-level swap, not a `dup2`, so a child inheriting fd 1 would write
log lines into the one stream that must carry exactly one JSON value.)

Everything that can refuse happens before anything that can write:

- **light install** — `runner serve` on a client-only install refuses by name
  (`runnerExtraMissing`, 65) with `pip install 'steerlab-server[runner]'`, and
  probes with `importlib.util.find_spec`, which resolves a name **without
  executing the module** — the light-install guarantee is not spent to report
  that it holds (`::test_a_light_install_is_refused_by_name_before_anything_starts`);
- **port** — `--port` is bind-tested first, so a collision is a typed refusal
  (`runnerPortUnavailable`, 65) naming the port and offering `runner
  capabilities` against it, rather than a uvicorn traceback from a child that
  dies half a second after the parent claimed success. No `--port` means an
  ephemeral one, picked and printed
  (`::test_a_second_runner_on_the_same_port_refuses_rather_than_serving_quietly`);
- **readiness** — a polled deadline on `GET /api/info` (stdlib `http.client`,
  not even httpx), with the subprocess watched for early exit; **any** status
  counts, 401 included, because in token mode an unauthenticated `/api/info` is
  *supposed* to be refused and the question is only whether an engine is
  answering. A child that exits first becomes `runnerStartFailed` (70) naming
  what to read.

**Stopping.** SIGINT *and* SIGTERM are handled, and both stop the child before
this process leaves: a shell puts a background job's SIGINT at `SIG_IGN` (which
CPython inherits, so `KeyboardInterrupt` would never arrive), and a plain
SIGTERM would kill the client outright and leave the engine listening with
nobody to stop it. One line on stderr on the way out, naming how long it served
and that the runner root was kept and no workspace was touched.

### 9.4 `--json`: one startup envelope, then a stream

Every other client verb writes exactly one document when it **finishes**. This
one finishes only when it is stopped, so:

> **`runner serve --json` emits ONE envelope on stdout the moment the engine is
> ready, and then serves, with every further line on stderr.**

An agent that had to wait for the process to die to learn the URL of the runner
it just started would have no use for the document at all. The envelope is an
ordinary §4 document — `state: ready`, sorted keys, a `nextAction` — whose
`result` carries `url`, `host`, `port`, `runnerRoot`, `runnerRootIsDefault`,
`tokenFilePresent`, `tokenFile`, `tokenFileMinted`, `authMode`, `executor`,
`runsDirectory`, `metadataRoot`, `enginePID`. It is flushed explicitly: a
buffered stdout on a pipe would hold the document until process exit, which is
never, on purpose. Mechanically the verb raises `client_cli.ServeCompleted`
when it stops, caught in `main` above the blanket handler, so no second
document can be written for the same invocation.

Pinned by `::test_the_document_is_a_startup_envelope_and_then_a_stream`, which
asserts stdout parses as exactly one value and that the engine's own artifact-
root banner and the verb's human banner are on **stderr**.

### 9.5 The acceptance test — the point of the phase

`test_local_runner.py::test_the_whole_round_trip_runs_against_a_managed_local_runner`
launches the verb the way an operator does (a subprocess of the real module, an
ephemeral port, an environment scrubbed of every `STEERLAB_*` that could point
it elsewhere), waits on the startup envelope and then on a polled `/api/info`
deadline — no sleeps anywhere — and drives the whole path with client verbs
against a temp **client** workspace:

author (`concept` files + `experiment create/attach/declare-condition/verify/
freeze --force`) → `bundle package` → `runner upload` (the digest the client
computed IS the one the runner read) → `runner submit --verb verify --executor
local` (a **real** execution: the only bundled verb that needs no model) →
poll `runner jobs <id>` to terminal → `runner evidence --out` (downloaded,
outer digest verified, deliberately **not** imported) → `bundle import
--sha256` into the client workspace.

Then the three assertions the phase exists for: **(a)** the runner root holds
the staging and the cache (`uploaded-bundle`, `submit-bundle`, `jobs.sqlite`,
the imported study) and the client workspace gained exactly one thing — the
imported run — with no staging directory, no job database and no token in it;
**(b)** the runner is stopped and its root deleted wholesale, after which the
imported run's bytes still hash to what they hashed before; **(c)** the
engine's `/api/info` root is the runner root and is not the workspace.

**What is real and what is not**, stated rather than implied: the runner, the
routes, the socket, the archive, the digests, the study, the submission and the
import are all genuine. The one thing skipped is the GPU compute that would
have produced a run directory — `verify` is the only model-free bundled verb
and it writes no run directory, so it packages no evidence — so the evidence
half is driven over a job **seeded into the runner's own durable job store
before it boots**, carrying an archive `bundles.package_evidence` really wrote
into the runner's runs root. That is the same allowance
`test_client_runner.py::_evidence_bearing_job` makes, one process further out.

### 9.6 What Phase 3 deliberately did NOT do

- **No daemon management.** `runner serve` is foreground, v1. No start/stop/
  status, no pidfile, no launchd/systemd unit, no auto-restart. Every one of
  those is a promise about a process nobody is watching, and the failure mode
  is a forgotten runner holding a port and a token.
- **No `--host`.** Binding a managed runner to the network is the engine's
  decision to make through `steerlab-server serve`, where the posture refusals
  that gate a non-loopback bind are written and tested (`api/posture`).
- **No engine change of any kind.** `steerlab-server`'s behaviour is
  byte-identical after this phase; the verb chooses the root, mints the token,
  picks the port, and waits.
- **No composite `run --wait`, still.** The round trip is still six explicit,
  separately refusable acts, and having a local runner makes that *more*
  valuable, not less: it is now cheap to exercise each one. Orchestrating them
  — the waiting policy, resume-after-disconnect, and when a client may import
  on its own — remains **Phase 5**, which wants exactly the pieces this phase
  made boring.

---

## 10. Phase 5 — the composite `steerlab run`

Phase 2 said it out loud (§8.7) and Phase 3 said it again (§9.6): **no
composite `run --wait`, not yet, because it wants the pieces below it to be
boring first.** They are. Phase 5 is the one command:

```bash
steerlab run <experiment> --runner <url>          # --wait is the DEFAULT
```

A frozen study in this workspace becomes verified evidence in this workspace.
The verb **composes** — every step below is a call into a piece that already
exists and is already separately tested, and every refusal a caller can meet
is one of theirs. Two files:

- `Server/steerlab_server/client_cli.py` — the `run` family and its machine
  (`_run`, `_run_machine`, `_run_wire`, `_poll_to_terminal`,
  `_refuse_unexecutable`, `_write_provenance`).
- `Server/tests/test_run_orchestration.py` — the pins below.

### 10.1 `run` is a SOLO family

`steerlab run <experiment>`, never `steerlab run run <experiment>`. The verb
table grew one entry whose family name *is* its verb (`SOLO_FAMILIES`), and
the synopsis, the help page, every repair sentence and the envelope's `verb`
all name the command a person types. Every other family is unchanged: the verb
word is still consumed as a verb word.

It is **not** in `AUTHORING_FAMILIES`, for §8.7's reason one level up. It does
author one thing — it *imports evidence* — which is exactly why it may not be
in the authoring set: an authoring verb may never declare a locator, and this
one must. `test_client_cli.py::test_no_authoring_verb_accepts_a_server_locator`
now asserts the excluded families are **exactly** `{runner, run}`, so a third
family cannot opt itself out quietly.

### 10.2 The stages, and what each may do

`result.stages` carries **one row per stage, always all nine**, each with a
`stage`, a `state`, and the facts the next stage needs. Absence is never the
signal: a stage that never happened says `notReached`, because "we never got
there" and "we forgot to record it" must not be the same observation. Human
mode streams one `run[<stage>]: …` line per transition to **stderr** (in both
modes — under `--json` stdout carries exactly one document).

| # | stage | what it does | what it composes | refuses with |
|---|---|---|---|---|
| 1 | `load` | load the local study, check it is **frozen**, re-verify every pin | `Manifest.load`, `experiment_store.load_raw`, `Manifest.verify` | `experimentNotFrozen` (65); `pinDrift` (65, **the store's own gate**, `error.gate` present) |
| 2 | `package` | package the run bundle here, record its sha256 | `bundles.package_experiment` | `bundleRefused` (65) |
| 3 | `capabilities` | ask the runner who it is and whether it can execute **this verb on this executor** | `RunnerClient.info` / `.capabilities` / `.identity` | `runnerCannotExecute` (65), naming what the runner offers |
| 4 | `upload` | stage the archive; the cross-socket digest agreement is the adapter's | `RunnerClient.upload_run_bundle` | `uploadDigestMismatch` (65) and the adapter's HTTP vocabulary |
| 5 | `submit` | submit with an explicit `--verb`, `--executor` and the submit endpoint's pass-throughs | `RunnerClient.submit_uploaded_bundle` (which pre-checks `--bundle-sha` against the runner's own inspect) | `bundleDigestMismatch` (65); `submitOutcomeUnknown` (70) |
| 6 | `wait` | poll to a terminal status | `RunnerClient.job` | `waitDeadlineExceeded` (70); `remoteJobFailed` (70, after stages 7–9) |
| 7 | `evidence` | fetch the **strongest evidence available** and verify its outer digest | `RunnerClient.evidence_reference`, `.download_bundle` | `evidenceNotPackaged` (65); `evidenceDigestMismatch` (65) |
| 8 | `import` | verify-and-extract into this workspace with the out-of-band pin | `bundles.import_bundle(expected_sha256=…)` | `bundleRefused` (65) |
| 9 | `provenance` | stamp the imported run, additively | `_write_provenance` | never — it skips and says why |

A failure mid-machine is a **typed envelope naming the stage it died in**:
`result.failedStage` plus the stage table, on top of whatever typed refusal the
underlying module produced. The translation itself is unchanged — a
`BundleError` is still `bundleRefused`, a mistyped study is still 66 — the
composite only adds *where*.

**Stage states** are a smaller, separate vocabulary from the envelope's
(`ok`, `skipped`, `refused`, `failed`, `notReached`), deliberately, so nobody
switches on them as if they were the twelve.

### 10.3 Refuse before you spend

Stage 3 exists so that an unexecutable submission never becomes staged disk on
somebody else's machine, and never becomes a scheduler allocation nobody gets
back. Three checks, all against what the runner **reports**:

- an unknown **study verb** — the client's `RUN_STUDY_VERBS` is a twin literal
  of `api/submissions.VALID_STUDY_VERBS` (a literal, not an import, because
  importing the route module pulls FastAPI into the light client; and not a
  read of the runner's `availableJobTypes`, because that is the *job-type*
  roster and does not contain `verify` — refusing the one model-free verb would
  be worse than not checking);
- an unknown **executor** — twin of `api/profile._VALID_EXECUTORS`;
- `--executor slurm` against a runner whose `schedulerMode` is not `slurm` —
  the submit route's own rule, read one hop earlier. `--dry-run` is exempt,
  because a dry run schedules nothing and the route accepts it anywhere; a
  precheck stricter than the rule it anticipates would refuse work that would
  have succeeded.

Every one of these refusals carries `result.runnerOffers` — `schedulerMode`,
`serverRole`, `availableJobTypes`, `engineVersion`, and the study-verb roster.
"Unsupported" with no list beside it sends the reader to a different machine to
find out what they should have typed.

| Contract | Guarantees | Pinned by |
|---|---|---|
| **capability-refusal-precedes-upload** | An unsupported verb/executor combination is refused with **nothing uploaded** — the fake adapter saw exactly one `GET /api/info` and zero uploads and zero submits. | `test_a_capability_refusal_happens_before_any_upload`, `test_an_unknown_study_verb_is_refused_before_any_upload`, `test_a_dry_run_may_name_slurm_on_any_runner` |
| **unfrozen-refuses-locally** | A draft refuses at stage 1 with the runner never addressed at all (`script.info_calls == 0`). | `test_an_unfrozen_study_is_refused_before_the_runner_is_addressed` |
| **drift-keeps-the-stores-gate** | A drifted pin refuses as `pinDrift` with `error.gate` present — the vocabulary `experiment verify` already answers in, not a composite's paraphrase. | `test_a_drifted_pin_is_refused_with_the_stores_own_gate` |

### 10.4 The retry table

| call | retried? | why |
|---|---|---|
| `GET /api/info`, `GET /api/capabilities`, `GET /api/jobs/{id}` | freely (they are the poll) | plain reads |
| **upload** | **yes — one automatic retry on a transport error** | each upload lands in its own server-minted staging directory, so a retry costs disk and produces a second path, never a second effect |
| **evidence download** | **yes — one automatic retry on a transport error** | a GET whose write is temp → verify → move; a half-file never reaches the destination |
| **submit** | **NEVER, not once, not ever automatically** | it creates a job and on Slurm spends an allocation. A transport failure here is genuinely ambiguous — the job may exist — so the machine stops with `submitOutcomeUnknown` (70) and the repair is to **LOOK**: `steerlab runner jobs --runner <url>`, with the `runner submit` line that resumes from the already-staged bundle printed beside it |
| `POST /api/jobs/{id}/cancel` | **never called by this verb at all** | see below |

Pinned by `test_a_submit_transport_error_is_never_retried_and_says_to_look`
(exactly one submit attempt, no poll, no download, no cancel) and
`test_an_upload_transport_error_is_retried_once`.

### 10.5 Waiting, and the two ways to stop

`--wait` is the **default** — orchestrating the wait is the point of the verb.
Polling is one cheap `GET /api/jobs/{id}` on a fixed interval with a gentle
backoff: 1 s, ×1.5, capped at 30 s, so a ten-second job stays responsive and an
eight-hour job is not asked twenty-eight thousand times.

**`--timeout <seconds>` on this verb is the WAIT deadline** (default 24 h), and
that is a different meaning from `--timeout` on the `runner` family, where it
is the per-request budget. The divergence is deliberate and both spellings are
present: a caller of a composite is thinking about how long the whole thing may
take, and the per-request budget keeps its own name here — `--request-timeout`
— so neither meaning has to be guessed from context.

**Nothing in this verb ever cancels a remote job.** Both ways of stopping
detach:

- **SIGINT during the wait** — the handler is installed for the duration of the
  wait only and restored afterwards; it sets a flag, the poll loop returns, and
  the verb answers `pending` (12) with `outcome: "interrupted"`,
  `cancelled: false`, and the three follow-up commands. A composite that killed
  a running cluster job because somebody pressed ctrl-c in a terminal would be
  destroying work it did not do and cannot give back. (The sleep is sliced into
  0.1 s steps: PEP 475 makes `time.sleep` *resume* after a handled signal, so a
  single 30-second sleep would swallow the interrupt for half a minute.)
- **`--timeout` expiry** — `waitDeadlineExceeded` (70), with `cancelled: false`
  in the result and "the job was NOT cancelled and is still on the runner" in
  the reason. The deadline bounds this client's patience, never the runner's
  work.

`--no-wait` detaches immediately after submit, answering `pending` (12) with
`outcome: "detached"` — the same state and the same exit code the engine's own
`study submit` already answers for asynchronous work in flight. `pending` is a
**success document**: no `error`, and `result.followUps` carries the three
commands that finish the job by hand, runnable as printed:

```
steerlab runner jobs <id> --runner <url>
steerlab runner evidence <id> --out <file.tar.gz> --runner <url>
steerlab bundle import <file.tar.gz> --sha256 <digest>
```

Pinned by `test_no_wait_detaches_and_prints_the_exact_follow_ups`,
`test_an_interrupt_during_the_wait_detaches_rather_than_cancelling` (which
raises a real SIGINT into the process and then asserts the handler was
restored) and `test_the_wait_deadline_stops_watching_and_never_cancels`.

### 10.6 The strongest-evidence rule

> **On a terminal state — success OR failure — fetch the strongest evidence the
> job has.**

A failed job that packaged partial evidence still gets its bundle downloaded,
verified and imported, and the job's failure is **loud** in the result:
`remoteJobFailed` (70), the runner's own error sentence in the reason, and a
repair that says a partial is evidence about a *failure*, never a result.
"The data still exists somewhere under /scratch" is not a retention policy —
the same sentence `bundles` already carries on the engine's side of the wire.

The other outcomes, all typed:

| situation | answer |
|---|---|
| job succeeded, evidence imported | `ready` (0), `outcome: "succeeded"` |
| job succeeded, **packaged none** | `evidenceNotPackaged` (65), `outcome: "noEvidence"` — the repair names why (`verify` writes no run directory by design; a dry run runs nothing) |
| job failed, evidence imported | `remoteJobFailed` (70), `imported: true`, stamp written |
| job failed, packaged none | `remoteJobFailed` (70), `imported: false` |
| `--no-evidence` / `--dry-run` | stages 7–9 `skipped`, with the flag as the `reason`; **not** a refusal — the caller asked for exactly this |
| download digest disagrees | the adapter's `evidenceDigestMismatch` (65); `import` and `provenance` are `notReached`, and nothing is written to the workspace |

The archive itself is a **courier, not an artifact**: without `--evidence-out`
it is downloaded into a temp directory of the verb's own, imported, and the
directory removed. A stray `.tar.gz` left in the workspace's `runs/` would look
like evidence that had not been imported. `--evidence-out <file>` keeps it, at
that path, and `--max-bytes` is the adapter's size cap.

Pinned by `test_a_failed_job_with_partial_evidence_still_comes_home`,
`test_a_terminal_job_with_no_evidence_reports_a_typed_outcome`,
`test_no_evidence_requested_is_a_skip_and_not_a_refusal` and
`test_a_download_whose_digest_disagrees_imports_nothing`.

### 10.7 Provenance, without rewriting the frozen study

The imported run's bytes cannot say **which** engine produced them, from
**which** bundle, under **which** job. `run` answers that with a small JSON
document written into the imported run directory at import time —
`runs/<runID>/remote-execution.json`.

**Where the shape comes from.** This is the house pattern, not a new one:
`adjudication.STAMP_FILENAME` (`adjudicated-endpoint.json`) is a named-constant
JSON stamp written into the run directory it describes, carrying a prose `NOTE`
that explains its own standing; and `bundles.import_bundle` already writes
`pipeline-portable.json` into an imported pipeline run as part of the landing
write. This is the same act, one caller out.

**What it is NOT allowed to touch.** Nothing under `experiments/`. Not the
manifest, not `pinned/`, not the freeze hash, not any content hash. The record
is additive and client-side, and that is checkable rather than believable:
`test_the_provenance_stamp_leaves_the_frozen_study_untouched` reads the entire
`experiments/` tree byte-for-byte before and after and asserts equality, and
the happy-path test re-asserts the frozen manifest's bytes on its own.

```jsonc
{
  "schemaVersion": 1,
  "kind": "remoteExecution",
  "note": "Remote execution provenance. …",     // prose, in the file itself
  "client":     { "program": "steerlab", "role": "client",
                  "version": "…", "engine": "python-hf-transformers" },
  "experiment": "<name>",
  "manifest":   { "status": "frozen", "contentHash": "<64 hex>",
                  "freezeHash": "<64 hex|null>", "freezeForced": false,
                  "modelID": "…", "modelRevision": "…" },
  "runner":     { "url": "https://…", "service": "steerlab-server",
                  "engineVersion": "steerlab-server <v>+<sha8>",
                  "root": "…", "schedulerMode": "local|slurm",
                  "serverRole": "…", "tokenPresent": true },
  "runBundle":  { "path": "<local>", "sha256": "<64 hex>",
                  "runnerPath": "<staged>", "bytes": 0, "entries": 0,
                  "experimentContentHash": "<64 hex>" },
  "job":        { "id": "…", "verb": "run", "executor": "local",
                  "status": "succeeded", "dryRun": false,
                  "slurmJobID": null, "shardJobIDs": [],
                  "error": null, "runDirectory": "<runner-side>" },
  "evidence":   { "sha256": "<64 hex>", "bytes": 0, "verified": true,
                  "runnerPath": "…", "archivePath": null,
                  "archiveRetained": false },
  "outcome":    "succeeded | jobFailed",
  "timestamps": { "startedAt": "…Z", "submittedAt": "…Z",
                  "terminalAt": "…Z", "importedAt": "…Z" }
}
```

Three properties worth naming:

- **Full digests, never elided.** An abbreviated hash in a provenance record is
  a provenance hole.
- **This client's own clock.** Every timestamp is an observation this process
  made, in the envelope's ISO-8601. Two machines' wall clocks disagree, and a
  record that mixed them would make its own durations meaningless.
- **`tokenPresent`, never the token.** §8.4's rule extended to the one durable
  surface this phase added — and the surface most likely to leak, because it is
  written to disk and travels with the run.
  `test_the_token_reaches_no_document_no_stamp_and_no_log_line` drives one
  distinctive fake token through a whole successful machine and then greps
  **every file in the workspace** for it.

Written with `O_EXCL`: a run's provenance is part of its landing write and is
never rewritten. In practice the importer refuses an existing member long
before this matters, which is exactly why the guard is cheap to keep. No run
directory to stamp (an evidence bundle that named no `runID`) is a `skipped`
stage with the reason, not an error.

### 10.8 The tests, and the one allowance

**Unit tier** — `test_run_orchestration.py`, an injected `FakeRunner` over
`runner_api.RunnerClient`, no sockets. Real parsing, real envelope, real
workspace, real `bundles`; only the wire is fake. It deliberately does **not**
re-pin the adapter's own integrity checks (those are `test_client_runner.py`'s,
over the genuine routes) — when a test wants "the runner hashed it
differently", the script raises the adapter's own `RunnerRefusal` with the
adapter's own code, and what is asserted is what the *machine* then does, and
above all **what never happened**.

**End-to-end tier** — the Phase-3 managed-runner harness, imported rather than
re-written. `test_the_whole_machine_runs_against_a_managed_runner` drives
`steerlab run <exp> --runner <url> --verb verify --executor local` through argv
against the genuine served engine over a real loopback socket in token mode:
stages 1–6 all `ok`, the job really `succeeded`, the runner's `/api/info` root
really is the runner root and not the workspace, the staged upload really is
inside the runner root, and the client workspace gained nothing but the run
bundle it packaged — no staging directory, no job database, no token.

**The allowance, stated rather than implied.** It ends at a *typed*
`evidenceNotPackaged`, and that is the honest outcome rather than a shortfall:
`verify` is the only bundled verb that needs no model, and it writes no run
directory, so it has no evidence to package
(`bundles._execute_run_bundle_inner`). The import-and-stamp half therefore gets
its own end-to-end test against the **same** runner —
`test_evidence_comes_home_verified_and_stamped_over_a_real_socket` — over the
job Phase 3 already seeds into the runner's durable store, carrying an archive
`bundles.package_evidence` really wrote. It enters at the stage boundary rather
than through argv, and everything it touches is the production path: the HTTP
download route, the outer-digest verification, the importer, the stamp writer.
It closes with Phase 3's two-roles assertions repeated for the composite's
landing write — the runner is stopped, its root deleted wholesale, and the
imported run's bytes and its provenance stamp are still here and still hash to
what they hashed before.

**Import guard** — `test_the_run_verb_stays_within_the_clients_light_import_set`
runs the whole machine out of process against a dead port (author → freeze →
`run` → transport failure, exit 70) and pins the import set: nothing heavy, and
nothing third-party outside the union of Phase 1b's authoring set and Phase 2's
httpx closure. The composite composes **light** pieces plus the adapter; if
this fails, the repair is to move the offending import inside the code that
needs it, never to weaken the guard.

### 10.9 What this completes, and what it deliberately did NOT do

This closes the eleven-step round trip this document opened with — author,
freeze, package, address, upload, submit, watch, fetch, verify, import, record
— as one command, with every hop still hash-pinned and every hop still
separately refusable. §8.7's and §9.6's standing "no composite `run --wait`,
still" is now spent.

- **No engine change of any kind.** `steerlab-server`'s behaviour is
  byte-identical after this phase; no route was added, versioned, or altered,
  and every payload key is transcribed from one that already existed. The only
  files that moved are the client's.
- **No auto-cancel, ever.** §10.5. There is no `--cancel-on-timeout` and there
  will not be one.
- **No resume-after-disconnect.** `--no-wait` and the detach paths print the
  three commands that re-attach by hand; the verb has no `--attach <job-id>` of
  its own. Re-entering a machine at stage 6 means answering what happens when
  the local bundle no longer matches the job's, which is a question this phase
  did not need to answer to be useful.
- **No sharding surface beyond `--parallel`.** The pass-throughs are exactly
  the ones `runner submit` already exposes; the composite invented none.
- **No second envelope.** `run` writes exactly one document, like every client
  verb except `runner serve` (§9.4) — the stage lines are diagnostics.

---

## 11. The route-ownership census — runner-profile narrowing, step 1

§9.1 made the ruling: there are **two service roles**. A **runner** executes
batches reached through the bundle protocol and owns a disposable root; a
**workbench** serves a live, authored workspace interactively. What the ruling
did not have was a list — one FastAPI app serves both roles, and nothing said
which of its routes belongs to which.

This step writes that list down. **It restricts nothing.**

- `Server/tests/route_roles.py` — **the census**. Every HTTP route the app
  exposes, with a declared role and one line of rationale.
- `Server/tests/test_route_roles.py` — **the gate** and the sanity checks.

### 11.1 The three roles

| role | what it means | examples |
|---|---|---|
| `runner` | Batch execution reached through the bundle protocol, plus what keeps such a deployment alive: bundle upload / inspect / evidence / download, submission, jobs and logs, the scheduler, the model cache. Its artifact root is a **cache**; every input arrives hash-pinned. | `POST /api/bundles/upload`, `POST /api/studies/submit-bundle`, `GET /api/jobs/{job_id}/stream`, `POST /api/models/install`, `POST /api/housekeeping/maintenance` |
| `workbench` | Interactive serving of a **live, authored** workspace: authoring writes, the workspace switch, concept and manifest writes, server-side freeze, the playground and every synchronous compute the app drives turn by turn, catalog browsing. | `POST /api/authoring/{name}/freeze`, `POST /api/workspace/switch`, `POST /api/generate`, `GET /api/concepts`, `PUT /api/experiment/{name}/manifest` |
| `both` | Genuinely used by **both roles today**. Identity reads, and the cluster deployment's remote-workbench surface. | `GET /api/info`, `POST /api/experiment/{name}/{verb}`, `POST /api/variants/upload`, the `/api/session` family, the deferred-judging intake |

**133 routes: 17 `runner`, 89 `workbench`, 27 `both`.**

The census is **documentation of TODAY, not aspiration.** Where a cluster
deployment legitimately serves the Mac app's interactive features, the route is
censused `both` and the rationale says so, rather than being labelled `runner`
because a tidier architecture would have wanted it that way. The four judgment
calls worth naming:

- **`POST /api/experiment/{name}/{verb}`** — genuine execution (runner-shaped),
  but reached *without* the bundle protocol, against a **server-resident**
  study. That is the app's remote-workbench path against a cluster, and it is
  in use. `both`.
- **The deferred-judging intake** (`…/sweep/awaiting`,
  `…/sweep/complete-judgment`, and the `evaluate` twins) — the blinded packets
  are the runner's own execution output; judging them and stamping the verdicts
  back is the workbench's act, landing in the runner's run directory. Both
  halves of the keyless-custody fork are real. `both`.
- **`POST /api/variants/upload`** — the app's variant-library push *and* how an
  agent artifact an execution needs reaches a cluster. `both`.
- **The `/api/session` family** — a scheduler operation (runner-shaped) whose
  entire product is an **interactive worker** for the playground. `both`.

`POST /api/studies/submit` and `POST /api/bundles/run` fall the same way for
the same reason: submission and packaging are runner acts, but these spellings
name a study **resident in the served workspace**, which a cache does not have.
`POST /api/studies/submit-bundle` is the pure-runner spelling beside them.

### 11.2 The gate

| Contract | Guarantees | Pinned by |
|---|---|---|
| **route-census-completeness** | Every `(method, template)` the **live app object** serves appears in the census — enumerated by walking the router tree, not by grepping the source, so `include_in_schema=False` routes are covered. A new route cannot ship undeclared. | `test_every_route_the_app_serves_is_censused` |
| **route-census-no-fiction** | The other direction: a censused route that no longer exists fails too, so the table cannot rot into a description of code that is gone. | `test_the_census_has_no_entries_for_routes_that_are_gone`, `…_declares_each_route_exactly_once`, `…_uses_the_methods_and_templates_the_router_declares`, `test_every_entry_carries_a_real_rationale` |
| **adapter-routes-are-runner-reachable** | Every route the Phase-2 client adapter speaks (§8.1) is censused `runner` or `both` — a narrowing that refused one would break submit-and-bring-evidence-home, which is the runner role's whole purpose. Both directions again: the adapter's **source** is scanned for `/api/` literals, so a new endpoint cannot escape the check. | `test_every_route_the_client_adapter_uses_is_runner_reachable`, `test_the_adapters_endpoint_scan_finds_nothing_undeclared` |
| **census-agrees-with-WP-S** | The mutating-by-default classification (`api/app.py`) answers a *different* question about the same table, and the two must not contradict. Every deliberately-open mutating route is censused `workbench` (an open mutating route neither writes nor spends, which is not a shape runner work takes); every mutating route the runner role keeps is token-gated; the two explicitly gated **reads** are runner-reachable. | `test_every_deliberately_open_mutating_route_is_workbench`, `test_every_mutating_runner_route_is_token_gated`, `test_the_read_side_privileged_prefixes_are_runner_reachable` |
| **no-restriction-is-active** | No production module references the census at all. If that changes, it must change in the diff that gives the table teeth. | `test_the_census_activates_no_restriction` |

The walk-and-require-declaration mechanism is deliberately the one
`Tests/ExperimentKitTests/CheckoutDependencyTests.swift` already uses for the
baked-path census: the same problem (a table everybody reads as evidence,
which is worthless the moment it silently falls behind the code) with the same
answer.

**Why the census lives beside the tests rather than in `steerlab_server/api/`.**
Because it governs nothing. `_PRIVILEGED_PREFIXES` and `_OPEN_MUTATING_PATHS`
live in `api/app.py` because `auth_middleware` branches on them; this table has
no branch anywhere, and a table shipped inside the installed package that no
code honours is a claim the package makes about itself and does not keep. It is
also inside the light-import surface §7 and §8.6 measure, for nothing. When a
runner profile really does refuse workbench routes, the table moves into the
package as part of **that** change.

### 11.3 One tension, recorded rather than forced

A runner's job roster (`GET /api/jobs`), one job's record
(`GET /api/jobs/{job_id}`) and its log stream (`GET /api/jobs/{job_id}/stream`)
are the runner role's entire observable surface, and **none of the three is
privileged** under WP-S — mutating-by-default gates writes, and these are
reads. There is no open runner today, because a real runner runs in **token
mode**, where `auth_mode == "token"` gates every `/api` route regardless of the
classification. But a runner profile that narrowed a deployment to exactly
these routes *without* token mode would be serving job logs to whoever can
reach the socket. Pinned as an observation by
`test_the_runner_reads_that_carry_no_token_gate_are_the_expected_three`, so a
fourth one cannot join it quietly.

### 11.4 What this step deliberately did NOT do

- **No restriction of any kind.** No route refuses anything it did not refuse
  before; the app's behaviour is byte-identical. There is no runner profile,
  no `--role` flag, and no way to turn one on.
- **No production module changed.** `api/routes.py`, `api/app.py` and every
  other engine module are untouched; the whole change is two new test-tier
  files and this section.
- **No re-labelling to make the table tidier.** Twenty-seven routes are `both`
  because they are used by both today. A census that recorded the architecture
  we would like would be useless as the input to a narrowing.

**The eventual runner profile** — the thing this census is step 1 of — would
read the table and refuse `workbench` routes with **typed refusals** carrying a
`repairAction`, in the vocabulary §4 already pins, so a client that reached for
an authoring verb against a runner would be told which service role it wanted
and where to find one. Every `both` route is a decision that profile still has
to make. **That is future work, and explicitly not this change.**

---

## 12. The neutral token-bank draw

A contract rather than a phase, recorded here because the 2026-08-28 math audit
found it was the one RNG seam in the extraction path that neither shared an
algorithm across engines nor documented an accepted divergence — the sibling
paths do one or the other (`repe_reader` shares SplitMix64; `vector_math` and
`StudyStatistics` document their divergence out loud), and this one did
neither.

The neutral token bank feeds the neutral-PC basis: a corpus is tokenized, every
token position past the reading position's start is a candidate row, and a
per-layer cap bounds how many are kept. **Which rows are kept is now the same
decision on both engines.**

| Contract | Guarantees | Pinned by |
|---|---|---|
| Seed derivation | `seed = first 8 bytes of SHA-256(corpusHash.utf8), big-endian` | `TokenBankDownsamplerTests.seedDerivationIsThePinnedCrossEngineRule` ↔ `test_token_bank_downsampling.py::test_seed_derivation_is_the_pinned_cross_engine_rule` (twin literals) |
| RNG | SplitMix64, the published constants, one 64-bit state word | `.splitMix64StreamIsThePinnedSequence` ↔ `test_splitmix64_stream_is_the_pinned_sequence` (twin literals) |
| Draw | seeded partial Fisher–Yates over `0 ..< count`, first `cap` entries, returned sorted ascending | `.selectionIsThePinnedDraw` ↔ `test_selection_is_the_pinned_draw` (twin literals) |
| Corpus hash from raw texts | SHA-256 over length-prefixed (`uint64` big-endian) UTF-8 texts | `.textCorpusHashIsLengthPrefixed` ↔ `test_corpus_hash_is_length_prefixed` |

Given the same `(count, cap, seed)` the two engines select **byte-identical**
rows. Before this the server derived its seed as `int(hash[:16], 16) % 2**63`
and drew with `random.sample` — a different number through a different
generator, and one whose stability CPython guarantees only for
`Random.random()`, so a Python upgrade could have silently rebuilt a basis from
different positions while every stamp said the recipe was unchanged. The
literals above are what make that impossible: a property-based test would have
passed straight through such a drift.

**The default cap deliberately did NOT converge.** Swift caps at 4096 rows per
layer (`TokenBankDownsampler.defaultRowCapPerLayer`) behind a
`NeutralBankBudget.preflight` refusal; the server caps at 2048
(`extractor.DEFAULT_MAX_TOKEN_ROWS`) and has no preflight, materializing kept
rows as Python floats for every captured layer. A cap is a memory bound, not a
rule: importing the larger default onto the engine with no preflight — the one
that runs on shared cluster GPUs — would double an unguarded transient for no
change in what the draw *means*. Pass the same `maxTokenRows` on both sides
when a bit-identical bank is what you want. The one remaining signature
difference is the server's `None` sentinel for "no cap at all", which Swift's
`Int` cap cannot express; a cap of `0` or less now selects nothing on both
engines (it used to mean "uncapped" on the server, turning a plausible typo
into an unbounded bank).

**Nothing pinned moved.** A neutral-PC basis travels by `neutralPCBasisHash`,
the SHA-256 of the basis file's own bytes, so every pinned basis keeps
verifying exactly as before — the draw only decides what a *new* basis is built
from, and bank-projected vectors were already held to the 0.98-cosine parity
bar rather than to fixture precision (the SVD-vs-Gram divergence recorded in
METHODS).
