import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// How `validate` resolves ONE pinned concept before it measures anything —
/// open-issues §21 (2026-08-20).
///
/// Three choices the server has always made and this engine did not:
///
/// 1. **Method.** The DATA method is the artifact's recorded SOURCE method
///    for an artifact-pinned concept, not the declared `pinnedArtifact`.
///    Switching on the declared method sent every artifact-pinned concept
///    down `.population` (the total-but-least-wrong answer
///    `ExtractionMethod.validationSemantics` gives an unresolved caller),
///    so a pinned CAA direction was scored against a corpus midpoint.
/// 2. **Data identity.** Stimuli and the held-out `validation.jsonl` live
///    under the DATA concept ("tidiness-gm" reads "tidiness"'s), which is
///    what `verify()` pins on BOTH engines. Looking them up under the
///    study-side name found nothing, so the probe silently did not run while
///    the freeze gate pinned a file the run never read.
/// 3. **Skip rule.** A direction with no source concept (OptVec, an imported
///    SAE decoder row) has no stimuli, no class means and no held-out set at
///    all. The server `continue`s; this engine had no equivalent.
///
/// The three travel together by construction: resolving (2) without (1)
/// changes which recipe root the lookup calls canonical, i.e. which file a
/// paired-vs-story concept reads.
///
/// The fixture manifest is committed and byte-identical to
/// `Server/tests/fixtures/validate-resolution/experiment.json`; the Python
/// twin (`Server/tests/test_validate_concept_resolution.py`) asserts the same
/// outcome table over the same bytes, so drift in either engine is a test
/// failure.
@Suite(.serialized) struct ValidateConceptResolutionTests {

    // MARK: - The committed cross-engine fixture

    private static let fixture: URL =
        URL(filePath: #filePath)  // …/Tests/ExperimentKitTests/<this file>
        .deletingLastPathComponent()
        .appending(components: "Fixtures", "validate-resolution", "experiment.json")

    private func loadFixture() throws -> ExperimentManifest {
        try JSONDecoder().decode(
            ExperimentManifest.self,
            from: try Data(contentsOf: Self.fixture))
    }

    private func concept(
        _ name: String, in manifest: ExperimentManifest
    ) throws -> ExperimentManifest.ConceptRef {
        try #require(manifest.concepts.first { $0.name == name })
    }

    private func resolvedProbe(
        _ name: String, in manifest: ExperimentManifest
    ) throws -> ExperimentTasks.ValidationProbePlan.Probe {
        switch try ExperimentTasks.validationProbePlan(
            for: try concept(name, in: manifest))
        {
        case .probe(let probe): return probe
        case .skipped(let reason):
            Issue.record("'\(name)' should owe a probe, but was skipped: \(reason)")
            throw ExperimentError(reason: "unexpected skip")
        }
    }

    // MARK: - (1) Method resolves through the ARTIFACT's source method

    @Test func anOrdinaryRecipeAnswersWithItsDeclaredMethod() throws {
        let manifest = try loadFixture()
        let probe = try resolvedProbe("tidiness", in: manifest)
        #expect(probe.method == .emotionGrandMean)
        #expect(probe.isGrandMean)
        #expect(probe.dataConcept == "tidiness")
        #expect(!probe.validationIsPaired)
    }

    @Test func anArtifactPinnedConceptAnswersWithItsSourceMethod() throws {
        let manifest = try loadFixture()
        // Declared `pinnedArtifact` — which declares no validation semantics
        // of its own and would have scored as `.population` by accident.
        let ref = try concept("stillness-pair", in: manifest)
        #expect(ref.options.method == .pinnedArtifact)
        #expect(ref.options.method.validationSemantics == .population)

        let probe = try resolvedProbe("stillness-pair", in: manifest)
        #expect(probe.method == .meanDifference)
        #expect(!probe.isGrandMean)  // the bug: contrastive, scored as population
        #expect(probe.method.usesContrastiveValidation)
    }

    @Test func anArtifactPinnedGrandMeanStaysPopulationScored() throws {
        let manifest = try loadFixture()
        let probe = try resolvedProbe("tidiness-gm", in: manifest)
        #expect(probe.method == .emotionGrandMean)
        #expect(probe.isGrandMean)
    }

    // MARK: - (2) Data identity is the DATA concept, as `verify()` pins it

    @Test func theProbeReadsTheDataConceptNotTheStudySideName() throws {
        let manifest = try loadFixture()
        #expect(try resolvedProbe("tidiness-gm", in: manifest).dataConcept == "tidiness")
        #expect(try resolvedProbe("stillness-pair", in: manifest).dataConcept == "stillness")
        // An ordinary recipe is its own data concept — the aligned rule is a
        // superset of the old one, never a change for unpinned studies.
        #expect(try resolvedProbe("tidiness", in: manifest).dataConcept == "tidiness")
    }

    @Test func theMethodDecidesWhichRecipeRootIsCanonical() throws {
        let manifest = try loadFixture()
        // Story-corpus recipes are canonical under prompts/emotions/, paired
        // recipes under prompts/concepts/ — resolved from the EFFECTIVE
        // method, which is why (1) and (2) cannot land separately.
        #expect(!(try resolvedProbe("tidiness-gm", in: manifest).validationIsPaired))
        #expect(try resolvedProbe("stillness-pair", in: manifest).validationIsPaired)
    }

    /// The lookup the validate loop performs with the plan's identities: the
    /// held-out set is found under the DATA concept, and the advisory that
    /// travels with it is labelled with the STUDY-side name (the server does
    /// exactly this — `validation_lookup_advisory(concept_name, location)`
    /// over `resolve_validation_file(data_concept, …)`).
    @Test func theDualRootLookupRunsUnderTheDataConceptAndAdvisesUnderTheStudyName()
        throws
    {
        let manifest = try loadFixture()
        try withBothRootsOverridden { root in
            let probe = try resolvedProbe("tidiness-gm", in: manifest)
            try plantStories("tidiness", root: root)
            let digest = try write(
                Self.scenario, to: root,
                path: "prompts/emotions/tidiness/validation.jsonl")

            // Under the STUDY-side name: nothing. This is the divergence —
            // the probe silently did not run.
            #expect(
                ExperimentStore.resolveConceptValidation(
                    name: "tidiness-gm", isPaired: probe.validationIsPaired) == nil)

            let location = try #require(
                ExperimentStore.resolveConceptValidation(
                    name: probe.dataConcept, isPaired: probe.validationIsPaired))
            #expect(
                location.relativePath == "prompts/emotions/tidiness/validation.jsonl")
            #expect(!location.usedFallback)
            #expect(ExperimentStore.conceptValidationHash(fileURL: location.url) == digest)
            // And it is the file `verify()` pins for this concept.
            #expect(
                ExperimentStore.conceptValidationHash(
                    name: try concept("tidiness-gm", in: manifest).dataConcept,
                    isPaired: probe.validationIsPaired) == digest)
        }
    }

    /// The dual-root FALLBACK still applies under the data concept: a
    /// grand-mean-sourced artifact whose held-out set was misfiled beside the
    /// paired stimuli is found, read, and advised about — under the study
    /// name, so the researcher sees the concept they declared.
    @Test func theFallbackRootStillAppliesUnderTheDataConcept() throws {
        let manifest = try loadFixture()
        try withBothRootsOverridden { root in
            let ref = try concept("tidiness-gm", in: manifest)
            let probe = try resolvedProbe("tidiness-gm", in: manifest)
            try plantStories("tidiness", root: root)
            let digest = try write(
                Self.scenario, to: root,
                path: "prompts/concepts/tidiness/validation.jsonl")

            let location = try #require(
                ExperimentStore.resolveConceptValidation(
                    name: probe.dataConcept, isPaired: probe.validationIsPaired))
            #expect(location.usedFallback)
            #expect(
                location.relativePath == "prompts/concepts/tidiness/validation.jsonl")
            #expect(
                location.canonicalRelativePath
                    == "prompts/emotions/tidiness/validation.jsonl")
            #expect(ExperimentStore.conceptValidationHash(fileURL: location.url) == digest)

            let advisory = try #require(
                ExperimentStore.validationLookupAdvisory(
                    concept: ref.name, location: location))
            // Labelled with the STUDY-side name, resolved under the data one.
            #expect(advisory.hasPrefix("concept 'tidiness-gm':"))
            #expect(advisory.contains("prompts/emotions/tidiness/validation.jsonl"))
        }
    }

    // MARK: - (3) No source concept ⇒ skipped, and never recorded vacuous

    @Test func aDirectionWithNoSourceConceptIsSkipped() throws {
        let manifest = try loadFixture()
        for name in ["drifting-optvec", "dictionary-row"] {
            let ref = try concept(name, in: manifest)
            switch try ExperimentTasks.validationProbePlan(for: ref) {
            case .skipped(let reason):
                #expect(reason.contains("no source concept"))
                #expect(reason.contains(name))
            case .probe(let probe):
                Issue.record("'\(name)' has nothing to probe, got \(probe)")
            }
        }
    }

    /// The skip/vacuity interaction, spelled out: a skipped concept OWES no
    /// held-out probe, so the ledger the validate loop keeps (which appends
    /// only for concepts that owe one) cannot record it as vacuous evidence.
    /// The server reaches the same state from the other direction — its
    /// ledger is seeded from `owes_held_out_probe` and these concepts are
    /// never in it — so the freeze gate sees the same verdict on both
    /// engines.
    @Test func aSkippedConceptIsNeverCountedAsVacuousEvidence() throws {
        let manifest = try loadFixture()
        for name in ["drifting-optvec", "dictionary-row"] {
            let ref = try concept(name, in: manifest)
            #expect(!ExperimentStore.owesHeldOutProbe(ref))
        }
        // A concept that DOES owe one stays on the hook — the skip is
        // narrow, not a hole in the vacuity gate.
        for name in ["tidiness", "tidiness-gm", "stillness-pair"] {
            #expect(ExperimentStore.owesHeldOutProbe(try concept(name, in: manifest)))
        }
        // And the path a vacuity refusal would NAME follows the same two
        // resolutions (effective method's family, data concept's directory).
        #expect(
            ExperimentStore.heldOutProbePath(try concept("tidiness-gm", in: manifest))
                == "prompts/emotions/tidiness/validation.jsonl")
        #expect(
            ExperimentStore.heldOutProbePath(
                try concept("stillness-pair", in: manifest))
                == "prompts/concepts/stillness/validation.jsonl")
    }

    // MARK: - Exhaustive semantics: refuse, never mis-branch

    @Test func aMethodWithNoValidationSemanticsRefuses() throws {
        let manifest = try loadFixture()
        // Reachable only by hand-editing a manifest (both engines' attach
        // refuses an artifact whose own source method is `pinnedArtifact`),
        // which is exactly why validate must refuse rather than fall into
        // whichever branch is syntactically last.
        let ref = try concept("undeclared-semantics", in: manifest)
        #expect(ref.effectiveMethod == .pinnedArtifact)
        do {
            _ = try ExperimentTasks.validationProbePlan(for: ref)
            Issue.record("a method with no validation semantics must refuse")
        } catch let error as ExperimentError {
            #expect(
                error.reason
                    == "concept 'undeclared-semantics': method 'pinnedArtifact' "
                    + "declares no validation semantics (neither contrastive "
                    + "nor grand-mean)")
        }
    }

    @Test func anUnknownSourceMethodRefusesRatherThanGuessing() throws {
        var ref = ExperimentManifest.ConceptRef(
            name: "from-the-future", stimulusSetHash: "h",
            options: ExtractionOptions(method: .pinnedArtifact))
        ref.vectorArtifact = ExperimentManifest.ConceptRef.VectorArtifactPin(
            path: "runs/x/from-the-future",
            sha256TensorHash: "t", sha256SidecarHash: "s",
            sourceMethod: "someMethodThisEngineDoesNotKnow",
            sourceConcept: "from-the-future",
            residualNormSource: "neutral-corpus")
        #expect(ref.effectiveMethod == nil)
        do {
            _ = try ExperimentTasks.validationProbePlan(for: ref)
            Issue.record("an unknown source method must refuse")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("someMethodThisEngineDoesNotKnow"))
            #expect(error.reason.contains("does not know"))
        }
    }

    // MARK: - Fixture helpers (temp-root planting, as ValidationDualRootTests)

    /// Both root seams at once: `ExperimentStore.rootOverride` (prompts/
    /// emotions) AND `WorkspaceRoot.programmaticOverride`
    /// (`VectorCatalog.projectRoot`, where the PAIRED home resolves).
    private func withBothRootsOverridden<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "valresolve") { root in
            let previous = WorkspaceRoot.programmaticOverride
            WorkspaceRoot.programmaticOverride = root
            defer { WorkspaceRoot.programmaticOverride = previous }
            return try body(root)
        }
    }

    private static let scenario =
        #"{"text": "a held-out scene", "expresses": true}"# + "\n"

    private func sha256Hex(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8))
            .map { String(format: "%02x", $0) }.joined()
    }

    @discardableResult
    private func write(_ text: String, to root: URL, path: String) throws -> String {
        let url = root.appending(path: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: url, atomically: true, encoding: .utf8)
        return sha256Hex(text)
    }

    private func plantStories(_ name: String, root: URL) throws {
        try write(
            """
            {"concept": "\(name)", "text": "a story about \(name)"}
            {"concept": "\(name)", "text": "another story about \(name)"}

            """,
            to: root, path: "prompts/emotions/\(name)/stories.jsonl")
    }
}
