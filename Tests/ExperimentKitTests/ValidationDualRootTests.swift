import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// Dual-root lookup for a concept's held-out validation.jsonl (2026-08-19).
///
/// A validation set has TWO possible homes and the RECIPE decides which is
/// canonical: paired recipes read `prompts/concepts/<name>/`, the grand-mean
/// recipe reads `prompts/emotions/<name>/`. Before this change a set filed
/// under the wrong recipe's root was treated as ABSENT — no hash pinned, no
/// error — so a measurement-side pin went quietly missing from the firewall.
///
/// The fix is lookup-only and deliberately narrow: the canonical home is
/// read first and always wins; the other home is a fallback; using the
/// fallback (or finding a file in both homes) is LOUD but never fatal. The
/// three-state pin semantics (hash / explicitly-absent / legacy-unpinned)
/// are untouched.
///
/// Python twin: `Server/tests/test_validation_dual_root.py`.
@Suite(.serialized) struct ValidationDualRootTests {

    // MARK: - Fixtures

    /// Both root seams at once: `ExperimentStore.rootOverride` (experiments,
    /// runs, prompts/emotions) AND `WorkspaceRoot.programmaticOverride`
    /// (`VectorCatalog.projectRoot`, which is where the PAIRED home
    /// resolves). Only with both pointing at the temp tree do the two homes
    /// of one concept live in the same workspace, which is the situation a
    /// misfiled set actually occurs in.
    private func withBothRootsOverridden<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "valroot") { root in
            let previous = WorkspaceRoot.programmaticOverride
            WorkspaceRoot.programmaticOverride = root
            defer { WorkspaceRoot.programmaticOverride = previous }
            return try body(root)
        }
    }

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

    private let scenario = #"{"text": "a hidden scenario", "expresses": true}"# + "\n"
    private let otherScenario =
        #"{"text": "a different hidden scenario", "expresses": false}"# + "\n"

    // MARK: - Resolution order

    @Test func canonicalHomeResolvesWithNoAdvisory() throws {
        try withBothRootsOverridden { root in
            try plantStories("serenity", root: root)
            let digest = try write(
                scenario, to: root,
                path: "prompts/emotions/serenity/validation.jsonl")

            let location = try #require(
                ExperimentStore.resolveConceptValidation(
                    name: "serenity", isPaired: false))
            #expect(location.relativePath == "prompts/emotions/serenity/validation.jsonl")
            #expect(!location.usedFallback)
            #expect(!location.bothHomesPresent)
            #expect(
                ExperimentStore.validationLookupAdvisory(
                    concept: "serenity", location: location) == nil)
            #expect(
                ExperimentStore.conceptValidationHash(
                    name: "serenity", isPaired: false) == digest)
        }
    }

    @Test func fallbackHomeResolvesLoudlyAndStillPins() throws {
        try withBothRootsOverridden { root in
            // Grand-mean recipe, but the held-out set was filed beside the
            // PAIRED stimuli — the misfiling this change exists for.
            try plantStories("serenity", root: root)
            let digest = try write(
                scenario, to: root,
                path: "prompts/concepts/serenity/validation.jsonl")

            let location = try #require(
                ExperimentStore.resolveConceptValidation(
                    name: "serenity", isPaired: false))
            #expect(location.usedFallback)
            #expect(!location.bothHomesPresent)
            #expect(location.relativePath == "prompts/concepts/serenity/validation.jsonl")
            #expect(
                location.canonicalRelativePath
                    == "prompts/emotions/serenity/validation.jsonl")

            let advisory = try #require(
                ExperimentStore.validationLookupAdvisory(
                    concept: "serenity", location: location))
            #expect(advisory.contains("OTHER recipe's root"))
            #expect(advisory.contains("prompts/emotions/serenity/validation.jsonl"))

            // The pin is computed from the file actually FOUND — it used to
            // be nil, i.e. an unpinned measurement-side input.
            #expect(
                ExperimentStore.conceptValidationHash(
                    name: "serenity", isPaired: false) == digest)
        }
    }

    @Test func bothHomesPresentCanonicalWinsWithAmbiguityAdvisory() throws {
        try withBothRootsOverridden { root in
            try plantStories("serenity", root: root)
            let canonical = try write(
                scenario, to: root,
                path: "prompts/emotions/serenity/validation.jsonl")
            let fallback = try write(
                otherScenario, to: root,
                path: "prompts/concepts/serenity/validation.jsonl")
            #expect(canonical != fallback)

            let location = try #require(
                ExperimentStore.resolveConceptValidation(
                    name: "serenity", isPaired: false))
            #expect(location.bothHomesPresent)
            #expect(!location.usedFallback)
            #expect(location.relativePath == location.canonicalRelativePath)

            let advisory = try #require(
                ExperimentStore.validationLookupAdvisory(
                    concept: "serenity", location: location))
            #expect(advisory.contains("BOTH"))
            #expect(advisory.contains(location.fallbackRelativePath))

            // Canonical bytes win.
            #expect(
                ExperimentStore.conceptValidationHash(
                    name: "serenity", isPaired: false) == canonical)
        }
    }

    @Test func neitherHomeStaysAbsentAsBefore() throws {
        try withBothRootsOverridden { root in
            try plantStories("serenity", root: root)
            #expect(
                ExperimentStore.resolveConceptValidation(
                    name: "serenity", isPaired: false) == nil)
            #expect(
                ExperimentStore.validationLookupAdvisory(
                    concept: "serenity", location: nil) == nil)
            #expect(
                ExperimentStore.conceptValidationHash(
                    name: "serenity", isPaired: false) == nil)
        }
    }

    // MARK: - Lifecycle: attach pins it, verify accepts it, freeze advises

    @Test func attachPinsMisfiledSetAndVerifyStaysClean() throws {
        try withBothRootsOverridden { root in
            var manifest = try ExperimentStore.create(
                name: "dual-root", description: "", modelID: "test/model")
            try plantStories("serenity", root: root)
            let digest = try write(
                scenario, to: root,
                path: "prompts/concepts/serenity/validation.jsonl")
            try ExperimentStore.attachGrandMeanConcepts(["serenity"], into: &manifest)

            #expect(manifest.concepts.first?.validationHash == digest)
            #expect(ExperimentStore.verify(manifest).isEmpty)

            // Drift in the FALLBACK-home file is still drift.
            try write(
                otherScenario, to: root,
                path: "prompts/concepts/serenity/validation.jsonl")
            #expect(
                ExperimentStore.verify(manifest).contains {
                    $0.contains("validation.jsonl changed since pinning")
                })
        }
    }

    @Test func freezeAdvisoryNamesTheMisfiledSet() throws {
        try withBothRootsOverridden { root in
            var manifest = try ExperimentStore.create(
                name: "dual-root-advice", description: "", modelID: "test/model")
            try plantStories("serenity", root: root)
            try write(
                scenario, to: root,
                path: "prompts/concepts/serenity/validation.jsonl")
            try ExperimentStore.attachGrandMeanConcepts(["serenity"], into: &manifest)

            let advisories = ExperimentStore.freezeAdvisories(for: manifest)
            #expect(
                advisories.contains {
                    $0.contains("OTHER recipe's root")
                        && $0.contains("prompts/emotions/serenity/validation.jsonl")
                })
        }
    }

    @Test func freezeAdvisorySilentWhenFilingIsCanonical() throws {
        try withBothRootsOverridden { root in
            var manifest = try ExperimentStore.create(
                name: "dual-root-clean", description: "", modelID: "test/model")
            try plantStories("serenity", root: root)
            try write(
                scenario, to: root,
                path: "prompts/emotions/serenity/validation.jsonl")
            try ExperimentStore.attachGrandMeanConcepts(["serenity"], into: &manifest)

            #expect(
                !ExperimentStore.freezeAdvisories(for: manifest).contains {
                    $0.contains("recipe's root") || $0.contains("BOTH recipe roots")
                })
        }
    }

    // MARK: - `data check` (StudyDataReadiness — pure, explicit root)

    private func readinessRows(
        root: URL, method: ExtractionMethod
    ) -> [DataRequirement] {
        var manifest = ExperimentManifest(
            name: "readiness", description: "", modelID: "test/model")
        manifest.concepts.append(
            ExperimentManifest.ConceptRef(
                name: "urgency", stimulusSetHash: "unused",
                options: ExtractionOptions(method: method)))
        return StudyDataReadiness.requirements(for: manifest, workspaceRoot: root)
    }

    @Test func dataCheckFlagsAMisfiledValidationSetWithoutBlocking() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "valroot-readiness-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try write(
            #"{"concept": "urgency", "text": "a story"}"# + "\n",
            to: root, path: "prompts/emotions/urgency/stories.jsonl")
        // Filed under the PAIRED root while the recipe is grand-mean.
        try write(
            scenario, to: root, path: "prompts/concepts/urgency/validation.jsonl")

        let row = try #require(
            readinessRows(root: root, method: .emotionGrandMean)
                .first { $0.id == "concept:urgency:validation" })
        // Non-blocking: the engine reads it. But it is a FINDING, and the
        // row names the canonical home.
        #expect(row.status == .partial)
        #expect(row.path == "prompts/emotions/urgency/validation.jsonl")
        #expect(row.detail.contains("prompts/concepts/urgency/validation.jsonl"))
        #expect(row.detail.contains("prompts/emotions/urgency/validation.jsonl"))
        #expect(
            StudyDataReadiness.summary(
                readinessRows(root: root, method: .emotionGrandMean)
            ).blockers.allSatisfy { $0.id != "concept:urgency:validation" })
    }

    @Test func dataCheckStaysCleanForACanonicallyFiledValidationSet() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "valroot-readiness-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try write(
            #"{"concept": "urgency", "text": "a story"}"# + "\n",
            to: root, path: "prompts/emotions/urgency/stories.jsonl")
        try write(
            scenario, to: root, path: "prompts/emotions/urgency/validation.jsonl")

        let row = try #require(
            readinessRows(root: root, method: .emotionGrandMean)
                .first { $0.id == "concept:urgency:validation" })
        #expect(row.status == .present)
        #expect(!row.detail.contains("NOTE:"))
    }

    @Test func dataCheckFlagsADuplicatedValidationSet() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "valroot-readiness-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try write(
            #"{"concept": "urgency", "text": "a story"}"# + "\n",
            to: root, path: "prompts/emotions/urgency/stories.jsonl")
        try write(
            scenario, to: root, path: "prompts/emotions/urgency/validation.jsonl")
        try write(
            otherScenario, to: root,
            path: "prompts/concepts/urgency/validation.jsonl")

        let row = try #require(
            readinessRows(root: root, method: .emotionGrandMean)
                .first { $0.id == "concept:urgency:validation" })
        #expect(row.status == .partial)
        #expect(row.detail.contains("a second validation.jsonl"))
    }
}
