import Foundation
import Testing

@testable import ExperimentKit

/// Exhaustive coverage of the pure evidence-note rule (`AgentEvidence.notes`)
/// and the latest-robustness lookup — fixture artifacts and temp run
/// directories only; no model, no live workspace, and nothing here blocks
/// anything (the notes are data, not gates).
@Suite struct AgentEvidenceTests {

    // MARK: - Fixtures

    private func makeArtifact(
        name: String = "agent-a",
        promotion: ModelVariantArtifact.Promotion? = nil
    ) -> ModelVariantArtifact {
        ModelVariantArtifact(
            name: name,
            baseModelID: "Qwen/Qwen3-4B-MLX-4bit",
            promptMode: "chatAssistant",
            qwenThinkingEnabled: false,
            temperature: 0,
            systemPrompt: "",
            promotion: promotion)
    }

    private func makePromotion(
        promotedBy: String = "criterion",
        overrideReason: String? = nil,
        metric: String? = "judgeScore",
        sweepRun: String? = "2026-07-01-exp-study-sweep",
        substrate: String = "swift-mlx"
    ) -> ModelVariantArtifact.Promotion {
        ModelVariantArtifact.Promotion(
            experiment: "study-1",
            experimentHash: "abc123",
            promotedAt: "2026-07-01T10:00:00Z",
            promotedBy: promotedBy,
            overrideReason: overrideReason,
            sweepRun: sweepRun,
            criterion: metric.map {
                ExperimentManifest.SweepSelection(
                    objective: .init(metric: $0))
            },
            substrate: substrate,
            appVersion: "test")
    }

    private func makeReport(
        variantName: String = "agent-a",
        artifactHash: String? = "hash-1",
        generatedAt: String = "2026-07-01T10:00:00Z",
        baselineAccuracy: Float = 0.95,
        variantAccuracy: Float = 0.92,
        warnings: [String] = []
    ) -> VariantRobustnessReport {
        VariantRobustnessReport(
            variantName: variantName,
            variantArtifactPath: "runs/model-variants/x/model-variant.json",
            variantArtifactHash: artifactHash,
            baseModelID: "Qwen/Qwen3-4B-MLX-4bit",
            substrate: "swift-mlx",
            presetID: "quick",
            batteryFile: "prompts/batteries/basic.jsonl",
            coherencePromptsFile: "prompts/dev/dev-prompts.jsonl",
            judgeModel: nil,
            generatedAt: generatedAt,
            baselineBatteryAccuracy: baselineAccuracy,
            variantBatteryAccuracy: variantAccuracy,
            meanBaselineDistinct2: 0.8,
            meanVariantDistinct2: 0.78,
            meanBaselineWords: 100,
            meanVariantWords: 98,
            batteryItems: [],
            coherenceItems: [],
            warnings: warnings)
    }

    private func labels(_ notes: [AgentEvidence.Note]) -> [String] {
        notes.map(\.label)
    }

    private func note(
        _ notes: [AgentEvidence.Note], _ label: String
    ) -> AgentEvidence.Note? {
        notes.first { $0.label == label }
    }

    // MARK: - (a) Hand-created / exploratory

    @Test func handCreatedArtifactGetsExploratoryNote() throws {
        let notes = AgentEvidence.notes(
            for: makeArtifact(),
            currentSubstrate: "swift-mlx",
            latestRobustness: makeReport())
        #expect(labels(notes) == ["Hand-created"])
        let handCreated = try #require(note(notes, "Hand-created"))
        #expect(handCreated.severity == .informational)
        #expect(handCreated.detail.contains("no sweep-selection provenance"))
        #expect(handCreated.detail.contains("sweep-promoted"))
    }

    @Test func handCreatedArtifactNeverGetsPromotionNotes() {
        let notes = AgentEvidence.notes(
            for: makeArtifact(),
            currentSubstrate: "python-hf-transformers",
            latestRobustness: makeReport())
        // No promotion → no override, criterion, or substrate claims (the
        // artifact itself carries no substrate stamp — only the promotion
        // birth certificate does).
        #expect(note(notes, "Manual override") == nil)
        #expect(note(notes, "Marker-density criterion") == nil)
        #expect(note(notes, "Substrate mismatch") == nil)
    }

    @Test func cleanPromotedAgentGetsNoNotes() {
        let notes = AgentEvidence.notes(
            for: makeArtifact(promotion: makePromotion(metric: "judgeScore")),
            currentSubstrate: "swift-mlx",
            latestRobustness: makeReport())
        #expect(notes.isEmpty)
    }

    // MARK: - (b) Manual override

    @Test func manualOverrideNoteIncludesReason() throws {
        let notes = AgentEvidence.notes(
            for: makeArtifact(
                promotion: makePromotion(
                    promotedBy: "manualOverride",
                    overrideReason: "recommended cell degenerated on longer prompts")),
            currentSubstrate: "swift-mlx",
            latestRobustness: makeReport())
        let override = try #require(note(notes, "Manual override"))
        #expect(override.severity == .caution)
        #expect(override.detail.contains("declared selection was bypassed"))
        #expect(override.detail.contains("recommended cell degenerated on longer prompts"))
    }

    @Test func manualOverrideWithoutReasonSaysUnrecorded() throws {
        let notes = AgentEvidence.notes(
            for: makeArtifact(
                promotion: makePromotion(
                    promotedBy: "manualOverride", overrideReason: nil)),
            currentSubstrate: "swift-mlx",
            latestRobustness: makeReport())
        let override = try #require(note(notes, "Manual override"))
        #expect(override.detail.contains("no override reason recorded"))
    }

    @Test func criterionPromotionGetsNoOverrideNote() {
        let notes = AgentEvidence.notes(
            for: makeArtifact(promotion: makePromotion(promotedBy: "criterion")),
            currentSubstrate: "swift-mlx",
            latestRobustness: makeReport())
        #expect(note(notes, "Manual override") == nil)
    }

    // MARK: - (c) Marker-density criterion

    @Test func markerDensityCriterionGetsConfoundCaution() throws {
        let notes = AgentEvidence.notes(
            for: makeArtifact(promotion: makePromotion(metric: "markerDensity")),
            currentSubstrate: "swift-mlx",
            latestRobustness: makeReport())
        let marker = try #require(note(notes, "Marker-density criterion"))
        #expect(marker.severity == .caution)
        #expect(marker.detail.contains("surface-prose confound"))
        #expect(marker.detail.contains("not an outcome instrument"))
    }

    @Test func legacyPromotionWithoutCriterionResolvesToMarkerDensity() {
        // A birth certificate that predates criterion stamping was selected
        // by the historical rule — marker density (SweepSelectionRule).
        let notes = AgentEvidence.notes(
            for: makeArtifact(promotion: makePromotion(metric: nil)),
            currentSubstrate: "swift-mlx",
            latestRobustness: makeReport())
        #expect(note(notes, "Marker-density criterion") != nil)
    }

    @Test func nonMarkerObjectivesGetNoCriterionNote() {
        for metric in ["judgeScore", "logprobShift"] {
            let notes = AgentEvidence.notes(
                for: makeArtifact(promotion: makePromotion(metric: metric)),
                currentSubstrate: "swift-mlx",
                latestRobustness: makeReport())
            #expect(note(notes, "Marker-density criterion") == nil)
        }
    }

    // MARK: - (d) Substrate mismatch

    @Test func substrateMismatchGetsCautionNamingBothSubstrates() throws {
        let notes = AgentEvidence.notes(
            for: makeArtifact(promotion: makePromotion(substrate: "swift-mlx")),
            currentSubstrate: "python-hf-transformers",
            latestRobustness: makeReport())
        let mismatch = try #require(note(notes, "Substrate mismatch"))
        #expect(mismatch.severity == .caution)
        #expect(mismatch.detail.contains("swift-mlx"))
        #expect(mismatch.detail.contains("python-hf-transformers"))
        #expect(mismatch.detail.contains("re-extracted"))
    }

    @Test func matchingSubstrateGetsNoNote() {
        let notes = AgentEvidence.notes(
            for: makeArtifact(promotion: makePromotion(substrate: "swift-mlx")),
            currentSubstrate: "swift-mlx",
            latestRobustness: makeReport())
        #expect(note(notes, "Substrate mismatch") == nil)
    }

    @Test func unknownCurrentSubstrateMakesNoSubstrateClaim() {
        let notes = AgentEvidence.notes(
            for: makeArtifact(promotion: makePromotion(substrate: "swift-mlx")),
            currentSubstrate: nil,
            latestRobustness: makeReport())
        #expect(note(notes, "Substrate mismatch") == nil)
    }

    // MARK: - (e) No robustness result

    @Test func missingRobustnessGetsInformationalNote() throws {
        let notes = AgentEvidence.notes(
            for: makeArtifact(promotion: makePromotion()),
            currentSubstrate: "swift-mlx",
            latestRobustness: nil)
        let robustness = try #require(note(notes, "No robustness result"))
        #expect(robustness.severity == .informational)
        #expect(robustness.detail.contains("robustness"))
    }

    @Test func presentRobustnessSuppressesTheNote() {
        let notes = AgentEvidence.notes(
            for: makeArtifact(promotion: makePromotion()),
            currentSubstrate: "swift-mlx",
            latestRobustness: makeReport())
        #expect(note(notes, "No robustness result") == nil)
    }

    // MARK: - Combinations and ordering

    @Test func worstCaseArtifactStacksAllNotesInStableOrder() {
        let notes = AgentEvidence.notes(
            for: makeArtifact(
                promotion: makePromotion(
                    promotedBy: "manualOverride",
                    overrideReason: "picked by eye",
                    metric: "markerDensity",
                    substrate: "python-hf-transformers")),
            currentSubstrate: "swift-mlx",
            latestRobustness: nil)
        #expect(labels(notes) == [
            "Manual override",
            "Marker-density criterion",
            "Substrate mismatch",
            "No robustness result",
        ])
    }

    @Test func handCreatedWithoutRobustnessStacksBothNotes() {
        let notes = AgentEvidence.notes(
            for: makeArtifact(),
            currentSubstrate: "swift-mlx",
            latestRobustness: nil)
        #expect(labels(notes) == ["Hand-created", "No robustness result"])
    }

    @Test func artifactNotFoundNoteIsACaution() {
        let missing = AgentEvidence.artifactNotFoundNote
        #expect(missing.severity == .caution)
        #expect(missing.detail.contains("pinned snapshot"))
        #expect(missing.displayLine.hasPrefix("Artifact not found — "))
    }

    // MARK: - Provenance line

    @Test func provenanceLineNamesRunAndVerbatimCriterion() {
        let line = AgentEvidence.provenanceLine(
            for: makePromotion(metric: "judgeScore"))
        #expect(line == "promoted from optimization run "
            + "2026-07-01-exp-study-sweep · criterion judgeScore")
    }

    @Test func provenanceLineFallsBackToExperimentWithoutSweepRun() {
        let line = AgentEvidence.provenanceLine(
            for: makePromotion(metric: nil, sweepRun: nil))
        #expect(line == "promoted from 'study-1'")
    }

    @Test func provenanceLineFlagsManualOverride() {
        let line = AgentEvidence.provenanceLine(
            for: makePromotion(promotedBy: "manualOverride", metric: "judgeScore"))
        #expect(line.hasSuffix("· manual override"))
    }

    // MARK: - Robustness summary line

    @Test func robustnessSummaryLineFormatsAccuraciesDateAndWarnings() {
        let clean = AgentEvidence.robustnessSummaryLine(
            makeReport(
                generatedAt: "2026-07-08T09:30:00Z",
                baselineAccuracy: 0.95, variantAccuracy: 0.92))
        #expect(clean == "robustness: battery 0.92 vs 0.95 baseline · 2026-07-08")

        let warned = AgentEvidence.robustnessSummaryLine(
            makeReport(
                generatedAt: "2026-07-08T09:30:00Z",
                warnings: ["capability battery dropped by more than 0.15"]))
        #expect(warned.hasSuffix("· 1 warning"))
    }

    // MARK: - Latest-robustness lookup (temp run directories)

    private func makeRunsRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appending(
            component: "agent-evidence-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        return root
    }

    private func writeReport(
        _ report: VariantRobustnessReport, dirName: String, under root: URL
    ) throws {
        let directory = root.appending(component: dirName)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(report).write(
            to: directory.appending(component: VariantRobustness.reportFileName))
    }

    @Test func scanFindsReportsNewestFirstAndSkipsNonReports() throws {
        let root = try makeRunsRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeReport(
            makeReport(generatedAt: "2026-07-01T10:00:00Z"),
            dirName: "2026-07-01-variant-robustness-a", under: root)
        try writeReport(
            makeReport(generatedAt: "2026-07-05T10:00:00Z"),
            dirName: "2026-07-05-variant-robustness-a", under: root)
        // A run directory without a report, and a malformed report: skipped.
        let plain = root.appending(component: "2026-07-02-exp-x-run")
        try FileManager.default.createDirectory(
            at: plain, withIntermediateDirectories: true)
        let broken = root.appending(component: "2026-07-03-variant-robustness-b")
        try FileManager.default.createDirectory(
            at: broken, withIntermediateDirectories: true)
        try Data("not json".utf8).write(
            to: broken.appending(component: VariantRobustness.reportFileName))

        let scanned = AgentEvidence.scanRobustnessReports(runsDirectory: root)
        #expect(scanned.count == 2)
        #expect(scanned.first?.report.generatedAt == "2026-07-05T10:00:00Z")
    }

    @Test func scanOfMissingRunsDirectoryIsEmpty() {
        let missing = FileManager.default.temporaryDirectory.appending(
            component: "agent-evidence-no-such-dir-\(UUID().uuidString)")
        #expect(AgentEvidence.scanRobustnessReports(runsDirectory: missing).isEmpty)
    }

    @Test func lookupReturnsNewestNameMatch() throws {
        let root = try makeRunsRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeReport(
            makeReport(artifactHash: "h1", generatedAt: "2026-07-01T10:00:00Z"),
            dirName: "2026-07-01-variant-robustness-a", under: root)
        try writeReport(
            makeReport(artifactHash: "h2", generatedAt: "2026-07-05T10:00:00Z"),
            dirName: "2026-07-05-variant-robustness-a", under: root)
        try writeReport(
            makeReport(
                variantName: "agent-b", artifactHash: "h9",
                generatedAt: "2026-07-06T10:00:00Z"),
            dirName: "2026-07-06-variant-robustness-b", under: root)

        let found = AgentEvidence.latestRobustness(
            variantName: "agent-a", artifactHash: nil, runsDirectory: root)
        #expect(found?.report.variantArtifactHash == "h2")
        #expect(
            found?.runDirectory.lastPathComponent
                == "2026-07-05-variant-robustness-a")
    }

    @Test func lookupPrefersHashMatchOverNewerNameMatch() throws {
        let root = try makeRunsRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeReport(
            makeReport(artifactHash: "h1", generatedAt: "2026-07-01T10:00:00Z"),
            dirName: "2026-07-01-variant-robustness-a", under: root)
        try writeReport(
            makeReport(artifactHash: "h2", generatedAt: "2026-07-05T10:00:00Z"),
            dirName: "2026-07-05-variant-robustness-a", under: root)

        // The older report tested exactly these artifact bytes — it wins
        // over the newer report of a since-edited artifact.
        let found = AgentEvidence.latestRobustness(
            variantName: "agent-a", artifactHash: "h1", runsDirectory: root)
        #expect(found?.report.variantArtifactHash == "h1")
    }

    @Test func lookupFallsBackToNameWhenNoHashMatches() throws {
        let root = try makeRunsRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeReport(
            makeReport(artifactHash: "h1", generatedAt: "2026-07-01T10:00:00Z"),
            dirName: "2026-07-01-variant-robustness-a", under: root)
        try writeReport(
            makeReport(artifactHash: nil, generatedAt: "2026-07-05T10:00:00Z"),
            dirName: "2026-07-05-variant-robustness-a", under: root)

        let found = AgentEvidence.latestRobustness(
            variantName: "agent-a", artifactHash: "h3", runsDirectory: root)
        // No report carries h3 — newest name match (the hashless legacy one).
        #expect(found?.report.variantArtifactHash == nil)
        #expect(found?.report.generatedAt == "2026-07-05T10:00:00Z")
    }

    @Test func lookupReturnsNilForUnknownAgent() throws {
        let root = try makeRunsRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try writeReport(
            makeReport(), dirName: "2026-07-01-variant-robustness-a", under: root)
        let found = AgentEvidence.latestRobustness(
            variantName: "agent-nope", artifactHash: nil, runsDirectory: root)
        #expect(found == nil)
    }
}
