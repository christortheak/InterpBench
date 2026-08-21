import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// The UI-helper layer for the two newest manifest declarations
/// (`numericParser` + `exclusionRules`): registry listing for the picker,
/// the draft-edit setters (the panel's one manifest-editing pathway), the
/// pre-run attention-check probe, and the Results pane's exclusion-stamp
/// reading + plain-language summary. Pure CPU; view code is exercised by
/// the build.
@Suite(.serialized) struct MeasurementUIHelpersTests {

    func withTempRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "measurement-ui", body)
    }

    private var shippedRegistryURL: URL {
        VectorCatalog.bundledSeedRoot.appending(path: ParserRegistry.registryFile)
    }

    private func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    @discardableResult
    private func writeShippedRegistry(into root: URL) throws -> String {
        let data = try Data(contentsOf: shippedRegistryURL)
        let destination = root.appending(path: ParserRegistry.registryFile)
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try data.write(to: destination)
        return sha256Hex(data)
    }

    // MARK: - ParserRegistryUI listing

    @Test func entriesListTheShippedRegistrySorted() throws {
        try withTempRoot { root in
            try writeShippedRegistry(into: root)
            let entries = ParserRegistryUI.entries()
            #expect(entries.map(\.name) == ["plain-number", "sentencing-months"])
            let sentencing = try #require(
                entries.first { $0.name == "sentencing-months" })
            #expect(sentencing.kind == "durationMonths")
            #expect(sentencing.plainKind == "a duration, read in months")
            #expect(!sentencing.description.isEmpty)
            let number = try #require(entries.first { $0.name == "plain-number" })
            #expect(number.plainKind == "a plain number")
            #expect(ParserRegistryUI.loadProblem() == nil)
        }
    }

    @Test func missingRegistryYieldsEmptyEntriesAndAPlainProblem() throws {
        try withTempRoot { _ in
            #expect(ParserRegistryUI.entries().isEmpty)
            let problem = try #require(ParserRegistryUI.loadProblem())
            #expect(problem.contains("no parser registry exists"))
        }
    }

    // MARK: - setNumericParser (the draft-edit pathway)

    @Test func setNumericParserDeclaresAndPinsAtTheMomentOfAction() throws {
        try withTempRoot { root in
            let hash = try writeShippedRegistry(into: root)
            _ = try ExperimentStore.create(
                name: "parser-set", description: "", modelID: "test/model")
            let updated = try ExperimentStore.setNumericParser(
                "sentencing-months", experimentName: "parser-set")
            #expect(updated.numericParser == "sentencing-months")
            #expect(updated.parserRegistryHash == hash)
            // Reloaded from disk — the write went through save().
            let reloaded = try ExperimentStore.load(name: "parser-set")
            #expect(reloaded.numericParser == "sentencing-months")
            #expect(reloaded.parserRegistryHash == hash)
        }
    }

    @Test func clearingTheParserClearsThePinToo() throws {
        try withTempRoot { root in
            try writeShippedRegistry(into: root)
            _ = try ExperimentStore.create(
                name: "parser-clear", description: "", modelID: "test/model")
            _ = try ExperimentStore.setNumericParser(
                "plain-number", experimentName: "parser-clear")
            let cleared = try ExperimentStore.setNumericParser(
                nil, experimentName: "parser-clear")
            #expect(cleared.numericParser == nil)
            // An unused pin certifies nothing — it must not linger.
            #expect(cleared.parserRegistryHash == nil)
        }
    }

    @Test func settingAnUndefinedParserRefusesWithThePlainReason() throws {
        try withTempRoot { root in
            try writeShippedRegistry(into: root)
            _ = try ExperimentStore.create(
                name: "parser-unknown", description: "", modelID: "test/model")
            #expect {
                try ExperimentStore.setNumericParser(
                    "no-such-parser", experimentName: "parser-unknown")
            } throws: { error in
                let reason = (error as? ExperimentError)?.reason ?? ""
                return reason.contains("no parser named 'no-such-parser'")
            }
            // A refused set leaves the manifest untouched.
            let manifest = try ExperimentStore.load(name: "parser-unknown")
            #expect(manifest.numericParser == nil)
            #expect(manifest.parserRegistryHash == nil)
        }
    }

    @Test func reDeclaringTheSameParserRePinsAfterRegistryEdit() throws {
        try withTempRoot { root in
            try writeShippedRegistry(into: root)
            _ = try ExperimentStore.create(
                name: "parser-repin", description: "", modelID: "test/model")
            _ = try ExperimentStore.setNumericParser(
                "plain-number", experimentName: "parser-repin")
            // Deliberate registry edit (still valid JSON) → drift.
            let registryURL = root.appending(path: ParserRegistry.registryFile)
            var text = try String(contentsOf: registryURL, encoding: .utf8)
            text = text.replacingOccurrences(
                of: "The first plain number", with: "The FIRST plain number")
            try text.write(to: registryURL, atomically: true, encoding: .utf8)
            let drifted = try ExperimentStore.load(name: "parser-repin")
            #expect(ParserRegistryUI.registryDrifted(drifted))
            #expect(
                ParserRegistryUI.problems(for: drifted).contains {
                    $0.contains("changed since pinning")
                })
            // The one-click repair: re-declare the same name.
            let repinned = try ExperimentStore.setNumericParser(
                "plain-number", experimentName: "parser-repin")
            #expect(!ParserRegistryUI.registryDrifted(repinned))
            #expect(ParserRegistryUI.problems(for: repinned).isEmpty)
        }
    }

    // MARK: - setExclusionRules (the draft-edit pathway)

    @Test func setExclusionRulesSavesValidRulesAndNormalizesEmptyToAbsent() throws {
        try withTempRoot { _ in
            _ = try ExperimentStore.create(
                name: "rules-set", description: "", modelID: "test/model")
            let rules = [
                ExclusionRule(rule: "unparseableEndpoint"),
                ExclusionRule(rule: "outOfRange", min: 0, max: 600),
            ]
            let updated = try ExperimentStore.setExclusionRules(
                rules, experimentName: "rules-set")
            #expect(updated.exclusionRules == rules)
            let cleared = try ExperimentStore.setExclusionRules(
                [], experimentName: "rules-set")
            // Empty normalizes to ABSENT — legacy manifests keep their
            // bytes and their content hash.
            #expect(cleared.exclusionRules == nil)
        }
    }

    @Test func setExclusionRulesRefusesMalformedRulesWithEngineWording() throws {
        try withTempRoot { _ in
            _ = try ExperimentStore.create(
                name: "rules-bad", description: "", modelID: "test/model")
            #expect {
                try ExperimentStore.setExclusionRules(
                    [ExclusionRule(rule: "outOfRange")],
                    experimentName: "rules-bad")
            } throws: { error in
                let reason = (error as? ExperimentError)?.reason ?? ""
                return reason.contains("outOfRange declares no bounds")
            }
            #expect(try ExperimentStore.load(name: "rules-bad").exclusionRules == nil)
        }
    }

    // MARK: - Editor copy + attention-check probe

    @Test func editorDescriptionsAreResearcherFacing() {
        #expect(
            ExclusionRulesUI.editorDescription(
                of: ExclusionRule(rule: "failedAttentionCheck"))
                .hasPrefix("Drop answers that failed their item's attention check"))
        #expect(
            ExclusionRulesUI.editorDescription(
                of: ExclusionRule(rule: "unparseableEndpoint"))
                == "Drop answers where no parsedMonths value could be read "
                + "(parse failures).")
        #expect(
            ExclusionRulesUI.editorDescription(
                of: ExclusionRule(rule: "outOfRange", min: 0, max: 600))
                == "Drop answers whose parsedMonths is outside 0 to 600.")
        #expect(
            ExclusionRulesUI.editorDescription(
                of: ExclusionRule(rule: "outOfRange", endpoint: "score", max: 7))
                == "Drop answers whose score is above 7.")
        #expect(ExclusionRulesUI.shortLabel(forRule: "outOfRange") == "out of range")
    }

    @Test func attentionCheckItemCountReadsTheCurrentPromptFile() throws {
        try withTempRoot { root in
            let file = "prompts/studies/probe-items.jsonl"
            let url = root.appending(path: file)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            let jsonl = """
                {"id": "p1", "text": "task one"}
                {"id": "p2", "text": "say YES", "attentionCheck": {"expected": "YES"}}
                {"id": "p3", "text": "task three"}
                """
            try jsonl.write(to: url, atomically: true, encoding: .utf8)
            #expect(
                ExclusionRulesUI.attentionCheckItemCount(taskPromptsFile: file) == 1)
            // Missing file → nil (other readiness rows own that finding).
            #expect(
                ExclusionRulesUI.attentionCheckItemCount(
                    taskPromptsFile: "prompts/studies/absent.jsonl") == nil)
        }
    }

    // MARK: - Results: stamp reading + plain-language summary

    private func makeStamp(
        excluded: Int,
        considered: [String: Int],
        byRule: [String: [String: Int]],
        surviving: [String: Int],
        rules: [ExclusionStamp.ResolvedRule]
    ) -> ExclusionStamp {
        ExclusionStamp(
            rules: rules,
            consideredN: considered,
            excludedByRule: byRule,
            excludedRecords: excluded,
            survivingN: surviving,
            pairwiseDeletion: true,
            note: "engine note")
    }

    private var twoRules: [ExclusionStamp.ResolvedRule] {
        [
            ExclusionStamp.ResolvedRule(
                rule: "failedAttentionCheck", checkedItems: 3,
                description: "failed check"),
            ExclusionStamp.ResolvedRule(
                rule: "unparseableEndpoint", endpoint: "parsedMonths",
                description: "no parse"),
        ]
    }

    @Test func summaryNamesEachFiringRuleAndThePairedDrop() {
        let stamp = makeStamp(
            excluded: 12,
            considered: ["baseline": 120, "steered": 120],
            byRule: [
                "baseline": ["failedAttentionCheck": 3, "unparseableEndpoint": 1],
                "steered": ["failedAttentionCheck": 5, "unparseableEndpoint": 3],
            ],
            surviving: ["baseline": 116, "steered": 112],
            rules: twoRules)
        #expect(
            ExclusionRulesUI.summary(of: stamp)
                == "12 of 240 answers were excluded: 8 failed attention "
                + "checks, 4 could not be parsed. Their paired baseline "
                + "answers were dropped from the affected comparisons too.")
        #expect(
            ExclusionRulesUI.ruleTotals(of: stamp).map(\.count) == [8, 4])
    }

    @Test func zeroExclusionsSaySoExplicitly() {
        let stamp = makeStamp(
            excluded: 0,
            considered: ["baseline": 50, "steered": 50],
            byRule: [
                "baseline": ["failedAttentionCheck": 0, "unparseableEndpoint": 0],
                "steered": ["failedAttentionCheck": 0, "unparseableEndpoint": 0],
            ],
            surviving: ["baseline": 50, "steered": 50],
            rules: twoRules)
        #expect(
            ExclusionRulesUI.summary(of: stamp)
                == "No answers were excluded — 2 declared rules were checked "
                + "against all 100 answers and none fired.")
    }

    @Test func overlappingRuleCountsAreCalledOut() {
        // 3 excluded records, 4 rule firings — one record failed both.
        let stamp = makeStamp(
            excluded: 3,
            considered: ["steered": 10],
            byRule: ["steered": ["failedAttentionCheck": 2, "unparseableEndpoint": 2]],
            surviving: ["steered": 7],
            rules: twoRules)
        let summary = ExclusionRulesUI.summary(of: stamp)
        #expect(summary.contains("3 of 10 answers were excluded"))
        #expect(
            summary.contains(
                "(An answer failing several rules is counted under each.)"))
    }

    @Test func loadStampReadsReportAnalysisAndStandaloneFiles() throws {
        let stamp = makeStamp(
            excluded: 1,
            considered: ["baseline": 4],
            byRule: ["baseline": ["unparseableEndpoint": 1]],
            surviving: ["baseline": 3],
            rules: [
                ExclusionStamp.ResolvedRule(
                    rule: "unparseableEndpoint", endpoint: "parsedMonths",
                    description: "no parse")
            ])
        let encoder = JSONEncoder()
        let stampData = try encoder.encode(stamp)
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "exclusion-stamp-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temp) }

        // report.json wrapper (run-inline stamp).
        let reportDir = temp.appending(component: "run")
        try FileManager.default.createDirectory(
            at: reportDir, withIntermediateDirectories: true)
        let wrapped = try JSONSerialization.data(
            withJSONObject: [
                "experiment": "x",
                "exclusions": JSONSerialization.jsonObject(with: stampData),
            ])
        try wrapped.write(to: reportDir.appending(component: "report.json"))
        #expect(ExclusionRulesUI.loadStamp(runDirectory: reportDir) == stamp)

        // analysis.json wrapper (analyze verb).
        let analyzeDir = temp.appending(component: "analyze")
        try FileManager.default.createDirectory(
            at: analyzeDir, withIntermediateDirectories: true)
        try wrapped.write(to: analyzeDir.appending(component: "analysis.json"))
        #expect(ExclusionRulesUI.loadStamp(runDirectory: analyzeDir) == stamp)

        // Standalone exclusions.json (both engines write it).
        let standaloneDir = temp.appending(component: "standalone")
        try FileManager.default.createDirectory(
            at: standaloneDir, withIntermediateDirectories: true)
        try stampData.write(
            to: standaloneDir.appending(component: "exclusions.json"))
        #expect(ExclusionRulesUI.loadStamp(runDirectory: standaloneDir) == stamp)

        // A run without any stamp: nil, never a fabricated section.
        let emptyDir = temp.appending(component: "empty")
        try FileManager.default.createDirectory(
            at: emptyDir, withIntermediateDirectories: true)
        try Data("{\"experiment\": \"x\"}".utf8).write(
            to: emptyDir.appending(component: "report.json"))
        #expect(ExclusionRulesUI.loadStamp(runDirectory: emptyDir) == nil)
    }
}
