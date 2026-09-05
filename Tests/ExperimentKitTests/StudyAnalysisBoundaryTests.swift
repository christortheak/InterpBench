import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// These tests deliberately use explicit repositories, never rootOverride.
struct StudyAnalysisBoundaryTests {
    private func temporaryRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appending(
            component: "analysis-boundary-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    private func manifest() -> ExperimentManifest {
        ExperimentManifest(
            name: "same-study", description: "", modelID: "test/model",
            createdAt: Date(timeIntervalSince1970: 0))
    }

    private func records(shift: Int) -> String {
        """
        {"condition":"baseline","seed":1,"promptID":"p","wordCount":10,"distinct2":0.5,"output":"might"}
        {"condition":"steered","seed":1,"promptID":"p","wordCount":\(10 + shift),"distinct2":0.5,"output":"might might"}
        """
    }

    private func run(root: URL, manifest: ExperimentManifest, text: String) throws -> URL {
        let directory = root.appending(path: "runs/20260905T000000000Z-exp-same-study-run")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(
            to: directory.appending(component: "experiment.json"))
        try Data(ExperimentStore.manifestHash(manifest).utf8).write(
            to: directory.appending(component: "experiment-hash.txt"))
        try Data(text.utf8).write(to: directory.appending(component: "generations.jsonl"))
        try Data("{}".utf8).write(to: directory.appending(component: "report.json"))
        return directory
    }

    private func bytes(in directory: URL) throws -> [String: Data] {
        try Dictionary(
            uniqueKeysWithValues: FileManager.default.contentsOfDirectory(atPath: directory.path)
                .map {
                    ($0, try Data(contentsOf: directory.appending(component: $0)))
                })
    }

    @Test func capturedEvidenceSurvivesSourceRemovalAndSameNamedWorkspacesStaySeparate() throws {
        let first = try temporaryRoot()
        let second = try temporaryRoot()
        defer {
            try? FileManager.default.removeItem(at: first)
            try? FileManager.default.removeItem(at: second)
        }
        let manifest = manifest()
        let source = try run(root: first, manifest: manifest, text: records(shift: 3))
        _ = try run(root: second, manifest: manifest, text: records(shift: 20))
        let a = StudyAnalysisRepository(workspaceRoot: first, promptRoot: first)
        let b = StudyAnalysisRepository(workspaceRoot: second, promptRoot: second)
        let input = try a.loadAnalysis(manifest: manifest, allowUnverifiedEpoch: false)
        let other = try b.loadAnalysis(manifest: manifest, allowUnverifiedEpoch: false)
        try FileManager.default.removeItem(at: source)
        let result = try StudyAnalysisCalculator.analyze(input)
        let otherResult = try StudyAnalysisCalculator.analyze(other)
        #expect(result.entries.first { $0.metric == "wordCount" }?.meanDiff == 3)
        #expect(otherResult.entries.first { $0.metric == "wordCount" }?.meanDiff == 20)
        let artifacts = try StudyAnalysisRendering.analyze(input: input, result: result)
        let report = try JSONDecoder().decode(
            ExperimentTasks.AnalyzeReport.self, from: #require(artifacts.files["analysis.json"]))
        #expect(report.sourceRunExperimentHash == ExperimentStore.manifestHash(manifest))
        #expect(report.sourceRun == source.lastPathComponent)
        #expect(Set(artifacts.files.keys) == ["analysis.json", "effect-sizes.csv"])
    }

    @Test func writerPreservesSourceBytesAndPublishesDistinctProvenanceStampedDirectories() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let manifest = manifest()
        let source = try run(root: root, manifest: manifest, text: records(shift: 4))
        let before = try bytes(in: source)
        let repository = StudyAnalysisRepository(workspaceRoot: root, promptRoot: root)
        let input = try repository.loadAnalysis(manifest: manifest, allowUnverifiedEpoch: false)
        let artifacts = try StudyAnalysisRendering.analyze(
            input: input, result: StudyAnalysisCalculator.analyze(input))
        let writer = StudyAnalysisWriter(workspaceRoot: root)
        let first = try writer.write(artifacts, manifest: manifest, task: .analyze)
        let second = try writer.write(artifacts, manifest: manifest, task: .analyze)
        #expect(first != second)
        #expect(first.deletingLastPathComponent().path == repository.runsDirectory.path)
        #expect(try bytes(in: source) == before)
        for (name, data) in artifacts.files {
            #expect(try Data(contentsOf: first.appending(component: name)) == data)
            #expect(try Data(contentsOf: second.appending(component: name)) == data)
        }
        let snapshot = try JSONDecoder().decode(
            ExperimentManifest.self,
            from: Data(contentsOf: first.appending(component: "experiment.json")))
        #expect(snapshot == manifest)
        let config = try #require(
            JSONSerialization.jsonObject(
                with: Data(contentsOf: first.appending(component: "config.json"))) as? [String: Any]
        )
        #expect(config["runType"] as? String == "analyze")
        #expect(config["experimentHash"] as? String == ExperimentStore.manifestHash(manifest))
        #expect(config["temperature"] is NSNull)
        #expect(config["samplesPerItem"] is NSNull)
        #expect((config["notes"] as? [String: Any])?.isEmpty == true)
    }

    @Test func evidenceReaderUsesExplicitPromptRootAndRetainsPinDriftGate() throws {
        let workspace = try temporaryRoot()
        let prompts = try temporaryRoot()
        defer {
            try? FileManager.default.removeItem(at: workspace)
            try? FileManager.default.removeItem(at: prompts)
        }
        var manifest = manifest()
        let data = Data(
            #"{"id":"p","prompt":"choose","target":"A","attentionCheck":{"expected":"A"}}"#.utf8)
        manifest.taskPromptsFile = "items.jsonl"
        manifest.taskPromptsHash = SHA256.hash(data: data).map { String(format: "%02x", $0) }
            .joined()
        manifest.exclusionRules = [ExclusionRule(rule: ExclusionEngine.ruleFailedAttentionCheck)]
        try data.write(to: prompts.appending(component: "items.jsonl"))
        _ = try run(root: workspace, manifest: manifest, text: records(shift: 1))
        let repository = StudyAnalysisRepository(workspaceRoot: workspace, promptRoot: prompts)
        let input = try repository.loadAnalysis(manifest: manifest, allowUnverifiedEpoch: false)
        #expect(input.exclusionChecks["p"]?.expected == "A")
        #expect(input.declaredTargets == ["p": true])
        try Data("changed".utf8).write(to: prompts.appending(component: "items.jsonl"))
        #expect(throws: ExperimentError.self) {
            try repository.loadAnalysis(manifest: manifest, allowUnverifiedEpoch: false)
        }
    }

    @Test func calculationPreservesNullVersusAbsentEndpointExclusions() throws {
        var manifest = manifest()
        manifest.exclusionRules = [ExclusionRule(rule: ExclusionEngine.ruleUnparseableEndpoint)]
        let text = """
            {"condition":"baseline","seed":1,"promptID":"p","wordCount":10}
            {"condition":"steered","seed":1,"promptID":"p","wordCount":20,"parsedMonths":null}
            {"condition":"baseline","seed":1,"promptID":"q","wordCount":10}
            {"condition":"steered","seed":1,"promptID":"q","wordCount":13}
            """
        let input = StudyAnalysisInput(
            manifest: manifest, sourceRunName: "captured", sourceRunExperimentHash: nil,
            epoch: .init(unverified: true), generations: text, style: nil)
        let result = try StudyAnalysisCalculator.analyze(input)
        #expect(result.exclusions?.excludedRecords == 1)
        #expect(result.sampledCount == 3)
        #expect(result.entries.first { $0.metric == "wordCount" }?.meanDiff == 3)
        let artifacts = try StudyAnalysisRendering.analyze(input: input, result: result)
        #expect(artifacts.files["exclusions.json"] != nil)
        let report = try JSONDecoder().decode(
            ExperimentTasks.AnalyzeReport.self, from: #require(artifacts.files["analysis.json"]))
        #expect(report.epochUnverified == true)
        #expect(report.exclusions?.excludedRecords == 1)
    }

    @Test func styleRescoringUsesCapturedTaxonomyAndRetainsDiagnosticProvenance() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let taxonomy = Data(
            #"{"schemaVersion":1,"name":"captured-style","features":[{"id":"hedge","title":"Hedge","kind":"wordList","patterns":["might"],"normalize":"rawCount"}]}"#
                .utf8)
        var manifest = manifest()
        manifest.reasoningStyleTaxonomyPath = "taxonomy.json"
        manifest.reasoningStyleTaxonomyHash = SHA256.hash(data: taxonomy).map {
            String(format: "%02x", $0)
        }.joined()
        try taxonomy.write(to: root.appending(component: "taxonomy.json"))
        _ = try run(root: root, manifest: manifest, text: records(shift: 1))
        let input = try StudyAnalysisRepository(workspaceRoot: root, promptRoot: root).loadRescore(
            manifest: manifest, runDirectoryName: nil, allowUnverifiedEpoch: false)
        try FileManager.default.removeItem(at: root.appending(component: "taxonomy.json"))
        let result = try StudyAnalysisCalculator.rescoreStyle(input)
        #expect(result.conditions["baseline"]?.features["hedge"]?.mean == 1)
        #expect(result.conditions["steered"]?.features["hedge"]?.mean == 2)
        let artifacts = try StudyAnalysisRendering.rescoreStyle(input: input, result: result)
        let report = try JSONDecoder().decode(
            ExperimentTasks.RescoreStyleReport.self,
            from: #require(artifacts.files["reasoning-style.json"]))
        #expect(report.taxonomyHash == manifest.reasoningStyleTaxonomyHash)
        #expect(report.taxonomyFile == "taxonomy.json")
        #expect(report.diagnosticOnly)
        #expect(
            String(decoding: try #require(artifacts.files["reasoning-style.csv"]), as: UTF8.self)
                == "condition,seed,promptIndex,promptID,rs_hedge\nbaseline,1,0,p,1.0\nsteered,1,0,p,2.0\n"
        )
    }

    @Test func epochRefusalCannotPublishAnAnalysis() throws {
        let root = try temporaryRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let old = manifest()
        _ = try run(root: root, manifest: old, text: records(shift: 1))
        let repository = StudyAnalysisRepository(workspaceRoot: root, promptRoot: root)
        let before = try FileManager.default.contentsOfDirectory(
            atPath: repository.runsDirectory.path)
        var changed = old
        changed.temperature = 0.8
        #expect(throws: ExperimentError.self) {
            try StudyAnalysisWorkflow.analyze(
                manifest: changed, repository: repository, allowUnverifiedEpoch: true)
        }
        #expect(
            try FileManager.default.contentsOfDirectory(atPath: repository.runsDirectory.path)
                == before)
    }
}
