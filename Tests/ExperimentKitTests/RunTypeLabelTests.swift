import Foundation
import Testing

@testable import ExperimentKit

/// The Results browser's study-vocabulary run-type label: sweep runs read as
/// "optimization (screen)" (pure mapping in `RunBrowser.displayRunType`,
/// surfaced through `Item.displayRunType`). Temp directories only.
@Suite struct RunTypeLabelTests {

    // MARK: Pure mapping

    @Test func sweepStampMapsToOptimizationLabel() {
        #expect(
            RunBrowser.displayRunType(stamped: "sweep")
                == RunBrowser.optimizationRunTypeLabel)
        #expect(RunBrowser.optimizationRunTypeLabel == "optimization (screen)")
    }

    @Test func stamplessDirectoryWithSweepArtifactsMapsToOptimizationLabel() {
        #expect(
            RunBrowser.displayRunType(stamped: nil, hasSweepArtifacts: true)
                == RunBrowser.optimizationRunTypeLabel)
    }

    @Test func otherStampsPassThroughVerbatim() {
        for stamp in ["run", "extract", "validate", "variant-robustness", "multi-agent"] {
            #expect(RunBrowser.displayRunType(stamped: stamp) == stamp)
            // The stamp is authoritative even when sweep artifacts sit next
            // to it (e.g. a study run copied into a sweep-shaped folder).
            #expect(
                RunBrowser.displayRunType(stamped: stamp, hasSweepArtifacts: true)
                    == stamp)
        }
    }

    @Test func stamplessDirectoryWithoutSweepArtifactsStaysUnlabeled() {
        #expect(RunBrowser.displayRunType(stamped: nil) == nil)
        #expect(
            RunBrowser.displayRunType(stamped: nil, hasSweepArtifacts: false) == nil)
    }

    // MARK: Shared row detail line (local + remote rows, WS6.1)

    @Test func rowDetailLineShowsRunTypeAndEngineBadges() {
        #expect(
            RunBrowser.rowDetailLine(
                runType: "run", substrate: "python-hf-transformers",
                experiment: "s1", modelID: "org/m")
                == "run · python-hf-transformers · exp s1 · org/m")
        #expect(
            RunBrowser.rowDetailLine(
                runType: "extract", substrate: "swift-mlx",
                experiment: nil, modelID: nil)
                == "extract · swift-mlx")
    }

    @Test func rowDetailLineForStamplessRunRendersLegacyCaption() {
        // Older runs without config.json must render exactly as before.
        #expect(
            RunBrowser.rowDetailLine(
                runType: nil, substrate: nil, experiment: nil, modelID: nil)
                == "no config.json stamp (legacy run type)")
    }

    // MARK: list() integration (temp runs tree)

    private func makeRunsRoot() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appending(
            component: "run-type-label-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        return root
    }

    private func makeRun(
        named name: String, under root: URL,
        configJSON: String? = nil, sweepCSV: Bool = false
    ) throws -> URL {
        let directory = root.appending(component: name)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        if let configJSON {
            try Data(configJSON.utf8).write(
                to: directory.appending(component: RunMetadata.fileName))
        }
        if sweepCSV {
            let csv = "concept,layer,alpha,markerDensity,distinct2,batteryAccuracy\n"
            try Data(csv.utf8).write(
                to: directory.appending(component: "sweep.csv"))
        }
        return directory
    }

    @Test func listLabelsSweepRunsAsOptimizations() throws {
        let root = try makeRunsRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        _ = try makeRun(
            named: "2026-07-01-exp-s1-sweep", under: root,
            configJSON: #"{"runType": "sweep"}"#, sweepCSV: true)
        _ = try makeRun(
            named: "2026-07-02-legacy-sweep", under: root,
            configJSON: nil, sweepCSV: true)
        _ = try makeRun(
            named: "2026-07-03-exp-s1-run", under: root,
            configJSON: #"{"runType": "run"}"#)
        _ = try makeRun(named: "2026-07-04-unstamped", under: root)

        let items = RunBrowser.list(runsDirectory: root)
        let byName = Dictionary(
            uniqueKeysWithValues: items.map { ($0.name, $0) })

        let stamped = try #require(byName["2026-07-01-exp-s1-sweep"])
        #expect(stamped.displayRunType == "optimization (screen)")
        // The raw stamp is never rewritten — display-only vocabulary.
        #expect(stamped.runType == "sweep")

        let legacy = try #require(byName["2026-07-02-legacy-sweep"])
        #expect(legacy.runType == nil)
        #expect(legacy.displayRunType == "optimization (screen)")

        let study = try #require(byName["2026-07-03-exp-s1-run"])
        #expect(study.displayRunType == "run")

        let unstamped = try #require(byName["2026-07-04-unstamped"])
        #expect(unstamped.displayRunType == nil)
    }
}
