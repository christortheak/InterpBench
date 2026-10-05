import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// A study learns what its workspace's compute cannot run while it is being
/// designed, not when a run stops — and is never refused for it.
///
/// One source: `docs/substrate-capabilities.json`, which the generator writes
/// into the shipped science catalog. Each operation carries its execution
/// profile there; `whereItRuns` holds the compute choices, the study
/// declarations each can run (with the advisory sentence), and the rows of
/// the What Runs Where sheet. The cross-engine byte parity of that catalog is
/// `ScienceCatalogTests.shippedCatalogGuidesAndHTTPMatchPythonBytes`; these
/// tests hold what the Mac does with it. Python twin: `test_compute_limits.py`.
///
/// Pure CPU: no model is loaded and nothing runs.
@Suite(.serialized) struct ComputeLimitsTests {

    static let features = ["probeMeasurements", "interventionPolicies", "saeLatentArms", "jlensReadout"]

    static let readout: JSONValue = .object([
        "lensID": .string("a-lens"),
        "lensSHA256": .string(String(repeating: "a", count: 64)),
        "configHash": .string(String(repeating: "b", count: 64)),
        "tokenizerHash": .string(String(repeating: "c", count: 64)),
        "qualificationID": .string("a-qualification"),
        "layers": .array([.number(10)]),
        "topK": .number(5),
    ])

    private func agent(policies: [JSONValue]? = nil) -> ModelVariantArtifact {
        ModelVariantArtifact(
            name: "agent", baseModelID: "test/model", adapters: [],
            injections: [], promptMode: "chatAssistant",
            qwenThinkingEnabled: false, temperature: 0, systemPrompt: "",
            interventionPolicies: policies)
    }

    /// A model-output study that declares exactly one feature.
    private func study(declaring feature: String?) -> ExperimentManifest {
        var manifest = ExperimentManifest(name: "limits", description: "", modelID: "test/model")
        switch feature {
        case "probeMeasurements":
            manifest.probeMeasurements = .object(["probes": .array([])])
        case "interventionPolicies":
            manifest.variantConditions = [
                .init(
                    name: "agent", artifactPath: "runs/model-variants/agent/model-variant.json",
                    artifactHash: String(repeating: "0", count: 64),
                    artifact: agent(policies: [
                        .object(["json": .string("{}"), "sha256": .string(String(repeating: "0", count: 64))])
                    ]))
            ]
        case "saeLatentArms":
            manifest.saeLatentConditions = .array([.object(["name": .string("latent")])])
        case "jlensReadout":
            manifest.jlensReadout = Self.readout
        default: break
        }
        return manifest
    }

    // MARK: - One source, carried by the catalog

    @Test func everyOperationCarriesItsProfileAndTheIndexSaysWhereItRuns() throws {
        let catalog = try ScienceCatalog.catalog()
        let statuses = catalog.whereItRuns.statuses
        for operation in catalog.operations {
            let profile = operation.executionProfile
            #expect(Set(profile.backends.keys) == ["cuda", "mps", "mlx"], "\(operation.id)")
            for backend in profile.backends.values {
                #expect(statuses[backend.status] == backend.label, "\(operation.id)")
            }
            #expect(!profile.runs.isEmpty && !profile.runs.contains("\n"))
        }
        let runs = Dictionary(
            uniqueKeysWithValues: try ScienceCatalog.brief().operations.map { ($0.id, $0.runs) })
        for operation in catalog.operations {
            #expect(runs[operation.id] == operation.executionProfile.runs)
        }
        // Hand-checked against the inventory's statuses.
        #expect(runs["optvec-train"] == "Python engine only")
        #expect(runs["finetune"] == "Python engine or built-in engine")
        #expect(runs["probe-train"] == "CPU only")
    }

    @Test func theMacCommandLineReturnsTheProfileForOneOperation() async throws {
        let outcome = await ExperimentCLIRunner(sink: .discarding).run(
            namespace: "science", ["operation", "probe-capture", "--json"])
        #expect(outcome.envelope.state == .ready)
        guard case .object(let profile)? = outcome.envelope.result?["executionProfile"],
            case .object(let backends)? = profile["backends"]
        else {
            Issue.record("no executionProfile in \(String(describing: outcome.envelope.result))")
            return
        }
        #expect(
            backends["mlx"]
                == .object(["status": .string("unsupported"), "label": .string("no native implementation")]))
        #expect(profile["runs"] == .string("Python engine only"))
    }

    /// The catalog's choices are the app's choices: same ids, names, and
    /// the bindings each writes to `.steerlab/workspace.json`.
    @Test func theCatalogsComputeChoicesAreTheAppsChoices() throws {
        let choices = try #require(ComputeLimits.shipped).computeChoices
        #expect(choices.map(\.id) == ComputeChoice.allCases.map(\.rawValue))
        for (entry, choice) in zip(choices, ComputeChoice.allCases) {
            #expect(entry.title == choice.title)
            #expect(entry.computeSubstrate == choice.binding.rawValue)
            #expect(entry.computeLocation == choice.location?.rawValue)
        }
    }

    // MARK: - The advisory

    @Test(arguments: ComputeLimitsTests.features)
    func eachFeatureIsAdvisedOnTheQuickStartAndNowhereElse(_ feature: String) throws {
        let manifest = study(declaring: feature)
        #expect(ComputeLimits.declaredFeatures(manifest) == [feature])
        let entry = try #require(ComputeLimits.shipped?.studyFeatures.first { $0.id == feature })

        let advice = ComputeLimits.advisories(for: manifest, choice: .macQuickStart)
        #expect(advice == [try #require(entry.advisories[ComputeChoice.macQuickStart.rawValue])])
        let sentence = try #require(advice.first)
        #expect(sentence.hasPrefix("This study declares \(entry.phrase)."))
        #expect(sentence.contains("\u{201C}\(ComputeChoice.macQuickStart.title)\u{201D}"))
        #expect(sentence.contains("Designing and freezing the study are not affected."))
        #expect(sentence.contains(
            "\u{201C}\(ComputeChoice.macFullCapabilities.title)\u{201D} or "
                + "\u{201C}\(ComputeChoice.anotherMachine.title)\u{201D} can run"))

        #expect(ComputeLimits.advisories(for: manifest, choice: .macFullCapabilities).isEmpty)
        #expect(ComputeLimits.advisories(for: manifest, choice: .anotherMachine).isEmpty)
    }

    @Test func aStudyDeclaringNothingNeedsNoAdvice() {
        var manifest = study(declaring: nil)
        manifest.saeLatentConditions = .array([])
        manifest.variantConditions = [
            .init(
                name: "agent", artifactPath: "runs/model-variants/agent/model-variant.json",
                artifactHash: String(repeating: "0", count: 64), artifact: agent(policies: []))
        ]
        #expect(ComputeLimits.declaredFeatures(manifest).isEmpty)
        #expect(ComputeLimits.advisories(for: manifest, choice: .macQuickStart).isEmpty)
        #expect(ComputeLimits.requirement(for: manifest, choice: .macQuickStart) == nil)
    }

    /// A multi-agent study runs a scenario and never arms the model-output
    /// blocks it carries; probe measurements are refused on the run path for
    /// every kind, so they still count.
    @Test func aMultiAgentStudyCountsOnlyWhatItsRunArms() {
        var everything = study(declaring: "jlensReadout")
        everything.probeMeasurements = study(declaring: "probeMeasurements").probeMeasurements
        everything.saeLatentConditions = study(declaring: "saeLatentArms").saeLatentConditions
        everything.variantConditions = study(declaring: "interventionPolicies").variantConditions
        #expect(ComputeLimits.declaredFeatures(everything) == Self.features)
        everything.studyKind = .multiAgent
        #expect(ComputeLimits.declaredFeatures(everything) == ["probeMeasurements"])
    }

    // MARK: - The readiness checklist

    @Test func theChecklistRowFollowsTheWorkspacesComputeAndNeverBlocks() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "compute-limits-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let manifest = study(declaring: "probeMeasurements")

        func row() -> DataRequirement? {
            StudyDataReadiness.requirements(for: manifest, workspaceRoot: root)
                .first { $0.id == "computeChoice" }
        }

        try WorkspaceCompute.declare(ComputeChoice.macQuickStart, root: root)
        let quickStart = try #require(row())
        #expect(quickStart.status == .partial)
        #expect(quickStart.kind == .computeChoice)
        #expect(quickStart.kind.authoringCategory == .compute)
        #expect(quickStart.detail.hasPrefix("This study declares probe measurements."))
        #expect(quickStart.detail.hasSuffix(ComputeLimits.switchClause))
        #expect(!StudyDataReadiness.summary([quickStart]).blockers.contains(quickStart))

        for choice in [ComputeChoice.macFullCapabilities, .anotherMachine] {
            try WorkspaceCompute.declare(choice, root: root)
            let python = try #require(row())
            #expect(python.status == .present)
            #expect(python.detail.contains(choice.title))
            #expect(python.detail.contains("probe measurements"))
        }

        // A study that declares none of these gets no row at all.
        #expect(
            !StudyDataReadiness.requirements(for: study(declaring: nil), workspaceRoot: root)
                .contains { $0.id == "computeChoice" })
    }

    // MARK: - verify, end to end

    /// The real verb in a temporary workspace: the study still verifies, and
    /// the advisory rides along only where the compute cannot run it.
    @Test func verifyAdvisesOnTheQuickStartAndStillVerifies() async throws {
        ExperimentRootOverrideLock.acquire()
        let root = FileManager.default.temporaryDirectory
            .appending(component: "compute-limits-verify-\(UUID().uuidString)")
        ExperimentStore.rootOverride = root
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: root)
            ExperimentRootOverrideLock.release()
        }

        // One plain agent as the arm, pinned by an artifact file written
        // here, so the draft verifies without any concept data.
        var manifest = try ExperimentStore.create(
            name: "limits-run", description: "", modelID: "test/model",
            modelRevision: "abc123")
        let artifact = agent()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(artifact)
        let relativePath = "runs/model-variants/agent/model-variant.json"
        let url = root.appending(path: relativePath)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url)
        manifest.variantConditions = [
            .init(
                name: "agent", artifactPath: relativePath,
                artifactHash: SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
                artifact: artifact)
        ]
        manifest.dtype = "float32"
        manifest.recordTokenIDs = true
        manifest.jlensReadout = Self.readout
        try ExperimentStore.save(manifest)
        let before = try Data(contentsOf: root.appending(path: "experiments/limits-run/experiment.json"))

        func verify() async -> ExperimentCLIOutcome {
            await ExperimentCLIRunner(sink: .discarding).run(
                namespace: "experiment", ["verify", "limits-run", "--json"])
        }
        func computeAdvisories(_ outcome: ExperimentCLIOutcome) -> [SteerLabCLIEnvelope.Advisory] {
            (outcome.envelope.advisories ?? []).filter { $0.code == CLIAdvisory.computeCannotRun.rawValue }
        }

        try WorkspaceCompute.declare(ComputeChoice.macQuickStart, root: root)
        let quickStart = await verify()
        #expect(quickStart.exitCode == 0)
        #expect(quickStart.envelope.result?["verified"] == .bool(true))
        let advice = computeAdvisories(quickStart)
        #expect(advice.count == 1)
        #expect(advice.first?.detail.hasPrefix("This study declares a J-lens readout.") == true)
        #expect(advice.first?.detail.hasSuffix(ComputeLimits.switchClause) == true)

        for choice in [ComputeChoice.macFullCapabilities, .anotherMachine] {
            try WorkspaceCompute.declare(choice, root: root)
            let python = await verify()
            #expect(python.exitCode == 0)
            #expect(computeAdvisories(python).isEmpty, "\(choice)")
        }
        // Advice writes nothing.
        #expect(
            try Data(contentsOf: root.appending(path: "experiments/limits-run/experiment.json"))
                == before)
    }

    // MARK: - What Runs Where reads the same data

    @Test func theSheetsRowsAreTheCatalogsRows() throws {
        let activities = try #require(ComputeLimits.shipped).activities
        #expect(!activities.isEmpty)
        #expect(ComputeGuide.rows.map(\.id) == activities.map(\.id))
        for (row, activity) in zip(ComputeGuide.rows, activities) {
            #expect(row.activity == activity.activity)
            for choice in ComputeChoice.allCases {
                #expect(row.runs(on: choice) == activity.runsOn[choice.rawValue], "\(row.id)")
            }
        }
        // The column headings are the catalog's short names, one per choice.
        #expect(
            ComputeChoice.allCases.map(ComputeGuide.columnTitle)
                == ["Quick start", "Full capabilities", "Another machine"])
    }

    /// A row is whatever the catalog says: change the data, change the row.
    @Test func aRowIsBuiltFromItsCatalogEntryAlone() throws {
        let json = #"{"id": "x", "activity": "Something new", "runsOn": {"mac-quick-start": false, "mac-full-capabilities": true}}"#
        let activity = try JSONDecoder().decode(
            ScienceCatalog.WhereItRuns.Activity.self, from: Data(json.utf8))
        let row = ComputeGuide.Row(activity)
        #expect(row.id == "x" && row.activity == "Something new")
        #expect(!row.runs(on: .macQuickStart))
        #expect(row.runs(on: .macFullCapabilities))
        // A choice the entry does not mention is not claimed.
        #expect(!row.runs(on: .anotherMachine))
        #expect(!row.pythonEngine)
    }

    /// The sentences around the table come from the same data: what "Yes"
    /// claims, and which declarations the quick start cannot run.
    @Test func theWordsAroundTheTableComeFromTheData() throws {
        let shipped = try #require(ComputeLimits.shipped)
        #expect(!shipped.qualifiedAnywhere)
        #expect(ComputeGuide.tableNote.contains("None of these has yet been measured"))
        let help = ComputeGuide.studyDeclarationsHelp
        for feature in shipped.studyFeatures where feature.runsOn[ComputeChoice.macQuickStart.rawValue] == false {
            #expect(help.lowercased().contains(feature.phrase.lowercased()), "\(feature.id)")
        }
        #expect(help.contains("cannot run on \u{201C}\(ComputeChoice.macQuickStart.title)\u{201D}"))
        #expect(!help.contains(ComputeChoice.macFullCapabilities.title))
        #expect(ComputeLimits.listed(["a", "b", "c"]) == "a, b, and c")
        #expect(ComputeLimits.listed(["a", "b"]) == "a and b")
    }
}
