import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// Gate-5 dry run #2's P1/P2/P3 punch list, Swift side.
///
/// Six promises the dry run found unkept. They are grouped here rather than
/// scattered into the suites they touch because each one is a CONTRACT defect
/// of the same kind — the surface said something an agent could act on, and
/// the thing it said was false or unactionable — and reading them together is
/// what makes the class visible.
///
/// Serialized and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct DryRun2PunchListTests {

    // MARK: Harness

    func withTempRoot<T>(_ body: (URL) async throws -> T) async throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "punch2-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        WorkspaceRoot.programmaticOverride = temp
        ExperimentStore.rootOverride = temp
        let concepts = temp.appending(components: "prompts", "concepts")
        try FileManager.default.createDirectory(
            at: concepts, withIntermediateDirectories: true)
        try FileManager.default.copyItem(
            at: CodeResources.compiledCheckoutPath.appending(
                components: "prompts", "concepts", "french"),
            to: concepts.appending(component: "french"))
        defer {
            ExperimentStore.rootOverride = nil
            WorkspaceRoot.programmaticOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body(temp)
    }

    @discardableResult
    func invoke(_ namespace: String, _ args: [String]) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(sink: .discarding).run(namespace: namespace, args)
    }

    // MARK: - #6. The neutral corpus is in the no-git reproducibility floor

    /// `neutralCorpusHash` is a pinned manifest field, `verify()` re-hashes it,
    /// the git gate covers it, and the run bundle packs it — but the `pinned/`
    /// snapshot did not copy it. A frozen study in an unversioned workspace
    /// could therefore lose the bytes that DENOMINATE norm-unit α, which is to
    /// say lose the meaning of every α in the manifest.
    @Test func freezeSnapshotsTheNeutralCorpusThatDenominatesAlpha() async throws {
        try await withTempRoot { root in
            let corpus = root.appending(
                components: "prompts", "neutral", "corpus.jsonl")
            try FileManager.default.createDirectory(
                at: corpus.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try #"{"text": "A neutral sentence."}"#.appending("\n")
                .write(to: corpus, atomically: true, encoding: .utf8)

            var manifest = try ExperimentStore.create(
                name: "alpha", description: "", modelID: "m")
            manifest.modelRevision = String(repeating: "a", count: 40)
            manifest.concepts = [
                ExperimentStore.makeConceptRef(
                    name: "french",
                    stimulusSetHash: String(repeating: "0", count: 64),
                    options: ExtractionOptions())
            ]
            manifest.neutralCorpusHash = String(repeating: "b", count: 64)
            try ExperimentStore.save(manifest)

            // The snapshot alone — freeze's gates are not what is under test.
            try ExperimentStore.snapshotPinnedInputs(for: manifest)

            let snapshot = ExperimentStore.directory.appending(
                components: "alpha", "pinned", "neutral", "corpus.jsonl")
            #expect(FileManager.default.fileExists(atPath: snapshot.path))
            #expect(
                try String(contentsOf: snapshot, encoding: .utf8)
                    == String(contentsOf: corpus, encoding: .utf8),
                "the snapshot must be BYTES, not a path rewrite")

            // …and only when the manifest actually pins one. An unpinned
            // corpus is not a pinned input, so copying it would claim a pin
            // that does not exist.
            var unpinned = manifest
            unpinned.name = "no-alpha"
            unpinned.neutralCorpusHash = nil
            try ExperimentStore.save(unpinned, allowCreate: true)
            try ExperimentStore.snapshotPinnedInputs(for: unpinned)
            #expect(
                !FileManager.default.fileExists(
                    atPath: ExperimentStore.directory.appending(
                        components: "no-alpha", "pinned", "neutral",
                        "corpus.jsonl").path))
        }
    }

    /// WP0 residual (c): the first cut of #6 mirrored the PIN SURFACE's
    /// predicate (`machinery && modelOutput`), which excludes exactly the
    /// studies that keep `neutralCorpusHash` without operating the concept
    /// machinery — a compare-agents study whose promoted agents' α is in norm
    /// units, and a panel carrying the pin forward. Their manifests still NAME
    /// a corpus hash, so the bytes behind it still have to survive.
    @Test func theNeutralCorpusIsSnapshottedByThePinNotByTheMachinery() async throws {
        try await withTempRoot { root in
            let corpus = root.appending(
                components: "prompts", "neutral", "corpus.jsonl")
            try FileManager.default.createDirectory(
                at: corpus.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try #"{"text": "A neutral sentence."}"#.appending("\n")
                .write(to: corpus, atomically: true, encoding: .utf8)

            // A compare-agents study: no concepts, no injection conditions —
            // `conceptMachineryOperative` is false — but α in norm units.
            var agents = try ExperimentStore.create(
                name: "agents", description: "", modelID: "m")
            agents.studyType = "agentComparison"
            agents.neutralCorpusHash = String(repeating: "b", count: 64)
            try ExperimentStore.save(agents)
            #expect(!ExperimentStore.conceptMachineryOperative(agents))
            try ExperimentStore.snapshotPinnedInputs(for: agents)
            #expect(
                FileManager.default.fileExists(
                    atPath: ExperimentStore.directory.appending(
                        components: "agents", "pinned", "neutral",
                        "corpus.jsonl").path),
                "a pinned corpus must be snapshotted with the machinery inert")

            // …and a panel, which is not a modelOutput study at all.
            var panel = agents
            panel.name = "panel"
            panel.studyKind = .multiAgent
            try ExperimentStore.save(panel, allowCreate: true)
            try ExperimentStore.snapshotPinnedInputs(for: panel)
            #expect(
                FileManager.default.fileExists(
                    atPath: ExperimentStore.directory.appending(
                        components: "panel", "pinned", "neutral",
                        "corpus.jsonl").path))
        }
    }

    // MARK: - #7. A repair names the engine that can satisfy the gate

    /// Under `--run-substrate server` the `validateEvidence` gate reads
    /// evidence stamped with the SERVER substrate; evidence from this engine
    /// can never satisfy it. The repair nonetheless said
    /// `steerlab-cli experiment validate` — a repair that cannot work, which
    /// dry run #1 already proved agents follow verbatim.
    @Test func theValidateEvidenceRepairNamesTheRunSubstratesEngine() {
        #expect(
            ExperimentStore.validateCLI(
                forRunSubstrate: ExperimentStore.evidenceSubstrate)
                == "steerlab-cli")
        #expect(
            ExperimentStore.validateCLI(
                forRunSubstrate: WorkspaceScoping.serverSubstrate)
                == "steerlab-server")

        var manifest = ExperimentManifest(name: "sub", description: "", modelID: "m")
        manifest.modelRevision = String(repeating: "a", count: 40)
        manifest.concepts = [
            ExperimentStore.makeConceptRef(
                name: "french", stimulusSetHash: String(repeating: "0", count: 64),
                options: ExtractionOptions())
        ]
        func repair(_ substrate: String) -> String? {
            let table = ExperimentStore.freezeGateTable(
                name: "sub", autoCommit: false, runSubstrate: substrate)
            return table.compactMap { $0.evaluate(manifest) }
                .first { $0.gate == .validateEvidence }?.repairAction
        }
        let local = try! #require(repair(ExperimentStore.evidenceSubstrate))
        let server = try! #require(repair(WorkspaceScoping.serverSubstrate))
        // The SENTENCE is unchanged; only the binary moves.
        #expect(local == "Run 'steerlab-cli experiment validate sub' first, or "
            + "freeze --force to record an unvalidated experiment")
        #expect(server == "Run 'steerlab-server experiment validate sub' first, "
            + "or freeze --force to record an unvalidated experiment")
    }

    /// The vacuous-evidence repair splits: `attach` is authoring (Mac-only),
    /// the trailing `validate` must run where the evidence will count.
    @Test func theVacuousRepairSplitsAuthoringFromEvidence() {
        var manifest = ExperimentManifest(name: "vac2", description: "", modelID: "m")
        manifest.concepts = [
            ExperimentStore.makeConceptRef(
                name: "french", stimulusSetHash: String(repeating: "0", count: 64),
                options: ExtractionOptions())
        ]
        let server = try! #require(
            ExperimentStore.vacuousValidationMachineRepair(
                for: manifest, vacuousConcepts: ["french"],
                runSubstrate: WorkspaceScoping.serverSubstrate))
        #expect(server.contains("steerlab-cli experiment attach vac2 french"))
        #expect(server.contains("steerlab-server experiment validate vac2"))
        // Default is unchanged, so every existing caller keeps its bytes.
        let local = try! #require(
            ExperimentStore.vacuousValidationMachineRepair(
                for: manifest, vacuousConcepts: ["french"]))
        #expect(local.contains("steerlab-cli experiment validate vac2"))
    }

    // MARK: - #9. Two untyped answers become typed ones

    /// Re-freezing was the LAST manifest-writing verb still answering
    /// `verbFailed`/70 — an operational failure, to an agent — where every
    /// other writer already said `statusImmutable`/65.
    @Test func refreezingAFrozenManifestIsStatusImmutable() async throws {
        try await withTempRoot { _ in
            var manifest = try ExperimentStore.create(
                name: "done", description: "", modelID: "m")
            manifest.status = .frozen
            try ExperimentStore.save(manifest, allowCreate: true)
            do {
                _ = try ExperimentStore.freeze(name: "done")
                Issue.record("re-freeze should have refused")
            } catch let error as ExperimentError {
                // Prose byte-stable; the structure is what moved.
                #expect(error.reason == "'done' is already frozen")
                let refusal = try #require(error.lifecycleRefusal)
                #expect(refusal.gate == .statusImmutable)
                #expect(
                    refusal.repairAction.hasPrefix(
                        "steerlab-cli experiment duplicate done done-v2"))
            }
            let outcome = await invoke("experiment", ["freeze", "done"])
            #expect(outcome.envelope.state == .refused)
            #expect(outcome.envelope.exitCode == 65)
            #expect(outcome.envelope.error?.code == LifecycleGate.statusImmutable.rawValue)
            #expect(outcome.envelope.error?.gate == LifecycleGate.statusImmutable.rawValue)
            // The placeholder is substituted with the verb actually typed.
            #expect(
                outcome.envelope.error?.repairAction.contains("experiment freeze done-v2")
                    == true)
        }
    }

    /// An out-of-vocabulary enum value is a MALFORMED INVOCATION (64), not an
    /// operational failure (70): nothing ran, and retrying cannot help — the
    /// caller has to retype. The legal values are the repair.
    @Test func anUnknownInstrumentIsAMalformedInvocation() async throws {
        try await withTempRoot { _ in
            _ = try ExperimentStore.create(
                name: "inst", description: "", modelID: "m")
            let outcome = await invoke(
                "experiment", ["set-instruments", "inst", "answerLogprob"])
            #expect(outcome.envelope.state == .blocked)
            #expect(outcome.envelope.exitCode == 64)
            #expect(outcome.envelope.error?.code == "usage")
            // Not a gate — a malformed invocation is not a refusal.
            #expect(outcome.envelope.error?.gate == nil)
            let repair = try #require(outcome.envelope.error?.repairAction)
            for instrument in ExperimentStore.knownOutcomeInstruments {
                #expect(repair.contains(instrument), "repair omits \(instrument)")
            }
            // Same class for the sibling vocabulary on the same verb.
            let ordinal = await invoke(
                "experiment",
                ["set-instruments", "inst", "ordinalScale",
                 "--ordinal-aggregation", "median"])
            #expect(ordinal.envelope.exitCode == 64)
            #expect(ordinal.envelope.error?.code == "usage")
        }
    }

    // MARK: - WP0 residual (b/a). The firewall behind the declaration gate

    /// `batteryEvidence` is the same class as #7: the gate reads evidence
    /// stamped with the RUN substrate, so under `--run-substrate server` a
    /// `steerlab-cli experiment validate` can never satisfy it.
    @Test func theBatteryEvidenceRepairNamesTheRunSubstratesEngine() {
        var manifest = ExperimentManifest(name: "bat", description: "", modelID: "m")
        manifest.modelRevision = String(repeating: "a", count: 40)
        manifest.variantConditions = [
            .init(
                name: "v1", artifactPath: "runs/model-variants/v1.json",
                artifactHash: String(repeating: "c", count: 64),
                artifact: .init(
                    name: "v1", baseModelID: "m", promptMode: "chatAssistant",
                    qwenThinkingEnabled: false, temperature: 0,
                    systemPrompt: ""))
        ]
        func batteryRepair(_ substrate: String) -> String? {
            ExperimentStore.freezeGateTable(
                name: "bat", autoCommit: false, runSubstrate: substrate)
                .compactMap { $0.evaluate(manifest) }
                .first { $0.gate == .batteryEvidence }?.repairAction
        }
        let local = try! #require(batteryRepair(ExperimentStore.evidenceSubstrate))
        let server = try! #require(
            batteryRepair(WorkspaceScoping.serverSubstrate))
        // The SENTENCE is unchanged; only the binary moves.
        #expect(local == "re-run 'steerlab-cli experiment validate bat' (each "
            + "variant condition runs the pinned battery), or freeze --force")
        #expect(server == "re-run 'steerlab-server experiment validate bat' "
            + "(each variant condition runs the pinned battery), or "
            + "freeze --force")
    }

    /// WP0 residual (a) — THE FIREWALL. The closed instrument vocabulary was
    /// enforced at DECLARATION only, which protects the one path that goes
    /// through `set-instruments`. A hand-edited or imported manifest carries
    /// whatever string it likes, every downstream reader is a set-membership
    /// test, and the study therefore RUNS while measuring nothing. Probed on
    /// the server before the fix: `sampledTxt` produced `state: "ready"`,
    /// exit 0, and a run directory holding 0 generations.
    @Test func anUnknownInstrumentRefusesAtRunStartNotOnlyAtDeclaration() throws {
        var manifest = ExperimentManifest(name: "typo", description: "", modelID: "m")
        // The near miss that motivates the gate: one character.
        manifest.outcomeInstruments = ["sampledTxt"]
        #expect(ExperimentStore.unknownOutcomeInstruments(manifest) == ["sampledTxt"])
        let prompts = [
            ExperimentTasks.StudyPrompt(
                id: "i1", text: "hello", options: nil, target: nil,
                anchorMonths: nil, severity: nil, arm: nil, caseID: nil)
        ]
        do {
            try ExperimentTasks.checkResponseFormats(prompts, manifest: manifest)
            Issue.record("an unknown instrument must refuse at run start")
        } catch let error as ExperimentError {
            let refusal = try #require(error.lifecycleRefusal)
            #expect(refusal.gate == .responseFormat)
            #expect(refusal.reason.contains("'sampledTxt'"))
            // Every legal value is IN the refusal, so the retype needs no
            // second lookup.
            for instrument in ExperimentStore.knownOutcomeInstruments {
                #expect(refusal.reason.contains(instrument))
                #expect(refusal.repairAction.contains(instrument))
            }
            // Authoring is Mac-authority, so both engines name this binary.
            #expect(
                refusal.repairAction.hasPrefix(
                    "steerlab-cli experiment set-instruments typo"))
        }
        // Every KNOWN value still passes, singly and together — the gate is a
        // vocabulary check, not a new restriction on what may be declared.
        for instrument in ExperimentStore.knownOutcomeInstruments {
            var ok = manifest
            ok.outcomeInstruments = [instrument]
            #expect(ExperimentStore.unknownOutcomeInstrumentProblem(ok) == nil)
        }
        var all = manifest
        all.outcomeInstruments = ExperimentStore.knownOutcomeInstruments
        #expect(ExperimentStore.unknownOutcomeInstrumentProblem(all) == nil)
        // …and an undeclared list is the engine default, not a violation.
        var none = manifest
        none.outcomeInstruments = nil
        #expect(ExperimentStore.unknownOutcomeInstrumentProblem(none) == nil)
    }

    // MARK: - #10 / #13. Honest repairs, and a name instead of an errno path

    @Test func theUntypedCatchAllNoLongerClaimsAnInputWasNamed() async throws {
        try await withTempRoot { _ in
            // The no-verb usage error points at --help, not at the roster it
            // has already printed as the reason.
            let unknown = await invoke("experiment", ["nonsense"])
            let repair = try #require(unknown.envelope.error?.repairAction)
            #expect(repair.contains("--help"))
            #expect(!repair.hasPrefix("verbs:"))
        }
    }

    @Test func theManifestShapeIsRecognisedInAMissingPath() {
        #expect(
            ExperimentCLIRunner.experimentName(
                inMissingPath: "/w/experiments/typo/experiment.json") == "typo")
        // Not a manifest: keep the file-shaped answer rather than inventing a
        // study name for a missing rubric.
        #expect(
            ExperimentCLIRunner.experimentName(
                inMissingPath: "/w/prompts/rubrics/missing.md") == nil)
        #expect(
            ExperimentCLIRunner.experimentName(
                inMissingPath: "/w/runs/x/experiment.json") == nil)
        #expect(ExperimentCLIRunner.experimentName(inMissingPath: "") == nil)
    }

    // MARK: - #16. Help and the refusal share one vocabulary

    @Test func setInstrumentsHelpPrintsTheLegalValues() {
        let spec = try! #require(
            ExperimentCLIParser.spec(namespace: "experiment", verb: "set-instruments"))
        let page = ExperimentCLIHelp.text(for: spec)
        for instrument in ExperimentStore.knownOutcomeInstruments {
            #expect(page.contains(instrument), "--help omits \(instrument)")
        }
        for aggregation in ExperimentStore.knownOrdinalAggregations {
            #expect(page.contains(aggregation), "--help omits \(aggregation)")
        }
    }

    // MARK: - #14. `workspace init` answers ABOUT the root it made

    @Test func workspaceInitReportsTheRootItCreated() async throws {
        try await withTempRoot { root in
            let target = root.appending(component: "fresh")
            let outcome = await invoke("workspace", ["init", target.path])
            #expect(outcome.envelope.state == .ready)
            #expect(outcome.envelope.workspace == target.path)
            // Every other verb still reports the RESOLVED root.
            let listed = await invoke("experiment", ["list"])
            #expect(listed.envelope.workspace == root.path)
        }
    }
}
