import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// A temperature-0 study with several seeds (public issue #9).
///
/// At temperature 0 every seed gives the same greedy response. This engine
/// ran such a study and wrote one identical record per listed seed per item
/// (each stamped `seedInert`); the Python engine refused it only after the
/// queue wait and the model load; and verify and freeze passed it on both.
/// The rule now has one sentence on both engines
/// (`StudySampling.greedySeedListProblem`, Python twin
/// `sampling.greedy_seed_list_problem`): a verify violation on a draft, so
/// freeze refuses, and a run-start refusal on this engine for a study frozen
/// under an older release, which still verifies so its runs stay readable.
/// `set-sampling --seeds` is the repair verb.
///
/// Serialized and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct GreedySeedListTests {

    func withTempRoot<T>(_ body: (URL) async throws -> T) async rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "greedy-seeds-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        WorkspaceRoot.programmaticOverride = temp
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            WorkspaceRoot.programmaticOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body(temp)
    }

    @discardableResult
    func invoke(_ args: [String]) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(sink: .discarding)
            .run(namespace: "experiment", args)
    }

    /// A draft with one concept attached, so verify has nothing else to say.
    func attachedDraft(_ root: URL, _ name: String) async throws {
        try ExperimentStore.create(name: name, description: "", modelID: "test/model")
        let concepts = root.appending(components: "prompts", "concepts")
        try FileManager.default.createDirectory(
            at: concepts, withIntermediateDirectories: true)
        try FileManager.default.copyItem(
            at: CodeResources.compiledCheckoutPath.appending(
                components: "prompts", "concepts", "french"),
            to: concepts.appending(component: "french"))
        #expect(await invoke(["attach", name, "french"]).exitCode == 0)
    }

    static let threeGreedy =
        "at temperature 0 decoding is greedy, so every seed gives the same "
        + "response, and the study's 3 seeds would write 3 identical records "
        + "per item. Set one seed, or raise the temperature above 0 to sample"

    // MARK: - verify

    @Test(arguments: [nil, "manifestSeeds", "derivedSHA256"] as [String?])
    func verifyRefusesAGreedyDraftWithSeveralSeeds(policy: String?) async throws {
        try await withTempRoot { _ in
            var manifest = ExperimentManifest(
                name: "greedy", description: "", modelID: "test/model")
            manifest.temperature = 0
            manifest.seeds = [11, 22, 33]
            manifest.seedPolicy = policy
            #expect(StudySampling.greedySeedListProblem(manifest) == Self.threeGreedy)
            #expect(ExperimentStore.verify(manifest).contains(Self.threeGreedy))
            #expect(VerificationRefusal.gate([Self.threeGreedy]) == .studyDeclaration)
            func greedy(_ study: ExperimentManifest) -> Bool {
                ExperimentStore.verify(study).contains {
                    $0.hasPrefix(StudySampling.greedySeedListPrefix)
                }
            }
            var one = manifest
            one.seeds = [11]
            #expect(!greedy(one))
            var warm = manifest
            warm.temperature = 0.7
            #expect(!greedy(warm))
            // Frozen under an older release: it still verifies, so analysis
            // and evaluation of its existing runs keep opening.
            var frozen = manifest
            frozen.status = .frozen
            #expect(!greedy(frozen))
        }
    }

    /// The advisory keeps only what is still true of a study that runs.
    @Test func theAdvisoryNeverCallsARefusedSettingOneWithNoEffect() {
        var manifest = ExperimentManifest(name: "s", description: "", modelID: "test/model")
        manifest.seedPolicy = "derivedSHA256"
        manifest.temperature = 0
        manifest.seeds = [11, 22, 33]
        #expect(StudySampling.unusedSeedSettings(manifest).isEmpty)
        manifest.seeds = [11]
        #expect(
            StudySampling.unusedSeedSettings(manifest).first?
                .contains("has no effect at temperature 0") == true)
        manifest.temperature = 0.7
        manifest.seeds = [11, 22, 33]
        #expect(
            StudySampling.unusedSeedSettings(manifest).first?
                .contains("(3 seeds) is not used") == true)
    }

    // MARK: - verify, freeze, and the repair verb, through the CLI

    @Test func verifyAndFreezeRefuseAndSetSamplingSeedsRepairsIt() async throws {
        try await withTempRoot { root in
            try await attachedDraft(root, "demo")
            let seeded = await invoke(["set-sampling", "demo", "--seeds", "11,22,33"])
            #expect(seeded.exitCode == 0)
            // Set one flag at a time: the joint rule surfaces at verify.
            for verb in ["verify", "freeze"] {
                let outcome = await invoke([verb, "demo"])
                #expect(outcome.envelope.exitCode == 65, "\(verb)")
                #expect(outcome.envelope.state == .refused, "\(verb)")
                #expect(outcome.envelope.error?.code == "studyDeclaration", "\(verb)")
                // verify lists its violations in the result; freeze says them.
                let listed = outcome.envelope.result?["violations"]
                    == .array([.string(Self.threeGreedy)])
                #expect(
                    listed
                        || outcome.envelope.error?.reason.contains(Self.threeGreedy) == true,
                    "\(verb): \(String(describing: outcome.envelope.result))")
                #expect(
                    outcome.envelope.error?.repairAction.hasPrefix(
                        "Set one seed: steerlab-cli experiment set-sampling demo "
                            + "--seeds <n>, or raise the temperature above 0 to "
                            + "sample. ") == true,
                    "\(verb)")
            }
            #expect(try ExperimentStore.load(name: "demo").status == .draft)
            // The repair, as written, verifies.
            #expect(await invoke(["set-sampling", "demo", "--seeds", "11"]).exitCode == 0)
            #expect(await invoke(["verify", "demo"]).envelope.exitCode == 0)
        }
    }

    // MARK: - set-sampling --seeds

    @Test func theSeedsFlagKeepsOrderAndEchoesTheList() async throws {
        try await withTempRoot { _ in
            try ExperimentStore.create(name: "demo", description: "", modelID: "test/model")
            let outcome = await invoke(
                ["set-sampling", "demo", "--seeds", "33, 11,22", "--temperature", "0.7"])
            #expect(outcome.envelope.exitCode == 0)
            #expect(try ExperimentStore.load(name: "demo").seeds == [33, 11, 22])
            #expect(
                outcome.envelope.result?["seeds"]
                    == .array([.string("33"), .string("11"), .string("22")]))
            #expect(outcome.envelope.message.contains("seeds 33, 11, 22"))
            // The largest seed the manifest can hold survives exactly.
            let largest = await invoke(
                ["set-sampling", "demo", "--seeds", String(UInt64.max)])
            #expect(largest.envelope.exitCode == 0)
            #expect(try ExperimentStore.load(name: "demo").seeds == [UInt64.max])
            #expect(largest.envelope.result?["seeds"] == .array([.string(String(UInt64.max))]))
        }
    }

    @Test(arguments: [
        ("7,7", "duplicate seeds"),
        ("", "the seed list cannot be empty"),
        (" , ", "the seed list cannot be empty"),
        ("1.5", "'1.5' is not a whole number seed"),
        ("-1", "'-1' is not a whole number seed"),
        ("12,abc", "'abc' is not a whole number seed"),
        ("+5", "'+5' is not a whole number seed"),
        ("18446744073709551616", "is not a whole number seed"),
    ])
    func aSeedListOutsideTheRulesIsMalformedAndWritesNothing(
        value: String, words: String
    ) async throws {
        try await withTempRoot { _ in
            try ExperimentStore.create(name: "demo", description: "", modelID: "test/model")
            let before = try ExperimentStore.load(name: "demo")
            let outcome = await invoke(
                ["set-sampling", "demo", "--seeds", value, "--temperature", "0.7"])
            #expect(outcome.envelope.exitCode == 64, "\(value)")
            #expect(outcome.envelope.error?.code == "usage", "\(value)")
            #expect(outcome.envelope.error?.reason.contains(words) == true, "\(value)")
            #expect(
                outcome.envelope.error?.repairAction.contains(
                    "steerlab-cli experiment set-sampling demo --seeds") == true)
            let after = try ExperimentStore.load(name: "demo")
            #expect(after.seeds == before.seeds)
            #expect(after.temperature == before.temperature)  // nothing else moved
        }
    }

    @Test func helpNamesTheSeedsFlag() async throws {
        try await withTempRoot { _ in
            let help = await invoke(["set-sampling", "--help"])
            #expect(help.envelope.exitCode == 0)
            let page = String(
                decoding: try JSONEncoder().encode(help.envelope.result), as: UTF8.self)
            #expect(page.contains("--seeds"), "\(page)")
            #expect(page.contains("<n>[,<n>…]"), "\(page)")
        }
    }

    // MARK: - only sampled text loops over the seeds (review round 1)

    /// A readout reads each item once whatever the seeds, so a readout-only
    /// study at temperature 0 with several seeds ran correctly on this engine
    /// before the rule, and still runs: never refused, told its list is not
    /// used.
    @Test func aReadoutOnlyStudyWithSeveralSeedsIsNotRefused() async throws {
        try await withTempRoot { _ in
            var readout = ExperimentManifest(
                name: "readout", description: "", modelID: "test/model")
            readout.temperature = 0
            readout.seeds = [11, 22, 33]
            readout.outcomeInstruments = ["answerTokenLogprob"]
            for status in [ExperimentManifest.Status.draft, .frozen] {
                readout.status = status
                #expect(StudySampling.greedySeedListProblem(readout) == nil)
                #expect(!ExperimentStore.verify(readout).contains {
                    $0.hasPrefix(StudySampling.greedySeedListPrefix)
                })
                try ExperimentTasks.refuseGreedySeedList(readout)
                #expect(StudySampling.frozenGreedySeedListAdvisory(readout) == nil)
                #expect(
                    StudySampling.unusedSeedSettings(readout) == [
                        "the seed list (3 seeds) is not used: this study records "
                            + "answer-token readouts only, and a readout reads each "
                            + "item once and draws no seed at any temperature. To "
                            + "generate responses that use the seeds, add "
                            + "sampledText to outcomeInstruments."
                    ])
            }
            // Beside sampled text, the text still loops over the seeds.
            var both = readout
            both.status = .draft
            both.outcomeInstruments = ["answerTokenLogprob", "sampledText"]
            #expect(StudySampling.greedySeedListProblem(both) == Self.threeGreedy)
            #expect(ExperimentStore.verify(both).contains(Self.threeGreedy))
        }
    }

    // MARK: - verify warns of a frozen study the rule will refuse (round 1)

    @Test func verifyWarnsOfAFrozenStudyTheSeedRuleRefuses() async throws {
        try await withTempRoot { root in
            try await attachedDraft(root, "demo")
            var manifest = try ExperimentStore.load(name: "demo")
            manifest.seeds = [11, 22, 33]
            manifest.status = .frozen
            manifest.freezeHash = ExperimentStore.manifestHash(manifest)
            try ExperimentStore.save(manifest)
            let verified = await invoke(["verify", "demo"])
            #expect(verified.envelope.exitCode == 0)
            #expect(verified.envelope.state == .okWithAdvisories)
            let notes = (verified.envelope.advisories ?? []).filter {
                $0.code == CLIAdvisory.refusedWhenRun.rawValue
            }
            #expect(
                notes.map(\.detail) == [
                    "this study will be refused when it is submitted or run, "
                        + "because \(Self.threeGreedy). To run it: "
                        + "steerlab-cli experiment duplicate demo demo-v2 && "
                        + "steerlab-cli experiment set-sampling demo-v2 --seeds <n>"
                        + "  (one seed; or raise the temperature above 0 to sample)"
                ])
            // A draft is refused at verify instead, with no such advisory.
            var draft = manifest
            draft.status = .draft
            #expect(StudySampling.frozenGreedySeedListAdvisory(draft) == nil)
        }
    }

    // MARK: - the client refuses before submitting (review round 1)

    /// Asked of every server, whatever its capabilities, so one from before
    /// the rule is never reached; a pipeline only when its chain runs the
    /// study. Python twin: `sampling.submission_greedy_seed_list_problem`.
    @Test func theSubmissionRuleFollowsTheVerbAndTheChain() {
        var manifest = ExperimentManifest(name: "demo", description: "", modelID: "test/model")
        manifest.temperature = 0
        manifest.seeds = [11, 22, 33]
        #expect(
            SubstrateRouting.greedySeedListSubmissionRefusal(manifest: manifest, verb: "run")
                == Self.threeGreedy)
        #expect(
            SubstrateRouting.greedySeedListSubmissionRefusal(manifest: manifest, verb: "analyze")
                == nil)
        // No block: the engine refuses the block itself.
        #expect(
            SubstrateRouting.greedySeedListSubmissionRefusal(manifest: manifest, verb: "pipeline")
                == nil)
        manifest.pipeline = .object(["stages": .array([.string("run"), .string("analyze")])])
        #expect(
            SubstrateRouting.greedySeedListSubmissionRefusal(manifest: manifest, verb: "pipeline")
                == Self.threeGreedy)
        manifest.pipeline = .object(["stages": .array([.string("extract"), .string("validate")])])
        #expect(
            SubstrateRouting.greedySeedListSubmissionRefusal(manifest: manifest, verb: "pipeline")
                == nil)
        // A readout-only study runs, so its submission is not refused.
        manifest.outcomeInstruments = ["answerTokenLogprob"]
        #expect(
            SubstrateRouting.greedySeedListSubmissionRefusal(manifest: manifest, verb: "run")
                == nil)
    }

    /// `remote submit-bundle`'s guard: the staged bundle's study is read
    /// from this workspace when its name and content hash match, and refused
    /// before anything is sent.
    @Test func aStagedBundleOfAGreedyStudyIsRefusedBeforeSubmitting() async throws {
        try await withTempRoot { root in
            try await attachedDraft(root, "demo")
            var manifest = try ExperimentStore.load(name: "demo")
            manifest.seeds = [11, 22, 33]
            manifest.status = .frozen
            manifest.freezeHash = ExperimentStore.manifestHash(manifest)
            try ExperimentStore.save(manifest)
            let client = ClusterClient(
                profile: .init(baseURL: URL(string: "https://runner.example.invalid")!))
            let staged = JSONValue.object([
                "experiment": .string("demo"),
                "experimentContentHash": .string(ExperimentStore.manifestHash(manifest)),
            ])
            #expect {
                try client.requireGreedySeedList(inspectedBundle: staged, verb: "run")
            } throws: { error in
                let refusal = (error as? ExperimentError)?.lifecycleRefusal
                return refusal?.gate == .studyDeclaration
                    && refusal?.reason == Self.threeGreedy
                    && refusal?.repairAction.hasPrefix(
                        "steerlab-cli experiment duplicate demo demo-v2 && ") == true
            }
            try client.requireGreedySeedList(inspectedBundle: staged, verb: "analyze")
            try client.requireGreedySeedList(
                inspectedBundle: .object([
                    "experiment": .string("demo"),
                    "experimentContentHash": .string(String(repeating: "0", count: 64)),
                ]), verb: "run")
        }
    }

    // MARK: - run start: the backstop for a study frozen under an older release

    @Test func runStartRefusesAStudyFrozenUnderAnOlderRelease() async throws {
        try await withTempRoot { root in
            try await attachedDraft(root, "demo")
            // As an older release froze it: a sound freeze hash, a rule it
            // did not have.
            var manifest = try ExperimentStore.load(name: "demo")
            manifest.seeds = [11, 22, 33]
            manifest.status = .frozen
            manifest.freezeHash = ExperimentStore.manifestHash(manifest)
            try ExperimentStore.save(manifest)
            #expect(ExperimentStore.verify(try ExperimentStore.load(name: "demo")).isEmpty)

            #expect {
                try ExperimentTasks.refuseGreedySeedList(manifest)
            } throws: { error in
                let refusal = (error as? ExperimentError)?.lifecycleRefusal
                return refusal?.gate == .studyDeclaration
                    && refusal?.reason == Self.threeGreedy
                    && refusal?.repairAction
                        == "steerlab-cli experiment duplicate demo demo-v2 && "
                        + "steerlab-cli experiment set-sampling demo-v2 --seeds <n>"
                        + "  (one seed; or raise the temperature above 0 to sample)"
            }
            // Through the run itself: refused before anything is read or
            // loaded, so no run directory exists.
            await #expect {
                try await ExperimentTasks.run(experimentName: "demo")
            } throws: { error in
                (error as? ExperimentError)?.lifecycleRefusal?.gate == .studyDeclaration
            }
            let runs = root.appending(component: "runs")
            let written = (try? FileManager.default.contentsOfDirectory(atPath: runs.path)) ?? []
            #expect(!written.contains { $0.contains("exp-demo-run") }, "\(written)")
            // One seed runs (the check is silent).
            var one = manifest
            one.seeds = [11]
            try ExperimentTasks.refuseGreedySeedList(one)
        }
    }
}
