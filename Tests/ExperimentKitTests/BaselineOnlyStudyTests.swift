import Foundation
import Testing

@testable import ExperimentKit

/// A study that measures the baseline agent alone (public issue #4).
///
/// A researcher replicating published studies on an unmodified model wants a
/// study whose only subject is the baseline agent: the pinned base model, with
/// no vectors or adapters, at the study's sampling settings. The sanctioned
/// spelling is a declared `studyType: agentComparison` with no concept, no
/// injection condition, no agent (variant) condition, and no SAE latent arm.
/// Before this suite, such a draft refused at `verify` (and so at freeze) as
/// the empty study, and the only workaround ran every item twice under two
/// identical arms.
///
/// What is pinned here:
///
/// 1. The cross-engine fixture `baseline-only-study.json`, written by the
///    Python engine: this engine gives the same answer for every shape
///    (baseline-only or not, the empty study or not, the arms a run executes),
///    the same sentences, and the same settings-summary Statistics section.
/// 2. Through the CLI: `verify` and `freeze` are clean, carry a
///    `baselineOnlyStudy` advisory, ask for no validation or battery evidence,
///    are not forced, and stamp no capability-battery exemption; the run's
///    start-time refusals all let the shape through to one arm.
/// 3. `analyze` of such a run has no rows and says `baselineOnlyStudy` instead
///    of `emptyAnalysis`.
/// 4. A paired rubric, which has nothing to pair with one arm, is refused at
///    verify under `studyDeclaration` with the per-response route as the
///    repair; a rubric that codes each response verifies. The empty study's
///    repair names the declaration a study of the baseline alone needs.
///
/// Server twin: `Server/tests/test_baseline_only_study.py`.
///
/// Serialized, and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct BaselineOnlyStudyTests {

    // MARK: Harness

    func withTempRoot<T>(_ body: (URL) async throws -> T) async throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "baseline-only-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body(temp)
    }

    @discardableResult
    func invoke(_ namespace: String, _ args: [String]) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(sink: ExperimentCLIRecorder().sink).run(
            namespace: namespace, args)
    }

    static let model = "mlx-community/gemma-3-4b-it-4bit"
    static let revision = String(repeating: "a", count: 40)

    func codes(_ outcome: ExperimentCLIOutcome) -> [String] {
        (outcome.envelope.advisories ?? []).map(\.code)
    }

    /// A draft of the shape, saved in the temp workspace.
    @discardableResult
    func makeBaselineOnlyDraft(_ name: String) throws -> ExperimentManifest {
        var manifest = try ExperimentStore.create(
            name: name, description: "", modelID: Self.model,
            modelRevision: Self.revision)
        manifest.studyType = "agentComparison"
        try ExperimentStore.save(manifest)
        return try ExperimentStore.load(name: name)
    }

    func fixture() throws -> [String: Any] {
        let url = CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cross-engine",
            "baseline-only-study.json")
        return try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: url))
                as? [String: Any])
    }

    /// Writes the fixture's rubrics into the workspace at their paths.
    func writeRubrics(_ fixture: [String: Any], into root: URL) throws {
        let rubrics = try #require(fixture["rubrics"] as? [String: String])
        for (relative, text) in rubrics {
            let url = root.appending(path: relative)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try text.write(to: url, atomically: true, encoding: .utf8)
        }
    }

    func decode(_ object: Any) throws -> ExperimentManifest {
        try JSONDecoder().decode(
            ExperimentManifest.self,
            from: JSONSerialization.data(withJSONObject: object))
    }

    // MARK: - 1. The cross-engine fixture

    @Test func theFixtureIsThisEnginesAnswerForEveryShape() async throws {
        try await withTempRoot { root in
            let fixture = try fixture()
            try writeRubrics(fixture, into: root)
            #expect(
                VerificationRefusal.baselineOnlyPairedRubricViolation(
                    rubricFile: "prompts/rubrics/paired.md")
                    == fixture["pairedRubricViolation"] as? String)
            #expect(ExperimentStore.baselineOnlyStudyAdvisory == fixture["advisory"] as? String)
            #expect(ExperimentStore.baselineOnlySummary == fixture["summary"] as? String)
            let effects = try #require(fixture["effectsSentence"] as? String)
            #expect(RunResults.Model.baselineOnlyEffects.hasPrefix(effects))

            let cases = try #require(fixture["cases"] as? [[String: Any]])
            #expect(cases.count > 5)
            for entry in cases {
                let label = try #require(entry["label"] as? String)
                let manifest = try decode(try #require(entry["manifest"]))
                #expect(
                    ExperimentStore.isBaselineOnlyStudy(manifest)
                        == (entry["baselineOnly"] as? Bool), "\(label)")
                let violations = ExperimentStore.verify(manifest)
                #expect(
                    violations.contains(VerificationRefusal.emptyModelOutputViolation)
                        == (entry["emptyStudy"] as? Bool), "\(label)")
                #expect(
                    violations.contains {
                        $0.hasPrefix(VerificationRefusal.baselineOnlyPairedRubricPrefix)
                    } == (entry["pairedRubricRefused"] as? Bool), "\(label)")
                if let executed = entry["executedConditions"] as? [String] {
                    #expect(
                        ExperimentTasks.effectiveConditionNames(for: manifest)
                            == executed, "\(label)")
                }
            }
        }
    }

    @Test func theSettingsSummaryPromisesNoPairedStatistics() async throws {
        try await withTempRoot { _ in
            let statistics = try #require(
                try fixture()["preregistrationStatistics"] as? [String: [String: Any]])
            #expect(Set(statistics.keys) == ["baselineOnly", "comparison"])
            for (key, entry) in statistics {
                let text = ExperimentStore.preregistrationMarkdown(
                    try decode(try #require(entry["manifest"])))
                let heading = try #require(text.range(of: "## Statistics\n\n"))
                let end = try #require(
                    text.range(of: "\n\n", range: heading.upperBound ..< text.endIndex))
                #expect(
                    String(text[heading.upperBound ..< end.lowerBound])
                        == entry["statisticsSection"] as? String,
                    "\(key): this engine's Statistics section differs")
            }
        }
    }

    // MARK: - 2. Verify, freeze, and the run's start

    @Test func verifyAndFreezeAreCleanAndSayWhatTheStudyMeasures() async throws {
        try await withTempRoot { root in
            try makeBaselineOnlyDraft("replication")
            // A named baseline is not an injection condition, and asks for no
            // validation evidence: there is no vector to validate.
            let declared = await invoke(
                "experiment", ["declare-condition", "replication", "baseline", "--baseline"])
            #expect(declared.exitCode == 0, "\(declared.envelope.message)")
            let withBaseline = try ExperimentStore.load(name: "replication")
            #expect(withBaseline.conditions.map(\.name) == ["baseline"])
            #expect(ExperimentStore.isBaselineOnlyStudy(withBaseline))

            let verified = await invoke("experiment", ["verify", "replication", "--json"])
            #expect(verified.exitCode == 0, "\(verified.envelope.message)")
            #expect(verified.envelope.state == .okWithAdvisories)
            #expect(verified.envelope.result?["violations"] == .array([]))
            #expect(codes(verified).contains(CLIAdvisory.baselineOnlyStudy.rawValue))
            #expect(
                verified.envelope.advisories?.first {
                    $0.code == CLIAdvisory.baselineOnlyStudy.rawValue
                }?.detail == ExperimentStore.baselineOnlyStudyAdvisory)

            let frozen = await invoke("experiment", ["freeze", "replication", "--json"])
            #expect(frozen.exitCode == 0, "\(frozen.envelope.message)")
            #expect(frozen.envelope.result?["forced"] == .bool(false))
            #expect(codes(frozen).contains(CLIAdvisory.baselineOnlyStudy.rawValue))
            #expect(!codes(frozen).contains(CLIAdvisory.freezeGateSkipped.rawValue))
            #expect(!codes(frozen).contains(CLIAdvisory.capabilityControlNotApplied.rawValue))

            // The battery checks whether steering broke the model, and this
            // study steers nothing: the gate is not asked of it, so nothing
            // is exempted, stamped, or forced.
            let stored = try ExperimentStore.load(name: "replication")
            #expect(stored.status == .frozen)
            #expect(stored.freezeForced == nil)
            #expect(stored.forcedGatesSkipped == nil)
            #expect(stored.capabilityBatteryNotApplied == nil)

            let summary = try String(
                contentsOf: root.appending(
                    components: "experiments", "replication",
                    ExperimentStore.preregistrationFilename),
                encoding: .utf8)
            #expect(summary.contains(ExperimentStore.baselineOnlySummary))
            #expect(!summary.contains("same-case baseline"))
        }
    }

    @Test func theRunsStartTimeRefusalsLetTheShapeThroughToOneArm() async throws {
        try await withTempRoot { _ in
            let manifest = try makeBaselineOnlyDraft("replication")
            #expect(ExperimentStore.inertConditionsProblem(manifest) == nil)
            #expect(ExperimentStore.noMeasuredConditionsProblem(manifest) == nil)
            #expect(ExperimentStore.latentArmsNotExecutableProblem(manifest) == nil)
            #expect(ExperimentStore.inertMachineryNote(manifest) == nil)
            #expect(!ExperimentStore.conceptMachineryOperative(manifest))
            #expect(
                ExperimentTasks.ordinaryRunConditions(for: manifest).map(\.name)
                    == ["baseline"])
            #expect(ExperimentTasks.effectiveConditionNames(for: manifest) == ["baseline"])
            // The dead end it used to be: the concept study's refusal names
            // the declaration, and the declaration now verifies.
            var concept = manifest
            concept.studyType = nil
            concept.concepts = [
                .init(name: "fear", stimulusSetHash: "h", options: .init(method: .meanDifference))
            ]
            let advice = try #require(ExperimentStore.noMeasuredConditionsProblem(concept))
            #expect(advice.contains("agentComparison"))
            #expect(advice.contains("baseline agent alone"))
        }
    }

    @Test func anUndeclaredOrConceptStudyDraftIsStillTheEmptyStudy() async throws {
        try await withTempRoot { _ in
            var manifest = try makeBaselineOnlyDraft("replication")
            for declared in [nil, "conceptStudy"] as [String?] {
                manifest.studyType = declared
                try ExperimentStore.save(manifest)
                let outcome = await invoke("experiment", ["verify", "replication", "--json"])
                #expect(outcome.envelope.error?.code == LifecycleGate.emptyStudy.rawValue)
                // The repair names the declaration a study of the baseline
                // alone needs, in this client's terms: this command line has
                // no verb for the study type.
                let repair = outcome.envelope.error?.repairAction ?? ""
                for phrase in [
                    "declare studyType agentComparison",
                    "steerlab-cli pack apply", "Compare agents",
                ] {
                    #expect(repair.contains(phrase), "repair omits '\(phrase)'")
                }
            }
        }
    }

    // MARK: - 4. Judging a study of the baseline alone

    @Test func aPairedRubricIsRefusedAtVerifyWithThePerResponseRoute() async throws {
        try await withTempRoot { root in
            let fixture = try fixture()
            try writeRubrics(fixture, into: root)
            try makeBaselineOnlyDraft("replication")

            let paired = await invoke(
                "experiment",
                [
                    "pin-rubric", "replication", "prompts/rubrics/paired.md",
                    "--judges", "coder:claude",
                ])
            #expect(paired.exitCode == 0, "\(paired.envelope.message)")
            for verb in ["verify", "freeze"] {
                let outcome = await invoke("experiment", [verb, "replication", "--json"])
                #expect(outcome.envelope.exitCode == 65, "\(verb): \(outcome.envelope.message)")
                #expect(
                    outcome.envelope.error?.code == LifecycleGate.studyDeclaration.rawValue,
                    "\(verb)")
                let repair = outcome.envelope.error?.repairAction ?? ""
                #expect(repair.contains("perResponseCoding"), "\(verb)")
                #expect(
                    repair.contains(
                        "steerlab-cli experiment pin-rubric replication prompts/rubrics/"),
                    "\(verb)")
            }
            let refused = await invoke("experiment", ["verify", "replication", "--json"])
            #expect(
                refused.envelope.result?["violations"]
                    == .array([
                        .string(
                            VerificationRefusal.baselineOnlyPairedRubricViolation(
                                rubricFile: "prompts/rubrics/paired.md"))
                    ]))
            #expect(try ExperimentStore.load(name: "replication").status == .draft)

            // The route, followed: a rubric that codes each response.
            let coded = await invoke(
                "experiment",
                [
                    "pin-rubric", "replication", "prompts/rubrics/coded.md",
                    "--judges", "coder:claude",
                ])
            #expect(coded.exitCode == 0, "\(coded.envelope.message)")
            let verified = await invoke("experiment", ["verify", "replication", "--json"])
            #expect(verified.exitCode == 0, "\(verified.envelope.message)")
            #expect(codes(verified).contains(CLIAdvisory.baselineOnlyStudy.rawValue))
        }
    }

    // MARK: - 3. Analyze

    @Test func analyzeHasNoRowsAndSaysBaselineOnlyInsteadOfEmptyAnalysis() async throws {
        try await withTempRoot { _ in
            let manifest = try makeBaselineOnlyDraft("replication")
            let run = ExperimentStore.runsDirectory.appending(
                component: "20261031T000000000Z-exp-replication-run")
            try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
            try JSONEncoder().encode(manifest).write(to: run.appending(component: "experiment.json"))
            try ExperimentStore.manifestHash(manifest).write(
                to: run.appending(component: "experiment-hash.txt"),
                atomically: true, encoding: .utf8)
            var lines = ""
            for (index, promptID) in ["p1", "p2", "p3"].enumerated() {
                lines += #"{"condition": "baseline", "seed": 1, "promptID": "\#(promptID)", "#
                    + #""promptIndex": \#(index), "wordCount": 12, "distinct2": 0.9}"# + "\n"
            }
            try lines.write(
                to: run.appending(component: "generations.jsonl"),
                atomically: true, encoding: .utf8)
            try "{}".write(to: run.appending(component: "report.json"), atomically: true, encoding: .utf8)

            let outcome = await invoke("experiment", ["analyze", "replication", "--json"])
            #expect(outcome.exitCode == 0, "\(outcome.envelope.message)")
            #expect(outcome.envelope.result?["effectSizeCount"] == .number(0))
            #expect(codes(outcome).contains(CLIAdvisory.baselineOnlyStudy.rawValue))
            #expect(!codes(outcome).contains(CLIAdvisory.emptyAnalysis.rawValue))

            // The same records under any other study keep the empty-analysis
            // advisory: there, no effects is a sign something is wrong.
            guard case .string(let analysisPath)? = outcome.envelope.result?["runDirectory"]
            else {
                Issue.record("analyze named no analysis directory")
                return
            }
            let analysis = URL(fileURLWithPath: analysisPath)
            #expect(
                ExperimentTasks.isBaselineOnlyAnalysis(
                    analysisDirectory: analysis, recordCount: 3, conditions: ["baseline"]))
            #expect(
                !ExperimentTasks.isBaselineOnlyAnalysis(
                    analysisDirectory: analysis, recordCount: 3,
                    conditions: ["baseline", "steered"]))
            #expect(
                !ExperimentTasks.isBaselineOnlyAnalysis(
                    analysisDirectory: analysis, recordCount: 0, conditions: []))
        }
    }
}
