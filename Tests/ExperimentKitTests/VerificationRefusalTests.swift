import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// A failed `verify` is three different refusals, and they are named apart.
///
/// Observed before this suite existed, on a draft with nothing attached:
///
/// - `steerlab-cli experiment freeze` answered `failed` / 70 / `verbFailed`
///   with the untyped repair ("this was not a typed refusal — read the
///   reason"), while the Python client answered `refused` for the same draft;
/// - `experiment validate` and `experiment verify` answered `pinDrift` and
///   advised restoring files to their pinned bytes, when nothing had been
///   pinned at all.
///
/// What is pinned here:
///
/// 1. The classification itself (`VerificationRefusal`), over sentences copied
///    from BOTH engines' `verify` — real drift must stay `pinDrift`, because
///    callers switch on it — and its agreement with the Python twin.
/// 2. The empty draft through the CLI: `refused`, 65, `emptyStudy`, and one
///    repair naming this client's own commands, identical from `freeze`,
///    `validate`, and `verify`.
/// 3. Real drift through the CLI: still `pinDrift`, with the repair it has
///    always carried.
/// 4. A declaration problem: neither of the above.
///
/// Python twin: `Server/tests/test_verification_refusal.py`.
///
/// Serialized, and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct VerificationRefusalTests {

    // MARK: Harness

    func withTempRoot<T>(_ body: (URL) async throws -> T) async throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "verification-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        // Experiments and runs live in the temp root. The one concept these
        // tests attach is READ from the checkout and never written, so the
        // process-wide workspace override stays untouched.
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body(temp)
    }

    @discardableResult
    func invoke(
        _ namespace: String, _ args: [String],
        recorder: ExperimentCLIRecorder = ExperimentCLIRecorder()
    ) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(sink: recorder.sink).run(
            namespace: namespace, args)
    }

    static let model = "mlx-community/gemma-3-4b-it-4bit"

    /// The refusal `validate`, `extract`, `sweep`, `run`, and `evaluate` give
    /// when the study does not verify. Driven at the task layer, at the one
    /// call they all make: the `validate` verb sets an MLX cache limit before
    /// it verifies, and plain `swift test` has no Metal library to answer it.
    func loadVerifiedRefusal(
        _ name: String, sourceLocation: SourceLocation = #_sourceLocation
    ) throws -> LifecycleRefusal {
        do {
            _ = try ExperimentTasks.loadVerified(name)
        } catch let error as ExperimentError {
            return try #require(error.lifecycleRefusal, sourceLocation: sourceLocation)
        }
        Issue.record("expected '\(name)' to fail verification", sourceLocation: sourceLocation)
        throw ExperimentError(reason: "expected a refusal")
    }

    // MARK: - 1. The classification

    /// Drift sentences as `verify` writes them, from both engines. Each
    /// reports bytes: a pinned file that changed, is gone, or appeared.
    static let driftSentences = [
        // Python engine
        "task prompts 'prompts/tasks/a.jsonl' changed since pinning (have 9711fc1dd6a1…, pinned 000000000000…)",
        "task prompts 'prompts/tasks/a.jsonl': file missing at prompts/tasks/a.jsonl",
        "concept 'french' stimuli changed since pinning (have 9711fc1dd6a1…, pinned 000000000000…)",
        "concept 'french': validation.jsonl appeared after pinning (validationHash pinned null) — re-attach to pin it (found at prompts/concepts/french/validation.jsonl)",
        "concept 'french': pinned validation.jsonl missing at prompts/concepts/french/validation.jsonl",
        "markers.json appeared after pinning (markersHash pinned null) — re-freeze on a duplicate to pin it",
        "pinned markers.json missing — no attached concept has a markers.json anymore",
        "concept 'calm': no stories.jsonl under prompts/emotions/",
        "manifest content changed after freeze (hash mismatch)",
        "pinned parser registry missing at prompts/parsers/registry.json",
        "pinned J-lens 'lens-a' is not importable in this workspace (FileNotFoundError) — import it before running, or the readout cannot be reproduced",
        // Swift engine
        "concept 'french': stimulus files changed since pinning (have 9711fc1dd6a1…, pinned 000000000000…)",
        "concept 'french': stimulus files missing/unreadable",
        "concept 'french': validation.jsonl appeared after attach (pinned as absent) — re-attach to pin it",
        "markersHash is pinned but no attached concept has a markers.json",
        "judge rubric missing (prompts/rubrics/r.md)",
        "variant 'tuned' artifact missing (runs/x/agent.json)",
        "multi-agent scenario changed since pinning (have 9711fc1dd6a1…, pinned 000000000000…)",
    ]

    /// Declaration sentences: the study's own settings, no file's bytes
    /// involved.
    static let declarationSentences = [
        "judge rubric pin is incomplete — judgeRubricFile and judgeRubricHash must both be set",
        "judge rubric is incompletely pinned (need file AND hash)",
        "unknown studyType 'survey' — one of conceptStudy, agentComparison, confirmAgent, multiAgent",
        "samplesPerItem > 1 requires temperature > 0 (greedy decoding makes every sample identical)",
        "condition 'steered': references unattached concept 'calm'",
        "variant 'tuned' uses org/other, not the study model org/m",
    ]

    @Test func classificationNamesTheThreeSituations() {
        for sentence in Self.driftSentences {
            #expect(VerificationRefusal.isDrift(sentence), "\(sentence)")
            #expect(VerificationRefusal.gate([sentence]) == .pinDrift, "\(sentence)")
        }
        for sentence in Self.declarationSentences {
            #expect(!VerificationRefusal.isDrift(sentence), "\(sentence)")
            #expect(
                VerificationRefusal.gate([sentence]) == .studyDeclaration,
                "\(sentence)")
        }
        for sentinel in [
            VerificationRefusal.emptyModelOutputViolation,
            VerificationRefusal.emptyMultiAgentViolation,
        ] {
            #expect(!VerificationRefusal.isDrift(sentinel))
            #expect(VerificationRefusal.gate([sentinel]) == .emptyStudy)
            // Still the empty study when a setting is also wrong: the first
            // thing to do is still to attach something.
            #expect(
                VerificationRefusal.gate([sentinel, Self.declarationSentences[0]])
                    == .emptyStudy)
        }
        // Drift outranks everything else in the list. Callers switch on
        // `pinDrift`, so a refusal that changed its name because a second
        // problem appeared beside the first would be worse than the mislabel
        // being fixed.
        let mixed = [
            VerificationRefusal.emptyModelOutputViolation,
            Self.declarationSentences[0], Self.driftSentences[0],
        ]
        #expect(VerificationRefusal.gate(mixed) == .pinDrift)
        #expect(
            VerificationRefusal.repair(name: "demo", violations: mixed)
                == ExperimentTasks.pinDriftRepair(name: "demo", violations: mixed))
    }

    /// The sentence `verify` actually writes for an empty study is the one the
    /// classifier matches. If the rule's text ever changes, this fails here
    /// rather than quietly sending empty drafts back to `studyDeclaration`.
    @Test func theSentinelsAreWhatVerifyWrites() {
        var manifest = ExperimentManifest(name: "empty", description: "", modelID: "m")
        #expect(
            ExperimentStore.verify(manifest)
                == [VerificationRefusal.emptyModelOutputViolation])
        manifest.studyKind = .multiAgent
        #expect(
            ExperimentStore.verify(manifest)
                == [VerificationRefusal.emptyMultiAgentViolation])
    }

    @Test func theEmptyReasonDropsNothingVerifySaid() {
        let reason = VerificationRefusal.emptyReason(
            name: "demo",
            violations: [
                VerificationRefusal.emptyModelOutputViolation,
                Self.declarationSentences[0],
            ])
        #expect(reason.hasPrefix("'demo' is empty: nothing is attached yet"))
        #expect(reason.contains(Self.declarationSentences[0]))
        #expect(!reason.contains(VerificationRefusal.emptyModelOutputViolation))
    }

    /// The two engines hold twin literals. Ask Python for its answers over the
    /// same cases and compare gate, reason, and repair.
    @Test func classificationAndRepairsMatchPython() throws {
        let repository = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        var cases = (Self.driftSentences + Self.declarationSentences).map { [$0] }
        cases.append([VerificationRefusal.emptyModelOutputViolation])
        cases.append([VerificationRefusal.emptyMultiAgentViolation])
        cases.append([
            VerificationRefusal.emptyModelOutputViolation,
            Self.declarationSentences[0],
        ])
        cases.append([
            VerificationRefusal.emptyModelOutputViolation, Self.driftSentences[0],
        ])
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        let python = ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"] ?? "python3"
        process.arguments = [
            python, "-c",
            """
            import json, sys
            from steerlab_server.experiment import verification_refusal as r
            rows = []
            for case in json.loads(sys.argv[1]):
                gate = r.gate(case)
                rows.append({
                    'gate': gate,
                    'repair': '' if gate == 'pinDrift' else r.repair('demo', case, program='steerlab-cli'),
                    'emptyReason': r.empty_reason('demo', case) if gate == 'emptyStudy' else '',
                })
            print(json.dumps({'rows': rows, 'markers': list(r.DRIFT_MARKERS),
                              'sentinels': [r.EMPTY_MODEL_OUTPUT_VIOLATION, r.EMPTY_MULTI_AGENT_VIOLATION]}))
            """,
            String(decoding: try JSONEncoder().encode(cases), as: UTF8.self),
        ]
        process.currentDirectoryURL = FileManager.default.temporaryDirectory
        process.environment = ProcessInfo.processInfo.environment.merging([
            "PYTHONPATH": repository.appending(path: "Server").path,
            "HF_HUB_OFFLINE": "1",
        ]) { _, new in new }
        let output = Pipe()
        let diagnostics = Pipe()
        process.standardOutput = output
        process.standardError = diagnostics
        try process.run()
        let bytes = output.fileHandleForReading.readDataToEndOfFile()
        let errors = diagnostics.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        #expect(process.terminationStatus == 0, "\(String(decoding: errors, as: UTF8.self))")

        struct Row: Decodable {
            let gate: String
            let repair: String
            let emptyReason: String
        }
        struct Answer: Decodable {
            let rows: [Row]
            let markers: [String]
            let sentinels: [String]
        }
        let answer = try JSONDecoder().decode(Answer.self, from: bytes)
        #expect(answer.markers == VerificationRefusal.driftMarkers)
        #expect(
            answer.sentinels == [
                VerificationRefusal.emptyModelOutputViolation,
                VerificationRefusal.emptyMultiAgentViolation,
            ])
        #expect(answer.rows.count == cases.count)
        for (violations, row) in zip(cases, answer.rows) {
            let gate = VerificationRefusal.gate(violations)
            #expect(gate.rawValue == row.gate, "\(violations)")
            // Drift's repair is each client's own long-standing sentence and
            // is not twinned; the two new repairs are.
            if gate != .pinDrift {
                #expect(
                    VerificationRefusal.repair(name: "demo", violations: violations)
                        == row.repair, "\(violations)")
            }
            if gate == .emptyStudy {
                #expect(
                    VerificationRefusal.emptyReason(name: "demo", violations: violations)
                        == row.emptyReason)
            }
        }
    }

    // MARK: - 2. The empty draft, through the CLI

    @Test func anEmptyDraftIsOneUsableRefusalFromFreezeValidateAndVerify() async throws {
        try await withTempRoot { _ in
            await invoke("experiment", ["create", "demo", "--model", Self.model])

            var refusals: [SteerLabCLIEnvelope.Failure] = []
            for verb in ["freeze", "verify"] {
                let outcome = await invoke("experiment", [verb, "demo", "--json"])
                #expect(outcome.envelope.state == .refused, "\(verb): state")
                #expect(outcome.envelope.exitCode == 65, "\(verb): exit code")
                #expect(outcome.envelope.changed == false, "\(verb): changed")
                let error = try #require(outcome.envelope.error, "\(verb)")
                #expect(error.code == "emptyStudy", "\(verb): code")
                #expect(error.gate == "emptyStudy", "\(verb): gate")
                refusals.append(error)
            }
            // `validate` (and every other verb that verifies before it works).
            let validating = try loadVerifiedRefusal("demo")
            #expect(validating.gate == .emptyStudy)

            // The SAME refusal from all three: reason and repair.
            #expect(Set(refusals.map(\.reason) + [validating.reason]).count == 1)
            #expect(Set(refusals.map(\.repairAction) + [validating.repairAction]).count == 1)

            let reason = refusals[0].reason
            #expect(reason.contains("nothing is attached yet"))
            let repair = refusals[0].repairAction
            // Plain words, and THIS client's commands: a concept, an agent, a
            // template, or the interview.
            for command in [
                "steerlab-cli experiment attach demo <concept>",
                "steerlab-cli experiment attach-agent demo",
                "steerlab-cli design list",
                "steerlab-cli authoring study conceptStudy --json",
            ] {
                #expect(repair.contains(command), "repair omits '\(command)'")
            }
            #expect(!repair.contains("not a typed refusal"))
            #expect(!repair.contains("restore"))

            // `verify` still reports what it found.
            let verified = await invoke("experiment", ["verify", "demo", "--json"])
            #expect(
                verified.envelope.result?["violations"]
                    == .array([.string(VerificationRefusal.emptyModelOutputViolation)]))
            // Neither refusal touched the draft.
            #expect(try ExperimentStore.load(name: "demo").status == .draft)
        }
    }

    /// The repair, followed, repairs it: attach a concept and the same
    /// `verify` passes.
    @Test func followingTheEmptyRepairClearsTheRefusal() async throws {
        try await withTempRoot { _ in
            await invoke("experiment", ["create", "demo", "--model", Self.model])
            let before = await invoke("experiment", ["verify", "demo"])
            #expect(before.envelope.error?.code == "emptyStudy")
            await invoke("experiment", ["attach", "demo", "french"])
            let after = await invoke("experiment", ["verify", "demo"])
            #expect(after.envelope.state == .ready, "\(after.envelope.message)")
        }
    }

    /// Human mode keeps exit 1, as every lifecycle refusal on this client
    /// does, and now says the repair as well as the reason.
    @Test func theEmptyRefusalSaysItsRepairInHumanMode() async throws {
        try await withTempRoot { _ in
            await invoke("experiment", ["create", "demo", "--model", Self.model])

            let frozen = await invoke("experiment", ["freeze", "demo"])
            #expect(frozen.exitCode == 1)
            let spoken = ExperimentCLIRenderer.standardErrorText(frozen) ?? ""
            #expect(spoken.contains("nothing is attached yet"))
            #expect(spoken.contains("steerlab-cli experiment attach demo <concept>"))

            let recorder = ExperimentCLIRecorder()
            let verified = await invoke("experiment", ["verify", "demo"], recorder: recorder)
            #expect(verified.exitCode == 1)
            #expect(recorder.standardOutput == "")
            #expect(recorder.standardError.contains("VIOLATION: "))
            #expect(recorder.standardError.contains("nothing is attached yet"))
            #expect(
                recorder.standardError.contains(
                    "steerlab-cli experiment attach demo <concept>"))
        }
    }

    @Test func anEmptyMultiAgentDraftIsPointedAtAPanel() async throws {
        try await withTempRoot { _ in
            await invoke("experiment", ["create", "demo", "--model", Self.model])
            var manifest = try ExperimentStore.load(name: "demo")
            manifest.studyKind = .multiAgent
            try ExperimentStore.save(manifest)

            let outcome = await invoke("experiment", ["verify", "demo", "--json"])
            #expect(outcome.envelope.exitCode == 65)
            #expect(outcome.envelope.error?.code == "emptyStudy")
            let repair = try #require(outcome.envelope.error?.repairAction)
            #expect(repair.contains("steerlab-cli panel list"))
            #expect(repair.contains("steerlab-cli panel compile"))
            #expect(repair.contains("steerlab-cli authoring study multiAgent --json"))
        }
    }

    // MARK: - 3. Real drift keeps its code and its repair

    @Test func realDriftIsStillPinDriftFromVerifyValidateAndFreeze() async throws {
        try await withTempRoot { _ in
            await invoke("experiment", ["create", "demo", "--model", Self.model])
            await invoke("experiment", ["attach", "demo", "french"])
            var manifest = try ExperimentStore.load(name: "demo")
            manifest.concepts[0].stimulusSetHash = String(repeating: "0", count: 64)
            try ExperimentStore.save(manifest)

            // Byte-for-byte the repair this refusal carried before the split.
            let driftRepair =
                "steerlab-cli experiment verify demo (names every drifted pin) "
                + "; then restore those files to their pinned bytes, or "
                + "steerlab-cli experiment duplicate demo demo-v2 and re-pin on "
                + "the copy"

            let verified = await invoke("experiment", ["verify", "demo", "--json"])
            #expect(verified.envelope.exitCode == 65)
            #expect(verified.envelope.error?.code == "pinDrift")
            #expect(verified.envelope.error?.gate == "pinDrift")
            #expect(
                verified.envelope.error?.reason
                    == "1 pinned input(s) of 'demo' no longer match their hashes")
            #expect(verified.envelope.error?.repairAction == driftRepair)

            let validating = try loadVerifiedRefusal("demo")
            #expect(validating.gate == .pinDrift)
            #expect(
                validating.reason.hasPrefix(
                    "experiment 'demo' failed verification:\n  - "))
            #expect(validating.repairAction == driftRepair)

            // Freeze meets the same drift and now names it the same way; it
            // used to answer `failed` / `verbFailed`.
            let frozen = await invoke("experiment", ["freeze", "demo", "--json"])
            #expect(frozen.envelope.state == .refused)
            #expect(frozen.envelope.exitCode == 65)
            #expect(frozen.envelope.error?.code == "pinDrift")
            #expect(
                frozen.envelope.error?.reason.hasPrefix("cannot freeze 'demo':\n  - ")
                    == true)
            #expect(frozen.envelope.error?.repairAction == driftRepair)
        }
    }

    // MARK: - 4. A declaration problem is neither

    @Test func aDeclarationProblemIsNotCalledDrift() async throws {
        try await withTempRoot { _ in
            await invoke("experiment", ["create", "demo", "--model", Self.model])
            await invoke("experiment", ["attach", "demo", "french"])
            // Half a pin: a rubric file named with no hash. No file changed.
            var manifest = try ExperimentStore.load(name: "demo")
            manifest.judgeRubricFile = "prompts/rubrics/rubric.md"
            manifest.judgeRubricHash = nil
            try ExperimentStore.save(manifest)

            let verified = await invoke("experiment", ["verify", "demo", "--json"])
            #expect(verified.envelope.state == .refused)
            #expect(verified.envelope.exitCode == 65)
            #expect(verified.envelope.error?.code == "studyDeclaration")
            #expect(verified.envelope.error?.gate == "studyDeclaration")
            let reason = try #require(verified.envelope.error?.reason)
            #expect(!reason.contains("no longer match their hashes"))
            #expect(reason.contains("declared settings of 'demo'"))
            let repair = try #require(verified.envelope.error?.repairAction)
            #expect(!repair.contains("restore"))
            #expect(repair.contains("steerlab-cli experiment verify demo"))
            #expect(
                verified.envelope.result?["violations"]
                    == .array([.string("judge rubric is incompletely pinned (need file AND hash)")]))

            let validating = try loadVerifiedRefusal("demo")
            #expect(validating.gate == .studyDeclaration)
            #expect(validating.repairAction == repair)

            // Freeze meets it too, typed where it used to be `verbFailed`.
            let frozen = await invoke("experiment", ["freeze", "demo", "--json"])
            #expect(frozen.envelope.exitCode == 65)
            #expect(frozen.envelope.error?.code == "studyDeclaration")
            #expect(frozen.envelope.error?.repairAction == repair)
        }
    }
}
