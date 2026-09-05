import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// The first in-process tests of the Swift CLI's agent-path verbs
/// (WP0-AGENT-SURFACE-AUDIT §7 step 4).
///
/// Before the move these verbs could only be exercised by spawning the built
/// binary — which needs `DYLD_FRAMEWORK_PATH`, a Metal toolchain, and a real
/// workspace on disk — so the entire refusal surface an agent will drive was
/// untested. `ExperimentCLIRunner` returns its outcome instead of calling
/// `exit`, and takes its output sink by injection, so a test can assert on the
/// exact bytes and the exact exit code.
///
/// Step 4 pinned several assertions that described behaviour it knew to be
/// WRONG, and said so; step 5 changed three of them deliberately and this
/// suite records what moved:
///
/// - `anUnknownFlagIsSTILLSwallowedSilently` → `anUnknownFlagIsAUsageError`.
///   The old pin asserted that `attach … --bogus-flag zzz` pinned french,
///   printed a success line, then died looking for a concept called `zzz`,
///   and that `list --bogus` exited 0. Both are now exit 64 with nothing
///   written (audit §8, P0-4: "a typo'd semantic flag changes study meaning
///   with no signal"). Justification: an undeclared flag is a MALFORMED
///   INVOCATION, never a refusal, so it does not wait for step 7's exit-code
///   migration — `cluster` and bare `steerlab-cli` have exited 64 for it all
///   along.
/// - The same test's "stdout LIES" assertion moves to
///   `aFailedAttachPrintsNoSuccessLine`, which reaches the same refusal
///   through a bad CONCEPT NAME rather than a bad flag (P1-6). Justification:
///   `attach` saves at the end, so the success line described a mutation that
///   never happened.
/// - `experiment verify`'s `VIOLATION:` lines are asserted on stderr, not
///   stdout (audit §2.1, "prose, stdout even on failure").
///
/// Everything else is unchanged, including every exit code in HUMAN mode.
/// The envelope's codes are asserted through `outcome.envelope`.
///
/// Serialized, and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct ExperimentCLITests {

    // MARK: Harness

    /// Runs `body` with the experiment/run roots pointed at a fresh temp
    /// directory. The async twin of `ExperimentRootOverrideLock.withTempRoot`
    /// — the runner's entry point is `async`, so the sync wrapper cannot host
    /// it.
    func withTempRoot<T>(_ body: (URL) async throws -> T) async throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "expcli-\(UUID().uuidString)")
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body(temp)
    }

    /// One invocation, with everything it wrote captured.
    @discardableResult
    func invoke(
        _ namespace: String, _ args: [String], recorder: ExperimentCLIRecorder
    ) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(sink: recorder.sink).run(namespace: namespace, args)
    }

    /// The bytes `main.swift` writes to stderr after the verb returns —
    /// asserted through the renderer, because that is the one place the prose
    /// is now produced.
    func renderedStandardError(_ outcome: ExperimentCLIOutcome) -> String {
        ExperimentCLIRenderer.standardErrorText(outcome) ?? ""
    }

    // MARK: `experiment list` on an empty workspace

    @Test func listOnAnEmptyWorkspaceSaysSoAndSucceeds() async throws {
        try await withTempRoot { _ in
            let recorder = ExperimentCLIRecorder()
            let outcome = await invoke("experiment", ["list"], recorder: recorder)
            #expect(outcome.exitCode == 0)
            #expect(outcome.failure == nil)
            #expect(outcome.verb == "experiment list")
            #expect(recorder.standardOutput == "no experiments\n")
            #expect(recorder.standardError == "")
        }
    }

    // MARK: create → attach round trip

    @Test func createThenAttachPinsTheStimulusHashAndSucceeds() async throws {
        try await withTempRoot { _ in
            let recorder = ExperimentCLIRecorder()
            let created = await invoke(
                "experiment",
                ["create", "demo", "--model", "mlx-community/gemma-3-4b-it-4bit"],
                recorder: recorder)
            #expect(created.exitCode == 0)
            #expect(
                recorder.standardOutput
                    == "created draft 'demo' (model mlx-community/gemma-3-4b-it-4bit)\n")

            // The real committed `french` stimulus set, read-only: concepts
            // resolve against the code checkout, not the overridden root.
            let attachRecorder = ExperimentCLIRecorder()
            let attached = await invoke(
                "experiment", ["attach", "demo", "french"], recorder: attachRecorder)
            #expect(attached.exitCode == 0)
            #expect(attached.failure == nil)

            let hash = try StimulusSet(
                directory: VectorCatalog.conceptsDirectory.appending(component: "french")
            ).hash
            #expect(
                attachRecorder.outputLines.first
                    == "pinned french @ \(hash.prefix(12))… (20+20 stimuli, "
                        + "meanDifference, last token)")

            // The round trip is durable, not just printed.
            let manifest = try ExperimentStore.load(name: "demo")
            #expect(manifest.concepts.map(\.name) == ["french"])
            #expect(manifest.concepts.first?.stimulusSetHash == hash)

            // …and the listing now sees it.
            let listRecorder = ExperimentCLIRecorder()
            await invoke("experiment", ["list"], recorder: listRecorder)
            #expect(
                listRecorder.standardOutput
                    == "demo  [draft]  model=mlx-community/gemma-3-4b-it-4bit  "
                        + "concepts=french  conditions=0\n")
        }
    }

    // MARK: the freeze refusal carries its gate ids through the runner

    @Test func freezeRefusalReachesTheCallerAsGateIDsNotOnlyProse() async throws {
        try await withTempRoot { _ in
            let setup = ExperimentCLIRecorder()
            await invoke(
                "experiment",
                ["create", "gated", "--model", "mlx-community/gemma-3-4b-it-4bit"],
                recorder: setup)
            await invoke("experiment", ["attach", "gated", "french"], recorder: setup)

            let recorder = ExperimentCLIRecorder()
            let outcome = await invoke(
                "experiment", ["freeze", "gated"], recorder: recorder)

            // Exit code and prose are exactly what they were before the move.
            #expect(outcome.exitCode == 1)
            #expect(recorder.standardOutput == "")
            let failure = try #require(outcome.failure)
            #expect(
                renderedStandardError(outcome)
                    == "steerlab-cli experiment: \(failure.reason)\n")
            #expect(failure.repairAction == nil)

            // What is NEW: the gate ids survive the trip out of the runner.
            // Before WP0 they existed only inside freeze's `--force` stamp.
            let gate = try #require(failure.gate)
            let gates = try #require(failure.gates)
            #expect(FreezeGate.vocabulary.contains(gate))
            #expect(gates.allSatisfy { FreezeGate.vocabulary.contains($0) })
            #expect(gates.contains(gate))
            // A draft with no validate run cannot satisfy the evidence gate,
            // and the refusal path evaluates ALL gates, so it is in the list
            // whether or not it is the gate whose prose was thrown.
            #expect(gates.contains(FreezeGate.validateEvidence.rawValue))
            // `gates` is reported in the closed vocabulary's order.
            #expect(
                gates
                    == FreezeGate.vocabulary.filter { gates.contains($0) })

            // …and they reach the ENVELOPE, with `gate` passed explicitly
            // rather than derived from `gates.first` (the step-2 divergence,
            // reconciled at step 5): `gate` names the gate whose prose is in
            // `message`.
            #expect(outcome.envelope.state == .refused)
            #expect(outcome.envelope.exitCode == 65)
            #expect(outcome.envelope.error?.code == "freezeGateFailed")
            #expect(outcome.envelope.error?.gate == gate)
            #expect(outcome.envelope.error?.gates == gates)
            #expect(outcome.envelope.message == failure.reason)
            #expect(outcome.envelope.error?.repairAction.isEmpty == false)
        }
    }

    @Test func aNonGateRefusalCarriesNoGateIDs() async throws {
        try await withTempRoot { _ in
            let recorder = ExperimentCLIRecorder()
            let outcome = await invoke(
                "experiment", ["verify", "nosuch"], recorder: recorder)
            #expect(outcome.exitCode == 1)
            #expect(outcome.failure?.gate == nil)
            #expect(outcome.failure?.gates == nil)
        }
    }

    // MARK: `data check` exits 65 (step 7's one human-mode migration)

    @Test func dataCheckReportsBlockersOnStdoutAndExitsTwo() async throws {
        try await withTempRoot { _ in
            let setup = ExperimentCLIRecorder()
            await invoke(
                "experiment",
                ["create", "needs", "--model", "mlx-community/gemma-3-4b-it-4bit"],
                recorder: setup)
            await invoke("experiment", ["attach", "needs", "french"], recorder: setup)

            let recorder = ExperimentCLIRecorder()
            let outcome = await invoke("data", ["check", "needs"], recorder: recorder)

            // MIGRATED at step 7. `data check` was the only verb in the
            // family that exited anything other than 1 on a bad answer, and
            // its 2 is the one human-mode migration audit §7 row 7 schedules
            // ("`data check` blockers (2 → 65)"); §2.3 asks for it to land
            // once rather than be aliased. Both modes now say 65, and the code
            // moved onto the closed `LifecycleGate` vocabulary.
            #expect(outcome.exitCode == 65)
            #expect(outcome.envelope.exitCode == 65)
            #expect(outcome.envelope.state == .refused)
            #expect(outcome.envelope.error?.code == "dataReadiness")
            // The classified list, with the PATHS an agent must author — the
            // thing the human report gives a person and stdout gave nobody.
            let result = try #require(outcome.envelope.result)
            guard case .array(let blockers)? = result["blockers"] else {
                Issue.record("no blockers array in the envelope")
                return
            }
            #expect(!blockers.isEmpty)
            #expect(
                blockers.contains {
                    guard case .object(let item) = $0,
                        case .string(let path)? = item["path"]
                    else { return false }
                    return path.hasSuffix("validation.jsonl")
                })
            // A mid-verb `exit(2)` printed no trailing error line, and still
            // does not: the outcome carries no failure.
            #expect(outcome.failure == nil)
            #expect(renderedStandardError(outcome) == "")
            // The blocker COUNT is the only thing on stderr; the blockers
            // themselves are stdout-only (recorded for step 5).
            #expect(recorder.standardError.hasPrefix("data check failed: "))
            #expect(recorder.standardError.hasSuffix(" blockers\n"))
            #expect(recorder.standardOutput.contains("[missing] validation set"))
            #expect(recorder.standardOutput.contains("[missing] task prompts"))
            #expect(recorder.outputLines.last?.contains("missing") == true)
        }
    }

    @Test func dataCheckOnAReadyManifestWouldExitZero() async throws {
        // The other half of the exit contract: `data check` is not
        // unconditionally 2. Asserted through the readiness layer's own
        // summary so the test does not need a fully-authored study on disk.
        try await withTempRoot { _ in
            let recorder = ExperimentCLIRecorder()
            let outcome = await invoke("data", ["check", "nosuch"], recorder: recorder)
            // A manifest that does not exist is an ordinary load failure —
            // exit 1, not the blocker code.
            #expect(outcome.exitCode == 1)
            #expect(outcome.failure != nil)
            // In the envelope it is finally distinguishable: `notFound` (66),
            // not the same 1 as a gate refusal and an operational failure.
            #expect(outcome.envelope.state == .notFound)
            #expect(outcome.envelope.exitCode == 66)
        }
    }

    // MARK: unknown verbs and unknown flags — AS THEY ARE TODAY

    @Test func anUnknownExperimentVerbListsTheVerbsAndExitsOne() async throws {
        try await withTempRoot { _ in
            let recorder = ExperimentCLIRecorder()
            let outcome = await invoke(
                "experiment", ["nonsense"], recorder: recorder)
            // PINNED CURRENT BEHAVIOUR, and wrong: a usage error should be 64
            // (audit appendix, "a usage error that should be 64"). Step 5.
            #expect(outcome.exitCode == 1)
            #expect(recorder.standardOutput == "")
            #expect(
                renderedStandardError(outcome)
                    == "steerlab-cli experiment: verbs: list | create "
                        + "| attach | detach | pin-prompts | pin-rubric "
                        + "| declare-condition | set-sweep-selection "
                        + "| set-sweep-grid | set-instruments "
                        + "| set-sampling | set-exclusions "
                        + "| set-system-prompt "
                        + "| set-parser | set-instrument-scope "
                        + "| set-evaluation-sampling "
                        + "| set-style-taxonomy | verify | freeze | duplicate "
                        + "| extract | validate | sweep | run | analyze "
                        + "| rescore-style | evaluate | promote | confirm\n")
        }
    }

    @Test func noVerbAtAllIsTheSameAsAnUnknownVerb() async throws {
        try await withTempRoot { _ in
            let bare = ExperimentCLIRecorder()
            let noVerb = await invoke("experiment", [], recorder: bare)
            let unknown = await invoke("experiment", ["nonsense"], recorder: bare)
            #expect(noVerb.exitCode == unknown.exitCode)
            #expect(noVerb.failure?.reason == unknown.failure?.reason)
            // The verb label degrades to the family name when nothing follows.
            #expect(noVerb.verb == "experiment")
        }
    }

    @Test func everyFamilyAnswersItsOwnUsageLine() async throws {
        try await withTempRoot { _ in
            let recorder = ExperimentCLIRecorder()
            let data = await invoke("data", [], recorder: recorder)
            #expect(data.exitCode == 1)
            #expect(
                renderedStandardError(data)
                    == "steerlab-cli data: usage: data check <experiment>\n")

            let workspace = await invoke("workspace", [], recorder: recorder)
            #expect(workspace.exitCode == 1)
            #expect(
                renderedStandardError(workspace)
                    == "steerlab-cli workspace: usage: workspace init <path>\n")

            let vectors = await invoke("vectors", [], recorder: recorder)
            #expect(vectors.exitCode == 1)
            #expect(renderedStandardError(vectors).hasPrefix("steerlab-cli vectors: usage: "))

            let remote = await invoke("remote", [], recorder: recorder)
            #expect(remote.exitCode == 1)
            #expect(
                renderedStandardError(remote).hasPrefix(
                    "steerlab-cli remote: usage: remote capabilities|"))
        }
    }

    /// WAS `anUnknownFlagIsSTILLSwallowedSilently` (step 4). The old pin is
    /// quoted in the suite's doc comment; this is the fix it predicted.
    @Test func anUnknownFlagIsAUsageErrorInBothModes() async throws {
        try await withTempRoot { _ in
            let setup = ExperimentCLIRecorder()
            await invoke(
                "experiment",
                ["create", "swallow", "--model", "mlx-community/gemma-3-4b-it-4bit"],
                recorder: setup)

            let recorder = ExperimentCLIRecorder()
            let outcome = await invoke(
                "experiment",
                ["attach", "swallow", "french", "--bogus-flag", "zzz"],
                recorder: recorder)

            // 64 in BOTH vocabularies: a malformed invocation was never a
            // refusal, so it does not wait for step 7.
            #expect(outcome.exitCode == 64)
            #expect(outcome.envelope.exitCode == 64)
            #expect(outcome.envelope.state == .blocked)
            #expect(outcome.envelope.error?.code == "unknownFlag")
            // NOTHING was written and nothing was pinned: the refusal lands
            // before the verb runs at all.
            #expect(recorder.standardOutput == "")
            // STEP 11 changed exactly one token of this expectation:
            // `--help` is now a DECLARED flag on every verb, so the repair
            // line names it. Everything else — the code, both exit codes, the
            // empty stdout, the unpinned manifest — is unchanged, which is the
            // point: `--help` became declared, nothing else stopped being 64.
            #expect(
                renderedStandardError(outcome)
                    == "steerlab-cli experiment: experiment attach does not "
                        + "accept --bogus-flag\n  experiment attach accepts: "
                        + "--corpus --extraction-rendering --help --json "
                        + "--method --out --pool-from --project-neutral "
                        + "--reading-position --reference\n")
            let manifest = try ExperimentStore.load(name: "swallow")
            #expect(manifest.concepts.isEmpty)

            // A flag with no value used to be swallowed even more quietly —
            // exit 0. It is the same usage error now.
            let quiet = ExperimentCLIRecorder()
            let listed = await invoke("experiment", ["list", "--bogus"], recorder: quiet)
            #expect(listed.exitCode == 64)
            #expect(listed.envelope.state == .blocked)
            #expect(quiet.standardOutput == "")
        }
    }

    // MARK: `--help` (WP0 step 11)

    /// `--help` runs NOTHING, exits 0, and answers before the verb's own
    /// positional requirements — a caller asking what the arguments are must
    /// not have to supply them first.
    @Test func helpRendersTheDeclaredSurfaceAndRunsNothing() async throws {
        try await withTempRoot { _ in
            let recorder = ExperimentCLIRecorder()
            let outcome = await invoke(
                "experiment", ["freeze", "--help"], recorder: recorder)

            #expect(outcome.exitCode == 0)
            #expect(outcome.envelope.exitCode == 0)
            #expect(outcome.envelope.state == .ready)
            #expect(outcome.envelope.changed == false)
            #expect(outcome.failure == nil)
            #expect(recorder.standardError == "")
            #expect(
                recorder.standardOutput.hasPrefix(
                    "usage: steerlab-cli experiment freeze <name> [--force]"))
            #expect(recorder.standardOutput.contains("--run-substrate <local|server>"))
            // No manifest named `--help` was created, loaded, or touched.
            #expect(ExperimentStore.list().isEmpty)

            // The same page as DATA, so a machine caller never parses columns.
            let flags = outcome.envelope.result?["flags"]
            #expect(flags != nil)
            #expect(outcome.envelope.result?["verb"] == .string("experiment freeze"))
        }
    }

    /// A family with no sub-verb lists its verbs rather than refusing.
    @Test func familyHelpListsEveryVerbAndExitsZero() async throws {
        try await withTempRoot { _ in
            let recorder = ExperimentCLIRecorder()
            let outcome = await invoke("experiment", ["--help"], recorder: recorder)
            #expect(outcome.exitCode == 0)
            #expect(recorder.standardOutput.contains("declare-condition"))
            #expect(recorder.standardOutput.contains("rescore-style"))
        }
    }

    /// The one parsing change of step 11, stated as a test: `--help` is
    /// DECLARED, and everything else is still a 64.
    @Test func helpIsDeclaredAndNeighbouringTyposAreStill64() async throws {
        try await withTempRoot { _ in
            let helped = ExperimentCLIRecorder()
            #expect(
                await invoke("data", ["check", "--help"], recorder: helped).exitCode
                    == 0)

            let typo = ExperimentCLIRecorder()
            let outcome = await invoke(
                "data", ["check", "--hepl"], recorder: typo)
            #expect(outcome.exitCode == 64)
            #expect(outcome.envelope.error?.code == "unknownFlag")
            #expect(
                outcome.envelope.error?.repairAction.contains("--help") == true)
        }
    }

    /// The generator verb agrees with the in-process gate.
    @Test func docsCLIReferenceChecksTheCommittedDocument() async throws {
        try await withTempRoot { _ in
            let recorder = ExperimentCLIRecorder()
            let outcome = await invoke(
                "docs", ["cli-reference", "--check"], recorder: recorder)
            #expect(outcome.exitCode == 0)
            #expect(outcome.envelope.state == .ready)
            #expect(outcome.envelope.changed == false)
        }
    }

    /// A drifted document refuses with a runnable repair rather than dying.
    @Test func docsCLIReferenceRefusesOnDrift() async throws {
        try await withTempRoot { root in
            let drifted = try String(
                contentsOf: #require(CLIReferenceDocument.committedURL),
                encoding: .utf8
            ).replacingOccurrences(
                of: "steerlab-cli experiment verify <name>",
                with: "steerlab-cli experiment verify <name> [--nonsense]")
            let path = root.appending(component: "drifted.md")
            try FileManager.default.createDirectory(
                at: root, withIntermediateDirectories: true)
            try drifted.write(to: path, atomically: true, encoding: .utf8)

            let recorder = ExperimentCLIRecorder()
            let outcome = await invoke(
                "docs", ["cli-reference", "--check", "--path", path.path],
                recorder: recorder)
            #expect(outcome.exitCode == 1)
            #expect(outcome.envelope.state == .refused)
            #expect(outcome.envelope.error?.code == "documentationDrift")
            #expect(
                outcome.envelope.error?.repairAction.contains(
                    "docs cli-reference --write") == true)
        }
    }

    /// WAS the "stdout LIES" half of `anUnknownFlagIsSTILLSwallowedSilently`
    /// (audit §8, P1-6). Same refusal, reached through a bad concept name now
    /// that a bad flag never gets that far.
    @Test func aFailedAttachPrintsNoSuccessLine() async throws {
        try await withTempRoot { _ in
            let setup = ExperimentCLIRecorder()
            await invoke(
                "experiment",
                ["create", "ordered", "--model", "mlx-community/gemma-3-4b-it-4bit"],
                recorder: setup)

            let recorder = ExperimentCLIRecorder()
            let outcome = await invoke(
                "experiment", ["attach", "ordered", "french", "zzz"],
                recorder: recorder)

            #expect(outcome.exitCode == 1)
            // The refusal is the ONLY thing said. `french` resolved, but
            // `attach` saves at the end, so a success line for it described a
            // mutation that never happened.
            #expect(recorder.standardOutput == "")
            // P2-10: BOTH missing files in one shot, and the row shape named
            // — the layout used to be discoverable one invocation at a time.
            #expect(
                outcome.failure?.reason.hasPrefix("stimulus files not found:") == true)
            #expect(
                outcome.failure?.reason.contains("/concepts/zzz/positive.jsonl") == true)
            #expect(
                outcome.failure?.reason.contains("/concepts/zzz/negative.jsonl") == true)
            #expect(outcome.failure?.reason.contains("\"text\"") == true)
            let manifest = try ExperimentStore.load(name: "ordered")
            #expect(manifest.concepts.isEmpty)
        }
    }

    /// The stream fix: a FAILURE reported on stdout is the audit's §2.1 note
    /// "prose, stdout even on failure".
    @Test func verifyViolationsGoToStderrAndTheOKLineStaysOnStdout() async throws {
        try await withTempRoot { _ in
            let setup = ExperimentCLIRecorder()
            await invoke(
                "experiment",
                ["create", "drifted", "--model", "mlx-community/gemma-3-4b-it-4bit"],
                recorder: setup)
            await invoke("experiment", ["attach", "drifted", "french"], recorder: setup)

            // Clean: the OK line is a success and stays on stdout.
            let ok = ExperimentCLIRecorder()
            let clean = await invoke("experiment", ["verify", "drifted"], recorder: ok)
            #expect(clean.exitCode == 0)
            #expect(ok.standardOutput.hasPrefix("OK [draft] —"))
            #expect(ok.standardError == "")

            // Drifted: rewrite the pinned stimulus hash so verify refuses.
            var manifest = try ExperimentStore.load(name: "drifted")
            manifest.concepts[0].stimulusSetHash = String(repeating: "0", count: 64)
            try ExperimentStore.save(manifest)

            let bad = ExperimentCLIRecorder()
            let outcome = await invoke("experiment", ["verify", "drifted"], recorder: bad)
            #expect(outcome.exitCode == 1)
            #expect(bad.standardOutput == "")
            #expect(bad.standardError.contains("VIOLATION: "))
            // …and the envelope names it as a refusal, not a crash.
            #expect(outcome.envelope.state == .refused)
            #expect(outcome.envelope.exitCode == 65)
            #expect(outcome.envelope.error?.code == "pinDrift")
        }
    }

    // MARK: the runner's own contract

    @Test func theRunnerNeverThrowsAndAlwaysNamesTheNamespace() async throws {
        try await withTempRoot { _ in
            let recorder = ExperimentCLIRecorder()
            for namespace in ExperimentCLIRunner.namespaces.sorted() {
                // `init` is the one family whose BARE form is a complete and
                // successful invocation — it takes no sub-verb — so running it
                // here would both break the exit-code expectation and
                // materialize a home layout on whatever machine runs the
                // suite. `HomeLayoutTests` drives it against a temp home.
                if namespace == "init" { continue }
                let outcome = await invoke(namespace, [], recorder: recorder)
                #expect(outcome.namespace == namespace)
                #expect(outcome.exitCode != 0)
            }
        }
    }

    @Test func aFamilyThisRunnerDoesNotOwnIsAUsageError() async {
        // `main.swift` checks `namespaces` first, so this is unreachable from
        // the binary — the runner still answers rather than trapping, because
        // the never-throws contract has to be total.
        let recorder = ExperimentCLIRecorder()
        let outcome = await invoke("cluster", ["status"], recorder: recorder)
        #expect(outcome.exitCode == 64)
        #expect(outcome.failure?.reason == "unknown verb family 'cluster'")
    }

    @Test func theNamespaceSetIsExactlyTheAgentPathFamiliesThatLivedInMain() {
        // Guards the split: `cluster` has its own runner; `serve`,
        // `artifacts`, and `--config` are not on the agent path (audit §2.1)
        // and deliberately stayed behind.
        //
        // STEP 11 adds `docs`, which is a new family rather than a changed
        // one: `docs cli-reference` regenerates this engine's marked regions
        // of `docs/CLI-REFERENCE.md` from the same verb table `--help` renders
        // from. It lives here because it is strict-parsed, enveloped, and
        // driveable in-process like every other agent-path verb — the
        // alternative was a second, untested dispatch in `main.swift`.
        //
        // STEP 12 adds `install`, for the same reason: `steerlab-cli
        // --version` is a REWRITE of `install version`, so the report, its
        // envelope, its `--json` mode, and its `--help` page all come from one
        // verb that tests can drive in process.
        //
        // OPEN-ISSUES §18 adds `panel`, which IS a changed one: the family
        // used to be dispatched from `main.swift` and is now here, because
        // `panel compile` writes a workspace input and pins it into a manifest
        // and had to be strict-parsed, enveloped, and testable in process like
        // every other authoring verb. `list`/`check` came with it so the
        // family has one help page and one output contract.
        //
        // WP1's last item adds `init`, also a new family: the home layout's
        // first-run materialization. It is the only BARE verb on the surface
        // (no sub-verb), and it is here rather than in `main.swift` for the
        // same reason as the rest — strict flags, one envelope, and a body a
        // test can drive against a temp home instead of the real `~/SteerLab`.
        //
        // `authoring` is the newest, and the odd one: it is the only family
        // that touches no manifest and writes nothing into the workspace. It
        // emits a generation prompt for a KIND of missing study data, which
        // is a question about data rather than about a study — and it is on
        // the agent path because an agent meets missing data far more often
        // than it meets a missing verb.
        #expect(
            ExperimentCLIRunner.namespaces
                == [
                    "init", "workspace", "data", "vectors", "remote",
                    "experiment", "docs", "install", "panel", "authoring",
                    // The chat-template capability record (2026-09-05).
                    "model",
                ])
    }

    @Test func theRendererReproducesBothStderrShapes() {
        // One line for a prose refusal…
        let prose = ExperimentCLIOutcome(
            namespace: "experiment", verb: "experiment freeze", exitCode: 1,
            failure: .init(reason: "nope"))
        #expect(
            ExperimentCLIRenderer.standardErrorText(prose)
                == "steerlab-cli experiment: nope\n")

        // …two for a typed `ClusterCLIError`, which `remote` catches and which
        // always names its own repair.
        let typed = ExperimentCLIOutcome(
            namespace: "remote", verb: "remote jobs", exitCode: 13,
            failure: .init(reason: "site not connected", repairAction: "cluster ensure"))
        #expect(
            ExperimentCLIRenderer.standardErrorText(typed)
                == "steerlab-cli remote: site not connected\n  cluster ensure\n")

        // Success writes nothing at all.
        #expect(
            ExperimentCLIRenderer.standardErrorText(
                ExperimentCLIOutcome(
                    namespace: "data", verb: "data check", exitCode: 0)) == nil)
    }

    @Test func theSinkKeepsStreamsSeparateAndOrdered() {
        let recorder = ExperimentCLIRecorder()
        let sink = recorder.sink
        sink.out("one")
        sink.outRaw("two")
        sink.out("", terminator: "")
        sink.err("bad\n")
        sink.out("three")
        #expect(recorder.standardOutput == "one\ntwothree\n")
        #expect(recorder.standardError == "bad\n")
        #expect(recorder.outputLines == ["one", "twothree"])
    }
}
