import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// `experiment set-evaluation-sampling` — the writer for the manifest's
/// `evaluationSampling` declaration, and the demotion of the evaluate sample
/// flags to a cross-check on a study that carries one.
///
/// THE RULING (review round 12, finding 4). The seeded evaluate subsample
/// shipped as CLI flags plus run stamps. A stamp records what HAPPENED, and
/// "preregistered" is a claim about what was decided BEFORE anything ran — a
/// claim that has to live in the artifact chain to be evidence. The reviewer
/// asked for a frozen, hashed design document; that does not fit this house's
/// flow, because judged re-measurement deliberately runs on never-frozen
/// duplicates. The adapted remedy keeps the substance: the design becomes a
/// DRAFT MANIFEST DECLARATION, and every run stamps the manifest snapshot
/// into its own `experiment.json`. `theDeclarationLandsInTheRunsOwnSnapshot`
/// below is the proof — it is the assertion the whole feature exists for.
///
/// Server twin: `Server/tests/test_evaluation_sampling.py`.
///
/// Serialized and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct EvaluationSamplingDeclarationTests {

    // MARK: Harness

    func withTempRoot<T>(_ body: (URL) async throws -> T) async rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "eval-sampling-\(UUID().uuidString)")
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
    func draft(_ name: String = "demo") throws -> ExperimentManifest {
        try ExperimentStore.create(
            name: name, description: "", modelID: "test/model")
    }

    @discardableResult
    func invoke(_ args: [String]) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(sink: .discarding)
            .run(namespace: "experiment", args)
    }

    // MARK: - The declaration

    /// The design is stored with the rule DERIVED, not typed — the same
    /// guarantee `parserRegistryHash` carries, and the reason there is no
    /// `--rule` flag on any surface.
    @Test func theDeclarationStoresTheDesignAndDerivesItsRule() async throws {
        try await withTempRoot { _ in
            try draft()
            let outcome = await invoke(
                ["set-evaluation-sampling", "demo", "2400", "0x2a"])
            #expect(outcome.envelope.state == .ready)
            #expect(outcome.envelope.exitCode == 0)
            #expect(outcome.envelope.result?["samplePerCondition"] == .number(2400))
            #expect(
                outcome.envelope.result?["sampleSeed"]
                    == .string("0x000000000000002a"))
            // Echoed in FULL: it is the derivation a reader recomputes the
            // membership from, so a truncated one would certify nothing.
            #expect(
                outcome.envelope.result?["rule"]
                    == .string(EvaluateSubsample.rule))
            // The MANIFEST, not the echo, is the contract.
            let design = try #require(
                try ExperimentStore.load(name: "demo").evaluationSampling)
            #expect(design.samplePerCondition == 2400)
            #expect(design.sampleSeed == "0x000000000000002a")
            #expect(design.rule == EvaluateSubsample.rule)
        }
    }

    /// A decimal seed means the decimal a researcher typed, and the stored
    /// spelling is the canonical hex every stamp carries — JSON has no
    /// unsigned 64-bit integer, and a decimal a reader's parser rounds is a
    /// seed that no longer redraws its own subsample.
    @Test func theStoredSeedIsTheCanonicalSpelling() async throws {
        try await withTempRoot { _ in
            try draft()
            await invoke(["set-evaluation-sampling", "demo", "40", "1234"])
            let design = try #require(
                try ExperimentStore.load(name: "demo").evaluationSampling)
            #expect(design.sampleSeed == "0x00000000000004d2")
        }
    }

    /// `""` clears — the affordance every declaration verb here carries, so a
    /// superseded design is removable without hand-editing the manifest.
    @Test func theEmptyStringClearsTheDeclaration() async throws {
        try await withTempRoot { _ in
            try draft()
            await invoke(["set-evaluation-sampling", "demo", "2400", "0x2a"])
            let outcome = await invoke(["set-evaluation-sampling", "demo", ""])
            #expect(outcome.envelope.exitCode == 0)
            #expect(outcome.envelope.result?["samplePerCondition"] == .null)
            #expect(outcome.envelope.result?["sampleSeed"] == .null)
            #expect(outcome.envelope.result?["rule"] == .null)
            let loaded = try ExperimentStore.load(name: "demo")
            #expect(loaded.evaluationSampling == nil)
            #expect(
                EvaluateSubsample.declarationViolations(
                    loaded.evaluationSampling
                ).isEmpty)
        }
    }

    /// Both halves or neither, INSIDE the declaration exactly as at the
    /// flags: a design nobody can redraw is not a preregistration, and a seed
    /// with no size is a stamp on a design it did not shape. Malformed (64),
    /// and nothing is written.
    @Test func halfADeclarationRefusesAndWritesNothing() async throws {
        try await withTempRoot { _ in
            try draft()
            let noSeed = await invoke(
                ["set-evaluation-sampling", "demo", "2400"])
            #expect(noSeed.envelope.state == .blocked)
            #expect(noSeed.envelope.exitCode == 64)
            #expect(noSeed.envelope.error?.code == "usage")
            #expect(noSeed.envelope.error?.gate == nil)
            #expect(
                noSeed.envelope.error?.reason
                    == "the sampling design named 2400 record(s) per "
                    + "condition with no seed — a subsample nobody can redraw "
                    + "is not a preregistration, so the declaration refuses "
                    + "rather than choosing a seed for you")
            #expect(try ExperimentStore.load(name: "demo")
                .evaluationSampling == nil)

            let noSize = await invoke(
                ["set-evaluation-sampling", "demo", "", "0x2a"])
            #expect(noSize.envelope.exitCode == 64)
            #expect(
                noSize.envelope.error?.reason
                    == "the sampling design named seed 0x2a with no "
                    + "per-condition size — with no size the full corpus is "
                    + "coded, and the seed would be stamped on a design it "
                    + "did not shape")
            #expect(try ExperimentStore.load(name: "demo")
                .evaluationSampling == nil)
        }
    }

    @Test(arguments: ["0", "-3", "2400.0", "many"])
    func aMalformedSizeRefuses(raw: String) async throws {
        try await withTempRoot { _ in
            try draft()
            let outcome = await invoke(
                ["set-evaluation-sampling", "demo", raw, "0x2a"])
            #expect(outcome.envelope.exitCode == 64)
            #expect(
                outcome.envelope.error?.reason
                    == "the sampling design's samplePerCondition must be a "
                    + "whole number of records of at least 1, not '\(raw)' — "
                    + "a subsample of zero records is a design nobody can "
                    + "report")
            #expect(try ExperimentStore.load(name: "demo")
                .evaluationSampling == nil)
        }
    }

    @Test(arguments: ["zz", "-1", "0x10000000000000000"])
    func aMalformedSeedRefuses(raw: String) async throws {
        try await withTempRoot { _ in
            try draft()
            let outcome = await invoke(
                ["set-evaluation-sampling", "demo", "40", raw])
            #expect(outcome.envelope.exitCode == 64)
            #expect(
                outcome.envelope.error?.reason
                    == "the sampling design's sampleSeed '\(raw)' is not a "
                    + "64-bit unsigned number — a seed is a decimal integer, "
                    + "or hexadecimal with or without a '0x' prefix, of at "
                    + "most 16 hex digits (the leading 16 of a digest are a "
                    + "fine seed, written down as such)")
        }
    }

    /// The draw rule is the ENGINE's, so it can never be typed. Structural,
    /// because the guarantee is the absence of a flag — the same assertion
    /// `set-parser` carries about its registry hash.
    @Test func theRuleIsNeverAnArgument() throws {
        let spec = try #require(
            ExperimentCLIParser.specs.first {
                $0.namespace == "experiment"
                    && $0.verb == "set-evaluation-sampling"
            })
        #expect(spec.valueFlags.isEmpty)
        #expect(spec.booleanFlags.isEmpty)
        #expect(!spec.purpose.lowercased().contains("--rule"))
        #expect(!spec.positional.contains("rule"))
    }

    // MARK: - verify(): what a desk can check, and what it cannot

    /// The declare-time/run-time split, asserted as a split. Shape, a whole
    /// positive `n` and a parseable seed are checkable at the desk; the
    /// POPULATION is not, because at verify time the source run this design
    /// will be drawn from need not exist — and usually does not, since
    /// declaring before running is the point.
    @Test func verifyChecksTheDesignButNeverThePopulation() async throws {
        try await withTempRoot { _ in
            try draft()
            // An `n` far above anything any run could hold verifies CLEAN:
            // inventing an obligation a draft cannot meet would make the
            // declaration unusable in the order a study is authored.
            await invoke(
                ["set-evaluation-sampling", "demo", "1000000", "0x2a"])
            var manifest = try ExperimentStore.load(name: "demo")
            #expect(
                EvaluateSubsample.declarationViolations(
                    manifest.evaluationSampling
                ).isEmpty)

            // A rule from another version IS a desk finding: the version
            // marker exists so a moved rule is visible rather than silent.
            manifest.evaluationSampling = .init(
                rule: "stratifiedByPromptID/v0 — something else",
                samplePerCondition: 40, sampleSeed: "0x000000000000002a")
            let moved = EvaluateSubsample.declarationViolations(
                manifest.evaluationSampling)
            #expect(moved.count == 1)
            #expect(
                moved.first?.contains(
                    "evaluationSampling.rule is not the draw rule this build "
                        + "derives") == true)

            manifest.evaluationSampling = .init(
                rule: EvaluateSubsample.rule,
                samplePerCondition: 0, sampleSeed: "not-a-seed")
            let broken = EvaluateSubsample.declarationViolations(
                manifest.evaluationSampling)
            #expect(broken.count == 2)
            #expect(
                broken.contains {
                    $0.contains("samplePerCondition must be a whole number")
                })
            #expect(
                broken.contains {
                    $0.contains(
                        "sampleSeed 'not-a-seed' is not a 64-bit unsigned")
                })
        }
    }

    /// ABSENT = no declaration = no violations, so every manifest written
    /// before this existed verifies exactly as it did: the study's verify
    /// verdict gains nothing that names the declaration.
    @Test func anUndeclaredStudyGainsNothingAtVerify() async throws {
        try await withTempRoot { _ in
            let manifest = try draft()
            #expect(manifest.evaluationSampling == nil)
            #expect(EvaluateSubsample.declarationViolations(nil).isEmpty)
            let violations = try ExperimentStore.verify(manifest)
            #expect(
                !violations.contains {
                    $0.contains(EvaluateSubsample.declarationKey)
                },
                "an undeclared study gained a sampling finding: \(violations)")
        }
    }

    // MARK: - The flags become a cross-check

    @Test func aDeclarationAloneIsTheEffectiveDraw() throws {
        let declaration = try EvaluateSubsample.declaredRequest(
            .init(
                rule: EvaluateSubsample.rule, samplePerCondition: 7,
                sampleSeed: "0x000000000000002a"),
            experiment: "demo", program: "steerlab-cli")
        let effective = try #require(
            try EvaluateSubsample.reconcile(
                flags: nil, declaration: declaration,
                program: "steerlab-cli"))
        #expect(effective.samplePerCondition == 7)
        #expect(effective.seed == 0x2A)
        #expect(effective.declared)
        // …and it reaches the stamp as the additive `declared: true`.
        let stamp = EvaluateSubsample.stamp(effective, sampled: 14, source: 24)
        #expect(stamp.declared == true)
    }

    /// Equal flags are a CROSS-CHECK that passes: the coding is still the
    /// declared one, so the stamp still says `declared: true`.
    @Test func agreeingFlagsPassAndTheDrawStaysTheDeclaredOne() throws {
        let declaration = try EvaluateSubsample.declaredRequest(
            .init(
                rule: EvaluateSubsample.rule, samplePerCondition: 7,
                sampleSeed: "0x000000000000002a"),
            experiment: "demo", program: "steerlab-cli")
        let effective = try #require(
            try EvaluateSubsample.reconcile(
                flags: .init(samplePerCondition: 7, seed: 0x2A),
                declaration: declaration, program: "steerlab-cli"))
        #expect(effective.declared)
    }

    /// A disagreeing flag REFUSES, naming both values. Never an override: a
    /// flag that won would code one design while the run's snapshot recorded
    /// another.
    @Test func aDisagreeingFlagRefusesNamingBothValues() throws {
        let declaration = try EvaluateSubsample.declaredRequest(
            .init(
                rule: EvaluateSubsample.rule, samplePerCondition: 7,
                sampleSeed: "0x000000000000002a"),
            experiment: "demo", program: "steerlab-cli")
        func reason(_ flags: EvaluateSubsample.Request) -> (String, String) {
            do {
                _ = try EvaluateSubsample.reconcile(
                    flags: flags, declaration: declaration,
                    program: "steerlab-cli")
                return ("", "")
            } catch let error as ExperimentError {
                return (
                    error.reason,
                    error.malformedInvocation?.repairAction ?? "")
            } catch {
                return ("", "")
            }
        }
        let (sizeReason, sizeRepair) = reason(
            .init(samplePerCondition: 9, seed: 0x2A))
        #expect(
            sizeReason == "--sample-per-condition 9 contradicts this study's "
                + "declared sampling design, which preregistered 7 record(s) "
                + "per condition. On a study that declares its design the "
                + "flag is a CROSS-CHECK, never an override: the declaration "
                + "is what the run's experiment.json snapshot carries, so a "
                + "flag that won would code one design and record another")
        // The repair is never "--force": it is to drop the flag, or to
        // declare the design you actually want.
        #expect(sizeRepair.hasPrefix("drop --sample-per-condition"))
        #expect(sizeRepair.contains("set-evaluation-sampling <name> 9 <seed>"))
        #expect(!sizeRepair.contains("--force"))

        let (seedReason, seedRepair) = reason(
            .init(samplePerCondition: 7, seed: 99))
        #expect(
            seedReason == "--sample-seed 0x0000000000000063 contradicts this "
                + "study's declared sampling design, which preregistered seed "
                + "0x000000000000002a. On a study that declares its design "
                + "the flag is a CROSS-CHECK, never an override: the "
                + "declaration is what the run's experiment.json snapshot "
                + "carries, so a flag that won would draw one subsample and "
                + "record another")
        #expect(seedRepair.hasPrefix("drop --sample-seed"))
        #expect(!seedRepair.contains("--force"))
    }

    /// An UNDECLARED study keeps the flags-only path exactly as it was — the
    /// ad-hoc path stays legal and stays loud, it simply cannot claim the
    /// provenance a declared one has.
    @Test func anUndeclaredStudyKeepsTheFlagsOnlyPath() throws {
        let effective = try #require(
            try EvaluateSubsample.reconcile(
                flags: .init(samplePerCondition: 7, seed: 0x2A),
                declaration: nil, program: "steerlab-cli"))
        #expect(effective.samplePerCondition == 7)
        #expect(!effective.declared)
        #expect(
            EvaluateSubsample.stamp(effective, sampled: 14, source: 24)
                .declared == nil)
    }

    // MARK: - The proof: the declaration reaches the run's snapshot

    /// THE assertion the feature exists for. A declared study, evaluated with
    /// NO flags, draws the declared subsample — and the run directory's own
    /// `experiment.json` (the manifest snapshot every run stamps) carries the
    /// declaration. That is the evidence chain: a plan document is
    /// pre-registration, and this snapshot is what proves the plan is the
    /// thing that ran.
    ///
    /// The synthetic run IS `EvaluateSubsampleTests`'s cross-engine fixture —
    /// 2 conditions × 3 promptIDs × 4 sampleIndexes at `n = 7, seed = 0x2a` —
    /// so the 14 coded triples are the literals both engines already pin.
    @Test func theDeclarationLandsInTheRunsOwnSnapshot() async throws {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "declared-sample-\(UUID().uuidString)")
        ExperimentStore.rootOverride = temp
        ExperimentTasks.codingOverrideForTesting = { _, _ in
            "{\"codes\": {\"mentionsLegalRule\": true, "
                + "\"mentionsEquity\": false}, \"brief_reason\": \"r\"}"
        }
        defer {
            ExperimentTasks.codingOverrideForTesting = nil
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }

        let study = "declared-sampling"
        var manifest = try ExperimentStore.create(
            name: study, description: "d", modelID: "test/model")
        // A pinned concept, so `loadVerified` has a manifest it can verify —
        // the same shape `ResponseCodingTests`'s fixtures use.
        manifest.concepts.append(
            .init(
                name: "french",
                stimulusSetHash: try StimulusSet(
                    directory: VectorCatalog.conceptsDirectory
                        .appending(component: "french")
                ).hash,
                options: .init()))
        manifest.judges = [.init(name: "judge-1", kind: "local", model: nil)]
        try ExperimentStore.save(manifest)
        // The DECLARATION, through the verb — not by hand.
        try ExperimentStore.declareEvaluationSampling(
            samplePerCondition: "7", sampleSeed: "0x2a",
            experimentName: study)
        let declared = try ExperimentStore.load(name: study)

        let source = ExperimentStore.runsDirectory.appending(
            component: "20260829T000000000Z-exp-\(study)-run")
        try FileManager.default.createDirectory(
            at: source, withIntermediateDirectories: true)
        try ExperimentStore.manifestHash(declared).write(
            to: source.appending(component: "experiment-hash.txt"),
            atomically: true, encoding: .utf8)
        var rows: [String] = []
        for condition in ["baseline", "injected"] {
            for prompt in ["p01", "p02", "p03"] {
                for index in 0..<4 {
                    rows.append(
                        "{\"experiment\":\"\(study)\","
                            + "\"condition\":\"\(condition)\",\"seed\":\(index),"
                            + "\"sampleIndex\":\(index),"
                            + "\"promptID\":\"\(prompt)\",\"prompt\":\"q\","
                            + "\"output\":\"\(condition) \(prompt) \(index)\"}")
                }
            }
        }
        try (rows.joined(separator: "\n") + "\n").write(
            to: source.appending(component: "generations.jsonl"),
            atomically: true, encoding: .utf8)

        // NO sample flags anywhere on this call: the declaration is the ask.
        let out = try await ExperimentTasks.evaluatePairedJudge(
            experimentName: study,
            sourceRunDirectory: source,
            evaluation: ExperimentManifest.EvaluationSpec(
                kind: .pairedJudge, judgeModel: "",
                judgePrompt: ResponseCodingTests.codingRubric))

        // 1. The declared draw is the one that ran — the same 14 triples the
        //    flags-driven twin codes.
        let lines = try String(
            contentsOf: out.appending(component: "codings.jsonl"),
            encoding: .utf8
        ).split(separator: "\n")
        #expect(lines.count == 14)

        // 2. THE SNAPSHOT. The run's own experiment.json carries the design,
        //    rule included, so the evidence travels with the run rather than
        //    only with the command line that started it.
        let snapshot = try #require(
            try JSONSerialization.jsonObject(
                with: Data(
                    contentsOf: out.appending(component: "experiment.json")))
                as? [String: Any])
        let design = try #require(
            snapshot[EvaluateSubsample.declarationKey] as? [String: Any])
        #expect(design["samplePerCondition"] as? Int == 7)
        #expect(design["sampleSeed"] as? String == "0x000000000000002a")
        #expect(design["rule"] as? String == EvaluateSubsample.rule)

        // 3. …and the coding stamp additionally notes that the draw was
        //    DECLARED, which the ad-hoc path cannot claim.
        let report = try #require(
            try JSONSerialization.jsonObject(
                with: Data(
                    contentsOf: out.appending(
                        component: "coding-report.json")))
                as? [String: Any])
        let sampling = try #require(report["sampling"] as? [String: Any])
        #expect(sampling["declared"] as? Bool == true)
        #expect(sampling["samplePerCondition"] as? Int == 7)
        #expect(sampling["sampledRecords"] as? Int == 14)
        let config = try #require(
            try JSONSerialization.jsonObject(
                with: Data(
                    contentsOf: out.appending(component: "config.json")))
                as? [String: Any])
        let notes = try #require(config["notes"] as? [String: Any])
        let stamped = try #require(notes["sampling"] as? [String: Any])
        #expect(stamped["declared"] as? Bool == true)
    }

    /// The cross-check, end to end: a flag that disagrees with the
    /// declaration refuses at 64 and writes no run directory. The refusal
    /// happens in the TASK, not at a CLI edge, so every caller that holds a
    /// manifest gets it.
    @Test func aDisagreeingFlagRefusesAtEvaluateAndWritesNothing() async throws {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "declared-conflict-\(UUID().uuidString)")
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        let study = "declared-conflict"
        var manifest = try ExperimentStore.create(
            name: study, description: "d", modelID: "test/model")
        manifest.concepts.append(
            .init(
                name: "french",
                stimulusSetHash: try StimulusSet(
                    directory: VectorCatalog.conceptsDirectory
                        .appending(component: "french")
                ).hash,
                options: .init()))
        manifest.judges = [.init(name: "judge-1", kind: "local", model: nil)]
        try ExperimentStore.save(manifest)
        try ExperimentStore.declareEvaluationSampling(
            samplePerCondition: "7", sampleSeed: "0x2a",
            experimentName: study)
        let declared = try ExperimentStore.load(name: study)
        let source = ExperimentStore.runsDirectory.appending(
            component: "20260829T000000000Z-exp-\(study)-run")
        try FileManager.default.createDirectory(
            at: source, withIntermediateDirectories: true)
        try ExperimentStore.manifestHash(declared).write(
            to: source.appending(component: "experiment-hash.txt"),
            atomically: true, encoding: .utf8)
        try "{\"condition\":\"baseline\",\"promptID\":\"p\",\"output\":\"o\"}\n"
            .write(
                to: source.appending(component: "generations.jsonl"),
                atomically: true, encoding: .utf8)

        let before = (try? FileManager.default.contentsOfDirectory(
            atPath: ExperimentStore.runsDirectory.path))?.sorted() ?? []
        await #expect(throws: ExperimentError.self) {
            try await ExperimentTasks.evaluatePairedJudge(
                experimentName: study,
                sourceRunDirectory: source,
                evaluation: ExperimentManifest.EvaluationSpec(
                    kind: .pairedJudge, judgeModel: "",
                    judgePrompt: ResponseCodingTests.codingRubric),
                subsample: .init(samplePerCondition: 9, seed: 0x2A))
        }
        let after = (try? FileManager.default.contentsOfDirectory(
            atPath: ExperimentStore.runsDirectory.path))?.sorted() ?? []
        #expect(after == before)
    }
}
