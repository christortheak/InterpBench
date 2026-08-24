import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// The second closed refusal vocabulary and its exhaustiveness
/// (WP0-AGENT-SURFACE-AUDIT §2.4, §7 step 7; gate-5 dry run #1's punch list).
///
/// Dry run #1 passed the lifecycle and named one systemic defect: `freeze` was
/// the ONLY verb whose refusal a machine could act on. Everything else — a
/// frozen manifest refusing an edit, a validate blocked by a pin that appeared
/// after attach, a promote with no sweep, a frozen run with no pinned prompts —
/// came back as `failed` / exit 70 / `code: "verbFailed"` with the repair "read
/// the reason and repair the named input". Four different gates, one
/// indistinguishable answer, and a repair naming no command.
///
/// Two things are asserted here, and they are different in kind:
///
/// 1. **Exhaustiveness over the registry** — every gate in the vocabulary is
///    claimed by a declared site, every site's repair is a runnable command,
///    and every verb a site names is a verb the CLI actually declares. This is
///    the guard against the vocabulary rotting into a list of hopes.
/// 2. **Reachability** — the four refusals dry run #1 hit are driven end to end
///    through `ExperimentCLIRunner` and asserted to come back `refused` / 65
///    with their id and a repair that starts with a command. A registry that
///    described refusals nothing produces would be worse than none.
///
/// Serialized, and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct RefusalSiteRegistryTests {

    // MARK: Harness

    func withTempRoot<T>(_ body: (URL) async throws -> T) async throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "refusal-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        WorkspaceRoot.programmaticOverride = temp
        ExperimentStore.rootOverride = temp
        // A REAL workspace, not just an experiments root: these tests attach a
        // concept and (for the appeared-after-attach loop) author its
        // validation.jsonl, and writing that under a checkout-resolved
        // concepts directory would edit the repository's own seed data.
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
        await ExperimentCLIRunner(sink: .discarding).run(
            namespace: namespace, args)
    }

    func write(_ text: String, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: url, atomically: true, encoding: .utf8)
    }

    /// A refusal, asserted the way an AGENT reads one: state, exit code, the
    /// id in both `code` and `gate`, and a repair whose first token is a
    /// command it can run.
    func expectRefusal(
        _ outcome: ExperimentCLIOutcome, _ gate: LifecycleGate,
        sourceLocation: SourceLocation = #_sourceLocation
    ) {
        #expect(
            outcome.envelope.state == .refused, "\(gate): state",
            sourceLocation: sourceLocation)
        #expect(
            outcome.envelope.exitCode == 65, "\(gate): exit code",
            sourceLocation: sourceLocation)
        #expect(
            outcome.envelope.error?.code == gate.rawValue, "\(gate): code",
            sourceLocation: sourceLocation)
        #expect(
            outcome.envelope.error?.gate == gate.rawValue, "\(gate): gate",
            sourceLocation: sourceLocation)
        let repair = outcome.envelope.error?.repairAction ?? ""
        #expect(
            repair.contains("steerlab-cli "),
            "\(gate): repair is not a command — '\(repair)'",
            sourceLocation: sourceLocation)
    }

    // MARK: - 1. Exhaustiveness

    @Test func everyAgentPathRefusalCarriesACode() {
        // Every gate is claimed. A gate with no site means a refusal class
        // was named and never wired — the failure mode this registry exists
        // to make impossible.
        for gate in LifecycleGate.allCases {
            #expect(
                RefusalSiteRegistry.site(for: gate) != nil,
                "no registry site claims '\(gate.rawValue)'")
        }
        // …and nothing claims a gate twice: two sites for one id would make
        // "which repair applies" a coin flip.
        let claimed = RefusalSiteRegistry.sites.map(\.gate.rawValue)
        #expect(Set(claimed).count == claimed.count, "duplicate gate claim")
        #expect(claimed.count == LifecycleGate.allCases.count)

        let declaredVerbs = Set(
            ExperimentCLIParser.specs.map { "\($0.namespace) \($0.verb)" })
        for site in RefusalSiteRegistry.sites {
            // A runnable repair. The dry run proved agents follow these
            // verbatim, so "duplicate to iterate" — true, and not a command —
            // is exactly the failure being fixed.
            #expect(
                site.repairAction.contains("steerlab-cli "),
                "\(site.gate.rawValue): repair names no command")
            #expect(!site.origin.isEmpty, "\(site.gate.rawValue): no origin")
            #expect(!site.verbs.isEmpty, "\(site.gate.rawValue): no verbs")
            for verb in site.verbs {
                #expect(
                    declaredVerbs.contains(verb),
                    "\(site.gate.rawValue) names undeclared verb '\(verb)'")
            }
        }
    }

    @Test func theTwoVocabulariesStayDisjoint() {
        // An id in both would make `error.gate` ambiguous about which switch
        // an agent should use — and the two have different skippability
        // classes: `--force` skips freeze gates, nothing skips these.
        #expect(!LifecycleGate.collidesWithFreezeVocabulary)
        #expect(LifecycleGate.vocabulary.count == LifecycleGate.allCases.count)
        #expect(Set(LifecycleGate.vocabulary).count == LifecycleGate.vocabulary.count)
        // The freeze seven are untouched by this step.
        #expect(
            FreezeGate.vocabulary == [
                "revision", "validateEvidence", "batteryEvidence",
                "judgeValidity", "variantValidity", "gitClean", "measurementPins",
            ])
    }

    @Test func onlyDataCheckMigratesItsHumanExitCode() {
        // Audit §7 row 7 names exactly ONE human-mode migration (`data check`
        // blockers, 2 → 65). Widening it would break `set -e` wrappers on a
        // change the row does not discuss, so the rest stay at 1 while the
        // ENVELOPE speaks 65 for all of them.
        for gate in LifecycleGate.allCases {
            let expected: Int32 = gate == .dataReadiness ? 65 : 1
            #expect(
                gate.humanExitCode == expected,
                "\(gate.rawValue) human exit code")
        }
        #expect(SteerLabCLIState.refused.exitCode == 65)
    }

    @Test func theAdvisoryVocabularyIsClosedAndDistinct() {
        #expect(Set(CLIAdvisory.vocabulary).count == CLIAdvisory.allCases.count)
        // The step-7 additions the punch list asked for, by name — a rename
        // here is a wire-contract change and must move the server twin too.
        for code in [
            "sweepRecommendationsOnly", "sweepSelectionDefaulted",
            "probeAtChanceFloor", "allEffectSizesZero",
            "choiceItemsWithoutInstrument",
        ] {
            #expect(CLIAdvisory.vocabulary.contains(code), "\(code)")
        }
        // Advisories never carry an exit code of their own.
        #expect(SteerLabCLIState.okWithAdvisories.exitCode == 0)
    }

    // MARK: - 2. Reachability: dry run #1's four repairable refusals

    /// P1a. `attach` on a frozen manifest.
    @Test func attachOnAFrozenManifestRefusesWithAnExecutableRepair() async throws {
        try await withTempRoot { _ in
            await invoke(
                "experiment",
                ["create", "frozen-demo", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            await invoke("experiment", ["attach", "frozen-demo", "french"])
            // Freeze the manifest directly: the gate evidence a real freeze
            // needs is a model run away, and this test is about what happens
            // AFTER a study is frozen.
            var manifest = try ExperimentStore.load(name: "frozen-demo")
            manifest.status = .frozen
            manifest.freezeHash = String(repeating: "a", count: 64)
            try ExperimentStore.save(manifest)  // draft → frozen is legal

            let outcome = await invoke("experiment", ["attach", "frozen-demo", "french"])
            expectRefusal(outcome, .statusImmutable)
            // The repair is the ONLY legal move, spelled as commands.
            let repair = try #require(outcome.envelope.error?.repairAction)
            #expect(repair.contains("experiment duplicate frozen-demo"))
            // Human mode is unchanged — exit 1, same prose.
            #expect(outcome.exitCode == 1)
            #expect(outcome.failure?.reason.contains("duplicate it to iterate") == true)
        }
    }

    /// P1b. A `validation.jsonl` authored AFTER attach pinned it as absent —
    /// the loop dry run #1 could not escape, because the freeze gate's own
    /// repair told it to author the file and re-validate, and re-validating
    /// then refused (§9, P5).
    @Test func validationThatAppearedAfterAttachNamesTheReAttach() async throws {
        try await withTempRoot { root in
            await invoke(
                "experiment",
                ["create", "appeared", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            await invoke("experiment", ["attach", "appeared", "french"])
            // Pinned as absent at attach…
            let manifest = try ExperimentStore.load(name: "appeared")
            let ref = try #require(manifest.concepts.first)
            #expect(ref.validationHash == nil)
            #expect(ref.validationHashPinnedAbsent)

            // …then the researcher does exactly what the vacuous-validation
            // repair asks for.
            try write(
                #"{"text": "Il fait beau aujourd'hui.", "expresses": true}"# + "\n"
                    + #"{"text": "The weather is fine today.", "expresses": false}"# + "\n",
                to: VectorCatalog.conceptsDirectory
                    .appending(components: "french", "validation.jsonl"))
            _ = root

            let outcome = await invoke("experiment", ["verify", "appeared"])
            expectRefusal(outcome, .pinDrift)
            let repair = try #require(outcome.envelope.error?.repairAction)
            // THE missing step, named: re-attach re-pins the hash.
            #expect(repair.contains("steerlab-cli experiment attach appeared french"))
            #expect(repair.contains("experiment validate appeared"))

            // And the repair, run, actually repairs it — the property the
            // shipped one did not have.
            await invoke("experiment", ["attach", "appeared", "french"])
            let after = await invoke("experiment", ["verify", "appeared"])
            #expect(after.envelope.state == .ready, "\(after.envelope.message)")
        }
    }

    /// P1c. `promote --cell` with no sweep at all — "hand-creation wearing a
    /// promotion badge" (audit §2.4's named gate-5 candidate).
    @Test func promoteWithNoSweepRefusesWithTheSweepCommand() async throws {
        try await withTempRoot { _ in
            await invoke(
                "experiment",
                ["create", "nosweep", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            await invoke("experiment", ["attach", "nosweep", "french"])
            let outcome = await invoke(
                "experiment",
                [
                    "promote", "nosweep", "french", "--cell", "17:0.4",
                    "--reason", "trying it",
                ])
            expectRefusal(outcome, .promotionEvidence)
            let repair = try #require(outcome.envelope.error?.repairAction)
            #expect(repair.contains("steerlab-cli experiment sweep nosweep"))
            #expect(repair.contains("steerlab-cli experiment promote nosweep french"))
        }
    }

    /// P1d. A frozen study with no pinned task prompts. The prose already
    /// named the three steps; nothing gave them to a machine as one runnable
    /// line, and the refusal was `failed`/70.
    @Test func frozenRunWithoutPinnedPromptsNamesTheDuplicateChain() throws {
        // Driven at the task layer: the run verb loads a model, and this
        // refusal fires strictly before that — which is the point of it.
        var manifest = ExperimentManifest(
            name: "frozen-run", description: "", modelID: "m")
        manifest.status = .frozen
        manifest.taskPromptsFile = nil
        manifest.taskPromptsHash = nil
        do {
            _ = try ExperimentTasks.loadTaskPrompts(for: manifest)
            Issue.record("expected a refusal")
        } catch let error as ExperimentError {
            let refusal = try #require(error.lifecycleRefusal)
            // A missing PIN, not drift: telling an agent to repair a hash
            // that was never written sends it in a circle.
            #expect(refusal.gate == .missingPrerequisite)
            #expect(
                refusal.repairAction.contains(
                    "steerlab-cli experiment duplicate frozen-run frozen-run-v2"))
            #expect(refusal.repairAction.contains("experiment pin-prompts frozen-run-v2"))
            #expect(refusal.repairAction.contains("experiment freeze frozen-run-v2"))
            // Prose unchanged: this is the string the verb has always thrown.
            #expect(error.reason.hasPrefix("frozen study has no pinned task prompts"))
        }
    }

    // MARK: - 3. The siblings on the agent path

    @Test func theEpochGuardCarriesItsGate() throws {
        var manifest = ExperimentManifest(name: "epoch", description: "", modelID: "m")
        manifest.status = .frozen
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "epoch-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temp) }
        try String(repeating: "b", count: 64).write(
            to: temp.appending(component: "experiment-hash.txt"),
            atomically: true, encoding: .utf8)
        do {
            _ = try ExperimentTasks.verifyRunEpoch(
                verb: "analyze", runDirectory: temp, manifest: manifest)
            Issue.record("expected an epoch refusal")
        } catch let error as ExperimentError {
            let refusal = try #require(error.lifecycleRefusal)
            #expect(refusal.gate == .manifestEpoch)
            #expect(refusal.repairAction.contains("steerlab-cli experiment run epoch"))
            #expect(refusal.repairAction.contains("--allow-unverified-epoch"))
        }
    }

    @Test func theGreedyOnlyPolicyCarriesItsGate() throws {
        var manifest = ExperimentManifest(name: "hot", description: "", modelID: "m")
        manifest.temperature = 0.7
        do {
            try ExperimentTasks.requireGreedyLocalDesign(manifest)
            Issue.record("expected a sampling-policy refusal")
        } catch let error as ExperimentError {
            let refusal = try #require(error.lifecycleRefusal)
            #expect(refusal.gate == .samplingPolicy)
            // The repair names the SUBSTRATE that can honour the declaration,
            // which is the fact an agent has no way to know: the local MLX
            // generator pins no per-run seed.
            #expect(refusal.repairAction.contains("steerlab-cli remote submit-bundle"))
        }
    }

    /// 2026-08-18, the ONBOARDING verification pass: `pin-rubric` against a
    /// path that is not on disk (a typo, or a workspace with no
    /// `prompts/rubrics/` yet) let Foundation's own error out unhandled. The
    /// human line was an `NSCocoaErrorDomain` dump — which embeds an
    /// `NSUnderlyingError` POINTER, so it is not even stable across
    /// invocations — and the envelope fell into the generic missing-file
    /// catch, whose repair reads "no file at /abs/path": no command, no
    /// convention, nothing an agent can act on.
    ///
    /// Its three siblings — every other verb whose argument is a workspace
    /// FILE — were in the same hole from the other side: a good sentence and
    /// an untyped `failed`/70/`verbFailed` envelope, the answer a crash
    /// gives. A path typo is the commonest authoring mistake there is, so all
    /// four now answer with the gate the RUN loop has always used for a named
    /// input that is not there, and a repair that names the convention
    /// directory and the verb to retry.
    @Test func aNamedInputFileThatIsNotThereRefusesWithItsConvention() async throws {
        try await withTempRoot { _ in
            await invoke(
                "experiment",
                ["create", "missing", "--model", "mlx-community/gemma-3-4b-it-4bit"])

            // THE bug.
            let rubric = await invoke(
                "experiment", ["pin-rubric", "missing", "prompts/rubrics/nope.md"])
            expectRefusal(rubric, .missingPrerequisite)
            #expect(rubric.envelope.message.hasPrefix("judge rubric file not found:"))
            let rubricRepair = try #require(rubric.envelope.error?.repairAction)
            #expect(rubricRepair.contains("under prompts/rubrics/"))
            #expect(rubricRepair.contains(JudgeRubricStore.defaultRubricFile))
            #expect(
                rubricRepair.contains(
                    "steerlab-cli experiment pin-rubric missing prompts/rubrics/nope.md"))
            // Neither surface dumps Foundation's error any more — and human
            // mode keeps exit 1, as every lifecycle gate but `data check` does.
            #expect(rubric.failure?.reason.contains("NSCocoaErrorDomain") != true)
            #expect(rubric.exitCode == 1)

            // The sibling whose sentence the fix mirrors: byte-identical to
            // what `ExperimentTasks.loadTaskPrompts` throws at RUN time, so
            // the same mistake reads the same at pin time and at run time.
            let prompts = await invoke(
                "experiment", ["pin-prompts", "missing", "prompts/tasks/nope.jsonl"])
            expectRefusal(prompts, .missingPrerequisite)
            #expect(prompts.envelope.message.hasPrefix("task prompt file not found:"))
            #expect(
                prompts.envelope.error?.repairAction.contains(
                    "steerlab-cli experiment pin-prompts missing prompts/tasks/nope.jsonl")
                    == true)

            let taxonomy = await invoke(
                "experiment",
                ["set-style-taxonomy", "missing", "prompts/taxonomies/nope.json"])
            expectRefusal(taxonomy, .missingPrerequisite)
            // Prose byte-preserved: it already named the convention directory.
            #expect(taxonomy.envelope.message.hasPrefix("no taxonomy file at "))
            #expect(
                taxonomy.envelope.error?.repairAction.contains(
                    "steerlab-cli experiment set-style-taxonomy missing") == true)

            // …and the same class one module down: a concept whose stimulus
            // files are not on disk. The prose is SteeringKit's — it names
            // every absent file and the row shape they need — and is
            // untouched; only the classification is new, and it is attached at
            // the ExperimentKit boundary because SteeringKit is
            // concept-agnostic by hard requirement.
            let attach = await invoke(
                "experiment", ["attach", "missing", "absent-concept"])
            expectRefusal(attach, .missingPrerequisite)
            #expect(attach.envelope.message.contains("stimulus files not found"))
            let attachRepair = try #require(attach.envelope.error?.repairAction)
            #expect(attachRepair.contains("prompts/concepts/<concept>/"))
            #expect(
                attachRepair.contains(
                    "steerlab-cli experiment attach missing absent-concept"))
        }
    }

    /// A MALFORMED stimulus row is not the same fact and must not borrow the
    /// same repair: telling an agent to author a file that already exists
    /// sends it in a circle. Only the absent-file cases are classified.
    @Test func aMalformedStimulusRowIsNotAMissingFile() {
        #expect(
            ExperimentCLIRunner.namesMissingFiles(
                .missingFiles([URL(filePath: "/tmp/positive.jsonl")])))
        #expect(ExperimentCLIRunner.namesMissingFiles(.missingFile(URL(filePath: "/tmp/x"))))
        #expect(
            !ExperimentCLIRunner.namesMissingFiles(
                .malformedTextRow(file: "positive.jsonl", line: 3)))
        #expect(
            !ExperimentCLIRunner.namesMissingFiles(
                .malformedLine(file: "positive.jsonl", line: 3)))
        #expect(!ExperimentCLIRunner.namesMissingFiles(.empty("french")))
    }

    @Test func aDraftEditOnACompleteStudyCarriesItsGate() async throws {
        try await withTempRoot { _ in
            await invoke(
                "experiment",
                ["create", "done", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            var manifest = try ExperimentStore.load(name: "done")
            manifest.status = .frozen
            try ExperimentStore.save(manifest)
            manifest.status = .complete
            try ExperimentStore.save(manifest)  // frozen → complete is legal
            let outcome = await invoke(
                "experiment",
                [
                    "declare-condition", "done", "arm", "--baseline",
                    "--alpha-units", "norm",
                ])
            expectRefusal(outcome, .statusImmutable)
        }
    }

    // MARK: - 4. The punch list's VALIDITY findings (P2, P3, P4, P5, P13, P14)

    /// P4. A chance-level probe froze and ran with no machine signal: the
    /// envelope reported only that validate had happened.
    @Test func validateCarriesItsProbeScoresAndFlagsTheChanceFloor() throws {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "vscore-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temp) }
        let report: [String: Any] = [
            "experiment": "demo",
            "validation": [
                "french": [
                    "scenarios": 12,
                    "depths": [[
                        "layer": 17,
                        "accuracy": 0.5,
                        "diagnostics": [
                            "auc": 0.52, "balancedAccuracy": 0.5,
                            "oneSidedPredictions": false,
                        ],
                    ]],
                ],
                "sympathy": [
                    "scenarios": 12,
                    "depths": [[
                        "layer": 17,
                        "accuracy": 0.92,
                        "diagnostics": [
                            "auc": 0.97, "balancedAccuracy": 0.91,
                            "oneSidedPredictions": false,
                        ],
                    ]],
                ],
                // A string entry is not a score — the vacuity ledger already
                // names it, and treating it as 0.0 would invent a measurement.
                "stoicism": "no validation.jsonl — convergent gate NOT run",
            ],
        ]
        try JSONSerialization.data(withJSONObject: report)
            .write(to: temp.appending(component: "report.json"))

        let scores = ExperimentCLIRunner.validationScores(inRunAt: temp)
        #expect(scores.map(\.concept) == ["french", "sympathy"])
        let french = try #require(scores.first)
        #expect(french.accuracy == 0.5)
        #expect(french.auc == 0.52)
        #expect(french.layer == 17)
        #expect(french.scenarios == 12)
        #expect(french.isAtOrBelowChance)
        // The advisory says the thing an agent cannot infer: this evidence
        // still SATISFIES the freeze gate.
        #expect(french.advisoryDetail.contains("SATISFIES"))
        #expect(!(scores.last?.isAtOrBelowChance ?? true))
    }

    @Test func aOneSidedThresholdIsAtTheFloorWhateverTheAccuracySays() {
        // The 2026-08-01 lesson, carried into the advisory: a transfer
        // threshold that puts every item on one side is measuring the
        // threshold, not the vector — accuracy 0.5 with AUC 0.855 was the
        // real case.
        let score = ExperimentCLIRunner.ValidationScore(
            concept: "c", layer: 17, accuracy: 0.86, balancedAccuracy: 0.86,
            auc: 0.9, scenarios: 20, oneSided: true)
        #expect(score.isAtOrBelowChance)
        #expect(score.advisoryDetail.contains("EVERY item on one side"))
    }

    /// P14. Zero ENTRIES had an advisory; entries that are all exactly zero
    /// did not — and they are a different fact.
    @Test func analyzeFlagsAnAllZeroEffectTable() throws {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "azero-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temp) }

        func writeAnalysis(_ diffs: [Double]) throws {
            let object: [String: Any] = [
                "effectSizes": diffs.map { value in
                    ["condition": "arm", "metric": "m", "n": 4, "meanDiff": value]
                }
            ]
            try JSONSerialization.data(withJSONObject: object)
                .write(to: temp.appending(component: "analysis.json"))
        }
        try writeAnalysis([0, 0, 0])
        #expect(ExperimentCLIRunner.allEffectSizesAreZero(inRunAt: temp))
        // One non-zero row is a real (if small) measurement, not the defect.
        try writeAnalysis([0, 0, 0.0001])
        #expect(!ExperimentCLIRunner.allEffectSizesAreZero(inRunAt: temp))
        // No entries at all is the OTHER advisory (emptyAnalysis).
        try writeAnalysis([])
        #expect(!ExperimentCLIRunner.allEffectSizesAreZero(inRunAt: temp))
    }

    /// P3. The rule the methods note is most emphatic about, made followable:
    /// the default fires silently on a choice-shaped task set.
    @Test func anUndeclaredCriterionOnAChoiceTaskIsAdvised() {
        let advisory = SweepSelectionRule.defaultedSelectionAdvisory(
            spec: nil, choiceItemCount: 8, totalItemCount: 8)
        let text = try! #require(advisory)
        #expect(text.contains("markerDensity"))
        #expect(text.contains("steerlab-cli experiment set-sweep-selection"))
        // Declared: nothing to say, whatever the task shape.
        #expect(
            SweepSelectionRule.defaultedSelectionAdvisory(
                spec: .init(objective: .init(metric: "judgeScore")),
                choiceItemCount: 8, totalItemCount: 8) == nil)
        // Not choice-shaped: marker density is a legitimate screen objective.
        #expect(
            SweepSelectionRule.defaultedSelectionAdvisory(
                spec: nil, choiceItemCount: 0, totalItemCount: 8) == nil)
    }

    /// P3, the authoring half: the criterion is now writable headlessly, and
    /// it is validated at DECLARATION rather than at sweep start.
    @Test func setSweepSelectionWritesAndValidatesTheCriterion() async throws {
        try await withTempRoot { _ in
            await invoke(
                "experiment",
                ["create", "crit", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            await invoke("experiment", ["attach", "crit", "french"])
            let outcome = await invoke(
                "experiment",
                [
                    "set-sweep-selection", "crit", "--objective", "markerDensity",
                    "--capability-tolerance", "0.1", "--coherence-floor", "0.5",
                    "--control-margin", "0.05",
                ])
            #expect(outcome.exitCode == 0)
            let manifest = try ExperimentStore.load(name: "crit")
            let selection = try #require(manifest.sweep?.selection)
            #expect(selection.objective?.metric == "markerDensity")
            #expect(selection.constraints?.capabilityTolerance == 0.1)
            #expect(selection.controls?.matchedNormRandomMargin == 0.05)
            // The resolved criterion is what the sweep will apply — reported,
            // not left for the caller to re-derive.
            guard case .string(let metric)? = outcome.envelope.result?["objective"]
            else {
                Issue.record("no objective in result")
                return
            }
            #expect(metric == "markerDensity")

            // An unknown metric is refused at DECLARATION.
            let bad = await invoke(
                "experiment",
                ["set-sweep-selection", "crit", "--objective", "vibes"])
            #expect(bad.exitCode != 0)
            #expect(bad.envelope.message.contains("unknown --objective"))
            // …and topK without a width is refused rather than defaulted: how
            // many cells a control covers is a preregistration decision.
            let topK = await invoke(
                "experiment",
                [
                    "set-sweep-selection", "crit", "--objective", "markerDensity",
                    "--control-margin", "0.05", "--control-apply-to", "topK",
                ])
            #expect(topK.exitCode != 0)
            #expect(topK.envelope.message.contains("--control-top-k"))
        }
    }

    /// P13. Pinned `options` + `target` do NOT engage the logprob instrument:
    /// the declaration is provenance and is never inferred from the data. The
    /// root cause is an authoring gap, and this is the verb that closes it.
    @Test func choiceItemsAdviseAndSetInstrumentsDeclares() async throws {
        try await withTempRoot { root in
            await invoke(
                "experiment",
                ["create", "choice", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            try write(
                #"{"id": "c1", "prompt": "Affirm or reverse?", "options": ["A", "B"], "target": "A"}"#
                    + "\n",
                to: root.appending(components: "prompts", "tasks", "items.jsonl"))
            let pinned = await invoke(
                "experiment",
                ["pin-prompts", "choice", "prompts/tasks/items.jsonl"])
            #expect(pinned.exitCode == 0)
            #expect(pinned.envelope.state == .okWithAdvisories)
            #expect(
                pinned.envelope.advisories?.contains {
                    $0.code == CLIAdvisory.choiceItemsWithoutInstrument.rawValue
                } == true)

            let declared = await invoke(
                "experiment", ["set-instruments", "choice", "answerTokenLogprob"])
            #expect(declared.exitCode == 0)
            let manifest = try ExperimentStore.load(name: "choice")
            #expect(manifest.outcomeInstruments == ["answerTokenLogprob"])
            #expect(
                !ExecutionPlan.resolve(instruments: manifest.outcomeInstruments)
                    .generatesSampledText)

            // Re-pinning now says nothing: the instrument is declared, so the
            // advisory would be noise.
            let again = await invoke(
                "experiment",
                ["pin-prompts", "choice", "prompts/tasks/items.jsonl"])
            #expect(again.envelope.state == .ready)

            // A typo cannot mint an instrument no engine implements.
            let bad = await invoke(
                "experiment", ["set-instruments", "choice", "answerLogprob"])
            #expect(bad.exitCode != 0)
            #expect(bad.envelope.message.contains("unknown outcome instrument"))
        }
    }

    /// P5. The `validateEvidence` gate's repair now names the re-attach its
    /// own steps required, while the cross-engine PROSE is untouched.
    @Test func theVacuousEvidenceRepairNamesTheReAttach() {
        var manifest = ExperimentManifest(
            name: "vac", description: "", modelID: "m")
        manifest.concepts = [
            ExperimentStore.makeConceptRef(
                name: "french", stimulusSetHash: String(repeating: "0", count: 64),
                options: ExtractionOptions())
        ]
        let prose = ExperimentStore.vacuousValidationRepairAction(
            for: manifest, vacuousConcepts: ["french"])
        let machine = ExperimentStore.vacuousValidationMachineRepair(
            for: manifest, vacuousConcepts: ["french"])
        let proseText = try! #require(prose)
        let machineText = try! #require(machine)
        // Byte-preserving: the machine form EXTENDS the prose, it does not
        // rewrite it (the prose is the cross-engine refusal string).
        #expect(machineText.hasPrefix(proseText))
        #expect(machineText.contains("steerlab-cli experiment attach vac french"))
        #expect(!proseText.contains("experiment attach"))
    }

    @Test func dataCheckBlockersAreTheOneMigratedHumanCode() async throws {
        try await withTempRoot { _ in
            await invoke(
                "experiment",
                ["create", "needs", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            await invoke("experiment", ["attach", "needs", "french"])
            let outcome = await invoke("data", ["check", "needs"])
            // BOTH modes now, which is the whole content of the migration.
            #expect(outcome.exitCode == 65)
            #expect(outcome.envelope.exitCode == 65)
            #expect(
                outcome.envelope.error?.code == LifecycleGate.dataReadiness.rawValue)
            #expect(
                outcome.envelope.error?.repairAction.contains(
                    "steerlab-cli data check needs") == true)
        }
    }
}
