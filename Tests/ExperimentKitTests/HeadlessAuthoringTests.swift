import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// The headless authoring verbs (WP0-AGENT-SURFACE-AUDIT §8, punch-list
/// P0-3; ladder amendment (b), the step inserted between 5 and 6).
///
/// The gate-5 dry run found that the Swift CLI could pin CONCEPTS and
/// nothing else: no task prompts, no rubric, no arms. An agent could
/// therefore walk create → attach → extract → validate → freeze → run →
/// analyze, exit 0 at every step, and have measured nothing — and every
/// refusal that named the missing piece named a remedy that existed only in
/// the Studies panel ("enter draft rubric text").
///
/// Three verbs close it, and this suite is the proof that they do:
///
///   * `experiment pin-prompts <name> <file>` — the measured task,
///   * `experiment pin-rubric <name> <file> [--judges …]` — the judging
///     instrument plus the panel and the `evaluation` declaration,
///   * `experiment declare-condition <name> <condition> --slots …` — the
///     arm, without which a concept study runs the implicit baseline alone.
///
/// The end-to-end test authors a complete minimal concept study through
/// nothing but the CLI and then asserts the run loop's no-conditions
/// refusal is GONE — by calling the manifest problem functions directly, the
/// way the firewall suite does, because asserting it by RUNNING would mean
/// loading a model.
///
/// Serialized, and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct HeadlessAuthoringTests {

    // MARK: Harness

    func withTempRoot<T>(_ body: (URL) async throws -> T) async throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "authoring-\(UUID().uuidString)")
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
    func invoke(_ args: [String]) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(sink: .discarding)
            .run(namespace: "experiment", args)
    }

    func write(_ text: String, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: url, atomically: true, encoding: .utf8)
    }

    static let model = "mlx-community/gemma-3-4b-it-4bit"

    static let taskPrompts =
        #"{"id": "c1", "prompt": "Decide the case and state the holding."}"# + "\n"
        + #"{"id": "c2", "prompt": "Decide the second case and state the holding."}"#
        + "\n"

    // MARK: - The end-to-end proof

    /// create → attach → pin-prompts → declare-condition → freeze --force,
    /// through the CLI only, and the study that comes out is one the run
    /// loop will MEASURE.
    ///
    /// `validate` is deliberately not run: the `french` fixture has no
    /// held-out probe, so its evidence is vacuous and freeze refuses it — the
    /// 2026-08-17 firewall repair working exactly as intended. `--force` is
    /// the sanctioned, stamped escape, and the point of this test is the
    /// AUTHORING surface, not the evidence tier. The freeze is asserted to be
    /// stamped non-citable so the test can never be read as blessing one.
    @Test func aMinimalMeasurableStudyIsAuthorableWithNothingButTheCLI() async throws {
        try await withTempRoot { root in
            #expect(await invoke(["create", "study", "--model", Self.model]).exitCode == 0)
            #expect(await invoke(["attach", "study", "french"]).exitCode == 0)

            // Before the arm and the prompts, the manifest measures nothing —
            // and says so.
            let bare = try ExperimentStore.load(name: "study")
            #expect(ExperimentStore.noMeasuredConditionsProblem(bare) != nil)
            #expect(bare.taskPromptsFile == nil)

            try write(
                Self.taskPrompts,
                to: root.appending(components: "prompts", "cases", "items.jsonl"))
            let pinned = await invoke(
                ["pin-prompts", "study", "prompts/cases/items.jsonl"])
            #expect(pinned.exitCode == 0)
            #expect(pinned.envelope.changed)

            let declared = await invoke(
                [
                    "declare-condition", "study", "french-hi",
                    "--slots", "french:17:0.4", "--alpha-units", "norm",
                ])
            #expect(declared.exitCode == 0)

            // An explicit baseline, so the arm has something to pair against.
            #expect(
                await invoke(
                    [
                        "declare-condition", "study", "baseline",
                        "--baseline", "--alpha-units", "norm",
                    ]
                ).exitCode == 0)

            // THE ASSERTION. The run loop's two silent-baseline refusals are
            // both silent now, and the conditions the run would execute are
            // the ones that were declared.
            let authored = try ExperimentStore.load(name: "study")
            #expect(ExperimentStore.noMeasuredConditionsProblem(authored) == nil)
            #expect(ExperimentStore.inertConditionsProblem(authored) == nil)
            #expect(
                ExperimentTasks.ordinaryRunConditions(for: authored).map(\.name).sorted()
                    == ["baseline", "french-hi"])
            // Nothing was pinned that verify() would refuse the moment it
            // was written.
            #expect(ExperimentStore.verify(authored).isEmpty)

            // And it freezes — loudly, and stamped non-citable.
            let frozen = await invoke(["freeze", "study", "--force"])
            #expect(frozen.exitCode == 0)
            let manifest = try ExperimentStore.load(name: "study")
            #expect(manifest.status == .frozen)
            #expect(manifest.freezeHash?.count == 64)
            #expect(manifest.freezeForced == true)
            #expect(manifest.forcedGatesSkipped?.isEmpty == false)
            // A frozen study still measures something: the pins survived.
            #expect(manifest.taskPromptsHash?.count == 64)
            #expect(ExperimentStore.noMeasuredConditionsProblem(manifest) == nil)
        }
    }

    /// The pins a CLI-authored study carries are the pins the panel writes —
    /// same manifest keys, same hash function, same bytes. If these ever
    /// diverge, a study authored by an agent and one authored by a
    /// researcher stop being the same object.
    @Test func theCLIsPinsAreByteIdenticalToTheStoreSetterTheGUIUses() async throws {
        try await withTempRoot { root in
            await invoke(["create", "cli", "--model", Self.model])
            await invoke(["create", "gui", "--model", Self.model])
            try write(
                Self.taskPrompts,
                to: root.appending(components: "prompts", "cases", "items.jsonl"))

            await invoke(["pin-prompts", "cli", "prompts/cases/items.jsonl"])
            // The path the Studies panel's setup save takes.
            var gui = try ExperimentStore.load(name: "gui")
            _ = try ExperimentStore.pinTaskPrompts(
                "prompts/cases/items.jsonl", into: &gui)
            try ExperimentStore.save(gui)

            let fromCLI = try ExperimentStore.load(name: "cli")
            let fromGUI = try ExperimentStore.load(name: "gui")
            #expect(fromCLI.taskPromptsFile == fromGUI.taskPromptsFile)
            #expect(fromCLI.taskPromptsHash == fromGUI.taskPromptsHash)
        }
    }

    // MARK: - Draft-only

    /// Every authoring verb refuses on a non-draft with the ONE immutability
    /// line `updateDraft` owns — not a per-verb variation of it.
    @Test func everyAuthoringVerbRefusesOnAFrozenStudy() async throws {
        try await withTempRoot { root in
            await invoke(["create", "sealed", "--model", Self.model])
            await invoke(["attach", "sealed", "french"])
            try write(
                Self.taskPrompts,
                to: root.appending(components: "prompts", "cases", "items.jsonl"))
            await invoke(["pin-prompts", "sealed", "prompts/cases/items.jsonl"])
            await invoke(
                [
                    "declare-condition", "sealed", "french-hi",
                    "--slots", "french:17:0.4", "--alpha-units", "norm",
                ])
            #expect(await invoke(["freeze", "sealed", "--force"]).exitCode == 0)

            let attempts: [[String]] = [
                ["pin-prompts", "sealed", "prompts/cases/items.jsonl"],
                ["pin-rubric", "sealed", "prompts/rubrics/default-paired-v1.md"],
                [
                    "declare-condition", "sealed", "other", "--slots",
                    "french:9:0.2", "--alpha-units", "norm",
                ],
                ["set-sampling", "sealed", "--temperature", "0.7"],
                ["set-exclusions", "sealed", "unparseableEndpoint"],
            ]
            for attempt in attempts {
                let outcome = await invoke(attempt)
                #expect(outcome.exitCode != 0, "\(attempt[0]) mutated a frozen study")
                #expect(
                    outcome.failure?.reason
                        == "experiment 'sealed' is frozen — duplicate it to iterate",
                    "\(attempt[0]) does not speak the immutability line")
            }
            // Nothing moved.
            let manifest = try ExperimentStore.load(name: "sealed")
            #expect(manifest.judgeRubricFile == nil)
            #expect(manifest.conditions.map(\.name) == ["french-hi"])
        }
    }

    // MARK: - declare-condition: the arm vocabulary

    @Test func aConditionMayNameAConceptTheStudyNeverPinned() async throws {
        try await withTempRoot { _ in
            await invoke(["create", "loose", "--model", Self.model])
            let outcome = await invoke(
                [
                    "declare-condition", "loose", "ghost-hi",
                    "--slots", "ghost:3:0.1", "--alpha-units", "norm",
                ])
            #expect(outcome.exitCode != 0)
            #expect(outcome.failure?.reason.contains("not attached") == true)
            #expect(outcome.failure?.reason.contains("attach (pin) it first") == true)
        }
    }

    /// BOTH concepts are authored HERE, in the temp workspace, rather than
    /// borrowed from the checkout's `prompts/concepts/`. That directory is
    /// study material and does not ship, so a test that reached into it
    /// passed in the research tree and failed in a released one. Two slots
    /// is the whole point of the test; whose stimuli they are is not.
    ///
    /// Attach resolves stimuli through `WorkspaceRoot.current`, which is a
    /// SEPARATE seam from `ExperimentStore.rootOverride`, so this test
    /// points both at the same temp tree. (Serialized suite, holding the
    /// shared root-override lock via `withTempRoot`.)
    @Test func aMultiSlotConditionIsTheLinearMixAndHashesAsOneCondition() async throws {
        try await withTempRoot { root in
            let previousWorkspace = WorkspaceRoot.programmaticOverride
            WorkspaceRoot.programmaticOverride = root
            defer { WorkspaceRoot.programmaticOverride = previousWorkspace }

            for (name, positive, negative) in [
                (
                    "cadence",
                    "The sentence moved in a slow, even measure.",
                    "the sentence just sort of went wherever it went"
                ),
                (
                    "brevity",
                    "He declined.",
                    "He indicated that it would not be possible for him to accept."
                ),
            ] {
                let directory = root.appending(path: "prompts/concepts/\(name)")
                try write(
                    #"{"text":"\#(positive)"}"# + "\n",
                    to: directory.appending(component: "positive.jsonl"))
                try write(
                    #"{"text":"\#(negative)"}"# + "\n",
                    to: directory.appending(component: "negative.jsonl"))
            }

            await invoke(["create", "mix", "--model", Self.model])
            #expect(await invoke(["attach", "mix", "cadence"]).exitCode == 0)
            #expect(await invoke(["attach", "mix", "brevity"]).exitCode == 0)
            let outcome = await invoke(
                [
                    "declare-condition", "mix", "both",
                    "--slots", "cadence:17:0.4,brevity:17:-0.2", "--band-width", "3",
                    "--alpha-units", "raw",
                ])
            #expect(outcome.exitCode == 0)
            let manifest = try ExperimentStore.load(name: "mix")
            let condition = try #require(manifest.conditions.first { $0.name == "both" })
            #expect(condition.slots.count == 2)
            #expect(condition.slots.map(\.concept) == ["cadence", "brevity"])
            #expect(condition.slots[1].alpha == -0.2)
            #expect(condition.bandWidth == 3)
            #expect(!condition.alphaInNormUnits)
            #expect(manifest.conditions.filter { $0.name == "both" }.count == 1)
        }
    }

    /// `--alpha-units` is REQUIRED, and a declaration without it is refused
    /// with the repair — including for a baseline.
    ///
    /// Phase-0 gap G6 (`docs/PORTABILITY-CONTRACTS.md`): the flag used to
    /// default to `norm` here while the server's `_condition_entry` defaulted
    /// to raw α, so the same undeclared arm authored a different study
    /// depending on which engine served it. α units are dose semantics — the
    /// same number is a different intervention in each convention — so neither
    /// engine guesses now. Server twin:
    /// `test_a_new_condition_that_declares_no_alpha_units_is_refused`.
    @Test func declaringAnArmWithoutItsAlphaUnitsIsRefused() async throws {
        try await withTempRoot { _ in
            await invoke(["create", "units", "--model", Self.model])
            for attempt in [
                ["declare-condition", "units", "arm", "--slots", "french:17:0.4"],
                ["declare-condition", "units", "baseline", "--baseline"],
            ] {
                let outcome = await invoke(attempt)
                #expect(outcome.exitCode != 0, "\(attempt[2]) was declared with no α unit")
                #expect(
                    outcome.failure?.reason.contains("--alpha-units norm|raw") == true,
                    "the refusal does not name the flag: \(outcome.failure?.reason ?? "")")
                let repair = outcome.failure?.repairAction ?? ""
                #expect(repair == ExperimentManifest.alphaUnitsRepairAction)
                // Both spellings: the manifest key and the CLI flag.
                #expect(repair.contains("\"alphaInNormUnits\""))
                #expect(repair.contains("--alpha-units norm|raw"))
                // Nothing was written.
                #expect((try? ExperimentStore.load(name: "units"))?.conditions == [])
            }
            // An out-of-vocabulary value is still its own refusal, naming the
            // two legal values rather than the missing-flag repair.
            let wrong = await invoke([
                "declare-condition", "units", "arm", "--slots", "french:17:0.4",
                "--alpha-units", "normal",
            ])
            #expect(wrong.exitCode != 0)
            #expect(wrong.failure?.reason.contains("norm | raw") == true)
        }
    }

    @Test func anAblationSlotCarriesItsModeAndAnAddSlotDoesNot() throws {
        let ablate = try ExperimentCLIRunner.parseSlots("fear:20:1.0:ablate")
        #expect(ablate[0].mode == .ablate)
        #expect(ablate[0].effectiveMode == .ablate)
        // An explicit `add` parses and is then dropped: writing the key on
        // every condition would re-identify every frozen study in the
        // workspace (manifest bytes ARE the content hash).
        let add = try ExperimentCLIRunner.parseSlots("fear:20:0.3:add")
        #expect(add[0].mode == nil)
        #expect(add[0].effectiveMode == .add)
    }

    @Test func aMalformedSlotNamesTheShapeItExpected() {
        for bad in ["fear", "fear:20", "fear:x:0.3", "fear:20:high", ":20:0.3"] {
            #expect(throws: ExperimentError.self) {
                _ = try ExperimentCLIRunner.parseSlots(bad)
            }
        }
        do {
            _ = try ExperimentCLIRunner.parseSlots("fear:20")
            Issue.record("a two-field slot should not parse")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("<concept>:<layer>:<alpha>"))
        } catch {
            Issue.record("wrong error type")
        }
        // An unknown mode names the closed vocabulary.
        do {
            _ = try ExperimentCLIRunner.parseSlots("fear:20:0.3:erase")
            Issue.record("an unknown mode should not parse")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("add"))
            #expect(error.reason.contains("ablate"))
        } catch {
            Issue.record("wrong error type")
        }
    }

    @Test func aControlCellIsDeclaredWithTheManifestsOwnVocabulary() async throws {
        try await withTempRoot { _ in
            await invoke(["create", "ctrl", "--model", Self.model])
            await invoke(["attach", "ctrl", "french"])
            await invoke(
                [
                    "declare-condition", "ctrl", "french-hi",
                    "--slots", "french:17:0.4", "--alpha-units", "norm",
                ])
            #expect(
                await invoke(
                    [
                        "declare-condition", "ctrl", "french-hi-random",
                        "--slots", "french:17:0.4", "--alpha-units", "norm",
                        "--control", "randomMatchedNorm",
                    ]
                ).exitCode == 0)
            let manifest = try ExperimentStore.load(name: "ctrl")
            #expect(
                manifest.conditions.first { $0.name == "french-hi-random" }?
                    .controlType == "randomMatchedNorm")
            // It is the same cell the store's own control constructor builds.
            let treatment = try #require(
                manifest.conditions.first { $0.name == "french-hi" })
            let constructed = ExperimentStore.randomControlCondition(for: treatment)
            #expect(constructed.controlType == "randomMatchedNorm")
            #expect(constructed.slots == treatment.slots)

            // An unknown control type is refused, not written.
            let bad = await invoke(
                [
                    "declare-condition", "ctrl", "nope", "--slots", "french:17:0.4",
                    "--alpha-units", "norm", "--control", "randomish",
                ])
            #expect(bad.exitCode != 0)
            #expect(bad.failure?.reason.contains("randomMatchedNorm") == true)
        }
    }

    @Test func baselineAndSlotsAreExclusiveAndOneOfThemIsRequired() async throws {
        try await withTempRoot { _ in
            await invoke(["create", "excl", "--model", Self.model])
            await invoke(["attach", "excl", "french"])
            let both = await invoke(
                [
                    "declare-condition", "excl", "x", "--baseline",
                    "--slots", "french:17:0.4", "--alpha-units", "norm",
                ])
            #expect(both.exitCode != 0)
            #expect(both.failure?.reason.contains("exclusive") == true)

            let neither = await invoke(
                ["declare-condition", "excl", "x", "--alpha-units", "norm"])
            #expect(neither.exitCode != 0)
            #expect(neither.failure?.reason.contains("--slots") == true)
            #expect(neither.failure?.reason.contains("--baseline") == true)
        }
    }

    // MARK: - pin-rubric: judges and the declaration

    @Test func aJudgePanelParsesIntoTheManifestsJudgeVocabulary() throws {
        let judges = try ExperimentCLIRunner.parseJudges(
            "j-1:claude,j-2:local:,j-3:openrouter:vendor/model:together")
        #expect(judges.map(\.name) == ["j-1", "j-2", "j-3"])
        #expect(judges.map(\.kind) == ["claude", "local", "openrouter"])
        // A blank model is ABSENT, not empty: a local judge resolves to the
        // study model, a claude judge to the default judge model.
        #expect(judges[1].model == nil)
        #expect(judges[2].model == "vendor/model")
        #expect(judges[2].provider == "together")
        // The provider is OpenRouter's pin alone.
        #expect(judges[0].provider == nil)

        #expect(throws: ExperimentError.self) {
            _ = try ExperimentCLIRunner.parseJudges("j-1:anthropic")
        }
        #expect(throws: ExperimentError.self) {
            _ = try ExperimentCLIRunner.parseJudges("j-1:local:m:together")
        }
        #expect(throws: ExperimentError.self) {
            _ = try ExperimentCLIRunner.parseJudges("j-1")
        }
    }

    @Test func aSingleJudgeIsAdvisedAgainstButNeverRefused() async throws {
        try await withTempRoot { _ in
            await invoke(["create", "one-judge", "--model", Self.model])
            let outcome = await invoke(
                [
                    "pin-rubric", "one-judge",
                    JudgeRubricStore.defaultRubricFile, "--judges", "solo:claude",
                ])
            // A one-judge DRAFT is legal; freeze is where it stops.
            #expect(outcome.exitCode == 0)
            #expect(
                (outcome.envelope.advisories ?? []).contains {
                    $0.code == "judgePanelTooSmall"
                })
        }
    }

    // MARK: - The refusals that named GUI-only remedies

    /// Each of these messages was a dead end for a headless caller. The
    /// remedy now exists, so the message names it — that is the whole
    /// contract of a repair action.
    @Test func everyUpdatedRefusalNamesTheVerbThatPerformsIt() throws {
        var manifest = ExperimentManifest(
            name: "nothing", description: "", modelID: "test/model")
        manifest.concepts = [
            .init(
                name: "fear", stimulusSetHash: "h",
                options: ExtractionOptions(method: .meanDifference))
        ]

        // run's no-conditions refusal (2026-08-17) said "declare a condition
        // naming a concept, layer and alpha" — an operation with no verb.
        let noArms = try #require(ExperimentStore.noMeasuredConditionsProblem(manifest))
        #expect(noArms.contains("experiment declare-condition nothing"))
        #expect(noArms.contains("--slots <concept>:<layer>:<alpha>"))
        // The remedies it already named are untouched.
        #expect(noArms.contains("promote"))
        #expect(noArms.contains("agentComparison"))

        // evaluate's rubric refusal said "enter draft rubric text" — a field
        // that exists only in the Studies panel.
        do {
            _ = try JudgeRubricStore.resolveRubric(for: manifest, inlineRubric: nil)
            Issue.record("a study with no rubric must not resolve one")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("experiment pin-rubric nothing"))
            #expect(!error.reason.contains("enter draft rubric text"))
        }

    }

    /// freeze's judge gate named "pin a judge rubric file from
    /// prompts/rubrics/" — a location, not an operation. Reached for real,
    /// through the CLI, so the assertion covers the thrown prose AND the
    /// envelope's `repairAction` (which is what an agent reads).
    @Test func theJudgeValidityGateNamesTheVerbThatPinsARubric() async throws {
        try await withTempRoot { root in
            await invoke(["create", "judged", "--model", Self.model])
            await invoke(["attach", "judged", "french"])
            try write(
                Self.taskPrompts,
                to: root.appending(components: "prompts", "cases", "items.jsonl"))
            await invoke(["pin-prompts", "judged", "prompts/cases/items.jsonl"])
            await invoke(
                [
                    "declare-condition", "judged", "french-hi",
                    "--slots", "french:17:0.4", "--alpha-units", "norm",
                ])
            // Judges with no rubric: exactly the shape the gate exists for.
            var manifest = try ExperimentStore.load(name: "judged")
            manifest.judges = [
                .init(name: "j-1", kind: "claude"),
                .init(name: "j-2", kind: "claude", model: "other"),
            ]
            try ExperimentStore.save(manifest)

            let outcome = await invoke(["freeze", "judged"])
            #expect(outcome.exitCode != 0)
            let gates = try #require(outcome.envelope.error?.gates)
            #expect(gates.contains(FreezeGate.judgeValidity.rawValue))

            // The gate's OWN prose (the refusal names whichever gate fires
            // first in the historical order, which need not be this one).
            let judgeGate = try #require(
                ExperimentStore.freezeGateTable(name: "judged", autoCommit: false)
                    .filter { $0.gate == .judgeValidity }
                    .compactMap { $0.evaluate(manifest) }
                    .first)
            #expect(judgeGate.refusal.contains("experiment pin-rubric judged"))
            #expect(judgeGate.repairAction.contains("experiment pin-rubric judged"))
            #expect(judgeGate.repairAction.contains("--judges"))
        }
    }

    /// The run path's frozen-study refusal, which told the caller to
    /// "duplicate, pin a prompt set, and re-freeze" — two of those three
    /// were verbs and one was not.
    @Test func theFrozenUnpinnedPromptsRefusalNamesBothVerbs() throws {
        var manifest = ExperimentManifest(
            name: "sealed", description: "", modelID: "test/model")
        manifest.status = .frozen
        manifest.taskPromptsFile = nil
        manifest.taskPromptsHash = nil
        do {
            _ = try ExperimentTasks.loadTaskPrompts(for: manifest)
            Issue.record("a frozen study with no pinned prompts must refuse")
        } catch let error as ExperimentError {
            #expect(error.reason.contains("experiment duplicate sealed"))
            #expect(error.reason.contains("experiment pin-prompts"))
        }
    }
}
