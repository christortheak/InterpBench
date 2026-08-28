import CryptoKit
import Foundation
import Testing
@testable import ExperimentKit

/// Local-judge model resolution for judgeScore sweeps (cross-engine rule,
/// 2026-07-08): a local judge with an empty/absent `model` resolves to the
/// STUDY model (`manifest.modelID`), logged at sweep start; a same-model
/// local judge generates through the sweep's ALREADY-LOADED study container
/// (never a second load); a different-model local judge refuses AT SWEEP
/// START on the local engine, which holds one loaded model. Pure-CPU — the
/// resolution, routing, and messaging are pure seams.
struct SweepJudgeResolutionTests {

    private func manifest(
        judges: [ExperimentManifest.JudgeRef]
    ) -> ExperimentManifest {
        var manifest = ExperimentManifest(
            name: "jr", description: "", modelID: "test/model")
        manifest.judges = judges
        return manifest
    }

    // MARK: (a) empty model → the study model

    @Test func emptyModelLocalJudgeResolvesToStudyModel() throws {
        let resolved = try ExperimentTasks.resolvedJudges(
            manifest: manifest(judges: [
                .init(name: "loc", kind: "local", model: nil),
                .init(name: "loc2", kind: "local", model: "   "),
                .init(name: "named", kind: "local", model: "test/model"),
            ]),
            evaluation: nil)
        // nil AND whitespace-only both mean "the study model" — never
        // judge-name-as-model-id, never a refusal.
        #expect(resolved.map(\.model) == ["test/model", "test/model", "test/model"])
        #expect(resolved.map(\.modelDefaulted) == [true, true, false])
        #expect(resolved.allSatisfy { $0.kind == "local" })
    }

    @Test func claudeJudgeBlankModelKeepsItsOwnDefault() throws {
        let resolved = try ExperimentTasks.resolvedJudges(
            manifest: manifest(judges: [.init(name: "cl", kind: "claude", model: nil)]),
            evaluation: nil)
        #expect(resolved.first?.model == ClaudePairedJudge.defaultModel)
        #expect(resolved.first?.modelDefaulted == false)
    }

    // MARK: openrouter judges — no defaults, provider is a pin (2026-07-19)

    @Test func openRouterJudgeResolvesWithItsPins() throws {
        let resolved = try ExperimentTasks.resolvedJudges(
            manifest: manifest(judges: [
                .init(name: "or", kind: "openrouter",
                      model: "google/gemma-3-27b-it", provider: "DeepInfra")
            ]),
            evaluation: nil)
        #expect(resolved.first?.kind == "openrouter")
        #expect(resolved.first?.model == "google/gemma-3-27b-it")
        #expect(resolved.first?.provider == "DeepInfra")
    }

    @Test func openRouterJudgeRefusesMissingModelOrProvider() {
        // No defaults to fill: a blank slug and a blank provider each
        // refuse at resolution, never mid-run.
        #expect(throws: ExperimentError.self) {
            _ = try ExperimentTasks.resolvedJudges(
                manifest: manifest(judges: [
                    .init(name: "or", kind: "openrouter", provider: "DeepInfra")
                ]),
                evaluation: nil)
        }
        #expect(throws: ExperimentError.self) {
            _ = try ExperimentTasks.resolvedJudges(
                manifest: manifest(judges: [
                    .init(name: "or", kind: "openrouter",
                          model: "google/gemma-3-27b-it")
                ]),
                evaluation: nil)
        }
    }

    @Test func openRouterJudgeRoutesToTheOpenRouterAPI() throws {
        let judge = ExperimentTasks.ResolvedJudge(
            name: "or", kind: "openrouter", model: "google/gemma-3-27b-it",
            provider: "DeepInfra")
        let route = try ExperimentTasks.sweepJudgeRoute(
            judge, studyModelID: "test/model", studyRevision: nil)
        #expect(route == .openRouterAPI(
            model: "google/gemma-3-27b-it", provider: "DeepInfra"))
        // Belt-and-braces: a hand-built panel entry without a provider
        // throws instead of routing anywhere.
        let bare = ExperimentTasks.ResolvedJudge(
            name: "or", kind: "openrouter", model: "google/gemma-3-27b-it")
        #expect(throws: ExperimentError.self) {
            _ = try ExperimentTasks.sweepJudgeRoute(
                bare, studyModelID: "test/model", studyRevision: nil)
        }
    }

    // MARK: (b)/(c) the one-model-slot rule, as pure seams

    @Test func slotProblemFiresOnlyForNonStudyLocalModels() {
        let same = ExperimentTasks.ResolvedJudge(
            name: "a", kind: "local", model: "test/model")
        let other = ExperimentTasks.ResolvedJudge(
            name: "b", kind: "local", model: "other/model")
        let claude = ExperimentTasks.ResolvedJudge(
            name: "c", kind: "claude", model: "other/model")
        // Same-model local judges and Claude judges (whatever their model)
        // never trip the slot rule.
        #expect(
            ExperimentTasks.localJudgeSlotProblem(
                [same, claude], studyModelID: "test/model",
                studyRevision: nil) == nil)
        let problem = ExperimentTasks.localJudgeSlotProblem(
            [same, other], studyModelID: "test/model", studyRevision: nil)
        #expect(problem?.contains("local judge 'b' uses model 'other/model'") == true)
        #expect(problem?.contains("not the study model 'test/model'") == true)
        #expect(problem?.contains("holds one loaded model") == true)
        #expect(
            problem?.contains("use the study model as judge or a claude judge")
                == true)
    }

    /// The slot the sweep holds is the STUDY's, loaded at the study's pin —
    /// so a judge that pins a different commit of the same repo is refused
    /// rather than quietly judged with weights it never declared (review
    /// round 9, finding 1's audit).
    @Test func slotProblemFiresOnADivergentRevisionOfTheStudyModel() {
        let pinned = ExperimentTasks.ResolvedJudge(
            name: "a", kind: "local", model: "test/model",
            revision: String(repeating: "a", count: 40))
        #expect(
            ExperimentTasks.localJudgeSlotProblem(
                [pinned], studyModelID: "test/model",
                studyRevision: String(repeating: "a", count: 40)) == nil)
        let problem = ExperimentTasks.localJudgeSlotProblem(
            [pinned], studyModelID: "test/model",
            studyRevision: String(repeating: "b", count: 40))
        #expect(problem?.contains("local judge 'a' pins revision aaaaaaaaaaaa…") == true)
        #expect(problem?.contains("which pins bbbbbbbbbbbb…") == true)
        #expect(
            problem?.contains("would judge with weights it did not declare")
                == true)
        // No pin on either side is not a divergence.
        let bare = ExperimentTasks.ResolvedJudge(
            name: "b", kind: "local", model: "test/model")
        #expect(
            ExperimentTasks.localJudgeSlotProblem(
                [bare], studyModelID: "test/model", studyRevision: nil) == nil)
        // …but a judge pinning where the study pins nothing still is: the
        // held container is whatever the cache resolved, not that commit.
        #expect(
            ExperimentTasks.localJudgeSlotProblem(
                [pinned], studyModelID: "test/model", studyRevision: nil)?
                .contains("which pins no revision") == true)
        #expect(throws: ExperimentError.self) {
            _ = try ExperimentTasks.sweepJudgeRoute(
                pinned, studyModelID: "test/model", studyRevision: nil)
        }
    }

    /// The held slot has a DTYPE as well as a revision, and it is the study's
    /// (review round 10, finding 10 — sorry, finding 2). `--judge-pin
    /// <name>=<revision>:<dtype>` makes a judge's precision declarable, so a
    /// same-model judge could pin one the container was never loaded at and
    /// have the provenance say so.
    @Test func slotProblemFiresOnADivergentDtypeOfTheStudyModel() {
        let pinned = ExperimentTasks.ResolvedJudge(
            name: "a", kind: "local", model: "test/model", dtype: "float16")
        // Agreement is silent, and it is CANONICAL agreement: `bf16` and
        // `bfloat16` are one dtype, not two.
        #expect(
            ExperimentTasks.localJudgeSlotProblem(
                [pinned], studyModelID: "test/model", studyRevision: nil,
                studyDtype: "float16") == nil)
        #expect(
            ExperimentTasks.localJudgeSlotProblem(
                [ExperimentTasks.ResolvedJudge(
                    name: "a", kind: "local", model: "test/model",
                    dtype: "bf16")],
                studyModelID: "test/model", studyRevision: nil,
                studyDtype: "bfloat16") == nil)
        // A judge with NO dtype inherits the study's, whatever that is.
        #expect(
            ExperimentTasks.localJudgeSlotProblem(
                [ExperimentTasks.ResolvedJudge(
                    name: "b", kind: "local", model: "test/model")],
                studyModelID: "test/model", studyRevision: nil,
                studyDtype: "bfloat16") == nil)

        let problem = ExperimentTasks.localJudgeSlotProblem(
            [pinned], studyModelID: "test/model", studyRevision: nil,
            studyDtype: "bfloat16")
        // Both dtypes named, and the repair spelled out.
        #expect(problem?.contains("local judge 'a' pins dtype 'float16'") == true)
        #expect(problem?.contains("which pins 'bfloat16'") == true)
        #expect(
            problem?.contains("would judge at a precision it did not declare")
                == true)
        #expect(
            problem?.contains(
                "drop the judge's dtype to judge with the study's, or use a "
                    + "claude judge") == true)

        // A judge pinning where the study pins NOTHING is a divergence too —
        // the container loads at whatever the device decides, not at float16.
        #expect(
            ExperimentTasks.localJudgeSlotProblem(
                [pinned], studyModelID: "test/model", studyRevision: nil)?
                .contains("which pins no dtype (the device decides)") == true)
        #expect(throws: ExperimentError.self) {
            _ = try ExperimentTasks.sweepJudgeRoute(
                pinned, studyModelID: "test/model", studyRevision: nil,
                studyDtype: "bfloat16")
        }
        // Claude judges are untouched by the slot rule, dtype and all.
        #expect(
            ExperimentTasks.localJudgeSlotProblem(
                [ExperimentTasks.ResolvedJudge(
                    name: "c", kind: "claude", model: "other/model",
                    dtype: "float32")],
                studyModelID: "test/model", studyRevision: nil,
                studyDtype: "bfloat16") == nil)
    }

    /// Resolution is where the inheritance happens: a study-model judge that
    /// declares no dtype comes out carrying the STUDY's, which is what makes
    /// "no declaration never refuses" true at the sweep's callsite rather than
    /// only in the pure rule.
    @Test func resolutionInheritsTheStudyDtypeForStudyModelJudges() throws {
        var manifest = ExperimentManifest(
            name: "s", description: "", modelID: "test/model")
        manifest.dtype = "bfloat16"
        manifest.judges = [
            .init(name: "blank", kind: "local"),
            .init(name: "named", kind: "local", model: "test/model"),
            .init(name: "pinned", kind: "local", model: "test/model",
                  dtype: "float16"),
        ]
        let panel = try ExperimentTasks.resolvedJudges(
            manifest: manifest, evaluation: nil)
        #expect(panel.first { $0.name == "blank" }?.dtype == "bfloat16")
        #expect(panel.first { $0.name == "named" }?.dtype == "bfloat16")
        #expect(panel.first { $0.name == "pinned" }?.dtype == "float16")
        // The first two are silent; the third is the refusal.
        #expect(
            ExperimentTasks.localJudgeSlotProblem(
                panel.filter { $0.name != "pinned" },
                studyModelID: manifest.modelID,
                studyRevision: manifest.modelRevision,
                studyDtype: manifest.dtype) == nil)
        #expect(
            ExperimentTasks.localJudgeSlotProblem(
                panel, studyModelID: manifest.modelID,
                studyRevision: manifest.modelRevision,
                studyDtype: manifest.dtype)?
                .contains("local judge 'pinned' pins dtype 'float16'") == true)
    }

    @Test func judgeRoutesUseTheHeldContainerOrTheAPI() throws {
        // A judge that DEFAULTED to the study model and one that names it
        // explicitly both route through the held study container — the
        // panel builder has no other local path, so no second load exists.
        let defaulted = ExperimentTasks.ResolvedJudge(
            name: "d", kind: "local", model: "test/model", modelDefaulted: true)
        let defaultedRoute = try ExperimentTasks.sweepJudgeRoute(
            defaulted, studyModelID: "test/model", studyRevision: nil)
        #expect(defaultedRoute == .heldStudyContainer)
        let named = ExperimentTasks.ResolvedJudge(
            name: "n", kind: "local", model: "test/model")
        let namedRoute = try ExperimentTasks.sweepJudgeRoute(
            named, studyModelID: "test/model", studyRevision: nil)
        #expect(namedRoute == .heldStudyContainer)
        let claude = ExperimentTasks.ResolvedJudge(
            name: "c", kind: "claude", model: "claude-x")
        let claudeRoute = try ExperimentTasks.sweepJudgeRoute(
            claude, studyModelID: "test/model", studyRevision: nil)
        #expect(claudeRoute == .claudeAPI(model: "claude-x"))
        // A different-model local judge throws the slot refusal instead of
        // routing anywhere.
        let other = ExperimentTasks.ResolvedJudge(
            name: "b", kind: "local", model: "other/model")
        #expect(throws: ExperimentError.self) {
            _ = try ExperimentTasks.sweepJudgeRoute(
                other, studyModelID: "test/model", studyRevision: nil)
        }
    }

    // MARK: save-time messaging (never refusals)

    @Test func saveTimeMessagingStatesTheRuleWithoutRefusing() {
        let blankPanel: [ExperimentManifest.JudgeRef] = [
            .init(name: "blank", kind: "local", model: nil),
            .init(name: "cl", kind: "claude", model: nil),
        ]
        let note = SweepSpecForm.localJudgeDefaultNote(
            judges: blankPanel, studyModelID: "test/model")
        #expect(note?.contains("'blank'") == true)
        #expect(note?.contains("legal") == true)
        #expect(note?.contains("study model") == true)
        #expect(note?.contains("test/model") == true)
        // Blank is legal — the slot warning never fires for it.
        #expect(
            SweepSpecForm.localJudgeSlotWarning(
                judges: blankPanel, studyModelID: "test/model") == nil)

        let differentPanel: [ExperimentManifest.JudgeRef] = [
            .init(name: "wrong", kind: "local", model: "other/model")
        ]
        let warning = SweepSpecForm.localJudgeSlotWarning(
            judges: differentPanel, studyModelID: "test/model")
        #expect(warning?.contains("local judge 'wrong'") == true)
        #expect(warning?.contains("refuse at start") == true)
        #expect(
            SweepSpecForm.localJudgeDefaultNote(
                judges: differentPanel, studyModelID: "test/model") == nil)
    }

    @Test func objectiveRequirementsAcceptBlankLocalJudgeModels() {
        // Save-time validation refuses missing rubric/judge PINS, never a
        // blank local judge model (blank = the study model).
        var manifest = ExperimentManifest(
            name: "ok", description: "", modelID: "test/model")
        manifest.judgeRubricFile = "prompts/rubrics/r.md"
        manifest.judgeRubricHash = String(repeating: "a", count: 64)
        manifest.judges = [.init(name: "blank", kind: "local", model: nil)]
        let spec = ExperimentManifest.SweepSelection(
            objective: .init(metric: "judgeScore"))
        #expect(
            SweepSpecForm.validateObjectiveRequirements(spec, manifest: manifest)
                == nil)
    }

    @Test func verifyNoLongerFlagsBlankLocalJudgeModels() {
        var manifest = ExperimentManifest(
            name: "vf", description: "", modelID: "test/model")
        manifest.judges = [.init(name: "blank", kind: "local", model: nil)]
        let violations = ExperimentStore.verify(manifest)
        #expect(!violations.contains { $0.contains("blank") })
    }
}

/// Sweep-START resolution against a real workspace (rubric file on disk) —
/// extends the serialized `ExperimentStoreTests` suite because it uses the
/// process-global workspace override.
extension ExperimentStoreTests {

    private func withJudgeWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "judge-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        // Shared cross-suite lock: the workspace root is process-global
        // (see ExperimentRootOverrideLock).
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = temp
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: temp)
        }
        return try body(temp)
    }

    private func judgeScoreManifest(
        root: URL, judges: [ExperimentManifest.JudgeRef]
    ) throws -> ExperimentManifest {
        let rubricDirectory = root.appending(components: "prompts", "rubrics")
        try FileManager.default.createDirectory(
            at: rubricDirectory, withIntermediateDirectories: true)
        let text = "Which response expresses more dread?\n"
        try text.write(
            to: rubricDirectory.appending(component: "r.md"),
            atomically: true, encoding: .utf8)
        var manifest = ExperimentManifest(
            name: "jsl", description: "", modelID: "test/model")
        manifest.judgeRubricFile = "prompts/rubrics/r.md"
        manifest.judgeRubricHash = SHA256.hash(data: Data(text.utf8))
            .map { String(format: "%02x", $0) }.joined()
        manifest.judges = judges
        return manifest
    }

    // MARK: per-concept choice files (choicePromptsFiles, 2026-08-02)

    private func choiceManifest(concepts: [String]) -> ExperimentManifest {
        var manifest = ExperimentManifest(
            name: "lpmap", description: "", modelID: "test/model")
        manifest.concepts = concepts.map {
            ExperimentManifest.ConceptRef(
                name: $0, stimulusSetHash: "h", options: .init())
        }
        return manifest
    }

    private func writeChoices(
        _ root: URL, _ name: String, target: String = "A"
    ) throws -> String {
        let dev = root.appending(components: "prompts", "dev")
        try FileManager.default.createDirectory(
            at: dev, withIntermediateDirectories: true)
        let line = #"{"id": "\#(name)-1", "text": "item", "#
            + #""options": ["A", "B"], "target": "\#(target)"}"#
        let data = Data((line + "\n").utf8)
        try data.write(to: dev.appending(component: "\(name).jsonl"))
        return SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }.joined()
    }

    @Test func perConceptChoiceFilesResolveEachConceptsOwnInstrument() throws {
        try withJudgeWorkspace { root in
            let fearHash = try writeChoices(root, "fear-choices")
            let hopeHash = try writeChoices(root, "hope-choices", target: "B")
            let manifest = choiceManifest(concepts: ["fear", "hope"])
            let criterion = try SweepSelectionRule.resolve(
                .init(objective: .init(metric: "logprobShift")))
            let objective = try SweepSelectionRule.resolveObjective(
                criterion: criterion,
                spec: .init(objective: .init(
                    metric: "logprobShift",
                    choicePromptsFiles: [
                        "fear": "prompts/dev/fear-choices.jsonl",
                        "hope": "prompts/dev/hope-choices.jsonl",
                    ])),
                manifest: manifest, root: root)
            let fear = try objective.choiceSet(for: "fear")
            let hope = try objective.choiceSet(for: "hope")
            #expect(fear.hash == fearHash)
            #expect(hope.hash == hopeHash)
            #expect(hope.rows.map(\.target) == ["B"])
            // Provenance pins the concept's OWN instrument.
            let block = criterion.asCriterion(
                objective: objective, concept: "hope").objective
            #expect(block?.choicePromptsFile == "prompts/dev/hope-choices.jsonl")
            #expect(block?.choicePromptsHash == hopeHash)
        }
    }

    @Test func perConceptChoiceFilesRefuseGapsTyposAndDoubleDeclaration() throws {
        try withJudgeWorkspace { root in
            _ = try writeChoices(root, "fear-choices")
            let criterion = try SweepSelectionRule.resolve(
                .init(objective: .init(metric: "logprobShift")))
            func resolve(
                concepts: [String],
                objective: ExperimentManifest.SweepSelection.Objective
            ) throws -> SweepSelectionRule.ResolvedObjective {
                try SweepSelectionRule.resolveObjective(
                    criterion: criterion, spec: .init(objective: objective),
                    manifest: choiceManifest(concepts: concepts), root: root)
            }
            // Server-identical refusal strings (cross-engine contract).
            func refusal(
                _ concepts: [String],
                _ objective: ExperimentManifest.SweepSelection.Objective
            ) -> String {
                do {
                    _ = try resolve(concepts: concepts, objective: objective)
                    Issue.record("expected a refusal")
                    return ""
                } catch let error as ExperimentError {
                    return error.reason
                } catch {
                    Issue.record("unexpected error \(error)")
                    return ""
                }
            }
            #expect(refusal(
                ["fear", "hope"],
                .init(metric: "logprobShift",
                      choicePromptsFiles: ["fear": "prompts/dev/fear-choices.jsonl"]))
                .contains("missing concepts this sweep would select for: hope"))
            #expect(refusal(
                ["fear"],
                .init(metric: "logprobShift",
                      choicePromptsFiles: [
                        "fear": "prompts/dev/fear-choices.jsonl",
                        "typo": "prompts/dev/fear-choices.jsonl",
                      ]))
                .contains("names concepts the study does not attach: typo"))
            #expect(refusal(
                ["fear"],
                .init(metric: "logprobShift",
                      choicePromptsFile: "prompts/dev/fear-choices.jsonl",
                      choicePromptsFiles: ["fear": "prompts/dev/fear-choices.jsonl"]))
                .contains("declare exactly one"))
            #expect(refusal(
                ["fear"],
                .init(metric: "logprobShift", choicePromptsFiles: [:]))
                .contains("non-empty object"))
            // A single declared file still resolves for every concept.
            let objective = try resolve(
                concepts: ["fear", "hope"],
                objective: .init(
                    metric: "logprobShift",
                    choicePromptsFile: "prompts/dev/fear-choices.jsonl"))
            #expect(try objective.choiceSet(for: "hope").rows.count == 1)
        }
    }

    @Test func perConceptChoiceFilesJoinThePinSurface() throws {
        var manifest = choiceManifest(concepts: ["fear"])
        manifest.sweep = .init(
            selection: .init(objective: .init(
                metric: "logprobShift",
                choicePromptsFiles: ["fear": "prompts/dev/fear-choices.jsonl"])))
        let labels = ExperimentStore.pinnedInputEntries(manifest)
            .filter(\.required).map(\.label)
        #expect(labels.contains("sweep choice prompts 'fear'"))
    }

    @Test func sweepStartDefaultsEmptyLocalJudgeModelToStudyModel() throws {
        try withJudgeWorkspace { root in
            let manifest = try judgeScoreManifest(
                root: root, judges: [.init(name: "j1", kind: "local", model: nil)])
            let criterion = try SweepSelectionRule.resolve(
                .init(objective: .init(metric: "judgeScore")))
            let objective = try SweepSelectionRule.resolveObjective(
                criterion: criterion, spec: nil, manifest: manifest,
                hasClaudeCredential: false)
            // The EXECUTING panel runs the study model; the provenance
            // block keeps the manifest's judges verbatim (model absent).
            #expect(objective.judgePanel.map(\.model) == ["test/model"])
            #expect(objective.judgePanel.first?.modelDefaulted == true)
            #expect(objective.judges == manifest.judges)
        }
    }

    @MainActor
    @Test func panelSaveProtocolPreservesOpenRouterProviderPins() throws {
        // Round-trip regression (engineer review 2026-07-19): saveProtocol
        // reconstructs JudgeRef field by field, and a reconstruction that
        // drops `provider` silently invalidates every openrouter judge on
        // the next save. Load → save → reload must be identity.
        try ExperimentRootOverrideLock.withTempRoot(prefix: "orsave") { root in
            // saveProtocol pins the default task-prompts file; without it
            // on disk the save throws inside its catch and the "round
            // trip" would compare the PLANTED bytes to themselves
            // (vacuous — found 2026-08-07).
            let promptsDir = root.appendingPathComponent("prompts/dev")
            try FileManager.default.createDirectory(
                at: promptsDir, withIntermediateDirectories: true)
            try Data("{\"prompt\": \"p1\"}\n".utf8).write(
                to: promptsDir.appendingPathComponent("dev-prompts.jsonl"))
            var manifest = ExperimentManifest(
                name: "or-save", description: "d", modelID: "test/model")
            manifest.judges = [
                .init(name: "or", kind: "openrouter",
                      model: "google/gemma-3-27b-it", provider: "DeepInfra"),
                .init(name: "cl", kind: "claude"),
            ]
            try ExperimentStore.save(manifest, allowCreate: true)

            let panel = ExperimentPanel()
            panel.selectedName = "or-save"
            panel.saveProtocol()

            let reloaded = try ExperimentStore.load(name: "or-save")
            // Canary that the save actually ran (it pins the prompts file).
            #expect(reloaded.taskPromptsHash != nil)
            #expect(reloaded.judges == manifest.judges)
            #expect(reloaded.judges?.first?.provider == "DeepInfra")
        }
    }

    @Test func sweepStartGatesOpenRouterJudgesOnTheJudgeKey() throws {
        // Local Swift sweeps judge INLINE only, so an openrouter judge
        // needs its credential at sweep start — exactly the claude rule.
        try withJudgeWorkspace { root in
            let manifest = try judgeScoreManifest(
                root: root,
                judges: [.init(name: "or", kind: "openrouter",
                               model: "google/gemma-3-27b-it",
                               provider: "DeepInfra")])
            let criterion = try SweepSelectionRule.resolve(
                .init(objective: .init(metric: "judgeScore")))
            do {
                _ = try SweepSelectionRule.resolveObjective(
                    criterion: criterion, spec: nil, manifest: manifest,
                    hasClaudeCredential: true, hasOpenRouterCredential: false)
                Issue.record("keyless openrouter judge must refuse at sweep start")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("'or' (openrouter)"))
                #expect(error.reason.contains("OPENROUTER_API_KEY"))
            }
            let objective = try SweepSelectionRule.resolveObjective(
                criterion: criterion, spec: nil, manifest: manifest,
                hasClaudeCredential: true, hasOpenRouterCredential: true)
            #expect(objective.judgePanel.map(\.provider) == ["DeepInfra"])
        }
    }

    @Test func sweepStartRefusesDifferentModelLocalJudge() throws {
        try withJudgeWorkspace { root in
            let manifest = try judgeScoreManifest(
                root: root,
                judges: [.init(name: "j1", kind: "local", model: "other/model")])
            let criterion = try SweepSelectionRule.resolve(
                .init(objective: .init(metric: "judgeScore")))
            do {
                _ = try SweepSelectionRule.resolveObjective(
                    criterion: criterion, spec: nil, manifest: manifest,
                    hasClaudeCredential: false)
                Issue.record("different-model local judge must refuse at sweep start")
            } catch let error as ExperimentError {
                #expect(
                    error.reason.contains("local judge 'j1' uses model 'other/model'"))
                #expect(error.reason.contains("holds one loaded model"))
            }
        }
    }

    // MARK: topK control selection (2026-08-03)

    @Test func controlApplyToResolvesAndRefusesLikeTheServer() throws {
        let resolved = try SweepSelectionRule.resolve(
            .init(
                objective: .init(metric: "markerDensity"),
                controls: .init(
                    matchedNormRandomMargin: 0, applyTo: "topK", topK: 3)))
        #expect(resolved.controlApplyTo == "topK")
        #expect(resolved.controlTopK == 3)
        let encoded = resolved.asCriterion.controls
        #expect(encoded?.applyTo == "topK" && encoded?.topK == 3)
        // Historical shape unchanged when absent.
        let winner = try SweepSelectionRule.resolve(
            .init(controls: .init(matchedNormRandomMargin: 0.1)))
        #expect(winner.controlApplyTo == "winner" && winner.controlTopK == nil)
        #expect(winner.asCriterion.controls?.applyTo == nil)
        // Server-identical refusals.
        func refusal(_ controls: ExperimentManifest.SweepSelection.Controls) -> String {
            do {
                _ = try SweepSelectionRule.resolve(.init(controls: controls))
                Issue.record("expected a refusal")
                return ""
            } catch let error as ExperimentError {
                return error.reason
            } catch { return "\(error)" }
        }
        #expect(refusal(.init(matchedNormRandomMargin: 0, applyTo: "best"))
            .contains("must be 'winner' or 'topK'"))
        #expect(refusal(.init(applyTo: "topK", topK: 3))
            .contains("declare matchedNormRandomMargin"))
        #expect(refusal(.init(matchedNormRandomMargin: 0, applyTo: "topK"))
            .contains("topK must be an integer"))
        #expect(refusal(.init(matchedNormRandomMargin: 0, topK: 3))
            .contains("only read with"))
    }

    @Test func rankedCandidatesOrderPromotableCellsOnly() throws {
        let baseline = SweepSelectionRule.Baseline(
            metric: 0, distinct2: 0.99, batteryAccuracy: 0.9)
        let criterion = try SweepSelectionRule.resolve(nil)
        let cells: [SweepSelectionRule.Cell] = [
            .init(layer: 1, alpha: 0.1, metric: 0.5, distinct2: 0.9,
                  batteryAccuracy: 0.9),
            .init(layer: 2, alpha: 0.2, metric: 2.0, distinct2: 0.9,
                  batteryAccuracy: 0.9),
            .init(layer: 3, alpha: 0.3, metric: 3.0, distinct2: 0.1,
                  batteryAccuracy: 0.9),   // incoherent — never ranks
            .init(layer: 4, alpha: 0.4, metric: -1, distinct2: 0.9,
                  batteryAccuracy: 0.9),   // below baseline — never ranks
        ]
        let ranked = SweepSelectionRule.rankedCandidates(
            cells: cells, baseline: baseline, criterion: criterion, k: 5)
        #expect(ranked.map(\.layer) == [2, 1])
        #expect(
            SweepSelectionRule.rankedCandidates(
                cells: cells, baseline: baseline, criterion: criterion, k: 1)
                .map(\.layer) == [2])
    }

    /// Cross-engine tie-break contract (2026-08-03): objective descending,
    /// then DECLARED GRID ORDER. Swift's `sorted` documents no stability,
    /// so ties carry an explicit grid index; judge scores tie in 0.5 steps,
    /// making equal metrics a real case. Server twin:
    /// `test_ranked_candidates_ties_break_by_declared_grid_order`.
    @Test func rankedCandidatesTiesBreakByDeclaredGridOrder() throws {
        let baseline = SweepSelectionRule.Baseline(
            metric: 0, distinct2: 0.99, batteryAccuracy: 0.9)
        let criterion = try SweepSelectionRule.resolve(nil)
        let cells: [SweepSelectionRule.Cell] = [
            .init(layer: 1, alpha: 0.1, metric: 0.5, distinct2: 0.9,
                  batteryAccuracy: 0.9),
            .init(layer: 1, alpha: 0.2, metric: 1.0, distinct2: 0.9,
                  batteryAccuracy: 0.9),
            .init(layer: 2, alpha: 0.1, metric: 0.5, distinct2: 0.9,
                  batteryAccuracy: 0.9),
            .init(layer: 2, alpha: 0.2, metric: 1.0, distinct2: 0.9,
                  batteryAccuracy: 0.9),
        ]
        let ranked = SweepSelectionRule.rankedCandidates(
            cells: cells, baseline: baseline, criterion: criterion, k: 4)
        #expect(ranked.map { [$0.layer, Int($0.alpha * 10)] } == [
            [1, 2], [2, 2], [1, 1], [2, 1]])
    }
}
