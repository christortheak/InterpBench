import Foundation
import Testing
@testable import ExperimentKit
@testable import SteeringKit

/// The Optimizations surface's pure machinery: sweep-spec form parsing/validation
/// (`SweepSpecForm`) and lifecycle-state derivation (`OptimizationLifecycle`).
/// Pure-CPU — no model, no filesystem; the store-backed setter tests live in
/// the serialized `ExperimentStoreTests` suite below.
struct OptimizationLifecycleTests {

    // MARK: number-list parsing (the editor's comma-separated fields)

    @Test func parsesCommaSeparatedNumbers() {
        #expect(SweepSpecForm.parseNumberList("0.35, 0.5, 0.65") == [0.35, 0.5, 0.65])
        #expect(SweepSpecForm.parseNumberList("0.1,0.2,0.4") == [0.1, 0.2, 0.4])
        // Tolerates stray whitespace and trailing commas.
        #expect(SweepSpecForm.parseNumberList(" 1 , 2 ,") == [1, 2])
        #expect(SweepSpecForm.parseNumberList("3") == [3])
    }

    @Test func rejectsEmptyAndMalformedLists() {
        #expect(SweepSpecForm.parseNumberList("") == nil)
        #expect(SweepSpecForm.parseNumberList("  ,  ") == nil)
        #expect(SweepSpecForm.parseNumberList("0.1, banana") == nil)
        #expect(SweepSpecForm.parseNumberList("0.1; 0.2") == nil)
        #expect(SweepSpecForm.parseNumberList("inf") == nil)
        #expect(SweepSpecForm.parseNumberList("nan") == nil)
    }

    @Test func numberListTextRoundTrips() {
        let values = [0.35, 0.5, 0.65, 1.0, 80.0]
        let text = SweepSpecForm.numberListText(values)
        #expect(SweepSpecForm.parseNumberList(text) == values)
        // Whole numbers render without a decimal tail.
        #expect(SweepSpecForm.numberListText([1.0, 2.0]) == "1, 2")
        #expect(SweepSpecForm.numberListText([0.1, 0.2]) == "0.1, 0.2")
    }

    // MARK: structural spec validation

    @Test func defaultSpecIsStructurallySound() {
        #expect(SweepSpecForm.validate(ExperimentManifest.SweepSpec()) == nil)
    }

    @Test func defaultGridStartsGentle() {
        // Researcher decision 2026-07-14 (live testing): stronger alphas
        // routinely push models into wasteful incoherence, and the live
        // optimum sits late in the network (L28/α0.08 on gemma-3-4b, ≈0.82
        // depth, inside this grid) — both engines default to the same grid.
        #expect(ExperimentManifest.SweepSpec().alphas == [0.05, 0.08, 0.1, 0.13])
        #expect(ExperimentManifest.SweepSpec().layerFractions == [0.5, 0.7, 0.85])
    }

    // MARK: depth-fraction → layer-index resolution (sweep-time)

    @Test func defaultFractionsResolveAgainstModelDepth() {
        // Truncating, clamped, deduped, sorted — the server's
        // `resolve_sweep_layers` applies the identical rule.
        let spec = ExperimentManifest.SweepSpec()
        // gemma-3-4b depth: 34 blocks. 0.85·34 = 28.9 → L28, the live
        // optimum cell.
        #expect(spec.resolvedLayers(layerCount: 34) == [17, 23, 28])
        // A 40-block model.
        #expect(spec.resolvedLayers(layerCount: 40) == [20, 28, 34])
    }

    @Test func fractionResolutionClampsAndDedups() {
        var spec = ExperimentManifest.SweepSpec()
        // 1.0 would name layerCount — clamps to the last valid block; 0
        // stays the first block; near-duplicates collapse to one cell.
        spec.layerFractions = [0.0, 1.0, 0.5, 0.51]
        #expect(spec.resolvedLayers(layerCount: 10) == [0, 5, 9])
    }

    @Test func explicitGridOverridesDefaults() throws {
        // A manifest that declares its own grid is untouched by the default
        // recalibration: decode round-trips the explicit values.
        let json = """
            {"layerFractions": [0.35, 0.5, 0.65],
             "alphas": [0.04, 0.08, 0.12],
             "devPromptsFile": "prompts/dev/dev-prompts.jsonl",
             "batteryFile": "prompts/batteries/basic.jsonl",
             "maxTokens": 80}
            """
        let spec = try JSONDecoder().decode(
            ExperimentManifest.SweepSpec.self, from: Data(json.utf8))
        #expect(spec.layerFractions == [0.35, 0.5, 0.65])
        #expect(spec.alphas == [0.04, 0.08, 0.12])
        #expect(spec.resolvedLayers(layerCount: 34) == [11, 17, 22])
    }

    // MARK: sweep live-log previews (cross-engine line format)

    @Test func generationPreviewCollapsesWhitespaceAndTruncates() {
        #expect(ExperimentTasks.generationPreview("short answer") == "short answer")
        // Whitespace runs — newlines included — collapse to single spaces,
        // and leading/trailing whitespace drops (the server's
        // `" ".join(text.split())`).
        #expect(
            ExperimentTasks.generationPreview(" line one\n\nline two\r\n\tline three ")
                == "line one line two line three")
        // Exactly at the limit: untouched, no ellipsis.
        let exact = String(repeating: "b", count: 160)
        #expect(ExperimentTasks.generationPreview(exact) == exact)
        // Over the limit: first 160 characters plus a single ellipsis.
        let long = String(repeating: "a", count: 200)
        #expect(
            ExperimentTasks.generationPreview(long)
                == String(repeating: "a", count: 160) + "…")
        // A cut that lands on a space never leaves "a …" — trailing
        // whitespace is trimmed before the ellipsis (server: rstrip()).
        let spaced = String(repeating: "c", count: 159) + " tail"
        #expect(
            ExperimentTasks.generationPreview(spaced)
                == String(repeating: "c", count: 159) + "…")
    }

    @Test func sweepDevPreviewLineMatchesServerFormat() {
        // `<label> dev <i>/<n>: "<preview>"` — the server's `_dev_texts`
        // emits the identical shape (cell labels carry alpha in %g, so no
        // trailing zeros).
        #expect(
            ExperimentTasks.sweepCellLabel(layer: 12, alpha: 0.04) == "L12 α0.04")
        #expect(
            ExperimentTasks.sweepCellLabel(layer: 8, alpha: 0.5) == "L8 α0.5")
        #expect(
            ExperimentTasks.sweepDevPreviewLine(
                label: ExperimentTasks.sweepCellLabel(layer: 12, alpha: 0.04),
                index: 3, total: 8, text: "Bonjour\nle monde")
                == "L12 α0.04 dev 3/8: \"Bonjour le monde\"")
        #expect(
            ExperimentTasks.sweepDevPreviewLine(
                label: "baseline", index: 1, total: 2, text: "ok")
                == "baseline dev 1/2: \"ok\"")
    }

    @Test func structuralProblemsAreNamed() {
        var spec = ExperimentManifest.SweepSpec()
        spec.layerFractions = []
        #expect(SweepSpecForm.validate(spec) != nil)

        spec = ExperimentManifest.SweepSpec()
        spec.layerFractions = [1.5]
        #expect(SweepSpecForm.validate(spec)?.contains("[0, 1]") == true)

        spec = ExperimentManifest.SweepSpec()
        spec.alphas = []
        #expect(SweepSpecForm.validate(spec) != nil)

        spec = ExperimentManifest.SweepSpec()
        spec.alphas = [0]
        #expect(SweepSpecForm.validate(spec)?.contains("baseline") == true)

        spec = ExperimentManifest.SweepSpec()
        spec.maxTokens = 0
        #expect(SweepSpecForm.validate(spec) != nil)

        spec = ExperimentManifest.SweepSpec()
        spec.devPromptsFile = "  "
        #expect(SweepSpecForm.validate(spec) != nil)

        spec = ExperimentManifest.SweepSpec()
        spec.batteryFile = ""
        #expect(SweepSpecForm.validate(spec) != nil)
    }

    // MARK: save-time criterion validation

    @Test func absentAndImplementedSelectionsAreValid() {
        #expect(SweepSpecForm.validateSelection(nil) == .valid)
        // All three metrics are implemented now (2026-07-08) — the
        // declared-ahead caption is gone; instrument REQUIREMENTS are
        // validateObjectiveRequirements' job.
        for metric in ["markerDensity", "judgeScore", "logprobShift"] {
            #expect(
                SweepSpecForm.validateSelection(.init(objective: .init(metric: metric)))
                    == .valid)
        }
    }

    // MARK: save-time objective-requirement validation

    private func manifest(
        rubricPinned: Bool = false, judges: [ExperimentManifest.JudgeRef] = []
    ) -> ExperimentManifest {
        var manifest = ExperimentManifest(
            name: "m", description: "", modelID: "test/model")
        if rubricPinned {
            manifest.judgeRubricFile = "prompts/rubrics/r.md"
            manifest.judgeRubricHash = String(repeating: "a", count: 64)
        }
        manifest.judges = judges.isEmpty ? nil : judges
        return manifest
    }

    @Test func judgeScoreRefusesAtSaveWithoutManifestPins() {
        let selection = ExperimentManifest.SweepSelection(
            objective: .init(metric: "judgeScore"))
        let noRubric = SweepSpecForm.validateObjectiveRequirements(
            selection, manifest: manifest())
        #expect(noRubric?.contains("pinned judge rubric") == true)
        let noJudges = SweepSpecForm.validateObjectiveRequirements(
            selection, manifest: manifest(rubricPinned: true))
        #expect(noJudges?.contains("at least one judge") == true)
        let pinned = SweepSpecForm.validateObjectiveRequirements(
            selection,
            manifest: manifest(
                rubricPinned: true,
                judges: [.init(name: "j1", kind: "local", model: "org/judge")]))
        #expect(pinned == nil)
    }

    @Test func logprobShiftRefusesAtSaveWhenTheChoiceFileIsUnusable() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "optimizations-choices-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: root.appending(components: "prompts", "dev"),
            withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        // No file declared.
        let undeclared = SweepSpecForm.validateObjectiveRequirements(
            .init(objective: .init(metric: "logprobShift")),
            manifest: manifest(), root: root)
        #expect(undeclared?.contains("choicePromptsFile") == true)
        // Declared but missing on disk.
        let selection = ExperimentManifest.SweepSelection(
            objective: .init(
                metric: "logprobShift",
                choicePromptsFile: "prompts/dev/choices.jsonl"))
        let missing = SweepSpecForm.validateObjectiveRequirements(
            selection, manifest: manifest(), root: root)
        #expect(missing?.contains("not found") == true)
        // Unparseable rows (an option-less row).
        let url = root.appending(components: "prompts", "dev", "choices.jsonl")
        try #"{"id": "c1", "prompt": "p"}"#.write(
            to: url, atomically: true, encoding: .utf8)
        let optionless = SweepSpecForm.validateObjectiveRequirements(
            selection, manifest: manifest(), root: root)
        #expect(optionless?.contains("at least 2 options") == true)
        // A sound file saves.
        try #"{"id": "c1", "prompt": "p", "options": ["A", "B"]}"#.write(
            to: url, atomically: true, encoding: .utf8)
        let sound = SweepSpecForm.validateObjectiveRequirements(
            selection, manifest: manifest(), root: root)
        #expect(sound == nil)
    }

    @Test func unknownMetricRefusesAtDeclaration() {
        let outcome = SweepSpecForm.validateSelection(
            .init(objective: .init(metric: "vibes")))
        guard case .invalid(let reason) = outcome else {
            Issue.record("expected .invalid, got \(outcome)")
            return
        }
        #expect(reason.contains("vibes"))
    }

    @Test func rangeErrorsRefuseEvenWithDeclaredAheadObjective() {
        // The range check must not be masked by the declared-ahead metric:
        // a bad number is a bad declaration on any engine.
        let outcome = SweepSpecForm.validateSelection(
            .init(
                objective: .init(metric: "judgeScore"),
                constraints: .init(coherenceFloor: 2.0)))
        guard case .invalid(let reason) = outcome else {
            Issue.record("expected .invalid, got \(outcome)")
            return
        }
        #expect(reason.contains("coherenceFloor"))
    }

    @Test func rangeErrorsMatchSweepStartRule() {
        // Same numbers `SweepSelectionRule.resolve` refuses at sweep start
        // must refuse at declaration — declaration and execution agree.
        let badSpecs: [ExperimentManifest.SweepSelection] = [
            .init(constraints: .init(capabilityTolerance: -0.1)),
            .init(constraints: .init(coherenceFloor: 1.5)),
            .init(controls: .init(matchedNormRandomMargin: -1)),
        ]
        for spec in badSpecs {
            guard case .invalid = SweepSpecForm.validateSelection(spec) else {
                Issue.record("expected .invalid for \(spec)")
                continue
            }
        }
    }

    // MARK: lifecycle derivation

    @Test func lifecycleStagesDeriveFromEvidence() {
        let undeclared = OptimizationLifecycle.derive(
            hasSweepSpec: false, hasSweepRun: false,
            hasRecommendation: false, hasPromotedAgent: false)
        #expect(undeclared == .init(
            declared: false, swept: false, recommended: false, promoted: false))

        let swept = OptimizationLifecycle.derive(
            hasSweepSpec: true, hasSweepRun: true,
            hasRecommendation: false, hasPromotedAgent: false)
        #expect(swept.declared == true)
        #expect(swept.swept)
        #expect(!swept.recommended)

        // A stamped recommendation is proof a sweep ran, even when the run
        // directory has been pruned.
        let recommendedOnly = OptimizationLifecycle.derive(
            hasSweepSpec: true, hasSweepRun: false,
            hasRecommendation: true, hasPromotedAgent: false)
        #expect(recommendedOnly.swept)
        #expect(recommendedOnly.recommended)

        // Server substrate: spec and promotion state are not knowable —
        // nil, never a false negative.
        let server = OptimizationLifecycle.derive(
            hasSweepSpec: nil, hasSweepRun: true,
            hasRecommendation: true, hasPromotedAgent: nil)
        #expect(server.declared == nil)
        #expect(server.promoted == nil)
    }

    @Test func promotedAgentDetectionMatchesBirthCertificate() {
        func agent(_ name: String, experiment: String?) -> ModelVariantArtifact {
            var promotion: ModelVariantArtifact.Promotion?
            if let experiment {
                promotion = .init(
                    experiment: experiment, experimentHash: "h",
                    promotedAt: "2026-07-07T00:00:00Z", promotedBy: "criterion",
                    substrate: "swift-mlx", appVersion: "test")
            }
            return ModelVariantArtifact(
                name: name, baseModelID: "m", promptMode: "chatAssistant",
                qwenThinkingEnabled: false, temperature: 0, systemPrompt: "",
                promotion: promotion)
        }
        let artifacts = [
            agent("hand-made", experiment: nil),
            agent("scr-fear-agent", experiment: "scr"),
        ]
        #expect(OptimizationLifecycle.hasPromotedAgent(experiment: "scr", in: artifacts))
        #expect(!OptimizationLifecycle.hasPromotedAgent(experiment: "other", in: artifacts))
        #expect(!OptimizationLifecycle.hasPromotedAgent(experiment: "scr", in: []))
    }

    @Test func nextStepPointsAtTheOneNextAction() {
        func states(
            _ declared: Bool?, _ swept: Bool, _ recommended: Bool, _ promoted: Bool?
        ) -> OptimizationLifecycle.States {
            .init(declared: declared, swept: swept,
                  recommended: recommended, promoted: promoted)
        }
        #expect(OptimizationLifecycle.nextStep(states(false, false, false, false))
            .contains("declare"))
        #expect(OptimizationLifecycle.nextStep(states(true, false, false, false))
            .contains("run the declared sweep"))
        #expect(OptimizationLifecycle.nextStep(states(true, true, false, false))
            .contains("no recommendation"))
        #expect(OptimizationLifecycle.nextStep(states(true, true, true, false))
            .contains("create the agent"))
        #expect(OptimizationLifecycle.nextStep(states(true, true, true, true))
            .contains("confirmation study"))
        // Not-derivable promotion state (server listing) never claims a stage.
        #expect(OptimizationLifecycle.nextStep(states(nil, true, true, nil))
            .contains("not derivable"))
    }
}

/// Store-backed `setSweepSpec` tests — extend the serialized
/// `ExperimentStoreTests` suite because they use the process-global
/// workspace override (same pattern as the promote tests).
extension ExperimentStoreTests {

    private func withOptimizationsWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "optimizations-\(UUID().uuidString)")
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

    @Test @MainActor func setSweepSpecRoundTripsThroughTheStore() throws {
        try withOptimizationsWorkspace { _ in
            _ = try ExperimentStore.create(
                name: "scr", description: "", modelID: "test/model")
            let panel = ExperimentPanel()

            var spec = ExperimentManifest.SweepSpec(
                layerFractions: [0.4, 0.6],
                alphas: [0.2, 0.5],
                devPromptsFile: "prompts/dev/dev-prompts.jsonl",
                batteryFile: "prompts/batteries/basic.jsonl",
                maxTokens: 64)
            spec.selection = .init(
                objective: .init(metric: "markerDensity"),
                constraints: .init(capabilityTolerance: 0.2, coherenceFloor: 0.5),
                controls: .init(matchedNormRandomMargin: 0.05))
            #expect(panel.setSweepSpec(spec, for: "scr"))
            let loaded = try ExperimentStore.load(name: "scr")
            #expect(loaded.sweep == spec)
        }
    }

    @Test @MainActor func setSweepSpecValidatesCriterionAtSaveTime() throws {
        try withOptimizationsWorkspace { _ in
            _ = try ExperimentStore.create(
                name: "scr2", description: "", modelID: "test/model")
            let panel = ExperimentPanel()
            var spec = ExperimentManifest.SweepSpec()

            // Unknown metric: refused, nothing written.
            spec.selection = .init(objective: .init(metric: "vibes"))
            #expect(!panel.setSweepSpec(spec, for: "scr2"))
            #expect(try ExperimentStore.load(name: "scr2").sweep == nil)

            // Out-of-range constraint: refused even alongside a legal metric.
            spec.selection = .init(
                objective: .init(metric: "markerDensity"),
                constraints: .init(coherenceFloor: 5))
            #expect(!panel.setSweepSpec(spec, for: "scr2"))

            // Structurally broken grid: refused.
            spec = ExperimentManifest.SweepSpec()
            spec.alphas = []
            #expect(!panel.setSweepSpec(spec, for: "scr2"))

            // judgeScore without the manifest's rubric/judge pins: refused
            // at SAVE (the objective's config comes from MANIFEST pins).
            spec = ExperimentManifest.SweepSpec()
            spec.selection = .init(objective: .init(metric: "judgeScore"))
            #expect(!panel.setSweepSpec(spec, for: "scr2"))
            #expect(panel.status?.contains("pinned judge rubric") == true)
            #expect(try ExperimentStore.load(name: "scr2").sweep == nil)

            // With the pins in place the same declaration saves.
            var manifest = try ExperimentStore.load(name: "scr2")
            manifest.judgeRubricFile = "prompts/rubrics/r.md"
            manifest.judgeRubricHash = String(repeating: "a", count: 64)
            manifest.judges = [.init(name: "j1", kind: "local", model: "org/judge")]
            try ExperimentStore.save(manifest)
            #expect(panel.setSweepSpec(spec, for: "scr2"))
            #expect(
                try ExperimentStore.load(name: "scr2")
                    .sweep?.selection?.objective?.metric == "judgeScore")

            // logprobShift without a usable choice file: refused at SAVE;
            // with one, the declaration saves and keeps the file reference.
            spec = ExperimentManifest.SweepSpec()
            spec.selection = .init(objective: .init(metric: "logprobShift"))
            #expect(!panel.setSweepSpec(spec, for: "scr2"))
            #expect(panel.status?.contains("choicePromptsFile") == true)
            let choicesURL = VectorCatalog.projectRoot
                .appending(components: "prompts", "dev", "choices.jsonl")
            try FileManager.default.createDirectory(
                at: choicesURL.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try #"{"id": "c1", "prompt": "p", "options": ["A", "B"]}"#.write(
                to: choicesURL, atomically: true, encoding: .utf8)
            spec.selection = .init(
                objective: .init(
                    metric: "logprobShift",
                    choicePromptsFile: "prompts/dev/choices.jsonl"))
            #expect(panel.setSweepSpec(spec, for: "scr2"))
            let saved = try ExperimentStore.load(name: "scr2").sweep?.selection
            #expect(saved?.objective?.metric == "logprobShift")
            #expect(saved?.objective?.choicePromptsFile == "prompts/dev/choices.jsonl")
        }
    }

    @Test @MainActor func setSweepSpecRefusesNonDraftManifests() throws {
        try withOptimizationsWorkspace { _ in
            _ = try ExperimentStore.create(
                name: "scr3", description: "", modelID: "test/model")
            // Stamp the manifest frozen on disk (bypassing the store's own
            // transition rules — this is the state under test, not the path).
            var manifest = try ExperimentStore.load(name: "scr3")
            manifest.status = .frozen
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(manifest)
                .write(to: ExperimentStore.manifestURL("scr3"))

            let panel = ExperimentPanel()
            #expect(!panel.setSweepSpec(.init(), for: "scr3"))
            #expect(panel.status?.contains("frozen") == true)
            #expect(try ExperimentStore.load(name: "scr3").sweep == nil)

            // Nonexistent experiments refuse with an error, not a crash.
            #expect(!panel.setSweepSpec(.init(), for: "no-such-study"))
        }
    }

    /// Finding 11a: every `setSweepSpec` refusal must ALSO land in
    /// `formErrors[.sweepSpec]`, so the editor can render it beside the Save
    /// button. Routing refusals only through `note` put them in the
    /// panel-top notice feed, hundreds of points above the control — which
    /// is how a control margin was believed saved for days (2026-07-26).
    @Test @MainActor func sweepSpecRefusalsReachTheFormNotOnlyTheNoticeFeed() throws {
        try withOptimizationsWorkspace { _ in
            _ = try ExperimentStore.create(
                name: "form", description: "", modelID: "test/model")
            let panel = ExperimentPanel()

            // A structurally invalid spec (no alphas) refuses.
            var bad = ExperimentManifest.SweepSpec()
            bad.alphas = []
            #expect(!panel.setSweepSpec(bad, for: "form"))
            let inline = try #require(panel.formErrors[.sweepSpec])
            #expect(inline == panel.status)

            // A subsequent SUCCESS must clear it — a stale refusal under a
            // now-saved spec is its own paper cut.
            #expect(panel.setSweepSpec(.init(), for: "form"))
            #expect(panel.formErrors[.sweepSpec] == nil)
        }
    }

    /// The α = 0 refusal that read as "Add Condition does nothing".
    @Test @MainActor func addConditionRefusalsReachTheForm() throws {
        try withOptimizationsWorkspace { _ in
            _ = try ExperimentStore.create(
                name: "cond", description: "", modelID: "test/model")
            let panel = ExperimentPanel()
            panel.refresh()
            panel.selectedName = "cond"
            panel.conditionConcept = "fear"
            panel.conditionLayerText = "41"
            panel.conditionAlphaText = "0"
            panel.addVectorCondition()

            let inline = try #require(panel.formErrors[.addCondition])
            #expect(inline.contains("nonzero"))
            #expect(try ExperimentStore.load(name: "cond").conditions.isEmpty)
        }
    }
}
