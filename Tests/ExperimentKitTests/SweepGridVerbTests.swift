import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// `experiment set-sweep-grid` — the sweep block's GRID half, and the writer
/// the authoring surface shipped without.
///
/// The gap this closes is where the passenger-concept problem lived. The
/// sweep's SELECTION rule had a headless writer (`set-sweep-selection`); the
/// layer × α grid it selects over had none outside the Optimizations panel. So
/// the only headless way to obtain a grid was to `duplicate` a study that
/// already had one — which carries the donor's CONCEPTS along with its sweep
/// block, and a concept that rides in that way is swept but cannot be cited.
///
/// The half that is not about convenience is the AXIS AUDIT. A grid whose
/// written form and run form disagree is the quiet-loss class this engine
/// refuses on principle: `resolvedLayers` sorts and deduplicates, so an
/// unordered or repeated declaration names cells the sweep will not run, and a
/// repeated α is a cell paid for twice and reported once.
///
/// Serialized and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct SweepGridVerbTests {

    // MARK: Harness

    /// A WORKSPACE override, not just an experiment-root override: the depth
    /// this verb resolves absolute layers against comes from the vector
    /// CATALOG, which scans through `WorkspaceRoot`. Overriding only the
    /// narrower seam would read the checkout's committed runs.
    func withTempRoot<T>(_ body: (URL) async throws -> T) async rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "sweep-grid-verb-\(UUID().uuidString)")
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
    func draft(_ name: String = "d", model: String = "test/model") throws
        -> ExperimentManifest
    {
        try ExperimentStore.create(
            name: name, description: "", modelID: model)
    }

    /// A minimal vector sidecar + tensor pair, which is the ONLY thing in a
    /// workspace that states how deep a model is. `VectorCatalog` needs both
    /// files, so writing the sidecar alone would leave the depth unknown and
    /// quietly change what these tests exercise.
    func plantSidecar(
        root: URL, model: String = "test/model", layerCount: Int = 34,
        concept: String = "alpha"
    ) throws {
        let run = root.appending(components: "runs", "20260101-000000-x")
        try FileManager.default.createDirectory(
            at: run, withIntermediateDirectories: true)
        // Every key `SteeringVectorSidecar` requires: a partial sidecar
        // fails to decode and the catalog skips the pair silently, which
        // would leave these tests exercising the unknown-depth path.
        let sidecar: [String: Any] = [
            "modelID": model, "layerCount": layerCount, "hiddenSize": 8,
            "concept": concept, "stimulusSetHash": String(repeating: "0", count: 64),
            "normsPerLayer": Array(repeating: 1.0, count: layerCount),
            "extractionDate": "2026-01-01T00:00:00Z",
        ]
        try JSONSerialization
            .data(withJSONObject: sidecar, options: [.sortedKeys])
            .write(to: run.appending(component: "\(concept).json"))
        try Data([0]).write(
            to: run.appending(component: "\(concept).safetensors"))
    }

    func lifecycleGate(_ error: any Error) -> LifecycleGate? {
        (error as? ExperimentError)?.lifecycleRefusal?.gate
    }

    // MARK: - The block it edits

    /// Setting ONE axis on a study that has never had a sweep block must not
    /// invent the others: the block is born at the documented defaults and the
    /// named axis moves. Anything else would make `--alphas` alone a silent
    /// declaration of a layer grid nobody chose.
    @Test func aDraftWithNoSweepBlockStartsFromTheEngineDefaults() async throws {
        try await withTempRoot { _ in
            try draft()
            let outcome = try ExperimentStore.setSweepGrid(
                experimentName: "d", alphas: [0.2, 0.4])
            let spec = try #require(outcome.manifest.sweep)
            #expect(spec.alphas == [0.2, 0.4])
            #expect(spec.layerFractions == [0.5, 0.7, 0.85])
            #expect(spec.devPromptsFile == "prompts/dev/dev-prompts.jsonl")
            #expect(spec.batteryFile == "prompts/batteries/basic.jsonl")
            #expect(spec.maxTokens == 80)
        }
    }

    @Test func eachArgumentEditsItsOwnFieldAndLeavesTheRest() async throws {
        try await withTempRoot { _ in
            try draft()
            _ = try ExperimentStore.setSweepGrid(
                experimentName: "d", layerFractions: [0.3, 0.6],
                alphas: [0.1], maxTokens: 120)
            let outcome = try ExperimentStore.setSweepGrid(
                experimentName: "d", alphas: [0.05, 0.1])
            let spec = try #require(outcome.manifest.sweep)
            #expect(spec.layerFractions == [0.3, 0.6])
            #expect(spec.maxTokens == 120)
            #expect(spec.alphas == [0.05, 0.1])
        }
    }

    /// The two verbs split one block. A grid edit that dropped the criterion
    /// would silently un-preregister the study.
    @Test func theSelectionCriterionIsUntouched() async throws {
        try await withTempRoot { _ in
            try draft()
            _ = try ExperimentStore.setSweepSelection(
                .init(objective: .init(metric: "markerDensity")),
                experimentName: "d")
            let before = try ExperimentStore.load(name: "d").sweep?.selection
            let outcome = try ExperimentStore.setSweepGrid(
                experimentName: "d", alphas: [0.05])
            #expect(outcome.manifest.sweep?.selection == before)
        }
    }

    /// A pin certifies BYTES. Kept over a path that just moved, it is a claim
    /// about a file nobody read — and `verify` would then compare the new file
    /// against the old file's hash and call it drift.
    @Test func repointingAnInstrumentClearsItsFreezePin() async throws {
        try await withTempRoot { _ in
            try draft()
            _ = try ExperimentStore.updateDraft(name: "d") { manifest in
                manifest.sweep = .init(
                    devPromptsFile: "prompts/dev/a.jsonl",
                    batteryFile: "prompts/batteries/b.jsonl",
                    devPromptsHash: String(repeating: "a", count: 64),
                    batteryHash: String(repeating: "b", count: 64))
            }
            let outcome = try ExperimentStore.setSweepGrid(
                experimentName: "d", devPromptsFile: "prompts/dev/c.jsonl")
            #expect(outcome.manifest.sweep?.devPromptsHash == nil)
            // The battery was NOT re-pointed, so its pin stands.
            #expect(
                outcome.manifest.sweep?.batteryHash
                    == String(repeating: "b", count: 64))
        }
    }

    // MARK: - The axis audit

    /// One malformed declaration and the sentence it must answer with. A
    /// named type rather than a tuple literal: the type checker times out on
    /// a nine-element heterogeneous literal inside the `@Test` macro.
    struct BadGrid: Sendable {
        let fractions: [Double]
        let alphas: [Double]
        let fragment: String
    }

    static let badGrids: [BadGrid] = [
        .init(fractions: [], alphas: [0.05],
              fragment: "the layer axis is empty"),
        .init(fractions: [1.5], alphas: [0.05],
              fragment: "layer fractions are depths in [0, 1]"),
        .init(fractions: [0.7, 0.5], alphas: [0.05],
              fragment: "the layer axis does not ascend at 0.5"),
        .init(fractions: [0.5, 0.5], alphas: [0.05],
              fragment: "the layer axis does not ascend at 0.5"),
        .init(fractions: [0.5], alphas: [],
              fragment: "the alpha axis is empty"),
        .init(fractions: [0.5], alphas: [0],
              fragment: "alphas are residual-norm units above 0"),
        .init(fractions: [0.5], alphas: [-0.1],
              fragment: "alphas are residual-norm units above 0"),
        .init(fractions: [0.5], alphas: [0.1, 0.05],
              fragment: "the alpha ladder does not ascend at 0.05"),
        .init(fractions: [0.5], alphas: [0.1, 0.1],
              fragment: "the alpha ladder does not ascend at 0.1"),
    ]

    @Test(arguments: SweepGridVerbTests.badGrids)
    func aGridNoEngineCouldSweepIsRefused(_ testCase: BadGrid) async throws {
        let fractions = testCase.fractions
        let alphas = testCase.alphas
        let fragment = testCase.fragment
        try await withTempRoot { _ in
            try draft()
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setSweepGrid(
                    experimentName: "d", layerFractions: fractions,
                    alphas: alphas)
            }
            do {
                _ = try ExperimentStore.setSweepGrid(
                    experimentName: "d", layerFractions: fractions,
                    alphas: alphas)
            } catch {
                #expect(lifecycleGate(error) == .sweepGridRule)
                #expect("\(error)".contains(fragment))
                #expect(
                    (error as? ExperimentError)?.lifecycleRefusal?.repairAction
                        == ExperimentStore.sweepGridRepair("d"))
            }
        }
    }

    @Test func aRefusedGridWritesNothing() async throws {
        try await withTempRoot { _ in
            try draft()
            _ = try ExperimentStore.setSweepGrid(
                experimentName: "d", layerFractions: [0.4], alphas: [0.05])
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setSweepGrid(
                    experimentName: "d", layerFractions: [0.9, 0.2],
                    alphas: [0.07])
            }
            let spec = try #require(ExperimentStore.load(name: "d").sweep)
            #expect(spec.layerFractions == [0.4])
            #expect(spec.alphas == [0.05])
        }
    }

    @Test func aFrozenStudyRefusesWithTheImmutabilityGate() async throws {
        try await withTempRoot { _ in
            try draft()
            var manifest = try ExperimentStore.load(name: "d")
            manifest.status = .frozen
            try ExperimentStore.save(manifest)
            do {
                _ = try ExperimentStore.setSweepGrid(
                    experimentName: "d", alphas: [0.05])
                Issue.record("a frozen study accepted a grid")
            } catch {
                #expect(lifecycleGate(error) == .statusImmutable)
            }
        }
    }

    // MARK: - Absolute layers, and the depth they are meaningless without

    /// Not clamped, not guessed at a default depth: a layer index against the
    /// wrong depth names a different cell, and a sweep that ran there would
    /// report a study nobody declared.
    @Test func absoluteLayersWithoutAKnownDepthAreRefused() async throws {
        try await withTempRoot { _ in
            try draft()
            do {
                _ = try ExperimentStore.setSweepGrid(
                    experimentName: "d", absoluteLayers: [13, 18])
                Issue.record("absolute layers resolved against no depth")
            } catch {
                #expect(lifecycleGate(error) == .missingPrerequisite)
                #expect(
                    "\(error)".contains(
                        "nothing in this workspace states how deep "
                            + "'test/model' is"))
                #expect(
                    (error as? ExperimentError)?.lifecycleRefusal?.repairAction
                        == ExperimentStore.absoluteLayersNeedDepthRepair("d"))
            }
            #expect(try ExperimentStore.load(name: "d").sweep == nil)
        }
    }

    @Test func absoluteLayersConvertAgainstTheCatalogsDepth() async throws {
        try await withTempRoot { root in
            try draft()
            try plantSidecar(root: root, layerCount: 34)
            let outcome = try ExperimentStore.setSweepGrid(
                experimentName: "d", absoluteLayers: [13, 18, 28])
            #expect(outcome.layerCount == 34)
            #expect(outcome.resolvedLayers == [13, 18, 28])
            #expect(outcome.declaredAbsoluteLayers)
            #expect(
                outcome.manifest.sweep?.resolvedLayers(layerCount: 34)
                    == [13, 18, 28])
        }
    }

    /// The manifest gains NO second spelling of the axis: an axis with two
    /// stored spellings is an axis that can disagree with itself, and the
    /// depth it resolved against is a property of the pinned model, which the
    /// manifest already names.
    @Test func theManifestStoresFractionsOnly() async throws {
        try await withTempRoot { root in
            try draft()
            try plantSidecar(root: root, layerCount: 34)
            _ = try ExperimentStore.setSweepGrid(
                experimentName: "d", absoluteLayers: [13, 28])
            let bytes = try Data(
                contentsOf: ExperimentStore.manifestURL("d"))
            let document = try #require(
                try JSONSerialization.jsonObject(with: bytes) as? [String: Any])
            let sweep = try #require(document["sweep"] as? [String: Any])
            #expect(
                Set(sweep.keys) == [
                    "layerFractions", "alphas", "devPromptsFile",
                    "batteryFile", "maxTokens",
                ])
        }
    }

    @Test func aLayerOutsideTheModelIsRefused() async throws {
        try await withTempRoot { root in
            try draft()
            try plantSidecar(root: root, layerCount: 34)
            do {
                _ = try ExperimentStore.setSweepGrid(
                    experimentName: "d", absoluteLayers: [13, 34])
                Issue.record("a layer outside the model was accepted")
            } catch {
                #expect(lifecycleGate(error) == .sweepGridRule)
                #expect(
                    "\(error)" == "layer 34 is outside 'test/model', which has "
                        + "34 block(s) — legal layers are 0…33")
                #expect(
                    (error as? ExperimentError)?.lifecycleRefusal?.repairAction
                        == ExperimentStore.absoluteLayersOutOfRangeRepair(
                            "d", depth: 34))
            }
        }
    }

    /// The same ascent rule fires on the fractions a moment later, but it
    /// would say "does not ascend at 0.42" about a declaration that named 13.
    @Test func unorderedAbsoluteLayersRefuseInTheirOwnVocabulary() async throws {
        try await withTempRoot { root in
            try draft()
            try plantSidecar(root: root, layerCount: 34)
            do {
                _ = try ExperimentStore.setSweepGrid(
                    experimentName: "d", absoluteLayers: [18, 13])
                Issue.record("an unordered layer axis was accepted")
            } catch {
                #expect(lifecycleGate(error) == .sweepGridRule)
                #expect(
                    "\(error)".hasPrefix(
                        "the layer axis does not ascend at 13"))
            }
        }
    }

    /// Not a refusal — the fractions are legal and the collapse is a property
    /// of THIS model's depth — but a grid of "three depths" that is really two
    /// is a silently smaller sweep.
    @Test func twoFractionsMayCollapseOntoOneLayerAndItIsReported() async throws {
        try await withTempRoot { root in
            try draft()
            try plantSidecar(root: root, layerCount: 26)
            let outcome = try ExperimentStore.setSweepGrid(
                experimentName: "d", layerFractions: [0.50, 0.51, 0.85])
            #expect(outcome.resolvedLayers == [13, 22])
            #expect(outcome.collapsedFractions == 1)
        }
    }

    /// A caller must be able to tell "no layers" from "not resolvable yet".
    @Test func anUnknownDepthIsReportedAsUnknownNeverAsZero() async throws {
        try await withTempRoot { _ in
            try draft()
            let outcome = try ExperimentStore.setSweepGrid(
                experimentName: "d", alphas: [0.05])
            #expect(outcome.layerCount == nil)
            #expect(outcome.resolvedLayers.isEmpty)
        }
    }

    @Test func theDepthComesFromThePinnedModelNotFromAnyVector() async throws {
        try await withTempRoot { root in
            try draft(model: "test/model")
            try plantSidecar(root: root, model: "other/model", layerCount: 62)
            do {
                _ = try ExperimentStore.setSweepGrid(
                    experimentName: "d", absoluteLayers: [13])
                Issue.record("a foreign model's depth was used")
            } catch {
                #expect(lifecycleGate(error) == .missingPrerequisite)
            }
        }
    }

    // MARK: - The verb, through the runner

    func invoke(_ args: [String]) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(sink: .discarding).run(
            namespace: "experiment", args)
    }

    @Test func theVerbEchoesBothFormsOfTheAxis() async throws {
        try await withTempRoot { root in
            try draft()
            try plantSidecar(root: root, layerCount: 34)
            let outcome = await invoke(
                ["set-sweep-grid", "d", "--layers", "13,18,28",
                 "--alphas", "0.05,0.1"])
            #expect(outcome.envelope.state == .ready)
            let result = try #require(outcome.envelope.result)
            #expect(
                result["resolvedLayers"]
                    == .array([.number(13), .number(18), .number(28)]))
            #expect(result["layerCount"] == .number(34))
            #expect(result["cellCount"] == .number(6))
            #expect(result["alphaUnits"] == .string("residualNorm"))
            #expect(result["declaredAbsoluteLayers"] == .bool(true))
        }
    }

    @Test func declaringBothSpellingsOfTheAxisIsAMalformedInvocation() async throws {
        try await withTempRoot { _ in
            try draft()
            let outcome = await invoke(
                ["set-sweep-grid", "d", "--layers", "13",
                 "--layer-fractions", "0.5"])
            #expect(outcome.envelope.state == .blocked)
            #expect(outcome.envelope.error?.code == "usage")
            #expect(
                outcome.envelope.error?.reason.contains(
                    "two spellings of ONE axis") == true)
        }
    }

    @Test func aCallThatWouldWriteNothingIsRefused() async throws {
        try await withTempRoot { _ in
            try draft()
            let outcome = await invoke(["set-sweep-grid", "d"])
            #expect(outcome.envelope.state == .blocked)
            #expect(
                outcome.envelope.error?.reason.contains(
                    "at least one axis or input") == true)
        }
    }

    /// Dropping it would report a four-cell sweep as the five-cell one that
    /// was asked for.
    @Test func anUnparseableAxisEntryRefusesRatherThanShrinkingTheGrid() async throws {
        try await withTempRoot { _ in
            try draft()
            let outcome = await invoke(
                ["set-sweep-grid", "d", "--alphas", "0.05,x,0.1"])
            #expect(outcome.envelope.state == .blocked)
            #expect(outcome.envelope.error?.reason.contains("'x'") == true)
            #expect(try ExperimentStore.load(name: "d").sweep == nil)
        }
    }

    // MARK: - The split with `set-sweep-selection`

    /// The redirect table is only useful while it names the flags
    /// `set-sweep-selection` actually takes — otherwise the pointer sends a
    /// caller to a verb that would refuse the same flag. Server twin:
    /// `test_the_selection_owned_flags_are_the_ones_that_verb_declares`.
    @Test func theRedirectedFlagsAreExactlyTheOtherVerbs() throws {
        let grid = try #require(
            ExperimentCLIParser.specs.first {
                $0.label == "experiment set-sweep-grid"
            })
        let selection = try #require(
            ExperimentCLIParser.specs.first {
                $0.label == "experiment set-sweep-selection"
            })
        #expect(
            Set(grid.redirectedFlags.keys) == selection.valueFlags,
            "set-sweep-grid redirects flags set-sweep-selection does not take")
        #expect(
            grid.redirectedFlags.values.allSatisfy {
                $0 == "experiment set-sweep-selection"
            })
        // And the two verbs own DISJOINT halves of one block: a flag both
        // accepted would be a field with two writers.
        #expect(grid.valueFlags.isDisjoint(with: selection.valueFlags))
    }

    /// A selection flag typed here is not a typo — it is a correct intent
    /// aimed one verb over — so the answer is the owner, not a flag list.
    @Test func aSelectionFlagIsAnsweredWithAPointerToItsOwner() throws {
        #expect(throws: ExperimentCLIUsageError.self) {
            try ExperimentCLIParser.parse(
                namespace: "experiment",
                ["set-sweep-grid", "d", "--objective", "judgeScore"])
        }
        do {
            _ = try ExperimentCLIParser.parse(
                namespace: "experiment",
                ["set-sweep-grid", "d", "--objective", "judgeScore"])
        } catch let error as ExperimentCLIUsageError {
            #expect(error.ownedBy == "experiment set-sweep-selection")
            #expect(
                error.reason
                    == "--objective is experiment set-sweep-selection's flag, "
                    + "not experiment set-sweep-grid's")
            #expect(
                error.repairAction.hasPrefix(
                    "steerlab-cli experiment set-sweep-selection <name> "
                        + "--objective <value>"))
        }
    }

    /// An ordinary typo keeps the ordinary answer: the pointer must not
    /// swallow the flag list every other refusal on this surface gives.
    @Test func anOrdinaryTypoKeepsTheOrdinaryRefusal() throws {
        do {
            _ = try ExperimentCLIParser.parse(
                namespace: "experiment",
                ["set-sweep-grid", "d", "--alfas", "0.05"])
            Issue.record("an undeclared flag parsed")
        } catch let error as ExperimentCLIUsageError {
            #expect(error.ownedBy == nil)
            #expect(
                error.reason
                    == "experiment set-sweep-grid does not accept --alfas")
            #expect(error.repairAction.hasPrefix("experiment set-sweep-grid accepts:"))
        }
    }
}
