import Foundation
import Testing

@testable import ExperimentKit

/// Pure-CPU tests for the server robustness path's "battery items → request
/// bodies" seam. The plan must mirror the local runner's arms exactly:
/// baseline battery → baseline coherence → variant battery → variant
/// coherence, greedy everywhere, the shared battery token cap, and
/// `stripInterventions` as the baseline arm (the same inline spec with its
/// interventions stripped server-side, mirroring the local empty-injections
/// + no-adapter baseline).
struct RobustnessServerPathTests {

    @Test func aFormatTwoBatteryPlansTwoWholeBatteryRequests() {
        // §23: a v2 battery is not N generations — it is ONE evaluation per
        // side, pinned by path + hash, with `stripInterventions` as the
        // baseline arm (the same arm flag the generate wire uses).
        let requests = VariantRobustness.serverBatteryRequests(
            batteryFile: "prompts/batteries/basic.jsonl", batteryHash: "abc123")
        #expect(requests.map(\.side) == ["Baseline", "Variant"])
        #expect(requests.map(\.stripInterventions) == [true, false])
        #expect(requests.allSatisfy { $0.batteryHash == "abc123" })
        #expect(
            requests.allSatisfy { $0.batteryFile == "prompts/batteries/basic.jsonl" })
    }

    @Test func armPlanMirrorsTheLocalRunnerOrder() {
        let requests = VariantRobustness.serverArmRequests(
            batteryPrompts: ["What is 2+2?", "Capital of France?"],
            coherencePrompts: ["Tell me about tea."],
            coherenceMaxTokens: 256)

        #expect(
            requests.map { "\($0.kind)|\($0.side)|\($0.index)" } == [
                "Capability|Baseline|1", "Capability|Baseline|2",
                "Coherence|Baseline|1",
                "Capability|Variant|1", "Capability|Variant|2",
                "Coherence|Variant|1",
            ])
    }

    @Test func baselineArmStripsInterventionsVariantArmKeepsThem() {
        let requests = VariantRobustness.serverArmRequests(
            batteryPrompts: ["What is 2+2?"],
            coherencePrompts: ["Tell me about tea."],
            coherenceMaxTokens: 256)
        for request in requests {
            #expect(request.stripInterventions == (request.side == "Baseline"))
        }
    }

    @Test func requestBodiesAreGreedyWithTheSharedTokenCaps() {
        // Greedy is a hard requirement for measured arms; battery items use
        // the SAME cap as the local runner (`batteryMaxTokens`), coherence
        // uses the preset's cap.
        let requests = VariantRobustness.serverArmRequests(
            batteryPrompts: ["What is 2+2?"],
            coherencePrompts: ["Tell me about tea."],
            coherenceMaxTokens: 320)
        for request in requests {
            #expect(request.temperature == 0)
            switch request.kind {
            case "Capability":
                #expect(request.maxTokens == VariantRobustness.batteryMaxTokens)
            case "Coherence":
                #expect(request.maxTokens == 320)
            default:
                Issue.record("unexpected kind \(request.kind)")
            }
        }
    }

    @Test func eachRequestCarriesASingleUserMessage() throws {
        let requests = VariantRobustness.serverArmRequests(
            batteryPrompts: ["What is 2+2?"],
            coherencePrompts: [],
            coherenceMaxTokens: 256)
        let request = try #require(requests.first)
        #expect(request.messages == [["role": "user", "content": "What is 2+2?"]])
        #expect(request.total == 1)
        #expect(request.prompt == "What is 2+2?")
    }

    @Test func emptyCoherenceListPlansOnlyBatteryArms() {
        let requests = VariantRobustness.serverArmRequests(
            batteryPrompts: ["a", "b"],
            coherencePrompts: [],
            coherenceMaxTokens: 256)
        #expect(requests.count == 4)
        #expect(requests.allSatisfy { $0.kind == "Capability" })
    }

    @Test func reportDecodesWithAndWithoutTheSubstrateStamp() throws {
        // Old reports (pre-stamp) must keep decoding; new ones carry the
        // engine stamp so the UI can label where generations ran.
        let legacy = """
            {"variantName": "v", "baseModelID": "m", "batteryFile": "b",
             "coherencePromptsFile": "c", "generatedAt": "2026-01-01",
             "baselineBatteryAccuracy": 1, "variantBatteryAccuracy": 1,
             "meanBaselineDistinct2": 0.5, "meanVariantDistinct2": 0.5,
             "meanBaselineWords": 10, "meanVariantWords": 10,
             "batteryItems": [], "coherenceItems": [], "warnings": []}
            """
        let decoded = try JSONDecoder().decode(
            VariantRobustnessReport.self, from: Data(legacy.utf8))
        #expect(decoded.substrate == nil)

        let stamped = legacy.replacingOccurrences(
            of: "\"variantName\": \"v\",",
            with: "\"variantName\": \"v\", \"substrate\": \"python-hf-transformers\",")
        let decodedStamped = try JSONDecoder().decode(
            VariantRobustnessReport.self, from: Data(stamped.utf8))
        #expect(decodedStamped.substrate == "python-hf-transformers")
    }
}

/// The server robustness path end to end, over stubbed wires (open issues
/// §23). Two behaviours are pinned against each other on the same code path:
/// a FORMAT-2 battery is scored by the dedicated battery wire, once per side,
/// under the battery's own arming; a FORMAT-1 battery keeps the generate wire
/// it always used, item by item, graded here by the historical text matcher.
///
/// Serialized, and every workspace-override window holds the shared
/// `ExperimentRootOverrideLock`: the workspace root is process-global, so a
/// parallel suite reading `VectorCatalog.projectRoot` must never observe it.
@Suite(.serialized) struct RobustnessServerBatteryWireTests {

    private static let v2Battery = """
        {"batteryFormat":2,"scoring":"choiceProbability","maxTokens":8,"promptMode":"chatAssistant"}
        {"id":"cap-fr","prompt":"What is the capital of France?","answer":"Paris","options":["Paris","Lyon","Nice"]}
        {"id":"sum","prompt":"What is 17 + 26?","answer":"43","options":["43","33","44"]}
        """

    private static let legacyBattery = """
        {"prompt":"What is the capital of France?","answer":"paris","grading":"token_exact"}
        {"prompt":"What is 17 + 26?","answer":"43","grading":"exact_number"}
        """

    private static let devPrompts = #"{"text":"Describe a walk through a quiet library."}"#

    private static let batteryFile = "prompts/batteries/probe.jsonl"
    private static let promptsFile = "prompts/dev/dev.jsonl"

    /// Records what each stubbed wire was asked for.
    actor Wires {
        private(set) var batteryRequests: [VariantRobustness.ServerBatteryRequest] = []
        private(set) var armRequests: [VariantRobustness.ServerArmRequest] = []
        func note(battery request: VariantRobustness.ServerBatteryRequest) {
            batteryRequests.append(request)
        }
        func note(arm request: VariantRobustness.ServerArmRequest) {
            armRequests.append(request)
        }
    }

    /// The variant under test carries a hostile instrument — the system
    /// prompt and prompt mode that must NOT reach a capability reading.
    private func variant() -> ModelVariantArtifact {
        ModelVariantArtifact(
            name: "fear-mix",
            baseModelID: "google/gemma-3-4b-it",
            promptMode: "rawCompletion",
            qwenThinkingEnabled: true,
            temperature: 0.7,
            systemPrompt: "You are a federal district judge. Respond in JSON.")
    }

    /// A temp workspace holding one battery file and one coherence prompt,
    /// installed as the process workspace for the duration of `body`.
    private func withWorkspace<T>(
        battery: String, _ body: (URL) async throws -> T
    ) async throws -> T {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "robust-wire-\(UUID().uuidString)")
            .resolvingSymlinksInPath().standardizedFileURL
        for (path, text) in [
            (Self.batteryFile, battery), (Self.promptsFile, Self.devPrompts),
        ] {
            let url = root.appending(path: path)
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try (text + "\n").write(to: url, atomically: true, encoding: .utf8)
        }
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = root
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: root)
        }
        return try await body(root)
    }

    /// One side's server response, in the wire's own shape.
    private func evaluation(
        side: String, correct: [Bool], hash: String
    ) -> ClusterClient.VariantBatteryEvaluation {
        let items:
            [(id: String, prompt: String, answer: String, options: [String])] = [
                ("cap-fr", "What is the capital of France?", "Paris",
                 ["Paris", "Lyon", "Nice"]),
                ("sum", "What is 17 + 26?", "43", ["43", "33", "44"]),
            ]
        let records = zip(items, correct).enumerated().map {
            index, pair -> ClusterClient.VariantBatteryEvaluation.Item in
            let (item, isCorrect) = pair
            let selected = isCorrect ? item.answer : item.options[1]
            return .init(
                promptIndex: index, promptID: item.id, prompt: item.prompt,
                answer: item.answer, scoring: "choiceProbability",
                options: item.options,
                choiceProbability: Dictionary(
                    uniqueKeysWithValues: item.options.map {
                        ($0, $0 == selected ? 0.8 : 0.1)
                    }),
                selected: selected, output: selected, correct: isCorrect)
        }
        return ClusterClient.VariantBatteryEvaluation(
            battery: Self.batteryFile, batteryHash: hash, batteryFormat: 2,
            stripInterventions: side == "Baseline", armingIsolated: true,
            armingPromptMode: "chatAssistant", armingSystemPrompt: false,
            armingMaxTokens: 8, items: records,
            summary: .init(
                accuracy: Double(correct.filter { $0 }.count) / Double(correct.count),
                itemCount: correct.count, batteryHash: hash))
    }

    @Test func aFormatTwoBatteryIsScoredByTheBatteryWireOnBothSides() async throws {
        try await withWorkspace(battery: Self.v2Battery) { root in
            let wires = Wires()
            let expectedHash = ExperimentStore.sha256Hex(
                try Data(contentsOf: root.appending(path: Self.batteryFile)))
            let result = try await VariantRobustness.runViaServer(
                variant: variant(), variantPath: nil, variantHash: nil,
                batteryFile: Self.batteryFile,
                coherencePromptsFile: Self.promptsFile,
                maxCoherencePrompts: 1,
                generate: { request in
                    await wires.note(arm: request)
                    return "A quiet library, then the rain outside."
                },
                scoreBattery: { request in
                    await wires.note(battery: request)
                    // Baseline gets both right; the variant loses one — the
                    // capability cost this check exists to see.
                    return self.evaluation(
                        side: request.side,
                        correct: request.side == "Baseline"
                            ? [true, true] : [true, false],
                        hash: request.batteryHash)
                })
            let report = result.report

            // Both sides went through the battery wire, pinned, baseline
            // first — the local runner's arm order.
            let batteryRequests = await wires.batteryRequests
            #expect(batteryRequests.map(\.side) == ["Baseline", "Variant"])
            #expect(batteryRequests.map(\.stripInterventions) == [true, false])
            #expect(batteryRequests.allSatisfy { $0.batteryHash == expectedHash })
            #expect(batteryRequests.allSatisfy { $0.batteryFile == Self.batteryFile })
            // …and NOTHING was generated for a capability item.
            let armRequests = await wires.armRequests
            #expect(armRequests.allSatisfy { $0.kind == "Coherence" })
            #expect(armRequests.map(\.side) == ["Baseline", "Variant"])

            // The server's readings ARE the report's readings — a choice
            // item's 0/1 is never re-graded here (there is no text to grade).
            #expect(report.baselineBatteryAccuracy == 1.0)
            #expect(report.variantBatteryAccuracy == 0.5)
            #expect(report.batteryItems.map(\.baselineResponse) == ["Paris", "43"])
            #expect(report.batteryItems.map(\.variantCorrect) == [true, false])
            #expect(report.substrate == WorkspaceScoping.serverSubstrate)
            // Stamped like a battery.jsonl row, so the report says on its
            // face that the two accuracies are instrument-comparable.
            #expect(report.batteryFormat == 2)
            #expect(report.armingIsolated == true)
            #expect(
                report.warnings.contains { $0.contains("capability battery dropped") })
        }
    }

    @Test func aFormatOneBatteryStillRunsOnTheGenerateWire() async throws {
        try await withWorkspace(battery: Self.legacyBattery) { _ in
            let wires = Wires()
            let result = try await VariantRobustness.runViaServer(
                variant: variant(), variantPath: nil, variantHash: nil,
                batteryFile: Self.batteryFile,
                coherencePromptsFile: Self.promptsFile,
                maxCoherencePrompts: 1,
                generate: { request in
                    await wires.note(arm: request)
                    return request.prompt.contains("capital") ? "Paris" : "43"
                },
                scoreBattery: { request in
                    await wires.note(battery: request)
                    Issue.record("a legacy battery must not touch the battery wire")
                    throw ExperimentError(reason: "unreachable")
                })

            // The legacy path is untouched: every item is a generation,
            // graded here by the historical text matcher, and the battery
            // wire — which the server refuses format 1 on — is never called.
            #expect(await wires.batteryRequests.isEmpty)
            let armRequests = await wires.armRequests
            #expect(
                armRequests.map { "\($0.kind)|\($0.side)" } == [
                    "Capability|Baseline", "Capability|Baseline", "Coherence|Baseline",
                    "Capability|Variant", "Capability|Variant", "Coherence|Variant",
                ])
            #expect(
                armRequests.filter { $0.kind == "Capability" }
                    .allSatisfy { $0.maxTokens == VariantRobustness.batteryMaxTokens })
            #expect(result.report.baselineBatteryAccuracy == 1.0)
            #expect(result.report.variantBatteryAccuracy == 1.0)
            // No format-2 stamps on a legacy reading — absent, never null,
            // the same rule battery.jsonl rows follow.
            #expect(result.report.batteryFormat == nil)
            #expect(result.report.armingIsolated == nil)
            // The one deliberate change to the legacy path: this reading IS
            // taken under the variant's system prompt, and the server path
            // now says so out loud exactly as the local runner already did.
            #expect(
                result.report.warnings.contains { $0.contains("(legacy)") })
        }
    }

    @Test func serverRecordsThatDoNotLineUpWithTheLocalBatteryRefuse() throws {
        let battery = try CapabilityBattery(
            data: Data((Self.v2Battery + "\n").utf8), file: "probe.jsonl")
        // One record short: the pinned hash makes this nearly impossible, and
        // a shorter list of readings must never be quietly averaged.
        let short = evaluation(side: "Variant", correct: [true], hash: "abc")
        #expect(throws: ExperimentError.self) {
            try VariantRobustness.batteryScores(battery, evaluation: short)
        }
        // Right count, wrong items.
        let mismatched = ClusterClient.VariantBatteryEvaluation(
            battery: "probe.jsonl", batteryHash: "abc", batteryFormat: 2,
            stripInterventions: false, armingIsolated: true,
            armingPromptMode: "chatAssistant", armingSystemPrompt: false,
            armingMaxTokens: 8,
            items: (0..<2).map {
                .init(
                    promptIndex: $0, promptID: "x", prompt: "a different item",
                    answer: "Paris", scoring: "choiceProbability",
                    options: ["Paris", "Lyon"], choiceProbability: ["Paris": 1],
                    selected: "Paris", output: "Paris", correct: true)
            },
            summary: .init(accuracy: 1, itemCount: 2, batteryHash: "abc"))
        #expect(throws: ExperimentError.self) {
            try VariantRobustness.batteryScores(battery, evaluation: mismatched)
        }
    }
}

/// View-model tests for the Robustness Check's target selection: local
/// records and SERVER-STORED agents share one selection type (the picker's
/// server group is the same source the agents list shows), and the legacy
/// local-ID accessor stays source-compatible for existing call sites.
@MainActor
struct RobustnessTargetSelectionTests {

    @Test func legacyLocalAccessorRoundTripsThroughTheUnifiedTarget() {
        let panel = FineTuningPanel()
        panel.robustnessTargetVariantID = "runs/model-variants/fear.json"
        #expect(panel.robustnessTarget == .local("runs/model-variants/fear.json"))
        #expect(panel.robustnessTargetVariantID == "runs/model-variants/fear.json")

        // A server-stored target is NOT a local record: the legacy accessor
        // reads nil, and the display name falls back to the path basename
        // until the server listing resolves it.
        panel.robustnessTarget = .server(path: "/srv/ws/runs/model-variants/awe.json")
        #expect(panel.robustnessTargetVariantID == nil)
        #expect(panel.robustnessTargetVariant == nil)
        #expect(panel.robustnessTargetName == "awe.json")

        panel.robustnessTargetVariantID = nil
        #expect(panel.robustnessTarget == nil)
        #expect(panel.robustnessTargetName == nil)
    }

    @Test func unresolvableTargetsNeverEnableTheRunButton() {
        let panel = FineTuningPanel()
        #expect(!panel.hasResolvableRobustnessTarget)
        // Server-stored target with no connected host/listing: not runnable.
        panel.robustnessTarget = .server(path: "/srv/x.json")
        #expect(!panel.hasResolvableRobustnessTarget)
        // Local target whose record is not in the scanned library: not runnable.
        panel.robustnessTarget = .local("no-such-record")
        #expect(!panel.hasResolvableRobustnessTarget)
    }
}
