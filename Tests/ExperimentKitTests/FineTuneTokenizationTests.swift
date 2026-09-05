import Foundation
import MLXLMCommon
import SteeringKit
import Testing
import Tokenizers

@testable import ExperimentKit

/// Instruction/chat TRAINING TARGETS, rendered by real chat templates.
///
/// The defect these pin (external review, 2026-09-05 — SCI-03): the trainer
/// rendered both ends of each example — prompt-only and completed — through
/// `MLXLMCommon.Tokenizer.applyChatTemplate(messages:tools:additionalContext:)`,
/// whose swift-transformers implementation hard-codes
/// `addGenerationPrompt: true` (1.3.x, `Sources/Tokenizers/Tokenizer.swift`).
/// The COMPLETED example therefore ended
/// `…<answer><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
/// on Qwen3 with thinking disabled, and the loss mask — every position from
/// the first assistant target on — supervised that trailing header, in
/// training and in validation alike.
///
/// These tests execute the REAL Jinja templates OFFLINE. `Tests/Fixtures/
/// chat-templates/<family>/` is a minimal swift-transformers model folder:
/// the verbatim `chat_template.jinja` pinned from a locally cached snapshot
/// (see `provenance.json` for source model, revision and date) beside a
/// synthetic byte-level BPE tokenizer — 256 single-byte tokens, no merges,
/// turn markers as added special tokens. So the template runs for real, the
/// ids never move, and nothing touches the network. `cachedSnapshotAgrees…`
/// then re-runs the same claims against the actual cached tokenizers (real
/// BPE, where a token-level prefix is a genuine question), skipping loudly
/// when a model is not downloaded.
@Suite struct FineTuneTokenizationTests {

    // MARK: - Fixtures

    private static var repoRoot: URL {
        URL(filePath: #filePath)  // …/Tests/ExperimentKitTests/<this file>
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static var templateDirectory: URL {
        repoRoot.appending(components: "Tests", "Fixtures", "chat-templates")
    }

    /// The pinned-template model folder for a family, loaded through the same
    /// entry point production uses (`AutoTokenizer.from(modelFolder:)`).
    private static func pinnedRenderer(
        _ family: String
    ) async throws -> CachedSnapshotInstructionRenderer {
        CachedSnapshotInstructionRenderer(
            tokenizer: try await AutoTokenizer.from(
                modelFolder: templateDirectory.appending(component: family)))
    }

    /// Local HF cache snapshot for a model id, or nil when not downloaded
    /// (same resolution as `SteeredContainerLoader.cachedSnapshotDirectory`).
    private static func cachedSnapshot(for modelID: String) -> URL? {
        SteeredContainerLoader.cachedSnapshotDirectory(for: modelID)
    }

    /// Decodes the supervised targets and the unsupervised prefix of a
    /// tokenized example. `weights[i]` scores the target `tokens[i + 1]`, so
    /// the first supervised index `i` means everything from `tokens[i + 1]` on
    /// is trained and `tokens[0...i]` is context.
    private static func split(
        _ example: FineTuneTrainer.TokenizedInstructionExample,
        with tokenizer: any Tokenizers.Tokenizer
    ) -> (context: String, supervised: String, contiguous: Bool) {
        guard let first = example.weights.firstIndex(of: 1) else {
            return ("", "", false)
        }
        let contiguous =
            example.weights[..<first].allSatisfy { $0 == 0 }
            && example.weights[first...].allSatisfy { $0 == 1 }
        return (
            tokenizer.decode(
                tokens: Array(example.tokens[...first]), skipSpecialTokens: false),
            tokenizer.decode(
                tokens: Array(example.tokens[(first + 1)...]), skipSpecialTokens: false),
            contiguous
        )
    }

    private static let terseExample = FineTuneTrainer.InstructionExample(
        system: "You are terse.", user: "Ping?", assistant: "Pong.")

    // MARK: - Qwen3: the render that carried the defect

    /// THE regression. The completed example ends at the answer's own
    /// end-of-turn — no second assistant header, no empty thinking block
    /// after the answer.
    @Test func qwen3CompletedExampleEndsAtTheAnswersEndOfTurn() async throws {
        let renderer = try await Self.pinnedRenderer("qwen3")
        let examples = try FineTuneTrainer.tokenizeInstructionExamples(
            [Self.terseExample], renderer: renderer, modelID: "Qwen/Qwen3-0.6B")
        #expect(examples.count == 1)
        let rendered = renderer.tokenizer.decode(
            tokens: examples[0].tokens, skipSpecialTokens: false)
        #expect(
            rendered.hasSuffix("Pong.<|im_end|>\n"),
            "completed example must end at the answer's end-of-turn: \(rendered)")
        #expect(
            !rendered.hasSuffix("</think>\n\n"),
            "a trailing generation prompt is the SCI-03 defect: \(rendered)")
        // The assistant header appears exactly once — the one the answer
        // belongs to.
        #expect(rendered.components(separatedBy: "<|im_start|>assistant").count == 2)
    }

    /// The mask supervises the answer and its end-of-turn, and nothing of the
    /// system or user turns.
    @Test func qwen3MaskSupervisesTheAnswerAndItsEndOfTurnOnly() async throws {
        let renderer = try await Self.pinnedRenderer("qwen3")
        let examples = try FineTuneTrainer.tokenizeInstructionExamples(
            [Self.terseExample], renderer: renderer, modelID: "Qwen/Qwen3-0.6B")
        let (context, supervised, contiguous) = Self.split(
            try #require(examples.first), with: renderer.tokenizer)
        #expect(contiguous, "the mask must be a single trailing run of 1s")
        #expect(supervised == "Pong.<|im_end|>\n")
        #expect(context.contains("You are terse."))
        #expect(context.contains("Ping?"))
        #expect(!supervised.contains("Ping?"))
        // Everything the model is not trained to produce ends at the header
        // it will actually be handed at inference time.
        #expect(context.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
    }

    /// Qwen3 with thinking DISABLED: `ExperimentTasks.qwenContext` must reach
    /// the template, so the empty `<think></think>` block appears in the
    /// context exactly once and never in the supervised span.
    @Test func qwen3ThinkingDisabledFlowsThroughQwenContext() async throws {
        let renderer = try await Self.pinnedRenderer("qwen3")
        let examples = try FineTuneTrainer.tokenizeInstructionExamples(
            [Self.terseExample], renderer: renderer, modelID: "Qwen/Qwen3-0.6B")
        let (context, supervised, _) = Self.split(
            try #require(examples.first), with: renderer.tokenizer)
        #expect(context.components(separatedBy: "<think>").count == 2)
        #expect(!supervised.contains("<think>"))
    }

    /// The invariant the trainer now verifies before deriving the mask.
    @Test func qwen3PromptTokensArePrefixOfCompletedTokens() async throws {
        let renderer = try await Self.pinnedRenderer("qwen3")
        let context = ExperimentTasks.qwenContext(
            modelID: "Qwen/Qwen3-0.6B", qwenThinkingEnabled: false)
        let prompt = try renderer.chatTemplateTokens(
            messages: [
                ["role": "system", "content": "You are terse."],
                ["role": "user", "content": "Ping?"],
            ],
            addGenerationPrompt: true,
            additionalContext: context)
        let examples = try FineTuneTrainer.tokenizeInstructionExamples(
            [Self.terseExample], renderer: renderer, modelID: "Qwen/Qwen3-0.6B")
        let full = try #require(examples.first).tokens
        #expect(full.starts(with: prompt))
        #expect(full.count > prompt.count)
    }

    // MARK: - Gemma: the other family, and the folded system message

    /// Gemma has no system role, so `instructionTokens` folds the system
    /// message into the user turn. The completed render must still end at the
    /// answer's `<end_of_turn>`.
    @Test func gemmaFoldsSystemIntoUserAndEndsAtTheAnswersEndOfTurn() async throws {
        let renderer = try await Self.pinnedRenderer("gemma3")
        let examples = try FineTuneTrainer.tokenizeInstructionExamples(
            [Self.terseExample], renderer: renderer, modelID: "google/gemma-3-4b-it")
        let example = try #require(examples.first)
        let rendered = renderer.tokenizer.decode(
            tokens: example.tokens, skipSpecialTokens: false)
        #expect(rendered.hasSuffix("Pong.<end_of_turn>\n"), "\(rendered)")
        #expect(!rendered.contains("<start_of_turn>system"))
        let (context, supervised, contiguous) = Self.split(
            example, with: renderer.tokenizer)
        #expect(contiguous)
        #expect(supervised == "Pong.<end_of_turn>\n")
        // The system text lives inside the user turn, ahead of the question.
        #expect(context.contains("<start_of_turn>user\nYou are terse.\n\nPing?"))
        #expect(context.hasSuffix("<start_of_turn>model\n"))
    }

    // MARK: - Edge cases the mask has to survive

    /// An empty answer still yields an honest example: the only thing
    /// supervised is the end-of-turn (i.e. "stop here").
    @Test func emptyAnswerSupervisesOnlyTheEndOfTurn() async throws {
        let renderer = try await Self.pinnedRenderer("qwen3")
        let examples = try FineTuneTrainer.tokenizeInstructionExamples(
            [.init(user: "Ping?", assistant: "")],
            renderer: renderer,
            modelID: "Qwen/Qwen3-0.6B")
        let (_, supervised, contiguous) = Self.split(
            try #require(examples.first), with: renderer.tokenizer)
        #expect(contiguous)
        #expect(supervised == "<|im_end|>\n")
    }

    /// The `maxTokens` ceiling still drops overlong rows rather than
    /// truncating them into a half-answer.
    @Test func overlongExamplesAreDroppedByTheTokenCeiling() async throws {
        let renderer = try await Self.pinnedRenderer("qwen3")
        let kept = try FineTuneTrainer.tokenizeInstructionExamples(
            [Self.terseExample],
            renderer: renderer,
            modelID: "Qwen/Qwen3-0.6B",
            maxTokens: 4096)
        #expect(kept.count == 1)
        let dropped = try FineTuneTrainer.tokenizeInstructionExamples(
            [Self.terseExample],
            renderer: renderer,
            modelID: "Qwen/Qwen3-0.6B",
            maxTokens: 8)
        #expect(dropped.isEmpty)
    }

    /// A template whose generation header disagrees with its rendered
    /// assistant turn leaves no honest span, so the trainer REFUSES by name
    /// rather than masking the wrong tokens. (Skipping instead would look
    /// like "no usable rows" for what is a template problem.)
    @Test func templateWithoutAPrefixRelationshipIsRefused() async throws {
        let renderer = try await Self.pinnedRenderer("prefix-mismatch")
        #expect(throws: FineTuneTrainingError.self) {
            try FineTuneTrainer.tokenizeInstructionExamples(
                [Self.terseExample], renderer: renderer, modelID: "synthetic/mismatch")
        }
        do {
            _ = try FineTuneTrainer.tokenizeInstructionExamples(
                [Self.terseExample], renderer: renderer, modelID: "synthetic/mismatch")
            Issue.record("expected a refusal")
        } catch let error as FineTuneTrainingError {
            #expect(error.description.contains("synthetic/mismatch"))
            #expect(error.description.contains("example 1"))
        }
    }

    /// The no-chat-template fallback render: the completed example runs
    /// straight into the answer and appends NOTHING after it.
    @Test func fallbackRenderAppendsNothingAfterTheAnswer() async throws {
        let base = try await Self.pinnedRenderer("qwen3")
        let renderer = NoChatTemplateRenderer(base: base)
        let examples = try FineTuneTrainer.tokenizeInstructionExamples(
            [Self.terseExample], renderer: renderer, modelID: "toy/base-lm")
        let example = try #require(examples.first)
        let rendered = base.tokenizer.decode(
            tokens: example.tokens, skipSpecialTokens: false)
        #expect(rendered == "System: You are terse.\nUser: Ping?\nAssistant: Pong.")
        let (context, supervised, contiguous) = Self.split(
            example, with: base.tokenizer)
        #expect(contiguous)
        #expect(supervised == " Pong.")
        #expect(context.hasSuffix("Assistant:"))
    }

    // MARK: - The pinned fixtures themselves

    @Test func pinnedTemplatesRecordTheirProvenance() throws {
        let data = try Data(
            contentsOf: Self.templateDirectory.appending(component: "provenance.json"))
        let object = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let families = try #require(object["families"] as? [String: Any])
        for family in ["qwen3", "gemma3"] {
            let entry = try #require(families[family] as? [String: Any])
            #expect((entry["sourceModelID"] as? String)?.isEmpty == false)
            #expect((entry["sourceRevision"] as? String)?.count == 40)
            #expect((entry["pinnedOn"] as? String)?.isEmpty == false)
            let template = Self.templateDirectory.appending(
                components: family, "chat_template.jinja")
            #expect(FileManager.default.fileExists(atPath: template.path))
        }
    }

    /// Route (a): the same claims against the ACTUAL cached tokenizers —
    /// real BPE vocabularies, where "the prompt ids are a prefix of the full
    /// ids" is a genuine question rather than a byte-level tautology. Also
    /// fails loudly if the upstream template has drifted from the pinned one.
    /// Skips (printed, named) when a model is not in the local cache.
    @Test(
        arguments: [
            ("Qwen/Qwen3-0.6B", "Pong.<|im_end|>\n"),
            ("google/gemma-3-4b-it", "Pong.<end_of_turn>\n"),
        ])
    func cachedSnapshotAgreesWithThePinnedFixture(
        modelID: String, expectedSupervised: String
    ) async throws {
        guard let snapshot = Self.cachedSnapshot(for: modelID) else {
            print(
                "SKIP FineTuneTokenizationTests.cachedSnapshotAgreesWithThePinnedFixture"
                    + "(\(modelID)): tokenizer not in the local HF cache — the pinned "
                    + "fixture still covers this family, but the live template and "
                    + "vocabulary were NOT checked")
            return
        }
        let renderer = CachedSnapshotInstructionRenderer(
            tokenizer: try await AutoTokenizer.from(modelFolder: snapshot))
        let family = modelID.lowercased().contains("gemma") ? "gemma3" : "qwen3"
        let pinned = try String(
            contentsOf: Self.templateDirectory.appending(
                components: family, "chat_template.jinja"),
            encoding: .utf8)
        let messages: [[String: any Sendable]] =
            family == "gemma3"
            ? [["role": "user", "content": "You are terse.\n\nPing?"]]
            : [
                ["role": "system", "content": "You are terse."],
                ["role": "user", "content": "Ping?"],
            ]
        let additional = ExperimentTasks.qwenContext(
            modelID: modelID, qwenThinkingEnabled: false)
        // Drift guard: the pinned bytes and the cached model's own template
        // must still render identically.
        let live = try renderer.chatTemplateTokens(
            messages: messages, addGenerationPrompt: true, additionalContext: additional)
        let fromPinned = try renderer.tokenizer.applyChatTemplate(
            messages: messages,
            chatTemplate: .literal(pinned),
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: nil,
            additionalContext: additional)
        #expect(
            live == fromPinned,
            "\(modelID)'s upstream chat template has drifted from the pinned fixture")

        let examples = try FineTuneTrainer.tokenizeInstructionExamples(
            [Self.terseExample], renderer: renderer, modelID: modelID)
        let example = try #require(examples.first)
        #expect(example.tokens.starts(with: live))
        let (_, supervised, contiguous) = Self.split(example, with: renderer.tokenizer)
        #expect(contiguous)
        #expect(supervised == expectedSupervised)
    }
}

/// A renderer whose tokenizer has no chat template — the fallback path.
private struct NoChatTemplateRenderer: FineTuneInstructionRenderer {
    let base: CachedSnapshotInstructionRenderer

    func chatTemplateTokens(
        messages: [[String: any Sendable]],
        addGenerationPrompt: Bool,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw Tokenizers.TokenizerError.missingChatTemplate
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        base.encode(text: text, addSpecialTokens: addSpecialTokens)
    }
}
