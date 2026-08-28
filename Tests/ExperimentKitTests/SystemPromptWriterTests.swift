import Foundation
import MLXLMCommon
import SteeringKit
import Testing

@testable import ExperimentKit

/// `experiment set-system-prompt` — the study's DEPLOYMENT FRAME, headlessly.
///
/// Field discovery (2026-08-28): the manifest's `systemPrompt` was writable
/// from the Study Setup panel's "Baseline system prompt" field and nowhere
/// else on this engine, so a replication study whose donor carries a
/// judge-persona frame could not be authored headlessly at all. Running it
/// without the persona would have been a different study, so the authoring
/// agent correctly refused to improvise one.
///
/// The suite tests the WRITE and the APPLICATION, because a writer whose
/// value never reaches the model would satisfy the first and fail the point:
/// what is written must be inserted as a genuine system turn where the
/// model's chat template has a system role, and prepended to the first user
/// turn where it does not. Both branches are asserted through
/// `ExperimentTasks.userInput` — the measured-generation entry point itself,
/// not a re-implementation of its rule.
///
/// Serialized and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct SystemPromptWriterTests {

    static let pinnedNow = Date(timeIntervalSince1970: 1_000)

    func withTempRoot<T>(_ body: (URL) async throws -> T) async throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "sysprompt-\(UUID().uuidString)")
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
        await ExperimentCLIRunner(sink: .discarding, now: { Self.pinnedNow })
            .run(namespace: "experiment", args)
    }

    func createStudy(_ name: String, model: String) async {
        await invoke(["create", name, "--model", model])
    }

    // MARK: - 1. The write

    /// Inline text, stored as the panel stores it: TRIMMED, and empty is
    /// ABSENT. The panel's save path is `nilIfEmpty`, and two spellings of one
    /// field that stored different bytes for the same typing would stamp two
    /// different effective `systemPromptHash` values for one study.
    @Test func theVerbStoresTheFrameTheWayThePanelDoes() async throws {
        try await withTempRoot { _ in
            await createStudy("demo", model: "Qwen/Qwen3-14B")
            let outcome = await invoke(
                ["set-system-prompt", "demo", "  You are a strict grader.  "])
            #expect(outcome.envelope.state == .ready)
            #expect(outcome.envelope.changed)
            let manifest = try ExperimentStore.load(name: "demo")
            #expect(manifest.systemPrompt == "You are a strict grader.")
        }
    }

    /// `""` clears, like every sibling declaration verb.
    @Test func theEmptyStringClearsTheDeclaration() async throws {
        try await withTempRoot { _ in
            await createStudy("demo", model: "Qwen/Qwen3-14B")
            await invoke(["set-system-prompt", "demo", "a frame"])
            let outcome = await invoke(["set-system-prompt", "demo", ""])
            #expect(outcome.envelope.state == .ready)
            let after = try ExperimentStore.load(name: "demo")
            #expect(after.systemPrompt == nil)
            #expect(
                outcome.envelope.message.contains(
                    "cleared the system prompt"))
        }
    }

    /// A whitespace-only frame is a cleared frame, not a frame of spaces —
    /// `SystemPromptComposition` treats whitespace as no content at all, so
    /// storing it would produce a declaration the renderer drops anyway.
    @Test func aWhitespaceOnlyFrameIsACleared0ne() async throws {
        try await withTempRoot { _ in
            await createStudy("demo", model: "Qwen/Qwen3-14B")
            await invoke(["set-system-prompt", "demo", "a frame"])
            await invoke(["set-system-prompt", "demo", "   \n  "])
            let after = try ExperimentStore.load(name: "demo")
            #expect(after.systemPrompt == nil)
        }
    }

    /// Draft-only through the standard gate — no second rule.
    @Test func aFrozenStudyRefusesWithTheImmutabilityLine() async throws {
        try await withTempRoot { _ in
            await createStudy("demo", model: "Qwen/Qwen3-14B")
            var manifest = try ExperimentStore.load(name: "demo")
            manifest.status = .frozen
            try ExperimentStore.save(manifest)
            let outcome = await invoke(
                ["set-system-prompt", "demo", "a frame"])
            #expect(outcome.envelope.state == .refused)
            #expect(outcome.envelope.exitCode == 65)
            let error = try #require(outcome.envelope.error)
            #expect(error.gate == LifecycleGate.statusImmutable.rawValue)
            #expect(error.reason.contains("duplicate it to iterate"))
            #expect(error.repairAction.contains("experiment duplicate demo"))
        }
    }

    /// No value at all is a MALFORMED invocation (64), and the usage line
    /// says what setting a frame physically does.
    @Test func aMissingValueIsAUsageRefusalThatExplainsDelivery() async throws {
        try await withTempRoot { _ in
            await createStudy("demo", model: "Qwen/Qwen3-14B")
            let outcome = await invoke(["set-system-prompt", "demo"])
            #expect(outcome.envelope.exitCode == 64)
            let reason = try #require(outcome.envelope.error?.reason)
            #expect(reason.contains("genuine system turn"))
            #expect(reason.contains("prepended to the first user turn"))
        }
    }

    // MARK: - 2. The application — what the model actually receives

    /// A family whose chat template HAS a system role gets a genuine system
    /// turn, carrying exactly the bytes the verb wrote. Driven through
    /// `ExperimentTasks.userInput`, the measured-generation entry point.
    @Test func aWrittenFrameBecomesASystemTurnOnASystemRoleFamily() async throws {
        try await withTempRoot { _ in
            await createStudy("demo", model: "Qwen/Qwen3-14B")
            await invoke(
                ["set-system-prompt", "demo", "You are a strict grader."])
            let manifest = try ExperimentStore.load(name: "demo")

            let input = ExperimentTasks.userInput(
                prompt: "Decide the item.", modelID: manifest.modelID,
                promptMode: manifest.promptMode ?? .chatAssistant,
                systemPrompt: manifest.systemPrompt,
                qwenThinkingEnabled: manifest.qwenThinkingEnabled ?? false)
            guard case .chat(let messages) = input.prompt else {
                Issue.record("chatAssistant must render as a chat")
                return
            }
            #expect(messages.map(\.role) == [.system, .user])
            #expect(messages[0].content == "You are a strict grader.")
            // …and the user turn is untouched: the frame did not eat it.
            #expect(messages[1].content == "Decide the item.")
            #expect(PromptRendering.hasSystemRole(manifest.modelID))
        }
    }

    /// A family with NO system role (Gemma) gets the SAME bytes prepended to
    /// the first user turn. The frame still reaches the model — which is the
    /// property the writer promises — by the only route the template allows.
    @Test func aWrittenFrameIsPrependedOnAFamilyWithoutASystemRole() async throws {
        try await withTempRoot { _ in
            await createStudy("demo", model: "google/gemma-3-27b-it")
            await invoke(
                ["set-system-prompt", "demo", "You are a strict grader."])
            let manifest = try ExperimentStore.load(name: "demo")

            let input = ExperimentTasks.userInput(
                prompt: "Decide the item.", modelID: manifest.modelID,
                promptMode: manifest.promptMode ?? .chatAssistant,
                systemPrompt: manifest.systemPrompt,
                qwenThinkingEnabled: false)
            guard case .chat(let messages) = input.prompt else {
                Issue.record("chatAssistant must render as a chat")
                return
            }
            #expect(messages.map(\.role) == [.user])
            #expect(
                messages[0].content
                    == "You are a strict grader.\n\nDecide the item.")
            #expect(!PromptRendering.hasSystemRole(manifest.modelID))
        }
    }

    /// The third route: `rawCompletion` has no template at all, and the frame
    /// is prepended to the prompt TEXT. There is no prompt mode on which a
    /// written frame is dropped, which is why the setter has no mode gate.
    @Test func aWrittenFrameIsPrependedUnderRawCompletion() async throws {
        try await withTempRoot { _ in
            await createStudy("demo", model: "meta/other-model")
            await invoke(["set-system-prompt", "demo", "Context."])
            await invoke(
                ["set-sampling", "demo", "--prompt-mode", "rawCompletion"])
            let manifest = try ExperimentStore.load(name: "demo")
            #expect(manifest.promptMode == .rawCompletion)

            let input = ExperimentTasks.userInput(
                prompt: "Continue.", modelID: manifest.modelID,
                promptMode: manifest.promptMode ?? .chatAssistant,
                systemPrompt: manifest.systemPrompt,
                qwenThinkingEnabled: false)
            guard case .text(let text) = input.prompt else {
                Issue.record("rawCompletion must render as text")
                return
            }
            #expect(text == "Context.\n\nContinue.")
        }
    }

    /// The echo names the ROUTE, not just the field: a researcher arming a
    /// persona has to know which one this model gives them.
    @Test func theEchoNamesTheDeliveryRoute() async throws {
        try await withTempRoot { _ in
            await createStudy("qwen", model: "Qwen/Qwen3-14B")
            await createStudy("gemma", model: "google/gemma-3-27b-it")
            let onQwen = await invoke(["set-system-prompt", "qwen", "F."])
            let onGemma = await invoke(["set-system-prompt", "gemma", "F."])
            #expect(
                onQwen.envelope.result?["delivery"] == .string("systemTurn"))
            #expect(
                onGemma.envelope.result?["delivery"]
                    == .string("prependedToFirstUserTurn"))
            // The DERIVED hash is the same on both — it is the frame's own
            // bytes, not the rendering.
            #expect(
                onQwen.envelope.result?["studyFrameHash"]
                    == onGemma.envelope.result?["studyFrameHash"])
            let frameHash = try #require(SystemPromptComposition.hash("F."))
            #expect(
                onQwen.envelope.result?["studyFrameHash"]
                    == .string(frameHash))
        }
    }

    /// An agent arm is not displaced: composition puts the persona first and
    /// this frame second, so declaring one adds a frame rather than replacing
    /// an identity.
    @Test func theFrameComposesAfterAnAgentPersona() async throws {
        try await withTempRoot { _ in
            await createStudy("demo", model: "Qwen/Qwen3-14B")
            await invoke(["set-system-prompt", "demo", "Answer in JSON."])
            let manifest = try ExperimentStore.load(name: "demo")
            #expect(
                SystemPromptComposition.compose(
                    agent: "You are a cautious reviewer.",
                    frame: manifest.systemPrompt)
                    == "You are a cautious reviewer.\n\nAnswer in JSON.")
        }
    }

    // MARK: - 3. The one path a frame does NOT reach

    /// A pinned item whose transcript opens with its OWN system turn replaces
    /// the study frame for that item. That is a declared cross-engine rule,
    /// not a bug — but it was silent, and a researcher who has just typed a
    /// persona is exactly the person who needs to hear it.
    @Test func transcriptItemsWithTheirOwnSystemTurnRaiseTheAdvisory() async throws {
        try await withTempRoot { root in
            await createStudy("demo", model: "Qwen/Qwen3-14B")
            let prompts = root.appending(
                components: "prompts", "tasks", "t.jsonl")
            try FileManager.default.createDirectory(
                at: prompts.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try
                (#"{"id": "a", "transcript": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Hi"}]}"#
                    + "\n"
                    + #"{"id": "b", "transcript": [{"role": "user", "content": "Hello"}]}"#
                    + "\n"
                    + #"{"id": "c", "prompt": "Plain item"}"# + "\n")
                .write(to: prompts, atomically: true, encoding: .utf8)
            await invoke(["pin-prompts", "demo", "prompts/tasks/t.jsonl"])

            let outcome = await invoke(["set-system-prompt", "demo", "Frame."])
            #expect(outcome.envelope.state == .okWithAdvisories)
            // Advisories NEVER change the exit code.
            #expect(outcome.envelope.exitCode == 0)
            let advisory = try #require((outcome.envelope.advisories ?? []).first)
            #expect(
                advisory.code == CLIAdvisory.systemPromptNotApplied.rawValue)
            #expect(advisory.detail.contains("1 of 3 pinned item(s)"))
            #expect(advisory.detail.contains("reaches the other 2 only"))
            #expect(
                outcome.envelope.result?["itemsWithOwnSystemTurn"]
                    == .number(1))
        }
    }

    /// No advisory when every pinned item is under the frame — and none when
    /// the frame is being CLEARED, where the substitution cannot matter.
    @Test func aFrameThatReachesEveryItemIsSilent() async throws {
        try await withTempRoot { root in
            await createStudy("demo", model: "Qwen/Qwen3-14B")
            let prompts = root.appending(
                components: "prompts", "tasks", "plain.jsonl")
            try FileManager.default.createDirectory(
                at: prompts.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try (#"{"id": "a", "prompt": "One"}"# + "\n"
                + #"{"id": "b", "prompt": "Two"}"# + "\n")
                .write(to: prompts, atomically: true, encoding: .utf8)
            await invoke(["pin-prompts", "demo", "prompts/tasks/plain.jsonl"])

            let declared = await invoke(["set-system-prompt", "demo", "F."])
            #expect(declared.envelope.state == .ready)
            #expect((declared.envelope.advisories ?? []).isEmpty)

            let cleared = await invoke(["set-system-prompt", "demo", ""])
            #expect((cleared.envelope.advisories ?? []).isEmpty)
        }
    }

    /// An unreadable pin is `verify`'s business, never this counter's: the
    /// advisory input degrades to "nothing to say" rather than refusing a
    /// write that has nothing to do with the prompts file.
    @Test func anUnreadablePromptsPinDoesNotBlockTheWrite() async throws {
        try await withTempRoot { root in
            await createStudy("demo", model: "Qwen/Qwen3-14B")
            let prompts = root.appending(
                components: "prompts", "tasks", "gone.jsonl")
            try FileManager.default.createDirectory(
                at: prompts.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try (#"{"id": "a", "prompt": "One"}"# + "\n")
                .write(to: prompts, atomically: true, encoding: .utf8)
            await invoke(["pin-prompts", "demo", "prompts/tasks/gone.jsonl"])
            try FileManager.default.removeItem(at: prompts)

            let outcome = await invoke(["set-system-prompt", "demo", "F."])
            #expect(outcome.envelope.state == .ready)
            #expect((outcome.envelope.advisories ?? []).isEmpty)
            let written = try ExperimentStore.load(name: "demo")
            #expect(written.systemPrompt == "F.")
        }
    }

    // MARK: - 4. Cross-engine sentence parity

    /// The advisory's detail is ONE sentence on both engines. Copied from
    /// `experiment_store.system_prompt_not_applied_detail`.
    @Test func theAdvisoryDetailTwinsTheServerSentence() {
        let serverLiteral =
            "1 of 3 pinned item(s) in prompts/tasks/t.jsonl carry their own "
            + "leading system turn, which REPLACES this study's system "
            + "prompt for those items — the frame declared here reaches the "
            + "other 2 only. Remove the transcripts' system turns to put "
            + "every item under one frame, or keep them and say in METHODS "
            + "that the frame is per-item."
        #expect(
            ExperimentStore.systemPromptNotAppliedDetail(
                replacedItems: 1, totalItems: 3,
                promptsFile: "prompts/tasks/t.jsonl") == serverLiteral)
    }
}
