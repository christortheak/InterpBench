import Foundation
import Testing

@testable import ExperimentKit

/// Send-as-assistant (seeded turns) + assistant-prefix continuation — the
/// metacognition instrument's pure-CPU contract:
///
/// 1. Seeding appends a MARKED assistant turn without generating.
/// 2. The transcript → wire mapping carries roles + seeded flags
///    (`{"role","content","seeded"?}` — the cross-engine message contract).
/// 3. Send-as-assistant is refused, with a self-naming reason, on servers
///    that do not advertise `chat.seededTurns` (and continuation on servers
///    without `chat.continueFinalMessage`).
/// 4. The family adapter maps transcripts to template messages with the
///    seeded flag INVISIBLE to rendering (byte-identity with real assistant
///    turns is pinned by the golden render fixtures).
/// 5. The continuation sentinel cut mirrors transformers'
///    `continue_final_message` algorithm exactly.
/// 6. Every export label distinguishes seeded turns from generations.
@MainActor
struct SeededTurnTests {

    private func freshDefaults(_ name: String) throws -> UserDefaults {
        let suite = "steerlab.tests.seeded-turns.\(name)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }

    private func serverWorkspaceService(
        _ name: String, capabilities: ClusterCapabilities?
    ) throws -> ChatService {
        let store = clusterStore(defaults: try freshDefaults(name))
        let entry = store.addServer(name: "test", urlString: "http://127.0.0.1:8080")
        store.activeWorkspace = .server(entry.id)
        store.capabilities = capabilities
        return ChatService(cluster: store)
    }

    private func decodedCapabilities(_ json: String) throws -> ClusterCapabilities {
        try JSONDecoder().decode(ClusterCapabilities.self, from: Data(json.utf8))
    }

    // MARK: - Capability decode (the feature gate's wire form)

    @Test func chatCapabilityFlagsDecodeFromTheServerSnapshot() throws {
        let capabilities = try decodedCapabilities(
            #"{"chat": {"seededTurns": true, "continueFinalMessage": true}}"#)
        #expect(capabilities.supportsSeededChatTurns)
        #expect(capabilities.supportsAssistantPrefillContinuation)
    }

    @Test func absentChatBlockMeansUnsupported() throws {
        // Older servers omit the block entirely — absent means unsupported,
        // never a guess.
        let capabilities = try decodedCapabilities(#"{"serverVersion": "0.9"}"#)
        #expect(!capabilities.supportsSeededChatTurns)
        #expect(!capabilities.supportsAssistantPrefillContinuation)
    }

    // MARK: - Seeding is blocked with a clear reason on non-supporting servers

    @Test func seedingRefusedOnServerWithoutCapability() throws {
        let service = try serverWorkspaceService(
            "no-cap",
            capabilities: try decodedCapabilities(#"{"serverVersion": "0.9"}"#))
        #expect(service.seedUnavailableReason?.contains("chat.seededTurns") == true)
        service.seedAssistantTurn("planted words")
        #expect(service.transcript.isEmpty, "refused seed must not append")
        #expect(service.errorMessage?.contains("chat.seededTurns") == true)
    }

    @Test func seedingRefusedBeforeCapabilitiesKnown() throws {
        let service = try serverWorkspaceService("unknown-cap", capabilities: nil)
        #expect(service.seedUnavailableReason?.contains("capabilities") == true)
        service.seedAssistantTurn("planted words")
        #expect(service.transcript.isEmpty)
    }

    @Test func seedingAppendsMarkedTurnWithoutGeneratingOnSupportingServer() throws {
        let service = try serverWorkspaceService(
            "with-cap",
            capabilities: try decodedCapabilities(
                #"{"chat": {"seededTurns": true, "continueFinalMessage": true}}"#))
        service.seedAssistantTurn("  I refuse to answer that.  ")
        #expect(service.transcript.count == 1)
        let turn = try #require(service.transcript.first)
        #expect(turn.role == .assistant)
        #expect(turn.seeded)
        #expect(turn.seededPrefixLength == nil)
        #expect(turn.continuationPrefixLength == nil)
        #expect(turn.text == "I refuse to answer that.")
        #expect(!service.isGenerating, "seeding must not trigger generation")
        #expect(service.errorMessage == nil)
    }

    @Test func seedingRequiresChatMode() throws {
        let service = try serverWorkspaceService(
            "raw-mode",
            capabilities: try decodedCapabilities(
                #"{"chat": {"seededTurns": true}}"#))
        service.promptMode = .rawCompletion
        #expect(service.seedUnavailableReason?.contains("chat mode") == true)
        service.seedAssistantTurn("planted")
        #expect(service.transcript.isEmpty)
    }

    @Test func localSeedingRequiresALoadedModel() throws {
        // Local workspace, nothing loaded: loading clears the transcript, so
        // an earlier seed would silently vanish — refuse instead.
        let service = ChatService(
            cluster: clusterStore(defaults: try freshDefaults("local")))
        #expect(service.seedUnavailableReason?.contains("load a model") == true)
        service.seedAssistantTurn("planted")
        #expect(service.transcript.isEmpty)
    }

    @Test func continuationRefusedOnServerWithoutPrefillCapability() throws {
        let service = try serverWorkspaceService(
            "seed-only",
            capabilities: try decodedCapabilities(
                #"{"chat": {"seededTurns": true}}"#))
        #expect(service.seedUnavailableReason == nil)
        #expect(
            service.continuationUnavailableReason?
                .contains("chat.continueFinalMessage") == true)
        service.continueSeededAssistantTurn("The court holds")
        #expect(service.transcript.isEmpty)
        #expect(!service.isGenerating)
    }

    // MARK: - Transcript → wire mapping (roles + seeded flags)

    @Test func wireMessagesCarryRolesAndSeededFlags() {
        let turns: [ChatService.ChatMessage] = [
            .init(role: .user, text: "State the holding."),
            .init(role: .assistant, text: "I refuse.", seeded: true),
            .init(role: .user, text: "Why did you refuse?"),
            .init(role: .assistant, text: "Because…"),
        ]
        let wire = ChatService.wireMessages(from: turns)
        #expect(wire.map(\.role) == ["user", "assistant", "user", "assistant"])
        #expect(wire.map(\.content) == [
            "State the holding.", "I refuse.", "Why did you refuse?", "Because…",
        ])
        // seeded is present ONLY on the seeded turn (absent = false on the
        // wire, so real turns carry no flag at all).
        #expect(wire.map(\.seeded) == [nil, true, nil, nil])
    }

    @Test func continuedTurnsAreSeededOnTheWire() {
        var continued = ChatService.ChatMessage(
            role: .assistant, text: "The court holds that X", seeded: true)
        continued.seededPrefixLength = 15
        let wire = ChatService.wireMessages(from: [continued])
        #expect(wire.first?.seeded == true)
    }

    @Test func seededFlagOmittedFromWireEncodingWhenAbsent() throws {
        // The cross-engine contract: absent = false. A real assistant turn
        // must serialize WITHOUT a "seeded" key.
        let data = try JSONEncoder().encode(
            ChatService.wireMessages(from: [
                .init(role: .assistant, text: "real generation")
            ]))
        let object = try #require(
            try JSONSerialization.jsonObject(with: data) as? [[String: Any]])
        #expect(object.first?.keys.contains("seeded") == false)
    }

    // MARK: - Family adapter: transcript → template messages

    @Test func chatHistoryDropsSeededFlagFromRendering() {
        // The adapter's output has no provenance channel at all — a seeded
        // turn maps to the SAME Chat.Message a real one does (byte-identity
        // through the template is pinned by the golden fixtures).
        let seeded = ExperimentTasks.chatHistoryMessages(
            turns: [
                .init(role: .user, text: "q"),
                .init(role: .assistant, text: "a", seeded: true),
            ],
            modelID: "Qwen/Qwen3-4B", systemPrompt: nil)
        let real = ExperimentTasks.chatHistoryMessages(
            turns: [
                .init(role: .user, text: "q"),
                .init(role: .assistant, text: "a"),
            ],
            modelID: "Qwen/Qwen3-4B", systemPrompt: nil)
        #expect(seeded.map(\.role) == real.map(\.role))
        #expect(seeded.map(\.content) == real.map(\.content))
        #expect(seeded.map(\.role) == [.user, .assistant])
    }

    @Test func gemmaHistoryFoldsSystemIntoEveryUserTurn() {
        // The LOCAL interactive convention (ChatService.chatPrompt prepends
        // per turn) — replay must reproduce what the model actually saw.
        let messages = ExperimentTasks.chatHistoryMessages(
            turns: [
                .init(role: .user, text: "first"),
                .init(role: .assistant, text: "reply"),
                .init(role: .user, text: "second"),
            ],
            modelID: "mlx-community/gemma-3-4b-it-4bit",
            systemPrompt: "Be terse.")
        #expect(messages[0].content == "Be terse.\n\nfirst")
        #expect(messages[1].content == "reply")
        #expect(messages[2].content == "Be terse.\n\nsecond")
    }

    @Test func nonGemmaHistoryAddsNoSystemMessage() {
        // ChatSession's `instructions:` supplies the system role; the
        // history adapter must not double it.
        let messages = ExperimentTasks.chatHistoryMessages(
            turns: [.init(role: .user, text: "q")],
            modelID: "Qwen/Qwen3-4B", systemPrompt: "Be terse.")
        #expect(messages.count == 1)
        #expect(messages[0].content == "q")
    }

    @Test func emptyTurnsAreDropped() {
        let messages = ExperimentTasks.chatHistoryMessages(
            turns: [
                .init(role: .user, text: "  "),
                .init(role: .assistant, text: "kept"),
            ],
            modelID: "Qwen/Qwen3-4B", systemPrompt: nil)
        #expect(messages.map(\.content) == ["kept"])
    }

    // MARK: - Continuation sentinel cut (transformers' algorithm, mirrored)

    @Test func continuationCutWithPreservedSpacing() throws {
        let sentinel = ExperimentTasks.continueFinalMessageSentinel
        let rendered = "<|assistant|>The court holds" + sentinel
        let cut = try ExperimentTasks.continuationCut(
            rendered: rendered, finalContent: "The court holds" + sentinel)
        #expect(cut == "<|assistant|>The court holds")
    }

    @Test func continuationCutTrimsWhenTemplateTrimmedTrailingSpace() throws {
        // The template trimmed the message tail, leaving the sentinel WITHOUT
        // its trailing space — transformers rstrips the head; mirror that.
        let rendered = "<|assistant|>The court holds \nCONTINUE_FINAL_MESSAGE_TAG"
        let cut = try ExperimentTasks.continuationCut(
            rendered: rendered,
            finalContent: "The court holds \nCONTINUE_FINAL_MESSAGE_TAG ")
        #expect(cut == "<|assistant|>The court holds")
    }

    @Test func continuationCutRefusesWhenTemplateTransformedContent() {
        // A template that deletes/transforms the final content (e.g. thinking
        // blocks) cannot be continued honestly — loud refusal, no guess.
        #expect(throws: (any Error).self) {
            try ExperimentTasks.continuationCut(
                rendered: "<|assistant|>something else entirely",
                finalContent: "The court holds")
        }
    }

    // MARK: - Export labels (persistence honesty)

    @Test func turnLabelsDistinguishSeededTurns() {
        #expect(
            ChatService.turnLabel(.init(role: .assistant, text: "x"))
                == "Assistant")
        #expect(
            ChatService.turnLabel(.init(role: .assistant, text: "x", seeded: true))
                .contains("seeded"))
        var continued = ChatService.ChatMessage(
            role: .assistant, text: "prefix rest", seeded: true)
        continued.seededPrefixLength = 6
        let label = ChatService.turnLabel(continued)
        #expect(label.contains("seeded prefix"))
        #expect(label.contains("6"))
    }

    @Test func transcriptMarkdownMarksSeededTurns() throws {
        let service = try serverWorkspaceService(
            "markdown",
            capabilities: try decodedCapabilities(
                #"{"chat": {"seededTurns": true}}"#))
        service.seedAssistantTurn("planted words")
        let markdown = service.transcriptMarkdown()
        #expect(markdown.contains("### Assistant (seeded — researcher-authored, not generated)"))
        #expect(markdown.contains("planted words"))
    }

    // MARK: - Family conversation constraints (the pre-flight table)

    @Test func constraintTableMatchesEmpiricalFamilyBehavior() {
        // Empirical (2026-07-13, real pinned tokenizers — the live-template
        // check lives in GoldenRenderFixtureTests): Gemma 3 enforces
        // user-first strict alternation; Qwen3 is fully permissive.
        let gemma = ExperimentTasks.conversationConstraints(
            modelID: "mlx-community/gemma-3-12b-it-8bit")
        #expect(gemma.requiresLeadingUserTurn)
        #expect(gemma.forbidsConsecutiveSameRole)
        let qwen = ExperimentTasks.conversationConstraints(
            modelID: "Qwen/Qwen3-14B-MLX-8bit")
        #expect(!qwen.requiresLeadingUserTurn)
        #expect(!qwen.forbidsConsecutiveSameRole)
    }

    @Test func assistantFirstBlockedOnGemmaAllowedOnQwen() {
        let blocked = ExperimentTasks.conversationConstraintViolation(
            appending: .assistant, toTranscriptRoles: [],
            modelID: "mlx-community/gemma-3-4b-it-4bit")
        #expect(blocked?.contains("start with a user turn") == true)
        #expect(blocked?.contains("gemma-3") == true)
        #expect(
            ExperimentTasks.conversationConstraintViolation(
                appending: .assistant, toTranscriptRoles: [],
                modelID: "Qwen/Qwen3-4B-MLX-4bit") == nil)
    }

    @Test func consecutiveSameRoleBlockedOnGemmaAllowedOnQwen() {
        let assistantPair = ExperimentTasks.conversationConstraintViolation(
            appending: .assistant, toTranscriptRoles: [.user, .assistant],
            modelID: "mlx-community/gemma-3-4b-it-4bit")
        #expect(assistantPair?.contains("consecutive assistant") == true)
        let userPair = ExperimentTasks.conversationConstraintViolation(
            appending: .user, toTranscriptRoles: [.user],
            modelID: "mlx-community/gemma-3-4b-it-4bit")
        #expect(userPair?.contains("consecutive user") == true)
        #expect(
            ExperimentTasks.conversationConstraintViolation(
                appending: .assistant, toTranscriptRoles: [.user, .assistant],
                modelID: "Qwen/Qwen3-4B-MLX-4bit") == nil)
        #expect(
            ExperimentTasks.conversationConstraintViolation(
                appending: .user, toTranscriptRoles: [.user],
                modelID: "Qwen/Qwen3-4B-MLX-4bit") == nil)
    }

    @Test func alternatingHistoryIsAlwaysLegal() {
        #expect(
            ExperimentTasks.conversationConstraintViolation(
                appending: .user, toTranscriptRoles: [.user, .assistant],
                modelID: "mlx-community/gemma-3-4b-it-4bit") == nil)
    }

    // MARK: - ChatService pre-flight (composer gate, before any request)

    private func gemmaCapableServerService(_ name: String) throws -> ChatService {
        let service = try serverWorkspaceService(
            name,
            capabilities: try decodedCapabilities(
                #"{"chat": {"seededTurns": true, "continueFinalMessage": true}}"#))
        service.selectedRemoteModelID = "mlx-community/gemma-3-4b-it-4bit"
        return service
    }

    @Test func assistantFirstSeedPreflightRefusesOnGemma() throws {
        // The user-reported defect's client half: the seed is refused with
        // an inline explanation instead of letting the next send 500.
        let service = try gemmaCapableServerService("preflight-gemma")
        #expect(service.seedUnavailableReason?.contains("start with a user turn") == true)
        service.seedAssistantTurn("planted words")
        #expect(service.transcript.isEmpty)
        #expect(service.errorMessage?.contains("start with a user turn") == true)
        // Continuation shares the same gate.
        #expect(
            service.continuationUnavailableReason?
                .contains("start with a user turn") == true)
    }

    @Test func assistantFirstSeedAllowedOnQwen() throws {
        let service = try serverWorkspaceService(
            "preflight-qwen",
            capabilities: try decodedCapabilities(
                #"{"chat": {"seededTurns": true}}"#))
        service.selectedRemoteModelID = "Qwen/Qwen3-4B-MLX-4bit"
        #expect(service.seedUnavailableReason == nil)
        service.seedAssistantTurn("planted words")
        #expect(service.transcript.count == 1)
    }

    @Test func gemmaUserFirstThenAssistantSeedIsTheCanonicalPath() throws {
        // Lead with a seeded user turn, then the assistant seed is legal —
        // exactly the fix the pre-flight message suggests.
        let service = try gemmaCapableServerService("preflight-canonical")
        #expect(service.seedUserUnavailableReason == nil)
        service.seedUserTurn("State the holding.")
        #expect(service.seedUnavailableReason == nil)
        service.seedAssistantTurn("I refuse.")
        #expect(service.transcript.map(\.role) == [.user, .assistant])
        // And a THIRD consecutive-role move is refused again.
        #expect(
            service.seedUnavailableReason?.contains("consecutive assistant") == true)
        #expect(
            service.turnConstraintReason(for: .user) == nil,
            "a user turn alternates correctly and must be allowed")
    }

    @Test func consecutiveUserSeedRefusedOnGemmaAllowedOnQwen() throws {
        let gemma = try gemmaCapableServerService("preflight-user-gemma")
        gemma.seedUserTurn("first question")
        #expect(gemma.seedUserUnavailableReason?.contains("consecutive user") == true)
        gemma.seedUserTurn("second question")
        #expect(gemma.transcript.count == 1, "refused seed must not append")
        #expect(gemma.turnConstraintReason(for: .user)?.contains("consecutive user") == true)

        let qwen = try serverWorkspaceService(
            "preflight-user-qwen",
            capabilities: try decodedCapabilities(#"{"chat": {"seededTurns": true}}"#))
        qwen.selectedRemoteModelID = "Qwen/Qwen3-4B-MLX-4bit"
        qwen.seedUserTurn("first")
        qwen.seedUserTurn("second")
        #expect(qwen.transcript.count == 2)
    }

    @Test func rawCompletionModeIsNeverConstrained() throws {
        let service = try gemmaCapableServerService("preflight-raw")
        service.promptMode = .rawCompletion
        #expect(service.turnConstraintReason(for: .assistant) == nil)
        #expect(service.turnConstraintReason(for: .user) == nil)
    }

    // MARK: - Seeding USER turns (scripted transcripts)

    @Test func seedUserTurnAppendsUnflaggedUserTurnWithoutGenerating() throws {
        let service = try serverWorkspaceService(
            "seed-user",
            capabilities: try decodedCapabilities(#"{"chat": {"seededTurns": true}}"#))
        service.seedUserTurn("  State the holding.  ")
        #expect(service.transcript.count == 1)
        let turn = try #require(service.transcript.first)
        #expect(turn.role == .user)
        #expect(!turn.seeded, "user turns carry no provenance flag")
        #expect(!turn.edited)
        #expect(!service.isGenerating)
        #expect(service.pendingSeededHistoryReplay, "seeded history must replay")
        // On the wire it is a plain user message — no flags at all.
        let wire = ChatService.wireMessages(from: service.transcript)
        #expect(wire == [ChatWireMessage(role: "user", content: "State the holding.")])
    }

    @Test func seedUserTurnDoesNotRequireSeededTurnsCapability() throws {
        // A seeded user turn is a plain message on the wire (no flag), so an
        // older server that lacks chat.seededTurns can still receive it.
        let service = try serverWorkspaceService(
            "seed-user-oldserver",
            capabilities: try decodedCapabilities(#"{"serverVersion": "0.9"}"#))
        #expect(service.seedUserUnavailableReason == nil)
        service.seedUserTurn("hello")
        #expect(service.transcript.count == 1)
    }

    @Test func localSeedUserTurnRequiresALoadedModel() throws {
        let service = ChatService(
            cluster: clusterStore(defaults: try freshDefaults("seed-user-local")))
        #expect(service.seedUserUnavailableReason?.contains("load a model") == true)
        service.seedUserTurn("hello")
        #expect(service.transcript.isEmpty)
    }

    // MARK: - Transcript editing (provenance rules + replay invalidation)

    private func editableService(_ name: String) throws -> ChatService {
        try serverWorkspaceService(
            name,
            capabilities: try decodedCapabilities(
                #"{"chat": {"seededTurns": true, "continueFinalMessage": true}}"#))
    }

    @Test func editingGeneratedAssistantTurnSetsEditedFlag() throws {
        let service = try editableService("edit-generated")
        service.transcript = [
            .init(role: .user, text: "q"),
            .init(role: .assistant, text: "pure generation"),
        ]
        service.pendingSeededHistoryReplay = false
        let id = try #require(service.transcript.last?.id)
        service.editTranscriptMessage(id: id, newText: "altered generation")
        let turn = try #require(service.transcript.last)
        #expect(turn.text == "altered generation")
        #expect(turn.edited)
        #expect(!turn.seeded, "edited is DISTINCT from seeded")
        #expect(service.pendingSeededHistoryReplay, "edit must invalidate replay")
        #expect(
            ChatService.turnLabel(turn)
                == "Assistant (generated, then edited by researcher)")
    }

    @Test func editingSeededTurnKeepsSeededWithoutEditedFlag() throws {
        let service = try editableService("edit-seeded")
        service.seedAssistantTurn("planted words")
        service.pendingSeededHistoryReplay = false
        let id = try #require(service.transcript.last?.id)
        service.editTranscriptMessage(id: id, newText: "replanted words")
        let turn = try #require(service.transcript.last)
        #expect(turn.seeded, "a seeded turn stays seeded")
        #expect(!turn.edited, "still authored wholesale — not generated-then-altered")
        #expect(service.pendingSeededHistoryReplay)
        #expect(ChatService.turnLabel(turn).contains("seeded"))
    }

    @Test func editingContinuedTurnKeepsSeededAddsEditedDropsPrefix() throws {
        let service = try editableService("edit-continued")
        var continued = ChatService.ChatMessage(
            role: .assistant, text: "prefix then generated tail", seeded: true)
        continued.seededPrefixLength = 6
        service.transcript = [.init(role: .user, text: "q"), continued]
        let id = try #require(service.transcript.last?.id)
        service.editTranscriptMessage(id: id, newText: "rewritten mixed turn")
        let turn = try #require(service.transcript.last)
        #expect(turn.seeded)
        #expect(turn.edited)
        #expect(
            turn.seededPrefixLength == nil,
            "the prefix boundary no longer describes the text")
        #expect(turn.continuationPrefixLength == nil)
        let label = ChatService.turnLabel(turn)
        #expect(label.contains("seeded") && label.contains("edited"))
    }

    @Test func generatedContinuationBoundaryDoesNotClaimResearcherAuthorship() {
        var continued = ChatService.ChatMessage(
            role: .assistant, text: "generated prefix and continued tail")
        continued.continuationPrefixLength = 16
        #expect(!continued.seeded)
        #expect(continued.seededPrefixLength == nil)
        #expect(ChatService.turnLabel(continued).contains("generated prefix"))
        #expect(ChatService.wireMessages(from: [continued]).first?.seeded == nil)
    }

    @Test func continuingAMixedTurnAgainPreservesTheAuthoredPrefixBoundary() {
        var continued = ChatService.ChatMessage(
            role: .assistant, text: "Hello world", seeded: true)
        continued.seededPrefixLength = 5
        continued.continuationPrefixLength = 5
        ChatService.stampContinuationProvenance(on: &continued)

        #expect(
            continued.seededPrefixLength == 5,
            "re-continuing must not relabel model-generated text as researcher-authored")
        #expect(
            continued.continuationPrefixLength == 11,
            "the new continuation boundary is the end of the current mixed turn")
    }

    @Test func firstContinuationOfAWholeSeedRecordsItsAuthoredBoundary() {
        var seeded = ChatService.ChatMessage(
            role: .assistant, text: "researcher prefix", seeded: true)
        ChatService.stampContinuationProvenance(on: &seeded)
        #expect(seeded.seededPrefixLength == 17)
        #expect(seeded.continuationPrefixLength == 17)
    }

    @Test func editingUserTurnSetsNothing() throws {
        let service = try editableService("edit-user")
        service.transcript = [.init(role: .user, text: "original question")]
        service.pendingSeededHistoryReplay = false
        let id = try #require(service.transcript.first?.id)
        service.editTranscriptMessage(id: id, newText: "revised question")
        let turn = try #require(service.transcript.first)
        #expect(turn.text == "revised question")
        #expect(!turn.seeded && !turn.edited)
        #expect(service.pendingSeededHistoryReplay, "edits always invalidate replay")
        #expect(ChatService.turnLabel(turn) == "User")
    }

    @Test func editingPreservesTheFirstOriginalButNeverSendsItToTheModel() throws {
        let service = try editableService("edit-original")
        service.transcript = [.init(role: .user, text: "first version")]
        let id = try #require(service.transcript.first?.id)

        service.editTranscriptMessage(id: id, newText: "second version")
        service.editTranscriptMessage(id: id, newText: "third version")

        let turn = try #require(service.transcript.first)
        #expect(turn.text == "third version")
        #expect(turn.originalText == "first version")
        let wire = ChatService.wireMessages(from: service.transcript)
        #expect(wire.map(\.content) == ["third version"])
        #expect(!String(decoding: try JSONEncoder().encode(wire), as: UTF8.self)
            .contains("first version"))
        #expect(service.transcriptMarkdown().contains("Original turn (not in active context)"))
        #expect(service.transcriptMarkdown().contains("first version"))
    }

    @Test func returnControlBranchesAfterEditedAssistantAndSelectsUserComposer() throws {
        let service = try editableService("edit-return-control")
        service.composerRole = .assistant
        service.transcript = [
            .init(role: .user, text: "question one"),
            .init(role: .assistant, text: "original answer"),
            .init(role: .user, text: "later question"),
            .init(role: .assistant, text: "later answer"),
        ]
        let answerID = service.transcript[1].id

        service.returnControlToUserFromAssistant(
            id: answerID, newText: "edited answer")

        #expect(service.transcript.map(\.text) == ["question one", "edited answer"])
        #expect(service.transcript.last?.originalText == "original answer")
        #expect(service.transcript.last?.edited == true)
        #expect(service.composerRole == .user)
        #expect(service.pendingSeededHistoryReplay)
    }

    @Test func unavailableRestartDoesNotDiscardTheExistingBranch() throws {
        let service = ChatService(
            cluster: clusterStore(defaults: try freshDefaults("edit-restart-refused")))
        service.transcript = [
            .init(role: .user, text: "original question"),
            .init(role: .assistant, text: "existing answer"),
        ]
        let questionID = service.transcript[0].id

        service.restartConversationFromUser(id: questionID, newText: "edited question")

        #expect(service.transcript.map(\.text) == ["original question", "existing answer"])
        #expect(service.errorMessage?.contains("load a model") == true)
    }

    @Test func editToEmptyTextIsRefused() throws {
        let service = try editableService("edit-empty")
        service.transcript = [.init(role: .assistant, text: "keep me")]
        let id = try #require(service.transcript.first?.id)
        service.editTranscriptMessage(id: id, newText: "   ")
        #expect(service.transcript.first?.text == "keep me")
        #expect(service.errorMessage?.contains("Remove turn") == true)
    }

    @Test func editedFlagRidesTheWireLikeSeeded() throws {
        var message = ChatService.ChatMessage(role: .assistant, text: "altered")
        message.edited = true
        let wire = ChatService.wireMessages(from: [message])
        #expect(wire.first?.edited == true)
        #expect(wire.first?.seeded == nil)
        // Absent = false: an unedited turn serializes WITHOUT the key.
        let plain = try JSONEncoder().encode(
            ChatService.wireMessages(from: [.init(role: .assistant, text: "real")]))
        let object = try #require(
            try JSONSerialization.jsonObject(with: plain) as? [[String: Any]])
        #expect(object.first?.keys.contains("edited") == false)
    }

    // MARK: - Removing the trailing turn

    @Test func removeLastTranscriptMessageRemovesOnlyTheTail() throws {
        let service = try editableService("remove-last")
        service.transcript = [
            .init(role: .user, text: "q"),
            .init(role: .assistant, text: "a"),
        ]
        service.pendingSeededHistoryReplay = false
        let first = try #require(service.transcript.first)
        let last = try #require(service.transcript.last)
        #expect(!service.canRemoveTranscriptMessage(first), "only the LAST turn")
        #expect(service.canRemoveTranscriptMessage(last))
        service.removeLastTranscriptMessage()
        #expect(service.transcript.map(\.text) == ["q"])
        #expect(service.pendingSeededHistoryReplay, "removal invalidates replay")
        service.removeLastTranscriptMessage()
        #expect(service.transcript.isEmpty)
    }
}
