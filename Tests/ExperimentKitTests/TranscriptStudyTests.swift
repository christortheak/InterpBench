import Foundation
import MLXLMCommon
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// Scripted-transcript study items (the metacognition-study instrument).
///
/// Task-prompt items may carry a `transcript` — a scripted multi-turn
/// conversation with researcher-authored assistant turns ("words in the
/// model's mouth"), pinned as hashed stimulus data through the ordinary
/// `taskPromptsHash`. This suite is the Swift twin of the server's
/// `test_transcript_study.py`; the validation MESSAGE strings are a
/// cross-engine contract replayed from the committed fixture
/// `prompts/fixtures/transcript-validation/cases.json`, and render byte
/// parity is pinned by the `study_transcript` golden fixtures
/// (`GoldenRenderFixtureTests`).
struct TranscriptStudyTests {

    // MARK: - Shared fixtures

    private static var repoRoot: URL {
        URL(filePath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private var transcript: [ExperimentTasks.TranscriptTurn] {
        [
            .init(role: "system", content: "You answer questions about your own prior statements."),
            .init(role: "user", content: "Name a color, and nothing else."),
            .init(role: "assistant", content: "I would rather not name a color."),
            .init(role: "user", content: "Did you just refuse? Answer yes or no."),
        ]
    }

    private var transcriptJSON: String {
        """
        [{"role": "system", "content": "You answer questions about your own prior statements."}, \
        {"role": "user", "content": "Name a color, and nothing else."}, \
        {"role": "assistant", "content": "I would rather not name a color.", "seeded": true}, \
        {"role": "user", "content": "Did you just refuse? Answer yes or no."}]
        """
    }

    private func manifest(
        modelID: String = "Qwen/Qwen3-14B-MLX-8bit"
    ) -> ExperimentManifest {
        ExperimentManifest(name: "meta", description: "", modelID: modelID)
    }

    // MARK: - 1. Validation messages: the committed cross-engine cases

    struct ValidationCase: Decodable {
        let name: String
        let itemID: String
        let modelID: String?
        let transcript: [ExperimentTasks.TranscriptTurn]
        let expect: String?
    }

    struct ValidationFixture: Decodable {
        let cases: [ValidationCase]
    }

    private static func loadCases() throws -> [ValidationCase] {
        let url = repoRoot.appending(
            components: "prompts", "fixtures", "transcript-validation", "cases.json")
        return try JSONDecoder()
            .decode(ValidationFixture.self, from: Data(contentsOf: url)).cases
    }

    @Test func validationMessagesMatchCommittedFixture() throws {
        let cases = try Self.loadCases()
        #expect(!cases.isEmpty)
        for c in cases {
            var violation = ExperimentTasks.transcriptSchemaViolation(
                c.transcript, itemID: c.itemID)
            if violation == nil, let modelID = c.modelID {
                violation = ExperimentTasks.transcriptFamilyViolation(
                    c.transcript, itemID: c.itemID, modelID: modelID)
            }
            #expect(
                violation == c.expect,
                """
                \(c.name): this engine's validation message diverged from the \
                cross-engine fixture (the Python twin replays the same file).
                expected: \(c.expect ?? "nil")
                got:      \(violation ?? "nil")
                """)
        }
    }

    @Test func fixtureCoversEveryRule() throws {
        let expects = try Self.loadCases().map(\.expect)
        for needle in [
            "transcript is empty", "allowed roles", "empty content",
            "at most one system turn", "assistant-prefix continuation",
            "must end with a user turn", "start with a user turn",
            "strict user/assistant alternation",
        ] {
            #expect(
                expects.contains { $0?.contains(needle) == true },
                "fixture lost coverage of rule: \(needle)")
        }
        #expect(expects.contains { $0 == nil }, "fixture needs valid cases too")
    }

    // MARK: - 2. parseTaskPrompts: transcript items

    @Test func parseAcceptsTranscriptItemAndDerivesDisplayText() throws {
        let jsonl = """
            {"id": "t1", "transcript": \(transcriptJSON), "options": ["yes", "no"], "target": "yes"}
            {"id": "p1", "prompt": "Plain item.", "options": ["yes", "no"]}
            """
        let prompts = try ExperimentTasks.parseTaskPrompts(Data(jsonl.utf8))
        #expect(prompts.count == 2)
        let t1 = prompts[0]
        #expect(t1.text == "Did you just refuse? Answer yes or no.")
        #expect(t1.transcript?.count == 4)
        // Unknown per-turn keys (the seeded flag) drop on decode — the
        // record's transcript copy is {role, content} only, matching the
        // server's normalization.
        #expect(
            t1.transcript?[2]
                == ExperimentTasks.TranscriptTurn(
                    role: "assistant", content: "I would rather not name a color."))
        #expect(t1.options == ["yes", "no"])
        #expect(prompts[1].transcript == nil)
    }

    @Test func parseExplicitTextWinsOverDerivedDisplayText() throws {
        let jsonl = #"{"id": "t1", "text": "custom display", "transcript": "#
            + transcriptJSON + "}"
        let prompts = try ExperimentTasks.parseTaskPrompts(Data(jsonl.utf8))
        #expect(prompts[0].text == "custom display")
    }

    @Test func parseRefusesSchemaViolationWithContractMessage() throws {
        let jsonl = """
            {"id": "bad-1", "transcript": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
            """
        #expect {
            try ExperimentTasks.parseTaskPrompts(Data(jsonl.utf8))
        } throws: { error in
            let reason = (error as? ExperimentError)?.reason ?? ""
            return reason.contains("item 'bad-1'")
                && reason.contains("assistant-prefix continuation is out of scope")
        }
    }

    @Test func parseStillRefusesItemsWithNeitherTextNorTranscript() {
        #expect(throws: ExperimentError.self) {
            _ = try ExperimentTasks.parseTaskPrompts(
                Data(#"{"id": "x", "options": ["a"]}"#.utf8))
        }
    }

    // MARK: - 3. Run-start + verify gates

    @Test func runStartRefusesFamilyIncompatibleTranscript() {
        let prompts = [
            ExperimentTasks.StudyPrompt(
                id: "g1", text: "d", options: nil, target: nil,
                anchorMonths: nil, severity: nil, arm: nil, caseID: nil,
                transcript: [
                    .init(role: "assistant", content: "Seeded first."),
                    .init(role: "user", content: "Q?"),
                ])
        ]
        #expect {
            try ExperimentTasks.checkTranscriptPrompts(
                prompts, manifest: manifest(modelID: "mlx-community/gemma-3-12b-it-8bit"))
        } throws: { error in
            let reason = (error as? ExperimentError)?.reason ?? ""
            return reason.contains("scripted transcripts are incompatible")
                && reason.contains("item 'g1'")
                && reason.contains("start with a user turn")
        }
    }

    @Test func runStartRefusesRawCompletionWithTranscripts() {
        var m = manifest()
        m.promptMode = .rawCompletion
        let prompts = [
            ExperimentTasks.StudyPrompt(
                id: "t1", text: "d", options: nil, target: nil,
                anchorMonths: nil, severity: nil, arm: nil, caseID: nil,
                transcript: transcript)
        ]
        #expect {
            try ExperimentTasks.checkTranscriptPrompts(prompts, manifest: m)
        } throws: { error in
            ((error as? ExperimentError)?.reason ?? "")
                .contains("render through the chat template by definition")
        }
    }

    @Test func runStartPassesCompatibleTranscriptsOnBothFamilies() throws {
        let prompts = [
            ExperimentTasks.StudyPrompt(
                id: "t1", text: "d", options: nil, target: nil,
                anchorMonths: nil, severity: nil, arm: nil, caseID: nil,
                transcript: transcript)
        ]
        try ExperimentTasks.checkTranscriptPrompts(
            prompts, manifest: manifest(modelID: "mlx-community/gemma-3-12b-it-8bit"))
        try ExperimentTasks.checkTranscriptPrompts(prompts, manifest: manifest())
    }

    @Test func pinViolationsNameItemAndRuleAtVerifyTime() throws {
        let assistantFirst = try ExperimentTasks.parseTaskPrompts(
            Data(
                """
                {"id": "g1", "transcript": [{"role": "assistant", "content": "Seeded."}, {"role": "user", "content": "Q?"}]}
                """.utf8))
        let violations = ExperimentTasks.transcriptPinViolations(
            assistantFirst,
            manifest: manifest(modelID: "mlx-community/gemma-3-12b-it-8bit"))
        #expect(violations.count == 1)
        #expect(violations[0].contains("item 'g1'"))
        #expect(violations[0].contains("start with a user turn"))
        // Qwen is permissive: the same transcript pins cleanly there.
        #expect(
            ExperimentTasks.transcriptPinViolations(
                assistantFirst, manifest: manifest()
            ).isEmpty)

        let valid = try ExperimentTasks.parseTaskPrompts(
            Data(("{\"id\": \"t1\", \"transcript\": " + transcriptJSON + "}").utf8))
        // Valid transcripts pin cleanly on the strict family too…
        #expect(
            ExperimentTasks.transcriptPinViolations(
                valid, manifest: manifest(modelID: "mlx-community/gemma-3-12b-it-8bit")
            ).isEmpty)
        // …but rawCompletion + transcripts is a manifest violation.
        var raw = manifest()
        raw.promptMode = .rawCompletion
        #expect(
            ExperimentTasks.transcriptPinViolations(valid, manifest: raw)
                == [ExperimentTasks.transcriptRawCompletionMessage])
    }

    // MARK: - 4. Rendering: composition rule + family conventions

    @Test func transcriptSystemTurnReplacesStudySystemPrompt() {
        let messages = ExperimentTasks.transcriptMessages(
            transcript, modelID: "Qwen/Qwen3-14B-MLX-8bit",
            studySystemPrompt: "STUDY SYSTEM — must not appear")
        #expect(messages.first?.role == .system)
        #expect(
            messages.first?.content
                == "You answer questions about your own prior statements.")
        #expect(!messages.contains { $0.content.contains("must not appear") })
        // Identical to rendering with no study system prompt at all.
        let twin = ExperimentTasks.transcriptMessages(
            transcript, modelID: "Qwen/Qwen3-14B-MLX-8bit", studySystemPrompt: nil)
        #expect(messages.map(\.content) == twin.map(\.content))
    }

    @Test func studySystemAppliesWhenTranscriptHasNoSystemTurn() {
        let messages = ExperimentTasks.transcriptMessages(
            Array(transcript.dropFirst()), modelID: "Qwen/Qwen3-14B-MLX-8bit",
            studySystemPrompt: "study system prompt")
        #expect(messages.first?.role == .system)
        #expect(messages.first?.content == "study system prompt")
    }

    @Test func gemmaFoldsEffectiveSystemIntoFirstUserTurnOnly() {
        // The SERVER transcript convention (render_messages): fold into the
        // FIRST user turn — not every turn (that is the local interactive
        // ChatService convention). Byte parity is pinned by the
        // study_transcript goldens; this asserts the message shape.
        let messages = ExperimentTasks.transcriptMessages(
            transcript, modelID: "mlx-community/gemma-3-12b-it-8bit",
            studySystemPrompt: nil)
        #expect(!messages.contains { $0.role == .system })
        #expect(messages.first?.role == .user)
        #expect(
            messages.first?.content
                == "You answer questions about your own prior statements.\n\n"
                + "Name a color, and nothing else.")
        // The second user turn is NOT folded.
        #expect(messages.last?.content == "Did you just refuse? Answer yes or no.")
    }

    @Test func studyUserInputRefusesRawCompletionForTranscripts() {
        #expect(throws: ExperimentError.self) {
            _ = try ExperimentTasks.studyUserInput(
                text: "d", transcript: transcript,
                modelID: "Qwen/Qwen3-14B-MLX-8bit",
                promptMode: .rawCompletion, systemPrompt: nil,
                qwenThinkingEnabled: false)
        }
    }

    @Test func studyUserInputFallsBackToPlainPathWithoutTranscript() throws {
        let input = try ExperimentTasks.studyUserInput(
            text: "Plain item.", transcript: nil,
            modelID: "Qwen/Qwen3-14B-MLX-8bit",
            promptMode: .rawCompletion, systemPrompt: nil,
            qwenThinkingEnabled: false)
        guard case .text(let text) = input.prompt else {
            Issue.record("plain rawCompletion item should build UserInput.Prompt.text")
            return
        }
        #expect(text.contains("Plain item."))
    }

    // MARK: - 5. Records carry the transcript + flag

    @Test func generationAndChoiceRecordsCarryTranscriptAndFlag() throws {
        var m = manifest()
        m.taskPromptsFile = "prompts/tasks/items.jsonl"
        m.taskPromptsHash = "aa"
        let prompt = ExperimentTasks.StudyPrompt(
            id: "t1", text: "Did you just refuse? Answer yes or no.",
            options: ["yes", "no"], target: "yes",
            anchorMonths: nil, severity: nil, arm: nil, caseID: nil,
            transcript: transcript)
        let row = ExperimentTasks.MetricRow(
            condition: "baseline", seed: 0, promptIndex: 1, promptID: "t1",
            wordCount: 3, distinct2: 1.0, markerDensity: [:])
        let record = ExperimentTasks.sampledGenerationRecord(
            manifest: m, experimentHash: "h", taskPromptsFile: "f",
            taskPromptsHash: "th", promptMode: .chatAssistant,
            systemPrompt: nil, qwenThinkingEnabled: false,
            condition: "baseline", seed: 0, promptIndex: 1, prompt: prompt,
            output: "The refusal was mine.", row: row)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let object = try #require(
            try JSONSerialization.jsonObject(with: encoder.encode(record))
                as? [String: Any])
        #expect(object["scriptedTranscript"] as? Bool == true)
        let turns = try #require(object["transcript"] as? [[String: Any]])
        #expect(turns.count == 4)
        #expect(turns[2]["role"] as? String == "assistant")
        #expect(Set(turns[2].keys) == ["role", "content"])
        #expect(object["prompt"] as? String == "Did you just refuse? Answer yes or no.")

        let choice = ChoiceResult(
            options: [
                OptionScore(option: "yes", tokenIDs: [1], tokenLogprobs: [-0.5]),
                OptionScore(option: "no", tokenIDs: [2], tokenLogprobs: [-2.0]),
            ],
            promptTokenCount: 5)
        let choiceRecord = ExperimentTasks.choiceRecord(
            manifest: m, experimentHash: "h", taskPromptsFile: "f",
            taskPromptsHash: "th", promptMode: .chatAssistant,
            systemPrompt: nil, qwenThinkingEnabled: false,
            condition: "baseline", promptIndex: 1, prompt: prompt,
            choice: choice)
        let choiceObject = try #require(
            try JSONSerialization.jsonObject(with: encoder.encode(choiceRecord))
                as? [String: Any])
        #expect(choiceObject["scriptedTranscript"] as? Bool == true)
        #expect((choiceObject["transcript"] as? [[String: Any]])?.count == 4)

        // Plain items omit BOTH keys — the record shape is unchanged.
        let plain = ExperimentTasks.StudyPrompt(
            id: "p1", text: "Plain.", options: ["yes", "no"], target: nil,
            anchorMonths: nil, severity: nil, arm: nil, caseID: nil)
        let plainRecord = ExperimentTasks.sampledGenerationRecord(
            manifest: m, experimentHash: "h", taskPromptsFile: "f",
            taskPromptsHash: "th", promptMode: .chatAssistant,
            systemPrompt: nil, qwenThinkingEnabled: false,
            condition: "baseline", seed: 0, promptIndex: 2, prompt: plain,
            output: "x", row: row)
        let plainObject = try #require(
            try JSONSerialization.jsonObject(with: encoder.encode(plainRecord))
                as? [String: Any])
        #expect(plainObject["scriptedTranscript"] == nil)
        #expect(plainObject["transcript"] == nil)
    }

    // MARK: - 6. TaskPromptsDocument round-trip (field-faithful transcripts)

    @Test func documentRoundTripsTranscriptLinesByteFaithfully() throws {
        let line1 =
            #"{"id": "t1", "options": ["yes", "no"], "transcript": "#
            + transcriptJSON + "}"
        let line2 = #"{"id": "p1", "text": "Plain item."}"#
        let data = Data((line1 + "\n" + line2 + "\n").utf8)
        let document = try TaskPromptsDocument.load(data)
        #expect(document.count == 2)
        #expect(document.transcriptItemCount == 1)
        // Untouched documents serialize byte-identically — the transcript
        // (seeded flag included) survives verbatim, so the pinned
        // taskPromptsHash still covers it.
        #expect(document.serialized() == data)
        // A transcript-only item's editable text IS the final user turn.
        #expect(document.texts[0] == "Did you just refuse? Answer yes or no.")
    }

    @Test func editingTranscriptOnlyItemRewritesFinalUserTurnNotAShadowText() throws {
        let line = #"{"id": "t1", "transcript": "# + transcriptJSON + "}"
        let document = try TaskPromptsDocument.load(Data((line + "\n").utf8))
        let edited = document.applyingEditedTexts(
            ["Did you refuse on purpose? Answer yes or no."])
        let out = edited.serialized()
        let object = try #require(
            try JSONSerialization.jsonObject(
                with: Data(String(decoding: out, as: UTF8.self)
                    .split(separator: "\n")[0].utf8)) as? [String: Any])
        // No shadow text key was invented…
        #expect(object["text"] == nil)
        // …the transcript's final user turn carries the edit.
        let turns = try #require(object["transcript"] as? [[String: Any]])
        #expect(
            turns.last?["content"] as? String
                == "Did you refuse on purpose? Answer yes or no.")
        // Other keys (the seeded flag on turn 3) survive the rewrite.
        #expect(turns[2]["seeded"] as? Bool == true)
        // And the run loop's parser reads the edited text back as the
        // display text.
        let reparsed = try ExperimentTasks.parseTaskPrompts(out)
        #expect(reparsed[0].text == "Did you refuse on purpose? Answer yes or no.")
    }

    @Test func documentWithExplicitTextEditsTextKeyAndKeepsTranscript() throws {
        let line =
            #"{"id": "t1", "text": "display", "transcript": "# + transcriptJSON + "}"
        let document = try TaskPromptsDocument.load(Data((line + "\n").utf8))
        #expect(document.texts[0] == "display")
        let edited = document.applyingEditedTexts(["new display"])
        let object = try #require(
            try JSONSerialization.jsonObject(
                with: Data(String(decoding: edited.serialized(), as: UTF8.self)
                    .split(separator: "\n")[0].utf8)) as? [String: Any])
        #expect(object["text"] as? String == "new display")
        #expect((object["transcript"] as? [Any])?.count == 4)
    }

    // MARK: - 7. Import preview

    @Test func importPreviewCountsTranscriptRecords() throws {
        let text = """
            {"id": "t1", "transcript": \(transcriptJSON), "options": ["yes", "no"]}
            {"id": "t2", "transcript": \(transcriptJSON)}
            {"id": "p1", "text": "Plain item.", "target": "x", "options": ["a", "b"]}
            """
        guard case .preview(let preview) = TaskPromptsImport.preview(text) else {
            Issue.record("expected a preview")
            return
        }
        #expect(preview.recordCount == 3)
        #expect(preview.transcriptCount == 2)
        #expect(preview.optionsCount == 2)
        #expect(preview.summaryLine.contains("2 with scripted transcripts"))
    }

    @Test func importPreviewRefusesSchemaViolationsLineAccurately() {
        let text = """
            {"id": "ok", "text": "fine"}
            {"id": "bad", "transcript": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
            """
        guard case .failure(let line, let message) = TaskPromptsImport.preview(text)
        else {
            Issue.record("expected a line-accurate failure")
            return
        }
        #expect(line == 2)
        #expect(message.contains("item 'bad'"))
        #expect(message.contains("assistant-prefix continuation"))
    }

    @Test func importPreviewAcceptsTranscriptOnlyRecords() {
        let text = #"{"id": "t1", "transcript": "# + transcriptJSON + "}"
        guard case .preview(let preview) = TaskPromptsImport.preview(text) else {
            Issue.record("transcript-only records must be importable")
            return
        }
        #expect(preview.recordCount == 1)
        #expect(preview.transcriptCount == 1)
        #expect(TaskPromptsImport.looksLikeJSONL(text))
    }

    // MARK: - 8. Readiness (data check): the task-prompts row

    private func withTempRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "transcript-readiness-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        return try body(root)
    }

    private func writePrompts(_ jsonl: String, to root: URL) throws -> String {
        let path = "prompts/tasks/meta-prompts.jsonl"
        let url = root.appending(path: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try jsonl.write(to: url, atomically: true, encoding: .utf8)
        return path
    }

    @Test func readinessDetailNamesFamilyIncompatibleItems() throws {
        try withTempRoot { root in
            var m = manifest(modelID: "mlx-community/gemma-3-12b-it-8bit")
            m.taskPromptsFile = try writePrompts(
                """
                {"id": "g1", "transcript": [{"role": "assistant", "content": "Seeded."}, {"role": "user", "content": "Q?"}]}
                """ + "\n", to: root)
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let row = try #require(rows.first { $0.id == "taskPrompts" })
            // Run-start refusals BLOCK (`.invalid`) — preflight matches
            // execution, never a "degraded run" promise.
            #expect(row.status == .invalid)
            #expect(row.detail.contains("item 'g1'"))
            #expect(row.detail.contains("start with a user turn"))
        }
    }

    @Test func readinessSurfacesSchemaViolationsAsInvalidBlockers() throws {
        try withTempRoot { root in
            var m = manifest()
            m.taskPromptsFile = try writePrompts(
                """
                {"id": "bad", "transcript": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]}
                """ + "\n", to: root)
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let row = try #require(rows.first { $0.id == "taskPrompts" })
            #expect(row.status == .invalid)
            #expect(row.detail.contains("item 'bad'"))
            #expect(row.detail.contains("assistant-prefix continuation"))
        }
    }

    @Test func readinessValidTranscriptsArePresentWithNote() throws {
        try withTempRoot { root in
            var m = manifest(modelID: "mlx-community/gemma-3-12b-it-8bit")
            m.taskPromptsFile = try writePrompts(
                "{\"id\": \"t1\", \"transcript\": " + transcriptJSON + "}\n",
                to: root)
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let row = try #require(rows.first { $0.id == "taskPrompts" })
            #expect(row.status == .present)
            #expect(row.detail.contains("1 with scripted transcripts"))
        }
    }

    // MARK: - 9. The bundled template is valid on both families

    @Test func bundledTranscriptTemplateParsesAndPassesEveryGate() throws {
        let url = Self.repoRoot.appending(
            components: "prompts", "templates", "task-prompts-transcript",
            "task-prompts-transcript-template.jsonl")
        let data = try Data(contentsOf: url)
        let prompts = try ExperimentTasks.parseTaskPrompts(data)
        #expect(prompts.count == 2)
        #expect(prompts.allSatisfy { $0.transcript?.isEmpty == false })
        // Domain-neutral template must satisfy BOTH families' constraints.
        for modelID in ["Qwen/Qwen3-14B-MLX-8bit", "mlx-community/gemma-3-12b-it-8bit"] {
            try ExperimentTasks.checkTranscriptPrompts(
                prompts, manifest: manifest(modelID: modelID))
        }
        // First item carries the instrument fields.
        #expect(prompts[0].options == ["yes", "no"])
        #expect(prompts[0].target == "yes")
        // Registered as a one-click template.
        #expect(DataTemplates.template(id: "task-prompts-transcript") != nil)
        #expect(DataTemplates.all.contains { $0.id == "task-prompts-transcript" })
    }
}
