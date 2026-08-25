import Foundation
import MLXLMCommon
import Testing

@testable import SteeringKit

/// Extraction RENDERING and the reading-position vocabulary (Swift twin of the
/// server's `test_extraction_rendering_and_positions.py`).
///
/// The bug this closes: extraction tokenized the raw stimulus while measured
/// generation rendered through the chat template, so which direction a study
/// got — and what the α denominator equalled — depended on an undeclared
/// implementation detail. These contracts pin the fix on this engine:
///
/// 1. **Absent is legacy raw** — a recipe that declares nothing encodes,
///    hashes, and stamps exactly as it always did.
/// 2. **The chat-template path reuses the MEASUREMENT renderer**
///    (`PromptRendering`), which `ExperimentTasks.userInput` also calls.
/// 3. **New positions resolve or refuse — never clamp**, and template-aware
///    roles refuse under raw rendering naming the dependency.
/// 4. **The stamp carries the request AND where it landed.**
/// 5. **A rendering form this engine cannot apply is a named refusal**, never
///    a silent fallback.
///
/// Pure CPU: a fake tokenizer, no model, no downloads.
@Suite struct ExtractionRenderingAndPositionsTests {

    // MARK: - fake tokenizer

    /// Ids shaped like a Gemma render: `<bos> … content … <end_of_turn> \n
    /// <start_of_turn> model \n`.
    static let bos = 2
    static let endOfTurn = 106
    static let newline = 107
    static let startOfTurn = 105
    static let model = 108

    struct FakeTokenizer: MLXLMCommon.Tokenizer {
        var chatTemplateFails = false

        var bosToken: String? { "<bos>" }
        var eosToken: String? { "<end_of_turn>" }
        var unknownToken: String? { "<unk>" }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            [ExtractionRenderingAndPositionsTests.bos] + text.utf8.map { Int($0) + 200 }
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map(String.init).joined(separator: " ")
        }

        func convertTokenToId(_ token: String) -> Int? {
            switch token {
            case "<bos>": ExtractionRenderingAndPositionsTests.bos
            case "<end_of_turn>": ExtractionRenderingAndPositionsTests.endOfTurn
            case "<start_of_turn>": ExtractionRenderingAndPositionsTests.startOfTurn
            case "<unk>": 3
            default: nil
            }
        }

        func convertIdToToken(_ id: Int) -> String? { "tok\(id)" }

        /// Stands in for the real template: the ids the generation-prompt
        /// render produces. `additionalContext` and the message dictionaries
        /// are recorded through `lastRender` so a test can assert the family
        /// rules reached it.
        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] {
            if chatTemplateFails {
                struct NoTemplate: Error, CustomStringConvertible {
                    var description: String { "this tokenizer has no chat template" }
                }
                throw NoTemplate()
            }
            let body = messages
                .compactMap { $0["content"] as? String }
                .joined(separator: "\u{01}")
            return [ExtractionRenderingAndPositionsTests.bos]
                + body.utf8.map { Int($0) + 200 }
                + [
                    ExtractionRenderingAndPositionsTests.endOfTurn,
                    ExtractionRenderingAndPositionsTests.newline,
                    ExtractionRenderingAndPositionsTests.startOfTurn,
                    ExtractionRenderingAndPositionsTests.model,
                    ExtractionRenderingAndPositionsTests.newline,
                ]
        }
    }

    /// A rendered Gemma-shaped sequence: content at 1...2, turn close at 3,
    /// then four scaffold tokens.
    static let renderedTokens = [
        bos, 201, 202, endOfTurn, newline, startOfTurn, model, newline,
    ]

    /// The other role word, for the turn-structured tokenizer below.
    static let userRole = 109

    /// A fake whose vocabulary PIECES are real, so the content mask can find
    /// the newline that ends a template's role tag. The flatter
    /// `FakeTokenizer` above is enough for roles that only need a turn CLOSE;
    /// the mask needs the turn's OPENING and its role tag to exist as tokens,
    /// because excluding them is the whole point.
    struct TurnFakeTokenizer: MLXLMCommon.Tokenizer {
        var bosToken: String? { "<bos>" }
        var eosToken: String? { "<end_of_turn>" }
        var unknownToken: String? { "<unk>" }

        func encode(text: String, addSpecialTokens: Bool) -> [Int] {
            [ExtractionRenderingAndPositionsTests.bos]
                + text.utf8.map { Int($0) + 200 }
        }

        func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
            tokenIds.map { convertIdToToken($0) ?? "" }.joined()
        }

        func convertTokenToId(_ token: String) -> Int? {
            switch token {
            case "<bos>": ExtractionRenderingAndPositionsTests.bos
            case "<end_of_turn>": ExtractionRenderingAndPositionsTests.endOfTurn
            case "<start_of_turn>": ExtractionRenderingAndPositionsTests.startOfTurn
            case "<unk>": 3
            default: nil
            }
        }

        func convertIdToToken(_ id: Int) -> String? {
            switch id {
            case ExtractionRenderingAndPositionsTests.bos: "<bos>"
            case ExtractionRenderingAndPositionsTests.startOfTurn: "<start_of_turn>"
            case ExtractionRenderingAndPositionsTests.endOfTurn: "<end_of_turn>"
            case ExtractionRenderingAndPositionsTests.newline: "\n"
            case ExtractionRenderingAndPositionsTests.model: "model"
            case ExtractionRenderingAndPositionsTests.userRole: "user"
            default: "tok\(id)"
            }
        }

        func applyChatTemplate(
            messages: [[String: any Sendable]],
            tools: [[String: any Sendable]]?,
            additionalContext: [String: any Sendable]?
        ) throws -> [Int] { [] }
    }

    /// A full Gemma-shaped render with the turn's OPENING present:
    /// `<bos><start_of_turn>user\n` · content 4…6 · `<end_of_turn>` at 7 ·
    /// `\n<start_of_turn>model\n`.
    static let turnTokens = [
        bos, startOfTurn, userRole, newline, 201, 202, 203, endOfTurn,
        newline, startOfTurn, model, newline,
    ]

    // MARK: - 1. absent is legacy raw

    @Test func anAbsentDeclarationIsRaw() {
        #expect(ExtractionRendering.raw.isRaw)
        #expect(ExtractionOptions().resolvedExtractionRendering.isRaw)
        #expect(VectorExtractionRecipe(
            name: "r", method: .caaMeanDifference, targetConcept: "c",
            datasets: []).resolvedExtractionRendering.isRaw)
    }

    /// Hash compatibility at the recipe level: a nil optional is OMITTED by
    /// Swift's synthesized encoder, so a recipe that declares no rendering
    /// encodes to the bytes it always did and keeps its `canonicalHash()`.
    @Test func anUndeclaredRenderingLeavesTheRecipeHashUntouched() throws {
        var recipe = VectorExtractionRecipe(
            name: "r", method: .caaMeanDifference, targetConcept: "c", datasets: [])
        let legacyHash = recipe.canonicalHash()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let legacyJSON = String(decoding: try encoder.encode(recipe), as: UTF8.self)
        #expect(!legacyJSON.contains("extractionRendering"))

        // Declaring raw explicitly is a different DOCUMENT (it says so out
        // loud) but must never be reached by accident…
        recipe.extractionRendering = .chatTemplate()
        #expect(recipe.canonicalHash() != legacyHash)
        recipe.extractionRendering = nil
        #expect(recipe.canonicalHash() == legacyHash)
    }

    /// Absent-not-null on the artifact: a raw extraction's sidecar is
    /// byte-identical to what this engine has always written.
    @Test func aRawSidecarStampsNeitherNewKey() throws {
        let sidecar = SteeringVectorSidecar(
            modelID: "m", concept: "c", stimulusSetHash: "h",
            vectors: ConceptVectors(perLayer: [[1, 0]]),
            options: ExtractionOptions(),
            residualNormPerLayer: [1], residualNormSource: "extraction-stimuli",
            residualNormConvention: ResidualNormConvention.current,
            residualNormRendering: ExtractionRendering.Mode.raw.rawValue,
            extractionRendering: .raw)
        #expect(sidecar.residualNormRendering == nil)
        #expect(sidecar.extractionRendering == nil)
        #expect(sidecar.readingPositionResolution == nil)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let json = String(decoding: try encoder.encode(sidecar), as: UTF8.self)
        #expect(!json.contains("extractionRendering"))
        #expect(!json.contains("residualNormRendering"))
        #expect(!json.contains("readingPositionResolution"))
    }

    @Test func aTemplatedSidecarStampsTheRenderingThatMadeTheVector() throws {
        let sidecar = SteeringVectorSidecar(
            modelID: "m", concept: "c", stimulusSetHash: "h",
            vectors: ConceptVectors(perLayer: [[1, 0]]),
            options: ExtractionOptions(
                readingPosition: .turnCloseToken,
                extractionRendering: .chatTemplate()),
            residualNormPerLayer: [1], residualNormSource: "neutral-corpus",
            residualNormConvention: ResidualNormConvention.current,
            residualNormRendering: ExtractionRendering.Mode.chatTemplate.rawValue)
        #expect(sidecar.residualNormRendering == "chatTemplate")
        #expect(sidecar.readingPosition == "turn close token")
        // Defaults stamped EXPLICITLY, so a reader never needs to know them.
        let stamped = try #require(sidecar.extractionRendering)
        #expect(stamped.addGenerationPrompt == true)
        #expect(stamped.qwenThinkingEnabled == false)
    }

    // MARK: - 2. the chat-template path reuses the measurement renderer

    /// Gemma 3 has no system role: the system text is folded into the user
    /// turn. This is the rule `ExperimentTasks.userInput` applies for MEASURED
    /// generation, and extraction now reaches it through the same function.
    @Test func gemmaFoldsTheSystemPromptIntoTheUserTurn() {
        let messages = PromptRendering.chatMessages(
            prompt: "the prompt", modelID: "google/gemma-3-4b-it",
            systemPrompt: "answer plainly")
        #expect(messages.count == 1)
        #expect(messages[0].role == .user)
        #expect(messages[0].content == "answer plainly\n\nthe prompt")
    }

    @Test func nonGemmaFamiliesGetARealSystemTurn() {
        let messages = PromptRendering.chatMessages(
            prompt: "the prompt", modelID: "Qwen/Qwen3-4B",
            systemPrompt: "answer plainly")
        #expect(messages.map(\.role) == [.system, .user])
        #expect(messages[1].content == "the prompt")
    }

    @Test func qwenThinkingTravelsAsTemplateContext() {
        let off = PromptRendering.qwenContext(
            modelID: "Qwen/Qwen3-4B", qwenThinkingEnabled: false)
        #expect(off?["enable_thinking"] as? Bool == false)
        #expect(PromptRendering.qwenContext(
            modelID: "google/gemma-3-4b-it", qwenThinkingEnabled: false) == nil)
    }

    @Test func theRawBranchIsTheHistoricalEncodeCall() throws {
        let tokenizer = FakeTokenizer()
        let ids = try PromptRendering.tokenIDs(
            tokenizer: tokenizer, modelID: "google/gemma-3-4b-it",
            text: "hi", rendering: .raw)
        #expect(ids == tokenizer.encode(text: "hi"))
    }

    @Test func theTemplateBranchAddsTheTemplatesOwnTokens() throws {
        let ids = try PromptRendering.tokenIDs(
            tokenizer: FakeTokenizer(), modelID: "google/gemma-3-4b-it",
            text: "hi", rendering: .chatTemplate())
        #expect(ids.count > FakeTokenizer().encode(text: "hi").count)
        #expect(ids.contains(Self.endOfTurn))
        #expect(ids.last == Self.newline)
    }

    // MARK: - 5. an unsupportable form is a NAMED refusal

    /// `addGenerationPrompt: false` is not reachable through MLXLMCommon's
    /// tokenizer bridge. It must refuse by name — never render WITH a
    /// generation prompt while stamping that the recipe asked for none.
    @Test func addGenerationPromptFalseRefusesNamingTheEngineAndTheRepair() {
        #expect(throws: PromptRendering.UnsupportedForm.self) {
            try PromptRendering.tokenIDs(
                tokenizer: FakeTokenizer(), modelID: "google/gemma-3-4b-it",
                text: "hi", rendering: .chatTemplate(addGenerationPrompt: false))
        }
        do {
            _ = try PromptRendering.tokenIDs(
                tokenizer: FakeTokenizer(), modelID: "google/gemma-3-4b-it",
                text: "hi", rendering: .chatTemplate(addGenerationPrompt: false))
            Issue.record("expected a refusal")
        } catch let error as PromptRendering.UnsupportedForm {
            #expect(error.reason.contains("python-hf-transformers"))
            #expect(error.reason.contains("Repair:"))
        } catch {
            Issue.record("wrong error type: \(error)")
        }
    }

    /// The extractor's wrapper names the engine and the model, so the refusal
    /// reads the same as every other typed refusal in the house style.
    @Test func theExtractorWrapsAnUnsupportedRenderingWithEngineAndRepair() {
        let error = ConceptExtractorError.renderingUnsupported(
            rendering: ExtractionRendering.chatTemplate().label,
            modelID: "google/gemma-3-4b-it",
            reason: "no chat template")
        #expect(error.description.contains("swift-mlx"))
        #expect(error.description.contains("google/gemma-3-4b-it"))
        #expect(error.description.contains("Repair:"))
    }

    // MARK: - 3./4. positions resolve or refuse, and the stamp records both

    @Test func offsetFromEndReadsTheIndexItNames() throws {
        let tokenizer = FakeTokenizer()
        let ids = Array(0 ..< 10)
        let resolved = try ReadingPosition.offsetFromEnd(3).resolve(
            tokens: ids, tokenizer: tokenizer, renderingIsRaw: true)
        #expect(resolved.startIndex == 6)
        #expect(resolved.endIndex == 7)
        #expect(resolved.offsetFromEnd == 3)
        #expect(!resolved.isPooled)

        // k = 0 is the last token, exactly.
        let zero = try ReadingPosition.offsetFromEnd(0).resolve(
            tokens: ids, tokenizer: tokenizer, renderingIsRaw: true)
        let last = try ReadingPosition.lastToken.resolve(
            tokens: ids, tokenizer: tokenizer, renderingIsRaw: true)
        #expect(zero.startIndex == last.startIndex)
    }

    @Test func offsetFromEndRefusesAShortSequenceInsteadOfClamping() {
        do {
            _ = try ReadingPosition.offsetFromEnd(7).resolve(
                tokens: [1, 2, 3], tokenizer: FakeTokenizer(), renderingIsRaw: true)
            Issue.record("expected a refusal")
        } catch let error as ReadingPositionError {
            #expect(error.reason.contains("too short"))
            #expect(error.reason.contains("needs 8"))
            #expect(error.reason.contains("never"))
            #expect(error.reason.contains("repair:"))
        } catch {
            Issue.record("wrong error type: \(error)")
        }
    }

    /// A raw stimulus has no turn-close token; saying so — and naming the
    /// dependency plus the escape hatch — is the repair.
    @Test func namedRolesRefuseUnderRawRenderingNamingTheDependency() {
        let roles: [ReadingPosition] = [
            .lastContentToken, .turnCloseToken, .postInstruction(1),
        ]
        for role in roles {
            do {
                _ = try role.resolve(
                    tokens: Self.renderedTokens, tokenizer: FakeTokenizer(),
                    renderingIsRaw: true)
                Issue.record("\(role.label) resolved under raw rendering")
            } catch let error as ReadingPositionError {
                #expect(error.reason.contains("templated rendering"))
                #expect(error.reason.contains("\"mode\": \"chatTemplate\""))
                #expect(error.reason.contains("offset from end k"))
            } catch {
                Issue.record("wrong error type: \(error)")
            }
        }
    }

    @Test func namedRolesResolveToTheIndicesTheTemplatePutsThemAt() throws {
        let tokenizer = FakeTokenizer()
        func index(_ position: ReadingPosition) throws -> Int {
            try position.resolve(
                tokens: Self.renderedTokens, tokenizer: tokenizer,
                renderingIsRaw: false
            ).startIndex
        }
        #expect(try index(.lastContentToken) == 2)
        #expect(try index(.turnCloseToken) == 3)
        // Arditi's convention: the i-th token AFTER the instruction content.
        #expect(try (1 ... 5).map { try index(.postInstruction($0)) } == [3, 4, 5, 6, 7])
        // The turn-close token IS where post-instruction 1 lands under a
        // generation prompt — a fact about the template, not a bug.
        #expect(try index(.turnCloseToken) == index(.postInstruction(1)))
    }

    /// `lastContentToken` anchors on the turn boundary, not on a special-token
    /// scan: on a Gemma render the sequence's own last token is a bare
    /// newline, which a special-token rule would wrongly accept as content.
    @Test func lastContentTokenIgnoresTheTrailingScaffoldNewline() throws {
        let resolved = try ReadingPosition.lastContentToken.resolve(
            tokens: Self.renderedTokens, tokenizer: FakeTokenizer(),
            renderingIsRaw: false)
        #expect(resolved.startIndex == 2)
        #expect(Self.renderedTokens[resolved.startIndex] == 202)
        #expect(resolved.source == "token before the last end-of-turn marker")
    }

    @Test func postInstructionRefusesWhenTheTemplateLeftNoRoom() {
        do {
            _ = try ReadingPosition.postInstruction(3).resolve(
                tokens: [Self.bos, 201, 202, Self.endOfTurn],
                tokenizer: FakeTokenizer(), renderingIsRaw: false)
            Issue.record("expected a refusal")
        } catch let error as ReadingPositionError {
            #expect(error.reason.contains("no token 3 after it"))
            #expect(error.reason.contains("addGenerationPrompt"))
        } catch {
            Issue.record("wrong error type: \(error)")
        }
    }

    @Test func postInstructionIsBoundedToTheDocumentedRange() {
        for i in [0, 6] {
            do {
                _ = try ReadingPosition.postInstruction(i).resolve(
                    tokens: Self.renderedTokens, tokenizer: FakeTokenizer(),
                    renderingIsRaw: false)
                Issue.record("postInstruction(\(i)) resolved")
            } catch let error as ReadingPositionError {
                #expect(error.reason.contains("i in 1...5"))
            } catch {
                Issue.record("wrong error type: \(error)")
            }
        }
    }

    @Test func aTurnCloseRoleRefusesWhenNoMarkerIsPresent() {
        do {
            _ = try ReadingPosition.turnCloseToken.resolve(
                tokens: [Self.bos, 201, 202], tokenizer: FakeTokenizer(),
                renderingIsRaw: false)
            Issue.record("expected a refusal")
        } catch let error as ReadingPositionError {
            #expect(error.reason.contains("no end-of-turn marker"))
        } catch {
            Issue.record("wrong error type: \(error)")
        }
    }

    /// The legacy pooled read must still produce exactly the window it always
    /// did, now that resolution runs through the same code path.
    @Test func theLegacyPooledReadResolvesToItsHistoricalWindow() throws {
        let tokenizer = FakeTokenizer()
        let ids = Array(0 ..< 10)
        let pooled = try ReadingPosition.meanFromToken(3).resolve(
            tokens: ids, tokenizer: tokenizer, renderingIsRaw: true)
        #expect(pooled.startIndex == 3)
        #expect(pooled.endIndex == 10)
        #expect(pooled.isPooled)
        #expect(pooled.offsetFromEnd == nil)

        let last = try ReadingPosition.lastToken.resolve(
            tokens: ids, tokenizer: tokenizer, renderingIsRaw: true)
        #expect(last.startIndex == 9)
        #expect(last.endIndex == 10)
    }

    // MARK: - the resolution stamp

    @Test func theStampCarriesTheRequestAndWhereItLanded() throws {
        let tokenizer = FakeTokenizer()
        let resolutions = try (0 ..< 4).map { _ in
            try ReadingPosition.postInstruction(2).resolve(
                tokens: Self.renderedTokens, tokenizer: tokenizer,
                renderingIsRaw: false)
        }
        let report = try #require(
            ReadingPositionResolutionReport.make(
                position: .postInstruction(2), rendering: .chatTemplate(),
                resolutions: resolutions))
        // BOTH halves, per the layerResolution precedent.
        #expect(report.requested == "post-instruction 2")
        #expect(report.mode == "postInstruction")
        #expect(report.parameter == 2)
        #expect(report.rendering == "chatTemplate")
        #expect(report.source == "last content token + 2")
        // One sequence shape: every stimulus read at the same place in its
        // template — which is what the offset-from-end grouping shows.
        #expect(report.shapes.count == 1)
        #expect(report.shapes[0].offsetFromEnd == 3)
        #expect(report.shapes[0].sequenceCount == 4)
        #expect(report.shapes[0].exampleIndex + 1 == report.shapes[0].exampleEndIndex)
    }

    @Test func theStampIsOmittedForTheLegacyPair() throws {
        let tokenizer = FakeTokenizer()
        for position in [ReadingPosition.lastToken, .meanFromToken(3)] {
            let resolved = try position.resolve(
                tokens: Array(0 ..< 10), tokenizer: tokenizer, renderingIsRaw: true)
            #expect(
                ReadingPositionResolutionReport.make(
                    position: position, rendering: .raw,
                    resolutions: [resolved]) == nil,
                "\(position.label) stamped a resolution it did not need")
        }
    }

    /// An explicit offset is the MECHANISM form — what lives three back is
    /// model- and rendering-specific — so the artifact records where it landed
    /// even under raw rendering.
    @Test func anExplicitOffsetIsStampedEvenUnderRawRendering() throws {
        let resolved = try ReadingPosition.offsetFromEnd(3).resolve(
            tokens: Array(0 ..< 10), tokenizer: FakeTokenizer(), renderingIsRaw: true)
        let report = try #require(
            ReadingPositionResolutionReport.make(
                position: .offsetFromEnd(3), rendering: .raw,
                resolutions: [resolved]))
        #expect(report.mode == "offsetFromEnd")
        #expect(report.rendering == "raw")
        #expect(report.shapes[0].offsetFromEnd == 3)
    }

    /// Two shapes are the loud signal: the stimuli did NOT all render the same
    /// way, so a reader sees it instead of averaging over it.
    @Test func aNonUniformRenderShowsUpAsTwoShapes() throws {
        let tokenizer = FakeTokenizer()
        let short = [Self.bos, 201, Self.endOfTurn, Self.newline]
        let resolutions = [
            try ReadingPosition.turnCloseToken.resolve(
                tokens: Self.renderedTokens, tokenizer: tokenizer,
                renderingIsRaw: false),
            try ReadingPosition.turnCloseToken.resolve(
                tokens: short, tokenizer: tokenizer, renderingIsRaw: false),
        ]
        let report = try #require(
            ReadingPositionResolutionReport.make(
                position: .turnCloseToken, rendering: .chatTemplate(),
                resolutions: resolutions))
        #expect(report.shapes.count == 2)
        #expect(report.shapes.map(\.offsetFromEnd) == [1, 4])
        #expect(report.parameter == nil)
    }

    // MARK: - Codable round trip (the cross-engine artifact contract)

    @Test func theRenderingBlockEncodesTheKeysTheServerReads() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let stamped = try #require(ExtractionRendering.chatTemplate().stamp)
        let json = String(decoding: try encoder.encode(stamped), as: UTF8.self)
        #expect(json == #"{"addGenerationPrompt":true,"mode":"chatTemplate","qwenThinkingEnabled":false}"#)

        let decoded = try JSONDecoder().decode(
            ExtractionRendering.self, from: Data(json.utf8))
        #expect(decoded == stamped)
    }

    /// The Swift Codable enum forms the server's manifest parser reads.
    @Test func everyPositionRoundTripsThroughItsSwiftCodableForm() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let positions: [ReadingPosition] = [
            .lastToken, .meanFromToken(50), .offsetFromEnd(3),
            .lastContentToken, .turnCloseToken, .postInstruction(2),
        ]
        for position in positions {
            let data = try encoder.encode(position)
            let decoded = try JSONDecoder().decode(ReadingPosition.self, from: data)
            #expect(decoded == position)
            // The case NAME is the wire key the server matches on.
            let json = String(decoding: data, as: UTF8.self)
            #expect(json.contains("\"\(position.identityMode)\""),
                    "\(position.label) encoded as \(json)")
        }
    }

    // MARK: - 6. THE WRITER: `--extraction-rendering`, parsed and refused here

    /// The option shipped with every consumer live and no writer at all. This
    /// section is the writer's contract: what the flag accepts, what it
    /// refuses, and — the load-bearing one — that declaring RAW writes
    /// exactly what declaring nothing writes.

    @Test func aRawDeclarationCanonicalizesToNothingAtAll() throws {
        // Every spelling of "the legacy rendering" resolves to nil, which is
        // what keeps an explicit declaration byte-identical to silence.
        for spelling in ["raw", "\"raw\"", "{\"mode\": \"raw\"}",
                         "  {\"mode\":\"raw\"}  "] {
            #expect(try ExtractionRendering.declared(spelling) == nil,
                    "spelling \(spelling) did not canonicalize to absent")
        }
    }

    @Test func aChatTemplateDeclarationComesBackWithItsDefaultsWrittenOut() throws {
        let bare = try #require(
            try ExtractionRendering.declared("{\"mode\": \"chatTemplate\"}"))
        #expect(bare.mode == .chatTemplate)
        // Resolved, not left to a reader to infer — the `stamp` discipline.
        #expect(bare.addGenerationPrompt == true)
        #expect(bare.qwenThinkingEnabled == false)
        #expect(bare.systemPrompt == nil)

        // The shell-friendly bare word is the same declaration.
        #expect(try ExtractionRendering.declared("chatTemplate") == bare)

        let full = try #require(
            try ExtractionRendering.declared(
                #"{"mode":"chatTemplate","qwenThinkingEnabled":true,"systemPrompt":"be brief"}"#))
        #expect(full.qwenThinkingEnabled == true)
        #expect(full.systemPrompt == "be brief")
    }

    /// Every parse refusal is typed and carries a repair — and NONE of them
    /// falls back to raw, which is the whole reason the option exists.
    @Test func everyMalformedDeclarationIsATypedRefusalWithARepair() {
        let bad = [
            "",                                     // no value
            "{\"mode\": \"chatTemplate\"",         // unterminated JSON
            "{\"mode\": \"templated\"}",           // out-of-vocabulary mode
            "someFutureForm",                       // …and its bare-word form
            "{\"mode\": \"raw\", \"systemPrompt\": \"x\"}",  // raw takes no parameters
            "{\"mode\": \"chatTemplate\", \"addGenerationPrompt\": 1}",
            "{\"mode\": \"chatTemplate\", \"qwenThinkingEnabled\": \"yes\"}",
            "{\"mode\": \"chatTemplate\", \"systemPrompt\": 7}",
        ]
        for declaration in bad {
            #expect(throws: ExtractionRendering.DeclarationError.self) {
                try ExtractionRendering.declared(declaration)
            }
            do {
                _ = try ExtractionRendering.declared(declaration)
                Issue.record("\(declaration) parsed instead of refusing")
            } catch let error as ExtractionRendering.DeclarationError {
                #expect(!error.reason.isEmpty)
                #expect(!error.repair.isEmpty, "\(declaration) refused with no repair")
            } catch {
                Issue.record("\(declaration) threw an untyped \(error)")
            }
        }
    }

    /// The unknown-mode refusal names THIS engine and the legal vocabulary —
    /// the same sentence shape the server twin prints.
    @Test func anUnknownModeNamesTheEngineAndTheVocabulary() {
        do {
            _ = try ExtractionRendering.declared("{\"mode\": \"templated\"}")
            Issue.record("an unknown mode parsed")
        } catch let error as ExtractionRendering.DeclarationError {
            #expect(error.reason.contains("templated"))
            #expect(error.reason.contains(ExtractionRendering.declarationEngine))
            #expect(error.repair.contains("raw"))
            #expect(error.repair.contains("chatTemplate"))
        } catch {
            Issue.record("untyped \(error)")
        }
    }

    /// THE ENGINE ASYMMETRY, MOVED FORWARD. `addGenerationPrompt: false` is a
    /// form this engine cannot render; it is refused where it is TYPED, with
    /// the identical sentence the extraction path would have thrown — so the
    /// declaration can never reach a frozen manifest and fail on a GPU.
    @Test func addGenerationPromptFalseIsRefusedAtDeclarationTime() {
        do {
            _ = try ExtractionRendering.declared(
                #"{"mode":"chatTemplate","addGenerationPrompt":false}"#)
            Issue.record("addGenerationPrompt false was accepted")
        } catch let error as ExtractionRendering.DeclarationError {
            #expect(error.reason == PromptRendering.addGenerationPromptFalseReason)
            #expect(error.repair.contains("python-hf-transformers"))
        } catch {
            Issue.record("untyped \(error)")
        }
        // …and the extraction path still throws the same sentence, so the two
        // refusals cannot drift into two explanations.
        #expect(PromptRendering.addGenerationPromptFalseReason
                .contains("addGenerationPrompt false is not available"))
    }

    /// The route's form: an already-decoded object takes the identical path.
    @Test func theObjectFormOfADeclarationSharesEveryRule() throws {
        #expect(try ExtractionRendering.declared(object: ["mode": "raw"]) == nil)
        let templated = try #require(
            try ExtractionRendering.declared(object: ["mode": "chatTemplate"]))
        #expect(templated.mode == .chatTemplate)
        #expect(throws: ExtractionRendering.DeclarationError.self) {
            try ExtractionRendering.declared(object: ["mode": "nope"])
        }
    }

    /// Cross-engine spelling parity, pinned as a literal on both sides: the
    /// flag, the mode words, and the parameter keys are ONE vocabulary.
    @Test func theDeclarationVocabularyIsTheCrossEngineSpelling() throws {
        #expect(ExtractionRendering.declarationFlag == "--extraction-rendering")
        #expect(ExtractionRendering.Mode.allCases.map(\.rawValue)
                == ["raw", "chatTemplate"])
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let declared = try #require(
            try ExtractionRendering.declared("chatTemplate"))
        #expect(String(decoding: try encoder.encode(declared), as: UTF8.self)
                == #"{"addGenerationPrompt":true,"mode":"chatTemplate","qwenThinkingEnabled":false}"#)
    }

    // MARK: - 7. THE VOICE, and what this engine can honor of it

    @Test func theVoiceVocabularyIsTheCrossEngineSpelling() {
        #expect(ExtractionRendering.Voice.allCases.map(\.rawValue)
                == ["user", "assistant"])
        #expect(ExtractionRendering.raw.resolvedVoice == .user)
    }

    /// THE VOICE'S HASH CONTRACT on this engine: absent ≡ explicit "user" ≡
    /// the exact bytes every chat-template recipe already encoded.
    @Test func anExplicitUserVoiceEncodesToTheBytesItAlwaysDid() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let absent = try #require(try ExtractionRendering.declared("chatTemplate"))
        let explicit = try #require(try ExtractionRendering.declared(
            #"{"mode":"chatTemplate","voice":"user"}"#))
        #expect(absent == explicit)
        #expect(explicit.voice == nil)
        #expect(String(decoding: try encoder.encode(explicit), as: UTF8.self)
                == #"{"addGenerationPrompt":true,"mode":"chatTemplate","qwenThinkingEnabled":false}"#)
    }

    /// The engine asymmetry, at DECLARATION time — the
    /// `addGenerationPrompt: false` precedent, one step further. It names the
    /// engine that can, and never falls back to the user voice, because the
    /// two voices are different directions.
    @Test func theAssistantVoiceIsRefusedNamingThePythonEngine() {
        do {
            _ = try ExtractionRendering.declared(
                #"{"mode":"chatTemplate","voice":"assistant"}"#)
            Issue.record("the assistant voice was accepted on swift-mlx")
        } catch let error as ExtractionRendering.DeclarationError {
            #expect(error.reason == PromptRendering.assistantVoiceReason)
            #expect(error.repair.contains("python-hf-transformers"))
        } catch {
            Issue.record("untyped \(error)")
        }
        #expect(PromptRendering.assistantVoiceReason.contains("swift-mlx"))
        #expect(PromptRendering.assistantVoiceReason.contains("different directions"))
    }

    /// The MEANINGLESS parameters are refused FIRST, with the server's twin
    /// texts: the declaration is malformed on both engines, so retyping is the
    /// first repair and the engine question only arises after it.
    @Test func theAssistantVoiceRefusesItsMeaninglessParametersFirst() {
        let cases = [
            (#"{"mode":"chatTemplate","voice":"assistant","addGenerationPrompt":true}"#,
             ExtractionRendering.assistantVoiceGenerationPromptReason),
            (#"{"mode":"chatTemplate","voice":"assistant","systemPrompt":"be brief"}"#,
             ExtractionRendering.assistantVoiceSystemPromptReason),
        ]
        for (declaration, reason) in cases {
            do {
                _ = try ExtractionRendering.declared(declaration)
                Issue.record("accepted \(declaration)")
            } catch let error as ExtractionRendering.DeclarationError {
                #expect(error.reason == reason, "\(declaration)")
            } catch {
                Issue.record("untyped \(error)")
            }
        }
    }

    @Test func anUnknownVoiceNamesTheEngineAndTheVocabulary() {
        do {
            _ = try ExtractionRendering.declared(
                #"{"mode":"chatTemplate","voice":"system"}"#)
            Issue.record("an unknown voice was accepted")
        } catch let error as ExtractionRendering.DeclarationError {
            #expect(error.reason.contains("system"))
            #expect(error.reason.contains(ExtractionRendering.declarationEngine))
            #expect(error.repair.contains("user") && error.repair.contains("assistant"))
        } catch {
            Issue.record("untyped \(error)")
        }
    }

    @Test func aRawRenderingStillTakesNoVoice() {
        #expect(throws: ExtractionRendering.DeclarationError.self) {
            try ExtractionRendering.declared(#"{"mode":"raw","voice":"assistant"}"#)
        }
    }

    /// This engine still UNDERSTANDS the voice everywhere but the renderer: a
    /// Mac that promotes a server-extracted artifact must decode its sidecar
    /// and stamp the voice back exactly as written.
    @Test func aVoicedSidecarDecodesAndStampsItsVoice() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let decoded = try JSONDecoder().decode(
            ExtractionRendering.self,
            from: Data(#"{"mode":"chatTemplate","qwenThinkingEnabled":false,"voice":"assistant"}"#.utf8))
        #expect(decoded.isAssistantVoice)
        let stamped = try #require(decoded.stamp)
        // addGenerationPrompt is NOT stamped under this voice — it is refused
        // at declaration, so an artifact may not claim it.
        #expect(String(decoding: try encoder.encode(stamped), as: UTF8.self)
                == #"{"mode":"chatTemplate","qwenThinkingEnabled":false,"voice":"assistant"}"#)
        #expect(decoded.label == "chatTemplate (voice=assistant)")
    }

    /// …and the render path refuses it too, so a hand-built rendering that
    /// never went through `declared` cannot slip past either.
    @Test func theRenderPathRefusesTheAssistantVoiceWithTheSameSentence() {
        let rendering = ExtractionRendering(mode: .chatTemplate, voice: .assistant)
        #expect(throws: PromptRendering.UnsupportedForm.self) {
            try PromptRendering.tokenIDs(
                tokenizer: FakeTokenizer(), modelID: "google/gemma-3-4b-it",
                text: "a stimulus", rendering: rendering)
        }
    }

    // MARK: - 8. contentOffset(k), and the content mask

    @Test func contentOffsetCountsBackFromTheContentBoundary() throws {
        let tokenizer = TurnFakeTokenizer()
        func resolve(_ position: ReadingPosition) throws -> ResolvedReadingPosition {
            try position.resolve(
                tokens: Self.turnTokens, tokenizer: tokenizer, renderingIsRaw: false)
        }
        #expect(try (0 ... 2).map { try resolve(.contentOffset($0)).startIndex }
                == [6, 5, 4])
        // k = 0 IS the last content token — same index, same value.
        #expect(try resolve(.contentOffset(0)).startIndex
                == resolve(.lastContentToken).startIndex)
        #expect(try resolve(.contentOffset(2)).source == "last content token − 2")
    }

    @Test func contentOffsetRefusesUnderflowInsteadOfClamping() {
        #expect(throws: ReadingPositionError.self) {
            try ReadingPosition.contentOffset(9).resolve(
                tokens: Self.turnTokens, tokenizer: TurnFakeTokenizer(),
                renderingIsRaw: false)
        }
        do {
            _ = try ReadingPosition.contentOffset(9).resolve(
                tokens: Self.turnTokens, tokenizer: TurnFakeTokenizer(),
                renderingIsRaw: false)
        } catch let error as ReadingPositionError {
            #expect(error.reason.contains("no token 9 before it"))
            #expect(error.reason.contains("never clamped"))
        } catch {
            Issue.record("untyped \(error)")
        }
    }

    @Test func contentOffsetRefusesUnderRawRenderingLikeEveryNamedRole() {
        #expect(ReadingPosition.contentOffset(1).requiresTemplatedRendering)
        do {
            _ = try ReadingPosition.contentOffset(1).resolve(
                tokens: Self.turnTokens, tokenizer: TurnFakeTokenizer(),
                renderingIsRaw: true)
            Issue.record("content offset resolved under raw rendering")
        } catch let error as ReadingPositionError {
            #expect(error.reason.contains("templated rendering"))
            #expect(error.reason.contains(ExtractionRendering.declarationFlag))
        } catch {
            Issue.record("untyped \(error)")
        }
    }

    @Test func contentOffsetRefusesANegativeK() {
        #expect(throws: ReadingPositionError.self) {
            try ReadingPosition.contentOffset(-1).resolve(
                tokens: Self.turnTokens, tokenizer: TurnFakeTokenizer(),
                renderingIsRaw: false)
        }
    }

    /// THE MASK, from ONE template map: the turn opens, its role tag is
    /// structure, it closes, and the trailing generation scaffold is past the
    /// content and therefore excluded by construction.
    @Test func theContentMaskKeepsOnlyTheStimulusOwnTokens() throws {
        let content = try ReadingPosition.contentIndices(
            tokens: Self.turnTokens, tokenizer: TurnFakeTokenizer(),
            label: "test")
        #expect(content == [4, 5, 6])
    }

    // MARK: - 9. meanContentFromToken(n)

    @Test func meanContentFromTokenPoolsContentAndCountsBothSides() throws {
        let resolved = try ReadingPosition.meanContentFromToken(1).resolve(
            tokens: Self.turnTokens, tokenizer: TurnFakeTokenizer(),
            renderingIsRaw: false)
        #expect(resolved.pooledIndices == [5, 6])
        #expect(resolved.startIndex == 5 && resolved.endIndex == 7)
        #expect(resolved.pooledTokenCount == 2)
        // 12 tokens, 3 of them content: the mask called nine of them structure.
        #expect(resolved.maskedTokenCount == 9)
        #expect(resolved.isPooled)
    }

    /// Under raw every token IS content, so the honest behavior is
    /// equivalence — the same window `meanFromToken(n)` reads, clamp included
    /// — not a refusal.
    @Test func meanContentFromTokenIsMeanFromTokenUnderRawRendering() throws {
        let tokens = Array(0 ..< 10)
        let masked = try ReadingPosition.meanContentFromToken(3).resolve(
            tokens: tokens, tokenizer: TurnFakeTokenizer(), renderingIsRaw: true)
        let plain = try ReadingPosition.meanFromToken(3).resolve(
            tokens: tokens, tokenizer: TurnFakeTokenizer(), renderingIsRaw: true)
        #expect(masked.startIndex == plain.startIndex)
        #expect(masked.endIndex == plain.endIndex)
        #expect(masked.pooledIndices == nil)      // a contiguous window
        #expect(masked.maskedTokenCount == 0)     // …masking nothing
        #expect(masked.source.contains("every token is content"))
        // …and the DECLARATION still differs, which is what the identity
        // records.
        #expect(masked.mode == "meanContentFromToken")
        #expect(!ReadingPosition.meanContentFromToken(3).requiresTemplatedRendering)
    }

    @Test func meanContentFromTokenRefusesWhenTheContentIsTooShort() {
        do {
            _ = try ReadingPosition.meanContentFromToken(5).resolve(
                tokens: Self.turnTokens, tokenizer: TurnFakeTokenizer(),
                renderingIsRaw: false)
            Issue.record("pooled from a content token that does not exist")
        } catch let error as ReadingPositionError {
            #expect(error.reason.contains("3 content tokens"))
            #expect(error.reason.contains("never clamped"))
        } catch {
            Issue.record("untyped \(error)")
        }
    }

    /// The stamp says how much was averaged and how much the mask called
    /// structure — and one uniform render is still ONE shapes row.
    @Test func theMaskedPoolStampCarriesBothCounts() throws {
        let tokenizer = TurnFakeTokenizer()
        let resolutions = try (0 ..< 3).map { _ in
            try ReadingPosition.meanContentFromToken(0).resolve(
                tokens: Self.turnTokens, tokenizer: tokenizer,
                renderingIsRaw: false)
        }
        let report = try #require(ReadingPositionResolutionReport.make(
            position: .meanContentFromToken(0),
            rendering: .chatTemplate(), resolutions: resolutions))
        #expect(report.mode == "meanContentFromToken")
        #expect(report.parameter == 0)
        #expect(report.shapes.count == 1)
        #expect(report.shapes[0].offsetFromEnd == nil)   // a pooled read
        #expect(report.shapes[0].examplePooledTokenCount == 3)
        #expect(report.shapes[0].exampleMaskedTokenCount == 9)
    }

    /// Every OTHER position's stamped bytes are untouched: the two counts are
    /// encoded only for a content-masked pool.
    @Test func theNewCountsAreAbsentFromEveryOtherStamp() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let resolutions = [try ReadingPosition.lastContentToken.resolve(
            tokens: Self.turnTokens, tokenizer: TurnFakeTokenizer(),
            renderingIsRaw: false)]
        let report = try #require(ReadingPositionResolutionReport.make(
            position: .lastContentToken, rendering: .chatTemplate(),
            resolutions: resolutions))
        let json = String(decoding: try encoder.encode(report), as: UTF8.self)
        #expect(!json.contains("examplePooledTokenCount"))
        #expect(!json.contains("exampleMaskedTokenCount"))
    }

    @Test func theNewPositionsRoundTripThroughLabelAndCodable() throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let positions: [ReadingPosition] = [
            .contentOffset(0), .contentOffset(4),
            .meanContentFromToken(0), .meanContentFromToken(12),
        ]
        for position in positions {
            #expect(ReadingPosition(label: position.label) == position)
            let data = try encoder.encode(position)
            #expect(try JSONDecoder().decode(ReadingPosition.self, from: data)
                    == position)
            #expect(String(decoding: data, as: UTF8.self)
                    .contains("\"\(position.identityMode)\""))
        }
        // …and the label of the position each is named after still parses to
        // that position, not to the new one.
        #expect(ReadingPosition(label: "mean from token 3") == .meanFromToken(3))
        #expect(ReadingPosition(label: "mean content from token 3")
                == .meanContentFromToken(3))
    }
}
