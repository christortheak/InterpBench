import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// "Build on server" and the two extraction declarations.
///
/// THE DEFECT (external review round 5, finding 2): the Concepts builder grew
/// a reading-position picker and an extraction-rendering picker, the LOCAL
/// build read both — and the SERVER build still posted `{method,
/// poolFromToken}`. Choosing `chatTemplate` + `last content token` and
/// clicking Build on server produced a RAW LAST-TOKEN vector while the panel
/// displayed different scientific settings. Nothing failed; the measurement
/// was simply not the one on screen.
///
/// Four properties hold the repair:
///
/// 1. **The whole declaration travels.** Every position outside the legacy
///    pooled pair, and every non-raw rendering, reaches the route as the
///    cross-engine label + the declaration object.
/// 2. **The legacy body is unchanged, byte for byte.** A default builder
///    posts exactly what it always posted, so a server that predates the
///    fields keeps building the recipes it can honor.
/// 3. **A declared axis is VERIFIED, not hoped for.** The routes echo what
///    they applied; a missing or disagreeing echo is a refusal, because an
///    old server would otherwise drop the declaration in silence — the exact
///    failure being fixed.
/// 4. **The engine asymmetry runs the right way.** `swift-mlx` cannot render
///    the assistant voice and the server can, so that refusal turns the LOCAL
///    build off and leaves the SERVER build on — its own repair text names
///    this server as the fix.
@Suite(.serialized) struct ServerExtractDeclarationClientTests {

    private static func client(
        handler: @escaping @Sendable (URLRequest) throws -> (Data, Int)
    ) -> ClusterClient {
        MockExtractURLProtocol.handler = handler
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockExtractURLProtocol.self]
        return ClusterClient(
            profile: ClusterConnectionProfile(
                baseURL: URL(string: "http://server.test")!),
            session: URLSession(configuration: configuration))
    }

    private static func body(from request: URLRequest) -> [String: Any]? {
        var data = request.httpBody
        if data == nil, let stream = request.httpBodyStream {
            stream.open()
            defer { stream.close() }
            var collected = Data()
            let size = 4096
            let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: size)
            defer { buffer.deallocate() }
            while stream.hasBytesAvailable {
                let count = stream.read(buffer, maxLength: size)
                if count <= 0 { break }
                collected.append(buffer, count: count)
            }
            data = collected
        }
        guard let data else { return nil }
        return try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    // MARK: - 1. the whole declaration travels

    @Test func conceptExtractPostsTheLabelAndTheRenderingObject() async throws {
        let client = Self.client { request in
            let object = try #require(Self.body(from: request))
            #expect(object["method"] as? String == "meanDifference")
            #expect(object["readingPosition"] as? String == "last content token")
            #expect(object["poolFromToken"] == nil)
            let rendering = try #require(
                object["extractionRendering"] as? [String: Any])
            #expect(rendering["mode"] as? String == "chatTemplate")
            return (Data("""
                {"jobId": "job-1",
                 "appliedExtraction": {"readingPosition": "last content token",
                                       "extractionRendering": {
                                           "mode": "chatTemplate",
                                           "addGenerationPrompt": true,
                                           "qwenThinkingEnabled": false}}}
                """.utf8), 200)
        }

        let jobID = try await client.conceptExtract(
            concept: "steadiness", method: "meanDifference",
            readingPosition: ReadingPosition.lastContentToken.label,
            extractionRendering: ExtractionRendering.chatTemplate())

        #expect(jobID == "job-1")
    }

    /// The assistant voice is the one this engine cannot render — the SERVER
    /// path must carry it anyway, because the server is where it works.
    @Test func conceptExtractCarriesTheAssistantVoiceAndVerifiesItsEcho() async throws {
        let client = Self.client { request in
            let object = try #require(Self.body(from: request))
            let rendering = try #require(
                object["extractionRendering"] as? [String: Any])
            #expect(rendering["voice"] as? String == "assistant")
            // The key both engines refuse under this voice is never sent.
            #expect(rendering["addGenerationPrompt"] == nil)
            return (Data("""
                {"jobId": "job-2",
                 "appliedExtraction": {"readingPosition": "last token",
                                       "extractionRendering": {
                                           "mode": "chatTemplate",
                                           "qwenThinkingEnabled": false,
                                           "voice": "assistant"}}}
                """.utf8), 200)
        }

        let jobID = try await client.conceptExtract(
            concept: "steadiness", method: "meanDifference",
            readingPosition: ReadingPosition.lastToken.label,
            extractionRendering: ExtractionRendering(
                mode: .chatTemplate, voice: .assistant))

        #expect(jobID == "job-2")
    }

    @Test func multiConceptExtractPostsTheSameDeclaration() async throws {
        let client = Self.client { request in
            #expect(request.url?.path == "/api/multiconcept/extract")
            let object = try #require(Self.body(from: request))
            #expect(object["readingPosition"] as? String
                == "mean content from token 4")
            #expect(object["poolFromToken"] == nil)
            return (Data("""
                {"jobId": "job-3",
                 "appliedExtraction": {
                     "readingPosition": "mean content from token 4",
                     "extractionRendering": {"mode": "chatTemplate",
                                             "addGenerationPrompt": true,
                                             "qwenThinkingEnabled": false}}}
                """.utf8), 200)
        }

        let jobID = try await client.multiConceptExtract(
            concepts: ["calm"], targets: ["calm"], poolFromToken: nil,
            readingPosition: ReadingPosition.meanContentFromToken(4).label,
            extractionRendering: ExtractionRendering.chatTemplate())

        #expect(jobID == "job-3")
    }

    // MARK: - 2. the legacy body is unchanged

    @Test func aLegacyCallPostsTheLegacyBodyAndNeedsNoEcho() async throws {
        let client = Self.client { request in
            let object = try #require(Self.body(from: request))
            #expect(object.keys.sorted() == ["method", "poolFromToken"])
            #expect(object["poolFromToken"] as? Int == 50)
            // Deliberately NO appliedExtraction: this is what an old server
            // answers, and a call that declared nothing new is content.
            return (Data(#"{"jobId": "job-legacy"}"#.utf8), 200)
        }

        let jobID = try await client.conceptExtract(
            concept: "steadiness", method: "lat", poolFromToken: 50)

        #expect(jobID == "job-legacy")
    }

    @Test func aDefaultCallPostsOnlyTheMethod() async throws {
        let client = Self.client { request in
            let object = try #require(Self.body(from: request))
            #expect(object.keys.sorted() == ["method"])
            return (Data(#"{"jobId": "job-bare"}"#.utf8), 200)
        }

        #expect(try await client.conceptExtract(
            concept: "steadiness", method: "meanDifference") == "job-bare")
    }

    // MARK: - 3. a declared axis is verified

    /// The version-skew case, exactly: an old server takes the fields,
    /// ignores them, and answers an ordinary job id. Silence is the bug, so
    /// the client refuses.
    @Test func aServerThatEchoesNothingIsRefusedNotTrusted() async throws {
        let client = Self.client { _ in
            (Data(#"{"jobId": "job-old"}"#.utf8), 200)
        }

        await #expect(throws: ClusterClient.DeclarationNotApplied.self) {
            _ = try await client.conceptExtract(
                concept: "steadiness", method: "meanDifference",
                readingPosition: ReadingPosition.turnCloseToken.label,
                extractionRendering: ExtractionRendering.chatTemplate())
        }
    }

    @Test func anEchoThatDisagreesIsRefused() async throws {
        let client = Self.client { _ in
            (Data("""
                {"jobId": "job-wrong",
                 "appliedExtraction": {"readingPosition": "last token",
                                       "extractionRendering": null}}
                """.utf8), 200)
        }

        await #expect(throws: ClusterClient.DeclarationNotApplied.self) {
            _ = try await client.conceptExtract(
                concept: "steadiness", method: "meanDifference",
                readingPosition: ReadingPosition.turnCloseToken.label)
        }
    }

    /// The position agrees and the RENDERING does not — the half a
    /// position-only check would wave through.
    @Test func anEchoThatDropsOnlyTheRenderingIsRefused() async throws {
        let client = Self.client { _ in
            (Data("""
                {"jobId": "job-halfway",
                 "appliedExtraction": {"readingPosition": "last token",
                                       "extractionRendering": null}}
                """.utf8), 200)
        }

        await #expect(throws: ClusterClient.DeclarationNotApplied.self) {
            _ = try await client.conceptExtract(
                concept: "steadiness", method: "meanDifference",
                readingPosition: ReadingPosition.lastToken.label,
                extractionRendering: ExtractionRendering.chatTemplate())
        }
    }

    // MARK: - 3b. a declared reference is verified the same way

    /// The designated-reference build's version-skew case: a server that
    /// predates the route's designated-reference support reads the PAIRED
    /// stimulus files, ignores `reference`, and answers an ordinary job id —
    /// a different recipe than the panel shows, in silence. The route echoes
    /// the pin it applied; this client refuses anything less.
    @Test func aDeclaredReferenceTravelsAndItsEchoIsVerified() async throws {
        let client = Self.client { request in
            let object = try #require(Self.body(from: request))
            #expect(object["method"] as? String == "designatedReference")
            #expect(object["reference"] as? String == "plain-exposition")
            #expect(object["poolFromToken"] as? Int == 50)
            return (Data("""
                {"jobId": "job-dr",
                 "designatedReference": {"name": "plain-exposition",
                                         "hash": "abc123"}}
                """.utf8), 200)
        }

        let jobID = try await client.conceptExtract(
            concept: "formality", method: "designatedReference",
            reference: "plain-exposition", poolFromToken: 50)

        #expect(jobID == "job-dr")
    }

    @Test func aServerThatEchoesNoReferenceIsRefusedNotTrusted() async throws {
        let client = Self.client { _ in
            (Data(#"{"jobId": "job-old-dr"}"#.utf8), 200)
        }

        do {
            _ = try await client.conceptExtract(
                concept: "formality", method: "designatedReference",
                reference: "plain-exposition", poolFromToken: 50)
            Issue.record("expected a refusal")
        } catch let error as ClusterClient.ReferenceNotApplied {
            #expect("\(error)".contains("predates designated-reference"))
            #expect("\(error)".contains("no legacy spelling"))
        }
    }

    @Test func aReferenceEchoThatDisagreesIsRefused() async throws {
        let client = Self.client { _ in
            (Data("""
                {"jobId": "job-wrong-dr",
                 "designatedReference": {"name": "neutral-stories",
                                         "hash": "def456"}}
                """.utf8), 200)
        }

        await #expect(throws: ClusterClient.ReferenceNotApplied.self) {
            _ = try await client.conceptExtract(
                concept: "formality", method: "designatedReference",
                reference: "plain-exposition", poolFromToken: 50)
        }
    }

    /// A call that declares no reference is the paired call it always was:
    /// no `reference` key in the body, no echo demanded.
    @Test func aReferencelessCallNeitherPostsNorDemandsTheEcho() async throws {
        let client = Self.client { request in
            let object = try #require(Self.body(from: request))
            #expect(object["reference"] == nil)
            return (Data(#"{"jobId": "job-plain"}"#.utf8), 200)
        }

        #expect(try await client.conceptExtract(
            concept: "steadiness", method: "meanDifference") == "job-plain")
    }

    /// Two spellings of one position are refused BEFORE the request — the
    /// engines' shared rule, in the engines' shared words.
    @Test func bothPositionSpellingsAreRefusedBeforeAnythingIsPosted() async throws {
        let client = Self.client { _ in
            Issue.record("the request must never be made")
            return (Data("{}".utf8), 200)
        }

        do {
            _ = try await client.conceptExtract(
                concept: "steadiness", method: "meanDifference",
                poolFromToken: 50,
                readingPosition: ReadingPosition.lastContentToken.label)
            Issue.record("expected a refusal")
        } catch let error as ClusterClient.ClientError {
            #expect("\(error)".contains("reading position declared twice"))
        }

        do {
            _ = try await client.multiConceptExtract(
                poolFromToken: 50,
                readingPosition: ReadingPosition.lastContentToken.label)
            Issue.record("expected a refusal")
        } catch let error as ClusterClient.ClientError {
            #expect("\(error)".contains("reading position declared twice"))
        }
    }
}

// MARK: - The builder's side of the declaration

@Suite(.serialized) @MainActor struct ServerExtractDeclarationBuilderTests {

    private func withBuilder<T>(_ body: (ConceptBuilder) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "server-extract") { root in
            let previous = WorkspaceRoot.programmaticOverride
            WorkspaceRoot.programmaticOverride = root
            defer { WorkspaceRoot.programmaticOverride = previous }
            return try body(ConceptBuilder())
        }
    }

    /// PROPERTY 2, builder side: an untouched builder declares the legacy
    /// spelling on the paired route (absent poolFromToken IS the last token
    /// there) and nothing else.
    @Test func theDefaultBuilderDeclaresTheLegacyBody() throws {
        try withBuilder { builder in
            let paired = builder.serverExtractionDeclaration(
                legacyPooledDefault: nil)
            #expect(paired.poolFromToken == nil)
            #expect(paired.readingPosition == nil)
            #expect(paired.extractionRendering == nil)
        }
    }

    /// …and the grand-mean family's pooled policy still travels as the legacy
    /// integer, because that is exactly what it means.
    @Test func theGrandMeanPooledPolicyStaysTheLegacyInteger() throws {
        try withBuilder { builder in
            builder.recipeFamily = .emotionGrandMean
            let grand = builder.serverExtractionDeclaration(
                legacyPooledDefault: 50)
            #expect(grand.poolFromToken == 50)
            #expect(grand.readingPosition == nil)
            #expect(grand.extractionRendering == nil)
        }
    }

    /// PROPERTY 1: the position the panel shows is the position the route
    /// gets, for every entry outside the legacy pair.
    @Test func everyOtherPositionTravelsAsItsLabel() throws {
        try withBuilder { builder in
            for (choice, parameter, label) in [
                (ReadingPositionChoice.lastContentToken, 0, "last content token"),
                (.turnCloseToken, 0, "turn close token"),
                (.offsetFromEnd, 3, "offset from end 3"),
                (.postInstruction, 2, "post-instruction 2"),
                (.contentOffset, 2, "content offset 2"),
                (.meanContentFromToken, 4, "mean content from token 4"),
            ] as [(ReadingPositionChoice, Int, String)] {
                builder.readingPositionParameter = parameter
                builder.readingPositionChoice = choice
                let declaration = builder.serverExtractionDeclaration(
                    legacyPooledDefault: nil)
                #expect(declaration.readingPosition == label)
                #expect(declaration.poolFromToken == nil)
            }
        }
    }

    /// `mean from token 0` has NO legacy spelling — a zero pool means
    /// something else on both routes — so it travels as its label.
    @Test func meanFromTokenZeroIsNotLegacyExpressible() throws {
        try withBuilder { builder in
            builder.readingPositionParameter = 0
            builder.readingPositionChoice = .meanFromToken
            let declaration = builder.serverExtractionDeclaration(
                legacyPooledDefault: nil)
            #expect(declaration.readingPosition == "mean from token 0")
            #expect(declaration.poolFromToken == nil)
        }
    }

    /// The last token has no legacy spelling on the grand-mean route either:
    /// an absent poolFromToken pools from 50 there.
    @Test func lastTokenTravelsAsItsLabelOnThePooledRoute() throws {
        try withBuilder { builder in
            builder.recipeFamily = .emotionGrandMean
            builder.readingPositionChoice = .lastToken
            let declaration = builder.serverExtractionDeclaration(
                legacyPooledDefault: 50)
            #expect(declaration.readingPosition == "last token")
            #expect(declaration.poolFromToken == nil)
        }
    }

    /// A declared rendering forces the label form even for a position the
    /// legacy field could carry: the two axes are one declaration, and the
    /// server must be told the whole of it or refuse.
    @Test func aDeclaredRenderingCarriesThePositionAsALabelToo() throws {
        try withBuilder { builder in
            builder.poolFromToken = 50
            builder.extractionRenderingChoice = ExtractionRenderingChoice(
                mode: .chatTemplate)
            let declaration = builder.serverExtractionDeclaration(
                legacyPooledDefault: nil)
            #expect(declaration.poolFromToken == nil)
            #expect(declaration.readingPosition == "mean from token 50")
            #expect(declaration.extractionRendering?.mode == .chatTemplate)
        }
    }

    // MARK: - 4. the engine asymmetry runs the right way

    /// THE SUBTLE ONE. `swift-mlx` refuses the assistant voice; the server
    /// renders it, and the refusal's own repair says so. The local build goes
    /// off, the SERVER build stays on, and the declaration that travels is
    /// the one on screen — not the nil this engine's parser produced.
    @Test func theAssistantVoiceBlocksTheLocalBuildAndNotTheServerOne() throws {
        try withBuilder { builder in
            builder.extractionRenderingChoice = ExtractionRenderingChoice(
                mode: .chatTemplate, voice: .assistant)

            #expect(builder.hasRefusedExtractionDeclaration)
            #expect(!builder.canSaveAndExtract)
            #expect(builder.extractionRenderingRefusalIsLocalEngineLimit)
            #expect(!builder.hasRefusedServerExtractionDeclaration)
            #expect(builder.serverExtractionDeclarationRefusal == nil)
            // The local parse is nil — that is the engine's limit, not the
            // declaration — and the SERVER declaration is the real one.
            #expect(builder.extractionRendering == nil)
            let rendering = try #require(builder.serverExtractionRendering)
            #expect(rendering.mode == .chatTemplate)
            #expect(rendering.isAssistantVoice)
        }
    }

    /// The same for the other asymmetry, whose repair text also names the
    /// server engine.
    @Test func addGenerationPromptFalseAlsoTravelsToTheServer() throws {
        try withBuilder { builder in
            builder.extractionRenderingChoice = ExtractionRenderingChoice(
                mode: .chatTemplate, addGenerationPrompt: false)

            #expect(builder.hasRefusedExtractionDeclaration)
            #expect(!builder.hasRefusedServerExtractionDeclaration)
            let rendering = try #require(builder.serverExtractionRendering)
            #expect(rendering.resolvedAddGenerationPrompt == false)
        }
    }

    /// An ENGINE-INDEPENDENT refusal stops both paths — nothing about it says
    /// the server would do better.
    @Test func anOutOfVocabularyPositionStopsTheServerBuildToo() throws {
        try withBuilder { builder in
            builder.readingPositionChoice = .postInstruction
            builder.readingPositionParameter = 9

            let refusal = try #require(builder.readingPositionRefusal)
            #expect(builder.hasRefusedExtractionDeclaration)
            #expect(builder.hasRefusedServerExtractionDeclaration)
            #expect(builder.serverExtractionDeclarationRefusal == refusal)
        }
    }

    /// A refused declaration never becomes a queued job: the server build
    /// path answers before it reaches a client at all (no host, no server
    /// workspace — the refusal is what runs first).
    @Test func aRefusedDeclarationNeverQueuesAServerJob() async throws {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "server-extract-refusal-\(UUID().uuidString)")
        ExperimentStore.rootOverride = temp
        let previous = WorkspaceRoot.programmaticOverride
        WorkspaceRoot.programmaticOverride = temp
        defer {
            WorkspaceRoot.programmaticOverride = previous
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        let builder = ConceptBuilder()
        builder.readingPositionChoice = .postInstruction
        builder.readingPositionParameter = 9
        let refusal = try #require(builder.readingPositionRefusal)

        await builder.buildVectorOnActiveServer()

        // The refusal, not "connect a server workspace first": it does not
        // depend on a connection, so it is answered first.
        #expect(builder.lastBuildError == refusal)
    }

    /// A server refusal reaches the panel in the SERVER's words, not wrapped
    /// in the transport's JSON.
    @Test func aServerRefusalIsUnwrappedForTheBuilderNotice() {
        let detail = "reading position 'middle token' is not one the "
            + "python-hf-transformers engine knows — repair: declare one of …"
        let error = ClusterClient.ClientError.badResponse(
            400, #"{"detail": "\#(detail)"}"#)
        #expect(ConceptBuilder.serverFailureText(error) == detail)

        let opaque = ClusterClient.ClientError.badResponse(500, "gateway boom")
        #expect(ConceptBuilder.serverFailureText(opaque)
            == "server returned 500: gateway boom")
    }
}

private final class MockExtractURLProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var handler:
        (@Sendable (URLRequest) throws -> (Data, Int))?

    override class func canInit(with request: URLRequest) -> Bool { true }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest {
        request
    }

    override func startLoading() {
        guard let handler = Self.handler else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }
        do {
            let (data, status) = try handler(request)
            let response = HTTPURLResponse(
                url: request.url!, statusCode: status, httpVersion: "HTTP/1.1",
                headerFields: ["Content-Type": "application/json"])!
            client?.urlProtocol(self, didReceive: response,
                                cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}
