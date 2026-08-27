import Foundation
import Network
import SteeringKit

/// Web front end for remote use (`steerlab-cli serve --port N`): a minimal
/// HTTP server wrapping the SAME ChatService/ConceptBuilder instances the
/// SwiftUI app renders, so the two front ends cannot diverge in behavior —
/// both are thin views over identical ExperimentKit state (CLAUDE.md ›
/// Conventions). The client (web/index.html) polls /api/state and posts
/// actions; that is the same render-observable-state model SwiftUI uses.
///
/// Deliberately dependency-free: Network.framework, HTTP/1.1,
/// connection-per-request. Single-researcher instrument, not a public
/// service — bind to localhost and reach it over an SSH tunnel.
///
/// The API is unauthenticated and can mutate experiment state, so it MUST NOT
/// be reachable off-box: the listener binds the loopback interface only
/// (`requiredLocalEndpoint`), and `route` additionally refuses any
/// browser-originated request whose Origin/Host isn't loopback, closing the
/// CSRF / DNS-rebinding path from a page open in the researcher's browser.
public final class SteerLabWebServer: Sendable {

    enum WebServerError: Error { case invalidPort(UInt16) }

    /// Host values that identify the loopback interface for Origin/Host checks.
    static let loopbackHosts: Set<String> = ["127.0.0.1", "localhost", "::1"]

    private let listener: NWListener
    private let queue = DispatchQueue(label: "steerlab.web")

    public init(port: UInt16) throws {
        guard let resolvedPort = NWEndpoint.Port(rawValue: port) else {
            throw WebServerError.invalidPort(port)
        }
        let parameters = NWParameters.tcp
        // Refuse every non-loopback interface: the unauthenticated API can
        // otherwise be driven by anyone on the same LAN or cluster node.
        parameters.requiredLocalEndpoint = NWEndpoint.hostPort(
            host: "127.0.0.1", port: resolvedPort)
        listener = try NWListener(using: parameters)
    }

    @MainActor
    public static func run(port: UInt16) async throws {
        let service = ChatService(cluster: ClusterConnectionStore())
        let server = try SteerLabWebServer(port: port)
        server.start(service: service)
        print("SteerLab web UI: http://localhost:\(port)  (Ctrl-C to stop)")
        // Keep the main actor free to process handler hops.
        while true {
            try await Task.sleep(for: .seconds(3600))
        }
    }

    /// ChatService is MainActor-isolated and non-Sendable; this box only
    /// ferries the reference through Network.framework's @Sendable callbacks
    /// to handlers that hop to the MainActor before touching it.
    private struct ServiceBox: @unchecked Sendable {
        let service: ChatService
    }

    public func start(service: ChatService) {
        let box = ServiceBox(service: service)
        listener.newConnectionHandler = { [queue] connection in
            connection.start(queue: queue)
            Self.receiveRequest(connection, buffer: Data(), box: box)
        }
        listener.start(queue: queue)
    }

    // MARK: - HTTP plumbing

    private static func receiveRequest(
        _ connection: NWConnection, buffer: Data, box: ServiceBox
    ) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) {
            data, _, isComplete, error in
            guard error == nil, let data, !data.isEmpty else {
                if isComplete { connection.cancel() }
                return
            }
            var buffer = buffer
            buffer.append(data)

            guard let headerEnd = buffer.range(of: Data("\r\n\r\n".utf8)) else {
                receiveRequest(connection, buffer: buffer, box: box)
                return
            }
            let headerText = String(decoding: buffer[..<headerEnd.lowerBound], as: UTF8.self)
            let headerLines = headerText.split(separator: "\r\n")
            func headerValue(_ name: String) -> String? {
                let prefix = name.lowercased() + ":"
                return headerLines.first { $0.lowercased().hasPrefix(prefix) }
                    .map { String($0.dropFirst(prefix.count)).trimmingCharacters(in: .whitespaces) }
            }
            let contentLength =
                headerValue("content-length").flatMap(Int.init) ?? 0
            let host = headerValue("host")
            let origin = headerValue("origin")
            // Browsers stamp Sec-Fetch-* / Origin on every request; curl, the
            // SwiftUI client, and SSH-tunnel users do not — so their absence
            // means "not a browser", and the loopback-origin guard is skipped.
            let isBrowserRequest = origin != nil || headerValue("sec-fetch-site") != nil
            // Data's indices are plain Ints over a zero-based buffer here.
            let bodyStart = headerEnd.upperBound
            guard buffer.count - (bodyStart - buffer.startIndex) >= contentLength else {
                receiveRequest(connection, buffer: buffer, box: box)
                return
            }

            let body = buffer.subdata(in: bodyStart ..< bodyStart + contentLength)
            let requestLine = headerText.split(separator: "\r\n").first.map(String.init) ?? ""
            let parts = requestLine.split(separator: " ")
            let method = parts.count > 0 ? String(parts[0]) : "GET"
            let path = parts.count > 1 ? String(parts[1]) : "/"

            Task { @MainActor in
                let response = await Self.route(
                    method: method, path: path, body: body,
                    host: host, origin: origin, isBrowserRequest: isBrowserRequest,
                    service: box.service)
                send(response, over: connection)
            }
        }
    }

    private struct Response {
        var status = "200 OK"
        var contentType = "application/json"
        var extraHeaders: [(String, String)] = []
        var body: Data

        static func json(_ object: some Encodable) -> Response {
            let encoder = JSONEncoder()
            let data = (try? encoder.encode(object)) ?? Data("{}".utf8)
            return Response(body: data)
        }

        static func ok() -> Response { Response(body: Data("{\"ok\":true}".utf8)) }

        static func error(_ message: String, status: String = "400 Bad Request") -> Response {
            let escaped = message
                .replacingOccurrences(of: "\\", with: "\\\\")
                .replacingOccurrences(of: "\"", with: "\\\"")
            return Response(status: status, body: Data("{\"error\":\"\(escaped)\"}".utf8))
        }
    }

    private static func send(_ response: Response, over connection: NWConnection) {
        var head = "HTTP/1.1 \(response.status)\r\n"
        head += "Content-Type: \(response.contentType); charset=utf-8\r\n"
        for (key, value) in response.extraHeaders {
            head += "\(key): \(value)\r\n"
        }
        head += "Content-Length: \(response.body.count)\r\n"
        head += "Cache-Control: no-store\r\n"
        head += "Connection: close\r\n\r\n"
        var payload = Data(head.utf8)
        payload.append(response.body)
        connection.send(
            content: payload,
            completion: .contentProcessed { _ in connection.cancel() })
    }

    private static func decode<T: Decodable>(_ type: T.Type, from body: Data) -> T? {
        try? JSONDecoder().decode(type, from: body)
    }

    private static func queryValue(_ name: String, in path: String) -> String? {
        guard let components = URLComponents(string: "http://localhost\(path)") else { return nil }
        return components.queryItems?.first { $0.name == name }?.value
    }

    // MARK: - Routes

    /// The exact CSRF decision `route` makes: refuse a browser-originated
    /// request that is not same-origin loopback. Non-browser clients (curl, the
    /// SwiftUI client, SSH-tunnel users) stamp no Origin/Sec-Fetch-* and are
    /// never refused. Factored out so the decision is testable without a live
    /// ChatService.
    static func isRequestRefused(isBrowserRequest: Bool, host: String?, origin: String?) -> Bool {
        isBrowserRequest && !isLoopbackContext(host: host, origin: origin)
    }

    /// True when the request's Origin (and, when present, Host) name the
    /// loopback interface. A `nil`/parse-failed Origin is treated as unsafe so
    /// that a cross-site request cannot slip through by omitting a parseable one.
    ///
    /// Same-origin means same port, not just same host: a hostile page served
    /// from another local port (another user's process on a shared node) is
    /// still cross-origin even though its host is loopback, so when both Origin
    /// and Host carry a port they must match.
    static func isLoopbackContext(host: String?, origin: String?) -> Bool {
        if let origin {
            // Foundation keeps IPv6 brackets on `.host` (e.g. "[::1]"); strip
            // them so the comparison matches the loopback set.
            guard let components = URLComponents(string: origin),
                let originHost = components.host,
                loopbackHosts.contains(
                    originHost.trimmingCharacters(in: CharacterSet(charactersIn: "[]")).lowercased())
            else { return false }
            if let host, let originPort = components.port {
                let hostPort = portOnly(host)
                if let hostPort, hostPort != originPort { return false }
            }
        }
        if let host {
            let bare = hostOnly(host)
            if !bare.isEmpty, !loopbackHosts.contains(bare) { return false }
        }
        return true
    }

    /// Strip a port (and IPv6 brackets) from a Host header value.
    static func hostOnly(_ value: String) -> String {
        let trimmed = value.trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("["), let end = trimmed.firstIndex(of: "]") {
            return String(trimmed[trimmed.index(after: trimmed.startIndex)..<end]).lowercased()
        }
        if trimmed.filter({ $0 == ":" }).count == 1, let colon = trimmed.firstIndex(of: ":") {
            return String(trimmed[..<colon]).lowercased()
        }
        return trimmed.lowercased()
    }

    /// Port from a Host header value, or nil when absent.
    static func portOnly(_ value: String) -> Int? {
        let trimmed = value.trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("["), let end = trimmed.firstIndex(of: "]") {
            let rest = trimmed[trimmed.index(after: end)...]
            return rest.hasPrefix(":") ? Int(rest.dropFirst()) : nil
        }
        if trimmed.filter({ $0 == ":" }).count == 1, let colon = trimmed.firstIndex(of: ":") {
            return Int(trimmed[trimmed.index(after: colon)...])
        }
        return nil
    }

    @MainActor
    private static func route(
        method: String, path: String, body: Data,
        host: String?, origin: String?, isBrowserRequest: Bool,
        service: ChatService
    ) async -> Response {
        // CSRF / DNS-rebinding defense: the listener is loopback-only, but a
        // page open in the researcher's browser can still POST to localhost.
        // Refuse any browser-originated request that isn't same-origin loopback.
        if isRequestRefused(isBrowserRequest: isBrowserRequest, host: host, origin: origin) {
            return .error("cross-origin request refused", status: "403 Forbidden")
        }
        let builder = service.concepts
        switch (method, path) {

        case ("GET", "/"), ("GET", "/index.html"):
            // The browser client is a code-shipped asset (the webAssets
            // family), not workspace data.
            guard let assets = try? CodeResources.webAssets(),
                let html = try? Data(
                    contentsOf: assets.appending(component: "index.html"))
            else {
                return .error("web/index.html not found", status: "404 Not Found")
            }
            return Response(contentType: "text/html", body: html)

        case ("GET", "/api/state"):
            return .json(StateDTO(service: service))

        case _ where method == "GET" && path.hasPrefix("/api/experiment/result/file"):
            guard let filename = queryValue("name", in: path) else {
                return .error("missing file name")
            }
            let allowed = Set(["generations.jsonl", "judgments.jsonl", "report.json", "judge-report.json"])
            guard allowed.contains(filename) else { return .error("file not downloadable") }
            guard let detail = service.experiments.selectedResult else {
                return .error("select a result artifact first")
            }
            let item = detail.item
            let judgeFiles = Set(["judgments.jsonl", "judge-report.json"])
            let basePath =
                judgeFiles.contains(filename)
                ? detail.judgeArtifactDirectory ?? item.path
                : item.path
            let url = URL(filePath: basePath).appending(component: filename)
            guard FileManager.default.fileExists(atPath: url.path),
                let data = try? Data(contentsOf: url)
            else {
                return .error("artifact file not found", status: "404 Not Found")
            }
            return Response(
                contentType: filename.hasSuffix(".jsonl") ? "application/x-ndjson" : "application/json",
                extraHeaders: [
                    (
                        "Content-Disposition",
                        "attachment; filename=\"\(item.directoryName)-\(filename)\""
                    )
                ],
                body: data)

        case ("POST", "/api/load"):
            struct Body: Decodable { let model: String }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            service.selectedModelID = request.model
            Task { await service.loadModel() }
            return .ok()

        case ("POST", "/api/chat"):
            struct Body: Decodable { let text: String }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            service.send(request.text)
            return .ok()

        case ("POST", "/api/chat/stop"):
            service.stopGeneration()
            return .ok()

        case ("POST", "/api/chat/reset"):
            service.resetChat()
            return .ok()

        case ("POST", "/api/steering"):
            struct Body: Decodable {
                var enabled: Bool?
                var bandWidth: Int?
                var normUnits: Bool?
                var removeNeutralDirections: Bool?
                var neutralCorpusID: String?
                var neutralPCBasisID: String?
                var temperature: Double?
                var qwenThinkingEnabled: Bool?
                var promptMode: ExperimentManifest.PromptMode?
                var systemPrompt: String?
            }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            if let value = request.enabled { service.steeringEnabled = value }
            if let value = request.bandWidth { service.layerBandWidth = value }
            if let value = request.normUnits { service.alphaInNormUnits = value }
            if let value = request.removeNeutralDirections {
                service.removeNeutralDirectionsAtSteering = value
            }
            if let value = request.neutralCorpusID {
                service.selectedNeutralCorpusID = value.isEmpty ? nil : value
            }
            if let value = request.neutralPCBasisID {
                service.selectedNeutralPCBasisID = value.isEmpty ? nil : value
            }
            if let value = request.temperature { service.temperature = value }
            if let value = request.qwenThinkingEnabled { service.qwenThinkingEnabled = value }
            if let value = request.promptMode { service.promptMode = value }
            if let value = request.systemPrompt { service.systemPrompt = value }
            return .ok()

        case ("POST", "/api/steering/slots"):
            struct SlotBody: Decodable {
                var id: String?
                var vectorID: String?
                var layer: Double
                var alpha: Double
                var enabled: Bool
            }
            struct Body: Decodable { let slots: [SlotBody] }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            let mapped = request.slots.map { slot in
                ChatService.SteerSlot(
                    id: slot.id.flatMap(UUID.init(uuidString:)) ?? UUID(),
                    vectorID: slot.vectorID?.isEmpty == true ? nil : slot.vectorID,
                    layer: slot.layer, alpha: slot.alpha, enabled: slot.enabled)
            }
            service.slots = mapped.isEmpty ? [ChatService.SteerSlot()] : mapped
            return .ok()

        case ("POST", "/api/steering/select-vector"):
            // Vector SELECTION is an engine decision, not a client one: the
            // slot's layer, α, and units come from what the artifact knows
            // about itself (`SlotAlphaDefault`), exactly as in the native
            // Playground. The response carries the resolved slot and the
            // decision so the client renders them without recomputing.
            struct Body: Decodable {
                var slotID: String?
                let vectorID: String
                var enabled: Bool?
            }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            let index =
                request.slotID
                .flatMap(UUID.init(uuidString:))
                .flatMap { id in service.slots.firstIndex(where: { $0.id == id }) } ?? 0
            guard service.slots.indices.contains(index) else {
                return .error("no steering slot")
            }
            service.slots[index].vectorID =
                request.vectorID.isEmpty ? nil : request.vectorID
            if let enabled = request.enabled { service.slots[index].enabled = enabled }
            service.applyDefaultsForSelectedVector(slotID: service.slots[index].id)
            let slot = service.slots[index]
            struct SelectVectorResponse: Encodable {
                let slot: StateDTO.SlotDTO
                let alphaDefault: StateDTO.AlphaDefaultDTO?
            }
            return .json(
                SelectVectorResponse(
                    slot: StateDTO.SlotDTO(
                        id: slot.id.uuidString, vectorID: slot.vectorID,
                        layer: slot.layer, alpha: slot.alpha, enabled: slot.enabled),
                    alphaDefault: service.alphaDefaultDecision.map(
                        StateDTO.AlphaDefaultDTO.init)))

        case ("POST", "/api/steering/slots/add"):
            // Same engine path as the native "Add vector" button: inherit the
            // previous box, default to the next unused vector of the active
            // catalog. No client-side α literal is involved.
            service.addSlot()
            return .ok()

        case ("POST", "/api/vectors/rescan"):
            service.refreshVectors()
            return .ok()

        case ("POST", "/api/neutral-pcs/build"):
            Task { @MainActor in
                await service.buildNeutralPCBasis()
            }
            return .ok()

        case ("POST", "/api/geometry/analyze"):
            struct Body: Decodable { let vectorIDs: [String] }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            let selected = service.compatibleVectors.filter { request.vectorIDs.contains($0.id) }
            guard selected.count == request.vectorIDs.count else {
                return .error("one or more selected vectors are not compatible with the loaded model")
            }
            do {
                let result = try GeometryAnalysis.analyze(artifacts: selected)
                return .json(GeometryResultDTO(result: result))
            } catch {
                return .error("\(error)")
            }

        case ("POST", "/api/gemmascope/run"):
            struct Body: Decodable {
                let vectorID: String
                let layer: Int
            }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            guard
                let artifact = service.compatibleVectors.first(where: { $0.id == request.vectorID })
            else {
                return .error("selected vector not found")
            }
            guard
                let info = GemmaScopeCatalog.info(
                    for: service.loadedModelID ?? service.selectedModelID,
                    layerCount: artifact.sidecar.layerCount,
                    preferredLayer: request.layer)
            else {
                return .error("load a Gemma 3 model before preparing Gemma Scope analysis")
            }
            do {
                return .json(
                    try await GemmaScopeAnalysis.run(
                        artifact: artifact, layer: info.recommendedLayer, info: info))
            } catch {
                return .error("\(error)")
            }

        case ("POST", "/api/gemmascope/import"):
            struct Body: Decodable {
                let reportID: String
                let feature: Int
                let source: String
            }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            guard
                let report = GemmaScopeReportCatalog.scan().first(where: { $0.id == request.reportID })
            else {
                return .error("Gemma Scope report not found")
            }
            guard
                let row = gemmaScopeRows(report: report, source: request.source).first(where: {
                    $0.feature == request.feature
                })
            else {
                return .error("feature not found in report")
            }
            do {
                let artifact = try GemmaScopeReportCatalog.importFeature(
                    report: report, row: row, source: request.source)
                service.refreshVectors()
                return .json(
                    GemmaScopeImportResultDTO(
                        vectorID: artifact.id,
                        label: artifact.label,
                        summary: "Imported feature \(row.feature) as \(artifact.sidecar.concept)"))
            } catch {
                return .error("\(error)")
            }

        case ("POST", "/api/concept/select"):
            struct Body: Decodable { let name: String? }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            builder.selectedExisting = request.name
            return .ok()

        case ("POST", "/api/concept/new"):
            builder.startNewConcept()
            return .ok()

        case ("POST", "/api/concept/delete"):
            builder.deleteSelectedConcept()
            return .ok()

        case ("POST", "/api/concept/name"):
            struct Body: Decodable { let name: String }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            builder.conceptName = request.name
            return .ok()

        case ("POST", "/api/concept/options"):
            struct Body: Decodable {
                var recipeFamily: String?
                var method: String?
                var poolFromToken: Int?
                var pooling: Bool?
                var buildAllGrandMeanConcepts: Bool?
                var designatedReference: String?
                var projectionNeutralCorpusName: String?
                var projectionNeutralConcepts: String?
                var projectionNeutralDomains: String?
                var projectionNeutralExclusions: String?
            }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            if let raw = request.recipeFamily,
                let family = ConceptBuilder.RecipeFamily(rawValue: raw)
            {
                builder.recipeFamily = family
            }
            if let raw = request.method, let method = ExtractionMethod(rawValue: raw) {
                builder.extractionMethod = method
            }
            if let pooling = request.pooling {
                builder.poolFromToken = pooling ? (request.poolFromToken ?? 0) : nil
            } else if let k = request.poolFromToken {
                builder.poolFromToken = k
            }
            if let buildAll = request.buildAllGrandMeanConcepts {
                builder.buildAllGrandMeanConcepts = buildAll
            }
            if let reference = request.designatedReference {
                builder.designatedReferenceConcept = reference
            }
            if let value = request.projectionNeutralCorpusName {
                builder.projectionNeutralCorpusName = value
                service.selectProjectionNeutralCorpus(named: value)
            }
            if let value = request.projectionNeutralConcepts {
                builder.projectionNeutralConceptsDraft = value
            }
            if let value = request.projectionNeutralDomains {
                builder.projectionNeutralDomainsDraft = value
            }
            if let value = request.projectionNeutralExclusions {
                builder.projectionNeutralExclusionsDraft = value
            }
            return .ok()

        case ("POST", "/api/concept/add"):
            struct Body: Decodable {
                let positive: String?
                let negative: String?
                let multiConcept: String?
                let multiConceptStoryConcept: String?
                let multiConceptTopic: String?
                let multiConceptSplit: String?
            }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            builder.positiveDraft = request.positive ?? ""
            builder.negativeDraft = request.negative ?? ""
            builder.multiConceptDraft = request.multiConcept ?? ""
            builder.multiConceptStoryConceptDraft = request.multiConceptStoryConcept ?? ""
            builder.multiConceptTopicDraft = request.multiConceptTopic ?? ""
            builder.multiConceptSplitDraft = request.multiConceptSplit ?? "build"
            await builder.addDrafts()
            return .ok()

        case ("POST", "/api/concept/topics"):
            struct Body: Decodable { let topics: [String]? }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            if let topics = request.topics {
                builder.setIncludedEmotionTopics(Set(topics))
            } else {
                builder.includeAllEmotionTopics()
            }
            return .ok()

        case ("POST", "/api/concept/emotion-concepts"):
            struct Body: Decodable { let concepts: [String]? }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            if let concepts = request.concepts {
                builder.setIncludedEmotionConcepts(Set(concepts))
            } else {
                builder.includeAllEmotionConcepts()
            }
            return .ok()

        case ("POST", "/api/concept/import"):
            struct Body: Decodable { let content: String; let filename: String }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            do {
                let pairs = try ConceptBuilder.parsePairs(
                    Data(request.content.utf8), filename: request.filename)
                builder.addPairs(pairs, source: request.filename)
                return .ok()
            } catch {
                return .error("import failed: \(error)")
            }

        case ("POST", "/api/concept/probe-import"):
            struct Body: Decodable { let content: String }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            builder.probeDraft = request.content
            await builder.addProbeDrafts()
            return .ok()

        case ("POST", "/api/concept/probe-train"):
            Task { await builder.trainReadingProbe() }
            return .ok()

        case ("POST", "/api/concept/remove"):
            struct Body: Decodable { let isPositive: Bool; let index: Int }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            builder.removeStimulus(isPositive: request.isPositive, index: request.index)
            return .ok()

        case ("POST", "/api/concept/rebuild"):
            Task { await builder.rebuild() }
            return .ok()

        case ("POST", "/api/concept/save"):
            Task { await builder.saveConceptAndExtract() }
            return .ok()

        case ("GET", "/api/concept/prompt"):
            guard let prompt = builder.generationPrompt() else {
                return .error(builder.status ?? "name the concept and add a pair first")
            }
            return Response(contentType: "text/plain", body: Data(prompt.utf8))

        case ("GET", "/api/concept/cowork-prompt"):
            guard let prompt = builder.coworkGenerationPrompt() else {
                return .error(builder.status ?? "select Grand mean and name the concept first")
            }
            return Response(contentType: "text/plain", body: Data(prompt.utf8))

        case ("GET", "/api/concept/probe-prompt"):
            guard let prompt = builder.probeGenerationPrompt() else {
                return .error(builder.status ?? "name the concept first")
            }
            return Response(contentType: "text/plain", body: Data(prompt.utf8))

        case ("GET", "/api/concept/neutral-prompt"):
            guard let prompt = builder.neutralCorpusPrompt() else {
                return .error(builder.status ?? "neutral corpus prompt template unavailable")
            }
            return Response(contentType: "text/plain", body: Data(prompt.utf8))

        case ("GET", "/api/concept/neutral-dialogue-prompt"):
            guard let prompt = builder.anthropicStyleNeutralDialoguePrompt() else {
                return .error(builder.status ?? "neutral dialogue prompt template unavailable")
            }
            return Response(contentType: "text/plain", body: Data(prompt.utf8))

        case ("POST", "/api/concept/neutral-import"):
            struct Body: Decodable { let content: String }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            builder.neutralCorpusDraft = request.content
            builder.importNeutralCorpusDraft()
            return .ok()

        case ("POST", "/api/concept/generate"):
            struct Body: Decodable { var count: Int?; var guidance: String? }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            if let count = request.count { builder.generationCount = count }
            if let guidance = request.guidance { builder.generationGuidance = guidance }
            Task { await builder.generateProposals() }
            return .ok()

        case ("POST", "/api/experiment/select"):
            struct Body: Decodable { let name: String? }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            service.experiments.selectedName = request.name
            return .ok()

        case ("POST", "/api/experiment/create"):
            struct Body: Decodable {
                let name: String
                let description: String?
            }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            service.experiments.newName = request.name
            service.experiments.newDescription = request.description ?? ""
            service.experiments.create()
            return .ok()

        case ("POST", "/api/experiment/protocol"):
            struct Body: Decodable {
                let description: String?
                let task: String?
                let outcomes: String?
                let judgeModel: String?
                let judgePrompt: String?
                let taskPromptsFile: String?
                let promptMode: ExperimentManifest.PromptMode?
                let systemPrompt: String?
                let qwenThinkingEnabled: Bool?
                let temperature: Double?
                let maxTokens: Int?
            }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            if let description = request.description {
                service.experiments.protocolDescription = description
            }
            if let task = request.task {
                service.experiments.taskDescription = task
            }
            if let outcomes = request.outcomes {
                service.experiments.outcomeMeasures = outcomes
            }
            if let judgeModel = request.judgeModel {
                service.experiments.judgeModel = judgeModel
            }
            if let judgePrompt = request.judgePrompt {
                service.experiments.evaluationPrompt = judgePrompt
            }
            if let taskPromptsFile = request.taskPromptsFile {
                service.experiments.taskPromptsFile = taskPromptsFile
            }
            if let promptMode = request.promptMode {
                service.experiments.promptMode = promptMode
            }
            if let systemPrompt = request.systemPrompt {
                service.experiments.systemPrompt = systemPrompt
            }
            if let qwenThinkingEnabled = request.qwenThinkingEnabled {
                service.experiments.qwenThinkingEnabled = qwenThinkingEnabled
            }
            if let temperature = request.temperature {
                service.experiments.runTemperature = temperature
            }
            if let maxTokens = request.maxTokens {
                service.experiments.runMaxTokens = maxTokens
            }
            // This request has no model field, so a headless protocol save is
            // never a base-model change: adopt the manifest's own model as
            // the panel's choice before delegating, or a stale panel field
            // would read as a model change and clear every variant condition
            // (open-issues §8, residual (b)).
            service.experiments.adoptSelectedManifestBaseModel()
            service.experiments.saveProtocol()
            return .ok()

        case ("POST", "/api/experiment/prompts/load"):
            service.experiments.loadTaskPrompts()
            return .ok()

        case ("POST", "/api/experiment/prompts/save"):
            struct Body: Decodable {
                let file: String?
                let text: String
            }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            if let file = request.file {
                service.experiments.taskPromptsFile = file
            }
            service.experiments.taskPromptsText = request.text
            service.experiments.saveTaskPrompts()
            return .ok()

        case ("POST", "/api/experiment/attach"):
            struct Body: Decodable { let concept: String }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            service.experiments.attachConcept(request.concept)
            return .ok()

        case ("POST", "/api/experiment/capture"):
            struct Body: Decodable { let name: String? }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            service.experiments.conditionName = request.name ?? ""
            service.experiments.captureCondition()
            return .ok()

        case ("POST", "/api/experiment/baseline"):
            struct Body: Decodable { let name: String? }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            service.experiments.conditionName = request.name ?? ""
            service.experiments.addBaselineCondition()
            return .ok()

        case ("POST", "/api/experiment/removeCondition"):
            struct Body: Decodable { let name: String }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            service.experiments.removeCondition(request.name)
            return .ok()

        case ("POST", "/api/experiment/freeze"):
            service.experiments.freeze()
            return .ok()

        case ("POST", "/api/experiment/duplicate"):
            service.experiments.duplicateSelected()
            return .ok()

        case ("POST", "/api/experiment/result/select"):
            struct Body: Decodable { let id: String? }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            service.experiments.selectedResultID = request.id
            return .ok()

        case ("POST", "/api/experiment/results/refresh"):
            service.experiments.refreshResults()
            return .ok()

        case ("POST", "/api/experiment/validate"):
            Task { @MainActor in await service.experiments.validateStudy() }
            return .ok()

        case ("POST", "/api/experiment/run"):
            Task { @MainActor in await service.experiments.runStudy() }
            return .ok()

        case ("POST", "/api/experiment/evaluate"):
            struct Body: Decodable {
                let judgeModel: String?
                let judgePrompt: String?
            }
            if let request = decode(Body.self, from: body) {
                if let judgeModel = request.judgeModel {
                    service.experiments.judgeModel = judgeModel
                }
                if let judgePrompt = request.judgePrompt {
                    service.experiments.evaluationPrompt = judgePrompt
                }
            }
            Task { @MainActor in await service.experiments.runPairedJudgeEvaluation() }
            return .ok()

        // RepE reader routes — minimal loopback mirrors of the Python server's
        // contract shapes (GET /api/readers, POST /api/reader/score) so
        // web/index.html can list and score local reader artifacts. Fitting is
        // deliberately NOT exposed here: the SwiftUI app and the server
        // workbench own the fit flow (the UI says so).
        case ("GET", "/api/readers"):
            return .json(
                ReaderListDTO(readers: VectorCatalog.scanReaders().map(ReaderListDTO.ReaderDTO.init)))

        case ("POST", "/api/reader/score"):
            struct Body: Decodable {
                let readerID: String
                let texts: [String]
            }
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            guard !request.readerID.isEmpty else { return .error("readerID is required") }
            guard !request.texts.isEmpty, request.texts.allSatisfy({ !$0.isEmpty }) else {
                return .error("texts must be a non-empty list of non-empty strings")
            }
            guard let container = service.containerForExtraction,
                let modelID = service.loadedModelID
            else {
                return .error("load a model first")
            }
            guard
                let record = VectorCatalog.scanReaders().first(where: { $0.id == request.readerID })
            else {
                return .error("no reader artifact with id '\(request.readerID)'")
            }
            do {
                // RepEReader.scoreTexts enforces the substrate/model guards
                // (a reader is a per-model, per-substrate instrument) and runs
                // the paper's exact inference — surfaced here as a 400.
                let scores = try await RepEReader.scoreTexts(
                    container: container, modelID: modelID,
                    reader: record.artifact, texts: request.texts)
                return .json(
                    ReaderScoreDTO(
                        readerID: request.readerID,
                        concept: record.artifact.concept,
                        layer: record.artifact.layer,
                        templateID: record.artifact.templateID,
                        scores: scores))
            } catch {
                return .error("\(error)")
            }

        case ("POST", "/api/concept/proposals"):
            struct Body: Decodable { let acceptedIDs: [String]? }  // nil = discard all
            guard let request = decode(Body.self, from: body) else { return .error("bad body") }
            if let ids = request.acceptedIDs {
                let accepted = Set(ids)
                for index in builder.proposals.indices {
                    builder.proposals[index].included =
                        accepted.contains(builder.proposals[index].id.uuidString)
                }
                await builder.acceptIncludedProposals()
            } else {
                builder.discardProposals()
            }
            return .ok()

        default:
            return .error("not found", status: "404 Not Found")
        }
    }

    private static func gemmaScopeRows(
        report: GemmaScopeReportArtifact, source: String
    ) -> [GemmaScopeFeatureRow] {
        switch source {
        case "top-negative": return report.report.topNegative
        case "top-absolute": return report.report.topAbsolute
        default: return report.report.topPositive
        }
    }
}

// MARK: - State snapshot DTOs

/// Mirrors the Python server's `GET /api/readers` shape (catalog
/// `ReaderSummary` field names) so the browser client renders both engines'
/// readers identically. Readers stay disjoint from the vector DTOs — they are
/// measurement instruments, not steering vectors.
struct ReaderListDTO: Encodable {
    struct ReaderDTO: Encodable {
        let id: String
        let label: String
        let runDirectory: String
        let name: String
        let concept: String
        let modelID: String
        let revision: String?
        let substrate: String
        let layer: Int
        let templateID: String
        let templateHash: String
        let templateDivergence: String?
        let datasetHash: String
        let latTokenPosition: String
        let trainAccuracy: Float
        let heldOutAccuracy: Float?
        /// WHAT the pair differences contrast, and HOW the sign was fixed —
        /// the two facts that distinguish one reader from another fitted on
        /// the same pairs. Legacy artifacts decode to the documented legacy
        /// values (`supervisedContent` / `trainMajority`), so the row always
        /// says what the artifact means rather than leaving it blank.
        let contrastMode: String
        let signConvention: String
        /// The set's argmax-held-out-accuracy layer — a RECOMMENDATION. Which
        /// layer a study reads is declared in its manifest; nothing here
        /// selects it. nil on a reader fitted before the stamp existed.
        let recommendedLayer: Int?
        let extracted: String

        init(record: VectorCatalog.ReaderArtifactRecord) {
            let artifact = record.artifact
            id = record.id
            label = record.label
            runDirectory = record.directory.path
            name = record.fileName.hasSuffix(".json")
                ? String(record.fileName.dropLast(".json".count)) : record.fileName
            concept = artifact.concept
            modelID = artifact.modelID
            revision = artifact.revision
            substrate = artifact.substrate
            layer = artifact.layer
            templateID = artifact.templateID
            templateHash = artifact.templateHash
            templateDivergence = artifact.template.divergence
            datasetHash = artifact.datasetHash
            latTokenPosition = artifact.latTokenPosition
            trainAccuracy = artifact.trainAccuracy
            heldOutAccuracy = artifact.heldOutAccuracy
            contrastMode = artifact.contrastMode.rawValue
            signConvention = artifact.signConvention.rawValue
            recommendedLayer = artifact.recommendedLayer
            extracted = artifact.extractionDate
        }
    }

    let readers: [ReaderDTO]
}

/// Mirrors the Python server's `POST /api/reader/score` response shape.
struct ReaderScoreDTO: Encodable {
    let readerID: String
    let concept: String
    let layer: Int
    let templateID: String
    let scores: [Float]
}

struct GemmaScopeImportResultDTO: Encodable {
    let vectorID: String
    let label: String
    let summary: String
}

struct GeometryResultDTO: Encodable {
    struct MatrixDTO: Encodable {
        let layer: Int
        let labels: [String]
        let values: [[Float]]
    }

    let artifactLabels: [String]
    let matrices: [MatrixDTO]
    let rsa: [[Float]]

    init(result: GeometryAnalysisResult) {
        artifactLabels = result.artifacts.map(\.label)
        matrices = result.matrices.map {
            MatrixDTO(layer: $0.layer, labels: $0.labels, values: $0.values)
        }
        rsa = result.rsa
    }
}

/// One JSON snapshot of everything both front ends render.
struct StateDTO: Encodable {
    struct MessageDTO: Encodable {
        let role: String
        let text: String
        /// Provenance: researcher-authored assistant turn (send-as-assistant)
        /// — nil for ordinary turns so older clients see no change. A seeded
        /// turn must never be indistinguishable from a real generation in
        /// any surface's record.
        let seeded: Bool?
        /// Provenance: generated, then altered by the researcher — nullable
        /// exactly like `seeded` (absent = false).
        let edited: Bool?
        /// First replaced version for revision inspection only. This field is
        /// never part of the model-facing chat message array.
        let originalText: String?
    }
    struct VectorDTO: Encodable {
        let id: String
        let label: String
        let stale: Bool
        let layerCount: Int
        let fixedLayer: Int?
        let concept: String
        let modelID: String
        let method: String?
        let reading: String?
        let extracted: String
        let stimulusHash: String
        let normsPerLayer: [Float]
        let hasResidualNorms: Bool
        let normSource: String?
    }
    struct GemmaScopeFeatureDTO: Encodable {
        let feature: Int
        let cosine: Float
        let sparsity: Float?
        let decoderAvailable: Bool
    }
    struct GemmaScopeReportDTO: Encodable {
        let id: String
        let label: String
        let path: String
        let concept: String
        let modelID: String
        let layer: Int
        let saeID: String
        let release: String
        let topPositive: [GemmaScopeFeatureDTO]
        let topNegative: [GemmaScopeFeatureDTO]
        let topAbsolute: [GemmaScopeFeatureDTO]
    }
    struct StimulusDTO: Encodable {
        let text: String
        let margin: Float?
    }
    struct ProposalDTO: Encodable {
        let id: String
        let positive: String
        let negative: String
        let included: Bool
    }
    struct StatsDTO: Encodable {
        let statsLayer: Int
        let heldOutAccuracy: Float?
        let heldOutCount: Int?
        let splitHalf: Float?
        let stability: Float?
        let normByLayer: [Float]
        let controlCosines: [String: Float]
    }
    struct SlotDTO: Encodable {
        let id: String
        let vectorID: String?
        let layer: Double
        let alpha: Double
        let enabled: Bool
    }
    /// The engine's α default for the most recent vector selection —
    /// `SlotAlphaDefault.Decision`, serialized verbatim. The client renders
    /// these strings; it never recomputes any part of the decision.
    struct AlphaDefaultDTO: Encodable {
        let alpha: Double
        /// "normUnits" | "raw"
        let units: String
        let label: String
        let rationale: String
        let conventionNote: String?
        let backfillHint: String?

        init(_ decision: SlotAlphaDefault.Decision) {
            alpha = decision.alpha
            units = decision.units.rawValue
            label = decision.alphaLabel
            rationale = decision.rationale
            conventionNote = decision.conventionNote
            backfillHint = decision.backfillHint
        }
    }
    struct NeutralPCBasisDTO: Encodable {
        let id: String
        let label: String
        let modelID: String
        let corpusHash: String
        let layers: Int
        let totalComponents: Int
        let tokenRows: Int
        let createdAt: String
    }
    struct NeutralCorpusDTO: Encodable {
        let id: String
        let label: String
        let kind: String
        let name: String
        let count: Int
        let hash: String?
    }
    struct ExperimentSummaryDTO: Encodable {
        let name: String
        let status: String
    }
    struct ExperimentDetailDTO: Encodable {
        struct ConceptRefDTO: Encodable {
            let name: String
            let hash: String
            let method: String
            let reading: String
            let neutralPCs: Int?
        }
        struct ConditionDTO: Encodable {
            let name: String
            let summary: String
        }
        let name: String
        let status: String
        let modelID: String
        let modelRevision: String?
        let description: String
        let taskDescription: String?
        let outcomeMeasures: String?
        let promptMode: String
        let promptModes: [PromptModeDTO]
        let systemPrompt: String
        let qwenThinkingEnabled: Bool
        let judgeModel: String
        let judgeModelOptions: [String]
        let judgePrompt: String
        let taskPromptsFile: String?
        let taskPromptsHash: String?
        let taskPromptsText: String
        let taskPromptsStatus: String?
        let temperature: Double
        let maxTokens: Int
        let isValidating: Bool
        let isRunning: Bool
        let isEvaluating: Bool
        let lastValidationDirectory: String?
        let lastRunDirectory: String?
        let lastEvaluationDirectory: String?
        let liveRunDirectory: String?
        let liveEvaluationDirectory: String?
        let liveActiveGeneration: LiveStudyGeneration?
        let liveActiveJudgment: LiveStudyJudgment?
        let liveGenerations: [StudyGenerationPreview]
        let liveJudgments: [StudyJudgePreview]
        let freezeHash: String?
        let gitCommit: String?
        let concepts: [ConceptRefDTO]
        let conditions: [ConditionDTO]
        let violations: [String]
        let attachable: [String]
        let resultRuns: [StudyRunListItem]
        let selectedResultID: String?
        let selectedResult: StudyRunDetail?
    }
    struct ConceptDTO: Encodable {
        let existing: [String]
        let selected: String?
        let name: String
        let recipeFamily: String
        let recipeFamilyLabel: String
        /// Family semantics the page must not re-derive: which dataset pane
        /// (paired vs story rows), and whether the grand-mean selectors
        /// apply. The client's fallback for an older DTO keeps the historic
        /// paired-unless-grand-mean reading.
        let recipeIsPaired: Bool
        let recipeUsesStoryCorpus: Bool
        /// designatedReference only: the reference-class selection, its
        /// candidates, and the standing refusal (a self-reference).
        let designatedReference: String
        let designatedReferenceOptions: [String]
        let designatedReferenceRefusal: String?
        let method: String
        let poolFromToken: Int?
        let positives: [StimulusDTO]
        let negatives: [StimulusDTO]
        let multiConceptRows: [MultiConceptRowDTO]
        let probeExamples: [ProbeExampleDTO]
        let probePositiveCount: Int
        let probeNegativeCount: Int
        let neutralCorpusCount: Int
        let neutralCorpusHash: String?
        let normNeutralCorpusCount: Int
        let normNeutralCorpusHash: String?
        let projectionNeutralCorpusName: String
        let projectionNeutralConcepts: String
        let projectionNeutralDomains: String
        let projectionNeutralExclusions: String
        let emotionConcepts: [String]
        let includedEmotionConcepts: [String]
        let buildAllGrandMeanConcepts: Bool
        let emotionTopics: [String]
        let includedEmotionTopics: [String]
        let emotionSummary: EmotionSummaryDTO
        let stats: StatsDTO?
        let statsStale: Bool
        let pendingPassCount: Int
        let unsavedChanges: Bool
        let canSaveAndExtract: Bool
        let isWorking: Bool
        let isGenerating: Bool
        let generationCount: Int
        let generationGuidance: String
        let proposals: [ProposalDTO]
        let status: String?
        let hasAPIKey: Bool
    }
    struct MultiConceptRowDTO: Encodable {
        let concept: String
        let topic: String?
        let text: String
        let split: String?
    }
    struct ProbeExampleDTO: Encodable {
        let id: String
        let text: String
        let expresses: Bool
        let topic: String?
        let split: String?
    }
    struct EmotionSummaryDTO: Encodable {
        let rowCount: Int
        let totalRowCount: Int
        let validationRowCount: Int
        let draftRowCount: Int
        let targetRowCount: Int
        let conceptCounts: [CountDTO]
        let topicCounts: [CountDTO]
        let cellCounts: [CellCountDTO]
        let missingCellCount: Int
        let minCellCount: Int
        let maxCellCount: Int
        let isBalanced: Bool
    }
    struct CountDTO: Encodable {
        let name: String
        let count: Int
    }
    struct CellCountDTO: Encodable {
        let topic: String
        let concept: String
        let count: Int
    }
    struct PromptModeDTO: Encodable {
        let value: String
        let label: String
    }

    let models: [String]
    let loadedModel: String?
    let selectedModel: String
    let modelState: String
    let isGenerating: Bool
    let errorMessage: String?
    let transcript: [MessageDTO]
    let steeringEnabled: Bool
    let bandWidth: Int
    let normUnits: Bool
    let normUnitsAvailable: Bool
    let removeNeutralDirections: Bool
    let neutralPCBasisID: String?
    let neutralPCBases: [NeutralPCBasisDTO]
    let neutralCorpusID: String?
    let neutralCorpora: [NeutralCorpusDTO]
    let neutralPCStatus: String?
    let isBuildingNeutralPCBasis: Bool
    let promptMode: String
    let promptModes: [PromptModeDTO]
    let systemPrompt: String
    let temperature: Double
    let qwenThinkingEnabled: Bool
    let qwenThinkingAvailable: Bool
    let activeSlotCount: Int
    let configuredSlotCount: Int
    let vectors: [VectorDTO]
    let allVectors: [VectorDTO]
    let gemmaScope: GemmaScopeInfo?
    let gemmaScopeReports: [GemmaScopeReportDTO]
    let slots: [SlotDTO]
    let alphaDefault: AlphaDefaultDTO?
    let experiments: [ExperimentSummaryDTO]
    let experiment: ExperimentDetailDTO?
    let experimentStatus: String?
    let concept: ConceptDTO

    @MainActor
    init(service: ChatService) {
        let builder = service.concepts
        models = ChatService.availableModels.map(\.id)
        loadedModel = service.loadedModelID
        selectedModel = service.selectedModelID
        modelState = {
            switch service.state {
            case .unloaded: return "unloaded"
            case .loading(let percent): return "loading \(percent)%"
            case .ready: return "ready"
            }
        }()
        isGenerating = service.isGenerating
        errorMessage = service.errorMessage
        transcript = service.transcript.map {
            MessageDTO(
                role: $0.role == .user ? "user" : "assistant",
                text: $0.text,
                seeded: $0.isSeededOrContinued ? true : nil,
                edited: $0.edited ? true : nil,
                originalText: $0.originalText)
        }
        steeringEnabled = service.steeringEnabled
        bandWidth = service.layerBandWidth
        normUnits = service.alphaInNormUnits
        normUnitsAvailable = service.normUnitsAvailable
        removeNeutralDirections = service.removeNeutralDirectionsAtSteering
        neutralPCBasisID = service.selectedNeutralPCBasisID
        neutralCorpusID = service.selectedNeutralCorpusID
        neutralCorpora = service.neutralCorpora.map { corpus in
            NeutralCorpusDTO(
                id: corpus.id,
                label: corpus.label,
                kind: corpus.kind.label,
                name: corpus.name,
                count: corpus.count,
                hash: corpus.hash.map { String($0.prefix(12)) })
        }
        neutralPCBases = service.compatibleNeutralPCBases.map { record in
            NeutralPCBasisDTO(
                id: record.id,
                label: record.label,
                modelID: record.basis.modelID,
                corpusHash: String(record.basis.corpusHash.prefix(12)),
                layers: record.basis.layers.count,
                totalComponents: record.basis.totalComponentCount,
                tokenRows: record.basis.tokenRowCount,
                createdAt: record.basis.createdAt)
        }
        neutralPCStatus = service.neutralPCStatus
        isBuildingNeutralPCBasis = service.isBuildingNeutralPCBasis
        promptMode = service.promptMode.rawValue
        promptModes = ExperimentManifest.PromptMode.allCases.map {
            PromptModeDTO(value: $0.rawValue, label: $0.label)
        }
        systemPrompt = service.systemPrompt
        temperature = service.temperature
        qwenThinkingEnabled = service.qwenThinkingEnabled
        qwenThinkingAvailable = service.selectedModelID.lowercased().contains("qwen")
        activeSlotCount = service.activeSlotCount
        configuredSlotCount = service.configuredSlotCount
        func vectorDTO(_ artifact: VectorArtifact) -> VectorDTO {
            VectorDTO(
                id: artifact.id, label: artifact.label,
                stale: service.staleVectorIDs.contains(artifact.id),
                layerCount: artifact.sidecar.layerCount,
                fixedLayer: artifact.fixedSteeringLayer,
                concept: artifact.sidecar.concept,
                modelID: artifact.sidecar.modelID,
                method: artifact.sidecar.extractionMethod,
                reading: artifact.sidecar.readingPosition,
                extracted: String(artifact.sidecar.extractionDate.prefix(10)),
                stimulusHash: String(artifact.sidecar.stimulusSetHash.prefix(12)),
                normsPerLayer: artifact.sidecar.normsPerLayer,
                hasResidualNorms: artifact.sidecar.residualNormPerLayer != nil,
                normSource: artifact.sidecar.residualNormSource)
        }
        vectors = service.compatibleVectors.map(vectorDTO)
        allVectors = service.vectors.map(vectorDTO)
        gemmaScope = GemmaScopeCatalog.info(
            for: service.loadedModelID ?? service.selectedModelID,
            layerCount: service.compatibleVectors.first?.sidecar.layerCount)
        gemmaScopeReports = GemmaScopeReportCatalog.scan().map { artifact in
            let report = artifact.report
            return GemmaScopeReportDTO(
                id: artifact.id,
                label: artifact.label,
                path: artifact.url.path,
                concept: report.vector.concept,
                modelID: report.vector.modelID,
                layer: report.vector.layer,
                saeID: report.gemmaScope.recommendedSAEID,
                release: report.gemmaScope.recommendedRelease,
                topPositive: report.topPositive.map(Self.featureDTO),
                topNegative: report.topNegative.map(Self.featureDTO),
                topAbsolute: report.topAbsolute.map(Self.featureDTO))
        }

        slots = service.slots.map {
            SlotDTO(
                id: $0.id.uuidString, vectorID: $0.vectorID,
                layer: $0.layer, alpha: $0.alpha, enabled: $0.enabled)
        }
        alphaDefault = service.alphaDefaultDecision.map(AlphaDefaultDTO.init)

        let panel = service.experiments
        experiments = panel.experiments.map {
            ExperimentSummaryDTO(name: $0.name, status: $0.status.rawValue)
        }
        experiment = panel.selected.map { manifest in
            ExperimentDetailDTO(
                name: manifest.name,
                status: manifest.status.rawValue,
                modelID: manifest.modelID,
                modelRevision: manifest.modelRevision.map { String($0.prefix(12)) },
                description: manifest.experimentDescription,
                taskDescription: manifest.taskDescription,
                outcomeMeasures: manifest.outcomeMeasures,
                promptMode: (manifest.promptMode ?? .chatAssistant).rawValue,
                promptModes: ExperimentManifest.PromptMode.allCases.map {
                    PromptModeDTO(value: $0.rawValue, label: $0.label)
                },
                systemPrompt: panel.systemPrompt,
                qwenThinkingEnabled: panel.qwenThinkingEnabled,
                judgeModel: panel.judgeModel,
                judgeModelOptions: panel.judgeModelOptions,
                judgePrompt: panel.evaluationPrompt,
                taskPromptsFile: manifest.taskPromptsFile,
                taskPromptsHash: manifest.taskPromptsHash.map { String($0.prefix(12)) },
                taskPromptsText: panel.taskPromptsText,
                taskPromptsStatus: panel.taskPromptsStatus,
                temperature: manifest.temperature,
                maxTokens: manifest.maxTokens,
                isValidating: panel.isValidating,
                isRunning: panel.isRunning,
                isEvaluating: panel.isEvaluating,
                lastValidationDirectory: panel.lastValidationDirectory,
                lastRunDirectory: panel.lastRunDirectory,
                lastEvaluationDirectory: panel.lastEvaluationDirectory,
                liveRunDirectory: panel.liveRunDirectory,
                liveEvaluationDirectory: panel.liveEvaluationDirectory,
                liveActiveGeneration: panel.liveActiveGeneration,
                liveActiveJudgment: panel.liveActiveJudgment,
                liveGenerations: panel.liveGenerations,
                liveJudgments: panel.liveJudgments,
                freezeHash: manifest.freezeHash.map { String($0.prefix(16)) },
                gitCommit: manifest.gitCommit.map { String($0.prefix(8)) },
                concepts: manifest.concepts.map {
                    ExperimentDetailDTO.ConceptRefDTO(
                        name: $0.name,
                        hash: String($0.stimulusSetHash.prefix(10)),
                        method: $0.options.method.label,
                        reading: $0.options.readingPosition.label,
                        neutralPCs: $0.options.neutralPCCount)
                },
                conditions: manifest.conditions.map { condition in
                    ExperimentDetailDTO.ConditionDTO(
                        name: condition.name,
                        summary: condition.slots.isEmpty
                            ? "no steering · baseline"
                            : condition.slots.map {
                                "\($0.concept) L\($0.layer) α\($0.alpha)"
                            }.joined(separator: " + ")
                                + (condition.alphaInNormUnits ? " (norm units)" : "")
                                + " · band \(condition.bandWidth)"
                                + (condition.neutralPCBasisLabel.map {
                                    " · neutral-removed: \($0)"
                                } ?? ""))
                },
                violations: panel.violations,
                attachable: panel.attachableConcepts,
                resultRuns: panel.resultRuns,
                selectedResultID: panel.selectedResultID,
                selectedResult: panel.selectedResult)
        }
        experimentStatus = panel.status

        let margins = builder.stats?.marginByStimulus ?? [:]
        let emotionSummary = builder.emotionCorpusSummary
        let neutralSummary = builder.neutralCorpusSummary
        let normNeutralSummary = builder.normNeutralCorpusSummary
        concept = ConceptDTO(
            existing: builder.existingConcepts,
            selected: builder.selectedExisting,
            name: builder.conceptName,
            recipeFamily: builder.recipeFamily.rawValue,
            recipeFamilyLabel: builder.recipeFamily.label,
            recipeIsPaired: builder.recipeFamily.isPaired,
            recipeUsesStoryCorpus: builder.recipeFamily.usesStoryCorpus,
            designatedReference: builder.designatedReferenceConcept,
            designatedReferenceOptions: builder.designatedReferenceOptions,
            designatedReferenceRefusal: builder.designatedReferenceRefusal,
            method: builder.extractionMethod.rawValue,
            poolFromToken: builder.poolFromToken,
            positives: builder.positives.enumerated().map { index, text in
                StimulusDTO(text: text, margin: margins["pos-\(index)"])
            },
            negatives: builder.negatives.enumerated().map { index, text in
                StimulusDTO(text: text, margin: margins["neg-\(index)"])
            },
            multiConceptRows: builder.multiConceptRows.map {
                MultiConceptRowDTO(
                    concept: $0.concept, topic: $0.topic, text: $0.text,
                    split: ConceptBuilder.canonicalSplit($0.split))
            },
            probeExamples: builder.probeExamples.map {
                ProbeExampleDTO(
                    id: $0.id, text: $0.text, expresses: $0.expresses,
                    topic: $0.topic, split: ConceptBuilder.canonicalSplit($0.split))
            },
            probePositiveCount: builder.probePositiveCount,
            probeNegativeCount: builder.probeNegativeCount,
            neutralCorpusCount: neutralSummary.count,
            neutralCorpusHash: neutralSummary.hash.map { String($0.prefix(12)) },
            normNeutralCorpusCount: normNeutralSummary.count,
            normNeutralCorpusHash: normNeutralSummary.hash.map { String($0.prefix(12)) },
            projectionNeutralCorpusName: builder.projectionNeutralCorpusName,
            projectionNeutralConcepts: builder.projectionNeutralConceptsDraft,
            projectionNeutralDomains: builder.projectionNeutralDomainsDraft,
            projectionNeutralExclusions: builder.projectionNeutralExclusionsDraft,
            emotionConcepts: builder.emotionConceptOptions,
            includedEmotionConcepts: builder.includedEmotionConcepts.sorted(),
            buildAllGrandMeanConcepts: builder.buildAllGrandMeanConcepts,
            emotionTopics: builder.emotionBuildTopics,
            includedEmotionTopics: builder.includedEmotionTopics.sorted(),
            emotionSummary: EmotionSummaryDTO(
                rowCount: emotionSummary.rowCount,
                totalRowCount: emotionSummary.totalRowCount,
                validationRowCount: emotionSummary.validationRowCount,
                draftRowCount: emotionSummary.draftRowCount,
                targetRowCount: emotionSummary.targetRowCount,
                conceptCounts: emotionSummary.conceptCounts.map {
                    CountDTO(name: $0.name, count: $0.count)
                },
                topicCounts: emotionSummary.topicCounts.map {
                    CountDTO(name: $0.name, count: $0.count)
                },
                cellCounts: emotionSummary.cellCounts.map {
                    CellCountDTO(topic: $0.topic, concept: $0.concept, count: $0.count)
                },
                missingCellCount: emotionSummary.missingCells.count,
                minCellCount: emotionSummary.minCellCount,
                maxCellCount: emotionSummary.maxCellCount,
                isBalanced: emotionSummary.isBalanced),
            stats: builder.stats.map { stats in
                StatsDTO(
                    statsLayer: stats.statsLayer,
                    heldOutAccuracy: stats.heldOut?.accuracy,
                    heldOutCount: stats.heldOut?.testCount,
                    splitHalf: stats.splitHalf,
                    stability: stats.stability,
                    normByLayer: stats.normByLayer,
                    controlCosines: Dictionary(
                        uniqueKeysWithValues: stats.controlCosines.map { ($0.name, $0.cosine) }))
            },
            statsStale: builder.statsStale,
            pendingPassCount: builder.pendingPassCount,
            unsavedChanges: builder.unsavedChanges,
            canSaveAndExtract: builder.canSaveAndExtract,
            isWorking: builder.isWorking,
            isGenerating: builder.isGeneratingProposals,
            generationCount: builder.generationCount,
            generationGuidance: builder.generationGuidance,
            proposals: builder.proposals.map {
                ProposalDTO(
                    id: $0.id.uuidString, positive: $0.positive,
                    negative: $0.negative, included: $0.included)
            },
            status: builder.status,
            hasAPIKey: ClaudeStimulusGenerator.apiKey != nil)
    }

    private static func featureDTO(_ row: GemmaScopeFeatureRow) -> GemmaScopeFeatureDTO {
        GemmaScopeFeatureDTO(
            feature: row.feature,
            cosine: row.cosine,
            sparsity: row.sparsity,
            decoderAvailable: row.decoderValues != nil)
    }
}
