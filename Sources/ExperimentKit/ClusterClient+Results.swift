import Foundation

/// One top-level file of a SERVER run directory with its listed byte size
/// (`fileEntries` of `GET /api/runs`, `asdict(catalog.RunFileEntry)`). The
/// size is what lets the client size-gate previews BEFORE fetching bytes.
public struct RemoteRunFileEntry: Codable, Sendable, Identifiable, Equatable {
    public var name: String
    public var size: Int

    public var id: String { name }

    public init(name: String, size: Int) {
        self.name = name
        self.size = size
    }
}

/// The enriched `GET /api/runs` record for remote Results browsing: the
/// legacy listing fields plus the canonical per-run `config.json` stamps the
/// local Results browser shows (RunMetadata contract) and the name+size file
/// entries. Every enrichment is optional — an older server omits them and
/// the browser degrades exactly like a legacy local run (absence is shown,
/// never invented). Kept separate from `RemoteRunRecord` so existing
/// consumers of the lean decode are untouched.
public struct RemoteStampedRunRecord: Codable, Sendable, Identifiable, Equatable {
    public var id: String
    public var path: String
    public var task: String?
    public var hasReport: Bool
    public var hasGenerations: Bool
    public var hasCosineMatrix: Bool
    public var vectorNames: [String]
    public var files: [String]
    public var fileEntries: [RemoteRunFileEntry]?
    public var runType: String?
    public var createdAt: String?
    public var modelID: String?
    public var revision: String?
    public var experiment: String?
    public var substrate: String?
    public var appVersion: String?

    public init(
        id: String, path: String, task: String? = nil,
        hasReport: Bool = false, hasGenerations: Bool = false,
        hasCosineMatrix: Bool = false, vectorNames: [String] = [],
        files: [String] = [], fileEntries: [RemoteRunFileEntry]? = nil,
        runType: String? = nil, createdAt: String? = nil,
        modelID: String? = nil, revision: String? = nil,
        experiment: String? = nil, substrate: String? = nil,
        appVersion: String? = nil
    ) {
        self.id = id
        self.path = path
        self.task = task
        self.hasReport = hasReport
        self.hasGenerations = hasGenerations
        self.hasCosineMatrix = hasCosineMatrix
        self.vectorNames = vectorNames
        self.files = files
        self.fileEntries = fileEntries
        self.runType = runType
        self.createdAt = createdAt
        self.modelID = modelID
        self.revision = revision
        self.experiment = experiment
        self.substrate = substrate
        self.appVersion = appVersion
    }

    /// File entries for preview assembly: the listed name+size pairs, or —
    /// against an older server that doesn't send sizes — the plain name list
    /// with size 0. Size 0 therefore means "size UNKNOWN", and consumers
    /// must treat it as completeness-unproven (F5): fetches stay
    /// head-bounded and any table derived from a cap-filling response
    /// carries the truncation caption rather than posing as complete.
    /// (The head route's per-response metadata — `RemoteRunFileHead` —
    /// supersedes these listed sizes wherever it is present.)
    public var previewFileEntries: [RemoteRunFileEntry] {
        fileEntries ?? files.map { RemoteRunFileEntry(name: $0, size: 0) }
    }
}

/// One bounded run-file head plus the server's response metadata: the ACTUAL
/// on-disk file size (`X-SteerLab-File-Size`) and whether the returned bytes
/// are a truncated head of it (`X-SteerLab-Truncated`). Both are nil against
/// an older server that doesn't stamp the headers — consumers then fall back
/// to the listed-size heuristic (F5). When present, this metadata is
/// AUTHORITATIVE: it fixes the falsy-zero listing class (a listed size of 0
/// meaning "unknown") outright, because completeness comes from the server
/// that read the bytes, not from a possibly-absent listing field.
public struct RemoteRunFileHead: Sendable, Equatable {
    public var data: Data
    /// Server-reported actual file size in bytes; nil on older servers.
    public var fileSize: Int?
    /// Server-reported "this response is a truncated head" flag; nil on
    /// older servers (unknown, NOT false).
    public var truncated: Bool?

    public init(data: Data, fileSize: Int? = nil, truncated: Bool? = nil) {
        self.data = data
        self.fileSize = fileSize
        self.truncated = truncated
    }

    /// Decode the head-route metadata headers from a live response.
    /// Unparseable or absent headers degrade to nil, never to a guess.
    public init(data: Data, response: HTTPURLResponse) {
        self.data = data
        self.fileSize = response.value(forHTTPHeaderField: "X-SteerLab-File-Size")
            .flatMap(Int.init)
        switch response.value(forHTTPHeaderField: "X-SteerLab-Truncated")?
            .lowercased()
        {
        case "true": self.truncated = true
        case "false": self.truncated = false
        default: self.truncated = nil
        }
    }
}

/// Remote Results additions to the cluster client. `ClusterClient`'s request
/// plumbing is private to its own file, so this extension carries a minimal
/// GET path of its own, built the same way (`baseURL.appending(path:)` so a
/// reverse-proxy base prefix is preserved; bearer token; non-2xx →
/// `ClientError.badResponse`).
extension ClusterClient {

    /// The server's `runs/` listing with config.json stamps + file sizes
    /// (same route as `runs()`; richer decode for the Results browser).
    public func stampedRuns() async throws -> [RemoteStampedRunRecord] {
        struct Response: Decodable { var runs: [RemoteStampedRunRecord] }
        let request = resultsRequest(path: "/api/runs", queryItems: [])
        let (data, response) = try await session.data(for: request)
        try validateResults(response: response, data: data)
        return try JSONDecoder().decode(Response.self, from: data).runs
    }

    /// The first `maxBytes` bytes of one SERVER run file
    /// (`GET /api/runs/{run_id}/file?name=…&head=…` — contained server-side
    /// to the runs root exactly like `runFile`, and capped server-side at
    /// 8 MiB, a documented hard max ≥ every client-side head cap). The
    /// result carries the server's file-size + truncated metadata headers
    /// when present (older servers omit them → nil). Both path parts are
    /// percent-encoded, matching the existing `runFile` style.
    public func runFileHead(
        runID: String, name: String, maxBytes: Int
    ) async throws -> RemoteRunFileHead {
        let safeID = runID.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? runID
        let request = resultsRequest(
            path: "/api/runs/\(safeID)/file",
            queryItems: [
                URLQueryItem(name: "name", value: name),
                URLQueryItem(name: "head", value: String(maxBytes)),
            ])
        let (data, response) = try await session.data(for: request)
        try validateResults(response: response, data: data)
        guard let http = response as? HTTPURLResponse else {
            return RemoteRunFileHead(data: data)
        }
        return RemoteRunFileHead(data: data, response: http)
    }

    private func resultsRequest(path: String, queryItems: [URLQueryItem]) -> URLRequest {
        let relative = path.hasPrefix("/") ? String(path.dropFirst()) : path
        var url = profile.baseURL.appending(path: relative)
        if !queryItems.isEmpty {
            url.append(queryItems: queryItems)
        }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        if let token, !token.isEmpty {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        return request
    }

    private func validateResults(response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200 ..< 300).contains(http.statusCode) else {
            let text = String(data: data, encoding: .utf8)
                ?? HTTPURLResponse.localizedString(forStatusCode: http.statusCode)
            throw ClientError.badResponse(http.statusCode, text)
        }
    }
}
