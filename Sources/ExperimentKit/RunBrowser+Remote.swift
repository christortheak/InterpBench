import Foundation

/// One assembled remote file preview: the listed file entry plus the same
/// `FilePreview` structure the local detail pane renders.
public struct RemoteRunFilePreviewItem: Identifiable, Sendable, Equatable {
    public var file: RemoteRunFileEntry
    public var preview: RunBrowser.FilePreview

    public var id: String { file.name }

    public init(file: RemoteRunFileEntry, preview: RunBrowser.FilePreview) {
        self.file = file
        self.preview = preview
    }
}

/// Remote (server-workspace) preview assembly for the Results browser.
///
/// Binding design decision: the server does NOT duplicate preview logic — it
/// provides the listing (with per-file sizes) and BOUNDED file reads
/// (`head=` byte cap); the client parses fetched bytes with the SAME pure
/// parsers (`jsonKeyValues` / `csvTable` / `jsonlRecords`) and the SAME byte
/// caps the local browser uses. The only difference is transport: a fetcher
/// closure (backed by `ClusterClient.runFile` / `runFileHead`) instead of
/// the local filesystem.
extension RunBrowser {

    /// How (and whether) a remote file's bytes are fetched for a preview —
    /// decided from the LISTED name+size BEFORE any bytes move, mirroring
    /// `preview(for:)`'s dispatch. Every fetch is head-bounded: a listed
    /// size of 0 can mean "older server, size unknown" (F5), so the
    /// unbounded whole-file route is never used here — for a small JSON the
    /// bounded head IS the whole file.
    public enum RemoteFetchPlan: Sendable, Equatable {
        /// Fetch a bounded head. `parseLimit` is the local parser's byte cap;
        /// the fetch requests `parseLimit + 1` so truncation is detectable
        /// from the returned bytes alone (same trick as `readHead`).
        case head(parseLimit: Int)
        /// No bytes should move; render this `unavailable` reason instead.
        case none(reason: String)

        /// Bytes to request from the server for this plan (nil = no fetch).
        public var requestBytes: Int? {
            switch self {
            case .none: nil
            case .head(let parseLimit): parseLimit + 1
            }
        }
    }

    /// Fetch plan by extension + listed size — the remote twin of
    /// `preview(for:)`'s dispatch, with identical caps and identical
    /// unavailable reasons. An oversized JSON is refused HERE, before any
    /// fetch, exactly as the local browser refuses before any read.
    public static func remoteFetchPlan(name: String, size: Int) -> RemoteFetchPlan {
        switch (name as NSString).pathExtension.lowercased() {
        case "json":
            guard size <= jsonPreviewByteLimit else {
                return .none(reason: "JSON too large to preview")
            }
            return .head(parseLimit: jsonPreviewByteLimit)
        case "csv":
            return .head(parseLimit: csvHeadByteLimit)
        case "jsonl":
            return .head(parseLimit: jsonlHeadByteLimit)
        case "txt", "md", "log":
            return .head(parseLimit: textHeadByteLimit)
        default:
            return .none(reason: "no preview for this file type")
        }
    }

    /// Pure parse of remotely fetched bytes into the same `FilePreview`
    /// structures the local browser renders. `size` is the LISTED file size
    /// (truncation is flagged when either the listing or the fetched byte
    /// count says the file exceeds the parser's cap).
    public static func remotePreview(name: String, size: Int, data: Data) -> FilePreview {
        switch (name as NSString).pathExtension.lowercased() {
        case "json":
            guard size <= jsonPreviewByteLimit, data.count <= jsonPreviewByteLimit else {
                return .unavailable(reason: "JSON too large to preview")
            }
            guard let rows = jsonKeyValues(data) else {
                return .unavailable(reason: "unreadable JSON")
            }
            return .keyValues(rows)
        case "csv":
            let head = headText(data, maxBytes: csvHeadByteLimit)
            guard let table = csvTable(head.text) else {
                return .unavailable(reason: "unreadable CSV")
            }
            let listedOverflow = size > csvHeadByteLimit
            return .table(
                header: table.header, rows: table.rows,
                truncated: head.truncated || table.truncated || listedOverflow)
        case "jsonl":
            let head = headText(data, maxBytes: jsonlHeadByteLimit)
            let parsed = jsonlRecords(head.text)
            let listedOverflow = size > jsonlHeadByteLimit
            return .records(
                parsed.records,
                truncated: head.truncated || parsed.truncated || listedOverflow)
        case "txt", "md", "log":
            let head = headText(data, maxBytes: textHeadByteLimit)
            return .text(head.text, truncated: head.truncated || size > textHeadByteLimit)
        default:
            return .unavailable(reason: "no preview for this file type")
        }
    }

    /// F5's completeness rule for a head-bounded remote fetch. The server's
    /// head-route metadata (`X-SteerLab-File-Size` / `X-SteerLab-Truncated`,
    /// threaded here as `serverFileSize`/`serverTruncated`) is AUTHORITATIVE
    /// when present: the server read the bytes, so its verdict beats any
    /// listing — including the falsy-zero case where a listed size of 0
    /// means "unknown". Against an older server (both nil) the heuristic
    /// stands: completeness is PROVEN only when the listing accounts for
    /// every returned byte, or when the server returned fewer bytes than
    /// requested (EOF before the cap); an unknown (0/negative) listed size
    /// with a cap-filling response counts as possibly truncated — a biased
    /// head must never render as a complete run. When truncated, the tail
    /// partial line is dropped so line-oriented parsers only see complete
    /// lines (same rule as `readHead`/`headText`).
    public static func remoteHead(
        data: Data, listedSize: Int, requestedBytes: Int,
        serverFileSize: Int? = nil, serverTruncated: Bool? = nil
    ) -> (data: Data, truncated: Bool) {
        let truncated: Bool
        if let serverTruncated {
            // Belt-and-braces: a size header contradicting the flag still
            // counts as truncation (the response demonstrably lacks bytes).
            truncated = serverTruncated || (serverFileSize ?? 0) > data.count
        } else if let serverFileSize {
            truncated = serverFileSize > data.count
        } else {
            truncated =
                listedSize > data.count
                || (listedSize <= 0 && data.count >= requestedBytes)
        }
        guard truncated else { return (data, false) }
        var head = data[...]
        if let lastNewline = head.lastIndex(of: 0x0A) {
            head = head[..<lastNewline]
        }
        return (Data(head), true)
    }

    /// Assemble one remote file's preview through a fetcher closure. Every
    /// request is head-bounded (see `RemoteFetchPlan`). A transport failure
    /// degrades to `unavailable` with the error text, never a thrown error
    /// — one unreadable file must not sink the run's whole detail pane.
    /// Runs in the CALLER's isolation (`#isolation`) so a non-Sendable
    /// fetcher never crosses actors.
    public static func remotePreview(
        name: String, size: Int,
        isolation: isolated (any Actor)? = #isolation,
        fetch: (_ name: String, _ maxBytes: Int?) async throws -> Data
    ) async -> FilePreview {
        let plan = remoteFetchPlan(name: name, size: size)
        if case .none(let reason) = plan {
            return .unavailable(reason: reason)
        }
        do {
            let data = try await fetch(name, plan.requestBytes)
            return remotePreview(name: name, size: size, data: data)
        } catch {
            return .unavailable(reason: "fetch failed: \(error)")
        }
    }
}
