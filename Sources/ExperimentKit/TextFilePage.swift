import Foundation

/// Bounded, read-only text pages. Byte cursors preserve UTF-8 boundaries and
/// allow arbitrarily long JSONL records to be inspected without loading a run.
public enum TextFilePage {
    public struct Page: Sendable {
        public let text: String
        public let nextOffset: UInt64
        public let totalBytes: UInt64
        public var hasMore: Bool { nextOffset < totalBytes }
    }

    public static func read(url: URL, offset: UInt64 = 0, byteLimit: Int = 64 * 1024) throws -> Page {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        let total = try file.seekToEnd()
        let start = min(offset, total)
        try file.seek(toOffset: start)
        let limit = max(4, min(byteLimit, 256 * 1024))
        let data = try file.read(upToCount: limit + 3) ?? Data()
        var end = min(limit, data.count)
        // Include the remainder of a UTF-8 scalar if the nominal boundary
        // falls inside it. Malformed input is still displayed lossily.
        while end < data.count && data[end] & 0xC0 == 0x80 { end += 1 }
        return Page(text: String(decoding: data.prefix(end), as: UTF8.self),
            nextOffset: start + UInt64(end), totalBytes: total)
    }
}
