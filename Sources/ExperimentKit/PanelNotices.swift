import Foundation
import Observation

/// One persistent panel notice (App gap A15). Panel `status` strings are
/// single-slot and overwritten by the next event; a notice is the same
/// message kept — so an overnight failure is still consultable in the
/// morning instead of living only in the run directory.
public struct PanelNotice: Identifiable, Codable, Sendable, Equatable {
    public enum Severity: String, Codable, Sendable, CaseIterable {
        case info
        case success
        case warning
        case error

        /// SF Symbol name for the severity icon.
        public var symbolName: String {
            switch self {
            case .info: "info.circle"
            case .success: "checkmark.circle"
            case .warning: "exclamationmark.triangle"
            case .error: "xmark.octagon"
            }
        }
    }

    public var id: UUID
    public var timestamp: Date
    /// Which panel spoke ("Studies", "Agents").
    public var source: String
    public var severity: Severity
    /// The legacy status string, verbatim.
    public var message: String

    public init(
        id: UUID = UUID(), timestamp: Date = Date(), source: String,
        severity: Severity, message: String
    ) {
        self.id = id
        self.timestamp = timestamp
        self.source = source
        self.severity = severity
        self.message = message
    }
}

/// Append-only in-memory ring of panel notices (cap `capacity`), persisted
/// per-workspace to `.steerlab/notices.json` — best-effort and atomic, like
/// the other `.steerlab/` state files. The store is a FEED, not a control
/// surface: panels append through their `note(...)` helpers (which also set
/// the legacy single-slot status string), and the bell button in the
/// Studies/Agents headers renders the ring with a Clear action. Errors
/// badge the bell until the feed is viewed (`markViewed`); the badge is
/// session state, deliberately not persisted — the ring itself is.
@Observable @MainActor
public final class PanelNotices {

    public static let capacity = 200
    /// Schema stamp inside notices.json so a future shape change can
    /// migrate instead of silently misparsing.
    static let schemaVersion = 1

    public static let shared = PanelNotices()

    /// Notices in append order (oldest first); the UI renders them newest
    /// first via `recentFirst`.
    public private(set) var notices: [PanelNotice] = []
    /// Error notices recorded since the feed was last viewed — drives the
    /// bell badge ("errors badge the bell until viewed").
    public private(set) var unseenErrorCount = 0

    /// Test seam: a fixed persistence file. nil = the live per-workspace
    /// location, resolved at every write so a workspace switch mid-session
    /// starts persisting to the new workspace's file.
    private let fixedFileURL: URL?

    /// Live location: `<workspace>/.steerlab/notices.json`.
    public var fileURL: URL {
        fixedFileURL
            ?? VectorCatalog.projectRoot.appending(
                components: ".steerlab", "notices.json")
    }

    public init(fileURL: URL? = nil) {
        self.fixedFileURL = fileURL
        load()
    }

    public var recentFirst: [PanelNotice] {
        notices.reversed()
    }

    public var hasUnseenErrors: Bool { unseenErrorCount > 0 }

    /// Append one notice, trim to capacity, persist best-effort.
    public func record(
        source: String, severity: PanelNotice.Severity, message: String
    ) {
        let trimmed = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        notices.append(
            PanelNotice(source: source, severity: severity, message: trimmed))
        if notices.count > Self.capacity {
            notices.removeFirst(notices.count - Self.capacity)
        }
        if severity == .error {
            unseenErrorCount += 1
        }
        persist()
    }

    /// The feed was opened: the error badge clears (the notices stay).
    public func markViewed() {
        unseenErrorCount = 0
    }

    /// Clear the ring (and its persisted file). The badge clears with it.
    public func clear() {
        notices = []
        unseenErrorCount = 0
        persist()
    }

    // MARK: - Persistence (best-effort, atomic, trimmed)

    private struct FileShape: Codable {
        var schemaVersion: Int
        var notices: [PanelNotice]
    }

    private func load() {
        guard
            let data = try? Data(contentsOf: fileURL),
            let file = try? Self.decoder.decode(FileShape.self, from: data),
            file.schemaVersion == Self.schemaVersion
        else { return }
        notices = Array(file.notices.suffix(Self.capacity))
    }

    private func persist() {
        let url = fileURL
        let file = FileShape(schemaVersion: Self.schemaVersion, notices: notices)
        guard let data = try? Self.encoder.encode(file) else { return }
        try? FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: url, options: .atomic)
    }

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }()

    private static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()
}
