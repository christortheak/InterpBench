import CryptoKit
import Foundation
import SteeringKit

public enum NeutralCorpusKind: String, Codable, Sendable, CaseIterable {
    case normCalibration
    case projection

    public var label: String {
        switch self {
        case .normCalibration: "Norm calibration"
        case .projection: "Projection basis"
        }
    }
}

public struct NeutralCorpusRecord: Identifiable, Sendable, Equatable {
    public let id: String
    public let kind: NeutralCorpusKind
    public let name: String
    public let url: URL
    public let count: Int
    public let hash: String?

    public var label: String {
        switch kind {
        case .normCalibration:
            return "Norm calibration corpus"
        case .projection:
            return name
        }
    }
}

public enum NeutralCorpusStore {
    private struct Row: Codable {
        let text: String
    }

    public static var normCorpusURL: URL { normCorpusURL(root: VectorCatalog.projectRoot) }

    public static var projectionRoot: URL { projectionRoot(root: VectorCatalog.projectRoot) }

    /// Root-parameterized forms. The resolved workspace is the default
    /// everywhere; an explicit root exists so a scanner that already knows
    /// which tree it is walking (`DatasetInventory.scan(root:)`) reuses THIS
    /// store rather than re-deriving the corpus layout.
    public static func normCorpusURL(root: URL) -> URL {
        root.appending(components: "prompts", "neutral", "corpus.jsonl")
    }

    public static func projectionRoot(root: URL) -> URL {
        root.appending(components: "prompts", "neutral", "projection")
    }

    public static var normCorpusID: String { "norm-calibration" }

    public static func projectionID(_ name: String) -> String {
        "projection:\(slugify(name))"
    }

    public static func scan(root: URL = VectorCatalog.projectRoot) -> [NeutralCorpusRecord] {
        var records: [NeutralCorpusRecord] = []
        let norm = corpusRecord(
            id: normCorpusID,
            kind: .normCalibration,
            name: "norm-calibration",
            url: normCorpusURL(root: root))
        records.append(norm)

        let fm = FileManager.default
        if let entries = try? fm.contentsOfDirectory(
            at: projectionRoot(root: root),
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles])
        {
            records.append(
                contentsOf: entries.compactMap { directory -> NeutralCorpusRecord? in
                    guard
                        (try? directory.resourceValues(forKeys: [.isDirectoryKey]).isDirectory)
                            == true
                    else { return nil }
                    let url = directory.appending(component: "corpus.jsonl")
                    return corpusRecord(
                        id: projectionID(directory.lastPathComponent),
                        kind: .projection,
                        name: directory.lastPathComponent,
                        url: url)
                })
        }
        return records.sorted {
            if $0.kind != $1.kind { return $0.kind == .normCalibration }
            return $0.name.localizedStandardCompare($1.name) == .orderedAscending
        }
    }

    public static func record(id: String?) -> NeutralCorpusRecord {
        guard let id else {
            return corpusRecord(
                id: normCorpusID,
                kind: .normCalibration,
                name: "norm-calibration",
                url: normCorpusURL)
        }
        if let existing = scan().first(where: { $0.id == id }) {
            return existing
        }
        if id.hasPrefix("projection:") {
            return projectionRecord(named: String(id.dropFirst("projection:".count)))
        }
        return corpusRecord(
            id: normCorpusID,
            kind: .normCalibration,
            name: "norm-calibration",
            url: normCorpusURL)
    }

    public static func loadTexts(record: NeutralCorpusRecord) throws -> (texts: [String], hash: String) {
        try StimulusSet.loadTexts(url: record.url)
    }

    public static func parseTexts(_ data: Data) throws -> [String] {
        let trimmed = String(decoding: data, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        var texts: [String] = []
        if trimmed.first == "[" {
            let object = try JSONSerialization.jsonObject(with: Data(trimmed.utf8))
            if let strings = object as? [String] {
                texts.append(contentsOf: strings)
            } else if let rows = object as? [[String: Any]] {
                texts.append(contentsOf: rows.compactMap { $0["text"] as? String })
            }
        } else {
            var decodedJSONL = 0
            for line in trimmed.split(whereSeparator: \.isNewline) {
                let raw = String(line).trimmingCharacters(in: .whitespacesAndNewlines)
                guard !raw.isEmpty, raw.first == "{" else { continue }
                let row = try JSONDecoder().decode(Row.self, from: Data(raw.utf8))
                texts.append(row.text)
                decodedJSONL += 1
            }
            if decodedJSONL == 0 {
                texts.append(
                    contentsOf: trimmed.components(separatedBy: "\n\n")
                        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) })
            }
        }

        var seen: Set<String> = []
        return texts.compactMap { text -> String? in
            let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !clean.isEmpty, !seen.contains(clean) else { return nil }
            seen.insert(clean)
            return clean
        }
    }

    @discardableResult
    public static func save(texts: [String], to record: NeutralCorpusRecord) throws -> NeutralCorpusRecord {
        try FileManager.default.createDirectory(
            at: record.url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let rows = texts.map { Row(text: $0) }
        let jsonl = rows.compactMap { row -> String? in
            guard let data = try? JSONEncoder().encode(row) else { return nil }
            return String(decoding: data, as: UTF8.self)
        }
        .joined(separator: "\n") + "\n"
        try jsonl.write(to: record.url, atomically: true, encoding: .utf8)
        return corpusRecord(
            id: record.id,
            kind: record.kind,
            name: record.name,
            url: record.url)
    }

    public static func projectionRecord(named rawName: String) -> NeutralCorpusRecord {
        let name = slugify(rawName.isEmpty ? "projection-neutral" : rawName)
        return corpusRecord(
            id: projectionID(name),
            kind: .projection,
            name: name,
            url: projectionRoot.appending(components: name, "corpus.jsonl"))
    }

    public static func relativePath(for url: URL) -> String {
        let root = VectorCatalog.projectRoot.path
        let path = url.path
        if path.hasPrefix(root + "/") {
            return String(path.dropFirst(root.count + 1))
        }
        return path
    }

    public static func corpusRecord(
        id: String,
        kind: NeutralCorpusKind,
        name: String,
        url: URL
    ) -> NeutralCorpusRecord {
        guard let loaded = try? StimulusSet.loadTexts(url: url) else {
            return NeutralCorpusRecord(id: id, kind: kind, name: name, url: url, count: 0, hash: nil)
        }
        return NeutralCorpusRecord(
            id: id,
            kind: kind,
            name: name,
            url: url,
            count: loaded.texts.count,
            hash: loaded.hash)
    }

    public static func slugify(_ value: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        let scalars = value.lowercased().unicodeScalars.map { scalar -> Character in
            allowed.contains(scalar) ? Character(scalar) : "-"
        }
        let slug = String(scalars)
            .replacingOccurrences(of: "--+", with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return slug.isEmpty ? "projection-neutral" : slug
    }
}
