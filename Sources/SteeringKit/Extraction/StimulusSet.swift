import CryptoKit
import Foundation

public enum StimulusSetError: Error, CustomStringConvertible {
    case missingFile(URL)
    /// EVERY missing file of one stimulus directory, in one refusal (WP0
    /// dry-run punch list P2-10: "one missing file per invocation" meant an
    /// agent authoring a concept discovered the layout one round trip at a
    /// time).
    case missingFiles([URL])
    case malformedLine(file: String, line: Int)
    /// A malformed row in a file whose row SHAPE we can name — the
    /// `{"text": …}` stimulus/corpus files. The plain `malformedLine` stays
    /// for the parsers whose shape is something else (paired stimuli,
    /// multi-concept corpora, scoring rows), where naming `text` would be a
    /// wrong answer confidently given.
    case malformedTextRow(file: String, line: Int)

    case empty(String)

    /// The row shape every `{"text": …}` stimulus file must have. Named in
    /// the refusal because it was otherwise discoverable only by brute force
    /// (P2-10: "errors name file:line, never the expected keys").
    public static let textRowShape =
        #"each line must be a JSON object with a "text" string, e.g. {"text": "…"}"#

    public var description: String {
        switch self {
        case .missingFile(let url): "stimulus file not found: \(url.path)"
        case .missingFiles(let urls):
            "stimulus file\(urls.count == 1 ? "" : "s") not found: "
                + urls.map(\.path).joined(separator: ", ")
                + " — \(Self.textRowShape)"
        case .malformedLine(let file, let line): "malformed JSONL at \(file):\(line)"
        case .malformedTextRow(let file, let line):
            "malformed JSONL at \(file):\(line) — \(Self.textRowShape)"
        case .empty(let name): "stimulus set \(name) has no usable pairs"
        }
    }
}

/// A contrastive stimulus set loaded from data files — adding a new concept
/// must require zero code changes (CLAUDE.md › Experiment A extension
/// points). Layout: `<directory>/{positive,negative}.jsonl`, one
/// `{"text": "..."}` object per line.
public struct StimulusSet: Sendable {
    public let name: String
    public let positive: [String]
    public let negative: [String]
    /// SHA-256 over the raw bytes of both files, hashed into run configs so
    /// results are traceable to the exact stimuli that produced the vector.
    public let hash: String

    /// In-memory class set — designated-reference extraction builds its
    /// classes from stories corpora, not a directory. `hash` is the
    /// caller's pinned identity for the positive class (the concept's
    /// stories.jsonl hash), matching what the manifest pinned at attach.
    public init(name: String, positive: [String], negative: [String], hash: String) {
        self.name = name
        self.positive = positive
        self.negative = negative
        self.hash = hash
    }

    public init(directory: URL) throws {
        self.name = directory.lastPathComponent

        let positiveURL = directory.appending(component: "positive.jsonl")
        let negativeURL = directory.appending(component: "negative.jsonl")

        // BOTH files are checked before either is read (P2-10): a concept
        // directory that has neither used to report `positive.jsonl` alone,
        // so authoring one cost two invocations to learn the layout.
        let absent = [positiveURL, negativeURL].filter {
            !FileManager.default.fileExists(atPath: $0.path)
        }
        guard absent.isEmpty else { throw StimulusSetError.missingFiles(absent) }

        let positiveData = try Data(contentsOf: positiveURL)
        let negativeData = try Data(contentsOf: negativeURL)

        self.positive = try Self.parse(positiveData, file: "positive.jsonl")
        self.negative = try Self.parse(negativeData, file: "negative.jsonl")

        guard !positive.isEmpty, !negative.isEmpty else {
            throw StimulusSetError.empty(name)
        }

        var digest = SHA256()
        digest.update(data: positiveData)
        digest.update(data: negativeData)
        self.hash = digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func read(_ url: URL) throws -> Data {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw StimulusSetError.missingFile(url)
        }
        return try Data(contentsOf: url)
    }

    private struct Line: Decodable {
        let text: String
    }

    /// Loads a plain {"text": …}-per-line JSONL file (dev prompts, neutral
    /// corpus) along with the SHA-256 of its bytes for pinning.
    public static func loadTexts(url: URL) throws -> (texts: [String], hash: String) {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw StimulusSetError.missingFile(url)
        }
        let data = try Data(contentsOf: url)
        let texts = try parse(data, file: url.lastPathComponent)
        var digest = SHA256()
        digest.update(data: data)
        let hash = digest.finalize().map { String(format: "%02x", $0) }.joined()
        return (texts, hash)
    }

    /// RepE/LAT-style paired data. The pair may be content-matched
    /// positive/negative text, or prompts ending in an answer scaffold. The
    /// sign contract is explicit: `positive` is the concept-present side.
    public struct PairedStimulus: Codable, Sendable, Equatable, Hashable {
        public let id: String?
        public let positive: String
        public let negative: String
        public let topic: String?
        public let split: String?
        public let notes: String?

        public init(
            id: String? = nil,
            positive: String,
            negative: String,
            topic: String? = nil,
            split: String? = nil,
            notes: String? = nil
        ) {
            self.id = id
            self.positive = positive
            self.negative = negative
            self.topic = topic
            self.split = split
            self.notes = notes
        }
    }

    public static func loadPairs(url: URL) throws -> (pairs: [PairedStimulus], hash: String) {
        let data = try read(url)
        let pairs = try parseObjects(PairedStimulus.self, data: data, file: url.lastPathComponent)
        guard !pairs.isEmpty else { throw StimulusSetError.empty(url.lastPathComponent) }
        return (pairs, sha256(data))
    }

    /// Multi-concept elicitation corpus, used by the emotion-vector method:
    /// vectors are concept mean minus grand mean over every row in this file.
    public struct MultiConceptStimulus: Codable, Sendable, Equatable, Hashable {
        public let id: String?
        public let concept: String
        public let topic: String?
        public let text: String
        public let split: String?
        public let source: String?
        public let notes: String?

        public init(
            id: String? = nil,
            concept: String,
            topic: String? = nil,
            text: String,
            split: String? = nil,
            source: String? = nil,
            notes: String? = nil
        ) {
            self.id = id
            self.concept = concept
            self.topic = topic
            self.text = text
            self.split = split
            self.source = source
            self.notes = notes
        }
    }

    public static func loadMultiConceptTexts(
        url: URL
    ) throws -> (rows: [MultiConceptStimulus], hash: String) {
        let data = try read(url)
        let rows = try parseObjects(
            MultiConceptStimulus.self, data: data, file: url.lastPathComponent)
        guard !rows.isEmpty else { throw StimulusSetError.empty(url.lastPathComponent) }
        return (rows, sha256(data))
    }

    /// A never-named held-out scenario for convergent validation: evokes the
    /// concept (or deliberately doesn't) WITHOUT its vocabulary — the
    /// emotion paper's discipline, enforced as a separate file
    /// (`validation.jsonl`) that plays no role in extraction.
    public struct ValidationScenario: Decodable, Sendable {
        public let text: String
        public let expresses: Bool
    }

    /// Loads `<directory>/validation.jsonl` if present; nil when the concept
    /// has no validation scenarios yet.
    public static func loadValidation(directory: URL) throws -> [ValidationScenario]? {
        let url = directory.appending(component: "validation.jsonl")
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        let decoder = JSONDecoder()
        let lines = String(decoding: try Data(contentsOf: url), as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: true)
        return try lines.enumerated().map { index, line in
            guard
                let scenario = try? decoder.decode(
                    ValidationScenario.self, from: Data(line.trimmingCharacters(in: .whitespaces).utf8))
            else {
                throw StimulusSetError.malformedLine(file: "validation.jsonl", line: index + 1)
            }
            return scenario
        }
    }

    private static func parse(_ data: Data, file: String) throws -> [String] {
        let decoder = JSONDecoder()
        var rows: [String] = []
        let lines = String(decoding: data, as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: true)
        for (index, line) in lines.enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { continue }
            guard let decoded = try? decoder.decode(Line.self, from: Data(trimmed.utf8))
            else {
                // Named shape (P2-10) — this parser, and only this parser,
                // knows the expected key is `text`.
                throw StimulusSetError.malformedTextRow(file: file, line: index + 1)
            }
            rows.append(decoded.text)
        }
        return rows
    }

    private static func parseObjects<T: Decodable>(
        _ type: T.Type, data: Data, file: String
    ) throws -> [T] {
        let decoder = JSONDecoder()
        var rows: [T] = []
        let lines = String(decoding: data, as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: true)
        for (index, line) in lines.enumerated() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { continue }
            guard let decoded = try? decoder.decode(T.self, from: Data(trimmed.utf8)) else {
                throw StimulusSetError.malformedLine(file: file, line: index + 1)
            }
            rows.append(decoded)
        }
        return rows
    }

    private static func sha256(_ data: Data) -> String {
        var digest = SHA256()
        digest.update(data: data)
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
