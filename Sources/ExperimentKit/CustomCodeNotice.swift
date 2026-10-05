import CryptoKit
import Foundation

/// Custom code a study carries, and the recorded acknowledgement before it runs.
///
/// Everything a workspace holds is data, with one exception: an intervention
/// policy may carry an *expert provider*, Python source the Python engine runs
/// with `exec` while the study runs. Its author sees a warning at authoring; a
/// study shared as a pack or a bundle carries the code to someone who never
/// did. This type is the notice for that recipient:
///
/// - `providers(in:workspaceRoot:)` finds every expert provider in a study
///   document, including inside the exact policy bytes an agent embeds and in
///   the agent and panel files the study references by path, and names each
///   by the SHA-256 of its source text, computed here rather than trusted.
/// - `acknowledge` records who acknowledged which source hash, and when, in
///   `custom-code-acknowledgements.json` at the workspace root, so the notice
///   is not repeated for the same code and a run can show that it was
///   acknowledged.
/// - `runRefusal` is the one gate: a step that can execute the study's agents
///   is not sent to an engine while a provider in it is unacknowledged.
///
/// Nothing here sandboxes anything, and nothing may say it does.
///
/// Python twin: `Server/steerlab_server/experiment/custom_code.py`. The file
/// name, record keys, notice sentence, and executing verbs are duplicated
/// there on purpose; `CustomCodeNoticeTests` and `test_custom_code.py` hold
/// both copies to the same literals.
public enum CustomCodeNotice {
    public static let fileName = "custom-code-acknowledgements.json"
    public static let schemaVersion = 1
    public static let notice =
        "This study contains custom code from its author. It runs with your "
        + "permissions when the study runs. Run it only if you trust the source."
    /// Study steps that can generate text with the study's agents. Every other
    /// step (extract, validate, evaluate, analyze, verify) is never held.
    public static let executingVerbs = ["pipeline", "run", "sweep"]
    /// The keys by which a study document names another file that may carry an
    /// agent. Same set as `InstrumentationSupport.references`.
    static let referenceKeys = ["artifactPath", "multiAgentScenarioPath", "variantArtifactPath"]
    static let program = "steerlab-cli"
    private static let maxDepth = 64
    private static let maxFiles = 1024

    public struct Provider: Codable, Sendable, Equatable {
        public let sha256: String
        public let policyNames: [String]
        public let sourceText: String
    }

    public struct Status: Sendable, Equatable {
        public let sha256: String
        public let policyNames: [String]
        public let sourceText: String
        public let acknowledgedAt: String?
        public let acknowledgedBy: String?
        public var acknowledged: Bool { acknowledgedAt != nil }

        /// The cross-engine row shape (`custom_code.status`), nulls explicit.
        public var json: JSONValue {
            .object([
                "sha256": .string(sha256),
                "policyNames": .array(policyNames.map { .string($0) }),
                "acknowledged": .bool(acknowledged),
                "acknowledgedAt": acknowledgedAt.map { .string($0) } ?? .null,
                "acknowledgedBy": acknowledgedBy.map { .string($0) } ?? .null,
            ])
        }
    }

    /// What a surface shows for one study: every provider and its state.
    public struct Review: Sendable, Equatable {
        public let study: String
        public let providers: [Status]
        public var pending: [Status] { providers.filter { !$0.acknowledged } }
        public var needsAcknowledgement: Bool { !pending.isEmpty }

        public var acknowledgeCommand: String {
            CustomCodeNotice.acknowledgeCommand(study: study, hashes: pending.map(\.sha256))
        }
        public var reviewCommand: String { CustomCodeNotice.reviewCommand(study: study) }

        /// The `customCode` block, key for key the Python client's.
        public var block: JSONValue {
            let waiting = needsAcknowledgement
            return .object([
                "notice": waiting ? .string(CustomCodeNotice.notice) : .null,
                "providers": .array(providers.map(\.json)),
                "acknowledged": .bool(!waiting),
                "recordFile": .string(CustomCodeNotice.fileName),
                "reviewCommand": waiting ? .string(reviewCommand) : .null,
                "acknowledgeCommand": waiting ? .string(acknowledgeCommand) : .null,
            ])
        }

        /// Human-mode lines; empty once everything is acknowledged.
        public var lines: [String] {
            guard needsAcknowledgement else { return [] }
            var lines = [CustomCodeNotice.notice]
            for row in pending {
                let named = row.policyNames.isEmpty ? "unnamed policy" : row.policyNames.joined(separator: ", ")
                lines.append("  custom code SHA-256 \(row.sha256) (policy: \(named))")
            }
            lines.append("Read it: \(reviewCommand)")
            lines.append("If you trust the source: \(acknowledgeCommand)")
            return lines
        }
    }

    // MARK: Finding providers

    public static func providers(in document: JSONValue, workspaceRoot: URL? = nil) -> [Provider] {
        var found: [String: (names: Set<String>, source: String)] = [:]
        for value in documents(document, workspaceRoot: workspaceRoot) {
            scan(value, into: &found, depth: 0)
        }
        return found.keys.sorted().map {
            Provider(sha256: $0, policyNames: found[$0]!.names.sorted(), sourceText: found[$0]!.source)
        }
    }

    static func sha256(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    private static func scan(
        _ value: JSONValue, into found: inout [String: (names: Set<String>, source: String)], depth: Int
    ) {
        guard depth <= maxDepth else { return }
        switch value {
        case .object(let object):
            if case .object(let provider) = object["provider"], case .string(let source) = provider["sourceText"] {
                let digest = sha256(source)
                var entry = found[digest] ?? (names: [], source: source)
                if case .string(let name) = object["name"], !name.isEmpty { entry.names.insert(name) }
                found[digest] = entry
            }
            if case .array(let attached) = object["interventionPolicies"] {
                for item in attached {
                    guard case .object(let attachment) = item, case .string(let text) = attachment["json"],
                        let parsed = try? JSONDecoder().decode(JSONValue.self, from: Data(text.utf8)) else { continue }
                    scan(parsed, into: &found, depth: depth + 1)
                }
            }
            for child in object.values { scan(child, into: &found, depth: depth + 1) }
        case .array(let values):
            for child in values { scan(child, into: &found, depth: depth + 1) }
        default:
            return
        }
    }

    static func references(_ value: JSONValue, depth: Int = 0) -> Set<String> {
        guard depth <= maxDepth else { return [] }
        switch value {
        case .object(let object):
            var result = Set<String>()
            for key in referenceKeys {
                if case .string(let path) = object[key], !path.isEmpty { result.insert(path) }
            }
            for child in object.values { result.formUnion(references(child, depth: depth + 1)) }
            return result
        case .array(let values):
            return values.reduce(into: Set<String>()) { $0.formUnion(references($1, depth: depth + 1)) }
        default:
            return []
        }
    }

    /// The document, then every referenced JSON file inside the workspace. A
    /// reference that leaves the workspace, is missing, or is not JSON is
    /// skipped: verification reports those, and a notice must not fail on them.
    static func documents(_ document: JSONValue, workspaceRoot: URL?) -> [JSONValue] {
        var result = [document]
        guard let workspaceRoot,
            let base = try? ManifestFileTransaction.canonicalPath(workspaceRoot) else { return result }
        var pending = references(document).sorted(by: >)
        var seen = Set<String>()
        while let relative = pending.popLast(), seen.count < maxFiles {
            guard seen.insert(relative).inserted else { continue }
            let candidate = URL(fileURLWithPath: base).appending(path: relative)
            guard let resolved = try? ManifestFileTransaction.canonicalPath(candidate),
                resolved.hasPrefix(base + "/"),
                let data = FileManager.default.contents(atPath: resolved),
                let parsed = try? JSONDecoder().decode(JSONValue.self, from: data) else { continue }
            result.append(parsed)
            pending.append(contentsOf: references(parsed).subtracting(seen).sorted(by: >))
        }
        return result
    }

    // MARK: The record

    public static func recordURL(workspaceRoot: URL) -> URL {
        workspaceRoot.appending(component: fileName)
    }

    private static func unreadable(_ detail: String) -> ExperimentError {
        .refusing(.missingPrerequisite,
            "The custom code acknowledgement record (\(fileName)) cannot be read: \(detail). Nothing was run.",
            repair: "Restore \(fileName) from the workspace history (it is an ordinary tracked file), "
                + "or move it aside and acknowledge the study's custom code again.")
    }

    /// The recorded entries, oldest first, kept verbatim (keys this build does
    /// not know survive a rewrite). Missing file: empty. Damaged: refused,
    /// because reading it as empty would silently re-show every notice.
    static func entries(workspaceRoot: URL) throws -> [JSONValue] {
        guard let data = FileManager.default.contents(atPath: recordURL(workspaceRoot: workspaceRoot).path) else {
            return []
        }
        guard let document = try? JSONDecoder().decode(JSONValue.self, from: data) else {
            throw unreadable("it is not valid JSON")
        }
        guard case .object(let object) = document, object["schemaVersion"] == .number(Double(schemaVersion)),
            case .array(let entries) = object["acknowledgements"] else {
            throw unreadable("it is not a version-1 acknowledgement record")
        }
        return entries.filter {
            guard case .object(let entry) = $0, case .string(let digest) = entry["providerSHA256"] else { return false }
            return isDigest(digest)
        }
    }

    static func isDigest(_ value: String) -> Bool {
        value.count == 64 && value.allSatisfy { "0123456789abcdef".contains($0) }
    }

    /// `providerSHA256 → (acknowledgedAt, acknowledgedBy)` of the first record.
    static func acknowledged(workspaceRoot: URL) throws -> [String: (at: String, by: String)] {
        var result: [String: (at: String, by: String)] = [:]
        for case .object(let entry) in try entries(workspaceRoot: workspaceRoot) {
            guard case .string(let digest) = entry["providerSHA256"], result[digest] == nil else { continue }
            var at = "", by = ""
            if case .string(let value) = entry["acknowledgedAt"] { at = value }
            if case .string(let value) = entry["acknowledgedBy"] { by = value }
            result[digest] = (at, by)
        }
        return result
    }

    /// Every provider the document carries and whether it was acknowledged;
    /// nil when it carries none.
    public static func review(
        of document: JSONValue, study: String, workspaceRoot: URL
    ) throws -> Review? {
        let found = providers(in: document, workspaceRoot: workspaceRoot)
        guard !found.isEmpty else { return nil }
        let seen = try acknowledged(workspaceRoot: workspaceRoot)
        return Review(study: study, providers: found.map {
            Status(sha256: $0.sha256, policyNames: $0.policyNames, sourceText: $0.sourceText,
                   acknowledgedAt: seen[$0.sha256].map(\.at), acknowledgedBy: seen[$0.sha256].map(\.by))
        })
    }

    /// The review for a study's manifest file in a workspace.
    public static func review(study: String, workspaceRoot: URL) throws -> Review? {
        let url = ExperimentRepository(workspaceRoot: workspaceRoot).manifestURL(study)
        let document = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: url))
        return try review(of: document, study: study, workspaceRoot: workspaceRoot)
    }

    public static func acknowledgeCommand(study: String, hashes: [String]) -> String {
        ([reviewCommand(study: study)] + hashes.map { "--sha256 \($0)" }).joined(separator: " ")
    }

    public static func reviewCommand(study: String) -> String {
        "\(program) experiment acknowledge-custom-code \(study)"
    }

    public struct Acknowledged: Sendable, Equatable {
        /// The records this call wrote, as written.
        public let added: [JSONValue]
        public let alreadyAcknowledged: [String]
    }

    /// Record an acknowledgement for each named source hash the study carries.
    /// Idempotent; a hash the study does not carry is refused, so a typo can
    /// never acknowledge code nobody looked at.
    @discardableResult
    public static func acknowledge(
        _ hashes: [String], in document: JSONValue, study: String, workspaceRoot: URL,
        client: String, account: String? = nil, now: String? = nil
    ) throws -> Acknowledged {
        let carried = Dictionary(
            providers(in: document, workspaceRoot: workspaceRoot).map { ($0.sha256, $0) },
            uniquingKeysWith: { first, _ in first })
        guard !carried.isEmpty else {
            throw ExperimentError.refusing(.missingPrerequisite,
                "'\(study)' carries no custom code, so there is nothing to acknowledge.",
                repair: "\(program) experiment verify \(study)")
        }
        var requested: [String] = []
        for hash in hashes where !requested.contains(hash) { requested.append(hash) }
        let unknown = requested.filter { carried[$0] == nil }
        guard unknown.isEmpty, !requested.isEmpty else {
            let lead = unknown.isEmpty
                ? "Name the custom code to acknowledge by its SHA-256. "
                : "'\(study)' carries no custom code with SHA-256 \(unknown.joined(separator: ", ")). "
            throw ExperimentError.refusing(.missingPrerequisite,
                lead + "The custom code it carries: \(carried.keys.sorted().joined(separator: ", ")).",
                repair: reviewCommand(study: study))
        }
        let url = recordURL(workspaceRoot: workspaceRoot)
        return try ManifestFileTransaction.withLock(manifestURL: url, workspaceRoot: workspaceRoot) {
            let existing = try entries(workspaceRoot: workspaceRoot)
            let seen = Set(existing.compactMap { value -> String? in
                guard case .object(let entry) = value, case .string(let digest) = entry["providerSHA256"] else { return nil }
                return digest
            })
            let stamp = now ?? timestamp()
            let who = account ?? (NSUserName().isEmpty ? "unknown" : NSUserName())
            let added: [JSONValue] = requested.filter { !seen.contains($0) }.map { digest in
                .object([
                    "providerSHA256": .string(digest), "acknowledgedAt": .string(stamp),
                    "acknowledgedBy": .string(who), "client": .string(client),
                    "study": .string(study),
                    "policyNames": .array(carried[digest]!.policyNames.map { .string($0) }),
                ])
            }
            if !added.isEmpty {
                let encoder = JSONEncoder()
                encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
                let record = JSONValue.object([
                    "schemaVersion": .number(Double(schemaVersion)),
                    "acknowledgements": .array(existing + added),
                ])
                var data = try encoder.encode(record)
                data.append(contentsOf: Array("\n".utf8))
                try data.write(to: url, options: .atomic)
            }
            return Acknowledged(added: added, alreadyAcknowledged: requested.filter { seen.contains($0) })
        }
    }

    static func timestamp() -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: Date())
    }

    // MARK: The gate

    /// The refusal for a step that would execute unacknowledged custom code, or
    /// nil. `verb == nil` means the step is not known yet (packaging a bundle to
    /// run elsewhere), which is treated as one that may execute.
    public static func runRefusal(
        document: JSONValue, study: String, verb: String?, workspaceRoot: URL,
        action: String = "sent to run"
    ) throws -> ExperimentError? {
        if let verb, !executingVerbs.contains(verb) { return nil }
        guard let review = try review(of: document, study: study, workspaceRoot: workspaceRoot),
            review.needsAcknowledgement else { return nil }
        let named = review.pending.map {
            "SHA-256 \($0.sha256) (policy: \($0.policyNames.isEmpty ? "unnamed" : $0.policyNames.joined(separator: ", ")))"
        }.joined(separator: "; ")
        return .refusing(.missingPrerequisite,
            "'\(study)' was not \(action): it contains custom code from its author that nobody has "
                + "acknowledged in this workspace (\(named)). \(notice)",
            repair: "Read the code with `\(review.reviewCommand)`; if you trust the source, acknowledge it "
                + "on the study's page in the app or run `\(review.acknowledgeCommand)`, then run the study again.")
    }

    /// The app's submission paths speak in one status line: the refusal's
    /// reason and repair, or nil when the step may proceed. A damaged
    /// acknowledgement record refuses too; a manifest that does not decode is
    /// left to the packaging step, which reports it in its own words.
    public static func submissionRefusalText(
        study: String, verb: String?, dryRun: Bool, workspaceRoot: URL?
    ) -> String? {
        guard let workspaceRoot else { return nil }
        do {
            guard let refusal = try runRefusal(
                study: study, verb: verb, dryRun: dryRun, workspaceRoot: workspaceRoot)
            else { return nil }
            return line(refusal)
        } catch let error as ExperimentError where error.lifecycleRefusal != nil {
            return line(error)
        } catch {
            return nil
        }
    }

    static func line(_ error: ExperimentError) -> String {
        guard let repair = error.lifecycleRefusal?.repairAction, !repair.isEmpty else { return error.reason }
        return error.reason + " " + repair
    }

    /// The same gate over a study's manifest file in a workspace. A study whose
    /// manifest is not in this workspace has nothing here to check.
    public static func runRefusal(
        study: String, verb: String?, dryRun: Bool = false, workspaceRoot: URL,
        action: String = "sent to run"
    ) throws -> ExperimentError? {
        guard !dryRun else { return nil }
        let url = ExperimentRepository(workspaceRoot: workspaceRoot).manifestURL(study)
        guard let data = FileManager.default.contents(atPath: url.path) else { return nil }
        let document = try JSONDecoder().decode(JSONValue.self, from: data)
        return try runRefusal(document: document, study: study, verb: verb,
                              workspaceRoot: workspaceRoot, action: action)
    }
}
