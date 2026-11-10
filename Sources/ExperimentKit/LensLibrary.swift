import CryptoKit
import Foundation

/// The Mac's half of the site lens library (public issues #13 and #14).
///
/// A site keeps a lens's converted file once, outside any workspace, filed
/// under its SHA-256 (`converted.sha256`). A lens that starts on this Mac is
/// sent there once, in parts that follow its layers; a request can then refer
/// to it instead of carrying it (`science package --lens-reference`). The
/// workspace stays the truth: anything in a site library can be sent again
/// from a workspace.
///
/// All lens handling is the shared Python client's. What is here is the
/// transport (`ClusterClient`'s lens routes), the order a push runs in, and
/// the sentences. Every sentence has a Python twin in
/// `steerlab_server.client.lens_transfer`, and the two are held to the same
/// bytes by `prompts/fixtures/lens-library/refusals.json`.
public enum LensLibrary {

    // MARK: An engine with no lens library

    /// The engine's own refusal of a staged request carries the same code.
    public static let unavailableCode = "lensLibraryUnavailable"

    /// Python twin: `lens_transfer.LIBRARY_UNAVAILABLE_REPAIR`.
    public static let unavailableRepair =
        "Package the request without --lens-reference so it carries its lens, or set the site "
        + "profile's lensLibrary folder to one the engine can write and run the command again. "
        + "An engine from before the lens library must be updated first."

    /// Why this client sends nothing to an engine with no usable lens
    /// library. `archive` is the upload of a request that refers to its lens.
    /// Python twin: `lens_transfer.library_unavailable_reason`.
    public static func unavailableReason(server: String, archive: Bool = false) -> String {
        (archive ? "This archive refers to its lens instead of carrying it, and the " : "The ")
            + "engine at \(server) does not offer a lens library (its capabilities do not report "
            + "lensLibrary.available as true). Nothing was sent."
    }

    // MARK: A site that forbids HTTP transfer

    public static let externalTransferCode = "externalTransferRequired"

    /// Python twin: `lens_transfer.external_transfer_reason`.
    public static func externalTransferReason(server: String) -> String {
        "The engine at \(server) does not accept file transfers over HTTP, so this client cannot "
            + "send the lens's bytes. Nothing was sent."
    }

    /// Python twin: `lens_transfer.external_transfer_repair`.
    public static func externalTransferRepair(file: String) -> String {
        "Copy \(file) to the site with the site's own transfer tool, then run lens-push again "
            + "with --adopt <the path of that copy on the site>: the site checks the copy against "
            + "the lens's hash and keeps it."
    }

    // MARK: An upload over the server's limit

    public static let oversizeCode = "uploadTooLarge"

    /// Python twin: `lens_transfer.OVERSIZE_LENS_ADVICE`.
    public static let oversizeAdvice =
        "A request that carries a lens can refer to it instead: send the lens once with "
        + "lens-push, then package the request with --lens-reference."

    /// The repair the Python client gives a 413, in this client's words.
    public static let oversizeRepair =
        "Raise the server's STEERLAB_MAX_UPLOAD_BYTES, or package a smaller bundle. "
        + oversizeAdvice + " Nothing was staged."

    public static func oversizeReason(detail: String) -> String {
        let said = detail.trimmingCharacters(in: CharacterSet(charactersIn: ". \n"))
        return said.isEmpty
            ? "The server refused the upload as larger than its limit."
            : "The server refused the upload as larger than its limit: \(said)."
    }

    // MARK: What happened, in one sentence

    /// Python twin: `lens_transfer.push_message`.
    public static func pushMessage(
        lensID: String, uploadedParts: Int, uploadedBytes: Int, adopted: Bool
    ) -> String {
        let sent: String
        if uploadedParts > 0 {
            sent = "Sent lens \(lensID) to the site's lens library (\(uploadedParts) parts, "
                + "\(uploadedBytes) bytes)"
        } else if adopted {
            sent = "The site checked its copy of lens \(lensID) and keeps it in its lens library"
        } else {
            sent = "The site's lens library already holds lens \(lensID), so nothing was uploaded"
        }
        return sent + "; the site has its record."
    }

    /// What `siteRecord` says in a sentence. Python twin:
    /// `lens_transfer._RECORD_CLAUSES`.
    static let recordClauses: [String: String] = [
        "registered": "the site has its record",
        "absent": "the site does not have its record",
        "different": "the site has a different lens under that name",
        "unknown": "this engine does not show its lens records here, so whether it has the "
            + "record is not known",
    ]

    /// Python twin: `lens_transfer.status_message`.
    public static func statusMessage(
        lensID: String, libraryAvailable: Bool, held: Bool, siteRecord: String
    ) -> String {
        let lead: String
        if !libraryAvailable {
            lead = "The engine does not offer a lens library, so lens \(lensID) is not held in one"
        } else if held {
            lead = "The site's lens library holds lens \(lensID)"
        } else {
            lead = "The site's lens library does not hold lens \(lensID)"
        }
        return lead + "; " + (recordClauses[siteRecord] ?? recordClauses["unknown"]!) + "."
    }

    /// Python twin: `lens_transfer.copy_message`.
    public static func copyMessage(
        lensID: String, existing: Bool, changed: Bool, fileCount: Int, bytes: Int,
        planSHA256: String, copyMethod: String?
    ) -> String {
        if existing { return "This workspace already holds lens \(lensID); nothing was copied." }
        if !changed {
            return "Reviewed a copy of lens \(lensID) (\(fileCount) files, \(bytes) bytes); "
                + "nothing was written. Repeat the command with --plan-sha256 \(planSHA256) to copy it."
        }
        let how = copyMethod == "clone" ? "as a clone that uses no extra disk space" : "as a full copy"
        return "Copied lens \(lensID) into this workspace \(how), under the same name and record."
    }

    /// The sentence for a `science lens-copy` answer from the Python client.
    static func copyMessage(for result: JSONValue) -> String? {
        guard case .object(let object) = result, case .string(let lensID)? = object["lensID"],
            case .bool(let existing)? = object["existing"], case .bool(let changed)? = object["changed"],
            case .array(let files)? = object["files"], let bytes = LensLibrary.whole(object["bytes"]),
            case .string(let digest)? = object["planSHA256"]
        else { return nil }
        var method: String?
        if case .string(let named)? = object["copyMethod"] { method = named }
        return copyMessage(
            lensID: lensID, existing: existing, changed: changed, fileCount: files.count,
            bytes: bytes, planSHA256: digest, copyMethod: method)
    }

    // MARK: Types

    /// A lens, a request, or an upload this client will not send as asked:
    /// `refused` (65) on the command line, with its code and repair. The app
    /// shows the reason and the repair together.
    public struct Refusal: Error, Sendable, Equatable, CustomStringConvertible, LocalizedError {
        public let code: String
        public let reason: String
        public let repairAction: String

        public init(code: String, reason: String, repairAction: String) {
            self.code = code
            self.reason = reason
            self.repairAction = repairAction
        }

        public var description: String { repairAction.isEmpty ? reason : reason + " " + repairAction }
        public var errorDescription: String? { description }
    }

    /// One part of a lens file: where it starts, how long it is, and its hash.
    struct Part: Codable, Sendable, Equatable {
        var index: Int
        var offset: Int
        var bytes: Int
        var sha256: String
    }

    /// What the Python client's `lens-push-plan` (with `parts`) or
    /// `lens-record` (without) answers about a workspace lens.
    struct Plan: Sendable, Equatable {
        var lensID: String
        var convertedSHA256: String
        var tensorSHA256: String?
        var bytes: Int
        /// The converted file, relative to the workspace.
        var file: String
        var parts: [Part]
        /// The lens record as exact JSON text (see `lensLibraryRegister`).
        var recordJSON: String

        init(_ value: JSONValue) throws {
            guard case .object(let object) = value, case .string(let lensID)? = object["lensID"],
                case .string(let sha)? = object["convertedSHA256"], let bytes = LensLibrary.whole(object["bytes"]),
                case .string(let file)? = object["file"], case .string(let record)? = object["recordJSON"]
            else { throw ExperimentError(reason: "The local Python client returned no lens plan.") }
            var parts: [Part] = []
            if case .array(let items)? = object["parts"] {
                for item in items {
                    guard case .object(let part) = item, let index = LensLibrary.whole(part["index"]),
                        let offset = LensLibrary.whole(part["offset"]),
                        let length = LensLibrary.whole(part["bytes"]),
                        case .string(let digest)? = part["sha256"]
                    else { throw ExperimentError(reason: "The local Python client returned a malformed lens part.") }
                    parts.append(Part(index: index, offset: offset, bytes: length, sha256: digest))
                }
            }
            self.lensID = lensID
            self.convertedSHA256 = sha
            if case .string(let tensor)? = object["tensorSHA256"] { self.tensorSHA256 = tensor }
            self.bytes = bytes
            self.file = file
            self.parts = parts
            self.recordJSON = record
        }
    }

    static func whole(_ value: JSONValue?) -> Int? {
        if case .number(let number)? = value { return Int(exactly: number) }
        return nil
    }

    // MARK: A request archive that refers to its lens

    /// The bound on an archive's metadata member. Python twin:
    /// `diagnostic_archives.MAX_METADATA`.
    static let metadataLimit = 8 * 1024 * 1024

    /// Whether a file is a request archive whose metadata says it refers to
    /// its lens instead of carrying it (`context.lensReference`). Any other
    /// file, and any archive this cannot read, is not judged here: the server
    /// inspects every upload. Python twin: `lens_transfer.archive_library_lenses`.
    static func archiveRefersToLibrary(_ url: URL) -> Bool {
        guard let data = archiveMetadata(url),
            case .object(let metadata)? = try? JSONDecoder().decode(JSONValue.self, from: data),
            case .object(let context)? = metadata["context"]
        else { return false }
        return context["lensReference"] == .bool(true)
    }

    /// The archive's `steerlab-bundle.json`, read by the system `tar` (as the
    /// bundle packer and importer do), or nil.
    static func archiveMetadata(_ url: URL) -> Data? {
        let process = Process()
        process.executableURL = URL(filePath: "/usr/bin/tar")
        process.arguments = ["-xOf", url.path, "steerlab-bundle.json"]
        let output = Pipe()
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        process.standardInput = FileHandle.nullDevice
        do { try process.run() } catch { return nil }
        var data = Data()
        while let chunk = try? output.fileHandleForReading.read(upToCount: 1 << 16), !chunk.isEmpty {
            data.append(chunk)
            if data.count > metadataLimit {
                process.terminate()
                break
            }
        }
        process.waitUntilExit()
        guard process.terminationStatus == 0, !data.isEmpty, data.count <= metadataLimit else { return nil }
        return data
    }

    // MARK: One part of a lens file

    /// Copy one part out of the lens file, a megabyte at a time, checking it
    /// against the hash the plan recorded.
    static func copyPart(_ part: Part, of file: URL, to piece: URL) throws {
        let changed = Refusal(
            code: "lensRefused",
            reason: "The converted lens file changed after it was checked. Nothing more was sent.",
            repairAction: "Run lens-push again once nothing else is writing the lens folder: it sends "
                + "only what the site still lacks.")
        let source = try FileHandle(forReadingFrom: file)
        defer { try? source.close() }
        guard FileManager.default.createFile(
            atPath: piece.path, contents: nil, attributes: [.posixPermissions: 0o600])
        else { throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: piece.path]) }
        let target = try FileHandle(forWritingTo: piece)
        defer { try? target.close() }
        try source.seek(toOffset: UInt64(part.offset))
        var hasher = SHA256()
        var remaining = part.bytes
        while remaining > 0 {
            guard let block = try source.read(upToCount: min(1 << 20, remaining)), !block.isEmpty else {
                throw changed
            }
            hasher.update(data: block)
            try target.write(contentsOf: block)
            remaining -= block.count
        }
        guard hasher.finalize().map({ String(format: "%02x", $0) }).joined() == part.sha256 else {
            throw changed
        }
    }
}

/// Sending a workspace lens to a site, and asking what the site holds.
public enum LensLibraryRemote {

    /// The Python client's local actions (`lens-push-plan`, `lens-record`).
    /// Injectable so the order of a push is testable without a Python
    /// environment.
    public typealias Bridge = @Sendable (String, [String: JSONValue]) async throws -> JSONValue

    public static let localBridge: Bridge = { try await DiagnosticWorkspace.perform($0, payload: $1) }

    /// What a push did. Python twin: the result of `lens_transfer.push`.
    public struct PushResult: Sendable, Equatable {
        public var lensID: String
        public var convertedSHA256: String
        public var bytes: Int
        /// The site held the bytes before this push sent any.
        public var alreadyHeld: Bool
        public var adopted: Bool
        public var uploadedParts: Int
        public var uploadedBytes: Int
        /// The server's answers to the registration, as it gave them.
        public var registered: Bool?
        public var recordChanged: Bool?

        public var changed: Bool { uploadedParts > 0 || adopted || recordChanged == true }

        public var json: JSONValue {
            .object([
                "lensID": .string(lensID), "convertedSHA256": .string(convertedSHA256),
                "bytes": .number(Double(bytes)), "held": .bool(true),
                "alreadyHeld": .bool(alreadyHeld), "adopted": .bool(adopted),
                "uploadedParts": .number(Double(uploadedParts)),
                "uploadedBytes": .number(Double(uploadedBytes)),
                "registered": registered.map(JSONValue.bool) ?? .null,
                "recordChanged": recordChanged.map(JSONValue.bool) ?? .null,
                "changed": .bool(changed),
            ])
        }

        public var message: String {
            LensLibrary.pushMessage(
                lensID: lensID, uploadedParts: uploadedParts, uploadedBytes: uploadedBytes,
                adopted: adopted)
        }
    }

    /// What a site holds of a workspace lens. Python twin: the result of
    /// `lens_transfer.status`.
    public struct StatusResult: Sendable, Equatable {
        public var lensID: String
        public var convertedSHA256: String
        public var bytes: Int
        public var libraryAvailable: Bool
        public var held: Bool
        /// The parts of an upload in progress the site has received.
        public var receivedParts: [Int]?
        /// `registered`, `absent`, `different`, or `unknown`.
        public var siteRecord: String

        public var json: JSONValue {
            .object([
                "lensID": .string(lensID), "convertedSHA256": .string(convertedSHA256),
                "bytes": .number(Double(bytes)), "libraryAvailable": .bool(libraryAvailable),
                "held": .bool(held),
                "receivedParts": receivedParts.map { .array($0.map { .number(Double($0)) }) } ?? .null,
                "siteRecord": .string(siteRecord), "changed": .bool(false),
            ])
        }

        public var message: String {
            LensLibrary.statusMessage(
                lensID: lensID, libraryAvailable: libraryAvailable, held: held, siteRecord: siteRecord)
        }
    }

    /// Send one lens to the site's library and register its record there.
    ///
    /// In order: the capability, what the site holds, the part manifest, only
    /// the parts the site lacks, the whole-file check, the record. Run again
    /// after an interruption, it sends only what is still missing; when the
    /// site already holds the bytes it sends none and still registers the
    /// record. With `adopt`, the bytes are a copy already on the site at that
    /// path, and the site checks it against the lens's hash.
    public static func push(
        _ lensID: String, adopt: String? = nil, client: ClusterClient, root: URL,
        bridge: Bridge = localBridge
    ) async throws -> PushResult {
        let server = client.profile.baseURL.absoluteString
        let capabilities = try await client.capabilities()
        guard capabilities.lensLibraryAvailable else {
            throw LensLibrary.Refusal(
                code: LensLibrary.unavailableCode,
                reason: LensLibrary.unavailableReason(server: server),
                repairAction: LensLibrary.unavailableRepair)
        }
        var payload: [String: JSONValue] = [
            "workspaceRoot": .string(root.path), "lensID": .string(lensID),
        ]
        let plan: LensLibrary.Plan
        if adopt != nil {
            plan = try LensLibrary.Plan(try await bridge("lens-record", payload))
        } else {
            guard let limit = capabilities.lensLibrary?.partMaxBytes, limit >= 1 else {
                throw ExperimentError(
                    reason: "The engine at \(server) reports a lens library with no usable partMaxBytes.")
            }
            payload["partMaxBytes"] = .number(Double(limit))
            plan = try LensLibrary.Plan(try await bridge("lens-push-plan", payload))
        }
        let sha = plan.convertedSHA256
        let status = try await client.lensLibraryStatus(sha)
        var result = PushResult(
            lensID: lensID, convertedSHA256: sha, bytes: plan.bytes, alreadyHeld: status.held,
            adopted: false, uploadedParts: 0, uploadedBytes: 0)
        if status.held, let held = status.bytes, held != plan.bytes {
            throw LensLibrary.Refusal(
                code: "lensLibraryEntryDiffers",
                reason: "The site holds \(held) bytes under this lens's hash, and the workspace file "
                    + "is \(plan.bytes) bytes. Nothing was sent.",
                repairAction: "Ask the site's operator to inspect that library entry; the workspace "
                    + "copy is the one its record describes.")
        }
        if !status.held, let adopt {
            _ = try await client.lensLibraryAdopt(path: adopt, sha256: sha)
            result.adopted = true
        } else if !status.held {
            let remote = capabilities.remoteStudy
            guard remote?.externalTransferRequired != true, remote?.httpTransfer != false else {
                throw LensLibrary.Refusal(
                    code: LensLibrary.externalTransferCode,
                    reason: LensLibrary.externalTransferReason(server: server),
                    repairAction: LensLibrary.externalTransferRepair(file: plan.file))
            }
            let begun = try await client.lensLibraryBegin(sha, bytes: plan.bytes, parts: plan.parts)
            if begun.held {
                result.alreadyHeld = true
            } else {
                let indices = Set(plan.parts.map(\.index))
                guard let missing = begun.missingParts, Set(missing).isSubset(of: indices) else {
                    throw ExperimentError(
                        reason: "The engine named parts to send that are not in the manifest it was given.")
                }
                let wanted = Set(missing)
                let file = root.appending(path: plan.file)
                for part in plan.parts where wanted.contains(part.index) {
                    try await client.lensLibraryPutPart(sha, part: part, of: file)
                    result.uploadedParts += 1
                    result.uploadedBytes += part.bytes
                }
                let done = try await client.lensLibraryComplete(sha)
                guard done.held, done.bytes == plan.bytes else {
                    throw ExperimentError(
                        reason: "The engine did not confirm that it holds the whole lens after the "
                            + "upload. Run the command again: it sends only what the site still lacks.")
                }
            }
        }
        let registration = try await client.lensLibraryRegister(recordJSON: plan.recordJSON)
        guard registration.lensID == lensID else {
            throw ExperimentError(
                reason: "The engine registered a record under another lens name than the one sent.")
        }
        result.registered = registration.registered
        result.recordChanged = registration.changed
        return result
    }

    /// Whether the site holds a workspace lens's bytes, and whether it has
    /// the lens's record. Reads only.
    public static func status(
        _ lensID: String, client: ClusterClient, root: URL, bridge: Bridge = localBridge
    ) async throws -> StatusResult {
        let plan = try LensLibrary.Plan(
            try await bridge(
                "lens-record", ["workspaceRoot": .string(root.path), "lensID": .string(lensID)]))
        var result = StatusResult(
            lensID: lensID, convertedSHA256: plan.convertedSHA256, bytes: plan.bytes,
            libraryAvailable: try await client.capabilities().lensLibraryAvailable, held: false,
            receivedParts: nil, siteRecord: "unknown")
        if result.libraryAvailable {
            let held = try await client.lensLibraryStatus(plan.convertedSHA256)
            result.held = held.held
            result.receivedParts = held.receivedParts
        }
        let site = try await client.siteLensRecord(lensID)
        if site.known {
            if let hashes = site.hashes {
                let same = hashes.converted == plan.convertedSHA256 && hashes.tensor == plan.tensorSHA256
                result.siteRecord = same ? "registered" : "different"
            } else {
                result.siteRecord = "absent"
            }
        }
        return result
    }
}

/// `remote lens-push` and `remote lens-status`.
enum LensLibraryRemoteCLI {
    static let verbs = ["lens-push", "lens-status"]

    static func run(
        _ args: [String], client: ClusterClient, root: URL, sink: ExperimentCLISink,
        bridge: @escaping LensLibraryRemote.Bridge = LensLibraryRemote.localBridge
    ) async throws -> ExperimentCLIResult {
        let parsed = try DiagnosticArguments(args, namespace: "remote", takesValue: true)
        let lensID = parsed.positional!
        let result: JSONValue
        let message: String
        let changed: Bool
        if parsed.verb == "lens-push" {
            let pushed = try await LensLibraryRemote.push(
                lensID, adopt: parsed.flags["--adopt"], client: client, root: root, bridge: bridge)
            (result, message, changed) = (pushed.json, pushed.message, pushed.changed)
        } else {
            let status = try await LensLibraryRemote.status(
                lensID, client: client, root: root, bridge: bridge)
            (result, message, changed) = (status.json, status.message, false)
        }
        sink.out(String(decoding: try JSONEncoder().encode(result), as: UTF8.self))
        return .init(
            message: message, changed: changed,
            payload: [
                "endpoint": .string(client.profile.baseURL.absoluteString),
                "workspaceRoot": .string(root.path), "response": result,
            ])
    }
}
