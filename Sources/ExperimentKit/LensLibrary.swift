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
///
/// "Lens library" here always means the site's. A workspace's own lenses are
/// the folders under `runs/jlens-lenses/`.
public enum LensLibrary {

    // MARK: An engine with no lens library

    /// The engine's own refusal of a staged request carries the same code.
    public static let unavailableCode = "lensLibraryUnavailable"

    /// The repair for the upload of a request that refers to its lens. Python
    /// twin: `lens_transfer.LIBRARY_UNAVAILABLE_REPAIR`.
    public static let archiveUnavailableRepair =
        "Package the request without --lens-reference so it carries its lens, or give the engine a "
        + "lens library folder it can write (on a cluster, the site profile's lensLibrary folder; on "
        + "any other engine, STEERLAB_LENS_LIBRARY_ROOT) and upload the request again. An engine from "
        + "before the lens library must be updated first."

    /// The repair for `lens-push`, which packages nothing. Python twin:
    /// `lens_transfer.PUSH_UNAVAILABLE_REPAIR`.
    public static let pushUnavailableRepair =
        "Give the engine a lens library folder it can write (on a cluster, the site profile's "
        + "lensLibrary folder; on any other engine, STEERLAB_LENS_LIBRARY_ROOT), then run the command "
        + "again. An engine from before the lens library must be updated first."

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

    // MARK: What a site holds, and an engine that does not keep the contract

    /// The site holds something else under the lens's hash.
    public static let entryDiffersCode = "lensLibraryEntryDiffers"
    /// Python twin: `lens_transfer.ENTRY_DIFFERS_REPAIR`.
    public static let entryDiffersRepair =
        "Ask the site's operator to inspect that library entry; the workspace copy is the one its "
        + "record describes."

    /// Python twin: `lens_transfer.entry_differs_reason`.
    public static func entryDiffersReason(held: Int, workspace: Int) -> String {
        "The site holds \(held) bytes under this lens's hash, and the workspace file is \(workspace) "
            + "bytes. Nothing was sent."
    }

    /// The engine answered a lens-library route with something the contract
    /// does not allow. A failure, not a refusal.
    public static let protocolCode = "lensLibraryProtocol"

    public enum ProtocolProblem: String, CaseIterable, Sendable {
        case partLimit, missingParts, notConfirmed, otherLens
    }

    /// The sentence and repair for an answer the engine should never give.
    /// Python twin: `lens_transfer.protocol_problem`.
    public static func protocolProblem(_ kind: ProtocolProblem, server: String = "")
        -> (reason: String, repair: String)
    {
        switch kind {
        case .partLimit:
            ("The engine at \(server) reports a lens library with no usable partMaxBytes.",
             "Update the engine; its capabilities must state the largest part it accepts.")
        case .missingParts:
            ("The engine named parts to send that are not in the manifest it was given.",
             "Run the command again; if it repeats, update the engine.")
        case .notConfirmed:
            ("The engine did not confirm that it holds the whole lens after the upload.", resumeRepair)
        case .otherLens:
            ("The engine registered a record under another lens name than the one sent.",
             "Run lens-status for this lens, and update the engine if it repeats.")
        }
    }

    static func protocolFailure(_ kind: ProtocolProblem, server: String = "") -> Refusal {
        let problem = protocolProblem(kind, server: server)
        return Refusal(code: protocolCode, reason: problem.reason, repairAction: problem.repair, state: .failed)
    }

    // MARK: A refusal that brings no code, no sentence, or no repair

    /// An engine's refusal on a lens-library route that names no code, gives
    /// no sentence, or names no repair gets these, on both clients. Python
    /// twins: `lens_transfer.REFUSAL_FALLBACK_CODE`,
    /// `refusal_fallback_reason`, and `REFUSAL_FALLBACK_REPAIR`.
    public static let refusalFallbackCode = "lensLibraryRefused"
    public static func refusalFallbackReason(status: Int) -> String {
        "The engine refused the lens library request (\(status)) without saying why."
    }
    public static let refusalFallbackRepair =
        "The engine named no repair. If its reason does not say what to change, look in the engine's "
        + "log for this request."
    /// No more of a refusal's text than this is shown, whatever sent it.
    /// Python twin: `lens_transfer.REFUSAL_TEXT_LIMIT`.
    public static let refusalTextLimit = 2000

    // MARK: An upload that did not finish, and a file that is not the plan's

    /// One of a push's calls to the engine did not get its answer.
    public static let interruptedCode = "lensUploadInterrupted"
    /// Python twin: `lens_transfer.RESUME_REPAIR`.
    public static let resumeRepair = "Run the command again: it sends only what the site still lacks."

    /// True of each call a push makes (the status read, the manifest, a
    /// part, the whole-file check, an adoption, the record): the push ends
    /// only when the site has confirmed both the lens and its record.
    /// Python twin: `lens_transfer.interrupted_reason`.
    public static func interruptedReason(lensID: String, cause: String) -> String {
        let said = cause.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmed = said.hasSuffix(".") ? String(said.dropLast()) : said
        return "Sending lens \(lensID) to the site was interrupted before the site confirmed that it "
            + "holds the whole lens and has its record (\(trimmed))."
    }

    /// One of lens-status's reads of the engine did not get its answer. A
    /// failure, not a refusal. Python twins:
    /// `lens_transfer.STATUS_UNANSWERED_CODE`, `STATUS_RETRY_REPAIR`, and
    /// `status_unanswered_reason`.
    public static let statusUnansweredCode = "lensStatusUnanswered"
    public static let statusRetryRepair = "Run the command again: lens-status only reads."
    public static func statusUnansweredReason(lensID: String, cause: String) -> String {
        let said = cause.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmed = said.hasSuffix(".") ? String(said.dropLast()) : said
        return "The engine did not answer lens-status about lens \(lensID) (\(trimmed)). Nothing was changed."
    }

    /// The lens file is not the one that was planned. Python twins:
    /// `lens_transfer.CHANGED_CODE`, `CHANGED_REASON`, and `CHANGED_REPAIR`.
    public static let changedCode = "lensRefused"
    public static let changedReason =
        "The converted lens file changed after it was checked. Nothing more was sent."
    public static let changedRepair =
        "Run lens-push again once nothing else is writing the lens folder: it sends only what the "
        + "site still lacks."

    // MARK: A lens copy asked for without its arguments

    /// Python twins: `lens_transfer.COPY_NEEDS_LENS` and `COPY_NEEDS_FROM`.
    public static let copyNeedsLens =
        "science lens-copy takes one lens ID: the name of the lens's folder under "
        + "runs/jlens-lenses/ in the workspace that holds it."
    public static let copyNeedsFrom =
        "science lens-copy needs --from <workspace directory>: the workspace that holds the lens."

    // MARK: What happened, in one sentence

    /// Python twin: `lens_transfer.push_message`.
    public static func pushMessage(
        lensID: String, uploadedParts: Int, uploadedBytes: Int, adopted: Bool, alreadyHeld: Bool,
        recordNonFiniteNumbers: Int = 0
    ) -> String {
        let sent: String
        if uploadedParts > 0 {
            sent = "Sent lens \(lensID) to the site's lens library (\(uploadedParts) parts, "
                + "\(uploadedBytes) bytes)"
        } else if adopted {
            sent = "The site checked its copy of lens \(lensID) and keeps it in its lens library"
        } else if alreadyHeld {
            sent = "The site's lens library already holds lens \(lensID), so nothing was uploaded"
        } else {
            // An earlier run sent every part and never got the whole-file check.
            sent = "The site had already received every part of lens \(lensID) and now holds it in its "
                + "lens library, so nothing was uploaded"
        }
        var nulls = ""
        if recordNonFiniteNumbers == 1 {
            nulls = ", with 1 number in the record that is not finite stored as null"
        } else if recordNonFiniteNumbers > 1 {
            nulls = ", with \(recordNonFiniteNumbers) numbers in the record that are not finite stored as null"
        }
        return sent + "; the site has its record" + nulls + "."
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

    /// What a `lens-copy` answer adds when this workspace holds the same lens
    /// under other names (`alsoHeldAs`), and nothing when it holds none. True
    /// before the copy and after it. Python twin:
    /// `lens_transfer.also_held_clause`.
    public static func alsoHeldClause(_ names: [String]) -> String {
        guard let last = names.last else { return "" }
        let listed: String
        switch names.count {
        case 1: listed = last
        case 2: listed = names[0] + " and " + last
        default: listed = names.dropLast().joined(separator: ", ") + ", and " + last
        }
        return " This workspace holds the same lens under "
            + (names.count == 1 ? "another name: " : "other names: ") + listed
            + ". Each name has its own record and qualifications."
    }

    /// What a `lens-copy` answer says, true in every case; it ends by naming
    /// any other names under which this workspace holds the same lens. Python
    /// twin: `lens_transfer.copy_message`.
    public static func copyMessage(
        lensID: String, action: String, existing: Bool, changed: Bool, restored: Bool = false,
        alsoHeldAs: [String] = [], fileCount: Int, bytes: Int, planSHA256: String,
        copyMethod: String?
    ) -> String {
        let again = "Repeat the command with --plan-sha256 \(planSHA256)"
        let how = copyMethod == "clone" ? "as a clone that uses no extra disk space" : "as a full copy"
        let said: String
        if restored {
            said = "Lens \(lensID) was already in this workspace; its missing converted file was put back \(how)."
        } else if action == "restore" {
            said = "Lens \(lensID) is already in this workspace, but its converted file is missing; "
                + "nothing was written. \(again) to put that file back."
        } else if existing {
            said = "This workspace already holds lens \(lensID); nothing was copied."
        } else if !changed {
            said = "Reviewed a copy of lens \(lensID) (\(fileCount) files, \(bytes) bytes); "
                + "nothing was written. \(again) to copy it."
        } else {
            said = "Copied lens \(lensID) into this workspace \(how), under the same name and record."
        }
        return said + alsoHeldClause(alsoHeldAs)
    }

    /// The sentence for a `science lens-copy` answer from the Python client.
    static func copyMessage(for result: JSONValue) -> String? {
        guard case .object(let object) = result, case .string(let lensID)? = object["lensID"],
            case .string(let action)? = object["action"],
            case .bool(let existing)? = object["existing"], case .bool(let changed)? = object["changed"],
            case .array(let files)? = object["files"], let bytes = LensLibrary.whole(object["bytes"]),
            case .string(let digest)? = object["planSHA256"]
        else { return nil }
        var method: String?
        if case .string(let named)? = object["copyMethod"] { method = named }
        var others: [String] = []
        if case .array(let names)? = object["alsoHeldAs"] {
            for case .string(let name) in names { others.append(name) }
        }
        return copyMessage(
            lensID: lensID, action: action, existing: existing, changed: changed,
            restored: object["restored"] == .bool(true), alsoHeldAs: others, fileCount: files.count,
            bytes: bytes, planSHA256: digest, copyMethod: method)
    }

    /// What an import says when the workspace already held the lens. Python
    /// twin: `lens_transfer.import_message`.
    public static func importMessage(lensID: String, restored: Bool) -> String {
        restored
            ? "This workspace already holds these files as lens \(lensID); its converted file was "
                + "missing and was put back."
            : "This workspace already holds these files as lens \(lensID), so nothing was imported."
    }

    /// The sentence for a `science artifact-import` answer, or nil for a new
    /// lens or any other instrument.
    static func importMessage(for result: [String: JSONValue]?) -> String? {
        guard let result, result["kind"] == .string("jlens"), result["existing"] == .bool(true),
            case .string(let lensID)? = result["lensID"]
        else { return nil }
        return importMessage(lensID: lensID, restored: result["restored"] == .bool(true))
    }

    // MARK: Types

    /// A lens, a request, or an upload this client will not send as asked, or
    /// could not finish sending. On the command line it is `refused` (65)
    /// unless `state` says it is a failure (70): an interrupted upload and an
    /// engine that broke the contract declined nothing. The app shows the
    /// reason and the repair together.
    public struct Refusal: Error, Sendable, Equatable, CustomStringConvertible, LocalizedError {
        public let code: String
        public let reason: String
        public let repairAction: String
        public let state: SteerLabCLIState

        public init(code: String, reason: String, repairAction: String, state: SteerLabCLIState = .refused) {
            self.code = code
            self.reason = reason
            self.repairAction = repairAction
            self.state = state
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
        /// The lens record as exact, strict JSON text (see `lensLibraryRegister`).
        var recordJSON: String
        /// How many numbers in the record were not finite, and are null in
        /// `recordJSON`.
        var nonFiniteNumbers: Int

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
            self.nonFiniteNumbers = LensLibrary.whole(object["nonFiniteNumbers"]) ?? 0
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

    /// Replaces `archiveRefersToLibrary` for the tasks under it, so a test can
    /// see whether an upload opened its archive at all.
    @TaskLocal static var archiveInspectorOverride: (@Sendable (URL) -> Bool)?

    /// Whether a file is a request archive whose metadata says it refers to
    /// its lens instead of carrying it (`context.lensReference`). Any other
    /// file, and any archive this cannot read, is not judged here: the server
    /// inspects every upload. Python twin: `lens_transfer.archive_library_lenses`.
    ///
    /// The metadata is the archive's last member, so this reads the whole
    /// archive. An upload asks it only of an engine with no lens library.
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

    /// Where a part's bytes are read from: a place in a file, read forward.
    /// A protocol so a test can see every read.
    protocol PartSource {
        func seek(to offset: UInt64) throws
        func read(upTo count: Int) throws -> Data?
    }

    struct FileSource: PartSource {
        let handle: FileHandle
        func seek(to offset: UInt64) throws { try handle.seek(toOffset: offset) }
        func read(upTo count: Int) throws -> Data? { try handle.read(upToCount: count) }
    }

    /// The most a part upload reads from the lens file at once.
    static let blockSize = 1 << 20

    /// The refusal for a lens file that is not the one that was planned.
    static var changedAfterCheck: Refusal {
        Refusal(code: changedCode, reason: changedReason, repairAction: changedRepair)
    }

    /// Copy one part out of the lens file into `piece`, checking it against
    /// the hash the plan recorded.
    ///
    /// The file is read a block at a time, from the part's first byte to its
    /// last and no further: never the whole lens, and never the whole part.
    /// A file that changed since the plan, or that ends early, is refused
    /// here, so no part of other bytes is ever uploaded.
    static func copyPart(
        _ part: Part, from source: any PartSource, to piece: URL, blockSize: Int = LensLibrary.blockSize
    ) throws {
        guard FileManager.default.createFile(
            atPath: piece.path, contents: nil, attributes: [.posixPermissions: 0o600])
        else { throw CocoaError(.fileWriteUnknown, userInfo: [NSFilePathErrorKey: piece.path]) }
        let target = try FileHandle(forWritingTo: piece)
        defer { try? target.close() }
        try source.seek(to: UInt64(part.offset))
        var hasher = SHA256()
        var remaining = part.bytes
        while remaining > 0 {
            guard let block = try source.read(upTo: min(blockSize, remaining)), !block.isEmpty else {
                throw changedAfterCheck
            }
            hasher.update(data: block)
            try target.write(contentsOf: block)
            remaining -= block.count
        }
        guard hasher.finalize().map({ String(format: "%02x", $0) }).joined() == part.sha256 else {
            throw changedAfterCheck
        }
    }

    static func copyPart(_ part: Part, of file: URL, to piece: URL) throws {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        try copyPart(part, from: FileSource(handle: handle), to: piece)
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
        /// How many numbers in the record were not finite. They are null in
        /// the record the site has.
        public var recordNonFiniteNumbers: Int = 0

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
                "recordNonFiniteNumbers": .number(Double(recordNonFiniteNumbers)),
                "changed": .bool(changed),
            ])
        }

        public var message: String {
            LensLibrary.pushMessage(
                lensID: lensID, uploadedParts: uploadedParts, uploadedBytes: uploadedBytes,
                adopted: adopted, alreadyHeld: alreadyHeld, recordNonFiniteNumbers: recordNonFiniteNumbers)
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

    /// One of a push's calls to the engine. When it gets no answer (the
    /// connection, a timeout, or a failure on the engine), say that running
    /// the command again resumes: what the site received, it keeps, and no
    /// job is involved. A refusal (any 4xx, a missing token among them) is
    /// not an interruption and is thrown as it is. Python twin:
    /// `lens_transfer._send`.
    static func sending<T>(_ lensID: String, _ call: () async throws -> T) async throws -> T {
        try await answered(call) { cause in
            LensLibrary.Refusal(
                code: LensLibrary.interruptedCode,
                reason: LensLibrary.interruptedReason(lensID: lensID, cause: cause),
                repairAction: LensLibrary.resumeRepair, state: .failed)
        }
    }

    /// One of lens-status's reads of the engine: its capabilities, what its
    /// lens library holds, or its record of the lens. When it gets no answer
    /// (the connection, a timeout, or a failure on the engine), say so and
    /// say to run the command again: nothing was changed, and no job is
    /// involved. Any 4xx is an answer, and is thrown as it is. Python twin:
    /// `lens_transfer._read`.
    static func reading<T>(_ lensID: String, _ call: () async throws -> T) async throws -> T {
        try await answered(call) { cause in
            LensLibrary.Refusal(
                code: LensLibrary.statusUnansweredCode,
                reason: LensLibrary.statusUnansweredReason(lensID: lensID, cause: cause),
                repairAction: LensLibrary.statusRetryRepair, state: .failed)
        }
    }

    /// Run one call to the engine. A lost connection, a timeout, or a 5xx
    /// becomes what `unanswered` makes of its cause; anything else is thrown
    /// as it is. The cause can be a whole response body, so no more of it is
    /// shown than of any other refusal's text.
    private static func answered<T>(
        _ call: () async throws -> T, unanswered: (String) -> LensLibrary.Refusal
    ) async throws -> T {
        func shown(_ cause: String) -> String {
            String(String.UnicodeScalarView(cause.unicodeScalars.prefix(LensLibrary.refusalTextLimit)))
        }
        do {
            return try await call()
        } catch let error as URLError {
            throw unanswered(shown(error.localizedDescription))
        } catch let error as ClusterClient.ClientError {
            if case .badResponse(let status, _) = error, status >= 500 { throw unanswered(shown(error.description)) }
            throw error
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
                repairAction: LensLibrary.pushUnavailableRepair)
        }
        var payload: [String: JSONValue] = [
            "workspaceRoot": .string(root.path), "lensID": .string(lensID),
        ]
        let plan: LensLibrary.Plan
        if adopt != nil {
            plan = try LensLibrary.Plan(try await bridge("lens-record", payload))
        } else {
            guard let limit = capabilities.lensLibrary?.partMaxBytes, limit >= 1 else {
                throw LensLibrary.protocolFailure(.partLimit, server: server)
            }
            payload["partMaxBytes"] = .number(Double(limit))
            plan = try LensLibrary.Plan(try await bridge("lens-push-plan", payload))
        }
        let sha = plan.convertedSHA256
        let status = try await sending(lensID) { try await client.lensLibraryStatus(sha) }
        var result = PushResult(
            lensID: lensID, convertedSHA256: sha, bytes: plan.bytes, alreadyHeld: status.held,
            adopted: false, uploadedParts: 0, uploadedBytes: 0,
            recordNonFiniteNumbers: plan.nonFiniteNumbers)
        if status.held, let held = status.bytes, held != plan.bytes {
            throw LensLibrary.Refusal(
                code: LensLibrary.entryDiffersCode,
                reason: LensLibrary.entryDiffersReason(held: held, workspace: plan.bytes),
                repairAction: LensLibrary.entryDiffersRepair)
        }
        if !status.held, let adopt {
            _ = try await sending(lensID) { try await client.lensLibraryAdopt(path: adopt, sha256: sha) }
            result.adopted = true
        } else if !status.held {
            let remote = capabilities.remoteStudy
            guard remote?.externalTransferRequired != true, remote?.httpTransfer != false else {
                throw LensLibrary.Refusal(
                    code: LensLibrary.externalTransferCode,
                    reason: LensLibrary.externalTransferReason(server: server),
                    repairAction: LensLibrary.externalTransferRepair(file: plan.file))
            }
            let begun = try await sending(lensID) {
                try await client.lensLibraryBegin(sha, bytes: plan.bytes, parts: plan.parts)
            }
            if begun.held {
                result.alreadyHeld = true
            } else {
                let indices = Set(plan.parts.map(\.index))
                guard let missing = begun.missingParts, Set(missing).isSubset(of: indices) else {
                    throw LensLibrary.protocolFailure(.missingParts)
                }
                let wanted = Set(missing)
                let file = root.appending(path: plan.file)
                for part in plan.parts where wanted.contains(part.index) {
                    try await sending(lensID) { try await client.lensLibraryPutPart(sha, part: part, of: file) }
                    result.uploadedParts += 1
                    result.uploadedBytes += part.bytes
                }
                let done = try await sending(lensID) { try await client.lensLibraryComplete(sha) }
                guard done.held, done.bytes == plan.bytes else {
                    throw LensLibrary.protocolFailure(.notConfirmed)
                }
            }
        }
        let registration = try await sending(lensID) {
            try await client.lensLibraryRegister(recordJSON: plan.recordJSON)
        }
        guard registration.lensID == lensID else { throw LensLibrary.protocolFailure(.otherLens) }
        result.registered = registration.registered
        result.recordChanged = registration.changed
        return result
    }

    /// Whether the site holds a workspace lens's bytes, and whether it has
    /// the lens's record. Reads only: the engine's capabilities, what its
    /// lens library holds, and its record of the lens.
    public static func status(
        _ lensID: String, client: ClusterClient, root: URL, bridge: Bridge = localBridge
    ) async throws -> StatusResult {
        let plan = try LensLibrary.Plan(
            try await bridge(
                "lens-record", ["workspaceRoot": .string(root.path), "lensID": .string(lensID)]))
        var result = StatusResult(
            lensID: lensID, convertedSHA256: plan.convertedSHA256, bytes: plan.bytes,
            libraryAvailable: try await reading(lensID) { try await client.capabilities() }.lensLibraryAvailable,
            held: false, receivedParts: nil, siteRecord: "unknown")
        if result.libraryAvailable {
            let held = try await reading(lensID) { try await client.lensLibraryStatus(plan.convertedSHA256) }
            result.held = held.held
            result.receivedParts = held.receivedParts
        }
        let site = try await reading(lensID) { try await client.siteLensRecord(lensID) }
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
