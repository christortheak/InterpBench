import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// The Mac's half of the site lens library (public issues #13 and #14): a lens
/// is sent once, in parts; a request can then refer to it; an upload is
/// streamed from its file; and a lens can be copied into a second workspace.
///
/// The server here is a small stand-in that speaks the lens-library routes as
/// the engine does, so every expectation is about what this CLIENT sends and
/// says. The sentences are held to the Python client's by one file,
/// `prompts/fixtures/lens-library/refusals.json`. Python twin:
/// `Server/tests/test_lens_transfer.py`.
@Suite(.serialized) struct LensLibraryTests {

    // MARK: - Fixtures

    private static var repository: URL {
        URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    private static func fixture() throws -> [String: JSONValue] {
        let url = repository.appending(path: "prompts/fixtures/lens-library/refusals.json")
        guard case .object(let object) = try JSONDecoder().decode(
            JSONValue.self, from: Data(contentsOf: url))
        else { throw ExperimentError(reason: "the lens-library fixture is not an object") }
        return object
    }

    private static func text(_ value: JSONValue?, _ key: String) throws -> String {
        guard case .object(let object)? = value, case .string(let text)? = object[key] else {
            throw ExperimentError(reason: "the lens-library fixture has no \(key)")
        }
        return text
    }

    private static func rows(_ fixture: [String: JSONValue], _ key: String) throws -> [[String: JSONValue]] {
        guard case .array(let items)? = fixture[key] else {
            throw ExperimentError(reason: "the lens-library fixture has no \(key)")
        }
        return items.compactMap { if case .object(let row) = $0 { row } else { nil } }
    }

    private static let server = "http://runner.example.invalid:8080"
    private static let lensID = "custom-lens-example"

    private static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// A workspace holding one 250-byte "lens file", and the plan the Python
    /// client would give for it at 100 bytes a part. The record carries an
    /// integer no double can hold, and a decimal.
    private struct Lens {
        var root: URL
        var data: Data
        var sha: String
        var parts: [LensLibrary.Part]
        var recordJSON: String
        var nonFiniteNumbers = 0
        var file: String { "runs/jlens-lenses/\(LensLibraryTests.lensID)/jacobians.safetensors" }
        var url: URL { root.appending(path: file) }

        init() throws {
            root = FileManager.default.temporaryDirectory
                .appending(component: "lens-library-\(UUID().uuidString)").resolvingSymlinksInPath()
            data = Data((0..<250).map { UInt8(($0 * 7 + 3) % 251) })
            sha = LensLibraryTests.sha256(data)
            let folder = root.appending(path: "runs/jlens-lenses/\(LensLibraryTests.lensID)")
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try data.write(to: folder.appending(component: "jacobians.safetensors"))
            let bytes = data
            parts = [(0, 100), (100, 100), (200, 50)].enumerated().map { index, range in
                LensLibrary.Part(
                    index: index, offset: range.0, bytes: range.1,
                    sha256: LensLibraryTests.sha256(bytes.subdata(in: range.0..<(range.0 + range.1))))
            }
            recordJSON = #"{"converted": {"path": "runs/jlens-lenses/custom-lens-example/jacobians.safetensors", "sha256": ""#
                + sha + #""}, "lensID": "custom-lens-example", "qualifications": [{"checks": {"seed": 18446744073709551615, "tolerance": 0.1}}], "source": {"tensorSHA256": ""#
                + String(repeating: "b", count: 64) + #""}}"#
        }

        func remove() { try? FileManager.default.removeItem(at: root) }

        func answer(parts included: Bool) -> JSONValue {
            var object: [String: JSONValue] = [
                "lensID": .string(LensLibraryTests.lensID), "convertedSHA256": .string(sha),
                "tensorSHA256": .string(String(repeating: "b", count: 64)),
                "bytes": .number(Double(data.count)), "file": .string(file),
                "recordJSON": .string(recordJSON), "nonFiniteNumbers": .number(Double(nonFiniteNumbers)),
                "changed": .bool(false),
            ]
            if included {
                object["parts"] = .array(parts.map {
                    .object([
                        "index": .number(Double($0.index)), "offset": .number(Double($0.offset)),
                        "bytes": .number(Double($0.bytes)), "sha256": .string($0.sha256),
                    ])
                })
            }
            return .object(object)
        }
    }

    /// The Python client's local actions, recorded.
    private final class Bridge: @unchecked Sendable {
        private let lock = NSLock()
        private var seen: [(String, [String: JSONValue])] = []
        let lens: Lens
        init(_ lens: Lens) { self.lens = lens }
        var calls: [(String, [String: JSONValue])] { lock.lock(); defer { lock.unlock() }; return seen }
        private func record(_ action: String, _ payload: [String: JSONValue]) {
            lock.lock()
            seen.append((action, payload))
            lock.unlock()
        }
        var run: LensLibraryRemote.Bridge {
            { [self] action, payload in
                record(action, payload)
                return lens.answer(parts: action == "lens-push-plan")
            }
        }
    }

    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var total = 0
        func bump() { lock.lock(); total += 1; lock.unlock() }
        var count: Int { lock.lock(); defer { lock.unlock() }; return total }
    }

    /// A server's lens library, in memory, speaking its routes as the engine does.
    private final class Site: @unchecked Sendable {
        private let lock = NSLock()
        var library: Bool? = true
        var partMax = 100
        var external = false
        var servesRecords = true
        /// Answer one part with a 500, or with no answer at all.
        var failPart: Int?
        var dropPart: Int?
        var failComplete = false
        /// An answer the contract does not allow, by route name.
        var answers: [String: [String: Any]] = [:]
        var entries: [String: Data] = [:]
        var manifests: [String: [[String: Any]]] = [:]
        var received: [String: [Int: Data]] = [:]
        var registered: [String: [String: Any]] = [:]
        var registerBodies: [Data] = []
        var adoptable: [String: Data] = [:]
        var adoptions: [[String: Any]] = []
        var uploads: [(bytes: Int, footprint: UInt64)] = []
        var uploadStatus = 200
        var calls: [String] = []

        func sent(_ method: String, containing fragment: String = "") -> [String] {
            calls.filter { $0.hasPrefix(method + " ") && (fragment.isEmpty || $0.contains(fragment)) }
        }

        private func json(_ object: Any, _ status: Int = 200) throws -> (Data, Int) {
            (try JSONSerialization.data(withJSONObject: object), status)
        }

        /// A refusal as the engine's lens-library routes write one.
        private func refusal(_ status: Int, _ code: String, _ sentence: String, _ repair: String) throws
            -> (Data, Int)
        {
            try json(
                ["detail": ["code": code, "reason": sentence, "message": sentence, "repairAction": repair]],
                status)
        }

        func respond(_ request: URLRequest) throws -> (Data, Int) {
            lock.lock()
            defer { lock.unlock() }
            let path = request.url?.path ?? ""
            let method = request.httpMethod ?? "GET"
            calls.append("\(method) \(path)")
            if path == "/api/capabilities" {
                var document: [String: Any] = [
                    "remoteStudy": ["httpTransfer": !external, "externalTransferRequired": external]
                ]
                if let library {
                    document["lensLibrary"] = [
                        "schemaVersion": 1, "available": library, "partMaxBytes": partMax,
                    ]
                }
                return try json(document)
            }
            if path == "/api/bundles/upload" {
                // Measured before the body is read: what the client holds
                // in memory while the upload is in flight.
                let footprint = LensLibraryTests.footprint()
                let count = LensLibraryTests.drain(request)
                uploads.append((count, footprint))
                if uploadStatus != 200 {
                    return try json(
                        ["detail": "upload exceeds 4294967296 bytes (STEERLAB_MAX_UPLOAD_BYTES)"],
                        uploadStatus)
                }
                return try json([
                    "path": "/staged/x.tar.gz", "filename": "x.tar.gz", "sha256": "", "bytes": count,
                    "executable": true, "stagingDirectory": "/staged",
                ])
            }
            if path.hasPrefix("/api/jlens/lenses/") {
                guard servesRecords else {
                    return try json(["detail": ["code": "workbench_required", "message": "runner only"]], 403)
                }
                guard let record = registered[String(path.dropFirst("/api/jlens/lenses/".count))] else {
                    return try json(["detail": "no such lens"], 404)
                }
                return try json(record)
            }
            guard library == true else {
                return try refusal(409, "lensLibraryUnavailable", "This engine has no lens library.", "Configure one.")
            }
            let body = LensLibraryTests.body(request)
            if path == "/api/lens-library/register" {
                registerBodies.append(body)
                // Strict JSON, as the engine reads it: a bare NaN is not JSON.
                guard let document = try? JSONSerialization.jsonObject(with: body) as? [String: Any],
                    let record = document["record"] as? [String: Any], let lensID = record["lensID"] as? String
                else { return try refusal(400, "lensRecordMalformed", "The record is not JSON.", "Send strict JSON.") }
                if let answer = answers["register"] { return try json(answer) }
                let known = registered[lensID].map { NSDictionary(dictionary: $0).isEqual(to: record) } ?? false
                registered[lensID] = record
                return try json(["lensID": lensID, "registered": true, "changed": !known])
            }
            if path == "/api/lens-library/adopt" {
                let document = try JSONSerialization.jsonObject(with: body) as! [String: Any]
                adoptions.append(document)
                guard let data = adoptable[document["path"] as! String] else {
                    return try refusal(409, "lensAdoptMissing", "No file is at that path.", "Copy the file there first.")
                }
                entries[document["sha256"] as! String] = data
                return try json(["held": true, "bytes": data.count])
            }
            let pieces = path.split(separator: "/").map(String.init)
            guard pieces.count >= 3, pieces[0] == "api", pieces[1] == "lens-library" else {
                return try json(["detail": "no such route"], 404)
            }
            let sha = pieces[2]
            if pieces.count == 3 {
                return try json([
                    "held": entries[sha] != nil, "bytes": entries[sha].map { $0.count as Any } ?? NSNull(),
                    "receivedParts": received[sha].map { $0.keys.sorted() as Any } ?? NSNull(),
                ])
            }
            switch pieces[3] {
            case "begin":
                if entries[sha] != nil { return try json(["held": true, "missingParts": [Int]()]) }
                let manifest = try JSONSerialization.jsonObject(with: body) as! [String: Any]
                let parts = manifest["parts"] as! [[String: Any]]
                var offset = 0
                for (position, part) in parts.enumerated() {
                    // Whole numbers on the wire, in order, and tiling the file.
                    guard part["index"] as? Int == position, part["offset"] as? Int == offset,
                        let bytes = part["bytes"] as? Int, bytes > 0, bytes <= partMax
                    else {
                        return try refusal(
                            400, "lensUploadPlanMalformed", "The parts do not tile the file.", "Plan it again.")
                    }
                    offset += bytes
                }
                guard offset == manifest["bytes"] as? Int else {
                    return try refusal(
                        400, "lensUploadPlanMalformed", "The parts do not tile the file.", "Plan it again.")
                }
                if let answer = answers["begin"] { return try json(answer) }
                manifests[sha] = parts
                if received[sha] == nil { received[sha] = [:] }
                let missing = parts.compactMap { $0["index"] as? Int }.filter { received[sha]![$0] == nil }
                return try json(["held": false, "missingParts": missing])
            case "parts":
                let index = Int(pieces[4])!
                if dropPart == index {
                    dropPart = nil
                    throw URLError(.timedOut)
                }
                if failPart == index {
                    failPart = nil
                    return try json(["detail": "the connection dropped"], 500)
                }
                let part = manifests[sha]![index]
                guard request.value(forHTTPHeaderField: "X-Content-SHA256") == part["sha256"] as? String,
                    LensLibraryTests.sha256(body) == part["sha256"] as? String, body.count == part["bytes"] as? Int
                else {
                    return try refusal(409, "lensUploadPartCorrupt", "The part differs from its plan.", "Send it again.")
                }
                received[sha]![index] = body
                return try json(["received": true])
            case "complete":
                if failComplete {
                    failComplete = false
                    return try json(["detail": "the engine stopped"], 500)
                }
                if let answer = answers["complete"] { return try json(answer) }
                if let held = entries[sha] { return try json(["held": true, "bytes": held.count]) }
                let parts = manifests[sha]!
                let data = parts.reduce(into: Data()) { $0.append(received[sha]![$1["index"] as! Int]!) }
                guard LensLibraryTests.sha256(data) == sha else {
                    return try refusal(409, "lensUploadCorrupt", "The whole file differs.", "Send it again.")
                }
                entries[sha] = data
                manifests[sha] = nil
                received[sha] = nil
                return try json(["held": true, "bytes": data.count])
            default:
                return try json(["detail": "no such route"], 404)
            }
        }

        func session() -> URLSession {
            LensStubURLProtocol.handler = { [self] request in try respond(request) }
            let configuration = URLSessionConfiguration.ephemeral
            configuration.protocolClasses = [LensStubURLProtocol.self]
            return URLSession(configuration: configuration)
        }

        func client() -> ClusterClient {
            ClusterClient(
                profile: .init(baseURL: URL(string: LensLibraryTests.server)!), session: session())
        }
    }

    fileprivate static func body(_ request: URLRequest) -> Data {
        if let body = request.httpBody { return body }
        guard let stream = request.httpBodyStream else { return Data() }
        stream.open()
        defer { stream.close() }
        var data = Data()
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: 1 << 16)
        defer { buffer.deallocate() }
        while true {
            let count = stream.read(buffer, maxLength: 1 << 16)
            if count <= 0 { break }
            data.append(buffer, count: count)
        }
        return data
    }

    /// Read a body to its end without keeping it; returns its length.
    fileprivate static func drain(_ request: URLRequest) -> Int {
        if let body = request.httpBody { return body.count }
        guard let stream = request.httpBodyStream else { return 0 }
        stream.open()
        defer { stream.close() }
        var total = 0
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: 1 << 20)
        defer { buffer.deallocate() }
        while true {
            let count = stream.read(buffer, maxLength: 1 << 20)
            if count <= 0 { break }
            total += count
        }
        return total
    }

    /// This process's memory footprint, in bytes.
    fileprivate static func footprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? UInt64(info.phys_footprint) : 0
    }

    /// A request archive with this metadata context, as `science package`
    /// writes one: the members, then `steerlab-bundle.json` last.
    private static func archive(in directory: URL, name: String, context: [String: Any]) throws -> URL {
        let payload = directory.appending(component: name + "-payload")
        try FileManager.default.createDirectory(
            at: payload.appending(component: "prompts"), withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: payload.appending(path: "prompts/request.json"))
        let metadata: [String: Any] = [
            "schemaVersion": 1, "kind": "diagnosticInput", "context": context, "entries": [Any](),
        ]
        try JSONSerialization.data(withJSONObject: metadata)
            .write(to: payload.appending(component: "steerlab-bundle.json"))
        let output = directory.appending(component: name + ".tar.gz")
        let process = Process()
        process.executableURL = URL(filePath: "/usr/bin/tar")
        process.currentDirectoryURL = payload
        process.arguments = ["-czf", output.path, "prompts/request.json", "steerlab-bundle.json"]
        try process.run()
        process.waitUntilExit()
        #expect(process.terminationStatus == 0)
        return output
    }

    private static func temporary() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "lens-library-\(UUID().uuidString)").resolvingSymlinksInPath()
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    // MARK: - The sentences

    @Test func theSentencesAreThePythonClientsWordForWord() throws {
        let fixture = try Self.fixture()
        guard case .string(let server)? = fixture["server"], case .string(let file)? = fixture["file"]
        else { Issue.record("the fixture names no server or file"); return }
        #expect(LensLibrary.unavailableCode == (try Self.text(fixture["unavailable"], "code")))
        #expect(LensLibrary.unavailableReason(server: server)
            == (try Self.text(fixture["unavailable"], "pushReason")))
        #expect(LensLibrary.pushUnavailableRepair == (try Self.text(fixture["unavailable"], "pushRepairAction")))
        #expect(LensLibrary.unavailableReason(server: server, archive: true)
            == (try Self.text(fixture["unavailable"], "archiveReason")))
        #expect(LensLibrary.archiveUnavailableRepair
            == (try Self.text(fixture["unavailable"], "archiveRepairAction")))
        #expect(LensLibrary.externalTransferCode == (try Self.text(fixture["externalTransfer"], "code")))
        #expect(LensLibrary.externalTransferReason(server: server)
            == (try Self.text(fixture["externalTransfer"], "reason")))
        #expect(LensLibrary.externalTransferRepair(file: file)
            == (try Self.text(fixture["externalTransfer"], "repairAction")))
        #expect(LensLibrary.oversizeAdvice == (try Self.text(fixture["oversize"], "advice")))
        #expect(LensLibrary.oversizeRepair == (try Self.text(fixture["oversize"], "macRepair")))

        // The site holds something else under the lens's hash.
        guard case .object(let entry)? = fixture["entryDiffers"], let held = LensLibrary.whole(entry["heldBytes"]),
            let workspace = LensLibrary.whole(entry["workspaceBytes"])
        else { Issue.record("the fixture has no entryDiffers"); return }
        #expect(LensLibrary.entryDiffersCode == (try Self.text(fixture["entryDiffers"], "code")))
        #expect(LensLibrary.entryDiffersReason(held: held, workspace: workspace)
            == (try Self.text(fixture["entryDiffers"], "reason")))
        #expect(LensLibrary.entryDiffersRepair == (try Self.text(fixture["entryDiffers"], "repairAction")))

        // The four answers an engine should never give: one sentence and one repair each.
        #expect(LensLibrary.protocolCode == (try Self.text(fixture["protocol"], "code")))
        guard case .object(let protocolBlock)? = fixture["protocol"], case .array(let cases)? = protocolBlock["cases"]
        else { Issue.record("the fixture has no protocol cases"); return }
        #expect(cases.count == LensLibrary.ProtocolProblem.allCases.count)
        for item in cases {
            guard let name = try? Self.text(item, "kind"), let kind = LensLibrary.ProtocolProblem(rawValue: name)
            else { Issue.record("an unknown protocol case"); continue }
            let problem = LensLibrary.protocolProblem(kind, server: server)
            #expect(problem.reason == (try Self.text(item, "reason")), "\(kind)")
            #expect(problem.repair == (try Self.text(item, "repairAction")), "\(kind)")
            let failure = LensLibrary.protocolFailure(kind, server: server)
            #expect(failure.state == .failed && failure.code == LensLibrary.protocolCode)
        }

        // An upload that did not finish, and a file that is not the plan's.
        #expect(LensLibrary.interruptedCode == (try Self.text(fixture["interrupted"], "code")))
        #expect(LensLibrary.interruptedReason(
            lensID: try Self.text(fixture["interrupted"], "lensID"), cause: try Self.text(fixture["interrupted"], "cause"))
            == (try Self.text(fixture["interrupted"], "reason")))
        #expect(LensLibrary.resumeRepair == (try Self.text(fixture["interrupted"], "repairAction")))
        #expect(LensLibrary.changedCode == (try Self.text(fixture["changed"], "code")))
        #expect(LensLibrary.changedReason == (try Self.text(fixture["changed"], "reason")))
        #expect(LensLibrary.changedRepair == (try Self.text(fixture["changed"], "repairAction")))

        // The verbs' purposes are the Python client's.
        #expect(ExperimentCLIParser.lensReferencePurpose == (try Self.text(fixture["purposes"], "lensReference")))
        #expect(ExperimentCLIParser.lensCopyPurpose == (try Self.text(fixture["purposes"], "lensCopy")))
        #expect(ExperimentCLIParser.lensPushPurpose == (try Self.text(fixture["purposes"], "lensPush")))
        #expect(ExperimentCLIParser.lensStatusPurpose == (try Self.text(fixture["purposes"], "lensStatus")))
        #expect(ExperimentCLIParser.artifactImportPurpose == (try Self.text(fixture["purposes"], "artifactImport")))
        func purpose(_ namespace: String, _ verb: String) -> String? {
            ExperimentCLIParser.spec(namespace: namespace, verb: verb)?.purpose
        }
        #expect(purpose("remote", "lens-push") == ExperimentCLIParser.lensPushPurpose)
        #expect(purpose("remote", "lens-status") == ExperimentCLIParser.lensStatusPurpose)
        #expect(purpose("science", "lens-copy") == ExperimentCLIParser.lensCopyPurpose)
        #expect(purpose("science", "artifact-import") == ExperimentCLIParser.artifactImportPurpose)
        for verb in ["input-plan", "package"] {
            #expect(purpose("science", verb)?.hasSuffix(ExperimentCLIParser.lensReferencePurpose) == true)
        }
    }

    @Test func theMessagesAreThePythonClientsWordForWord() throws {
        let fixture = try Self.fixture()
        let pushes = try Self.rows(fixture, "pushMessages")
        let statuses = try Self.rows(fixture, "statusMessages")
        let copies = try Self.rows(fixture, "copyMessages")
        let imports = try Self.rows(fixture, "importMessages")
        #expect(pushes.count == 6 && statuses.count == 5 && copies.count == 8 && imports.count == 2)
        for row in pushes {
            guard case .string(let lens)? = row["lensID"], let parts = LensLibrary.whole(row["uploadedParts"]),
                let bytes = LensLibrary.whole(row["uploadedBytes"]), case .bool(let adopted)? = row["adopted"],
                case .bool(let already)? = row["alreadyHeld"], let nulls = LensLibrary.whole(row["recordNonFiniteNumbers"])
            else { Issue.record("malformed push row"); continue }
            #expect(.string(LensLibrary.pushMessage(
                lensID: lens, uploadedParts: parts, uploadedBytes: bytes, adopted: adopted, alreadyHeld: already,
                recordNonFiniteNumbers: nulls)) == row["message"])
        }
        for row in statuses {
            guard case .string(let lens)? = row["lensID"], case .bool(let available)? = row["libraryAvailable"],
                case .bool(let held)? = row["held"], case .string(let record)? = row["siteRecord"]
            else { Issue.record("malformed status row"); continue }
            #expect(.string(LensLibrary.statusMessage(
                lensID: lens, libraryAvailable: available, held: held, siteRecord: record)) == row["message"])
        }
        for row in copies {
            guard case .string(let lens)? = row["lensID"], case .string(let action)? = row["action"],
                case .bool(let existing)? = row["existing"], case .bool(let changed)? = row["changed"],
                case .bool(let restored)? = row["restored"], let files = LensLibrary.whole(row["fileCount"]),
                let bytes = LensLibrary.whole(row["bytes"]), case .string(let digest)? = row["planSHA256"]
            else { Issue.record("malformed copy row"); continue }
            var method: String?
            if case .string(let named)? = row["copyMethod"] { method = named }
            var other: String?
            if case .string(let named)? = row["existingLensID"] { other = named }
            #expect(.string(LensLibrary.copyMessage(
                lensID: lens, action: action, existing: existing, changed: changed, restored: restored,
                existingLensID: other, fileCount: files, bytes: bytes, planSHA256: digest, copyMethod: method))
                == row["message"])
            // …and from the answer the Python client gives the command line,
            // where a field that does not apply is absent.
            var answer = row.filter { $0.value != .null }
            answer["files"] = .array(Array(repeating: .null, count: files))
            if !restored { answer["restored"] = nil }
            #expect(LensLibrary.copyMessage(for: .object(answer)).map(JSONValue.string) == row["message"])
        }
        for row in imports {
            guard case .string(let lens)? = row["lensID"], case .bool(let restored)? = row["restored"]
            else { Issue.record("malformed import row"); continue }
            #expect(.string(LensLibrary.importMessage(lensID: lens, restored: restored)) == row["message"])
            var answer: [String: JSONValue] = ["kind": .string("jlens"), "existing": .bool(true), "lensID": .string(lens)]
            if restored { answer["restored"] = .bool(true) }
            #expect(LensLibrary.importMessage(for: answer).map(JSONValue.string) == row["message"])
        }
        // A new lens, and any other instrument, keep the family's sentence.
        #expect(LensLibrary.importMessage(for: ["kind": .string("jlens"), "lensID": .string("x"), "changed": .bool(true)]) == nil)
        #expect(LensLibrary.importMessage(for: ["kind": .string("sae-decoder"), "existing": .bool(true)]) == nil)
        #expect(LensLibrary.importMessage(for: nil) == nil)
    }

    @Test func theCapabilityIsReadAndAnEngineWithoutItHasNoLibrary() throws {
        let decoder = JSONDecoder()
        let present = try decoder.decode(
            ClusterCapabilities.self,
            from: Data(#"{"lensLibrary":{"schemaVersion":1,"available":true,"partMaxBytes":268435456}}"#.utf8))
        #expect(present.lensLibraryAvailable)
        #expect(present.lensLibrary?.partMaxBytes == 268_435_456)
        let unconfigured = try decoder.decode(
            ClusterCapabilities.self,
            from: Data(#"{"lensLibrary":{"schemaVersion":1,"available":false,"partMaxBytes":268435456}}"#.utf8))
        #expect(!unconfigured.lensLibraryAvailable)
        let older = try decoder.decode(ClusterCapabilities.self, from: Data("{}".utf8))
        #expect(older.lensLibrary == nil && !older.lensLibraryAvailable)
    }

    // MARK: - Sending a lens

    @Test func aFreshPushSendsEveryPartOnceInOrderAndRegistersTheExactRecord() async throws {
        let lens = try Lens()
        defer { lens.remove() }
        let site = Site()
        let bridge = Bridge(lens)
        let result = try await LensLibraryRemote.push(
            Self.lensID, client: site.client(), root: lens.root, bridge: bridge.run)

        #expect(site.entries == [lens.sha: lens.data])
        #expect(result == .init(
            lensID: Self.lensID, convertedSHA256: lens.sha, bytes: 250, alreadyHeld: false,
            adopted: false, uploadedParts: 3, uploadedBytes: 250, registered: true, recordChanged: true))
        #expect(result.changed)
        // The contract's order: the capability, what is held, the manifest,
        // the parts, the whole, the record.
        #expect(site.calls == [
            "GET /api/capabilities", "GET /api/lens-library/\(lens.sha)",
            "POST /api/lens-library/\(lens.sha)/begin",
            "PUT /api/lens-library/\(lens.sha)/parts/0", "PUT /api/lens-library/\(lens.sha)/parts/1",
            "PUT /api/lens-library/\(lens.sha)/parts/2",
            "POST /api/lens-library/\(lens.sha)/complete", "POST /api/lens-library/register",
        ])
        // The plan was asked for with the server's part limit, once.
        #expect(bridge.calls.map(\.0) == ["lens-push-plan"])
        #expect(bridge.calls.first?.1 == [
            "workspaceRoot": .string(lens.root.path), "lensID": .string(Self.lensID),
            "partMaxBytes": .number(100),
        ])
        // The record arrives as the exact text the Python client read: an
        // integer no double can hold is still that integer.
        #expect(site.registerBodies == [Data((#"{"record":"# + lens.recordJSON + "}").utf8)])
        #expect(String(decoding: site.registerBodies.first ?? Data(), as: UTF8.self)
            .contains("18446744073709551615"))
        #expect(result.message == "Sent lens custom-lens-example to the site's lens library "
            + "(3 parts, 250 bytes); the site has its record.")
        #expect(result.json == .object([
            "lensID": .string(Self.lensID), "convertedSHA256": .string(lens.sha), "bytes": .number(250),
            "held": .bool(true), "alreadyHeld": .bool(false), "adopted": .bool(false),
            "uploadedParts": .number(3), "uploadedBytes": .number(250), "registered": .bool(true),
            "recordChanged": .bool(true), "recordNonFiniteNumbers": .number(0), "changed": .bool(true),
        ]))
    }

    /// A part or the whole-file check that gets no answer: one sentence, the
    /// Python client's, saying that running the command again sends only
    /// what the site still lacks. Then it does.
    @Test(arguments: ["a 500 on a part", "no answer to a part", "a 500 on the whole-file check"])
    func anInterruptedPushSaysToRunItAgainAndThenSendsOnlyWhatIsMissing(_ failure: String) async throws {
        let lens = try Lens()
        defer { lens.remove() }
        let site = Site()
        switch failure {
        case "a 500 on a part": site.failPart = 1
        case "no answer to a part": site.dropPart = 1
        default: site.failComplete = true
        }
        let bridge = Bridge(lens)
        let client = site.client()
        do {
            _ = try await LensLibraryRemote.push(Self.lensID, client: client, root: lens.root, bridge: bridge.run)
            Issue.record("an interrupted push must not succeed")
        } catch let stopped as LensLibrary.Refusal {
            #expect(stopped.code == "lensUploadInterrupted" && stopped.state == .failed)
            #expect(stopped.repairAction == (try Self.text(try Self.fixture()["interrupted"], "repairAction")))
            #expect(stopped.reason.hasPrefix("Sending lens custom-lens-example to the site was interrupted "
                + "before the site confirmed that it holds the whole lens ("))
            #expect(!(stopped.reason + stopped.repairAction).contains("job"))
        }
        #expect(site.entries.isEmpty && site.registered.isEmpty)
        let received = site.received[lens.sha]?.keys.sorted() ?? []
        #expect(received == (failure == "a 500 on the whole-file check" ? [0, 1, 2] : [0]))
        site.calls.removeAll()

        let result = try await LensLibraryRemote.push(
            Self.lensID, client: client, root: lens.root, bridge: bridge.run)
        let missing = [0, 1, 2].filter { !received.contains($0) }
        #expect(site.sent("PUT") == missing.map { "PUT /api/lens-library/\(lens.sha)/parts/\($0)" })
        #expect(result.uploadedParts == missing.count && !result.alreadyHeld)
        #expect(site.entries == [lens.sha: lens.data] && site.registered[Self.lensID] != nil)
        if missing.isEmpty {
            #expect(result.message.hasPrefix("The site had already received every part of lens "
                + "custom-lens-example and now holds it"))
        }
    }

    @Test func aLensTheSiteHoldsUploadsNothingAndStillRegisters() async throws {
        let lens = try Lens()
        defer { lens.remove() }
        let site = Site()
        site.entries[lens.sha] = lens.data      // fitted on the site, or sent from another workspace
        let bridge = Bridge(lens)
        let client = site.client()
        let result = try await LensLibraryRemote.push(
            Self.lensID, client: client, root: lens.root, bridge: bridge.run)
        #expect(site.sent("PUT").isEmpty && site.sent("POST", containing: "begin").isEmpty
            && site.sent("POST", containing: "complete").isEmpty)
        #expect(result.alreadyHeld && result.uploadedParts == 0 && result.uploadedBytes == 0)
        #expect(result.registered == true && result.changed)    // the record was new to the site
        #expect(site.registered[Self.lensID] != nil)
        #expect(result.message == "The site's lens library already holds lens custom-lens-example, "
            + "so nothing was uploaded; the site has its record.")
        let again = try await LensLibraryRemote.push(
            Self.lensID, client: client, root: lens.root, bridge: bridge.run)
        #expect(!again.changed && again.recordChanged == false)
    }

    /// Older (the capability is absent) or unconfigured (it is false): one
    /// sentence, the Python client's, and no request but the capability read.
    @Test(arguments: [nil, false] as [Bool?])
    func anEngineWithNoLensLibraryIsSentNothing(_ library: Bool?) async throws {
        let lens = try Lens()
        defer { lens.remove() }
        let site = Site()
        site.library = library
        let bridge = Bridge(lens)
        do {
            _ = try await LensLibraryRemote.push(
                Self.lensID, client: site.client(), root: lens.root, bridge: bridge.run)
            Issue.record("an engine with no lens library must be refused")
        } catch let refusal as LensLibrary.Refusal {
            let fixture = try Self.fixture()
            #expect(refusal.code == "lensLibraryUnavailable" && refusal.state == .refused)
            #expect(refusal.reason == (try Self.text(fixture["unavailable"], "pushReason")))
            // A push packages nothing, so its repair does not talk about packaging a request.
            #expect(refusal.repairAction == (try Self.text(fixture["unavailable"], "pushRepairAction")))
            #expect(!refusal.repairAction.contains("--lens-reference"))
        }
        #expect(site.calls == ["GET /api/capabilities"])
        #expect(bridge.calls.isEmpty, "nothing is hashed for an engine that cannot take it")
    }

    @Test func aSiteThatForbidsHTTPTransferAdoptsACopyAlreadyOnIt() async throws {
        let lens = try Lens()
        defer { lens.remove() }
        let site = Site()
        site.external = true
        let bridge = Bridge(lens)
        let client = site.client()
        do {
            _ = try await LensLibraryRemote.push(Self.lensID, client: client, root: lens.root, bridge: bridge.run)
            Issue.record("a site that forbids HTTP transfer must be refused")
        } catch let refusal as LensLibrary.Refusal {
            #expect(refusal.code == "externalTransferRequired")
            #expect(refusal.reason == LensLibrary.externalTransferReason(server: Self.server))
            #expect(refusal.repairAction == LensLibrary.externalTransferRepair(file: lens.file))
        }
        #expect(site.sent("POST").isEmpty && site.sent("PUT").isEmpty)

        site.adoptable["/scratch/incoming/jacobians.safetensors"] = lens.data
        let result = try await LensLibraryRemote.push(
            Self.lensID, adopt: "/scratch/incoming/jacobians.safetensors", client: client,
            root: lens.root, bridge: bridge.run)
        #expect(result.adopted && result.uploadedParts == 0 && result.changed)
        #expect(site.entries == [lens.sha: lens.data] && site.registered[Self.lensID] != nil)
        #expect(site.sent("PUT").isEmpty && site.sent("POST", containing: "begin").isEmpty)
        #expect(site.adoptions.count == 1)
        #expect(site.adoptions.first?["path"] as? String == "/scratch/incoming/jacobians.safetensors")
        #expect(site.adoptions.first?["sha256"] as? String == lens.sha)
        // An adoption hashes nothing here: the site checks its own copy.
        #expect(bridge.calls.map(\.0) == ["lens-push-plan", "lens-record"])
        #expect(result.message == "The site checked its copy of lens custom-lens-example and keeps "
            + "it in its lens library; the site has its record.")
    }

    /// The engine's own code, sentence, and repair, as the Python client
    /// reports them too: from `message`, or from `reason` where a route
    /// gives only that.
    @Test func aServersRefusalArrivesWithItsOwnCodeAndRepair() async throws {
        let lens = try Lens()
        defer { lens.remove() }
        let site = Site()
        site.partMax = 50          // the plan's parts are 100 bytes: the manifest is refused
        do {
            _ = try await LensLibraryRemote.push(
                Self.lensID, client: site.client(), root: lens.root, bridge: Bridge(lens).run)
            Issue.record("a manifest the server refuses must be refused")
        } catch let refusal as LensLibrary.Refusal {
            #expect(refusal.code == "lensUploadPlanMalformed" && refusal.state == .refused)
            #expect(refusal.reason == "The parts do not tile the file.")
            #expect(refusal.repairAction == "Plan it again.")
        }
        #expect(site.sent("PUT").isEmpty)

        let onlyReason = ClusterClient.refusalDetail(Data(
            #"{"detail": {"code": "lensRecordDiffers", "reason": "The site has another record.", "repairAction": "Compare the two records."}}"#.utf8))
        #expect(onlyReason.code == "lensRecordDiffers" && onlyReason.message == "The site has another record."
            && onlyReason.repair == "Compare the two records.")
        let bare = ClusterClient.refusalDetail(Data(#"{"detail": "the body is too large"}"#.utf8))
        #expect(bare.code == nil && bare.message == "the body is too large" && bare.repair == nil)
    }

    @Test func anEntryOfAnotherSizeAndAnAnswerTheContractDoesNotAllow() async throws {
        let lens = try Lens()
        defer { lens.remove() }
        let other = Site()
        other.entries[lens.sha] = lens.data + Data([33])
        do {
            _ = try await LensLibraryRemote.push(
                Self.lensID, client: other.client(), root: lens.root, bridge: Bridge(lens).run)
            Issue.record("an entry of another size must be refused")
        } catch let refusal as LensLibrary.Refusal {
            #expect(refusal.code == "lensLibraryEntryDiffers" && refusal.state == .refused)
            #expect(refusal.reason == LensLibrary.entryDiffersReason(held: 251, workspace: 250))
            #expect(refusal.repairAction == LensLibrary.entryDiffersRepair)
        }
        #expect(other.sent("POST").isEmpty && other.sent("PUT").isEmpty)

        let broken: [(LensLibrary.ProtocolProblem, (Site) -> Void)] = [
            (.partLimit, { $0.partMax = 0 }),
            (.missingParts, { $0.answers["begin"] = ["held": false, "missingParts": [99]] }),
            (.notConfirmed, { $0.answers["complete"] = ["held": false, "bytes": NSNull()] }),
            (.otherLens, { $0.answers["register"] = ["lensID": "another-lens", "registered": true, "changed": true] }),
        ]
        for (kind, change) in broken {
            let site = Site()
            change(site)
            do {
                _ = try await LensLibraryRemote.push(
                    Self.lensID, client: site.client(), root: lens.root, bridge: Bridge(lens).run)
                Issue.record("\(kind) must not succeed")
            } catch let failure as LensLibrary.Refusal {
                // A failure (70), not a refusal: nothing declined, the engine misbehaved.
                #expect(failure == LensLibrary.protocolFailure(kind, server: Self.server), "\(kind)")
                #expect(failure.state == .failed)
            }
        }
    }

    /// The record the site receives is strict JSON, and the result says how
    /// many numbers in it were not finite.
    @Test func aRecordWithNumbersThatAreNotFiniteIsCountedInTheResult() async throws {
        var lens = try Lens()
        defer { lens.remove() }
        lens.nonFiniteNumbers = 2
        lens.recordJSON = lens.recordJSON.replacingOccurrences(of: #""tolerance": 0.1"#, with: #""score": null, "margin": null"#)
        let site = Site()
        let result = try await LensLibraryRemote.push(
            Self.lensID, client: site.client(), root: lens.root, bridge: Bridge(lens).run)
        #expect(result.recordNonFiniteNumbers == 2 && result.registered == true)
        #expect(result.json == .object([
            "lensID": .string(Self.lensID), "convertedSHA256": .string(lens.sha), "bytes": .number(250),
            "held": .bool(true), "alreadyHeld": .bool(false), "adopted": .bool(false),
            "uploadedParts": .number(3), "uploadedBytes": .number(250), "registered": .bool(true),
            "recordChanged": .bool(true), "recordNonFiniteNumbers": .number(2), "changed": .bool(true),
        ]))
        #expect(result.message.hasSuffix("; the site has its record, with 2 numbers in the record that are "
            + "not finite stored as null."))
        #expect(site.registered[Self.lensID] != nil)
    }

    @Test func statusSaysWhatTheSiteHoldsAndChangesNothing() async throws {
        let lens = try Lens()
        defer { lens.remove() }
        let site = Site()
        let bridge = Bridge(lens)
        let client = site.client()
        let before = try await LensLibraryRemote.status(
            Self.lensID, client: client, root: lens.root, bridge: bridge.run)
        #expect(!before.held && before.siteRecord == "absent" && before.libraryAvailable)
        #expect(site.calls.allSatisfy { $0.hasPrefix("GET ") })
        #expect(bridge.calls.map(\.0) == ["lens-record"], "a status read hashes nothing")

        _ = try await LensLibraryRemote.push(Self.lensID, client: client, root: lens.root, bridge: bridge.run)
        let after = try await LensLibraryRemote.status(
            Self.lensID, client: client, root: lens.root, bridge: bridge.run)
        #expect(after.held && after.siteRecord == "registered")
        #expect(after.message == "The site's lens library holds lens custom-lens-example; the site has its record.")
        #expect(after.json == .object([
            "lensID": .string(Self.lensID), "convertedSHA256": .string(lens.sha), "bytes": .number(250),
            "libraryAvailable": .bool(true), "held": .bool(true), "receivedParts": .null,
            "siteRecord": .string("registered"), "changed": .bool(false),
        ]))

        site.registered[Self.lensID]?["converted"] = ["sha256": String(repeating: "f", count: 64)]
        #expect(try await LensLibraryRemote.status(
            Self.lensID, client: client, root: lens.root, bridge: bridge.run).siteRecord == "different")
        site.servesRecords = false          // a runner that serves no lens records
        #expect(try await LensLibraryRemote.status(
            Self.lensID, client: client, root: lens.root, bridge: bridge.run).siteRecord == "unknown")

        let old = Site()
        old.library = nil
        let none = try await LensLibraryRemote.status(
            Self.lensID, client: old.client(), root: lens.root, bridge: bridge.run)
        #expect(!none.libraryAvailable && !none.held)
        #expect(old.sent("GET", containing: "/api/lens-library/").isEmpty)
    }

    @Test func theCommandLineVerbsAreDeclaredAndReport() async throws {
        for verb in LensLibraryRemoteCLI.verbs {
            let spec = try #require(ExperimentCLIParser.spec(namespace: "remote", verb: verb))
            #expect(spec.positional == "<lensID>")
            #expect(spec.valueFlags.isSuperset(of: ["--site", "--url"]))
            #expect(!spec.valueFlags.contains("--token"), "the token comes from the Keychain")
            #expect(ExperimentCLIRunner.needsWorkspace(namespace: "remote", verb: verb))
        }
        #expect(ExperimentCLIParser.spec(namespace: "remote", verb: "lens-push")?.valueFlags.contains("--adopt") == true)
        #expect(ExperimentCLIParser.spec(namespace: "remote", verb: "lens-status")?.valueFlags.contains("--adopt") == false)

        let lens = try Lens()
        defer { lens.remove() }
        let site = Site()
        let bridge = Bridge(lens)
        let client = site.client()
        let pushed = try await LensLibraryRemoteCLI.run(
            ["lens-push", Self.lensID, "--url", Self.server], client: client, root: lens.root,
            sink: .discarding, bridge: bridge.run)
        #expect(pushed.changed)
        #expect(pushed.message.hasPrefix("Sent lens custom-lens-example to the site's lens library"))
        guard case .object(let response)? = pushed.payload["response"] else {
            Issue.record("no response"); return
        }
        #expect(response["uploadedBytes"] == .number(250) && response["registered"] == .bool(true))
        #expect(pushed.payload["endpoint"] == .string(Self.server))
        let status = try await LensLibraryRemoteCLI.run(
            ["lens-status", Self.lensID, "--url", Self.server], client: client, root: lens.root,
            sink: .discarding, bridge: bridge.run)
        #expect(!status.changed)
        #expect(status.message == "The site's lens library holds lens custom-lens-example; the site has its record.")
        await #expect(throws: (any Error).self) {
            _ = try await LensLibraryRemoteCLI.run(
                ["lens-push", "--url", Self.server], client: client, root: lens.root,
                sink: .discarding, bridge: bridge.run)
        }

        // What the command line makes of each kind of stop: a refusal is 65,
        // an interrupted upload or a misbehaving engine is a failure, 70.
        let runner = ExperimentCLIRunner(sink: .discarding)
        let refused = runner.outcome(
            for: .init(code: "lensLibraryUnavailable", reason: "no library", repairAction: "configure one"),
            namespace: "remote", verb: "remote lens-push")
        #expect(refused.envelope.state == .refused && refused.exitCode(json: true) == 65
            && refused.exitCode(json: false) == 1)
        #expect(refused.envelope.error?.code == "lensLibraryUnavailable"
            && refused.envelope.error?.repairAction == "configure one")
        let stopped = runner.outcome(
            for: .init(
                code: LensLibrary.interruptedCode, reason: "interrupted", repairAction: LensLibrary.resumeRepair,
                state: .failed),
            namespace: "remote", verb: "remote lens-push")
        #expect(stopped.envelope.state == .failed && stopped.exitCode(json: true) == 70
            && stopped.exitCode(json: false) == 1)
        #expect(stopped.envelope.error?.code == "lensUploadInterrupted"
            && stopped.envelope.error?.repairAction == LensLibrary.resumeRepair)
    }

    // MARK: - One part of a lens file

    /// Every read a part upload makes of the lens file, recorded.
    private final class RecordingSource: LensLibrary.PartSource {
        let handle: FileHandle
        var position: UInt64 = 0
        var seeks: [UInt64] = []
        var reads: [(position: UInt64, asked: Int, got: Int)] = []
        init(_ url: URL) throws { handle = try FileHandle(forReadingFrom: url) }
        func seek(to offset: UInt64) throws {
            seeks.append(offset)
            position = offset
            try handle.seek(toOffset: offset)
        }
        func read(upTo count: Int) throws -> Data? {
            let data = try handle.read(upToCount: count)
            reads.append((position, count, data?.count ?? 0))
            position += UInt64(data?.count ?? 0)
            return data
        }
    }

    /// Never the whole lens, and never the whole part: every read is at most
    /// one block, and stays inside the part being sent.
    @Test func aPartIsReadABlockAtATimeAndNeverBeyondItself() throws {
        let lens = try Lens()
        defer { lens.remove() }
        #expect(LensLibrary.blockSize == 1 << 20)
        for part in lens.parts {
            let source = try RecordingSource(lens.url)
            let piece = lens.root.appending(component: "part-\(part.index)")
            try LensLibrary.copyPart(part, from: source, to: piece, blockSize: 16)
            #expect(source.seeks == [UInt64(part.offset)], "read from the part's own first byte")
            #expect(source.reads.count == (part.bytes + 15) / 16)
            for read in source.reads {
                #expect(read.asked > 0 && read.asked <= 16, "never more than one block at once")
                #expect(Int(read.position) >= part.offset
                    && Int(read.position) + read.asked <= part.offset + part.bytes, "never beyond the part")
                #expect(read.got == read.asked)
            }
            #expect(source.reads.map(\.got).reduce(0, +) == part.bytes)
            #expect(try Data(contentsOf: piece) == lens.data.subdata(in: part.offset..<(part.offset + part.bytes)))
        }
    }

    /// A lens file that is not the plan's is refused here, with the sentence
    /// the Python client says, and the part is never uploaded.
    @Test func aLensFileThatChangedAfterThePlanStopsTheUpload() async throws {
        let fixture = try Self.fixture()
        for change in ["altered", "shortened"] {
            let lens = try Lens()
            defer { lens.remove() }
            var bytes = lens.data
            if change == "altered" { bytes[249] ^= 1 } else { bytes = bytes.prefix(240) }
            try bytes.write(to: lens.url)      // after the plan (the bridge's answer) was made
            let site = Site()
            do {
                _ = try await LensLibraryRemote.push(
                    Self.lensID, client: site.client(), root: lens.root, bridge: Bridge(lens).run)
                Issue.record("a changed lens file must stop the upload")
            } catch let refusal as LensLibrary.Refusal {
                #expect(refusal.code == (try Self.text(fixture["changed"], "code")) && refusal.state == .refused)
                #expect(refusal.reason == (try Self.text(fixture["changed"], "reason")))
                #expect(refusal.repairAction == (try Self.text(fixture["changed"], "repairAction")))
            }
            // The parts before the changed one arrived; that one was never sent.
            #expect(site.sent("PUT") == [0, 1].map { "PUT /api/lens-library/\(lens.sha)/parts/\($0)" }, "\(change)")
            #expect(site.entries.isEmpty && site.registered.isEmpty)
            #expect(site.sent("POST", containing: "complete").isEmpty)
        }
    }

    // MARK: - Uploading an archive

    /// The archive is streamed from its file. A quarter-gigabyte file (sparse,
    /// so it costs no disk) is uploaded, and the process's memory footprint
    /// while the upload is in flight stays far below the file's size. Reading
    /// the archive whole, as the upload used to, holds every byte of it.
    @Test func anUploadIsStreamedFromItsFileAndNeverReadWhole() async throws {
        let directory = try Self.temporary()
        defer { try? FileManager.default.removeItem(at: directory) }
        let archive = directory.appending(component: "large.tar.gz")
        #expect(FileManager.default.createFile(atPath: archive.path, contents: nil))
        let size = 256 * 1024 * 1024
        let handle = try FileHandle(forWritingTo: archive)
        try handle.truncate(atOffset: UInt64(size))
        try handle.close()

        let site = Site()
        let client = site.client()
        let before = Self.footprint()
        let uploaded = try await client.uploadBundle(archive)
        #expect(uploaded.bytes == size)
        let upload = try #require(site.uploads.first)
        #expect(site.uploads.count == 1 && upload.bytes == size, "every byte was sent")
        let growth = Int64(upload.footprint) - Int64(before)
        #expect(growth < Int64(size / 2), "the upload held \(growth) more bytes in memory")
        #expect(site.sent("POST") == ["POST /api/bundles/upload"])
    }

    @Test func anUploadOverTheLimitIsRefusedWithItsRepair() async throws {
        let directory = try Self.temporary()
        defer { try? FileManager.default.removeItem(at: directory) }
        let archive = try Self.archive(in: directory, name: "carrying", context: ["request": [String: Any]()])
        let site = Site()
        site.uploadStatus = 413
        do {
            _ = try await site.client().uploadBundle(archive)
            Issue.record("a 413 must be refused")
        } catch let refusal as LensLibrary.Refusal {
            #expect(refusal.code == "uploadTooLarge")
            #expect(refusal.reason == "The server refused the upload as larger than its limit: "
                + "upload exceeds 4294967296 bytes (STEERLAB_MAX_UPLOAD_BYTES).")
            #expect(refusal.repairAction == (try Self.text(try Self.fixture()["oversize"], "macRepair")))
            #expect(refusal.repairAction.contains(LensLibrary.oversizeAdvice))
            // The app shows an error's description: the reason and the repair.
            #expect(refusal.localizedDescription == refusal.reason + " " + refusal.repairAction)
        }

        // On the command line: refused (65), with the code and the repair.
        let session = site.session()
        ExperimentRootOverrideLock.acquire()
        ExperimentStore.rootOverride = directory
        defer {
            ExperimentStore.rootOverride = nil
            ExperimentRootOverrideLock.release()
        }
        let outcome = await ExperimentCLIRunner.$remoteSessionOverride.withValue(session) {
            await ExperimentCLIRunner(sink: .discarding).run(
                namespace: "remote", ["upload", archive.path, "--url", Self.server, "--token", "stub"])
        }
        #expect(outcome.envelope.state == .refused)
        #expect(outcome.exitCode(json: true) == 65 && outcome.exitCode(json: false) == 1)
        #expect(outcome.envelope.error?.code == "uploadTooLarge")
        #expect(outcome.envelope.error?.repairAction == LensLibrary.oversizeRepair)
    }

    @Test func aRequestThatRefersToItsLensIsNotUploadedToAnEngineWithoutALibrary() async throws {
        let directory = try Self.temporary()
        defer { try? FileManager.default.removeItem(at: directory) }
        let lenses: [[String: Any]] = [
            ["lensID": Self.lensID, "convertedSHA256": String(repeating: "a", count: 64), "bytes": 7]
        ]
        let referring = try Self.archive(
            in: directory, name: "referring",
            context: ["request": [String: Any](), "lensReference": true, "libraryLenses": lenses])
        let carrying = try Self.archive(in: directory, name: "carrying", context: ["request": [String: Any]()])
        #expect(LensLibrary.archiveRefersToLibrary(referring))
        #expect(!LensLibrary.archiveRefersToLibrary(carrying))
        #expect(!LensLibrary.archiveRefersToLibrary(directory.appending(component: "absent.tar.gz")))
        let notAnArchive = directory.appending(component: "plain.tar.gz")
        try Data("not an archive".utf8).write(to: notAnArchive)
        #expect(!LensLibrary.archiveRefersToLibrary(notAnArchive))

        for library in [nil, false] as [Bool?] {
            let site = Site()
            site.library = library
            let client = site.client()
            do {
                _ = try await client.uploadBundle(referring)
                Issue.record("a referring request must not go to an engine without a library")
            } catch let refusal as LensLibrary.Refusal {
                let fixture = try Self.fixture()
                #expect(refusal.code == "lensLibraryUnavailable")
                #expect(refusal.reason == (try Self.text(fixture["unavailable"], "archiveReason")))
                #expect(refusal.repairAction == (try Self.text(fixture["unavailable"], "archiveRepairAction")))
            }
            #expect(site.sent("POST").isEmpty, "nothing may be sent")
            // A request that carries its lens goes to any engine, as before.
            _ = try await client.uploadBundle(carrying)
            #expect(site.sent("POST") == ["POST /api/bundles/upload"])
        }
        let site = Site()
        _ = try await site.client().uploadBundle(referring)
        #expect(site.sent("POST") == ["POST /api/bundles/upload"])
    }

    /// An up-to-date engine costs an upload nothing extra: one capability
    /// read, and the archive is never opened to look for a lens reference.
    @Test func theCapabilityIsAskedFirstAndTheArchiveIsOpenedOnlyWithoutALibrary() async throws {
        let directory = try Self.temporary()
        defer { try? FileManager.default.removeItem(at: directory) }
        let carrying = try Self.archive(in: directory, name: "carrying", context: ["request": [String: Any]()])
        for (library, expected) in [(true, 0), (false, 1), (nil, 1)] as [(Bool?, Int)] {
            let site = Site()
            site.library = library
            let opened = Counter()
            let client = site.client()
            _ = try await LensLibrary.$archiveInspectorOverride.withValue({ url in
                opened.bump()
                return LensLibrary.archiveRefersToLibrary(url)
            }) {
                try await client.uploadBundle(carrying)
            }
            #expect(opened.count == expected, "library \(String(describing: library))")
            #expect(site.calls == ["GET /api/capabilities", "POST /api/bundles/upload"])
        }
    }

    // MARK: - The command line's requests to the Python client

    @Test func theLensReferenceFlagIsSentOnlyWhenItIsTyped() throws {
        let root = URL(filePath: "/workspace")
        let carried = try DiagnosticWorkspaceCLI.payload(
            DiagnosticArguments(["input-plan", "request.json"], namespace: "science", takesValue: true),
            root: root)
        // Carry mode: the request the Python client received before the flag existed.
        #expect(carried == ["workspaceRoot": .string("/workspace"), "requestFile": .string("request.json")])
        let referred = try DiagnosticWorkspaceCLI.payload(
            DiagnosticArguments(
                ["input-plan", "request.json", "--lens-reference"], namespace: "science", takesValue: true),
            root: root)
        #expect(referred == carried.merging(["lensReference": .bool(true)]) { _, new in new })
        let packaged = try DiagnosticWorkspaceCLI.payload(
            DiagnosticArguments(
                ["package", "request.json", "--archive", "out.tar.gz", "--plan-sha256", "abc",
                 "--lens-reference"], namespace: "science", takesValue: true),
            root: root)
        #expect(packaged == [
            "workspaceRoot": .string("/workspace"), "requestFile": .string("request.json"),
            "archivePath": .string("out.tar.gz"), "planSHA256": .string("abc"),
            "lensReference": .bool(true),
        ])
        // No other verb takes the flag.
        #expect(throws: (any Error).self) {
            try ExperimentCLIParser.parse(namespace: "science", ["import", "a.tar.gz", "--sha256", "abc", "--lens-reference"])
        }
    }

    @Test func aLensCopyNamesTheOtherWorkspaceByItsFullPath() throws {
        let spec = try #require(ExperimentCLIParser.spec(namespace: "science", verb: "lens-copy"))
        #expect(spec.requiredFlags == ["--from"] && spec.valueFlags == ["--from", "--plan-sha256"])
        #expect(DiagnosticWorkspace.localActions == ["lens-push-plan", "lens-record", "lens-copy"])
        #expect(Set(DiagnosticWorkspace.localActions).isDisjoint(with: DiagnosticWorkspace.actions))
        let directory = URL(filePath: "/somewhere/current")
        let review = DiagnosticWorkspaceCLI.lensCopyPayload(
            try DiagnosticArguments(
                ["lens-copy", Self.lensID, "--from", "../first"], namespace: "science", takesValue: true),
            root: URL(filePath: "/workspace"), directory: directory)
        #expect(review == [
            "workspaceRoot": .string("/workspace"), "lensID": .string(Self.lensID),
            "fromWorkspace": .string("/somewhere/first"),
        ])
        let write = DiagnosticWorkspaceCLI.lensCopyPayload(
            try DiagnosticArguments(
                ["lens-copy", Self.lensID, "--from", "/elsewhere/first", "--plan-sha256", "abc"],
                namespace: "science", takesValue: true),
            root: URL(filePath: "/workspace"), directory: directory)
        #expect(write["fromWorkspace"] == .string("/elsewhere/first") && write["planSHA256"] == .string("abc"))
        #expect(throws: (any Error).self) {
            try DiagnosticArguments(["lens-copy", Self.lensID], namespace: "science", takesValue: true)
        }

        // The verb says what is missing, in the Python client's words.
        let fixture = try Self.fixture()
        #expect(LensLibrary.copyNeedsLens == (try Self.text(fixture["copyArguments"], "lensID")))
        #expect(LensLibrary.copyNeedsFrom == (try Self.text(fixture["copyArguments"], "from")))
        let usage = "steerlab-cli science lens-copy <lensID> --from <workspace directory>"
        let missing: [([String], String)] = [
            (["lens-copy", Self.lensID], LensLibrary.copyNeedsFrom),
            (["lens-copy", Self.lensID, "--from"], LensLibrary.copyNeedsFrom),
            (["lens-copy", "--from", "/elsewhere/first"], LensLibrary.copyNeedsLens),
            (["lens-copy", "one", "two", "--from", "/elsewhere/first"], LensLibrary.copyNeedsLens),
        ]
        for (args, sentence) in missing {
            do {
                try DiagnosticWorkspaceCLI.checkLensCopyArguments(args)
                Issue.record("\(args) must be refused")
            } catch let error as ExperimentError {
                #expect(error.reason == sentence && error.malformedInvocation?.repairAction == usage)
            }
        }
        try DiagnosticWorkspaceCLI.checkLensCopyArguments(["lens-copy", Self.lensID, "--from", "../first"])
        try DiagnosticWorkspaceCLI.checkLensCopyArguments(
            ["lens-copy", Self.lensID, "--plan-sha256", "abc", "--from", "/elsewhere/first"])
    }

    /// The local HTTP service serves `actions` only: a caller there cannot
    /// name a second folder on this Mac.
    @Test func theLocalActionsAreNotServedOverHTTP() async throws {
        let root = try Self.temporary()
        defer { try? FileManager.default.removeItem(at: root) }
        for action in DiagnosticWorkspace.localActions {
            let body = try JSONEncoder().encode(JSONValue.object([
                "workspaceRoot": .string(root.path), "lensID": .string(Self.lensID),
                "fromWorkspace": .string("/elsewhere"),
            ]))
            let response = await DiagnosticWorkspace.http(action, body: body, root: root)
            #expect(!response.succeeded)
            #expect(String(decoding: response.body, as: UTF8.self).contains("Unknown diagnostic workspace action"))
        }
    }

    // MARK: - Through the real Python client

    /// A two-layer lens is imported, planned, and copied into a second
    /// workspace through the real bridge; its missing converted file is put
    /// back by an import and by a copy; and a record that holds numbers that
    /// are not finite still reaches this Mac.
    @Test func theRealBridgePlansAPushAndCopiesALens() async throws {
        let python = URL(filePath: try #require(
            ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"],
            "Set STEERLAB_TEST_PYTHON to run the real Python client."))
        let source = Self.repository.appending(component: "Server")
        let directory = try Self.temporary()
        defer { try? FileManager.default.removeItem(at: directory) }
        let first = directory.appending(component: "first")
        let second = directory.appending(component: "second")
        for root in [first, second] {
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        }
        // Two 2x2 float32 layers in one safetensors file.
        let header: [String: Any] = [
            "a": ["dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]],
            "b": ["dtype": "F32", "shape": [2, 2], "data_offsets": [16, 32]],
        ]
        var headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while headerData.count % 8 != 0 { headerData.append(0x20) }
        var length = UInt64(headerData.count).littleEndian
        var tensor = withUnsafeBytes(of: &length) { Data($0) }
        tensor.append(headerData)
        for value in (1...8).map(Float.init) {
            var bits = value.bitPattern.littleEndian
            tensor.append(withUnsafeBytes(of: &bits) { Data($0) })
        }
        try tensor.write(to: first.appending(component: "weights.safetensors"))
        let description = first.appending(component: "import.json")
        let spec: [String: Any] = [
            "schemaVersion": 1, "kind": "jlens", "modelID": "example/model", "hiddenSize": 2,
            "layerCount": 3, "tensorFile": "weights.safetensors",
            "lens": ["targetLayer": 2, "layers": ["0": "a", "1": "b"]],
        ]
        try JSONSerialization.data(withJSONObject: spec).write(to: description)
        func perform(_ action: String, _ payload: [String: JSONValue]) async throws -> [String: JSONValue] {
            guard case .object(let object) = try await DiagnosticWorkspace.perform(
                action, payload: payload, python: python, source: source)
            else { throw ExperimentError(reason: "no answer") }
            return object
        }
        var importing: [String: JSONValue] = [
            "workspaceRoot": .string(first.path), "descriptionFile": .string(description.path),
        ]
        let reviewed = try await perform("artifact-plan", importing)
        importing["planSHA256"] = reviewed["planSHA256"]
        let imported = try await perform("artifact-import", importing)
        guard case .string(let lensID)? = imported["lensID"] else { Issue.record("no lens"); return }
        // One lens, one name: the same reviewed files find the lens they made.
        let repeated = try await perform("artifact-import", importing)
        #expect(repeated["lensID"] == .string(lensID) && repeated["existing"] == .bool(true)
            && repeated["changed"] == .bool(false))
        #expect(LensLibrary.importMessage(for: repeated)
            == "This workspace already holds these files as lens \(lensID), so nothing was imported.")

        let folder = first.appending(path: "runs/jlens-lenses/\(lensID)")
        let convertedURL = folder.appending(component: "jacobians.safetensors")
        let converted = try Data(contentsOf: convertedURL)
        let headerEnd = 8 + Int(converted.prefix(8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })
        let planning: [String: JSONValue] = [
            "workspaceRoot": .string(first.path), "lensID": .string(lensID), "partMaxBytes": .number(1_048_576),
        ]
        let plan = try LensLibrary.Plan(.object(try await perform("lens-push-plan", planning)))
        // The header, then one part per layer; they tile the file, each with its hash.
        #expect(plan.parts.map(\.offset) == [0, headerEnd, headerEnd + 16])
        #expect(plan.parts.map(\.bytes) == [headerEnd, 16, 16])
        #expect(plan.bytes == converted.count && plan.convertedSHA256 == Self.sha256(converted))
        for part in plan.parts {
            #expect(part.sha256 == Self.sha256(converted.subdata(in: part.offset..<(part.offset + part.bytes))))
        }
        #expect(plan.file == "runs/jlens-lenses/\(lensID)/jacobians.safetensors")
        #expect(plan.recordJSON.contains(#""lensID": ""# + lensID + "\""))
        #expect(!plan.recordJSON.contains(first.path), "no path on this Mac goes to a site")
        #expect(plan.nonFiniteNumbers == 0)
        let record = try LensLibrary.Plan(.object(try await perform(
            "lens-record", ["workspaceRoot": .string(first.path), "lensID": .string(lensID)])))
        #expect(record.parts.isEmpty && record.convertedSHA256 == plan.convertedSHA256
            && record.recordJSON == plan.recordJSON)

        // The repair every lens refusal names: import again. A missing
        // converted file is put back from the reviewed sources.
        try FileManager.default.removeItem(at: convertedURL)
        do {
            _ = try await perform("lens-push-plan", planning)
            Issue.record("a lens with no converted file must be refused")
        } catch let refusal as DiagnosticWorkspace.Refusal {
            #expect(refusal.code == "lensFileMissing")
            #expect(refusal.repairAction.hasPrefix("Import the lens again from its source files"))
        }
        let restored = try await perform("artifact-import", importing)
        #expect(restored["existing"] == .bool(true) && restored["changed"] == .bool(true)
            && restored["restored"] == .bool(true))
        #expect(try Data(contentsOf: convertedURL) == converted)
        #expect(LensLibrary.importMessage(for: restored)
            == "This workspace already holds these files as lens \(lensID); its converted file was missing "
            + "and was put back.")

        // The same lens in a second workspace: reviewed, then copied as a clone.
        var copying: [String: JSONValue] = [
            "workspaceRoot": .string(second.path), "lensID": .string(lensID), "fromWorkspace": .string(first.path),
        ]
        let preview = try await perform("lens-copy", copying)
        #expect(preview["changed"] == .bool(false) && preview["action"] == .string("copy"))
        #expect(!FileManager.default.fileExists(atPath: second.appending(component: "runs").path))
        copying["planSHA256"] = preview["planSHA256"]
        let copied = try await perform("lens-copy", copying)
        #expect(copied["changed"] == .bool(true) && copied["copyMethod"] == .string("clone"))
        for name in ["lens.json", "jacobians.safetensors", "import-receipt.json"] {
            let relative = "runs/jlens-lenses/\(lensID)/\(name)"
            #expect(try Data(contentsOf: second.appending(path: relative))
                == Data(contentsOf: first.appending(path: relative)))
        }
        copying["planSHA256"] = nil
        let again = try await perform("lens-copy", copying)
        #expect(again["existing"] == .bool(true) && again["action"] == .string("none"))
        #expect(LensLibrary.copyMessage(for: .object(again))
            == "This workspace already holds lens \(lensID); nothing was copied.")
        // A lens already here whose converted file is missing gets that file back.
        let copy = second.appending(path: "runs/jlens-lenses/\(lensID)/jacobians.safetensors")
        try FileManager.default.removeItem(at: copy)
        let missing = try await perform("lens-copy", copying)
        #expect(missing["action"] == .string("restore") && missing["changed"] == .bool(false))
        copying["planSHA256"] = missing["planSHA256"]
        let putBack = try await perform("lens-copy", copying)
        #expect(putBack["restored"] == .bool(true) && putBack["changed"] == .bool(true))
        #expect(try Data(contentsOf: copy) == converted)
        #expect(LensLibrary.copyMessage(for: .object(putBack))
            == "Lens \(lensID) was already in this workspace; its missing converted file was put back as a "
            + "clone that uses no extra disk space.")
        copying["planSHA256"] = nil
        do {
            _ = try await perform("lens-copy", copying.merging(["lensID": .string("absent")]) { _, new in new })
            Issue.record("a lens the other workspace does not hold must be refused")
        } catch let refusal as DiagnosticWorkspace.Refusal {
            #expect(refusal.code == "lensNotFound")
        }

        // A record that holds numbers that are not finite (a qualification's
        // measured values): the bridge's answer is strict JSON, so it decodes
        // here, where a bare NaN used to read as no answer at all.
        let recordURL = folder.appending(component: "lens.json")
        var stored = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: recordURL)) as? [String: Any])
        stored["qualifications"] = [[
            "qualificationID": "q1", "modelID": "example/model", "revision": "r", "dtype": "float32",
            "checks": ["causal": ["score": 12345.5, "margin": 54321.5]],
        ]]
        let text = String(decoding: try JSONSerialization.data(withJSONObject: stored), as: UTF8.self)
            .replacingOccurrences(of: "12345.5", with: "NaN").replacingOccurrences(of: "54321.5", with: "Infinity")
        #expect(text.contains("NaN") && text.contains("Infinity"))
        try Data(text.utf8).write(to: recordURL)
        for (action, payload) in [("lens-record", ["workspaceRoot": .string(first.path), "lensID": .string(lensID)]),
                                  ("lens-push-plan", planning)] as [(String, [String: JSONValue])] {
            let described = try LensLibrary.Plan(.object(try await perform(action, payload)))
            #expect(described.nonFiniteNumbers == 2, "\(action)")
            #expect(!described.recordJSON.contains("NaN") && !described.recordJSON.contains("Infinity"))
            #expect(described.recordJSON.contains(#""score": null"#) && described.recordJSON.contains(#""margin": null"#))
        }
        // A repeated import answers with the record, qualifications and all.
        let found = try await perform("artifact-import", importing)
        #expect(found["existing"] == .bool(true))
        guard case .object(let answered)? = found["record"], case .array(let qualifications)? = answered["qualifications"],
            case .object(let qualification)? = qualifications.first, case .object(let checks)? = qualification["checks"],
            case .object(let causal)? = checks["causal"]
        else { Issue.record("the record's qualification did not arrive"); return }
        #expect(causal["score"] == .null && causal["margin"] == .null)
    }
}

/// The stub transport for this suite alone (its own handler, so it shares
/// nothing with another suite's stub).
private final class LensStubURLProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var handler: (@Sendable (URLRequest) throws -> (Data, Int))?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let handler = Self.handler else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }
        do {
            let (data, status) = try handler(request)
            let response = HTTPURLResponse(
                url: request.url!, statusCode: status, httpVersion: "HTTP/1.1",
                headerFields: ["Content-Type": "application/json"])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}
