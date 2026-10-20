import Foundation
import Testing

@testable import ExperimentKit

// The wiring half of `SubmitExecutorDefault` (2026-10-20 cluster smoke test):
// what `steerlab-cli remote submit-bundle` actually puts on the wire, said on
// stderr, and echoes, against a server stubbed below the client. The pure
// rule has its own suite (`SubmitExecutorDefaultTests`).
@Suite(.serialized) struct SubmitBundleExecutorWiringTests {

    struct Outcome {
        var envelope: SteerLabCLIEnvelope
        var stderr: String
        var submitBodies: [[String: JSONValue]]
        var paths: [String]
    }

    /// Runs one `remote submit-bundle` against a stub whose capabilities
    /// declare `serverExecutor` (or fail with a 500 when nil).
    func submit(_ args: [String], serverExecutor: String?) async throws -> Outcome {
        let recorder = Recorder()
        SubmitWiringURLProtocol.handler = { request in
            let path = request.url?.path ?? ""
            recorder.paths.withLock { $0.append(path) }
            switch path {
            case "/api/capabilities":
                guard let serverExecutor else {
                    return (Data(#"{"detail":"unavailable"}"#.utf8), 500)
                }
                let body: [String: Any] = [
                    "apiSchemaVersion": "2026-07-01",
                    "engine": "python-hf-transformers",
                    "instrumentation": [
                        "policy-evidence-v2", "policy-v1", "probe-readings-v1",
                    ],
                    "profile": ["profile": "cluster", "executor": serverExecutor],
                ]
                return (try JSONSerialization.data(withJSONObject: body), 200)
            case "/api/bundles/inspect":
                return (Data(#"{"kind":"runBundle","experiment":"demo"}"#.utf8), 200)
            case "/api/studies/submit-bundle":
                let data = Self.bodyData(from: request) ?? Data()
                let sent = try JSONDecoder().decode([String: JSONValue].self, from: data)
                recorder.bodies.withLock { $0.append(sent) }
                var executor = "?"
                if case .string(let value)? = sent["executor"] { executor = value }
                let reply: [String: Any] = [
                    "jobId": "stub0001", "experiment": "demo", "verb": "run",
                    "executor": executor, "dryRun": false, "runBundle": [:],
                    "command": ["python"], "recordsDirectory": "/stub/records",
                    "submissionDirectory": "/stub",
                ]
                return (try JSONSerialization.data(withJSONObject: reply), 200)
            default:
                return (Data(#"{"detail":"no such route"}"#.utf8), 404)
            }
        }
        defer { SubmitWiringURLProtocol.handler = nil }
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [SubmitWiringURLProtocol.self]
        let session = URLSession(configuration: configuration)

        let stderr = WiringLocked("")
        let sink = ExperimentCLISink(
            out: { _ in }, outRaw: { _ in },
            err: { text in stderr.withLock { $0 += text } })
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "submit-wiring-\(UUID().uuidString)")
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        let result = await ExperimentCLIRunner.$remoteSessionOverride.withValue(session) {
            await ExperimentCLIRunner(sink: sink).run(
                namespace: "remote",
                ["submit-bundle", "/srv/b.tar.gz", "--url", "http://127.0.0.1:1",
                 "--token", "stub"] + args)
        }
        return Outcome(
            envelope: result.envelope, stderr: stderr.withLock { $0 },
            submitBodies: recorder.bodies.withLock { $0 },
            paths: recorder.paths.withLock { $0 })
    }

    private func executor(_ outcome: Outcome) -> String? {
        guard case .string(let value)? = outcome.submitBodies.first?["executor"] else {
            return nil
        }
        return value
    }

    private func source(_ outcome: Outcome) -> String? {
        guard case .string(let value)? = outcome.envelope.result?["executorSource"] else {
            return nil
        }
        return value
    }

    @Test func aModelVerbWithoutAFlagIsSentToTheServersOwnExecutor() async throws {
        let outcome = try await submit(["--verb", "run"], serverExecutor: "slurm")
        #expect(outcome.envelope.exitCode == 0)
        #expect(executor(outcome) == "slurm")
        #expect(source(outcome) == "serverProfile")
        #expect(!outcome.stderr.contains("controller's own"))
    }

    /// Before, `--parallel` without `--executor slurm` was silently reduced
    /// to one job, because the client believed the executor was local.
    @Test func parallelIsEncodedUnderTheServersSlurmDefault() async throws {
        let outcome = try await submit(
            ["--verb", "run", "--parallel", "3"], serverExecutor: "slurm")
        #expect(outcome.submitBodies.first?["parallelJobs"] == .number(3))
        #expect(outcome.envelope.result?["parallelJobsEncoded"] == .number(3))
    }

    @Test func verbsThatLoadNoModelStayOnTheController() async throws {
        for verb in ["analyze", "verify"] {
            let outcome = try await submit(["--verb", verb], serverExecutor: "slurm")
            #expect(executor(outcome) == "local", "\(verb)")
            #expect(source(outcome) == "loadsNoModel", "\(verb)")
            #expect(!outcome.stderr.contains("controller's own"), "\(verb)")
            // Named local, a verb that loads no model has no warning to look
            // up, so its submission waits on no extra request.
            let named = try await submit(
                ["--verb", verb, "--executor", "local"], serverExecutor: "slurm")
            #expect(executor(named) == "local", "\(verb)")
            #expect(
                named.paths.filter { $0 == "/api/capabilities" }.count
                    == outcome.paths.filter { $0 == "/api/capabilities" }.count,
                "\(verb)")
        }
    }

    @Test func namingLocalForAModelVerbIsSentAndWarns() async throws {
        let outcome = try await submit(
            ["--verb", "extract", "--executor", "local"], serverExecutor: "slurm")
        #expect(outcome.envelope.exitCode == 0)
        #expect(executor(outcome) == "local")
        #expect(source(outcome) == "flag")
        #expect(outcome.stderr.contains("warning: --executor local runs this extract"))
    }

    @Test func anUnreadableServerExecutorRefusesInsteadOfGuessingLocal() async throws {
        let outcome = try await submit(["--verb", "run"], serverExecutor: nil)
        #expect(outcome.envelope.exitCode == 65)
        #expect(outcome.envelope.error?.code == "missingPrerequisite")
        #expect(
            outcome.envelope.error?.repairAction
                == "steerlab-cli remote submit-bundle /srv/b.tar.gz --verb run "
                + "--executor slurm (its own Slurm job), or --executor local "
                + "(the controller), to choose it yourself")
        #expect(outcome.envelope.error?.reason.contains("Nothing was submitted") == true)
        #expect(outcome.submitBodies.isEmpty)
        #expect(!outcome.paths.contains("/api/studies/submit-bundle"))
    }

    // MARK: - Stub plumbing

    final class Recorder: Sendable {
        let bodies = WiringLocked<[[String: JSONValue]]>([])
        let paths = WiringLocked<[String]>([])
    }

    static func bodyData(from request: URLRequest) -> Data? {
        if let body = request.httpBody { return body }
        guard let stream = request.httpBodyStream else { return nil }
        stream.open()
        defer { stream.close() }
        var data = Data()
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: 4096)
        defer { buffer.deallocate() }
        while stream.hasBytesAvailable {
            let count = stream.read(buffer, maxLength: 4096)
            guard count > 0 else { break }
            data.append(buffer, count: count)
        }
        return data
    }
}

/// A lock-guarded value, for the stub's records and the captured stderr.
final class WiringLocked<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var value: Value
    init(_ value: Value) { self.value = value }
    func withLock<T>(_ body: (inout Value) throws -> T) rethrows -> T {
        lock.lock()
        defer { lock.unlock() }
        return try body(&value)
    }
}

private final class SubmitWiringURLProtocol: URLProtocol, @unchecked Sendable {
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
