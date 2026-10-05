import Foundation
import Testing

@testable import ExperimentKit

/// "The Mac and Python sources differ" — each distinct cause, its wording,
/// and its repair.
///
/// The comparison is local: this build's compiled identity against the
/// Python client files it resolves on this machine. No server takes part, so
/// no wording sends the reader to one, and the repair for a mismatch never
/// leads with setting the helper up again (which cannot change which files
/// these are). The remote stage now confirms the match before it asks the
/// server for anything.
@Suite(.serialized) struct ClientIdentityFailureTests {

    private static let expected = String(repeating: "a", count: 64)
    private static let found = String(repeating: "b", count: 64)
    private static let app = URL(filePath: "/Applications/SteerLab.app/Contents/Resources/ServerPayload")
    private static let python = URL(filePath: "/usr/bin/python3")

    private func failure(_ answer: JSONValue?, source: URL = ClientIdentityFailureTests.app,
                         exit: Int32 = 65) -> ClientIdentityFailure? {
        DiagnosticWorkspace.identityFailure(
            answer: answer, expected: Self.expected, source: source,
            interpreter: Self.python, exitStatus: exit, errorOutput: { "Traceback | ImportError: no module" })
    }

    // MARK: - One cause per shape of answer

    @Test func aDifferentIdentityNamesBothSidesAndThePath() throws {
        let reported = "/Applications/SteerLab.app/Contents/Resources/ServerPayload"
        let got = try #require(failure(.object([
            "ok": .bool(false), "clientSHA256": .string(Self.found),
            "clientRoot": .string(reported + "/steerlab_server"),
            "reason": .string("The Mac build and local Python client sources differ"),
        ])))
        #expect(got.cause == .sourcesDiffer)
        #expect(got.layout == .appBundle)
        #expect(got.reason.contains("expects sources aaaaaaaaaaaa"))
        #expect(got.reason.contains(reported + "/steerlab_server are bbbbbbbbbbbb"))
        #expect(!got.reason.contains(Self.expected))  // short hashes in prose
        #expect(got.details.contains(Self.expected) && got.details.contains(Self.found))
        // The repair leads with a fresh start and never with the setup verbs.
        #expect(got.repair.hasPrefix("Start again first"))
        #expect(!got.repair.contains("setup plan") && !got.repair.contains("setup apply"))
        #expect(got.repair.contains(ClientIdentityFailure.noServerTakesPart))
    }

    @Test func aMatchingIdentityIsNoFailure() {
        #expect(failure(.object(["ok": .bool(true), "clientSHA256": .string(Self.expected)])) == nil)
    }

    /// Python answered but stopped before it could hash its files (an
    /// unreadable or symlinked file, say): not a mismatch, and not called one.
    @Test func anAnswerWithoutAnIdentityIsNotCalledAMismatch() throws {
        let got = try #require(failure(.object([
            "ok": .bool(false), "clientSHA256": .null,
            "reason": .string("Python client sources must be ordinary installed files."),
        ])))
        #expect(got.cause == .identityNotReported)
        #expect(got.reason.contains("This is not a version mismatch"))
        #expect(got.reason.contains("ordinary installed files."))
        #expect(!got.reason.contains("expects sources"))
        #expect(got.repair.contains("complete, ordinary, readable files"))
        #expect(!got.repair.hasPrefix(ScientificPythonRuntime.setupHint))
        // An older client that omits the key entirely is the same cause.
        #expect(failure(.object(["ok": .bool(true)]))?.cause == .identityNotReported)
    }

    /// No answer at all: the helper did not start. Setting it up again is
    /// the right repair here, and only here.
    @Test func noAnswerIsAStartupProblemWithTheSetupRepair() throws {
        let got = try #require(failure(nil, exit: 1))
        #expect(got.cause == .noAnswer)
        #expect(got.reason.contains("did not answer (exit status 1)"))
        #expect(got.reason.contains("not a version mismatch"))
        #expect(got.reason.contains("ImportError: no module"))
        #expect(got.repair == ScientificPythonRuntime.setupHint)
    }

    // MARK: - The repair fits where the files are

    @Test func theRepairFollowsTheLayoutAndAReplacedInstall() {
        let checkout = URL(filePath: "/work/checkout/Server")
        let classified = ClientIdentityFailure.classify(
            payload: checkout, fileExists: { $0 == "/work/checkout/Package.swift" })
        #expect(classified.layout == .developerCheckout)
        #expect(ClientIdentityFailure.classify(payload: Self.app).installation.path
            == "/Applications/SteerLab.app")
        #expect(ClientIdentityFailure.classify(
            payload: URL(filePath: "/opt/payload"), fileExists: { _ in false }).layout == .other)

        var developer = ClientIdentityFailure(
            cause: .sourcesDiffer, expected: Self.expected, actual: Self.found,
            payloadPath: checkout.path, layout: .developerCheckout)
        #expect(developer.repair.contains("check-python-client-identity.py --write"))
        developer.layout = .appBundle
        developer.replacedWhileRunning = true
        #expect(developer.repair.hasPrefix("Run the command again, or quit SteerLab"))
        #expect(developer.reason.contains("replaced after this process started"))
        #expect(developer.appSummary == "SteerLab was updated on this Mac while this copy was still open.")
    }

    @Test func aReplacedInstallIsSeenFromTheInstallationsOwnFolders() {
        let start = Date(timeIntervalSince1970: 1_000)
        let installation = URL(filePath: "/Applications/SteerLab.app")
        func replaced(_ changed: Date?, layout: ClientIdentityFailure.Layout = .appBundle) -> Bool {
            ClientIdentityFailure.replacedSinceLaunch(
                payload: Self.app, layout: layout, installation: installation,
                processStart: start, changeDate: { $0 == installation ? changed : nil })
        }
        #expect(replaced(Date(timeIntervalSince1970: 2_000)))
        #expect(!replaced(Date(timeIntervalSince1970: 500)))
        #expect(!replaced(nil))
        // A checkout is never claimed: its files change one at a time.
        #expect(!replaced(Date(timeIntervalSince1970: 2_000), layout: .developerCheckout))
    }

    // MARK: - The app's words, the command line's words

    @Test func theAppSaysItPlainlyAndTheCommandLineKeepsTheDetail() throws {
        let developerText = ["steerlab-cli", "STEERLAB_", "--", "payload", "Python",
                             "/", "aaaaaa", "bbbbbb", "checkout", "rebuild"]
        for cause in [ClientIdentityFailure.Cause.sourcesDiffer, .identityNotReported, .noAnswer] {
            for replaced in [false, true] {
                for staged in [false, true] {
                    let item = ClientIdentityFailure(
                        cause: cause, expected: Self.expected, actual: Self.found,
                        payloadPath: Self.app.path, layout: .appBundle,
                        replacedWhileRunning: replaced, afterRemoteStage: staged)
                    for text in [item.appSummary, item.appNextStep] {
                        let offences = developerText.filter { text.contains($0) }
                        #expect(offences.isEmpty, "\(offences) in: \(text)")
                    }
                    let error = ExperimentError.clientIdentity(item)
                    // The app reads localizedDescription; the command line
                    // reads the reason and the repair, unchanged in class.
                    #expect(error.localizedDescription == item.appSummary + " " + item.appNextStep)
                    #expect(error.reason == item.reason)
                    #expect(error.malformedInvocation?.repairAction == item.repair)
                    // Research Setup keeps its own setup sentence for the one
                    // cause that is a setup problem.
                    let shown = ResearchSetupCopy.failure(error)
                    #expect(shown.reason == item.appSummary)
                    #expect(shown.repair == (cause == .noAnswer
                        ? ResearchSetupCopy.helperRepair : item.appNextStep))
                }
            }
        }
    }

    @Test func afterTheServerStagedTheWordingSaysWhatThatLeaves() {
        let item = ClientIdentityFailure(
            cause: .sourcesDiffer, expected: Self.expected, actual: Self.found,
            payloadPath: Self.app.path, layout: .appBundle, replacedWhileRunning: true,
            afterRemoteStage: true)
        #expect(item.reason.hasPrefix("The controller staged the archive, but this Mac could not then record it"))
        #expect(item.repair.contains("run the same science-stage command again"))
        #expect(item.repair.contains("nothing is uploaded again"))
        #expect(item.appSummary.hasPrefix("The server prepared your archive"))
    }

    // MARK: - The remote stage checks first

    /// A mismatch is found before the server is asked to do anything: the
    /// stage request is never sent.
    @Test func theStageConfirmsTheMatchBeforeAskingTheServer() async throws {
        IdentityStageProtocol.requests = 0
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [IdentityStageProtocol.self]
        let client = ClusterClient(
            profile: .init(baseURL: URL(string: "http://server.test")!),
            session: URLSession(configuration: configuration))
        let digest = String(repeating: "c", count: 64)
        let mismatch = ExperimentError.clientIdentity(.init(
            cause: .sourcesDiffer, expected: Self.expected, actual: Self.found,
            payloadPath: Self.app.path, layout: .appBundle))

        await #expect(throws: ExperimentError.self) {
            _ = try await DiagnosticRemote.stage(
                path: "/runs/input.tar.gz", sha256: digest, client: client,
                root: URL(filePath: "/tmp"), confirm: { throw mismatch },
                record: { _ in Issue.record("recorded without confirming"); return .null })
        }
        #expect(IdentityStageProtocol.requests == 0)

        // A match proceeds; if the files are replaced while the server works,
        // the failure says the server already staged.
        var thrown: ExperimentError?
        do {
            _ = try await DiagnosticRemote.stage(
                path: "/runs/input.tar.gz", sha256: digest, client: client,
                root: URL(filePath: "/tmp"), confirm: {}, record: { _ in throw mismatch })
        } catch let error as ExperimentError { thrown = error }
        #expect(IdentityStageProtocol.requests == 1)
        #expect(thrown?.clientIdentityFailure?.afterRemoteStage == true)
        #expect(thrown?.reason.hasPrefix("The controller staged the archive") == true)
    }
}

private final class IdentityStageProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var requests = 0
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        Self.requests += 1
        let body = Data(#"{"inputBundleSHA256":"\#(String(repeating: "c", count: 64))","request":{"inputBundleSHA256":"\#(String(repeating: "c", count: 64))"}}"#.utf8)
        let response = HTTPURLResponse(
            url: request.url!, statusCode: 200, httpVersion: nil,
            headerFields: ["Content-Type": "application/json"])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: body)
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}
