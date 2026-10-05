import Foundation
import Testing

@testable import ExperimentKit

/// A refused local workspace action answers `refused` (65), on both clients.
///
/// The Python client classifies every declined bridge action in one place
/// (`diagnostic_commands.refusal_fields`) and says which it is in its answer:
/// `refused` for a well-formed request it declined, `blocked` for a request
/// in the wrong shape. The Mac used to read every declined answer as a
/// malformed invocation, so `science report` on a folder with no report
/// exited 64. Python twin: `Server/tests/test_bridge_refusal_state.py`.
struct DiagnosticBridgeRefusalTests {

    private var repository: URL {
        URL(filePath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    @Test func aRefusedAnswerBecomesARefusal() throws {
        let error = DiagnosticWorkspace.failure(from: [
            "ok": .bool(false), "state": .string("refused"), "code": .string("reportRefused"),
            "reason": .string("This folder holds no report."),
            "repairAction": .string("Give a run folder that holds assessment-report.json."),
        ])
        let refusal = try #require(error as? DiagnosticWorkspace.Refusal)
        #expect(refusal.code == "reportRefused")
        #expect(refusal.reason == "This folder holds no report.")
        #expect(refusal.repairAction.contains("assessment-report.json"))
        // The app shows the reason, as it did before the refusal had a type.
        #expect(refusal.localizedDescription == "This folder holds no report.")
    }

    @Test func aMalformedAnswerStaysAMalformedInvocation() throws {
        for state: JSONValue? in [.string("blocked"), nil] {
            var object: [String: JSONValue] = [
                "ok": .bool(false), "reason": .string("Supply the declared fields."),
                "repairAction": .string("science report --help"),
            ]
            if let state { object["state"] = state }
            let error = try #require(DiagnosticWorkspace.failure(from: object) as? ExperimentError)
            #expect(error.malformedInvocation?.repairAction == "science report --help")
        }
    }

    /// The command line's answer: `refused`, 65 in JSON mode, with the Python
    /// client's code and repair; human mode keeps exit 1.
    @Test func theCommandLineAnswersARefusalWith65() {
        let outcome = ExperimentCLIRunner(sink: .discarding).outcome(
            for: .init(
                code: "reportRefused", reason: "This folder holds no report.",
                repairAction: "Give a run folder that holds assessment-report.json."),
            namespace: "science", verb: "science report")
        #expect(outcome.envelope.state == .refused)
        #expect(outcome.exitCode(json: true) == 65)
        #expect(outcome.exitCode(json: false) == 1)
        #expect(outcome.envelope.error?.code == "reportRefused")
        #expect(outcome.envelope.error?.repairAction.contains("assessment-report.json") == true)
    }

    /// Through the real Python client: a refused report and a malformed
    /// request arrive as the two different errors.
    @Test func theRealBridgeSaysWhichIsWhich() async throws {
        let python = URL(filePath: try #require(
            ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"],
            "Set STEERLAB_TEST_PYTHON to run the real Python client."))
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
            .resolvingSymlinksInPath()
        try FileManager.default.createDirectory(
            at: root.appending(component: "runs"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = repository.appending(path: "Server")
        do {
            _ = try await DiagnosticWorkspace.perform(
                ScienceReport.action,
                payload: ["workspaceRoot": .string(root.path), "path": .string("runs/missing")],
                python: python, source: source)
            Issue.record("A missing report must be refused.")
        } catch let refusal as DiagnosticWorkspace.Refusal {
            #expect(refusal.code == "reportRefused")
            #expect(refusal.reason.contains("Nothing was found at runs/missing"))
        }
        do {
            _ = try await DiagnosticWorkspace.perform(
                ScienceReport.action, payload: ["workspaceRoot": .string(root.path)],
                python: python, source: source)
            Issue.record("A request with no path must be refused as malformed.")
        } catch let error as ExperimentError {
            #expect(error.malformedInvocation != nil)
        }
    }
}
