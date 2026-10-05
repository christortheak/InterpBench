import Foundation
import Testing

@testable import ExperimentKit

/// Research Setup says why the helper could not be used, in the failure's own
/// words, when the readiness check finds the helper's files are not this
/// build's.
///
/// Before, the readiness report kept only the command line's reason and
/// repair, so the sheet fell back to "The helper is not installed yet. Choose
/// Review Setup Plan…" — untrue, and a setup plan cannot change which files
/// these are. The failure now travels with the report, the helper step says
/// its plain sentence and its one step, and the technical detail sits behind
/// a "Details" disclosure.
@MainActor
@Suite(.serialized) struct ResearchSetupIdentityTests {

    private static let expected = PythonClientIdentity.sourceSHA256
    private static let other = String(repeating: "b", count: 64)

    private func failure(_ cause: ClientIdentityFailure.Cause) -> ClientIdentityFailure {
        ClientIdentityFailure(
            cause: cause, expected: Self.expected,
            actual: cause == .sourcesDiffer ? Self.other : nil,
            payloadPath: "/Applications/SteerLab.app/Contents/Resources/ServerPayload",
            layout: .appBundle, exitStatus: cause == .sourcesDiffer ? nil : 1)
    }

    // MARK: - The words

    @Test func anIdentityFailureSaysItsOwnSentenceNotNotInstalled() throws {
        for cause in [ClientIdentityFailure.Cause.sourcesDiffer, .identityNotReported] {
            let item = failure(cause)
            let guidance = try #require(
                ResearchSetupCopy.helperGuidance(
                    clientReady: false, basicClientReady: false, identityFailure: item))
            #expect(guidance == item.appSummary + " " + item.appNextStep)
            #expect(guidance != ResearchSetupCopy.helperNeeded)
            #expect(!guidance.contains("not installed"))
            // A setup plan cannot fix a mismatch, so it is not the step.
            #expect(!guidance.contains("Review Setup Plan"))
            #expect(
                ResearchSetupCopy.helperTitle(
                    clientReady: false, basicClientReady: false, identityFailure: item)
                    == ResearchSetupCopy.helperProblemTitle)
        }
    }

    /// The helper did not start: a setup problem, and the only cause whose
    /// step is this sheet's setup plan.
    @Test func aHelperThatDidNotStartKeepsTheSetupStep() throws {
        let item = failure(.noAnswer)
        let guidance = try #require(
            ResearchSetupCopy.helperGuidance(
                clientReady: false, basicClientReady: false, identityFailure: item))
        #expect(guidance.hasPrefix(item.appSummary))
        #expect(guidance.hasSuffix(ResearchSetupCopy.helperRepair))
    }

    @Test func withoutAFailureTheOrdinaryWordsStand() {
        #expect(
            ResearchSetupCopy.helperGuidance(
                clientReady: false, basicClientReady: false, identityFailure: nil)
                == ResearchSetupCopy.helperNeeded)
        #expect(
            ResearchSetupCopy.helperGuidance(
                clientReady: false, basicClientReady: true, identityFailure: nil)
                == ResearchSetupCopy.helperUpdateNeeded)
        #expect(
            ResearchSetupCopy.helperGuidance(
                clientReady: true, basicClientReady: true, identityFailure: failure(.sourcesDiffer))
                == nil)
        #expect(
            ResearchSetupCopy.helperTitle(
                clientReady: false, basicClientReady: false, identityFailure: nil)
                == ResearchSetupCopy.helperSetupTitle)
    }

    // MARK: - The readiness check carries the failure to the sheet

    /// A stand-in helper that answers with another build's identity — the
    /// real readiness path, with no Python installed.
    private func fakeHelper(in directory: URL) throws -> (python: URL, source: URL) {
        let source = directory.appending(component: "Server")
        let client = source.appending(components: "steerlab_server", "client")
        try FileManager.default.createDirectory(at: client, withIntermediateDirectories: true)
        try Data().write(to: source.appending(components: "steerlab_server", "__init__.py"))
        try Data().write(to: client.appending(component: "diagnostic_workspace.py"))
        let python = directory.appending(component: "python")
        let script = """
            #!/bin/sh
            cat > /dev/null
            printf '%s' '{"ok": false, "clientSHA256": "\(Self.other)", "reason": "sources differ"}'
            """
        try Data(script.utf8).write(to: python)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: python.path)
        return (python, source)
    }

    @Test func theReadinessCheckKeepsTheFailureAndTheSheetSaysIt() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "setup-identity-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let helper = try fakeHelper(in: directory)

        let report = await ClientSetup.inspectReport(python: helper.python, source: helper.source)
        let carried = try #require(report.identityFailure)
        #expect(carried.cause == .sourcesDiffer)
        #expect(carried.actual == Self.other)
        // The command line's report is unchanged in shape: same keys, and
        // the failure's own repair.
        #expect(report.readiness["clientReady"] == .bool(false))
        #expect(report.readiness["repairAction"] == .string(carried.repair))
        #expect(
            await ClientSetup.inspect(python: helper.python, source: helper.source).keys.sorted()
                == report.readiness.keys.sorted())

        let model = ResearchSetupModel()
        model.absorb(report)
        let guidance = try #require(model.helperGuidance)
        #expect(guidance != ResearchSetupCopy.helperNeeded)
        #expect(guidance.hasPrefix(carried.appSummary))
        #expect(model.helperTitle == ResearchSetupCopy.helperProblemTitle)
        // The disclosure's content: both identities in full and the files.
        let details = try #require(model.identityFailure?.details)
        #expect(details.contains(Self.expected))
        #expect(details.contains(Self.other))
        #expect(details.contains(helper.source.path))

        // A later check that finds the helper ready clears it.
        model.absorb(.init(readiness: ["clientReady": .bool(true)]))
        #expect(model.identityFailure == nil)
        #expect(model.helperGuidance == nil)
    }

    /// The same failure from a plan or an install shows its detail too.
    @Test func aFailedPlanCarriesItsDetails() {
        let model = ResearchSetupModel()
        let item = failure(.sourcesDiffer)
        model.record(ExperimentError.clientIdentity(item))
        #expect(model.error == item.appSummary)
        #expect(model.errorRepair == item.appNextStep)
        #expect(model.errorDetails == item.details)

        model.record(ExperimentError(reason: "something else"))
        #expect(model.errorDetails == nil)
    }
}
