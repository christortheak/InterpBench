import Foundation
import Testing
@testable import ExperimentKit

/// Research Setup's Cancel, through the model (release review A4). The model
/// runs a stand-in installer from a temporary folder, with a temporary runtime
/// and log folder, so nothing is installed and the real home folder is never
/// touched. The real installer's own signal handling is held by
/// Server/tests/test_client_installer.py.
@MainActor @Suite(.serialized)
struct ResearchSetupCancelTests {
    /// A stand-in for install-client.sh: `plan` answers at once; `install`
    /// stalls until it is sent TERM, then reports a cancellation the way the
    /// real installer does. It stops its own child, so nothing outlives it.
    private static let stalledInstaller = #"""
        #!/bin/sh
        here=$(dirname "$0")
        case "$1" in
            plan)
                printf '{"ok":true,"changed":false,"planSHA256":"fixture-plan","runtime":"%s","actions":["Download (about 1 MB in all)"]}\n' "$3" ;;
            install|repair)
                sleep 60 >/dev/null 2>&1 </dev/null &
                child=$!
                trap 'kill "$child" 2>/dev/null; : > "$here/terminated"; printf "%s\n" "{\"ok\":false,\"changed\":false,\"code\":\"cancelled\",\"reason\":\"Setup was cancelled.\",\"repairAction\":\"Nothing was changed.\"}"; exit 130' TERM
                : > "$here/started"
                wait "$child"
                printf '{"ok":true,"changed":true}\n' ;;
        esac
        """#

    private func makeRelease(_ root: URL) throws -> URL {
        let release = root.appending(component: "release")
        try FileManager.default.createDirectory(at: release, withIntermediateDirectories: true)
        try Self.stalledInstaller.write(
            to: release.appending(component: "install-client.sh"), atomically: true, encoding: .utf8)
        // The installer must belong to this build, as the app's own does.
        try PythonClientIdentity.sourceSHA256.write(
            to: release.appending(component: "source.sha256"), atomically: true, encoding: .utf8)
        return release
    }

    private func waitFor(_ url: URL, seconds: Double = 20) async throws {
        let deadline = Date().addingTimeInterval(seconds)
        while !FileManager.default.fileExists(atPath: url.path) {
            guard Date() < deadline else {
                Issue.record("\(url.lastPathComponent) never appeared")
                throw CancellationError()
            }
            try await Task.sleep(for: .milliseconds(50))
        }
    }

    @Test func cancelStopsARunningInstallAndReportsNoError() async throws {
        let root = FileManager.default.temporaryDirectory.appending(component: "research-setup-cancel-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let release = try makeRelease(root)
        let model = ResearchSetupModel(locations: .init(
            release: release, runtime: root.appending(component: "client-runtime"),
            logDirectory: root.appending(component: "logs")))

        await model.preview()
        #expect(model.planHash == "fixture-plan")
        #expect(model.error == nil)

        let began = Date()
        let install = Task { await model.install(workspace: nil) }
        try await waitFor(release.appending(component: "started"))
        #expect(model.busy && model.installing && !model.cancelling)

        model.cancelInstall()
        #expect(model.cancelling)
        #expect(model.message == ResearchSetupCopy.cancellingInstall)
        await install.value

        // Stopped promptly, not after the stand-in's 60-second download.
        #expect(Date().timeIntervalSince(began) < 30)
        #expect(FileManager.default.fileExists(atPath: release.appending(component: "terminated").path))
        // A cancellation the researcher asked for is not shown as an error.
        #expect(model.error == nil && model.errorRepair == nil)
        #expect(model.message == ResearchSetupCopy.installCancelled)
        #expect(!model.busy && !model.installing && !model.cancelling)
        #expect(model.planHash == nil, "a cancelled plan must be reviewed again")
        #expect(!FileManager.default.fileExists(atPath: root.appending(component: "client-runtime").path))
    }

    @Test func cancelOutsideAnInstallDoesNothing() async throws {
        let root = FileManager.default.temporaryDirectory.appending(component: "research-setup-cancel-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = ResearchSetupModel(locations: .init(
            release: try makeRelease(root), runtime: root.appending(component: "client-runtime"),
            logDirectory: root.appending(component: "logs")))
        await model.preview()
        model.cancelInstall()
        #expect(!model.cancelling && model.message == nil)
        #expect(model.planHash == "fixture-plan", "the plan survives a stray Cancel")
    }

    /// A cancellation that reaches the installer before it starts never
    /// starts it.
    @Test func aCancelBeforeLaunchNeverStartsTheInstaller() throws {
        let control = InstallerProcessControl()
        control.cancel()
        let process = Process()
        process.executableURL = URL(filePath: "/bin/sh")
        process.arguments = ["-c", "exit 0"]
        #expect(throws: CancellationError.self) { try control.launch(process) }
        #expect(!process.isRunning)
    }
}
