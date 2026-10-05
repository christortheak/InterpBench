import Foundation
import Testing
@testable import ExperimentKit

/// Remote job origins recorded in the workspace: what lets the app import
/// from, or act on, a job a command line submitted, with no reconnect.
@MainActor @Suite(.serialized)
struct WorkspaceJobOriginTests {
    private var repository: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
    }

    private func withScratch<T>(_ body: (URL, UserDefaults) throws -> T) throws -> T {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "job-origins-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let suite = "job-origins-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer {
            defaults.removePersistentDomain(forName: suite)
            try? FileManager.default.removeItem(at: root)
        }
        return try body(root, defaults)
    }

    private func client(_ url: String, identity: String? = nil) -> ClusterClient {
        ClusterClient(profile: .init(name: "Connected", baseURL: URL(string: url)!, serverIdentity: identity))
    }

    /// The shape `steerlab-cli remote submit-bundle --site <id>` writes.
    private func macCommandLineOrigin(root: URL, identity: String = "ssh://cluster.invalid:8080") -> WorkspaceJobOrigin {
        WorkspaceJobOrigin(
            serverIdentity: identity, endpoint: "http://127.0.0.1:8743", siteID: "lab",
            servingRoot: "/scratch/steerlab", workspaceRoot: root.path,
            submittedBy: WorkspaceJobOrigin.macCommandLine, experiment: "study-a",
            verb: "run", operation: "submit-bundle")
    }

    @Test func aMacCommandLineOriginLetsTheAppActWithNoReconnect() throws {
        try withScratch { root, defaults in
            try WorkspaceJobOrigins.record(macCommandLineOrigin(root: root), jobID: "job-1", workspaceRoot: root)
            let jobs = StudyRemoteJobController(defaults: defaults, workspaceRoot: { root })
            let origin = try jobs.origin(for: "job-1")
            #expect(origin.workspaceRoot == root.standardizedFileURL)
            #expect(origin.connection.serverIdentity == "ssh://cluster.invalid:8080")
            // The app's tunnel to the same site came up on ANOTHER local port:
            // the durable identity decides, not the port.
            let sameSite = client("http://127.0.0.1:8799", identity: "ssh://cluster.invalid:8080")
            #expect(try jobs.clientForJob("job-1", connected: sameSite).profile == sameSite.profile)
            let otherSite = client("http://127.0.0.1:8743", identity: "ssh://other.invalid:8080")
            #expect(throws: (any Error).self) { try jobs.clientForJob("job-1", connected: otherSite) }
            // The app's own preferences were not needed and were not written.
            #expect(defaults.data(forKey: "SteerLabRemoteJobOrigins") == nil)
        }
    }

    @Test func thePythonClientsRecordIsReadByTheAppsController() throws {
        try withScratch { root, defaults in
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            let localPython = repository.appending(path: "Server/.venv.nosync/bin/python").path
            let configured = ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"]
            let python = configured ?? (FileManager.default.isExecutableFile(atPath: localPython) ? localPython : "python3")
            // The real Python writer, exactly as `steerlab runner submit
            // --root <workspace>` calls it.
            process.arguments = [python, "-c", """
            import sys
            from steerlab_server.client import job_origins
            job_origins.record(sys.argv[1], job_id='py-job', endpoint='http://127.0.0.1:8080',
                               serving_root='/srv/runner', experiment='study-b', verb='run',
                               operation='submit-bundle')
            """, root.path]
            var environment = ProcessInfo.processInfo.environment
            environment["PYTHONPATH"] = repository.appending(component: "Server").path
            process.environment = environment
            let errors = Pipe()
            process.standardError = errors
            try process.run()
            let diagnostics = errors.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            #expect(process.terminationStatus == 0, "\(String(decoding: diagnostics, as: UTF8.self))")

            let recorded = try #require(WorkspaceJobOrigins.origins(forJob: "py-job", workspaceRoot: root).first)
            #expect(recorded.submittedBy == WorkspaceJobOrigin.pythonClient)
            #expect(recorded.isCommandLine)
            #expect(recorded.experiment == "study-b")
            #expect(recorded.servingRoot == "/srv/runner")
            let jobs = StudyRemoteJobController(defaults: defaults, workspaceRoot: { root })
            let origin = try jobs.origin(for: "py-job")
            #expect(origin.workspaceRoot == root.standardizedFileURL)
            // The app connected to that runner as a saved direct server.
            let connected = client(
                "http://127.0.0.1:8080",
                identity: ClusterConnectionStore.normalizedEndpointKey("http://127.0.0.1:8080"))
            #expect(throws: Never.self) { try jobs.clientForJob("py-job", connected: connected) }
        }
    }

    @Test func theWorkspaceRecordIsReadBeforeTheAppsPreferences() throws {
        try withScratch { root, defaults in
            let preferred = StudyRemoteJobController(defaults: defaults, workspaceRoot: { root })
            preferred.recordOrigin(
                .init(connection: .init(baseURL: URL(string: "http://app.invalid")!), workspaceRoot: root),
                jobID: "job")
            #expect(try preferred.origin(for: "job").connection.baseURL.host() == "app.invalid")
            try WorkspaceJobOrigins.record(macCommandLineOrigin(root: root), jobID: "job", workspaceRoot: root)
            let restarted = StudyRemoteJobController(defaults: defaults, workspaceRoot: { root })
            #expect(try restarted.origin(for: "job").connection.serverIdentity == "ssh://cluster.invalid:8080")
            // Preferences are still kept, for an older build of the app.
            #expect(defaults.data(forKey: "SteerLabRemoteJobOrigins") != nil)
        }
    }

    @Test func aJobIDTwoServersShareStaysAmbiguousAndRefuses() throws {
        try withScratch { root, defaults in
            try WorkspaceJobOrigins.record(macCommandLineOrigin(root: root), jobID: "same", workspaceRoot: root)
            try WorkspaceJobOrigins.record(
                macCommandLineOrigin(root: root, identity: "http://second.invalid:80"), jobID: "same",
                workspaceRoot: root)
            let jobs = StudyRemoteJobController(defaults: defaults, workspaceRoot: { root })
            #expect(throws: (any Error).self) { try jobs.origin(for: "same") }
            // The same job on the same server is replaced, never duplicated.
            try WorkspaceJobOrigins.record(macCommandLineOrigin(root: root), jobID: "same", workspaceRoot: root)
            #expect(WorkspaceJobOrigins.origins(forJob: "same", workspaceRoot: root).count == 2)
        }
    }

    @Test func aWriteKeepsRowsANewerClientWroteAndSetsAsideAnUnreadableFile() throws {
        try withScratch { root, _ in
            let folder = WorkspaceJobOrigins.directory(workspaceRoot: root)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let future = #"{"schemaVersion": 2, "extra": 1, "jobs": {"old": [{"serverIdentity": "http://a.invalid:80", "endpoint": "http://a.invalid", "workspaceRoot": "/w", "submittedBy": "steerlab", "recordedAt": "x", "futureField": true}, "not-a-row"]}}"#
            try Data(future.utf8).write(to: WorkspaceJobOrigins.fileURL(workspaceRoot: root))
            try WorkspaceJobOrigins.record(macCommandLineOrigin(root: root), jobID: "new", workspaceRoot: root)
            let written = try #require(
                try JSONSerialization.jsonObject(with: Data(contentsOf: WorkspaceJobOrigins.fileURL(workspaceRoot: root)))
                    as? [String: Any])
            #expect(written["schemaVersion"] as? Int == 2)
            #expect(written["extra"] as? Int == 1)
            let old = try #require((written["jobs"] as? [String: Any])?["old"] as? [Any])
            #expect(old.count == 2)
            #expect((old.first as? [String: Any])?["futureField"] as? Bool == true)
            #expect(WorkspaceJobOrigins.load(workspaceRoot: root)["old"]?.count == 1)
            let ignore = try String(contentsOf: folder.appending(component: ".gitignore"), encoding: .utf8)
            #expect(ignore == "*\n")

            try Data("{\"jobs\": [".utf8).write(to: WorkspaceJobOrigins.fileURL(workspaceRoot: root))
            #expect(WorkspaceJobOrigins.load(workspaceRoot: root).isEmpty)
            try WorkspaceJobOrigins.record(macCommandLineOrigin(root: root), jobID: "after", workspaceRoot: root)
            #expect(Array(WorkspaceJobOrigins.load(workspaceRoot: root).keys) == ["after"])
            let aside = try FileManager.default.contentsOfDirectory(atPath: folder.path)
                .filter { $0.hasPrefix("origins.json.unreadable-") }
            #expect(aside.count == 1)
        }
    }

    /// Writers on separate threads open separate descriptors, so the flock
    /// serializes them exactly as it serializes two processes.
    @Test func concurrentWritersNeverCorruptOrLoseRecords() throws {
        try withScratch { root, _ in
            let base = macCommandLineOrigin(root: root)
            let failures = Locked(0)
            DispatchQueue.concurrentPerform(iterations: 8) { writer in
                for index in 0..<10 {
                    do {
                        try WorkspaceJobOrigins.record(base, jobID: "w\(writer)-\(index)", workspaceRoot: root)
                    } catch {
                        failures.withLock { $0 += 1 }
                    }
                }
            }
            #expect(failures.withLock { $0 } == 0)
            #expect(WorkspaceJobOrigins.load(workspaceRoot: root).count == 80)
        }
    }

    @Test func discoveredCommandLineJobsAreNamedAndImportableInRecentJobs() throws {
        try withScratch { root, _ in
            let origin = macCommandLineOrigin(root: root)
            let connected = client("http://127.0.0.1:8799", identity: origin.serverIdentity)
            let found = try #require(StudyRemoteJobController.commandLineOrigin([origin], for: connected))
            #expect(found.experiment == "study-a")
            let label = StudyRemoteJobController.recentVerbLabel(kind: "study-submit-bundle", origin: found)
            #expect(label == "run (bundle)")
            #expect(ExperimentPanel.jobOffersEvidenceImport(verb: label, state: "succeeded"))
            // Another server's record does not describe this server's job.
            let elsewhere = client("http://127.0.0.1:8799", identity: "ssh://other.invalid:8080")
            #expect(StudyRemoteJobController.commandLineOrigin([origin], for: elsewhere) == nil)

            try WorkspaceJobOrigins.record(origin, jobID: "cli-job", workspaceRoot: root)
            let marked = WorkspaceJobOrigins.commandLineOrigins(
                workspaceRoot: root, serverIdentity: origin.serverIdentity, servingRoot: "/scratch/steerlab")
            #expect(marked["cli-job"]?.submitterDescription == "the Mac command line (steerlab-cli)")
            #expect(marked["cli-job"]?.commandLineSummary.contains("study-a") == true)
            let otherRoot = WorkspaceJobOrigins.commandLineOrigins(
                workspaceRoot: root, serverIdentity: origin.serverIdentity, servingRoot: "/elsewhere")
            #expect(otherRoot.isEmpty)
        }
    }
}

/// A tiny lock-protected box for the concurrency test's counter.
private final class Locked<Value>: @unchecked Sendable {
    private var value: Value
    private let lock = NSLock()
    init(_ value: Value) { self.value = value }
    func withLock<T>(_ body: (inout Value) -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body(&value)
    }
}
