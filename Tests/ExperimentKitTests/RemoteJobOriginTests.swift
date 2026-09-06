import Foundation
import Testing
@testable import ExperimentKit

@MainActor struct RemoteJobOriginTests {
    @Test func originPersistsAndSelectionCannotRedirectJobActions() throws {
        let suite = "job-origin-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let first = ClusterClient(profile: .init(name: "First", baseURL: URL(string: "http://first.invalid")!))
        let second = ClusterClient(profile: .init(name: "Second", baseURL: URL(string: "http://second.invalid")!))
        let root = URL(filePath: "/private/tmp/fictional-workspace")
        let jobs = StudyRemoteJobController(defaults: defaults)
        jobs.recordOrigin(.init(connection: first.profile, workspaceRoot: root), jobID: "job")
        let restarted = StudyRemoteJobController(defaults: defaults)
        #expect(try restarted.origin(for: "job").workspaceRoot == root.standardizedFileURL)
        #expect(try restarted.clientForJob("job", connected: first).profile == first.profile)
        #expect(throws: (any Error).self) { try restarted.clientForJob("job", connected: second) }
        #expect(throws: (any Error).self) { try restarted.clientForJob("legacy", connected: first) }
    }

    @Test func collidingJobIDsNeverAcquireTheOtherServersAuthority() throws {
        let suite = "job-collision-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let jobs = StudyRemoteJobController(defaults: defaults)
        for host in ["first.invalid", "second.invalid"] {
            jobs.recordOrigin(.init(connection: .init(baseURL: URL(string: "http://\(host)")!),
                                    workspaceRoot: URL(filePath: "/private/tmp/fictional-workspace")), jobID: "same-id")
        }
        #expect(throws: (any Error).self) { try jobs.origin(for: "same-id") }
    }
}
