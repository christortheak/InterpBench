import Foundation
import Testing
@testable import ExperimentKit

@Suite(.serialized) struct WorkspaceRepositoryTests {
    @Test func manifestsWithTheSameNameStayInTheirWorkspace() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "repository-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let first = ExperimentRepository(workspaceRoot: root.appending(component: "first"))
        let second = ExperimentRepository(workspaceRoot: root.appending(component: "second"))
        let a = ExperimentManifest(name: "shared", description: "first", modelID: "test/a")
        let b = ExperimentManifest(name: "shared", description: "second", modelID: "test/b")
        // This internal primitive is only used to seed fixtures here. Production
        // mutations still go through ExperimentStore's lifecycle admission.
        try first.persistAdmitted(a)
        try second.persistAdmitted(b)
        #expect(try first.load(name: "shared") == a)
        #expect(try second.load(name: "shared") == b)
        #expect(first.list() == [a])
        #expect(second.list() == [b])
    }

    @Test func resultDiscoveryUsesTheRepositoryWorkspace() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "results-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let firstRoot = root.appending(component: "first")
        let secondRoot = root.appending(component: "second")
        let manifest = ExperimentManifest(name: "shared", description: "", modelID: "test/model")
        let run = firstRoot.appending(components: "runs", "20260905T120000-exp-shared-run")
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(to: run.appending(component: "experiment.json"))
        let first = StudyResultRepository(workspaceRoot: firstRoot)
        let second = StudyResultRepository(workspaceRoot: secondRoot)
        let items = first.list(experimentName: "shared")
        #expect(items.count == 1)
        let item = try #require(items.first)
        #expect(URL(filePath: item.path).resolvingSymlinksInPath().path
            == run.resolvingSymlinksInPath().path)
        #expect(second.list(experimentName: "shared").isEmpty)
        let detail = first.detail(for: item)
        #expect(detail.generations.isEmpty)
        #expect(detail.judgments.isEmpty)
        #expect(detail.report == nil)
    }
}
