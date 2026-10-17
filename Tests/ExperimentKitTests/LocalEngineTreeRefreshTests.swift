import Foundation
import Testing

@testable import ExperimentKit

/// Refresh Engine: after an app update the materialized engine was copied by
/// the earlier build. It is OFFERED (never part of the setup plan), and it
/// replaces the engine's sources while keeping the installed packages.
struct LocalEngineTreeRefreshTests {

    // MARK: - When it is offered

    private func observation(_ source: LocalEngineObservation.SourceState, app: String? = "new12345")
        -> LocalEngineObservation
    {
        LocalEngineObservation(source: source, appRevision: app)
    }

    @Test func anEngineCopiedByAnotherBuildIsOfferedARefresh() {
        let stale = observation(.engineRoot(path: "/fixture/Engine", stampRevision: "old99999"))
        #expect(
            LocalEngineDecisions.refreshOffer(stale)
                == .refreshEngine(destination: "/fixture/Engine", from: "old99999", to: "new12345"))
        // Still never a step of the plan: the setup does not force it.
        #expect(LocalEngineDecisions.engineSourceState(stale).actions.isEmpty)
        #expect(LocalEngineDecisions.engineSourceState(stale).advisory?.contains("Refresh Engine") == true)

        let unstamped = observation(.engineRoot(path: "/fixture/Engine", stampRevision: nil))
        #expect(
            LocalEngineDecisions.refreshOffer(unstamped)
                == .refreshEngine(destination: "/fixture/Engine", from: nil, to: "new12345"))
    }

    @Test func nothingIsOfferedWhenThereIsNothingToRefresh() {
        #expect(
            LocalEngineDecisions.refreshOffer(
                observation(.engineRoot(path: "/fixture/Engine", stampRevision: "new12345"))) == nil)
        #expect(LocalEngineDecisions.refreshOffer(observation(.checkout(path: "/code"))) == nil)
        #expect(
            LocalEngineDecisions.refreshOffer(
                observation(.materializable(destination: "/fixture/Engine", payload: "/p"))) == nil)
        // A dev bundle with no revision of its own cannot tell.
        #expect(
            LocalEngineDecisions.refreshOffer(
                observation(.engineRoot(path: "/fixture/Engine", stampRevision: "old99999"), app: nil))
                == nil)
    }

    // MARK: - What the first launch after an update reads

    @Test func theFirstLaunchAfterAnUpdateReadsTheEngineStamp() throws {
        let fm = FileManager.default
        let engine = fm.temporaryDirectory.appending(component: "steerlab-engine-\(UUID().uuidString)")
        let home = fm.temporaryDirectory.appending(component: "steerlab-home-\(UUID().uuidString)")
        try fm.createDirectory(at: engine.appending(path: "Server/steerlab_server"), withIntermediateDirectories: true)
        try fm.createDirectory(at: home, withIntermediateDirectories: true)
        defer {
            try? fm.removeItem(at: engine)
            try? fm.removeItem(at: home)
        }
        ExperimentRootOverrideLock.acquire()
        CodeResources.modeOverrideForTesting = .release
        CodeResources.engineRootOverrideForTesting = engine
        CodeResources.executableHomesOverrideForTesting = [home]
        defer {
            CodeResources.modeOverrideForTesting = nil
            CodeResources.engineRootOverrideForTesting = nil
            CodeResources.executableHomesOverrideForTesting = nil
            ExperimentRootOverrideLock.release()
        }
        func stamp(_ revision: String?) throws {
            let stamp = CodeResources.EngineStamp(
                sourceRevision: revision, appVersion: "0.9.9", materializedAt: "2026-10-17T00:00:00Z",
                fileCount: 1)
            try JSONEncoder().encode(stamp).write(
                to: engine.appending(component: CodeResources.EngineStamp.fileName))
        }
        try stamp("old99999")
        #expect(LocalEngineDecisions.materializedEngineIsFromAnotherBuild(appRevision: "new12345"))
        try stamp("new12345")
        #expect(!LocalEngineDecisions.materializedEngineIsFromAnotherBuild(appRevision: "new12345"))
        // No revision of its own (a dev bundle): it cannot tell, so it is quiet.
        try stamp("old99999")
        #expect(!LocalEngineDecisions.materializedEngineIsFromAnotherBuild(appRevision: nil))
    }

    // MARK: - The file work

    private struct Tree {
        let root: URL
        var server: URL { root.appending(components: "Engine", "Server") }
        var payload: URL { root.appending(component: "ServerPayload") }

        init() throws {
            root = FileManager.default.temporaryDirectory
                .appending(component: "steerlab-engine-refresh-\(UUID().uuidString)")
            let fm = FileManager.default
            // The engine as an earlier build left it, with its environment.
            try fm.createDirectory(
                at: server.appending(components: "steerlab_server", "removed_module"),
                withIntermediateDirectories: true)
            try Data("old".utf8).write(to: server.appending(components: "steerlab_server", "__init__.py"))
            try fm.createDirectory(
                at: server.appending(components: ".venv.nosync", "bin"), withIntermediateDirectories: true)
            try Data("python".utf8).write(to: server.appending(components: ".venv.nosync", "bin", "python"))
            try fm.createDirectory(
                at: server.appending(component: "steerlab_server.egg-info"), withIntermediateDirectories: true)
            // This build's payload.
            try fm.createDirectory(
                at: payload.appending(components: "steerlab_server", "new_module"),
                withIntermediateDirectories: true)
            try Data("new".utf8).write(to: payload.appending(components: "steerlab_server", "__init__.py"))
            try Data("x".utf8).write(to: payload.appending(components: "steerlab_server", "new_module", "a.py"))
            try Data("lock".utf8).write(to: payload.appending(component: "requirements-macos-arm64.lock"))
        }

        func remove() { try? FileManager.default.removeItem(at: root) }

        func read(_ parts: String...) throws -> String {
            var url = server
            for part in parts { url = url.appending(component: part) }
            return try String(contentsOf: url, encoding: .utf8)
        }
    }

    @Test func aRefreshReplacesTheSourcesAndKeepsTheInstalledPackages() throws {
        let tree = try Tree()
        defer { tree.remove() }
        let copied = try LocalEngineTreeRefresh.replaceSources(in: tree.server, from: tree.payload)
        #expect(copied == 3)
        #expect(try tree.read("steerlab_server", "__init__.py") == "new")
        #expect(try tree.read("steerlab_server", "new_module", "a.py") == "x")
        #expect(try tree.read(".venv.nosync", "bin", "python") == "python")
        let fm = FileManager.default
        #expect(fm.fileExists(atPath: tree.server.appending(component: "steerlab_server.egg-info").path))
        #expect(!fm.fileExists(atPath: tree.server.appending(components: "steerlab_server", "removed_module").path))
        // No staging folders are left beside it.
        let siblings = try fm.contentsOfDirectory(atPath: tree.server.deletingLastPathComponent().path)
        #expect(siblings == ["Server"])
    }

    /// Fails the final rename (the new tree into place).
    private final class FailingFinalMove: FileManager, @unchecked Sendable {
        override func moveItem(at srcURL: URL, to dstURL: URL) throws {
            if srcURL.lastPathComponent.hasPrefix("Server.incoming-"), dstURL.lastPathComponent == "Server" {
                throw CocoaError(.fileWriteNoPermission)
            }
            try super.moveItem(at: srcURL, to: dstURL)
        }
    }

    @Test func aFailedRefreshLeavesTheOriginalTreeAsItWas() throws {
        let tree = try Tree()
        defer { tree.remove() }
        #expect(throws: (any Error).self) {
            try LocalEngineTreeRefresh.replaceSources(
                in: tree.server, from: tree.payload, fileManager: FailingFinalMove())
        }
        #expect(try tree.read("steerlab_server", "__init__.py") == "old")
        #expect(try tree.read(".venv.nosync", "bin", "python") == "python")
        let fm = FileManager.default
        #expect(fm.fileExists(atPath: tree.server.appending(components: "steerlab_server", "removed_module").path))
        #expect(fm.fileExists(atPath: tree.server.appending(component: "steerlab_server.egg-info").path))
        let siblings = try fm.contentsOfDirectory(atPath: tree.server.deletingLastPathComponent().path)
        #expect(siblings == ["Server"])
    }
}
