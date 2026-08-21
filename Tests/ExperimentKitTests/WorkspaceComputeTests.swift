import Foundation
import Testing

@testable import ExperimentKit

/// A workspace declares what it computes on; the app stops guessing.
///
/// Every verb used to re-derive that intent from the live server pairing, and
/// they disagreed — freeze matched evidence against the server's substrate in
/// Mac-authority mode while promotion kept keying on this engine's compile-time
/// constant. In a cluster workspace that made the workspace's own vectors
/// foreign to it.
struct WorkspaceComputeTests {

    private func withRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "wc-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: root.appending(component: "runs"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        return try body(root)
    }

    private func writeRun(_ root: URL, name: String, substrate: String?) {
        let directory = root.appending(components: "runs", name)
        try? FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        var config: [String: Any] = ["runType": "sweep"]
        if let substrate { config["substrate"] = substrate }
        try? JSONSerialization.data(withJSONObject: config)
            .write(to: directory.appending(component: "config.json"))
    }

    @Test func aDeclaredBindingRoundTripsAndWins() throws {
        try withRoot { root in
            #expect(WorkspaceCompute.declared(root: root) == nil)
            #expect(!WorkspaceCompute.isDeclared(root: root))
            try WorkspaceCompute.declare(.cluster, root: root)
            #expect(WorkspaceCompute.declared(root: root) == .cluster)
            #expect(WorkspaceCompute.isDeclared(root: root))
            #expect(WorkspaceCompute.resolved(root: root) == .cluster)

            // A declaration beats the evidence: a cluster workspace that ran
            // one local smoke test is still a cluster workspace.
            for index in 0..<5 {
                writeRun(root, name: "r\(index)", substrate: "swift-mlx")
            }
            #expect(WorkspaceCompute.resolved(root: root) == .cluster)
        }
    }

    /// Existing workspaces predate the declaration. One holding fifty server
    /// runs must not be told it is a local MLX workspace because a config
    /// file is missing — that is precisely the state that produced the live
    /// failure.
    @Test func anUndeclaredWorkspaceIsReadFromItsOwnRuns() throws {
        try withRoot { root in
            for index in 0..<48 {
                writeRun(root, name: "s\(index)", substrate: "python-hf-transformers")
            }
            writeRun(root, name: "local", substrate: "swift-mlx")
            #expect(WorkspaceCompute.inferred(root: root) == .cluster)
            #expect(WorkspaceCompute.resolved(root: root) == .cluster)
            #expect(
                !WorkspaceCompute.isDeclared(root: root),
                "an inferred binding must not read as one the researcher chose")
        }
    }

    @Test func anEmptyWorkspaceInfersNothingAndFallsBackToLocal() throws {
        try withRoot { root in
            #expect(WorkspaceCompute.inferred(root: root) == nil)
            #expect(WorkspaceCompute.resolved(root: root) == .localMLX)
        }
    }

    /// A tie is not an inference. Guessing on a 1–1 split would flip the
    /// workspace's identity on the next run either way.
    @Test func aTieInfersNothing() throws {
        try withRoot { root in
            writeRun(root, name: "a", substrate: "python-hf-transformers")
            writeRun(root, name: "b", substrate: "swift-mlx")
            writeRun(root, name: "c", substrate: nil)
            #expect(WorkspaceCompute.inferred(root: root) == nil)
        }
    }

    @Test func theInferenceIsNotWrittenDown() throws {
        try withRoot { root in
            writeRun(root, name: "a", substrate: "python-hf-transformers")
            #expect(WorkspaceCompute.resolved(root: root) == .cluster)
            #expect(
                WorkspaceCompute.declared(root: root) == nil,
                "a persisted guess is indistinguishable from a decision")
        }
    }

    @Test func substratesAndExecutionPolicyMatchTheEngines() {
        #expect(WorkspaceCompute.cluster.substrate == "python-hf-transformers")
        #expect(WorkspaceCompute.localMLX.substrate == "swift-mlx")
        #expect(!WorkspaceCompute.cluster.allowsLocalExecution)
        #expect(WorkspaceCompute.localMLX.allowsLocalExecution)
    }
}
