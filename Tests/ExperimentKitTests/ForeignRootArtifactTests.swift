import Foundation
import Testing

@testable import ExperimentKit

/// An artifact reference recorded on the CLUSTER's filesystem must still
/// resolve on the Mac.
///
/// Live failure, 2026-07-26: a hand-created agent in a cluster workspace
/// stored `/scratch/<user>/steerlab-workspace/runs/<run>/neuroticism` as its
/// `vectorArtifactID` — the path the researcher was looking at on the server.
/// The same run was sitting in the Mac's `runs/`, imported, but the upload
/// packer threw `MissingVariantDependency` because nothing resolved. The
/// relative/absolute rule this type was built for assumes both shapes name
/// the same filesystem; a cluster workspace breaks that assumption.
struct ForeignRootArtifactTests {

    private func withWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "foreign-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        ExperimentRootOverrideLock.acquire()
        WorkspaceRoot.programmaticOverride = temp
        defer {
            WorkspaceRoot.programmaticOverride = nil
            ExperimentRootOverrideLock.release()
            try? FileManager.default.removeItem(at: temp)
        }
        return try body(temp)
    }

    private func plantVector(_ workspace: URL, run: String, leaf: String) {
        let directory = workspace.appending(components: "runs", run)
        try? FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try? Data("{}".utf8).write(
            to: directory.appending(component: "\(leaf).json"))
        try? Data("w".utf8).write(
            to: directory.appending(component: "\(leaf).safetensors"))
    }

    @Test func aClusterPathResolvesToTheImportedRunOfTheSameName() throws {
        try withWorkspace { workspace in
            let run = "20260726T162230362-exp-optimize-neuroticism-sweep"
            plantVector(workspace, run: run, leaf: "neuroticism")
            let resolved = ArtifactIdentity.resolve(
                "/scratch/user/steerlab-workspace/runs/\(run)/neuroticism")
            #expect(
                resolved.path
                    == workspace.appending(components: "runs", run, "neuroticism")
                        .standardizedFileURL.path,
                "the cluster path did not rebase: \(resolved.path)")
        }
    }

    /// The rebase is a fallback, never a redirect. A reference that already
    /// resolves is untouched, so this cannot silently repoint a working agent.
    @Test func aReferenceThatAlreadyResolvesIsNeverRebased() throws {
        try withWorkspace { workspace in
            plantVector(workspace, run: "r", leaf: "fear")
            let direct = workspace.appending(components: "runs", "r", "fear")
                .standardizedFileURL.path
            #expect(ArtifactIdentity.resolve(direct).path == direct)
        }
    }

    /// A foreign path whose tail is NOT present stays as it was — the
    /// refusal must still name the real missing dependency rather than
    /// pointing at some unrelated file.
    @Test func anAbsentTailIsLeftAloneRatherThanGuessedAt() throws {
        try withWorkspace { _ in
            let reference = "/scratch/u/ws/runs/never-imported/fear"
            #expect(ArtifactIdentity.resolve(reference).path == reference)
        }
    }

    /// A cluster root that itself contains a `runs` component must rebase on
    /// the LAST one, not the first.
    @Test func theRebaseUsesTheLastMatchingRootComponent() throws {
        try withWorkspace { workspace in
            plantVector(workspace, run: "r2", leaf: "v")
            let resolved = ArtifactIdentity.resolve(
                "/scratch/runs/steerlab/runs/r2/v")
            #expect(
                resolved.path
                    == workspace.appending(components: "runs", "r2", "v")
                        .standardizedFileURL.path)
        }
    }

    /// Relative references keep working exactly as before.
    @Test func workspaceRelativeReferencesAreUnaffected() throws {
        try withWorkspace { workspace in
            plantVector(workspace, run: "r", leaf: "fear")
            #expect(
                ArtifactIdentity.resolve("runs/r/fear").path
                    == workspace.appending(components: "runs", "r", "fear")
                        .standardizedFileURL.path)
        }
    }

    /// The WRITE-side inverse (`workspaceRelative`): an absolute path under
    /// the current workspace serializes as the portable relative form — the
    /// shape the Python engine writes (`os.path.relpath`, promote.py) and
    /// the only shape that survives shipping the artifact to the cluster.
    /// Live failure, 2026-08-04: six app-promoted agents carried absolute
    /// Mac paths and every cluster panel/run died resolving them literally.
    @Test func anAbsolutePathUnderTheWorkspaceSerializesRelative() throws {
        try withWorkspace { workspace in
            let absolute = workspace
                .appending(components: "runs", "r", "fear").path
            #expect(ArtifactIdentity.workspaceRelative(absolute)
                    == "runs/r/fear")
            // Round trip: the relative form resolves back to the same file.
            #expect(ArtifactIdentity.canonical(
                        ArtifactIdentity.workspaceRelative(absolute))
                    == ArtifactIdentity.canonical(absolute))
        }
    }

    /// Outside the workspace there is nothing to relativize against — the
    /// reference passes through untouched rather than being disguised as
    /// portable.
    @Test func pathsOutsideTheWorkspacePassThroughUnchanged() throws {
        try withWorkspace { _ in
            let foreign = "/scratch/u/other-workspace/runs/r/fear"
            #expect(ArtifactIdentity.workspaceRelative(foreign) == foreign)
            #expect(ArtifactIdentity.workspaceRelative("runs/r/fear")
                    == "runs/r/fear")
        }
    }

    /// Review round 10, finding 7: an already-relative reference is NORMALIZED
    /// rather than passed through. `prompts/x`, `./prompts/x` and
    /// `prompts/a/../x` name one file and used to compare as three, so a
    /// re-declaration spelling the path the other way read as a different file
    /// and could drop the hash pin standing beside it.
    @Test func relativeReferencesAreLexicallyNormalized() throws {
        try withWorkspace { workspace in
            let canonical = "prompts/tasks/x.jsonl"
            for spelling in [
                "prompts/tasks/x.jsonl",
                "./prompts/tasks/x.jsonl",
                "prompts/tasks/./x.jsonl",
                "prompts//tasks/x.jsonl",
                "prompts/tasks/a/../x.jsonl",
                "prompts/other/../tasks/x.jsonl",
                "./prompts/./tasks/a/b/../../x.jsonl",
            ] {
                #expect(
                    ArtifactIdentity.workspaceRelative(spelling) == canonical,
                    "\(spelling) must normalize to \(canonical)")
            }

            // A `..` that walks off the TOP escapes the root: it cannot be
            // resolved lexically without inventing one, so — like an absolute
            // path outside the workspace — it is returned verbatim.
            for escaping in [
                "../outside/x", "a/../../outside/x", "../../x", "..",
            ] {
                #expect(ArtifactIdentity.workspaceRelative(escaping) == escaping)
            }

            // References that reduce to nothing, and the empty string, are
            // returned as written for the same reason.
            #expect(ArtifactIdentity.workspaceRelative(".") == ".")
            #expect(ArtifactIdentity.workspaceRelative("a/..") == "a/..")
            #expect(ArtifactIdentity.workspaceRelative("") == "")

            // A trailing separator is SHAPE, not spelling: a directory
            // reference (`adapterDirectory`) keeps it.
            #expect(
                ArtifactIdentity.workspaceRelative("./runs/r/adapter/")
                    == "runs/r/adapter/")

            // And the normalization agrees with `canonical`: three spellings,
            // one resolved file.
            plantVector(workspace, run: "r", leaf: "fear")
            let resolved = ArtifactIdentity.canonical("runs/r/fear")
            #expect(ArtifactIdentity.canonical("./runs/r/fear") == resolved)
            #expect(ArtifactIdentity.canonical("runs/x/../r/fear") == resolved)
        }
    }
}
