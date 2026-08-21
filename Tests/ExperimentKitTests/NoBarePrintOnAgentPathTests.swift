import Foundation
import Testing

@testable import ExperimentKit

/// The named guard for WP0-AGENT-SURFACE-AUDIT §7 step 4: **no bare `print`,
/// no bare file handle, and no `exit` in the moved dispatch.**
///
/// A grep-style test over source text, in the idiom
/// `NormBackfillGuidanceTests` established: the failure mode is a future edit
/// re-introducing a direct write, and no behavioural test would notice —
/// a stray `print` still puts the right bytes on the right stream when the
/// binary runs. It is only in-process testing, and step 5's `--json` mode,
/// that it breaks: a diagnostic printed to stdout outside the sink would
/// corrupt the one-JSON-document invariant that the whole envelope contract
/// rests on.
///
/// The sink's own definition (`ExperimentCLISink.standard`) is the one place
/// `print` and `FileHandle` legitimately appear; it lives in its own file, so
/// this test greps only the dispatch.
@Suite struct NoBarePrintOnAgentPathTests {

    private var repositoryRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // ExperimentKitTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repo root
    }

    private func source(_ path: String) throws -> String {
        try String(contentsOf: repositoryRoot.appending(path: path), encoding: .utf8)
    }

    /// Source with `//` comments stripped, so the file's own prose about
    /// `print(` and `exit(` does not trip its own guard.
    private func code(_ path: String) throws -> [(number: Int, text: String)] {
        try source(path).components(separatedBy: "\n").enumerated().compactMap {
            index, line in
            let text =
                line.range(of: "//").map { String(line[line.startIndex..<$0.lowerBound]) }
                ?? line
            return text.trimmingCharacters(in: .whitespaces).isEmpty
                ? nil : (index + 1, text)
        }
    }

    private static let dispatch = "Sources/ExperimentKit/ExperimentCLIRunner.swift"

    @Test func theMovedDispatchNeverCallsPrint() throws {
        let offenders = try code(Self.dispatch).filter { $0.text.contains("print(") }
        #expect(
            offenders.isEmpty,
            "bare print( in the moved dispatch at line(s) \(offenders.map(\.number)) — route it through `sink.out`")
    }

    @Test func theMovedDispatchNeverTouchesAFileHandleDirectly() throws {
        let offenders = try code(Self.dispatch).filter {
            $0.text.contains("FileHandle.standard")
        }
        #expect(
            offenders.isEmpty,
            "bare FileHandle write in the moved dispatch at line(s) \(offenders.map(\.number)) — route it through `sink.err`/`sink.outRaw`")
    }

    @Test func theMovedDispatchNeverCallsExit() throws {
        // The runner returns its exit code; a verb that called `exit` would be
        // untestable in-process and would kill the app if it ever drove the
        // same code path.
        let offenders = try code(Self.dispatch).filter {
            $0.text.range(of: "\\bexit\\(", options: .regularExpression) != nil
        }
        #expect(
            offenders.isEmpty,
            "bare exit( in the moved dispatch at line(s) \(offenders.map(\.number)) — throw `ExperimentCLIStop(exitCode:)` instead")
    }

    @Test func mainSwiftNoLongerCarriesTheMovedDispatch() throws {
        // The other half of the move: the binary must not keep a second copy.
        let main = try source("Sources/steerlab-cli/main.swift")
        for marker in [
            "func runExperimentCommand", "func runDataCommand",
            "func runVectorsCommand", "func runRemoteCommand",
            "func runWorkspaceCommand",
            "case \"set-style-taxonomy\":", "case \"rescore-style\":",
            "data check failed:",
        ] {
            #expect(!main.contains(marker), "main.swift still contains '\(marker)'")
        }
        // …and it does hand every one of those families over.
        #expect(main.contains("ExperimentCLIRunner.namespaces.contains(arguments[1])"))
        #expect(main.contains("ExperimentCLIRenderer.standardErrorText(outcome)"))
    }

    @Test func theSinkIsTheOnlyPlaceTheProcessStreamsAreNamed() throws {
        // Positive control: the guard above would pass trivially if the sink
        // had stopped writing anywhere at all.
        let sink = try source("Sources/ExperimentKit/ExperimentCLISink.swift")
        #expect(sink.contains("print(text, terminator: \"\")"))
        #expect(sink.contains("FileHandle.standardOutput.write"))
        #expect(sink.contains("FileHandle.standardError.write"))
    }
}
