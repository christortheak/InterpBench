import Foundation
import Testing

@testable import ExperimentKit

/// `vectors compare`'s THREE-OUTCOME exit contract (WP0-AGENT-SURFACE-AUDIT
/// §2.4), over the full 3×2 matrix of outcome × mode.
///
/// The audit's requirement is one sentence long and this suite is its whole
/// enforcement: *a CI script must be able to tell pass, diverged, and
/// could-not-compare apart from exit codes alone.* Before 2026-08-18 it could
/// not — could-not-compare arrived as `failed`/70 in JSON mode (the answer a
/// crash gives) and exit 1 in human mode (the answer a real divergence gives),
/// so a harness pointed at a mistyped path reported the same thing as a
/// harness that had found genuine cross-substrate drift.
///
/// Every expectation here is duplicated in `Server/tests/test_vector_parity.py`
/// (`test_compare_exit_matrix_*`) against the byte-identical fixture copies.
/// Two literals, one contract — the parity idiom this repo uses everywhere the
/// engines cannot import each other.
@Suite struct VectorsCompareExitContractTests {

    private static var fixturesDirectory: URL {
        URL(filePath: #filePath)
            .deletingLastPathComponent()
            .appending(components: "Fixtures", "parity")
    }

    private func fixture(_ name: String) -> String {
        Self.fixturesDirectory.appending(component: "\(name).safetensors").path
    }

    /// One invocation through the full parse → run path. `json` picks which
    /// half of the matrix is being asserted: the process exits with
    /// `outcome.exitCode` in human mode and `envelope.exitCode` in JSON mode,
    /// which is the split `main.swift` performs.
    private func run(_ args: [String]) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(sink: .discarding).run(
            namespace: "vectors", args)
    }

    private func exitCodes(_ args: [String]) async -> (human: Int32, json: Int32) {
        let human = await run(args)
        let machine = await run(args + ["--json"])
        return (human.exitCode, machine.envelope.exitCode)
    }

    // MARK: - The matrix

    @Test func passIsZeroInBothModes() async throws {
        let codes = await exitCodes([
            "compare", fixture("identical-a"), fixture("identical-b"),
        ])
        #expect(codes.human == 0)
        #expect(codes.json == 0)
        let outcome = await run([
            "compare", fixture("identical-a"), fixture("identical-b"), "--json",
        ])
        #expect(outcome.envelope.state == .ready)
        #expect(outcome.envelope.error == nil)
    }

    @Test func comparedAndDivergedIsOneAnd65WithTheParityGate() async throws {
        let codes = await exitCodes([
            "compare", fixture("orthogonal-a"), fixture("orthogonal-b"),
        ])
        #expect(codes.human == 1)
        #expect(codes.json == SteerLabCLIState.refused.exitCode)

        let outcome = await run([
            "compare", fixture("orthogonal-a"), fixture("orthogonal-b"),
            "--json",
        ])
        #expect(outcome.envelope.state == .refused)
        let error = try #require(outcome.envelope.error)
        #expect(error.code == LifecycleGate.parityThreshold.rawValue)
        // A gate-shaped refusal names its gate. This is the one outcome of the
        // three that is one, and keeping `gate` populated is what lets an
        // agent switch on the lifecycle vocabulary here.
        #expect(error.gate == LifecycleGate.parityThreshold.rawValue)
        #expect(error.repairAction.contains("extract"))
        // It COMPARED: the report is still the verb's product.
        #expect(outcome.envelope.result?["report"] != nil)
    }

    @Test func missingArtifactIsTwoAnd66NotFound() async throws {
        let missing = Self.fixturesDirectory
            .appending(component: "no-such-artifact.safetensors").path
        let codes = await exitCodes([
            "compare", missing, fixture("identical-b"),
        ])
        #expect(codes.human == 2)
        #expect(codes.json == SteerLabCLIState.notFound.exitCode)

        let outcome = await run([
            "compare", missing, fixture("identical-b"), "--json",
        ])
        #expect(outcome.envelope.state == .notFound)
        let error = try #require(outcome.envelope.error)
        #expect(error.code == "notFound")
        // NOT gate-shaped: nothing declined a well-formed request, the request
        // could not be answered. A `gate` here would put `notFound` in a
        // vocabulary it is not a member of.
        #expect(error.gate == nil)
        // The repair names BOTH operands and the shape of an artifact.
        #expect(error.repairAction.contains(missing))
        #expect(error.repairAction.contains(fixture("identical-b")))
        #expect(error.repairAction.contains(".safetensors"))
        #expect(error.repairAction.contains("sidecar"))
        // …and they are machine-readable, not only in prose.
        let paths = try #require(outcome.envelope.result?["operandPaths"])
        guard case .array(let operands) = paths else {
            Issue.record("operandPaths is not an array")
            return
        }
        #expect(operands.count == 2)
    }

    /// Hidden-size mismatch: the artifacts EXIST and cannot be compared. Same
    /// outcome as a missing file (2/66) because the caller's question has no
    /// answer either way — with a different repair, because "check your paths"
    /// would send them in a circle.
    @Test func hiddenSizeMismatchIsCouldNotCompareNotAFailedComparison() async throws {
        let codes = await exitCodes([
            "compare", fixture("narrow-a"), fixture("identical-b"),
        ])
        #expect(codes.human == 2)
        #expect(codes.json == SteerLabCLIState.notFound.exitCode)

        let outcome = await run([
            "compare", fixture("narrow-a"), fixture("identical-b"), "--json",
        ])
        let error = try #require(outcome.envelope.error)
        #expect(error.code == "notFound")
        #expect(error.reason.contains("hidden-size mismatch"))
        #expect(error.repairAction.contains("SAME model"))
        // Never the parity gate: this is not a comparison that failed.
        #expect(error.code != LifecycleGate.parityThreshold.rawValue)
    }

    /// Layer-count mismatch is the CONTRAST that keeps the third outcome
    /// meaningful: those artifacts are comparable, the intersection is
    /// compared, and the run succeeds.
    @Test func layerCountMismatchStaysAReportNotAnOutcome() async throws {
        let codes = await exitCodes([
            "compare", fixture("truncated-a"), fixture("identical-b"),
        ])
        #expect(codes.human == 0)
        #expect(codes.json == 0)
    }

    @Test func usageErrorsStayAt64() async throws {
        let outcome = await run(["compare", fixture("identical-a"), "--json"])
        #expect(outcome.envelope.exitCode == SteerLabCLIState.blocked.exitCode)
    }

    // MARK: - The error's own classification

    @Test func parityErrorKindSeparatesUnreadableFromIncomparable() throws {
        // The default must be the unreadable class — every reader throw site
        // relies on it.
        #expect(VectorParity.ParityError(reason: "x").kind == .unreadableArtifact)
        do {
            _ = try VectorParity.compare(
                nameA: "a", perLayerA: [[1, 2]],
                nameB: "b", perLayerB: [[1, 2, 3]])
            Issue.record("hidden-size mismatch did not throw")
        } catch let error as VectorParity.ParityError {
            #expect(error.kind == .incomparableArtifacts)
        }
    }

    @Test func theRepairNamesBothOperandsInEitherClass() {
        for kind in [
            VectorParity.ParityError.Kind.unreadableArtifact,
            .incomparableArtifacts,
        ] {
            let repair = ExperimentCLIRunner.parityCouldNotCompareRepair(
                kind: kind, pathA: "/a/one.safetensors", pathB: "/b/two.safetensors")
            #expect(repair.contains("/a/one.safetensors"), "\(kind)")
            #expect(repair.contains("/b/two.safetensors"), "\(kind)")
            #expect(repair.contains("experiment extract"), "\(kind)")
        }
    }
}
