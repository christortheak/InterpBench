import Foundation
import Testing

@testable import ExperimentKit

/// The judge panel's LOADED-MODEL cache, keyed by model AND revision.
///
/// Review round 9, finding 1: the cache was keyed by `judge.model` alone. A
/// panel naming one model at two revisions — a stability check across
/// checkpoints is exactly that panel — loaded revision A for the first judge
/// and then handed A's container to the judge pinned to B, while the judgment
/// rows stamped `judgeRevision: B`. Wrong weights under a provenance stamp
/// that named bytes which never ran.
///
/// The loads are counted through `loadLocalJudgeContainers`' generic
/// `Container`: the study path instantiates it with a `ModelContainer`, these
/// tests with a marker string, so the cache identity is provable on a machine
/// with no weights on it at all.
@Suite struct JudgeContainerCacheTests {

    /// What the fake loader returns: the exact (model, revision) pair the
    /// load was asked for. A judge that reads back a marker naming a revision
    /// other than its own is judging with weights it did not declare.
    private func marker(_ judge: ExperimentTasks.ResolvedJudge) -> String {
        "\(judge.model)@\(judge.revision ?? "unpinned")"
    }

    private func judge(
        _ name: String, model: String = "test/model", revision: String? = nil,
        kind: String = "local"
    ) -> ExperimentTasks.ResolvedJudge {
        ExperimentTasks.ResolvedJudge(
            name: name, kind: kind, model: model, revision: revision)
    }

    @Test func twoRevisionsOfOneModelLoadTwiceAndNeverShareWeights() async throws {
        let a = judge("judge-1", revision: String(repeating: "a", count: 40))
        let b = judge("judge-2", revision: String(repeating: "b", count: 40))
        var loaded: [String] = []
        let containers = try await ExperimentTasks.loadLocalJudgeContainers(
            for: [a, b]
        ) { judge in
            loaded.append(self.marker(judge))
            return self.marker(judge)
        }
        #expect(loaded.count == 2, "one model at two revisions is two loads")
        #expect(containers.count == 2)
        // The evidence claim: each judge reads back the container loaded for
        // ITS OWN revision — the revision its judgment rows stamp.
        #expect(
            containers[ExperimentTasks.LoadedModelKey(a)] == marker(a))
        #expect(
            containers[ExperimentTasks.LoadedModelKey(b)] == marker(b))
    }

    @Test func oneModelAtOneRevisionStillLoadsOnce() async throws {
        let pin = String(repeating: "c", count: 40)
        let judges = [
            judge("judge-1", revision: pin), judge("judge-2", revision: pin),
        ]
        var loads = 0
        let containers = try await ExperimentTasks.loadLocalJudgeContainers(
            for: judges
        ) { judge in
            loads += 1
            return self.marker(judge)
        }
        // The sharing the cache exists for is intact: same model, same pin,
        // one set of weights for both judges.
        #expect(loads == 1)
        #expect(containers.count == 1)
        for judge in judges {
            #expect(
                containers[ExperimentTasks.LoadedModelKey(judge)]
                    == "test/model@\(pin)")
        }
    }

    @Test func anUnpinnedJudgeIsItsOwnKeyNotAWildcard() async throws {
        let unpinned = judge("judge-1")
        let pinned = judge("judge-2", revision: String(repeating: "d", count: 40))
        var loads = 0
        let containers = try await ExperimentTasks.loadLocalJudgeContainers(
            for: [unpinned, pinned]
        ) { judge in
            loads += 1
            return self.marker(judge)
        }
        // An unpinned judge loads through the loader's own resolution and
        // stamps nil; a judge that named a commit declared something else.
        // Two declarations, two containers — never one standing for both.
        #expect(loads == 2)
        #expect(
            containers[ExperimentTasks.LoadedModelKey(unpinned)]
                == "test/model@unpinned")
        #expect(
            containers[ExperimentTasks.LoadedModelKey(pinned)]
                == marker(pinned))
    }

    @Test func onlyLocalJudgesLoadAnything() async throws {
        let judges = [
            judge("claude", model: "claude-x", kind: "claude"),
            judge("or", model: "google/gemma-3-27b-it", kind: "openrouter"),
        ]
        var loads = 0
        let containers = try await ExperimentTasks.loadLocalJudgeContainers(
            for: judges
        ) { judge in
            loads += 1
            return self.marker(judge)
        }
        #expect(loads == 0)
        #expect(containers.isEmpty)
    }

    /// A load failure is not swallowed by the cache: the first judge's
    /// refusal is the run's refusal, and nothing partial is handed back.
    @Test func aLoadFailureStopsThePanel() async {
        let judges = [judge("judge-1"), judge("judge-2", model: "other/model")]
        await #expect(throws: ExperimentError.self) {
            _ = try await ExperimentTasks.loadLocalJudgeContainers(for: judges) {
                judge in
                guard judge.model == "test/model" else {
                    throw ExperimentError(
                        reason: ExperimentTasks.localJudgeNotInstalledMessage(
                            judgeName: judge.name, model: judge.model,
                            revision: judge.revision))
                }
                return self.marker(judge)
            }
        }
    }

    /// The log line the two loops print names the pin, so two loads of one
    /// model id are distinguishable in a run log.
    @Test func theLoadLineNamesThePinnedRevision() {
        #expect(
            ExperimentTasks.localJudgeLoadLogLine(judge("judge-1"))
                == "loading local judge model test/model")
        #expect(
            ExperimentTasks.localJudgeLoadLogLine(
                judge("judge-2", revision: String(repeating: "e", count: 40)))
                == "loading local judge model test/model at revision eeeeeeeeeeee…")
        // A blank revision is no revision, not an empty pin printed as one.
        #expect(
            ExperimentTasks.localJudgeLoadLogLine(judge("judge-3", revision: "  "))
                == "loading local judge model test/model")
    }
}
