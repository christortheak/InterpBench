import Foundation
import Testing
@testable import ExperimentKit

/// Kind-owned judge fields (field bug 2026-08-07): switching a judge row's
/// kind in the Studies panel used to persist the previous kind's fields into
/// the manifest — a local judge kept `provider` from its OpenRouter past, a
/// claim about a pin that does not exist for that kind. The design under
/// test: (1) the panel keeps a per-row, per-kind stash so an exploratory
/// toggle loses nothing within the session; (2) the manifest write funnel
/// serializes only the fields the current kind owns; (3) an existing stale
/// manifest draws a non-blocking advisory, never a refusal. Pure-CPU.
struct JudgeKindFieldOwnershipTests {

    /// The default task-prompts file `saveProtocol` pins for a model-output
    /// study — plant it so the save path can complete in a bare temp root.
    private func plantDefaultTaskPrompts(under root: URL) throws {
        let directory = root.appendingPathComponent("prompts/dev")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try Data("{\"prompt\": \"p1\"}\n".utf8).write(
            to: directory.appendingPathComponent("dev-prompts.jsonl"))
    }

    // MARK: - The ownership filter itself

    @Test func kindOwnedFilterKeepsExactlyTheSchemaFields() {
        let full = ExperimentManifest.JudgeRef(
            name: "j", kind: "local", model: "other/judge-12b",
            provider: "google-ai-studio", revision: "cafe01", dtype: "bfloat16")

        var local = full
        local.kind = "local"
        let keptLocal = local.keepingKindOwnedFields()
        #expect(keptLocal.provider == nil)
        #expect(keptLocal.model == "other/judge-12b")
        #expect(keptLocal.revision == "cafe01")
        #expect(keptLocal.dtype == "bfloat16")

        var openrouter = full
        openrouter.kind = "openrouter"
        let keptOR = openrouter.keepingKindOwnedFields()
        #expect(keptOR.provider == "google-ai-studio")
        #expect(keptOR.model == "other/judge-12b")
        #expect(keptOR.revision == nil)
        #expect(keptOR.dtype == nil)

        var claude = full
        claude.kind = "claude"
        let keptClaude = claude.keepingKindOwnedFields()
        #expect(keptClaude.model == "other/judge-12b")
        #expect(keptClaude.provider == nil)
        #expect(keptClaude.revision == nil)
        #expect(keptClaude.dtype == nil)

        // A blank kind resolves to claude (the resolvedJudgeIdentity rule).
        var blank = full
        blank.kind = "  "
        #expect(blank.keepingKindOwnedFields().provider == nil)

        // An unrecognized kind keeps every field — this build cannot know
        // what it owns, and destroying data is worse than carrying it.
        var unknown = full
        unknown.kind = "future-kind"
        #expect(unknown.keepingKindOwnedFields() == unknown)
    }

    // MARK: - The per-row, per-kind stash (panel state)

    @MainActor
    @Test func kindSwitchRoundTripRestoresEachKindsFields() {
        let panel = ExperimentPanel()
        panel.judges = [
            .init(name: "judge-1", kind: "openrouter",
                  model: "google/gemma-3-27b-it", provider: "google-ai-studio")
        ]

        // openrouter → local: the row swaps to local's (empty) field set —
        // nothing of the OpenRouter past renders or saves under local.
        panel.setJudgeKind(at: 0, to: "local")
        #expect(panel.judges[0].kind == "local")
        #expect(panel.judges[0].model == nil)
        #expect(panel.judges[0].provider == nil)

        // The researcher configures the local kind.
        panel.judges[0].model = "other/judge-12b"
        panel.judges[0].revision = "cafe01"
        panel.judges[0].dtype = "bfloat16"

        // local → openrouter: the hand-discovered provider slug is restored
        // — losing it to an exploratory toggle would be hostile.
        panel.setJudgeKind(at: 0, to: "openrouter")
        #expect(panel.judges[0].model == "google/gemma-3-27b-it")
        #expect(panel.judges[0].provider == "google-ai-studio")
        #expect(panel.judges[0].revision == nil)
        #expect(panel.judges[0].dtype == nil)

        // openrouter → local again: the local entries came back too.
        panel.setJudgeKind(at: 0, to: "local")
        #expect(panel.judges[0].model == "other/judge-12b")
        #expect(panel.judges[0].revision == "cafe01")
        #expect(panel.judges[0].dtype == "bfloat16")
        #expect(panel.judges[0].provider == nil)

        // Same-kind and stale-index writes are no-ops, not traps.
        panel.setJudgeKind(at: 0, to: "local")
        #expect(panel.judges[0].model == "other/judge-12b")
        panel.setJudgeKind(at: 7, to: "openrouter")
        #expect(panel.judges.count == 1)
    }

    @MainActor
    @Test func removingAJudgeRowShiftsTheStashWithTheRows() {
        let panel = ExperimentPanel()
        panel.judges = [
            .init(name: "a", kind: "openrouter", model: "slug/a", provider: "pa"),
            .init(name: "b", kind: "openrouter", model: "slug/b", provider: "pb"),
        ]
        panel.setJudgeKind(at: 0, to: "local")
        panel.setJudgeKind(at: 1, to: "local")

        panel.removeJudge(at: 0)
        #expect(panel.judges.map(\.name) == ["b"])
        // Row b kept ITS stash under its new index: toggling back restores
        // b's fields, not a's.
        panel.setJudgeKind(at: 0, to: "openrouter")
        #expect(panel.judges[0].model == "slug/b")
        #expect(panel.judges[0].provider == "pb")
    }

    // MARK: - The write funnel (saveProtocol)

    @MainActor
    @Test func savedManifestCarriesOnlyKindOwnedJudgeFields() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "kindown") { root in
            // saveProtocol pins the default task-prompts file; without it on
            // disk the save throws and the round trip silently tests
            // nothing (the reload would show the PLANTED bytes).
            try plantDefaultTaskPrompts(under: root)
            var manifest = ExperimentManifest(
                name: "kind-owned", description: "d", modelID: "test/model")
            // A stale manifest as the field bug produced it: the local judge
            // kept "provider" from its OpenRouter past; the openrouter judge
            // kept local pins from a toggle the other way.
            manifest.judges = [
                .init(name: "loc", kind: "local", model: "other/judge-12b",
                      provider: "google-ai-studio", revision: "cafe01",
                      dtype: "bfloat16"),
                .init(name: "or", kind: "openrouter",
                      model: "google/gemma-3-27b-it", provider: "DeepInfra",
                      revision: "beef02", dtype: "float16"),
            ]
            try ExperimentStore.save(manifest, allowCreate: true)

            let panel = ExperimentPanel()
            panel.selectedName = "kind-owned"
            panel.saveProtocol()

            let reloaded = try ExperimentStore.load(name: "kind-owned")
            // Canary that the save actually ran (it pins the prompts file):
            // a swallowed save error would leave the planted bytes intact
            // and pass the stale-field checks below for the wrong reason.
            #expect(reloaded.taskPromptsHash != nil)
            let loc = reloaded.judges?.first { $0.name == "loc" }
            // The stale provider is gone; the local pins SURVIVE the save
            // (the reconstruction used to drop revision/dtype entirely).
            #expect(loc?.provider == nil)
            #expect(loc?.model == "other/judge-12b")
            #expect(loc?.revision == "cafe01")
            #expect(loc?.dtype == "bfloat16")
            let or = reloaded.judges?.first { $0.name == "or" }
            #expect(or?.provider == "DeepInfra")
            #expect(or?.revision == nil)
            #expect(or?.dtype == nil)
        }
    }

    // MARK: - The cleanliness advisory (non-blocking, freeze-advisory precedent)

    @Test func advisoryFiresOnAPlantedStaleManifestAndNamesTheFields() throws {
        var manifest = ExperimentManifest(
            name: "adv", description: "d", modelID: "test/model")
        // Blank-model local judge (resolves to the study model — needs no
        // pins of its own) carrying a stale OpenRouter provider: exactly the
        // field-bug artifact.
        manifest.judges = [
            .init(name: "loc", kind: "local", provider: "google-ai-studio")
        ]
        let advisory = ExperimentStore.kindForeignJudgeFieldsAdvisory(manifest)
        #expect(advisory?.contains("'loc'") == true)
        #expect(advisory?.contains("provider") == true)

        // The freeze-advisory surface carries it, non-blocking.
        try ExperimentRootOverrideLock.withTempRoot(prefix: "kindadv") { _ in
            let advisories = ExperimentStore.freezeAdvisories(for: manifest)
            #expect(advisories.contains {
                $0.contains("fields their kind does not own")
            })
        }

        // An openrouter judge with stale local pins is named field by field.
        manifest.judges = [
            .init(name: "or", kind: "openrouter", model: "slug/m",
                  provider: "DeepInfra", revision: "beef02", dtype: "float16")
        ]
        let orAdvisory = ExperimentStore.kindForeignJudgeFieldsAdvisory(manifest)
        #expect(orAdvisory?.contains("revision") == true)
        #expect(orAdvisory?.contains("dtype") == true)
        #expect(orAdvisory?.contains("provider") == false)

        // Clean panels (and whitespace-only leftovers) draw no advisory.
        manifest.judges = [
            .init(name: "loc", kind: "local", model: "other/judge-12b",
                  provider: "   ", revision: "cafe01", dtype: "bfloat16"),
            .init(name: "or", kind: "openrouter", model: "slug/m",
                  provider: "DeepInfra"),
        ]
        #expect(ExperimentStore.kindForeignJudgeFieldsAdvisory(manifest) == nil)
    }
}
