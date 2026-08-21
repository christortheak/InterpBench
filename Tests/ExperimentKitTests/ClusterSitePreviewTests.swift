import Foundation
import Testing

@testable import ExperimentKit

/// WP5 Step 5 — the preview surfaces (`docs/WP5-PROFILE-MATERIALIZATION-AUDIT.md`
/// §3.3, §5). Pure-CPU tests for `ClusterSitePreview`, the one model behind all
/// three surfaces (site editor, setup wizard, `steerlab-cli cluster preview`).
///
/// The load-bearing assertion is `theThreeSurfacesShowTheCommittedGoldenBytes`:
/// a preview whose bytes are not the renderer's bytes is not a preview, it is a
/// second opinion — and a site admin approving it would be approving something
/// other than what runs. Everything else here (the default-set disclosure, the
/// secret rule, job-class filtering) is a property of THAT projection.
@MainActor
struct ClusterSitePreviewTests {

    // MARK: Fixtures

    /// The committed cross-engine profiles (WP5 §2.0). Same list as
    /// `ClusterEnvironmentRendererTests.crossEngineFixtures` — the preview is
    /// pinned to the SAME bytes the two renderers are. `nonisolated` so the
    /// `@Test(arguments:)` macro can read it from outside the main actor.
    nonisolated static let fixtures = ClusterEnvironmentRendererTests.crossEngineFixtures

    private static var fixtureDirectory: URL {
        URL(filePath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appending(components: "prompts", "fixtures", "cluster-site-profile")
    }

    private func fixtureProfile(_ name: String) throws -> ClusterSiteProfile {
        try ClusterSiteProfile.decode(
            from: try Data(
                contentsOf: Self.fixtureDirectory.appending(component: "\(name).json")))
    }

    private func golden(_ name: String) throws -> String {
        try String(
            contentsOf: Self.fixtureDirectory.appending(component: name), encoding: .utf8)
    }

    // MARK: - The projection (the gate)

    /// Every pane, on every committed fixture, byte-equal to the golden the two
    /// RENDERERS are pinned to — reached through the preview path rather than
    /// through the renderer directly, because that is the path the editor, the
    /// wizard, and the CLI actually take.
    @Test(arguments: ClusterSitePreviewTests.fixtures)
    func theThreeSurfacesShowTheCommittedGoldenBytes(_ fixture: String) throws {
        let profile = try fixtureProfile(fixture)

        // 1. The CLI / wizard path: the profile straight through the preview.
        let direct = ClusterSitePreview(profile)
        #expect(direct.envFile == (try golden("\(fixture).env.golden.txt")))
        #expect(direct.headerDocument == (try golden("\(fixture).headers.golden.txt")))
        #expect(
            direct.unresolvedDocument == (try golden("\(fixture).unresolved.golden.txt")))

        // 2. The editor path: the same profile through the form model. A clean
        //    load must not be dirty — if it were, the preview would be showing
        //    an environment the SAVED site does not imply.
        let model = SiteEditorModel(profile: profile)
        #expect(!model.isDirty, "\(fixture) did not round-trip through the editor")
        #expect(model.preview == direct)
        #expect(model.preview.envFile == (try golden("\(fixture).env.golden.txt")))
    }

    /// Editing a field moves every pane immediately — the preview is derived,
    /// never a snapshot taken at load.
    @Test func theEditorsPreviewFollowsTheFieldsAsTheyChange() throws {
        let model = SiteEditorModel(profile: try fixtureProfile("v2-neutral"))
        #expect(!model.preview.envFile.contains("STEERLAB_ROOT"))
        #expect(
            model.preview.unresolvedFacts.contains { $0.key == "STEERLAB_ROOT" },
            "an undeclared workspace root is an unresolved fact")

        model.workspaceRoot = "/scratch/me/ws"
        #expect(model.preview.envFile.contains("export STEERLAB_ROOT=\"/scratch/me/ws\""))
        // Declaring it retires the fact — the pane is a to-do list, so a stale
        // entry would send an admin to fix something already fixed.
        #expect(!model.preview.unresolvedFacts.contains { $0.key == "STEERLAB_ROOT" })
    }

    // MARK: - Which defaults are in play (Step 5 requirement 4)

    /// An admin reading a v1 site's preview must be told that LEGACY defaults
    /// apply, and told it in words — a schema number is not a disclosure to a
    /// reader who has not read the audit.
    @Test func everyPreviewStatesItsSchemaAndItsDefaultSet() throws {
        let v1 = ClusterSitePreview(try fixtureProfile("v1-maximal"))
        #expect(v1.schemaVersion == 1)
        #expect(v1.defaultSet == "legacyV1")
        #expect(v1.defaultSetSummary.contains("schema 1"))
        #expect(v1.defaultSetSummary.contains("LEGACY"))
        #expect(v1.defaultSetSummary.contains("bootstrap.sh"))

        let v2 = ClusterSitePreview(try fixtureProfile("v2-neutral"))
        #expect(v2.schemaVersion == ClusterSiteProfile.currentSchemaVersion)
        #expect(v2.defaultSet == "neutralV2")
        #expect(v2.defaultSetSummary.contains("NEUTRAL"))
        // Both the machine field and the sentence agree with the renderer's own
        // determination — one source, three spellings.
        for fixture in Self.fixtures {
            let profile = try fixtureProfile(fixture)
            #expect(
                ClusterSitePreview(profile).defaultSet
                    == ClusterEnvironmentRenderer.defaultSet(for: profile).rawValue)
        }
        // The human pane leads with it, so it cannot be scrolled past.
        #expect(v1.humanLines.first?.contains("LEGACY") == true)
    }

    // MARK: - The secret rule (§3.3 pane 1)

    /// The token reaches the preview as the `$(cat …)` PATH indirection the env
    /// file carries — never a value, on any surface, in any mode. Structural:
    /// there is no property on the preview that could hold one.
    @Test func noPaneCanCarryASecretValue() throws {
        var profile = try fixtureProfile("v2-maximal")
        profile.environment.tokenFilePath = "~/.secrets/steerlab-token"
        let preview = ClusterSitePreview(profile)
        #expect(
            preview.envFile.contains(
                "export STEERLAB_AUTH_TOKEN=\"$(cat \"$HOME/.secrets/steerlab-token\")\""))

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let json = try encoder.encode(preview)
        guard let object = try JSONSerialization.jsonObject(with: json) as? [String: Any]
        else {
            Issue.record("the preview did not encode to a JSON object")
            return
        }
        var keys: Set<String> = []
        func walk(_ value: Any) {
            if let object = value as? [String: Any] {
                for (key, nested) in object {
                    keys.insert(key.lowercased())
                    walk(nested)
                }
            } else if let array = value as? [Any] {
                array.forEach(walk)
            }
        }
        walk(object)
        for forbidden in ["token", "secret", "password", "credential"] {
            #expect(
                !keys.contains { $0.contains(forbidden) },
                "the preview grew a \(forbidden)-shaped field")
        }
    }

    // MARK: - Panes

    /// Pane 3 narrows to one class; nothing else does. A `--job-class` reader
    /// still needs the environment, the commands, and the fallbacks.
    @Test func filteringNarrowsOnlyTheHeaderPane() throws {
        let profile = try fixtureProfile("v2-maximal")
        let all = ClusterSitePreview(profile)
        let onlyStudy = ClusterSitePreview(profile, jobClasses: [.study])
        #expect(all.headers.count == ClusterEnvironmentRenderer.JobClass.allCases.count)
        #expect(onlyStudy.headers.map(\.jobClass) == ["study"])
        #expect(onlyStudy.headers.first?.lines == all.headers.first?.lines)
        #expect(onlyStudy.envFile == all.envFile)
        #expect(onlyStudy.gpuVocabulary == all.gpuVocabulary)
        #expect(onlyStudy.unresolvedFacts == all.unresolvedFacts)
        #expect(onlyStudy.schedulerCommands == all.schedulerCommands)
    }

    /// G4: the table an admin reads and the string the env file carries are the
    /// same inventory, so one can be checked against the other.
    @Test func theGPUPaneShowsTheInventoryAndTheValueItRendersTo() throws {
        let declared = ClusterSitePreview(try fixtureProfile("v2-maximal"))
        #expect(declared.gpuVocabulary.declared)
        #expect(!declared.gpuVocabulary.entries.isEmpty)
        #expect(
            declared.gpuVocabulary.typesValue
                == declared.gpuVocabulary.entries.map(\.type).joined(separator: ","))
        #expect(declared.gpuDocument.contains("STEERLAB_SLURM_GPU_TYPES="))
        #expect(declared.envFile.contains(
            "export STEERLAB_SLURM_GPU_TYPES=\"\(declared.gpuVocabulary.typesValue)\""))

        // A v1 site with no inventory inherits the legacy constant — and the
        // pane SAYS the profile declared none, which is the whole point of
        // previewing a legacy site.
        var bare = try fixtureProfile("v1-maximal")
        if case .slurm(var slurm) = bare.scheduler {
            slurm.gpus = []
            slurm.gpuTypes = []
            slurm.gpuVRAMGB = [:]
            bare.scheduler = .slurm(slurm)
        }
        let inherited = ClusterSitePreview(bare)
        #expect(!inherited.gpuVocabulary.declared)
        #expect(inherited.gpuVocabulary.typesValue == "L4,A100,H100")
        #expect(inherited.gpuDocument.contains("inherited from the legacy constant"))
        #expect(
            inherited.unresolvedFacts.contains { $0.key == "STEERLAB_SLURM_GPU_TYPES" })

        // A site with no scheduler at all: no vocabulary, no commands, and the
        // panes say so instead of rendering empty.
        let workstation = ClusterSitePreview(ClusterSiteProfile.gpuWorkstation)
        #expect(workstation.gpuVocabulary.isEmpty)
        #expect(workstation.gpuDocument.contains("declares no inventory"))
        #expect(workstation.schedulerCommands.isEmpty)
        #expect(workstation.schedulerCommandDocument.contains("no scheduler"))
        #expect(workstation.headers.allSatisfy { $0.lines.isEmpty })
    }

    /// §3.3 pane 2: a site that wraps `squeue` must show the wrapper, or the
    /// preview would describe commands the site never runs.
    @Test func theCommandPaneShowsWrappersNotTheRawBinaries() throws {
        var profile = try fixtureProfile("v2-maximal")
        if case .slurm(var slurm) = profile.scheduler {
            slurm.commands.query = "sq"
            profile.scheduler = .slurm(slurm)
        }
        let preview = ClusterSitePreview(profile)
        #expect(preview.schedulerCommands.map(\.role) == ["submit", "query", "accounting", "cancel"])
        #expect(preview.schedulerCommands.first { $0.role == "query" }?.command == "sq")
        #expect(preview.schedulerCommandDocument.contains("query\tsq"))
    }

    /// Pane 4 with nothing to say is a stated absence, never a blank box.
    @Test func anEmptyUnresolvedPaneSaysSoInBothForms() throws {
        let complete = ClusterSitePreview(try fixtureProfile("v2-maximal"))
        #expect(complete.unresolvedFacts.isEmpty)
        // The golden's convention for an empty pane (fixtures README).
        #expect(complete.unresolvedDocument == "\n")
        #expect(complete.humanLines.contains { $0.contains("unresolved facts (none") })
    }

    // MARK: - Determinism (Step 6 folds these bytes into the plan hash)

    @Test func rerenderingAnUnchangedProfileIsAByteIdenticalNoOp() throws {
        for fixture in Self.fixtures {
            let profile = try fixtureProfile(fixture)
            #expect(ClusterSitePreview(profile) == ClusterSitePreview(profile))
            #expect(
                ClusterSitePreview(profile).humanLines
                    == ClusterSitePreview(profile).humanLines)
        }
    }

    /// The human panes print the documents VERBATIM: an admin copies the env
    /// file out of a terminal, so no line may acquire a leading indent.
    @Test func humanPaneBodiesAreCopyableVerbatim() throws {
        let preview = ClusterSitePreview(try fixtureProfile("v1-maximal"))
        let lines = preview.humanLines
        for exported in preview.envFile.split(separator: "\n") {
            #expect(
                lines.contains(String(exported)),
                "env line was reformatted for the human pane: \(exported)")
        }
        for block in preview.headers {
            for line in block.lines { #expect(lines.contains(line)) }
        }
    }
}
