import Foundation
import Testing
@testable import ExperimentKit
@testable import SteeringKit

/// The judge-rubric template (2026-07-20 researcher round, item 4a): a
/// workspace with no `prompts/rubrics/` files rendered an empty, useless
/// rubric picker — the template gives the researcher a domain-neutral
/// starting point. These tests pin the registration, the shipped seed, the
/// readiness-row wiring, and the scaffold's never-overwrite rule.
struct RubricTemplateTests {

    private func withTempRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "rubric-template-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        return try body(root)
    }

    private func judgedManifest(name: String = "rubric-study") -> ExperimentManifest {
        var manifest = ExperimentManifest(
            name: name, description: "", modelID: "test/model")
        manifest.judges = [
            ExperimentManifest.JudgeRef(name: "j1", kind: "claude"),
            ExperimentManifest.JudgeRef(name: "j2", kind: "local"),
        ]
        return manifest
    }

    // MARK: - Registration

    @Test func judgeRubricTemplateIsRegistered() {
        #expect(DataTemplates.template(id: "judge-rubric") == DataTemplates.judgeRubric)
        #expect(DataTemplates.all.contains(DataTemplates.judgeRubric))
        #expect(
            DataTemplates.judgeRubric.seedRelativePath
                == "prompts/templates/rubrics/rubric-template.md")
    }

    @Test func judgeRubricDestinationLivesWhereTheStoreScans() {
        // The rubric picker offers `JudgeRubricStore.list()` (a scan of
        // prompts/rubrics/), so the scaffold destination must land there.
        #expect(
            DataTemplates.judgeRubricDestination(experiment: "s1")
                == "prompts/rubrics/s1-rubric.md")
        #expect(
            DataTemplates.judgeRubricDestination(experiment: "s1")
                .hasPrefix(JudgeRubricStore.relativeDirectory + "/"))
    }

    // MARK: - Shipped seed validity

    @Test func judgeRubricSeedIsNonEmptyUTF8Text() throws {
        let seed = DataTemplates.seedURL(
            for: DataTemplates.judgeRubric,
            workspaceRoot: URL(filePath: "/nonexistent"))
        let data = try Data(contentsOf: seed)
        let text = String(decoding: data, as: UTF8.self)
        // Rubrics are raw text handed to the judge whole — the only shape
        // requirement is real, decodable text.
        #expect(Data(text.utf8) == data)
        #expect(!text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        // Domain-neutral by policy: the template must not smuggle in the
        // judicial program's vocabulary.
        #expect(!text.lowercased().contains("sentenc"))
        #expect(!text.lowercased().contains("judicial"))
    }

    // MARK: - Readiness-row wiring

    @Test func unpinnedJudgeRubricRowCarriesTemplateAndFileDestination() throws {
        try withTempRoot { root in
            let manifest = judgedManifest()
            let rows = StudyDataReadiness.requirements(
                for: manifest, workspaceRoot: root)
            let rubric = try #require(rows.first { $0.id == "judgeRubric" })
            #expect(rubric.status == .missing)
            #expect(rubric.templateID == DataTemplates.judgeRubric.id)
            // A concrete FILE destination (not the bare directory), so the
            // scaffold can create the starting file there.
            #expect(rubric.path == "prompts/rubrics/rubric-study-rubric.md")
        }
    }

    @Test func pinnedJudgeRubricRowsNeverOfferTheTemplate() throws {
        // Template bytes can never restore a PINNED rubric's hash — the
        // pinned rows (present or missing-on-disk) must not offer creation.
        try withTempRoot { root in
            var manifest = judgedManifest()
            manifest.judgeRubricFile = "prompts/rubrics/r1.md"

            let missing = StudyDataReadiness.requirements(
                for: manifest, workspaceRoot: root)
            let missingRow = try #require(missing.first { $0.id == "judgeRubric" })
            #expect(missingRow.status == .missing)
            #expect(missingRow.templateID == nil)

            let url = root.appending(path: "prompts/rubrics/r1.md")
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try "Compare the responses.".write(
                to: url, atomically: true, encoding: .utf8)
            let present = StudyDataReadiness.requirements(
                for: manifest, workspaceRoot: root)
            let presentRow = try #require(present.first { $0.id == "judgeRubric" })
            #expect(presentRow.status == .present)
            #expect(presentRow.templateID == nil)
        }
    }

    // MARK: - Scaffolding

    @Test func scaffoldCreatesRubricFromSeedAndRefusesOverwrite() throws {
        try withTempRoot { root in
            let manifest = judgedManifest()
            let rows = StudyDataReadiness.requirements(
                for: manifest, workspaceRoot: root)
            let rubric = try #require(rows.first { $0.id == "judgeRubric" })

            let created = try StudyDataReadiness.scaffold(
                requirement: rubric, in: root)
            #expect(
                created.path
                    == root.appending(
                        path: "prompts/rubrics/rubric-study-rubric.md").path)
            let createdText = try String(contentsOf: created, encoding: .utf8)
            let seed = DataTemplates.seedURL(
                for: DataTemplates.judgeRubric, workspaceRoot: root)
            #expect(createdText == (try String(contentsOf: seed, encoding: .utf8)))

            // Never overwrites — the researcher's edits are safe.
            #expect(throws: ExperimentError.self) {
                try StudyDataReadiness.scaffold(requirement: rubric, in: root)
            }
        }
    }

    // MARK: - The no-rubric refusal is TYPED (WP0 dry run #2, skipped check)

    /// `evaluate` on a study with a completed run and no rubric anywhere was
    /// refused in prose but untyped: `verbFailed` / exit 70 / "read the
    /// reason and repair the named input", which tells an agent nothing it
    /// can act on. It is a `missingPrerequisite` — the verb needs something
    /// the study never declared — and its repair is a runnable command. The
    /// reason string is unchanged, and is byte-identical to the server's
    /// `tasks.no_rubric_refusal` (which names `steerlab-cli` on purpose:
    /// authoring is Mac-authority and the Python CLI has no `pin-rubric`).
    @Test func anEvaluationWithNoRubricRefusesAsAMissingPrerequisite() throws {
        let manifest = judgedManifest(name: "no-rubric")
        let error = #expect(throws: ExperimentError.self) {
            _ = try JudgeRubricStore.resolveRubric(
                for: manifest, inlineRubric: nil)
        }
        let refusal = try #require(error?.lifecycleRefusal)
        #expect(refusal.gate == .missingPrerequisite)
        #expect(refusal.gateID == "missingPrerequisite")
        #expect(
            refusal.reason
                == "study 'no-rubric' has no judge rubric — pin one: "
                + "'steerlab-cli experiment pin-rubric no-rubric "
                + "prompts/rubrics/default-paired-v1.md' (any file under "
                + "prompts/rubrics/; inline draft text is draft-only and "
                + "cannot freeze)")
        #expect(refusal.reason == JudgeRubricStore.noRubricRefusal("no-rubric"))
        #expect(refusal.repairAction.hasPrefix("steerlab-cli experiment pin-rubric"))
        #expect(refusal.repairAction.contains("experiment evaluate no-rubric"))
    }

    /// Whitespace is not a rubric — the same refusal, so neither engine can
    /// hand a judge a blank instruction sheet.
    @Test func aWhitespaceOnlyInlineRubricIsNoRubric() {
        let manifest = judgedManifest(name: "blank-rubric")
        #expect(throws: ExperimentError.self) {
            _ = try JudgeRubricStore.resolveRubric(
                for: manifest, inlineRubric: "   \n\t ")
        }
    }
}
