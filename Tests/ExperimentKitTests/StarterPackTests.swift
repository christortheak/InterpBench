import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// The sample workspace (`SampleWorkspace/` + docs/examples/starter-study-pack.json):
/// working, domain-neutral defaults a new researcher modifies instead of
/// authoring blank files. Every file must parse with its REAL consumer — the
/// loaders the run paths use, never a lenient test-only parser — and the
/// whole pack must stay free of study-domain vocabulary (a domain leak in
/// the worked example would contaminate the stimulus-independence firewall
/// for anyone who starts from it).
///
/// WP1 moved this material: it is no longer seeded into every new workspace
/// (new workspaces are concept-empty). It ships as a SEPARATE, recipe-only
/// folder the user copies or opens on purpose — decision 5, "no vector bytes
/// ship, ever". `prompts/starter/` is retained in the research tree as
/// history and neither ships nor seeds; `SampleWorkspace/` is the live copy,
/// so these gates point there.
struct StarterPackTests {

    /// The shipped sample workspace, in the CODE checkout.
    private var starterRoot: URL {
        VectorCatalog.bundledSeedRoot.appending(
            components: "SampleWorkspace", "prompts")
    }

    private var conceptDirectory: URL {
        starterRoot.appending(components: "concepts", "formality")
    }

    private var tasksURL: URL {
        starterRoot.appending(components: "tasks", "starter-prompts.jsonl")
    }

    private var batteryURL: URL {
        starterRoot.appending(components: "batteries", "starter-battery.jsonl")
    }

    private var packURL: URL {
        VectorCatalog.bundledSeedRoot.appending(
            components: "docs", "examples", "starter-study-pack.json")
    }

    // MARK: - The worked concept loads through the real extraction loader

    @Test func starterConceptLoadsAsAStimulusSet() throws {
        let set = try StimulusSet(directory: conceptDirectory)
        // ≥20 genuinely contrastive pairs; equal counts because each line is
        // the content-matched twin of the same line on the other side.
        #expect(set.positive.count >= 20)
        #expect(set.negative.count == set.positive.count)
        #expect(!set.hash.isEmpty)
        // Content-matched, not identical: every pair differs.
        for (positive, negative) in zip(set.positive, set.negative) {
            #expect(positive != negative)
        }
    }

    @Test func starterValidationScenariosLoadThroughTheGateLoader() throws {
        let scenarios = try #require(
            try StimulusSet.loadValidation(directory: conceptDirectory))
        #expect(scenarios.count >= 12)
        // Both classes present — a one-sided validation set cannot gate.
        #expect(scenarios.contains { $0.expresses })
        #expect(scenarios.contains { !$0.expresses })
        // Never-named: the held-out scenarios must not use the concept's
        // own vocabulary (the emotion-paper discipline the validate gate
        // depends on).
        let conceptVocabulary: Set<String> = [
            "formal", "formality", "informal", "casual", "polite",
            "politeness", "register",
        ]
        for scenario in scenarios {
            let tokens = Set(MarkerRubric.tokens(of: scenario.text))
            #expect(
                tokens.isDisjoint(with: conceptVocabulary),
                "validation scenario names the concept: \(scenario.text)")
        }
    }

    @Test func starterMarkersLoadAndFire() throws {
        let rubric = try #require(MarkerRubric(directory: conceptDirectory))
        #expect(!rubric.words.isEmpty)
        // The rubric must actually fire on formal-register text …
        #expect(rubric.count(in: "We respectfully advise you to proceed.") > 0)
        // … and stay silent on casual text.
        #expect(rubric.count(in: "hey, come over whenever!") == 0)
    }

    // MARK: - Battery loads through the real battery runner

    /// Was `starterBatteryLoadsWithPinnedGradingModesOnly` — the shipped
    /// battery is format 2 since 2026-08-19, so its items declare `options`
    /// and no `grading` (nothing is generated to grade). The INTENT is
    /// unchanged: the shipped bytes load through the REAL runner, every item
    /// is decidable, and the shared validating pin accepts them.
    @Test func starterBatteryLoadsAsAnIsolatedChoiceBattery() throws {
        let battery = try CapabilityBattery(url: batteryURL)
        #expect(battery.items.count >= 15)
        // Isolated arming: the battery declares its own context, so the
        // reading means the same thing under any study that pins it.
        #expect(battery.isolated)
        #expect(battery.formatVersion == CapabilityBattery.currentFormat)
        #expect(battery.systemPrompt == nil)
        #expect(battery.promptMode == .chatAssistant)
        // Every item is a decidable choice: ≥3 options, the answer among
        // them, and a stable id (the resume key on the server).
        var identifiers = Set<String>()
        for item in battery.items {
            #expect(battery.scoring(for: item) == .choiceProbability)
            let options = item.options ?? []
            #expect(options.count >= 3)
            #expect(options.contains(item.answer))
            #expect(Set(options).count == options.count)
            if let id = item.id { identifiers.insert(id) }
        }
        #expect(identifiers.count == battery.items.count)
        // The shared validating pin accepts it (the same check the study
        // pack import and freeze evidence path run).
        let data = try Data(contentsOf: batteryURL)
        #expect(
            PinShapeValidation.capabilityBatteryShapeProblem(
                data, file: "prompts/batteries/starter-battery.jsonl") == nil)
    }

    // MARK: - Task prompts load through the run loop's parser

    @Test func starterTaskPromptsParseWithChoiceFieldsIntact() throws {
        let data = try Data(contentsOf: tasksURL)
        // The REAL consumer — duplicate-id refusal and schema checks
        // included.
        let prompts = try ExperimentTasks.parseTaskPrompts(data)
        #expect(prompts.count >= 15)
        // Half the set carries options+target so the answer-token
        // instrument works out of the box.
        let choice = prompts.filter { $0.options != nil }
        #expect(choice.count >= 7)
        for prompt in choice {
            let options = try #require(prompt.options)
            #expect(options.count == 2)
            let target = try #require(prompt.target)
            #expect(options.contains(target))
        }
        // The open items are real prompts, not empty rows.
        for prompt in prompts {
            #expect(!prompt.text.trimmingCharacters(in: .whitespaces).isEmpty)
        }
    }

    // MARK: - The study pack imports through the real importer

    @Test(
        .enabled(
            if: ResearchTreeFixtures.hasStarterStudyPack,
            """
            docs/examples/starter-study-pack.json is research-side (the \
            shipped docs set is curated and its walkthrough is not in it) — \
            nothing in a released tree can import the pack
            """))
    func starterStudyPackImportsAsADraftWithOnlyTheAttachViolation() throws {
        let json = try String(contentsOf: packURL, encoding: .utf8)
        try ExperimentRootOverrideLock.withTempRoot(prefix: "starterpack") { root in
            let (imported, violations, written) =
                try ExperimentStore.importStudyJSON(json)
            #expect(imported.name == "starter-study")
            #expect(imported.status == .draft)
            // The pack's data files landed in the workspace …
            #expect(written.sorted() == [
                "prompts/batteries/starter-battery.jsonl",
                "prompts/tasks/starter-prompts.jsonl",
            ])
            #expect(FileManager.default.fileExists(
                atPath: root.appending(
                    path: "prompts/tasks/starter-prompts.jsonl").path))
            // … and the named inputs were pinned from the written bytes.
            #expect(imported.taskPromptsHash != nil)
            #expect(imported.capabilityBatteryHash != nil)
            // The ONE expected finding — concepts attach in the app because
            // attaching computes the pins (documented in
            // docs/examples/README.md). Anything else is a broken pack.
            #expect(violations == ["no concepts or variants attached"])
        }
    }

    /// The pack's embedded files are byte-identical to the sample
    /// workspace's copies, so importing the pack into the sample workspace
    /// is an idempotent no-op for them — packs never overwrite differing
    /// files, so drift here would make the starter pack refuse in exactly
    /// the workspace it is written for.
    @Test(
        .enabled(
            if: ResearchTreeFixtures.hasStarterStudyPack,
            """
            docs/examples/starter-study-pack.json is research-side — see \
            starterStudyPackImportsAsADraftWithOnlyTheAttachViolation
            """))
    func starterStudyPackFilesMatchTheSeedBytes() throws {
        struct Pack: Decodable { let files: [String: String] }
        let pack = try JSONDecoder().decode(
            Pack.self, from: Data(contentsOf: packURL))
        let expected: [String: URL] = [
            "prompts/tasks/starter-prompts.jsonl": tasksURL,
            "prompts/batteries/starter-battery.jsonl": batteryURL,
        ]
        #expect(Set(pack.files.keys) == Set(expected.keys))
        for (path, seedURL) in expected {
            let seed = try String(contentsOf: seedURL, encoding: .utf8)
            #expect(pack.files[path] == seed, "pack drifted from seed: \(path)")
        }
    }

    // MARK: - The sample workspace opens as a workspace, and is recipe-only

    /// It must be usable exactly as shipped: `isWorkspace` accepts it (a
    /// prompts/ directory is enough — no marker with a creation timestamp
    /// belongs in a versioned tree), and its content sits at the FUNCTIONAL
    /// paths the tooling scans, so opening the folder is the whole setup.
    @Test func sampleWorkspaceOpensAndCarriesRecipeInputsOnly() throws {
        let root = VectorCatalog.bundledSeedRoot.appending(component: "SampleWorkspace")
        let fm = FileManager.default
        #expect(WorkspaceStore.isWorkspace(url: root))
        #expect(fm.fileExists(atPath: root.appending(component: "README.md").path))

        // Loadable through the real loaders from its shipped location.
        let concept = root.appending(components: "prompts", "concepts", "formality")
        #expect(try StimulusSet(directory: concept).positive.count >= 20)
        #expect((try StimulusSet.loadValidation(directory: concept) ?? []).count >= 12)
        #expect(MarkerRubric(directory: concept) != nil)
        #expect(
            try ExperimentTasks.parseTaskPrompts(Data(contentsOf: tasksURL))
                .count >= 15)
        #expect(try CapabilityBattery(url: batteryURL).items.count >= 15)

        // Recipe-only (decision 5): re-derivable inputs, no measurements.
        // Vectors do not transfer across substrates, and a shipped run
        // would be someone else's model on someone else's hardware.
        for absent in ["runs", "experiments", "adapters"] {
            #expect(
                !fm.fileExists(atPath: root.appending(component: absent).path),
                "the sample workspace must ship no \(absent)/")
        }
        var offenders: [String] = []
        if let enumerator = fm.enumerator(at: root, includingPropertiesForKeys: nil) {
            for case let url as URL in enumerator
            where ["safetensors", "npz", "bin", "pt"].contains(url.pathExtension) {
                offenders.append(url.lastPathComponent)
            }
        }
        #expect(offenders.isEmpty, "vector/weight bytes in the sample: \(offenders)")
    }

    /// New workspaces are CONCEPT-EMPTY since WP1: the worked example is a
    /// folder you open, not something every workspace silently inherits.
    @Test func workspaceSeedingNoLongerCarriesTheSampleContent() throws {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "starter-seed-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temp) }
        _ = try WorkspaceStore.create(at: temp)

        for absent in [
            "prompts/concepts/formality", "prompts/tasks/starter-prompts.jsonl",
            "prompts/batteries/starter-battery.jsonl",
        ] {
            #expect(
                !FileManager.default.fileExists(
                    atPath: temp.appending(path: absent).path),
                "sample content seeded into a new workspace: \(absent)")
        }
    }

    // MARK: - Hygiene: UTF-8, non-empty, domain-neutral

    /// Every starter data file (and the pack) is valid non-empty UTF-8.
    @Test func starterFilesAreNonEmptyUTF8() throws {
        // The sample-workspace files always; the pack only where it exists
        // (research-side). Everything else here SHIPS, so this gate keeps
        // its teeth in a released tree instead of skipping wholesale.
        var files = [tasksURL, batteryURL]
        if ResearchTreeFixtures.hasStarterStudyPack { files.append(packURL) }
        for name in ["positive.jsonl", "negative.jsonl", "validation.jsonl", "markers.json"] {
            files.append(conceptDirectory.appending(component: name))
        }
        for url in files {
            let data = try Data(contentsOf: url)
            let text = try #require(
                String(data: data, encoding: .utf8),
                "not UTF-8: \(url.lastPathComponent)")
            #expect(
                !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                "empty starter file: \(url.lastPathComponent)")
        }
    }

    /// The starter data is domain-neutral by hard rule: it seeds every new
    /// workspace, and the lead study's circularity firewall requires
    /// stimuli independent of legal outcomes — so no legal-domain
    /// vocabulary may appear anywhere in the starter data.
    @Test func starterDataContainsNoLegalDomainVocabulary() throws {
        let forbidden: Set<String> = [
            "judge", "judges", "court", "courts", "courtroom", "sentence",
            "sentences", "sentencing", "verdict", "plaintiff", "defendant",
            "jury", "attorney", "lawsuit", "criminal", "prison",
        ]
        var files = [tasksURL, batteryURL]
        for name in ["positive.jsonl", "negative.jsonl", "validation.jsonl", "markers.json"] {
            files.append(conceptDirectory.appending(component: name))
        }
        for url in files {
            let text = try String(contentsOf: url, encoding: .utf8)
            let tokens = Set(MarkerRubric.tokens(of: text))
            let hits = tokens.intersection(forbidden)
            #expect(
                hits.isEmpty,
                "legal-domain vocabulary \(hits.sorted()) in \(url.lastPathComponent)")
        }
    }
}
