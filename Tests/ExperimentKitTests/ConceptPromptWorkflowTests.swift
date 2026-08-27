import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

@Suite(.serialized) @MainActor
struct ConceptPromptWorkflowTests {
    private func withTempWorkspace<T>(
        _ body: (URL) throws -> T
    ) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "concept-prompts") { root in
            let previous = WorkspaceRoot.programmaticOverride
            WorkspaceRoot.programmaticOverride = root
            defer { WorkspaceRoot.programmaticOverride = previous }
            return try body(root)
        }
    }

    @Test func everyRecipeCanGenerateAFirstDatasetPromptInAnOldWorkspace() throws {
        try withTempWorkspace { _ in
            let builder = ConceptBuilder()
            builder.conceptName = "practical wisdom"

            // Scoped to the families that read STIMULI. A derived direction
            // (J-lens) has no stimulus set to generate a prompt for, so
            // offering one would imply provenance it does not have — the
            // builder returns nil there deliberately. Keyed on the property
            // rather than naming the case, so a future derived family is
            // covered without editing this test.
            for recipe in ConceptBuilder.RecipeFamily.allCases
            where recipe.extractsFromStimuli {
                builder.recipeFamily = recipe
                let prompt = try #require(builder.generationPrompt())
                #expect(!prompt.isEmpty)
                #expect(!prompt.contains("Template unavailable"))
                #expect(prompt.lowercased().contains("practical-wisdom"))
            }

            // The complement, asserted rather than left implicit: a derived
            // family must NOT hand back a stimulus-generation prompt.
            for recipe in ConceptBuilder.RecipeFamily.allCases
            where !recipe.extractsFromStimuli {
                builder.recipeFamily = recipe
                #expect(builder.generationPrompt() == nil)
            }

            builder.recipeFamily = .emotionGrandMean
            let cowork = try #require(builder.coworkGenerationPrompt())
            #expect(!cowork.isEmpty)
            #expect(!cowork.contains("Template unavailable"))
        }
    }

    @Test func caaPromptDoesNotRequireAnExistingPair() {
        withTempWorkspace { _ in
            let builder = ConceptBuilder()
            builder.conceptName = "integrity"
            builder.recipeFamily = .caaMeanDifference
            #expect(builder.positives.isEmpty)
            #expect(builder.negatives.isEmpty)
            #expect(builder.generationPrompt()?.contains("integrity") == true)
        }
    }

    @Test func savingANewConceptMakesItSelectableWithoutDatasetRows() {
        withTempWorkspace { root in
            let builder = ConceptBuilder()
            builder.startNewConcept()
            builder.conceptName = "Practical Wisdom"
            builder.saveNewConcept()

            #expect(builder.selectedExisting == "practical-wisdom")
            #expect(builder.existingConcepts.contains("practical-wisdom"))
            #expect(
                FileManager.default.fileExists(
                    atPath: root.appending(
                        components: "prompts", "concepts", "practical-wisdom",
                        "concept.json").path))
        }
    }

    /// Selecting an existing concept must restore the recipe its dataset was
    /// built under, so "Copy LLM prompt" copies THAT recipe's prompt. The
    /// shared paired files can't distinguish CAA from RepE rows, but the
    /// recipe mirrors can (`prompts/repe/<name>/pairs.jsonl`,
    /// `prompts/readers/<name>/pairs.jsonl`) — before 2026-07-14 every
    /// paired concept defaulted the picker back to CAA on selection, so a
    /// RepE concept quietly copied the CAA dataset prompt.
    @Test func selectingAConceptRestoresItsRecipeForPromptCopy() throws {
        try withTempWorkspace { root in
            func writePairedFiles(_ concept: String) throws {
                let dir = root.appending(components: "prompts", "concepts", concept)
                try FileManager.default.createDirectory(
                    at: dir, withIntermediateDirectories: true)
                try #"{"text": "brave act"}"#.write(
                    to: dir.appending(component: "positive.jsonl"),
                    atomically: true, encoding: .utf8)
                try #"{"text": "plain act"}"#.write(
                    to: dir.appending(component: "negative.jsonl"),
                    atomically: true, encoding: .utf8)
            }
            func writeMirror(_ subtree: String, _ concept: String) throws {
                let dir = root.appending(components: "prompts", subtree, concept)
                try FileManager.default.createDirectory(
                    at: dir, withIntermediateDirectories: true)
                try #"{"positive": "brave act", "negative": "plain act"}"#.write(
                    to: dir.appending(component: "pairs.jsonl"),
                    atomically: true, encoding: .utf8)
            }
            try writePairedFiles("courage")
            try writeMirror("repe", "courage")
            try writePairedFiles("integrity")

            let builder = ConceptBuilder()
            builder.refreshConceptList()

            // The RepE-built concept restores RepE — and copies ITS prompt.
            builder.selectedExisting = "courage"
            #expect(builder.recipeFamily == .pairedDifferencePCA)
            let repePrompt = try #require(builder.generationPrompt())
            #expect(repePrompt.contains("Representation Engineering"))

            // A plain paired concept (no mirror) is CAA, prompt to match.
            builder.selectedExisting = "integrity"
            #expect(builder.recipeFamily == .caaMeanDifference)
            let caaPrompt = try #require(builder.generationPrompt())
            #expect(caaPrompt.contains("contrastive activation addition"))
            #expect(!caaPrompt.contains("Representation Engineering"))
        }
    }

    /// The mirror rule: neither mirror → CAA; exactly one mirror → its family.
    /// Both mirrors is the interesting case — it is decided by RECORDED
    /// PROVENANCE (a fitted reader artifact, or a vector sidecar's
    /// recipeMethod), never by file modification times, which a git checkout,
    /// a workspace copy, an rsync or a bundle unpack rewrites wholesale. With
    /// nothing recorded the answer is nil and the caller keeps its selection.
    @Test func pairedRecipeFamilyOnDiskReadsRecordedProvenance() throws {
        try withTempWorkspace { root in
            func writeMirror(_ subtree: String, date: Date) throws {
                let dir = root.appending(components: "prompts", subtree, "valor")
                try FileManager.default.createDirectory(
                    at: dir, withIntermediateDirectories: true)
                let url = dir.appending(component: "pairs.jsonl")
                try #"{"positive": "p", "negative": "n"}"#.write(
                    to: url, atomically: true, encoding: .utf8)
                // Deliberately BACKDATED relative to the other mirror: the old
                // rule read this attribute, and the new one must not.
                try FileManager.default.setAttributes(
                    [.modificationDate: date], ofItemAtPath: url.path)
            }
            #expect(
                ConceptBuilder.pairedRecipeFamilyOnDisk(for: "valor").family
                    == .caaMeanDifference)
            try writeMirror("repe", date: Date(timeIntervalSince1970: 2_000))
            #expect(
                ConceptBuilder.pairedRecipeFamilyOnDisk(for: "valor").family
                    == .pairedDifferencePCA)

            // Both mirrors, nothing recorded: unknowable, and said so.
            try writeMirror("readers", date: Date(timeIntervalSince1970: 3_000))
            let ambiguous = ConceptBuilder.pairedRecipeFamilyOnDisk(for: "valor")
            #expect(ambiguous.family == nil)
            #expect(ambiguous.source.contains("cannot be read off disk"))

            // A recorded vector build decides it — and keeps deciding it when
            // the OTHER mirror is touched last, which the old rule would have
            // flipped.
            let runs = VectorCatalog.runsDirectory(root: root)
            let runDirectory = runs.appending(component: "20260101T000000000-extract")
            try FileManager.default.createDirectory(
                at: runDirectory, withIntermediateDirectories: true)
            var sidecar = SteeringVectorSidecar(
                modelID: "org/m", revision: nil, concept: "valor",
                stimulusSetHash: "sh",
                vectors: ConceptVectors(perLayer: [[1, 0]]))
            sidecar.recipeMethod = "repeLAT"
            try SteeringVectorStore.save(
                vectors: ConceptVectors(perLayer: [[1, 0]]), sidecar: sidecar,
                to: runDirectory, name: "valor")
            try writeMirror("readers", date: Date(timeIntervalSince1970: 9_000))
            let recorded = ConceptBuilder.pairedRecipeFamilyOnDisk(for: "valor")
            #expect(recorded.family == .pairedDifferencePCA)
            #expect(recorded.source.contains("recipeMethod"))
        }
    }

    // MARK: - Activity live-log behavior (guide, notices, recovery)

    private func freshService(_ name: String) throws -> ChatService {
        let suite = "steerlab.tests.concept-prompts.\(name)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        return ChatService(cluster: clusterStore(defaults: defaults))
    }

    private func liveLogCount(_ service: ChatService, containing needle: String) -> Int {
        service.transcript.filter { $0.text.contains(needle) }.count
    }

    /// F6 + reviewer finding 4: any chat send wipes live logs; the guide
    /// must exist again whenever the panel asks for it — recreate-when-
    /// absent (same family included), suppress only live-entry churn. A
    /// same-family panel appearance while the entry is LIVE updates in place
    /// (never a second bubble); after a clear, returning to Concept Lab with
    /// the SAME recipe re-creates the guide instead of leaving the requested
    /// explanation absent.
    @Test func recipeGuideSurvivesTranscriptClears() throws {
        try withTempWorkspace { _ in
            let service = try freshService("guide")
            let builder = service.concepts
            builder.recipeFamily = .pairedDifferencePCA  // didSet presents the guide
            #expect(liveLogCount(service, containing: "**Technique — ") == 1)

            // Panel appearance re-presents: updates in place, no new entry.
            builder.presentRecipeGuide()
            #expect(liveLogCount(service, containing: "**Technique — ") == 1)

            // A chat send clears live logs; returning to the panel with the
            // SAME family re-creates the guide (recreate-when-absent)…
            service.clearLiveLogs()
            builder.presentRecipeGuide()
            #expect(liveLogCount(service, containing: "**Technique — ") == 1)
            // …as ONE entry that subsequent appearances update, not append.
            builder.presentRecipeGuide()
            #expect(liveLogCount(service, containing: "**Technique — ") == 1)

            // Switching the recipe re-presents it too (acceptance #6).
            builder.recipeFamily = .emotionGrandMean
            #expect(liveLogCount(service, containing: "**Technique — Grand mean") == 1)
            #expect(liveLogCount(service, containing: "**Technique — ") == 1)

            // And it keeps working across further clears, not just once.
            service.clearLiveLogs()
            builder.recipeFamily = .pairedDifferencePCA
            #expect(liveLogCount(service, containing: "**Technique — ") == 1)
        }
    }

    /// F7: a missing template (packaged/relocated app + template-less
    /// workspace) must land in the failure path — nil prompt, action-needed
    /// notice, recovery slot untouched — never a sentinel string that the
    /// copy path would put on the clipboard under a success message.
    @Test func missingTemplateRoutesToTheFailurePath() throws {
        try withTempWorkspace { root in
            let emptySeed = root.appending(component: "empty-seed-root")
            try FileManager.default.createDirectory(
                at: emptySeed, withIntermediateDirectories: true)
            ConceptBuilder.seedRootOverrideForTesting = emptySeed
            defer { ConceptBuilder.seedRootOverrideForTesting = nil }

            let service = try freshService("sentinel")
            let builder = service.concepts
            builder.conceptName = "integrity"
            builder.recipeFamily = .pairedDifferencePCA

            #expect(builder.generationPrompt() == nil)
            #expect(builder.lastCopiedPrompt == nil)
            #expect(builder.status?.contains("unavailable") == true)
            #expect(liveLogCount(service, containing: "action needed") == 1)

            // Template-backed siblings take the same path; repeated failures
            // coalesce into the one notice entry instead of appending.
            #expect(builder.neutralCorpusPrompt() == nil)
            #expect(builder.probeGenerationPrompt() == nil)
            #expect(liveLogCount(service, containing: "action needed") == 1)

            // The sentinel text never surfaces anywhere.
            #expect(
                service.transcript.allSatisfy {
                    !$0.text.contains("Template unavailable")
                })

            // The CAA prompt is code-built, not template-backed — it still
            // works with the seed data gone.
            builder.recipeFamily = .caaMeanDifference
            #expect(builder.generationPrompt() != nil)
        }
    }

    /// The "Last copied prompt" disclosure is the manual-recovery path: it
    /// must hold the generated prompt in BOTH pasteboard branches —
    /// precisely when the clipboard fails is when recovery matters.
    @Test func copyFailureStillRecordsThePromptForManualRecovery() throws {
        try withTempWorkspace { _ in
            let service = try freshService("recovery")
            let builder = service.concepts
            builder.conceptName = "integrity"

            builder.recordCopiedPrompt("PROMPT-A", message: "copied A")
            #expect(builder.lastCopiedPrompt == "PROMPT-A")

            builder.recordCopyFailure(
                "PROMPT-B", message: "could not write the prompt to the macOS clipboard")
            #expect(builder.lastCopiedPrompt == "PROMPT-B")
            #expect(builder.status?.contains("could not write") == true)
            #expect(liveLogCount(service, containing: "action needed") == 1)
        }
    }

    /// Copy-tweak-copy loops must update ONE Activity entry per notice
    /// category (like the recipe guide), not append unboundedly — and the
    /// stable id must recover after a transcript clear.
    @Test func copyConfirmationsCoalesceIntoOneActivityEntry() throws {
        try withTempWorkspace { _ in
            let service = try freshService("coalesce")
            let builder = service.concepts
            builder.conceptName = "integrity"

            builder.recordCopiedPrompt("P1", message: "copied 1")
            builder.recordCopiedPrompt("P2", message: "copied 2")
            builder.recordCopiedPrompt("P3", message: "copied 3")
            #expect(
                liveLogCount(service, containing: "**Dataset prompt — CAA mean difference**")
                    == 1)
            #expect(liveLogCount(service, containing: "copied 3") == 1)
            #expect(liveLogCount(service, containing: "copied 1") == 0)

            // After a chat send clears the transcript, the stale id falls
            // back to a fresh entry instead of silently rendering nothing.
            service.clearLiveLogs()
            builder.recordCopiedPrompt("P4", message: "copied 4")
            #expect(liveLogCount(service, containing: "copied 4") == 1)
        }
    }

    // MARK: - Empty-examples prompt coherence (fresh-concept flow)

    @Test func freshConceptPromptOmitsTheEmptyExamplesSection() {
        let fresh = ClaudeStimulusGenerator.buildPrompt(
            concept: "integrity", guidance: "",
            examplePositives: [], exampleNegatives: [], count: 8)
        #expect(!fresh.contains("Existing pairs in the set"))
        #expect(!fresh.contains("examples below"))
        #expect(fresh.contains("Generate exactly 8 new pairs."))

        let seasoned = ClaudeStimulusGenerator.buildPrompt(
            concept: "integrity", guidance: "",
            examplePositives: ["a calm hand steadies the ladder"],
            exampleNegatives: ["a hand rests on the ladder"], count: 8)
        #expect(
            seasoned.contains(
                "Existing pairs in the set (for style; do NOT duplicate their scenarios):"))
        #expect(seasoned.contains("do not reuse the scenarios of the examples below"))
        #expect(seasoned.contains("1. positive: a calm hand steadies the ladder"))
    }

    // MARK: - Shared seed-resolution rule

    @Test func workspaceOrSeedURLPrefersTheWorkspaceCopy() throws {
        try withTempWorkspace { root in
            let seedRoot = root.appending(component: "seed-root")
            let rel = "prompts/generation/example.md"
            let workspaceCopy = root.appending(path: rel)

            // No workspace copy: resolves into the seed root.
            let seedResolved = DataTemplates.workspaceOrSeedURL(
                relativePath: rel, workspaceRoot: root, seedRoot: seedRoot)
            #expect(seedResolved.path == seedRoot.appending(path: rel).path)

            // A workspace copy wins once it exists.
            try FileManager.default.createDirectory(
                at: workspaceCopy.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try Data("hello".utf8).write(to: workspaceCopy)
            let workspaceResolved = DataTemplates.workspaceOrSeedURL(
                relativePath: rel, workspaceRoot: root, seedRoot: seedRoot)
            #expect(workspaceResolved.path == workspaceCopy.path)

            // Template scaffolding routes through the same rule (default
            // seed root = `CodeResources.workspaceSeed()`, which since WP1
            // is the curated `WorkspaceSeed/` tree rather than the checkout).
            let template = DataTemplates.validation
            let templateURL = DataTemplates.seedURL(for: template, workspaceRoot: root)
            #expect(
                templateURL.path
                    == (try CodeResources.workspaceSeed())
                        .appending(path: template.seedRelativePath).path)
        }
    }
}
