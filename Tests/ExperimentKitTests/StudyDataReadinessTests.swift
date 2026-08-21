import Foundation
import Testing
@testable import ExperimentKit
@testable import SteeringKit

/// Study-data readiness: the manifest-derived checklist of what data a study
/// still needs, where each file goes, and the template scaffolding. Pure
/// functions over (manifest, explicit workspace root) — no process globals,
/// so this suite runs unserialized. Deliberately domain-neutral fixtures:
/// requirements derive ONLY from manifest declarations, never from any study
/// domain.
struct StudyDataReadinessTests {

    // MARK: - Fixture helpers

    private func withTempRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "readiness-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        return try body(root)
    }

    private func write(_ text: String, to root: URL, path: String) throws {
        let url = root.appending(path: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: url, atomically: true, encoding: .utf8)
    }

    private func manifest(name: String = "readiness-study") -> ExperimentManifest {
        ExperimentManifest(name: name, description: "", modelID: "test/model")
    }

    private func attachGrandMean(
        _ concept: String, to manifest: inout ExperimentManifest
    ) {
        manifest.concepts.append(
            ExperimentManifest.ConceptRef(
                name: concept, stimulusSetHash: "unused",
                options: ExtractionOptions(method: .emotionGrandMean)))
    }

    private func attachPaired(
        _ concept: String, to manifest: inout ExperimentManifest
    ) {
        manifest.concepts.append(
            ExperimentManifest.ConceptRef(
                name: concept, stimulusSetHash: "unused",
                options: ExtractionOptions(method: .meanDifference)))
    }

    private func requirement(
        _ id: String, in rows: [DataRequirement]
    ) -> DataRequirement? {
        rows.first { $0.id == id }
    }

    // MARK: - Concept-derived rows

    @Test func grandMeanConceptWithoutValidationIsABlocker() throws {
        try withTempRoot { root in
            var m = manifest()
            attachGrandMean("urgency", to: &m)
            try write(
                #"{"concept": "urgency", "text": "a story"}"# + "\n",
                to: root, path: "prompts/emotions/urgency/stories.jsonl")
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)

            let stimuli = try #require(requirement("concept:urgency:stimuli", in: rows))
            #expect(stimuli.status == .present)
            #expect(stimuli.path == "prompts/emotions/urgency/stories.jsonl")

            let validation = try #require(
                requirement("concept:urgency:validation", in: rows))
            #expect(validation.status == .missing)
            #expect(validation.path == "prompts/emotions/urgency/validation.jsonl")
            #expect(validation.detail.contains("before validate can gate freeze"))
            #expect(validation.templateID == "validation")

            let markers = try #require(requirement("concept:urgency:markers", in: rows))
            #expect(markers.status == .optional)
            #expect(markers.path == "prompts/concepts/urgency/markers.json")
            #expect(markers.detail.contains("marker-density"))
        }
    }

    @Test func pairedConceptRowsResolveUnderPromptsConcepts() throws {
        try withTempRoot { root in
            var m = manifest()
            attachPaired("calm", to: &m)
            try write(
                #"{"text": "a positive stimulus"}"# + "\n",
                to: root, path: "prompts/concepts/calm/positive.jsonl")
            try write(
                #"{"text": "a negative stimulus"}"# + "\n",
                to: root, path: "prompts/concepts/calm/negative.jsonl")
            try write(
                #"{"text": "an unhurried afternoon", "expresses": true}"# + "\n",
                to: root, path: "prompts/concepts/calm/validation.jsonl")
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)

            #expect(requirement("concept:calm:stimuli", in: rows)?.status == .present)
            let validation = try #require(requirement("concept:calm:validation", in: rows))
            #expect(validation.status == .present)
            #expect(validation.path == "prompts/concepts/calm/validation.jsonl")
        }
    }

    @Test func degenerateStimulusClassIsABlocker() throws {
        // The live failure this guards: a derivation script iterated a
        // reference corpus's NAME as a string, writing one-character rows
        // ('n','e','u','t','r',...). Presence checks passed; last-token
        // extraction would have produced garbage class means silently.
        try withTempRoot { root in
            var m = manifest()
            attachPaired("courage-vs-neutral", to: &m)
            try write(
                #"{"text": "A long true story about someone holding steady under real danger, told plainly."}"#
                    + "\n",
                to: root, path: "prompts/concepts/courage-vs-neutral/positive.jsonl")
            try write(
                ["n", "e", "u", "t", "r", "a", "l"]
                    .map { #"{"text": "\#($0)"}"# }
                    .joined(separator: "\n") + "\n",
                to: root, path: "prompts/concepts/courage-vs-neutral/negative.jsonl")
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let stimuli = try #require(
                requirement("concept:courage-vs-neutral:stimuli", in: rows))
            #expect(stimuli.status == .invalid)
            #expect(stimuli.detail.contains("degenerate"))
            #expect(stimuli.detail.contains("derivation bug"))
        }
    }

    @Test func pooledReadingTooShortRowsAreABlocker() throws {
        // Healthy prose, but shorter than the pinned pooling start: the
        // extractor refuses such rows at run time, so the checklist says so
        // before any model loads.
        try withTempRoot { root in
            var m = manifest()
            var options = ExtractionOptions(method: .meanDifference)
            options.readingPosition = .meanFromToken(50)
            m.concepts.append(
                ExperimentManifest.ConceptRef(
                    name: "calm", stimulusSetHash: "unused", options: options))
            try write(
                #"{"text": "An unhurried afternoon by the window."}"# + "\n",
                to: root, path: "prompts/concepts/calm/positive.jsonl")
            try write(
                #"{"text": "A frantic morning of missed alarms and spilled coffee everywhere."}"#
                    + "\n",
                to: root, path: "prompts/concepts/calm/negative.jsonl")
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let stimuli = try #require(requirement("concept:calm:stimuli", in: rows))
            #expect(stimuli.status == .invalid)
            #expect(stimuli.detail.contains("mean from token 50"))
        }
    }

    @Test func shortButHealthyPairedStimuliStayPresent() throws {
        // Minimal pairs are a legitimate design — no absolute per-row floor.
        try withTempRoot { root in
            var m = manifest()
            attachPaired("fear", to: &m)
            try write(
                #"{"text": "I am terrified."}"# + "\n"
                    + #"{"text": "My hands are shaking."}"# + "\n",
                to: root, path: "prompts/concepts/fear/positive.jsonl")
            try write(
                #"{"text": "I am calm."}"# + "\n"
                    + #"{"text": "The evening is quiet."}"# + "\n",
                to: root, path: "prompts/concepts/fear/negative.jsonl")
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            #expect(requirement("concept:fear:stimuli", in: rows)?.status == .present)
        }
    }

    @Test func missingStimuliAreABlocker() throws {
        try withTempRoot { root in
            var m = manifest()
            attachPaired("calm", to: &m)
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            #expect(requirement("concept:calm:stimuli", in: rows)?.status == .missing)
        }
    }

    // MARK: - Task prompts + instrument options

    @Test func optionsLackingTaskPromptsArePartialWithCounts() throws {
        try withTempRoot { root in
            var m = manifest()
            m.taskPromptsFile = "prompts/tasks/items.jsonl"
            m.outcomeInstruments = ["answerTokenLogprob"]
            let lines = [
                #"{"text": "pick one", "options": ["left", "right"], "target": "left"}"#,
                #"{"text": "no options here"}"#,
                #"{"text": "also no options"}"#,
                #"{"text": "choose", "options": ["up", "down"]}"#,
                #"{"text": "third without"}"#,
            ]
            try write(
                lines.joined(separator: "\n") + "\n",
                to: root, path: "prompts/tasks/items.jsonl")
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let prompts = try #require(requirement("taskPrompts", in: rows))
            #expect(prompts.status == .partial)
            #expect(
                prompts.detail.contains(
                    "3 of 5 task prompts lack the `options` field"))
        }
    }

    @Test func fullyOptionedTaskPromptsArePresent() throws {
        try withTempRoot { root in
            var m = manifest()
            m.taskPromptsFile = "prompts/tasks/items.jsonl"
            m.outcomeInstruments = ["choiceProbability"]
            try write(
                #"{"text": "pick", "options": ["a", "b"], "target": "a"}"# + "\n",
                to: root, path: "prompts/tasks/items.jsonl")
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            #expect(requirement("taskPrompts", in: rows)?.status == .present)
        }
    }

    @Test func zeroOptionTaskPromptsAreABlockerWithTheRunRefusal() throws {
        // 2026-08-06 field incident: an option-consuming instrument declared
        // while NO item carries options now refuses at run start — the
        // readiness row states the run loop's own refusal, not a "degraded
        // run" partial.
        try withTempRoot { root in
            var m = manifest()
            m.taskPromptsFile = "prompts/tasks/items.jsonl"
            m.outcomeInstruments = ["answerTokenLogprob"]
            let lines = [
                #"{"text": "no options"}"#,
                #"{"text": "none here either"}"#,
            ]
            try write(
                lines.joined(separator: "\n") + "\n",
                to: root, path: "prompts/tasks/items.jsonl")
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let prompts = try #require(requirement("taskPrompts", in: rows))
            #expect(prompts.status == .missing)
            #expect(
                prompts.detail.contains(
                    "none of the 2 task items carries options"))
            #expect(prompts.detail.contains("silently produce zero records"))
        }
    }

    @Test func ladderWindowAdvisoryRowMirrorsTheRunWarning() throws {
        // The 1–7 ladder / min 0 max 100 field case: legal, inert, surfaced
        // as an advisory row (never a blocker — the endpoint may lawfully
        // take non-ladder values).
        try withTempRoot { root in
            var m = manifest()
            m.taskPromptsFile = "prompts/tasks/items.jsonl"
            m.exclusionRules = [
                ExclusionRule(rule: "outOfRange", min: 0, max: 100)
            ]
            try write(
                #"{"text": "rate", "options": ["1","2","3","4","5","6","7"]}"#
                    + "\n",
                to: root, path: "prompts/tasks/items.jsonl")
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let row = try #require(requirement("exclusionLadderWindow", in: rows))
            #expect(row.status == .partial)
            #expect(row.detail.contains("min 0 and max 100"))
            #expect(row.detail.contains("spans 1 to 7"))
        }
    }

    @Test func undeclaredTaskPromptsAreABlockerForModelOutputStudies() throws {
        try withTempRoot { root in
            let m = manifest(name: "fresh")
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let prompts = try #require(requirement("taskPrompts", in: rows))
            #expect(prompts.status == .missing)
            #expect(prompts.path == "prompts/tasks/fresh-prompts.jsonl")

            // Multi-agent studies have no task-prompts row at all.
            var multi = manifest()
            multi.studyKind = .multiAgent
            let multiRows = StudyDataReadiness.requirements(
                for: multi, workspaceRoot: root)
            #expect(requirement("taskPrompts", in: multiRows) == nil)
        }
    }

    // MARK: - Load-refusing task prompts are blockers (invalid, not partial)

    @Test func malformedTaskPromptsAreInvalidBlockers() throws {
        // `.partial` promises a degraded run; a file the run REFUSES to load
        // must block preflight the way it blocks execution.
        try withTempRoot { root in
            var m = manifest()
            m.taskPromptsFile = "prompts/tasks/items.jsonl"
            try write(
                "this is not json\n", to: root, path: "prompts/tasks/items.jsonl")
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let prompts = try #require(requirement("taskPrompts", in: rows))
            #expect(prompts.status == .invalid)
            #expect(prompts.detail.contains("malformed task prompt JSONL at line 1"))
            #expect(prompts.detail.contains("refuses to load"))

            let summary = StudyDataReadiness.summary(rows)
            #expect(summary.invalidCount == 1)
            #expect(summary.blockers.contains { $0.id == "taskPrompts" })
            #expect(!summary.isReady)  // → data check exit 2
            #expect(summary.line.contains("1 invalid"))
        }
    }

    @Test func transcriptSchemaViolationsAreInvalidBlockers() throws {
        try withTempRoot { root in
            var m = manifest()
            m.taskPromptsFile = "prompts/tasks/items.jsonl"
            // Ends with an assistant turn — the run loop's parser refuses.
            try write(
                #"{"id": "t1", "transcript": [{"role": "assistant", "content": "hi"}]}"#
                    + "\n",
                to: root, path: "prompts/tasks/items.jsonl")
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let prompts = try #require(requirement("taskPrompts", in: rows))
            #expect(prompts.status == .invalid)
            #expect(prompts.detail.contains("t1"))
            #expect(StudyDataReadiness.summary(rows).blockers.contains {
                $0.id == "taskPrompts"
            })
        }
    }

    @Test func transcriptFamilyViolationsAreInvalidBlockers() throws {
        try withTempRoot { root in
            var m = manifest(name: "gemma-study")
            m.modelID = "mlx-community/gemma-3-4b-it-4bit"
            m.taskPromptsFile = "prompts/tasks/items.jsonl"
            // Schema-valid, but consecutive user turns violate Gemma's
            // strict-alternation template — a run-START refusal.
            try write(
                #"{"id": "t1", "transcript": ["#
                    + #"{"role": "user", "content": "one"}, "#
                    + #"{"role": "user", "content": "two"}]}"# + "\n",
                to: root, path: "prompts/tasks/items.jsonl")
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let prompts = try #require(requirement("taskPrompts", in: rows))
            #expect(prompts.status == .invalid)
            #expect(prompts.detail.contains("alternation"))
        }
    }

    // MARK: - Judge protocol

    @Test func judgeProtocolWithOneJudgeAndNoRubricIsPartialAndMissing() throws {
        try withTempRoot { root in
            var m = manifest()
            m.evaluation = ExperimentManifest.EvaluationSpec(kind: .pairedJudge)
            m.judges = [ExperimentManifest.JudgeRef(name: "j1", kind: "claude")]
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)

            let rubric = try #require(requirement("judgeRubric", in: rows))
            #expect(rubric.status == .missing)

            let panel = try #require(requirement("judgePanel", in: rows))
            #expect(panel.status == .partial)
            #expect(panel.detail.contains("1 of the ≥2 judges"))

            // Two judges + an existing pinned rubric file → both present.
            m.judges?.append(ExperimentManifest.JudgeRef(name: "j2", kind: "local"))
            m.judgeRubricFile = "prompts/rubrics/r1.md"
            try write("Compare the two responses.", to: root, path: "prompts/rubrics/r1.md")
            let rows2 = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            #expect(requirement("judgeRubric", in: rows2)?.status == .present)
            #expect(requirement("judgePanel", in: rows2)?.status == .present)
        }
    }

    @Test func judgePanelCollapsingToOneResolvedJudgeIsInvalid() throws {
        // Finding 4 (2026-07-22): two blank-model local judges resolve to
        // the SAME deterministic judge — count says panel, identity says no.
        try withTempRoot { root in
            var m = manifest()
            m.evaluation = ExperimentManifest.EvaluationSpec(kind: .pairedJudge)
            m.judges = [
                ExperimentManifest.JudgeRef(name: "judge-1", kind: "local"),
                ExperimentManifest.JudgeRef(name: "judge-2", kind: "local"),
            ]
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let panel = try #require(requirement("judgePanel", in: rows))
            #expect(panel.status == .invalid)
            #expect(panel.detail.contains("agree perfectly by construction"))

            // Making one judge a different model restores the panel.
            m.judges?[1].model = "other/model"
            let rows2 = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            #expect(requirement("judgePanel", in: rows2)?.status == .present)
        }
    }

    @Test func judgePanelRoutesEvaluateFanoutAndBlocksJudgedSweep() throws {
        // Finding 1, fan-out era (2026-07-23; live incident 2026-07-22): a
        // pipeline's EVALUATE stage with a foreign-model local judge ROUTES
        // to the server's post-generation judge fan-out (informational); a
        // judged SWEEP with such judges is still invalid — no fan-out
        // exists for sweep-interleaved judging.
        try withTempRoot { root in
            var m = manifest()
            m.evaluation = ExperimentManifest.EvaluationSpec(kind: .pairedJudge)
            m.pipeline = .object(
                ["stages": .array([.string("run"), .string("evaluate")])])
            m.judges = [
                ExperimentManifest.JudgeRef(name: "judge-1", kind: "local"),
                ExperimentManifest.JudgeRef(
                    name: "judge-2", kind: "local", model: "other/judge-12b"),
            ]
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let panel = try #require(requirement("judgePanel", in: rows))
            #expect(panel.status == .present)
            #expect(panel.detail.contains("post-generation judge fan-out"))

            // A judgeScore sweep stage blocks with the freeze gate's words.
            m.pipeline = .object(
                ["stages": .array([.string("sweep"), .string("run")])])
            m.sweep = .init(
                selection: .init(objective: .init(metric: "judgeScore")))
            let rows2 = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let panel2 = try #require(requirement("judgePanel", in: rows2))
            #expect(panel2.status == .invalid)
            #expect(panel2.detail.contains("sweep stage holds ONE model"))

            // Without the pipeline the same panel is fine (evaluate outside
            // the chain may hold two models where the substrate allows it).
            m.pipeline = nil
            let rows3 = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            #expect(requirement("judgePanel", in: rows3)?.status == .present)
        }
    }

    @Test func noJudgeProtocolMeansNoJudgeRows() throws {
        try withTempRoot { root in
            let rows = StudyDataReadiness.requirements(
                for: manifest(), workspaceRoot: root)
            #expect(requirement("judgeRubric", in: rows) == nil)
            #expect(requirement("judgePanel", in: rows) == nil)
        }
    }

    // MARK: - Human baseline

    @Test func humanBaselineUnpinnedIsOneOptionalRow() throws {
        try withTempRoot { root in
            let rows = StudyDataReadiness.requirements(
                for: manifest(name: "study-x"), workspaceRoot: root)
            let baseline = try #require(requirement("humanBaseline", in: rows))
            #expect(baseline.status == .optional)
            #expect(baseline.path == "prompts/baselines/study-x-human-baseline.csv")
            #expect(baseline.detail.contains("human-anchored (R) claims"))
            #expect(baseline.templateID == "human-baseline")
        }
    }

    @Test func pinnedHumanBaselineIsCheckedOnDiskAndForShape() throws {
        try withTempRoot { root in
            var m = manifest()
            m.humanBaseline = ExperimentManifest.HumanBaseline(
                path: "prompts/baselines/effects.csv", hash: "aa")
            let missing = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            #expect(requirement("humanBaseline", in: missing)?.status == .missing)

            // Present but the WRONG SHAPE (the pre-2026-07-19 template
            // columns): invalid — analyze would refuse it — with the
            // plain detail naming the loader's columns.
            try write(
                "source,measure,population,delta,ci_low,ci_high,n,notes\n",
                to: root, path: "prompts/baselines/effects.csv")
            let invalid = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let invalidRow = try #require(requirement("humanBaseline", in: invalid))
            #expect(invalidRow.status == .invalid)
            #expect(invalidRow.detail.contains("endpoint, deltaHuman, ciLower, ciUpper"))

            try FileManager.default.removeItem(
                at: root.appending(path: "prompts/baselines/effects.csv"))
            try write(
                "endpoint,deltaHuman,ciLower,ciUpper\nrate,0.1,0.0,0.2\n",
                to: root, path: "prompts/baselines/effects.csv")
            let present = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            #expect(requirement("humanBaseline", in: present)?.status == .present)
        }
    }

    // MARK: - Multi-agent scenario

    @Test func declaredScenarioMissingOnDiskIsABlocker() throws {
        try withTempRoot { root in
            var m = manifest()
            m.studyKind = .multiAgent
            m.multiAgentScenarioPath = "runs/multi-agent-scenarios/s/scenario.json"
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let scenario = try #require(requirement("multiAgentScenario", in: rows))
            #expect(scenario.status == .missing)

            // A parsing scenario (the shipped template) flips it to present.
            let seed = DataTemplates.seedURL(
                for: DataTemplates.scenario, workspaceRoot: root)
            let destination = root.appending(
                path: "runs/multi-agent-scenarios/s/scenario.json")
            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: seed, to: destination)
            let rows2 = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            #expect(requirement("multiAgentScenario", in: rows2)?.status == .present)

            // A file that exists but does not parse is partial, not present.
            try Data(#"{"schemaVersion": 1}"#.utf8).write(to: destination)
            let rows3 = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            #expect(requirement("multiAgentScenario", in: rows3)?.status == .partial)
        }
    }

    // MARK: - Battery and neutral corpus

    @Test func unpinnedBatteryIsOptionalAndSweepBatteryIsChecked() throws {
        try withTempRoot { root in
            var m = manifest()
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let battery = try #require(requirement("capabilityBattery", in: rows))
            #expect(battery.status == .optional)
            #expect(battery.detail.contains("evidence-grade"))

            // A declared sweep makes its battery file a hard requirement.
            m.sweep = ExperimentManifest.SweepSpec()
            let sweepRows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            #expect(requirement("capabilityBattery", in: sweepRows)?.status == .missing)
            try write(
                #"{"prompt": "2+2? Answer with the number.", "answer": "4", "grading": "exact_number"}"#
                    + "\n",
                to: root, path: "prompts/batteries/basic.jsonl")
            let sweepRows2 = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            #expect(requirement("capabilityBattery", in: sweepRows2)?.status == .present)
        }
    }

    @Test func neutralCorpusIsABlockerOnceConceptsAttach() throws {
        try withTempRoot { root in
            var m = manifest()
            // No concepts: informational only.
            let bare = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            #expect(requirement("neutralCorpus", in: bare)?.status == .optional)

            attachPaired("calm", to: &m)
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            #expect(requirement("neutralCorpus", in: rows)?.status == .missing)

            try write(
                #"{"text": "a plain sentence"}"# + "\n",
                to: root, path: "prompts/neutral/corpus.jsonl")
            let rows2 = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            #expect(requirement("neutralCorpus", in: rows2)?.status == .present)
        }
    }

    // MARK: - Summary math

    @Test func summaryCountsAndBlockersAreConsistent() throws {
        try withTempRoot { root in
            var m = manifest()
            attachGrandMean("urgency", to: &m)  // stimuli+validation missing, markers optional
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let summary = StudyDataReadiness.summary(rows)
            #expect(
                summary.presentCount + summary.partialCount + summary.invalidCount
                    + summary.missingCount + summary.optionalCount == rows.count)
            #expect(
                summary.blockers
                    == rows.filter { $0.status == .missing || $0.status == .invalid })
            #expect(summary.invalidCount == 0)  // nothing load-refusing here
            #expect(summary.blockers.allSatisfy { $0.status == .missing })
            #expect(!summary.isReady)
            // stories + validation + task prompts + neutral corpus missing.
            #expect(summary.missingCount == 4)
        }
    }

    // MARK: - Scaffolding

    @Test func scaffoldCreatesValidationAtMethodResolvedDestination() throws {
        try withTempRoot { root in
            var m = manifest()
            attachGrandMean("urgency", to: &m)
            attachPaired("calm", to: &m)
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)

            let grandMean = try #require(
                requirement("concept:urgency:validation", in: rows))
            let created = try StudyDataReadiness.scaffold(
                requirement: grandMean, in: root)
            #expect(
                created.path
                    == root.appending(
                        path: "prompts/emotions/urgency/validation.jsonl").path)
            #expect(FileManager.default.fileExists(atPath: created.path))

            let paired = try #require(requirement("concept:calm:validation", in: rows))
            let createdPaired = try StudyDataReadiness.scaffold(
                requirement: paired, in: root)
            #expect(
                createdPaired.path
                    == root.appending(
                        path: "prompts/concepts/calm/validation.jsonl").path)

            // Refuses to overwrite what it just created.
            #expect(throws: ExperimentError.self) {
                try StudyDataReadiness.scaffold(requirement: grandMean, in: root)
            }
        }
    }

    @Test func scaffoldRefusesRequirementsWithoutTemplates() throws {
        try withTempRoot { root in
            var m = manifest()
            attachPaired("calm", to: &m)
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let stimuli = try #require(requirement("concept:calm:stimuli", in: rows))
            #expect(stimuli.templateID == nil)
            #expect(throws: ExperimentError.self) {
                try StudyDataReadiness.scaffold(requirement: stimuli, in: root)
            }
        }
    }

    // MARK: - Template validity (the shipped seed data)

    @Test func scenarioTemplateParsesWithTheRealLoader() throws {
        let seed = DataTemplates.seedURL(
            for: DataTemplates.scenario,
            workspaceRoot: URL(filePath: "/nonexistent"))
        let data = try Data(contentsOf: seed)
        // The exact decode the scenario store and the readiness check use.
        let scenario = try JSONDecoder().decode(MultiAgentScenario.self, from: data)
        #expect(scenario.schemaVersion == 1)
        #expect(scenario.agents.count == 2)
        #expect(scenario.turns.count == 2)
        let agentIDs = Set(scenario.agents.map(\.id))
        #expect(scenario.turns.allSatisfy { agentIDs.contains($0.speakerAgentID) })
    }

    @Test func validationTemplateRowsDecode() throws {
        let seed = DataTemplates.seedURL(
            for: DataTemplates.validation,
            workspaceRoot: URL(filePath: "/nonexistent"))
        let data = try Data(contentsOf: seed)
        // Text side loads through the field-preserving document…
        let document = try TaskPromptsDocument.load(data)
        #expect(document.count == 4)
        // …and every row carries the boolean `expresses` label with both
        // classes represented.
        var labels: [Bool] = []
        for line in String(decoding: data, as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: true)
        {
            let object = try #require(
                try JSONSerialization.jsonObject(with: Data(line.utf8))
                    as? [String: Any])
            labels.append(try #require(object["expresses"] as? Bool))
        }
        #expect(labels.contains(true) && labels.contains(false))
    }

    @Test func choiceTemplateRowsCarryOptionsAndParse() throws {
        let seed = DataTemplates.seedURL(
            for: DataTemplates.taskPromptsChoice,
            workspaceRoot: URL(filePath: "/nonexistent"))
        let data = try Data(contentsOf: seed)
        let document = try TaskPromptsDocument.load(data)
        #expect(document.count == 3)
        #expect(document.optionsItemCount == 3)
        // The run loop's own parser accepts it too.
        let prompts = try ExperimentTasks.parseTaskPrompts(data)
        #expect(prompts.allSatisfy { ($0.options?.count ?? 0) >= 2 })
    }

    /// Finding B1 (readiness half): a variant study with NO pinned battery
    /// relies on the shared DEFAULT battery, which freeze/validate pin
    /// nil-tolerantly — a malformed default is skipped like an absent one —
    /// so readiness must surface WHY as an `.invalid` blocker carrying the
    /// plain shape detail. An absent or well-formed default keeps the
    /// historical row.
    @Test func malformedDefaultBatteryIsInvalidForVariantStudies() throws {
        try withTempRoot { root in
            var m = manifest()
            m.variantConditions = [
                ExperimentManifest.VariantCondition(
                    name: "agent-a", artifactPath: "p", artifactHash: "h",
                    artifact: .init(
                        name: "agent-a", baseModelID: "test/model",
                        promptMode: "chatAssistant", qwenThinkingEnabled: false,
                        temperature: 0, systemPrompt: ""))
            ]
            let file = VariantRobustness.defaultPreset.batteryFile
            try write("not a battery item\n", to: root, path: file)
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let battery = try #require(requirement("capabilityBattery", in: rows))
            #expect(battery.status == .invalid)
            #expect(battery.path == file)
            #expect(battery.detail.contains("line 1"))
            #expect(
                StudyDataReadiness.summary(rows).blockers.contains {
                    $0.kind == .capabilityBattery
                })
            // A well-formed default battery is not a blocker…
            try write(
                #"{"prompt": "2+2?", "answer": "4"}"# + "\n",
                to: root, path: file)
            let healthy = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            #expect(
                requirement("capabilityBattery", in: healthy)?.status == .optional)
            // …and an ABSENT default keeps the historical optional row.
            try FileManager.default.removeItem(at: root.appending(path: file))
            let absent = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            #expect(
                requirement("capabilityBattery", in: absent)?.status == .optional)
        }
    }

    @Test func markersTemplateHasWordsList() throws {
        let seed = DataTemplates.seedURL(
            for: DataTemplates.markers,
            workspaceRoot: URL(filePath: "/nonexistent"))
        let object = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: seed))
                as? [String: Any])
        #expect(((object["words"] as? [Any])?.isEmpty) == false)
    }

    /// The template's header IS the loader contract (residuals.py
    /// `HUMAN_BASELINE_FIELDS`) plus provenance extras — locked exactly, so
    /// a template edit that breaks the analyze loader fails here first.
    @Test func humanBaselineTemplateHasTheLoaderColumns() throws {
        let seed = DataTemplates.seedURL(
            for: DataTemplates.humanBaseline,
            workspaceRoot: URL(filePath: "/nonexistent"))
        let text = try String(contentsOf: seed, encoding: .utf8)
        let header = try #require(text.split(separator: "\n").first)
        #expect(header == "endpoint,deltaHuman,ciLower,ciUpper,source,n,notes")
        // …and therefore the template itself pins cleanly.
        #expect(
            PinShapeValidation.humanBaselineShapeProblem(
                Data(text.utf8), file: "template") == nil)
    }

    /// Hard requirement: templates (and their READMEs) are domain-neutral —
    /// the readiness layer must serve any agentic-behavior study, so no seed
    /// file may smuggle in a study domain.
    @Test func templatesAreDomainNeutral() throws {
        let forbidden = [
            "legal", "judicial", "courtroom", "defendant", "plaintiff",
            "verdict", "statute", "sentencing",
        ]
        for template in DataTemplates.all {
            let seed = DataTemplates.seedURL(
                for: template, workspaceRoot: URL(filePath: "/nonexistent"))
            let directory = seed.deletingLastPathComponent()
            let files = try FileManager.default.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: nil)
            #expect(
                files.contains { $0.lastPathComponent == "README.md" },
                "template '\(template.id)' has no sibling README.md")
            for file in files {
                let text = try String(contentsOf: file, encoding: .utf8).lowercased()
                for word in forbidden {
                    #expect(
                        !text.contains(word),
                        "\(file.lastPathComponent) mentions '\(word)'")
                }
            }
        }
    }

    // MARK: - Numeric parser (registry) rows

    private let registryJSON = """
        {"schemaVersion": 1, "parsers": {"months": {"kind": "durationMonths", \
        "units": {"years": 12, "months": 1}}}}
        """

    @Test func noNamedParserMeansNoParserRow() throws {
        try withTempRoot { root in
            let rows = StudyDataReadiness.requirements(
                for: manifest(), workspaceRoot: root)
            #expect(requirement("numericParser", in: rows) == nil)
        }
    }

    @Test func namedParserWithoutRegistryIsABlocker() throws {
        try withTempRoot { root in
            var m = manifest()
            m.numericParser = "months"
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let row = try #require(requirement("numericParser", in: rows))
            #expect(row.status == .missing)
            #expect(row.kind == .numericParser)
            #expect(row.path == "prompts/parsers/parser-registry.json")
            #expect(row.detail.contains("no parser registry exists"))
            #expect(row.detail.contains("remove the study's numericParser"))
            #expect(StudyDataReadiness.summary(rows).blockers.contains {
                $0.id == "numericParser"
            })
        }
    }

    @Test func namedParserAbsentFromRegistryIsABlocker() throws {
        try withTempRoot { root in
            try write(registryJSON, to: root, path: ParserRegistry.registryFile)
            var m = manifest()
            m.numericParser = "sentences"
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let row = try #require(requirement("numericParser", in: rows))
            #expect(row.status == .invalid)
            #expect(row.detail.contains("defines no parser named 'sentences'"))
            #expect(row.detail.contains("the run refuses to start"))
        }
    }

    @Test func driftedRegistryPinIsABlockerPhrasedAsDrift() throws {
        try withTempRoot { root in
            try write(registryJSON, to: root, path: ParserRegistry.registryFile)
            var m = manifest()
            m.numericParser = "months"
            m.parserRegistryHash = String(repeating: "0", count: 64)
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let row = try #require(requirement("numericParser", in: rows))
            #expect(row.status == .invalid)
            #expect(row.detail.contains("drifted from the pinned hash"))
        }
    }

    @Test func definedParserWithMatchingPinIsReady() throws {
        try withTempRoot { root in
            try write(registryJSON, to: root, path: ParserRegistry.registryFile)
            var m = manifest()
            m.numericParser = "months"
            m.parserRegistryHash = ParserRegistry.liveHash(
                at: root.appending(path: ParserRegistry.registryFile))
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let row = try #require(requirement("numericParser", in: rows))
            #expect(row.status == .present)
            #expect(row.detail.contains("parser 'months' (durationMonths)"))
            #expect(row.detail.contains("drift is checked by hash"))

            // Unpinned (draft) but defined: still ready, with the freeze
            // note instead of the drift note.
            m.parserRegistryHash = nil
            let unpinned = try #require(
                requirement(
                    "numericParser",
                    in: StudyDataReadiness.requirements(for: m, workspaceRoot: root)))
            #expect(unpinned.status == .present)
            #expect(unpinned.detail.contains("freeze pins the registry"))
        }
    }

    // MARK: - Exclusion-rule / attention-check rows

    @Test func noRulesAndNoChecksMeansNoExclusionRows() throws {
        try withTempRoot { root in
            var m = manifest()
            m.taskPromptsFile = "prompts/tasks/t.jsonl"
            try write(
                #"{"id": "p1", "prompt": "a"}"# + "\n",
                to: root, path: "prompts/tasks/t.jsonl")
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            #expect(!rows.contains { $0.kind == .exclusionRules })
        }
    }

    @Test func declaredAttentionRuleWithZeroCheckedItemsIsABlocker() throws {
        try withTempRoot { root in
            var m = manifest()
            m.taskPromptsFile = "prompts/tasks/t.jsonl"
            m.exclusionRules = [ExclusionRule(rule: "failedAttentionCheck")]
            try write(
                #"{"id": "p1", "prompt": "a"}"# + "\n"
                    + #"{"id": "p2", "prompt": "b"}"# + "\n",
                to: root, path: "prompts/tasks/t.jsonl")
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let row = try #require(requirement("attentionChecks", in: rows))
            #expect(row.status == .missing)
            #expect(row.kind == .exclusionRules)
            #expect(
                row.detail.contains(
                    "none of the 2 task-prompt items carries an attentionCheck"))
            #expect(row.detail.contains("or remove the rule"))
            #expect(row.detail.contains("refuses to start"))
            #expect(StudyDataReadiness.summary(rows).blockers.contains {
                $0.id == "attentionChecks"
            })
        }
    }

    @Test func declaredAttentionRuleWithCheckedItemsIsReady() throws {
        try withTempRoot { root in
            var m = manifest()
            m.taskPromptsFile = "prompts/tasks/t.jsonl"
            m.exclusionRules = [ExclusionRule(rule: "failedAttentionCheck")]
            try write(
                #"{"id": "p1", "prompt": "a", "attentionCheck": {"expected": "7"}}"#
                    + "\n" + #"{"id": "p2", "prompt": "b"}"# + "\n",
                to: root, path: "prompts/tasks/t.jsonl")
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let row = try #require(requirement("attentionChecks", in: rows))
            #expect(row.status == .present)
            #expect(row.detail.contains("1 of 2 task-prompt items"))
        }
    }

    @Test func checksWithoutTheRuleAreAnUnusedAdvisory() throws {
        try withTempRoot { root in
            var m = manifest()
            m.taskPromptsFile = "prompts/tasks/t.jsonl"
            try write(
                #"{"id": "p1", "prompt": "a", "attentionCheck": {"expected": "7"}}"#
                    + "\n",
                to: root, path: "prompts/tasks/t.jsonl")
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let row = try #require(requirement("attentionChecks", in: rows))
            #expect(row.status == .partial)  // advisory, never a blocker
            #expect(row.detail.contains("checks exist but are unused"))
            #expect(row.detail.contains("failedAttentionCheck"))
            #expect(!StudyDataReadiness.summary(rows).blockers.contains {
                $0.id == "attentionChecks"
            })
        }
    }

    @Test func malformedRulesAreABlockerRow() throws {
        try withTempRoot { root in
            var m = manifest()
            m.exclusionRules = [ExclusionRule(rule: "outliers")]
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let row = try #require(requirement("exclusionRules", in: rows))
            #expect(row.status == .invalid)
            #expect(row.detail.contains("'outliers' is not recognized"))
            #expect(row.detail.contains("the run refuses to start"))
        }
    }

    @Test func unproducibleDefaultEndpointIsAnAdvisoryNotABlocker() throws {
        try withTempRoot { root in
            var m = manifest()
            m.exclusionRules = [
                ExclusionRule(rule: "outOfRange", min: 0, max: 600)
            ]
            // No numericParser, case family not sentencing: parsedMonths can
            // never appear on a record — the rule is inert as declared.
            let rows = StudyDataReadiness.requirements(for: m, workspaceRoot: root)
            let row = try #require(
                requirement("exclusionEndpoint:outOfRange", in: rows))
            #expect(row.status == .partial)
            #expect(row.detail.contains("no record of this study can carry it"))
            #expect(!StudyDataReadiness.summary(rows).blockers.contains {
                $0.id.hasPrefix("exclusionEndpoint")
            })

            // The sentencing case family produces parsedMonths — no advisory.
            m.caseFamily = "sentencing"
            #expect(
                requirement(
                    "exclusionEndpoint:outOfRange",
                    in: StudyDataReadiness.requirements(for: m, workspaceRoot: root))
                    == nil)

            // A declared numeric parser produces it too — no advisory.
            m.caseFamily = nil
            m.numericParser = "months"
            #expect(
                requirement(
                    "exclusionEndpoint:outOfRange",
                    in: StudyDataReadiness.requirements(for: m, workspaceRoot: root))
                    == nil)

            // A custom endpoint name is deliberately NOT checked (whether a
            // record carries it needs run data, not manifest inference).
            m.numericParser = nil
            m.exclusionRules = [
                ExclusionRule(rule: "outOfRange", endpoint: "wordCount", max: 900)
            ]
            #expect(
                !StudyDataReadiness.requirements(for: m, workspaceRoot: root)
                    .contains { $0.id.hasPrefix("exclusionEndpoint") })
        }
    }
}
