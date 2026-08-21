import Foundation
import Testing

@testable import ExperimentKit

/// Chain-runner state models + the pipeline-block verify mirror (stage 5).
/// Pure CPU: decode contracts and display helpers on the client side, and
/// the Swift twin of the server resolver's refusals.
struct PipelineStateTests {

    @Test func pipelineRunSummaryDecodesTheServerShape() throws {
        let json = #"""
            {"run": "20260718T000003000-exp-chain-pipeline", "schema": 2,
             "disposition": "aborted", "experimentHash": "abc",
             "stages": [
               {"stage": "extract", "status": "completed",
                "runID": "20260718T000001000-exp-chain-extract"},
               {"stage": "validate", "status": "completed",
                "runID": "20260718T000002000-exp-chain-validate"},
               {"stage": "sweep", "status": "pending", "runID": null},
               {"stage": "promote", "status": "pending", "runID": null},
               {"stage": "run", "status": "pending", "runID": null}],
             "abort": {"stage": "validate",
                       "gates": [{"passed": false, "stage": "validate",
                                   "gate": "minScenarioAccuracy",
                                   "detail": "concept 'fear' scenario accuracy 0.3 vs floor 0.6",
                                   "measured": 0.3, "threshold": 0.6}],
                       "evidenceRunID": "20260718T000002000-exp-chain-validate"},
             "promotedAgents": {"fear": {
                "artifact": "chain-fear-agent.json", "hash": "ff",
                "sweepRun": "20260718T000000000-exp-chain-sweep",
                "winningCell": {"layer": 2, "alpha": 4.0}}}}
            """#
        let summary = try JSONDecoder().decode(
            ClusterClient.PipelineRunSummary.self, from: Data(json.utf8))
        #expect(summary.stateLabel == "aborted (gate)")
        // The failing stage renders ✕ (it never COMPLETED — the gate said
        // stop after validate; sweep onward never ran).
        #expect(summary.stageSummaryLine
            == "extract ✓ · validate ✓ · sweep – · promote – · run –")
        let gate = try #require(summary.abort?.gates?.first)
        #expect(gate.measured == 0.3)
        #expect(gate.threshold == 0.6)
        #expect(summary.promotedAgents?["fear"]?.winningCell?.layer == 2)

        // No terminal disposition: honestly "unfinished" (the listing
        // cannot know whether a job still runs it — the updatedAt stamp is
        // the evidence); a started stage renders ▸.
        let unfinished = try JSONDecoder().decode(
            ClusterClient.PipelineRunSummary.self, from: Data(#"""
                {"run": "r", "updatedAt": "2026-07-18T12:00:00Z",
                 "stages": [
                   {"stage": "extract", "status": "completed"},
                   {"stage": "run", "status": "started"}]}
                """#.utf8))
        #expect(unfinished.stateLabel == "unfinished")
        #expect(unfinished.updatedAt == "2026-07-18T12:00:00Z")
        #expect(unfinished.stageSummaryLine == "extract ✓ · run ▸")
    }

    @Test func parkedAndEpochNotesDecodeAndLabel() throws {
        // The startup reconciler's orphan stamp (2026-08-06) and the resume
        // guard's loud-and-stamped tolerance notes — cross-engine keys
        // match tasks.list_pipeline_runs exactly.
        let summary = try JSONDecoder().decode(
            ClusterClient.PipelineRunSummary.self, from: Data(#"""
                {"run": "20260806T015828917-exp-replication-1-pipeline",
                 "experiment": "replication-1",
                 "disposition": null,
                 "parked": {"at": "2026-08-06T09:00:00Z",
                            "by": "startup-reconcile",
                            "reason": "remaining stage(s) analyze need the study model",
                            "completedStages": ["run"],
                            "remainingStages": ["analyze"]},
                 "epochDriftAtContinuation": [
                   {"ledgerHash": "623224f0aaaa",
                    "liveHash": "64d52e6fbbbb"}],
                 "stages": [{"stage": "run", "status": "completed",
                             "runID": "20260806T015828917-exp-replication-1-run"},
                            {"stage": "analyze", "status": "pending"}]}
                """#.utf8))
        #expect(summary.stateLabel == "parked")
        #expect(summary.experiment == "replication-1")
        #expect(summary.hasCompletedStages)
        #expect(summary.parked?.reason?.contains("study model") == true)
        #expect(summary.parked?.completedStages == ["run"])
        #expect(summary.epochDriftAtContinuation?.first?.ledgerHash
            == "623224f0aaaa")
        #expect(summary.epochDriftAtContinuation?.first?.liveHash
            == "64d52e6fbbbb")
    }

    @Test func importTriageSurfacesActionableUnimportedChains() throws {
        func row(
            _ run: String, disposition: String? = nil,
            parked: ClusterClient.PipelineRunSummary.Parked? = nil,
            stageStatus: String = "completed"
        ) -> ClusterClient.PipelineRunSummary {
            ClusterClient.PipelineRunSummary(
                run: run, disposition: disposition, parked: parked,
                stages: [.init(stage: "run", status: stageStatus,
                               runID: nil)])
        }
        let parked = ClusterClient.PipelineRunSummary.Parked(
            at: nil, by: nil, reason: "orphaned", completedStages: ["run"],
            remainingStages: [])
        let rows = [
            row("parked-chain", parked: parked),
            row("completed-chain", disposition: "completed"),
            row("aborted-chain", disposition: "aborted"),
            row("plausibly-live-chain"),  // null disposition, not parked
            row("nothing-done-chain", parked: parked, stageStatus: "pending"),
            row("already-ledgered", disposition: "completed"),
            row("already-local", disposition: "completed"),
        ]
        let surfaced = PipelineImportTriage.awaitingImport(
            rows,
            importedRunIDs: ["already-ledgered"],
            localRunExists: { $0 == "already-local" })
        #expect(surfaced.map(\.run)
            == ["parked-chain", "completed-chain", "aborted-chain"])
    }

    @Test func pipelineDraftRoundTripsTheCrossEngineSchema() throws {
        // Composer draft → JSONValue → draft is identity, and the encoded
        // block is exactly what the server resolver consumes.
        var draft = PipelineDraft(
            stages: ["extract", "validate", "sweep", "promote", "run"],
            minScenarioAccuracy: 0.6, maxCrossConceptCosine: 0.8,
            requireSelectionForEveryConcept: true)
        #expect(PipelineDraft.parse(draft.encoded()) == draft)
        #expect(!draft.isGateless)
        // The encoded block passes the verify mirror.
        #expect(ExperimentStore.pipelineBlockViolations(draft.encoded()).isEmpty)

        // Stage toggling preserves canonical order regardless of toggle
        // sequence — the composer can never produce an out-of-order chain.
        draft.setStage("analyze", enabled: true)
        draft.setStage("evaluate", enabled: true)
        #expect(draft.stages == PipelineDraft.allStages)
        draft.setStage("sweep", enabled: false)
        #expect(draft.stages == [
            "extract", "validate", "promote", "run", "evaluate", "analyze",
        ])

        // Gate-less drafts say so, and encode without a gates key.
        let bare = PipelineDraft()
        #expect(bare.isGateless)
        guard case .object(let block) = bare.encoded() else {
            Issue.record("encoded draft must be an object")
            return
        }
        #expect(block["gates"] == nil)
        // Lenient parse: nil and non-object blocks read as nil.
        #expect(PipelineDraft.parse(nil) == nil)
        #expect(PipelineDraft.parse(.string("x")) == nil)
    }

    @Test func pipelineBlockViolationsMirrorTheServerResolver() {
        func block(_ json: String) -> JSONValue? {
            try? JSONDecoder().decode(JSONValue.self, from: Data(json.utf8))
        }
        // Absent and well-formed blocks are clean.
        #expect(ExperimentStore.pipelineBlockViolations(nil).isEmpty)
        #expect(ExperimentStore.pipelineBlockViolations(block(#"""
            {"stages": ["extract", "validate", "sweep", "promote", "run"],
             "gates": {"validate": {"minScenarioAccuracy": 0.6,
                                     "maxCrossConceptCosine": 0.8},
                        "sweep": {"requireSelectionForEveryConcept": true}}}
            """#)).isEmpty)
        // Refusals mirror the server: unknown stage, order, duplicates,
        // evaluate-without-run, unknown keys, gate-for-absent-stage,
        // out-of-range threshold.
        func violations(_ json: String) -> [String] {
            ExperimentStore.pipelineBlockViolations(block(json))
        }
        #expect(violations(#"{"stages": ["teleport"]}"#)
            .contains { $0.contains("unknown stage") })
        #expect(violations(#"{"stages": ["run", "extract"]}"#)
            .contains { $0.contains("canonical order") })
        #expect(violations(#"{"stages": ["extract", "extract"]}"#)
            .contains { $0.contains("duplicates") })
        #expect(violations(#"{"stages": ["extract", "analyze"]}"#)
            .contains { $0.contains("require 'run'") })
        #expect(violations(#"{"stage": ["extract"]}"#)
            .contains { $0.contains("unknown pipeline key") })
        #expect(
            violations(#"{"stages": ["extract", "run"], "gates": {"validate": {}}}"#)
                .contains { $0.contains("not in the stage list") })
        #expect(violations(#"{"gates": {"promote": {}}}"#)
            .contains { $0.contains("no gate is defined") })
        #expect(violations(
            #"{"gates": {"validate": {"minScenarioAccuracy": 1.5}}}"#)
            .contains { $0.contains("[0, 1]") })
        #expect(violations(
            #"{"gates": {"sweep": {"requireSelection": true}}}"#)
            .contains { $0.contains("unknown sweep-gate key") })
    }

    @Test func manifestVerifyCarriesPipelineBlockViolations() throws {
        let json = #"""
            {"name": "pv", "experimentDescription": "d",
             "createdAt": "2026-07-18", "modelID": "test/model",
             "status": "draft",
             "pipeline": {"stages": ["run", "extract"]}}
            """#
        let manifest = try JSONDecoder().decode(
            ExperimentManifest.self, from: Data(json.utf8))
        #expect(ExperimentStore.verify(manifest).contains {
            $0.contains("canonical order")
        })
    }

    @Test func gatelessPipelineAdvisoryAndForwardRefsAreNotHandCreated()
        throws
    {
        var manifest = ExperimentManifest(
            name: "adv", description: "", modelID: "test/model")
        manifest.pipeline = try JSONDecoder().decode(
            JSONValue.self, from: Data(#"{"stages": ["extract", "run"]}"#.utf8))
        // A forward-referenced condition: sweep-promoted by definition —
        // it must NOT trip the hand-created advisory.
        manifest.variantConditions = [
            .init(
                name: "fear-agent", artifactPath: "", artifactHash: "",
                artifact: .init(
                    name: "", baseModelID: "", promptMode: "",
                    qwenThinkingEnabled: false, temperature: 0,
                    systemPrompt: ""),
                fromPromotion: .init(concept: "fear"))
        ]
        let advisories = ExperimentStore.freezeAdvisories(for: manifest)
        #expect(advisories.contains { $0.contains("pipeline declares no gates") })
        #expect(!advisories.contains { $0.contains("hand-created") })

        // Gates present → the advisory clears.
        manifest.pipeline = try JSONDecoder().decode(
            JSONValue.self,
            from: Data(#"""
                {"gates": {"sweep": {"requireSelectionForEveryConcept": true}}}
                """#.utf8))
        #expect(!ExperimentStore.freezeAdvisories(for: manifest).contains {
            $0.contains("pipeline declares no gates")
        })
    }

    /// Review 2026-08-02 (P1): a metric selected with no minimum ENCODED
    /// as no floor at all — the stripped block validated clean, isGateless
    /// went quiet, and Save succeeded. A researcher could believe a
    /// preregistered AUC gate existed while the manifest carried none.
    @Test func metricWithoutMinimumIsACompletenessViolationNotASilentNoGate() throws {
        var draft = PipelineDraft()
        draft.accuracyFloorMetric = "auc"
        // The encoding DOES drop the incomplete block — that is exactly
        // why completeness must be checked before it.
        let encoded = draft.encoded()
        #expect(
            ExperimentStore.pipelineBlockViolations(encoded).isEmpty)
        #expect(!draft.completenessViolations.isEmpty)
        #expect(draft.completenessViolations.first?.contains(
            "NO accuracy gate") == true)
        // isGateless is quiet (a gate is INTENDED) — the completeness
        // violation is what carries the refusal.
        #expect(!draft.isGateless)
        // Completing the declaration clears it and the block round-trips.
        draft.accuracyFloorMinimum = 0.7
        #expect(draft.completenessViolations.isEmpty)
        let parsed = try #require(PipelineDraft.parse(draft.encoded()))
        #expect(parsed.accuracyFloorMetric == "auc")
        #expect(parsed.accuracyFloorMinimum == 0.7)
        // The inverse hole (minimum, no metric) is named too.
        var inverse = PipelineDraft()
        inverse.accuracyFloorMinimum = 0.7
        #expect(inverse.completenessViolations.first?.contains(
            "no metric") == true)
    }

    /// The declared-floor block mirrors the server resolver's refusals.
    @Test func accuracyFloorBlockViolationsMirrorTheServerResolver() throws {
        func violations(_ json: String) throws -> [String] {
            ExperimentStore.pipelineBlockViolations(
                try JSONDecoder().decode(JSONValue.self, from: Data(json.utf8)))
        }
        #expect(try violations(#"""
            {"gates": {"validate": {"accuracyFloor":
                {"metric": "auc", "minimum": 0.7}}}}
            """#).isEmpty)
        #expect(try violations(#"""
            {"gates": {"validate": {"accuracyFloor":
                {"metric": "vibes", "minimum": 0.7}}}}
            """#).contains { $0.contains("unknown accuracyFloor metric") })
        #expect(try violations(#"""
            {"gates": {"validate": {"accuracyFloor": {"metric": "auc"}}}}
            """#).contains { $0.contains("must be an object") })
        #expect(try violations(#"""
            {"gates": {"validate": {"minScenarioAccuracy": 0.6,
                "accuracyFloor": {"metric": "auc", "minimum": 0.7}}}}
            """#).contains { $0.contains("declare exactly one") })
        // String/bool minimums refuse here exactly as the server does
        // since 2026-08-02 (float("0.7") used to resolve there).
        #expect(try violations(#"""
            {"gates": {"validate": {"accuracyFloor":
                {"metric": "auc", "minimum": "0.7"}}}}
            """#).contains { $0.contains("must be a number") })
    }
}
