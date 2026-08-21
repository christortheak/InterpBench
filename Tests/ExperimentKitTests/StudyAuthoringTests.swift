import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// Study-authoring streamline (2026-07-19): intent derivation (a VIEW
/// classification — filters presentation, never data), the Data & Prompts
/// grouping vocabulary, and study.json export/import with the firewall
/// applied identically to pasted and hand-built studies.
struct StudyAuthoringTests {

    private func manifest(_ mutate: (inout ExperimentManifest) -> Void = { _ in })
        -> ExperimentManifest
    {
        var manifest = ExperimentManifest(
            name: "sa", description: "", modelID: "test/model")
        mutate(&manifest)
        return manifest
    }

    @Test func intentDerivesFromManifestContent() {
        #expect(StudyIntent.derive(from: manifest()) == .conceptStudy)
        #expect(StudyIntent.derive(from: manifest {
            $0.variantConditions = [
                .init(name: "a", artifactPath: "p", artifactHash: "h",
                      artifact: .init(
                          name: "a", baseModelID: "test/model",
                          promptMode: "chatAssistant",
                          qwenThinkingEnabled: false, temperature: 0,
                          systemPrompt: ""))
            ]
        }) == .agentComparison)
        // Concepts win over variants: a hybrid is a concept study.
        #expect(StudyIntent.derive(from: manifest {
            $0.concepts = [ExperimentManifest.ConceptRef(
                name: "fear", stimulusSetHash: "00",
                options: ExtractionOptions(method: .meanDifference))]
            $0.variantConditions = [
                .init(name: "a", artifactPath: "p", artifactHash: "h",
                      artifact: .init(
                          name: "a", baseModelID: "test/model",
                          promptMode: "chatAssistant",
                          qwenThinkingEnabled: false, temperature: 0,
                          systemPrompt: ""))
            ]
        }) == .conceptStudy)
        #expect(StudyIntent.derive(from: manifest {
            $0.studyKind = .multiAgent
        }) == .multiAgent)
        // A perturbation policy is the CONCEPT study's confirm phase
        // (2026-07-19 fold-in — mechanically the same machinery).
        #expect(StudyIntent.derive(from: manifest {
            $0.perturbationPolicy = .init(
                sourceAgent: .init(
                    name: "a", artifactPath: "p", artifactHash: "h",
                    promoted: true),
                concept: "fear",
                cell: .init(layer: 10, alpha: 4.0),
                alphaDeltas: [0.2],
                includeMatchedNormControl: true,
                declaredAt: "2026-07-19")
        }) == .conceptStudy)
        // Legacy alias: "confirmAgent" (a top-level type until 2026-07-19)
        // still parses — as the concept study.
        #expect(StudyIntent.parse("confirmAgent") == .conceptStudy)
        #expect(StudyIntent.parse("conceptStudy") == .conceptStudy)
        #expect(StudyIntent.parse("vibes") == nil)
        #expect(StudyIntent.derive(from: manifest {
            $0.studyType = "confirmAgent"
        }) == .conceptStudy)
        // …and the contract accepts the alias (never "unknown studyType").
        #expect(ExperimentStore.studyTypeContractViolations(manifest {
            $0.studyType = "confirmAgent"
        }).isEmpty)
    }

    /// The single classifier replaced the Study stage / Study Focus duo —
    /// every type maps onto a manifest kind (no more pointer stages), and
    /// the panel setter writes studyKind on drafts only.
    @Test func unifiedStudyTypeWritesDraftKindAndNeverFrozen() {
        #expect(StudyIntent.multiAgent.mappedKind == .multiAgent)
        for intent in [StudyIntent.conceptStudy, .agentComparison] {
            #expect(intent.mappedKind == .modelOutput)
        }
        for intent in StudyIntent.allCases {
            #expect(!intent.displayName.isEmpty)
            #expect(!intent.explanation.isEmpty)
            // The structured guide (display pane) is complete for every
            // type: tagline, prose, provided-items, measured-claims.
            #expect(!intent.tagline.isEmpty)
            #expect(!intent.whatItIs.isEmpty)
            #expect(!intent.youProvide.isEmpty)
            #expect(!intent.itMeasures.isEmpty)
            #expect(intent.youProvide.contains { $0.required })
        }
    }

    /// Durability (engineer finding, P2): the declared type persists in
    /// the manifest and wins over content derivation — an empty comparison
    /// stays "Compare agents" across selection changes. An INCONSISTENT
    /// declaration (hand-edited JSON whose studyKind says otherwise) is
    /// ignored: the run path never follows a label it contradicts.
    @Test func declaredStudyTypePersistsAndWinsWhenConsistent() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "satype") { _ in
            _ = try ExperimentStore.create(
                name: "typed", description: "d", modelID: "test/model")
            let updated = try ExperimentStore.setStudyType(
                .agentComparison, experimentName: "typed")
            #expect(updated.studyType == "agentComparison")
            #expect(updated.studyKind == .modelOutput)
            #expect(StudyIntent.derive(from: updated) == .agentComparison)
            // Round-trips through disk.
            let loaded = try ExperimentStore.load(name: "typed")
            #expect(StudyIntent.derive(from: loaded) == .agentComparison)
            // Inconsistent declaration loses to the engine-facing kind.
            var inconsistent = loaded
            inconsistent.studyKind = .multiAgent
            #expect(StudyIntent.derive(from: inconsistent) == .multiAgent)
            // Draft-only: freezing on disk refuses the setter.
            var frozen = loaded
            frozen.status = .frozen
            try ExperimentStore.save(frozen)
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.setStudyType(
                    .conceptStudy, experimentName: "typed")
            }
        }
    }

    /// Engineer finding (P1): pasted names become directory paths — the
    /// import applies create()'s sanitization, so traversal or separator
    /// characters can never escape the experiments store.
    @Test func importSanitizesPathUnsafeNames() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "saname") { root in
            let escape = try ExperimentStore.exportStudyJSON(manifest {
                $0.name = "../escape"
            })
            let (imported, _, _) = try ExperimentStore.importStudyJSON(escape)
            #expect(imported.name == "escape")
            #expect(FileManager.default.fileExists(
                atPath: root.appending(
                    components: "experiments", "escape", "experiment.json").path))
            // Nothing landed OUTSIDE experiments/.
            #expect(!FileManager.default.fileExists(
                atPath: root.appending(component: "escape").path))
            let nested = try ExperimentStore.exportStudyJSON(manifest {
                $0.name = "a/b Study"
            })
            let (flattened, _, _) = try ExperimentStore.importStudyJSON(nested)
            #expect(flattened.name == "ab-study")
            // A name that is NOTHING BUT unsafe characters refuses.
            let hopeless = try ExperimentStore.exportStudyJSON(manifest {
                $0.name = "../.."
            })
            #expect(throws: ExperimentError.self) {
                _ = try ExperimentStore.importStudyJSON(hopeless)
            }
        }
    }

    /// Engineer finding (P1): a fresh pipeline declaration seeds only the
    /// stages the study type makes relevant — a compare-agents chain
    /// starts as `run`, never extract → ….
    @Test func pipelineSeedStagesDeriveFromStudyType() {
        #expect(PipelineDraft.seedStages(
            relevant: StudyIntent.conceptStudy.relevantPipelineStages)
            == PipelineDraft.defaultStages)
        #expect(PipelineDraft.seedStages(
            relevant: StudyIntent.agentComparison.relevantPipelineStages)
            == ["run"])
        #expect(PipelineDraft.seedStages(
            relevant: StudyIntent.multiAgent.relevantPipelineStages)
            == ["run"])
    }

    /// Type-switch preservation, closed (engineer finding 2026-07-19):
    /// carried configuration from another study type must be INERT — it
    /// cannot block verification, does not enter the pin/packaging
    /// surface, and is announced as a freeze advisory instead.
    @Test func carriedConfigIsInertForTheOtherStudyKind() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "sacarry") { _ in
            var carried = manifest {
                $0.studyKind = .multiAgent
                // Stale model-output config a type switch preserved: a
                // concept whose stimuli do not exist, a missing prompts
                // file, an agent whose artifact is gone.
                $0.concepts = [ExperimentManifest.ConceptRef(
                    name: "ghost", stimulusSetHash: "00",
                    options: ExtractionOptions(method: .meanDifference))]
                $0.taskPromptsFile = "prompts/absent.jsonl"
                $0.taskPromptsHash = "11"
                $0.variantConditions = [
                    .init(name: "v", artifactPath: "missing.json",
                          artifactHash: "deadbeef",
                          artifact: .init(
                              name: "v", baseModelID: "test/model",
                              promptMode: "chatAssistant",
                              qwenThinkingEnabled: false, temperature: 0,
                              systemPrompt: ""))
                ]
            }
            // Without a scenario the ONLY violation is the scenario rule —
            // none of the carried (broken) config blocks.
            #expect(ExperimentStore.verify(carried)
                == ["multi-agent study needs a pinned scenario"])
            // The pin surface skips carried config: no concept, prompt,
            // or variant entries in a multi-agent study's bundle.
            let labels = ExperimentStore.pinnedInputEntries(carried).map(\.label)
            #expect(!labels.contains {
                $0.contains("concept") || $0.contains("task prompts")
                    || $0.contains("variant")
            })
            // The carried state is SAID at freeze time.
            #expect(ExperimentStore.freezeAdvisories(for: carried)
                .contains { $0.contains("another study type") })

            // The mirror direction: a model-output study carrying a
            // scenario pin neither verifies nor packages it.
            carried.studyKind = .modelOutput
            carried.multiAgentScenarioPath = "prompts/scenarios/ghost.json"
            carried.multiAgentScenarioHash = "22"
            let moLabels = ExperimentStore.pinnedInputEntries(carried).map(\.label)
            #expect(!moLabels.contains { $0.contains("scenario") })
            #expect(moLabels.contains { $0.contains("task prompts") })
            #expect(!ExperimentStore.verify(carried).contains {
                $0.contains("scenario")
            })
        }
    }

    /// studyType contract (engineer finding 2026-07-19): unknown or
    /// kind-contradicting labels are verify violations — LLM-authored
    /// JSON gets loud feedback, never a silent re-derive.
    @Test func studyTypeContractIsValidatedLoudly() {
        #expect(ExperimentStore.studyTypeContractViolations(manifest {
            $0.studyType = "vibes"
        }).first?.contains("unknown studyType") == true)
        #expect(ExperimentStore.studyTypeContractViolations(manifest {
            $0.studyType = "multiAgent"  // studyKind stays modelOutput
        }).first?.contains("contradicts studyKind") == true)
        #expect(ExperimentStore.studyTypeContractViolations(manifest {
            $0.studyType = "agentComparison"
        }).isEmpty)
        #expect(ExperimentStore.studyTypeContractViolations(manifest()).isEmpty)
        // And verify() carries the contract.
        #expect(ExperimentStore.verify(manifest { $0.studyType = "vibes" })
            .contains { $0.contains("unknown studyType") })
    }

    /// Second engineer round (2026-07-19): the operative surface is keyed
    /// to the durable studyType WITHIN model-output — a declared
    /// compare-agents study without forward references carries its
    /// concept machinery inert (verify, packaging, readiness, and the
    /// RUN's condition selection all agree), while forward references or
    /// a concept-study/confirmation type keep it fully active.
    @Test func conceptMachineryFollowsTheDeclaredStudyType() {
        let ghost = ExperimentManifest.ConceptRef(
            name: "ghost", stimulusSetHash: "00",
            options: ExtractionOptions(method: .meanDifference))
        let injection = ExperimentManifest.Condition(
            name: "ghost-4", slots: [.init(concept: "ghost", layer: 4, alpha: 2)])
        let agent = ExperimentManifest.VariantCondition(
            name: "a", artifactPath: "p", artifactHash: "h",
            artifact: .init(
                name: "a", baseModelID: "test/model",
                promptMode: "chatAssistant", qwenThinkingEnabled: false,
                temperature: 0, systemPrompt: ""))

        // Declared compare-agents with carried concepts, no forward refs:
        // machinery INERT — no concept violations, no injection conditions
        // at run time, carried advisory fires.
        let comparison = manifest {
            $0.studyType = "agentComparison"
            $0.concepts = [ghost]
            $0.conditions = [injection]
            $0.variantConditions = [agent]
        }
        #expect(!ExperimentStore.conceptMachineryOperative(comparison))
        #expect(!ExperimentStore.verify(comparison).contains {
            $0.contains("ghost")
        })
        #expect(ExperimentTasks.ordinaryRunConditions(for: comparison)
            .map(\.name) == ["baseline"])
        #expect(ExperimentStore.freezeAdvisories(for: comparison)
            .contains { $0.contains("another study type") })
        // The readiness scan downgrades carried concepts to one optional
        // row — never a red blocker in the Issues box.
        let rows = StudyDataReadiness.requirements(
            for: comparison, workspaceRoot: FileManager.default.temporaryDirectory)
        #expect(!rows.contains {
            $0.kind == .conceptStimuli && $0.status == .missing
        })
        #expect(rows.contains { $0.id == "concepts:carried" })

        // A forward reference flips the machinery ON (the study's own
        // sweep must extract and select on that concept).
        var forward = comparison
        forward.variantConditions.append(
            .init(name: "ghost-agent", artifactPath: "", artifactHash: "",
                  artifact: .init(
                      name: "", baseModelID: "", promptMode: "",
                      qwenThinkingEnabled: false, temperature: 0,
                      systemPrompt: ""),
                  fromPromotion: .init(concept: "ghost")))
        #expect(ExperimentStore.conceptMachineryOperative(forward))

        // Concept studies and confirmations keep the machinery active,
        // and their runs execute the injection conditions.
        let concept = manifest {
            $0.studyType = "conceptStudy"
            $0.concepts = [ghost]
            $0.conditions = [injection]
        }
        #expect(ExperimentStore.conceptMachineryOperative(concept))
        #expect(ExperimentTasks.ordinaryRunConditions(for: concept)
            .map(\.name) == ["baseline", "ghost-4"])
    }

    /// Fail-closed frozen-on-server guard (second engineer round): a known
    /// draft is clean, a known frozen/complete status refuses, and an
    /// UNKNOWN status refuses too — "could not check" must never read as
    /// "checked and clean" on the evidence path (404 alone means "no
    /// same-named study" and is handled by the async caller).
    @Test func frozenOnServerGuardFailsClosed() {
        #expect(ClusterClient.frozenOnServerConflictMessage(
            study: "s", remoteStatus: "draft") == nil)
        // An ABSENT status is not clean — same fail-closed refusal as an
        // unreachable server (second-round finding: nil read as ok).
        #expect(ClusterClient.frozenOnServerConflictMessage(
            study: "s", remoteStatus: nil)?.contains("refusing") == true)
        #expect(ClusterClient.frozenOnServerConflictMessage(
            study: "s", remoteStatus: "frozen")?.contains("frozen") == true)
        #expect(ClusterClient.frozenOnServerConflictMessage(
            study: "s", remoteStatus: "complete")?.contains("complete") == true)
        let refusal = ClusterClient.statusUnavailableRefusal(
            study: "s", error: "timeout")
        #expect(refusal.contains("refusing"))
        #expect(refusal.contains("timeout"))
    }

    /// Study packs ("one file drives the study"): the envelope's files
    /// land contained under prompts/, identical re-imports are idempotent,
    /// differing overwrites refuse, traversal refuses, and named inputs
    /// are pinned from the just-written bytes.
    @Test func studyPackWritesContainedFilesAndPinsNamedInputs() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "sapack") { root in
            let promptsLine = "{\"text\": \"first prompt\"}\n"
            let pack = """
            {
              "study": {
                "name": "packed", "status": "draft",
                "studyType": "agentComparison",
                "studyKind": "modelOutput",
                "experimentDescription": "packed study",
                "modelID": "test/model", "createdAt": "2026-07-19",
                "temperature": 0, "maxTokens": 64, "seeds": [0],
                "multiAgentIncludeBaseline": true,
                "taskPromptsFile": "prompts/tasks/packed.jsonl",
                "concepts": [], "conditions": [], "variantConditions": []
              },
              "files": {
                "prompts/tasks/packed.jsonl": "{\\"text\\": \\"first prompt\\"}\\n"
              }
            }
            """
            let (imported, _, written) = try ExperimentStore.importStudyJSON(pack)
            #expect(written == ["prompts/tasks/packed.jsonl"])
            let url = root.appending(path: "prompts/tasks/packed.jsonl")
            #expect(try String(contentsOf: url, encoding: .utf8) == promptsLine)
            // The named task prompts were PINNED from the written bytes.
            #expect(imported.taskPromptsHash != nil)

            // Traversal and absolute paths refuse.
            let evil = pack
                .replacingOccurrences(of: "\"packed\"", with: "\"packed2\"")
                .replacingOccurrences(
                    of: "prompts/tasks/packed.jsonl",
                    with: "prompts/../../etc/owned")
            #expect(throws: ExperimentError.self) {
                _ = try ExperimentStore.importStudyJSON(evil)
            }

            // A differing existing file refuses (packs never overwrite).
            try Data("different\n".utf8).write(
                to: root.appending(path: "prompts/tasks/other.jsonl"))
            let clash = pack
                .replacingOccurrences(of: "\"packed\"", with: "\"packed3\"")
                .replacingOccurrences(
                    of: "prompts/tasks/packed.jsonl",
                    with: "prompts/tasks/other.jsonl")
            #expect(throws: ExperimentError.self) {
                _ = try ExperimentStore.importStudyJSON(clash)
            }
        }
    }

    /// Third engineer round (2026-07-19): mixed arm modes refuse — when
    /// agent conditions exist the engines run agents only, so a
    /// hand-declared injection condition would silently vanish from the
    /// evidence. Sweep-stamped conditions (selection provenance) are
    /// exempt: the promoted agent IS their executable form.
    @Test func mixedHandDeclaredArmsAreAVerifyViolation() {
        let ghost = ExperimentManifest.ConceptRef(
            name: "ghost", stimulusSetHash: "00",
            options: ExtractionOptions(method: .meanDifference))
        let hand = ExperimentManifest.Condition(
            name: "ghost-4", slots: [.init(concept: "ghost", layer: 4, alpha: 2)])
        let agent = ExperimentManifest.VariantCondition(
            name: "a", artifactPath: "p", artifactHash: "h",
            artifact: .init(
                name: "a", baseModelID: "test/model",
                promptMode: "chatAssistant", qwenThinkingEnabled: false,
                temperature: 0, systemPrompt: ""))
        let mixed = manifest {
            $0.studyType = "conceptStudy"
            $0.concepts = [ghost]
            $0.conditions = [hand]
            $0.variantConditions = [agent]
        }
        #expect(ExperimentStore.verify(mixed).contains {
            $0.contains("mixed arm modes") && $0.contains("ghost-4")
        })
        // STRUCTURALLY CONSISTENT selection provenance exempts the
        // condition (fourth round: presence alone is not proof — the
        // condition must BE the winning cell under the sweep's name
        // convention; run-directory existence is deliberately not
        // required because runs are per-substrate trees) — but ONLY
        // together with its EXECUTABLE TWIN among the agent arms (fifth
        // round: self-consistency alone did not prove the promoted agent
        // actually runs; sixth round: the twin needs COMPLETE
        // birth-certificate identity — concept + sweepRun + cell — and a
        // forward reference, which binds to no selection, no longer
        // counts).
        func recommended(
            name: String, layer: Int, alpha: Double,
            cell: ExperimentManifest.SelectionProvenance.Cell
        ) -> ExperimentManifest.Condition {
            ExperimentManifest.Condition(
                name: name,
                slots: [.init(concept: "ghost", layer: layer, alpha: alpha)],
                selection: ExperimentManifest.SelectionProvenance(
                    sweepRun: "runs/x",
                    criterion: ExperimentManifest.SweepSelection(
                        objective: .init(metric: "markerDensity")),
                    devPromptsHash: "00",
                    winningCell: cell,
                    metrics: [:]))
        }
        let forwardTwin = ExperimentManifest.VariantCondition(
            name: "ghost-agent", artifactPath: "", artifactHash: "",
            artifact: .init(
                name: "", baseModelID: "", promptMode: "",
                qwenThinkingEnabled: false, temperature: 0,
                systemPrompt: ""),
            fromPromotion: .init(concept: "ghost"))
        // Identity-complete twin: injects the concept, names the SAME
        // sweep run, and certifies the SAME winning cell. (The Promotion
        // certificate has no concept field — concept identity is read
        // from the artifact's injections, which promote always writes.)
        let promotedTwin = ExperimentManifest.VariantCondition(
            name: "ghost-promoted", artifactPath: "p", artifactHash: "h",
            artifact: .init(
                name: "ghost-promoted", baseModelID: "test/model",
                injections: [
                    .init(concept: "ghost", vectorArtifactID: "v",
                          layer: 4, alpha: 2)
                ],
                promptMode: "chatAssistant", qwenThinkingEnabled: false,
                temperature: 0, systemPrompt: "",
                promotion: .init(
                    experiment: "mix", experimentHash: "00",
                    promotedAt: "2026-07-19T00:00:00Z",
                    promotedBy: "criterion", sweepRun: "runs/x",
                    winningCell: .init(layer: 4, alpha: 2),
                    substrate: "swift-mlx", appVersion: "test")))
        // Stamped + ONLY a forward reference for the concept: REFUSED
        // (sixth round — the rule changed deliberately). A forward
        // reference binds to no selection; its future sweep may pick a
        // different cell, so it cannot prove the recommended cell runs.
        // The refusal names the sweep run the recommendation is bound to.
        var stamped = mixed
        stamped.conditions = [recommended(
            name: "ghost-recommended", layer: 4, alpha: 2,
            cell: .init(layer: 4, alpha: 2))]
        stamped.variantConditions = [agent, forwardTwin]
        #expect(ExperimentStore.verify(stamped).contains {
            $0.contains("ghost-recommended")
                && $0.contains("not among this study's arms")
                && $0.contains("bound to sweep run 'runs/x'")
        })
        // Stamped + a concrete agent whose birth certificate carries the
        // COMPLETE identity (concept + sweep run + cell): exempt.
        var stampedConcrete = stamped
        stampedConcrete.variantConditions = [promotedTwin]
        #expect(!ExperimentStore.verify(stampedConcrete).contains {
            $0.contains("mixed arm modes") || $0.contains("not among")
        })
        // Stamped with ONLY an unrelated agent: the recommended cell's
        // promoted agent is not among the arms, so nothing would execute
        // it — refused, with the concept named.
        var stampedOrphan = stamped
        stampedOrphan.variantConditions = [agent]
        #expect(ExperimentStore.verify(stampedOrphan).contains {
            $0.contains("ghost-recommended")
                && $0.contains("not among this study's arms")
                && $0.contains("'ghost'")
        })
        // A birth certificate for a DIFFERENT cell is not the twin either.
        var wrongCell = stamped
        var otherPromotion = promotedTwin
        otherPromotion.artifact.promotion?.winningCell = .init(
            layer: 9, alpha: 2)
        wrongCell.variantConditions = [otherPromotion]
        #expect(ExperimentStore.verify(wrongCell).contains {
            $0.contains("not among this study's arms")
        })
        // Same cell, DIFFERENT concept injected: not the twin (the agent
        // executes another concept's vector).
        var wrongConcept = stamped
        var otherConceptTwin = promotedTwin
        otherConceptTwin.artifact.injections = [
            .init(concept: "other", vectorArtifactID: "v", layer: 4, alpha: 2)
        ]
        wrongConcept.variantConditions = [otherConceptTwin]
        #expect(ExperimentStore.verify(wrongConcept).contains {
            $0.contains("not among this study's arms")
        })
        // Same concept + cell but a DIFFERENT sweep run: not the twin
        // (the certificate binds the agent to another selection).
        var wrongRun = stamped
        var otherRunTwin = promotedTwin
        otherRunTwin.artifact.promotion?.sweepRun = "runs/y"
        wrongRun.variantConditions = [otherRunTwin]
        #expect(ExperimentStore.verify(wrongRun).contains {
            $0.contains("not among this study's arms")
        })
        // An EMPTY promotion sweepRun refuses too: a stamped selection
        // always names its run, so an unbound certificate proves nothing
        // (the old "when both carry one" hole).
        var emptyRun = stamped
        var unboundTwin = promotedTwin
        unboundTwin.artifact.promotion?.sweepRun = nil
        emptyRun.variantConditions = [unboundTwin]
        #expect(ExperimentStore.verify(emptyRun).contains {
            $0.contains("not among this study's arms")
        })
        // Seventh round (2026-07-20): the certificate proved what was
        // CLAIMED, not what RUNS. A certificate carrying the right
        // identity while the artifact INJECTS a different cell is not
        // the twin — what executes is not what the sweep selected.
        var injectionCell = stamped
        var mismatchedInjectionTwin = promotedTwin
        mismatchedInjectionTwin.artifact.injections = [
            .init(concept: "ghost", vectorArtifactID: "v",
                  layer: 9, alpha: 9)
        ]
        injectionCell.variantConditions = [mismatchedInjectionTwin]
        #expect(ExperimentStore.verify(injectionCell).contains {
            $0.contains("not among this study's arms")
        })
        // An EXTRA second injection is not the twin either: promote
        // mints exactly one, and a second vector means the arm executes
        // a mix the sweep never selected.
        var extraInjection = stamped
        var extraInjectionTwin = promotedTwin
        extraInjectionTwin.artifact.injections.append(
            .init(concept: "stow", vectorArtifactID: "w",
                  layer: 6, alpha: 1))
        extraInjection.variantConditions = [extraInjectionTwin]
        #expect(ExperimentStore.verify(extraInjection).contains {
            $0.contains("not among this study's arms")
        })
        // A well-shaped but INCONSISTENT block does not exempt: the slot
        // is not the winning cell, so what runs is not what the sweep
        // selected.
        var forged = mixed
        forged.conditions = [recommended(
            name: "ghost-recommended", layer: 4, alpha: 9,
            cell: .init(layer: 4, alpha: 2))]
        #expect(ExperimentStore.verify(forged).contains {
            $0.contains("mixed arm modes")
        })
        // Wrong name convention refuses too.
        var misnamed = mixed
        misnamed.conditions = [recommended(
            name: "my-arm", layer: 4, alpha: 2,
            cell: .init(layer: 4, alpha: 2))]
        #expect(ExperimentStore.verify(misnamed).contains {
            $0.contains("mixed arm modes")
        })
        // The canonical empty baseline is exempt (Add Baseline creates
        // it; the agent path runs an equivalent) — a CUSTOM-named empty
        // condition still refuses (its identity would vanish).
        var withBaseline = mixed
        withBaseline.conditions = [
            ExperimentManifest.Condition(name: "baseline", slots: [])
        ]
        #expect(!ExperimentStore.verify(withBaseline).contains {
            $0.contains("mixed arm modes")
        })
        var customEmpty = mixed
        customEmpty.conditions = [
            ExperimentManifest.Condition(name: "my-baseline", slots: [])
        ]
        #expect(ExperimentStore.verify(customEmpty).contains {
            $0.contains("mixed arm modes")
        })
        // Inert machinery (declared comparison) never fires it — the
        // carried-config advisory covers that state instead.
        var carried = mixed
        carried.studyType = "agentComparison"
        #expect(!ExperimentStore.verify(carried).contains {
            $0.contains("mixed arm modes")
        })
    }

    /// The degenerate inert case (observed live 2026-08-11: the c20-*
    /// cluster fan-out produced baseline-only evidence): a declared
    /// agentComparison with NO agent arms at all but carrying injection
    /// conditions would run baseline only, silently — verify names every
    /// dropped condition, and the run path refuses through the same rule
    /// (Python twin `inert_conditions_problem`).
    @Test func inertConditionsWithNoAgentArmsAreAVerifyViolation() {
        let ghost = ExperimentManifest.ConceptRef(
            name: "ghost", stimulusSetHash: "00",
            options: ExtractionOptions(method: .meanDifference))
        let inert = manifest {
            $0.studyType = "agentComparison"
            $0.concepts = [ghost]
            $0.conditions = [
                .init(name: "ghost-4",
                      slots: [.init(concept: "ghost", layer: 4, alpha: 2)]),
                .init(name: "ghost-9",
                      slots: [.init(concept: "ghost", layer: 9, alpha: 1)]),
            ]
            $0.variantConditions = []
        }
        #expect(ExperimentStore.inertConditionsProblem(inert) != nil)
        #expect(ExperimentStore.verify(inert).contains {
            $0.contains("BASELINE ONLY") && $0.contains("ghost-4")
                && $0.contains("ghost-9")
        })
        // And the run path would drop them: the pure condition resolver
        // returns baseline only for this shape — exactly why the refusal
        // exists.
        #expect(ExperimentTasks.ordinaryRunConditions(for: inert)
            .map(\.name) == ["baseline"])

        // Undeclared type: content derivation makes it a concept study —
        // operative, no problem.
        var undeclared = inert
        undeclared.studyType = nil
        #expect(ExperimentStore.inertConditionsProblem(undeclared) == nil)
        #expect(!ExperimentStore.verify(undeclared).contains {
            $0.contains("BASELINE ONLY")
        })
        // With an agent arm the 2026-07-19 rule applies instead (agents
        // run; carried conditions are the mixed-arm/advisory surface).
        var withAgent = inert
        withAgent.variantConditions = [
            .init(name: "a", artifactPath: "p", artifactHash: "h",
                  artifact: .init(
                      name: "a", baseModelID: "test/model",
                      promptMode: "chatAssistant",
                      qwenThinkingEnabled: false, temperature: 0,
                      systemPrompt: ""))
        ]
        #expect(ExperimentStore.inertConditionsProblem(withAgent) == nil)
        // Baseline-only content has nothing to drop.
        var baselineOnly = inert
        baselineOnly.conditions = [.init(name: "baseline", slots: [])]
        #expect(ExperimentStore.inertConditionsProblem(baselineOnly) == nil)

        // The shape that legally PROCEEDS (agent arms exist, carried
        // machinery inert) gets the loud run-start note instead — naming
        // the declared type and every inert condition — while operative
        // and empty shapes stay quiet (2026-08-11 follow-up: baseline-only
        // results must never look ordinary).
        let note = ExperimentStore.inertMachineryNote(withAgent)
        #expect(note != nil)
        #expect(note?.contains("agentComparison") == true)
        #expect(note?.contains("ghost-4") == true)
        #expect(note?.contains("ghost-9") == true)
        #expect(ExperimentStore.inertMachineryNote(undeclared) == nil)
        var carriedConceptsOnly = withAgent
        carriedConceptsOnly.conditions = []
        #expect(ExperimentStore.inertMachineryNote(carriedConceptsOnly) != nil)
        var nothingCarried = withAgent
        nothingCarried.conditions = []
        nothingCarried.concepts = []
        #expect(ExperimentStore.inertMachineryNote(nothingCarried) == nil)
    }

    /// Third engineer round: a carried (inert) sweep must not surface a
    /// false battery blocker, and the pack importer is symlink-safe and
    /// transactional.
    @Test func carriedSweepBatteryIsNotABlocker() {
        let comparison = manifest {
            $0.studyType = "agentComparison"
            $0.variantConditions = [
                .init(name: "a", artifactPath: "p", artifactHash: "h",
                      artifact: .init(
                          name: "a", baseModelID: "test/model",
                          promptMode: "chatAssistant",
                          qwenThinkingEnabled: false, temperature: 0,
                          systemPrompt: ""))
            ]
            $0.sweep = ExperimentManifest.SweepSpec(
                devPromptsFile: "prompts/dev/none.jsonl",
                batteryFile: "prompts/batteries/none.jsonl")
        }
        let rows = StudyDataReadiness.requirements(
            for: comparison,
            workspaceRoot: FileManager.default.temporaryDirectory)
        let batteryBlocked = rows.contains { row in
            row.kind == .capabilityBattery && row.status == .missing
        }
        #expect(!batteryBlocked)
    }

    @Test func studyPackRefusesSymlinkEscapesAndRollsBackOnFailure() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "sapack2") { root in
            let fm = FileManager.default
            // A symlink planted under prompts/ pointing outside the root.
            let outside = root.deletingLastPathComponent()
                .appending(component: "outside-\(root.lastPathComponent)")
            try fm.createDirectory(at: outside, withIntermediateDirectories: true)
            defer { try? fm.removeItem(at: outside) }
            let promptsDir = root.appending(path: "prompts")
            try fm.createDirectory(at: promptsDir, withIntermediateDirectories: true)
            try fm.createSymbolicLink(
                at: promptsDir.appending(component: "link"),
                withDestinationURL: outside)

            func pack(name: String, files: [String: String]) -> String {
                let fileLines = files
                    .map { "\"\($0.key)\": \"\($0.value)\"" }
                    .joined(separator: ", ")
                return """
                {"study": {"name": "\(name)", "status": "draft",
                 "experimentDescription": "d", "modelID": "test/model",
                 "createdAt": "2026-07-19", "temperature": 0,
                 "maxTokens": 64, "seeds": [0],
                 "multiAgentIncludeBaseline": true,
                 "concepts": [], "conditions": [], "variantConditions": []},
                 "files": {\(fileLines)}}
                """
            }

            // Symlink escape refuses and writes nothing outside.
            #expect(throws: ExperimentError.self) {
                _ = try ExperimentStore.importStudyJSON(pack(
                    name: "sym",
                    files: ["prompts/link/evil.jsonl": "x"]))
            }
            #expect(!fm.fileExists(
                atPath: outside.appending(component: "evil.jsonl").path))
            // Nested case (fourth round): containment resolves the
            // nearest EXISTING ancestor BEFORE any directory creation —
            // prompts/link/new/… must not plant a "new/" directory
            // outside the workspace on its way to refusing.
            #expect(throws: ExperimentError.self) {
                _ = try ExperimentStore.importStudyJSON(pack(
                    name: "sym2",
                    files: ["prompts/link/new/evil.jsonl": "x"]))
            }
            #expect(!fm.fileExists(
                atPath: outside.appending(component: "new").path))

            // Transactional: a later collision means the earlier file is
            // NOT left behind (validation precedes the first write).
            try Data("existing\n".utf8).write(
                to: promptsDir.appending(component: "taken.jsonl"))
            #expect(throws: ExperimentError.self) {
                _ = try ExperimentStore.importStudyJSON(pack(
                    name: "txn",
                    files: [
                        "prompts/a-first.jsonl": "aaa",
                        "prompts/taken.jsonl": "differing",
                    ]))
            }
            #expect(!fm.fileExists(
                atPath: promptsDir.appending(component: "a-first.jsonl").path))
        }
    }

    /// The LLM co-authoring prompt teaches the real contract — spot-check
    /// the load-bearing rules so drift between prompt and code fails here.
    @Test func coauthoringPromptTeachesTheContract() {
        // One prompt PER STUDY TYPE (2026-07-19 feedback), each teaching
        // the study-pack envelope and its own type's interview.
        for intent in StudyIntent.allCases {
            let prompt = StudyCoauthoring.prompt(for: intent)
            #expect(prompt.contains("NEVER fabricate hashes"))
            #expect(prompt.contains("\"status\" must be \"draft\""))
            #expect(prompt.contains("lowercase letters, digits, hyphens"))
            #expect(prompt.contains("Paste Study JSON"))
            #expect(prompt.contains("\"files\""))
            #expect(prompt.contains("\"studyType\": \"\(intent.rawValue)\""))
        }
    }

    /// Review 2026-08-02 (P1): the prompt taught `{"text","label"}` while
    /// both engines require `{"text","expresses":bool}` — once the Python
    /// loader turned strict, data authored from the app's own prompt
    /// refused at load. The documented example must PARSE through the real
    /// loader, and the old schema must be gone (a Python twin feeds the
    /// same example line through `load_validation`).
    @Test func documentedValidationExampleParsesThroughTheRealLoader() throws {
        for intent in StudyIntent.allCases {
            let prompt = StudyCoauthoring.prompt(for: intent)
            if prompt.contains("validation.jsonl") {
                #expect(prompt.contains(#""expresses": true"#))
                #expect(!prompt.contains(#""label""#))
            }
        }
        // The example row, as the prompt teaches it, through the loader
        // evaluation uses.
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "doc-example-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try #"{"text": "The clerk hesitated at the counter.", "expresses": true}"#
            .appending("\n")
            .appending(#"{"text": "The bus arrived at seven.", "expresses": false}"#)
            .write(
                to: directory.appending(component: "validation.jsonl"),
                atomically: true, encoding: .utf8)
        let scenarios = try #require(
            try StimulusSet.loadValidation(directory: directory))
        #expect(scenarios.count == 2)
        #expect(scenarios[0].expresses == true)
        #expect(scenarios[1].expresses == false)
    }

    @Test func intentFiltersNeverOrphanContentSilently() {
        // agentComparison hides concept machinery — but a manifest CARRYING
        // concepts says so loudly.
        let hybrid = manifest {
            $0.concepts = [ExperimentManifest.ConceptRef(
                name: "fear", stimulusSetHash: "00",
                options: ExtractionOptions(method: .meanDifference))]
        }
        let noteText = StudyIntent.agentComparison.hiddenContentNote(for: hybrid)
        #expect(noteText?.contains("1 attached concept") == true)
        #expect(StudyIntent.agentComparison.hiddenContentNote(for: manifest()) == nil)
        // Pipeline stage relevance: an agent comparison chains run →
        // evaluate → analyze; the funnel stages belong to concept studies.
        #expect(StudyIntent.agentComparison.relevantPipelineStages
            == ["run", "evaluate", "analyze"])
        #expect(StudyIntent.conceptStudy.relevantPipelineStages
            == PipelineDraft.allStages)
        // Every readiness kind maps into exactly one authoring category.
        for kind in [DataRequirement.Kind.conceptStimuli, .conceptValidation,
                     .conceptMarkers, .taskPrompts, .judgeRubric, .judgePanel,
                     .humanBaseline, .multiAgentScenario, .capabilityBattery,
                     .neutralCorpus, .reasoningStyleTaxonomy]
        {
            _ = kind.authoringCategory  // exhaustive switch = compile proof
        }
    }

    @Test func studyJSONRoundTripsAndImportsAsDraftOnly() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "sajson") { _ in
            var original = manifest { $0.name = "sa-export" }
            original.judges = [
                .init(name: "or", kind: "openrouter",
                      model: "google/gemma-3-27b-it", provider: "DeepInfra")
            ]
            let json = try ExperimentStore.exportStudyJSON(original)
            // Round trip: reimporting under a fresh name preserves content.
            let renamed = json.replacingOccurrences(
                of: "\"sa-export\"", with: "\"sa-import\"")
            let (imported, violations, _) = try ExperimentStore.importStudyJSON(renamed)
            #expect(imported.name == "sa-import")
            #expect(imported.judges == original.judges)
            // verify() ran: an empty study's one honest complaint.
            #expect(violations == ["no concepts or variants attached"])
            // Saved and reloadable.
            #expect(try ExperimentStore.load(name: "sa-import").judges
                == original.judges)

            // A pasted "frozen" study CANNOT mint a preregistered object:
            // freeze metadata is stripped and the import is a draft.
            let frozenJSON = try ExperimentStore.exportStudyJSON(manifest {
                $0.name = "sa-frozen"
                $0.status = .frozen
                $0.freezeHash = "0"
                $0.frozenAt = "2026-07-18"
                $0.freezeForced = true
                $0.forcedGatesSkipped = ["revision"]
            })
            let (defrosted, _, _) = try ExperimentStore.importStudyJSON(frozenJSON)
            #expect(defrosted.status == .draft)
            #expect(defrosted.freezeHash == nil)
            #expect(defrosted.frozenAt == nil)
            #expect(defrosted.freezeForced == nil)

            // Name collisions refuse — never a silent overwrite.
            #expect(throws: ExperimentError.self) {
                _ = try ExperimentStore.importStudyJSON(renamed)
            }
            // Garbage refuses with the remedy named.
            #expect(throws: ExperimentError.self) {
                _ = try ExperimentStore.importStudyJSON("not json")
            }

            // A malformed-but-decodable study imports WITH its violations
            // surfaced (the firewall annotates, the researcher decides).
            let badJSON = try ExperimentStore.exportStudyJSON(manifest {
                $0.name = "sa-bad"
                $0.judges = [.init(name: "or", kind: "openrouter")]
            })
            let (_, badViolations, _) = try ExperimentStore.importStudyJSON(badJSON)
            #expect(badViolations.contains { $0.contains("no model slug") })
        }
    }
}
