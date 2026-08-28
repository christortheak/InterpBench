import CryptoKit
import Foundation
import SteeringKit
import Testing
@testable import ExperimentKit

/// Pre-freeze evidence tier: judge-rubric versioning (pinned rubric file +
/// judge panel + agreement stats) and capability-battery-as-evidence
/// (per-condition battery results in validation evidence, scope-hashed).
/// Extends the serialized `ExperimentStoreTests` suite for its temp-root
/// seam. Cross-engine JSON keys are pinned; the server side mirrors them.
extension ExperimentStoreTests {

    private func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Writes a rubric file under the temp root and returns (path, hash).
    private func plantRubric(
        _ relativePath: String = "prompts/rubrics/default-paired-v1.md",
        text: String = "Prefer the response the rubric describes."
    ) throws -> (file: String, hash: String) {
        let root = try #require(ExperimentStore.rootOverride)
        let url = root.appending(path: relativePath)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let data = Data(text.utf8)
        try data.write(to: url)
        return (relativePath, sha256Hex(data))
    }

    /// Minimal variant-only study under the temp root (no adapters or
    /// injections, so the variant-validity gate passes and the battery gate
    /// is what freeze exercises).
    @discardableResult
    private func makeBatteryVariantStudy(
        name: String = "bvs", variantName: String = "fear-lora"
    ) throws -> ExperimentManifest {
        let root = try #require(ExperimentStore.rootOverride)
        let artifact = ModelVariantArtifact(
            name: variantName,
            baseModelID: "test/model",
            adapters: [],
            injections: [],
            promptMode: "chatAssistant",
            qwenThinkingEnabled: false,
            temperature: 0,
            systemPrompt: "")
        let relativePath = "runs/model-variants/\(variantName)/model-variant.json"
        let artifactURL = root.appending(path: relativePath)
        try FileManager.default.createDirectory(
            at: artifactURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(artifact)
        try data.write(to: artifactURL)
        var manifest = try ExperimentStore.create(
            name: name, description: "", modelID: "test/model", modelRevision: "abc123")
        manifest.variantConditions = [
            .init(
                name: variantName, artifactPath: relativePath,
                artifactHash: sha256Hex(data), artifact: artifact)
        ]
        try ExperimentStore.save(manifest)
        return manifest
    }

    // MARK: - Manifest keys

    /// New evidence-tier keys round-trip, and nil fields are omitted from
    /// the encoding so every existing manifest keeps its content hash.
    @Test func evidenceTierFieldsRoundTripAndLegacyHashStable() throws {
        let plain = ExperimentManifest(name: "et", description: "", modelID: "test/model")
        let plainHash = ExperimentStore.manifestHash(plain)

        let encoder = JSONEncoder()
        let json = try #require(
            try JSONSerialization.jsonObject(with: encoder.encode(plain))
                as? [String: Any])
        for key in [
            "judgeRubricFile", "judgeRubricHash", "judges", "humanValidation",
            "capabilityBatteryFile", "capabilityBatteryHash",
        ] {
            #expect(json[key] == nil, "nil \(key) must be omitted from encoding")
        }

        var manifest = plain
        manifest.judgeRubricFile = "prompts/rubrics/default-paired-v1.md"
        manifest.judgeRubricHash = "rh"
        manifest.judges = [
            .init(name: "claude-judge", kind: "claude", model: nil),
            // Deliberately UNPINNED: this test round-trips the judge shape
            // and asserts nil fields are omitted from the JSON. It never
            // freezes, so the foreign-local-judge pin gate does not apply.
            .init(name: "local-judge", kind: "local", model: "test/judge"),
        ]
        manifest.humanValidation = .init(path: "prompts/human/subset.jsonl", hash: "hh")
        manifest.capabilityBatteryFile = "prompts/batteries/basic.jsonl"
        manifest.capabilityBatteryHash = "bh"
        let decoded = try JSONDecoder().decode(
            ExperimentManifest.self, from: encoder.encode(manifest))
        #expect(decoded == manifest)
        #expect(decoded.judges?.count == 2)
        #expect(decoded.judges?[0].kind == "claude")
        #expect(ExperimentStore.manifestHash(manifest) != plainHash)

        // Exact JSON keys of a judge entry (cross-engine contract).
        let judgeJSON = try #require(
            try JSONSerialization.jsonObject(
                with: encoder.encode(manifest.judges?[1]))
                as? [String: Any])
        #expect(Set(judgeJSON.keys) == Set(["name", "kind", "model"]))
    }

    // MARK: - verify() pins

    @Test func verifyFlagsRubricDriftMissingAndHalfPin() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "rub", description: "", modelID: "test/model")
            let rubric = try plantRubric()
            manifest.judgeRubricFile = rubric.file
            manifest.judgeRubricHash = rubric.hash
            #expect(!ExperimentStore.verify(manifest).contains { $0.contains("rubric") })

            // Drift after pinning.
            let root = try #require(ExperimentStore.rootOverride)
            try Data("EDITED".utf8).write(to: root.appending(path: rubric.file))
            var violations = ExperimentStore.verify(manifest)
            #expect(
                violations.contains {
                    $0.contains("judge rubric") && $0.contains("changed since pinning")
                }, "\(violations)")

            // Missing file.
            manifest.judgeRubricFile = "prompts/rubrics/nowhere.md"
            violations = ExperimentStore.verify(manifest)
            #expect(violations.contains { $0.contains("judge rubric missing") })

            // Half-pin: file without hash.
            manifest.judgeRubricFile = rubric.file
            manifest.judgeRubricHash = nil
            violations = ExperimentStore.verify(manifest)
            #expect(violations.contains { $0.contains("incompletely pinned") })
        }
    }

    @Test func pipelineBlockSurvivesASwiftRoundTrip() throws {
        // Chain-runner passthrough (server stage 3, 2026-07-18): Swift does
        // not interpret the `pipeline` block, but a decode → re-encode must
        // carry it VERBATIM — a closed CodingKeys enum silently destroying
        // a server-authored key is the JudgeRef.provider bug class.
        let json = #"""
            {"name": "chain", "experimentDescription": "d",
             "createdAt": "2026-07-18", "modelID": "org/m",
             "status": "draft",
             "pipeline": {
               "stages": ["extract", "validate", "sweep", "promote", "run"],
               "gates": {"validate": {"minScenarioAccuracy": 0.6,
                                       "maxCrossConceptCosine": 0.8},
                          "sweep": {"requireSelectionForEveryConcept": true}}}}
            """#
        let manifest = try JSONDecoder().decode(
            ExperimentManifest.self, from: Data(json.utf8))
        #expect(manifest.pipeline != nil)
        let encoded = try JSONEncoder().encode(manifest)
        let reloaded = try JSONDecoder().decode(
            ExperimentManifest.self, from: encoded)
        #expect(reloaded.pipeline == manifest.pipeline)
        guard case .object(let block)? = reloaded.pipeline,
            case .array(let stages)? = block["stages"]
        else {
            Issue.record("pipeline block lost its structure on round-trip")
            return
        }
        #expect(stages.count == 5)
        #expect(block["gates"] != nil)
        // A manifest WITHOUT the key stays byte-stable: nil is not encoded,
        // so pre-pipeline manifests keep their hash.
        var bare = manifest
        bare.pipeline = nil
        let bareObject = try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(bare)) as? [String: Any]
        #expect(bareObject?["pipeline"] == nil)
    }

    @Test func forwardReferencedConditionSurvivesAndVerifies() throws {
        // Stage 4 (server-resolved): a forward-referenced variant
        // condition decodes without artifact keys, re-encodes byte-
        // faithfully (ONE identity — no placeholder artifact), and
        // verify() checks the declaration, not the unminted artifact.
        try withTempRoot {
            let json = #"""
                {"name": "fr", "experimentDescription": "d",
                 "createdAt": "2026-07-18", "modelID": "test/model",
                 "status": "draft",
                 "variantConditions": [
                    {"name": "fear-agent",
                     "fromPromotion": {"concept": "fear"}}]}
                """#
            var manifest = try JSONDecoder().decode(
                ExperimentManifest.self, from: Data(json.utf8))
            manifest.concepts = [
                ExperimentManifest.ConceptRef(
                    name: "fear", stimulusSetHash: "00",
                    options: ExtractionOptions(method: .meanDifference))
            ]
            let (variant) = try #require(manifest.variantConditions.first)
            #expect(variant.fromPromotion?.concept == "fear")
            #expect(variant.artifactPath.isEmpty)
            // Round-trip: the encoded condition carries name +
            // fromPromotion ONLY — a fabricated empty artifact would make
            // the server flag "both identities declared".
            let encoded = try JSONSerialization.jsonObject(
                with: JSONEncoder().encode(manifest)) as? [String: Any]
            let conditions = encoded?["variantConditions"] as? [[String: Any]]
            #expect(conditions?.first?.keys.sorted()
                == ["fromPromotion", "name"])
            // verify(): attached concept → no variant violations; the
            // usual artifact checks (missing file, base-model) are exempt.
            #expect(!ExperimentStore.verify(manifest).contains {
                $0.contains("fear-agent")
            })
            // Unattached concept and both-identities each flag.
            var bad = manifest
            bad.variantConditions = [
                .init(name: "ghost-agent", artifactPath: "", artifactHash: "",
                      artifact: variant.artifact,
                      fromPromotion: .init(concept: "ghost")),
                .init(name: "double", artifactPath: "runs/x/agent.json",
                      artifactHash: "00", artifact: variant.artifact,
                      fromPromotion: .init(concept: "fear")),
            ]
            let violations = ExperimentStore.verify(bad)
            #expect(violations.contains {
                $0.contains("ghost-agent") && $0.contains("not attached")
            })
            #expect(violations.contains {
                $0.contains("double") && $0.contains("BOTH")
            })
        }
    }

    @Test func verifyFlagsBadJudgeEntriesAndHumanValidationDrift() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "jv", description: "", modelID: "test/model")
            manifest.judges = [
                .init(name: "weird", kind: "gpt", model: "x"),
                .init(name: "loc", kind: "local", model: nil),
                .init(name: "", kind: "claude", model: nil),
                .init(name: "or-bare", kind: "openrouter"),
                .init(name: "or-ok", kind: "openrouter",
                      model: "google/gemma-3-27b-it", provider: "DeepInfra"),
            ]
            let violations = ExperimentStore.verify(manifest)
            #expect(violations.contains { $0.contains("unknown kind 'gpt'") })
            // Cross-engine rule (2026-07-08): a local judge with no model is
            // LEGAL — it resolves to the study model. Never a violation.
            #expect(!violations.contains { $0.contains("'loc'") })
            #expect(violations.contains { $0.contains("empty name") })
            // OpenRouter judges have NO defaults (2026-07-19): a missing
            // model slug and a missing provider are each violations; a
            // fully pinned entry is clean.
            #expect(violations.contains {
                $0.contains("'or-bare'") && $0.contains("no model slug")
            })
            #expect(violations.contains {
                $0.contains("'or-bare'") && $0.contains("no pinned provider")
            })
            #expect(!violations.contains { $0.contains("'or-ok'") })

            // Human validation subset: drift and missing are violations.
            let root = try #require(ExperimentStore.rootOverride)
            let humanPath = "prompts/human/subset.jsonl"
            let humanURL = root.appending(path: humanPath)
            try FileManager.default.createDirectory(
                at: humanURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            // The unified row contract (2026-08-01): outcome ∈
            // baseline|variant|tie — verify parses the pinned rows with the
            // same parser evaluation uses, so a legacy-shaped row would be a
            // violation here, not merely quaint.
            let bytes = Data(
                #"{"condition":"c1","promptID":"p1","outcome":"variant"}"#.utf8)
            try bytes.write(to: humanURL)
            manifest.judges = nil
            manifest.humanValidation = .init(path: humanPath, hash: sha256Hex(bytes))
            #expect(
                !ExperimentStore.verify(manifest).contains { $0.contains("human validation") })
            manifest.humanValidation = .init(path: humanPath, hash: "not-the-hash")
            #expect(
                ExperimentStore.verify(manifest).contains {
                    $0.contains("human validation set") && $0.contains("changed since pinning")
                })
            manifest.humanValidation = .init(path: humanPath + ".missing", hash: "h")
            #expect(
                ExperimentStore.verify(manifest).contains {
                    $0.contains("human validation set missing")
                })
        }
    }

    @Test func verifyFlagsCapabilityBatteryDrift() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "bat", description: "", modelID: "test/model")
            let root = try #require(ExperimentStore.rootOverride)
            let batteryPath = "prompts/batteries/basic.jsonl"
            let batteryURL = root.appending(path: batteryPath)
            try FileManager.default.createDirectory(
                at: batteryURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            let bytes = Data(#"{"prompt":"2+2?","answer":"4"}"#.utf8)
            try bytes.write(to: batteryURL)

            manifest.capabilityBatteryFile = batteryPath
            manifest.capabilityBatteryHash = sha256Hex(bytes)
            #expect(
                !ExperimentStore.verify(manifest).contains {
                    $0.contains("capability battery")
                })

            manifest.capabilityBatteryHash = "not-the-hash"
            #expect(
                ExperimentStore.verify(manifest).contains {
                    $0.contains("capability battery") && $0.contains("changed since pinning")
                })

            manifest.capabilityBatteryFile = batteryPath + ".missing"
            #expect(
                ExperimentStore.verify(manifest).contains {
                    $0.contains("capability battery missing")
                })

            manifest.capabilityBatteryFile = nil
            #expect(
                ExperimentStore.verify(manifest).contains {
                    $0.contains("capability battery is incompletely pinned")
                })
        }
    }

    // MARK: - Validation scope

    /// The battery hash joins the validation scope ONLY when variant
    /// conditions exist; legacy and non-variant scope hashes are unchanged.
    @Test func batteryHashScopesValidationOnlyForVariantStudies() throws {
        var plain = ExperimentManifest(name: "s", description: "", modelID: "test/model")
        let baseScope = ExperimentStore.validationScopeHash(plain)
        plain.capabilityBatteryHash = "bh"
        #expect(ExperimentStore.validationScopeHash(plain) == baseScope)

        var variantStudy = plain
        let artifact = ModelVariantArtifact(
            name: "v", baseModelID: "test/model", adapters: [], injections: [],
            promptMode: "chatAssistant", qwenThinkingEnabled: false, temperature: 0,
            systemPrompt: "")
        variantStudy.variantConditions = [
            .init(name: "v", artifactPath: "p", artifactHash: "h", artifact: artifact)
        ]
        let withBattery = ExperimentStore.validationScopeHash(variantStudy)
        variantStudy.capabilityBatteryHash = "other-battery"
        #expect(ExperimentStore.validationScopeHash(variantStudy) != withBattery)
    }

    // MARK: - Freeze gates

    @Test func freezeRequiresBatteryEvidencePerVariantCondition() throws {
        try withTempRoot {
            let manifest = try makeBatteryVariantStudy()

            // No validate evidence at all → the battery gate names the need.
            do {
                _ = try ExperimentStore.freeze(name: "bvs")
                Issue.record("expected freeze to require battery evidence")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("validate"))
            }

            // Evidence WITHOUT battery results → violation names the
            // uncovered condition.
            try fabricateValidationEvidence(for: manifest)
            do {
                _ = try ExperimentStore.freeze(name: "bvs")
                Issue.record("expected freeze to name the uncovered condition")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("capability-battery"))
                #expect(error.reason.contains("fear-lora"))
            }

            // --force skips loudly, like the other evidence gates.
            let forced = try ExperimentStore.freeze(name: "bvs", force: true)
            #expect(forced.status == .frozen)
        }
    }

    @Test func freezePassesWithBatteryEvidenceCoveringAllConditions() throws {
        try withTempRoot {
            let manifest = try makeBatteryVariantStudy(name: "bvs2", variantName: "v2")
            try fabricateValidationEvidence(
                for: manifest,
                capabilityBattery: [
                    .init(
                        condition: "baseline", batteryHash: "bh", total: 12,
                        correct: 12, accuracy: 1),
                    .init(
                        condition: "v2", batteryHash: "bh", total: 12, correct: 11,
                        accuracy: 11.0 / 12.0),
                ])
            let frozen = try ExperimentStore.freeze(name: "bvs2")
            #expect(frozen.status == .frozen)
        }
    }

    /// The panel-size rule after the 2026-08-28 ruling: ONE judge is a legal
    /// design and freezes cleanly, ZERO is the invalid state the gate exists
    /// to refuse, and two or more is byte-identical to before.
    @Test func freezeRequiresPinnedRubricAndAtLeastOneJudge() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "jg", description: "", modelID: "test/model",
                modelRevision: "abc123def456")
            manifest.concepts.append(
                .init(name: "french", stimulusSetHash: try realFrenchHash(), options: .init()))
            manifest.evaluation = .init(
                kind: .pairedJudge, judgeModel: "test/judge",
                judgePrompt: "inline draft rubric")
            try ExperimentStore.save(manifest)
            try fabricateValidationEvidence(for: manifest)

            // Inline-only rubric → no freeze.
            do {
                _ = try ExperimentStore.freeze(name: "jg")
                Issue.record("expected freeze to require a pinned rubric file")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("rubric"))
                #expect(error.reason.contains("prompts/rubrics"))
            }

            // Pinned rubric, judged evaluation, and NO judge → the invalid
            // state: a judged instrument with no judge codes nothing.
            let rubric = try plantRubric()
            manifest.judgeRubricFile = rubric.file
            manifest.judgeRubricHash = rubric.hash
            manifest.judges = nil
            try ExperimentStore.save(manifest)
            try fabricateValidationEvidence(for: manifest)
            do {
                _ = try ExperimentStore.freeze(name: "jg")
                Issue.record("expected freeze to require at least one judge")
            } catch let error as ExperimentError {
                #expect(
                    error.reason
                        == "cannot freeze 'jg': "
                        + ExperimentStore.noJudgeDeclaredReason(
                            experimentName: "jg"))
            }

            // A SINGLE judge freezes — a single-coder design is legal — and
            // carries the advisory that says what it costs. Its own study,
            // because freezing 'jg' would seal it before the two-judge arm.
            var soloDraft = manifest
            soloDraft.name = "jg-solo"
            soloDraft.judges = [.init(name: "solo", kind: "claude", model: nil)]
            try ExperimentStore.save(soloDraft, allowCreate: true)
            try fabricateValidationEvidence(for: soloDraft)
            let solo = try ExperimentStore.freeze(name: "jg-solo")
            #expect(solo.status == .frozen)
            #expect(solo.freezeForced != true, "a legal design must not be forced")
            #expect(
                ExperimentStore.singleJudgePanelAdvisory(solo)
                    == ExperimentStore.singleJudgePanelAdvisoryText)
            #expect(
                ExperimentStore.freezeAdvisories(for: solo)
                    .contains(ExperimentStore.singleJudgePanelAdvisoryText))

            // Rubric + two judges → freezes; --force also works from the
            // unpinned state (loud skip, never silent).
            manifest.judges = [
                .init(name: "claude-judge", kind: "claude", model: nil),
                .init(name: "local-judge", kind: "local", model: "test/judge",
                  revision: "cafe01", dtype: "bfloat16"),
            ]
            try ExperimentStore.save(manifest)
            try fabricateValidationEvidence(for: manifest)
            let frozen = try ExperimentStore.freeze(name: "jg")
            #expect(frozen.status == .frozen)
        }
    }

    // MARK: - Judge-panel distinctness (finding 4, 2026-07-22)

    @Test func judgePanelIndistinctProblemResolvesIdentities() {
        var manifest = ExperimentManifest(
            name: "jp", description: "", modelID: "test/model")
        // Two blank-model local judges: both ARE the study model at
        // temperature 0 — the exact cross-engine wording.
        manifest.judges = [
            .init(name: "judge-1", kind: "local", model: nil),
            .init(name: "judge-2", kind: "local", model: ""),
        ]
        let problem = ExperimentStore.judgePanelIndistinctProblem(manifest)
        #expect(
            problem
                == "judges 'judge-1' and 'judge-2' both resolve to the same "
                + "deterministic judge (the study model at temperature 0) — "
                + "they would agree perfectly by construction; use judges "
                + "with different models, kinds, or providers")
        // An explicit study-model local + a blank local collapse the same way.
        manifest.judges = [
            .init(name: "a", kind: "local", model: "test/model"),
            .init(name: "b", kind: "local", model: nil),
        ]
        #expect(ExperimentStore.judgePanelIndistinctProblem(manifest) != nil)
        // Three collapsing judges read "all", not "both".
        manifest.judges = [
            .init(name: "a", kind: "local", model: nil),
            .init(name: "b", kind: "local", model: nil),
            .init(name: "c", kind: "local", model: nil),
        ]
        #expect(
            ExperimentStore.judgePanelIndistinctProblem(manifest)?
                .contains("'a', 'b' and 'c' all resolve") == true)
        // Two identical claude judges also collapse (generic wording).
        manifest.judges = [
            .init(name: "c1", kind: "claude", model: nil),
            .init(name: "c2", kind: "claude", model: ClaudePairedJudge.defaultModel),
        ]
        #expect(
            ExperimentStore.judgePanelIndistinctProblem(manifest)?
                .contains("the claude judge") == true)
        // Distinct panels pass: blank local + claude…
        manifest.judges = [
            .init(name: "judge-1", kind: "local", model: nil),
            .init(name: "claude", kind: "claude", model: nil),
        ]
        #expect(ExperimentStore.judgePanelIndistinctProblem(manifest) == nil)
        // …two distinct local models…
        manifest.judges = [
            .init(name: "judge-1", kind: "local", model: nil),
            .init(name: "judge-2", kind: "local", model: "other/model"),
        ]
        #expect(ExperimentStore.judgePanelIndistinctProblem(manifest) == nil)
        // …and same openrouter slug via different pinned providers.
        manifest.judges = [
            .init(name: "o1", kind: "openrouter", model: "org/slug", provider: "p1"),
            .init(name: "o2", kind: "openrouter", model: "org/slug", provider: "p2"),
        ]
        #expect(ExperimentStore.judgePanelIndistinctProblem(manifest) == nil)
        // A routing slug and OpenRouter's response display name identify the
        // same endpoint; they cannot masquerade as two independent judges.
        manifest.judges = [
            .init(
                name: "o1", kind: "openrouter",
                model: "google/gemini-3.6-flash",
                provider: "google-ai-studio"),
            .init(
                name: "o2", kind: "openrouter",
                model: "google/gemini-3.6-flash",
                provider: "Google AI Studio"),
        ]
        #expect(
            ExperimentStore.judgePanelIndistinctProblem(manifest)?
                .contains("openrouter judge") == true)
    }

    @Test func freezeRefusesIndistinctJudgePanelAndForceStampsJudgeValidity() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "jgi", description: "", modelID: "test/model",
                modelRevision: "abc123def456")
            manifest.concepts.append(
                .init(name: "french", stimulusSetHash: try realFrenchHash(), options: .init()))
            let rubric = try plantRubric()
            manifest.judgeRubricFile = rubric.file
            manifest.judgeRubricHash = rubric.hash
            manifest.judges = [
                .init(name: "judge-1", kind: "local", model: nil),
                .init(name: "judge-2", kind: "local", model: nil),
            ]
            try ExperimentStore.save(manifest)
            try fabricateValidationEvidence(for: manifest)

            // Two blank locals are ONE deterministic judge: refused, with
            // the cross-engine wording, and surfaced pre-freeze as an
            // advisory too.
            #expect(
                ExperimentStore.freezeAdvisories(for: manifest)
                    .contains { $0.contains("agree perfectly by construction") })
            do {
                _ = try ExperimentStore.freeze(name: "jgi")
                Issue.record("expected freeze to refuse an indistinct judge panel")
            } catch let error as ExperimentError {
                #expect(
                    error.reason.contains(
                        "judges 'judge-1' and 'judge-2' both resolve to the "
                            + "same deterministic judge (the study model at "
                            + "temperature 0)"))
            }

            // --force skips loudly and stamps the existing judgeValidity id.
            let forced = try ExperimentStore.freeze(name: "jgi", force: true)
            #expect(forced.freezeForced == true)
            #expect(forced.forcedGatesSkipped?.contains("judgeValidity") == true)
        }

        // A distinct panel (blank local + claude) freezes cleanly.
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "jgd", description: "", modelID: "test/model",
                modelRevision: "abc123def456")
            manifest.concepts.append(
                .init(name: "french", stimulusSetHash: try realFrenchHash(), options: .init()))
            let rubric = try plantRubric()
            manifest.judgeRubricFile = rubric.file
            manifest.judgeRubricHash = rubric.hash
            manifest.judges = [
                .init(name: "judge-1", kind: "local", model: nil),
                .init(name: "claude", kind: "claude", model: nil),
            ]
            try ExperimentStore.save(manifest)
            try fabricateValidationEvidence(for: manifest)
            let frozen = try ExperimentStore.freeze(name: "jgd")
            #expect(frozen.status == .frozen)
        }
    }

    // MARK: - Local judges in a judged pipeline (finding 1, 2026-07-23)

    @Test func localJudgePipelineProblemIsSweepOnlyAndEvaluateRoutes() {
        var manifest = ExperimentManifest(
            name: "lp", description: "", modelID: "test/model")
        manifest.judges = [
            .init(name: "judge-1", kind: "local", model: nil),
            .init(name: "judge-2", kind: "local", model: "other/judge-12b",
                  revision: "cafe01", dtype: "bfloat16"),
        ]
        // Evaluate stage: NEVER a gate — it routes to the server's
        // post-generation judge fan-out (2026-07-23).
        manifest.pipeline = .object(
            ["stages": .array([.string("run"), .string("evaluate")])])
        #expect(ExperimentStore.localJudgePipelineProblem(manifest) == nil)
        let note = ExperimentStore.localJudgeFanoutNote(manifest)
        #expect(note?.contains("post-generation judge fan-out") == true)
        #expect(note?.contains("'judge-2' (model 'other/judge-12b')") == true)
        // A judgeScore sweep stage still refuses (no fan-out exists for
        // sweep-interleaved judging).
        manifest.pipeline = .object(
            ["stages": .array([.string("sweep"), .string("run")])])
        manifest.sweep = .init(
            selection: .init(objective: .init(metric: "judgeScore")))
        let problem = ExperimentStore.localJudgePipelineProblem(manifest)
        #expect(problem?.contains("sweep stage holds ONE model") == true)
        #expect(problem?.contains("'judge-2' (model 'other/judge-12b')") == true)
        #expect(problem?.contains("fan-out covers the evaluate stage only") == true)
        // A markerDensity sweep does not judge — no problem.
        manifest.sweep = .init(
            selection: .init(objective: .init(metric: "markerDensity")))
        #expect(ExperimentStore.localJudgePipelineProblem(manifest) == nil)
        // No pipeline declared → neither problem nor note.
        manifest.pipeline = nil
        manifest.sweep = .init(
            selection: .init(objective: .init(metric: "judgeScore")))
        #expect(ExperimentStore.localJudgePipelineProblem(manifest) == nil)
        #expect(ExperimentStore.localJudgeFanoutNote(manifest) == nil)
        // A judged pipeline whose local judges all resolve to the study
        // model needs neither.
        manifest.pipeline = .object(
            ["stages": .array([.string("run"), .string("evaluate")])])
        manifest.judges = [
            .init(name: "judge-1", kind: "local", model: nil),
            .init(name: "claude", kind: "claude", model: nil),
        ]
        #expect(ExperimentStore.localJudgeFanoutNote(manifest) == nil)
    }

    @Test func freezeRefusesForeignLocalJudgeInJudgedSweepPipeline() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "lpg", description: "", modelID: "test/model",
                modelRevision: "abc123def456")
            manifest.concepts.append(
                .init(name: "french", stimulusSetHash: try realFrenchHash(), options: .init()))
            let rubric = try plantRubric()
            manifest.judgeRubricFile = rubric.file
            manifest.judgeRubricHash = rubric.hash
            // Sweep inputs must exist for freeze to pin them (the sweep
            // gate under test comes after).
            _ = try plantRubric(
                "prompts/dev/dev-prompts.jsonl",
                text: "{\"id\": \"d1\", \"prompt\": \"Describe the cellar.\"}\n")
            _ = try plantRubric(
                "prompts/batteries/basic.jsonl",
                text: "{\"id\": \"b1\", \"prompt\": \"2+2?\", \"answer\": \"4\"}\n")
            manifest.pipeline = .object(
                ["stages": .array([.string("sweep"), .string("run")])])
            manifest.sweep = .init(
                selection: .init(objective: .init(metric: "judgeScore")))
            manifest.judges = [
                .init(name: "judge-1", kind: "local", model: nil),
                .init(name: "judge-2", kind: "local", model: "other/judge-12b",
                  revision: "cafe01", dtype: "bfloat16"),
            ]
            try ExperimentStore.save(manifest)
            try fabricateValidationEvidence(for: manifest)
            #expect(
                ExperimentStore.freezeAdvisories(for: manifest)
                    .contains { $0.contains("sweep stage holds ONE model") })
            do {
                _ = try ExperimentStore.freeze(name: "lpg")
                Issue.record(
                    "expected freeze to refuse a judged-sweep pipeline whose local judge cannot load inside the chain")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("sweep stage holds ONE model"))
                #expect(error.reason.contains("other/judge-12b"))
            }
            let forced = try ExperimentStore.freeze(name: "lpg", force: true)
            #expect(forced.forcedGatesSkipped?.contains("judgeValidity") == true)
        }
    }

    @Test func evaluatePipelineWithForeignLocalJudgeFreezesWithFanoutNote() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "lpe", description: "", modelID: "test/model",
                modelRevision: "abc123def456")
            manifest.concepts.append(
                .init(name: "french", stimulusSetHash: try realFrenchHash(), options: .init()))
            let rubric = try plantRubric()
            manifest.judgeRubricFile = rubric.file
            manifest.judgeRubricHash = rubric.hash
            manifest.pipeline = .object(
                ["stages": .array([.string("run"), .string("evaluate")])])
            manifest.judges = [
                .init(name: "judge-1", kind: "local", model: nil),
                .init(name: "judge-2", kind: "local", model: "other/judge-12b",
                  revision: "cafe01", dtype: "bfloat16"),
            ]
            try ExperimentStore.save(manifest)
            try fabricateValidationEvidence(for: manifest)
            #expect(
                ExperimentStore.freezeAdvisories(for: manifest)
                    .contains { $0.contains("post-generation judge fan-out") })
            let frozen = try ExperimentStore.freeze(name: "lpe")
            #expect(frozen.status == .frozen)
        }
    }

    @Test func freezePinsBlankLocalJudgeRevisionFromStudyPin() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "lpr", description: "", modelID: "test/model",
                modelRevision: "abc123def456")
            manifest.concepts.append(
                .init(name: "french", stimulusSetHash: try realFrenchHash(), options: .init()))
            let rubric = try plantRubric()
            manifest.judgeRubricFile = rubric.file
            manifest.judgeRubricHash = rubric.hash
            manifest.judges = [
                .init(name: "judge-1", kind: "local", model: nil),
                .init(name: "claude", kind: "claude", model: nil),
                .init(
                    name: "pinned", kind: "local", model: "test/model",
                    revision: "beef06"),
            ]
            try ExperimentStore.save(manifest)
            try fabricateValidationEvidence(for: manifest)
            let frozen = try ExperimentStore.freeze(name: "lpr")
            let byName = Dictionary(
                uniqueKeysWithValues: (frozen.judges ?? []).map { ($0.name, $0) })
            // The blank local judge resolved to the study model: revision
            // pinned from the study pin (cross-engine key
            // judges[].revision).
            #expect(byName["judge-1"]?.revision == "abc123def456")
            // Claude judges gain no revision; a declared revision is never
            // overwritten.
            #expect(byName["claude"]?.revision == nil)
            #expect(byName["pinned"]?.revision == "beef06")
        }
    }

    @Test func judgeRefRevisionAndDtypeRoundTripOmitWhenNil() throws {
        let plain = ExperimentManifest.JudgeRef(
            name: "j", kind: "local", model: nil)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let plainJSON = String(
            decoding: try encoder.encode(plain), as: UTF8.self)
        #expect(!plainJSON.contains("revision"))
        #expect(!plainJSON.contains("dtype"))
        let pinned = ExperimentManifest.JudgeRef(
            name: "j", kind: "local", model: "m/x", revision: "r1",
            dtype: "bfloat16")
        let data = try encoder.encode(pinned)
        let decoded = try JSONDecoder().decode(
            ExperimentManifest.JudgeRef.self, from: data)
        #expect(decoded.revision == "r1")
        #expect(decoded.dtype == "bfloat16")
    }

    @Test func forceSkipsJudgeGateLoudly() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "jgf", description: "", modelID: "test/model",
                modelRevision: "abc123def456")
            manifest.concepts.append(
                .init(name: "french", stimulusSetHash: try realFrenchHash(), options: .init()))
            manifest.evaluation = .init(
                kind: .pairedJudge, judgeModel: "test/judge",
                judgePrompt: "inline draft rubric")
            try ExperimentStore.save(manifest)
            try fabricateValidationEvidence(for: manifest)
            // Judge-evaluated without a pinned rubric: gated…
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.freeze(name: "jgf")
            }
            // …and --force records it deliberately.
            let forced = try ExperimentStore.freeze(name: "jgf", force: true)
            #expect(forced.status == .frozen)
        }
    }

    // MARK: - Agreement plumbing (pure)

    private func judgeRecord(
        judge: String, condition: String, sampleIndex: UInt64, promptID: String,
        result: String
    ) -> ExperimentTasks.PairedJudgeRecord {
        ExperimentTasks.PairedJudgeRecord(
            experiment: "e", experimentHash: "h", sourceRunDirectory: "d",
            judgeName: judge, judgeKind: "claude", judgeModel: "m",
            judgePrompt: "r", judgeRubricFile: nil, judgeRubricHash: nil,
            structuredPrompt: nil, condition: condition,
            sampleIndex: sampleIndex, baselineSeed: 11, variantSeed: 22,
            promptID: promptID, prompt: "p", baselineWas: "A", conditionWas: "B",
            judgment: PairedJudgeResponse(
                aScores: nil, bScores: nil, structuredFields: nil,
                winner: "B", confidence: 0.9, briefReason: ""),
            conditionResult: result)
    }

    @Test func judgeAgreementPairsAndKappa() throws {
        // Two judges, three shared items: agree on two, disagree on one.
        let records = [
            judgeRecord(judge: "j1", condition: "c", sampleIndex: 1, promptID: "p1", result: "condition"),
            judgeRecord(judge: "j1", condition: "c", sampleIndex: 1, promptID: "p2", result: "baseline"),
            judgeRecord(judge: "j1", condition: "c", sampleIndex: 1, promptID: "p3", result: "tie"),
            judgeRecord(judge: "j2", condition: "c", sampleIndex: 1, promptID: "p1", result: "condition"),
            judgeRecord(judge: "j2", condition: "c", sampleIndex: 1, promptID: "p2", result: "baseline"),
            judgeRecord(judge: "j2", condition: "c", sampleIndex: 1, promptID: "p3", result: "condition"),
        ]
        let agreement = try #require(
            ExperimentTasks.judgeAgreement(records: records, judges: ["j1", "j2"]))
        #expect(agreement.count == 1)
        #expect(agreement[0].judgeA == "j1" && agreement[0].judgeB == "j2")
        #expect(agreement[0].items == 3)
        #expect(abs(agreement[0].percentAgreement - 2.0 / 3.0) < 1e-9)
        // Single judge → no pairwise section.
        #expect(ExperimentTasks.judgeAgreement(records: records, judges: ["j1"]) == nil)
    }

    @Test func humanAgreementMatchesWithAndWithoutSeed() throws {
        let records = [
            judgeRecord(judge: "j1", condition: "c", sampleIndex: 7, promptID: "p1", result: "condition"),
            judgeRecord(judge: "j1", condition: "c", sampleIndex: 7, promptID: "p2", result: "tie"),
        ]
        // Cross-engine row shape: "outcome" in baseline|variant|tie, where
        // "variant" maps onto this engine's "condition" result label.
        let human = try ExperimentTasks.parseHumanValidation(
            Data(
                """
                {"condition":"c","promptID":"p1","outcome":"variant"}
                {"condition":"c","promptID":"p2","seed":7,"outcome":"baseline"}
                """.utf8))
        let reports = ExperimentTasks.humanAgreement(
            records: records, judges: ["j1"], human: human)
        #expect(reports.count == 1)
        #expect(reports[0].items == 2)
        #expect(abs(reports[0].percentAgreement - 0.5) < 1e-9)

        // Malformed rows fail loudly (bad outcome label).
        #expect(throws: ExperimentError.self) {
            try ExperimentTasks.parseHumanValidation(
                Data(#"{"condition":"c","promptID":"p","outcome":"maybe"}"#.utf8))
        }
    }

    // MARK: - Canonical per-run config.json

    /// THE cross-engine contract (schema 4). The literal list is duplicated
    /// on purpose: adding/removing/renaming a top-level key must fail THIS
    /// test until `RunMetadata.schemaVersion` is bumped and this list (plus
    /// the Python twin in `Server/tests/test_run_config.py::CONTRACT_KEYS`)
    /// is updated in the same change. `notes` is the only escape hatch for
    /// engine-specific extras — never a new top-level key.
    @Test func runMetadataPayloadHasExactPinnedKeys() throws {
        let contractKeys = [
            "appVersion", "createdAt", "dtype", "experiment",
            "experimentHash", "jobId", "modelID", "notes", "platform",
            "pythonEnvironment", "revision", "runId", "runType",
            "samplesPerItem", "schemaVersion", "seedPolicy", "substrate",
            "temperature",
        ]
        let payload = RunMetadata.payload(
            runID: "19700101T000000000-exp-e-validate",
            runType: "validate", createdAt: Date(timeIntervalSince1970: 0),
            modelID: "test/model", revision: nil, experiment: "e",
            experimentHash: "eh", jobID: nil)
        #expect(payload.keys.sorted() == contractKeys)
        #expect(RunMetadata.contractKeys == contractKeys)
        #expect(payload["schemaVersion"] as? Int == 4)
        // Schema 3: this engine always writes null — MLX study models are
        // quantized repos with no single parameter dtype, and the Mac is a
        // testing substrate. Null means "not recorded", explicitly.
        #expect(payload["dtype"] is NSNull)
        // Schema 4: same engine-conditional shape — there is no Python
        // environment under Swift/MLX, so the key is present and null rather
        // than absent (the Python engine fills it with resolved versions).
        #expect(payload["pythonEnvironment"] is NSNull)
        #expect(payload["runId"] as? String == "19700101T000000000-exp-e-validate")
        #expect(payload["runType"] as? String == "validate")
        #expect(payload["createdAt"] as? String == "1970-01-01T00:00:00Z")
        #expect(payload["substrate"] as? String == RepEReader.substrate)
        #expect(payload["modelID"] as? String == "test/model")
        #expect(payload["revision"] is NSNull, "absent values encode as JSON null")
        #expect(payload["experiment"] as? String == "e")
        #expect(payload["experimentHash"] as? String == "eh")
        #expect(payload["temperature"] is NSNull)
        #expect(payload["samplesPerItem"] is NSNull)
        #expect(payload["seedPolicy"] is NSNull)
        #expect(payload["jobId"] is NSNull)
        #expect((payload["notes"] as? [String: String]) == [:])
        #expect(payload["appVersion"] as? String == SteerLabVersion.current)
        #expect(
            (payload["appVersion"] as? String)?.hasPrefix("swift-app ") == true,
            "engine version stamp format is 'swift-app <version>[+shortSHA]'")
        // platform is OS + architecture only (e.g. "macOS-arm64") — never a
        // hostname.
        #expect(payload["platform"] as? String == RunMetadata.platform)
        #expect(
            RunMetadata.platform.hasPrefix("macOS-")
                || RunMetadata.platform.hasPrefix("linux-"))
    }

    /// Two run-directory writers verified end to end: the experiment task
    /// path (`makeRunDirectory`) and the robustness report writer.
    @Test func runDirectoryWritersStampConfigJSON() throws {
        try withTempRoot {
            var manifest = try ExperimentStore.create(
                name: "cfg", description: "", modelID: "test/model",
                modelRevision: "abc123")
            manifest.status = .draft
            let runDirectory = try ExperimentTasks.makeRunDirectory(
                experiment: manifest, task: "validate")
            let configURL = runDirectory.appending(component: "config.json")
            let config = try #require(
                try JSONSerialization.jsonObject(
                    with: Data(contentsOf: configURL)) as? [String: Any])
            #expect(config["runType"] as? String == "validate")
            #expect(config["runId"] as? String == runDirectory.lastPathComponent)
            #expect(config["modelID"] as? String == "test/model")
            #expect(config["revision"] as? String == "abc123")
            #expect(config["experiment"] as? String == "cfg")
            #expect(
                config["experimentHash"] as? String
                    == ExperimentStore.manifestHash(manifest))
            #expect(config["substrate"] as? String == "swift-mlx")
            // validate does not sample by the manifest's policy → nulls.
            #expect(config["temperature"] is NSNull)
            #expect(config["samplesPerItem"] is NSNull)
            #expect(config["seedPolicy"] is NSNull)

            // Multi-agent task name maps to the pinned "multi-agent" type,
            // and study tasks stamp the manifest's sampling policy (defaults
            // mirror the server: samplesPerItem 1, seedPolicy manifestSeeds).
            let multiAgentDir = try ExperimentTasks.makeRunDirectory(
                experiment: manifest, task: "multi-agent-run")
            let multiAgentConfig = try #require(
                try JSONSerialization.jsonObject(
                    with: Data(
                        contentsOf: multiAgentDir.appending(component: "config.json")))
                    as? [String: Any])
            #expect(multiAgentConfig["runType"] as? String == "multi-agent")
            #expect(multiAgentConfig["temperature"] as? Double == manifest.temperature)
            #expect(multiAgentConfig["samplesPerItem"] as? Int == 1)
            #expect(multiAgentConfig["seedPolicy"] as? String == "manifestSeeds")

            manifest.temperature = 0.7
            manifest.samplesPerItem = 3
            manifest.seedPolicy = "derivedSHA256"
            let studyDir = try ExperimentTasks.makeRunDirectory(
                experiment: manifest, task: "run")
            let studyConfig = try #require(
                try JSONSerialization.jsonObject(
                    with: Data(
                        contentsOf: studyDir.appending(component: "config.json")))
                    as? [String: Any])
            #expect(studyConfig["runType"] as? String == "run")
            #expect(studyConfig["temperature"] as? Double == 0.7)
            #expect(studyConfig["samplesPerItem"] as? Int == 3)
            #expect(studyConfig["seedPolicy"] as? String == "derivedSHA256")
        }

        // Writer 2: the robustness report writer (hermetic — explicit dir).
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "robust-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temp) }
        let report = VariantRobustnessReport(
            variantName: "v", variantArtifactPath: nil, variantArtifactHash: nil,
            baseModelID: "test/model", substrate: "swift-mlx", presetID: "quick",
            batteryFile: "b", coherencePromptsFile: "c", judgeModel: nil,
            generatedAt: "2026-07-06T00:00:00Z", baselineBatteryAccuracy: 1,
            variantBatteryAccuracy: 1, meanBaselineDistinct2: 1,
            meanVariantDistinct2: 1, meanBaselineWords: 1, meanVariantWords: 1,
            batteryItems: [], coherenceItems: [], warnings: [])
        try VariantRobustness.write(report, to: temp)
        let config = try #require(
            try JSONSerialization.jsonObject(
                with: Data(contentsOf: temp.appending(component: "config.json")))
                as? [String: Any])
        #expect(config["runType"] as? String == "variant-robustness")
        #expect(config["modelID"] as? String == "test/model")
        #expect(config["experiment"] is NSNull)
    }
}
