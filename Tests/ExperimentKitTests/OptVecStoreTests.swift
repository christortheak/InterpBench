import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// OptVec v1 surface: bundle store decode + the data-check mirror of
/// `steerlab-server data check optvec`, the run store's offline readings,
/// and the Swift half of the attach-artifact contract (af1af0e — pinned
/// bytes as a `pinnedArtifact` concept). Serialized: every test moves the
/// process-global workspace root through the shared override lock.
@Suite(.serialized) struct OptVecStoreTests {

    func withTempRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "optvec", body)
    }

    // MARK: - Fixture planting

    /// A tiny VALID bundle: every choice file balanced 1/1 (A/B), ids
    /// unique bundle-wide, neutral file to spec, bundle.json hash-true.
    @discardableResult
    private func plantBundle(
        named name: String = "test-1", root: URL,
        mutate: (inout [String: String]) -> Void = { _ in }
    ) throws -> URL {
        let directory = root.appending(components: "prompts", "optvec", name)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        var files: [String: String] = [:]
        for file in OptVecBundleStore.choiceFiles {
            let stem = file.replacingOccurrences(of: ".jsonl", with: "")
            files[file] = """
            {"id": "ovt-\(name)-\(stem)-001", "text": "Case one. Answer with exactly one letter: A or B.", "options": ["A", "B"], "target": "A"}
            {"id": "ovt-\(name)-\(stem)-002", "text": "Case two. Answer with exactly one letter: A or B.", "options": ["A", "B"], "target": "B"}

            """
        }
        files[OptVecBundleStore.neutralFile] = """
        {"text": "A plain paragraph about tide pools."}
        {"text": "A plain paragraph about typefaces."}

        """
        mutate(&files)
        for (file, contents) in files {
            try contents.write(
                to: directory.appending(component: file),
                atomically: true, encoding: .utf8)
        }
        var pins: [String: [String: String]] = [:]
        for file in OptVecBundleStore.bundleFiles {
            let data = try Data(
                contentsOf: directory.appending(component: file))
            pins[file] = [
                "path": file,
                "sha256": OptVecBundleStore.sha256Hex(data),
                "reader": "optimizer",
            ]
        }
        let payload: [String: Any] = [
            "bundle": name,
            "targetIssue": "rule vs equity; shift toward equity",
            "shiftDirection": "toward equity",
            "caseFamilies": ["filing-deadline"],
            "anchorIssues": ["settled-evidence"],
            "files": pins,
        ]
        let json = try JSONSerialization.data(
            withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
        try json.write(
            to: directory.appending(
                component: OptVecBundleStore.bundleJSON))
        return directory
    }

    private func requirement(
        _ report: OptVecBundleStore.Report, _ name: String
    ) -> OptVecBundleStore.Requirement? {
        report.requirements.first { $0.name == name }
    }

    // MARK: - Bundle listing and decode

    @Test func listDecodesBundleJSONAndSurvivesMalformation() throws {
        try withTempRoot { root in
            try plantBundle(named: "rve-mini", root: root)
            // A second folder with an undecodable bundle.json still LISTS.
            let broken = root.appending(
                components: "prompts", "optvec", "broken")
            try FileManager.default.createDirectory(
                at: broken, withIntermediateDirectories: true)
            try "not json".write(
                to: broken.appending(component: "bundle.json"),
                atomically: true, encoding: .utf8)

            let entries = OptVecBundleStore.list()
            #expect(entries.map(\.name) == ["broken", "rve-mini"])
            let bundle = try #require(
                entries.first { $0.name == "rve-mini" }?.bundle)
            #expect(bundle.bundle == "rve-mini")
            #expect(bundle.targetIssue?.lines
                == ["rule vs equity; shift toward equity"])
            #expect(bundle.targetIssue?.wasScalar == true)
            #expect(bundle.caseFamilies?.lines == ["filing-deadline"])
            #expect(bundle.filesByBasename["target-train.jsonl"]?.reader
                == "optimizer")
            #expect(
                entries.first { $0.name == "broken" }?.decodeFailure != nil)
        }
    }

    @Test func missingOptvecDirectoryListsEmpty() throws {
        try withTempRoot { _ in
            #expect(OptVecBundleStore.list().isEmpty)
        }
    }

    // MARK: - Data-check mirror

    @Test func validBundlePassesAllTwelveRequirements() throws {
        try withTempRoot { root in
            let directory = try plantBundle(root: root)
            let report = OptVecBundleStore.check(directory: directory)
            // 9 data files + bundle.json + REPORT.md + bundle ids.
            #expect(report.requirements.count == 12)
            // REPORT.md was not planted: partial, non-blocking — so the
            // bundle is still ready.
            #expect(requirement(report, "REPORT.md")?.status == .partial)
            #expect(report.ready)
            #expect(report.blockers.isEmpty)
            let toc = try #require(
                requirement(report, OptVecBundleStore.bundleJSON))
            #expect(toc.status == .present)
            #expect(toc.detail.contains("hashes agree"))
            // Every passing file gets a paste-able hash; the target file's
            // agrees with its bytes.
            let target = try #require(
                requirement(report, "target-train.jsonl"))
            #expect(target.status == .present)
            #expect(target.rows == 2)
            let bytes = try Data(
                contentsOf: directory.appending(
                    component: "target-train.jsonl"))
            #expect(target.sha256 == OptVecBundleStore.sha256Hex(bytes))
            #expect(
                requirement(report, OptVecBundleStore.bundleIDsRequirement)?
                    .status == .present)
        }
    }

    @Test func staleHashInBundleJSONIsABlocker() throws {
        try withTempRoot { root in
            let directory = try plantBundle(root: root)
            // Edit a file AFTER pinning — the drift bundle.json exists to
            // prevent.
            try """
            {"id": "x-1", "text": "Edited. Answer with exactly one letter: A or B.", "options": ["A", "B"], "target": "A"}
            {"id": "x-2", "text": "Edited too. Answer with exactly one letter: A or B.", "options": ["A", "B"], "target": "B"}

            """.write(
                to: directory.appending(component: "anchor-val.jsonl"),
                atomically: true, encoding: .utf8)
            let report = OptVecBundleStore.check(directory: directory)
            let toc = try #require(
                requirement(report, OptVecBundleStore.bundleJSON))
            #expect(toc.status == .invalid)
            #expect(toc.detail.contains("pinned hash disagrees"))
            #expect(toc.detail.contains("anchor-val.jsonl"))
            #expect(!report.ready)
            // Blockers sort first — the server's ordering.
            #expect(report.requirements.first?.blocker == true)
        }
    }

    @Test func missingFileAndDirectiveBlock() throws {
        try withTempRoot { root in
            let directory = try plantBundle(root: root)
            try FileManager.default.removeItem(
                at: directory.appending(component: "capability-eval.jsonl"))
            // Blank a directive too.
            var payload = try JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appending(
                    component: "bundle.json"))) as! [String: Any]
            payload["shiftDirection"] = "   "
            try JSONSerialization.data(withJSONObject: payload)
                .write(to: directory.appending(component: "bundle.json"))

            let report = OptVecBundleStore.check(directory: directory)
            #expect(
                requirement(report, "capability-eval.jsonl")?.status
                    == .missing)
            let toc = try #require(
                requirement(report, OptVecBundleStore.bundleJSON))
            #expect(toc.status == .invalid)
            #expect(toc.detail.contains("shiftDirection"))
            // With a choice file unparsed, id uniqueness is only partial.
            #expect(
                requirement(report, OptVecBundleStore.bundleIDsRequirement)?
                    .status == .partial)
            #expect(!report.ready)
        }
    }

    @Test func crossFileDuplicateIDsAndMultiCharOptionsBlock() throws {
        try withTempRoot { root in
            let directory = try plantBundle(root: root) { files in
                // target-val reuses target-train's ids → cross-file dup.
                files["target-val.jsonl"] = files["target-train.jsonl"]!
                    .replacingOccurrences(
                        of: "target-train-00", with: "target-train-00")
                // capability-train gets a word option.
                files["capability-train.jsonl"] = """
                {"id": "cap-1", "text": "Pick. Answer with exactly one letter: A or B.", "options": ["(A)", "B"], "target": "B"}
                {"id": "cap-2", "text": "Pick. Answer with exactly one letter: A or B.", "options": ["A", "B"], "target": "A"}

                """
            }
            // Re-pin hashes so bundle.json is not the failing check here.
            var pins: [String: [String: String]] = [:]
            for file in OptVecBundleStore.bundleFiles {
                let data = try Data(
                    contentsOf: directory.appending(component: file))
                pins[file] = [
                    "path": file, "sha256": OptVecBundleStore.sha256Hex(data),
                ]
            }
            var payload = try JSONSerialization.jsonObject(
                with: Data(contentsOf: directory.appending(
                    component: "bundle.json"))) as! [String: Any]
            payload["files"] = pins
            try JSONSerialization.data(withJSONObject: payload)
                .write(to: directory.appending(component: "bundle.json"))

            let report = OptVecBundleStore.check(directory: directory)
            let ids = try #require(
                requirement(report, OptVecBundleStore.bundleIDsRequirement))
            #expect(ids.status == .invalid)
            #expect(ids.detail.contains("duplicated across files"))
            let capability = try #require(
                requirement(report, "capability-train.jsonl"))
            #expect(capability.status == .invalid)
            #expect(capability.detail.contains("not a single character"))
            // The failed file hands out no hash.
            #expect(capability.sha256 == nil)
        }
    }

    @Test func unbalancedTargetsBlock() throws {
        try withTempRoot { root in
            let directory = try plantBundle(root: root) { files in
                files["anchor-test.jsonl"] = """
                {"id": "ab-1", "text": "One. Answer with exactly one letter: A or B.", "options": ["A", "B"], "target": "A"}
                {"id": "ab-2", "text": "Two. Answer with exactly one letter: A or B.", "options": ["A", "B"], "target": "A"}
                {"id": "ab-3", "text": "Three. Answer with exactly one letter: A or B.", "options": ["A", "B"], "target": "A"}

                """
            }
            let report = OptVecBundleStore.check(directory: directory)
            let anchor = try #require(
                requirement(report, "anchor-test.jsonl"))
            #expect(anchor.status == .invalid)
            #expect(anchor.detail.contains("100.0%"))
            #expect(anchor.detail.contains("balance window"))
        }
    }

    // MARK: - Run store

    private func plantRun(
        named name: String, root: URL, runType: String,
        notes: [String: Any] = [:], files: [String: String] = [:]
    ) throws -> URL {
        let directory = root.appending(components: "runs", name)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let config: [String: Any] = [
            "schemaVersion": 3, "runId": name, "runType": runType,
            "createdAt": "2026-08-10T12:00:00Z", "substrate": "python-hf",
            "modelID": "test/model", "notes": notes,
        ]
        try JSONSerialization.data(withJSONObject: config)
            .write(to: directory.appending(component: "config.json"))
        for (file, contents) in files {
            try contents.write(
                to: directory.appending(component: file),
                atomically: true, encoding: .utf8)
        }
        return directory
    }

    @Test func runListFindsOptvecRunTypesOnly() throws {
        try withTempRoot { root in
            _ = try plantRun(
                named: "20260810T130000000-optvec-a", root: root,
                runType: "optvec-train", notes: ["stage": "optimizing"])
            _ = try plantRun(
                named: "20260810T120000000-optvec-eval-a", root: root,
                runType: "optvec-eval")
            _ = try plantRun(
                named: "20260810T110000000-study", root: root,
                runType: "study")
            let runs = OptVecRunStore.list()
            #expect(runs.map(\.kind) == [.train, .eval])
            #expect(runs.first?.stage == "optimizing")
        }
    }

    @Test func trainProgressReadsStageMetricsTailAndArtifact() throws {
        try withTempRoot { root in
            let sidecar: [String: Any] = [
                "modelID": "test/model", "concept": "rve-L31",
                "stimulusSetHash": "optvec:abc", "layerCount": 4,
                "hiddenSize": 8, "normsPerLayer": [1.0, 1.0, 1.0, 1.0],
                "extractionDate": "2026-08-10", "extractionMethod": "optvec",
            ]
            let directory = try plantRun(
                named: "20260810T130000000-optvec-rve", root: root,
                runType: "optvec-train", notes: ["stage": "complete"],
                files: [
                    "metrics.jsonl": """
                    {"step": 1, "loss": 3.0}
                    {"step": 2, "loss": 2.5, "valShiftRate": 0.4, "valComposite": 0.4}

                    """,
                    "rve-L31.safetensors": "fake-tensor-bytes",
                ])
            try JSONSerialization.data(withJSONObject: sidecar)
                .write(to: directory.appending(component: "rve-L31.json"))

            let progress = OptVecRunStore.trainProgress(runURL: directory)
            #expect(progress.stage == "complete")
            #expect(progress.lastMetrics?.step == 2)
            #expect(progress.lastMetrics?.valShiftRate == 0.4)
            #expect(
                progress.artifactReference
                    == "runs/20260810T130000000-optvec-rve/rve-L31")
        }
    }

    @Test func campaignStatusDerivesOffline() throws {
        try withTempRoot { root in
            let directory = try plantRun(
                named: "20260810T130000000-optvec-campaign-rve", root: root,
                runType: "optvec-campaign")
            let plan: [String: Any] = [
                "schemaVersion": 1, "name": "rve",
                "config": ["slurm": ["maxResubmits": 1]],
                "cells": [
                    ["cellID": "s2-L31-s0", "condition": "s2", "layer": 31,
                     "seed": 0],
                    ["cellID": "s2-L31-s1", "condition": "s2", "layer": 31,
                     "seed": 1],
                    ["cellID": "s1-L31-s0", "condition": "s1", "layer": 31,
                     "seed": 0],
                    ["cellID": "s0-L31-s0", "condition": "s0", "layer": 31,
                     "seed": 0],
                ],
            ]
            try JSONSerialization.data(withJSONObject: plan)
                .write(to: directory.appending(component: "campaign.json"))
            let state: [String: Any] = [
                "schemaVersion": 1,
                "cells": [
                    "s2-L31-s0": [
                        "attempts": [["jobID": "101", "outcome": "submitted"]],
                        "lastJobID": "101",
                    ],
                    "s2-L31-s1": [
                        "attempts": [
                            ["outcome": "failed", "exitCode": 1],
                            ["outcome": "failed", "exitCode": 1],
                        ]
                    ],
                    "s1-L31-s0": [
                        "attempts": [["jobID": "103", "outcome": "submitted"]],
                        "lastJobID": "103",
                    ],
                ],
            ]
            try JSONSerialization.data(withJSONObject: state)
                .write(
                    to: directory.appending(
                        component: "campaign-state.json"))
            // s1-L31-s0 finished: the COMPLETED marker is the authority.
            let cellDirectory = directory.appending(
                components: "cells", "s1-L31-s0")
            try FileManager.default.createDirectory(
                at: cellDirectory, withIntermediateDirectories: true)
            try Data().write(
                to: cellDirectory.appending(component: "COMPLETED"))

            let status = try #require(
                OptVecRunStore.campaignStatus(runURL: directory))
            let byID = Dictionary(
                uniqueKeysWithValues: status.cells.map { ($0.id, $0) })
            #expect(byID["s2-L31-s0"]?.status == .submitted)
            // budget = 1 + maxResubmits(1) = 2 attempts → exhausted.
            #expect(byID["s2-L31-s1"]?.status == .exhausted)
            #expect(byID["s1-L31-s0"]?.status == .completed)
            #expect(byID["s0-L31-s0"]?.status == .planned)
            #expect(status.totals[.completed] == 1)
        }
    }

    @Test func evalReportDecodesDoseRowsAndSkippedShapes() throws {
        try withTempRoot { root in
            let eval: [String: Any] = [
                "schemaVersion": 1, "runType": "optvec-eval",
                "claim": "sufficiency", "split": "test",
                "firewall": "test split only: …",
                "artifact": ["reference": "runs/x/rve-L31", "name": "rve-L31"],
                "alphaMultiples": [1.0],
                "doseResponse": [
                    [
                        "alphaMultiple": 0.0, "isBaseline": true,
                        "target": ["itemCount": 30],
                        "anchor": ["itemCount": 0],
                        "fluency": [
                            "textCount": 0, "scoredTextCount": 0,
                            "meanTokenLogprob": NSNull(),
                            "note": "no neutralTexts declared",
                        ],
                    ],
                    [
                        "alphaMultiple": 1.0, "alphaAbsolute": 4.7,
                        "isBaseline": false,
                        "target": [
                            "itemCount": 30, "shiftRate": 0.57,
                            "meanLogOddsMovement": 1.92,
                        ],
                        "anchor": [
                            "itemCount": 24, "flipRate": 0.04,
                            "meanKLFromBaseline": 0.02,
                        ],
                        "capability": [
                            "itemCount": 60, "accuracy": 0.93,
                            "accuracyDelta": -0.02,
                        ],
                    ],
                ],
                "library": [
                    "layer": 31, "comparedCount": 2,
                    "topK": [
                        [
                            "reference": "runs/y/crit", "concept": "crit",
                            "cosine": 0.31, "nullPercentile": 99.8,
                        ]
                    ],
                ],
            ]
            let directory = try plantRun(
                named: "20260810T140000000-optvec-eval-rve", root: root,
                runType: "optvec-eval")
            try JSONSerialization.data(withJSONObject: eval)
                .write(to: directory.appending(component: "eval.json"))

            let report = try #require(
                OptVecRunStore.evalReport(runURL: directory))
            #expect(report.claim == "sufficiency")
            #expect(report.doseResponse?.count == 2)
            #expect(report.doseResponse?[0].fluency?.note != nil)
            #expect(report.doseResponse?[1].target?.shiftRate == 0.57)
            #expect(report.library?.topK?.first?.nullPercentile == 99.8)
        }
    }

    // MARK: - Attach-artifact (the Swift half of the af1af0e contract)

    /// A saved OptVec artifact pair in a fake training run dir; returns the
    /// workspace-relative extension-less reference.
    private func plantOptvecArtifact(
        root: URL, name: String = "rve-L31",
        substrate: String? = "python-hf-transformers",
        residualNormSource: String? = "neutral-corpus",
        optvecBlock: [String: Any]? = [
            "layer": 31, "seed": 0,
            "runID": "20260810T130000000-optvec-rve",
            "evalRun": "20260810T140000000-optvec-eval-rve",
        ],
        extractionMethod: String = "optvec",
        modelID: String = "test/model",
        rendering: [String: Any]? = nil
    ) throws -> String {
        let runDirectory = root.appending(
            components: "runs", "20260810T130000000-optvec-rve")
        try FileManager.default.createDirectory(
            at: runDirectory, withIntermediateDirectories: true)
        try Data("fake-tensor".utf8).write(
            to: runDirectory.appending(component: "\(name).safetensors"))
        var sidecar: [String: Any] = [
            "modelID": modelID, "concept": name,
            "stimulusSetHash": "optvec:composite-abc", "layerCount": 4,
            "hiddenSize": 8, "normsPerLayer": [1.0, 1.0, 1.0, 1.0],
            "extractionDate": "2026-08-10",
            "extractionMethod": extractionMethod,
            "readingPosition": "last token",
            "neutralCorpusHash": "beefcafe",
        ]
        if let substrate { sidecar["substrate"] = substrate }
        if let residualNormSource {
            sidecar["residualNormSource"] = residualNormSource
        }
        if let optvecBlock { sidecar["optvec"] = optvecBlock }
        if let rendering { sidecar["extractionRendering"] = rendering }
        try JSONSerialization.data(withJSONObject: sidecar)
            .write(to: runDirectory.appending(component: "\(name).json"))
        // Attach verifies the eval-run citation since 2026-08-10: when the
        // provenance block names one, plant the run directory with an
        // eval.json certifying the tensor bytes written above.
        if let optvecBlock,
            let evalRun = ExperimentStore.recordedOptvecEvalRun(optvecBlock)
        {
            try plantCertifyingEvalRun(
                root: root, named: URL(filePath: evalRun).lastPathComponent)
        }
        return "runs/20260810T130000000-optvec-rve/\(name)"
    }

    /// An eval run directory whose eval.json certifies the helper's
    /// "fake-tensor" bytes (or an explicit foreign hash, for the mismatch
    /// case).
    private func plantCertifyingEvalRun(
        root: URL, named: String, tensorHash: String? = nil
    ) throws {
        let evalDir = root.appending(components: "runs", named)
        try FileManager.default.createDirectory(
            at: evalDir, withIntermediateDirectories: true)
        let hash =
            tensorHash
            ?? OptVecBundleStore.sha256Hex(Data("fake-tensor".utf8))
        try JSONSerialization.data(
            withJSONObject: ["artifact": ["tensorSHA256": hash]]
        ).write(to: evalDir.appending(component: "eval.json"))
    }

    @Test func attachOptvecArtifactPinsBytesAndProvenance() throws {
        try withTempRoot { root in
            try WorkspaceCompute.declare(.cluster, root: root)
            _ = try ExperimentStore.create(
                name: "optvec-confirm", description: "",
                modelID: "test/model")
            let reference = try plantOptvecArtifact(root: root)

            let manifest = try ExperimentStore.attachArtifact(
                "rve", artifact: reference,
                experimentName: "optvec-confirm")
            let ref = try #require(
                manifest.concepts.first { $0.name == "rve" })
            #expect(ref.options.method == .pinnedArtifact)
            // The composite dataset hash travels VERBATIM.
            #expect(ref.stimulusSetHash == "optvec:composite-abc")
            // validationHash pinned EXPLICITLY null — not legacy-absent.
            #expect(ref.validationHash == nil)
            #expect(ref.validationHashPinnedAbsent)
            let pin = try #require(ref.vectorArtifact)
            #expect(pin.path == reference)
            #expect(pin.sourceMethod == "optvec")
            #expect(pin.sourceConcept == "rve")
            #expect(pin.residualNormSource == "neutral-corpus")
            #expect(pin.normCorpusHash == "beefcafe")
            #expect(pin.optvecLayer == 31)
            #expect(pin.optvecSeed == 0)
            #expect(pin.optvecTrainingRun == "20260810T130000000-optvec-rve")
            #expect(pin.optvecEvalRun == "20260810T140000000-optvec-eval-rve")
            // The citation was resolved and hash-checked, not trusted by
            // name.
            #expect(pin.optvecEvalRunVerified == true)
            #expect(pin.optvecEvalRunUnverifiedReason == nil)
            let tensorBytes = try Data(
                contentsOf: root.appending(
                    path: "\(reference).safetensors"))
            #expect(
                pin.sha256TensorHash
                    == OptVecBundleStore.sha256Hex(tensorBytes))
            #expect(ref.effectiveMethod == .optvec)
            #expect(ref.dataConcept == "rve")
            // The pin passes verify the moment it is written.
            #expect(ExperimentStore.verify(manifest).isEmpty)
            // Round-trip through disk keeps the pin (custom Codable).
            let reloaded = try #require(
                ExperimentStore.list().first {
                    $0.name == "optvec-confirm"
                })
            #expect(reloaded.concepts.first?.vectorArtifact == pin)
            #expect(reloaded.concepts.first?.validationHashPinnedAbsent == true)
        }
    }

    /// The RENDERING is recipe identity exactly as the reading position is,
    /// so it travels from the sidecar at attach and a manifest that
    /// contradicts the artifact's own stamp is a verify violation. Compared
    /// CANONICALLY: a legacy artifact carries no stamp at all, so an absent
    /// AND an explicitly raw declaration both stay clean. Server twin:
    /// `test_extraction_rendering_disagreement_is_a_violation`.
    @Test func extractionRenderingContradictionIsAViolation() throws {
        try withTempRoot { root in
            try WorkspaceCompute.declare(.cluster, root: root)
            _ = try ExperimentStore.create(
                name: "rendering-pin", description: "", modelID: "test/model")

            let legacy = try plantOptvecArtifact(root: root, name: "legacy")
            var manifest = try ExperimentStore.attachArtifact(
                "legacy", artifact: legacy, experimentName: "rendering-pin")
            #expect(manifest.concepts.first?.options.extractionRendering == nil)
            #expect(ExperimentStore.verify(manifest).isEmpty)

            // An explicit raw declaration IS the absent one, canonically.
            manifest.concepts[0].options.extractionRendering = .raw
            #expect(ExperimentStore.verify(manifest).isEmpty)

            // A chat-template declaration over a raw artifact is not.
            manifest.concepts[0].options.extractionRendering = .chatTemplate()
            #expect(
                ExperimentStore.verify(manifest).contains {
                    $0.contains(
                        "held-out activations must be read as the vector was "
                            + "rendered")
                })
        }
    }

    /// Attach never writes a pin the very next verify rejects, so a
    /// chat-template artifact attaches with its rendering copied — and
    /// dropping that declaration afterwards is the contradiction above.
    @Test func attachCopiesTheArtifactsRendering() throws {
        try withTempRoot { root in
            try WorkspaceCompute.declare(.cluster, root: root)
            _ = try ExperimentStore.create(
                name: "rendering-copy", description: "", modelID: "test/model")

            let templated = try plantOptvecArtifact(
                root: root, name: "templated",
                rendering: [
                    "mode": "chatTemplate",
                    "addGenerationPrompt": false,
                    "qwenThinkingEnabled": false,
                ])
            var manifest = try ExperimentStore.attachArtifact(
                "templated", artifact: templated,
                experimentName: "rendering-copy")
            let declared = try #require(
                manifest.concepts.first?.options.extractionRendering)
            #expect(declared.mode == .chatTemplate)
            #expect(declared.resolvedAddGenerationPrompt == false)
            #expect(ExperimentStore.verify(manifest).isEmpty)

            manifest.concepts[0].options.extractionRendering = nil
            #expect(
                ExperimentStore.verify(manifest).contains {
                    $0.contains(
                        "held-out activations must be read as the vector was "
                            + "rendered")
                })
        }
    }

    @Test func attachRefusalsMirrorTheServer() throws {
        try withTempRoot { root in
            try WorkspaceCompute.declare(.cluster, root: root)
            _ = try ExperimentStore.create(
                name: "refusals", description: "", modelID: "test/model")

            // Born without norms → the backfill refusal, naming the verb.
            let unbackfilled = try plantOptvecArtifact(
                root: root, name: "raw", residualNormSource: nil)
            do {
                _ = try ExperimentStore.attachArtifact(
                    "raw", artifact: unbackfilled, experimentName: "refusals")
                Issue.record("norm-less optvec artifact must refuse")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("backfill"))
            }

            // Stripped optvec block → refuse.
            let stripped = try plantOptvecArtifact(
                root: root, name: "stripped", optvecBlock: nil)
            do {
                _ = try ExperimentStore.attachArtifact(
                    "stripped", artifact: stripped,
                    experimentName: "refusals")
                Issue.record("stripped sidecar must refuse")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("provenance block"))
            }

            // A source concept makes no sense for an optvec direction.
            let good = try plantOptvecArtifact(root: root, name: "good")
            do {
                _ = try ExperimentStore.attachArtifact(
                    "mine", artifact: good, sourceConcept: "crit",
                    experimentName: "refusals")
                Issue.record("sourceConcept must refuse for optvec")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("no source concept"))
            }

            // Wrong model → directions do not transfer.
            let foreign = try plantOptvecArtifact(
                root: root, name: "foreign", modelID: "other/model")
            do {
                _ = try ExperimentStore.attachArtifact(
                    "foreign", artifact: foreign, experimentName: "refusals")
                Issue.record("foreign-model artifact must refuse")
            } catch let error as ExperimentError {
                #expect(
                    error.reason.contains("does not transfer between models"))
            }

            // Foreign substrate vs the workspace's declared compute.
            try WorkspaceCompute.declare(.localMLX, root: root)
            do {
                _ = try ExperimentStore.attachArtifact(
                    "good", artifact: good, experimentName: "refusals")
                Issue.record("foreign-substrate artifact must refuse")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("across engines"))
            }
        }
    }

    @Test func evalRunCitationsAreVerifiedNotTrustedByName() throws {
        try withTempRoot { root in
            try WorkspaceCompute.declare(.cluster, root: root)
            _ = try ExperimentStore.create(
                name: "evidence", description: "", modelID: "test/model")
            let reference = try plantOptvecArtifact(root: root)
            let evalDir = root.appending(
                components: "runs", "20260810T140000000-optvec-eval-rve")

            // A citation naming NO run directory refuses (typo'd citations
            // are input errors, catchable now or never).
            try FileManager.default.removeItem(at: evalDir)
            do {
                _ = try ExperimentStore.attachArtifact(
                    "rve", artifact: reference, experimentName: "evidence")
                Issue.record("missing eval run must refuse")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("names no run directory"))
            }

            // An eval run that certifies a DIFFERENT tensor refuses.
            try plantCertifyingEvalRun(
                root: root, named: "20260810T140000000-optvec-eval-rve",
                tensorHash: String(repeating: "ab", count: 32))
            do {
                _ = try ExperimentStore.attachArtifact(
                    "rve", artifact: reference, experimentName: "evidence")
                Issue.record("foreign-tensor eval run must refuse")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("DIFFERENT direction"))
            }

            // A run directory without eval.json attaches UNVERIFIED (a
            // crashed or partially imported eval may complete later), and
            // the freeze advisory downgrades it.
            try FileManager.default.removeItem(
                at: evalDir.appending(component: "eval.json"))
            let manifest = try ExperimentStore.attachArtifact(
                "rve", artifact: reference, experimentName: "evidence")
            let pin = try #require(
                manifest.concepts.first { $0.name == "rve" }?.vectorArtifact)
            #expect(pin.optvecEvalRunVerified == false)
            #expect(
                pin.optvecEvalRunUnverifiedReason?.contains("no eval.json")
                    == true)
            let advisories = ExperimentStore.freezeAdvisories(for: manifest)
            #expect(
                advisories.contains {
                    $0.contains("could NOT be verified at attach")
                })

            // A legacy pin (recorded before verification existed) is flagged
            // as never-checked rather than described as evidence.
            var legacy = manifest
            legacy.concepts[0].vectorArtifact?.optvecEvalRunVerified = nil
            legacy.concepts[0].vectorArtifact?
                .optvecEvalRunUnverifiedReason = nil
            #expect(
                ExperimentStore.freezeAdvisories(for: legacy).contains {
                    $0.contains("never checked against this artifact")
                })
        }
    }

    @Test func attachContainmentResolvesSymlinks() throws {
        try withTempRoot { root in
            try WorkspaceCompute.declare(.cluster, root: root)
            _ = try ExperimentStore.create(
                name: "contain", description: "", modelID: "test/model")
            // Real bytes OUTSIDE the workspace; a symlinked directory inside
            // runs/ points at them, so the lexical check alone would pass.
            let outside = root.deletingLastPathComponent()
                .appending(component: "outside-\(UUID().uuidString)")
            try FileManager.default.createDirectory(
                at: outside, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: outside) }
            try Data("fake-tensor".utf8).write(
                to: outside.appending(component: "esc.safetensors"))
            try JSONSerialization.data(withJSONObject: ["modelID": "test/model"])
                .write(to: outside.appending(component: "esc.json"))
            let runsDir = root.appending(component: "runs")
            try FileManager.default.createDirectory(
                at: runsDir, withIntermediateDirectories: true)
            try FileManager.default.createSymbolicLink(
                at: runsDir.appending(component: "linked"),
                withDestinationURL: outside)
            do {
                _ = try ExperimentStore.attachArtifact(
                    "esc", artifact: "runs/linked/esc",
                    experimentName: "contain")
                Issue.record("symlink escape must refuse")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("resolves outside the workspace"))
            }
        }
    }

    @Test func verifyReportsArtifactDriftAndAttachConceptRefusesNonRecipe()
        throws
    {
        try withTempRoot { root in
            try WorkspaceCompute.declare(.cluster, root: root)
            _ = try ExperimentStore.create(
                name: "drift", description: "", modelID: "test/model")
            let reference = try plantOptvecArtifact(root: root)
            _ = try ExperimentStore.attachArtifact(
                "rve", artifact: reference, experimentName: "drift")

            // Mutate the tensor bytes AFTER pinning.
            try Data("tampered".utf8).write(
                to: root.appending(path: "\(reference).safetensors"))
            let manifest = try #require(
                ExperimentStore.list().first { $0.name == "drift" })
            let violations = ExperimentStore.verify(manifest)
            #expect(
                violations.contains {
                    $0.contains("vector artifact")
                        && $0.contains("changed since pinning")
                })

            // Recipe attach refuses the non-recipe methods outright.
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.attachConcept(
                    "rve", method: .pinnedArtifact, experimentName: "drift")
            }
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.attachConcept(
                    "rve", method: .optvec, experimentName: "drift")
            }
        }
    }

    @Test func optvecOnlyStudiesAreExemptFromTheValidateGateWithAdvisory()
        throws
    {
        try withTempRoot { root in
            try WorkspaceCompute.declare(.cluster, root: root)
            _ = try ExperimentStore.create(
                name: "exempt", description: "", modelID: "test/model")
            let reference = try plantOptvecArtifact(root: root)
            let manifest = try ExperimentStore.attachArtifact(
                "rve", artifact: reference, experimentName: "exempt")

            #expect(ExperimentStore.optvecExemptFromValidateGate(manifest))
            let advisories = ExperimentStore.freezeAdvisories(for: manifest)
            #expect(
                advisories.contains {
                    $0.contains("OptVec direction")
                        && $0.contains("20260810T140000000-optvec-eval-rve")
                })

            // A mixed study keeps the gate.
            var mixed = manifest
            mixed.concepts.append(
                .init(name: "fear", stimulusSetHash: "abc", options: .init()))
            #expect(!ExperimentStore.optvecExemptFromValidateGate(mixed))
        }
    }

}
