import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// Pin-time shape validation (Usability Plan Phase 0 item 2): a pin
/// validates the file's SHAPE when it is made, not only its bytes — a
/// schema-garbage file must refuse at the pin with a plain-language error
/// instead of pinning cleanly and failing much later at analyze/run.
/// Serialized: several tests hold the experiment root override.
@Suite(.serialized) struct PinShapeValidationTests {

    private func withTempRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "pinshape", body)
    }

    private func write(_ text: String, to root: URL, path: String) throws {
        let url = root.appending(path: path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: url, atomically: true, encoding: .utf8)
    }

    private let validBaseline =
        "endpoint,deltaHuman,ciLower,ciUpper\nrate,0.12,0.05,0.19\n"
    private let validBattery = #"{"prompt": "2+2?", "answer": "4"}"# + "\n"
        + #"{"prompt": "Sky color?", "answer": "blue", "grading": "token_exact"}"#
        + "\n"

    // MARK: - Human-baseline contract

    /// Lock the exact loader column list (residuals.py
    /// `HUMAN_BASELINE_FIELDS` — the loader's columns ARE the contract).
    @Test func requiredColumnsLockTheLoaderContract() {
        #expect(
            PinShapeValidation.humanBaselineRequiredColumns
                == ["endpoint", "deltaHuman", "ciLower", "ciUpper"])
    }

    /// The shipped template satisfies the contract exactly (its own lock
    /// lives in StudyDataReadinessTests; this asserts template → loader
    /// agreement through the same shape check the pin runs).
    @Test func templateMatchesTheLoaderColumns() throws {
        let seed = DataTemplates.seedURL(
            for: DataTemplates.humanBaseline,
            workspaceRoot: URL(filePath: "/nonexistent"))
        let data = try Data(contentsOf: seed)
        #expect(
            PinShapeValidation.humanBaselineShapeProblem(data, file: "t") == nil)
        // Required columns lead the header, extras (provenance) follow.
        let header = PinShapeValidation.csvHeaderColumns(data)
        #expect(
            Array(header.prefix(4))
                == PinShapeValidation.humanBaselineRequiredColumns)
    }

    @Test func headerParsingTrimsQuotesWhitespaceAndBOM() {
        let data = Data("\u{FEFF}\"endpoint\" , deltaHuman,ciLower,ciUpper\n".utf8)
        #expect(
            PinShapeValidation.csvHeaderColumns(data)
                == ["endpoint", "deltaHuman", "ciLower", "ciUpper"])
        #expect(
            PinShapeValidation.humanBaselineShapeProblem(data, file: "f") == nil)
    }

    @Test func baselineShapeProblemNamesBothColumnLists() throws {
        let problem = PinShapeValidation.humanBaselineShapeProblem(
            Data("source,measure,population,delta,ci_low,ci_high,n,notes\n".utf8),
            file: "prompts/baselines/old.csv")
        let text = try #require(problem)
        // The required list, the found list, and the remedy — plain words.
        #expect(text.contains("endpoint, deltaHuman, ciLower, ciUpper"))
        #expect(text.contains("source, measure, population"))
        #expect(text.contains("Create from template"))
    }

    @Test func extraColumnsAreFine() {
        #expect(
            PinShapeValidation.humanBaselineShapeProblem(
                Data("notes,endpoint,deltaHuman,ciLower,ciUpper,n\n".utf8),
                file: "f") == nil)
    }

    // MARK: - Row-level validation (the loader's tolerances)

    /// Every form the analyze loader (residuals.py, `csv.DictReader` +
    /// `float()`) accepts must pin: quoted numbers, surrounding
    /// whitespace, scientific notation, signs, bare decimal points,
    /// Python digit grouping, blank lines, extra columns.
    @Test func rowLevelAcceptsWhatTheLoaderAccepts() {
        let csv = "endpoint,deltaHuman,ciLower,ciUpper,source\n"
            + "rate,0.12,0.05,0.19,plain\n"
            + "flip,\"0.31\",\" -0.12 \",4.8e-1,quoted and padded\n"
            + "months,-2.5,+.5,5.,signs and bare points\n"
            + "\n"
            + "grouped,1_000.5,0.1,0.2,python digit grouping\n"
        #expect(
            PinShapeValidation.humanBaselineShapeProblem(
                Data(csv.utf8), file: "f") == nil)
        // Header-only files load as zero endpoints — allowed, like the loader.
        #expect(
            PinShapeValidation.humanBaselineShapeProblem(
                Data("endpoint,deltaHuman,ciLower,ciUpper\n".utf8), file: "f")
                == nil)
    }

    @Test func rowLevelRefusesNamingFirstBadRowAndField() throws {
        let csv = "endpoint,deltaHuman,ciLower,ciUpper\n"
            + "rate,0.12,0.05,0.19\n"
            + "flip,about 0.3,0.1,0.5\n"
        let problem = try #require(
            PinShapeValidation.humanBaselineShapeProblem(
                Data(csv.utf8), file: "b.csv"))
        #expect(problem.contains("data row 2"))
        #expect(problem.contains("deltaHuman"))
        #expect(problem.contains("about 0.3"))
        #expect(problem.contains("not a number"))
    }

    @Test func rowLevelRefusesMissingNumericCell() throws {
        // A short row leaves ciUpper absent — the loader dies on it
        // (`float(None)`), so the pin refuses naming the row and field.
        let short = "endpoint,deltaHuman,ciLower,ciUpper\nrate,0.12,0.05\n"
        let problem = try #require(
            PinShapeValidation.humanBaselineShapeProblem(
                Data(short.utf8), file: "b.csv"))
        #expect(problem.contains("data row 1"))
        #expect(problem.contains("ciUpper"))
        // An empty cell likewise (`float("")` refuses in the loader).
        let empty = "endpoint,deltaHuman,ciLower,ciUpper\nrate,,0.05,0.19\n"
        #expect(
            PinShapeValidation.humanBaselineShapeProblem(
                Data(empty.utf8), file: "b.csv")?.contains("deltaHuman") == true)
    }

    /// The numeric acceptance rule follows the LOADER, not Swift: hex
    /// floats (Swift-only) refuse; inf/nan/scientific/underscore forms
    /// (Python `float()`) accept.
    @Test func loaderNumberRuleFollowsPythonFloat() {
        for good in ["0.5", " 7 ", "+.5", "5.", "1e3", "-4.8E-1",
                     "inf", "-Infinity", "nan", "1_000", "1_000.5"] {
            #expect(PinShapeValidation.parsesAsLoaderNumber(good), "\(good)")
        }
        for bad in ["", "  ", "0x1p3", "0X1P3", "_1", "1_", "1__0", "1_e3",
                    "about 0.3", "1,5"] {
            #expect(!PinShapeValidation.parsesAsLoaderNumber(bad), "\(bad)")
        }
    }

    @Test func pinHumanBaselineRefusesMalformedRowNamingRowAndField() throws {
        try withTempRoot { root in
            try ExperimentStore.create(
                name: "shape-row", description: "d", modelID: "test/model")
            let path = "prompts/baselines/shape-row.csv"
            try write(
                "endpoint,deltaHuman,ciLower,ciUpper\nrate,0.1,low,0.2\n",
                to: root, path: path)
            do {
                _ = try ExperimentStore.pinHumanBaseline(
                    path: path, experimentName: "shape-row")
                Issue.record("expected the row-level refusal")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("data row 1"))
                #expect(error.reason.contains("ciLower"))
            }
            #expect(try ExperimentStore.load(name: "shape-row").humanBaseline == nil)
        }
    }

    // MARK: - pinHumanBaseline refuses / accepts at pin time

    @Test func pinHumanBaselineRefusesWrongColumnsWithPlainError() throws {
        try withTempRoot { root in
            try ExperimentStore.create(
                name: "shape-a", description: "d", modelID: "test/model")
            let path = "prompts/baselines/shape-a.csv"
            try write(
                "source,measure,delta,ci_low\nX,rate,0.1,0.0\n",
                to: root, path: path)
            do {
                _ = try ExperimentStore.pinHumanBaseline(
                    path: path, experimentName: "shape-a")
                Issue.record("expected the shape refusal")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("endpoint, deltaHuman, ciLower, ciUpper"))
                #expect(error.reason.contains("source, measure, delta, ci_low"))
            }
            // Nothing was pinned by the refusal.
            #expect(try ExperimentStore.load(name: "shape-a").humanBaseline == nil)
        }
    }

    @Test func pinHumanBaselineAcceptsValidHeaderWithExtras() throws {
        try withTempRoot { root in
            try ExperimentStore.create(
                name: "shape-b", description: "d", modelID: "test/model")
            let path = "prompts/baselines/shape-b.csv"
            try write(
                "endpoint,deltaHuman,ciLower,ciUpper,source,notes\n"
                    + "rate,0.12,0.05,0.19,\"Example (2020)\",\"checked\"\n",
                to: root, path: path)
            let pinned = try ExperimentStore.pinHumanBaseline(
                path: path, experimentName: "shape-b")
            #expect(pinned.hash.count == 64)
            #expect(try ExperimentStore.load(name: "shape-b").humanBaseline == pinned)
        }
    }

    // MARK: - Capability battery shape

    @Test func batteryShapeAcceptsWhatTheRunnerLoads() {
        #expect(
            PinShapeValidation.capabilityBatteryShapeProblem(
                Data(validBattery.utf8), file: "f") == nil)
    }

    @Test func batteryShapeRefusesGarbageNamingTheLine() throws {
        let garbage = #"{"prompt": "2+2?", "answer": "4"}"# + "\n"
            + "not json at all\n"
        let problem = PinShapeValidation.capabilityBatteryShapeProblem(
            Data(garbage.utf8), file: "prompts/batteries/bad.jsonl")
        let text = try #require(problem)
        #expect(text.contains("line 2"))
        #expect(text.contains("prompts/batteries/bad.jsonl"))
        #expect(text.contains("prompts/batteries/basic.jsonl"))
        // A row missing the required keys refuses too (the runner would).
        #expect(
            PinShapeValidation.capabilityBatteryShapeProblem(
                Data(#"{"prompt": "no answer key"}"#.utf8), file: "f") != nil)
        // …and an empty battery refuses (0-of-0 scoring is an accident).
        #expect(
            PinShapeValidation.capabilityBatteryShapeProblem(
                Data("\n".utf8), file: "f") != nil)
    }

    @Test func pinCapabilityBatteryValidatesAtPinTime() throws {
        try withTempRoot { root in
            try write(validBattery, to: root, path: "prompts/batteries/ok.jsonl")
            try write(
                "garbage\n", to: root, path: "prompts/batteries/bad.jsonl")
            var manifest = ExperimentManifest(
                name: "bat", description: "", modelID: "test/model")
            let hash = try ExperimentStore.pinCapabilityBattery(
                "prompts/batteries/ok.jsonl", into: &manifest)
            #expect(hash.count == 64)
            #expect(manifest.capabilityBatteryFile == "prompts/batteries/ok.jsonl")
            #expect(manifest.capabilityBatteryHash == hash)

            var untouched = ExperimentManifest(
                name: "bat2", description: "", modelID: "test/model")
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.pinCapabilityBattery(
                    "prompts/batteries/bad.jsonl", into: &untouched)
            }
            #expect(untouched.capabilityBatteryFile == nil)
            #expect(untouched.capabilityBatteryHash == nil)
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.pinCapabilityBattery(
                    "prompts/batteries/absent.jsonl", into: &untouched)
            }
        }
    }

    /// Finding B1: the nil-tolerant freeze-time overload
    /// (`pinCapabilityBattery(into:)` — manifest's file or the shared
    /// DEFAULT) used to pin malformed bytes. Now a malformed file is
    /// treated exactly like an ABSENT one: the pin is skipped, nothing is
    /// stamped, and freeze behavior matches the absent-file case.
    @Test func defaultBatteryPinSkipsMalformedFileLikeAbsent() throws {
        try withTempRoot { root in
            let file = VariantRobustness.defaultPreset.batteryFile
            try write("not a battery item\n", to: root, path: file)
            var manifest = ExperimentManifest(
                name: "vb", description: "", modelID: "test/model")
            #expect(ExperimentStore.pinCapabilityBattery(into: &manifest) == nil)
            #expect(manifest.capabilityBatteryFile == nil)
            #expect(manifest.capabilityBatteryHash == nil)
            // Fixing the file makes the very same pin succeed.
            try write(validBattery, to: root, path: file)
            let pinned = ExperimentStore.pinCapabilityBattery(into: &manifest)
            #expect(pinned?.file == file)
            #expect(manifest.capabilityBatteryFile == file)
            #expect(manifest.capabilityBatteryHash == pinned?.hash)
        }
    }

    // MARK: - Study-pack auto-pin surfaces (never swallows) a shape problem

    private func pack(name: String, batteryLine: String) -> String {
        """
        {
          "study": {
            "name": "\(name)", "status": "draft",
            "studyType": "agentComparison",
            "studyKind": "modelOutput",
            "experimentDescription": "packed study",
            "modelID": "test/model", "createdAt": "2026-07-19",
            "temperature": 0, "maxTokens": 64, "seeds": [0],
            "multiAgentIncludeBaseline": true,
            "capabilityBatteryFile": "prompts/batteries/\(name).jsonl",
            "concepts": [], "conditions": [], "variantConditions": []
          },
          "files": {
            "prompts/batteries/\(name).jsonl": "\(batteryLine)\\n"
          }
        }
        """
    }

    @Test func packedValidBatteryIsPinnedOnImport() throws {
        try withTempRoot { _ in
            let (imported, violations, written) = try ExperimentStore.importStudyJSON(
                pack(
                    name: "packed-ok",
                    batteryLine:
                        #"{\"prompt\": \"2+2?\", \"answer\": \"4\"}"#))
            #expect(written == ["prompts/batteries/packed-ok.jsonl"])
            #expect(imported.capabilityBatteryHash != nil)
            #expect(!violations.contains { $0.contains("capability battery") })
        }
    }

    // MARK: - Verify-time shape backstop (Finding 3, 2026-07-19)

    private func sha256(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8))
            .map { String(format: "%02x", $0) }.joined()
    }

    /// A present-but-MALFORMED pinned file whose hash MATCHES used to pass
    /// verify and die much later (battery at task load, baseline at
    /// analyze) — files legally enter manifests without the pin path
    /// (hand-edit, import, legacy), so verify is the backstop. The shape
    /// violation fires ONLY on a hash match; drift and missing files keep
    /// their own messages (never double-reported).
    @Test func verifyFlagsMalformedButHashMatchingBatteryAndBaseline() throws {
        try withTempRoot { root in
            let badBattery = "not a battery item\n"
            try write(badBattery, to: root, path: "prompts/batteries/mal.jsonl")
            let badBaseline = "caseID,delta\nA,0.12\n"
            try write(badBaseline, to: root, path: "prompts/baselines/mal.csv")

            var manifest = ExperimentManifest(
                name: "shape-verify", description: "", modelID: "test/model")
            manifest.capabilityBatteryFile = "prompts/batteries/mal.jsonl"
            manifest.capabilityBatteryHash = sha256(badBattery)
            manifest.humanBaseline = .init(
                path: "prompts/baselines/mal.csv", hash: sha256(badBaseline))

            let violations = ExperimentStore.verify(manifest)
            // The battery violation names the problem the run loop would hit.
            #expect(violations.contains {
                $0.contains("prompts/batteries/mal.jsonl")
                    && $0.contains("line 1")
            })
            // The baseline violation names both column lists.
            #expect(violations.contains {
                $0.contains("endpoint, deltaHuman, ciLower, ciUpper")
                    && $0.contains("caseID, delta")
            })
            // Shape only — no spurious drift/missing report rides along.
            #expect(!violations.contains { $0.contains("changed since pinning") })

            // A malformed ROW (valid header) is named row-and-field.
            let badRow = "endpoint,deltaHuman,ciLower,ciUpper\nrate,oops,0.1,0.2\n"
            try write(badRow, to: root, path: "prompts/baselines/row.csv")
            manifest.humanBaseline = .init(
                path: "prompts/baselines/row.csv", hash: sha256(badRow))
            #expect(ExperimentStore.verify(manifest).contains {
                $0.contains("data row 1") && $0.contains("deltaHuman")
            })
        }
    }

    @Test func verifyShapeChecksStayCleanOnValidFilesAndDeferToDrift() throws {
        try withTempRoot { root in
            let baseline =
                "endpoint,deltaHuman,ciLower,ciUpper\nrate,0.12,0.05,0.19\n"
            try write(validBattery, to: root, path: "prompts/batteries/ok.jsonl")
            try write(baseline, to: root, path: "prompts/baselines/ok.csv")

            var manifest = ExperimentManifest(
                name: "shape-clean", description: "", modelID: "test/model")
            manifest.capabilityBatteryFile = "prompts/batteries/ok.jsonl"
            manifest.capabilityBatteryHash = sha256(validBattery)
            manifest.humanBaseline = .init(
                path: "prompts/baselines/ok.csv", hash: sha256(baseline))
            let clean = ExperimentStore.verify(manifest)
            #expect(!clean.contains { $0.contains("capability battery") })
            #expect(!clean.contains { $0.contains("human baseline") })
            #expect(!clean.contains { $0.contains("analyze step") })

            // DRIFT keeps its own message and does NOT double-report shape
            // (the drifted bytes on disk are malformed here, but the hash
            // mismatch is the violation).
            try write("garbage\n", to: root, path: "prompts/batteries/ok.jsonl")
            try write("bad,header\nx,1\n", to: root, path: "prompts/baselines/ok.csv")
            let drifted = ExperimentStore.verify(manifest)
            #expect(drifted.contains {
                $0.contains("capability battery")
                    && $0.contains("changed since pinning")
            })
            #expect(drifted.contains {
                $0.contains("human baseline")
                    && $0.contains("changed since pinning")
            })
            #expect(!drifted.contains { $0.contains("analyze step") })
            #expect(!drifted.contains { $0.contains("not shaped the way") })

            // MISSING keeps its own message too.
            manifest.capabilityBatteryFile = "prompts/batteries/absent.jsonl"
            manifest.humanBaseline = .init(
                path: "prompts/baselines/absent.csv", hash: sha256(baseline))
            let missing = ExperimentStore.verify(manifest)
            #expect(missing.contains { $0.contains("capability battery missing") })
            #expect(missing.contains { $0.contains("human baseline missing") })
        }
    }

    @Test func packedShapeInvalidBatterySurfacesAsFindings() throws {
        try withTempRoot { root in
            let (imported, violations, written) = try ExperimentStore.importStudyJSON(
                pack(name: "packed-bad", batteryLine: "not a battery item"))
            // The file landed (the pack is data), but the shape-garbage
            // battery was NOT silently pinned…
            #expect(written == ["prompts/batteries/packed-bad.jsonl"])
            #expect(imported.capabilityBatteryHash == nil)
            // …and the problem surfaces twice: verify names the incomplete
            // pin, and readiness marks the file invalid with the plain
            // shape detail (a blocker, not a silent disappearance).
            #expect(violations.contains {
                $0.contains("capability battery is incompletely pinned")
            })
            let rows = StudyDataReadiness.requirements(
                for: imported, workspaceRoot: root)
            let battery = try #require(
                rows.first { $0.kind == .capabilityBattery })
            #expect(battery.status == .invalid)
            #expect(battery.detail.contains("line 1"))
            #expect(StudyDataReadiness.summary(rows).blockers.contains {
                $0.kind == .capabilityBattery
            })
        }
    }
}
