import CryptoKit
import Foundation
import SteeringKit
import Testing
@testable import ExperimentKit

/// The foreign-local-judge pin gate (external review round 2, finding 3).
///
/// A local judge naming a model OTHER than the study model must pin the
/// exact bytes that will judge — `revision` and `dtype`. Without it,
/// targeted retry compares two sessions' recorded judge identities to
/// decide whether earlier verdicts may be REUSED, and `nil == nil` passes
/// while the two sessions loaded different defaults.
///
/// The gate shipped in 10adf47d8 with no direct coverage on either engine
/// (external review round 4, finding 1) — existing fixtures were merely
/// edited to survive it, which proves nothing about whether it fires.
/// These are the tests that prove it. Python twin:
/// `Server/tests/test_foreign_local_judge_pins.py`.
extension ExperimentStoreTests {

    private func plantFile(_ relativePath: String, _ text: String) throws -> String {
        let root = try #require(ExperimentStore.rootOverride)
        let url = root.appending(path: relativePath)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let data = Data(text.utf8)
        try data.write(to: url)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// A judged study whose only possible freeze problem is the judge pins.
    private func makeJudgePinStudy(
        _ name: String, judges: [ExperimentManifest.JudgeRef]
    ) throws -> ExperimentManifest {
        var manifest = try ExperimentStore.create(
            name: name, description: "", modelID: "test/model",
            modelRevision: "abc123def456")
        manifest.concepts.append(
            .init(name: "french", stimulusSetHash: try realFrenchHash(),
                  options: .init()))
        let rubricPath = "prompts/rubrics/pin-\(name).md"
        manifest.judgeRubricFile = rubricPath
        manifest.judgeRubricHash = try plantFile(
            rubricPath, "Prefer the response the rubric describes.")
        manifest.judges = judges
        try ExperimentStore.save(manifest)
        try fabricateValidationEvidence(for: manifest)
        return manifest
    }

    private func foreignPanel(
        revision: String? = nil, dtype: String? = nil
    ) -> [ExperimentManifest.JudgeRef] {
        [.init(name: "judge-1", kind: "local", model: nil),
         .init(name: "judge-2", kind: "local", model: "other/judge-12b",
               revision: revision, dtype: dtype)]
    }

    /// The reason freeze refused, or nil when it did not refuse.
    private func freezeRefusal(_ name: String) -> String? {
        do {
            _ = try ExperimentStore.freeze(name: name)
            return nil
        } catch let error as ExperimentError {
            return error.reason
        } catch {
            return "\(error)"
        }
    }

    // MARK: - The gate fires, per missing field

    @Test func freezeRefusesForeignLocalJudgeMissingRevision() throws {
        try withTempRoot {
            _ = try makeJudgePinStudy("fpr", judges: foreignPanel(dtype: "bfloat16"))
            let reason = try #require(freezeRefusal("fpr"))
            #expect(reason.contains("pin the exact bytes"))
            // Only the field actually absent is named as missing (the
            // remedy sentence later mentions both, which is correct).
            #expect(reason.contains(
                "'judge-2' (model 'other/judge-12b') is missing revision."))
        }
    }

    @Test func freezeRefusesForeignLocalJudgeMissingDtype() throws {
        try withTempRoot {
            _ = try makeJudgePinStudy("fpd", judges: foreignPanel(revision: "cafe01"))
            let reason = try #require(freezeRefusal("fpd"))
            #expect(reason.contains(
                "'judge-2' (model 'other/judge-12b') is missing dtype"))
        }
    }

    @Test func freezeNamesBothFieldsWhenBothAreAbsent() throws {
        try withTempRoot {
            _ = try makeJudgePinStudy("fpb", judges: foreignPanel())
            let reason = try #require(freezeRefusal("fpb"))
            #expect(reason.contains("is missing revision and dtype"))
        }
    }

    /// A field present but blank must not satisfy the gate — the recorded
    /// identity would still be empty at retry time.
    @Test func whitespaceIsNotAPin() throws {
        try withTempRoot {
            _ = try makeJudgePinStudy(
                "fpw", judges: foreignPanel(revision: "   ", dtype: "\t"))
            let reason = try #require(freezeRefusal("fpw"))
            #expect(reason.contains("is missing revision and dtype"))
        }
    }

    @Test func everyOffenderIsNamedNotJustTheFirst() throws {
        try withTempRoot {
            var judges = foreignPanel(revision: "cafe01", dtype: "bfloat16")
            judges.append(
                .init(name: "judge-3", kind: "local", model: "third/judge"))
            _ = try makeJudgePinStudy("fpa", judges: judges)
            let reason = try #require(freezeRefusal("fpa"))
            #expect(reason.contains("judge-3"))
        }
    }

    // MARK: - The gate stays silent where it should

    @Test func fullyPinnedForeignLocalJudgeFreezes() throws {
        try withTempRoot {
            _ = try makeJudgePinStudy(
                "fpk", judges: foreignPanel(revision: "cafe01", dtype: "bfloat16"))
            let frozen = try ExperimentStore.freeze(name: "fpk")
            #expect(frozen.status == .frozen)
            #expect(frozen.freezeForced != true)
        }
    }

    /// A local judge resolving to the STUDY model inherits the study's
    /// pinned revision, so there is nothing for it to pin — both the blank
    /// form and the explicitly-named form.
    ///
    /// Asserted on the rule itself for the two-study-model panel, because
    /// FREEZING that panel is refused by a different gate (both judges are
    /// the same deterministic judge); the freeze half uses a distinct panel.
    @Test func studyModelLocalJudgesNeedNoPins() throws {
        try withTempRoot {
            var manifest = ExperimentManifest(
                name: "both", description: "", modelID: "test/model")
            manifest.judges = [
                .init(name: "judge-1", kind: "local", model: nil),
                .init(name: "judge-2", kind: "local", model: "test/model"),
            ]
            #expect(
                ExperimentStore.unpinnedForeignLocalJudgeProblem(manifest) == nil)

            _ = try makeJudgePinStudy("fps", judges: [
                .init(name: "judge-1", kind: "local", model: "test/model"),
                .init(name: "judge-2", kind: "openrouter",
                      model: "anthropic/claude-opus-4", provider: "Anthropic"),
            ])
            #expect(try ExperimentStore.freeze(name: "fps").status == .frozen)
        }
    }

    @Test func externalJudgesAreUntouched() {
        var manifest = ExperimentManifest(
            name: "ext", description: "", modelID: "test/model")
        manifest.judges = [
            .init(name: "judge-1", kind: "local", model: nil),
            .init(name: "judge-2", kind: "openrouter",
                  model: "anthropic/claude-opus-4", provider: "Anthropic"),
        ]
        #expect(ExperimentStore.unpinnedForeignLocalJudgeProblem(manifest) == nil)
    }

    // MARK: - The closed dtype vocabulary (review round 4, finding 2)

    @Test func aliasesNormalizeToCanonicalSpellings() {
        for (alias, canonical) in [
            ("bfloat16", "bfloat16"), ("bf16", "bfloat16"), ("BF16", "bfloat16"),
            ("float16", "float16"), ("fp16", "float16"),
            ("float32", "float32"), ("fp32", "float32"),
            ("  bf16  ", "bfloat16"),
        ] {
            #expect(ExperimentStore.normalizeJudgeDtype(alias) == canonical)
        }
        for unknown in ["banana", "int8", "float64", "bfloat", "", "auto"] {
            #expect(ExperimentStore.normalizeJudgeDtype(unknown) == nil)
        }
    }

    @Test func freezeRefusesAJudgeDtypeOutsideTheVocabulary() throws {
        try withTempRoot {
            _ = try makeJudgePinStudy("fpx", judges: [
                .init(name: "judge-1", kind: "local", model: nil),
                .init(name: "judge-2", kind: "local", model: "other/judge-12b",
                      revision: "cafe01", dtype: "banana"),
            ])
            let reason = try #require(freezeRefusal("fpx"))
            #expect(reason.contains("cannot load"))
            #expect(reason.contains("'judge-2' declares dtype 'banana'"))
            for name in ExperimentStore.judgeDtypeVocabulary {
                #expect(reason.contains(name))
            }
        }
    }

    /// A study-model judge needs no PIN, but a dtype it does declare still
    /// has to be loadable — the server's loader would refuse it on the
    /// compute node, after the queue wait.
    @Test func aBadDtypeIsCaughtEvenOnAStudyModelJudge() throws {
        try withTempRoot {
            _ = try makeJudgePinStudy("fpy", judges: [
                .init(name: "judge-1", kind: "local", model: nil, dtype: "int4"),
                .init(name: "judge-2", kind: "openrouter",
                      model: "anthropic/claude-opus-4", provider: "Anthropic"),
            ])
            let reason = try #require(freezeRefusal("fpy"))
            #expect(reason.contains("cannot load"))
        }
    }

    @Test func aliasSpellingsAreAcceptedByTheGate() throws {
        try withTempRoot {
            _ = try makeJudgePinStudy(
                "fpz", judges: foreignPanel(revision: "cafe01", dtype: "bf16"))
            #expect(try ExperimentStore.freeze(name: "fpz").status == .frozen)
        }
    }

    // MARK: - Study-model judges cannot pin another identity (round 5, F1)

    /// A judgeScore sweep, with its pinned inputs planted so freeze reaches
    /// the judge gate rather than stopping at the sweep pins.
    private func makeJudgedSweepStudy(
        _ name: String, judges: [ExperimentManifest.JudgeRef]
    ) throws -> ExperimentManifest {
        // The sweep's DEFAULT input paths — an operative sweep pins these
        // at freeze, so they must exist for the freeze attempt to reach the
        // judge gate under test.
        _ = try plantFile(
            "prompts/dev/dev-prompts.jsonl",
            "{\"id\": \"d1\", \"prompt\": \"Describe the cellar.\"}\n")
        _ = try plantFile(
            "prompts/batteries/basic.jsonl",
            "{\"id\": \"b1\", \"prompt\": \"2+2?\", \"answer\": \"4\"}\n")
        var manifest = try makeJudgePinStudy(name, judges: judges)
        manifest.sweep = .init(
            selection: .init(objective: .init(metric: "judgeScore")))
        try ExperimentStore.save(manifest)
        try fabricateValidationEvidence(for: manifest)
        return manifest
    }

    private func studyModelJudgePanel(
        model: String? = "test/model", revision: String? = nil,
        dtype: String? = nil
    ) -> [ExperimentManifest.JudgeRef] {
        [.init(name: "judge-1", kind: "local", model: model,
               revision: revision, dtype: dtype),
         .init(name: "judge-2", kind: "openrouter",
               model: "anthropic/claude-opus-4", provider: "Anthropic")]
    }

    /// The sweep judges with the HELD weights and never loads a second
    /// revision, so the pin would be silently ignored — while `evaluate`
    /// DOES honor it. One manifest, two identities, depending on the verb.
    @Test func freezeRefusesAStudyModelJudgePinningAnotherRevision() throws {
        try withTempRoot {
            _ = try makeJudgedSweepStudy(
                "smr", judges: studyModelJudgePanel(revision: "beef02"))
            let reason = try #require(freezeRefusal("smr"))
            #expect(reason.contains("cannot pin a different identity"))
            #expect(reason.contains("'judge-1' pins revision 'beef02'"))
            #expect(reason.contains("is pinned at 'abc123def456'"))
        }
    }

    @Test func freezeRefusesAStudyModelJudgePinningAnotherDtype() throws {
        try withTempRoot {
            _ = try makeJudgedSweepStudy(
                "smd", judges: studyModelJudgePanel(dtype: "float32"))
            let reason = try #require(freezeRefusal("smd"))
            // The study pins no dtype, so ANY declared dtype is a claim the
            // sweep cannot honor.
            #expect(reason.contains("pins none (the device decides)"))
        }
    }

    /// Blank model and explicit study model both resolve to the study model,
    /// so both must be checked.
    @Test func aBlankModelJudgeIsCoveredToo() throws {
        try withTempRoot {
            _ = try makeJudgedSweepStudy(
                "smb", judges: studyModelJudgePanel(model: nil, revision: "beef03"))
            let reason = try #require(freezeRefusal("smb"))
            #expect(reason.contains("cannot pin a different identity"))
        }
    }

    /// Redundant, not wrong — the gate is about DIVERGENCE.
    @Test func pinsThatAgreeWithTheStudyAreLegal() throws {
        try withTempRoot {
            _ = try makeJudgedSweepStudy(
                "sma", judges: studyModelJudgePanel(revision: "abc123def456"))
            #expect(try ExperimentStore.freeze(name: "sma").status == .frozen)
        }
    }

    @Test func aForeignJudgeIsNotAffectedByThisGate() {
        var manifest = ExperimentManifest(
            name: "smf", description: "", modelID: "test/model")
        manifest.modelRevision = "abc"
        manifest.sweep = .init(
            selection: .init(objective: .init(metric: "judgeScore")))
        manifest.judges = [
            .init(name: "j", kind: "local", model: "other/judge-12b",
                  revision: "cafe01", dtype: "bfloat16"),
        ]
        #expect(ExperimentStore.studyModelJudgePinConflict(manifest) == nil)
    }

    /// NOT a blanket rule: evaluate genuinely LOADS a declared judge
    /// revision, so a different checkpoint of the study repo is a
    /// legitimate judge there. Only a judgeScore sweep cannot honor it.
    @Test func evaluateOnlyStudiesMayJudgeWithAnotherCheckpoint() throws {
        try withTempRoot {
            _ = try makeJudgePinStudy(
                "sme", judges: studyModelJudgePanel(revision: "beef04"))
            #expect(try ExperimentStore.freeze(name: "sme").status == .frozen)
        }
    }

    // MARK: - A pin must name a commit, not a moving ref (round 5, F4)

    /// A branch is re-pointed by definition and a tag can be moved, so
    /// neither identifies the bytes a run used — the old gate only required
    /// non-emptiness, and the loader recorded the symbolic name it was
    /// handed rather than the commit it resolved to.
    @Test func movingReferencesAreNotPins() {
        for moving in ["main", "master", "HEAD", "refs/pr/1", "v1.0",
                       "latest", "release-2026"] {
            #expect(!ExperimentStore.isCommitLike(moving))
            var manifest = ExperimentManifest(
                name: "mv", description: "", modelID: "test/model")
            manifest.modelRevision = moving
            let problem = ExperimentStore.symbolicRevisionProblem(manifest)
            #expect(problem?.contains("the study model pins '\(moving)'") == true)
        }
    }

    @Test func commitHashesPass() {
        for commit in ["abc123", "cafe01", String(repeating: "0", count: 40),
                       "deadbeef", "ABC123"] {
            #expect(ExperimentStore.isCommitLike(commit))
            var manifest = ExperimentManifest(
                name: "ok", description: "", modelID: "test/model")
            manifest.modelRevision = commit
            #expect(ExperimentStore.symbolicRevisionProblem(manifest) == nil)
        }
    }

    @Test func judgeRevisionsAreCheckedToo() throws {
        try withTempRoot {
            _ = try makeJudgePinStudy(
                "srj", judges: foreignPanel(revision: "main", dtype: "bfloat16"))
            let reason = try #require(freezeRefusal("srj"))
            #expect(reason.contains("moving reference"))
            #expect(reason.contains("judge 'judge-2' pins 'main'"))
            // The remedy points at the thing that now works (finding 5).
            #expect(reason.contains("Resolve button"))
        }
    }

    @Test func forceFreezeStampsItUnderTheRevisionGate() throws {
        try withTempRoot {
            _ = try makeJudgePinStudy(
                "srf", judges: foreignPanel(revision: "main", dtype: "bfloat16"))
            let forced = try ExperimentStore.freeze(name: "srf", force: true)
            #expect(forced.forcedGatesSkipped?.contains("revision") == true)
        }
    }

    /// Absence is the OTHER gate's concern (unpinned foreign judge); this
    /// one only judges the shape of a revision that exists.
    @Test func anAbsentRevisionIsNotThisGatesBusiness() {
        var manifest = ExperimentManifest(
            name: "abs", description: "", modelID: "test/model")
        #expect(ExperimentStore.symbolicRevisionProblem(manifest) == nil)
        manifest.modelRevision = ""
        #expect(ExperimentStore.symbolicRevisionProblem(manifest) == nil)
    }

    // MARK: - The study-level dtype pin (2026-07-24)

    /// The Mac is the AUTHORING surface, so it validates a key only the
    /// server consumes: a manifest must not reach the cluster carrying a
    /// dtype that refuses at load after a queue wait.
    @Test func freezeRefusesAnUnloadableStudyDtype() throws {
        try withTempRoot {
            var manifest = try makeJudgePinStudy(
                "sdt", judges: foreignPanel(revision: "cafe01", dtype: "bfloat16"))
            manifest.dtype = "banana"
            try ExperimentStore.save(manifest)
            let reason = try #require(freezeRefusal("sdt"))
            #expect(reason.contains("study dtype 'banana'"))
            #expect(reason.contains("can load"))
        }
    }

    /// `measurementPins` was RESERVED in the cross-engine gate vocabulary.
    /// The study dtype is the first gate to use it.
    @Test func forceFreezeStampsTheStudyDtypeUnderMeasurementPins() throws {
        try withTempRoot {
            var manifest = try makeJudgePinStudy(
                "sdf", judges: foreignPanel(revision: "cafe01", dtype: "bfloat16"))
            manifest.dtype = "banana"
            try ExperimentStore.save(manifest)
            let forced = try ExperimentStore.freeze(name: "sdf", force: true)
            #expect(forced.forcedGatesSkipped?.contains("measurementPins") == true)
        }
    }

    @Test func vocabularySpellingsFreezeAndAbsenceIsUntouched() throws {
        try withTempRoot {
            for (index, spelling) in ["bfloat16", "bf16", "fp16", "float32"]
                .enumerated()
            {
                var manifest = try makeJudgePinStudy(
                    "sdok\(index)",
                    judges: foreignPanel(revision: "cafe01", dtype: "bfloat16"))
                manifest.dtype = spelling
                try ExperimentStore.save(manifest)
                #expect(
                    try ExperimentStore.freeze(name: "sdok\(index)").status == .frozen)
            }
            // Absent = "let the device decide", the historical behaviour.
            let plain = ExperimentManifest(
                name: "sdnone", description: "", modelID: "test/model")
            #expect(ExperimentStore.unloadableStudyDtypeProblem(plain) == nil)
        }
    }

    /// Legacy manifests carry no `dtype` key; decoding must not invent one,
    /// and re-encoding must not add one (it would change the content hash).
    @Test func theStudyDtypeIsOmittedWhenNil() throws {
        let manifest = ExperimentManifest(
            name: "sdr", description: "", modelID: "test/model")
        #expect(manifest.dtype == nil)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let json = try #require(
            String(data: try encoder.encode(manifest), encoding: .utf8))
        #expect(!json.contains("\"dtype\""))

        var pinned = manifest
        pinned.dtype = "bfloat16"
        let round = try JSONDecoder().decode(
            ExperimentManifest.self, from: try encoder.encode(pinned))
        #expect(round.dtype == "bfloat16")
    }

    // MARK: - Pin prefill for the judge editor (review round 4, finding 5)

    @Test func prefillFillsBlankPinsFromTheModelCache() throws {
        let filled = try #require(ExperimentStore.judgePinPrefill(
            model: "other/judge-12b", studyModel: "test/model",
            revision: nil, dtype: nil,
            resolveRevision: { _ in "cafe01" }))
        #expect(filled.revision == "cafe01")
        #expect(filled.dtype == "bfloat16")
    }

    /// A pin the researcher set is never rewritten by re-picking the SAME
    /// model.
    @Test func prefillNeverOverwritesAPinAlreadySet() throws {
        let filled = try #require(ExperimentStore.judgePinPrefill(
            model: "other/judge-12b", previousModel: "other/judge-12b",
            studyModel: "test/model",
            revision: "mine", dtype: "float16",
            resolveRevision: { _ in "cafe01" }))
        #expect(filled.revision == "mine")
        #expect(filled.dtype == "float16")
    }

    /// Review round 5, finding 2. A revision pins nothing on its own — only
    /// a (model, revision) PAIR identifies bytes. Carrying model A's
    /// revision onto model B freezes happily and then dies on the compute
    /// node, because model B has no such commit.
    @Test func prefillClearsTheRevisionWhenTheModelChanges() throws {
        let filled = try #require(ExperimentStore.judgePinPrefill(
            model: "other/judge-B", previousModel: "other/judge-A",
            studyModel: "test/model",
            revision: "revision-of-A", dtype: "float16",
            resolveRevision: { model in
                model == "other/judge-B" ? "revision-of-B" : "revision-of-A"
            }))
        #expect(filled.revision == "revision-of-B")
        // dtype describes how to LOAD, not which bytes, so it survives.
        #expect(filled.dtype == "float16")
    }

    /// A model change to something the cache does not hold clears rather
    /// than keeping the stale pin: nil asks for the commit, the old value
    /// would be a confident lie.
    @Test func prefillClearsEvenWhenTheNewModelCannotBeResolved() throws {
        let filled = try #require(ExperimentStore.judgePinPrefill(
            model: "other/judge-B", previousModel: "other/judge-A",
            studyModel: "test/model",
            revision: "revision-of-A", dtype: nil,
            resolveRevision: { _ in nil }))
        #expect(filled.revision == nil)
    }

    /// The same notification fires when a DIFFERENT STUDY is loaded into the
    /// editor (nil/empty previous). Clearing there would wipe a pin the
    /// manifest legitimately carries.
    @Test func prefillLeavesPinsAloneWhenThereIsNoPriorModel() throws {
        for previous in [nil, "", "   "] as [String?] {
            let filled = try #require(ExperimentStore.judgePinPrefill(
                model: "other/judge-12b", previousModel: previous,
                studyModel: "test/model",
                revision: "from-the-manifest", dtype: "float16",
                resolveRevision: { _ in "cafe01" }))
            #expect(filled.revision == "from-the-manifest")
        }
    }

    @Test func prefillLeavesRevisionNilWhenTheModelIsNotCached() throws {
        let filled = try #require(ExperimentStore.judgePinPrefill(
            model: "other/judge-12b", studyModel: "test/model",
            revision: nil, dtype: nil,
            resolveRevision: { _ in nil }))
        // Nil, not a fabricated value — the row then says so and asks for
        // the commit, rather than pinning something untrue.
        #expect(filled.revision == nil)
        #expect(filled.dtype == "bfloat16")
    }

    @Test func prefillDeclinesForJudgesThatNeedNoPins() {
        // The study model inherits the study's own pin.
        #expect(ExperimentStore.judgePinPrefill(
            model: "test/model", studyModel: "test/model",
            revision: nil, dtype: nil, resolveRevision: { _ in "x" }) == nil)
        // A blank model means "study model (default)".
        #expect(ExperimentStore.judgePinPrefill(
            model: nil, studyModel: "test/model",
            revision: nil, dtype: nil, resolveRevision: { _ in "x" }) == nil)
        #expect(ExperimentStore.judgePinPrefill(
            model: "   ", studyModel: "test/model",
            revision: nil, dtype: nil, resolveRevision: { _ in "x" }) == nil)
    }

    /// What the editor prefills must be something the gate accepts, or the
    /// UI would be handing researchers a guaranteed freeze refusal.
    @Test func whatThePrefillProducesSatisfiesTheGate() throws {
        try withTempRoot {
            let filled = try #require(ExperimentStore.judgePinPrefill(
                model: "other/judge-12b", studyModel: "test/model",
                revision: nil, dtype: nil,
                resolveRevision: { _ in "cafe01" }))
            _ = try makeJudgePinStudy("fpp", judges: [
                .init(name: "judge-1", kind: "local", model: nil),
                .init(name: "judge-2", kind: "local", model: "other/judge-12b",
                      revision: filled.revision, dtype: filled.dtype),
            ])
            #expect(try ExperimentStore.freeze(name: "fpp").status == .frozen)
        }
    }

    // MARK: - Draft visibility and force provenance

    @Test func aDraftShowsTheProblemBeforeFreezeRefuses() throws {
        try withTempRoot {
            let manifest = try makeJudgePinStudy("fpv", judges: foreignPanel())
            #expect(
                ExperimentStore.freezeAdvisories(for: manifest)
                    .contains { $0.contains("pin the exact bytes") })
        }
    }

    @Test func forceFreezeSkipsItLoudlyAndStampsJudgeValidity() throws {
        try withTempRoot {
            _ = try makeJudgePinStudy("fpf", judges: foreignPanel())
            let forced = try ExperimentStore.freeze(name: "fpf", force: true)
            #expect(forced.freezeForced == true)
            #expect(forced.forcedGatesSkipped?.contains("judgeValidity") == true)
        }
    }
}
