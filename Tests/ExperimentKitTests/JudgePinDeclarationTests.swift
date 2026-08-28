import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// `pin-rubric --judge-pin` — the LOCAL-judge pins (`judges[].revision`,
/// `judges[].dtype`) that had no CLI spelling at all, and the panel merge
/// that stopped `--judges` from wiping them.
///
/// Field-discovered, and blocking a maintainer-specified design: a local
/// judge naming a model OTHER than the study model must pin both fields or
/// freeze refuses under `judgeValidity` — so a cross-substrate panel (a
/// gemma judge on a qwen study) could not be authored headlessly at all,
/// because `--judges`'s positional grammar
/// (`<name>:<kind>[:<model>[:<provider>]]`) has no room for them and no
/// other flag wrote them. Worse, `--judges` REPLACED the panel with
/// nil-revision/nil-dtype rows, so a headless re-declaration silently wiped
/// pins the app had written and the study then refused at freeze for want of
/// pins it used to have — the same silent-drop class the sweep-selection
/// merge exists to kill, one level down.
///
/// Serialized and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct JudgePinDeclarationTests {

    /// A commit-shaped revision (hex, as `isCommitLike` requires).
    static let commit = "9f3c1ab77de40521"
    static let otherCommit = "0011aaff2244cc88"
    /// The cross-substrate case: a gemma judge on a qwen study.
    static let studyModel = "Qwen/Qwen3-8B"
    static let judgeModel = "google/gemma-3-27b-it"

    // MARK: Harness

    func withTempRoot<T>(_ body: (URL) async throws -> T) async rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "judge-pin-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        WorkspaceRoot.programmaticOverride = temp
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            WorkspaceRoot.programmaticOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body(temp)
    }

    @discardableResult
    func invoke(_ args: [String]) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(sink: .discarding)
            .run(namespace: "experiment", args)
    }

    /// A draft on the STUDY model, with a rubric file on disk to pin.
    @discardableResult
    func study(_ name: String = "judged", in root: URL) async throws -> String {
        _ = try ExperimentStore.create(
            name: name, description: "", modelID: Self.studyModel,
            modelRevision: "beef0123")
        let rubric = root.appending(
            components: "prompts", "rubrics", "paired.md")
        try FileManager.default.createDirectory(
            at: rubric.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try "Which response reasons more formally?\n".write(
            to: rubric, atomically: true, encoding: .utf8)
        return "prompts/rubrics/paired.md"
    }

    func judge(
        _ manifest: ExperimentManifest, _ name: String
    ) throws -> ExperimentManifest.JudgeRef {
        try #require((manifest.judges ?? []).first { $0.name == name })
    }

    // MARK: - The blocked design, unblocked

    /// The whole point: a cross-substrate panel authored in ONE headless
    /// invocation, and the `judgeValidity` gate's own rule satisfied by what
    /// the CLI wrote.
    @Test func theCrossSubstrateJudgePanelIsAuthorableHeadlessly() async throws {
        try await withTempRoot { root in
            let rubric = try await study(in: root)
            let outcome = await invoke([
                "pin-rubric", "judged", rubric,
                "--judges", "strict:local:\(Self.judgeModel),lenient:claude",
                "--judge-pin", "strict=\(Self.commit):bfloat16",
            ])
            #expect(outcome.envelope.state == .ready)
            #expect(outcome.envelope.exitCode == 0)

            // The MANIFEST is the contract.
            let manifest = try ExperimentStore.load(name: "judged")
            let strict = try judge(manifest, "strict")
            #expect(strict.kind == "local")
            #expect(strict.model == Self.judgeModel)
            #expect(strict.revision == Self.commit)
            #expect(strict.dtype == "bfloat16")
            // The other judge is untouched — pins are per judge.
            let lenient = try judge(manifest, "lenient")
            #expect(lenient.revision == nil)
            #expect(lenient.dtype == nil)

            // The gate that blocked this design now passes on the manifest
            // the CLI authored — and the revision it wrote is a commit, not
            // a moving reference.
            #expect(
                ExperimentStore.unpinnedForeignLocalJudgeProblem(manifest) == nil)
            #expect(ExperimentStore.symbolicRevisionProblem(manifest) == nil)

            // …and the same panel WITHOUT the pin is exactly what the gate
            // refuses, so the test is measuring the gate and not itself.
            var unpinned = manifest
            unpinned.judges = [
                .init(name: "strict", kind: "local", model: Self.judgeModel),
                .init(name: "lenient", kind: "claude"),
            ]
            let refusal = try #require(
                ExperimentStore.unpinnedForeignLocalJudgeProblem(unpinned))
            #expect(refusal.contains("is missing revision and dtype"))
        }
    }

    /// The echo carries the pins it wrote — they used to be absent from the
    /// document even when the manifest held them.
    @Test func theEchoCarriesTheJudgePins() async throws {
        try await withTempRoot { root in
            let rubric = try await study(in: root)
            let outcome = await invoke([
                "pin-rubric", "judged", rubric,
                "--judges", "strict:local:\(Self.judgeModel),lenient:claude",
                "--judge-pin", "strict=\(Self.commit):float16",
            ])
            guard case .array(let judges)? = outcome.envelope.result?["judges"]
            else {
                Issue.record("no judges array in the echo")
                return
            }
            guard case .object(let strict) = judges[0] else {
                Issue.record("judge row is not an object")
                return
            }
            #expect(strict["revision"] == .string(Self.commit))
            #expect(strict["dtype"] == .string("float16"))
            // Omit-when-nil, like the manifest's own encoding.
            guard case .object(let lenient) = judges[1] else {
                Issue.record("judge row is not an object")
                return
            }
            #expect(lenient["revision"] == nil)
            #expect(lenient["dtype"] == nil)
        }
    }

    // MARK: - The wipe, and the merge that ends it

    /// THE pinned regression. Re-declaring the same roster with no
    /// `--judge-pin` used to write nil-revision/nil-dtype rows over the
    /// pins, and the study then refused at freeze for want of pins it had.
    /// Now the pins are inherited field by field, and the echo says so.
    @Test func reDeclaringTheRosterKeepsThePinsAndSaysSo() async throws {
        try await withTempRoot { root in
            let rubric = try await study(in: root)
            await invoke([
                "pin-rubric", "judged", rubric,
                "--judges", "strict:local:\(Self.judgeModel),lenient:claude",
                "--judge-pin", "strict=\(Self.commit):bfloat16",
            ])
            // A second declaration that says nothing about the pins.
            let outcome = await invoke([
                "pin-rubric", "judged", rubric,
                "--judges", "strict:local:\(Self.judgeModel),lenient:claude",
            ])
            #expect(outcome.envelope.exitCode == 0)
            let manifest = try ExperimentStore.load(name: "judged")
            let strict = try judge(manifest, "strict")
            #expect(strict.revision == Self.commit, "the revision pin was wiped")
            #expect(strict.dtype == "bfloat16", "the dtype pin was wiped")
            #expect(
                ExperimentStore.unpinnedForeignLocalJudgeProblem(manifest) == nil)

            // A merge is honest when it is said out loud — the same key the
            // sweep-selection merge echoes under.
            guard
                case .array(let kept)? = outcome.envelope
                    .result?["inheritedFromExistingDeclaration"]
            else {
                Issue.record("the inheritance was silent")
                return
            }
            #expect(kept.count == 2)
            #expect(
                outcome.envelope.message.contains(
                    "kept from the existing declaration"))
        }
    }

    /// A pin declared in the same breath WINS over the inherited one — the
    /// merge fills what was not declared, never overrides what was.
    @Test func aDeclaredPinOverridesTheInheritedOne() async throws {
        try await withTempRoot { root in
            let rubric = try await study(in: root)
            await invoke([
                "pin-rubric", "judged", rubric,
                "--judges", "strict:local:\(Self.judgeModel)",
                "--judge-pin", "strict=\(Self.commit):bfloat16",
            ])
            await invoke([
                "pin-rubric", "judged", rubric,
                "--judges", "strict:local:\(Self.judgeModel)",
                "--judge-pin", "strict=\(Self.otherCommit)",
            ])
            let strict = try judge(
                try ExperimentStore.load(name: "judged"), "strict")
            #expect(strict.revision == Self.otherCommit)
            // The dtype was not re-declared, so it is inherited, not reset.
            #expect(strict.dtype == "bfloat16")
        }
    }

    /// A judge whose MODEL changed drops its pins — they identify the old
    /// bytes — and the drop is named, exactly as a re-declared
    /// choice-prompts file names the pin that goes with it.
    @Test func changingAJudgesModelDropsItsPinsOutLoud() async throws {
        try await withTempRoot { root in
            let rubric = try await study(in: root)
            await invoke([
                "pin-rubric", "judged", rubric,
                "--judges", "strict:local:\(Self.judgeModel)",
                "--judge-pin", "strict=\(Self.commit):bfloat16",
            ])
            let outcome = await invoke([
                "pin-rubric", "judged", rubric,
                "--judges", "strict:local:other/judge-12b",
            ])
            #expect(outcome.envelope.exitCode == 0)
            let strict = try judge(
                try ExperimentStore.load(name: "judged"), "strict")
            #expect(strict.revision == nil)
            #expect(strict.dtype == nil)
            guard
                case .array(let notes)? = outcome.envelope
                    .result?["inheritedFromExistingDeclaration"]
            else {
                Issue.record("the drop was silent")
                return
            }
            #expect(
                notes.contains(
                    .string(
                        "dropped judge 'strict' revision/dtype pins — it now "
                            + "names a different model")))
        }
    }

    /// A judge that stops being local drops them too, and says which reason.
    @Test func aJudgeThatStopsBeingLocalDropsItsPinsOutLoud() async throws {
        try await withTempRoot { root in
            let rubric = try await study(in: root)
            await invoke([
                "pin-rubric", "judged", rubric,
                "--judges", "strict:local:\(Self.judgeModel)",
                "--judge-pin", "strict=\(Self.commit):bfloat16",
            ])
            let outcome = await invoke([
                "pin-rubric", "judged", rubric, "--judges", "strict:claude",
            ])
            #expect(outcome.envelope.exitCode == 0)
            let strict = try judge(
                try ExperimentStore.load(name: "judged"), "strict")
            #expect(strict.revision == nil)
            #expect(strict.dtype == nil)
            guard
                case .array(let notes)? = outcome.envelope
                    .result?["inheritedFromExistingDeclaration"]
            else {
                Issue.record("the drop was silent")
                return
            }
            #expect(
                notes.contains(
                    .string(
                        "dropped judge 'strict' revision/dtype pins — it is no "
                            + "longer a local judge")))
        }
    }

    /// `--judge-pin` with no `--judges` pins the panel that is already
    /// there — the follow-up a panel authored in the app needs.
    @Test func aPinAloneAppliesToTheExistingPanel() async throws {
        try await withTempRoot { root in
            let rubric = try await study(in: root)
            await invoke([
                "pin-rubric", "judged", rubric,
                "--judges", "strict:local:\(Self.judgeModel),lenient:claude",
            ])
            #expect(
                try judge(
                    try ExperimentStore.load(name: "judged"), "strict"
                ).revision == nil)
            let outcome = await invoke([
                "pin-rubric", "judged", rubric,
                "--judge-pin", "strict=\(Self.commit):bf16",
            ])
            #expect(outcome.envelope.exitCode == 0)
            let manifest = try ExperimentStore.load(name: "judged")
            #expect(manifest.judges?.map(\.name) == ["strict", "lenient"])
            let strict = try judge(manifest, "strict")
            #expect(strict.revision == Self.commit)
            // An ALIAS is accepted and stored canonically, so every manifest
            // spells the same precision the same way.
            #expect(strict.dtype == "bfloat16")
        }
    }

    // MARK: - Refusals

    /// A pin aimed at no declared judge is malformed, never silently
    /// attached to nothing.
    @Test func aPinNamingNoDeclaredJudgeIsMalformed() async throws {
        try await withTempRoot { root in
            let rubric = try await study(in: root)
            let outcome = await invoke([
                "pin-rubric", "judged", rubric,
                "--judges", "strict:local:\(Self.judgeModel)",
                "--judge-pin", "strikt=\(Self.commit)",
            ])
            #expect(outcome.envelope.state == .blocked)
            #expect(outcome.envelope.exitCode == 64)
            #expect(outcome.envelope.error?.code == "usage")
            #expect(
                outcome.envelope.error?.reason
                    == "--judge-pin 'strikt' names no judge in the panel — "
                    + "declared: strict")
            // Nothing was written — not even the rubric half.
            #expect(try ExperimentStore.load(name: "judged").judges == nil)
        }
    }

    /// `keepingKindOwnedFields` would drop a claude judge's pins without a
    /// word; refuse instead of writing a pin that evaporates.
    @Test func aPinOnANonLocalJudgeIsMalformed() async throws {
        try await withTempRoot { root in
            let rubric = try await study(in: root)
            let outcome = await invoke([
                "pin-rubric", "judged", rubric, "--judges", "lenient:claude",
                "--judge-pin", "lenient=\(Self.commit)",
            ])
            #expect(outcome.envelope.exitCode == 64)
            let reason = try #require(outcome.envelope.error?.reason)
            #expect(
                reason == "judge 'lenient' is claude, which pins no revision "
                    + "or dtype — those are local-judge pins (a claude judge's "
                    + "identity is its model slug)")
            #expect(try ExperimentStore.load(name: "judged").judges == nil)
        }
    }

    /// The revision rule freeze applies, said at the declaration: a branch
    /// is re-pointed by definition, so it identifies nothing.
    @Test func aSymbolicRevisionIsRefusedAtTheDeclaration() async throws {
        try await withTempRoot { root in
            let rubric = try await study(in: root)
            let outcome = await invoke([
                "pin-rubric", "judged", rubric,
                "--judges", "strict:local:\(Self.judgeModel)",
                "--judge-pin", "strict=main:bfloat16",
            ])
            #expect(outcome.envelope.exitCode == 64)
            let reason = try #require(outcome.envelope.error?.reason)
            #expect(
                reason == "judge pin 'strict' names revision 'main', which is "
                    + "not a commit hash — a branch or tag is re-pointed by "
                    + "definition, so it cannot identify the weights a run used")
            #expect(try ExperimentStore.load(name: "judged").judges == nil)
        }
    }

    /// A dtype outside the loader's vocabulary is refused with the loader's
    /// own words — an unrecognized value used to load float32 silently.
    @Test func anUnknownDtypeIsRefusedWithTheLoadersVocabulary() async throws {
        try await withTempRoot { root in
            let rubric = try await study(in: root)
            let outcome = await invoke([
                "pin-rubric", "judged", rubric,
                "--judges", "strict:local:\(Self.judgeModel)",
                "--judge-pin", "strict=\(Self.commit):int8",
            ])
            #expect(outcome.envelope.exitCode == 64)
            let reason = try #require(outcome.envelope.error?.reason)
            #expect(reason.hasPrefix("unknown judge dtype 'int8' — "))
            for dtype in ExperimentStore.judgeDtypeVocabulary {
                #expect(reason.contains(dtype), "reason omits \(dtype)")
            }
            let repair = try #require(outcome.envelope.error?.repairAction)
            #expect(repair.contains("bfloat16|float16|float32"))
        }
    }

    /// The shape refusals, on the pure parser.
    @Test func theJudgePinGrammarRefusesWhatItCannotRead() throws {
        for raw in ["strict", "=abc123", "strict=", "strict=abc:bf16:extra"] {
            #expect {
                _ = try ExperimentCLIRunner.parseJudgePin(raw)
            } throws: { error in
                (error as? ExperimentError)?.malformedInvocation != nil
            }
        }
        // The FIRST `=` splits, like `--seat`.
        let parsed = try ExperimentCLIRunner.parseJudgePin("strict=abc123")
        #expect(parsed == .init(name: "strict", revision: "abc123", dtype: nil))
    }

    /// A one-judge panel is a legal DESIGN (maintainer ruling, 2026-08-28),
    /// so `pin-rubric` advises about the cost rather than about a gate that
    /// no longer refuses it — and the sentence is the one freeze says.
    @Test func aSingleJudgePanelAdvisesAboutTheCostNotAGate() async throws {
        try await withTempRoot { root in
            let rubric = try await study(in: root)
            let solo = await invoke([
                "pin-rubric", "judged", rubric, "--judges", "solo:claude",
            ])
            #expect(solo.envelope.exitCode == 0)
            let advisories = solo.envelope.advisories ?? []
            #expect(
                advisories.contains {
                    $0.code == CLIAdvisory.judgePanelTooSmall.rawValue
                        && $0.detail
                            == ExperimentStore.singleJudgePanelAdvisoryText
                })
            // The old wording named a gate requirement that is now false.
            #expect(
                !(advisories.first?.detail.contains("requires at least 2")
                    ?? false))
            // Two judges say nothing.
            let pair = await invoke([
                "pin-rubric", "judged", rubric,
                "--judges", "solo:claude,other:local:\(Self.judgeModel)",
                "--judge-pin", "other=\(Self.commit):bfloat16",
            ])
            #expect(pair.envelope.advisories?.isEmpty != false)
        }
    }

    /// The `--judges` grammar is untouched by the new flag: its fourth field
    /// is still, and only, OpenRouter's serving provider.
    @Test func theJudgesGrammarIsUnchanged() throws {
        let panel = try ExperimentCLIRunner.parseJudges(
            "a:claude,b:local:\(Self.judgeModel),c:openrouter:v/m:together")
        #expect(panel.map(\.name) == ["a", "b", "c"])
        #expect(panel[2].provider == "together")
        #expect(panel.allSatisfy { $0.revision == nil && $0.dtype == nil })
        #expect {
            _ = try ExperimentCLIRunner.parseJudges(
                "b:local:\(Self.judgeModel):\(Self.commit)")
        } throws: { error in
            ((error as? ExperimentError)?.reason ?? "")
                .contains("which pins no serving provider")
        }
    }
}
