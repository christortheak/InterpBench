import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// Two-phase Claude-judged sweeps, Mac side (key-custody design
/// 2026-07-18): the runner judges every (packet × pinned judge) pair with
/// an injectable judge, never sees cell identity or orientation, and the
/// completion payload carries exactly the server's contract keys.
struct SweepJudgmentTests {

    private func packet(_ id: String, a: String, b: String)
        -> ClusterClient.SweepJudgingPacket
    {
        ClusterClient.SweepJudgingPacket(
            packetID: id, prompt: "Write about the town.",
            responseA: a, responseB: b)
    }

    private func judgeEntry(
        _ name: String, model: String? = "claude-pinned"
    ) -> ClusterClient.AwaitingSweepJudgment.JudgeEntry {
        ClusterClient.AwaitingSweepJudgment.JudgeEntry(
            name: name, kind: "claude", model: model)
    }

    /// Winner-only fake verdict (most tests only steer the winner).
    private static func verdict(
        _ winner: String, confidence: Double = 0.9
    ) -> PairedJudgeResponse {
        PairedJudgeResponse(
            winner: winner, confidence: confidence, briefReason: "because")
    }

    @Test func judgesEveryPacketWithEveryPinnedJudge() async throws {
        let packets = [packet("p1", a: "calm", b: "dread"),
                       packet("p2", a: "dread", b: "calm")]
        let judges = [judgeEntry("j1"), judgeEntry("j2", model: "claude-x")]
        // A content-keyed fake: prefers "dread" wherever it sits.
        let entries = try await SweepJudgmentRunner.judge(
            packets: packets, judges: judges, rubric: "more dread wins",
            using: { _, _, _, _, a, b in
                if a.contains("dread") { return Self.verdict("A") }
                if b.contains("dread") { return Self.verdict("B") }
                return Self.verdict("tie")
            })
        #expect(entries.count == 4)  // 2 packets × 2 judges — full coverage
        #expect(Set(entries.map(\.judge)) == ["j1", "j2"])
        // The emission-pinned model is stamped into every judgment
        // verbatim — this Mac never substitutes its own default.
        let models = Dictionary(grouping: entries, by: \.judge)
            .mapValues { Set($0.compactMap(\.model)) }
        #expect(models["j2"] == ["claude-x"])
        #expect(models["j1"] == ["claude-pinned"])
        let byKey = Dictionary(
            uniqueKeysWithValues: entries.map {
                ("\($0.judge)|\($0.packetID)", $0.winner)
            })
        // The runner passes responses through untouched: the fake's verdict
        // tracks CONTENT position, proving no re-blinding happens here (the
        // server's map is the only unblinding authority).
        #expect(byKey["j1|p1"] == "B")
        #expect(byKey["j1|p2"] == "A")
    }

    @Test func emptyJudgePanelRefuses() async {
        do {
            _ = try await SweepJudgmentRunner.judge(
                packets: [packet("p", a: "a", b: "b")], judges: [],
                rubric: "r", using: { _, _, _, _, _, _ in Self.verdict("tie") })
            Issue.record("expected a refusal for an empty judge panel")
        } catch let error as PairedJudgeError {
            #expect(error.reason.contains("no judges"))
        } catch {
            Issue.record("unexpected error \(error)")
        }
    }

    @Test func judgmentEntriesEncodeTheServerContract() throws {
        let entry = ClusterClient.SweepJudgmentEntry(
            packetID: "abc", judge: "opus-judge", winner: "tie",
            model: "claude-x")
        let data = try JSONEncoder().encode(entry)
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(object?.keys.sorted() == ["judge", "model", "packetID", "winner"])
        // model is optional on the wire (older payloads omit it).
        let bare = try JSONEncoder().encode(ClusterClient.SweepJudgmentEntry(
            packetID: "abc", judge: "opus-judge", winner: "tie"))
        let bareObject = try JSONSerialization.jsonObject(with: bare) as? [String: Any]
        #expect(bareObject?.keys.sorted() == ["judge", "packetID", "winner"])
        // OpenRouter judgments additionally stamp the verified provider
        // under the server's contract key.
        let orData = try JSONEncoder().encode(ClusterClient.SweepJudgmentEntry(
            packetID: "abc", judge: "or-judge", winner: "A",
            model: "google/gemma-3-27b-it", provider: "DeepInfra"))
        let orObject = try JSONSerialization.jsonObject(with: orData) as? [String: Any]
        #expect(orObject?.keys.sorted()
            == ["judge", "model", "packetID", "provider", "winner"])
        #expect(orObject?["provider"] as? String == "DeepInfra")
        // Winner-only closure (2026-07-20): a full verdict rides under the
        // server's "judgment" key (snake_case verdict fields — the inline
        // path's shape) plus a top-level "confidence".
        let fullData = try JSONEncoder().encode(ClusterClient.SweepJudgmentEntry(
            packetID: "abc", judge: "opus-judge", winner: "A",
            model: "claude-x", confidence: 0.8,
            judgment: PairedJudgeResponse(
                aScores: ["dread": 4], winner: "A", confidence: 0.8,
                briefReason: "more dread")))
        let fullObject = try JSONSerialization.jsonObject(with: fullData) as? [String: Any]
        #expect(fullObject?.keys.sorted() == [
            "confidence", "judge", "judgment", "model", "packetID", "winner",
        ])
        let payload = fullObject?["judgment"] as? [String: Any]
        #expect(payload?.keys.sorted() == [
            "a_scores", "brief_reason", "confidence", "winner",
        ])
        #expect(payload?["winner"] as? String == "A")
    }

    @Test func fullVerdictsRideWithTheJudgments() async throws {
        // Winner-only closure (2026-07-20): the runner keeps the judge's
        // whole verdict — confidence and payload land on every entry, and
        // the payload's winner always agrees with the recorded winner (the
        // completion verb refuses inconsistency).
        let entries = try await SweepJudgmentRunner.judge(
            packets: [packet("p1", a: "calm", b: "dread")],
            judges: [judgeEntry("j1")], rubric: "more dread wins",
            using: { _, _, _, _, _, _ in
                PairedJudgeResponse(
                    aScores: ["dread": 1], bScores: ["dread": 4],
                    winner: "B", confidence: 0.85, briefReason: "B has dread")
            })
        let entry = try #require(entries.first)
        #expect(entry.winner == "B")
        #expect(entry.confidence == 0.85)
        #expect(entry.judgment?.briefReason == "B has dread")
        #expect(entry.judgment?.winner == entry.winner)
    }

    @Test func invalidVerdictRetriesOnceThenRefuses() async throws {
        // Invalid-verdict closure (2026-07-20): an out-of-vocabulary winner
        // used to be silently recorded as a substantive tie — invented data
        // that corrupts the preference mean. Judges are LLMs, so ONE
        // malformed response retries…
        let flaky = CallCounter()
        let recovered = try await SweepJudgmentRunner.judge(
            packets: [packet("p1", a: "calm", b: "dread")],
            judges: [judgeEntry("j1")], rubric: "r",
            using: { _, _, _, _, _, _ in
                await flaky.next() == 1 ? Self.verdict("both") : Self.verdict("B")
            })
        #expect(await flaky.value == 2)  // retried exactly once
        #expect(recovered.first?.winner == "B")
        #expect(recovered.first?.judgment?.winner == "B")
        #expect(recovered.first?.confidence == 0.9)

        // …and a second invalid verdict REFUSES the whole phase: the throw
        // happens before any completion verb runs, so nothing is recorded
        // and the run can be re-judged.
        let stubborn = CallCounter()
        do {
            _ = try await SweepJudgmentRunner.judge(
                packets: [packet("p1", a: "calm", b: "dread")],
                judges: [judgeEntry("j1")], rubric: "r",
                using: { _, _, _, _, _, _ in
                    _ = await stubborn.next()
                    return Self.verdict("both")
                })
            Issue.record("expected a refusal after two invalid verdicts")
        } catch let error as PairedJudgeError {
            #expect(error.reason.contains("judge 'j1'"))
            #expect(error.reason.contains("invalid verdict twice"))
            #expect(error.reason.contains("p1"))
            #expect(error.reason.contains("'both'"))
            #expect(error.reason.contains("expected A, B, or tie"))
        } catch {
            Issue.record("unexpected error \(error)")
        }
        #expect(await stubborn.value == 2)  // one retry, never a third call
    }

    @Test func preflightFailsClosedBeforeAnyAPISpend() async {
        // Engineer review 2026-07-18: every malformed input refuses BEFORE
        // the judge callable runs — a judge call is money.
        let spent = SpentFlag()
        let judgeCall: SweepJudgmentRunner.JudgeCall = { _, _, _, _, _, _ in
            await spent.mark()
            return Self.verdict("tie")
        }
        let packets = [packet("p1", a: "a", b: "b")]
        // Nameless judge.
        await #expect(throws: PairedJudgeError.self) {
            _ = try await SweepJudgmentRunner.judge(
                packets: packets, judges: [judgeEntry("  ")], rubric: "r",
                using: judgeCall)
        }
        // Duplicate judge names.
        await #expect(throws: PairedJudgeError.self) {
            _ = try await SweepJudgmentRunner.judge(
                packets: packets,
                judges: [judgeEntry("j1"), judgeEntry("j1")], rubric: "r",
                using: judgeCall)
        }
        // A local judge in a deferred panel is malformed by construction.
        await #expect(throws: PairedJudgeError.self) {
            _ = try await SweepJudgmentRunner.judge(
                packets: packets,
                judges: [ClusterClient.AwaitingSweepJudgment.JudgeEntry(
                    name: "j1", kind: "local", model: "m")],
                rubric: "r", using: judgeCall)
        }
        // A missing/unknown KIND refuses — never shrugged past.
        await #expect(throws: PairedJudgeError.self) {
            _ = try await SweepJudgmentRunner.judge(
                packets: packets,
                judges: [ClusterClient.AwaitingSweepJudgment.JudgeEntry(
                    name: "j1", kind: nil, model: "m")],
                rubric: "r", using: judgeCall)
        }
        // A missing pinned MODEL refuses — this Mac never substitutes its
        // own default (the sweep pins the model at emission).
        await #expect(throws: PairedJudgeError.self) {
            _ = try await SweepJudgmentRunner.judge(
                packets: packets, judges: [judgeEntry("j1", model: nil)],
                rubric: "r", using: judgeCall)
        }
        // An openrouter judge without its pinned PROVIDER refuses — an
        // unpinned provider is not a pinned judge (2026-07-19).
        await #expect(throws: PairedJudgeError.self) {
            _ = try await SweepJudgmentRunner.judge(
                packets: packets,
                judges: [ClusterClient.AwaitingSweepJudgment.JudgeEntry(
                    name: "or", kind: "openrouter",
                    model: "anthropic/claude-opus-4.8")],
                rubric: "r", using: judgeCall)
        }
        #expect(await spent.value == false)
    }

    @Test func openRouterJudgesDispatchWithTheirPins() async throws {
        // Deferred panels are external-only: claude AND openrouter kinds
        // judge side by side, each call receiving its OWN entry so the
        // dispatch can honor per-judge model/provider pins (2026-07-19).
        let judges = [
            judgeEntry("claude-j", model: "claude-pinned"),
            ClusterClient.AwaitingSweepJudgment.JudgeEntry(
                name: "or-j", kind: "openrouter",
                model: "google/gemma-3-27b-it", provider: "DeepInfra"),
        ]
        let entries = try await SweepJudgmentRunner.judge(
            packets: [packet("p1", a: "calm", b: "dread")], judges: judges,
            rubric: "r",
            using: { judgeRef, _, _, _, _, _ in
                Self.verdict(judgeRef.kind == "openrouter" ? "B" : "A")
            })
        let byJudge = Dictionary(
            uniqueKeysWithValues: entries.map { ($0.judge, $0) })
        #expect(byJudge["claude-j"]?.winner == "A")
        #expect(byJudge["or-j"]?.winner == "B")
        // Each judgment stamps ITS entry's emission-pinned model — and, for
        // openrouter, the emission-pinned PROVIDER (the completion verb
        // verifies both; a recorded string is provenance only if verified).
        #expect(byJudge["or-j"]?.model == "google/gemma-3-27b-it")
        #expect(byJudge["claude-j"]?.model == "claude-pinned")
        #expect(byJudge["or-j"]?.provider == "deepinfra")
        #expect(byJudge["claude-j"]?.provider == nil)
    }

    @Test func evaluateAwaitingRecordDecodesAndRoutes() throws {
        // Stage 2 (2026-07-19): evaluate awaiting runs share the record
        // shape plus kind/sourceRun/structured-prompt pins.
        let json = #"""
            {"run": "20260719T010203000-exp-ev-evaluate", "kind": "evaluate",
             "sourceRun": "20260719T000000000-exp-ev-run", "packetCount": 2,
             "judges": [{"name": "or-judge", "kind": "openrouter",
                         "model": "anthropic/claude-opus-4.8",
                         "provider": "Anthropic"}],
             "rubric": "r", "rubricTextSha256": "aa",
             "structuredPrompt": "compare severity",
             "structuredPromptSha256": "bb",
             "packetsSha256": "cc"}
            """#
        let awaiting = try JSONDecoder().decode(
            ClusterClient.AwaitingSweepJudgment.self, from: Data(json.utf8))
        #expect(awaiting.isEvaluate)
        #expect(awaiting.sourceRun == "20260719T000000000-exp-ev-run")
        #expect(awaiting.structuredPrompt == "compare severity")
        #expect(awaiting.judges?.first?.provider == "Anthropic")
        // Sweep records (no kind) stay non-evaluate.
        let sweep = try JSONDecoder().decode(
            ClusterClient.AwaitingSweepJudgment.self,
            from: Data(#"{"run": "x"}"#.utf8))
        #expect(!sweep.isEvaluate)
    }

    @Test func structuredPromptPinIsVerifiedBeforeAnySpend() async {
        // A drifted (or unpinned) structured prompt refuses BEFORE the
        // packets are even fetched — it is part of the criterion.
        let client = ClusterClient(profile: ClusterConnectionProfile(
            baseURL: URL(string: "http://127.0.0.1:1")!))
        let rubric = "more dread wins"
        var awaiting = ClusterClient.AwaitingSweepJudgment(
            run: "r1", kind: "evaluate")
        awaiting.rubric = rubric
        awaiting.rubricTextSha256 = SHA256.hash(data: Data(rubric.utf8))
            .map { String(format: "%02x", $0) }.joined()
        awaiting.structuredPrompt = "compare severity"
        awaiting.structuredPromptSha256 = "not-the-hash"
        do {
            _ = try await SweepJudgmentRunner.judgeAndComplete(
                client: client, experiment: "ev", awaiting: awaiting,
                using: { _, _, _, _, _, _ in Self.verdict("tie") })
            Issue.record("expected a structured-prompt pin refusal")
        } catch let error as PairedJudgeError {
            #expect(error.reason.contains("structured prompt"))
        } catch {
            let detail = "unexpected error \(error) — a pin refusal must "
                + "fire before any network call"
            Issue.record(Comment(rawValue: detail))
        }
    }

    @Test func awaitingRecordDecodesTheServerShape() throws {
        let json = #"""
            {"run": "20260718T010203000-exp-js-sweep", "packetCount": 24,
             "judges": [{"name": "opus-judge", "kind": "claude"}],
             "rubricFile": "prompts/rubrics/r.md", "rubricHash": "aa",
             "rubricTextSha256": "dd",
             "rubric": "Which response expresses more dread?",
             "experimentHash": "bb", "packetsFile": "judging-packets.jsonl",
             "packetsSha256": "cc"}
            """#
        let awaiting = try JSONDecoder().decode(
            ClusterClient.AwaitingSweepJudgment.self, from: Data(json.utf8))
        #expect(awaiting.packetCount == 24)
        #expect(awaiting.judges?.first?.name == "opus-judge")
        #expect(awaiting.rubric?.contains("dread") == true)
        #expect(awaiting.rubricTextSha256 == "dd")
    }
}

/// Actor-isolated flag so a @Sendable judge fake can record it was called.
private actor SpentFlag {
    private(set) var value = false
    func mark() { value = true }
}

/// Actor-isolated call counter so a @Sendable judge fake can vary its
/// verdict per call (garbage-once-then-valid retry tests).
private actor CallCounter {
    private(set) var value = 0
    func next() -> Int {
        value += 1
        return value
    }
}
