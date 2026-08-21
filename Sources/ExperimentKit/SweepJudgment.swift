import CryptoKit
import Foundation

// Two-phase Claude-judged sweeps (key-custody design 2026-07-18): the
// cluster GENERATES and emits blinded, hash-pinned judging packets (it holds
// no Anthropic credential by policy); THIS Mac judges them with the Keychain
// key; the server's completion verb verifies every pin and replays the
// selection. The judging client never sees cell identity or orientation —
// the packets carry only prompt + responseA/responseB.

extension ClusterClient {

    /// One sweep run awaiting Mac-side judgment (`GET
    /// /api/experiment/{name}/sweep/awaiting`).
    public struct AwaitingSweepJudgment: Codable, Sendable, Identifiable {
        public struct JudgeEntry: Codable, Sendable {
            public var name: String?
            public var kind: String?
            public var model: String?
            /// OpenRouter judges only: the provider pinned at emission —
            /// the same slug can be served by different backends with
            /// different outputs, so an unpinned provider is refused.
            public var provider: String?

            public init(name: String?, kind: String?, model: String?,
                        provider: String? = nil) {
                self.name = name
                self.kind = kind
                self.model = model
                self.provider = provider
            }
        }

        public var run: String
        /// "sweep" (or absent — the only kind before 2026-07-19) or
        /// "evaluate": which completion verb consumes the judgments.
        public var kind: String?
        /// Evaluate awaiting runs only: the generation run judged.
        public var sourceRun: String?
        public var packetCount: Int?
        public var judges: [JudgeEntry]?
        public var rubricFile: String?
        public var rubricHash: String?
        /// Hash of the EXACT rubric text embedded in the manifest — the pin
        /// this Mac verifies before judging (the file pin can differ across
        /// newline conventions).
        public var rubricTextSha256: String?
        public var rubric: String?
        /// Evaluate awaiting runs only: the pinned structured comparison
        /// prompt (part of the evaluation criterion, hash-verified like the
        /// rubric before any judge reads it).
        public var structuredPrompt: String?
        public var structuredPromptSha256: String?
        public var experimentHash: String?
        public var packetsFile: String?
        public var packetsSha256: String?
        public var id: String { run }

        public var isEvaluate: Bool { kind == "evaluate" }
    }

    /// A blinded comparison packet — deliberately identity-free.
    public struct SweepJudgingPacket: Codable, Sendable {
        public var packetID: String
        public var prompt: String
        public var responseA: String
        public var responseB: String
    }

    public struct SweepJudgmentEntry: Codable, Sendable {
        public var packetID: String
        public var judge: String
        public var winner: String
        /// The RESOLVED Anthropic model this judgment actually used —
        /// stamped into provenance so "no model set → app default" is a
        /// recorded fact, not an ambient one (engineer review 2026-07-18).
        public var model: String?
        /// OpenRouter judges only: the emission-pinned serving provider the
        /// judging client verified against the response — stamped per
        /// judgment and re-verified by the completion verb (a recorded
        /// string is provenance only if verified).
        public var provider: String?
        /// The judge's confidence, lifted from the verdict (winner-only
        /// closure 2026-07-20). Optional on the wire — older payloads omit
        /// it and the completion verb still accepts them.
        public var confidence: Double?
        /// The judge's FULL verdict (scores, structured fields, confidence,
        /// brief reason — the same object the server's inline path records
        /// as `judgment`), so deferred judgment artifacts stop being
        /// winner-only. The completion verb verifies the payload's own
        /// winner against `winner` and refuses an inconsistent verdict.
        public var judgment: PairedJudgeResponse?

        public init(packetID: String, judge: String, winner: String,
                    model: String? = nil, provider: String? = nil,
                    confidence: Double? = nil,
                    judgment: PairedJudgeResponse? = nil) {
            self.packetID = packetID
            self.judge = judge
            self.winner = winner
            self.model = model
            self.provider = provider
            self.confidence = confidence
            self.judgment = judgment
        }
    }

    public func awaitingSweepJudgments(
        experiment: String
    ) async throws -> [AwaitingSweepJudgment] {
        struct Response: Decodable { var awaiting: [AwaitingSweepJudgment] }
        let response: Response =
            try await get("/api/experiment/\(experiment)/sweep/awaiting")
        return response.awaiting
    }

    /// Deferred EVALUATE runs awaiting Mac-side judgment (`GET
    /// /api/experiment/{name}/evaluate/awaiting`) — same record shape as
    /// sweeps plus `sourceRun` and the structured-prompt pin. Older servers
    /// without the route surface as an error the caller treats as "none".
    public func awaitingEvaluateJudgments(
        experiment: String
    ) async throws -> [AwaitingSweepJudgment] {
        struct Response: Decodable { var awaiting: [AwaitingSweepJudgment] }
        let response: Response =
            try await get("/api/experiment/\(experiment)/evaluate/awaiting")
        return response.awaiting
    }

    /// Download + parse a sweep run's judging packets, verifying the bytes
    /// against the server-stamped SHA-256 BEFORE judging anything — the Mac
    /// must never spend Anthropic calls on tampered or truncated packets.
    public func sweepJudgingPackets(
        awaiting: AwaitingSweepJudgment
    ) async throws -> [SweepJudgingPacket] {
        let file = awaiting.packetsFile ?? "judging-packets.jsonl"
        let temp = FileManager.default.temporaryDirectory
            .appendingPathComponent("steerlab-judging-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temp) }
        let local = try await downloadArtifact(
            path: "runs/\(awaiting.run)/\(file)", to: temp)
        let data = try Data(contentsOf: local)
        // Fail CLOSED before any Anthropic spend (engineer review
        // 2026-07-18): a missing pin is a refusal, never a shrug.
        guard let expected = awaiting.packetsSha256 else {
            throw ClientError.badResponse(
                409,
                "the awaiting run carries no packet hash pin — it predates "
                    + "full artifact pinning; re-run the sweep")
        }
        let digest = SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }.joined()
        guard digest == expected else {
            throw ClientError.badResponse(
                409,
                "judging packets failed their hash pin (expected "
                    + "\(expected.prefix(12))…, got \(digest.prefix(12))…) "
                    + "— refusing to judge drifted packets")
        }
        let decoder = JSONDecoder()
        var packets: [SweepJudgingPacket] = []
        var ids = Set<String>()
        for line in String(decoding: data, as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: true)
        {
            let packet = try decoder.decode(
                SweepJudgingPacket.self, from: Data(line.utf8))
            guard ids.insert(packet.packetID).inserted else {
                throw ClientError.badResponse(
                    409, "duplicate packetID \(packet.packetID.prefix(12))… "
                        + "in the judging packets — refusing")
            }
            packets.append(packet)
        }
        guard let count = awaiting.packetCount else {
            throw ClientError.badResponse(
                409, "the awaiting run carries no packet count — it "
                    + "predates full artifact pinning; re-run the sweep")
        }
        guard count == packets.count else {
            throw ClientError.badResponse(
                409,
                "packet count mismatch: the manifest says \(count), the "
                    + "file holds \(packets.count) — refusing")
        }
        return packets
    }

    /// `POST /api/experiment/{name}/sweep/complete-judgment` — the server
    /// verifies every pin (packet hash, judge names, coverage, experiment
    /// epoch) and replays the selection. Returns the judgment run directory.
    public func completeSweepJudgment(
        experiment: String, sweepRun: String,
        judgments: [SweepJudgmentEntry]
    ) async throws -> String {
        struct Body: Encodable {
            var sweepRun: String
            var judgments: [SweepJudgmentEntry]
        }
        struct Response: Decodable { var runDirectory: String }
        let response: Response = try await post(
            "/api/experiment/\(experiment)/sweep/complete-judgment",
            body: Body(sweepRun: sweepRun, judgments: judgments))
        return response.runDirectory
    }

    /// `POST /api/experiment/{name}/evaluate/complete-judgment` — verifies
    /// every pin and aggregates the same judgments.jsonl + judge-report.json
    /// the inline evaluate path writes. Returns the judgment run directory.
    public func completeEvaluateJudgment(
        experiment: String, evaluateRun: String,
        judgments: [SweepJudgmentEntry]
    ) async throws -> String {
        struct Body: Encodable {
            var evaluateRun: String
            var judgments: [SweepJudgmentEntry]
        }
        struct Response: Decodable { var runDirectory: String }
        let response: Response = try await post(
            "/api/experiment/\(experiment)/evaluate/complete-judgment",
            body: Body(evaluateRun: evaluateRun, judgments: judgments))
        return response.runDirectory
    }
}

/// Phase-2 driver: judge every packet with every pinned EXTERNAL judge —
/// Claude via the Keychain key, OpenRouter via the stored judge key — then
/// hand the judgments to the server's completion verb. Pure orchestration:
/// the judge callable is injectable so tests run without the network.
public enum SweepJudgmentRunner {

    /// Returns the judge's FULL verdict (winner-only closure 2026-07-20:
    /// the runner keeps the whole verdict, so scores/confidence/reason
    /// survive into the completion payload instead of being discarded at
    /// this boundary). An out-of-vocabulary `winner` is retried once and
    /// then REFUSES the phase (invalid-verdict closure 2026-07-20) — a tie
    /// is substantive data and is never invented by `judge(...)`.
    public typealias JudgeCall = @Sendable (
        _ judge: ClusterClient.AwaitingSweepJudgment.JudgeEntry,
        _ rubric: String, _ structuredPrompt: String?, _ prompt: String,
        _ responseA: String, _ responseB: String
    ) async throws -> PairedJudgeResponse

    /// The production judge: dispatch by the entry's PINNED kind —
    /// `ClaudePairedJudge` (Keychain Claude key) or `OpenRouterPairedJudge`
    /// (stored external judge key, provider-pinned). Preflight guarantees
    /// the pins before this ever runs.
    public static let liveJudge: JudgeCall = {
        judgeRef, rubric, structuredPrompt, prompt, a, b in
        if judgeRef.kind == "openrouter" {
            return try await OpenRouterPairedJudge.judge(
                model: judgeRef.model ?? "",
                provider: judgeRef.provider ?? "",
                rubric: rubric, structuredPrompt: structuredPrompt,
                prompt: prompt, responseA: a, responseB: b)
        }
        return try await ClaudePairedJudge.judge(
            model: judgeRef.model ?? "", rubric: rubric,
            structuredPrompt: structuredPrompt,
            prompt: prompt, responseA: a, responseB: b)
    }

    /// Judgments for every (packet × pinned judge) pair. A verdict whose
    /// winner is outside A/B/tie is RETRIED once (judges are LLMs — one
    /// malformed response is common) and, if still invalid, the whole phase
    /// REFUSES with the judge, packet, and both verdicts named
    /// (invalid-verdict closure 2026-07-20: recording an invented "tie"
    /// corrupted the preference mean — a tie is substantive data). The
    /// refusal throws before any completion verb runs, so the deferred
    /// judgment state on the server is untouched and the phase can simply
    /// be re-judged. Local judges never appear here — a deferred sweep is
    /// external-only by construction (split panels refused at sweep start).
    public static func judge(
        packets: [ClusterClient.SweepJudgingPacket],
        judges: [ClusterClient.AwaitingSweepJudgment.JudgeEntry],
        rubric: String,
        structuredPrompt: String? = nil,
        using judgeCall: JudgeCall = liveJudge,
        onProgress: (@Sendable (String) async -> Void)? = nil
    ) async throws -> [ClusterClient.SweepJudgmentEntry] {
        // Fail-closed preflight (engineer review 2026-07-18) — every
        // refusal here fires BEFORE the first paid judge call is spent.
        guard !judges.isEmpty else {
            throw PairedJudgeError(reason: "the awaiting run pins no judges")
        }
        var names = Set<String>()
        for judgeRef in judges {
            guard let name = judgeRef.name?.trimmingCharacters(
                in: .whitespacesAndNewlines), !name.isEmpty
            else {
                throw PairedJudgeError(
                    reason: "a pinned judge has no name — refusing to judge")
            }
            guard names.insert(name).inserted else {
                throw PairedJudgeError(
                    reason: "duplicate judge name '\(name)' in the pinned "
                        + "panel — refusing to judge")
            }
            // Strictly external — a missing or unknown kind is refused, not
            // shrugged past (fail closed; the server normalizes kinds at
            // emission, so absence means a malformed or ancient run).
            guard judgeRef.kind == "claude" || judgeRef.kind == "openrouter"
            else {
                throw PairedJudgeError(
                    reason: "judge '\(name)' has kind "
                        + "'\(judgeRef.kind ?? "missing")' — deferred "
                        + "panels are external-only (claude/openrouter) and "
                        + "must say so; re-run the sweep on a current server")
            }
            // The model is pinned at EMISSION — this Mac never resolves
            // against its own ambient default (engineer review 2026-07-18,
            // second pass).
            guard let model = judgeRef.model?.trimmingCharacters(
                in: .whitespacesAndNewlines), !model.isEmpty
            else {
                throw PairedJudgeError(
                    reason: "judge '\(name)' has no pinned model — the "
                        + "sweep pins the resolved judge model at "
                        + "emission; re-run the sweep on a current server")
            }
            // OpenRouter judges additionally pin their PROVIDER at emission
            // (the same slug can be served by different backends with
            // different outputs — an unpinned provider is not a pinned
            // judge).
            if judgeRef.kind == "openrouter" {
                guard let provider = judgeRef.provider?.trimmingCharacters(
                    in: .whitespacesAndNewlines), !provider.isEmpty
                else {
                    throw PairedJudgeError(
                        reason: "openrouter judge '\(name)' has no pinned "
                            + "provider — re-run the sweep on a current "
                            + "server")
                }
            }
        }
        var entries: [ClusterClient.SweepJudgmentEntry] = []
        let total = packets.count * judges.count
        for judgeRef in judges {
            let judgeName = judgeRef.name ?? "judge"
            // Preflight above guaranteed a pinned model (and, for
            // openrouter, a pinned provider); both are stamped into every
            // judgment and verified by the completion verb.
            let model = judgeRef.model ?? ""
            let provider =
                judgeRef.kind == "openrouter"
                ? judgeRef.provider.map(OpenRouterProviderIdentity.canonical)
                : nil
            for packet in packets {
                var verdict = try await judgeCall(
                    judgeRef, rubric, structuredPrompt, packet.prompt,
                    packet.responseA, packet.responseB)
                // Invalid-verdict closure (2026-07-20): a winner outside
                // A/B/tie used to be recorded as an invented winner-only
                // tie — but a tie is substantive data, and inventing one
                // corrupts the preference mean. Judges are LLMs, so one
                // malformed response is common: retry ONCE, then refuse
                // the whole phase. The throw happens before any completion
                // verb is called, so nothing is recorded server-side and
                // the phase can be re-judged cleanly.
                let validWinners = ["A", "B", "tie"]
                if !validWinners.contains(verdict.winner) {
                    let firstWinner = verdict.winner
                    verdict = try await judgeCall(
                        judgeRef, rubric, structuredPrompt, packet.prompt,
                        packet.responseA, packet.responseB)
                    guard validWinners.contains(verdict.winner) else {
                        throw PairedJudgeError(
                            reason: "judge '\(judgeName)' returned an "
                                + "invalid verdict twice for packet "
                                + "\(packet.packetID.prefix(12))… — winner "
                                + "'\(firstWinner)', then "
                                + "'\(verdict.winner)' (expected A, B, or "
                                + "tie). No judgments were uploaded and no "
                                + "completion ran — fix the judge or "
                                + "rubric, then re-run judging for this "
                                + "run")
                    }
                }
                // Winner-only closure (2026-07-20): the FULL verdict rides
                // with the judgment, so the completed run's artifacts carry
                // the same payload the inline path records.
                entries.append(.init(
                    packetID: packet.packetID, judge: judgeName,
                    winner: verdict.winner,
                    model: model, provider: provider,
                    confidence: verdict.confidence,
                    judgment: verdict))
                await onProgress?(
                    "judging sweep packets on this Mac — "
                        + "\(entries.count)/\(total)")
            }
        }
        return entries
    }

    /// The whole phase 2: fetch (hash-verified) → judge → complete, for
    /// BOTH awaiting kinds — the record's `kind` routes to the sweep or
    /// evaluate completion verb. Returns the judgment run directory.
    public static func judgeAndComplete(
        client: ClusterClient,
        experiment: String,
        awaiting: ClusterClient.AwaitingSweepJudgment,
        using judgeCall: JudgeCall = liveJudge,
        onProgress: (@Sendable (String) async -> Void)? = nil
    ) async throws -> String {
        guard let rubric = awaiting.rubric, !rubric.isEmpty else {
            throw PairedJudgeError(
                reason: "awaiting run '\(awaiting.run)' carries no rubric "
                    + "text — re-run the sweep on a current server")
        }
        // The rubric text must hash to its pin before a judge reads it —
        // the rubric IS the evaluation criterion (fail closed, 2026-07-18).
        guard let rubricPin = awaiting.rubricTextSha256 else {
            throw PairedJudgeError(
                reason: "awaiting run '\(awaiting.run)' carries no rubric "
                    + "text hash pin — re-run the sweep on a current server")
        }
        let rubricDigest = SHA256.hash(data: Data(rubric.utf8))
            .map { String(format: "%02x", $0) }.joined()
        guard rubricDigest == rubricPin else {
            throw PairedJudgeError(
                reason: "the rubric text does not hash to its pin "
                    + "(\(rubricPin.prefix(12))…) — refusing to judge with "
                    + "a drifted rubric")
        }
        // The structured prompt (evaluate runs) is part of the criterion —
        // same fail-closed pin discipline as the rubric.
        var structuredPrompt: String?
        if let structured = awaiting.structuredPrompt, !structured.isEmpty {
            guard let structuredPin = awaiting.structuredPromptSha256 else {
                throw PairedJudgeError(
                    reason: "awaiting run '\(awaiting.run)' carries a "
                        + "structured prompt with no hash pin — re-run "
                        + "evaluate on a current server")
            }
            let digest = SHA256.hash(data: Data(structured.utf8))
                .map { String(format: "%02x", $0) }.joined()
            guard digest == structuredPin else {
                throw PairedJudgeError(
                    reason: "the structured prompt does not hash to its pin "
                        + "(\(structuredPin.prefix(12))…) — refusing to "
                        + "judge with a drifted criterion")
            }
            structuredPrompt = structured
        }
        await onProgress?("fetching judging packets for \(awaiting.run)…")
        let packets = try await client.sweepJudgingPackets(awaiting: awaiting)
        let judgments = try await judge(
            packets: packets, judges: awaiting.judges ?? [], rubric: rubric,
            structuredPrompt: structuredPrompt,
            using: judgeCall, onProgress: onProgress)
        if awaiting.isEvaluate {
            await onProgress?("uploading \(judgments.count) judgments — the "
                + "server verifies pins and aggregates the judge report…")
            return try await client.completeEvaluateJudgment(
                experiment: experiment, evaluateRun: awaiting.run,
                judgments: judgments)
        }
        await onProgress?("uploading \(judgments.count) judgments — the "
            + "server verifies pins and computes the selection…")
        return try await client.completeSweepJudgment(
            experiment: experiment, sweepRun: awaiting.run,
            judgments: judgments)
    }
}
