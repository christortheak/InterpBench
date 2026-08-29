import CryptoKit
import Foundation
import SteeringKit

/// Seeded, stratified subsampling for the per-response coding evaluate —
/// the line-for-line twin of
/// `Server/steerlab_server/experiment/evaluate_subsample.py`.
///
/// **Why this exists (field discovery 2026-08-29).** A 7,200-record corpus
/// needed judged classification and the preregistered design was a stratified
/// 2,400-record subsample. `evaluate` codes a whole source run and nothing
/// else, so the operation had no honest spelling: the only routes available
/// were coding all 7,200 (not the declared design, and far more judge calls
/// than the power computation asked for) or hand-building a run directory
/// holding the chosen records — which is evidence-chain corruption under
/// immutable `runs/` and was correctly refused. The design was legitimate and
/// preregistered; the instrument simply could not say it.
///
/// **The contract.** `--sample-per-condition <n>` and `--sample-seed
/// <hex-or-int>` are BOTH given or NEITHER. A sample without a seed is a
/// subsample nobody can redraw; a seed without a sample size is a stamp on a
/// coding it did not shape. Either half alone is a malformed invocation (exit
/// 64) — no defaulted seed, no inferred size. There is deliberately no
/// fraction spelling: an absolute per-condition `n` is what a power
/// computation produces.
///
/// **Never clamp.** An `n` larger than a condition's codeable population
/// refuses. Clamping would code a smaller design than the preregistered one
/// while every stamp still said `samplePerCondition: n`.
///
/// **The draw is the house RNG.** Every choice is a partial Fisher–Yates over
/// `SplitMix64`, seeded through SHA-256 — the same primitive and the same
/// argument as `TokenBankDownsampler` (2026-08-28 audit, convention note 9):
/// a subsample whose membership can move under a runtime upgrade while its
/// seed stamp stays identical is not reproducible evidence.
/// `EvaluateSubsampleTests` pins the output with integer literals that appear
/// byte-identically in `Server/tests/test_evaluate_subsample.py`.
///
/// **Ordering is by UTF-8 bytes, on purpose.** Swift's `<` on `String` is
/// Unicode canonical-equivalence-aware and Python's `sorted` is code-point
/// order; the two agree on ASCII and are not guaranteed to agree beyond it.
/// The stratum order decides WHICH records are chosen, so both engines sort
/// promptIDs (and conditions) by their UTF-8 byte sequences.
///
/// **The design is DECLARED, not merely typed (review round 12, finding 4).**
/// The flags above record what HAPPENED; the word "preregistered" is a claim
/// about what was decided BEFORE anything ran, and a claim like that has to
/// live in the artifact chain or it is not evidence. So the sampling design is
/// a manifest declaration — `evaluationSampling` — written by `experiment
/// set-evaluation-sampling` on both authoring surfaces, and every run stamps
/// the manifest snapshot into its own `experiment.json`. That snapshot is the
/// provenance: a plan document is pre-registration, and the snapshot is what
/// proves the plan is the thing that ran.
///
/// When a study declares one, `evaluate` samples by it with no flags at all.
/// The flags may still be typed, and then they are a CROSS-CHECK: any
/// inequality with the declaration refuses at 64 naming both values
/// (`reconcile`). They are never an override — a flag that won would put the
/// coding and the snapshot in disagreement, which is precisely the loss the
/// declaration exists to prevent. A study that declares nothing keeps the
/// flags-only path exactly as it was.
///
/// **Declare-time vs run-time validation.** What can be known at the desk is
/// checked at the desk: a whole `samplePerCondition` of at least 1, a
/// `sampleSeed` that parses as a 64-bit unsigned number, and a `rule` derived
/// here rather than typed. What CANNOT be known at the desk is the
/// population: at declaration time there is no source run, and the same
/// design is legitimately declared before the run that will satisfy it
/// exists. So the over-ask refusal stays where the records are —
/// `selectedPositions`, at evaluate — and `declarationViolations` is the
/// verify-surface check that never invents an obligation a draft cannot meet.
public enum EvaluateSubsample {

    /// The derivation, stated once and stamped VERBATIM into every sampled
    /// coding report and run config. A reader who has the seed, the source
    /// run and this string can recompute the subsample's membership by hand;
    /// the version marker moves if the rule ever does. Twin literal:
    /// `evaluate_subsample.RULE`.
    public static let rule = """
        stratifiedByPromptID/v1 — within each condition, floor(n / P) records \
        per promptID over that condition's P promptIDs, the n mod P remainder \
        given one at a time to promptIDs in seeded order (a promptID already \
        at its codeable population is skipped and its quota passes on, so \
        exactly n records are always drawn); within each (condition, \
        promptID) cell the records are drawn over sampleIndex ascending. \
        Every draw is a partial Fisher-Yates over SplitMix64 seeded from the \
        first 8 bytes, big-endian, of SHA-256(seed as 8-byte big-endian || \
        each part length-prefixed as an 8-byte big-endian UTF-8 byte count \
        followed by those bytes): the parts are (condition,) for the promptID \
        order and (condition, promptID) for a cell draw. promptIDs are \
        ordered by UTF-8 bytes. Kept records stay in their source-run order.
        """

    /// The manifest key the sampling DESIGN is declared under (cross-engine
    /// contract key; server twin `evaluate_subsample.DECLARATION_KEY`).
    public static let declarationKey = "evaluationSampling"

    // MARK: - The request

    /// A validated `(n, seed)` ask. `seedText` is the canonical
    /// `0x`-prefixed 16-hex-digit spelling that every stamp carries — JSON
    /// has no unsigned 64-bit integer, and a decimal a reader's JSON parser
    /// rounds is a seed that no longer redraws its own subsample.
    public struct Request: Sendable, Equatable {
        public let samplePerCondition: Int
        public let seed: UInt64
        /// True when this draw came from the study's `evaluationSampling`
        /// declaration rather than from flags alone. It rides on the REQUEST
        /// so nothing downstream has to thread a second argument, and it
        /// reaches the stamp as the additive `declared: true` key — the one
        /// difference between a declared coding's stamp and an ad-hoc one's.
        /// Defaulted false so every existing construction is unchanged.
        public let declared: Bool

        public init(
            samplePerCondition: Int, seed: UInt64, declared: Bool = false
        ) {
            self.samplePerCondition = samplePerCondition
            self.seed = seed
            self.declared = declared
        }

        public var seedText: String { EvaluateSubsample.format(seed: seed) }
    }

    /// The study's DECLARED sampling design, as it sits in the manifest.
    ///
    /// `rule` is DERIVED at the write from `EvaluateSubsample.rule` and never
    /// accepted from a caller — the same guarantee as `parserRegistryHash`,
    /// for the same reason: a typed rule would let a study claim a derivation
    /// nothing performed. Decoding keeps whatever the file holds, so a
    /// hand-edited or older-version rule survives to be REPORTED by
    /// `declarationViolations` rather than silently corrected.
    public struct Declaration: Codable, Sendable, Equatable {
        public var rule: String
        public var samplePerCondition: Int
        public var sampleSeed: String

        public init(
            rule: String, samplePerCondition: Int, sampleSeed: String
        ) {
            self.rule = rule
            self.samplePerCondition = samplePerCondition
            self.sampleSeed = sampleSeed
        }
    }

    /// The `sampling` block. Additive by construction: its ABSENCE is what a
    /// full-corpus coding looks like, so every report written before this
    /// existed reads back byte-identically and no reader has to interpret a
    /// missing block as anything but "all of it".
    public struct Stamp: Codable, Sendable, Equatable {
        public let rule: String
        public let samplePerCondition: Int
        public let sampleSeed: String
        public let sampledRecords: Int
        public let sourceRecords: Int
        /// Present (and true) when the draw came from the study's
        /// `evaluationSampling` declaration; ABSENT when the flags alone
        /// asked for it. Additive for the same reason `sampling` itself is:
        /// a key that were always present would make `declared: false` read
        /// as a finding rather than as the older, still-honest spelling.
        public let declared: Bool?

        public init(
            rule: String, samplePerCondition: Int, sampleSeed: String,
            sampledRecords: Int, sourceRecords: Int, declared: Bool? = nil
        ) {
            self.rule = rule
            self.samplePerCondition = samplePerCondition
            self.sampleSeed = sampleSeed
            self.sampledRecords = sampledRecords
            self.sourceRecords = sourceRecords
            self.declared = declared
        }

        /// The same keys as a `[String: Any]`, for `RunMetadata`'s
        /// JSONSerialization payload — which takes a plain object rather
        /// than an `Encodable`. `declared` is omitted when nil, exactly as
        /// the encoder omits it.
        public var jsonObject: [String: Any] {
            var object: [String: Any] = [
                "rule": rule,
                "samplePerCondition": samplePerCondition,
                "sampleSeed": sampleSeed,
                "sampledRecords": sampledRecords,
                "sourceRecords": sourceRecords,
            ]
            if let declared { object["declared"] = declared }
            return object
        }
    }

    /// The canonical seed spelling: `0x` + 16 lowercase hex digits.
    public static func format(seed: UInt64) -> String {
        String(format: "0x%016lx", seed)
    }

    /// `--sample-seed` as a 64-bit unsigned integer.
    ///
    /// Accepts `0x`-prefixed hex, a bare decimal, or bare hex — an
    /// all-digits string is read as DECIMAL, because a seed a researcher
    /// typed as a number must mean the number they typed. Anything else, and
    /// anything that does not fit in 64 bits, refuses: a silently truncated
    /// seed stamps a value that does not redraw its own subsample.
    public static func parseSeed(_ text: String, program: String) throws -> UInt64 {
        let raw = text.trimmingCharacters(in: .whitespaces)
        guard !raw.isEmpty else {
            throw ExperimentError.malformed(
                "--sample-seed is empty — a subsample's seed is the only "
                    + "thing that lets anyone redraw it, so it cannot be blank",
                repair: seedRepair(program: program))
        }
        guard let value = seedValue(raw) else {
            // `UInt64(_:radix:)` fails identically for a non-number and for a
            // value too large to represent, so the message names both: the
            // repair is the same sentence either way.
            throw ExperimentError.malformed(
                "--sample-seed '\(raw)' is not a 64-bit unsigned number — a "
                    + "seed is a decimal integer, or hexadecimal with or "
                    + "without a '0x' prefix, of at most 16 hex digits (the "
                    + "leading 16 of a digest are a fine seed, written down "
                    + "as such)",
                repair: seedRepair(program: program))
        }
        return value
    }

    /// The seed GRAMMAR, without a sentence: the 64-bit unsigned value, or
    /// nil when the text does not name one.
    ///
    /// ONE grammar for every surface that reads a seed — the flag, the
    /// manifest declaration, and verify — so the three cannot drift into
    /// accepting different sets of strings while all three call the result
    /// "the seed". Server twin: `evaluate_subsample.seed_value`.
    public static func seedValue(_ text: String) -> UInt64? {
        var body = text.trimmingCharacters(in: .whitespaces)
        guard !body.isEmpty else { return nil }
        var radix = 10
        if body.count > 2, body.prefix(2).lowercased() == "0x" {
            body = String(body.dropFirst(2))
            radix = 16
        } else if !body.allSatisfy({ $0.isNumber }) {
            radix = 16
        }
        guard !body.isEmpty else { return nil }
        return UInt64(body, radix: radix)
    }

    static func seedRepair(program: String) -> String {
        "\(program) experiment evaluate <name> --sample-per-condition <n> "
            + "--sample-seed 0x5eed0a5e5eed0a5e  (any 64-bit value; record it "
            + "in the preregistration — the same seed always draws the same "
            + "records)"
    }

    /// Validate the flag PAIR before anything is read or written.
    ///
    /// `nil` for the full-corpus case (neither flag given), a validated
    /// request when both are, and a malformed-invocation refusal when exactly
    /// one is. Called at the CLI edge AND at the top of the task, so no
    /// caller can reach the draw with half a request.
    public static func resolveRequest(
        samplePerCondition: String?, sampleSeed: String?, program: String
    ) throws -> Request? {
        let sizeText = samplePerCondition?.trimmingCharacters(in: .whitespaces)
        let seedText = sampleSeed?.trimmingCharacters(in: .whitespaces)
        let hasSize = !(sizeText ?? "").isEmpty
        let hasSeed = !(seedText ?? "").isEmpty
        if !hasSize, !hasSeed { return nil }
        if hasSize, !hasSeed {
            throw ExperimentError.malformed(
                "--sample-per-condition \(sizeText ?? "") was given without "
                    + "--sample-seed: a subsample nobody can redraw is not "
                    + "evidence, so the draw refuses rather than choosing a "
                    + "seed for you",
                repair: seedRepair(program: program))
        }
        if hasSeed, !hasSize {
            throw ExperimentError.malformed(
                "--sample-seed \(seedText ?? "") was given without "
                    + "--sample-per-condition: with no sample size the full "
                    + "corpus is coded, and the seed would be stamped on a "
                    + "coding it did not shape",
                repair: "add --sample-per-condition <n> to draw a subsample, "
                    + "or drop --sample-seed and run \(program) experiment "
                    + "evaluate <name> to code the full corpus")
        }
        guard let count = Int(sizeText ?? ""), count >= 1 else {
            throw ExperimentError.malformed(
                "--sample-per-condition must be a whole number of records "
                    + "of at least 1, not '\(sizeText ?? "")' — a subsample "
                    + "of zero records is a design nobody can report",
                repair: "\(program) experiment evaluate <name> "
                    + "--sample-per-condition 2400 --sample-seed "
                    + "<seed>, or drop both flags to code the full corpus")
        }
        return Request(
            samplePerCondition: count,
            seed: try parseSeed(seedText ?? "", program: program))
    }

    // MARK: - The DECLARATION

    // `evaluationSampling`, and the flags' demotion to a cross-check. Server
    // twins: `evaluate_subsample.resolve_declaration`, `.declared_request`,
    // `.reconcile`, `.declaration_violations`.

    /// The retype for a refused sampling declaration. Names the verb, this
    /// study, and BOTH halves, because both-or-neither is the rule the
    /// refusals below are almost always enforcing.
    public static func declarationRepair(
        experiment: String, program: String
    ) -> String {
        "\(program) experiment set-evaluation-sampling \(experiment) 2400 "
            + "0x5eed0a5e5eed0a5e  (a per-condition size and the seed that "
            + "draws it — both, always; \"\" clears the declaration)"
    }

    /// `(n, seed)` for a well-formed declaration, nil for the CLEAR.
    ///
    /// The single grammar behind both the writer (`resolveDeclaration`) and
    /// the reader (`declaredRequest`), so a block this engine wrote can never
    /// fail to read back, and a hand-edited one refuses with the sentence its
    /// author would have got at the verb.
    static func declarationParts(
        samplePerCondition: String?, sampleSeed: String?,
        experiment: String, program: String
    ) throws -> (count: Int, seed: UInt64)? {
        let sizes = (samplePerCondition ?? "")
            .trimmingCharacters(in: .whitespaces)
        let seeds = (sampleSeed ?? "").trimmingCharacters(in: .whitespaces)
        if sizes.isEmpty, seeds.isEmpty { return nil }
        if !sizes.isEmpty, seeds.isEmpty {
            throw ExperimentError.malformed(
                "the sampling design named \(sizes) record(s) per condition "
                    + "with no seed — a subsample nobody can redraw is not a "
                    + "preregistration, so the declaration refuses rather "
                    + "than choosing a seed for you",
                repair: declarationRepair(
                    experiment: experiment, program: program))
        }
        if !seeds.isEmpty, sizes.isEmpty {
            throw ExperimentError.malformed(
                "the sampling design named seed \(seeds) with no "
                    + "per-condition size — with no size the full corpus is "
                    + "coded, and the seed would be stamped on a design it "
                    + "did not shape",
                repair: declarationRepair(
                    experiment: experiment, program: program))
        }
        guard let count = Int(sizes), count >= 1 else {
            throw ExperimentError.malformed(
                "the sampling design's samplePerCondition must be a whole "
                    + "number of records of at least 1, not '\(sizes)' — a "
                    + "subsample of zero records is a design nobody can "
                    + "report",
                repair: declarationRepair(
                    experiment: experiment, program: program))
        }
        guard let seed = seedValue(seeds) else {
            throw ExperimentError.malformed(
                "the sampling design's sampleSeed '\(seeds)' is not a 64-bit "
                    + "unsigned number — a seed is a decimal integer, or "
                    + "hexadecimal with or without a '0x' prefix, of at most "
                    + "16 hex digits (the leading 16 of a digest are a fine "
                    + "seed, written down as such)",
                repair: declarationRepair(
                    experiment: experiment, program: program))
        }
        return (count, seed)
    }

    /// The `evaluationSampling` block for a declaration, or nil when both
    /// halves are empty — which is the CLEAR, the affordance every other
    /// declaration verb carries.
    ///
    /// Only the DESK-KNOWABLE rules run here: a whole `n` of at least 1 and a
    /// seed that parses. The population check cannot run — the source run
    /// this design will be drawn from need not exist yet, and usually does
    /// not, since declaring the design before running is the entire point —
    /// so it stays in `selectedPositions`.
    public static func resolveDeclaration(
        samplePerCondition: String?, sampleSeed: String?,
        experiment: String, program: String
    ) throws -> Declaration? {
        guard
            let parts = try declarationParts(
                samplePerCondition: samplePerCondition, sampleSeed: sampleSeed,
                experiment: experiment, program: program)
        else { return nil }
        // `rule` is DERIVED, never typed: the same argument as the parser
        // registry's hash. Stamped verbatim so a reader of the run's manifest
        // snapshot can recompute the membership without this build.
        return Declaration(
            rule: rule, samplePerCondition: parts.count,
            sampleSeed: format(seed: parts.seed))
    }

    /// A stored declaration read back as a request, or nil when the study
    /// declares nothing. `declared` is true on whatever comes back, which is
    /// what puts `declared: true` in the coding stamp.
    public static func declaredRequest(
        _ declaration: Declaration?, experiment: String, program: String
    ) throws -> Request? {
        guard let declaration else { return nil }
        guard
            let parts = try declarationParts(
                samplePerCondition: String(declaration.samplePerCondition),
                sampleSeed: declaration.sampleSeed,
                experiment: experiment, program: program)
        else { return nil }
        return Request(
            samplePerCondition: parts.count, seed: parts.seed, declared: true)
    }

    /// The effective draw, given what the flags asked for and what the study
    /// declared. `program` names the AUTHORING binary the repair points at.
    ///
    /// * No declaration → the flags, unchanged. The ad-hoc path is untouched
    ///   and stays loud: its stamps still say SUBSAMPLE on every line.
    /// * Declaration, no flags → the declaration. This is the point of the
    ///   feature: a declared study needs no flags at all.
    /// * Both, and equal → the declaration (so `declared: true` is stamped).
    /// * Both, and unequal → REFUSED at 64, naming both values.
    ///
    /// The flags are a cross-check, never an override. A flag that won would
    /// code one design while the run's `experiment.json` snapshot — the
    /// artifact a reader trusts — recorded another, which is exactly the
    /// silent substitution the whole refusal vocabulary exists to prevent.
    /// The repair is therefore never "pass --force": it is to drop the flag,
    /// or to declare the design you actually want on a draft.
    public static func reconcile(
        flags: Request?, declaration: Request?, program: String
    ) throws -> Request? {
        guard let declaration else { return flags }
        guard let flags else { return declaration }
        if flags.samplePerCondition != declaration.samplePerCondition {
            throw ExperimentError.malformed(
                "--sample-per-condition \(flags.samplePerCondition) "
                    + "contradicts this study's declared sampling design, "
                    + "which preregistered \(declaration.samplePerCondition) "
                    + "record(s) per condition. On a study that declares its "
                    + "design the flag is a CROSS-CHECK, never an override: "
                    + "the declaration is what the run's experiment.json "
                    + "snapshot carries, so a flag that won would code one "
                    + "design and record another",
                repair: "drop --sample-per-condition (the declaration "
                    + "already supplies \(declaration.samplePerCondition)), "
                    + "or declare the design you actually want on a draft: "
                    + "\(program) experiment set-evaluation-sampling <name> "
                    + "\(flags.samplePerCondition) <seed>")
        }
        if flags.seed != declaration.seed {
            throw ExperimentError.malformed(
                "--sample-seed \(flags.seedText) contradicts this study's "
                    + "declared sampling design, which preregistered seed "
                    + "\(declaration.seedText). On a study that declares its "
                    + "design the flag is a CROSS-CHECK, never an override: "
                    + "the declaration is what the run's experiment.json "
                    + "snapshot carries, so a flag that won would draw one "
                    + "subsample and record another",
                repair: "drop --sample-seed (the declaration already "
                    + "supplies \(declaration.seedText)), or declare the "
                    + "design you actually want on a draft: \(program) "
                    + "experiment set-evaluation-sampling <name> "
                    + "\(declaration.samplePerCondition) \(flags.seedText)")
        }
        return declaration
    }

    /// The verify() surface for a stored `evaluationSampling` block.
    ///
    /// ABSENT = no declaration = no violations, so every manifest written
    /// before this existed verifies exactly as it did. What is checked is
    /// what a desk can check: a whole positive `n`, a parseable seed, and
    /// that the `rule` is the one THIS build derives — a declaration carrying
    /// an older `stratifiedByPromptID` version would not redraw the same
    /// records, and the version marker exists so that is visible rather than
    /// silent.
    ///
    /// What is NOT checked here is the population: no run exists yet at
    /// verify time, and inventing an obligation a draft cannot meet would
    /// make the declaration unusable in the order a study is actually
    /// authored. That check lives in `selectedPositions`, where the records
    /// are. Server twin: `evaluate_subsample.declaration_violations`.
    public static func declarationViolations(
        _ declaration: Declaration?
    ) -> [String] {
        guard let declaration else { return [] }
        var problems: [String] = []
        if declaration.samplePerCondition < 1 {
            problems.append(
                "\(declarationKey).samplePerCondition must be a whole number "
                    + "of records of at least 1 (declared: "
                    + "\(declaration.samplePerCondition))")
        }
        if seedValue(declaration.sampleSeed) == nil {
            problems.append(
                "\(declarationKey).sampleSeed '\(declaration.sampleSeed)' is "
                    + "not a 64-bit unsigned number")
        }
        if declaration.rule != rule {
            problems.append(
                "\(declarationKey).rule is not the draw rule this build "
                    + "derives — the declaration was written under a "
                    + "different version of stratifiedByPromptID and would "
                    + "not redraw the same records; re-declare the design to "
                    + "derive the current rule")
        }
        return problems
    }

    // MARK: - The draw

    /// The per-stratum stream seed: the first 8 bytes, big-endian, of
    /// SHA-256(seed as 8-byte big-endian ‖ each length-prefixed UTF-8 part).
    ///
    /// Length-prefixing is not decoration: without it `("ab", "c")` and
    /// `("a", "bc")` would hash identically, so two different cells of one
    /// condition could share a draw — the same reason
    /// `TokenBankDownsampler.corpusHash(texts:)` length-prefixes.
    public static func streamSeed(_ seed: UInt64, _ parts: String...) -> UInt64 {
        streamSeed(seed, parts: parts)
    }

    public static func streamSeed(_ seed: UInt64, parts: [String]) -> UInt64 {
        var hasher = SHA256()
        hasher.update(data: bigEndianBytes(seed))
        for part in parts {
            let encoded = Data(part.utf8)
            hasher.update(data: bigEndianBytes(UInt64(encoded.count)))
            hasher.update(data: encoded)
        }
        return hasher.finalize().prefix(8).reduce(UInt64(0)) {
            ($0 << 8) | UInt64($1)
        }
    }

    private static func bigEndianBytes(_ value: UInt64) -> Data {
        var big = value.bigEndian
        return withUnsafeBytes(of: &big) { Data($0) }
    }

    /// A full Fisher–Yates permutation of `0..<count` over SplitMix64 — the
    /// same loop as `TokenBankDownsampler.selectedIndices`, without the final
    /// sort, because here the ORDER is the answer (which promptID receives
    /// the next remainder record).
    public static func seededOrder(count: Int, seed: UInt64) -> [Int] {
        guard count > 0 else { return [] }
        var rng = SplitMix64(seed: seed)
        var pool = Array(0..<count)
        for position in 0..<count {
            let remaining = count - position
            let offset = Int(rng.next() % UInt64(remaining))
            pool.swapAt(position, position + offset)
        }
        return pool
    }

    /// How many records each promptID contributes: `floor(total / P)` each,
    /// then the remainder handed out one at a time in seeded promptID order,
    /// skipping any promptID already at its population and passing its quota
    /// on.
    ///
    /// The skip is what makes "exactly `total` records, always" an invariant
    /// on a ragged run (a partial or resumed source run need not be
    /// rectangular). It terminates because `select` has already refused when
    /// the condition's total population is below `total`, so headroom always
    /// exists until the last record is placed.
    static func allotments(
        populations: [Int], total: Int, seed: UInt64
    ) -> [Int] {
        let count = populations.count
        guard count > 0, total > 0 else { return Array(repeating: 0, count: count) }
        var allot = populations.map { Swift.min(total / count, $0) }
        let order = seededOrder(count: count, seed: seed)
        var remaining = total - allot.reduce(0, +)
        while remaining > 0 {
            var placed = 0
            for index in order where remaining > 0 {
                if allot[index] < populations[index] {
                    allot[index] += 1
                    remaining -= 1
                    placed += 1
                }
            }
            if placed == 0 { break }  // guarded by the population refusal
        }
        return allot
    }

    /// One record's stratum coordinates, so the draw does not depend on the
    /// concrete record type. `ExperimentTasks.EvaluationGeneration` maps onto
    /// it directly.
    public struct Coordinate: Sendable, Equatable {
        public let condition: String
        public let promptID: String
        public let sampleIndex: UInt64

        public init(condition: String, promptID: String, sampleIndex: UInt64) {
            self.condition = condition
            self.promptID = promptID
            self.sampleIndex = sampleIndex
        }
    }

    /// Codeable records per condition — what an over-ask is measured against.
    public static func population(
        of coordinates: [Coordinate]
    ) -> [String: Int] {
        var counts: [String: Int] = [:]
        for coordinate in coordinates {
            counts[coordinate.condition, default: 0] += 1
        }
        return counts
    }

    /// The source-run positions to KEEP, sorted ascending.
    ///
    /// Refuses — writing nothing, because this runs before the evaluate run
    /// directory is minted — when any condition holds fewer than
    /// `request.samplePerCondition` codeable records.
    public static func selectedPositions(
        _ coordinates: [Coordinate], request: Request, program: String
    ) throws -> [Int] {
        let populations = population(of: coordinates)
        let short = populations
            .filter { $0.value < request.samplePerCondition }
            .sorted {
                $0.value == $1.value
                    ? utf8Precedes($0.key, $1.key) : $0.value < $1.value
            }
        if let (tightest, available) = short.first {
            let named = short.map { "'\($0.key)' has \($0.value)" }
                .joined(separator: ", ")
            throw ExperimentError.malformed(
                "--sample-per-condition \(request.samplePerCondition) exceeds "
                    + "what the source run holds: \(named) codeable "
                    + "record(s). A subsample cannot be larger than the "
                    + "stratum it is drawn from, and clamping it would code a "
                    + "smaller design than the one that was preregistered "
                    + "while every stamp still said "
                    + "\(request.samplePerCondition)",
                repair: "re-run with --sample-per-condition \(available) or "
                    + "less (condition '\(tightest)' is the binding stratum), "
                    + "or drop both sample flags and run \(program) "
                    + "experiment evaluate <name> to code all "
                    + "\(coordinates.count) record(s)")
        }

        var keep: Set<Int> = []
        for condition in populations.keys.sorted(by: utf8Precedes) {
            var cells: [String: [Int]] = [:]
            for (position, coordinate) in coordinates.enumerated()
            where coordinate.condition == condition {
                cells[coordinate.promptID, default: []].append(position)
            }
            let prompts = cells.keys.sorted(by: utf8Precedes)
            for prompt in prompts {
                cells[prompt]?.sort {
                    coordinates[$0].sampleIndex == coordinates[$1].sampleIndex
                        ? $0 < $1
                        : coordinates[$0].sampleIndex
                            < coordinates[$1].sampleIndex
                }
            }
            let allot = allotments(
                populations: prompts.map { cells[$0]?.count ?? 0 },
                total: request.samplePerCondition,
                seed: streamSeed(request.seed, condition))
            for (index, prompt) in prompts.enumerated() {
                let positions = cells[prompt] ?? []
                let chosen = TokenBankDownsampler.selectedIndices(
                    count: positions.count, cap: allot[index],
                    seed: streamSeed(request.seed, condition, prompt))
                for offset in chosen { keep.insert(positions[offset]) }
            }
        }
        return keep.sorted()
    }

    /// The stamp for a completed draw.
    ///
    /// `declared: true` is additive inside the block, for the same reason the
    /// block itself is and one level down: present when the draw came from
    /// the study's `evaluationSampling` declaration, ABSENT when the flags
    /// alone asked for it. Both stamps say SUBSAMPLE on every line — the
    /// ad-hoc path is not quieter, it simply cannot claim the provenance the
    /// declared one has.
    public static func stamp(
        _ request: Request, sampled: Int, source: Int
    ) -> Stamp {
        Stamp(
            rule: rule, samplePerCondition: request.samplePerCondition,
            sampleSeed: request.seedText, sampledRecords: sampled,
            sourceRecords: source,
            declared: request.declared ? true : nil)
    }

    /// The human count every line says: `"7200 record(s)"` for a full corpus,
    /// `"2400 of 7200 record(s) (seeded subsample)"` for a sampled one.
    ///
    /// One function so no line can drift into implying a full-corpus coding —
    /// the whole point of the loud stamping is that a reader CANNOT mistake
    /// the two, and a report is read line by line, not block by block.
    public static func codedPhrase(_ stamp: Stamp?, total: Int) -> String {
        guard let stamp else { return "\(total) record(s)" }
        return "\(stamp.sampledRecords) of \(stamp.sourceRecords) record(s) "
            + "(seeded subsample)"
    }

    /// The sample flags on a pairedJudge evaluate.
    ///
    /// Scoped deliberately (2026-08-29): the coding instrument's unit of
    /// analysis is a RECORD, which is what a per-condition stratified draw is
    /// defined over. The paired judge's unit is a PAIR — a variant record
    /// joined to the baseline record of its (promptID, sampleIndex) cell —
    /// where "baseline" is not a sampled condition at all but the other half
    /// of every comparison, so "n records per condition" does not name a set
    /// of pairs. Silently ignoring the flags would be the trap the `--shard`
    /// refusal exists to prevent: a correct-looking command line the CLI
    /// half-executes.
    public static func pairedRefusal(program: String) -> ExperimentError {
        ExperimentError.malformed(
            "--sample-per-condition/--sample-seed apply to the per-response "
                + "coding instrument only: this study's pinned rubric is a "
                + "paired comparison, whose unit is a (baseline, variant) "
                + "PAIR rather than a record, so a per-condition record count "
                + "does not name a set of pairs to judge",
            repair: "drop both sample flags and run \(program) experiment "
                + "evaluate <name> to judge every pair, or pin a "
                + "perResponseCoding rubric if per-record coding is the design")
    }

    /// Byte-order comparison over UTF-8, the cross-engine sort. Swift's `<`
    /// on `String` normalizes; Python's `sorted` does not, and the stratum
    /// order decides which records the draw picks.
    static func utf8Precedes(_ lhs: String, _ rhs: String) -> Bool {
        Array(lhs.utf8).lexicographicallyPrecedes(Array(rhs.utf8))
    }
}
