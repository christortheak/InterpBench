import CryptoKit
import Foundation
import SteeringKit

/// The sweep's selection decision as a pure, unit-testable rule (cross-engine
/// contract with the server's `sweep_selection.py`).
///
/// The manifest's `sweep.selection` block declares HOW the recommended
/// layer×alpha cell is chosen — objective metric, capability/coherence
/// constraints, optional matched-norm random control margin — so the choice
/// is preregistered data, not engine code. An absent block resolves to the
/// historical hardcoded behavior, keeping old manifests' meaning (and content
/// hashes) stable.
public enum SweepSelectionRule {
    /// The shared metric enum — all three are implemented on both engines
    /// (2026-07-08); unknown strings still FAIL FAST rather than silently
    /// fall back. The known/implemented split survives so an engine that
    /// temporarily lags the other refuses loudly instead of mis-selecting.
    public static let knownMetrics = ["markerDensity", "judgeScore", "logprobShift"]
    public static let implementedMetrics = ["markerDensity", "judgeScore", "logprobShift"]

    public static let defaultCapabilityTolerance = 0.15
    /// The LEGACY absolute distinct-2 floor. Still the resolved value for
    /// every criterion declared before the baseline-relative form existed, and
    /// still what a criterion declaring `coherenceFloor` alone means.
    public static let defaultCoherenceFloor = 0.45

    // MARK: - The baseline-relative coherence floor
    //
    // An ABSOLUTE distinct-2 floor gates against a fixed number, and a fixed
    // number cannot know what the model's own prose looks like. A sweep
    // admitted a cell at distinct-2 0.535 against a baseline of 0.989 —
    // barely half the coherence the unsteered model produced, and 65% longer
    // output — and its logprobShift was REPETITION rather than steering, which
    // is precisely the failure the floor exists to catch. 0.535 clears 0.45,
    // so the gate said yes.
    //
    // The floor a sweep declares from now on is therefore relative to the α=0
    // baseline cell, with an absolute backstop underneath it: a cell passes
    // when its distinct-2 is at least `ratio ×` the baseline's AND at least
    // `backstop`. The backstop is what keeps a degenerate BASELINE from
    // licensing a degenerate winner.
    //
    // Existing pinned criteria are untouched, forever: a constraints block
    // with neither new field means the ABSOLUTE rule at its declared (or
    // default) `coherenceFloor`, which is exactly what those studies ran.

    /// Default `distinct2 ≥ ratio × baseline.distinct2`.
    public static let defaultCoherenceRatio = 0.85
    /// Default absolute backstop under the relative floor.
    public static let defaultCoherenceBackstop = 0.60
    /// A cell's mean output length above this multiple of the baseline's is
    /// FLAGGED in the sweep report. A flag, never a gate: length inflation is
    /// evidence a reader needs when interpreting a metric, not a rule about
    /// which cells may win.
    public static let lengthInflationFactor = 1.5

    /// What a capability tolerance can actually gate on, given how many
    /// items the battery holds (C4).
    ///
    /// Battery accuracy moves in steps of 1/N, so a declared tolerance that
    /// falls between steps is not the tolerance that operates. With N = 10
    /// and tolerance 0.15, a one-item drop (0.1) passes and a two-item drop
    /// (0.2) fails — the gate is effectively 0.2, not the 0.15 the manifest
    /// declares and the sweep reports. Stating the operative number is the
    /// point; nothing here changes the rule.
    public struct BatteryResolution: Sendable, Equatable {
        public var itemCount: Int
        public var declaredTolerance: Double
        /// The smallest drop the battery can express that actually fails the
        /// constraint.
        public var effectiveTolerance: Double
        /// The declared value is materially finer than the battery can see.
        public var isCoarse: Bool

        public var summary: String {
            let declared = declaredTolerance.formatted(
                .number.precision(.fractionLength(0 ... 3)))
            let effective = effectiveTolerance.formatted(
                .number.precision(.fractionLength(0 ... 3)))
            let step = (1.0 / Double(itemCount)).formatted(
                .number.precision(.fractionLength(0 ... 3)))
            return "capability battery has \(itemCount) item"
                + "\(itemCount == 1 ? "" : "s"), so accuracy moves in steps of "
                + "\(step); the declared tolerance \(declared) therefore gates "
                + "at the first larger step, \(effective)"
        }
    }

    /// Resolve what the tolerance really gates at. Nil when the battery is
    /// empty (nothing to say).
    public static func batteryResolution(
        itemCount: Int, capabilityTolerance: Double
    ) -> BatteryResolution? {
        guard itemCount > 0, capabilityTolerance.isFinite,
            capabilityTolerance >= 0
        else { return nil }
        let n = Double(itemCount)
        // The constraint is `cell >= baseline - tolerance`, so a drop EQUAL
        // to the tolerance passes. The first failing drop is the smallest
        // k/N strictly greater than the tolerance.
        let k = floor(capabilityTolerance * n + 1e-9) + 1
        let effective = k / n
        return BatteryResolution(
            itemCount: itemCount,
            declaredTolerance: capabilityTolerance,
            effectiveTolerance: effective,
            // "Materially" = the operative gate is at least 1.5× what was
            // declared. An exactly-on-a-step tolerance lands at 2× and is
            // the common case worth flagging.
            isCoarse: effective >= capabilityTolerance * 1.5)
    }

    /// The RESOLVED criterion a sweep actually applies (defaults filled in).
    public struct Resolved: Sendable, Equatable {
        public var metric: String
        public var capabilityTolerance: Double
        /// The ABSOLUTE distinct-2 floor. Under the legacy rule this IS the
        /// gate; under the baseline-relative rule it carries the backstop, so
        /// every surface that reads one absolute number keeps reading a true
        /// one (the number below which no cell passes either way).
        public var coherenceFloor: Double
        /// Non-nil = the BASELINE-RELATIVE rule: a cell passes coherence only
        /// when its distinct-2 is at least this multiple of the α=0
        /// baseline's AND at least `coherenceFloor` (the backstop). Nil = the
        /// legacy absolute rule, which is what every criterion declared
        /// before this form means and will mean forever.
        public var coherenceRatioToBaseline: Double?
        public var matchedNormRandomMargin: Double?
        /// "winner" (historical) or "topK" — see the Controls doc.
        public var controlApplyTo: String = "winner"
        public var controlTopK: Int?

        /// Whether this criterion gates coherence against the baseline.
        public var isBaselineRelativeCoherence: Bool {
            coherenceRatioToBaseline != nil
        }

        /// The coherence rule in one clause, in the words both engines print.
        /// Server twin: `SelectionCriterion.coherence_summary`.
        /// The numbers are rendered the way Python's `:g` renders them
        /// (`SweepSelectionRule.g`), so a ratio of 1 reads "1×" on both
        /// engines rather than "1×" on one and "1.0×" on the other.
        public var coherenceSummary: String {
            guard let ratio = coherenceRatioToBaseline else {
                return "coherence floor \(SweepSelectionRule.g(coherenceFloor)) "
                    + "(absolute distinct-2)"
            }
            return "coherence floor \(SweepSelectionRule.g(ratio))× the α=0 "
                + "baseline's distinct-2, backstop "
                + "\(SweepSelectionRule.g(coherenceFloor))"
        }

        /// The resolved criterion in the manifest's own JSON shape — embedded
        /// verbatim in selection provenance and promotion birth certificates.
        public var asCriterion: ExperimentManifest.SweepSelection {
            asCriterion(objective: nil)
        }

        /// The resolved criterion with the objective's per-metric pins
        /// (choice file + hash, judge rubric hash + judges) embedded in the
        /// `objective` block, so provenance pins the instrument's data.
        /// Under the per-concept declaration, `concept` selects WHICH
        /// instrument this provenance block pins — the one the concept's
        /// cells were actually scored on.
        public func asCriterion(
            objective resolved: ResolvedObjective?,
            concept: String? = nil
        ) -> ExperimentManifest.SweepSelection {
            var objective = ExperimentManifest.SweepSelection.Objective(metric: metric)
            if let resolved {
                if let sets = resolved.choiceSets, let concept,
                    let chosen = sets[concept]
                {
                    objective.choicePromptsFile = chosen.file
                    objective.choicePromptsHash = chosen.hash
                } else {
                    objective.choicePromptsFile = resolved.choicePromptsFile
                    objective.choicePromptsHash = resolved.choicePromptsHash
                }
                objective.judgeRubricHash = resolved.judgeRubricHash
                objective.judges =
                    resolved.judges.isEmpty ? nil : resolved.judges
            }
            return ExperimentManifest.SweepSelection(
                objective: objective,
                constraints: .init(
                    capabilityTolerance: capabilityTolerance,
                    coherenceFloor: coherenceFloor,
                    // Emitted only under the relative rule, so a legacy
                    // criterion round-trips to byte-identical JSON and keeps
                    // its content hash — and so "no new fields" keeps meaning
                    // "the absolute rule", forever.
                    coherenceRatioToBaseline: coherenceRatioToBaseline,
                    coherenceAbsoluteBackstop: coherenceRatioToBaseline == nil
                        ? nil : coherenceFloor),
                controls: matchedNormRandomMargin.map {
                    .init(
                        matchedNormRandomMargin: $0,
                        applyTo: controlApplyTo == "winner" ? nil : controlApplyTo,
                        topK: controlApplyTo == "winner" ? nil : controlTopK)
                })
        }
    }

    /// The advisory a sweep owes a caller that declared no selection rule
    /// while measuring a CHOICE task (WP0 step 7, punch list #1 P3).
    ///
    /// An absent `sweep.selection` resolves to `markerDensity` — surface-prose
    /// expression — which `docs/CLAUDE.md` and the methods note both forbid as
    /// a promotion objective for decision studies ("marker density is a
    /// diagnostic/manipulation check — never the promotion objective when
    /// the claim is about a substantive outcome"). Dry run #1 watched that
    /// default fire silently on
    /// a study whose every pinned item carried `options` + `target`, i.e. on
    /// exactly the study the rule exists for.
    ///
    /// It stays an ADVISORY, not a refusal: a legal manifest may sweep on
    /// marker density deliberately, and a sweep that refuses to run because a
    /// field is absent would break every existing screening study. What
    /// changes is that the choice is now visible at the moment it is made.
    ///
    /// nil when a selection WAS declared, or when the task set is not
    /// choice-shaped, or when the item file could not be read (silence beats a
    /// guess).
    public static func defaultedSelectionAdvisory(
        spec: ExperimentManifest.SweepSelection?,
        choiceItemCount: Int,
        totalItemCount: Int
    ) -> String? {
        guard spec?.objective?.metric == nil, choiceItemCount > 0 else { return nil }
        return "no sweep.selection is declared, so the winning cell will be "
            + "chosen by markerDensity — a SURFACE-PROSE diagnostic — while "
            + "\(choiceItemCount) of \(totalItemCount) pinned item(s) carry "
            + "options/target and could be scored deterministically. Declare "
            + "the criterion: steerlab-cli experiment set-sweep-selection "
            + "<name> --objective logprobShift --choice-prompts <file>  (or "
            + "--objective judgeScore with a pinned rubric). Marker density is "
            + "a manipulation check, not a decision objective."
    }

    /// Resolve a manifest `sweep.selection` block (or nil) to the criterion
    /// the sweep applies. Throws at sweep START for declared-but-unimplemented
    /// metrics and for unknown metric strings — never mid-run.
    // MARK: - Coherence refusals (cross-engine literals; server twin:
    // `sweep_selection.resolve_selection`)

    public static func coherenceRatioRangeRefusal(_ value: Double) -> String {
        "sweep.selection coherenceRatioToBaseline must be a finite number in "
            + "(0, 1] — got \(value). It is a FRACTION of the α=0 baseline's "
            + "distinct-2, so 1 means 'as coherent as the unsteered model' and "
            + "anything above 1 asks a steered cell to beat it"
    }

    public static func coherenceBackstopRangeRefusal(_ value: Double) -> String {
        "sweep.selection coherenceAbsoluteBackstop must be a finite number in "
            + "[0, 1) — got \(value). It is the absolute distinct-2 no cell may "
            + "fall below however incoherent the baseline was; 1 would admit "
            + "nothing"
    }

    public static func coherenceOrderRefusal(
        ratio: Double, backstop: Double
    ) -> String {
        "sweep.selection declares a baseline-relative coherence floor of "
            + "\(ratio)× with an absolute backstop of \(backstop), but the "
            + "backstop must sit UNDER the relative bar (backstop < ratio). A "
            + "baseline's distinct-2 is at most 1, so a bar of \(ratio)× can "
            + "never demand more than \(ratio) — a backstop of \(backstop) "
            + "would gate every cell absolutely while the criterion reads as "
            + "relative"
    }

    /// Does this cell clear the coherence gate? THE one place the rule lives,
    /// so selection, ranking, the no-selection reason and the grid's cell
    /// marking cannot drift from each other — the drift that let a degenerate
    /// cell through in the first place. Server twin: `coherence_passes`.
    public static func coherencePasses(
        distinct2: Double, baselineDistinct2: Double, criterion: Resolved
    ) -> Bool {
        guard let ratio = criterion.coherenceRatioToBaseline else {
            return distinct2 >= criterion.coherenceFloor
        }
        return distinct2 >= ratio * baselineDistinct2
            && distinct2 >= criterion.coherenceFloor
    }

    /// A cell's distinct-2 as a fraction of the baseline's — the number the
    /// relative floor gates on, reported for EVERY cell whichever rule is in
    /// force. Nil when the baseline's own distinct-2 is 0 (the ratio is
    /// undefined, and reporting 0 or ∞ would be an invented fact).
    public static func distinct2Ratio(
        distinct2: Double, baselineDistinct2: Double
    ) -> Double? {
        guard baselineDistinct2 > 0 else { return nil }
        return distinct2 / baselineDistinct2
    }

    /// Whether this cell's mean output length exceeds
    /// `lengthInflationFactor ×` the baseline's — a REPORTED column, never a
    /// gate. The degenerate cell that motivated the relative floor ran 65%
    /// long, and a reader looking at a logprobShift owes themselves that fact.
    public static func lengthInflated(
        meanWords: Double, baselineMeanWords: Double
    ) -> Bool {
        baselineMeanWords > 0
            && meanWords > lengthInflationFactor * baselineMeanWords
    }

    public static func resolve(
        _ spec: ExperimentManifest.SweepSelection?
    ) throws -> Resolved {
        let metric = spec?.objective?.metric ?? "markerDensity"
        guard knownMetrics.contains(metric) else {
            throw ExperimentError(
                reason: "unknown selection metric '\(metric)' — known metrics: "
                    + knownMetrics.joined(separator: ", "))
        }
        guard implementedMetrics.contains(metric) else {
            throw ExperimentError(
                reason: "selection metric '\(metric)' is not implemented on this "
                    + "engine yet — it is declared for forward compatibility; "
                    + "use markerDensity")
        }
        // Range validation (cross-engine contract with the server's
        // `resolve_selection`): declared numbers outside their meaningful
        // ranges fail at sweep START, never mid-run.
        let tolerance = spec?.constraints?.capabilityTolerance
            ?? defaultCapabilityTolerance
        guard tolerance.isFinite, tolerance >= 0, tolerance <= 1 else {
            throw ExperimentError(
                reason: "sweep.selection capabilityTolerance must be a finite "
                    + "number in [0, 1] — got \(tolerance)")
        }
        // WHICH coherence rule this criterion declares is decided by the
        // PRESENCE of either relative field — never by their values — so a
        // constraints block written before the relative form existed keeps
        // its absolute semantics permanently, and a stamped criterion decodes
        // to the rule that actually ran.
        let declaredRatio = spec?.constraints?.coherenceRatioToBaseline
        let declaredBackstop = spec?.constraints?.coherenceAbsoluteBackstop
        let relative = declaredRatio != nil || declaredBackstop != nil
        let declaredFloor = spec?.constraints?.coherenceFloor
        let floor: Double
        if relative {
            let ratio = declaredRatio ?? defaultCoherenceRatio
            let backstop = declaredBackstop ?? defaultCoherenceBackstop
            guard ratio.isFinite, ratio > 0, ratio <= 1 else {
                throw ExperimentError(
                    reason: coherenceRatioRangeRefusal(ratio))
            }
            guard backstop.isFinite, backstop >= 0, backstop < 1 else {
                throw ExperimentError(
                    reason: coherenceBackstopRangeRefusal(backstop))
            }
            // Ascending sanity: the backstop sits UNDER the relative bar. A
            // backstop at or above the ratio can never be the looser of the
            // two (the baseline's distinct-2 is at most 1, so the relative
            // bar is at most `ratio`), which means the declaration says
            // "relative" and behaves absolutely — a criterion that reads as
            // one thing and gates as another.
            guard backstop < ratio else {
                throw ExperimentError(
                    reason: coherenceOrderRefusal(ratio: ratio, backstop: backstop))
            }
            floor = backstop
        } else {
            floor = declaredFloor ?? defaultCoherenceFloor
            guard floor.isFinite, floor >= 0, floor <= 1 else {
                throw ExperimentError(
                    reason: "sweep.selection coherenceFloor must be a finite "
                        + "number in [0, 1] — got \(floor)")
            }
        }
        let margin = spec?.controls?.matchedNormRandomMargin
        if let margin {
            guard margin.isFinite, margin >= 0 else {
                throw ExperimentError(
                    reason: "sweep.selection matchedNormRandomMargin must be a "
                        + "finite number ≥ 0 — got \(margin)")
            }
        }
        // Control application mode (cross-engine contract with the server's
        // `resolve_selection` — identical refusal strings).
        let applyTo = spec?.controls?.applyTo ?? "winner"
        let topK = spec?.controls?.topK
        guard applyTo == "winner" || applyTo == "topK" else {
            throw ExperimentError(
                reason: "sweep.selection controls.applyTo must be 'winner' "
                    + "or 'topK' — got '\(applyTo)'")
        }
        if applyTo != "winner", margin == nil {
            throw ExperimentError(
                reason: "sweep.selection controls.applyTo declares how the "
                    + "matched-norm control is applied — declare "
                    + "matchedNormRandomMargin too")
        }
        if applyTo == "topK" {
            guard let topK, topK >= 1 else {
                throw ExperimentError(
                    reason: "sweep.selection controls.topK must be an "
                        + "integer ≥ 1 — got "
                        + (topK.map(String.init) ?? "nil"))
            }
        } else if topK != nil {
            throw ExperimentError(
                reason: "sweep.selection controls.topK is only read with "
                    + "applyTo: 'topK' — remove it, or declare applyTo")
        }
        return Resolved(
            metric: metric,
            capabilityTolerance: tolerance,
            coherenceFloor: floor,
            coherenceRatioToBaseline: relative
                ? (declaredRatio ?? defaultCoherenceRatio) : nil,
            matchedNormRandomMargin: margin,
            controlApplyTo: applyTo,
            controlTopK: applyTo == "topK" ? topK : nil)
    }

    /// One logprobShift measurement item: the study path's choice-row schema
    /// (`prompt`/`text` + `options` + optional `target`), with the target
    /// designation resolved exactly as the study runner resolves it —
    /// item `target`, else the first option.
    public struct ChoiceRow: Sendable, Equatable {
        public var id: String
        public var prompt: String
        public var options: [String]
        public var target: String

        public init(id: String, prompt: String, options: [String], target: String) {
            self.id = id
            self.prompt = prompt
            self.options = options
            self.target = target
        }
    }

    /// One concept's logprobShift instrument: the declared file, the
    /// SHA-256 of its raw bytes, and the parsed rows.
    public struct ChoiceSet: Sendable, Equatable {
        public var file: String
        public var hash: String
        public var rows: [ChoiceRow]
    }

    /// The per-metric instrument config a sweep runs with, resolved at sweep
    /// START (never mid-grid). markerDensity carries nothing extra;
    /// logprobShift pins its choice file (path + SHA-256 + parsed rows);
    /// judgeScore carries the manifest's rubric pin + text and the judge
    /// panel (verbatim for provenance, resolved for execution).
    public struct ResolvedObjective: Sendable {
        public var metric: String
        // logprobShift — the singular declaration
        public var choicePromptsFile: String?
        public var choicePromptsHash: String?
        public var choiceRows: [ChoiceRow] = []
        /// logprobShift — the per-concept declaration (choicePromptsFiles,
        /// 2026-08-02): each concept's cells are scored on its OWN rows.
        public var choiceSets: [String: ChoiceSet]?
        // judgeScore
        public var judgeRubricHash: String?
        public var judges: [ExperimentManifest.JudgeRef] = []
        var judgeRubricText: String?
        var judgePanel: [ExperimentTasks.ResolvedJudge] = []

        public init(metric: String) { self.metric = metric }

        /// The instrument this CONCEPT's cells are scored on: its own set
        /// under the per-concept declaration, else the study-wide one.
        /// Coverage is validated at resolve time, so a miss here throws
        /// loudly rather than scoring on a silent empty instrument.
        public func choiceSet(for concept: String) throws -> ChoiceSet {
            if let choiceSets {
                guard let chosen = choiceSets[concept] else {
                    throw ExperimentError(
                        reason: "no choice set resolved for concept "
                            + "'\(concept)' — resolveObjective validates "
                            + "coverage at sweep start, so this concept was "
                            + "not part of the resolved manifest")
                }
                return chosen
            }
            return ChoiceSet(
                file: choicePromptsFile ?? "",
                hash: choicePromptsHash ?? "", rows: choiceRows)
        }
    }

    /// The engine's path resolution for a choice-prompts file: absolute
    /// paths pass through, relative paths resolve against the workspace
    /// root. Shared with the authoring preview
    /// (`SweepSpecForm.previewChoicePrompts`) so the two can never disagree
    /// about which file is meant.
    public static func choicePromptsURL(file: String, root: URL) -> URL {
        file.hasPrefix("/") ? URL(filePath: file) : root.appending(path: file)
    }

    /// Load + validate a logprobShift choice-prompt file. Refusals here are
    /// the resolve-time gate: no declared file, a missing/unreadable file,
    /// no rows, a row without ≥2 options, or a target outside its options.
    /// (The token-count option-length guard additionally fires during the
    /// baseline choice pass — after model load, still before any grid cell.)
    public static func loadChoiceRows(
        file: String?, root: URL
    ) throws -> (rows: [ChoiceRow], file: String, hash: String) {
        guard let file, !file.trimmingCharacters(in: .whitespaces).isEmpty else {
            throw ExperimentError(
                reason: "logprobShift objective needs sweep.selection.objective."
                    + "choicePromptsFile — a JSONL of choice rows "
                    + "(prompt + ≥2 options) the shift is measured on")
        }
        let url = choicePromptsURL(file: file, root: root)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ExperimentError(
                reason: "logprobShift choice prompts file not found: \(url.path)")
        }
        let data = try Data(contentsOf: url)
        let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        let prompts = try ExperimentTasks.parseTaskPrompts(data)
        guard !prompts.isEmpty else {
            throw ExperimentError(
                reason: "logprobShift choice prompts '\(file)' has no rows")
        }
        var rows: [ChoiceRow] = []
        for prompt in prompts {
            guard let options = prompt.options, options.count >= 2 else {
                throw ExperimentError(
                    reason: "logprobShift choice prompts '\(file)' row "
                        + "'\(prompt.id)' needs at least 2 options")
            }
            let target = (prompt.target?.isEmpty == false ? prompt.target : nil)
                ?? options[0]
            guard options.contains(target) else {
                throw ExperimentError(
                    reason: "logprobShift choice prompts '\(file)' row "
                        + "'\(prompt.id)': target '\(target)' is not one of "
                        + "its options")
            }
            rows.append(
                ChoiceRow(id: prompt.id, prompt: prompt.text,
                          options: options, target: target))
        }
        return (rows, file, hash)
    }

    /// Resolve the objective's instrument config at sweep START, before the
    /// model loads — never mid-grid. logprobShift reads and hashes the
    /// criterion's choice file; judgeScore takes its config from the
    /// MANIFEST pins (rubric file + hash, judge panel) and refuses — naming
    /// the judge — when a Claude judge has no API credential or a LOCAL
    /// judge names a model other than the study model (a local judge with
    /// no model resolves to the study model; the local sweep holds one
    /// loaded model). (The ≥2-judge freeze gate for evidence is unchanged;
    /// screening may use one judge.)
    public static func resolveObjective(
        criterion: Resolved,
        spec: ExperimentManifest.SweepSelection?,
        manifest: ExperimentManifest,
        hasClaudeCredential: Bool? = nil,
        hasOpenRouterCredential: Bool? = nil,
        root: URL? = nil
    ) throws -> ResolvedObjective {
        var objective = ResolvedObjective(metric: criterion.metric)
        switch criterion.metric {
        case "logprobShift":
            let declared = spec?.objective?.choicePromptsFile
            let declaredMap = spec?.objective?.choicePromptsFiles
            if declared != nil, declaredMap != nil {
                throw ExperimentError(
                    reason: "sweep.selection.objective declares both "
                        + "choicePromptsFile and choicePromptsFiles — the "
                        + "sweep cannot know which instrument scores a "
                        + "concept; declare exactly one")
            }
            if let declaredMap {
                // The per-concept form (2026-08-02): every attached concept
                // needs its own entry, and no entry may name a concept the
                // study does not attach — a typo here would otherwise leave
                // some concept scored on the wrong file.
                guard !declaredMap.isEmpty else {
                    throw ExperimentError(
                        reason: "choicePromptsFiles must be a non-empty "
                            + "object of {concept: path} — one choice file "
                            + "per attached concept")
                }
                for (name, rel) in declaredMap
                where rel.trimmingCharacters(in: .whitespaces).isEmpty {
                    throw ExperimentError(
                        reason: "choicePromptsFiles['\(name)'] must be a "
                            + "file path (got '\(rel)')")
                }
                let attached = manifest.concepts.map(\.name)
                let unknown = Set(declaredMap.keys).subtracting(attached).sorted()
                if !unknown.isEmpty {
                    throw ExperimentError(
                        reason: "choicePromptsFiles names concepts the study "
                            + "does not attach: "
                            + unknown.joined(separator: ", ")
                            + " — a typo here would otherwise leave some "
                            + "concept scored on the wrong file")
                }
                let missing = attached.filter { declaredMap[$0] == nil }
                if !missing.isEmpty {
                    throw ExperimentError(
                        reason: "choicePromptsFiles is missing concepts this "
                            + "sweep would select for: "
                            + missing.joined(separator: ", ")
                            + " — every attached concept needs its own "
                            + "choice file (or declare the single "
                            + "choicePromptsFile)")
                }
                var sets: [String: ChoiceSet] = [:]
                for name in attached {
                    let (rows, file, hash) = try loadChoiceRows(
                        file: declaredMap[name],
                        root: root ?? VectorCatalog.projectRoot)
                    sets[name] = ChoiceSet(file: file, hash: hash, rows: rows)
                }
                objective.choiceSets = sets
                break
            }
            let (rows, file, hash) = try loadChoiceRows(
                file: declared,
                root: root ?? VectorCatalog.projectRoot)
            objective.choicePromptsFile = file
            objective.choicePromptsHash = hash
            objective.choiceRows = rows
        case "judgeScore":
            guard manifest.judgeRubricFile != nil,
                let pinnedHash = manifest.judgeRubricHash
            else {
                throw ExperimentError(
                    reason: "judgeScore objective needs a pinned judge rubric "
                        + "(judgeRubricFile + judgeRubricHash) in the manifest "
                        + "— pin one under prompts/rubrics/ before sweeping")
            }
            guard let judges = manifest.judges, !judges.isEmpty else {
                throw ExperimentError(
                    reason: "judgeScore objective needs at least one judge "
                        + "pinned in manifest.judges")
            }
            let panel = try ExperimentTasks.resolvedJudges(
                manifest: manifest, evaluation: nil)
            let credential =
                hasClaudeCredential ?? (ClaudeStimulusGenerator.apiKey != nil)
            for judge in panel where judge.kind == "claude" && !credential {
                throw ExperimentError(
                    reason: "judge '\(judge.name)' (claude) has no API "
                        + "credential — set ANTHROPIC_API_KEY or save a key in "
                        + "the Compute section before the sweep starts")
            }
            // Local Swift sweeps judge INLINE only (deferral is a
            // cluster-emission concept), so an openrouter judge needs its
            // key before the model loads, exactly like a claude judge.
            let openRouterCredential = hasOpenRouterCredential
                ?? (JudgeKeyStore.resolveKey(kind: "openrouter") != nil)
            for judge in panel
            where judge.kind == "openrouter" && !openRouterCredential {
                throw ExperimentError(
                    reason: "judge '\(judge.name)' (openrouter) has no API "
                        + "credential — save an external judge key in the "
                        + "Compute section or set OPENROUTER_API_KEY before "
                        + "the sweep starts")
            }
            // One-model-slot rule for THIS engine's sweep (the server applies
            // its own capacity policy): a local judge either uses the study
            // model — an empty judge model already resolved to it — or the
            // sweep refuses here, before the model loads.
            if let problem = ExperimentTasks.localJudgeSlotProblem(
                panel, studyModelID: manifest.modelID,
                studyRevision: manifest.modelRevision)
            {
                throw ExperimentError(reason: problem)
            }
            // The pinned rubric file, drift-checked at read time.
            let rubric = try JudgeRubricStore.resolveRubric(
                for: manifest, inlineRubric: nil)
            // A coding rubric declares no preference — the judgeScore
            // objective would force an improvised winner (2026-08-04;
            // server twin: `response_coding.refuse_if_coding`).
            try ResponseCoding.refuseIfCoding(
                rubric.text, context: "the sweep's judgeScore objective",
                rubricFile: rubric.file)
            objective.judgeRubricHash = pinnedHash
            objective.judgeRubricText = rubric.text
            objective.judges = judges
            objective.judgePanel = panel
        default:
            break
        }
        return objective
    }

    /// The baseline (no-injection) cell's objective value, by construction:
    /// markerDensity measures the baseline texts; judgeScore is the 0.5 tie
    /// (a response never beats itself); logprobShift is 0 (no shift from
    /// itself). `select` then requires the winner to EXCEED this.
    public static func baselineMetric(
        _ metric: String, baselineDensity: Double
    ) -> Double {
        switch metric {
        case "judgeScore": 0.5
        case "logprobShift": 0
        default: baselineDensity
        }
    }

    /// One measured layer×alpha cell (`alpha` in residual-norm units).
    public struct Cell: Sendable, Equatable {
        public var layer: Int
        public var alpha: Double
        public var metric: Double  // objective metric value (markerDensity today)
        public var distinct2: Double
        public var batteryAccuracy: Double
        /// Mean output length in whitespace words — carried so the selection
        /// can stamp `lengthInflated` for the winning cell (server twin:
        /// `SweepCell.words`). Nil when the cell was rebuilt from a record
        /// that predates the column, in which case the flag is simply not
        /// stamped. Selection never reads it.
        public var words: Double?

        public init(
            layer: Int, alpha: Double, metric: Double,
            distinct2: Double, batteryAccuracy: Double,
            words: Double? = nil
        ) {
            self.layer = layer
            self.alpha = alpha
            self.metric = metric
            self.distinct2 = distinct2
            self.batteryAccuracy = batteryAccuracy
            self.words = words
        }
    }

    /// The no-injection cell the constraints are anchored to.
    public struct Baseline: Sendable, Equatable {
        public var metric: Double
        public var distinct2: Double
        public var batteryAccuracy: Double

        public init(metric: Double, distinct2: Double, batteryAccuracy: Double) {
            self.metric = metric
            self.distinct2 = distinct2
            self.batteryAccuracy = batteryAccuracy
        }
    }

    /// The winning cell, or nil when nothing passes.
    ///
    /// A cell is eligible when its battery accuracy stays within
    /// `capabilityTolerance` of baseline AND its distinct-2 stays at or above
    /// `coherenceFloor`; the best eligible cell by highest objective metric
    /// wins — and must actually EXCEED the baseline metric (a "winner" that
    /// expresses no more than baseline is no winner).
    public static func select(
        cells: [Cell], baseline: Baseline, criterion: Resolved
    ) -> Cell? {
        var best: Cell?
        for cell in cells {
            let eligible =
                cell.batteryAccuracy >= baseline.batteryAccuracy - criterion.capabilityTolerance
                && coherencePasses(
                    distinct2: cell.distinct2,
                    baselineDistinct2: baseline.distinct2, criterion: criterion)
            guard eligible else { continue }
            if cell.metric > (best?.metric ?? baseline.metric) {
                best = cell
            }
        }
        return best
    }

    /// WHY no cell was selected — the constraints, or the objective.
    ///
    /// This engine always said "no cell passed the capability/coherence
    /// gates". That is one of two possible reasons and often the wrong one. A
    /// grid whose cells are all perfectly eligible but none of which beats the
    /// baseline objective is a completely different result: the constraints
    /// were never the obstacle, the direction simply did not move the
    /// objective the declared way. Reported as a gate failure, it sends the
    /// researcher to loosen a tolerance that was never binding.
    ///
    /// Observed live on the server (2026-07-26): a practicalwisdom sweep
    /// where all 36 cells sat inside both constraints and every objective
    /// value was NEGATIVE — the vector moved the objective the opposite way —
    /// yet the run recorded "no cell passed the capability/coherence gates".
    /// The server learned the distinction then; this engine kept printing the
    /// old sentence for the same grid until review round 9, finding 6.
    ///
    /// Server twin: `sweep_selection.no_selection_reason`, word for word.
    public static func noSelectionReason(
        cells: [Cell], baseline: Baseline, criterion: Resolved
    ) -> String {
        guard !cells.isEmpty else { return "the sweep measured no cells" }
        let eligible = cells.filter {
            $0.batteryAccuracy >= baseline.batteryAccuracy - criterion.capabilityTolerance
                && coherencePasses(
                    distinct2: $0.distinct2,
                    baselineDistinct2: baseline.distinct2, criterion: criterion)
        }
        guard let best = eligible.map(\.metric).max() else {
            return "no cell passed the capability/coherence gates "
                + "(tolerance \(g(criterion.capabilityTolerance)), "
                + "\(criterion.coherenceSummary))"
        }
        let blocked = cells.count - eligible.count
        let detail =
            blocked > 0
            ? "; \(blocked) of \(cells.count) cells also failed the "
                + "capability/coherence gates"
            : "; all \(cells.count) cells were inside both constraints"
        let direction =
            best < baseline.metric
            ? " — the objective moved the OPPOSITE way from the declared "
                + "direction"
            : ""
        return "no eligible cell beat the baseline \(criterion.metric) "
            + "(\(g(best)) vs baseline \(g(baseline.metric)))\(direction)"
            + "\(detail)"
    }

    /// Python's `:g` for the numbers these cross-engine sentences carry, so
    /// the twin texts are equal byte for byte and not merely in wording.
    static func g(_ value: Double) -> String {
        String(format: "%g", value)
    }

    /// The top `k` PROMOTABLE cells in objective order: eligible under the
    /// capability/coherence constraints AND beating the baseline objective —
    /// exactly the cells `select` would pick if the ones above them
    /// vanished. The topK control walks this list and promotes the first
    /// cell that beats its OWN matched-norm control. Server twin:
    /// `ranked_candidates`.
    public static func rankedCandidates(
        cells: [Cell], baseline: Baseline, criterion: Resolved, k: Int
    ) -> [Cell] {
        // Tie-break is CONTRACT, not accident (review 2026-08-03): objective
        // descending, then declared grid order. Swift's `sorted` documents
        // no stability, so ties carry an explicit grid index — the server's
        // stable sort yields the identical ranking; judge-score ties (0.5
        // steps) make this reachable in practice.
        cells
            .enumerated()
            .filter {
                $0.element.batteryAccuracy
                    >= baseline.batteryAccuracy - criterion.capabilityTolerance
                    && coherencePasses(
                        distinct2: $0.element.distinct2,
                        baselineDistinct2: baseline.distinct2,
                        criterion: criterion)
                    && $0.element.metric > baseline.metric
            }
            .sorted {
                $0.element.metric != $1.element.metric
                    ? $0.element.metric > $1.element.metric
                    : $0.offset < $1.offset
            }
            .prefix(max(k, 0))
            .map(\.element)
    }

    /// True when the winner beats the matched-norm random control by at
    /// least `margin` — the noise-floor check applied AFTER selection.
    public static func controlPasses(
        bestMetric: Double, controlMetric: Double, margin: Double
    ) -> Bool {
        bestMetric - controlMetric >= margin
    }

    /// One evaluated candidate control (topK mode) — stamped into selection
    /// provenance as `controlsEvaluated`.
    public struct EvaluatedControl: Sendable, Equatable {
        public var layer: Int
        public var alpha: Double
        public var metricValue: Double
        public var controlMetricValue: Double
        public var passed: Bool
    }

    /// Server twin: `top_k_control_failure_message`.
    public static func topKControlFailureMessage(
        _ evaluated: [EvaluatedControl], margin: Double
    ) -> String {
        let cells = evaluated
            .map {
                "L\($0.layer) α\($0.alpha): \($0.metricValue) vs control "
                    + "\($0.controlMetricValue)"
            }
            .joined(separator: "; ")
        return "all \(evaluated.count) top candidate cell(s) failed the "
            + "matched-norm control margin (\(margin)): \(cells)"
    }

    public static func controlFailureMessage(
        bestMetric: Double, controlMetric: Double, margin: Double
    ) -> String {
        "winning cell failed the matched-norm control margin "
            + "(best \(bestMetric) vs control \(controlMetric), margin \(margin))"
    }

    /// Deterministic seed for a control cell's random direction, derived from
    /// stable identifiers only (condition|concept|layer — the same seed-text
    /// convention as the server's `_matched_norm_random`; the RNGs differ per
    /// substrate, which is fine — random controls never cross engines, they
    /// only need to be reproducible within one).
    public static func controlSeed(_ seedText: String) -> UInt64 {
        let digest = SHA256.hash(data: Data(seedText.utf8))
        return digest.prefix(8).reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
    }

    /// A reproducible random direction rescaled to `norm` for control cells.
    public static func controlVector(
        seedText: String, dimension: Int, norm: Float
    ) throws -> [Float] {
        var rng = SplitMix64(seed: controlSeed(seedText))
        return try SteeringVectorMath.randomVector(
            dimension: dimension, norm: norm, using: &rng)
    }
}
