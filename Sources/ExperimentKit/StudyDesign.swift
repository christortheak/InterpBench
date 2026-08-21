import Foundation

// The design/study split as the UI expresses it (2026-08-06, researcher
// walkthrough).
//
// A TEMPLATE is an experimental DESIGN: everything that changes what a result
// MEANS — task prompts and their pin, instruments and their scope, sampling
// policy, parser, exclusions, battery, judges and rubric, pipeline, model, and
// (for a panel) the semantic scenario. A STUDY is a design BOUND to agents or a
// seat casting, plus the compute logistics that get it run. `StudyTemplate`
// already embodies the split; this file is what lets the interface say it: a
// read-only design summary, the "start a new study from…" picker model, and the
// live divergence check that tells a minted study whether it still IS the
// design it names.
//
// Decision logic only — no view may compute any of this, and nothing here is
// read by a run, a freeze, an analysis, or the Python engine.

// MARK: - What a new study starts from

/// The first choice in the new-study flow: a blank draft, or a saved design.
///
/// Modelled as a closed choice rather than an optional template name so the
/// blank path is a NAMED option in the picker instead of the absence of one —
/// "From scratch" is a decision the researcher makes, and an empty selection
/// reads as "I have not chosen yet".
public enum StudyDesignChoice: Hashable, Sendable, Identifiable {
    case fromScratch
    case design(String)

    /// Tag-safe id. The blank path takes a name no directory can have, so it
    /// can never collide with a design called "from-scratch".
    public var id: String {
        switch self {
        case .fromScratch: "\u{0}from-scratch"
        case .design(let name): name
        }
    }

    public var designName: String? {
        guard case .design(let name) = self else { return nil }
        return name
    }

    public var label: String {
        switch self {
        case .fromScratch: "From scratch"
        case .design(let name): name
        }
    }

    /// The picker's rows, blank path first.
    public static func choices(designs: [StudyTemplate]) -> [StudyDesignChoice] {
        [.fromScratch] + designs.map { .design($0.name) }
    }

    /// A remembered choice re-checked against the CURRENT library.
    ///
    /// A design can be renamed or deleted while a choice is parked in the
    /// picker; falling back to `.fromScratch` is the honest resolution, because
    /// the alternative — a selection naming a design that no longer exists —
    /// would mint nothing and say nothing about why.
    public static func resolve(
        _ choice: StudyDesignChoice, designs: [StudyTemplate]
    ) -> StudyDesignChoice {
        guard let name = choice.designName else { return .fromScratch }
        return designs.contains { $0.name == name } ? choice : .fromScratch
    }
}

// MARK: - The read-only design summary

/// What a design SAYS, rendered honestly: the manifest facts that decide what a
/// result means, in one scannable block.
///
/// Read-only on purpose. Revising a design is a round trip — instantiate it,
/// tweak the draft in the Studies editor, save the draft back as a design — and
/// a second manifest editor in the Templates tab would be a parallel
/// implementation of the Studies one that drifts from it the first time a field
/// is added to only one. (Settled scoping decision, 2026-08-06.)
public enum StudyDesignSummary {

    /// One labelled fact. Values are already display-ready — a view formats
    /// nothing.
    public struct Row: Sendable, Equatable, Identifiable {
        public let label: String
        public let value: String
        public var id: String { label }

        public init(_ label: String, _ value: String) {
            self.label = label
            self.value = value
        }
    }

    /// Absent is said out loud ("none", "every item") rather than left blank: a
    /// design with no judges and a design whose judges failed to render look
    /// identical when both are empty strings.
    public static func rows(for template: StudyTemplate) -> [Row] {
        let study = template.study
        var rows: [Row] = [
            Row("Study type", template.intent.displayName),
            Row("Model", modelLine(study)),
            Row("Task file", pinnedFile(study.taskPromptsFile, hash: study.taskPromptsHash)),
            Row("Instruments", instrumentLine(study)),
            Row("Instrument scope", scopeLine(study)),
            Row("Sampling", samplingLine(study)),
            Row("Prompting", promptLine(study)),
            Row("Parser", study.numericParser ?? "none declared"),
            Row("Exclusions", exclusionLine(study)),
            Row(
                "Capability battery",
                pinnedFile(
                    study.capabilityBatteryFile, hash: study.capabilityBatteryHash)),
            Row("Judging", judgingLine(study)),
            Row("Pipeline", study.pipeline == nil ? "none" : "declared (server-run)"),
        ]
        if !study.concepts.isEmpty {
            rows.append(
                Row("Concepts", study.concepts.map(\.name).joined(separator: ", ")))
        }
        if let scenario = template.semanticScenario {
            rows.append(
                Row("Panel", "\(scenario.path)  @ \(short(scenario.hash))"))
        }
        return rows
    }

    // MARK: Lines

    private static func modelLine(_ study: ExperimentManifest) -> String {
        guard let revision = study.modelRevision, !revision.isEmpty else {
            return "\(study.modelID)  ·  revision unpinned"
        }
        return "\(study.modelID)  @ \(short(revision))"
    }

    private static func pinnedFile(_ path: String?, hash: String?) -> String {
        guard let path, !path.isEmpty else { return "none" }
        guard let hash, !hash.isEmpty else { return "\(path)  ·  unpinned" }
        return "\(path)  @ \(short(hash))"
    }

    private static func instrumentLine(_ study: ExperimentManifest) -> String {
        let declared = study.outcomeInstruments ?? []
        guard !declared.isEmpty else { return "sampled text (default)" }
        var line = declared.joined(separator: ", ")
        if declared.contains("ordinalScale") {
            line += "  ·  aggregation: \(study.ordinalAggregation ?? "undeclared")"
        }
        return line
    }

    private static func scopeLine(_ study: ExperimentManifest) -> String {
        guard let scope = study.outcomeInstrumentScope else {
            return "every item"
        }
        return "\(scope.responseFormats.joined(separator: ", "))  ·  "
            + "\(scope.itemCount) item(s) @ \(short(scope.itemIDsHash))"
    }

    private static func samplingLine(_ study: ExperimentManifest) -> String {
        let samples = max(1, study.samplesPerItem ?? 1)
        var line = "\(samples) sample(s) × temperature \(number(study.temperature))"
        line += " × up to \(study.maxTokens) tokens"
        if study.temperature > 0 {
            // The substrate rule, stated where the number is read: a local MLX
            // run refuses a non-zero temperature outright.
            line += "  ·  server substrate only"
        }
        return line
    }

    private static func promptLine(_ study: ExperimentManifest) -> String {
        var parts = [(study.promptMode ?? .chatAssistant).label]
        if study.systemPrompt?.isEmpty == false { parts.append("system prompt set") }
        if study.qwenThinkingEnabled == true { parts.append("thinking ON") }
        return parts.joined(separator: "  ·  ")
    }

    private static func exclusionLine(_ study: ExperimentManifest) -> String {
        let rules = study.exclusionRules ?? []
        guard !rules.isEmpty else { return "none declared" }
        return rules.map { rule -> String in
            guard let endpoint = rule.endpoint, !endpoint.isEmpty else {
                return rule.rule
            }
            return "\(rule.rule)(\(endpoint))"
        }.joined(separator: ", ")
    }

    private static func judgingLine(_ study: ExperimentManifest) -> String {
        let judges = study.judges ?? []
        guard !judges.isEmpty || study.judgeRubricFile != nil else { return "none" }
        var parts: [String] = []
        if judges.isEmpty {
            parts.append("no judges pinned")
        } else {
            parts.append(
                "\(judges.count) judge(s): "
                    + judges.map(\.name).joined(separator: ", "))
        }
        parts.append(
            "rubric "
                + pinnedFile(study.judgeRubricFile, hash: study.judgeRubricHash))
        return parts.joined(separator: "  ·  ")
    }

    // MARK: Formatting

    static func short(_ hash: String) -> String {
        hash.count > 12 ? String(hash.prefix(12)) + "…" : hash
    }

    /// Temperatures read as `0`, `0.7` — never `0.0` or `0.7000000000000001`.
    static func number(_ value: Double) -> String {
        value == value.rounded()
            ? String(Int(value))
            : String(format: "%g", value)
    }
}

// MARK: - Does a minted study still agree with its design?

extension StudyTemplateStore {

    /// Whether a study still matches the design it was minted from.
    ///
    /// Editing a minted draft is LEGAL and must never be blocked — a study is
    /// where the researcher works, and a design is a starting point, not a
    /// contract. But a study whose sampling policy or instrument set has
    /// changed is no longer the replication its lineage line claims, and a
    /// researcher reading "from 'vg-arm-a'" on a study that now measures
    /// something else has been misled by the interface rather than by anything
    /// in the data. So: edits are allowed, and divergence is SAID.
    public enum DesignAgreement: Sendable, Equatable {
        /// No lineage stamp — a hand-authored study.
        case noLineage
        /// The stripped form still hashes to the stamp.
        case matches
        /// Some design-bearing setting changed after minting.
        case diverged
        /// The design it names is gone (renamed or deleted). Its own state:
        /// nothing can be compared, and calling that "diverged" would blame
        /// the study for a change made to the library.
        case designMissing
    }

    /// The lineage reading in full: TWO independent facts, because one of them
    /// answers a question the other cannot.
    ///
    /// `agreement` compares the study against its BIRTH CERTIFICATE — the hash
    /// stamped at mint time. That comparison is deliberate and stays (revising
    /// a design must never re-file older instances as diverged; see
    /// `saveStudyBackToDesign`). But it has a silent consequence: after a
    /// design is revised, every older instance keeps reading "matches" while
    /// the recipe under that NAME has moved on, so a lineage line that shows
    /// only `agreement` tells a researcher the study replicates the current
    /// 'vg-arm-a' when it replicates the retired one.
    ///
    /// `designRevised` is that second fact, reported beside the first rather
    /// than folded into it: the design still in the library under the stamped
    /// name no longer carries the stamped hash. It says nothing about whether
    /// the STUDY moved, and the study's own reading says nothing about
    /// whether the design did.
    public struct DesignLineage: Sendable, Equatable {
        /// Fact (a): does the study still hash to its birth certificate?
        public let agreement: DesignAgreement
        /// Fact (b): does the design that still bears the stamped NAME still
        /// carry the stamped hash? False whenever there is nothing to compare
        /// (no lineage, or the design is gone).
        public let designRevised: Bool

        public init(agreement: DesignAgreement, designRevised: Bool) {
            self.agreement = agreement
            self.designRevised = designRevised
        }
    }

    /// The live agreement check: strip the study exactly as `templateFromStudy`
    /// does, and compare the result to the hash stamped at mint time.
    ///
    /// Compared against `templateProvenance.templateHash` rather than the
    /// design's CURRENT hash: the stamp records what the study was minted from,
    /// which is the question "has this study drifted?" actually asks. (A design
    /// re-minted under the same name therefore also reads as divergence, which
    /// is correct — the study no longer matches the recipe under that name.)
    ///
    /// Whether the design ITSELF has since been revised is the separate fact
    /// `lineage(of:).designRevised` — use that call wherever the answer is
    /// shown to a researcher.
    ///
    /// File I/O per call (it re-reads the study's compiled panel), so callers
    /// cache it — `ExperimentPanel` computes it once per refresh, never per
    /// frame.
    public static func agreement(of study: ExperimentManifest) -> DesignAgreement {
        lineage(of: study).agreement
    }

    /// Both lineage facts from one pass over the files (see `DesignLineage`).
    ///
    /// File I/O per call, for the same reason `agreement(of:)` is: callers
    /// cache it per refresh, never per frame.
    public static func lineage(of study: ExperimentManifest) -> DesignLineage {
        guard let provenance = study.templateProvenance else {
            return DesignLineage(agreement: .noLineage, designRevised: false)
        }
        guard let design = try? load(name: provenance.template) else {
            return DesignLineage(agreement: .designMissing, designRevised: false)
        }
        // Fact (b), read off the design alone: its CURRENT content hash
        // against the one stamped into the study at mint time. `hash` covers
        // the design's body and semantic panel and excludes its name and
        // description, so a renamed note cannot read as a revision.
        let designRevised = hash(design) != provenance.templateHash
        func reading(_ agreement: DesignAgreement) -> DesignLineage {
            DesignLineage(agreement: agreement, designRevised: designRevised)
        }
        // The panel is compared STRUCTURALLY for the reason `templateFromStudy`
        // gives: the study's scenario is a compiled, seat-bound file and the
        // design's is the semantic form, so comparing bytes would call every
        // instance diverged.
        if let path = study.multiAgentScenarioPath, !path.isEmpty {
            guard let hoist = try? PanelComposition.hoistLegacyScenario(path: path),
                sameSemanticPanel(hoist.semantic, as: design)
            else { return reading(.diverged) }
        } else if design.semanticScenario != nil {
            return reading(.diverged)
        }
        var candidate = design
        candidate.study = strippedBody(study)
        return reading(
            hash(candidate) == provenance.templateHash ? .matches : .diverged)
    }
}
