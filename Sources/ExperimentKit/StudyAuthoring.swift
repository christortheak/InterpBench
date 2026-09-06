import Foundation

// Study-authoring streamline (2026-07-19, researcher walkthrough feedback):
// the manifest is the single study.json — this file adds (1) a PRESENTATION
// classification of what the study is trying to do, so the UI can show only
// relevant sections with a real explanation, and (2) manifest JSON
// export/import, so a study is buildable by hand, by the app, or by an LLM
// and pasted in — with the same firewall (verify + freeze gates) either way.

/// THE study classifier (2026-07-19 second pass): one question, asked once,
/// at the top of the page. Replaces the former trio of overlapping controls
/// (manifest studyKind picker / "Study stage" / "Study Focus") whose
/// disagreements the researcher had to reconcile. A VIEW-layer
/// classification derived from manifest content, never stored in it (the
/// manifest stays the cross-engine contract; hiding a section must never
/// change the data) — but the picker DOES write the manifest's studyKind
/// on drafts via `mappedKind`, so the one control the user sets and the
/// one that filters the page can never contradict each other again.
public enum StudyIntent: String, CaseIterable, Sendable, Identifiable {
    /// Derive steering conditions from concepts and measure them. TWO
    /// PHASES under one type (2026-07-19 fold-in): the SCREEN phase runs
    /// the derivation funnel (extract → validate → sweep → promote →
    /// run); the CONFIRM phase re-tests one promoted agent's cell under
    /// a perturbation policy (α ± δ, matched-norm control) on held-out
    /// prompts. The manifest's `phase` field selects the variation —
    /// mechanically both are the same concept machinery.
    case conceptStudy
    /// Compare saved agents (and/or a baseline) on task prompts, a
    /// capability battery, and optional judging. No concept machinery.
    case agentComparison
    /// Multi-agent scenario transcripts (panels, turns, visibility).
    case multiAgent

    public var id: String { rawValue }

    /// Parse INCLUDING legacy aliases: "confirmAgent" was a top-level
    /// study type until 2026-07-19 and lives on in shipped manifests and
    /// LLM-authored packs — it reads as the concept study (its confirm
    /// phase).
    public static func parse(_ raw: String) -> StudyIntent? {
        raw == "confirmAgent" ? .conceptStudy : StudyIntent(rawValue: raw)
    }

    public var displayName: String {
        switch self {
        case .conceptStudy: "Concept study (derive & test)"
        case .agentComparison: "Compare agents"
        case .multiAgent: "Multi-agent scenario"
        }
    }

    /// The manifest kind this study type writes on DRAFTS. Every type maps
    /// — the picker never leaves the manifest ambiguous (the old "pointer
    /// stages" that mapped to nothing are gone from the vocabulary; a
    /// picker should not offer things the page cannot do).
    public var mappedKind: ExperimentManifest.StudyKind {
        self == .multiAgent ? .multiAgent : .modelOutput
    }

    /// The display-pane explanation: what this study kind is, which data it
    /// needs, and what is researcher-only. Shown verbatim in the app.
    public var explanation: String {
        switch self {
        case .agentComparison:
            return """
            Compare saved agents against a baseline. You need: the base \
            model, the agents to compare (Add agent — each is pinned by \
            path + hash), and TASK PROMPTS (the prompts every arm answers, \
            alongside any declared system prompt). \
            Optional: a capability battery (did general ability survive?), \
            judges + a rubric (blinded A/B judging of outputs), and a \
            reasoning-style taxonomy. Not needed: concepts, conditions, \
            controls, sweeps — that machinery derives injection conditions, \
            and your arms are already-built agents. Description, task \
            description, outcome measures, phase, and case family are notes \
            for the record — never sent to any model.
            """
        case .conceptStudy:
            return """
            Derive steering conditions from concept data and measure their \
            effect, in two phases. SCREEN (the default): concept stimuli → \
            extract a vector → validate it (never-named scenarios + \
            cross-concept cosines) → sweep layer×alpha on dev prompts → \
            promote the winning cell to an agent → run the frozen \
            condition matrix. CONFIRM (set Funnel phase to "confirm"): \
            re-test one promoted agent's cell at α ± δ against a \
            matched-norm control on HELD-OUT prompts, as a NEW \
            preregistered study — Holm-corrected, with the held-out pool \
            enforced at verify. You need: concept data (stimuli, \
            validation, markers), task prompts, and — for evidence-grade \
            claims — a pinned judge rubric with a judge panel (two or more \
            distinct judges buy inter-rater agreement; one is legal and \
            reports none). Conditions are \
            named injection arms. Matched-norm random controls help test \
            whether effects exceed comparable random perturbations; they \
            do not establish construct specificity. The whole funnel can run as one declared pipeline. Description, \
            outcome measures, and case family are notes for the record — \
            never sent to any model.
            """
        case .multiAgent:
            return """
            Run a multi-agent scenario (panel of agents, turn structure, \
            visibility rules) and record the transcript. You need: a pinned \
            scenario file and the agents it casts. Task prompts do not \
            apply — the scenario file defines what agents see. Judging \
            applies to transcripts if declared. Description and the science \
            manifest are notes for the record — never sent to any model.
            """
        }
    }

    // MARK: Structured guide (rendered in the display pane)

    /// One item the researcher provides for a study of this type.
    public struct GuideItem: Sendable, Identifiable {
        public let name: String
        public let detail: String
        public let required: Bool
        public var id: String { name }

        public init(_ name: String, _ detail: String, required: Bool) {
            self.name = name
            self.detail = detail
            self.required = required
        }
    }

    public var systemImage: String {
        switch self {
        case .conceptStudy: "waveform.path.ecg"
        case .agentComparison: "person.2.crop.square.stack"
        case .multiAgent: "person.3.sequence"
        }
    }

    /// One-line answer to "what is this?" — shown beside the picker; the
    /// full guide renders in the display pane.
    public var tagline: String {
        switch self {
        case .conceptStudy:
            "Derive steering vectors from concept data, tune them, and "
                + "measure what they change — screen phase; the confirm "
                + "phase re-tests a promoted cell on held-out prompts."
        case .agentComparison:
            "Run saved agents and a baseline over the same prompts and "
                + "compare their outputs."
        case .multiAgent:
            "Run a scripted panel of agents and record the transcript."
        }
    }

    /// The guide's opening paragraph: what this study type IS and how it
    /// runs, in full sentences.
    public var whatItIs: String {
        switch self {
        case .conceptStudy:
            return """
            A concept study has two phases under one type. The SCREEN \
            phase runs the full derivation funnel: from your concept data \
            the engine extracts a steering vector (a direction in the \
            model's residual stream), validates that the vector actually \
            reads its concept, sweeps layer × strength on dev prompts to \
            find the best injection cell, promotes that winning cell to a \
            named agent, and runs the frozen condition matrix — every \
            task prompt through every condition, paired to the same-case \
            baseline. The whole funnel can run as one declared pipeline \
            with scientific stop conditions (gates) between stages.

            The CONFIRM phase (set Funnel phase to "confirm") is the \
            second half of screen → confirm: one promoted agent — \
            normally carrying a sweep-selection birth certificate — is \
            re-run on HELD-OUT prompts at its anchor strength and at \
            declared symmetric offsets (α ± δ), optionally against a \
            matched-norm random control. A real effect grows and shrinks \
            with the dose and beats the control; confirm-phase analysis \
            uses the stricter Holm correction, and verify enforces that \
            the prompt pool is disjoint from the screen's. A confirmation \
            is always a NEW preregistered study (duplicate, never edit) — \
            testing the screening claim inside the object that generated \
            it would be exactly the circularity the firewall prevents.
            """
        case .agentComparison:
            return """
            An agent comparison takes agents that already exist — saved \
            in your library, or forward-referenced from a sweep that has \
            not run yet — and runs each of them, plus the unmodified \
            baseline model, over the same task prompts. Outputs are \
            paired per prompt, so every comparison is like-for-like; the \
            capability battery runs through every condition to show \
            whether general ability survived; judging (if declared) is \
            blinded A/B against baseline.
            """
        case .multiAgent:
            return """
            A multi-agent study runs a pinned scenario file — which \
            agents sit on the panel, who speaks when, and who sees what — \
            and records the full transcript as the run artifact. \
            Optionally it also records a stripped-baseline transcript: \
            the same panel with every steering vector and adapter \
            removed, so the intervention's effect on the group dynamic is \
            visible by comparison. Task prompts do not apply; the \
            scenario file defines everything the agents see.
            """
        }
    }

    /// What the researcher provides, item by item.
    public var youProvide: [GuideItem] {
        switch self {
        case .conceptStudy:
            return [
                GuideItem(
                    "Concept data",
                    "stimulus texts per concept (the extraction recipe), "
                        + "plus a never-named validation set and optional "
                        + "markers", required: true),
                GuideItem(
                    "Task prompts",
                    "what the model answers in the measured run (alongside "
                        + "any declared system prompt)",
                    required: true),
                GuideItem(
                    "Judges + rubric",
                    "a pinned judge panel and a rubric file for evidence-grade "
                        + "blinded A/B judging (two or more distinct judges "
                        + "buy inter-rater agreement)", required: false),
                GuideItem(
                    "Capability battery",
                    "held-out ability probes run through every condition",
                    required: false),
                GuideItem(
                    "Confirm phase: perturbation policy",
                    "α deltas + matched-norm control around one promoted "
                        + "agent's cell, expanded into hashed conditions "
                        + "(set Funnel phase to \"confirm\")",
                    required: false),
                GuideItem(
                    "Confirm phase: held-out task prompts",
                    "disjoint from the sweep's dev split — reusing dev "
                        + "prompts would confirm the selection, not the "
                        + "effect", required: false),
                GuideItem(
                    "Human baseline",
                    "measured human effect sizes — required only for "
                        + "human-anchored (R = Δmodel − Δhuman) claims",
                    required: false),
            ]
        case .agentComparison:
            return [
                GuideItem(
                    "Agents",
                    "saved agents from the library (pinned by path + hash), "
                        + "or forward references to a sweep's future "
                        + "promotion", required: true),
                GuideItem(
                    "Task prompts",
                    "the prompts every arm answers", required: true),
                GuideItem(
                    "Judges + rubric",
                    "blinded A/B judging of outputs vs baseline",
                    required: false),
                GuideItem(
                    "Capability battery",
                    "did general ability survive the intervention?",
                    required: false),
            ]
        case .multiAgent:
            return [
                GuideItem(
                    "Scenario file",
                    "the panel, turn structure, and visibility rules, "
                        + "pinned by hash", required: true),
                GuideItem(
                    "Agents",
                    "the saved agents the scenario casts", required: true),
                GuideItem(
                    "Judges + rubric",
                    "transcript-level judging, if declared",
                    required: false),
            ]
        }
    }

    /// What the study measures / what its results can claim.
    public var itMeasures: [String] {
        switch self {
        case .conceptStudy:
            return [
                "Whether each concept vector moves behavior at all — and at "
                    + "which layer and strength (the sweep grid)",
                "Effect sizes per condition, paired to the same-case "
                    + "baseline, with bootstrap confidence intervals",
                "Direction vs energy: matched-norm random controls test "
                    + "whether the effect exceeds nonspecific perturbation; "
                    + "they do not rule out confounds or establish construct validity",
                "Capability cost: the battery says whether steering broke "
                    + "general ability",
                "Confirm phase: replication on held-out prompts, dose "
                    + "response across α ± δ, and beating the matched-norm "
                    + "control",
            ]
        case .agentComparison:
            return [
                "Output differences between each agent and baseline on "
                    + "identical prompts (paired, never cross-prompt)",
                "Judge preferences (blinded A/B) and structured fields, "
                    + "when judging is declared",
                "Capability battery per condition — ability survival",
            ]
        case .multiAgent:
            return [
                "The transcript itself — group dynamics under intervention",
                "Intervention vs stripped baseline: what the steering "
                    + "changed in the panel's behavior",
            ]
        }
    }

    /// Pipeline stages that make sense for this intent — the composer
    /// shows only these (a chain for an agent-comparison study has nothing
    /// to extract or sweep).
    public var relevantPipelineStages: [String] {
        switch self {
        case .agentComparison: ["run", "evaluate", "analyze"]
        case .conceptStudy:
            ["extract", "validate", "sweep", "promote", "run", "evaluate",
             "analyze"]
        case .multiAgent: ["run"]
        }
    }

    /// Readiness categories relevant to this intent (see
    /// `DataRequirement.Kind.authoringCategory`).
    public var relevantDataCategories: Set<DataCategory> {
        switch self {
        case .agentComparison:
            [.taskPrompts, .battery, .judging, .style]
        case .conceptStudy:
            [.taskPrompts, .conceptData, .battery, .judging, .style,
             .comparison]
        case .multiAgent: [.scenario, .judging, .style]
        }
    }

    /// The effective study type. A DECLARED `manifest.studyType` (written
    /// by the picker on drafts, durable across selection changes; legacy
    /// "confirmAgent" reads as conceptStudy via `parse`) wins — as long
    /// as it is consistent with the engine-facing studyKind, the
    /// fail-safe against hand-edited JSON that says one thing and runs
    /// another. Otherwise derive from content: a perturbation policy is
    /// the concept study's confirm phase; concepts (or conditions or a
    /// declared sweep) win over variants — a hybrid study is a concept
    /// study that also carries agents.
    public static func derive(from manifest: ExperimentManifest) -> StudyIntent {
        if let declared = manifest.studyType.flatMap(StudyIntent.parse(_:)),
            declared.mappedKind == manifest.studyKind
        {
            return declared
        }
        if manifest.studyKind == .multiAgent { return .multiAgent }
        if manifest.perturbationPolicy != nil { return .conceptStudy }
        // Injection conditions count as concept-study evidence too: a
        // manifest carrying them without a DECLARED type is running the
        // concept machinery (conditions reference concept vectors).
        if !manifest.concepts.isEmpty || !manifest.conditions.isEmpty
            || manifest.sweep != nil
        {
            return .conceptStudy
        }
        if !manifest.variantConditions.isEmpty { return .agentComparison }
        return .conceptStudy
    }

    /// Data the manifest carries that this intent's filtered view would
    /// HIDE — surfaced so hiding never silently orphans content. Saving
    /// under one type PRESERVES the other types' configuration, so every
    /// direction needs its note.
    public func hiddenContentNote(for manifest: ExperimentManifest) -> String? {
        var carried: [String] = []
        switch self {
        case .agentComparison:
            if !manifest.concepts.isEmpty {
                carried.append("\(manifest.concepts.count) attached concept(s)")
            }
            if !manifest.conditions.isEmpty {
                carried.append(
                    "\(manifest.conditions.count) injection condition(s)")
            }
            if manifest.multiAgentScenarioPath != nil {
                carried.append("a pinned multi-agent scenario")
            }
        case .conceptStudy:
            if manifest.multiAgentScenarioPath != nil {
                carried.append("a pinned multi-agent scenario")
            }
        case .multiAgent:
            if !manifest.concepts.isEmpty {
                carried.append("\(manifest.concepts.count) attached concept(s)")
            }
            if !manifest.conditions.isEmpty {
                carried.append(
                    "\(manifest.conditions.count) injection condition(s)")
            }
            if !manifest.variantConditions.isEmpty {
                carried.append(
                    "\(manifest.variantConditions.count) agent condition(s)")
            }
            if manifest.taskPromptsFile != nil {
                carried.append("a pinned task-prompts file")
            }
        }
        guard !carried.isEmpty else { return nil }
        return "this study also carries " + carried.joined(separator: ", ")
            + " — switch the study type to see and edit them (nothing is "
            + "deleted by this view filter)"
    }
}

/// Grouping vocabulary for the Data & Prompts pane — one pane, subdivided
/// by what each file IS FOR, instead of two panes (Data Readiness + Input
/// Data) whose difference the researcher had to guess.
public enum DataCategory: String, CaseIterable, Sendable, Identifiable {
    case taskPrompts
    case conceptData
    case judging
    case battery
    case comparison
    case scenario
    case style
    case readout

    public var id: String { rawValue }

    /// Group titles NAME the pane each file feeds (2026-07-19 feedback:
    /// the researcher should be able to trace a data row to the pane that
    /// uses it without guessing).
    public var title: String {
        switch self {
        case .taskPrompts: "Task prompts — what every arm answers"
        case .conceptData:
            "Concept data — feeds Build & Validate Concept Vectors"
        case .judging: "Judging — feeds the Evaluation pane"
        case .battery: "Capability battery — runs through every arm (Evaluation)"
        case .comparison: "Comparison data — human baselines (for R claims)"
        case .scenario: "Scenario — feeds the Conditions pane (multi-agent)"
        case .style: "Reasoning style — output taxonomy (Evaluation)"
        case .readout:
            "J-Space readout — what is read from the residual stream"
        }
    }
}

extension DataRequirement.Kind {
    /// Which Data & Prompts subgroup a requirement renders under.
    public var authoringCategory: DataCategory {
        switch self {
        // The parser registry and exclusion/attention-check rows read and
        // grade task-prompt outputs — they render with the prompt set they
        // gate.
        case .taskPrompts, .numericParser, .exclusionRules: .taskPrompts
        case .conceptStimuli, .conceptValidation, .conceptMarkers,
             .neutralCorpus: .conceptData
        case .judgeRubric, .judgePanel: .judging
        case .capabilityBattery: .battery
        case .humanBaseline: .comparison
        case .multiAgentScenario: .scenario
        case .reasoningStyleTaxonomy: .style
        // Not a data FILE but a manifest declaration, and it belongs in
        // this checklist for the same reason the others do: every way it
        // can be wrong is a freeze refusal met at the end of authoring,
        // or — for retention — only after the run, when it is too late.
        case .jlensReadout: .readout
        }
    }
}

// MARK: - LLM co-authoring prompt

/// The "work with an LLM" bridge (2026-07-19): a researcher copies this
/// prompt into any capable LLM, the LLM interviews them and produces a
/// complete STUDY PACK — the manifest plus the data files it names — and
/// Paste Study JSON imports it as a draft with verification running
/// immediately. The prompt is DATA here (versioned, testable) so the
/// contract it teaches stays in sync with the code, and it is keyed to
/// the study TYPE the researcher selected.
public enum StudyCoauthoring {
    /// One maintained interview shared by the app, HTTP service and both CLIs.
    public static func prompt(for intent: StudyIntent = .conceptStudy) -> String {
        switch intent {
        case .conceptStudy: StudyInterviewText.conceptStudy
        case .agentComparison: StudyInterviewText.agentComparison
        case .multiAgent: StudyInterviewText.multiAgent
        }
    }
}

// MARK: - Manifest JSON export / import

extension ExperimentStore {

    /// The selected study as ONE JSON document — the same
    /// `experiment.json` every engine reads, pretty-printed for pasting
    /// into an editor or an LLM conversation.
    public static func exportStudyJSON(_ manifest: ExperimentManifest) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return String(decoding: try encoder.encode(manifest), as: UTF8.self)
    }

    /// Import a pasted study JSON as a NEW DRAFT. The firewall applies
    /// identically to pasted and hand-built studies:
    ///
    /// - The imported manifest is ALWAYS a draft — a pasted "frozen" status
    ///   would fake preregistration, so freeze metadata is stripped and the
    ///   one-way freeze must be earned through its gates.
    /// - The name must be free (no silent overwrite; rename in the JSON).
    /// - `verify()` runs immediately; violations are returned for display —
    ///   an importable-but-broken study arrives loudly annotated.
    ///
    /// STUDY PACKS (2026-07-19, "one file drives the study"): the document
    /// may be an envelope `{"study": {…}, "files": {"prompts/…": "…"}}`.
    /// The files land in the workspace FIRST (contained under prompts/,
    /// never overwriting differing bytes), then anything the manifest
    /// names but has not pinned (task prompts, rubric, battery) is pinned
    /// from the just-written bytes — so an LLM-authored pack arrives
    /// runnable without hand-computed hashes.
    public static func importStudyJSON(
        _ json: String
    ) throws -> (manifest: ExperimentManifest, violations: [String],
                 filesWritten: [String]) {
        let data = Data(json.utf8)
        let root = workspaceRoot
        let preview = try StudyPackAuthoring.preview(data, workspaceRoot: root)
        let result = try StudyPackAuthoring.apply(data, workspaceRoot: root,
            expectedReviewSHA256: preview.reviewSHA256)
        return (result.study.manifest, result.violations, result.filesWritten)
    }
}
