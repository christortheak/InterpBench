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
            named injection arms; controls (matched-norm random) prove an \
            effect is the DIRECTION, not the perturbation energy. The \
            whole funnel can run as one declared pipeline. Description, \
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
                "Direction vs energy: matched-norm random controls prove an "
                    + "effect is the concept's direction, not perturbation "
                    + "noise",
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

    /// The type-specific interview + skeleton section.
    static func typeSection(for intent: StudyIntent) -> String {
        switch intent {
        case .conceptStudy:
            return """
            ## This study is a CONCEPT STUDY (studyType "conceptStudy")

            Derive steering vectors from concept data, then extract → \
            validate → sweep → promote → run — the full funnel, runnable \
            as one gated pipeline. Ask the researcher:
            - Which CONCEPTS, and what stimulus texts exist for each? \
            Paired extraction needs positive.jsonl + negative.jsonl under \
            prompts/concepts/<name>/ (one {"text": ...} per line); \
            grand-mean extraction needs prompts/emotions/<name>/\
            stories.jsonl. You can AUTHOR these with the researcher and \
            ship them in the pack's "files" map.
            - A never-named validation set per concept \
            (prompts/concepts/<name>/validation.jsonl: scenario rows \
            {"text": "...", "expresses": true} — "expresses" is a JSON \
            boolean, true for concept-expressing scenarios and false for \
            matched non-expressing ones; both loaders REFUSE any other \
            label shape) — the held-out check of whether a vector \
            SEPARATES its concept's scenarios (read the calibrated \
            accuracy and AUC diagnostics; the midpoint-threshold accuracy \
            alone can sit at chance for threshold reasons that say \
            nothing about the vector).
            - Task prompts (what the model answers in the measured run).
            - The sweep: layer × alpha grid and the selection criterion \
            (judgeScore / logprobShift whenever the claim is about a \
            substantive outcome — never markerDensity as the promotion \
            objective).
            - Gates for the pipeline (validation floor, distinctness cap, \
            selection-required).
            IMPORTANT: concepts enter the manifest via ATTACH in the app \
            (attaching computes the stimulus hashes) — leave "concepts": \
            [] in the JSON and tell the researcher to click Attach after \
            the pack imports; the stimulus files you shipped will be \
            found and pinned.

            CONFIRM PHASE: if the researcher is confirming an already \
            promoted agent (screen → confirm, phase 2), this is still a \
            concept study — set "phase": "confirm". It needs HELD-OUT \
            task prompts disjoint from the sweep's dev split (author them \
            together and ship them in "files"; the manifest must pin \
            screenTaskPromptsHash — the screen pool it is held out from). \
            The perturbation policy itself (anchor α ± δ, matched-norm \
            control) is declared in the app, where Attach Policy expands \
            it into hashed conditions.
            """
        case .agentComparison:
            return """
            ## This study is an AGENT COMPARISON (studyType "agentComparison")

            Run saved agents against the unmodified baseline over the \
            same task prompts. No concept derivation happens here. Ask \
            the researcher:
            - Which agents? Existing library agents are added IN THE APP \
            (their artifacts are pinned by hash); agents that do not \
            exist yet can be declared as forward references to a sweep — \
            but that makes it a concept study's machinery; keep this \
            pack focused on prompts + judging.
            - Task prompts: the prompts every arm answers — author them \
            together and ship them in "files" (one {"text": ...} per \
            line; add "options" + "target" per item for the answer-token \
            instrument).
            - A capability battery (did general ability survive?), \
            judges (≥2 for evidence grade) and a rubric criterion.
            Leave "variantConditions": [] — agents attach in the app.
            """
        case .multiAgent:
            return """
            ## This study is a MULTI-AGENT SCENARIO (studyType "multiAgent")

            Run a pinned scenario (panel of agents, turn structure, \
            visibility rules) and record the transcript. Ask the \
            researcher:
            - Which scenario file, and which saved agents does it cast? \
            The scenario is selected and pinned in the app's Conditions \
            pane; task prompts do not apply.
            - Whether to include the stripped-baseline transcript \
            ("multiAgentIncludeBaseline": true).
            - Transcript judging, if any (judges + rubric).
            """
        }
    }

    /// The full co-authoring prompt for the selected study type.
    public static func prompt(for intent: StudyIntent = .conceptStudy) -> String {
        """
        You are helping a researcher author a study for SteerLab, an \
        activation-steering workbench for open-weight language models. \
        Your job: interview the researcher, then produce ONE complete \
        JSON document — a STUDY PACK — they will paste into the app via \
        "Paste Study JSON". A pack is:

        ```json
        {
          "study": { …the experiment manifest… },
          "files": { "prompts/tasks/my-study.jsonl": "<file contents>" }
        }
        ```

        "files" lets the pack CARRY the study's data (task prompts, \
        concept stimuli, validation sets, rubrics) so one document \
        drives the study. Paths must be workspace-relative under \
        prompts/ (no "..", no absolute paths); the app writes them, \
        refuses to overwrite differing existing files, PINS what the \
        manifest names (task prompts, rubric, battery), and imports the \
        study as a DRAFT with verification running immediately — \
        missing pieces surface as named violations with remedies. A \
        bare manifest without the envelope is also accepted.

        \(typeSection(for: intent))

        ## Hard rules

        1. "name": lowercase letters, digits, hyphens ONLY (it becomes a \
        directory name). The app drops everything else.
        2. "status" must be "draft". Never emit freeze fields (frozenAt, \
        freezeHash, frozenBy, gitCommit) — the app strips them; \
        preregistration is earned through gates, never pasted.
        3. NEVER fabricate hashes. Omit every *Hash field. Hashes are \
        pins the app computes from the real files when the researcher \
        attaches them. A made-up hash = instant verify violation.
        4. Leave "concepts": [] and "variantConditions": [] unless the \
        researcher gives you exact existing artifacts — attaching \
        concepts and agents in the app is what pins them correctly.
        5. Researcher-notes fields (never sent to any model): \
        "experimentDescription", "taskDescription", "outcomeMeasures", \
        "phase" (screen|confirm|triangulate|panel — picks the analysis \
        correction). Behavior fields: "modelID" (a Hugging Face id), \
        "temperature" (0 for locally-measured runs), "maxTokens", \
        "samplesPerItem" + "seedPolicy" (stochastic designs — server/\
        cluster runs only), "systemPrompt", "promptMode" \
        ("chatAssistant" | "rawCompletion"), "numericParser" (the name of an \
        entry in prompts/parsers/parser-registry.json — this is how a study \
        declares the grammar its numeric answers are parsed with). Do NOT \
        reach for "caseFamily": it is a provenance label. The value \
        "sentencing" still implicitly selects a built-in duration parser for \
        manifests that already depend on it, and that implicit selection is \
        DEPRECATED — declare "numericParser" instead.
        6. Judges: "judges" is a list of {"name", "kind", "model", \
        "provider"}. kind "openrouter" REQUIRES both a model slug and a \
        pinned provider. Two or more judges for evidence-grade studies.
        7. A pipeline (one cluster job chaining stages, with gates) is \
        declared as {"stages": [...], "gates": {...}}. Stages in order \
        from: extract, validate, sweep, promote, run, evaluate, analyze. \
        Concept studies use the full funnel; comparisons chain run → \
        evaluate → analyze; evaluate/analyze require run in the chain. \
        Gates: {"validate": {"minScenarioAccuracy": 0.6, \
        "maxCrossConceptCosine": 0.8}, "sweep": \
        {"requireSelectionForEveryConcept": true}}.

        Ask about the research question, the model (exact Hugging Face \
        id; local runs are greedy temperature 0, cluster runs may be \
        stochastic), judging, and the pipeline + gates — then AUTHOR the \
        data files with the researcher rather than leaving placeholders.

        ## Output

        When you have enough, output the complete STUDY PACK in one \
        ```json fenced block — nothing else in that block. The "study" \
        skeleton (keep every key you do not change; set studyType/\
        studyKind for the type above):

        ```json
        {
          "study": {
            "name": "my-study",
            "status": "draft",
            "studyType": "\(intent.rawValue)",
            "studyKind": "\(intent.mappedKind.rawValue)",
            "experimentDescription": "what this study asks",
            "taskDescription": "what the model will do",
            "outcomeMeasures": "what will be measured",
            "modelID": "Qwen/Qwen3-4B-MLX-4bit",
            "createdAt": "1970-01-01T00:00:00Z",
            "promptMode": "chatAssistant",
            "qwenThinkingEnabled": false,
            "temperature": 0,
            "maxTokens": 512,
            "seeds": [0],
            "taskPromptsFile": "prompts/tasks/my-study.jsonl",
            "concepts": [],
            "conditions": [],
            "variantConditions": [],
            "multiAgentIncludeBaseline": true,
            "judges": []
          },
          "files": {
            "prompts/tasks/my-study.jsonl": "{\\"text\\": \\"first prompt\\"}\\n{\\"text\\": \\"second prompt\\"}\\n"
          }
        }
        ```

        After the block, tell the researcher: paste it via Paste Study \
        JSON in the app's Studies panel — the files land in the \
        workspace and named ones are pinned; then attach concepts/\
        agents there. The Issues box and Data & Prompts pane list \
        exactly what is still needed.
        """
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
        var manifest: ExperimentManifest
        var packFiles: [String: String] = [:]
        let raw = try? JSONSerialization.jsonObject(with: Data(json.utf8))
        let isPack = (raw as? [String: Any])?["study"] != nil
        do {
            if isPack {
                struct StudyPack: Decodable {
                    var study: ExperimentManifest
                    var files: [String: String]?
                }
                let pack = try JSONDecoder().decode(
                    StudyPack.self, from: Data(json.utf8))
                manifest = pack.study
                packFiles = pack.files ?? [:]
            } else {
                manifest = try JSONDecoder().decode(
                    ExperimentManifest.self, from: Data(json.utf8))
            }
        } catch {
            throw ExperimentError(
                reason: "study JSON did not decode: \(error). The document "
                    + "must be a complete experiment manifest, or a study "
                    + "pack {\"study\": {…}, \"files\": {…}} (start from "
                    + "Copy Study JSON on an existing draft)")
        }
        // The name becomes an experiments/<name>/ path component — apply
        // the SAME sanitization create() applies, so a pasted "../escape"
        // or "a/b" can never write outside the store (and the saved
        // manifest carries the sanitized name it actually lives under).
        let name = sanitizedExperimentName(manifest.name)
        guard !name.isEmpty else {
            throw ExperimentError(
                reason: "study JSON has no usable name (after dropping "
                    + "path-unsafe characters: lowercase letters, digits, "
                    + "and hyphens survive)")
        }
        manifest.name = name
        if (try? load(name: name)) != nil {
            throw ExperimentError(
                reason: "a study named '\(name)' already exists — change "
                    + "\"name\" in the JSON (freeze-and-iterate duplicates, "
                    + "never overwrites)")
        }
        // Draft-only import: freeze is one-way and GATED — pasted text
        // cannot mint a preregistered object.
        manifest.status = .draft
        manifest.frozenAt = nil
        manifest.freezeHash = nil
        manifest.frozenBy = nil
        manifest.gitCommit = nil
        manifest.freezeForced = nil
        manifest.forcedGatesSkipped = nil
        manifest.preregistrationHash = nil
        manifest.preregistrationGeneratedHash = nil
        // Pack files land BEFORE the manifest so pinning sees real bytes.
        // TRANSACTIONAL (engineer finding 2026-07-19): every path is
        // validated before the FIRST write, and a later failure —
        // including the manifest save — rolls the newly created files
        // back, so a refused import never leaves half a pack behind.
        let filesWritten = try writePackFiles(packFiles)
        do {
            autoPinNamedInputs(into: &manifest)
            try save(manifest, allowCreate: true)
        } catch {
            for path in filesWritten {
                try? FileManager.default.removeItem(at: resolveProjectPath(path))
            }
            throw error
        }
        return (manifest, verify(manifest), filesWritten)
    }

    /// Write a study pack's data files into the workspace, two-phase:
    /// VALIDATE everything (lexical containment, symlink-real containment,
    /// collision) before the first write, then write. Existing identical
    /// files are fine (idempotent re-import); existing DIFFERING files
    /// refuse — a pasted pack never silently rewrites data another study
    /// may pin. Containment is checked against RESOLVED paths (engineer
    /// finding 2026-07-19: lexical checks alone would follow a symlink
    /// planted under prompts/ out of the workspace).
    private static func writePackFiles(
        _ files: [String: String]
    ) throws -> [String] {
        guard !files.isEmpty else { return [] }
        let fm = FileManager.default
        let entries = files.sorted(by: { $0.key < $1.key })
        // A symlinked prompts/ root itself is SUPPORTED: containment is
        // judged between RESOLVED paths, so everything must land inside
        // wherever prompts/ really lives.
        let promptsRoot = resolveProjectPath("prompts")
            .resolvingSymlinksInPath().path
        var toWrite: [(path: String, url: URL, data: Data)] = []
        // Phase 1 — validate every entry BEFORE any filesystem effect
        // (fourth round: creating parent directories before the symlink
        // check could plant directories OUTSIDE the workspace and leave
        // them behind). Containment resolves the NEAREST EXISTING
        // ancestor: the not-yet-existing tail components are plain names
        // (no "..", checked above) and cannot be symlinks, so if the
        // existing ancestor resolves inside prompts/, the final parent
        // will too.
        for (path, content) in entries {
            guard !path.hasPrefix("/"), !path.contains(".."),
                path.hasPrefix("prompts/")
            else {
                throw ExperimentError(
                    reason: "study pack file '\(path)' refused — pack files "
                        + "must be workspace-relative paths under prompts/ "
                        + "(no absolute paths, no ..)")
            }
            let url = resolveProjectPath(path)
            // A destination that is itself a symlink is refused outright
            // (Data(contentsOf:)/write would follow it).
            if (try? fm.destinationOfSymbolicLink(atPath: url.path)) != nil {
                throw ExperimentError(
                    reason: "study pack file '\(path)' refused — the "
                        + "destination is a symlink")
            }
            var ancestor = url.deletingLastPathComponent()
            while !fm.fileExists(atPath: ancestor.path),
                ancestor.pathComponents.count > 1
            {
                ancestor = ancestor.deletingLastPathComponent()
            }
            let ancestorReal = ancestor.resolvingSymlinksInPath().path
            guard ancestorReal == promptsRoot
                || ancestorReal.hasPrefix(promptsRoot + "/")
                || promptsRoot.hasPrefix(ancestorReal + "/")
            else {
                throw ExperimentError(
                    reason: "study pack file '\(path)' refused — its "
                        + "directory resolves outside the workspace's "
                        + "prompts/ tree (symlink)")
            }
            let data = Data(content.utf8)
            if fm.fileExists(atPath: url.path) {
                guard (try? Data(contentsOf: url)) == data else {
                    throw ExperimentError(
                        reason: "study pack file '\(path)' differs from the "
                            + "existing file — packs never overwrite; rename "
                            + "the pack's path or reconcile by hand")
                }
                continue  // identical bytes: idempotent, nothing to write
            }
            toWrite.append((path, url, data))
        }
        // Phase 2 — create directories and write; roll created files back
        // on failure (directories created inside prompts/ may remain —
        // contained and empty, they are harmless).
        var written: [String] = []
        do {
            for (path, url, data) in toWrite {
                try fm.createDirectory(
                    at: url.deletingLastPathComponent(),
                    withIntermediateDirectories: true)
                try data.write(to: url, options: .atomic)
                written.append(path)
            }
        } catch {
            for path in written {
                try? fm.removeItem(at: resolveProjectPath(path))
            }
            throw error
        }
        return written
    }

    /// Pin inputs the manifest NAMES but has not pinned, from bytes now on
    /// disk — an LLM cannot compute hashes, so the app pins on arrival
    /// (task prompts, judge rubric, capability battery). Every pin goes
    /// through its validating helper: a file that is present but the wrong
    /// SHAPE stays UNPINNED, so the problem SURFACES — verify() reports the
    /// incomplete pin and readiness reports the file as invalid with the
    /// plain-language detail — instead of pinning garbage that would fail
    /// much later at run/analyze. A per-input no-op when the file is
    /// absent, for the same reason: verify() is the honest finding.
    private static func autoPinNamedInputs(into manifest: inout ExperimentManifest) {
        if let file = manifest.taskPromptsFile, manifest.taskPromptsHash == nil {
            _ = try? pinTaskPrompts(file, into: &manifest)
        }
        if let file = manifest.judgeRubricFile, manifest.judgeRubricHash == nil {
            _ = try? JudgeRubricStore.pin(file, into: &manifest)
        }
        if let file = manifest.capabilityBatteryFile,
            manifest.capabilityBatteryHash == nil
        {
            // The shared validating pin (Phase 0 item 2) — never a raw
            // hash of unparsed bytes.
            _ = try? pinCapabilityBattery(file, into: &manifest)
        }
    }
}
