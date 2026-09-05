import ExperimentKit
import SwiftUI

// The standard visible-help affordance (2026-07-19 usability program,
// Phase 1): a plain question-mark button that opens a popover carrying a
// REAL multi-sentence explanation. Hover `.help` tooltips remain the
// secondary layer — never the only one. The pattern originated in the
// Pipeline composer; this file is its shared home plus the explanation
// corpus for the Studies surfaces.

/// A visible ⓘ button. Owns its own presentation state so call sites need
/// no `@State` plumbing (important in `ExperimentsPanelView`, which sits at
/// the type-checker's limits).
struct InfoButton: View {
    let text: String
    @State private var isPresented = false

    var body: some View {
        Button {
            isPresented = true
        } label: {
            Image(systemName: "questionmark.circle")
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Explain")
        .popover(isPresented: $isPresented) {
            Text(text)
                .font(.caption)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: 380)
                .padding(12)
                .textSelection(.enabled)
        }
    }
}

/// A section header with the ⓘ affordance beside the title — for Form
/// sections whose whole purpose deserves a visible explanation.
struct InfoSectionHeader: View {
    let title: String
    let text: String

    var body: some View {
        HStack(spacing: 6) {
            Text(title)
            InfoButton(text: text)
        }
    }
}

// MARK: - Explanation corpus for the Studies surfaces

/// Plain-language explanations, one per jargon cluster the 2026-07-19
/// audit called out. Convention throughout: plain words first, the
/// technical term second, and no punitive language. These are UI copy —
/// cross-engine manifest keys and report vocabulary never change here.
enum StudyInfo {
    static let studyType = """
        The study type says what this study is trying to do, and the page \
        arranges itself around that answer: only the sections that type \
        needs are shown below. Switching types never deletes anything — \
        data a type's view hides is called out right under the picker, and \
        switching back shows it again. On drafts the choice is saved into \
        the study itself; on frozen studies it only changes what the page \
        displays.

        The buttons alongside let the whole study travel as one JSON \
        document: copy it out, work on it by hand or with an LLM (Copy LLM \
        Prompt teaches the format), and paste the result back as a NEW \
        draft. Pasted studies always arrive as drafts and are checked \
        immediately — pasting can never mint a preregistered study.
        """

    static let funnelPhase = """
        Studies in this program move through a funnel: screen (try many \
        concepts cheaply), confirm (re-test the survivors), triangulate \
        (cross-check with a different instrument), and panel (multi-agent).

        Declaring where this study sits has two real effects. First, the \
        analysis picks its multiple-comparison correction from it — \
        screening studies get the BH-FDR correction, confirmation studies \
        the stricter Holm correction. Second, a 'confirm' study must use \
        prompts the screening study never saw (held-out prompts), and that \
        separation is checked — a confirmation cannot quietly re-use the \
        prompts that selected the effect.

        The phase changes no prompt text and is never sent to any model.
        """

    static let sampling = """
        How many responses the run generates per (condition, prompt) pair, \
        and how randomness is seeded.

        1 sample means deterministic 'greedy' decoding — the model always \
        takes its most likely token, so the same prompt gives the same \
        answer. More than 1 sample is a stochastic design: it needs \
        temperature > 0 and runs on the Python server, which seeds every \
        record individually so any single response can be reproduced \
        exactly. The local Mac engine has no per-run sampling seed, so \
        local measured runs stay greedy.

        Seed policy: the fixed seed list (engine term 'manifestSeeds', the \
        default) reuses the study's declared seeds for every record; \
        derived per record ('derivedSHA256') derives each record's seed \
        from (condition, prompt, sample index), so samples differ but each \
        one is individually reproducible — required for samples per \
        item > 1.

        The study's sampling policy governs the baseline AND every saved \
        agent in the design — an agent's Playground temperature is \
        provenance only, never used in measured runs.
        """

    static let seedPolicy = """
        How each generated response gets its random seed on stochastic \
        runs. Only the Python server reads this — the local Mac engine \
        decodes greedily (temperature 0) and never uses a sampling seed.

        Derived per record (engine term 'derivedSHA256'): every \
        (condition, prompt, sample) derives its own seed from those \
        coordinates, so samples differ from each other but any single \
        response can be reproduced exactly. The right choice whenever \
        samples per item > 1 or temperature > 0 — and required for \
        samples per item > 1.

        Fixed seed list ('manifestSeeds' — the default when nothing is \
        declared): every record reuses the study's fixed seed list. \
        Leaving the policy undeclared and declaring 'manifestSeeds' \
        explicitly behave identically on both engines — the default IS \
        the fixed list.
        """

    static let promptMode = """
        How the baseline condition presents prompts to the model. 'Chat \
        assistant' wraps each task prompt in the model's chat template \
        with user/assistant roles — the way the model was trained to \
        converse. 'Raw completion' sends the literal text with no template \
        and the model simply continues it.

        Saved agents carry their own prompt mode, so this setting shapes \
        only the baseline arm. It changes the exact prompt bytes that are \
        hashed and pinned, so it is a real part of the study design, not a \
        display preference.
        """

    static let conditionsArms = """
        The arms of the study — what is actually compared. An agent \
        comparison runs the baseline model against saved agents (steered \
        or fine-tuned variants). A multi-agent study runs a scenario. A \
        concept study's confirm phase declares a perturbation policy \
        around a promoted agent's operating point and expands it into \
        ordinary conditions.

        Every arm becomes a hashed condition in the manifest, so the \
        comparison is pinned before anything runs — that is the \
        preregistration working, not extra ceremony.
        """

    static let conceptVectors = """
        A concept enters the study as a recipe, not as bytes: attaching \
        pins the concept's stimulus files by cryptographic hash together \
        with the extraction options (method, reading position). Runs \
        re-derive the steering vector deterministically from that recipe, \
        so the same study produces the same vector on any machine.

        Before experimental use a vector must earn its name: validation \
        checks that it moves held-out probes for its own concept, and \
        that distinct concepts do not collapse into one direction \
        (cross-concept similarity is reported). Freeze requires that \
        evidence for evidence-grade studies.
        """

    static let injectionConditions = """
        Each condition steers generation by adding a concept's direction \
        into the model's residual stream — its internal working memory — \
        at a chosen layer and strength, on every generated token.

        A defensible study pairs each treatment with controls: a \
        direction control (same vector, negative strength — the effect \
        should reverse) and a random-direction control at the same \
        strength (technically 'matched-norm random' — if a random \
        direction moves outputs just as much, the 'effect' was mere \
        perturbation energy, not the concept). A no-steering baseline is \
        always included at run time.
        """

    static let strengthLayerNorm = """
        Layer: which of the model's transformer layers the steering is \
        applied at. The middle third of the network is the usual sweet \
        spot; sweeps find the best layer empirically.

        Strength (α): how hard to push along the concept's direction. \
        Negative values push the opposite way — that is a legal condition \
        (a direction control), not an error.

        Relative strength ('norm units') measures α relative to the \
        model's own activity at that layer — the typical size of its \
        residual-stream activations on a pinned neutral corpus. With it \
        on, α 0.5 means "half as strong as what the model is already \
        doing there", which is comparable across concepts and layers; \
        with it off, α is in raw activation units.
        """

    static let controls = """
        A counterpart is a control condition that makes a treatment \
        interpretable: without it, an "effect" could be mere perturbation \
        or a one-off. 'Add Baseline' adds an explicit no-steering arm \
        (added automatically at run time when absent). '+ sign control' \
        on a condition repeats it with the strength negated (−α) — a real \
        directional effect should reverse. '+ random control' keeps the \
        same layers and strength but steers along a deterministic random \
        direction (technically 'matched-norm random') — if it moves \
        outputs as much as the concept does, the effect was perturbation \
        energy, not meaning.

        'Scaffold Control Matrix' does this for the whole study in one \
        click: for EVERY treatment condition it adds the missing \
        counterparts — the direction-flip control at −α and the \
        random-direction control at the same strength — plus one explicit \
        no-steering baseline if none exists. It is idempotent: run it \
        again and existing conditions are never touched or duplicated \
        (matching is by condition name).

        It also says what it CANNOT scaffold, because those need a \
        researcher: a positive-control concept (e.g. sympathy) must be \
        authored as its own concept — stimuli written, attached, and \
        added as a condition; dose-response needs you to add further \
        strengths per treatment; and the capability battery must be \
        pinned so every condition is battery-scored during the run.
        """

    static let dataPrompts = """
        Everything this study reads from disk, in one pane, grouped by \
        what each file is for. Each row tracks one file the manifest \
        needs or pins: green = present, amber = partially usable, red = \
        missing or in a shape the run would refuse.

        Files are pinned by hash, so editing a pinned file later shows up \
        as drift instead of silently changing the study — re-pin after \
        deliberate edits. 'Create from template' scaffolds a starting \
        file with example rows; the row's buttons view it, reveal it in \
        Finder, or open it in your editor.
        """

    static let evaluationOutcome = """
        This choice dispatches real measurement machinery — it is not a \
        pre-declaration note. It decides what the run generates and what \
        lands in its records.

        'Generated choice': the model writes text for every item, and the \
        answer is parsed out of the prose — the run's records are sampled \
        text plus parsed endpoints.

        'Answer-token probability': no prose for the endpoint at all — \
        the run scores how likely each declared answer option is, \
        directly from the model's token probabilities. Deterministic, \
        temperature-free, immune to prose confounds; the preferred \
        instrument for categorical outcomes. It needs per-item 'options' \
        in the prompt file, and thinking mode off.

        'Both' records the two side by side: sampled-text records AND \
        answer-token records for every item.

        Results views differ accordingly — and paired judging reads only \
        sampled text, so a logprob-only run has nothing for judges to \
        score. The declaration is written into the study (the \
        'outcomeInstruments' key), chosen before results exist, never \
        inferred from the data afterwards.
        """

    static let judges = """
        Judging is optional. A study with no judges measures its declared \
        endpoints alone (parsed answers, answer-token probabilities) — a \
        complete, valid design; the shakedown guide itself recommends it \
        for first runs.

        When declared, judging is paired and blinded: a judge sees a \
        baseline output and a condition output for the same item, in \
        randomized order, without knowing which is which, and scores the \
        pair against the pinned rubric. Judges are PINNED identities — \
        name, kind, and model (plus serving provider for OpenRouter) are \
        hashed into the study — and a frozen judged study needs at least \
        2 so the report can carry agreement statistics; one judge's \
        quirks are not evidence.

        Three judge kinds. 'claude' and 'openrouter' are API calls and \
        need keys (rows say so when the key is missing). 'local' runs on \
        this substrate: an EMPTY local-judge model means the study's own \
        model judges, generating through the already-loaded weights; a \
        DIFFERENT local model needs an engine that can hold two models \
        at once — the server with STEERLAB_MAX_LOADED_MODELS ≥ 2 — and \
        refuses at start otherwise (local Mac sweeps always hold one \
        model, so they always refuse a second).

        With no judges pinned, the ad-hoc judge model below is used only \
        by Run Paired Judge in Results — one unpinned judge for \
        exploration, never freeze-grade evidence.
        """

    static let caseFamily = """
        Which family of cases this study's prompts belong to. A \
        PROVENANCE LABEL: free text, saved into the study, printed in the \
        report and the preregistration, and behaviorless. It changes no \
        prompt and is never sent to any model.

        Declare a 'Numeric parser' (below) to say how this study's \
        numeric answers are read. That is the mechanism: the parser \
        registry, as workspace data, pinned by hash at freeze.

        One DEPRECATED exception survives for studies that already \
        depend on it: with no numeric parser declared, the exact label \
        'sentencing' still makes both engines parse a duration-in-months \
        value out of every sampled record (stamped 'parsedMonths'). \
        Those studies keep running and keep producing the same numbers — \
        the run now says out loud that its endpoint was chosen by \
        inference rather than declaration. New studies should declare the \
        registry's 'sentencing-months' entry, which reproduces that \
        parser exactly. A declared parser always wins over the label.

        A/B choice parsing is separate from the family either way — it \
        reads from each item's declared 'options'.
        """

    static let numericParser = """
        How a numeric answer is read out of each response — declared as \
        data, not code. The workspace keeps a small registry file \
        (prompts/parsers/parser-registry.json) of named parsers: a \
        'duration, read in months' parser declares its unit words (years: \
        12, months: 1), how compound answers like '8 years 3 months' \
        combine, and what a range like '18 to 24 months' means; a 'plain \
        number' parser declares its range and percent policies.

        Picking one writes its name into the study (the manifest key \
        'numericParser') and pins the registry file by cryptographic hash, \
        so the exact reading rule is part of the preregistered study — \
        editing the registry afterwards shows up as drift, never as a \
        silent change of measurement. With 'None', the built-in behavior \
        applies: only the case family 'sentencing' parses a numeric \
        endpoint (parsedMonths).

        An answer no rule can read is counted as a parse failure — it is \
        never coerced to 0 or a partial number.
        """

    static let exclusionRules = """
        Which answers the analysis may set aside, declared up front as \
        part of the study — never decided after seeing the results. Three \
        rules exist: drop answers that failed their item's attention check \
        (items declare the expected answer in the prompt file), drop \
        answers where no numeric value could be read, and drop answers \
        whose value falls outside a declared range.

        Exclusions apply at analysis time only: the raw responses on disk \
        are never touched, and the report states exactly which rules were \
        active and how many answers each one dropped, per condition — \
        zeros included. Because the statistics compare each answer with \
        its same-item baseline, an excluded answer also drops from that \
        comparison, and an excluded baseline answer drops its item from \
        every comparison (pairwise deletion).

        Declaring the attention-check rule while the prompt file has no \
        checked items makes the run refuse at start — a declared rule \
        that can never fire is a data bug, not a no-op.
        """

    static let runExclusions = """
        This run's study declared exclusion rules, so the analysis says \
        exactly what they did: which rules were active, how many answers \
        each rule dropped in each condition (zeros included — a rule that \
        never fired says so), how many answers survived into the paired \
        statistics, and the declared scope.

        Scope, spelled out: at analysis time the rules consider all record \
        types. A failed attention check drops the whole condition-item \
        cell — including its deterministic instrument readouts (answer-token \
        logprob, ordinal position) — once every sampled answer of the cell \
        fails its check. Range and parse rules only ever read a value the \
        record itself carries, never a proxy from another record. Before \
        paired judging, only sampled answers are considered, because only \
        they are judged.

        The raw responses in generations.jsonl are untouched — exclusions \
        only remove records from the statistics. Because those statistics \
        are paired, dropping an answer also drops its comparison with the \
        same item's baseline, and a dropped baseline answer removes that \
        item from every condition's comparison (pairwise deletion).
        """

    static let rubricFile = """
        One rubric, one file. The judge rubric is a file in the \
        workspace, pinned by hash at Save Evaluation Settings — the \
        exact criteria the judges read are part of the preregistered \
        study, and later edits surface as drift until re-pinned. The \
        menu lists prompts/rubrics/; the folder button picks any file \
        inside the workspace. The eye views it, the pencil edits it in \
        the app.

        Freezing a judge-evaluated study requires a pinned rubric file — \
        unpinned text leaves no drift-checkable record of what the \
        judges were asked.
        """

    static let rubricDraft = """
        The rubric FILE (chosen above, pinned by hash) is the \
        instrument. This scratchpad appears only while no file is \
        chosen: it lets you draft wording, and Run Paired Judge in \
        Results can use it for an exploratory judging pass.

        'Save scratchpad as rubric file' writes the text to the study's \
        standard rubric path so it can be pinned — it never overwrites a \
        file with different contents. Once a file is chosen, the file \
        always wins at evaluation time and this scratchpad is put away.
        """

    static let structuredJudgeFields = """
        Optional. Besides picking a winner, each judge can ALSO return \
        machine-readable fields you name here — for example \
        'holding_changed boolean; severity_delta number; \
        main_difference short string'. The judge fills them per pair \
        (null when undeterminable), and the report summarizes them \
        across all paired judgments (counts for booleans, means for \
        numbers, tallies for categories).

        The app builds the output instructions itself: every judge is \
        told to answer in JSON (winner, confidence, brief reason), and \
        fields named here are requested additionally under \
        'structured_fields'. Your rubric only needs to state the \
        judging criteria — never the output format.

        Leave it empty to judge on the rubric alone.
        """

    static let judgeKindsAndKeys = """
        Three judge kinds, and where each one's API key lives:

        'claude' calls the Claude API with your personal Anthropic key \
        (Compute section; stored in the macOS Keychain, or the \
        ANTHROPIC_API_KEY environment variable). That key never leaves \
        this Mac — pushing it to a cluster is deliberately unsupported.

        'openrouter' pins a model + serving provider on OpenRouter. It \
        uses OPENROUTER_API_KEY if set, else the dedicated 'external \
        judge key' from the Compute section (also Keychain-stored) — a \
        separate, ideally spend-capped key. That dedicated key is the \
        only key that ever travels: it is copied to the cluster as \
        ~/.steerlab/judge-key (permissions 600) at every connect so \
        cluster runs can judge inline, and it is removed there when you \
        clear it. Keys are never written into study bundles or job \
        environments — only the file's path travels.

        'local' is a model running on this substrate (blank = the study \
        model judges its own outputs). No key at all.
        """

    static let taskPromptsSavePin = """
        Save & Pin writes this editor's text back to the file named \
        above — the file on disk is replaced by what you see here (one \
        prompt per block, blocks separated by a line containing only \
        ---; per-item instrument fields like options/target are \
        preserved byte-faithfully) — and then re-pins the file's new \
        SHA-256 into the study. Saving IS a file write plus a re-pin, \
        not an app-internal save.
        """

    static let importRecordStructure = """
        Required record structure — one JSON object per line:

        {"text": "…the prompt…", "id": "item-1", \
        "options": ["A", "B"], "target": "A"}

        "text" is required (the key "prompt" is accepted as a synonym). \
        Optional per record: "id" (stable item id — must be unique), \
        "options" (array of allowed answers, required by the \
        answer-token instrument), and "target" (the option of \
        interest). Unknown keys are preserved verbatim. A line that is \
        not a JSON object, or lacks "text"/"prompt", refuses the whole \
        import with its line number.
        """

    /// Import-destination semantics, stated as the code behaves: the sheet
    /// import writes the study's standard destination file under the
    /// study-pack write rule (identical bytes idempotent; differing bytes
    /// refuse unless the Replace checkbox opts in), points the manifest at
    /// it, and pins the new hash.
    static func importDestination(_ destination: String) -> String {
        """
        On Import & Pin the records are written to \(destination) — this \
        study's standard task-prompts file. The study's manifest then \
        points at that file and pins its SHA-256. If that file already \
        exists with identical contents the import just re-pins; if its \
        contents DIFFER the import refuses — imports never overwrite \
        silently — unless "Replace the existing file" is checked, which \
        replaces the previous records deliberately (draft studies only — \
        a frozen study refuses the import entirely).
        """
    }

    static let inlineFileEditor = """
        Saving writes the file in place on disk (atomically). If a study \
        pins this file by hash, the pin still records the OLD bytes — \
        the edit surfaces as drift ('problem found' on frozen studies, a \
        readiness finding on drafts) until the file is re-pinned where \
        it was pinned (e.g. Save & Pin for task prompts, Save Evaluation \
        Settings for rubrics, the Pin button for baselines). Nothing is \
        lost either way: drift is loud, never silent.
        """

    static let optionLengths = """
        A subtlety of the answer-token instrument: answer options that \
        tokenize to different lengths are not directly comparable — joint \
        probability naturally favors shorter options, which can \
        masquerade as an effect.

        The run refuses unequal option sets unless this box acknowledges \
        the issue deliberately. Prefer equal-length option labels (single \
        letters, matched phrasing) where the design allows it.
        """

    static let issues = """
        The one-glance answer to "is this study sound, and if not, why".

        Problems found (red) are blocking: a pinned file that no longer \
        matches its recorded hash (the engine calls these 'verify \
        violations'), or a data file the run or freeze will refuse \
        without. Notes (orange; 'advisories' in reports) are non-blocking: \
        they affect what grade of evidence the study can claim, not \
        whether it can run.

        Nothing here is punitive — the list exists so drift and gaps are \
        visible the moment they appear, not at a failing gate later.
        """

    /// Per-group explanations for the Data & Prompts pane.
    static func dataCategory(_ category: DataCategory) -> String {
        switch category {
        case .taskPrompts:
            """
            The prompts every arm answers — the study's actual task \
            items, one JSONL record per line. Records may carry per-item \
            instrument fields (answer options, target) that the \
            answer-token instrument needs; the editors here preserve \
            those fields byte-faithfully. The file is pinned by hash, so \
            the exact prompt set is part of the preregistered study.
            """
        case .conceptData:
            """
            What concept vectors are built from and checked against: the \
            stimulus sets extraction reads, the concept's validation \
            scenarios (held-out probes a real vector must move), style \
            markers, and the neutral corpus that fixes what strength 1.0 \
            means. All pinned by hash — the study pins the recipe, never \
            vector bytes.
            """
        case .judging:
            """
            What the Evaluation pane reads: the judge rubric (a \
            git-versioned file, pinned by hash) and the judge panel. \
            Judges score blinded baseline-vs-condition pairs against the \
            rubric; freezing a judged study requires a pinned rubric \
            file and at least one judge — a single-coder design is legal \
            and freezes with an advisory, because no inter-rater \
            agreement will exist for its codings.
            """
        case .battery:
            """
            A fixed set of capability probes run through every arm, \
            checking that steering has not simply broken the model — an \
            arm that wins the task by losing coherence is not a finding. \
            The battery runs per condition inside the study run and is \
            reported alongside the outcomes.
            """
        case .comparison:
            """
            Human-baseline measurements for the same manipulation, \
            transcribed from published studies into a CSV. Pinning one \
            enables the headline comparison R = delta_model − \
            delta_human; without it, results stay model-internal (which \
            is a legitimate, smaller claim).
            """
        case .scenario:
            """
            The saved multi-agent scenario this study runs — the panel \
            composition, turn order, and what each agent can see. \
            Authored in the Multi-Agent panel; the study pins and runs \
            it as a condition.
            """
        case .style:
            """
            The output taxonomy for reasoning-style measurement — how \
            responses are classified beyond their outcomes (e.g. \
            formalist vs consequentialist argument moves). The third leg \
            alongside a study's categorical outcomes and its numeric \
            magnitudes.
            """
        case .readout:
            """
            What a run reads out of the residual stream WHILE it generates \
            — the J-lens readout block: which lens, which layers, which \
            tokens are watched or ranked. Authored on the SERVER \
            (server-only by rule; any model with a published lens); this app renders and \
            carries it, and never produces one. Every field is pinned, \
            because a readout that cannot be reproduced cannot be cited — \
            and freeze additionally requires a PASSING qualification for \
            the study's exact model, revision, dtype and quantization, \
            since geometry cannot see numerics.
            """
        }
    }
}
