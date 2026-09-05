import Foundation
import SteeringKit

/// One item of study data an experiment manifest implies it needs — with
/// where the file goes, whether it is there, and (where one exists) the
/// template that scaffolds it.
///
/// Domain-neutral by construction (hard requirement): every requirement is
/// derived ONLY from what the manifest declares — attached concepts, pinned
/// files, declared instruments, judge protocol, study kind. No study-domain
/// assumption lives here; concepts, templates, and rubrics are data.
public struct DataRequirement: Identifiable, Sendable, Equatable {
    public enum Kind: String, Sendable, Codable {
        case conceptStimuli
        case conceptValidation
        case conceptMarkers
        case taskPrompts
        case judgeRubric
        case judgePanel
        case humanBaseline
        case multiAgentScenario
        case capabilityBattery
        case neutralCorpus
        case reasoningStyleTaxonomy
        case numericParser
        case exclusionRules
        case jlensReadout
    }

    public enum Status: String, Sendable, Codable {
        /// The file exists (and parses, where a parser gates use).
        case present
        /// The file exists but is incomplete for what the manifest declares
        /// (e.g. task-prompt items lacking `options` under a logprob
        /// instrument, a judge panel of 1). A partial study still RUNS —
        /// it only degrades what the study can claim.
        case partial
        /// The file exists but the run REFUSES to load it (duplicate item
        /// ids, transcript schema/family violations, malformed JSONL) — a
        /// blocker exactly like `.missing`, so preflight matches execution
        /// instead of promising a degraded run that would never start.
        case invalid
        /// Required by the declared protocol and absent — a blocker.
        case missing
        /// Not required by this manifest's declarations; listed so the
        /// researcher sees what claims it would unlock.
        case optional
    }

    public var id: String
    public var title: String
    public var kind: Kind
    public var status: Status
    /// Where the file goes — workspace-relative unless the manifest pinned
    /// an absolute path.
    public var path: String
    /// One human sentence: what is wrong / what this is for.
    public var detail: String
    /// `DataTemplates` id when one-click scaffolding exists for this kind.
    public var templateID: String?

    public init(
        id: String, title: String, kind: Kind, status: Status,
        path: String, detail: String, templateID: String? = nil
    ) {
        self.id = id
        self.title = title
        self.kind = kind
        self.status = status
        self.path = path
        self.detail = detail
        self.templateID = templateID
    }
}

/// Aggregate counts over a requirement list. `blockers` are the `.missing`
/// AND `.invalid` items — `.partial` items run but degrade what the study
/// can claim, while an `.invalid` file makes the run refuse to load.
public struct ReadinessSummary: Sendable, Equatable {
    public var presentCount: Int
    public var partialCount: Int
    public var invalidCount: Int
    public var missingCount: Int
    public var optionalCount: Int
    public var blockers: [DataRequirement]

    public var isReady: Bool { blockers.isEmpty }

    public var line: String {
        "\(presentCount) present · \(partialCount) partial · "
            + (invalidCount > 0 ? "\(invalidCount) invalid · " : "")
            + "\(missingCount) missing · \(optionalCount) optional"
    }
}

/// Registry of one-click data templates: repo seed data under
/// `prompts/templates/<id>/` (each with a sibling README.md documenting the
/// schema and destination), plus the workspace-relative destination rules.
/// Templates are deliberately domain-neutral example data — never study
/// content.
public enum DataTemplates {
    public struct Template: Sendable, Equatable {
        public let id: String
        /// Seed file, relative to a prompts-bearing root (workspace first,
        /// then the code checkout's seed data).
        public let seedRelativePath: String
        public let summary: String
    }

    public static let validation = Template(
        id: "validation",
        seedRelativePath: "prompts/templates/validation/validation-template.jsonl",
        summary: "never-named validation scenarios ({text, expresses})")

    public static let markers = Template(
        id: "markers",
        seedRelativePath: "prompts/templates/markers/markers-template.json",
        summary: "surface-marker word list for marker-density metrics")

    public static let humanBaseline = Template(
        id: "human-baseline",
        seedRelativePath:
            "prompts/templates/human-baseline/human-baseline-template.csv",
        summary: "transcribed human-effect table for human-anchored (R) claims")

    public static let scenario = Template(
        id: "scenario",
        seedRelativePath: "prompts/templates/scenario/scenario-template.json",
        summary: "minimal two-agent multi-agent scenario")

    public static let taskPromptsChoice = Template(
        id: "task-prompts-choice",
        seedRelativePath:
            "prompts/templates/task-prompts-choice/task-prompts-choice-template.jsonl",
        summary: "task prompts with options/target for answer-logprob instruments")

    public static let taskPromptsTranscript = Template(
        id: "task-prompts-transcript",
        seedRelativePath:
            "prompts/templates/task-prompts-transcript/task-prompts-transcript-template.jsonl",
        summary: "task prompts with scripted transcripts (seeded assistant "
            + "turns) for metacognition studies")

    /// Two example taxonomies live in the template directory (word-list and
    /// regex/structural); the scaffold copies the word-list one — both are
    /// STARTING POINTS the researcher must adapt and hash-pin.
    public static let reasoningStyle = Template(
        id: "reasoning-style",
        seedRelativePath:
            "prompts/templates/reasoning-style/reasoning-style-generic-template.json",
        summary: "reasoning-style taxonomy (rs_<feature> metrics + effect sizes)")

    /// Starting-point paired-judging rubric (domain-neutral criteria the
    /// judge scores a blinded response pair against) — so a workspace with
    /// no `prompts/rubrics/` files can start from a default and modify.
    public static let judgeRubric = Template(
        id: "judge-rubric",
        seedRelativePath: "prompts/templates/rubrics/rubric-template.md",
        summary: "domain-neutral paired-judging rubric (criteria to adapt and pin)")

    public static let all: [Template] = [
        validation, markers, humanBaseline, scenario, taskPromptsChoice,
        taskPromptsTranscript, reasoningStyle, judgeRubric,
    ]

    public static func template(id: String) -> Template? {
        all.first { $0.id == id }
    }

    /// Seed file for a template: the workspace's own copy when the workspace
    /// was seeded with templates, else the shipped seed data
    /// (`CodeResources.workspaceSeed()`).
    public static func seedURL(for template: Template, workspaceRoot: URL) -> URL {
        workspaceOrSeedURL(
            relativePath: template.seedRelativePath, workspaceRoot: workspaceRoot)
    }

    /// The one seed-resolution rule, shared by template scaffolding and the
    /// Concept Lab's prompt templates: the workspace's own copy wins when it
    /// exists, else the code-shipped seed copy (nil resolves through
    /// `CodeResources.workspaceSeed()` — `<checkout>/WorkspaceSeed/` in
    /// developer mode, the bundled seed tree in a packaged build; every
    /// template path below is a `seedManifest` entry, so both resolutions
    /// hit real bytes). `seedRoot` is parameterized
    /// so tests can simulate a packaged/relocated app whose code checkout
    /// (and therefore its seed data) is absent. When NO seed source exists,
    /// the (absent) workspace path is returned so the caller's readability
    /// check routes the failure — never a fabricated path.
    public static func workspaceOrSeedURL(
        relativePath: String,
        workspaceRoot: URL,
        seedRoot: URL? = nil
    ) -> URL {
        let workspaceCopy = workspaceRoot.appending(path: relativePath)
        if FileManager.default.fileExists(atPath: workspaceCopy.path) {
            return workspaceCopy
        }
        guard let resolvedSeed = seedRoot ?? (try? CodeResources.workspaceSeed())
        else { return workspaceCopy }
        return resolvedSeed.appending(path: relativePath)
    }

    // MARK: - Destination rules (workspace-relative)

    /// Validation sets live where the validate gate reads them — the rule is
    /// `ExperimentStore.conceptValidationRelativePath` (paired vs grand-mean
    /// concepts resolve to different trees).
    public static func validationDestination(concept: String, isPaired: Bool) -> String {
        ExperimentStore.conceptValidationRelativePath(name: concept, isPaired: isPaired)
    }

    /// Markers live where scoring reads them for EVERY concept, grand-mean
    /// included (`ExperimentStore.markersRelativePath`).
    public static func markersDestination(concept: String) -> String {
        ExperimentStore.markersRelativePath(concept: concept)
    }

    public static func humanBaselineDestination(experiment: String) -> String {
        "prompts/baselines/\(experiment)-human-baseline.csv"
    }

    /// Scenario files live where `MultiAgentScenarioStore.scan` discovers
    /// them: `prompts/panels/<slug>.json` (a git-versioned
    /// subtree, like model-variants).
    public static func scenarioDestination(experiment: String) -> String {
        "prompts/panels/\(experiment)-scenario.json"
    }

    /// Task prompt sets are authored where the Data inventory enumerates them
    /// (`VectorCatalog.taskPromptsRelativeDirectory`) — one constant, so the
    /// place this tells a researcher to write and the place the workbench
    /// looks cannot drift apart.
    public static func taskPromptsDestination(experiment: String) -> String {
        "\(VectorCatalog.taskPromptsRelativeDirectory)/\(experiment)-prompts.jsonl"
    }

    /// Rubrics live where `JudgeRubricStore` scans and pins them
    /// (`prompts/rubrics/`, workspace-relative).
    public static func judgeRubricDestination(experiment: String) -> String {
        "\(JudgeRubricStore.relativeDirectory)/\(experiment)-rubric.md"
    }

    /// Taxonomies live where `set-style-taxonomy` expects them
    /// (`prompts/taxonomies/`, workspace-relative).
    public static func reasoningStyleDestination(experiment: String) -> String {
        "\(ExperimentStore.taxonomiesRelativeDirectory())/\(experiment)-reasoning-style.json"
    }
}

/// Derives, from an experiment manifest alone, the checklist of study data
/// the workspace still needs — so gaps surface in the panel and the CLI
/// instead of only when a gate fails. Pure functions over (manifest,
/// workspace root); no globals, unit-testable against temp roots.
public enum StudyDataReadiness {

    // MARK: - Derivation

    public static func requirements(
        for manifest: ExperimentManifest, workspaceRoot: URL,
        workspaceIsServer: Bool = false
    ) -> [DataRequirement] {
        var rows: [DataRequirement] = []
        let fm = FileManager.default

        func resolve(_ path: String) -> URL {
            path.hasPrefix("/")
                ? URL(filePath: path)
                : workspaceRoot.appending(path: path)
        }
        func exists(_ path: String) -> Bool {
            fm.fileExists(atPath: resolve(path).path)
        }

        // Per-concept data: stimuli for the pinned extraction method,
        // never-named validation set, optional markers. Only when the
        // concept machinery is OPERATIVE for this study type (engineer
        // finding 2026-07-19): carried-but-inert concepts must not read
        // as blockers in the Issues box — they get one honest optional
        // row instead.
        let machinery = ExperimentStore.conceptMachineryOperative(manifest)
        if !machinery, !manifest.concepts.isEmpty {
            rows.append(
                DataRequirement(
                    id: "concepts:carried",
                    title: "carried concept data (inactive)",
                    kind: .conceptStimuli,
                    status: .optional,
                    path: "prompts/concepts/",
                    detail: "\(manifest.concepts.count) concept(s) carried "
                        + "from another study type — preserved but not used, "
                        + "verified, or packaged by this study; switch the "
                        + "study type to work with them"))
        }
        for ref in machinery ? manifest.concepts : [] {
            let isPaired = ref.options.method.isPaired
            let name = ref.name

            if isPaired {
                let directory = "prompts/concepts/\(name)"
                let stimuli = try? StimulusSet(directory: resolve(directory))
                if let stimuli {
                    // Content check, not just presence (2026-07-31): a live
                    // workspace shipped 16 derived concept dirs whose
                    // negative class was the reference corpus's NAME split
                    // into one-character rows — a string-for-list bug in a
                    // derivation script. Presence checks passed; extraction
                    // under a pooled reading refused at run time; last-token
                    // reading would have produced silently garbage class
                    // means. Degenerate rows are a BLOCKER here, before any
                    // model loads.
                    let problem = Self.stimulusContentProblem(
                        positive: stimuli.positive, negative: stimuli.negative,
                        readingPosition: ref.options.readingPosition)
                    rows.append(
                        DataRequirement(
                            id: "concept:\(name):stimuli",
                            title: "stimuli — \(name)",
                            kind: .conceptStimuli,
                            status: problem == nil ? .present : .invalid,
                            path: directory + "/positive.jsonl (+ negative.jsonl)",
                            detail: problem
                                ?? "paired stimulus set loads "
                                + "(\(stimuli.positive.count) + "
                                + "\(stimuli.negative.count) rows); the pinned "
                                + "recipe re-derives the vector from it"))
                } else {
                    rows.append(
                        DataRequirement(
                            id: "concept:\(name):stimuli",
                            title: "stimuli — \(name)",
                            kind: .conceptStimuli,
                            status: .missing,
                            path: directory + "/positive.jsonl (+ negative.jsonl)",
                            detail: "paired extraction needs positive.jsonl and "
                                + "negative.jsonl in \(directory)/ — extraction "
                                + "cannot run without them"))
                }
            } else {
                let methodLabel = ref.options.method == .designatedReference
                    ? "designated-reference" : "grand-mean"
                let stories = "prompts/emotions/\(name)/stories.jsonl"
                let loaded = try? StimulusSet.loadMultiConceptTexts(url: resolve(stories))
                let problem = loaded.flatMap {
                    Self.stimulusContentProblem(
                        positive: $0.rows.map(\.text), negative: [],
                        readingPosition: ref.options.readingPosition)
                }
                rows.append(
                    DataRequirement(
                        id: "concept:\(name):stimuli",
                        title: "stories — \(name)",
                        kind: .conceptStimuli,
                        status: loaded == nil ? .missing
                            : (problem == nil ? .present : .invalid),
                        path: stories,
                        detail: problem
                            ?? (loaded != nil
                                ? "story corpus loads (\(loaded?.rows.count ?? 0) "
                                    + "rows); \(methodLabel) extraction re-derives "
                                    + "the vector from it"
                                : "\(methodLabel) extraction needs \(stories) — "
                                    + "extraction cannot run without it")))
                // The designated REFERENCE is recipe data too (external
                // review 2026-07-31, finding 5): a green checklist must mean
                // the reference corpus exists and holds real prose, or the
                // failure surfaces only after a cluster queue + model load.
                if ref.options.method == .designatedReference {
                    if let pin = ref.designatedReference {
                        let refStories = "prompts/emotions/\(pin.name)/stories.jsonl"
                        let refLoaded = try? StimulusSet.loadMultiConceptTexts(
                            url: resolve(refStories))
                        let refProblem = refLoaded.flatMap {
                            Self.stimulusContentProblem(
                                positive: $0.rows.map(\.text), negative: [],
                                readingPosition: ref.options.readingPosition)
                        }
                        rows.append(
                            DataRequirement(
                                id: "concept:\(name):reference",
                                title: "reference — \(pin.name) (for \(name))",
                                kind: .conceptStimuli,
                                status: refLoaded == nil ? .missing
                                    : (refProblem == nil ? .present : .invalid),
                                path: refStories,
                                detail: refProblem
                                    ?? (refLoaded != nil
                                        ? "designated reference loads "
                                            + "(\(refLoaded?.rows.count ?? 0) rows) — "
                                            + "the vector subtracts its mean"
                                        : "the pinned designated reference needs "
                                            + "\(refStories) — extraction cannot "
                                            + "run without it")))
                    } else {
                        rows.append(
                            DataRequirement(
                                id: "concept:\(name):reference",
                                title: "reference — \(name)",
                                kind: .conceptStimuli,
                                status: .invalid,
                                path: "prompts/emotions/<reference>/stories.jsonl",
                                detail: "designated-reference concept has no pinned "
                                    + "reference — re-attach with a reference"))
                    }
                }
            }

            // Both homes are offered to the checklist: the recipe's
            // canonical one and the OTHER recipe's, which the engine's
            // lookup falls back to (`ExperimentStore.resolveConceptValidation`).
            // A misfiled set is READ, so it is not a blocker — but it is a
            // finding, and the row names where it belongs.
            let validationPath = DataTemplates.validationDestination(
                concept: name, isPaired: isPaired)
            let otherValidationPath = DataTemplates.validationDestination(
                concept: name, isPaired: !isPaired)
            rows.append(
                validationRequirement(
                    concept: name, path: validationPath,
                    url: resolve(validationPath),
                    fallbackPath: otherValidationPath,
                    fallbackURL: resolve(otherValidationPath)))

            let markersPath = DataTemplates.markersDestination(concept: name)
            rows.append(
                markersRequirement(
                    concept: name, path: markersPath, url: resolve(markersPath)))
        }

        // Task prompts (model-output studies): the measured item set, plus
        // the per-item `options` the categorical instruments require.
        if manifest.studyKind == .modelOutput {
            rows.append(
                taskPromptsRequirement(
                    manifest: manifest, resolve: resolve))
        }

        // Declared numeric parser: the registry file, the named entry, and
        // the freeze pin are all run-start gates — say so here first. No
        // parser named = no row (legacy studies stay silent).
        if let row = numericParserRequirement(manifest: manifest, resolve: resolve) {
            rows.append(row)
        }

        // Declared exclusion rules / per-item attention checks: the
        // run/analyze-start refusals surface here first, and checks that
        // exist without the rule that would apply them get an honest
        // "unused" row.
        rows.append(
            contentsOf: exclusionRulesRequirements(
                manifest: manifest, resolve: resolve))

        // Judge protocol: declared by a paired-judge evaluation, a pinned
        // rubric, or a judge panel — any of them implies a pinned rubric
        // file and at least one judge.
        let judgeCount = manifest.judges?.count ?? 0
        let judgeProtocolDeclared =
            manifest.evaluation?.kind == .pairedJudge
            || manifest.judgeRubricFile != nil
            || judgeCount > 0
        if judgeProtocolDeclared {
            if let rubric = manifest.judgeRubricFile {
                let present = exists(rubric)
                rows.append(
                    DataRequirement(
                        id: "judgeRubric",
                        title: "judge rubric",
                        kind: .judgeRubric,
                        status: present ? .present : .missing,
                        path: rubric,
                        detail: present
                            ? "pinned rubric file exists (verified by hash at "
                                + "evaluate time)"
                            : "the pinned rubric file is missing — restore it or pin "
                                + "another from prompts/rubrics/"))
            } else {
                // Concrete file destination (not the bare directory) so the
                // judge-rubric template can scaffold a starting file there.
                rows.append(
                    DataRequirement(
                        id: "judgeRubric",
                        title: "judge rubric",
                        kind: .judgeRubric,
                        status: .missing,
                        path: DataTemplates.judgeRubricDestination(
                            experiment: manifest.name),
                        detail: "no rubric file pinned — inline draft text cannot "
                            + "freeze a judge-evaluated study; pin a file under "
                            + "prompts/rubrics/ (e.g. "
                            + "\(JudgeRubricStore.defaultRubricFile))",
                        templateID: DataTemplates.judgeRubric.id))
            }
            // Count alone is not a panel (finding 4, 2026-07-22): two judges
            // that RESOLVE to the same deterministic judge would agree
            // perfectly by construction — the freeze gate refuses, so the
            // data check says so first.
            let indistinct = ExperimentStore.judgePanelIndistinctProblem(manifest)
            // A judged SWEEP whose local judge cannot load inside the
            // chain (finding 1, live incident 2026-07-22) is a freeze
            // gate; the data check says so first, with the same wording.
            // An evaluate stage with such judges ROUTES to the server's
            // post-generation judge fan-out (2026-07-23) — informational,
            // never a blocker.
            let pipelineProblem = ExperimentStore.localJudgePipelineProblem(manifest)
            let fanoutNote = ExperimentStore.localJudgeFanoutNote(manifest)
            let panelProblem = indistinct ?? pipelineProblem
            // One judge is PRESENT, not partial: a single-coder design is a
            // legal methodology and freeze accepts it (2026-08-28 ruling).
            // What the row says is what that design cannot report — the same
            // sentence the advisory and the freeze envelope carry.
            let panelStatus: DataRequirement.Status =
                judgeCount >= 1
                ? (panelProblem == nil ? .present : .invalid)
                : .missing
            rows.append(
                DataRequirement(
                    id: "judgePanel",
                    title: "judge panel",
                    kind: .judgePanel,
                    status: panelStatus,
                    path: "experiments/\(manifest.name)/experiment.json",
                    detail: judgeCount >= 2
                        ? (panelProblem
                            ?? "\(judgeCount) judges pinned — agreement statistics can "
                                + "be computed"
                                + (fanoutNote.map { "; \($0)" } ?? ""))
                        : judgeCount == 1
                            ? (panelProblem
                                ?? ExperimentStore.singleJudgePanelAdvisoryText
                                    + (fanoutNote.map { "; \($0)" } ?? ""))
                            : "no judge pinned — a judged instrument with no "
                                + "judge codes nothing, and freeze refuses"))
        }

        // Pipeline evaluate-stage coherence (2026-07-22 incident): a chain
        // declaring 'evaluate' with no effective paired-judge declaration
        // (no explicit evaluation block, and not the judges + rubric-file
        // pin pair) dies at the evaluate stage after generation — a blocker
        // here with the same remedy the verify violation gives. Only read
        // once the pipeline block itself parses (a malformed block is the
        // verify violation's job).
        if ExperimentStore.pipelineBlockViolations(manifest.pipeline).isEmpty,
            let pipelineDraft = PipelineDraft.parse(manifest.pipeline),
            pipelineDraft.stages.contains("evaluate"),
            ExperimentStore.effectiveEvaluation(manifest)?.spec.kind != .pairedJudge
        {
            rows.append(
                DataRequirement(
                    id: "pipelineEvaluate",
                    title: "pipeline evaluate stage",
                    kind: .judgePanel,
                    status: .missing,
                    path: "experiments/\(manifest.name)/experiment.json",
                    detail: ExperimentStore.evaluateWithoutJudgingViolation))
        }

        // Human baseline: pinned → checked on disk AND for the header shape
        // the analyze loader reads (a present-but-wrong-shape file is
        // `.invalid` — analyze would refuse it); unpinned → the optional
        // row that says what pinning it would unlock.
        if let baseline = manifest.humanBaseline {
            let data = try? Data(contentsOf: resolve(baseline.path))
            let shapeProblem = data.flatMap {
                PinShapeValidation.humanBaselineShapeProblem(
                    $0, file: baseline.path)
            }
            rows.append(
                DataRequirement(
                    id: "humanBaseline",
                    title: "human baseline",
                    kind: .humanBaseline,
                    status: data == nil
                        ? .missing : (shapeProblem == nil ? .present : .invalid),
                    path: baseline.path,
                    detail: data == nil
                        ? "the pinned human-baseline table is missing — restore the "
                            + "exact pinned file"
                        : shapeProblem
                            ?? "pinned human-effect table exists and carries the "
                                + "columns analyze reads (drift is checked by hash)",
                    templateID: DataTemplates.humanBaseline.id))
        } else {
            rows.append(
                DataRequirement(
                    id: "humanBaseline",
                    title: "human-baseline CSV",
                    kind: .humanBaseline,
                    status: .optional,
                    path: DataTemplates.humanBaselineDestination(
                        experiment: manifest.name),
                    detail: "human-baseline CSV — required only for human-anchored "
                        + "(R) claims; without it results stay model-internal",
                    templateID: DataTemplates.humanBaseline.id))
        }

        // Reasoning-style taxonomy: pinned → checked (present iff it loads);
        // unpinned → the optional row that says what pinning it would unlock.
        rows.append(
            reasoningStyleRequirement(manifest: manifest, resolve: resolve))

        // Multi-agent scenario: the study's executable protocol.
        if manifest.studyKind == .multiAgent {
            rows.append(
                scenarioRequirement(manifest: manifest, resolve: resolve,
                                    workspaceIsServer: workspaceIsServer))
        }

        // Capability battery: pinned → checked; declared via the sweep →
        // checked at the sweep's path; otherwise optional evidence.
        rows.append(batteryRequirement(manifest: manifest, resolve: resolve))

        // Neutral corpus: the α denominator (norm-unit steering strengths).
        let corpusPath = "prompts/neutral/corpus.jsonl"
        let corpusPresent =
            (try? StimulusSet.loadTexts(url: resolve(corpusPath))) != nil
        // Needed only where concepts are OPERATIVE (carried-inert concepts
        // must not make the corpus read as a blocker).
        let corpusNeeded = machinery && !manifest.concepts.isEmpty
        rows.append(
            DataRequirement(
                id: "neutralCorpus",
                title: "neutral corpus",
                kind: .neutralCorpus,
                status: corpusPresent ? .present : (corpusNeeded ? .missing : .optional),
                path: corpusPath,
                detail: corpusPresent
                    ? "present — the fixed denominator that makes α comparable "
                        + "across concepts"
                    : corpusNeeded
                        ? "missing — residual-norm units (the α denominator) are "
                            + "measured on this corpus; attach pins it when present"
                        : "no concepts attached yet — needed once a concept's α is "
                            + "reported in residual-norm units"))

        rows.append(jlensReadoutRequirement(manifest: manifest))

        return rows
    }

    /// Counts by status; `blockers` = the `.missing` + `.invalid` items
    /// (both stop the run — absence, or a file the run refuses to load).
    public static func summary(_ requirements: [DataRequirement]) -> ReadinessSummary {
        ReadinessSummary(
            presentCount: requirements.count { $0.status == .present },
            partialCount: requirements.count { $0.status == .partial },
            invalidCount: requirements.count { $0.status == .invalid },
            missingCount: requirements.count { $0.status == .missing },
            optionalCount: requirements.count { $0.status == .optional },
            blockers: requirements.filter {
                $0.status == .missing || $0.status == .invalid
            })
    }

    // MARK: - Scaffolding

    /// Copies the requirement's template to its destination (creating
    /// directories), refusing to overwrite anything. Returns the created
    /// file URL.
    @discardableResult
    public static func scaffold(
        requirement: DataRequirement, in workspaceRoot: URL
    ) throws -> URL {
        guard
            let templateID = requirement.templateID,
            let template = DataTemplates.template(id: templateID)
        else {
            throw ExperimentError(
                reason: "no template exists for '\(requirement.title)' — author the "
                    + "file at \(requirement.path)")
        }
        let seed = DataTemplates.seedURL(for: template, workspaceRoot: workspaceRoot)
        guard FileManager.default.fileExists(atPath: seed.path) else {
            throw ExperimentError(
                reason: "template seed missing: \(template.seedRelativePath)")
        }
        let destination =
            requirement.path.hasPrefix("/")
            ? URL(filePath: requirement.path)
            : workspaceRoot.appending(path: requirement.path)
        guard !FileManager.default.fileExists(atPath: destination.path) else {
            throw ExperimentError(
                reason: "refusing to overwrite existing file: \(requirement.path) — "
                    + "edit it in place instead")
        }
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: seed, to: destination)
        return destination
    }

    /// Pin a file this checklist just scaffolded, for requirement kinds where
    /// the PIN — not the file — is what was missing.
    ///
    /// The multi-agent scenario is the case that matters: its requirement
    /// begins "is a scenario pinned?", so creating the file alone can never
    /// satisfy it. Scaffolding then left the blocker standing with the file
    /// sitting right there, which reads as the tool being broken.
    ///
    /// Returns true when it pinned something, so a caller knows to save.
    @discardableResult
    public static func pinScaffolded(
        requirement: DataRequirement, createdPath: String,
        into manifest: inout ExperimentManifest, workspaceRoot: URL
    ) throws -> Bool {
        guard requirement.kind == .multiAgentScenario else { return false }
        let url = createdPath.hasPrefix("/")
            ? URL(filePath: createdPath)
            : workspaceRoot.appending(path: createdPath)
        manifest.multiAgentScenarioPath = createdPath
        manifest.multiAgentScenarioHash = try MultiAgentScenarioStore.hash(url)
        return true
    }

    // MARK: - Per-kind derivations

    private static func validationRequirement(
        concept: String, path: String, url: URL,
        fallbackPath: String, fallbackURL: URL
    ) -> DataRequirement {
        let id = "concept:\(concept):validation"
        let title = "validation set — \(concept)"
        let fm = FileManager.default
        // Same resolution order the engine uses
        // (`ExperimentStore.resolveConceptValidation`): the recipe's
        // canonical home first, the other recipe's home as a fallback. A
        // misfiled set is found and READ, so it never reads as missing here
        // either — it reads as a non-blocking finding naming its canonical
        // home. `path` stays the canonical destination: that is where the
        // template scaffolds to and where the file belongs.
        let canonicalExists = fm.fileExists(atPath: url.path)
        let fallbackExists =
            fm.fileExists(atPath: fallbackURL.path)
            && fallbackURL.standardizedFileURL != url.standardizedFileURL
        let readURL = canonicalExists ? url : fallbackURL
        let usedFallback = !canonicalExists && fallbackExists
        // One sentence about WHERE, appended to whatever the row says about
        // the contents.
        let filingNote: String? =
            usedFallback
            ? " — NOTE: it is filed under the other recipe's root "
                + "(\(fallbackPath)); the engine falls back to it, but this "
                + "recipe's canonical home is \(path)"
            : (canonicalExists && fallbackExists
                ? " — NOTE: a second validation.jsonl also sits under "
                    + "\(fallbackPath); the canonical \(path) is the one read "
                    + "and pinned, so delete or merge the other"
                : nil)
        // The gate's own loader: rows must be {"text", "expresses"} objects.
        // `try?` of an optional-returning throwing call: outer nil = a row is
        // malformed (threw); inner nil = no file.
        let loaded =
            (canonicalExists || fallbackExists)
            ? (try? StimulusSet.loadValidation(
                directory: readURL.deletingLastPathComponent()))
            : nil
        if let rows = loaded ?? nil, !rows.isEmpty {
            return DataRequirement(
                id: id, title: title, kind: .conceptValidation,
                // A misfiled or duplicated set works, so it is never a
                // blocker — but it is not clean either.
                status: filingNote == nil ? .present : .partial,
                path: path,
                detail: "\(rows.count) never-named scenarios — validate reads these "
                    + "as the convergent-validity gate" + (filingNote ?? ""),
                templateID: DataTemplates.validation.id)
        }
        if canonicalExists || fallbackExists {
            return DataRequirement(
                id: id, title: title, kind: .conceptValidation, status: .partial,
                path: path,
                detail: "validation.jsonl exists but has no readable "
                    + "{\"text\": …, \"expresses\": …} rows — fix it before validate"
                    + (filingNote ?? ""),
                templateID: DataTemplates.validation.id)
        }
        return DataRequirement(
            id: id, title: title, kind: .conceptValidation, status: .missing,
            path: path,
            detail: "never-named validation.jsonl is required before validate "
                + "can gate freeze — scenarios must evoke the concept without "
                + "naming it",
            templateID: DataTemplates.validation.id)
    }

    private static func markersRequirement(
        concept: String, path: String, url: URL
    ) -> DataRequirement {
        let id = "concept:\(concept):markers"
        let title = "markers — \(concept)"
        guard let data = try? Data(contentsOf: url) else {
            return DataRequirement(
                id: id, title: title, kind: .conceptMarkers, status: .optional,
                path: path,
                detail: "needed only for marker-density metrics (a diagnostic / "
                    + "manipulation check, never a promotion objective)",
                templateID: DataTemplates.markers.id)
        }
        let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        let words = object?["words"] as? [Any]
        guard let words, !words.isEmpty else {
            return DataRequirement(
                id: id, title: title, kind: .conceptMarkers, status: .partial,
                path: path,
                detail: "markers.json exists but has no {\"words\": [...]} list — "
                    + "marker-density metrics will read zero everywhere",
                templateID: DataTemplates.markers.id)
        }
        return DataRequirement(
            id: id, title: title, kind: .conceptMarkers, status: .present,
            path: path,
            detail: "\(words.count) marker words — drives marker-density diagnostics",
            templateID: DataTemplates.markers.id)
    }

    private static func taskPromptsRequirement(
        manifest: ExperimentManifest, resolve: (String) -> URL
    ) -> DataRequirement {
        let id = "taskPrompts"
        let title = "task prompts"
        let templateID = DataTemplates.taskPromptsChoice.id
        guard let file = manifest.taskPromptsFile else {
            return DataRequirement(
                id: id, title: title, kind: .taskPrompts, status: .missing,
                path: DataTemplates.taskPromptsDestination(experiment: manifest.name),
                detail: "no task-prompts file declared — a measured study needs a "
                    + "pinned prompt set (frozen studies refuse to run without one); "
                    + "author the file at this path, then 'steerlab-cli experiment "
                    + "pin-prompts \(manifest.name) <path>'",
                templateID: templateID)
        }
        guard let data = try? Data(contentsOf: resolve(file)) else {
            return DataRequirement(
                id: id, title: title, kind: .taskPrompts, status: .missing,
                path: file,
                detail: "the declared task-prompts file is missing — restore it or "
                    + "point the study at another set",
                templateID: templateID)
        }
        let document: TaskPromptsDocument
        do {
            document = try TaskPromptsDocument.load(data)
        } catch {
            // A file the JSONL parser refuses is a file the run refuses —
            // a blocker (`.invalid`), never a "degraded run" promise.
            return DataRequirement(
                id: id, title: title, kind: .taskPrompts, status: .invalid,
                path: file,
                detail: "\(error) — the run refuses to load this file",
                templateID: templateID)
        }
        guard document.count > 0 else {
            return DataRequirement(
                id: id, title: title, kind: .taskPrompts, status: .partial,
                path: file,
                detail: "task-prompts file exists but has no readable JSONL rows",
                templateID: templateID)
        }
        // Duplicate item ids corrupt pairing and reporting on BOTH engines,
        // and the run loop refuses the file at load (identical cross-engine
        // message) — a blocker here so preflight matches execution, naming
        // EVERY duplicate so one edit fixes the file (the parser stops at
        // the first).
        let duplicateIDs = ExperimentTasks.duplicateTaskPromptIDs(data)
        if !duplicateIDs.isEmpty {
            let named = duplicateIDs.map { duplicate in
                "'\(duplicate.id)' (items "
                    + duplicate.items.map(String.init).joined(separator: ", ") + ")"
            }.joined(separator: "; ")
            return DataRequirement(
                id: id, title: title, kind: .taskPrompts, status: .invalid,
                path: file,
                detail: "duplicate item ids — \(named) — ids must be unique for "
                    + "pairing and reporting; the run refuses to load this file",
                templateID: templateID)
        }
        // Scripted-transcript items get the run loop's own validation here at
        // `data check` time (the earliest gate): schema first (identical
        // messages on both engines), then the study model family's
        // chat-template constraints — the detail line names the invalid
        // items, so a Gemma-incompatible transcript surfaces long before the
        // run-start refusal.
        if document.transcriptItemCount > 0 {
            let prompts: [ExperimentTasks.StudyPrompt]
            do {
                prompts = try ExperimentTasks.parseTaskPrompts(data)
            } catch {
                // The run loop's own parser refused (schema violation) — the
                // run will refuse this file identically, so it blocks here.
                return DataRequirement(
                    id: id, title: title, kind: .taskPrompts, status: .invalid,
                    path: file,
                    detail: (error as? ExperimentError)?.reason ?? "\(error)",
                    templateID: templateID)
            }
            var violations = prompts.compactMap { prompt in
                prompt.transcript.flatMap {
                    ExperimentTasks.transcriptFamilyViolation(
                        $0, itemID: prompt.id, modelID: manifest.modelID)
                }
            }
            if (manifest.promptMode ?? .chatAssistant) == .rawCompletion {
                violations.append(ExperimentTasks.transcriptRawCompletionMessage)
            }
            if !violations.isEmpty {
                // Family/prompt-mode constraints refuse at run start — a
                // blocker, not a degraded run.
                return DataRequirement(
                    id: id, title: title, kind: .taskPrompts, status: .invalid,
                    path: file,
                    detail: violations.joined(separator: "; "),
                    templateID: templateID)
            }
        }
        let instruments = Set(manifest.outcomeInstruments ?? [])
        let needsOptions = !instruments.isDisjoint(
            with: ["answerTokenLogprob", "choiceProbability", "ordinalScale"])
        if needsOptions {
            // Blocker, not a partial: this exact combination now refuses at
            // run start, so reporting it as "degraded" would understate it.
            let unscorable = document.unscorableOptionItemCount
            if unscorable > 0, manifest.outcomeInstrumentScope == nil {
                return DataRequirement(
                    id: id, title: title, kind: .taskPrompts, status: .missing,
                    path: file,
                    detail: "\(unscorable) of \(document.count) items declare a "
                        + "responseFormat the declared answer-token instrument "
                        + "cannot read (the choice is not at the first generated "
                        + "position) — the run refuses this combination; change "
                        + "those rows to responseFormat 'label', drop the "
                        + "instrument, or declare outcomeInstrumentScope",
                    templateID: templateID)
            }
            // Every remaining run-start refusal of this combination —
            // ZERO items carrying options, a scope selecting zero items —
            // is a blocker stated with the run loop's own refusal string,
            // so preflight matches execution exactly (2026-08-06: the
            // zero-options case used to read as a "degraded" partial here
            // while the instrument silently produced zero records).
            if let refusal = ResponseFormat.refusal(
                items: document.responseFormatItems,
                declaredInstruments: manifest.outcomeInstruments,
                declaredScope: manifest.outcomeInstrumentScope)
            {
                return DataRequirement(
                    id: id, title: title, kind: .taskPrompts, status: .missing,
                    path: file,
                    detail: refusal,
                    templateID: templateID)
            }
            let lacking = document.count - document.optionsItemCount
            if lacking > 0 {
                return DataRequirement(
                    id: id, title: title, kind: .taskPrompts, status: .partial,
                    path: file,
                    detail: "\(lacking) of \(document.count) task prompts lack the "
                        + "`options` field the logprob instrument requires",
                    templateID: templateID)
            }
            return DataRequirement(
                id: id, title: title, kind: .taskPrompts, status: .present,
                path: file,
                detail: "all \(document.count) items carry the `options` field the "
                    + "declared answer-token instrument reads",
                templateID: templateID)
        }
        // Instrument-activation honesty (team finding P1): items CARRY
        // options but the manifest declares no categorical instrument —
        // fields preserved ≠ measurement enabled. Partial, never a blocker:
        // the study runs, but only generates and parses answer text.
        if document.optionsItemCount > 0 {
            // ...but only recommend the instrument the data can actually
            // support. A file whose rows ask for a JSON response cannot be
            // read by answer-token scoring at all (the first generated token
            // is `{`), and the run loop refuses that combination — so
            // advising it here would send the researcher to a dead end.
            let unscorable = document.unscorableOptionItemCount
            let advice: String
            if unscorable >= document.optionsItemCount {
                advice = "every one asks for a JSON or free-text response, so "
                    + "answer-token probability cannot read them — sampled "
                    + "text parsed for a choice is the right instrument here"
            } else if unscorable > 0 {
                advice = "\(unscorable) ask for a JSON or free-text response "
                    + "that answer-token probability cannot read; declaring it "
                    + "needs an outcome-instrument scope limiting it to the "
                    + "label rows"
            } else {
                advice = "declare outcomeInstruments (answerTokenLogprob) to "
                    + "measure choice probabilities"
            }
            return DataRequirement(
                id: id, title: title, kind: .taskPrompts, status: .partial,
                path: file,
                detail: "\(document.optionsItemCount) of \(document.count) items "
                    + "carry `options`, but no categorical outcome instrument is "
                    + "declared — the study will only generate and parse answer "
                    + "text; " + advice,
                templateID: templateID)
        }
        let transcriptNote =
            document.transcriptItemCount > 0
            ? " (\(document.transcriptItemCount) with scripted transcripts — "
                + "family constraints check out)"
            : ""
        return DataRequirement(
            id: id, title: title, kind: .taskPrompts, status: .present,
            path: file,
            detail: "\(document.count) task prompts load" + transcriptNote,
            templateID: templateID)
    }

    private static func scenarioRequirement(
        manifest: ExperimentManifest, resolve: (String) -> URL,
        workspaceIsServer: Bool = false
    ) -> DataRequirement {
        let id = "multiAgentScenario"
        let title = "multi-agent scenario"
        let templateID = DataTemplates.scenario.id
        guard let path = manifest.multiAgentScenarioPath else {
            return DataRequirement(
                id: id, title: title, kind: .multiAgentScenario, status: .missing,
                path: DataTemplates.scenarioDestination(experiment: manifest.name),
                detail: "no scenario pinned. Selecting one in Study Setup does "
                    + "NOT pin it — the pin is written when you SAVE the study "
                    + "setup (or the seat casting). Pick a scenario, cast its "
                    + "seats and save, or create one from the template here "
                    + "(which pins it for you).",
                templateID: templateID)
        }
        guard let data = try? Data(contentsOf: resolve(path)) else {
            return DataRequirement(
                id: id, title: title, kind: .multiAgentScenario, status: .missing,
                path: path,
                detail: workspaceIsServer
                    ? "the pinned scenario is not in THIS workspace. Under a "
                        + "server target the study may reference a file that "
                        + "exists on the server: this checklist only reads the "
                        + "local tree, so verify there before re-authoring it."
                    : "the pinned scenario is missing — restore it or pin "
                        + "another scenario",
                templateID: templateID)
        }
        guard
            let scenario = try? JSONDecoder().decode(MultiAgentScenario.self, from: data)
        else {
            return DataRequirement(
                id: id, title: title, kind: .multiAgentScenario, status: .partial,
                path: path,
                detail: "scenario file exists but does not parse as a scenario — "
                    + "fix it before the study can run",
                templateID: templateID)
        }
        return DataRequirement(
            id: id, title: title, kind: .multiAgentScenario, status: .present,
            path: path,
            detail: "scenario parses: \(scenario.agents.count) agents, "
                + "\(scenario.turns.count) turns",
            templateID: templateID)
    }

    /// J-lens readout readiness, from the MANIFEST ALONE.
    ///
    /// Server-only by hard requirement (CLAUDE.md): imported lens artifacts
    /// are PyTorch/HF-native, this engine never produces one, and it does not
    /// resolve, load, or verify the lens here either. What it can honestly
    /// check is what the study DECLARED — and that is worth checking, because
    /// every one of these is a freeze refusal the researcher would otherwise
    /// meet only at the end.
    ///
    /// The pin set mirrors the server's `_check_jlens_readout`. It is
    /// duplicated rather than shared because the two engines cannot import
    /// each other; a cross-engine test pins the key list so the copies cannot
    /// drift silently.
    static func jlensReadoutRequirement(
        manifest: ExperimentManifest
    ) -> DataRequirement {
        let id = "jlensReadout"
        let title = "J-lens readout"
        let path = "experiments/\(manifest.name)/experiment.json"

        // A panel study runs a SCENARIO and never arms a readout. It may
        // carry a block from before a kind switch (the never-delete rule), so
        // grading it here would report a blocker — `data check` exits 2 on
        // `.missing`/`.invalid` — for an instrument this study cannot run
        // (round-12 sweep).
        guard manifest.studyKind == .modelOutput else {
            return DataRequirement(
                id: id, title: title, kind: .jlensReadout, status: .optional,
                path: path,
                detail: "not applicable — a multi-agent study runs a scenario "
                    + "and arms no J-lens readout. Any block here is carried "
                    + "configuration from another study kind: preserved, never "
                    + "executed, and not graded")
        }

        guard case .object(let block)? = manifest.jlensReadout else {
            return DataRequirement(
                id: id, title: title, kind: .jlensReadout, status: .optional,
                path: path,
                detail: "no J-lens readout declared — without one a run records "
                    + "no readout at all. Authored on the SERVER (any published lens, "
                    + "server-only); this engine renders and carries the block, "
                    + "it never produces one.")
        }

        func text(_ key: String) -> String? {
            if case .string(let value)? = block[key], !value.isEmpty {
                return value
            }
            return nil
        }
        func nonEmptyArray(_ key: String) -> Bool {
            if case .array(let values)? = block[key] { return !values.isEmpty }
            return false
        }

        // Exactly the server's freeze demands, in its order.
        var missing: [String] = []
        // `qualificationID` joins the pin set for the same reason the others
        // are in it: without it a frozen study resolves whichever
        // qualification is newest, so appending one later moves the ground
        // under it (external review round 3). The server enforces the same
        // list; a test on each side asserts it.
        for key in ["lensID", "lensSHA256", "configHash", "tokenizerHash",
                    "qualificationID"]
        where text(key) == nil {
            missing.append(key)
        }
        if !nonEmptyArray("layers") { missing.append("layers") }
        if !missing.isEmpty {
            return DataRequirement(
                id: id, title: title, kind: .jlensReadout, status: .missing,
                path: path,
                detail: "jlensReadout is missing \(missing.joined(separator: ", "))"
                    + " — freeze refuses a readout that is not fully pinned, "
                    + "because one that cannot be reproduced cannot be cited.")
        }

        // Declaring neither means the run records nothing while looking armed.
        let records = nonEmptyArray("watchlist") || {
            if case .number(let k)? = block["topK"] { return k > 0 }
            return false
        }()
        if !records {
            return DataRequirement(
                id: id, title: title, kind: .jlensReadout, status: .invalid,
                path: path,
                detail: "jlensReadout declares neither a token watchlist nor a "
                    + "top-k width, so it would record nothing — freeze refuses "
                    + "it, and a run would otherwise complete looking armed.")
        }

        // Qualification is keyed by model + revision + dtype + quantization,
        // so a study missing either pin cannot be matched against one.
        var unpinned: [String] = []
        if (manifest.modelRevision ?? "").isEmpty { unpinned.append("modelRevision") }
        if (manifest.dtype ?? "").isEmpty { unpinned.append("dtype") }
        if !unpinned.isEmpty {
            return DataRequirement(
                id: id, title: title, kind: .jlensReadout, status: .missing,
                path: path,
                detail: "the study pins no \(unpinned.joined(separator: " and "))"
                    + " — a J-lens qualification is keyed by model, revision, "
                    + "dtype AND quantization, because geometry cannot see "
                    + "numerics: a float16 or quantized runtime passes every "
                    + "shape check while presenting different numbers to the "
                    + "same Jacobian.")
        }

        // Runs fine; degrades what the study can do LATER. Retention is not
        // retroactive, so this is worth saying before the run, not after.
        if !manifest.recordTokenIDs {
            return DataRequirement(
                id: id, title: title, kind: .jlensReadout, status: .partial,
                path: path,
                detail: "pinned, but recordTokenIDs is off: the exact sampled "
                    + "sequence will not be kept, so you could not go back and read a "
                    + "layer, token, or position this block did not name. "
                    + "Re-deriving ids from stored text is not a round trip — a "
                    + "generation that stops naturally ends with <end_of_turn>, "
                    + "which the streamer skips.")
        }

        let layerCount: Int = {
            if case .array(let values)? = block["layers"] { return values.count }
            return 0
        }()
        return DataRequirement(
            id: id, title: title, kind: .jlensReadout, status: .present,
            path: path,
            detail: "pinned at \(layerCount) layer(s); token ids retained, so "
                + "the exact sampled sequence is kept for later teacher-forced "
                + "replay (which reproduces top-1 and coarse rank movement, "
                + "subject to measured decode-vs-batch divergence — not a "
                + "bit-identical re-run). Freeze still requires a PASSING "
                + "qualification for this exact runtime, which only the server "
                + "can produce (jlens qualify).")
    }

    private static func reasoningStyleRequirement(
        manifest: ExperimentManifest, resolve: (String) -> URL
    ) -> DataRequirement {
        let id = "reasoningStyleTaxonomy"
        let title = "reasoning-style taxonomy"
        let templateID = DataTemplates.reasoningStyle.id
        guard let path = manifest.reasoningStyleTaxonomyPath else {
            return DataRequirement(
                id: id, title: title, kind: .reasoningStyleTaxonomy,
                status: .optional,
                path: DataTemplates.reasoningStyleDestination(
                    experiment: manifest.name),
                detail: "reasoning-style taxonomy — required for reasoning-style "
                    + "claims (rs_<feature> metrics + effect sizes); without it "
                    + "style is measured only by marker density and the judge",
                templateID: templateID)
        }
        guard let data = try? Data(contentsOf: resolve(path)) else {
            return DataRequirement(
                id: id, title: title, kind: .reasoningStyleTaxonomy,
                status: .missing,
                path: path,
                detail: "the pinned taxonomy file is missing — restore the exact "
                    + "pinned file (drift is checked by hash)",
                templateID: templateID)
        }
        do {
            let taxonomy = try ReasoningStyleTaxonomy.load(data: data)
            return DataRequirement(
                id: id, title: title, kind: .reasoningStyleTaxonomy,
                status: .present,
                path: path,
                detail: "\(taxonomy.features.count) feature(s) — scores rs_<id> "
                    + "columns and effect sizes per generation",
                templateID: templateID)
        } catch {
            return DataRequirement(
                id: id, title: title, kind: .reasoningStyleTaxonomy,
                status: .partial,
                path: path,
                detail: "taxonomy file exists but does not load: "
                    + "\((error as? ExperimentError)?.reason ?? "\(error)")",
                templateID: templateID)
        }
    }

    private static func batteryRequirement(
        manifest: ExperimentManifest, resolve: (String) -> URL
    ) -> DataRequirement {
        let id = "capabilityBattery"
        let title = "capability battery"
        // A present-but-wrong-shape battery is `.invalid` (a blocker): the
        // battery runner (`CapabilityBattery`) would refuse the file, so
        // preflight says so with the same plain detail the pin gives.
        func shapeProblem(_ file: String) -> String?? {
            guard let data = try? Data(contentsOf: resolve(file)) else {
                return nil  // outer nil: file missing
            }
            return .some(
                PinShapeValidation.capabilityBatteryShapeProblem(data, file: file))
        }
        if let file = manifest.capabilityBatteryFile {
            switch shapeProblem(file) {
            case nil:
                return DataRequirement(
                    id: id, title: title, kind: .capabilityBattery,
                    status: .missing,
                    path: file,
                    detail: "the pinned capability battery is missing — restore it "
                        + "or pin another from prompts/batteries/")
            case .some(let problem?):
                return DataRequirement(
                    id: id, title: title, kind: .capabilityBattery,
                    status: .invalid,
                    path: file,
                    detail: problem)
            case .some(nil):
                return DataRequirement(
                    id: id, title: title, kind: .capabilityBattery,
                    status: .present,
                    path: file,
                    detail: "pinned battery loads — validate scores it through "
                        + "every condition as evidence")
            }
        }
        // The DEFAULT-battery path (no pinned file, variant conditions
        // present): validate/freeze pin the shared default preset battery,
        // and that pin SKIPS a malformed file (treated as absent — the
        // nil-tolerant freeze-time contract), so readiness must say WHY:
        // the file is invalid, with the same plain shape detail the pin
        // computes. An ABSENT default keeps the historical optional row
        // below — absence was always the honestly-reported state.
        if !manifest.variantConditions.isEmpty,
            case .some(let problem?) = shapeProblem(
                VariantRobustness.defaultPreset.batteryFile)
        {
            return DataRequirement(
                id: id, title: title, kind: .capabilityBattery,
                status: .invalid,
                path: VariantRobustness.defaultPreset.batteryFile,
                detail: problem)
        }
        // A sweep-declared battery is required only while the concept
        // machinery is OPERATIVE — a compare-agents study carrying an old
        // sweep must not show a false battery blocker (engineer finding
        // 2026-07-19).
        if let sweep = manifest.sweep,
            ExperimentStore.conceptMachineryOperative(manifest)
        {
            switch shapeProblem(sweep.batteryFile) {
            case nil:
                return DataRequirement(
                    id: id, title: title, kind: .capabilityBattery,
                    status: .missing,
                    path: sweep.batteryFile,
                    detail: "the declared sweep points at a battery file that does "
                        + "not exist — the sweep will refuse")
            case .some(let problem?):
                return DataRequirement(
                    id: id, title: title, kind: .capabilityBattery,
                    status: .invalid,
                    path: sweep.batteryFile,
                    detail: problem)
            case .some(nil):
                return DataRequirement(
                    id: id, title: title, kind: .capabilityBattery,
                    status: .present,
                    path: sweep.batteryFile,
                    detail: "the declared sweep's battery loads (capability "
                        + "guardrail for cell selection)")
            }
        }
        return DataRequirement(
            id: id, title: title, kind: .capabilityBattery, status: .optional,
            path: "prompts/batteries/",
            detail: "required for evidence-grade variant/sweep gates — pin one "
                + "from prompts/batteries/ (e.g. basic.jsonl)")
    }

    /// The declared numeric parser's registry row: file missing, named
    /// entry undefined/malformed, or registry drifted from the pinned hash
    /// — each a blocker, because the run refuses to start on exactly these
    /// (`ParserRegistry.resolveNumericParser`). A study that names no
    /// parser gets NO row: legacy behavior is the built-in parser and
    /// stays silent.
    private static func numericParserRequirement(
        manifest: ExperimentManifest, resolve: (String) -> URL
    ) -> DataRequirement? {
        let name = manifest.numericParser?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !name.isEmpty else { return nil }
        let id = "numericParser"
        let title = "numeric parser — \(name)"
        let path = ParserRegistry.registryFile
        let url = resolve(path)
        guard let liveHash = ParserRegistry.liveHash(at: url) else {
            return DataRequirement(
                id: id, title: title, kind: .numericParser, status: .missing,
                path: path,
                detail: "the study names numeric parser '\(name)' but no "
                    + "parser registry exists at \(path) — create the registry "
                    + "(new workspaces are seeded with a template), or remove "
                    + "the study's numericParser; the run refuses to start "
                    + "without it")
        }
        if let pinned = manifest.parserRegistryHash, pinned != liveHash {
            return DataRequirement(
                id: id, title: title, kind: .numericParser, status: .invalid,
                path: path,
                detail: "the parser registry drifted from the pinned hash "
                    + "(have \(liveHash.prefix(12))…, pinned "
                    + "\(pinned.prefix(12))…) — restore the pinned file, or "
                    + "duplicate the study to re-pin; the run refuses drifted "
                    + "registries")
        }
        do {
            let spec = try ParserRegistry.spec(named: name, at: url)
            return DataRequirement(
                id: id, title: title, kind: .numericParser, status: .present,
                path: path,
                detail: "parser '\(name)' (\(spec.kind ?? "?")) is defined in "
                    + "the registry — the run parses numeric answers with this "
                    + "declared grammar"
                    + (manifest.parserRegistryHash != nil
                        ? " (drift is checked by hash)"
                        : "; freeze pins the registry by hash"))
        } catch {
            return DataRequirement(
                id: id, title: title, kind: .numericParser, status: .invalid,
                path: path,
                detail: ((error as? ExperimentError)?.reason ?? "\(error)")
                    + " — the run refuses to start until the registry "
                    + "defines it")
        }
    }

    /// Exclusion-rule / attention-check rows, derived only from what the
    /// manifest and the task-prompt items declare:
    /// - malformed `exclusionRules` → blocker (verify and run refuse on the
    ///   same plain-language problems);
    /// - `failedAttentionCheck` declared but zero items carry an
    ///   `attentionCheck` → blocker (the run refuses at start — this row
    ///   says it earlier, with the remedy);
    /// - items carry checks but no rule applies them → advisory (checks
    ///   exist but are unused);
    /// - an endpoint-reading rule (`unparseableEndpoint`/`outOfRange`)
    ///   whose default `parsedMonths` endpoint can never be produced (no
    ///   `numericParser`, case family not "sentencing") → advisory. Custom
    ///   endpoint names are NOT checked (whether a record carries them
    ///   needs run data, not manifest inference).
    /// No rules declared and no checks present = no rows (legacy silence).
    private static func exclusionRulesRequirements(
        manifest: ExperimentManifest, resolve: (String) -> URL
    ) -> [DataRequirement] {
        let rules = manifest.exclusionRules ?? []
        let manifestPath = "experiments/\(manifest.name)/experiment.json"

        // Malformed rules refuse at verify AND run start — one blocker row
        // naming every problem.
        let ruleProblems = ExclusionEngine.violations(
            rules.isEmpty ? nil : rules)
        if !ruleProblems.isEmpty {
            return [
                DataRequirement(
                    id: "exclusionRules",
                    title: "exclusion rules",
                    kind: .exclusionRules,
                    status: .invalid,
                    path: manifestPath,
                    detail: ruleProblems.joined(separator: "; ")
                        + " — the run refuses to start until the declared "
                        + "exclusionRules are fixed")
            ]
        }

        // Per-item checks, from the declared task-prompts file. An
        // unreadable/unparseable file is the task-prompts row's blocker —
        // no duplicate finding here (checkedCount nil = unknown).
        var checkedCount: Int?
        var itemCount = 0
        var optionLadders: [[String]] = []
        if let file = manifest.taskPromptsFile,
            let data = try? Data(contentsOf: resolve(file)),
            let prompts = try? ExperimentTasks.parseTaskPrompts(data)
        {
            itemCount = prompts.count
            checkedCount = prompts.count { $0.attentionCheck != nil }
            optionLadders = prompts.compactMap(\.options)
        }

        var rows: [DataRequirement] = []
        let declaresAttention = rules.contains {
            $0.rule == ExclusionEngine.ruleFailedAttentionCheck
        }
        if declaresAttention {
            if let checkedCount, checkedCount > 0 {
                rows.append(
                    DataRequirement(
                        id: "attentionChecks",
                        title: "attention checks",
                        kind: .exclusionRules,
                        status: .present,
                        path: manifest.taskPromptsFile ?? manifestPath,
                        detail: "\(checkedCount) of \(itemCount) task-prompt "
                            + "items carry an attentionCheck — records that "
                            + "fail them drop from the paired statistics and "
                            + "judging, and a cell whose every sampled answer "
                            + "fails also drops its instrument readouts "
                            + "(stamped in the report)"))
            } else {
                // Zero checked items (or no readable prompt set at all):
                // the run/analyze refusal, said early and in plain words.
                let counted =
                    checkedCount == nil
                    ? "the study declares no readable task-prompts file"
                    : "none of the \(itemCount) task-prompt items carries "
                        + "an attentionCheck"
                rows.append(
                    DataRequirement(
                        id: "attentionChecks",
                        title: "attention checks",
                        kind: .exclusionRules,
                        status: .missing,
                        path: manifest.taskPromptsFile ?? manifestPath,
                        detail: "the failedAttentionCheck exclusion rule is "
                            + "declared but \(counted) — add "
                            + "{\"attentionCheck\": {\"expected\": …}} to at "
                            + "least one item (graded with the battery's "
                            + "vocabulary), or remove the rule; the run "
                            + "refuses to start otherwise"))
            }
        } else if let checkedCount, checkedCount > 0 {
            rows.append(
                DataRequirement(
                    id: "attentionChecks",
                    title: "attention checks",
                    kind: .exclusionRules,
                    status: .partial,
                    path: manifest.taskPromptsFile ?? manifestPath,
                    detail: "\(checkedCount) task-prompt item(s) carry an "
                        + "attentionCheck but no failedAttentionCheck "
                        + "exclusion rule is declared — the checks exist but "
                        + "are unused; declare exclusionRules "
                        + "[{\"rule\": \"failedAttentionCheck\"}] to apply "
                        + "them"))
        }

        // Endpoint-reading rules whose default endpoint can never appear
        // on a record: parsedMonths is produced only by a declared
        // numericParser or the sentencing case family (the record
        // factory's dispatch rule).
        let producesParsedMonths =
            !(manifest.numericParser ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || manifest.caseFamily == "sentencing"
        for rule in rules
        where rule.rule != ExclusionEngine.ruleFailedAttentionCheck {
            let endpoint = ExclusionEngine.resolvedEndpoint(rule)
            guard endpoint == ExclusionEngine.defaultEndpoint,
                !producesParsedMonths
            else { continue }
            rows.append(
                DataRequirement(
                    id: "exclusionEndpoint:\(rule.rule)",
                    title: "exclusion rule — \(rule.rule)",
                    kind: .exclusionRules,
                    status: .partial,
                    path: manifestPath,
                    detail: "the \(rule.rule) rule reads '\(endpoint)', but "
                        + "no record of this study can carry it — "
                        + "parsedMonths is parsed only when the study "
                        + "declares a numericParser or the sentencing case "
                        + "family; the rule will exclude nothing as declared"))
        }

        // Ladder-window advisory (2026-08-06) — the readiness-layer mirror
        // of the run-start warning both engines log: a declared outOfRange
        // keep-window whose every bound lies outside the scale the items'
        // numeric options ladders imply cannot bind that scale. Advisory,
        // not a blocker (the endpoint may lawfully take non-ladder values).
        for warning in ExclusionEngine.ladderWarnings(
            rules: rules.isEmpty ? nil : rules, optionLadders: optionLadders)
        {
            rows.append(
                DataRequirement(
                    id: "exclusionLadderWindow",
                    title: "exclusion rule — outOfRange window",
                    kind: .exclusionRules,
                    status: .partial,
                    path: manifestPath,
                    detail: warning))
        }
        return rows
    }

    /// A blocker description when either stimulus class is degenerate, or
    /// nil when the data is extraction-worthy. Domain-neutral, and
    /// deliberately class-level: SHORT stimuli are a legitimate paired
    /// design ("I am terrified."), so no per-row absolute floor exists —
    /// the observed failure mode is a whole class of debris (a reference
    /// corpus's NAME iterated character-by-character into one-character
    /// rows), which the MEDIAN catches without flagging real minimal
    /// pairs. The reading-position floor is per-row and strict because it
    /// mirrors a hard runtime rule: extraction refuses any row shorter
    /// than the pooling start, so a row under ~2 chars/pooled-token (far
    /// below English's ~4) fails here, before any model loads. The
    /// extractor's own refusal remains the loud backstop for what these
    /// heuristics let through.
    static func stimulusContentProblem(
        positive: [String], negative: [String],
        readingPosition: ReadingPosition
    ) -> String? {
        for (label, rows) in [("positive", positive), ("negative", negative)] {
            let lengths = rows.map {
                $0.trimmingCharacters(in: .whitespacesAndNewlines).count
            }.sorted()
            guard let shortest = lengths.first else { continue }
            let median = lengths[lengths.count / 2]
            // Narrowed after external review (2026-07-31, finding 5):
            // median < 5 is the observed derivation-bug signature
            // (one-character rows median 1); a terse-but-legitimate
            // minimal-pair set can median in the low teens, so that band
            // is a caller-surfaced ADVISORY, not a blocker.
            if median < 5 {
                return "\(label) class is degenerate — median row length "
                    + "\(median) character(s) over \(lengths.count) row(s) "
                    + "(shortest: \(shortest)); real stimuli are prose, and "
                    + "a class of fragments this short is the signature of "
                    + "a derivation bug (the class mean would be garbage)"
            }
            if case .meanFromToken(let k) = readingPosition, k > 0 {
                let floor = 2 * k
                let tooShort = lengths.filter { $0 < floor }.count
                if tooShort > 0 {
                    return "\(label) class has \(tooShort) row(s) under "
                        + "\(floor) characters — almost certainly shorter "
                        + "than \(k) tokens, and the pinned reading position "
                        + "'mean from token \(k)' makes extraction refuse "
                        + "such rows"
                }
            }
        }
        return nil
    }
}
