import Foundation

/// Which outcome leads a results summary, and the study's declared one.
///
/// A study's summary leads with the outcome the study is about. The rule, in
/// order: the outcome the researcher declared (manifest key
/// `primaryOutcome`); else a judged outcome; else a declared choice or
/// numeric outcome; else a reader or probe score; else a reasoning-style
/// feature; else marker density; else a surface measure such as word count.
/// Every summary says which rule chose the headline.
///
/// The mapping from outcome names to tiers is DATA:
/// `Server/steerlab_server/client/resources/headline-outcomes.json`, compiled
/// in here as `HeadlineOutcomeData` by
/// `scripts/ci/check-headline-outcomes.py`. The Python engine reads the file
/// itself and the results explorer reads its own generated copy; all three
/// are held to one set of cases,
/// `Tests/Fixtures/cross-engine/headline-outcome.json`.
///
/// Selection happens when a summary is SHOWN. Nothing here adds, removes, or
/// reorders the rows an analysis emits, and nothing here computes a
/// statistic.
///
/// Server twin: `Server/steerlab_server/experiment/headline_outcome.py`.
public enum HeadlineOutcome {

    /// The manifest key a study declares its primary outcome under.
    public static let manifestKey = "primaryOutcome"

    /// The one outcome that comes from the evaluation report.
    public static let judged = "judged"

    /// The files a judged outcome is read from, in a run directory.
    public static let evaluationReportFiles = [
        "judge-report.json", "coding-report.json",
    ]

    public enum Rule: String, Sendable, Equatable {
        case declared
        case defaultOrder
    }

    public enum Source: String, Sendable, Equatable {
        case analysisRows
        case evaluationReport
    }

    // MARK: - The mapping

    struct Mapping: Decodable {
        struct Tier: Decodable {
            let id: String
            let label: String
        }
        struct Outcome: Decodable {
            let name: String
            let match: String
            let tier: String
            let source: String
            let requires: String
            let emittedBy: [String]
            let plain: String
        }
        let schemaVersion: Int
        let manifestKey: String
        let rules: [String: String]
        let tiers: [Tier]
        let unlistedTier: String
        /// What each `requires` value needs, in plain words ("" for the two
        /// that need nothing a study could add).
        let requirements: [String: String]
        let outcomes: [Outcome]
    }

    /// The compiled mapping. The generator and `HeadlineOutcomeTests` both
    /// check that it decodes, so a failure here is a build that skipped its
    /// own checks, not a state a researcher can reach.
    static let mapping: Mapping = {
        do {
            return try JSONDecoder().decode(
                Mapping.self, from: Data(HeadlineOutcomeData.json.utf8))
        } catch {
            preconditionFailure(
                "the compiled headline-outcome mapping does not decode: \(error)")
        }
    }()

    /// The first mapping entry that names `name`, or nil. A prefix or suffix
    /// entry needs something left over: `rs_` alone is not a reasoning-style
    /// feature, and `MarkerDensity` alone names no concept.
    private static func entryIndex(_ name: String) -> Int? {
        mapping.outcomes.firstIndex { entry in
            switch entry.match {
            case "exact":
                return name == entry.name
            case "prefix":
                return name.hasPrefix(entry.name)
                    && name.utf8.count > entry.name.utf8.count
            case "suffix":
                return name.hasSuffix(entry.name)
                    && name.utf8.count > entry.name.utf8.count
            default:
                return false
            }
        }
    }

    /// The tier id of an outcome name. Names the mapping does not list fall
    /// in the last tier, after the surface measures.
    public static func tier(of name: String) -> String {
        entryIndex(name).map { mapping.outcomes[$0].tier } ?? mapping.unlistedTier
    }

    /// Plain words for an outcome name, or nil when the mapping does not
    /// list it. `{part}` in a pattern entry is the rest of the name.
    public static func plainPhrase(_ name: String) -> String? {
        guard let index = entryIndex(name) else { return nil }
        let entry = mapping.outcomes[index]
        let part: String
        switch entry.match {
        case "prefix": part = String(name.dropFirst(entry.name.count))
        case "suffix": part = String(name.dropLast(entry.name.count))
        default: part = ""
        }
        return entry.plain.replacingOccurrences(of: "{part}", with: part)
    }

    /// Default-order comparison: tier, then entry order, then the name
    /// itself by code point, so three codebases agree on every tie.
    private static func precedes(_ left: String, _ right: String) -> Bool {
        func key(_ name: String) -> (Int, Int) {
            let tierIDs = mapping.tiers.map(\.id)
            let index = entryIndex(name)
            return (
                tierIDs.firstIndex(of: tier(of: name)) ?? tierIDs.count,
                index ?? mapping.outcomes.count
            )
        }
        let (leftKey, rightKey) = (key(left), key(right))
        if leftKey != rightKey { return leftKey < rightKey }
        return left.unicodeScalars.map(\.value)
            .lexicographicallyPrecedes(right.unicodeScalars.map(\.value))
    }

    // MARK: - Selection

    /// What leads a summary, and why.
    ///
    /// `outcome` is nil when the run has nothing to lead with.
    /// `declaredAbsent` is true when the study declared a primary outcome
    /// this run does not have, so the summary fell back; `chosenBy` always
    /// says so in words.
    public struct Selection: Sendable, Equatable {
        public var outcome: String?
        public var tier: String?
        public var rule: Rule?
        public var source: Source?
        public var declaredOutcome: String?
        public var declaredAbsent: Bool
        public var chosenBy: String

        /// Nothing to lead with, and nothing declared.
        public static let none = HeadlineOutcome.select(
            declared: nil, analysisOutcomes: [])

        /// The outcome in plain words with the engine's name after it, e.g.
        /// "the target-choice rate (choiceRate)".
        public var outcomePhrase: String? {
            outcome.map(EffectNarrative.metricPhrase)
        }

        /// One line for a header: what leads, and how it was chosen.
        public var summaryLine: String {
            guard let phrase = outcomePhrase else { return chosenBy }
            return "\(phrase), \(chosenBy)"
        }

        /// The block an envelope carries. Keys match the server twin's.
        public var payload: [String: JSONValue] {
            var block: [String: JSONValue] = [
                "outcome": outcome.map { .string($0) } ?? .null,
                "tier": tier.map { .string($0) } ?? .null,
                "rule": rule.map { .string($0.rawValue) } ?? .null,
                "source": source.map { .string($0.rawValue) } ?? .null,
                "chosenBy": .string(chosenBy),
                "declaredAbsent": .bool(declaredAbsent),
            ]
            if let declaredOutcome {
                block["declaredOutcome"] = .string(declaredOutcome)
            }
            if let outcome, let plain = HeadlineOutcome.plainPhrase(outcome) {
                block["plain"] = .string(plain)
            }
            return block
        }
    }

    private static func clean(_ declared: String?) -> String? {
        guard let trimmed = declared?.trimmingCharacters(
            in: .whitespacesAndNewlines), !trimmed.isEmpty
        else { return nil }
        return trimmed
    }

    private static func unique(_ names: [String]) -> [String] {
        var seen = Set<String>()
        return names.filter { !$0.isEmpty && seen.insert($0).inserted }
    }

    /// Choose the headline from what a run HAS.
    ///
    /// `analysisOutcomes` are the outcome names of the pooled effect rows;
    /// `evaluationOutcomes` are the names an evaluation report supplies
    /// (today only `judged`). A judged outcome is never an analysis row,
    /// which is why the two arrive separately.
    public static func select(
        declared: String?, analysisOutcomes: [String],
        evaluationOutcomes: [String] = []
    ) -> Selection {
        let declared = clean(declared)
        let evaluation = unique(evaluationOutcomes)
        let available =
            evaluation + unique(analysisOutcomes).filter { !evaluation.contains($0) }
        func source(of name: String) -> Source {
            evaluation.contains(name) ? .evaluationReport : .analysisRows
        }
        if let declared, available.contains(declared) {
            return Selection(
                outcome: declared, tier: tier(of: declared), rule: .declared,
                source: source(of: declared), declaredOutcome: declared,
                declaredAbsent: false,
                chosenBy: mapping.rules[Rule.declared.rawValue] ?? "")
        }
        let absent =
            declared.map {
                "; the declared primary outcome '\($0)' is not in this run"
            } ?? ""
        guard let first = available.min(by: precedes) else {
            return Selection(
                outcome: nil, tier: nil, rule: nil, source: nil,
                declaredOutcome: declared, declaredAbsent: declared != nil,
                chosenBy: "no outcome to lead with" + absent)
        }
        return Selection(
            outcome: first, tier: tier(of: first), rule: .defaultOrder,
            source: source(of: first), declaredOutcome: declared,
            declaredAbsent: declared != nil,
            chosenBy: (mapping.rules[Rule.defaultOrder.rawValue] ?? "") + absent)
    }

    /// The measure a chart of `metrics` should open on, and how it was
    /// chosen: the run's headline when it is one of the chart's measures.
    /// A judged headline lives in the evaluation report, not in the effect
    /// rows a chart draws, so the chart then opens on the first of the
    /// measures it does have, by the same default order.
    public static func chartLead(
        _ headline: Selection, among metrics: [String]
    ) -> Selection {
        if let outcome = headline.outcome, metrics.contains(outcome) {
            return headline
        }
        return select(declared: nil, analysisOutcomes: metrics)
    }

    // MARK: - What a study can produce

    /// The outcomes a study's settings can produce, in default order.
    ///
    /// `names` are complete outcome names. `patterns` are families whose
    /// members cannot be listed from the settings alone (reasoning-style
    /// features live in the pinned taxonomy file): any name of that shape is
    /// accepted.
    public struct Producible: Sendable, Equatable {
        public var names: [String]
        public var patterns: [String]

        /// One retypeable list: `a | b | rs_<name>`.
        public var choices: String {
            (names + patterns).joined(separator: " | ")
        }

        public var payload: JSONValue {
            .object([
                "names": .array(names.map { .string($0) }),
                "patterns": .array(patterns.map { .string($0) }),
            ])
        }
    }

    private static func requirementMet(
        _ requirement: String, _ manifest: ExperimentManifest
    ) -> Bool {
        let instruments = manifest.outcomeInstruments ?? []
        let declaredParser = !(manifest.numericParser ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        switch requirement {
        case "always":
            return true
        case "never":
            return false
        case "judging":
            // The store's own resolution: an explicit evaluation block
            // decides; without one, a pinned judge plus a pinned rubric
            // file is a judging declaration.
            return ExperimentStore.effectiveEvaluation(manifest)?.spec.kind
                == .pairedJudge
        case "declaredNumericParser":
            return declaredParser
        case "numericEndpoint":
            return declaredParser || manifest.usesImplicitCaseFamilyEndpoint
        case "readers":
            return instruments.contains("repeReaderScore")
                && !(manifest.readerRefs ?? []).isEmpty
        case "reasoningStyleTaxonomy":
            return !(manifest.reasoningStyleTaxonomyPath ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        case "concepts":
            return !manifest.concepts.isEmpty
        default:
            if requirement.hasPrefix("instrument:") {
                return instruments.contains(
                    String(requirement.dropFirst("instrument:".count)))
            }
            return false
        }
    }

    private static func pattern(_ entry: Mapping.Outcome) -> String {
        entry.match == "prefix" ? entry.name + "<name>" : "<name>" + entry.name
    }

    /// Read from the manifest's own declarations; no file is opened.
    public static func producible(_ manifest: ExperimentManifest) -> Producible {
        var names: [String] = []
        var patterns: [String] = []
        func add(_ name: String) {
            if !names.contains(name) { names.append(name) }
        }
        for entry in mapping.outcomes
        where requirementMet(entry.requires, manifest) {
            if entry.match == "exact" {
                names.append(entry.name)
            } else if entry.requires == "readers" {
                for ref in manifest.readerRefs ?? [] where !ref.concept.isEmpty {
                    add(entry.name + ref.concept)
                }
            } else if entry.requires == "concepts" {
                for concept in manifest.concepts where !concept.name.isEmpty {
                    add(concept.name + entry.name)
                }
            } else {
                patterns.append(pattern(entry))
            }
        }
        return Producible(names: names, patterns: patterns)
    }

    /// Whether `outcome` is one of the study's producible outcomes.
    public static func canProduce(
        _ manifest: ExperimentManifest, outcome: String
    ) -> Bool {
        let listing = producible(manifest)
        if listing.names.contains(outcome) { return true }
        guard let index = entryIndex(outcome) else { return false }
        let entry = mapping.outcomes[index]
        return entry.match != "exact" && listing.patterns.contains(pattern(entry))
    }

    /// Why a study's settings cannot produce `outcome`, in plain words: what
    /// the outcome needs, or that nothing reports it. For the refusal a
    /// declaration gives; call it only for an outcome `canProduce` declined.
    public static func cannotProduceReason(_ outcome: String) -> String {
        guard let index = entryIndex(outcome) else {
            return "it is not an outcome SteerLab reports"
        }
        let needs = mapping.requirements[mapping.outcomes[index].requires] ?? ""
        return needs.isEmpty
            ? "no engine reports it as a paired effect" : "it needs " + needs
    }

    /// The primary-outcome line of the generated settings summary. Server
    /// twin: `headline_outcome.settings_summary_line` (same words).
    public static func settingsSummaryLine(_ manifest: ExperimentManifest) -> String {
        guard let declared = clean(manifest.primaryOutcome) else {
            return "- **Primary outcome:** not declared; summaries lead with "
                + "the default order"
        }
        return "- **Primary outcome:** \(declared)"
            + (plainPhrase(declared).map { " (\($0))" } ?? "")
            + ", declared by the researcher"
    }

    // MARK: - What a run has

    private static func jsonObject(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    private static func directoryName(_ raw: String) -> String {
        var trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        while trimmed.hasSuffix("/") { trimmed.removeLast() }
        return trimmed.split(separator: "/").last.map(String.init) ?? ""
    }

    /// The directory name of the run a derived run (analyze, evaluate)
    /// read, or this run's own name when it is a source run itself.
    private static func sourceRunName(_ runDirectory: URL) -> String {
        if let stamped = try? String(
            contentsOf: runDirectory.appending(component: "source-run.txt"),
            encoding: .utf8), !directoryName(stamped).isEmpty
        {
            return directoryName(stamped)
        }
        if let analysis = jsonObject(
            runDirectory.appending(component: "analysis.json")),
            let source = analysis["sourceRun"] as? String,
            !directoryName(source).isEmpty
        {
            return directoryName(source)
        }
        return runDirectory.standardizedFileURL.lastPathComponent
    }

    /// The evaluation report that judged the same source run as
    /// `runDirectory`, or nil.
    ///
    /// Looked for in the run directory itself first, then in the sibling
    /// directories whose name marks them as an evaluation and whose report
    /// names the same source run (the Mac stamps `sourceRunDirectory`, a
    /// path; the server stamps `sourceRun`, a name); the latest by directory
    /// name wins. Read-only.
    public static func evaluationReport(forRunAt runDirectory: URL) -> URL? {
        let fileManager = FileManager.default
        for name in evaluationReportFiles {
            let own = runDirectory.appending(component: name)
            if fileManager.fileExists(atPath: own.path) { return own }
        }
        let source = sourceRunName(runDirectory)
        let parent = runDirectory.standardizedFileURL.deletingLastPathComponent()
        guard
            let siblings = try? fileManager.contentsOfDirectory(
                atPath: parent.path)
        else { return nil }
        for sibling in siblings.sorted(by: >) where sibling.contains("-evaluate") {
            for name in evaluationReportFiles {
                let url = parent.appending(components: sibling, name)
                guard let report = jsonObject(url) else { continue }
                let stamped =
                    (report["sourceRun"] as? String)
                    ?? (report["sourceRunDirectory"] as? String) ?? ""
                if directoryName(stamped) == source { return url }
            }
        }
        return nil
    }

    /// The primary outcome a run's own manifest snapshot declares, read
    /// tolerantly: a snapshot this engine's strict decoder cannot read still
    /// says which outcome its study declared.
    public static func declaredOutcome(inSnapshot data: Data?) -> String? {
        guard
            let data,
            let object = (try? JSONSerialization.jsonObject(with: data))
                as? [String: Any]
        else { return nil }
        return clean(object[manifestKey] as? String)
    }

    /// The headline of a run directory: the outcome names given (its pooled
    /// effect rows), the evaluation report for the same source run when
    /// there is one, and the primary outcome declared in the run's own
    /// manifest snapshot.
    public static func forRun(
        at runDirectory: URL, analysisOutcomes: [String]
    ) -> Selection {
        select(
            declared: declaredOutcome(
                inSnapshot: try? Data(
                    contentsOf: runDirectory.appending(component: "experiment.json"))),
            analysisOutcomes: analysisOutcomes,
            evaluationOutcomes: evaluationReport(forRunAt: runDirectory) == nil
                ? [] : [judged])
    }
}

// MARK: - The declaration

extension ExperimentStore {

    /// Declare (or clear) the outcome a study is about.
    ///
    /// The declared outcome leads every results summary of the study, and
    /// the summary says it was "declared by the researcher". It is written
    /// to the manifest key `primaryOutcome`, so it is frozen with the study
    /// and travels in every run's manifest snapshot. Declaring nothing is
    /// fine: the summary then leads by the default order and says so.
    ///
    /// An outcome the study's settings cannot produce is refused here, with
    /// the list of the ones they can — a declaration nobody could ever read
    /// back is a mistake worth catching while the study is still a draft.
    /// It is a typed MALFORMED invocation (64), like a parser the registry
    /// does not define: a value outside this study's vocabulary, not a gate
    /// declining a healthy request. Nothing is written when it refuses.
    /// Server twin: `experiment_store.set_primary_outcome` (same words).
    @discardableResult
    public static func setPrimaryOutcome(
        _ outcome: String?, experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            let trimmed = outcome?.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let value = trimmed, !value.isEmpty else {
                manifest.primaryOutcome = nil
                return
            }
            guard HeadlineOutcome.canProduce(manifest, outcome: value) else {
                throw ExperimentError.malformed(
                    "this study's settings cannot produce the outcome "
                        + "'\(value)': "
                        + HeadlineOutcome.cannotProduceReason(value)
                        + ". The outcomes they can produce: "
                        + HeadlineOutcome.producible(manifest).choices,
                    repair: primaryOutcomeRepair(manifest))
            }
            manifest.primaryOutcome = value
        }
    }

    /// The retype for a refused primary-outcome declaration: the verb, this
    /// study, and the outcomes its settings can produce.
    static func primaryOutcomeRepair(_ manifest: ExperimentManifest) -> String {
        "steerlab-cli experiment set-primary-outcome \(manifest.name) <"
            + HeadlineOutcome.producible(manifest).choices
            + ">  (\"\" clears the declaration)"
    }
}
