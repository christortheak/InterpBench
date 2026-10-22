import Foundation
import SteeringKit

/// Filesystem evidence for an already verified manifest. The compatibility
/// facade owns manifest admission; this reader never chooses a global root.
struct StudyAnalysisRepository {
    let workspaceRoot: URL
    let promptRoot: URL
    var runsDirectory: URL { workspaceRoot.appending(component: "runs") }
    private var prompts: StudyPromptRepository { StudyPromptRepository(projectRoot: promptRoot) }

    func newestCompletedRun(experimentName: String) -> URL? {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: runsDirectory, includingPropertiesForKeys: nil)
        else { return nil }
        for entry in entries.sorted(by: { $0.lastPathComponent > $1.lastPathComponent }) {
            guard
                entry.lastPathComponent.range(
                    of: "-exp-\(experimentName)-run(-\\d+)?$",
                    options: .regularExpression) != nil,
                fm.fileExists(
                    atPath: entry.appending(component: "generations.jsonl").path),
                fm.fileExists(atPath: entry.appending(component: "report.json").path),
                let data = try? Data(
                    contentsOf: entry.appending(component: "experiment.json")),
                let snapshot = try? JSONDecoder().decode(
                    ExperimentManifest.self, from: data),
                snapshot.name == experimentName
            else { continue }
            return entry
        }
        return nil
    }

    /// The directory a run reference names: an absolute path, a path from
    /// the workspace (`runs/<dir>`, the form the results commands print), or
    /// a directory name under `runs/`. Only resolved, never checked: the
    /// caller decides what the run must hold.
    func runDirectory(named reference: String) -> URL {
        if reference.hasPrefix("/") { return URL(filePath: reference) }
        let fromWorkspace = workspaceRoot.appending(path: reference)
        var isDirectory: ObjCBool = false
        if reference.hasPrefix("runs/"),
            FileManager.default.fileExists(atPath: fromWorkspace.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        {
            return fromWorkspace
        }
        return runsDirectory.appending(path: reference)
    }

    /// The completed run of `experimentName` that `reference` names
    /// (`runDirectory(named:)`), or a refusal saying why it is not one. An
    /// analysis names the evidence it reads, so a reference is never
    /// widened to another run: a directory outside `runs/`, an unfinished
    /// run, or another study's run is refused.
    func completedRun(named reference: String, experimentName: String) throws -> URL {
        let fm = FileManager.default
        let directory = runDirectory(named: reference).standardizedFileURL
        let listRuns = "steerlab-cli experiment analyze \(experimentName)  (without --run, "
            + "the newest completed run; run directories are listed under runs/)"
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: directory.path, isDirectory: &isDirectory),
            isDirectory.boolValue,
            directory.resolvingSymlinksInPath().deletingLastPathComponent().path
                == runsDirectory.resolvingSymlinksInPath().standardizedFileURL.path
        else {
            throw ExperimentError.refusing(
                .missingPrerequisite,
                "'\(reference)' is not a run directory in this workspace's runs/ folder",
                repair: listRuns)
        }
        let missing = ["generations.jsonl", "report.json"].filter {
            !fm.fileExists(atPath: directory.appending(component: $0).path)
        }
        guard missing.isEmpty else {
            throw ExperimentError.refusing(
                .missingPrerequisite,
                "run '\(directory.lastPathComponent)' is not a completed study run: it has no "
                    + missing.joined(separator: " and ")
                    + " (a run writes report.json when it completes)",
                repair: listRuns)
        }
        let owner =
            (try? Data(contentsOf: directory.appending(component: "experiment.json")))
            .flatMap { try? JSONDecoder().decode(ExperimentManifest.self, from: $0) }?.name
        guard owner == experimentName else {
            throw ExperimentError.refusing(
                .missingPrerequisite,
                "run '\(directory.lastPathComponent)' is "
                    + (owner.map { "a run of the study '\($0)'" } ?? "not stamped with its study")
                    + ", not of '\(experimentName)'",
                repair: owner.map {
                    "steerlab-cli experiment analyze \($0) --run runs/\(directory.lastPathComponent)"
                } ?? listRuns)
        }
        return directory
    }

    static func verifyRunEpoch(
        verb: String,
        runDirectory: URL,
        manifest: ExperimentManifest,
        allowUnverified: Bool = false
    ) throws -> RunEpoch.Check {
        let check = RunEpoch.check(
            verb: verb, experiment: manifest.name,
            liveHash: ExperimentStore.manifestHash(manifest),
            runDirectory: runDirectory, liveManifest: manifest,
            allowUnverified: allowUnverified,
            tolerateMeasurementDrift: true,
            // The whole family reads the source run's RECORDS, so a run from
            // the other engine is refused here rather than measured into an
            // empty result that exits 0 (WP0 dry run #2, P0).
            refuseForeignSubstrate: true)
        if let refusal = check.refusal {
            // A foreign run's repair is not "re-run" and is certainly not
            // `--allow-unverified-epoch` (which forgives a missing stamp, and
            // would leave this run just as unreadable) — it is the same verb
            // on the engine that wrote the records.
            let repair =
                RunEpoch.foreignSubstrate(runDirectory) != nil
                ? "steerlab-server experiment \(verb) \(manifest.name)  "
                    + "(on the engine that produced the run; the Mac reads "
                    + "its results, it does not re-measure them)"
                : "steerlab-cli experiment run \(manifest.name)  "
                    + "(a run of the CURRENT manifest), or read the older run "
                    + "under its own epoch with steerlab-cli experiment "
                    + "\(verb) \(manifest.name) --allow-unverified-epoch  "
                    + "(which only bypasses an UNSTAMPED run, never a "
                    + "mismatched one)"
            throw ExperimentError.refusing(.manifestEpoch, refusal, repair: repair)
        }
        // Tolerated is never silent (server twin: the `_log` warnings in
        // `tasks.evaluate`/`analyze`/`rescore_style`).
        if let drift = check.measurementDrift {
            print(
                "WARNING: '\(manifest.name)' drifted from source run "
                    + "'\(runDirectory.lastPathComponent)' in MEASUREMENT-side "
                    + "fields only (\(drift)) — the generations are "
                    + "unaffected; \(verb) proceeds under the LIVE settings "
                    + "and the output is stamped measurementDrift")
        }
        if check.unverified {
            print(
                "WARNING: source run '\(runDirectory.lastPathComponent)' "
                    + "carries no experiment-hash stamp — \(verb) under "
                    + "allowUnverifiedEpoch; the output is stamped "
                    + "epochUnverified")
        }
        return check
    }

    static func analysisSourceRecords(
        at runDirectory: URL
    ) -> (recordCount: Int?, conditions: Set<String>) {
        guard
            let text = try? String(
                contentsOf: runDirectory.appending(component: "generations.jsonl"),
                encoding: .utf8)
        else { return (nil, []) }
        struct ConditionOnly: Decodable { let condition: String }
        var conditions = Set<String>()
        var count = 0
        let decoder = JSONDecoder()
        for line in text.split(separator: "\n") {
            count += 1
            if let record = try? decoder.decode(
                ConditionOnly.self, from: Data(line.utf8))
            {
                conditions.insert(record.condition)
            }
        }
        return (count, conditions)
    }

    /// `sourceRunName` (`analyze --run`): the run to analyze, as
    /// `completedRun(named:experimentName:)` reads it; nil analyzes the
    /// newest completed run.
    func loadAnalysis(
        manifest: ExperimentManifest, allowUnverifiedEpoch: Bool, rereadChoices: Bool = false,
        sourceRunName: String? = nil
    ) throws -> StudyAnalysisInput {
        let experimentName = manifest.name
        let named = try sourceRunName.map {
            try completedRun(named: $0, experimentName: experimentName)
        }
        guard let sourceRun = named ?? newestCompletedRun(experimentName: experimentName) else {
            throw ExperimentError.refusing(
                .missingPrerequisite,
                "no completed study run found for '\(experimentName)' "
                    + "(need generations.jsonl + report.json under runs/)",
                repair: "steerlab-cli experiment run \(experimentName) && "
                    + "steerlab-cli experiment analyze \(experimentName)")
        }
        let epoch = try Self.verifyRunEpoch(
            verb: "analyze", runDirectory: sourceRun, manifest: manifest,
            allowUnverified: allowUnverifiedEpoch)

        // Reasoning-style values are derived, not stored: recompute them from
        // each record's output through the pinned (hash-checked) taxonomy so
        // rs_<featureID> joins the same paired effect-size machinery.
        let style = try ExperimentStore.loadPinnedReasoningStyle(manifest, root: workspaceRoot)

        // Declared exclusion rules join HERE — records drop from the paired
        // statistics only (pairwise deletion falls out of the per-item
        // baseline join: a cell keeps its surviving samples, and an item
        // pairs only while both arms still have one), never from
        // generations.jsonl, and the stamp lands in analysis.json +
        // exclusions.json. Scope is
        // allRecordTypes (the engine default): instrument readouts are
        // considered too — endpoint rules read endpoints the record itself
        // carries (e.g. ordinalPosition), and a cell whose every sampled
        // record failed its attention check drops its instrument readout
        // from the ordinal pairing with it. No rules declared = today's
        // behavior byte-for-byte. Server twin: `tasks.analyze`.
        let exclusionRules = manifest.exclusionRules ?? []
        let ruleProblems = ExclusionEngine.violations(exclusionRules)
        guard ruleProblems.isEmpty else {
            throw ExperimentError(reason: ruleProblems.joined(separator: "; "))
        }
        var exclusionChecks: [String: AttentionCheck] = [:]
        if ExclusionEngine.needsChecks(exclusionRules) {
            guard manifest.taskPromptsHash != nil else {
                // WP0 step 8: the deferred cross-engine-twinned message gets
                // its id on BOTH engines. The STRING is unchanged and stays
                // byte-identical to the server's `PIN_REQUIRED_MESSAGE`
                // (asserted on both sides); only the gate id and the runnable
                // repair are new.
                throw ExperimentError.refusing(
                    .missingPrerequisite, ExclusionEngine.pinRequiredMessage,
                    repair: ExclusionEngine.pinRequiredRepair)
            }
            exclusionChecks = StudyPromptParsing.attentionChecks(
                of: try prompts.load(for: manifest).prompts)
            guard !exclusionChecks.isEmpty else {
                throw ExperimentError(reason: ExclusionEngine.noChecksMessage)
            }
        }
        let text = try String(
            contentsOf: sourceRun.appending(component: "generations.jsonl"),
            encoding: .utf8)
        var declaredTargets: [String: Bool]? = nil
        if manifest.taskPromptsHash != nil,
            let loaded = try? prompts.load(for: manifest)
        {
            declaredTargets = Dictionary(
                loaded.prompts.map { ($0.id, $0.target?.isEmpty == false) },
                uniquingKeysWith: { first, _ in first })
        }
        var input = StudyAnalysisInput(
            manifest: manifest, sourceRunName: sourceRun.lastPathComponent,
            sourceRunExperimentHash: RunEpoch.stampedExperimentHash(sourceRun),
            epoch: epoch, generations: text, style: style,
            exclusionChecks: exclusionChecks, declaredTargets: declaredTargets,
            numericParser: numericParser(for: manifest))
        if rereadChoices {
            // Every stored choice read again with the current parser, against
            // the hash-verified task prompts the run names; refuses rather
            // than guess when they cannot be found. The run is not touched.
            let reread = try ChoiceReread.forAnalysis(
                runDirectory: sourceRun, generations: text,
                workspaceRoot: workspaceRoot, study: manifest.name)
            input.choiceRereads = reread.fresh
            input.choiceRereadStamp = reread.stamp
        }
        return input
    }

    /// The declared numeric parser, as far as `analyze` needs it: its KIND,
    /// which decides whether the parsed numbers are also reported under the
    /// neutral `parsedValue…` names (any kind but `durationMonths`).
    ///
    /// Never a refusal. The analysis pairs the numbers the run recorded and
    /// re-parses nothing, so a registry that is missing, malformed, or
    /// changed since the study pinned it costs only those two names — and
    /// the analysis says so (`StudyAnalysisOutcomes
    /// .numericParserUnreadableReason`). The Python engine, which DOES
    /// re-parse unread responses under the registry's grammar, refuses in
    /// the same state.
    func numericParser(for manifest: ExperimentManifest) -> AnalysisNumericParser {
        let name =
            manifest.numericParser?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !name.isEmpty else { return .undeclared }
        let registry = ExperimentStore.resolveProjectPath(
            ParserRegistry.registryFile, root: workspaceRoot)
        do {
            let spec = try ParserRegistry.spec(named: name, at: registry)
            if let pinned = manifest.parserRegistryHash,
                let live = ParserRegistry.liveHash(at: registry), live != pinned
            {
                return .unreadable(
                    name: name,
                    reason: "the parser registry changed since the study "
                        + "pinned it (have \(live.prefix(12))…, pinned "
                        + "\(pinned.prefix(12))…)")
            }
            return .resolved(name: name, kind: spec.kind ?? "")
        } catch {
            return .unreadable(
                name: name,
                reason: (error as? ExperimentError)?.reason ?? "\(error)")
        }
    }

    func loadRescore(
        manifest: ExperimentManifest, runDirectoryName: String?,
        allowUnverifiedEpoch: Bool
    ) throws -> StudyAnalysisInput {
        let experimentName = manifest.name
        guard
            let style = try ExperimentStore.loadPinnedReasoningStyle(manifest, root: workspaceRoot)
        else {
            throw ExperimentError(
                reason: "experiment '\(experimentName)' pins no reasoning-style "
                    + "taxonomy — pin one first: steerlab-cli experiment "
                    + "set-style-taxonomy \(experimentName) "
                    + "prompts/taxonomies/<name>.json")
        }
        let sourceRun: URL
        if let runDirectoryName {
            sourceRun = runDirectory(named: runDirectoryName)
        } else if let newest = newestCompletedRun(experimentName: experimentName) {
            sourceRun = newest
        } else {
            throw ExperimentError.refusing(
                .missingPrerequisite,
                "no completed study run found for '\(experimentName)' "
                    + "(need generations.jsonl + report.json under runs/) — "
                    + "run it first, or pass --run",
                repair: "steerlab-cli experiment run \(experimentName) && "
                    + "steerlab-cli experiment rescore-style \(experimentName)")
        }
        let epoch = try Self.verifyRunEpoch(
            verb: "rescore-style", runDirectory: sourceRun, manifest: manifest,
            allowUnverified: allowUnverifiedEpoch)

        let text = try String(
            contentsOf: sourceRun.appending(component: "generations.jsonl"),
            encoding: .utf8)
        return StudyAnalysisInput(
            manifest: manifest, sourceRunName: sourceRun.lastPathComponent,
            sourceRunExperimentHash: RunEpoch.stampedExperimentHash(sourceRun),
            epoch: epoch, generations: text, style: style)
    }
}
