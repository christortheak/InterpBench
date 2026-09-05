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

    func loadAnalysis(
        manifest: ExperimentManifest, allowUnverifiedEpoch: Bool
    ) throws -> StudyAnalysisInput {
        let experimentName = manifest.name
        guard let sourceRun = newestCompletedRun(experimentName: experimentName) else {
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
        // statistics only (pairwise deletion falls out of the (seed,
        // promptID) baseline join), never from generations.jsonl, and the
        // stamp lands in analysis.json + exclusions.json. Scope is
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
        return StudyAnalysisInput(
            manifest: manifest, sourceRunName: sourceRun.lastPathComponent,
            sourceRunExperimentHash: RunEpoch.stampedExperimentHash(sourceRun),
            epoch: epoch, generations: text, style: style,
            exclusionChecks: exclusionChecks, declaredTargets: declaredTargets)
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
            sourceRun =
                runDirectoryName.hasPrefix("/")
                ? URL(filePath: runDirectoryName)
                : runsDirectory.appending(path: runDirectoryName)
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
