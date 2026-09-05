import Foundation
import SteeringKit

/// Serialized report products, with diagnostics retained as values until publication.
struct StudyAnalysisArtifacts {
    private(set) var files: [String: Data] = [:]
    var diagnostics: [StudyAnalysisDiagnostic] = []

    mutating func add<T: Encodable>(_ name: String, json: T) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        files[name] = try encoder.encode(json)
    }
    mutating func add(_ name: String, text: String) { files[name] = Data(text.utf8) }
    mutating func log(_ text: String) { diagnostics.append(StudyAnalysisDiagnostic(text: text)) }
}

/// Wire formats and human-readable summaries, independent of filesystem policy.
enum StudyAnalysisRendering {
    typealias AnalyzeReport = ExperimentTasks.AnalyzeReport
    typealias RescoreStyleReport = ExperimentTasks.RescoreStyleReport
    typealias EffectSizeEntry = ExperimentTasks.EffectSizeEntry

    static func analyze(input: StudyAnalysisInput, result: StudyAnalysisResult) throws
        -> StudyAnalysisArtifacts
    {
        let manifest = input.manifest
        let epoch = input.epoch
        let entries = result.entries
        let exclusionStamp = result.exclusions
        let choiceDeltas = result.choiceDeltas
        let marginReports = result.margins
        var artifacts = StudyAnalysisArtifacts(diagnostics: result.diagnostics)
        let report = AnalyzeReport(
            experiment: manifest.name,
            experimentHash: ExperimentStore.manifestHash(manifest),
            sourceRun: input.sourceRunName,
            sourceRunExperimentHash: input.sourceRunExperimentHash,
            epochUnverified: epoch.unverified ? true : nil,
            measurementDrift: epoch.measurementDrift,
            effectSizes: entries,
            exclusions: exclusionStamp)
        try artifacts.add("analysis.json", json: report)
        if let exclusionStamp {
            // The stamp file the server also writes — one artifact name to
            // look for on either engine.
            try artifacts.add("exclusions.json", json: exclusionStamp)
        }
        artifacts.add("effect-sizes.csv", text: effectSizesCSV(entries))
        // Per-item choice deltas (server twin: choice-deltas.csv +
        // choice-deltas.json). Absence over empty artifacts: a run with no
        // non-baseline choice readouts grows no table implying it had some.
        // When there ARE readouts the file is written even if every one of
        // them was skipped — the skip counts are the finding in that case.
        if !choiceDeltas.summary.conditions.isEmpty {
            artifacts.add("choice-deltas.csv", text: ChoiceDeltas.csv(choiceDeltas.rows))
            try artifacts.add("choice-deltas.json", json: choiceDeltas.summary)
            let flips = choiceDeltas.summary.conditions.values
                .reduce(0) { $0 + $1.flipped }
            artifacts.log(
                "choice deltas: \(choiceDeltas.rows.count) paired item(s) "
                    + "across \(choiceDeltas.summary.conditions.count) "
                    + "condition(s), \(flips) flip(s), "
                    + "\(choiceDeltas.summary.skippedNoBaseline) skipped "
                    + "(no baseline partner) → choice-deltas.csv")
            if choiceDeltas.summary.skippedNoTargetValue > 0 {
                artifacts.log(
                    "choice deltas: \(choiceDeltas.summary.skippedNoTargetValue) "
                        + "readout(s) skipped — no log-odds entry for the "
                        + "item's own target option")
            }
        }
        // D3: a large joint-logprob margin means the FLIP RATE has poor
        // sensitivity — an intervention can move the log-odds a long way
        // without flipping any item — while the log-odds itself keeps moving
        // continuously. Calling that "saturation" invites the wrong
        // conclusion; true numerical saturation is the separately counted
        // clamp incidence.
        if !marginReports.isEmpty {
            try artifacts.add("choice-margins.json", json: marginReports)
            for (condition, block) in marginReports.sorted(by: { $0.key < $1.key }) {
                artifacts.log("\(condition): \(block.interpretation ?? "")")
            }
        }
        let ordinalNote =
            result.ordinalCount == 0
            ? "" : " + \(result.ordinalCount) ordinal readouts"
        let stratifiedCount = entries.count - result.pooledCount
        artifacts.log(
            "analyzed \(result.sampledCount) generations\(ordinalNote) from "
                + "\(input.sourceRunName): "
                + "\(result.pooledCount) effect-size "
                + "entr\(result.pooledCount == 1 ? "y" : "ies")"
                + (stratifiedCount > 0 ? " + \(stratifiedCount) stratified" : ""))
        return artifacts
    }

    static func rescoreStyle(input: StudyAnalysisInput, result: StudyStyleResult) throws
        -> StudyAnalysisArtifacts
    {
        guard let style = input.style else {
            throw ExperimentError(reason: "rescore-style requires a pinned taxonomy")
        }
        let manifest = input.manifest
        let epoch = input.epoch
        let rows = result.rows
        var artifacts = StudyAnalysisArtifacts()
        let header =
            ["condition", "seed", "promptIndex", "promptID"]
            + style.taxonomy.featureIDs.map { "rs_\($0)" }
        var lines = [header.joined(separator: ",")]
        for row in rows {
            let cells =
                [
                    csvEscape(row.condition),
                    String(row.seed),
                    String(row.promptIndex),
                    csvEscape(row.promptID),
                ] + style.taxonomy.featureIDs.map { String(row.reasoningStyle[$0] ?? 0) }
            lines.append(cells.joined(separator: ","))
        }
        artifacts.add("reasoning-style.csv", text: (lines.joined(separator: "\n") + "\n"))

        let report = RescoreStyleReport(
            experiment: manifest.name,
            experimentHash: ExperimentStore.manifestHash(manifest),
            sourceRun: input.sourceRunName,
            sourceRunExperimentHash: input.sourceRunExperimentHash,
            epochUnverified: epoch.unverified ? true : nil,
            measurementDrift: epoch.measurementDrift,
            taxonomy: style.taxonomy.name,
            taxonomyHash: style.hash,
            taxonomyFile: style.path,
            diagnosticOnly: true,
            conditions: result.conditions)
        try artifacts.add("reasoning-style.json", json: report)
        artifacts.log(
            "rescored \(rows.count) generations from \(input.sourceRunName): "
                + "\(style.taxonomy.featureIDs.count) feature(s) × "
                + "\(result.conditions.count) condition(s)")
        return artifacts
    }

    static func effectSizesCSV(_ entries: [EffectSizeEntry]) -> String {
        var lines = [
            "condition,metric,n,meanDiff,ciLower,ciUpper,wilcoxonW,wilcoxonP,"
                + "adjustedP,correction,stratifyBy,stratum,unit,estimand,"
                + "inference"
        ]
        for entry in entries {
            lines.append(
                [
                    csvEscape(entry.condition),
                    csvEscape(entry.metric),
                    String(entry.n),
                    String(entry.meanDiff),
                    String(entry.ciLower),
                    String(entry.ciUpper),
                    entry.wilcoxonW.map { String($0) } ?? "",
                    entry.wilcoxonP.map { String($0) } ?? "",
                    entry.adjustedP.map { String($0) } ?? "",
                    entry.correction.map(csvEscape) ?? "",
                    csvEscape(entry.stratifyBy ?? "pooled"),
                    entry.stratum.map(csvEscape) ?? "",
                    entry.unit.map(csvEscape) ?? "",
                    entry.estimand.map(csvEscape) ?? "",
                    entry.inference.map(csvEscape) ?? "",
                ].joined(separator: ","))
        }
        return lines.joined(separator: "\n") + "\n"
    }

    static func csvEscape(_ value: String) -> String {
        guard value.contains(",") || value.contains("\"") || value.contains("\n") else {
            return value
        }
        return "\"" + value.replacingOccurrences(of: "\"", with: "\"\"") + "\""
    }
}
