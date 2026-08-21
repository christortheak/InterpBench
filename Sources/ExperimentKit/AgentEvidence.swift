import Foundation

/// Pure, NON-BLOCKING evidence notes for one saved agent (variant artifact),
/// plus the latest-robustness lookup that feeds them.
///
/// These are honesty chips in the freeze-advisory tradition
/// (`ExperimentStore.freezeAdvisories`): they surface provenance and evidence
/// gaps at the moment a researcher looks at an agent — in the Agent Library
/// and under attached study conditions — and they never gate anything.
/// Nothing here prevents attach, freeze, or run.
///
/// Views render notes verbatim; every rule is plain data in, plain data out,
/// unit-tested without a model or a live workspace.
public enum AgentEvidence {

    // MARK: - Notes

    public enum Severity: String, Sendable, Equatable {
        /// Worth knowing; expected during exploration.
        case informational
        /// Worth pausing on before treating a result as evidence.
        case caution
    }

    public struct Note: Identifiable, Sendable, Equatable, Hashable {
        /// Short chip label ("Hand-created", "Manual override", …). Each
        /// rule fires at most once per artifact, so the label is a stable id.
        public var label: String
        /// One-line explanation in study vocabulary.
        public var detail: String
        public var severity: Severity

        public var id: String { label }

        /// Caption-ready one-liner for compact UI rows.
        public var displayLine: String { "\(label) — \(detail)" }

        public init(label: String, detail: String, severity: Severity) {
            self.label = label
            self.detail = detail
            self.severity = severity
        }
    }

    /// Evidence notes for one agent, in stable display order: provenance
    /// (hand-created / override), selection criterion, substrate, robustness.
    ///
    /// - `currentSubstrate`: the engine of the ACTIVE workspace
    ///   (`RepEReader.substrate` locally, `WorkspaceScoping.serverSubstrate`
    ///   for a server workspace); nil = unknown, in which case no substrate
    ///   claim is made at all (honest, not optimistic).
    /// - `latestRobustness`: the newest robustness report found for this
    ///   agent (see `latestRobustness(in:variantName:artifactHash:)`); nil
    ///   means none was found and the no-robustness note fires.
    ///
    /// Substrate rule: the artifact itself carries no substrate stamp — only
    /// the promotion birth certificate records the engine that minted it
    /// (`Promotion.substrate`). The mismatch note therefore fires only for
    /// promoted agents; a hand-created agent already carries the
    /// hand-created note and makes no substrate claim.
    ///
    /// Criterion rule: the birth certificate embeds the RESOLVED criterion
    /// verbatim (never hashed). A promotion with no criterion block predates
    /// criterion stamping, and the historical selection rule WAS marker
    /// density (`SweepSelectionRule.resolve`), so an absent objective reads
    /// as markerDensity here too.
    public static func notes(
        for artifact: ModelVariantArtifact,
        currentSubstrate: String?,
        latestRobustness: VariantRobustnessReport?
    ) -> [Note] {
        var notes: [Note] = []
        if let promotion = artifact.promotion {
            if promotion.promotedBy == "manualOverride" {
                var detail = "promoted by manual override — the declared selection was bypassed"
                if let reason = promotion.overrideReason {
                    detail += " — reason: \(reason)"
                } else {
                    detail += "; no override reason recorded"
                }
                notes.append(
                    Note(label: "Manual override", detail: detail, severity: .caution))
            }
            let metric = promotion.criterion?.objective?.metric ?? "markerDensity"
            if metric == "markerDensity" {
                notes.append(
                    Note(
                        label: "Marker-density criterion",
                        detail: "selected by marker density — surface-prose confound: "
                            + "a diagnostic metric, not an outcome instrument; "
                            + "a claim about a substantive outcome should select "
                            + "on judgeScore or logprobShift",
                        severity: .caution))
            }
            if let currentSubstrate, promotion.substrate != currentSubstrate {
                notes.append(
                    Note(
                        label: "Substrate mismatch",
                        detail: "promoted on '\(promotion.substrate)' but this workspace "
                            + "runs '\(currentSubstrate)' — vectors must be re-extracted "
                            + "and re-validated on the substrate a study runs on",
                        severity: .caution))
            }
        } else {
            notes.append(
                Note(
                    label: "Hand-created",
                    detail: "no sweep-selection provenance — fine for exploration; "
                        + "evidence-grade studies should compare sweep-promoted agents",
                    severity: .informational))
        }
        if latestRobustness == nil {
            notes.append(
                Note(
                    label: "No robustness result",
                    detail: "no robustness report for this agent in this workspace's "
                        + "runs — run a robustness check before evidence use",
                    severity: .informational))
        }
        return notes
    }

    /// The attach-site note for a variant condition whose referenced artifact
    /// no longer resolves in the workspace's agent library. Still
    /// non-blocking: the manifest snapshots the full artifact, so the study
    /// runs from its pinned copy regardless.
    public static let artifactNotFoundNote = Note(
        label: "Artifact not found",
        detail: "this agent's artifact was not found in this workspace's library — "
            + "the study still runs from the manifest's pinned snapshot",
        severity: .caution)

    /// Provenance one-liner for an attached promoted agent:
    /// "promoted from optimization run <sweepRun> · criterion <metric>".
    /// Falls back to the experiment name when the birth certificate predates
    /// the sweep-run stamp; the criterion metric is read verbatim from the
    /// stamped criterion (never hashed, never re-derived).
    public static func provenanceLine(
        for promotion: ModelVariantArtifact.Promotion
    ) -> String {
        var line: String
        if let run = promotion.sweepRun {
            line = "promoted from optimization run \(run)"
        } else {
            line = "promoted from '\(promotion.experiment)'"
        }
        if let metric = promotion.criterion?.objective?.metric {
            line += " · criterion \(metric)"
        }
        if promotion.promotedBy == "manualOverride" {
            line += " · manual override"
        }
        return line
    }

    // MARK: - Latest-robustness lookup

    /// One robustness report plus the immutable run directory it lives in.
    public struct RobustnessEvidence: Sendable {
        public var report: VariantRobustnessReport
        public var runDirectory: URL

        public init(report: VariantRobustnessReport, runDirectory: URL) {
            self.report = report
            self.runDirectory = runDirectory
        }
    }

    /// All decodable robustness reports in the workspace's runs/ tree,
    /// newest first. One directory listing plus one small-JSON decode per
    /// robustness run — cheap enough for a panel refresh, not meant for
    /// per-frame calls. Uses the same workspace resolution as the rest of
    /// ExperimentKit (`ExperimentStore.runsDirectory` → `VectorCatalog.projectRoot`,
    /// the RunBrowser/SweepRunCatalog mechanism). Directories without a
    /// report, and reports that fail to decode (pre-schema legacy runs), are
    /// skipped silently — absence of evidence is reported by the caller as
    /// the no-robustness note, never invented here.
    public static func scanRobustnessReports(
        runsDirectory: URL = ExperimentStore.runsDirectory
    ) -> [RobustnessEvidence] {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: runsDirectory,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles])
        else { return [] }
        let decoder = JSONDecoder()
        var results: [RobustnessEvidence] = []
        for url in entries {
            guard
                (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
            else { continue }
            let reportURL = url.appending(component: VariantRobustness.reportFileName)
            guard
                fm.fileExists(atPath: reportURL.path),
                let data = try? Data(contentsOf: reportURL),
                let report = try? decoder.decode(VariantRobustnessReport.self, from: data)
            else { continue }
            results.append(RobustnessEvidence(report: report, runDirectory: url))
        }
        return results.sorted { recencyKey($0) > recencyKey($1) }
    }

    /// The newest robustness evidence for one agent among already-scanned
    /// reports (scan once per refresh, match per row — pure and cheap).
    ///
    /// Matching: reports identify their variant by NAME
    /// (`report.variantName`) and, when the report was written by a current
    /// engine, by artifact-bytes hash (`report.variantArtifactHash`, the
    /// `ModelVariantStore.hash` convention). When both the caller and a
    /// report carry a hash, hash matches are preferred — they survive
    /// renames and distinguish edited artifacts. With no hash match (legacy
    /// reports, or an artifact edited since its check), the newest
    /// name-match is returned: a name match alone cannot prove the report
    /// tested these exact bytes.
    public static func latestRobustness(
        in evidence: [RobustnessEvidence],
        variantName: String,
        artifactHash: String?
    ) -> RobustnessEvidence? {
        let named = evidence.filter { $0.report.variantName == variantName }
        if let artifactHash {
            let hashMatched = named.filter {
                $0.report.variantArtifactHash == artifactHash
            }
            if let newest = hashMatched.max(by: { recencyKey($0) < recencyKey($1) }) {
                return newest
            }
        }
        return named.max(by: { recencyKey($0) < recencyKey($1) })
    }

    /// Convenience: scan + match in one call for one-off lookups.
    public static func latestRobustness(
        variantName: String,
        artifactHash: String?,
        runsDirectory: URL = ExperimentStore.runsDirectory
    ) -> RobustnessEvidence? {
        latestRobustness(
            in: scanRobustnessReports(runsDirectory: runsDirectory),
            variantName: variantName,
            artifactHash: artifactHash)
    }

    /// Caption one-liner for an agent row:
    /// "robustness: battery 0.92 vs 0.95 baseline · 2026-07-08 · 1 warning".
    public static func robustnessSummaryLine(
        _ report: VariantRobustnessReport
    ) -> String {
        let battery = String(format: "%.2f", report.variantBatteryAccuracy)
        let baseline = String(format: "%.2f", report.baselineBatteryAccuracy)
        var parts = ["robustness: battery \(battery) vs \(baseline) baseline"]
        let day = String(report.generatedAt.prefix(10))
        if !day.isEmpty { parts.append(day) }
        if !report.warnings.isEmpty {
            let count = report.warnings.count
            parts.append("\(count) warning\(count == 1 ? "" : "s")")
        }
        return parts.joined(separator: " · ")
    }

    /// Recency ordering: the report's own ISO-8601 `generatedAt` stamp
    /// (lexicographic == chronological), with the timestamp-prefixed run
    /// directory name as the tiebreak.
    private static func recencyKey(_ evidence: RobustnessEvidence) -> String {
        evidence.report.generatedAt + "|" + evidence.runDirectory.lastPathComponent
    }
}
