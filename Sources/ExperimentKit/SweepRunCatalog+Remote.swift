import Foundation

/// Remote (server-workspace) discovery for the Optimizations surface.
///
/// Both engines name sweep run directories identically
/// (`<stamp>-exp-<experiment>-sweep[-N]` — Swift `ExperimentTasks` and the
/// server's `make_unique_run_directory` share the convention), so the SAME
/// `directoryNameMatches` rule applies to a server's `/api/runs` ids as to
/// local directory names. Parsing likewise goes through the same pure entry
/// points (`parseCSV` / `parseRecommendations`); only the byte transport
/// differs (`ClusterClient.runFile` instead of the local filesystem).
extension SweepRunCatalog {

    /// Newest server run record that is this experiment's sweep run: id
    /// matches the sweep-task naming AND the listing shows a `sweep.csv`
    /// (the same two gates local discovery applies). Server ids are
    /// timestamp-prefixed, so lexicographic order is creation order.
    public static func newestRemoteSweepRunRecord(
        experiment: String, in records: [RemoteRunRecord]
    ) -> RemoteRunRecord? {
        records
            .filter {
                directoryNameMatches($0.id, experiment: experiment)
                    && $0.files.contains("sweep.csv")
            }
            .sorted { $0.id < $1.id }
            .last
    }

    /// Assemble a `SweepRun` from remotely fetched file contents through the
    /// same parsers the local loader uses. `runPath` is the run directory's
    /// path ON THE SERVER's tree (display identity only — nothing local ever
    /// resolves it); `recommendationsData` is optional exactly like the local
    /// loader treats a missing `recommendations.json`.
    public static func remoteSweepRun(
        runPath: String,
        csvText: String,
        recommendationsData: Data?
    ) throws -> SweepRun {
        let rows = try parseCSV(csvText)
        var recommendations: [String: Recommendation] = [:]
        if let recommendationsData, !recommendationsData.isEmpty {
            recommendations = try parseRecommendations(recommendationsData)
        }
        return SweepRun(
            directory: URL(filePath: runPath),
            rows: rows,
            recommendations: recommendations)
    }
}
