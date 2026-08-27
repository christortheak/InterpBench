import Foundation
import SteeringKit

/// A persisted steering vector discovered on disk (a `.safetensors` +
/// sidecar pair inside a run directory).
public struct VectorArtifact: Identifiable, Sendable {
    public let directory: URL
    /// Base file name shared by `<name>.safetensors` and `<name>.json`.
    public let name: String
    public let sidecar: SteeringVectorSidecar

    public var id: String { directory.appending(component: name).path }

    /// e.g. "french · french-qwen3 · paired-difference PCA · pooled@50 · 2026-06-10". Method
    /// and pooling appear so that vectors extracted from the same stimuli
    /// under different options stay distinguishable in the picker.
    public var label: String {
        var parts = [sidecar.concept, name]
        if let recipe = sidecar.recipeMethod {
            switch recipe {
            case VectorExtractionRecipe.Method.caaMeanDifference.rawValue:
                parts.append("CAA")
            case VectorExtractionRecipe.Method.pairedDifferencePCA.rawValue:
                parts.append("paired-difference PCA")
            case VectorExtractionRecipe.Method.emotionGrandMean.rawValue:
                parts.append("emotion-grand-mean")
            default:
                parts.append(recipe)
            }
        } else {
            switch sidecar.extractionMethod {
            case "lat": parts.append("paired-difference PCA")
            case "meanDifference": parts.append("mean-diff")
            case "repeReaderLAT": parts.append("RepE reader-derived")
            default: break  // pre-options artifact: mean difference implied
            }
        }
        if let position = sidecar.readingPosition, position != "last token" {
            parts.append(
                position.replacingOccurrences(of: "mean from token ", with: "pooled@"))
        }
        parts.append(String(sidecar.extractionDate.prefix(10)))
        return parts.joined(separator: " · ")
    }

    public var fixedSteeringLayer: Int? {
        guard sidecar.concept.hasPrefix("sae:") else { return nil }
        let parts = sidecar.concept.split(separator: ":")
        guard parts.count >= 4, parts[2].hasPrefix("L") else { return nil }
        return Int(parts[2].dropFirst())
    }
}

public struct ArtifactAuditFinding: Codable, Sendable, Identifiable {
    public enum Severity: String, Codable, Sendable {
        case info
        case warning
        case invalidForReportedRuns
    }

    public let id: String
    public let severity: Severity
    public let artifactPath: String
    public let issue: String
    public let recommendation: String

    public init(
        severity: Severity,
        artifactPath: String,
        issue: String,
        recommendation: String
    ) {
        self.id = "\(artifactPath)|\(issue)"
        self.severity = severity
        self.artifactPath = artifactPath
        self.issue = issue
        self.recommendation = recommendation
    }
}

public enum VectorCatalog {

    /// The WORKSPACE root — the data folder holding prompts/, experiments/,
    /// runs/. Runtime-resolved (see `WorkspaceRoot` for the exact precedence:
    /// STEERLAB_WORKSPACE env → programmatic override → persisted app choice
    /// → the legacy code-checkout fallback), so dev and test flows with
    /// nothing configured behave exactly as before. The app's working
    /// directory is unrelated to the workspace when launched from Xcode.
    public static var projectRoot: URL {
        WorkspaceRoot.current
    }

    /// The CODE repository root — a compatibility alias for
    /// `CodeResources.compiledCheckoutPath` (the one sanctioned derivation).
    /// Used by the workspace/checkout conflation guards, the legacy
    /// workspace-root fallback, and tests that read committed fixtures —
    /// never as a data root when a workspace is configured. New code
    /// resolving shipped resources should use the typed `CodeResources`
    /// accessors instead.
    public static var bundledSeedRoot: URL {
        CodeResources.compiledCheckoutPath
    }

    public enum PathError: Error, LocalizedError {
        case outsideProjectRoot(String)
        case runDirectoryNamesExhausted(slug: String, stamp: String, attempts: Int)
        public var errorDescription: String? {
            switch self {
            case .outsideProjectRoot(let path):
                return "path escapes the project root: \(path)"
            case .runDirectoryNamesExhausted(let slug, let stamp, let attempts):
                return """
                    could not create a unique run directory for slug "\(slug)": \
                    \(attempts) consecutive names at stamp \(stamp) were already taken
                    """
            }
        }
    }

    /// Resolve a caller-supplied relative path against the project root,
    /// refusing any result outside it (absolute paths, `..` traversal). Use
    /// this for API-writable files such as task-prompt sets: the web server is
    /// unauthenticated, so an unchecked `projectRoot.appending(path:)` would be
    /// an arbitrary-file read/write primitive (CLAUDE.md security posture).
    public static func projectFile(_ relative: String) throws -> URL {
        let root = projectRoot.standardizedFileURL
        let candidate = root.appending(path: relative).standardizedFileURL
        let rootPath = root.path
        guard candidate.path == rootPath || candidate.path.hasPrefix(rootPath + "/") else {
            throw PathError.outsideProjectRoot(relative)
        }
        return candidate
    }

    public static var runsDirectory: URL { runsDirectory(root: projectRoot) }

    public static var conceptsDirectory: URL { conceptsDirectory(root: projectRoot) }

    public static var emotionsDirectory: URL { emotionsDirectory(root: projectRoot) }

    public static var probesDirectory: URL { probesDirectory(root: projectRoot) }

    public static var batteriesDirectory: URL { batteriesDirectory(root: projectRoot) }

    public static var taskPromptsDirectory: URL { taskPromptsDirectory(root: projectRoot) }

    public static var devPromptsDirectory: URL { devPromptsDirectory(root: projectRoot) }

    /// Root-parameterized forms of the three per-concept dataset roots.
    ///
    /// Same convention as `NeutralCorpusStore.normCorpusURL(root:)`: the
    /// resolved workspace is the default everywhere, and an explicit root
    /// exists so a caller that already knows which tree it is working in —
    /// `DatasetCreationPlanner.plan(_:root:)` previewing a destination —
    /// resolves through THIS authority instead of re-deriving the layout.
    public static func conceptsDirectory(root: URL) -> URL {
        root.appending(components: "prompts", "concepts")
    }

    public static func emotionsDirectory(root: URL) -> URL {
        root.appending(components: "prompts", "emotions")
    }

    public static func probesDirectory(root: URL) -> URL {
        root.appending(components: "prompts", "probes")
    }

    /// The runs tree of a NAMED workspace. Same reason as the three roots
    /// above: the derived-artifact inventory scans a workspace it already
    /// knows (`DerivedArtifactInventory.scan(root:)`), and the stores it
    /// calls take a runs root, so the layout is resolved here once rather
    /// than re-derived at each call site.
    public static func runsDirectory(root: URL) -> URL {
        root.appending(component: "runs")
    }

    /// The three workspace prompt roots that had no named form before the
    /// Data workbench enumerated them. Each is the directory every existing
    /// reference already names as a relative path — `prompts/batteries/…`
    /// (`VariantRobustness.presets`, `SweepSpec.batteryFile`,
    /// `WorkspaceStore.seedManifest`), `prompts/tasks/…`
    /// (`DataTemplates.taskPromptsDestination`, `AgentContract`'s data
    /// contract for measured task prompts), `prompts/dev/…`
    /// (`SweepSpec.devPromptsFile`, `VariantRobustness`'s coherence prompts)
    /// — named ONCE here instead of scattered string literals, in the shape
    /// `JudgeRubricStore.relativeDirectory` already established.
    ///
    /// Naming the directory is not the same as owning what may live in it. A
    /// manifest pins a task-prompt set or a battery by workspace-relative
    /// PATH (`pinTaskPrompts`, `pinCapabilityBattery` accept any path, the
    /// engine's own default task-prompt file being under `prompts/dev/`), so
    /// a set filed elsewhere is legal. For TASK PROMPTS that gap is now
    /// closed by enumeration rather than disclosure: `TaskPromptsStore.list`
    /// unions these roots with every file an `experiments/` manifest actually
    /// pins, labelled with the study that pins it. Batteries still rely on
    /// the convention, and the inventory's battery rows say so.
    public static let batteriesRelativeDirectory = "prompts/batteries"
    public static let taskPromptsRelativeDirectory = "prompts/tasks"
    public static let devPromptsRelativeDirectory = "prompts/dev"

    public static func batteriesDirectory(root: URL) -> URL {
        root.appending(path: batteriesRelativeDirectory)
    }

    // MARK: Paired-pairs roots (prompts/repe, prompts/readers)

    /// The `pairs.jsonl` roots the two PAIRED recipes read, named here
    /// instead of assembled at each call site.
    ///
    /// Before the Data workbench's phase 4 these paths were built inline in
    /// five places inside `ConceptBuilder` (concept delete, the on-disk
    /// recipe probe, the local reader fit, the server reader fit, and the
    /// RepE-LAT vector build) plus the dataset inventory's own family loop —
    /// six independent spellings of one layout, and the RepE build
    /// additionally repeated the relative path as a string literal for its
    /// provenance stamp. Same reason `conceptsDirectory(root:)` exists: a
    /// layout change moves every reader and writer with it.
    ///
    /// The two families share a directory shape and a filename but NOT a row
    /// shape — `prompts/repe/` holds `StimulusSet.PairedStimulus` rows
    /// (`positive`/`negative`) and `prompts/readers/` holds `RepEReader.Pair`
    /// rows (`concept`/`positiveStimulus`/`negativeStimulus`/`templateID`).
    /// `loader` names which, so a caller reads a file with the loader its
    /// recipe actually uses.
    public enum PairedStimulusFamily: String, Sendable, CaseIterable, Identifiable, Codable {
        /// RepE/LAT vector extraction (`prompts/repe/<name>/pairs.jsonl`).
        case repe
        /// Fitted RepE reader instruments (`prompts/readers/<name>/pairs.jsonl`).
        case readers

        public var id: String { rawValue }

        /// The sub-family label the inventory shows. Deliberately the raw
        /// directory name — the researcher reads these rows against
        /// `prompts/<label>/`.
        public var label: String { rawValue }

        public var relativeDirectory: String { "prompts/\(rawValue)" }

        public var title: String {
            switch self {
            case .repe: "Paired-difference PCA pairs"
            case .readers: "Reader pairs"
            }
        }

        /// What reads it, and the row shape its loader requires.
        public var detail: String {
            switch self {
            case .repe:
                "content-matched pairs for the RepE/LAT vector recipe — "
                    + #"{"positive", "negative"} rows, read by "#
                    + "StimulusSet.loadPairs"
            case .readers:
                "the pinned dataset a fitted RepE reader is trained from — "
                    + #"{"concept", "positiveStimulus", "negativeStimulus", "#
                    + #""templateID"} rows, read by RepEReader.loadPairs"#
            }
        }
    }

    /// The one filename both paired families use.
    public static let pairedStimuliFileName = "pairs.jsonl"

    /// One paired family's ROOT (`prompts/repe`, `prompts/readers`).
    public static func pairedStimuliRoot(
        family: PairedStimulusFamily, root: URL = projectRoot
    ) -> URL {
        root.appending(path: family.relativeDirectory)
    }

    /// One concept's directory under a paired family
    /// (`prompts/<family>/<name>`).
    public static func pairedStimuliDirectory(
        family: PairedStimulusFamily, name: String, root: URL = projectRoot
    ) -> URL {
        pairedStimuliRoot(family: family, root: root).appending(component: name)
    }

    /// One concept's `pairs.jsonl` under a paired family.
    public static func pairedStimuliFile(
        family: PairedStimulusFamily, name: String, root: URL = projectRoot
    ) -> URL {
        pairedStimuliDirectory(family: family, name: name, root: root)
            .appending(component: pairedStimuliFileName)
    }

    /// The WORKSPACE-RELATIVE path of that file — what a sidecar stamps as
    /// its canonical dataset path. Derived from the same authority as the URL
    /// so the stamp cannot drift from the bytes it names.
    public static func pairedStimuliRelativePath(
        family: PairedStimulusFamily, name: String
    ) -> String {
        "\(family.relativeDirectory)/\(name)/\(pairedStimuliFileName)"
    }

    public static func taskPromptsDirectory(root: URL) -> URL {
        root.appending(path: taskPromptsRelativeDirectory)
    }

    public static func devPromptsDirectory(root: URL) -> URL {
        root.appending(path: devPromptsRelativeDirectory)
    }

    /// Bounded retry budget for `makeUniqueRunDirectory` (mirrors the server's
    /// `MAX_RUN_DIRECTORY_ATTEMPTS`). Far above any real same-millisecond
    /// fan-out; exhausting it means something is wrong, so we throw.
    public static let maxRunDirectoryAttempts = 500

    /// Creates a fresh run directory `<runs>/<stamp>-<slug>` that is
    /// guaranteed never to alias an existing one: fractional-second stamps
    /// plus a numeric suffix on collision. Run directories are immutable —
    /// two invocations in the same instant must not share one (CLAUDE.md ›
    /// Data & reproducibility: never overwrite or mutate a run).
    ///
    /// Creation itself is the exclusivity test. Probing with `fileExists` and
    /// *then* creating leaves a TOCTOU window: two processes resolving the
    /// same millisecond stamp (concurrent generation shards on a shared
    /// cluster filesystem) both see "free" and both create — and because
    /// `withIntermediateDirectories: true` does NOT fail on an existing
    /// directory, both would silently proceed to write into one run
    /// directory. `withIntermediateDirectories: false` is an atomic
    /// `mkdir(2)`: exactly one caller wins a name, the loser retries under the
    /// next suffix. (Same fix, same suffix convention, as the server's
    /// `make_unique_run_directory`.)
    public static func makeUniqueRunDirectory(
        slug: String, under runsRoot: URL = runsDirectory
    ) throws -> URL {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let stamp = formatter.string(from: Date())
            .replacingOccurrences(of: ":", with: "")
            .replacingOccurrences(of: ".", with: "")
        let fm = FileManager.default
        // The parent may legitimately be missing/nested; only the leaf carries
        // the exclusivity guarantee.
        try fm.createDirectory(at: runsRoot, withIntermediateDirectories: true)
        var url = runsRoot.appending(component: "\(stamp)-\(slug)")
        var counter = 1
        for _ in 0 ..< maxRunDirectoryAttempts {
            do {
                try fm.createDirectory(at: url, withIntermediateDirectories: false)
                return url
            } catch {
                // Only a name that is now taken is a collision; anything else
                // (permissions, read-only volume, ENOSPC) must surface.
                guard fm.fileExists(atPath: url.path) else { throw error }
                counter += 1
                url = runsRoot.appending(component: "\(stamp)-\(slug)-\(counter)")
            }
        }
        throw PathError.runDirectoryNamesExhausted(
            slug: slug, stamp: stamp, attempts: maxRunDirectoryAttempts)
    }

    /// The subdirectory names under one `prompts/<family>/` root, sorted.
    ///
    /// The ONE directory-walk primitive for the per-concept dataset layouts:
    /// `conceptNames()` unions it across the families a concept can appear
    /// in, and `DatasetInventory` walks each family separately. Adding a
    /// second scanner is how the two would drift.
    public static func datasetDirectoryNames(in root: URL) -> [String] {
        guard
            let entries = try? FileManager.default.contentsOfDirectory(
                at: root, includingPropertiesForKeys: [.isDirectoryKey])
        else { return [] }
        return entries
            .filter {
                (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
            }
            .map(\.lastPathComponent)
            .sorted()
    }

    /// Concept stimulus sets on disk (directory names under prompts/concepts).
    public static func conceptNames() -> [String] {
        let roots = [conceptsDirectory, emotionsDirectory, probesDirectory]
        return Set(roots.flatMap { datasetDirectoryNames(in: $0) }).sorted()
    }

    /// True when the concept's stimulus files have changed since this vector
    /// was extracted — the vector is still a valid artifact of its recorded
    /// hash, but it no longer represents the current stimulus set.
    public static func isStale(_ artifact: VectorArtifact) -> Bool {
        let directory = conceptsDirectory.appending(component: artifact.sidecar.concept)
        guard let set = try? StimulusSet(directory: directory) else { return false }
        return set.hash != artifact.sidecar.stimulusSetHash
    }

    /// All vectors under `runs/`, newest run first.
    public static func scan(runsDirectory: URL = runsDirectory) -> [VectorArtifact] {
        let fm = FileManager.default
        guard
            let runDirs = try? fm.contentsOfDirectory(
                at: runsDirectory, includingPropertiesForKeys: nil)
        else { return [] }

        var artifacts: [VectorArtifact] = []
        for runDir in runDirs.sorted(by: { $0.lastPathComponent > $1.lastPathComponent }) {
            guard
                let files = try? fm.contentsOfDirectory(
                    at: runDir, includingPropertiesForKeys: nil)
            else { continue }
            for file in files where file.pathExtension == "safetensors" {
                let name = file.deletingPathExtension().lastPathComponent
                let sidecarURL = runDir.appending(component: "\(name).json")
                guard
                    let data = try? Data(contentsOf: sidecarURL),
                    let sidecar = try? JSONDecoder().decode(
                        SteeringVectorSidecar.self, from: data)
                else { continue }
                artifacts.append(
                    VectorArtifact(directory: runDir, name: name, sidecar: sidecar))
            }
        }
        return artifacts
    }

    /// A fitted RepE reader artifact discovered on disk (a standalone
    /// `<name>.json` with `artifactType: "repe-reader-lat"` in a run
    /// directory). Readers are measurement instruments, not steering vectors
    /// — they never appear in `scan()`.
    public struct ReaderArtifactRecord: Identifiable, Sendable {
        public let directory: URL
        /// File name including the `.json` extension.
        public let fileName: String
        public let artifact: RepEReader.Artifact

        public var url: URL { directory.appending(component: fileName) }
        public var id: String { url.path }

        public var label: String {
            var parts = [
                artifact.concept, "layer \(artifact.layer)", artifact.templateID,
            ]
            if let held = artifact.heldOutAccuracy {
                parts.append(String(format: "held-out %.0f%%", held * 100))
            }
            parts.append(String(artifact.extractionDate.prefix(10)))
            return parts.joined(separator: " · ")
        }
    }

    /// All RepE reader artifacts under `runs/`, newest run first. A steering
    /// vector's sidecar JSON never decodes as a reader (no `artifactType`),
    /// so the two catalogs stay disjoint.
    public static func scanReaders(
        runsDirectory: URL = runsDirectory
    ) -> [ReaderArtifactRecord] {
        struct Peek: Decodable {
            let artifactType: String?
        }
        let fm = FileManager.default
        guard
            let runDirs = try? fm.contentsOfDirectory(
                at: runsDirectory, includingPropertiesForKeys: nil)
        else { return [] }

        var records: [ReaderArtifactRecord] = []
        for runDir in runDirs.sorted(by: { $0.lastPathComponent > $1.lastPathComponent }) {
            guard
                let files = try? fm.contentsOfDirectory(
                    at: runDir, includingPropertiesForKeys: nil)
            else { continue }
            for file in files.sorted(by: { $0.lastPathComponent < $1.lastPathComponent })
            where file.pathExtension == "json" {
                guard
                    let data = try? Data(contentsOf: file),
                    (try? JSONDecoder().decode(Peek.self, from: data))?.artifactType
                        == RepEReader.artifactType,
                    let artifact = try? JSONDecoder().decode(
                        RepEReader.Artifact.self, from: data)
                else { continue }
                records.append(
                    ReaderArtifactRecord(
                        directory: runDir,
                        fileName: file.lastPathComponent,
                        artifact: artifact))
            }
        }
        return records
    }

    public static func auditArtifacts(
        runsDirectory: URL = runsDirectory
    ) -> [ArtifactAuditFinding] {
        scan(runsDirectory: runsDirectory).flatMap { artifact -> [ArtifactAuditFinding] in
            let sidecar = artifact.sidecar
            let path = artifact.directory.appending(component: "\(artifact.name).json").path
            var findings: [ArtifactAuditFinding] = []

            if sidecar.schemaVersion == nil {
                findings.append(
                    ArtifactAuditFinding(
                        severity: .warning,
                        artifactPath: path,
                        issue: "sidecar predates schema-versioned provenance",
                        recommendation: "treat as legacy unless regenerated with current extraction"))
            }

            if sidecar.recipeMethod == VectorExtractionRecipe.Method.emotionGrandMean.rawValue,
                sidecar.extractionMethod != nil
            {
                findings.append(
                    ArtifactAuditFinding(
                        severity: .invalidForReportedRuns,
                        artifactPath: path,
                        issue: "grand-mean artifact also stamps contrastive extractionMethod",
                        recommendation: "regenerate so recipeMethod is emotionGrandMean and extractionMethod is empty"))
            }

            if let projection = sidecar.confoundProjection, !projection.isEmpty {
                findings.append(
                    ArtifactAuditFinding(
                        severity: .invalidForReportedRuns,
                        artifactPath: path,
                        issue: "uses legacy pooled neutral projection: \(projection)",
                        recommendation: "regenerate with token-bank PCA or no neutral projection"))
            }

            if sidecar.neutralProjection == nil {
                findings.append(
                    ArtifactAuditFinding(
                        severity: .warning,
                        artifactPath: path,
                        issue: "neutral projection mode was not recorded",
                        recommendation: "regenerate before using in reported comparisons"))
            } else if sidecar.neutralProjection?.hasPrefix("legacy-pooled") == true {
                findings.append(
                    ArtifactAuditFinding(
                        severity: .invalidForReportedRuns,
                        artifactPath: path,
                        issue: "neutral projection is explicitly legacy pooled",
                        recommendation: "use only for draft exploration; verified runs should use token-bank PCA or none"))
            }

            if sidecar.recipeMethod == VectorExtractionRecipe.Method.emotionGrandMean.rawValue,
                sidecar.sourceStimulusCount == nil
                    || sidecar.includedStimulusCount == nil
                    || sidecar.excludedShortStimulusCount == nil
            {
                findings.append(
                    ArtifactAuditFinding(
                        severity: .warning,
                        artifactPath: path,
                        issue: "grand-mean artifact lacks short-text screening counts",
                        recommendation: "regenerate so token-50 exclusions are provenance-stamped"))
            }

            // WS7.2: a Gemma-Scope-sourced artifact without the cross-engine
            // import-convention stamp predates the decided convention (the
            // server used to save the RAW decoder row; local imports rescaled
            // but did not declare it) — its scaling cannot be proven from the
            // sidecar. The server twin warns at load (vector_store.load).
            let fromGemmaScope =
                sidecar.extractionMethod == "gemmaScopeSAE"
                || sidecar.stimulusSetHash.hasPrefix("gemmascope:")
            if fromGemmaScope, sidecar.gemmascopeConvention == nil {
                findings.append(
                    ArtifactAuditFinding(
                        severity: .warning,
                        artifactPath: path,
                        issue:
                            "Gemma Scope SAE import lacks the gemmascopeConvention stamp: pre-convention import",
                        recommendation:
                            "re-import the feature before evidence use (decided convention: \(GemmaScopeReportCatalog.importConvention))"))
            }

            return findings
        }
    }
}
