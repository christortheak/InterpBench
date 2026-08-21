import Foundation
import Observation
import SteeringKit

/// The Data section's dataset-first inventory (WP-Data phase 1): one typed
/// row per data artifact the resolved WORKSPACE actually holds.
///
/// Design rules this file keeps:
///
/// - **It reports; it never seeds.** A fresh CONCEPT-EMPTY workspace yields
///   exactly the generic instruments its seed genuinely contains (today: the
///   neutral norm-calibration corpus) and nothing else. Nothing is created,
///   defaulted, or invented to fill a table.
/// - **One scanner.** Every kind below is enumerated through the store that
///   already owns that family — `VectorCatalog.datasetDirectoryNames` for the
///   `prompts/<family>/<name>/` layouts, `NeutralCorpusStore.scan` for
///   corpora, `StimulusSet` / `ConceptBuilder.parseProbeExamples` for row
///   counts, and the existing SHA-256 conventions for hashes. No second
///   filesystem convention is introduced here.
/// - **Absent, never faked.** `itemCount` and `contentHash` are optional. A
///   file too large for the scan budget, a malformed row, or a family whose
///   store computes no hash yields `nil` plus an `issue` string — never a
///   zero standing in for "unknown".
public enum DatasetInventory {

    /// Per-entry parse budget. Row counts and content hashes come from the
    /// existing loaders, which read whole files; above this the scan degrades
    /// to stat-level (size/date only) rather than pulling megabytes off disk
    /// on a UI refresh. 8 MB comfortably covers every stimulus set, probe
    /// set, and neutral corpus this instrument has ever produced.
    public static let maximumParsedBytes: Int64 = 8 * 1024 * 1024

    // MARK: Scan

    /// Enumerate the workspace's datasets. Pure and synchronous — callers run
    /// it off the main actor (`DatasetInventoryModel.refresh`).
    ///
    /// `root` defaults to the RESOLVED workspace
    /// (`VectorCatalog.projectRoot` → `WorkspaceRoot.current`: the
    /// `STEERLAB_WORKSPACE` → programmatic override → persisted choice →
    /// legacy-checkout chain), so a workspace switch changes the result with
    /// no caller involvement.
    public static func scan(root: URL = VectorCatalog.projectRoot) -> [DatasetInventoryEntry] {
        var entries: [DatasetInventoryEntry] = []
        let prompts = root.appending(component: "prompts")

        entries.append(contentsOf: conceptEntries(promptsRoot: prompts))
        entries.append(contentsOf: pairedEntries(root: root))
        entries.append(contentsOf: grandMeanEntries(promptsRoot: prompts))
        entries.append(contentsOf: probeEntries(promptsRoot: prompts))
        entries.append(contentsOf: neutralEntries(root: root))
        entries.append(contentsOf: batteryEntries(root: root))
        entries.append(contentsOf: promptSetEntries(root: root))

        return entries.sorted { left, right in
            if left.kind != right.kind {
                return left.kind.sortIndex < right.kind.sortIndex
            }
            return left.name.localizedStandardCompare(right.name) == .orderedAscending
        }
    }

    // MARK: Concept stimulus sets (prompts/concepts/<name>/)

    private static func conceptEntries(promptsRoot: URL) -> [DatasetInventoryEntry] {
        let familyRoot = promptsRoot.appending(component: "concepts")
        var entries: [DatasetInventoryEntry] = []
        for name in VectorCatalog.datasetDirectoryNames(in: familyRoot) {
            let directory = familyRoot.appending(component: name)
            let positive = directory.appending(component: "positive.jsonl")
            let negative = directory.appending(component: "negative.jsonl")
            let stat = FileStat.total(of: [positive, negative])

            if !stat.present.isEmpty {
                var count: Int?
                var hash: String?
                var issue: String?
                if stat.present.count < 2 {
                    let missing = [positive, negative]
                        .filter { !stat.present.contains($0) }
                        .map(\.lastPathComponent)
                        .joined(separator: ", ")
                    issue = "incomplete pair — missing \(missing)"
                } else if stat.bytes > maximumParsedBytes {
                    issue = FileStat.oversizeNote(stat.bytes)
                } else {
                    do {
                        // The same loader extraction uses, so a count shown
                        // here is the count the recipe would read.
                        let set = try StimulusSet(directory: directory)
                        count = set.positive.count + set.negative.count
                        hash = set.hash
                    } catch {
                        issue = "\(error)"
                    }
                }
                entries.append(
                    DatasetInventoryEntry(
                        kind: .conceptStimuli,
                        name: name,
                        directory: directory,
                        files: stat.present,
                        itemCount: count,
                        byteSize: stat.bytes,
                        modified: stat.modified,
                        contentHash: hash,
                        issue: issue,
                        conceptName: name))
            }

            if let validation = validationEntry(
                conceptDirectory: directory, name: name, familyLabel: "paired")
            {
                entries.append(validation)
            }
        }
        return entries
    }

    /// The concept's never-named held-out set. Its own row, not a footnote on
    /// the stimuli row: `validation.jsonl` is a measurement-side PIN
    /// (`ConceptRef.validationHash`) and a freeze gate, so its presence and
    /// size are things the researcher checks directly.
    ///
    /// It has TWO homes, and the recipe decides which
    /// (`ExperimentStore.conceptValidationRelativePath`): paired recipes read
    /// `prompts/concepts/<name>/validation.jsonl`, grand-mean reads
    /// `prompts/emotions/<name>/validation.jsonl`. Both are scanned, and the
    /// row is labelled with which — a validation set filed under the wrong
    /// recipe's root is exactly the invisible mistake this table should make
    /// visible.
    private static func validationEntry(
        conceptDirectory: URL, name: String, familyLabel: String
    ) -> DatasetInventoryEntry? {
        let url = conceptDirectory.appending(component: "validation.jsonl")
        let stat = FileStat.total(of: [url])
        guard !stat.present.isEmpty else { return nil }

        var count: Int?
        var hash: String?
        var issue: String?
        if stat.bytes > maximumParsedBytes {
            issue = FileStat.oversizeNote(stat.bytes)
        } else {
            do {
                count = try StimulusSet.loadValidation(directory: conceptDirectory)?.count
                // Same bytes, same SHA-256 the manifest pins at attach.
                hash = ExperimentStore.conceptValidationHash(fileURL: url)
            } catch {
                issue = "\(error)"
            }
        }
        return DatasetInventoryEntry(
            kind: .validationSet,
            name: name,
            directory: conceptDirectory,
            files: stat.present,
            itemCount: count,
            byteSize: stat.bytes,
            modified: stat.modified,
            contentHash: hash,
            issue: issue,
            conceptName: name,
            familyLabel: familyLabel)
    }

    // MARK: Paired stimuli (prompts/repe|readers/<name>/pairs.jsonl)

    /// Both paired roots, enumerated through the layout authority
    /// (`VectorCatalog.PairedStimulusFamily`) rather than a local list of
    /// directory names.
    ///
    /// The two families share a filename and NOT a row shape, so each is
    /// counted by the loader its own recipe reads with
    /// (`pairedRowCount(family:url:)`). Counting both with the RepE-LAT
    /// loader — as this did before phase 4 — reported every real reader
    /// dataset as malformed, including one this workspace's own creation
    /// flow had just validated and filed.
    private static func pairedEntries(root: URL) -> [DatasetInventoryEntry] {
        var entries: [DatasetInventoryEntry] = []
        for family in VectorCatalog.PairedStimulusFamily.allCases {
            let familyRoot = VectorCatalog.pairedStimuliRoot(family: family, root: root)
            for name in VectorCatalog.datasetDirectoryNames(in: familyRoot) {
                let directory = VectorCatalog.pairedStimuliDirectory(
                    family: family, name: name, root: root)
                let url = VectorCatalog.pairedStimuliFile(
                    family: family, name: name, root: root)
                let stat = FileStat.total(of: [url])
                guard !stat.present.isEmpty else { continue }

                var count: Int?
                var hash: String?
                var issue: String?
                if stat.bytes > maximumParsedBytes {
                    issue = FileStat.oversizeNote(stat.bytes)
                } else {
                    do {
                        (count, hash) = try pairedRowCount(family: family, url: url)
                    } catch {
                        issue = "\(error)"
                    }
                }
                entries.append(
                    DatasetInventoryEntry(
                        kind: .pairedStimuli,
                        name: name,
                        directory: directory,
                        files: stat.present,
                        itemCount: count,
                        byteSize: stat.bytes,
                        modified: stat.modified,
                        contentHash: hash,
                        issue: issue,
                        conceptName: name,
                        familyLabel: family.label,
                        note: family.detail))
            }
        }
        return entries
    }

    /// Rows + pinned digest for one paired file, read with the loader that
    /// family's recipe uses. Both loaders hash the RAW BYTES, so the two
    /// digests are the same convention (and the same value a build stamps).
    static func pairedRowCount(
        family: VectorCatalog.PairedStimulusFamily, url: URL
    ) throws -> (Int, String) {
        switch family {
        case .repe:
            let loaded = try StimulusSet.loadPairs(url: url)
            return (loaded.pairs.count, loaded.hash)
        case .readers:
            let loaded = try RepEReader.loadPairs(url: url)
            return (loaded.pairs.count, loaded.hash)
        }
    }

    // MARK: Grand-mean story corpora (prompts/emotions/<name>/stories.jsonl)

    private static func grandMeanEntries(promptsRoot: URL) -> [DatasetInventoryEntry] {
        let familyRoot = promptsRoot.appending(component: "emotions")
        var entries: [DatasetInventoryEntry] = []
        for name in VectorCatalog.datasetDirectoryNames(in: familyRoot) {
            let directory = familyRoot.appending(component: name)

            // The grand-mean recipe's held-out set lives HERE, not under
            // prompts/concepts/ (see `validationEntry`).
            if let validation = validationEntry(
                conceptDirectory: directory, name: name, familyLabel: "grand-mean")
            {
                entries.append(validation)
            }

            let url = directory.appending(component: "stories.jsonl")
            let stat = FileStat.total(of: [url])
            guard !stat.present.isEmpty else { continue }

            var count: Int?
            var hash: String?
            var issue: String?
            if stat.bytes > maximumParsedBytes {
                issue = FileStat.oversizeNote(stat.bytes)
            } else {
                do {
                    let loaded = try StimulusSet.loadMultiConceptTexts(url: url)
                    count = loaded.rows.count
                    hash = loaded.hash
                } catch {
                    issue = "\(error)"
                }
            }
            entries.append(
                DatasetInventoryEntry(
                    kind: .grandMeanCorpus,
                    name: name,
                    directory: directory,
                    files: stat.present,
                    itemCount: count,
                    byteSize: stat.bytes,
                    modified: stat.modified,
                    contentHash: hash,
                    issue: issue,
                    conceptName: name))
        }
        return entries
    }

    // MARK: Probe item sets (prompts/probes/<name>/items.jsonl)

    private static func probeEntries(promptsRoot: URL) -> [DatasetInventoryEntry] {
        let familyRoot = promptsRoot.appending(component: "probes")
        var entries: [DatasetInventoryEntry] = []
        for name in VectorCatalog.datasetDirectoryNames(in: familyRoot) {
            let directory = familyRoot.appending(component: name)
            let url = directory.appending(component: "items.jsonl")
            let stat = FileStat.total(of: [url])
            guard !stat.present.isEmpty else { continue }

            var count: Int?
            var issue: String?
            if stat.bytes > maximumParsedBytes {
                issue = FileStat.oversizeNote(stat.bytes)
            } else {
                do {
                    // The Concept Lab's own parser, so a row it would reject
                    // is a row this inventory reports as an issue.
                    count = try ConceptBuilder.parseProbeExamples(
                        Data(contentsOf: url),
                        filename: url.lastPathComponent,
                        concept: name).count
                } catch {
                    issue = "\(error)"
                }
            }
            entries.append(
                DatasetInventoryEntry(
                    kind: .probeItems,
                    name: name,
                    directory: directory,
                    files: stat.present,
                    itemCount: count,
                    byteSize: stat.bytes,
                    modified: stat.modified,
                    // No store computes a pinned hash for items.jsonl, so
                    // none is shown. Absent, not invented.
                    contentHash: nil,
                    issue: issue,
                    conceptName: name))
        }
        return entries
    }

    // MARK: Neutral corpora (prompts/neutral/…)

    private static func neutralEntries(root: URL) -> [DatasetInventoryEntry] {
        NeutralCorpusStore.scan(root: root).compactMap { record in
            let stat = FileStat.total(of: [record.url])
            // `scan()` always synthesizes the norm-calibration record, even
            // when the file is absent; the inventory lists what EXISTS.
            guard !stat.present.isEmpty else { return nil }
            return DatasetInventoryEntry(
                kind: .neutralCorpus,
                name: record.name,
                directory: record.url.deletingLastPathComponent(),
                files: stat.present,
                itemCount: record.count,
                byteSize: stat.bytes,
                modified: stat.modified,
                contentHash: record.hash,
                issue: record.hash == nil ? "could not be parsed as a text corpus" : nil,
                conceptName: nil,
                familyLabel: record.kind.label)
        }
    }

    // MARK: Capability batteries (prompts/batteries/*.jsonl)

    /// The workspace's capability batteries, read by the engine's OWN battery
    /// loader (`CapabilityBattery`, which parses both the legacy headerless
    /// format and the header-declared format 2 since the twin landed). A file
    /// this row counts is a file `run`'s per-condition battery pass would
    /// read, and a file the loader rejects is reported as an issue rather
    /// than silently dropped.
    ///
    /// Two honesty notes the rows carry:
    ///
    /// - The directory is a CONVENTION, not an exhaustive authority. A
    ///   manifest pins a battery by workspace-relative path
    ///   (`ExperimentStore.pinCapabilityBattery` accepts any path), so a
    ///   battery filed elsewhere is legal and simply not listed here.
    /// - The hash is the one the PIN uses: `ExperimentStore.sha256Hex` over
    ///   the raw file bytes, byte-identical to `sweep.batteryHash` and
    ///   `manifest.capabilityBatteryHash`. No canonicalization — which is
    ///   exactly why a format-1 file's historical pin still means what it
    ///   always meant.
    private static func batteryEntries(root: URL) -> [DatasetInventoryEntry] {
        jsonlFiles(in: VectorCatalog.batteriesDirectory(root: root)).map { url in
            let stat = FileStat.total(of: [url])
            var count: Int?
            var hash: String?
            var issue: String?
            var format: Int?
            var description: String?

            if stat.bytes > maximumParsedBytes {
                issue = FileStat.oversizeNote(stat.bytes)
            } else if let data = try? Data(contentsOf: url) {
                hash = ExperimentStore.sha256Hex(data)
                do {
                    let battery = try CapabilityBattery(
                        data: data, file: url.lastPathComponent)
                    count = battery.items.count
                    format = battery.formatVersion
                    // The header's free-text `description`, read through the
                    // SAME header parser the loader uses to identify format 2.
                    // The loader itself ignores the key (it is not part of
                    // the arming contract), so this is the one place it is
                    // surfaced — as documentation, never as behaviour.
                    if let first = PinShapeValidation.batteryLines(data).first,
                        let header = PinShapeValidation.batteryHeaderObject(first),
                        case .string(let text)? = header["description"],
                        !text.isEmpty
                    {
                        description = text
                    }
                } catch {
                    issue = "\(error)"
                }
            } else {
                issue = "could not be read"
            }

            return DatasetInventoryEntry(
                kind: .capabilityBattery,
                name: url.deletingPathExtension().lastPathComponent,
                directory: url.deletingLastPathComponent(),
                files: stat.present,
                itemCount: count,
                byteSize: stat.bytes,
                modified: stat.modified,
                contentHash: hash,
                issue: issue,
                conceptName: nil,
                familyLabel: format.map { "format \($0)" },
                note: batteryNote(format: format, description: description))
        }
    }

    private static func batteryNote(format: Int?, description: String?) -> String? {
        var parts: [String] = []
        if let description { parts.append(description) }
        if format == CapabilityBattery.legacyFormat {
            // Not a defect — legacy files pin and run exactly as before. But
            // the authoring path has moved on, and the lint verb says so.
            parts.append(
                "legacy headerless format: it still pins and runs unchanged, "
                    + "but its arming comes from whatever instrument loads it. "
                    + "Format 2 declares its own scoring, prompt mode, and "
                    + "token cap so a reading is comparable across "
                    + "instruments — steerlab-server battery lint <file> "
                    + "reports what a repair would change.")
        }
        return parts.isEmpty ? nil : parts.joined(separator: " · ")
    }

    // MARK: Task and dev prompt sets (prompts/tasks/, prompts/dev/)

    /// The workspace's prompt sets, enumerated by `TaskPromptsStore` and read
    /// by the loader that OWNS each:
    ///
    /// - `prompts/tasks/*.jsonl` — the measured task prompts. Counted with
    ///   `ExperimentTasks.parseTaskPrompts`, the parser both
    ///   `ExperimentStore.pinTaskPrompts` and the run loop use; the hash is
    ///   the raw-bytes SHA-256 pinned as `taskPromptsHash`. The directory is
    ///   where `DataTemplates.taskPromptsDestination` tells a researcher to
    ///   author them.
    /// - `prompts/dev/*.jsonl` — sweep dev splits and robustness coherence
    ///   prompts. Counted with `StimulusSet.loadTexts`, the loader the sweep
    ///   itself uses, whose hash IS `sweep.devPromptsHash`.
    /// - Plus any file an `experiments/` manifest actually pins as its
    ///   `taskPromptsFile`, wherever it lives — labelled `pinned` and naming
    ///   the study that pins it. Phase 3 could only DISCLOSE that such files
    ///   might exist; phase 4 lists them, because the manifests know.
    ///
    /// `TaskPromptsStore` owns the exact rule (and what it still refuses to
    /// invent: no recursive walk, and a pinned path with no file on disk is
    /// not listed).
    private static func promptSetEntries(root: URL) -> [DatasetInventoryEntry] {
        TaskPromptsStore.list(root: root).map(promptSetEntry)
    }

    private static func promptSetEntry(_ record: TaskPromptsStore.Record)
        -> DatasetInventoryEntry
    {
        let url = record.url
        let stat = FileStat.total(of: [url])
        var count: Int?
        var hash: String?
        var issue: String?

        if stat.bytes > maximumParsedBytes {
            issue = FileStat.oversizeNote(stat.bytes)
        } else if let data = try? Data(contentsOf: url) {
            // Both pins are raw-bytes SHA-256 through the same helper, so the
            // hash is reportable even when the family's parser rejects the
            // rows — which is the case a researcher most wants to see.
            hash = ExperimentStore.sha256Hex(data)
            do {
                switch record.readingFamily {
                case .task:
                    count = try ExperimentTasks.parseTaskPrompts(data).count
                case .dev:
                    count = try StimulusSet.loadTexts(url: url).texts.count
                }
            } catch {
                issue = "\(error)"
            }
        } else {
            issue = "could not be read"
        }

        return DatasetInventoryEntry(
            kind: .promptSet,
            name: url.deletingPathExtension().lastPathComponent,
            directory: url.deletingLastPathComponent(),
            files: stat.present,
            itemCount: count,
            byteSize: stat.bytes,
            modified: stat.modified,
            contentHash: hash,
            issue: issue,
            conceptName: nil,
            familyLabel: record.familyLabel,
            note: record.note)
    }

    /// Sorted `*.jsonl` files directly inside one directory — the listing
    /// primitive the flat-file families need. Non-recursive on purpose: these
    /// roots are flat by convention, and descending into subdirectories would
    /// be inventing a layout. Shared with `TaskPromptsStore`, which applies
    /// the same rule to the two prompt-set roots.
    private static func jsonlFiles(in directory: URL) -> [URL] {
        TaskPromptsStore.jsonlFiles(in: directory)
    }

    // MARK: Stat helpers (the only filesystem primitive in this file)

    enum FileStat {
        struct Total {
            var present: [URL]
            var bytes: Int64
            var modified: Date?
        }

        /// Stat-level roll-up over a candidate file list: which exist, their
        /// summed size, and the newest modification date.
        static func total(of candidates: [URL]) -> Total {
            var total = Total(present: [], bytes: 0, modified: nil)
            for url in candidates {
                guard
                    let values = try? url.resourceValues(forKeys: [
                        .fileSizeKey, .contentModificationDateKey, .isRegularFileKey,
                    ]),
                    values.isRegularFile == true
                else { continue }
                total.present.append(url)
                total.bytes += Int64(values.fileSize ?? 0)
                if let date = values.contentModificationDate {
                    total.modified = max(total.modified ?? date, date)
                }
            }
            return total
        }

        static func oversizeNote(_ bytes: Int64) -> String {
            "not counted — \(DatasetInventoryEntry.formattedSize(bytes)) exceeds the "
                + "inventory's \(DatasetInventoryEntry.formattedSize(maximumParsedBytes)) "
                + "parse budget; open it in the Concept Builder to read it"
        }
    }
}

// MARK: - Kinds

/// The dataset families this scope enumerates. Deliberately DATASETS only —
/// derived artifacts (vectors, trained reading probes, adapters, neutral-PC
/// bases, agents) are the Derived scope's rows
/// (`DerivedArtifactInventory`), because mixing them in would make "items"
/// and "size" mean two different things in one column.
///
/// Phase 3 added the two INSTRUMENT families phase 1 deferred: capability
/// batteries and task/dev prompt sets. Both are workspace-wide instruments
/// rather than concept-scoped data, and both are pinned inputs — which is
/// why they belong in a table the researcher checks before a freeze.
public enum DatasetKind: String, Sendable, CaseIterable, Identifiable, Codable {
    case conceptStimuli
    case pairedStimuli
    case grandMeanCorpus
    case probeItems
    case validationSet
    case neutralCorpus
    case capabilityBattery
    case promptSet

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .conceptStimuli: "Concept stimuli"
        case .pairedStimuli: "Paired stimuli"
        case .grandMeanCorpus: "Story corpus"
        case .probeItems: "Probe items"
        case .validationSet: "Validation set"
        case .neutralCorpus: "Neutral corpus"
        case .capabilityBattery: "Capability battery"
        case .promptSet: "Prompt set"
        }
    }

    /// One line of what the family IS and what reads it — the detail pane's
    /// orientation text, kept beside the kind so the view stays thin.
    public var detail: String {
        switch self {
        case .conceptStimuli:
            "contrastive positive/negative stimuli — the CAA recipe's input "
                + "(prompts/concepts/<name>/{positive,negative}.jsonl)"
        case .pairedStimuli:
            "content-matched pairs for the RepE/LAT and reader recipes "
                + "(prompts/{repe,readers}/<name>/pairs.jsonl)"
        case .grandMeanCorpus:
            "multi-concept story rows for the grand-mean recipe "
                + "(prompts/emotions/<name>/stories.jsonl)"
        case .probeItems:
            "labelled items a linear reading probe trains and scores on "
                + "(prompts/probes/<name>/items.jsonl)"
        case .validationSet:
            "the never-named held-out scenarios the convergent gate reads; "
                + "pinned as validationHash and checked at freeze "
                + "(prompts/concepts/<name>/validation.jsonl)"
        case .neutralCorpus:
            "the pinned neutral denominator — residual-norm calibration and "
                + "projection bases (prompts/neutral/…)"
        case .capabilityBattery:
            "deterministic capability probes a run scores per condition — the "
                + "control that separates 'the concept moved the holding' from "
                + "'the injection broke the model' (prompts/batteries/*.jsonl)"
        case .promptSet:
            "the prompts a study or a sweep actually runs: measured task "
                + "prompts (prompts/tasks/), sweep dev / coherence splits "
                + "(prompts/dev/), and any file a manifest pins as its task "
                + "prompts wherever it lives"
        }
    }

    /// Table/sort order: the concept-scoped families first, the workspace-wide
    /// instruments last.
    var sortIndex: Int {
        switch self {
        case .conceptStimuli: 0
        case .validationSet: 1
        case .pairedStimuli: 2
        case .grandMeanCorpus: 3
        case .probeItems: 4
        case .neutralCorpus: 5
        case .promptSet: 6
        case .capabilityBattery: 7
        }
    }

    /// True for the FLAT-FILE families: one directory holds many independent
    /// datasets, so a row's identity is its file, not its folder.
    var identifiesByFile: Bool {
        switch self {
        case .capabilityBattery, .promptSet: true
        case .conceptStimuli, .pairedStimuli, .grandMeanCorpus, .probeItems,
            .validationSet, .neutralCorpus:
            false
        }
    }
}

// MARK: - Entry

/// One row of the inventory. Value type, `Sendable`, built entirely off the
/// main actor.
public struct DatasetInventoryEntry: Identifiable, Sendable, Equatable, Hashable {
    public let kind: DatasetKind
    public let name: String
    /// The directory that OWNS this dataset (a concept folder, or the corpus
    /// folder) — what "Reveal in Finder" opens when there is more than one file.
    public let directory: URL
    /// The files that actually exist on disk, in the order the family reads them.
    public let files: [URL]
    /// Rows, when a store could cheaply count them. `nil` = unknown (too large
    /// to parse, or malformed) — never 0 standing in for unknown.
    public let itemCount: Int?
    public let byteSize: Int64
    public let modified: Date?
    /// The SHA-256 an existing store already computes for this family's bytes.
    /// `nil` where no store defines one (probe items) — the inventory adds no
    /// new hashing of its own.
    public let contentHash: String?
    /// A non-fatal problem: a missing half of a pair, a malformed row, a file
    /// past the parse budget. The entry is still listed.
    public let issue: String?
    /// The concept this dataset belongs to, when it belongs to one — the
    /// routing key for "Open in Concept Builder".
    public let conceptName: String?
    /// Sub-family label where one family has several roots (repe vs readers,
    /// norm-calibration vs projection, battery format, task vs dev).
    public let familyLabel: String?
    /// A per-ROW fact the kind's shared `detail` sentence cannot carry: a
    /// battery's declared description and its legacy-format caption, or which
    /// loader and which pin a prompt-set root belongs to. Not a problem (that
    /// is `issue`) — orientation.
    public let note: String?

    public init(
        kind: DatasetKind,
        name: String,
        directory: URL,
        files: [URL],
        itemCount: Int?,
        byteSize: Int64,
        modified: Date?,
        contentHash: String?,
        issue: String?,
        conceptName: String?,
        familyLabel: String? = nil,
        note: String? = nil
    ) {
        self.kind = kind
        self.name = name
        self.directory = directory
        self.files = files
        self.itemCount = itemCount
        self.byteSize = byteSize
        self.modified = modified
        self.contentHash = contentHash
        self.issue = issue
        self.conceptName = conceptName
        self.familyLabel = familyLabel
        self.note = note
        // Flat-file families put several datasets in ONE directory, so the
        // owning directory alone is not an identity there (phase 1's
        // per-concept families each own their folder). The row's primary
        // FILE is what distinguishes `basic.jsonl` from `factual-short.jsonl`.
        self.id = kind.identifiesByFile && files.count == 1
            ? Self.id(kind: kind, fileURL: files[0])
            : Self.id(kind: kind, directory: directory)
    }

    /// Stable across scans of the same workspace (path-derived), and distinct
    /// between kinds that share a directory (stimuli vs validation set).
    /// Computed once at construction — the formula resolves symlinks, which
    /// is a `stat` per component and has no business running inside a table
    /// row's identity getter.
    public let id: String

    /// THE row-identity formula, exposed so the creation flow
    /// (`DatasetCreationPlan.inventoryEntryID`) can name the row a dataset
    /// WILL have before it exists, and select it after the re-scan. One
    /// formula — a second one is how the two would stop matching.
    ///
    /// The path is resolved, not merely absolute: `contentsOfDirectory`
    /// hands back `/private/var/…` where the store's own URL builder yields
    /// `/var/…`, so two callers naming the SAME directory produced two ids
    /// (found by the creation flow's round-trip test, on the projection
    /// corpora — the one family the scan enumerates rather than constructs).
    public static func id(kind: DatasetKind, directory: URL) -> String {
        "\(kind.rawValue):\(directory.standardizedFileURL.resolvingSymlinksInPath().path)"
    }

    /// The FILE-addressed form, for the flat-file families where one
    /// directory holds many datasets (batteries, prompt sets). Same formula,
    /// same resolution — the creation flow uses it to name a row before it
    /// exists (`DatasetCreationPlan.inventoryEntryID`).
    public static func id(kind: DatasetKind, fileURL: URL) -> String {
        "\(kind.rawValue):\(fileURL.standardizedFileURL.resolvingSymlinksInPath().path)"
    }

    public var kindLabel: String {
        guard let familyLabel else { return kind.label }
        return "\(kind.label) · \(familyLabel)"
    }

    /// The single path the row identifies: the file when there is exactly
    /// one, the owning directory otherwise.
    public var primaryURL: URL { files.count == 1 ? files[0] : directory }

    /// Workspace-relative where possible (the researcher reads these against
    /// `prompts/…`, not against `/Users/…`).
    public func displayPath(root: URL = VectorCatalog.projectRoot) -> String {
        let rootPath = root.standardizedFileURL.path
        let path = primaryURL.standardizedFileURL.path
        if path.hasPrefix(rootPath + "/") {
            return String(path.dropFirst(rootPath.count + 1))
        }
        return path
    }

    // MARK: Sort/display helpers (non-optional so `KeyPathComparator` works)

    public var sortableItemCount: Int { itemCount ?? -1 }
    public var sortableModified: Date { modified ?? .distantPast }

    public var itemCountText: String { itemCount.map(String.init) ?? "—" }
    public var sizeText: String { Self.formattedSize(byteSize) }
    public var modifiedText: String {
        guard let modified else { return "—" }
        return Self.dateFormatter.string(from: modified)
    }
    /// Same 12-hex-prefix convention the rest of the app uses for SHA-256.
    public var shortHash: String? { contentHash.map { String($0.prefix(12)) } }

    public static func formattedSize(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .file
        formatter.allowedUnits = [.useBytes, .useKB, .useMB, .useGB]
        return formatter.string(fromByteCount: bytes)
    }

    private static let dateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        return formatter
    }()
}

// MARK: - Observable model

/// What the Data section's Inventory region renders — BOTH scopes. Owns the
/// scan lifecycle (off the main actor, latest-wins) so the view holds no
/// logic.
///
/// One model, one refresh, two lists: the scope switch is a view-level
/// choice of which list to show, never a reason to re-scan. That also means
/// the single existing refresh seam (`WorkspaceControls.resetCatalogs`)
/// keeps both scopes current with no new wiring.
@Observable @MainActor
public final class DatasetInventoryModel {
    public private(set) var entries: [DatasetInventoryEntry] = []
    public private(set) var derived: [DerivedArtifactEntry] = []
    public private(set) var isScanning = false
    /// The workspace the current `entries` describe — nil before the first
    /// scan completes. Rendered so a stale table can never masquerade as the
    /// workspace the toolbar names.
    public private(set) var scannedRoot: URL?
    /// True once a scan has completed, so "no datasets" and "not scanned yet"
    /// are distinguishable empty states.
    public private(set) var hasScanned = false

    /// Latest-wins guard: a workspace switch during an in-flight scan must not
    /// let the previous root's entries land afterwards.
    private var generation = 0

    public init() {}

    /// Re-scan the RESOLVED workspace. Cheap to call — it is wired to the same
    /// seams as the other catalog refreshes (`WorkspaceControls.resetCatalogs`)
    /// and to the region appearing.
    public func refresh() {
        let root = VectorCatalog.projectRoot
        generation &+= 1
        let token = generation
        isScanning = true
        Task.detached(priority: .userInitiated) {
            let scanned = DatasetInventory.scan(root: root)
            let scannedDerived = DerivedArtifactInventory.scan(root: root)
            await MainActor.run { [weak self] in
                guard let self, token == self.generation else { return }
                self.entries = scanned
                self.derived = scannedDerived
                self.scannedRoot = root
                self.hasScanned = true
                self.isScanning = false
            }
        }
    }

    /// The kinds actually present, in table order — the filter menu's options
    /// (a filter for a kind with no rows is noise).
    public var presentKinds: [DatasetKind] {
        var seen: Set<DatasetKind> = []
        return entries.compactMap { seen.insert($0.kind).inserted ? $0.kind : nil }
    }

    public var presentDerivedKinds: [DerivedArtifactKind] {
        var seen: Set<DerivedArtifactKind> = []
        return derived.compactMap { seen.insert($0.kind).inserted ? $0.kind : nil }
    }

    public var issueCount: Int { entries.filter { $0.issue != nil }.count }
}
