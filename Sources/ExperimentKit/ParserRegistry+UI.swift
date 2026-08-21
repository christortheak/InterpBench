import Foundation

// UI-facing read helpers + the manifest setter for the declared numeric
// parser (USABILITY-PLAN Phase-4 item 18's UI affordance). A NEW file by
// the ownership rule: `ParserRegistry.swift` and `ExperimentStore.swift`
// carry the cross-engine science contract and are not edited for UI needs.
// Everything here is presentation-side: listing the registry for a picker,
// surfacing pin problems in plain words, and writing the researcher's
// choice through the ordinary draft-edit gate.

/// Read-only registry views for the study panel's parser picker.
public enum ParserRegistryUI {

    /// One registry entry as the picker shows it: the name the manifest
    /// stores, the declared kind, and the author's plain-words description.
    public struct Entry: Identifiable, Sendable, Equatable {
        public let name: String
        public let kind: String
        public let description: String

        public var id: String { name }

        /// The kind in plain words (the engine vocabulary stays in `kind`).
        public var plainKind: String {
            switch kind {
            case "durationMonths": "a duration, read in months"
            case "number": "a plain number"
            default: kind
            }
        }
    }

    /// The workspace registry's entries, sorted by name — empty when the
    /// registry is missing or unreadable (`loadProblem()` says why). Every
    /// declared entry is listed, including a malformed one: picking it
    /// surfaces its shape problem at the moment of selection, in plain
    /// words, instead of hiding the entry.
    public static func entries() -> [Entry] {
        guard let specs = try? ParserRegistry.loadSpecs() else { return [] }
        return specs
            .map { name, spec in
                Entry(
                    name: name,
                    kind: spec.kind ?? "",
                    description: spec.description ?? "")
            }
            .sorted { $0.name < $1.name }
    }

    /// Plain-language reason the registry cannot be listed (missing file,
    /// unreadable JSON, wrong schema) — nil when it loads.
    public static func loadProblem() -> String? {
        do {
            _ = try ParserRegistry.loadSpecs()
            return nil
        } catch {
            return (error as? ExperimentError)?.reason ?? "\(error)"
        }
    }

    /// The declared parser's problems for THIS study, in plain words:
    /// registry drift from the pinned hash, a missing registry, an
    /// undefined or malformed entry, or an unused pin. Empty when the
    /// declaration is healthy (or when nothing is declared and nothing is
    /// pinned). Wraps the engine's own verify-surface checks so the panel
    /// never re-implements them.
    public static func problems(for manifest: ExperimentManifest) -> [String] {
        ParserRegistry.pinViolations(manifest)
    }

    /// True when the manifest pins a registry hash that no longer matches
    /// the file's current bytes — the drift the standard affordance shows.
    public static func registryDrifted(_ manifest: ExperimentManifest) -> Bool {
        guard let pinned = manifest.parserRegistryHash else { return false }
        return ParserRegistry.liveHash() != pinned
    }
}

extension ExperimentStore {

    /// Declare (or clear) the study's registry parser, through the same
    /// draft-edit gate as every science-manifest setter. Declaring a name
    /// shape-checks the entry NOW (feedback at the moment of action) and
    /// pins the registry file's current SHA-256 as `parserRegistryHash` —
    /// later registry edits surface as drift, never as a silent change of
    /// measurement. Re-declaring the same name deliberately re-pins the
    /// current bytes (the drift-repair affordance). Clearing the parser
    /// clears the pin too (an unused pin certifies nothing and is a verify
    /// finding).
    @discardableResult
    public static func setNumericParser(
        _ name: String?, experimentName: String
    ) throws -> ExperimentManifest {
        try updateDraft(name: experimentName) { manifest in
            let trimmed = name?.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let value = trimmed, !value.isEmpty else {
                manifest.numericParser = nil
                manifest.parserRegistryHash = nil
                return
            }
            // Validates existence and shape with ParserRegistry's own
            // plain-language refusals.
            _ = try ParserRegistry.spec(named: value)
            guard let hash = ParserRegistry.liveHash() else {
                throw ExperimentError(
                    reason: "no parser registry exists at "
                        + ParserRegistry.registryFile)
            }
            manifest.numericParser = value
            manifest.parserRegistryHash = hash
        }
    }
}
