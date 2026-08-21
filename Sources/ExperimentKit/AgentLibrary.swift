import Foundation

/// Pure, unit-testable rules behind the Agent Library UI (the Agents
/// section): agent KIND classification, readiness chips, virtual baseline
/// rows, and list filters.
///
/// Views render these verbatim — no science or availability logic lives in
/// SwiftUI. Everything arrives as data (sets of ids the active workspaces can
/// resolve), so the rules run in plain CPU tests.
public enum AgentLibrary {

    // MARK: Kind

    /// One user-facing kind per row. Priority: promotion provenance beats
    /// composition (a sweep-promoted agent with an adapter is still
    /// "sweep-promoted" — the birth certificate is the headline).
    public enum Kind: String, Sendable, CaseIterable {
        /// Virtual row derived from a base model id — never on disk.
        case baseline
        case sweepPromoted
        case overridePromoted
        case adapter
        case vectorOnly
        case exploratory

        public var label: String {
            switch self {
            case .baseline: "baseline (virtual)"
            case .sweepPromoted: "sweep-promoted"
            case .overridePromoted: "override-promoted"
            case .adapter: "adapter"
            case .vectorOnly: "vector-only"
            case .exploratory: "exploratory"
            }
        }
    }

    public static func kind(of artifact: ModelVariantArtifact) -> Kind {
        if let promotion = artifact.promotion {
            return promotion.promotedBy == "manualOverride"
                ? .overridePromoted : .sweepPromoted
        }
        if !artifact.adapters.isEmpty { return .adapter }
        if !artifact.injections.isEmpty { return .vectorOnly }
        return .exploratory
    }

    // MARK: Availability (data in, chips out)

    /// What the active workspaces can actually resolve, passed in as plain
    /// data. The server verdict REUSES the existing pure rule
    /// (`WorkspaceScoping.serverDefinitionApplicability`) — the caller
    /// evaluates it against the live server catalog and hands over the
    /// boolean; nil means no server workspace is active or connected, in
    /// which case no server chip is claimed at all (honest, not optimistic).
    public struct Availability: Sendable {
        public var localModelIDs: Set<String>
        public var localVectorIDs: Set<String>
        /// Registered adapter identities: artifactPath ∪ adapterDirectory of
        /// every locally registered adapter artifact.
        public var localAdapterKeys: Set<String>
        public var serverApplicability: Bool?
        public var serverVectorIDs: Set<String>

        public init(
            localModelIDs: Set<String> = [],
            localVectorIDs: Set<String> = [],
            localAdapterKeys: Set<String> = [],
            serverApplicability: Bool? = nil,
            serverVectorIDs: Set<String> = []
        ) {
            self.localModelIDs = localModelIDs
            self.localVectorIDs = localVectorIDs
            self.localAdapterKeys = localAdapterKeys
            self.serverApplicability = serverApplicability
            self.serverVectorIDs = serverVectorIDs
        }
    }

    // MARK: Readiness chips

    public enum ChipTone: String, Sendable {
        case positive
        case neutral
        case warning
    }

    public struct Chip: Sendable, Equatable, Hashable, Identifiable {
        public var label: String
        public var tone: ChipTone
        public var help: String

        public var id: String { label }

        public init(label: String, tone: ChipTone, help: String) {
            self.label = label
            self.tone = tone
            self.help = help
        }
    }

    /// True when every component of the definition resolves in the LOCAL
    /// workspace: base model in the local tiers, every vector ref a local
    /// artifact, every adapter ref a registered local adapter.
    public static func isRunnableLocally(
        _ artifact: ModelVariantArtifact, availability: Availability
    ) -> Bool {
        guard availability.localModelIDs.contains(artifact.baseModelID) else {
            return false
        }
        // Path-shape tolerant (B3): catalog ids are absolute, but a
        // server-promoted agent stores workspace-relative references, so a
        // verbatim string compare reported every imported agent as missing
        // its vector no matter how many times it was imported.
        for injection in artifact.injections
        where !ArtifactIdentity.contains(
            availability.localVectorIDs, injection.vectorArtifactID)
        {
            return false
        }
        for adapter in artifact.adapters
        where !ArtifactIdentity.contains(
            availability.localAdapterKeys, adapter.artifactPath)
            && !ArtifactIdentity.contains(
                availability.localAdapterKeys, adapter.adapterDirectory)
        {
            return false
        }
        return true
    }

    /// Vector refs that resolve in NO provided catalog (local or server) —
    /// the "Missing artifact" evidence. When no server catalog was provided,
    /// the judgment is against the local catalog alone.
    public static func missingVectorRefs(
        _ artifact: ModelVariantArtifact, availability: Availability
    ) -> [String] {
        artifact.injections.map(\.vectorArtifactID).filter {
            !ArtifactIdentity.contains(availability.localVectorIDs, $0)
                && !ArtifactIdentity.contains(availability.serverVectorIDs, $0)
        }
    }

    /// Adapter refs whose RECORDED hash disagrees with the registered
    /// adapter artifact's CURRENT hash for the same adapter directory —
    /// `currentAdapterHashes` maps adapterDirectory → the hash the local
    /// adapter registry records today. Only refs where both sides record a
    /// hash can drift; unknown stays silent, never guessed.
    public static func driftedAdapterRefs(
        _ artifact: ModelVariantArtifact,
        currentAdapterHashes: [String: String]
    ) -> [String] {
        artifact.adapters.compactMap { ref in
            guard let recorded = ref.adapterHash,
                let current = currentAdapterHashes[ref.adapterDirectory],
                recorded != current
            else { return nil }
            return ref.name
        }
    }

    /// The readiness chip row for one saved agent, in stable display order:
    /// provenance, composition, runnability, then integrity warnings.
    public static func chips(
        for artifact: ModelVariantArtifact,
        availability: Availability,
        currentAdapterHashes: [String: String] = [:]
    ) -> [Chip] {
        var chips: [Chip] = []

        switch kind(of: artifact) {
        case .sweepPromoted:
            chips.append(
                Chip(
                    label: "Sweep-promoted", tone: .positive,
                    help: "settings selected on dev data by the declared sweep "
                        + "criterion, then promoted (birth certificate attached)"))
        case .overridePromoted:
            chips.append(
                Chip(
                    label: "Override", tone: .warning,
                    help: "promoted from a sweep, but the declared selection was "
                        + "bypassed — stamped promotedBy: manualOverride"))
        default:
            chips.append(
                Chip(
                    label: "Exploratory", tone: .neutral,
                    help: "hand-created agent; its settings were not selected by "
                        + "a declared sweep criterion"))
        }

        if !artifact.adapters.isEmpty {
            chips.append(
                Chip(
                    label: "Uses adapter", tone: .neutral,
                    help: "applies \(artifact.adapters.count) LoRA/DoRA "
                        + "adapter\(artifact.adapters.count == 1 ? "" : "s")"))
        }
        if !artifact.injections.isEmpty {
            chips.append(
                Chip(
                    label: "Uses steering", tone: .neutral,
                    help: "injects \(artifact.injections.count) steering "
                        + "vector\(artifact.injections.count == 1 ? "" : "s") "
                        + "into the residual stream"))
        }

        if isRunnableLocally(artifact, availability: availability) {
            chips.append(
                Chip(
                    label: "Runnable locally", tone: .positive,
                    help: "base model, vectors, and adapters all resolve in the "
                        + "Local (MLX) workspace"))
        }
        if availability.serverApplicability == true {
            chips.append(
                Chip(
                    label: "Runnable on server", tone: .positive,
                    help: "base model and refs resolve on the active server "
                        + "(appliable as an inline spec)"))
        }

        let missing = missingVectorRefs(artifact, availability: availability)
        if !missing.isEmpty {
            chips.append(
                Chip(
                    label: "Missing artifact", tone: .warning,
                    help: "vector ref\(missing.count == 1 ? "" : "s") not found "
                        + "in any available catalog: "
                        + missing.joined(separator: ", ")))
        }

        let drifted = driftedAdapterRefs(
            artifact, currentAdapterHashes: currentAdapterHashes)
        if !drifted.isEmpty {
            chips.append(
                Chip(
                    label: "Hash drift", tone: .warning,
                    help: "recorded adapter hash no longer matches the registered "
                        + "adapter artifact: " + drifted.joined(separator: ", ")))
        }

        return chips
    }

    // MARK: Virtual baseline rows

    /// One derived, never-persisted row per distinct base model among the
    /// saved agents — "chat with the unmodified base" as a first-class list
    /// entry, without inventing artifacts on disk.
    public struct BaselineRow: Sendable, Equatable, Identifiable {
        public var baseModelID: String

        public var id: String { "baseline:\(baseModelID)" }
        public var name: String { "\(baseModelID) (baseline)" }

        public init(baseModelID: String) {
            self.baseModelID = baseModelID
        }
    }

    public static func baselineRows(
        for artifacts: [ModelVariantArtifact]
    ) -> [BaselineRow] {
        Set(artifacts.map(\.baseModelID))
            .sorted()
            .map(BaselineRow.init(baseModelID:))
    }

    // MARK: Filters

    public struct Filter: Sendable, Equatable {
        /// nil = all base models.
        public var baseModelID: String?
        public var sweepPromotedOnly: Bool
        public var hasAdapter: Bool
        public var runnableHere: Bool

        public init(
            baseModelID: String? = nil,
            sweepPromotedOnly: Bool = false,
            hasAdapter: Bool = false,
            runnableHere: Bool = false
        ) {
            self.baseModelID = baseModelID
            self.sweepPromotedOnly = sweepPromotedOnly
            self.hasAdapter = hasAdapter
            self.runnableHere = runnableHere
        }

        /// Provenance/composition filters cannot apply to a derived baseline
        /// row (it has no artifact) — such rows are excluded whenever one of
        /// those filters is on.
        public var excludesBaselines: Bool {
            sweepPromotedOnly || hasAdapter
        }
    }

    public static func matches(
        _ artifact: ModelVariantArtifact,
        filter: Filter,
        runnableHere: Bool
    ) -> Bool {
        if let base = filter.baseModelID, artifact.baseModelID != base {
            return false
        }
        if filter.sweepPromotedOnly, artifact.promotion == nil { return false }
        if filter.hasAdapter, artifact.adapters.isEmpty { return false }
        if filter.runnableHere, !runnableHere { return false }
        return true
    }

    public static func matches(
        baseline row: BaselineRow,
        filter: Filter,
        runnableHere: Bool
    ) -> Bool {
        guard !filter.excludesBaselines else { return false }
        if let base = filter.baseModelID, row.baseModelID != base { return false }
        if filter.runnableHere, !runnableHere { return false }
        return true
    }

    // MARK: Row summaries

    public static func componentsSummary(_ artifact: ModelVariantArtifact) -> String {
        var parts: [String] = []
        if !artifact.adapters.isEmpty {
            parts.append(
                "\(artifact.adapters.count) adapter\(artifact.adapters.count == 1 ? "" : "s")")
        }
        if !artifact.injections.isEmpty {
            parts.append(
                "\(artifact.injections.count) vector\(artifact.injections.count == 1 ? "" : "s")")
        }
        if let label = artifact.neutralPCBasisLabel {
            parts.append("neutral \(label)")
        }
        if parts.isEmpty { parts.append("base model settings only") }
        return parts.joined(separator: " · ")
    }
}
