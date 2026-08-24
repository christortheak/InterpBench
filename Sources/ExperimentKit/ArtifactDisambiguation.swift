import Foundation
import SteeringKit

/// Telling near-identically-named artifacts apart by what actually differs.
///
/// The lifecycle says iterate by DUPLICATING, never editing, so a workspace
/// accumulates `case-replication-with-fear-agent`, `…-2`, `…-2-2`; and a
/// concept re-extracted three times in an afternoon yields three vectors
/// whose names are identical and whose run directories differ only in
/// milliseconds. A timestamp does not help when the work happened on the same
/// day — it tells you WHICH came later, never WHY there are two.
///
/// The useful distinguisher is already computed elsewhere: two vectors with
/// the same `recipeIdentityHash` are interchangeable (that is exactly why
/// promote picks the newest among them without agonising), and two with
/// different identities differ in NAMEABLE fields — `RecipeIdentity.diffFields`
/// has produced that wording for refusals since 2026-07-14. This turns the
/// same comparison into a label.
///
/// Two cases are worth distinguishing sharply:
///
/// - **Different identity** — the artifacts genuinely differ; name the field.
///   "fear (method: meanDifference)" vs "fear (method: lat)".
/// - **Same identity** — they are the SAME recipe, so nothing about them
///   differs except when they were made. Saying "interchangeable" is more
///   useful than a hash, because the researcher's real question is "does it
///   matter which I pick?" and the answer is no.
public enum ArtifactDisambiguation {

    public struct Candidate: Sendable, Equatable {
        /// Stable identity for the artifact itself (path, run id).
        public var id: String
        /// What the researcher sees today — often colliding.
        public var displayName: String
        /// Canonical comparable fields. Values are pre-rendered strings so
        /// this stays free of any particular identity schema.
        public var identity: [String: String]
        /// Fallback ordering when identities are equal.
        public var createdAt: String?

        public init(
            id: String, displayName: String, identity: [String: String],
            createdAt: String? = nil
        ) {
            self.id = id
            self.displayName = displayName
            self.identity = identity
            self.createdAt = createdAt
        }
    }

    public struct Label: Sendable, Equatable {
        public var id: String
        public var displayName: String
        /// The differing fields that separate this one, already rendered.
        /// Empty when the name was already unique.
        public var distinguisher: String?
        /// This artifact shares its identity with another candidate — they
        /// are the same recipe and interchangeable.
        public var isInterchangeable: Bool

        /// One line for a picker row.
        public var text: String {
            guard let distinguisher else { return displayName }
            return "\(displayName) — \(distinguisher)"
        }

        public var help: String? {
            guard isInterchangeable else { return nil }
            return "identical recipe to another artifact here — they are "
                + "interchangeable, so it does not matter which you pick; "
                + "they differ only in when they were produced"
        }
    }

    /// Label every candidate, adding a distinguisher only where the name
    /// alone is ambiguous.
    public static func labels(_ candidates: [Candidate]) -> [Label] {
        var byName: [String: [Candidate]] = [:]
        for candidate in candidates {
            byName[candidate.displayName, default: []].append(candidate)
        }
        var labelled: [String: Label] = [:]
        for (_, group) in byName {
            guard group.count > 1 else {
                for candidate in group {
                    labelled[candidate.id] = Label(
                        id: candidate.id, displayName: candidate.displayName,
                        distinguisher: nil, isInterchangeable: false)
                }
                continue
            }
            for label in label(group: group) { labelled[label.id] = label }
        }
        return candidates.compactMap { labelled[$0.id] }
    }

    /// Label one ambiguous group.
    ///
    /// A group need not separate cleanly. Real workspaces contain families
    /// where SOME members differ and the rest are byte-identical in every
    /// compared field — observed live: `case-replication-with-fear-agent`
    /// differs from its two duplicates by having an agent attached, while
    /// `-2` and `-2-2` differ from each other in nothing at all. Labelling
    /// the whole group all-or-nothing would give those two the same text and
    /// leave the researcher exactly where they started, so members are
    /// handled per SIGNATURE: the ones a field separates get the field, and
    /// the ones nothing separates are told they are interchangeable.
    static func label(group: [Candidate]) -> [Label] {
        let keys = distinguishingKeys(group)
        func signature(_ candidate: Candidate) -> String {
            keys.map { candidate.identity[$0] ?? "" }.joined(separator: "\u{1}")
        }
        var counts: [String: Int] = [:]
        for candidate in group { counts[signature(candidate), default: 0] += 1 }

        return group.map { candidate in
            let rendered = keys.isEmpty
                ? nil
                : keys.map { "\($0): \(candidate.identity[$0] ?? "—")" }
                    .joined(separator: ", ")
            guard (counts[signature(candidate)] ?? 0) > 1 else {
                return Label(
                    id: candidate.id, displayName: candidate.displayName,
                    distinguisher: rendered, isInterchangeable: false)
            }
            // Nothing compared separates this one from a sibling. Say so, and
            // fall back to time only as an ordering aid — never as though it
            // were the reason they differ.
            var parts: [String] = []
            if let rendered { parts.append(rendered) }
            parts.append("identical to a sibling here")
            if let createdAt = candidate.createdAt { parts.append("made \(createdAt)") }
            return Label(
                id: candidate.id, displayName: candidate.displayName,
                distinguisher: parts.joined(separator: " · "),
                isInterchangeable: true)
        }
    }

    /// The smallest set of fields that separates a colliding group.
    ///
    /// Greedy rather than exact: minimum set cover is NP-hard, and with the
    /// handful of fields a recipe identity carries the greedy answer is
    /// either optimal or one field longer. A slightly longer label is a much
    /// better failure than a slow picker.
    ///
    /// Empty result = every field agrees, i.e. the artifacts are the same
    /// recipe.
    static func distinguishingKeys(_ group: [Candidate]) -> [String] {
        let allKeys = Set(group.flatMap { $0.identity.keys })
        let varying = allKeys.filter { key in
            Set(group.map { $0.identity[key] ?? "" }).count > 1
        }
        guard !varying.isEmpty else { return [] }

        func groupsRemaining(_ chosen: [String]) -> Int {
            Set(
                group.map { candidate in
                    chosen.map { candidate.identity[$0] ?? "" }.joined(separator: "\u{1}")
                }
            ).count
        }

        var chosen: [String] = []
        var pool = varying.sorted()
        while groupsRemaining(chosen) < group.count, !pool.isEmpty {
            // Pick the field that separates the most, ties broken by name so
            // the label is deterministic across runs and engines.
            let best = pool.max { a, b in
                let ca = groupsRemaining(chosen + [a])
                let cb = groupsRemaining(chosen + [b])
                return ca == cb ? a > b : ca < cb
            }
            guard let best else { break }
            chosen.append(best)
            pool.removeAll { $0 == best }
        }
        return chosen.sorted()
    }
}

extension ArtifactDisambiguation {

    /// Candidates for the vector catalog: name collisions are the norm
    /// (one concept, re-extracted), and the identity fields are the ones a
    /// researcher would actually name out loud.
    public static func vectorCandidates(
        _ artifacts: [VectorArtifact]
    ) -> [Candidate] {
        artifacts.map { artifact in
            let sidecar = artifact.sidecar
            var identity: [String: String] = [
                "model": sidecar.modelID,
                "method": sidecar.extractionMethod ?? "meanDifference",
            ]
            if let revision = sidecar.revision {
                identity["revision"] = String(revision.prefix(8))
            }
            identity["stimuli"] = String(sidecar.stimulusSetHash.prefix(8))
            if let reading = sidecar.readingPosition {
                identity["reading"] = reading
            }
            // Two artifacts of one concept can differ ONLY in how the
            // stimulus reached the model, so the rendering has to be a
            // disambiguating field — otherwise the chooser would call them
            // identical and pick the newest, which is exactly the silent win
            // this surface exists to prevent. Absent means legacy raw.
            identity["rendering"] =
                sidecar.extractionRendering?.mode.rawValue
                ?? ExtractionRendering.Mode.raw.rawValue
            if let projection = sidecar.neutralProjection {
                identity["projection"] = projection
            }
            if let source = sidecar.residualNormSource {
                identity["normSource"] = source
            }
            // The recipe hash is the LAST resort: it separates anything, but
            // says nothing about why. Included so a group is always
            // separable, and ranked last by the greedy chooser only because
            // every other field ties first.
            if let recipe = sidecar.recipeIdentityHash {
                identity["recipe"] = String(recipe.prefix(8))
            }
            return Candidate(
                id: artifact.id,
                displayName: sidecar.concept,
                identity: identity,
                // A sidecar can carry an EMPTY date but never a missing one
                // (the field is required, so a keyless sidecar fails decoding
                // and never reaches the catalog; "" is the legacy convention
                // for "unknown"). The run-directory name embeds the creation
                // timestamp, so it still orders the group.
                createdAt: sidecar.extractionDate.isEmpty
                    ? artifact.directory.lastPathComponent
                    : sidecar.extractionDate)
        }
    }

    /// Candidates for studies. Duplicating-to-iterate makes `x`, `x-2`,
    /// `x-2-2` — names that ARE distinct but carry no information about what
    /// changed, so the identity fields are the pinned design decisions a
    /// researcher would compare.
    public static func studyCandidates(
        _ manifests: [ExperimentManifest]
    ) -> [Candidate] {
        manifests.map { manifest in
            var identity: [String: String] = [
                "model": manifest.modelID,
                "status": manifest.status.rawValue,
            ]
            if let revision = manifest.modelRevision {
                identity["revision"] = String(revision.prefix(8))
            }
            let concepts = manifest.concepts.map(\.name).sorted()
            if !concepts.isEmpty {
                identity["concepts"] = concepts.joined(separator: "+")
            }
            let agents = manifest.variantConditions.map(\.name).sorted()
            if !agents.isEmpty {
                identity["agents"] = agents.joined(separator: "+")
            }
            if let instruments = manifest.outcomeInstruments, !instruments.isEmpty {
                identity["instruments"] = instruments.sorted().joined(separator: "+")
            }
            if let prompts = manifest.taskPromptsFile, !prompts.isEmpty {
                identity["prompts"] = URL(filePath: prompts).lastPathComponent
            }
            identity["conditions"] = String(manifest.conditions.count)
            return Candidate(
                id: manifest.name,
                displayName: manifest.name,
                identity: identity,
                createdAt: manifest.frozenAt)
        }
    }

    /// Studies whose names differ only by the duplicate suffix, grouped under
    /// the stem they were duplicated from.
    ///
    /// `case-replication-with-fear-agent`, `…-2` and `…-2-2` are three
    /// DISTINCT names, so the collision rule above never fires for them — yet
    /// they are exactly the set a researcher cannot tell apart. Grouping by
    /// stem lets the same field-diff run over them.
    public static func duplicateFamilies(
        _ manifests: [ExperimentManifest]
    ) -> [String: [Candidate]] {
        var families: [String: [Candidate]] = [:]
        for candidate in studyCandidates(manifests) {
            families[duplicateStem(candidate.displayName), default: []]
                .append(candidate)
        }
        return families.filter { $0.value.count > 1 }
    }

    /// Strips trailing `-<digits>` groups, the shape `duplicate` appends.
    static func duplicateStem(_ name: String) -> String {
        var stem = name[...]
        while let range = stem.range(of: #"-\d+$"#, options: .regularExpression) {
            stem = stem[..<range.lowerBound]
        }
        return String(stem)
    }

    /// Label a duplicate family by what actually differs between its members.
    public static func familyLabels(
        _ manifests: [ExperimentManifest]
    ) -> [String: [Label]] {
        duplicateFamilies(manifests).mapValues { label(group: $0) }
    }
}

extension ArtifactDisambiguation {

    /// Comparable fields for an agent.
    ///
    /// Promoting one concept twice — a re-sweep, a different cell, a manual
    /// override — produces same-named agents routinely, and the cell is
    /// almost always what a researcher means when they ask "which one is
    /// this?". The promotion's own birth certificate answers it, so the
    /// fields come from there when present.
    public static func agentIdentity(
        _ artifact: ModelVariantArtifact
    ) -> [String: String] {
        var identity: [String: String] = ["model": artifact.baseModelID]
        if let revision = artifact.baseRevision {
            identity["revision"] = String(revision.prefix(8))
        }
        let injections = artifact.injections
        if !injections.isEmpty {
            identity["concepts"] = injections.map(\.concept).sorted()
                .joined(separator: "+")
            identity["cell"] = injections
                .map {
                    "L\($0.layer)α"
                        + $0.alpha.formatted(.number.precision(.fractionLength(0 ... 4)))
                }
                .joined(separator: "/")
        }
        if !artifact.adapters.isEmpty {
            identity["adapters"] = artifact.adapters.map(\.name).sorted()
                .joined(separator: "+")
        }
        if let promotion = artifact.promotion {
            // How it was chosen is a first-class difference: a
            // criterion-selected agent and a hand-overridden one at the same
            // cell are NOT the same evidence.
            identity["promotedBy"] = promotion.promotedBy
            if let sweepRun = promotion.sweepRun {
                identity["sweep"] = sweepRun
            }
        } else {
            identity["promotedBy"] = "handCreated"
        }
        return identity
    }
}
