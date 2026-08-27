import Foundation
import SteeringKit

/// Agents → New Agent → "Optimize from concept vector": one form that
/// creates the real optimization manifest itself, so agent creation never
/// detours through Studies. The composer pins vector RECIPES, not vector
/// bytes — the user picks artifacts from the active substrate, and each
/// sidecar/catalog record names the recipe (concept + CURRENT stimulus hash + extraction
/// options) the manifest pins; runs re-derive vectors deterministically
/// (CLAUDE.md › Experiment lifecycle).
///
/// Everything except `declare` is pure logic over plain values so the
/// refusal rules (mixed models, missing stimulus data, unmappable recipes,
/// legacy sidecars) are unit-testable without a workspace. Refusals exist
/// ONLY where pinning would be dishonest; advisory conditions (stimulus
/// drift, non-comparable corpus hashes) surface loudly but never gate.
public enum OptimizationComposer {

    // MARK: - Refusals

    /// A refusal in user vocabulary plus the precise technical detail views
    /// put in a tooltip. The message says what happened and what the user
    /// can do about it; the detail names the exact sidecar field or
    /// workspace path involved so nothing precise is lost — it just is not
    /// the headline.
    public struct Refusal: Sendable, Equatable {
        public var message: String
        public var detail: String?

        public init(_ message: String, detail: String? = nil) {
            self.message = message
            self.detail = detail
        }
    }

    // MARK: - Method mapping

    /// Mapping a sidecar's recorded method stamps onto the manifest's
    /// `ExtractionMethod` vocabulary. The newer `recipeMethod` stamp wins
    /// when present (it distinguishes grand-mean artifacts even when legacy
    /// fields coexist); a sidecar with NO recorded method refuses — a recipe
    /// field that selection provenance depends on is never silently
    /// defaulted.
    public enum MethodMapping: Equatable, Sendable {
        case mapped(ExtractionMethod)
        case refused(Refusal)
    }

    private static let readerRefusal = Refusal(
        "this vector was converted from a fitted reading probe, not "
            + "extracted from stimulus data — its recipe can't be reproduced; "
            + "extract a CAA, LAT, or grand-mean vector for this concept in "
            + "Data instead",
        detail: "Sidecar method repeReaderLAT: a reader conversion is not a "
            + "stimulus-set recipe the manifest can pin.")

    /// A J-lens direction is DERIVED from a lens plus a token id, so like a
    /// reader conversion it has no stimulus-set recipe a manifest could pin and
    /// re-derive. Refused for the same reason, with its own wording: telling a
    /// researcher to "re-extract it in Data" would be wrong advice here — the
    /// artifact is reproducible, just not as an extraction.
    private static let jlensRefusal = Refusal(
        "this vector was derived from a J-lens token direction, not extracted "
            + "from stimulus data — an optimizer can't re-derive it as a "
            + "recipe; sweep layer and alpha for it as a fixed vector, or "
            + "extract a CAA, LAT, or grand-mean vector for this concept "
            + "instead",
        detail: "Sidecar method jlensTokenDirection: the identity is a lens "
            + "plus an exact token id, not a stimulus set.")

    public static func mapMethod(
        recipeMethodRaw: String?, extractionMethodRaw: String?
    ) -> MethodMapping {
        if let raw = recipeMethodRaw {
            switch VectorExtractionRecipe.Method(rawValue: raw) {
            case .caaMeanDifference:
                return .mapped(.meanDifference)
            case .pairedDifferencePCA:
                return .mapped(.pairedDifferencePCA)
            case .emotionGrandMean:
                return .mapped(.emotionGrandMean)
            case .repeReaderLAT:
                return .refused(readerRefusal)
            case .jlensTokenDirection:
                return .refused(jlensRefusal)
            case nil:
                return .refused(
                    Refusal(
                        "this vector's saved recipe ('\(raw)') isn't one this "
                            + "app can reproduce — re-extract it in Data to "
                            + "make it optimizable",
                        detail: "Sidecar recipeMethod '\(raw)' does not map "
                            + "onto a manifest extraction recipe."))
            }
        }
        if let raw = extractionMethodRaw {
            if raw == "repeReaderLAT" {
                return .refused(readerRefusal)
            }
            if raw == "jlensTokenDirection" {
                return .refused(jlensRefusal)
            }
            if let method = ExtractionMethod(rawValue: raw) {
                return .mapped(method)
            }
            return .refused(
                Refusal(
                    "this vector's saved extraction method ('\(raw)') isn't "
                        + "one this app can reproduce — re-extract it in Data "
                        + "to make it optimizable",
                    detail: "Sidecar extractionMethod '\(raw)' does not map "
                        + "onto a manifest extraction recipe."))
        }
        return .refused(
            Refusal(
                "this vector predates recipe recording (no extraction method "
                    + "saved) — re-extract it in Data to make it optimizable",
                detail: "The sidecar records neither recipeMethod nor "
                    + "extractionMethod; recipe pinning will not guess a "
                    + "field the selection provenance depends on."))
    }

    // MARK: - Artifact assessment

    /// Plain values distilled from one artifact's sidecar plus the current
    /// workspace truth about its stimulus data. `assess` is pure over this.
    public struct ArtifactFacts: Sendable, Equatable {
        public var artifactID: String
        public var concept: String
        public var modelID: String
        public var recordedStimulusHash: String
        public var recipeMethodRaw: String?
        public var extractionMethodRaw: String?
        public var readingPositionLabel: String?
        /// Grand-mean corpus membership recorded at extraction
        /// (`sidecar.comparisonConcepts`).
        public var grandMeanCorpusConcepts: [String]?
        /// CURRENT hash of `prompts/concepts/<concept>/` (paired recipes);
        /// nil = no readable stimulus directory in this workspace.
        public var currentPairedStimulusHash: String?
        /// CURRENT hash of `prompts/emotions/<concept>/stories.jsonl`
        /// (grand-mean recipes); nil = no stories file.
        public var currentStoriesHash: String?
        /// Recorded corpus members whose stories.jsonl is missing locally.
        public var missingCorpusMembers: [String]
        /// designatedReference only: the reference corpus name recorded at
        /// extraction (`sidecar.designatedReference.name`).
        public var designatedReferenceName: String?
        /// designatedReference only: the reference stories hash recorded at
        /// extraction.
        public var recordedReferenceHash: String?
        /// CURRENT hash of the reference's stories.jsonl; nil = no such
        /// file in this workspace.
        public var currentReferenceHash: String?

        public init(
            artifactID: String,
            concept: String,
            modelID: String,
            recordedStimulusHash: String,
            recipeMethodRaw: String? = nil,
            extractionMethodRaw: String? = nil,
            readingPositionLabel: String? = nil,
            grandMeanCorpusConcepts: [String]? = nil,
            currentPairedStimulusHash: String? = nil,
            currentStoriesHash: String? = nil,
            missingCorpusMembers: [String] = [],
            designatedReferenceName: String? = nil,
            recordedReferenceHash: String? = nil,
            currentReferenceHash: String? = nil
        ) {
            self.artifactID = artifactID
            self.concept = concept
            self.modelID = modelID
            self.recordedStimulusHash = recordedStimulusHash
            self.recipeMethodRaw = recipeMethodRaw
            self.extractionMethodRaw = extractionMethodRaw
            self.readingPositionLabel = readingPositionLabel
            self.grandMeanCorpusConcepts = grandMeanCorpusConcepts
            self.currentPairedStimulusHash = currentPairedStimulusHash
            self.currentStoriesHash = currentStoriesHash
            self.missingCorpusMembers = missingCorpusMembers
            self.designatedReferenceName = designatedReferenceName
            self.recordedReferenceHash = recordedReferenceHash
            self.currentReferenceHash = currentReferenceHash
        }
    }

    /// The recipe one artifact names, resolved against current workspace
    /// data — everything `declare` needs to pin the concept the way the
    /// engine verifies it.
    public struct ConceptPin: Sendable, Equatable {
        public var concept: String
        public var modelID: String
        public var method: ExtractionMethod
        public var readingPosition: ReadingPosition
        /// CURRENT stimulus hash (freeze verify demands file truth; declare
        /// recomputes it at pin time — this value is the assessment-time
        /// reading for display).
        public var currentStimulusHash: String
        public var recordedStimulusHash: String
        /// Grand-mean only: the pinned population (always includes the
        /// target).
        public var corpusConcepts: [String]
        /// Grand-mean only: the pooled reading start token.
        public var poolFromToken: Int?
        /// designatedReference only: the reference corpus name — part of
        /// the recipe (mean(concept stories) − mean(REFERENCE stories)).
        public var designatedReferenceName: String?
        /// designatedReference only: the CURRENT reference stories hash the
        /// declaration will pin.
        public var currentReferenceHash: String?
        /// Loud advisory (stimulus drift) — never a gate.
        public var caution: String?

        /// The FULL recipe identity, canonical: every field that makes two
        /// artifacts interchangeable FOR PINNING — including the sorted
        /// grand-mean corpus membership, pooling token, and the designated
        /// reference (name + current hash), because population and
        /// reference ARE part of the recipe. The composer's duplicate
        /// collapse keys on THIS, never a hand-selected subset (review
        /// 2026-08-02 round 4, P1: a subset key omitted `corpusConcepts`
        /// and could merge grand-mean artifacts built from different
        /// populations).
        ///
        /// `recordedStimulusHash` is deliberately NOT identity: the
        /// declaration pins CURRENT bytes and the sweep re-derives from
        /// them, so two artifacts extracted before and after a stimulus
        /// edit pin the identical study — listing both as different
        /// recipes was exactly the "duplicates with a drift caution"
        /// confusion (field report 2026-08-02). `caution` is display
        /// state, not identity, and stays out too.
        public var recipeIdentity: String {
            ([
                concept, modelID, "\(method)", "\(readingPosition)",
                currentStimulusHash,
                poolFromToken.map(String.init) ?? "-",
                designatedReferenceName ?? "-",
                currentReferenceHash ?? "-",
            ] + corpusConcepts.sorted()).joined(separator: "|")
        }

        public init(
            concept: String,
            modelID: String,
            method: ExtractionMethod,
            readingPosition: ReadingPosition,
            currentStimulusHash: String,
            recordedStimulusHash: String,
            corpusConcepts: [String] = [],
            poolFromToken: Int? = nil,
            designatedReferenceName: String? = nil,
            currentReferenceHash: String? = nil,
            caution: String? = nil
        ) {
            self.concept = concept
            self.modelID = modelID
            self.method = method
            self.readingPosition = readingPosition
            self.currentStimulusHash = currentStimulusHash
            self.recordedStimulusHash = recordedStimulusHash
            self.corpusConcepts = corpusConcepts
            self.poolFromToken = poolFromToken
            self.designatedReferenceName = designatedReferenceName
            self.currentReferenceHash = currentReferenceHash
            self.caution = caution
        }
    }

    public enum ArtifactVerdict: Sendable, Equatable {
        case pinnable(ConceptPin)
        case refused(Refusal)
    }

    /// Disk-backed facts for a catalog artifact: the sidecar's recorded
    /// recipe stamps plus the CURRENT hashes of the stimulus data those
    /// stamps name.
    public static func facts(for artifact: VectorArtifact) -> ArtifactFacts {
        let sidecar = artifact.sidecar
        return facts(
            artifactID: artifact.id,
            concept: sidecar.concept,
            modelID: sidecar.modelID,
            recordedStimulusHash: sidecar.stimulusSetHash,
            recipeMethodRaw: sidecar.recipeMethod,
            extractionMethodRaw: sidecar.extractionMethod,
            readingPositionLabel: sidecar.readingPosition,
            grandMeanCorpusConcepts: sidecar.comparisonConcepts
                ?? sidecar.grandMeanPopulation.map {
                    Array($0.keys).sorted()
                },
            designatedReferenceName: sidecar.designatedReference?["name"],
            recordedReferenceHash: sidecar.designatedReference?["hash"])
    }

    /// Server-catalog twin of ``facts(for:)``. A remote vector is only the
    /// researcher's entry point: optimization still resolves the recipe
    /// against the CURRENT workspace stimulus files and pins those inputs.
    /// The remote artifact path is provenance/display identity, never vector
    /// bytes imported across substrates.
    public static func facts(for record: RemoteVectorRecord) -> ArtifactFacts {
        facts(
            artifactID: record.id,
            concept: record.concept,
            modelID: record.modelID,
            recordedStimulusHash: record.stimulusSetHash ?? "",
            recipeMethodRaw: record.recipeMethod,
            extractionMethodRaw: record.extractionMethod ?? record.method,
            readingPositionLabel: record.resolvedReadingPosition,
            grandMeanCorpusConcepts: record.comparisonConcepts
                ?? record.grandMeanPopulation.map { Array($0.keys).sorted() },
            designatedReferenceName: record.designatedReference?["name"],
            recordedReferenceHash: record.designatedReference?["hash"])
    }

    private static func facts(
        artifactID: String,
        concept: String,
        modelID: String,
        recordedStimulusHash: String,
        recipeMethodRaw: String?,
        extractionMethodRaw: String?,
        readingPositionLabel: String?,
        grandMeanCorpusConcepts: [String]?,
        designatedReferenceName: String? = nil,
        recordedReferenceHash: String? = nil
    ) -> ArtifactFacts {
        let conceptDirectory = VectorCatalog.conceptsDirectory.appending(
            component: concept)
        let pairedHash = (try? StimulusSet(directory: conceptDirectory))?.hash
        let storiesHash = ExperimentStore.storiesHash(for: concept)
        let members = grandMeanCorpusConcepts ?? []
        let missingMembers = members.filter {
            ExperimentStore.storiesHash(for: $0) == nil
        }
        return ArtifactFacts(
            artifactID: artifactID,
            concept: concept,
            modelID: modelID,
            recordedStimulusHash: recordedStimulusHash,
            recipeMethodRaw: recipeMethodRaw,
            extractionMethodRaw: extractionMethodRaw,
            readingPositionLabel: readingPositionLabel,
            grandMeanCorpusConcepts: grandMeanCorpusConcepts,
            currentPairedStimulusHash: pairedHash,
            currentStoriesHash: storiesHash,
            missingCorpusMembers: missingMembers,
            designatedReferenceName: designatedReferenceName,
            recordedReferenceHash: recordedReferenceHash,
            currentReferenceHash: designatedReferenceName.flatMap {
                ExperimentStore.storiesHash(for: $0)
            })
    }

    /// Can this artifact name a recipe the manifest could pin honestly?
    /// Pure over `ArtifactFacts`. Refusals only where pinning would be
    /// dishonest; hash drift is a loud caution on a pinnable verdict.
    public static func assess(_ facts: ArtifactFacts) -> ArtifactVerdict {
        guard !facts.recordedStimulusHash.isEmpty else {
            return .refused(
                Refusal(
                    "this server catalog does not expose the vector's source "
                        + "data hash — refresh after updating the server, or "
                        + "re-extract it in Data",
                    detail: "Optimization pins a reproducible recipe rather "
                        + "than vector bytes; a missing stimulusSetHash cannot "
                        + "be guessed."))
        }
        let method: ExtractionMethod
        switch mapMethod(
            recipeMethodRaw: facts.recipeMethodRaw,
            extractionMethodRaw: facts.extractionMethodRaw)
        {
        case .refused(let refusal):
            return .refused(refusal)
        case .mapped(let mapped):
            method = mapped
        }
        guard let label = facts.readingPositionLabel else {
            return .refused(
                Refusal(
                    "this vector predates recipe recording (no reading "
                        + "position saved) — re-extract it in Data to make it "
                        + "optimizable",
                    detail: "The sidecar records no readingPosition — a "
                        + "recipe field the manifest pins cannot be guessed."))
        }
        guard let reading = ReadingPosition(label: label) else {
            return .refused(
                Refusal(
                    "this vector's saved reading position ('\(label)') isn't "
                        + "one this app can reproduce — re-extract it in Data "
                        + "to make it optimizable",
                    detail: "Sidecar readingPosition '\(label)' did not "
                        + "parse; refusing to guess a recipe field the "
                        + "selection provenance depends on."))
        }
        if method == .emotionGrandMean {
            return assessGrandMean(facts, reading: reading, label: label)
        }
        if method == .designatedReference {
            // mean(concept stories) − mean(REFERENCE stories): the recipe
            // reads prompts/emotions/, never prompts/concepts/ — falling
            // through to the paired path here is what falsely refused every
            // stance vector as "no stimulus data" (field diagnosis
            // 2026-08-02).
            return assessDesignatedReference(facts, reading: reading)
        }
        guard let current = facts.currentPairedStimulusHash else {
            return .refused(
                Refusal(
                    "no stimulus data for '\(facts.concept)' in this "
                        + "workspace — the recipe can't be pinned here",
                    detail: "Expected prompts/concepts/\(facts.concept)/ "
                        + "(positive.jsonl + negative.jsonl). Recipe pinning "
                        + "re-derives vectors from stimulus files, so an "
                        + "imported or feature vector without stimulus data "
                        + "cannot seed an optimization."))
        }
        var caution: String?
        if current != facts.recordedStimulusHash {
            caution =
                "'\(facts.concept)': stimulus data changed since this vector "
                + "was extracted (recorded "
                + "\(facts.recordedStimulusHash.prefix(12))…, current "
                + "\(current.prefix(12))…) — the optimization re-derives "
                + "vectors from the CURRENT data"
        }
        return .pinnable(
            ConceptPin(
                concept: facts.concept,
                modelID: facts.modelID,
                method: method,
                readingPosition: reading,
                currentStimulusHash: current,
                recordedStimulusHash: facts.recordedStimulusHash,
                caution: caution))
    }

    private static func assessDesignatedReference(
        _ facts: ArtifactFacts, reading: ReadingPosition
    ) -> ArtifactVerdict {
        guard let current = facts.currentStoriesHash else {
            return .refused(
                Refusal(
                    "no stories data for '\(facts.concept)' in this "
                        + "workspace — the recipe can't be pinned here",
                    detail: "Expected prompts/emotions/\(facts.concept)/"
                        + "stories.jsonl — the designated-reference recipe "
                        + "re-derives vectors from the target stories minus "
                        + "the reference stories."))
        }
        guard let referenceName = facts.designatedReferenceName,
            !referenceName.isEmpty
        else {
            return .refused(
                Refusal(
                    "this vector doesn't record its designated reference — "
                        + "refresh the server catalog after updating the "
                        + "server, or re-extract it in Data",
                    detail: "The reference corpus is part of the recipe "
                        + "(mean(concept stories) − mean(reference stories)) "
                        + "and cannot be defaulted; older server catalogs "
                        + "omitted the sidecar's designatedReference field."))
        }
        guard let currentReference = facts.currentReferenceHash else {
            return .refused(
                Refusal(
                    "this vector's designated reference '\(referenceName)' "
                        + "has no stimulus data in this workspace, so the "
                        + "recipe can't be pinned here",
                    detail: "Expected prompts/emotions/\(referenceName)/"
                        + "stories.jsonl — the reference corpus is pinned "
                        + "beside the target and must exist to re-derive."))
        }
        var cautions: [String] = []
        if current != facts.recordedStimulusHash {
            cautions.append(
                "'\(facts.concept)': stimulus data changed since this vector "
                    + "was extracted (recorded "
                    + "\(facts.recordedStimulusHash.prefix(12))…, current "
                    + "\(current.prefix(12))…) — the optimization re-derives "
                    + "vectors from the CURRENT data")
        }
        if let recorded = facts.recordedReferenceHash,
            recorded != currentReference
        {
            cautions.append(
                "'\(facts.concept)': reference '\(referenceName)' stories "
                    + "changed since extraction (recorded "
                    + "\(recorded.prefix(12))…, current "
                    + "\(currentReference.prefix(12))…) — the optimization "
                    + "re-derives from the CURRENT reference")
        }
        return .pinnable(
            ConceptPin(
                concept: facts.concept,
                modelID: facts.modelID,
                method: .designatedReference,
                readingPosition: reading,
                currentStimulusHash: current,
                recordedStimulusHash: facts.recordedStimulusHash,
                designatedReferenceName: referenceName,
                currentReferenceHash: currentReference,
                caution: cautions.isEmpty
                    ? nil : cautions.joined(separator: "\n")))
    }

    private static func assessGrandMean(
        _ facts: ArtifactFacts, reading: ReadingPosition, label: String
    ) -> ArtifactVerdict {
        guard let current = facts.currentStoriesHash else {
            return .refused(
                Refusal(
                    "no stimulus data for '\(facts.concept)' in this "
                        + "workspace — the recipe can't be pinned here",
                    detail: "Expected prompts/emotions/\(facts.concept)/"
                        + "stories.jsonl — the grand-mean recipe re-derives "
                        + "vectors from the stories data."))
        }
        let members = facts.grandMeanCorpusConcepts ?? []
        guard members.contains(where: { $0 != facts.concept }) else {
            return .refused(
                Refusal(
                    "this vector doesn't record which concepts made up its "
                        + "grand-mean corpus, so its recipe can't be "
                        + "reproduced — rebuild it in Data to make it "
                        + "optimizable",
                    detail: "The sidecar's comparisonConcepts field names no "
                        + "concept beyond the target — the population the "
                        + "grand mean was computed over is part of the recipe "
                        + "and cannot be defaulted."))
        }
        guard facts.missingCorpusMembers.isEmpty else {
            return .refused(
                Refusal(
                    "this vector's grand-mean corpus includes "
                        + facts.missingCorpusMembers.joined(separator: ", ")
                        + " — no stimulus data for that in this workspace, so "
                        + "the recipe can't be pinned here",
                    detail: "No stories.jsonl under prompts/emotions/"
                        + "<concept>/ for the named corpus member(s); the "
                        + "pinned population must exist for the recipe to "
                        + "re-derive."))
        }
        guard case .meanFromToken(let token) = reading else {
            return .refused(
                Refusal(
                    "this vector reads at '\(label)', which a grand-mean "
                        + "optimization can't reproduce — re-extract it in "
                        + "Data with a pooled reading (mean from token k)",
                    detail: "The manifest's grand-mean pin helper expresses "
                        + "only pooled reading positions; refusing to "
                        + "mispin."))
        }
        var caution: String?
        if current != facts.recordedStimulusHash {
            caution =
                "'\(facts.concept)': the current stories data differs from "
                + "this vector's recorded corpus hash (recorded "
                + "\(facts.recordedStimulusHash.prefix(12))…, current stories "
                + "\(current.prefix(12))…) — grand-mean artifacts hash their "
                + "selected build rows, so a difference is expected; the "
                + "optimization re-derives vectors from the CURRENT pinned "
                + "stories data"
        }
        return .pinnable(
            ConceptPin(
                concept: facts.concept,
                modelID: facts.modelID,
                method: .emotionGrandMean,
                readingPosition: reading,
                currentStimulusHash: current,
                recordedStimulusHash: facts.recordedStimulusHash,
                corpusConcepts: members,
                poolFromToken: token,
                caution: caution))
    }

    // MARK: - Plan

    public struct Plan: Sendable, Equatable {
        public var modelID: String
        public var pins: [ConceptPin]
        public var cautions: [String]

        public init(modelID: String, pins: [ConceptPin], cautions: [String]) {
            self.modelID = modelID
            self.pins = pins
            self.cautions = cautions
        }
    }

    /// Structural problem with the selected pins, or nil when a plan can be
    /// made. Refusals only where pinning would be dishonest.
    public static func planProblem(_ pins: [ConceptPin]) -> String? {
        guard !pins.isEmpty else {
            return "select at least one optimizable vector"
        }
        let models = Set(pins.map(\.modelID))
        if models.count > 1 {
            return "selected vectors come from different models ("
                + models.sorted().joined(separator: ", ")
                + ") — one optimization pins ONE base model; select artifacts "
                + "extracted from a single model"
        }
        let duplicates = Dictionary(grouping: pins, by: \.concept)
            .filter { $0.value.count > 1 }
            .keys.sorted()
        if !duplicates.isEmpty {
            return "concept(s) " + duplicates.joined(separator: ", ")
                + " selected more than once — an optimization pins ONE recipe "
                + "per concept name; pick a single artifact per concept"
        }
        return nil
    }

    public static func makePlan(_ pins: [ConceptPin]) throws -> Plan {
        if let problem = planProblem(pins) {
            throw ExperimentError(reason: problem)
        }
        return Plan(
            modelID: pins[0].modelID,
            pins: pins,
            cautions: pins.compactMap(\.caution))
    }

    // MARK: - Model availability

    /// Substrate scoping for the vector picker: split artifacts by whether
    /// their base model is in the active workspace's installed-model list.
    /// Display/selection filtering ONLY — never a new pin gate: an artifact
    /// for an unavailable model is un-selectable because the sweep loads
    /// that model to re-derive vectors and generate, which would fail on
    /// this substrate. Order is preserved within each half.
    ///
    /// An EMPTY `availableModels` list means the inventory is unknown
    /// (e.g. a server workspace before the server reports its models) —
    /// scoping would then be a dishonest guess, so everything stays
    /// available and the engine's own model-load error remains the
    /// backstop.
    public static func partitionByModelAvailability<Element>(
        _ artifacts: [Element],
        availableModels: [String],
        modelID: (Element) -> String
    ) -> (available: [Element], unavailable: [Element]) {
        guard !availableModels.isEmpty else { return (artifacts, []) }
        let installed = Set(availableModels)
        var available: [Element] = []
        var unavailable: [Element] = []
        for artifact in artifacts {
            if installed.contains(modelID(artifact)) {
                available.append(artifact)
            } else {
                unavailable.append(artifact)
            }
        }
        return (available, unavailable)
    }

    /// Catalog convenience over the sidecar's recorded base model.
    public static func partitionByModelAvailability(
        _ artifacts: [VectorArtifact], availableModels: [String]
    ) -> (available: [VectorArtifact], unavailable: [VectorArtifact]) {
        partitionByModelAvailability(
            artifacts, availableModels: availableModels,
            modelID: { $0.sidecar.modelID })
    }

    // MARK: - Name suggestion

    /// "optimize-<firstConcept>-<yyyy-MM-dd>", pre-sanitized the same way
    /// `ExperimentStore.create` sanitizes, so the suggestion IS the name the
    /// store will use.
    public static func suggestedName(
        firstConcept: String?, date: Date = Date()
    ) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyy-MM-dd"
        let sanitized = (firstConcept ?? "").lowercased()
            .replacingOccurrences(of: " ", with: "-")
            .filter { $0.isLetter || $0.isNumber || $0 == "-" }
        let stem = sanitized.isEmpty ? "concept" : sanitized
        return "optimize-\(stem)-\(formatter.string(from: date))"
    }

    // MARK: - Instrument scan

    /// Candidate instrument files: `.jsonl` under the workspace's `prompts/`
    /// (recursively, small depth cap), as workspace-relative paths — so dev
    /// prompts, capability batteries, and choice prompts become pickers over
    /// real files. Pure over the given root; sorted for stable UI.
    public static func scanInstrumentFiles(
        root: URL = VectorCatalog.projectRoot, maxDepth: Int = 4
    ) -> [String] {
        let fm = FileManager.default
        var results: [String] = []
        func walk(_ directory: URL, depth: Int, relative: String) {
            guard depth <= maxDepth else { return }
            guard
                let entries = try? fm.contentsOfDirectory(
                    at: directory,
                    includingPropertiesForKeys: [.isDirectoryKey],
                    options: [.skipsHiddenFiles])
            else { return }
            for entry in entries {
                let name = entry.lastPathComponent
                let isDirectory =
                    (try? entry.resourceValues(forKeys: [.isDirectoryKey])
                        .isDirectory) == true
                if isDirectory {
                    walk(entry, depth: depth + 1, relative: relative + name + "/")
                } else if entry.pathExtension == "jsonl" {
                    results.append("prompts/" + relative + name)
                }
            }
        }
        walk(root.appending(component: "prompts"), depth: 0, relative: "")
        return results.sorted()
    }

    // MARK: - Declare

    /// Creates the real optimization manifest: create the draft study, stamp
    /// the same defaults `ExperimentPanel.create()` stamps, attach every
    /// concept pin at its CURRENT hash (recomputed here — freeze verify
    /// demands file truth), pin the neutral corpus (norm denominator), pin
    /// judge rubric + judges for a judgeScore objective, save, and THEN set
    /// the sweep spec through `ExperimentPanel.setSweepSpec` so the
    /// criterion validation (judge/choice pins) runs exactly the engine's
    /// way, in an order where the pins already exist.
    ///
    /// The criterion travels VERBATIM in `spec.selection` — never hashed.
    /// Returns the created (sanitized) study name.
    @MainActor
    @discardableResult
    public static func declare(
        name: String,
        description: String,
        plan: Plan,
        spec: ExperimentManifest.SweepSpec,
        judgeRubricFile: String?,
        judges: [ExperimentManifest.JudgeRef],
        panel: ExperimentPanel
    ) throws -> String {
        let metric = spec.selection?.objective?.metric
            .trimmingCharacters(in: .whitespaces)
        guard let metric, !metric.isEmpty else {
            throw ExperimentError(
                reason: "declare requires an explicitly chosen selection "
                    + "objective — the criterion is pre-declared data, never "
                    + "an implied default")
        }
        if let problem = planProblem(plan.pins) {
            throw ExperimentError(reason: problem)
        }
        // Pre-flight every throwing read BEFORE creating the manifest, so a
        // missing file cannot leave a half-declared study behind. Paired
        // hashes are recomputed here — the pin is file truth at declare
        // time, not the assessment-time reading.
        var pairedHashes: [String: String] = [:]
        // designatedReference: {target stories hash, reference stories hash}
        // recomputed at declare time (file truth, like the paired hashes).
        var referencePins: [String: (target: String, reference: String)] = [:]
        for pin in plan.pins {
            if pin.method == .emotionGrandMean {
                for member in Set(pin.corpusConcepts + [pin.concept])
                where ExperimentStore.storiesHash(for: member) == nil {
                    throw ExperimentError(
                        reason: "no stories.jsonl for grand-mean corpus member "
                            + "'\(member)' under prompts/emotions/")
                }
            } else if pin.method == .designatedReference {
                // The stance recipe reads prompts/emotions/, never
                // prompts/concepts/ — the paired preflight below would
                // falsely refuse it (field diagnosis 2026-08-02).
                guard let referenceName = pin.designatedReferenceName,
                    !referenceName.isEmpty
                else {
                    throw ExperimentError(
                        reason: "designatedReference pin for '\(pin.concept)' "
                            + "names no reference corpus — the reference is "
                            + "part of the recipe")
                }
                guard let targetHash = ExperimentStore.storiesHash(
                    for: pin.concept)
                else {
                    throw ExperimentError(
                        reason: "no stories.jsonl for concept "
                            + "'\(pin.concept)' under prompts/emotions/")
                }
                guard let referenceHash = ExperimentStore.storiesHash(
                    for: referenceName)
                else {
                    throw ExperimentError(
                        reason: "no stories.jsonl for reference "
                            + "'\(referenceName)' under prompts/emotions/")
                }
                referencePins[pin.concept] = (targetHash, referenceHash)
            } else {
                let directory = VectorCatalog.conceptsDirectory.appending(
                    component: pin.concept)
                pairedHashes[pin.concept] = try StimulusSet(
                    directory: directory).hash
            }
        }
        var cleanedJudges: [ExperimentManifest.JudgeRef] = []
        if metric == "judgeScore" {
            let rubric = (judgeRubricFile ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !rubric.isEmpty else {
                throw ExperimentError(
                    reason: "judgeScore objective needs a rubric file to pin "
                        + "(prompts/rubrics/)")
            }
            cleanedJudges = judges
                .map { judge in
                    ExperimentManifest.JudgeRef(
                        name: judge.name.trimmingCharacters(
                            in: .whitespacesAndNewlines),
                        kind: judge.kind,
                        model: judge.model.flatMap {
                            let trimmed = $0.trimmingCharacters(
                                in: .whitespacesAndNewlines)
                            return trimmed.isEmpty ? nil : trimmed
                        },
                        // The provider is a PIN (openrouter judges) — a
                        // declare that drops it invalidates the judge
                        // (2026-07-19).
                        provider: judge.provider.flatMap {
                            let trimmed = $0.trimmingCharacters(
                                in: .whitespacesAndNewlines)
                            return trimmed.isEmpty ? nil : trimmed
                        })
                        // Judge write funnels serialize only kind-owned
                        // fields (field bug 2026-08-07) — this path never
                        // carried revision/dtype, but the filter keeps the
                        // rule in one place for every caller.
                        .keepingKindOwnedFields()
                }
                .filter { !$0.name.isEmpty }
            guard !cleanedJudges.isEmpty else {
                throw ExperimentError(
                    reason: "judgeScore objective needs at least one judge")
            }
        }

        // A name collision (or empty name) surfaces the store's own error
        // verbatim.
        var manifest = try ExperimentStore.create(
            name: name, description: description, modelID: plan.modelID)
        do {
            // Same defaults ExperimentPanel.create() stamps.
            manifest.studyKind = .modelOutput
            manifest.temperature = 0
            manifest.maxTokens = 2048
            manifest.promptMode = .chatAssistant
            manifest.qwenThinkingEnabled = false
            for pin in plan.pins where pin.method != .emotionGrandMean {
                let options = ExtractionOptions(
                    method: pin.method, readingPosition: pin.readingPosition)
                manifest.concepts.removeAll { $0.name == pin.concept }
                if pin.method == .designatedReference,
                    let pins = referencePins[pin.concept],
                    let referenceName = pin.designatedReferenceName
                {
                    // A REAL designated-reference concept pin — target
                    // stories as the stimulus hash, reference pinned
                    // beside it (same shape the attach path writes; the
                    // old paired fallthrough omitted the reference
                    // entirely — field diagnosis 2026-08-02, item 4).
                    var ref = ExperimentStore.makeConceptRef(
                        name: pin.concept,
                        stimulusSetHash: pins.target,
                        options: options)
                    ref.designatedReference = .init(
                        name: referenceName, hash: pins.reference)
                    manifest.concepts.append(ref)
                } else {
                    manifest.concepts.append(
                        ExperimentStore.makeConceptRef(
                            name: pin.concept,
                            stimulusSetHash: pairedHashes[pin.concept]
                                ?? pin.currentStimulusHash,
                            options: options))
                }
            }
            for pin in plan.pins where pin.method == .emotionGrandMean {
                try ExperimentStore.attachGrandMeanConcepts(
                    [pin.concept],
                    corpusConcepts: pin.corpusConcepts,
                    poolFromToken: pin.poolFromToken,
                    into: &manifest)
            }
            ExperimentStore.pinNeutralCorpus(into: &manifest)  // norm denominator
            try ExperimentStore.save(manifest)
            if metric == "judgeScore" {
                try JudgeRubricStore.pin(
                    (judgeRubricFile ?? "").trimmingCharacters(
                        in: .whitespacesAndNewlines),
                    into: &manifest)
                manifest.judges = cleanedJudges
                try ExperimentStore.save(manifest)
            }
        } catch let error as ExperimentError {
            throw ExperimentError(
                reason: error.reason
                    + " — draft study '\(manifest.name)' was created without "
                    + "its pins; fix the data and re-declare, or delete the "
                    + "draft in Studies")
        }
        guard panel.setSweepSpec(spec, for: manifest.name) else {
            throw ExperimentError(
                reason: (panel.status ?? "sweep spec not saved")
                    + " — draft study '\(manifest.name)' was created with its "
                    + "pins but without the declared criterion; declare one on "
                    + "it in Agents → Optimizations, or delete the draft in "
                    + "Studies")
        }
        return manifest.name
    }
}
