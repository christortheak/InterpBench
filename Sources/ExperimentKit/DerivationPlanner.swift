import Foundation
import SteeringKit

/// What the user is getting into *before* a derivation runs: reuse a fresh
/// artifact, be warned that an existing one is stale, or see the honest cost
/// of a new build. The view renders this verdict verbatim — there is never a
/// silent-reuse path (a stale match offers only re-extraction).
public enum DerivationAdvice: Sendable, Equatable {
    /// A verifiably fresh artifact already exists on the workspace — offer
    /// "Use existing / Re-extract anyway". `date` is the extraction stamp.
    case reusable(SubstrateVectorRecord, date: String?)
    /// A concept+model artifact exists but a pin diverged; `reason` is the
    /// classifier's canonical staleness string, shown verbatim. The only
    /// action is re-extraction.
    case staleExists(SubstrateVectorRecord, reason: String)
    /// Nothing to reuse — show the cost of the build (cheap arithmetic from
    /// the on-disk recipe files, no fake precision, no time estimates).
    case new(costLine: String)
}

/// Pure pre-derivation planning over already-fetched substrate records: the
/// freshness classification (`SubstrateCatalog.classify`) plus the honest
/// cost line. Lives in ExperimentKit — not in a view — so it is unit-testable
/// with synthetic record lists and counts; callers (the Concept Vector
/// Builder) inject the records, the live recipe pins, and the counts.
@MainActor
public enum DerivationPlanner {

    /// Cheap, honest cost arithmetic for a derivation. Counts come from the
    /// on-disk recipe files (recipes are local truth).
    public enum Cost: Sendable, Equatable {
        /// CAA / RepE-LAT: one forward pass per stimulus
        /// (positive + negative counts).
        case paired(stimulusCount: Int)
        /// Grand mean: one pass per build story row, across the corpus's
        /// concepts.
        case grandMean(storyCount: Int, conceptCount: Int)
        /// Reading-probe training: one pass per labeled example.
        case probe(exampleCount: Int)
    }

    /// "N stimuli × model" — with a note when the work runs as a server job.
    public static func costLine(
        _ cost: Cost,
        modelID: String,
        workspace: ClusterConnectionStore.Workspace
    ) -> String {
        let base: String
        switch cost {
        case .paired(let stimulusCount):
            base = "\(stimulusCount) stimuli × \(modelID)"
        case .grandMean(let storyCount, let conceptCount):
            base = "\(storyCount) stories across \(conceptCount) concepts × \(modelID)"
        case .probe(let exampleCount):
            base = "\(exampleCount) probe examples × \(modelID)"
        }
        switch workspace {
        case .local:
            return base
        case .server:
            return base + " — runs as a server job"
        }
    }

    /// The method-identity string the freshness comparison uses, per
    /// workspace. Local artifacts saved by the builder stamp the *recipe*
    /// method (`recipeMethod`: "caaMeanDifference"/"repeLAT"/
    /// "emotionGrandMean", preferred by the record normalization); the
    /// server's catalog exposes the sidecar's `extractionMethod`
    /// ("meanDifference"/"lat", and "emotionGrandMean" for grand-mean jobs).
    static func methodPin(
        for method: VectorExtractionRecipe.Method,
        workspace: ClusterConnectionStore.Workspace
    ) -> String {
        switch workspace {
        case .local:
            return method.rawValue
        case .server:
            switch method {
            case .caaMeanDifference: return ExtractionMethod.meanDifference.rawValue
            case .pairedDifferencePCA: return ExtractionMethod.pairedDifferencePCA.rawValue
            // Reader fits are server jobs now, but readers are measurement
            // artifacts outside the substrate VECTOR catalog; the raw value
            // keeps the pin honest if a vector record ever appears.
            case .repeReaderLAT: return method.rawValue
            case .emotionGrandMean: return ExtractionMethod.emotionGrandMean.rawValue
            // A J-lens direction has no ExtractionMethod twin — it is derived
            // from a lens plus a token id, not extracted from stimuli. The raw
            // value keeps the pin honest; freshness for these compares the
            // derivation identity, not a stimulus hash.
            case .jlensTokenDirection: return method.rawValue
            }
        }
    }

    /// Classify the requested derivation against the workspace's records.
    ///
    /// A nil `stimulusHash` (the live recipe could not be read/hashed) never
    /// yields `.reusable` — an unverifiable recipe must not offer reuse — it
    /// falls straight through to `.new` with the cost line.
    public static func advise(
        records: [SubstrateVectorRecord],
        concept: String,
        method: VectorExtractionRecipe.Method,
        workspace: ClusterConnectionStore.Workspace,
        modelID: String,
        revision: String? = nil,
        stimulusHash: String?,
        cost: Cost
    ) -> DerivationAdvice {
        let newAdvice = DerivationAdvice.new(
            costLine: costLine(cost, modelID: modelID, workspace: workspace))
        guard let stimulusHash else { return newAdvice }
        switch SubstrateCatalog.classify(
            records: records,
            concept: concept,
            modelID: modelID,
            revision: revision,
            stimulusHash: stimulusHash,
            method: methodPin(for: method, workspace: workspace))
        {
        case .fresh(let record):
            return .reusable(record, date: record.extractionDate)
        case .stale(let record, let reason):
            return .staleExists(record, reason: reason)
        case nil:
            return newAdvice
        }
    }
}
