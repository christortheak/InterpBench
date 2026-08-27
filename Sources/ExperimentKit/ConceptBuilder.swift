import CryptoKit
import Foundation
import MLXLMCommon
import Observation
import SteeringKit

/// Interactive concept construction: paste contrasting stimuli, rebuild the
/// CAA direction live, and watch design stats update. A front-end to the
/// exact same files and pipeline the CLI uses — saving writes
/// `prompts/concepts/<name>/{positive,negative}.jsonl` and extracts through
/// `ConceptExtractor` into a normal run directory with full provenance.
///
/// Per-stimulus activations are cached by (model, text), so adding a pair
/// costs two forward passes; the means and stats recompute on CPU.
@Observable @MainActor
public final class ConceptBuilder {

    public enum RecipeFamily: String, Codable, Sendable, CaseIterable, Identifiable {
        case caaMeanDifference
        /// PCA over normalized paired differences — RepE-INSPIRED direction
        /// math, not the paper's pipeline. Raw value pinned to the legacy
        /// `"repeLAT"` so persisted builder state and every sidecar written
        /// under the old symbol keep decoding (naming ruling 2026-08-27).
        case pairedDifferencePCA = "repeLAT"
        /// The faithful template-mediated RepE reader.
        case repeReaderLAT
        case emotionGrandMean
        /// METHODS amendment ii: mean(concept stories) − mean(a DESIGNATED
        /// reference class's stories) — unpaired class-vs-reference mean
        /// difference over the `prompts/emotions/` story layout. The
        /// reference corpus is part of the recipe: the artifact pins it as
        /// {name, hash}, exactly as `experiment attach --method
        /// designatedReference --reference <concept>` does.
        case designatedReference
        /// A direction for one exact vocabulary token, derived from an imported
        /// Jacobian lens. Unlike every other family this reads NO activations:
        /// no stimuli, no reading position, no pooling, and no loaded model —
        /// it slices two tensors out of the checkpoint. Server-only, because
        /// lens artifacts are PyTorch/HF-native.
        case jlensTokenDirection

        public var id: String { rawValue }

        public var label: String {
            switch self {
            case .caaMeanDifference: "CAA mean difference"
            case .pairedDifferencePCA: "Paired-difference PCA (RepE-inspired)"
            case .repeReaderLAT: "RepE reader LAT"
            case .emotionGrandMean: "Grand mean"
            case .designatedReference: "Designated reference (stories − reference)"
            case .jlensTokenDirection: "J-lens token direction"
            }
        }

        public var recipeMethod: VectorExtractionRecipe.Method {
            switch self {
            case .caaMeanDifference: .caaMeanDifference
            case .pairedDifferencePCA: .pairedDifferencePCA
            case .repeReaderLAT: .repeReaderLAT
            case .emotionGrandMean: .emotionGrandMean
            case .designatedReference: .designatedReference
            case .jlensTokenDirection: .jlensTokenDirection
            }
        }

        public var isPaired: Bool {
            switch self {
            case .caaMeanDifference, .pairedDifferencePCA, .repeReaderLAT: true
            case .emotionGrandMean, .designatedReference, .jlensTokenDirection:
                false
            }
        }

        /// True when the family's stimuli are story corpora under
        /// `prompts/emotions/<concept>/stories.jsonl` rather than paired
        /// files — the dataset pane then edits story rows, and `addDrafts`
        /// parses story JSONL. Mirrors ``ExtractionMethod/usesStoryCorpus``.
        public var usesStoryCorpus: Bool {
            switch self {
            case .emotionGrandMean, .designatedReference: true
            default: false
            }
        }

        /// The direction-finding method this family ALREADY means, or nil for
        /// a family that reads no activations. The extraction-method picker
        /// consults it before demoting a family: `repeReaderLAT` means a
        /// paired-difference PCA read, and `emotionGrandMean` means a mean
        /// difference over its corpus, so seeing those methods arrive is not
        /// evidence that the researcher left the family.
        public var impliedExtractionMethod: ExtractionMethod? {
            switch self {
            case .caaMeanDifference: .meanDifference
            case .pairedDifferencePCA: .pairedDifferencePCA
            case .repeReaderLAT: .pairedDifferencePCA
            case .emotionGrandMean: .meanDifference
            // First-class in the method vocabulary, and the family's own
            // didSet pins exactly it — so a designated-reference family
            // seeing `designatedReference` arrive is not evidence that the
            // researcher left the family (the demotion the guard below
            // exists to prevent).
            case .designatedReference: .designatedReference
            case .jlensTokenDirection: nil
            }
        }

        /// True when the family reads activations from the loaded model over a
        /// stimulus set. False for a DERIVED direction, whose pane hides the
        /// stimulus, pooling, and reading-position controls because none of
        /// them apply — and would silently imply provenance it does not have.
        public var extractsFromStimuli: Bool {
            switch self {
            case .caaMeanDifference, .pairedDifferencePCA, .repeReaderLAT,
                .emotionGrandMean, .designatedReference:
                true
            case .jlensTokenDirection:
                false
            }
        }

        /// Families that exist only on the server. Lens artifacts are
        /// PyTorch/HF-native and activations do not transfer across substrates,
        /// so there is nothing to fall back to locally.
        public var isServerOnly: Bool {
            switch self {
            case .jlensTokenDirection: true
            default: false
            }
        }

        /// Whether a fresh artifact of this family arrives with residual norms.
        /// Derivation measures no neutral-corpus denominator, so norm-unit alpha
        /// needs the ordinary backfill first — worth saying before someone
        /// reaches for it and gets a refusal later.
        public var arrivesWithResidualNorms: Bool { extractsFromStimuli }

        /// Plain-language method guidance shown in the Concept Lab's Activity
        /// pane. This is instructional UI only; extraction behavior remains in
        /// the recipe implementation and its pinned options.
        public var activityGuide: [String] {
            switch self {
            case .caaMeanDifference:
                [
                    "CAA mean difference builds one direction per layer from mean(concept-present activations) - mean(matched-control activations).",
                    "Data needed: matched positive/negative sentences that differ mainly in the concept.",
                    "Workflow: save or select a concept, copy the LLM prompt, paste JSONL into the two dataset boxes (or import a file), click Add to set, then build the vector.",
                ]
            case .pairedDifferencePCA:
                [
                    "Paired-difference PCA is the RepE-inspired PCA direction over matched positive-minus-negative activation differences — the paper's direction math without its template-mediated pipeline.",
                    "Data needed: matched concept-present/control pairs with train/test metadata when available.",
                    "Workflow: copy the recipe prompt, paste or import its JSONL pairs, add them to the dataset, then build the per-layer direction.",
                ]
            case .repeReaderLAT:
                [
                    "RepE reader LAT fits a scalar reading direction from paired examples rendered through a selected reader template.",
                    "Data needed: an independent paired reader dataset; held-out pairs estimate reader accuracy.",
                    "Workflow: copy the LLM prompt, paste/import pairs, choose the reader template and held-out count, then use Build Reader rather than the ordinary vector button.",
                ]
            case .emotionGrandMean:
                [
                    "Grand mean builds each selected concept as mean(concept stories) - mean(all included concept stories), following the emotion-vector corpus design.",
                    "Data needed: a balanced concept x topic story corpus; build rows extract vectors and validation rows stay held out.",
                    "Workflow: copy the LLM prompt for a small corpus or the Claude Cowork prompt for parallel multi-concept generation, paste the returned JSONL into the story box, add the rows, choose included/build concepts and topics, then build.",
                ]
            case .designatedReference:
                [
                    "Designated reference builds each layer's direction as mean(concept stories) - mean(reference stories): an unpaired class-vs-reference mean difference, where the reference is a deliberately authored corpus, not a population centroid.",
                    "Data needed: the concept's story corpus plus a reference corpus in the same register and on the same topic grid (narrative concepts reference neutral stories; expository concepts reference plain exposition) — a register mismatch injects an essay-vs-story component into the vector.",
                    "Workflow: author both corpora as story concepts (copy the LLM prompt, paste the returned JSONL), pick the reference class, then build. The artifact pins the reference by name and stories hash — the reference corpus is part of the recipe.",
                    "Negative-dose semantics: -alpha steers toward the reference class, i.e. toward an ordinary telling of the same scenes.",
                ]
            case .jlensTokenDirection:
                [
                    "J-lens token direction derives one vector per layer as J_l^T (g . u_t) for ONE exact vocabulary token, from an imported Jacobian lens.",
                    "It reads no activations: no stimuli, no reading position, no pooling, and no loaded model — it slices the token's embedding row and the final-norm gain out of the checkpoint.",
                    "Workflow: pick the model, confirm its lens is imported, type a word, then SELECT AN EXACT TOKEN. A word that is not a single token shows its components; choosing one derives a direction for that component, not the word.",
                    "The token id is the durable identity, not the word, so it stays in the saved name.",
                    "Server-only: lens artifacts are PyTorch/HF-native and activations do not transfer across substrates.",
                    "This is an instrument primitive and a positive control for the readout — not evidence on its own that the token names a psychological concept.",
                ]
            }
        }
    }

    public struct OutlierDisplay: Identifiable, Sendable {
        public let id: String
        public let classLabel: String
        public let textPrefix: String
        public let margin: Float
    }

    public struct ControlCosine: Identifiable, Sendable {
        public let name: String
        public let cosine: Float
        public var id: String { name }
    }

    public struct ProbeExample: Identifiable, Codable, Sendable, Hashable {
        public var id: String
        public var text: String
        public var expresses: Bool
        public var topic: String?
        public var split: String?
        public var notes: String?
    }

    public struct Stats: Sendable {
        public var positiveCount: Int
        public var negativeCount: Int
        public var statsLayer: Int
        public var heldOut: ConceptStats.HeldOut?
        public var splitHalf: Float?
        /// Cosine of the current direction vs the previous rebuild.
        public var stability: Float?
        public var normByLayer: [Float]
        public var outliers: [OutlierDisplay]
        public var controlCosines: [ControlCosine]
        /// Decision margin for every stimulus, keyed "pos-<i>"/"neg-<i>" —
        /// negative means the stimulus sits on the wrong side of the
        /// direction. Drives the browse view.
        public var marginByStimulus: [String: Float]
    }

    public internal(set) weak var host: ChatService?

    // MARK: Editing state

    public var conceptName = ""
    /// The DATASET selection: the concept whose stimulus files the Concept
    /// Index / Dataset Builder are editing.
    ///
    /// Changing it to a concept also moves the vector builder's build target
    /// (below) — one-way, at the model layer, so every route into a selection
    /// gets it: the two dataset pickers, the Data-section inventory's "Open in
    /// Concept Builder", and the web client's `/api/concept/select`. Before
    /// 2026-08-19 the two selections drifted silently, so the panel showed one
    /// concept while a build targeted another. The build target stays
    /// independently changeable AFTERWARDS — deliberate divergence is still
    /// possible, it just cannot happen by accident.
    public var selectedExisting: String? {
        didSet {
            guard oldValue != selectedExisting else { return }
            loadSelectedExisting()
            if let selectedExisting { vectorBuilderSelectedExisting = selectedExisting }
        }
    }
    /// The BUILD target: the concept the Concept Vector Builder extracts.
    /// Follows `selectedExisting` when that changes; independently settable.
    public var vectorBuilderSelectedExisting: String? {
        didSet {
            guard oldValue != vectorBuilderSelectedExisting else { return }
            refreshStaleness()
        }
    }
    public var positiveDraft = ""
    public var negativeDraft = ""
    public var pairedJSONLDraft = ""
    public var multiConceptDraft = ""
    public var probeDraft = ""
    public var neutralCorpusDraft = ""
    public var projectionNeutralCorpusName = "assistant-dialogue-neutral"
    public var projectionNeutralConceptsDraft = ""
    public var projectionNeutralDomainsDraft = ""
    public var projectionNeutralExclusionsDraft = ""
    public var multiConceptStoryConceptDraft = ""
    public var multiConceptTopicDraft = ""
    public var multiConceptSplitDraft = "build"
    public var includedEmotionTopics: Set<String> = []
    public var includedEmotionConcepts: Set<String> = []
    public var grandMeanBuildConcepts: Set<String> = []
    public var grandMeanBuildConceptsAreExplicit = false
    public var buildAllGrandMeanConcepts = true {
        didSet {
            guard oldValue != buildAllGrandMeanConcepts else { return }
            noteMutation()
        }
    }

    public private(set) var positives: [String] = []
    public private(set) var negatives: [String] = []
    public private(set) var multiConceptRows: [StimulusSet.MultiConceptStimulus] = []
    public private(set) var probeExamples: [ProbeExample] = []
    public private(set) var existingConcepts: [String] = []
    public private(set) var stats: Stats?
    public private(set) var isWorking = false
    public private(set) var activeTaskLabel: String?
    public private(set) var status: String?
    /// Most recent vector/reader/probe BUILD failure — rendered as a
    /// banner-style row in the Concepts panel, because a failed build in the
    /// caption-sized status line is invisible in practice (missing/unsynced
    /// stimulus files died silently there). Set only on failure (success
    /// stays quiet), cleared when a new build starts or the user dismisses.
    public private(set) var lastBuildError: String?

    public func clearBuildError() { lastBuildError = nil }

    /// Route a build failure where it can be seen: the status caption
    /// (existing behavior), the panel's error banner, and — unless the
    /// failure already streamed through a followed job's live log — the
    /// shared live-log/Activity pane (`ChatService.startLiveLog`, the same
    /// host the experiment runner logs through).
    private func reportBuildFailure(
        _ message: String, title: String, logToActivity: Bool = true
    ) {
        status = message
        lastBuildError = message
        if logToActivity, let host {
            _ = host.startLiveLog(title: title, initialLine: message)
        }
    }

    public var currentConceptName: String {
        selectedExisting ?? Self.sanitizedName(conceptName)
    }

    public var vectorBuilderConceptName: String {
        vectorBuilderSelectedExisting ?? ""
    }

    public struct EmotionCorpusSummary: Sendable {
        public let targetConcept: String
        public let totalRowCount: Int
        public let rowCount: Int
        public let validationRowCount: Int
        public let draftRowCount: Int
        public let targetRowCount: Int
        public let conceptCounts: [(name: String, count: Int)]
        public let topicCounts: [(name: String, count: Int)]
        public let cellCounts: [(topic: String, concept: String, count: Int)]
        public let missingCells: [(topic: String, concept: String)]
        public let minCellCount: Int
        public let maxCellCount: Int

        public var conceptCount: Int { conceptCounts.count }
        public var topicCount: Int { topicCounts.count }
        public var isBalanced: Bool {
            rowCount > 0 && missingCells.isEmpty && maxCellCount - minCellCount <= 1
        }
    }

    // MARK: Claude-generated proposals (pending human review)

    public struct Proposal: Identifiable, Sendable {
        public let id = UUID()
        public let positive: String
        public let negative: String
        public var included = true
    }

    public var proposals: [Proposal] = []
    public private(set) var isGeneratingProposals = false
    public var generationCount = 10
    public var probeGenerationCount = 240
    public var generationGuidance = ""
    public private(set) var lastCopiedPrompt: String?
    private var recipeGuideLogID: UUID?
    /// Stable live-log ids per notice category: copy-tweak-copy loops update
    /// one Activity entry instead of appending a new bubble per click.
    private var copiedPromptLogID: UUID?
    private var promptNoticeLogID: UUID?
    /// Test seam for the packaged-app case: where `templatePrompt` looks for
    /// the code-shipped seed copy. nil (the real checkout) in production.
    internal static var seedRootOverrideForTesting: URL?

    public var probePositiveCount: Int {
        probeExamples.filter(\.expresses).count
    }

    public var probeNegativeCount: Int {
        probeExamples.filter { !$0.expresses }.count
    }

    public var neutralCorpusSummary: (count: Int, hash: String?) {
        let record = NeutralCorpusStore.record(id: host?.selectedNeutralCorpusID)
        return (record.count, record.hash)
    }

    public var normNeutralCorpusSummary: (count: Int, hash: String?) {
        let record = NeutralCorpusStore.record(id: NeutralCorpusStore.normCorpusID)
        return (record.count, record.hash)
    }

    // MARK: Extraction options (METHODS.md › Method options)

    public var recipeFamily: RecipeFamily = .caaMeanDifference {
        didSet {
            guard oldValue != recipeFamily else { return }
            syncingRecipeFamily = true
            defer { syncingRecipeFamily = false }
            switch recipeFamily {
            case .caaMeanDifference:
                extractionMethod = .meanDifference
                if poolFromToken == 50 { poolFromToken = nil }
            case .pairedDifferencePCA:
                extractionMethod = .pairedDifferencePCA
                if poolFromToken == 50 { poolFromToken = nil }
            case .repeReaderLAT:
                // Reader fit is template-rendered LAT at the scaffold's final
                // token; the pooled reading position does not apply.
                extractionMethod = .pairedDifferencePCA
                poolFromToken = nil
                refreshReaderTemplates()
            case .jlensTokenDirection:
                // Nothing to extract and nothing to pool: the direction comes
                // from a lens matrix and a token row, not from activations.
                poolFromToken = nil
            case .emotionGrandMean:
                extractionMethod = .meanDifference
                poolFromToken = 50
                if generationCount == 10 { generationCount = 12 }
            case .designatedReference:
                // The attach verb's method policy, mirrored: pooled reading
                // from token 50 unless a position is declared explicitly.
                extractionMethod = .designatedReference
                poolFromToken = 50
            }
            stats = nil
            lastDirection = nil
            noteMutation()
            presentRecipeGuide()
        }
    }

    public var extractionMethod: ExtractionMethod = .meanDifference {
        didSet {
            guard oldValue != extractionMethod else { return }
            // Reverse sync, but only when the standing family does NOT
            // already imply this method (audit finding 4, fixed 2026-08-27).
            // The old rule was unconditional. Setting the method to the very
            // value a repeReaderLAT (or emotionGrandMean) family already
            // implies then DEMOTED the family — a reader concept silently
            // became a paired-difference-PCA one, and the pane's Build Reader
            // button, its template picker and its held-out control all
            // vanished with it. A family that already means this method is
            // not news.
            if !syncingRecipeFamily,
                recipeFamily.impliedExtractionMethod != extractionMethod
            {
                recipeFamily =
                    switch extractionMethod {
                    case .pairedDifferencePCA: .pairedDifferencePCA
                    case .designatedReference: .designatedReference
                    default: .caaMeanDifference
                    }
            }
            lastDirection = nil
            noteMutation()
        }
    }
    /// designatedReference only: the REFERENCE class — the stories concept
    /// whose mean is subtracted from the target's. Part of the recipe (the
    /// artifact pins {name, hash}), so it is a deliberate selection with an
    /// empty default, never a guess.
    public var designatedReferenceConcept: String = "" {
        didSet {
            guard oldValue != designatedReferenceConcept else { return }
            lastDirection = nil
            noteMutation()
        }
    }

    /// Reference-class candidates: concepts with a local
    /// `prompts/emotions/<name>/stories.jsonl`. The build target may appear
    /// here too — selecting it stands refused below rather than hidden, so
    /// the mistake is explained instead of silently unavailable.
    public var designatedReferenceOptions: [String] {
        localStoryConceptNames()
    }

    /// The standing refusal for the current reference selection, or nil. An
    /// EMPTY selection is not a refusal (the picker says "select reference…"
    /// and the build button stays off); a self-reference is.
    public var designatedReferenceRefusal: String? {
        let reference = designatedReferenceConcept
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !reference.isEmpty, reference == buildTargetName else { return nil }
        return "the reference must be a different stories concept — "
            + "subtracting a concept's own corpus extracts the zero vector — "
            + "repair: pick the deliberately authored reference class "
            + "(a neutral-stories or plain-exposition corpus in the same register)"
    }

    /// WHERE the residual stream is read, as a picker holds it — the WHOLE
    /// declarable vocabulary, not only the legacy pooled pair this pane used
    /// to offer. The position itself is re-derived through the engine's own
    /// strict parser (``ReadingPositionChoice/declaredPosition(parameter:)``),
    /// so an out-of-vocabulary parameter is REFUSED in the engine's words
    /// rather than clamped into a recipe nobody asked for.
    public var readingPositionChoice: ReadingPositionChoice = .lastToken {
        didSet {
            guard oldValue != readingPositionChoice else { return }
            // Carry the number into the new kind's range. A convenience, not
            // a validation — the field still accepts anything, and the
            // engine still answers for what is typed there.
            let stepped = readingPositionChoice.steppedParameter(
                from: readingPositionParameter)
            if stepped != readingPositionParameter {
                readingPositionParameter = stepped  // redeclares in its didSet
            } else {
                redeclareReadingPosition()
            }
        }
    }
    /// The K/k/i/n beside the position, when it takes one.
    public var readingPositionParameter = 0 {
        didSet {
            guard oldValue != readingPositionParameter else { return }
            redeclareReadingPosition()
        }
    }
    /// The declared position, or `.lastToken` — the value every extraction
    /// path here reads.
    public private(set) var readingPosition: ReadingPosition = .lastToken
    /// The engine's own refusal for the current selection, or nil. Never this
    /// pane's words: the vocabulary and the repair come from `ReadingPosition`.
    public private(set) var readingPositionRefusal: String?

    private func redeclareReadingPosition() {
        do {
            readingPosition =
                try readingPositionChoice.declaredPosition(
                    parameter: readingPositionParameter) ?? .lastToken
            readingPositionRefusal = nil
        } catch let error as ReadingPosition.DeclarationError {
            readingPositionRefusal = "\(error.reason) — repair: \(error.repair)"
        } catch {
            readingPositionRefusal = "\(error)"
        }
        lastDirection = nil
        noteMutation()
    }

    /// The LEGACY spelling of exactly one position (`mean from token k`), kept
    /// because the server-extract routes and the web client type it. nil for
    /// every other reading position — `--pool-from K` and
    /// `--reading-position 'mean from token K'` are one declaration, never two.
    public var poolFromToken: Int? {
        get {
            readingPositionChoice == .meanFromToken ? readingPositionParameter : nil
        }
        set {
            if let newValue {
                readingPositionParameter = newValue
                readingPositionChoice = .meanFromToken
            } else if readingPositionChoice == .meanFromToken {
                readingPositionChoice = .lastToken
            }
        }
    }

    /// HOW each stimulus reaches the model. Raw declares NOTHING — and
    /// nothing is what an untouched builder passes, so the artifact it writes
    /// is byte-identical to what this pane always wrote.
    public var extractionRenderingChoice = ExtractionRenderingChoice() {
        didSet {
            guard oldValue != extractionRenderingChoice else { return }
            redeclareExtractionRendering()
        }
    }
    /// The declared rendering, or nil for the legacy raw one.
    public private(set) var extractionRendering: ExtractionRendering?
    /// The engine's own refusal for the current selection, or nil — this is
    /// where the assistant-voice and `addGenerationPrompt: false` engine
    /// asymmetries speak.
    public private(set) var extractionRenderingRefusal: String?
    /// True when the standing rendering refusal is THIS engine's limit rather
    /// than a malformed declaration — the assistant voice and
    /// `addGenerationPrompt: false`, both of which the server engine renders
    /// happily. Their own repair text says so ("extract this concept on the
    /// python-hf-transformers engine, which supports it"), so a server build
    /// under one of them must not be blocked by a refusal that names the
    /// server as the fix.
    public private(set) var extractionRenderingRefusalIsLocalEngineLimit = false

    /// The refusals ``ExtractionRendering/declared(object:)`` raises for what
    /// the swift-mlx renderer cannot do, as opposed to what no engine accepts.
    /// Matched by REASON — the same constants the parser throws, so the two
    /// cannot drift into two lists.
    private static let localEngineRenderingLimits = [
        PromptRendering.assistantVoiceReason,
        PromptRendering.addGenerationPromptFalseReason,
    ]

    private func redeclareExtractionRendering() {
        do {
            extractionRendering = try extractionRenderingChoice.declared()
            extractionRenderingRefusal = nil
            extractionRenderingRefusalIsLocalEngineLimit = false
        } catch let error as ExtractionRendering.DeclarationError {
            extractionRenderingRefusal = "\(error.reason) — repair: \(error.repair)"
            extractionRenderingRefusalIsLocalEngineLimit =
                Self.localEngineRenderingLimits.contains(error.reason)
        } catch {
            extractionRenderingRefusal = "\(error)"
            extractionRenderingRefusalIsLocalEngineLimit = false
        }
        lastDirection = nil
        noteMutation()
    }

    /// True while either declaration stands refused: the LOCAL build button is
    /// off, because extracting under the last VALID declaration would silently
    /// produce a vector nobody asked for.
    public var hasRefusedExtractionDeclaration: Bool {
        readingPositionRefusal != nil || extractionRenderingRefusal != nil
    }

    /// The same question for a SERVER build, which is not the same question.
    ///
    /// The engines are asymmetric in exactly one direction: `swift-mlx`
    /// refuses the assistant voice and `addGenerationPrompt: false` (its
    /// tokenizer bridge cannot render either), and `python-hf-transformers`
    /// renders both. Those two refusals were computed by the LOCAL parser and
    /// literally name the server as their repair, so letting them switch off
    /// the server build would refuse a declaration the server accepts. Every
    /// other refusal — an out-of-vocabulary reading position, a malformed
    /// rendering — is engine-independent and stops both paths.
    ///
    /// This gate is belt-and-braces, not the validation: the routes re-parse
    /// the declaration with the SERVER's own parsers and answer a 400 in the
    /// server engine's words, which is what the panel shows.
    public var hasRefusedServerExtractionDeclaration: Bool {
        if readingPositionRefusal != nil { return true }
        guard extractionRenderingRefusal != nil else { return false }
        return !extractionRenderingRefusalIsLocalEngineLimit
    }

    /// The standing refusal that blocks a server build, or nil.
    public var serverExtractionDeclarationRefusal: String? {
        guard hasRefusedServerExtractionDeclaration else { return nil }
        return readingPositionRefusal ?? extractionRenderingRefusal
    }

    /// The rendering a SERVER build declares, built from the picker state
    /// directly rather than through this engine's ``declared()`` parser.
    ///
    /// Deliberately not `extractionRendering`: that property is the LOCAL
    /// engine's parse, and it is nil under the assistant voice precisely
    /// because this engine cannot render it — sending nil there would post a
    /// raw build under a panel showing a chat template, which is the silent
    /// substitution this whole path exists to end. The server re-parses this
    /// with its own strict parser and refuses anything it cannot honor.
    public var serverExtractionRendering: ExtractionRendering? {
        guard extractionRenderingChoice.mode == .chatTemplate else { return nil }
        if extractionRenderingChoice.voice == .assistant {
            // The declaration OMITS addGenerationPrompt under this voice —
            // both engines refuse the key there as meaningless.
            return ExtractionRendering(mode: .chatTemplate, voice: .assistant)
        }
        return ExtractionRendering(
            mode: .chatTemplate,
            addGenerationPrompt: extractionRenderingChoice.addGenerationPrompt
                ? nil : false)
    }

    /// The whole extraction declaration in the shape the server-extract
    /// routes take it.
    public struct ServerExtractionDeclaration: Sendable, Equatable {
        /// The LEGACY spelling of one position, sent only when it says
        /// everything the declaration says.
        public var poolFromToken: Int?
        /// The cross-engine label, for every other position.
        public var readingPosition: String?
        public var extractionRendering: ExtractionRendering?
    }

    /// What ``buildVectorOnActiveServer()`` declares, for a route whose
    /// pre-declaration reading of an ABSENT `poolFromToken` was
    /// `legacyPooledDefault` (nil where the route read the last token — the
    /// paired route; 50 where it pooled — the grand-mean route).
    ///
    /// THE COMPATIBILITY RULE, stated once: a declaration the legacy
    /// `poolFromToken` field expresses COMPLETELY — a pooled mean from token
    /// K, or the last token where absence already means that — is sent in
    /// that legacy spelling, so the request bytes are identical to the ones
    /// this panel has always posted and a server that predates the
    /// declaration fields keeps building the recipes it can actually honor.
    /// Anything else is sent as the label plus the rendering object, and the
    /// client then REQUIRES the server's echo
    /// (``ClusterClient/AppliedExtraction``): a declaration that cannot be
    /// confirmed is a refusal, never a build.
    ///
    /// `mean from token 0` is deliberately NOT legacy-expressible on either
    /// route: both read a zero pool as something else (the last token; a pool
    /// from 50), so the label is the only spelling that says it.
    public func serverExtractionDeclaration(
        legacyPooledDefault: Int?
    ) -> ServerExtractionDeclaration {
        let rendering = serverExtractionRendering
        let parameter = readingPositionParameter
        if rendering == nil {
            switch readingPositionChoice {
            case .lastToken where legacyPooledDefault == nil:
                return ServerExtractionDeclaration(poolFromToken: nil)
            case .meanFromToken where parameter != 0:
                return ServerExtractionDeclaration(poolFromToken: parameter)
            default:
                break
            }
        }
        return ServerExtractionDeclaration(
            readingPosition: readingPositionChoice.declarationLabel(
                parameter: parameter),
            extractionRendering: rendering)
    }

    // MARK: Staleness & dirty tracking

    /// True when the displayed stats no longer describe the current
    /// (stimuli, method, reading position, model).
    public private(set) var statsStale = false
    /// Forward passes a rebuild would need right now (uncached stimuli,
    /// including control concepts for the discriminant cosines). 0 means a
    /// rebuild is pure CPU over cached activations.
    public private(set) var pendingPassCount = 0
    /// True when the working set or options differ from the files on disk /
    /// the newest saved artifact for this concept and model.
    public private(set) var unsavedChanges = false

    public var canSaveAndExtract: Bool {
        // A standing declaration refusal disables the build: extracting under
        // the last VALID declaration would write a vector nobody asked for.
        if hasRefusedExtractionDeclaration { return false }
        if recipeFamily == .emotionGrandMean {
            return !grandMeanTargetConceptNames.isEmpty
        }
        if recipeFamily == .designatedReference {
            // Both classes must exist as non-empty story corpora, and the
            // reference must be a different concept — the same gates the
            // build itself refuses on, so the button can never be enabled
            // for a build that would refuse.
            let name = buildTargetName
            guard !name.isEmpty, designatedReferenceRefusal == nil else {
                return false
            }
            let reference = designatedReferenceConcept
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !reference.isEmpty else { return false }
            return storyCorpusRowCount(for: name) > 0
                && storyCorpusRowCount(for: reference) > 0
        }
        if recipeFamily == .repeReaderLAT {
            return false  // reader family builds through buildReader()
        }
        // Same resolution the build itself uses, so the button can never be
        // enabled for one concept and extract another.
        let name = buildTargetName
        guard !name.isEmpty else { return false }
        if name == currentName {
            return positives.count >= 4 && negatives.count >= 4
        }
        let directory = VectorCatalog.conceptsDirectory.appending(component: name)
        guard let set = try? StimulusSet(directory: directory) else { return false }
        return set.positive.count >= 4 && set.negative.count >= 4
    }

    private var statsFingerprint: Int?
    private var freeRebuildTask: Task<Void, Never>?
    private var syncingRecipeFamily = false

    private var currentFingerprint: Int {
        var hasher = Hasher()
        hasher.combine(host?.loadedModelID)
        hasher.combine(recipeFamily)
        hasher.combine(vectorBuilderSelectedExisting)
        hasher.combine(extractionMethod)
        hasher.combine(extractionOptions.readingPosition.label)
        hasher.combine(extractionOptions.resolvedExtractionRendering.label)
        hasher.combine(positives)
        hasher.combine(negatives)
        hasher.combine(multiConceptRows)
        hasher.combine(includedEmotionTopics.sorted())
        hasher.combine(includedEmotionConcepts.sorted())
        hasher.combine(grandMeanBuildConcepts.sorted())
        hasher.combine(grandMeanBuildConceptsAreExplicit)
        hasher.combine(buildAllGrandMeanConcepts)
        hasher.combine(designatedReferenceConcept)
        return hasher.finalize()
    }

    /// Recomputes staleness, the pass-count estimate, and the unsaved flag.
    /// Call after anything that changes what the stats describe.
    public func refreshStaleness() {
        statsStale = stats == nil || statsFingerprint != currentFingerprint
        pendingPassCount = computePendingPasses()
        unsavedChanges = computeUnsavedChanges()
    }

    /// The compute policy: recompute automatically only when free (zero
    /// forward passes — method flips, deletions, reverts, all served from
    /// cache). Costed rebuilds wait for the explicit button, which shows
    /// the pass count — at 27B scale a pass is real money.
    private func noteMutation() {
        refreshStaleness()
        guard statsStale, stats != nil, pendingPassCount == 0 else { return }
        freeRebuildTask?.cancel()
        freeRebuildTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(300))
            guard !Task.isCancelled else { return }
            await self?.rebuild()
        }
    }

    private func computePendingPasses() -> Int {
        let options = extractionOptions
        if recipeFamily == .designatedReference {
            let texts = designatedReferenceClassTexts()
            guard let modelID = host?.loadedModelID else { return texts.count }
            return texts.count {
                activationCache[
                    Self.activationCacheKey(
                        modelID: modelID, options: options, text: $0)] == nil
            }
        }
        if recipeFamily == .emotionGrandMean {
            let rows = emotionRowsForExtraction()
            guard let modelID = host?.loadedModelID else { return rows.count }
            return rows.count {
                activationCache[
                    Self.activationCacheKey(
                        modelID: modelID, options: options, text: $0.text)] == nil
            }
        }
        guard let modelID = host?.loadedModelID else {
            return positives.count + negatives.count
        }
        func missing(_ texts: [String]) -> Int {
            texts.count {
                activationCache[
                    Self.activationCacheKey(
                        modelID: modelID, options: options, text: $0)] == nil
            }
        }
        var passes = missing(positives) + missing(negatives)
        for name in existingConcepts where name != currentName {
            let directory = VectorCatalog.conceptsDirectory.appending(component: name)
            if let set = try? StimulusSet(directory: directory) {
                passes += missing(set.positive) + missing(set.negative)
            }
        }
        return passes
    }

    private func computeUnsavedChanges() -> Bool {
        let name = currentName
        guard !name.isEmpty else {
            return !positives.isEmpty || !negatives.isEmpty || !multiConceptRows.isEmpty
        }
        if recipeFamily == .designatedReference {
            if !multiConceptDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return true
            }
            guard let modelID = host?.loadedModelID else { return false }
            let target = buildTargetName
            let reference = designatedReferenceConcept
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard !target.isEmpty, !reference.isEmpty,
                let targetHash = storyCorpusHash(for: target),
                let referenceHash = storyCorpusHash(for: reference)
            else { return false }
            let newest = VectorCatalog.scan().first {
                $0.sidecar.concept == target && $0.sidecar.modelID == modelID
            }
            guard let newest else { return true }
            return newest.sidecar.recipeMethod
                != VectorExtractionRecipe.Method.designatedReference.rawValue
                || newest.sidecar.stimulusSetHash != targetHash
                || newest.sidecar.designatedReference?["name"] != reference
                || newest.sidecar.designatedReference?["hash"] != referenceHash
                || (newest.sidecar.readingPosition ?? ReadingPosition.lastToken.label)
                    != extractionOptions.readingPosition.label
                || Self.renderingLabel(newest.sidecar.extractionRendering)
                    != extractionOptions.resolvedExtractionRendering.label
        }
        if recipeFamily == .emotionGrandMean {
            if !multiConceptDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return true
            }
            let url = VectorCatalog.projectRoot.appending(
                components: "prompts", "emotions", name, "stories.jsonl")
            guard let disk = try? StimulusSet.loadMultiConceptTexts(url: url) else {
                return !multiConceptRows.isEmpty
            }
            if disk.rows.map(Self.canonicalized).filter({ $0.concept == name })
                != multiConceptRows
            { return true }
            guard let modelID = host?.loadedModelID else { return false }
            let newest = VectorCatalog.scan().first {
                $0.sidecar.concept == name && $0.sidecar.modelID == modelID
            }
            guard let newest else { return true }
            let extractionRows = emotionRowsForExtraction()
            guard !extractionRows.isEmpty else { return false }
            let selectedCorpusHash = Self.sha256(multiConceptJSONL(extractionRows))
            let selectedConcepts = selectedEmotionConceptNames().sorted()
            let selectedTopics = includedEmotionTopics.isEmpty
                ? Array(Set(extractionRows.map { normalizedTopic($0.topic) })).sorted()
                : includedEmotionTopics.sorted()
            return newest.sidecar.recipeMethod != VectorExtractionRecipe.Method
                .emotionGrandMean.rawValue
                || newest.sidecar.stimulusSetHash != selectedCorpusHash
                || (newest.sidecar.readingPosition ?? ReadingPosition.lastToken.label)
                    != extractionOptions.readingPosition.label
                || Self.renderingLabel(newest.sidecar.extractionRendering)
                    != extractionOptions.resolvedExtractionRendering.label
                || newest.sidecar.comparisonConcepts != selectedConcepts
                || newest.sidecar.selectedTopics != selectedTopics
        }
        let directory = VectorCatalog.conceptsDirectory.appending(component: name)
        guard let disk = try? StimulusSet(directory: directory) else {
            return !positives.isEmpty
        }
        if disk.positive != positives || disk.negative != negatives { return true }
        guard let modelID = host?.loadedModelID else { return false }
        // Newest artifact for this concept+model must match current options.
        let newest = VectorCatalog.scan().first {
            $0.sidecar.concept == name && $0.sidecar.modelID == modelID
        }
        guard let newest else { return true }
        return newest.sidecar.extractionMethod != extractionMethod.rawValue
            || (newest.sidecar.readingPosition ?? ReadingPosition.lastToken.label)
                != extractionOptions.readingPosition.label
            || Self.renderingLabel(newest.sidecar.extractionRendering)
                != extractionOptions.resolvedExtractionRendering.label
    }

    /// An artifact's rendering as a comparable label. An ABSENT stamp is the
    /// legacy raw rendering, never "unknown" — that is the whole convention.
    nonisolated static func renderingLabel(_ rendering: ExtractionRendering?) -> String {
        (rendering ?? .raw).label
    }

    public var extractionOptions: ExtractionOptions {
        ExtractionOptions(
            method: extractionMethod,
            readingPosition: readingPosition,
            extractionRendering: extractionRendering)
    }

    /// Activations keyed by "modelID|text"; values are [layer][hidden].
    private var activationCache: [String: [[Float]]] = [:]
    private var lastDirection: (model: String, vector: [Float])?

    public init() {
        refreshConceptList()
        refreshReaderTemplates()
        if let first = existingConcepts.first {
            selectedExisting = first
            vectorBuilderSelectedExisting = first
            loadSelectedExisting()
        }
    }

    public var emotionCorpusSummary: EmotionCorpusSummary {
        let target = currentName
        let buildRows = emotionRowsForExtraction()
        let selectedRows = emotionRowsForSelectedConcepts()
        let conceptCounts = counts(buildRows.map(\.concept))
        let topicCounts = counts(buildRows.map { normalizedTopic($0.topic) })
        let concepts = conceptCounts.map(\.name)
        let topics = topicCounts.map(\.name)
        var cellMap: [String: Int] = [:]
        for row in buildRows {
            cellMap["\(normalizedTopic(row.topic))|\(row.concept)", default: 0] += 1
        }
        var cells: [(topic: String, concept: String, count: Int)] = []
        var missing: [(topic: String, concept: String)] = []
        var minCell = Int.max
        var maxCell = 0
        for topic in topics {
            for concept in concepts {
                let count = cellMap["\(topic)|\(concept)", default: 0]
                cells.append((topic, concept, count))
                if count == 0 { missing.append((topic, concept)) }
                minCell = min(minCell, count)
                maxCell = max(maxCell, count)
            }
        }
        if cells.isEmpty { minCell = 0 }
        return EmotionCorpusSummary(
            targetConcept: target,
            totalRowCount: selectedRows.count,
            rowCount: buildRows.count,
            validationRowCount: selectedRows.count {
                Self.canonicalSplit($0.split) == "validation"
            },
            draftRowCount: selectedRows.count {
                Self.canonicalSplit($0.split) == "draft"
            },
            targetRowCount: buildRows.count { $0.concept == target },
            conceptCounts: conceptCounts,
            topicCounts: topicCounts,
            cellCounts: cells,
            missingCells: missing,
            minCellCount: minCell,
            maxCellCount: maxCell)
    }

    public var selectedEmotionRows: [StimulusSet.MultiConceptStimulus] {
        emotionRowsForSelectedConcepts()
    }

    public var emotionBuildTopics: [String] {
        counts(
            emotionRowsForSelectedConcepts()
                .filter { Self.canonicalSplit($0.split) == "build" }
                .map { normalizedTopic($0.topic) }
        ).map(\.name)
    }

    public var emotionConceptOptions: [String] {
        let fm = FileManager.default
        var names = Set(existingConcepts.map(Self.sanitizedName).filter { !$0.isEmpty })
        if
            let entries = try? fm.contentsOfDirectory(
                at: VectorCatalog.emotionsDirectory,
                includingPropertiesForKeys: [.isDirectoryKey])
        {
            for entry in entries
            where (try? entry.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
            {
                let stories = entry.appending(component: "stories.jsonl")
                if fm.fileExists(atPath: stories.path) {
                    names.insert(entry.lastPathComponent)
                }
            }
        }
        if !currentName.isEmpty, !multiConceptRows.isEmpty { names.insert(currentName) }
        return names.sorted()
    }

    public func emotionTopicIsIncluded(_ topic: String) -> Bool {
        includedEmotionTopics.isEmpty || includedEmotionTopics.contains(topic)
    }

    public func setEmotionTopic(_ topic: String, included: Bool) {
        let allTopics = Set(emotionBuildTopics)
        var selected = includedEmotionTopics.isEmpty ? allTopics : includedEmotionTopics
        if included {
            selected.insert(topic)
        } else {
            selected.remove(topic)
        }
        includedEmotionTopics = selected
        noteMutation()
    }

    public func includeAllEmotionTopics() {
        includedEmotionTopics = []
        noteMutation()
    }

    public func setIncludedEmotionTopics(_ topics: Set<String>) {
        includedEmotionTopics = topics
        noteMutation()
    }

    public func emotionConceptIsIncluded(_ concept: String) -> Bool {
        includedEmotionConcepts.isEmpty || includedEmotionConcepts.contains(concept)
    }

    public func grandMeanConceptWillBuild(_ concept: String) -> Bool {
        grandMeanTargetConceptNames.contains(Self.sanitizedName(concept))
    }

    public func setEmotionConcept(_ concept: String, included: Bool) {
        let concept = Self.sanitizedName(concept)
        let allConcepts = Set(emotionConceptOptions)
        var selected = includedEmotionConcepts.isEmpty ? allConcepts : includedEmotionConcepts
        if included {
            selected.insert(concept)
        } else {
            selected.remove(concept)
            if grandMeanBuildConceptsAreExplicit {
                grandMeanBuildConcepts.remove(concept)
            }
        }
        includedEmotionConcepts = selected
        noteMutation()
    }

    public func setGrandMeanBuildConcept(_ concept: String, build: Bool) {
        let concept = Self.sanitizedName(concept)
        let included = selectedEmotionConceptNames()
        if !grandMeanBuildConceptsAreExplicit {
            grandMeanBuildConcepts = included
            grandMeanBuildConceptsAreExplicit = true
        }
        if build {
            if includedEmotionConcepts.isEmpty {
                includedEmotionConcepts = Set(emotionConceptOptions)
            }
            includedEmotionConcepts.insert(concept)
            grandMeanBuildConcepts.insert(concept)
        } else {
            grandMeanBuildConcepts.remove(concept)
        }
        noteMutation()
    }

    public func includeAllEmotionConcepts() {
        includedEmotionConcepts = []
        grandMeanBuildConceptsAreExplicit = false
        grandMeanBuildConcepts = []
        noteMutation()
    }

    public func setIncludedEmotionConcepts(_ concepts: Set<String>) {
        includedEmotionConcepts = Set(concepts.map(Self.sanitizedName).filter { !$0.isEmpty })
        if grandMeanBuildConceptsAreExplicit {
            grandMeanBuildConcepts = grandMeanBuildConcepts.intersection(includedEmotionConcepts)
        }
        noteMutation()
    }

    // MARK: - Concept list / loading

    public func refreshConceptList() {
        existingConcepts = VectorCatalog.conceptNames()
        if let selected = vectorBuilderSelectedExisting,
            !existingConcepts.contains(selected)
        {
            vectorBuilderSelectedExisting = existingConcepts.first
        } else if vectorBuilderSelectedExisting == nil {
            vectorBuilderSelectedExisting = existingConcepts.first
        }
    }

    /// THE model-layer seam for "open this concept in the builder": refresh
    /// the index so a concept authored outside this app session (or seconds
    /// ago by the New Dataset flow) is a valid option, then set the dataset
    /// selection — whose `didSet` loads its files and moves the build target
    /// with it.
    ///
    /// One seam, so every route in gets the same behaviour: the Data
    /// section's inventory routing, the Concepts panel's own New Dataset
    /// hand-off, and the web client's `/api/concept/select`. Before this
    /// existed the routing was open-coded in the view, which is how the two
    /// selections drifted apart in the first place.
    ///
    /// Returns false when the name is not (or is no longer) a concept in this
    /// workspace — the selection is then left alone rather than set to
    /// something the picker cannot show.
    @discardableResult
    public func selectConcept(_ name: String) -> Bool {
        refreshConceptList()
        guard existingConcepts.contains(name) else { return false }
        selectedExisting = name
        // A repeat selection is a no-op for `didSet`; make the sync explicit
        // so re-routing to the already-selected concept still lands the build
        // target on it.
        vectorBuilderSelectedExisting = name
        return true
    }

    public func startNewConcept() {
        selectedExisting = nil
        conceptName = ""
        positiveDraft = ""
        negativeDraft = ""
        pairedJSONLDraft = ""
        multiConceptDraft = ""
        probeDraft = ""
        multiConceptStoryConceptDraft = ""
        multiConceptTopicDraft = ""
        multiConceptSplitDraft = "build"
        positives = []
        negatives = []
        multiConceptRows = []
        probeExamples = []
        includedEmotionTopics = []
        includedEmotionConcepts = []
        grandMeanBuildConcepts = []
        grandMeanBuildConceptsAreExplicit = false
        stats = nil
        lastDirection = nil
        status = "new concept"
        refreshStaleness()
    }

    /// Persist the concept primitive before any recipe-specific dataset
    /// exists. A small metadata file keeps the otherwise-empty directory in
    /// git; recipe datasets remain independent children added later.
    public func saveNewConcept() {
        let name = Self.sanitizedName(conceptName)
        guard !name.isEmpty else {
            promptNotice("enter a concept name before saving")
            return
        }
        let directory = VectorCatalog.conceptsDirectory.appending(component: name)
        do {
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            let metadataURL = directory.appending(component: "concept.json")
            if !FileManager.default.fileExists(atPath: metadataURL.path) {
                let metadata: [String: Any] = [
                    "artifactType": "steerlab-concept",
                    "schemaVersion": 1,
                    "name": name,
                    "createdAt": ISO8601DateFormatter().string(from: Date()),
                ]
                let data = try JSONSerialization.data(
                    withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys])
                try data.write(to: metadataURL, options: .atomic)
            }
            conceptName = name
            refreshConceptList()
            selectedExisting = name
            vectorBuilderSelectedExisting = name
            status = "saved concept \(name) — choose a recipe and create its dataset"
            if let host {
                _ = host.startLiveLog(
                    title: "Concept — \(name)",
                    initialLine: "Saved the concept primitive. Choose a vector recipe, copy its dataset prompt, paste/import the returned data, then build a model-specific vector.")
            }
        } catch {
            promptNotice("could not save concept \(name): \(error)")
        }
    }

    /// UI-surfaced status feedback (e.g. "prompt copied").
    public func setStatus(_ message: String) {
        status = message
    }

    public func removeMultiConceptRow(index: Int) {
        guard multiConceptRows.indices.contains(index) else { return }
        let removed = multiConceptRows.remove(at: index)
        do {
            try writeEmotionRows(multiConceptRows, for: currentName)
            status = "removed 1 \(removed.concept) story row"
            noteMutation()
        } catch {
            status = "remove failed: \(error)"
        }
    }

    public func removeContrastiveStimulus(isPositive: Bool, index: Int) {
        let name = currentName
        guard !name.isEmpty else { return }
        if isPositive {
            guard positives.indices.contains(index) else { return }
            positives.remove(at: index)
        } else {
            guard negatives.indices.contains(index) else { return }
            negatives.remove(at: index)
        }
        do {
            let directory = VectorCatalog.conceptsDirectory.appending(component: name)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try jsonl(positives).write(
                to: directory.appending(component: "positive.jsonl"),
                atomically: true,
                encoding: .utf8)
            try jsonl(negatives).write(
                to: directory.appending(component: "negative.jsonl"),
                atomically: true,
                encoding: .utf8)
            status = "removed 1 \(isPositive ? "positive" : "negative") stimulus"
            noteMutation()
        } catch {
            status = "remove failed: \(error)"
        }
    }

    public func removeEmotionRow(_ row: StimulusSet.MultiConceptStimulus) {
        let concept = Self.sanitizedName(row.concept)
        guard !concept.isEmpty else { return }
        do {
            var rows = (try? loadEmotionRows(for: concept)) ?? []
            let targetIdentity = Self.rowIdentity(row)
            guard let index = rows.firstIndex(where: { Self.rowIdentity($0) == targetIdentity }) else {
                status = "could not find that \(concept) story row on disk"
                return
            }
            rows.remove(at: index)
            try writeEmotionRows(rows, for: concept)
            if concept == currentName {
                multiConceptRows = rows
            }
            status = "removed 1 \(concept) story row"
            noteMutation()
        } catch {
            status = "remove failed: \(error)"
        }
    }

    public func deleteSelectedConcept() {
        let name = currentName
        guard !name.isEmpty else { return }
        let fm = FileManager.default
        let roots = [
            VectorCatalog.conceptsDirectory.appending(component: name),
            VectorCatalog.emotionsDirectory.appending(component: name),
            VectorCatalog.probesDirectory.appending(component: name),
            // Unchanged scope: the RepE mirror is swept, the readers mirror
            // is not (a fitted reader's pinned dataset outlives the concept's
            // editable stimuli). Only the PATH moved to the authority.
            VectorCatalog.pairedStimuliDirectory(family: .repe, name: name),
        ]
        do {
            for url in roots where fm.fileExists(atPath: url.path) {
                try fm.removeItem(at: url)
            }
            refreshConceptList()
            if let first = existingConcepts.first {
                selectedExisting = first
                vectorBuilderSelectedExisting = first
            } else {
                vectorBuilderSelectedExisting = nil
                startNewConcept()
            }
            status = "deleted editable datasets for \(name); reader pair sets are kept, "
                + "and saved vector artifacts remain in runs/"
        } catch {
            status = "delete failed: \(error)"
        }
    }

    private func loadSelectedExisting() {
        stats = nil
        lastDirection = nil
        guard let name = selectedExisting else {
            conceptName = ""
            positives = []
            negatives = []
            multiConceptRows = []
            probeExamples = []
            includedEmotionTopics = []
            includedEmotionConcepts = []
            grandMeanBuildConcepts = []
            grandMeanBuildConceptsAreExplicit = false
            return
        }
        conceptName = name
        let directory = VectorCatalog.conceptsDirectory.appending(component: name)
        var notes: [String] = []
        if let set = try? StimulusSet(directory: directory) {
            positives = set.positive
            negatives = set.negative
            if !set.positive.isEmpty || !set.negative.isEmpty {
                let inferred = Self.pairedRecipeFamilyOnDisk(for: name)
                if let family = inferred.family {
                    recipeFamily = family
                } else {
                    // Unknowable, not guessable: keep the standing selection
                    // and say why, rather than flipping the picker (and with
                    // it "Copy LLM prompt") on no evidence.
                    promptNotice(inferred.source)
                }
                notes.append("\(set.positive.count)+\(set.negative.count) stimuli")
            }
        } else {
            positives = []
            negatives = []
        }
        multiConceptRows = (try? loadEmotionRows(for: name)) ?? []
        if !multiConceptRows.isEmpty {
            if positives.isEmpty, negatives.isEmpty {
                // A story corpus alone cannot distinguish grand-mean from
                // designated-reference (the file layout is shared on
                // purpose), but this concept's newest artifact can — the same
                // must-restore-ITS-recipe rule `pairedRecipeFamilyOnDisk`
                // exists for. The artifact also carries the reference pin, so
                // the picker comes back showing the recipe that built it.
                if let reference = Self.designatedReferencePinOnDisk(for: name) {
                    recipeFamily = .designatedReference
                    designatedReferenceConcept = reference
                } else {
                    recipeFamily = .emotionGrandMean
                }
            }
            notes.append("\(multiConceptRows.count) story rows")
        }
        if positives.isEmpty, negatives.isEmpty, multiConceptRows.isEmpty {
            status = "loaded concept \(name) — no recipe dataset yet"
        } else {
            status = "loaded " + notes.joined(separator: " · ")
        }
        if recipeFamily != .emotionGrandMean {
            includedEmotionTopics = []
            includedEmotionConcepts = []
            grandMeanBuildConcepts = []
            grandMeanBuildConceptsAreExplicit = false
        } else if positives.isEmpty, negatives.isEmpty {
            includedEmotionTopics = []
            includedEmotionConcepts = []
            grandMeanBuildConcepts = []
            grandMeanBuildConceptsAreExplicit = false
        }
        probeExamples = (try? loadProbeExamples(for: name)) ?? []
        if positives.isEmpty, negatives.isEmpty, multiConceptRows.isEmpty,
            !probeExamples.isEmpty
        {
            status = "loaded \(probeExamples.count) probe examples"
        }
        probeDraft = ""
        refreshStaleness()
    }

    /// The reference pin from a concept's newest designated-reference
    /// artifact, or nil when its newest artifact was built by another recipe
    /// (or no artifact exists). This is RECORDED PROVENANCE in the same sense
    /// as `recordedRecipeFamily` — `VectorCatalog.scan` enumerates run
    /// directories newest-first by their timestamped names, never by a
    /// filesystem attribute — so the selection restores the recipe the
    /// researcher actually ran, reference pin and all.
    nonisolated static func designatedReferencePinOnDisk(
        for name: String, root: URL? = nil
    ) -> String? {
        let runs = VectorCatalog.runsDirectory(root: root ?? VectorCatalog.projectRoot)
        guard
            let newest = VectorCatalog.scan(runsDirectory: runs).first(where: {
                $0.sidecar.concept == name
            }),
            VectorRecipeGrouping.normalizedMethod(
                recipeMethod: newest.sidecar.recipeMethod,
                extractionMethod: newest.sidecar.extractionMethod)
                == ExtractionMethod.designatedReference.rawValue
        else { return nil }
        return newest.sidecar.designatedReference?["name"]
    }

    /// Which paired recipe produced a concept's on-disk dataset, and WHERE
    /// that answer came from. The shared paired files
    /// (`prompts/concepts/<name>/{positive,negative}.jsonl`) cannot
    /// distinguish CAA from paired-difference-PCA rows by shape, but the
    /// recipe-specific mirrors can: a paired-difference-PCA vector build also
    /// writes `prompts/repe/<name>/pairs.jsonl`, and a reader fit writes
    /// `prompts/readers/<name>/pairs.jsonl`. Selecting an existing concept
    /// must restore ITS recipe — before 2026-07-14 this defaulted the picker
    /// (and therefore "Copy LLM prompt") back to CAA for every paired
    /// concept, so a paired-difference-PCA concept quietly copied the CAA
    /// dataset prompt.
    ///
    /// **The tie-break is RECORDED PROVENANCE, not file mtimes (audit finding
    /// 4, fixed 2026-08-27).** When both mirrors exist the old rule picked
    /// whichever file the filesystem said was touched last — a quantity that
    /// a `git checkout`, a workspace copy, an rsync, a bundle unpack, or an
    /// editor's save-in-place rewrites wholesale, and that says nothing about
    /// which recipe the researcher actually ran. The mirrors answer only when
    /// exactly one exists (then they are unambiguous); otherwise the question
    /// is put to the artifacts the concept has actually produced — a fitted
    /// reader artifact, or the newest vector artifact's sidecar
    /// `recipeMethod`. When nothing is recorded either, the answer is **nil**:
    /// the family genuinely is not knowable from disk, and the caller keeps
    /// its current selection and says so, rather than switching the picker on
    /// a coin flip.
    ///
    /// Returns `(family, source)`; `source` is a phrase naming the evidence,
    /// for the status line.
    nonisolated static func pairedRecipeFamilyOnDisk(
        for name: String, root: URL? = nil
    ) -> (family: RecipeFamily?, source: String) {
        let base = root ?? VectorCatalog.projectRoot
        let fm = FileManager.default
        let hasRepe = fm.fileExists(
            atPath: VectorCatalog.pairedStimuliFile(
                family: .repe, name: name, root: base).path)
        let hasReaders = fm.fileExists(
            atPath: VectorCatalog.pairedStimuliFile(
                family: .readers, name: name, root: base).path)
        switch (hasRepe, hasReaders) {
        case (false, false):
            return (
                .caaMeanDifference,
                "no recipe-specific mirror on disk — plain paired data is CAA")
        case (true, false):
            return (.pairedDifferencePCA, "the only recipe mirror on disk (prompts/repe/)")
        case (false, true):
            return (.repeReaderLAT, "the only recipe mirror on disk (prompts/readers/)")
        case (true, true):
            if let recorded = recordedRecipeFamily(for: name, root: base) {
                return recorded
            }
            return (
                nil,
                "both recipe mirrors exist for '\(name)' and no build under it is "
                    + "recorded in runs/ — which recipe this concept belongs to "
                    + "cannot be read off disk, so the picker was left as it was. "
                    + "Repair: pick the recipe deliberately, or build once so the "
                    + "artifact records it")
        }
    }

    /// The recipe family a concept's own ARTIFACTS record: a fitted reader
    /// wins (it is the more specific instrument, and it is what a reader fit
    /// writes), otherwise the newest vector artifact's sidecar. Both catalogs
    /// enumerate run directories newest-first by their timestamped names,
    /// which is recorded provenance rather than a filesystem attribute.
    nonisolated static func recordedRecipeFamily(
        for name: String, root: URL
    ) -> (family: RecipeFamily?, source: String)? {
        let runs = VectorCatalog.runsDirectory(root: root)
        if VectorCatalog.scanReaders(runsDirectory: runs)
            .contains(where: { $0.artifact.concept == name })
        {
            return (.repeReaderLAT, "a fitted reader artifact under runs/")
        }
        for artifact in VectorCatalog.scan(runsDirectory: runs)
        where artifact.sidecar.concept == name {
            if let recipe = artifact.sidecar.recipeMethod,
                let family = RecipeFamily(rawValue: recipe)
            {
                return (family, "the newest vector artifact's recipeMethod '\(recipe)'")
            }
            if let method = artifact.sidecar.extractionMethod,
                let family = Self.family(forExtractionMethod: method)
            {
                return (
                    family,
                    "the newest vector artifact's extractionMethod '\(method)'")
            }
        }
        return nil
    }

    /// Manifest-vocabulary method → builder family, for sidecars that stamp
    /// only `extractionMethod`. nil for a method with no paired family here.
    nonisolated static func family(forExtractionMethod method: String) -> RecipeFamily? {
        switch ExtractionMethod(rawValue: method) {
        case .meanDifference: .caaMeanDifference
        case .pairedDifferencePCA: .pairedDifferencePCA
        case .repeReaderLAT: .repeReaderLAT
        case .emotionGrandMean: .emotionGrandMean
        default: nil
        }
    }

    // MARK: - Adding stimuli

    /// `templatePrompt` with the failure path attached: a missing template
    /// posts an action-needed notice and yields nil, so every copy button
    /// lands in its failure branch (clipboard untouched) instead of copying
    /// a sentinel under a success message.
    private func templatePromptOrNotice(
        filename: String, replacements: [String: String]
    ) -> String? {
        guard
            let text = Self.templatePrompt(
                filename: filename, replacements: replacements)
        else {
            promptNotice(
                "prompt template \(filename) is unavailable — neither the workspace's prompts/generation/ nor the app's seed copy has it; nothing was copied")
            return nil
        }
        return text
    }

    /// The full instruction prompt for generating more pairs with any LLM,
    /// including all current pairs (as style examples and to discourage
    /// duplication) and a strict output format the app can parse back.
    public func generationPrompt() -> String? {
        let name = currentName
        guard !name.isEmpty else {
            promptNotice("save or name the concept before copying a prompt")
            return nil
        }
        switch recipeFamily {
        case .jlensTokenDirection:
            // There is no stimulus set to generate — the inputs are a lens and
            // a token id. Offering a generation prompt here would imply this
            // family has stimulus provenance it does not have.
            promptNotice(
                "a J-lens direction has no stimulus set — pick a token instead "
                    + "of generating stimuli")
            return nil
        case .caaMeanDifference:
            return ClaudeStimulusGenerator.clipboardPrompt(
                concept: name, guidance: generationGuidance,
                examplePositives: positives, exampleNegatives: negatives,
                count: generationCount)
        case .pairedDifferencePCA, .repeReaderLAT:
            // The reader recipe consumes matched positive/negative STIMULI —
            // the task scaffold is applied at fit time from the selected
            // template, so the generation prompt asks for plain scenario
            // pairs, never pre-scaffolded text.
            return templatePromptOrNotice(
                filename: "repe-paired-reader-data.md",
                replacements: [
                    "concept": name,
                    "count": "\(generationCount)",
                    "template_or_scaffold": recipeFamily == .repeReaderLAT
                        ? "Produce plain matched scenario pairs (no scaffold text; "
                            + "the reader task template is applied separately)."
                        : (generationGuidance.isEmpty
                            ? "Use matched prompt fragments or an answer scaffold appropriate to the concept."
                            : generationGuidance),
                ])
        case .emotionGrandMean, .designatedReference:
            // The designated-reference layout IS the story-corpus layout
            // (METHODS amendment ii keeps `prompts/emotions/` for it), so the
            // same corpus prompt authors either class — the reference corpus
            // is authored as its own story concept through this same flow.
            return templatePromptOrNotice(
                filename: "emotion-grand-mean-stories.md",
                replacements: [
                    "concepts": name,
                    "topics": generationGuidance.isEmpty
                        ? "ordinary non-study topics matched across concepts"
                        : generationGuidance,
                    "stories_per_concept_topic": "\(max(1, generationCount))",
                    "concept": name,
                    "topic": "topic",
                ])
        }
    }

    public func coworkGenerationPrompt() -> String? {
        let name = currentName
        guard !name.isEmpty else {
            promptNotice("save or name the concept before copying a cowork prompt")
            return nil
        }
        guard recipeFamily == .emotionGrandMean else {
            promptNotice("cowork corpus prompts are for Grand mean multi-concept story corpora")
            return nil
        }
        let concepts = selectedEmotionConceptNames().sorted()
        let topics = emotionBuildTopics
        let storiesPerCell = max(12, generationCount)
        let expectedRows =
            topics.isEmpty ? "concepts × chosen topics × \(storiesPerCell)" :
            "\(concepts.count * topics.count * storiesPerCell)"
        return templatePromptOrNotice(
            filename: "grand-mean-cowork-agent.md",
            replacements: [
                "concepts": concepts.isEmpty
                    ? name
                    : concepts.joined(separator: ", "),
                "topics": topics.isEmpty
                    ? "choose 6-10 ordinary, study-neutral topics and use the same topics for every concept"
                    : topics.joined(separator: ", "),
                "stories_per_concept_topic": "\(storiesPerCell)",
                "expected_rows": expectedRows,
                "split": "build",
            ])
    }

    public func probeGenerationPrompt() -> String? {
        let name = currentName
        guard !name.isEmpty else {
            status = "name the concept before copying a probe prompt"
            return nil
        }
        return templatePromptOrNotice(
            filename: "probe-validation-items.md",
            replacements: [
                "concept": name,
                "count": "\(max(20, probeGenerationCount))",
            ])
    }

    public func neutralCorpusPrompt() -> String? {
        return templatePromptOrNotice(
            filename: "neutral-norm-corpus.md",
            replacements: [
                "count": "\(max(200, generationCount * 20))",
                "minimum_words": "90",
            ])
    }

    public func anthropicStyleNeutralDialoguePrompt() -> String? {
        let concepts = projectionNeutralConceptsDraft
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let domains = projectionNeutralDomainsDraft
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let exclusions = projectionNeutralExclusionsDraft
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return templatePromptOrNotice(
            filename: "neutral-dialogues-anthropic-style.md",
            replacements: [
                "count": "\(max(200, generationCount * 20))",
                "minimum_words": "120",
                "neutral_concepts": concepts.isEmpty
                    ? "the concept set currently under study; ask the user to provide concepts before generating final data"
                    : concepts,
                "matched_domains": domains.isEmpty
                    ? "coding, factual questions, math, geography, workplace summarization, practical how-to, formatting, classification, list generation, simple planning, household tasks, inventory, scheduling, and technical explanation"
                    : domains,
                "avoid_settings": exclusions.isEmpty
                    ? "danger, illness, violence, conflict, romance, death, law, politics, religion, identity groups, moral judgment, persuasion, experiments, AI safety, or model behavior"
                    : exclusions,
            ])
    }

    public func recordCopiedPrompt(_ prompt: String, message: String) {
        lastCopiedPrompt = prompt
        status = message
        guard let host else { return }
        let title = "Dataset prompt — \(recipeFamily.label)"
        let line =
            message
            + "\nThe full prompt is available under Last copied prompt. Run it in your LLM, then paste the JSONL result into the visible dataset editor or import it where offered."
        if let copiedPromptLogID,
            host.updateLiveLog(id: copiedPromptLogID, title: title, lines: [line])
        {
            return
        }
        copiedPromptLogID = host.startLiveLog(title: title, initialLine: line)
    }

    /// Clipboard-failure twin of `recordCopiedPrompt`: the pasteboard write
    /// failed, so the "Last copied prompt" disclosure is the manual-recovery
    /// path — it must hold the generated prompt precisely when the clipboard
    /// does not. The failure itself is still reported as a notice.
    public func recordCopyFailure(_ prompt: String, message: String) {
        lastCopiedPrompt = prompt
        promptNotice(message)
    }

    /// Show or update one stable recipe guide in Activity instead of adding a
    /// fresh log bubble each time the same selector changes elsewhere.
    ///
    /// Rule (reviewer finding, 2026-07-14): the guide must EXIST whenever the
    /// panel asks for it — recreate-when-absent, suppress only live-entry
    /// churn. A live entry is updated in place (same id, so same-family
    /// appearances never append a second bubble); an absent entry (chat sends
    /// clear live logs) is always re-created, same family included —
    /// returning to Concept Lab after a clear must not leave the requested
    /// explanation missing.
    public func presentRecipeGuide() {
        guard let host else { return }
        let lines = recipeFamily.activityGuide
        let title = "Technique — \(recipeFamily.label)"
        if let recipeGuideLogID,
            host.updateLiveLog(id: recipeGuideLogID, title: title, lines: lines)
        {
            return
        }
        recipeGuideLogID = host.startLiveLog(
            title: title, initialLine: lines.joined(separator: "\n"))
    }

    private func promptNotice(_ message: String) {
        status = message
        guard let host else { return }
        let title = "Dataset prompt — action needed"
        if let promptNoticeLogID,
            host.updateLiveLog(id: promptNoticeLogID, title: title, lines: [message])
        {
            return
        }
        promptNoticeLogID = host.startLiveLog(title: title, initialLine: message)
    }

    public func importNeutralCorpusDraft() {
        do {
            let texts = try NeutralCorpusStore.parseTexts(Data(neutralCorpusDraft.utf8))
            guard !texts.isEmpty else {
                status = "neutral corpus paste did not contain any text rows"
                return
            }
            let target = host?.selectedNeutralCorpusForImport()
                ?? NeutralCorpusStore.record(id: NeutralCorpusStore.normCorpusID)
            let saved = try NeutralCorpusStore.save(texts: texts, to: target)
            host?.refreshNeutralCorpora()
            neutralCorpusDraft = ""
            status =
                "saved \(saved.label) (\(texts.count) rows"
                + (saved.hash.map { ", \(String($0.prefix(12)))" } ?? "")
                + ")"
        } catch {
            status = "couldn't import neutral norm corpus: \(error)"
        }
    }

    /// Appends the pasted lines (one stimulus per line) to the working set.
    /// Stats go stale rather than rebuilding — new stimuli cost one forward
    /// pass each, and the user decides when to spend that.
    ///
    /// Pasted pair-JSONL (the copy-prompt round trip) is detected in either
    /// box and parsed as pairs.
    public func addDrafts() async {
        if recipeFamily.usesStoryCorpus {
            do {
                let rows = try Self.parseMultiConceptRows(
                    Data(multiConceptDraft.utf8),
                    filename: "pasted stories",
                    defaultConcept: storyConceptForDraft(),
                    defaultTopic: multiConceptTopicDraft.trimmingCharacters(
                        in: .whitespacesAndNewlines).isEmpty
                        ? "manual"
                        : multiConceptTopicDraft.trimmingCharacters(
                            in: .whitespacesAndNewlines),
                    defaultSplit: multiConceptSplitDraft.trimmingCharacters(
                        in: .whitespacesAndNewlines).isEmpty
                        ? "build"
                        : multiConceptSplitDraft.trimmingCharacters(
                            in: .whitespacesAndNewlines))
                let addedByConcept = try persistEmotionRowsByConcept(rows)
                let name = currentName
                multiConceptRows = try loadEmotionRows(for: name)
                multiConceptDraft = ""
                let added = addedByConcept.values.reduce(0, +)
                let concepts = addedByConcept.keys.sorted().joined(separator: ", ")
                status = "distributed \(added) story rows"
                    + (concepts.isEmpty ? "" : " across \(concepts)")
                    + (added < rows.count ? " (\(rows.count - added) duplicate(s) skipped)" : "")
                refreshConceptList()
                noteMutation()
            } catch {
                status = "couldn't parse story JSONL: \(error)"
            }
            return
        }

        let pairedJSONL = pairedJSONLDraft.trimmingCharacters(
            in: .whitespacesAndNewlines)
        if !pairedJSONL.isEmpty {
            do {
                let pairs = try Self.parsePairs(
                    Data(pairedJSONL.utf8), filename: "pasted paired JSONL")
                pairedJSONLDraft = ""
                addPairs(pairs, source: "pasted paired JSONL")
            } catch {
                status = "couldn't parse paired JSONL: \(error)"
            }
            return
        }

        for draft in [positiveDraft, negativeDraft] {
            do {
                guard let pairs = try Self.parsePastedPairs(draft) else { continue }
                positiveDraft = ""
                negativeDraft = ""
                addPairs(pairs, source: "pasted JSONL")
            } catch {
                status = "couldn't parse pasted JSONL: \(error)"
            }
            return
        }

        let newPositives = parse(positiveDraft)
        let newNegatives = parse(negativeDraft)
        guard !newPositives.isEmpty || !newNegatives.isEmpty else { return }

        let existingPositives = Set(positives)
        let existingNegatives = Set(negatives)
        let addedPositives = newPositives.filter { !existingPositives.contains($0) }
        let addedNegatives = newNegatives.filter { !existingNegatives.contains($0) }
        let skipped =
            (newPositives.count - addedPositives.count)
            + (newNegatives.count - addedNegatives.count)

        positives.append(contentsOf: addedPositives)
        negatives.append(contentsOf: addedNegatives)
        positiveDraft = ""
        negativeDraft = ""

        var notes: [String] = []
        if skipped > 0 { notes.append("\(skipped) duplicate(s) skipped") }
        if positives.count != negatives.count {
            notes.append(
                "unbalanced set (\(positives.count)+ / \(negatives.count)−) — "
                    + "content-matched pairs control confounds best")
        }
        status = notes.isEmpty ? nil : notes.joined(separator: "; ")

        noteMutation()
    }

    public func addProbeDrafts() async {
        let name = currentName
        guard !name.isEmpty else {
            status = "name the concept before importing probe examples"
            return
        }
        do {
            let incoming = try Self.parseProbeExamples(
                Data(probeDraft.utf8),
                filename: "pasted probe examples",
                concept: name)
            let existingKeys = Set(probeExamples.map(Self.probeIdentity))
            let added = incoming.filter { !existingKeys.contains(Self.probeIdentity($0)) }
            probeExamples.append(contentsOf: added)
            try writeProbeExamples(probeExamples, for: name)
            probeDraft = ""
            refreshConceptList()
            let skipped = incoming.count - added.count
            status = "imported \(added.count) probe examples"
                + (skipped > 0 ? " (\(skipped) duplicate(s) skipped)" : "")
        } catch {
            status = "couldn't parse probe examples: \(error)"
        }
    }

    public func removeProbeExample(index: Int) {
        let name = currentName
        guard !name.isEmpty, probeExamples.indices.contains(index) else { return }
        probeExamples.remove(at: index)
        do {
            try writeProbeExamples(probeExamples, for: name)
            status = "removed 1 probe example"
        } catch {
            status = "remove probe example failed: \(error)"
        }
    }

    public func trainReadingProbe() async {
        guard !isWorking else { return }
        guard let host else {
            status = "load a model before training a probe"
            return
        }
        let name = currentName
        guard !name.isEmpty else {
            status = "select or name a concept before training a probe"
            return
        }
        let split = Self.probeTrainValidationSplit(probeExamples)
        let positives = split.train.filter(\.expresses).map(\.text)
        let negatives = split.train.filter { !$0.expresses }.map(\.text)
        let validationPositives = split.validation.filter(\.expresses).map(\.text)
        let validationNegatives = split.validation.filter { !$0.expresses }.map(\.text)
        guard positives.count >= 2, negatives.count >= 2 else {
            status = "need at least 2 build concept-present and 2 build control probe examples"
            return
        }
        guard validationPositives.count + validationNegatives.count >= 2 else {
            status = "need held-out validation probe examples before training a probe"
            return
        }

        isWorking = true
        activeTaskLabel = "Training probe"
        defer {
            isWorking = false
            activeTaskLabel = nil
        }

        // Probe fitting reads activations, so the same rule applies: the
        // selected model drives it, and a probe stamped with the wrong model
        // would silently be unusable for the vectors it is meant to check.
        do {
            try await ensureSelectedLocalModelLoaded(host: host)
        } catch {
            reportBuildFailure(
                "probe training did not start: \(error)", title: "Probe Training — failed")
            return
        }
        guard let container = host.containerForExtraction,
            let modelID = host.loadedModelID
        else {
            status = "load a model before training a probe"
            return
        }

        // A probe must be trained the way the concept is READ (audit finding
        // 4, fixed 2026-08-27). This flow used to hard-code `.lastToken` and
        // `meanDifference` and then stamp the recipe `caaMeanDifference`,
        // whatever the concept declared: a pooled-at-50 grand-mean concept, or
        // a paired-difference-PCA one, got a probe fitted at a different
        // position with a different direction and a recipe that named neither.
        // A probe fitted somewhere the vectors are not read is not a check on
        // those vectors; it is a different measurement wearing their name.
        guard recipeFamily.extractsFromStimuli else {
            status = "\(recipeFamily.label) reads no activations — there is nothing "
                + "for a probe to be fitted on. Repair: train a probe under a "
                + "family that extracts from stimuli"
            return
        }
        guard readingPositionRefusal == nil, extractionRenderingRefusal == nil else {
            status = "probe training did not start: the declared reading position or "
                + "rendering stands refused — repair the declaration first"
            return
        }
        let probePosition = readingPosition
        let probeRendering = extractionRendering
        let probeMethod = extractionMethod
        do {
            status = "recording probe activations..."
            func probeActivations(_ texts: [String]) async throws -> StimulusActivations {
                try await ConceptExtractor.activations(
                    container: container, texts: texts, position: probePosition,
                    rendering: probeRendering ?? .raw)
            }
            let positive = try await probeActivations(positives)
            let negative = try await probeActivations(negatives)
            let validationPositive = validationPositives.isEmpty
                ? nil : try await probeActivations(validationPositives)
            let validationNegative = validationNegatives.isEmpty
                ? nil : try await probeActivations(validationNegatives)
            guard let layerCount = positive.values.first?.count,
                layerCount > 0,
                negative.values.allSatisfy({ $0.count == layerCount }),
                validationPositive?.values.allSatisfy({ $0.count == layerCount }) ?? true,
                validationNegative?.values.allSatisfy({ $0.count == layerCount }) ?? true
            else {
                status = "probe training failed: layer mismatch"
                return
            }

            var best: (layer: Int, probe: SteeringVectorMath.ScalarProbe, accuracy: Float)?
            for layer in 0 ..< layerCount {
                let pos = positive.values.map { $0[layer] }
                let neg = negative.values.map { $0[layer] }
                let direction = try SteeringVectorMath.direction(
                    positive: pos, negative: neg, method: probeMethod)
                let center = try SteeringVectorMath.mean(pos + neg)
                let probe = try SteeringVectorMath.scalarProbe(
                    direction: direction,
                    positive: pos,
                    negative: neg,
                    activationCenter: center)
                let validationPos = validationPositive?.values.map { $0[layer] } ?? []
                let validationNeg = validationNegative?.values.map { $0[layer] } ?? []
                let correctPositive = try validationPos.filter { try probe.classifiesPositive($0) }
                    .count
                let correctNegative = try validationNeg.filter {
                    !(try probe.classifiesPositive($0))
                }.count
                let validationCount = validationPos.count + validationNeg.count
                let accuracy = Float(correctPositive + correctNegative) / Float(validationCount)
                if let current = best {
                    let margin = abs(probe.positiveMean - probe.negativeMean) / probe.projectionScale
                    let currentMargin =
                        abs(current.probe.positiveMean - current.probe.negativeMean)
                        / current.probe.projectionScale
                    if accuracy > current.accuracy
                        || (accuracy == current.accuracy && margin > currentMargin)
                    {
                        best = (layer, probe, accuracy)
                    }
                } else {
                    best = (layer, probe, accuracy)
                }
            }

            guard let best else {
                status = "probe training failed: no usable layers"
                return
            }
            try writeProbeExamples(probeExamples, for: name)
            let calibrationText = probeJSONL(split.train)
            let validationText = probeJSONL(split.validation)
            let calibrationHash = Self.sha256(calibrationText)
            let validationHash = Self.sha256(validationText)
            let recipe = VectorExtractionRecipe(
                name: "\(name)-reading-probe",
                method: recipeFamily.recipeMethod,
                targetConcept: name,
                datasets: [
                    .init(
                        role: .probeCalibration,
                        path: "prompts/probes/\(name)/items.jsonl",
                        sha256: calibrationHash)
                ],
                readingPosition: probePosition,
                extractionRendering: probeRendering,
                notes: "Scalar reading probe trained from labeled build examples, "
                    + "read at \(probePosition.label) under the concept's declared "
                    + "\(probeMethod.label) direction. Layer selected by held-out "
                    + "validation accuracy across all layers.")
            let artifact = ReadingProbeArtifact(
                modelID: modelID,
                revision: SteeredContainerLoader.cachedRevision(for: modelID),
                concept: name,
                layer: best.layer,
                recipeName: recipe.name,
                recipeHash: recipe.canonicalHash(),
                calibrationHash: calibrationHash,
                validationHash: validationHash,
                probe: best.probe,
                notes: "Training examples: \(positives.count)+/\(negatives.count)-. Held-out examples: \(validationPositives.count)+/\(validationNegatives.count)-. Validation accuracy \(Int((best.accuracy * 100).rounded()))%.")
            let runDirectory = try VectorCatalog.makeUniqueRunDirectory(slug: "probe-\(name)")
            try RunMetadata.write(
                runType: "probe-train", to: runDirectory,
                modelID: modelID,
                revision: SteeredContainerLoader.cachedRevision(for: modelID))
            let artifactName = "\(name)-probe-L\(best.layer)-\(host.selectedModel.family)"
            let url = try ProbeCatalog.save(artifact, to: runDirectory, name: artifactName)
            host.refreshVectors()
            host.selectedProbeID = url.path
            status = "trained \(name) probe at L\(best.layer) (\(Int((best.accuracy * 100).rounded()))% held-out) → \(runDirectory.lastPathComponent)"
        } catch {
            status = "probe training failed: \(error)"
        }
    }

    nonisolated static func probeTrainValidationSplit(
        _ examples: [ProbeExample]
    ) -> (train: [ProbeExample], validation: [ProbeExample]) {
        let explicitTrain = examples.filter { canonicalSplit($0.split) == "build" }
        let explicitValidation = examples.filter { canonicalSplit($0.split) == "validation" }
        if explicitTrain.contains(where: \.expresses),
            explicitTrain.contains(where: { !$0.expresses }),
            !explicitValidation.isEmpty
        {
            return (explicitTrain, explicitValidation)
        }

        // Fallback split (CROSS-ENGINE CONTRACT, 2026-07-13; the Python
        // engine implements the identical rule): explicitly tagged examples
        // keep their tags; the UNTAGGED pool is sorted ascending by the
        // SHA-256 hex of each example's text, and every 5th of sorted order
        // (0-based index % 5 == 4) goes to validation. Content-derived, so
        // the same items land in the same split on both engines regardless
        // of file order — the historical positional rule diverged whenever
        // row order differed. Pools with fewer than 5 untagged items get NO
        // holdout (the rule applies literally; training then refuses for
        // lack of held-out examples, which is honest).
        var train = explicitTrain
        var validation = explicitValidation
        let untagged = examples.filter {
            let split = canonicalSplit($0.split)
            return split != "build" && split != "validation"
        }
        let sorted = untagged.sorted { sha256($0.text) < sha256($1.text) }
        for (index, example) in sorted.enumerated() {
            if index % 5 == 4 {
                validation.append(example)
            } else {
                train.append(example)
            }
        }
        return (train, validation)
    }

    // MARK: - Import

    /// Imports contrastive pairs from a file produced elsewhere (human
    /// coders or another Claude). Supported formats: JSONL with
    /// {"positive": …, "negative": …} per line; a JSON array of the same
    /// objects; or two-column CSV (optional positive,negative header).
    public func importPairs(from url: URL) async {
        do {
            let accessing = url.startAccessingSecurityScopedResource()
            defer { if accessing { url.stopAccessingSecurityScopedResource() } }
            let data = try Data(contentsOf: url)
            let pairs = try Self.parsePairs(data, filename: url.lastPathComponent)
            addPairs(pairs, source: url.lastPathComponent)
        } catch {
            status = "import failed: \(error)"
        }
    }

    /// Shared append path for imported pairs (file importer and web client).
    func addPairs(_ pairs: [ImportedPair], source: String) {
        let existingPositives = Set(positives)
        let added = pairs.filter { !existingPositives.contains($0.positive) }
        positives.append(contentsOf: added.map(\.positive))
        negatives.append(contentsOf: added.map(\.negative))

        let skipped = pairs.count - added.count
        status = "imported \(added.count) pairs from \(source)"
            + (skipped > 0 ? " (\(skipped) duplicate(s) skipped)" : "")
        noteMutation()
    }

    struct ImportedPair: Decodable {
        let positive: String
        let negative: String
    }

    /// Detect pair-shaped content pasted into one of the legacy manual-side
    /// boxes. The dedicated paired-JSONL box is preferred, but keeping this
    /// path makes old copy/paste instructions and saved user habits work.
    nonisolated static func parsePastedPairs(_ text: String) throws -> [ImportedPair]? {
        let normalized = normalizedPairImportText(text)
        guard !normalized.isEmpty else { return nil }
        let lowercased = normalized.lowercased()
        let looksLikePairs =
            normalized.hasPrefix("{") || normalized.hasPrefix("[")
            || (lowercased.contains("\"positive\"")
                && lowercased.contains("\"negative\""))
        guard looksLikePairs else { return nil }
        return try parsePairs(Data(normalized.utf8), filename: "pasted JSONL")
    }

    nonisolated static func parsePairs(_ data: Data, filename: String) throws -> [ImportedPair] {
        let text = normalizedPairImportText(String(decoding: data, as: UTF8.self))
        guard !text.isEmpty else {
            throw StimulusSetError.empty(filename)
        }

        // JSON array of {"positive","negative"} objects.
        if text.hasPrefix("[") {
            return try JSONDecoder().decode([ImportedPair].self, from: Data(text.utf8))
        }

        // JSONL: one object per line.
        if text.hasPrefix("{") {
            return try text.split(separator: "\n").enumerated().compactMap { index, line in
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard !trimmed.isEmpty else { return nil }
                guard
                    let pair = try? JSONDecoder().decode(
                        ImportedPair.self, from: Data(trimmed.utf8))
                else {
                    throw StimulusSetError.malformedLine(file: filename, line: index + 1)
                }
                return pair
            }
        }

        // CSV: two columns, optional header.
        var pairs: [ImportedPair] = []
        for (index, line) in text.split(separator: "\n").enumerated() {
            let columns = Self.parseCSVColumns(String(line))
            guard columns.count >= 2 else {
                throw StimulusSetError.malformedLine(file: filename, line: index + 1)
            }
            let first = columns[0].trimmingCharacters(in: .whitespaces)
            let second = columns[1].trimmingCharacters(in: .whitespaces)
            if index == 0, first.lowercased() == "positive" { continue }
            if first.isEmpty || second.isEmpty { continue }
            pairs.append(ImportedPair(positive: first, negative: second))
        }
        return pairs
    }

    /// LLMs sometimes wrap otherwise-valid JSONL in a Markdown code fence
    /// despite a no-fences instruction. Accept exactly the fenced payload
    /// (and a stray standalone `jsonl`/`json` language marker) without
    /// weakening row validation or treating JSON objects as one-sided text.
    private nonisolated static func normalizedPairImportText(_ raw: String) -> String {
        let hasFence = raw.contains("```")
        var insideFence = false
        var lines: [String] = []
        for rawLine in raw.components(separatedBy: .newlines) {
            let trimmed = rawLine.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("```") {
                insideFence.toggle()
                continue
            }
            if hasFence, !insideFence { continue }
            if trimmed.lowercased() == "jsonl" || trimmed.lowercased() == "json" {
                continue
            }
            lines.append(rawLine)
        }
        return lines.joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Minimal RFC-4180 two-column field splitter (handles quoted commas).
    private nonisolated static func parseCSVColumns(_ line: String) -> [String] {
        var columns: [String] = []
        var current = ""
        var inQuotes = false
        var characters = line.makeIterator()
        while let character = characters.next() {
            switch character {
            case "\"":
                inQuotes.toggle()
            case "," where !inQuotes:
                columns.append(current)
                current = ""
            default:
                current.append(character)
            }
        }
        columns.append(current)
        return columns
    }

    // MARK: - Claude generation

    public func generateProposals() async {
        guard !isGeneratingProposals else { return }
        let name = currentName
        guard !name.isEmpty else {
            status = "name the concept before generating"
            return
        }
        guard recipeFamily.isPaired else {
            status = "Claude candidate review is currently for paired CAA/RepE data; use Copy LLM prompt for emotion story corpora"
            return
        }
        guard !positives.isEmpty, !negatives.isEmpty else {
            status = "add at least one example pair first — generation extends your examples"
            return
        }

        isGeneratingProposals = true
        defer { isGeneratingProposals = false }
        status = "asking Claude for \(generationCount) candidate pairs…"

        do {
            let generated = try await ClaudeStimulusGenerator.generatePairs(
                concept: name, guidance: generationGuidance,
                examplePositives: positives, exampleNegatives: negatives,
                count: generationCount)
            let existingPositives = Set(positives)
            proposals = generated
                .filter { !existingPositives.contains($0.positive) }
                .map { Proposal(positive: $0.positive, negative: $0.negative) }
            status = proposals.isEmpty
                ? "Claude returned no new pairs"
                : "review \(proposals.count) proposals below — nothing is added until you accept"
        } catch {
            status = "generation failed: \(error)"
        }
    }

    /// Appends the checked proposals to the working set and rebuilds.
    public func acceptIncludedProposals() async {
        let accepted = proposals.filter(\.included)
        guard !accepted.isEmpty else { return }
        positives.append(contentsOf: accepted.map(\.positive))
        negatives.append(contentsOf: accepted.map(\.negative))
        proposals = []
        status = "accepted \(accepted.count) pairs — save to persist"
        noteMutation()
    }

    public func discardProposals() {
        proposals = []
        status = nil
    }

    /// Removes one stimulus from the working set (files on disk are
    /// untouched until save) and rebuilds stats from cached activations.
    public func removeStimulus(isPositive: Bool, index: Int) {
        if isPositive {
            guard positives.indices.contains(index) else { return }
            positives.remove(at: index)
        } else {
            guard negatives.indices.contains(index) else { return }
            negatives.remove(at: index)
        }
        status = "removed 1 stimulus — save to persist"
        noteMutation()  // deletion needs zero new passes → auto-recomputes
    }

    private func parse(_ draft: String) -> [String] {
        draft.split(separator: "\n")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    private struct ProbeExamplePayload: Decodable {
        var id: String?
        var text: String
        var expresses: Bool?
        var label: String?
        var topic: String?
        var split: String?
        var notes: String?
    }

    nonisolated static func parseProbeExamples(
        _ data: Data, filename: String, concept: String
    ) throws -> [ProbeExample] {
        let rawText = String(decoding: data, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !rawText.isEmpty else { throw StimulusSetError.empty(filename) }
        let text = rawText
            .split(separator: "\n")
            .map { String($0).trimmingCharacters(in: .whitespaces) }
            .filter { !$0.hasPrefix("```") }
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let cleanConcept = sanitizedName(concept)

        func convert(_ payload: ProbeExamplePayload, index: Int) throws -> ProbeExample {
            let trimmedText = payload.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedText.isEmpty else {
                throw StimulusSetError.malformedLine(file: filename, line: index)
            }
            let expresses: Bool
            if let value = payload.expresses {
                expresses = value
            } else if let label = payload.label?.lowercased() {
                if ["positive", "pos", "concept", "present", "true", "1"].contains(label) {
                    expresses = true
                } else if ["negative", "neg", "control", "absent", "false", "0"].contains(label) {
                    expresses = false
                } else {
                    throw StimulusSetError.malformedLine(file: filename, line: index)
                }
            } else {
                throw StimulusSetError.malformedLine(file: filename, line: index)
            }
            let trimmedID = payload.id?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let resolvedID = trimmedID.isEmpty
                ? "\(cleanConcept)-probe-\(String(format: "%03d", index))"
                : trimmedID
            return ProbeExample(
                id: resolvedID,
                text: trimmedText,
                expresses: expresses,
                topic: payload.topic,
                split: canonicalSplit(payload.split ?? "validation"),
                notes: payload.notes)
        }

        if text.hasPrefix("[") {
            let payloads = try JSONDecoder().decode(
                [ProbeExamplePayload].self, from: Data(text.utf8))
            return try payloads.enumerated().map { try convert($0.element, index: $0.offset + 1) }
        }

        if text.hasPrefix("{") {
            return try text.split(separator: "\n").enumerated().compactMap { index, line in
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard !trimmed.isEmpty else { return nil }
                guard
                    let payload = try? JSONDecoder().decode(
                        ProbeExamplePayload.self, from: Data(trimmed.utf8))
                else {
                    throw StimulusSetError.malformedLine(file: filename, line: index + 1)
                }
                return try convert(payload, index: index + 1)
            }
        }

        var examples: [ProbeExample] = []
        for (index, line) in text.split(separator: "\n").enumerated() {
            let raw = String(line).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !raw.isEmpty else { continue }
            let lower = raw.lowercased()
            let expresses: Bool
            let body: String
            if lower.hasPrefix("positive:") || lower.hasPrefix("pos:") || lower.hasPrefix("+") {
                expresses = true
                body = raw
                    .replacingOccurrences(of: #"^(positive:|pos:|\+)\s*"#,
                                           with: "", options: [.regularExpression, .caseInsensitive])
            } else if lower.hasPrefix("negative:") || lower.hasPrefix("neg:")
                || lower.hasPrefix("control:") || lower.hasPrefix("-")
            {
                expresses = false
                body = raw
                    .replacingOccurrences(
                        of: #"^(negative:|neg:|control:|-)\s*"#,
                        with: "", options: [.regularExpression, .caseInsensitive])
            } else {
                throw StimulusSetError.malformedLine(file: filename, line: index + 1)
            }
            let trimmedBody = body.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedBody.isEmpty else {
                throw StimulusSetError.malformedLine(file: filename, line: index + 1)
            }
            examples.append(
                ProbeExample(
                    id: "\(cleanConcept)-probe-\(String(format: "%03d", examples.count + 1))",
                    text: trimmedBody,
                    expresses: expresses,
                    topic: nil,
                    split: "validation",
                    notes: nil))
        }
        return examples
    }

    private func loadProbeExamples(for concept: String) throws -> [ProbeExample] {
        let name = Self.sanitizedName(concept)
        guard !name.isEmpty else { return [] }
        let url = VectorCatalog.probesDirectory
            .appending(component: name)
            .appending(component: "items.jsonl")
        guard FileManager.default.fileExists(atPath: url.path) else { return [] }
        return try Self.parseProbeExamples(
            Data(contentsOf: url), filename: url.lastPathComponent, concept: name)
    }

    private func writeProbeExamples(_ examples: [ProbeExample], for concept: String) throws {
        let name = Self.sanitizedName(concept)
        guard !name.isEmpty else { return }
        let directory = VectorCatalog.probesDirectory.appending(component: name)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try probeJSONL(examples).write(
            to: directory.appending(component: "items.jsonl"),
            atomically: true,
            encoding: .utf8)
    }

    private nonisolated static func probeIdentity(_ example: ProbeExample) -> String {
        "\(example.expresses)|\(example.text.lowercased())"
    }

    private func storyConceptForDraft() -> String {
        let typed = multiConceptStoryConceptDraft
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return typed.isEmpty ? currentName : Self.sanitizedName(typed)
    }

    private func loadEmotionRows(for concept: String) throws -> [StimulusSet.MultiConceptStimulus] {
        let name = Self.sanitizedName(concept)
        guard !name.isEmpty else { return [] }
        let url = VectorCatalog.emotionsDirectory
            .appending(component: name)
            .appending(component: "stories.jsonl")
        guard FileManager.default.fileExists(atPath: url.path) else { return [] }
        return try StimulusSet.loadMultiConceptTexts(url: url).rows
            .map(Self.canonicalized)
            .filter { $0.concept == name }
    }

    private func writeEmotionRows(
        _ rows: [StimulusSet.MultiConceptStimulus], for concept: String
    ) throws {
        let name = Self.sanitizedName(concept)
        guard !name.isEmpty else { return }
        let directory = VectorCatalog.emotionsDirectory.appending(component: name)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let filtered = rows.map(Self.canonicalized).filter { $0.concept == name }
        try multiConceptJSONL(filtered).write(
            to: directory.appending(component: "stories.jsonl"),
            atomically: true,
            encoding: .utf8)
    }

    private func writeNeutralCorpus(_ texts: [String]) throws {
        let url = Self.neutralCorpusURL()
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let rows = texts.map { NeutralTextRow(text: $0) }
        let jsonl = rows.compactMap { row -> String? in
            guard let data = try? JSONEncoder().encode(row) else { return nil }
            return String(decoding: data, as: UTF8.self)
        }
        .joined(separator: "\n") + "\n"
        try jsonl.write(to: url, atomically: true, encoding: .utf8)
    }

    @discardableResult
    private func persistEmotionRowsByConcept(
        _ rows: [StimulusSet.MultiConceptStimulus]
    ) throws -> [String: Int] {
        let grouped = Dictionary(grouping: rows.map(Self.canonicalized), by: \.concept)
        var addedByConcept: [String: Int] = [:]
        for (concept, incoming) in grouped {
            let existing = try loadEmotionRows(for: concept)
            let existingKeys = Set(existing.map(Self.rowIdentity))
            let added = incoming.filter { !existingKeys.contains(Self.rowIdentity($0)) }
            guard !added.isEmpty else {
                addedByConcept[concept] = 0
                continue
            }
            try writeEmotionRows(existing + added, for: concept)
            addedByConcept[concept] = added.count
        }
        return addedByConcept
    }

    private nonisolated static func rowIdentity(
        _ row: StimulusSet.MultiConceptStimulus
    ) -> String {
        row.id ?? "\(row.concept)|\(row.topic ?? "")|\(row.text)"
    }

    private func normalizedTopic(_ topic: String?) -> String {
        let trimmed = (topic ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty ? "manual" : trimmed
    }

    public nonisolated static func canonicalSplit(_ split: String?) -> String {
        let trimmed = (split ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        switch trimmed {
        case "", "draft":
            return "draft"
        case "build", "train", "training":
            return "build"
        case "validation", "validate", "dev", "test", "heldout", "held-out", "holdout":
            return "validation"
        default:
            return trimmed
        }
    }

    private func counts(_ values: [String]) -> [(name: String, count: Int)] {
        Dictionary(grouping: values, by: { $0 })
            .map { (name: $0.key, count: $0.value.count) }
            .sorted {
                $0.count == $1.count ? $0.name < $1.name : $0.count > $1.count
            }
    }

    // MARK: - Rebuild

    /// Set when a rebuild request arrives while one is running; the running
    /// rebuild loops once more so the final options/stimuli always win.
    private var rebuildQueued = false

    public func rebuild() async {
        guard recipeFamily.isPaired else {
            status = "live pair-margin stats are for CAA/RepE paired data; "
                + "use Save & generate vector to extract the "
                + "\(recipeFamily.label) vector"
            stats = nil
            refreshStaleness()
            return
        }
        guard recipeFamily != .repeReaderLAT else {
            // Honest scoping: reader statistics come from the template-
            // rendered fit (train/held-out probe accuracy), not raw-stimulus
            // pair margins — scoring unrendered text would not be the
            // instrument the artifact records.
            status = "RepE reader accuracy comes from Build reader "
                + "(template-rendered fit), not live pair margins"
            stats = nil
            refreshStaleness()
            return
        }
        guard !isWorking else {
            rebuildQueued = true
            return
        }
        guard let host, let container = host.containerForExtraction,
            let modelID = host.loadedModelID
        else {
            status = "load a model first"
            return
        }
        guard positives.count >= 4, negatives.count >= 4 else {
            status = "need at least 4 stimuli per side for stats"
            return
        }

        isWorking = true
        activeTaskLabel = "Rebuilding stats"
        defer {
            isWorking = false
            activeTaskLabel = nil
        }

        repeat {
            rebuildQueued = false
            await rebuildOnce(container: container, modelID: modelID)
        } while rebuildQueued
    }

    private func rebuildOnce(container: ModelContainer, modelID: String) async {
        do {
            status = "computing activations…"
            let posActs = try await activations(for: positives, container: container, modelID: modelID)
            let negActs = try await activations(for: negatives, container: container, modelID: modelID)

            guard let layerCount = posActs.first?.count, layerCount > 0 else {
                status = "no activations recorded"
                return
            }
            let statsLayer = layerCount / 2
            let method = extractionMethod
            let posAtLayer = posActs.map { $0[statsLayer] }
            let negAtLayer = negActs.map { $0[statsLayer] }
            let direction = try SteeringVectorMath.direction(
                positive: posAtLayer, negative: negAtLayer, method: method)

            let normByLayer: [Float] = try (0 ..< layerCount).map { layer in
                let v = try SteeringVectorMath.direction(
                    positive: posActs.map { $0[layer] },
                    negative: negActs.map { $0[layer] },
                    method: method)
                return SteeringVectorMath.l2Norm(v)
            }

            var stability: Float?
            if let last = lastDirection, last.model == modelID {
                stability = try? SteeringVectorMath.cosineSimilarity(last.vector, direction)
            }
            lastDirection = (modelID, direction)

            let outliers = ConceptStats.outliers(
                direction: direction, positive: posAtLayer, negative: negAtLayer
            ).map { outlier in
                let text = outlier.isPositive
                    ? positives[outlier.index] : negatives[outlier.index]
                return OutlierDisplay(
                    id: outlier.id,
                    classLabel: outlier.isPositive ? "+" : "−",
                    textPrefix: String(text.prefix(48)),
                    margin: outlier.margin)
            }

            let allMargins = ConceptStats.outliers(
                direction: direction, positive: posAtLayer, negative: negAtLayer,
                count: posAtLayer.count + negAtLayer.count)
            let marginByStimulus = Dictionary(
                uniqueKeysWithValues: allMargins.map { ($0.id, $0.margin) })

            status = "computing control cosines…"
            let controls = try await controlCosines(
                against: direction, statsLayer: statsLayer,
                container: container, modelID: modelID)

            stats = Stats(
                positiveCount: positives.count,
                negativeCount: negatives.count,
                statsLayer: statsLayer,
                heldOut: ConceptStats.heldOutAccuracy(
                    positive: posAtLayer, negative: negAtLayer, method: method),
                splitHalf: ConceptStats.splitHalfCosine(
                    positive: posAtLayer, negative: negAtLayer, method: method),
                stability: stability,
                normByLayer: normByLayer,
                outliers: outliers,
                controlCosines: controls,
                marginByStimulus: marginByStimulus)
            statsFingerprint = currentFingerprint
            refreshStaleness()
            status = nil
        } catch {
            status = "rebuild failed: \(error)"
        }
    }

    /// Cosines against every other concept on disk (including the built-in
    /// negative-valence and arousal controls) — the live early warning that
    /// two concepts are collapsing into one direction.
    private func controlCosines(
        against direction: [Float], statsLayer: Int,
        container: ModelContainer, modelID: String
    ) async throws -> [ControlCosine] {
        var result: [ControlCosine] = []
        for name in existingConcepts where name != currentName {
            let directory = VectorCatalog.conceptsDirectory.appending(component: name)
            guard let set = try? StimulusSet(directory: directory) else { continue }
            let posActs = try await activations(
                for: set.positive, container: container, modelID: modelID)
            let negActs = try await activations(
                for: set.negative, container: container, modelID: modelID)
            guard posActs.first?.count ?? 0 > statsLayer else { continue }
            let vector = try SteeringVectorMath.meanDifference(
                positive: posActs.map { $0[statsLayer] },
                negative: negActs.map { $0[statsLayer] })
            if let cosine = try? SteeringVectorMath.cosineSimilarity(direction, vector) {
                result.append(ControlCosine(name: name, cosine: cosine))
            }
        }
        return result.sorted { abs($0.cosine) > abs($1.cosine) }
    }

    private var currentName: String {
        selectedExisting ?? Self.sanitizedName(conceptName)
    }

    private var vectorBuildName: String {
        vectorBuilderSelectedExisting ?? ""
    }

    /// The concept a BUILD extracts. The Concept Vector Builder's own picker
    /// is the target (its caption says so, and the button's enablement, the
    /// freshness advice, the reader fit, and the server build all read it);
    /// the dataset selection is only the fallback for flows that never touch
    /// that picker — a brand-new concept typed into the Dataset Builder, and
    /// the web client, which selects concepts through `selectedExisting`.
    private var buildTargetName: String {
        let name = vectorBuildName
        return name.isEmpty ? currentName : name
    }

    private func pairedRowsForBuildConcept(_ name: String) throws -> (positive: [String], negative: [String]) {
        if name == currentName {
            return (positives, negatives)
        }
        let directory = VectorCatalog.conceptsDirectory.appending(component: name)
        let set = try StimulusSet(directory: directory)
        return (set.positive, set.negative)
    }

    /// Activations are keyed by MODEL, READING POSITION and RENDERING: a
    /// pooled read and a last-token read are different measurements of the
    /// same stimulus, and so are a raw render and a templated one — reusing
    /// one for the other would silently mix two directions in one vector.
    static func activationCacheKey(
        modelID: String, options: ExtractionOptions, text: String
    ) -> String {
        "\(modelID)|\(options.readingPosition.label)"
            + "|\(options.resolvedExtractionRendering.label)|\(text)"
    }

    private func activations(
        for texts: [String], container: ModelContainer, modelID: String
    ) async throws -> [[[Float]]] {
        let options = extractionOptions
        func key(_ text: String) -> String {
            Self.activationCacheKey(modelID: modelID, options: options, text: text)
        }

        let missing = texts.filter { activationCache[key($0)] == nil }
        if !missing.isEmpty {
            let computed = try await ConceptExtractor.activations(
                container: container, texts: missing,
                position: options.readingPosition,
                rendering: options.resolvedExtractionRendering)
            for (text, acts) in zip(missing, computed.values) {
                activationCache[key(text)] = acts
            }
        }
        return texts.compactMap { activationCache[key($0)] }
    }

    // MARK: - Server-side builds (server workspace)

    /// Queue a vector extraction on the active server as a durable job. The
    /// server extracts from *its* checkout of the git-versioned recipe data
    /// (`prompts/concepts/…`, story corpus) with this panel's current
    /// options, on the server's loaded model — loading the selected server
    /// model first when needed. The finished artifact lands in the server's
    /// vector catalog; it is a per-substrate artifact, never merged locally.

    // MARK: J-lens token direction

    /// The lens resolved for the selected model. NOT a user choice: the
    /// published artifacts are one lens per model (the fitting corpus is a path
    /// segment, so more could exist later), so this is provenance the pane
    /// displays, not a picker. Nil means no lens is imported for this model yet.
    public var jlensLensID: String?
    /// Everything the pane shows about that lens, for the provenance rows.
    public var jlensLens: JLensRecord?
    /// Candidates for the typed word. The service never recommends one.
    public var jlensTokenOptions: JLensTokenOptions?
    public var jlensQuery: String = ""
    public var jlensIncludeCaseVariants = false
    /// The selected token. A direction is indexed by ONE exact id, and the id —
    /// never the word — is the durable identity.
    public var jlensSelectedTokenID: Int?
    public var jlensSelectedPiece: String?
    /// The researcher's own label. Combined with the token id on save.
    public var jlensLabel: String = ""
    /// OPTIONAL concept association, empty by default.
    ///
    /// Empty means the direction groups only with itself. Choosing a concept is
    /// a deliberate act with a real consequence: in every concept-grouped view
    /// the direction then sits beside that concept's stimulus-extracted vectors,
    /// as though they were alternative extractions of one thing. Sometimes that
    /// is exactly what a researcher wants (validating a lens token against the
    /// concept's held-out probe is a genuine test); it must never happen as a
    /// side effect of typing a name.
    public var jlensAssociatedConcept: String = ""

    /// The artifact name: the researcher's label with the token id appended.
    ///
    /// The id is ALWAYS appended, and that is the whole point. Two directions
    /// labelled "courage" could be token 23648 (" courage") and 236755 ("c") —
    /// the exact confusion the token picker exists to prevent. A bare label plus
    /// version numbers ("courage", "courage-2") records that there are two and
    /// nothing about which is which; the id makes the distinction visible in the
    /// artifact name itself, and makes collisions informative:
    /// same label + same token is a duplicate to reuse, same label + different
    /// token gets distinct names for free.
    public var jlensArtifactName: String? {
        guard let tokenID = jlensSelectedTokenID else { return nil }
        let trimmed = jlensLabel.trimmingCharacters(in: .whitespacesAndNewlines)
        let base = trimmed.isEmpty
            ? "jlens-token-\(Self.safePiece(jlensSelectedPiece))"
            : Self.safeLabel(trimmed)
        return "\(base)-id\(tokenID)"
    }

    /// True when an artifact with this exact name already exists for the
    /// workspace — which, because the id is in the name, means the SAME
    /// derivation, not a name clash to version around.
    public func jlensDuplicateExists(in records: [SubstrateVectorRecord]) -> Bool {
        guard let name = jlensArtifactName else { return false }
        // The derived artifact's `concept` IS its name (derive stamps both), and
        // the name carries the token id — so a match here is the same lens, the
        // same token, the same bytes, not a label clash.
        return records.contains { (record: SubstrateVectorRecord) in
            record.concept == name
        }
    }

    static func safeLabel(_ text: String) -> String {
        let kept = text.map { ch -> Character in
            ch.isLetter || ch.isNumber || ch == "-" || ch == "_" ? ch : "-"
        }
        let joined = String(kept).trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return joined.isEmpty ? "jlens-token" : String(joined.prefix(48))
    }

    static func safePiece(_ piece: String?) -> String {
        safeLabel(piece ?? "")
    }

    /// Resolve the lens for the selected model, so the pane can show what it is
    /// or say that none is imported.
    public func refreshJLensLens() async {
        guard let host, let client = host.cluster.client,
              let modelID = host.workspaceSelectedModelID, !modelID.isEmpty
        else { jlensLensID = nil; jlensLens = nil; return }
        guard let catalog = try? await client.jlensCatalog() else { return }
        let match = catalog.lenses.first { $0.fit?.modelID == modelID }
        jlensLensID = match?.lensID
        jlensLens = match
        jlensLensTier = catalog.supported
            .first { $0.modelID == modelID }?.tier
    }

    /// Bounded, deterministic candidates for the typed word.
    public func lookUpJLensTokens() async {
        guard let host, let client = host.cluster.client,
              let modelID = host.workspaceSelectedModelID, !modelID.isEmpty,
              !jlensQuery.isEmpty
        else { return }
        jlensSelectedTokenID = nil
        jlensSelectedPiece = nil
        do {
            jlensTokenOptions = try await client.jlensTokenOptions(
                modelID: modelID, text: jlensQuery,
                includeCaseVariants: jlensIncludeCaseVariants)
        } catch {
            status = "token lookup failed: \(error.localizedDescription)"
        }
    }

    /// The evidence tier of the resolved lens's model, from the server's own
    /// supported table — never inferred from the model id here, so one authority
    /// decides what counts as evidence.
    public var jlensLensTier: String?

    /// Import the published lens for the selected model (a server job).
    ///
    /// Acquire-then-import in one action from the builder: a researcher who
    /// wants a vector should not have to learn the lens lifecycle to get one.
    public func importJLensForSelectedModel() async {
        guard let host, let client = host.cluster.client,
              let modelID = host.workspaceSelectedModelID, !modelID.isEmpty
        else { status = "connect a server workspace and select a model"; return }
        isWorking = true
        activeTaskLabel = "Importing J-lens"
        defer { isWorking = false; activeTaskLabel = nil }
        do {
            // Acquire first: import is offline by construction and refuses when
            // the bytes are not cached, so doing both here is what makes this
            // one button rather than two.
            let acquireJob = try await client.jlensAcquire(modelID: modelID)
            _ = await host.followServerJobInActivity(
                jobID: acquireJob, client: client,
                title: "J-lens acquire: \(modelID)")
            let importJob = try await client.jlensImport(modelID: modelID)
            _ = await host.followServerJobInActivity(
                jobID: importJob, client: client,
                title: "J-lens import: \(modelID)")
            await refreshJLensLens()
        } catch {
            status = "lens import failed: \(error.localizedDescription)"
        }
    }

    public func selectJLensToken(_ candidate: JLensTokenCandidate) {
        jlensSelectedTokenID = candidate.tokenID
        jlensSelectedPiece = candidate.decoded ?? candidate.piece
    }

    public func buildVectorOnActiveServer() async {
        guard !isWorking else { return }
        // The reader recipe is its own flow (a fitted measurement instrument,
        // not a steering vector) with its own validation — delegate so a
        // repeReaderLAT build queues the server fit job.
        if recipeFamily == .repeReaderLAT {
            await buildReaderOnActiveServer()
            return
        }
        // A declaration the SERVER would also refuse never becomes a queued
        // job — answered first, because it does not depend on a connection.
        // The two engine-asymmetry refusals are deliberately NOT here: their
        // own repair text names this server as the fix.
        if recipeFamily.extractsFromStimuli,
            let refusal = serverExtractionDeclarationRefusal
        {
            reportBuildFailure(refusal, title: "Server Vector Build — refused")
            return
        }
        guard let host, case .server = host.cluster.activeWorkspace,
            let client = host.cluster.client
        else {
            status = "connect a server workspace first"
            return
        }
        let name = vectorBuildName
        // A J-lens direction names itself from the label + token id, so it has
        // no concept to select.
        if recipeFamily != .emotionGrandMean, recipeFamily != .jlensTokenDirection,
            name.isEmpty {
            status = "select a vector-build concept"
            return
        }

        isWorking = true
        activeTaskLabel = "Generating vector (server)"
        lastBuildError = nil
        defer {
            isWorking = false
            activeTaskLabel = nil
        }

        do {
            try await ensureSelectedServerModelLoaded(host: host, client: client)
            let jobID: String
            let title: String
            switch recipeFamily {
            case .jlensTokenDirection:
                // Derivation is its own server verb: it takes a lens id and a
                // token id, not a stimulus set, so it does not pass through the
                // extraction job at all.
                guard let lensID = jlensLensID,
                      let tokenID = jlensSelectedTokenID,
                      let selectedModel = host.workspaceSelectedModelID
                else {
                    status = "import a lens and select an exact token first"
                    return
                }
                let association = jlensAssociatedConcept
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                jobID = try await client.jlensDerive(
                    lensID: lensID, modelID: selectedModel, tokenID: tokenID,
                    piece: jlensSelectedPiece, name: jlensArtifactName,
                    concept: association.isEmpty ? nil : association)
                title = "J-lens derive: \(jlensArtifactName ?? "token \(tokenID)")"
            case .emotionGrandMean:
                let included = includedEmotionConcepts.isEmpty
                    ? nil : includedEmotionConcepts.sorted()
                let targets = grandMeanTargetConceptNames.isEmpty
                    ? nil : grandMeanTargetConceptNames.sorted()
                // The job reads the SERVER's story corpora — and its corpus
                // loader silently skips missing files, so an unsynced concept
                // would silently pool a smaller corpus, not fail. Push the
                // included corpora first; anything required that cannot be
                // pushed (no local stories) must already exist server-side.
                let missing = try await syncStoryCorporaToServer(
                    plan: Self.grandMeanServerSyncPlan(
                        included: included, targets: targets,
                        localStoryConcepts: localStoryConceptNames()),
                    client: client)
                if !missing.isEmpty {
                    reportBuildFailure(
                        "cannot queue grand-mean build: the server has "
                            + "no stories for \(missing.joined(separator: ", ")) "
                            + "(prompts/emotions/<concept>/stories.jsonl on the "
                            + "server's tree) and there are none locally to sync",
                        title: "Server Grand Mean Vector Build — failed")
                    return
                }
                // The grand-mean route pools from token 50 when the body says
                // nothing, so THAT is its legacy default — and the whole
                // declaration travels either way.
                let declaration = serverExtractionDeclaration(
                    legacyPooledDefault: 50)
                jobID = try await client.multiConceptExtract(
                    concepts: included, targets: targets,
                    poolFromToken: declaration.poolFromToken,
                    readingPosition: declaration.readingPosition,
                    extractionRendering: declaration.extractionRendering)
                title = "Server Grand Mean Vector Build"
            case .designatedReference:
                let reference = designatedReferenceConcept
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard !reference.isEmpty else {
                    reportBuildFailure(
                        "designated-reference build needs a reference stories "
                            + "concept — the reference corpus is part of the "
                            + "recipe",
                        title: "Server Vector Build — refused")
                    return
                }
                if let refusal = designatedReferenceRefusal {
                    reportBuildFailure(
                        refusal, title: "Server Vector Build — refused")
                    return
                }
                // The job reads the SERVER's story corpora for BOTH classes —
                // push what exists locally, and anything unpushable must
                // already be in the server's corpus listing, answered before
                // queuing rather than after the GPU warms up.
                let missing = try await syncStoryCorporaToServer(
                    plan: Self.grandMeanServerSyncPlan(
                        included: [name, reference], targets: nil,
                        localStoryConcepts: localStoryConceptNames()),
                    client: client)
                if !missing.isEmpty {
                    reportBuildFailure(
                        "cannot queue designated-reference build: the server "
                            + "has no stories for \(missing.joined(separator: ", ")) "
                            + "(prompts/emotions/<concept>/stories.jsonl on the "
                            + "server's tree) and there are none locally to sync",
                        title: "Server Vector Build — refused")
                    return
                }
                // Pooled-from-50 is this method's attach policy, so it is the
                // route's legacy reading of an absent position — and the
                // reference itself is verified via the response echo: a
                // server that cannot confirm the pin never gets to build.
                let declaration = serverExtractionDeclaration(
                    legacyPooledDefault: 50)
                jobID = try await client.conceptExtract(
                    concept: name,
                    method: ExtractionMethod.designatedReference.rawValue,
                    reference: reference,
                    poolFromToken: declaration.poolFromToken,
                    readingPosition: declaration.readingPosition,
                    extractionRendering: declaration.extractionRendering)
                title = "Server Vector Build: \(name) − '\(reference)'"
            case .caaMeanDifference, .pairedDifferencePCA:
                // The job reads the SERVER's checkout of
                // prompts/concepts/<name> — persist the drafted dataset
                // locally first (the git-versioned recipe, the same write the
                // local build does), then PREFLIGHT against the server's
                // concept catalog: push when the server has nothing to
                // clobber, skip the redundant write when the copies already
                // match, and REFUSE when the server holds DIFFERENT data —
                // never silently overwrite it, never queue a job against
                // data other than what this panel shows.
                let rows = try persistServerBuildDataset(name: name)
                let localContentHash = Self.stimulusContentHash(
                    positive: rows.positive, negative: rows.negative)
                var catalogKnown = false
                var record: ClusterClient.RemoteConceptRecord?
                if let listing = try? await client.remoteConcepts() {
                    catalogKnown = true
                    serverConcepts = listing
                    record = listing.first { $0.name == name }
                }
                let plan = WorkspaceScoping.serverConceptBuildSyncPlan(
                    concept: name,
                    catalogKnown: catalogKnown,
                    existsLocally: true,
                    existsOnServer: (record.map { $0.positiveCount + $0.negativeCount } ?? 0) > 0,
                    localContentHash: localContentHash,
                    serverContentHash: record?.contentHash)
                switch plan {
                case .refuse(let reason):
                    reportBuildFailure(
                        reason, title: "Server Vector Build — refused")
                    return
                case .push:
                    status = "syncing '\(name)' stimuli to the server…"
                    let saved = try await client.saveConcept(
                        name: name, positive: rows.positive, negative: rows.negative)
                    if let serverHash = saved.contentHash, let localContentHash,
                        serverHash != localContentHash
                    {
                        reportBuildFailure(
                            "after syncing, the server's '\(name)' content hash "
                                + "(\(serverHash.prefix(12))) still differs from "
                                + "local (\(localContentHash.prefix(12))) — "
                                + "refusing to queue extraction against different "
                                + "data",
                            title: "Server Vector Build — refused")
                        return
                    }
                case .skipPush:
                    status = "server's '\(name)' already matches the local "
                        + "dataset — extracting from its copy"
                }
                // The paired route reads the last token when the body says
                // nothing about the position — its legacy default.
                let declaration = serverExtractionDeclaration(
                    legacyPooledDefault: nil)
                jobID = try await client.conceptExtract(
                    concept: name, method: extractionMethod.rawValue,
                    poolFromToken: declaration.poolFromToken,
                    readingPosition: declaration.readingPosition,
                    extractionRendering: declaration.extractionRendering)
                title = "Server Vector Build: \(name)"
            case .repeReaderLAT:
                return  // delegated to buildReaderOnActiveServer above
            }
            status = "server extraction queued as job \(jobID)…"
            await host.cluster.refreshRemoteState()
            guard let job = await followServerJobInDisplay(
                jobID: jobID,
                client: client,
                host: host,
                title: title,
                label: "vector extraction"
            ) else {
                status = "server extraction is still running as job \(jobID) — "
                    + "open Compute to continue watching progress"
                return
            }
            await host.cluster.refreshRemoteState()
            switch job.status {
            case "succeeded":
                await host.catalog.refreshRemoteVectors()
                let runDirectory = job.result.flatMap { result -> String? in
                    if case .string(let path)? = result["runDirectory"] { return path }
                    return nil
                }
                status = "server extraction finished"
                    + (runDirectory.map { " → \($0)" } ?? "")
                    + " — new vector artifact is in the server catalog"
            case "cancelled":
                status = "server extraction was cancelled (job \(jobID))"
            default:
                // The followed job's live log already carries the error tail;
                // surface it in the banner without a duplicate log entry.
                reportBuildFailure(
                    "server extraction failed: "
                        + (job.error ?? job.logTail.last ?? "job \(jobID) \(job.status)"),
                    title: title, logToActivity: false)
            }
        } catch {
            reportBuildFailure(
                "server extraction failed: \(Self.serverFailureText(error))",
                title: "Server Vector Build — failed")
        }
    }

    /// A server failure in the SERVER's words.
    ///
    /// A refused declaration comes back as `{"detail": "<the engine's refusal
    /// — repair: …>"}`; showing that JSON wrapper in a builder notice buries
    /// the repair inside quoting and escapes. Every other error prints as it
    /// always did.
    static func serverFailureText(_ error: any Error) -> String {
        guard case ClusterClient.ClientError.badResponse(let code, let text) = error
        else { return "\(error)" }
        guard let data = text.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data)
                as? [String: Any],
            let detail = object["detail"] as? String
        else { return "server returned \(code): \(text)" }
        return detail
    }

    /// Queue reading-probe training on the active server (durable job over
    /// the server's checkout of `prompts/probes/<concept>/items.jsonl`) and
    /// follow it in the shared live-log/Activity pane — same treatment as
    /// the server vector build, so training no longer queues silently.
    public func trainProbeOnActiveServer() async {
        guard !isWorking else { return }
        guard let host, case .server = host.cluster.activeWorkspace,
            let client = host.cluster.client
        else {
            status = "connect a server workspace first"
            return
        }
        let name = currentName
        guard !name.isEmpty else {
            status = "select or name a concept before training a probe"
            return
        }

        isWorking = true
        activeTaskLabel = "Training probe (server)"
        lastBuildError = nil
        defer {
            isWorking = false
            activeTaskLabel = nil
        }

        let title = "Server Probe Training: \(name)"
        do {
            try await ensureSelectedServerModelLoaded(host: host, client: client)
            // Same persistence hole as the vector build: the job reads the
            // SERVER's prompts/probes/<name>/items.jsonl, so push the
            // builder's items first (local file: the git-versioned recipe;
            // server: the probe-items API). Empty builder items push
            // nothing — the server's own copy, if any, stays untouched.
            if !probeExamples.isEmpty {
                try writeProbeExamples(probeExamples, for: name)
                status = "syncing '\(name)' probe items to the server…"
                try await client.saveProbeItems(concept: name, rows: probeExamples)
            }
            let jobID = try await client.probeTrain(concept: name)
            status = "server probe training queued as job \(jobID) — uses the "
                + "server's probe items for '\(name)'"
            await host.cluster.refreshRemoteState()
            guard let job = await followServerJobInDisplay(
                jobID: jobID,
                client: client,
                host: host,
                title: title,
                label: "probe training"
            ) else {
                status = "server probe training is still running as job \(jobID) — "
                    + "open Compute to continue watching progress"
                return
            }
            await host.cluster.refreshRemoteState()
            switch job.status {
            case "succeeded":
                let runDirectory = job.result.flatMap { result -> String? in
                    if case .string(let path)? = result["runDirectory"] { return path }
                    return nil
                }
                let quality = job.result.flatMap { result -> String? in
                    guard case .number(let layer)? = result["layer"],
                        case .number(let accuracy)? = result["accuracy"]
                    else { return nil }
                    return String(
                        format: "best layer %d, held-out acc %.0f%%",
                        Int(layer), accuracy * 100)
                }
                status = "server probe training finished"
                    + (runDirectory.map { " → \($0)" } ?? "")
                    + (quality.map { " (\($0))" } ?? "")
                    + " — the probe artifact stays in the SERVER's runs/ tree"
            case "cancelled":
                status = "server probe training was cancelled (job \(jobID))"
            default:
                // The followed job's live log already carries the error tail;
                // surface it in the banner without a duplicate log entry.
                reportBuildFailure(
                    "server probe training failed: "
                        + (job.error ?? job.logTail.last ?? "job \(jobID) \(job.status)"),
                    title: title, logToActivity: false)
            }
        } catch {
            reportBuildFailure(
                "server probe training failed: \(error)",
                title: "Server Probe Training — failed")
        }
    }

    /// The server-build twin of the local save's dataset write: a queued
    /// extract job reads the server's checkout of `prompts/concepts/<name>`,
    /// so the drafted dataset must exist as files BEFORE the job runs. When
    /// the build target is the editor's current concept, the unsaved drafts
    /// are written locally first (concept datasets are git-versioned recipes
    /// — the exact write `saveConceptAndExtract` does); a picked saved
    /// concept already has its files on disk. Returns the rows the caller
    /// pushes to the server via the concept-save API.
    private func persistServerBuildDataset(
        name: String
    ) throws -> (positive: [String], negative: [String]) {
        let rows = try pairedRowsForBuildConcept(name)
        guard !rows.positive.isEmpty, !rows.negative.isEmpty else {
            throw ChatServiceError(
                reason: "concept '\(name)' has no paired stimuli to extract "
                    + "from — draft or import stimuli first")
        }
        if name == currentName {
            let directory = VectorCatalog.conceptsDirectory.appending(component: name)
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            try jsonl(rows.positive).write(
                to: directory.appending(component: "positive.jsonl"),
                atomically: true, encoding: .utf8)
            try jsonl(rows.negative).write(
                to: directory.appending(component: "negative.jsonl"),
                atomically: true, encoding: .utf8)
            refreshConceptList()
        }
        return rows
    }

    /// Pure seam for the server grand-mean build (unit-tested without a
    /// server): which local story corpora to PUSH through the stories API
    /// before queuing — the included set, or every locally present corpus
    /// when `included == nil` (nil pools the server's whole corpus) — and
    /// which required concepts cannot be pushed (no local stories) and must
    /// therefore already exist on the server (pre-flighted against
    /// `GET /api/multiconcept/concepts`, so a gap is a clear status error
    /// before queuing, never a job-side silently-smaller corpus).
    nonisolated static func grandMeanServerSyncPlan(
        included: [String]?, targets: [String]?, localStoryConcepts: [String]
    ) -> (push: [String], requireOnServer: [String]) {
        let local = Set(localStoryConcepts)
        let pooled = included.map { Set($0) } ?? local
        let required = pooled.union(targets ?? [])
        let push = required.intersection(local)
        return (push.sorted(), required.subtracting(push).sorted())
    }

    /// Execute a story-corpus sync plan against the active server: push the
    /// local corpora the build needs and preflight the rest against the
    /// server's corpus listing. Returns the concepts the server cannot
    /// provide (nothing local to push, nothing in its listing) — non-empty
    /// means refuse before queuing, because the job-side corpus loaders
    /// answer a missing file only after the GPU is already warm.
    private func syncStoryCorporaToServer(
        plan: (push: [String], requireOnServer: [String]),
        client: ClusterClient
    ) async throws -> [String] {
        var mustExistOnServer = plan.requireOnServer
        for concept in plan.push {
            let rows = (try? loadEmotionRows(for: concept)) ?? []
            guard !rows.isEmpty else {
                // Nothing to push — never overwrite server stories with an
                // empty file; require the server's copy.
                mustExistOnServer.append(concept)
                continue
            }
            status = "syncing '\(concept)' stories to the server…"
            try await client.saveStories(concept: concept, rows: rows)
        }
        guard !mustExistOnServer.isEmpty else { return [] }
        let listing = Set(try await client.storyConcepts())
        return mustExistOnServer.filter { !listing.contains($0) }.sorted()
    }

    /// One concept's stories corpus from disk — the class loader every
    /// designated-reference path here shares (the hash is the raw file
    /// bytes', matching what attach pins). nil when the file is missing or
    /// unreadable.
    private func storyCorpus(for concept: String)
        -> (rows: [StimulusSet.MultiConceptStimulus], hash: String)?
    {
        try? StimulusSet.loadMultiConceptTexts(
            url: VectorCatalog.emotionsDirectory.appending(
                components: concept, "stories.jsonl"))
    }

    private func storyCorpusRowCount(for concept: String) -> Int {
        storyCorpus(for: concept)?.rows.count ?? 0
    }

    private func storyCorpusHash(for concept: String) -> String? {
        storyCorpus(for: concept)?.hash
    }

    /// Both designated-reference classes' texts, for the pass-count estimate.
    private func designatedReferenceClassTexts() -> [String] {
        let target = buildTargetName
        let reference = designatedReferenceConcept
            .trimmingCharacters(in: .whitespacesAndNewlines)
        var texts: [String] = []
        if !target.isEmpty, let corpus = storyCorpus(for: target) {
            texts += corpus.rows.map(\.text)
        }
        if !reference.isEmpty, reference != target,
            let corpus = storyCorpus(for: reference)
        {
            texts += corpus.rows.map(\.text)
        }
        return texts
    }

    /// Concepts with a local `prompts/emotions/<name>/stories.jsonl`.
    private func localStoryConceptNames() -> [String] {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: VectorCatalog.emotionsDirectory,
                includingPropertiesForKeys: [.isDirectoryKey])
        else { return [] }
        return entries
            .filter {
                fm.fileExists(
                    atPath: $0.appending(component: "stories.jsonl").path)
            }
            .map(\.lastPathComponent)
            .sorted()
    }

    /// Local counterpart to `ensureSelectedServerModelLoaded`: the builder's
    /// model selector DRIVES local extraction. Where a server build loads the
    /// selected model on the server before queueing, a local build unloads
    /// whatever the Playground has resident and loads the selected model
    /// through the SAME path the Playground's Load button uses
    /// (`ChatService.ensureSelectedLocalModelLoaded` → `loadModel()`), rather
    /// than advising in a caption and extracting on the wrong model.
    ///
    /// No-op in a server workspace (that side has its own ensure) and when the
    /// selected model is already loaded. Throws — never proceeds — when the
    /// load fails or when a chat generation / another load is in flight.
    /// Callers must read `containerForExtraction` and `loadedModelID` AFTER
    /// awaiting this: a load replaces the container.
    private func ensureSelectedLocalModelLoaded(host: ChatService) async throws {
        guard case .local = host.cluster.activeWorkspace else { return }
        if host.loadedModelID != host.selectedModelID {
            status = "loading \(host.selectedModelID) for this build…"
        }
        try await host.ensureSelectedLocalModelLoaded()
    }

    /// Server extraction/probe jobs run on the server's *loaded* model. When
    /// the builder's selected server model differs (or nothing is loaded),
    /// load it first so the artifact is stamped with the model the user
    /// actually picked.
    private func ensureSelectedServerModelLoaded(
        host: ChatService, client: ClusterClient
    ) async throws {
        guard let model = host.selectedRemoteModelID, !model.isEmpty else {
            throw ChatServiceError(reason: "select a server model first")
        }
        if host.cluster.remoteState?.loadedModel != model {
            status = "loading \(model) on the server…"
            try await client.loadModel(model)
            await host.cluster.refreshRemoteState()
        }
    }

    // MARK: - Server concept catalog (Concept Lab workspace scoping)
    //
    // Under a server compute target the SERVER's `prompts/concepts/` tree is
    // what extraction jobs actually read — on an unpaired cluster that is a
    // DIFFERENT tree than the one this panel edits. The catalog below is the
    // server's own listing (read-only browse), the drift verdicts compare it
    // against the local workspace, and upload/fetch are the explicit sync
    // affordances. Editing always stays local.

    /// The ACTIVE server's concept datasets (`GET /api/concepts`), listed
    /// read-only next to the local concept index — never merged into
    /// `existingConcepts`.
    public private(set) var serverConcepts: [ClusterClient.RemoteConceptRecord] = []

    /// Set when a fetch would overwrite non-empty local stimulus files; the
    /// view confirms and re-calls `fetchConceptFromServer(_:overwrite: true)`.
    public var pendingServerFetchConcept: String?

    /// Re-list the active server's concept catalog. Quiet on failure (an
    /// older server or a dropped connection must not spam the status line) —
    /// the rows simply stay empty and drift stays `unknown`.
    public func refreshServerConcepts() async {
        guard let host, case .server = host.cluster.activeWorkspace,
            let client = host.cluster.client
        else {
            serverConcepts = []
            return
        }
        serverConcepts = (try? await client.remoteConcepts()) ?? []
    }

    /// Cross-engine CONTENT hash of a paired stimulus set: SHA-256 over the
    /// compact, key-sorted JSON `{"negative":[…],"positive":[…]}` (UTF-8,
    /// slashes unescaped). Byte-for-byte the server's
    /// `authoring.stimulus_content_hash` — golden-fixture tested on both
    /// engines. Raw file hashes (`StimulusSet.hash`) differ across engines
    /// for the same texts (each engine's JSONL formatting differs), so drift
    /// comparison MUST use this hash; manifest pins still use the raw hash
    /// of the executing substrate's files.
    public nonisolated static func stimulusContentHash(
        positive: [String], negative: [String]
    ) -> String? {
        guard !positive.isEmpty || !negative.isEmpty else { return nil }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard let data = try? encoder.encode(["negative": negative, "positive": positive])
        else { return nil }
        var digest = SHA256()
        digest.update(data: data)
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Local paired stimuli for a concept: whether any pair file has rows,
    /// and the content hash when it does. Tolerant of one-sided/legacy
    /// datasets (StimulusSet itself requires both sides).
    func localConceptStimuli(_ name: String) -> (exists: Bool, contentHash: String?) {
        let concept = Self.sanitizedName(name)
        guard !concept.isEmpty else { return (false, nil) }
        let rows: (positive: [String], negative: [String])
        if concept == currentName {
            rows = (positives, negatives)
        } else {
            let directory = VectorCatalog.conceptsDirectory.appending(component: concept)
            rows = (
                (try? StimulusSet.loadTexts(
                    url: directory.appending(component: "positive.jsonl")).texts) ?? [],
                (try? StimulusSet.loadTexts(
                    url: directory.appending(component: "negative.jsonl")).texts) ?? []
            )
        }
        guard !rows.positive.isEmpty || !rows.negative.isEmpty else { return (false, nil) }
        return (
            true,
            Self.stimulusContentHash(positive: rows.positive, negative: rows.negative)
        )
    }

    /// The server's catalog row for a concept, if it has one with rows.
    public func serverConceptRecord(_ name: String) -> ClusterClient.RemoteConceptRecord? {
        serverConcepts.first { $0.name == Self.sanitizedName(name) }
    }

    /// Drift verdict for one concept against the active server's catalog
    /// (`.unknown` until `refreshServerConcepts()` has run, or on an older
    /// server without content hashes).
    public func conceptDrift(for name: String) -> WorkspaceScoping.ConceptDrift {
        let local = localConceptStimuli(name)
        let record = serverConceptRecord(name)
        let serverHasRows = (record.map { $0.positiveCount + $0.negativeCount } ?? 0) > 0
        return WorkspaceScoping.conceptDrift(
            existsLocally: local.exists,
            existsOnServer: serverHasRows,
            localContentHash: local.contentHash,
            serverContentHash: record?.contentHash)
    }

    /// Explicit sync, local → server: overwrite the server's
    /// `prompts/concepts/<name>/{positive,negative}.jsonl` with the local
    /// workspace's texts (the same save route the server build uses), then
    /// verify the server recomputed the same content hash.
    public func uploadConceptToServer(_ name: String) async {
        guard !isWorking else { return }
        guard let host, case .server = host.cluster.activeWorkspace,
            let client = host.cluster.client
        else {
            status = "connect a server workspace first"
            return
        }
        let concept = Self.sanitizedName(name)
        guard !concept.isEmpty else { return }
        do {
            let rows = try pairedRowsForBuildConcept(concept)
            guard !rows.positive.isEmpty, !rows.negative.isEmpty else {
                status = "'\(concept)' has no complete paired dataset to upload "
                    + "(both positive and negative rows are required)"
                return
            }
            let localHash = Self.stimulusContentHash(
                positive: rows.positive, negative: rows.negative)
            status = "uploading '\(concept)' stimuli to \(host.cluster.substrateLabel)…"
            let saved = try await client.saveConcept(
                name: concept, positive: rows.positive, negative: rows.negative)
            await refreshServerConcepts()
            if let serverHash = saved.contentHash, let localHash, serverHash != localHash {
                reportBuildFailure(
                    "uploaded '\(concept)' but the server recomputed a different "
                        + "content hash (local \(localHash.prefix(12)) vs server "
                        + "\(serverHash.prefix(12))) — the copies are NOT in sync; "
                        + "do not queue server extractions for this concept",
                    title: "Concept Upload — hash mismatch")
            } else {
                status = "uploaded '\(concept)' "
                    + "(\(saved.positiveCount)+/\(saved.negativeCount)-) to the "
                    + "server's tree — copies are in sync"
            }
        } catch {
            reportBuildFailure(
                "upload of '\(concept)' failed: \(error)",
                title: "Concept Upload — failed")
        }
    }

    /// Explicit sync, server → local: download the server's copy into the
    /// local workspace. Refuses to overwrite existing local stimuli without
    /// confirmation (`pendingServerFetchConcept` drives the view's dialog).
    public func fetchConceptFromServer(_ name: String, overwrite: Bool = false) async {
        guard !isWorking else { return }
        guard let host, case .server = host.cluster.activeWorkspace,
            let client = host.cluster.client
        else {
            status = "connect a server workspace first"
            return
        }
        let concept = Self.sanitizedName(name)
        guard !concept.isEmpty else { return }
        if !overwrite, localConceptStimuli(concept).exists {
            pendingServerFetchConcept = concept
            status = "'\(concept)' already has local stimulus files — confirm "
                + "overwrite to replace them with the server's copy"
            return
        }
        do {
            status = "fetching '\(concept)' from \(host.cluster.substrateLabel)…"
            let contents = try await client.remoteConceptContents(name: concept)
            guard !contents.positive.isEmpty || !contents.negative.isEmpty else {
                status = "server's '\(concept)' has no paired stimuli to fetch"
                return
            }
            let directory = VectorCatalog.conceptsDirectory.appending(component: concept)
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            try jsonl(contents.positive).write(
                to: directory.appending(component: "positive.jsonl"),
                atomically: true, encoding: .utf8)
            try jsonl(contents.negative).write(
                to: directory.appending(component: "negative.jsonl"),
                atomically: true, encoding: .utf8)
            refreshConceptList()
            if selectedExisting == concept {
                loadSelectedExisting()
            }
            let localHash = Self.stimulusContentHash(
                positive: contents.positive, negative: contents.negative)
            let verified: String
            if let serverHash = contents.contentHash, let localHash {
                verified = serverHash == localHash
                    ? " — content hash verified" : " — WARNING: content hash mismatch"
            } else {
                verified = ""
            }
            status = "fetched '\(concept)' "
                + "(\(contents.positive.count)+/\(contents.negative.count)-) from "
                + "the server into the local workspace\(verified)"
        } catch {
            reportBuildFailure(
                "fetch of '\(concept)' failed: \(error)",
                title: "Concept Fetch — failed")
        }
    }

    // MARK: - RepE reader (faithful reader recipe; REPE-IMPLEMENTATION-BRIEF §8)
    //
    // Mental model: concept data → reader artifact → optional steering
    // variant — never "positive/negative text → vector". The reader is a
    // fitted measurement instrument; the steering vector is an explicit,
    // provenance-stamped conversion.

    /// Registry templates from `prompts/templates/` (shared data files — the
    /// same JSON the server reads; never duplicated in code).
    public private(set) var readerTemplates: [RepEReader.TaskTemplate] = []
    public var selectedReaderTemplateID: String?
    public var useCustomReaderTemplate = false
    public var customReaderTemplateText = ""
    /// The LAST k pairs of the working set are written with split "test" and
    /// score the fitted probe they did not train.
    public var readerHeldOutPairCount = 0
    /// The seed for the unsupervised T+/T− orientation draw, stamped into
    /// every artifact of an `unsupervisedTemplatePair` fit. Declarable because
    /// the reference implementation's shuffle is unseeded and therefore
    /// irreproducible; the default is the engine's own.
    public var readerOrientationSeed: UInt64 = RepEReader.defaultOrientationSeed

    public struct ReaderLayerScore: Identifiable, Sendable {
        public let layer: Int
        public let trainAccuracy: Float
        public let heldOutAccuracy: Float?
        /// PC1's share of the DIFFERENCE CLOUD's variance. nil when the cloud
        /// carries none to apportion (every difference identical) — absent,
        /// never 0, which would read as "PC1 explains nothing".
        public let explainedVariance: Float?
        /// How this layer's sign was fixed, and the held-out agreement behind
        /// it when the held-out split decided.
        public let signConvention: RepEReader.SignConvention
        public let signHeldOutAccuracy: Float?
        /// Why the held-out sign rule stood down at this layer, when it did —
        /// the artifact's own `signFallbackReason`, verbatim. Present exactly
        /// when `signConvention == .trainMajority` on a fit that tried.
        public let signFallbackReason: String?
        /// WHICH cloud the explained variance describes: `"differenceCloud"`,
        /// `"degenerateDifferenceCloud"`, or a schema-1 artifact's legacy
        /// `"alternatedRows"`. Displayed beside the number, never silently
        /// relabelled — the two bases are not comparable.
        public let explainedVarianceBasis: String
        /// True for the layer the fit RECOMMENDS (argmax held-out accuracy).
        /// A recommendation: the layer a study reads stays declarable.
        public let isRecommendedLayer: Bool
        public var id: Int { layer }

        /// True when this row's variance number is a pre-2026-08-27 artifact's
        /// — measured over the ± alternated copies PCA is fitted on, where the
        /// cloud is centred near zero by construction and the ratio flatters.
        public var explainedVarianceIsLegacyBasis: Bool {
            explainedVarianceBasis == "alternatedRows"
        }

        /// The variance cell's basis, in the words a reader needs beside it.
        public var explainedVarianceBasisLabel: String {
            switch explainedVarianceBasis {
            case "differenceCloud": "of the difference cloud"
            case "degenerateDifferenceCloud":
                "difference cloud is degenerate — every difference identical, "
                    + "so there is no variance to apportion"
            case "alternatedRows":
                "legacy schema-1 basis: measured over the ± ALTERNATED rows, "
                    + "not the differences — systematically flattering, and not "
                    + "comparable with a schema-2 number"
            default: explainedVarianceBasis
            }
        }

        /// The explained-variance cell, formatted. "—" when the difference
        /// cloud carries no variance to apportion: PC1's share of nothing is
        /// undefined, and "0.00" would read as "PC1 explains nothing", which
        /// is the opposite of what an all-identical cloud means.
        public var differenceCloudVarianceLabel: String {
            guard let explainedVariance else { return "—" }
            return String(format: "%.2f", explainedVariance)
        }

        public init(
            layer: Int, trainAccuracy: Float, heldOutAccuracy: Float?,
            explainedVariance: Float?,
            signConvention: RepEReader.SignConvention = .trainMajority,
            signHeldOutAccuracy: Float? = nil,
            signFallbackReason: String? = nil,
            explainedVarianceBasis: String = "differenceCloud",
            isRecommendedLayer: Bool = false
        ) {
            self.layer = layer
            self.trainAccuracy = trainAccuracy
            self.heldOutAccuracy = heldOutAccuracy
            self.explainedVariance = explainedVariance
            self.signConvention = signConvention
            self.signHeldOutAccuracy = signHeldOutAccuracy
            self.signFallbackReason = signFallbackReason
            self.explainedVarianceBasis = explainedVarianceBasis
            self.isRecommendedLayer = isRecommendedLayer
        }

        /// Projection of one fitted artifact — the ONE place a reader artifact
        /// becomes a display row, so the grid, the artifact list, and a legacy
        /// schema-1 artifact all read the same fields the same way.
        public init(artifact: RepEReader.Artifact) {
            self.init(
                layer: artifact.layer,
                trainAccuracy: artifact.trainAccuracy,
                heldOutAccuracy: artifact.heldOutAccuracy,
                explainedVariance: artifact.differenceCloudExplainedVariance,
                signConvention: artifact.signConvention,
                signHeldOutAccuracy: artifact.signHeldOutAccuracy,
                signFallbackReason: artifact.signFallbackReason,
                explainedVarianceBasis: artifact.explainedVarianceBasis,
                isRecommendedLayer: artifact.recommendedLayer == artifact.layer)
        }
    }

    /// The stamps that describe a fitted SET rather than one of its layers —
    /// the contrast the differences express, the orientation seed behind them,
    /// the recommended layer, and how the scaffolds reached the model.
    ///
    /// Read off the artifacts themselves. Nothing here is re-derived in the
    /// pane: two places computing "best layer" is how a pane and an artifact
    /// come to disagree, and the recommendation is the artifact's word.
    public struct ReaderFitStamps: Sendable, Equatable {
        public let contrastMode: RepEReader.ContrastMode
        public let orientationSeed: UInt64?
        public let recommendedLayer: Int?
        public let recommendedLayerAccuracy: Float?
        public let layerRecommendationBasis: String?
        /// The artifact's own rendering stamp — "raw" when absent, which is
        /// what every reader fitted before the option existed carries.
        public let renderingLabel: String
        public let renderingConvention: String
        /// True when the set carries a schema-1 explained-variance basis.
        public let hasLegacyVarianceBasis: Bool

        /// The recommendation line, in the ARTIFACT's own words — never
        /// "selected", because nothing selects a layer at use time.
        public var recommendationLine: String? {
            guard let recommendedLayer, let recommendedLayerAccuracy else { return nil }
            let basis = layerRecommendationBasis == "heldOutAccuracy"
                ? "held-out" : "train"
            return String(
                format: "recommended layer %d (%@ %.0f%%). ",
                recommendedLayer, basis, recommendedLayerAccuracy * 100)
                + RepEReader.Artifact.layerRecommendationNote
        }

        /// The orientation-draw line, or nil under `supervisedContent` (whose
        /// ± alternation needs no RNG and stamps no seed).
        public var orientationSeedLine: String? {
            guard let orientationSeed else { return nil }
            return "orientation seed \(orientationSeed) — the T+/T− draw is "
                + "SEEDED and stamped, so this fit is reproducible from the "
                + "artifact alone (the reference implementation's shuffle is not)"
        }
    }

    // MARK: Reader fit-score grid (formatting, not layout — the pane is thin)

    /// The fit-score grid's columns, in the order ``readerLayerCells``
    /// returns. "★" on a layer marks the fit's RECOMMENDED layer (argmax
    /// held-out accuracy) — a recommendation, not a selection: which layer a
    /// study reads is declared in its manifest.
    public nonisolated static let readerLayerColumns = [
        "layer", "train", "held-out", "PC1 var", "basis", "sign from",
    ]

    /// One reader-layer row's cells, as plain strings.
    ///
    /// An absent explained variance prints "—", not "0.00": PC1's share of a
    /// difference cloud with no variance is undefined, and 0 would read as
    /// "PC1 explains nothing" — the opposite of what an all-identical cloud
    /// means. The sign cell says whether the HELD-OUT split fixed this layer's
    /// direction (the paper's step 4) or the train labels did.
    ///
    /// The BASIS cell is not decoration: a schema-1 artifact's number was
    /// measured over the ± ALTERNATED rows and is systematically flattering,
    /// so it is never printed beside a schema-2 number without saying which is
    /// which.
    public nonisolated static func readerLayerCells(
        _ score: ReaderLayerScore
    ) -> [String] {
        let layer = score.isRecommendedLayer
            ? "\(score.layer) ★" : "\(score.layer)"
        let train = String(format: "%.0f%%", score.trainAccuracy * 100)
        let held =
            score.heldOutAccuracy.map { String(format: "%.0f%%", $0 * 100) } ?? "—"
        let sign: String
        switch score.signConvention {
        case .heldOutPairAgreement:
            sign = score.signHeldOutAccuracy
                .map { String(format: "held-out %.0f%%", $0 * 100) } ?? "held-out"
        case .trainMajority:
            sign = "train"
        }
        return [
            layer, train, held, score.differenceCloudVarianceLabel,
            readerVarianceBasisCell(score.explainedVarianceBasis), sign,
        ]
    }

    /// The basis column's short form. An unrecognized basis passes through
    /// verbatim rather than being coerced into one of the known three — a
    /// future basis must read as unfamiliar, not as one of these.
    public nonisolated static func readerVarianceBasisCell(_ basis: String) -> String {
        switch basis {
        case "differenceCloud": "diffs"
        case "degenerateDifferenceCloud": "degenerate"
        case "alternatedRows": "alternated (legacy)"
        default: basis
        }
    }

    /// Distinct sign-fallback reasons, in first-seen layer order — the same
    /// reason repeated across 30 layers is one line, not thirty.
    public nonisolated static func readerSignFallbackReasons(
        _ scores: [ReaderLayerScore]
    ) -> [String] {
        var seen: Set<String> = []
        return scores.compactMap { score -> String? in
            guard let reason = score.signFallbackReason,
                seen.insert(reason).inserted
            else { return nil }
            return reason
        }
    }

    /// "id — T+/T− pair" for a template-pair template, the bare id otherwise.
    /// The marker is on the MENU entry so a template that will refuse against
    /// the authored rows is recognizable before it is chosen — the picker
    /// shows every template rather than filtering the mismatch out of sight.
    public nonisolated static func readerTemplateMenuLabel(
        _ template: RepEReader.TaskTemplate
    ) -> String {
        template.isTemplatePair ? "\(template.id) — T+/T− pair" : template.id
    }

    /// Set-level stamps from a freshly fitted (or freshly read) artifact set.
    public nonisolated static func readerFitStamps(
        from artifacts: [RepEReader.Artifact]
    ) -> ReaderFitStamps? {
        guard let first = artifacts.first else { return nil }
        return ReaderFitStamps(
            contrastMode: first.contrastMode,
            orientationSeed: first.orientationSeed,
            recommendedLayer: first.recommendedLayer,
            recommendedLayerAccuracy: first.recommendedLayerAccuracy,
            layerRecommendationBasis: first.layerRecommendationBasis,
            renderingLabel: renderingLabel(first.extractionRendering),
            renderingConvention: first.renderingConvention,
            hasLegacyVarianceBasis: artifacts.contains {
                $0.explainedVarianceBasis == "alternatedRows"
            })
    }

    /// One labelled line of a reader artifact's provenance. `isCaution` marks
    /// a line a reader must not skim past — a stood-down sign rule, a legacy
    /// variance basis, a divergence stamp.
    public struct ReaderDetailRow: Sendable, Equatable, Identifiable {
        public let label: String
        public let value: String
        public let isCaution: Bool
        public var id: String { label }

        public init(label: String, value: String, isCaution: Bool = false) {
            self.label = label
            self.value = value
            self.isCaution = isCaution
        }
    }

    /// Everything a fitted reader artifact stamps, as display lines — schema 2
    /// and schema 1 alike. A schema-1 artifact is NOT dressed up as a schema-2
    /// one: its explained variance says which cloud it measured, its absent
    /// contrast/sign stamps show as the legacy defaults they decode to, and the
    /// rows say so in words.
    public nonisolated static func readerArtifactDetails(
        _ artifact: RepEReader.Artifact
    ) -> [ReaderDetailRow] {
        var rows: [ReaderDetailRow] = [
            ReaderDetailRow(
                label: "contrast mode", value: artifact.contrastMode.label),
            ReaderDetailRow(
                label: "sign convention",
                value: artifact.signConvention.label
                    + (artifact.signHeldOutAccuracy.map {
                        String(format: " · held-out agreement %.0f%%", $0 * 100)
                    } ?? ""),
                isCaution: artifact.signConvention == .trainMajority),
        ]
        if let reason = artifact.signFallbackReason {
            rows.append(
                ReaderDetailRow(
                    label: "sign fallback", value: reason, isCaution: true))
        }
        let score = ReaderLayerScore(artifact: artifact)
        rows.append(
            ReaderDetailRow(
                label: "PC1 explained variance",
                value: score.differenceCloudVarianceLabel
                    + " (" + score.explainedVarianceBasisLabel + ")",
                isCaution: score.explainedVarianceIsLegacyBasis
                    || artifact.differenceCloudExplainedVariance == nil))
        if let layer = artifact.recommendedLayer,
            let accuracy = artifact.recommendedLayerAccuracy
        {
            let basis = artifact.layerRecommendationBasis == "heldOutAccuracy"
                ? "held-out" : "train"
            rows.append(
                ReaderDetailRow(
                    label: "recommended layer",
                    value: String(format: "%d (%@ %.0f%%). ", layer, basis, accuracy * 100)
                        + RepEReader.Artifact.layerRecommendationNote))
        }
        if let seed = artifact.orientationSeed {
            rows.append(
                ReaderDetailRow(label: "orientation seed", value: String(seed)))
        }
        rows.append(
            ReaderDetailRow(
                label: "extraction rendering",
                value: renderingLabel(artifact.extractionRendering)
                    + (artifact.extractionRendering == nil
                        ? " (absent stamp = raw, the pre-declaration default)" : "")))
        rows.append(
            ReaderDetailRow(
                label: "template",
                value: "\(artifact.templateID) · \(artifact.templateHash.prefix(12))…"))
        if let divergence = artifact.template.divergence {
            rows.append(
                ReaderDetailRow(
                    label: "template divergence", value: divergence, isCaution: true))
        }
        rows.append(
            ReaderDetailRow(
                label: "dataset", value: "\(artifact.datasetHash.prefix(12))… · "
                    + "\(artifact.trainPairCount) train / \(artifact.heldOutPairCount) held out"))
        rows.append(
            ReaderDetailRow(
                label: "binding",
                value: "\(artifact.modelID) · substrate \(artifact.substrate)"))
        return rows
    }

    /// Per-layer train/held-out accuracy from the newest local reader fit.
    public private(set) var readerLayerScores: [ReaderLayerScore] = []
    /// Set-level stamps of the newest local reader fit.
    public private(set) var readerFitStamps: ReaderFitStamps?
    public private(set) var lastReaderRunDirectory: URL?
    /// Fitted reader artifacts discovered under runs/ (all concepts; the
    /// panel filters to the selected one).
    public private(set) var readerArtifacts: [VectorCatalog.ReaderArtifactRecord] = []
    /// Reader artifacts on the ACTIVE server (`GET /api/readers`) — readers
    /// are substrate-specific instruments, so these are listed read-only next
    /// to the local rows and never merged into `readerArtifacts`.
    public private(set) var serverReaders: [RemoteReaderRecord] = []

    /// Re-list the active server's fitted readers. Quiet on failure (an older
    /// server without the readers route, or a dropped connection, should not
    /// spam the builder status line) — the rows simply stay empty.
    public func refreshServerReaders() async {
        guard let host, case .server = host.cluster.activeWorkspace,
            let client = host.cluster.client
        else {
            serverReaders = []
            return
        }
        serverReaders = (try? await client.listReaders()) ?? []
    }

    public static var readerTemplatesDirectory: URL {
        VectorCatalog.projectRoot.appending(components: "prompts", "templates")
    }

    public var selectedReaderTemplate: RepEReader.TaskTemplate? {
        readerTemplates.first { $0.id == selectedReaderTemplateID }
    }

    public func refreshReaderTemplates() {
        let fm = FileManager.default
        let entries = (try? fm.contentsOfDirectory(
            at: Self.readerTemplatesDirectory, includingPropertiesForKeys: nil)) ?? []
        readerTemplates = entries
            .filter { $0.pathExtension == "json" }
            .compactMap { try? RepEReader.loadTemplate(url: $0) }
            .sorted { $0.id < $1.id }
        if selectedReaderTemplateID == nil {
            selectedReaderTemplateID = readerTemplates.first?.id
        }
        refreshReaderArtifacts()
    }

    public func refreshReaderArtifacts() {
        readerArtifacts = VectorCatalog.scanReaders()
    }

    // MARK: Reader dataset shape (REPE-IMPLEMENTATION-BRIEF §3)

    /// WHICH row shape the reader dataset is authored in. This is a DATA
    /// choice — how many stimuli a row carries — and it is the only half of
    /// the contrast the researcher declares. The other half is the template,
    /// and the CONTRAST MODE is derived from the two by the engine's
    /// `resolveContrastMode`; it is never offered as a third control, because
    /// a mode chosen independently of the data that supports it is a mode the
    /// fit would have to refuse.
    public enum ReaderRowShape: String, Codable, Sendable, CaseIterable, Identifiable {
        /// Two DIFFERENT stimuli under ONE template.
        case contentPair
        /// ONE stimulus under a template's T+ and T− instructions.
        case singleStimulus

        public var id: String { rawValue }

        public var label: String {
            switch self {
            case .contentPair: "content pairs (positive / matched control)"
            case .singleStimulus: "single stimuli (the template's T+/T− pair)"
            }
        }

        /// What the rows say the file is, in the loader's own vocabulary.
        public var datasetShape: RepEReader.Dataset.Shape {
            switch self {
            case .contentPair: .contentPair
            case .singleStimulus: .singleStimulus
            }
        }

        /// The row-id stem the encoder writes, which is also what the split
        /// preview names (`<concept>-pair-<i>` / `<concept>-row-<i>`, the
        /// brief's §3 examples).
        var rowIDStem: String {
            switch self {
            case .contentPair: "pair"
            case .singleStimulus: "row"
            }
        }
    }

    /// The row shape the reader editor is authoring.
    public var readerRowShape: ReaderRowShape = .contentPair {
        didSet {
            guard oldValue != readerRowShape else { return }
            noteMutation()
        }
    }

    /// The single-stimulus working set — the shape `readerPairRows` could not
    /// write before 2026-08-27 (it encoded content pairs only). Each entry
    /// becomes ONE row; the contrast is the template's instruction pair, so a
    /// second text here would be a confound, not data.
    public private(set) var readerStimuli: [String] = []
    /// Draft box for the single-stimulus editor: one stimulus per line.
    public var readerStimulusDraft = ""

    /// Append the draft's non-empty lines to the single-stimulus set.
    public func addReaderStimuli() {
        let added = readerStimulusDraft
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        guard !added.isEmpty else {
            status = "nothing to add — write one stimulus per line"
            return
        }
        readerStimuli.append(contentsOf: added)
        readerStimulusDraft = ""
        status = "added \(added.count) single stimuli (\(readerStimuli.count) rows)"
        noteMutation()
    }

    public func clearReaderStimuli() {
        readerStimuli = []
        noteMutation()
    }

    /// How many rows the current shape would write.
    public var readerRowCount: Int {
        switch readerRowShape {
        case .contentPair: min(positives.count, negatives.count)
        case .singleStimulus: readerStimuli.count
        }
    }

    // MARK: Reader template + contrast resolution (engine-owned refusals)

    /// The template the current selection resolves to, parsed through the
    /// ENGINE's own schema checks — including `validateInstructionSlot`, so a
    /// custom scaffold carrying a `{{instruction}}` slot with no instruction
    /// pair refuses here in exactly the words `loadTemplate` would use.
    ///
    /// A custom template is hashed over the same bytes the fit persists, so
    /// what the pane shows is what the artifact will pin.
    nonisolated static func resolveReaderTemplate(
        conceptName: String, customText: String?,
        registry: RepEReader.TaskTemplate?
    ) throws -> RepEReader.TaskTemplate {
        if let customText {
            guard customText.contains("{{stimulus}}") else {
                throw ChatServiceError(reason: "custom template needs a {{stimulus}} slot")
            }
            let data = try customReaderTemplateData(
                conceptName: conceptName, text: customText)
            var template = try JSONDecoder().decode(
                RepEReader.TaskTemplate.self, from: data)
            template.hash = SHA256.hash(data: data)
                .map { String(format: "%02x", $0) }.joined()
            try template.validateInstructionSlot()
            return template
        }
        guard let registry else {
            throw ChatServiceError(
                reason: "select a task template (prompts/templates/) or write a custom one")
        }
        return registry
    }

    /// The template the reader flow would fit through right now, or nil while
    /// the selection refuses (see ``readerTemplateRefusal``).
    public var resolvedReaderTemplate: RepEReader.TaskTemplate? {
        try? Self.resolveReaderTemplate(
            conceptName: currentConceptName,
            customText: useCustomReaderTemplate ? customReaderTemplateText : nil,
            registry: selectedReaderTemplate)
    }

    /// The engine's refusal for the current template selection, verbatim.
    public var readerTemplateRefusal: String? {
        do {
            _ = try Self.resolveReaderTemplate(
                conceptName: currentConceptName,
                customText: useCustomReaderTemplate ? customReaderTemplateText : nil,
                registry: selectedReaderTemplate)
            return nil
        } catch let error as ChatServiceError {
            return error.reason
        } catch let error as RepEReader.ReaderError {
            return error.reason
        } catch {
            return "\(error)"
        }
    }

    /// The contrast mode the current (row shape × template) DERIVES, or nil
    /// when the two disagree. Never a control: see ``ReaderRowShape``.
    public var readerContrastMode: RepEReader.ContrastMode? {
        guard let template = resolvedReaderTemplate else { return nil }
        return try? RepEReader.resolveContrastMode(
            shape: readerRowShape.datasetShape, template: template)
    }

    /// The ENGINE's typed refusal when the chosen template and the authored
    /// row shape cannot make a contrast — the pinned `resolveContrastMode`
    /// twin literal, shown rather than a silently-filtered picker that would
    /// hide the mismatch from the person who has to fix it.
    public var readerContrastRefusal: String? {
        guard let template = resolvedReaderTemplate else { return nil }
        do {
            _ = try RepEReader.resolveContrastMode(
                shape: readerRowShape.datasetShape, template: template)
            return nil
        } catch let error as RepEReader.ReaderError {
            return error.reason
        } catch {
            return "\(error)"
        }
    }

    // MARK: Held-out split preview (REPE-IMPLEMENTATION-BRIEF §5)

    /// What the writer will actually stamp `split: "test"` on, before it
    /// writes anything: the clamp that keeps two train rows, the ids of the
    /// held-out tail, and whether the split is large enough for the paper's
    /// held-out sign rule to run at all.
    public struct ReaderSplitPreview: Sendable, Equatable {
        public let totalRows: Int
        public let requestedHeldOut: Int
        public let heldOutRows: Int
        public let trainRows: Int
        /// The row ids that will carry `split: "test"`, in file order.
        public let heldOutRowIDs: [String]

        /// True when the request was cut back to protect the train split.
        public var wasClamped: Bool { heldOutRows != requestedHeldOut }

        /// True when the held-out split cannot decide any layer's sign, so
        /// every artifact will stamp `signConvention: "trainMajority"` plus a
        /// `signFallbackReason`. The threshold is the engine's, not a number
        /// this type chose.
        public var signSelectionWillFallBack: Bool {
            heldOutRows < RepEReader.minimumHeldOutPairsForSignSelection
        }

        /// One line saying what the split buys and what it costs.
        public var note: String {
            let minimum = RepEReader.minimumHeldOutPairsForSignSelection
            if totalRows == 0 { return "no rows authored yet" }
            var parts = ["\(trainRows) train / \(heldOutRows) held out"]
            if wasClamped {
                parts.append(
                    "clamped from \(requestedHeldOut): at least 2 rows must stay train")
            }
            if signSelectionWillFallBack {
                parts.append(
                    "below the \(minimum)-pair minimum, so every layer's sign falls "
                        + "back to train-label majority and stamps a signFallbackReason "
                        + "— accuracies are train-only")
            } else {
                parts.append(
                    "enough for the paper's held-out sign selection (minimum \(minimum))")
            }
            return parts.joined(separator: " — ")
        }
    }

    /// Pure preview of the split the row encoder will write. Same clamp as
    /// ``readerPairRows``/``readerStimulusRows``, computed once here so the
    /// pane and the writer cannot disagree about which rows are held out.
    public nonisolated static func readerSplitPreview(
        concept: String, rowCount: Int, requestedHeldOut: Int,
        rowShape: ReaderRowShape
    ) -> ReaderSplitPreview {
        let heldOut = min(max(0, requestedHeldOut), max(0, rowCount - 2))
        let ids = (max(0, rowCount - heldOut) ..< max(0, rowCount)).map {
            "\(concept)-\(rowShape.rowIDStem)-\($0)"
        }
        return ReaderSplitPreview(
            totalRows: rowCount, requestedHeldOut: max(0, requestedHeldOut),
            heldOutRows: heldOut, trainRows: max(0, rowCount - heldOut),
            heldOutRowIDs: ids)
    }

    public var readerSplitPreview: ReaderSplitPreview {
        Self.readerSplitPreview(
            concept: currentConceptName, rowCount: readerRowCount,
            requestedHeldOut: readerHeldOutPairCount, rowShape: readerRowShape)
    }

    // MARK: Reader fit gate

    /// The standing refusal that blocks a reader fit on this target, or nil.
    /// Every one of them is an ENGINE refusal quoted, never a message this
    /// type composed.
    ///
    /// Two deliberate scopings. The READING POSITION is not consulted: a
    /// reader reads at its template's LAT token, so a reading-position
    /// declaration standing refused for the vector builder is not this fit's
    /// business. The RENDERING is — fitting under the last valid rendering
    /// while the pane shows another is exactly the silent substitution the
    /// declaration exists to end — except for the two refusals that are THIS
    /// engine's limit and name the server as their repair, which must not
    /// block a fit queued on the server.
    public func readerFitRefusal(onServer: Bool) -> String? {
        if let refusal = readerTemplateRefusal ?? readerContrastRefusal {
            return refusal
        }
        guard let rendering = extractionRenderingRefusal else { return nil }
        if onServer, extractionRenderingRefusalIsLocalEngineLimit { return nil }
        return rendering
    }

    // MARK: Reader fit request assembly (pure; unit-tested)

    /// The pieces of the server `POST /api/reader/fit` body this builder
    /// sends: exactly one of `templateID` (registry) / `templateJSON`
    /// (custom, unregistered), plus the inline pairs JSONL. `modelID`, when
    /// set, pins which model the server acquires for the fit (the builder's
    /// selected server model) instead of whatever happens to be loaded.
    struct ReaderFitRequest: Equatable, Sendable {
        var concept: String
        var modelID: String?
        var templateID: String?
        var templateJSON: String?
        var pairsJSONL: String
        /// HOW the scaffold reaches the model, declared for the fit. nil is
        /// the raw default and is what an untouched panel sends, so the
        /// request bytes stay identical to every fit queued before the
        /// rendering became declarable.
        var extractionRendering: ExtractionRendering?
        /// The unsupervised T+/T− orientation draw's seed. nil lets the
        /// engine stamp its own default.
        var orientationSeed: UInt64?
    }

    /// Sorted-keys JSONL rows for a reader pairs file: one `RepEReader.Pair`
    /// per matched (positive, negative) index, the LAST k pairs written with
    /// split "test". This single encoder feeds BOTH the local pinned copy
    /// (`prompts/readers/<concept>/pairs.jsonl`, git-versioned truth shared
    /// by both engines) and the inline `pairsJSONL` payload of the server fit
    /// route — so the written bytes, and therefore the dataset hash, are
    /// identical on both substrates.
    nonisolated static func readerPairRows(
        concept: String, positives: [String], negatives: [String],
        heldOutPairCount: Int, templateID: String
    ) throws -> [String] {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let heldOut = min(max(0, heldOutPairCount), max(0, positives.count - 2))
        var lines: [String] = []
        for (index, pair) in zip(positives, negatives).enumerated() {
            let row = RepEReader.Pair(
                id: "\(concept)-pair-\(index)",
                concept: concept,
                positiveStimulus: pair.0,
                negativeStimulus: pair.1,
                split: index >= positives.count - heldOut ? "test" : "train",
                templateID: templateID)
            lines.append(String(decoding: try encoder.encode(row), as: UTF8.self))
        }
        return lines
    }

    /// Sorted-keys JSONL rows for a SINGLE-STIMULUS reader dataset — one
    /// `stimulus` row per text, the LAST k written with split "test"
    /// (REPE-IMPLEMENTATION-BRIEF §3, the paper's §3.1 step 1b shape). The
    /// engine wave left this shape unauthorable: `readerPairRows` writes
    /// content pairs only, so a T+/T− template had no dataset the Concept Lab
    /// could produce for it.
    ///
    /// Row ids are `<concept>-row-<i>` rather than `-pair-<i>`: a row here is
    /// one stimulus, and the two shapes must not be mistakable for one another
    /// in a diff.
    nonisolated static func readerStimulusRows(
        concept: String, stimuli: [String],
        heldOutPairCount: Int, templateID: String
    ) throws -> [String] {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let heldOut = min(max(0, heldOutPairCount), max(0, stimuli.count - 2))
        var lines: [String] = []
        for (index, stimulus) in stimuli.enumerated() {
            let row = RepEReader.Pair.templatePair(
                id: "\(concept)-row-\(index)",
                concept: concept,
                stimulus: stimulus,
                split: index >= stimuli.count - heldOut ? "test" : "train",
                templateID: templateID)
            lines.append(String(decoding: try encoder.encode(row), as: UTF8.self))
        }
        return lines
    }

    /// Raw JSON bytes of a one-off (unregistered) reader template — the same
    /// object the local path persists into its run directory, so a custom
    /// scaffold sent to the server as `templateJSON` is byte-identical to the
    /// one a local build would pin. `conceptSlot` is derived from a
    /// `{{concept}}` slot in the text; the divergence stamp says the template
    /// is not a registry file.
    nonisolated static func customReaderTemplateData(
        conceptName: String, text: String
    ) throws -> Data {
        let object: [String: Any] = [
            "id": "custom-\(conceptName)-v1",
            "conceptSlot": text.contains("{{concept}}"),
            "text": text,
            "latToken": "final",
            "divergence": "custom-unregistered",
        ]
        return try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }

    /// Assembles the server fit request from builder state: template
    /// selection resolves to exactly one of templateID/templateJSON, and the
    /// rows carry the same ids, split assignment, and sorted-keys encoding as
    /// the local pairs file.
    ///
    /// `rowShape` picks WHICH encoder writes the rows — content pairs from
    /// `positives`/`negatives`, or single stimuli from `stimuli` — and the
    /// server derives the contrast mode from the rows and the template exactly
    /// as the local fit does. `extractionRendering` and `orientationSeed` are
    /// nil-defaulted so an untouched panel posts the same bytes it always did.
    nonisolated static func readerFitRequest(
        concept: String,
        positives: [String],
        negatives: [String],
        heldOutPairCount: Int,
        registryTemplateID: String?,
        customTemplateText: String?,
        modelID: String? = nil,
        rowShape: ReaderRowShape = .contentPair,
        stimuli: [String] = [],
        extractionRendering: ExtractionRendering? = nil,
        orientationSeed: UInt64? = nil
    ) throws -> ReaderFitRequest {
        let templateID: String?
        let templateJSON: String?
        let pinnedTemplateID: String
        if let customTemplateText {
            guard customTemplateText.contains("{{stimulus}}") else {
                throw ChatServiceError(reason: "custom template needs a {{stimulus}} slot")
            }
            let data = try customReaderTemplateData(
                conceptName: concept, text: customTemplateText)
            templateID = nil
            templateJSON = String(decoding: data, as: UTF8.self)
            pinnedTemplateID = "custom-\(concept)-v1"
        } else if let registryTemplateID {
            templateID = registryTemplateID
            templateJSON = nil
            pinnedTemplateID = registryTemplateID
        } else {
            throw ChatServiceError(
                reason: "select a task template (prompts/templates/) or write a custom one")
        }
        let lines: [String]
        switch rowShape {
        case .contentPair:
            lines = try readerPairRows(
                concept: concept, positives: positives, negatives: negatives,
                heldOutPairCount: heldOutPairCount, templateID: pinnedTemplateID)
        case .singleStimulus:
            lines = try readerStimulusRows(
                concept: concept, stimuli: stimuli,
                heldOutPairCount: heldOutPairCount, templateID: pinnedTemplateID)
        }
        return ReaderFitRequest(
            concept: concept, modelID: modelID,
            templateID: templateID, templateJSON: templateJSON,
            pairsJSONL: lines.joined(separator: "\n") + "\n",
            extractionRendering: extractionRendering,
            orientationSeed: orientationSeed)
    }

    /// Queue a RepE reader fit on the active server as a durable job
    /// (`POST /api/reader/fit`; same job pattern as probe training /
    /// extraction). Validation mirrors the local `buildReader` path. The
    /// pairs file is still written locally under
    /// `prompts/readers/<name>/pairs.jsonl` (the git-versioned recipe truth
    /// shared by both engines) AND sent inline, so the server pins
    /// byte-identical rows on its tree. The fitted reader artifacts land in
    /// the SERVER's runs/ — readers are substrate-specific and never merge
    /// into the local catalog.
    public func buildReaderOnActiveServer() async {
        guard !isWorking else { return }
        guard let host, case .server = host.cluster.activeWorkspace,
            let client = host.cluster.client
        else {
            status = "connect a server workspace first"
            return
        }
        let name = vectorBuildName
        guard !name.isEmpty else {
            status = "select a vector-build concept"
            return
        }
        // Every gate is an ENGINE refusal, quoted — surfaced here so it
        // arrives before a job is queued rather than as a failed job.
        if let refusal = readerFitRefusal(onServer: true) {
            reportBuildFailure(refusal, title: "Server Reader Fit — refused")
            return
        }
        let pairedRows: (positive: [String], negative: [String])
        switch readerRowShape {
        case .contentPair:
            do {
                pairedRows = try pairedRowsForBuildConcept(name)
            } catch {
                reportBuildFailure(
                    "could not read paired dataset for \(name): \(error)",
                    title: "Reader Fit — failed")
                return
            }
            guard pairedRows.positive.count == pairedRows.negative.count else {
                status = "reader pairs must be matched: \(pairedRows.positive.count)+ vs "
                    + "\(pairedRows.negative.count)- rows"
                return
            }
        case .singleStimulus:
            pairedRows = ([], [])
        }
        let rowCount = readerRowShape == .contentPair
            ? pairedRows.positive.count : readerStimuli.count
        let heldOut = min(max(0, readerHeldOutPairCount), max(0, rowCount - 2))
        guard rowCount - heldOut >= 2 else {
            status = "need at least 2 train rows (have \(rowCount) rows, "
                + "\(heldOut) held out)"
            return
        }

        isWorking = true
        activeTaskLabel = "Building reader (server)"
        lastBuildError = nil
        defer {
            isWorking = false
            activeTaskLabel = nil
        }

        do {
            let request = try Self.readerFitRequest(
                concept: name,
                positives: pairedRows.positive,
                negatives: pairedRows.negative,
                heldOutPairCount: readerHeldOutPairCount,
                registryTemplateID: useCustomReaderTemplate ? nil : selectedReaderTemplateID,
                customTemplateText: useCustomReaderTemplate ? customReaderTemplateText : nil,
                modelID: host.selectedRemoteModelID,
                rowShape: readerRowShape,
                stimuli: readerStimuli,
                extractionRendering: serverExtractionRendering,
                orientationSeed: readerRowShape == .singleStimulus
                    ? readerOrientationSeed : nil)

            // Pin the reader dataset locally exactly as the local build does —
            // the pairs file is shared recipe data, not a per-substrate
            // artifact; the server writes the same bytes on its tree.
            let readersDirectory = VectorCatalog.pairedStimuliDirectory(
                family: .readers, name: name)
            try FileManager.default.createDirectory(
                at: readersDirectory, withIntermediateDirectories: true)
            try request.pairsJSONL.write(
                to: VectorCatalog.pairedStimuliFile(family: .readers, name: name),
                atomically: true, encoding: .utf8)

            try await ensureSelectedServerModelLoaded(host: host, client: client)
            let jobID = try await client.fitReader(
                concept: request.concept,
                modelID: request.modelID,
                templateID: request.templateID,
                templateJSON: request.templateJSON,
                pairsJSONL: request.pairsJSONL,
                extractionRendering: request.extractionRendering,
                orientationSeed: request.orientationSeed)
            status = "server reader fit queued as job \(jobID)…"
            await host.cluster.refreshRemoteState()

            // Follow the durable job in the shared live-log/Activity pane to a
            // terminal state — same treatment as vector extraction, probe
            // training, studies, and Gemma Scope (no more silent polling). The
            // reader artifact does not exist until the fit finishes, so
            // refreshing the catalog at queue time would show nothing.
            let title = "Server Reader Fit: \(name)"
            guard let job = await followServerJobInDisplay(
                jobID: jobID,
                client: client,
                host: host,
                title: title,
                label: "reader fit"
            ) else {
                // Timed out (or the panel's task was cancelled): the job keeps
                // running server-side; say so instead of spinning forever.
                status = "reader fit is still running as job \(jobID) — watch it "
                    + "in Compute; the reader appears in the server "
                    + "catalog when it finishes"
                return
            }
            await host.cluster.refreshRemoteState()
            switch job.status {
            case "succeeded":
                await refreshServerReaders()
                let runDirectory = job.result.flatMap { result -> String? in
                    if case .string(let path)? = result["runDirectory"] { return path }
                    return nil
                }
                status = "server reader fit finished"
                    + (runDirectory.map { " → \($0)" } ?? "")
                    + " — artifacts land in the server's runs/; readers are "
                    + "substrate-specific, so they never merge into the local catalog"
            case "cancelled":
                status = "server reader fit was cancelled (job \(jobID))"
            default:
                // The followed job's live log already carries the error tail;
                // surface it in the banner without a duplicate log entry.
                reportBuildFailure(
                    "server reader fit failed: "
                        + (job.error ?? job.logTail.last ?? "job \(jobID) \(job.status)"),
                    title: title, logToActivity: false)
            }
        } catch let error as ChatServiceError {
            reportBuildFailure(error.reason, title: "Server Reader Fit — failed")
        } catch {
            reportBuildFailure(
                "server reader fit failed: \(error)",
                title: "Server Reader Fit — failed")
        }
    }

    /// Terminal statuses of the server's durable job store
    /// (`Server/steerlab_server/api/dto.py`: pending | running | succeeded |
    /// failed | cancelled, plus "prepared" for staged dry runs and "parked"
    /// for a worker that stopped short with durable state and a recovery
    /// action; Slurm adds transient states, so `finishedAt` is also honored
    /// as terminal). "cancelling" is deliberately NOT terminal — it is the
    /// window between a cancel request and the worker's acknowledgement;
    /// followers keep following through it.
    private static let terminalJobStatuses: Set<String> = [
        "succeeded", "failed", "cancelled", "prepared", "parked",
    ]

    /// Poll a server job every ~2 s until it reaches a terminal state,
    /// surfacing its latest log line through `status` as progress. Returns
    /// nil on timeout (default 10 min — the job keeps running server-side)
    /// or when the surrounding task is cancelled (navigating away must not
    /// crash or spin); transient fetch errors are tolerated and retried.
    private func pollServerJob(
        jobID: String,
        client: ClusterClient,
        label: String,
        every interval: Duration = .seconds(2),
        timeout: TimeInterval = 600
    ) async -> RemoteJobRecord? {
        let deadline = Date().addingTimeInterval(timeout)
        while !Task.isCancelled {
            if let job = try? await client.job(jobID) {
                if Self.terminalJobStatuses.contains(job.status) || job.finishedAt != nil {
                    return job
                }
                status = "\(label) job \(jobID) \(job.status)"
                    + (job.logTail.last.map { ": \($0)" } ?? "…")
            }
            guard Date() < deadline else { return nil }
            try? await Task.sleep(for: interval)
        }
        return nil
    }

    private func followServerJobInDisplay(
        jobID: String,
        client: ClusterClient,
        host: ChatService,
        title: String,
        label: String,
        maxLines: Int = 400
    ) async -> RemoteJobRecord? {
        var lines = ["queued job \(jobID)"]
        let logID = host.startLiveLog(title: title, initialLine: lines[0])

        do {
            try await client.streamJobLog(jobID: jobID) { line in
                await MainActor.run {
                    lines.append(line)
                    if lines.count > maxLines {
                        lines.removeFirst(lines.count - maxLines)
                    }
                    host.updateLiveLog(id: logID, title: title, lines: lines)
                    self.status = "\(label) job \(jobID): \(line)"
                }
            }
        } catch is CancellationError {
            return nil
        } catch {
            lines.append("log stream ended: \(error.localizedDescription)")
            host.updateLiveLog(id: logID, title: title, lines: lines)
        }

        if let job = try? await client.job(jobID),
            Self.terminalJobStatuses.contains(job.status) || job.finishedAt != nil
        {
            lines.append("job \(job.status)")
            if let error = job.error, !error.isEmpty {
                lines.append(error)
            }
            host.updateLiveLog(id: logID, title: title, lines: lines)
            return job
        }
        // A broken stream can leave the job in a NON-terminal state (e.g.
        // "running" or "cancelling") — returning it here would make the
        // status switches above misread an in-flight job as a failure, so
        // keep polling until it actually resolves.
        return await pollServerJob(jobID: jobID, client: client, label: label)
    }

    /// Fits one reader per layer from the working paired set through the
    /// selected task template. In a server workspace this delegates to
    /// `buildReaderOnActiveServer` (durable server job over the same pinned
    /// pairs bytes); locally it writes the pinned pairs file
    /// (`prompts/readers/<concept>/pairs.jsonl`), fits on the loaded model,
    /// and saves one artifact JSON per layer into a fresh immutable run
    /// directory.
    public func buildReader() async {
        guard !isWorking else { return }
        if let host, case .server = host.cluster.activeWorkspace {
            await buildReaderOnActiveServer()
            return
        }
        guard let host else {
            status = "load a model first"
            return
        }
        let name = buildTargetName
        guard !name.isEmpty else {
            status = "select a vector-build concept"
            return
        }
        // The template's schema refusal, the (shape × template) contrast
        // refusal, and a standing rendering refusal are all the ENGINE's, and
        // they are the refusals this fit would raise anyway — surfaced before
        // a forward pass is spent.
        if let refusal = readerFitRefusal(onServer: false) {
            reportBuildFailure(refusal, title: "Reader Fit — refused")
            return
        }
        let pairedRows: (positive: [String], negative: [String])
        switch readerRowShape {
        case .contentPair:
            do {
                pairedRows = try pairedRowsForBuildConcept(name)
            } catch {
                reportBuildFailure(
                    "could not read paired dataset for \(name): \(error)",
                    title: "Reader Fit — failed")
                return
            }
            guard pairedRows.positive.count == pairedRows.negative.count else {
                status = "reader pairs must be matched: \(pairedRows.positive.count)+ vs "
                    + "\(pairedRows.negative.count)- rows"
                return
            }
        case .singleStimulus:
            pairedRows = ([], [])
        }
        let rowCount = readerRowShape == .contentPair
            ? pairedRows.positive.count : readerStimuli.count
        let heldOut = min(max(0, readerHeldOutPairCount), max(0, rowCount - 2))
        guard rowCount - heldOut >= 2 else {
            status = "need at least 2 train rows (have \(rowCount) rows, "
                + "\(heldOut) held out)"
            return
        }

        isWorking = true
        activeTaskLabel = "Building reader"
        lastBuildError = nil
        defer {
            isWorking = false
            activeTaskLabel = nil
        }

        // Same rule as the vector build: the selected model drives the fit.
        do {
            try await ensureSelectedLocalModelLoaded(host: host)
        } catch {
            reportBuildFailure(
                "reader fit did not start: \(error)", title: "Reader Fit — failed")
            return
        }
        guard let container = host.containerForExtraction,
            let modelID = host.loadedModelID
        else {
            status = "load a model first"
            return
        }

        do {
            let runDirectory = try VectorCatalog.makeUniqueRunDirectory(
                slug: "reader-\(name)")
            try RunMetadata.write(
                runType: "reader-fit", to: runDirectory,
                modelID: modelID,
                revision: SteeredContainerLoader.cachedRevision(for: modelID))
            let template: RepEReader.TaskTemplate
            if useCustomReaderTemplate {
                // A custom scaffold is persisted into the run directory and
                // hashed from its raw bytes — same pin discipline as the
                // registry, without polluting the shared template files. The
                // divergence is stamped so the artifact says it is not a
                // registry template.
                let text = customReaderTemplateText
                guard text.contains("{{stimulus}}") else {
                    status = "custom template needs a {{stimulus}} slot"
                    return
                }
                // Shared encoder with the server fit path (`templateJSON`),
                // so a custom scaffold pins the same bytes on both substrates.
                let data = try Self.customReaderTemplateData(conceptName: name, text: text)
                let templateURL = runDirectory.appending(
                    component: "custom-\(name)-v1.json")
                try data.write(to: templateURL)
                template = try RepEReader.loadTemplate(url: templateURL)
            } else if let selected = selectedReaderTemplate {
                template = selected
            } else {
                status = "select a task template (prompts/templates/) or write a custom one"
                return
            }

            // Pin the reader dataset: unrendered stimuli + templateID + split,
            // hashed over the written bytes (server pairs.jsonl contract).
            let readersDirectory = VectorCatalog.pairedStimuliDirectory(
                family: .readers, name: name)
            try FileManager.default.createDirectory(
                at: readersDirectory, withIntermediateDirectories: true)
            // Shared row encoder with the server fit path (`pairsJSONL`), so
            // both engines pin byte-identical rows → identical dataset hash.
            let lines: [String]
            switch readerRowShape {
            case .contentPair:
                lines = try Self.readerPairRows(
                    concept: name,
                    positives: pairedRows.positive,
                    negatives: pairedRows.negative,
                    heldOutPairCount: readerHeldOutPairCount, templateID: template.id)
            case .singleStimulus:
                lines = try Self.readerStimulusRows(
                    concept: name, stimuli: readerStimuli,
                    heldOutPairCount: readerHeldOutPairCount, templateID: template.id)
            }
            let pairsURL = VectorCatalog.pairedStimuliFile(family: .readers, name: name)
            try (lines.joined(separator: "\n") + "\n").write(
                to: pairsURL, atomically: true, encoding: .utf8)
            let dataset = try RepEReader.loadPairs(url: pairsURL)

            status = "fitting reader (\(dataset.pairs.count * 2) scaffold passes)…"
            let artifacts = try await RepEReader.fit(
                container: container,
                modelID: modelID,
                revision: SteeredContainerLoader.cachedRevision(for: modelID),
                dataset: dataset,
                template: template,
                orientationSeed: readerOrientationSeed,
                extractionRendering: extractionRendering)
            for artifact in artifacts {
                try RepEReader.saveArtifact(artifact, to: runDirectory)
            }
            readerLayerScores = artifacts.map(ReaderLayerScore.init(artifact:))
            readerFitStamps = Self.readerFitStamps(from: artifacts)
            lastReaderRunDirectory = runDirectory
            refreshReaderArtifacts()
            // The RECOMMENDATION comes off the artifacts themselves (the fit
            // stamps it), not from a second argmax here: two places computing
            // "best layer" is how the pane and the artifact come to disagree.
            let bestNote = artifacts.first.flatMap { first -> String? in
                guard let layer = first.recommendedLayer,
                    let accuracy = first.recommendedLayerAccuracy
                else { return nil }
                return String(
                    format: "recommended layer %d (%@ %.0f%%) — a recommendation; "
                        + "the study declares which layer it reads",
                    layer, first.layerRecommendationBasis == "heldOutAccuracy"
                        ? "held-out" : "train", accuracy * 100)
            }
            status = "built \(artifacts.count) per-layer readers for '\(name)' "
                + "→ \(runDirectory.lastPathComponent)"
                + (bestNote.map { "; \($0)" } ?? "")
                + (heldOut == 0
                    ? " — no held-out pairs; accuracies are train-only" : "")
        } catch {
            reportBuildFailure(
                "reader build failed: \(error)", title: "Reader Fit — failed")
        }
    }

    /// What a reader-derived steering vector carries, and what it still owes
    /// before a study can cite it (REPE-IMPLEMENTATION-BRIEF §6).
    ///
    /// The refusal is not composed here: it is
    /// ``ExperimentStore/readerDerivedNormBackfillRefusal(artifact:)``, the
    /// literal `attachArtifact` raises and the twin of the server's
    /// `experiment_store.reader_derived_norm_backfill_refusal`. Showing it at
    /// the derive step is the only way the missing LIFECYCLE STEP arrives
    /// before the attach that would refuse.
    public struct DerivedReaderVectorSummary: Sendable, Equatable {
        /// Workspace-relative locator (`runs/<run>/<name>`), the same string
        /// an attach takes.
        public let reference: String
        public let details: [ReaderDetailRow]
        /// The engine's own attach refusal while this direction has no
        /// residual-norm denominator; nil once one has been measured.
        public let normBackfillRefusal: String?
        /// True when every gate a reader-derived direction must pass is
        /// passed: the method is in the vocabulary and the denominator exists.
        public var isAttachable: Bool { normBackfillRefusal == nil }

        /// What attaching this direction means, said once at the affordance.
        public var attachabilityNote: String {
            isAttachable
                ? "attachable: `repeReaderLAT` is in the ExtractionMethod "
                    + "vocabulary, so this pins as a pinnedArtifact whose SOURCE "
                    + "method resolves. It has no source concept — its stimuli are "
                    + "the reader's dataset and its held-out evidence is the reader "
                    + "artifact's own accuracy, so no validation.jsonl is owed."
                : "not attachable yet — measure norms first (below), then attach "
                    + "the BACKFILLED copy."
        }
    }

    /// Pure projection of a derived artifact's sidecar into the provenance the
    /// derive affordance shows.
    public nonisolated static func derivedReaderVectorSummary(
        sidecar: SteeringVectorSidecar, reference: String
    ) -> DerivedReaderVectorSummary {
        var details: [ReaderDetailRow] = [
            ReaderDetailRow(
                label: "source", value: sidecar.source ?? RepEReader.artifactType),
            ReaderDetailRow(
                label: "control mode",
                value: sidecar.controlMode ?? RepEReader.controlMode),
        ]
        if let layer = sidecar.readerLayer {
            details.append(ReaderDetailRow(label: "reader layer", value: "\(layer)"))
        }
        if let template = sidecar.readerTemplateID {
            details.append(
                ReaderDetailRow(
                    label: "reader template",
                    value: template
                        + (sidecar.readerTemplateHash.map { " · \($0.prefix(12))…" } ?? "")))
        }
        if let mode = sidecar.readerContrastMode {
            details.append(
                ReaderDetailRow(
                    label: "contrast mode",
                    value: RepEReader.ContrastMode(rawValue: mode)?.label ?? mode))
        }
        if let convention = sidecar.readerSignConvention {
            details.append(
                ReaderDetailRow(
                    label: "sign convention",
                    value: RepEReader.SignConvention(rawValue: convention)?.label
                        ?? convention,
                    isCaution: convention
                        == RepEReader.SignConvention.trainMajority.rawValue))
        }
        if let orientation = sidecar.readerProbeOrientation {
            details.append(
                ReaderDetailRow(
                    label: "probe orientation",
                    value: orientation < 0
                        ? "−1 — the probe reads ANTI-aligned with PC1, so the sign "
                            + "is folded into the vector bytes (a steering vector has "
                            + "no orientation field to carry it)"
                        : "+1 — the stored direction already points at more concept"))
        }
        let refusal: String? =
            sidecar.residualNormSource == nil
            ? ExperimentStore.readerDerivedNormBackfillRefusal(artifact: reference)
            : nil
        return DerivedReaderVectorSummary(
            reference: reference, details: details, normBackfillRefusal: refusal)
    }

    /// The newest reader → steering-vector conversion, with its provenance and
    /// its standing refusal. Cleared when the family or concept moves.
    public private(set) var lastDerivedReaderVector: DerivedReaderVectorSummary?

    /// Explicit reader → steering-vector conversion (REPE-IMPLEMENTATION-BRIEF §6). The derived
    /// artifact's sidecar stamps `source`, `readerID`, `readerHash`, and the
    /// honest `controlMode`, and it enters the normal vector catalog.
    public func deriveSteeringVectorFromReader(
        _ record: VectorCatalog.ReaderArtifactRecord
    ) {
        do {
            let runDirectory = try VectorCatalog.makeUniqueRunDirectory(
                slug: "derived-\(record.artifact.concept)-repe-reader")
            try RunMetadata.write(
                runType: "derive-reader-vector", to: runDirectory,
                modelID: record.artifact.modelID,
                revision: record.artifact.revision)
            let base = try RepEReader.deriveSteeringVector(
                readerURL: record.url, into: runDirectory)
            let sidecar = try JSONDecoder().decode(
                SteeringVectorSidecar.self,
                from: Data(contentsOf: base.appendingPathExtension("json")))
            let root = VectorCatalog.projectRoot.standardizedFileURL.path + "/"
            let path = base.standardizedFileURL.path
            lastDerivedReaderVector = Self.derivedReaderVectorSummary(
                sidecar: sidecar,
                reference: path.hasPrefix(root)
                    ? String(path.dropFirst(root.count)) : path)
            host?.refreshVectors()
            host?.selectVector(base.path)
            status = "derived steering vector from reader "
                + "'\(record.fileName)' → \(runDirectory.lastPathComponent) "
                + "(controlMode: \(RepEReader.controlMode)); selected for steering"
        } catch {
            lastDerivedReaderVector = nil
            status = "derive failed: \(error)"
        }
    }

    // MARK: - Residual-norm backfill (norm-unit denominator for legacy artifacts)

    /// Measure per-layer residual norms for an existing LOCAL artifact whose
    /// sidecar lacks them, on the same pinned neutral corpus extraction uses
    /// (`prompts/neutral/corpus.jsonl` — the norm-calibration corpus, not a
    /// second picker), and write a NEW artifact into a fresh run directory.
    /// Originals are immutable: backfill never overwrites.
    public func backfillNormsLocally(_ artifact: VectorArtifact) async {
        guard !isWorking else { return }
        guard let host, let container = host.containerForExtraction,
            let modelID = host.loadedModelID
        else {
            status = "load a model first"
            return
        }
        guard artifact.sidecar.residualNormPerLayer == nil else {
            status = "'\(artifact.name)' already records residual norms — "
                + "backfill never overwrites"
            return
        }
        guard artifact.sidecar.modelID == modelID else {
            status = "load \(artifact.sidecar.modelID) first — residual norms "
                + "are a per-model measurement"
            return
        }

        isWorking = true
        activeTaskLabel = "Measuring norms"
        defer {
            isWorking = false
            activeTaskLabel = nil
        }

        do {
            let runDirectory = try VectorCatalog.makeUniqueRunDirectory(
                slug: "norms-\(artifact.sidecar.concept)")
            status = "measuring residual norms on the neutral corpus…"
            let base = try await NormBackfill.backfillNorms(
                container: container,
                modelID: modelID,
                artifact: URL(filePath: artifact.id),
                corpusURL: Self.neutralCorpusURL(),
                runDirectory: runDirectory)
            host.refreshVectors()
            status = "backfilled norms for '\(artifact.name)' → "
                + "\(runDirectory.lastPathComponent)/\(base.lastPathComponent) "
                + "(original untouched)"
        } catch {
            status = "norm backfill failed: \(error)"
        }
    }

    /// Where the norm-unit denominator comes from, for display at every
    /// backfill affordance: the workspace's pinned neutral corpus and its
    /// raw-bytes SHA-256 (the engines' single-file corpus hash convention —
    /// `stimulus_set.py` mirrors `StimulusSet.swift`). Shown because the
    /// measurement becomes the new artifact's `residualNormSource:
    /// "neutral-corpus"` + `neutralCorpusHash` provenance. Hash nil when the
    /// workspace has no corpus file yet.
    public nonisolated static func neutralCorpusProvenance() -> (path: String, hash: String?) {
        let relativePath = "prompts/neutral/corpus.jsonl"
        guard let data = try? Data(contentsOf: neutralCorpusURL()) else {
            return (relativePath, nil)
        }
        let digest = SHA256.hash(data: data)
        return (relativePath, digest.map { String(format: "%02x", $0) }.joined())
    }

    /// Queue a residual-norm backfill on the active server as a durable job
    /// (`POST /api/vectors/backfill-norms`; same poll pattern as reader fit).
    /// The server measures on ITS checkout of `prompts/neutral/corpus.jsonl`
    /// and writes the new artifact into its own runs/ tree — per-substrate,
    /// never merged locally.
    public func backfillNormsOnActiveServer(_ record: RemoteVectorRecord) async {
        guard !isWorking else { return }
        isWorking = true
        activeTaskLabel = "Measuring norms (server)"
        defer {
            isWorking = false
            activeTaskLabel = nil
        }
        _ = await backfillNormsOnActiveServerCore(record)
    }

    /// Backfill EVERY listed norms-less record for the selected server model,
    /// sequentially (one durable job each — the server serializes model use
    /// anyway). Records for other models are skipped loudly, not silently: a
    /// norm table is a per-model measurement, and the per-record path
    /// requires the artifact's own model selected.
    public func backfillAllNormsOnActiveServer(_ records: [RemoteVectorRecord]) async {
        guard !isWorking else { return }
        guard let host, case .server = host.cluster.activeWorkspace,
            host.cluster.client != nil
        else {
            status = "connect a server workspace first"
            return
        }
        let selected = host.selectedRemoteModelID ?? ""
        let eligible = records.filter { record in
            record.residualNormPerLayer == nil && record.hasResidualNorms != true
                && (selected.isEmpty || record.modelID == selected)
        }
        let skipped = records.count - eligible.count
        guard !eligible.isEmpty else {
            status = "no backfillable vectors for the selected server model — "
                + "select the artifacts' model first (norms are a per-model "
                + "measurement)"
            return
        }

        isWorking = true
        activeTaskLabel = "Measuring norms (server)"
        defer {
            isWorking = false
            activeTaskLabel = nil
        }
        var succeeded = 0
        for (index, record) in eligible.enumerated() {
            status = "measuring norms \(index + 1)/\(eligible.count): "
                + "'\(record.name)'…"
            if await backfillNormsOnActiveServerCore(record) { succeeded += 1 }
        }
        status = "norm backfill finished for \(succeeded)/\(eligible.count) "
            + "vectors"
            + (skipped > 0
                ? " (\(skipped) skipped: other models — select their model "
                    + "and run again)"
                : "")
            + " — new artifacts are in the server catalog; originals untouched"
    }

    /// One record's backfill: guards, job submission, poll, status. Returns
    /// success so `backfillAllNormsOnActiveServer` can count honestly. The
    /// caller owns `isWorking`/`activeTaskLabel`.
    private func backfillNormsOnActiveServerCore(_ record: RemoteVectorRecord) async -> Bool {
        guard let host, case .server = host.cluster.activeWorkspace,
            let client = host.cluster.client
        else {
            status = "connect a server workspace first"
            return false
        }
        guard record.residualNormPerLayer == nil, record.hasResidualNorms != true
        else {
            status = "'\(record.name)' already records residual norms — "
                + "backfill never overwrites"
            return false
        }
        if let selected = host.selectedRemoteModelID, !selected.isEmpty,
            selected != record.modelID
        {
            status = "select \(record.modelID) as the server model first — "
                + "residual norms are a per-model measurement"
            return false
        }

        do {
            try await ensureSelectedServerModelLoaded(host: host, client: client)
            let jobID = try await client.backfillNorms(
                vectorID: record.id,
                neutralCorpusPath: "prompts/neutral/corpus.jsonl",
                modelID: host.selectedRemoteModelID)
            status = "server norm backfill queued as job \(jobID)…"
            guard
                let job = await pollServerJob(
                    jobID: jobID, client: client, label: "norm backfill")
            else {
                status = "norm backfill is still running as job \(jobID) — the new "
                    + "artifact appears in the server catalog when it finishes"
                // Still running is not a failure; the artifact lands later.
                return true
            }
            await host.cluster.refreshRemoteState()
            switch job.status {
            case "succeeded":
                await host.catalog.refreshRemoteVectors()
                let runDirectory = job.result.flatMap { result -> String? in
                    if case .string(let path)? = result["runDirectory"] { return path }
                    return nil
                }
                status = "server norm backfill finished"
                    + (runDirectory.map { " → \($0)" } ?? "")
                    + " — new artifact in the server catalog; the original is untouched"
                return true
            case "cancelled":
                status = "server norm backfill was cancelled (job \(jobID))"
                return false
            default:
                // The backfill is polled, not log-followed — nothing else put
                // this failure in the banner/Activity pane, so route it there
                // (same treatment as the reader fit).
                reportBuildFailure(
                    "server norm backfill failed: "
                        + (job.error ?? job.logTail.last ?? "job \(jobID) \(job.status)"),
                    title: "Server Norm Backfill — failed")
                return false
            }
        } catch let error as ChatServiceError {
            reportBuildFailure(error.reason, title: "Server Norm Backfill — failed")
            return false
        } catch {
            reportBuildFailure(
                "server norm backfill failed: \(error)",
                title: "Server Norm Backfill — failed")
            return false
        }
    }

    // MARK: - Pre-derivation reuse check (reuse/cost surfacing)

    /// What the Concept Vector Builder's build action would get into right
    /// now on the active workspace: a fresh artifact to reuse, a stale one
    /// (re-extract only — never silent reuse), or the honest cost of a new
    /// extraction. The live pins come from the on-disk recipe files (recipes
    /// are local truth): the CAA/LAT stimulus-set hash for paired recipes,
    /// the selected build-row corpus hash (local) or the target's raw
    /// `stories.jsonl` hash (server — what its jobs stamp) for grand-mean.
    /// Returns nil when there is not enough state to advise (no model
    /// picked, no named concept/target, empty recipe).
    public func vectorDerivationAdvice() -> DerivationAdvice? {
        guard let host, let modelID = host.workspaceSelectedModelID, !modelID.isEmpty
        else { return nil }
        let workspace = host.cluster.activeWorkspace
        let records = host.catalog.vectorArtifacts(for: workspace)
        switch recipeFamily {
        case .jlensTokenDirection:
            // Freshness here would have to compare a DERIVATION identity (lens
            // hash + token id), not a stimulus hash. Advising from a stimulus
            // comparison would be meaningless, and guessing is worse than
            // staying quiet — the derive verb is seconds either way.
            return nil
        case .repeReaderLAT:
            // Readers are measurement artifacts outside the substrate vector
            // index; Build reader reports its own pass cost.
            return nil
        case .caaMeanDifference, .pairedDifferencePCA:
            let name = vectorBuildName
            guard !name.isEmpty else { return nil }
            let directory = VectorCatalog.conceptsDirectory.appending(component: name)
            let disk = try? StimulusSet(directory: directory)
            let stimulusCount: Int
            if let disk {
                stimulusCount = disk.positive.count + disk.negative.count
            } else if name == currentName {
                stimulusCount = positives.count + negatives.count
            } else {
                stimulusCount = 0
            }
            guard stimulusCount > 0 else { return nil }
            return DerivationPlanner.advise(
                records: records,
                concept: name,
                method: recipeFamily.recipeMethod,
                workspace: workspace,
                modelID: modelID,
                stimulusHash: disk?.hash,
                cost: .paired(stimulusCount: stimulusCount))
        case .designatedReference:
            let name = buildTargetName
            guard !name.isEmpty else { return nil }
            guard let target = storyCorpus(for: name), !target.rows.isEmpty
            else { return nil }
            let reference = designatedReferenceConcept
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let referenceCount = reference.isEmpty
                ? 0 : storyCorpusRowCount(for: reference)
            // Both engines stamp the TARGET's raw stories.jsonl hash as the
            // artifact's stimulus identity (the reference rides alongside as
            // its own pin), so freshness compares that hash on either
            // workspace. Cost is one pass per story on either side.
            return DerivationPlanner.advise(
                records: records,
                concept: name,
                method: recipeFamily.recipeMethod,
                workspace: workspace,
                modelID: modelID,
                stimulusHash: target.hash,
                cost: .paired(stimulusCount: target.rows.count + referenceCount))
        case .emotionGrandMean:
            let rows = emotionRowsForExtraction()
            guard !rows.isEmpty else { return nil }
            let targets = grandMeanTargetConceptNames
            guard
                let target = targets.contains(currentName)
                    ? currentName : targets.sorted().first
            else { return nil }
            let stimulusHash: String?
            switch workspace {
            case .local:
                stimulusHash = Self.sha256(multiConceptJSONL(rows))
            case .server:
                let url = VectorCatalog.emotionsDirectory
                    .appending(component: target)
                    .appending(component: "stories.jsonl")
                stimulusHash = (try? StimulusSet.loadMultiConceptTexts(url: url))?.hash
            }
            return DerivationPlanner.advise(
                records: records,
                concept: target,
                method: recipeFamily.recipeMethod,
                workspace: workspace,
                modelID: modelID,
                stimulusHash: stimulusHash,
                cost: .grandMean(
                    storyCount: rows.count,
                    conceptCount: Set(rows.map(\.concept)).count))
        }
    }

    /// Cost hint for probe training: one forward pass per labeled example.
    /// No reuse check — probe artifacts are not in the substrate vector
    /// index — so this is honesty about cost only. nil when there is nothing
    /// to count or no model is picked.
    public func probeTrainingCostLine() -> String? {
        guard let host, let modelID = host.workspaceSelectedModelID, !modelID.isEmpty,
            !probeExamples.isEmpty
        else { return nil }
        return DerivationPlanner.costLine(
            .probe(exampleCount: probeExamples.count),
            modelID: modelID,
            workspace: host.cluster.activeWorkspace)
    }

    // MARK: - Save

    /// Writes the stimulus files and extracts the canonical vector through
    /// the same pipeline the CLI uses, into a fresh immutable run directory.
    public func saveConceptAndExtract() async {
        guard !isWorking else { return }
        guard let host else {
            status = "load a model first"
            return
        }
        let datasetName = currentName
        // Grand mean builds the emotion selector's target set, which is
        // anchored on the dataset selection and its story corpus; every other
        // family builds the Concept Vector Builder's own picker.
        let name = recipeFamily == .emotionGrandMean ? datasetName : buildTargetName
        guard !name.isEmpty else {
            status = "need a concept name"
            return
        }

        isWorking = true
        activeTaskLabel = "Generating vector"
        lastBuildError = nil
        defer {
            isWorking = false
            activeTaskLabel = nil
        }

        // The selected model DRIVES the build (substrate-as-workspace): load
        // it before touching the container, and fail the build on a load
        // error rather than extracting on whatever was resident.
        do {
            try await ensureSelectedLocalModelLoaded(host: host)
        } catch {
            reportBuildFailure(
                "vector build did not start: \(error)", title: "Vector Build — failed")
            return
        }
        guard let container = host.containerForExtraction,
            let modelID = host.loadedModelID
        else {
            status = "load a model first"
            return
        }

        do {
            if recipeFamily == .repeReaderLAT {
                status = "RepE readers are built with Build reader (concept data "
                    + "→ reader artifact → optional steering variant), not saved "
                    + "as a plain vector"
                return
            }
            if recipeFamily == .emotionGrandMean {
                if !multiConceptDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    await addDrafts()
                    if !multiConceptDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        return
                    }
                }
                try await saveEmotionGrandMeanConcept(
                    name: name, container: container, modelID: modelID, host: host)
                return
            }
            if recipeFamily == .designatedReference {
                if !multiConceptDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    await addDrafts()
                    if !multiConceptDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        return
                    }
                }
                try await saveDesignatedReferenceConcept(
                    name: name, container: container, modelID: modelID, host: host)
                return
            }
            let pairedRows = try pairedRowsForBuildConcept(name)
            guard pairedRows.positive.count >= 4, pairedRows.negative.count >= 4 else {
                status = "need at least 4 paired stimuli per side"
                return
            }
            let directory = VectorCatalog.conceptsDirectory.appending(component: name)
            if name == currentName {
                try FileManager.default.createDirectory(
                    at: directory, withIntermediateDirectories: true)
                try jsonl(pairedRows.positive).write(
                    to: directory.appending(component: "positive.jsonl"),
                    atomically: true, encoding: .utf8)
                try jsonl(pairedRows.negative).write(
                    to: directory.appending(component: "negative.jsonl"),
                    atomically: true, encoding: .utf8)
            }
            let canonicalDatasetHash: String
            let canonicalDatasetPath: String
            if recipeFamily == .pairedDifferencePCA {
                let repeDirectory = VectorCatalog.pairedStimuliDirectory(
                    family: .repe, name: name)
                try FileManager.default.createDirectory(
                    at: repeDirectory, withIntermediateDirectories: true)
                let pairsURL = VectorCatalog.pairedStimuliFile(family: .repe, name: name)
                try pairJSONL(
                    concept: name,
                    positives: pairedRows.positive,
                    negatives: pairedRows.negative
                ).write(
                    to: pairsURL, atomically: true, encoding: .utf8)
                let loadedPairs = try StimulusSet.loadPairs(url: pairsURL)
                canonicalDatasetHash = loadedPairs.hash
                canonicalDatasetPath = VectorCatalog.pairedStimuliRelativePath(
                    family: .repe, name: name)
            } else {
                canonicalDatasetHash = try StimulusSet(directory: directory).hash
                canonicalDatasetPath = "prompts/concepts/\(name)"
            }

            status = "extracting vector…"
            let stimuli = try StimulusSet(directory: directory)
            // Neutral corpus (when present) = the norm-unit denominator,
            // fixed across concepts (emotion paper convention).
            let corpusURL = VectorCatalog.projectRoot.appending(
                components: "prompts", "neutral", "corpus.jsonl")
            let corpus = try? StimulusSet.loadTexts(url: corpusURL)
            let extraction = try await ConceptExtractor.extract(
                container: container, stimuli: stimuli, options: extractionOptions,
                neutralTexts: corpus?.texts)
            let vectors = extraction.vectors

            let runDirectory = try VectorCatalog.makeUniqueRunDirectory(
                slug: "concept-\(name)")
            try RunMetadata.write(
                runType: "extract", to: runDirectory,
                modelID: modelID,
                revision: SteeredContainerLoader.cachedRevision(for: modelID))
            let normSource =
                extraction.residualNormSource == "neutral-corpus"
                ? "neutral-corpus \(corpus?.hash.prefix(12) ?? "")"
                : extraction.residualNormSource
            let recipe = VectorExtractionRecipe(
                name: "\(name)-\(recipeFamily.rawValue)",
                method: recipeFamily.recipeMethod,
                targetConcept: name,
                datasets: [
                    .init(
                        role: recipeFamily == .pairedDifferencePCA ? .pairedContrastive : .positive,
                        path: canonicalDatasetPath,
                        sha256: canonicalDatasetHash)
                ],
                readingPosition: extraction.options.readingPosition,
                neutralPCSelection: .none,
                // nil for the legacy raw rendering, which keeps the recipe's
                // encoded bytes — and therefore its canonicalHash — exactly
                // where they were.
                extractionRendering: extraction.options.extractionRendering,
                notes: recipeFamily == .pairedDifferencePCA
                    ? "Paired-difference PCA direction is norm-matched to the CAA mean-difference vector for direction-only comparison — OUR normalization, not the paper's."
                    : nil)
            let sidecar = SteeringVectorSidecar(
                modelID: modelID,
                revision: SteeredContainerLoader.cachedRevision(for: modelID),
                concept: name,
                stimulusSetHash: stimuli.hash, vectors: vectors,
                options: extraction.options,
                residualNormPerLayer: extraction.residualNormPerLayer,
                residualNormSource: normSource,
                residualNormConvention: extraction.residualNormConvention,
                recipeMethod: recipe.method,
                recipeHash: recipe.canonicalHash(),
                recipeName: recipe.name)
            let artifactName = "\(name)-\(host.selectedModel.family)"
            try SteeringVectorStore.save(
                vectors: vectors, sidecar: sidecar, to: runDirectory, name: artifactName,
                neutralMeanPerLayer: extraction.neutralMeanPerLayer)

            refreshConceptList()
            // Selecting the built concept is for the just-created-from-drafts
            // case (the dataset selection was nil). A build of a DIFFERENT
            // target must not yank the Dataset Builder away from the concept
            // the researcher is editing — and would discard its unsaved rows.
            if name == datasetName { selectedExisting = name }
            host.refreshVectors()
            let artifactID = runDirectory.appending(component: artifactName).path
            host.selectVector(artifactID)
            host.steeringEnabled = true
            status = "saved \(name) (\(stimuli.positive.count)+\(stimuli.negative.count) stimuli) "
                + "→ \(runDirectory.lastPathComponent); selected for steering"
            refreshStaleness()
        } catch {
            reportBuildFailure("save failed: \(error)", title: "Vector Build — failed")
        }
    }

    /// The local designated-reference build: both classes are stories
    /// corpora read whole from disk (the same class loader the CLI lifecycle
    /// uses — no split filtering; validation lives in its own file), the math
    /// is the unpaired class-mean difference, and the artifact pins the
    /// reference {name, hash} exactly as `experiment attach` would.
    private func saveDesignatedReferenceConcept(
        name: String, container: ModelContainer, modelID: String, host: ChatService
    ) async throws {
        let reference = designatedReferenceConcept
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !reference.isEmpty else {
            reportBuildFailure(
                "designated-reference build needs a reference stories concept "
                    + "— the reference corpus is part of the recipe",
                title: "Vector Build — refused")
            return
        }
        if let refusal = designatedReferenceRefusal {
            reportBuildFailure(refusal, title: "Vector Build — refused")
            return
        }
        guard let target = storyCorpus(for: name), !target.rows.isEmpty else {
            reportBuildFailure(
                "no stories.jsonl for concept '\(name)' under prompts/emotions/ "
                    + "— add story rows (or import a corpus) before building",
                title: "Vector Build — refused")
            return
        }
        guard let referenceCorpus = storyCorpus(for: reference),
            !referenceCorpus.rows.isEmpty
        else {
            reportBuildFailure(
                "no stories.jsonl for reference '\(reference)' under "
                    + "prompts/emotions/ — author the reference corpus first",
                title: "Vector Build — refused")
            return
        }

        status = "extracting designated-reference vector…"
        let stimuli = StimulusSet(
            name: name,
            positive: target.rows.map(\.text),
            negative: referenceCorpus.rows.map(\.text),
            hash: target.hash)
        let corpusURL = VectorCatalog.projectRoot.appending(
            components: "prompts", "neutral", "corpus.jsonl")
        let corpus = try? StimulusSet.loadTexts(url: corpusURL)
        let extraction = try await ConceptExtractor.extract(
            container: container, stimuli: stimuli, options: extractionOptions,
            neutralTexts: corpus?.texts)

        let runDirectory = try VectorCatalog.makeUniqueRunDirectory(
            slug: "concept-\(name)")
        try RunMetadata.write(
            runType: "extract", to: runDirectory,
            modelID: modelID,
            revision: SteeredContainerLoader.cachedRevision(for: modelID))
        let normSource =
            extraction.residualNormSource == "neutral-corpus"
            ? "neutral-corpus \(corpus?.hash.prefix(12) ?? "")"
            : extraction.residualNormSource
        let recipe = VectorExtractionRecipe(
            name: "\(name)-designated-reference",
            method: .designatedReference,
            targetConcept: name,
            datasets: [
                .init(
                    role: .positive,
                    path: "prompts/emotions/\(name)/stories.jsonl",
                    sha256: target.hash),
                .init(
                    role: .negative,
                    path: "prompts/emotions/\(reference)/stories.jsonl",
                    sha256: referenceCorpus.hash,
                    notes: "designated reference '\(reference)'"),
            ],
            readingPosition: extraction.options.readingPosition,
            neutralPCSelection: .none,
            extractionRendering: extraction.options.extractionRendering,
            notes: "Vector = mean(concept stories) - mean(reference "
                + "'\(reference)' stories); -alpha steers toward the "
                + "reference class.")
        var sidecar = SteeringVectorSidecar(
            modelID: modelID,
            revision: SteeredContainerLoader.cachedRevision(for: modelID),
            concept: name,
            stimulusSetHash: target.hash, vectors: extraction.vectors,
            options: extraction.options,
            residualNormPerLayer: extraction.residualNormPerLayer,
            residualNormSource: normSource,
            residualNormConvention: extraction.residualNormConvention,
            recipeMethod: recipe.method,
            recipeHash: recipe.canonicalHash(),
            recipeName: recipe.name)
        sidecar.designatedReference = ["name": reference, "hash": referenceCorpus.hash]
        let artifactName = "\(name)-\(host.selectedModel.family)"
        try SteeringVectorStore.save(
            vectors: extraction.vectors, sidecar: sidecar, to: runDirectory,
            name: artifactName,
            neutralMeanPerLayer: extraction.neutralMeanPerLayer)

        refreshConceptList()
        if name == currentName { selectedExisting = name }
        host.refreshVectors()
        let artifactID = runDirectory.appending(component: artifactName).path
        host.selectVector(artifactID)
        host.steeringEnabled = true
        status = "saved \(name) (\(target.rows.count) stories − "
            + "\(referenceCorpus.rows.count) reference '\(reference)') "
            + "→ \(runDirectory.lastPathComponent); selected for steering"
        refreshStaleness()
    }

    private func saveEmotionGrandMeanConcept(
        name: String, container: ModelContainer, modelID: String, host: ChatService
    ) async throws {
        let diskRows = (try? loadEmotionRows(for: name)) ?? []
        if multiConceptRows.isEmpty, !diskRows.isEmpty {
            multiConceptRows = diskRows
        }
        let extractionRows = emotionRowsForExtraction()
        let requestedTargets = grandMeanTargetConceptNames
        let availableTargets = Set(extractionRows.map(\.concept))
        let buildTargets = requestedTargets.intersection(availableTargets).sorted()
        let skippedTargets = requestedTargets.subtracting(availableTargets).sorted()
        guard extractionRows.count >= 4 else {
            status = "need at least 4 build rows in the selected topics before extracting"
            return
        }
        guard Set(extractionRows.map(\.concept)).count >= 2 else {
            status = "emotion grand-mean needs at least two build concepts in the selected topics; otherwise target mean equals grand mean"
            return
        }
        guard !buildTargets.isEmpty else {
            let counts = counts(extractionRows.map(\.concept))
                .map { "\($0.name):\($0.count)" }
                .joined(separator: ", ")
            let targetDescription = requestedTargets.sorted().joined(separator: ", ")
            status = "selected build rows must include stories for at least one target concept"
                + (targetDescription.isEmpty ? "" : " (\(targetDescription))")
                + (counts.isEmpty ? "" : "; selected rows contain \(counts)")
            return
        }
        let selectedCorpusText = multiConceptJSONL(extractionRows)
        let selectedCorpusHash = Self.sha256(selectedCorpusText)

        status = buildTargets.count == 1
            ? "extracting grand-mean vector…"
            : "extracting \(buildTargets.count) grand-mean vectors…"
        let corpusURL = VectorCatalog.projectRoot.appending(
            components: "prompts", "neutral", "corpus.jsonl")
        let corpus = try? StimulusSet.loadTexts(url: corpusURL)
        let extraction = try await ConceptExtractor.extractGrandMean(
            container: container,
            corpus: extractionRows,
            targetConcepts: Set(buildTargets),
            readingPosition: extractionOptions.readingPosition,
            neutralTexts: corpus?.texts,
            extractionRendering: extractionOptions.resolvedExtractionRendering)

        let runSlug = buildTargets.count == 1
            ? "concept-\(buildTargets[0])"
            : "grand-mean-\(host.selectedModel.family)"
        let runDirectory = try VectorCatalog.makeUniqueRunDirectory(slug: runSlug)
        try RunMetadata.write(
            runType: "extract", to: runDirectory,
            modelID: modelID,
            revision: SteeredContainerLoader.cachedRevision(for: modelID))
        let normSource =
            extraction.residualNormSource == "neutral-corpus"
            ? "neutral-corpus \(corpus?.hash.prefix(12) ?? "")"
            : extraction.residualNormSource
        var savedArtifactIDs: [(concept: String, id: String)] = []
        for target in buildTargets {
            guard let vectors = extraction.vectorsByConcept[target] else { continue }
            let recipe = VectorExtractionRecipe(
                name: "\(target)-grand-mean",
                method: .emotionGrandMean,
                targetConcept: target,
                datasets: [
                    .init(
                        role: .multiConceptStories,
                        path: "prompts/emotions/*/stories.jsonl",
                        sha256: selectedCorpusHash)
                ],
                readingPosition: extraction.readingPosition,
                neutralPCSelection: .none,
                extractionRendering: extractionRendering,
                notes: "Vector = mean(target concept build stories) - grand mean(selected build story rows). Validation/draft rows are saved in per-concept corpora but excluded from extraction. Concepts: \(selectedEmotionConceptDescription()). Topics: \(selectedEmotionTopicDescription()).")
            let sidecar = SteeringVectorSidecar(
                modelID: modelID,
                revision: SteeredContainerLoader.cachedRevision(for: modelID),
                concept: target,
                stimulusSetHash: selectedCorpusHash,
                vectors: vectors,
                options: nil,
                residualNormPerLayer: extraction.residualNormPerLayer,
                residualNormSource: normSource,
                residualNormConvention: extraction.residualNormConvention,
                extractionRendering: extractionRendering,
                recipeMethod: recipe.method,
                recipeHash: recipe.canonicalHash(),
                recipeName: recipe.name,
                appliedReadingPosition: extraction.readingPosition,
                neutralProjection: extraction.neutralProjectionDescription,
                neutralCorpusHash: corpus?.hash,
                screening: extraction.screening,
                neutralScreening: extraction.neutralScreening,
                comparisonConcepts: selectedEmotionConceptNames().sorted(),
                selectedTopics: includedEmotionTopics.isEmpty
                    ? Array(Set(extractionRows.map { normalizedTopic($0.topic) })).sorted()
                    : includedEmotionTopics.sorted(),
                selectedSplits: ["build"])
            let artifactName = "\(target)-\(host.selectedModel.family)"
            try SteeringVectorStore.save(
                vectors: vectors, sidecar: sidecar, to: runDirectory, name: artifactName,
                neutralMeanPerLayer: extraction.neutralMeanPerLayer)
            savedArtifactIDs.append(
                (target, runDirectory.appending(component: artifactName).path))
        }
        guard !savedArtifactIDs.isEmpty else {
            status = "no grand-mean vectors were produced for the selected concepts"
            return
        }
        refreshConceptList()
        if savedArtifactIDs.contains(where: { $0.concept == name }) {
            selectedExisting = name
        } else if let first = savedArtifactIDs.first?.concept {
            selectedExisting = first
        }
        host.refreshVectors()
        let selectedArtifact = savedArtifactIDs.first(where: { $0.concept == name })
            ?? savedArtifactIDs.first
        if let selectedArtifact {
            host.selectVector(selectedArtifact.id)
        }
        host.steeringEnabled = true
        let excluded = extraction.screening.excludedShortCount
        let neutralNote: String
        if let neutral = extraction.neutralScreening, neutral.excludedShortCount > 0 {
            neutralNote =
                "; neutral norm corpus kept \(neutral.includedCount)/\(neutral.sourceCount) "
                + "rows, using \(extraction.residualNormSource) norms"
        } else {
            neutralNote = ""
        }
        let skippedNote = skippedTargets.isEmpty
            ? ""
            : "; skipped missing targets \(skippedTargets.joined(separator: ", "))"
        let savedConceptList = savedArtifactIDs.map { $0.concept }.joined(separator: ", ")
        let vectorNoun = savedArtifactIDs.count == 1 ? "vector" : "vectors"
        let screeningNote = "(\(extraction.screening.includedCount)/\(extraction.screening.sourceCount) selected build rows"
            + (excluded > 0 ? ", excluded \(excluded) short" : "")
            + ")"
        status = "saved \(savedArtifactIDs.count) grand-mean \(vectorNoun) "
            + "(\(savedConceptList)) \(screeningNote) → "
            + "\(runDirectory.lastPathComponent); selected for steering"
            + neutralNote + skippedNote
        refreshStaleness()
    }

    private func emotionRowsForExtraction() -> [StimulusSet.MultiConceptStimulus] {
        let selectedTopics = includedEmotionTopics
        return emotionRowsForSelectedConcepts().filter { row in
            Self.canonicalSplit(row.split) == "build"
                && (selectedTopics.isEmpty || selectedTopics.contains(normalizedTopic(row.topic)))
        }
    }

    private func emotionRowsForSelectedConcepts() -> [StimulusSet.MultiConceptStimulus] {
        var rows: [StimulusSet.MultiConceptStimulus] = []
        for concept in selectedEmotionConceptNames().sorted() {
            if concept == currentName {
                rows.append(contentsOf: (try? loadEmotionRows(for: concept)) ?? [])
                rows.append(contentsOf: multiConceptRows)
            } else {
                rows.append(contentsOf: (try? loadEmotionRows(for: concept)) ?? [])
            }
        }
        var seen: Set<String> = []
        return rows.filter { row in
            let key = Self.rowIdentity(row)
            guard !seen.contains(key) else { return false }
            seen.insert(key)
            return true
        }
    }

    private func selectedEmotionConceptNames() -> Set<String> {
        let options = Set(emotionConceptOptions)
        let selected = includedEmotionConcepts.isEmpty ? options : includedEmotionConcepts
        let names = selected.map(Self.sanitizedName).filter { !$0.isEmpty }
        return Set(names)
    }

    private var grandMeanTargetConceptNames: Set<String> {
        let included = selectedEmotionConceptNames()
        if grandMeanBuildConceptsAreExplicit {
            return grandMeanBuildConcepts.intersection(included)
        }
        return included
    }

    private func selectedEmotionTopicDescription() -> String {
        includedEmotionTopics.isEmpty
            ? "all build topics"
            : includedEmotionTopics.sorted().joined(separator: ", ")
    }

    private func selectedEmotionConceptDescription() -> String {
        includedEmotionConcepts.isEmpty
            ? "all available story concepts"
            : selectedEmotionConceptNames().sorted().joined(separator: ", ")
    }

    private func sanitized(_ name: String) -> String {
        Self.sanitizedName(name)
    }

    public nonisolated static func sanitizedName(_ name: String) -> String {
        name.lowercased()
            .replacingOccurrences(of: " ", with: "-")
            .filter { $0.isLetter || $0.isNumber || $0 == "-" }
    }

    private func jsonl(_ texts: [String]) -> String {
        texts.compactMap { text -> String? in
            guard let data = try? JSONEncoder().encode(["text": text]) else { return nil }
            return String(decoding: data, as: UTF8.self)
        }
        .joined(separator: "\n") + "\n"
    }

    private func pairJSONL(concept: String, positives: [String], negatives: [String]) -> String {
        zip(positives, negatives).enumerated().compactMap { index, pair -> String? in
            let row = StimulusSet.PairedStimulus(
                id: "\(concept)-pair-\(index + 1)",
                positive: pair.0,
                negative: pair.1,
                split: index.isMultiple(of: 5) ? "validation" : "build")
            guard let data = try? JSONEncoder().encode(row) else { return nil }
            return String(decoding: data, as: UTF8.self)
        }
        .joined(separator: "\n") + "\n"
    }

    private func multiConceptJSONL(_ rows: [StimulusSet.MultiConceptStimulus]) -> String {
        rows.compactMap { row -> String? in
            let row = Self.canonicalized(row)
            guard let data = try? JSONEncoder().encode(row) else { return nil }
            return String(decoding: data, as: UTF8.self)
        }
        .joined(separator: "\n") + "\n"
    }

    private func probeJSONL(_ examples: [ProbeExample]) -> String {
        examples.compactMap { example -> String? in
            guard let data = try? JSONEncoder().encode(example) else { return nil }
            return String(decoding: data, as: UTF8.self)
        }
        .joined(separator: "\n") + "\n"
    }

    private nonisolated static func sha256(_ text: String) -> String {
        let digest = SHA256.hash(data: Data(text.utf8))
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    private struct NeutralTextRow: Codable {
        let text: String
    }

    private nonisolated static func neutralCorpusURL() -> URL {
        VectorCatalog.projectRoot.appending(
            components: "prompts", "neutral", "corpus.jsonl")
    }

    nonisolated static func parseNeutralTexts(_ data: Data) throws -> [String] {
        let trimmed = String(decoding: data, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        var texts: [String] = []
        if trimmed.first == "[" {
            let object = try JSONSerialization.jsonObject(with: Data(trimmed.utf8))
            if let strings = object as? [String] {
                texts.append(contentsOf: strings)
            } else if let rows = object as? [[String: Any]] {
                texts.append(contentsOf: rows.compactMap { $0["text"] as? String })
            }
        } else {
            var decodedJSONL = 0
            for line in trimmed.split(whereSeparator: \.isNewline) {
                let raw = String(line).trimmingCharacters(in: .whitespacesAndNewlines)
                guard !raw.isEmpty, raw.first == "{" else { continue }
                let row = try JSONDecoder().decode(
                    NeutralTextRow.self,
                    from: Data(raw.utf8))
                texts.append(row.text)
                decodedJSONL += 1
            }
            if decodedJSONL == 0 {
                texts.append(
                    contentsOf: trimmed.components(separatedBy: "\n\n")
                        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) })
            }
        }

        var seen: Set<String> = []
        return texts.compactMap { text -> String? in
            let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !clean.isEmpty, !seen.contains(clean) else { return nil }
            seen.insert(clean)
            return clean
        }
    }

    nonisolated static func parseMultiConceptRows(
        _ data: Data, filename: String, defaultConcept: String? = nil,
        defaultTopic: String? = nil,
        defaultSplit: String = "build"
    ) throws -> [StimulusSet.MultiConceptStimulus] {
        let text = String(decoding: data, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { throw StimulusSetError.empty(filename) }
        if text.hasPrefix("[") {
            return try JSONDecoder().decode(
                [StimulusSet.MultiConceptStimulus].self, from: Data(text.utf8))
                .map(canonicalized)
        }
        if text.hasPrefix("{") {
            return try text.split(separator: "\n").enumerated().compactMap { index, line in
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard !trimmed.isEmpty else { return nil }
                guard
                    let row = try? JSONDecoder().decode(
                        StimulusSet.MultiConceptStimulus.self, from: Data(trimmed.utf8))
                else {
                    throw StimulusSetError.malformedLine(file: filename, line: index + 1)
                }
                return canonicalized(row)
            }
        }
        guard let defaultConcept, !defaultConcept.isEmpty else {
            throw StimulusSetError.malformedLine(file: filename, line: 1)
        }
        let chunks = text.components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        let stories = chunks.isEmpty
            ? text.split(separator: "\n").map {
                $0.trimmingCharacters(in: .whitespacesAndNewlines)
            }.filter { !$0.isEmpty }
            : chunks
        return stories.enumerated().map { index, story in
            StimulusSet.MultiConceptStimulus(
                id: "\(defaultConcept)-manual-\(index + 1)",
                concept: defaultConcept,
                topic: defaultTopic,
                text: story,
                split: canonicalSplit(defaultSplit),
                source: "manual")
        }
    }

    private nonisolated static func canonicalized(
        _ row: StimulusSet.MultiConceptStimulus
    ) -> StimulusSet.MultiConceptStimulus {
        StimulusSet.MultiConceptStimulus(
            id: row.id,
            concept: sanitizedName(row.concept),
            topic: row.topic,
            text: row.text,
            split: canonicalSplit(row.split),
            source: row.source,
            notes: row.notes)
    }

    /// Renders a prompt template resolved by the one shared seed rule
    /// (`DataTemplates.workspaceOrSeedURL`): the workspace's own copy
    /// when the workspace was seeded with `prompts/generation/`, else the
    /// code-shipped seed copy. Returns nil when neither is readable — e.g. a
    /// packaged/relocated app plus a template-less workspace — so callers
    /// route the failure path; a sentinel string must never reach the
    /// clipboard under a success message.
    private static func templatePrompt(
        filename: String, replacements: [String: String]
    ) -> String? {
        let url = DataTemplates.workspaceOrSeedURL(
            relativePath: "prompts/generation/\(filename)",
            workspaceRoot: VectorCatalog.projectRoot,
            seedRoot: seedRootOverrideForTesting)
        guard var text = try? String(contentsOf: url, encoding: .utf8) else {
            return nil
        }
        for (key, value) in replacements {
            text = text.replacingOccurrences(of: "{{\(key)}}", with: value)
        }
        return text
    }
}
