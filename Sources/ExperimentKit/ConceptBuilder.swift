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
        case repeLAT
        case repeReaderLAT
        case emotionGrandMean
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
            case .repeLAT: "LAT paired direction (RepE-inspired)"
            case .repeReaderLAT: "RepE reader LAT"
            case .emotionGrandMean: "Grand mean"
            case .jlensTokenDirection: "J-lens token direction"
            }
        }

        public var recipeMethod: VectorExtractionRecipe.Method {
            switch self {
            case .caaMeanDifference: .caaMeanDifference
            case .repeLAT: .repeLAT
            case .repeReaderLAT: .repeReaderLAT
            case .emotionGrandMean: .emotionGrandMean
            case .jlensTokenDirection: .jlensTokenDirection
            }
        }

        public var isPaired: Bool {
            switch self {
            case .caaMeanDifference, .repeLAT, .repeReaderLAT: true
            case .emotionGrandMean, .jlensTokenDirection: false
            }
        }

        /// True when the family reads activations from the loaded model over a
        /// stimulus set. False for a DERIVED direction, whose pane hides the
        /// stimulus, pooling, and reading-position controls because none of
        /// them apply — and would silently imply provenance it does not have.
        public var extractsFromStimuli: Bool {
            switch self {
            case .caaMeanDifference, .repeLAT, .repeReaderLAT, .emotionGrandMean:
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
            case .repeLAT:
                [
                    "LAT paired direction is the RepE-inspired PCA direction over matched positive-minus-negative activation differences.",
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
            case .repeLAT:
                extractionMethod = .lat
                if poolFromToken == 50 { poolFromToken = nil }
            case .repeReaderLAT:
                // Reader fit is template-rendered LAT at the scaffold's final
                // token; the pooled reading position does not apply.
                extractionMethod = .lat
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
            if !syncingRecipeFamily {
                recipeFamily = extractionMethod == .lat ? .repeLAT : .caaMeanDifference
            }
            lastDirection = nil
            noteMutation()
        }
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
    static func renderingLabel(_ rendering: ExtractionRendering?) -> String {
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
                recipeFamily = Self.pairedRecipeFamilyOnDisk(for: name)
                notes.append("\(set.positive.count)+\(set.negative.count) stimuli")
            }
        } else {
            positives = []
            negatives = []
        }
        multiConceptRows = (try? loadEmotionRows(for: name)) ?? []
        if !multiConceptRows.isEmpty {
            if positives.isEmpty, negatives.isEmpty {
                recipeFamily = .emotionGrandMean
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

    /// Which paired recipe produced a concept's on-disk dataset. The shared
    /// paired files (`prompts/concepts/<name>/{positive,negative}.jsonl`)
    /// cannot distinguish CAA from RepE rows by shape, but the
    /// recipe-specific mirrors can: a RepE-LAT vector build also writes
    /// `prompts/repe/<name>/pairs.jsonl`, and a reader fit writes
    /// `prompts/readers/<name>/pairs.jsonl`. Selecting an existing concept
    /// must restore ITS recipe — before 2026-07-14 this defaulted the picker
    /// (and therefore "Copy LLM prompt") back to CAA for every paired
    /// concept, so a RepE concept quietly copied the CAA dataset prompt.
    /// When both mirrors exist, the most recently written one wins (the
    /// user's latest build under that concept); with neither, plain paired
    /// data is CAA.
    nonisolated static func pairedRecipeFamilyOnDisk(for name: String) -> RecipeFamily {
        func modificationDate(_ url: URL) -> Date? {
            (try? FileManager.default.attributesOfItem(atPath: url.path))?[
                .modificationDate] as? Date
        }
        let repePairs = modificationDate(
            VectorCatalog.pairedStimuliFile(family: .repe, name: name))
        let readerPairs = modificationDate(
            VectorCatalog.pairedStimuliFile(family: .readers, name: name))
        switch (repePairs, readerPairs) {
        case (nil, nil): return .caaMeanDifference
        case (.some, nil): return .repeLAT
        case (nil, .some): return .repeReaderLAT
        case let (.some(repe), .some(reader)):
            return reader > repe ? .repeReaderLAT : .repeLAT
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
        case .repeLAT, .repeReaderLAT:
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
        case .emotionGrandMean:
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
        if recipeFamily == .emotionGrandMean {
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

        do {
            status = "recording probe activations..."
            let positive = try await ConceptExtractor.activations(
                container: container, texts: positives, position: .lastToken)
            let negative = try await ConceptExtractor.activations(
                container: container, texts: negatives, position: .lastToken)
            let validationPositive = validationPositives.isEmpty
                ? nil
                : try await ConceptExtractor.activations(
                    container: container, texts: validationPositives, position: .lastToken)
            let validationNegative = validationNegatives.isEmpty
                ? nil
                : try await ConceptExtractor.activations(
                    container: container, texts: validationNegatives, position: .lastToken)
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
                let direction = try SteeringVectorMath.meanDifference(positive: pos, negative: neg)
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
                method: .caaMeanDifference,
                targetConcept: name,
                datasets: [
                    .init(
                        role: .probeCalibration,
                        path: "prompts/probes/\(name)/items.jsonl",
                        sha256: calibrationHash)
                ],
                readingPosition: .lastToken,
                notes: "Scalar reading probe trained from labeled build examples. Layer selected by held-out validation accuracy across all layers.")
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
            status = "live pair-margin stats are for CAA/RepE paired data; save to extract the emotion grand-mean vector"
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
                let plan = Self.grandMeanServerSyncPlan(
                    included: included, targets: targets,
                    localStoryConcepts: localStoryConceptNames())
                var mustExistOnServer = plan.requireOnServer
                for concept in plan.push {
                    let rows = (try? loadEmotionRows(for: concept)) ?? []
                    guard !rows.isEmpty else {
                        // Nothing to push — never overwrite server stories
                        // with an empty file; require the server's copy.
                        mustExistOnServer.append(concept)
                        continue
                    }
                    status = "syncing '\(concept)' stories to the server…"
                    try await client.saveStories(concept: concept, rows: rows)
                }
                if !mustExistOnServer.isEmpty {
                    let listing = Set(try await client.storyConcepts())
                    let missing = mustExistOnServer.filter { !listing.contains($0) }
                        .sorted()
                    if !missing.isEmpty {
                        reportBuildFailure(
                            "cannot queue grand-mean build: the server has "
                                + "no stories for \(missing.joined(separator: ", ")) "
                                + "(prompts/emotions/<concept>/stories.jsonl on the "
                                + "server's tree) and there are none locally to sync",
                            title: "Server Grand Mean Vector Build — failed")
                        return
                    }
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
            case .caaMeanDifference, .repeLAT:
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

    // MARK: - RepE reader (faithful reader recipe; brief §8)
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

    public struct ReaderLayerScore: Identifiable, Sendable {
        public let layer: Int
        public let trainAccuracy: Float
        public let heldOutAccuracy: Float?
        public let explainedVariance: Float
        public var id: Int { layer }
    }

    /// Per-layer train/held-out accuracy from the newest local reader fit.
    public private(set) var readerLayerScores: [ReaderLayerScore] = []
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
    /// pairs rows carry the same ids, split assignment, and sorted-keys
    /// encoding as the local pairs file.
    nonisolated static func readerFitRequest(
        concept: String,
        positives: [String],
        negatives: [String],
        heldOutPairCount: Int,
        registryTemplateID: String?,
        customTemplateText: String?,
        modelID: String? = nil
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
        let lines = try readerPairRows(
            concept: concept, positives: positives, negatives: negatives,
            heldOutPairCount: heldOutPairCount, templateID: pinnedTemplateID)
        return ReaderFitRequest(
            concept: concept, modelID: modelID,
            templateID: templateID, templateJSON: templateJSON,
            pairsJSONL: lines.joined(separator: "\n") + "\n")
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
        let pairedRows: (positive: [String], negative: [String])
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
        let heldOut = min(max(0, readerHeldOutPairCount), max(0, pairedRows.positive.count - 2))
        guard pairedRows.positive.count - heldOut >= 2 else {
            status = "need at least 2 train pairs (have \(pairedRows.positive.count) pairs, "
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
                modelID: host.selectedRemoteModelID)

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
                pairsJSONL: request.pairsJSONL)
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
        let pairedRows: (positive: [String], negative: [String])
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
        let heldOut = min(max(0, readerHeldOutPairCount), max(0, pairedRows.positive.count - 2))
        guard pairedRows.positive.count - heldOut >= 2 else {
            status = "need at least 2 train pairs (have \(pairedRows.positive.count) pairs, "
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
            let lines = try Self.readerPairRows(
                concept: name,
                positives: pairedRows.positive,
                negatives: pairedRows.negative,
                heldOutPairCount: readerHeldOutPairCount, templateID: template.id)
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
                template: template)
            for artifact in artifacts {
                try RepEReader.saveArtifact(artifact, to: runDirectory)
            }
            readerLayerScores = artifacts.map {
                ReaderLayerScore(
                    layer: $0.layer,
                    trainAccuracy: $0.trainAccuracy,
                    heldOutAccuracy: $0.heldOutAccuracy,
                    explainedVariance: $0.pc1ExplainedVariance)
            }
            lastReaderRunDirectory = runDirectory
            refreshReaderArtifacts()
            let best = artifacts.max {
                ($0.heldOutAccuracy ?? $0.trainAccuracy)
                    < ($1.heldOutAccuracy ?? $1.trainAccuracy)
            }
            let bestNote = best.map {
                String(
                    format: "best layer %d (%@ %.0f%%)", $0.layer,
                    $0.heldOutAccuracy == nil ? "train" : "held-out",
                    ($0.heldOutAccuracy ?? $0.trainAccuracy) * 100)
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

    /// Explicit reader → steering-vector conversion (brief §6). The derived
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
            host?.refreshVectors()
            host?.selectVector(base.path)
            status = "derived steering vector from reader "
                + "'\(record.fileName)' → \(runDirectory.lastPathComponent) "
                + "(controlMode: \(RepEReader.controlMode)); selected for steering"
        } catch {
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
        case .caaMeanDifference, .repeLAT:
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
            if recipeFamily == .repeLAT {
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
                        role: recipeFamily == .repeLAT ? .pairedContrastive : .positive,
                        path: canonicalDatasetPath,
                        sha256: canonicalDatasetHash)
                ],
                readingPosition: extraction.options.readingPosition,
                neutralPCSelection: .none,
                // nil for the legacy raw rendering, which keeps the recipe's
                // encoded bytes — and therefore its canonicalHash — exactly
                // where they were.
                extractionRendering: extraction.options.extractionRendering,
                notes: recipeFamily == .repeLAT
                    ? "RepE/LAT direction is norm-matched to the CAA mean-difference vector for direction-only comparison."
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
