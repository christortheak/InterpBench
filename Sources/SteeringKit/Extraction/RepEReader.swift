import CryptoKit
import Foundation
import MLXLMCommon

/// Template-mediated RepE reader LAT (Zou et al., arXiv:2310.01405 §3.1,
/// App. C.1) — the Swift twin of
/// `Server/steerlab_server/steering/repe_reader.py`, which is the schema/math
/// source of truth. A reader is a fitted **measurement instrument**, not a
/// steering vector: task template + LAT token position + PCA direction with
/// training normalization + held-out sign/layer selection + held-out scalar
/// accuracy (see `docs/REPE-IMPLEMENTATION-BRIEF.md`). Pipeline:
///
///     stimulus → render task template → capture hidden state at the LAT token
///     → per-pair differences → centered PCA → PC1 signed on the HELD-OUT
///     split → ScalarProbe fitted on the TRAIN activations
///
/// Inference renders the *same* template under the *same* rendering, captures
/// the *same* token position, and scores through the stored probe (training
/// center/scale) — never a raw cosine-to-vector shortcut. A reader can
/// *derive* a steering vector (`deriveSteeringVector`), but the derived
/// artifact stamps its reader provenance so "reading-vector activation
/// addition" is never conflated with the paper's full control experiments.
///
/// **Faithful, and where it still departs.** Implemented from the paper and
/// the reference implementation: the task template (§3.1) with its LAT token
/// at the rendered scaffold's final position (`rep_token=-1`); PCA over paired
/// differences with `n_components=1` and mean-centering only
/// (`repe/rep_readers.py`, `PCARepReader`); the difference construction
/// `[::2] − [1::2]` over per-pair randomized orientation
/// (`rep_reading_pipeline.py` + the dataset builder's `random.shuffle(d)`);
/// BOTH contrast constructions — the supervised content contrast and the
/// paper's unsupervised T+/T− instruction pair (§3.1 step 1b); held-out sign
/// AND layer selection (the paper's step 4, e.g. its 25 ARC-Challenge
/// validation examples); and either rendering — raw scaffold or the family
/// chat template, the repo's `user_tag`/`assistant_tag` analogue.
///
/// The departures that REMAIN, each deliberate and each stamped:
///
/// - **`latToken` supports only `"final"`.** The paper reads other positions
///   in some experiments; the registry schema carries the field so a second
///   position is a data change, but only `final` is implemented, and any other
///   value is refused rather than silently coerced.
/// - **Scoring is a calibrated scalar probe, not the paper's logistic /
///   `pca_model.transform` pipeline.** The center and scale come from the
///   train projections and are persisted, which reproduces the paper's
///   "normalize test activations with the training parameters"; the classifier
///   on top is our midpoint rule.
/// - **Reader artifacts are substrate-specific by rule.** Activations do not
///   transfer between MLX and PyTorch, so a reader fitted on one engine is
///   readable but never usable on the other. The paper has one substrate.
/// - **Datasets are authored, not borrowed.** The paper fits on published
///   corpora (TQA, ARC); a study here fits on its own pinned pairs.
///
/// Concept-agnostic by design: concepts, templates, and stimuli enter as data
/// (`prompts/templates/`, reader pairs JSONL). The shared inputs are the
/// template/dataset bytes, whose SHA-256 hashes are identical across engines.
public enum RepEReader {

    public static let artifactType = "repe-reader-lat"
    /// This engine's substrate stamp (the server's is
    /// "python-hf-transformers"). Scoring and manifest verification refuse
    /// readers fitted elsewhere.
    public static let substrate = "swift-mlx"
    /// Honest control-mode label stamped on derived steering vectors.
    public static let controlMode = "reading-vector activation addition"
    /// Rendering contract stamped into a RAW-rendered artifact (server twin:
    /// `READER_RENDERING_CONVENTION`; identical semantics, this engine's
    /// extraction path named). Byte-frozen: it is the value every reader
    /// artifact written before the rendering became declarable carries, and
    /// an absent `extractionRendering` means exactly this.
    public static let renderingConvention =
        "rawCompletion scaffold: no chat template, no system role, no family "
        + "thinking suffix; tokenized by the extraction path "
        + "(ConceptExtractor.activations) with the tokenizer's default special "
        + "tokens — single BOS added by the tokenizer, LAT token = final "
        + "scaffold token"
    /// Rendering contract stamped into a CHAT-TEMPLATE-rendered artifact
    /// (server twin: `READER_CHAT_TEMPLATE_RENDERING_CONVENTION`). This is the
    /// repo's `user_tag`/`assistant_tag` construction: the scaffold is the user
    /// turn's content and the family template supplies every special token,
    /// so the final token is the generation prompt's tail — exactly what
    /// `rep_token=-1` reads in `f"{user_tag} {instruction} {scenario}
    /// {assistant_tag}"`.
    public static let chatTemplateRenderingConvention =
        "chatTemplate scaffold: the rendered scaffold is the USER TURN's "
        + "content and the family chat template supplies every special token "
        + "(single BOS, turn markers, generation prompt); tokenized by the "
        + "extraction path (ConceptExtractor.activations) through "
        + "PromptRendering — LAT token = final token of the generation prompt, "
        + "the repo's assistant_tag position"

    /// The rendering convention string for a declared rendering. Raw resolves
    /// to the byte-frozen legacy constant so a raw fit is byte-identical to
    /// every reader written before the option existed.
    public static func renderingConvention(
        for rendering: ExtractionRendering?
    ) -> String {
        guard let rendering, !rendering.isRaw else { return renderingConvention }
        return chatTemplateRenderingConvention
    }

    public struct ReaderError: Error, CustomStringConvertible, Equatable {
        public let reason: String
        public var description: String { reason }

        public init(reason: String) {
            self.reason = reason
        }
    }

    static func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    // MARK: - Contrast construction (paper §3.1 steps 1a / 1b)

    /// WHAT the two activations in a pair differ by. Both are first-class:
    /// the paper describes both, and neither is a degraded form of the other.
    /// Server twin: `repe_reader.CONTRAST_MODES`.
    public enum ContrastMode: String, Codable, Sendable, CaseIterable {
        /// Two DIFFERENT stimuli under ONE template — the concept-present
        /// stimulus against its matched control. This is what every reader
        /// fitted before 2026-08-27 did, so an absent stamp means this.
        case supervisedContent
        /// ONE stimulus under TWO templates — the paper's experimental /
        /// reference instruction pair (T+ / T−, e.g. "pretend you're an honest
        /// person…" against "pretend you're a dishonest person…"). The
        /// stimulus is held fixed and the INSTRUCTION carries the contrast, so
        /// the direction cannot be a content artifact of two different texts.
        case unsupervisedTemplatePair

        public var label: String {
            switch self {
            case .supervisedContent: "supervised content contrast"
            case .unsupervisedTemplatePair: "unsupervised template pair (T+/T−)"
            }
        }
    }

    /// HOW a layer's PC1 sign was fixed. Server twin:
    /// `repe_reader.SIGN_CONVENTIONS`.
    public enum SignConvention: String, Codable, Sendable, CaseIterable {
        /// The paper's step 4: the held-out split decides. The sign is the one
        /// under which held-out (positive − negative) differences project
        /// positively — held-out classification accuracy on the paired
        /// discrimination task, maximized over the two available signs.
        case heldOutPairAgreement
        /// `get_signs`' train-label agreement, which is what the reference
        /// implementation actually ships and what every reader fitted before
        /// 2026-08-27 used. An absent stamp means this.
        case trainMajority

        public var label: String {
            switch self {
            case .heldOutPairAgreement: "held-out pair agreement (paper step 4)"
            case .trainMajority: "train-label majority (reference get_signs)"
            }
        }
    }

    /// The fewest held-out pairs that may decide a sign. Below it the held-out
    /// vote is one or two coin flips wearing the authority of a validation
    /// split, so the fit falls back to train-majority AND says so in the
    /// artifact (`signFallbackReason`) rather than quietly pretending.
    public static let minimumHeldOutPairsForSignSelection = 2

    /// Default seed for the unsupervised mode's per-row orientation draw.
    /// Deterministic and stamped: the reference implementation shuffles each
    /// pair with an unseeded `random.shuffle`, which makes its direction
    /// irreproducible; we keep the paper's ± symmetry and drop its
    /// irreproducibility. `2310_01405` is the paper's arXiv id.
    public static let defaultOrientationSeed: UInt64 = 231_001_405

    /// SplitMix64 — the shared cross-engine PRNG for the orientation draw.
    /// Chosen because it is four lines in both languages and produces
    /// bit-identical streams, so an unsupervised fit is reproducible across
    /// substrates from the seed alone. Server twin: `repe_reader._split_mix64`.
    static func splitMix64(_ state: inout UInt64) -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// `count` orientations in ±1, drawn in row order from `seed`. Server
    /// twin: `repe_reader.orientation_signs`.
    public static func orientationSigns(count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0 ..< max(0, count)).map { _ in
            splitMix64(&state) & 1 == 0 ? 1 : -1
        }
    }

    // MARK: - Task templates (prompts/templates/<id>.json — shared data files)

    /// One registry entry from `prompts/templates/<id>.json`.
    ///
    /// `hash` is the SHA-256 of the file's raw bytes (stimulus-set
    /// convention): changing the template changes every artifact fitted
    /// through it. `divergence` marks deliberate departures from the paper
    /// (e.g. the unnamed clean-room scaffold, which never names the concept).
    /// JSON keys match the server's `TaskTemplate.to_dict` exactly.
    public struct TaskTemplate: Codable, Sendable, Equatable {

        /// The paper's T+ / T− instruction pair (§3.1 step 1b). A template
        /// that declares one is a TEMPLATE-PAIR template: the same stimulus is
        /// rendered twice, once under each instruction, and the difference is
        /// H(T+) − H(T−). Both strings substitute into the template's
        /// `{{instruction}}` slot.
        ///
        /// Hygiene rules apply exactly as they do to a scaffold: an
        /// instruction that names the study concept makes the reader a
        /// concept-word detector, so the shipped example names none.
        public struct InstructionPair: Codable, Sendable, Equatable {
            /// T+ — the instruction under which the concept is present.
            public var experimental: String
            /// T− — the matched reference instruction.
            public var reference: String

            public init(experimental: String, reference: String) {
                self.experimental = experimental
                self.reference = reference
            }
        }

        public var id: String
        public var conceptSlot: Bool
        public var text: String
        public var latToken: String
        /// Not present in the template *file*; carried in embedded copies.
        public var hash: String
        public var divergence: String?
        /// Absent for a single-template (supervised content) template; present
        /// for a T+/T− template-pair template. Absent is the legacy shape, so
        /// every template file written before 2026-08-27 keeps its hash.
        public var instructionPair: InstructionPair?

        /// The instruction slot a template-pair template must carry.
        public static let instructionSlot = "{{instruction}}"

        enum CodingKeys: String, CodingKey {
            case id, conceptSlot, text, latToken, hash, divergence, instructionPair
        }

        public init(
            id: String, conceptSlot: Bool, text: String,
            latToken: String = "final", hash: String = "",
            divergence: String? = nil,
            instructionPair: InstructionPair? = nil
        ) {
            self.id = id
            self.conceptSlot = conceptSlot
            self.text = text
            self.latToken = latToken
            self.hash = hash
            self.divergence = divergence
            self.instructionPair = instructionPair
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            id = try container.decode(String.self, forKey: .id)
            conceptSlot = try container.decodeIfPresent(Bool.self, forKey: .conceptSlot) ?? false
            text = try container.decode(String.self, forKey: .text)
            latToken = try container.decodeIfPresent(String.self, forKey: .latToken) ?? "final"
            hash = try container.decodeIfPresent(String.self, forKey: .hash) ?? ""
            divergence = try container.decodeIfPresent(String.self, forKey: .divergence)
            instructionPair = try container.decodeIfPresent(
                InstructionPair.self, forKey: .instructionPair)
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(id, forKey: .id)
            try container.encode(conceptSlot, forKey: .conceptSlot)
            try container.encode(text, forKey: .text)
            try container.encode(latToken, forKey: .latToken)
            try container.encode(hash, forKey: .hash)
            try container.encodeIfPresent(divergence, forKey: .divergence)
            try container.encodeIfPresent(instructionPair, forKey: .instructionPair)
        }

        /// Whether this template carries the paper's T+/T− instruction pair.
        public var isTemplatePair: Bool { instructionPair != nil }

        public func readingPosition() throws -> ReadingPosition {
            guard latToken == "final" else {
                throw ReaderError(
                    reason: "template '\(id)': unsupported latToken '\(latToken)' "
                        + "(only 'final' is implemented)")
            }
            return .lastToken
        }

        /// Schema coherence between the instruction pair and the slot. Called
        /// by `loadTemplate` so a malformed registry file is refused at load
        /// rather than producing a scaffold with a literal `{{instruction}}`
        /// in it.
        public func validateInstructionSlot() throws {
            let hasSlot = text.contains(Self.instructionSlot)
            if let instructionPair {
                guard hasSlot else {
                    throw ReaderError(
                        reason: "template '\(id)' declares an instructionPair but its "
                            + "text has no \(Self.instructionSlot) slot — the T+/T− "
                            + "instructions would never reach the model. Repair: add "
                            + "\(Self.instructionSlot) to the text, or drop "
                            + "instructionPair to make this a single-template reader")
                }
                guard !instructionPair.experimental.isEmpty,
                    !instructionPair.reference.isEmpty
                else {
                    throw ReaderError(
                        reason: "template '\(id)': instructionPair needs both "
                            + "'experimental' (T+) and 'reference' (T−) — one empty "
                            + "instruction makes the contrast a rendering artifact. "
                            + "Repair: write both instructions")
                }
                guard instructionPair.experimental != instructionPair.reference else {
                    throw ReaderError(
                        reason: "template '\(id)': instructionPair's experimental and "
                            + "reference instructions are identical — every difference "
                            + "would be exactly zero. Repair: write two instructions "
                            + "that differ in the quality under study")
                }
            } else if hasSlot {
                throw ReaderError(
                    reason: "template '\(id)' has a \(Self.instructionSlot) slot but "
                        + "declares no instructionPair — nothing would fill it. "
                        + "Repair: add an instructionPair with 'experimental' and "
                        + "'reference', or remove the slot")
            }
        }

        /// Pure slot substitution; the family-aware scaffold pass happens in
        /// `renderScaffold`. `instruction` fills `{{instruction}}` and is
        /// required exactly when the template declares an instruction pair.
        public func render(
            stimulus: String, concept: String? = nil, instruction: String? = nil
        ) throws -> String {
            guard text.contains("{{stimulus}}") else {
                throw ReaderError(reason: "template '\(id)' has no {{stimulus}} slot")
            }
            var rendered = text
            if conceptSlot {
                guard let concept, !concept.isEmpty else {
                    throw ReaderError(
                        reason: "template '\(id)' names the concept but none was given")
                }
                rendered = rendered.replacingOccurrences(of: "{{concept}}", with: concept)
            } else if rendered.contains("{{concept}}") {
                throw ReaderError(
                    reason: "template '\(id)' declares conceptSlot=false but contains "
                        + "a {{concept}} slot")
            }
            if isTemplatePair {
                guard let instruction, !instruction.isEmpty else {
                    throw ReaderError(
                        reason: "template '\(id)' is a T+/T− template-pair template but "
                            + "no instruction was given — render it once per instruction")
                }
                rendered = rendered.replacingOccurrences(
                    of: Self.instructionSlot, with: instruction)
            } else if rendered.contains(Self.instructionSlot) {
                throw ReaderError(
                    reason: "template '\(id)' declares no instructionPair but contains "
                        + "an \(Self.instructionSlot) slot")
            }
            return rendered.replacingOccurrences(of: "{{stimulus}}", with: stimulus)
        }
    }

    /// Loads one template JSON; hash = SHA-256 over the file's raw bytes.
    /// The registry is one file per id: the id must match the filename.
    public static func loadTemplate(url: URL) throws -> TaskTemplate {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ReaderError(reason: "missing template file: \(url.path)")
        }
        let data = try Data(contentsOf: url)
        var template: TaskTemplate
        do {
            template = try JSONDecoder().decode(TaskTemplate.self, from: data)
        } catch {
            throw ReaderError(reason: "\(url.path): invalid template JSON: \(error)")
        }
        template.hash = sha256Hex(data)
        let expected = url.deletingPathExtension().lastPathComponent
        guard url.pathExtension != "json" || template.id == expected else {
            throw ReaderError(
                reason: "\(url.path): template id '\(template.id)' does not match "
                    + "filename '\(expected)' — the registry is one file per id")
        }
        try template.validateInstructionSlot()
        return template
    }

    // MARK: - Family-aware scaffold pass (server twin: render_reader)

    /// Chat-template / special-token markers that must never appear in a
    /// reader scaffold. Under a RAW rendering the scaffold IS the whole token
    /// sequence, so an embedded BOS or turn marker is the double-BOS /
    /// hand-tokenized-template hazard; under a CHAT-TEMPLATE rendering the
    /// scaffold is the user turn's CONTENT, so an embedded marker forges a
    /// turn boundary inside a turn. Same list, two different reasons — see
    /// `renderReaderScaffold`.
    static let forbiddenScaffoldMarkers = [
        "<bos>", "<eos>", "<start_of_turn>", "<end_of_turn>",  // Gemma
        "<|im_start|>", "<|im_end|>", "<|endoftext|>",  // Qwen/ChatML
        "</s>",
    ]

    /// Family-aware pass for a RepE reader scaffold (measurement, not chat).
    ///
    /// **Under `raw` (the default, and every reader fitted before the
    /// rendering became declarable):** scaffolds are rawCompletion-style plain
    /// text. The LAT reader reads the hidden state at the scaffold's final
    /// token, so the rendered string must reach the model through exactly the
    /// tokenizer conventions the extractor uses
    /// (`ConceptExtractor.activations` tokenizes with the tokenizer defaults —
    /// single BOS, no chat template). Family notes:
    ///
    /// - **Gemma**: no system role exists and the tokenizer adds BOS itself; a
    ///   scaffold that embeds `<bos>`/turn markers would double them.
    /// - **Qwen**: no ` /no_think` suffix is appended (unlike the
    ///   rawCompletion *generation* path) — nothing is generated, and
    ///   appending it would move the LAT token off the scaffold's final
    ///   position.
    ///
    /// **Under `chatTemplate`:** the family template IS the marker mechanism —
    /// it supplies BOS, the turn markers, and the generation prompt, exactly
    /// once, and the scaffold becomes the user turn's content. So the
    /// double-BOS rationale above does not apply and its refusal must not
    /// fire. What survives is a different and narrower hazard: a marker
    /// embedded in the CONTENT forges a turn boundary inside the turn, which
    /// both splits the scaffold the model was meant to read as one unit and
    /// moves the LAT token off the generation prompt's tail. The manual-`<s>`
    /// check is dropped there, because under a template a leading `<s>` in
    /// content is ordinary text, not a hand-written BOS.
    ///
    /// The convention actually applied is stamped into every reader artifact
    /// as `renderingConvention`, beside the structured `extractionRendering`.
    public static func renderReaderScaffold(
        _ text: String, modelID: String,
        rendering: ExtractionRendering? = nil
    ) throws -> String {
        _ = modelID  // both families share the plain-text contract today
        let isRaw = rendering?.isRaw ?? true
        for marker in forbiddenScaffoldMarkers where text.contains(marker) {
            if isRaw {
                throw ReaderError(
                    reason: "reader scaffold embeds special/chat-template marker '\(marker)' — "
                        + "scaffolds must be plain text; the tokenizer adds special tokens "
                        + "exactly once (double-BOS / hand-tokenized-template hazard)")
            }
            throw ReaderError(
                reason: "reader scaffold embeds turn marker '\(marker)' inside the user "
                    + "turn's content — under a chatTemplate rendering the template "
                    + "supplies the markers, so an embedded one forges a turn boundary "
                    + "and moves the LAT token off the generation prompt. Repair: write "
                    + "the scaffold as plain content and let the template do the framing")
        }
        if isRaw, text.drop(while: \.isWhitespace).hasPrefix("<s>") {
            throw ReaderError(
                reason: "reader scaffold embeds a manual '<s>' BOS — the tokenizer adds it")
        }
        return text
    }

    /// Template substitution + the family-aware scaffold pass.
    public static func renderScaffold(
        template: TaskTemplate, stimulus: String, concept: String?,
        modelID: String, instruction: String? = nil,
        rendering: ExtractionRendering? = nil
    ) throws -> String {
        try renderReaderScaffold(
            template.render(
                stimulus: stimulus, concept: concept, instruction: instruction),
            modelID: modelID, rendering: rendering)
    }

    // MARK: - Reader dataset (pairs.jsonl)

    /// One row: unrendered stimuli + the template that will render them
    /// (REPE-IMPLEMENTATION-BRIEF §3). Storing the raw stimulus keeps the
    /// corpus re-renderable per model family while the hash pins the bytes. JSON keys match the server's
    /// `pairs.jsonl` contract exactly.
    ///
    /// Two shapes, one file format:
    ///
    /// - **content pair** — `positiveStimulus` + `negativeStimulus`: two
    ///   different texts under one template (the legacy and still-default
    ///   shape).
    /// - **single stimulus** — `stimulus`: ONE text, rendered under a
    ///   template-pair template's T+ and T− instructions. The contrast is the
    ///   instruction, so a second text would be a confound, not data.
    public struct Pair: Codable, Sendable, Equatable {
        public var id: String?
        public var concept: String
        public var positiveStimulus: String
        public var negativeStimulus: String
        /// Set (with both stimuli empty) for a template-pair row. Optional and
        /// nil-omitted so a content-pair row encodes byte-identically to every
        /// row written before this shape existed.
        public var stimulus: String?
        public var topic: String?
        public var split: String
        public var templateID: String

        enum CodingKeys: String, CodingKey {
            case id, concept, positiveStimulus, negativeStimulus, stimulus
            case topic, split, templateID
        }

        public init(
            id: String? = nil, concept: String,
            positiveStimulus: String, negativeStimulus: String,
            stimulus: String? = nil,
            topic: String? = nil, split: String = "train", templateID: String
        ) {
            self.id = id
            self.concept = concept
            self.positiveStimulus = positiveStimulus
            self.negativeStimulus = negativeStimulus
            self.stimulus = stimulus
            self.topic = topic
            self.split = split
            self.templateID = templateID
        }

        /// A template-pair row: one stimulus, no content contrast.
        public static func templatePair(
            id: String? = nil, concept: String, stimulus: String,
            topic: String? = nil, split: String = "train", templateID: String
        ) -> Pair {
            Pair(
                id: id, concept: concept, positiveStimulus: "", negativeStimulus: "",
                stimulus: stimulus, topic: topic, split: split, templateID: templateID)
        }

        public var isTemplatePairRow: Bool { stimulus != nil }

        /// Written so a template-pair row encodes to the shape the LOADER
        /// accepts. The synthesized encoding emitted `positiveStimulus` and
        /// `negativeStimulus` unconditionally — as empty strings on a
        /// template-pair row — and `parsePairs` refuses exactly that
        /// combination ("row declares both 'stimulus' and a positive/negative
        /// pair"), so the type could parse a shape it could not write. A
        /// content-pair row encodes byte-identically to the synthesized form
        /// (optionals still nil-omitted), which is what keeps the pinned
        /// cross-engine dataset hash where it was.
        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encodeIfPresent(id, forKey: .id)
            try container.encode(concept, forKey: .concept)
            if let stimulus {
                try container.encode(stimulus, forKey: .stimulus)
            } else {
                try container.encode(positiveStimulus, forKey: .positiveStimulus)
                try container.encode(negativeStimulus, forKey: .negativeStimulus)
            }
            try container.encodeIfPresent(topic, forKey: .topic)
            try container.encode(split, forKey: .split)
            try container.encode(templateID, forKey: .templateID)
        }
    }

    public struct Dataset: Sendable, Equatable {

        /// Which row shape the whole file uses. A dataset is one shape or the
        /// other, never both: the two produce different differences, so a
        /// mixed file has no single meaning.
        public enum Shape: String, Sendable, Equatable {
            case contentPair
            case singleStimulus
        }

        public let concept: String
        public let pairs: [Pair]
        /// SHA-256 over the file's raw bytes (stimulus-set convention).
        public let hash: String

        public var train: [Pair] { pairs.filter { $0.split == "train" } }
        public var heldOut: [Pair] { pairs.filter { $0.split != "train" } }

        public var shape: Shape {
            pairs.first?.isTemplatePairRow == true ? .singleStimulus : .contentPair
        }

        public init(concept: String, pairs: [Pair], hash: String) {
            self.concept = concept
            self.pairs = pairs
            self.hash = hash
        }
    }

    /// Loads `pairs.jsonl`; hash = SHA-256 over raw bytes. Rows must share one
    /// concept — a reader measures exactly one.
    public static func loadPairs(url: URL) throws -> Dataset {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ReaderError(reason: "missing reader pairs file: \(url.path)")
        }
        let data = try Data(contentsOf: url)
        return try parsePairs(data, source: url.path)
    }

    static func requireRowKeys(
        _ pairs: [(String, String?)], source: String
    ) throws {
        for (key, value) in pairs where value == nil {
            throw ReaderError(reason: "\(source): row missing '\(key)'")
        }
    }

    /// Split from file IO so the parse is unit-testable (and reusable by the
    /// Concept Lab editor).
    public static func parsePairs(_ data: Data, source: String) throws -> Dataset {
        struct Row: Decodable {
            let id: String?
            let concept: String?
            let positiveStimulus: String?
            let negativeStimulus: String?
            let stimulus: String?
            let topic: String?
            let split: String?
            let templateID: String?
        }
        let decoder = JSONDecoder()
        var pairs: [Pair] = []
        let lines = String(decoding: data, as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: true)
        for raw in lines {
            let line = raw.trimmingCharacters(in: .whitespaces)
            guard !line.isEmpty else { continue }
            guard let row = try? decoder.decode(Row.self, from: Data(line.utf8)) else {
                throw ReaderError(reason: "\(source): invalid JSON line")
            }
            let split = (row.split?.isEmpty == false ? row.split! : "train").lowercased()
            // Shape first (which keys a row OWES depends on it), then the two
            // keys every row owes whatever its shape.
            if let stimulus = row.stimulus {
                guard row.positiveStimulus == nil, row.negativeStimulus == nil else {
                    throw ReaderError(
                        reason: "\(source): row declares both 'stimulus' and a "
                            + "positive/negative pair — a template-pair row holds ONE "
                            + "stimulus (the T+/T− instructions carry the contrast). "
                            + "Repair: drop 'stimulus' for a content pair, or drop "
                            + "'positiveStimulus'/'negativeStimulus' for a template pair")
                }
                try requireRowKeys(
                    [("concept", row.concept), ("templateID", row.templateID)],
                    source: source)
                pairs.append(
                    Pair.templatePair(
                        id: row.id, concept: row.concept!, stimulus: stimulus,
                        topic: row.topic, split: split, templateID: row.templateID!))
                continue
            }
            try requireRowKeys(
                [
                    ("positiveStimulus", row.positiveStimulus),
                    ("negativeStimulus", row.negativeStimulus),
                    ("concept", row.concept),
                    ("templateID", row.templateID),
                ], source: source)
            pairs.append(
                Pair(
                    id: row.id, concept: row.concept!,
                    positiveStimulus: row.positiveStimulus!,
                    negativeStimulus: row.negativeStimulus!,
                    topic: row.topic, split: split, templateID: row.templateID!))
        }
        guard !pairs.isEmpty else {
            throw ReaderError(reason: "empty reader pairs file: \(source)")
        }
        let concepts = Set(pairs.map(\.concept))
        guard concepts.count == 1 else {
            throw ReaderError(
                reason: "\(source): mixed concepts \(concepts.sorted()) — one reader "
                    + "dataset per concept")
        }
        let shapes = Set(pairs.map(\.isTemplatePairRow))
        guard shapes.count == 1 else {
            throw ReaderError(
                reason: "\(source): mixes content-pair rows (positiveStimulus/"
                    + "negativeStimulus) with template-pair rows (stimulus) — the two "
                    + "produce different differences, so one file cannot mean both. "
                    + "Repair: split them into two datasets")
        }
        return Dataset(concept: pairs[0].concept, pairs: pairs, hash: sha256Hex(data))
    }

    /// Which contrast a (dataset, template) combination declares — derived,
    /// never passed loosely, so the fit cannot be asked for a construction its
    /// inputs do not support.
    public static func resolveContrastMode(
        dataset: Dataset, template: TaskTemplate
    ) throws -> ContrastMode {
        try resolveContrastMode(shape: dataset.shape, template: template)
    }

    /// The same derivation over a dataset SHAPE alone, for a caller that has
    /// the shape before it has the rows — the Concept Lab's reader editor,
    /// which must show the derived contrast (or the refusal) while the pairs
    /// are still being authored. Same switch, same refusal literals, one
    /// owner: a pane that re-worded either would be inventing a message the
    /// CLI does not use.
    public static func resolveContrastMode(
        shape: Dataset.Shape, template: TaskTemplate
    ) throws -> ContrastMode {
        switch (shape, template.isTemplatePair) {
        case (.contentPair, false):
            return .supervisedContent
        case (.singleStimulus, true):
            return .unsupervisedTemplatePair
        case (.contentPair, true):
            throw ReaderError(
                reason: "template '\(template.id)' declares a T+/T− instructionPair but "
                    + "the dataset holds content pairs (positiveStimulus/"
                    + "negativeStimulus) — under a template pair the contrast is the "
                    + "INSTRUCTION and a second stimulus would be a confound. Repair: "
                    + "fit these pairs through a single-template reader template, or "
                    + "rewrite the dataset as one-stimulus ('stimulus') rows")
        case (.singleStimulus, false):
            throw ReaderError(
                reason: "the dataset holds one-stimulus ('stimulus') rows but template "
                    + "'\(template.id)' declares no instructionPair — there is nothing "
                    + "to contrast the stimulus against. Repair: choose a template-pair "
                    + "template (one with 'instructionPair'), or rewrite the dataset as "
                    + "positiveStimulus/negativeStimulus content pairs")
        }
    }

    // MARK: - Reader artifact

    /// The fitted instrument (REPE-IMPLEMENTATION-BRIEF §4): one per concept × layer × template ×
    /// model × substrate. The full template record is embedded so inference is
    /// standalone and drift-proof; `templateID`/`templateHash` remain the
    /// registry pins. The encoded JSON shape matches the server's
    /// `ReaderArtifact.to_dict` byte-for-byte in schema (keys, nested "probe",
    /// nil omission) so artifacts are cross-engine *readable* — but never
    /// cross-engine *usable*: `substrate` gates scoring and verification.
    ///
    /// **Schema growth rule.** Every field added after schema 1 decodes with an
    /// explicit LEGACY default, named here, so a reader artifact written by an
    /// older engine keeps loading and keeps meaning what it meant:
    /// `contrastMode` absent = `supervisedContent`, `signConvention` absent =
    /// `trainMajority`, `extractionRendering` absent = raw,
    /// `pc1ExplainedVarianceOfDifferences` absent = the legacy
    /// `pc1ExplainedVariance` under basis `alternatedRows`.
    public struct Artifact: Codable, Sendable, Equatable {
        public var modelID: String
        public var revision: String?
        public var concept: String
        public var layer: Int
        public var template: TaskTemplate
        public var datasetHash: String
        public var probe: SteeringVectorMath.ScalarProbe
        /// Share of the DIFFERENCE CLOUD's variance that PC1 explains.
        ///
        /// Before 2026-08-27 this number was computed over the ALTERNATED rows
        /// — the ± symmetrized copies PCA is actually fitted on — where the
        /// cloud's mean is near zero by construction and the ratio is
        /// systematically flattering. A reader reasonably assumes it describes
        /// the differences themselves, so it now does. `explainedVarianceBasis`
        /// says which of the two a given artifact carries.
        /// nil when the cloud has no variance to apportion — see
        /// `explainedVarianceBasis == "degenerateDifferenceCloud"`.
        public var differenceCloudExplainedVariance: Float?
        /// `"differenceCloud"` (fitted here), `"degenerateDifferenceCloud"`
        /// (fitted here, but every difference was identical so there is no
        /// variance to apportion and the value is absent), or
        /// `"alternatedRows"` (decoded from a pre-2026-08-27 artifact's
        /// `pc1ExplainedVariance`, which measured the ± symmetrized copies).
        public var explainedVarianceBasis: String
        public var trainAccuracy: Float
        public var heldOutAccuracy: Float?
        public var trainPairCount: Int
        public var heldOutPairCount: Int
        /// Which contrast the pair differences express (absent = legacy
        /// `supervisedContent`).
        public var contrastMode: ContrastMode
        /// How this layer's sign was fixed (absent = legacy `trainMajority`).
        public var signConvention: SignConvention
        /// Held-out paired-discrimination accuracy of the CHOSEN sign, when
        /// the held-out split fixed it. nil under `trainMajority`.
        public var signHeldOutAccuracy: Float?
        /// Why the held-out sign rule stood down, when it did. Present exactly
        /// when `signConvention == .trainMajority` on a fit that tried.
        public var signFallbackReason: String?
        /// The seed the unsupervised orientation draw used. nil under
        /// `supervisedContent` (whose alternation needs no RNG).
        public var orientationSeed: UInt64?
        /// The layer with the highest held-out accuracy in the SET this
        /// artifact was fitted with — a RECOMMENDATION, never a selection.
        /// Which layer a study reads is a declarable choice that lives in the
        /// manifest; nothing here or at use time picks it silently.
        public var recommendedLayer: Int?
        public var recommendedLayerAccuracy: Float?
        /// `"heldOutAccuracy"`, or `"trainAccuracy"` when the fit had no
        /// held-out pairs to recommend from.
        public var layerRecommendationBasis: String?
        public var substrate: String
        /// HOW the scaffold reached the model. Absent = legacy raw, which is
        /// why a raw fit encodes byte-identically to a pre-2026-08-27 one.
        public var extractionRendering: ExtractionRendering?
        public var renderingConvention: String
        public var extractionDate: String

        public var templateID: String { template.id }
        public var templateHash: String { template.hash }
        public var latTokenPosition: String { template.latToken }

        /// The rendering actually applied — absent resolves to legacy raw.
        public var resolvedExtractionRendering: ExtractionRendering {
            extractionRendering ?? .raw
        }

        public func readingPosition() throws -> ReadingPosition {
            try template.readingPosition()
        }

        /// What a reader must be told about this instrument's layer, in one
        /// sentence, wherever the recommendation is surfaced.
        public static let layerRecommendationNote =
            "recommendedLayer is the argmax of held-out accuracy over the layers "
            + "fitted together; it is a RECOMMENDATION. The layer a study reads is "
            + "declared in the manifest and is never selected automatically at "
            + "use time."

        enum CodingKeys: String, CodingKey {
            case artifactType, schemaVersion
            case modelID, revision, substrate, concept, layer
            case templateID, templateHash, template, templateDivergence
            case datasetHash, latTokenPosition, readingPosition
            case probe
            case pc1ExplainedVariance  // legacy key; decode-only
            case pc1ExplainedVarianceOfDifferences, pc1ExplainedVarianceBasis
            case trainAccuracy, heldOutAccuracy
            case trainPairCount, heldOutPairCount
            case contrastMode, signConvention, signHeldOutAccuracy
            case signFallbackReason, orientationSeed
            case recommendedLayer, recommendedLayerAccuracy
            case layerRecommendationBasis, layerRecommendationNote
            case extractionRendering
            case renderingConvention, extractionDate
        }

        public init(
            modelID: String, revision: String?, concept: String, layer: Int,
            template: TaskTemplate, datasetHash: String,
            probe: SteeringVectorMath.ScalarProbe,
            differenceCloudExplainedVariance: Float?,
            explainedVarianceBasis: String? = nil,
            trainAccuracy: Float,
            heldOutAccuracy: Float?, trainPairCount: Int, heldOutPairCount: Int,
            contrastMode: ContrastMode = .supervisedContent,
            signConvention: SignConvention = .trainMajority,
            signHeldOutAccuracy: Float? = nil,
            signFallbackReason: String? = nil,
            orientationSeed: UInt64? = nil,
            recommendedLayer: Int? = nil,
            recommendedLayerAccuracy: Float? = nil,
            layerRecommendationBasis: String? = nil,
            substrate: String = RepEReader.substrate,
            extractionRendering: ExtractionRendering? = nil,
            renderingConvention: String? = nil,
            extractionDate: String = Artifact.timestamp()
        ) {
            self.modelID = modelID
            self.revision = revision
            self.concept = concept
            self.layer = layer
            self.template = template
            self.datasetHash = datasetHash
            self.probe = probe
            self.differenceCloudExplainedVariance = differenceCloudExplainedVariance
            self.explainedVarianceBasis =
                explainedVarianceBasis
                ?? (differenceCloudExplainedVariance == nil
                    ? "degenerateDifferenceCloud" : "differenceCloud")
            self.trainAccuracy = trainAccuracy
            self.heldOutAccuracy = heldOutAccuracy
            self.trainPairCount = trainPairCount
            self.heldOutPairCount = heldOutPairCount
            self.contrastMode = contrastMode
            self.signConvention = signConvention
            self.signHeldOutAccuracy = signHeldOutAccuracy
            self.signFallbackReason = signFallbackReason
            self.orientationSeed = orientationSeed
            self.recommendedLayer = recommendedLayer
            self.recommendedLayerAccuracy = recommendedLayerAccuracy
            self.layerRecommendationBasis = layerRecommendationBasis
            self.substrate = substrate
            self.extractionRendering = extractionRendering?.stamp
            self.renderingConvention =
                renderingConvention
                ?? RepEReader.renderingConvention(for: extractionRendering)
            self.extractionDate = extractionDate
        }

        public static func timestamp(_ date: Date = Date()) -> String {
            let formatter = ISO8601DateFormatter()
            formatter.formatOptions = [.withInternetDateTime]
            return formatter.string(from: date)
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            let kind = try container.decodeIfPresent(String.self, forKey: .artifactType)
            guard kind == RepEReader.artifactType else {
                throw ReaderError(
                    reason: "not a \(RepEReader.artifactType) artifact "
                        + "(artifactType=\(kind ?? "nil"))")
            }
            modelID = try container.decode(String.self, forKey: .modelID)
            revision = try container.decodeIfPresent(String.self, forKey: .revision)
            concept = try container.decode(String.self, forKey: .concept)
            layer = try container.decode(Int.self, forKey: .layer)
            var decodedTemplate = try container.decode(TaskTemplate.self, forKey: .template)
            // The registry pin wins over the embedded copy's own hash field
            // (server `ReaderArtifact.from_dict` behavior).
            if let pinned = try container.decodeIfPresent(String.self, forKey: .templateHash) {
                decodedTemplate.hash = pinned
            }
            template = decodedTemplate
            datasetHash = try container.decode(String.self, forKey: .datasetHash)
            probe = try container.decode(
                SteeringVectorMath.ScalarProbe.self, forKey: .probe)
            let declaredBasis = try container.decodeIfPresent(
                String.self, forKey: .pc1ExplainedVarianceBasis)
            if let current = try container.decodeIfPresent(
                Float.self, forKey: .pc1ExplainedVarianceOfDifferences)
            {
                differenceCloudExplainedVariance = current
                explainedVarianceBasis = declaredBasis ?? "differenceCloud"
            } else if let legacy = try container.decodeIfPresent(
                Float.self, forKey: .pc1ExplainedVariance)
            {
                // A pre-2026-08-27 artifact: the number is over the ALTERNATED
                // rows, and the basis stamp says so rather than relabelling it.
                differenceCloudExplainedVariance = legacy
                explainedVarianceBasis = "alternatedRows"
            } else if let declaredBasis {
                // A degenerate difference cloud: the basis is stamped and the
                // number is deliberately absent, because zero variance has no
                // share to apportion and writing 0 would read as "PC1 explains
                // nothing" — the opposite of what a one-point cloud means.
                differenceCloudExplainedVariance = nil
                explainedVarianceBasis = declaredBasis
            } else {
                throw ReaderError(
                    reason: "reader artifact carries neither "
                        + "'pc1ExplainedVarianceOfDifferences', its "
                        + "'pc1ExplainedVarianceBasis', nor the legacy "
                        + "'pc1ExplainedVariance' — one of them is required")
            }
            trainAccuracy = try container.decode(Float.self, forKey: .trainAccuracy)
            heldOutAccuracy = try container.decodeIfPresent(
                Float.self, forKey: .heldOutAccuracy)
            trainPairCount = try container.decodeIfPresent(
                Int.self, forKey: .trainPairCount) ?? 0
            heldOutPairCount = try container.decodeIfPresent(
                Int.self, forKey: .heldOutPairCount) ?? 0
            contrastMode =
                try container.decodeIfPresent(ContrastMode.self, forKey: .contrastMode)
                ?? .supervisedContent
            signConvention =
                try container.decodeIfPresent(SignConvention.self, forKey: .signConvention)
                ?? .trainMajority
            signHeldOutAccuracy = try container.decodeIfPresent(
                Float.self, forKey: .signHeldOutAccuracy)
            signFallbackReason = try container.decodeIfPresent(
                String.self, forKey: .signFallbackReason)
            orientationSeed = try container.decodeIfPresent(
                UInt64.self, forKey: .orientationSeed)
            recommendedLayer = try container.decodeIfPresent(
                Int.self, forKey: .recommendedLayer)
            recommendedLayerAccuracy = try container.decodeIfPresent(
                Float.self, forKey: .recommendedLayerAccuracy)
            layerRecommendationBasis = try container.decodeIfPresent(
                String.self, forKey: .layerRecommendationBasis)
            substrate = try container.decodeIfPresent(String.self, forKey: .substrate) ?? ""
            extractionRendering = try container.decodeIfPresent(
                ExtractionRendering.self, forKey: .extractionRendering)
            renderingConvention = try container.decodeIfPresent(
                String.self, forKey: .renderingConvention) ?? ""
            extractionDate = try container.decodeIfPresent(
                String.self, forKey: .extractionDate) ?? ""
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(RepEReader.artifactType, forKey: .artifactType)
            try container.encode(2, forKey: .schemaVersion)
            try container.encode(modelID, forKey: .modelID)
            try container.encodeIfPresent(revision, forKey: .revision)
            try container.encode(substrate, forKey: .substrate)
            try container.encode(concept, forKey: .concept)
            try container.encode(layer, forKey: .layer)
            try container.encode(template.id, forKey: .templateID)
            try container.encode(template.hash, forKey: .templateHash)
            try container.encode(template, forKey: .template)
            try container.encodeIfPresent(template.divergence, forKey: .templateDivergence)
            try container.encode(datasetHash, forKey: .datasetHash)
            try container.encode(template.latToken, forKey: .latTokenPosition)
            try container.encode(try readingPosition().label, forKey: .readingPosition)
            try container.encode(probe, forKey: .probe)
            // The legacy `pc1ExplainedVariance` key is deliberately NOT
            // written: the number's basis changed, and writing the old key
            // with the new semantics would make every pre-existing consumer
            // silently wrong instead of visibly out of date.
            try container.encodeIfPresent(
                differenceCloudExplainedVariance,
                forKey: .pc1ExplainedVarianceOfDifferences)
            try container.encode(explainedVarianceBasis, forKey: .pc1ExplainedVarianceBasis)
            try container.encode(trainAccuracy, forKey: .trainAccuracy)
            try container.encodeIfPresent(heldOutAccuracy, forKey: .heldOutAccuracy)
            try container.encode(trainPairCount, forKey: .trainPairCount)
            try container.encode(heldOutPairCount, forKey: .heldOutPairCount)
            try container.encode(contrastMode, forKey: .contrastMode)
            try container.encode(signConvention, forKey: .signConvention)
            try container.encodeIfPresent(signHeldOutAccuracy, forKey: .signHeldOutAccuracy)
            try container.encodeIfPresent(signFallbackReason, forKey: .signFallbackReason)
            try container.encodeIfPresent(orientationSeed, forKey: .orientationSeed)
            try container.encodeIfPresent(recommendedLayer, forKey: .recommendedLayer)
            try container.encodeIfPresent(
                recommendedLayerAccuracy, forKey: .recommendedLayerAccuracy)
            try container.encodeIfPresent(
                layerRecommendationBasis, forKey: .layerRecommendationBasis)
            if recommendedLayer != nil {
                try container.encode(
                    Artifact.layerRecommendationNote, forKey: .layerRecommendationNote)
            }
            try container.encodeIfPresent(extractionRendering, forKey: .extractionRendering)
            try container.encode(renderingConvention, forKey: .renderingConvention)
            try container.encode(extractionDate, forKey: .extractionDate)
        }
    }

    /// Writes `<name>.json` into a run directory; returns the file URL.
    @discardableResult
    public static func saveArtifact(
        _ artifact: Artifact, to directory: URL, name: String? = nil
    ) throws -> URL {
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let name = name ?? "reader-\(artifact.concept)-layer\(artifact.layer)"
        let url = directory.appending(component: "\(name).json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(artifact).write(to: url)
        return url
    }

    public static func loadArtifact(url: URL) throws -> Artifact {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ReaderError(reason: "missing reader artifact: \(url.path)")
        }
        let data = try Data(contentsOf: url)
        do {
            return try JSONDecoder().decode(Artifact.self, from: data)
        } catch let error as ReaderError {
            throw error
        } catch {
            throw ReaderError(reason: "\(url.path): unreadable reader artifact: \(error)")
        }
    }

    // MARK: - Fit

    /// One layer's direction, with the provenance of how its sign was fixed.
    struct FittedDirection {
        var component: [Float]
        var differenceCloudExplainedVariance: Float?
        var signConvention: SignConvention
        var signHeldOutAccuracy: Float?
        var signFallbackReason: String?
    }

    /// PC1 of the paired differences, signed by the HELD-OUT split when it can
    /// be (paper step 4) and by train-label majority when it cannot.
    ///
    /// Construction by contrast mode:
    ///
    /// - `supervisedContent`: each difference is L2-normalized before PCA (OUR
    ///   departure, not the paper's — see `SteeringVectorMath.direction`), then
    ///   enters in alternating ± orientation. Alternation matters: once
    ///   magnitudes are normalized away, labeled differences all point the same
    ///   way, and centering would subtract the shared concept direction out of
    ///   the data entirely.
    /// - `unsupervisedTemplatePair`: the reference implementation's own
    ///   construction — NO normalization, per-row random orientation
    ///   (`random.shuffle(d)` then `[::2] − [1::2]`), mean-centering, PCA with
    ///   `n_components=1`. The orientation draw is seeded and the seed is
    ///   stamped, which is the one place we improve on the reference: its
    ///   shuffle is unseeded, so its direction cannot be reproduced.
    ///
    /// Why the sign cannot come from the probe's own accuracy: `scalarProbe`
    /// derives `orientation` from the train class means, so flipping the
    /// direction flips the orientation too and leaves every score identical.
    /// Sign selection therefore scores the PAIRED discrimination — does a
    /// held-out (positive − negative) difference project positive? — which is
    /// exactly what `get_signs` asks of the train split, asked of held-out
    /// data instead.
    static func fitDirection(
        posTrain: [[Float]], negTrain: [[Float]],
        posHeld: [[Float]], negHeld: [[Float]],
        contrastMode: ContrastMode,
        orientationSeed: UInt64
    ) throws -> FittedDirection {
        guard posTrain.count == negTrain.count,
            posTrain.first?.count == negTrain.first?.count
        else {
            throw ReaderError(
                reason: "unpaired activations: positive \(posTrain.count) vs "
                    + "negative \(negTrain.count)")
        }
        var diffs: [[Float]] = []
        for (p, n) in zip(posTrain, negTrain) {
            let d = zip(p, n).map(-)
            let norm = SteeringVectorMath.l2Norm(d)
            guard norm > 0 else { continue }
            switch contrastMode {
            case .supervisedContent: diffs.append(d.map { $0 / norm })
            case .unsupervisedTemplatePair: diffs.append(d)
            }
        }
        guard diffs.count >= 2 else {
            throw ReaderError(reason: "need at least 2 non-degenerate train pairs")
        }

        let oriented: [[Float]]
        switch contrastMode {
        case .supervisedContent:
            oriented = diffs.enumerated().map { index, d in
                index.isMultiple(of: 2) ? d : d.map { -$0 }
            }
        case .unsupervisedTemplatePair:
            let signs = orientationSigns(count: diffs.count, seed: orientationSeed)
            oriented = zip(diffs, signs).map { d, s in s > 0 ? d : d.map { -$0 } }
        }

        let result: SteeringVectorMath.PrincipalComponentsResult
        do {
            result = try SteeringVectorMath.principalComponentsWithVariance(
                of: oriented, count: 1)
        } catch {
            throw ReaderError(reason: "degenerate pair differences (no PC1)")
        }
        guard var pc = result.components.first else {
            throw ReaderError(reason: "degenerate pair differences (no PC1)")
        }
        let explained = explainedVarianceOfDifferenceCloud(diffs, component: pc)

        // --- sign: held-out first (paper step 4), train majority as fallback
        let heldDiffs = zip(posHeld, negHeld).map { p, n in zip(p, n).map(-) }
        let heldScores = heldDiffs.map { SteeringVectorMath.dot($0, pc) }
        let agree = heldScores.count { $0 > 0 }
        let disagree = heldScores.count { $0 < 0 }
        let decided = agree + disagree

        var convention = SignConvention.heldOutPairAgreement
        var signHeldOutAccuracy: Float?
        var fallbackReason: String?
        var flip = false

        if decided >= minimumHeldOutPairsForSignSelection, agree != disagree {
            flip = disagree > agree
            signHeldOutAccuracy = Float(max(agree, disagree)) / Float(decided)
        } else {
            convention = .trainMajority
            fallbackReason = heldOutSignFallbackReason(
                heldOutPairCount: heldDiffs.count, decided: decided,
                agree: agree, disagree: disagree)
            let scores = diffs.map { SteeringVectorMath.dot($0, pc) }
            let positiveScores = scores.count { $0 > 0 }
            if positiveScores * 2 == scores.count {
                let meanDiff = try SteeringVectorMath.meanDifference(
                    positive: posTrain, negative: negTrain)
                flip = SteeringVectorMath.dot(pc, meanDiff) < 0  // tie: class means
            } else {
                flip = positiveScores * 2 < scores.count
            }
        }
        // Flipping the component does not change the held-out accuracy that
        // was just recorded: it IS the accuracy of the chosen sign, which is
        // the larger of the two by construction.
        if flip { pc = pc.map { -$0 } }
        return FittedDirection(
            component: pc, differenceCloudExplainedVariance: explained,
            signConvention: convention, signHeldOutAccuracy: signHeldOutAccuracy,
            signFallbackReason: fallbackReason)
    }

    /// Why the held-out sign rule stood down — stamped into the artifact, so a
    /// fit that fell back cannot be mistaken for one that did not.
    static func heldOutSignFallbackReason(
        heldOutPairCount: Int, decided: Int, agree: Int, disagree: Int
    ) -> String {
        if heldOutPairCount == 0 {
            return "no held-out pairs (every row is split 'train'): the sign follows "
                + "train-label majority, the reference implementation's get_signs. "
                + "Repair: mark some rows with a non-'train' split so the paper's "
                + "held-out sign selection can run"
        }
        if decided < minimumHeldOutPairsForSignSelection {
            return "\(decided) held-out pair(s) projected off zero, below the minimum "
                + "\(minimumHeldOutPairsForSignSelection): a one-pair vote is a coin "
                + "flip wearing a validation split's authority, so the sign follows "
                + "train-label majority instead"
        }
        return "held-out pairs split evenly (\(agree) for, \(disagree) against): the "
            + "held-out set does not discriminate at this layer, so the sign follows "
            + "train-label majority. Read this layer's heldOutAccuracy before trusting "
            + "its direction"
    }

    /// PC1's share of the DIFFERENCE CLOUD's variance: the centered
    /// differences' projection energy over their total energy. Computed on the
    /// differences themselves, NOT on the ± alternated copies PCA is fitted on
    /// — the alternated cloud is centered near zero by construction, which
    /// inflates the ratio into a number that means less than a reader assumes.
    /// nil when the centered cloud carries no variance at all (every
    /// difference identical) — there is then no share to report, and 0 would
    /// say the opposite of what a one-point cloud means.
    static func explainedVarianceOfDifferenceCloud(
        _ differences: [[Float]], component: [Float]
    ) -> Float? {
        guard let centered = try? SteeringVectorMath.centeredRows(differences) else {
            return nil
        }
        let total = SteeringVectorMath.varianceTrace(ofCenteredRows: centered)
        guard total > 0 else { return nil }
        let captured = centered.reduce(Float(0)) { partial, row in
            let projection = SteeringVectorMath.dot(row, component)
            return partial + projection * projection
        }
        return captured / total
    }

    static func pairAccuracy(
        probe: SteeringVectorMath.ScalarProbe,
        positive: [[Float]], negative: [[Float]]
    ) throws -> Float? {
        let total = positive.count + negative.count
        guard total > 0 else { return nil }
        var correct = 0
        for row in positive {
            if try probe.classifiesPositive(row) { correct += 1 }
        }
        for row in negative {
            if try !probe.classifiesPositive(row) { correct += 1 }
        }
        return Float(correct) / Float(total)
    }

    /// Pure fit over pre-captured LAT-token activations — the unit-testable
    /// math half. `capturedValues[textIndex][layer]` follows the rendered-text
    /// order the async `fit` produces: train rows then held-out rows, the
    /// positive/T+ rendering before the negative/T− rendering within each row.
    public static func fit(
        dataset: Dataset,
        template: TaskTemplate,
        capturedValues: [[[Float]]],
        modelID: String,
        revision: String?,
        layers: [Int]? = nil,
        orientationSeed: UInt64 = RepEReader.defaultOrientationSeed,
        extractionRendering: ExtractionRendering? = nil
    ) throws -> [Artifact] {
        for pair in dataset.pairs where pair.templateID != template.id {
            throw ReaderError(
                reason: "pair '\(pair.id ?? "?")' pins template '\(pair.templateID)' "
                    + "but the fit uses '\(template.id)'")
        }
        let contrastMode = try resolveContrastMode(dataset: dataset, template: template)
        let train = dataset.train
        let held = dataset.heldOut
        guard train.count >= 2 else {
            throw ReaderError(
                reason: "need at least 2 train pairs, have \(train.count) "
                    + "(rows default to split 'train')")
        }
        guard capturedValues.count == 2 * (train.count + held.count) else {
            throw ReaderError(
                reason: "captured \(capturedValues.count) activations for "
                    + "\(train.count + held.count) pairs — expected two per pair")
        }
        let layerCount = capturedValues.first?.count ?? 0
        guard layerCount > 0 else {
            throw ReaderError(reason: "no layers captured")
        }
        let chosen = layers ?? Array(0 ..< layerCount)

        let nTrain = train.count
        var artifacts: [Artifact] = []
        for layer in chosen {
            guard (0 ..< layerCount).contains(layer) else {
                throw ReaderError(
                    reason: "layer \(layer) out of range 0..\(layerCount - 1)")
            }
            let posTrain = (0 ..< nTrain).map { capturedValues[2 * $0][layer] }
            let negTrain = (0 ..< nTrain).map { capturedValues[2 * $0 + 1][layer] }
            let posHeld = (0 ..< held.count).map { capturedValues[2 * (nTrain + $0)][layer] }
            let negHeld = (0 ..< held.count).map { capturedValues[2 * (nTrain + $0) + 1][layer] }

            let fitted = try fitDirection(
                posTrain: posTrain, negTrain: negTrain,
                posHeld: posHeld, negHeld: negHeld,
                contrastMode: contrastMode, orientationSeed: orientationSeed)
            // Training normalization: center at the train activation mean,
            // score scale/center from the train projections — the "fit
            // params" the paper's inference reuses on new text.
            let center = try SteeringVectorMath.mean(posTrain + negTrain)
            let probe = try SteeringVectorMath.scalarProbe(
                direction: fitted.component, positive: posTrain, negative: negTrain,
                activationCenter: center)
            let trainAccuracy = try pairAccuracy(
                probe: probe, positive: posTrain, negative: negTrain)
            let heldAccuracy = try pairAccuracy(
                probe: probe, positive: posHeld, negative: negHeld)
            artifacts.append(
                Artifact(
                    modelID: modelID, revision: revision,
                    concept: dataset.concept, layer: layer, template: template,
                    datasetHash: dataset.hash, probe: probe,
                    differenceCloudExplainedVariance:
                        fitted.differenceCloudExplainedVariance,
                    trainAccuracy: trainAccuracy ?? 0,
                    heldOutAccuracy: heldAccuracy,
                    trainPairCount: nTrain, heldOutPairCount: held.count,
                    contrastMode: contrastMode,
                    signConvention: fitted.signConvention,
                    signHeldOutAccuracy: fitted.signHeldOutAccuracy,
                    signFallbackReason: fitted.signFallbackReason,
                    orientationSeed: contrastMode == .unsupervisedTemplatePair
                        ? orientationSeed : nil,
                    extractionRendering: extractionRendering))
        }
        return stampLayerRecommendation(artifacts)
    }

    /// Stamps the argmax-held-out-accuracy layer into every artifact of a set
    /// (paper step 4's layer half). A RECOMMENDATION: nothing downstream may
    /// read it as a selection, because which layer a study reads is a
    /// declarable choice recorded in its manifest.
    static func stampLayerRecommendation(_ artifacts: [Artifact]) -> [Artifact] {
        guard artifacts.count > 1 else { return artifacts }
        let basis =
            artifacts.contains { $0.heldOutAccuracy != nil }
            ? "heldOutAccuracy" : "trainAccuracy"
        func score(_ artifact: Artifact) -> Float {
            basis == "heldOutAccuracy"
                ? (artifact.heldOutAccuracy ?? -1) : artifact.trainAccuracy
        }
        // Ties go to the LOWER layer index: an arbitrary tiebreak, but a
        // stated and stable one, so two fits of the same data recommend the
        // same layer on both engines.
        guard
            let best = artifacts.min(by: {
                score($0) != score($1) ? score($0) > score($1) : $0.layer < $1.layer
            })
        else { return artifacts }
        return artifacts.map { artifact in
            var copy = artifact
            copy.recommendedLayer = best.layer
            copy.recommendedLayerAccuracy = score(best)
            copy.layerRecommendationBasis = basis
            return copy
        }
    }

    /// The rendered texts a fit reads, in the order `fit` expects: train rows
    /// then held-out rows, positive/T+ before negative/T− within each row.
    /// Pure (no model), so the rendering contract is unit-testable.
    public static func fitTexts(
        dataset: Dataset, template: TaskTemplate, modelID: String,
        rendering: ExtractionRendering? = nil
    ) throws -> [String] {
        let contrastMode = try resolveContrastMode(dataset: dataset, template: template)
        var texts: [String] = []
        texts.reserveCapacity(dataset.pairs.count * 2)
        for pair in dataset.train + dataset.heldOut {
            switch contrastMode {
            case .supervisedContent:
                for stimulus in [pair.positiveStimulus, pair.negativeStimulus] {
                    texts.append(
                        try renderScaffold(
                            template: template, stimulus: stimulus,
                            concept: dataset.concept, modelID: modelID,
                            rendering: rendering))
                }
            case .unsupervisedTemplatePair:
                guard let stimulus = pair.stimulus,
                    let instructions = template.instructionPair
                else {
                    throw ReaderError(
                        reason: "row '\(pair.id ?? "?")' carries no 'stimulus' for a "
                            + "template-pair fit")
                }
                for instruction in [instructions.experimental, instructions.reference] {
                    texts.append(
                        try renderScaffold(
                            template: template, stimulus: stimulus,
                            concept: dataset.concept, modelID: modelID,
                            instruction: instruction, rendering: rendering))
                }
            }
        }
        return texts
    }

    /// Fits one reader per layer from the dataset's train split; held-out rows
    /// (any other `split` value) fix each layer's SIGN and score the
    /// instrument they did not fit.
    ///
    /// Rendering goes through `renderScaffold` (family-aware) and then the
    /// declared `extractionRendering`; activations are captured by the same
    /// extraction path every other recipe uses, at the template's LAT token
    /// position.
    public static func fit(
        container: ModelContainer,
        modelID: String,
        revision: String?,
        dataset: Dataset,
        template: TaskTemplate,
        layers: [Int]? = nil,
        orientationSeed: UInt64 = RepEReader.defaultOrientationSeed,
        extractionRendering: ExtractionRendering? = nil
    ) async throws -> [Artifact] {
        let texts = try fitTexts(
            dataset: dataset, template: template, modelID: modelID,
            rendering: extractionRendering)
        let captured = try await ConceptExtractor.activations(
            container: container, texts: texts,
            position: template.readingPosition(),
            rendering: extractionRendering ?? .raw)
        return try fit(
            dataset: dataset, template: template, capturedValues: captured.values,
            modelID: modelID, revision: revision, layers: layers,
            orientationSeed: orientationSeed,
            extractionRendering: extractionRendering)
    }

    // MARK: - Exact inference

    /// Pure probe scoring for a pre-captured LAT-token activation.
    public static func scoreActivation(
        _ reader: Artifact, activation: [Float]
    ) throws -> Float {
        try reader.probe.score(activation)
    }

    /// Preconditions for scoring texts through a fitted reader. Internal but
    /// test-visible (pure CPU, no model) so the guard itself is unit-tested:
    /// a reader is fitted per substrate AND per model — its probe reads a
    /// specific model's residual geometry, so scoring through the wrong model
    /// would return silently meaningless numbers, not an error.
    static func validateForScoring(reader: Artifact, modelID: String) throws {
        guard reader.substrate == substrate else {
            throw ReaderError(
                reason: "reader was fitted on substrate '\(reader.substrate)'; this "
                    + "engine is '\(substrate)' — reader artifacts are "
                    + "substrate-specific, re-fit here")
        }
        guard reader.modelID == modelID else {
            throw ReaderError(
                reason: "reader was fitted on model '\(reader.modelID)'; the loaded "
                    + "model is '\(modelID)' — a reader is a per-model measurement "
                    + "instrument, re-fit it for this model")
        }
    }

    /// The paper's inference, exactly: render the SAME template around each
    /// new stimulus under the SAME rendering the fit used, capture the LAT
    /// token at the reader's layer, normalize with the training parameters
    /// (inside the probe), and project. Not cosine-to-vector.
    ///
    /// A template-pair reader scores under its EXPERIMENTAL (T+) instruction:
    /// the direction was fitted as H(T+) − H(T−), so the T+ rendering is the
    /// one its probe's center and scale were calibrated on.
    public static func scoreTexts(
        container: ModelContainer, modelID: String,
        reader: Artifact, texts: [String]
    ) async throws -> [Float] {
        try validateForScoring(reader: reader, modelID: modelID)
        let rendering = reader.resolvedExtractionRendering
        let rendered = try texts.map {
            try renderScaffold(
                template: reader.template, stimulus: $0,
                concept: reader.concept, modelID: modelID,
                instruction: reader.template.instructionPair?.experimental,
                rendering: rendering)
        }
        let captured = try await ConceptExtractor.activations(
            container: container, texts: rendered,
            position: reader.readingPosition(), rendering: rendering)
        return try captured.values.map { values in
            guard reader.layer < values.count else {
                throw ReaderError(
                    reason: "reader layer \(reader.layer) out of range for a "
                        + "\(values.count)-layer capture — wrong model?")
            }
            return try scoreActivation(reader, activation: values[reader.layer])
        }
    }

    public static func scoreText(
        container: ModelContainer, modelID: String,
        reader: Artifact, text: String
    ) async throws -> Float {
        try await scoreTexts(
            container: container, modelID: modelID, reader: reader, texts: [text])[0]
    }

    // MARK: - Derive-steering conversion (REPE-IMPLEMENTATION-BRIEF §6)

    /// Pure half of the reader → steering-vector conversion: the reading
    /// direction **in reading orientation** placed at the reader's layer
    /// (zeros below, Gemma-Scope import convention) plus a sidecar stamping
    /// `source: repe-reader-lat`, the reader file id + SHA-256, and the honest
    /// `controlMode` — because "we steered with a RepE reader direction" is a
    /// different claim from "we reproduced RepE control".
    ///
    /// **The orientation is applied here (audit finding 1, fixed 2026-08-27).**
    /// `ScalarProbe.score` computes `orientation · (a·direction − center)`, so
    /// the stored `direction` points at "more concept" only when
    /// `orientation == +1`. Half the readers a study fits — every one where
    /// PC1 came out anti-aligned with the positive class — carry
    /// `orientation == −1`, and shipping the raw direction as a steering
    /// vector injected the concept BACKWARDS while every provenance stamp said
    /// forwards. A steering vector has no orientation field to carry the sign
    /// in, so the sign must be folded into the bytes.
    public static func deriveSteeringArtifact(
        from reader: Artifact, readerFileName: String, readerBytes: Data
    ) throws -> (vectors: ConceptVectors, sidecar: SteeringVectorSidecar) {
        let probeDirection = reader.probe.direction
        guard !probeDirection.isEmpty else {
            throw ReaderError(reason: "reader probe has an empty direction")
        }
        let orientation = reader.probe.orientation
        let direction = orientation < 0 ? probeDirection.map { -$0 } : probeDirection
        let hidden = direction.count
        var perLayer = [[Float]](
            repeating: [Float](repeating: 0, count: hidden), count: reader.layer)
        perLayer.append(direction)
        let vectors = ConceptVectors(perLayer: perLayer)
        var sidecar = SteeringVectorSidecar(
            modelID: reader.modelID,
            revision: reader.revision,
            concept: reader.concept,
            stimulusSetHash: reader.datasetHash,
            vectors: vectors,
            appliedReadingPosition: try reader.readingPosition())
        sidecar.extractionMethod = ExtractionMethod.repeReaderLAT.rawValue
        sidecar.recipeMethod = VectorExtractionRecipe.Method.repeReaderLAT.rawValue
        sidecar.source = artifactType
        sidecar.readerID = readerFileName
        sidecar.readerHash = sha256Hex(readerBytes)
        sidecar.controlMode = controlMode
        sidecar.readerLayer = reader.layer
        sidecar.readerTemplateID = reader.templateID
        sidecar.readerTemplateHash = reader.templateHash
        sidecar.readerContrastMode = reader.contrastMode.rawValue
        sidecar.readerSignConvention = reader.signConvention.rawValue
        sidecar.readerProbeOrientation = orientation
        sidecar.signConvention = reader.signConvention.rawValue
        if let rendering = reader.extractionRendering {
            sidecar.extractionRendering = rendering.stamp
        }
        return (vectors, sidecar)
    }

    /// Converts a saved reader into a standard steering-vector artifact in
    /// `runDirectory` (`<name>.safetensors` + `<name>.json`). Returns the
    /// artifact base URL (`<runDirectory>/<name>`).
    @discardableResult
    public static func deriveSteeringVector(
        readerURL: URL, into runDirectory: URL, name: String? = nil
    ) throws -> URL {
        guard FileManager.default.fileExists(atPath: readerURL.path) else {
            throw ReaderError(reason: "missing reader artifact: \(readerURL.path)")
        }
        let bytes = try Data(contentsOf: readerURL)
        let reader: Artifact
        do {
            reader = try JSONDecoder().decode(Artifact.self, from: bytes)
        } catch let error as ReaderError {
            throw error
        } catch {
            throw ReaderError(
                reason: "\(readerURL.path): unreadable reader artifact: \(error)")
        }
        let (vectors, sidecar) = try deriveSteeringArtifact(
            from: reader, readerFileName: readerURL.lastPathComponent, readerBytes: bytes)
        let name = name ?? "\(reader.concept)-repe-reader"
        try SteeringVectorStore.save(
            vectors: vectors, sidecar: sidecar, to: runDirectory, name: name)
        return runDirectory.appending(component: name)
    }
}
