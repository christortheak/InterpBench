import MLX
import MLXLMCommon
import Synchronization

/// A reading position that cannot be honored for a given sequence.
///
/// Typed and carrying a repair, per house style. Never clamped: silently
/// reading token 0 because the caller asked for "7 back from the end" of a
/// 5-token stimulus is the class of quiet substitution the freeze discipline
/// exists to forbid.
public struct ReadingPositionError: Error, CustomStringConvertible {
    public let reason: String
    public init(reason: String) { self.reason = reason }
    public var description: String { reason }
}

/// Where in the stimulus the residual stream is read during extraction
/// (METHODS.md › Method options).
///
/// MAINTAINER RULING (recorded here because it is the whole point of the
/// named roles): **raw negative indices are the MECHANISM, named roles are
/// the PORTABLE FORM.** `offsetFromEnd(3)` says "three back from the end" and
/// nothing more — on Gemma 3 under the chat template that token is
/// `<start_of_turn>`, on Llama-3 it is something else entirely, and on a raw
/// stimulus it is a word. A study should therefore prefer `lastContentToken`,
/// `turnCloseToken`, and `postInstruction`, which name the thing being read
/// and resolve to whatever concrete index the model family's template puts it
/// at. `offsetFromEnd` exists so an arbitrary offset is DECLARABLE rather than
/// smuggled in, not because it travels.
///
/// Two axes, deliberately separate: **shape-only positions** (`lastToken`,
/// `meanFromToken`, `offsetFromEnd`) resolve from the sequence LENGTH alone
/// and mean the same thing under any rendering; **template-aware roles**
/// (`lastContentToken`, `turnCloseToken`, `postInstruction`) resolve against
/// the TOKEN IDS and refuse under raw rendering, because a raw stimulus has
/// no turn-close token to find.
///
/// Server twin: `Server/steerlab_server/steering/reading_position.py`. The
/// `label` strings are the sidecar contract and must match it byte-for-byte.
public enum ReadingPosition: Codable, Sendable, Equatable {
    /// Hidden state of the final token (RepE's convention; Phase 0 default).
    case lastToken
    /// Mean over token positions from `k` onward (the emotion paper pools
    /// from token 50 of paragraph stories). Extraction callers must enforce
    /// that token `k` exists; otherwise this reading position is not valid.
    case meanFromToken(Int)
    /// The single token at index `last − k`, `k ≥ 0`. `k = 0` is exactly
    /// `.lastToken` — same index, same value — and the recipe identity
    /// canonicalizes it as such. The MECHANISM form; prefer a named role.
    case offsetFromEnd(Int)
    /// The last token of the stimulus's own content: the token immediately
    /// before the marker that closes its turn. Under a generation prompt the
    /// sequence's own last token is template, not content (on Gemma 3 it is a
    /// bare newline), so this names the boundary a study usually means by
    /// "read the end of the prompt".
    case lastContentToken
    /// The template's end-of-turn token — Gemma 3's `<end_of_turn>`, ChatML's
    /// `<|im_end|>`, Llama-3's `<|eot_id|>` — taken as the LAST such marker
    /// in the sequence.
    case turnCloseToken
    /// Arditi's post-instruction convention: the `i`-th token AFTER the end
    /// of the instruction content, `i ∈ 1...5`. Those positions are the
    /// template's own trailing tokens; naming the role is what makes the
    /// convention portable instead of a hard-coded `−5`.
    case postInstruction(Int)
    /// The single token at `lastContentToken − k`, `k ≥ 0` — the CONTENT
    /// -coordinate sibling of `offsetFromEnd`. `offsetFromEnd` counts from the
    /// end of the SEQUENCE, so under a generation prompt its first few
    /// positions are template scaffolding and how many is a fact about the
    /// family; `contentOffset(2)` says "two tokens back into what the stimulus
    /// itself said" and lands in the same place everywhere. `k = 0` is exactly
    /// `.lastContentToken`, and the recipe identity canonicalizes it as such.
    case contentOffset(Int)
    /// Mean over CONTENT token positions from the `n`-th content token on —
    /// the masked sibling of `meanFromToken`, which under a chat template
    /// would pool the stimulus and the template's structure together. Under
    /// RAW rendering every token is content, so it resolves exactly as
    /// `meanFromToken(n)` rather than refusing; the stamp says so.
    case meanContentFromToken(Int)

    public var label: String {
        switch self {
        case .lastToken: "last token"
        case .meanFromToken(let k): "mean from token \(k)"
        case .offsetFromEnd(let k): "offset from end \(k)"
        case .lastContentToken: "last content token"
        case .turnCloseToken: "turn close token"
        case .postInstruction(let i): "post-instruction \(i)"
        case .contentOffset(let k): "content offset \(k)"
        case .meanContentFromToken(let n): "mean content from token \(n)"
        }
    }

    /// Minimum token count required for this reading position to mean what it
    /// says. `meanFromToken(50)` needs token index 50 to exist. For a named
    /// role the real bound is not a length — it is "the template put this
    /// anchor in the sequence" — which `resolve` enforces against the ids.
    public var minimumTokenCount: Int {
        switch self {
        case .lastToken: 1
        case .meanFromToken(let k): max(1, k + 1)
        case .offsetFromEnd(let k): max(1, k + 1)
        case .lastContentToken, .turnCloseToken, .postInstruction: 1
        case .contentOffset: 1
        // The honest bound under RAW rendering, and a necessary (not
        // sufficient) one under a template, where the real requirement is
        // "the content is longer than n" — enforced by `resolve`, which
        // refuses rather than clamps.
        case .meanContentFromToken(let n): max(1, n + 1)
        }
    }

    /// The first position of a POOLED read, for callers that bank rows from
    /// there (the neutral token bank). nil for every single-token read —
    /// `offsetFromEnd` deliberately included, because `k` counts from the end
    /// and answering with it would bank the wrong rows.
    public var requestedStartIndex: Int? {
        switch self {
        case .meanFromToken(let k): k
        case .lastToken, .offsetFromEnd, .lastContentToken, .turnCloseToken,
            .postInstruction, .contentOffset:
            nil
        // DELIBERATELY nil: `n` counts in CONTENT coordinates, which the
        // neutral token bank's raw index space cannot express — answering
        // with n would bank rows from the wrong place under any template.
        case .meanContentFromToken: nil
        }
    }

    /// True for roles that only exist inside a rendered chat turn.
    ///
    /// `meanContentFromToken` is deliberately absent: under raw rendering
    /// every token IS content, so the position is perfectly meaningful there
    /// (and equal to `meanFromToken(n)`) — refusing would be pedantry.
    public var requiresTemplatedRendering: Bool {
        switch self {
        case .lastContentToken, .turnCloseToken, .postInstruction,
            .contentOffset:
            true
        case .lastToken, .meanFromToken, .offsetFromEnd, .meanContentFromToken:
            false
        }
    }

    /// The canonical mode token this position contributes to the recipe
    /// identity (`RecipeIdentity.canonicalReading` applies the
    /// `offsetFromEnd(0) ≡ lastToken` rule on top).
    public var identityMode: String {
        switch self {
        case .lastToken: "lastToken"
        case .meanFromToken: "meanFromToken"
        case .offsetFromEnd: "offsetFromEnd"
        case .lastContentToken: "lastContentToken"
        case .turnCloseToken: "turnCloseToken"
        case .postInstruction: "postInstruction"
        case .contentOffset: "contentOffset"
        case .meanContentFromToken: "meanContentFromToken"
        }
    }

    public var identityParameter: Int? {
        switch self {
        case .meanFromToken(let k): k
        case .offsetFromEnd(let k): k
        case .postInstruction(let i): i
        case .contentOffset(let k): k
        case .meanContentFromToken(let n): n
        case .lastToken, .lastContentToken, .turnCloseToken: nil
        }
    }

    /// Inverse of `label`: parse a sidecar's stamped reading-position string
    /// back into a position. Used by jobs that must re-measure at an
    /// artifact's recorded position (e.g. residual-norm backfill). Nil for
    /// unrecognized labels — callers must fail loudly, not guess.
    public init?(label: String) {
        switch label {
        case ReadingPosition.lastToken.label: self = .lastToken
        case ReadingPosition.lastContentToken.label: self = .lastContentToken
        case ReadingPosition.turnCloseToken.label: self = .turnCloseToken
        default:
            func number(after prefix: String) -> Int? {
                guard label.hasPrefix(prefix) else { return nil }
                let rest = label.dropFirst(prefix.count)
                guard !rest.isEmpty, rest.allSatisfy(\.isNumber) else { return nil }
                return Int(rest)
            }
            // "mean content from token k" is checked FIRST only for
            // readability: it does not carry the "mean from token " prefix, so
            // the two can never be confused.
            if let n = number(after: "mean content from token ") {
                self = .meanContentFromToken(n)
            } else if let k = number(after: "mean from token ") {
                self = .meanFromToken(k)
            } else if let k = number(after: "offset from end ") {
                self = .offsetFromEnd(k)
            } else if let k = number(after: "content offset ") {
                self = .contentOffset(k)
            } else if let i = number(after: "post-instruction "),
                (1 ... 5).contains(i)
            {
                self = .postInstruction(i)
            } else {
                return nil
            }
        }
    }
}

/// What was ACTUALLY read, for one sequence.
///
/// `startIndex`/`endIndex` are a half-open window, so a single-token read is
/// `(i, i + 1)` and a pooled read is `(k, n)`. Artifacts stamp this alongside
/// the requested position so a reader can see what happened without
/// re-deriving template internals — the same both-halves discipline
/// `layerResolution` follows (layer AND depth fraction AND the rule).
public struct ResolvedReadingPosition: Sendable, Equatable {
    public let requested: String
    public let mode: String
    public let parameter: Int?
    public let startIndex: Int
    public let endIndex: Int
    public let tokenCount: Int
    /// How the index was derived, in words ("sequence end", "last content
    /// token + 2", …). The provenance half of the stamp.
    public let source: String
    /// A CONTENT-MASKED pool's exact positions, or nil for the contiguous
    /// `startIndex ..< endIndex` window every other position reads. The window
    /// still bounds it (first pooled index → last pooled index + 1), so every
    /// consumer that only understands windows still sees a truthful span; the
    /// recorder reads exactly these positions.
    public let pooledIndices: [Int]?
    /// How many positions of the whole sequence THE MASK called structure, or
    /// nil for a read that masks nothing at all. Zero is meaningful (under raw
    /// rendering every token is content), so nil and 0 say different things.
    /// Deliberately NOT "everything not pooled": the tokens a
    /// `meanContentFromToken(n)` skips before its n-th content token were
    /// excluded by the REQUEST, not by the mask.
    public let maskedTokenCount: Int?

    public init(
        requested: String, mode: String, parameter: Int?, startIndex: Int,
        endIndex: Int, tokenCount: Int, source: String,
        pooledIndices: [Int]? = nil, maskedTokenCount: Int? = nil
    ) {
        self.requested = requested
        self.mode = mode
        self.parameter = parameter
        self.startIndex = startIndex
        self.endIndex = endIndex
        self.tokenCount = tokenCount
        self.source = source
        self.pooledIndices = pooledIndices
        self.maskedTokenCount = maskedTokenCount
    }

    public var isPooled: Bool { endIndex - startIndex > 1 }

    /// How many positions were actually averaged.
    public var pooledTokenCount: Int {
        pooledIndices?.count ?? (endIndex - startIndex)
    }

    /// Distance of the read position from the sequence end, or nil for a
    /// pooled read. This is the SEQUENCE-SHAPE invariant: under a fixed
    /// template the offset is constant while the absolute index moves with
    /// stimulus length.
    public var offsetFromEnd: Int? {
        isPooled ? nil : tokenCount - 1 - startIndex
    }
}

/// The stamped `readingPositionResolution` block.
///
/// Records BOTH halves: the REQUESTED position (name + parameter) and what it
/// RESOLVED to — mirroring the way `layerResolution` records the layer
/// together with its depth fraction and the rule that chose it. A reader can
/// then see what was actually read without re-deriving template internals.
///
/// Resolutions are grouped by OFFSET FROM END, the sequence-shape invariant:
/// under one template the offset is constant while the absolute index moves
/// with stimulus length, so a single row means "every stimulus was read at the
/// same place in its template", and two rows are a loud sign the template did
/// not render uniformly.
///
/// Server twin: `extractor.resolution_report`.
public struct ReadingPositionResolutionReport: Codable, Sendable, Equatable {

    public struct Shape: Codable, Sendable, Equatable {
        public var offsetFromEnd: Int?
        public var sequenceCount: Int
        public var exampleIndex: Int
        public var exampleEndIndex: Int
        public var exampleTokenCount: Int
        /// Content-masked pools only (nil elsewhere, so no other position's
        /// stamped bytes move): how many positions were averaged, and how
        /// many the mask called structure. EXAMPLES, like every other
        /// `example*` field — they vary with stimulus length, and grouping by
        /// them would turn one honest "every stimulus read the same way" row
        /// into one row per stimulus.
        public var examplePooledTokenCount: Int?
        public var exampleMaskedTokenCount: Int?
    }

    public var requested: String
    public var mode: String
    /// Encoded only when the position takes one, so a parameterless role
    /// stamps no null.
    public var parameter: Int?
    public var rendering: String
    public var source: String
    public var shapes: [Shape]

    /// The report for one extraction pass, or nil to OMIT the key.
    ///
    /// nil for the legacy pair (a shape-only position under raw rendering),
    /// whose resolved index is fully implied by its label — so legacy
    /// artifacts keep byte-identical sidecars.
    public static func make(
        position: ReadingPosition,
        rendering: ExtractionRendering,
        resolutions: [ResolvedReadingPosition]
    ) -> ReadingPositionResolutionReport? {
        let stampedModes: Set<String> = [
            "offsetFromEnd", "lastContentToken", "turnCloseToken", "postInstruction",
            "contentOffset", "meanContentFromToken",
        ]
        guard !rendering.isRaw || stampedModes.contains(position.identityMode) else {
            return nil
        }
        guard let first = resolutions.first else { return nil }

        var order: [Int?] = []
        var byOffset: [Int: Shape] = [:]
        var pooled: Shape?
        for resolved in resolutions {
            let key = resolved.offsetFromEnd
            let fresh = Shape(
                offsetFromEnd: key, sequenceCount: 1,
                exampleIndex: resolved.startIndex,
                exampleEndIndex: resolved.endIndex,
                exampleTokenCount: resolved.tokenCount,
                examplePooledTokenCount: resolved.maskedTokenCount == nil
                    ? nil : resolved.pooledTokenCount,
                exampleMaskedTokenCount: resolved.maskedTokenCount)
            if let key {
                if byOffset[key] == nil {
                    byOffset[key] = fresh
                    order.append(key)
                } else {
                    byOffset[key]?.sequenceCount += 1
                }
            } else if pooled == nil {
                pooled = fresh
                order.append(nil)
            } else {
                pooled?.sequenceCount += 1
            }
        }
        // Sorted by offset, with the pooled row (no single offset) last —
        // matching the server's `(offsetFromEnd is None, offsetFromEnd)` key.
        var shapes = order.compactMap { $0 }.sorted().compactMap { byOffset[$0] }
        if let pooled { shapes.append(pooled) }

        return ReadingPositionResolutionReport(
            requested: position.label,
            mode: position.identityMode,
            parameter: position.identityParameter,
            rendering: rendering.mode.rawValue,
            source: first.source,
            shapes: shapes)
    }
}

// MARK: - The WRITER: a reading position declared on the command line

/// Pinning a reading position into a study.
///
/// The gap this closes (2026-08-25) is the `extractionRendering` gap one layer
/// down: the vocabulary above had a manifest parser, a recipe identity and
/// sidecar stamps, and no way to PIN anything but `.lastToken` and
/// `.meanFromToken` — `attach` took a `--pool-from` token index and nothing
/// else, so every named role was reachable only from an ad-hoc extract call
/// that pins nothing. A grid that varies the reading position has to run
/// study-disciplined, so the writer exists. Server twin:
/// `reading_position.parse_declaration`.
extension ReadingPosition {

    /// The flag both engines' CLIs pin a reading position with. One constant,
    /// so a help page, a refusal and a parser cannot name three flags.
    public static let declarationFlag = "--reading-position"

    /// The LEGACY spelling for one specific position (`mean from token k`).
    /// It predates the label vocabulary and stays, because studies and scripts
    /// type it — but the two may never be declared together.
    public static let poolFromFlag = "--pool-from"

    /// Every spelling `declared(_:)` accepts, as a person would type it. This
    /// IS the cross-engine vocabulary (the committed fixture
    /// `extraction-rendering-and-positions.json` pins the concrete labels),
    /// and it is what an unknown-label refusal lists. Server twin:
    /// `reading_position.DECLARABLE_LABELS`.
    public static let declarableLabels = [
        "last token",
        "mean from token <k>",
        "offset from end <k>",
        "last content token",
        "turn close token",
        "post-instruction <i>  (i in 1..5)",
        "content offset <k>",
        "mean content from token <n>",
    ]

    /// This engine's name in a refusal about an undeclarable position.
    public static let declarationEngine = "swift-mlx"

    /// A declaration this engine will not accept. Typed and carrying a
    /// repair, per house style — the CLI turns it into a malformed-invocation
    /// refusal (exit 64: nothing ran, and retrying cannot help).
    public struct DeclarationError: Error, CustomStringConvertible, Equatable {
        public let reason: String
        public let repair: String
        public init(reason: String, repair: String) {
            self.reason = reason
            self.repair = repair
        }
        public var description: String { reason }
    }

    /// Parse what was typed after `--reading-position`.
    ///
    /// STRICT: an unrecognized label is a typed refusal naming the vocabulary,
    /// never a fall back to last-token. `init?(label:)`'s tolerance exists for
    /// READING old artifacts; a WRITER that guessed would pin a recipe nobody
    /// asked for. nil for an absent declaration, which keeps the recipe's
    /// default — and therefore its identity — exactly where it was.
    public static func declared(_ text: String?) throws -> ReadingPosition? {
        guard let text else { return nil }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw DeclarationError(
                reason: "\(declarationFlag) was given no value",
                repair: "name a reading position, e.g. \(declarationFlag) "
                    + "'last content token' — or omit the flag, which keeps "
                    + "the recipe's default")
        }
        guard let position = ReadingPosition(label: trimmed) else {
            throw DeclarationError(
                reason: "reading position '\(trimmed)' is not one the "
                    + "\(declarationEngine) engine knows",
                repair: "declare one of "
                    + declarableLabels.joined(separator: "; "))
        }
        return position
    }

    /// The refusal for declaring a reading position in BOTH spellings, or nil.
    ///
    /// `--pool-from K` is exactly `--reading-position 'mean from token K'`;
    /// accepting both would mean silently picking one, and which one it picked
    /// would be the recipe. Twin text on the server
    /// (`reading_position.declaration_conflict`).
    public static func declarationConflict(
        _ declaration: String?, poolFromToken: Int?
    ) -> String? {
        guard let declaration, let poolFromToken else { return nil }
        return "reading position declared twice: \(declarationFlag) "
            + "'\(declaration)' and \(poolFromFlag) \(poolFromToken) name two "
            + "recipes, and a concept pins exactly one — repair: drop "
            + "\(poolFromFlag) (\(declarationFlag) 'mean from token "
            + "\(poolFromToken)' is its spelling), or drop \(declarationFlag)"
    }

    /// End-of-turn markers probed by NAME, in addition to whatever the
    /// tokenizer declares as its eos. Cross-engine contract: the server twin
    /// (`reading_position.END_OF_TURN_TOKENS`) probes the same strings in the
    /// same order.
    public static let endOfTurnTokens = [
        "<end_of_turn>",    // Gemma 3
        "<|im_end|>",       // Qwen / ChatML
        "<|eot_id|>",       // Llama 3
        "<|end|>",          // Phi
        "</s>",             // Llama 2 / Mistral
    ]

    /// Turn-OPEN markers, the symmetric half of `endOfTurnTokens` and the
    /// other half of the one template map. Probed by NAME in the same way and
    /// the same order as the server twin
    /// (`reading_position.TURN_OPEN_TOKENS`). Llama 3 contributes both header
    /// markers: the role tag lives BETWEEN them, so both are structure.
    public static let turnOpenTokens = [
        "<start_of_turn>",      // Gemma 3
        "<|im_start|>",         // Qwen / ChatML
        "<|start_header_id|>",  // Llama 3
        "<|end_header_id|>",    // Llama 3
        "<|user|>",             // Phi
        "<|assistant|>",        // Phi
    ]

    /// Sequence-level markers that are structure wherever they appear (server
    /// twin: `reading_position.STRUCTURAL_TOKENS`).
    public static let structuralTokens = [
        "<bos>",                // Gemma 3
        "<|begin_of_text|>",    // Llama 3
        "<s>",                  // Llama 2 / Mistral
        "<|endoftext|>",        // GPT-2 / Qwen padding
    ]

    /// How far past a turn-open marker the ROLE TAG may run before the mask
    /// gives up looking for its newline. Every vendored family writes
    /// `<open><role>\n` in at most three tokens; the cap keeps a tokenizer
    /// that never emits a newline token from eating the stimulus.
    static let roleTagMaxTokens = 4

    static func ids(
        forTokens names: [String], tokenizer: any MLXLMCommon.Tokenizer
    ) -> Set<Int> {
        var ids = Set<Int>()
        let unknown = tokenizer.unknownTokenId
        for token in names {
            if let id = tokenizer.convertTokenToId(token), id >= 0, id != unknown {
                ids.insert(id)
            }
        }
        return ids
    }

    static func endOfTurnIDs(tokenizer: any MLXLMCommon.Tokenizer) -> Set<Int> {
        var ids = self.ids(forTokens: endOfTurnTokens, tokenizer: tokenizer)
        if let eos = tokenizer.eosTokenId, eos >= 0 { ids.insert(eos) }
        return ids
    }

    /// Every id THE TEMPLATE MAP calls structure: turn-open markers,
    /// turn-close markers (the tokenizer's declared eos included), and the
    /// sequence-level markers. The same inventory the named roles resolve
    /// against, read as a set instead of scanned for a boundary — one
    /// definition, two uses.
    static func structuralIDs(tokenizer: any MLXLMCommon.Tokenizer) -> Set<Int> {
        ids(forTokens: turnOpenTokens, tokenizer: tokenizer)
            .union(endOfTurnIDs(tokenizer: tokenizer))
            .union(ids(forTokens: structuralTokens, tokenizer: tokenizer))
    }

    /// The vocabulary PIECE for one id, with the two BPE surface conventions
    /// normalized (`Ċ` is a newline, `Ġ`/`▁` are spaces). Used only to find
    /// the newline that ends a template's role tag; a tokenizer that answers
    /// nothing yields "", and the caller then leaves the tag unmasked rather
    /// than guessing.
    static func tokenPiece(
        _ id: Int, tokenizer: any MLXLMCommon.Tokenizer
    ) -> String {
        guard let piece = tokenizer.convertIdToToken(id) else { return "" }
        return piece.replacingOccurrences(of: "Ċ", with: "\n")
            .replacingOccurrences(of: "Ġ", with: " ")
            .replacingOccurrences(of: "▁", with: " ")
    }

    /// The positions of the stimulus's OWN CONTENT in a rendered sequence.
    ///
    /// THE MASK, defined once and used by every content-coordinate position.
    /// It derives entirely from the template map the named roles already use:
    ///
    /// 1. the read turn CLOSES at `turnCloseIndex` — the last end-of-turn
    ///    marker — and content cannot follow it, which is what excludes the
    ///    trailing generation scaffold (on Gemma 3 that scaffold ends in a
    ///    bare newline, an ordinary token no special-token test would catch);
    /// 2. the read turn OPENS at the last `turnOpenTokens` marker before that
    ///    close (position 0 when a template writes none);
    /// 3. the template's ROLE TAG — `<start_of_turn>` `model` `\n`, or
    ///    ChatML's `<|im_start|>assistant\n` — is structure, and every
    ///    vendored family terminates it with a newline, so the tokens up to
    ///    and including the first newline piece after the open marker are
    ///    dropped (bounded by `roleTagMaxTokens`; a tokenizer that reports no
    ///    newline keeps its tag rather than losing stimulus tokens to a
    ///    guess);
    /// 4. anything still inside that the map calls structure is dropped.
    ///
    /// Server twin: `reading_position.content_indices`.
    static func contentIndices(
        tokens: [Int], tokenizer: any MLXLMCommon.Tokenizer, label: String
    ) throws -> [Int] {
        let close = try turnCloseIndex(
            tokens: tokens, tokenizer: tokenizer, label: label)
        guard close > 0 else {
            throw ReadingPositionError(
                reason: "reading position '\(label)' did not resolve: the turn "
                    + "closes at token 0, so the stimulus contributed no content "
                    + "— repair: check the stimulus is non-empty and that the "
                    + "rendering is the one you meant")
        }
        let opens = ids(forTokens: turnOpenTokens, tokenizer: tokenizer)
        var start = 0
        for index in stride(from: close - 1, through: 0, by: -1)
        where opens.contains(tokens[index]) {
            start = index + 1
            for offset in 0 ..< min(roleTagMaxTokens, close - start)
            where tokenPiece(tokens[start + offset], tokenizer: tokenizer)
                .contains("\n")
            {
                start += offset + 1
                break
            }
            break
        }
        let structural = structuralIDs(tokenizer: tokenizer)
        let content = (start ..< close).filter { !structural.contains(tokens[$0]) }
        guard !content.isEmpty else {
            throw ReadingPositionError(
                reason: "reading position '\(label)' did not resolve: the "
                    + "rendered turn (\(tokens.count) tokens) carries no content "
                    + "tokens — every position between the turn's opening and "
                    + "its close is template structure — repair: check the "
                    + "stimulus is non-empty and that the rendering is the one "
                    + "you meant")
        }
        return content
    }

    /// Index of the LAST end-of-turn marker in the sequence.
    ///
    /// ONE mechanism for all three named roles, and one both engines have. A
    /// "last token that is not special" rule would have been engine-specific
    /// AND wrong — on Gemma 3 the generation prompt ends in a bare newline, an
    /// ordinary content token, so a special-token scan lands past the
    /// instruction rather than inside it.
    static func turnCloseIndex(
        tokens: [Int], tokenizer: any MLXLMCommon.Tokenizer, label: String
    ) throws -> Int {
        let markers = endOfTurnIDs(tokenizer: tokenizer)
        guard !markers.isEmpty else {
            throw ReadingPositionError(
                reason: "reading position '\(label)' did not resolve on the "
                    + "swift-mlx engine: this tokenizer declares no end-of-turn "
                    + "token (looked for \(endOfTurnTokens.joined(separator: ", ")) "
                    + "and the tokenizer's own eos) — repair: read at an explicit "
                    + "'offset from end k', which needs no template anchor")
        }
        for index in stride(from: tokens.count - 1, through: 0, by: -1)
        where markers.contains(tokens[index]) {
            return index
        }
        throw ReadingPositionError(
            reason: "reading position '\(label)' did not resolve: the rendered "
                + "sequence (\(tokens.count) tokens) contains no end-of-turn "
                + "marker — repair: check that extractionRendering really is "
                + "'chatTemplate' for this concept, or read at an explicit "
                + "'offset from end k'")
    }

    static func lastContentIndex(
        tokens: [Int], tokenizer: any MLXLMCommon.Tokenizer, label: String
    ) throws -> Int {
        let close = try turnCloseIndex(tokens: tokens, tokenizer: tokenizer, label: label)
        guard close > 0 else {
            throw ReadingPositionError(
                reason: "reading position '\(label)' did not resolve: the turn "
                    + "closes at token 0, so the stimulus contributed no content "
                    + "— repair: check the stimulus is non-empty and that the "
                    + "rendering is the one you meant")
        }
        return close - 1
    }

    func requireTemplated(renderingIsRaw: Bool) throws {
        guard renderingIsRaw, let refusal = Self.templatedRenderingRefusal(self)
        else { return }
        throw ReadingPositionError(reason: refusal)
    }

    /// The refusal for a template-aware role under RAW rendering, or nil when
    /// the position has no such dependency.
    ///
    /// ONE sentence, TWO moments. It was written for resolution time, where a
    /// role that needs a chat turn meets a raw stimulus; since 2026-08-25 the
    /// ATTACH path asks the same question at DECLARATION time — a pin that
    /// could never resolve is answered while the person is typing, not hours
    /// later on a GPU (the `addGenerationPrompt: false` precedent). Hoisting
    /// it here is what keeps the two moments from drifting into two
    /// explanations. Server twin:
    /// `reading_position.templated_rendering_refusal`.
    public static func templatedRenderingRefusal(
        _ position: ReadingPosition
    ) -> String? {
        guard position.requiresTemplatedRendering else { return nil }
        return "reading position '\(position.label)' needs templated "
            + "rendering: a raw stimulus has no chat turn, so there is no turn "
            + "boundary to read at — repair: re-attach the concept with "
            + "\(ExtractionRendering.declarationFlag) "
            + "'{\"mode\": \"chatTemplate\"}', or choose a "
            + "rendering-independent position ('last token', 'offset from "
            + "end k', 'mean from token k')"
    }

    /// The concrete read window for one tokenized stimulus.
    ///
    /// `tokens` is the sequence actually fed to the model — already rendered,
    /// so a templated stimulus resolves against the template's own tokens.
    /// Throws `ReadingPositionError` (typed, with a repair) rather than
    /// clamping.
    public func resolve(
        tokens: [Int], tokenizer: any MLXLMCommon.Tokenizer, renderingIsRaw: Bool
    ) throws -> ResolvedReadingPosition {
        try requireTemplated(renderingIsRaw: renderingIsRaw)
        let n = tokens.count

        func single(_ index: Int, _ source: String) -> ResolvedReadingPosition {
            ResolvedReadingPosition(
                requested: label, mode: identityMode, parameter: identityParameter,
                startIndex: index, endIndex: index + 1, tokenCount: n,
                source: source)
        }

        switch self {
        case .lastToken:
            guard n >= 1 else { throw tooShort(have: n) }
            return single(n - 1, "sequence end")

        case .meanFromToken(let k):
            guard n >= minimumTokenCount else { throw tooShort(have: n) }
            // Historical clamp preserved EXACTLY: this position has always
            // pooled from min(max(0, k), n − 1).
            let start = min(max(0, k), n - 1)
            return ResolvedReadingPosition(
                requested: label, mode: identityMode, parameter: k,
                startIndex: start, endIndex: n, tokenCount: n,
                source: "token \(k) through sequence end")

        case .offsetFromEnd(let k):
            guard k >= 0 else {
                throw ReadingPositionError(
                    reason: "offsetFromEnd needs k ≥ 0, got \(k) — repair: "
                        + "offsets count BACKWARD from the last token, so k=0 is "
                        + "the last token and k=3 is three before it")
            }
            guard n >= minimumTokenCount else { throw tooShort(have: n) }
            return single(n - 1 - k, "sequence end − \(k)")

        case .lastContentToken:
            let index = try Self.lastContentIndex(
                tokens: tokens, tokenizer: tokenizer, label: label)
            return single(index, "token before the last end-of-turn marker")

        case .turnCloseToken:
            let index = try Self.turnCloseIndex(
                tokens: tokens, tokenizer: tokenizer, label: label)
            return single(index, "last end-of-turn marker")

        case .postInstruction(let i):
            guard (1 ... 5).contains(i) else {
                throw ReadingPositionError(
                    reason: "postInstruction needs i in 1...5, got \(i) — repair: "
                        + "the convention covers the five template tokens that "
                        + "follow the instruction; for anything further out "
                        + "declare an explicit 'offset from end k'")
            }
            let content = try Self.lastContentIndex(
                tokens: tokens, tokenizer: tokenizer, label: label)
            let index = content + i
            guard index < n else {
                throw ReadingPositionError(
                    reason: "reading position '\(label)' did not resolve: the "
                        + "instruction ends at token \(content) of \(n), so there "
                        + "is no token \(i) after it — repair: lower i, or render "
                        + "with addGenerationPrompt true so the template's "
                        + "trailing tokens are present")
            }
            return single(index, "last content token + \(i)")

        case .contentOffset(let k):
            guard k >= 0 else {
                throw ReadingPositionError(
                    reason: "contentOffset needs k ≥ 0, got \(k) — repair: "
                        + "offsets count BACKWARD from the last content token, "
                        + "so k=0 is the last content token and k=3 is three "
                        + "before it")
            }
            let content = try Self.lastContentIndex(
                tokens: tokens, tokenizer: tokenizer, label: label)
            let index = content - k
            guard index >= 0 else {
                throw ReadingPositionError(
                    reason: "reading position '\(label)' did not resolve: the "
                        + "stimulus's content ends at token \(content) of \(n), "
                        + "so there is no token \(k) before it — repair: lower "
                        + "k, or drop the short stimuli (this position is never "
                        + "clamped, because reading the turn's first token "
                        + "instead would silently change the recipe)")
            }
            return single(index, "last content token − \(k)")

        case .meanContentFromToken(let n_):
            guard n_ >= 0 else {
                throw ReadingPositionError(
                    reason: "meanContentFromToken needs n ≥ 0, got \(n_) — "
                        + "repair: n counts FORWARD from the first content "
                        + "token, so n=0 pools the whole content")
            }
            if renderingIsRaw {
                guard n >= minimumTokenCount else { throw tooShort(have: n) }
                // The meanFromToken clamp, preserved exactly: under raw the
                // two positions must read identical rows, or "equivalent to
                // meanFromToken(n)" would be a claim the numbers contradict.
                let start = min(max(0, n_), n - 1)
                return ResolvedReadingPosition(
                    requested: label, mode: identityMode, parameter: n_,
                    startIndex: start, endIndex: n, tokenCount: n,
                    source: "token \(n_) through sequence end (raw rendering: "
                        + "every token is content)",
                    pooledIndices: nil, maskedTokenCount: 0)
            }
            let content = try Self.contentIndices(
                tokens: tokens, tokenizer: tokenizer, label: label)
            guard content.count > n_ else {
                throw ReadingPositionError(
                    reason: "reading position '\(label)' did not resolve: the "
                        + "rendered turn carries \(content.count) content "
                        + "tokens, so there is no content token \(n_) to pool "
                        + "from — repair: lower n, or drop the short stimuli "
                        + "(this position is never clamped, because pooling "
                        + "from the turn's first token instead would silently "
                        + "change the recipe)")
            }
            let pooled = Array(content[n_...])
            return ResolvedReadingPosition(
                requested: label, mode: identityMode, parameter: n_,
                startIndex: pooled[0], endIndex: pooled[pooled.count - 1] + 1,
                tokenCount: n,
                source: "content token \(n_) through the last content token",
                pooledIndices: pooled,
                maskedTokenCount: n - content.count)
        }
    }

    func tooShort(have: Int) -> ReadingPositionError {
        ReadingPositionError(
            reason: "sequence is too short for '\(label)': \(have) tokens, needs "
                + "\(minimumTokenCount) — repair: drop the short stimuli, or "
                + "choose a position that fits them (this position is never "
                + "clamped, because reading a different token instead would "
                + "silently change the recipe)")
    }
}

/// Records the residual stream at configured layers during a forward pass.
///
/// Values are evaluated and copied to CPU inside the hook — capturing lazy
/// GPU arrays for every layer/token explodes unified memory (CLAUDE.md ›
/// MLX gotchas). The eager copy makes this recorder appropriate for
/// extraction passes (one prefill per stimulus, a handful of layers), not
/// for per-token capture over long generations.
///
/// Assumes batch size 1.
public final class ActivationRecorder: LayerIntervention {
    public struct Capture: Sendable {
        /// Transformer block whose output was recorded.
        public let layer: Int
        /// KV-cache position of the first token of the recorded pass
        /// (0 = prefill).
        public let offset: Int
        /// Number of token positions in this forward pass.
        public let tokenCount: Int
        /// First position included in the pooled readout.
        public let startIndex: Int
        /// Hidden vector at the reading position, copied to CPU as float32.
        public let values: [Float]
        /// Mean L2 norm of the residual stream over the positions read —
        /// the "typical residual-stream norm" used to express steering
        /// strength in norm units (the emotion paper's convention).
        public let residualNorm: Float

        public init(
            layer: Int, offset: Int, tokenCount: Int = 0, startIndex: Int = 0,
            values: [Float], residualNorm: Float
        ) {
            self.layer = layer
            self.offset = offset
            self.tokenCount = tokenCount
            self.startIndex = startIndex
            self.values = values
            self.residualNorm = residualNorm
        }
    }

    private let layers: Set<Int>
    private let position: ReadingPosition
    private let storage = Mutex<[Capture]>([])
    /// Half-open read window pinned by the driver for the next forward pass,
    /// or nil for the historical length-only behavior. See `setWindow`.
    private let window = Mutex<(start: Int, end: Int)?>(nil)
    /// A content-masked pool's exact positions, or nil for the contiguous
    /// window. See `setWindow(start:end:indices:)`.
    private let pooledIndices = Mutex<[Int]?>(nil)

    /// - Parameters:
    ///   - layers: block indices to record; record only what you need —
    ///     every capture is an eager GPU→CPU copy.
    ///   - position: where in the sequence to read (default last token).
    public init(layers: some Sequence<Int>, position: ReadingPosition = .lastToken) {
        self.layers = Set(layers)
        self.position = position
    }

    /// All captures so far, in hook-firing order.
    public var captures: [Capture] {
        storage.withLock { $0 }
    }

    public func reset() {
        storage.withLock { $0.removeAll() }
    }

    /// Pin the half-open read window for the NEXT forward pass.
    ///
    /// Template-aware reading positions ("turn close token",
    /// "post-instruction 2") resolve against the TOKEN IDS, which only the
    /// extraction driver holds — the hook sees hidden states. So the driver
    /// resolves and pins the window here, and this recorder stays free of
    /// template knowledge. Unset (the default) keeps the historical
    /// length-only behavior for the two shape-only positions, byte-for-byte.
    /// - Parameter indices: a CONTENT-MASKED pool's exact positions, for a
    ///   reading whose window is not contiguous ("mean content from token n"
    ///   skips the template's structure). nil — every caller before
    ///   2026-08-25 and every position but that one — reads the contiguous
    ///   window exactly as before.
    public func setWindow(start: Int, end: Int, indices: [Int]? = nil) {
        window.withLock { $0 = (start, end) }
        pooledIndices.withLock { $0 = (indices?.isEmpty == false) ? indices : nil }
    }

    public func clearWindow() {
        window.withLock { $0 = nil }
        pooledIndices.withLock { $0 = nil }
    }

    public func apply(_ h: MLXArray, layer: Int, offset: Int) -> MLXArray {
        guard layers.contains(layer) else { return h }

        let length = h.dim(1)
        let startIndex: Int
        let endIndex: Int
        if let pinned = window.withLock({ $0 }) {
            startIndex = pinned.start
            endIndex = min(pinned.end, length)
        } else {
            switch position {
            case .lastToken, .offsetFromEnd, .lastContentToken, .turnCloseToken,
                .postInstruction, .contentOffset:
                // Only the two shape-only cases can reach here unpinned; the
                // rest are always driver-resolved. Last-token is the honest
                // fallback for a hand-built recorder.
                startIndex = length - 1
            case .meanFromToken(let k):
                startIndex = min(max(0, k), length - 1)
            case .meanContentFromToken(let n):
                // Unpinned, this position has no template map to consult, so
                // it reads what it means under RAW rendering — the window
                // `meanFromToken(n)` reads. A driver-resolved pass pins the
                // mask and never reaches here.
                startIndex = min(max(0, n), length - 1)
            }
            endIndex = length
        }

        // rows: [positions, hidden] over the reading window. `h[0, s..<length]`
        // is the same tensor `h[0, s...]` was, so every legacy recipe's
        // pooled numbers are unchanged. A content-masked pool gathers exactly
        // the pinned positions instead; the window still bounds them, so a
        // pass that came out shorter than the resolution expected drops the
        // out-of-range tail rather than indexing past the end.
        var readStart = startIndex
        let rows: MLXArray
        if let masked = pooledIndices.withLock({ $0 }) {
            let kept = masked.filter { $0 < length }
            guard !kept.isEmpty else { return h }
            readStart = kept[0]
            rows = h[0].take(MLXArray(kept.map { Int32($0) }), axis: 0)
                .asType(.float32)
        } else {
            rows = h[0, startIndex ..< endIndex, 0...].asType(.float32)
        }
        let pooled = rows.mean(axis: 0)
        let norms = sqrt(rows.square().sum(axis: -1)).mean()

        // asArray/item force evaluation and copy to CPU promptly.
        let values = pooled.asArray(Float.self)
        let residualNorm = norms.item(Float.self)
        storage.withLock {
            $0.append(
                Capture(
                    layer: layer, offset: offset, tokenCount: length, startIndex: readStart,
                    values: values, residualNorm: residualNorm))
        }
        return h
    }
}

/// Records one residual-stream row per token position, for calibration banks
/// such as Anthropic-style neutral transcript PCA. This is intentionally
/// separate from `ActivationRecorder`, whose contract is one pooled vector per
/// stimulus.
///
/// **Ingestion is bounded, not post-hoc trimmed.** The row cap is applied
/// HERE, as rows arrive, via `selectedRowIndices` — a precomputed set of
/// global bank positions from `TokenBankDownsampler.selectedIndexSet`. Only
/// the selected positions are gathered off the GPU, so the transient CPU
/// float32 copy is `kept × hidden`, never `tokens × hidden`. Collecting every
/// row first and downsampling afterwards (the shape this recorder had until
/// 2026-08-18) makes an all-layer capture over a real corpus a multi-GB ×
/// layers transient, which fails inside MLX/native allocation and kills the
/// process before a Swift `catch` can run (CLAUDE.md › MLX gotchas).
///
/// Positions are counted per layer in ARRIVAL order — text order, then token
/// order — which is exactly the row order the old two-phase downsample
/// indexed, so the streaming filter keeps the identical rows. The cursor
/// therefore persists across `reset()` (which drains rows only); use
/// `resetAll()` to start a fresh bank.
///
/// Residual norms are accumulated for EVERY position regardless of selection:
/// one float per token is cheap, and the norm denominator should describe the
/// whole corpus, not the draw.
public final class ActivationBankRecorder: LayerIntervention {
    public struct Row: Sendable {
        public let layer: Int
        public let offset: Int
        public let tokenIndex: Int
        public let values: [Float]
        public let residualNorm: Float
    }

    /// A residual norm for a position that was NOT banked. Callers fold these
    /// into the per-layer norm average alongside the banked rows.
    public struct SkippedNorm: Sendable {
        public let layer: Int
        public let residualNorm: Float
    }

    private struct Storage {
        var rows: [Row] = []
        var skippedNorms: [SkippedNorm] = []
        /// Next global bank position for each layer. Survives `reset()`.
        var nextPositionByLayer: [Int: Int] = [:]
        /// High-water mark of `rows.count` — the retention instrument the
        /// bounded-ingestion test asserts against.
        var peakRowCount = 0
    }

    private let layers: Set<Int>
    private let startIndex: Int
    private let selectedRowIndices: Set<Int>?
    private let storage = Mutex(Storage())

    /// - Parameters:
    ///   - layers: block indices to record. Restrict this — an all-layer bank
    ///     costs `rows × layers × hidden` float32 even when bounded.
    ///   - startIndex: first token position to bank.
    ///   - selectedRowIndices: global bank positions to keep, or nil to keep
    ///     every position. Pass `TokenBankDownsampler.selectedIndexSet`.
    public init(
        layers: some Sequence<Int>,
        startIndex: Int = 0,
        selectedRowIndices: Set<Int>? = nil
    ) {
        self.layers = Set(layers)
        self.startIndex = max(0, startIndex)
        self.selectedRowIndices = selectedRowIndices
    }

    public var rows: [Row] {
        storage.withLock { $0.rows }
    }

    /// Residual norms for positions the row cap excluded, so the norm average
    /// still covers the whole corpus.
    public var skippedNorms: [SkippedNorm] {
        storage.withLock { $0.skippedNorms }
    }

    /// Largest number of rows this recorder has ever held at once. The
    /// bounded-ingestion guarantee is observable: with a selection set of size
    /// `cap`, this never exceeds `cap × layers` across a whole bank, and
    /// exactly `cap` when the driver drains per pass.
    public var peakRetainedRowCount: Int {
        storage.withLock { $0.peakRowCount }
    }

    /// Drains banked rows and skipped norms. The per-layer position cursor
    /// deliberately survives, so a driver can drain after every forward pass
    /// while the global selection stays aligned.
    public func reset() {
        storage.withLock {
            $0.rows.removeAll()
            $0.skippedNorms.removeAll()
        }
    }

    /// Full reset, cursors included — a new bank over a new corpus.
    public func resetAll() {
        storage.withLock { $0 = Storage() }
    }

    public func apply(_ h: MLXArray, layer: Int, offset: Int) -> MLXArray {
        guard layers.contains(layer) else { return h }
        let length = h.dim(1)
        guard startIndex < length else { return h }
        let positionCount = length - startIndex

        // Claim this pass's slice of the global position space before doing
        // any work, so selection is stable no matter how passes interleave.
        let base = storage.withLock { state -> Int in
            let base = state.nextPositionByLayer[layer] ?? 0
            state.nextPositionByLayer[layer] = base + positionCount
            return base
        }

        let keptPositions: [Int]
        if let selectedRowIndices {
            keptPositions = (0 ..< positionCount).filter {
                selectedRowIndices.contains(base + $0)
            }
        } else {
            keptPositions = Array(0 ..< positionCount)
        }

        // Norms first: one float per position, so this stays cheap even when
        // nothing in this pass is banked.
        let window = h[0, startIndex..., 0...].asType(.float32)
        let residualNorms = sqrt(window.square().sum(axis: -1)).asArray(Float.self)

        // Gather ONLY the kept rows on the GPU before the CPU copy. This is
        // the bound: `values` is kept × hidden, not positionCount × hidden.
        var values: [Float] = []
        if !keptPositions.isEmpty {
            let gathered =
                keptPositions.count == positionCount
                ? window
                : window.take(MLXArray(keptPositions.map { Int32($0) }), axis: 0)
            values = gathered.asArray(Float.self)
        }
        let hidden = h.dim(2)
        let keptSet = keptPositions.isEmpty ? Set<Int>() : Set(keptPositions)

        storage.withLock { state in
            for (slot, position) in keptPositions.enumerated() {
                let lower = slot * hidden
                let upper = lower + hidden
                guard upper <= values.count, position < residualNorms.count else { continue }
                state.rows.append(
                    Row(
                        layer: layer,
                        offset: offset,
                        tokenIndex: startIndex + position,
                        values: Array(values[lower ..< upper]),
                        residualNorm: residualNorms[position]))
            }
            for position in 0 ..< positionCount where !keptSet.contains(position) {
                guard position < residualNorms.count else { continue }
                state.skippedNorms.append(
                    SkippedNorm(layer: layer, residualNorm: residualNorms[position]))
            }
            state.peakRowCount = max(state.peakRowCount, state.rows.count)
        }
        return h
    }
}
