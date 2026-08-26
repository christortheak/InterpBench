import Foundation
import SteeringKit

/// The two extraction DECLARATIONS — WHERE the residual stream is read and
/// HOW the stimulus reached the model — in the shape a picker holds them.
///
/// The engines have had both declarations since 2026-08-25 (`--reading-position`,
/// `--extraction-rendering`, and the typed refusals behind them), and the app
/// could declare NEITHER: the Studies attach row offered a `--pool-from` token
/// index, and the Concepts builder a pooling toggle, so six of the eight
/// positions and every template rendering were reachable only from a command
/// line. These types are what a SwiftUI picker binds to; they carry no rules
/// of their own — every choice is turned back into the cross-engine
/// declaration text and handed to ``ReadingPosition/declared(_:)`` /
/// ``ExtractionRendering/declared(object:)``, so the app refuses exactly what
/// the CLI refuses, in the engines' own words.

// MARK: - Reading position

/// One entry of the reading-position picker: the position's KIND, with its
/// parameter (K/k/i/n) held beside it so a picker selection stays stable while
/// the number is typed.
public enum ReadingPositionChoice: String, CaseIterable, Sendable, Hashable,
    Identifiable
{
    /// Declare nothing — the recipe keeps its default, and the manifest keeps
    /// its bytes. Offered only where the caller has a nil to pass (the study
    /// attach); the Concepts builder always reads at a concrete position.
    case recipeDefault
    case lastToken
    case meanFromToken
    case offsetFromEnd
    case lastContentToken
    case turnCloseToken
    case postInstruction
    case contentOffset
    case meanContentFromToken

    public var id: String { rawValue }

    /// Everything a person can actually declare — the whole cross-engine
    /// vocabulary, in the order ``ReadingPosition/declarableLabels`` lists it.
    public static let declarableCases: [ReadingPositionChoice] = [
        .lastToken, .meanFromToken, .offsetFromEnd, .lastContentToken,
        .turnCloseToken, .postInstruction, .contentOffset, .meanContentFromToken,
    ]

    /// What the menu row says. The parameterized entries end in an ellipsis
    /// because the number lives in the field beside the picker.
    public var menuLabel: String {
        switch self {
        case .recipeDefault: "recipe default"
        case .lastToken: "last token"
        case .meanFromToken: "mean from token…"
        case .offsetFromEnd: "offset from end…"
        case .lastContentToken: "last content token"
        case .turnCloseToken: "turn close token"
        case .postInstruction: "post-instruction…"
        case .contentOffset: "content offset…"
        case .meanContentFromToken: "mean content from token…"
        }
    }

    /// The parameter's name in the vocabulary, or nil for a position that
    /// takes none (which is what hides the field).
    public var parameterCaption: String? {
        switch self {
        case .meanFromToken: "K"
        case .offsetFromEnd, .contentOffset: "k"
        case .postInstruction: "i"
        case .meanContentFromToken: "n"
        case .recipeDefault, .lastToken, .lastContentToken, .turnCloseToken: nil
        }
    }

    public var takesParameter: Bool { parameterCaption != nil }

    /// True for the roles that only exist inside a rendered chat turn — the
    /// ones the store refuses under a raw rendering. Read by the picker's help
    /// text only; the REFUSAL still comes from the store.
    public var requiresTemplatedRendering: Bool {
        switch self {
        case .lastContentToken, .turnCloseToken, .postInstruction, .contentOffset:
            true
        case .recipeDefault, .lastToken, .meanFromToken, .offsetFromEnd,
            .meanContentFromToken:
            false
        }
    }

    /// A parameter this kind can carry, for the moment the picker moves from
    /// one kind to another. NOT a validation: an out-of-vocabulary number
    /// typed into the field still reaches the engine's own refusal.
    public func steppedParameter(from parameter: Int) -> Int {
        switch self {
        case .postInstruction: min(max(parameter, 1), 5)
        case .meanFromToken, .offsetFromEnd, .contentOffset, .meanContentFromToken:
            max(parameter, 0)
        case .recipeDefault, .lastToken, .lastContentToken, .turnCloseToken:
            parameter
        }
    }

    /// The cross-engine LABEL this choice declares, or nil for "declare
    /// nothing" — which is the byte-identity contract, not an omission.
    public func declarationLabel(parameter: Int) -> String? {
        switch self {
        case .recipeDefault: nil
        case .lastToken: ReadingPosition.lastToken.label
        case .meanFromToken: ReadingPosition.meanFromToken(parameter).label
        case .offsetFromEnd: ReadingPosition.offsetFromEnd(parameter).label
        case .lastContentToken: ReadingPosition.lastContentToken.label
        case .turnCloseToken: ReadingPosition.turnCloseToken.label
        case .postInstruction: ReadingPosition.postInstruction(parameter).label
        case .contentOffset: ReadingPosition.contentOffset(parameter).label
        case .meanContentFromToken:
            ReadingPosition.meanContentFromToken(parameter).label
        }
    }

    /// The position itself, through the engine's own STRICT parser — so a
    /// `post-instruction 9` typed into the field is refused here in the same
    /// words the CLI uses, rather than being clamped into a recipe nobody
    /// asked for. nil for ``recipeDefault``.
    public func declaredPosition(parameter: Int) throws -> ReadingPosition? {
        try ReadingPosition.declared(declarationLabel(parameter: parameter))
    }

    /// The picker state that displays an existing position.
    public static func choice(
        for position: ReadingPosition
    ) -> (choice: ReadingPositionChoice, parameter: Int) {
        switch position {
        case .lastToken: (.lastToken, 0)
        case .meanFromToken(let k): (.meanFromToken, k)
        case .offsetFromEnd(let k): (.offsetFromEnd, k)
        case .lastContentToken: (.lastContentToken, 0)
        case .turnCloseToken: (.turnCloseToken, 0)
        case .postInstruction(let i): (.postInstruction, i)
        case .contentOffset(let k): (.contentOffset, k)
        case .meanContentFromToken(let n): (.meanContentFromToken, n)
        }
    }
}

// MARK: - Extraction rendering

/// The rendering picker's state: a mode, and the two chat-template parameters
/// this surface offers. `qwenThinkingEnabled` and `systemPrompt` are
/// deliberately absent — they are declarable on the command line and have no
/// control here yet.
///
/// **Raw is the default and declares NOTHING** (``declared()`` answers nil),
/// which is the same rule ``ExtractionRendering/declared(object:)`` applies to
/// an explicit `{"mode": "raw"}`: absent is the legacy rendering, and a
/// manifest that chose it keeps its bytes and its recipe identity.
public struct ExtractionRenderingChoice: Sendable, Equatable {
    public var mode: ExtractionRendering.Mode
    public var voice: ExtractionRendering.Voice
    /// User voice only. The assistant voice refuses the key outright (it
    /// reaches nothing there), so the declaration omits it.
    public var addGenerationPrompt: Bool

    public init(
        mode: ExtractionRendering.Mode = .raw,
        voice: ExtractionRendering.Voice = .user,
        addGenerationPrompt: Bool = true
    ) {
        self.mode = mode
        self.voice = voice
        self.addGenerationPrompt = addGenerationPrompt
    }

    public var isRaw: Bool { mode == .raw }

    /// The declaration object, in exactly the JSON shape both engines parse.
    /// nil for raw — nothing to declare.
    public var declaration: [String: Any]? {
        guard mode == .chatTemplate else { return nil }
        var object: [String: Any] = ["mode": ExtractionRendering.Mode.chatTemplate.rawValue]
        if voice == .assistant {
            object["voice"] = ExtractionRendering.Voice.assistant.rawValue
        } else if !addGenerationPrompt {
            object["addGenerationPrompt"] = false
        }
        return object
    }

    /// The rendering, through the engine's own declaration parser — which is
    /// where the assistant-voice and `addGenerationPrompt: false` engine
    /// asymmetries are answered, in their own words. nil for raw.
    public func declared() throws -> ExtractionRendering? {
        guard let declaration else { return nil }
        return try ExtractionRendering.declared(object: declaration)
    }

    /// The picker state that displays an existing rendering.
    public static func choice(for rendering: ExtractionRendering?) -> Self {
        guard let rendering, !rendering.isRaw else { return Self() }
        return Self(
            mode: .chatTemplate,
            voice: rendering.resolvedVoice,
            addGenerationPrompt: rendering.resolvedAddGenerationPrompt)
    }
}
