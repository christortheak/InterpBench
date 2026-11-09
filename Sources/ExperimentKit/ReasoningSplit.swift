import Foundation

/// Split a generated text into the reasoning a model wrote and the answer
/// after it.
///
/// One rule, shared by every reader that has to tell the two apart: the
/// paired judge (which reads a verdict out of the answer) and the panel
/// runner (which passes a seat's answer, and never its reasoning, on to its
/// colleagues). Python twin: `reasoning_split.split`, pinned to this type by
/// `Tests/Fixtures/cross-engine/reasoning-split.json`.
///
/// The text carries one of three shapes:
///
/// - `closed`: a reasoning block that ends. Either it opens with a leading
///   `<think>` tag (whitespace allowed before it, attributes allowed inside
///   it) and a later `</think>` closes it, or the first `</think>` comes
///   before any opening tag — what a chat template that opens the block in
///   the PROMPT leaves behind. The reasoning is the text inside the block
///   and the answer is everything after the closing tag, byte for byte.
/// - `unclosed`: a leading opening tag with no closing tag after it.
///   Everything after the opening tag is reasoning, and there is no answer.
/// - `none`: neither shape. There is no reasoning block, and the whole text
///   is the answer.
///
/// The scan is literal and works on Unicode scalars (Python's code points):
/// the tags match with ASCII case folding only, and whitespace is the
/// scalar's White_Space property. Both engines apply the same rule to the
/// same scalars, so they cannot disagree about where a block ends.
public struct ReasoningSplit: Sendable, Equatable {
    public enum State: String, Sendable, Equatable {
        case closed, unclosed, none
    }

    /// nil exactly when `state` is `.none`.
    public let reasoning: String?
    /// Empty when `state` is `.unclosed`.
    public let answer: String
    public let state: State

    public init(reasoning: String?, answer: String, state: State) {
        self.reasoning = reasoning
        self.answer = answer
        self.state = state
    }

    static let closeTag = Array("</think>".unicodeScalars)
    static let openName = Array("<think".unicodeScalars)

    private static func fold(_ scalar: Unicode.Scalar) -> Unicode.Scalar {
        (scalar.value >= 65 && scalar.value <= 90)
            ? Unicode.Scalar(scalar.value + 32)! : scalar
    }

    private static func matches(
        _ scalars: [Unicode.Scalar], at index: Int, _ needle: [Unicode.Scalar]
    ) -> Bool {
        guard index >= 0, index + needle.count <= scalars.count else { return false }
        for offset in 0..<needle.count where fold(scalars[index + offset]) != needle[offset] {
            return false
        }
        return true
    }

    /// The index just past an opening tag that starts at `index`, or nil.
    private static func openTagEnd(_ scalars: [Unicode.Scalar], at index: Int) -> Int? {
        guard matches(scalars, at: index, openName) else { return nil }
        let after = index + openName.count
        guard after < scalars.count else { return nil }
        if scalars[after] == ">" { return after + 1 }
        // `<thinking>` is not a reasoning tag.
        guard scalars[after].properties.isWhitespace else { return nil }
        var cursor = after
        while cursor < scalars.count {
            if scalars[cursor] == ">" { return cursor + 1 }
            cursor += 1
        }
        return nil
    }

    private static func firstClose(_ scalars: [Unicode.Scalar], from start: Int = 0) -> Int? {
        var index = start
        while index < scalars.count {
            if scalars[index] == "<", matches(scalars, at: index, closeTag) { return index }
            index += 1
        }
        return nil
    }

    private static func firstOpen(_ scalars: [Unicode.Scalar]) -> Int? {
        var index = 0
        while index < scalars.count {
            if scalars[index] == "<", openTagEnd(scalars, at: index) != nil { return index }
            index += 1
        }
        return nil
    }

    private static func string(_ scalars: ArraySlice<Unicode.Scalar>) -> String {
        var view = String.UnicodeScalarView()
        view.append(contentsOf: scalars)
        return String(view)
    }

    /// The reasoning block in `text`, and the answer after it.
    public static func split(_ text: String) -> ReasoningSplit {
        let scalars = Array(text.unicodeScalars)
        var start = 0
        while start < scalars.count, scalars[start].properties.isWhitespace { start += 1 }
        if let openEnd = openTagEnd(scalars, at: start) {
            guard let close = firstClose(scalars, from: openEnd) else {
                return ReasoningSplit(
                    reasoning: string(scalars[openEnd...]), answer: "", state: .unclosed)
            }
            return ReasoningSplit(
                reasoning: string(scalars[openEnd..<close]),
                answer: string(scalars[(close + closeTag.count)...]), state: .closed)
        }
        if let close = firstClose(scalars) {
            let open = firstOpen(scalars)
            if open == nil || close < open! {
                return ReasoningSplit(
                    reasoning: string(scalars[..<close]),
                    answer: string(scalars[(close + closeTag.count)...]), state: .closed)
            }
        }
        return ReasoningSplit(reasoning: nil, answer: text, state: .none)
    }

    /// `text` without its leading whitespace: the form a panel routes an
    /// answer in. Python twin: `reasoning_split.leading_whitespace_removed`.
    public static func leadingWhitespaceRemoved(_ text: String) -> String {
        let scalars = Array(text.unicodeScalars)
        var index = 0
        while index < scalars.count, scalars[index].properties.isWhitespace { index += 1 }
        return string(scalars[index...])
    }

    /// `text` without leading or trailing White_Space. Python twin:
    /// `reasoning_split.strip_space`.
    public static func trimmingWhitespace(_ text: String) -> String {
        let scalars = Array(text.unicodeScalars)
        var start = 0
        var end = scalars.count
        while start < end, scalars[start].properties.isWhitespace { start += 1 }
        while end > start, scalars[end - 1].properties.isWhitespace { end -= 1 }
        return string(scalars[start..<end])
    }
}
