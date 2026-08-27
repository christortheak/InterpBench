import Foundation
import SteeringKit

/// What the Robustness Check's Prompts row KNOWS: how many coherence prompts
/// the selected file actually holds, what a typed count clamps to, and what
/// happens when the request outruns the supply.
///
/// The engine takes `prefix(maxCoherencePrompts)` on both routes, so asking
/// for more prompts than the file has is not an error — it is a silent
/// shortfall, and the row read as a fixed knob with no relation to the file
/// underneath it. The prefix behaviour is left exactly as it is; this type
/// exists so the UI can TELL THE TRUTH about it: the supply inline, and a
/// warning caption when the request exceeds it.
public enum RobustnessPromptSupply {

    /// The stepper/field range the row enforces. Unchanged from the row's
    /// historical `1 ... 12` — the supply is reported, never used to clamp,
    /// because a file can be swapped under a saved count.
    public static let range = 1...12

    /// How many `{"text": …}` rows a coherence prompt file holds, or nil when
    /// the file cannot be read or parsed. nil is a real answer: a path that
    /// is not yet typed in full must not be reported as an empty file.
    public static func count(file: String) -> Int? {
        let trimmed = file.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        let url = VectorCatalog.projectRoot.appending(path: trimmed)
        guard let loaded = try? StimulusSet.loadTexts(url: url) else { return nil }
        return loaded.texts.count
    }

    /// The inline supply note beside the count — "of 6 in dev-prompts.jsonl".
    /// nil when the supply is unknown, so the row simply says nothing rather
    /// than guessing.
    public static func supplyCaption(file: String, supply: Int?) -> String? {
        guard let supply else { return nil }
        let name = (file as NSString).lastPathComponent
        guard !name.isEmpty else { return "of \(supply)" }
        return "of \(supply) in \(name)"
    }

    /// The warning caption when the requested count outruns the file. nil
    /// when the request fits (or the supply is unknown).
    public static func truncationCaption(requested: Int, supply: Int?) -> String? {
        guard let supply, requested > supply else { return nil }
        return "file has \(supply) — extra prompts are ignored"
    }

    /// Parse-and-clamp for the numeric text field, the same shape the row's
    /// other numeric field uses: unparseable or out-of-range text falls back
    /// to the nearest legal value rather than zeroing the count.
    public static func clamp(_ requested: Int) -> Int {
        min(max(requested, range.lowerBound), range.upperBound)
    }

    /// Clamp typed text, keeping `current` when the text is not a number yet
    /// (mid-edit, or emptied to retype).
    public static func clamp(text: String, current: Int) -> Int {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let value = Int(trimmed) else { return clamp(current) }
        return clamp(value)
    }
}

/// Display strings for a finished robustness report — written here, where
/// they can be asserted, rather than inside the pane that shows them.
public enum VariantRobustnessReadout {

    /// Who judged: the model, its backend kind, and an OpenRouter pin when
    /// there is one. A report from before the kind stamp names the model
    /// alone rather than guessing a backend for it.
    public static func judgeStamp(
        model: String, kind: String?, provider: String?
    ) -> String {
        guard let kind, !kind.isEmpty else { return model }
        guard let provider, !provider.isEmpty else { return "\(model) (\(kind))" }
        return "\(model) (\(kind) · \(provider))"
    }
}
