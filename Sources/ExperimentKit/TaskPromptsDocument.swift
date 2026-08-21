import Foundation

/// Round-trips a task-prompts JSONL file for the Studies editor WITHOUT
/// destroying the science-layer fields the logprob instrument depends on.
///
/// The historical editor read only `.texts` and re-encoded `{"text": …}`
/// lines — silently stripping `options`, `target`, `anchorMonths`, and every
/// other per-item key on save (the task-prompts data-loss bug, fixed
/// 2026-07-13). This document instead keeps every line's ORIGINAL bytes:
/// only the editable prompt text is exposed; on save, untouched items are
/// written back byte-identically and edited items are re-encoded with all
/// their other keys preserved verbatim.
///
/// Scripted-transcript items (`transcript` array — the metacognition-study
/// instrument) round-trip field-faithfully like every other key. When such
/// an item has no `text`/`prompt` of its own, the editable prompt text IS
/// the transcript's final user turn (the run loop's display-text rule), and
/// an edit rewrites that turn's `content` in place — never a divergent
/// shadow `text` key.
///
/// Not `Sendable` (holds parsed JSON objects) — it lives on the main-actor
/// panel and in tests.
public struct TaskPromptsDocument {
    /// Keys an item may carry besides the prompt text; shown to the
    /// researcher as a non-editable instrument badge so the preserved
    /// fields are visible even though the editor cannot change them.
    struct Item {
        /// The original JSONL line, verbatim (no trailing newline).
        var rawLine: String
        /// The line's declared `id`, when present. Identity for the
        /// ResponseFormat scope rule — it must speak the SAME id vocabulary
        /// as both engines' run loaders (real id, else `prompt-<ordinal>`),
        /// or a scope pinned in the editor refuses at run start
        /// (2026-08-03 field incident: the editor synthesized `item-N`).
        var id: String?
        /// "text" or "prompt" — whichever key the line used (new items get
        /// "text", the Swift-side convention). nil for transcript-only items
        /// (editing rewrites the transcript's final user turn instead).
        var textKey: String?
        /// The editable prompt text.
        var text: String
        /// True when the line carries a non-empty `options` array (the
        /// categorical-instrument fields).
        var hasOptions: Bool
        /// True when the line declares a non-empty `target` — the option a
        /// target-dependent choice instrument reads (open-issues #6).
        /// Defaults false: every construction site other than `load` mints an
        /// option-less item, where the field cannot matter.
        var hasTarget: Bool = false
        /// True when the line carries a non-empty `transcript` array (a
        /// scripted-transcript item).
        var hasTranscript: Bool
        /// The declared `responseFormat`, or nil when the key is absent or
        /// unrecognised. The EDITOR is deliberately lenient where the run
        /// loop refuses: this document exists to round-trip bytes without
        /// data loss, and refusing to open a file with one bad value would
        /// leave the researcher no way to fix it here.
        var responseFormat: ResponseFormat?
    }

    var items: [Item]

    public struct ParseError: Error, CustomStringConvertible {
        public let line: Int
        public var description: String {
            "malformed task prompt JSONL at line \(line)"
        }
    }

    // MARK: - Load

    public static func load(_ data: Data) throws -> TaskPromptsDocument {
        var items: [Item] = []
        let lines = String(decoding: data, as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: true)
        for (index, raw) in lines.enumerated() {
            let trimmed = raw.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { continue }
            guard
                let object = (try? JSONSerialization.jsonObject(with: Data(trimmed.utf8)))
                    as? [String: Any]
            else { throw ParseError(line: index + 1) }
            let transcript = object["transcript"] as? [Any]
            let hasTranscript = (transcript?.isEmpty == false)
            // Same precedence as the run loop's parser: `prompt` (server
            // style) wins over legacy `text`; a transcript-only item derives
            // its display text from the transcript's final user turn.
            let textKey: String?
            let text: String
            if let prompt = object["prompt"] as? String {
                textKey = "prompt"
                text = prompt
            } else if let legacy = object["text"] as? String {
                textKey = "text"
                text = legacy
            } else if hasTranscript {
                textKey = nil
                text = Self.finalTurnContent(transcript) ?? ""
            } else {
                throw ParseError(line: index + 1)
            }
            let options = object["options"] as? [Any]
            items.append(
                Item(
                    rawLine: trimmed,
                    id: (object["id"] as? String).flatMap {
                        $0.isEmpty ? nil : $0
                    },
                    textKey: textKey,
                    text: text,
                    hasOptions: (options?.isEmpty == false),
                    hasTarget: ((object["target"] as? String)?.isEmpty == false),
                    hasTranscript: hasTranscript,
                    // Lenient by design (see the field's doc): an
                    // unrecognised value reads as nil here so the file can
                    // still be OPENED and corrected; the run loop refuses it.
                    responseFormat: try? ResponseFormat.parse(
                        object["responseFormat"] as? String)))
        }
        return TaskPromptsDocument(items: items)
    }

    /// Fresh document from plain prompt texts (the "typed straight into the
    /// editor" path) — every item is a new `{"text": …}` line.
    public static func fromTexts(_ texts: [String]) -> TaskPromptsDocument {
        TaskPromptsDocument(
            items: texts.map { text in
                Item(
                    rawLine: Self.encodeLine(["text": text]),
                    id: nil,
                    textKey: "text",
                    text: text,
                    hasOptions: false,
                    hasTranscript: false,
                    responseFormat: nil)
            })
    }

    /// The `content` of a transcript's final turn (the run loop's
    /// display-text rule; the schema validator guarantees it is a user turn).
    private static func finalTurnContent(_ transcript: [Any]?) -> String? {
        guard let last = transcript?.last as? [String: Any] else { return nil }
        return last["content"] as? String
    }

    // MARK: - Editor surface

    public static let editorSeparator = "\n\n---\n\n"

    public var texts: [String] { items.map(\.text) }

    public var editorText: String {
        items.map(\.text).joined(separator: Self.editorSeparator)
    }

    public var count: Int { items.count }

    /// Items carrying a non-empty `options` array (instrument fields the
    /// editor preserves but cannot edit).
    public var optionsItemCount: Int { items.count(where: \.hasOptions) }

    /// The `ResponseFormat` rule's view of these items. The id vocabulary
    /// is the RUN LOADERS' rule on both engines — the file's real `id`,
    /// else `prompt-<ordinal>` (1-based) — because the scope pinned here is
    /// verified at run start against ids the loaders resolve. The editor's
    /// former synthesized `item-N` ids produced a pin no engine could ever
    /// match: same count, disjoint vocabulary (2026-08-03 field incident).
    public var responseFormatItems: [ResponseFormat.Item] {
        items.enumerated().map { index, item in
            .init(
                id: item.id ?? "prompt-\(index + 1)",
                hasOptions: item.hasOptions,
                hasTarget: item.hasTarget,
                format: item.responseFormat)
        }
    }

    /// Items carrying options whose declared format the answer-token
    /// instruments CANNOT read (`json`, `freeText`). Drives the honest
    /// version of the activation advice.
    public var unscorableOptionItemCount: Int {
        ResponseFormat.unscorableItems(responseFormatItems).count
    }

    /// Items carrying a non-empty scripted `transcript` array.
    public var transcriptItemCount: Int { items.count(where: \.hasTranscript) }

    /// One-line badge for the editor, or nil when no item carries
    /// instrument fields.
    public var instrumentSummary: String? {
        var parts: [String] = []
        if optionsItemCount > 0 {
            parts.append(
                "\(optionsItemCount) of \(count) item\(count == 1 ? "" : "s") carry "
                    + "options/instrument fields — preserved on save, not editable here")
        }
        if transcriptItemCount > 0 {
            parts.append(
                "\(transcriptItemCount) with scripted transcripts (editing the "
                    + "prompt text edits the final user turn)")
        }
        return parts.isEmpty ? nil : parts.joined(separator: "; ")
    }

    /// Splits the editor text back into prompt blocks (blocks separated by a
    /// line containing only `---`, matching the historical editor).
    public static func editorBlocks(_ text: String) -> [String] {
        text.split(separator: "\n---\n", omittingEmptySubsequences: false)
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    // MARK: - Apply edits

    /// Pairs edited blocks with existing items BY INDEX: block i replaces
    /// item i's text (all other keys kept verbatim); extra blocks become new
    /// `{"text": …}` items; missing blocks drop the tail items.
    public func applyingEditedTexts(_ blocks: [String]) -> TaskPromptsDocument {
        var updated: [Item] = []
        for (index, block) in blocks.enumerated() {
            if index < items.count {
                var item = items[index]
                if item.text != block {
                    item = Self.rewriting(item, text: block)
                }
                updated.append(item)
            } else {
                updated.append(
                    Item(
                        rawLine: Self.encodeLine(["text": block]),
                        id: nil,
                        textKey: "text",
                        text: block,
                        hasOptions: false,
                        hasTranscript: false,
                        responseFormat: nil))
            }
        }
        return TaskPromptsDocument(items: updated)
    }

    /// Re-encodes one item with a new prompt text. Every OTHER key of the
    /// original line is preserved (unknown keys included); only the text key
    /// — or, for a transcript-only item, the transcript's final turn content
    /// — changes. Key order is canonical (sorted) on rewritten lines —
    /// untouched lines keep their exact original bytes.
    private static func rewriting(_ item: Item, text: String) -> Item {
        guard
            var object = (try? JSONSerialization.jsonObject(with: Data(item.rawLine.utf8)))
                as? [String: Any]
        else {
            // Unreachable for items that came through `load`, but never
            // corrupt data: fall back to a fresh line.
            return Item(
                rawLine: encodeLine([item.textKey ?? "text": text]),
                id: item.id,
                textKey: item.textKey ?? "text", text: text,
                hasOptions: false, hasTranscript: false, responseFormat: nil)
        }
        if let textKey = item.textKey {
            object[textKey] = text
        } else if var transcript = object["transcript"] as? [Any],
            var last = transcript.last as? [String: Any]
        {
            // Transcript-only item: the editable text IS the final user
            // turn's content.
            last["content"] = text
            transcript[transcript.count - 1] = last
            object["transcript"] = transcript
        } else {
            object["text"] = text
        }
        return Item(
            rawLine: encodeLine(object),
            id: item.id,
            textKey: item.textKey,
            text: text,
            hasOptions: item.hasOptions,
            hasTarget: item.hasTarget,
            hasTranscript: item.hasTranscript,
            responseFormat: item.responseFormat)
    }

    // MARK: - Serialize

    /// One JSONL line per item, trailing newline after each (the format
    /// every loader in both engines reads).
    public func serialized() -> Data {
        var out = ""
        for item in items {
            out += item.rawLine + "\n"
        }
        return Data(out.utf8)
    }

    private static func encodeLine(_ object: [String: Any]) -> String {
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: object, options: [.sortedKeys]),
            let line = String(data: data, encoding: .utf8)
        else { return "{}" }
        return line
    }
}
