import Foundation

// MARK: - Method guides, read as formatted text in the app
//
// The method guides (`WorkspaceSeed/prompts/method-guides/*.md`) are written
// once for both readers: a coding assistant, which reads them through
// `science guide <method>`, and a researcher, who reads them in the app's
// "Research methods and guides". The text is the same for both; only the
// order and the presentation differ.
//
// For the researcher the app shows, in this order: the method's purpose, the
// shapes its data takes (tables and JSON examples), the rest of the guide as
// formatted text, and then a clearly labelled "For coding assistants" part
// holding the command lines and the prompts written for an assistant to
// follow. Nothing is dropped and no sentence is rewritten: every block of the
// guide lands in exactly one of those parts.
//
// This file parses the small Markdown subset the guides use (headings,
// paragraphs, lists, pipe tables, and fenced code) and decides where each
// block goes. The app's `GuideMarkdownView` only lays the result out.

public enum GuideMarkdown {

    // MARK: Blocks

    public enum Block: Equatable, Sendable {
        case heading(level: Int, text: String)
        case paragraph(String)
        case list(ordered: Bool, items: [String])
        case table(header: [String], rows: [[String]])
        case code(language: String?, text: String)
    }

    /// Parse a guide into blocks. Inline formatting (code spans, emphasis,
    /// links) stays in the block's text; `inline(_:)` renders it.
    public static func parse(_ markdown: String) -> [Block] {
        let lines = markdown.replacingOccurrences(of: "\r\n", with: "\n")
            .split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        var blocks: [Block] = []
        var paragraph: [String] = []
        func flush() {
            if !paragraph.isEmpty {
                blocks.append(.paragraph(paragraph.joined(separator: " ")))
                paragraph = []
            }
        }

        var index = 0
        while index < lines.count {
            let line = lines[index]
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if let fence = fenceMarker(trimmed) {
                flush()
                let language = String(trimmed.dropFirst(fence.count))
                    .trimmingCharacters(in: .whitespaces)
                let indent = line.prefix { $0 == " " }.count
                var code: [String] = []
                index += 1
                while index < lines.count,
                    !lines[index].trimmingCharacters(in: .whitespaces).hasPrefix(fence)
                {
                    code.append(dropIndent(lines[index], upTo: indent))
                    index += 1
                }
                index += 1  // the closing fence
                blocks.append(
                    .code(
                        language: language.isEmpty ? nil : language,
                        text: code.joined(separator: "\n")))
                continue
            }

            if trimmed.isEmpty {
                flush()
                index += 1
                continue
            }

            if let heading = heading(trimmed) {
                flush()
                blocks.append(heading)
                index += 1
                continue
            }

            if trimmed.hasPrefix("|"), index + 1 < lines.count,
                isTableSeparator(lines[index + 1])
            {
                flush()
                let header = cells(trimmed)
                var rows: [[String]] = []
                index += 2
                while index < lines.count {
                    let row = lines[index].trimmingCharacters(in: .whitespaces)
                    guard row.hasPrefix("|") else { break }
                    rows.append(fitted(cells(row), to: header.count))
                    index += 1
                }
                blocks.append(.table(header: header, rows: rows))
                continue
            }

            if let first = listItem(line) {
                flush()
                var items = [first.text]
                index += 1
                while index < lines.count {
                    let next = lines[index]
                    let nextTrimmed = next.trimmingCharacters(in: .whitespaces)
                    if nextTrimmed.isEmpty || fenceMarker(nextTrimmed) != nil { break }
                    if let item = listItem(next), item.ordered == first.ordered {
                        items.append(item.text)
                    } else if next.hasPrefix(" ") || next.hasPrefix("\t") {
                        items[items.count - 1] += " " + nextTrimmed
                    } else {
                        break
                    }
                    index += 1
                }
                blocks.append(.list(ordered: first.ordered, items: items))
                continue
            }

            paragraph.append(trimmed)
            index += 1
        }
        flush()
        return blocks
    }

    // MARK: Where each block goes

    /// One table or data example, with the heading it sat under and the
    /// sentence that introduced it, when there was one.
    public struct DataShape: Equatable, Sendable {
        public let section: String?
        public let caption: String?
        public let block: Block
    }

    /// Something in the guide's own flow: a block, or a note where a command
    /// line was moved to the coding-assistant part.
    public enum Element: Equatable, Sendable {
        case block(Block)
        case movedToAssistants
    }

    /// One heading's worth of the coding-assistant part.
    public struct AssistantSection: Equatable, Sendable {
        public let section: String?
        public var blocks: [Block]
    }

    public struct Layout: Equatable, Sendable {
        /// The guide's own title (its first-level heading).
        public var title: String?
        /// The sentence or two that say what the method is for.
        public var purpose: [Block]
        /// Every table and data example, in the order the guide gives them.
        public var dataShapes: [DataShape]
        /// The rest of the guide, in order.
        public var body: [Element]
        /// Command lines, and the prompts written for an assistant.
        public var forCodingAssistants: [AssistantSection]
    }

    /// The headings of the guide's two app-facing parts.
    public static let dataShapesTitle = "What the data looks like"
    public static let assistantsTitle = "For coding assistants"
    public static let assistantsIntroduction =
        "The command lines and prompts in this guide, for a coding assistant to use "
        + "on your behalf. You can copy them into its chat."
    /// Left in the guide where a command line was moved.
    public static let movedNote =
        "The command for this step is listed under For coding assistants, below."

    /// Arrange a guide for a researcher reading it in the app.
    public static func layout(_ markdown: String) -> Layout {
        var blocks = parse(markdown)[...]
        var layout = Layout(
            title: nil, purpose: [], dataShapes: [], body: [], forCodingAssistants: [])
        if case .heading(1, let title)? = blocks.first {
            layout.title = title
            blocks = blocks.dropFirst()
        }
        if case .paragraph? = blocks.first, let purpose = blocks.first {
            layout.purpose = [purpose]
            blocks = blocks.dropFirst()
        }

        var section: String?
        var promptLevel: Int?
        /// The paragraph just before this block, when it ends with a colon:
        /// it introduces the block and travels with it.
        func takeIntroduction() -> String? {
            guard case .block(.paragraph(let text))? = layout.body.last,
                text.hasSuffix(":")
            else { return nil }
            layout.body.removeLast()
            return text
        }
        func assistantSection(_ name: String?) {
            if layout.forCodingAssistants.last?.section != name
                || layout.forCodingAssistants.isEmpty
            {
                layout.forCodingAssistants.append(AssistantSection(section: name, blocks: []))
            }
        }

        for block in blocks {
            if case .heading(let level, let text) = block {
                if let current = promptLevel, level <= current { promptLevel = nil }
                section = text
                if promptLevel == nil, isAssistantHeading(text) {
                    promptLevel = level
                    layout.forCodingAssistants.append(AssistantSection(section: text, blocks: []))
                    continue
                }
                if promptLevel == nil {
                    layout.body.append(.block(block))
                    continue
                }
            }
            if promptLevel != nil {
                layout.forCodingAssistants[layout.forCodingAssistants.count - 1].blocks.append(block)
                continue
            }
            switch block {
            case .table:
                layout.dataShapes.append(
                    DataShape(section: section, caption: takeIntroduction(), block: block))
            case .code(let language, let text):
                switch codeKind(language: language, text: text) {
                case .data:
                    layout.dataShapes.append(
                        DataShape(section: section, caption: takeIntroduction(), block: block))
                case .command:
                    let introduction = takeIntroduction()
                    assistantSection(section ?? layout.title)
                    if let introduction {
                        layout.forCodingAssistants[layout.forCodingAssistants.count - 1]
                            .blocks.append(.paragraph(introduction))
                    }
                    layout.forCodingAssistants[layout.forCodingAssistants.count - 1]
                        .blocks.append(block)
                    if layout.body.last != .movedToAssistants {
                        layout.body.append(.movedToAssistants)
                    }
                case .other:
                    layout.body.append(.block(block))
                }
            default:
                layout.body.append(.block(block))
            }
        }
        return layout
    }

    /// A section written for an assistant to follow: the author and reviewer
    /// prompts every guide ends with, and the prompts some guides offer to
    /// copy. Recognised by its heading naming one prompt ("Coworker author
    /// prompt", "Prompt to offer a data author"); a heading that only
    /// mentions prompts in general stays in the guide.
    static func isAssistantHeading(_ text: String) -> Bool {
        text.range(of: #"\bprompt\b"#, options: [.regularExpression, .caseInsensitive]) != nil
    }

    // MARK: Code

    public enum CodeKind: Equatable, Sendable {
        /// A command line to run.
        case command
        /// The shape of a data file or request: JSON, JSON Lines, CSV.
        case data
        /// Anything else, such as a Python function an expert supplies.
        case other
    }

    static let commandLanguages: Set<String> = ["sh", "bash", "zsh", "shell", "console", "terminal"]
    static let dataLanguages: Set<String> = [
        "json", "jsonl", "jsonc", "ndjson", "csv", "tsv", "yaml", "yml",
    ]
    static let commandPrefixes = [
        "steerlab ", "steerlab-cli ", "steerlab-server ", "$ ", "curl ", "python3 ", "python ",
        "pip ", "git ",
    ]

    public static func codeKind(language: String?, text: String) -> CodeKind {
        if let language = language?.lowercased() {
            if commandLanguages.contains(language) { return .command }
            if dataLanguages.contains(language) { return .data }
            return .other
        }
        let first = text.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .first { !$0.isEmpty } ?? ""
        if first.hasPrefix("{") || first.hasPrefix("[") { return .data }
        if commandPrefixes.contains(where: { first.hasPrefix($0) }) { return .command }
        return .other
    }

    // MARK: Inline text

    /// Inline Markdown (code spans, emphasis, links) as attributed text.
    ///
    /// The guides write placeholders as `<concept>` or `<workspace>`, often
    /// outside a code span. Markdown would read those as HTML tags and drop
    /// them, so every `<` outside a code span is escaped first, except where
    /// it opens a web link.
    public static func inline(_ text: String) -> AttributedString {
        let options = AttributedString.MarkdownParsingOptions(
            allowsExtendedAttributes: false,
            interpretedSyntax: .inlineOnlyPreservingWhitespace,
            failurePolicy: .returnPartiallyParsedIfPossible)
        return (try? AttributedString(markdown: escapingPlaceholders(text), options: options))
            ?? AttributedString(text)
    }

    static func escapingPlaceholders(_ text: String) -> String {
        var result = ""
        var inCode = false
        var rest = text[...]
        while let character = rest.first {
            if character == "`" {
                inCode.toggle()
                result.append(character)
            } else if character == "\\", !inCode, let next = rest.dropFirst().first {
                // Already escaped: keep the pair as written.
                result.append(character)
                result.append(next)
                rest = rest.dropFirst()
            } else if character == "<", !inCode,
                !["<http://", "<https://", "<mailto:"].contains(where: { rest.hasPrefix($0) })
            {
                result.append("\\<")
            } else {
                result.append(character)
            }
            rest = rest.dropFirst()
        }
        return result
    }

    // MARK: Line rules

    private static func fenceMarker(_ trimmed: String) -> String? {
        if trimmed.hasPrefix("```") { return "```" }
        if trimmed.hasPrefix("~~~") { return "~~~" }
        return nil
    }

    private static func dropIndent(_ line: String, upTo indent: Int) -> String {
        let leading = line.prefix { $0 == " " }.count
        return String(line.dropFirst(min(leading, indent)))
    }

    private static func heading(_ trimmed: String) -> Block? {
        let hashes = trimmed.prefix { $0 == "#" }.count
        guard (1...6).contains(hashes) else { return nil }
        let rest = trimmed.dropFirst(hashes)
        guard rest.first == " " else { return nil }
        var text = rest.trimmingCharacters(in: .whitespaces)
        while text.hasSuffix("#") { text.removeLast() }
        return .heading(level: hashes, text: text.trimmingCharacters(in: .whitespaces))
    }

    private static func isTableSeparator(_ line: String) -> Bool {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        return trimmed.hasPrefix("|") && trimmed.contains("-")
            && trimmed.allSatisfy { "|-: ".contains($0) }
    }

    /// A row's cells: split on pipes that are neither escaped nor inside a
    /// code span, with the outer pipes dropped.
    static func cells(_ row: String) -> [String] {
        var cells: [String] = []
        var current = ""
        var inCode = false
        var previous: Character?
        for character in row {
            if character == "`" { inCode.toggle() }
            if character != "|" || inCode {
                current.append(character)
            } else if previous == "\\" {
                current.removeLast()
                current.append(character)
            } else {
                cells.append(current)
                current = ""
            }
            previous = character
        }
        cells.append(current)
        if cells.first?.trimmingCharacters(in: .whitespaces).isEmpty == true {
            cells.removeFirst()
        }
        if cells.last?.trimmingCharacters(in: .whitespaces).isEmpty == true {
            cells.removeLast()
        }
        return cells.map { $0.trimmingCharacters(in: .whitespaces) }
    }

    private static func fitted(_ row: [String], to count: Int) -> [String] {
        guard count > 0 else { return row }
        if row.count >= count {
            return Array(row.prefix(count - 1)) + [row.dropFirst(count - 1).joined(separator: " | ")]
        }
        return row + Array(repeating: "", count: count - row.count)
    }

    private struct ListItem {
        let ordered: Bool
        let text: String
    }

    private static func listItem(_ line: String) -> ListItem? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        for marker in ["- ", "* ", "+ "] where trimmed.hasPrefix(marker) {
            return ListItem(ordered: false, text: String(trimmed.dropFirst(2)))
        }
        let digits = trimmed.prefix { $0.isNumber }
        guard !digits.isEmpty, digits.count <= 3 else { return nil }
        let rest = trimmed.dropFirst(digits.count)
        guard let mark = rest.first, mark == "." || mark == ")",
            rest.dropFirst().first == " "
        else { return nil }
        return ListItem(
            ordered: true, text: rest.dropFirst(2).trimmingCharacters(in: .whitespaces))
    }
}
