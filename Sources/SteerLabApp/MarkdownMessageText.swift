import SwiftUI

/// Lightweight markdown rendering for chat bubbles. SwiftUI's `Text` only
/// renders *inline* markdown (bold/italic/code/links), so block structure is
/// handled here: bullets, numbered lists, headings, and code fences — the
/// constructs chat models actually emit. Anything unrecognized falls back to
/// plain text, which also keeps partially streamed markdown rendering sanely.
struct MarkdownMessageText: View {
    let text: String

    private enum Block {
        case paragraph(String)
        case heading(String)
        case bullet(String)
        case numbered(marker: String, content: String)
        case code(String)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                switch block {
                case .paragraph(let content):
                    inline(content)
                case .heading(let content):
                    inline(content).font(.headline)
                case .bullet(let content):
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Text("•")
                        inline(content)
                    }
                    .padding(.leading, 8)
                case .numbered(let marker, let content):
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Text(marker)
                        inline(content)
                    }
                    .padding(.leading, 8)
                case .code(let content):
                    Text(content)
                        .font(.system(.callout, design: .monospaced))
                        .padding(6)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(
                            .black.opacity(0.12), in: RoundedRectangle(cornerRadius: 6))
                }
            }
        }
    }

    private func inline(_ content: String) -> Text {
        if let attributed = try? AttributedString(
            markdown: content,
            options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))
        {
            return Text(attributed)
        }
        return Text(content)
    }

    private var blocks: [Block] {
        var result: [Block] = []
        var inCodeFence = false
        var codeLines: [String] = []

        for rawLine in text.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = String(rawLine)
            let trimmed = line.trimmingCharacters(in: .whitespaces)

            if trimmed.hasPrefix("```") {
                if inCodeFence {
                    result.append(.code(codeLines.joined(separator: "\n")))
                    codeLines = []
                }
                inCodeFence.toggle()
                continue
            }
            if inCodeFence {
                codeLines.append(line)
                continue
            }
            if trimmed.isEmpty { continue }

            if let content = trimmed.droppingPrefixMatch(of: /#{1,6}\s+/) {
                result.append(.heading(content))
            } else if let content = trimmed.droppingPrefixMatch(of: /[-*+]\s+/) {
                result.append(.bullet(content))
            } else if let match = trimmed.wholeMatch(of: /(\d{1,3})[.)]\s+(.*)/) {
                result.append(.numbered(marker: "\(match.1).", content: String(match.2)))
            } else {
                result.append(.paragraph(trimmed))
            }
        }
        // Unterminated fence (mid-stream): show what we have as code.
        if inCodeFence, !codeLines.isEmpty {
            result.append(.code(codeLines.joined(separator: "\n")))
        }
        return result
    }
}

extension String {
    fileprivate func droppingPrefixMatch(of regex: some RegexComponent) -> String? {
        guard let match = prefixMatch(of: regex) else { return nil }
        return String(self[match.range.upperBound...])
    }
}
