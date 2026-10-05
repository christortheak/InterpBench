import AppKit
import ExperimentKit
import SwiftUI

// The method guides' Markdown, drawn as formatted text: headings, paragraphs,
// lists, tables, and code. What a block is, and which part of the guide it
// belongs to, is `GuideMarkdown`'s (ExperimentKit, unit-tested); these views
// only draw it. They sit inside the guide sheet's scroll view, never directly
// in a split-view column.

/// A run of guide blocks, top to bottom.
struct GuideBlocksView: View {
    let blocks: [GuideMarkdown.Block]
    /// Offer a Copy button under each code block (the coding-assistant part).
    var copyableCode = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                GuideBlockView(block: block, copyableCode: copyableCode)
            }
        }
    }
}

/// One block.
struct GuideBlockView: View {
    let block: GuideMarkdown.Block
    var copyableCode = false

    var body: some View {
        switch block {
        case .heading(let level, let text):
            Text(GuideMarkdown.inline(text))
                .font(level <= 1 ? .title2.bold() : level == 2 ? .title3.weight(.semibold) : .headline)
                .padding(.top, level <= 2 ? 6 : 2)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityAddTraits(.isHeader)
        case .paragraph(let text):
            Text(GuideMarkdown.inline(text))
                .fixedSize(horizontal: false, vertical: true)
        case .list(let ordered, let items):
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.offset) { index, item in
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        Text(ordered ? "\(index + 1)." : "•")
                            .monospacedDigit()
                            .foregroundStyle(.secondary)
                        Text(GuideMarkdown.inline(item))
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        case .table(let header, let rows):
            GuideTableView(header: header, rows: rows)
        case .code(_, let text):
            GuideCodeView(text: text, copyable: copyableCode)
        }
    }
}

/// A pipe table as a grid with a bold header row and a rule between rows.
struct GuideTableView: View {
    let header: [String]
    let rows: [[String]]

    var body: some View {
        Grid(alignment: .topLeading, horizontalSpacing: 14, verticalSpacing: 6) {
            GridRow {
                ForEach(header.indices, id: \.self) { column in
                    cell(header[column]).font(.callout.weight(.semibold))
                }
            }
            Divider()
            ForEach(rows.indices, id: \.self) { row in
                GridRow {
                    ForEach(rows[row].indices, id: \.self) { column in
                        cell(rows[row][column]).font(.callout)
                    }
                }
                if row < rows.count - 1 { Divider() }
            }
        }
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.07)))
        .textSelection(.enabled)
    }

    private func cell(_ text: String) -> some View {
        Text(GuideMarkdown.inline(text))
            .fixedSize(horizontal: false, vertical: true)
    }
}

/// A code block: monospaced, scrolling sideways rather than wrapping, with an
/// optional Copy button.
struct GuideCodeView: View {
    let text: String
    var copyable = false
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            ScrollView(.horizontal) {
                Text(text)
                    .font(.system(.callout, design: .monospaced))
                    .textSelection(.enabled)
                    .fixedSize()
                    .padding(8)
            }
            .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.07)))
            if copyable {
                Button(copied ? "Copied" : "Copy") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(text, forType: .string)
                    copied = true
                }
                .controlSize(.small)
                .help("copy this text, to paste into a coding assistant's chat")
            }
        }
    }
}
