import ExperimentKit
import SteeringKit
import SwiftUI
import UniformTypeIdentifiers

struct ResultReviewSheet: Identifiable {
    enum Mode {
        case generations
        case judgments
    }

    let id = UUID()
    let mode: Mode
    let detail: StudyRunDetail
    /// A judge directory other than the detail's own: an evaluation that
    /// stopped before it wrote a report, whose kept rows are still
    /// reviewable.
    var judgmentsDirectory: String? = nil

    var title: String {
        switch mode {
        case .generations: "Generated Responses"
        case .judgments: "Judge Responses"
        }
    }

    /// The directory whose file this sheet pages through.
    var directory: URL {
        switch mode {
        case .generations:
            URL(filePath: detail.item.path)
        case .judgments:
            URL(
                filePath: judgmentsDirectory ?? detail.judgeArtifactDirectory
                    ?? detail.item.path)
        }
    }

    var fileURL: URL {
        switch mode {
        case .generations:
            StudyResultRepository.responsesURL(runDirectory: directory)
        case .judgments:
            StudyResultRepository.judgmentsURL(directory: directory)
        }
    }
}

struct ResultReviewWindow: View {
    let sheet: ResultReviewSheet
    @State private var showingInstrumentation = false
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(sheet.title)
                        .font(.title2.weight(.semibold))
                    Text(sheet.directory.lastPathComponent)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
                Spacer()
                // Audit headline 12: the sheet had no way out — no Close
                // button and no cancelAction, so Escape was at best
                // undiscoverable.
                Button("Close", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
            }

            Button("Compare probe readings and policy actions…") { showingInstrumentation = true }
            Divider()

            // Every line of the file is a row here, a page at a time
            // (`StudyRecordReview`): the header counts the whole file, and
            // the caption says which part of it is on screen.
            switch sheet.mode {
            case .generations:
                StudyRecordReviewPane<StudyResponseRow, ResponseRowCard>(
                    url: sheet.fileURL
                ) { ResponseRowCard(row: $0) }
            case .judgments:
                StudyRecordReviewPane<StudyJudgmentRow, JudgmentRowCard>(
                    url: sheet.fileURL
                ) { JudgmentRowCard(row: $0) }
            }
        }
        .padding(18)
        .frame(minWidth: 760, minHeight: 560)
        .sheet(isPresented: $showingInstrumentation) {
            InstrumentationEvidenceView(root: ExperimentStore.workspaceRoot, path: sheet.detail.item.path)
        }
    }
}

/// One evidence file, counted in full and shown a page at a time.
///
/// The file is read and counted off the main thread when the sheet opens,
/// and each page decodes only its own rows, so a file with thousands of
/// records opens as promptly as a short one. Nothing is left out: the
/// header's counts cover the whole file, every kind of row has a filter,
/// and Previous and Next reach every page.
private struct StudyRecordReviewPane<Row: StudyReviewRow, Card: View>: View {
    let url: URL
    @ViewBuilder let card: (Row) -> Card

    private struct Request: Hashable {
        var page = 0
        var kind: Row.Kind?
    }

    @State private var review: StudyRecordReview<Row>?
    @State private var request = Request()
    @State private var shown: StudyRecordReview<Row>.Page?
    /// Counts the pages put on screen, so each one starts at its top.
    @State private var shownVersion = 0

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let review {
                if review.file.isReadable {
                    summary(review)
                    controls(review)
                    rows
                } else {
                    Text("\(url.lastPathComponent) is not in this directory, so there are no rows to review.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                    Spacer()
                }
            } else {
                ProgressView("Counting every row in \(url.lastPathComponent)…")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .task(id: url) {
            let loaded = await StudyRecordReview<Row>.load(url: url)
            guard !Task.isCancelled else { return }
            review = loaded.review
            shown = loaded.firstPage
            shownVersion += 1
        }
        .task(id: request) {
            guard let review else { return }
            let page = await review.loadPage(request.page, kind: request.kind)
            guard !Task.isCancelled else { return }
            shown = page
            shownVersion += 1
        }
    }

    /// The whole file's counts, one per kind of row — zero or not, so
    /// "none failed" is stated rather than left to be assumed.
    private func summary(_ review: StudyRecordReview<Row>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("\(Row.noun.counted(review.counts.total)) in \(url.lastPathComponent)")
                .font(.callout.weight(.semibold))
            HStack(spacing: 8) {
                ForEach(review.counts.summary) { item in
                    Text("\(item.label) \(StudyReviewText.grouped(item.count))")
                        .font(.caption)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(
                            (item.needsAttention ? Color.orange : Color.secondary)
                                .opacity(0.15),
                            in: Capsule())
                        .foregroundStyle(item.needsAttention ? Color.orange : Color.secondary)
                }
            }
        }
    }

    private func controls(_ review: StudyRecordReview<Row>) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                Picker(
                    "Show",
                    selection: Binding<Row.Kind?>(
                        get: { request.kind },
                        // A different filter is a different list: start it
                        // at its first page.
                        set: { request = Request(page: 0, kind: $0) })
                ) {
                    Text("All rows (\(StudyReviewText.grouped(review.counts.total)))")
                        .tag(Row.Kind?.none)
                    ForEach(Array(Row.Kind.allCases), id: \.self) { kind in
                        Text("\(kind.label) (\(StudyReviewText.grouped(review.counts.count(kind))))")
                            .tag(Row.Kind?.some(kind))
                    }
                }
                .fixedSize()
                .help(
                    "which rows to page through — every kind of row in the "
                        + "file is listed with its count, and All shows them "
                        + "in file order")
                Spacer()
                if let shown {
                    Button("Previous") { request.page = shown.page - 1 }
                        .disabled(!shown.paging.hasPrevious(shown.page))
                        .help("the page before this one")
                    Button("Next") { request.page = shown.page + 1 }
                        .disabled(!shown.paging.hasNext(shown.page))
                        .help("the page after this one")
                }
            }
            // Which rows are on screen, out of how many: a page is never
            // left to be mistaken for the whole file.
            if let shown {
                Text(shown.caption)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    @ViewBuilder
    private var rows: some View {
        if let shown {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    ForEach(shown.rows) { row in
                        card(row)
                    }
                }
                .padding(.trailing, 8)
            }
            // A new page or filter starts at its top, not wherever the
            // last one was scrolled to.
            .id(shownVersion)
        } else {
            Spacer()
        }
    }
}

/// A long prompt or response, shown in part with the amount stated and a
/// way to see all of it — never trimmed behind a bare ellipsis.
private struct ReviewExcerpt: View {
    let text: String
    let font: Font
    @State private var showingAll = false

    var body: some View {
        let excerpt = StudyReviewText.excerpt(text)
        VStack(alignment: .leading, spacing: 4) {
            Text(showingAll ? text : excerpt.shown)
                .font(font)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            if let note = excerpt.note {
                HStack(spacing: 6) {
                    Text(showingAll ? "Showing all of it." : note)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Button(showingAll ? "Show less" : "Show all") { showingAll.toggle() }
                        .buttonStyle(.link)
                        .font(.caption)
                }
            }
        }
    }
}

/// The plain-language line that says what a row is when it is not an
/// ordinary, complete record.
private struct ReviewStatusNote: View {
    let text: String
    let needsAttention: Bool

    var body: some View {
        Label(
            text,
            systemImage: needsAttention ? "exclamationmark.triangle.fill" : "info.circle"
        )
        .font(.callout)
        .foregroundStyle(needsAttention ? Color.orange : Color.secondary)
        .fixedSize(horizontal: false, vertical: true)
    }
}

private struct ReviewCard<Content: View>: View {
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 8, content: content)
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.background)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
    }
}

private struct ResponseRowCard: View {
    let row: StudyResponseRow

    var body: some View {
        ReviewCard {
            HStack {
                Text(row.title)
                    .font(.headline)
                Spacer()
                Text(measures)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let note = row.statusNote {
                ReviewStatusNote(text: note, needsAttention: row.kind.needsAttention)
            }
            if let failure = row.failure {
                Text(failure)
                    .font(.callout.monospaced())
                    .textSelection(.enabled)
            }
            if let note = row.answerNote {
                ReviewStatusNote(text: note, needsAttention: true)
            }
            if let selected = row.selected {
                Text("Preferred option: \(selected)" + (row.margin.map { " · margin " + $0.formatted(.number.precision(.fractionLength(3))) } ?? ""))
                    .font(.callout)
                    .textSelection(.enabled)
            }
            if let decisions = row.interventionDecisions {
                DisclosureGroup("Intervention decisions") {
                    Text("These records identify the policy, the consumed token position, the probe scores, and the requested strengths. Partial or failed responses are not completed evidence.").font(.caption)
                    ScrollView([.horizontal, .vertical]) {
                        PolicyJSONView(value: decisions)
                    }.frame(maxHeight: 300)
                }
            }
            if let readings = row.probeMeasurements { ProbeMeasurementResultsView(value: readings) }
            if let prompt = row.prompt {
                ReviewExcerpt(text: prompt, font: .caption)
                    .foregroundStyle(.secondary)
            }
            if let output = row.output {
                ReviewExcerpt(text: output, font: .body.monospaced())
                    .padding(10)
                    .background(.quaternary.opacity(0.35))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }
            if row.kind == .failed || row.kind == .unreadable {
                DisclosureGroup("The line as written") {
                    ReviewExcerpt(text: row.rawJSON, font: .caption.monospaced())
                }
                .help("this line of generations.jsonl, exactly as the run recorded it")
            }
        }
    }

    /// "record 12 · 148 words · distinct-2 0.912" — a measure appears only
    /// when the record stores it.
    private var measures: String {
        var parts = ["record \(StudyReviewText.grouped(row.record))"]
        if let words = row.wordCount { parts.append("\(StudyReviewText.grouped(words)) words") }
        if let distinct = row.distinct2 {
            parts.append("distinct-2 " + distinct.formatted(.number.precision(.fractionLength(3))))
        }
        return parts.joined(separator: " · ")
    }
}

private struct JudgmentRowCard: View {
    let row: StudyJudgmentRow

    var body: some View {
        ReviewCard {
            HStack {
                Text(row.title)
                    .font(.headline)
                Spacer()
                Text(verdict)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Text(provenance)
                .font(.caption)
                .foregroundStyle(.secondary)
            if let note = row.statusNote {
                ReviewStatusNote(text: note, needsAttention: true)
            }
            if let reason = row.reason {
                ReviewExcerpt(text: reason, font: .callout.monospaced())
            }
            if let prompt = row.prompt {
                ReviewExcerpt(text: prompt, font: .caption)
                    .foregroundStyle(.secondary)
            } else if row.kind != .unreadable {
                Text("This row does not store the prompt. It is in the run record with the same prompt ID.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let briefReason = row.briefReason {
                Text(briefReason)
                    .textSelection(.enabled)
            }
            if let scores = scoreSummary {
                Text(scores)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            if let structured = structuredFieldsSummary {
                Text(structured)
                    .font(.caption.monospaced())
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
            DisclosureGroup("The row as recorded") {
                ReviewExcerpt(text: row.rawJSON, font: .caption.monospaced())
            }
            .help("this line of judgments.jsonl, exactly as the evaluation recorded it")
        }
    }

    /// "record 7 · winner A · condition · confidence 0.82", or what stands
    /// in for a verdict on a row that has none.
    private var verdict: String {
        var parts = ["record \(StudyReviewText.grouped(row.record))"]
        switch row.kind {
        case .verdict:
            if let winner = row.winner { parts.append("winner \(winner)") }
            if let result = row.result { parts.append(result) }
            parts.append(
                row.confidence.map {
                    "confidence " + $0.formatted(.number.precision(.fractionLength(2)))
                } ?? "confidence not recorded")
        case .noncompliant:
            parts.append("no verdict")
        case .unreadable:
            parts.append("unreadable")
        }
        return parts.joined(separator: " · ")
    }

    private var provenance: String {
        var parts: [String] = []
        if let judge = row.judge { parts.append("judge \(judge)") }
        if let model = row.judgeModel { parts.append(model) }
        if let baselineWas = row.baselineWas { parts.append("baseline shown as \(baselineWas)") }
        if let conditionWas = row.conditionWas { parts.append("condition shown as \(conditionWas)") }
        return parts.isEmpty ? "no judge or labels recorded" : parts.joined(separator: " · ")
    }

    private var scoreSummary: String? {
        let a = (row.aScores ?? [:]).map { "\($0.key): \($0.value.displayString)" }.sorted()
            .joined(separator: ", ")
        let b = (row.bScores ?? [:]).map { "\($0.key): \($0.value.displayString)" }.sorted()
            .joined(separator: ", ")
        guard !a.isEmpty || !b.isEmpty else { return nil }
        return "A [\(a.isEmpty ? "no scores" : a)] · B [\(b.isEmpty ? "no scores" : b)]"
    }

    private var structuredFieldsSummary: String? {
        guard let fields = row.structuredFields, !fields.isEmpty else { return nil }
        let summary = fields.map { "\($0.key): \($0.value.displayString)" }
            .sorted()
            .joined(separator: ", ")
        return "structured_fields [\(summary)]"
    }
}

/// Import JSONL… sheet for the Input Data section: paste (text area) or
/// choose a file, watch the live parse preview — record count, how many
/// records carry `options`/`target`, or the FIRST error with its line
/// number — and import only when every line parses. Garbage is refused,
/// never coerced into prompt text. All parsing rules live in
/// `TaskPromptsImport` (ExperimentKit, unit-tested); this sheet renders
/// them.
