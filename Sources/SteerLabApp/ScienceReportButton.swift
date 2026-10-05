import ExperimentKit
import SwiftUI
import WebKit

/// "Open Report" for a run folder that holds a stored scientific report.
///
/// The app draws nothing itself. It shows the one self-contained HTML file
/// the shared Python client makes (`ScienceReport`): the page the engine wrote
/// with a new run, or one drawn on request outside the run folder. That is the
/// same file a browser opens and a colleague can be sent.
struct ScienceReportButton: View {
    let runDirectory: URL
    let root: URL
    @State private var page: ScienceReport.Page?
    @State private var failure: String?
    @State private var busy = false

    var body: some View {
        if ScienceReport.hasReport(in: runDirectory) {
            Button {
                open()
            } label: {
                Label("Open Report", systemImage: "doc.richtext")
            }
            .controlSize(.small)
            .disabled(busy)
            .help(
                "open this assessment as a readable page — the same HTML file "
                + "you can open in a browser or send to a colleague; the run "
                + "folder itself is never written to")
            .sheet(item: $page) { page in
                ScienceReportSheet(page: page)
            }
            .alert(
                "The report could not be opened",
                isPresented: Binding(get: { failure != nil }, set: { if !$0 { failure = nil } })
            ) {
                Button("OK") { failure = nil }
            } message: {
                Text(failure ?? "")
            }
        }
    }

    private func open() {
        busy = true
        Task {
            defer { busy = false }
            do {
                page = try await ScienceReport.page(for: runDirectory, root: root)
            } catch {
                failure = error.localizedDescription
            }
        }
    }
}

/// The report page in a sheet, with the two ways out of the app a researcher
/// needs: a browser (to print or save as PDF) and Finder (to send the file).
struct ScienceReportSheet: View {
    let page: ScienceReport.Page
    /// Says where the page came from, before its path. Nil keeps the wording
    /// for a stored scientific report.
    var origin: String? = nil
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("Report").font(.headline)
                    Text((origin.map { $0 + ": " } ?? (page.inRunDirectory
                        ? "Stored with the run: "
                        : "Drawn from the run’s stored report, outside the run folder: ")) + page.url.path)
                        .font(.caption).foregroundStyle(.secondary)
                        .lineLimit(1).truncationMode(.middle)
                        .textSelection(.enabled).help(page.url.path)
                }
                Spacer()
                Button("Open in Browser") { NSWorkspace.shared.open(page.url) }
                    .help("open this same file in your browser, to print it or save it as a PDF")
                Button("Reveal in Finder") { NSWorkspace.shared.activateFileViewerSelecting([page.url]) }
                    .help("show the file, a single self-contained page you can send to a colleague")
                Button("Done") { dismiss() }.keyboardShortcut(.defaultAction)
            }
            .padding(12)
            Divider()
            ScienceReportPage(url: page.url)
        }
        .frame(minWidth: 860, minHeight: 640)
    }
}

/// One local HTML file, shown as it is. Script is off and nothing but that
/// file may load: the page needs neither, and the view should not be a way to
/// reach anything else.
private struct ScienceReportPage: NSViewRepresentable {
    let url: URL

    func makeCoordinator() -> Coordinator { Coordinator(file: url) }

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.defaultWebpagePreferences.allowsContentJavaScript = false
        configuration.websiteDataStore = .nonPersistent()
        let view = WKWebView(frame: .zero, configuration: configuration)
        view.navigationDelegate = context.coordinator
        view.loadFileURL(url, allowingReadAccessTo: url)
        return view
    }

    func updateNSView(_ view: WKWebView, context: Context) {}

    @MainActor
    final class Coordinator: NSObject, WKNavigationDelegate {
        private let file: URL

        init(file: URL) { self.file = file.standardizedFileURL }

        func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction) async -> WKNavigationActionPolicy {
            // In-page links (the contents list, comparison numbers) stay on
            // this file. Any other destination is refused.
            guard let target = navigationAction.request.url, target.isFileURL,
                  target.standardizedFileURL.path == file.path else { return .cancel }
            return .allow
        }
    }
}
