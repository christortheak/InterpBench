import AppKit
import ExperimentKit
import SwiftUI
import UniformTypeIdentifiers
import WebKit

/// The native Results Explorer pane: the explorer SPA (source at
/// `results-explorer/`, built into the app's `web/results-explorer/`)
/// presented in a WKWebView — no browser, no server, no ports. A custom URL
/// scheme serves the bundled assets AND the page's `/api/tree` +
/// `/api/file` reads, answered directly from the active workspace's `runs/`
/// directory through the containment-checked `ResultsExplorerBridge`.
/// Everything stays on-box, and the workspace is only ever read: the one
/// thing the page can write is a file the reader places with a save panel
/// (`ResultsExplorerSaveHandler`).
final class ResultsExplorerSchemeHandler: NSObject, WKURLSchemeHandler {
    static let scheme = "steerlab-explorer"
    private let runsRoot: URL

    init(runsRoot: URL) {
        self.runsRoot = runsRoot
    }

    func webView(_ webView: WKWebView, start task: WKURLSchemeTask) {
        guard let url = task.request.url else { return }
        do {
            let (data, contentType) = try respond(to: url)
            let response = HTTPURLResponse(
                url: url, statusCode: 200, httpVersion: "HTTP/1.1",
                headerFields: [
                    "Content-Type": contentType,
                    "Cache-Control": "no-cache",
                ])!
            task.didReceive(response)
            task.didReceive(data)
            task.didFinish()
        } catch {
            let body = Data(error.localizedDescription.utf8)
            let response = HTTPURLResponse(
                url: url, statusCode: 404, httpVersion: "HTTP/1.1",
                headerFields: ["Content-Type": "text/plain; charset=utf-8"])!
            task.didReceive(response)
            task.didReceive(body)
            task.didFinish()
        }
    }

    func webView(_ webView: WKWebView, stop task: WKURLSchemeTask) {}

    private func respond(to url: URL) throws -> (Data, String) {
        let path = url.path.isEmpty ? "/" : url.path
        switch path {
        case "/api/tree":
            let relative = Self.queryValue("path", in: url) ?? ""
            let entries = try ResultsExplorerBridge.tree(
                path: relative, under: runsRoot)
            return (try JSONEncoder().encode(entries),
                    "application/json; charset=utf-8")
        case "/api/file":
            let relative = Self.queryValue("path", in: url) ?? ""
            let data = try ResultsExplorerBridge.fileData(
                path: relative, under: runsRoot,
                offset: Self.queryValue("offset", in: url).flatMap(Int.init),
                length: Self.queryValue("length", in: url).flatMap(Int.init))
            return (data,
                    ResultsExplorerBridge.contentType(
                        for: (relative as NSString).lastPathComponent))
        default:
            // Bundled SPA assets — same containment discipline, rooted at
            // the shipped web/results-explorer directory.
            let assetsRoot = try CodeResources.webAssets()
                .appending(component: "results-explorer")
            let assetPath = (path == "/" || path == "/index.html")
                ? "index.html"
                : String(path.dropFirst())
            let data = try ResultsExplorerBridge.fileData(
                path: assetPath, under: assetsRoot)
            return (data,
                    ResultsExplorerBridge.contentType(
                        for: (assetPath as NSString).lastPathComponent))
        }
    }

    private static func queryValue(_ name: String, in url: URL) -> String? {
        URLComponents(url: url, resolvingAgainstBaseURL: false)?
            .queryItems?
            .first { $0.name == name }?
            .value
    }
}

/// The explorer's export and download controls, inside the app. A web view
/// has no downloads folder, so the page posts what it wants saved (a table
/// it built, or the path of one file in `runs/`) and this handler asks the
/// reader where to put it. The page never names the destination, and
/// `ResultsExplorerBridge.save` refuses any place inside `runs/`.
///
/// The page's side is `results-explorer/app/lib/save.ts`; the reply it
/// reads is `{"state": "saved", "name": …}`, `{"state": "cancelled"}`, or
/// `{"state": "failed", "message": …}`.
final class ResultsExplorerSaveHandler: NSObject, WKScriptMessageHandlerWithReply {
    static let name = "steerlabSave"
    private let runsRoot: URL

    init(runsRoot: URL) {
        self.runsRoot = runsRoot
    }

    func userContentController(
        _ userContentController: WKUserContentController,
        didReceive message: WKScriptMessage,
        replyHandler: @escaping @MainActor @Sendable (Any?, String?) -> Void
    ) {
        // Only the explorer's own page, in the main frame, may ask.
        guard message.frameInfo.isMainFrame,
            message.frameInfo.securityOrigin.protocol
                == ResultsExplorerSchemeHandler.scheme,
            let parsed = ResultsExplorerBridge.saveRequest(
                fromMessage: message.body)
        else {
            replyHandler(
                [
                    "state": "failed",
                    "message": "The explorer sent a save request the app "
                        + "did not understand, so nothing was saved.",
                ], nil)
            return
        }
        let request = parsed.request
        let root = runsRoot
        let panel = NSSavePanel()
        panel.nameFieldStringValue = parsed.suggestedName
        panel.canCreateDirectories = true
        panel.isExtensionHidden = false
        let fileExtension =
            (parsed.suggestedName as NSString).pathExtension
        if let type = UTType(filenameExtension: fileExtension) {
            panel.allowedContentTypes = [type]
        }
        panel.message =
            "Choose where to save a copy. Nothing in the workspace's runs "
            + "folder is changed."
        let finish: (NSApplication.ModalResponse) -> Void = { response in
            guard response == .OK, let destination = panel.url else {
                replyHandler(["state": "cancelled"], nil)
                return
            }
            // Off the main actor: a run's generations file can be large.
            Task.detached {
                let outcome: Result<Int, any Error> = Result {
                    try ResultsExplorerBridge.save(
                        request, to: destination, runsRoot: root)
                }
                let name = destination.lastPathComponent
                await MainActor.run {
                    switch outcome {
                    case .success:
                        replyHandler(["state": "saved", "name": name], nil)
                    case .failure(let error):
                        let reason =
                            (error as? ExperimentError)?.reason
                            ?? error.localizedDescription
                        replyHandler(
                            ["state": "failed", "message": reason], nil)
                    }
                }
            }
        }
        if let window = message.webView?.window {
            panel.beginSheetModal(for: window, completionHandler: finish)
        } else {
            panel.begin(completionHandler: finish)
        }
    }
}

/// The WKWebView hosting the embedded explorer, deep-linked to a run.
struct ResultsExplorerPane: NSViewRepresentable {
    let runName: String?

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.setURLSchemeHandler(
            ResultsExplorerSchemeHandler(
                runsRoot: ExperimentStore.runsDirectory),
            forURLScheme: ResultsExplorerSchemeHandler.scheme)
        configuration.userContentController.addScriptMessageHandler(
            ResultsExplorerSaveHandler(
                runsRoot: ExperimentStore.runsDirectory),
            contentWorld: .page,
            name: ResultsExplorerSaveHandler.name)
        let webView = WKWebView(frame: .zero, configuration: configuration)
        var components = URLComponents()
        components.scheme = ResultsExplorerSchemeHandler.scheme
        components.host = "app"
        components.path = "/index.html"
        var items = [URLQueryItem(name: "embedded", value: "steerlab")]
        if let runName {
            items.append(URLQueryItem(name: "run", value: runName))
        }
        items.append(
            URLQueryItem(
                name: "workspace",
                value: ExperimentStore.runsDirectory
                    .deletingLastPathComponent().lastPathComponent))
        components.queryItems = items
        if let url = components.url {
            webView.load(URLRequest(url: url))
        }
        return webView
    }

    func updateNSView(_ webView: WKWebView, context: Context) {}
}

/// The Results tab affordance: opens the explorer in a large sheet —
/// deep-linked when a run is given, on the whole workspace's run picker
/// otherwise.
struct ResultsExplorerButton: View {
    var runName: String?
    @State private var showingExplorer = false

    var body: some View {
        Button {
            showingExplorer = true
        } label: {
            Label("Results Explorer", systemImage: "chart.bar.doc.horizontal")
        }
        .controlSize(.small)
        .help(
            runName == nil
                ? "browse every run in this workspace in the embedded "
                    + "Results Explorer — read-only"
                : "open this run in the embedded Results Explorer — read-only")
        .sheet(isPresented: $showingExplorer) {
            VStack(spacing: 0) {
                HStack {
                    Text(runName.map { "Results Explorer — \($0)" }
                        ?? "Results Explorer")
                        .font(.callout.monospaced())
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Spacer()
                    Button("Done") { showingExplorer = false }
                        .keyboardShortcut(.cancelAction)
                        .help("close the explorer and go back to the run list")
                }
                .padding(10)
                Divider()
                if Self.assetsAvailable {
                    ResultsExplorerPane(runName: runName)
                } else {
                    assetsMissingState
                }
            }
            // Ideal, not minimum: 1180 x 780 as a floor is taller than the
            // usable height of a 13-inch display, and a sheet cannot be
            // resized below its minimum.
            .frame(
                minWidth: 720, idealWidth: 1180, maxWidth: .infinity,
                minHeight: 460, idealHeight: 780, maxHeight: .infinity)
        }
    }

    /// A NATIVE state for a build whose embedded explorer assets are
    /// missing. Without it the WKWebView renders the scheme handler's 404
    /// body as bare text, which reads like a broken page rather than an
    /// incomplete install.
    private var assetsMissingState: some View {
        ContentUnavailableView {
            Label(
                "The embedded Results Explorer is not in this build",
                systemImage: "questionmark.folder")
        } description: {
            Text(
                "Its web assets are missing, so there is nothing to serve. "
                    + "The run's artifacts are still readable in the Results "
                    + "detail pane, and in Finder.")
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// Cheap presence check for the shipped SPA — one `fileExists`, run when
    /// the sheet is presented.
    private static var assetsAvailable: Bool {
        guard let root = try? CodeResources.webAssets() else { return false }
        return FileManager.default.fileExists(
            atPath: root
                .appending(components: "results-explorer", "index.html").path)
    }
}

/// The remote (server) detail header's variant: the explorer reads LOCAL
/// run directories, so a server run offers it only once its directory
/// exists in the local workspace (Import Evidence / downloaded results);
/// until then the button shows disabled with the reason.
struct RemoteResultsExplorerButton: View {
    let runID: String

    var body: some View {
        if FileManager.default.fileExists(
            atPath: ExperimentStore.runsDirectory
                .appending(component: runID).path)
        {
            ResultsExplorerButton(runName: runID)
        } else {
            VStack(alignment: .trailing, spacing: 1) {
                Button {} label: {
                    Label(
                        "Results Explorer",
                        systemImage: "chart.bar.doc.horizontal")
                }
                .controlSize(.small)
                .disabled(true)
                .help(
                    "the embedded Results Explorer reads local run directories "
                        + "— Import Evidence (or download this run's results) "
                        + "and it becomes viewable here")
                // A disabled button whose only explanation is a tooltip is
                // an explanation most researchers never see.
                Text("no local copy yet — Import Evidence first")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
        }
    }
}
