import ExperimentKit
import SwiftUI
import WebKit

/// The native Results Explorer pane: the separate team's explorer SPA
/// (vendored at `results-explorer/`, embedded build committed at
/// `web/results-explorer/`) presented in a WKWebView — no browser, no
/// server, no ports. A custom URL scheme serves the bundled assets AND the
/// page's `/api/tree` + `/api/file` reads, answered directly from the
/// active workspace's `runs/` directory through the containment-checked
/// `ResultsExplorerBridge`. Everything stays on-box and read-only.
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
            let body = Data("\(error)".utf8)
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

/// The WKWebView hosting the embedded explorer, deep-linked to a run.
struct ResultsExplorerPane: NSViewRepresentable {
    let runName: String?

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.setURLSchemeHandler(
            ResultsExplorerSchemeHandler(
                runsRoot: ExperimentStore.runsDirectory),
            forURLScheme: ResultsExplorerSchemeHandler.scheme)
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
            (runName == nil
                ? "browse every run in this workspace with the embedded "
                : "open this run in the embedded ")
                + "Results Explorer — the reading views (overview, concept "
                + "evidence, optimization, effects, panels, generations, "
                + "provenance) served natively from the workspace, "
                + "read-only, no browser")
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
                }
                .padding(10)
                Divider()
                ResultsExplorerPane(runName: runName)
            }
            .frame(minWidth: 1180, minHeight: 780)
        }
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
        }
    }
}
