import ExperimentKit
import SwiftUI

struct TextFileReader: View {
    let url: URL
    @Environment(\.dismiss) private var dismiss
    @State private var offset: UInt64 = 0
    @State private var previous: [UInt64] = []
    @State private var page: TextFilePage.Page?
    @State private var failure: String?
    @State private var busy = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(url.lastPathComponent).font(.headline)
                Spacer()
                Button("Done") { dismiss() }
            }
            Text("Read-only file contents. Pages are bounded for responsiveness; a long record may continue on the next page.")
                .font(.caption).foregroundStyle(.secondary)
            if let failure { Text(failure).foregroundStyle(.red) }
            ScrollView([.vertical, .horizontal]) {
                Text(page?.text ?? "Loading…")
                    .font(.system(.body, design: .monospaced)).textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            HStack {
                Button("Previous page") { offset = previous.removeLast() }
                    .disabled(previous.isEmpty || busy)
                Button("Next page") {
                    if let page { previous.append(offset); offset = page.nextOffset }
                }.disabled(page?.hasMore != true || busy)
                if let page {
                    Text("Bytes \(offset)–\(page.nextOffset) of \(page.totalBytes)")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if busy { ProgressView().controlSize(.small) }
            }
        }.padding().frame(minWidth: 700, minHeight: 500)
        .task(id: offset) {
            busy = true; failure = nil
            let requested = offset
            do {
                let loaded = try await Task.detached { try TextFilePage.read(url: url, offset: requested) }.value
                guard !Task.isCancelled else { return }
                page = loaded
            } catch { failure = error.localizedDescription }
            busy = false
        }
    }
}

struct ReadTextFileButton: View {
    let url: URL
    var title = "Read file…"
    @State private var showing = false
    var body: some View {
        Button(title) { showing = true }
            .sheet(isPresented: $showing) { TextFileReader(url: url) }
    }
}

/// Browsing reads directory metadata; file contents are loaded only on selection.
struct TrainingDataBrowseButton: View {
    let path: String
    @State private var showing = false
    var body: some View {
        Button("Browse examples…") { showing = true }
            .disabled(path.isEmpty)
            .sheet(isPresented: $showing) { TrainingDataFiles(path: path) }
    }
}

private struct TrainingDataFiles: View {
    let path: String
    @Environment(\.dismiss) private var dismiss
    @State private var files: [URL] = []
    @State private var failure: String?
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Training or validation source files").font(.headline)
                Spacer()
                Button("Done") { dismiss() }
            }
            Text("Read the source examples here. Use Analyze Training Plan in the trainer for parsed counts and format checks before training.")
                .font(.caption).foregroundStyle(.secondary)
            if let failure { Text(failure) }
            List(files, id: \.self) { file in
                if ["jsonl", "json", "txt", "md", "csv"].contains(file.pathExtension.lowercased()) {
                    ReadTextFileButton(url: file, title: file.lastPathComponent)
                } else {
                    Text(file.lastPathComponent + " — preview this format in its document app")
                }
            }
        }.padding().frame(width: 640, height: 420)
        .task {
            do {
                let folder = URL(filePath: path)
                files = try await Task.detached {
                    if try folder.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true {
                        return [folder]
                    }
                    return try FileManager.default.contentsOfDirectory(at: folder,
                        includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles])
                        .filter { try $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true }
                        .sorted { $0.lastPathComponent < $1.lastPathComponent }
                }.value
                if files.isEmpty { failure = "No source files in this folder. Choose a data folder or drop files on the data row." }
            } catch { failure = error.localizedDescription }
        }
    }
}
