import AppKit
import ExperimentKit
import SwiftUI
import UniformTypeIdentifiers

/// Source choice and review call the same portable owner as the client and API.
struct FittingCorpusPreparationSheet: View {
    let root: URL
    let modelID: String
    let revision: String
    let onSaved: ([String: String]) -> Void
    @Environment(\.dismiss) private var dismiss
    @State private var sourceKind = "local"
    @State private var files: [String] = []
    @State private var dataset = ""
    @State private var datasetRevision = "main"
    @State private var patterns = ""
    @State private var textColumn = "text"
    @State private var documentColumn = ""
    @State private var selection = "seeded"
    @State private var count = "1000"
    @State private var seed = "0"
    @State private var minChars = "1"
    @State private var scanLimit = "100000"
    @State private var passageChars = "0"
    @State private var tokenPreview = false
    @State private var maxTokens = "128"
    @State private var skipFirst = "16"
    @State private var destination = "prompts/fitting/" + UUID().uuidString.lowercased()
    @State private var showingFiles = false
    @State private var preview: JSONValue?
    @State private var reviewedSpec: String?
    @State private var busy = false
    @State private var failure: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Prepare fitting text").font(.title2)
                Spacer()
                Button("Done") { dismiss() }.disabled(busy)
            }
            Text("Choose existing text from the population you want the lens to represent. This prepares a corpus; it does not generate text, load model weights, or start fitting.")
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    GroupBox("1. Choose the source") {
                        VStack(alignment: .leading, spacing: 10) {
                            Picker("Source", selection: $sourceKind) {
                                Text("Local files").tag("local")
                                Text("Public Hugging Face dataset").tag("huggingface")
                            }.pickerStyle(.segmented)
                            if sourceKind == "local" {
                                Text("Text files are whole documents. JSONL, CSV, and Parquet files supply one record per row. Choosing files copies them into this workspace so the original files remain unchanged.").font(.caption)
                                Button("Choose source files…") { showingFiles = true }
                                ForEach(files, id: \.self) { Text($0).font(.caption).textSelection(.enabled) }
                            } else {
                                Text("Preview downloads the selected public data files (up to 2 GiB) into the Hugging Face cache and records the resolved dataset commit. Dataset scripts and stored credentials are not used.").font(.caption)
                                Button("Use WikiText-103 training files") {
                                    dataset = "Salesforce/wikitext"; datasetRevision = "main"
                                    patterns = "wikitext-103-raw-v1/train-*.parquet"
                                    textColumn = "text"; minChars = "600"
                                }
                                Text("WikiText supplies general encyclopedia prose. Choose your own source if a different text population matters; this example does not start a download.").font(.caption)
                                TextField("Dataset, such as owner/dataset", text: $dataset)
                                TextField("Dataset revision", text: $datasetRevision)
                                TextField("Repository file paths or patterns, one per line", text: $patterns, axis: .vertical)
                                Text("Choose paths for the intended configuration and split from the dataset's Files page. For example, a train/*.parquet pattern selects matching training shards. Use local exports for gated datasets.").font(.caption)
                            }
                            TextField("Text column", text: $textColumn)
                                .help("The column containing the source prose, usually text. Plain text files always use text. Extra source columns are not passed to fitting.")
                            TextField("Document ID column (optional)", text: $documentColumn)
                                .help("Record document identifiers in provenance so you can check whether separately prepared fitting and assessment corpora share documents. This does not split or deduplicate documents automatically.")
                        }.padding(8)
                    }
                    GroupBox("2. Choose the sample") {
                        VStack(alignment: .leading, spacing: 10) {
                            Picker("Selection", selection: $selection) {
                                Text("Reproducible seeded sample").tag("seeded")
                                Text("First eligible records").tag("first")
                            }
                            TextField("Number of records", text: $count)
                            DisclosureGroup("Sampling details") {
                            TextField("Sampling seed", text: $seed)
                                .help("The same source bytes and settings produce the same sample. Seeded selection removes exact duplicate source text.")
                            TextField("Minimum characters after trimming", text: $minChars)
                            TextField("Maximum source records to scan", text: $scanLimit)
                                .help("Sampling covers only the scanned portion of the selected files. Raise this limit when a broader sample is needed; the receipt records the actual scope.")
                            TextField("Passage characters (0 keeps whole records)", text: $passageChars)
                                .help("A positive number takes one seeded character window from each selected record. It can cut through a word. Fitting applies its separate token limit afterward.")
                            }
                            Text("Sampling considers at most \(scanLimit) records. The preview reports how much of the source was actually read.").font(.caption)
                            Text("Keep assessment text separate, preferably using a different source split. A successful fit alone does not establish readout quality.").font(.caption)
                            DisclosureGroup("Token preview") {
                                Toggle("Check with the selected model's locally cached tokenizer", isOn: $tokenPreview)
                                Text("Optional. No tokenizer or model weights are downloaded. If the exact tokenizer is unavailable, the corpus can still be prepared and the fitting pilot checks lengths.").font(.caption)
                                TextField("Token limit", text: $maxTokens)
                                TextField("Leading token positions to omit", text: $skipFirst)
                            }
                        }.padding(8)
                    }
                    GroupBox("3. Review and save") {
                        VStack(alignment: .leading, spacing: 10) {
                            Button(sourceKind == "local" ? "Read files and preview sample" : "Download selected data and preview sample") {
                                perform {
                                    let spec = try specification()
                                    preview = try await DiagnosticWorkspace.perform("corpus-preview", payload: ["workspaceRoot": .string(root.path), "specText": .string(spec)])
                                    reviewedSpec = spec
                                }
                            }
                            if let preview { previewView(preview) }
                            DisclosureGroup("Storage details") {
                                Text("Preview keeps a candidate in workspace scratch. Save creates a new folder containing corpus.jsonl and preparation.json; it never replaces a previous corpus.").font(.caption)
                                TextField("New corpus folder", text: $destination)
                            }
                            Button("Save corpus and use for fitting") {
                                perform {
                                    guard reviewedSpec == (try specification()), let id = text(preview, "previewID"), let hash = text(preview, "planSHA256") else {
                                        throw ExperimentError(reason: "The source or settings changed. Preview the sample again before saving.")
                                    }
                                    let result = try await DiagnosticWorkspace.perform("corpus-publish", payload: ["workspaceRoot": .string(root.path), "previewID": .string(id), "planSHA256": .string(hash), "destination": .string(destination)])
                                    guard case .object(let object) = result, case .object(let inputs) = object["fittingInputs"] else { throw ExperimentError(reason: "Corpus publication returned no fitting inputs.") }
                                    var paths: [String: String] = [:]
                                    for (key, value) in inputs { if let path = text(value, "path") { paths[key] = path } }
                                    onSaved(paths); dismiss()
                                }
                            }.disabled(preview == nil)
                        }.padding(8)
                    }
                }
            }.disabled(busy)
            if busy { ProgressView("Reading source data and preparing the sample. Large files and downloads can take several minutes.") }
            if let failure { Text(failure).foregroundStyle(.red).textSelection(.enabled) }
        }.padding(24).frame(minWidth: 760, minHeight: 700)
        .interactiveDismissDisabled(busy)
        .fileImporter(isPresented: $showingFiles, allowedContentTypes: [.data], allowsMultipleSelection: true) { result in
            perform {
                let selected = try result.get()
                let workspace = root
                files = try await Task.detached {
                    let folder = workspace.appending(path: ".steerlab/corpus-sources/" + UUID().uuidString.lowercased())
                    // Refuse pre-existing symlink ancestors before staging chosen files.
                    var ancestor = workspace
                    for component in [".steerlab", "corpus-sources"] {
                        ancestor.append(path: component)
                        if (try? ancestor.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) == true {
                            throw ExperimentError(reason: "The corpus source directory is a link. Choose an ordinary workspace directory.")
                        }
                    }
                    var total: Int64 = 0
                    for url in selected {
                        let properties = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
                        guard properties.isRegularFile == true, properties.isSymbolicLink != true else { throw ExperimentError(reason: "Choose ordinary source files.") }
                        total += Int64(properties.fileSize ?? 0)
                    }
                    guard total <= 2 * 1024 * 1024 * 1024 else { throw ExperimentError(reason: "Choose source files totaling at most 2 GiB.") }
                    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                    do {
                        var imported: [String] = []
                        for (index, url) in selected.enumerated() {
                            let target = folder.appending(component: "\(index)-" + url.lastPathComponent)
                            try FileManager.default.copyItem(at: url, to: target)
                            imported.append(String(target.path.dropFirst(workspace.path.count + 1)))
                        }
                        return imported
                    } catch {
                        try? FileManager.default.removeItem(at: folder)
                        throw error
                    }
                }.value
                preview = nil; reviewedSpec = nil
            }
        }
    }

    @ViewBuilder private func previewView(_ value: JSONValue) -> some View {
        if case .object(let object) = value {
            if case .object(let counts) = object["counts"] {
                Text("Sample: \(number(counts["selected"])) records from \(number(counts["scanned"])) scanned; \(number(counts["tooShort"])) too short, \(number(counts["missingText"])) without text, and \(number(counts["duplicateText"])) duplicates skipped.")
            }
            if case .array(let warnings) = object["warnings"] {
                ForEach(Array(warnings.enumerated()), id: \.offset) { _, warning in
                    if case .string(let message) = warning { Text(message).font(.caption).foregroundStyle(.secondary) }
                }
            }
            if case .object(let tokens) = object["tokenReview"] {
                if case .string(let message) = tokens["message"] { Text(message).font(.caption) }
                if tokens["status"] == .string("measured") {
                    Text("Token preview: \(number(tokens["truncatedRows"])) rows will be truncated; \(number(tokens["tooShortRows"])) leave too few positions to fit.").font(.caption)
                }
            }
            if case .array(let examples) = object["examples"] {
                ForEach(Array(examples.enumerated()), id: \.offset) { index, example in
                    if let excerpt = text(example, "text") {
                        VStack(alignment: .leading) {
                            Text("Sample \(index + 1) (first 500 characters)").font(.caption.bold())
                            Text(excerpt).textSelection(.enabled)
                        }.padding(8).frame(maxWidth: .infinity, alignment: .leading).background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
                    }
                }
            }
        }
    }
    private func number(_ value: JSONValue?) -> String {
        guard let value, let data = try? JSONEncoder().encode(value) else { return "?" }
        return String(decoding: data, as: UTF8.self)
    }
    private func text(_ value: JSONValue?, _ key: String) -> String? {
        guard case .object(let object) = value, case .string(let text) = object[key] else { return nil }
        return text
    }
    private func specification() throws -> String {
        var spec: [String: Any] = ["textColumn": textColumn, "documentColumn": documentColumn.isEmpty ? NSNull() : documentColumn, "selection": selection]
        for (key, value) in [("count", count), ("seed", seed), ("minChars", minChars), ("scanLimit", scanLimit), ("passageChars", passageChars)] {
            guard let parsed = UInt64(value) else { throw ExperimentError(reason: key + " needs a nonnegative whole number.") }
            spec[key] = parsed
        }
        spec["source"] = sourceKind == "local" ? ["kind": "local", "files": files] : ["kind": "huggingface", "dataset": dataset, "revision": datasetRevision, "files": patterns.split(whereSeparator: \.isNewline).map(String.init)]
        if tokenPreview {
            guard let limit = Int(maxTokens), let skip = Int(skipFirst) else { throw ExperimentError(reason: "Token settings need whole numbers.") }
            spec["tokenizer"] = ["modelID": modelID, "revision": revision, "maxSeqLen": limit, "skipFirst": skip]
        }
        return String(decoding: try JSONSerialization.data(withJSONObject: spec, options: [.sortedKeys]), as: UTF8.self)
    }
    private func perform(_ operation: @escaping @MainActor () async throws -> Void) {
        busy = true; failure = nil
        Task { do { try await operation() } catch { failure = error.localizedDescription }; busy = false }
    }
}
