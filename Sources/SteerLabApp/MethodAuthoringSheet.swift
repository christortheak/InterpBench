import ExperimentKit
import SwiftUI
import UniformTypeIdentifiers

/// Form and agent answers reach one portable authoring owner. Numeric answers
/// remain strings until Python parses them, including integers beyond 2^53.
struct MethodAuthoringSheet: View {
    let workflow: ScienceCatalog.Workflow
    let root: URL
    let client: ClusterClient?
    @Environment(\.dismiss) private var dismiss
    @State private var purpose = ""
    @State private var claim = ""
    @State private var controls = ""
    @State private var selection = ""
    @State private var fields: [String: String] = [:]
    @State private var advanced = "{}"
    @State private var destination = "requests/"
    @State private var review: JSONValue?
    @State private var reviewedAnswers: String?
    @State private var published: String?
    @State private var busy = false
    @State private var failure: String?
    @State private var choosingField: String?
    @State private var showingImporter = false
    @State private var showingExecution = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text(workflow.title).font(.title2)
                Spacer(); Button("Done") { dismiss() }.disabled(busy)
            }
            Text(workflow.purpose)
            Text("Workspace: " + root.path).font(.caption).textSelection(.enabled)
            Form {
                Section("Research decisions") {
                    TextField("What question are you asking?", text: $purpose, axis: .vertical)
                    TextField("What claim could these measurements support?", text: $claim, axis: .vertical)
                    TextField("What baseline, controls and independent data will you use?", text: $controls, axis: .vertical)
                    TextField("How were inputs, dose and settings selected?", text: $selection, axis: .vertical)
                    Text(workflow.claimBoundary).font(.caption).foregroundStyle(.secondary)
                }
                Section("Method inputs and settings") {
                    ForEach(workflow.fields) { field in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                if field.kind == "boolean" {
                                    Picker(field.label, selection: fieldBinding(field)) {
                                        Text("Unspecified").tag("")
                                        Text("True").tag("true"); Text("False").tag("false")
                                    }
                                } else {
                                    TextField(field.label + (field.required ? " *" : ""), text: fieldBinding(field), axis: .vertical)
                                }
                                if ["file", "fileRef", "documentFile"].contains(field.kind) {
                                    Button("Choose…") { choosingField = field.id; showingImporter = true }
                                }
                            }
                            Text(field.help).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    DisclosureGroup("Additional engine settings") {
                        Text("Optional JSON object. These settings cannot override answers above; the engine validates the full config before execution.").font(.caption)
                        TextEditor(text: $advanced).font(.system(.body, design: .monospaced)).frame(minHeight: 90)
                    }
                }
                Section("Review and publish") {
                    Button("Review request and input bytes") { perform {
                        let text = try answerText()
                        review = try await call("draft", text: text)
                        reviewedAnswers = text; published = nil
                    } }
                    if let review {
                        Text("The request is authored, not executed or scientifically qualified. Read the captured inputs and settings below.").font(.caption)
                        reviewText(review)
                    }
                    TextField("New request directory", text: $destination)
                    Button("Publish reviewed request") { perform {
                        guard let reviewedAnswers, reviewedAnswers == (try answerText()), let hash = string(review, "planSHA256") else {
                            throw ExperimentError(reason: "Answers changed; review the request again.")
                        }
                        let result = try await call("publish", text: reviewedAnswers, extra: ["destination": .string(destination), "planSHA256": .string(hash)])
                        published = string(result, "requestFile")
                    } }.disabled(review == nil || published != nil)
                    if let published {
                        Text(published).font(.caption).textSelection(.enabled)
                        Button("Open execution and evidence…") { showingExecution = true }
                    }
                }
            }.formStyle(.grouped).disabled(busy)
            if busy { ProgressView() }
            if let failure { Text(failure).foregroundStyle(.red).textSelection(.enabled) }
        }.padding().frame(minWidth: 820, minHeight: 720)
        .interactiveDismissDisabled(busy)
        .onChange(of: fields) { _, _ in invalidateReview() }
        .onChange(of: purpose) { _, _ in invalidateReview() }
        .onChange(of: claim) { _, _ in invalidateReview() }
        .onChange(of: controls) { _, _ in invalidateReview() }
        .onChange(of: selection) { _, _ in invalidateReview() }
        .onChange(of: advanced) { _, _ in invalidateReview() }
        .onAppear { fields = Dictionary(uniqueKeysWithValues: workflow.fields.map { ($0.id, $0.default ?? "") }) }
        .sheet(isPresented: $showingExecution) {
            DiagnosticLifecycleSheet(root: root, client: client, initialJobID: nil, initialRequestFile: published)
        }
        .fileImporter(isPresented: $showingImporter, allowedContentTypes: [.data]) { result in
            do {
                let url = try result.get().standardizedFileURL
                let prefix = root.standardizedFileURL.path + "/"
                guard url.path.hasPrefix(prefix), let choosingField else { throw ExperimentError(reason: "Choose an input in the captured workspace.") }
                fields[choosingField] = String(url.path.dropFirst(prefix.count))
            } catch { failure = error.localizedDescription }
        }
    }
    private func invalidateReview() { review = nil; reviewedAnswers = nil; published = nil }
    private func fieldBinding(_ field: ScienceCatalog.WorkflowField) -> Binding<String> {
        Binding(get: { fields[field.id] ?? field.default ?? "" }, set: { fields[field.id] = $0 })
    }
    private func answerText() throws -> String {
        // Compose the outer JSON with the advanced object's original text. No
        // generic floating-point decoder can round a researcher's integer seed.
        guard (try JSONSerialization.jsonObject(with: Data(advanced.utf8))) is [String: Any] else { throw ExperimentError(reason: "Additional settings must be one JSON object.") }
        let values: [String: JSONValue] = ["purpose": .string(purpose), "claim": .string(claim), "controls": .string(controls), "selection": .string(selection), "fields": .object(fields.mapValues(JSONValue.string))]
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let text = String(decoding: try encoder.encode(values), as: UTF8.self)
        return String(text.dropLast()) + ",\"advanced\":" + advanced + "}"
    }
    private func call(_ action: String, text: String, extra: [String: JSONValue] = [:]) async throws -> JSONValue {
        try await DiagnosticWorkspace.perform(action, payload: extra.merging(["workspaceRoot": .string(root.path), "operation": .string(workflow.id), "answersText": .string(text)]) { _, new in new })
    }
    private func string(_ value: JSONValue?, _ key: String) -> String? {
        guard case .object(let object) = value, case .string(let text) = object[key] else { return nil }; return text
    }
    private func reviewText(_ value: JSONValue) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Exact request").font(.headline)
            Text(string(value, "requestJSON") ?? "Exact request unavailable; review again.")
                .font(.system(.caption, design: .monospaced)).textSelection(.enabled)
            Text("Captured input files").font(.headline)
            if case .object(let object) = value, case .object(let inputs) = object["inputs"], case .array(let files) = inputs["files"] {
                ForEach(Array(files.enumerated()), id: \.offset) { _, entry in
                    VStack(alignment: .leading) {
                        Text(string(entry, "path") ?? "Missing path")
                        Text(string(entry, "sha256") ?? "Missing hash").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary)
                    }.textSelection(.enabled)
                }
            }
        }
    }
    private func perform(_ action: @escaping @MainActor () async throws -> Void) {
        busy = true; failure = nil
        Task { defer { busy = false }; do { try await action() } catch { failure = error.localizedDescription } }
    }
}
