import ExperimentKit
import AppKit
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
    @State private var destination = "requests/" + UUID().uuidString.lowercased()
    @State private var step = 0
    @State private var modelOptions: [String] = []
    @State private var modelNotice: String?
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
            Picker("Authoring step", selection: $step) {
                ForEach(Array(availableSteps.enumerated()), id: \.element) { index, value in
                    Text("\(index + 1). " + stepNames[value]).tag(value)
                }
            }.pickerStyle(.segmented)
            Form {
                if step == 0 {
                    Section("Choose the model") {
                        Text("Fields marked required must be filled before review. Choosing inputs does not start training or download model weights.")
                            .font(.caption).foregroundStyle(.secondary)
                        ForEach(workflow.fields.filter { ["modelID", "revision"].contains($0.id) }) { field in
                            inputRow(field)
                        }
                    }
                } else if step == 1 {
                    Section("Supply your data") {
                        Text(workflow.id == "jlens-fit"
                             ? "Supply varied text from the population where you want to use the lens. This is fitting text, not positive/negative concept data. Reserve separate passages to assess readouts afterward. Files must be in this workspace before selection."
                             : "Use existing files, prepare them yourself, or ask an agent for an authoring prompt. Keep training, validation, and final-test examples separate. Files must be in this workspace before you select them.")
                            .font(.caption).foregroundStyle(.secondary)
                        ForEach(workflow.fields.filter { isData($0) }) { field in inputRow(field) }
                        if workflow.id == "jlens-fit" {
                            Button("Copy corpus instructions for my agent") {
                                do {
                                    let guide = try ScienceCatalog.guide("jlens").text
                                    NSPasteboard.general.clearContents()
                                    NSPasteboard.general.setString(guide, forType: .string)
                                } catch { failure = error.localizedDescription }
                            }
                        }
                    }
                } else if step == 2 {
                    Section("Training or evaluation settings") {
                        ForEach(workflow.fields.filter { !isData($0) && !["modelID", "revision"].contains($0.id) }) { field in inputRow(field) }
                        DisclosureGroup("Additional engine settings") {
                            Text("Optional JSON object. These settings cannot override answers above; the engine validates the full config before execution.").font(.caption)
                            TextEditor(text: $advanced).font(.system(.body, design: .monospaced)).frame(minHeight: 90)
                        }
                    }
                } else if step == 3 {
                    Section("Explain the plan in your own words") {
                        Text("These notes are saved beside the request so you and collaborators can understand the plan later. They do not set the optimizer, choose data, or score your scientific claims. Each note needs a brief answer to save. If a choice is unresolved, say so; these fields do not require a polished research proposal.")
                            .font(.callout)
                        note("Research question", text: $purpose, example: workflow.id == "jlens-fit" ? "How useful are the lens readouts on the text population I want to study?" : "Does this intervention increase the chosen behavior on new prompts?")
                        note("What the result could show", text: $claim, example: workflow.id == "jlens-fit" ? "An exploratory instrument for this model and corpus. Fitting alone does not validate the meaning of a readout." : "An exploratory behavioral effect on this model and prompt population; not a general claim about the concept.")
                        note("Comparisons and checks", text: $controls, example: workflow.id == "jlens-fit" ? "Qualify the lens on its exact runtime, then assess readouts on separate text. Compare fits from different corpus samples if appropriate." : "Compare the same model with and without the vector. Check preserved judgments and abilities on separate examples.")
                        note("How settings and data were chosen", text: $selection, example: workflow.id == "jlens-fit" ? "Start with a four-row timing pilot. Explain the corpus source, layer coverage, and any later changes made after inspecting results." : "State whether settings are exploratory, chosen on validation data, or fixed in advance. Reserve final-test data for the final evaluation.")
                        Text(workflow.claimBoundary).font(.caption).foregroundStyle(.secondary)
                    }
                } else {
                    Section("Check the plan before saving") {
                        Text("Review reads the selected files, records their fingerprints, and assembles your settings and notes. It does not run the model. Saving checks that those inputs still match, then creates a new request in this workspace. Execution is a separate step.")
                            .font(.callout)
                        Button("Check inputs and preview plan") { perform {
                            let text = try answerText()
                            review = try await call("draft", text: text)
                            reviewedAnswers = text; published = nil
                        } }
                        if let review { reviewText(review) }
                        DisclosureGroup("Storage details") {
                            Text("A unique folder is chosen automatically under requests/. Existing requests are never replaced. You normally do not need to change this location.").font(.caption)
                            TextField("Workspace-relative folder", text: $destination)
                                .disabled(published != nil)
                        }
                        Button("Save reviewed plan") { perform {
                            guard let reviewedAnswers, reviewedAnswers == (try answerText()), let hash = string(review, "planSHA256") else {
                                throw ExperimentError(reason: "Answers changed; review the request again.")
                            }
                            let result = try await call("publish", text: reviewedAnswers, extra: ["destination": .string(destination), "planSHA256": .string(hash)])
                            published = string(result, "requestFile")
                        } }.disabled(review == nil || published != nil)
                        if let published {
                            Text(published).font(.caption).textSelection(.enabled)
                            Button("Choose execution and collect results…") { showingExecution = true }
                        }
                    }
                }
            }.formStyle(.grouped).disabled(busy)
            HStack {
                Button("Back") { moveStep(-1) }.disabled(step == availableSteps.first || busy)
                Spacer()
                Button("Next") { moveStep(1) }.disabled(step == availableSteps.last || busy)
            }
            if busy { ProgressView() }
            if let failure { Text(failure).foregroundStyle(.red).textSelection(.enabled) }
        }.padding().frame(minWidth: 820, minHeight: 720)
        .interactiveDismissDisabled(busy)
        .onChange(of: fields) { old, new in
            if old["modelID"] != new["modelID"] { fields["revision"] = ""; modelNotice = nil }
            invalidateReview()
        }
        .onChange(of: purpose) { _, _ in invalidateReview() }
        .onChange(of: claim) { _, _ in invalidateReview() }
        .onChange(of: controls) { _, _ in invalidateReview() }
        .onChange(of: selection) { _, _ in invalidateReview() }
        .onChange(of: advanced) { _, _ in invalidateReview() }
        .onAppear {
            fields = Dictionary(uniqueKeysWithValues: workflow.fields.map { ($0.id, $0.default ?? "") })
            step = availableSteps.first ?? 3
        }
        .task {
            guard let client else { return }
            do { modelOptions = try await client.state().models }
            catch { modelNotice = "Could not list engine models. You can still enter a verified model identifier: " + error.localizedDescription }
        }
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
    private let stepNames = ["Model", "Data", "Settings", "Notes", "Review"]
    private var availableSteps: [Int] {
        var result: [Int] = []
        if workflow.fields.contains(where: { ["modelID", "revision"].contains($0.id) }) { result.append(0) }
        if workflow.fields.contains(where: isData) { result.append(1) }
        result.append(contentsOf: [2, 3, 4])
        return result
    }
    private func moveStep(_ delta: Int) {
        guard let index = availableSteps.firstIndex(of: step), availableSteps.indices.contains(index + delta) else { return }
        step = availableSteps[index + delta]
    }
    private func isData(_ field: ScienceCatalog.WorkflowField) -> Bool {
        ["file", "fileRef", "documentFile", "files"].contains(field.kind)
    }
    private func note(_ title: String, text: Binding<String>, example: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title).font(.headline)
            TextField(title, text: text, axis: .vertical).labelsHidden()
            Text("Example to adapt: " + example).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
    }
    private func inputRow(_ field: ScienceCatalog.WorkflowField) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(field.label + (field.required ? " (required)" : " (optional)")).font(.headline)
            if field.id == "modelID" {
                if client == nil {
                    Text("Choose a Python engine in Compute to list its models and inspect their versions. You can also enter a verified model identifier from another execution environment below.")
                        .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
                if !modelOptions.isEmpty {
                    Picker("Prepared model", selection: fieldBinding(field)) {
                        Text("Choose a model…").tag("")
                        ForEach(Array(Set(modelOptions + [fields[field.id] ?? ""])).filter { !$0.isEmpty }.sorted(), id: \.self) { model in
                            Text(model).tag(model)
                        }
                    }
                }
                DisclosureGroup(modelOptions.isEmpty ? "Enter a model identifier" : "Use another model") {
                    TextField("Model identifier, such as owner/model-name", text: fieldBinding(field))
                    Text("Use the Python/Hugging Face model identifier. An MLX conversion is a different artifact. Models for another engine can be declared here and prepared there before execution.").font(.caption)
                }
                if let modelNotice { Text(modelNotice).font(.caption).foregroundStyle(.secondary) }
            } else if field.id == "revision" {
                HStack {
                    TextField("Verified model commit", text: fieldBinding(field)).labelsHidden()
                    Button("Inspect selected model") { inspectModel() }
                        .disabled(client == nil || (fields["modelID"] ?? "").isEmpty || busy)
                }
            } else if field.kind == "boolean" {
                Picker(field.label, selection: fieldBinding(field)) {
                    Text("Unspecified").tag("")
                    Text("True").tag("true"); Text("False").tag("false")
                }.labelsHidden()
            } else {
                HStack {
                    TextField(field.label, text: fieldBinding(field), axis: .vertical).labelsHidden()
                    if ["file", "fileRef", "documentFile"].contains(field.kind) {
                        Button("Choose file…") { choosingField = field.id; showingImporter = true }
                    }
                }
            }
            Text(field.help).font(.caption).foregroundStyle(.secondary)
            if let example = field.example {
                DisclosureGroup("File format and example") {
                    Text("One JSON object per line. Adapt this syntax to your research question; this example is not used as training data.").font(.caption)
                    Text(example).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                    if example.contains("\"options\"") {
                        Text("Choices must tokenize as single tokens for OptVec training. Spell out their meaning in the prompt. Give each row a unique id, and name the intended answer explicitly in target.").font(.caption)
                    }
                }
            }
        }.padding(.vertical, 4)
    }
    private func inspectModel() {
        guard let client else { return }
        let model = fields["modelID"] ?? ""
        perform {
            let result = try await client.modelLoadPreflight(model)
            guard fields["modelID"] == model else { return }
            guard let revision = result.revision, revision.count == 40 else {
                throw ExperimentError(reason: "No exact commit was returned. Prepare this model in Compute, or supply a verified commit from its execution environment.")
            }
            fields["revision"] = revision
            modelNotice = result.cached ? "Model version inspected; no weights downloaded." : "Model version inspected. Weights still need preparation before execution."
        }
    }
    private func invalidateReview() {
        if published != nil { destination = "requests/" + UUID().uuidString.lowercased() }
        review = nil; reviewedAnswers = nil; published = nil
    }
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
            Text("Settings included in this plan").font(.headline)
            if case .object(let object) = value, case .object(let answers) = object["effectiveAnswers"] {
                ForEach(workflow.fields) { field in
                    if case .string(let answer) = answers[field.id] {
                        Text(field.label + ": " + answer).font(.caption).textSelection(.enabled)
                    }
                }
            }
            DisclosureGroup("Research notes saved with the plan") {
                Text(purpose + "\n\n" + claim + "\n\n" + controls + "\n\n" + selection).font(.caption).textSelection(.enabled)
            }
            DisclosureGroup("Exact engine request (technical details)") {
                Text(string(value, "requestJSON") ?? "Exact request unavailable; review again.")
                    .font(.system(.caption, design: .monospaced)).textSelection(.enabled)
            }
            Text("Files included in this plan").font(.headline)
            if case .object(let object) = value, case .object(let inputs) = object["inputs"], case .array(let files) = inputs["files"] {
                ForEach(Array(files.enumerated()), id: \.offset) { _, entry in
                    VStack(alignment: .leading) {
                        Text(string(entry, "path") ?? "Missing path")
                        DisclosureGroup("File fingerprint") { Text(string(entry, "sha256") ?? "Missing hash").font(.system(.caption, design: .monospaced)).foregroundStyle(.secondary) }
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
