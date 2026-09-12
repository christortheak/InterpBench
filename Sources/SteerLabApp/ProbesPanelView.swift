import AppKit
import ExperimentKit
import SwiftUI

/// Captured workspace identity resets the view, while selection tasks discard
/// late responses. Scanning/reading runs in the shared Python worker off-main.
struct ProbesPanelView: View {
    let root: URL
    let openLegacyTraining: () -> Void
    @State private var inventory: ProbeLibrary.Inventory?
    @State private var selection: String?
    @State private var inspected: ProbeLibrary.Record?
    @State private var parameterPreview = ""
    @State private var error: String?
    @State private var loading = false
    @State private var inspecting = false
    @State private var refreshID = UUID()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Probes").font(.title2)
                Spacer()
                Button("Refresh") { refreshID = UUID() }.disabled(loading)
            }
            Text("A probe reads a model’s activations and returns a score. It measures the model; it does not change its behavior.")
            Text("Library in \(root.path)").font(.caption).textSelection(.enabled)
            Button("Train with the existing reader builder…", action: openLegacyTraining)
                .help("Opens the current labeled-reader training controls in Data → Concepts & Vectors.")
            if loading { ProgressView("Reading the probe library…") }
            if inspecting { ProgressView("Reading the selected probe…") }
            if let error { Text(error).foregroundStyle(.orange).textSelection(.enabled) }
            if let inventory {
                if inventory.probes.isEmpty {
                    Text("No saved probes yet. Existing native and Python reading probes appear here after their run evidence is in this workspace.")
                } else {
                    List(inventory.probes, selection: $selection) { probe in
                        VStack(alignment: .leading, spacing: 3) {
                            Text(probe.label)
                            Text("\(probe.modelID) · layer \(probe.layer) · \(probe.methodLabel)").font(.caption)
                        }.tag(probe.path)
                    }.frame(minHeight: 140, maxHeight: 260)
                }
                ForEach(inventory.issues.indices, id: \.self) { index in
                    Text("Could not inspect \(inventory.issues[index].path): \(inventory.issues[index].reason)")
                        .font(.caption).foregroundStyle(.orange).textSelection(.enabled)
                }
            }
            if let inspected {
                ScrollView {
                    VStack(alignment: .leading, spacing: 10) {
                        Text(inspected.label).font(.headline)
                        Text("\(inspected.methodLabel) · \(inspected.formatLabel)").font(.caption)
                        Text("Model: \(inspected.modelID)\nLayer: \(inspected.layer)").textSelection(.enabled)
                        if let date = inspected.createdAt { Text("Created: \(date)") }
                        else { Text("Creation time was not recorded in this artifact.").font(.caption) }
                        ForEach(inspected.limitations, id: \.self) { Text($0).font(.callout) }
                        Text("Artifact: \(inspected.path)\nSHA-256: \(inspected.sha256)").font(.caption).textSelection(.enabled)
                        Button("Reveal artifact") { NSWorkspace.shared.activateFileViewerSelecting([root.appending(path: inspected.path)]) }
                        DisclosureGroup("Stored parameters and provenance") {
                            Text(parameterPreview).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                        }
                    }.frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            Spacer(minLength: 0)
        }.padding()
        .task(id: refreshID) {
            loading = true; error = nil; inspected = nil; selection = nil
            defer { if !Task.isCancelled { loading = false } }
            do {
                let result = try await ProbeLibrary.inventory(root: root)
                guard !Task.isCancelled else { return }
                inventory = result
            } catch { if !Task.isCancelled { self.error = error.localizedDescription; inventory = nil } }
        }
        .task(id: selection) {
            inspected = nil; inspecting = false
            guard let selection else { return }
            inspecting = true
            defer { if !Task.isCancelled { inspecting = false } }
            do {
                let result = try await ProbeLibrary.inspect(path: selection, root: root)
                let preview = await Task.detached(priority: .utility) { Self.formatted(result.document) }.value
                guard !Task.isCancelled, self.selection == selection else { return }
                inspected = result; parameterPreview = preview; error = nil
            } catch { if !Task.isCancelled { self.error = error.localizedDescription } }
        }
    }

    nonisolated private static func formatted(_ document: JSONValue?) -> String {
        guard let document else { return "No parameters available." }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let bytes = try? encoder.encode(document) else { return "Could not display parameters." }
        let preview = String(decoding: bytes.prefix(65_536), as: UTF8.self)
        return bytes.count <= 65_536 ? preview : preview + "\n…\nParameter preview shortened. Reveal the artifact to read all stored parameters."

    }
}
