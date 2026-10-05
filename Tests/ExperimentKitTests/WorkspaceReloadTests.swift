import Foundation
import Testing
@testable import ExperimentKit

/// The app notices what a coding assistant did: a study written on disk
/// after load appears on reload (activation or Refresh) without discarding an
/// unsaved edit, and a manifest the app cannot read is listed with a reason
/// instead of vanishing.
@MainActor @Suite(.serialized)
struct WorkspaceReloadTests {
    private func withWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let root = FileManager.default.temporaryDirectory.appending(
            component: "workspace-reload-\(UUID())")
        let previous = WorkspaceRoot.programmaticOverride
        WorkspaceRoot.programmaticOverride = root
        defer {
            WorkspaceRoot.programmaticOverride = previous
            try? FileManager.default.removeItem(at: root)
            ExperimentRootOverrideLock.release()
        }
        return try body(root)
    }

    private func write(_ text: String, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(text.utf8).write(to: url)
    }

    @Test func aStudyWrittenOnDiskAfterLoadAppearsOnReloadAndUnsavedEditsSurvive() throws {
        try withWorkspace { root in
            let suite = "steerlab.tests.workspace-reload.\(UUID().uuidString)"
            let defaults = try #require(UserDefaults(suiteName: suite))
            defer { defaults.removePersistentDomain(forName: suite) }
            let service = ChatService(cluster: clusterStore(defaults: defaults))
            let management = service.experiments.management
            _ = try ExperimentStore.create(name: "existing", description: "saved words", modelID: "test/model")
            service.reloadWorkspaceListsFromDisk()
            #expect(management.experiments.map(\.name) == ["existing"])
            management.selectedName = "existing"
            #expect(service.experiments.draft.protocolDescription == "saved words")
            #expect(!management.selectedDraftNeedsReload)

            // The researcher is mid-edit in the app…
            service.experiments.draft.protocolDescription = "unsaved words"
            // …while a coding assistant, through a command line, creates a
            // study and changes the one on screen.
            _ = try ExperimentStore.create(name: "from-assistant", description: "", modelID: "test/model")
            var changed = try ExperimentStore.load(name: "existing")
            changed.experimentDescription = "changed on disk"
            try ExperimentStore.save(changed)

            service.reloadWorkspaceListsFromDisk()
            #expect(Set(management.experiments.map(\.name)) == ["existing", "from-assistant"])
            // Nothing typed is discarded; the editor says the saved study
            // changed and offers "Discard edits and reload" instead.
            #expect(service.experiments.draft.protocolDescription == "unsaved words")
            #expect(management.selectedName == "existing")
            #expect(management.selectedDraftNeedsReload)
            #expect(management.selected?.experimentDescription == "changed on disk")
            _ = root
        }
    }

    @Test func anUnreadableStudyIsListedWithItsReasonAndNeverWritten() throws {
        try withWorkspace { root in
            _ = try ExperimentStore.create(name: "readable", description: "", modelID: "test/model")
            let experiments = root.appending(component: "experiments")
            let broken = experiments.appending(components: "broken", "experiment.json")
            let brokenText = #"{"name": "broken", "description": "half-written"#
            try write(brokenText, to: broken)
            let incomplete = experiments.appending(components: "incomplete", "experiment.json")
            try write(#"{"name": "incomplete"}"#, to: incomplete)
            var misnamed = try ExperimentStore.load(name: "readable")
            misnamed.name = "someone-else"
            let encoder = JSONEncoder()
            let misnamedURL = experiments.appending(components: "misnamed", "experiment.json")
            try FileManager.default.createDirectory(
                at: misnamedURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try encoder.encode(misnamed).write(to: misnamedURL)
            // Not studies at all: a draft moved to trash, an empty folder, a
            // stray file. None is listed, readable or not.
            try write("{", to: experiments.appending(components: ".trash-2026", "x", "experiment.json"))
            try FileManager.default.createDirectory(
                at: experiments.appending(component: "empty-folder"), withIntermediateDirectories: true)
            try write("notes", to: experiments.appending(component: "notes.txt"))

            let owner = StudyManagementController(draft: StudyDraftState())
            owner.refresh()
            #expect(owner.experiments.map(\.name) == ["readable"])
            let unreadable = owner.unreadableStudies
            #expect(unreadable.map(\.name) == ["broken", "incomplete", "misnamed"])
            let byName = Dictionary(uniqueKeysWithValues: unreadable.map { ($0.name, $0) })
            #expect(byName["broken"]?.reason.contains("not valid JSON") == true)
            #expect(byName["incomplete"]?.reason.contains("is missing") == true)
            #expect(byName["misnamed"]?.reason.contains("different study than its folder") == true)
            #expect(byName["broken"]?.url.standardizedFileURL == broken.standardizedFileURL)
            #expect(byName["broken"]?.relativePath(workspaceRoot: root) == "experiments/broken/experiment.json")
            // Read, never written.
            #expect(try String(contentsOf: broken, encoding: .utf8) == brokenText)

            // Repaired on disk, it simply joins the list on the next reload.
            try write(String(decoding: try encoder.encode(try ExperimentStore.load(name: "readable")), as: UTF8.self)
                .replacingOccurrences(of: "\"readable\"", with: "\"broken\""), to: broken)
            owner.refresh()
            #expect(Set(owner.experiments.map(\.name)) == ["readable", "broken"])
            #expect(owner.unreadableStudies.map(\.name) == ["incomplete", "misnamed"])
        }
    }

    @Test func anUnreadableTemplateIsListedWithItsReason() throws {
        try withWorkspace { root in
            let file = root.appending(components: "templates", "broken-design", "template.json")
            try write("[1, 2", to: file)
            let owner = StudyManagementController(draft: StudyDraftState())
            owner.refresh()
            #expect(owner.designs.templates.isEmpty)
            let listed = try #require(owner.designs.unreadableTemplates.first)
            #expect(listed.name == "broken-design")
            #expect(listed.reason.contains("not valid JSON"))
            #expect(try String(contentsOf: file, encoding: .utf8) == "[1, 2")
        }
    }

    @Test func readFailuresAreExplainedInPlainWords() {
        struct Shape: Decodable { let count: Int; let items: [String] }
        func reason(_ json: String) -> String {
            do {
                _ = try JSONDecoder().decode(Shape.self, from: Data(json.utf8))
                return "decoded"
            } catch {
                return UnreadableManifest.reason(for: error, folder: "study")
            }
        }
        #expect(reason(#"{"count": 1}"#) == "A required setting, “items”, is missing.")
        #expect(reason(#"{"count": "one", "items": []}"#)
            == "The setting “count” holds the wrong kind of value (expected a number).")
        #expect(reason(#"{"count": null, "items": []}"#).contains("is empty where a number is required"))
        #expect(reason("{").hasPrefix("The file is not valid JSON"))
    }
}
