import Foundation
import Testing

@testable import ExperimentKit

/// The app's one presentation for a refusal: the plain reason, what to do
/// (an app action for the gates the Studies views can act on), and the
/// command-line repair behind a disclosure. Never a raw error string.
@MainActor @Suite(.serialized) struct RefusalPresentationTests {

    @Test func aFrozenStudyOffersDuplicateAndKeepsTheCommandLineRepair() {
        let error = ExperimentError.refusing(
            .statusImmutable, "experiment 'kept' is frozen — duplicate it to iterate",
            repair: "steerlab-cli experiment duplicate kept kept-v2")
        let refusal = RefusalPresentation(error, context: "Couldn't add the condition.")
        #expect(refusal.reason == "Couldn't add the condition. Experiment 'kept' is frozen — duplicate it to iterate.")
        #expect(refusal.appAction == .duplicateStudy)
        #expect(refusal.appAction?.title == "Duplicate this study")
        #expect(refusal.commandLine == "steerlab-cli experiment duplicate kept kept-v2")
        #expect(refusal.code == "statusImmutable")
        #expect(refusal.whatToDo.contains("Duplicate it"))
        #expect(refusal.summary.hasPrefix(refusal.reason))
    }

    @Test func aChangedFileOffersReloadAndKeepsProseOutOfTheDisclosure() {
        let error = ExperimentError.refusing(
            .staleManifest, "The reviewed draft is unavailable in this workspace.",
            repair: "Use Discard edits and reload to review the saved study, then apply the edit again.")
        let refusal = RefusalPresentation(error)
        #expect(refusal.appAction == .reloadStudy)
        #expect(refusal.appAction?.title == "Discard edits and reload")
        // Prose advice is not a command line.
        #expect(refusal.commandLine == nil)
    }

    @Test func knownFamilyCodesMapWithoutInventingActions() {
        let changed = RefusalPresentation(StudyDesignAuthoringError(
            code: "designChanged", reason: "The design changed after it was reviewed.",
            repairAction: "Inspect the named design again."))
        #expect(changed.appAction == nil)
        #expect(changed.whatToDo.contains("template changed"))

        let inUse = RefusalPresentation(WorkspaceHousekeeping.Refusal(
            code: WorkspaceHousekeeping.agentInUseCode, path: "runs/model-variants/a/model-variant.json",
            usedBy: ["study"], reason: "This agent is used by a study: study.",
            repairAction: "steerlab-cli experiment manifest study  (see which arm uses it)"))
        #expect(inUse.appAction == nil)
        #expect(inUse.whatToDo.contains("Remove the agent"))
        #expect(inUse.commandLine?.hasPrefix("steerlab-cli experiment manifest study") == true)
    }

    @Test func anUnknownRefusalUsesItsOwnProseRepair() {
        let refusal = RefusalPresentation(
            ExperimentError.malformed("a study named 'x' already exists", repair: "choose a different name"))
        #expect(refusal.reason == "A study named 'x' already exists.")
        #expect(refusal.whatToDo == "Choose a different name.")
        #expect(refusal.appAction == nil)
        #expect(refusal.commandLine == nil)
    }

    @Test func foundationAndDecodingErrorsFallBackCleanly() {
        let missing = CocoaError(.fileReadNoSuchFile, userInfo: [NSFilePathErrorKey: "/tmp/nowhere.json"])
        let refusal = RefusalPresentation(missing, context: "Couldn't pin the file.", advice: "Choose a file that exists.")
        #expect(!refusal.reason.contains("NSCocoaErrorDomain"))
        #expect(!refusal.reason.contains("Code="))
        #expect(refusal.reason.hasPrefix("Couldn't pin the file. "))
        #expect(refusal.whatToDo == "Choose a file that exists.")

        let decoding = DecodingError.keyNotFound(
            AnyKey("modelID"), .init(codingPath: [], debugDescription: "missing"))
        let plain = RefusalPresentation.plainReason(decoding)
        #expect(plain == "The file is missing the field “modelID”.")
        #expect(!RefusalPresentation(decoding).reason.contains("DecodingError"))

        let generic = RefusalPresentation(missing)
        #expect(generic.whatToDo == RefusalPresentation.genericAdvice)
    }

    @Test func aRefusalArrivingAsACodeAndRepairIsPresented() {
        let refusal = RefusalPresentation(
            code: "custodyMismatch", repair: "steerlab-cli data verify-custody <receipt>",
            context: "The evidence import from job 7 was refused (custodyMismatch).")
        #expect(refusal.reason == "The evidence import from job 7 was refused (custodyMismatch).")
        #expect(refusal.commandLine == "steerlab-cli data verify-custody <receipt>")
        #expect(refusal.whatToDo == RefusalPresentation.genericAdvice)
    }

    // MARK: - Through the Studies panel

    @Test func theStudiesPanelSpeaksRefusalsAndPlainNotesClearThem() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "refusal-panel") { root in
            let panel = ExperimentPanel()
            panel.notices = PanelNotices(fileURL: root.appending(component: "notices.json"))
            panel.noteRefusal("Couldn't remove the control.", ExperimentError.refusing(
                .statusImmutable, "experiment 'kept' is complete and immutable",
                repair: "steerlab-cli experiment duplicate kept kept-v2"))
            let refusal = try #require(panel.statusRefusal)
            #expect(panel.status == refusal.summary)
            #expect(panel.notices.notices.last?.refusal == refusal)
            #expect(panel.notices.notices.last?.message == refusal.summary)

            panel.note("saved", severity: .success)
            #expect(panel.statusRefusal == nil)

            panel.refuse(.addCondition, "Couldn't add the condition.", ExperimentError.malformed(
                "no concept named 'x'", repair: "attach the concept first"))
            #expect(panel.draft.formErrors[.addCondition]
                == "Couldn't add the condition. No concept named 'x'. Attach the concept first.")
        }
    }

    /// End to end: the app's own Rename on a frozen study now arrives typed,
    /// so the Studies view can offer Duplicate rather than a raw error.
    @Test func renamingAFrozenStudyInTheAppOffersDuplicate() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "refusal-rename") { root in
            _ = try ExperimentStore.create(name: "kept", description: "d", modelID: "test/model")
            var manifest = try ExperimentStore.load(name: "kept")
            manifest.status = .frozen
            try ExperimentStore.save(manifest)
            let panel = ExperimentPanel()
            panel.notices = PanelNotices(fileURL: root.appending(component: "notices.json"))
            panel.management.selectedName = "kept"
            let reviewed = try panel.management.reviewStudy(named: "kept")
            #expect(!panel.management.rename(reviewed: reviewed, canonicalName: "renamed", label: nil))
            let refusal = try #require(panel.statusRefusal)
            #expect(refusal.code == "statusImmutable")
            #expect(refusal.appAction == .duplicateStudy)
            #expect(refusal.reason.hasPrefix("Couldn't rename the study; nothing changed."))
            #expect(panel.draft.formErrors[.rename] == refusal.summary)
        }
    }

    @Test func noticesWrittenBeforeTheRefusalFieldStillLoad() throws {
        let url = FileManager.default.temporaryDirectory
            .appending(component: "notices-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        let legacy = """
            {"schemaVersion": 1, "notices": [{"id": "\(UUID().uuidString)",
            "timestamp": "2026-10-01T00:00:00Z", "source": "Studies",
            "severity": "error", "message": "an older notice"}]}
            """
        try Data(legacy.utf8).write(to: url)
        let notices = PanelNotices(fileURL: url)
        #expect(notices.notices.map(\.message) == ["an older notice"])
        #expect(notices.notices.first?.refusal == nil)
    }
}

private struct AnyKey: CodingKey {
    var stringValue: String
    var intValue: Int? { nil }
    init(_ string: String) { stringValue = string }
    init?(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { nil }
}
