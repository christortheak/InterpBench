import Foundation
import Testing

@testable import ExperimentKit

/// Viewer-state hygiene for the workbench's right-hand display pane
/// ("phase 0" of the Data-workbench program).
///
/// Reported live: "display panel data from one tab persisting into the
/// display panel of another tab — it feels half-baked". Three distinct
/// mechanisms wore that appearance; these tests pin the rules that separate
/// them:
///
/// 1. The pane follows the SELECTED SECTION. A pin is the only thing that
///    survives a switch, and a pinned viewer showing one section's own data
///    under a different section must say whose data it is.
/// 2. The Activity feed is ONE workspace-wide log. Persisting across
///    sections is correct — so it is never labelled with a false origin, and
///    never cleared.
/// 3. Live activity logs and chat turns share `ChatService.transcript` but
///    belong to different panes. Rendering the raw array in the Playground
///    put another section's vector build into the chat.
@Suite struct WorkbenchViewerHygieneTests {

    // The app target is an executable and cannot be imported, so the pin
    // model is generic over the view's mode enum; this stands in for
    // `ActivityViewerMode`.
    private enum Mode: Equatable, Sendable {
        case chat
        case resultsRun
        case analysisGeometry
        case activity
    }

    private static let resultsViewer = WorkbenchViewerIdentity(
        title: "Selected Run", sectionLabel: "Results", ownership: .section)
    private static let analysisViewer = WorkbenchViewerIdentity(
        title: "Vector Geometry", sectionLabel: "Analysis", ownership: .section)
    private static let computeActivity = WorkbenchViewerIdentity(
        title: "Activity", sectionLabel: "Compute", ownership: .workspaceWide)
    private static let homeActivity = WorkbenchViewerIdentity(
        title: "Recent Activity", sectionLabel: "Home", ownership: .workspaceWide)

    // MARK: - The reset rule

    @Test func anUnpinnedViewerFollowsTheSelectedSection() {
        let pin = WorkbenchViewerPin<Mode>()
        // Select Results, then Compute: the pane holds no mode of its own,
        // so each answer is that section's own viewer.
        #expect(pin.resolvedMode(section: .resultsRun) == .resultsRun)
        #expect(pin.resolvedMode(section: .activity) == .activity)
        #expect(pin.resolvedIdentity(section: Self.computeActivity)
            == Self.computeActivity)
        #expect(pin.badge(section: Self.computeActivity) == nil)
        #expect(!pin.showsForeignSectionData(section: Self.computeActivity))
    }

    @Test func onlyAPinSurvivesASectionSwitch() {
        var pin = WorkbenchViewerPin<Mode>()
        pin.toggle(mode: .resultsRun, identity: Self.resultsViewer)
        #expect(pin.isPinned)
        // Now standing in Compute, whose own viewer is the Activity feed.
        #expect(pin.resolvedMode(section: .activity) == .resultsRun)
        #expect(pin.resolvedIdentity(section: Self.computeActivity)
            == Self.resultsViewer)
    }

    @Test func unpinningReturnsThePaneToTheSelectedSection() {
        var pin = WorkbenchViewerPin<Mode>()
        pin.toggle(mode: .resultsRun, identity: Self.resultsViewer)
        pin.toggle(mode: .resultsRun, identity: Self.resultsViewer)  // toggle off
        #expect(!pin.isPinned)
        #expect(pin.resolvedMode(section: .activity) == .activity)

        pin.toggle(mode: .analysisGeometry, identity: Self.analysisViewer)
        pin.unpin()
        #expect(!pin.isPinned)
        #expect(pin.resolvedMode(section: .chat) == .chat)
    }

    // MARK: - The label rule

    @Test func aForeignSectionsDataIsAlwaysLabelledWithItsOrigin() {
        var pin = WorkbenchViewerPin<Mode>()
        pin.toggle(mode: .resultsRun, identity: Self.resultsViewer)
        // Standing in Compute, looking at Results' run.
        #expect(pin.showsForeignSectionData(section: Self.computeActivity))
        #expect(pin.badge(section: Self.computeActivity) == "pinned · Results")
        // Back in Results, the same pin is not foreign — no origin needed.
        #expect(!pin.showsForeignSectionData(section: Self.resultsViewer))
        #expect(pin.badge(section: Self.resultsViewer) == "pinned")
    }

    /// The invariant the whole pass exists to guarantee: section-specific
    /// content never renders under another section without a label.
    @Test func sectionSpecificContentNeverRendersUnlabelled() {
        let sections = [Self.resultsViewer, Self.analysisViewer,
                        Self.computeActivity, Self.homeActivity]
        for origin in sections {
            var pin = WorkbenchViewerPin<Mode>()
            pin.toggle(mode: .resultsRun, identity: origin)
            for current in sections {
                let badge = pin.badge(section: current)
                #expect(badge != nil, "a pinned viewer is always badged")
                if pin.showsForeignSectionData(section: current) {
                    #expect(
                        badge?.contains(origin.sectionLabel) == true,
                        "\(origin.sectionLabel)'s own data under \(current.sectionLabel) must name its origin")
                }
            }
        }
    }

    @Test func theWorkspaceWideFeedIsNeverGivenAFalseOrigin() {
        var pin = WorkbenchViewerPin<Mode>()
        // Pinned from Home, viewed from Compute: the SAME log, so claiming
        // it "came from Home" would assert something untrue.
        pin.toggle(mode: .activity, identity: Self.homeActivity)
        #expect(!pin.showsForeignSectionData(section: Self.computeActivity))
        #expect(pin.badge(section: Self.computeActivity) == "pinned")
        #expect(
            pin.scopeDescription(section: Self.computeActivity)
                .contains("workspace-wide"))
    }

    @Test func theScopeDescriptionNamesWhoseContentIsOnScreen() {
        var pin = WorkbenchViewerPin<Mode>()
        #expect(
            pin.scopeDescription(section: Self.resultsViewer).contains("Results"))
        #expect(
            pin.scopeDescription(section: Self.computeActivity)
                .contains("workspace-wide"))
        pin.toggle(mode: .resultsRun, identity: Self.resultsViewer)
        let pinnedDescription = pin.scopeDescription(section: Self.analysisViewer)
        #expect(pinnedDescription.contains("Results"))
        #expect(pinnedDescription.contains("unpin"))
    }

    @Test func thePinControlSaysWhatItWillDo() {
        var pin = WorkbenchViewerPin<Mode>()
        #expect(pin.pinControlHelp(section: Self.resultsViewer).hasPrefix("pin "))
        pin.toggle(mode: .resultsRun, identity: Self.resultsViewer)
        // Foreign: the help names the section whose data is on screen.
        let foreign = pin.pinControlHelp(section: Self.computeActivity)
        #expect(foreign.hasPrefix("unpin"))
        #expect(foreign.contains("Results"))
        #expect(foreign.contains("Compute"))
        // Same section: plain unpin.
        #expect(pin.pinControlHelp(section: Self.resultsViewer).hasPrefix("unpin"))
    }
}

/// The transcript half of the same hygiene pass: which pane renders which
/// entries, and what a workspace switch retires.
@MainActor
struct WorkbenchViewerTranscriptTests {

    private func freshService(_ name: String) throws -> ChatService {
        let suite = "steerlab.tests.viewer-hygiene.\(name)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        return ChatService(cluster: ClusterConnectionStore(defaults: defaults))
    }

    /// The reported leak: a build started in Data streamed into the shared
    /// transcript, and the Playground rendered the whole array.
    @Test func theChatPaneShowsConversationTurnsOnly() throws {
        let service = try freshService("chat-pane")
        // Staged directly (no model loaded in a unit test), same as the
        // seeded-turn suite.
        service.transcript.append(.init(role: .user, text: "what is the holding?"))
        _ = service.startLiveLog(
            title: "Build vector · courage", initialLine: "reading stimuli…")
        service.transcript.append(.init(role: .assistant, text: "The court held…"))

        #expect(service.transcript.count == 3)
        let conversation = service.conversationTranscript
        #expect(conversation.count == 2)
        #expect(conversation.allSatisfy { !service.isLiveActivityLog($0) })
        #expect(!conversation.contains { $0.text.contains("Build vector") })
    }

    @Test func theActivityFeedShowsLogsOnly() throws {
        let service = try freshService("activity-pane")
        service.transcript.append(.init(role: .user, text: "hello"))
        _ = service.startLiveLog(title: "Study run", initialLine: "condition 1/4")

        let logs = service.activityLogTranscript
        #expect(logs.count == 1)
        #expect(logs.allSatisfy { service.isLiveActivityLog($0) })
        #expect(logs[0].text.contains("Study run"))
        // The two panes partition the transcript — nothing is shown twice,
        // nothing is dropped.
        #expect(
            service.conversationTranscript.count + logs.count
                == service.transcript.count)
    }

    /// A live activity log is not a model turn: exporting one under an
    /// "Assistant" heading would put a vector build's progress lines into the
    /// record as generated output.
    @Test func theTranscriptExportOmitsActivityLogs() throws {
        let service = try freshService("export")
        _ = service.startLiveLog(
            title: "Build vector · courage", initialLine: "reading stimuli…")
        service.transcript.append(.init(role: .user, text: "hello"))

        let markdown = service.transcriptMarkdown()
        #expect(markdown.contains("hello"))
        #expect(!markdown.contains("Build vector"))
    }

    /// After Open Workspace… the Results viewer used to keep showing the
    /// PREVIOUS workspace's run — with a "reveal in Finder" button pointing
    /// outside the workspace. Each viewer already had an empty state; the
    /// selection just needed retiring.
    @Test func switchingWorkspacesRetiresSectionViewerSelections() throws {
        let service = try freshService("workspace-switch")
        let runURL = URL(fileURLWithPath: "/tmp/old-workspace/runs/2026-01-01-run")
        service.selectedResultsRun = RunBrowser.Item(
            url: runURL, name: "2026-01-01-run", runType: "study")
        service.experiments.selectedResultsFile = RunBrowser.FileEntry(
            url: runURL.appending(path: "report.json"), name: "report.json",
            size: 10, isDirectory: false)

        service.resetSectionViewers()

        #expect(service.selectedResultsRun == nil)
        #expect(service.experiments.selectedResultsFile == nil)
        #expect(service.experiments.selectedRemoteResultsRun == nil)
        #expect(service.geometry.result == nil)
        #expect(service.geometry.activeMatrix == nil)
        #expect(service.geometry.serverMatrix == nil)
        #expect(service.multiAgent.liveTurnResults.isEmpty)
        #expect(service.fineTuning.robustnessReport == nil)
    }

    /// The workspace-wide Activity log is a LOG: jobs started before the
    /// switch keep writing to it, so it is deliberately not cleared.
    @Test func switchingWorkspacesKeepsTheActivityLog() throws {
        let service = try freshService("workspace-switch-log")
        _ = service.startLiveLog(title: "Study run", initialLine: "condition 1/4")
        service.resetSectionViewers()
        #expect(service.activityLogTranscript.count == 1)
    }
}
