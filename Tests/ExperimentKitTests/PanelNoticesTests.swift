import Foundation
import Testing
@testable import ExperimentKit

/// A15: the persistent panel-notices store (append/trim/persist/badge) and
/// the panels' `note(...)` helper — the legacy single-slot status string must
/// keep working verbatim while the feed records the same message durably.
@MainActor
@Suite struct PanelNoticesTests {

    private func tempFeedURL() -> URL {
        FileManager.default.temporaryDirectory.appending(
            components: "steerlab-notices-\(UUID().uuidString)", "notices.json")
    }

    @Test func recordAppendsInOrderAndTrimsAtCapacity() {
        let notices = PanelNotices(fileURL: tempFeedURL())
        for index in 0..<(PanelNotices.capacity + 25) {
            notices.record(source: "Studies", severity: .info, message: "m\(index)")
        }
        #expect(notices.notices.count == PanelNotices.capacity)
        // Oldest entries fell off the front; the ring stays append-ordered.
        #expect(notices.notices.first?.message == "m25")
        #expect(notices.notices.last?.message == "m\(PanelNotices.capacity + 24)")
        // The UI reads newest first.
        #expect(notices.recentFirst.first?.message == notices.notices.last?.message)
    }

    @Test func persistsAndReloadsTrimmedRing() throws {
        let url = tempFeedURL()
        let notices = PanelNotices(fileURL: url)
        notices.record(source: "Studies", severity: .error, message: "overnight failure")
        notices.record(source: "Agents", severity: .success, message: "training complete")
        #expect(FileManager.default.fileExists(atPath: url.path))

        // A fresh instance (a new app session) reads the same ring back.
        let reloaded = PanelNotices(fileURL: url)
        #expect(reloaded.notices.count == 2)
        #expect(reloaded.notices.first?.message == "overnight failure")
        #expect(reloaded.notices.first?.severity == .error)
        #expect(reloaded.notices.first?.source == "Studies")
        #expect(reloaded.notices.last?.severity == .success)
        // The badge is session state — a reload does not resurrect it.
        #expect(!reloaded.hasUnseenErrors)
    }

    @Test func errorBadgeStaysUntilViewed() {
        let notices = PanelNotices(fileURL: tempFeedURL())
        notices.record(source: "Studies", severity: .warning, message: "cancelled")
        #expect(!notices.hasUnseenErrors)
        notices.record(source: "Studies", severity: .error, message: "run failed")
        notices.record(source: "Agents", severity: .error, message: "train failed")
        #expect(notices.hasUnseenErrors)
        #expect(notices.unseenErrorCount == 2)
        // Non-error traffic never clears the badge — only viewing does.
        notices.record(source: "Studies", severity: .info, message: "still going")
        #expect(notices.hasUnseenErrors)
        notices.markViewed()
        #expect(!notices.hasUnseenErrors)
        // A new error re-badges.
        notices.record(source: "Studies", severity: .error, message: "again")
        #expect(notices.unseenErrorCount == 1)
    }

    @Test func clearEmptiesRingAndPersistedFile() throws {
        let url = tempFeedURL()
        let notices = PanelNotices(fileURL: url)
        notices.record(source: "Studies", severity: .error, message: "boom")
        notices.clear()
        #expect(notices.notices.isEmpty)
        #expect(!notices.hasUnseenErrors)
        #expect(PanelNotices(fileURL: url).notices.isEmpty)
    }

    @Test func blankMessagesAreNeverRecorded() {
        let notices = PanelNotices(fileURL: tempFeedURL())
        notices.record(source: "Studies", severity: .info, message: "   ")
        #expect(notices.notices.isEmpty)
    }

    // MARK: - The panels' note() helper (legacy status + feed, verbatim)

    @Test func experimentPanelNoteSetsLegacyStatusAndRecords() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "notices") { _ in
            let panel = ExperimentPanel()
            let feed = PanelNotices(fileURL: tempFeedURL())
            panel.notices = feed
            panel.note("study run failed: boom", severity: .error)
            // Legacy single-slot status still set, verbatim.
            #expect(panel.status == "study run failed: boom")
            // And the same message persists in the feed.
            #expect(feed.notices.last?.message == "study run failed: boom")
            #expect(feed.notices.last?.severity == .error)
            #expect(feed.notices.last?.source == "Studies")
            #expect(feed.hasUnseenErrors)
        }
    }

    @Test func fineTuningPanelSetStatusRoutesThroughFeed() {
        let panel = FineTuningPanel()
        let feed = PanelNotices(fileURL: tempFeedURL())
        panel.notices = feed
        panel.setStatus("registered adapter x")
        #expect(panel.status == "registered adapter x")
        #expect(feed.notices.last?.message == "registered adapter x")
        #expect(feed.notices.last?.source == "Agents")
        // nil clears the single-slot line without touching the ring.
        panel.setStatus(nil)
        #expect(panel.status == nil)
        #expect(feed.notices.count == 1)
    }
}
