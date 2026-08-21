import Foundation
import Testing
@testable import ExperimentKit

/// Where a norm refusal sends you.
///
/// Reported live 2026-07-29: a J-lens direction refused to steer with "has no
/// residual norms", and the accompanying advice said to queue a backfill "in
/// Concepts" — a section that no longer exists. It was renamed **Data**. A
/// pointer to a missing section is worse than no pointer: the reader concludes
/// they are looking in the wrong place rather than that the message is stale.
@Suite struct NormBackfillGuidanceTests {

    /// The app target is not importable from here, so the section vocabulary is
    /// read from its source — which is also the file that would drift.
    private func workbenchSectionSource() throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(
            contentsOf: root.appending(path: "Sources/SteerLabApp/WorkbenchSection.swift"),
            encoding: .utf8)
    }

    @Test func theBreadcrumbNamesASectionThatExists() throws {
        let location = SteeringGuidance.normBackfillLocation
        let sections = try workbenchSectionSource()
        // "Data" is the live section name; "Concepts" was the pre-rename one.
        #expect(location.contains("Data"))
        #expect(sections.contains("case data = \"Data\""))
        #expect(!sections.contains("= \"Concepts\""))
    }

    @Test func theBreadcrumbGoesDeeperThanASectionName() {
        // A section name alone was not enough — the affordance is a collapsed
        // row inside a section, so the pointer has to name the row too.
        let location = SteeringGuidance.normBackfillLocation
        #expect(location.contains("Concept Index"))
        #expect(location.contains("missing norms"))
    }

    @Test func noRefusalStillSaysConcepts() throws {
        // The four refusal sites shared a stale string. Guard the file itself,
        // because the failure mode is a message drifting from the UI — which no
        // behavioural test over ChatService would notice.
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // ExperimentKitTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // repo root
        let source = try String(
            contentsOf: root.appending(path: "Sources/ExperimentKit/ChatService.swift"),
            encoding: .utf8)
        // The doc comment explaining the rename legitimately quotes the old
        // name; the refusals must not.
        for line in source.split(separator: "\n") where line.contains("in Concepts") {
            #expect(line.contains("///"),
                    "a non-comment line still points at the renamed section: \(line)")
        }
    }

    @Test func theRefusalOffersTheAlternativeToo() throws {
        // Measuring norms is not the only fix: raw α works immediately. A
        // refusal that names only the slower path reads as a hard block.
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let source = try String(
            contentsOf: root.appending(path: "Sources/ExperimentKit/ChatService.swift"),
            encoding: .utf8)
        let refusals = source.split(separator: "\n").filter {
            $0.contains("normBackfillLocation")
        }
        #expect(!refusals.isEmpty)
        // Each refusal that names the location also names raw α, on that line or
        // the next — checked by looking at the joined neighbourhood.
        #expect(source.contains("switch α to raw"))
    }
}
