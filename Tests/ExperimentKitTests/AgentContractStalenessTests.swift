import Foundation
import Testing

@testable import ExperimentKit

/// The workspace-contract staleness advisory: awareness, not force.
///
/// `AGENTS.md` is written once and **never overwritten**, because it is data
/// in the researcher's own repository. The cost of that (correct) rule is that
/// a workspace made before a contract revision keeps the old text forever,
/// silently, while its runner agent reads instructions that no longer describe
/// the CLI. These tests pin the cheap fix: classify, and say so once.
///
/// The classification's load-bearing decision is the HEADER-INTACT HEURISTIC —
/// `AgentContract.generatedHeader` present byte-for-byte means "SteerLab wrote
/// this and nobody has touched the line that says so", i.e. safe to
/// regenerate; absent or altered means the researcher owns the file and every
/// surface stays silent. Four of the tests below are that line, from both
/// sides.
///
/// Serialized and holding `ExperimentRootOverrideLock` for the end-to-end
/// case: `ExperimentStore.rootOverride` is a process-global seam shared with
/// every other lifecycle suite.
@Suite(.serialized) struct AgentContractStalenessTests {

    private func tempDirectory() -> URL {
        FileManager.default.temporaryDirectory
            .appending(component: "contract-\(UUID().uuidString)")
    }

    /// A workspace-shaped folder (the `prompts/` half of `isWorkspace`) with
    /// whatever `AGENTS.md` the caller wants — or none.
    private func makeWorkspace(contract: String?) throws -> URL {
        let root = tempDirectory()
        try FileManager.default.createDirectory(
            at: root.appending(component: "prompts"), withIntermediateDirectories: true)
        if let contract {
            try contract.write(
                to: root.appending(component: AgentContract.fileName),
                atomically: true, encoding: .utf8)
        }
        return root
    }

    /// The shipped contract with a suffix of its body lopped off — the shape
    /// of the real drift: a workspace seeded from an older build, header and
    /// all, missing the sections added since.
    private func truncatedContract(keepingBodyLines keep: Int) -> String {
        let lines = AgentContract.body.split(
            separator: "\n", omittingEmptySubsequences: false)
        let kept = lines.prefix(keep).joined(separator: "\n")
        return AgentContract.generatedHeader + "\n\n" + kept + "\n"
    }

    // MARK: - The four classifications

    @Test func aFreshlyWrittenContractIsCurrent() throws {
        let root = try makeWorkspace(contract: AgentContract.contents())
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(AgentContract.status(at: root) == .current)
        #expect(AgentContract.stalenessAdvisory(at: root) == nil)
    }

    @Test func anIntactHeaderOverAnOlderBodyIsStaleUnedited() throws {
        let stale = truncatedContract(keepingBodyLines: 40)
        let root = try makeWorkspace(contract: stale)
        defer { try? FileManager.default.removeItem(at: root) }

        guard case .staleUnedited(let linesBehind) = AgentContract.status(at: root)
        else {
            Issue.record("expected staleUnedited, got \(AgentContract.status(at: root))")
            return
        }
        // Honest, not magic: the count is exactly what the multiset difference
        // reports for these two bodies, and it is positive because a truncated
        // copy is missing lines the shipped one has.
        #expect(linesBehind > 0)
        #expect(
            linesBehind
                == AgentContract.missingLineCount(
                    shipped: AgentContract.body,
                    workspace: String(stale.dropFirst(
                        (AgentContract.generatedHeader + "\n\n").count))))
    }

    /// The heuristic from the "hands off" side, #1: the header is still one
    /// line of HTML comment at the top, but it is not OUR line. Someone has
    /// been in this file with intent.
    @Test func aTamperedHeaderIsEdited() throws {
        let tampered =
            "<!-- Written by SteerLab workspace seeding. -->\n\n"
            + AgentContract.body
        let root = try makeWorkspace(contract: tampered)
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(AgentContract.status(at: root) == .edited)
        #expect(AgentContract.stalenessAdvisory(at: root) == nil)
    }

    /// The heuristic from the "hands off" side, #2 — and the more common one:
    /// a contract the researcher rewrote from their own text, with no machine
    /// header at all. Note the BODY here is the current shipped body: even a
    /// byte-current contract is `edited` without the header, because the
    /// header is the only evidence we ever had that the file is ours.
    @Test func aMissingHeaderIsEditedEvenWhenTheBodyIsCurrent() throws {
        let root = try makeWorkspace(contract: AgentContract.body)
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(AgentContract.status(at: root) == .edited)
        #expect(AgentContract.stalenessAdvisory(at: root) == nil)
    }

    @Test func noContractFileIsAbsent() throws {
        let root = try makeWorkspace(contract: nil)
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(AgentContract.status(at: root) == .absent)
        // Silent: the ensure-path regenerates an absent contract on its own,
        // so there is nothing for a person to repair.
        #expect(AgentContract.stalenessAdvisory(at: root) == nil)
    }

    // MARK: - The line count

    /// Neutral fixtures, and the two properties the advisory relies on: it
    /// counts what the workspace is MISSING, and it is a multiset difference
    /// rather than an LCS diff (so a copy that only added lines reports 0, and
    /// the advisory words that case as "out of date with").
    @Test func missingLineCountIsAMultisetDifferenceInOneDirection() {
        #expect(
            AgentContract.missingLineCount(
                shipped: "alpha\nbeta\ngamma\n", workspace: "alpha\ngamma\n") == 1)
        #expect(
            AgentContract.missingLineCount(
                shipped: "alpha\nbeta\n", workspace: "alpha\nbeta\n") == 0)
        // Additions alone are invisible to it, by construction.
        #expect(
            AgentContract.missingLineCount(
                shipped: "alpha\nbeta\n", workspace: "alpha\nbeta\ndelta\n") == 0)
        // Duplicates are counted, not collapsed.
        #expect(
            AgentContract.missingLineCount(
                shipped: "alpha\nalpha\nalpha\n", workspace: "alpha\n") == 2)
    }

    // MARK: - What the advisory says

    @Test func theAdvisoryNamesTheFileAndTheRepair() throws {
        let root = try makeWorkspace(contract: truncatedContract(keepingBodyLines: 40))
        defer { try? FileManager.default.removeItem(at: root) }
        let advisory = try #require(AgentContract.stalenessAdvisory(at: root))

        // The file it is about, by absolute path — an agent must not have to
        // guess which AGENTS.md.
        #expect(
            advisory.contains(
                root.appending(component: AgentContract.fileName).path))
        // Why we are confident it is safe to regenerate.
        #expect(advisory.contains("machine header shows it unedited"))
        // The repair, in full, and it is the one the surfaces actually honour.
        #expect(advisory.contains("delete"))
        #expect(advisory.contains("reopen the workspace"))
        #expect(advisory.contains("run any workspace verb"))
        #expect(advisory.contains("regenerate"))
        // One line. An advisory that wraps into a paragraph gets skimmed.
        #expect(!advisory.contains("\n"))
    }

    // MARK: - Regeneration still never overwrites

    /// The promise the whole design rests on, re-pinned from the staleness
    /// side: the advisory tells a PERSON to delete the file. Nothing in this
    /// feature deletes or rewrites one that is there — not a stale one, and
    /// not an edited one.
    @Test func regenerationNeverOverwritesAnExistingContract() throws {
        let stale = truncatedContract(keepingBodyLines: 40)
        let root = try makeWorkspace(contract: stale)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(component: AgentContract.fileName)

        #expect(WorkspaceStore.ensureAgentContract(at: root) == false)
        #expect(try String(contentsOf: url, encoding: .utf8) == stale)
        // …and the CLI's resolution-path helper, which regenerates in the
        // same breath as it advises, leaves it alone too.
        _ = ExperimentCLIRunner.agentContractAdvisory(root: root)
        #expect(try String(contentsOf: url, encoding: .utf8) == stale)

        let owned = "# AGENTS.md\n\nmy own notes\n"
        try owned.write(to: url, atomically: true, encoding: .utf8)
        _ = ExperimentCLIRunner.agentContractAdvisory(root: root)
        #expect(try String(contentsOf: url, encoding: .utf8) == owned)

        // Absent is the ONE case that writes — which is what makes the
        // advisory's "delete it and run any workspace verb" repair true on
        // the command line, where nothing calls `WorkspaceStore.open`.
        try FileManager.default.removeItem(at: url)
        #expect(ExperimentCLIRunner.agentContractAdvisory(root: root) == nil)
        #expect(try String(contentsOf: url, encoding: .utf8) == AgentContract.contents())
    }

    /// A folder that is not a workspace is never seeded with a contract for a
    /// workspace it is not — the guard that keeps every temp-root CLI test's
    /// stderr empty.
    @Test func aNonWorkspaceFolderIsLeftAlone() throws {
        let root = tempDirectory()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(ExperimentCLIRunner.agentContractAdvisory(root: root) == nil)
        #expect(
            !FileManager.default.fileExists(
                atPath: root.appending(component: AgentContract.fileName).path))
    }

    // MARK: - Through the real resolution path, exactly once

    /// The frequency guard, end to end. The advisory lives on
    /// `ExperimentCLIRunner.run(_:)` — which runs once per process — and not
    /// on any read path, so ONE invocation against a stale workspace produces
    /// exactly ONE stderr line, whatever the verb does afterwards.
    @Test func aStaleWorkspaceAdvisesExactlyOncePerInvocation() async throws {
        ExperimentRootOverrideLock.acquire()
        let root = FileManager.default.temporaryDirectory
            .appending(component: "contract-cli-\(UUID().uuidString)")
        ExperimentStore.rootOverride = root
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: root)
            ExperimentRootOverrideLock.release()
        }

        // A real workspace, made the real way, then aged backwards.
        try WorkspaceStore.create(at: root)
        let url = root.appending(component: AgentContract.fileName)
        try truncatedContract(keepingBodyLines: 40).write(
            to: url, atomically: true, encoding: .utf8)

        let stale = ExperimentCLIRecorder()
        let outcome = await ExperimentCLIRunner(sink: stale.sink)
            .run(namespace: "experiment", ["list"])
        // Non-blocking: the verb succeeded and said its own piece on stdout.
        #expect(outcome.exitCode == 0)
        #expect(outcome.failure == nil)
        #expect(stale.standardOutput == "no experiments\n")
        // No `advisories[]` entry — `CLIAdvisory` is a closed cross-engine
        // vocabulary and this fact is about the local checkout's shipped text.
        #expect(outcome.envelope.advisories == nil)

        let lines = stale.standardError.split(
            separator: "\n", omittingEmptySubsequences: true)
        #expect(lines.count == 1)
        #expect(lines.first?.hasPrefix("advisory: ") == true)
        #expect(lines.first?.contains(AgentContract.fileName) == true)
        #expect(lines.first?.contains("regenerate") == true)

        // The repair, performed: delete and run any verb. The contract comes
        // back, and the next invocation is silent.
        try FileManager.default.removeItem(at: url)
        let repaired = ExperimentCLIRecorder()
        _ = await ExperimentCLIRunner(sink: repaired.sink)
            .run(namespace: "experiment", ["list"])
        #expect(repaired.standardError == "")
        #expect(try String(contentsOf: url, encoding: .utf8) == AgentContract.contents())

        // And a current contract stays silent for good.
        let quiet = ExperimentCLIRecorder()
        _ = await ExperimentCLIRunner(sink: quiet.sink)
            .run(namespace: "experiment", ["list"])
        #expect(quiet.standardError == "")

        // A contract the researcher owns is silent on this path too.
        try "# AGENTS.md\n\nmy own notes\n".write(
            to: url, atomically: true, encoding: .utf8)
        let owned = ExperimentCLIRecorder()
        _ = await ExperimentCLIRunner(sink: owned.sink)
            .run(namespace: "experiment", ["list"])
        #expect(owned.standardError == "")
    }

    // MARK: - The app surface

    /// The app's twin: one notice per OPEN, into the same `"Workspace"` feed
    /// the cluster store's synchronization problems use — no new UI chrome.
    @MainActor
    @Test func openingAStaleWorkspaceRecordsOneWorkspaceNotice() throws {
        let parent = tempDirectory()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: parent) }

        let root = parent.appending(component: "ws")
        try WorkspaceStore.create(at: root)
        try truncatedContract(keepingBodyLines: 40).write(
            to: root.appending(component: AgentContract.fileName),
            atomically: true, encoding: .utf8)

        // A private feed file, so the live ring is never touched.
        let feed = PanelNotices(
            fileURL: parent.appending(component: "notices.json"))
        let store = WorkspaceStore()
        store.notices = feed

        let recorded = store.noteAgentContractStaleness(at: root)
        #expect(recorded != nil)
        #expect(feed.notices.count == 1)
        #expect(feed.notices.first?.source == "Workspace")
        #expect(feed.notices.first?.severity == .warning)
        #expect(feed.notices.first?.message.contains("regenerate") == true)
        // No `advisory: ` prefix here — the severity carries that, and the
        // sentence is shared with the CLI rather than duplicated.
        #expect(feed.notices.first?.message.hasPrefix("advisory:") == false)

        // Current contract: nothing recorded, and the feed does not grow.
        try AgentContract.contents().write(
            to: root.appending(component: AgentContract.fileName),
            atomically: true, encoding: .utf8)
        #expect(store.noteAgentContractStaleness(at: root) == nil)
        #expect(feed.notices.count == 1)

        // Edited contract: still nothing. The researcher chose their text.
        try "# AGENTS.md\n\nmy own notes\n".write(
            to: root.appending(component: AgentContract.fileName),
            atomically: true, encoding: .utf8)
        #expect(store.noteAgentContractStaleness(at: root) == nil)
        #expect(feed.notices.count == 1)
    }
}
