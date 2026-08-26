import Foundation
import Testing

@testable import ExperimentKit

/// The workspace-contract upkeep path: refresh what we can PROVE is ours,
/// advise about what we cannot, and never touch the researcher's own text.
///
/// `AGENTS.md` is documentation — it alters no run — so a workspace whose copy
/// is behind the shipped contract hands its runner agent instructions that no
/// longer describe the CLI, for no benefit at all. The fix is a hash: the
/// generated header carries the SHA-256 of the body it wrote, so a file whose
/// header hash still matches its body is one nobody has edited, and rewriting
/// it risks nothing.
///
/// That proof is the load-bearing decision, and these tests are it from every
/// side:
///
/// - **hashed header, hash matches, body behind** → rewritten in place,
///   atomically, with one notice;
/// - **hashed header, hash does NOT match** → the body was edited under our
///   header: `edited`, never written, silent;
/// - **legacy hashless header** → exactly the pre-hash behaviour, advisory
///   only, bytes untouched, and the one manual regeneration it names
///   graduates the file into the hashed regime;
/// - **no header** → the researcher's file, on every surface.
///
/// Serialized and holding `ExperimentRootOverrideLock` for the end-to-end
/// cases: `ExperimentStore.rootOverride` is a process-global seam shared with
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

    /// The shipped body with a suffix lopped off — the shape of the real
    /// drift: a workspace seeded from an older build, missing the sections
    /// added since.
    private func truncatedBody(keepingLines keep: Int) -> String {
        let lines = AgentContract.body.split(
            separator: "\n", omittingEmptySubsequences: false)
        return lines.prefix(keep).joined(separator: "\n") + "\n"
    }

    /// What an OLDER BUILD wrote: that build's body under that build's own
    /// (correct) hash. The only honest way to age a fixture backwards — a
    /// mismatched hash means something else entirely.
    private func machineWritten(body: String) -> String {
        AgentContract.header(for: body) + "\n\n" + body
    }

    /// What a PRE-HASH build wrote: the legacy header, no proof attached.
    private func legacyWritten(body: String) -> String {
        AgentContract.legacyGeneratedHeader + "\n\n" + body
    }

    // MARK: - The header format

    /// Round trip: what the writer emits is what the classifier proves. The
    /// header is one HTML comment line carrying `sha256:<64 lowercase hex>`,
    /// the hex is the hash of exactly the body that follows it, and a file
    /// built that way reads back as `current`.
    @Test func theHashedHeaderRoundTrips() throws {
        let contents = AgentContract.contents()
        let headerLine = String(contents.prefix(while: { $0 != "\n" }))

        #expect(headerLine == AgentContract.generatedHeader)
        #expect(headerLine.hasPrefix("<!--"))
        #expect(headerLine.hasSuffix("-->"))
        #expect(!headerLine.contains("\n"))
        #expect(headerLine.contains("sha256:"))

        let declared = try #require(
            AgentContract.declaredBodyHash(inHeaderLine: headerLine))
        #expect(declared.count == 64)
        #expect(declared == AgentContract.sha256Hex(AgentContract.body))
        let isAllHex = declared.allSatisfy { $0.isHexDigit }
        #expect(isAllHex)
        #expect(declared.lowercased() == declared)

        #expect(AgentContract.classify(contents) == .current)

        // The hash covers the body in the SAME normalized form `classify`
        // compares, so the blank line between header and body is tolerated on
        // both sides of the proof — a copy that lost it still verifies.
        #expect(
            AgentContract.classify(AgentContract.generatedHeader + "\n" + AgentContract.body)
                == .current)
    }

    /// The legacy header is recognised and is NOT the hashed one: it declares
    /// no hash, so it can never be mistaken for proof.
    @Test func theLegacyHeaderCarriesNoProof() {
        #expect(
            AgentContract.declaredBodyHash(
                inHeaderLine: AgentContract.legacyGeneratedHeader) == nil)
        #expect(AgentContract.legacyGeneratedHeader != AgentContract.generatedHeader)
        #expect(!AgentContract.legacyGeneratedHeader.contains("sha256:"))
    }

    // MARK: - The classifications

    @Test func aFreshlyWrittenContractIsCurrent() throws {
        let root = try makeWorkspace(contract: AgentContract.contents())
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(AgentContract.status(at: root) == .current)
        #expect(AgentContract.stalenessAdvisory(at: root) == nil)
        #expect(WorkspaceStore.upkeepAgentContract(at: root) == .unchanged)
    }

    /// The state the whole feature exists for: an older build's body under
    /// that build's own hash. The hash matches, so nobody has edited it, so it
    /// is safe to rewrite — `staleProven`, not the heuristic's `staleUnedited`.
    @Test func anOlderHashedContractIsProvenStale() throws {
        let old = truncatedBody(keepingLines: 40)
        let root = try makeWorkspace(contract: machineWritten(body: old))
        defer { try? FileManager.default.removeItem(at: root) }

        guard case .staleProven(let linesBehind) = AgentContract.status(at: root)
        else {
            Issue.record("expected staleProven, got \(AgentContract.status(at: root))")
            return
        }
        // Honest, not magic: the count is exactly what the multiset difference
        // reports for these two bodies, and it is positive because a truncated
        // copy is missing lines the shipped one has.
        #expect(linesBehind > 0)
        #expect(
            linesBehind
                == AgentContract.missingLineCount(
                    shipped: AgentContract.body, workspace: old))
        // No advisory: this state does not ask a person for anything.
        #expect(AgentContract.stalenessAdvisory(at: root) == nil)
    }

    /// The proof from the "hands off" side, and the case the old heuristic got
    /// WRONG: the header is intact but the body under it has been edited. The
    /// hash no longer matches, so the file is the researcher's — silent, and
    /// never rewritten.
    @Test func aBodyEditedUnderAHashedHeaderIsEdited() throws {
        let edited = AgentContract.contents() + "\nMy own note at the end.\n"
        let root = try makeWorkspace(contract: edited)
        defer { try? FileManager.default.removeItem(at: root) }

        #expect(AgentContract.status(at: root) == .edited)
        #expect(AgentContract.stalenessAdvisory(at: root) == nil)
        #expect(WorkspaceStore.upkeepAgentContract(at: root) == .unchanged)
        #expect(
            try String(
                contentsOf: root.appending(component: AgentContract.fileName),
                encoding: .utf8) == edited)
    }

    /// The legacy regime, unchanged: a pre-hash header over an older body is
    /// `staleUnedited` — the heuristic, advisory only.
    @Test func aLegacyHeaderOverAnOlderBodyIsStaleUnedited() throws {
        let old = truncatedBody(keepingLines: 40)
        let root = try makeWorkspace(contract: legacyWritten(body: old))
        defer { try? FileManager.default.removeItem(at: root) }

        guard case .staleUnedited(let linesBehind) = AgentContract.status(at: root)
        else {
            Issue.record("expected staleUnedited, got \(AgentContract.status(at: root))")
            return
        }
        #expect(linesBehind > 0)
        #expect(
            linesBehind
                == AgentContract.missingLineCount(
                    shipped: AgentContract.body, workspace: old))
        #expect(AgentContract.stalenessAdvisory(at: root) != nil)
    }

    /// A pre-hash header over the CURRENT body: nothing is behind, so there is
    /// nothing to do and nothing to say — exactly the pre-hash answer. It
    /// stays hashless until the shipped contract moves and the person performs
    /// the one manual regeneration.
    @Test func aLegacyHeaderOverTheCurrentBodyIsCurrent() throws {
        let root = try makeWorkspace(contract: legacyWritten(body: AgentContract.body))
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(AgentContract.status(at: root) == .current)
        #expect(WorkspaceStore.upkeepAgentContract(at: root) == .unchanged)
    }

    /// The "hands off" side, #1: the header is still one line of HTML comment
    /// at the top, but it is not OUR line — neither format. Someone has been
    /// in this file with intent.
    @Test func aTamperedHeaderIsEdited() throws {
        let tampered =
            "<!-- Written by SteerLab workspace seeding. -->\n\n"
            + AgentContract.body
        let root = try makeWorkspace(contract: tampered)
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(AgentContract.status(at: root) == .edited)
        #expect(AgentContract.stalenessAdvisory(at: root) == nil)
    }

    /// The "hands off" side, #2 — a header shaped like ours with a hash that
    /// is not the body's. Whatever produced it, we cannot vouch for the text,
    /// so we do not touch it.
    @Test func aHashedHeaderWhoseHashDoesNotMatchIsEdited() throws {
        let wrong = AgentContract.header(for: "something else entirely\n")
        let root = try makeWorkspace(contract: wrong + "\n\n" + AgentContract.body)
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(AgentContract.status(at: root) == .edited)
        #expect(WorkspaceStore.upkeepAgentContract(at: root) == .unchanged)
    }

    /// The "hands off" side, #3 — and the more common one: a contract the
    /// researcher rewrote from their own text, with no machine header at all.
    /// Note the BODY here is the current shipped body: even a byte-current
    /// contract is `edited` without a header, because the header is the only
    /// evidence we ever had that the file is ours.
    @Test func aMissingHeaderIsEditedEvenWhenTheBodyIsCurrent() throws {
        let root = try makeWorkspace(contract: AgentContract.body)
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(AgentContract.status(at: root) == .edited)
        #expect(AgentContract.stalenessAdvisory(at: root) == nil)
    }

    /// A file with no newline at all cannot have a header line. Hands off.
    @Test func aSingleLineFileIsEdited() throws {
        let root = try makeWorkspace(contract: "notes")
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(AgentContract.status(at: root) == .edited)
    }

    @Test func noContractFileIsAbsent() throws {
        let root = try makeWorkspace(contract: nil)
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(AgentContract.status(at: root) == .absent)
        // Silent: the upkeep path regenerates an absent contract on its own,
        // so there is nothing for a person to repair.
        #expect(AgentContract.stalenessAdvisory(at: root) == nil)
    }

    // MARK: - The line counts

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

    /// The refresh notice counts BOTH directions, so a contract revision that
    /// only added lines still reports what changed.
    @Test func changedLineCountIsSymmetric() {
        #expect(
            AgentContract.changedLineCount(
                from: "alpha\nbeta\n", to: "alpha\nbeta\n") == 0)
        #expect(
            AgentContract.changedLineCount(
                from: "alpha\n", to: "alpha\nbeta\n") == 1)
        #expect(
            AgentContract.changedLineCount(
                from: "alpha\nbeta\n", to: "alpha\n") == 1)
        #expect(
            AgentContract.changedLineCount(
                from: "alpha\nbeta\n", to: "alpha\ngamma\n") == 2)
        // A pure reorder is invisible to a multiset difference — which is why
        // the notice words 0 rather than printing "0 lines changed".
        #expect(
            AgentContract.changedLineCount(
                from: "alpha\nbeta\n", to: "beta\nalpha\n") == 0)
    }

    // MARK: - What the two sentences say

    /// The legacy advisory: the same repair as before, plus the fact that it
    /// is the LAST time the repair is manual.
    @Test func theAdvisoryNamesTheFileAndTheRepair() throws {
        let root = try makeWorkspace(
            contract: legacyWritten(body: truncatedBody(keepingLines: 40)))
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
        // And what performing it buys: never having to perform it again.
        #expect(advisory.contains("hashed header"))
        #expect(advisory.contains("refreshes"))
        // One line. An advisory that wraps into a paragraph gets skimmed.
        #expect(!advisory.contains("\n"))
    }

    /// The refresh notice: a report of completed work, naming the file, the
    /// extent, and why the rewrite was safe. It asks for nothing.
    @Test func theRefreshNoticeReportsCompletedWork() throws {
        let root = try makeWorkspace(contract: nil)
        defer { try? FileManager.default.removeItem(at: root) }
        let notice = AgentContract.refreshNotice(linesChanged: 12, at: root)

        #expect(notice.hasPrefix("refreshed "))
        #expect(
            notice.contains(root.appending(component: AgentContract.fileName).path))
        #expect(notice.contains("(12 lines changed)"))
        #expect(notice.contains("nobody had edited it"))
        // No repair verb: there is nothing to do.
        #expect(!notice.contains("delete"))
        #expect(!notice.contains("\n"))

        // Singular, and the honest wording for a pure reorder.
        #expect(
            AgentContract.refreshNotice(linesChanged: 1, at: root)
                .contains("(1 line changed)"))
        #expect(
            AgentContract.refreshNotice(linesChanged: 0, at: root)
                .contains("(same lines, reordered)"))
    }

    // MARK: - The write, and everything it must not touch

    /// The refresh itself: a proven-stale file is rewritten to this build's
    /// contract, the result classifies `current`, and the directory is left
    /// with no staging debris — the write goes through a temp file beside it
    /// and an atomic exchange, so the path never holds a half-written
    /// contract.
    @Test func aProvenStaleContractIsRefreshedInPlace() throws {
        let old = truncatedBody(keepingLines: 40)
        let root = try makeWorkspace(contract: machineWritten(body: old))
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(component: AgentContract.fileName)

        guard case .refreshed(let linesChanged) =
            WorkspaceStore.upkeepAgentContract(at: root)
        else {
            Issue.record("expected .refreshed")
            return
        }
        #expect(linesChanged > 0)
        // Counted over the BODIES: the header line always differs (its hash
        // moved with the text) and is not a changed line of contract.
        #expect(
            linesChanged
                == AgentContract.changedLineCount(from: old, to: AgentContract.body))
        #expect(try String(contentsOf: url, encoding: .utf8) == AgentContract.contents())
        #expect(AgentContract.status(at: root) == .current)

        // Nothing staged, nothing stray — the whole directory, hidden files
        // included, is what it was plus the (rewritten) contract.
        let entries = Set(
            try FileManager.default.contentsOfDirectory(atPath: root.path))
        #expect(entries == [AgentContract.fileName, "prompts"])

        // Idempotent: a second pass has nothing to do and says nothing.
        #expect(WorkspaceStore.upkeepAgentContract(at: root) == .unchanged)
    }

    /// An absent contract is regenerated — silently, and with the hashed
    /// header, so the file it writes is one the next build can refresh
    /// without asking.
    @Test func anAbsentContractIsRegeneratedWithTheHashedHeader() throws {
        let root = try makeWorkspace(contract: nil)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(component: AgentContract.fileName)

        #expect(WorkspaceStore.upkeepAgentContract(at: root) == .regenerated)
        let written = try String(contentsOf: url, encoding: .utf8)
        #expect(written == AgentContract.contents())
        #expect(written.hasPrefix(AgentContract.headerPrefix))
        #expect(AgentContract.status(at: root) == .current)
        // Silent on every surface: nobody has anything to repair.
        #expect(ExperimentCLIRunner.agentContractUpkeepLine(root: root) == nil)
    }

    /// The promise the whole design rests on, from the other three sides: a
    /// LEGACY stale file, an EDITED file, and a body edited under our own
    /// hashed header are all left byte-for-byte alone by every write path.
    @Test func upkeepNeverOverwritesWhatItCannotProve() throws {
        let legacy = legacyWritten(body: truncatedBody(keepingLines: 40))
        let root = try makeWorkspace(contract: legacy)
        defer { try? FileManager.default.removeItem(at: root) }
        let url = root.appending(component: AgentContract.fileName)

        #expect(WorkspaceStore.ensureAgentContract(at: root) == false)
        #expect(try String(contentsOf: url, encoding: .utf8) == legacy)
        // …and the upkeep pass, which advises about exactly this file, leaves
        // it alone too. Byte-for-byte.
        guard case .legacyStale = WorkspaceStore.upkeepAgentContract(at: root) else {
            Issue.record("expected .legacyStale")
            return
        }
        #expect(try String(contentsOf: url, encoding: .utf8) == legacy)
        _ = ExperimentCLIRunner.agentContractUpkeepLine(root: root)
        #expect(try String(contentsOf: url, encoding: .utf8) == legacy)

        let owned = "# AGENTS.md\n\nmy own notes\n"
        try owned.write(to: url, atomically: true, encoding: .utf8)
        _ = ExperimentCLIRunner.agentContractUpkeepLine(root: root)
        #expect(try String(contentsOf: url, encoding: .utf8) == owned)

        let underOurHeader = AgentContract.contents() + "\nand my own note.\n"
        try underOurHeader.write(to: url, atomically: true, encoding: .utf8)
        _ = ExperimentCLIRunner.agentContractUpkeepLine(root: root)
        #expect(try String(contentsOf: url, encoding: .utf8) == underOurHeader)

        // Absent is one of the two cases that write — which is what makes the
        // advisory's "delete it and run any workspace verb" repair true on
        // the command line, where nothing calls `WorkspaceStore.open`.
        try FileManager.default.removeItem(at: url)
        #expect(ExperimentCLIRunner.agentContractUpkeepLine(root: root) == nil)
        #expect(try String(contentsOf: url, encoding: .utf8) == AgentContract.contents())
    }

    /// A folder that is not a workspace is never seeded with a contract for a
    /// workspace it is not — the guard that keeps every temp-root CLI test's
    /// stderr empty.
    @Test func aNonWorkspaceFolderIsLeftAlone() throws {
        let root = tempDirectory()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(ExperimentCLIRunner.agentContractUpkeepLine(root: root) == nil)
        #expect(
            !FileManager.default.fileExists(
                atPath: root.appending(component: AgentContract.fileName).path))
    }

    // MARK: - Through the real resolution path, exactly once

    /// The frequency guard, end to end. The upkeep line lives on
    /// `ExperimentCLIRunner.run(_:)` — which runs once per process — and not
    /// on any read path, so ONE invocation against a stale workspace produces
    /// exactly ONE stderr line, whatever the verb does afterwards. And because
    /// the same invocation performs the refresh, the NEXT one is silent.
    @Test func aProvenStaleWorkspaceRefreshesAndSaysSoExactlyOnce() async throws {
        ExperimentRootOverrideLock.acquire()
        let root = FileManager.default.temporaryDirectory
            .appending(component: "contract-cli-\(UUID().uuidString)")
        ExperimentStore.rootOverride = root
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: root)
            ExperimentRootOverrideLock.release()
        }

        // A real workspace, made the real way, then aged backwards — an older
        // build's body under that build's own hash.
        try WorkspaceStore.create(at: root)
        let url = root.appending(component: AgentContract.fileName)
        try machineWritten(body: truncatedBody(keepingLines: 40)).write(
            to: url, atomically: true, encoding: .utf8)

        let refreshing = ExperimentCLIRecorder()
        let outcome = await ExperimentCLIRunner(sink: refreshing.sink)
            .run(namespace: "experiment", ["list"])
        // Non-blocking: the verb succeeded and said its own piece on stdout.
        #expect(outcome.exitCode == 0)
        #expect(outcome.failure == nil)
        #expect(refreshing.standardOutput == "no experiments\n")
        // No `advisories[]` entry — `CLIAdvisory` is a closed cross-engine
        // vocabulary and this fact is about the local checkout's shipped text.
        #expect(outcome.envelope.advisories == nil)

        let lines = refreshing.standardError.split(
            separator: "\n", omittingEmptySubsequences: true)
        #expect(lines.count == 1)
        #expect(lines.first?.hasPrefix("notice: refreshed ") == true)
        #expect(lines.first?.contains(AgentContract.fileName) == true)
        #expect(lines.first?.contains("line") == true)

        // The work is already done: the file is current, and the next
        // invocation has nothing to say.
        #expect(try String(contentsOf: url, encoding: .utf8) == AgentContract.contents())
        let quiet = ExperimentCLIRecorder()
        _ = await ExperimentCLIRunner(sink: quiet.sink)
            .run(namespace: "experiment", ["list"])
        #expect(quiet.standardError == "")

        // A contract the researcher owns is silent on this path too, and
        // survives untouched.
        let owned = "# AGENTS.md\n\nmy own notes\n"
        try owned.write(to: url, atomically: true, encoding: .utf8)
        let hands = ExperimentCLIRecorder()
        _ = await ExperimentCLIRunner(sink: hands.sink)
            .run(namespace: "experiment", ["list"])
        #expect(hands.standardError == "")
        #expect(try String(contentsOf: url, encoding: .utf8) == owned)
    }

    /// The legacy file on the same path: ONE `advisory: ` line, the file
    /// untouched, and the same line again on the next invocation — because
    /// nothing here writes. Performing the manual repair once graduates it,
    /// and from then on the surface is silent.
    @Test func aLegacyStaleWorkspaceAdvisesAndIsNeverWritten() async throws {
        ExperimentRootOverrideLock.acquire()
        let root = FileManager.default.temporaryDirectory
            .appending(component: "contract-cli-\(UUID().uuidString)")
        ExperimentStore.rootOverride = root
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: root)
            ExperimentRootOverrideLock.release()
        }

        try WorkspaceStore.create(at: root)
        let url = root.appending(component: AgentContract.fileName)
        let legacy = legacyWritten(body: truncatedBody(keepingLines: 40))
        try legacy.write(to: url, atomically: true, encoding: .utf8)

        let advising = ExperimentCLIRecorder()
        let outcome = await ExperimentCLIRunner(sink: advising.sink)
            .run(namespace: "experiment", ["list"])
        #expect(outcome.exitCode == 0)
        let lines = advising.standardError.split(
            separator: "\n", omittingEmptySubsequences: true)
        #expect(lines.count == 1)
        #expect(lines.first?.hasPrefix("advisory: ") == true)
        #expect(lines.first?.contains("regenerate") == true)
        // Byte-for-byte: this surface does not write pre-hash files, ever.
        #expect(try String(contentsOf: url, encoding: .utf8) == legacy)

        // Still advising, still not writing.
        let again = ExperimentCLIRecorder()
        _ = await ExperimentCLIRunner(sink: again.sink)
            .run(namespace: "experiment", ["list"])
        #expect(again.standardError.split(separator: "\n").count == 1)
        #expect(try String(contentsOf: url, encoding: .utf8) == legacy)

        // The repair, performed by hand exactly once: delete and run any verb.
        // What comes back carries a hashed header, and the surface goes quiet
        // for good.
        try FileManager.default.removeItem(at: url)
        let repaired = ExperimentCLIRecorder()
        _ = await ExperimentCLIRunner(sink: repaired.sink)
            .run(namespace: "experiment", ["list"])
        #expect(repaired.standardError == "")
        #expect(try String(contentsOf: url, encoding: .utf8) == AgentContract.contents())
        #expect(
            AgentContract.declaredBodyHash(
                inHeaderLine: AgentContract.generatedHeader) != nil)
    }

    // MARK: - The app surface

    /// The app's twin: one line per OPEN, into the same `"Workspace"` feed the
    /// cluster store's synchronization problems use — no new UI chrome. A
    /// refresh is `.info` (work already done); a legacy file is `.warning` (a
    /// gesture is wanted).
    @MainActor
    @Test func openingAStaleWorkspaceRecordsOneWorkspaceNotice() throws {
        let parent = tempDirectory()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: parent) }

        let root = parent.appending(component: "ws")
        try WorkspaceStore.create(at: root)
        let url = root.appending(component: AgentContract.fileName)
        try machineWritten(body: truncatedBody(keepingLines: 40)).write(
            to: url, atomically: true, encoding: .utf8)

        // A private feed file, so the live ring is never touched.
        let feed = PanelNotices(
            fileURL: parent.appending(component: "notices.json"))
        let store = WorkspaceStore()
        store.notices = feed

        let recorded = try #require(store.noteAgentContractUpkeep(at: root))
        #expect(recorded.hasPrefix("refreshed "))
        #expect(feed.notices.count == 1)
        #expect(feed.notices.first?.source == "Workspace")
        #expect(feed.notices.first?.severity == .info)
        // No `advisory: ` / `notice: ` prefix here — the severity carries
        // that, and the sentence is shared with the CLI rather than
        // duplicated.
        #expect(feed.notices.first?.message.hasPrefix("notice:") == false)
        #expect(try String(contentsOf: url, encoding: .utf8) == AgentContract.contents())

        // Once per open: the refreshed contract is current, so nothing more is
        // recorded however often this is called.
        #expect(store.noteAgentContractUpkeep(at: root) == nil)
        #expect(feed.notices.count == 1)

        // Edited contract: still nothing, and it is not rewritten. The
        // researcher chose their text.
        let owned = "# AGENTS.md\n\nmy own notes\n"
        try owned.write(to: url, atomically: true, encoding: .utf8)
        #expect(store.noteAgentContractUpkeep(at: root) == nil)
        #expect(feed.notices.count == 1)
        #expect(try String(contentsOf: url, encoding: .utf8) == owned)

        // A pre-hash file speaks, at warning, and keeps its bytes.
        let legacy = legacyWritten(body: truncatedBody(keepingLines: 40))
        try legacy.write(to: url, atomically: true, encoding: .utf8)
        let advisory = try #require(store.noteAgentContractUpkeep(at: root))
        #expect(advisory.contains("delete"))
        #expect(feed.notices.count == 2)
        #expect(feed.notices.last?.severity == .warning)
        #expect(try String(contentsOf: url, encoding: .utf8) == legacy)
    }

    /// The seam `switchTo` rests on: `open` performs the upkeep and HANDS THE
    /// OUTCOME UP, so the notice path does not have to re-classify a file that
    /// same call has already rewritten (and would then find `current`, saying
    /// nothing about a refresh that did happen).
    @Test func openPerformsTheUpkeepAndReturnsWhatItDid() throws {
        let parent = tempDirectory()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: parent) }

        let root = parent.appending(component: "ws")
        try WorkspaceStore.create(at: root)
        let url = root.appending(component: AgentContract.fileName)
        try machineWritten(body: truncatedBody(keepingLines: 40)).write(
            to: url, atomically: true, encoding: .utf8)

        let suiteName = "workspace-contract-\(UUID().uuidString)"
        let defaults = try #require(UserDefaults(suiteName: suiteName))
        defer { defaults.removePersistentDomain(forName: suiteName) }

        let upkeep = try WorkspaceStore.open(
            at: root, defaults: defaults, setOverride: { _ in })
        guard case .refreshed(let linesChanged) = upkeep else {
            Issue.record("expected .refreshed, got \(upkeep)")
            return
        }
        #expect(linesChanged > 0)
        #expect(upkeep.sentence(at: root)?.hasPrefix("refreshed ") == true)
        #expect(upkeep.severity == .info)
        #expect(try String(contentsOf: url, encoding: .utf8) == AgentContract.contents())

        // Second open: nothing left to do, and nothing to say.
        let again = try WorkspaceStore.open(
            at: root, defaults: defaults, setOverride: { _ in })
        #expect(again == .unchanged)
        #expect(again.sentence(at: root) == nil)
    }
}
