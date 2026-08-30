import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// Golden envelope fixtures for the Swift agent-path verbs
/// (WP0-AGENT-SURFACE-AUDIT §7 step 5).
///
/// One committed document per verb under `Tests/Fixtures/cli-envelopes/`, each
/// produced by driving `ExperimentCLIRunner` in-process against a scratch
/// workspace with a PINNED clock — the `now:` seam `ClusterCLIRunner` already
/// has, whose tests pin it to `Date(timeIntervalSince1970: 1_000)`
/// (`ClusterCLITests.runner(now:)`). Without that seam every fixture would
/// churn on `observedAt`.
///
/// Two things vary between machines and are canonicalised before comparison:
/// the scratch workspace's absolute path (`<workspace>`) and the code
/// checkout's (`<checkout>`). Everything else — including every hash — is
/// byte-stable, which is the point: the fixtures are what would catch a verb
/// silently dropping `freezeHash`, or eliding it back to twelve characters.
///
/// **A missing fixture is written rather than failed.** The structural
/// assertions (`exactlyOneJSONDocumentIsProduced`, the closed key set, the
/// state/exit-code agreement) still run on the freshly produced document, so a
/// new verb cannot land without being checked; only the byte comparison waits
/// for the file to be committed. A fixture that EXISTS must match exactly.
///
/// Coverage is every agent-path verb that is drivable without a model, a
/// server, or a completed run. The model-loading verbs (`extract`, `validate`,
/// `sweep`, `run`, `evaluate`) and the network verbs (`remote *`) build their
/// envelopes through the same `ExperimentCLIResult` seam and are covered
/// structurally by `everyDeclaredVerbHasASpec` plus the runner's own tests;
/// pinning a golden for them would mean pinning a model download into the
/// suite.
///
/// Serialized, and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct ExperimentCLIEnvelopeTests {

    // MARK: Harness

    /// The clock every fixture is produced under.
    static let pinnedNow = Date(timeIntervalSince1970: 1_000)

    static var fixtureDirectory: URL {
        CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cli-envelopes")
    }

    func withTempRoot<T>(_ body: (URL) async throws -> T) async throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "expenv-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body(temp)
    }

    /// One invocation through the full parse → run path, with the clock
    /// pinned and everything the verb printed discarded (JSON mode routes it
    /// to stderr in the binary; here we only want the document).
    @discardableResult
    func invoke(_ namespace: String, _ args: [String]) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(
            sink: .discarding, now: { Self.pinnedNow }
        ).run(namespace: namespace, args)
    }

    /// Machine-varying absolute paths → placeholders, so a fixture is the
    /// same on every checkout.
    func canonicalize(_ text: String, root: URL) -> String {
        // Every SPELLING of each root is substituted — raw, standardized,
        // and symlink-resolved. The clean-clone rehearsal (2026-08-20)
        // caught the gap: under a /tmp-resident clone the runtime resolves
        // /private/tmp while the compiled literal says /tmp (or vice
        // versa), and a single-spelling replacement leaves a machine path
        // in the golden. Same failure family as the DatasetInventory
        // symlink-id bug.
        var out = text
        let workspaceSpellings = [
            root.resolvingSymlinksInPath().path,
            root.standardizedFileURL.path,
            root.path,
        ]
        let checkout = CodeResources.compiledCheckoutPath
        let checkoutSpellings = [
            checkout.resolvingSymlinksInPath().path,
            checkout.standardizedFileURL.path,
            checkout.path,
        ]
        for spelling in Set(workspaceSpellings) where !spelling.isEmpty {
            out = out.replacingOccurrences(of: spelling, with: "<workspace>")
        }
        for spelling in Set(checkoutSpellings) where !spelling.isEmpty {
            out = out.replacingOccurrences(of: spelling, with: "<checkout>")
        }
        return out
    }

    /// Assert the structural contract, then compare against (or write) the
    /// golden.
    func check(
        _ outcome: ExperimentCLIOutcome, fixture name: String, root: URL,
        file: String = #filePath, line: Int = #line
    ) throws {
        let document = try outcome.envelope.jsonText()

        // EXACTLY ONE JSON DOCUMENT. `JSONSerialization` stops at the end of
        // the first value, so a second document (or a stray printed line
        // ahead of it) shows up as leftover bytes.
        let data = Data(document.utf8)
        let parsed = try JSONSerialization.jsonObject(with: data)
        let object = try #require(parsed as? [String: Any], "\(name): not an object")
        let reserialized = try JSONSerialization.data(withJSONObject: object)
        #expect(!reserialized.isEmpty)
        #expect(document.hasSuffix("}\n"), "\(name): must end with one newline")
        #expect(
            document.components(separatedBy: "\n").filter { $0 == "}" }.count == 1,
            "\(name): more than one top-level document on stdout")

        // The CLOSED key set.
        let allowed = Set(
            SteerLabCLIEnvelope.contractHeaderKeys
                + SteerLabCLIEnvelope.contractOptionalKeys)
        for key in SteerLabCLIEnvelope.contractHeaderKeys {
            #expect(object[key] != nil, "\(name): header key '\(key)' missing")
        }
        for key in object.keys {
            #expect(allowed.contains(key), "\(name): undeclared top-level key '\(key)'")
        }

        // The document and the process can never disagree.
        #expect(outcome.envelope.exitCode == outcome.envelope.state.exitCode)
        #expect(outcome.envelope.engine == SteerLabCLIEnvelope.localEngine)
        #expect(outcome.envelope.verb == outcome.verb)
        // An error is present exactly when the state is not a success.
        #expect((outcome.envelope.error != nil) == !outcome.envelope.state.isSuccess)
        // It decodes back — an agent's side of the contract.
        let round = try SteerLabCLIEnvelope.decode(fromJSON: document)
        #expect(round.state == outcome.envelope.state)

        // The golden.
        let canonical = canonicalize(document, root: root)
        let url = Self.fixtureDirectory.appending(component: "\(name).json")
        guard let existing = try? String(contentsOf: url, encoding: .utf8) else {
            try FileManager.default.createDirectory(
                at: Self.fixtureDirectory, withIntermediateDirectories: true)
            try canonical.write(to: url, atomically: true, encoding: .utf8)
            return
        }
        #expect(
            canonical == existing,
            "\(name): envelope drifted from Tests/Fixtures/cli-envelopes/\(name).json")
    }

    // MARK: - The goldens, one per verb

    @Test func workspaceInitEnvelope() async throws {
        try await withTempRoot { root in
            let target = root.appending(component: "ws")
            let outcome = await invoke("workspace", ["init", target.path])
            #expect(outcome.envelope.state == .ready)
            #expect(outcome.envelope.changed)
            try check(outcome, fixture: "workspace-init", root: root)
        }
    }

    @Test func experimentListEmptyEnvelope() async throws {
        try await withTempRoot { root in
            let outcome = await invoke("experiment", ["list"])
            try check(outcome, fixture: "experiment-list-empty", root: root)
        }
    }

    @Test func experimentCreateEnvelope() async throws {
        try await withTempRoot { root in
            let outcome = await invoke(
                "experiment",
                [
                    "create", "demo", "--model", "mlx-community/gemma-3-4b-it-4bit",
                    "--revision", "0123456789abcdef0123456789abcdef01234567",
                ])
            #expect(outcome.envelope.changed)
            // FULL revision, not the human line's twelve characters.
            guard case .string(let revision)? = outcome.envelope.result?["modelRevision"]
            else {
                Issue.record("no modelRevision in result")
                return
            }
            #expect(revision == "0123456789abcdef0123456789abcdef01234567")
            try check(outcome, fixture: "experiment-create", root: root)
        }
    }

    @Test func experimentAttachEnvelope() async throws {
        try await withTempRoot { root in
            await invoke(
                "experiment",
                ["create", "demo", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            let outcome = await invoke("experiment", ["attach", "demo", "french"])
            // FULL stimulus hash — the human line elides it to twelve.
            let hash = try StimulusSet(
                directory: VectorCatalog.conceptsDirectory.appending(component: "french")
            ).hash
            #expect(hash.count == 64)
            let document = try outcome.envelope.jsonText()
            #expect(document.contains(hash))
            try check(outcome, fixture: "experiment-attach", root: root)
        }
    }

    @Test func experimentDetachEnvelope() async throws {
        try await withTempRoot { root in
            await invoke(
                "experiment",
                ["create", "demo", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            await invoke("experiment", ["attach", "demo", "french"])
            let outcome = await invoke("experiment", ["detach", "demo", "french"])
            #expect(outcome.envelope.state == .ready)
            try check(outcome, fixture: "experiment-detach", root: root)
        }
    }

    /// The refusal an agent meets when a declaration still names the concept —
    /// pinned as a golden because its `error.code`/`error.gate`/`repairAction`
    /// are the machine surface a caller acts on.
    @Test func experimentDetachRefusalEnvelope() async throws {
        try await withTempRoot { root in
            await invoke(
                "experiment",
                ["create", "demo", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            await invoke("experiment", ["attach", "demo", "french"])
            await invoke(
                "experiment",
                ["declare-condition", "demo", "arm", "--slots", "french:10:0.1",
                 "--alpha-units", "norm"])
            let outcome = await invoke("experiment", ["detach", "demo", "french"])
            #expect(outcome.envelope.state == .refused)
            #expect(outcome.envelope.error?.code == "conceptInUse")
            #expect(outcome.envelope.error?.gate == "conceptInUse")
            try check(outcome, fixture: "experiment-detach-refused", root: root)
        }
    }

    /// The grid, echoed. A grid is the one declaration whose written form and
    /// its run form differ (depths become blocks), so `layerFractions` AND
    /// `resolvedLayers` both belong in the document — the contract's grid
    /// dialog asks a human to read them side by side. This golden is produced
    /// with nothing extracted, which is the honest common case for a fresh
    /// draft: `layerCount` and `resolvedLayers` are explicitly null, never
    /// zero and never an empty list that reads as "no layers".
    @Test func experimentSetSweepGridEnvelope() async throws {
        try await withTempRoot { root in
            await invoke(
                "experiment",
                ["create", "demo", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            let outcome = await invoke(
                "experiment",
                ["set-sweep-grid", "demo", "--layer-fractions", "0.5,0.7,0.85",
                 "--alphas", "0.05,0.08,0.1,0.13"])
            #expect(outcome.envelope.state == .ready)
            try check(outcome, fixture: "experiment-set-sweep-grid", root: root)
        }
    }

    /// The sampling protocol, echoed — the whole row, not just the flags
    /// that moved, because the samples × temperature × tokens product is the
    /// study's cost and replication shape. This golden is the motivating
    /// stochastic replication arm (25 × 0.7 × 1024, derivedSHA256).
    @Test func experimentSetSamplingEnvelope() async throws {
        try await withTempRoot { root in
            await invoke(
                "experiment",
                ["create", "demo", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            let outcome = await invoke(
                "experiment",
                ["set-sampling", "demo", "--temperature", "0.7",
                 "--max-tokens", "1024", "--samples-per-item", "25",
                 "--seed-policy", "derivedSHA256"])
            #expect(outcome.envelope.state == .ready)
            try check(outcome, fixture: "experiment-set-sampling", root: root)
        }
    }

    /// The declared exclusion rules, echoed as stored (omit-when-nil, like
    /// the manifest's own encoding) — the echo IS the declaration.
    @Test func experimentSetExclusionsEnvelope() async throws {
        try await withTempRoot { root in
            await invoke(
                "experiment",
                ["create", "demo", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            let outcome = await invoke(
                "experiment",
                ["set-exclusions", "demo", "unparseableEndpoint,outOfRange",
                 "--min", "0", "--max", "600"])
            #expect(outcome.envelope.state == .ready)
            try check(outcome, fixture: "experiment-set-exclusions", root: root)
        }
    }

    /// The study's deployment FRAME, echoed flat: the text as stored, the
    /// DERIVED hash of it, and the delivery route this model family gives it
    /// — the fact a researcher arming a persona actually needs.
    @Test func experimentSetSystemPromptEnvelope() async throws {
        try await withTempRoot { root in
            await invoke(
                "experiment",
                ["create", "demo", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            let outcome = await invoke(
                "experiment",
                ["set-system-prompt", "demo", "You are a strict grader."])
            #expect(outcome.envelope.state == .ready)
            try check(
                outcome, fixture: "experiment-set-system-prompt", root: root)
        }
    }

    /// A one-entry registry with FIXED bytes, so the `parserRegistryHash` in
    /// the golden is stable across checkouts (the shipped seed registry is
    /// not a fixture of this suite and may legitimately gain entries).
    static let parserRegistryJSON =
        #"{"schemaVersion": 1, "parsers": {"months": {"kind": "durationMonths", "description": "Months.", "units": {"years": 12, "months": 1}}}}"#
        + "\n"

    /// The declaration AND its derived pin: the caller names a parser, the
    /// registry's current bytes become `parserRegistryHash`, and the echo
    /// carries the full hash plus the registry file — the provenance a later
    /// report.json is compared against.
    @Test func experimentSetParserEnvelope() async throws {
        try await withTempRoot { root in
            await invoke(
                "experiment",
                ["create", "demo", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            try write(
                Self.parserRegistryJSON,
                to: root.appending(path: ParserRegistry.registryFile))
            let outcome = await invoke(
                "experiment", ["set-parser", "demo", "months"])
            #expect(outcome.envelope.state == .ready)
            let hash = try #require(
                try ExperimentStore.load(name: "demo").parserRegistryHash)
            // FULL hash in the document — the human line elides it.
            #expect(try outcome.envelope.jsonText().contains(hash))
            try check(outcome, fixture: "experiment-set-parser", root: root)
        }
    }

    /// The scope pin, echoed as stored: the formats the caller chose plus
    /// the `itemCount`/`itemIDsHash` the engine computed from the study's
    /// own items — the three fields the run-start drift rule re-checks.
    @Test func experimentSetInstrumentScopeEnvelope() async throws {
        try await withTempRoot { root in
            await invoke(
                "experiment",
                ["create", "demo", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            try write(
                #"{"id": "c1", "prompt": "Affirm or reverse?", "options": ["A", "B"], "target": "A", "responseFormat": "label"}"#
                    + "\n"
                    + #"{"id": "r1", "prompt": "Affirm or reverse, with reasons.", "options": ["A", "B"], "target": "A", "responseFormat": "json"}"#
                    + "\n",
                to: root.appending(components: "prompts", "cases", "mixed.jsonl"))
            await invoke(
                "experiment", ["pin-prompts", "demo", "prompts/cases/mixed.jsonl"])
            let outcome = await invoke(
                "experiment", ["set-instrument-scope", "demo", "label"])
            #expect(outcome.envelope.state == .ready)
            try check(
                outcome, fixture: "experiment-set-instrument-scope", root: root)
        }
    }

    /// The refusal an agent meets on a ladder that doubles back — pinned as a
    /// golden because its `error.code`/`error.gate`/`repairAction` are the
    /// machine surface a caller acts on.
    @Test func experimentSetSweepGridRefusalEnvelope() async throws {
        try await withTempRoot { root in
            await invoke(
                "experiment",
                ["create", "demo", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            let outcome = await invoke(
                "experiment", ["set-sweep-grid", "demo", "--alphas", "0.1,0.05"])
            #expect(outcome.envelope.state == .refused)
            #expect(outcome.envelope.error?.code == "sweepGridRule")
            #expect(outcome.envelope.error?.gate == "sweepGridRule")
            try check(
                outcome, fixture: "experiment-set-sweep-grid-refused", root: root)
        }
    }

    // MARK: The three authoring verbs (WP0 step 5½)

    /// A workspace override, not just an experiment-root override: a rubric
    /// resolves through `VectorCatalog.projectRoot`, so pinning one under the
    /// narrower seam would read the CHECKOUT's file and pin a hash that moves
    /// whenever that committed file is edited.
    func withTempWorkspace<T>(_ body: (URL) async throws -> T) async throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "expenv-ws-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        WorkspaceRoot.programmaticOverride = temp
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            WorkspaceRoot.programmaticOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body(temp)
    }

    func write(_ text: String, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: url, atomically: true, encoding: .utf8)
    }

    /// Two items, fixed bytes — so the pinned hash in the golden is stable.
    static let taskPromptsJSONL =
        #"{"id": "c1", "prompt": "Decide the case and state the holding."}"# + "\n"
        + #"{"id": "c2", "prompt": "Decide the case and state the holding."}"# + "\n"

    @Test func experimentPinPromptsEnvelope() async throws {
        try await withTempRoot { root in
            await invoke(
                "experiment",
                ["create", "demo", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            try write(
                Self.taskPromptsJSONL,
                to: root.appending(components: "prompts", "cases", "demo-items.jsonl"))
            let outcome = await invoke(
                "experiment",
                ["pin-prompts", "demo", "prompts/cases/demo-items.jsonl"])
            #expect(outcome.exitCode == 0)
            #expect(outcome.envelope.changed)

            // The pin is the manifest's, hashed exactly as the panel's is.
            let manifest = try ExperimentStore.load(name: "demo")
            #expect(manifest.taskPromptsFile == "prompts/cases/demo-items.jsonl")
            let hash = try #require(manifest.taskPromptsHash)
            #expect(hash.count == 64)
            // FULL hash in the document — the human line elides it.
            #expect(try outcome.envelope.jsonText().contains(hash))
            try check(outcome, fixture: "experiment-pin-prompts", root: root)
        }
    }

    @Test func experimentPinRubricEnvelope() async throws {
        try await withTempWorkspace { root in
            await invoke(
                "experiment",
                ["create", "demo", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            try write(
                "Which response reasons more formally?\n",
                to: root.appending(
                    components: "prompts", "rubrics", "demo-rubric.md"))
            let outcome = await invoke(
                "experiment",
                [
                    "pin-rubric", "demo", "prompts/rubrics/demo-rubric.md",
                    "--judges", "j-1:claude,j-2:local",
                ])
            #expect(outcome.exitCode == 0)

            let manifest = try ExperimentStore.load(name: "demo")
            #expect(manifest.judgeRubricFile == "prompts/rubrics/demo-rubric.md")
            #expect(manifest.judgeRubricHash?.count == 64)
            #expect(manifest.judges?.map(\.name) == ["j-1", "j-2"])
            // The 2026-07-22 incident's fix reaches the CLI: pins + judges
            // WRITE the explicit declaration, so a frozen study cannot die
            // at the evaluate stage for want of one.
            #expect(manifest.evaluation?.kind == .pairedJudge)
            // The pin verify() would have refused is never the one written:
            // no rubric or judge violation, the moment it lands. (The study
            // has no concepts yet, which verify says separately — this
            // fixture is about the rubric.)
            let violations = ExperimentStore.verify(manifest)
            #expect(
                !violations.contains { $0.contains("rubric") || $0.contains("judge") },
                "\(violations)")
            try check(outcome, fixture: "experiment-pin-rubric", root: root)
        }
    }

    @Test func experimentDeclareConditionEnvelope() async throws {
        try await withTempRoot { root in
            await invoke(
                "experiment",
                ["create", "demo", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            await invoke("experiment", ["attach", "demo", "french"])
            let outcome = await invoke(
                "experiment",
                [
                    "declare-condition", "demo", "french-hi",
                    "--slots", "french:17:0.4", "--alpha-units", "norm",
                ])
            #expect(outcome.exitCode == 0)
            #expect(outcome.envelope.changed)

            let manifest = try ExperimentStore.load(name: "demo")
            let condition = try #require(
                manifest.conditions.first { $0.name == "french-hi" })
            #expect(condition.slots.count == 1)
            #expect(condition.slots[0].concept == "french")
            #expect(condition.slots[0].layer == 17)
            #expect(condition.slots[0].alpha == 0.4)
            // An explicit `add` is never WRITTEN — manifest bytes are the
            // content hash.
            #expect(condition.slots[0].mode == nil)
            #expect(condition.alphaInNormUnits)
            try check(outcome, fixture: "experiment-declare-condition", root: root)
        }
    }

    @Test func experimentListPopulatedEnvelope() async throws {
        try await withTempRoot { root in
            await invoke(
                "experiment",
                ["create", "demo", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            await invoke("experiment", ["attach", "demo", "french"])
            let outcome = await invoke("experiment", ["list"])
            try check(outcome, fixture: "experiment-list", root: root)
        }
    }

    @Test func experimentVerifyEnvelope() async throws {
        try await withTempRoot { root in
            await invoke(
                "experiment",
                ["create", "demo", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            await invoke("experiment", ["attach", "demo", "french"])
            let outcome = await invoke("experiment", ["verify", "demo"])
            #expect(outcome.envelope.state == .ready)
            try check(outcome, fixture: "experiment-verify", root: root)
        }
    }

    @Test func experimentVerifyViolationEnvelope() async throws {
        try await withTempRoot { root in
            await invoke(
                "experiment",
                ["create", "demo", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            await invoke("experiment", ["attach", "demo", "french"])
            var manifest = try ExperimentStore.load(name: "demo")
            manifest.concepts[0].stimulusSetHash = String(repeating: "0", count: 64)
            try ExperimentStore.save(manifest)
            let outcome = await invoke("experiment", ["verify", "demo"])
            #expect(outcome.envelope.state == .refused)
            try check(outcome, fixture: "experiment-verify-violation", root: root)
        }
    }

    @Test func experimentDuplicateEnvelope() async throws {
        try await withTempRoot { root in
            await invoke(
                "experiment",
                ["create", "demo", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            let outcome = await invoke("experiment", ["duplicate", "demo", "demo-2"])
            try check(outcome, fixture: "experiment-duplicate", root: root)
        }
    }

    @Test func experimentFreezeRefusalEnvelope() async throws {
        try await withTempRoot { root in
            await invoke(
                "experiment",
                ["create", "demo", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            await invoke("experiment", ["attach", "demo", "french"])
            let outcome = await invoke("experiment", ["freeze", "demo"])
            #expect(outcome.envelope.state == .refused)
            #expect(outcome.envelope.exitCode == 65)
            // The gate ids reach the document — the whole point of steps 2–5.
            let gates = try #require(outcome.envelope.error?.gates)
            #expect(gates.contains(FreezeGate.validateEvidence.rawValue))
            try check(outcome, fixture: "experiment-freeze-refused", root: root)
        }
    }

    @Test func dataCheckRefusalEnvelope() async throws {
        try await withTempRoot { root in
            await invoke(
                "experiment",
                ["create", "demo", "--model", "mlx-community/gemma-3-4b-it-4bit"])
            await invoke("experiment", ["attach", "demo", "french"])
            let outcome = await invoke("data", ["check", "demo"])
            // Step 7's one human-mode migration (audit §7 row 7): both
            // modes now answer 65, so the family has a single refusal code.
            #expect(outcome.exitCode == 65)
            #expect(outcome.envelope.exitCode == 65)
            try check(outcome, fixture: "data-check-refused", root: root)
        }
    }

    @Test func unknownFlagEnvelope() async throws {
        try await withTempRoot { root in
            let outcome = await invoke("experiment", ["list", "--bogus"])
            #expect(outcome.exitCode == 64)
            #expect(outcome.envelope.exitCode == 64)
            try check(outcome, fixture: "experiment-unknown-flag", root: root)
        }
    }

    @Test func unknownVerbEnvelope() async throws {
        try await withTempRoot { root in
            let outcome = await invoke("experiment", ["nonsense"])
            // Human mode keeps exit 1; the envelope calls it what it is — a
            // malformed invocation (audit appendix, "a usage error that
            // should be 64").
            #expect(outcome.exitCode == 1)
            #expect(outcome.envelope.state == .blocked)
            #expect(outcome.envelope.exitCode == 64)
            #expect(outcome.envelope.error?.code == "unknownVerb")
            try check(outcome, fixture: "experiment-unknown-verb", root: root)
        }
    }

    @Test func notFoundEnvelope() async throws {
        try await withTempRoot { root in
            let outcome = await invoke("experiment", ["verify", "nosuch"])
            #expect(outcome.envelope.state == .notFound)
            #expect(outcome.envelope.exitCode == 66)
            try check(outcome, fixture: "experiment-not-found", root: root)
        }
    }

    // MARK: - The contract itself

    /// Every fixture on disk is exactly one parseable document with the
    /// closed key set — the guard against a committed golden rotting into
    /// something no agent can read.
    @Test func everyCommittedFixtureIsOneValidEnvelope() throws {
        let files =
            (try? FileManager.default.contentsOfDirectory(
                at: Self.fixtureDirectory, includingPropertiesForKeys: nil)) ?? []
        let goldens = files.filter { $0.pathExtension == "json" }
        #expect(!goldens.isEmpty, "no committed cli-envelope fixtures")
        let allowed = Set(
            SteerLabCLIEnvelope.contractHeaderKeys
                + SteerLabCLIEnvelope.contractOptionalKeys)
        for url in goldens {
            let text = try String(contentsOf: url, encoding: .utf8)
            #expect(text.hasSuffix("}\n"), "\(url.lastPathComponent): stray trailing bytes")
            let object = try #require(
                try JSONSerialization.jsonObject(with: Data(text.utf8))
                    as? [String: Any])
            for key in object.keys {
                #expect(
                    allowed.contains(key),
                    "\(url.lastPathComponent): undeclared key '\(key)'")
            }
            let envelope = try SteerLabCLIEnvelope.decode(fromJSON: text)
            #expect(envelope.schemaVersion == SteerLabCLIEnvelope.schemaVersion)
            #expect(envelope.engine == SteerLabCLIEnvelope.localEngine)
            #expect((envelope.error != nil) == !envelope.state.isSuccess)
            // No fixture may carry a credential-shaped key, at any depth.
            let lowered = text.lowercased()
            for forbidden in ["\"token\"", "\"password\"", "\"apikey\"", "\"secret\""] {
                #expect(!lowered.contains(forbidden), "\(url.lastPathComponent): \(forbidden)")
            }
        }
    }

    // MARK: - The flag table

    @Test func everyRunnerOwnedVerbHasAFlagSpec() {
        // The table is what makes strict parsing possible; a verb missing
        // from it silently reverts to "swallow anything".
        // `label`, not `namespace + " " + verb`: a bare-verb family (`init`)
        // has no sub-verb, and its label is the family name alone.
        let declared = Set(ExperimentCLIParser.specs.map(\.label))
        let expected: Set<String> = [
            // WP1's last item: the home layout's first-run materialization.
            // The one BARE verb — `steerlab-cli init [--home <dir>]` — and a
            // new family rather than a changed one; nothing in the lifecycle
            // moved.
            "init",
            "workspace init", "data check",
            // POLE MIRRORING: the opposite pole of a contrastive direction as
            // its own artifact. A new verb in an existing family, not a
            // changed one — nothing in the lifecycle moved, and the experiment
            // count asserted below is untouched.
            "vectors compare", "vectors backfill-norms", "vectors mirror-poles",
            "remote capabilities", "remote package", "remote upload",
            "remote submit-bundle", "remote jobs", "remote logs", "remote cancel",
            // The managed resume of a checkpointed job (field incident
            // 2026-08-29: the designed resume had no managed spelling and
            // forced a hand-rolled scheduler command). A new verb in an
            // existing family; nothing in the lifecycle moved.
            "remote resubmit",
            "remote fetch", "remote import", "remote import-chain",
            "remote variants", "remote chat",
            "experiment list", "experiment create", "experiment attach",
            // `attach`'s inverse — the one authoring pin that could be added
            // and never removed, so a draft carried whatever it was first
            // attached with.
            "experiment detach",
            "experiment pin-prompts", "experiment pin-rubric",
            "experiment declare-condition",
            "experiment set-sweep-selection",
            // The other half of the sweep block: the RULE had a headless
            // writer, the GRID it selects over had none, so the only way to
            // obtain one was `duplicate` — with the donor's concepts aboard.
            "experiment set-sweep-grid",
            "experiment set-instruments",
            // The generation protocol and the exclusion rules — the six
            // manifest fields (temperature, maxTokens, promptMode,
            // samplesPerItem, seedPolicy, exclusionRules) that were writable
            // only from the Study Setup panel and the SwiftUI rules editor,
            // so a stochastic replication arm could not be authored
            // headlessly and was cut from a study design.
            "experiment set-sampling", "experiment set-exclusions",
            // The study's SYSTEM PROMPT — the deployment frame every arm is
            // read under. Writable from the Study Setup panel and nowhere
            // else, so a replication study whose donor carries a
            // judge-persona frame could not be authored headlessly at all,
            // and running it without the persona would have been a different
            // study.
            "experiment set-system-prompt",
            // The two remaining measurement declarations the app's pickers
            // owned exclusively: HOW a numeric endpoint is read
            // (`numericParser` + the `parserRegistryHash` pin) and WHICH
            // rows the option-consuming instruments read
            // (`outcomeInstrumentScope`). Without the first, a headless
            // numeric study fell back to the DEPRECATED implicit selection;
            // without the second, a mixed-format study could only follow the
            // run-start gate's LOSSY repair and drop its instrument.
            "experiment set-parser", "experiment set-instrument-scope",
            // The third declaration of that shape: the evaluate subsample's
            // DESIGN (`evaluationSampling`), whose draw rule is derived from
            // the engine and can no more be typed than a registry hash can.
            "experiment set-evaluation-sampling",
            "experiment set-style-taxonomy", "experiment verify",
            "experiment freeze", "experiment duplicate", "experiment extract",
            "experiment validate", "experiment sweep", "experiment run",
            "experiment analyze", "experiment rescore-style",
            "experiment evaluate", "experiment promote", "experiment confirm",
            // STEP 11: the generator behind this document's marked regions.
            // A new verb, not a changed one — the lifecycle set below is
            // unchanged, and this pin stays exhaustive.
            "docs cli-reference",
            // STEP 12: the binary talking about itself. Also new rather than
            // changed — nothing in the lifecycle moved — and strict-parsed
            // like everything else, so a mistyped flag on `--version` is a
            // usage error rather than a token the verb quietly ignores.
            "install version", "install stamp", "install verify",
            // OPEN-ISSUES §18: the panel family. `compile` is the new verb —
            // headless seat casting, previously reachable only from the app,
            // so the only headless path was a re-implementation of the compile
            // transform outside the engine. `list` and `check` are the
            // pre-existing read verbs, declared when the family joined the
            // agent path; their human output is unchanged.
            "panel list", "panel check", "panel compile",
            // The generation-prompt emitter: the only verb here that touches
            // no manifest and writes nothing into the workspace. It answers
            // "this study is missing X; what do I ask an LLM for", which is a
            // question about a KIND of data rather than about a study — and
            // it is on the agent path because an agent meets missing data far
            // more often than it meets a missing verb.
            "authoring prompt",
        ]
        #expect(declared == expected)
        // The audit's sixteen lifecycle verbs, the three headless authoring
        // verbs step 5½ added (P0-3), step 7's two (punch list #1 P3 +
        // P13: the sweep's selection criterion and the study's outcome
        // instruments were manifest data no headless caller could declare, so
        // the sweep silently selected on marker density and a pinned choice
        // task was measured as prose), `detach` — the owed inverse of
        // `attach`, without which a concept pin was the one authoring
        // declaration that could only ever be ADDED — and `set-sweep-grid`,
        // the sweep block's other half, without which a grid could only be
        // inherited by duplicating a study and its concepts with it — plus
        // `set-sampling` and `set-exclusions`, the writers for the six
        // protocol fields the panel owned exclusively (a stochastic
        // replication arm could not be authored headlessly) — and finally
        // `set-parser` and `set-instrument-scope`, the last two measurement
        // declarations the app's pickers owned alone (the numeric-endpoint
        // grammar with its registry pin, and the row subset the
        // option-consuming instruments read) — and `set-system-prompt`, the
        // study's deployment FRAME, the last panel-only authoring field: a
        // replication whose donor carries a judge persona could not be
        // authored headlessly at all, and running it without the persona
        // would have been a different study — and `set-evaluation-sampling`,
        // the third declaration of that derived-pin shape (review round 12): a
        // preregistered coding subsample could be TYPED but not DECLARED, so
        // the design lived in a command line rather than in the artifact
        // chain the evidence travels in.
        #expect(
            declared.filter { $0.hasPrefix("experiment ") }.count == 29,
            "the experiment lifecycle is twenty-nine verbs (audit §2.1, §8 P0-3, §9 P3/P13)")
    }

    @Test func everySpecIsInARunnerOwnedNamespace() {
        for spec in ExperimentCLIParser.specs {
            #expect(ExperimentCLIRunner.namespaces.contains(spec.namespace))
        }
    }

    @Test func jsonIsBooleanAndOutTakesThePath() throws {
        let parsed = try ExperimentCLIParser.parse(
            namespace: "experiment", ["freeze", "demo", "--json", "--force"])
        #expect(parsed.json)
        #expect(parsed.outPath == nil)
        #expect(parsed.args == ["freeze", "demo", "--force"])

        let withOut = try ExperimentCLIParser.parse(
            namespace: "experiment", ["freeze", "demo", "--out", "/tmp/e.json"])
        #expect(!withOut.json)
        #expect(withOut.outPath == "/tmp/e.json")
        #expect(withOut.args == ["freeze", "demo"])
    }

    @Test func theLegacyJSONPathSpellingStillWorksAndWarns() throws {
        // `vectors compare --json OUT` is the one surviving file form.
        let legacy = try ExperimentCLIParser.parse(
            namespace: "vectors", ["compare", "a", "b", "--json", "/tmp/r.json"])
        #expect(!legacy.json, "the path form is NOT the mode flag")
        #expect(legacy.args == ["compare", "a", "b", "--json", "/tmp/r.json"])
        #expect(legacy.deprecations.count == 1)
        #expect(legacy.deprecations[0].contains("--out /tmp/r.json"))

        // …and the bare form on the same verb is the mode.
        let modern = try ExperimentCLIParser.parse(
            namespace: "vectors", ["compare", "a", "b", "--json"])
        #expect(modern.json)
        #expect(modern.deprecations.isEmpty)
        #expect(modern.args == ["compare", "a", "b"])
    }

    @Test func aVerbThatOwnsOutKeepsIt() throws {
        // `remote fetch --out dir` is a DOWNLOAD DIRECTORY. Repurposing it as
        // the envelope's destination would send an evidence bundle somewhere
        // nobody asked for.
        let parsed = try ExperimentCLIParser.parse(
            namespace: "remote", ["fetch", "runs/x", "--out", "./downloads"])
        #expect(parsed.outPath == nil)
        #expect(parsed.args == ["fetch", "runs/x", "--out", "./downloads"])
        #expect(!parsed.declaredOutIsAvailable)
    }

    @Test func anUnknownFlagNamesTheDeclaredOnes() {
        do {
            _ = try ExperimentCLIParser.parse(
                namespace: "experiment", ["create", "demo", "--reivsion", "abc"])
        } catch let error as ExperimentCLIUsageError {
            #expect(error.flag == "--reivsion")
            #expect(error.verb == "experiment create")
            #expect(error.declaredFlags.contains("--revision"))
            #expect(error.repairAction.contains("--revision"))
        } catch {
            Issue.record("wrong error type")
        }
    }

    /// Review round 10, finding 5. A DECLARED value flag with no value used to
    /// be kept and handed on, "because the verb refuses" — and the verbs that
    /// read flags through the tolerant helper do not: `experiment set-sampling
    /// demo --temperature` WROTE the protocol at defaults and reported success.
    /// Both shapes of missing value now refuse at 64, at the one shared
    /// preprocessor, before any verb runs.
    @Test func aValueFlagWithNoValueIsAMalformedInvocation() {
        // (a) end of args, across several verbs.
        for (namespace, args, flag) in [
            ("experiment", ["set-sampling", "demo", "--temperature"], "--temperature"),
            ("experiment", ["set-exclusions", "demo", "--min"], "--min"),
            ("remote", ["submit-bundle", "b.tar", "--parallel"], "--parallel"),
        ] {
            do {
                _ = try ExperimentCLIParser.parse(namespace: namespace, args)
                Issue.record("\(flag) with no value must not parse")
            } catch let error as ExperimentCLIUsageError {
                #expect(error.flag == flag)
                #expect(error.code == "missingFlagValue")
                #expect(error.reason.contains("flag \(flag) expects a value ("))
                #expect(error.reason.hasSuffix("and none followed"))
                // The repair is the verb's own usage line, not its flag list:
                // the flag was right, its value was absent.
                #expect(error.repairAction.contains(error.verb))
                #expect(error.repairAction.contains("steerlab-cli"))
            } catch {
                Issue.record("wrong error type for \(flag)")
            }
        }

        // (b) the swallowed-flag variant: `--temperature --json` read `--json`
        // as the temperature and left the caller with neither.
        do {
            _ = try ExperimentCLIParser.parse(
                namespace: "experiment",
                ["set-sampling", "demo", "--temperature", "--json"])
            Issue.record("--temperature --json must not parse")
        } catch let error as ExperimentCLIUsageError {
            #expect(error.flag == "--temperature")
            #expect(error.followedBy == "--json")
            #expect(
                error.reason.contains(
                    "the next token --json is another experiment set-sampling "
                        + "flag, not a value"))
        } catch {
            Issue.record("wrong error type")
        }
    }

    /// The three things that must KEEP parsing, or the refusal above has
    /// broken more than it fixed.
    @Test func onlyAGenuinelyMissingValueRefuses() throws {
        // (a) An explicit EMPTY token is a value, and several verbs use it as
        // their clear affordance. It must reach them untouched.
        let cleared = try ExperimentCLIParser.parse(
            namespace: "experiment",
            ["set-sweep-selection", "demo", "--control-margin", ""])
        #expect(cleared.args == ["set-sweep-selection", "demo", "--control-margin", ""])

        // (b) A BOOLEAN flag following a valued flag's value is ordinary
        // adjacency, not a swallowed value.
        let dry = try ExperimentCLIParser.parse(
            namespace: "remote",
            ["submit-bundle", "b.tar", "--parallel", "4", "--dry-run"])
        #expect(dry.args == ["submit-bundle", "b.tar", "--parallel", "4", "--dry-run"])

        // (c) A flag-SHAPED token that is not one of this verb's flags is the
        // value the caller typed. Reinterpreting it as a flag would be this
        // parser inventing an intent.
        let negative = try ExperimentCLIParser.parse(
            namespace: "experiment",
            ["set-sampling", "demo", "--temperature", "--0.5"])
        #expect(negative.args == ["set-sampling", "demo", "--temperature", "--0.5"])
    }

    /// `--out` is a declared value flag and answers the same event with the
    /// same sentence — it kept a bespoke one until the shared mechanism
    /// existed. Its CONDITION stays stricter: the envelope's destination is a
    /// path this parser writes to, so any flag-shaped token there is refused
    /// rather than taken at its word.
    @Test func theSharedOutFlagUsesTheSameMissingValueRefusal() {
        for tail in [["freeze", "demo", "--out"],
                     ["freeze", "demo", "--out", "--json"],
                     ["freeze", "demo", "--out", "--not-a-flag"]] {
            do {
                _ = try ExperimentCLIParser.parse(
                    namespace: "experiment", tail)
                Issue.record("\(tail) must not parse")
            } catch let error as ExperimentCLIUsageError {
                #expect(error.flag == "--out")
                #expect(error.code == "missingFlagValue")
                #expect(error.reason.contains("flag --out expects a value ("))
                #expect(error.repairAction.contains("experiment freeze"))
            } catch {
                Issue.record("wrong error type for \(tail)")
            }
        }
        // A flag-shaped token that is not this verb's flag is named honestly
        // — it is not "another freeze flag".
        do {
            _ = try ExperimentCLIParser.parse(
                namespace: "experiment", ["freeze", "demo", "--out", "--nope"])
        } catch let error as ExperimentCLIUsageError {
            #expect(error.reason.contains("--nope is flag-shaped, not a value"))
        } catch {
            Issue.record("wrong error type")
        }
    }

    @Test func anUnknownSUBverbIsNotAFlagError() throws {
        // Telling someone their flag is wrong on a verb that does not exist
        // is the less useful of the two messages; the dispatch's verb list
        // wins.
        let parsed = try ExperimentCLIParser.parse(
            namespace: "experiment", ["nonsense", "--whatever"])
        // The word is carried through as typed (it labels the envelope), and
        // NOTHING was stripped — the dispatch answers with its verb list.
        #expect(parsed.verb == "nonsense")
        #expect(parsed.args == ["nonsense", "--whatever"])
        #expect(ExperimentCLIParser.spec(namespace: "experiment", verb: "nonsense") == nil)
    }

    @Test func aBareJSONFlagSurvivesAnUnparseableVerb() throws {
        // "`--json` is honoured even when parsing itself fails" (audit §2.2).
        let parsed = try ExperimentCLIParser.parse(
            namespace: "experiment", ["nonsense", "--json"])
        #expect(parsed.json)
    }

    // MARK: - The gate/gates reconciliation

    @Test func theEnvelopeNamesTheGateItsMessageDescribes() {
        // freeze's refusal order and the stamp vocabulary's order are
        // different permutations, so `gates.first` is NOT the gate whose
        // prose was thrown. Step 1's factory derived it that way; step 5
        // passes it explicitly. Here `gitClean` is the thrown gate while
        // `revision` sorts first in the vocabulary.
        let refusal = FreezeRefusal(
            gate: .gitClean, gates: [.revision, .gitClean],
            reason: "the workspace has uncommitted pinned inputs",
            repairAction: "commit them")
        let envelope = SteerLabCLIEnvelope.refusal(
            verb: "experiment freeze", engine: SteerLabCLIEnvelope.localEngine,
            code: "freezeGateFailed", gate: refusal.gateID, gates: refusal.gateIDs,
            reason: refusal.reason, repairAction: refusal.repairAction)
        #expect(envelope.error?.gate == "gitClean")
        #expect(envelope.error?.gates == ["revision", "gitClean"])
        #expect(envelope.error?.gates?.first != envelope.error?.gate)
        #expect(envelope.message == refusal.reason)
        // The invariant that survives: `gate` is always a MEMBER of `gates`.
        #expect(envelope.error?.gates?.contains(envelope.error?.gate ?? "") == true)
    }

    @Test func anOmittedGateStillFallsBackToTheListsFirst() {
        let envelope = SteerLabCLIEnvelope.refusal(
            verb: "experiment freeze", engine: SteerLabCLIEnvelope.localEngine,
            code: "freezeGateFailed", gates: ["validateEvidence"],
            reason: "no validate run", repairAction: "validate")
        #expect(envelope.error?.gate == "validateEvidence")
    }

    @Test func aNamedGateMissingFromTheListIsAddedNotDropped() {
        let envelope = SteerLabCLIEnvelope.refusal(
            verb: "experiment freeze", engine: SteerLabCLIEnvelope.localEngine,
            code: "freezeGateFailed", gate: "judgeValidity", gates: ["revision"],
            reason: "…", repairAction: "…")
        #expect(envelope.error?.gate == "judgeValidity")
        #expect(envelope.error?.gates == ["judgeValidity", "revision"])
    }
}

extension ExperimentCLIInvocation {
    /// Test convenience: whether the shared `--out` was available to this
    /// invocation's verb at all.
    var declaredOutIsAvailable: Bool {
        ExperimentCLIParser.spec(namespace: namespace, verb: verb)
            .map { !$0.ownsOutFlag } ?? true
    }
}
