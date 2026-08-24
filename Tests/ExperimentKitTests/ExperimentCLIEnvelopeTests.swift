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
            "vectors compare", "vectors backfill-norms",
            "remote capabilities", "remote package", "remote upload",
            "remote submit-bundle", "remote jobs", "remote logs", "remote cancel",
            "remote fetch", "remote import", "remote import-chain",
            "remote variants", "remote chat",
            "experiment list", "experiment create", "experiment attach",
            "experiment pin-prompts", "experiment pin-rubric",
            "experiment declare-condition",
            "experiment set-sweep-selection", "experiment set-instruments",
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
        ]
        #expect(declared == expected)
        // The audit's sixteen lifecycle verbs, the three headless authoring
        // verbs step 5½ added (P0-3), and step 7's two (punch list #1 P3 +
        // P13: the sweep's selection criterion and the study's outcome
        // instruments were manifest data no headless caller could declare, so
        // the sweep silently selected on marker density and a pinned choice
        // task was measured as prose).
        #expect(
            declared.filter { $0.hasPrefix("experiment ") }.count == 21,
            "the experiment lifecycle is twenty-one verbs (audit §2.1, §8 P0-3, §9 P3/P13)")
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
