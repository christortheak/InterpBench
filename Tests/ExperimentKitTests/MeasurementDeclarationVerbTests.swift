import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// `experiment set-parser` + `experiment set-instrument-scope` — the writers
/// for the three manifest fields the app's pickers owned exclusively
/// (`numericParser`, `parserRegistryHash`, `outcomeInstrumentScope`), plus
/// `remote submit-bundle --parallel`, the fan-out parameter whose machinery
/// was complete underneath and reachable only from the app's stepper.
///
/// All three gaps are the same shape as the `set-sampling` gap: a field the
/// engine already READS, declared in the panel, with no way to type it. Here
/// the consequence was sharper than an unauthorable arm — a headless study
/// with a numeric endpoint fell back to the DEPRECATED implicit selection
/// (`caseFamily: "sentencing"` → the built-in duration parser), and a
/// mixed-format study could only follow the run-start gate's LOSSY repair
/// (`set-instruments … sampledText`, which drops the instrument) because the
/// non-lossy one the same refusal names had no writer.
///
/// Serialized and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct MeasurementDeclarationVerbTests {

    // MARK: Harness

    func withTempRoot<T>(_ body: (URL) async throws -> T) async rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "measurement-verb-\(UUID().uuidString)")
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

    @discardableResult
    func draft(_ name: String = "demo") throws -> ExperimentManifest {
        try ExperimentStore.create(
            name: name, description: "", modelID: "test/model")
    }

    @discardableResult
    func invoke(_ args: [String]) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(sink: .discarding)
            .run(namespace: "experiment", args)
    }

    /// The registry the workspace template seeds, copied into the temp root.
    /// Returns its SHA-256 — the value a declaration must pin.
    @discardableResult
    func plantRegistry(into root: URL) throws -> String {
        let source = VectorCatalog.bundledSeedRoot
            .appending(path: ParserRegistry.registryFile)
        let data = try Data(contentsOf: source)
        let destination = root.appending(path: ParserRegistry.registryFile)
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try data.write(to: destination)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }
            .joined()
    }

    /// A genuinely MIXED task file: two `label` rows and one `json` reasons
    /// row, ALL carrying options and a target. That is the shape the
    /// run-start refusal exists for — the reasons arm asks the same question
    /// but wants the choice inside a JSON object, so the first generated
    /// token is `{` and an answer-token instrument cannot read it. A json
    /// row with no options would not trip the gate at all (the rule only
    /// counts option-carrying rows), and would prove nothing here.
    @discardableResult
    func plantMixedPrompts(
        _ relativePath: String = "prompts/tasks/mixed.jsonl"
    ) throws -> String {
        let rows = [
            #"{"id":"case-1","text":"Affirm or reverse?","options":["A","B"],"target":"A","responseFormat":"label"}"#,
            #"{"id":"case-2","text":"Affirm or reverse?","options":["A","B"],"target":"B","responseFormat":"label"}"#,
            #"{"id":"reasons-1","text":"Affirm or reverse, with reasons.","options":["A","B"],"target":"A","responseFormat":"json"}"#,
        ]
        let url = ExperimentStore.resolveProjectPath(relativePath)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try (rows.joined(separator: "\n") + "\n").write(
            to: url, atomically: true, encoding: .utf8)
        return relativePath
    }

    // MARK: - set-parser

    /// The declaration and its pin, in one invocation: the name is stored,
    /// the registry's CURRENT bytes are hashed into `parserRegistryHash`,
    /// and the echo carries the kind read back out of the registry.
    @Test func theDeclaredParserPinsTheRegistrysCurrentBytes() async throws {
        try await withTempRoot { root in
            let hash = try plantRegistry(into: root)
            try draft()
            let outcome = await invoke(
                ["set-parser", "demo", "sentencing-months"])
            #expect(outcome.envelope.state == .ready)
            #expect(outcome.envelope.exitCode == 0)
            #expect(
                outcome.envelope.result?["numericParser"]
                    == .string("sentencing-months"))
            #expect(outcome.envelope.result?["parserKind"] == .string("durationMonths"))
            #expect(
                outcome.envelope.result?["parserRegistryHash"] == .string(hash))
            #expect(
                outcome.envelope.result?["registryFile"]
                    == .string(ParserRegistry.registryFile))
            // The MANIFEST, not the echo, is the contract.
            let loaded = try ExperimentStore.load(name: "demo")
            #expect(loaded.numericParser == "sentencing-months")
            #expect(loaded.parserRegistryHash == hash)
            // …and the declaration resolves to a runnable parser, which is
            // the whole point of pinning it.
            let resolved = try #require(
                try ParserRegistry.resolveNumericParser(loaded))
            #expect(resolved.name == "sentencing-months")
            #expect(resolved.parse("8 years 3 months") == 99)
        }
    }

    /// "" clears the declaration AND its pin — an unused pin certifies
    /// nothing and is a verify finding, so the two move together.
    @Test func theEmptyStringClearsTheParserAndItsPin() async throws {
        try await withTempRoot { root in
            try plantRegistry(into: root)
            try draft()
            await invoke(["set-parser", "demo", "plain-number"])
            let outcome = await invoke(["set-parser", "demo", ""])
            #expect(outcome.envelope.exitCode == 0)
            #expect(outcome.envelope.result?["numericParser"] == .null)
            #expect(outcome.envelope.result?["parserRegistryHash"] == .null)
            let loaded = try ExperimentStore.load(name: "demo")
            #expect(loaded.numericParser == nil)
            #expect(loaded.parserRegistryHash == nil)
            #expect(ParserRegistry.pinViolations(loaded).isEmpty)
        }
    }

    /// A name the registry does not define is a MALFORMED invocation (64)
    /// carrying the registry's own sentence, and its repair names the
    /// parsers that ARE defined — read from the registry, so the repair and
    /// the refusal cannot disagree.
    @Test func anUndefinedParserIsMalformedAndWritesNothing() async throws {
        try await withTempRoot { root in
            try plantRegistry(into: root)
            try draft()
            let outcome = await invoke(["set-parser", "demo", "no-such-parser"])
            #expect(outcome.envelope.state == .blocked)
            #expect(outcome.envelope.exitCode == 64)
            #expect(outcome.envelope.error?.code == "usage")
            #expect(outcome.envelope.error?.gate == nil)
            let reason = try #require(outcome.envelope.error?.reason)
            #expect(
                reason == "the registry defines no parser named "
                    + "'no-such-parser' — defined: plain-number, "
                    + "sentencing-months")
            let repair = try #require(outcome.envelope.error?.repairAction)
            for defined in ["plain-number", "sentencing-months"] {
                #expect(repair.contains(defined), "repair omits \(defined)")
            }
            let loaded = try ExperimentStore.load(name: "demo")
            #expect(loaded.numericParser == nil)
            #expect(loaded.parserRegistryHash == nil)
        }
    }

    /// No registry at all is the same class of refusal, not an operational
    /// failure: the caller named a parser on a workspace that declares none.
    @Test func aMissingRegistryIsMalformedToo() async throws {
        try await withTempRoot { _ in
            try draft()
            let outcome = await invoke(["set-parser", "demo", "sentencing-months"])
            #expect(outcome.envelope.exitCode == 64)
            #expect(
                outcome.envelope.error?.reason.contains(
                    "no parser registry exists at "
                        + ParserRegistry.registryFile) == true)
        }
    }

    /// The registry is the AUTHORITY on which parser version a study
    /// preregistered, so the hash can never be typed. Structural, because
    /// the guarantee is the absence of a flag: `set-parser` declares no
    /// value flags at all.
    @Test func theRegistryHashIsNeverAnArgument() throws {
        let spec = try #require(
            ExperimentCLIParser.specs.first {
                $0.namespace == "experiment" && $0.verb == "set-parser"
            })
        #expect(spec.valueFlags.isEmpty)
        #expect(spec.booleanFlags.isEmpty)
        #expect(!spec.purpose.lowercased().contains("--registry-hash"))
    }

    /// Clearing the declaration hands a `sentencing` study back to the
    /// DEPRECATED implicit rule — said at the moment of clearing, in the one
    /// sentence every firing site uses.
    @Test func clearingAdvisesWhenTheImplicitRuleTakesOver() async throws {
        try await withTempRoot { root in
            try plantRegistry(into: root)
            var manifest = try draft("sentencing-study")
            manifest.caseFamily = "sentencing"
            try ExperimentStore.save(manifest)
            let declared = await invoke(
                ["set-parser", "sentencing-study", "sentencing-months"])
            // A DECLARED parser displaces the implicit rule, so nothing is
            // said on the way in.
            #expect(declared.envelope.advisories?.isEmpty != false)
            let cleared = await invoke(["set-parser", "sentencing-study", ""])
            #expect(cleared.envelope.exitCode == 0)
            #expect(
                cleared.envelope.advisories?.contains {
                    $0.code == CLIAdvisory.deprecatedImplicitSelection.rawValue
                        && $0.detail
                            == ExperimentManifest.implicitCaseFamilyAdvisory
                } == true)
        }
    }

    // MARK: - set-instrument-scope

    /// The motivating case, end to end: a mixed json+label file with a
    /// choice instrument declared is REFUSED at run start, and the scope
    /// declaration — the non-lossy repair the refusal itself names — makes
    /// the same study pass without dropping the instrument.
    @Test func theScopeIsTheNonLossyRepairForAMixedFormatStudy() async throws {
        try await withTempRoot { _ in
            try draft("mixed")
            let file = try plantMixedPrompts()
            #expect(await invoke(["pin-prompts", "mixed", file]).exitCode == 0)
            #expect(
                await invoke([
                    "set-instruments", "mixed", "answerTokenLogprob,sampledText",
                ]).exitCode == 0)

            // Before: the run-start gate refuses, and its own words offer
            // both repairs — drop the instrument, or declare the scope.
            let prompts = try ExperimentTasks.loadTaskPrompts(
                for: try ExperimentStore.load(name: "mixed"))
            #expect {
                try ExperimentTasks.checkResponseFormats(
                    prompts.prompts,
                    manifest: try ExperimentStore.load(name: "mixed"))
            } throws: { error in
                let reason = (error as? ExperimentError)?.reason ?? ""
                return reason.contains("declare outcomeInstrumentScope")
            }

            let outcome = await invoke(
                ["set-instrument-scope", "mixed", "label"])
            #expect(outcome.envelope.state == .ready)
            #expect(outcome.envelope.exitCode == 0)
            #expect(
                outcome.envelope.result?["responseFormats"]
                    == .array([.string("label")]))
            #expect(outcome.envelope.result?["itemCount"] == .number(2))

            // The MANIFEST carries the whole pin, computed from the study's
            // own items — the researcher picked a format, never a hash.
            let scoped = try ExperimentStore.load(name: "mixed")
            let pin = try #require(scoped.outcomeInstrumentScope)
            #expect(pin.responseFormats == ["label"])
            #expect(pin.itemCount == 2)
            #expect(
                pin.itemIDsHash
                    == ResponseFormat.Scope.idsHash([
                        .init(id: "case-1", hasOptions: true, format: .label),
                        .init(id: "case-2", hasOptions: true, format: .label),
                    ]))
            #expect(outcome.envelope.result?["itemIDsHash"] == .string(pin.itemIDsHash))

            // After: the same gate passes, with `answerTokenLogprob` still
            // declared alongside `sampledText`.
            #expect(scoped.outcomeInstruments == ["answerTokenLogprob", "sampledText"])
            try ExperimentTasks.checkResponseFormats(
                prompts.prompts, manifest: scoped)
        }
    }

    /// REPLACE, not merge — like `set-instruments` and `set-exclusions`. The
    /// pin is one subset: a second declaration recomputes the count and the
    /// hash from the whole new format list rather than appending to the old.
    @Test func aSecondDeclarationReplacesThePinRatherThanMerging() async throws {
        try await withTempRoot { _ in
            try draft("mixed")
            let file = try plantMixedPrompts()
            await invoke(["pin-prompts", "mixed", file])
            await invoke(["set-instrument-scope", "mixed", "label"])
            #expect(
                try ExperimentStore.load(name: "mixed")
                    .outcomeInstrumentScope?.itemCount == 2)
            let outcome = await invoke(
                ["set-instrument-scope", "mixed", "label,json"])
            #expect(outcome.envelope.exitCode == 0)
            let pin = try #require(
                try ExperimentStore.load(name: "mixed").outcomeInstrumentScope)
            #expect(pin.responseFormats == ["label", "json"])
            #expect(pin.itemCount == 3)
        }
    }

    @Test func theEmptyStringClearsTheScope() async throws {
        try await withTempRoot { _ in
            try draft("mixed")
            let file = try plantMixedPrompts()
            await invoke(["pin-prompts", "mixed", file])
            await invoke(["set-instrument-scope", "mixed", "label"])
            let outcome = await invoke(["set-instrument-scope", "mixed", ""])
            #expect(outcome.envelope.exitCode == 0)
            #expect(outcome.envelope.result?["itemCount"] == .number(0))
            #expect(outcome.envelope.result?["itemIDsHash"] == .null)
            #expect(
                try ExperimentStore.load(name: "mixed")
                    .outcomeInstrumentScope == nil)
        }
    }

    /// Clearing needs no prompts pin — it derives nothing (review round 10,
    /// finding 10).
    ///
    /// The prompts-file guard and the file read used to precede the empty
    /// branch, so `""` refused in exactly the three states that make clearing
    /// necessary: no pin at all, a pin whose file is gone, and a pin that
    /// drifted. A stale scope was then unremovable except by hand-editing the
    /// manifest. DECLARING still requires the pin, which the last case here
    /// holds still.
    @Test func theEmptyStringClearsEvenWhenThePromptsPinIsBroken() async throws {
        try await withTempRoot { _ in
            // (a) No prompts pin at all — nothing to clear yet, but the verb
            // must not refuse the request to clear.
            try draft("nopin")
            let unpinned = await invoke(["set-instrument-scope", "nopin", ""])
            #expect(unpinned.envelope.state == .ready)
            #expect(unpinned.envelope.exitCode == 0)
            #expect(
                try ExperimentStore.load(name: "nopin")
                    .outcomeInstrumentScope == nil)

            // (b) A scope declared against a real file, then the FILE goes
            // missing. This is the state the fix exists for: a pinned path
            // that no longer resolves, and a scope still standing.
            try draft("missing")
            let file = try plantMixedPrompts("prompts/tasks/gone.jsonl")
            #expect(await invoke(["pin-prompts", "missing", file]).exitCode == 0)
            #expect(
                await invoke(["set-instrument-scope", "missing", "label"])
                    .exitCode == 0)
            #expect(
                try ExperimentStore.load(name: "missing")
                    .outcomeInstrumentScope != nil)
            try FileManager.default.removeItem(
                at: ExperimentStore.resolveProjectPath(file))
            let cleared = await invoke(["set-instrument-scope", "missing", ""])
            #expect(cleared.envelope.state == .ready)
            #expect(cleared.envelope.exitCode == 0)
            #expect(
                try ExperimentStore.load(name: "missing")
                    .outcomeInstrumentScope == nil)

            // (c) DRIFT — the pinned file's bytes changed under the scope. The
            // clear still lands.
            try draft("drifted")
            let drifting = try plantMixedPrompts("prompts/tasks/drift.jsonl")
            #expect(
                await invoke(["pin-prompts", "drifted", drifting]).exitCode == 0)
            #expect(
                await invoke(["set-instrument-scope", "drifted", "label"])
                    .exitCode == 0)
            try #"{"id":"only","text":"?","options":["A","B"],"target":"A","responseFormat":"json"}"#
                .appending("\n")
                .write(
                    to: ExperimentStore.resolveProjectPath(drifting),
                    atomically: true, encoding: .utf8)
            let afterDrift = await invoke(["set-instrument-scope", "drifted", ""])
            #expect(afterDrift.envelope.exitCode == 0)
            #expect(
                try ExperimentStore.load(name: "drifted")
                    .outcomeInstrumentScope == nil)

            // …and DECLARING (non-empty) still requires the pin: the guard
            // moved, it did not go.
            let declaring = await invoke(
                ["set-instrument-scope", "nopin", "label"])
            #expect(declaring.envelope.exitCode != 0)
            #expect(
                declaring.envelope.message
                    .contains("declare the task prompts first"))
        }
    }

    /// An unrecognised format would select nothing and pin "zero items" —
    /// `Scope.includes` is a raw string comparison. Refused at 64 with the
    /// vocabulary named, and nothing written.
    @Test func anUnknownResponseFormatIsMalformedAndWritesNothing() async throws {
        try await withTempRoot { _ in
            try draft("mixed")
            let file = try plantMixedPrompts()
            await invoke(["pin-prompts", "mixed", file])
            let outcome = await invoke(
                ["set-instrument-scope", "mixed", "label,lbl"])
            #expect(outcome.envelope.state == .blocked)
            #expect(outcome.envelope.exitCode == 64)
            #expect(outcome.envelope.error?.code == "usage")
            #expect(
                outcome.envelope.error?.reason
                    == "unknown responseFormat 'lbl' — known: label, json, "
                    + "freeText")
            let repair = try #require(outcome.envelope.error?.repairAction)
            for format in ExperimentStore.knownResponseFormats {
                #expect(repair.contains(format), "repair omits \(format)")
            }
            #expect(
                try ExperimentStore.load(name: "mixed")
                    .outcomeInstrumentScope == nil)
        }
    }

    /// A well-spelled format the file has NO rows of is refused at the
    /// declaration rather than at the run: a scope selecting zero items
    /// would send the instrument at nothing and produce zero records, which
    /// is the very failure the run-start zero-item rules exist for.
    @Test func aScopeThatSelectsNothingIsRefusedAtDeclaration() async throws {
        try await withTempRoot { _ in
            try draft("mixed")
            let file = try plantMixedPrompts()
            await invoke(["pin-prompts", "mixed", file])
            let outcome = await invoke(
                ["set-instrument-scope", "mixed", "freeText"])
            #expect(outcome.envelope.state == .blocked)
            #expect(outcome.envelope.exitCode == 64)
            let reason = try #require(outcome.envelope.error?.reason)
            #expect(
                reason == "the declared outcomeInstrumentScope selects zero "
                    + "task items of '\(file)' — the instruments would run on "
                    + "nothing and silently produce zero records")
            #expect(
                try ExperimentStore.load(name: "mixed")
                    .outcomeInstrumentScope == nil)
        }
    }

    /// The scope pins rows of a FILE, so the file has to be pinned first —
    /// the refusal names the verb that does it.
    @Test func aScopeWithoutPinnedPromptsNamesPinPrompts() async throws {
        try await withTempRoot { _ in
            try draft("bare")
            let outcome = await invoke(["set-instrument-scope", "bare", "label"])
            #expect(outcome.envelope.exitCode != 0)
            #expect(
                outcome.envelope.error?.reason.contains(
                    "experiment pin-prompts bare") == true)
            #expect(
                try ExperimentStore.load(name: "bare")
                    .outcomeInstrumentScope == nil)
        }
    }

    // MARK: - remote submit-bundle --parallel

    /// A non-integer fan-out is a malformed invocation, refused before any
    /// request leaves the machine (the endpoint here answers nothing).
    @Test func aNonIntegerParallelIsMalformedBeforeAnyRequest() async throws {
        let outcome = await ExperimentCLIRunner(sink: .discarding).run(
            namespace: "remote",
            [
                "submit-bundle", "/srv/bundles/demo", "--url",
                "http://127.0.0.1:1", "--verb", "run", "--executor", "slurm",
                "--parallel", "four",
            ])
        #expect(outcome.envelope.state == .blocked)
        #expect(outcome.envelope.exitCode == 64)
        #expect(outcome.envelope.error?.code == "usage")
        #expect(
            outcome.envelope.error?.reason
                == "--parallel must be a positive integer, not 'four'")
        #expect(
            outcome.envelope.error?.repairAction.contains("--parallel 4") == true)
    }

    @Test func aZeroParallelIsMalformedToo() async throws {
        let outcome = await ExperimentCLIRunner(sink: .discarding).run(
            namespace: "remote",
            [
                "submit-bundle", "/srv/bundles/demo", "--url",
                "http://127.0.0.1:1", "--parallel", "0",
            ])
        #expect(outcome.envelope.exitCode == 64)
        #expect(
            outcome.envelope.error?.reason
                == "--parallel must be a positive integer, not '0'")
    }

    /// The flag is declared on the verb (so the parser accepts it rather
    /// than rejecting it as undeclared), and its help carries the field
    /// caveat: a fan-out can partially fail while the submit exits 0.
    @Test func theParallelFlagIsDeclaredAndItsHelpCarriesTheCaveat() throws {
        let spec = try #require(
            ExperimentCLIParser.specs.first {
                $0.namespace == "remote" && $0.verb == "submit-bundle"
            })
        #expect(spec.valueFlags.contains("--parallel"))
        #expect(CLIFlagVocabulary.metavar("--parallel") == "<n>")
        let help = CLIFlagVocabulary.purpose("--parallel")
        #expect(help.contains("verify the shard jobs landed"))
    }

    // MARK: - remote submit-bundle --source

    /// The source-run override is declared on the verb, spelled `--source`
    /// like every other surface (`bundle execute --source`, `study submit
    /// --source`), and its help names the workflow it exists for: a renamed
    /// duplicate has no runs under its own name, so without the override a
    /// measurement submission dies at name-scoped discovery before the epoch
    /// tolerance can rule.
    @Test func theSourceFlagIsDeclaredAndItsHelpNamesTheDuplicatePath() throws {
        let spec = try #require(
            ExperimentCLIParser.specs.first {
                $0.namespace == "remote" && $0.verb == "submit-bundle"
            })
        #expect(spec.valueFlags.contains("--source"))
        #expect(CLIFlagVocabulary.metavar("--source") == "<run-dir>")
        let help = CLIFlagVocabulary.purpose("--source")
        #expect(help.contains("renamed"))
        #expect(help.contains("SUBMIT time"))
    }
}
