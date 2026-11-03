import Foundation
import Testing

@testable import ExperimentKit

/// `promptReading` on the Mac (issue #6): the manifest key, `experiment
/// set-prompt-reading`, the refusal to send a declaring study to an engine
/// without `remoteStudy.promptReading`, and the one advisory a declaring
/// study gets on this engine.
///
/// The hand-authored contract `Tests/Fixtures/cross-engine/prompt-reading.json`
/// is checked here and in `Server/tests/test_prompt_reading_authoring.py`:
/// the block both clients write, their printed line, their refusal sentences,
/// and the rule's chunking decisions. One parity test also runs the Python
/// client in a subprocess and requires the same block bytes.
///
/// Serialized and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct PromptReadingTests {

    private var repository: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
    }

    private func contract() throws -> [String: Any] {
        let url = repository.appending(path: "Tests/Fixtures/cross-engine/prompt-reading.json")
        return try #require(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }

    func withTempRoot<T>(_ body: (URL) async throws -> T) async rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "prompt-reading-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: temp, withIntermediateDirectories: true)
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
            name: name, description: "", modelID: "org/m",
            modelRevision: String(repeating: "0", count: 40))
    }

    func invoke(_ args: [String]) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(sink: .discarding).run(namespace: "experiment", args)
    }

    /// Python's `json.dumps(…, sort_keys=True, separators=(",", ":"))` for a
    /// block of strings and integers.
    func canonical(_ value: ExactJSONValue?) throws -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(try #require(value)), as: UTF8.self)
    }

    /// A fixture value as exact JSON, number spelling kept (`4096.0` stays a
    /// non-integer, as the Python client reads it).
    func exact(_ object: Any) throws -> ExactJSONValue {
        PromptReading.exactValue(serialized: object)
    }

    // MARK: - The contract

    @Test func theVerbWritesTheContractBlockAndLine() async throws {
        let cases = try #require(try contract()["declarations"] as? [[String: Any]])
        #expect(cases.count == 3)
        for entry in cases {
            try await withTempRoot { _ in
                try draft()
                let flags = try #require(entry["flags"] as? [String])
                let outcome = await invoke(["set-prompt-reading", "demo"] + flags)
                #expect(outcome.envelope.state == .ready, "\(outcome.envelope.error?.reason ?? "")")
                #expect(outcome.envelope.message == entry["line"] as? String)
                #expect(outcome.envelope.changed)
                let written = try ExperimentStore.load(name: "demo").promptReading
                #expect(try canonical(written) == entry["canonical"] as? String)
                #expect(outcome.envelope.result?["reading"] == .string(entry["reading"] as! String))
                #expect(outcome.envelope.result?["written"] == .bool(true))
                #expect(outcome.envelope.result?["previousPromptReading"] == .null)
            }
        }
    }

    @Test func malformedFlagsAreRefusedWithTheContractSentence() async throws {
        let cases = try #require(try contract()["malformedFlags"] as? [[String: Any]])
        try await withTempRoot { _ in
            try draft()
            for entry in cases {
                let flags = try #require(entry["flags"] as? [String])
                let outcome = await invoke(["set-prompt-reading", "demo"] + flags)
                #expect(outcome.envelope.exitCode == 64, "\(flags)")
                #expect(outcome.envelope.error?.reason == entry["reason"] as? String, "\(flags)")
                #expect(outcome.envelope.error?.repairAction.contains(
                    "steerlab-cli experiment set-prompt-reading demo --single-pass") == true)
            }
            #expect(try ExperimentStore.load(name: "demo").promptReading == nil)
        }
    }

    @Test func aHandEditedBlockReadsAsTheContractSays() throws {
        let cases = try #require(try contract()["violations"] as? [[String: Any]])
        #expect(cases.count == 11)
        for entry in cases {
            let raw = try #require(entry["block"])
            let block = try exact(raw)
            #expect(PromptReading.violations(block) == entry["violations"] as? [String],
                    "\(String(describing: entry["block"]))")
        }
    }

    @Test func theRuleChunksAsTheContractSays() throws {
        let cases = try #require(try contract()["chunking"] as? [[String: Any]])
        for entry in cases {
            let raw = try #require(entry["rule"])
            let rule = try #require(PromptReading.rule(try exact(raw)))
            let tokens = try #require(entry["promptTokens"] as? Int)
            #expect(PromptReading.chunkSize(rule, promptTokens: tokens)
                == entry["chunkTokens"] as? Int, "\(entry)")
        }
    }

    @Test func theOutdatedEngineSentenceIsTheContractOne() throws {
        let outdated = try #require(try contract()["outdatedEngine"] as? [String: String])
        #expect(PromptReading.outdatedEngineReason(server: outdated["where"]!)
            == outdated["reason"])
        #expect(PromptReading.outdatedEngineRepair == outdated["repair"])
    }

    // MARK: - The verb's other paths

    @Test func aDryRunWritesNothingAndTheSameDeclarationChangesNothing() async throws {
        try await withTempRoot { _ in
            try draft()
            let dry = await invoke(["set-prompt-reading", "demo", "--single-pass", "--dry-run"])
            #expect(dry.envelope.message.hasPrefix("(dry run, nothing written) would declare"))
            #expect(!dry.envelope.changed)
            #expect(try ExperimentStore.load(name: "demo").promptReading == nil)
            _ = await invoke(["set-prompt-reading", "demo", "--single-pass"])
            let again = await invoke(["set-prompt-reading", "demo", "--single-pass"])
            #expect(!again.envelope.changed)
            #expect(again.envelope.message.hasPrefix("'demo' already declares"))
        }
    }

    @Test func clearRemovesTheDeclaration() async throws {
        try await withTempRoot { _ in
            try draft()
            _ = await invoke(["set-prompt-reading", "demo", "--chunk-tokens", "2048",
                              "--above-tokens", "8192"])
            let cleared = await invoke(["set-prompt-reading", "demo", "--clear"])
            #expect(cleared.envelope.changed)
            #expect(cleared.envelope.result?["promptReading"] == .null)
            #expect(cleared.envelope.result?["previousPromptReading"] == .object([
                "mode": .string("chunked"), "chunkTokens": .number(2048),
                "aboveTokens": .number(8192)]))
            #expect(try ExperimentStore.load(name: "demo").promptReading == nil)
            let nothing = await invoke(["set-prompt-reading", "demo", "--clear"])
            #expect(!nothing.envelope.changed)
            #expect(nothing.envelope.message.contains("nothing to clear"))
        }
    }

    @Test func aFrozenStudyRefusesWithTheDuplicateRepair() async throws {
        try await withTempRoot { root in
            try draft()
            let url = root.appending(path: "experiments/demo/experiment.json")
            var object = try #require(
                JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
            object["status"] = "frozen"
            try JSONSerialization.data(withJSONObject: object).write(to: url)
            for flags in [["--single-pass"], ["--single-pass", "--dry-run"]] {
                let outcome = await invoke(["set-prompt-reading", "demo"] + flags)
                #expect(outcome.envelope.exitCode == 65)
                #expect(outcome.envelope.error?.gate == LifecycleGate.statusImmutable.rawValue)
                #expect(outcome.envelope.error?.repairAction.contains("duplicate demo") == true)
            }
        }
    }

    // MARK: - The manifest key

    @Test func theDeclarationIsContentAndSurvivesARoundTrip() async throws {
        try await withTempRoot { root in
            try draft()
            let before = ExperimentStore.manifestHash(try ExperimentStore.load(name: "demo"))
            _ = await invoke(["set-prompt-reading", "demo", "--chunk-tokens", "4096",
                              "--above-tokens", "8192"])
            let loaded = try ExperimentStore.load(name: "demo")
            #expect(ExperimentStore.manifestHash(loaded) != before)
            #expect(loaded.unknownTopLevelFields.isEmpty)
            #expect(PromptReading.rule(loaded.promptReading) == .chunked(4096, above: 8192))
            #expect(ExperimentStore.verify(loaded).allSatisfy { !$0.contains("promptReading") })
            // The file holds integers, as the Python client writes them.
            let url = root.appending(path: "experiments/demo/experiment.json")
            let object = try #require(
                JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
            let block = try #require(object["promptReading"] as? [String: Any])
            #expect(block["chunkTokens"] as? Int == 4096)
            #expect(block["mode"] as? String == "chunked")
        }
    }

    @Test func aPythonWrittenManifestIsModelledNotCarriedAsUnknown() throws {
        // A key this build did not model would block freeze here; this one
        // is modelled, and a null reads as no declaration.
        let base: [String: Any] = [
            "name": "x", "experimentDescription": "", "modelID": "org/m",
            "status": "draft", "createdAt": "1970-01-01T00:00:00Z",
        ]
        var with = base
        with["promptReading"] = ["mode": "chunked", "chunkTokens": 1024, "aboveTokens": 0]
        let decoded = try JSONDecoder().decode(
            ExperimentManifest.self, from: JSONSerialization.data(withJSONObject: with))
        #expect(decoded.unknownTopLevelFields.isEmpty)
        #expect(PromptReading.rule(decoded.promptReading) == .chunked(1024, above: 0))
        var null = base
        null["promptReading"] = NSNull()
        let nulled = try JSONDecoder().decode(
            ExperimentManifest.self, from: JSONSerialization.data(withJSONObject: null))
        #expect(nulled.promptReading == nil)
    }

    /// `JSONDecoder` reads `"chunkTokens": 4096.0` as 4096; verify reads the
    /// file as written and refuses it with the Python client's sentence, and
    /// declaring 4096 again rewrites it rather than saying it is already so.
    @Test func aFractionalSpellingInTheFileIsAVerifyViolation() async throws {
        try await withTempRoot { root in
            try draft()
            _ = await invoke(["set-prompt-reading", "demo", "--chunk-tokens", "4096",
                              "--above-tokens", "8192"])
            let url = root.appending(path: "experiments/demo/experiment.json")
            let text = try String(contentsOf: url, encoding: .utf8)
            #expect(text.contains("\"chunkTokens\" : 4096"))
            try text.replacingOccurrences(of: "\"chunkTokens\" : 4096",
                                          with: "\"chunkTokens\" : 4096.0")
                .write(to: url, atomically: true, encoding: .utf8)
            let loaded = try ExperimentStore.load(name: "demo")
            #expect(ExperimentStore.verify(loaded).contains(
                "promptReading chunkTokens must be a whole number from 1 to 100,000,000"))
            let again = await invoke(["set-prompt-reading", "demo", "--chunk-tokens", "4096",
                                      "--above-tokens", "8192"])
            #expect(again.envelope.changed)
            #expect(ExperimentStore.verify(try ExperimentStore.load(name: "demo"))
                .allSatisfy { !$0.contains("promptReading") })
        }
    }

    @Test func aMalformedBlockIsAVerifyViolation() async throws {
        try await withTempRoot { _ in
            var manifest = try draft()
            manifest.promptReading = try exact(["mode": "chunked", "chunkTokens": 0,
                                                "aboveTokens": 8192])
            try ExperimentStore.save(manifest)
            #expect(ExperimentStore.verify(try ExperimentStore.load(name: "demo")).contains(
                "promptReading chunkTokens must be a whole number from 1 to 100,000,000"))
        }
    }

    // MARK: - Cross-client parity

    /// The Python client and this verb, on identical drafts, write the same
    /// block bytes.
    @Test func bothClientsWriteTheSameBlock() async throws {
        try await withTempRoot { root in
            try draft("mac")
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
            let local = repository.appending(path: "Server/.venv.nosync/bin/python").path
            let executable = ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"]
                ?? (FileManager.default.isExecutableFile(atPath: local) ? local : "python3")
            process.arguments = [executable, "-c", """
                import contextlib, io, json, sys
                from steerlab_server import client_cli
                from steerlab_server.experiment import experiment_store as store
                root = sys.argv[1]
                store.create("py", model_id="org/m", revision="0" * 40, root=root)
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code = client_cli.main(["--root", root, "experiment", "set-prompt-reading",
                        "py", "--chunk-tokens", "2048", "--above-tokens", "8192", "--json"])
                block = store.load_raw("py", root)["promptReading"]
                print(json.dumps({"code": code,
                    "canonical": json.dumps(block, sort_keys=True, separators=(",", ":"))}))
                """, root.path]
            var environment = ProcessInfo.processInfo.environment
            environment["PYTHONPATH"] = repository.appending(component: "Server").path
            environment["HF_HUB_OFFLINE"] = "1"
            process.environment = environment
            let output = Pipe(), errors = Pipe()
            process.standardOutput = output
            process.standardError = errors
            try process.run()
            let data = output.fileHandleForReading.readDataToEndOfFile()
            let diagnostics = errors.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            #expect(process.terminationStatus == 0, "\(String(decoding: diagnostics, as: UTF8.self))")
            let python = try #require(
                JSONSerialization.jsonObject(with: data) as? [String: Any])
            #expect(python["code"] as? Int == 0)
            let outcome = await invoke(["set-prompt-reading", "mac", "--chunk-tokens", "2048",
                                        "--above-tokens", "8192"])
            #expect(outcome.envelope.state == .ready)
            #expect(try canonical(ExperimentStore.load(name: "mac").promptReading)
                == python["canonical"] as? String)
        }
    }

    // MARK: - Engines

    @Test func aDeclaringStudyIsRefusedByAServerWithoutTheCapability() throws {
        var manifest = ExperimentManifest(name: "demo", description: "", modelID: "org/m")
        manifest.promptReading = PromptReading.Rule.single.jsonValue
        let old = try JSONDecoder().decode(
            ClusterCapabilities.self,
            from: Data(#"{"remoteStudy": {"declaredSeedPolicy": true}}"#.utf8))
        for server in [old, nil] {
            for verb in ["run", "pipeline", "sweep"] {
                #expect(SubstrateRouting.promptReadingSubmissionRefusal(
                    manifest: manifest, verb: verb, capabilities: server,
                    server: "http://runner.invalid")
                    == PromptReading.outdatedEngineReason(server: "http://runner.invalid"))
            }
        }
        // A verb whose prompts the declaration does not govern (the engine
        // refuses a plan for it), a study that declares nothing, and an
        // engine that honours it all proceed.
        #expect(PromptReading.readingVerbs == ["run", "sweep", "pipeline"])
        for verb in ["extract", "validate", "evaluate", "verify", "analyze"] {
            #expect(SubstrateRouting.promptReadingSubmissionRefusal(
                manifest: manifest, verb: verb, capabilities: old, server: "s") == nil)
        }
        #expect(SubstrateRouting.promptReadingSubmissionRefusal(
            manifest: ExperimentManifest(name: "x", description: "", modelID: "m"),
            verb: "run", capabilities: old, server: "s") == nil)
        for json in [#"{"remoteStudy": {"promptReading": true}}"#,
                     #"{"features": {"remoteStudy.promptReading": true}}"#] {
            let new = try JSONDecoder().decode(ClusterCapabilities.self, from: Data(json.utf8))
            #expect(new.supportsPromptReading)
            #expect(SubstrateRouting.promptReadingSubmissionRefusal(
                manifest: manifest, verb: "run", capabilities: new, server: "s") == nil)
        }
        // A nested value wins over the dotted one, as on the Python client.
        let nested = try JSONDecoder().decode(
            ClusterCapabilities.self,
            from: Data(#"{"remoteStudy": {"promptReading": false}, "features": {"remoteStudy.promptReading": true}}"#.utf8))
        #expect(!nested.supportsPromptReading)
    }

    @Test func theMacEngineAdvisesAndNamesItsOwnStep() throws {
        var manifest = ExperimentManifest(name: "demo", description: "", modelID: "org/m")
        #expect(PromptReading.macEngineAdvisory(manifest) == nil)
        manifest.promptReading = PromptReading.Rule.chunked(4096, above: 8192).jsonValue
        let advisory = try #require(PromptReading.macEngineAdvisory(manifest))
        #expect(advisory.contains("The declaration applies to the Python engine"))
        #expect(advisory.contains("in its own steps of 512 tokens"))
        #expect(advisory.contains("chunks of 4,096 tokens"))
        #expect(PromptReading.mlxPrefillStepSize == 512)
    }
}
