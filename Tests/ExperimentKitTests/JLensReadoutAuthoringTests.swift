import Foundation
import Testing

@testable import ExperimentKit

/// `experiment set-jlens-readout` on this command line, and its parity with
/// the Python client's twin (`Server/tests/test_jlens_readout_authoring.py`).
///
/// This engine never reads a lens record, so the lens pins are flags. The
/// parity test runs the Python client against a real lens record in a
/// subprocess, feeds its `result.macCommand` to this verb on an identical
/// draft, and requires the same block — compared as Python's canonical JSON
/// bytes, configuration hash included — and the same design identity on both
/// clients' hashers.
///
/// Serialized and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct JLensReadoutAuthoringTests {

    static let model = "google/gemma-3-27b-it"
    static let revision = "005ad3404e59d6023443cb575daa05336842228a"
    static let lensSHA = String(repeating: "a", count: 64)
    static let tokenizer = String(repeating: "c", count: 64)

    private var repository: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
    }

    // MARK: Harness

    func withTempRoot<T>(_ body: (URL) async throws -> T) async rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "jlens-readout-verb-\(UUID().uuidString)")
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

    /// A draft on the evidence-tier model with the pins a readout keys on.
    @discardableResult
    func draft(_ name: String = "demo", maxTokens: Int = 2048) throws -> ExperimentManifest {
        var manifest = try ExperimentStore.create(
            name: name, description: "", modelID: Self.model, modelRevision: Self.revision)
        manifest.dtype = "bfloat16"
        manifest.maxTokens = maxTokens
        try ExperimentStore.save(manifest)
        return try ExperimentStore.load(name: name)
    }

    func invoke(_ args: [String]) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(sink: .discarding).run(namespace: "experiment", args)
    }

    /// The lens pins, as the Python client prints them.
    func declare(_ extra: [String], name: String = "demo", lens: String = "lens-a",
                 lensModel: String = model, fitted: String = "0-60") -> [String]
    {
        ["set-jlens-readout", name, lens, "--lens-sha256", Self.lensSHA,
         "--lens-model", lensModel, "--fitted-layers", fitted] + extra
    }

    func block(_ name: String = "demo") throws -> [String: JSONValue]? {
        guard case .object(let block)? = try ExperimentStore.load(name: name).jlensReadout
        else { return nil }
        return block
    }

    func python(_ script: String, root: URL) throws -> [String: Any] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        let local = repository.appending(path: "Server/.venv.nosync/bin/python").path
        let executable = ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"]
            ?? (FileManager.default.isExecutableFile(atPath: local) ? local : "python3")
        process.arguments = [executable, "-c", """
            import contextlib, io, json, os, sys
            root = sys.argv[1]
            os.environ["HF_HUB_CACHE"] = os.path.join(root, "hub")
            from steerlab_server import client_cli
            from steerlab_server.client import design_identity
            from steerlab_server.experiment import experiment_store as store
            from steerlab_server.jlens import lens_store, schemas
            """ + "\n" + script, root.path]
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
        guard process.terminationStatus == 0 else {
            throw ExperimentError(reason: String(decoding: diagnostics, as: UTF8.self)
                + "\nSet TEST_RUNNER_STEERLAB_TEST_PYTHON=<client-python> in the shell "
                + "environment before xcodebuild.")
        }
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    // MARK: - Cross-client parity

    /// The Python client reads the lens record and the model cache; this verb
    /// takes the pins it printed. Same block bytes, same configuration hash,
    /// same design identity — with every optional key present, a qualification
    /// id that needs escaping, and a declared budget raise.
    @Test func bothClientsWriteTheSameBlockAndDesignIdentity() async throws {
        try await withTempRoot { root in
            try draft("py")
            try draft("mac")
            let first = try python("""
                model = "\(Self.model)"
                revision = "\(Self.revision)"
                record = schemas.JLensRecord(
                    lensID="lens-a",
                    source=schemas.SourceRef(repo="org/lenses", folder="f", tensorFile="t.pt",
                                             configFile="c.yaml", tensorSHA256="\(Self.lensSHA)"),
                    fit=schemas.FitProvenance(modelID=model),
                    sourceLayers=list(range(0, 61)), dModel=8, targetLayer=61, nPrompts=1,
                    converted=schemas.ConvertedRef(path="runs/jlens-lenses/lens-a/j.safetensors",
                                                   dtype="float16", sha256="b" * 64))
                record.qualifications.append(schemas.Qualification(
                    qualificationID="q\\u2014one", modelID=model, revision=revision,
                    dtype="bfloat16", tier="evidence", passed=True, lensSHA256=record.source.tensorSHA256,
                    convertedSHA256="b" * 64, layers=[20, 21, 22, 26]))
                lens_store.save(record, root)
                snapshot = os.path.join(root, "hub", "models--" + model.replace("/", "--"),
                                        "snapshots", revision)
                os.makedirs(snapshot)
                with open(os.path.join(snapshot, "tokenizer.json"), "w") as handle:
                    handle.write('{"model": "parity"}')
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code = client_cli.main(["--root", root, "experiment", "set-jlens-readout", "py",
                        "lens-a", "--layers", "20-22,26", "--top-k", "7", "--top-k-layers", "21,26",
                        "--watchlist", "11,3", "--no-companion", "--max-full-vocab-projections",
                        "500000", "--keep-tokens", "--json"])
                envelope = json.loads(out.getvalue())
                block = store.load_raw("py", root)["jlensReadout"]
                print(json.dumps({"code": code, "envelope": envelope,
                    "canonical": json.dumps(block, sort_keys=True, separators=(",", ":"))}))
                """, root: root)
            #expect(first["code"] as? Int == 0)
            let envelope = try #require(first["envelope"] as? [String: Any])
            let result = try #require(envelope["result"] as? [String: Any])
            let command = try #require(result["macCommand"] as? String)

            // The printed spelling, aimed at the twin draft, run as typed.
            var words = command.split(separator: " ").map(String.init)
            #expect(Array(words.prefix(4)) == ["steerlab-cli", "experiment", "set-jlens-readout", "py"])
            words[3] = "mac"
            let outcome = await invoke(Array(words.dropFirst(2)))
            #expect(outcome.envelope.state == .ready, "\(outcome.envelope.error?.reason ?? "")")

            let pythonBlock = try #require(try block("py"))
            let macBlock = try #require(try block("mac"))
            #expect(pythonBlock == macBlock)
            #expect(JLensReadoutAuthoring.pythonJSON(.object(macBlock))
                == first["canonical"] as? String)
            #expect(macBlock["qualificationID"] == .string("q\u{2014}one"))
            #expect(macBlock["configHash"] != nil)
            let recomputed = JLensReadoutAuthoring.configurationHash(macBlock)
            #expect(pythonBlock["configHash"] == .string(recomputed))
            #expect(try ExperimentStore.load(name: "mac").recordTokenIDs)

            // The same design on both clients' hashers, for both manifests.
            func identity(_ name: String) throws -> String {
                let study = try ExperimentStore.load(name: name)
                return try PortableDesignIdentity.hash(
                    StudyTemplate(name: "identity", study: StudyTemplateStore.strippedBody(study)))
            }
            #expect(try identity("py") == identity("mac"))
            let hashes = try python("""
                def identity(name):
                    with open(os.path.join(root, "experiments", name, "experiment.json")) as handle:
                        study = json.load(handle)
                    return design_identity.content_hash({"schemaVersion": 1,
                        "study": design_identity.stripped(study), "semanticScenario": None})
                print(json.dumps({"py": identity("py"), "mac": identity("mac")}))
                """, root: root)
            #expect(hashes["py"] as? String == (try identity("py")))
            #expect(hashes["mac"] as? String == (try identity("mac")))

            // The echoes agree too: the same flat keys, the same pins, the
            // same price.
            let mac = try #require(outcome.envelope.result)
            #expect(Set(mac.keys) == Set(result.keys))
            #expect(mac["pinsComplete"] == .bool(true))
            if case .object(let estimate)? = mac["estimate"] {
                let pythonEstimate = try #require(result["estimate"] as? [String: Any])
                #expect(estimate["responsesAllowed"]
                    == .number(Double(pythonEstimate["responsesAllowed"] as? Int ?? -1)))
            } else {
                Issue.record("no estimate in the Swift echo")
            }
        }
    }

    // MARK: - The block

    @Test func theDeclaredBlockIsWhatTheDataCheckAndVerifyRead() async throws {
        try await withTempRoot { _ in
            try draft()
            let outcome = await invoke(declare([
                "--layers", "20,26", "--top-k", "10", "--watchlist", "5,9",
                "--tokenizer-hash", Self.tokenizer, "--qualification", "q1",
                "--keep-tokens",
            ]))
            #expect(outcome.envelope.state == .ready)
            #expect(outcome.envelope.exitCode == 0)
            #expect(outcome.envelope.changed)
            let written = try #require(try block())
            #expect(written["lensID"] == .string("lens-a"))
            #expect(written["layers"] == .array([.number(20), .number(26)]))
            #expect(written["logitLensCompanion"] == .bool(true))
            #expect(written["topKLayers"] == nil && written["budget"] == nil)
            #expect(!written.values.contains(.null))
            let manifest = try ExperimentStore.load(name: "demo")
            #expect(ExperimentStore.jlensReadoutViolations(manifest).isEmpty)
            let row = StudyDataReadiness.jlensReadoutRequirement(manifest: manifest)
            #expect(row.status == .present)
        }
    }

    @Test func withoutATokenizerHashTheHashPinsAreLeftOut() async throws {
        try await withTempRoot { _ in
            try draft()
            let outcome = await invoke(declare(["--layers", "20", "--top-k", "3"]))
            #expect(outcome.envelope.state == .ready)
            let written = try #require(try block())
            #expect(written["tokenizerHash"] == nil && written["configHash"] == nil)
            #expect(outcome.envelope.result?["pinsComplete"] == .bool(false))
        }
    }

    // MARK: - The budget

    /// At a 2,048-token cap, top-50 at 8 layers with the companion costs
    /// 32,768 projections and about 41 MB of trace per response, so the
    /// default ceilings cover 6 responses.
    @Test func theWorkedExampleIsPricedBeforeAnythingRuns() async throws {
        try await withTempRoot { _ in
            try draft()
            let outcome = await invoke(declare(["--layers", "20-27", "--top-k", "50", "--dry-run"]))
            #expect(outcome.envelope.state == .ready)
            #expect(!outcome.envelope.changed)
            #expect(try block() == nil)
            guard case .object(let estimate)? = outcome.envelope.result?["estimate"],
                case .object(let per)? = estimate["perResponse"]
            else {
                Issue.record("no estimate")
                return
            }
            #expect(per["fullVocabProjections"] == .number(32_768))
            #expect(per["observations"] == .number(16_384))
            #expect(per["projectedTraceBytes"] == .number(41_287_680))
            #expect(estimate["responsesAllowed"] == .number(6))
            #expect(estimate["limitedBy"] == .string("maxFullVocabProjections"))
        }
    }

    @Test func aReadoutThatCannotFitOneResponseIsRefused() async throws {
        try await withTempRoot { _ in
            try draft()
            let outcome = await invoke(declare(["--layers", "0-11", "--top-k", "5"]))
            #expect(outcome.envelope.exitCode == 64)
            #expect(outcome.envelope.error?.reason.contains("12 armed layers exceeds 8") == true)
            #expect(outcome.envelope.error?.repairAction.contains("--max-armed-layers") == true)
            #expect(try block() == nil)
            let raised = await invoke(declare([
                "--layers", "0-11", "--top-k", "5", "--max-armed-layers", "12",
                "--max-full-vocab-projections", "1000000",
            ]))
            #expect(raised.envelope.state == .ready)
        }
    }

    // MARK: - Refusals

    static let sixtyFiveTokens: String = (0..<65).map { String($0) }.joined(separator: ",")
    static let malformedCases: [([String], String)] = [
        (["--top-k", "3"], "needs --layers"),
        (["--layers", "20"], "neither a --watchlist nor a --top-k"),
        (["--layers", "20", "--top-k", "51"], "--top-k must be a whole number"),
        (["--layers", "26-20", "--top-k", "3"], "runs backwards"),
        (["--layers", "20", "--watchlist", "5,5"], "names token 5 twice"),
        (["--layers", "20", "--watchlist", sixtyFiveTokens], "watches at most 64"),
        (["--layers", "20", "--top-k", "3", "--top-k-layers", "21"], "--layers does not arm"),
        (["--layers", "70", "--top-k", "3"], "layer 70 is not a fitted source layer"),
        (["--layers", "20", "--top-k", "3", "--tokenizer-hash", "ABC"], "SHA-256 digest"),
    ]

    @Test(arguments: malformedCases)
    func outOfRangeValuesAreMalformed(args: [String], fragment: String) async throws {
        try await withTempRoot { _ in
            try draft()
            let outcome = await invoke(declare(args))
            #expect(outcome.envelope.exitCode == 64)
            #expect(outcome.envelope.state == .blocked)
            #expect(outcome.envelope.error?.reason.contains(fragment) == true,
                    "\(outcome.envelope.error?.reason ?? "")")
            #expect(outcome.envelope.error?.repairAction.isEmpty == false)
            #expect(try block() == nil)
        }
    }

    @Test func theLensPinsAreRequiredAndNamedInTheRepair() async throws {
        try await withTempRoot { _ in
            try draft()
            let outcome = await invoke(["set-jlens-readout", "demo", "lens-a", "--layers", "20", "--top-k", "3"])
            #expect(outcome.envelope.exitCode == 64)
            #expect(outcome.envelope.error?.reason.contains("--lens-sha256, --lens-model, --fitted-layers") == true)
            #expect(outcome.envelope.error?.repairAction.contains("--dry-run --json") == true)
        }
    }

    @Test func aLensForAnotherModelOrRevisionIsRefused() async throws {
        try await withTempRoot { _ in
            try draft()
            let other = await invoke(declare(["--layers", "20", "--top-k", "3"], lensModel: "org/other-model"))
            #expect(other.envelope.exitCode == 65)
            #expect(other.envelope.error?.gate == LifecycleGate.artifactPin.rawValue)
            let revision = await invoke(declare(
                ["--layers", "20", "--top-k", "3"],
                lensModel: Self.model + "@" + String(repeating: "0", count: 40)))
            #expect(revision.envelope.error?.gate == LifecycleGate.artifactPin.rawValue)
            #expect(revision.envelope.error?.repairAction.contains("pin-revision") == true)
            #expect(try block() == nil)
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
            let outcome = await invoke(declare(["--layers", "20", "--top-k", "3"]))
            #expect(outcome.envelope.exitCode == 65)
            #expect(outcome.envelope.error?.gate == LifecycleGate.statusImmutable.rawValue)
            #expect(outcome.envelope.error?.repairAction.contains("duplicate demo") == true)
            let dry = await invoke(declare(["--layers", "20", "--top-k", "3", "--dry-run"]))
            #expect(dry.envelope.error?.gate == LifecycleGate.statusImmutable.rawValue)
        }
    }

    // MARK: - Clearing

    @Test func theEmptyLensIDClearsAndKeepsTokenRetention() async throws {
        try await withTempRoot { _ in
            try draft()
            _ = await invoke(declare(["--layers", "20", "--top-k", "3", "--keep-tokens"]))
            let cleared = await invoke(["set-jlens-readout", "demo", ""])
            #expect(cleared.envelope.state == .ready)
            #expect(cleared.envelope.changed)
            #expect(cleared.envelope.result?["jlensReadout"] == .null)
            let manifest = try ExperimentStore.load(name: "demo")
            #expect(manifest.jlensReadout == nil)
            #expect(manifest.recordTokenIDs)
            let again = await invoke(["set-jlens-readout", "demo", ""])
            #expect(!again.envelope.changed)
            let withSettings = await invoke(["set-jlens-readout", "demo", "", "--layers", "20"])
            #expect(withSettings.envelope.exitCode == 64)
            #expect(withSettings.envelope.error?.reason.contains("a clear takes no settings") == true)
        }
    }

    // MARK: - The canonical serializer

    /// The configuration hash is SHA-256 over Python's `json.dumps(…,
    /// sort_keys=True, separators=(",", ":"))`, ensure_ascii included.
    @Test func theCanonicalSerializerSpellsWhatPythonSpells() {
        let value: JSONValue = .object([
            "b": .array([.number(1), .number(2_147_483_648), .bool(false), .null]),
            "a": .string("q\u{2014}\"\\\n/\u{7F}\u{1F600}"),
        ])
        // Python's escapes spelled out: a backslash, then `u` and four hex
        // digits, one per UTF-16 unit outside printable ASCII.
        let escape = { (hex: String) in "\\" + "u" + hex }
        let expected = #"{"a":"q"# + escape("2014") + #"\"\\\n/"# + escape("007f")
            + escape("d83d") + escape("de00") + #"","b":[1,2147483648,false,null]}"#
        #expect(JLensReadoutAuthoring.pythonJSON(value) == expected)
        #expect(JLensReadoutAuthoring.compactLayers([5, 0, 1, 2, 7, 8]) == "0-2,5,7-8")
    }
}
