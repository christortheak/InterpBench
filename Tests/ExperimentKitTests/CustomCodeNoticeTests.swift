import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// Custom code a shared study carries is noticed, acknowledged, and recorded
/// on the Mac exactly as on the Python client (twin: `test_custom_code.py`):
/// pack apply and attach-agent return the notice with the provider's source
/// hash; `experiment acknowledge-custom-code` shows the code and records who
/// acknowledged which hash and when; a step that would execute unacknowledged
/// code is not sent to run, and nothing else is held.
@MainActor
struct CustomCodeNoticeTests {
    static let source = "def decide(context, tensor, scores, state, rng, assets):\n    return [Decision('change', 1)]\n"
    static let digest = sha(source)

    // Cross-engine literals (Python twin: test_custom_code.py).
    static let fileNameLiteral = "custom-code-acknowledgements.json"
    static let noticeLiteral = "This study contains custom code from its author. It runs with your permissions when the study runs. Run it only if you trust the source."
    static let executingLiteral = ["pipeline", "run", "sweep"]

    static func sha(_ text: String) -> String {
        SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    static func policy(source: String? = source, declared: String? = nil) throws -> JSONValue {
        var document: [String: JSONValue] = [
            "schemaVersion": .number(1), "name": .string("expert-policy"),
            "binding": .object([
                "modelID": .string("test/model"), "revision": .string(String(repeating: "0", count: 40)),
                "tokenizerSHA256": .null, "rendering": .string("chatTemplate"),
                "coordinateConvention": .string("hf-decoder-block-v1/model.layers"),
            ]),
            "site": .object(["kind": .string("residualPost"), "layer": .number(0)]),
            "stages": .array([.string("decode")]), "positions": .string("lastPosition"),
            "probes": .array([]),
            "actions": .array([.object(["id": .string("change"), "kind": .string("add"),
                                        "bounds": .array([.number(0), .number(2)]),
                                        "vector": .array([.number(1), .number(0)])])]),
            "rules": .array([]), "onError": .string("stop"), "maxEvents": .number(8),
        ]
        if let source {
            document["provider"] = .object([
                "sourceText": .string(source), "sourceSHA256": .string(declared ?? sha(source)),
                "assets": .object([:]),
            ])
        }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let text = String(decoding: try encoder.encode(JSONValue.object(document)), as: UTF8.self)
        return .object(["json": .string(text), "sha256": .string(sha(text))])
    }

    /// A temp workspace holding a draft `study` and an agent whose policy
    /// carries an expert provider, on disk under runs/.
    private func withWorkspace(_ body: (URL, AgentArtifactSnapshot) throws -> Void) throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "custom-code") { root in
            _ = try ExperimentStore.create(name: "study", description: "Shared", modelID: "test/model")
            let path = "runs/model-variants/expert-agent/model-variant.json"
            let url = root.appending(path: path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            let artifact = ModelVariantArtifact(
                name: "expert-agent", baseModelID: "test/model", promptMode: "chatAssistant",
                qwenThinkingEnabled: false, temperature: 0, systemPrompt: "",
                interventionPolicies: [try Self.policy()])
            try JSONEncoder().encode(artifact).write(to: url)
            try body(root, try AgentArtifactSnapshot(workspaceRoot: root, path: path))
        }
    }

    private func attach(_ root: URL, _ agent: AgentArtifactSnapshot, study: String = "study") throws -> ExperimentCLIResult {
        let reviewed = try DraftAuthoringSnapshot(workspaceRoot: root, name: study)
        let invocation = try ExperimentCLIParser.parse(namespace: "experiment", [
            "attach-agent", study, "--artifact", agent.path, "--artifact-sha256", agent.file.sha256,
            "--manifest-sha256", reviewed.file.sha256, "--json"])
        return try StudyAgentCLI.attach(invocation, workspaceRoot: root, sink: .discarding)
    }

    private func acknowledgeVerb(_ root: URL, _ extra: [String] = []) throws -> ExperimentCLIResult {
        let invocation = try ExperimentCLIParser.parse(
            namespace: "experiment", ["acknowledge-custom-code", "study"] + extra + ["--json"])
        return try CustomCodeCLI.acknowledge(invocation, workspaceRoot: root, sink: .discarding)
    }

    private func object(_ value: JSONValue?) -> [String: JSONValue] {
        guard case .object(let object) = value else { return [:] }
        return object
    }

    // MARK: The owner

    @Test func crossEngineLiteralsAreThePythonOnes() {
        #expect(CustomCodeNotice.fileName == Self.fileNameLiteral)
        #expect(CustomCodeNotice.notice == Self.noticeLiteral)
        #expect(CustomCodeNotice.executingVerbs == Self.executingLiteral)
    }

    @Test func providersAreFoundInsideAttachedPolicyBytesAndNamedByTheirCode() throws {
        let plain = JSONValue.object(["interventionPolicies": .array([try Self.policy(source: nil)])])
        let expert = JSONValue.object(["interventionPolicies": .array([try Self.policy()])])
        let document = JSONValue.object(["variantConditions": .array([
            .object(["artifact": expert]), .object(["artifact": plain])])])
        let found = CustomCodeNotice.providers(in: document)
        #expect(found.map(\.sha256) == [Self.digest])
        #expect(found.first?.policyNames == ["expert-policy"])
        #expect(found.first?.sourceText == Self.source)
        #expect(CustomCodeNotice.providers(in: plain).isEmpty)
        // The hash is the code's own, never the document's claim about it.
        let claimed = JSONValue.object(["interventionPolicies": .array([try Self.policy(declared: String(repeating: "f", count: 64))])])
        #expect(CustomCodeNotice.providers(in: claimed).map(\.sha256) == [Self.digest])
    }

    @Test func aPanelSeatsAgentIsFoundThroughItsReferences() throws {
        try withWorkspace { root, agent in
            let panel = root.appending(path: "prompts/panels/compiled/panel.json")
            try FileManager.default.createDirectory(at: panel.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONEncoder().encode(JSONValue.object(["agents": .array([.object([
                "id": .string("a"), "variantArtifactPath": .string(agent.path)])])])).write(to: panel)
            let document = JSONValue.object([
                "multiAgentScenarioPath": .string("prompts/panels/compiled/panel.json"),
                "variantConditions": .array([.object(["artifactPath": .string("../outside.json")])]),
            ])
            #expect(CustomCodeNotice.providers(in: document).isEmpty)
            #expect(CustomCodeNotice.providers(in: document, workspaceRoot: root).map(\.sha256) == [Self.digest])
        }
    }

    @Test func acknowledgeRecordsWhoWhichHashAndWhenOnce() throws {
        try withWorkspace { root, _ in
            let document = JSONValue.object(["variantConditions": .array([.object(["artifact": .object([
                "interventionPolicies": .array([try Self.policy()])])])])])
            let first = try CustomCodeNotice.acknowledge(
                [Self.digest], in: document, study: "s", workspaceRoot: root, client: "steerlab-cli",
                account: "researcher", now: "2026-10-05T00:00:00Z")
            let expected = JSONValue.object([
                "providerSHA256": .string(Self.digest), "acknowledgedAt": .string("2026-10-05T00:00:00Z"),
                "acknowledgedBy": .string("researcher"), "client": .string("steerlab-cli"),
                "study": .string("s"), "policyNames": .array([.string("expert-policy")]),
            ])
            #expect(first.added == [expected])
            let url = root.appending(component: Self.fileNameLiteral)
            let record = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: url))
            #expect(record == .object(["schemaVersion": .number(1), "acknowledgements": .array([expected])]))
            let again = try CustomCodeNotice.acknowledge(
                [Self.digest], in: document, study: "other", workspaceRoot: root, client: "steerlab-cli")
            #expect(again.added.isEmpty && again.alreadyAcknowledged == [Self.digest])
            #expect(try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: url)) == record)
            let review = try #require(try CustomCodeNotice.review(of: document, study: "s", workspaceRoot: root))
            #expect(!review.needsAcknowledgement)
            #expect(object(review.block)["notice"] == .null)
            #expect(review.providers.first?.acknowledgedBy == "researcher")
            // A hash the study does not carry is refused.
            #expect(throws: ExperimentError.self) {
                try CustomCodeNotice.acknowledge(
                    [String(repeating: "a", count: 64)], in: document, study: "s", workspaceRoot: root,
                    client: "steerlab-cli")
            }
        }
    }

    @Test func onlyStepsThatExecuteAgentsAreHeldAndADamagedRecordRefuses() throws {
        try withWorkspace { root, _ in
            let document = JSONValue.object(["variantConditions": .array([.object(["artifact": .object([
                "interventionPolicies": .array([try Self.policy()])])])])])
            for verb in ["extract", "validate", "evaluate", "analyze", "verify"] {
                #expect(try CustomCodeNotice.runRefusal(document: document, study: "s", verb: verb, workspaceRoot: root) == nil)
            }
            for verb in Self.executingLiteral + [nil] as [String?] {
                let refusal = try #require(try CustomCodeNotice.runRefusal(
                    document: document, study: "s", verb: verb, workspaceRoot: root))
                #expect(refusal.reason.contains(Self.digest) && refusal.reason.contains(Self.noticeLiteral))
                #expect(refusal.lifecycleRefusal?.gate == .missingPrerequisite)
                #expect(refusal.lifecycleRefusal?.repairAction.contains("--sha256 \(Self.digest)") == true)
            }
            try Data("{not json".utf8).write(to: root.appending(component: Self.fileNameLiteral))
            #expect(throws: ExperimentError.self) {
                try CustomCodeNotice.runRefusal(document: document, study: "s", verb: "run", workspaceRoot: root)
            }
            // A study with no custom code is never held up by the record.
            #expect(try CustomCodeNotice.runRefusal(document: .object([:]), study: "s", verb: "run", workspaceRoot: root) == nil)
        }
    }

    // MARK: The command line

    @Test func attachAgentAndPackApplyGiveTheNoticeUntilAcknowledged() throws {
        try withWorkspace { root, agent in
            let attached = try attach(root, agent)
            let block = object(attached.payload["customCode"])
            #expect(block["notice"] == .string(Self.noticeLiteral))
            #expect(block["acknowledgeCommand"] == .string(
                "steerlab-cli experiment acknowledge-custom-code study --sha256 \(Self.digest)"))
            guard case .array(let rows) = block["providers"] else {
                Issue.record("expected provider rows")
                return
            }
            #expect(object(rows.first)["sha256"] == .string(Self.digest))

            // A shared pack carrying the same agent.
            let exported = try StudyPackAuthoring.export(reviewed: DraftAuthoringSnapshot(workspaceRoot: root, name: "study"))
            var pack = try JSONDecoder().decode(JSONValue.self, from: exported.data)
            func renamed(_ name: String) throws -> URL {
                guard case .object(var top) = pack, case .object(var study) = top["study"] else { throw CancellationError() }
                study["name"] = .string(name)
                top["study"] = .object(study)
                pack = .object(top)
                let url = root.appending(component: "\(name).pack.json")
                try JSONEncoder().encode(pack).write(to: url)
                return url
            }
            func apply(_ url: URL) throws -> ExperimentCLIResult {
                let review = try StudyPackAuthoring.preview(Data(contentsOf: url), workspaceRoot: root)
                return try StudyPackCLI.run(
                    ExperimentCLIParser.parse(namespace: "pack", ["apply", url.path, "--review-sha256", review.reviewSHA256, "--json"]),
                    workspaceRoot: root, sink: .discarding)
            }
            let applied = try apply(try renamed("shared-copy"))
            #expect(object(applied.payload["customCode"])["notice"] == .string(Self.noticeLiteral))

            // Review mode shows the code and writes nothing.
            let reviewed = try acknowledgeVerb(root)
            #expect(!reviewed.changed)
            guard case .array(let shown) = reviewed.payload["providers"] else {
                Issue.record("expected provider rows")
                return
            }
            #expect(object(shown.first)["sourceText"] == .string(Self.source))
            let record = root.appending(component: Self.fileNameLiteral)
            #expect(!FileManager.default.fileExists(atPath: record.path))

            let recorded = try acknowledgeVerb(root, ["--sha256", Self.digest])
            #expect(recorded.changed)
            guard case .array(let added) = recorded.payload["acknowledged"] else {
                Issue.record("expected acknowledgement records")
                return
            }
            #expect(object(added.first)["client"] == .string("steerlab-cli"))
            #expect(throws: ExperimentError.self) { try acknowledgeVerb(root, ["--sha256", String(repeating: "b", count: 64)]) }

            // The same code arriving again is not noticed again.
            let again = try apply(try renamed("third-copy"))
            #expect(object(again.payload["customCode"])["notice"] == .null)
            #expect(object(again.payload["customCode"])["acknowledged"] == .bool(true))
        }
    }

    // MARK: The app

    @Test func theStudyPageAcknowledgesExactlyWhatItShowed() throws {
        try withWorkspace { root, agent in
            _ = try attach(root, agent)
            let panel = ExperimentPanel()
            let shown = panel.customCodeState(for: "study", workspaceRoot: root)
            let review = try #require(shown.review)
            #expect(review.needsAcknowledgement && shown.problem == nil)
            #expect(review.pending.map(\.sha256) == [Self.digest])
            #expect(panel.acknowledgeCustomCode(review, workspaceRoot: root))
            let after = panel.customCodeState(for: "study", workspaceRoot: root)
            #expect(after.review?.needsAcknowledgement == false)
            #expect(after.review?.providers.first?.acknowledgedAt != nil)
            let record = try JSONDecoder().decode(
                JSONValue.self, from: Data(contentsOf: root.appending(component: Self.fileNameLiteral)))
            guard case .object(let top) = record, case .array(let entries) = top["acknowledgements"] else {
                Issue.record("expected a record")
                return
            }
            #expect(object(entries.first)["client"] == .string("SteerLab app"))
            // A study with no custom code shows nothing.
            _ = try ExperimentStore.create(name: "plain", description: "", modelID: "test/model")
            #expect(panel.customCodeState(for: "plain", workspaceRoot: root).review == nil)
            // A damaged record still shows the notice and names the problem.
            try Data("{".utf8).write(to: root.appending(component: Self.fileNameLiteral))
            let damaged = panel.customCodeState(for: "study", workspaceRoot: root)
            #expect(damaged.review?.needsAcknowledgement == true)
            #expect(damaged.problem?.contains("cannot be read") == true)
        }
    }

    // MARK: The run gate on the app's submission path

    @Test func bundleSubmissionRefusesUnacknowledgedCodeBeforePackaging() async throws {
        ExperimentRootOverrideLock.acquire()
        let root = FileManager.default.temporaryDirectory.appending(component: "custom-code-submit-\(UUID())")
        ExperimentStore.rootOverride = root
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: root)
            ExperimentRootOverrideLock.release()
        }
        do {
            _ = try ExperimentStore.create(name: "study", description: "Shared", modelID: "test/model")
            let path = "runs/model-variants/expert-agent/model-variant.json"
            let url = root.appending(path: path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            let artifact = ModelVariantArtifact(
                name: "expert-agent", baseModelID: "test/model", promptMode: "chatAssistant",
                qwenThinkingEnabled: false, temperature: 0, systemPrompt: "",
                interventionPolicies: [try Self.policy()])
            try JSONEncoder().encode(artifact).write(to: url)
            _ = try attach(root, try AgentArtifactSnapshot(workspaceRoot: root, path: path))
            let manifest = try ExperimentStore.load(name: "study")

            var packaged = 0
            var io = StudyBundleTransport(
                frozenConflict: { _ in nil },
                package: { _ in packaged += 1; return URL(filePath: "/private/tmp/fake-study.tar.gz") },
                upload: { _ in "uploaded/study.tar.gz" },
                submit: { _, _ in
                    .init(jobId: "job", experiment: "study", verb: "run", executor: "local", dryRun: false,
                          runBundle: [:], slurmBundle: nil, slurmJobID: nil, command: [],
                          recordsDirectory: "records", submissionDirectory: "submission")
                })
            io.origin = RemoteJobOrigin(
                connection: ClusterConnectionProfile(baseURL: URL(string: "http://127.0.0.1:9")!),
                workspaceRoot: root)
            let jobs = StudyRemoteJobController()
            let options = StudySubmissionOptions()
            options.remoteVerb = "run"
            let refused = await StudyBundleSubmissionController(jobs: jobs).submit(
                manifest, request: options.snapshot, capabilities: nil, substrate: nil, transport: io)
            guard case .failure(let failure) = refused else {
                Issue.record("must refuse")
                return
            }
            #expect(failure.reason.contains(Self.digest))
            #expect(packaged == 0)

            // A step that runs no agent is not held.
            options.remoteVerb = "validate"
            _ = await StudyBundleSubmissionController(jobs: jobs).submit(
                manifest, request: options.snapshot, capabilities: nil, substrate: nil, transport: io)
            #expect(packaged == 1)

            let document = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: ExperimentStore.manifestURL("study")))
            try CustomCodeNotice.acknowledge([Self.digest], in: document, study: "study", workspaceRoot: root, client: "SteerLab app")
            options.remoteVerb = "run"
            let accepted = await StudyBundleSubmissionController(jobs: jobs).submit(
                manifest, request: options.snapshot, capabilities: nil, substrate: nil, transport: io)
            guard case .success = accepted else {
                Issue.record("an acknowledged study is sent")
                return
            }
            #expect(packaged == 2)
        }
    }

    /// The app's diagnostic sheet reads the plan's `customCode` block: the
    /// notice, the providers still to acknowledge, and the value that
    /// acknowledges them when the inputs are packaged.
    @Test func aDiagnosticPlanNamesTheCodeStillToAcknowledge() throws {
        func row(_ sha256: String, acknowledged: Bool) -> JSONValue {
            .object(["sha256": .string(sha256), "policyNames": .array([.string("expert")]),
                     "sourceText": .string("def decide(context):\n    return []\n"), "acknowledged": .bool(acknowledged)])
        }
        let plan: JSONValue = .object(["planSHA256": .string("p"), "customCode": .object([
            "notice": .string("This diagnostic's inputs contain custom code."),
            "providers": .array([row("a", acknowledged: false), row("b", acknowledged: true), row("c", acknowledged: false)])])])
        let review = try #require(CustomCodeNotice.DiagnosticReview(plan: plan))
        #expect(review.notice == "This diagnostic's inputs contain custom code.")
        #expect(review.pending.map(\.sha256) == ["a", "c"] && review.acknowledgement == "a,c")
        #expect(review.pending.first?.policyNames == ["expert"])
        #expect(CustomCodeNotice.DiagnosticReview(plan: .object(["planSHA256": .string("p")])) == nil)
        // Once everything is acknowledged the plan's notice is null: nothing to ask.
        let done: JSONValue = .object(["customCode": .object(["notice": .null, "providers": .array([row("a", acknowledged: true)])])])
        #expect(CustomCodeNotice.DiagnosticReview(plan: done) == nil)
    }
}
