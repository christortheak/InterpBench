import CryptoKit
import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// `steerlab-cli agent create` (public issue #17) and an agent's thinking
/// level (public issue #16). Python twin: `Server/tests/test_agent_create.py`.
struct StudyAgentCreationTests {
    static let model = "test/model"
    static let revision = String(repeating: "a", count: 40)

    private func sha(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// A vector pair: the tensor file is only hashed, and the sidecar carries
    /// what the checks read.
    @discardableResult
    private func vector(
        _ root: URL, name: String = "formality", model: String = Self.model,
        revision: String? = Self.revision, norms: Bool = true
    ) throws -> (reference: String, sha256: String) {
        let run = root.appending(components: "runs", "20261108-extract")
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        let tensor = Data("tensor-\(name)".utf8)
        try tensor.write(to: run.appending(component: "\(name).safetensors"))
        var sidecar: [String: Any] = [
            "modelID": model, "concept": name, "stimulusSetHash": "h", "layerCount": 4,
            "hiddenSize": 2, "normsPerLayer": [1.0, 0.0, 3.0, 4.0], "extractionDate": "2026-11-08",
        ]
        if let revision { sidecar["revision"] = revision }
        if norms { sidecar["residualNormPerLayer"] = [10.0, 10.0, 10.0, 10.0] }
        try JSONSerialization.data(withJSONObject: sidecar)
            .write(to: run.appending(component: "\(name).json"))
        return ("runs/20261108-extract/\(name)", sha(tensor))
    }

    private func adapter(_ root: URL, model: String = Self.model) throws -> (String, String) {
        let run = root.appending(components: "runs", "20261108-lora-tone")
        let folder = run.appending(component: "tone")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let weights = Data("weights".utf8)
        let config = Data(#"{"peft_type": "LORA"}"#.utf8)
        try weights.write(to: folder.appending(component: "adapter_model.safetensors"))
        try config.write(to: folder.appending(component: "adapter_config.json"))
        try JSONSerialization.data(withJSONObject: [
            "name": "tone", "baseModelID": model, "revision": Self.revision,
            "adapterFormat": "hf-peft-lora", "adapterBytesHash": sha(weights),
            "adapterConfigHash": sha(config),
        ]).write(to: run.appending(component: "tone.json"))
        return ("runs/20261108-lora-tone/tone", sha(weights))
    }

    private func base(_ name: String = "steered", model: String = Self.model) -> [String] {
        ["create", name, "--model", model, "--revision", Self.revision]
    }

    private func run(_ root: URL, _ args: [String]) throws -> ExperimentCLIResult {
        try StudyAgentCreation.run(
            ExperimentCLIParser.parse(namespace: "agent", args + ["--json"]),
            workspaceRoot: root, sink: .discarding)
    }

    private func create(_ root: URL, _ args: [String]) throws -> ExperimentCLIResult {
        let preview = try run(root, args)
        guard case .string(let digest)? = preview.payload["reviewSHA256"] else {
            Issue.record("no reviewSHA256")
            return preview
        }
        return try run(root, args + ["--review-sha256", digest])
    }

    private func refusal(_ root: URL, _ args: [String]) -> LifecycleRefusal? {
        do {
            _ = try run(root, args)
            Issue.record("expected a refusal")
            return nil
        } catch let error as ExperimentError {
            return error.lifecycleRefusal
        } catch {
            Issue.record("unexpected \(error)")
            return nil
        }
    }

    private func libraryAgents(_ root: URL) -> [URL] {
        let library = root.appending(components: "runs", "model-variants")
        let folders = (try? FileManager.default.contentsOfDirectory(
            at: library, includingPropertiesForKeys: nil)) ?? []
        return folders.map { $0.appending(component: "model-variant.json") }
            .filter { FileManager.default.fileExists(atPath: $0.path) }
    }

    // MARK: - Preview, then the reviewed write

    @Test func previewWritesNothingAndTheReviewedWriteMakesALibraryAgent() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "agent-create") { root in
            let (reference, vectorSHA) = try vector(root)
            let (adapterPath, adapterSHA) = try adapter(root)
            let prompt = root.appending(component: "persona.txt")
            try Data("Answer as a careful reviewer.".utf8).write(to: prompt)
            let args = base() + [
                "--adapter", adapterPath, "--adapter-sha256", adapterSHA,
                "--vector", reference, "--vector-sha256", vectorSHA, "--layer", "2",
                "--strength", "0.12", "--system-prompt-file", prompt.path,
                "--reason", "the published setting from a prior study",
            ]
            let preview = try run(root, args)
            #expect(preview.changed == false)
            #expect(preview.payload["written"] == .bool(false))
            #expect(libraryAgents(root).isEmpty)
            guard case .string(let digest)? = preview.payload["reviewSHA256"] else {
                Issue.record("no digest"); return
            }
            let written = try run(root, args + ["--review-sha256", digest])
            #expect(written.changed)
            let files = libraryAgents(root)
            #expect(files.count == 1)
            let data = try Data(contentsOf: try #require(files.first))
            #expect(written.payload["artifactFileSHA256"] == .string(sha(data)))
            let artifact = try JSONDecoder().decode(ModelVariantArtifact.self, from: data)
            #expect(artifact.baseRevision == Self.revision)
            #expect(artifact.alphaInNormUnits && artifact.bandWidth == 1)
            #expect(artifact.injections == [.init(
                concept: "formality", vectorArtifactID: reference, layer: 2, alpha: 0.12)])
            #expect(artifact.adapters.first?.adapterHash == adapterSHA)
            #expect(artifact.systemPromptHash == sha(Data("Answer as a careful reviewer.".utf8)))
            #expect(artifact.reasoningEffort == nil && artifact.qwenThinkingEnabled == false)
            #expect(artifact.promotion == nil)
            #expect(artifact.kind == .handmade)
            guard case .object(let record)? = artifact.handmade else {
                Issue.record("no handmade record"); return
            }
            #expect(record["madeWith"] == .string("agent create"))
            #expect(record["reason"] == .string("the published setting from a prior study"))
            #expect(record["createdAt"] == .string(artifact.createdAt))
            guard case .array(let command)? = record["command"] else {
                Issue.record("no command"); return
            }
            #expect(Array(command.prefix(3)) == [.string("steerlab-cli"), .string("agent"), .string("create")])
            #expect(!command.contains(.string("--review-sha256")))
            // The library agent is one every client lists.
            let listed = try StudyAgentAuthoring.list(workspaceRoot: root)
            #expect(listed.agents.map(\.agent.kind) == [.handmade])
        }
    }

    @Test func aStaleReviewDigestRefusesAndWritesNothing() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "agent-create") { root in
            let (reference, vectorSHA) = try vector(root)
            let vectorArgs = ["--vector", reference, "--vector-sha256", vectorSHA, "--layer", "2"]
            let preview = try run(root, base() + vectorArgs + ["--strength", "0.1"])
            guard case .string(let stale)? = preview.payload["reviewSHA256"] else { return }
            let refused = refusal(root, base() + vectorArgs + ["--strength", "0.2",
                                                               "--review-sha256", stale])
            #expect(refused?.gate == .staleManifest)
            #expect(refused?.repairAction.contains("Preview again") == true)
            #expect(libraryAgents(root).isEmpty)
        }
    }

    // MARK: - Each check refuses with its repair

    @Test(arguments: [
        "revision", "vectorHash", "vectorModel", "vectorRevision", "layer", "emptyLayer",
        "norms", "adapterHash", "twoAdapters", "effort", "name",
    ])
    func eachCheckRefusesPlainlyWithARepair(_ check: String) throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "agent-create") { root in
            let (reference, vectorSHA) = try vector(root)
            let (adapterPath, adapterSHA) = try adapter(root)
            func vectorArgs(_ ref: String, _ digest: String, layer: String = "2") -> [String] {
                ["--vector", ref, "--vector-sha256", digest, "--layer", layer, "--strength", "0.1"]
            }
            var args = base() + vectorArgs(reference, vectorSHA)
            var reason = "", repair = ""
            switch check {
            case "revision":
                args = ["create", "steered", "--model", Self.model, "--revision", "main"]
                reason = "is not a pinned commit"; repair = "40-character commit hash"
            case "vectorHash":
                args = base() + vectorArgs(reference, String(repeating: "0", count: 64))
                reason = "not the one you reviewed"; repair = "inspect-artifact"
            case "vectorModel":
                let other = try vector(root, name: "other", model: "other/model")
                args = base() + vectorArgs(other.reference, other.sha256)
                reason = "was extracted from other/model"; repair = "base model"
            case "vectorRevision":
                let other = try vector(root, name: "other", revision: String(repeating: "b", count: 40))
                args = base() + vectorArgs(other.reference, other.sha256)
                reason = "revision " + String(repeating: "b", count: 40); repair = "--revision"
            case "layer":
                args = base() + vectorArgs(reference, vectorSHA, layer: "9")
                reason = "Layer 9 is not in the vector"; repair = "--layer"
            case "emptyLayer":
                args = base() + vectorArgs(reference, vectorSHA, layer: "1")
                reason = "is empty at layer 1"; repair = "normsPerLayer"
            case "norms":
                let bare = try vector(root, name: "bare", norms: false)
                args = base() + vectorArgs(bare.reference, bare.sha256)
                reason = "no residual norm at layer 2"; repair = "backfill-norms"
            case "adapterHash":
                args = base() + ["--adapter", adapterPath, "--adapter-sha256", String(repeating: "1", count: 64)]
                reason = "not the ones you reviewed"; repair = "shasum -a 256"
            case "twoAdapters":
                args = base() + ["--adapter", adapterPath, "--adapter-sha256", adapterSHA,
                                 "--adapter", adapterPath, "--adapter-sha256", adapterSHA]
                reason = "An agent runs one adapter"; repair = "Give one --adapter"
            case "effort":
                args = base() + ["--reasoning-effort", "medium"]
                reason = "has no thinking switch"; repair = "--reasoning-effort off"
            default:  // name
                _ = try create(root, base("taken"))
                args = base("taken")
                reason = "already exists"; repair = "agent list"
            }
            let refused = refusal(root, args)
            #expect(refused?.reason.contains(reason) == true, "\(String(describing: refused))")
            #expect(refused?.repairAction.contains(repair) == true, "\(String(describing: refused))")
            #expect(libraryAgents(root).count == (check == "name" ? 1 : 0))
        }
    }

    @Test func thereIsNoTemperatureFlag() {
        #expect(throws: (any Error).self) {
            _ = try ExperimentCLIParser.parse(namespace: "agent", base() + ["--temperature", "0.7"])
        }
        #expect(!StudyAgentCreation.spec.valueFlags.contains("--temperature"))
    }

    @Test func aLevelTheTemplateAcceptsIsWrittenWithItsShadow() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "agent-create") { root in
            let written = try create(
                root, base("thinker", model: "Qwen/Qwen3.8-27B") + ["--reasoning-effort", "medium"])
            let data = try Data(contentsOf: try #require(libraryAgents(root).first))
            let object = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
            #expect(object["reasoningEffort"] as? String == "medium")
            #expect(object["qwenThinkingEnabled"] as? Bool == true)
            guard case .object(let origin)? = written.payload["origin"] else { return }
            #expect(origin["reasoningEffort"] == .string("medium"))
            #expect(origin["reasoningEffortSource"] == .string("agent"))
        }
    }

    // MARK: - The thinking level on the agent (#16)

    private func decode(_ object: [String: Any]) throws -> ModelVariantArtifact {
        var full: [String: Any] = ["name": "a", "baseModelID": Self.model]
        full.merge(object) { _, new in new }
        return try JSONDecoder().decode(
            ModelVariantArtifact.self, from: JSONSerialization.data(withJSONObject: full))
    }

    @Test(arguments: [("medium", false), ("off", true)])
    func aFileWhoseEffortAndSwitchDisagreeIsRefused(_ effort: String, _ thinking: Bool) throws {
        // It still decodes (so a study holding it decodes, and its verify
        // names the problem); every reader that uses it refuses.
        let artifact = try decode(["reasoningEffort": effort, "qwenThinkingEnabled": thinking])
        #expect(artifact.fileProblem?.contains(
            "one says the agent thinks and the other says it does not") == true)
    }

    @Test func anUnknownEffortAndTwoRecordsAreRefused() throws {
        #expect(try decode(["reasoningEffort": "hgih"]).fileProblem?.contains(
            "reasoningEffort \"hgih\" is not a level") == true)
        // A non-string level gets the server's sentence, not a decode error.
        #expect(try decode(["reasoningEffort": 5]).fileProblem?.contains(
            "reasoningEffort 5 is not a level") == true)
        var promoted = try decode([:])
        promoted.promotion = .init(
            experiment: "s", experimentHash: "h", promotedAt: "t", promotedBy: "criterion",
            substrate: "swift-mlx", appVersion: "0")
        promoted.handmade = .object(["madeWith": .string("agent create")])
        let data = try JSONEncoder().encode(promoted)
        let read = try JSONDecoder().decode(ModelVariantArtifact.self, from: data)
        #expect(read.fileProblem?.contains("An agent is made one way") == true)
        // A non-object handmade is no record, on both engines.
        var odd = try decode([:])
        odd.handmade = .string("agent create")
        #expect(odd.kind == .handmade && odd.fileProblem == nil)
    }

    @Test func theThinkingFieldsReadAsTheServerReadsThem() throws {
        let url = CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cross-engine", "agent-reasoning-effort.json")
        let cases = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [[String: Any]])
        #expect(cases.count == 33)
        for entry in cases {
            let object = try #require(entry["artifact"] as? [String: Any])
            let artifact = try JSONDecoder().decode(
                ModelVariantArtifact.self, from: JSONSerialization.data(withJSONObject: object))
            let label = "\(object)"
            if let problem = entry["problem"] as? String {
                #expect(artifact.fileProblem == problem, "\(label)")
            } else {
                #expect(artifact.fileProblem == nil, "\(label)")
                #expect(artifact.resolvedReasoningEffort.rawValue
                    == entry["reasoningEffort"] as? String, "\(label)")
                #expect(artifact.qwenThinkingEnabled == entry["qwenThinkingEnabled"] as? Bool,
                    "\(label)")
            }
        }
    }

    @Test(arguments: [
        (#"{"qwenThinkingEnabled": false}"#, ReasoningEffort.off),
        (#"{"qwenThinkingEnabled": true}"#, .xhigh),
        ("{}", .off),
        (#"{"reasoningEffort": "medium", "qwenThinkingEnabled": true}"#, .medium),
        (#"{"reasoningEffort": "on"}"#, .on),
        (#"{"reasoningEffort": "off", "qwenThinkingEnabled": false}"#, .off),
    ])
    func theHelperReadsTheLevelOrTheLegacySwitch(_ json: String, _ expected: ReasoningEffort) throws {
        let object = try #require(
            try JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        let artifact = try decode(object)
        #expect(artifact.resolvedReasoningEffort == expected)
        #expect(artifact.qwenThinkingEnabled == expected.isOn)
        let again = try JSONDecoder().decode(
            ModelVariantArtifact.self, from: JSONEncoder().encode(artifact))
        #expect(again.resolvedReasoningEffort == expected)
    }

    @Test func aLegacyAgentEncodesWithoutANewKey() throws {
        let legacy = ModelVariantArtifact(
            name: "a", baseModelID: Self.model, promptMode: "chatAssistant",
            qwenThinkingEnabled: true, temperature: 0, systemPrompt: "")
        let object = try #require(try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(legacy)) as? [String: Any])
        #expect(object["reasoningEffort"] == nil && object["handmade"] == nil)
        // Off, picked in the editor, writes the boolean alone.
        let off = ModelVariantArtifact(
            name: "a", baseModelID: Self.model, promptMode: "chatAssistant",
            qwenThinkingEnabled: true, temperature: 0, systemPrompt: "", reasoningEffort: .off)
        #expect(off.reasoningEffort == nil && off.qwenThinkingEnabled == false)
    }

    @Test func anOrdinaryStudyRunsTheAgentAtItsOwnLevel() throws {
        var manifest = ExperimentManifest(name: "s", description: "", modelID: "Qwen/Qwen3.8-27B")
        manifest.reasoningEffort = "xhigh"
        manifest.reasoningMaxTokens = 256
        let agent = try decode(["reasoningEffort": "medium", "qwenThinkingEnabled": true])
        #expect(StudySampling.reasoningEffort(manifest, variant: agent) == .medium)
        #expect(StudySampling.reasoningEffort(manifest, variant: nil) == .xhigh)
        let legacy = try decode(["qwenThinkingEnabled": true])
        #expect(StudySampling.reasoningEffort(manifest, variant: legacy) == .xhigh)
    }

    @Test func theTemplateRuleIsTheStudyRule() throws {
        let qwen = try decode(["baseModelID": "Qwen/Qwen3.8-27B", "reasoningEffort": "medium"])
        #expect(qwen.effortViolations().isEmpty)
        let none = try decode(["reasoningEffort": "on"])
        #expect(none.effortViolations() == [
            ReasoningEffort.effortWithoutThinkingModeReason(.on, modelID: Self.model)])
    }

    @Test func theEditorOffersOnlyWhatTheTemplateAccepts() {
        let none = PromptRendering.capabilities(for: Self.model, explicit: nil)
        #expect(ModelVariantArtifact.offeredReasoningEfforts(
            capabilities: none, current: .off) == [.off])
        // A DECLARED level the template does not take is never offered…
        #expect(ModelVariantArtifact.offeredReasoningEfforts(
            capabilities: none, current: .xhigh) == [.off])
        // …while an older file's thinking switch shows as what it reads as,
        // so the agent saves unchanged.
        #expect(ModelVariantArtifact.offeredReasoningEfforts(
            capabilities: none, current: .xhigh, currentDeclared: false) == [.off, .xhigh])
        let qwen = PromptRendering.capabilities(for: "Qwen/Qwen3.8-27B", explicit: nil)
        let offered = ModelVariantArtifact.offeredReasoningEfforts(capabilities: qwen, current: .off)
        #expect(offered.first == .off && offered.contains(.on))
        for level in offered where level.isLevel {
            #expect([.accepted, .assumed].contains(qwen.effortVerdict(level.rawValue)))
        }
    }

    @Test func promotionKeepsTheStudysDeclaredLevel() {
        var declared = ExperimentManifest(name: "s", description: "", modelID: "m")
        declared.reasoningEffort = "medium"
        declared.reasoningMaxTokens = 64
        #expect(AgentPromotion.promotedReasoningEffort(declared) == .medium)
        // A manifest from before the effort existed spells only the boolean.
        var legacy = ExperimentManifest(name: "s", description: "", modelID: "m")
        legacy.reasoningEffort = nil
        legacy.qwenThinkingEnabled = true
        #expect(AgentPromotion.promotedReasoningEffort(legacy) == nil)
        var off = ExperimentManifest(name: "s", description: "", modelID: "m")
        off.reasoningEffort = "off"
        #expect(AgentPromotion.promotedReasoningEffort(off) == nil)
    }

    // MARK: - Swept and handmade, everywhere

    @Test func inspectAndAttachNameTheKind() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "agent-create") { root in
            let written = try create(root, base("plain"))
            guard case .string(let path)? = written.payload["path"],
                case .string(let artifactSHA)? = written.payload["artifactFileSHA256"]
            else { Issue.record("no path"); return }
            let inspected = try AgentArtifactDocument(AgentArtifactSnapshot(workspaceRoot: root, path: path))
            #expect(inspected.agent.kind == .handmade)
            #expect(inspected.agent.summary == "Made by hand with agent create. The researcher "
                + "chose its settings. No reason was stated.")
            #expect(inspected.agent.reasoningEffort == "off")
            #expect(inspected.agent.reasoningEffortSource == "thinkingSwitch")
            var swept = inspected.artifact
            swept.handmade = nil
            swept.promotion = .init(
                experiment: "screen", experimentHash: "h", promotedAt: "t",
                promotedBy: "criterion", substrate: "swift-mlx", appVersion: "0")
            #expect(swept.origin.summary == "A sweep chose this agent's settings: experiment "
                + "promote made it from study screen, using the cell the study's declared rule "
                + "selected.")
            // The Python client's sentence for the same file (cross-engine).
            _ = try ExperimentStore.create(name: "study", description: "d", modelID: Self.model)
            let study = try DraftAuthoringSnapshot(workspaceRoot: root, name: "study")
            let invocation = try ExperimentCLIParser.parse(namespace: "experiment", [
                "attach-agent", "study", "--artifact", path, "--artifact-sha256", artifactSHA,
                "--manifest-sha256", study.file.sha256, "--json"])
            let attached = try StudyAgentCLI.attach(invocation, workspaceRoot: root, sink: .discarding)
            guard case .object(let agent)? = attached.payload["agent"] else {
                Issue.record("no agent block"); return
            }
            #expect(agent["kind"] == .string("handmade"))
        }
    }

    // MARK: - Round 2

    @Test func aFileOutsideTheWorkspaceIsRecordedByNameAndHash() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "agent-create") { root in
            let outside = FileManager.default.temporaryDirectory
                .appending(component: "fr6b-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: outside) }
            let prompt = outside.appending(component: "persona.txt")
            try Data("Answer as a careful reviewer.".utf8).write(to: prompt)
            _ = try create(root, base() + ["--system-prompt-file", prompt.path])
            let data = try Data(contentsOf: try #require(libraryAgents(root).first))
            let text = String(decoding: data, as: UTF8.self)
            let homeRoot = "/" + "Users" + "/"  // spelled apart
            #expect(!text.contains(outside.path) && !text.contains(NSHomeDirectory())
                && !text.contains(homeRoot) && !text.contains("~/"))
            let artifact = try JSONDecoder().decode(ModelVariantArtifact.self, from: data)
            guard case .object(let record)? = artifact.handmade,
                case .object(let inputs)? = record["inputs"],
                case .object(let file)? = inputs["systemPromptFile"],
                case .array(let command)? = record["command"]
            else { Issue.record("no record"); return }
            #expect(file["name"] == .string("persona.txt") && file["path"] == nil)
            #expect(command.contains(.string("persona.txt")))
            #expect(inputs["temperature"] == .null)
            // A `~` path that cannot be read says so without the home folder.
            do {
                _ = try run(root, base("other") + ["--system-prompt-file", "~/fr6b-missing.txt"])
                Issue.record("expected a refusal")
            } catch let error as ExperimentError {
                #expect(!error.reason.contains(NSHomeDirectory()))
                #expect(error.reason.contains("fr6b-missing.txt"))
            }
        }
    }

    @Test func bandWidthCenteringAndNamesFollowTheEditor() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "agent-create") { root in
            let (reference, digest) = try vector(root)
            let preview = try run(root, base() + [
                "--vector", reference, "--vector-sha256", digest, "--layer", "2",
                "--strength", "1", "--mode", "ablate", "--band-width", "3"])
            guard case .object(let document)? = preview.payload["document"],
                case .array(let notes)? = preview.payload["advisories"]
            else { Issue.record("no document"); return }
            #expect(document["bandWidth"] == .number(3))
            #expect(notes.contains { $0.agentRecordString?.contains("ablated without centering") == true })
            let refused = refusal(root, base() + [
                "--vector", reference, "--vector-sha256", digest, "--layer", "2",
                "--strength", "1", "--mode", "ablate", "--centering", "neutralMean"])
            #expect(refused?.reason.contains("stores no neutral mean") == true)
            #expect(throws: ExperimentError.self) { _ = try run(root, base() + ["--band-width", "4"]) }
            _ = try create(root, base("taken"))
            #expect(refusal(root, base("TAKEN"))?.reason.contains("ignoring case") == true)
        }
    }

    @Test func anAdapterTrainedOnABranchIsReadByItsResolvedCommit() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "agent-create") { root in
            let (path, digest) = try adapter(root)
            let recordURL = root.appending(path: path + ".json")
            var record = try #require(
                try JSONSerialization.jsonObject(with: Data(contentsOf: recordURL)) as? [String: Any])
            record["revision"] = "main"
            record["revisionResolved"] = Self.revision
            try JSONSerialization.data(withJSONObject: record).write(to: recordURL)
            _ = try run(root, base() + ["--adapter", path, "--adapter-sha256", digest])
        }
    }

    @Test func theRepairsAreTrueForTheRecordInHand() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "agent-create") { root in
            let refused = refusal(root, ["create", "a", "--model", Self.model, "--revision", "main"])
            #expect(refused?.repairAction.contains("model plan \(Self.model) --json") == true)
            // The command the repair names is one this command line accepts.
            _ = try ExperimentCLIParser.parse(namespace: "model", ["plan", Self.model, "--json"])
        }
        let qwen = PromptRendering.capabilities(for: "Qwen/Qwen3.8-27B", explicit: nil)
        let repair = StudyAgentCreation.effortRepair(qwen)
        #expect(!repair.contains("accepts: "))
        #expect(repair.contains("assumed from the model id, with no probed template: low, medium, xhigh"))
    }

    private func study(
        _ root: URL, artifact: ModelVariantArtifact, instruments: [String]? = nil,
        status: ExperimentManifest.Status = .draft
    ) throws -> ExperimentManifest {
        var manifest = try ExperimentStore.create(
            name: "study", description: "", modelID: artifact.baseModelID,
            modelRevision: Self.revision)
        manifest.variantConditions = [.init(
            name: "arm", artifactPath: "runs/x/a.json",
            artifactHash: String(repeating: "a", count: 64), artifact: artifact)]
        manifest.outcomeInstruments = instruments
        manifest.status = status
        return manifest
    }

    @Test func aDraftVerifyNamesEachAgentArmRule() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "agent-arms") { root in
            func violations(_ object: [String: Any], instruments: [String]? = nil) throws -> [String] {
                let manifest = try study(root, artifact: try decode(object), instruments: instruments)
                try? FileManager.default.removeItem(
                    at: ExperimentRepository(workspaceRoot: root).manifestURL("study")
                        .deletingLastPathComponent())
                return ExperimentStore.verify(manifest).filter { $0.contains("agent arm 'arm'") }
            }
            #expect(try violations(["reasoningEffort": "medium"]).contains {
                $0.contains("has no thinking switch") })
            #expect(try violations(["reasoningEffort": "medium", "qwenThinkingEnabled": false])
                .contains { $0.contains("one says the agent thinks") })
            #expect(try violations(["adapters": [["adapterDirectory": "a"], ["adapterDirectory": "b"]]])
                .contains { $0.contains("lists 2 adapters") })
            #expect(try violations(["qwenThinkingEnabled": true], instruments: ["answerTokenLogprob"])
                .contains { $0.contains("answer-token instrument") })
            #expect(try violations(["baseRevision": String(repeating: "b", count: 40)])
                .contains { $0.contains("made for revision") })
            let (reference, _) = try vector(root, name: "bare", norms: false)
            #expect(try violations(["alphaInNormUnits": true, "injections": [
                ["concept": "bare", "vectorArtifactID": reference, "layer": 9, "alpha": 0.1]]])
                .contains { $0.contains("at layer 9, which its vector does not have") })
            #expect(try violations(["alphaInNormUnits": true, "injections": [
                ["concept": "bare", "vectorArtifactID": reference, "layer": 2, "alpha": 0.1]]])
                .contains { $0.contains("records no residual norm at layer 2") })
        }
    }

    @Test func aFrozenStudyIsToldAtVerifyAndRefusedAtSubmissionAndRunStart() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "agent-arms") { root in
            let manifest = try study(
                root, artifact: try decode(["qwenThinkingEnabled": true]),
                instruments: ["answerTokenLogprob"], status: .frozen)
            #expect(!ExperimentStore.verify(manifest).contains { $0.contains("agent arm") })
            let note = ModelVariantArtifact.frozenArmProblemsAdvisory(manifest, workspaceRoot: root)
            #expect(note?.hasPrefix("this study will be refused when its run is submitted") == true)
            #expect(note?.contains("experiment duplicate study study-v2") == true)
            #expect(SubstrateRouting.agentArmSubmissionRefusal(
                manifest: manifest, verb: "run", workspaceRoot: root)?.lifecycleRefusal?.gate
                == .studyDeclaration)
            #expect(SubstrateRouting.agentArmSubmissionRefusal(
                manifest: manifest, verb: "analyze", workspaceRoot: root) == nil)
            #expect(ModelVariantArtifact.armRefusal(manifest, workspaceRoot: root) != nil)
        }
    }

    @Test func aRecordStampsWhatReachedTheRenderer() {
        #expect(ModelVariantArtifact.renderedReasoningEffort(.xhigh, modelID: Self.model) == .off)
        #expect(ModelVariantArtifact.renderedReasoningEffort(
            .medium, modelID: "Qwen/Qwen3.8-27B") == .medium)
        let qwen = ModelCapabilities.heuristicView(modelID: "Qwen/Qwen3.8-27B", revision: nil)
        let ignoring = ModelCapabilities(
            modelID: qwen.modelID, revision: nil, source: .probe, systemRole: qwen.systemRole,
            foldSeparator: qwen.foldSeparator, thinkingSwitch: qwen.thinkingSwitch,
            thinkOpenInPrompt: qwen.thinkOpenInPrompt, effortVariableRead: false,
            effortLevels: ["low": .ignored, "medium": .ignored, "xhigh": .ignored],
            thinkTokens: qwen.thinkTokens, architecture: nil, templateSha256: nil,
            tokenizerConfigSha256: nil)
        #expect(ModelVariantArtifact.renderedReasoningEffort(
            .medium, modelID: "Qwen/Qwen3.8-27B", capabilities: ignoring) == .on)
    }

    @Test func anOffStudyMayDeclareABudgetForItsThinkingAgents() throws {
        var manifest = ExperimentManifest(name: "s", description: "", modelID: "Qwen/Qwen3.8-27B")
        manifest.reasoningEffort = "off"
        manifest.reasoningMaxTokens = 256
        #expect(manifest.reasoningProtocolViolations == [ReasoningEffort.budgetWithoutEffortReason])
        let thinking = try decode(["baseModelID": "Qwen/Qwen3.8-27B", "reasoningEffort": "on"])
        manifest.variantConditions = [.init(
            name: "arm", artifactPath: "runs/x/a.json",
            artifactHash: String(repeating: "a", count: 64), artifact: thinking)]
        #expect(manifest.reasoningProtocolViolations.isEmpty)
        // The budget reaches only the arms that think.
        #expect(StudySampling.reasoningBudget(manifest, variant: thinking) == 256)
        #expect(StudySampling.reasoningBudget(manifest, variant: nil) == nil)
        #expect(StudySampling.reasoningBudget(
            manifest, variant: try decode(["baseModelID": "Qwen/Qwen3.8-27B"])) == nil)
    }

    @Test func attachRefusesAnAgentMadeForAnotherRevision() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "agent-attach") { root in
            let written = try create(root, base("plain"))
            guard case .string(let path)? = written.payload["path"] else { return }
            var manifest = try ExperimentStore.create(
                name: "study", description: "", modelID: Self.model,
                modelRevision: String(repeating: "b", count: 40))
            let snapshot = try AgentArtifactSnapshot(workspaceRoot: root, path: path)
            do {
                try ExperimentStore.attachAgent(snapshot.record, into: &manifest, workspaceRoot: root)
                Issue.record("expected a refusal")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("was made for revision"))
            }
        }
    }

    @Test func resavingAnOlderSwitchAsItsLevelIsNotAnEdit() throws {
        // An older file's thinking switch (true, read as xhigh) saved back
        // as an explicit xhigh is the same agent: its record stays.
        var stored = try decode(["qwenThinkingEnabled": true])
        stored.promotion = .init(
            experiment: "s", experimentHash: "h", promotedAt: "t", promotedBy: "criterion",
            substrate: "swift-mlx", appVersion: "0")
        var edited = stored
        edited.promotion = nil
        edited.reasoningEffort = ReasoningEffort.xhigh.rawValue
        let merge = AgentEditProvenance.merge(edited: edited, over: stored)
        #expect(!merge.droppedPromotion)
        #expect(merge.artifact.promotion == stored.promotion)
    }

    @Test func theEditorRecordsTheAgentsItSaves() throws {
        let fresh = try decode([:])
        let recorded = fresh.withEditorRecordIfUnrecorded()
        #expect(recorded.handmade?.agentRecordObject?["madeWith"] == .string("agent editor"))
        #expect(recorded.origin.summary.hasPrefix("Made by hand in the app's agent editor."))
        var promoted = fresh
        promoted.promotion = .init(
            experiment: "s", experimentHash: "h", promotedAt: "t", promotedBy: "criterion",
            substrate: "swift-mlx", appVersion: "0")
        #expect(promoted.withEditorRecordIfUnrecorded().handmade == nil)
    }

    @Test func anUneditedSaveKeepsTheHandmadeRecordAndAnEditDropsIt() throws {
        var stored = try decode(["reasoningEffort": "on"])
        stored.handmade = .object(["madeWith": .string("agent create")])
        var unedited = stored
        unedited.handmade = nil  // the editor has no control for the record
        #expect(AgentEditProvenance.merge(edited: unedited, over: stored).artifact.handmade != nil)
        var edited = unedited
        edited.reasoningEffort = nil
        edited.qwenThinkingEnabled = false
        #expect(AgentEditProvenance.merge(edited: edited, over: stored).artifact.handmade == nil)
    }
}
