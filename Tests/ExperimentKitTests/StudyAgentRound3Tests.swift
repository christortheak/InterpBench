import CryptoKit
import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// FR6-B round 3: the revision rule refuses only commits that differ, a
/// repair that replaces an arm and clears the refusal, the editor's record of
/// what an agent was edited from, single-layer ablations, a check for panel
/// seats, and the one budget predicate. Python twin:
/// `Server/tests/test_agent_round3.py`.
@Suite(.serialized) struct StudyAgentRound3Tests {
    static let model = "test/model"
    static let qwen = "Qwen/Qwen3.8-27B"
    static let revision = String(repeating: "a", count: 40)
    static let other = String(repeating: "b", count: 40)

    private func sha(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// A temporary workspace both the store and the command line resolve.
    private func withWorkspace<T>(_ body: (URL) async throws -> T) async throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "fr6b-r3-\(UUID().uuidString)")
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

    private func invoke(_ namespace: String, _ args: [String]) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(sink: .discarding).run(namespace: namespace, args + ["--json"])
    }

    private func decode(_ object: [String: Any]) throws -> ModelVariantArtifact {
        var full: [String: Any] = ["name": "a", "baseModelID": Self.model]
        for (key, value) in object { full[key] = value }
        return try JSONDecoder().decode(
            ModelVariantArtifact.self, from: JSONSerialization.data(withJSONObject: full))
    }

    /// An agent file in the library, written by hand (so it may break a rule
    /// `agent create` would refuse): its workspace path and file digest.
    @discardableResult
    private func agentFile(
        _ root: URL, _ name: String, _ fields: [String: Any] = [:]
    ) throws -> (path: String, sha256: String, artifact: ModelVariantArtifact) {
        var document: [String: Any] = [
            "schemaVersion": 1, "name": name, "baseModelID": Self.model,
            "alphaInNormUnits": true, "createdAt": "2026-11-08T00:00:00Z",
        ]
        for (key, value) in fields { document[key] = value }
        let folder = root.appending(components: "runs", "model-variants", name)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let data = try JSONSerialization.data(withJSONObject: document)
        try data.write(to: folder.appending(component: "model-variant.json"))
        return ("runs/model-variants/\(name)/model-variant.json", sha(data),
                try JSONDecoder().decode(ModelVariantArtifact.self, from: data))
    }

    private func manifestDigest(_ root: URL, _ name: String = "study") throws -> String {
        try DraftAuthoringSnapshot(workspaceRoot: root, name: name).file.sha256
    }

    // MARK: - R1: the revision rule

    @Test(arguments: [
        ("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", String(repeating: "a", count: 40), "match"),
        ("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", String(repeating: "a", count: 40), "match"),
        (String(repeating: "a", count: 40), String(repeating: "b", count: 40), "differ"),
        ("main", String(repeating: "a", count: 40), "unchecked"),
        (String(repeating: "a", count: 40), "main", "unchecked"),
        ("aaaaaaa", String(repeating: "a", count: 40), "unchecked"),
        ("bbbbbbb", String(repeating: "a", count: 40), "differ"),
        ("aaaaaaa", "aaaaaaab", "unchecked"),
        ("v1.0", "v1.1", "unchecked"),
    ])
    func revisionsDifferOnlyWhenBothNameCommits(agent: String, study: String, verdict: String) {
        #expect(ModelVariantArtifact.compareRevisions(agent, study).rawValue == verdict)
    }

    @Test func verifyRefusesDifferingCommitsAndNotesTheRest() async throws {
        try await withWorkspace { root in
            for (revision, refused, phrase) in [
                ("main", false, "main is a branch or tag name"),
                ("aaaaaaa", false, "a shortened commit hash that the other begins with"),
                (nil, false, "the agent records no model revision"),
                ("bbbbbbb", true, ""),
                (Self.other, true, ""),
            ] as [(String?, Bool, String)] {
                var manifest = ExperimentManifest(
                    name: "study", description: "", modelID: Self.model)
                manifest.modelRevision = Self.revision
                manifest.variantConditions = [.init(
                    name: "arm", artifactPath: "runs/x/a.json",
                    artifactHash: String(repeating: "a", count: 64),
                    artifact: try decode(revision.map { ["baseRevision": $0] } ?? [:]))]
                let violations = ModelVariantArtifact.manifestArmProblems(
                    manifest, workspaceRoot: root)
                let notes = ModelVariantArtifact.manifestArmNotes(manifest)
                if refused {
                    #expect(violations.contains { $0.contains("was made for revision") })
                    #expect(notes.isEmpty)
                } else {
                    #expect(violations.isEmpty, "\(violations)")
                    #expect(notes.count == 1 && notes[0].contains(phrase), "\(notes)")
                    #expect(notes.first?.contains("did not check") == true)
                    // Agents saved today can carry none too, so no note
                    // blames their age.
                    #expect(notes.first?.contains("saved before") == false)
                }
            }
        }
    }

    @Test func attachRefusesOnlyDifferingCommits() async throws {
        try await withWorkspace { root in
            for (revision, refused) in [("main", false), ("aaaaaaa", false),
                                        ("bbbbbbb", true), (Self.other, true)] {
                var manifest = try ExperimentStore.create(
                    name: "study-\(revision.count)-\(revision.prefix(4))", description: "",
                    modelID: Self.model, modelRevision: Self.revision)
                let file = try agentFile(root, "pinned-\(revision.count)-\(revision.prefix(4))",
                                         ["baseRevision": revision])
                let snapshot = try AgentArtifactSnapshot(workspaceRoot: root, path: file.path)
                do {
                    try ExperimentStore.attachAgent(
                        snapshot.record, into: &manifest, workspaceRoot: root)
                    #expect(!refused, "\(revision) should be refused")
                } catch let error as ExperimentError {
                    #expect(refused, "\(revision): \(error.reason)")
                    #expect(error.reason.hasPrefix("agent 'pinned-"))
                    #expect(!error.reason.contains("': agent '"))
                }
            }
        }
    }

    // MARK: - R2: --replace, and repairs that clear the refusal

    @Test func replacePutsTheAgentInTheArmsPlace() async throws {
        try await withWorkspace { root in
            var manifest = try ExperimentStore.create(
                name: "study", description: "", modelID: Self.model,
                modelRevision: Self.revision)
            for name in ["first", "second", "third"] {
                let file = try agentFile(root, name)
                try ExperimentStore.attachAgent(
                    try AgentArtifactSnapshot(workspaceRoot: root, path: file.path).record,
                    into: &manifest, workspaceRoot: root)
            }
            let fixed = try agentFile(root, "fixed")
            try ExperimentStore.attachAgent(
                try AgentArtifactSnapshot(workspaceRoot: root, path: fixed.path).record,
                into: &manifest, workspaceRoot: root, replacing: "second")
            #expect(manifest.variantConditions.map(\.name) == ["first", "fixed", "third"])
            let late = try agentFile(root, "late")
            do {
                try ExperimentStore.attachAgent(
                    try AgentArtifactSnapshot(workspaceRoot: root, path: late.path).record,
                    into: &manifest, workspaceRoot: root, replacing: "absent")
                Issue.record("an absent arm must refuse")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("no agent arm named absent"))
            }
        }
    }

    /// Each command-line command a repair names, as written.
    /// Each command a repair names, as written: its steps in order (`; then`),
    /// taking alternative `choice` where a step offers several (`; or`).
    private func commands(_ repair: String, choice: Int = 0) -> [String] {
        let pattern = #"steerlab-cli (.+?)(?= \(or the app's agent editor\)|, then |$)"#
        let regex = try! NSRegularExpression(pattern: pattern)
        return repair.components(separatedBy: "; then ").flatMap { step -> [String] in
            let alternatives = step.components(separatedBy: "; or ")
            let chosen = alternatives[min(choice, alternatives.count - 1)]
            let range = NSRange(chosen.startIndex..., in: chosen)
            return regex.matches(in: chosen, range: range).compactMap {
                Range($0.range(at: 1), in: chosen).map { String(chosen[$0]) }
            }
        }
    }

    /// Runs a repair's commands as written, filling only its placeholders:
    /// the new agent's name and the digests the earlier commands return.
    private func follow(_ root: URL, _ repair: String, choice: Int = 0) async throws {
        var made: (path: String, sha256: String)?
        for command in commands(repair, choice: choice) {
            var words = command.replacingOccurrences(of: " …", with: "")
                .split(separator: " ").map(String.init)
            let namespace = words.removeFirst()
            if namespace == "agent", words.first == "create" {
                words = words.map { $0 == "<new-name>" ? "fixed" : $0 }
                let preview = await invoke(namespace, words)
                guard case .string(let digest)? = preview.envelope.result?["reviewSHA256"] else {
                    Issue.record("no preview for \(command): \(preview.envelope)"); return
                }
                let written = await invoke(namespace, words + ["--review-sha256", digest])
                guard case .string(let path)? = written.envelope.result?["path"],
                    case .string(let file)? = written.envelope.result?["artifactFileSHA256"]
                else { Issue.record("no agent written: \(written.envelope)"); return }
                made = (path, file)
                continue
            }
            if words.first == "attach-agent", let made {
                var digests = [made.sha256, try manifestDigest(root, words[1])].makeIterator()
                words = words.map {
                    $0 == "<path>" ? made.path : $0 == "<digest>" ? digests.next() ?? $0 : $0
                }
            }
            let outcome = await invoke(namespace, words)
            #expect(outcome.envelope.exitCode == 0, "\(command): \(outcome.envelope)")
        }
    }

    private func brokenStudy(_ root: URL, _ rule: String, frozen: Bool) throws {
        let model = rule == "instrument" ? Self.qwen : Self.model
        var manifest = try ExperimentStore.create(
            name: "study", description: "", modelID: model, modelRevision: Self.revision)
        let file: (path: String, sha256: String, artifact: ModelVariantArtifact)
        switch rule {
        case "level":
            file = try agentFile(root, "thinker", [
                "baseRevision": Self.revision, "reasoningEffort": "medium",
                "qwenThinkingEnabled": true])
        case "revision":
            file = try agentFile(root, "elsewhere", ["baseRevision": Self.other])
        default:
            file = try agentFile(root, "reasoner", [
                "baseModelID": Self.qwen, "baseRevision": Self.revision,
                "reasoningEffort": "on", "qwenThinkingEnabled": true])
            manifest.outcomeInstruments = ["answerTokenLogprob"]
        }
        manifest.variantConditions = [.init(
            name: file.artifact.name, artifactPath: file.path, artifactHash: file.sha256,
            artifact: file.artifact)]
        if frozen { manifest.status = .frozen }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(
            to: ExperimentRepository(workspaceRoot: root).manifestURL("study"))
    }

    /// Each alternative a repair offers clears it alone: for a thinking agent
    /// beside an answer-token instrument, either new instruments (choice 0)
    /// or an agent remade with thinking off (choice 1).
    @Test(arguments: [("level", false, 0), ("level", true, 0), ("revision", false, 0),
                      ("revision", true, 0), ("instrument", false, 0), ("instrument", true, 0),
                      ("instrument", false, 1), ("instrument", true, 1)])
    func eachRepairFollowedAsWrittenClearsTheRefusal(
        rule: String, frozen: Bool, choice: Int
    ) async throws {
        try await withWorkspace { root in
            try brokenStudy(root, rule, frozen: frozen)
            let manifest = try ExperimentStore.load(name: "study")
            let refusal = try #require(
                ModelVariantArtifact.armRefusal(manifest, workspaceRoot: root))
            let repair = try #require(refusal.lifecycleRefusal?.repairAction)
            #expect(repair.contains("experiment duplicate study study-v2") == frozen)
            #expect(repair.contains("--replace"))
            if rule == "instrument" {
                #expect(repair.contains("set-instruments")
                    && repair.contains("--reasoning-effort off"))
            }
            try await follow(root, repair, choice: choice)
            let repaired = try ExperimentStore.load(name: frozen ? "study-v2" : "study")
            #expect(ModelVariantArtifact.armRefusal(repaired, workspaceRoot: root) == nil,
                    "\(ModelVariantArtifact.manifestArmProblems(repaired, workspaceRoot: root))")
        }
    }

    // MARK: - R5: what the editor's record says

    @Test func anEditedAgentNamesWhatItWasEditedFrom() async throws {
        try await withWorkspace { root in
            var promoted = try decode([
                "injections": [["concept": "c", "vectorArtifactID": "runs/x/c", "layer": 2,
                                "alpha": 0.1]]])
            promoted.promotion = .init(
                experiment: "tone", experimentHash: "h", promotedAt: "t",
                promotedBy: "criterion", sweepRun: "sweep-1", substrate: "swift-mlx",
                appVersion: "0")
            let stored = try ModelVariantStore.save(
                promoted, library: root.appending(components: "runs", "model-variants"))
            let before = try ModelVariantStore.hash(stored.url)
            var edited = promoted
            edited.promotion = nil
            edited.systemPrompt = "Answer as a careful reviewer."
            edited.reasoningEffort = "off"
            let outcome = try ModelVariantStore.saveChanges(
                edited, to: stored, takenNames: [promoted.name])
            guard case .updated(let updated, let dropped) = outcome else {
                Issue.record("expected an update in place"); return
            }
            #expect(dropped)
            let record = try #require(updated.artifact.handmade?.agentRecordObject)
            let source = try #require(record["editedFrom"]?.agentRecordObject)
            #expect(source["kind"] == .string("swept"))
            #expect(source["experiment"] == .string("tone"))
            #expect(source["sweepRun"] == .string("sweep-1"))
            #expect(source["artifactFileSHA256"] == .string(before))
            #expect(record["changed"] == .array([.string("systemPrompt")]))
            #expect(updated.artifact.origin.summary == "Made by hand in the app's agent "
                + "editor. The researcher edited it from an agent experiment promote made "
                + "from study tone (sweep run sweep-1) and changed its persona. No reason "
                + "was stated.")
            #expect(!updated.artifact.origin.summary.contains("no sweep selected"))
        }
    }

    @Test func theEditedFromRecordNamesTheSourcesFileByItsHash() async throws {
        try await withWorkspace { root in
            let source = try decode(["reasoningEffort": "on", "baseModelID": Self.qwen])
            let stored = try ModelVariantStore.save(
                source, library: root.appending(components: "runs", "model-variants"))
            let before = try ModelVariantStore.hash(stored.url)
            var edited = source
            edited.bandWidth = 3
            let copy = try ModelVariantStore.saveAsNewAgent(
                edited, takenNames: [source.name], source: stored)
            let record = try #require(copy.artifact.handmade?.agentRecordObject)
            let from = try #require(record["editedFrom"]?.agentRecordObject)
            #expect(from["artifactFileSHA256"] == .string(before))
            #expect(from["kind"] == .string("handmade"))
            #expect(record["changed"] == .array([.string("bandWidth")]))
            let text = String(decoding: try Data(contentsOf: copy.url), as: UTF8.self)
            #expect(!text.contains(root.path))
        }
    }

    @Test func theOriginSummariesMatchTheServers() throws {
        let url = CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cross-engine", "agent-origin-summaries.json")
        let cases = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [[String: Any]])
        #expect(cases.count >= 6)
        for entry in cases {
            let artifact = try JSONDecoder().decode(
                ModelVariantArtifact.self,
                from: JSONSerialization.data(withJSONObject: entry["artifact"] as Any))
            #expect(artifact.origin.summary == entry["summary"] as? String)
        }
    }

    // MARK: - R6: an ablation's layers

    private func vector(_ root: URL, residualNorms: [Double] = [10, 10, 10, 10]) throws
        -> (reference: String, sha256: String)
    {
        let run = root.appending(components: "runs", "20261108-extract")
        try FileManager.default.createDirectory(at: run, withIntermediateDirectories: true)
        let tensor = Data("tensor-formality".utf8)
        try tensor.write(to: run.appending(component: "formality.safetensors"))
        try JSONSerialization.data(withJSONObject: [
            "modelID": Self.model, "concept": "formality", "stimulusSetHash": "h",
            "layerCount": 4, "hiddenSize": 2, "normsPerLayer": [1.0, 0.0, 3.0, 4.0],
            "extractionDate": "2026-11-08", "revision": Self.revision,
            "residualNormPerLayer": residualNorms,
        ] as [String: Any]).write(to: run.appending(component: "formality.json"))
        return ("runs/20261108-extract/formality", sha(tensor))
    }

    private func preview(_ root: URL, _ args: [String]) async -> ExperimentCLIOutcome {
        await invoke("agent", ["create", "steered", "--model", Self.model,
                               "--revision", Self.revision] + args)
    }

    @Test func anAblationAtOneLayerListsThatLayer() async throws {
        try await withWorkspace { root in
            let (reference, digest) = try vector(root)
            let ablate = ["--vector", reference, "--vector-sha256", digest, "--strength", "1",
                          "--mode", "ablate"]
            let one = await preview(root, ablate + ["--layer", "2"])
            guard case .object(let document)? = one.envelope.result?["document"],
                case .array(let injections)? = document["injections"],
                case .object(let injection)? = injections.first
            else { Issue.record("no document: \(one.envelope)"); return }
            #expect(injection["layers"] == .array([.number(2)]))
            let every = await preview(root, ablate + ["--layer", "all"])
            guard case .object(let all)? = every.envelope.result?["document"],
                case .array(let allInjections)? = all["injections"],
                case .object(let allInjection)? = allInjections.first,
                case .object(let record)? = all["handmade"],
                case .object(let inputs)? = record["inputs"],
                case .array(let vectors)? = inputs["vectors"],
                case .object(let recorded)? = vectors.first,
                case .array(let notes)? = every.envelope.result?["advisories"]
            else { Issue.record("no document: \(every.envelope)"); return }
            #expect(allInjection["layers"] == nil)
            #expect(recorded["layer"] == .string("all"))
            #expect(notes.contains { $0.agentRecordString?.contains("ablated at every layer") == true })
            let steering = await preview(root, [
                "--vector", reference, "--vector-sha256", digest, "--strength", "0.1",
                "--layer", "all"])
            #expect(steering.envelope.exitCode == 64)
        }
    }

    @Test func theArmRulesCheckAnAblationsLayersAndSkipAZeroLayer() async throws {
        try await withWorkspace { root in
            let (reference, digest) = try vector(root, residualNorms: [10])
            var manifest = ExperimentManifest(name: "study", description: "", modelID: Self.model)
            func problems(_ injection: [String: Any], band: Int = 1) throws -> [String] {
                manifest.variantConditions = [.init(
                    name: "arm", artifactPath: "runs/x/a.json",
                    artifactHash: String(repeating: "a", count: 64),
                    artifact: try decode(["alphaInNormUnits": true, "bandWidth": band,
                                          "injections": [injection]]))]
                return ModelVariantArtifact.manifestArmProblems(manifest, workspaceRoot: root)
            }
            #expect(try problems([
                "concept": "formality", "vectorArtifactID": reference, "layer": 2, "alpha": 1,
                "mode": "ablate", "layers": [2, 9]])
                .contains { $0.contains("the ablation of 'formality' lists layer 9") })
            // Layer 1 has no residual norm, but the vector is zero there: the
            // engine skips it, so neither the rules nor agent create ask.
            #expect(try problems([
                "concept": "formality", "vectorArtifactID": reference, "layer": 0,
                "alpha": 0.1], band: 3).isEmpty)
            let created = await preview(root, [
                "--vector", reference, "--vector-sha256", digest, "--layer", "0",
                "--strength", "0.1", "--band-width", "3"])
            #expect(created.envelope.exitCode == 0, "\(created.envelope)")
        }
    }

    // MARK: - Nits

    @Test func anAdapterOnABranchIsNotedNotLooped() async throws {
        try await withWorkspace { root in
            func adapter(_ revision: String) throws -> (String, String) {
                let run = root.appending(components: "runs", "lora-\(revision)")
                let folder = run.appending(component: "tone")
                try FileManager.default.createDirectory(
                    at: folder, withIntermediateDirectories: true)
                let weights = Data("weights-\(revision)".utf8)
                try weights.write(to: folder.appending(component: "adapter_model.safetensors"))
                try Data("{}".utf8).write(to: folder.appending(component: "adapter_config.json"))
                try JSONSerialization.data(withJSONObject: [
                    "name": "tone", "baseModelID": Self.model, "revision": revision,
                    "adapterFormat": "hf-peft-lora",
                ]).write(to: run.appending(component: "tone.json"))
                return ("runs/lora-\(revision)/tone", sha(weights))
            }
            let (branch, branchDigest) = try adapter("main")
            let noted = await preview(root, ["--adapter", branch, "--adapter-sha256", branchDigest])
            guard case .array(let notes)? = noted.envelope.result?["advisories"] else {
                Issue.record("no preview: \(noted.envelope)"); return
            }
            #expect(notes.contains { $0.agentRecordString?.contains("records revision main") == true })
            let (short, shortDigest) = try adapter("bbbbbbb")
            let refused = await preview(root, ["--adapter", short, "--adapter-sha256", shortDigest])
            #expect(refused.envelope.exitCode == 65)
            #expect(refused.envelope.error?.repairAction.contains(
                "full 40-character commit it names") == true)
        }
    }

    @Test func aReasonNamingAHomeFolderIsKeptAndWarned() async throws {
        try await withWorkspace { root in
            let reason = "copied from " + "/" + "Users" + "/someone/notes.txt"
            let outcome = await preview(root, ["--reason", reason])
            guard case .object(let document)? = outcome.envelope.result?["document"],
                case .object(let record)? = document["handmade"],
                case .array(let notes)? = outcome.envelope.result?["advisories"]
            else { Issue.record("no preview: \(outcome.envelope)"); return }
            #expect(record["reason"] == .string(reason))
            #expect(notes.contains { $0.agentRecordString?.contains("kept word for word") == true })
        }
    }

    @Test func aRejectedLevelIsStampedAsWhatRendered() {
        let qwen = ModelCapabilities.heuristicView(modelID: Self.qwen, revision: nil)
        let rejecting = ModelCapabilities(
            modelID: qwen.modelID, revision: nil, source: .probe, systemRole: qwen.systemRole,
            foldSeparator: qwen.foldSeparator, thinkingSwitch: qwen.thinkingSwitch,
            thinkOpenInPrompt: qwen.thinkOpenInPrompt, effortVariableRead: true,
            effortLevels: ["low": .rejected, "medium": .accepted, "xhigh": .accepted],
            thinkTokens: qwen.thinkTokens, architecture: nil, templateSha256: nil,
            tokenizerConfigSha256: nil)
        // `thinkingContext` renders a rejected level as plain thinking.
        #expect(PromptRendering.thinkingContext(
            modelID: Self.qwen, effort: .low, capabilities: rejecting)?["reasoning_effort"] == nil)
        #expect(ModelVariantArtifact.renderedReasoningEffort(
            .low, modelID: Self.qwen, capabilities: rejecting) == .on)
        #expect(ModelVariantArtifact.renderedReasoningEffort(
            .medium, modelID: Self.qwen, capabilities: rejecting) == .medium)
    }

    /// A panel loads each seat at its own revision, so a seat agent made for
    /// another commit is legitimate; the level rule still applies.
    @Test func aCastAgentMeetsTheRulesButTheAnswerTokenAndRevisionOnes() throws {
        let root = FileManager.default.temporaryDirectory
        let thinking = try decode(["reasoningEffort": "medium", "baseRevision": Self.other])
        let problems = thinking.castAgentProblems(
            label: "seat 'judge'", modelID: Self.model, capabilities: nil, workspaceRoot: root)
        #expect(problems.first?.hasPrefix("seat 'judge': ") == true)
        #expect(!problems.contains { $0.contains("made for revision") })
        let legacy = try decode([
            "baseModelID": Self.qwen, "qwenThinkingEnabled": true, "baseRevision": Self.other])
        #expect(legacy.castAgentProblems(
            label: "seat 'judge'", modelID: Self.qwen, capabilities: nil,
            workspaceRoot: root).isEmpty)
    }

    // MARK: - Round 4

    @Test func replaceRefusesToDropAnotherArmAndNamesTheRename() async throws {
        try await withWorkspace { root in
            _ = try ExperimentStore.create(
                name: "study", description: "", modelID: Self.model,
                modelRevision: Self.revision)
            var files: [String: (path: String, sha256: String)] = [:]
            for name in ["first", "second", "fixed"] {
                let file = try agentFile(root, name)
                files[name] = (file.path, file.sha256)
            }
            func attach(_ name: String, _ extra: [String] = []) async throws -> ExperimentCLIOutcome {
                let file = try #require(files[name])
                return await invoke("experiment", [
                    "attach-agent", "study", "--artifact", file.path,
                    "--artifact-sha256", file.sha256,
                    "--manifest-sha256", try manifestDigest(root)] + extra)
            }
            #expect(try await attach("first").envelope.exitCode == 0)
            #expect(try await attach("second").envelope.exitCode == 0)
            // The agent "first" is attached already: putting it in second's
            // place would drop arm first as well.
            let clash = try await attach("first", ["--replace", "second"])
            #expect(clash.envelope.exitCode == 65)
            #expect(clash.envelope.error?.reason.contains("would drop that arm too") == true)
            #expect(clash.envelope.error?.repairAction.contains("--replace first") == true)
            #expect(try ExperimentStore.load(name: "study").variantConditions.map(\.name)
                == ["first", "second"])
            let renamed = try await attach("fixed", ["--replace", "second"])
            #expect(renamed.envelope.exitCode == 0, "\(renamed.envelope)")
            guard case .object(let replaced)? = renamed.envelope.result?["replaced"] else {
                Issue.record("no replaced block: \(renamed.envelope)"); return
            }
            #expect(replaced["renamed"] == .bool(true) && replaced["now"] == .string("fixed"))
            #expect(renamed.envelope.message.contains("Arm second is now arm fixed"))
        }
    }

    @Test func anUnchangedCopyAndAChainOfEditsKeepTheirOrigin() throws {
        var promoted = try decode([:])
        promoted.promotion = .init(
            experiment: "tone", experimentHash: "h", promotedAt: "t", promotedBy: "criterion",
            sweepRun: "sweep-1", substrate: "swift-mlx", appVersion: "0")
        var copy = promoted
        copy.promotion = nil
        copy = copy.withEditorRecordIfUnrecorded(editedFrom: (promoted, "c"))
        #expect(copy.origin.summary.contains(
            "saved it as an unchanged copy of an agent experiment promote made from study tone"))
        #expect(!copy.origin.summary.contains("edited"))
        // An edit of an edit keeps the sweep origin and every recorded change.
        var first = copy
        first.handmade = nil
        first.systemPrompt = "Be brief."
        first = first.withEditorRecordIfUnrecorded(editedFrom: (copy, "d"))
        var second = first
        second.handmade = nil
        second.bandWidth = 3
        second = second.withEditorRecordIfUnrecorded(editedFrom: (first, "e"))
        let record = try #require(second.handmade?.agentRecordObject)
        let origin = try #require(record["editedFrom"]?.agentRecordObject)
        #expect(origin["kind"] == .string("swept") && origin["sweepRun"] == .string("sweep-1"))
        #expect(origin["artifactFileSHA256"] == .string("e"))
        #expect(record["changed"] == .array([.string("bandWidth"), .string("systemPrompt")]))
        #expect(second.origin.summary.contains("edited it from an agent experiment promote "
            + "made from study tone (sweep run sweep-1) and changed its layer band and persona"))
    }

    @Test func aPromotionWhoseSweepRunCannotBeReadSaysWhy() async throws {
        try await withWorkspace { _ in
            do {
                _ = try AgentPromotion.sweepRunLevel(
                    sweepRun: nil, experimentName: "s", concept: "c")
                Issue.record("expected a refusal")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("names no sweep run"))
                #expect(!error.reason.contains("None"))
            }
            do {
                _ = try AgentPromotion.sweepRunLevel(
                    sweepRun: "20260707-exp-s-sweep", experimentName: "s", concept: "c")
                Issue.record("expected a refusal")
            } catch let error as ExperimentError {
                #expect(error.reason.contains("is not in this workspace's runs/ folder"))
                #expect(error.lifecycleRefusal?.repairAction.contains(
                    "fetch the sweep's results into runs/") == true)
            }
        }
    }

    @Test func aPinnedReuseRepairNamesWhereAPinnedPromotionRuns() {
        let pinned = AgentPromotion.promoteCommand(
            "pr", "fear", agentName: "pr-fear-agent-medium", cell: nil, overrideReason: nil,
            pins: .init(sweepRun: "run-1", experimentHash: "e", winningCell: (layer: 3, alpha: 0.4)))
        #expect(pinned.hasPrefix("this command line cannot pin a promotion to a sweep run"))
        #expect(pinned.contains("steerlab-server experiment promote pr fear --agent-name "
            + "pr-fear-agent-medium --sweep-run run-1 --expect-cell 3:0.4 --expect-epoch e "
            + "--root <workspace>"))
        #expect(AgentPromotion.promoteCommand(
            "pr", "fear", agentName: "a", cell: nil, overrideReason: nil, pins: nil)
            == "steerlab-cli experiment promote pr fear --agent-name a")
    }

    @Test func theBudgetGatesAskTheOnePredicate() throws {
        var manifest = ExperimentManifest(name: "s", description: "", modelID: Self.qwen)
        manifest.reasoningEffort = "off"
        #expect(!ModelVariantArtifact.budgetPermittedWithoutEffort(manifest))
        #expect(throws: ExperimentError.self) {
            try ManifestDraftEdits.setSamplingProtocol(
                reasoningMaxTokens: 128, experimentName: "s", manifest: &manifest)
        }
        manifest.variantConditions = [.init(
            name: "arm", artifactPath: "runs/x/a.json",
            artifactHash: String(repeating: "a", count: 64),
            artifact: try decode(["baseModelID": Self.qwen, "reasoningEffort": "on"]))]
        #expect(ModelVariantArtifact.budgetPermittedWithoutEffort(manifest))
        try ManifestDraftEdits.setSamplingProtocol(
            reasoningMaxTokens: 128, experimentName: "s", manifest: &manifest)
        #expect(manifest.reasoningMaxTokens == 128)
        #expect(manifest.reasoningProtocolViolations.isEmpty)
        // The study's own arms never think here, so none of them names it.
        #expect(StudySampling.studyArmBudget(manifest) == nil)
        manifest.reasoningEffort = "on"
        #expect(StudySampling.studyArmBudget(manifest) == 128)
    }
}
