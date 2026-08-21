import Foundation
import Testing

@testable import ExperimentKit

/// `steerlab-cli panel compile` — headless seat casting (open-issues §18).
///
/// The defect this closes is not a missing convenience. `PanelComposition
/// .compileAndPin` was reachable only from the app, so the only headless way to
/// cast a panel was to re-implement the compile transform outside the engine
/// and hand-edit `multiAgentScenarioPath`/`Hash` plus the semantic pins into a
/// draft — which is exactly what the 2026-08-19 Latin-square preparation did,
/// in a workspace Python script, with `experiment verify` as the only net. Two
/// implementations of one transform is a standing drift risk against the
/// measured artifact.
///
/// So the load-bearing test here is `castingFromTheCLIAndFromTheUIPathAgreeByte
/// ForByte`: the same panel, the same seats, the same study settings, cast once
/// through the verb and once through the call the app's Seats section makes,
/// must produce identical compiled BYTES and identical manifest pins. Everything
/// else is the refusal surface an agent has to be able to act on.
///
/// Serialized, like every suite that moves the process-global workspace root:
/// the scenario library, the agent library and `experiments/` all have to land
/// in the same temporary tree.
@Suite(.serialized) struct PanelCompileCLITests {

    // MARK: - Fixtures

    func withTempWorkspace<T>(_ body: (URL) async throws -> T) async rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        // Symlinks RESOLVED: the temp directory is /var → /private/var, and
        // this suite compares scanned record ids (absolute paths) against ids
        // built from the store's own roots.
        let temp = FileManager.default.temporaryDirectory
            .resolvingSymlinksInPath()
            .appending(component: "panel-compile-\(UUID().uuidString)")
        let previous = WorkspaceRoot.programmaticOverride
        WorkspaceRoot.programmaticOverride = temp
        defer {
            WorkspaceRoot.programmaticOverride = previous
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body(temp)
    }

    @discardableResult
    func invoke(_ args: [String]) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(sink: .discarding).run(namespace: "panel", args)
    }

    /// An agent artifact in the library, under an obviously synthetic name.
    @discardableResult
    private func plantAgent(
        named name: String, baseModel: String = "test/model"
    ) throws -> ModelVariantRecord {
        let directory = ModelVariantStore.directory.appending(component: name)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let artifact = ModelVariantArtifact(
            name: name, baseModelID: baseModel,
            injections: [
                .init(
                    concept: "alpha", vectorArtifactID: "runs/vec-\(name)",
                    layer: 12, alpha: 0.1)
            ],
            promptMode: "chatAssistant", qwenThinkingEnabled: false,
            temperature: 0, systemPrompt: "")
        let url = directory.appending(component: "model-variant.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(artifact).write(to: url)
        return ModelVariantRecord(url: url, artifact: artifact)
    }

    /// A two-seat SEMANTIC scenario: roles, turns, materials, no bindings.
    private func semanticScenario(
        named name: String = "widget-panel"
    ) -> MultiAgentScenario {
        PanelComposition.semanticForm(
            MultiAgentScenario(
                name: name,
                baseModelID: "",
                sharedMaterials: "Two teams share 100 widgets.",
                agents: [
                    .init(
                        id: "seat-one", name: "First", baseModelID: "",
                        systemPrompt: "You represent Team One."),
                    .init(
                        id: "seat-two", name: "Second", baseModelID: "",
                        systemPrompt: "You represent Team Two."),
                ],
                turns: [
                    .init(
                        id: "turn-propose", title: "Proposal",
                        speakerAgentID: "seat-one",
                        promptTemplate: "Propose a split.",
                        outputLabel: "proposal"),
                    .init(
                        id: "turn-review", title: "Review",
                        speakerAgentID: "seat-two",
                        promptTemplate: "Accept or decline.",
                        outputLabel: "review"),
                ]))
    }

    /// A multi-agent draft carrying the three study parameters a casting binds.
    @discardableResult
    private func makeDraft(
        named name: String, model: String = "test/model",
        temperature: Double = 0, maxTokens: Int = 512
    ) throws -> ExperimentManifest {
        var manifest = try ExperimentStore.create(
            name: name, description: "Widget panel", modelID: model,
            modelRevision: "abc123")
        manifest.studyKind = .multiAgent
        manifest.studyType = StudyIntent.multiAgent.rawValue
        manifest.temperature = temperature
        manifest.maxTokens = maxTokens
        try ExperimentStore.save(manifest)
        return manifest
    }

    private func compiledScenario(
        at relativePath: String
    ) throws -> MultiAgentScenario {
        try JSONDecoder().decode(
            MultiAgentScenario.self,
            from: Data(contentsOf: ExperimentStore.resolveProjectPath(relativePath)))
    }

    private func string(_ payload: [String: JSONValue]?, _ key: String) -> String? {
        guard case .string(let value)? = payload?[key] else { return nil }
        return value
    }

    // MARK: - 1. The casting

    @Test func castingATwoSeatPanelPinsBothScenariosIntoTheDraft() async throws {
        try await withTempWorkspace { _ in
            try makeDraft(named: "widget-study")
            let panel = try MultiAgentScenarioStore.save(semanticScenario())
            let agent = try plantAgent(named: "alpha-agent")

            let outcome = await invoke([
                "compile", "widget-panel", "--experiment", "widget-study",
                "--seat", "seat-two=\(ModelVariantStore.relativePath(for: agent))",
            ])
            #expect(outcome.exitCode == 0, "\(outcome.envelope.message)")
            #expect(outcome.envelope.state == .ready)
            #expect(outcome.envelope.changed)

            // The file the verb wrote, where the compiler always writes.
            let compiledPath = try #require(
                string(outcome.envelope.result, "compiledPath"))
            #expect(compiledPath.hasPrefix("prompts/panels/compiled/"))
            #expect(compiledPath.hasSuffix("widget-study.json"))
            let compiled = try compiledScenario(at: compiledPath)

            // Bound: every seat carries the study's model, and only the named
            // seat carries an agent.
            #expect(compiled.baseModelID == "test/model")
            #expect(compiled.maxTokens == 512)
            #expect(compiled.agents.allSatisfy { $0.baseModelID == "test/model" })
            let first = try #require(compiled.agents.first { $0.id == "seat-one" })
            #expect(first.variantArtifactPath == nil)
            let second = try #require(compiled.agents.first { $0.id == "seat-two" })
            #expect(
                second.variantArtifactPath
                    == ModelVariantStore.relativePath(for: agent))
            #expect(
                second.variantArtifactHash
                    == (try ModelVariantStore.hash(agent.url)))
            // Casting changes WHO speaks, never what the prompt says.
            #expect(compiled.turns.map(\.promptTemplate)
                == ["Propose a split.", "Accept or decline."])

            // Both pin pairs, in the manifest, in one step.
            let manifest = try ExperimentStore.load(name: "widget-study")
            #expect(manifest.multiAgentScenarioPath == compiledPath)
            #expect(
                manifest.multiAgentScenarioHash
                    == string(outcome.envelope.result, "scenarioHash"))
            #expect(
                manifest.multiAgentScenarioHash
                    == (try MultiAgentScenarioStore.hash(
                        ExperimentStore.resolveProjectPath(compiledPath))))
            // Workspace-RELATIVE, never the filesystem's own absolute spelling:
            // an absolute provenance path does not survive moving the
            // workspace.
            #expect(
                manifest.multiAgentSemanticScenarioPath
                    == "prompts/panels/widget-panel.json")
            #expect(
                manifest.multiAgentSemanticScenarioPath
                    == FineTuneStore.relativePath(for: panel.url))
            #expect(
                manifest.multiAgentSemanticScenarioHash
                    == (try MultiAgentScenarioStore.hash(panel.url)))
            // …and the study still verifies against its own bytes.
            #expect(ExperimentStore.verify(manifest).isEmpty)
        }
    }

    /// THE test. Two implementations of one transform is the defect; this is
    /// the assertion that there is only one.
    @Test func castingFromTheCLIAndFromTheUIPathAgreeByteForByte() async throws {
        try await withTempWorkspace { _ in
            try makeDraft(named: "cli-cast")
            var uiManifest = try makeDraft(named: "ui-cast")
            let panel = try MultiAgentScenarioStore.save(semanticScenario())
            let agent = try plantAgent(named: "alpha-agent")
            let artifactPath = ModelVariantStore.relativePath(for: agent)

            let outcome = await invoke([
                "compile", "widget-panel", "--experiment", "cli-cast",
                "--seat", "seat-two=\(artifactPath)",
            ])
            #expect(outcome.exitCode == 0, "\(outcome.envelope.message)")

            // The app's "save the seats" call, with the same casting built the
            // way the seat picker builds it.
            let semantic = PanelComposition.semanticForm(panel.scenario)
            let assignment = SeatAssignment(
                seatIDs: PanelComposition.seatIDs(semantic),
                occupants: [
                    "seat-one": .baseline,
                    "seat-two": SeatCasting.occupant(for: agent),
                ])
            let uiCompiled = try SeatCasting.compile(
                assignment, semantic: semantic,
                semanticPath: FineTuneStore.relativePath(for: panel.url),
                into: &uiManifest)
            try ExperimentStore.save(uiManifest)

            // Same bytes. The compiled scenario is a function of the panel, the
            // casting and the three study parameters — nothing about which
            // study it belongs to leaks into it — so the two files are equal
            // and so are their hashes.
            let cliPath = try #require(string(outcome.envelope.result, "compiledPath"))
            let cliBytes = try Data(
                contentsOf: ExperimentStore.resolveProjectPath(cliPath))
            let uiBytes = try Data(
                contentsOf: ExperimentStore.resolveProjectPath(uiCompiled.path))
            #expect(cliBytes == uiBytes)
            #expect(uiCompiled.hash == string(outcome.envelope.result, "scenarioHash"))

            // …and the manifests agree on every field a casting writes.
            let cliManifest = try ExperimentStore.load(name: "cli-cast")
            let savedUI = try ExperimentStore.load(name: "ui-cast")
            #expect(cliManifest.multiAgentScenarioHash == savedUI.multiAgentScenarioHash)
            #expect(
                cliManifest.multiAgentSemanticScenarioPath
                    == savedUI.multiAgentSemanticScenarioPath)
            #expect(
                cliManifest.multiAgentSemanticScenarioHash
                    == savedUI.multiAgentSemanticScenarioHash)
            #expect(cliManifest.studyKind == savedUI.studyKind)
            // The one field that legitimately differs is the compiled file's
            // NAME, which is slugged from the study.
            #expect(cliManifest.multiAgentScenarioPath != savedUI.multiAgentScenarioPath)
        }
    }

    @Test func unnamedSeatsStayBaselineAndFileSlugNamesTheFile() async throws {
        try await withTempWorkspace { _ in
            try makeDraft(named: "widget-study")
            try MultiAgentScenarioStore.save(semanticScenario())

            let outcome = await invoke([
                "compile", "widget-panel", "--experiment", "widget-study",
                "--file-slug", "arm-a",
            ])
            #expect(outcome.exitCode == 0, "\(outcome.envelope.message)")
            let path = try #require(string(outcome.envelope.result, "compiledPath"))
            #expect(path == "prompts/panels/compiled/arm-a.json")
            // An all-baseline casting is the CONTROL composition, not an
            // absence: every seat is bound to the model and none to an agent.
            let compiled = try compiledScenario(at: path)
            #expect(compiled.agents.allSatisfy { $0.variantArtifactPath == nil })
            #expect(compiled.agents.allSatisfy { $0.baseModelID == "test/model" })
            guard case .number(let treated)? = outcome.envelope.result?["treatedSeatCount"]
            else {
                Issue.record("no treatedSeatCount in result")
                return
            }
            #expect(treated == 0)

            // A second casting under the same slug is uniqued rather than
            // overwritten — what makes a batch of castings safe to script.
            try makeDraft(named: "widget-study-2")
            let again = await invoke([
                "compile", "widget-panel", "--experiment", "widget-study-2",
                "--file-slug", "arm-a",
            ])
            #expect(
                string(again.envelope.result, "compiledPath")
                    == "prompts/panels/compiled/arm-a-2.json")
        }
    }

    // MARK: - 2. Study parameters: defaults, then overrides

    @Test func theStudyParametersDefaultFromTheManifest() async throws {
        try await withTempWorkspace { _ in
            try makeDraft(
                named: "widget-study", model: "test/other-model", maxTokens: 1234)
            try MultiAgentScenarioStore.save(semanticScenario())

            let outcome = await invoke([
                "compile", "widget-panel", "--experiment", "widget-study",
            ])
            #expect(outcome.exitCode == 0, "\(outcome.envelope.message)")
            let compiled = try compiledScenario(
                at: try #require(string(outcome.envelope.result, "compiledPath")))
            #expect(compiled.baseModelID == "test/other-model")
            #expect(compiled.maxTokens == 1234)
            #expect(compiled.temperature == 0)
            // Nothing was overridden, so the manifest is unchanged in those
            // three fields.
            let manifest = try ExperimentStore.load(name: "widget-study")
            #expect(manifest.modelID == "test/other-model")
            #expect(manifest.maxTokens == 1234)
        }
    }

    @Test func explicitFlagsOverrideAndAreWrittenToTheManifest() async throws {
        try await withTempWorkspace { _ in
            try makeDraft(named: "widget-study", model: "test/model", maxTokens: 512)
            try MultiAgentScenarioStore.save(semanticScenario())

            let outcome = await invoke([
                "compile", "widget-panel", "--experiment", "widget-study",
                "--model", "test/override-model", "--temperature", "0.7",
                "--max-tokens", "2048",
            ])
            #expect(outcome.exitCode == 0, "\(outcome.envelope.message)")
            let compiled = try compiledScenario(
                at: try #require(string(outcome.envelope.result, "compiledPath")))
            #expect(compiled.baseModelID == "test/override-model")
            #expect(compiled.temperature == 0.7)
            #expect(compiled.maxTokens == 2048)

            // The manifest is the ONE place those three are decided, so an
            // override moves it too — a compiled scenario that disagreed with
            // its own study would be a second answer.
            let manifest = try ExperimentStore.load(name: "widget-study")
            #expect(manifest.modelID == "test/override-model")
            #expect(manifest.temperature == 0.7)
            #expect(manifest.maxTokens == 2048)
        }
    }

    @Test func aNonNumericSamplingOverrideIsAMalformedInvocation() async throws {
        try await withTempWorkspace { _ in
            try makeDraft(named: "widget-study")
            try MultiAgentScenarioStore.save(semanticScenario())
            let outcome = await invoke([
                "compile", "widget-panel", "--experiment", "widget-study",
                "--temperature", "warm",
            ])
            #expect(outcome.envelope.state == .blocked)
            #expect(outcome.envelope.exitCode == 64)
            #expect(outcome.envelope.error?.code == "usage")
            #expect(outcome.envelope.message.contains("--temperature"))
            // Nothing was written.
            let manifest = try ExperimentStore.load(name: "widget-study")
            #expect(manifest.multiAgentScenarioPath == nil)
        }
    }

    // MARK: - 3. Refusals

    /// An id the panel does not have is a MALFORMED INVOCATION, not a gate
    /// refusal: nothing is wrong with the study or the panel, the caller typed
    /// a value outside the panel's own vocabulary, and retyping fixes it —
    /// the same class an unknown `set-instruments` value lands in. So the
    /// refusal's whole job is printing the vocabulary.
    @Test func anUnknownSeatNamesThePanelsActualSeats() async throws {
        try await withTempWorkspace { _ in
            try makeDraft(named: "widget-study")
            let agent = try plantAgent(named: "alpha-agent")
            try MultiAgentScenarioStore.save(semanticScenario())

            let outcome = await invoke([
                "compile", "widget-panel", "--experiment", "widget-study",
                "--seat", "judge-2=\(ModelVariantStore.relativePath(for: agent))",
            ])
            #expect(outcome.envelope.state == .blocked)
            #expect(outcome.envelope.exitCode == 64)
            #expect(outcome.envelope.error?.code == "usage")
            #expect(outcome.envelope.message.contains("no seat 'judge-2'"))
            let repair = try #require(outcome.envelope.error?.repairAction)
            #expect(repair.contains("seat-one"))
            #expect(repair.contains("seat-two"))
            // No file, no pin: a refused casting writes nothing at all.
            #expect(
                !FileManager.default.fileExists(
                    atPath: PanelComposition.compiledDirectory.path))
            #expect(
                try ExperimentStore.load(name: "widget-study")
                    .multiAgentScenarioPath == nil)
        }
    }

    /// A `--seat` value that is not on disk is the pin-verbs' fact, so it gets
    /// the pin-verbs' gate: author (or promote) the input, then re-run.
    @Test func aSeatArtifactThatIsNotThereRefusesWithMissingPrerequisite()
        async throws
    {
        try await withTempWorkspace { root in
            try makeDraft(named: "widget-study")
            try MultiAgentScenarioStore.save(semanticScenario())

            let outcome = await invoke([
                "compile", "widget-panel", "--experiment", "widget-study",
                "--seat", "seat-two=runs/model-variants/nope/model-variant.json",
            ])
            #expect(outcome.envelope.state == .refused)
            #expect(outcome.envelope.exitCode == 65)
            #expect(
                outcome.envelope.error?.gate
                    == LifecycleGate.missingPrerequisite.rawValue)
            #expect(outcome.envelope.message.hasPrefix("agent artifact not found:"))
            #expect(
                outcome.envelope.error?.repairAction.contains("steerlab-cli ")
                    == true)
            // Human mode keeps exit 1, as every lifecycle gate but `data check`
            // does.
            #expect(outcome.exitCode == 1)

            // A file that exists but is not an agent artifact is the same gate
            // with a different sentence — telling an agent to author a file
            // that is already there would send it in a circle.
            let stray = root.appending(
                components: "runs", "model-variants", "stray", "model-variant.json")
            try FileManager.default.createDirectory(
                at: stray.deletingLastPathComponent(), withIntermediateDirectories: true)
            try "{\"not\": \"an artifact\"}".write(
                to: stray, atomically: true, encoding: .utf8)
            let malformed = await invoke([
                "compile", "widget-panel", "--experiment", "widget-study",
                "--seat", "seat-two=runs/model-variants/stray/model-variant.json",
            ])
            #expect(malformed.envelope.state == .refused)
            #expect(malformed.envelope.message.hasPrefix("not an agent artifact:"))
        }
    }

    /// The immutability gate, and the property that makes it worth checking
    /// separately from every other `statusImmutable` site: the compiled file is
    /// written BEFORE the manifest, so a refusal that arrived at save time
    /// would leave an orphan casting in `prompts/panels/compiled/` for a study
    /// that never adopted it.
    @Test func aFrozenStudyRefusesBeforeAnythingIsCompiled() async throws {
        try await withTempWorkspace { _ in
            var manifest = try makeDraft(named: "widget-study")
            try MultiAgentScenarioStore.save(semanticScenario())
            manifest.status = .frozen
            manifest.freezeHash = String(repeating: "a", count: 64)
            try ExperimentStore.save(manifest)

            let outcome = await invoke([
                "compile", "widget-panel", "--experiment", "widget-study",
            ])
            #expect(outcome.envelope.state == .refused)
            #expect(outcome.envelope.exitCode == 65)
            #expect(
                outcome.envelope.error?.gate == LifecycleGate.statusImmutable.rawValue)
            #expect(outcome.envelope.message.contains("duplicate it to iterate"))
            // The repair is THIS verb's argument shape, not `experiment
            // <the verb you just ran>` — a repair that is not runnable sends an
            // agent in a circle.
            let repair = try #require(outcome.envelope.error?.repairAction)
            #expect(
                repair.contains("steerlab-cli experiment duplicate widget-study "
                    + "widget-study-v2"))
            #expect(
                repair.contains(
                    "steerlab-cli panel compile widget-panel --experiment "
                        + "widget-study-v2"))
            #expect(outcome.exitCode == 1)

            // Nothing was compiled: no orphan file.
            #expect(
                !FileManager.default.fileExists(
                    atPath: PanelComposition.compiledDirectory.path))
        }
    }

    /// A hand-authored panel carries its casting INSIDE the file, and other
    /// studies may pin the same file, so re-casting it from a study would
    /// rewrite an input under them. The app's Seats section states the same
    /// rule as `legacyBound`; the CLI states it as a refusal.
    @Test func aBoundPanelIsRefusedRatherThanSilentlyStripped() async throws {
        try await withTempWorkspace { _ in
            try makeDraft(named: "widget-study")
            var bound = semanticScenario(named: "bound-panel")
            bound.baseModelID = "test/model"
            bound.agents = bound.agents.map {
                var seat = $0
                seat.baseModelID = "test/model"
                return seat
            }
            try MultiAgentScenarioStore.save(bound)

            let outcome = await invoke([
                "compile", "bound-panel", "--experiment", "widget-study",
            ])
            #expect(outcome.envelope.state == .blocked)
            #expect(outcome.envelope.exitCode == 64)
            #expect(outcome.envelope.message.contains("is not semantic"))
            #expect(
                outcome.envelope.error?.repairAction.contains("steerlab-cli panel list")
                    == true)
        }
    }

    @Test func aMissingExperimentAndAMissingPanelAreBothNotFound() async throws {
        try await withTempWorkspace { _ in
            try MultiAgentScenarioStore.save(semanticScenario())
            let noStudy = await invoke([
                "compile", "widget-panel", "--experiment", "no-such-study",
            ])
            #expect(noStudy.envelope.state == .notFound)
            #expect(noStudy.envelope.exitCode == 66)

            try makeDraft(named: "widget-study")
            let noPanel = await invoke([
                "compile", "no-such-panel", "--experiment", "widget-study",
            ])
            #expect(noPanel.envelope.state == .notFound)
            #expect(noPanel.envelope.exitCode == 66)
            #expect(noPanel.envelope.message.contains("no panel 'no-such-panel'"))
            #expect(
                noPanel.envelope.error?.repairAction.contains("steerlab-cli panel list")
                    == true)
        }
    }

    @Test func aMalformedSeatPairAndAMissingExperimentFlagAreUsageErrors()
        async throws
    {
        try await withTempWorkspace { _ in
            try makeDraft(named: "widget-study")
            try MultiAgentScenarioStore.save(semanticScenario())

            // No `--experiment` at all.
            let noFlag = await invoke(["compile", "widget-panel"])
            #expect(noFlag.envelope.exitCode == 64)
            #expect(noFlag.envelope.message.hasPrefix("usage: panel compile"))

            // A `--seat` value with no `=`.
            let noEquals = await invoke([
                "compile", "widget-panel", "--experiment", "widget-study",
                "--seat", "seat-two",
            ])
            #expect(noEquals.envelope.exitCode == 64)
            #expect(noEquals.envelope.message.hasPrefix("usage: panel compile"))

            // A flag the verb does not declare: refused before any work, in
            // both output modes.
            let typo = await invoke([
                "compile", "widget-panel", "--experiment", "widget-study",
                "--seet", "seat-two=x",
            ])
            #expect(typo.exitCode == 64)
            #expect(typo.envelope.error?.code == ExperimentCLIUsageError.code)
        }
    }

    // MARK: - 4. The declaration a casting implies

    /// A panel scenario is read only by the multi-agent run path, so pinning
    /// one into a `modelOutput` study leaves a study that looks armed and
    /// measures nothing — the 2026-08-11 inert-conditions shape from the other
    /// side. The verb makes the declaration (there is no headless setter for
    /// it) and says so; it is never silent.
    @Test func castingDeclaresTheStudyMultiAgentAndReportsIt() async throws {
        try await withTempWorkspace { _ in
            var manifest = try makeDraft(named: "widget-study")
            manifest.studyKind = .modelOutput
            manifest.studyType = StudyIntent.agentComparison.rawValue
            try ExperimentStore.save(manifest)
            try MultiAgentScenarioStore.save(semanticScenario())

            let outcome = await invoke([
                "compile", "widget-panel", "--experiment", "widget-study",
            ])
            #expect(outcome.exitCode == 0, "\(outcome.envelope.message)")
            guard case .bool(let declared)? =
                outcome.envelope.result?["studyKindDeclared"]
            else {
                Issue.record("no studyKindDeclared in result")
                return
            }
            #expect(declared)
            let after = try ExperimentStore.load(name: "widget-study")
            #expect(after.studyKind == .multiAgent)
            #expect(after.studyType == StudyIntent.multiAgent.rawValue)

            // A study that was already multi-agent reports false rather than
            // claiming a change it did not make.
            try makeDraft(named: "already-panel")
            let second = await invoke([
                "compile", "widget-panel", "--experiment", "already-panel",
            ])
            guard case .bool(let again)? =
                second.envelope.result?["studyKindDeclared"]
            else {
                Issue.record("no studyKindDeclared in result")
                return
            }
            #expect(!again)
        }
    }

    // MARK: - 5. The envelope

    @Test func theEnvelopeCarriesTheCastingAsData() async throws {
        try await withTempWorkspace { _ in
            try makeDraft(named: "widget-study")
            try MultiAgentScenarioStore.save(semanticScenario())
            let agent = try plantAgent(named: "alpha-agent")

            let outcome = await invoke([
                "compile", "widget-panel", "--experiment", "widget-study",
                "--seat", "seat-two=\(ModelVariantStore.relativePath(for: agent))",
            ])
            #expect(outcome.envelope.verb == "panel compile")
            #expect(outcome.envelope.engine == SteerLabCLIEnvelope.localEngine)
            let result = try #require(outcome.envelope.result)
            for key in [
                "experiment", "panel", "compiledPath", "scenarioHash",
                "semanticScenarioPath", "semanticScenarioHash", "modelID",
                "temperature", "maxTokens", "studyKind", "studyKindDeclared",
                "seatCount", "treatedSeatCount", "seats",
            ] {
                #expect(result[key] != nil, "result is missing \(key)")
            }
            // FULL hashes: an elided one is a provenance hole in a machine
            // document.
            #expect(string(outcome.envelope.result, "scenarioHash")?.count == 64)

            guard case .array(let seats)? = result["seats"] else {
                Issue.record("no seats array in result")
                return
            }
            #expect(seats.count == 2)
            guard case .object(let treatedSeat) = seats[1] else {
                Issue.record("seat entry is not an object")
                return
            }
            #expect(treatedSeat["seat"] == .string("seat-two"))
            #expect(treatedSeat["occupant"] == .string("agent"))
            #expect(treatedSeat["agentName"] == .string("alpha-agent"))
            #expect(
                treatedSeat["artifactHash"]
                    == .string(try ModelVariantStore.hash(agent.url)))
            // The baseline seat is a CONDITION, not an absence: it is present
            // with explicit nulls rather than omitted.
            guard case .object(let baselineSeat) = seats[0] else {
                Issue.record("seat entry is not an object")
                return
            }
            #expect(baselineSeat["occupant"] == .string("baseline"))
            #expect(baselineSeat["artifactPath"] == .null)

            // Exactly one JSON document, and it round-trips.
            let text = try outcome.envelope.jsonText()
            #expect(text.hasSuffix("\n"))
            _ = try JSONSerialization.jsonObject(with: Data(text.utf8))
        }
    }

    @Test func helpRunsNothingAndNamesEverySeatFlag() async throws {
        try await withTempWorkspace { _ in
            let outcome = await invoke(["compile", "--help"])
            #expect(outcome.exitCode == 0)
            let page = ExperimentCLIHelp.text(
                for: try #require(
                    ExperimentCLIParser.spec(namespace: "panel", verb: "compile")))
            for flag in ["--experiment", "--seat", "--model", "--temperature",
                         "--max-tokens", "--file-slug", "--json", "--help"] {
                #expect(page.contains(flag), "the help page omits \(flag)")
            }
            // `--experiment` is required, and the synopsis says so by leaving
            // it unbracketed.
            #expect(page.contains("--experiment <name>"))
            #expect(!page.contains("[--experiment"))
        }
    }

    // MARK: - 6. The read verbs, which came along

    @Test func listAndCheckStillAnswerAndNowReportSemanticness() async throws {
        try await withTempWorkspace { _ in
            let empty = await invoke(["list"])
            #expect(empty.exitCode == 0)
            #expect(empty.envelope.message.hasPrefix("no panels in "))

            try MultiAgentScenarioStore.save(semanticScenario())
            let listed = await invoke(["list"])
            #expect(listed.exitCode == 0)
            guard case .array(let panels)? = listed.envelope.result?["panels"],
                case .object(let first) = panels[0]
            else {
                Issue.record("no panels array in result")
                return
            }
            #expect(first["name"] == .string("widget-panel"))
            #expect(first["semantic"] == .bool(true))

            // A SEMANTIC panel is structurally not runnable — the run-time
            // validator rejects a seat with no model, deliberately, so an
            // uncast panel can never quietly generate from whatever model
            // happened to be resident. `check` therefore refuses it, which is
            // the pre-existing behaviour and the reason `compile` exists.
            let semanticCheck = await invoke(["check", "widget-panel"])
            #expect(semanticCheck.exitCode != 0)

            // A bound panel is what `check` validates.
            var bound = semanticScenario(named: "bound-panel")
            bound.baseModelID = "test/model"
            bound.agents = bound.agents.map {
                var seat = $0
                seat.baseModelID = "test/model"
                return seat
            }
            _ = try MultiAgentScenarioStore.save(bound)
            let checked = await invoke(["check", "bound-panel"])
            #expect(checked.exitCode == 0, "\(checked.envelope.message)")
            #expect(checked.envelope.message.hasPrefix("valid: 2 agents, 2 turns"))
            #expect(checked.envelope.result?["panelAdvisories"] != nil)
            #expect(checked.envelope.result?["semantic"] == .bool(false))

            let missing = await invoke(["check", "no-such-panel"])
            #expect(missing.envelope.state == .notFound)
            #expect(missing.envelope.exitCode == 66)
            #expect(missing.envelope.message.contains("no panel 'no-such-panel'"))
        }
    }
}
