import CryptoKit
import Foundation
import MLX
import MLXLMCommon
import SteeringKit

public struct MultiAgentTurnResult: Codable, Identifiable, Sendable, Equatable {
    public var id: String { turnID }
    public let turnID: String
    public let turnIndex: Int
    public let title: String
    public let speakerAgentID: String
    public let speakerName: String
    /// Revision of the model that actually generated this turn (pinned when
    /// the study pins one, else the local cache's resolved commit). Optional
    /// so pre-existing turns.jsonl files still decode.
    public let modelRevision: String?
    public let prompt: String
    public let output: String
    public let outputLabel: String
    public let routedAgentIDs: [String]
    /// Which independent play-through of the scenario this turn belongs to.
    /// Optional so pre-existing turns.jsonl files still decode; absent means
    /// the single-replicate era, i.e. 0.
    public let replicateIndex: Int?
    /// Sampling temperature this turn actually generated at. Optional for the
    /// same reason; absent means the greedy-only era, i.e. 0.
    public let temperature: Double?
    /// The declared endpoint's parse, stamped at write time (Wave-2
    /// contract). nil ⇒ key omitted, which is exactly the case where the turn
    /// declared no endpoint — a turn that declared one and could not be read
    /// carries `{"name": …, "value": null, "unparsed": true}`, never a guess.
    /// Server twin: the `endpoint` key on `multi_agent.run_scenario`'s turn
    /// record.
    public let endpoint: TurnEndpointStamp?
    /// Which renderer produced `prompt`: `"contract-v1"` or `"template-v1"`.
    /// Optional so pre-contract turns.jsonl files still decode; absent means a
    /// run from before the two renderers existed, which was always the
    /// template one. Stamped here rather than derived later because the
    /// scenario file can be edited after the run and the record cannot.
    public let promptRenderer: String?
    /// The turn's voice lint (spec §5), stamped at write time on every
    /// generated turn. Optional so pre-lint turns.jsonl files still decode;
    /// absent means a run from before the lint existed, never "clean".
    /// Server twin: the `voiceLint` key on `multi_agent.run_scenario`'s turn
    /// record.
    public let voiceLint: VoiceLintStamp?
    /// WHICH LEVELS composed the system prompt this turn generated under
    /// (2026-08-24 casting ruling): the cast agent artifact's persona and the
    /// seat's cast-entry role text. Optional so pre-composition turns.jsonl
    /// files still decode, and so an absent key stays absent rather than
    /// claiming both levels were empty; every turn this engine generates
    /// stamps it. Server twin: the `systemPromptComposition` key on
    /// `multi_agent.run_scenario`'s turn record.
    public let systemPromptComposition: PanelSystemPromptCompositionStamp?
    /// Why this turn's generation ended (`ExperimentTasks.FinishReason`),
    /// classified against the TURN's own token budget — the only place that
    /// number is known, which is why the runner stamps it at write time
    /// instead of leaving the flattener to re-derive it against a manifest
    /// cap the generation never ran under. Optional so pre-existing
    /// turns.jsonl files still decode; absent means a turn nobody classified,
    /// never a turn that finished. Server twin: the `finishReason` key on
    /// `multi_agent.run_scenario`'s turn record.
    public let finishReason: String?

    public init(
        turnID: String, turnIndex: Int, title: String, speakerAgentID: String,
        speakerName: String, modelRevision: String?, prompt: String, output: String,
        outputLabel: String, routedAgentIDs: [String],
        replicateIndex: Int? = nil, temperature: Double? = nil,
        endpoint: TurnEndpointStamp? = nil, promptRenderer: String? = nil,
        voiceLint: VoiceLintStamp? = nil,
        systemPromptComposition: PanelSystemPromptCompositionStamp? = nil,
        finishReason: String? = nil
    ) {
        self.turnID = turnID
        self.turnIndex = turnIndex
        self.title = title
        self.speakerAgentID = speakerAgentID
        self.speakerName = speakerName
        self.modelRevision = modelRevision
        self.prompt = prompt
        self.output = output
        self.outputLabel = outputLabel
        self.routedAgentIDs = routedAgentIDs
        self.replicateIndex = replicateIndex
        self.temperature = temperature
        self.endpoint = endpoint
        self.promptRenderer = promptRenderer
        self.voiceLint = voiceLint
        self.systemPromptComposition = systemPromptComposition
        self.finishReason = finishReason
    }
}

public struct MultiAgentRunReport: Codable, Sendable, Equatable {
    public let scenarioName: String
    public let conditionName: String
    public let strippedInterventions: Bool
    public let scenarioHash: String
    public let baseModelID: String
    public let runDirectory: String
    public let startedAt: String
    public let completedAt: String
    public let turnCount: Int
    public let warnings: [MultiAgentRunWarning]
    /// Sampling provenance. Optional so pre-existing report.json files still
    /// decode; absent means the greedy, single-replicate era.
    public let temperature: Double?
    public let replicateIndex: Int?
    /// "greedy" locally at temperature 0; "seedInert" for a warm LOCAL run —
    /// varied but not re-runnable, because MLX exposes no per-run sampling
    /// seed. The server stamps "derivedSHA256" for its warm runs.
    public let seedPolicy: String?
}

public struct MultiAgentRunWarning: Codable, Sendable, Equatable {
    public let agentName: String
    public let variantArtifactPath: String?
    public let expectedHash: String?
    public let actualHash: String?
    public let message: String
}

public struct MultiAgentRunErrorReport: Codable, Sendable, Equatable {
    public let scenarioName: String
    public let runDirectory: String
    public let failedAt: String
    public let turnIndex: Int?
    public let turnTitle: String?
    public let speakerName: String?
    public let error: String
}

public enum MultiAgentRunProgress: Sendable {
    case runDirectory(String)
    case turnStarted(index: Int, title: String, speaker: String)
    case turnChunk(index: Int, title: String, speaker: String, output: String)
    case turnCompleted(MultiAgentTurnResult)
    case runWarning(MultiAgentRunWarning)
    case runFailed(MultiAgentRunErrorReport)
}

public typealias MultiAgentRunProgressHandler = @Sendable (MultiAgentRunProgress) async -> Void

public enum MultiAgentRunner {
    private struct RuntimeAgent {
        var spec: MultiAgentScenario.Agent
        var context: [String] = []
    }

    private struct RuntimeModel {
        let container: ModelContainer
        let revision: String?
    }

    @discardableResult
    public static func run(
        scenario: MultiAgentScenario,
        scenarioURL: URL? = nil,
        runDirectory requestedRunDirectory: URL? = nil,
        conditionName: String = "configured",
        stripInterventions: Bool = false,
        defaultRevision: String? = nil,
        temperature: Double? = nil,
        replicateIndex: Int = 0,
        progress: MultiAgentRunProgressHandler? = nil
    ) async throws -> URL {
        try validate(scenario)
        // Positive temperature is supported here (mlx-swift-lm samples warm
        // through CategoricalSampler/TopPSampler — the Playground has always
        // used it). What local MLX cannot do is SEED that sampling:
        // GenerateParameters exposes no per-run seed, so a warm local
        // transcript is varied but not re-runnable, and every record stays
        // stamped `seedInert: true`. That is the documented substrate
        // difference — stochastic evidence comes from the server, which seeds
        // per turn — not a reason to refuse an exploratory warm run.
        // `temperature` overrides the scenario's authoring value: a STUDY
        // manifest owns measured-run sampling policy.
        let effectiveTemperature = temperature ?? scenario.temperature
        guard effectiveTemperature >= 0 else {
            throw ExperimentError(
                reason: "temperature must be >= 0, got \(effectiveTemperature)")
        }

        MLX.Memory.cacheLimit = 2 * 1024 * 1024 * 1024
        let runDirectory =
            if let requestedRunDirectory {
                requestedRunDirectory
            } else {
                try makeRunDirectory(scenario: scenario)
            }
        try FileManager.default.createDirectory(
            at: runDirectory, withIntermediateDirectories: true)
        await progress?(.runDirectory(runDirectory.path))

        let scenarioHash =
            if let scenarioURL {
                (try? MultiAgentScenarioStore.hash(scenarioURL)) ?? hash(scenario)
            } else {
                hash(scenario)
            }
        let scenarioCopy = runDirectory.appending(component: "scenario.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(scenario).write(to: scenarioCopy, options: .atomic)

        // Turn-level durability, matching the server. turns.jsonl used to be
        // written once at the END, so a cancelled or crashed run left NOTHING
        // on disk — the F2 cancellation note promised partial turns that did
        // not exist. Open it now, append as each turn lands, and replay
        // whatever a previous attempt finished.
        let turnsURL = runDirectory.appending(component: "turns.jsonl")
        // Truncate BEFORE loading. The other order loses data, and it is MORE
        // likely here than on the server because the record and its newline
        // are two separate writes: a record whose JSON landed but whose
        // newline did not is complete and parses, so loading first admits it
        // to `completed` — and then truncation, which can only see "no
        // trailing newline", deletes it. The turn would be neither on disk nor
        // regenerated. This way the worst case is a regenerated turn.
        truncateTornTail(at: turnsURL)
        let completed = completedTurns(at: turnsURL)
        if !completed.isEmpty {
            print("resuming transcript: \(completed.count) turn(s) already complete")
        }
        if !FileManager.default.fileExists(atPath: turnsURL.path) {
            FileManager.default.createFile(atPath: turnsURL.path, contents: nil)
        }
        let turnsHandle = try FileHandle(forWritingTo: turnsURL)
        defer { try? turnsHandle.close() }
        try turnsHandle.seekToEnd()
        let turnEncoder = JSONEncoder()
        turnEncoder.outputFormatting = [.sortedKeys]

        var agents = Dictionary(
            uniqueKeysWithValues: scenario.agents.map { ($0.id, RuntimeAgent(spec: $0)) })
        var outputsByLabel: [String: String] = [:]
        var models: [ExperimentTasks.LoadedModelKey: RuntimeModel] = [:]
        var results: [MultiAgentTurnResult] = []
        var warnings: [MultiAgentRunWarning] = []
        var emittedWarningKeys = Set<String>()
        let startedAt = ISO8601DateFormatter().string(from: Date())

        for (index, turn) in scenario.turns.enumerated() {
            do {
                try Task.checkCancellation()
                guard let speaker = agents[turn.speakerAgentID] else {
                    throw ExperimentError(reason: "turn '\(turn.title)' has no valid speaker")
                }
                // Finished on a previous attempt: replay its recorded output
                // into the context and label state so later turns see exactly
                // what they would have, and generate nothing.
                if let replayed = completed[turn.id] {
                    let label = normalizedOutputLabel(turn: turn, index: index)
                    outputsByLabel[label] = replayed.output
                    for id in routedAgentIDs(for: turn, agents: scenario.agents) {
                        agents[id]?.context.append(
                            contextEntry(
                                turn: turn, outputLabel: label, speaker: speaker.spec,
                                output: replayed.output, reader: id))
                    }
                    results.append(replayed)
                    await progress?(.turnCompleted(replayed))
                    continue
                }
                let (
                    variant, modelID, promptMode, systemPrompt, qwenThinking,
                    injections, turnWarnings, systemComposition
                ) =
                    try runtimeSettings(for: speaker.spec, stripInterventions: stripInterventions)
                for warning in turnWarnings {
                    let key = warningKey(warning)
                    guard emittedWarningKeys.insert(key).inserted else { continue }
                    warnings.append(warning)
                    await progress?(.runWarning(warning))
                }
                // A variant pins its own base revision; bare agents inherit
                // the study's pinned revision instead of loading unpinned.
                let runtime = try await model(
                    for: modelID,
                    revision: variant?.baseRevision ?? defaultRevision,
                    models: &models)
                let turnMaxTokens = max(1, turn.maxTokens ?? scenario.maxTokens)
                let prompt = renderPrompt(
                    scenario: scenario,
                    turn: turn,
                    speakerName: speaker.spec.name,
                    speakerContext: speaker.context.joined(separator: "\n\n"),
                    outputsByLabel: outputsByLabel)

                await progress?(
                    .turnStarted(index: index + 1, title: turn.title, speaker: speaker.spec.name))
                let adapter: LoRAContainer?
                if let variant {
                    adapter = try await ExperimentTasks.loadAdapter(variant, into: runtime.container)
                } else {
                    adapter = nil
                }

                // `generateMeasured`, not `generate`: the plain wrapper
                // discards the stream's own stop reason, and a panel turn's
                // budget is the TURN's — so this is the only place the
                // classification can honestly be made.
                let measured: ExperimentTasks.MeasuredGeneration
                do {
                    measured = try await ExperimentTasks.generateMeasured(
                        runtime.container,
                        prompt: prompt,
                        modelID: modelID,
                        maxTokens: turnMaxTokens,
                        temperature: effectiveTemperature,
                        injections: injections,
                        promptMode: promptMode,
                        systemPrompt: systemPrompt,
                        qwenThinkingEnabled: qwenThinking
                    ) { output in
                        await progress?(
                            .turnChunk(
                                index: index + 1,
                                title: turn.title,
                                speaker: speaker.spec.name,
                                output: output))
                    }
                } catch {
                    if let adapter {
                        await ExperimentTasks.unloadAdapter(adapter, from: runtime.container)
                    }
                    try? await ExperimentTasks.setInterventions(runtime.container, [])
                    throw error
                }
                if let adapter {
                    await ExperimentTasks.unloadAdapter(adapter, from: runtime.container)
                }
                try await ExperimentTasks.setInterventions(runtime.container, [])
                try Task.checkCancellation()

                let output = measured.text
                let routed = routedAgentIDs(for: turn, agents: scenario.agents)
                let outputLabel = normalizedOutputLabel(turn: turn, index: index)
                outputsByLabel[outputLabel] = output
                for id in routed {
                    agents[id]?.context.append(
                        contextEntry(
                            turn: turn,
                            outputLabel: outputLabel,
                            speaker: speaker.spec,
                            output: output,
                            reader: id))
                }

                let result = MultiAgentTurnResult(
                    turnID: turn.id,
                    turnIndex: index + 1,
                    title: turn.title,
                    speakerAgentID: speaker.spec.id,
                    speakerName: speaker.spec.name,
                    modelRevision: runtime.revision,
                    prompt: prompt,
                    output: output,
                    outputLabel: outputLabel,
                    routedAgentIDs: routed,
                    replicateIndex: replicateIndex,
                    temperature: effectiveTemperature,
                    // Parsed HERE, at write time, from the text this turn just
                    // produced — one parse, one place, and the stamp lands on
                    // the same fsync as the output it describes. Nothing
                    // downstream re-reads the prose.
                    endpoint: endpointStamp(for: turn, output: output),
                    promptRenderer: promptRenderer(for: turn),
                    // Same seam, same fsync: the voice stamp describes the
                    // text it lands beside.
                    voiceLint: voiceLintStamp(
                        for: turn, output: output, agents: scenario.agents),
                    // …and WHICH LEVELS composed the system prompt this turn
                    // actually generated under. Built with the effective text
                    // in `runtimeSettings`, so the stamp and the arming can
                    // never describe two different things.
                    systemPromptComposition: systemComposition,
                    // The stream's own account of why this turn ended,
                    // against the TURN's budget. Same seam, same fsync.
                    finishReason: measured.finishReason)
                results.append(result)
                // Flush BEFORE the next turn starts: an unflushed transcript
                // is a transcript replayed from turn 1.
                try turnsHandle.write(contentsOf: turnEncoder.encode(result))
                try turnsHandle.write(contentsOf: Data("\n".utf8))
                try turnsHandle.synchronize()
                await progress?(.turnCompleted(result))
            } catch is CancellationError {
                // F2: a cancelled run is not a failed run. Stamping error.json
                // made a deliberate stop indistinguishable from a crash, and
                // left it visible to analyze/evaluate as if it were a
                // measurement — the study runner's cancellation note exists
                // for exactly this. Completed turns are already on disk —
                // they are appended as they land — so re-running this
                // scenario resumes from the last one.
                writeCancellationNote(
                    scenario: scenario, runDirectory: runDirectory,
                    turnIndex: index + 1, turnTitle: turn.title)
                throw CancellationError()
            } catch {
                let speakerName = agents[turn.speakerAgentID]?.spec.name
                let report = writeFailure(
                    scenario: scenario,
                    runDirectory: runDirectory,
                    turnIndex: index + 1,
                    turnTitle: turn.title,
                    speakerName: speakerName,
                    error: error)
                await progress?(.runFailed(report))
                throw error
            }
        }

        let completedAt = ISO8601DateFormatter().string(from: Date())
        let report = MultiAgentRunReport(
            scenarioName: scenario.name,
            conditionName: conditionName,
            strippedInterventions: stripInterventions,
            scenarioHash: scenarioHash,
            baseModelID: scenario.baseModelID,
            runDirectory: runDirectory.path,
            startedAt: startedAt,
            completedAt: completedAt,
            turnCount: results.count,
            warnings: warnings,
            temperature: effectiveTemperature,
            replicateIndex: replicateIndex,
            seedPolicy: effectiveTemperature > 0 ? "seedInert" : "greedy")
        try encoder.encode(report).write(to: runDirectory.appending(component: "report.json"))

        // turns.jsonl is already complete — it was appended turn by turn.
        try markdownTranscript(scenario: scenario, results: results).write(
            to: runDirectory.appending(component: "transcript.md"),
            atomically: true,
            encoding: .utf8)
        return runDirectory
    }

    /// Authoring problems that are real but must not block a draft (plan F1),
    /// the twin of the server's `multi_agent.advisories`.
    ///
    /// All three fail SILENTLY today, which is the worst outcome — the run
    /// completes and the prompts are quietly wrong: a duplicate output label
    /// means the later turn wins every `{{outputs.X}}` interpolation; a label
    /// colliding with the `turn_<n>` default does the same; and an
    /// `{{outputs.X}}` naming a label no EARLIER turn produces is left
    /// verbatim, so the model is shown literal placeholder text.
    ///
    /// Advisory while drafting, freeze gate at the firewall: an
    /// authoring-time hard stop would block a researcher mid-thought for
    /// something freeze catches anyway, with better information.
    ///
    /// A turn may DECLARE the cross-routing reads it makes on purpose
    /// (`acknowledgedInputs`), silencing the private-read advisory for
    /// exactly those labels; `acknowledgmentAdvisories` then keeps the
    /// declarations honest. Emission order is part of the cross-engine
    /// contract: scenario-level fence first, then per turn in turn order —
    /// unknown `{{outputs.X}}` reference, unacknowledged private read,
    /// acknowledgment hygiene, strict-format budget, duplicate label.
    public static func advisories(_ scenario: MultiAgentScenario) -> [String] {
        var out: [String] = []
        // Scenario-level first: the fence marker the contract renderer builds
        // its blocks from, appearing inside the material those blocks wrap.
        if scenario.sharedMaterials.contains(contractFence) {
            out.append(
                "shared materials contain the fence marker '\(contractFence)' — a "
                    + "contract turn's fenced blocks would be ambiguous, and the model "
                    + "cannot tell where the record ends")
        }
        var produced = Set<String>()
        var producers: [String: MultiAgentScenario.Turn] = [:]
        var seenLabels: [String: String] = [:]
        for (index, turn) in scenario.turns.enumerated() {
            let explicit = turn.outputLabel.trimmingCharacters(in: .whitespacesAndNewlines)
            let label = explicit.isEmpty ? "turn_\(index + 1)" : explicit
            for reference in outputReferences(in: turn.promptTemplate)
            where !produced.contains(reference) {
                out.append(
                    "turn '\(turn.title)' interpolates {{outputs.\(reference)}} but "
                        + "no earlier turn produces the label '\(reference)' — the "
                        + "placeholder will be sent to the model verbatim")
            }
            // A read of an output this speaker was never routed. Legal —
            // `{{outputs.X}}` and contract inputs bypass routing by design —
            // but it is also exactly what a leaking private turn looks like,
            // and nothing else in the pipeline would ever say so.
            //
            // Routed MEMBERSHIP is the whole test. The spec's extra "and
            // routing ≠ all" clause is vacuous — `all` routes to every agent,
            // so the speaker is always a member — and spelling it out would
            // suggest there is a case where the two disagree. The Python twin
            // tests membership alone for the same reason.
            //
            // …unless the turn DECLARED the read (`acknowledgedInputs`), in
            // which case the design has already answered the question this
            // advisory asks, for exactly those labels and no others.
            let readLabels = outputReferences(in: turn.promptTemplate)
                + (turn.contract?.inputs ?? [])
            let acknowledged = Set(turn.acknowledgedInputs ?? [])
            for reference in readLabels where !acknowledged.contains(reference) {
                guard let producer = producers[reference],
                    !routedAgentIDs(for: producer, agents: scenario.agents)
                        .contains(turn.speakerAgentID)
                else { continue }
                out.append(
                    "turn '\(turn.title)' reads the output '\(reference)' of turn "
                        + "'\(producer.title)', which was never routed to this turn's "
                        + "speaker — check whether a private turn is leaking")
            }
            out += acknowledgmentAdvisories(
                scenario: scenario, turn: turn, readLabels: readLabels,
                producers: producers)
            // A strict-format turn with room to append a whole document is how
            // a disposition line came back with an opinion stapled to it.
            if let endpoint = turn.endpoint {
                let budget = turn.maxTokens ?? scenario.maxTokens
                if budget > strictFormatTokenBudget {
                    out.append(
                        "turn '\(turn.title)' declares the endpoint '\(endpoint.name)' but "
                            + "allows \(budget) tokens — a strict-format turn with room to "
                            + "append an opinion invites format contamination")
                }
            }
            if let owner = seenLabels[label] {
                let kind = explicit.isEmpty ? "default output label" : "output label"
                out.append(
                    "turn '\(turn.title)' reuses the \(kind) '\(label)' (also on "
                        + "'\(owner)') — the later turn silently wins every "
                        + "{{outputs.\(label)}} interpolation")
            }
            seenLabels[label] = turn.title
            produced.insert(label)
            producers[label] = turn
        }
        return out
    }

    /// Keeps `acknowledgedInputs` honest (spec §4.1), the twin of the
    /// server's `_acknowledgment_advisories`.
    ///
    /// An acknowledgment is a claim about the DESIGN — "this turn reads a
    /// colleague's private memo on purpose" — so it has to stay attached to a
    /// real read, or it is a silencer left behind after the read it silenced
    /// was edited away. Two ways it goes wrong, both advisory, both reported
    /// in the order the author declared them:
    ///
    /// - STALE: acknowledged but not read (or read but produced by nothing at
    ///   all, which is the same thing here — no advisory could ever have
    ///   fired for it);
    /// - NO-OP: read, but the producer's routing already includes this
    ///   speaker, so nothing would have fired. Harmless, but it says the
    ///   author believes a read is private when it is not, and that belief is
    ///   worth correcting before it becomes a design.
    ///
    /// Duplicates are not collapsed, for the same reason repeated reads are
    /// not: each declaration is a line in the file to fix.
    static func acknowledgmentAdvisories(
        scenario: MultiAgentScenario,
        turn: MultiAgentScenario.Turn,
        readLabels: [String],
        producers: [String: MultiAgentScenario.Turn]
    ) -> [String] {
        var out: [String] = []
        for label in turn.acknowledgedInputs ?? [] {
            guard readLabels.contains(label), let producer = producers[label] else {
                out.append(
                    "turn '\(turn.title)' acknowledges the read of '\(label)' but does "
                        + "not read it — remove the stale acknowledgment")
                continue
            }
            if routedAgentIDs(for: producer, agents: scenario.agents)
                .contains(turn.speakerAgentID)
            {
                out.append(
                    "turn '\(turn.title)' acknowledges the read of '\(label)', but its "
                        + "speaker was routed that output anyway — the acknowledgment "
                        + "is doing nothing")
            }
        }
        return out
    }

    /// The marker the contract renderer builds its block fences from.
    static let contractFence = "====="

    /// Above this, a turn that declares an endpoint has room to write past its
    /// declared format.
    static let strictFormatTokenBudget = 512

    /// The first layout placeholder a contract field is not allowed to carry
    /// (spec §1.3), spelled as it appears, or nil when the text is clean.
    static func forbiddenContractPlaceholder(in text: String) -> String? {
        for placeholder in ["{{scenario.materials}}", "{{agent.context}}"]
        where text.contains(placeholder) {
            return placeholder
        }
        guard text.contains("{{outputs.") else { return nil }
        return outputReferences(in: text).first.map { "{{outputs.\($0)}}" } ?? "{{outputs."
    }

    /// Labels referenced by `{{outputs.<label>}}` in a template, in order.
    static func outputReferences(in template: String) -> [String] {
        var out: [String] = []
        var rest = Substring(template)
        while let open = rest.range(of: "{{outputs."),
            let close = rest[open.upperBound...].range(of: "}}")
        {
            out.append(String(rest[open.upperBound ..< close.lowerBound]))
            rest = rest[close.upperBound...]
        }
        return out
    }

    public static func validate(_ scenario: MultiAgentScenario) throws {
        let name = scenario.name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { throw ExperimentError(reason: "scenario needs a name") }
        guard !scenario.agents.isEmpty else {
            throw ExperimentError(reason: "scenario needs at least one agent")
        }
        guard !scenario.turns.isEmpty else {
            throw ExperimentError(reason: "scenario needs at least one turn")
        }
        let agentIDs = Set(scenario.agents.map(\.id))
        guard agentIDs.count == scenario.agents.count else {
            throw ExperimentError(reason: "agent IDs must be unique")
        }
        for agent in scenario.agents {
            guard !agent.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ExperimentError(reason: "every agent needs a name")
            }
            guard !agent.baseModelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ExperimentError(reason: "agent '\(agent.name)' needs a base model")
            }
        }
        // Labels produced by turns already seen, so a contract input can be
        // checked for "exists AND is earlier" in one pass.
        var produced = Set<String>()
        for (index, turn) in scenario.turns.enumerated() {
            defer { produced.insert(normalizedOutputLabel(turn: turn, index: index)) }
            guard agentIDs.contains(turn.speakerAgentID) else {
                throw ExperimentError(reason: "turn '\(turn.title)' speaker is not an agent")
            }
            let template = turn.promptTemplate.trimmingCharacters(in: .whitespacesAndNewlines)
            if let contract = turn.contract {
                // One renderer per turn. A contract plus a template is not a
                // merge to resolve, it is an author who meant one of the two.
                guard template.isEmpty else {
                    throw ExperimentError(
                        reason: "turn '\(turn.title)' declares both a contract and a "
                            + "prompt template — a turn has exactly one renderer")
                }
                guard !contract.task.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                else {
                    throw ExperimentError(
                        reason: "turn '\(turn.title)' declares a contract with no task")
                }
                for (field, text) in contract.textFields {
                    guard let placeholder = forbiddenContractPlaceholder(in: text) else {
                        continue
                    }
                    throw ExperimentError(
                        reason: "turn '\(turn.title)' contract \(field) uses \(placeholder), "
                            + "which the contract renderer places in its own canonical slot")
                }
                // Stricter than the template advisory on purpose: a contract's
                // inputs are structured data, not prose that might happen to
                // contain a placeholder, so an unresolvable one is a mistake
                // rather than a risk.
                for label in contract.inputs where !produced.contains(label) {
                    throw ExperimentError(
                        reason: "turn '\(turn.title)' contract input '\(label)' is not "
                            + "produced by any earlier turn")
                }
            } else {
                guard !template.isEmpty else {
                    throw ExperimentError(reason: "turn '\(turn.title)' needs a prompt template")
                }
            }
            if turn.routing == .selected {
                let unknown = turn.routedAgentIDs.filter { !agentIDs.contains($0) }
                guard unknown.isEmpty else {
                    throw ExperimentError(
                        reason: "turn '\(turn.title)' routes to unknown agent IDs")
                }
            }
            if let endpoint = turn.endpoint {
                // Re-checked here as well as on decode: a scenario built in
                // code (the Scenario Lab, a test) never passed through
                // `init(from:)`, and an endpoint that parses nothing must not
                // reach a measured run. Server twin: `multi_agent.validate`.
                try validate(endpoint, turn: turn.title)
            }
        }
    }

    /// Declaration rules, re-run over an already-constructed endpoint by
    /// round-tripping it through the validating decoder — one definition of
    /// "well formed", not two that can drift.
    private static func validate(_ endpoint: TurnEndpoint, turn: String) throws {
        do {
            _ = try JSONDecoder().decode(
                TurnEndpoint.self, from: try JSONEncoder().encode(endpoint))
        } catch let error as ExperimentError {
            throw ExperimentError(reason: "turn '\(turn)': \(error.reason)")
        }
    }

    /// The turn's declared endpoint parsed out of `output`, or nil when the
    /// turn declares none (in which case no key is stamped at all and the
    /// record is what it always was).
    ///
    /// A named seam so the artifact contract is testable without a model:
    /// this is exactly the expression the run loop uses.
    static func endpointStamp(
        for turn: MultiAgentScenario.Turn, output: String
    ) -> TurnEndpointStamp? {
        turn.endpoint.map { TurnEndpointParser.stamp($0, in: output) }
    }

    /// The turn's voice lint (spec §5). Stamped on EVERY turn — no declaration
    /// gates it, because "did this seat write as itself" applies to every turn
    /// a panel produces — and never blocking: a noncompliant turn is recorded
    /// and the run completes. Regenerating it would select on the dependent
    /// variable, since the failure rate is what differs between arms.
    ///
    /// A named seam, like `endpointStamp`: exactly the expression the run loop
    /// uses, testable without a model.
    static func voiceLintStamp(
        for turn: MultiAgentScenario.Turn,
        output: String,
        agents: [MultiAgentScenario.Agent]
    ) -> VoiceLintStamp {
        let speaker = agents.first { $0.id == turn.speakerAgentID }
        return VoiceLint.stamp(
            in: output,
            speaker: speaker?.name ?? "",
            others: agents.filter { $0.id != turn.speakerAgentID }.map(\.name))
    }

    /// Resolve one seat's runtime settings.
    ///
    /// `systemPrompt` is the seat's EFFECTIVE system prompt — the composition
    /// of the cast agent artifact's persona and the seat's own cast-entry role
    /// text, never one of them alone (see the casting note below) — and
    /// `systemComposition` is the additive provenance stamp for exactly that
    /// text, built beside it here so a turn record cannot stamp a composition
    /// its generation did not run under. Server twin:
    /// `multi_agent._runtime_settings`.
    static func runtimeSettings(
        for agent: MultiAgentScenario.Agent,
        stripInterventions: Bool = false
    ) throws -> (
        variant: ModelVariantArtifact?,
        modelID: String,
        promptMode: ExperimentManifest.PromptMode,
        systemPrompt: String?,
        qwenThinking: Bool,
        injections: [ExperimentTasks.CellInjection],
        warnings: [MultiAgentRunWarning],
        systemComposition: PanelSystemPromptCompositionStamp
    ) {
        var warnings: [MultiAgentRunWarning] = []
        let variant = try agent.variantArtifactPath.map { path -> ModelVariantArtifact in
            let url = ModelVariantStore.absoluteURL(path)
            if let expected = agent.variantArtifactHash {
                let actual = try ModelVariantStore.hash(url)
                if actual != expected {
                    warnings.append(
                        MultiAgentRunWarning(
                            agentName: agent.name,
                            variantArtifactPath: path,
                            expectedHash: expected,
                            actualHash: actual,
                            message: "variant artifact drifted for agent '\(agent.name)'; using the current artifact on disk"))
                }
            }
            let data = try Data(contentsOf: url)
            return try JSONDecoder().decode(ModelVariantArtifact.self, from: data)
        }
        let modelID = variant?.baseModelID ?? agent.baseModelID
        let promptMode = variant.flatMap { ExperimentManifest.PromptMode(rawValue: $0.promptMode) }
            ?? .chatAssistant
        // Casting COMPOSES; it no longer replaces (maintainer ruling,
        // 2026-08-24). A seat has two levels of system content, and they used
        // to be mutually exclusive here — a cast entry with role text silently
        // discarded the agent's persona, and a seat with no role text ran on
        // the persona alone:
        //
        //   * the AGENT ARTIFACT's `systemPrompt` — the persona, who the model
        //     is;
        //   * the CAST ENTRY's `systemPrompt` — the role, "you represent Team
        //     South", which `PanelComposition.semanticForm` deliberately keeps
        //     on the seat because the role is the EXPERIMENT and the agent is
        //     what is cast into it.
        //
        // Persona first, role second, joined by one blank line — the SAME
        // order and the same principle as the study rule
        // (`SystemPromptComposition.compose`): identity precedes instruction,
        // and a cast role is situational instruction TO whoever the agent is,
        // exactly as a study frame is. Composed through that type's own
        // primitive rather than re-spelled here, so the joiner and the four
        // degradation cases cannot drift between the study path and this one.
        //
        // Degradation carries the legacy lock: every agent in the workspace
        // today has an EMPTY persona, so `compose` returns the cast text
        // itself — the same string, not a re-joined copy — and every existing
        // panel renders the bytes it always did.
        let seat = SeatSystemPrompt(
            agent: variant?.systemPrompt, cast: agent.systemPrompt)
        let qwenThinking = variant?.qwenThinkingEnabled ?? false
        let injections =
            stripInterventions
            ? []
            : (try variant.map(ExperimentTasks.injections(for:)) ?? [])
        let adapterVariant = stripInterventions ? nil : variant
        return (
            adapterVariant, modelID, promptMode, seat.effective, qwenThinking,
            injections, warnings, seat.stamp)
    }

    /// The model this turn generates through, held per (id, revision) for the
    /// rest of the run.
    ///
    /// The key carries the REVISION (review round 9, finding 1, the same
    /// class as the judge cache): a panel whose variants pin two base
    /// revisions of one checkpoint used to find the first-loaded container
    /// under the bare model id, so every later turn generated with revision
    /// A's weights while the variant declared B — and `modelRevision` on the
    /// turn record stamped A, an honest stamp for a declaration that was
    /// silently discarded.
    private static func model(
        for modelID: String,
        revision requestedRevision: String?,
        models: inout [ExperimentTasks.LoadedModelKey: RuntimeModel]
    ) async throws -> RuntimeModel {
        let key = ExperimentTasks.LoadedModelKey(
            model: modelID, revision: requestedRevision)
        if let runtime = models[key] { return runtime }
        let container = try await SteeredContainerLoader.load(
            modelID: modelID, revision: requestedRevision)
        let revision = requestedRevision ?? SteeredContainerLoader.cachedRevision(for: modelID)
        let runtime = RuntimeModel(container: container, revision: revision)
        models[key] = runtime
        return runtime
    }

    /// Render one turn's prompt. Pure over strings — the speaker arrives as a
    /// name plus already-joined context — so it mirrors the server's
    /// `multi_agent._render_prompt` signature and is unit-testable without a
    /// runtime agent. Internal rather than private for that reason.
    static func renderPrompt(
        scenario: MultiAgentScenario,
        turn: MultiAgentScenario.Turn,
        speakerName: String,
        speakerContext joinedContext: String,
        outputsByLabel: [String: String]
    ) -> String {
        // One renderer per turn, chosen by the turn's own declaration — never
        // a merge of the two. `validate` refuses a turn that carries both.
        if let contract = turn.contract {
            return renderContractPrompt(
                scenario: scenario, turn: turn, contract: contract,
                speakerName: speakerName, speakerContext: joinedContext,
                outputsByLabel: outputsByLabel)
        }
        var prompt = turn.promptTemplate
        let materials = turn.includeScenarioMaterials ? scenario.sharedMaterials : ""
        let speakerContext = turn.includeSpeakerContext ? joinedContext : ""
        let replacements: [String: String] = [
            "{{scenario.name}}": scenario.name,
            "{{scenario.description}}": scenario.description,
            "{{scenario.materials}}": materials,
            "{{agent.name}}": speakerName,
            "{{agent.context}}": speakerContext,
            "{{turn.title}}": turn.title,
        ]
        for (key, value) in replacements {
            prompt = prompt.replacingOccurrences(of: key, with: value)
        }
        for (label, output) in outputsByLabel {
            prompt = prompt.replacingOccurrences(of: "{{outputs.\(label)}}", with: output)
        }
        // The fallback is for templates that never ASKED for the context or
        // the materials — test the ORIGINAL template, not the substituted
        // string. Substitution has by definition just removed the placeholder,
        // so testing `prompt` here made both guards unconditionally true and
        // appended a second copy to every template that did interpolate them
        // (every built-in one). The server tests `turn.prompt_template` for the
        // same reason — see `multi_agent._render_prompt`.
        //
        // PREPENDED, not appended (spec §3.2, 2026-08-17). Appending put the
        // case record AFTER the instruction, which reads as an afterthought
        // and left the last thing before the model's turn being material
        // rather than the task; record first, transcript second, instruction
        // last is the order the contract renderer uses and the order that
        // stopped the voice failures.
        let template = turn.promptTemplate
        var sections: [String] = []
        if !template.contains("{{scenario.materials}}"), !materials.isEmpty {
            sections.append("Shared scenario materials:\n\(materials)")
        }
        if !template.contains("{{agent.context}}"), !speakerContext.isEmpty {
            sections.append("Visible prior context:\n\(speakerContext)")
        }
        sections.append(prompt)
        return sections.joined(separator: "\n\n")
    }

    // MARK: - Contract rendering (spec §2)

    /// The canonical sandwich. Pure over the scenario and the strings a turn
    /// has in hand, so the byte-level contract is testable without a model —
    /// which is the point: the Python twin asserts the identical worked
    /// example, and a divergence has to fail somewhere cheap.
    ///
    /// Blocks are joined by exactly one blank line and the prompt has no
    /// trailing newline. A block whose content is empty is OMITTED ENTIRELY:
    /// no empty fence, and no glue line left dangling with nothing above it.
    static func renderContractPrompt(
        scenario: MultiAgentScenario,
        turn: MultiAgentScenario.Turn,
        contract: TurnContract,
        speakerName: String,
        speakerContext joinedContext: String,
        outputsByLabel: [String: String]
    ) -> String {
        let speakerID = turn.speakerAgentID
        let speaker = scenario.agents.first { $0.id == speakerID }
        let colleagues = scenario.agents.filter { $0.id != speakerID }.map(\.name)
        func substitute(_ text: String) -> String {
            contractSubstitutions(
                text, scenario: scenario, turn: turn, speakerName: speakerName)
        }
        var blocks: [String] = []

        // Block 1 — identity opener (always).
        var opener = "You are \(speakerName)"
        if let role = MultiAgentScenario.Agent.normalizedRole(speaker?.role) {
            opener += ", \(role)"
        }
        opener += "."
        if !colleagues.isEmpty {
            opener += " The other participants are \(nameList(colleagues, conjunction: "and"))."
        }
        // `stage` and `task` are TRIMMED before being joined into a sentence —
        // they are clauses this renderer concatenates, and a stray newline in
        // an authored field would otherwise land mid-sentence. `format` and
        // `materialsTitle` are verbatim: one is a literal output specimen and
        // the other is a fence label. Same split in the Python twin.
        let stage = substitute(contract.stage)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !stage.isEmpty { opener += " \(stage)" }
        blocks.append(opener)

        // Block 2 — shared materials, plus its glue line as its own block.
        let materials = turn.includeScenarioMaterials ? scenario.sharedMaterials : ""
        if !materials.isEmpty {
            let title = substitute(contract.materialsTitle)
            blocks.append(
                "===== \(title) =====\n\(materials)\n===== END OF \(title) =====")
            blocks.append("That was the shared material. Every participant has read it.")
        }

        // Block 3 — inputs, in declared order, attributed by the PRODUCING
        // turn's speaker. Own-authored inputs are named as the reader's own
        // work: presenting a judge's own draft in the same third person as a
        // colleague's is how a panel ends up signing someone else's opinion.
        let producers = earlierProducers(scenario: scenario, before: turn)
        var showedOtherOutput = false
        for label in contract.inputs {
            guard let producer = producers[label] else { continue }
            let output = outputsByLabel[label] ?? ""
            guard !output.isEmpty else { continue }
            if producer.speakerAgentID == speakerID {
                blocks.append(
                    "===== YOUR OWN EARLIER OUTPUT — \(producer.title) =====\n\(output)\n"
                        + "===== END OF YOUR OWN EARLIER OUTPUT =====")
                blocks.append(
                    "That was your own earlier output, written by you, \(speakerName).")
            } else {
                let producerName =
                    scenario.agents.first { $0.id == producer.speakerAgentID }?.name
                    ?? producer.speakerAgentID
                blocks.append(
                    "===== OUTPUT OF \(producerName) — \(producer.title) =====\n\(output)\n"
                        + "===== END OF OUTPUT OF \(producerName) =====")
                showedOtherOutput = true
            }
        }
        if showedOtherOutput {
            blocks.append(
                "Those were the contributions of the other participants. "
                    + "You have now read them.")
        }

        // Block 4 — routed transcript.
        let context = turn.includeSpeakerContext ? joinedContext : ""
        let showedTranscript = !context.isEmpty
        if showedTranscript {
            blocks.append(
                "===== TRANSCRIPT SO FAR =====\n\(context)\n===== END OF TRANSCRIPT SO FAR =====")
        }

        // Block 5 — task (always).
        let task = substitute(contract.task).trimmingCharacters(in: .whitespacesAndNewlines)
        blocks.append("===== YOUR TASK =====\nYou are \(speakerName). \(task)")

        // Block 6 — own-voice constraint. Pointless with a single agent:
        // there is no one else to speak for.
        if contract.ownVoice, scenario.agents.count > 1 {
            var line =
                "Write only your own response, in your own voice, as \(speakerName) "
                + "and no one else. Do not write, draft, continue, quote at length, "
                + "or reply on behalf of \(nameList(colleagues, conjunction: "or"))."
            if showedOtherOutput || showedTranscript {
                line +=
                    " Their contributions above are finished documents; "
                    + "you are adding one document of your own."
            }
            blocks.append(line)
        }

        // Block 7 — format, verbatim.
        let format = substitute(contract.format)
        if !format.isEmpty { blocks.append(format) }

        // Block 8 — closing reminder (always).
        blocks.append(
            "Reminder: you are \(speakerName). Respond as \(speakerName) and as no one else.")

        return blocks.joined(separator: "\n\n")
    }

    /// The four substitutions a contract's text may use (spec §1.3). The
    /// layout placeholders — `{{scenario.materials}}`, `{{agent.context}}`,
    /// `{{outputs.*}}` — are deliberately absent: they have canonical slots,
    /// and `validate` refuses a contract that tries to place them by hand.
    static func contractSubstitutions(
        _ text: String,
        scenario: MultiAgentScenario,
        turn: MultiAgentScenario.Turn,
        speakerName: String
    ) -> String {
        var out = text
        for (key, value) in [
            ("{{scenario.name}}", scenario.name),
            ("{{scenario.description}}", scenario.description),
            ("{{agent.name}}", speakerName),
            ("{{turn.title}}", turn.title),
        ] {
            out = out.replacingOccurrences(of: key, with: value)
        }
        return out
    }

    /// `"Ben"`, `"Ben and Cal"`, `"Ben, Cal, and Dee"` — Oxford comma at three
    /// or more. `conjunction` is `"and"` for the roll call and `"or"` for the
    /// own-voice prohibition.
    static func nameList(_ names: [String], conjunction: String) -> String {
        switch names.count {
        case 0: ""
        case 1: names[0]
        case 2: "\(names[0]) \(conjunction) \(names[1])"
        default:
            names.dropLast().joined(separator: ", ") + ", \(conjunction) " + (names.last ?? "")
        }
    }

    /// Output label → the turn that produces it, restricted to turns STRICTLY
    /// EARLIER than `turn`. A turn absent from the script (one built in a
    /// test) sees the whole script as earlier. Duplicate labels resolve the
    /// way the run loop's `outputsByLabel` does: the later turn wins.
    static func earlierProducers(
        scenario: MultiAgentScenario,
        before turn: MultiAgentScenario.Turn
    ) -> [String: MultiAgentScenario.Turn] {
        let limit = scenario.turns.firstIndex { $0.id == turn.id } ?? scenario.turns.count
        var out: [String: MultiAgentScenario.Turn] = [:]
        for (index, candidate) in scenario.turns.enumerated() where index < limit {
            out[normalizedOutputLabel(turn: candidate, index: index)] = candidate
        }
        return out
    }

    /// Which renderer a turn goes through, as stamped on its record.
    static func promptRenderer(for turn: MultiAgentScenario.Turn) -> String {
        turn.contract == nil ? "template-v1" : "contract-v1"
    }

    /// Internal, not private: the cross-engine render fixture replays a
    /// scenario through the REAL routing and label rules rather than a copy
    /// of them that could drift.
    static func routedAgentIDs(
        for turn: MultiAgentScenario.Turn,
        agents: [MultiAgentScenario.Agent]
    ) -> [String] {
        switch turn.routing {
        case .all:
            return agents.map(\.id)
        case .speakerOnly:
            return [turn.speakerAgentID]
        case .selected:
            return turn.routedAgentIDs
        case .none:
            return []
        }
    }

    static func normalizedOutputLabel(
        turn: MultiAgentScenario.Turn,
        index: Int
    ) -> String {
        let explicit = turn.outputLabel.trimmingCharacters(in: .whitespacesAndNewlines)
        if !explicit.isEmpty { return explicit }
        return "turn_\(index + 1)"
    }

    /// One transcript entry, rendered FOR A PARTICULAR READER (spec §3.1).
    ///
    /// The same output is a different thing to different readers: to everyone
    /// else it is a colleague's document, and to its author it is the thing
    /// they themselves wrote. Rendering both the same way — `— Judge Marsden`
    /// either way — is what taught the model to continue its own prior turn in
    /// the third person, and then to continue everyone else's. So the entry is
    /// computed per RECEIVING agent, not once per turn.
    static func contextEntry(
        turn: MultiAgentScenario.Turn,
        outputLabel: String,
        speaker: MultiAgentScenario.Agent,
        output: String,
        reader readerID: String
    ) -> String {
        let attribution =
            readerID == speaker.id
            ? "your own earlier output (\(speaker.name))"
            : speaker.name
        return """
            [\(outputLabel)] \(turn.title) — \(attribution)
            \(output)
            """
    }

    /// Turns a previous attempt finished, by turn id. A torn final line is
    /// discarded rather than repaired — half a turn is not a turn, and
    /// trusting it would poison every turn after it.
    static func completedTurns(at url: URL) -> [String: MultiAgentTurnResult] {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return [:] }
        let decoder = JSONDecoder()
        var out: [String: MultiAgentTurnResult] = [:]
        for line in text.split(separator: "\n", omittingEmptySubsequences: true) {
            guard
                let turn = try? decoder.decode(
                    MultiAgentTurnResult.self, from: Data(line.utf8))
            else { continue }
            out[turn.turnID] = turn
        }
        return out
    }

    /// Cut the file back to its last COMPLETE line. Skipping a torn tail is
    /// not enough: the fragment stays on disk and the next append lands on the
    /// same line, leaving a permanently malformed record that whatever parses
    /// turns.jsonl strictly will die on later, far from the kill that caused it.
    static func truncateTornTail(at url: URL) {
        guard let data = try? Data(contentsOf: url), !data.isEmpty else { return }
        guard data.last != UInt8(ascii: "\n") else { return }
        let newline = data.lastIndex(of: UInt8(ascii: "\n"))
        let tailStart = newline.map { data.index(after: $0) } ?? data.startIndex
        let tail = Data(data[tailStart...])
        // Complete JSON that merely lost its newline is a FINISHED turn.
        // Terminate it instead of discarding it.
        if (try? JSONDecoder().decode(MultiAgentTurnResult.self, from: tail)) != nil {
            try? (data + Data("\n".utf8)).write(to: url, options: .atomic)
            return
        }
        if let newline {
            try? Data(data[...newline]).write(to: url, options: .atomic)
        } else {
            try? Data().write(to: url, options: .atomic)
        }
    }

    private static func makeRunDirectory(scenario: MultiAgentScenario) throws -> URL {
        let directory = try VectorCatalog.makeUniqueRunDirectory(
            slug: "multi-agent-\(FineTuneStore.slugify(scenario.name))-run",
            under: VectorCatalog.runsDirectory)
        try RunMetadata.write(
            runType: "multi-agent", to: directory, modelID: scenario.baseModelID)
        return directory
    }

    private static func warningKey(_ warning: MultiAgentRunWarning) -> String {
        [
            warning.agentName,
            warning.variantArtifactPath ?? "",
            warning.expectedHash ?? "",
            warning.actualHash ?? "",
        ].joined(separator: "|")
    }

    /// Marks a run stopped by the user — deliberately a DIFFERENT artifact
    /// from error.json, so a stop never reads as a crash and never enters the
    /// analysis path as a measurement.
    private static func writeCancellationNote(
        scenario: MultiAgentScenario, runDirectory: URL,
        turnIndex: Int, turnTitle: String
    ) {
        let note = [
            "cancelled": true,
            "task": "multi-agent scenario run",
            "scenarioName": scenario.name,
            "stoppedAtTurnIndex": turnIndex,
            "stoppedAtTurnTitle": turnTitle,
            "cancelledAt": ISO8601DateFormatter().string(from: Date()),
        ] as [String: Any]
        guard
            let data = try? JSONSerialization.data(
                withJSONObject: note, options: [.prettyPrinted, .sortedKeys])
        else { return }
        try? data.write(to: runDirectory.appending(component: "cancelled.json"))
    }

    private static func writeFailure(
        scenario: MultiAgentScenario,
        runDirectory: URL,
        turnIndex: Int?,
        turnTitle: String?,
        speakerName: String?,
        error: any Error
    ) -> MultiAgentRunErrorReport {
        let report = MultiAgentRunErrorReport(
            scenarioName: scenario.name,
            runDirectory: runDirectory.path,
            failedAt: ISO8601DateFormatter().string(from: Date()),
            turnIndex: turnIndex,
            turnTitle: turnTitle,
            speakerName: speakerName,
            error: String(describing: error))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try? encoder.encode(report).write(to: runDirectory.appending(component: "error.json"))
        return report
    }

    private static func hash(_ scenario: MultiAgentScenario) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = (try? encoder.encode(scenario)) ?? Data()
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static func markdownTranscript(
        scenario: MultiAgentScenario,
        results: [MultiAgentTurnResult]
    ) -> String {
        var lines: [String] = [
            "# \(scenario.name)",
            "",
            scenario.description,
            "",
            "## Turns",
        ]
        for result in results {
            lines.append("")
            lines.append("### \(result.turnIndex). \(result.title)")
            lines.append("")
            lines.append("Speaker: \(result.speakerName)")
            lines.append("Routed to: \(result.routedAgentIDs.joined(separator: ", "))")
            lines.append("")
            lines.append(result.output)
        }
        return lines.joined(separator: "\n")
    }
}
