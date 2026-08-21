import Foundation
import MLXLMCommon

/// Running a capability battery on THIS engine — the Swift twin of the
/// server's battery scoring path (`steerlab_server/experiment/tasks.py`
/// `_battery_backends` / `_run_capability_battery`; the contract authority is
/// `battery.py`).
///
/// The whole point of the split below is that a battery's ARMING comes from
/// the battery's format, never from the caller's convenience:
///
/// - a **format-2** battery is scored under its OWN promptMode/systemPrompt/
///   thinking flag/token cap, identically under baseline, steering and
///   variant conditions, and its choice items are read by the answer-token
///   logprob instrument (nothing is generated, so the reading cannot move
///   with response length or format compliance);
/// - a **format-1** battery keeps the historical arming — the surrounding
///   instrument's — so its pinned hash keeps its historical meaning, and the
///   contaminated case says so out loud (`contaminationAdvisory`).
///
/// Both back-ends are handed to `CapabilityBattery.scoreItem` as closures, so
/// the scoring/record arithmetic is testable without a model (server parity:
/// `generate_fn` / `choice_fn`).
extension ExperimentTasks {

    /// The two scoring back-ends a battery item can use, bound to ONE
    /// condition's interventions. Both take the battery's arming — not the
    /// study's rendering context — so a format-2 battery is scored
    /// identically under every condition and the intervention is the only
    /// thing that differs.
    struct BatteryBackends {
        let generate: (_ prompt: String, _ arming: BatteryArming) async throws -> String
        let choice:
            (_ prompt: String, _ options: [String], _ arming: BatteryArming) async throws
                -> (selected: String, probability: [String: Double])
    }

    /// `onChunk` streams a GENERATED item's partial text to the caller's
    /// progress surface; a choice item generates nothing, so it never fires.
    static func batteryBackends(
        _ container: ModelContainer, modelID: String,
        injections: [CellInjection],
        onChunk: GenerationChunkHandler? = nil
    ) -> BatteryBackends {
        BatteryBackends(
            generate: { prompt, arming in
                try await generate(
                    container, prompt: prompt, modelID: modelID,
                    maxTokens: arming.maxTokens,
                    temperature: 0,
                    injections: injections,
                    promptMode: arming.promptMode,
                    systemPrompt: arming.systemPrompt,
                    qwenThinkingEnabled: arming.qwenThinkingEnabled,
                    onChunk: onChunk)
            },
            choice: { prompt, options, arming in
                // The instrument's own decode-identical injection semantics:
                // single-chunk prefill, one step per option token under the
                // KV cache. Nothing here re-plumbs steering.
                let result = try await LogprobInstrument.scoreOptions(
                    container, prompt: prompt, options: options,
                    modelID: modelID,
                    injections: injections,
                    promptMode: arming.promptMode,
                    systemPrompt: arming.systemPrompt,
                    qwenThinkingEnabled: arming.qwenThinkingEnabled)
                return (result.selected, result.probability)
            })
    }

    /// One battery item's record under one condition (engine-pure: the
    /// reading is already taken). Format-1 rows carry the exact six keys they
    /// always did; format-2 rows add the scoring readout and the arming
    /// provenance under the server's key names.
    static func batteryRecord(
        condition: String,
        promptIndex: Int,
        item: CapabilityBattery.Item,
        score: BatteryItemScore,
        battery: CapabilityBattery,
        arming: BatteryArming
    ) -> BatteryGenerationRecord {
        var record = BatteryGenerationRecord(
            condition: condition,
            promptIndex: promptIndex,
            prompt: item.prompt,
            expected: item.answer,
            output: score.output,
            correct: score.correct)
        guard battery.isolated else { return record }
        let fields = arming.recordFields
        record.batteryFormat = battery.formatVersion
        record.scoring = score.scoring.rawValue
        record.options = score.options
        record.choiceProbability = score.choiceProbability
        record.selected = score.selected
        record.armingIsolated = fields.isolated
        record.armingPromptMode = fields.promptMode
        record.armingSystemPrompt = fields.systemPrompt
        record.armingMaxTokens = fields.maxTokens
        return record
    }

    static func batterySummary(
        _ scores: [BatteryItemScore], batteryHash: String
    ) -> CapabilityBatterySummary {
        CapabilityBatterySummary(
            accuracy: scores.isEmpty
                ? 0 : Double(scores.count { $0.correct }) / Double(scores.count),
            itemCount: scores.count,
            batteryHash: batteryHash)
    }

    /// One condition's records + summary from readings ALREADY taken — the
    /// engine-pure half of the battery pass (the GPU never enters here), so
    /// the record/summary contract is unit-testable on fixtures.
    static func scoreBatteryReadings(
        condition: String,
        battery: CapabilityBattery,
        arming: BatteryArming,
        scores: [BatteryItemScore],
        batteryHash: String
    ) -> (records: [BatteryGenerationRecord], summary: CapabilityBatterySummary) {
        let records = zip(battery.items, scores).enumerated().map {
            index, pair in
            batteryRecord(
                condition: condition, promptIndex: index + 1, item: pair.0,
                score: pair.1, battery: battery, arming: arming)
        }
        return (records, batterySummary(scores, batteryHash: batteryHash))
    }

    /// Scores one condition's whole battery: reads every item through the
    /// arming-appropriate back-end and returns the records plus the summary.
    /// Returns nil when `cancel` observes a cancellation between items — a
    /// partially scored battery is NEVER reported as an accuracy.
    static func runBattery(
        _ container: ModelContainer,
        battery: CapabilityBattery,
        batteryHash: String,
        condition: String,
        modelID: String,
        injections: [CellInjection],
        arming: BatteryArming,
        cancel: (@Sendable (Int) async -> Bool)? = nil
    ) async throws
        -> (records: [BatteryGenerationRecord], summary: CapabilityBatterySummary)?
    {
        let backends = batteryBackends(
            container, modelID: modelID, injections: injections)
        var scores: [BatteryItemScore] = []
        for (index, item) in battery.items.enumerated() {
            if let cancel, await cancel(index) { return nil }
            scores.append(
                try await battery.scoreItem(
                    item, arming: arming,
                    generate: backends.generate, choice: backends.choice))
        }
        return scoreBatteryReadings(
            condition: condition, battery: battery, arming: arming,
            scores: scores, batteryHash: batteryHash)
    }
}
