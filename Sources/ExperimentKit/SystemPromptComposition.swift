import CryptoKit
import Foundation

/// How an arm's system prompt is COMPOSED, and how that composition is
/// stamped.
///
/// Server twin: `steerlab_server/experiment/system_prompt.py` — same joiner,
/// same degradation rules, same stamp key spellings. Both engines must
/// compose byte-identically or two runs of the same study on two substrates
/// are two different studies.
///
/// Two levels of system prompt exist, and they used to REPLACE one another:
///
/// - the **agent's** `systemPrompt` — who the model is (a persona carried on
///   a saved agent artifact), and
/// - the **study's** `manifest.systemPrompt` — the deployment frame every arm
///   of the study is read under (response format, task framing).
///
/// Replacement made those two mutually exclusive: a variant/agent arm ran
/// under its persona with the study's frame simply not applied, while
/// baseline and steering arms ran under the frame with no persona. The arms
/// were then not comparable — the frame was part of the contrast rather than
/// held constant.
///
/// **The rule (maintainer ruling, 2026-08-24).** The effective system prompt
/// of an arm is the AGENT's text first, then the STUDY's frame, joined by one
/// blank line. Persona first because identity precedes instruction: the
/// frame's "answer in this format" is an instruction TO whoever the model is
/// being.
///
/// **Degradation is graceful, and byte-exact.** An empty agent yields the
/// frame itself — the SAME string, not a re-joined copy — so every historical
/// empty-persona run stamps and renders exactly the bytes it always did. An
/// empty frame yields the persona alone. Both empty yields nothing. Emptiness
/// is whitespace-insensitive, but a non-empty value is never trimmed: what
/// the researcher wrote is what the model is armed with.
///
/// Batteries compose the same way with a different second term — the
/// battery's own declared arming text, never the study frame (see
/// `CapabilityBattery.resolveArming`).
///
/// **Panels compose the same way too** (maintainer ruling, 2026-08-24). A
/// panel seat carries a CAST ENTRY prompt — the role, "you represent Team
/// South" — and the agent artifact cast into that seat carries the persona.
/// Casting used to REPLACE, exactly as the study levels did
/// (`MultiAgentRunner.runtimeSettings`). The order is the SAME as the study
/// rule and holds for the same reason: identity precedes instruction, and a
/// cast role is situational instruction TO whoever the agent is, just as the
/// study frame is. One uniform rule everywhere, so there is deliberately no
/// second entry point — the panel path calls `compose(agent:frame:)` with the
/// cast text as the `frame` and stamps with
/// `PanelSystemPromptCompositionStamp`.
public enum SystemPromptComposition {

    /// The separator between the two levels. One blank line, which every chat
    /// template renders as a paragraph break inside a single system turn.
    public static let joiner = "\n\n"

    /// True when this text carries no system content at all.
    static func isEmpty(_ text: String?) -> Bool {
        (text ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    /// The effective system prompt: `agent` first, then `frame`.
    ///
    /// Returns the surviving side UNCHANGED when the other is empty —
    /// identity, not reconstruction, which is what makes the empty-persona
    /// case byte-exact against every run recorded before this rule existed.
    public static func compose(agent: String?, frame: String?) -> String? {
        if isEmpty(agent) { return frame }
        if isEmpty(frame) { return agent }
        return agent! + joiner + frame!
    }

    /// SHA-256 of a system prompt, or nil when there is nothing to hash.
    ///
    /// The same rule the per-condition `systemPromptHash` stamp has always
    /// used on the server (`tasks._sha256_text`), so the composition stamp's
    /// hashes and the effective hash beside them are comparable without a
    /// second convention.
    public static func hash(_ text: String?) -> String? {
        guard let text, !text.isEmpty else { return nil }
        return SHA256.hash(data: Data(text.utf8))
            .map { String(format: "%02x", $0) }.joined()
    }

    /// The loud, non-blocking run-start advisory for arms armed with
    /// DIFFERENT effective system content.
    ///
    /// `arms` is in the run's own emission order. nil when every arm shares
    /// one effective content — the universal case, which must stay silent.
    ///
    /// Why an advisory and not a refusal: a deliberately persona-varying
    /// design IS a legitimate study (that is the point of promoting agents),
    /// so this cannot gate. What it must not do is let the difference pass
    /// unremarked, because a contrast between two arms armed differently
    /// mixes identity and framing into whatever the intervention did.
    /// Server twin: `system_prompt.divergence_advisory`, same wording.
    public static func divergenceAdvisory(
        arms: [(name: String, systemPrompt: String?)]
    ) -> String? {
        let listed = arms.map { ($0.name, hash($0.systemPrompt)) }
        guard Set(listed.map { $0.1 ?? "" }).count >= 2 else { return nil }
        let parts = listed.map { name, digest in
            "\(name) (\(digest.map { $0.prefix(12) + "…" } ?? "none"))"
        }.joined(separator: ", ")
        return "arms in this run are armed with DIFFERENT effective system "
            + "prompts — \(parts). A contrast between two such arms mixes "
            + "identity and framing with the intervention; hold the system "
            + "prompt constant across arms, or report the difference as part "
            + "of the design. Not a refusal."
    }
}

/// The additive provenance block beside an effective `systemPromptHash`:
/// which level contributed what.
///
/// Both keys are ALWAYS encoded (`null` when that level contributed nothing)
/// — an absent key would read as "this engine does not stamp composition", a
/// different claim from "this level was empty". The server's
/// `system_prompt.composition` emits the identical shape.
public struct SystemPromptCompositionStamp: Codable, Equatable, Sendable {
    /// SHA-256 of the AGENT's persona text; nil ⇒ JSON `null` (no persona).
    public let agent: String?
    /// SHA-256 of the STUDY frame; nil ⇒ JSON `null` (no frame).
    public let study: String?

    public init(agent: String?, study: String?) {
        self.agent = agent
        self.study = study
    }

    /// The stamp for one arm, from the two raw texts.
    public init(agentText: String?, studyText: String?) {
        self.init(
            agent: SystemPromptComposition.hash(agentText),
            study: SystemPromptComposition.hash(studyText))
    }

    /// Neither level contributed anything (both keys `null`).
    public static let none = SystemPromptCompositionStamp(agent: nil, study: nil)

    enum CodingKeys: String, CodingKey { case agent, study }

    /// Explicit `encode`, not `encodeIfPresent`: nil must reach the wire as
    /// `null`, matching the server's always-present keys.
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(agent, forKey: .agent)
        try container.encode(study, forKey: .study)
    }
}

/// ONE arm's resolved system-prompt arming: the effective text generation runs
/// under, and the stamp that says which levels produced it.
///
/// The two are built together, from the same two inputs, so a record can never
/// stamp a composition its generation did not run under. Server twin: the
/// `system_prompt`/`agent_system_prompt`/`study_system_prompt` triple on
/// `tasks.EffectiveCondition`.
public struct ArmSystemPrompt: Sendable, Equatable {
    /// What reaches the renderer, and what a record's `systemPrompt` carries.
    public let effective: String?
    /// What a record's `systemPromptComposition` carries.
    public let stamp: SystemPromptCompositionStamp

    /// - Parameters:
    ///   - agent: the arm's agent persona; nil for baseline, every steering
    ///     condition, and every arm that is not agent-backed.
    ///   - study: the study's deployment frame (`manifest.systemPrompt`).
    public init(agent: String?, study: String?) {
        self.effective = SystemPromptComposition.compose(agent: agent, frame: study)
        self.stamp = SystemPromptCompositionStamp(agentText: agent, studyText: study)
    }
}

/// The PANEL twin of `SystemPromptCompositionStamp`: a cast seat's second term
/// is the CAST ENTRY's role text, never the study frame (the study frame
/// reaches no panel turn at all) — so the key is spelled `cast`, and the
/// difference in spelling is the point. `agent` stays first, so all three
/// stamp shapes read alike. Server twin:
/// `system_prompt.composition(frame_key="cast")`.
public struct PanelSystemPromptCompositionStamp: Codable, Equatable, Sendable {
    /// SHA-256 of the cast AGENT ARTIFACT's persona; nil ⇒ JSON `null`.
    public let agent: String?
    /// SHA-256 of the seat's CAST ENTRY role text; nil ⇒ JSON `null`.
    public let cast: String?

    public init(agent: String?, cast: String?) {
        self.agent = agent
        self.cast = cast
    }

    /// The stamp for one seat, from the two raw texts.
    public init(agentText: String?, castText: String?) {
        self.init(
            agent: SystemPromptComposition.hash(agentText),
            cast: SystemPromptComposition.hash(castText))
    }

    enum CodingKeys: String, CodingKey { case agent, cast }

    /// Explicit `encode`, not `encodeIfPresent`: nil must reach the wire as
    /// `null`, matching the server's always-present keys.
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(agent, forKey: .agent)
        try container.encode(cast, forKey: .cast)
    }
}

/// ONE panel seat's resolved arming: the effective text the seat generates
/// under, and the stamp that says which levels produced it.
///
/// The panel twin of `ArmSystemPrompt`, and built the same way — the two
/// together, from the same two inputs, so a turn record can never stamp a
/// composition its generation did not run under. Server twin: the
/// `system`/`system_composition` pair returned by
/// `multi_agent._runtime_settings`.
public struct SeatSystemPrompt: Sendable, Equatable {
    /// What reaches the renderer, and what the turn generated under.
    public let effective: String?
    /// What a turn record's `systemPromptComposition` carries.
    public let stamp: PanelSystemPromptCompositionStamp

    /// - Parameters:
    ///   - agent: the persona on the agent artifact cast into this seat; nil
    ///     for a baseline seat and for every agent with no persona — which,
    ///     today, is every agent in the workspace.
    ///   - cast: the seat's own cast-entry role text.
    public init(agent: String?, cast: String?) {
        self.effective = SystemPromptComposition.compose(agent: agent, frame: cast)
        self.stamp = PanelSystemPromptCompositionStamp(
            agentText: agent, castText: cast)
    }
}

/// The battery twin of `SystemPromptCompositionStamp`: a battery generation's
/// second term is the BATTERY FILE's declared arming text, never the study
/// frame — so the key is spelled `battery`, and the difference in spelling is
/// the point. Server twin: `system_prompt.composition(frame_key="battery")`.
public struct BatteryArmingCompositionStamp: Codable, Equatable, Sendable {
    public let agent: String?
    public let battery: String?

    public init(agent: String?, battery: String?) {
        self.agent = agent
        self.battery = battery
    }

    public init(agentText: String?, batteryText: String?) {
        self.init(
            agent: SystemPromptComposition.hash(agentText),
            battery: SystemPromptComposition.hash(batteryText))
    }

    enum CodingKeys: String, CodingKey { case agent, battery }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(agent, forKey: .agent)
        try container.encode(battery, forKey: .battery)
    }
}
