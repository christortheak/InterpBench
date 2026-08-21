import Foundation

/// A panel turn's prompt declared as STRUCTURE rather than as free text
/// (panel turn contracts, 2026-08-17).
///
/// ```json
/// "contract": {
///   "stage": "The panel has exchanged initial memoranda …",
///   "task": "Write a memorandum to your colleagues …",
///   "format": "Use exactly this format and nothing else:\nDisposition: affirm OR reverse",
///   "inputs": ["r1_judge-1", "r1_judge-2"],
///   "ownVoice": true,
///   "materialsTitle": "THE RECORD ON APPEAL"
/// }
/// ```
///
/// The renderer owns the LAYOUT — where the record sits relative to the
/// instruction, how another participant's output is fenced and attributed,
/// where the own-voice constraint lands — because that layout is what a
/// free-text template kept getting wrong: the case record appended after the
/// task, and the reader's own prior turn presented in the same third person as
/// everyone else's, which is how 75% of one round's opinions came back
/// carrying a colleague's signature block. A contract turn therefore declares
/// only its own content, never its own arrangement.
///
/// Server twin: `Server/steerlab_server/experiment/multi_agent.py`. Both
/// engines must render byte-identical prompts from the same contract; the
/// worked example in `docs/PANEL-TURN-CONTRACTS-SPEC.md` §2 is the fixture
/// both suites assert against.
public struct TurnContract: Codable, Sendable, Equatable {
    /// Fence title used when a contract declares none.
    public static let defaultMaterialsTitle = "SHARED MATERIALS"

    /// One or two sentences: where the scenario stands right now.
    public var stage: String
    /// The instruction. Non-empty — an empty task is a validate error, not a
    /// turn that quietly asks for nothing.
    public var task: String
    /// Output-format block, rendered verbatim after the own-voice block.
    public var format: String
    /// Ordered output labels of EARLIER turns to show this speaker, fenced,
    /// with own/other attribution derived from the producing turn's speaker.
    public var inputs: [String]
    /// Injects the standard own-voice constraint block.
    public var ownVoice: Bool
    /// Fence title for the shared-materials block, verbatim.
    public var materialsTitle: String

    public init(
        stage: String = "",
        task: String,
        format: String = "",
        inputs: [String] = [],
        ownVoice: Bool = true,
        materialsTitle: String = TurnContract.defaultMaterialsTitle
    ) {
        self.stage = stage
        self.task = task
        self.format = format
        self.inputs = inputs
        self.ownVoice = ownVoice
        self.materialsTitle = materialsTitle
    }

    /// Tolerant decode, deliberately unlike `TurnEndpoint`'s validating one: a
    /// contract's failure modes are AUTHORING mistakes with better wording
    /// available at validate time (which turn, which field), and refusing on
    /// decode would make the whole panel file unreadable in the picker instead
    /// of showing the researcher the one turn to fix. See
    /// `MultiAgentRunner.validate` for the four refusals.
    ///
    /// Defaults fire on absence for every field but one: `materialsTitle` is
    /// TRIMMED at decode, and blank-after-trim falls back to the default —
    /// matching the Python twin's `from_dict`, which stores the stripped
    /// title. It is a fence label; a blank or whitespace-padded one would
    /// render `=====  =====`, and the two engines must agree on the repair,
    /// not just the happy path.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.stage = try container.decodeIfPresent(String.self, forKey: .stage) ?? ""
        self.task = try container.decodeIfPresent(String.self, forKey: .task) ?? ""
        self.format = try container.decodeIfPresent(String.self, forKey: .format) ?? ""
        self.inputs = try container.decodeIfPresent([String].self, forKey: .inputs) ?? []
        self.ownVoice = try container.decodeIfPresent(Bool.self, forKey: .ownVoice) ?? true
        let title = (try container.decodeIfPresent(String.self, forKey: .materialsTitle) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        self.materialsTitle = title.isEmpty ? Self.defaultMaterialsTitle : title
    }

    /// The fields that carry author text, in the order validate reports them.
    /// Named here so the placeholder rule (§1.3) has one list, not four
    /// call sites that can drift apart.
    var textFields: [(field: String, text: String)] {
        [("stage", stage), ("task", task), ("format", format),
         ("materialsTitle", materialsTitle)]
    }
}
