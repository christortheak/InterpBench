import Foundation

/// The shared CLI/HTTP casting document. A comparison supplies `agents` (an
/// empty array means baseline); a panel supplies `seats`, with null meaning
/// baseline and each treated seat naming a reviewed artifact file and digest.
public enum StudyDesignCastingInput {
    public static func resolve(_ data: Data, reviewed: StudyDesignSnapshot) throws -> StudyDesignInstantiation.Casting {
        guard let fields = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any], fields.count == 1 else {
            throw malformed()
        }
        func agent(_ value: Any) throws -> AgentArtifactSnapshot {
            guard let object = value as? [String: Any], Set(object.keys) == ["artifactPath", "artifactFileSHA256"],
                let path = object["artifactPath"] as? String, let hash = object["artifactFileSHA256"] as? String else {
                throw malformed()
            }
            return try StudyAgentAuthoring.reviewArtifact(path: path, workspaceRoot: reviewed.workspaceRoot, expectedFileSHA256: hash)
        }
        if let values = fields["agents"] as? [Any] {
            return .agents(try values.map(agent))
        }
        if let values = fields["seats"] as? [String: Any] {
            guard let ref = reviewed.template.semanticScenario else { throw malformed() }
            let seatIDs = PanelComposition.seatIDs(try StudyTemplateStore.loadSemanticPanel(ref, workspaceRoot: reviewed.workspaceRoot))
            var occupants: [String: SeatOccupant] = [:]
            for (id, value) in values {
                if value is NSNull { occupants[id] = .baseline }
                else {
                    let reviewedAgent = try agent(value)
                    occupants[id] = .agent(name: reviewedAgent.record.artifact.name,
                        artifactPath: reviewedAgent.path, artifactHash: reviewedAgent.file.sha256)
                }
            }
            return .seating(.init(seatIDs: seatIDs, occupants: occupants))
        }
        throw malformed()
    }

    private static func malformed() -> ExperimentError {
        .malformed("A casting contains exactly one of agents (an array) or seats (a seat-ID object). Each agent names artifactPath and artifactFileSHA256; a baseline seat is null.",
            repair: "Inspect the design and agent files, then supply only the explicit casting fields. Use {\"agents\":[]} for a baseline comparison.")
    }
}
