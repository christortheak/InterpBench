import Foundation

/// Read-only plans for the same permutations and composition sweep the app uses.
public enum StudyDesignExpansion {
    public struct Result: Encodable {
        public let mode: String
        public let designFileSHA256: String
        public let count: Int
        public let batch: JSONValue
    }

    public static func expand(_ data: Data, mode: String, reviewed: StudyDesignSnapshot) throws -> Result {
        guard let ref = reviewed.template.semanticScenario else { throw StudyPanelAuthoring.malformed("Expansion requires a panel design.") }
        let panel = try StudyTemplateStore.loadSemanticPanel(ref, workspaceRoot: reviewed.workspaceRoot)
        let ids = panel.agents.map(\.id)
        guard !ids.isEmpty, ids.count <= 64, Set(ids).count == ids.count else { throw StudyPanelAuthoring.malformed("Expansion requires 1–64 unique seats.") }
        let casting = try StudyDesignCastingInput.resolve(data, reviewed: reviewed)
        let assignments: [SeatAssignment]
        switch (mode, casting) {
        case ("permutations", .seating(let seats)):
            guard Set(seats.occupants.keys) == Set(ids) else { throw StudyPanelAuthoring.malformed("Cast every seat explicitly before expansion.") }
            let ordered = ids.map { seats.occupants[$0]! }
            var grouped: [SeatOccupant: Int] = [:]
            for occupant in ordered { grouped[occupant, default: 0] += 1 }
            var count = 1.0, placed = 0
            for n in grouped.values {
                for index in 1...n { count *= Double(placed + index) / Double(index) }
                placed += n
            }
            guard count < 4096.5 else { throw StudyPanelAuthoring.malformed("Expansion exceeds 4096 rows; narrow the explicit design.") }
            assignments = try PanelComposition.distinctAssignments(seatIDs: ids, occupants: ordered)
        case ("composition", .agents(let agents)) where agents.count == 1:
            let agent = agents[0]
            assignments = PanelComposition.compositionSweep(seatIDs: ids, agent: .agent(name: agent.record.artifact.name,
                artifactPath: agent.path, artifactHash: agent.file.sha256))
        default: throw StudyPanelAuthoring.malformed("Choose permutations with seats, or composition with exactly one reviewed agent.")
        }
        for occupant in Set(assignments.flatMap { Array($0.occupants.values) }) {
                if case .agent(_, let path, let hash) = occupant {
                    let agent = try StudyAgentAuthoring.reviewArtifact(path: path, workspaceRoot: reviewed.workspaceRoot, expectedFileSHA256: hash)
                    guard agent.record.artifact.baseModelID == reviewed.template.study.modelID else { throw StudyPanelAuthoring.malformed("A casting agent uses another base model.") }
                }
        }
        let rows: [JSONValue] = assignments.map { assignment in
            var seats: [String: JSONValue] = [:]
            for (id, occupant) in assignment.occupants {
                switch occupant {
                case .baseline: seats[id] = .null
                case .agent(_, let path, let hash): seats[id] = .object(["artifactPath": .string(path), "artifactFileSHA256": .string(hash)])
                }
            }
            return .object(["casting": .object(["seats": .object(seats)])])
        }
        return Result(mode: mode, designFileSHA256: reviewed.file.sha256, count: rows.count, batch: .object(["rows": .array(rows)]))
    }
}
