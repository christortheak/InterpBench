import Foundation

/// The agent guide's topics, as this client renders them.
///
/// A workspace's `AGENTS.md` is a short core contract (`AgentContract`). The
/// depth lives here, served on demand by `workspace guide [<topic>]`, so the
/// text always describes the installed client and can never go stale in a
/// workspace. The sources are `WorkspaceGuide/topics/*.md`;
/// `scripts/ci/check-workspace-bootstrap.py` renders them once per client —
/// this is the Mac rendering, with `steerlab-cli` commands — and the Python
/// client carries its own rendering under the same topic names.
public enum WorkspaceGuide {

    /// One topic: its name as typed, a one-line summary, and its full text.
    public struct Topic: Codable, Sendable, Equatable {
        public let name: String
        public let summary: String
        public let text: String
    }

    /// The executable whose commands these topics show.
    public static let client = "steerlab-cli"

    /// Every topic, in the order the core guide lists them.
    public static let topics: [Topic] = {
        // Generated text compiled into the binary: a decode failure is a build
        // defect, and the drift gate plus `WorkspaceGuideTests` catch it first.
        try! JSONDecoder().decode(
            [Topic].self, from: Data(WorkspaceBootstrapText.guideTopicsJSON.utf8))
    }()

    public static func topic(named name: String) -> Topic? {
        topics.first { $0.name == name }
    }

    /// The topic list a caller can choose from: names and summaries, no text.
    public static var index: JSONValue {
        .array(
            topics.map {
                .object(["name": .string($0.name), "summary": .string($0.summary)])
            })
    }
}
