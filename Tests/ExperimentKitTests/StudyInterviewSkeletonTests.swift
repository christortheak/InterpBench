import Foundation
import Testing

@testable import ExperimentKit

/// The study interviews end with a "study" skeleton a coding assistant copies
/// into the pack it writes. It must show the current way to declare thinking
/// off (`reasoningEffort: "off"`, what `set-sampling --reasoning-effort off`
/// writes) rather than the legacy `qwenThinkingEnabled` boolean, and it must
/// still read as a manifest on this engine.
struct StudyInterviewSkeletonTests {

    static let interviews: [(name: String, text: String)] = [
        ("conceptStudy", StudyInterviewText.conceptStudy),
        ("agentComparison", StudyInterviewText.agentComparison),
        ("multiAgent", StudyInterviewText.multiAgent),
    ]

    /// The fenced JSON block that carries the skeleton pack.
    static func skeleton(_ text: String) throws -> String {
        let blocks = text.components(separatedBy: "```json\n").dropFirst()
            .compactMap { $0.components(separatedBy: "```").first }
        return try #require(
            blocks.first { $0.contains(#""name": "my-study""#) },
            "no study skeleton in the interview")
    }

    @Test(arguments: interviews.map(\.name))
    func theSkeletonDeclaresThinkingOffTheCurrentWay(_ name: String) throws {
        let text = try #require(Self.interviews.first { $0.name == name }?.text)
        let skeleton = try Self.skeleton(text)
        #expect(skeleton.contains(#""reasoningEffort": "off""#), "\(name)")
        #expect(!skeleton.contains("qwenThinkingEnabled"), "\(name)")

        let pack = try #require(
            try JSONSerialization.jsonObject(with: Data(skeleton.utf8)) as? [String: Any])
        let study = try #require(pack["study"] as? [String: Any])
        let manifest = try JSONDecoder().decode(
            ExperimentManifest.self, from: JSONSerialization.data(withJSONObject: study))
        #expect(manifest.reasoningEffort == "off")
        #expect(manifest.qwenThinkingEnabled == nil)
        #expect(manifest.resolvedReasoningEffort == .off)
    }

    /// The legacy key is still described (a manifest frozen under it still
    /// reads), and several samples per item are not called server-only.
    @Test(arguments: interviews.map(\.name))
    func theRulesDescribeTheCurrentSettings(_ name: String) throws {
        let text = try #require(Self.interviews.first { $0.name == name }?.text)
        #expect(text.contains("set-sampling --reasoning-effort"), "\(name)")
        #expect(text.contains(#"the legacy "qwenThinkingEnabled" boolean still reads"#), "\(name)")
        #expect(text.contains("run by either engine"), "\(name)")
        #expect(!text.contains("server/cluster runs only"), "\(name)")
    }
}
