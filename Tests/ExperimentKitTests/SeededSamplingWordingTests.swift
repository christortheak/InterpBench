import Foundation
import Testing

@testable import ExperimentKit

/// The Mac engine samples each record from its own seeded stream (seeded
/// MLX sampling, 0.9.6). Help text, the CLI reference, the study interviews,
/// and the browser companion all once said a local run took temperature 0
/// only. A researcher who believed them would send a warm study to a server
/// for no reason, and the companion's Run Study button refused one outright.
/// These checks keep the retired claim from coming back.
@Suite struct SeededSamplingWordingTests {

    private static var root: URL { CodeResources.compiledCheckoutPath }

    /// Phrases that only ever described the retired greedy-only rule.
    static let retiredClaims = [
        "greedy-only",
        "refuses any other value",
        "requires Temp = 0",
        "currently require 0",
        "local runs are greedy",
        "Only the Python server reads this",
        "server/cluster runs only",
        "0 for locally-measured runs",
        "does not expose a per-run seed",
    ]

    /// Maintained sources a researcher or a coding assistant reads. Dated
    /// documents and the changelog keep their history and are not listed.
    static let maintainedSources = [
        "docs/CLI-REFERENCE.md",
        "web/index.html",
        "Sources/ExperimentKit/CLIHelp.swift",
        "Sources/ExperimentKit/StudyInterviewText.swift",
        "Sources/SteerLabApp/InfoPopover.swift",
        "WorkspaceSeed/prompts/study-interviews/study-conceptStudy.md",
        "WorkspaceSeed/prompts/study-interviews/study-agentComparison.md",
        "WorkspaceSeed/prompts/study-interviews/study-multiAgent.md",
    ]

    @Test func setSamplingTemperatureHelpDescribesSeededSampling() {
        let help = CLIFlagVocabulary.purpose(
            "--temperature", verb: "experiment set-sampling")
        #expect(!help.contains("refuses"), "\(help)")
        #expect(help.contains("greedy"), "\(help)")
        #expect(help.contains("seed"), "\(help)")
    }

    @Test func maintainedSourcesCarryNoRetiredGreedyOnlyClaim() throws {
        for path in Self.maintainedSources {
            let url = Self.root.appending(path: path)
            guard FileManager.default.fileExists(atPath: url.path) else { continue }
            let text = try String(contentsOf: url, encoding: .utf8)
            for claim in Self.retiredClaims {
                let stillSaysIt = text.contains(claim)
                #expect(
                    !stillSaysIt,
                    "\(path) still says \"\(claim)\"; the Mac engine runs any temperature ≥ 0 with a seed per record")
            }
        }
    }

    /// The companion's Run Study button is disabled only by a running job or
    /// a verification problem — never by the study's temperature.
    @Test(.enabled(
        if: FileManager.default.fileExists(
            atPath: CodeResources.compiledCheckoutPath.appending(path: "web/index.html").path),
        "web/index.html is not in this checkout"))
    func companionRunButtonIsNotGatedOnTemperature() throws {
        let source = try String(
            contentsOf: Self.root.appending(path: "web/index.html"), encoding: .utf8)
        let start = try #require(source.range(of: #"<button id="ex-run""#))
        let end = try #require(source.range(of: ">", range: start.upperBound..<source.endIndex))
        let tag = String(source[start.lowerBound..<end.lowerBound])
        #expect(!tag.contains("temperature"), "\(tag)")
    }
}
