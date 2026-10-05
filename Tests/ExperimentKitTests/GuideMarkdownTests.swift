import Foundation
import Testing

@testable import ExperimentKit

/// Method guides as formatted text in the app: the Markdown the guides use,
/// and the order a researcher reads them in (purpose, data shapes, the guide,
/// then the part for coding assistants).
@Suite struct GuideMarkdownTests {

    /// A small guide with every construct the shipped guides use.
    static let sample = """
        # Choose a direction

        Compare two populations before testing an intervention.

        Choose the contrast with the researcher.

        | Method | Inputs |
        |---|---|
        | meanDifference | prompts/concepts/<concept>/positive.jsonl |
        | lat | `a|b` paired rows \\| aligned |

        ## Dataset

        Each row looks like this:

        ```json
        {"id":"row-1","text":"A statement."}
        ```

        - first point,
          continued here
        - second point
        1. one
        2. two

        Bring the results home:

        ```sh
        steerlab science import <bundle> --root <workspace> --json
        ```

        ```python
        def decide(context):
            return []
        ```

        ## Coworker author prompt

        Help author the inputs.

        ### A deeper prompt heading

        Still for the assistant.

        ## After the prompts

        Back in the guide.
        """

    // MARK: Parsing

    @Test func headingsParagraphsListsTablesAndCodeParse() {
        let blocks = GuideMarkdown.parse(Self.sample)
        #expect(blocks.first == .heading(level: 1, text: "Choose a direction"))
        #expect(blocks.contains(.paragraph("Choose the contrast with the researcher.")))
        #expect(
            blocks.contains(
                .table(
                    header: ["Method", "Inputs"],
                    rows: [
                        ["meanDifference", "prompts/concepts/<concept>/positive.jsonl"],
                        // A pipe inside a code span, or escaped, is not a column break.
                        ["lat", "`a|b` paired rows | aligned"],
                    ])))
        #expect(blocks.contains(.code(language: "json", text: #"{"id":"row-1","text":"A statement."}"#)))
        #expect(blocks.contains(.list(ordered: false, items: ["first point, continued here", "second point"])))
        #expect(blocks.contains(.list(ordered: true, items: ["one", "two"])))
        #expect(
            blocks.contains(
                .code(
                    language: "python",
                    text: "def decide(context):\n    return []")))
        #expect(blocks.contains(.heading(level: 3, text: "A deeper prompt heading")))
    }

    // MARK: The order a researcher reads it in

    @Test func purposeAndDataShapesComeFirstAndCommandsMoveToTheAssistantPart() throws {
        let layout = GuideMarkdown.layout(Self.sample)
        #expect(layout.title == "Choose a direction")
        #expect(layout.purpose == [.paragraph("Compare two populations before testing an intervention.")])

        // Data shapes: the table, then the JSON example with the sentence
        // that introduced it.
        #expect(layout.dataShapes.count == 2)
        #expect(layout.dataShapes[0].section == nil)
        if case .table(let header, _) = layout.dataShapes[0].block {
            #expect(header == ["Method", "Inputs"])
        } else {
            Issue.record("the first data shape is not the table")
        }
        #expect(layout.dataShapes[1].section == "Dataset")
        #expect(layout.dataShapes[1].caption == "Each row looks like this:")

        // The guide's own flow keeps its prose, lists, and expert code, and
        // marks where the command was.
        #expect(layout.body.first == .block(.paragraph("Choose the contrast with the researcher.")))
        #expect(layout.body.contains(.block(.heading(level: 2, text: "Dataset"))))
        #expect(layout.body.contains(.movedToAssistants))
        #expect(layout.body.contains { if case .block(.code("python", _)) = $0 { true } else { false } })
        #expect(!layout.body.contains(.block(.paragraph("Bring the results home:"))))
        // A heading that only mentions prompts ends the prompt section and
        // stays in the guide.
        #expect(layout.body.last == .block(.paragraph("Back in the guide.")))
        #expect(layout.body.contains(.block(.heading(level: 2, text: "After the prompts"))))
        #expect(GuideMarkdown.isAssistantHeading("Prompt to offer a data author"))
        #expect(!GuideMarkdown.isAssistantHeading("After the prompts"))

        // The assistant part: the command with its introduction, then the
        // prompt section with everything under it.
        let assistants = layout.forCodingAssistants
        #expect(assistants.map(\.section) == ["Dataset", "Coworker author prompt"])
        #expect(assistants[0].blocks.first == .paragraph("Bring the results home:"))
        #expect(
            assistants[0].blocks.last
                == .code(language: "sh", text: "steerlab science import <bundle> --root <workspace> --json"))
        #expect(
            assistants[1].blocks == [
                .paragraph("Help author the inputs."),
                .heading(level: 3, text: "A deeper prompt heading"),
                .paragraph("Still for the assistant."),
            ])
    }

    @Test func codeIsToldApartByLanguageOrByItsFirstLine() {
        #expect(GuideMarkdown.codeKind(language: "sh", text: "anything") == .command)
        #expect(GuideMarkdown.codeKind(language: "JSON", text: "{}") == .data)
        #expect(GuideMarkdown.codeKind(language: "python", text: "steerlab x") == .other)
        #expect(GuideMarkdown.codeKind(language: nil, text: "\n  steerlab-cli science list --json") == .command)
        #expect(GuideMarkdown.codeKind(language: nil, text: "[1, 2]") == .data)
        #expect(GuideMarkdown.codeKind(language: nil, text: "plain words") == .other)
    }

    // MARK: Inline text

    @Test func placeholdersSurviveAndCodeSpansStayCode() {
        let text = GuideMarkdown.inline(
            "Use `science guide <method>` on prompts/<concept>/a.jsonl, **not** <https://example.org>.")
        let plain = String(text.characters)
        #expect(plain == "Use science guide <method> on prompts/<concept>/a.jsonl, not https://example.org.")
        let code = text.runs.filter { $0.inlinePresentationIntent?.contains(.code) == true }
        #expect(code.map { String(text[$0.range].characters) } == ["science guide <method>"])
        let bold = text.runs.filter { $0.inlinePresentationIntent?.contains(.stronglyEmphasized) == true }
        #expect(bold.map { String(text[$0.range].characters) } == ["not"])
        #expect(text.runs.contains { $0.link?.absoluteString == "https://example.org" })
    }

    // MARK: Every shipped guide

    /// Every block of every shipped guide lands in exactly one part, no
    /// command line stays in the guide's own flow, and every guide's author
    /// and reviewer prompts are in the assistant part.
    @Test func everyShippedGuideIsLaidOutWithNothingDropped() throws {
        let catalog = try ScienceCatalog.catalog()
        #expect(!catalog.methods.isEmpty)
        for method in catalog.methods {
            let text = try ScienceCatalog.guide(method.id).text
            let parsed = GuideMarkdown.parse(text)
            let layout = GuideMarkdown.layout(text)

            let promptHeadings = layout.forCodingAssistants.filter {
                GuideMarkdown.isAssistantHeading($0.section ?? "")
            }.count
            let bodyBlocks = layout.body.filter { $0 != .movedToAssistants }.count
            let placed =
                (layout.title == nil ? 0 : 1) + layout.purpose.count + layout.dataShapes.count
                + layout.dataShapes.filter { $0.caption != nil }.count + bodyBlocks
                + layout.forCodingAssistants.reduce(0) { $0 + $1.blocks.count } + promptHeadings
            #expect(placed == parsed.count, "\(method.id): \(parsed.count) blocks, \(placed) placed")

            #expect(layout.title != nil, "\(method.id) has no title")
            #expect(!layout.purpose.isEmpty, "\(method.id) has no purpose line")
            for element in layout.body {
                if case .block(.code(let language, let code)) = element {
                    #expect(
                        GuideMarkdown.codeKind(language: language, text: code) == .other,
                        "\(method.id) keeps a command or data block in its body")
                }
            }
            let sections = layout.forCodingAssistants.compactMap(\.section)
            #expect(sections.contains("Coworker author prompt"), "\(method.id)")
            #expect(sections.contains("Independent review prompt"), "\(method.id)")
            for shape in layout.dataShapes {
                if case .table(let header, let rows) = shape.block {
                    #expect(rows.allSatisfy { $0.count == header.count }, "\(method.id)")
                }
            }
        }
    }

    @Test func theGuideWithTablesAndCommandsShowsBoth() throws {
        let layout = GuideMarkdown.layout(try ScienceCatalog.guide("jlens").text)
        #expect(layout.dataShapes.contains { if case .table = $0.block { true } else { false } })
        #expect(layout.body.contains(.movedToAssistants))
        #expect(
            layout.forCodingAssistants.flatMap(\.blocks).contains {
                if case .code("sh", let text) = $0 { text.hasPrefix("steerlab ") } else { false }
            })
    }
}
