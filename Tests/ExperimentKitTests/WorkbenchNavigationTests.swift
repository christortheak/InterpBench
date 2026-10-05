import Foundation
import Testing

@testable import ExperimentKit

/// The sidebar's two groups, and the Data section's tools split the same way.
///
/// The app target is not importable from here, so the section and tool names
/// are read from the app's own source, which is also the file that would
/// drift: a section added there without a place in a group fails this suite.
@Suite struct WorkbenchNavigationTests {

    static let repository = URL(filePath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    /// The raw values of the cases of `enum <name>` in an app source file.
    private func caseNames(ofEnum name: String, in file: String) throws -> [String] {
        let source = try String(
            contentsOf: Self.repository.appending(path: "Sources/SteerLabApp/\(file)"),
            encoding: .utf8)
        let start = try #require(source.range(of: "enum \(name):"))
        let body = source[start.upperBound...]
        let end = body.range(of: "var id")?.lowerBound ?? body.endIndex
        let regex = try NSRegularExpression(pattern: #"case \w+ = "([^"]+)""#)
        let text = String(body[..<end])
        return regex.matches(in: text, range: NSRange(text.startIndex..., in: text)).compactMap {
            Range($0.range(at: 1), in: text).map { String(text[$0]) }
        }
    }

    @Test func theBasicsAndAdvancedAreTheAgreedLists() {
        #expect(WorkbenchNavigation.sections.map(\.group) == [.basics, .advanced])
        #expect(
            WorkbenchNavigation.sections[0].names
                == ["Home", "Playground", "Data", "Agents", "Templates", "Studies", "Results", "Compute"])
        #expect(WorkbenchNavigation.sections[1].names == ["Probes", "Multi-Agent", "Analysis"])
        #expect(WorkbenchNavigation.Group.allCases.map(\.rawValue) == ["Basics", "Advanced"])
        #expect(WorkbenchNavigation.group(ofSection: "Studies") == .basics)
        #expect(WorkbenchNavigation.group(ofSection: "Probes") == .advanced)
        #expect(WorkbenchNavigation.group(ofSection: "Nowhere") == nil)
    }

    @Test func everySectionTheAppHasIsInExactlyOneGroup() throws {
        let sections = try caseNames(ofEnum: "WorkbenchSection", in: "WorkbenchSection.swift")
        #expect(sections.count == 11)
        let listed = WorkbenchNavigation.sections.flatMap(\.names)
        #expect(Set(listed) == Set(sections), "the sidebar groups and the app's sections differ")
        #expect(listed.count == Set(listed).count, "a section is listed twice")
    }

    @Test func theDataToolsSplitIntoBasicAndAdvanced() throws {
        let tools = try caseNames(ofEnum: "Tool", in: "SectionContainers.swift")
        #expect(tools == WorkbenchNavigation.basicDataTools + WorkbenchNavigation.advancedDataTools)
        #expect(WorkbenchNavigation.advancedDataTools == ["Adapter Training", "OptVec"])
        #expect(WorkbenchNavigation.isAdvancedDataTool("OptVec"))
        #expect(!WorkbenchNavigation.isAdvancedDataTool("Inventory"))
    }
}
