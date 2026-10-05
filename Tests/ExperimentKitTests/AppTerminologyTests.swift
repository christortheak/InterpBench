import Foundation
import Testing

@testable import ExperimentKit

/// The app's words for its own things, held by reading the app's sources
/// (the app target has no unit tests of its own).
///
/// The release review (D5) found the terms colliding: "agent" meant the
/// researcher's AI tool in one sheet and a configured model in the next;
/// "design" sat beside "template", with "instantiate", "cast", and "mint"
/// for the same few actions; the Playground still said "variant" for an
/// agent. The choices, now:
///
/// - an **agent** is a configured model under study; the researcher's AI tool
///   is a **coding assistant**;
/// - reusable study settings are a **template**: studies are *created* from
///   one, and a multi-agent study's seats are *assigned* (the `design`
///   command family keeps its name where a command is shown);
/// - a robustness report's agent side is shown as the **agent**, though its
///   files still record it as "variant".
///
/// Only text a researcher reads is checked: string literals and the lines
/// of multi-line string literals, outside comments.
struct AppTerminologyTests {

    private static var repoRoot: URL {
        URL(filePath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().standardizedFileURL
    }

    /// Retired phrase → the word that replaced it.
    private static let retired: [(phrase: String, use: String)] = [
        ("coding agent", "coding assistant"),
        ("your agent", "your coding assistant"),
        ("my agent", "my coding assistant"),
        ("Review agent draft", "Review the returned draft"),
        ("Save back to design", "Save back to template"),
        ("Save as new design", "Save as new template"),
        ("A saved design", "A saved template"),
        ("No saved designs", "No saved templates"),
        ("from this design", "from this template"),
        ("reload design", "reload template"),
        ("Instantiate", "Create Studies"),
        ("instantiated", "created from"),
        ("instantiation", "creating studies from"),
        ("minted", "created"),
        ("minting", "creating"),
        ("mints ", "creates"),
        ("mint an", "create an"),
        ("mint a ", "create a"),
        ("Save Casting", "Save Seat Assignments"),
        ("casting", "seat assignment"),
        ("Casting", "Seat assignment"),
        ("cast in", "assigned to"),
        ("recast", "assign again"),
        ("casts the seats", "assigns the seats"),
        (" variant ·", " agent ·"),
        ("· variant", "· agent"),
        ("variant artifact", "agent file"),
        ("model variant", "agent"),
        ("Python Compute", "the Python engine"),
        ("Local (MLX)", ComputeChoice.macQuickStart.title),
    ]

    /// Files another stream owns in this wave; their wording is changed
    /// there. Kept to the minimum, with the reason.
    private static let excused: [String: String] = [
        // Method guides view (stream C, wave 3): its one line about
        // "scenarios cast from" a template is reworded with that view.
        "Sources/SteerLabApp/ScienceGuidesView.swift": "stream C",
    ]

    /// The text a researcher could read on this line, or nil for code.
    private static func readableText(_ line: Substring, inMultiline: Bool) -> String? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        if trimmed.hasPrefix("//") { return nil }
        if inMultiline { return String(line) }
        var pieces: [String] = []
        var current = ""
        var inString = false
        var escaped = false
        var previous: Character?
        for character in line {
            if inString {
                if escaped {
                    escaped = false
                    current.append(character)
                } else if character == "\\" {
                    escaped = true
                    current.append(character)
                } else if character == "\"" {
                    pieces.append(current)
                    current = ""
                    inString = false
                } else {
                    current.append(character)
                }
            } else if character == "\"" {
                inString = true
            } else if character == "/", previous == "/" {
                // A trailing comment after code is not text a researcher
                // reads.
                break
            }
            previous = character
        }
        return pieces.isEmpty ? nil : pieces.joined(separator: " ")
    }

    @Test func theAppSaysAgentCodingAssistantAndTemplate() throws {
        let enumerator = try #require(
            FileManager.default.enumerator(
                at: Self.repoRoot.appending(components: "Sources", "SteerLabApp"),
                includingPropertiesForKeys: nil))
        var offenders: [String] = []
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            let relative = String(
                url.standardizedFileURL.path.dropFirst(Self.repoRoot.path.count + 1))
            guard Self.excused[relative] == nil,
                let source = try? String(contentsOf: url, encoding: .utf8)
            else { continue }
            var inMultiline = false
            for (index, line) in source.split(separator: "\n", omittingEmptySubsequences: false)
                .enumerated()
            {
                let delimiters = line.components(separatedBy: "\"\"\"").count - 1
                let text = Self.readableText(line, inMultiline: inMultiline && delimiters == 0)
                if delimiters % 2 == 1 { inMultiline.toggle() }
                guard let text else { continue }
                for (phrase, use) in Self.retired where text.contains(phrase) {
                    offenders.append("\(relative):\(index + 1): '\(phrase)' — say '\(use)'")
                }
            }
        }
        #expect(offenders.isEmpty, "\(offenders.joined(separator: "\n"))")
    }

    /// The robustness report keeps recording the agent's side as "variant";
    /// what is shown is the agent.
    @Test func theAgentSideOfARobustnessReportIsShownAsTheAgent() {
        #expect(VariantRobustnessReadout.armLabel("variant") == "agent")
        #expect(VariantRobustnessReadout.armLabel("Variant") == "Agent")
        #expect(VariantRobustnessReadout.armLabel("baseline") == "baseline")
        #expect(VariantRobustnessReadout.armLabel("tie") == "tie")
        let output = FineTuningPanel.LiveRobustnessOutput(
            kind: "Coherence", index: 2, total: 5, prompt: "p", side: "Variant",
            output: "", isComplete: false)
        #expect(output.title == "Coherence 2/5 · Agent")
        // The identity still uses the stored side.
        #expect(output.id == "Coherence-2-Variant")
    }

    /// Lineage and batch lines the app shows from ExperimentKit name the
    /// template, and point at buttons that exist by those names.
    @Test func templateLinesNameTheTemplateAndTheRealButtons() throws {
        let library = try String(
            contentsOf: Self.repoRoot.appending(
                path: "Sources/ExperimentKit/StudyDesignLibrary.swift"), encoding: .utf8)
        #expect(library.contains("use Save as new template"))
        #expect(!library.contains("Save as new design"))
        let actions = try String(
            contentsOf: Self.repoRoot.appending(
                path: "Sources/SteerLabApp/StudyDesignActionsView.swift"), encoding: .utf8)
        #expect(actions.contains("Button(\"Save as new template\")"))
        #expect(actions.contains("Button(\"Save back to template"))
    }
}
