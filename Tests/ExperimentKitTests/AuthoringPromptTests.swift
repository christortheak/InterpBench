import Foundation
import Testing

@testable import ExperimentKit

/// `authoring prompt <kind>` — the generation-prompt emitter, and its registry.
///
/// The gap it closes: a study is blocked by MISSING DATA far more often than by
/// a missing verb, and every re-improvised generation prompt re-learned the
/// same lessons the hard way while losing the audit numbers, which are the only
/// part an acceptor can check.
///
/// What this suite is really guarding is the seam between EMITTING and
/// ACCEPTING. The verb renders text and writes nothing into the workspace; its
/// `nextAction` names a second reviewer; and the hash it stamps is what makes
/// the exact wording a corpus was generated from recoverable afterwards. Every
/// test below is about one of those three.
///
/// Server twin: `Server/tests/test_authoring_prompts.py`.
@Suite(.serialized) struct AuthoringPromptTests {

    /// One legal argument set per kind — enough to emit. Kept beside the
    /// registry so a kind added without an emittable example fails here rather
    /// than in a study.
    static let legalArguments: [String: [String: String]] = [
        "contrastive-pairs": [
            "concept": "tidiness", "positive": "P", "negative": "N",
        ],
        "choice-prompts": ["concept": "tidiness", "decision": "D"],
        "validation-set": [
            "concept": "tidiness", "positive": "P", "negative": "N",
        ],
        "reader-pairs": [
            "concept": "tidiness", "positive": "P", "negative": "N",
            "templateID": "T",
        ],
        "battery": [:],
    ]

    static let kindIDs = AuthoringPrompts.kinds.map(\.id)

    // MARK: - The registry

    /// A kind in the table with no file is a verb that refuses when it is
    /// used, which is the one moment nobody is looking at this table.
    @Test func everyDeclaredKindHasATemplateOnDisk() throws {
        for kind in AuthoringPrompts.kinds {
            let url = AuthoringPrompts.templateURL(kind.templateFileName)
            #expect(
                FileManager.default.fileExists(atPath: url.path),
                "\(kind.id) has no \(url.path)")
        }
    }

    /// The DIRECTORY IS THE INDEX, so a stray `.md` at the top level would look
    /// like a kind the verb cannot reach. `_`-prefixed files are partials and
    /// README.md is prose.
    @Test func theRegistryDirectoryHoldsNoUndeclaredKind() throws {
        let seed = try CodeResources.workspaceSeed()
        let directory = seed.appending(
            path: AuthoringPrompts.registryRelativeDirectory)
        let names = try FileManager.default
            .contentsOfDirectory(atPath: directory.path)
            .filter {
                $0.hasSuffix(".md") && !$0.hasPrefix("_") && $0 != "README.md"
            }
            .map { String($0.dropLast(3)) }
        #expect(Set(names) == Set(Self.kindIDs))
    }

    /// A plausible default for "what is the positive pole" is a study nobody
    /// declared. Counts and shape choices may default; descriptions may not.
    @Test func nothingThatDescribesTheStudyCarriesADefault() {
        let describing: Set = [
            "concept", "positive", "negative", "decision", "templateID",
        ]
        for kind in AuthoringPrompts.kinds {
            for parameter in kind.parameters
            where describing.contains(parameter.key) {
                #expect(
                    parameter.isRequired,
                    "\(kind.id).\(parameter.key) defaults")
            }
            #expect(kind.purpose.hasSuffix("."))
            #expect(!kind.destination.isEmpty)
        }
    }

    // MARK: - Emission

    /// An unsubstituted `{{placeholder}}` is a hole an LLM would answer
    /// literally — so the emitter leaves unknown keys VERBATIM (never blank)
    /// and this test is what notices.
    @Test(arguments: AuthoringPromptTests.kindIDs)
    func everyKindEmitsWithNoPlaceholderLeft(_ kindID: String) throws {
        let emission = try AuthoringPrompts.emit(
            kind: kindID, arguments: Self.legalArguments[kindID] ?? [:])
        #expect(
            !emission.text.contains("{{"),
            "\(kindID) left an unsubstituted placeholder")
    }

    @Test(arguments: AuthoringPromptTests.kindIDs)
    func everyEmissionCarriesItsHashAndItsAuditBattery(_ kindID: String) throws {
        let emission = try AuthoringPrompts.emit(
            kind: kindID, arguments: Self.legalArguments[kindID] ?? [:])
        let first = try #require(emission.text.split(separator: "\n").first)
        #expect(first.hasPrefix("<!-- steerlab authoring prompt"))
        #expect(first.contains("sha256:\(emission.promptSpecHash)"))
        #expect(emission.promptSpecHash.count == 64)
        let isHex = emission.promptSpecHash.allSatisfy(\.isHexDigit)
        #expect(isHex)
        // The battery is the part an acceptor re-runs; a prompt without one is
        // a request with no way to check the answer.
        #expect(emission.text.contains("audit battery — compute these and report them"))
        #expect(emission.text.contains("Compute every audit number. Never assert one."))
        #expect(emission.text.contains("prompt is never its acceptor"))
    }

    @Test(arguments: AuthoringPromptTests.kindIDs)
    func emissionIsDeterministic(_ kindID: String) throws {
        let arguments = Self.legalArguments[kindID] ?? [:]
        let first = try AuthoringPrompts.emit(kind: kindID, arguments: arguments)
        let second = try AuthoringPrompts.emit(kind: kindID, arguments: arguments)
        #expect(first.text == second.text)
        #expect(first.promptSpecHash == second.promptSpecHash)
    }

    /// `promptSpecHash` is what a study's provenance cites, so it must be
    /// reproducible from the named files by anyone holding them.
    @Test func theHashIsOverThePartialsAndTheTemplateInAssemblyOrder() throws {
        let emission = try AuthoringPrompts.emit(
            kind: "reader-pairs",
            arguments: Self.legalArguments["reader-pairs"] ?? [:])
        let directory = AuthoringPrompts.registryRelativeDirectory
        #expect(
            emission.templateFiles == [
                "\(directory)/_reader-shape-contentPair.md",
                "\(directory)/_discipline.md",
                "\(directory)/_delivery.md",
                "\(directory)/reader-pairs.md",
            ])
        var joined = Data()
        for relative in emission.templateFiles {
            let name = String(relative.split(separator: "/").last ?? "")
            let url = AuthoringPrompts.templateURL(name)
            try joined.append(Data(contentsOf: url))
        }
        #expect(ExperimentStore.sha256Hex(joined) == emission.promptSpecHash)
    }

    /// The shapes fit DIFFERENT contrasts. One prompt for both, or one hash for
    /// both, would make the distinction unrecoverable from a delivery.
    @Test func theTwoReaderShapesProduceDifferentPromptsAndHashes() throws {
        var arguments = Self.legalArguments["reader-pairs"] ?? [:]
        arguments["shape"] = "contentPair"
        let content = try AuthoringPrompts.emit(
            kind: "reader-pairs", arguments: arguments)
        arguments["shape"] = "singleStimulus"
        let single = try AuthoringPrompts.emit(
            kind: "reader-pairs", arguments: arguments)
        #expect(content.promptSpecHash != single.promptSpecHash)
        #expect(content.text.contains("positiveStimulus"))
        #expect(single.text.contains("read under TWO templates"))
    }

    /// Editing the wording for a study is the point — and the emission must
    /// then cite the edited bytes, not the shipped ones, or the provenance is a
    /// citation of a prompt nobody used.
    @Test func aWorkspaceCopyWinsAndChangesTheHash() throws {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "authoring-prompt-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temp) }
        try FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        let shipped = try AuthoringPrompts.emit(
            kind: "battery", arguments: [:], workspaceRoot: temp)
        #expect(!shipped.fromWorkspaceCopy)

        let directory = temp.appending(
            path: AuthoringPrompts.registryRelativeDirectory)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let original = try String(
            contentsOf: AuthoringPrompts.templateURL("_discipline.md"),
            encoding: .utf8)
        try (original + "\n**6. One more rule for this study.**\n").write(
            to: directory.appending(component: "_discipline.md"),
            atomically: true, encoding: .utf8)
        let edited = try AuthoringPrompts.emit(
            kind: "battery", arguments: [:], workspaceRoot: temp)
        #expect(edited.fromWorkspaceCopy)
        #expect(edited.promptSpecHash != shipped.promptSpecHash)
        #expect(edited.text.contains("One more rule for this study"))
    }

    /// A parameter value containing `{{count}}` must not then be substituted:
    /// the seam is one-way, and a study that talks about braces is not a
    /// template.
    @Test func aStudysOwnWordsAreDataNotTemplate() throws {
        let emission = try AuthoringPrompts.emit(
            kind: "choice-prompts",
            arguments: ["concept": "c", "decision": "Pick {{count}} of them."])
        #expect(emission.text.contains("Pick {{count}} of them."))
    }

    @Test func theDestinationIsRenderedOverTheSameParameters() throws {
        let emission = try AuthoringPrompts.emit(
            kind: "reader-pairs",
            arguments: Self.legalArguments["reader-pairs"] ?? [:])
        #expect(emission.destination == "prompts/readers/tidiness/pairs.jsonl")
        #expect(emission.text.contains(emission.destination))
    }

    // MARK: - Refusals

    func malformedRepair(_ error: any Error) -> String? {
        (error as? ExperimentError)?.malformedInvocation?.repairAction
    }

    @Test func anUnknownKindIsRefusedWithTheRoster() throws {
        do {
            _ = try AuthoringPrompts.emit(kind: "bogus", arguments: [:])
            Issue.record("an unknown kind emitted")
        } catch {
            #expect("\(error)".contains("known kinds:"))
            #expect(malformedRepair(error) == AuthoringPrompts.unknownKindRepair())
            #expect((error as? ExperimentError)?.lifecycleRefusal == nil)
        }
    }

    @Test func aMissingDescriptionIsRefusedAndSaysWhy() throws {
        do {
            _ = try AuthoringPrompts.emit(
                kind: "contrastive-pairs", arguments: ["concept": "c"])
            Issue.record("a pole-less contrastive prompt emitted")
        } catch {
            #expect("\(error)".contains("--positive"))
            #expect("\(error)".contains("--negative"))
            #expect("\(error)".contains("a study nobody declared"))
        }
    }

    /// Ignoring it leaves a caller convinced they set something.
    @Test func aParameterTheKindDoesNotOwnIsRefusedNotIgnored() throws {
        do {
            _ = try AuthoringPrompts.emit(
                kind: "battery", arguments: ["concept": "c"])
            Issue.record("a foreign parameter was accepted")
        } catch {
            #expect("\(error)".contains("takes no parameter 'concept'"))
        }
    }

    @Test func anUnknownReaderShapeIsRefusedWithTheVocabulary() throws {
        var arguments = Self.legalArguments["reader-pairs"] ?? [:]
        arguments["shape"] = "both"
        do {
            _ = try AuthoringPrompts.emit(
                kind: "reader-pairs", arguments: arguments)
            Issue.record("an unknown shape emitted")
        } catch {
            #expect(
                "\(error)".contains(
                    "known shapes: contentPair, singleStimulus"))
        }
    }

    /// Never a prompt with a hole in it. The gate is the workspace-state one,
    /// because the repair is to restore a file.
    @Test func aMissingRegistryFileIsATypedPrerequisiteRefusal() throws {
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "authoring-empty-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: temp) }
        try FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        do {
            _ = try AuthoringPrompts.emit(
                kind: "battery", arguments: [:],
                workspaceRoot: temp, seedRoot: temp)
            Issue.record("an empty registry emitted a prompt")
        } catch {
            let refusal = (error as? ExperimentError)?.lifecycleRefusal
            #expect(refusal?.gate == .missingPrerequisite)
            #expect(
                "\(error)".contains(
                    "the authoring-prompt registry has no"))
            #expect(
                refusal?.repairAction
                    == AuthoringPrompts.missingTemplateRepair(
                        "\(AuthoringPrompts.registryRelativeDirectory)/"
                            + "_discipline.md"))
        }
    }

    // MARK: - The verb

    func invoke(_ args: [String]) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(sink: .discarding).run(
            namespace: "authoring", args)
    }

    @Test func theVerbEmitsAndReportsItsProvenance() async throws {
        let outcome = await invoke(
            ["prompt", "contrastive-pairs", "--concept", "tidiness",
             "--positive", "P", "--negative", "N"])
        #expect(outcome.envelope.state == .ready)
        let result = try #require(outcome.envelope.result)
        #expect(result["kind"] == .string("contrastive-pairs"))
        #expect(result["destination"] == .string("prompts/concepts/tidiness/"))
        if case .string(let hash)? = result["promptSpecHash"] {
            #expect(hash.hasPrefix("sha256:"))
        } else {
            Issue.record("no promptSpecHash in the result")
        }
        if case .string(let prompt)? = result["prompt"] {
            #expect(prompt.hasPrefix("<!-- steerlab authoring prompt"))
        } else {
            Issue.record("no prompt in the result")
        }
    }

    /// The emitter is never the acceptor. `changed` is false and the next
    /// action requires a human — anything else would read as "installed".
    @Test func theVerbWritesNothingAndSaysAHumanMustReview() async throws {
        let outcome = await invoke(["prompt", "battery"])
        #expect(outcome.envelope.changed == false)
        #expect(outcome.envelope.nextAction?.requiresHuman == true)
        #expect(
            outcome.envelope.nextAction?.verb.contains("SECOND reviewer")
                == true)
    }

    @Test func aForeignFlagIsRefusedByTheVerbToo() async throws {
        let outcome = await invoke(["prompt", "battery", "--concept", "c"])
        #expect(outcome.envelope.state == .blocked)
        #expect(
            outcome.envelope.error?.reason.contains("does not take --concept")
                == true)
    }

    /// This verb exists on BOTH surfaces, so its repairs name a program — and
    /// the Mac's spelling is the default one.
    @Test func theVerbNamesItsOwnBinaryInARepair() async throws {
        let outcome = await invoke(["prompt", "bogus"])
        #expect(outcome.envelope.state == .blocked)
        #expect(
            outcome.envelope.error?.repairAction
                .hasPrefix("steerlab-cli authoring") == true)
    }

    // MARK: - The cross-engine literals

    /// Copied from `authoring_prompts.THRESHOLDS`. Two engines emitting
    /// different numbers for one kind would be two different instruments
    /// wearing one name. Server twin test:
    /// `test_thresholds_match_the_swift_literal`.
    @Test func thresholdsMatchTheServerLiteral() {
        #expect(
            AuthoringPrompts.thresholds == [
                "stemCapPercent": "40",
                "frameCapPercent": "25",
                "parityPercent": "10",
                "lengthDeltaWords": "10",
                "minWords": "60",
                "maxWords": "90",
                "balanceLowPercent": "45",
                "balanceHighPercent": "55",
                "optionLengthRatio": "3",
                "minItems": "10",
                "minOptions": "3",
                "maxTokens": "24",
            ])
    }

    /// Server twin test: `test_the_kind_roster_matches_the_swift_literal`.
    @Test func kindsMatchTheServerLiteral() {
        #expect(
            Self.kindIDs == [
                "contrastive-pairs", "choice-prompts", "validation-set",
                "reader-pairs", "battery",
            ])
        #expect(AuthoringPrompts.readerShapes == ["contentPair", "singleStimulus"])
        #expect(
            AuthoringPrompts.registryRelativeDirectory
                == "prompts/authoring-prompts")
    }

    /// The keys ARE the wire vocabulary an HTTP caller or a test uses, so a
    /// rename on one engine is a rename of the surface. Server twin test:
    /// `test_the_parameter_table_matches_the_swift_literal`.
    @Test func parametersMatchTheServerLiteral() {
        func table(_ id: String) -> [[String]] {
            (AuthoringPrompts.kind(id)?.parameters ?? []).map {
                [$0.key, $0.flag, $0.defaultValue ?? "—"]
            }
        }
        #expect(
            table("contrastive-pairs") == [
                ["concept", "--concept", "—"],
                ["positive", "--positive", "—"],
                ["negative", "--negative", "—"],
                ["count", "--count", "48"],
                ["validationCount", "--validation-count", "40"],
            ])
        #expect(
            table("choice-prompts") == [
                ["concept", "--concept", "—"],
                ["decision", "--decision", "—"],
                ["count", "--count", "40"],
            ])
        #expect(
            table("validation-set") == [
                ["concept", "--concept", "—"],
                ["positive", "--positive", "—"],
                ["negative", "--negative", "—"],
                ["count", "--count", "40"],
            ])
        #expect(
            table("reader-pairs") == [
                ["concept", "--concept", "—"],
                ["positive", "--positive", "—"],
                ["negative", "--negative", "—"],
                ["templateID", "--template-id", "—"],
                ["shape", "--shape", "contentPair"],
                ["count", "--count", "40"],
                ["heldOut", "--held-out", "10"],
            ])
        #expect(
            table("battery") == [
                ["name", "--name", "capability"],
                ["count", "--count", "20"],
            ])
    }

    /// Server twin test: `test_the_repairs_match_the_swift_literals`.
    @Test func repairsMatchTheServerLiterals() throws {
        #expect(
            AuthoringPrompts.unknownKindRepair()
                == "steerlab-cli authoring prompt <contrastive-pairs|"
                + "choice-prompts|validation-set|reader-pairs|battery> …")
        let kind = try #require(AuthoringPrompts.kind("contrastive-pairs"))
        let missing = kind.parameters.filter {
            $0.key == "positive" || $0.key == "negative"
        }
        #expect(
            AuthoringPrompts.missingParametersRepair(
                kind: kind, missing: missing)
                == "steerlab-cli authoring prompt contrastive-pairs "
                + "--positive \"…\" --negative \"…\"")
        #expect(
            AuthoringPrompts.missingTemplateRepair(
                "prompts/authoring-prompts/x.md")
                == "restore prompts/authoring-prompts/x.md from the shipped "
                + "seed tree, or re-create the workspace with steerlab-cli "
                + "workspace init <path>  (the emitter reads the workspace's "
                + "copy first and the shipped copy second, and refuses rather "
                + "than emitting a prompt with a hole in it)")
    }
}
