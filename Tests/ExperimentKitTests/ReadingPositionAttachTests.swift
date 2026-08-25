import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// `experiment attach … --reading-position '<label>'` — the WRITER the reading
/// -position vocabulary shipped without.
///
/// The gap is the `extractionRendering` gap one layer down. `ReadingPosition`
/// has had a full vocabulary — `lastContentToken`, `turnCloseToken`,
/// `postInstruction`, and since 2026-08-25 `contentOffset` and
/// `meanContentFromToken` — a manifest that encodes every one of them, a
/// recipe identity that hashes them, and sidecar stamps that record where they
/// landed. What it did not have was a way to PIN one: `attach` took
/// `--pool-from K` and nothing else, so a study could declare exactly two
/// positions and every other one was reachable only from an ad-hoc extract
/// call that pins nothing at all. A grid that varies the reading position has
/// to run study-disciplined, so the writer exists.
///
/// Two halves, the same two the rendering writer holds:
///
/// 1. **The declaration reaches the consumers** — `ConceptRef.options
///    .readingPosition`, which is where `RecipeIdentity.required` and the
///    extraction paths read it — and it survives `duplicate`, the only
///    sanctioned way to iterate a frozen study.
/// 2. **Nothing else moves.** An attach that declares nothing writes the
///    manifest it always did, byte for byte, and `--pool-from K` and
///    `--reading-position 'mean from token K'` write the same bytes as each
///    other: one recipe, one encoding.
///
/// Server twin: `Server/tests/test_reading_position_attach.py`.
///
/// Serialized, and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct ReadingPositionAttachTests {

    // MARK: Harness

    static let model = "mlx-community/gemma-3-4b-it-4bit"
    static let chatTemplate = #"{"mode":"chatTemplate"}"#

    func withTempRoot<T>(_ body: (URL) async throws -> T) async throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "position-attach-\(UUID().uuidString)")
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body(temp)
    }

    @discardableResult
    func invoke(_ args: [String]) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(sink: .discarding).run(namespace: "experiment", args)
    }

    /// The manifest exactly as it sits on disk — the bytes, not a re-encode,
    /// because "byte-identical" is the claim under test.
    func manifestBytes(_ name: String) throws -> Data {
        try Data(contentsOf: ExperimentStore.directory
            .appending(components: name, "experiment.json"))
    }

    // MARK: - 1. the declaration reaches the consumers

    /// The whole vocabulary is declarable — that IS the gap — and each label
    /// lands as the position the manifest parser already reads.
    @Test func attachPinsEveryLabelInTheVocabulary() async throws {
        let cases: [(String, ReadingPosition)] = [
            ("last token", .lastToken),
            ("mean from token 50", .meanFromToken(50)),
            ("offset from end 3", .offsetFromEnd(3)),
            ("last content token", .lastContentToken),
            ("turn close token", .turnCloseToken),
            ("post-instruction 2", .postInstruction(2)),
            ("content offset 2", .contentOffset(2)),
            ("mean content from token 0", .meanContentFromToken(0)),
        ]
        try await withTempRoot { _ in
            for (index, (label, expected)) in cases.enumerated() {
                let name = "pinned\(index)"
                await invoke(["create", name, "--model", Self.model])
                // A template-aware role needs a rendering that HAS a turn; the
                // rest are rendering-independent.
                var args = [
                    "attach", name, "french",
                    ReadingPosition.declarationFlag, label,
                ]
                if expected.requiresTemplatedRendering {
                    args += ["--extraction-rendering", Self.chatTemplate]
                }
                let attached = await invoke(args)
                #expect(attached.exitCode == 0, "\(label): \(attached.envelope.state)")
                let ref = try #require(
                    try ExperimentStore.load(name: name).concepts.first)
                #expect(ref.options.readingPosition == expected, "\(label)")
                // The envelope reports the declaration rather than leaving an
                // agent to re-read the manifest to learn what it just wrote.
                #expect(attached.envelope.result?["readingPosition"] != nil, "\(label)")
            }
        }
    }

    /// Position IS identity: a declared position must reach the identity an
    /// extraction has to satisfy, or promotion would match an artifact read
    /// somewhere else entirely.
    @Test func aDeclaredPositionReachesTheIdentityAnExtractionMustSatisfy()
        async throws
    {
        try await withTempRoot { _ in
            var hashes: Set<String> = []
            let labels: [String?] = [
                nil, "mean from token 4", "mean content from token 4",
                "offset from end 3",
            ]
            for (index, label) in labels.enumerated() {
                let name = "identity\(index)"
                await invoke(["create", name, "--model", Self.model])
                var args = ["attach", name, "french"]
                if let label { args += [ReadingPosition.declarationFlag, label] }
                #expect(await invoke(args).exitCode == 0)
                let manifest = try ExperimentStore.load(name: name)
                let ref = try #require(manifest.concepts.first)
                hashes.insert(
                    RecipeIdentity.hash(
                        try RecipeIdentity.required(manifest: manifest, ref: ref)))
            }
            #expect(hashes.count == 4, "declared positions collapsed into one recipe")
        }
    }

    /// DECLARE → RESOLVE → STAMP, without a model: the position the manifest
    /// pins is the position that resolves against a rendered sequence, and the
    /// report an artifact carries says so. (The server twin runs the same loop
    /// through a real extraction; this engine's extraction needs a loaded
    /// container, and the chain the writer had to reach is this one.)
    @Test func thePinnedPositionResolvesAndStampsWhatWasDeclared() async throws {
        try await withTempRoot { _ in
            await invoke(["create", "stamped", "--model", Self.model])
            #expect(
                await invoke([
                    "attach", "stamped", "french",
                    ReadingPosition.declarationFlag, "content offset 1",
                    "--extraction-rendering", Self.chatTemplate,
                ]).exitCode == 0)
            let ref = try #require(
                try ExperimentStore.load(name: "stamped").concepts.first)
            let position = ref.options.readingPosition
            #expect(position == .contentOffset(1))

            let tokenizer = ExtractionRenderingCrossEngineTests.FixtureTokenizer()
            let tokens = [2, 105, 109, 107, 201, 202, 203, 106, 107, 105, 108, 107]
            let resolved = try position.resolve(
                tokens: tokens, tokenizer: tokenizer, renderingIsRaw: false)
            let rendering = try #require(ref.options.extractionRendering)
            let report = try #require(ReadingPositionResolutionReport.make(
                position: position, rendering: rendering,
                resolutions: [resolved]))
            #expect(report.requested == "content offset 1")
            #expect(report.mode == "contentOffset")
            #expect(report.parameter == 1)
            #expect(report.rendering == "chatTemplate")
            #expect(report.source == "last content token − 1")
            #expect(report.shapes.count == 1)
        }
    }

    /// The grand-mean path is a DIFFERENT constructor with its own pooled
    /// POLICY (token 50). A declaration is the researcher overriding that
    /// policy, which is exactly what a rendering×position grid does.
    @Test func theGrandMeanPathTakesTheDeclarationOverItsPoolDefault() async throws {
        try await withTempRoot { root in
            for concept in ["fear", "joy"] {
                let url = root.appending(
                    components: "prompts", "emotions", concept, "stories.jsonl")
                try FileManager.default.createDirectory(
                    at: url.deletingLastPathComponent(),
                    withIntermediateDirectories: true)
                try (#"{"concept":"\#(concept)","text":"a story about \#(concept)."}"#
                    + "\n").write(to: url, atomically: true, encoding: .utf8)
            }
            await invoke(["create", "grand", "--model", Self.model])
            #expect(
                await invoke([
                    "attach", "grand", "fear", "--method", "emotionGrandMean",
                    "--corpus", "joy",
                    ReadingPosition.declarationFlag, "mean content from token 4",
                    "--extraction-rendering", Self.chatTemplate,
                ]).exitCode == 0)
            let ref = try #require(
                try ExperimentStore.load(name: "grand").concepts
                    .first { $0.name == "fear" })
            #expect(ref.options.readingPosition == .meanContentFromToken(4))

            // …and without one, the method's own default is untouched.
            await invoke(["create", "grand2", "--model", Self.model])
            #expect(
                await invoke([
                    "attach", "grand2", "fear", "--method", "emotionGrandMean",
                    "--corpus", "joy",
                ]).exitCode == 0)
            let plain = try #require(
                try ExperimentStore.load(name: "grand2").concepts
                    .first { $0.name == "fear" })
            #expect(plain.options.readingPosition == .meanFromToken(50))
        }
    }

    /// `duplicate` is the ONLY way to iterate a frozen study, so a pin it
    /// dropped would silently re-derive every vector at last-token — a
    /// different recipe wearing the same name.
    @Test func duplicateCarriesThePinnedPosition() async throws {
        try await withTempRoot { _ in
            await invoke(["create", "origin", "--model", Self.model])
            #expect(
                await invoke([
                    "attach", "origin", "french",
                    ReadingPosition.declarationFlag, "turn close token",
                    "--extraction-rendering", Self.chatTemplate,
                ]).exitCode == 0)
            #expect(await invoke(["duplicate", "origin", "origin-v2"]).exitCode == 0)

            let copy = try ExperimentStore.load(name: "origin-v2")
            let ref = try #require(copy.concepts.first)
            #expect(ref.options.readingPosition == .turnCloseToken)
            // The two drafts demand the SAME recipe identity, which is what
            // makes the iteration an iteration rather than a new study.
            let source = try ExperimentStore.load(name: "origin")
            #expect(
                try RecipeIdentity.hash(
                    RecipeIdentity.required(
                        manifest: source, ref: #require(source.concepts.first)))
                    == RecipeIdentity.hash(
                        RecipeIdentity.required(manifest: copy, ref: ref)))
        }
    }

    // MARK: - 2. nothing else moves

    /// THE HARD CONSTRAINT at the writer: a study that says nothing about the
    /// reading position produces exactly the manifest it always did.
    @Test func anAbsentDeclarationIsByteIdenticalToToday() async throws {
        try await withTempRoot { _ in
            await invoke(["create", "silent", "--model", Self.model])
            await invoke(["create", "loud", "--model", Self.model])
            #expect(await invoke(["attach", "silent", "french"]).exitCode == 0)
            #expect(await invoke(["attach", "loud", "french"]).exitCode == 0)
            let silent = try normalized(try manifestBytes("silent"))
            let loud = try normalized(try manifestBytes("loud"))
            #expect(silent == loud)
            let ref = try #require(
                try ExperimentStore.load(name: "silent").concepts.first)
            #expect(ref.options.readingPosition == .lastToken)
        }
    }

    /// `--pool-from K` IS `--reading-position 'mean from token K'`. Two
    /// spellings of one recipe must produce one set of bytes, or a study that
    /// switched spellings would look like a different recipe to promote.
    @Test func theLegacySpellingAndTheLabelWriteTheSameRecipe() async throws {
        try await withTempRoot { _ in
            await invoke(["create", "legacy", "--model", Self.model])
            await invoke(["create", "labelled", "--model", Self.model])
            #expect(
                await invoke(["attach", "legacy", "french", "--pool-from", "50"])
                    .exitCode == 0)
            #expect(
                await invoke([
                    "attach", "labelled", "french",
                    ReadingPosition.declarationFlag, "mean from token 50",
                ]).exitCode == 0)
            let legacy = try normalized(try manifestBytes("legacy"))
            let labelled = try normalized(try manifestBytes("labelled"))
            #expect(legacy == labelled)
        }
    }

    // MARK: - 3. every refusal fires at declaration time

    @Test func declaringBothSpellingsRefusesNamingBothFlags() async throws {
        try await withTempRoot { _ in
            await invoke(["create", "both", "--model", Self.model])
            let outcome = await invoke([
                "attach", "both", "french", "--pool-from", "50",
                ReadingPosition.declarationFlag, "last content token",
            ])
            #expect(outcome.envelope.exitCode == 64)
            #expect(outcome.envelope.state == .blocked)
            let reason = try #require(outcome.failure?.reason)
            #expect(reason.contains(ReadingPosition.declarationFlag))
            #expect(reason.contains(ReadingPosition.poolFromFlag))
            #expect(reason.contains("a concept pins exactly one"))
            #expect(try ExperimentStore.load(name: "both").concepts.isEmpty)
        }
    }

    @Test func anUnknownLabelRefusesNamingTheEngineAndTheVocabulary() async throws {
        try await withTempRoot { _ in
            await invoke(["create", "unknown", "--model", Self.model])
            for label in ["middle token", "post-instruction 9",
                          "offset from end -2", "  "] {
                let outcome = await invoke([
                    "attach", "unknown", "french",
                    ReadingPosition.declarationFlag, label,
                ])
                #expect(outcome.envelope.exitCode == 64, "\(label) was accepted")
                #expect(outcome.envelope.error?.code == "usage", "\(label)")
                #expect(outcome.envelope.error?.repairAction.isEmpty == false,
                        "\(label) refused with no repair")
            }
            let outcome = await invoke([
                "attach", "unknown", "french",
                ReadingPosition.declarationFlag, "middle token",
            ])
            let reason = try #require(outcome.failure?.reason)
            #expect(reason.contains(ReadingPosition.declarationEngine))
            // The vocabulary is named, so the repair is executable without docs.
            let repair = try #require(outcome.envelope.error?.repairAction)
            #expect(repair.contains("content offset <k>"))
            #expect(repair.contains("mean content from token <n>"))
            #expect(try ExperimentStore.load(name: "unknown").concepts.isEmpty)
        }
    }

    /// DECLARATION-TIME BEATS EXTRACTION-TIME (the addGenerationPrompt-false
    /// precedent). This pin could never resolve — a raw stimulus has no turn —
    /// and the refusal names the flag that fixes it, in the same sentence the
    /// resolution path throws.
    @Test func aTemplateRoleUnderRawRenderingRefusesAtAttach() async throws {
        try await withTempRoot { _ in
            await invoke(["create", "raw", "--model", Self.model])
            for label in ["last content token", "turn close token",
                          "post-instruction 1", "content offset 2"] {
                let outcome = await invoke([
                    "attach", "raw", "french",
                    ReadingPosition.declarationFlag, label,
                ])
                #expect(outcome.envelope.exitCode == 64, "\(label) was accepted")
                let reason = try #require(outcome.failure?.reason, "\(label)")
                #expect(reason.contains("needs templated rendering"), "\(label)")
                #expect(outcome.envelope.error?.repairAction
                    .contains(ExtractionRendering.declarationFlag) == true, "\(label)")
            }
            // An explicitly-raw rendering is the same condition, said out loud.
            #expect(
                await invoke([
                    "attach", "raw", "french",
                    ReadingPosition.declarationFlag, "last content token",
                    "--extraction-rendering", #"{"mode":"raw"}"#,
                ]).envelope.exitCode == 64)
            #expect(try ExperimentStore.load(name: "raw").concepts.isEmpty)

            // …and the rendering-independent positions attach under raw
            // exactly as they always did.
            #expect(
                await invoke([
                    "attach", "raw", "french",
                    ReadingPosition.declarationFlag, "offset from end 3",
                ]).exitCode == 0)
            let ref = try #require(
                try ExperimentStore.load(name: "raw").concepts.first)
            #expect(ref.options.readingPosition == .offsetFromEnd(3))
        }
    }

    /// The store enforces every rule the CLI does — a panel or a future route
    /// that calls it directly cannot slip past the parse.
    @Test func theStoreEnforcesTheSameRulesAsTheFlag() throws {
        #expect(throws: ExperimentError.self) {
            try ExperimentStore.declaredReadingPosition(
                "last content token", poolFromToken: nil,
                extractionRendering: nil)
        }
        #expect(throws: ExperimentError.self) {
            try ExperimentStore.declaredReadingPosition(
                "middle token", poolFromToken: nil,
                extractionRendering: .chatTemplate())
        }
        #expect(throws: ExperimentError.self) {
            try ExperimentStore.declaredReadingPosition(
                "last token", poolFromToken: 50, extractionRendering: nil)
        }
        let declared = try ExperimentStore.declaredReadingPosition(
            "content offset 2", poolFromToken: nil,
            extractionRendering: .chatTemplate())
        #expect(declared == .contentOffset(2))
        let absent = try ExperimentStore.declaredReadingPosition(
            nil, poolFromToken: 50, extractionRendering: nil)
        #expect(absent == nil)
    }

    /// The flag is DECLARED on the verb — an undeclared one is an unknownFlag
    /// refusal, which is how the vocabulary stayed unwritable for as long as
    /// it did — and the help page names it.
    @Test func theFlagIsDeclaredOnTheVerbAndDocumented() {
        let spec = ExperimentCLIParser.specs.first { $0.label == "experiment attach" }
        #expect(spec?.valueFlags.contains(ReadingPosition.declarationFlag) == true)
        #expect(CLIFlagVocabulary.metavar(ReadingPosition.declarationFlag)
                == "<label>")
        #expect(CLIFlagVocabulary.purpose(ReadingPosition.declarationFlag)
                .contains("last content token"))
    }

    // MARK: helpers

    /// The manifest bytes with the two per-study stamps removed, so "the same
    /// manifest" means the same recipe rather than the same clock reading.
    func normalized(_ data: Data) throws -> String {
        var object = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        object.removeValue(forKey: "createdAt")
        object.removeValue(forKey: "name")
        let canonical = try JSONSerialization.data(
            withJSONObject: object, options: [.sortedKeys])
        return String(decoding: canonical, as: UTF8.self)
    }
}
