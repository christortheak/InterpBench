import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// `experiment attach … --extraction-rendering '<json>'` — the WRITER the
/// rendering option shipped without.
///
/// `extractionRendering` landed 2026-08-24 with every CONSUMER live: recipe
/// identity hashes it, the α denominator follows it, the template-aware
/// reading positions require it, the sidecars stamp it, and two refusals ask
/// out loud for it to be declared. Nothing could declare it. There was no
/// flag, no route field, and no store parameter — so the engines' own repair
/// text named a command that did not exist.
///
/// This suite is that command's contract, and it has exactly two halves:
///
/// 1. **The declaration reaches the consumers.** A chat-template declaration
///    typed at `attach` lands in `ConceptRef.options.extractionRendering`,
///    which is where `RecipeIdentity.required` and the extraction paths read
///    it, and it survives `duplicate` — the only sanctioned way to iterate a
///    frozen study.
/// 2. **Nothing else moves.** An attach that declares nothing, and an attach
///    that declares `{"mode": "raw"}`, write BYTE-IDENTICAL manifests. Raw is
///    the legacy rendering; saying it out loud may not fork a recipe's
///    identity, its validation scope, or its freeze hash away from every
///    study frozen before the option existed.
///
/// Serialized, and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct ExtractionRenderingAttachTests {

    // MARK: Harness

    static let model = "mlx-community/gemma-3-4b-it-4bit"

    func withTempRoot<T>(_ body: (URL) async throws -> T) async throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "rendering-attach-\(UUID().uuidString)")
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

    /// The round trip: declared at attach, read by the identity the
    /// extraction must satisfy, and matched by the stamp an artifact carries.
    /// No model is loaded — the manifest→identity→stamp chain is the part the
    /// writer had to reach, and it is entirely pure.
    @Test func aDeclaredRenderingReachesTheIdentityAnExtractionMustSatisfy() async throws {
        try await withTempRoot { _ in
            #expect(await invoke(["create", "declared", "--model", Self.model]).exitCode == 0)
            let attached = await invoke(
                [
                    "attach", "declared", "french",
                    "--extraction-rendering", #"{"mode":"chatTemplate"}"#,
                ])
            #expect(attached.exitCode == 0)
            #expect(attached.envelope.changed)

            let manifest = try ExperimentStore.load(name: "declared")
            let ref = try #require(manifest.concepts.first)
            // WHERE THE CONSUMERS READ IT.
            let declared = try #require(ref.options.extractionRendering)
            #expect(declared.mode == .chatTemplate)
            // Resolved defaults written out, so the manifest says what the
            // extraction will do without a reader knowing the type's defaults.
            #expect(declared.addGenerationPrompt == true)
            #expect(declared.reasoningEffort == .off)
            #expect(declared.qwenThinkingEnabled == nil)
            #expect(ref.options.resolvedExtractionRendering.isRaw == false)

            // The identity the extraction must reproduce carries it…
            let required = try RecipeIdentity.required(manifest: manifest, ref: ref)
            #expect(required.extractionRendering?.mode == .chatTemplate)
            // …and an artifact stamped with the SAME rendering matches, while
            // one extracted raw does not. That is the whole point of the
            // declaration: the two are no longer interchangeable.
            var raw = required
            raw.extractionRendering = nil
            #expect(RecipeIdentity.hash(required) != RecipeIdentity.hash(raw))

            // The envelope reports the declaration rather than leaving an
            // agent to re-read the manifest to learn what it just wrote.
            #expect(attached.envelope.result?["extractionRendering"] != nil)
        }
    }

    /// Every rendering PARAMETER survives the flag, not just the mode.
    @Test func theRenderingParametersSurviveTheFlagVerbatim() async throws {
        try await withTempRoot { _ in
            // A thinking-on rendering is only declarable on a family with a
            // thinking mode (the attach gate refuses it on any other).
            await invoke(["create", "params", "--model", "Qwen/Qwen3-0.6B"])
            #expect(
                await invoke(
                    [
                        "attach", "params", "french", "--extraction-rendering",
                        #"{"mode":"chatTemplate","qwenThinkingEnabled":true,"systemPrompt":"be brief"}"#,
                    ]
                ).exitCode == 0)
            let ref = try #require(
                try ExperimentStore.load(name: "params").concepts.first)
            let declared = try #require(ref.options.extractionRendering)
            // The legacy boolean lands resolved in the effort spelling.
            #expect(declared.reasoningEffort == .xhigh)
            #expect(declared.resolvedQwenThinkingEnabled == true)
            #expect(declared.systemPrompt == "be brief")

            // …and on a family without a thinking mode it is refused by name.
            await invoke(["create", "gemma", "--model", "google/gemma-3-4b-it"])
            let refused = await invoke(
                [
                    "attach", "gemma", "french", "--extraction-rendering",
                    #"{"mode":"chatTemplate","reasoningEffort":"low"}"#,
                ])
            #expect(refused.envelope.exitCode == 64)
            #expect(refused.envelope.error?.reason.contains("no thinking switch") == true)
        }
    }

    /// The grand-mean attach path is a DIFFERENT constructor, and it carries
    /// the declaration too — a study whose recipe is mean(concept) − mean(corpus)
    /// needs the rendering pinned exactly as a paired one does.
    @Test func theGrandMeanAttachPathCarriesTheDeclaration() async throws {
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
            let attached = await invoke(
                [
                    "attach", "grand", "fear", "--method", "emotionGrandMean",
                    "--corpus", "joy",
                    "--extraction-rendering", #"{"mode":"chatTemplate"}"#,
                ])
            #expect(attached.exitCode == 0)

            let manifest = try ExperimentStore.load(name: "grand")
            #expect(!manifest.concepts.isEmpty)
            for ref in manifest.concepts {
                #expect(ref.options.extractionRendering?.mode == .chatTemplate,
                        "grand-mean concept \(ref.name) lost the declaration")
            }
            // The identity a grand-mean extraction must satisfy carries it
            // too — the population and the rendering are both recipe.
            let required = try RecipeIdentity.required(
                manifest: manifest, ref: #require(manifest.concepts.first))
            #expect(required.extractionRendering?.mode == .chatTemplate)
        }
    }

    /// `duplicate` is the ONLY way to iterate a frozen study, so a declaration
    /// it dropped would silently re-derive every vector under a different
    /// rendering than the study it descends from.
    @Test func duplicateCarriesTheDeclarationIntoTheNewDraft() async throws {
        try await withTempRoot { _ in
            await invoke(["create", "source", "--model", Self.model])
            #expect(
                await invoke(
                    [
                        "attach", "source", "french", "--extraction-rendering",
                        #"{"mode":"chatTemplate","systemPrompt":"be brief"}"#,
                    ]
                ).exitCode == 0)
            #expect(await invoke(["duplicate", "source", "source-v2"]).exitCode == 0)

            let copy = try ExperimentStore.load(name: "source-v2")
            let ref = try #require(copy.concepts.first)
            let carried = try #require(ref.options.extractionRendering)
            #expect(carried.mode == .chatTemplate)
            #expect(carried.systemPrompt == "be brief")
            // And the two drafts demand the SAME recipe identity, which is
            // what makes the iteration an iteration rather than a new study.
            let source = try ExperimentStore.load(name: "source")
            #expect(
                try RecipeIdentity.hash(
                    RecipeIdentity.required(
                        manifest: source, ref: #require(source.concepts.first)))
                    == RecipeIdentity.hash(
                        RecipeIdentity.required(manifest: copy, ref: ref)))
        }
    }

    // MARK: - 2. nothing else moves

    /// THE HARD CONSTRAINT, at the WRITER. An attach that says nothing and an
    /// attach that says `{"mode": "raw"}` produce the same manifest, byte for
    /// byte — so declaring the legacy rendering out loud cannot move a recipe
    /// hash, a validation scope, or a freeze hash for any study that predates
    /// the option.
    @Test func anExplicitRawAttachIsByteIdenticalToDeclaringNothing() async throws {
        try await withTempRoot { _ in
            await invoke(["create", "silent", "--model", Self.model])
            #expect(await invoke(["attach", "silent", "french"]).exitCode == 0)

            await invoke(["create", "loud", "--model", Self.model])
            #expect(
                await invoke(
                    [
                        "attach", "loud", "french",
                        "--extraction-rendering", #"{"mode":"raw"}"#,
                    ]
                ).exitCode == 0)

            let silent = try ExperimentStore.load(name: "silent")
            let loud = try ExperimentStore.load(name: "loud")
            #expect(silent.concepts.first?.options.extractionRendering == nil)
            #expect(loud.concepts.first?.options.extractionRendering == nil)
            #expect(silent.concepts == loud.concepts)
            #expect(
                ExperimentStore.validationScopeHash(silent)
                    == ExperimentStore.validationScopeHash(loud))

            // …and the bytes on disk, with only the two fields that are
            // legitimately per-study normalized away.
            func normalized(_ name: String) throws -> String {
                var text = String(decoding: try manifestBytes(name), as: UTF8.self)
                text = text.replacingOccurrences(of: "\"\(name)\"", with: "\"NAME\"")
                return text.split(separator: "\n")
                    .filter { !$0.contains("\"createdAt\"") }
                    .joined(separator: "\n")
            }
            let silentBytes = try normalized("silent")
            let loudBytes = try normalized("loud")
            #expect(silentBytes == loudBytes)
            #expect(!loudBytes.contains("extractionRendering"))
        }
    }

    /// The bare word is the same declaration as the object — including for
    /// raw, whose bare form must also canonicalize to absent.
    @Test func theBareModeWordDeclaresExactlyWhatTheObjectDoes() async throws {
        try await withTempRoot { _ in
            func declaration() throws -> ExtractionRendering? {
                try ExperimentStore.load(name: "bare")
                    .concepts.first?.options.extractionRendering
            }
            await invoke(["create", "bare", "--model", Self.model])
            #expect(await invoke(["attach", "bare", "french",
                                  "--extraction-rendering", "raw"]).exitCode == 0)
            let afterRaw = try declaration()
            #expect(afterRaw == nil)

            #expect(await invoke(["attach", "bare", "french",
                                  "--extraction-rendering", "chatTemplate"]).exitCode == 0)
            let afterTemplate = try declaration()
            #expect(afterTemplate?.mode == .chatTemplate)

            // Re-attaching raw CLEARS a previous declaration rather than
            // leaving a stale one behind: attach rebuilds the pin.
            #expect(await invoke(["attach", "bare", "french",
                                  "--extraction-rendering", "raw"]).exitCode == 0)
            let cleared = try declaration()
            #expect(cleared == nil)
        }
    }

    // MARK: - 3. the refusals are typed, and land before anything is written

    /// A malformed declaration is a MALFORMED INVOCATION — exit 64 in both
    /// vocabularies, with a repair — and it lands before the concept is
    /// pinned. Retrying cannot help; the study is not what needs repairing.
    @Test func aMalformedDeclarationIsExitSixtyFourAndWritesNothing() async throws {
        try await withTempRoot { _ in
            await invoke(["create", "bad", "--model", Self.model])
            let cases = [
                #"{"mode":"chatTemplate""#,        // unterminated JSON
                #"{"mode":"templated"}"#,          // out-of-vocabulary mode
                #"{"mode":"raw","systemPrompt":"x"}"#,  // raw takes no parameters
                #"{"mode":"chatTemplate","addGenerationPrompt":1}"#,
                // …and a stranger under chatTemplate, which used to be read as
                // "nothing declared" and silently kept the default.
                #"{"mode":"chatTemplate","addGenerationPromt":false}"#,
            ]
            for declaration in cases {
                let outcome = await invoke(
                    ["attach", "bad", "french", "--extraction-rendering", declaration])
                // The house rule for a malformed VALUE: human exit stays 1,
                // the envelope is `blocked`/64/`usage` — "retype this", not
                // "retry, the system is unwell".
                #expect(outcome.exitCode == 1, "\(declaration) was not refused")
                #expect(outcome.envelope.exitCode == 64,
                        "\(declaration) did not land as a malformed invocation")
                #expect(outcome.envelope.state == .blocked)
                #expect(outcome.envelope.error?.code == "usage")
                #expect(outcome.envelope.error?.repairAction.isEmpty == false,
                        "\(declaration) refused with no repair")
                // NOTHING was pinned: the parse happens before the manifest
                // is even loaded.
                let untouched = try ExperimentStore.load(name: "bad")
                #expect(untouched.concepts.isEmpty)
            }
        }
    }

    /// The engine asymmetry, refused where it is TYPED. `addGenerationPrompt:
    /// false` is a form this engine cannot render, and the honest moment to
    /// say so is now — not after the manifest is frozen and an extraction job
    /// is queued behind it.
    @Test func addGenerationPromptFalseIsRefusedAtAttachNotAtExtraction() async throws {
        try await withTempRoot { _ in
            await invoke(["create", "asym", "--model", Self.model])
            let outcome = await invoke(
                [
                    "attach", "asym", "french", "--extraction-rendering",
                    #"{"mode":"chatTemplate","addGenerationPrompt":false}"#,
                ])
            #expect(outcome.exitCode == 1)
            #expect(outcome.envelope.exitCode == 64)
            #expect(outcome.envelope.state == .blocked)
            // The SAME sentence the extraction path throws — one explanation,
            // two places it can be met.
            #expect(outcome.failure?.reason
                == PromptRendering.addGenerationPromptFalseReason)
            #expect(outcome.envelope.error?.repairAction
                .contains("python-hf-transformers") == true)
            let untouched = try ExperimentStore.load(name: "asym")
            #expect(untouched.concepts.isEmpty)
        }
    }

    /// THE MISSPELLING BUG (review 2026-08-26), answered where the author can
    /// still fix it. `addGenerationPromt: false` — one transposed letter — used
    /// to attach cleanly and pin the DEFAULT `true`: a manifest that reads as
    /// one recipe and extracts as another. The refusal names the offending key
    /// and offers the vocabulary that would have worked.
    @Test func aMisspelledParameterIsRefusedNamingItAndTheVocabulary() async throws {
        try await withTempRoot { _ in
            await invoke(["create", "typo", "--model", Self.model])
            let outcome = await invoke(
                [
                    "attach", "typo", "french", "--extraction-rendering",
                    #"{"mode":"chatTemplate","addGenerationPromt":false}"#,
                ])
            #expect(outcome.exitCode == 1)
            #expect(outcome.envelope.exitCode == 64)
            #expect(outcome.envelope.state == .blocked)
            let reason = try #require(outcome.failure?.reason)
            #expect(reason.contains("addGenerationPromt"))
            let repair = try #require(outcome.envelope.error?.repairAction)
            for key in ExtractionRendering.chatTemplateKeys {
                #expect(repair.contains(key), "the repair omits '\(key)'")
            }
            let untouched = try ExperimentStore.load(name: "typo")
            #expect(untouched.concepts.isEmpty)
        }
    }

    // MARK: - 4. reading a RECORDED rendering is as strict as declaring one

    /// An artifact on disk whose sidecar is written as RAW JSON, so the test
    /// can put a key in it that no type on this engine can produce — which is
    /// exactly the shape a NEWER engine's stamp would have. The tensor is a
    /// stub: a pin hashes those bytes, it never decodes them.
    func plantArtifact(
        root: URL, name: String, rendering: [String: Any]
    ) throws -> (reference: String, sidecar: URL, tensor: URL) {
        let sidecar = SteeringVectorSidecar(
            modelID: Self.model, concept: "french",
            stimulusSetHash: "stim-hash",
            vectors: ConceptVectors(perLayer: [[1, 0], [0, 1]]),
            residualNormPerLayer: [7.0, 7.5],
            residualNormSource: "neutral-corpus")
        var object = try #require(
            try JSONSerialization.jsonObject(
                with: try JSONEncoder().encode(sidecar)) as? [String: Any])
        object["extractionRendering"] = rendering
        let run = "runs/20260826T000000000-planted"
        let directory = root.appending(path: run)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let sidecarURL = directory.appending(component: "\(name).json")
        let tensorURL = directory.appending(component: "\(name).safetensors")
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
            .write(to: sidecarURL)
        try Data("stub-tensor".utf8).write(to: tensorURL)
        return ("\(run)/\(name)", sidecarURL, tensorURL)
    }

    /// A stranger in a RECORDED stamp can only be a newer engine's parameter
    /// or a typo, and copying the block minus that key would pin a rendering
    /// the artifact was not extracted under. Attach refuses by name — never a
    /// decoder dump, and never a quiet reinterpretation.
    @Test func aStrangerInTheArtifactsRenderingRefusesTheAttachByName() async throws {
        try await withTempRoot { root in
            await invoke(["create", "pinned", "--model", Self.model])
            let planted = try plantArtifact(
                root: root, name: "planted-french",
                rendering: ["mode": "chatTemplate", "addGenerationPromt": false])
            do {
                _ = try ExperimentStore.attachArtifact(
                    "planted-french", artifact: planted.reference,
                    experimentName: "pinned")
                Issue.record("a stranger in a recorded rendering must refuse")
            } catch let error as ExperimentError {
                #expect(error.reason.contains(
                    "records an extractionRendering this engine cannot read"))
                #expect(error.reason.contains("addGenerationPromt"))
                #expect(error.reason.contains(
                    "re-attach on the engine that wrote it"))
            }
        }
    }

    /// The same strictness at the OTHER read: a study that ALREADY pins such
    /// an artifact reports a named violation rather than silently skipping
    /// every pin check for that concept. Server twin: the same violation in
    /// `Manifest._verify_vector_artifact_pins`.
    @Test func aStrangerInAPinnedSidecarIsANamedVerifyViolation() async throws {
        try await withTempRoot { root in
            await invoke(["create", "pinned-verify", "--model", Self.model])
            let planted = try plantArtifact(
                root: root, name: "planted-verify",
                rendering: ["mode": "chatTemplate", "addGenerationPromt": false])
            var manifest = try ExperimentStore.load(name: "pinned-verify")
            manifest.concepts = [
                ExperimentManifest.ConceptRef(
                    name: "planted-verify", stimulusSetHash: "stim-hash",
                    options: ExtractionOptions(method: .pinnedArtifact),
                    vectorArtifact: .init(
                        path: planted.reference,
                        sha256TensorHash: ExperimentStore.sha256Hex(
                            try Data(contentsOf: planted.tensor)),
                        sha256SidecarHash: ExperimentStore.sha256Hex(
                            try Data(contentsOf: planted.sidecar)),
                        sourceMethod: "meanDifference",
                        sourceConcept: "planted-verify",
                        residualNormSource: "neutral-corpus")),
            ]
            let violations = ExperimentStore.verify(manifest)
            #expect(violations.contains {
                $0.contains(
                    "declares an extractionRendering this engine cannot read")
            }, "\(violations)")
            #expect(violations.contains { $0.contains("addGenerationPromt") },
                    "\(violations)")
        }
    }

    /// The flag is DECLARED on the verb, so the strict parser accepts it and
    /// `--help` and the generated reference both carry it. (A value flag the
    /// table did not declare would exit 64 as an unknown flag, which is how
    /// this stayed unwritable for as long as it did.)
    @Test func theFlagIsPartOfTheDeclaredAttachSurface() throws {
        let spec = try #require(
            ExperimentCLIParser.specs.first { $0.label == "experiment attach" })
        #expect(spec.valueFlags.contains(ExtractionRendering.declarationFlag))
        #expect(spec.declaredFlags.contains(ExtractionRendering.declarationFlag))
        // Documented, not merely accepted: `--help` renders a metavar and a
        // purpose for every value flag it prints.
        let help = ExperimentCLIHelp.text(for: spec)
        #expect(help.contains(ExtractionRendering.declarationFlag))
        #expect(help.contains("chatTemplate"))
    }
}
