import Foundation
import Testing

@testable import ExperimentKit

/// Pure-CPU tests for the alpha unit-mode contract:
///
/// 1. Norm units are the DEFAULT for a fresh session (decision of record —
///    the instrument's native denomination), with raw as the explicitly
///    labeled fallback.
/// 2. Stored artifacts are NEVER reinterpreted: their explicit
///    `alphaInNormUnits` flag rules, and an ABSENT flag decodes as raw on
///    both engines (mirrored by the Python server's
///    `ModelVariant.from_dict` — the conservative literal α·v reading for
///    legacy artifacts that predate norm capture).
/// 3. Seeding a variant that flips the session's unit mode announces the
///    flip (`ChatService.alphaUnitFlipAnnouncement`), never silently
///    rescaling what the user's typed numbers mean.
@MainActor
struct ChatServiceAlphaUnitsTests {

    private func freshDefaults(_ name: String) throws -> UserDefaults {
        let suite = "steerlab.tests.alpha-units.\(name)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }

    private func makeArtifact(
        name: String = "french-demo", normUnits: Bool
    ) -> ModelVariantArtifact {
        ModelVariantArtifact(
            name: name,
            baseModelID: "Qwen/Qwen3-4B",
            alphaInNormUnits: normUnits,
            promptMode: "chatAssistant",
            qwenThinkingEnabled: false,
            temperature: 0,
            systemPrompt: "")
    }

    // MARK: - Fresh-session default

    @Test func freshSessionDefaultsToNormUnits() throws {
        // The unit mode is not persisted across launches, so the property
        // default IS the fresh-session behavior.
        let service = ChatService(
            cluster: ClusterConnectionStore(defaults: try freshDefaults("fresh")))
        #expect(service.alphaInNormUnits)
    }

    // MARK: - Decode default (the cross-engine absent-key rule)

    @Test func absentFlagDecodesAsRaw() throws {
        // Same rule as the server's `ModelVariant.from_dict`: a spec without
        // alphaInNormUnits is a legacy raw-α artifact. Same bytes must mean
        // the same injection on both engines.
        let json = #"{"name": "legacy", "baseModelID": "org/m"}"#
        let decoded = try JSONDecoder().decode(
            ModelVariantArtifact.self, from: Data(json.utf8))
        #expect(!decoded.alphaInNormUnits)
    }

    @Test func explicitFlagIsHonoredOnDecode() throws {
        for flag in [true, false] {
            let json = """
                {"name": "v", "baseModelID": "org/m", "alphaInNormUnits": \(flag)}
                """
            let decoded = try JSONDecoder().decode(
                ModelVariantArtifact.self, from: Data(json.utf8))
            #expect(decoded.alphaInNormUnits == flag)
        }
    }

    @Test func encoderAlwaysStampsTheFlag() throws {
        // Explicit stamping on every write is what makes the absent-key
        // decode default safe for artifacts this engine produces.
        for flag in [true, false] {
            let data = try JSONEncoder().encode(makeArtifact(normUnits: flag))
            let object = try #require(
                JSONSerialization.jsonObject(with: data) as? [String: Any])
            #expect(object["alphaInNormUnits"] as? Bool == flag)
        }
    }

    // MARK: - The default swap must not reinterpret stored artifacts

    @Test func explicitRawSurvivesSeedCaptureRoundTrip() throws {
        // A raw-α artifact (e.g. the french-demo shakedown variant) decoded,
        // adopted by seeding (seeding applies the ARTIFACT's flag, not the
        // session default), composed back into a control-state capture, and
        // re-encoded: explicit false at every step — the norm-units default
        // never leaks into stored meaning.
        let json = """
            {"name": "raw-mix", "baseModelID": "Qwen/Qwen3-4B",
             "alphaInNormUnits": false,
             "injections": [{"concept": "french",
                             "vectorArtifactID": "/srv/runs/x/french",
                             "layer": 12, "alpha": 6}]}
            """
        let decoded = try JSONDecoder().decode(
            ModelVariantArtifact.self, from: Data(json.utf8))
        #expect(!decoded.alphaInNormUnits)

        // What seedServerControls does: the live mode becomes the artifact's.
        let seededMode = decoded.alphaInNormUnits
        let state = InlineVariantComposer.ControlState(
            baseModelID: decoded.baseModelID,
            slots: decoded.injections.map {
                InlineVariantComposer.Slot(
                    concept: $0.concept, vectorPath: $0.vectorArtifactID,
                    layer: $0.layer, alpha: $0.alpha)
            },
            adapters: [],
            bandWidth: decoded.bandWidth,
            alphaInNormUnits: seededMode,
            neutralPCBasisPath: nil,
            neutralPCBasisLabel: nil,
            promptMode: decoded.promptMode,
            qwenThinkingEnabled: decoded.qwenThinkingEnabled,
            temperature: decoded.temperature,
            systemPrompt: "")
        let captured = try #require(
            InlineVariantComposer.captureArtifact(
                state: state, name: "raw-mix", loadedServerModelID: nil))
        #expect(!captured.alphaInNormUnits)
        #expect(captured.injections.first?.alpha == 6)

        let reDecoded = try JSONDecoder().decode(
            ModelVariantArtifact.self, from: JSONEncoder().encode(captured))
        #expect(!reDecoded.alphaInNormUnits)
    }

    // MARK: - Unit-flip announcements on variant seeding

    @Test func flipToRawIsAnnounced() throws {
        let note = try #require(
            ChatService.alphaUnitFlipAnnouncement(
                currentNormUnits: true,
                applying: makeArtifact(normUnits: false)))
        #expect(note == "variant 'french-demo' uses raw alphas "
            + "(α values are literal α·v coefficients)")
    }

    @Test func flipToNormUnitsIsAnnounced() throws {
        let note = try #require(
            ChatService.alphaUnitFlipAnnouncement(
                currentNormUnits: false,
                applying: makeArtifact(normUnits: true)))
        #expect(note == "variant 'french-demo' uses residual-norm alphas "
            + "(α is a fraction of the layer's residual-stream norm)")
    }

    @Test func noFlipIsSilent() {
        #expect(
            ChatService.alphaUnitFlipAnnouncement(
                currentNormUnits: true, applying: makeArtifact(normUnits: true))
                == nil)
        #expect(
            ChatService.alphaUnitFlipAnnouncement(
                currentNormUnits: false, applying: makeArtifact(normUnits: false))
                == nil)
    }
}
