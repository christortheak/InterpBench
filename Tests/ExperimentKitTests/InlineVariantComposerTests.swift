import Foundation
import Testing

@testable import ExperimentKit

/// Pure-CPU tests for the server-workspace inline-variant machinery: the
/// "compose a variant spec from the live chat controls" assembly and the
/// dirty-since-seeded rule that decides whether a send carries the stored
/// `variantPath` + hash (exact provenance) or an ephemeral inline spec.
///
/// Contract under test (the server's `/api/variant/generate[/stream]` inline
/// form): the inline `variant` object is field-for-field the `variant` field
/// of the `POST /api/variants/upload` payload — baseModelID (required), name,
/// baseRevision, injections, adapters, bandWidth, alphaInNormUnits,
/// neutralPCBasisPath, promptMode, qwenThinkingEnabled, temperature,
/// systemPrompt — with vector refs addressed by SERVER artifact path.
@Suite struct InlineVariantComposerTests {

    private func makeState(
        slots: [InlineVariantComposer.Slot] = [
            InlineVariantComposer.Slot(
                concept: "fear",
                vectorPath: "/srv/runs/2026-07-01T000000Z-concept-fear/fear",
                layer: 18, alpha: 4.5)
        ],
        adapters: [ModelVariantArtifact.AdapterRef] = [],
        systemPrompt: String = "You are terse."
    ) -> InlineVariantComposer.ControlState {
        InlineVariantComposer.ControlState(
            baseModelID: "Qwen/Qwen3-4B",
            slots: slots,
            adapters: adapters,
            bandWidth: 3,
            alphaInNormUnits: true,
            neutralPCBasisPath: nil,
            neutralPCBasisLabel: nil,
            promptMode: "chatAssistant",
            qwenThinkingEnabled: false,
            temperature: 0.7,
            systemPrompt: systemPrompt)
    }

    // MARK: - Composed spec matches the upload-payload schema verbatim

    @Test func composedSpecEncodesTheUploadPayloadSchema() throws {
        let adapter = ModelVariantArtifact.AdapterRef(
            name: "judicial-lora",
            artifactPath: "/srv/runs/2026-07-02T000000Z-lora-judicial",
            adapterDirectory: "/srv/runs/2026-07-02T000000Z-lora-judicial")
        let spec = InlineVariantComposer.compose(makeState(adapters: [adapter]))

        let data = try JSONEncoder().encode(spec)
        let object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])

        // Only contract keys appear (the upload payload also stamps
        // schemaVersion/systemPromptHash/createdAt — same encoder, so the
        // inline spec matches the stored-artifact schema by construction).
        let contractKeys: Set<String> = [
            "schemaVersion", "name", "baseModelID", "baseRevision",
            "adapters", "injections", "bandWidth", "alphaInNormUnits",
            "neutralPCBasisPath", "neutralPCBasisLabel", "promptMode",
            "qwenThinkingEnabled", "temperature", "systemPromptHash",
            "systemPrompt", "createdAt",
        ]
        #expect(Set(object.keys).isSubset(of: contractKeys), "keys: \(object.keys)")

        #expect(object["baseModelID"] as? String == "Qwen/Qwen3-4B")
        #expect(object["name"] as? String == "chat-inline")
        #expect(object["bandWidth"] as? Int == 3)
        #expect(object["alphaInNormUnits"] as? Bool == true)
        #expect(object["promptMode"] as? String == "chatAssistant")
        #expect(object["qwenThinkingEnabled"] as? Bool == false)
        #expect(object["temperature"] as? Double == 0.7)
        #expect(object["systemPrompt"] as? String == "You are terse.")
        // Unset optionals are ABSENT from the wire, not JSON null.
        #expect(object["neutralPCBasisPath"] == nil)
        #expect(object["baseRevision"] == nil)

        let injections = try #require(object["injections"] as? [[String: Any]])
        #expect(injections.count == 1)
        // Vector refs are SERVER artifact paths — never local ids.
        #expect(
            injections[0]["vectorArtifactID"] as? String
                == "/srv/runs/2026-07-01T000000Z-concept-fear/fear")
        #expect(injections[0]["concept"] as? String == "fear")
        #expect(injections[0]["layer"] as? Int == 18)
        #expect(injections[0]["alpha"] as? Double == 4.5)

        let adapters = try #require(object["adapters"] as? [[String: Any]])
        #expect(
            adapters.first?["adapterDirectory"] as? String
                == "/srv/runs/2026-07-02T000000Z-lora-judicial")
    }

    @Test func emptySystemPromptEncodesAbsent() throws {
        // Same normalization a saved local variant gets: trimmed-empty system
        // prompt drops both systemPrompt and systemPromptHash from the wire.
        let spec = InlineVariantComposer.compose(makeState(systemPrompt: "  \n"))
        let data = try JSONEncoder().encode(spec)
        let object = try #require(
            JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["systemPrompt"] == nil)
        #expect(object["systemPromptHash"] == nil)
    }

    @Test func seededNeutralBasisPinSurvivesComposition() throws {
        // The basis is not editable in chat; an edited spec must not silently
        // drop a seeded variant's pinned neutral basis.
        var state = makeState()
        state.neutralPCBasisPath = "/srv/runs/2026-06-30T000000Z-neutral/basis.safetensors"
        state.neutralPCBasisLabel = "neutral-v1"
        let spec = InlineVariantComposer.compose(state)
        #expect(spec.neutralPCBasisPath == state.neutralPCBasisPath)
        #expect(spec.neutralPCBasisLabel == "neutral-v1")
    }

    @Test func inlineSpecRoundTripsThroughTheDetailDecoder() throws {
        // The composed spec must decode through the same tolerant decoder the
        // variant-detail seed path uses, byte-for-byte equal in every
        // contract field (the seed → edit → compose round trip).
        let spec = InlineVariantComposer.compose(makeState())
        let decoded = try JSONDecoder().decode(
            ModelVariantArtifact.self, from: JSONEncoder().encode(spec))
        #expect(decoded == spec)
    }

    // MARK: - Dirty-since-seeded (stored path vs inline spec vs plain)

    @Test func untouchedSeedSendsTheStoredVariant() {
        let seeded = makeState()
        #expect(
            InlineVariantComposer.disposition(
                storedVariantSelected: true, seeded: seeded, current: seeded)
                == .storedVariant)
    }

    @Test func anyEditSwitchesToInline() {
        let seeded = makeState()

        var alphaTouched = seeded
        alphaTouched.slots[0].alpha = 5.0
        #expect(
            InlineVariantComposer.disposition(
                storedVariantSelected: true, seeded: seeded, current: alphaTouched)
                == .inlineVariant)

        var promptTouched = seeded
        promptTouched.systemPrompt += "!"
        #expect(
            InlineVariantComposer.disposition(
                storedVariantSelected: true, seeded: seeded, current: promptTouched)
                == .inlineVariant)

        var bandTouched = seeded
        bandTouched.bandWidth = 5
        #expect(
            InlineVariantComposer.disposition(
                storedVariantSelected: true, seeded: seeded, current: bandTouched)
                == .inlineVariant)

        // Toggling steering off empties the effective slot list — that IS an
        // edit, so the send stops claiming the stored variant's identity.
        var steeringOff = seeded
        steeringOff.slots = []
        #expect(
            InlineVariantComposer.disposition(
                storedVariantSelected: true, seeded: seeded, current: steeringOff)
                == .inlineVariant)
    }

    @Test func unseedableStoredSelectionKeepsExactProvenance() {
        // Older server without /api/variant/detail: the controls could not be
        // seeded, dirty-tracking is impossible, so the stored path is sent
        // regardless of what the (unrelated) controls hold.
        #expect(
            InlineVariantComposer.disposition(
                storedVariantSelected: true, seeded: nil, current: makeState())
                == .storedVariant)
    }

    @Test func noStoredVariantWithControlsComposesInline() {
        #expect(
            InlineVariantComposer.disposition(
                storedVariantSelected: false, seeded: nil, current: makeState())
                == .inlineVariant)
    }

    @Test func noVariantAndNoInterventionsIsPlainGenerate() {
        // "None" + steering off → plain /api/generate/stream, unchanged.
        let bare = makeState(slots: [], adapters: [])
        #expect(!bare.hasInterventions)
        #expect(
            InlineVariantComposer.disposition(
                storedVariantSelected: false, seeded: nil, current: bare)
                == .plainGenerate)
    }

    // MARK: - Server-workspace Save Variant (captureArtifact)

    @Test func serverWorkspaceCaptureRecordsServerModelAndServerRefs() throws {
        // A server-workspace save is the composed control state verbatim:
        // the SERVER base model id and SERVER vector paths — never the local
        // loadedModelID or local artifact ids.
        let adapter = ModelVariantArtifact.AdapterRef(
            name: "judicial-lora",
            artifactPath: "/srv/runs/2026-07-02T000000Z-lora-judicial",
            adapterDirectory: "/srv/runs/2026-07-02T000000Z-lora-judicial")
        let captured = try #require(
            InlineVariantComposer.captureArtifact(
                state: makeState(adapters: [adapter]),
                name: "fear-mix",
                loadedServerModelID: nil))

        #expect(captured.name == "fear-mix")
        #expect(captured.baseModelID == "Qwen/Qwen3-4B")
        #expect(
            captured.injections.map(\.vectorArtifactID)
                == ["/srv/runs/2026-07-01T000000Z-concept-fear/fear"])
        #expect(captured.injections.first?.concept == "fear")
        #expect(captured.injections.first?.layer == 18)
        #expect(captured.injections.first?.alpha == 4.5)
        #expect(
            captured.adapters.map(\.adapterDirectory)
                == ["/srv/runs/2026-07-02T000000Z-lora-judicial"])
        #expect(captured.bandWidth == 3)
        #expect(captured.alphaInNormUnits)
        #expect(captured.temperature == 0.7)
        #expect(captured.systemPrompt == "You are terse.")
        // The captured definition IS the inline-send spec for the same state.
        var asSent = InlineVariantComposer.compose(makeState(adapters: [adapter]))
        asSent.name = captured.name
        asSent.createdAt = captured.createdAt
        #expect(captured == asSent)
    }

    @Test func captureFallsBackToTheLoadedServerModel() throws {
        // No server model picked yet, but one is loaded on the server: that
        // is the base the chat would run, so it's the base the save records.
        var state = makeState()
        state.baseModelID = ""
        let captured = try #require(
            InlineVariantComposer.captureArtifact(
                state: state, name: "v", loadedServerModelID: "Qwen/Qwen3-0.6B"))
        #expect(captured.baseModelID == "Qwen/Qwen3-0.6B")
    }

    @Test func captureRefusesWithoutAnyServerModelIdentity() {
        // No selected and no loaded server model → no truthful base model to
        // record → nil (the panel reports instead of writing a bogus recipe).
        var state = makeState()
        state.baseModelID = ""
        #expect(
            InlineVariantComposer.captureArtifact(
                state: state, name: "v", loadedServerModelID: nil)
                == nil)
        #expect(
            InlineVariantComposer.captureArtifact(
                state: state, name: "v", loadedServerModelID: "")
                == nil)
    }

    // MARK: - Honest slot resolution (the silent-drop fix)

    private static let catalogConcepts = [
        "/srv/runs/2026-07-01T000000Z-concept-fear/fear": "fear",
        "/srv/runs/2026-07-03T000000Z-concept-independent/independent": "independent",
    ]

    private func input(
        _ vectorID: String?, enabled: Bool = true, layer: Int = 12, alpha: Double = 2
    ) -> InlineVariantComposer.SlotInput {
        InlineVariantComposer.SlotInput(
            vectorID: vectorID, layer: layer, alpha: alpha, enabled: enabled)
    }

    @Test func resolvableEnabledSlotsCompose() {
        let resolution = InlineVariantComposer.resolveSlots(
            steeringEnabled: true,
            slots: [
                input("/srv/runs/2026-07-01T000000Z-concept-fear/fear", layer: 18, alpha: 4.5),
                input(
                    "/srv/runs/2026-07-03T000000Z-concept-independent/independent",
                    layer: 10, alpha: -1),
            ],
            concept: { Self.catalogConcepts[$0] })
        #expect(resolution.unresolvedVectorIDs.isEmpty)
        #expect(
            resolution.slots == [
                InlineVariantComposer.Slot(
                    concept: "fear",
                    vectorPath: "/srv/runs/2026-07-01T000000Z-concept-fear/fear",
                    layer: 18, alpha: 4.5),
                InlineVariantComposer.Slot(
                    concept: "independent",
                    vectorPath: "/srv/runs/2026-07-03T000000Z-concept-independent/independent",
                    layer: 10, alpha: -1),
            ])
    }

    @Test func unresolvableEnabledSlotIsReportedNotSwallowed() {
        // The user-facing bug class: an enabled slot whose id is not in the
        // server catalog (local leftover, stale listing, failed fetch) must
        // be excluded from the spec AND show up in the drop report.
        let localLeftover = "/Users/x/runs/2026-07-01T000000Z-concept-fear/fear"
        let resolution = InlineVariantComposer.resolveSlots(
            steeringEnabled: true,
            slots: [
                input(localLeftover),
                input("/srv/runs/2026-07-01T000000Z-concept-fear/fear"),
            ],
            concept: { Self.catalogConcepts[$0] })
        #expect(resolution.slots.map(\.vectorPath) == [
            "/srv/runs/2026-07-01T000000Z-concept-fear/fear"
        ])
        #expect(resolution.unresolvedVectorIDs == [localLeftover])
    }

    @Test func disabledAndEmptySlotsAreNotDrops() {
        // A disabled box and an enabled-but-unpicked box are intentional
        // states, not silent drops — they must not trigger the warning.
        let resolution = InlineVariantComposer.resolveSlots(
            steeringEnabled: true,
            slots: [
                input("nowhere/unresolvable", enabled: false),
                input(nil, enabled: true),
            ],
            concept: { Self.catalogConcepts[$0] })
        #expect(resolution.slots.isEmpty)
        #expect(resolution.unresolvedVectorIDs.isEmpty)
    }

    @Test func masterSwitchOffMeansNoSlotsAndNoDrops() {
        // Steering off is deliberate: no injections and nothing to warn
        // about, even when boxes hold unresolvable ids.
        let resolution = InlineVariantComposer.resolveSlots(
            steeringEnabled: false,
            slots: [input("nowhere/unresolvable")],
            concept: { _ in nil })
        #expect(resolution.slots.isEmpty)
        #expect(resolution.unresolvedVectorIDs.isEmpty)
    }

    // MARK: - The loudness surfaces (refusal / caption / send warning)

    @Test func refusalTextNamesTheSlotsAndTheRemedy() {
        #expect(
            InlineVariantComposer.unresolvedSlotRefusal(unresolvedVectorIDs: []) == nil)
        #expect(
            InlineVariantComposer.unresolvedSlotRefusal(unresolvedVectorIDs: ["a/b"])
                == "not saved: slot 'a/b' does not resolve in the server catalog "
                + "listing — hit Refresh artifacts (or re-pick the vector), then save again")
        #expect(
            InlineVariantComposer.unresolvedSlotRefusal(unresolvedVectorIDs: ["a/b", "c/d"])
                == "not saved: 2 slots ('a/b', 'c/d') do not resolve in the server "
                + "catalog listing — hit Refresh artifacts (or re-pick the vectors), "
                + "then save again")
    }

    @Test func captionWarnsWhileSlotsWouldDrop() {
        #expect(InlineVariantComposer.unresolvedSlotCaption(count: 0) == nil)
        #expect(
            InlineVariantComposer.unresolvedSlotCaption(count: 1)
                == "1 enabled slot unresolved — NOT sent; Refresh artifacts")
        #expect(
            InlineVariantComposer.unresolvedSlotCaption(count: 2)
                == "2 enabled slots unresolved — NOT sent; Refresh artifacts")
    }

    @Test func sendWarningIsExplicitAboutWhatRan() {
        #expect(
            InlineVariantComposer.unresolvedSendWarning(unresolvedVectorIDs: []) == nil)
        let single = InlineVariantComposer.unresolvedSendWarning(
            unresolvedVectorIDs: ["a/b"])
        #expect(
            single
                == "1 enabled steering slot ('a/b') did not resolve in the server "
                + "catalog listing and was NOT sent — this generation ran without it; "
                + "hit Refresh artifacts or re-pick the vector")
        let double = InlineVariantComposer.unresolvedSendWarning(
            unresolvedVectorIDs: ["a/b", "c/d"])
        #expect(
            double
                == "2 enabled steering slots ('a/b', 'c/d') did not resolve in the "
                + "server catalog listing and were NOT sent — this generation ran "
                + "without them; hit Refresh artifacts or re-pick the vector")
    }

    @Test func captureRefusesRatherThanSavingAnInjectionlessLie() {
        // The observed bug end-to-end in pure logic: a steered chat whose
        // only enabled slot fails to resolve must NOT become an
        // empty-injections definition — the refusal fires first.
        let resolution = InlineVariantComposer.resolveSlots(
            steeringEnabled: true,
            slots: [input("stale/or/local/path")],
            concept: { _ in nil })
        #expect(resolution.slots.isEmpty)
        let refusal = InlineVariantComposer.unresolvedSlotRefusal(
            unresolvedVectorIDs: resolution.unresolvedVectorIDs)
        #expect(refusal != nil)
        // Had the refusal not fired, captureArtifact would happily record
        // zero injections — exactly the provenance hazard being closed.
        var state = makeState(slots: resolution.slots)
        state.baseModelID = "google/gemma-3-4b-it"
        let wouldHaveSaved = InlineVariantComposer.captureArtifact(
            state: state, name: "x", loadedServerModelID: nil)
        #expect(wouldHaveSaved?.injections.isEmpty == true)
    }

    @Test func adapterAloneCountsAsAnIntervention() {
        let adapter = ModelVariantArtifact.AdapterRef(
            name: "lora", artifactPath: "/srv/a", adapterDirectory: "/srv/a")
        let state = makeState(slots: [], adapters: [adapter])
        #expect(state.hasInterventions)
        #expect(
            InlineVariantComposer.disposition(
                storedVariantSelected: false, seeded: nil, current: state)
                == .inlineVariant)
    }
}
