import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// What the Robustness Check's judge controls OFFER, SPELL, and DISPATCH TO,
/// tested where those are written.
///
/// Three defects live behind this suite:
///
/// - the picker concatenated the raw cache scan, so non-generative artifacts
///   appeared as judges (fixed by a capability filter in the offer
///   composition, never in the scan — the scan is also the is-installed test);
/// - the dispatch knew two backends where the engine has three, so a fully
///   implemented judge client was unreachable and the server route's skip
///   warning swallowed every spelling that was not Claude's;
/// - the Prompts count silently outran the file it read from.
///
/// No test here holds, reads, or renders a credential: key state reaches the
/// composition as a PRESENCE boolean and nothing else.
@Suite(.serialized) struct RobustnessJudgeSurfaceTests {

    // MARK: - Fixtures

    /// A capability seam with an explicit verdict per id. Anything unnamed is
    /// capable, so a test states only what it cares about.
    private static func capability(
        refusing refusals: [String: String]
    ) -> JudgeModelOffers.CapabilityCheck {
        { id in
            refusals[id].map { .refused($0) } ?? .capable
        }
    }

    // MARK: - The spelling

    /// The spelling is NOT new vocabulary: it is the study path's judge spec
    /// (`ExperimentCLIRunner.parseJudges`, `<name>:<kind>[:<model>[:<provider>]]`)
    /// minus the name field — same kind token, same separator, same order,
    /// same rule that the provider field is OpenRouter's alone.
    @Test func theOpenRouterSpellingFollowsTheStudyPathsFieldOrder() throws {
        let parsed = try #require(
            JudgeModelSpelling.parse("openrouter:vendor/model:together"))
        #expect(parsed == .openRouter(model: "vendor/model", provider: "together"))
        #expect(parsed.kind == "openrouter")
        #expect(parsed.model == "vendor/model")
        #expect(parsed.provider == "together")

        // The kind token is the manifest's own.
        #expect(ExperimentStore.knownJudgeKinds.contains(parsed.kind))

        // The CLI parser reads the same three fields off the same string,
        // once a name is put in front of it.
        let viaCLI = try ExperimentCLIRunner.parseJudges(
            "j-1:openrouter:vendor/model:together")
        #expect(viaCLI[0].kind == parsed.kind)
        #expect(viaCLI[0].model == parsed.model)
        #expect(viaCLI[0].provider == parsed.provider)
    }

    @Test func claudeAndLocalSpellingsAreUnchanged() throws {
        #expect(JudgeModelSpelling.parse("claude-opus-4-8") == .claude(model: "claude-opus-4-8"))
        #expect(
            JudgeModelSpelling.parse("anthropic:some-model")
                == .claude(model: "anthropic:some-model"))
        #expect(
            JudgeModelSpelling.parse("vendor/model-small-4bit")
                == .local(model: "vendor/model-small-4bit"))
        #expect(JudgeModelSpelling.parse("   ") == nil)
        #expect(JudgeModelSpelling.parse(nil) == nil)
    }

    /// An unpinned provider is not a pinned judge — the same sentence the
    /// sweep route and the catalogue preflight use. It must NOT fall through
    /// to the local branch and be loaded as a repo id.
    @Test func anOpenRouterSpellingWithNoProviderIsItsOwnCase() {
        #expect(
            JudgeModelSpelling.parse("openrouter:vendor/model")
                == .openRouterUnpinned(model: "vendor/model"))
        #expect(
            JudgeModelSpelling.parse("openrouter:")
                == .openRouterUnpinned(model: ""))
        #expect(
            JudgeModelSpelling.unpinnedProviderRefusal(model: "vendor/model")
                == "openrouter judge for 'vendor/model' has no pinned provider "
                    + "— an unpinned provider is not a pinned judge")
    }

    @Test func spellingRoundTripsThroughTheOneWriter() throws {
        let spelled = JudgeModelSpelling.spellOpenRouter(
            model: "vendor/model", provider: "together")
        #expect(spelled == "openrouter:vendor/model:together")
        #expect(
            JudgeModelSpelling.parse(spelled)
                == .openRouter(model: "vendor/model", provider: "together"))
    }

    // MARK: - Picker composition (defect A)

    @Test func nonGenerativeCacheArtifactsAreFilteredOutOfTheOffers() {
        let offers = JudgeModelOffers.compose(
            selected: "",
            candidates: [
                .cached("vendor/chat-model-small"),
                .cached("vendor/dictionary-artifact"),
                .cached("vendor/lens-artifact"),
            ],
            openRouterKeyPresent: false,
            capability: Self.capability(refusing: [
                "vendor/dictionary-artifact":
                    "the cached snapshot has no config.json",
                "vendor/lens-artifact":
                    "the cached snapshot has no config.json",
            ]))
        #expect(offers.models.map(\.id) == ["vendor/chat-model-small"])
    }

    /// Curated entries have nothing on disk to inspect, so they are never
    /// capability-checked — even when the seam would refuse them.
    @Test func curatedEntriesPassByConstruction() {
        let offers = JudgeModelOffers.compose(
            selected: "",
            candidates: [
                .cached("vendor/dictionary-artifact"),
                .curated("vendor/tier-model"),
                .curated("claude-opus-4-8"),
            ],
            openRouterKeyPresent: false,
            capability: Self.capability(refusing: [
                "vendor/dictionary-artifact": "no config.json",
                "vendor/tier-model": "no cached snapshot",
                "claude-opus-4-8": "no cached snapshot",
            ]))
        #expect(offers.models.map(\.id) == ["vendor/tier-model", "claude-opus-4-8"])
    }

    /// A stored choice that fails the filter is FLAGGED, never dropped: a
    /// picker that quietly deletes a saved selection leaves a blank field and
    /// no account of what happened to it.
    @Test func aFailingStoredSelectionIsFlaggedNotDropped() throws {
        let offers = JudgeModelOffers.compose(
            selected: "vendor/dictionary-artifact",
            candidates: [.cached("vendor/chat-model-small")],
            openRouterKeyPresent: false,
            capability: Self.capability(refusing: [
                "vendor/dictionary-artifact":
                    "the cached snapshot has no config.json — this repo is "
                    + "not a loadable text model"
            ]))
        let flagged = try #require(offers.models.first)
        #expect(flagged.id == "vendor/dictionary-artifact")
        #expect(flagged.label == "vendor/dictionary-artifact (cannot judge)")
        #expect(flagged.isFlagged)
        #expect(
            offers.selectionCaption
                == "'vendor/dictionary-artifact' cannot judge: the cached "
                    + "snapshot has no config.json — this repo is not a "
                    + "loadable text model. It stays listed because it is "
                    + "your stored choice — pick another judge to clear this.")
        // …and the capable option is still offered alongside it.
        #expect(offers.models.map(\.id).contains("vendor/chat-model-small"))
    }

    @Test func aCapableStoredSelectionIsNotFlagged() {
        let offers = JudgeModelOffers.compose(
            selected: "vendor/chat-model-small",
            candidates: [.cached("vendor/chat-model-small")],
            openRouterKeyPresent: false,
            capability: Self.capability(refusing: [:]))
        #expect(offers.models.map(\.id) == ["vendor/chat-model-small"])
        #expect(offers.selectionCaption == nil)
        #expect(offers.models.allSatisfy { !$0.isFlagged })
    }

    // MARK: - Picker composition (defect B: the OpenRouter section)

    @Test func theOpenRouterSectionAppearsOnlyWhenAKeyIsPresent() {
        let withKey = JudgeModelOffers.compose(
            selected: "", candidates: [.curated("claude-opus-4-8")],
            openRouterKeyPresent: true,
            capability: Self.capability(refusing: [:]))
        #expect(withKey.openRouter.map(\.id) == ["openrouter:"])
        #expect(withKey.openRouter.map(\.label) == ["OpenRouter judge…"])
        #expect(withKey.openRouterHint == nil)

        let withoutKey = JudgeModelOffers.compose(
            selected: "", candidates: [.curated("claude-opus-4-8")],
            openRouterKeyPresent: false,
            capability: Self.capability(refusing: [:]))
        #expect(withoutKey.openRouter.isEmpty)
        #expect(
            withoutKey.openRouterHint
                == "No OpenRouter key on this Mac — set the external judge "
                    + "key in Compute (stored in the macOS Keychain), or put "
                    + "OPENROUTER_API_KEY in the environment, and OpenRouter "
                    + "judges appear here.")
    }

    /// Key presence gates the SECTION, never the stored choice: a saved
    /// OpenRouter judge stays listed on a machine that has since lost its key,
    /// alongside the hint that says where to put one back.
    @Test func aStoredOpenRouterSelectionSurvivesAnAbsentKey() {
        let offers = JudgeModelOffers.compose(
            selected: "openrouter:vendor/model:together",
            candidates: [.curated("claude-opus-4-8")],
            openRouterKeyPresent: false,
            capability: Self.capability(refusing: [:]))
        #expect(offers.openRouter.map(\.id) == ["openrouter:vendor/model:together"])
        #expect(offers.openRouterHint != nil)
        #expect(offers.selectionCaption == nil)
    }

    // MARK: - Dispatch (defect B)

    /// Replays one canned OpenRouter response and records what was sent —
    /// the same scripted-transport shape the other OpenRouter suites use.
    private actor ScriptedTransport {
        private(set) var requestBodies: [Data] = []
        private let body: String

        init(body: String) { self.body = body }

        func handle(_ request: URLRequest) throws -> (Data, URLResponse) {
            requestBodies.append(request.httpBody ?? Data())
            let response = HTTPURLResponse(
                url: OpenRouterPairedJudge.endpoint, statusCode: 200,
                httpVersion: nil, headerFields: nil)!
            return (Data(body.utf8), response)
        }
    }

    private static let openRouterVerdict = """
        {"provider": "Together", "choices": [{"message": {"content": \
        "{\\"winner\\": \\"B\\", \\"confidence\\": 0.7, \\"brief_reason\\": \
        \\"B reads better.\\"}"}, "finish_reason": "stop"}]}
        """

    @Test func bothRoutesDispatchAnOpenRouterSelectionToTheOpenRouterClient()
        async throws
    {
        let script = ScriptedTransport(body: Self.openRouterVerdict)
        let previousKey = OpenRouterPairedJudge.keyOverrideForTesting
        OpenRouterPairedJudge.keyOverrideForTesting = "test-only-not-a-key"
        defer { OpenRouterPairedJudge.keyOverrideForTesting = previousKey }

        // localContainer: nil is the SERVER route; the local route differs
        // only in holding one, which an OpenRouter pick never consults.
        let route = VariantRobustness.judgeRoute(
            JudgeModelSpelling.parse("openrouter:vendor/model:together"),
            rubric: "rubric text",
            localContainer: nil,
            openRouterTransport: { try await script.handle($0) })
        let judge = try #require(route.judge)
        #expect(route.warnings.isEmpty)

        let verdict = try #require(
            try await judge("a prompt", "baseline text", "variant text"))
        #expect(verdict.winner.lowercased() == "b")

        // The request went to OpenRouter carrying BOTH pins.
        let bodies = await script.requestBodies
        let firstBody = try #require(bodies.first)
        let sent = try #require(
            try JSONSerialization.jsonObject(with: firstBody) as? [String: Any])
        #expect(sent["model"] as? String == "vendor/model")
        let provider = try #require(sent["provider"] as? [String: Any])
        #expect(provider["order"] as? [String] == ["together"])
        #expect(provider["allow_fallbacks"] as? Bool == false)
    }

    /// The server route's skip warning stays for a GENUINELY local pick — and
    /// no longer swallows the other two kinds.
    @Test func theServerRouteSkipsOnlyLocalJudges() {
        let local = VariantRobustness.judgeRoute(
            JudgeModelSpelling.parse("vendor/chat-model-small"),
            rubric: "r", localContainer: nil)
        #expect(local.judge == nil)
        #expect(
            local.warnings == [
                "coherence judge 'vendor/chat-model-small' is a local model — "
                    + "judging skipped in a server workspace; pick a Claude or "
                    + "OpenRouter judge"
            ])

        let claude = VariantRobustness.judgeRoute(
            JudgeModelSpelling.parse("claude-opus-4-8"),
            rubric: "r", localContainer: nil)
        #expect(claude.judge != nil)
        #expect(claude.warnings.isEmpty)

        let openRouter = VariantRobustness.judgeRoute(
            JudgeModelSpelling.parse("openrouter:vendor/model:together"),
            rubric: "r", localContainer: nil)
        #expect(openRouter.judge != nil)
        #expect(openRouter.warnings.isEmpty)
    }

    @Test func anUnpinnedOpenRouterSelectionJudgesNothingAndSaysWhy() {
        let route = VariantRobustness.judgeRoute(
            JudgeModelSpelling.parse("openrouter:vendor/model"),
            rubric: "r", localContainer: nil)
        #expect(route.judge == nil)
        #expect(
            route.warnings == [
                "openrouter judge for 'vendor/model' has no pinned provider — "
                    + "an unpinned provider is not a pinned judge"
            ])
    }

    @Test func noSelectionIsNoJudgeAndNoWarning() {
        let route = VariantRobustness.judgeRoute(
            nil, rubric: "r", localContainer: nil)
        #expect(route.judge == nil)
        #expect(route.warnings.isEmpty)
    }

    /// The ad-hoc judge selector reads the SAME spelling, so an OpenRouter
    /// pick there resolves to an openrouter judge with its provider intact
    /// rather than being mistaken for a repo id.
    @Test func theAdHocJudgeResolvesTheSameSpelling() throws {
        var manifest = ExperimentManifest(
            name: "surface", description: "", modelID: "vendor/base-model")
        manifest.judges = nil
        let resolved = try ExperimentTasks.resolvedJudges(
            manifest: manifest,
            evaluation: ExperimentManifest.EvaluationSpec(
                kind: .pairedJudge,
                judgeModel: "openrouter:vendor/model:together"))
        #expect(resolved.count == 1)
        #expect(resolved[0].kind == "openrouter")
        #expect(resolved[0].model == "vendor/model")
        #expect(resolved[0].provider == "together")
    }

    // MARK: - The report stamp

    @Test func theJudgeStampNamesTheBackendAndAnyPin() {
        #expect(
            VariantRobustnessReadout.judgeStamp(
                model: "vendor/model", kind: "openrouter", provider: "together")
                == "vendor/model (openrouter · together)")
        #expect(
            VariantRobustnessReadout.judgeStamp(
                model: "claude-opus-4-8", kind: "claude", provider: nil)
                == "claude-opus-4-8 (claude)")
        // A report from before the stamp names the model alone rather than
        // guessing a backend for it.
        #expect(
            VariantRobustnessReadout.judgeStamp(
                model: "vendor/model", kind: nil, provider: nil)
                == "vendor/model")
    }

    // MARK: - The Prompts row (defect C)

    @Test func theSupplyCaptionNamesTheFileAndItsCount() {
        #expect(
            RobustnessPromptSupply.supplyCaption(
                file: "prompts/dev/dev-prompts.jsonl", supply: 6)
                == "of 6 in dev-prompts.jsonl")
        // An unknown supply says nothing rather than guessing zero.
        #expect(
            RobustnessPromptSupply.supplyCaption(
                file: "prompts/dev/dev-prompts.jsonl", supply: nil) == nil)
    }

    /// The engine still takes a prefix; the row now says so out loud instead
    /// of truncating in silence.
    @Test func anOversizedRequestSurfacesInsteadOfTruncatingSilently() {
        #expect(
            RobustnessPromptSupply.truncationCaption(requested: 9, supply: 6)
                == "file has 6 — extra prompts are ignored")
        #expect(RobustnessPromptSupply.truncationCaption(requested: 6, supply: 6) == nil)
        #expect(RobustnessPromptSupply.truncationCaption(requested: 3, supply: 6) == nil)
        #expect(RobustnessPromptSupply.truncationCaption(requested: 9, supply: nil) == nil)
    }

    @Test func typedCountsClampIntoTheRowsRange() {
        #expect(RobustnessPromptSupply.clamp(0) == 1)
        #expect(RobustnessPromptSupply.clamp(99) == 12)
        #expect(RobustnessPromptSupply.clamp(5) == 5)
        // Mid-edit text keeps the current value rather than zeroing it.
        #expect(RobustnessPromptSupply.clamp(text: "", current: 4) == 4)
        #expect(RobustnessPromptSupply.clamp(text: "abc", current: 4) == 4)
        #expect(RobustnessPromptSupply.clamp(text: " 7 ", current: 4) == 7)
        #expect(RobustnessPromptSupply.clamp(text: "40", current: 4) == 12)
    }

    /// The supply is READ from the file, not assumed from the preset.
    @Test func theSupplyIsCountedFromTheFileItself() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "prompt-supply") { root in
            let previous = WorkspaceRoot.programmaticOverride
            WorkspaceRoot.programmaticOverride = root
            defer { WorkspaceRoot.programmaticOverride = previous }

            let directory = root.appending(path: "prompts/dev")
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
            try """
                {"text":"one"}
                {"text":"two"}
                {"text":"three"}
                """
                .write(
                    to: directory.appending(component: "sample-prompts.jsonl"),
                    atomically: true, encoding: .utf8)

            #expect(
                RobustnessPromptSupply.count(
                    file: "prompts/dev/sample-prompts.jsonl") == 3)
            #expect(RobustnessPromptSupply.count(file: "prompts/dev/absent.jsonl") == nil)
            #expect(RobustnessPromptSupply.count(file: "  ") == nil)
        }
    }
}

/// The two panes that own a single-string judge picker, composing through the
/// SAME rule. These tests exist to prove the seams are actually wired into the
/// panel properties — a composition that is correct but not reached would fix
/// nothing. Key state arrives as a presence boolean; nothing here is, holds,
/// or renders a credential.
@Suite(.serialized) @MainActor
struct JudgePickerPanelSurfaceTests {

    private static let refusesDictionaryArtifacts: JudgeModelOffers.CapabilityCheck = {
        $0 == "vendor/dictionary-artifact"
            ? .refused("the cached snapshot has no config.json")
            : .capable
    }

    @Test func theRobustnessPickerFiltersTheScanAndFlagsAStoredFailure() throws {
        let panel = FineTuningPanel()
        panel.judgeCapabilityOverrideForTesting = Self.refusesDictionaryArtifacts
        panel.localModelScanOverrideForTesting = [
            "vendor/chat-model-small", "vendor/dictionary-artifact",
        ]
        panel.judgeKeyPresenceOverrideForTesting = false
        panel.robustnessJudgeModel = "vendor/dictionary-artifact"

        let offers = panel.robustnessJudgeOffers
        let ids = offers.models.map(\.id)
        // The stored choice survives, flagged and first…
        #expect(ids.first == "vendor/dictionary-artifact")
        #expect(offers.models.first?.isFlagged == true)
        #expect(offers.selectionCaption?.contains("stays listed") == true)
        // …it is not offered a SECOND time from the scan, and the capable
        // scanned model is.
        #expect(ids.filter { $0 == "vendor/dictionary-artifact" }.count == 1)
        #expect(ids.contains("vendor/chat-model-small"))
        // No key: no section, but a hint saying where to put one.
        #expect(offers.openRouter.isEmpty)
        #expect(offers.openRouterHint == JudgeModelOffers.openRouterKeyHint)
    }

    @Test func theRobustnessPickerOffersOpenRouterOnlyWithAKey() {
        let panel = FineTuningPanel()
        panel.judgeCapabilityOverrideForTesting = { _ in .capable }
        panel.localModelScanOverrideForTesting = []
        panel.judgeKeyPresenceOverrideForTesting = true
        #expect(panel.robustnessJudgeOffers.openRouter.map(\.id) == ["openrouter:"])
        #expect(panel.robustnessJudgeOffers.openRouterHint == nil)
    }

    /// The two fields the pane reveals write through the ONE spelling writer,
    /// so the pane never hand-assembles a judge string.
    @Test func theOpenRouterFieldsWriteTheCanonicalSpelling() {
        let panel = FineTuningPanel()
        panel.robustnessJudgeModel = JudgeModelOffers.openRouterSentinel
        panel.robustnessOpenRouterModel = "vendor/model"
        #expect(panel.robustnessJudgeModel == "openrouter:vendor/model")
        #expect(panel.robustnessJudgeSelection == .openRouterUnpinned(model: "vendor/model"))

        panel.robustnessOpenRouterProvider = "together"
        #expect(panel.robustnessJudgeModel == "openrouter:vendor/model:together")
        #expect(
            panel.robustnessJudgeSelection
                == .openRouter(model: "vendor/model", provider: "together"))
        #expect(panel.robustnessOpenRouterModel == "vendor/model")
        #expect(panel.robustnessOpenRouterProvider == "together")
    }

    /// The Studies-side twin composes by the same rule — one fix, two panes.
    @Test func theAdHocPickerFiltersTheScanAndGatesOpenRouterTheSameWay() {
        let panel = ExperimentPanel()
        panel.judgeCapabilityOverrideForTesting = Self.refusesDictionaryArtifacts
        panel.localModelScanOverrideForTesting = [
            "vendor/chat-model-small", "vendor/dictionary-artifact",
        ]
        panel.judgeKeyPresenceOverrideForTesting = false
        panel.judgeModel = ""

        let offers = panel.judgeModelOffers
        #expect(!offers.models.map(\.id).contains("vendor/dictionary-artifact"))
        #expect(offers.models.map(\.id).contains("vendor/chat-model-small"))
        #expect(offers.openRouter.isEmpty)
        #expect(offers.openRouterHint == JudgeModelOffers.openRouterKeyHint)

        // The flat list the web surface consumes is the same offers,
        // flattened — the filter reaches it too.
        #expect(!panel.judgeModelOptions.contains("vendor/dictionary-artifact"))

        panel.judgeKeyPresenceOverrideForTesting = true
        #expect(panel.judgeModelOffers.openRouter.map(\.id) == ["openrouter:"])
        #expect(panel.judgeModelOptions.contains("openrouter:"))
    }
}
