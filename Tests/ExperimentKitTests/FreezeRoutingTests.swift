import Foundation
import Testing

@testable import ExperimentKit

/// The Freeze control's substrate routing + the advisory-surfacing rule —
/// pure decision logic (the view renders exactly what `decide`/`present`
/// return). The substrate-consistent lifecycle: in a server workspace the
/// freeze routes to the SERVER, whose gates run against server-substrate
/// evidence, because validation evidence counts only on the substrate that
/// performs the freeze.
struct FreezeRoutingTests {

    // MARK: Routing

    @Test func localWorkspaceFreezesLocally() {
        let decision = FreezeRouting.decide(
            FreezeRouting.Inputs(activeWorkspaceIsServer: false))
        #expect(decision.target == .thisMac)
        #expect(decision.buttonLabel == "Freeze Study…")
        #expect(decision.confirmLabel == "Freeze")
        #expect(decision.executorNote == nil)
        #expect(decision.blockedReason == nil)
    }

    @Test func serverWorkspaceRoutesToTheServerAndSaysSo() {
        let decision = FreezeRouting.decide(
            FreezeRouting.Inputs(
                activeWorkspaceIsServer: true,
                serverConnected: true,
                serverLabel: "example-hpc",
                serverHasSelectedStudy: true))
        #expect(decision.target == .server)
        // The executing substrate must be unmistakable on the button itself.
        #expect(decision.buttonLabel == "Freeze (on example-hpc)…")
        #expect(decision.confirmLabel == "Freeze on example-hpc")
        #expect(decision.executorNote?.contains("SERVER-substrate") == true)
        #expect(
            decision.executorNote?.contains(
                "validation evidence counts on the substrate that freezes") == true)
        #expect(decision.blockedReason == nil)
    }

    // MARK: Mac-authority routing (2026-07-21)

    @Test func knownUnpairedServerWorkspaceFreezesLocally() {
        // The researcher's real cluster configuration: server Compute whose
        // workspace is KNOWN unpaired — the Mac is the source of truth, so
        // Freeze runs in this workspace with the full local gate set, and
        // the button says so.
        let decision = FreezeRouting.decide(
            FreezeRouting.Inputs(
                activeWorkspaceIsServer: true,
                serverConnected: true,
                serverLabel: "example-hpc",
                serverHasSelectedStudy: false,
                workspaceKnownUnpaired: true))
        #expect(decision.target == .thisMac)
        #expect(decision.buttonLabel == "Freeze (in this workspace)…")
        #expect(decision.confirmLabel == "Freeze")
        // Non-residency must NOT block: the frozen study travels as a
        // bundle; the note names the run-substrate evidence rule.
        #expect(decision.blockedReason == nil)
        #expect(decision.executorNote?.contains("Mac authority") == true)
        #expect(decision.executorNote?.contains("example-hpc") == true)
    }

    @Test func knownUnpairedNeedsNoServerConnectionToFreeze() {
        // Local freeze needs nothing from the server — a disconnected
        // unpaired site must not gate the lifecycle action.
        let decision = FreezeRouting.decide(
            FreezeRouting.Inputs(
                activeWorkspaceIsServer: true,
                serverConnected: false,
                serverLabel: "example-hpc",
                workspaceKnownUnpaired: true))
        #expect(decision.target == .thisMac)
        #expect(decision.blockedReason == nil)
    }

    @Test func unknownPairingKeepsTheServerRoute() {
        // Mac-authority requires a CONFIRMED unpaired answer; unknown
        // pairing keeps the historical server route (the identity precheck
        // is the backstop there).
        let decision = FreezeRouting.decide(
            FreezeRouting.Inputs(
                activeWorkspaceIsServer: true,
                serverConnected: true,
                serverLabel: "example-hpc",
                workspaceKnownUnpaired: false))
        #expect(decision.target == .server)
        #expect(decision.buttonLabel == "Freeze (on example-hpc)…")
    }

    @Test func localViolationsStillDisableAMacAuthorityFreeze() {
        // Unpaired-local routing uses the LOCAL gates — a local pin
        // violation disables the button exactly like a plain local freeze.
        let decision = FreezeRouting.decide(
            FreezeRouting.Inputs(
                activeWorkspaceIsServer: true,
                serverConnected: true,
                workspaceKnownUnpaired: true))
        #expect(
            FreezeRouting.freezeButtonDisabled(
                decision: decision, hasLocalViolations: true))
        #expect(
            !FreezeRouting.freezeButtonDisabled(
                decision: decision, hasLocalViolations: false))
    }

    @Test func disconnectedServerBlocksWithConnectHint() {
        let decision = FreezeRouting.decide(
            FreezeRouting.Inputs(
                activeWorkspaceIsServer: true,
                serverConnected: false,
                serverLabel: "example-hpc"))
        #expect(decision.target == .server)
        #expect(decision.blockedReason?.contains("connect to example-hpc") == true)
    }

    @Test func nonResidentStudyBlocksAndNamesBothWaysForward() {
        let decision = FreezeRouting.decide(
            FreezeRouting.Inputs(
                activeWorkspaceIsServer: true,
                serverConnected: true,
                serverLabel: "example-hpc",
                serverHasSelectedStudy: false))
        #expect(decision.blockedReason?.contains("server-resident copy only") == true)
        #expect(decision.blockedReason?.contains("serve --root") == true)
        #expect(decision.blockedReason?.contains("Local (MLX)") == true)
    }

    @Test func unknownResidencyNeverBlocks() {
        // Listing failed / not yet checked: the server's own refusal is the
        // backstop — a transient error must not disable the lifecycle.
        let decision = FreezeRouting.decide(
            FreezeRouting.Inputs(
                activeWorkspaceIsServer: true,
                serverConnected: true,
                serverLabel: "example-hpc",
                serverHasSelectedStudy: nil))
        #expect(decision.blockedReason == nil)
    }

    @Test func emptyServerLabelFallsBackToGenericName() {
        let decision = FreezeRouting.decide(
            FreezeRouting.Inputs(
                activeWorkspaceIsServer: true,
                serverConnected: true,
                serverLabel: "  "))
        #expect(decision.buttonLabel == "Freeze (on server)…")
    }

    // MARK: Button enablement (server gates decide remote readiness)

    @Test func localViolationsDisableOnlyALocalFreeze() {
        let local = FreezeRouting.decide(
            FreezeRouting.Inputs(activeWorkspaceIsServer: false))
        #expect(FreezeRouting.freezeButtonDisabled(
            decision: local, hasLocalViolations: true))
        #expect(!FreezeRouting.freezeButtonDisabled(
            decision: local, hasLocalViolations: false))
    }

    @Test func serverRoutedFreezeIgnoresLocalViolations() {
        // The finding: local verification failures disabled the server
        // freeze button even when the remote copy is valid. Server-side
        // gates decide remote readiness — never local violations.
        let server = FreezeRouting.decide(
            FreezeRouting.Inputs(
                activeWorkspaceIsServer: true,
                serverConnected: true,
                serverLabel: "example-hpc",
                serverHasSelectedStudy: true))
        #expect(!FreezeRouting.freezeButtonDisabled(
            decision: server, hasLocalViolations: true))
    }

    @Test func blockedReasonAlwaysDisables() {
        let disconnected = FreezeRouting.decide(
            FreezeRouting.Inputs(
                activeWorkspaceIsServer: true,
                serverConnected: false,
                serverLabel: "example-hpc"))
        #expect(FreezeRouting.freezeButtonDisabled(
            decision: disconnected, hasLocalViolations: false))
    }

    @Test func localViolationsContextNoteIsInformationalAndCounted() {
        let note = FreezeRouting.localViolationsContextNote(
            count: 2, serverLabel: "example-hpc")
        #expect(note?.contains("2 issues") == true)
        #expect(note?.contains("informational") == true)
        #expect(note?.contains("example-hpc") == true)
        #expect(FreezeRouting.localViolationsContextNote(
            count: 0, serverLabel: "example-hpc") == nil)
        // Singular form.
        #expect(FreezeRouting.localViolationsContextNote(
            count: 1, serverLabel: "example-hpc")?.contains("1 issue ") == true)
    }

    // MARK: Remote-freeze manifest identity (block/proceed rules)

    @Test func verifiedEqualProceedsSilently() {
        let precheck = FreezeRouting.remoteFreezePrecheck(
            identity: .verifiedEqual, study: "s", serverLabel: "example-hpc",
            workspacePaired: false)
        #expect(precheck.proceed)
        #expect(precheck.message == nil)
    }

    @Test func mismatchBlocksWithFieldSummaryAndRemedy() {
        let precheck = FreezeRouting.remoteFreezePrecheck(
            identity: .mismatch([
                "conditions: differs (local 3, server 5)",
                "modelID: differs",
            ]),
            study: "case1", serverLabel: "example-hpc", workspacePaired: false)
        #expect(!precheck.proceed)
        #expect(precheck.message?.contains("NOT the manifest you are looking at") == true)
        #expect(precheck.message?.contains("conditions: differs (local 3, server 5)") == true)
        #expect(precheck.message?.contains("modelID: differs") == true)
        // The remedy, verbatim intent: pair or sync.
        #expect(precheck.message?.contains("pair the workspace") == true
            || precheck.message?.contains("Pair the workspace") == true)
    }

    @Test func mismatchBlocksEvenOnAPairedWorkspace() {
        // A measured difference always outranks pairing heuristics.
        let precheck = FreezeRouting.remoteFreezePrecheck(
            identity: .mismatch(["temperature: differs"]),
            study: "case1", serverLabel: "example-hpc", workspacePaired: true)
        #expect(!precheck.proceed)
    }

    @Test func serverOnlyCopyProceedsButNamesWhatWillFreeze() {
        let precheck = FreezeRouting.remoteFreezePrecheck(
            identity: .localMissing(
                serverStatus: "draft",
                canonicalBodyHash: String(repeating: "a", count: 64)),
            study: "case1", serverLabel: "example-hpc", workspacePaired: false)
        #expect(precheck.proceed)
        #expect(precheck.message?.contains("ONLY copy") == true)
        #expect(precheck.message?.contains("case1") == true)
        #expect(precheck.message?.contains("status draft") == true)
        // Hash-of-canonicalized-body, truncated for display.
        #expect(precheck.message?.contains("aaaaaaaaaaaa…") == true)
    }

    @Test func unverifiableBlocksOnUnpairedProceedsOnPaired() {
        let unpaired = FreezeRouting.remoteFreezePrecheck(
            identity: .unverifiable("older server"),
            study: "case1", serverLabel: "example-hpc", workspacePaired: false)
        #expect(!unpaired.proceed)
        #expect(unpaired.message?.contains("unpaired workspace") == true)
        #expect(unpaired.message?.contains("serve --root") == true)

        let paired = FreezeRouting.remoteFreezePrecheck(
            identity: .unverifiable("older server"),
            study: "case1", serverLabel: "example-hpc", workspacePaired: true)
        #expect(paired.proceed)
        #expect(paired.message?.contains("could not verify") == true)
    }

    // MARK: One-click server-draft sync (2026-07-21 incident, part 3)

    @Test func syncOfferedExactlyForAMismatchedLocalDraft() {
        // The mismatch block used to name a remedy ("sync the study") the
        // app didn't offer — now it does, for local DRAFTS only.
        #expect(
            FreezeRouting.canOfferServerDraftSync(
                identity: .mismatch(["temperature: differs"]), localIsDraft: true))
        // Frozen local manifests never push (duplicate-never-edit).
        #expect(
            !FreezeRouting.canOfferServerDraftSync(
                identity: .mismatch(["temperature: differs"]), localIsDraft: false))
        // The other cases have nothing trustworthy to push over.
        #expect(
            !FreezeRouting.canOfferServerDraftSync(
                identity: .unverifiable("older server"), localIsDraft: true))
        #expect(
            !FreezeRouting.canOfferServerDraftSync(
                identity: .localMissing(serverStatus: "draft", canonicalBodyHash: nil),
                localIsDraft: true))
        #expect(
            !FreezeRouting.canOfferServerDraftSync(
                identity: .verifiedEqual, localIsDraft: true))
    }

    @Test func syncOutcomeReportsVerifiedEqualAfterTheRecheck() {
        let outcome = FreezeRouting.serverDraftSyncOutcome(
            recheck: .verifiedEqual,
            study: "case1", serverLabel: "example-hpc",
            canonicalBodyHash: String(repeating: "b", count: 64))
        #expect(outcome.resolved)
        #expect(outcome.message.contains("verified equal"))
        #expect(outcome.message.contains("case1"))
        #expect(outcome.message.contains("bbbbbbbbbbbb…"))
    }

    @Test func syncOutcomeReportsARemainingDiffHonestly() {
        // Pushed but the re-check still differs (racing writer): the claim
        // is the measurement, never "pushed, so it must match".
        let outcome = FreezeRouting.serverDraftSyncOutcome(
            recheck: .mismatch(["conditions: differs (local 3, server 5)"]),
            study: "case1", serverLabel: "example-hpc", canonicalBodyHash: nil)
        #expect(!outcome.resolved)
        #expect(outcome.message.contains("STILL differs"))
        #expect(outcome.message.contains("conditions: differs (local 3, server 5)"))
        #expect(outcome.message.contains("do not freeze"))
    }

    @Test func syncOutcomeReportsAnUnverifiableRecheck() {
        let outcome = FreezeRouting.serverDraftSyncOutcome(
            recheck: .unverifiable("manifest body fetch failed: timeout"),
            study: "case1", serverLabel: "example-hpc", canonicalBodyHash: nil)
        #expect(!outcome.resolved)
        #expect(outcome.message.contains("could not verify"))
        #expect(outcome.message.contains("timeout"))
    }

    @Test func manifestReplaceResultDecodesPreservedServerPins() throws {
        // Merge semantics (2026-08-06): the server names the auto-pins it
        // KEPT when the pushed document omitted them — the app adopts the
        // revision locally so the identity re-check converges. Keys match
        // experiment_store.replace_draft_manifest exactly.
        let json = #"""
            {"name": "demo", "status": "draft", "canonicalBodyHash": "abc",
             "preserved": {"modelRevision": "005ad3404e59",
                           "conditions": ["fear-recommended"],
                           "capabilityBattery": {
                               "file": "prompts/battery.jsonl",
                               "hash": "bb"}}}
            """#
        let result = try JSONDecoder().decode(
            ClusterClient.RemoteManifestReplaceResult.self,
            from: Data(json.utf8))
        #expect(result.preserved?.modelRevision == "005ad3404e59")
        #expect(result.preserved?.conditions == ["fear-recommended"])
        #expect(result.preserved?.capabilityBattery?.file
            == "prompts/battery.jsonl")
        // Older server / nothing preserved: the key is simply absent.
        let bare = try JSONDecoder().decode(
            ClusterClient.RemoteManifestReplaceResult.self,
            from: Data(#"{"name": "demo"}"#.utf8))
        #expect(bare.preserved == nil)
    }

    // MARK: The advisory-surfacing rule

    @Test func recognizesBothEnginesCrossSubstrateWording() {
        // Swift's wording…
        #expect(
            FreezeRouting.isCrossSubstrateAdvisory(
                "validation evidence was produced on python-hf-transformers; "
                    + "runs on swift-mlx should re-validate on-substrate"))
        // …and the server's (identical clause, engines swapped).
        #expect(
            FreezeRouting.isCrossSubstrateAdvisory(
                "validation evidence was produced on swift-mlx; runs on "
                    + "python-hf-transformers should re-validate on-substrate"))
        // Other advisories stay regular.
        #expect(
            !FreezeRouting.isCrossSubstrateAdvisory(
                "2 variant condition(s) are hand-created (no sweep-selection "
                    + "provenance) — fine for exploration"))
    }

    @Test func presentPromotesCrossSubstrateAdvisoriesWithTheRule() {
        let cross =
            "validation evidence was produced on python-hf-transformers; "
            + "runs on swift-mlx should re-validate on-substrate"
        let regular = "forced freeze — gates skipped: revision — non-citable"
        let presentation = FreezeRouting.present(advisories: [cross, regular])
        #expect(presentation.prominent.count == 1)
        #expect(presentation.prominent.first?.hasPrefix(cross) == true)
        // The one-line explanation rides with the prominent rendering.
        #expect(
            presentation.prominent.first?.contains(
                FreezeRouting.crossSubstrateRule) == true)
        #expect(presentation.regular == [regular])
    }
}
