import Foundation
import Testing

@testable import ExperimentKit

/// WS6.3 unified-Run rules: picker defaults, stochastic pinning, connect-
/// first greying, and the WS4 preflight presentation mapping (ok silent /
/// warn inline / fail blocks behind a restating confirmation). These are
/// the helpers the Studies view renders verbatim.
struct SubstrateRoutingTests {

    private func inputs(
        temperature: Double? = 0,
        samplesPerItem: Int? = nil,
        serverScope: Bool = false,
        siteRegistered: Bool = true,
        siteName: String? = "gpu-a",
        connected: Bool = false,
        userSelection: SubstrateRouting.Substrate? = nil
    ) -> SubstrateRouting.Inputs {
        SubstrateRouting.Inputs(
            temperature: temperature,
            samplesPerItem: samplesPerItem,
            activeWorkspaceIsServer: serverScope,
            siteRegistered: siteRegistered,
            siteName: siteName,
            serverConnected: connected,
            userSelection: userSelection)
    }

    // MARK: Defaults follow the active scope

    @Test func defaultsToTheActiveScope() {
        let local = SubstrateRouting.decide(inputs())
        #expect(local.selection == .thisMac)
        #expect(!local.pinnedToServer)
        #expect(local.stochasticNote == nil)
        #expect(local.runBlockedReason == nil)

        let server = SubstrateRouting.decide(inputs(serverScope: true, connected: true))
        #expect(server.selection == .server)
        #expect(server.serverSelectable)
        #expect(server.runBlockedReason == nil)
        #expect(server.serverLabel == "gpu-a")
    }

    @Test func userOverrideWinsForDeterministicDesigns() {
        let decision = SubstrateRouting.decide(
            inputs(serverScope: true, connected: true, userSelection: .thisMac))
        #expect(decision.selection == .thisMac)
        // Substrate is a scope: local execution needs the Local scope, and
        // the decision says so instead of running into the wrong tree.
        #expect(decision.runBlockedReason != nil)
        #expect(decision.localHint?.contains("Local (MLX)") == true)
    }

    // MARK: Stochastic designs retain the selected backend

    @Test func stochasticTemperaturePreservesTheLocalChoice() {
        let decision = SubstrateRouting.decide(
            inputs(temperature: 0.7, serverScope: false, connected: true,
                   userSelection: .thisMac))
        #expect(decision.selection == .thisMac)
        #expect(!decision.pinnedToServer)
        #expect(decision.stochasticNote?.contains("seeded") == true)
        #expect(decision.runBlockedReason == nil)  // connected → runnable
    }

    @Test func multiSamplePreservesTheLocalDefault() {
        let decision = SubstrateRouting.decide(
            inputs(temperature: 0, samplesPerItem: 8, connected: true))
        #expect(decision.selection == .thisMac)
        #expect(!decision.pinnedToServer)
        // Greedy single-sample stays free.
        let plain = SubstrateRouting.decide(inputs(temperature: 0, samplesPerItem: 1))
        #expect(!plain.pinnedToServer)
    }

    // MARK: Connect-first greying

    @Test func disconnectedSiteGreysTheServerArmWithAHint() {
        let decision = SubstrateRouting.decide(inputs(connected: false, userSelection: .server))
        #expect(!decision.serverSelectable)
        #expect(decision.serverHint?.contains("connect first") == true)
        #expect(decision.runBlockedReason?.contains("gpu-a") == true)
    }

    @Test func noRegisteredSitePointsAtAddingOne() {
        let decision = SubstrateRouting.decide(
            inputs(siteRegistered: false, siteName: nil, userSelection: .server))
        #expect(decision.serverLabel == "Server")
        #expect(!decision.serverSelectable)
        #expect(decision.serverHint?.contains("add a cluster site") == true)
        #expect(decision.runBlockedReason != nil)
    }

    @Test func stochasticLocalRunDoesNotRequireAServer() {
        // Local seeded generation needs no connected server.
        let decision = SubstrateRouting.decide(inputs(temperature: 1.0, connected: false))
        #expect(decision.selection == .thisMac)
        #expect(!decision.pinnedToServer)
        #expect(decision.stochasticNote != nil)
        #expect(decision.runBlockedReason == nil)
    }

    // MARK: Button label says what it does

    @Test func runButtonLabelNamesSubstrateVerbAndDryRun() {
        let local = SubstrateRouting.decide(inputs())
        #expect(SubstrateRouting.runButtonLabel(
            decision: local, verb: "run", dryRun: false, isBusy: false) == "Run Study")

        let server = SubstrateRouting.decide(inputs(serverScope: true, connected: true))
        #expect(SubstrateRouting.runButtonLabel(
            decision: server, verb: "run", dryRun: false, isBusy: false)
            == "Run on gpu-a: run")
        #expect(SubstrateRouting.runButtonLabel(
            decision: server, verb: "sweep", dryRun: true, isBusy: false)
            == "Run on gpu-a: sweep (dry run)")
        #expect(SubstrateRouting.runButtonLabel(
            decision: server, verb: "run", dryRun: false, isBusy: true) == "Running…")
    }

    // MARK: Preflight presentation (ok silent / warn inline / fail blocks)

    private func report(verdict: String, checks: [(String, String, String)]) -> PreflightReport {
        PreflightReport(
            checks: checks.map { PreflightCheck(id: $0.0, status: $0.1, message: $0.2) },
            verdict: verdict)
    }

    @Test func okVerdictProceedsSilently() throws {
        let ok = try #require(
            PreflightPresentation.from(
                report(verdict: "ok", checks: [("memoryFit", "ok", "fits in 80 GB")])))
        #expect(!ok.blocksSubmission)
        #expect(ok.inlineSummary == nil)     // silent
        #expect(ok.attentionLines.isEmpty)   // nothing rendered
        #expect(PreflightPresentation.from(nil) == nil)  // older servers: nothing
    }

    @Test func warnVerdictShowsTheNonOKChecksAndProceeds() throws {
        let presentation = try #require(
            PreflightPresentation.from(
                report(
                    verdict: "warn",
                    checks: [
                        ("memoryFit", "ok", "fits"),
                        ("walltime", "warn", "estimate 9h exceeds the 8h requested"),
                    ])))
        #expect(!presentation.blocksSubmission)
        #expect(presentation.attentionLines.count == 1)
        #expect(presentation.attentionLines.first?.message.contains("9h") == true)
        #expect(presentation.inlineSummary?.contains("warning") == true)
    }

    @Test func failVerdictBlocksAndTheOverrideRestatesTheFailingChecks() throws {
        let presentation = try #require(
            PreflightPresentation.from(
                report(
                    verdict: "fail",
                    checks: [
                        ("memoryFit", "fail",
                         "needs ≥ 39 GB: use gpu:A100:1, not gpu:L4:1"),
                        ("quota", "warn", "scratch at 82%"),
                    ])))
        #expect(presentation.blocksSubmission)
        #expect(presentation.failingLines.count == 1)
        // The forced-override confirmation restates the failing checks
        // verbatim — the override is informed, never silent.
        #expect(presentation.overrideConfirmationMessage.contains("gpu:A100:1"))
        #expect(presentation.overrideConfirmationMessage.contains("bypasses"))
        #expect(presentation.inlineSummary?.contains("refused") == true)
    }

    @Test func refusalMapsOnlyGenuinePreflightFailures() {
        let refusalBody = """
            {"detail": {"message": "refused", "preflight": {"verdict": "fail",
             "checks": [{"id": "quota", "status": "fail",
                         "message": "predicted output exceeds /scratch headroom"}]}}}
            """
        let refusal = PreflightPresentation.refusal(fromErrorBody: refusalBody)
        #expect(refusal?.blocksSubmission == true)
        #expect(refusal?.failingLines.first?.message.contains("/scratch") == true)

        // A warn-verdict report inside an error body is NOT a preflight
        // refusal (something else failed); nor is a plain error.
        let warnBody = """
            {"preflight": {"verdict": "warn", "checks": []}, "detail": "x"}
            """
        #expect(PreflightPresentation.refusal(fromErrorBody: warnBody) == nil)
        #expect(PreflightPresentation.refusal(fromErrorBody: #"{"detail":"boom"}"#) == nil)
    }

    @Test func refusalParsesTheServersActualStringDetail() throws {
        // The landed server refuses with a plain-string 400 detail
        // (`submissions._gate_on_preflight`): em-dash, "; "-joined failing
        // checks, trailing force/dry-run hint. This is the shape the app
        // meets in the wild — pinned here.
        let body = """
            {"detail": "preflight failed — memoryFit: needs ≥ 39 GB: use \
            gpu:A100:1, not gpu:L4:1; quotaHeadroom: predicted output exceeds \
            /scratch headroom (pass force=true to submit anyway, or \
            dryRun=true to inspect the report without submitting)"}
            """
        let refusal = try #require(PreflightPresentation.refusal(fromErrorBody: body))
        #expect(refusal.blocksSubmission)
        #expect(refusal.failingLines.count == 2)
        #expect(refusal.failingLines.first?.id == "memoryFit")
        #expect(refusal.failingLines.first?.message.contains("gpu:A100:1") == true)
        #expect(refusal.failingLines.last?.id == "quotaHeadroom")
        // The hint is stripped — the app renders its own override affordance.
        #expect(!refusal.overrideConfirmationMessage.contains("pass force=true"))
        #expect(refusal.overrideConfirmationMessage.contains("gpu:A100:1"))

        // Bare (non-JSON) text bodies parse too; unrelated errors never do.
        #expect(PreflightPresentation.refusal(
            fromErrorBody: "preflight failed — walltime: estimate 30h exceeds gpu_p cap")?
            .failingLines.first?.id == "walltime")
        #expect(PreflightPresentation.refusal(
            fromErrorBody: "bundle hash mismatch for prompts/dev.jsonl") == nil)
    }

    // MARK: Study-owned sampling — the old-server saved-agent guard (2026-07-21)

    private func capabilities(_ json: String) throws -> ClusterCapabilities {
        try JSONDecoder().decode(ClusterCapabilities.self, from: Data(json.utf8))
    }

    @Test func stochasticAgentSubmissionRefusesOnAServerWithoutTheCapability()
        throws
    {
        // A server that predates study-owned sampling would run each saved
        // agent one greedy path while the baseline samples N draws — the
        // unbalanced design must refuse BEFORE packaging, in plain language.
        let old = try capabilities(
            #"{"serverVersion": "1.0", "remoteStudy": {"submitBundle": true}}"#)
        let refusal = SubstrateRouting.stochasticVariantSubmissionRefusal(
            temperature: 0.7, samplesPerItem: 30, variantConditionCount: 2,
            verb: "run", capabilities: old)
        #expect(refusal?.contains("predates study-owned sampling") == true)
        #expect(refusal?.contains("greedy while the baseline samples") == true)

        // samplesPerItem > 1 alone is stochastic too, and unknown
        // capabilities (never fetched) must refuse rather than assume.
        #expect(
            SubstrateRouting.stochasticVariantSubmissionRefusal(
                temperature: 0, samplesPerItem: 30, variantConditionCount: 1,
                verb: "run", capabilities: old) != nil)
        #expect(
            SubstrateRouting.stochasticVariantSubmissionRefusal(
                temperature: 0.7, samplesPerItem: 1, variantConditionCount: 1,
                verb: "run", capabilities: nil) != nil)
        // "pipeline" executes the study matrix as its last stage — gated.
        #expect(
            SubstrateRouting.stochasticVariantSubmissionRefusal(
                temperature: 0.7, samplesPerItem: 30, variantConditionCount: 1,
                verb: "pipeline", capabilities: old) != nil)
    }

    @Test func stochasticAgentSubmissionProceedsWhenTheServerHasTheCapability()
        throws
    {
        let new = try capabilities(
            #"{"remoteStudy": {"variantStudySampling": true}}"#)
        #expect(
            SubstrateRouting.stochasticVariantSubmissionRefusal(
                temperature: 0.7, samplesPerItem: 30, variantConditionCount: 2,
                verb: "run", capabilities: new) == nil)
        // The dotted features spelling is honored (key-move safety, same
        // rule as supportsWorkspaceSwitch).
        let dotted = try capabilities(
            #"{"features": {"remoteStudy.variantStudySampling": true}}"#)
        #expect(
            SubstrateRouting.stochasticVariantSubmissionRefusal(
                temperature: 0.7, samplesPerItem: 30, variantConditionCount: 1,
                verb: "run", capabilities: dotted) == nil)
    }

    @Test func deterministicAgentlessOrNonMatrixSubmissionsAreNeverGated()
        throws
    {
        let old = try capabilities(#"{"serverVersion": "1.0"}"#)
        // Greedy design with agents: balanced (everyone greedy) — fine.
        #expect(
            SubstrateRouting.stochasticVariantSubmissionRefusal(
                temperature: 0, samplesPerItem: 1, variantConditionCount: 3,
                verb: "run", capabilities: old) == nil)
        // Stochastic but agent-free: no variant path involved — fine.
        #expect(
            SubstrateRouting.stochasticVariantSubmissionRefusal(
                temperature: 0.7, samplesPerItem: 30, variantConditionCount: 0,
                verb: "run", capabilities: old) == nil)
        // Non-matrix verbs never execute conditions — fine. analyze is the
        // statistics-only verb the submit surfaces learned on 2026-08-06.
        for verb in ["verify", "extract", "validate", "sweep", "evaluate",
                     "analyze"] {
            #expect(
                SubstrateRouting.stochasticVariantSubmissionRefusal(
                    temperature: 0.7, samplesPerItem: 30,
                    variantConditionCount: 2, verb: verb,
                    capabilities: old) == nil)
        }
    }
}
