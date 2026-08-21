import Foundation
import Testing

@testable import ExperimentKit

/// The rendered controller script's provenance (open-issues §1 field report,
/// 2026-08-20).
///
/// The defect these tests pin: `~/.steerlab/controller-job.sbatch` is a
/// RENDERED CHILD of `Server/scripts/controller-job.sbatch.template`, the
/// walltime self-chain is shell code in the TEMPLATE, and a deploy refreshes
/// the template without touching the child. serverd 47564632 therefore ran the
/// chain fix's code for 24 h under an Aug-17 launching script and left no
/// successor, and nothing in the system could see the gap.
///
/// Two invariants carry the fix: the artifact STATES which template bytes it
/// came from, and every reader recomputes that digest from the deployed
/// template rather than trusting a claim.
struct ControllerRenderStampTests {

    static func templateText() throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // ExperimentKitTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // repo root
            .appendingPathComponent("Server/scripts/controller-job.sbatch.template")
        return try String(contentsOf: url, encoding: .utf8)
    }

    private func daemonSite() -> ClusterSiteProfile {
        ClusterSiteProfile(
            name: "Test Cluster",
            transport: .ssh(
                host: "login.test", proxyJump: nil, remotePort: 8080,
                vpnExpected: false),
            topology: .daemonInJob,
            scheduler: .slurm(
                ClusterSiteProfile.SlurmSiteData(
                    partitions: [.init(name: "batch", maxWalltimeHours: 168)],
                    defaultPartition: "batch")),
            constraints: ClusterSiteProfile.SiteConstraints(
                computeEgress: .unknown,
                storageRoots: [
                    "metadata": "/home/me/.steerlab",
                    "workspace": "/scratch/me/ws",
                ]))
    }

    // MARK: The template's half of the contract

    @Test func templateCarriesTheStampLineAndTheChainExportBesideTheTrap() throws {
        let template = try Self.templateText()
        let stampLine = template.split(separator: "\n", omittingEmptySubsequences: false)
            .first { $0.hasPrefix(ClusterProvisioner.renderStampMarker) }
        let line = try #require(
            stampLine.map(String.init),
            "the template must carry the machine-readable stamp line")
        for field in ["sha256=\(ClusterProvisioner.renderStampPlaceholders.sha)",
                      "renderedAt=\(ClusterProvisioner.renderStampPlaceholders.renderedAt)",
                      "source=\(ClusterProvisioner.renderStampPlaceholders.source)"]
        {
            #expect(line.contains(field), "stamp line is missing \(field)")
        }
        // The export lives BESIDE the trap on purpose: it is a claim about the
        // file that carries the chain, so the two must not be separable.
        let export = "export \(ClusterProvisioner.controllerChainEnvironmentVariable)="
            + "\"template-\(ClusterProvisioner.renderStampPlaceholders.sha)\""
        #expect(template.contains(export))
        let exportIndex = try #require(template.range(of: export)?.lowerBound)
        let trapIndex = try #require(
            template.range(of: "\ntrap chain_successor USR1\n")?.lowerBound)
        #expect(exportIndex < trapIndex, "the chain marker must be exported before the trap")
    }

    @Test func theChainIsTemplateSideAndUnchangedApartFromTheStamp() throws {
        // The ledger's open question, asserted rather than remembered: the
        // successor chain is shell code in this file. If any of these move to
        // the server, this test is the place that must be reconsidered.
        let template = try Self.templateText()
        #expect(template.contains("chain_successor() {"))
        #expect(template.contains("\ntrap chain_successor USR1\n"))
        #expect(template.contains("sbatch --export=NONE \"$STEERLAB_CONTROLLER_SCRIPT\""))
        #expect(template.contains("serverd.chain.json"))
        // Byte-compatible aside from the stamp/export: still no `exec`, still
        // a background child the trap can outlive.
        #expect(!template.contains("exec \"@PYTHON@\""))
        #expect(template.contains("\"@PYTHON@\" -m steerlab_server.cli serve --port \"$PORT\" &"))
    }

    // MARK: The render command

    @Test func theRenderCommandComputesAndSubstitutesEveryStampField() {
        let command = ClusterProvisioner.controllerRemoteCommand(
            site: daemonSite(), remoteRepoPath: "~/steerlab", envPrefix: nil,
            envFile: "~/steerlab-cluster.env")
        #expect(command.contains("STEERLAB_TPL_SHA=\"$( { sha256sum "))
        #expect(command.contains("|| shasum -a 256 "))
        #expect(command.contains("STEERLAB_RENDERED_AT=\"$(date -u +%Y-%m-%dT%H:%M:%SZ"))
        #expect(command.contains("~/steerlab/Server/steerlab_server/BUILD_COMMIT"))
        for (placeholder, variable) in [
            (ClusterProvisioner.renderStampPlaceholders.sha, "$STEERLAB_TPL_SHA"),
            (ClusterProvisioner.renderStampPlaceholders.renderedAt, "$STEERLAB_RENDERED_AT"),
            (ClusterProvisioner.renderStampPlaceholders.source, "$STEERLAB_SOURCE_REV"),
        ] {
            #expect(command.contains("-e \"s|\(placeholder)|\(variable)|g\""))
        }
    }

    @Test func everyStampSubstitutionSurvivesAMissingFact() {
        // The `&&` chain is the danger: in POSIX sh `VAR="$(cat missing)"`
        // returns the substitution's status, so one absent BUILD_COMMIT would
        // silently SKIP the render — the failure mode this whole change
        // exists to make visible. Each substitution therefore ends in a
        // command that always succeeds.
        let assignments = ClusterProvisioner.controllerStampAssignments(
            templatePath: "/opt/t.template", remoteRepoPath: "/opt/repo")
        #expect(assignments.contains("|| echo unknown; } | awk '{print $1}' )"))
        #expect(assignments.contains("|| echo unknown)"))
        #expect(assignments.contains("|| echo unknown; } | head -n 1 )"))
        // …and an EMPTY value (a zero-byte BUILD_COMMIT) still reads as a
        // statement rather than a blank.
        #expect(assignments.contains("STEERLAB_SOURCE_REV=\"${STEERLAB_SOURCE_REV:-unknown}\""))
        #expect(assignments.hasSuffix("&& "))
    }

    @Test func renderOnlyStopsBeforeSbatchAndEchoesTheStamp() {
        let site = daemonSite()
        let submitting = ClusterProvisioner.controllerRemoteCommand(
            site: site, remoteRepoPath: "~/steerlab", envPrefix: nil,
            envFile: "~/steerlab-cluster.env")
        let renderOnly = ClusterProvisioner.controllerRemoteCommand(
            site: site, remoteRepoPath: "~/steerlab", envPrefix: nil,
            envFile: "~/steerlab-cluster.env", submit: false)
        // "sbatch" also appears in the FILE NAME, so the assertion has to be
        // about the invocation, not the substring.
        #expect(submitting.contains("&& cd \"$WS\" && sbatch "))
        #expect(!renderOnly.contains("&& sbatch "))
        #expect(!renderOnly.contains("&& cd \"$WS\" && sbatch "))
        // Same render, byte for byte, up to the redirect: two renderers would
        // be exactly the drift this item is about.
        let prefix = "> /home/me/.steerlab/controller-job.sbatch"
        let submitPrefix = String(submitting.prefix(
            through: submitting.range(of: prefix)!.upperBound))
        let renderPrefix = String(renderOnly.prefix(
            through: renderOnly.range(of: prefix)!.upperBound))
        #expect(submitPrefix == renderPrefix)
        #expect(renderOnly.contains(
            "sed -n 's|^\(ClusterProvisioner.renderStampMarker) ||p'"))
    }

    // MARK: The staleness probe

    private func probeLines(
        exists: String = "yes", stamp: String, chain: String, template: String
    ) -> [String] {
        ["rendered-exists: \(exists)", "rendered-stamp: \(stamp)",
         "rendered-chain: \(chain)", "template-sha256: \(template)"]
    }

    private let digestA = String(repeating: "a", count: 64)
    private let digestB = String(repeating: "b", count: 64)

    @Test func aMatchingStampIsCurrent() {
        let observation = ClusterProvisioner.parseControllerScriptProbe(
            exitCode: 0,
            lines: probeLines(
                stamp: "template=controller-job.sbatch.template sha256=\(digestA) "
                    + "renderedAt=2026-08-20T09:00:00Z source=52a7176",
                chain: "template-\(digestA)", template: digestA))
        #expect(observation == .current(
            templateHash: digestA, renderedAt: "2026-08-20T09:00:00Z"))
        #expect(observation.advisory(siteID: "site-1") == nil)
    }

    @Test func aDifferentTemplateDigestIsStaleAndNamesBothSides() {
        let observation = ClusterProvisioner.parseControllerScriptProbe(
            exitCode: 0,
            lines: probeLines(
                stamp: "sha256=\(digestA) renderedAt=2026-08-17T04:00:00Z",
                chain: "template-\(digestA)", template: digestB))
        guard case .stale(let reason) = observation else {
            Issue.record("expected .stale, got \(observation)")
            return
        }
        #expect(reason.contains("aaaaaaaaaaaa"))
        #expect(reason.contains("bbbbbbbbbbbb"))
        #expect(reason.contains("2026-08-17T04:00:00Z"))
        let advisory = try? #require(observation.advisory(siteID: "test-site"))
        #expect(advisory?.contains(
            "steerlab-cli cluster controller start --site test-site --render-only") == true)
    }

    @Test func anUnstampedScriptIsStaleAsTheEraItComesFrom() {
        // The 47564632 case exactly: a rendered copy from before the stamp
        // existed. It must not read as "unknown" — its era is precisely the
        // one whose rendered copies drop the chain.
        for stamp in ["", "sha256=\(ClusterProvisioner.renderStampPlaceholders.sha)",
                      "sha256=unknown"]
        {
            let observation = ClusterProvisioner.parseControllerScriptProbe(
                exitCode: 0,
                lines: probeLines(stamp: stamp, chain: "", template: digestA))
            guard case .stale(let reason) = observation else {
                Issue.record("expected .stale for stamp \(stamp.isEmpty ? "<empty>" : stamp)")
                return
            }
            #expect(reason.contains("NO render stamp"))
        }
    }

    @Test func aStampedScriptWithNoChainExportIsStale() {
        let observation = ClusterProvisioner.parseControllerScriptProbe(
            exitCode: 0,
            lines: probeLines(stamp: "sha256=\(digestA)", chain: "", template: digestA))
        guard case .stale(let reason) = observation else {
            Issue.record("expected .stale, got \(observation)")
            return
        }
        #expect(reason.contains("STEERLAB_CONTROLLER_CHAIN"))
    }

    @Test func aMissingArtifactIsAbsentNotStale() {
        let observation = ClusterProvisioner.parseControllerScriptProbe(
            exitCode: 0,
            lines: probeLines(exists: "no", stamp: "", chain: "", template: digestA))
        #expect(observation == .absent)
        // Absent is not a complaint — `controller start` writes one — but it
        // still tells the reader how to render without submitting.
        #expect(observation.advisory(siteID: "site-1")?
            .contains("--render-only") == true)
    }

    @Test func anUnreadableProbeOrTemplateIsUnknownNeverStale() {
        // Doctrine: an unproven fact never licenses an action. A wrong "stale"
        // sends an operator re-rendering over a working script, and the third
        // time that happens they stop reading the finding.
        let failed = ClusterProvisioner.parseControllerScriptProbe(
            exitCode: 255, lines: [])
        guard case .unknown = failed else {
            Issue.record("expected .unknown for a failed probe, got \(failed)")
            return
        }
        let unhashable = ClusterProvisioner.parseControllerScriptProbe(
            exitCode: 0,
            lines: probeLines(stamp: "sha256=\(digestA)",
                              chain: "template-\(digestA)", template: "unknown"))
        guard case .unknown(let reason) = unhashable else {
            Issue.record("expected .unknown, got \(unhashable)")
            return
        }
        #expect(reason.contains("aaaaaaaaaaaa"))
    }

    @Test func theProbeCommandReadsTheArtifactAndHashesTheDeployedTemplate() {
        let command = ClusterProvisioner.controllerScriptProbeCommand(
            site: daemonSite(), remoteRepoPath: "~/steerlab")
        #expect(command.contains("rendered-exists: "))
        #expect(command.contains("/home/me/.steerlab/controller-job.sbatch"))
        #expect(command.contains("~/steerlab/Server/scripts/controller-job.sbatch.template"))
        // Read-only: nothing here writes, redirects, or submits ("sbatch"
        // as a bare substring is in the artifact's own file name).
        #expect(!command.contains("&& sbatch "))
        #expect(!command.contains(" > "))
    }

    // MARK: The layer and its advisory

    @Test func theObservedStateCarriesTheLayerAndTheAdvisory() {
        var observed = ClusterObservedState(siteID: "test-site", siteName: "Test Site")
        observed.controllerScript = .stale(reason: "rendered from template aaa")
        #expect(observed.layerSummaries.contains { $0.layer == "controllerScript" })
        #expect(observed.advisories.count == 1)
        #expect(observed.advisories[0].contains("--render-only"))
        // Silence when there is nothing to say — an advisory channel that
        // always speaks is one nobody reads.
        observed.controllerScript = .current(templateHash: "abc", renderedAt: "now")
        #expect(observed.advisories.isEmpty)
        observed.controllerScript = .notApplicable
        #expect(observed.advisories.isEmpty)
    }

    @Test func theRerenderCommandIsOneStringEverySurfaceNames() {
        // Byte-identical to Server/steerlab_server/controller_render.py's
        // RERENDER_COMMAND (its own test asserts this literal), so the ledger,
        // the qualify repair, the boot warning, and the runbook cannot drift
        // into four near-identical commands.
        #expect(
            ClusterProvisioner.renderControllerCommand(siteID: "")
                == "steerlab-cli cluster controller start --site <site> --render-only")
        #expect(
            ClusterProvisioner.renderControllerCommand(siteID: "test-site")
                == "steerlab-cli cluster controller start --site test-site --render-only")
    }

    @Test func renderOnlyIsAFlagOnTheVerbThatAlreadyOwnedTheRender() throws {
        let invocation = try ClusterCLIParser.parse(
            ["controller", "start", "--site", "test-site", "--render-only"])
        #expect(invocation.verb == .controllerStart)
        #expect(invocation.renderOnly)
        // …and only there: a flag silently accepted by a verb that ignores it
        // is worse than no flag.
        #expect(throws: ClusterCLIError.self) {
            _ = try ClusterCLIParser.parse(["status", "--site", "test-site", "--render-only"])
        }
    }
}
