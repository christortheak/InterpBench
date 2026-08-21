import Foundation
import Testing

@testable import ExperimentKit

// MARK: - Scripted runner (no ssh, no processes)

/// Replays canned responses in order and records every argv, so tests can
/// assert the EXACT commands the step machine composes.
private actor ScriptedProvisionRunner: ProvisionCommandRunner {
    struct Response {
        var lines: [String]
        var exit: Int32
        init(lines: [String] = [], exit: Int32 = 0) {
            self.lines = lines
            self.exit = exit
        }
    }

    private var responses: [Response]
    private(set) var calls: [[String]] = []

    init(_ responses: [Response] = []) {
        self.responses = responses
    }

    func enqueue(lines: [String] = [], exit: Int32 = 0) {
        responses.append(Response(lines: lines, exit: exit))
    }

    func recordedCalls() -> [[String]] { calls }
    func callCount() -> Int { calls.count }

    func run(
        _ argv: [String], onLine: @escaping @Sendable (String) -> Void
    ) async throws -> Int32 {
        calls.append(argv)
        guard !responses.isEmpty else { return 0 }
        let response = responses.removeFirst()
        for line in response.lines { onLine(line) }
        return response.exit
    }
}

// MARK: - Tests

/// WS5 step-machine tests over the scripted runner: exact argv templates for
/// push / bootstrap dry-run + real / validate / controller sbatch (all
/// site-driven), the dry-run-before-real gate, bootstrap JSON report parsing
/// (ok and failed steps, stderr tail verbatim), the daemon-in-a-job branch
/// (job-id parse → serverd.host poll), loud skips with idempotent re-runs,
/// and transcript capture.
@MainActor
struct ClusterProvisionerTests {

    // MARK: Render-provenance literals (open-issues §1 field report, 2026-08-20)

    /// The remote-shell assignments the render stamp is built from, for the
    /// site the byte-exact golden below uses (template under `~/steerlab`).
    /// Transcribed by hand on purpose: this is the text an operator would read
    /// in a transcript, and a helper that regenerated it from the same code
    /// under test would assert nothing.
    static let expectedStampAssignments =
        "STEERLAB_TPL_SHA=\"$( { sha256sum "
        + "~/steerlab/Server/scripts/controller-job.sbatch.template 2>/dev/null "
        + "|| shasum -a 256 "
        + "~/steerlab/Server/scripts/controller-job.sbatch.template 2>/dev/null "
        + "|| echo unknown; } | awk '{print $1}' )\" && "
        + "STEERLAB_RENDERED_AT=\"$(date -u +%Y-%m-%dT%H:%M:%SZ || echo unknown)\" && "
        + "STEERLAB_SOURCE_REV=\"$( { cat "
        + "~/steerlab/Server/steerlab_server/BUILD_COMMIT 2>/dev/null "
        + "|| cat ~/steerlab/steerlab_server/BUILD_COMMIT 2>/dev/null "
        + "|| echo unknown; } | head -n 1 )\" && "
        + "STEERLAB_TPL_SHA=\"${STEERLAB_TPL_SHA:-unknown}\" "
        + "STEERLAB_RENDERED_AT=\"${STEERLAB_RENDERED_AT:-unknown}\" "
        + "STEERLAB_SOURCE_REV=\"${STEERLAB_SOURCE_REV:-unknown}\" && "

    /// The three stamp substitutions, DOUBLE-quoted (they expand remote shell
    /// variables) where the two body placeholders stay single-quoted.
    static let expectedStampSedExpressions =
        " -e \"s|@TEMPLATE_SHA256@|$STEERLAB_TPL_SHA|g\""
        + " -e \"s|@RENDERED_AT@|$STEERLAB_RENDERED_AT|g\""
        + " -e \"s|@SOURCE_REVISION@|$STEERLAB_SOURCE_REV|g\""

    private func slurmSite(
        topology: ClusterSiteProfile.Topology = .loginDaemon,
        proxyJump: String? = nil,
        account: String? = nil,
        bootstrapPath: String? = nil,
        storageRoots: [String: String] = [
            "workspace": "/scratch/me/ws", "hfCache": "/work/lab/hf",
        ]
    ) -> ClusterSiteProfile {
        ClusterSiteProfile(
            name: "Test Cluster",
            transport: .ssh(
                host: "login.test", proxyJump: proxyJump, remotePort: 8080, vpnExpected: false),
            topology: topology,
            scheduler: .slurm(
                ClusterSiteProfile.SlurmSiteData(
                    partitions: [
                        .init(name: "gpu_p", maxWalltimeHours: 168),
                        .init(name: "batch", maxWalltimeHours: 168),
                    ],
                    gpuTypes: ["A100"],
                    gpuVRAMGB: ["A100": 80],
                    defaultGres: "gpu:A100:1",
                    defaultPartition: "gpu_p",
                    accountRequired: account != nil,
                    account: account)),
            constraints: ClusterSiteProfile.SiteConstraints(
                computeEgress: .unknown, storageRoots: storageRoots),
            bootstrapPath: bootstrapPath)
    }

    private let controlPrefix = [
        "/usr/bin/ssh", "-o", "ControlPath=~/.ssh/steerlab-cm-%C", "-o", "BatchMode=yes",
    ]

    // MARK: Authenticate

    @Test func authenticateCheckUsesTheSharedControlMasterProbe() async {
        let runner = ScriptedProvisionRunner([.init(exit: 0)])
        let provisioner = ClusterProvisioner(runner: runner, site: slurmSite())
        let alive = await provisioner.runAuthenticateCheck()
        #expect(alive)
        let calls = await runner.recordedCalls()
        #expect(
            calls.first == [
                "/usr/bin/ssh", "-o", "ControlPath=~/.ssh/steerlab-cm-%C",
                "-O", "check", "login.test",
            ])
        guard case .succeeded = provisioner.record(for: .authenticate).status else {
            Issue.record("expected .succeeded, got \(provisioner.record(for: .authenticate).status)")
            return
        }
    }

    @Test func authenticateCheckStaysOpenWhileTheMasterIsAbsent() async {
        let runner = ScriptedProvisionRunner([.init(exit: 255)])
        let provisioner = ClusterProvisioner(runner: runner, site: slurmSite())
        let alive = await provisioner.runAuthenticateCheck()
        #expect(!alive)
        #expect(provisioner.record(for: .authenticate).status == .running)
        #expect(
            provisioner.record(for: .authenticate).transcript.contains {
                $0.contains("authenticate in Terminal")
            })
    }

    @Test func authenticateSkipsForDirectTransport() async {
        let runner = ScriptedProvisionRunner()
        let provisioner = ClusterProvisioner(runner: runner, site: .gpuWorkstation)
        let alive = await provisioner.runAuthenticateCheck()
        #expect(alive)
        guard case .skipped = provisioner.record(for: .authenticate).status else {
            Issue.record("expected .skipped for direct transport")
            return
        }
        #expect(await runner.callCount() == 0)
    }

    // MARK: Push code

    @Test func pushArgvIsTheExactRsyncTemplate() {
        let argv = ClusterProvisioner.pushArgv(
            site: slurmSite(), localRepoPath: "/Users/me/Project", remoteRepoPath: "~/steerlab")
        #expect(
            argv == [
                "/usr/bin/rsync", "-az", "--delete", "--prune-empty-dirs",
                "--exclude", ".venv*",
                "--exclude", "runs",
                "--exclude", "__pycache__",
                "--exclude", ".pytest_cache",
                "--exclude", ".mypy_cache",
                "--exclude", ".ruff_cache",
                "--exclude", "*.egg-info",
                "--exclude", "*.pyc",
                "--exclude", ".coverage*",
                "--exclude", ".DS_Store",
                "--include", "/Server/",
                "--include", "/Server/***",
                "--include", "/prompts/",
                "--include", "/prompts/fixtures/",
                "--include", "/prompts/fixtures/***",
                "--exclude", "*",
                "-e", "ssh -o ControlPath=~/.ssh/steerlab-cm-%C -o BatchMode=yes",
                "/Users/me/Project/", "login.test:~/steerlab/",
            ])
    }

    @Test func pushFiltersAreAllowlistedAndCannotDeleteExcludedRemoteFiles() {
        let filters = ClusterProvisioner.pushFilterArguments
        #expect(filters.contains("/Server/***"))
        #expect(filters.contains("/prompts/fixtures/***"))
        #expect(filters.suffix(2) == ["--exclude", "*"])
        #expect(!filters.contains("--delete-excluded"))

        let argv = ClusterProvisioner.pushArgv(
            site: slurmSite(), localRepoPath: "/p", remoteRepoPath: "~/s") ?? []
        #expect(argv.contains("--delete"))
        #expect(!argv.contains("--delete-excluded"))
    }

    @Test func pushArgvHonorsProxyJumpAndRefusesDirectTransport() {
        let jumped = ClusterProvisioner.pushArgv(
            site: slurmSite(proxyJump: "jump.host"), localRepoPath: "/p", remoteRepoPath: "~/s")
        #expect(
            jumped?.contains(
                "ssh -o ControlPath=~/.ssh/steerlab-cm-%C -o BatchMode=yes -J jump.host")
                == true)
        #expect(
            ClusterProvisioner.pushArgv(
                site: .gpuWorkstation, localRepoPath: "/p", remoteRepoPath: "~/s") == nil)
    }

    @Test func runPushCodeStreamsTheTranscriptAndSucceeds() async {
        let runner = ScriptedProvisionRunner([
            .init(lines: ["sending incremental file list", "Server/", "sent 42 bytes"], exit: 0)
        ])
        let provisioner = ClusterProvisioner(runner: runner, site: slurmSite())
        provisioner.localRepoPath = "/Users/me/Project"
        await provisioner.runPushCode()
        guard case .succeeded(let note) = provisioner.record(for: .pushCode).status else {
            Issue.record("expected .succeeded, got \(provisioner.record(for: .pushCode).status)")
            return
        }
        #expect(note.contains("login.test"))
        let transcript = provisioner.record(for: .pushCode).transcript
        #expect(transcript.first?.hasPrefix("$ /usr/bin/rsync") == true)
        #expect(transcript.contains("sending incremental file list"))
        #expect(transcript.contains("sent 42 bytes"))
    }

    @Test func failedPushIsStampedFailed() async {
        let runner = ScriptedProvisionRunner([.init(lines: ["rsync: connection refused"], exit: 255)])
        let provisioner = ClusterProvisioner(runner: runner, site: slurmSite())
        await provisioner.runPushCode()
        guard case .failed(let note) = provisioner.record(for: .pushCode).status else {
            Issue.record("expected .failed")
            return
        }
        #expect(note.contains("255"))
    }

    // MARK: Push — deployment-manifest verification (packaged payloads)

    /// A staged payload directory in the ResourceManifest layout; caller
    /// removes it. `drifted` mutates a file AFTER manifest generation.
    private func stagePackagedPayload(drifted: Bool) throws -> URL {
        let fm = FileManager.default
        let payload = fm.temporaryDirectory.appending(
            component: "steerlab-packaged-payload-\(UUID().uuidString)")
        let scripts = payload.appending(components: "Server", "scripts")
        try fm.createDirectory(at: scripts, withIntermediateDirectories: true)
        let bootstrap = scripts.appending(component: "bootstrap.sh")
        try "#!/bin/sh\necho ok\n".write(to: bootstrap, atomically: true, encoding: .utf8)
        let manifest = try ResourceManifest.generate(
            over: payload, serverVersion: "0.0-test", protocolVersion: 1)
        try manifest.write(
            to: payload.appending(
                component: ClusterProvisioner.deploymentManifestFileName))
        if drifted {
            try "#!/bin/sh\necho drifted\n".write(
                to: bootstrap, atomically: true, encoding: .utf8)
        }
        return payload
    }

    @Test func pushArgvShipsTheDeploymentManifestOnlyWhenAsked() {
        let plain = ClusterProvisioner.pushArgv(
            site: slurmSite(), localRepoPath: "/p", remoteRepoPath: "~/s") ?? []
        let shipping = ClusterProvisioner.pushArgv(
            site: slurmSite(), localRepoPath: "/p", remoteRepoPath: "~/s",
            includeDeploymentManifest: true) ?? []
        #expect(!plain.contains("/deployment-manifest.json"))
        let index = shipping.firstIndex(of: "/deployment-manifest.json")
        #expect(index != nil)
        guard let index else { return }
        // The include lands immediately before the catch-all exclude…
        #expect(shipping[index - 1] == "--include")
        #expect(Array(shipping[(index + 1)...(index + 2)]) == ["--exclude", "*"])
        // …and stripping it restores today's dev template byte for byte.
        var stripped = shipping
        stripped.removeSubrange((index - 1)...index)
        #expect(stripped == plain)
    }

    @Test func pushRefusesADriftedPackagedPayloadBeforeAnyBytesMove() async throws {
        let payload = try stagePackagedPayload(drifted: true)
        defer { try? FileManager.default.removeItem(at: payload) }
        let runner = ScriptedProvisionRunner()
        let provisioner = ClusterProvisioner(runner: runner, site: slurmSite())
        provisioner.localRepoPath = payload.path
        await provisioner.runPushCode()
        guard case .failed(let note) = provisioner.record(for: .pushCode).status else {
            Issue.record("expected .failed, got \(provisioner.record(for: .pushCode).status)")
            return
        }
        #expect(note.contains("does not match its manifest"))
        #expect(note.contains("Server/scripts/bootstrap.sh"))
        #expect(note.contains("make-server-payload.sh"))
        #expect(await runner.callCount() == 0)  // refused BEFORE rsync ran
    }

    @Test func pushShipsTheManifestWithACleanPackagedPayload() async throws {
        let payload = try stagePackagedPayload(drifted: false)
        defer { try? FileManager.default.removeItem(at: payload) }
        let runner = ScriptedProvisionRunner([.init(lines: ["sent 42 bytes"], exit: 0)])
        let provisioner = ClusterProvisioner(runner: runner, site: slurmSite())
        provisioner.localRepoPath = payload.path
        await provisioner.runPushCode()
        guard case .succeeded = provisioner.record(for: .pushCode).status else {
            Issue.record("expected .succeeded, got \(provisioner.record(for: .pushCode).status)")
            return
        }
        let calls = await runner.recordedCalls()
        #expect(calls.count == 1)
        #expect(calls.first?.contains("/deployment-manifest.json") == true)
        #expect(provisioner.pushCommandPreview?.contains("/deployment-manifest.json") == true)
    }

    @Test func pushFailsClosedOnABlankPayloadPath() async {
        // A release build missing its bundled payload resolves the default
        // path to "" — the push must refuse plainly, not rsync the CWD.
        let runner = ScriptedProvisionRunner()
        let provisioner = ClusterProvisioner(runner: runner, site: slurmSite())
        provisioner.localRepoPath = ""
        await provisioner.runPushCode()
        guard case .failed = provisioner.record(for: .pushCode).status else {
            Issue.record("expected .failed, got \(provisioner.record(for: .pushCode).status)")
            return
        }
        #expect(provisioner.pushCommandPreview == nil)
        #expect(await runner.callCount() == 0)
    }

    // MARK: Bootstrap — dry-run ALWAYS before real

    private func dryRunReportLine(
        ok: Bool = true, envFile: String = "/home/me/steerlab-cluster.env",
        prefix: String = "/home/me/envs/steerlab"
    ) -> String {
        """
        {"ok":\(ok),"steps":{"condaDetect":"planned","envCreate":"planned",\
        "manifestCheck":"skipped","torchInstall":"planned","serverInstall":"planned",\
        "envFile":"planned","profileValidate":"planned","helloJob":"skipped"},\
        "envFile":"\(envFile)","prefix":"\(prefix)"}
        """
    }

    /// Audit c30 (WP5 Step 12): the torch build is a SITE fact. `bootstrap.sh`
    /// still defaults to a CUDA 12.8 index for a hand run, but a site that
    /// declares `environment.torchIndexURL` must have it passed — on ROCm,
    /// older CUDA, or CPU-only nodes the default installs wheels whose kernels
    /// the hardware cannot run. Undeclared stays silent, so no existing
    /// invocation gains a word.
    @Test func bootstrapArgvPassesADeclaredTorchIndexAndOmitsAnUndeclaredOne() throws {
        var site = slurmSite()
        site.environment.torchIndexURL = "https://download.pytorch.org/whl/rocm6.2"
        let declared = try #require(
            ClusterProvisioner.bootstrapArgv(
                site: site, remoteRepoPath: "~/steerlab", envPrefix: "", pythonVersion: "",
                executionTarget: .sshHost, jobPartition: "batch", jobCPUs: 4,
                jobMemory: "16G", jobWalltime: "02:00:00", squeueCommand: "squeue",
                force: false, hello: false, dryRun: true))
        #expect(
            declared.joined(separator: " ")
                .contains("--torch-index https://download.pytorch.org/whl/rocm6.2"))

        let undeclared = try #require(
            ClusterProvisioner.bootstrapArgv(
                site: slurmSite(), remoteRepoPath: "~/steerlab", envPrefix: "",
                pythonVersion: "", executionTarget: .sshHost, jobPartition: "batch",
                jobCPUs: 4, jobMemory: "16G", jobWalltime: "02:00:00",
                squeueCommand: "squeue", force: false, hello: false, dryRun: true))
        #expect(!undeclared.joined(separator: " ").contains("--torch-index"))
    }

    @Test func bootstrapDryRunComposesTheSiteDrivenInvocation() async {
        let runner = ScriptedProvisionRunner([
            .init(lines: ["bootstrap.sh plan (dry run — nothing executed):", dryRunReportLine()])
        ])
        let site = slurmSite(account: "lab1")
        let provisioner = ClusterProvisioner(runner: runner, site: site)
        await provisioner.runBootstrap(dryRun: true)
        let calls = await runner.recordedCalls()
        // WP5 Step 7: the wizard materializes by default, so the invocation
        // names the rendered env file and the digest that puts it inside the
        // reviewed plan hash. The digest is the renderer's — asserting it
        // literally here would pin the env-file goldens twice.
        let digest = ClusterEnvironmentRenderer.envFileDigest(site)
        #expect(
            calls.first
                == controlPrefix + [
                    "login.test",
                    "bash ~/steerlab/Server/scripts/submit-bootstrap-job.sh "
                        + "--bootstrap-script ~/steerlab/Server/scripts/bootstrap.sh "
                        + "--job-workspace /scratch/me/ws --job-partition batch "
                        + "--job-cpus 4 --job-memory 16G --job-walltime 02:00:00 "
                        + "--squeue-command squeue "
                        + "--job-account lab1 --dry-run -- --repo ~/steerlab "
                        + "--workspace /scratch/me/ws --hf-cache /work/lab/hf "
                        + "--partition gpu_p --gres gpu:A100:1 --account lab1 "
                        + "--walltime 24:00:00 "
                        + "--env-file-from ~/.steerlab/rendered-cluster.env "
                        + "--env-file-sha256 \(digest)",
                ])
        guard case .awaitingConfirmation = provisioner.record(for: .bootstrap).status else {
            Issue.record("dry-run success should await the real-run confirmation")
            return
        }
        #expect(provisioner.dryRunReport?.steps["condaDetect"] == "planned")
        #expect(provisioner.dryRunReport?.envFile == "/home/me/steerlab-cluster.env")
        #expect(provisioner.realBootstrapUnlocked)
    }

    /// **WP5 Step 9 (audit c19): the setup job's shape is SITE data.** Its
    /// resources were wizard constants (4 / 16G / 02:00:00), so a site could
    /// describe its bootstrap job and be ignored. They now seed from
    /// `scheduler.setupJob` — and because they ride in the helper's argv, the
    /// reviewed bootstrap plan hash covers them.
    @Test func bootstrapJobResourcesComeFromTheSitesSetupClass() async {
        var site = slurmSite(account: "lab1")
        guard case .slurm(var slurm) = site.scheduler else {
            Issue.record("expected a Slurm site")
            return
        }
        slurm.setupJob = .init(
            partition: "long", cpusPerTask: 6, memory: "32G", walltime: "03:00:00",
            extraSbatch: ["--hint=nomultithread", "bad directive"])
        site.scheduler = .slurm(slurm)
        let runner = ScriptedProvisionRunner([.init(lines: [dryRunReportLine()])])
        let provisioner = ClusterProvisioner(runner: runner, site: site)
        #expect(provisioner.bootstrapJobCPUs == 6)
        #expect(provisioner.bootstrapJobMemory == "32G")
        #expect(provisioner.bootstrapJobWalltime == "03:00:00")
        #expect(provisioner.bootstrapJobPartition == "long")
        await provisioner.runBootstrap(dryRun: true)
        let submitted = (await runner.recordedCalls()).first?.last ?? ""
        #expect(
            submitted.contains(
                "--job-partition long --job-cpus 6 --job-memory 32G "
                    + "--job-walltime 03:00:00"))
        // The class's own directives ride along; a "directive" that is not one
        // is dropped here rather than word-split into the helper's argv.
        #expect(submitted.contains("--job-extra-sbatch --hint=nomultithread"))
        #expect(!submitted.contains("bad directive"))
        // The headless configuration resolves the same facts as the wizard.
        let configuration = ClusterProvisioningConfiguration.defaults(for: site)
        #expect(configuration.bootstrapJobCPUs == 6)
        #expect(configuration.bootstrapJobMemory == "32G")
        #expect(configuration.bootstrapJobWalltime == "03:00:00")
        // An undeclared class keeps the helper script's own fallbacks, which
        // is what `submit-bootstrap-job.sh` still carries.
        let undeclared = ClusterProvisioningConfiguration.defaults(for: slurmSite())
        #expect(undeclared.bootstrapJobCPUs == 4)
        #expect(undeclared.bootstrapJobMemory == "16G")
        #expect(undeclared.bootstrapJobWalltime == "02:00:00")
    }

    @Test func bootstrapHonorsBootstrapPathPrefixPythonAndToggles() async {
        let runner = ScriptedProvisionRunner([.init(lines: [dryRunReportLine()])])
        let provisioner = ClusterProvisioner(
            runner: runner,
            site: slurmSite(bootstrapPath: "/opt/steerlab/Server/scripts/bootstrap.sh"))
        provisioner.envPrefix = "/home/me/envs/custom"
        provisioner.pythonVersion = "3.12"
        provisioner.bootstrapForce = true
        provisioner.bootstrapHello = true
        await provisioner.runBootstrap(dryRun: true)
        let calls = await runner.recordedCalls()
        let remote = calls.first?.last ?? ""
        #expect(remote.hasPrefix("bash ~/steerlab/Server/scripts/submit-bootstrap-job.sh "))
        #expect(remote.contains("--bootstrap-script /opt/steerlab/Server/scripts/bootstrap.sh"))
        #expect(remote.contains("--prefix /home/me/envs/custom"))
        #expect(remote.contains("--python 3.12"))
        #expect(remote.contains("--force"))
        #expect(remote.contains("--hello"))
        #expect(remote.contains("--dry-run -- --repo"))
    }

    @Test func realBootstrapRefusesWithoutAMatchingDryRun() async {
        let runner = ScriptedProvisionRunner()
        let provisioner = ClusterProvisioner(runner: runner, site: slurmSite())
        await provisioner.runBootstrap(dryRun: false)
        guard case .failed(let note) = provisioner.record(for: .bootstrap).status else {
            Issue.record("expected refusal before any dry-run")
            return
        }
        #expect(note.contains("dry-run"))
        #expect(await runner.callCount() == 0)  // refused before composing anything
    }

    @Test func settingsChangeInvalidatesTheDryRunGate() async {
        let runner = ScriptedProvisionRunner([.init(lines: [dryRunReportLine()])])
        let provisioner = ClusterProvisioner(runner: runner, site: slurmSite())
        await provisioner.runBootstrap(dryRun: true)
        #expect(provisioner.realBootstrapUnlocked)
        provisioner.bootstrapForce = true  // plan changed — must re-review
        #expect(!provisioner.realBootstrapUnlocked)
        await provisioner.runBootstrap(dryRun: false)
        guard case .failed = provisioner.record(for: .bootstrap).status else {
            Issue.record("expected refusal after a settings change")
            return
        }
        #expect(await runner.callCount() == 1)  // only the dry-run ever ran
    }

    @Test func realBootstrapReRunsTheReviewedPlanMinusDryRunAndParsesTheReport() async {
        let realReport = """
            {"ok":true,"steps":{"condaDetect":"ok","envCreate":"ok","manifestCheck":"ok",\
            "torchInstall":"ok","serverInstall":"ok","envFile":"ok","profileValidate":"ok",\
            "helloJob":"skipped"},\
            "envFile":"/home/me/steerlab-cluster.env","prefix":"/home/me/envs/steerlab"}
            """
        let runner = ScriptedProvisionRunner([
            .init(lines: [dryRunReportLine()]),
            // WP5 Step 7: the rendered environment is pushed first, before
            // anything is submitted.
            .init(),
            // The helper submits and returns; the job id arrives before
            // anything could interrupt the session (review finding 5).
            .init(lines: [
                "STEERLAB_BOOTSTRAP_JOB_ID=7788",
                "STEERLAB_BOOTSTRAP_STATUS_FILE=/scratch/me/ws/b.status",
            ]),
            // …and a separate, short status command carries the verdict plus
            // the job log the report is parsed from.
            .init(lines: [
                "STEERLAB_BOOTSTRAP_STATUS=completed-ok job=7788 exit=0",
                "bootstrap.sh: using /apps/mamba",
                realReport,
            ]),
        ])
        let provisioner = ClusterProvisioner(runner: runner, site: slurmSite())
        provisioner.bootstrapPollInterval = .zero
        provisioner.bootstrapPollAttempts = 3
        await provisioner.runBootstrap(dryRun: true)
        await provisioner.runBootstrap(dryRun: false)
        let calls = await runner.recordedCalls()
        // dry run → env push → submit → status probe (WP5 Step 7 added the push).
        #expect(calls.count == 4)
        guard calls.count == 4 else { return }
        let dry = calls[0].last ?? ""
        #expect(calls[1].last?.contains("rendered-cluster.env") == true)
        let real = calls[2].last ?? ""
        #expect(dry.replacingOccurrences(of: " --dry-run -- ", with: " -- ") == real)
        // The follow is a status read, never a second submission.
        let probe = calls[3].last ?? ""
        #expect(probe.contains("--status --job-workspace /scratch/me/ws --job-id 7788"))
        #expect(!probe.contains("--bootstrap-script"))
        guard case .succeeded(let note) = provisioner.record(for: .bootstrap).status else {
            Issue.record("expected .succeeded, got \(provisioner.record(for: .bootstrap).status)")
            return
        }
        #expect(note.contains("/home/me/envs/steerlab"))
        #expect(note.contains("job 7788"))
        #expect(provisioner.bootstrapJobID == "7788")
        #expect(provisioner.realReport?.ok == true)
        #expect(provisioner.realReport?.orderedSteps.map(\.name) == BootstrapReport.stepOrder)
    }

    @Test func failedBootstrapStepSurfacesItsStderrTailVerbatim() async {
        let failedLine =
            "bootstrap.sh: STEP FAILED [torchInstall]: pip install torch from "
            + "https://download.pytorch.org/whl/cu128 failed — no egress from this node?"
        let report = """
            {"ok":false,"steps":{"condaDetect":"ok","envCreate":"ok",\
            "manifestCheck":"skipped","torchInstall":"failed","serverInstall":"ok",\
            "envFile":"ok","profileValidate":"skipped","helloJob":"skipped"},\
            "envFile":"/home/me/steerlab-cluster.env","prefix":"/home/me/envs/steerlab"}
            """
        let runner = ScriptedProvisionRunner([
            .init(lines: [dryRunReportLine(ok: true)]),
            .init(),  // WP5 Step 7: the rendered environment push.
            .init(lines: [
                "STEERLAB_BOOTSTRAP_JOB_ID=7788",
                "STEERLAB_BOOTSTRAP_STATUS_FILE=/scratch/me/ws/b.status",
            ]),
            // A detached bootstrap must still say WHICH step failed and show
            // the failure's own stderr tail: the status probe forwards the
            // job log behind its verdict line.
            .init(
                lines: [
                    "STEERLAB_BOOTSTRAP_STATUS=completed-code-1 job=7788 exit=1",
                    "Collecting torch",
                    "ERROR: Could not find a version that satisfies the requirement torch",
                    failedLine,
                    report,
                ]),
        ])
        let provisioner = ClusterProvisioner(runner: runner, site: slurmSite())
        provisioner.bootstrapPollInterval = .zero
        provisioner.bootstrapPollAttempts = 3
        await provisioner.runBootstrap(dryRun: true)
        await provisioner.runBootstrap(dryRun: false)
        guard case .failed(let note) = provisioner.record(for: .bootstrap).status else {
            Issue.record("expected .failed")
            return
        }
        #expect(note.contains("torchInstall"))
        #expect(provisioner.realReport?.failedStepNames == ["torchInstall"])
        let detail = BootstrapReport.failureDetail(
            forStep: "torchInstall",
            inTranscript: provisioner.record(for: .bootstrap).transcript)
        #expect(detail.last == failedLine)  // the script's diagnostic, verbatim
        #expect(detail.contains("ERROR: Could not find a version that satisfies the requirement torch"))
    }

    @Test func slurmBootstrapRequiresExplicitWorkspaceAndHFCacheRoots() async {
        let runner = ScriptedProvisionRunner()
        let provisioner = ClusterProvisioner(
            runner: runner, site: slurmSite(storageRoots: [:]))
        #expect(provisioner.bootstrapExecutionTarget == .slurmBatch)
        #expect(provisioner.bootstrapConfigurationErrors.contains { $0.contains("workspace") })
        #expect(provisioner.bootstrapConfigurationErrors.contains { $0.contains("HF cache") })
        await provisioner.runBootstrap(dryRun: true)
        #expect(await runner.callCount() == 0)
        guard case .failed(let note) = provisioner.record(for: .bootstrap).status else {
            Issue.record("expected missing storage roots to fail before ssh")
            return
        }
        #expect(note.contains("HF cache"))
    }

    @Test func directSSHBootstrapIsExplicitAndStillDryRunGated() async {
        let runner = ScriptedProvisionRunner([.init(lines: [dryRunReportLine()])])
        let provisioner = ClusterProvisioner(runner: runner, site: slurmSite())
        provisioner.bootstrapExecutionTarget = .sshHost
        await provisioner.runBootstrap(dryRun: true)
        let remote = await runner.recordedCalls().first?.last ?? ""
        #expect(remote.hasPrefix("bash ~/steerlab/Server/scripts/bootstrap.sh "))
        #expect(remote.hasSuffix("--dry-run"))
        #expect(provisioner.bootstrapConfigurationWarnings.contains { $0.contains("login node") })
        #expect(provisioner.realBootstrapUnlocked)
    }

    @Test func bootstrapJobResourceChangeInvalidatesReviewedPlan() async {
        let runner = ScriptedProvisionRunner([.init(lines: [dryRunReportLine()])])
        let provisioner = ClusterProvisioner(runner: runner, site: slurmSite())
        await provisioner.runBootstrap(dryRun: true)
        #expect(provisioner.realBootstrapUnlocked)
        provisioner.bootstrapJobMemory = "24G"
        #expect(!provisioner.realBootstrapUnlocked)
    }

    @Test func bootstrapQueueCommandIsReviewedAndValidated() async {
        let runner = ScriptedProvisionRunner([.init(lines: [dryRunReportLine()])])
        let provisioner = ClusterProvisioner(runner: runner, site: slurmSite())
        provisioner.bootstrapSqueueCommand = "sq"
        await provisioner.runBootstrap(dryRun: true)
        let remote = await runner.recordedCalls().first?.last ?? ""
        #expect(remote.contains("--squeue-command sq"))
        #expect(provisioner.realBootstrapUnlocked)

        provisioner.bootstrapSqueueCommand = "sq --me"
        #expect(provisioner.bootstrapConfigurationErrors.contains { $0.contains("queue query") })
        #expect(!provisioner.realBootstrapUnlocked)
    }

    @Test func bootstrapReportParsingIgnoresTrailingNoise() {
        let report = BootstrapReport.parse(fromTranscript: [
            "junk before",
            #"{"ok":true,"steps":{"condaDetect":"ok"},"envFile":"/e","prefix":"/p","helloJobId":"77"}"#,
            "stderr noise after the report line",
            "{not json",
        ])
        #expect(report?.ok == true)
        #expect(report?.helloJobId == "77")
        #expect(report?.envFile == "/e")
        #expect(BootstrapReport.parse(fromTranscript: ["no json here"]) == nil)
    }

    // MARK: Validate

    @Test func validateSourcesTheEnvFileAndUsesThePrefixBinary() async {
        let runner = ScriptedProvisionRunner([
            .init(lines: [dryRunReportLine()]),
            .init(
                lines: [
                    "OK   profile: cluster/login/slurm",
                    "WARN bind: binds 0.0.0.0 (non-loopback) for the daemon-in-a-job topology",
                    "0 failure(s), 1 warning(s)",
                ], exit: 0),
        ])
        let provisioner = ClusterProvisioner(runner: runner, site: slurmSite())
        await provisioner.runBootstrap(dryRun: true)  // supplies envFile + prefix
        await provisioner.runValidate()
        let calls = await runner.recordedCalls()
        #expect(
            calls.last
                == controlPrefix + [
                    "login.test",
                    "bash -lc '. /home/me/steerlab-cluster.env && "
                        + "/home/me/envs/steerlab/bin/steerlab-server profile validate'",
                ])
        guard case .succeeded(let note) = provisioner.record(for: .validate).status else {
            Issue.record("expected .succeeded, got \(provisioner.record(for: .validate).status)")
            return
        }
        #expect(note.contains("1 warning"))
        #expect(provisioner.validateLines.map(\.kind) == [.ok, .warn, .other])
    }

    @Test func validateWithoutABootstrapReportUsesTheDocumentedDefaults() async {
        let runner = ScriptedProvisionRunner([
            .init(lines: ["FAIL auth: STEERLAB_AUTH_TOKEN is required for token auth"], exit: 1)
        ])
        let provisioner = ClusterProvisioner(runner: runner, site: slurmSite())
        await provisioner.runValidate()
        let calls = await runner.recordedCalls()
        // No bootstrap report in this session: the fallback must not be the
        // BARE binary (which fails on env files predating the PATH export) —
        // it prefers the env file's own STEERLAB_PREFIX.
        #expect(
            calls.first?.last
                == "bash -lc '. ~/steerlab-cluster.env && "
                + "${STEERLAB_PREFIX:+$STEERLAB_PREFIX/bin/}steerlab-server "
                + "profile validate'")
        guard case .failed(let note) = provisioner.record(for: .validate).status else {
            Issue.record("expected .failed on exit 1")
            return
        }
        #expect(note.contains("STEERLAB_AUTH_TOKEN"))  // FAIL lines verbatim
    }

    // MARK: Controller job (daemon-in-a-job only)

    @Test func controllerJobSkipsLoudlyForOtherTopologies() async {
        let runner = ScriptedProvisionRunner()
        let provisioner = ClusterProvisioner(runner: runner, site: slurmSite(topology: .loginDaemon))
        await provisioner.runControllerJob()
        guard case .skipped(let reason) = provisioner.record(for: .controllerJob).status else {
            Issue.record("expected .skipped")
            return
        }
        #expect(reason.contains("loginDaemon"))
        #expect(await runner.callCount() == 0)
        #expect(provisioner.summaryLines.contains { $0.contains("SKIPPED") })
    }

    @Test func controllerJobRendersSubmitsParsesTheJobIDAndPollsForTheNode() async {
        let runner = ScriptedProvisionRunner([
            .init(lines: ["Submitted batch job 4242"], exit: 0),
            .init(lines: [], exit: 1),  // serverd.host not written yet
            .init(lines: ["c4-12"], exit: 0),
        ])
        let provisioner = ClusterProvisioner(
            runner: runner, site: slurmSite(topology: .daemonInJob, account: "lab1"))
        provisioner.pollInterval = .zero
        await provisioner.runControllerJob()
        let calls = await runner.recordedCalls()
        #expect(calls.count == 3)
        guard calls.count == 3 else { return }
        // Submit: render the job in-file (the resubmit chain re-sbatches the
        // FILE, so CLI flags would not survive), then sbatch it. WP5 Step 9:
        // the `#SBATCH` block is COMPOSED from the profile by the shared header
        // renderer and printed under the shebang — partition, account, walltime,
        // cores and memory are the site's facts now, not four placeholders over
        // a fixed block in the template.
        #expect(
            calls[0]
                == controlPrefix + [
                    "login.test",
                    // The site declares a workspace storage root, so $WS is
                    // that concrete path; the runs/ log dir is mkdir'd before
                    // sbatch — Slurm silently drops logs otherwise.
                    "WS=/scratch/me/ws && "
                        + "mkdir -p ~/.steerlab \"$WS/runs\" && "
                        // Render provenance, computed in the REMOTE shell so
                        // the command string itself stays deterministic
                        // (open-issues §1 field report, 2026-08-20). Every
                        // substitution exits 0 even when the fact is missing,
                        // or the `&&` chain would skip the render.
                        + Self.expectedStampAssignments
                        + "{ printf '%s\\n' '#!/usr/bin/env bash' "
                        + "'#SBATCH --job-name=steerlab-serverd' "
                        + "'#SBATCH --partition=batch' "
                        + "'#SBATCH --account=lab1' "
                        // min(cap 168h, 24h): the resubmit chain covers longer
                        // sessions; a cap-length request just collides with
                        // maintenance reservations (live 2026-07-21).
                        + "'#SBATCH --time=24:00:00' "
                        + "'#SBATCH --ntasks=1' "
                        + "'#SBATCH --cpus-per-task=1' "
                        + "'#SBATCH --mem=16G' "
                        // The pre-expiry hook the template's successor chain
                        // hangs off (open-issues §1, 2026-08-18). Always the
                        // BATCH form: a step-targeted USR1 would land on the
                        // Python server, which has no handler for it.
                        + "'#SBATCH --signal=B:USR1@600' "
                        // Bundle-owned, appended after the site block — the
                        // same split the study path uses.
                        + "\"#SBATCH --output=$WS/runs/steerlab-serverd.%j.out\" "
                        + "\"#SBATCH --error=$WS/runs/steerlab-serverd.%j.err\"; "
                        + "tail -n +2 "
                        + "~/steerlab/Server/scripts/controller-job.sbatch.template "
                        + "| sed -e 's|@PYTHON@|$HOME/envs/steerlab/bin/python|g' "
                        + "-e 's|@PORT@|8080|g'"
                        + Self.expectedStampSedExpressions + "; } "
                        + "> ~/.steerlab/controller-job.sbatch && "
                        // Submit FROM the workspace: the controller (and its
                        // auto-resubmit chain) inherits a scratch-backed cwd.
                        + "cd \"$WS\" && sbatch ~/.steerlab/controller-job.sbatch",
                ])
        // Poll: the tunnel's own remote-read convention.
        #expect(calls[1] == controlPrefix + ["login.test", "cat", "~/.steerlab/serverd.host"])
        #expect(calls[2] == calls[1])
        #expect(provisioner.controllerJobID == "4242")
        #expect(provisioner.daemonHost == "c4-12")
        guard case .succeeded(let note) = provisioner.record(for: .controllerJob).status else {
            Issue.record("expected .succeeded, got \(provisioner.record(for: .controllerJob).status)")
            return
        }
        #expect(note.contains("4242"))
        #expect(note.contains("c4-12"))
    }

    @Test func controllerJobUsesTheBootstrapPrefixAndBootstrapPathWhenKnown() async {
        let runner = ScriptedProvisionRunner([
            .init(lines: [dryRunReportLine(prefix: "/home/me/envs/steerlab")]),
            .init(lines: ["Submitted batch job 7"], exit: 0),
            .init(lines: ["node-1"], exit: 0),
        ])
        let site = slurmSite(
            topology: .daemonInJob,
            bootstrapPath: "/opt/steerlab/Server/scripts/bootstrap.sh",
            storageRoots: ["metadata": "/home/me/.steerlab"])
        let provisioner = ClusterProvisioner(runner: runner, site: site)
        provisioner.pollInterval = .zero
        // This test deliberately omits a workspace root to exercise the
        // controller's env-file fallback. Use the advanced direct dry run so
        // the separate Slurm-bootstrap safety gate does not reject that setup.
        provisioner.bootstrapExecutionTarget = .sshHost
        await provisioner.runBootstrap(dryRun: true)
        await provisioner.runControllerJob()
        let calls = await runner.recordedCalls()
        #expect(calls.count == 3)
        guard calls.count == 3 else { return }
        let submit = calls[1].last ?? ""
        #expect(submit.contains("'s|@PYTHON@|/home/me/envs/steerlab/bin/python|g'"))
        #expect(submit.contains("/opt/steerlab/Server/scripts/controller-job.sbatch.template"))
        #expect(submit.contains("mkdir -p /home/me/.steerlab"))
        #expect(calls[2].suffix(2) == ["cat", "/home/me/.steerlab/serverd.host"])
    }

    @Test func controllerJobResolvesWorkspaceFromTheEnvFileWhenUnconfigured() async {
        // No workspace storage root on this site: $WS resolves remotely from
        // the bootstrap env file (STEERLAB_ROOT) — and with neither source it
        // FAILS CLOSED (exit 64) rather than quietly parking job logs in
        // /home, the exact thing the surrounding contract forbids.
        let runner = ScriptedProvisionRunner([
            .init(lines: ["Submitted batch job 11"], exit: 0),
            .init(lines: ["node-2"], exit: 0),
        ])
        let site = slurmSite(topology: .daemonInJob, storageRoots: [:])
        let provisioner = ClusterProvisioner(runner: runner, site: site)
        provisioner.pollInterval = .zero
        await provisioner.runControllerJob()
        let submit = (await runner.recordedCalls()).first?.last ?? ""
        #expect(
            submit.hasPrefix(
                "WS=\"$( . ~/steerlab-cluster.env 2>/dev/null; "
                    + "printf '%s' \"${STEERLAB_ROOT:-}\" )\" && [ -n \"$WS\" ] || "))
        #expect(submit.contains("exit 64"))
        #expect(!submit.contains("$HOME/runs"))
        #expect(!submit.contains("{STEERLAB_ROOT:-$HOME}"))
        #expect(submit.contains("mkdir -p ~/.steerlab \"$WS/runs\""))
        #expect(submit.contains("\"#SBATCH --output=$WS/runs/steerlab-serverd.%j.out\""))
        #expect(submit.contains("cd \"$WS\" && sbatch "))
    }

    @Test func controllerCommandSubstitutesEveryTemplatePlaceholder() throws {
        // Drift guard for the render contract: every @PLACEHOLDER@ the
        // committed template declares must appear in the app's render command
        // (the @WORKSPACE@ regression shipped exactly because it did not).
        //
        // WP5 Step 9 narrowed what the template may declare: the site-fact
        // placeholders are gone with the `#SBATCH` block they filled, so the
        // template carries only the BODY's own two — and it must carry no
        // scheduler directive of its own, or the rendered file would hold two
        // contradicting blocks.
        let templateURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // ExperimentKitTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // repo root
            .appendingPathComponent("Server/scripts/controller-job.sbatch.template")
        let template = try String(contentsOf: templateURL, encoding: .utf8)
        var placeholders = Set<String>()
        // `[A-Z0-9_]+`, not `[A-Z]+`: the render-provenance placeholders carry
        // digits and underscores, and the narrower pattern would have skipped
        // them silently — which is the exact shape of the @WORKSPACE@ bug this
        // guard exists to prevent.
        for match in template.matches(of: /@([A-Z0-9_]+)@/) {
            placeholders.insert(String(match.1))
        }
        #expect(
            placeholders.subtracting(["PLACEHOLDER"])
                == ["PYTHON", "PORT", "TEMPLATE_SHA256", "RENDERED_AT",
                    "SOURCE_REVISION"])
        #expect(!template.split(separator: "\n").contains { $0.hasPrefix("#SBATCH ") })
        var site = slurmSite(topology: .daemonInJob, account: "lab1")
        site.constraints.storageRoots["workspace"] = "/scratch/me/ws"
        let command = ClusterProvisioner.controllerRemoteCommand(
            site: site, remoteRepoPath: "~/steerlab", envPrefix: nil,
            envFile: "~/steerlab-cluster.env")
        for placeholder in placeholders.subtracting(["PLACEHOLDER"]) {
            #expect(
                command.contains("@\(placeholder)@"),
                "template placeholder @\(placeholder)@ has no substitution in the render command")
        }
        // The block the command prints is the RENDERER's, verbatim — one
        // header vocabulary for the preview, the goldens, and what runs.
        for line in ClusterEnvironmentRenderer.renderSchedulerHeaders(
            site, jobClass: .controller)
        {
            #expect(
                command.contains("'\(line)'"),
                "controller header \(line) is not printed by the render command")
        }
    }

    @Test func controllerHeadersFollowTheSitesDeclaredJobClass() {
        // The point of Step 9: `scheduler.controllerJob` reaches the job that
        // runs, instead of the template's fixed 1 CPU / 16G / derived
        // partition. `--nice=100` proves the CLASS's own extraSbatch is
        // emitted — the escape hatch Step 6 promised the CPU-only classes when
        // it stopped them inheriting the site-wide placement directives.
        var site = slurmSite(topology: .daemonInJob, account: "lab1")
        guard case .slurm(var slurm) = site.scheduler else {
            Issue.record("expected a Slurm site")
            return
        }
        slurm.controllerJob = .init(
            partition: "long", cpusPerTask: 2, memory: "24G", walltime: "48:00:00",
            port: 9090, extraSbatch: ["--nice=100"])
        site.scheduler = .slurm(slurm)
        let command = ClusterProvisioner.controllerRemoteCommand(
            site: site, remoteRepoPath: "~/steerlab", envPrefix: nil,
            envFile: "~/steerlab-cluster.env")
        for directive in [
            "'#SBATCH --partition=long'", "'#SBATCH --cpus-per-task=2'",
            "'#SBATCH --mem=24G'", "'#SBATCH --time=48:00:00'",
            "'#SBATCH --nice=100'",
        ] {
            #expect(command.contains(directive), "missing \(directive)")
        }
        // A GPU-only class fact must NOT leak onto this CPU-only daemon.
        #expect(!command.contains("--gres"))
        // audit a12: the declared controller port becomes the rendered
        // default, ahead of the transport's remote port (8080 here).
        #expect(command.contains("'s|@PORT@|9090|g'"))
    }

    @Test func controllerJobFailsWhenSbatchReportsNoJobID() async {
        let runner = ScriptedProvisionRunner([
            .init(lines: ["sbatch: error: invalid partition specified"], exit: 1)
        ])
        let provisioner = ClusterProvisioner(runner: runner, site: slurmSite(topology: .daemonInJob))
        provisioner.pollInterval = .zero
        await provisioner.runControllerJob()
        guard case .failed(let note) = provisioner.record(for: .controllerJob).status else {
            Issue.record("expected .failed")
            return
        }
        #expect(note.contains("job id"))
        #expect(provisioner.controllerJobID == nil)
    }

    @Test func controllerJobFailsAfterThePollBudgetRunsOut() async {
        let runner = ScriptedProvisionRunner([
            .init(lines: ["Submitted batch job 9"], exit: 0),
            .init(lines: [], exit: 1),
            .init(lines: [], exit: 1),
        ])
        let provisioner = ClusterProvisioner(runner: runner, site: slurmSite(topology: .daemonInJob))
        provisioner.pollInterval = .zero
        provisioner.pollAttempts = 2
        await provisioner.runControllerJob()
        guard case .failed(let note) = provisioner.record(for: .controllerJob).status else {
            Issue.record("expected .failed")
            return
        }
        #expect(note.contains("serverd.host"))
    }

    @Test func controllerWalltimeIsAtMostOneDayAndRespectsShorterCaps() {
        // The auto-resubmit chain covers longer sessions: requesting the
        // partition cap buys nothing and held the controller PENDING behind
        // a maintenance reservation (observed live, 2026-07-21).
        func slurmData(cap: Int?) -> ClusterSiteProfile.SlurmSiteData {
            ClusterSiteProfile.SlurmSiteData(
                partitions: [.init(name: "batch", maxWalltimeHours: cap)])
        }
        #expect(ClusterProvisioner.controllerWalltime(for: slurmData(cap: 168)) == "24:00:00")
        #expect(ClusterProvisioner.controllerWalltime(for: slurmData(cap: 12)) == "12:00:00")
        #expect(ClusterProvisioner.controllerWalltime(for: slurmData(cap: nil)) == "24:00:00")
    }

    @Test func sbatchJobIDParsing() {
        #expect(
            ClusterProvisioner.parseSbatchJobID(from: ["Submitted batch job 123"]) == "123")
        #expect(
            ClusterProvisioner.parseSbatchJobID(
                from: ["noise", "Submitted batch job 99 on cluster example-hpc"]) == "99")
        #expect(ClusterProvisioner.parseSbatchJobID(from: ["sbatch: error"]) == nil)
        #expect(ClusterProvisioner.parseSbatchJobID(from: ["Submitted batch job abc"]) == nil)
    }

    // MARK: Skip / re-run / site switch

    @Test func skipStampsLoudlyAndAReRunOverwritesIt() async {
        let runner = ScriptedProvisionRunner([.init(lines: ["done"], exit: 0)])
        let provisioner = ClusterProvisioner(runner: runner, site: slurmSite())
        provisioner.skip(.pushCode)
        #expect(
            provisioner.summaryLines.contains {
                $0.contains("Push code") && $0.contains("SKIPPED (skipped by user)")
            })
        await provisioner.runPushCode()  // steps stay idempotent to re-run
        guard case .succeeded = provisioner.record(for: .pushCode).status else {
            Issue.record("a re-run after skip should succeed")
            return
        }
        #expect(!provisioner.summaryLines.contains { $0.contains("SKIPPED") })
    }

    @Test func selectingADifferentSiteResetsDownstreamState() async {
        let runner = ScriptedProvisionRunner([.init(lines: [dryRunReportLine()])])
        let provisioner = ClusterProvisioner(runner: runner, site: slurmSite())
        await provisioner.runBootstrap(dryRun: true)
        #expect(provisioner.dryRunReport != nil)
        #expect(provisioner.realBootstrapUnlocked)
        provisioner.selectSite(slurmSite(topology: .daemonInJob, account: "other"))
        #expect(provisioner.dryRunReport == nil)
        #expect(!provisioner.realBootstrapUnlocked)  // stale plans never carry over
        #expect(provisioner.record(for: .bootstrap).status == .pending)
        #expect(provisioner.record(for: .bootstrap).transcript.isEmpty)
        guard case .succeeded = provisioner.record(for: .site).status else {
            Issue.record("site step should stamp the new selection")
            return
        }
    }

    // MARK: Connect + capability summary

    @Test func connectStepRunsTheInjectedHandler() async {
        let provisioner = ClusterProvisioner(runner: ScriptedProvisionRunner(), site: slurmSite())
        provisioner.connectHandler = { "engine pytorch · devices: cuda:0" }
        await provisioner.runConnect()
        #expect(
            provisioner.record(for: .connect).status
                == .succeeded("engine pytorch · devices: cuda:0"))

        struct Boom: Error, LocalizedError {
            var errorDescription: String? { "tunnel did not come up" }
        }
        provisioner.connectHandler = { throw Boom() }
        await provisioner.runConnect()
        guard case .failed(let note) = provisioner.record(for: .connect).status else {
            Issue.record("expected .failed")
            return
        }
        #expect(note.contains("tunnel did not come up"))
    }

    // MARK: Connect-step controller-job honesty (live incident 2026-07-21:
    // controller PENDING behind a maintenance reservation, wizard showed the
    // generic "tunnel opened but not answering" line)

    private struct TunnelSilence: Error, LocalizedError {
        var errorDescription: String? {
            "the SSH tunnel opened, but the SteerLab controller is not answering behind it"
        }
    }

    @Test func failedConnectQueriesSlurmAndExplainsAPendingControllerJob() async {
        let runner = ScriptedProvisionRunner([
            .init(lines: ["Submitted batch job 4242"], exit: 0),
            .init(lines: ["c4-12"], exit: 0),  // serverd.host read-back
            .init(
                lines: ["PENDING|ReqNodeNotAvail, Reserved for maintenance"],
                exit: 0),
        ])
        let provisioner = ClusterProvisioner(
            runner: runner, site: slurmSite(topology: .daemonInJob))
        provisioner.pollInterval = .zero
        await provisioner.runControllerJob()
        provisioner.connectHandler = { throw TunnelSilence() }
        await provisioner.runConnect()
        // The probe rides the same ControlMaster, honors the squeue wrapper
        // name, and asks for state|reason in one shot.
        let calls = await runner.recordedCalls()
        #expect(calls.last == controlPrefix + ["login.test", "squeue -h -j 4242 -o '%T|%r'"])
        guard case .failed(let note) = provisioner.record(for: .connect).status else {
            Issue.record("expected .failed")
            return
        }
        #expect(note.contains("not answering"))  // the original failure survives
        #expect(note.contains("controller job 4242 is still waiting in the queue"))
        #expect(note.contains("reason: ReqNodeNotAvail, Reserved for maintenance"))
        #expect(note.contains("maintenance reservation"))
        #expect(note.contains("scontrol update JobId=4242 TimeLimit=1-00:00:00"))
        #expect(note.contains("Try Connect again once it shows RUNNING"))
    }

    @Test func failedConnectFallsBackHonestlyWhenTheProbeItselfFails() async {
        let runner = ScriptedProvisionRunner([
            .init(lines: ["Submitted batch job 8"], exit: 0),
            .init(lines: ["node-9"], exit: 0),
            .init(lines: [], exit: 255),  // ssh hiccup — not a job answer
        ])
        let provisioner = ClusterProvisioner(
            runner: runner, site: slurmSite(topology: .daemonInJob))
        provisioner.pollInterval = .zero
        await provisioner.runControllerJob()
        provisioner.connectHandler = { throw TunnelSilence() }
        await provisioner.runConnect()
        guard case .failed(let note) = provisioner.record(for: .connect).status else {
            Issue.record("expected .failed")
            return
        }
        #expect(note.contains("not answering"))
        #expect(note.contains("could not be checked"))
        #expect(note.contains("squeue -j 8"))
    }

    @Test func failedConnectDoesNotProbeWithoutAControllerJob() async {
        // loginDaemon topology (and any session with no submitted job id)
        // keeps the plain failure — no phantom squeue call.
        let runner = ScriptedProvisionRunner()
        let provisioner = ClusterProvisioner(runner: runner, site: slurmSite())
        provisioner.connectHandler = { throw TunnelSilence() }
        await provisioner.runConnect()
        #expect(await runner.callCount() == 0)
        guard case .failed(let note) = provisioner.record(for: .connect).status else {
            Issue.record("expected .failed")
            return
        }
        #expect(!note.contains("controller job"))
    }

    @Test func controllerJobProbeParsing() {
        #expect(
            ClusterProvisioner.parseControllerJobProbe(
                exitCode: 0,
                lines: ["PENDING|ReqNodeNotAvail, Reserved for maintenance"])
                == .state("PENDING", reason: "ReqNodeNotAvail, Reserved for maintenance"))
        #expect(
            ClusterProvisioner.parseControllerJobProbe(exitCode: 0, lines: ["RUNNING|None"])
                == .state("RUNNING", reason: "None"))
        // Reason column can be empty; state alone still parses.
        #expect(
            ClusterProvisioner.parseControllerJobProbe(exitCode: 0, lines: ["FAILED|"])
                == .state("FAILED", reason: ""))
        // Clean exit, no rows: the job left the queue.
        #expect(
            ClusterProvisioner.parseControllerJobProbe(exitCode: 0, lines: []) == .absent)
        // Aged-out job id is an ANSWER (absent), not a failed probe.
        #expect(
            ClusterProvisioner.parseControllerJobProbe(
                exitCode: 1, lines: ["slurm_load_jobs error: Invalid job id specified"])
                == .absent)
        // Any other failure says nothing about the job.
        #expect(
            ClusterProvisioner.parseControllerJobProbe(exitCode: 255, lines: [])
                == .unavailable)
        #expect(
            ClusterProvisioner.parseControllerJobProbe(
                exitCode: 1, lines: ["ssh: connect to host login.test: timed out"])
                == .unavailable)
    }

    @Test func maintenanceReservationReasonClassification() {
        for reason in [
            "ReqNodeNotAvail, Reserved for maintenance",
            "ReqNodeNotAvail, UnavailableNodes:c1-[1-8]",
            "(ReqNodeNotAvail, Reserved for maintenance)",
            "Reserved",
            "reqnodenotavail",
        ] {
            #expect(
                ClusterProvisioner.reasonSuggestsMaintenanceReservation(reason),
                "\(reason) should read as a maintenance reservation")
        }
        for reason in ["Priority", "Resources", "Dependency", "None", ""] {
            #expect(
                !ClusterProvisioner.reasonSuggestsMaintenanceReservation(reason),
                "\(reason) should NOT read as a maintenance reservation")
        }
    }

    @Test func controllerJobDiagnosisCoversEveryState() {
        let site = slurmSite(topology: .daemonInJob)

        let pendingPlain = ClusterProvisioner.controllerJobDiagnosis(
            jobID: "7", probe: .state("PENDING", reason: "Priority"), site: site)
        #expect(pendingPlain.contains("still waiting in the queue"))
        #expect(pendingPlain.contains("reason: Priority"))
        #expect(!pendingPlain.contains("maintenance"))  // no false alarms
        #expect(pendingPlain.contains("Try Connect again once it shows RUNNING"))

        let pendingMaintenance = ClusterProvisioner.controllerJobDiagnosis(
            jobID: "7",
            probe: .state("PENDING", reason: "ReqNodeNotAvail, Reserved for maintenance"),
            site: site)
        #expect(pendingMaintenance.contains("maintenance reservation"))
        #expect(pendingMaintenance.contains("scontrol update JobId=7 TimeLimit=1-00:00:00"))

        let running = ClusterProvisioner.controllerJobDiagnosis(
            jobID: "7", probe: .state("RUNNING", reason: "None"), site: site)
        #expect(running.contains("RUNNING"))
        #expect(running.contains("not answering"))
        #expect(running.contains("serverd.host"))
        #expect(running.contains("/scratch/me/ws/runs/steerlab-serverd.7.out"))

        for state in ["FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"] {
            let ended = ClusterProvisioner.controllerJobDiagnosis(
                jobID: "7", probe: .state(state, reason: ""), site: site)
            #expect(ended.contains("not running (state: \(state))"))
            #expect(ended.contains("re-run the Controller Job step"))
            #expect(ended.contains("/scratch/me/ws/runs/steerlab-serverd.7.out"))
        }

        let absent = ClusterProvisioner.controllerJobDiagnosis(
            jobID: "7", probe: .absent, site: site)
        #expect(absent.contains("no longer in Slurm's queue"))
        #expect(absent.contains("re-run the Controller Job step"))

        let unavailable = ClusterProvisioner.controllerJobDiagnosis(
            jobID: "7", probe: .unavailable, site: site)
        #expect(unavailable.contains("could not be checked"))
        #expect(unavailable.contains("squeue -j 7"))

        // Without a declared workspace root the log hint stays useful.
        let bare = slurmSite(topology: .daemonInJob, storageRoots: [:])
        let noWorkspace = ClusterProvisioner.controllerJobDiagnosis(
            jobID: "7", probe: .absent, site: bare)
        #expect(noWorkspace.contains("steerlab-serverd.7.out"))
    }

    @Test func connectWithoutAHandlerFailsInsteadOfPretending() async {
        let provisioner = ClusterProvisioner(runner: ScriptedProvisionRunner(), site: slurmSite())
        await provisioner.runConnect()
        guard case .failed = provisioner.record(for: .connect).status else {
            Issue.record("expected .failed without a handler")
            return
        }
    }

    @Test func capabilitySummaryNamesEngineDevicesAndFeatureFlags() throws {
        let capabilities = try JSONDecoder().decode(
            ClusterCapabilities.self,
            from: Data(
                """
                {"engine":"pytorch","deviceInventory":["cuda:0 H100 80GB"],
                 "housekeeping":true}
                """.utf8))
        #expect(
            ClusterProvisioner.capabilitySummary(capabilities)
                == "engine pytorch · devices: cuda:0 H100 80GB · housekeeping ✓ · preflight —")
        let bare = try JSONDecoder().decode(
            ClusterCapabilities.self, from: Data("{}".utf8))
        #expect(
            ClusterProvisioner.capabilitySummary(bare)
                == "engine unknown · housekeeping — · preflight —")
    }

    // MARK: Shell quoting

    @Test func shellQuotingLeavesSafeWordsBareAndQuotesTheRest() {
        #expect(ClusterProvisioner.shellQuoted("~/steerlab") == "~/steerlab")
        #expect(ClusterProvisioner.shellQuoted("gpu:A100:1") == "gpu:A100:1")
        #expect(ClusterProvisioner.shellQuoted("$HOME/envs") == "$HOME/envs")
        #expect(
            ClusterProvisioner.shellQuoted("s|@PARTITION@|batch|g")
                == "'s|@PARTITION@|batch|g'")
        #expect(ClusterProvisioner.shellQuoted("two words") == "'two words'")
        #expect(ClusterProvisioner.shellQuoted("don't") == "'don'\\''t'")
    }

    /// A shared site file's storage root is username-agnostic on purpose —
    /// `/scratch/${USER:-$(id -un)}/steerlab-workspace` is bootstrap.sh's own
    /// default shape — and single-quoting it froze the placeholder literal:
    /// the first live cluster start ran
    /// `mkdir -p '/scratch/${USER:-$(id -un)}/…'` and died with "Permission
    /// denied" creating a directory literally named `${USER:-$(id -un)}`
    /// (2026-08-21). Such words are double-quoted so the REMOTE shell expands
    /// them; words carrying their own quoting characters keep the literal
    /// single-quote path.
    @Test func remoteExpansionSyntaxIsDoubleQuotedNotFrozenLiteral() {
        #expect(
            ClusterProvisioner.shellQuoted(
                "/scratch/${USER:-$(id -un)}/steerlab-workspace")
                == "\"/scratch/${USER:-$(id -un)}/steerlab-workspace\"")
        #expect(ClusterProvisioner.shellQuoted("/scratch/$USER/ws")
            == "/scratch/$USER/ws")  // already bare-safe, unchanged
        // A `$`-carrying word with characters outside the expansion set is
        // NOT the expandable case — sed programs stay literal.
        #expect(
            ClusterProvisioner.shellQuoted("s|@X@|$HOME/envs|g")
                == "'s|@X@|$HOME/envs|g'")
        // No `$` at all: plain unsafe words keep the single-quote path even
        // when their characters are all in the expansion set.
        #expect(ClusterProvisioner.shellQuoted("two words") == "'two words'")
    }

    @Test func selectingASiteKeepsMaterializationAtItsDeclaredDefault() {
        // Regression (WP5 Step 9 flag): this reset once wrote the stale Step 6
        // opt-in literal and silently defeated default-on materialization the
        // moment any site was picked.
        let provisioner = ClusterProvisioner(
            runner: ScriptedProvisionRunner(), site: slurmSite())
        provisioner.bootstrapMaterializeEnvironment = false
        provisioner.selectSite(slurmSite(topology: .daemonInJob, account: "other"))
        #expect(
            provisioner.bootstrapMaterializeEnvironment
                == ClusterProvisioner.materializesEnvironmentByDefault)
        #expect(ClusterProvisioner.materializesEnvironmentByDefault)
    }
}
