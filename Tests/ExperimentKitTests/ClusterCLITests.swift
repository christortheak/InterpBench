import Foundation
import Testing

@testable import ExperimentKit

// =============================================================================
// The `cluster` CLI surface (CLUSTER-CLI-LIFECYCLE-PLAN Phase C).
//
// The plan's Phase C acceptance gate is that a coding agent can determine the
// exact next action from JSON alone, including human-auth, approval, pending,
// repairable, and ready states. Everything here runs against fakes — no ssh, no
// sockets, no Keychain, no process spawn — because the parsing and
// serialization deliberately live in ExperimentKit rather than in the binary.
// =============================================================================

// MARK: - Fakes

private actor FakeCLIShell: ClusterShellRunner {
    private struct Rule {
        var match: @Sendable ([String]) -> Bool
        var result: ClusterShellResult
    }

    private var rules: [Rule] = []
    private var calls: [[String]] = []

    func on(_ needle: String, exit: Int32 = 0, lines: [String] = []) {
        rules.insert(
            Rule(
                match: { $0.joined(separator: " ").contains(needle) },
                result: ClusterShellResult(exitCode: exit, lines: lines)),
            at: 0)
    }

    func run(_ argv: [String]) async -> ClusterShellResult {
        calls.append(argv)
        for rule in rules where rule.match(argv) { return rule.result }
        return ClusterShellResult(exitCode: 1)
    }

    func joinedCalls() -> [String] { calls.map { $0.joined(separator: " ") } }
    func callCount(containing needle: String) -> Int {
        calls.filter { $0.joined(separator: " ").contains(needle) }.count
    }
}

/// Holds a KNOWN token so every redaction assertion has something to hunt for.
private final class FakeCLISecrets: ClusterSecretStore, @unchecked Sendable {
    // @unchecked Sendable is justified: mutated only from the serialized test
    // body and one task chain, and it never escapes a test.
    private var storage: [String: String]
    /// The two reads are counted SEPARATELY so a presence claim that secretly
    /// reads the value is a failing assertion, not a silent regression: on the
    /// real Keychain the data read is the one that raises a password prompt.
    private(set) var dataReadCount = 0
    private(set) var presenceReadCount = 0
    init(_ storage: [String: String] = [:]) { self.storage = storage }
    func token(forKey key: String) -> String? {
        dataReadCount += 1
        return storage[key]
    }
    func hasToken(forKey key: String) -> Bool {
        presenceReadCount += 1
        return storage[key] != nil
    }
    func store(_ token: String, forKey key: String) throws { storage[key] = token }
    func removeToken(forKey key: String) { storage[key] = nil }
}

private struct FakeCLIProbe: ClusterEndpointProbe {
    var reachable = true
    var mismatch = false
    var build: String? = "steerlab-server 0.1.0+46bb4f9a"

    func probe(baseURL: URL, token: String?) async -> ClusterEndpointProbeResult {
        if mismatch {
            return ClusterEndpointProbeResult(
                reachable: false, detail: "identity mismatch")
        }
        guard token != nil else {
            return ClusterEndpointProbeResult(reachable: false, authFailed: true)
        }
        guard reachable else {
            return ClusterEndpointProbeResult(reachable: false, detail: "no answer")
        }
        return ClusterEndpointProbeResult(
            reachable: true, serverBuild: build, serverRole: "controller",
            root: "/scratch/me/ws")
    }
}

private final class FakeCLITunnel: ClusterTunnelControlling, @unchecked Sendable {
    var observation: ClusterTunnelObservation
    private(set) var openCount = 0
    private(set) var closeCount = 0

    init(observation: ClusterTunnelObservation = .absent) {
        self.observation = observation
    }

    func observe(
        site: ClusterSiteProfile, persistedPort: Int?, targetHost: String?
    ) async -> ClusterTunnelObservation { observation }

    func open(
        site: ClusterSiteProfile, targetHost: String, persistedPort: Int?
    ) async -> ClusterTunnelOpenOutcome {
        openCount += 1
        let port = persistedPort ?? 8712
        observation = .up(localPort: port)
        return ClusterTunnelOpenOutcome(
            observation: observation, message: "forward installed",
            forwardIdentity: "fake://\(targetHost):\(port)", changed: true)
    }

    func close(
        site: ClusterSiteProfile, localPort: Int, targetHost: String
    ) async -> String? {
        closeCount += 1
        observation = .absent
        return nil
    }
}

private final class FakeCLILauncher: ClusterAuthenticationLauncher, @unchecked Sendable {
    private(set) var openCount = 0
    func authenticationCommand(for site: ClusterSiteProfile) -> String? {
        ClusterTunnel.authenticationCommand(for: site)
    }
    func openAuthenticationTerminal(for site: ClusterSiteProfile) async -> Bool {
        openCount += 1
        return true
    }
}

private final class FakeCLILogStreamer: ClusterLogStreamer, @unchecked Sendable {
    var lines: [String] = ["controller up", "listening"]
    private(set) var streamCount = 0

    func stream(
        _ argv: [String], onLine: @escaping @Sendable (String) -> Void
    ) async -> ClusterShellResult {
        streamCount += 1
        for line in lines { onLine(line) }
        return ClusterShellResult(exitCode: 0, lines: lines)
    }
}

// MARK: - Parsing

/// Flag parsing is a CONTRACT, not a convenience: an agent that mistypes a
/// permission flag must be refused loudly rather than quietly denied the
/// mutation it asked for.
struct ClusterCLIParsingTests {

    @Test func everyLifecycleTargetParsesInBothSpellings() throws {
        for target in ClusterLifecycleTarget.allCases {
            let kebab = try ClusterCLIParser.parse(
                ["ensure", "--site", "s", "--target", target.cliName])
            #expect(kebab.target == target)
            let camel = try ClusterCLIParser.parse(
                ["ensure", "--site", "s", "--target", target.rawValue])
            #expect(camel.target == target)
        }
    }

    @Test func aBadTargetIsAUsageErrorNamingTheLadder() throws {
        do {
            _ = try ClusterCLIParser.parse(
                ["ensure", "--site", "s", "--target", "sideways"])
            Issue.record("expected a refusal")
        } catch let error as ClusterCLIError {
            #expect(error.code == "invalidTarget")
            #expect(error.exitCode == 64)
            #expect(error.repairAction.contains("controller-running"))
        }
    }

    @Test func theDefaultTargetIsConnected() throws {
        #expect(try ClusterCLIParser.parse(["ensure", "--site", "s"]).target == .connected)
        #expect(try ClusterCLIParser.parse(["plan", "--site", "s"]).target == .connected)
    }

    @Test func eachMutationHasItsOwnFlagAndNoneIsImplied() throws {
        let none = try ClusterCLIParser.parse(["ensure", "--site", "s"])
        #expect(none.permissions.isEmpty)

        let push = try ClusterCLIParser.parse(["ensure", "--site", "s", "--allow-push"])
        #expect(push.permissions == .push)

        let all = try ClusterCLIParser.parse([
            "ensure", "--site", "s", "--allow-push", "--allow-bootstrap",
            "--allow-controller-start",
        ])
        #expect(all.permissions == .allMutations)
        // Opening the Terminal is separately authorized — every MUTATION
        // permission is not the same as permission to open a window (§4.3).
        #expect(!all.permissions.contains(.openAuthTerminal))

        // The plan spells it without `--allow-`; the derived spelling is
        // accepted too so the vocabulary's own `flags` rendering is never a
        // dead end.
        for spelling in ["--open-auth-terminal", "--allow-open-auth-terminal"] {
            let parsed = try ClusterCLIParser.parse(["ensure", "--site", "s", spelling])
            #expect(parsed.permissions == .openAuthTerminal)
        }
    }

    @Test func anUnknownVerbOrFlagIsAUsageErrorWithARepairAction() throws {
        for arguments in [["bogus"], ["sites"], ["--site", "s"]] {
            do {
                _ = try ClusterCLIParser.parse(arguments)
                Issue.record("expected a refusal for \(arguments)")
            } catch let error as ClusterCLIError {
                #expect(error.code == "unknownVerb")
                #expect(error.exitCode == 64)
            }
        }
        do {
            _ = try ClusterCLIParser.parse(["ensure", "--site", "s", "--allow-everything"])
            Issue.record("expected a refusal")
        } catch let error as ClusterCLIError {
            #expect(error.code == "unknownFlag")
            #expect(error.exitCode == 64)
        }
        // A flag that belongs to ANOTHER verb is still unknown here — silently
        // ignoring it would let `status --follow` look like it did something.
        do {
            _ = try ClusterCLIParser.parse(["status", "--site", "s", "--follow"])
            Issue.record("expected a refusal")
        } catch let error as ClusterCLIError {
            #expect(error.code == "unknownFlag")
        }
    }

    @Test func requiredArgumentsAreEnforcedBeforeAnythingRuns() throws {
        func expect(_ arguments: [String], code: String) {
            do {
                _ = try ClusterCLIParser.parse(arguments)
                Issue.record("expected a refusal for \(arguments)")
            } catch let error as ClusterCLIError {
                #expect(error.code == code, "\(arguments) gave \(error.code)")
                #expect(error.exitCode == 64)
            } catch {
                Issue.record("unexpected error \(error)")
            }
        }
        expect(["status"], code: "missingSite")
        expect(["sites", "export", "--site", "s"], code: "missingArgument")
        expect(["sites", "import"], code: "missingArgument")
        expect(["bootstrap", "apply", "--site", "s"], code: "missingArgument")
        expect(["controller", "adopt", "--site", "s"], code: "missingArgument")
        expect(["ensure", "--site"], code: "missingFlagValue")
        // The two registry-wide verbs are exempt.
        #expect(try ClusterCLIParser.parse(["sites", "list"]).siteReference == nil)
        // Every controller verb that READS a job may name one; only `start`,
        // which mints one, may not.
        for verb in ["status", "logs", "stop", "adopt"] {
            let parsed = try ClusterCLIParser.parse(
                ["controller", verb, "--site", "s", "--job-id", "4242"])
            #expect(parsed.jobID == "4242")
        }
        expect(
            ["controller", "start", "--site", "s", "--job-id", "1"],
            code: "unknownFlag")
    }

    @Test func twoWordVerbsAreNeverShadowedByTheirFirstWord() throws {
        #expect(try ClusterCLIParser.parse(["sites", "list"]).verb == .sitesList)
        #expect(
            try ClusterCLIParser.parse(["controller", "adopt", "--site", "s",
                                        "--job-id", "1"]).verb == .controllerAdopt)
        #expect(try ClusterCLIParser.parse(["status", "--site", "s"]).verb == .status)
    }

    @Test func provisioningOverridesLayerOntoTheSitesOwnDefaults() throws {
        let parsed = try ClusterCLIParser.parse([
            "push", "--site", "s", "--remote-repo", "/scratch/me/steerlab",
            "--squeue", "gsqueue",
        ])
        #expect(parsed.overrides.remoteRepoPath == "/scratch/me/steerlab")
        let profile = ClusterSiteProfile(
            name: "T",
            transport: .ssh(host: "h", proxyJump: nil, remotePort: 8080, vpnExpected: false),
            topology: .daemonInJob)
        let configuration = parsed.overrides.resolved(for: profile)
        #expect(configuration.remoteRepoPath == "/scratch/me/steerlab")
        #expect(configuration.squeueCommand == "gsqueue")
        // Unset overrides keep the site's defaults rather than blanking them.
        #expect(configuration.bootstrapJobCPUs == 4)
    }

    /// **WP5 Step 7 — materialization is the default, and opting out has to be
    /// typed.** A caller who says nothing installs the environment rendered
    /// from the site profile; `--no-materialize-env` is the one spelling of
    /// "let bootstrap.sh synthesize its own", and `--materialize-env` survives
    /// as an explicit way to name the default (Step 6 scripts keep working).
    @Test func materializeEnvIsTheDefaultAndOptingOutMustBeTyped() throws {
        let profile = ClusterSiteProfile(
            name: "T",
            transport: .ssh(host: "h", proxyJump: nil, remotePort: 8080, vpnExpected: false),
            topology: .daemonInJob)

        let plain = try ClusterCLIParser.parse(["bootstrap", "plan", "--site", "s"])
        #expect(plain.overrides.materializeEnvironmentFile == nil)
        #expect(plain.overrides.resolved(for: profile).materializeEnvironmentFile == true)

        let opted = try ClusterCLIParser.parse([
            "bootstrap", "plan", "--site", "s", "--materialize-env",
        ])
        #expect(opted.overrides.materializeEnvironmentFile == true)
        #expect(opted.overrides.resolved(for: profile).materializeEnvironmentFile == true)

        let refused = try ClusterCLIParser.parse([
            "bootstrap", "plan", "--site", "s", "--no-materialize-env",
        ])
        #expect(refused.overrides.materializeEnvironmentFile == false)
        #expect(refused.overrides.resolved(for: profile).materializeEnvironmentFile == false)

        // A verb that runs no remote command rejects both spellings rather than
        // ignoring them — `preview` renders the SAVED profile and has no
        // bootstrap to gate.
        #expect(throws: ClusterCLIError.self) {
            try ClusterCLIParser.parse(["preview", "--site", "s", "--materialize-env"])
        }
        #expect(throws: ClusterCLIError.self) {
            try ClusterCLIParser.parse(["preview", "--site", "s", "--no-materialize-env"])
        }
        #expect(throws: ClusterCLIError.self) {
            try ClusterCLIParser.parse(["sites", "list", "--no-materialize-env"])
        }
    }

    // MARK: preview (WP5 §3.3)

    @Test func previewParsesEveryJobClassInBothSpellings() throws {
        // No --job-class previews every class; the runner turns nil into
        // `allCases`, which is what the committed header golden covers.
        #expect(try ClusterCLIParser.parse(["preview", "--site", "s"]).jobClass == nil)
        for jobClass in ClusterEnvironmentRenderer.JobClass.allCases {
            for spelling in [jobClass.rawValue, jobClass.cliName, jobClass.cliName.uppercased()] {
                let parsed = try ClusterCLIParser.parse(
                    ["preview", "--site", "s", "--job-class", spelling])
                #expect(parsed.jobClass == jobClass, "\(spelling) did not parse")
            }
        }
        #expect(try ClusterCLIParser.parse(["preview", "--site", "s", "--json"]).json)
        // Reading what will run is never a mutation: the runner may skip the
        // per-site lock for it.
        #expect(ClusterCLIVerb.preview.isReadOnly)
    }

    @Test func aBadJobClassIsAUsageErrorNamingTheClasses() throws {
        do {
            _ = try ClusterCLIParser.parse(
                ["preview", "--site", "s", "--job-class", "sideways"])
            Issue.record("expected a refusal")
        } catch let error as ClusterCLIError {
            #expect(error.code == "invalidJobClass")
            #expect(error.exitCode == 64)
            #expect(error.repairAction.contains("gpu-session"))
            #expect(error.repairAction.contains("study"))
        }
    }

    @Test func previewRefusesAProvisioningOverrideAndRequiresASite() throws {
        // It renders the SAVED profile, so an override would show an
        // environment the stored site does not imply.
        do {
            _ = try ClusterCLIParser.parse(
                ["preview", "--site", "s", "--env-prefix", "/tmp/x"])
            Issue.record("expected a refusal")
        } catch let error as ClusterCLIError {
            #expect(error.code == "unknownFlag")
        }
        do {
            _ = try ClusterCLIParser.parse(["preview"])
            Issue.record("expected a refusal")
        } catch let error as ClusterCLIError {
            #expect(error.code == "missingSite")
        }
    }

    @Test func theWatchVerbIsAbsentFromThisCut() {
        // §0.3: polling ensure/status with retryAfterSeconds IS the v1
        // contract. If `watch` ever lands, this test should be the thing that
        // makes someone decide deliberately.
        #expect(ClusterCLIVerb(rawValue: "watch") == nil)
    }

    @Test func siteAndURLAreMutuallyExclusive() throws {
        #expect(
            try ClusterRemoteSiteResolver.choose(site: "uga", url: nil) == .site("uga"))
        #expect(
            try ClusterRemoteSiteResolver.choose(site: nil, url: "http://x")
                == .url("http://x"))
        #expect(try ClusterRemoteSiteResolver.choose(site: nil, url: nil) == .url(nil))
        do {
            _ = try ClusterRemoteSiteResolver.choose(site: "uga", url: "http://x")
            Issue.record("expected a refusal")
        } catch let error as ClusterCLIError {
            #expect(error.code == "siteAndURLAreMutuallyExclusive")
            #expect(error.exitCode == 64)
        }
    }
}

// MARK: - Envelope

/// The machine protocol (§6.5). The JSON state is authoritative and the exit
/// code is a convenience for shell callers — so they must be derived from the
/// same place, and the required keys must actually be there.
struct ClusterCLIEnvelopeTests {

    @Test func exitCodesFollowTheStateVocabularyExactly() {
        let expected: [ClusterLifecycleState: Int32] = [
            .ready: 0, .planned: 0, .running: 0,
            .needsHumanAuthentication: 10, .needsApproval: 11, .pending: 12,
            .degraded: 13, .blocked: 64, .failed: 70,
        ]
        for state in ClusterLifecycleState.allCases {
            let envelope = ClusterCLIEnvelope(
                verb: "cluster ensure", state: state, message: "m")
            #expect(envelope.exitCode == expected[state])
            // The envelope's own code and the vocabulary's can never diverge.
            #expect(envelope.exitCode == state.exitCode)
        }
    }

    @Test func theRequiredCommonFieldsAreAlwaysPresent() throws {
        let envelope = ClusterCLIEnvelope(
            verb: "cluster status", state: .ready, message: "m")
        let object = try Self.decode(envelope)
        for key in ["schemaVersion", "verb", "state", "changed", "observedAt", "message"] {
            #expect(object[key] != nil, "missing required key \(key)")
        }
        #expect(object["schemaVersion"] as? Int == 1)
        // Absent optionals are OMITTED rather than emitted as null: a caller
        // testing for a key gets a straight answer.
        #expect(object["nextAction"] == nil)
        #expect(object["endpoint"] == nil)
    }

    @Test func aLifecycleResultProjectsEveryDocumentedField() throws {
        let observed = ClusterObservedState(
            siteID: "uga", siteName: "Example HPC",
            controlMaster: .alive, tunnel: .up(localPort: 8718),
            bearerToken: .accepted)
        let plan = ClusterLifecyclePlan(
            target: .connected,
            transitions: [
                ClusterLifecycleTransition(
                    step: .pushCode, reason: "the payload is stale",
                    gating: ClusterTransitionGating(requiredPermission: .push))
            ])
        let result = ClusterLifecycleResult(
            operationID: "cluster-1", siteID: "uga", siteName: "Example HPC",
            target: .connected, state: .needsApproval, step: .pushCode,
            changed: false, message: "approval required",
            nextAction: .init(
                verb: "cluster ensure", missingPermissionFlags: ["--allow-push"]),
            endpoint: "http://127.0.0.1:8718", tokenAvailable: true,
            tokenSource: "keychain", serverBuild: "steerlab-server 0.1.0",
            observed: observed, plan: plan)
        let envelope = ClusterCLIEnvelope.lifecycle(verb: .ensure, result: result)
        let object = try Self.decode(envelope)

        #expect(object["operationID"] as? String == "cluster-1")
        #expect(object["siteID"] as? String == "uga")
        #expect(object["siteName"] as? String == "Example HPC")
        #expect(object["target"] as? String == "connected")
        #expect(object["state"] as? String == "needsApproval")
        #expect(object["step"] as? String == "pushCode")
        #expect(object["endpoint"] as? String == "http://127.0.0.1:8718")
        #expect(object["tokenAvailable"] as? Bool == true)
        #expect(object["tokenSource"] as? String == "keychain")
        #expect(envelope.exitCode == 11)

        // The next action is machine-readable: no prose parsing required.
        let next = try #require(object["nextAction"] as? [String: Any])
        #expect(next["verb"] as? String == "cluster ensure")
        #expect(next["missingPermissionFlags"] as? [String] == ["--allow-push"])

        // Every layer is reported independently — never one `connected` Bool.
        let layers = try #require(object["layers"] as? [[String: Any]])
        #expect(layers.count == observed.layerSummaries.count)
        #expect(layers.contains { $0["layer"] as? String == "controlMaster" })
        #expect(layers.contains { $0["layer"] as? String == "bearerToken" })

        let steps = try #require(object["plan"] as? [[String: Any]])
        #expect(steps.first?["step"] as? String == "pushCode")
        #expect(steps.first?["requiredPermissionFlags"] as? [String] == ["--allow-push"])
    }

    @Test func commandsAreArgvArraysNotShellStrings() throws {
        var envelope = ClusterCLIEnvelope(
            verb: "cluster auth command", state: .ready, message: "m")
        envelope.command = ["ssh", "-o", "ControlMaster=auto", "login.test"]
        let object = try Self.decode(envelope)
        #expect(object["command"] as? [String] != nil)
    }

    @Test func exactlyOneJSONDocumentIsProduced() throws {
        let text = try ClusterCLIEnvelope(
            verb: "cluster status", state: .ready, message: "m").jsonText()
        #expect(text.hasSuffix("\n"))
        // One document: trimming the trailing newline leaves a single parse.
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        #expect(trimmed.hasPrefix("{"))
        #expect(trimmed.hasSuffix("}"))
        #expect(!trimmed.dropLast().contains("}\n{"))
        // No ANSI, ever (§6.5).
        #expect(!text.contains("\u{1B}"))
    }

    @Test func redactionRemovesUsernamesAndHomePathsButKeepsTheDiagnosis() {
        let text = "rsync to someone@login.slurm.example.edu:/home/someone/steerlab"
        let redacted = ClusterCLIRedaction.redact(text)
        #expect(!redacted.contains("someone"))
        #expect(redacted.contains("login.slurm.example.edu"))
        #expect(redacted.contains("/home/<user>/steerlab"))
        #expect(ClusterCLIRedaction.redact("/scratch/me/ws").contains("/scratch/<user>"))
    }

    private static func decode(_ envelope: ClusterCLIEnvelope) throws -> [String: Any] {
        let data = Data(try envelope.jsonText().utf8)
        return try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}

// MARK: - Runner

/// Verb dispatch against fakes. The plan's acceptance gate: the exact next
/// action is determinable from the machine output alone.
struct ClusterCLIRunnerTests {

    /// The token every leak assertion hunts for.
    private static let secretToken = "sk-CLI-SECRET-never-print-77"

    /// Every key in a JSON document, lowercased and recursive — so the
    /// "no credential-shaped field" invariant is checked structurally rather
    /// than by grepping prose.
    static func allKeys(in json: String) -> Set<String> {
        func walk(_ value: Any, into keys: inout Set<String>) {
            if let object = value as? [String: Any] {
                for (key, nested) in object {
                    keys.insert(key.lowercased())
                    walk(nested, into: &keys)
                }
            } else if let array = value as? [Any] {
                for element in array { walk(element, into: &keys) }
            }
        }
        var keys: Set<String> = []
        if let parsed = try? JSONSerialization.jsonObject(with: Data(json.utf8)) {
            walk(parsed, into: &keys)
        }
        return keys
    }

    private struct Harness {
        var runner: ClusterCLIRunner
        var repository: ClusterSiteRepository
        var operationStore: ClusterOperationStore
        var shell: FakeCLIShell
        var secrets: FakeCLISecrets
        var tunnel: FakeCLITunnel
        var launcher: FakeCLILauncher
        var streamer: FakeCLILogStreamer
        var root: URL
        var payloadRoot: URL
        var manifestBytes: String
        /// What `--activate-in-app` wrote, instead of touching UserDefaults.
        var activations: ActivationRecorder
    }

    private final class ActivationRecorder: @unchecked Sendable {
        private(set) var endpoints: [String] = []
        func record(_ endpoint: String) { endpoints.append(endpoint) }
    }

    private func temporaryDirectory(_ name: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appending(components: "steerlab-cluster-cli", "\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func slurmProfile() -> ClusterSiteProfile {
        ClusterSiteProfile(
            name: "Test Cluster",
            transport: .ssh(
                host: "login.test", proxyJump: nil, remotePort: 8080, vpnExpected: false),
            topology: .daemonInJob,
            scheduler: .slurm(
                ClusterSiteProfile.SlurmSiteData(
                    partitions: [.init(name: "batch", maxWalltimeHours: 168)],
                    defaultPartition: "batch")),
            constraints: ClusterSiteProfile.SiteConstraints(
                storageRoots: ["workspace": "/scratch/me/ws"]))
    }

    private func makeHarness(
        _ name: String,
        storedToken: String? = nil,
        tunnel: ClusterTunnelObservation = .absent,
        probe: FakeCLIProbe = FakeCLIProbe(),
        now: Date = Date(timeIntervalSince1970: 1_000)
    ) throws -> Harness {
        let root = try temporaryDirectory(name)
        let payloadRoot = root.appending(component: "payload")
        try FileManager.default.createDirectory(
            at: payloadRoot, withIntermediateDirectories: true)
        let manifestBytes = #"{"schemaVersion":1,"files":[]}"#
        try Data(manifestBytes.utf8).write(
            to: payloadRoot.appending(
                component: ClusterProvisioner.deploymentManifestFileName))

        let repository = ClusterSiteRepository(
            fileURL: root.appending(component: "cluster-sites.json"),
            legacyRegistryData: { nil })
        _ = try repository.upsert(profile: slurmProfile())
        let operationStore = ClusterOperationStore(
            rootDirectory: root.appending(component: "cluster-operations"))
        let shell = FakeCLIShell()
        let secrets = FakeCLISecrets(storedToken.map { ["login.test:8080": $0] } ?? [:])
        let tunnelController = FakeCLITunnel(observation: tunnel)
        let launcher = FakeCLILauncher()
        let streamer = FakeCLILogStreamer()
        let activations = ActivationRecorder()

        let runner = ClusterCLIRunner(
            repository: repository, operationStore: operationStore, shell: shell,
            secrets: secrets, tunnel: tunnelController, endpointProbe: probe,
            authenticationLauncher: launcher, logStreamer: streamer,
            now: { now }, activationSink: { activations.record($0) },
            // Bootstrap polling is a cadence of separate short commands; a
            // test must not wait it out.
            bootstrapPollDelay: .zero, bootstrapPollLimit: 3)

        return Harness(
            runner: runner, repository: repository, operationStore: operationStore,
            shell: shell, secrets: secrets, tunnel: tunnelController,
            launcher: launcher, streamer: streamer, root: root,
            payloadRoot: payloadRoot, manifestBytes: manifestBytes,
            activations: activations)
    }

    /// Every override the CLI would pass for this harness's payload.
    private func overrides(_ harness: Harness) -> ClusterCLIOverrides {
        var overrides = ClusterCLIOverrides()
        overrides.localPayloadPath = harness.payloadRoot.path
        overrides.bootstrapPartition = "batch"
        return overrides
    }

    private func invocation(
        _ harness: Harness, _ verb: ClusterCLIVerb,
        permissions: ClusterLifecyclePermissions = [],
        jobID: String? = nil, planHash: String? = nil, outPath: String? = nil,
        jobClass: ClusterEnvironmentRenderer.JobClass? = nil,
        positional: String? = nil, follow: Bool = false, dryRun: Bool = false,
        redact: Bool = false, activateInApp: Bool = false,
        renderOnly: Bool = false
    ) -> ClusterCLIInvocation {
        ClusterCLIInvocation(
            verb: verb, siteReference: "test-cluster", permissions: permissions,
            redact: redact, dryRun: dryRun, follow: follow,
            activateInApp: activateInApp, renderOnly: renderOnly,
            planHash: planHash, jobID: jobID,
            outPath: outPath, jobClass: jobClass, positional: positional,
            overrides: overrides(harness))
    }

    private func scriptHealthySite(
        _ harness: Harness, controllerState: String = "RUNNING|None",
        daemonHost: String = "c4-12"
    ) async {
        await harness.shell.on("-O check", exit: 0)
        // The command probe behind `-O check`: a live master runs `true`.
        await harness.shell.on("ConnectTimeout", exit: 0)
        await harness.shell.on(
            ClusterProvisioner.deploymentManifestFileName, exit: 0,
            lines: [harness.manifestBytes])
        await harness.shell.on(
            "STEERLAB_PREFIX", exit: 0, lines: ["/home/me/envs/steerlab"])
        await harness.shell.on("squeue", exit: 0, lines: [controllerState])
        await harness.shell.on("serverd.host", exit: 0, lines: [daemonHost])
        await harness.shell.on(
            "profile validate", exit: 0, lines: ["OK   profile: cluster/login/slurm"])
        await harness.shell.on(".steerlab-token", exit: 0, lines: [Self.secretToken])
        // The rendered controller script vs the deployed template (§1 field
        // report, 2026-08-20). A healthy site's artifact is the current
        // template's child, so the layer reads `current` and no advisory
        // fires — an advisory channel that always speaks is one nobody reads.
        let digest = String(repeating: "a", count: 64)
        await harness.shell.on(
            "rendered-exists", exit: 0,
            lines: [
                "rendered-exists: yes",
                "rendered-stamp: template=controller-job.sbatch.template "
                    + "sha256=\(digest) renderedAt=2026-08-20T09:00:00Z "
                    + "source=52a7176",
                "rendered-chain: template-\(digest)",
                "template-sha256: \(digest)",
            ])
    }

    private func seedControllerJob(_ harness: Harness, jobID: String = "4242") throws {
        var record = ClusterOperationRecord(
            operationID: "op-seed", siteID: "test-cluster", siteProfileHash: "seed",
            target: .controllerRunning, state: .pending,
            startedAt: Date(timeIntervalSince1970: 900))
        record.controllerJobID = jobID
        try harness.operationStore.save(record)
    }

    // MARK: sites

    @Test func sitesListReportsPresenceNotSecrets() async throws {
        let harness = try makeHarness("sites-list", storedToken: Self.secretToken)
        let outcome = await harness.runner.run(
            ClusterCLIInvocation(verb: .sitesList))
        #expect(outcome.exitCode == 0)
        let sites = try #require(outcome.envelope.sites)
        #expect(sites.count == 1)
        #expect(sites[0].id == "test-cluster")
        #expect(sites[0].transport == "ssh://login.test:8080")
        #expect(sites[0].tokenAvailable)
        // Presence, never the value — in the JSON or the human rendering.
        let json = try outcome.envelope.jsonText()
        #expect(!json.contains(Self.secretToken))
        #expect(!ClusterCLIRenderer.humanText(outcome.envelope).contains(Self.secretToken))
    }

    /// The keychain-prompt hazard, as a test rather than a warning in a doc.
    ///
    /// `tokenAvailable` on a read-only listing verb is a PRESENCE claim. On the
    /// real Keychain, answering it by reading the secret is an ACL-gated access
    /// that a freshly installed binary — a new binary identity — cannot make
    /// without a system password dialog, and an unattended agent waits at that
    /// dialog forever. So: presence reads, zero data reads.
    @Test func sitesListAnswersTokenPresenceWithoutReadingTheSecret() async throws {
        let harness = try makeHarness("sites-list-presence", storedToken: Self.secretToken)
        let outcome = await harness.runner.run(
            ClusterCLIInvocation(verb: .sitesList))
        #expect(outcome.exitCode == 0)
        let sites = try #require(outcome.envelope.sites)
        #expect(sites[0].tokenAvailable)
        #expect(harness.secrets.presenceReadCount == 1)
        #expect(harness.secrets.dataReadCount == 0)

        // `sites show` reports the same field from the same offline record.
        let shown = await harness.runner.run(invocation(harness, .sitesShow))
        #expect(shown.exitCode == 0)
        #expect(shown.envelope.tokenAvailable == true)
        #expect(harness.secrets.dataReadCount == 0)
    }

    @Test func exportWritesTheProfileAndImportRoundTripsIt() async throws {
        let harness = try makeHarness("sites-export", storedToken: Self.secretToken)
        let out = harness.root.appending(component: "profile.json")
        let exported = await harness.runner.run(
            invocation(harness, .sitesExport, outPath: out.path))
        #expect(exported.exitCode == 0)
        #expect(exported.envelope.outputPath == out.path)

        // No credentials and no ephemeral tunnel detail in the shared document.
        let text = try String(contentsOf: out, encoding: .utf8)
        #expect(!text.contains(Self.secretToken))
        #expect(!text.contains("127.0.0.1"))
        // Nothing token-shaped survives except the site's DECLARED token-file
        // PATH (schema 2, `environment.tokenFilePath`) — an indirection the
        // renderer dereferences on the cluster, never a value here.
        #expect(
            !ClusterProfileTokenScrub.withoutDeclaredTokenPath(text).lowercased()
                .contains("token"))

        // Re-importing dedupes by canonical identity rather than forking.
        let imported = await harness.runner.run(
            ClusterCLIInvocation(verb: .sitesImport, positional: out.path))
        #expect(imported.exitCode == 0)
        #expect(imported.envelope.siteID == "test-cluster")
        #expect(try harness.repository.sites().count == 1)
    }

    // MARK: preview (WP5 §3.3 — the surface that makes the rest reviewable)

    /// The agent path (WP0): a stable JSON document carrying the complete
    /// environment, the headers per job class, the GPU vocabulary, the
    /// unresolved facts, which default set applied, and the profile's schema —
    /// produced without touching the cluster, because a preview that needed a
    /// live cluster could not be read BEFORE anything runs.
    @Test func previewRendersTheWholeEnvironmentWithoutTouchingTheCluster() async throws {
        let harness = try makeHarness("preview", storedToken: Self.secretToken)
        let outcome = await harness.runner.run(invocation(harness, .preview))
        #expect(outcome.exitCode == 0)
        #expect(outcome.envelope.state == "ready")
        #expect(outcome.envelope.changed == false)
        // Read-only means read-only: not one command ran.
        #expect(await harness.shell.joinedCalls().isEmpty)

        let json = try outcome.envelope.jsonText()
        guard
            let object = try JSONSerialization.jsonObject(with: Data(json.utf8))
                as? [String: Any],
            let preview = object["preview"] as? [String: Any]
        else {
            Issue.record("the envelope carried no preview object")
            return
        }
        // The documented schema shape.
        for key in [
            "siteName", "schemaVersion", "defaultSet", "defaultSetSummary", "envFile",
            "headers", "schedulerCommands", "gpuVocabulary", "unresolvedFacts",
        ] {
            #expect(preview[key] != nil, "preview is missing \(key)")
        }
        #expect(preview["defaultSet"] as? String == "neutralV2")
        #expect(preview["schemaVersion"] as? Int == ClusterSiteProfile.currentSchemaVersion)
        // The envelope's own schemaVersion is the machine protocol's; the
        // preview's is the PROFILE's. They are different numbers on purpose.
        #expect(object["schemaVersion"] as? Int == ClusterLifecycleResult.schemaVersion)

        let headers = preview["headers"] as? [[String: Any]] ?? []
        #expect(
            headers.compactMap { $0["jobClass"] as? String }
                == ClusterEnvironmentRenderer.JobClass.allCases.map(\.rawValue))
        #expect(headers.allSatisfy { $0["lines"] is [Any] })
        let facts = preview["unresolvedFacts"] as? [[String: Any]] ?? []
        #expect(facts.allSatisfy { $0["key"] is String && $0["detail"] is String })
        // Every pane is the renderer's own bytes — the preview adds no facts.
        let site = try #require(try harness.repository.resolve(reference: "test-cluster"))
        #expect(
            preview["envFile"] as? String
                == ClusterEnvironmentRenderer.renderEnvFile(site.profile))
        // …and the secret stays an indirection even with a token in the store.
        #expect(!json.contains(Self.secretToken))
        #expect((preview["envFile"] as? String)?.contains("$(cat ") == true)
    }

    @Test func previewNarrowsToOneJobClassOnRequest() async throws {
        let harness = try makeHarness("preview-class")
        let outcome = await harness.runner.run(
            invocation(harness, .preview, jobClass: .controller))
        #expect(outcome.envelope.preview?.headers.map(\.jobClass) == ["controller"])
        // The human view prints the panes verbatim, so an admin can copy them.
        let text = ClusterCLIRenderer.humanText(outcome.envelope)
        #expect(text.contains("--- environment file ---"))
        #expect(text.contains("--- unresolved facts"))
        #expect(text.contains("#SBATCH --job-name=steerlab-serverd"))
        #expect(!text.contains("#SBATCH --job-name=steerlab-gpu-session"))
        // Requirement 4: the reader is told which defaults are in play.
        #expect(text.contains("NEUTRAL defaults"))
    }

    @Test func anUnknownSiteRefusesWithARepairAction() async throws {
        let harness = try makeHarness("unknown-site")
        let outcome = await harness.runner.run(
            ClusterCLIInvocation(verb: .status, siteReference: "nope"))
        #expect(outcome.envelope.error?.code == "unknownSite")
        #expect(outcome.envelope.error?.repairAction.contains("cluster sites list") == true)
        #expect(outcome.exitCode == 70)
    }

    // MARK: status

    @Test func statusReportsEveryLayerIndependently() async throws {
        let harness = try makeHarness(
            "status", storedToken: Self.secretToken, tunnel: .up(localPort: 8712))
        await scriptHealthySite(harness)
        try seedControllerJob(harness)
        _ = try harness.repository.noteConnection(
            siteID: "test-cluster", endpoint: "http://127.0.0.1:8712",
            serverBuild: "steerlab-server 0.1.0")

        let outcome = await harness.runner.run(invocation(harness, .status))
        let layers = try #require(outcome.envelope.layers)
        let byName = Dictionary(
            uniqueKeysWithValues: layers.map { ($0.layer, $0.state) })
        #expect(byName["controlMaster"] == "alive")
        #expect(byName["controller"] == "running (4242)")
        #expect(byName["daemonHost"] == "current (c4-12)")
        #expect(byName["tunnel"] == "up (127.0.0.1:8712)")
        #expect(byName["serverHTTP"] == "reachable")
        #expect(byName["bearerToken"] == "accepted")
        // Read-only: nothing mutated, and approval is never the answer for a
        // question about what is true.
        #expect(!outcome.envelope.changed)
        #expect(outcome.envelope.state != ClusterLifecycleState.needsApproval.rawValue)
        #expect(byName["controllerScript"] == "current (aaaaaaaaaaaa)")
        #expect(outcome.envelope.advisories == nil)
        #expect(await harness.shell.callCount(containing: "/usr/bin/rsync") == 0)
        // "sbatch" is also a substring of the rendered artifact's FILE NAME,
        // which the read-only controller-script probe names — so the assertion
        // has to be about the SUBMISSION, not the substring.
        #expect(await harness.shell.callCount(containing: "&& sbatch ") == 0)
    }

    // MARK: auth — the human boundary (§4.1, release-blocking)

    @Test func authOpenIsIdempotentAndNeverSpawnsASecondWindow() async throws {
        let harness = try makeHarness("auth-open")
        // Cold: no master, so one Terminal opens.
        let first = await harness.runner.run(invocation(harness, .authOpen))
        #expect(first.envelope.state == ClusterLifecycleState.needsHumanAuthentication.rawValue)
        #expect(first.exitCode == 10)
        #expect(harness.launcher.openCount == 1)
        #expect(first.envelope.nextAction?.requiresHuman == true)
        #expect(first.envelope.retryAfterSeconds == 5)

        // Repeating inside the window REPORTS the open attempt instead of
        // spawning another window (§6.2).
        let second = await harness.runner.run(invocation(harness, .authOpen))
        #expect(harness.launcher.openCount == 1)
        #expect(second.exitCode == 10)
        #expect(second.envelope.message.contains("already opened"))
        #expect(!second.envelope.changed)

        // A live master returns ready WITHOUT opening anything.
        await harness.shell.on("-O check", exit: 0)
        await harness.shell.on("ConnectTimeout", exit: 0)
        let warm = await harness.runner.run(invocation(harness, .authOpen))
        #expect(warm.exitCode == 0)
        #expect(warm.envelope.state == ClusterLifecycleState.ready.rawValue)
        #expect(harness.launcher.openCount == 1)
    }

    @Test func authVerbsCarryNoCredentialFieldAndOnlyEverPrintACommand() async throws {
        let harness = try makeHarness("auth-command")
        let command = await harness.runner.run(invocation(harness, .authCommand))
        let argv = try #require(command.envelope.command)
        // An argv ARRAY, and the composition the user runs themselves.
        #expect(argv.first == "ssh")
        #expect(argv.contains("ControlMaster=auto"))
        #expect(argv.last == "login.test")
        // No credential-shaped FIELD exists anywhere in the document. (The
        // message may well say the word "password" — it says SteerLab never
        // sees one — so the invariant is about keys, not prose.)
        let keys = Self.allKeys(in: try command.envelope.jsonText())
        for forbidden in ["password", "passcode", "secret", "credential", "token"] {
            #expect(!keys.contains(forbidden), "a '\(forbidden)' field exists")
        }
        // The only token-shaped keys are presence and provenance.
        #expect(keys.filter { $0.contains("token") }.isSubset(
            of: ["tokenavailable", "tokensource"]))

        // auth status classifies the master rather than guessing.
        let cold = await harness.runner.run(invocation(harness, .authStatus))
        #expect(cold.exitCode == 10)
        await harness.shell.on("-O check", exit: 0)
        await harness.shell.on("ConnectTimeout", exit: 0)
        let warm = await harness.runner.run(invocation(harness, .authStatus))
        #expect(warm.exitCode == 0)
    }

    @Test func authCloseTargetsOnlySteerLabsOwnControlPath() async throws {
        let harness = try makeHarness("auth-close")
        await harness.shell.on("-O exit", exit: 0)
        let outcome = await harness.runner.run(invocation(harness, .authClose))
        #expect(outcome.exitCode == 0)
        let argv = try #require(outcome.envelope.command)
        #expect(argv.contains("-O"))
        #expect(argv.contains("exit"))
        // Scoped to SteerLab's control path: it cannot touch another session.
        #expect(argv.contains { $0.contains("steerlab-cm") })
    }

    // MARK: push / bootstrap

    @Test func pushDryRunShowsTheExactCommandAndRunsNothing() async throws {
        let harness = try makeHarness("push-dry")
        let outcome = await harness.runner.run(invocation(harness, .push, dryRun: true))
        #expect(outcome.envelope.state == ClusterLifecycleState.planned.rawValue)
        let argv = try #require(outcome.envelope.command)
        #expect(argv.first == ClusterProvisioner.rsyncExecutablePath)
        #expect(await harness.shell.callCount(containing: "/usr/bin/rsync") == 0)
        #expect(!outcome.envelope.changed)
    }

    @Test func bootstrapApplyRefusesAStalePlanHashBeforeAnythingRuns() async throws {
        let harness = try makeHarness("bootstrap-gate")
        await scriptHealthySite(harness)
        await harness.shell.on(
            "bootstrap", exit: 0,
            lines: [#"{"ok":true,"steps":{"condaDetect":"planned"},"envFile":"/e","prefix":"/p"}"#])
        // WP5 Step 7: apply materializes by default, so the rendered env file
        // is pushed before anything is submitted. Matched on the push heredoc's
        // delimiter, which only the push carries — the bootstrap argv now names
        // the rendered file's PATH, so a path needle would swallow it too.
        await harness.shell.on("STEERLAB_RENDERED_ENV_EOF", exit: 0)

        let planned = await harness.runner.run(invocation(harness, .bootstrapPlan))
        let planHash = try #require(planned.envelope.planHash)
        #expect(planned.envelope.state == ClusterLifecycleState.planned.rawValue)
        #expect(planned.envelope.nextAction?.detail?.contains(planHash) == true)
        // The review is DURABLE: a later process finds it in the store.
        #expect(harness.operationStore.lastBootstrapPlanHash(forSite: "test-cluster")
            == planHash)

        let realCallsBefore = await harness.shell.joinedCalls()
            .filter { $0.contains("bootstrap") && !$0.contains("--dry-run") }.count
        let stale = await harness.runner.run(
            invocation(harness, .bootstrapApply, planHash: "deadbeef"))
        #expect(stale.envelope.error?.code == "bootstrapPlanMismatch")
        #expect(stale.exitCode == 70)
        let realCallsAfter = await harness.shell.joinedCalls()
            .filter { $0.contains("bootstrap") && !$0.contains("--dry-run") }.count
        #expect(realCallsAfter == realCallsBefore)

        // The real apply is submit → persist → poll (review finding 5): the
        // helper hands back the job id at once, and a SEPARATE status probe
        // carries the verdict. These rules are registered now, after the dry
        // run, because both commands share the helper's path.
        await harness.shell.on(
            "submit-bootstrap-job.sh", exit: 0,
            lines: [
                "STEERLAB_BOOTSTRAP_JOB_ID=4242",
                "STEERLAB_BOOTSTRAP_STATUS_FILE=/scratch/me/ws/b.status",
            ])
        await harness.shell.on(
            "--status", exit: 0,
            lines: [
                "STEERLAB_BOOTSTRAP_STATUS=completed-ok job=4242 exit=0",
                #"{"ok":true,"steps":{"condaDetect":"ok"},"envFile":"/e","prefix":"/p"}"#,
            ])

        let applied = await harness.runner.run(
            invocation(harness, .bootstrapApply, planHash: planHash))
        #expect(applied.exitCode == 0)
        #expect(applied.envelope.changed)
        #expect(applied.envelope.schedulerJobID == "4242")
        // The submitted job is durable BEFORE anything could interrupt it,
        // and closed out once it finished.
        let records = harness.operationStore.records(forSite: "test-cluster")
        #expect(records.contains { $0.bootstrapJobID == "4242" })
        #expect(records.contains {
            $0.bootstrapStatusFile == "/scratch/me/ws/b.status"
        })
        #expect(harness.operationStore.pendingBootstrapJob(forSite: "test-cluster") == nil)
    }

    @Test func aQueuedBootstrapJobIsResumedByTheNextInvocation() async throws {
        // The recovery the finding asked for: the Mac sleeps while the job is
        // still queued, so the first apply returns `pending` with the job
        // recorded, and the SECOND apply polls that job instead of sbatching
        // a second bootstrap.
        let harness = try makeHarness("bootstrap-resume")
        await scriptHealthySite(harness)
        await harness.shell.on(
            "bootstrap", exit: 0,
            lines: [#"{"ok":true,"steps":{"condaDetect":"planned"},"envFile":"/e","prefix":"/p"}"#])
        // WP5 Step 7: every apply pushes the rendered environment first — twice
        // here, because resuming a queued job re-installs the same bytes rather
        // than assuming the earlier push survived. Matched on the push heredoc's
        // delimiter; see the note in the case above.
        await harness.shell.on("STEERLAB_RENDERED_ENV_EOF", exit: 0)
        let planned = await harness.runner.run(invocation(harness, .bootstrapPlan))
        let planHash = try #require(planned.envelope.planHash)

        await harness.shell.on(
            "submit-bootstrap-job.sh", exit: 0,
            lines: [
                "STEERLAB_BOOTSTRAP_JOB_ID=4242",
                "STEERLAB_BOOTSTRAP_STATUS_FILE=/scratch/me/ws/b.status",
            ])
        await harness.shell.on(
            "--status", exit: 0,
            lines: ["STEERLAB_BOOTSTRAP_STATUS=pending job=4242 state=PENDING"])

        let queued = await harness.runner.run(
            invocation(harness, .bootstrapApply, planHash: planHash))
        #expect(queued.envelope.state == ClusterLifecycleState.pending.rawValue)
        #expect(queued.envelope.schedulerJobID == "4242")
        // A queued job is a WAIT, not an error to repair.
        #expect(queued.envelope.error == nil)
        #expect(queued.envelope.nextAction?.detail?.contains("not be resubmitted") == true)
        let pending = try #require(
            harness.operationStore.pendingBootstrapJob(forSite: "test-cluster"))
        #expect(pending.jobID == "4242")

        let submitsAfterFirst = await harness.shell.callCount(
            containing: "submit-bootstrap-job.sh --bootstrap-script")
        await harness.shell.on(
            "--status", exit: 0,
            lines: [
                "STEERLAB_BOOTSTRAP_STATUS=completed-ok job=4242 exit=0",
                #"{"ok":true,"steps":{"condaDetect":"ok"},"envFile":"/e","prefix":"/p"}"#,
            ])
        let resumed = await harness.runner.run(
            invocation(harness, .bootstrapApply, planHash: planHash))
        #expect(resumed.exitCode == 0)
        #expect(resumed.envelope.schedulerJobID == "4242")
        // NOTHING was submitted the second time.
        #expect(
            await harness.shell.callCount(
                containing: "submit-bootstrap-job.sh --bootstrap-script")
                == submitsAfterFirst)
        #expect(harness.operationStore.pendingBootstrapJob(forSite: "test-cluster") == nil)
    }

    // MARK: controller

    @Test func controllerStartNeverSubmitsASecondJob() async throws {
        let harness = try makeHarness("controller-double")
        await scriptHealthySite(harness, controllerState: "PENDING|Resources")
        try seedControllerJob(harness)

        let outcome = await harness.runner.run(invocation(harness, .controllerStart))
        #expect(outcome.exitCode == 12)
        #expect(outcome.envelope.schedulerJobID == "4242")
        #expect(outcome.envelope.retryAfterSeconds == 30)
        #expect(await harness.shell.callCount(containing: "&& sbatch ") == 0)
    }

    /// The repair for the site that needs it MOST — one whose controller is
    /// already running — so it must run BEFORE the "never submit a second job"
    /// reconcile, or the fix would be a no-op exactly where it matters
    /// (open-issues §1 field report, 2026-08-20).
    @Test func controllerStartRenderOnlyRefreshesTheScriptAndSubmitsNothing() async throws {
        let harness = try makeHarness("controller-render-only")
        await scriptHealthySite(harness, controllerState: "RUNNING|None")
        try seedControllerJob(harness)
        let digest = String(repeating: "d", count: 64)
        await harness.shell.on(
            "> ~/.steerlab/controller-job.sbatch", exit: 0,
            lines: ["template=controller-job.sbatch.template sha256=\(digest)"])

        let outcome = await harness.runner.run(
            invocation(harness, .controllerStart, renderOnly: true))
        #expect(outcome.exitCode == 0)
        #expect(outcome.envelope.changed)
        #expect(outcome.envelope.message.contains("re-rendered"))
        #expect(outcome.envelope.message.contains("Nothing was submitted"))
        #expect(outcome.envelope.message.contains(digest))
        // The running controller is untouched, and the operator is told the
        // one thing re-rendering does NOT fix.
        #expect(outcome.envelope.nextAction?.detail?.contains("cannot chain") == true)
        #expect(await harness.shell.callCount(containing: "&& sbatch ") == 0)
    }

    @Test func aQueuedControllerIsPendingNotFailed() async throws {
        let harness = try makeHarness("controller-pending")
        await scriptHealthySite(harness, controllerState: "PENDING|Resources")
        try seedControllerJob(harness)
        let outcome = await harness.runner.run(invocation(harness, .controllerStatus))
        #expect(outcome.envelope.state == ClusterLifecycleState.pending.rawValue)
        #expect(outcome.exitCode == 12)
        #expect(outcome.envelope.retryAfterSeconds == 30)
        // An unreadable scheduler is DEGRADED — retryable, and never a licence
        // to resubmit.
        await harness.shell.on("squeue", exit: 255, lines: ["ssh: timed out"])
        let degraded = await harness.runner.run(invocation(harness, .controllerStatus))
        #expect(degraded.exitCode == 13)
    }

    @Test func controllerAdoptRefusesEveryUnverifiableJob() async throws {
        // The scheduler answered that the job is gone.
        let absent = try makeHarness("adopt-absent")
        await scriptHealthySite(absent)
        await absent.shell.on("squeue", exit: 0, lines: [])
        let gone = await absent.runner.run(
            invocation(absent, .controllerAdopt, jobID: "9999"))
        #expect(gone.envelope.error?.code == "controllerAdoptionUnverified")
        #expect(gone.exitCode == 70)
        // Nothing was recorded — a poisoned id would break every later
        // reconciliation.
        #expect(absent.operationStore.lastControllerJobID(forSite: "test-cluster") == nil)

        // The scheduler could not be read at all.
        let unknown = try makeHarness("adopt-unknown")
        await scriptHealthySite(unknown)
        await unknown.shell.on("squeue", exit: 255, lines: ["ssh: timed out"])
        let unreadable = await unknown.runner.run(
            invocation(unknown, .controllerAdopt, jobID: "4242"))
        #expect(unreadable.envelope.error?.code == "controllerAdoptionUnverified")
        #expect(unknown.operationStore.lastControllerJobID(forSite: "test-cluster") == nil)

        // Running, but it never published where it runs.
        let hostless = try makeHarness("adopt-hostless")
        await scriptHealthySite(hostless)
        await hostless.shell.on("serverd.host", exit: 1)
        let noHost = await hostless.runner.run(
            invocation(hostless, .controllerAdopt, jobID: "4242"))
        #expect(noHost.envelope.error?.code == "controllerAdoptionUnverified")
        #expect(noHost.envelope.error?.reason.contains("serverd.host") == true)

        // A dead job's leftover host file is not proof of life.
        let dead = try makeHarness("adopt-dead")
        await scriptHealthySite(dead, controllerState: "TIMEOUT|TimeLimit")
        let failed = await dead.runner.run(
            invocation(dead, .controllerAdopt, jobID: "4242"))
        #expect(failed.envelope.error?.code == "controllerAdoptionUnverified")
    }

    @Test func controllerAdoptRecordsAVerifiedJobSoLaterCallsReconcileIt() async throws {
        let harness = try makeHarness(
            "adopt-ok", storedToken: Self.secretToken, tunnel: .up(localPort: 8712))
        await scriptHealthySite(harness)
        // Nothing is recorded yet: this is exactly the site whose controller
        // predates the operation store.
        #expect(harness.operationStore.lastControllerJobID(forSite: "test-cluster") == nil)

        let outcome = await harness.runner.run(
            invocation(harness, .controllerAdopt, jobID: "4242"))
        #expect(outcome.exitCode == 0)
        #expect(outcome.envelope.changed)
        #expect(outcome.envelope.schedulerJobID == "4242")
        #expect(outcome.envelope.schedulerState == "RUNNING")
        #expect(harness.operationStore.lastControllerJobID(forSite: "test-cluster") == "4242")

        // And the reconciliation now works: status sees a running controller
        // rather than an unattributable host file.
        let status = await harness.runner.run(invocation(harness, .controllerStatus))
        #expect(status.exitCode == 0)
        #expect(status.envelope.schedulerJobID == "4242")
        // The adoption document names every layer it checked, and leaks nothing.
        let layers = try #require(outcome.envelope.layers).map(\.layer)
        #expect(layers.contains("controller"))
        #expect(layers.contains("daemonHost"))
        #expect(layers.contains("serverHTTP"))
        #expect(!(try outcome.envelope.jsonText()).contains(Self.secretToken))
    }

    @Test func controllerLogsTailsWithoutAJobIdRefusingHelpfully() async throws {
        let harness = try makeHarness("logs")
        await scriptHealthySite(harness)
        let none = await harness.runner.run(invocation(harness, .controllerLogs))
        #expect(none.envelope.nextAction?.verb == "cluster controller adopt")

        try seedControllerJob(harness)
        await harness.shell.on("tail", exit: 0, lines: ["line one", "line two"])
        let recorder = ActivationRecorder()
        let tailed = await harness.runner.run(invocation(harness, .controllerLogs)) {
            recorder.record($0)
        }
        #expect(tailed.exitCode == 0)
        #expect(recorder.endpoints == ["line one", "line two"])
        #expect(tailed.envelope.logPath?.contains("steerlab-serverd.4242.out") == true)

        // --follow goes through the streamer, so lines arrive as they land
        // rather than after the process ends.
        let followed = await harness.runner.run(
            invocation(harness, .controllerLogs, follow: true)) { _ in }
        #expect(harness.streamer.streamCount == 1)
        #expect(followed.exitCode == 0)
    }

    // MARK: tunnel / connect

    @Test func tunnelOpenRefusesWithoutATrustworthyNodeRecord() async throws {
        let harness = try makeHarness("tunnel-stale")
        await scriptHealthySite(harness, controllerState: "PENDING|Resources")
        try seedControllerJob(harness)
        // The host file exists but its job is queued — stale, so there is
        // nothing to forward to.
        let outcome = await harness.runner.run(invocation(harness, .tunnelOpen))
        #expect(outcome.exitCode == 13)
        #expect(harness.tunnel.openCount == 0)
        #expect(outcome.envelope.message.contains("stale"))
    }

    @Test func connectProvesIdentityAndRegistersExactlyOneEndpoint() async throws {
        let harness = try makeHarness("connect", tunnel: .up(localPort: 8712))
        await scriptHealthySite(harness)
        try seedControllerJob(harness)

        let outcome = await harness.runner.run(
            invocation(harness, .connect, activateInApp: true))
        #expect(outcome.exitCode == 0)
        #expect(outcome.envelope.endpoint == "http://127.0.0.1:8712")
        #expect(outcome.envelope.tokenAvailable == true)
        #expect(outcome.envelope.tokenSource == "keychain")
        // One registration, updated in place.
        let stored = try #require(try harness.repository.site(id: "test-cluster"))
        #expect(stored.lastEndpoint == "http://127.0.0.1:8712")
        #expect(harness.activations.endpoints == ["http://127.0.0.1:8712"])
        #expect(outcome.envelope.message.contains("Phase D"))
        // The imported token never appears in the document.
        #expect(!(try outcome.envelope.jsonText()).contains(Self.secretToken))
    }

    @Test func disconnectClosesTheForwardAndKeepsTheToken() async throws {
        let harness = try makeHarness(
            "disconnect", storedToken: Self.secretToken, tunnel: .up(localPort: 8712))
        await scriptHealthySite(harness)
        try seedControllerJob(harness)
        _ = try harness.repository.noteConnection(
            siteID: "test-cluster", endpoint: "http://127.0.0.1:8712", serverBuild: nil)

        let outcome = await harness.runner.run(invocation(harness, .disconnect))
        #expect(outcome.exitCode == 0)
        #expect(harness.tunnel.closeCount == 1)
        #expect(try harness.repository.site(id: "test-cluster")?.lastEndpoint == nil)
        // The bearer token is the user's credential for a server that is still
        // there — disconnecting a tunnel is not a reason to discard it.
        #expect(harness.secrets.token(forKey: "login.test:8080") == Self.secretToken)
    }

    // MARK: ensure — the boundaries an agent must be able to read

    @Test func ensureStopsAtEachBoundaryWithAMachineReadableNextAction() async throws {
        // Human authentication.
        let cold = try makeHarness("ensure-cold")
        let human = await cold.runner.run(invocation(cold, .ensure))
        #expect(human.exitCode == 10)
        #expect(human.envelope.nextAction?.requiresHuman == true)
        #expect(human.envelope.retryAfterSeconds == 5)
        #expect(human.envelope.step == "authenticate")
        // The Terminal did NOT open: that is its own authorization.
        #expect(cold.launcher.openCount == 0)

        // Approval, with the exact flag to add.
        let approval = try makeHarness("ensure-approval")
        await approval.shell.on("-O check", exit: 0)
        await approval.shell.on("ConnectTimeout", exit: 0)
        await approval.shell.on(ClusterProvisioner.deploymentManifestFileName, exit: 1)
        let needsApproval = await approval.runner.run(
            ClusterCLIInvocation(
                verb: .ensure, siteReference: "test-cluster", target: .codeDeployed,
                overrides: overrides(approval)))
        #expect(needsApproval.exitCode == 11)
        #expect(
            needsApproval.envelope.nextAction?.missingPermissionFlags == ["--allow-push"])
        #expect(await approval.shell.callCount(containing: "/usr/bin/rsync") == 0)

        // Ready, and repeating changes nothing.
        let warm = try makeHarness(
            "ensure-warm", storedToken: Self.secretToken, tunnel: .up(localPort: 8712))
        await scriptHealthySite(warm)
        try seedControllerJob(warm)
        _ = try warm.repository.noteConnection(
            siteID: "test-cluster", endpoint: "http://127.0.0.1:8712",
            serverBuild: "steerlab-server 0.1.0")
        let ready = await warm.runner.run(
            invocation(warm, .ensure, permissions: .allMutations))
        #expect(ready.exitCode == 0)
        let again = await warm.runner.run(
            invocation(warm, .ensure, permissions: .allMutations))
        #expect(again.exitCode == 0)
        #expect(!again.envelope.changed)
        #expect(warm.tunnel.openCount == 0)
    }

    /// open-issues §17: whatever `ensure` re-persists, it must not be the ssh
    /// destination's login. A registry that quietly loses the `user@` half
    /// authenticates as the LOCAL account, and an SSO cluster reports that as
    /// an unenrolled user — the 2026-08-18 Duo loop, which read as an upstream
    /// outage for an afternoon.
    @Test func ensureNeverRewritesTheSSHLoginOutOfTheRegistry() async throws {
        let harness = try makeHarness(
            "ensure-login", storedToken: Self.secretToken,
            tunnel: .up(localPort: 8712))
        // Give the saved site a login, the way the researcher's own registry
        // carries one (same canonical identity, so this refreshes the record).
        var withLogin = slurmProfile()
        withLogin.transport = .ssh(
            host: "alice@login.test", proxyJump: nil, remotePort: 8080,
            vpnExpected: false)
        _ = try harness.repository.upsert(profile: withLogin)
        await scriptHealthySite(harness)
        try seedControllerJob(harness)
        _ = try harness.repository.noteConnection(
            siteID: "test-cluster", endpoint: "http://127.0.0.1:8712",
            serverBuild: "steerlab-server 0.1.0")

        _ = await harness.runner.run(
            invocation(harness, .ensure, permissions: .allMutations))

        let stored = try #require(try harness.repository.site(id: "test-cluster"))
        guard case .ssh(let host, _, _, _) = stored.profile.transport else {
            Issue.record("the transport stopped being ssh")
            return
        }
        #expect(host == "alice@login.test")
    }

    /// The write-time guard, through the verb that hit it: importing a
    /// login-less profile over the login-carrying site refuses, names the
    /// repair, and leaves the registry exactly as it was.
    @Test func sitesImportRefusesAProfileThatWouldDropTheSSHLogin() async throws {
        let harness = try makeHarness("sites-import-login")
        var withLogin = slurmProfile()
        withLogin.transport = .ssh(
            host: "alice@login.test", proxyJump: nil, remotePort: 8080,
            vpnExpected: false)
        _ = try harness.repository.upsert(profile: withLogin)

        let path = harness.root.appending(component: "loginless.json")
        try slurmProfile().encoded().write(to: path)
        let refused = await harness.runner.run(
            ClusterCLIInvocation(verb: .sitesImport, positional: path.path))
        #expect(refused.exitCode != 0)
        #expect(refused.envelope.error?.code == "sshLoginDropped")
        #expect(refused.envelope.error?.repairAction
            .contains("alice@login.test") == true)

        let stored = try #require(try harness.repository.site(id: "test-cluster"))
        guard case .ssh(let host, _, _, _) = stored.profile.transport else {
            Issue.record("the transport stopped being ssh")
            return
        }
        #expect(host == "alice@login.test")
    }

    /// The other half of the rule: a site that carries no login ANYWHERE is
    /// legal (`~/.ssh/config` can supply `User`) and imports — loudly.
    @Test func sitesImportWarnsButAcceptsALoginlessSite() async throws {
        let harness = try makeHarness("sites-import-warn")
        let path = harness.root.appending(component: "loginless.json")
        try slurmProfile().encoded().write(to: path)
        let imported = await harness.runner.run(
            ClusterCLIInvocation(verb: .sitesImport, positional: path.path))
        #expect(imported.exitCode == 0)
        #expect(imported.envelope.message.contains("WARNING"))
        #expect(imported.envelope.message.contains("~/.ssh/config"))
    }

    @Test func theAuthTerminalOpensFromEnsureOnlyWhenSeparatelyAuthorized() async throws {
        let harness = try makeHarness("ensure-terminal")
        _ = await harness.runner.run(
            invocation(harness, .ensure, permissions: .allMutations))
        #expect(harness.launcher.openCount == 0)
        _ = await harness.runner.run(
            invocation(harness, .ensure, permissions: [.openAuthTerminal]))
        #expect(harness.launcher.openCount == 1)
    }

    @Test func aSecondProcessObservesRatherThanDuplicating() async throws {
        let harness = try makeHarness("ensure-locked")
        await scriptHealthySite(harness)
        let lock = try #require(
            try harness.operationStore.acquireLock(siteID: "test-cluster"))
        defer { lock.release() }
        var running = ClusterOperationRecord(
            operationID: "op-other", siteID: "test-cluster", siteProfileHash: "h",
            target: .connected, state: .pending)
        running.controllerJobID = "4242"
        try harness.operationStore.save(running)

        let outcome = await harness.runner.run(
            invocation(harness, .ensure, permissions: .allMutations))
        #expect(outcome.envelope.error?.code == "operationInProgress")
        #expect(outcome.envelope.error?.repairAction.contains("cluster status") == true)
        #expect(await harness.shell.callCount(containing: "&& sbatch ") == 0)
        // Inspection is ALWAYS allowed, lock or no lock.
        let status = await harness.runner.run(invocation(harness, .status))
        #expect(status.envelope.error == nil)
    }

    // MARK: diagnose

    @Test func diagnoseRedactsUsersAndPathsWithoutLosingTheDiagnosis() async throws {
        let harness = try makeHarness(
            "diagnose", storedToken: Self.secretToken, tunnel: .up(localPort: 8712))
        await scriptHealthySite(harness)
        try seedControllerJob(harness)

        let plain = await harness.runner.run(invocation(harness, .diagnose))
        #expect(plain.envelope.command?.first == "ssh")
        #expect(plain.envelope.logPath?.contains("/scratch/me/ws") == true)

        let redacted = await harness.runner.run(invocation(harness, .diagnose, redact: true))
        let text = try redacted.envelope.jsonText()
        #expect(!text.contains("/scratch/me/"))
        #expect(text.contains("/scratch/<user>"))
        // Still diagnosable: the layers and their states survive.
        #expect(redacted.envelope.layers?.count == plain.envelope.layers?.count)
        #expect(!text.contains(Self.secretToken))
    }

    // MARK: Secrets, across the whole surface

    @Test func noVerbEverEmitsTokenMaterial() async throws {
        let harness = try makeHarness(
            "no-leak", storedToken: Self.secretToken, tunnel: .up(localPort: 8712))
        await scriptHealthySite(harness)
        try seedControllerJob(harness)
        _ = try harness.repository.noteConnection(
            siteID: "test-cluster", endpoint: "http://127.0.0.1:8712", serverBuild: nil)

        let verbs: [ClusterCLIVerb] = [
            .sitesList, .sitesShow, .preview, .status, .diagnose, .authCommand, .authStatus,
            .bootstrapStatus, .validate, .controllerStatus, .tunnelStatus,
            .plan, .ensure,
        ]
        for verb in verbs {
            let outcome = await harness.runner.run(
                invocation(harness, verb, permissions: .allMutations))
            let json = try outcome.envelope.jsonText()
            #expect(
                !json.contains(Self.secretToken),
                "\(verb.displayName) leaked token material into its JSON")
            #expect(
                !ClusterCLIRenderer.humanText(outcome.envelope).contains(Self.secretToken),
                "\(verb.displayName) leaked token material into its human output")
        }
        // Nor into any durable byte the surface wrote.
        for record in harness.operationStore.records(forSite: "test-cluster") {
            #expect(!record.transcript.joined().contains(Self.secretToken))
            #expect(!record.steps.map(\.message).joined().contains(Self.secretToken))
        }
    }
}

// MARK: - remote --site

/// `remote … --site <id>` (§5.4): an agent should not need the ephemeral local
/// port or the bearer token to use the data plane.
struct ClusterRemoteSiteResolutionTests {

    private static let token = "sk-REMOTE-SECRET-13"

    private func harness(
        _ name: String, endpoint: String? = "http://127.0.0.1:8712",
        storedToken: String? = ClusterRemoteSiteResolutionTests.token,
        tunnel: ClusterTunnelObservation = .up(localPort: 8712)
    ) throws -> (ClusterSiteRepository, FakeCLISecrets, FakeCLITunnel) {
        let root = FileManager.default.temporaryDirectory
            .appending(components: "steerlab-remote-site", "\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let repository = ClusterSiteRepository(
            fileURL: root.appending(component: "cluster-sites.json"),
            legacyRegistryData: { nil })
        let profile = ClusterSiteProfile(
            name: "Test Cluster",
            transport: .ssh(
                host: "login.test", proxyJump: nil, remotePort: 8080, vpnExpected: false),
            topology: .daemonInJob)
        _ = try repository.upsert(profile: profile)
        if let endpoint {
            _ = try repository.noteConnection(
                siteID: "test-cluster", endpoint: endpoint, serverBuild: nil)
        }
        return (
            repository,
            FakeCLISecrets(storedToken.map { ["login.test:8080": $0] } ?? [:]),
            FakeCLITunnel(observation: tunnel)
        )
    }

    @Test func aConnectedSiteResolvesEndpointAndTokenWithoutRevealingEither()
        async throws
    {
        let (repository, secrets, tunnel) = try harness("connected")
        let resolved = try await ClusterRemoteSiteResolver.resolve(
            reference: "test-cluster", repository: repository, secrets: secrets,
            tunnel: tunnel)
        #expect(resolved.siteID == "test-cluster")
        #expect(resolved.baseURL.absoluteString == "http://127.0.0.1:8712")
        // The token IS resolved (the transport needs it) …
        #expect(resolved.token == Self.token)
        #expect(resolved.tokenAvailable)
        // … and the only thing anything may PRINT is presence and provenance.
        #expect(!resolved.redactedSummary.contains(Self.token))
        #expect(resolved.redactedSummary.contains("keychain"))
        #expect(resolved.tokenSource == "keychain")
    }

    @Test func siteReferencesResolveByIdOrUniqueDisplayName() async throws {
        let (repository, secrets, tunnel) = try harness("by-name")
        let byName = try await ClusterRemoteSiteResolver.resolve(
            reference: "Test Cluster", repository: repository, secrets: secrets,
            tunnel: tunnel)
        #expect(byName.siteID == "test-cluster")
    }

    @Test func anUnconnectedSiteRefusesWithTheExactEnsureCommand() async throws {
        // No endpoint has ever been registered.
        let (noEndpoint, secrets, tunnel) = try harness("no-endpoint", endpoint: nil)
        do {
            _ = try await ClusterRemoteSiteResolver.resolve(
                reference: "test-cluster", repository: noEndpoint, secrets: secrets,
                tunnel: tunnel)
            Issue.record("expected a refusal")
        } catch let error as ClusterCLIError {
            #expect(error.code == "siteNotConnected")
            #expect(error.exitCode == 13)
            #expect(
                error.repairAction
                    == "run `steerlab-cli cluster ensure --site test-cluster --target connected`")
        }

        // A registered endpoint whose forward is gone is STALE, not usable:
        // sending a request at a dead port would just time out later.
        let (stale, staleSecrets, staleTunnel) = try harness(
            "stale",
            tunnel: .stale(localPort: 8712, reason: "nothing is listening on the port"))
        do {
            _ = try await ClusterRemoteSiteResolver.resolve(
                reference: "test-cluster", repository: stale, secrets: staleSecrets,
                tunnel: staleTunnel)
            Issue.record("expected a refusal")
        } catch let error as ClusterCLIError {
            #expect(error.code == "siteNotConnected")
            #expect(error.reason.contains("stale"))
        }

        // Connected, but no token: refuse rather than send an unauthenticated
        // request that the server will reject confusingly.
        let (noToken, emptySecrets, liveTunnel) = try harness(
            "no-token", storedToken: nil)
        do {
            _ = try await ClusterRemoteSiteResolver.resolve(
                reference: "test-cluster", repository: noToken, secrets: emptySecrets,
                tunnel: liveTunnel)
            Issue.record("expected a refusal")
        } catch let error as ClusterCLIError {
            #expect(error.reason.contains("Keychain"))
        }
    }

    @Test func anUnknownSiteRefusesWithTheStableLifecycleCode() async throws {
        let (repository, secrets, tunnel) = try harness("unknown")
        do {
            _ = try await ClusterRemoteSiteResolver.resolve(
                reference: "nope", repository: repository, secrets: secrets,
                tunnel: tunnel)
            Issue.record("expected a refusal")
        } catch let error as ClusterLifecycleError {
            #expect(error.code == "unknownSite")
        }
    }

    /// `--activate-in-app` writes the app's own persisted server-URL key. The
    /// runner mirrors the spelling because `ClusterConnectionStore` is
    /// main-actor isolated; if the app ever renames it, this fails.
    @MainActor
    @Test func theActivationKeyMatchesTheAppsOwnPreference() {
        #expect(
            ClusterCLIRunner.appServerURLDefaultsKey
                == ClusterConnectionStore.serverURLDefaultsKey)
    }
}
