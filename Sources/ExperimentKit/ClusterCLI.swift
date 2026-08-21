import Foundation

// =============================================================================
// The `cluster` verb surface (CLUSTER-CLI-LIFECYCLE-PLAN §6) — Phase C.
//
// Argument parsing lives HERE, not in `steerlab-cli/main.swift`, for the same
// reason the orchestration does (plan §7.1): the CLI binary parses, calls
// ExperimentKit, and serializes. Everything a test would want to assert about
// the surface — which flags a verb accepts, how a target name is spelled, what
// an unknown flag costs — is reachable without spawning a process.
//
// Two hard rules from the plan carry into this file:
//   * §4.1/§4.2 — there is NO flag, field, or environment variable anywhere in
//     this surface that can carry a password, a Duo response, or a bearer
//     token. `--token` survives only on the legacy `remote` verbs, and the
//     site-aware path deliberately replaces it.
//   * §6.5 — a caller must be able to determine the exact next action from the
//     machine output alone, so every refusal here has a STABLE code, a
//     plain-language reason, and a concrete repair action.
// =============================================================================

// MARK: - Verbs

/// Every `cluster` verb, spelled exactly as it is typed.
///
/// `watch` is deliberately absent: §0.3 of the plan drops it from this cut, and
/// polling `ensure`/`status` with `retryAfterSeconds` is the v1 contract.
public enum ClusterCLIVerb: String, CaseIterable, Sendable, Equatable {
    case sitesList = "sites list"
    case sitesShow = "sites show"
    case sitesExport = "sites export"
    case sitesImport = "sites import"
    /// WP5 §3.3: read the complete generated environment and scheduler
    /// commands BEFORE anything runs. Read-only and offline — it touches no
    /// shell, no scheduler, and no network; it renders the saved profile.
    case preview = "preview"
    case status = "status"
    case diagnose = "diagnose"
    case authCommand = "auth command"
    case authOpen = "auth open"
    case authStatus = "auth status"
    case authClose = "auth close"
    case push = "push"
    case bootstrapPlan = "bootstrap plan"
    case bootstrapApply = "bootstrap apply"
    case bootstrapStatus = "bootstrap status"
    case validate = "validate"
    case controllerStart = "controller start"
    case controllerStatus = "controller status"
    case controllerLogs = "controller logs"
    case controllerStop = "controller stop"
    /// Beyond plan §6.3: record a controller job that was started by hand (or
    /// by a build that predates the operation store) AFTER verifying it. Without
    /// it, a site whose controller is already running can never be reconciled —
    /// inspection has no job id to ask the scheduler about, so it reports
    /// `unknown` forever and the planner (correctly) refuses to submit a second.
    case controllerAdopt = "controller adopt"
    case tunnelOpen = "tunnel open"
    case tunnelStatus = "tunnel status"
    case tunnelClose = "tunnel close"
    case connect = "connect"
    case disconnect = "disconnect"
    /// Bring the paired cluster workspace's run directories home under the
    /// shared import policy (open-issues §20). Mac-authority by the
    /// source-of-truth rule: the workspace is durable here, cluster scratch
    /// purges, and one code path serves both this verb and the app's hook.
    case importRuns = "import"
    case plan = "plan"
    case ensure = "ensure"

    /// How the verb is named in output and in `nextAction.verb`.
    public var displayName: String { "cluster \(rawValue)" }

    /// The words that select it, in order.
    public var words: [String] { rawValue.split(separator: " ").map(String.init) }

    /// Whether `--site` is required. Only the two verbs that operate on the
    /// registry as a whole are exempt.
    public var requiresSite: Bool {
        switch self {
        case .sitesList, .sitesImport: false
        default: true
        }
    }

    /// Whether the verb may change anything at all. Inspection is always
    /// allowed (§4.3), and the runner uses this to decide whether to take the
    /// per-site lock.
    public var isReadOnly: Bool {
        switch self {
        case .sitesList, .sitesShow, .sitesExport, .preview, .status, .diagnose,
            .authCommand, .authStatus, .bootstrapStatus, .controllerStatus,
            .controllerLogs, .tunnelStatus, .plan:
            true
        // `import` writes into the LOCAL workspace and changes nothing on the
        // cluster — it enumerates and copies, and it deletes nothing on either
        // side. Read-only in the sense this flag means (no site mutation), so
        // it never contends with a lifecycle operation for the site lock.
        case .importRuns:
            true
        default:
            false
        }
    }

    /// Valueless flags this verb accepts, beyond the universal `--json`.
    var booleanFlags: Set<String> {
        var flags: Set<String>
        switch self {
        case .status: flags = ["--refresh"]
        case .diagnose: flags = ["--redact"]
        case .push: flags = ["--dry-run"]
        case .importRuns: flags = ["--dry-run"]
        case .controllerLogs: flags = ["--follow"]
        case .connect: flags = ["--activate-in-app"]
        // §1 field report (2026-08-20): the render half of `controller start`,
        // on its own. THE re-render command every other surface names.
        case .controllerStart: flags = ["--render-only"]
        case .ensure: flags = [
            "--allow-push", "--allow-bootstrap", "--allow-controller-start",
            "--open-auth-terminal", "--allow-open-auth-terminal",
        ]
        default: flags = []
        }
        // The wizard's editable provisioning TOGGLES, as CLI overrides. Same
        // rule as the value flags: a verb that runs no remote command rejects
        // them rather than silently ignoring them.
        if acceptsConfigurationOverrides {
            flags.formUnion(ClusterCLIOverrides.booleanFlagNames)
        }
        return flags
    }

    /// Flags that take a value, beyond the universal `--site`.
    var valueFlags: Set<String> {
        var flags: Set<String> = []
        switch self {
        case .sitesExport: flags.insert("--out")
        case .preview: flags.insert("--job-class")
        case .bootstrapApply: flags.insert("--plan-hash")
        // Every controller verb may name a job explicitly; without it they use
        // the durably recorded one.
        case .controllerAdopt, .controllerStatus, .controllerLogs, .controllerStop:
            flags.insert("--job-id")
        case .plan, .ensure: flags.insert("--target")
        case .importRuns: flags.insert("--since")
        default: break
        }
        // The wizard's editable provisioning fields. A site whose remote root,
        // partition, or `squeue` wrapper differs from the defaults is otherwise
        // undriveable headlessly.
        if acceptsConfigurationOverrides { flags.formUnion(ClusterCLIOverrides.flagNames) }
        return flags
    }

    /// Whether the provisioning-configuration overrides apply. They are inputs
    /// to remote commands, so the registry-only and purely local verbs reject
    /// them rather than silently ignoring them.
    var acceptsConfigurationOverrides: Bool {
        switch self {
        // `preview` renders the SAVED PROFILE. A provisioning override is not
        // profile data, so accepting one here would show an environment the
        // stored site does not imply — the exact confusion the preview exists
        // to remove.
        case .sitesList, .sitesShow, .sitesExport, .sitesImport, .preview,
            .authCommand, .authOpen, .authStatus, .authClose,
            // `import` reads the site's declared storage roots and its ssh
            // transport, and nothing else the provisioning overrides carry
            // (payload path, env prefix, bootstrap partition, `squeue`
            // wrapper). Accepting one would silently ignore it.
            .importRuns:
            false
        default:
            true
        }
    }

    /// Match a verb from the leading words, returning it plus how many words it
    /// consumed. Two-word verbs are tried first so `sites` alone cannot shadow
    /// `sites list`.
    public static func match(_ words: [String]) -> (verb: ClusterCLIVerb, consumed: Int)? {
        if words.count >= 2 {
            let pair = "\(words[0]) \(words[1])"
            if let verb = ClusterCLIVerb(rawValue: pair) { return (verb, 2) }
        }
        if let first = words.first, let verb = ClusterCLIVerb(rawValue: first) {
            return (verb, 1)
        }
        return nil
    }

    /// The one positional argument the verb takes, spelled as the usage text
    /// prints it. Empty for the twenty-seven that take none.
    public var positional: String {
        self == .sitesImport ? "<profile.json>" : ""
    }

    /// One line saying what running the verb does. CONTRACT TEXT (WP0 step
    /// 11): imperative, neutral, and about the effect — the rationale, the
    /// incident history, and the security essay stay in the reference
    /// document's prose, which the generation never touches.
    public var purpose: String {
        switch self {
        case .sitesList: "List every saved site, with token presence only."
        case .sitesShow: "Print one site's registry record without probing it."
        case .sitesExport: "Write the site's profile — never a credential — to a file."
        case .sitesImport: "Upsert a site profile by its canonical remote identity."
        case .preview:
            "Render the environment and scheduler commands this site will run."
        case .status: "Report each lifecycle layer's state, read-only."
        case .diagnose: "Report status plus the auth command, log path, and last operations."
        case .authCommand: "Print the SSH ControlMaster command for a human to run."
        case .authOpen: "Open a visible Terminal holding the authentication command."
        case .authStatus: "Check whether the shared SSH session is alive."
        case .authClose: "Close SteerLab's own SSH session, and no other."
        case .push: "Deploy the allowlisted server payload and re-stamp its build id."
        case .bootstrapPlan: "Render the bootstrap dry run and print its plan hash."
        case .bootstrapApply: "Run exactly the reviewed bootstrap plan."
        case .bootstrapStatus: "Report whether the remote environment is valid."
        case .validate: "Run the remote engine's own profile validation."
        case .controllerStart: "Submit the controller job, then return."
        case .controllerStatus: "Report the controller job's scheduler state and log path."
        case .controllerLogs: "Read the tail of the controller job's log."
        case .controllerStop: "Cancel the recorded controller job."
        case .controllerAdopt: "Record a hand-started controller after verifying it."
        case .tunnelOpen: "Install or adopt the local forward to the controller."
        case .tunnelStatus: "Report the local forward as absent, up, stale, or conflicted."
        case .tunnelClose: "Cancel SteerLab's exact forward, and no other."
        case .connect: "Reach the connected rung without any mutation authority."
        case .disconnect: "Close the forward and clear the registered endpoint."
        case .importRuns:
            "Bring this site's run directories into the workspace under the "
                + "import policy, verify them by content, and rebuild the catalog."
        case .plan: "Print what reaching the target would do, and execute nothing."
        case .ensure: "Advance the site to the target rung, idempotently."
        }
    }

    /// The verb's declared flags, sorted, with the universal ones included.
    /// `--site` is listed only where it applies, which is what makes the
    /// generated usage honest about the two registry-wide verbs.
    public var declaredFlags: [String] {
        var flags = booleanFlags.union(valueFlags)
        if requiresSite { flags.insert("--site") }
        flags.insert("--json")
        flags.insert(ExperimentCLIParser.helpFlag)
        return flags.sorted()
    }

    /// Whether a declared flag takes the next argument as its value.
    public func takesValue(_ flag: String) -> Bool {
        flag == "--site" || valueFlags.contains(flag)
    }

    /// `cluster ensure --site <id> [--target <t>] …` — generated, never
    /// hand-kept. Until step 11 this was a hand-written literal with no test
    /// asserting it covered `allCases` or agreed with the flag sets, which is
    /// the drift the declarative table exists to prevent.
    public var synopsis: String {
        var parts = ["steerlab-cli", displayName]
        if !positional.isEmpty { parts.append(positional) }
        for flag in declaredFlags {
            let spelling = CLIFlagVocabulary.spelling(
                flag, verb: displayName, takesValue: takesValue(flag))
            // `--site` is required wherever it is accepted, so it is not
            // optional-bracketed: an agent reading the synopsis must be able
            // to see the difference.
            parts.append(flag == "--site" ? spelling : "[\(spelling)]")
        }
        return parts.joined(separator: " ")
    }

    /// The whole verb surface, GENERATED from `allCases` (audit §5.3,
    /// `ClusterUsageTextIsGenerated`). The trailing paragraphs are the two
    /// facts a caller must know before running anything, and are the only
    /// hand-written text left here.
    public static var usageText: String {
        var lines = ["usage: steerlab-cli cluster <verb> [--site <id>] [--json]", ""]
        let width = min(allCases.map { $0.rawValue.count }.max() ?? 0, 20)
        for verb in allCases {
            let padding = String(
                repeating: " ", count: max(2, width + 2 - verb.rawValue.count))
            lines.append("  " + verb.rawValue + padding + verb.purpose)
        }
        lines.append("")
        lines.append(
            "steerlab-cli cluster <verb> --help prints that verb's arguments.")
        lines.append("")
        lines.append(
            """
            targets: \(ClusterLifecycleTarget.allCases.map(\.cliName).joined(separator: " | "))\
              (default: connected)
            job classes: \
            \(ClusterEnvironmentRenderer.JobClass.allCases.map(\.cliName).joined(separator: " | "))\
              (default: all)

            Authentication is human-only: the CLI can open a visible Terminal, and
            can never accept, read, or forward a password or multi-factor response.
            """)
        return lines.joined(separator: "\n")
    }

    /// One verb's `--help` page.
    public var helpText: String {
        var lines = ["usage: " + synopsis, "", purpose]
        lines.append("")
        lines.append("flags:")
        let spellings = declaredFlags.map {
            CLIFlagVocabulary.spelling(
                $0, verb: displayName, takesValue: takesValue($0))
        }
        let width = min(spellings.map(\.count).max() ?? 0, 34)
        for (spelling, flag) in zip(spellings, declaredFlags) {
            let purpose = CLIFlagVocabulary.purpose(flag, verb: displayName)
            let padding = String(
                repeating: " ", count: max(2, width + 2 - spelling.count))
            lines.append("  " + spelling + (purpose.isEmpty ? "" : padding + purpose))
        }
        lines.append("")
        lines.append(ExperimentCLIHelp.exitCodeLine)
        return lines.joined(separator: "\n") + "\n"
    }
}

// MARK: - Provisioning overrides

/// The wizard's editable provisioning fields, as CLI overrides over
/// `ClusterProvisioningConfiguration.defaults(for:)`.
///
/// These are configuration, not secrets: paths, a partition name, a `squeue`
/// wrapper. Nothing here can carry a credential.
public struct ClusterCLIOverrides: Sendable, Equatable {
    public var localPayloadPath: String?
    public var remoteRepoPath: String?
    public var envPrefix: String?
    public var pythonVersion: String?
    public var bootstrapPartition: String?
    public var squeueCommand: String?
    public var envFile: String?
    /// **WP5 Step 7 — materialization is the default; `--no-materialize-env`
    /// opts out.**
    ///
    /// Nil (no flag typed) leaves
    /// `ClusterProvisioningConfiguration.materializeEnvironmentFile` at its own
    /// `true`, so the cluster sources the environment rendered from the site
    /// profile. `--no-materialize-env` sets it false: `bootstrap.sh` then writes
    /// its built-in fallback env file, whose values are the script's defaults
    /// rather than this site's declared facts, and the plan transcript carries a
    /// WARNING saying exactly that. `--materialize-env` is still accepted — it
    /// now names the default, so scripts written against Step 6 keep working and
    /// mean what they say.
    public var materializeEnvironmentFile: Bool?

    public init() {}

    static let flagNames: Set<String> = [
        "--payload", "--remote-repo", "--env-prefix", "--python-version",
        "--bootstrap-partition", "--squeue", "--env-file",
    ]

    static let booleanFlagNames: Set<String> = [
        "--materialize-env", "--no-materialize-env",
    ]

    mutating func apply(flag: String, value: String) {
        switch flag {
        case "--payload": localPayloadPath = value
        case "--remote-repo": remoteRepoPath = value
        case "--env-prefix": envPrefix = value
        case "--python-version": pythonVersion = value
        case "--bootstrap-partition": bootstrapPartition = value
        case "--squeue": squeueCommand = value
        case "--env-file": envFile = value
        default: break
        }
    }

    mutating func apply(booleanFlag: String) {
        switch booleanFlag {
        case "--materialize-env": materializeEnvironmentFile = true
        case "--no-materialize-env": materializeEnvironmentFile = false
        default: break
        }
    }

    /// Layer the overrides onto the site's own defaults.
    public func resolved(
        for site: ClusterSiteProfile
    ) -> ClusterProvisioningConfiguration {
        var configuration = ClusterProvisioningConfiguration.defaults(for: site)
        if let localPayloadPath { configuration.localPayloadPath = localPayloadPath }
        if let remoteRepoPath { configuration.remoteRepoPath = remoteRepoPath }
        if let envPrefix { configuration.envPrefix = envPrefix }
        if let pythonVersion { configuration.pythonVersion = pythonVersion }
        if let bootstrapPartition {
            configuration.bootstrapJobPartition = bootstrapPartition
        }
        if let squeueCommand { configuration.squeueCommand = squeueCommand }
        if let envFile { configuration.bootstrapEnvFile = envFile }
        if let materializeEnvironmentFile {
            configuration.materializeEnvironmentFile = materializeEnvironmentFile
        }
        return configuration
    }
}

// MARK: - Job classes on the command line

extension ClusterEnvironmentRenderer.JobClass {
    /// How the class is typed. `gpuSession` is spelled `gpu-session` on a
    /// command line, matching how the lifecycle targets are spelled, while the
    /// wire/JSON name stays the camel-cased `rawValue` the renderer and the
    /// committed header goldens use.
    public var cliName: String {
        self == .gpuSession ? "gpu-session" : rawValue
    }

    /// Accept either spelling, case-insensitively — an agent that read the
    /// JSON (`gpuSession`) and an admin who read the usage text
    /// (`gpu-session`) must both be understood.
    public static func parse(_ text: String) -> Self? {
        let normalized = text.trimmingCharacters(in: .whitespaces)
            .lowercased()
            .replacingOccurrences(of: "-", with: "")
            .replacingOccurrences(of: "_", with: "")
        return allCases.first { $0.rawValue.lowercased() == normalized }
    }
}

// MARK: - Invocation

/// One parsed `cluster …` command line.
public struct ClusterCLIInvocation: Sendable, Equatable {
    public var verb: ClusterCLIVerb
    public var siteReference: String?
    public var target: ClusterLifecycleTarget
    public var permissions: ClusterLifecyclePermissions
    public var json: Bool
    public var refresh: Bool
    public var redact: Bool
    public var dryRun: Bool
    public var follow: Bool
    public var activateInApp: Bool
    /// `controller start --render-only`: refresh the RENDERED controller
    /// script from the current template and stop before `sbatch`
    /// (open-issues §1 field report, 2026-08-20). The repair for a site whose
    /// controller is already running — the artifact goes back into step with
    /// the template, and no second controller joins the queue.
    public var renderOnly: Bool
    public var planHash: String?
    public var jobID: String?
    public var outPath: String?
    /// `preview --job-class <class>`: narrow the header pane to one class. Nil
    /// previews every class.
    public var jobClass: ClusterEnvironmentRenderer.JobClass?
    /// `import --since <t>`, normalized to the run stamp's own lexicographic
    /// space so the filter reads the run's START, not a filesystem mtime a
    /// later touch would move.
    public var since: String?
    /// The one positional argument any verb takes (`sites import <file>`).
    public var positional: String?
    public var overrides: ClusterCLIOverrides
    /// `--help`: print the verb's declared surface and run nothing (exit 0).
    public var help: Bool

    public init(
        verb: ClusterCLIVerb,
        siteReference: String? = nil,
        target: ClusterLifecycleTarget = .connected,
        permissions: ClusterLifecyclePermissions = [],
        json: Bool = false,
        refresh: Bool = false,
        redact: Bool = false,
        dryRun: Bool = false,
        follow: Bool = false,
        activateInApp: Bool = false,
        renderOnly: Bool = false,
        planHash: String? = nil,
        jobID: String? = nil,
        outPath: String? = nil,
        jobClass: ClusterEnvironmentRenderer.JobClass? = nil,
        since: String? = nil,
        positional: String? = nil,
        overrides: ClusterCLIOverrides = ClusterCLIOverrides(),
        help: Bool = false
    ) {
        self.since = since
        self.verb = verb
        self.siteReference = siteReference
        self.target = target
        self.permissions = permissions
        self.json = json
        self.refresh = refresh
        self.redact = redact
        self.dryRun = dryRun
        self.follow = follow
        self.activateInApp = activateInApp
        self.renderOnly = renderOnly
        self.planHash = planHash
        self.jobID = jobID
        self.outPath = outPath
        self.jobClass = jobClass
        self.positional = positional
        self.overrides = overrides
        self.help = help
    }
}

// MARK: - Errors

/// Typed CLI-surface failures. Every case carries a STABLE code, a
/// plain-language reason, and a concrete repair action (plan §6.5), plus the
/// process exit code a shell caller should see.
public enum ClusterCLIError: Error, LocalizedError, Equatable {
    case unknownVerb(String)
    case unknownFlag(flag: String, verb: ClusterCLIVerb)
    case missingFlagValue(String)
    case missingSite(ClusterCLIVerb)
    case missingArgument(verb: ClusterCLIVerb, what: String)
    case invalidTarget(String)
    case invalidJobClass(String)
    /// `--since` that no date grammar accepts. A silently ignored window flag
    /// would look like "nothing new to import".
    case invalidSince(String)
    case unexpectedArgument(String)
    /// `remote --site` and `remote --url` both supplied.
    case siteAndURLAreMutuallyExclusive
    /// The site has no usable connection to reach through.
    case siteNotConnected(siteID: String, reason: String)

    public var code: String {
        switch self {
        case .unknownVerb: "unknownVerb"
        case .unknownFlag: "unknownFlag"
        case .missingFlagValue: "missingFlagValue"
        case .missingSite: "missingSite"
        case .missingArgument: "missingArgument"
        case .invalidTarget: "invalidTarget"
        case .invalidJobClass: "invalidJobClass"
        case .invalidSince: "invalidSince"
        case .unexpectedArgument: "unexpectedArgument"
        case .siteAndURLAreMutuallyExclusive: "siteAndURLAreMutuallyExclusive"
        case .siteNotConnected: "siteNotConnected"
        }
    }

    /// Plain-language reason, no ANSI, no prose the caller must parse.
    public var reason: String {
        switch self {
        case .unknownVerb(let word):
            "'\(word)' is not a cluster verb"
        case .unknownFlag(let flag, let verb):
            "\(verb.displayName) does not accept \(flag)"
        case .missingFlagValue(let flag):
            "\(flag) requires a value"
        case .missingSite(let verb):
            "\(verb.displayName) requires --site <id>"
        case .missingArgument(let verb, let what):
            "\(verb.displayName) requires \(what)"
        case .invalidTarget(let text):
            "'\(text)' is not a lifecycle target"
        case .invalidJobClass(let text):
            "'\(text)' is not a job class"
        case .invalidSince(let text):
            "'\(text)' is not a date this verb understands"
        case .unexpectedArgument(let word):
            "unexpected argument '\(word)'"
        case .siteAndURLAreMutuallyExclusive:
            "--site and --url are mutually exclusive"
        case .siteNotConnected(let siteID, let reason):
            "site '\(siteID)' has no usable connection: \(reason)"
        }
    }

    /// The exact next command, never a hint to go read something.
    public var repairAction: String {
        switch self {
        case .unknownVerb:
            "run `steerlab-cli cluster` for the verb list"
        case .unknownFlag(_, let verb):
            "drop the flag, or run `steerlab-cli cluster` for \(verb.displayName)'s flags"
        case .missingFlagValue(let flag):
            "supply a value after \(flag)"
        case .missingSite:
            "list the saved sites with `steerlab-cli cluster sites list`"
        case .missingArgument(_, let what):
            "supply \(what)"
        case .invalidTarget:
            "targets: "
                + ClusterLifecycleTarget.allCases.map(\.cliName).joined(separator: " | ")
        case .invalidJobClass:
            "job classes: "
                + ClusterEnvironmentRenderer.JobClass.allCases.map(\.cliName)
                    .joined(separator: " | ")
        case .invalidSince:
            "pass a day (2026-08-01 or 20260801) or an instant "
                + "(2026-08-01T09:30:00), or a run stamp copied from a run "
                + "directory's name"
        case .unexpectedArgument:
            "remove the extra argument"
        case .siteAndURLAreMutuallyExclusive:
            "pass --site <id> for a saved cluster site, or --url <server> for an "
                + "unmanaged server — not both"
        case .siteNotConnected(let siteID, _):
            "run `steerlab-cli cluster ensure --site \(siteID) --target connected`"
        }
    }

    public var errorDescription: String? { "\(reason) — \(repairAction)" }

    /// Plan §6.5: 64 for invalid usage or invalid site configuration.
    /// A site that is merely not connected yet is a retryable degraded state,
    /// not a malformed command.
    public var exitCode: Int32 {
        switch self {
        case .siteNotConnected: 13
        default: 64
        }
    }
}

// MARK: - Parser

/// Pure argument parsing for the `cluster` namespace.
public enum ClusterCLIParser {

    /// Parse `cluster`'s arguments (everything AFTER the word `cluster`).
    ///
    /// Strict by design: an unknown verb or an unknown flag is a usage error
    /// rather than a silent no-op, because an agent that mistypes a permission
    /// flag must be told rather than quietly denied the mutation.
    public static func parse(_ arguments: [String]) throws -> ClusterCLIInvocation {
        guard let matched = ClusterCLIVerb.match(arguments) else {
            throw ClusterCLIError.unknownVerb(arguments.first ?? "")
        }
        var invocation = ClusterCLIInvocation(verb: matched.verb)

        // `--help` is answered before `--site`, `--plan-hash`, or anything
        // else this parser requires: a caller asking what a verb takes must
        // not have to supply it first (WP0 step 11).
        if arguments.contains(ExperimentCLIParser.helpFlag) {
            invocation.help = true
            invocation.json = arguments.contains("--json")
            return invocation
        }
        let rest = Array(arguments.dropFirst(matched.consumed))
        let booleans = matched.verb.booleanFlags
        let valued = matched.verb.valueFlags

        var index = 0
        while index < rest.count {
            let word = rest[index]
            if word == "--json" {
                invocation.json = true
                index += 1
                continue
            }
            if word == "--site" || valued.contains(word) {
                guard index + 1 < rest.count else {
                    throw ClusterCLIError.missingFlagValue(word)
                }
                let value = rest[index + 1]
                switch word {
                case "--site": invocation.siteReference = value
                case "--target":
                    guard let target = ClusterLifecycleTarget.parse(value) else {
                        throw ClusterCLIError.invalidTarget(value)
                    }
                    invocation.target = target
                case "--job-class":
                    guard let jobClass = ClusterEnvironmentRenderer.JobClass.parse(value)
                    else {
                        throw ClusterCLIError.invalidJobClass(value)
                    }
                    invocation.jobClass = jobClass
                case "--since":
                    guard let normalized = WorkspaceImportPolicy.normalizedSince(value)
                    else { throw ClusterCLIError.invalidSince(value) }
                    invocation.since = normalized
                case "--out": invocation.outPath = value
                case "--plan-hash": invocation.planHash = value
                case "--job-id": invocation.jobID = value
                default: invocation.overrides.apply(flag: word, value: value)
                }
                index += 2
                continue
            }
            if booleans.contains(word) {
                switch word {
                case "--refresh": invocation.refresh = true
                case "--redact": invocation.redact = true
                case "--dry-run": invocation.dryRun = true
                case "--follow": invocation.follow = true
                case "--activate-in-app": invocation.activateInApp = true
                case "--render-only": invocation.renderOnly = true
                case "--allow-push": invocation.permissions.insert(.push)
                case "--allow-bootstrap": invocation.permissions.insert(.bootstrap)
                case "--allow-controller-start":
                    invocation.permissions.insert(.controllerStart)
                // The plan spells this one WITHOUT `--allow-` (§4.3); the
                // `--allow-` form is accepted so the permission vocabulary's
                // own `flags` rendering is never a dead end.
                case "--open-auth-terminal", "--allow-open-auth-terminal":
                    invocation.permissions.insert(.openAuthTerminal)
                default: invocation.overrides.apply(booleanFlag: word)
                }
                index += 1
                continue
            }
            if word.hasPrefix("--") {
                throw ClusterCLIError.unknownFlag(flag: word, verb: matched.verb)
            }
            guard invocation.positional == nil else {
                throw ClusterCLIError.unexpectedArgument(word)
            }
            invocation.positional = word
            index += 1
        }

        if matched.verb.requiresSite, invocation.siteReference == nil {
            throw ClusterCLIError.missingSite(matched.verb)
        }
        switch matched.verb {
        case .sitesImport where invocation.positional == nil:
            throw ClusterCLIError.missingArgument(
                verb: matched.verb, what: "a profile JSON path")
        case .sitesExport where invocation.outPath == nil:
            throw ClusterCLIError.missingArgument(
                verb: matched.verb, what: "--out <file>")
        case .bootstrapApply where invocation.planHash == nil:
            throw ClusterCLIError.missingArgument(
                verb: matched.verb,
                what: "--plan-hash <sha256> from `cluster bootstrap plan`")
        case .controllerAdopt where invocation.jobID == nil:
            throw ClusterCLIError.missingArgument(
                verb: matched.verb, what: "--job-id <scheduler job id>")
        default:
            break
        }
        return invocation
    }
}
