import Foundation
import SteeringKit

// =============================================================================
// Strict flag parsing and the agent-path invocation (WP0-AGENT-SURFACE-AUDIT
// §7 step 5, punch-list P0-4).
//
// Until now every non-cluster verb read its flags with an ad-hoc
// `args.firstIndex(of:)` scan, which means an argument list is never
// VALIDATED: a mistyped `--reivsion` is not an error, it is a silently
// unpinned model revision, and `--pool-from` typed as `--poolfrom` silently
// changes the reading position of every vector the study extracts. The audit's
// dry run found this on the live surface and escalated it to P0-4: "a typo'd
// semantic flag changes study meaning with no signal".
//
// The fix is the one `cluster` already has (`ClusterCLIParser`): a declared
// flag set per verb, and anything else is a usage error — exit 64, in BOTH
// output modes, before the verb does any work. That last clause is the point.
// A refusal after the first concept is pinned is not much better than no
// refusal at all.
//
// The table is also what step 11 generates `--help` and `CLI-REFERENCE.md`
// from; it is written as data for that reason and not folded into the switch
// statements it guards.
// =============================================================================

/// One verb's declared argument surface.
public struct ExperimentCLIVerbSpec: Sendable {

    /// The verb family — `experiment`, `data`, `vectors`, `remote`,
    /// `workspace`, `docs`.
    public let namespace: String
    /// The sub-verb as typed, e.g. `attach`. EMPTY for a family that IS one
    /// verb and takes no sub-verb — `init`, whose whole spelling is
    /// `steerlab-cli init [--home <dir>]`. A bare-verb spec is looked up when
    /// the argument list opens with a flag (or is empty), and everything else
    /// about it — strict flags, `--help`, the envelope, the generated
    /// reference region — is identical to a two-word verb's.
    public let verb: String
    /// The positional arguments, spelled as `--help` and the reference
    /// document print them: `<name> <concept>…`. Empty when the verb takes
    /// none. CONTRACT TEXT (step 11) — neutral, imperative, agent-readable.
    public let positional: String
    /// One line saying what running the verb does. CONTRACT TEXT: it is what
    /// `--help` renders and what the generated `CLI-REFERENCE.md` regions
    /// carry, so it states the effect, not the rationale.
    public let purpose: String
    /// Flags that stand alone.
    public let booleanFlags: Set<String>
    /// Flags that take the next argument as their value.
    public let valueFlags: Set<String>
    /// Flags the verb REFUSES without. Rendered unbracketed in `--help` and in
    /// the reference document, because "optional" and "the verb will not run
    /// without it" are the difference between a usable synopsis and a
    /// misleading one.
    public let requiredFlags: Set<String>
    /// True when the verb already owns `--out` with a meaning of its own, so
    /// the shared "write the envelope here" flag must not steal it.
    /// `remote fetch` and `remote import` are the two: their `--out` is the
    /// download DIRECTORY, and quietly repurposing it would send an evidence
    /// bundle somewhere the caller did not ask for.
    public let ownsOutFlag: Bool
    /// True when this verb historically spelled `--json <path>` — a file
    /// destination rather than a mode. Accepted for one release with a
    /// deprecation warning (audit §2.2); `--out <path>` is the replacement.
    public let acceptsLegacyJSONPath: Bool
    /// Flags this verb does NOT accept but a caller plausibly reaches for
    /// here, mapped to the command that does own them.
    ///
    /// The case that earned it: `set-sweep-grid` and `set-sweep-selection`
    /// write two halves of ONE manifest block, so `--objective` typed on the
    /// grid verb is not a typo — it is a correct intent aimed one verb over.
    /// The generic refusal ("accepts: …") answers a different question, and
    /// leaves the caller to infer which of twenty-three verbs owns the field.
    /// Declaring the owner as DATA keeps the pointer beside the split it
    /// describes.
    public let redirectedFlags: [String: String]

    public init(
        namespace: String, verb: String,
        positional: String = "", purpose: String = "",
        booleanFlags: Set<String> = [], valueFlags: Set<String> = [],
        requiredFlags: Set<String> = [],
        ownsOutFlag: Bool = false, acceptsLegacyJSONPath: Bool = false,
        redirectedFlags: [String: String] = [:]
    ) {
        self.namespace = namespace
        self.verb = verb
        self.positional = positional
        self.purpose = purpose
        self.booleanFlags = booleanFlags
        self.valueFlags = valueFlags
        self.requiredFlags = requiredFlags
        self.ownsOutFlag = ownsOutFlag
        self.acceptsLegacyJSONPath = acceptsLegacyJSONPath
        self.redirectedFlags = redirectedFlags
    }

    /// `experiment attach` — the verb as it is typed and as the envelope
    /// header carries it. A bare-verb spec's label is the family alone
    /// (`init`), never `init ` with a trailing space: the label is what
    /// `--help` prints and what the envelope's `verb` carries, and both have
    /// to be typeable.
    public var label: String { verb.isEmpty ? namespace : "\(namespace) \(verb)" }

    /// Every flag this verb accepts, including the shared agent flags, sorted
    /// — the list a refusal prints so the repair is mechanical.
    ///
    /// `--help` is in the list from step 11 on: it is a DECLARED flag on every
    /// verb, and a caller told "this verb accepts …" should be told the one
    /// flag that answers the question it just failed to answer for itself.
    public var declaredFlags: [String] {
        var flags = booleanFlags.union(valueFlags)
        flags.insert(ExperimentCLIParser.helpFlag)
        flags.insert(ExperimentCLIParser.jsonFlag)
        if !ownsOutFlag { flags.insert(ExperimentCLIParser.outFlag) }
        return flags.sorted()
    }
}

/// A flag this verb does not accept. Exit 64 (`EX_USAGE`) in both output
/// modes: the audit's compatibility posture holds human exit codes still for
/// refusals, but a MALFORMED INVOCATION was never a refusal — `cluster` has
/// exited 64 for it since Phase C, and `steerlab-cli` with no arguments at all
/// already does.
public struct ExperimentCLIUsageError: Error, CustomStringConvertible, Equatable {

    /// The offending token, e.g. `--reivsion`.
    public let flag: String
    /// The verb as typed, e.g. `experiment attach`.
    public let verb: String
    /// Every flag the verb does accept.
    public let declaredFlags: [String]
    /// The verb that DOES own this flag, when the table declares one
    /// (`ExperimentCLIVerbSpec.redirectedFlags`). nil for an ordinary typo.
    public let ownedBy: String?
    /// A DECLARED value flag that arrived with no value: the metavar it wanted
    /// (`CLIFlagVocabulary.metavar`). nil for an ordinary undeclared-flag
    /// error, which is what every other field describes.
    public let expectedValue: String?
    /// The token standing where the value should have been — another of this
    /// verb's own declared flags, which cannot be a value. nil when the flag
    /// simply ended the argument list.
    public let followedBy: String?
    /// The verb's one-line synopsis, the runnable repair for a missing value
    /// (the flag list alone would answer a question nobody asked: the flag was
    /// right, its value was absent).
    public let usage: String?

    public init(
        flag: String, verb: String, declaredFlags: [String],
        ownedBy: String? = nil, expectedValue: String? = nil,
        followedBy: String? = nil, usage: String? = nil
    ) {
        self.flag = flag
        self.verb = verb
        self.declaredFlags = declaredFlags
        self.ownedBy = ownedBy
        self.expectedValue = expectedValue
        self.followedBy = followedBy
        self.usage = usage
    }

    public var reason: String {
        if let expectedValue {
            let head = "flag \(flag) expects a value (\(expectedValue))"
            guard let followedBy else { return head + " and none followed" }
            return head + " and none followed — the next token \(followedBy) "
                + (declaredFlags.contains(followedBy)
                    ? "is another \(verb) flag, not a value"
                    : "is flag-shaped, not a value")
        }
        guard let ownedBy else { return "\(verb) does not accept \(flag)" }
        return "\(flag) is \(ownedBy)'s flag, not \(verb)'s"
    }

    /// The envelope's failure code. A declared flag missing its value is not
    /// an unknown flag — the caller spelled it right — so it says so.
    public var code: String {
        expectedValue == nil ? Self.code : "missingFlagValue"
    }

    public var repairAction: String {
        if let usage, expectedValue != nil { return usage }
        if let ownedBy {
            // The owner FIRST: the caller's intent was right and only aimed
            // one verb over, so the runnable answer is the other command, not
            // this one's flag list. The list still follows, because the next
            // question is what this verb would have taken.
            return "steerlab-cli \(ownedBy) <name> \(flag) <value>"
                + (declaredFlags.isEmpty
                    ? ""
                    : "  ; \(verb) accepts: "
                        + declaredFlags.joined(separator: " "))
        }
        return declaredFlags.isEmpty
            ? "\(verb) takes no flags — remove \(flag)"
            : "\(verb) accepts: \(declaredFlags.joined(separator: " "))"
    }

    /// Rendered exactly as every other runner failure is: `reason` on the
    /// first line, the repair indented on the second.
    public var description: String { reason }

    public static let code = "unknownFlag"
}

/// One parsed agent-path invocation: which verb, what is left for it to read,
/// and the two shared agent flags lifted out of the way.
public struct ExperimentCLIInvocation: Sendable {

    /// The verb family — `arguments[1]`.
    public var namespace: String
    /// The sub-verb, when one was given and recognised.
    public var verb: String?
    /// What the verb dispatch sees: the sub-verb and its own arguments, with
    /// the shared agent flags removed. A legacy `--json <path>` is left IN,
    /// because the verb that owns that spelling still reads it.
    public var args: [String]
    /// `--json`: one envelope document on stdout, diagnostics on stderr.
    public var json: Bool
    /// `--out <path>`: write the same document to a file.
    public var outPath: String?
    /// Deprecation notices to print on stderr before the verb runs.
    public var deprecations: [String]
    /// `--help`: print the declared surface and run nothing. Exit 0 — asking
    /// what a verb accepts is not an error, and an agent that has to parse a
    /// failure to read a manual has no manual.
    public var help: Bool

    public init(
        namespace: String, verb: String?, args: [String], json: Bool = false,
        outPath: String? = nil, deprecations: [String] = [], help: Bool = false
    ) {
        self.namespace = namespace
        self.verb = verb
        self.args = args
        self.json = json
        self.outPath = outPath
        self.deprecations = deprecations
        self.help = help
    }

    /// The verb label the envelope's header carries — exactly as typed.
    public var label: String {
        verb.map { "\(namespace) \($0)" } ?? namespace
    }
}

/// The declared argument surface of every agent-path verb, and the strict
/// parse over it.
public enum ExperimentCLIParser {

    /// The mode flag. Boolean everywhere from this step forward.
    public static let jsonFlag = "--json"
    /// The file form, replacing the historical `--json <path>`.
    public static let outFlag = "--out"
    /// Print the verb's declared surface and run NOTHING. Declared on every
    /// verb from step 11 on: until then a strict parser answered `--help` with
    /// exit 64, which is the one refusal a caller cannot repair by reading it.
    public static let helpFlag = "--help"

    /// Flags every `remote` verb shares — the site-resolution preamble.
    private static let remoteConnection: Set<String> = ["--site", "--url", "--token"]

    /// The table. Data on purpose: step 11 generates `--help` and the
    /// reference document's flag rows from it, and a switch statement cannot
    /// be enumerated.
    public static let specs: [ExperimentCLIVerbSpec] = [
        // init — the home layout (GENERAL-DISTRIBUTION-WORK-PLAN decision 8,
        // work item (a)). The one BARE verb: a git repository cannot ship its
        // own parent directory, so the home folder that holds `Workspaces/`,
        // `Sites/`, and the checkout as siblings has to be materialized once,
        // and `steerlab-cli init` is how a first-run caller types that. It
        // creates directories and nothing else — no workspace, no git, no
        // persisted preference — so the workspace resolution chain is exactly
        // what it was.
        .init(
            namespace: "init", verb: "",
            purpose: "Create the SteerLab home layout's Workspaces/ and Sites/ "
                + "directories (default home ~/SteerLab).",
            valueFlags: ["--home"]),

        // workspace
        .init(
            namespace: "workspace", verb: "init", positional: "<path>",
            purpose: "Create and seed a data workspace, and git-init it."),

        // data
        .init(
            namespace: "data", verb: "check", positional: "<experiment>",
            purpose: "Report which study-data inputs the manifest still needs."),

        // vectors
        .init(
            namespace: "vectors", verb: "compare",
            positional: "<a.safetensors> <b.safetensors>",
            purpose: "Compare two vector artifacts and refuse below the cosine "
                + "threshold.",
            valueFlags: ["--threshold"], acceptsLegacyJSONPath: true),
        .init(
            namespace: "vectors", verb: "backfill-norms",
            positional: "<runDir/name>",
            purpose: "Measure per-layer residual norms for an existing artifact "
                + "into a new artifact, stamped with the current denominator "
                + "convention (the opt-in migration for legacy unstamped "
                + "artifacts).",
            booleanFlags: ["--redenominate"], valueFlags: ["--corpus", "--model"]),
        // Pole mirroring: the OTHER end of a contrastive direction, as an
        // artifact rather than as a negative α. `--concept` is required and
        // enforced in the verb body rather than by the parser, because the
        // refusal has to explain WHY a new name is not optional (two artifacts
        // under one concept name pointing opposite ways) and a bare
        // "missing required flag" would not.
        .init(
            namespace: "vectors", verb: "mirror-poles",
            positional: "<runDir/name>",
            purpose: "Mint the opposite pole of a contrastive direction as a "
                + "new artifact — every layer negated bit-exactly, under a "
                + "required new concept name, with a negatedFrom stamp.",
            valueFlags: ["--concept", "--output-name"],
            requiredFlags: ["--concept"]),

        // panel — the multi-agent authoring family (open-issues §18). `list`
        // and `check` are the pre-existing read verbs, declared here when the
        // family joined the agent path so `--help` and the reference document
        // can see them; `compile` is the casting verb, and the reason the
        // family moved: `PanelComposition.compileAndPin` was reachable only
        // from the app, so every headless casting was a re-implementation of
        // the compile transform outside the engine.
        .init(
            namespace: "panel", verb: "list",
            purpose: "List this workspace's panel scenarios."),
        .init(
            namespace: "panel", verb: "check", positional: "<path-or-name>",
            purpose: "Validate one panel scenario and report its advisories."),
        .init(
            namespace: "panel", verb: "compile", positional: "<path-or-name>",
            purpose: "Cast a semantic panel's seats and pin the compiled "
                + "scenario into a draft study.",
            valueFlags: [
                "--experiment", "--seat", "--model", "--temperature",
                "--max-tokens", "--file-slug",
            ],
            requiredFlags: ["--experiment"]),

        // remote
        .init(
            namespace: "remote", verb: "capabilities",
            purpose: "Report the paired server's capability snapshot.",
            valueFlags: remoteConnection),
        .init(
            namespace: "remote", verb: "package", positional: "<experiment>",
            purpose: "Build a hash-pinned run bundle locally and print its path.",
            valueFlags: remoteConnection),
        .init(
            namespace: "remote", verb: "upload", positional: "<bundle>",
            purpose: "Upload a run bundle to the server.",
            valueFlags: remoteConnection),
        .init(
            namespace: "remote", verb: "submit-bundle",
            positional: "<server-bundle-path>",
            purpose: "Submit an uploaded bundle as a server job.",
            booleanFlags: ["--dry-run"],
            valueFlags: remoteConnection.union([
                "--bundle", "--verb", "--executor", "--gres", "--walltime",
                "--parallel", "--source",
                // The evaluate subsample, submitted rather than typed
                // locally: the cluster is where a judged evaluate of a large
                // corpus actually runs, so the flags have to reach it through
                // the same channel `--source` does.
                "--sample-per-condition", "--sample-seed",
            ])),
        .init(
            namespace: "remote", verb: "jobs",
            purpose: "List the server's jobs.", valueFlags: remoteConnection),
        .init(
            namespace: "remote", verb: "logs", positional: "<job-id>",
            purpose: "Stream one job's log.", valueFlags: remoteConnection),
        .init(
            namespace: "remote", verb: "cancel", positional: "<job-id>",
            purpose: "Request cancellation of one job.",
            valueFlags: remoteConnection),
        .init(
            namespace: "remote", verb: "resubmit", positional: "<job-id>",
            purpose: "Resume a checkpointed job: the server re-submits the "
                + "job's own rendered sbatch script byte-for-byte, "
                + "optionally under a longer walltime.",
            valueFlags: remoteConnection.union(["--walltime"])),
        .init(
            namespace: "remote", verb: "fetch", positional: "<artifact-path>",
            purpose: "Download one server artifact without importing it.",
            valueFlags: remoteConnection.union(["--path", "--out"]),
            ownsOutFlag: true),
        .init(
            namespace: "remote", verb: "import",
            positional: "<server-evidence-path>",
            purpose: "Download, hash-verify, and import an evidence bundle into "
                + "runs/.",
            valueFlags: remoteConnection.union(["--path", "--out", "--sha256"]),
            ownsOutFlag: true),
        .init(
            namespace: "remote", verb: "import-chain",
            positional: "<pipeline-run-id-or-experiment>",
            purpose: "Import a whole pipeline chain, skipping directories "
                + "already present.",
            valueFlags: remoteConnection),
        .init(
            namespace: "remote", verb: "variants",
            purpose: "List the server's variant artifacts.",
            valueFlags: remoteConnection),
        .init(
            namespace: "remote", verb: "chat",
            purpose: "Generate one completion through a server-resident variant.",
            booleanFlags: ["--strip"],
            valueFlags: remoteConnection.union([
                "--variant", "--prompt", "--hash", "--max-tokens", "--temperature",
                "--prompt-mode", "--system",
            ]),
            requiredFlags: ["--variant", "--prompt"]),

        // experiment — the lifecycle nineteen (sixteen at step 5, plus the
        // three authoring verbs step 5½ added)
        .init(
            namespace: "experiment", verb: "list",
            purpose: "List this workspace's experiments with their status."),
        .init(
            namespace: "experiment", verb: "create", positional: "<name>",
            purpose: "Create a draft manifest pinned to a model.",
            valueFlags: ["--model", "--revision", "--description"],
            requiredFlags: ["--model"]),
        .init(
            namespace: "experiment", verb: "attach",
            positional: "<name> <concept>…",
            purpose: "Pin each named concept's stimulus hash and extraction "
                + "options.",
            valueFlags: [
                "--method", "--pool-from", "--corpus", "--reference",
                "--project-neutral", ExtractionRendering.declarationFlag,
                ReadingPosition.declarationFlag,
            ]),
        // `attach`'s inverse, and the writer the authoring surface shipped
        // without: every other pin here could be replaced or cleared, and a
        // CONCEPT pin could only ever be ADDED. A draft therefore carried
        // whatever it was first attached with — a mis-pinned concept had to
        // ride along to freeze, and re-pointing one concept across a shelf of
        // drafts was a mechanical edit no command expressed.
        .init(
            namespace: "experiment", verb: "detach",
            positional: "<name> <concept>…",
            purpose: "Remove each named concept's pin from a draft — refused "
                + "while a declaration still names one."),
        // The headless authoring three (WP0 step 5½, punch-list P0-3): pin
        // the measured task, pin the judging instrument, declare an arm.
        .init(
            namespace: "experiment", verb: "pin-prompts",
            positional: "<name> <prompts/…/file.jsonl>",
            purpose: "Pin the measured task-prompt file and its hash "
                + "(\"\" clears the pin)."),
        .init(
            namespace: "experiment", verb: "pin-rubric",
            positional: "<name> <prompts/rubrics/file.md>",
            purpose: "Pin the judging rubric, the judge panel, and the "
                + "evaluation declaration they imply; --judge-pin declares a "
                + "local judge's revision and dtype (repeat per judge).",
            valueFlags: ["--judges", "--judge-pin"]),
        .init(
            namespace: "experiment", verb: "declare-condition",
            positional: "<name> <condition>",
            purpose: "Declare one experimental arm, or the explicit baseline.",
            booleanFlags: ["--baseline"],
            valueFlags: ["--slots", "--band-width", "--alpha-units", "--control"]),
        // Step 7's two authoring additions (punch list #1, P3 + P13): the
        // sweep's selection criterion and the study's outcome instruments
        // were manifest data with no headless authoring path at all, so the
        // sweep silently selected on marker density and a pinned choice task
        // was scored as prose.
        .init(
            namespace: "experiment", verb: "set-sweep-selection",
            positional: "<name>",
            purpose: "Declare the sweep's selection criterion as manifest data.",
            valueFlags: [
                "--objective", "--choice-prompts", "--capability-tolerance",
                "--coherence-floor", "--coherence-ratio", "--coherence-backstop",
                "--control-margin", "--control-apply-to",
                "--control-top-k",
            ]),
        // The grid's other half, and the gap that made the passenger-concept
        // problem possible: `sweep.selection` had a writer, the GRID it
        // selects over had none outside the Optimizations panel. The only
        // headless way to obtain a grid was to `duplicate` a study that had
        // one — which brings the donor's concepts with it.
        //
        // The two verbs split the block cleanly: `set-sweep-selection` owns
        // `sweep.selection` entire, this one owns the axes, the two
        // instrument files, and the per-cell budget. The selection flags are
        // DECLARED here as redirects so typing one lands on a pointer to its
        // owner rather than on a flag list.
        .init(
            namespace: "experiment", verb: "set-sweep-grid",
            positional: "<name>",
            purpose: "Declare the sweep's layer × alpha grid, its instrument "
                + "files, and its per-cell token budget.",
            valueFlags: [
                "--layer-fractions", "--layers", "--alphas", "--dev-prompts",
                "--battery", "--max-tokens",
            ],
            redirectedFlags: Dictionary(
                uniqueKeysWithValues: ExperimentStore.sweepSelectionOwnedFlags
                    .map { ($0, "experiment set-sweep-selection") })),
        // The legal values come from `ExperimentStore.knownOutcomeInstruments`
        // — the SAME constant the 64-refusal prints — so `--help` and the
        // refusal cannot name different vocabularies (gate-5 dry run #2, P3:
        // the help page named none, so the only way to learn the values was
        // to type a wrong one).
        .init(
            namespace: "experiment", verb: "set-instruments",
            positional: "<name> <instrument>[,…]",
            purpose: "Declare which outcome instruments the run measures ("
                + ExperimentStore.knownOutcomeInstruments.joined(separator: ", ")
                + "; \"\" clears the declaration).",
            valueFlags: ["--ordinal-aggregation"]),
        // The generation protocol's writer — the fields the Study Setup
        // panel owned exclusively until now. Field-discovered gap: a
        // stochastic replication arm (N samples × temperature × token
        // budget) could not be authored headlessly, so it was cut from a
        // study design. Same ownership reasoning as the sweep split: the
        // manifest's TOP-LEVEL sampling fields live here; the sweep block's
        // per-cell `--max-tokens` stays on `set-sweep-grid`.
        .init(
            namespace: "experiment", verb: "set-sampling",
            positional: "<name>",
            purpose: "Declare the generation protocol: temperature, token "
                + "budget, prompt mode ("
                + ExperimentStore.knownPromptModes.joined(separator: ", ")
                + "), and the stochastic replication policy (samples per "
                + "item × seed policy: "
                + ExperimentStore.knownSeedPolicies.joined(separator: ", ")
                + ").",
            valueFlags: [
                "--temperature", "--max-tokens", "--prompt-mode",
                "--samples-per-item", "--seed-policy",
            ]),
        // The legal values come from `ExclusionEngine.ruleVocabulary` — the
        // SAME constant the refusal prints — so `--help` and the refusal
        // cannot name different vocabularies.
        .init(
            namespace: "experiment", verb: "set-exclusions",
            positional: "<name> <rule>[,…]",
            purpose: "Declare the record-exclusion rules analysis applies ("
                + ExclusionEngine.ruleVocabulary.joined(separator: ", ")
                + "; \"\" clears the declaration).",
            valueFlags: ["--endpoint", "--min", "--max"]),
        // The study's system prompt — the deployment frame every arm is read
        // under. Field-discovered gap: writable from the Study Setup panel
        // and nowhere else, so a replication study whose donor carries a
        // judge-persona frame could not be authored headlessly at all.
        //
        // The purpose says what setting one PHYSICALLY DOES, because the
        // delivery route is capability-dependent and a researcher arming a
        // persona has to know which one they get: a family whose chat
        // template has a system role gets a genuine system turn; a family
        // without one (Gemma) gets the same text prepended to the first user
        // turn. Positional text, no `--file`: the manifest holds the frame as
        // text with no path and no pin beside it.
        .init(
            namespace: "experiment", verb: "set-system-prompt",
            positional: "<name> <text>",
            purpose: "Declare the study's system prompt — the deployment "
                + "frame every arm is read under. Inserted as a genuine "
                + "system turn where the model's chat template has a system "
                + "role; prepended to the first user turn where it does not "
                + "(e.g. Gemma-family); prepended to the prompt text under "
                + "rawCompletion. An agent arm reads under its persona then "
                + "this frame. \"\" clears the declaration."),
        // The numeric-endpoint grammar and its registry pin. The legal
        // values are workspace DATA (the registry file), not a compiled
        // vocabulary, so the purpose names the FILE and the refusal — which
        // can read it — names the parsers defined there. There is
        // deliberately no `--registry-hash`: the registry is the authority
        // on which parser version a study preregistered, so the hash is
        // derived at the write and can never be typed.
        .init(
            namespace: "experiment", verb: "set-parser",
            positional: "<name> <parser>",
            purpose: "Declare the numeric-endpoint parser from "
                + ParserRegistry.registryFile
                + " and pin that registry's hash (\"\" clears both)."),
        // The applicability subset the option-consuming instruments read.
        // Vocabulary from the type that owns the field, so `--help` and the
        // refusal cannot name different formats.
        .init(
            namespace: "experiment", verb: "set-instrument-scope",
            positional: "<name> <responseFormat>[,…]",
            purpose: "Declare which response formats the option-consuming "
                + "outcome instruments apply to ("
                + ExperimentStore.knownResponseFormats.joined(separator: ", ")
                + "), pinning the row set they select; \"\" clears the "
                + "declaration."),
        // The EVALUATION SAMPLING DESIGN (review round 12, finding 4). A verb
        // for the same reason as the two above: not a field assignment. It
        // takes the design a power computation produced — a per-condition
        // size and the seed that draws it — and DERIVES the draw rule from
        // `EvaluateSubsample.rule`, so `rule` can no more be typed here than
        // a registry hash can.
        .init(
            namespace: "experiment", verb: "set-evaluation-sampling",
            positional: "<name> <n> <seed>",
            purpose: "Declare the evaluate subsample this study "
                + "preregistered — n records per condition and the seed that "
                + "draws them — so the design travels in the manifest and "
                + "lands in every run's snapshot. evaluate then needs no "
                + "flags, and a flag that disagrees with the declaration "
                + "refuses. \"\" clears the declaration."),
        .init(
            namespace: "experiment", verb: "set-style-taxonomy",
            positional: "<name> <prompts/taxonomies/file.json>",
            purpose: "Pin the reasoning-style taxonomy and its hash."),
        .init(
            namespace: "experiment", verb: "verify", positional: "<name>",
            purpose: "Re-check every pinned input against the file bytes on "
                + "disk."),
        .init(
            namespace: "experiment", verb: "freeze", positional: "<name>",
            purpose: "Freeze the manifest one-way, after the evidence gates "
                + "pass.",
            booleanFlags: ["--force"], valueFlags: ["--run-substrate"]),
        .init(
            namespace: "experiment", verb: "duplicate",
            positional: "<name> <new-name>",
            purpose: "Copy a manifest into a new draft — how a frozen study is "
                + "iterated."),
        .init(
            namespace: "experiment", verb: "extract", positional: "<name>",
            purpose: "Derive the manifest's concept vectors on this engine."),
        .init(
            namespace: "experiment", verb: "validate", positional: "<name>",
            purpose: "Score each vector on its held-out probe and report "
                + "cross-concept similarity."),
        .init(
            namespace: "experiment", verb: "sweep", positional: "<name>",
            purpose: "Sweep layer × alpha on the dev split and record a "
                + "recommendation per concept."),
        .init(
            namespace: "experiment", verb: "run", positional: "<name>",
            purpose: "Generate the measured run for every declared condition.",
            valueFlags: ["--prompts"]),
        .init(
            namespace: "experiment", verb: "analyze", positional: "<name>",
            purpose: "Compute paired effect sizes from the newest completed "
                + "run into a fresh run directory.",
            booleanFlags: ["--allow-unverified-epoch"]),
        .init(
            namespace: "experiment", verb: "rescore-style", positional: "<name>",
            purpose: "Re-score reasoning style over a completed run into a "
                + "fresh run directory.",
            booleanFlags: ["--allow-unverified-epoch"], valueFlags: ["--run"]),
        .init(
            namespace: "experiment", verb: "evaluate", positional: "<name>",
            purpose: "Judge a completed run with the pinned rubric and judges.",
            booleanFlags: ["--allow-unverified-epoch"],
            // `--sample-per-condition n` + `--sample-seed s` code a seeded,
            // stratified SUBSAMPLE of the source run instead of all of it
            // (2026-08-29) — the preregistered design a power computation
            // produces. Both or neither: either half alone refuses at 64,
            // because a subsample nobody can redraw is not evidence and a
            // seed with no size stamps a coding it did not shape.
            valueFlags: [
                "--run", "--sample-per-condition", "--sample-seed",
            ]),
        .init(
            namespace: "experiment", verb: "promote",
            positional: "<name> <concept>",
            purpose: "Mint a variant artifact from the sweep-selected cell, "
                + "with its birth certificate.",
            valueFlags: ["--agent-name", "--cell", "--reason"]),
        .init(
            namespace: "experiment", verb: "confirm", positional: "<name>",
            purpose: "Expand a perturbation policy around a promoted agent "
                + "into hashed conditions.",
            booleanFlags: ["--no-control"], valueFlags: ["--agent", "--deltas"],
            requiredFlags: ["--agent"]),

        // install — the shipped binary talking about itself (step 12). These
        // three are the only verbs that answer questions a caller has BEFORE
        // it trusts the install: which build is this, is the tree intact, and
        // will a GPU verb find its shaders.
        .init(
            namespace: "install", verb: "version",
            purpose: "Report this build's version and where it is installed."),
        .init(
            namespace: "install", verb: "stamp",
            purpose: "Hash the installed tree into its resource manifest.",
            valueFlags: ["--revision", "--root"]),
        .init(
            namespace: "install", verb: "verify",
            purpose: "Check the installed tree against its resource manifest.",
            booleanFlags: ["--gpu"], valueFlags: ["--root"]),

        // authoring — the generation-prompt emitter. A family of its own
        // rather than an `experiment` verb because it takes NO experiment: it
        // answers "this study is missing X; what do I ask an LLM for", which
        // is a question about a KIND of data, not about a manifest. Nothing
        // here writes anything into the workspace — the emitter is never the
        // acceptor, and installing generated data is a separate, reviewed act.
        .init(
            namespace: "authoring", verb: "prompt",
            positional: "<kind>",
            purpose: "Emit the generation prompt for one kind of missing "
                + "study data, with its audit battery as numbers.",
            valueFlags: Set(
                AuthoringPrompts.kinds.flatMap { $0.parameters.map(\.flag) }),
            ownsOutFlag: true),

        // docs — the generator behind the reference document's marked regions
        // (step 11). It reads the same table `--help` renders.
        .init(
            namespace: "docs", verb: "cli-reference",
            purpose: "Regenerate this engine's marked regions in "
                + "docs/CLI-REFERENCE.md, or check them for drift.",
            booleanFlags: ["--check", "--write"], valueFlags: ["--path"]),
    ]

    private static let specsByLabel: [String: ExperimentCLIVerbSpec] = Dictionary(
        specs.map { ($0.label, $0) }, uniquingKeysWith: { a, _ in a })

    /// The spec for a verb, or nil when the verb is not one we declare — an
    /// unknown sub-verb, which the dispatch itself answers with its own verb
    /// list. Flag checking is skipped there on purpose: telling someone their
    /// flag is wrong on a verb that does not exist is the less useful of the
    /// two messages.
    ///
    /// A family that declares a BARE verb (`init`) resolves to it whatever the
    /// first argument is — the family takes no sub-verb, so a leading word is a
    /// stray POSITIONAL and the verb's own `usage:` line is the right answer to
    /// it, not "unknown sub-verb". Every other family behaves exactly as
    /// before.
    public static func spec(namespace: String, verb: String?) -> ExperimentCLIVerbSpec? {
        if let bare = bareSpec(namespace: namespace) { return bare }
        guard let verb else { return nil }
        return specsByLabel["\(namespace) \(verb)"]
    }

    /// The family's sub-verb-less spec, when it declares one. Unambiguous by
    /// construction: a bare spec's label IS its family name, and no two-word
    /// verb can collide with it.
    public static func bareSpec(namespace: String) -> ExperimentCLIVerbSpec? {
        guard let spec = specsByLabel[namespace], spec.verb.isEmpty else {
            return nil
        }
        return spec
    }

    /// Parse one invocation's arguments (everything AFTER the namespace).
    ///
    /// Throws `ExperimentCLIUsageError` for an undeclared flag and nothing
    /// else — a missing positional or a bad flag VALUE stays the verb's own
    /// refusal, so its long-standing `usage:` prose is untouched.
    public static func parse(
        namespace: String, _ args: [String]
    ) throws -> ExperimentCLIInvocation {
        // A bare-verb family has no sub-verb to read: the whole argument list
        // belongs to the one verb.
        let verb =
            bareSpec(namespace: namespace) != nil
            ? nil : args.first.flatMap { $0.hasPrefix("--") ? nil : $0 }

        // `--help` is answered BEFORE anything else is validated, including
        // the verb's own positional requirements: a caller asking what the
        // arguments are must not have to supply them first. It runs nothing,
        // so it can never be the flag that mutates a manifest.
        if args.contains(helpFlag) {
            return ExperimentCLIInvocation(
                namespace: namespace,
                verb: spec(namespace: namespace, verb: verb) == nil ? nil : verb,
                args: verb.map { [$0] } ?? [], json: args.contains(jsonFlag),
                help: true)
        }

        guard let spec = spec(namespace: namespace, verb: verb) else {
            // Unknown or absent sub-verb: keep the arguments intact so the
            // dispatch prints the verb list it always printed, but still
            // honour a bare `--json` so a machine caller gets a document
            // back on the one path it cannot parse (audit §2.2).
            return ExperimentCLIInvocation(
                namespace: namespace, verb: verb, args: args,
                json: args.contains(jsonFlag))
        }

        // A bare-verb spec contributes no sub-verb token, so its dispatch sees
        // its own flags and nothing else.
        var kept: [String] = spec.verb.isEmpty ? [] : [spec.verb]
        var json = false
        var outPath: String?
        var deprecations: [String] = []
        // Two-word verbs start after the sub-verb token; a bare verb's first
        // argument is already its own.
        var index = spec.verb.isEmpty ? 0 : 1

        while index < args.count {
            let token = args[index]
            guard token.hasPrefix("--") else {
                kept.append(token)
                index += 1
                continue
            }
            let next = index + 1 < args.count ? args[index + 1] : nil

            if token == jsonFlag {
                // The historical file form: `--json <path>` on a verb that
                // had it. Left in `kept` so the verb's own reader still finds
                // it, and warned about once.
                if spec.acceptsLegacyJSONPath, let next, !next.hasPrefix("--") {
                    deprecations.append(
                        "warning: `--json <path>` is deprecated on "
                            + "\(namespace) \(spec.verb) — use `--out \(next)`; "
                            + "`--json` alone now means machine output")
                    kept.append(token)
                    kept.append(next)
                    index += 2
                    continue
                }
                json = true
                index += 1
                continue
            }

            if token == outFlag, !spec.ownsOutFlag {
                // `--out` is a declared value flag like any other, so it
                // refuses through the SAME missing-value mechanism (review
                // round 10, finding 5) rather than a bespoke sentence. It
                // kept its own before that mechanism existed. Its CONDITION is
                // deliberately stricter than the general one: the envelope's
                // destination is a file path this parser writes to, so any
                // flag-shaped token there is refused rather than taken at its
                // word.
                guard let next, !next.hasPrefix("--") else {
                    throw missingValueError(
                        flag: token, next: next, spec: spec)
                }
                outPath = next
                index += 2
                continue
            }

            if spec.booleanFlags.contains(token) {
                kept.append(token)
                index += 1
                continue
            }

            if spec.valueFlags.contains(token) {
                // A declared value flag with NO value is a malformed
                // invocation, refused here (review round 10, finding 5).
                //
                // This used to be left to the verb "which already answers with
                // `usage:`" — and the verbs that read their flags through the
                // strict reader do. The verbs that read through the tolerant
                // `flag()` helper see nil instead and fall back to a DEFAULT:
                // `experiment set-sampling <name> --temperature` wrote the
                // sampling protocol at defaults and reported success, which is
                // the flag-that-exits-0-having-done-nothing class this parser
                // exists to close. Refusing at the shared preprocessor fixes it
                // for every verb at once and cannot be forgotten by a new one.
                //
                // TWO shapes are missing values, and only two:
                //   * end of args — `--temperature` was the last token;
                //   * the next token is another DECLARED flag of this verb —
                //     `--temperature --json` ate `--json` as the temperature.
                // Everything else parses, deliberately: an explicit empty
                // token (`--description ""`) is a VALUE, and several verbs
                // accept it as their clear affordance, so it must reach them.
                // A next token that merely looks flag-shaped but is NOT this
                // verb's flag also parses — it is a value the verb will judge
                // (and an undeclared flag typed in a value slot is the value
                // the caller typed, not a flag we may reinterpret).
                guard let next, !isDeclaredFlag(next, spec: spec) else {
                    throw missingValueError(
                        flag: token, next: next, spec: spec)
                }
                kept.append(token)
                kept.append(next)
                index += 2
                continue
            }

            throw ExperimentCLIUsageError(
                flag: token, verb: spec.label,
                declaredFlags: spec.declaredFlags,
                ownedBy: spec.redirectedFlags[token])
        }

        return ExperimentCLIInvocation(
            namespace: namespace, verb: spec.verb.isEmpty ? nil : spec.verb,
            args: kept, json: json, outPath: outPath, deprecations: deprecations)
    }

    /// Whether a token is one of THIS verb's flags — the vocabulary the
    /// missing-value check consults so it can tell `--temperature --json`
    /// (a swallowed flag) from `--temperature --2` (an odd but genuine value).
    ///
    /// `declaredFlags` is the whole surface: the verb's own boolean and value
    /// flags plus the shared `--help`/`--json`/`--out`, which is exactly the
    /// set this loop would otherwise have consumed as flags one token later.
    static func isDeclaredFlag(
        _ token: String, spec: ExperimentCLIVerbSpec
    ) -> Bool {
        token.hasPrefix("--") && spec.declaredFlags.contains(token)
    }

    /// THE missing-value refusal, so `--out` and every other declared value
    /// flag answer the same event with the same sentence. `next` is the token
    /// that stood where the value should have been, or nil at end-of-args.
    static func missingValueError(
        flag: String, next: String?, spec: ExperimentCLIVerbSpec
    ) -> ExperimentCLIUsageError {
        ExperimentCLIUsageError(
            flag: flag, verb: spec.label, declaredFlags: spec.declaredFlags,
            expectedValue: CLIFlagVocabulary.metavar(flag, verb: spec.label),
            followedBy: next,
            usage: ExperimentCLIHelp.synopsis(spec))
    }
}
