import Foundation
import SteeringKit

// =============================================================================
// `--help`, rendered from the declarative verb tables (WP0-AGENT-SURFACE-AUDIT
// §5.3, §7 step 11).
//
// `docs/CLI-REFERENCE.md` recorded "Neither CLI has `--help`. This document is
// the substitute" — which, for an agent driving a shipped binary with no
// checkout to read, is the substitution the wrong way round. The declarative
// tables step 5 built as DATA (`ExperimentCLIParser.specs`) and the ones
// `cluster` has had since Phase C (`ClusterCLIVerb`) make the manual free, and
// generating BOTH `--help` and the reference document's marked regions from the
// same table is what makes them incapable of disagreeing.
//
// Everything in this file is CONTRACT TEXT: neutral, imperative, and readable
// by a caller with no prior context. No institution names, no study names, no
// incident history — those belong in the reference document's prose, which this
// generation deliberately leaves alone.
// =============================================================================

/// How one flag is spelled and what it does, shared by `--help` and the
/// generated reference regions.
///
/// A glossary rather than a field on the spec: the same flag means the same
/// thing on twelve verbs, and thirteen copies of a sentence is thirteen places
/// for it to drift. Where a flag genuinely means something else on one verb
/// — `remote fetch --out` is a download DIRECTORY, not the envelope's file —
/// the verb-qualified key wins.
public enum CLIFlagVocabulary {

    /// What a value flag's argument is called. Boolean flags have none.
    private static let metavars: [String: String] = [
        "--agent": "<name-or-path>",
        "--agent-name": "<name>",
        "--alphas": "<a1,a2,…>",
        "--alpha-units": "<norm|raw>",
        "--battery": "<path>",
        "--band-width": "<k>",
        "--bootstrap-partition": "<partition>",
        "--bundle": "<server-path>",
        "--capability-tolerance": "<ratio>",
        "--cell": "<layer>:<alpha>",
        "--choice-prompts": "<path>",
        "--coherence-backstop": "<ratio>",
        "--coherence-floor": "<ratio>",
        "--coherence-ratio": "<ratio>",
        "--concept": "<name>",
        "--control": "<name>",
        "--control-apply-to": "<winner|topK>",
        "--control-margin": "<margin>",
        "--control-top-k": "<k>",
        "--corpus": "<path>",
        "--count": "<n>",
        "--decision": "<text>",
        "--deltas": "<d1,d2>",
        "--description": "<text>",
        "--dev-prompts": "<path>",
        "--env-file": "<path>",
        "--env-prefix": "<path>",
        "--experiment": "<name>",
        "--extraction-rendering": "<json>",
        "--file-slug": "<slug>",
        "--executor": "<local|slurm>",
        "--gres": "<spec>",
        "--hash": "<sha256>",
        "--held-out": "<n>",
        "--home": "<dir>",
        "--job-class": "<class>",
        "--job-id": "<id>",
        "--judges": "<spec>",
        "--layer-fractions": "<f1,f2,…>",
        "--layers": "<L1,L2,…>",
        "--max-tokens": "<n>",
        "--method": "<name>",
        "--model": "<id>",
        "--name": "<name>",
        "--negative": "<text>",
        "--objective": "<metric>",
        "--ordinal-aggregation": "<name>",
        "--out": "<file>",
        "--path": "<server-path>",
        "--payload": "<path>",
        "--plan-hash": "<sha256>",
        "--pool-from": "<k>",
        "--positive": "<text>",
        "--reading-position": "<label>",
        "--python-version": "<version>",
        "--prompt": "<text>",
        "--prompt-mode": "<mode>",
        "--prompts": "<path>",
        "--project-neutral": "<k>",
        "--reason": "<text>",
        "--reference": "<concept>",
        "--remote-repo": "<path>",
        "--revision": "<commit>",
        "--root": "<dir>",
        "--run": "<run-dir>",
        "--run-substrate": "<local|server>",
        "--seat": "<seat>=<agent-artifact-path>",
        "--shape": "<contentPair|singleStimulus>",
        "--sha256": "<hex>",
        "--since": "<date>",
        "--site": "<id>",
        "--slots": "<spec>",
        "--squeue": "<command>",
        "--system": "<text>",
        "--target": "<rung>",
        "--template-id": "<id>",
        "--temperature": "<t>",
        "--threshold": "<ratio>",
        "--token": "<token>",
        "--url": "<server>",
        "--validation-count": "<n>",
        "--variant": "<server-path>",
        "--verb": "<verb>",
        "--walltime": "<hh:mm:ss>",
    ]

    /// Verb-qualified spellings, for the flags whose ARGUMENT differs.
    private static let verbMetavars: [String: String] = [
        "experiment attach --corpus": "<a,b,c>",
        // From the same closed vocabulary the 64-refusal prints, so help and
        // refusal cannot disagree (gate-5 dry run #2, P3).
        "experiment set-instruments --ordinal-aggregation":
            "<" + ExperimentStore.knownOrdinalAggregations.joined(separator: "|") + ">",
        "remote fetch --out": "<dir>",
        "remote import --out": "<dir>",
        "docs cli-reference --path": "<file>",
    ]

    /// One line per flag. Imperative, and about the effect rather than the
    /// history.
    private static let purposes: [String: String] = [
        "--agent": "The promoted agent the policy perturbs.",
        "--agent-name": "Name the minted variant artifact.",
        "--allow-bootstrap": "Authorize the bootstrap step to run.",
        "--allow-controller-start": "Authorize submitting the controller job.",
        "--allow-open-auth-terminal": "Authorize opening the authentication Terminal.",
        "--allow-push": "Authorize deploying the server payload.",
        "--allow-unverified-epoch":
            "Accept a legacy run that carries no experiment-hash stamp.",
        "--alpha-units": "Whether alpha is denominated by the residual-stream norm.",
        "--alphas":
            "The sweep's dose ladder, ascending, in residual-norm units above "
            + "0 (0 is the baseline cell every sweep runs anyway).",
        "--band-width": "Layers per slot.",
        "--battery": "The capability battery every swept cell is scored on.",
        "--baseline": "Declare the explicit no-intervention arm.",
        "--bootstrap-partition": "Scheduler partition the bootstrap job runs in.",
        "--bundle": "The bundle path, as an alternative to the positional.",
        "--capability-tolerance": "Allowed capability drop against baseline.",
        "--cell": "Override the sweep-selected cell, loudly.",
        "--check": "Compare the committed document with the tables; refuse on drift.",
        "--choice-prompts": "The hashed choice-prompt file the objective scores.",
        "--coherence-backstop": "Absolute distinct-2 no cell may fall below.",
        "--coherence-floor": "Absolute distinct-2 floor (the legacy rule).",
        "--coherence-ratio": "Coherence floor as a fraction of the baseline's distinct-2.",
        "--concept": "The concept this data is for; also names its destination.",
        "--control": "Attach a matched control condition to this arm.",
        "--control-apply-to": "Which swept cells the control runs against.",
        "--control-margin": "Margin the winner must beat its control by.",
        "--control-top-k": "How many cells topK covers.",
        "--corpus": "The neutral corpus the norms are measured on.",
        "--count": "How many rows the prompt asks for.",
        "--decision": "The decision each choice row puts to the model.",
        "--deltas": "Perturbation deltas around the anchor cell (default 0.2).",
        "--description": "Free text stored on the manifest.",
        "--dev-prompts": "The dev split the sweep generates on.",
        "--dry-run": "Print what would be submitted and submit nothing.",
        "--env-file": "Path of the environment file the bootstrap step installs.",
        "--env-prefix": "Remote path prefix for the created environment.",
        "--executor": "Where the server runs the job.",
        "--experiment": "The draft study the compiled scenario is pinned into.",
        "--extraction-rendering":
            "How the stimulus reaches the model: '{\"mode\": \"chatTemplate\"}' "
            + "or '{\"mode\": \"raw\"}' (the default, and what absent means). "
            + "A chat-template rendering may add '\"voice\": \"assistant\"' — "
            + "the stimulus as the model's own output — which this engine "
            + "refuses and the python-hf-transformers engine runs.",
        "--file-slug":
            "Name the compiled scenario file (default: the study's name).",
        "--follow": "Keep reading the log as it grows.",
        "--force":
            "Skip the evidence gates and stamp the manifest freezeForced "
            + "(non-citable, by stamp).",
        "--gpu": "Also dispatch one Metal kernel, to prove the shaders load.",
        "--gres": "Scheduler GPU resource request.",
        "--hash": "Expected artifact hash of the variant.",
        "--held-out": "Trailing rows marked split \"test\" — they decide the reader's sign.",
        "--help": "Print this surface and run nothing.",
        "--home": "Materialize the layout here instead of ~/SteerLab.",
        "--job-class": "Narrow the preview to one scheduler job class.",
        "--job-id": "Name the scheduler job explicitly instead of the recorded one.",
        "--json": "Print exactly one machine-readable envelope on stdout.",
        "--judges": "Judge panel: <name>:<kind>[:<model>[:<provider>]], comma-separated.",
        "--layer-fractions":
            "The sweep's layer axis as depths in [0, 1], ascending — the "
            + "portable form, resolved against whatever model is loaded.",
        "--layers":
            "The sweep's layer axis as absolute block indices, ascending — "
            + "converted to depths against the pinned model, which something "
            + "must already have been extracted for.",
        "--materialize-env":
            "Install the env file rendered from the site profile (the default).",
        "--max-tokens": "Maximum tokens to generate (default 512).",
        "--method": "Extraction recipe for the named concepts.",
        "--model": "The model id.",
        "--name": "Names the file the delivered data lands at.",
        "--negative": "What the negative pole IS — a second considered position, never the absence of the first.",
        "--no-control": "Omit the control condition.",
        "--no-materialize-env":
            "Let the bootstrap script synthesize its own env file from built-in "
            + "constants instead.",
        "--objective": "The sweep's selection metric.",
        "--open-auth-terminal": "Authorize opening the authentication Terminal.",
        "--ordinal-aggregation": "How ordinal readouts are aggregated.",
        "--out": "Write the same document to this file.",
        "--path": "The server-side path, as an alternative to the positional.",
        "--payload": "Local directory pushed as the server payload.",
        "--plan-hash": "The reviewed plan this apply is allowed to run.",
        "--pool-from": "Read from token K onward instead of the last token.",
        "--positive": "What the positive pole IS, in a sentence or two.",
        "--reading-position":
            "Where the residual stream is read, by name: 'last content token', "
            + "'content offset 2', 'mean content from token 0', 'offset from "
            + "end 3', … (the legacy spelling of one of them is --pool-from; "
            + "declaring both is refused).",
        "--python-version": "Python version the remote environment is built with.",
        "--project-neutral":
            "Legacy pooled projection — draft-only and verification-blocked.",
        "--prompt": "The user turn to send.",
        "--prompt-mode": "How the prompt is rendered (default chatAssistant).",
        "--prompts": "Override the pinned task-prompt file (pin-checked when frozen).",
        "--reason": "Why the manual cell was chosen; recorded on the certificate.",
        "--redact": "Strip usernames and home paths so the report can be shared.",
        "--redenominate": "Also rewrite the artifact's norm units.",
        "--reference": "designatedReference only: the concept subtracted from the target.",
        "--refresh": "Also run the read-only remote validation before reporting.",
        "--remote-repo": "Remote path the payload is deployed to.",
        "--revision": "Pin the model commit.",
        "--root": "Act on this install root instead of the running binary's own.",
        "--run": "The source run directory; absent, the newest completed run.",
        "--run-substrate": "Which engine's validate and battery evidence the gates read.",
        "--seat":
            "Seat one agent artifact; repeat per seat. Unnamed seats stay baseline.",
        "--sha256": "Expected digest of the downloaded bundle.",
        "--since":
            "Only consider run directories stamped at or after this day or "
            + "instant.",
        "--site": "Name the server through the saved site registry.",
        "--slots": "Arm slots: <concept>:<layer>:<alpha>[:add|ablate], comma-separated.",
        "--squeue": "Command that queries the scheduler's queue.",
        "--shape": "Which reader contrast the rows carry: content pair or single stimulus.",
        "--strip": "Strip the variant's interventions before generating.",
        "--system": "System text prepended to the turn.",
        "--target": "The lifecycle rung to reach (default connected).",
        "--template-id": "The task template the reader fit will use; every row declares it.",
        "--temperature": "Sampling temperature.",
        "--threshold": "Minimum cosine below which the comparison refuses.",
        "--token": "Bearer token for --url; --site reads it from the Keychain instead.",
        "--url": "Name an unmanaged server by URL.",
        "--validation-count": "How many held-out probe rows the prompt asks for.",
        "--variant": "The server-resident variant to generate through.",
        "--verb": "The verb the submitted job runs (defaults to run).",
        "--walltime": "Scheduler wall-time request.",
        "--write": "Rewrite this engine's marked regions in place.",
    ]

    private static let verbPurposes: [String: String] = [
        "experiment attach --corpus": "Extra corpus members for emotionGrandMean.",
        // A compiled panel binds the STUDY's model and sampling settings, so
        // these three default from the target manifest and, when given,
        // overwrite it — the manifest stays the one place they are decided.
        "panel compile --model":
            "Override the study's base model before casting (default: the "
            + "manifest's).",
        "panel compile --temperature":
            "Override the study's temperature before casting (default: the "
            + "manifest's).",
        "panel compile --max-tokens":
            "Override the study's max tokens before casting (default: the "
            + "manifest's).",
        "experiment set-sweep-grid --max-tokens":
            "Tokens generated per swept cell (default 80) — the grid's cost "
            + "multiplier, not the study's generation length.",
        "experiment create --model": "The model this experiment is pinned to (required).",
        "vectors backfill-norms --model": "Load this model for the measurement.",
        "vectors mirror-poles --concept":
            "The mirrored pole's concept name (required) — it must differ from "
            + "the source's, because the negated direction points at the "
            + "opposite pole.",
        "vectors mirror-poles --output-name":
            "File name for the new artifact inside its run directory "
            + "(default: the mirrored concept name).",
        "remote fetch --out": "Download into this directory (default `.`).",
        "remote import --out": "Download into this directory (default .steerlab-downloads).",
        "docs cli-reference --path": "The document to read or rewrite.",
        "cluster controller start --allow-controller-start":
            "Accepted and ignored: typing this verb IS the authorization.",
        "cluster sites export --out": "Write the profile to this file.",
        "cluster sites import --force":
            "Replace a site the registry already holds (default: refuse).",
        "cluster import --dry-run":
            "Print the classification, what would transfer, and the "
            + "purge-eligibility report; transfer and write nothing.",
        "install stamp --revision":
            "Record this source revision in the stamp (the build's git SHA).",
    ]

    /// `--method <name>`, or `--force` for a valueless flag.
    public static func spelling(_ flag: String, verb: String? = nil, takesValue: Bool)
        -> String
    {
        guard takesValue else { return flag }
        return "\(flag) \(metavar(flag, verb: verb))"
    }

    /// The argument's placeholder. `<value>` is the honest fallback for a flag
    /// nobody has glossed yet — never an invented shape.
    public static func metavar(_ flag: String, verb: String? = nil) -> String {
        if let verb, let named = verbMetavars["\(verb) \(flag)"] { return named }
        return metavars[flag] ?? "<value>"
    }

    /// The flag's one line, or empty when it has none.
    public static func purpose(_ flag: String, verb: String? = nil) -> String {
        if let verb, let named = verbPurposes["\(verb) \(flag)"] { return named }
        return purposes[flag] ?? ""
    }
}

/// Renders one verb's (or one family's) declared surface.
public enum ExperimentCLIHelp {

    /// The binary's name in every synopsis line.
    public static let program = "steerlab-cli"

    /// Every family the declarative table covers, in reading order.
    public static let families: [String] = {
        var seen: Set<String> = []
        return ExperimentCLIParser.specs.compactMap { spec in
            seen.insert(spec.namespace).inserted ? spec.namespace : nil
        }
    }()

    /// The verbs of one family, in table order.
    public static func specs(inFamily family: String) -> [ExperimentCLIVerbSpec] {
        ExperimentCLIParser.specs.filter { $0.namespace == family }
    }

    /// `steerlab-cli experiment attach <name> <concept>… [--method <name>] …` —
    /// the one-line synopsis, flags in sorted order so the rendering is stable.
    public static func synopsis(
        _ spec: ExperimentCLIVerbSpec, includeProgram: Bool = true,
        includeSharedFlags: Bool = true
    ) -> String {
        var parts: [String] = []
        if includeProgram { parts.append(program) }
        parts.append(spec.label)
        if !spec.positional.isEmpty { parts.append(spec.positional) }
        for flag in orderedFlags(spec) {
            // The reference document lists the three shared agent flags once
            // per region rather than on all thirty-nine lines; `--help` prints
            // them, because a caller reading one page has no region note.
            if !includeSharedFlags, Self.sharedFlags.contains(flag),
                !(flag == ExperimentCLIParser.outFlag && spec.ownsOutFlag)
            {
                continue
            }
            let takesValue = takesValue(flag, spec: spec)
            let spelling = CLIFlagVocabulary.spelling(
                flag, verb: spec.label, takesValue: takesValue)
            parts.append(
                spec.requiredFlags.contains(flag) ? spelling : "[\(spelling)]")
        }
        return parts.joined(separator: " ")
    }

    /// The flags every agent-path verb carries.
    public static let sharedFlags: Set<String> = [
        ExperimentCLIParser.helpFlag, ExperimentCLIParser.jsonFlag,
        ExperimentCLIParser.outFlag,
    ]

    /// Every flag the verb accepts, sorted, `--help` included. The shared agent
    /// flags are part of the surface: a caller cannot be expected to know that
    /// `--json` is universal but `--out` is not.
    public static func orderedFlags(_ spec: ExperimentCLIVerbSpec) -> [String] {
        spec.declaredFlags
    }

    public static func takesValue(_ flag: String, spec: ExperimentCLIVerbSpec) -> Bool {
        if spec.valueFlags.contains(flag) { return true }
        if spec.booleanFlags.contains(flag) { return false }
        return flag == ExperimentCLIParser.outFlag
    }

    /// One verb's `--help` page.
    public static func text(for spec: ExperimentCLIVerbSpec) -> String {
        var lines: [String] = ["usage: " + synopsis(spec)]
        if !spec.purpose.isEmpty { lines.append(""); lines.append(spec.purpose) }
        let flags = orderedFlags(spec)
        if !flags.isEmpty {
            lines.append("")
            lines.append("flags:")
            let spellings = flags.map {
                CLIFlagVocabulary.spelling(
                    $0, verb: spec.label, takesValue: takesValue($0, spec: spec))
            }
            let width = min(spellings.map(\.count).max() ?? 0, 34)
            for (spelling, flag) in zip(spellings, flags) {
                var purpose = CLIFlagVocabulary.purpose(flag, verb: spec.label)
                if spec.requiredFlags.contains(flag) {
                    purpose = purpose.isEmpty
                        ? "Required." : purpose + "  (required)"
                }
                if purpose.isEmpty {
                    lines.append("  " + spelling)
                } else {
                    let padding = String(
                        repeating: " ", count: max(2, width + 2 - spelling.count))
                    lines.append("  " + spelling + padding + purpose)
                }
            }
        }
        lines.append("")
        lines.append(exitCodeLine)
        return lines.joined(separator: "\n") + "\n"
    }

    /// One family's `--help` page: every verb it owns, with its purpose.
    ///
    /// A family that IS one verb (`init`) has no roster to print — the verb
    /// page is the family page, and printing a one-row table with an empty
    /// verb name would be a worse answer to the same question.
    public static func familyText(for family: String) -> String {
        if let bare = ExperimentCLIParser.bareSpec(namespace: family) {
            return text(for: bare)
        }
        let specs = specs(inFamily: family)
        guard !specs.isEmpty else {
            return "usage: \(program) <verb> …\n  families: "
                + families.joined(separator: " | ") + "\n"
        }
        var lines = ["usage: \(program) \(family) <verb> …", ""]
        let width = min(specs.map { $0.verb.count }.max() ?? 0, 22)
        for spec in specs {
            let padding = String(
                repeating: " ", count: max(2, width + 2 - spec.verb.count))
            lines.append("  " + spec.verb + padding + spec.purpose)
        }
        lines.append("")
        lines.append(
            "\(program) \(family) <verb> --help prints one verb's arguments.")
        lines.append(exitCodeLine)
        return lines.joined(separator: "\n") + "\n"
    }

    // MARK: The top-level page

    /// One line of `steerlab-cli --help`: how the entry is typed and what it
    /// is for. A table rather than a string literal because the audit's drift
    /// finding D8 WAS that literal — it omitted `panel` and `vectors`, and an
    /// unlisted family is indistinguishable from an absent one to a caller
    /// with no checkout to read.
    public struct TopLevelEntry: Sendable {
        public let synopsis: String
        public let purpose: String

        public init(synopsis: String, purpose: String) {
            self.synopsis = synopsis
            self.purpose = purpose
        }
    }

    /// Global flags: the ones that are read before any verb sees the argument
    /// list, and so belong to no family.
    public static let globalFlags: [TopLevelEntry] = [
        .init(
            synopsis: "--workspace <dir>",
            purpose: "Data workspace root (precedence: STEERLAB_WORKSPACE env > "
                + "--workspace > app-persisted choice > the code checkout)."),
        .init(
            synopsis: "--version",
            purpose: "Print this build's version and install layout, and run "
                + "nothing (the same report as `install version`)."),
        .init(
            synopsis: "--help",
            purpose: "Print this page. `<family> --help` lists a family's "
                + "verbs; `<family> <verb> --help` prints one verb's arguments."),
        .init(
            synopsis: "--json",
            purpose: "Declared on every agent-path verb: exactly one envelope "
                + "on stdout, every diagnostic on stderr."),
    ]

    /// Every family the binary dispatches, in reading order. The families the
    /// declarative table covers are generated from it; the three that live
    /// outside it — `cluster` has its own table, and `artifacts`, `serve`,
    /// `--config` are not on the agent path — are declared here, and
    /// `topLevelNamesEveryDispatchedFamily` is what stops that list rotting.
    /// (`panel` joined the table with `panel compile`, open-issues §18.)
    public static let topLevelEntries: [TopLevelEntry] = [
            .init(
                synopsis: "init [--home <dir>]",
                purpose: "Create the home layout's Workspaces/ and Sites/."),
            .init(
                synopsis: "workspace init <path>",
                purpose: "Create and seed a data workspace."),
            .init(
                synopsis: "experiment <verb> <name> …",
                purpose: "The study lifecycle."),
            .init(
                synopsis: "data check <experiment>",
                purpose: "Study-data readiness."),
            .init(
                synopsis: "vectors <verb> …", purpose: "Vector artifacts."),
            .init(
                synopsis: "remote <verb> (--site <id> | --url <server>)",
                purpose: "Cluster client."),
            .init(synopsis: "cluster <verb> …", purpose: "Cluster lifecycle."),
            .init(
                synopsis: "install version | stamp | verify",
                purpose: "This build's identity and the integrity of its "
                    + "install."),
            .init(
                synopsis: "authoring prompt <kind> …",
                purpose: "Generation prompts for missing study data."),
            .init(
                synopsis: "docs cli-reference [--check | --write]",
                purpose: "Regenerate the reference document."),
            .init(
                synopsis: "panel <verb> …",
                purpose: "Panel scenarios and seat casting."),
            .init(
                synopsis: "artifacts audit [--json]",
                purpose: "Vector-sidecar audit."),
            .init(
                synopsis: "serve [--port N]",
                purpose: "The loopback web front end."),
        .init(
            synopsis: "--config <path.json>",
            purpose: "Smoke-test / toy-concept tasks."),
    ]

    /// `steerlab-cli --help`, and the body of the generated `swift-global`
    /// region.
    public static var topLevelText: String {
        var lines = [
            "usage: \(program) [--workspace <dir>] <family> <verb> … "
                + "[--help] [--json]",
            "",
        ]
        let width = min(
            topLevelEntries.map(\.synopsis.count).max() ?? 0, 44)
        for entry in topLevelEntries {
            let padding = String(
                repeating: " ", count: max(2, width + 2 - entry.synopsis.count))
            lines.append("  " + entry.synopsis + padding + entry.purpose)
        }
        lines.append("")
        lines.append("global flags:")
        let flagWidth = min(globalFlags.map(\.synopsis.count).max() ?? 0, 22)
        for flag in globalFlags {
            let padding = String(
                repeating: " ", count: max(2, flagWidth + 2 - flag.synopsis.count))
            lines.append("  " + flag.synopsis + padding + flag.purpose)
        }
        lines.append("")
        lines.append(exitCodeLine)
        return lines.joined(separator: "\n") + "\n"
    }

    /// The state vocabulary, in one line, on every page — an agent reading a
    /// help page is about to read an exit code.
    public static let exitCodeLine =
        "exit codes: 0 ok · 64 malformed invocation · 65 refused · 66 not found "
        + "· 70 failed  (--json: the envelope's `state` is authoritative)"

    /// The same page as data, for `--json`. A machine caller should not have to
    /// parse the columns a human reads.
    public static func payload(for spec: ExperimentCLIVerbSpec) -> [String: JSONValue] {
        var flags: [JSONValue] = []
        for flag in orderedFlags(spec) {
            let takesValue = takesValue(flag, spec: spec)
            var entry: [String: JSONValue] = [
                "flag": .string(flag),
                "takesValue": .bool(takesValue),
                "purpose": .string(CLIFlagVocabulary.purpose(flag, verb: spec.label)),
            ]
            if takesValue {
                entry["value"] = .string(
                    CLIFlagVocabulary.metavar(flag, verb: spec.label))
            }
            flags.append(.object(entry))
        }
        return [
            "verb": .string(spec.label),
            "purpose": .string(spec.purpose),
            "positional": .string(spec.positional),
            "synopsis": .string(synopsis(spec)),
            "flags": .array(flags),
        ]
    }

    /// The family page as data.
    public static func familyPayload(for family: String) -> [String: JSONValue] {
        if let bare = ExperimentCLIParser.bareSpec(namespace: family) {
            return payload(for: bare)
        }
        return [
            "family": .string(family),
            "verbs": .array(
                specs(inFamily: family).map { spec in
                    .object([
                        "verb": .string(spec.label),
                        "purpose": .string(spec.purpose),
                        "synopsis": .string(synopsis(spec)),
                    ])
                }),
        ]
    }
}
