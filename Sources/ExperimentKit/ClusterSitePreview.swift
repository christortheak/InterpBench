import Foundation

// =============================================================================
// The WP5 §3.3 preview surface — `docs/WP5-PROFILE-MATERIALIZATION-AUDIT.md`
// Step 5, the deliverable that makes every other step reviewable.
//
// WP5 requires "a UI/CLI preview of the complete environment and scheduler
// commands, so a site admin can read what will run before it runs". Before this
// type the two previews (the site editor and the setup wizard) showed
// `ClusterSiteProfile.environmentExports()` — six keys, none of which any
// engine has ever written. That is not a preview of what runs; it is a preview
// of a function nothing consumes.
//
// `ClusterSitePreview` is the ONE preview model. It is a projection of
// `ClusterEnvironmentRenderer` and adds no facts of its own: every string here
// either comes from the renderer verbatim or is a label naming which renderer
// output it is. Three surfaces read it — the site editor, the setup wizard, and
// `steerlab-cli cluster preview` — so an admin, a researcher, and an agent are
// looking at the SAME bytes.
//
// **Pure and timestamp-free**, for the same reason the renderer is: Step 6
// folds these bytes into the reviewed bootstrap plan hash, and a clock in the
// preview would invalidate every review. Rendering the same profile twice
// yields identical bytes, so a preview diff is a profile diff.
//
// **No secret can appear here.** The bearer token reaches the env file as the
// `$(cat …)` PATH indirection the renderer emits (`executors.py`'s
// `_refuse_secret_env` rule), and this type copies that text rather than
// resolving it. There is no property on it, or on any type it nests, that can
// hold a credential.
// =============================================================================

/// Everything a site admin must be able to read before anything runs: the
/// complete rendered environment file, the `#SBATCH` block for every job class,
/// the scheduler binaries this site invokes, the GPU vocabulary that is emitted,
/// and every fact the profile did NOT state.
///
/// The four panes are WP5 §3.3's enumeration; `schedulerCommands` is its pane 2
/// reduced to what is knowable at Step 5 (the binary NAMES the site will call —
/// full argv composition lands with Steps 6 and 9, and inventing it here would
/// be a second cluster implementation).
public struct ClusterSitePreview: Sendable, Equatable, Encodable {

    // MARK: Nested payloads

    /// One job class's `#SBATCH` block.
    public struct HeaderBlock: Sendable, Equatable, Encodable {
        /// `study` / `controller` / `setup` / `gpuSession`.
        public var jobClass: String
        /// Literal `#SBATCH …` lines, in the renderer's one canonical order.
        public var lines: [String]

        public init(jobClass: String, lines: [String]) {
            self.jobClass = jobClass
            self.lines = lines
        }
    }

    /// One GPU type in the site's vocabulary.
    public struct GPUEntry: Sendable, Equatable, Encodable {
        public var type: String
        /// Absent when the profile declares the type without its VRAM — which
        /// is exactly what makes the memory-fit preflight unable to judge it.
        public var vramGB: Int?

        public init(type: String, vramGB: Int?) {
            self.type = type
            self.vramGB = vramGB
        }
    }

    /// G4: the vocabulary, plus the two env values it renders to, so a reader
    /// can check the emitted string against the inventory that produced it.
    public struct GPUVocabulary: Sendable, Equatable, Encodable {
        public var entries: [GPUEntry]
        /// `STEERLAB_SLURM_GPU_TYPES` value, empty when nothing is emitted.
        public var typesValue: String
        /// `STEERLAB_SLURM_GPU_VRAM` value, empty when nothing is emitted.
        public var vramValue: String
        /// Whether the site declared an inventory at all (as opposed to
        /// inheriting the legacy constant, or emitting nothing).
        public var declared: Bool

        public var isEmpty: Bool { entries.isEmpty }
    }

    /// One scheduler binary this site invokes. NAMES only — a site that wraps
    /// `squeue` says so here, and nothing composes a shell string.
    public struct SchedulerCommand: Sendable, Equatable, Encodable {
        /// `submit` / `query` / `accounting` / `cancel`.
        public var role: String
        public var command: String
    }

    /// A fact the profile did not state, and what happened instead.
    public struct Fact: Sendable, Equatable, Encodable {
        public var key: String
        public var detail: String
    }

    // MARK: Stored

    /// The site's display name, for the pane header.
    public var siteName: String
    /// The PROFILE's schema stamp (not this document's): it is what selects the
    /// default set, so an admin reading a v1 site's preview can see that legacy
    /// defaults are in play. See `ClusterCLIEnvelope.schemaVersion` for the
    /// machine-protocol version of the surrounding document.
    public var schemaVersion: Int
    /// `legacyV1` or `neutralV2`.
    public var defaultSet: String
    /// The same fact as a sentence, so no surface has to invent the wording.
    public var defaultSetSummary: String
    /// Pane 1: the complete rendered env file, verbatim, secrets as `$(cat …)`
    /// indirections exactly as they will appear.
    public var envFile: String
    /// SHA-256 of `envFile`'s exact bytes — the review token of WP5
    /// materialization (Step 6's mechanism, the default path since Step 7).
    ///
    /// The same digest rides in the bootstrap argv as `--env-file-sha256`, so
    /// it is inside the plan hash `bootstrap apply --plan-hash` demands, and
    /// `bootstrap.sh` re-computes it on the far side before installing. A
    /// reviewer who reads this pane can therefore check that the environment
    /// approved here is the environment that landed:
    /// `sha256sum ~/.steerlab/rendered-cluster.env`.
    ///
    /// It is deliberately NOT `ClusterCLIEnvelope.planHash`: that field is the
    /// bootstrap plan hash (argv + profile + this digest), and a preview cannot
    /// compute it — it renders the saved profile and knows nothing about the
    /// provisioning configuration the argv comes from.
    public var envFileSHA256: String
    /// Pane 3: the `#SBATCH` block per job class.
    public var headers: [HeaderBlock]
    /// Pane 2 (reduced): the scheduler binaries, wrappers included.
    public var schedulerCommands: [SchedulerCommand]
    /// The GPU vocabulary and VRAM table that will be emitted.
    public var gpuVocabulary: GPUVocabulary
    /// Pane 4: every field that fell back to a default rather than being said.
    public var unresolvedFacts: [Fact]

    // MARK: Construction

    /// Project a profile through the renderer.
    ///
    /// `jobClasses` narrows pane 3 (`cluster preview --job-class study`); the
    /// default is every class, which is what the committed header golden covers.
    public init(
        _ profile: ClusterSiteProfile,
        jobClasses: [ClusterEnvironmentRenderer.JobClass] =
            ClusterEnvironmentRenderer.JobClass.allCases
    ) {
        let defaultSet = ClusterEnvironmentRenderer.defaultSet(for: profile)
        let vocabulary = ClusterEnvironmentRenderer.gpuVocabulary(for: profile)
        var declaredInventory = false
        var commands: [SchedulerCommand] = []
        if case .slurm(let slurm) = profile.scheduler {
            declaredInventory = !slurm.resolvedGPUs.isEmpty
            commands = [
                SchedulerCommand(role: "submit", command: slurm.commands.submit),
                SchedulerCommand(role: "query", command: slurm.commands.query),
                SchedulerCommand(role: "accounting", command: slurm.commands.accounting),
                SchedulerCommand(role: "cancel", command: slurm.commands.cancel),
            ]
        }
        siteName = profile.name
        schemaVersion = profile.schemaVersion
        self.defaultSet = defaultSet.rawValue
        defaultSetSummary = Self.summary(defaultSet, schemaVersion: profile.schemaVersion)
        envFile = ClusterEnvironmentRenderer.renderEnvFile(profile)
        envFileSHA256 = ClusterEnvironmentRenderer.envFileDigest(profile)
        headers = jobClasses.map { jobClass in
            HeaderBlock(
                jobClass: jobClass.rawValue,
                lines: ClusterEnvironmentRenderer.renderSchedulerHeaders(
                    profile, jobClass: jobClass))
        }
        schedulerCommands = commands
        gpuVocabulary = GPUVocabulary(
            entries: vocabulary.types.map {
                GPUEntry(type: $0, vramGB: vocabulary.vramGB[$0])
            },
            typesValue: vocabulary.typesValue,
            vramValue: vocabulary.vramValue,
            declared: declaredInventory)
        unresolvedFacts = ClusterEnvironmentRenderer.unresolvedFacts(profile).map {
            Fact(key: $0.key, detail: $0.detail)
        }
    }

    /// Which default set applied, and what that MEANS — the sentence WP5 §5
    /// Step 5 requires on every surface, because "legacy defaults are in play"
    /// is not inferable from a version number by a reader who has not read the
    /// audit.
    static func summary(
        _ defaultSet: ClusterEnvironmentRenderer.DefaultSet, schemaVersion: Int
    ) -> String {
        switch defaultSet {
        case .legacyV1:
            return "profile schema \(schemaVersion) — LEGACY defaults: anything this "
                + "profile does not state is filled from today's bootstrap.sh / "
                + "executors.py constants, so materializing this site changes nothing"
        case .neutralV2:
            return "profile schema \(schemaVersion) — NEUTRAL defaults: anything this "
                + "profile does not state is omitted, and the consuming engine's own "
                + "built-in default applies"
        }
    }

    // MARK: Pane documents

    /// Pane 3 as one document, `# --- <class> ---` markers included. Byte-equal
    /// to the committed `<profile>.headers.golden.txt` for an unfiltered
    /// preview — the marker is the fixture's format and this is the one place
    /// that owns it.
    public var headerDocument: String {
        headers
            .map { (["# --- \($0.jobClass) ---"] + $0.lines).joined(separator: "\n") }
            .joined(separator: "\n\n") + "\n"
    }

    /// Pane 4 as one document, `key<TAB>detail` per line (TAB because details
    /// contain colons). Byte-equal to the committed
    /// `<profile>.unresolved.golden.txt`; an empty pane is a lone newline.
    public var unresolvedDocument: String {
        unresolvedFacts.map { "\($0.key)\t\($0.detail)" }.joined(separator: "\n") + "\n"
    }

    /// The GPU pane: the inventory as rows, then the two values the env file
    /// actually carries, so the table and the emitted string are checkable
    /// against each other.
    public var gpuDocument: String {
        guard !gpuVocabulary.isEmpty else {
            return "no GPU vocabulary is emitted — this site declares no inventory\n"
        }
        var lines = gpuVocabulary.entries.map { entry in
            "\(entry.type)\t"
                + (entry.vramGB.map { "\($0) GB" } ?? "VRAM not declared")
        }
        if !gpuVocabulary.declared {
            lines.append("(inherited from the legacy constant — this profile declares none)")
        }
        lines.append("STEERLAB_SLURM_GPU_TYPES=\(gpuVocabulary.typesValue)")
        if gpuVocabulary.vramValue.isEmpty {
            lines.append("STEERLAB_SLURM_GPU_VRAM is not emitted (no VRAM declared)")
        } else {
            lines.append("STEERLAB_SLURM_GPU_VRAM=\(gpuVocabulary.vramValue)")
        }
        return lines.joined(separator: "\n") + "\n"
    }

    /// The scheduler-command pane. Empty for a site with no scheduler.
    public var schedulerCommandDocument: String {
        guard !schedulerCommands.isEmpty else {
            return "no scheduler — this site runs jobs locally\n"
        }
        return schedulerCommands.map { "\($0.role)\t\($0.command)" }
            .joined(separator: "\n") + "\n"
    }

    /// The whole preview as plain text lines, for the CLI's human mode. Pane
    /// bodies are NOT indented: an admin copies the env file out of a terminal,
    /// and leading spaces would corrupt it.
    public var humanLines: [String] {
        // The digest is a header fact, not a pane title: the pane titles are a
        // stable vocabulary callers match on, and the env pane's body must stay
        // copyable verbatim.
        var lines = [
            "  \(defaultSetSummary)",
            "  environment file sha256: \(envFileSHA256)",
        ]
        func pane(_ title: String, _ document: String) {
            lines.append("  --- \(title) ---")
            lines += document.split(separator: "\n", omittingEmptySubsequences: false)
                .dropLast()
                .map(String.init)
        }
        pane("environment file", envFile)
        pane("scheduler headers", headerDocument)
        pane("scheduler commands", schedulerCommandDocument)
        pane("GPU vocabulary", gpuDocument)
        pane(
            unresolvedFacts.isEmpty
                ? "unresolved facts (none — the profile states everything)"
                : "unresolved facts (\(unresolvedFacts.count))",
            unresolvedDocument)
        return lines
    }

    /// One-line summary for a header or an envelope message.
    public var summaryLine: String {
        "\(defaultSet) defaults · \(headers.count) job class(es) · "
            + "\(gpuVocabulary.entries.count) GPU type(s) · "
            + "\(unresolvedFacts.count) unresolved fact(s)"
    }
}
