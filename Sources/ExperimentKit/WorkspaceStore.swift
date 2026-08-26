import Foundation
import Observation

// The workspace split (decision of record): the app stops using its source
// checkout as the data root. A *workspace* is a plain folder containing
// prompts/, experiments/, runs/ — created/opened by the app (or
// `steerlab-cli workspace init`), git-managed invisibly. What a new
// workspace is BORN with comes from one curated tree — `WorkspaceSeed/`
// (WP1), resolved through `CodeResources.workspaceSeed()` — through the
// explicit file allowlist `seedManifest` below. The research checkout's own
// `prompts/` is no longer a seed source; it is study/template data and test
// fixtures. The substrate switcher is called "Compute" in the UI, so
// "Workspace" unambiguously means the data folder.

/// Runtime resolution of the workspace root every artifact path funnels
/// through (`VectorCatalog.projectRoot`). Precedence, exactly:
///
/// 1. `STEERLAB_WORKSPACE` environment variable
/// 2. the process-wide programmatic override (CLI `--workspace`, the app's
///    workspace manager)
/// 3. UserDefaults key `"SteerLabWorkspaceRoot"` (the app's persisted choice
///    — honored only while the directory actually exists, so a deleted
///    workspace degrades to the dev fallback instead of a dangling root)
/// 4. the legacy repo root (the compiled-in checkout path,
///    `CodeResources.compiledCheckoutPath` via its `bundledSeedRoot` alias) —
///    the dev/test fallback, so every existing test and dev flow is
///    byte-identical when nothing is configured.
public enum WorkspaceRoot {
    public static let environmentKey = "STEERLAB_WORKSPACE"
    public static let defaultsKey = "SteerLabWorkspaceRoot"

    /// Precedence #2. nonisolated(unsafe) is justified the same way as
    /// `ExperimentStore.rootOverride`: written once at CLI startup (before
    /// any verb runs) or from the main thread by the app's workspace manager
    /// before catalogs re-scan — never raced against readers by construction.
    nonisolated(unsafe) public static var programmaticOverride: URL?

    /// Precedence #1, read once — a process's environment cannot change.
    private static let environmentValue: String? =
        ProcessInfo.processInfo.environment[environmentKey]

    /// True when the root is pinned by `STEERLAB_WORKSPACE` — the app's
    /// switcher surfaces this instead of pretending a switch would stick.
    public static var isEnvironmentPinned: Bool {
        !(environmentValue ?? "").trimmingCharacters(in: .whitespaces).isEmpty
    }

    /// Pure, injectable resolution (the testable seam: environment dict +
    /// persisted-defaults value arrive as data; no process state).
    public static func resolve(
        environment: [String: String],
        programmaticOverride: URL?,
        persistedPath: String?,
        fallback: URL
    ) -> URL {
        if let env = environment[environmentKey]?
            .trimmingCharacters(in: .whitespacesAndNewlines), !env.isEmpty
        {
            return URL(filePath: env).standardizedFileURL
        }
        if let programmaticOverride {
            return programmaticOverride.standardizedFileURL
        }
        if let persistedPath, !persistedPath.isEmpty {
            var isDirectory: ObjCBool = false
            if FileManager.default.fileExists(
                atPath: persistedPath, isDirectory: &isDirectory),
                isDirectory.boolValue
            {
                return URL(filePath: persistedPath).standardizedFileURL
            }
        }
        return fallback
    }

    /// The live workspace root. UserDefaults is read per call (cheap, and the
    /// app can change it); the environment is cached.
    public static var current: URL {
        resolve(
            environment: environmentValue.map { [environmentKey: $0] } ?? [:],
            programmaticOverride: programmaticOverride,
            persistedPath: UserDefaults.standard.string(forKey: defaultsKey),
            fallback: VectorCatalog.bundledSeedRoot)
    }
}

/// Workspace lifecycle: create (make dirs + seed from the code repo + git
/// init), open (validate + persist + set the runtime override), and the
/// app-facing observable wrapper the toolbar switcher binds to. The static
/// core is nonisolated so the CLI and tests use it without an actor hop.
@MainActor @Observable
public final class WorkspaceStore {

    // MARK: Static core (CLI + app + tests)

    /// Every prompts/ subdirectory a fresh workspace carries, populated or not.
    /// `taxonomies` is here because `experiment set-style-taxonomy` names
    /// `prompts/taxonomies/<file>.json` in its own usage and refusals
    /// (`ExperimentStore.taxonomiesRelativeDirectory`), and seeding never
    /// created it — the verb promised a directory that did not exist.
    public nonisolated static let promptSubdirectories = [
        "concepts", "emotions", "rubrics", "batteries", "neutral", "templates",
        "tasks", "readers", "probes", "dev", "generation", "parsers", "panels",
        "taxonomies",
    ]

    /// THE SEED ALLOWLIST (WP1): every file a new workspace receives, named
    /// one by one, relative to the seed root (`CodeResources.workspaceSeed()`
    /// — `WorkspaceSeed/` in the checkout, `WorkspaceSeed/` in a bundle).
    ///
    /// This replaced a directory SWEEP minus an exclusion list. The sweep's
    /// boundary was a subtraction: in developer mode the seed root *was* the
    /// research checkout, so every file the researcher added under
    /// `prompts/` seeded every new workspace unless someone remembered to
    /// exclude it — and what leaked was not neutral instrument data but this
    /// study's own material (a coding rubric naming its source authors,
    /// authoring notes naming case families, an appellate protocol carrying
    /// the study it was minted from, demo concepts, a starter pack). An
    /// allowlist inverts the default: an unlisted file is absent, and adding
    /// one is a deliberate edit here, reviewed as code.
    ///
    /// The rules this list keeps:
    /// - **Instruments and templates only.** No concepts, probes, panels,
    ///   cases, emotion corpora, or authoring job cards. A fresh workspace
    ///   starts CONCEPT-EMPTY; `SampleWorkspace/` is where a worked example
    ///   lives, and it is a separate folder the user opens on purpose.
    /// - **Neutral bytes.** Nothing here names a study, an institution, a
    ///   person, or a task domain. `AgentContractTests` walks
    ///   `WorkspaceSeed/` against the private name denylist, and asserts
    ///   this list and the tree are the same set — no stowaways.
    /// - **Engine defaults resolve.** Several paths below are hard-coded
    ///   defaults in both engines (`prompts/batteries/basic.jsonl`,
    ///   `prompts/dev/dev-prompts.jsonl`,
    ///   `prompts/dev/robustness-coherence.jsonl`,
    ///   `prompts/neutral/corpus.jsonl`,
    ///   `prompts/parsers/parser-registry.json`, and the batteries the
    ///   robustness presets name), so a workspace without them would have
    ///   defaults pointing at nothing.
    ///
    /// Nothing is removed from the repository — existing workspaces keep
    /// every file they were ever given. Only NEW workspaces change.
    public nonisolated static let seedManifest: [String] = [
        // Capability probes. `basic.jsonl` is the engine default; the rest
        // are the sets `VariantRobustness` presets name.
        "prompts/batteries/basic.jsonl",
        "prompts/batteries/factual-short.jsonl",
        "prompts/batteries/instruction-following.jsonl",
        "prompts/batteries/reasoning-small.jsonl",
        "prompts/batteries/study-guardrail.jsonl",
        "prompts/batteries/truthfulness-small.jsonl",
        // Sweep dev split + robustness/coherence prompts (both engine
        // defaults; generic, concept-content-free).
        "prompts/dev/dev-prompts.jsonl",
        "prompts/dev/robustness-coherence.jsonl",
        // Dataset-generation prompt templates. Both engines RENDER these
        // files by name (`ConceptBuilder.templatePrompt`,
        // `authoring.TEMPLATE_FILES`), resolving the workspace's copy first
        // and the seed copy second — a workspace without them makes the
        // Concept Lab's "copy LLM prompt" buttons and the server's
        // `authoring template` endpoint fail. The study's own authoring job
        // cards are NOT here; only the seven the code names, plus a README
        // written for this list.
        "prompts/generation/README.md",
        "prompts/generation/caa-paired-stimuli.md",
        "prompts/generation/emotion-grand-mean-stories.md",
        "prompts/generation/grand-mean-cowork-agent.md",
        "prompts/generation/neutral-dialogues-anthropic-style.md",
        "prompts/generation/neutral-norm-corpus.md",
        "prompts/generation/probe-validation-items.md",
        "prompts/generation/repe-paired-reader-data.md",
        // The residual-norm denominator corpus — a fixed denominator is what
        // makes α comparable across concepts.
        "prompts/neutral/corpus.jsonl",
        // Declared numeric unit grammars; its default entry reproduces the
        // built-in parser, so `numericParser` is nameable on day one.
        "prompts/parsers/parser-registry.json",
        // Judge rubrics: the pin convention, and the historical default
        // criterion (pinnable so pre-versioning behavior is reproducible).
        "prompts/rubrics/README.md",
        "prompts/rubrics/default-paired-v1.md",
        // A worked task-prompts file showing `responseFormat` (which
        // instruments may legitimately read an item) and `options`/`target`.
        "prompts/tasks/example-task-prompts.jsonl",
        // Templates: the shapes every study-data scaffold copies from
        // (`StudyDataReadiness.Template.seedRelativePath` names several of
        // these exactly), plus the two RepE reader task templates the
        // Concept Lab's registry picker scans.
        "prompts/templates/amount-in-scenario-v1.json",
        "prompts/templates/unnamed-scenario-v1.json",
        "prompts/templates/battery/capability-battery-v2.template.jsonl",
        "prompts/templates/human-baseline/README.md",
        "prompts/templates/human-baseline/human-baseline-template.csv",
        "prompts/templates/markers/README.md",
        "prompts/templates/markers/markers-template.json",
        "prompts/templates/reasoning-style/README.md",
        "prompts/templates/reasoning-style/reasoning-style-generic-template.json",
        "prompts/templates/reasoning-style/reasoning-style-structure-template.json",
        "prompts/templates/rubrics/README.md",
        "prompts/templates/rubrics/rubric-template.md",
        "prompts/templates/scenario/README.md",
        "prompts/templates/scenario/scenario-template.json",
        "prompts/templates/task-prompts-choice/README.md",
        "prompts/templates/task-prompts-choice/task-prompts-choice-template.jsonl",
        "prompts/templates/task-prompts-transcript/README.md",
        "prompts/templates/task-prompts-transcript/task-prompts-transcript-template.jsonl",
        "prompts/templates/validation/README.md",
        "prompts/templates/validation/validation-template.jsonl",
    ]

    public nonisolated static let markerFileName = "WORKSPACE.md"

    /// A folder is a workspace when it carries the marker or at least a
    /// prompts/ directory (pre-marker folders someone assembled by hand).
    public nonisolated static func isWorkspace(url: URL) -> Bool {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: url.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else { return false }
        if fm.fileExists(atPath: url.appending(component: markerFileName).path) {
            return true
        }
        var promptsIsDirectory: ObjCBool = false
        return fm.fileExists(
            atPath: url.appending(component: "prompts").path,
            isDirectory: &promptsIsDirectory) && promptsIsDirectory.boolValue
    }

    /// Creates a workspace at `url`: the full directory skeleton, the
    /// `seedManifest` files copied from the shipped seed tree
    /// (`CodeResources.workspaceSeed()` — `WorkspaceSeed/` in the checkout,
    /// `WorkspaceSeed/` in a packaged build; pass `seedingFrom:` to
    /// override — the workspace is CONCEPT-EMPTY), a WORKSPACE.md
    /// marker, and a git repo with an initial commit (silent no-op when git
    /// is unavailable). Does NOT switch the running process to it —
    /// `open(at:)` does that. Throws the locator's plain-language error when
    /// no seed source exists (a release build missing its bundled seed data
    /// fails closed rather than minting an unseeded workspace).
    @discardableResult
    public nonisolated static func create(
        at url: URL,
        seedingFrom seedRootOverride: URL? = nil
    ) throws -> URL {
        let fm = FileManager.default
        let root = url.standardizedFileURL
        guard !isWorkspace(url: root) else {
            throw ExperimentError(
                reason: "\(root.path) is already a SteerLab workspace")
        }
        // Preflight: a non-empty destination is refused rather than merged
        // into. Creation is not an import verb, and merging is how a
        // half-built workspace used to be indistinguishable from a real one.
        if fm.fileExists(atPath: root.path) {
            let entries = (try? fm.contentsOfDirectory(atPath: root.path))?
                .filter { $0 != ".DS_Store" } ?? []
            guard entries.isEmpty else {
                throw ExperimentError(
                    reason: "\(root.path) already exists and is not empty — "
                        + "create the workspace in a new or empty folder")
            }
        }
        let seedRoot = try seedRootOverride ?? CodeResources.workspaceSeed()

        // STAGE AND INSTALL. Building in place meant any mid-copy failure —
        // an unreadable seed file, a full disk — left a directory that
        // `isWorkspace` accepts (a `prompts/` directory is the whole test)
        // but that is missing an arbitrary suffix of the seed. The
        // researcher's next verb then read a workspace that looked fine and
        // was not. Instead: build a complete tree in a temp SIBLING (same
        // volume, so the install is a rename), install it with one atomic
        // move, and on ANY failure remove the staging tree and leave the
        // destination absent. Half a workspace is never observable.
        try fm.createDirectory(
            at: root.deletingLastPathComponent(), withIntermediateDirectories: true)
        let staging = root.deletingLastPathComponent().appending(
            component: ".\(root.lastPathComponent).steerlab-staging-\(UUID().uuidString)")
        do {
            try seedWorkspace(at: staging, from: seedRoot)
            // The vetted destination is empty or absent; either way it must
            // be gone for the rename to land the staged tree whole.
            if fm.fileExists(atPath: root.path) { try fm.removeItem(at: root) }
            try fm.moveItem(at: staging, to: root)
        } catch {
            try? fm.removeItem(at: staging)
            throw error
        }
        return root
    }

    /// Builds a complete workspace at `root`, which must not yet exist. Every
    /// failure propagates: the caller discards the whole tree, so there is no
    /// partial state to reason about here.
    private nonisolated static func seedWorkspace(
        at root: URL, from seedRoot: URL
    ) throws {
        let fm = FileManager.default

        for sub in promptSubdirectories {
            try fm.createDirectory(
                at: root.appending(components: "prompts", sub),
                withIntermediateDirectories: true)
        }
        try fm.createDirectory(
            at: root.appending(component: "experiments"),
            withIntermediateDirectories: true)
        try fm.createDirectory(
            at: root.appending(component: "runs"), withIntermediateDirectories: true)
        // Top-level home for per-adapter training data + outputs (like
        // runs/, not under prompts/). Existing workspaces get it lazily the
        // first time the adapter UI needs it (`FineTuneStore.createAdapterHome`).
        try fm.createDirectory(
            at: root.appending(component: "adapters"),
            withIntermediateDirectories: true)

        // Seed: exactly the files `seedManifest` names, at the same relative
        // paths. A seed root that lacks one is skipped silently (a partial
        // staged bundle degrades to a thinner workspace, never to a crash);
        // an existing destination is never overwritten.
        for relative in seedManifest {
            try copySeedFile(relative, from: seedRoot, into: root)
        }

        try markerContents().write(
            to: root.appending(component: markerFileName), atomically: true,
            encoding: .utf8)
        // The agent contract, written before `git init` so it is in the
        // initial commit. Generated, never seeded (audit §4.3).
        try agentContractContents().write(
            to: root.appending(component: AgentContract.fileName), atomically: true,
            encoding: .utf8)
        // runs/ are immutable bulk outputs; versioning them would make every
        // freeze auto-commit enormous. The pinned/ snapshot inside each
        // frozen experiment is the reproducibility floor for run inputs.
        // adapters/ keeps its (researcher-authored, provenance-relevant)
        // training data versioned but excludes trained weight binaries —
        // the same bulk-output rationale as runs/.
        // `catalog/` is a GENERATED symlink forest over runs/ (open-issues
        // §20): navigation only, rebuilt idempotently after every import, and
        // nothing in it is paper-relevant. Committing it would put a tree of
        // dangling-by-design links into every freeze auto-commit.
        // `WorkspaceRunCatalog.ensureGitignored` adds the line to workspaces
        // made before this rule; here it is present from the first commit.
        try "runs/\ncatalog/\nadapters/**/*.safetensors\n.DS_Store\n".write(
            to: root.appending(component: ".gitignore"), atomically: true,
            encoding: .utf8)

        initializeGit(at: root)
    }

    /// Switches the process to a workspace: validates it, persists the choice
    /// (UserDefaults), and sets the runtime override. `defaults` and
    /// `setOverride` are injection seams so tests never mutate process-global
    /// state; production callers use the defaults.
    @discardableResult
    public nonisolated static func open(
        at url: URL,
        defaults: UserDefaults = .standard,
        setOverride: (URL) -> Void = { WorkspaceRoot.programmaticOverride = $0 }
    ) throws -> AgentContractUpkeep {
        let root = url.standardizedFileURL
        guard isWorkspace(url: root) else {
            throw ExperimentError(
                reason: "\(root.path) does not look like a SteerLab workspace "
                    + "(no prompts/ directory or \(markerFileName) marker)")
        }
        let upkeep = upkeepAgentContract(at: root)
        defaults.set(root.path, forKey: WorkspaceRoot.defaultsKey)
        setOverride(root)
        // Returned rather than announced: `switchTo` is the surface that
        // speaks, and handing the outcome up is what keeps it from having to
        // classify a file this call has already rewritten.
        return upkeep
    }

    /// What one pass of contract upkeep DID, and therefore what the surface
    /// that asked for it should say. Both the app and the CLI drive the same
    /// pass and read the same answer, so the two cannot drift.
    public enum AgentContractUpkeep: Sendable, Equatable {

        /// Nothing to do, or nothing we are allowed to do: the contract is
        /// current, or the file is the researcher's and stays theirs. Silent.
        case unchanged

        /// The file was absent and has been written. Silent by design — a
        /// missing machine file is trivially safe to write and there is
        /// nothing for anyone to repair.
        case regenerated

        /// A file whose hashed header PROVED it machine-written and unedited
        /// was rewritten to this build's contract. One notice, once per open.
        case refreshed(linesChanged: Int)

        /// A file with a pre-hash header is behind, and this build will not
        /// touch it: the header is a heuristic, not a proof. One advisory,
        /// once per open, naming the manual regeneration that graduates the
        /// file into the hashed regime.
        case legacyStale(linesBehind: Int)

        /// The ONE sentence this outcome is worth, or nil when it is silent.
        /// Prefix-free — the CLI stamps its own label, the app carries a
        /// severity.
        public func sentence(at root: URL) -> String? {
            switch self {
            case .unchanged, .regenerated:
                return nil
            case .refreshed(let linesChanged):
                return AgentContract.refreshNotice(
                    linesChanged: linesChanged, at: root)
            case .legacyStale(let linesBehind):
                return AgentContract.stalenessAdvisory(
                    for: .staleUnedited(linesBehind: linesBehind), at: root)
            }
        }

        /// How loud the app's notice feed should be. A refresh is a report of
        /// completed work; the legacy advisory asks for a gesture.
        public var severity: PanelNotice.Severity {
            if case .legacyStale = self { return .warning }
            return .info
        }
    }

    /// Classify this workspace's `AGENTS.md`, then do the one thing that
    /// classification permits.
    ///
    /// Two outcomes write, and only two:
    ///
    /// - **absent** → write the contract. `create` runs once and thousands of
    ///   workspace-days predate the contract, so opening an older workspace is
    ///   what hands it the file (audit §4.3).
    /// - **`staleProven`** → rewrite it, atomically, in place. The header
    ///   carries a SHA-256 of the body it wrote and that hash still matches,
    ///   so the file provably holds no human text: the contract is
    ///   documentation, it alters no run, and keeping it current is what the
    ///   researcher would have done by hand.
    /// - everything else → **nothing**. `current` needs no work;
    ///   `staleUnedited` is a pre-hash file whose header is a heuristic rather
    ///   than a proof, so it gets the advisory and keeps its bytes; `edited`
    ///   is the researcher's file and is never touched on any path.
    ///
    /// Failures are swallowed: a read-only or otherwise unwritable workspace
    /// must still open, and a contract refresh is the last thing that should
    /// stop it.
    @discardableResult
    public nonisolated static func upkeepAgentContract(at root: URL) -> AgentContractUpkeep {
        let url = root.appending(component: AgentContract.fileName)
        switch AgentContract.status(at: root) {
        case .absent:
            do {
                try agentContractContents().write(
                    to: url, atomically: true, encoding: .utf8)
                return .regenerated
            } catch {
                return .unchanged
            }
        case .staleProven:
            // Counted over the BODIES, not the files: the header line always
            // differs (its hash moved with the text), and reporting that as
            // two changed lines would inflate every count by two.
            let previous = AgentContract.bodyText(
                of: (try? String(contentsOf: url, encoding: .utf8)) ?? "")
            do {
                try atomicallyReplace(url, with: agentContractContents())
            } catch {
                return .unchanged
            }
            return .refreshed(
                linesChanged: AgentContract.changedLineCount(
                    from: previous, to: AgentContract.body))
        case .staleUnedited(let linesBehind):
            return .legacyStale(linesBehind: linesBehind)
        case .current, .edited:
            return .unchanged
        }
    }

    /// Writes `AGENTS.md` if the workspace has none — the absent-only half of
    /// `upkeepAgentContract`, kept as its own verb for the callers that want
    /// exactly that and nothing else. Returns whether it wrote.
    @discardableResult
    public nonisolated static func ensureAgentContract(at root: URL) -> Bool {
        let url = root.appending(component: AgentContract.fileName)
        guard !FileManager.default.fileExists(atPath: url.path) else { return false }
        do {
            try agentContractContents().write(to: url, atomically: true, encoding: .utf8)
            return true
        } catch {
            return false
        }
    }

    /// Replace an EXISTING file's contents without a window in which the path
    /// holds a half-written contract: a temp file beside it (same directory,
    /// so the rename cannot cross a filesystem), then an atomic exchange. A
    /// failure anywhere leaves the original exactly as it was.
    private nonisolated static func atomicallyReplace(
        _ url: URL, with text: String
    ) throws {
        let fm = FileManager.default
        let staging = url.deletingLastPathComponent()
            .appending(component: ".\(url.lastPathComponent).\(UUID().uuidString).tmp")
        do {
            try Data(text.utf8).write(to: staging, options: .atomic)
            _ = try fm.replaceItemAt(url, withItemAt: staging)
        } catch {
            try? fm.removeItem(at: staging)
            throw error
        }
    }

    /// Copies ONE manifest entry, seed-root-relative to workspace-relative.
    /// Absent at the source = silent skip; already present at the
    /// destination = never overwritten (the historical seed rule, and what
    /// makes re-seeding an existing tree harmless).
    private nonisolated static func copySeedFile(
        _ relative: String, from seedRoot: URL, into workspace: URL
    ) throws {
        let fm = FileManager.default
        let source = seedRoot.appending(path: relative)
        guard fm.fileExists(atPath: source.path) else { return }
        let destination = workspace.appending(path: relative)
        guard !fm.fileExists(atPath: destination.path) else { return }
        try fm.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try fm.copyItem(at: source, to: destination)
    }

    private nonisolated static func markerContents() -> String {
        """
        # SteerLab Workspace

        Created by SteerLab (\(SteerLabVersion.current)) on \
        \(ISO8601DateFormatter().string(from: Date())).

        This folder is a SteerLab *data workspace* — the app's Workspace menu
        points here; the code repository keeps only seed/template data.

        Layout:
        - `prompts/` — git-versioned inputs: `concepts/` (contrastive stimulus
          sets), `emotions/` (grand-mean story corpora), `rubrics/`,
          `batteries/`, `neutral/`, `templates/`, `tasks/`, `readers/`,
          `probes/`, `dev/`
        - `experiments/` — experiment manifests (freezable recipes; frozen
          ones carry a `pinned/` snapshot of every pinned input)
        - `runs/` — immutable run outputs (gitignored; never edited)
        - `adapters/` — per-adapter homes (`adapters/<name>/` with
          `training/` and `validation/` data folders; trained weight files
          are gitignored, the data is versioned)

        The workspace is git-managed invisibly: creating it makes the initial
        commit, and freezing an experiment commits the workspace as part of
        the freeze gesture.
        """
    }

    /// The workspace-facing agent contract (`AGENTS.md`). Generated in code,
    /// deliberately — see the header of `AgentContract.swift` and audit §4.3.
    /// **Edits go to `docs/AGENTS-WORKSPACE-DRAFT.md` first** and are mirrored
    /// into `AgentContract.body`; a drift gate holds the two together.
    private nonisolated static func agentContractContents() -> String {
        AgentContract.contents()
    }

    /// git init + initial commit, silent no-op when git is unavailable. A
    /// local fallback identity is configured only when the machine has none,
    /// so app-made commits never fail on an unconfigured git.
    private nonisolated static func initializeGit(at root: URL) {
        guard runGit(["init"], in: root) != nil else { return }
        if runGit(["config", "user.email"], in: root) == nil {
            _ = runGit(["config", "user.name", "SteerLab"], in: root)
            _ = runGit(["config", "user.email", "steerlab@localhost"], in: root)
        }
        _ = runGit(["add", "-A", "."], in: root)
        _ = runGit(
            [
                "commit", "-m",
                "workspace created (seeded from SteerLab \(SteerLabVersion.current))",
            ], in: root)
    }

    private nonisolated static func runGit(
        _ arguments: [String], in root: URL
    ) -> String? {
        let process = Process()
        process.executableURL = URL(filePath: "/usr/bin/git")
        process.arguments = ["-C", root.path] + arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        do {
            try process.run()
        } catch {
            return nil
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }
        return String(decoding: data, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: App-facing observable state

    /// The workspace root as currently resolved; refreshed after a switch.
    public private(set) var rootURL: URL

    public init() {
        rootURL = WorkspaceRoot.current
    }

    /// Folder name for the toolbar label; help text shows the full path.
    public var displayName: String { rootURL.lastPathComponent }

    /// True while running against the legacy code-checkout fallback (#4) —
    /// the switcher labels this honestly instead of pretending it's a
    /// managed workspace.
    public var isLegacyRepoRoot: Bool {
        rootURL.standardizedFileURL.path
            == VectorCatalog.bundledSeedRoot.standardizedFileURL.path
    }

    /// True when `STEERLAB_WORKSPACE` pins the root — switching in the UI
    /// could not take effect, so the menu disables New/Open and says why.
    public var isEnvironmentPinned: Bool { WorkspaceRoot.isEnvironmentPinned }

    /// The ONE workspace-root-change observation seam: fired after every
    /// successful root change — the toolbar pickers and any programmatic
    /// switch all funnel through `switchTo`. The app wires exactly one
    /// observer here (the cluster store's same-machine server
    /// synchronization coordinator); WorkspaceStore itself stays ignorant of
    /// cluster connections.
    public var onRootChange: ((URL) -> Void)?

    /// Where the open-time contract notice lands. `.shared` in production —
    /// the same feed the bell button in the Studies/Agents headers renders,
    /// and the same one `ClusterConnectionStore` posts its `"Workspace"`
    /// synchronization problems to. Injectable so a test never appends to the
    /// live ring.
    public var notices: PanelNotices = .shared

    /// **One line per workspace OPEN**, never per action: perform contract
    /// upkeep and record the single sentence it earned. Returns what it
    /// recorded, or nil when there was nothing to say.
    ///
    /// Deliberately on the OPEN path and not on any read path. The contract is
    /// consulted by the researcher's agent, not by the app, so the moment
    /// worth interrupting is the one where a person chose this workspace —
    /// and a notice that reappeared on every panel refresh would train them to
    /// dismiss the bell.
    ///
    /// Two things can be worth saying. A **refresh** — a hashed, provably
    /// unedited contract brought up to this build — is `.info`: work already
    /// done, nothing to repair. A **legacy stale** file is `.warning`: this
    /// build will not touch it, and the one manual regeneration is the
    /// gesture that ends the nagging for good. Silent for `current`; for
    /// `absent`, which upkeep has just rewritten and which nobody has to
    /// repair by hand in any case; and for `edited` — the researcher's own
    /// text is not a defect and gets no notice on any surface. Non-blocking:
    /// the open has already succeeded when this runs.
    ///
    /// `root` defaults to the live one; it is a parameter for the same reason
    /// `open(at:defaults:setOverride:)` takes its two — so a test can point
    /// the check at a fixture without moving the process's workspace.
    @discardableResult
    public func noteAgentContractUpkeep(at root: URL? = nil) -> String? {
        let root = root ?? rootURL
        return record(Self.upkeepAgentContract(at: root), at: root)
    }

    /// Record an upkeep outcome `open` already performed — so `switchTo` does
    /// not classify (and could not possibly rewrite) a file the open path has
    /// just handled.
    @discardableResult
    private func record(_ upkeep: AgentContractUpkeep, at root: URL) -> String? {
        guard let sentence = upkeep.sentence(at: root) else { return nil }
        notices.record(
            source: "Workspace", severity: upkeep.severity, message: sentence)
        return sentence
    }

    public func switchTo(_ url: URL) throws {
        let upkeep = try Self.open(at: url)
        rootURL = WorkspaceRoot.current
        record(upkeep, at: rootURL)
        onRootChange?(rootURL)
    }

    /// Creates the workspace and switches to it. `computing:` DECLARES what
    /// the workspace is for at the moment it is made — the one point where
    /// the answer is never ambiguous. Omitted, the workspace stays
    /// undeclared and `WorkspaceCompute` infers from its runs.
    public func createAndSwitch(
        to url: URL, computing compute: WorkspaceCompute? = nil
    ) throws {
        _ = try Self.create(at: url)
        if let compute { try WorkspaceCompute.declare(compute, root: url) }
        try switchTo(url)
    }

    // MARK: Compute binding

    /// What this workspace computes on — declared, or inferred from its own
    /// runs. Read by the UI; the lifecycle reads
    /// `ExperimentStore.computeSubstrate`, which resolves the same way.
    public var compute: WorkspaceCompute {
        WorkspaceCompute.resolved(root: rootURL)
    }

    /// False when the binding above is standing in for a declaration. An
    /// inference must read as a suggestion, never as a setting the
    /// researcher chose.
    public var isComputeDeclared: Bool {
        WorkspaceCompute.isDeclared(root: rootURL)
    }

    public func declareCompute(_ compute: WorkspaceCompute) throws {
        try WorkspaceCompute.declare(compute, root: rootURL)
        // Republish: `compute`/`isComputeDeclared` are derived from disk, so
        // Observation has nothing to notice without a stored-property touch.
        rootURL = rootURL
    }
}
