import Foundation

// MARK: - Demo Workspaces: worked examples, opened as a verified copy
//
// A Demo Workspace is a complete study someone can read before downloading a
// model: a finished study with its results, and a draft copy ready to run.
// One is built per compute backend, because a workspace is bound to one
// backend and its vectors and evidence are stamped by it.
//
// The product CARRIES them (`CodeResources.Family.demoWorkspaces`:
// `DemoWorkspaces/<backend>/` in a checkout and in the app bundle) and never
// opens one in place. `open` copies a carried tree to a folder the researcher
// chooses, checks every copied byte against the original, checks each study
// the way `experiment verify` does, records the backend's compute binding,
// and commits the result once. A build may carry any subset of the backends,
// including none.
//
// Python twin: `Server/steerlab_server/client/demo_workspaces.py`. The two
// read the same `demo.json`, leave the same files out of a copy, record the
// same binding bytes, and refuse in the same words.

public enum DemoWorkspace {

    // MARK: Backends

    /// The backends a Demo Workspace can be built for, in the order they are
    /// offered. The raw value is the folder name under `DemoWorkspaces/`.
    public enum Backend: String, CaseIterable, Sendable, Codable, Identifiable {
        /// The engine built into the app.
        case mlx
        /// The Python engine on this Mac's own graphics processor.
        case mps
        /// The Python engine on a workstation or a cluster.
        case cuda

        public var id: String { rawValue }

        /// The compute choice a copy is set to, so it opens already bound to
        /// the engine its vectors and evidence came from.
        public var computeChoice: ComputeChoice {
            switch self {
            case .mlx: .macQuickStart
            case .mps: .macFullCapabilities
            case .cuda: .anotherMachine
            }
        }

        /// Where this backend's studies run, in the words the compute choice
        /// uses everywhere else.
        public var title: String { computeChoice.title }
    }

    // MARK: What a carried demo says about itself (`demo.json`)

    public struct Model: Sendable, Equatable {
        /// The model the demo's studies run on.
        public let id: String
        /// Roughly how much there is to download before the first run.
        public let approximateDownloadGB: Double
    }

    public struct Study: Sendable, Equatable {
        public let name: String
        /// One line about the study, when the demo gives one.
        public let summary: String?
    }

    public struct Description: Sendable, Equatable {
        public static let schemaVersion = 1

        public let backend: Backend
        public let title: String
        /// One line: what the demo shows.
        public let summary: String
        public let model: Model
        public let studies: [Study]
        /// `demo.json` as written, so keys a later version adds travel
        /// through untouched.
        public let document: JSONValue
    }

    /// One carried demo: where its tree is, and what it says about itself.
    public struct Entry: Sendable, Equatable, Identifiable {
        public let directory: URL
        public let description: Description
        public var backend: Backend { description.backend }
        public var id: String { backend.rawValue }
    }

    // MARK: Refusals

    /// A demo that cannot be opened: a stable code and plain words.
    ///
    /// `repair` is written for a person. The command line replaces it with a
    /// command where there is one to run; the app shows it as it is.
    public struct Refusal: Error, Equatable, Sendable, CustomStringConvertible {
        public enum Code: String, Sendable {
            /// This build carries no demo for the backend asked for.
            case demoNotCarried
            /// The carried demo is incomplete or unreadable.
            case demoDamaged
            /// The chosen folder exists and holds something.
            case destinationNotEmpty
            /// The copy differs from the original, or a study in it did not
            /// pass verification. Nothing was created.
            case demoCopyUnverified
        }

        public let code: Code
        public let reason: String
        public let repair: String
        public var backend: Backend?
        /// The backends this build does carry, on `demoNotCarried`.
        public var carried: [Backend] = []
        /// Each study's check, when a study failed in the copy.
        public var studies: [StudyCheck] = []
        /// The files whose bytes differed, when the copy did not match.
        public var differingFiles: [String] = []

        public var description: String { reason }

        static let reinstall = "Reinstall SteerLab, then open the demo again in a new folder."

        static func damaged(_ backend: String, _ detail: String) -> Refusal {
            Refusal(
                code: .demoDamaged,
                reason: "The \(backend) Demo Workspace in this copy of SteerLab is "
                    + "incomplete: \(detail)",
                repair: reinstall, backend: Backend(rawValue: backend))
        }
    }

    // MARK: Results

    /// One study, checked exactly as `experiment verify` checks it.
    public struct StudyCheck: Sendable, Equatable {
        public let name: String
        /// `draft` or `frozen`; nil when the study could not be read.
        public let status: String?
        public let verified: Bool
        public let violations: [String]
    }

    /// What `open` made.
    public struct Opened: Sendable, Equatable {
        public let root: URL
        public let entry: Entry
        /// How many carried files were copied, and their total size.
        public let fileCount: Int
        public let byteCount: Int
        /// Each study's check in the copy, or nil when the caller asked `open`
        /// not to check them (the app checks after it has switched to the copy).
        public let studies: [StudyCheck]?

        public var readme: URL { root.appending(component: "README.md") }
    }

    // MARK: Reading what the build carries

    /// Written fresh for every copy by workspace creation, so a carried tree's
    /// own copies of them are never used.
    static let generatedFiles: Set<String> = [AgentContract.fileName, WorkspaceStore.markerFileName]

    /// Where this build keeps its Demo Workspaces, or nil when it has none.
    public static func carriedRoot() -> URL? {
        try? CodeResources.demoWorkspaces()
    }

    /// Read and check one carried demo's `demo.json`.
    ///
    /// Checked: the schema version, that the backend matches the folder's
    /// name, a title and a one-line summary, the model and its approximate
    /// download size, that every named study is present, and the README.
    public static func describe(_ directory: URL) throws -> Description {
        let folder = directory.lastPathComponent
        let fm = FileManager.default
        guard let data = try? Data(contentsOf: directory.appending(component: "demo.json")),
            let document = try? JSONDecoder().decode(JSONValue.self, from: data)
        else {
            throw Refusal.damaged(folder, "demo.json could not be read.")
        }
        guard case .object(let object) = document else {
            throw Refusal.damaged(folder, "demo.json is not an object.")
        }
        func line(_ value: JSONValue?) -> String? {
            guard case .string(let text)? = value,
                !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                !text.contains("\n")
            else { return nil }
            return text
        }
        guard object["schemaVersion"] == .number(Double(Description.schemaVersion)) else {
            throw Refusal.damaged(
                folder, "demo.json needs \"schemaVersion\": \(Description.schemaVersion).")
        }
        guard case .string(let named)? = object["backend"], named == folder,
            let backend = Backend(rawValue: named)
        else {
            let named: String = { if case .string(let text)? = object["backend"] { "'\(text)'" } else { "None" } }()
            throw Refusal.damaged(
                folder,
                "demo.json names the backend \(named), but its folder is '\(folder)'.")
        }
        guard let title = line(object["title"]) else {
            throw Refusal.damaged(folder, "demo.json needs a one-line \"title\".")
        }
        guard let summary = line(object["summary"]) else {
            throw Refusal.damaged(folder, "demo.json needs a one-line \"summary\".")
        }
        guard case .object(let model)? = object["model"], let modelID = line(model["id"]) else {
            throw Refusal.damaged(
                folder,
                "demo.json needs \"model\": {\"id\": …, \"approximateDownloadGB\": …}.")
        }
        guard case .number(let size)? = model["approximateDownloadGB"], size > 0 else {
            throw Refusal.damaged(
                folder,
                "demo.json needs the model's \"approximateDownloadGB\" as a number above zero.")
        }
        guard case .array(let listed)? = object["studies"], !listed.isEmpty else {
            throw Refusal.damaged(
                folder, "demo.json needs a \"studies\" list that names at least one study.")
        }
        var studies: [Study] = []
        for entry in listed {
            guard case .object(let study) = entry, let name = line(study["name"]),
                !name.contains("/"), name != ".", name != ".."
            else {
                throw Refusal.damaged(folder, "every entry in \"studies\" needs a \"name\".")
            }
            if study["summary"] != nil, line(study["summary"]) == nil {
                throw Refusal.damaged(
                    folder, "the study '\(name)' has a \"summary\" that is not one line of text.")
            }
            let experiments = directory.appending(component: "experiments")
            guard
                fm.fileExists(
                    atPath: experiments.appending(components: name, "experiment.json").path)
                    || fm.fileExists(atPath: experiments.appending(component: "\(name).json").path)
            else {
                throw Refusal.damaged(
                    folder,
                    "the study '\(name)' is named in demo.json but is not under experiments/.")
            }
            studies.append(Study(name: name, summary: line(study["summary"])))
        }
        guard Set(studies.map(\.name)).count == studies.count else {
            throw Refusal.damaged(folder, "demo.json names a study twice.")
        }
        guard
            let readme = try? String(
                contentsOf: directory.appending(component: "README.md"), encoding: .utf8),
            !readme.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw Refusal.damaged(folder, "its README.md is missing or empty.")
        }
        return Description(
            backend: backend, title: title, summary: summary,
            model: Model(id: modelID, approximateDownloadGB: size),
            studies: studies, document: document)
    }

    /// The files a copy receives, as sorted workspace-relative paths.
    ///
    /// Left out: anything whose path has a dot-prefixed part (workspace-local
    /// state, and files an installer may leave), Python bytecode caches, and
    /// the two files workspace creation writes fresh for every copy. A
    /// symbolic link is refused: a copy must be made of this tree's own bytes.
    public static func files(in directory: URL) throws -> [String] {
        let fm = FileManager.default
        var names: [String] = []
        func walk(_ folder: URL, _ prefix: String) throws {
            let entries = (try? fm.contentsOfDirectory(atPath: folder.path)) ?? []
            for name in entries.sorted() where !name.hasPrefix(".") {
                let url = folder.appending(component: name)
                let relative = prefix.isEmpty ? name : "\(prefix)/\(name)"
                let values = try? url.resourceValues(forKeys: [.isSymbolicLinkKey, .isDirectoryKey])
                if values?.isSymbolicLink == true {
                    throw Refusal.damaged(
                        directory.lastPathComponent, "\(relative) is a symbolic link.")
                }
                if values?.isDirectory == true {
                    if name != "__pycache__" { try walk(url, relative) }
                } else if !name.hasSuffix(".pyc"), !generatedFiles.contains(relative) {
                    names.append(relative)
                }
            }
        }
        try walk(directory, "")
        return names.sorted()
    }

    /// The demos this build carries and can open, in `Backend` order. A
    /// damaged one is left out here and named when it is asked for.
    public static func available(in root: URL? = nil) -> [Entry] {
        guard let root = root ?? carriedRoot() else { return [] }
        return Backend.allCases.compactMap { backend in
            let directory = root.appending(component: backend.rawValue)
            guard let description = try? describe(directory) else { return nil }
            return Entry(directory: directory, description: description)
        }
    }

    /// One carried demo, or a plain refusal.
    public static func carried(_ backend: Backend, in root: URL? = nil) throws -> Entry {
        var isDirectory: ObjCBool = false
        guard let root = root ?? carriedRoot(),
            FileManager.default.fileExists(
                atPath: root.appending(component: backend.rawValue).path,
                isDirectory: &isDirectory), isDirectory.boolValue
        else {
            let others = available(in: root).map(\.backend)
            let carries =
                others.isEmpty
                ? "It carries none."
                : "It carries one for \(joined(others.map(\.rawValue)))."
            throw Refusal(
                code: .demoNotCarried,
                reason: "This copy of SteerLab carries no Demo Workspace for "
                    + "\(backend.rawValue) (\(backend.title)). \(carries)",
                repair: "Open a demo this copy carries, or create an ordinary new workspace.",
                backend: backend, carried: others)
        }
        let directory = root.appending(component: backend.rawValue)
        return Entry(directory: directory, description: try describe(directory))
    }

    /// `a`, `a and b`, or `a, b, and c`.
    static func joined(_ words: [String]) -> String {
        words.count < 3
            ? words.joined(separator: " and ")
            : words.dropLast().joined(separator: ", ") + ", and " + (words.last ?? "")
    }

    // MARK: Opening a copy

    /// Copy one carried demo to `destination` and return what was made.
    ///
    /// The copy is staged beside the destination and installed only when all
    /// of it holds: every file is byte-for-byte the carried one and, when
    /// `verifyingStudies` is set, every study the demo names passes
    /// verification in the copy. A refusal leaves nothing behind. The seed
    /// fills in any file a new workspace would have and the demo does not
    /// carry; it never replaces one the demo does carry.
    ///
    /// `verifyingStudies` reads the staged copy AS the workspace for the
    /// duration of the check (`WorkspaceRoot.reading`), which is right for
    /// the command line — one verb, one process — and wrong for the app,
    /// where other work is reading the current workspace at the same time.
    /// The app passes false, switches to the copy, and then calls
    /// `checkStudiesInCurrentWorkspace`.
    @discardableResult
    public static func open(
        _ backend: Backend, at destination: URL, in root: URL? = nil,
        seedingFrom seedRoot: URL? = nil, verifyingStudies: Bool
    ) throws -> Opened {
        let entry = try carried(backend, in: root)
        let names = try files(in: entry.directory)
        let fm = FileManager.default
        let target = destination.standardizedFileURL
        if fm.fileExists(atPath: target.path) {
            var isDirectory: ObjCBool = false
            fm.fileExists(atPath: target.path, isDirectory: &isDirectory)
            let contents = (try? fm.contentsOfDirectory(atPath: target.path))?
                .filter { $0 != ".DS_Store" } ?? []
            guard isDirectory.boolValue, contents.isEmpty else {
                throw Refusal(
                    code: .destinationNotEmpty,
                    reason: "The folder \(target.path) already exists and is not empty, "
                        + "so nothing was copied into it.",
                    repair: "Choose a new or empty folder for the copy.", backend: backend)
            }
        }

        var byteCount = 0
        var studies: [StudyCheck]?
        let created = try WorkspaceStore.create(
            at: target, seedingFrom: seedRoot,
            placingFirst: { staged in
                for name in names {
                    let copy = staged.appending(path: name)
                    try fm.createDirectory(
                        at: copy.deletingLastPathComponent(), withIntermediateDirectories: true)
                    try fm.copyItem(at: entry.directory.appending(path: name), to: copy)
                }
            },
            beforeCommit: { staged in
                var differing: [String] = []
                for name in names {
                    let original = try Data(contentsOf: entry.directory.appending(path: name))
                    let copy = try? Data(contentsOf: staged.appending(path: name))
                    byteCount += original.count
                    if copy != original { differing.append(name) }
                }
                guard differing.isEmpty else {
                    throw Refusal(
                        code: .demoCopyUnverified,
                        reason: "The copy of the \(backend.rawValue) Demo Workspace does not "
                            + "match the original in \(differing.count) file(s), so nothing "
                            + "was created.",
                        repair: "Check that the disk has free space, then try again in a new folder.",
                        backend: backend, differingFiles: differing)
                }
                if verifyingStudies {
                    let checks = WorkspaceRoot.reading(staged) {
                        checkStudies(entry.description.studies.map(\.name))
                    }
                    if let refusal = unverified(checks, backend: backend, created: false) {
                        throw refusal
                    }
                    studies = checks
                }
                try WorkspaceCompute.declare(backend.computeChoice, root: staged)
            },
            commitMessage: "demo workspace opened (\(backend.rawValue), from SteerLab "
                + "\(SteerLabVersion.current))")
        return Opened(
            root: created, entry: entry, fileCount: names.count, byteCount: byteCount,
            studies: studies)
    }

    /// Each named study, checked exactly as `experiment verify` checks it,
    /// against the workspace this process is reading right now.
    static func checkStudies(_ names: [String]) -> [StudyCheck] {
        names.map { name in
            do {
                let manifest = try ExperimentStore.load(name: name)
                let violations = ExperimentStore.verify(manifest)
                return StudyCheck(
                    name: name, status: manifest.status.rawValue,
                    verified: violations.isEmpty, violations: violations)
            } catch {
                return StudyCheck(
                    name: name, status: nil, verified: false,
                    violations: ["the study could not be read (\(error))"])
            }
        }
    }

    /// The refusal for studies that did not verify, or nil when all did.
    /// `created` says whether the copy exists: the command line checks before
    /// it installs the copy, and the app checks after it has opened it.
    static func unverified(
        _ checks: [StudyCheck], backend: Backend, created: Bool
    ) -> Refusal? {
        let failed = checks.filter { !$0.verified }.map(\.name)
        guard !failed.isEmpty else { return nil }
        return Refusal(
            code: .demoCopyUnverified,
            reason: "The \(backend.rawValue) Demo Workspace was copied, but "
                + "\(joined(failed)) did not pass verification in the copy"
                + (created ? "." : ", so nothing was created."),
            repair: Refusal.reinstall, backend: backend, studies: checks)
    }

    /// The app's half of study verification: after it has switched to a copy
    /// `open` made, check that copy's studies in place. Returns nil — nothing
    /// checked — unless `opened.root` is the workspace this process is reading,
    /// so it can never report on some other workspace's studies.
    public static func checkStudiesInCurrentWorkspace(_ opened: Opened) -> [StudyCheck]? {
        guard
            ExperimentStore.workspaceRoot.standardizedFileURL.path
                == opened.root.standardizedFileURL.path
        else { return nil }
        return checkStudies(opened.entry.description.studies.map(\.name))
    }

    // MARK: Words

    /// What a demo needs, in one plain line: the model and its download, and
    /// where its studies run.
    public static func needsLine(_ description: Description) -> String {
        let gigabytes = description.model.approximateDownloadGB
        let size = String(format: gigabytes == gigabytes.rounded() ? "%.0f" : "%.1f", gigabytes)
        let model = "Needs the model \(description.model.id), a download of about \(size) GB."
        switch description.backend {
        case .mlx:
            return model + " Runs on this Mac with nothing else to install."
        case .mps:
            return model + " Runs on this Mac with every method, after a one-time "
                + "setup of several gigabytes."
        case .cuda:
            return model + " Runs on a workstation or a cluster you connect to."
        }
    }

    /// What a demo shows, in one plain line: its own summary.
    public static func showsLine(_ description: Description) -> String {
        description.summary
    }

    /// After opening a Demo Workspace: its README first, then its studies.
    /// Python twin: `bootstrap_commands.demo_next_action`, which names
    /// `--root` where this names `--workspace`.
    static func nextAction(
        rootPath: String, workspaceFlag: String = "--workspace"
    ) -> SteerLabCLIEnvelope.NextAction {
        .init(
            verb: "experiment list",
            detail: "Read \(rootPath)/README.md first. It says what this demo study asks, "
                + "and gives the steps from here to an exported result. Then list the "
                + "studies: name the workspace with \(workspaceFlag) \(rootPath), or export "
                + "STEERLAB_WORKSPACE=\(rootPath).")
    }

    /// One study check as the envelope carries it. Same keys as the Python
    /// client's.
    static func json(_ check: StudyCheck) -> JSONValue {
        .object([
            "name": .string(check.name),
            "status": check.status.map(JSONValue.string) ?? .null,
            "verified": .bool(check.verified),
            "violations": .array(check.violations.map(JSONValue.string)),
        ])
    }
}

// MARK: - The app's side

/// The app's words for Demo Workspaces, for a researcher with no
/// machine-learning background. Data rather than view text, so the suite can
/// hold them to Research Setup's rule: no command, no flag, no path.
public enum DemoWorkspaceCopy {
    /// The button in Research Setup's first step.
    public static let button = "Open Demo Workspace…"

    public static let title = "Open a Demo Workspace"

    public static let introduction =
        "A Demo Workspace is a finished study you can read before you download "
        + "a model, with a draft copy you can run yourself. SteerLab makes a "
        + "copy in a folder you choose and opens the copy. The original is "
        + "never changed."

    /// Shown in place of the list when this build carries no demo.
    public static let noneCarried =
        "This copy of SteerLab does not include a Demo Workspace. You can "
        + "still create a new workspace and design your own study."

    /// One row's heading: the demo's own title, and where its studies run.
    public static func rowTitle(_ description: DemoWorkspace.Description) -> String {
        "\(description.title) (\(description.backend.title))"
    }

    public static let openButton = "Choose a Folder and Open…"
    public static let panelTitle = "Choose a Folder for the Demo Copy"
    public static let panelPrompt = "Open Copy Here"
    public static let defaultFolderName = "SteerLab Demo Workspace"

    /// Said once the copy is open.
    public static let opened =
        "The demo is open, and this copy is yours to change. Start with its "
        + "guide: the file named README at the top of the workspace folder."

    public static let showGuide = "Open the Demo's Guide"

    /// Said when the copy opened but a study in it did not pass its check.
    public static func unverified(_ names: [String]) -> String {
        "The demo was copied and opened, but \(DemoWorkspace.joined(names)) did "
            + "not pass verification in the copy. This copy of SteerLab may be "
            + "damaged. Reinstall it, then open the demo again in a new folder."
    }

    public static let failedTitle = "Could not open the Demo Workspace"

    /// Why the button is unavailable while the workspace is fixed from outside
    /// the app.
    public static let unavailableWhilePinned =
        "The workspace was fixed when SteerLab was started, so it cannot be "
        + "changed here. Quit SteerLab and open it again in the usual way."
}

extension WorkspaceStore {

    /// What opening a Demo Workspace in the app came to.
    public struct DemoOpening: Sendable, Equatable {
        public let opened: DemoWorkspace.Opened
        /// Each study's check in the opened copy, or nil when the app was not
        /// reading the copy and so could not check it.
        public let studies: [DemoWorkspace.StudyCheck]?

        /// The studies that did not pass, by name.
        public var unverified: [String] {
            (studies ?? []).filter { !$0.verified }.map(\.name)
        }
    }

    /// Open a copy of a Demo Workspace this build carries, the way the app
    /// does it: copy it to `url` with every byte checked, switch to the copy,
    /// and then check each of its studies in place.
    ///
    /// The studies are checked AFTER the switch because the check reads the
    /// current workspace, and the app must never point its readers at another
    /// folder behind their backs (the command line, one verb per process,
    /// checks before it installs the copy). A study that fails is reported in
    /// the result; the copy is already open, and the researcher is told.
    @discardableResult
    public func openDemoWorkspace(
        _ backend: DemoWorkspace.Backend, at url: URL, in root: URL? = nil
    ) throws -> DemoOpening {
        try openDemoWorkspace(backend, at: url, in: root) { try self.switchTo($0) }
    }

    /// The same, with the switch injected — the seam that lets a test stand
    /// in the app's sequence without writing the saved workspace choice.
    func openDemoWorkspace(
        _ backend: DemoWorkspace.Backend, at url: URL, in root: URL?,
        switching: (URL) throws -> Void
    ) throws -> DemoOpening {
        guard !isEnvironmentPinned else {
            throw ExperimentError(reason: DemoWorkspaceCopy.unavailableWhilePinned)
        }
        let opened = try DemoWorkspace.open(
            backend, at: url, in: root, verifyingStudies: false)
        try switching(opened.root)
        return DemoOpening(
            opened: opened, studies: DemoWorkspace.checkStudiesInCurrentWorkspace(opened))
    }
}

extension WorkspaceRoot {

    /// Run `body` with `root` read as THE workspace, ahead of every other
    /// rule — `STEERLAB_WORKSPACE` included — and put everything back after.
    ///
    /// For one purpose: `workspace init <dir> --demo <backend>` checks the
    /// studies of a copy it has not installed yet, and the study checks read
    /// the process-wide root. A shell that exported `STEERLAB_WORKSPACE` for
    /// an earlier workspace would otherwise have the check read that one.
    ///
    /// ONLY for a process that runs one verb and exits. The switch is
    /// process-wide for the duration of `body`: in the app, where other work
    /// reads the current workspace at the same time, it would hand that work
    /// the wrong folder. The app switches to the copy first and checks there
    /// (`DemoWorkspace.checkStudiesInCurrentWorkspace`).
    static func reading<T>(_ root: URL, _ body: () throws -> T) rethrows -> T {
        let previous = (scopedReadRoot, ExperimentStore.rootOverride)
        scopedReadRoot = root.standardizedFileURL
        ExperimentStore.rootOverride = root.standardizedFileURL
        defer { (scopedReadRoot, ExperimentStore.rootOverride) = previous }
        return try body()
    }
}
