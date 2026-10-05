import Foundation

// MARK: - The first-study checklist on Home
//
// Seven steps from nothing to an exported result, in order: choose a
// workspace, choose where studies run, get a model, open the Demo Workspace or
// start a study, run it, read the results, and export them. Each step is done
// or not done because of something the workspace or the app actually holds,
// never because a button was pressed: a folder on disk, a declaration in the
// workspace's settings, a model the active engine reports, a run directory, an
// analysis of that run, or an export folder.
//
// The app shows it on Home (`FirstStudyChecklistView`); this file owns the
// facts, the judgement, and the words, so the suite can hold all three. The
// words follow Research Setup's rule: no command, no flag, and no path.

public enum FirstStudyChecklist {

    // MARK: Steps

    public enum Step: String, CaseIterable, Sendable, Identifiable {
        case workspace
        case compute
        case model
        case study
        case run
        case results
        case export

        public var id: String { rawValue }
    }

    // MARK: What a step can take the researcher to

    /// One button. The app performs it; this type only names it, so a step
    /// never offers something the app has no way to do.
    public enum Action: Hashable, Sendable {
        case newWorkspace
        case openWorkspace
        /// The three places studies can run, offered as one menu.
        case chooseCompute
        case openPlayground
        case openDemoWorkspace
        /// Studies, with this study selected when one is named.
        case openStudies(select: String?)
        case openResults

        public var title: String {
            switch self {
            case .newWorkspace: "New Workspace…"
            case .openWorkspace: "Open Workspace…"
            case .chooseCompute: "Choose…"
            case .openPlayground: "Open Playground"
            case .openDemoWorkspace: DemoWorkspaceCopy.button
            case .openStudies: "Open Studies"
            case .openResults: "Open Results"
            }
        }

        /// Whether the button can do anything before a workspace exists.
        /// Creating, opening, and the Demo Workspace each make one; every
        /// other destination works inside a workspace.
        public var needsWorkspace: Bool {
            switch self {
            case .newWorkspace, .openWorkspace, .openDemoWorkspace: false
            case .chooseCompute, .openPlayground, .openStudies, .openResults: true
            }
        }
    }

    // MARK: Facts

    /// One study in the workspace: its folder name and its status as the
    /// manifest records it (`draft`, `frozen`, or `complete`).
    public struct Study: Equatable, Sendable {
        public let name: String
        public let status: String?

        public init(name: String, status: String?) {
            self.name = name
            self.status = status
        }
    }

    /// One completed run of a study: a run directory holding both its
    /// responses and its report.
    public struct Run: Equatable, Sendable {
        public let directoryName: String
        public let study: String?

        public init(directoryName: String, study: String?) {
            self.directoryName = directoryName
            self.study = study
        }
    }

    /// Everything the checklist is judged from.
    ///
    /// `scan(root:carriedDemos:)` reads all of it from a workspace folder
    /// except `modelCount`, which the app reads from the active engine's
    /// model inventory, and which it may refresh `computeChoice` from when
    /// the researcher changes it.
    public struct Facts: Equatable, Sendable {
        public var hasWorkspace: Bool
        /// The place this workspace declares its studies run, or nil when it
        /// has declared none.
        public var computeChoice: ComputeChoice?
        public var modelCount: Int
        public var studies: [Study]
        /// True for a copy of a Demo Workspace (its `demo.json` is at the top).
        public var isDemoCopy: Bool
        /// The model a Demo Workspace copy's studies use, when it names one.
        public var demoModelID: String?
        /// Completed runs made in this workspace, newest first. Runs that came
        /// with a Demo Workspace are not counted: they were not run here.
        public var completedRuns: [Run]
        /// The completed runs above that have an analysis or an evaluation.
        public var runsWithResults: [String]
        /// Results exports of runs made in this workspace.
        public var exportCount: Int
        /// Studies with a completed run of any kind, the Demo Workspace's own
        /// finished run included. Used only to choose which study the Run
        /// step opens.
        public var studiesWithAnyRun: Set<String>

        public init(
            hasWorkspace: Bool, computeChoice: ComputeChoice? = nil, modelCount: Int = 0,
            studies: [Study] = [], isDemoCopy: Bool = false, demoModelID: String? = nil,
            completedRuns: [Run] = [], runsWithResults: [String] = [], exportCount: Int = 0,
            studiesWithAnyRun: Set<String> = []
        ) {
            self.hasWorkspace = hasWorkspace
            self.computeChoice = computeChoice
            self.modelCount = modelCount
            self.studies = studies
            self.isDemoCopy = isDemoCopy
            self.demoModelID = demoModelID
            self.completedRuns = completedRuns
            self.runsWithResults = runsWithResults
            self.exportCount = exportCount
            self.studiesWithAnyRun = studiesWithAnyRun
        }

        /// Before any workspace exists.
        public static let noWorkspace = Facts(hasWorkspace: false)
    }

    // MARK: Items

    public struct Item: Equatable, Sendable, Identifiable {
        public let step: Step
        public let title: String
        public let detail: String
        public let isDone: Bool
        public let actions: [Action]
        public var id: Step { step }
    }

    /// The seven steps, in order, judged from `facts`.
    public static func items(_ facts: Facts) -> [Item] {
        Step.allCases.map { item($0, facts) }
    }

    /// The first step that is not done, or nil when every step is.
    public static func nextStep(_ items: [Item]) -> Step? {
        items.first { !$0.isDone }?.step
    }

    /// "3 of 7 steps done", for the checklist's heading.
    public static func progress(_ items: [Item]) -> String {
        let done = items.filter(\.isDone).count
        return done == items.count
            ? "All \(items.count) steps done"
            : "\(done) of \(items.count) steps done"
    }

    public static let title = "Your first study"

    /// Said under the heading while steps remain.
    public static let introduction =
        "The steps from a new workspace to results you can share. Each one is "
        + "checked from what this workspace holds, so it stays done once it is done."

    /// Said once every step is done, when the steps are hidden.
    public static let finished =
        "Every step to a first exported result is done. The steps stay here if "
        + "you want to see them again."

    static func item(_ step: Step, _ facts: Facts) -> Item {
        switch step {
        case .workspace:
            return Item(
                step: step, title: "Choose a workspace",
                detail: facts.hasWorkspace
                    ? "This workspace keeps your studies, their inputs, and their "
                        + "results together in one folder."
                    : "A workspace is a folder that keeps your studies, their inputs, "
                        + "and their results together. Create one, open one you have, "
                        + "or open a copy of the Demo Workspace.",
                isDone: facts.hasWorkspace,
                actions: [.newWorkspace, .openWorkspace])

        case .compute:
            let detail: String =
                if let choice = facts.computeChoice {
                    "Where this workspace's studies run: \(choice.title)."
                } else {
                    "Pick one of three places: this Mac with nothing else to install, "
                        + "this Mac with every method after a one-time setup, or "
                        + "another machine such as a workstation or a cluster."
                }
            return Item(
                step: step, title: "Choose where studies run", detail: detail,
                isDone: facts.hasWorkspace && facts.computeChoice != nil,
                actions: [.chooseCompute])

        case .model:
            let count = facts.modelCount
            let detail: String =
                if count > 0 {
                    "\(count) \(count == 1 ? "model is" : "models are") ready where "
                        + "your studies run."
                } else if facts.isDemoCopy, let model = facts.demoModelID {
                    "The demo's studies use the model \(model). Choose it in the "
                        + "Playground and download it there."
                } else {
                    "Studies run on an open-weight language model. Choose one in the "
                        + "Playground and download it there."
                }
            return Item(
                step: step, title: "Get a model", detail: detail,
                isDone: facts.hasWorkspace && count > 0, actions: [.openPlayground])

        case .study:
            let count = facts.studies.count
            let detail: String =
                if count > 0, facts.isDemoCopy {
                    "This is a copy of the Demo Workspace: a finished study to read and "
                        + "a draft you can run yourself."
                } else if count > 0 {
                    "This workspace has \(count) \(count == 1 ? "study" : "studies")."
                } else {
                    "The Demo Workspace is a finished study you can read, with a draft "
                        + "copy you can run. Or start your own draft in Studies."
                }
            return Item(
                step: step, title: "Open the Demo Workspace or start a study",
                detail: detail, isDone: facts.hasWorkspace && count > 0,
                actions: [.openDemoWorkspace, .openStudies(select: nil)])

        case .run:
            let count = facts.completedRuns.count
            let detail: String =
                if count > 0 {
                    "\(count) completed \(count == 1 ? "run" : "runs"). A run's "
                        + "responses are saved as they were recorded and never changed."
                } else if facts.studies.isEmpty {
                    "Once there is a study, freeze its design and run it."
                } else if facts.isDemoCopy {
                    "Run the demo's draft: open it in Studies, freeze its design, and "
                        + "run it. The finished study's run came with the demo."
                } else {
                    "Open a study, freeze its design, and run it. Freezing fixes what "
                        + "the study measures before any response is recorded."
                }
            return Item(
                step: step, title: "Run the study", detail: detail,
                isDone: facts.hasWorkspace && count > 0,
                actions: [.openStudies(select: studyToRun(facts))])

        case .results:
            let done = facts.hasWorkspace && !facts.runsWithResults.isEmpty
            var detail =
                done
                ? "Your run has its analysis: each condition compared with the baseline."
                : "Open your run in Results and analyze it. The analysis compares each "
                    + "condition with the baseline, and nothing in the run is changed."
            if !done, facts.isDemoCopy {
                detail += " The demo's finished study can be read in Results now."
            }
            return Item(
                step: step, title: "Read the results", detail: detail, isDone: done,
                actions: [.openResults])

        case .export:
            let done = facts.hasWorkspace && facts.exportCount > 0
            return Item(
                step: step, title: "Export the results",
                detail: done
                    ? "Exported: tables, transcripts, and a methods summary are in the "
                        + "exports folder of this workspace."
                    : "Export Results, in a study's results, writes tables for R, Stata, "
                        + "SPSS, or a spreadsheet, with transcripts, a methods summary, "
                        + "and a codebook. The run itself is never changed.",
                isDone: done,
                actions: [.openStudies(select: studyToExport(facts))])
        }
    }

    /// The study the Run step opens: one that has not run yet, preferring one
    /// whose design is already frozen; else the first study.
    static func studyToRun(_ facts: Facts) -> String? {
        let waiting = facts.studies.filter {
            !facts.studiesWithAnyRun.contains($0.name) && $0.status != "complete"
        }
        return (waiting.first { $0.status == "frozen" } ?? waiting.first ?? facts.studies.first)?
            .name
    }

    /// The study the Export step opens: the one with the newest completed run.
    static func studyToExport(_ facts: Facts) -> String? {
        facts.completedRuns.first?.study ?? studyToRun(facts)
    }

    // MARK: Reading a workspace

    /// Read every fact a workspace folder holds. Reads only; nothing is
    /// written. `carriedDemos` is where this build keeps its Demo Workspaces
    /// (`DemoWorkspace.carriedRoot()`): a run or export a copy carried in from
    /// there was not made in this workspace, so it does not count. When the
    /// build no longer carries the demo a copy came from, every run counts.
    public static func scan(root: URL, carriedDemos: URL?) -> Facts {
        let demo = demoDescription(root: root)
        let carriedRuns = demo.flatMap { backend in
            carriedDemos.map {
                directoryNames($0.appending(components: backend.backend, "runs"))
            }
        } ?? []
        let carriedExports = demo.flatMap { backend in
            carriedDemos.map {
                directoryNames($0.appending(components: backend.backend, "exports"))
            }
        } ?? []

        let runsRoot = root.appending(component: "runs")
        var completed: [Run] = []
        var studiesWithAnyRun = Set<String>()
        var sources = Set<String>()
        for name in directoryNames(runsRoot).sorted(by: >) {
            let directory = runsRoot.appending(component: name)
            if let study = study(ofRun: name, in: directory) ?? matched(name, runPattern) {
                guard isCompleteRun(directory) else { continue }
                studiesWithAnyRun.insert(study)
                if !carriedRuns.contains(name) {
                    completed.append(Run(directoryName: name, study: study))
                }
            } else if matched(name, analysisPattern) != nil {
                if let source = analysisSource(directory) { sources.insert(source) }
            } else if matched(name, evaluationPattern) != nil {
                if let source = evaluationSource(directory) { sources.insert(source) }
            }
        }
        let ownRuns = Set(completed.map(\.directoryName))

        return Facts(
            hasWorkspace: true,
            computeChoice: WorkspaceCompute.declaredChoice(root: root),
            modelCount: 0,
            studies: studies(root: root),
            isDemoCopy: demo != nil,
            demoModelID: demo?.modelID,
            completedRuns: completed,
            runsWithResults: completed.map(\.directoryName).filter(sources.contains),
            exportCount: exportCount(
                root: root, carriedExports: carriedExports, carriedRuns: carriedRuns,
                ownRuns: ownRuns),
            studiesWithAnyRun: studiesWithAnyRun)
    }

    // MARK: Reading, one fact at a time

    /// Run directories end `-exp-<study>-run`, with an optional `multi-agent-`
    /// before `run` and an optional `-N` after it (`VectorCatalog`'s naming).
    static let runPattern = #"-exp-(.+?)-(?:multi-agent-)?run(?:-\d+)?$"#
    static let analysisPattern = #"-exp-(.+?)-analyze(?:-\d+)?$"#
    static let evaluationPattern = #"-exp-(.+?)-evaluate(?:-judgment)?(?:-\d+)?$"#

    /// The study named in a directory name, when the name matches `pattern`.
    static func matched(_ name: String, _ pattern: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern),
            let match = regex.firstMatch(
                in: name, range: NSRange(name.startIndex..., in: name)),
            let range = Range(match.range(at: 1), in: name)
        else { return nil }
        return String(name[range])
    }

    /// A run's study as its own stamp says, for a run-named directory only.
    private static func study(ofRun name: String, in directory: URL) -> String? {
        guard matched(name, runPattern) != nil else { return nil }
        if let config = object(directory.appending(component: "config.json")),
            let experiment = config["experiment"] as? String, !experiment.isEmpty
        {
            return experiment
        }
        if let snapshot = object(directory.appending(component: "experiment.json")),
            let study = snapshot["name"] as? String, !study.isEmpty
        {
            return study
        }
        return nil
    }

    /// A run is complete once both its responses and its report are written.
    private static func isCompleteRun(_ directory: URL) -> Bool {
        let fm = FileManager.default
        return ["generations.jsonl", "report.json"].allSatisfy {
            fm.fileExists(atPath: directory.appending(component: $0).path)
        }
    }

    /// The run an analysis analyzed (a directory name), when it holds results.
    private static func analysisSource(_ directory: URL) -> String? {
        let fm = FileManager.default
        let report = object(directory.appending(component: "analysis.json"))
        guard
            report != nil
                || fm.fileExists(atPath: directory.appending(component: "effect-sizes.csv").path)
        else { return nil }
        if let text = try? String(
            contentsOf: directory.appending(component: "source-run.txt"), encoding: .utf8),
            let line = text.split(whereSeparator: \.isNewline).first
        {
            return baseName(String(line))
        }
        return (report?["sourceRun"] as? String).map(baseName)
    }

    /// The run an evaluation judged (a directory name), once it has a report.
    private static func evaluationSource(_ directory: URL) -> String? {
        for file in ["judge-report.json", "coding-report.json"] {
            guard let report = object(directory.appending(component: file)) else { continue }
            for key in ["sourceRun", "sourceRunDirectory"] {
                if let value = report[key] as? String, !value.isEmpty { return baseName(value) }
            }
        }
        return nil
    }

    private static func baseName(_ path: String) -> String {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        let withoutSlash = trimmed.hasSuffix("/") ? String(trimmed.dropLast()) : trimmed
        return withoutSlash.split(separator: "/").last.map(String.init) ?? withoutSlash
    }

    /// Studies under `experiments/`: a folder holding `experiment.json`, or an
    /// older flat `<name>.json`. Sorted by name.
    static func studies(root: URL) -> [Study] {
        let experiments = root.appending(component: "experiments")
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: experiments.path) else {
            return []
        }
        var found: [Study] = []
        for entry in entries where !entry.hasPrefix(".") {
            let url = experiments.appending(component: entry)
            let folderManifest = url.appending(component: "experiment.json")
            if fm.fileExists(atPath: folderManifest.path) {
                found.append(
                    Study(name: entry, status: object(folderManifest)?["status"] as? String))
            } else if entry.hasSuffix(".json") {
                found.append(
                    Study(
                        name: String(entry.dropLast(".json".count)),
                        status: object(url)?["status"] as? String))
            }
        }
        return found.sorted { $0.name < $1.name }
    }

    /// Exports under `exports/` that a results export wrote (its manifest
    /// names the kind), leaving out any a Demo Workspace carried in and any
    /// of a run that came with one.
    private static func exportCount(
        root: URL, carriedExports: Set<String>, carriedRuns: Set<String>, ownRuns: Set<String>
    ) -> Int {
        let exports = root.appending(component: "exports")
        return directoryNames(exports).filter { name in
            guard !carriedExports.contains(name),
                let manifest = object(
                    exports.appending(components: name, "manifest.json")),
                manifest["kind"] as? String == resultsExportKind
            else { return false }
            guard let run = manifest["run"] as? String else { return true }
            return !carriedRuns.contains(baseName(run))
        }.count
    }

    /// The kind a results export's `manifest.json` records
    /// (`results_export.EXPORT_KIND` in the Python client, which writes it).
    static let resultsExportKind = "steerlab.resultsExport"

    /// What a Demo Workspace copy's `demo.json` says about where it came from.
    private struct DemoOrigin {
        let backend: String
        let modelID: String?
    }

    private static func demoDescription(root: URL) -> DemoOrigin? {
        guard let document = object(root.appending(component: "demo.json")),
            let backend = document["backend"] as? String,
            DemoWorkspace.Backend(rawValue: backend) != nil
        else { return nil }
        let model = (document["model"] as? [String: Any])?["id"] as? String
        return DemoOrigin(backend: backend, modelID: model)
    }

    private static func directoryNames(_ url: URL) -> Set<String> {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: url.path) else { return [] }
        return Set(
            entries.filter { name in
                var isDirectory: ObjCBool = false
                return !name.hasPrefix(".")
                    && fm.fileExists(
                        atPath: url.appending(component: name).path, isDirectory: &isDirectory)
                    && isDirectory.boolValue
            })
    }

    private static func object(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }
}
