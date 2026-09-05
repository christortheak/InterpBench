import Foundation
import Observation
import SteeringKit

/// Local operation lifetime, cancellation and live progress. Draft selection is supplied at launch.
@Observable @MainActor
public final class StudyLocalJobController {
    public internal(set) var isValidating = false
    public internal(set) var isRunning = false
    public internal(set) var isEvaluating = false
    /// A local extraction is executing (A11). Extraction shares the GPU with
    /// runs/validation/sweeps, so all four flags gate each other.
    public internal(set) var isExtracting = false
    public internal(set) var lastExtractDirectory: String?
    /// A local sweep is executing (Optimizations' Optimize). Sweeps share the GPU
    /// with runs/validation, so all three flags gate each other.
    public internal(set) var isSweeping = false
    public internal(set) var lastValidationDirectory: String?
    public internal(set) var lastRunDirectory: String?
    public internal(set) var lastEvaluationDirectory: String?
    public internal(set) var liveRunDirectory: String?
    public internal(set) var liveEvaluationDirectory: String?
    public internal(set) var liveActiveGeneration: LiveStudyGeneration?
    public internal(set) var liveActiveJudgment: LiveStudyJudgment?
    public internal(set) var liveGenerations: [StudyGenerationPreview] = []
    public internal(set) var liveJudgments: [StudyJudgePreview] = []
    /// A local-sweep cancellation was requested (Optimizations' Cancel):
    /// `ExperimentTasks.sweep` polls this between generations and stops
    /// after the current one, keeping partial grid rows. Reset when the
    /// next local sweep starts.
    public internal(set) var sweepCancelRequested = false
    /// A local study-run cancellation was requested: `ExperimentTasks.run`
    /// polls this between generations/choice items/battery items and stops
    /// after the current one — partial artifacts stay, marked by a
    /// cancelled.txt note, and no report.json is written. Reset when the
    /// next local run starts. (Server-routed runs are durable jobs with
    /// their own Cancel Server Job control.)
    public internal(set) var studyRunCancelRequested = false
    /// Same flag for `ExperimentTasks.validate` (polled between concepts,
    /// scenarios, control extractions, and battery items; a cancelled
    /// validation writes NO evidence).
    public internal(set) var validationCancelRequested = false
    /// Same flag for `ExperimentTasks.evaluatePairedJudge` (polled between
    /// judgments; completed judgments stay, no judge report is written).
    public internal(set) var evaluationCancelRequested = false
    /// Same flag for `ExperimentTasks.extract` (A11; polled between
    /// concepts — completed concepts keep their sidecar artifacts, and the
    /// run directory is marked cancelled).
    public internal(set) var extractCancelRequested = false
    @ObservationIgnored var presentation = StudyJobPresentation() {
        didSet { displayLog.presentation = presentation }
    }
    @ObservationIgnored private let displayLog = StudyDisplayLog()
    private var status: String? { didSet { presentation.status(status) } }
    public init() {}
    private func note(_ message: String, severity: PanelNotice.Severity) {
        presentation.note(message, severity)
    }
    func beginDisplayLog(title: String, initialLine: String) {
        displayLog.begin(title: title, initialLine: initialLine)
    }
    func appendDisplayLog(_ line: String) { displayLog.append(line) }
    func endDisplayLog(_ line: String? = nil) { displayLog.end(line) }

    public func clearLiveViewer() {
        resetLiveViewer()
    }
    func resetLiveViewer() {
        liveRunDirectory = nil
        liveEvaluationDirectory = nil
        liveActiveGeneration = nil
        liveActiveJudgment = nil
        liveGenerations = []
        liveJudgments = []
    }
    func handleStudyProgress(_ event: ExperimentTasks.StudyTaskProgress) {
        switch event {
        case .runDirectory(let path):
            liveRunDirectory = path
            status = "writing study artifacts to \(URL(filePath: path).lastPathComponent)…"
            appendDisplayLog("run directory: \(URL(filePath: path).lastPathComponent)")
        case .generationStarted(let condition, let promptID, let prompt):
            liveActiveGeneration = LiveStudyGeneration(
                condition: condition,
                promptID: promptID,
                prompt: prompt,
                output: "")
            status = "generating \(condition) · \(promptID)…"
            appendDisplayLog("generating [\(condition)] \(promptID)…")
        case .generationChunk(let condition, let promptID, let output):
            // Per-token chunks update only the in-panel viewer — mirroring
            // every chunk would flood the display-pane log.
            liveActiveGeneration = LiveStudyGeneration(
                condition: condition,
                promptID: promptID,
                prompt: liveActiveGeneration?.prompt ?? "",
                output: output)
        case .generationCompleted(let generation):
            liveGenerations.insert(generation, at: 0)
            liveActiveGeneration = nil
            status = "generated \(generation.condition) · \(generation.promptID)"
            appendDisplayLog(
                "generated [\(generation.condition)] \(generation.promptID): "
                    + "\(generation.wordCount) words")
        case .evaluationDirectory(let path):
            liveEvaluationDirectory = path
            status = "writing judge artifacts to \(URL(filePath: path).lastPathComponent)…"
            appendDisplayLog("judge directory: \(URL(filePath: path).lastPathComponent)")
        case .judgmentStarted(let condition, let promptID):
            liveActiveJudgment = LiveStudyJudgment(condition: condition, promptID: promptID)
            status = "judging \(condition) · \(promptID)…"
        case .judgmentCompleted(let judgment):
            liveJudgments.insert(judgment, at: 0)
            liveActiveJudgment = nil
            status =
                "judged \(judgment.condition) · \(judgment.promptID): \(judgment.conditionResult)"
            appendDisplayLog(
                "judged [\(judgment.condition)] \(judgment.promptID): "
                    + "\(judgment.conditionResult)")
        case .codingCompleted(let coding):
            liveActiveJudgment = nil
            let codes = coding.codes.sorted { $0.key < $1.key }
                .map { "\($0.key)=\($0.value.displayString)" }
                .joined(separator: ", ")
            status =
                "coded \(coding.condition) · \(coding.promptID) "
                + "[\(coding.judge)]"
            appendDisplayLog(
                "coded [\(coding.condition)] \(coding.promptID) "
                    + "(\(coding.judge)): \(codes)")
        }
    }
    public func cancelExtract() {
        guard isExtracting, !extractCancelRequested else { return }
        extractCancelRequested = true
        note(
            "cancelling extraction — stops after the current concept; "
                + "completed vectors stay in the run directory", severity: .warning)
        appendDisplayLog(
            "cancellation requested — extraction stops after the current "
                + "concept; completed vectors stay")
    }
    public func cancelStudyRun() {
        guard isRunning, !studyRunCancelRequested else { return }
        studyRunCancelRequested = true
        note(
            "cancelling study run — stops after the current generation; "
                + "partial artifacts stay in the run directory", severity: .warning)
        appendDisplayLog(
            "cancellation requested — the run stops after the current "
                + "generation; partial artifacts stay (no report.json)")
    }
    public func cancelValidation() {
        guard isValidating, !validationCancelRequested else { return }
        validationCancelRequested = true
        note(
            "cancelling validation — stops after the current unit; "
                + "no validation evidence will be written", severity: .warning)
        appendDisplayLog(
            "cancellation requested — validation stops after the current "
                + "unit; no evidence is written")
    }
    public func cancelPairedJudge() {
        guard isEvaluating, !evaluationCancelRequested else { return }
        evaluationCancelRequested = true
        note(
            "cancelling paired judge — stops after the current judgment; "
                + "completed judgments stay, no judge report is written", severity: .warning)
        appendDisplayLog(
            "cancellation requested — judging stops after the current "
                + "judgment; no judge report is written")
    }
    public func runStudy(experimentName name: String) async {
        guard !isRunning else { return }
        isRunning = true
        studyRunCancelRequested = false
        resetLiveViewer()
        note("running study '\(name)'…", severity: .info)
        beginDisplayLog(
            title: "Study run — \(name)",
            initialLine: "verifying pins and loading the pinned model…")
        defer {
            isRunning = false
            presentation.refresh()
        }
        do {
            let runDirectory = try await ExperimentTasks.run(
                experimentName: name,
                shouldCancel: { [weak self] in
                    await self?.studyRunCancelRequested ?? false
                },
                progress: { [weak self] event in
                    await MainActor.run {
                        self?.handleStudyProgress(event)
                    }
                })
            lastRunDirectory = runDirectory.path
            presentation.refresh()
            presentation.selectResult(name, runDirectory.lastPathComponent)
            if studyRunCancelRequested {
                // Cancelled by user: partial artifacts on disk, honestly
                // marked — never reported as an error or a completion.
                note(
                    "study run cancelled by user — partial artifacts kept "
                        + "in \(runDirectory.lastPathComponent) (no report.json; "
                        + "not a completed run)", severity: .warning)
                endDisplayLog(
                    "study run cancelled by user — partial artifacts kept in "
                        + runDirectory.lastPathComponent)
            } else {
                note("study run complete: \(runDirectory.lastPathComponent)", severity: .success)
                endDisplayLog("study run complete: \(runDirectory.lastPathComponent)")
            }
        } catch {
            presentation.refresh()
            note(
                "The study run failed — no report.json was written; any "
                    + "partial run directory remains on disk for inspection. "
                    + "Details: \(error)",
                severity: .error)
            endDisplayLog("study run failed: \(error)")
        }
    }
    public func validateStudy(experimentName name: String) async {
        guard !isValidating else { return }
        isValidating = true
        validationCancelRequested = false
        note("validating study '\(name)'…", severity: .info)
        beginDisplayLog(
            title: "Study validation — \(name)",
            initialLine: "verifying pins and loading the pinned model…")
        defer {
            isValidating = false
            presentation.refresh()
        }
        do {
            let runDirectory = try await ExperimentTasks.validate(
                experimentName: name,
                shouldCancel: { [weak self] in
                    await self?.validationCancelRequested ?? false
                },
                log: { [weak self] line in
                    await MainActor.run {
                        self?.appendDisplayLog(line)
                        self?.status = line
                    }
                })
            if validationCancelRequested {
                note(
                    "validation cancelled by user — partial artifacts kept "
                        + "in \(runDirectory.lastPathComponent); no validation "
                        + "evidence was written", severity: .warning)
                endDisplayLog(
                    "validation cancelled by user — no evidence written")
            } else {
                lastValidationDirectory = runDirectory.path
                note("validation complete: \(runDirectory.lastPathComponent)", severity: .success)
                endDisplayLog("validation complete: \(runDirectory.lastPathComponent)")
            }
        } catch {
            note(
                "Validation failed — no validation evidence was written, so "
                    + "freeze will still ask for a matching validate run. Fix "
                    + "the cause and validate again. Details: \(error)",
                severity: .error)
            endDisplayLog("validation failed: \(error)")
        }
    }
    public func extractStudy(experimentName name: String) async {
        guard !isExtracting else { return }
        isExtracting = true
        extractCancelRequested = false
        note("extracting vectors for '\(name)'…", severity: .info)
        beginDisplayLog(
            title: "Vector extraction — \(name)",
            initialLine: "verifying pins and loading the pinned model…")
        defer {
            isExtracting = false
            presentation.refresh()
        }
        do {
            try await ExperimentTasks.extract(
                experimentName: name,
                shouldCancel: { [weak self] in
                    await self?.extractCancelRequested ?? false
                })
            let runDirectory = ExperimentStore.newestRunDirectory(
                experimentName: name, task: "extract")
            lastExtractDirectory = runDirectory?.path
            presentation.refresh()
            if let id = runDirectory?.lastPathComponent {
                presentation.selectResult(name, id)
            }
            if extractCancelRequested {
                note(
                    "extraction cancelled by user — completed vectors kept in "
                        + (runDirectory?.lastPathComponent ?? "the run directory")
                        + " (marked cancelled)",
                    severity: .warning)
                endDisplayLog("extraction cancelled by user — partial vectors kept")
            } else {
                note(
                    "extraction complete: "
                        + (runDirectory?.lastPathComponent ?? "see runs/"),
                    severity: .success)
                endDisplayLog(
                    "extraction complete: "
                        + (runDirectory?.lastPathComponent ?? "see runs/"))
            }
        } catch {
            presentation.refresh()
            note(
                "Vector extraction failed — any completed vectors remain in "
                    + "the run directory; nothing pinned in the study changed. "
                    + "Details: \(error)",
                severity: .error)
            endDisplayLog("extraction failed: \(error)")
        }
    }
    public func runSweep(experimentName name: String) async {
        guard !isSweeping, !isRunning, !isValidating else { return }
        isSweeping = true
        sweepCancelRequested = false
        note("sweeping '\(name)'…", severity: .info)
        beginDisplayLog(
            title: "Optimization sweep — \(name)",
            initialLine: "verifying pins — the sweep loads the pinned model itself…")
        defer {
            isSweeping = false
            presentation.refresh()
        }
        do {
            try await ExperimentTasks.sweep(
                experimentName: name,
                shouldCancel: { [weak self] in
                    await self?.sweepCancelRequested ?? false
                },
                log: { [weak self] line in
                    await MainActor.run {
                        self?.appendDisplayLog(line)
                        self?.status = line
                    }
                })
            presentation.refresh()
            if sweepCancelRequested {
                // The sweep returned normally with a partial grid (server
                // parity) — never report a cancelled run as complete.
                note(
                    "sweep cancelled for '\(name)' — partial grid rows "
                        + "kept in the run directory; no recommendation from an "
                        + "incomplete grid", severity: .warning)
                endDisplayLog("sweep cancelled for '\(name)' — partial rows kept")
            } else {
                note(
                    "sweep complete for '\(name)' — grid and recommendations updated",
                    severity: .success)
                endDisplayLog("sweep complete for '\(name)'")
            }
        } catch {
            presentation.refresh()
            note("sweep failed: \(error)", severity: .error)
            endDisplayLog("sweep failed: \(error)")
        }
    }
    public func runPairedJudgeEvaluation(
        experimentName name: String, sourceRun item: StudyRunListItem,
        evaluation: ExperimentManifest.EvaluationSpec?, hasPinnedRubric: Bool
    ) async {
        guard !isEvaluating else { return }
        isEvaluating = true
        evaluationCancelRequested = false
        liveEvaluationDirectory = nil
        liveActiveJudgment = nil
        liveJudgments = []
        note("running paired judge for '\(item.directoryName)'…", severity: .info)
        // A pinned rubric file makes inline draft text optional; with
        // neither, there is nothing to judge with.
        if evaluation == nil,
            !hasPinnedRubric
        {
            note("enter a paired judge rubric or pin a rubric file first", severity: .info)
            isEvaluating = false
            return
        }
        beginDisplayLog(
            title: "Paired judge — \(name)",
            initialLine: "judging run \(item.directoryName)…")
        defer {
            isEvaluating = false
            presentation.refresh()
        }
        do {
            let url = try await ExperimentTasks.evaluatePairedJudge(
                experimentName: name,
                sourceRunDirectory: URL(filePath: item.path),
                evaluation: evaluation,
                shouldCancel: { [weak self] in
                    await self?.evaluationCancelRequested ?? false
                },
                progress: { [weak self] event in
                    await MainActor.run {
                        self?.handleStudyProgress(event)
                    }
                })
            presentation.refresh()
            presentation.selectResult(name, item.id)
            if evaluationCancelRequested {
                note(
                    "paired judge cancelled by user — completed judgments "
                        + "kept in \(url.lastPathComponent); no judge report written",
                    severity: .warning)
                endDisplayLog(
                    "paired judge cancelled by user — no judge report written")
            } else {
                lastEvaluationDirectory = url.path
                note("paired judge complete: \(url.lastPathComponent)", severity: .success)
                endDisplayLog("paired judge complete: \(url.lastPathComponent)")
            }
        } catch {
            presentation.refresh()
            note(
                "Paired judging failed — no judge report was written; the "
                    + "source run is untouched, so judging can simply be run "
                    + "again. Details: \(error)",
                severity: .error)
            endDisplayLog("paired judge failed: \(error)")
        }
    }
    public func cancelSweep() {
        guard isSweeping, !sweepCancelRequested else { return }
        sweepCancelRequested = true
        note("cancelling optimization — stops after the current generation", severity: .warning)
        appendDisplayLog(
            "cancellation requested — the sweep stops after the current "
                + "generation; partial rows stay in the run directory")
    }
}
