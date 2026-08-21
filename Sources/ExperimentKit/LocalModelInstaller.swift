import Foundation
import Observation
import SteeringKit

/// Local (MLX) counterpart to `ClusterConnectionStore.installModel`: fetch a
/// Hugging Face repo into this Mac's HF cache as its own visible, cancellable
/// operation, so the installed-models registry every builder reads
/// (`SubstrateCatalog` over `SteeredContainerLoader.localModelIDs`) can be
/// grown without a Load button silently pulling several gigabytes.
///
/// The server workspace has had this shape since the beginning — a named
/// install that queues a durable job and reports progress; Local had only the
/// implicit download hidden inside `ChatService.loadModel()`. This is the same
/// affordance for the same registry, not a special case: install here, and the
/// model shows up installed in the Playground picker *and* in every builder's
/// installed-models selector.
///
/// Owned by `ChatService` (one instance per app) so a multi-gigabyte fetch
/// survives view churn, exactly like `LocalServerController` and
/// `LocalEngineProvisioner` are owned above the toolbar views that drive them.
@Observable @MainActor
public final class LocalModelInstaller {

    public enum Phase: Equatable, Sendable {
        case idle
        /// Fetching; `percent` is the hub client's own progress (0…100).
        case installing(modelID: String, percent: Int)
        case finished(modelID: String)
        case cancelled(modelID: String)
        case failed(modelID: String, reason: String)
    }

    public private(set) var phase: Phase = .idle

    /// The fetch itself, injectable so the state machine is testable without
    /// a network or a multi-gigabyte download. Reports 0…100.
    private let fetch:
        @Sendable (String, @Sendable @escaping (Int) -> Void) async throws -> Void

    private var task: Task<Void, Never>?
    /// Bumped per install so a cancelled predecessor's tail cannot clobber
    /// its successor's phase or task handle.
    private var epoch = 0

    public init(
        fetch: (
            @Sendable (String, @Sendable @escaping (Int) -> Void) async throws -> Void
        )? = nil
    ) {
        self.fetch = fetch ?? { modelID, report in
            try await SteeredContainerLoader.downloadSnapshot(modelID: modelID) { progress in
                report(Int(progress.fractionCompleted * 100))
            }
        }
    }

    // MARK: Derived state

    public var isInstalling: Bool {
        if case .installing = phase { return true }
        return false
    }

    public var isFailed: Bool {
        if case .failed = phase { return true }
        return false
    }

    /// The model currently being fetched, if any — so a picker can badge that
    /// row rather than only showing a status line somewhere else.
    public var installingModelID: String? {
        if case .installing(let modelID, _) = phase { return modelID }
        return nil
    }

    /// One sentence for the UI. Never empty while something is in flight, and
    /// a failure keeps its reason on screen until the next install starts.
    public var statusLine: String? {
        switch phase {
        case .idle:
            return nil
        case .installing(let modelID, let percent):
            return "downloading \(modelID) — \(percent)%"
        case .finished(let modelID):
            return "\(modelID) is installed and ready to load"
        case .cancelled(let modelID):
            return "download of \(modelID) cancelled — partial files stay in "
                + "the Hugging Face cache and a re-run resumes"
        case .failed(let modelID, let reason):
            return "could not install \(modelID): \(reason)"
        }
    }

    // MARK: Actions

    /// Start an install. Returns the refusal reason when it will not start
    /// (already installing, empty/malformed slug), nil when it did.
    @discardableResult
    public func install(_ modelID: String) -> String? {
        let trimmed = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        if let refusal = Self.installRefusal(slug: modelID, isInstalling: isInstalling) {
            // A refusal must be visible, not silent — except when it is "one
            // is already running", whose in-flight status line is the more
            // useful thing to keep on screen.
            if !isInstalling {
                phase = .failed(modelID: trimmed, reason: refusal)
            }
            return refusal
        }
        phase = .installing(modelID: trimmed, percent: 0)
        epoch += 1
        let epoch = self.epoch
        let fetch = self.fetch
        // Strong capture: the fetch is the whole point of this object, and a
        // dropped installer mid-download would leave the phase — and the
        // Playground's progress row — frozen at whatever it last showed. The
        // task ends, so the reference does too.
        task = Task { [self] in
            do {
                try await fetch(trimmed) { percent in
                    Task { @MainActor in
                        self.reportProgress(epoch: epoch, modelID: trimmed, percent: percent)
                    }
                }
                // A cancelled-then-restarted install has a live successor: its
                // predecessor's tail must not overwrite the new phase or steal
                // the new task handle.
                guard epoch == self.epoch else { return }
                if Task.isCancelled {
                    phase = .cancelled(modelID: trimmed)
                } else {
                    phase = .finished(modelID: trimmed)
                    onFinished?(trimmed)
                }
            } catch is CancellationError {
                guard epoch == self.epoch else { return }
                phase = .cancelled(modelID: trimmed)
            } catch {
                guard epoch == self.epoch else { return }
                phase = .failed(modelID: trimmed, reason: error.localizedDescription)
            }
            if epoch == self.epoch { task = nil }
        }
        return nil
    }

    /// Progress arrives off the main actor and can land after a cancel or a
    /// restart — only the still-running install may move the bar.
    private func reportProgress(epoch: Int, modelID: String, percent: Int) {
        guard epoch == self.epoch else { return }
        guard case .installing(let current, _) = phase, current == modelID else { return }
        phase = .installing(modelID: modelID, percent: max(0, min(100, percent)))
    }

    /// Stop waiting for (and stop fetching) the current install. Partial
    /// files stay in the HF cache; a re-run resumes from them.
    public func cancel() {
        task?.cancel()
        task = nil
        if case .installing(let modelID, _) = phase {
            phase = .cancelled(modelID: modelID)
        }
    }

    public func clearStatus() {
        guard !isInstalling else { return }
        phase = .idle
    }

    /// Called on the main actor after a successful install — the seam the
    /// host uses to re-scan the installed-models registry so pickers flip
    /// from "not installed" to selectable without a relaunch.
    public var onFinished: (@MainActor (String) -> Void)?

    // MARK: Pure rules (unit-tested without a network)

    /// Why this install cannot start, or nil when it may. Slug shape is the
    /// hub's own: `owner/repo`, no whitespace, no leading or trailing slash.
    public nonisolated static func installRefusal(
        slug: String, isInstalling: Bool
    ) -> String? {
        if isInstalling {
            return "an install is already running — wait for it to finish or cancel it"
        }
        let trimmed = slug.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return "enter a Hugging Face repo id to install" }
        if trimmed.contains(" ") {
            return "a Hugging Face repo id has no spaces (e.g. mlx-community/gemma-3-4b-it-4bit)"
        }
        let parts = trimmed.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2, !parts[0].isEmpty, !parts[1].isEmpty else {
            return "use the owner/repo form (e.g. mlx-community/gemma-3-4b-it-4bit)"
        }
        return nil
    }
}
