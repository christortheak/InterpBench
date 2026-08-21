import ExperimentKit
import SwiftUI

// Cluster-testing item 2 (2026-07-21): ONE shared "no GPU session" warning
// for every app-side submission of model-running cluster work — server LoRA
// training, server study run/sweep/extract/validate — instead of per-panel
// dialogs. Extended the same day by the incident's part 1: bundle
// submissions also carry their OWN resource request (executor + gres), and
// a submission that requests no GPU on a Slurm site executes inside the
// controller's small CPU allocation and dies mid-load. The predicate and
// copy live in `GPUSessionPreflight` + `ModelJobSubmissionPreflight`
// (ExperimentKit, unit-tested); this file is the presentation glue.
//
// One dialog, the most relevant message: a missing allocation is MORE
// specific than a missing session (it wins the title and the primary fix
// affordance); when both apply the single dialog says both.
//
// Honest-researcher rule: this is a warning with an override, never a hard
// refusal — Slurm queuing a job for later, and CPU smoke tests with tiny
// models, are legitimate choices. Deliberately unguarded: non-model jobs
// (bundle import, file ops, status polls), dry-run bundle submissions
// (nothing executes), the `analyze` verb (statistics over an existing run),
// and interactive model load/generate — those routes are proxied through
// the session itself and already answer with the server's own "no GPU
// session — start one" refusal hint.

/// A model-running server submission parked while the dialog asks.
struct PendingModelJob: Identifiable {
    let id = UUID()
    /// Plain-language name of what would be submitted ("study run",
    /// "server LoRA training") — shown in the dialog title.
    let label: String
    /// Which warning this dialog carries (allocation beats session).
    let concern: ModelJobSubmissionPreflight.Concern
    /// The submission's Remote options, for the message copy (nil for
    /// session-only warnings on non-bundle submissions).
    let options: ModelJobSubmissionPreflight.BundleOptions?
    /// The "Fix options (request a GPU)" action: reveal + prefill the
    /// Remote options. Nil hides the affordance.
    let fixOptions: (@MainActor () -> Void)?
    let action: @MainActor () async -> Void
}

@MainActor
enum ModelJobGPUGate {
    /// Route a model-running server submission through the warning. Two
    /// composed rules, one dialog (`ModelJobSubmissionPreflight.concern`):
    ///
    /// - `bundleOptions` non-nil (bundle submissions): on a Slurm-scheduler
    ///   site, a non-slurm executor or empty gres parks the job behind the
    ///   no-GPU-ALLOCATION dialog (fix options / submit anyway / cancel) —
    ///   the 2026-07-21 controller-allocation failure. Dry runs and the
    ///   analyze verb pass straight through, and a CORRECT Slurm bundle
    ///   (executor slurm, non-empty gres) submits silently even with no
    ///   session: it owns its GPU allocation (external review 2026-07-22).
    /// - otherwise the historical no-GPU-SESSION warning: active server
    ///   workspace whose profile offers GPU sessions and none is live or
    ///   starting.
    ///
    /// No concern → the action runs immediately.
    static func submit(
        _ label: String,
        service: ChatService,
        pending: Binding<PendingModelJob?>,
        bundleOptions: ModelJobSubmissionPreflight.BundleOptions? = nil,
        fixOptions: (@MainActor () -> Void)? = nil,
        action: @escaping @MainActor () async -> Void
    ) {
        let cluster = service.cluster
        let sessionWarning: Bool = {
            guard case .server = cluster.activeWorkspace else { return false }
            return cluster.gpuSession.shouldWarnBeforeModelRunningJob
        }()
        let siteUsesSlurm: Bool = {
            if case .slurm = cluster.activeSite?.scheduler { return true }
            return false
        }()
        guard let concern = ModelJobSubmissionPreflight.concern(
            siteUsesSlurmScheduler: siteUsesSlurm,
            options: bundleOptions,
            sessionWarning: sessionWarning)
        else {
            Task { await action() }
            return
        }
        pending.wrappedValue = PendingModelJob(
            label: label, concern: concern, options: bundleOptions,
            fixOptions: fixOptions, action: action)
    }
}

extension View {
    /// Attach once per guarded view (each keeps its own
    /// `@State var pendingModelJob: PendingModelJob?`); presents the
    /// dialog for the job parked in `pending`.
    func modelJobGPUWarning(
        pending: Binding<PendingModelJob?>, service: ChatService
    ) -> some View {
        modifier(ModelJobGPUWarningModifier(pending: pending, service: service))
    }
}

private struct ModelJobGPUWarningModifier: ViewModifier {
    @Binding var pending: PendingModelJob?
    let service: ChatService

    func body(content: Content) -> some View {
        content.confirmationDialog(
            pending.map {
                ModelJobSubmissionPreflight.title(for: $0.concern, jobLabel: $0.label)
            } ?? "Submit this job?",
            isPresented: presented,
            titleVisibility: .visible,
            presenting: pending
        ) { job in
            switch job.concern {
            case .noGPUAllocation:
                if let fixOptions = job.fixOptions {
                    Button(ModelJobSubmissionPreflight.fixOptionsButtonLabel) {
                        fixOptions()
                    }
                }
                Button("Submit anyway") {
                    Task { await job.action() }
                }
                Button("Cancel", role: .cancel) {}
            case .noGPUSession:
                Button("Start GPU session, then submit") {
                    let service = service
                    Task {
                        // Same defaults machinery the Playground start sheet
                        // pre-fills; the toolbar control shows the session
                        // queueing while the job is submitted behind it.
                        await GPUSessionStartFlow.startWithDefaults(service: service)
                        await job.action()
                    }
                }
                Button("Submit anyway") {
                    Task { await job.action() }
                }
                Button("Cancel", role: .cancel) {}
            }
        } message: { job in
            Text(ModelJobSubmissionPreflight.message(
                for: job.concern,
                substrate: service.cluster.substrateLabel,
                options: job.options))
        }
    }

    private var presented: Binding<Bool> {
        Binding(
            get: { pending != nil },
            set: { if !$0 { pending = nil } })
    }
}
