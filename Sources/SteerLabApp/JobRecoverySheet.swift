import ExperimentKit
import SwiftUI

/// Controller-recovery review: read the server's ownership evidence for a job,
/// then — only after asserting by hand that the recorded owner has exited —
/// record that assertion. It restarts nothing and authorizes no second
/// submission; the review token binds the assertion to the evidence that was
/// on screen when it was made.
struct JobRecoverySheet: View {
    let client: ClusterClient
    let jobID: String
    @Environment(\.dismiss) private var dismiss
    @State private var review: [String: JSONValue] = [:]
    @State private var report = ""
    @State private var reason = ""
    @State private var confirmed = false
    @State private var busy = false
    @State private var recovered = false
    /// The last failure, in its own line rather than pasted over the report:
    /// an error used to REPLACE the evidence the researcher was reading.
    @State private var failure: String?
    @State private var confirmingRecover = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            ScrollView {
                Text(report)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            failureLine
            TextField("Reason and evidence that the owner exited", text: $reason)
                .accessibilityLabel("Reason and evidence that the owner exited")
                .help("what you checked and what it showed — this is recorded "
                    + "with the assertion and is what a reader of the run will "
                    + "have to judge it by")
            Toggle("I have established that the recorded owner has exited",
                   isOn: $confirmed)
                .help("tick only after checking the owner yourself — the "
                    + "server records your assertion, it does not verify it")
            controls
        }
        .padding()
        .frame(minWidth: 760, minHeight: 540)
        .task { await load() }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Review controller recovery")
                .font(.title2)
            Text(client.profile.baseURL.absoluteString + " · " + jobID)
                .font(.caption)
                .textSelection(.enabled)
            Text("Recovery records your assertion that the original "
                + "controller has exited. Inspect its ownership evidence "
                + "first. It does not restart the computation or authorize a "
                + "second submission.")
        }
    }

    /// Always present so the sheet does not resize when a request fails.
    private var failureLine: some View {
        Label(failure ?? " ", systemImage: "exclamationmark.triangle.fill")
            .font(.caption)
            .foregroundStyle(.orange)
            .textSelection(.enabled)
            .lineLimit(2)
            .opacity(failure == nil ? 0 : 1)
            .accessibilityHidden(failure == nil)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 6) {
            // Why Recover is dim, in the sheet rather than only on hover.
            Text(recoverBlockedReason ?? " ")
                .font(.caption)
                .foregroundStyle(.orange)
                .lineLimit(1)
            HStack(spacing: 8) {
                Button(busy ? "Reading…" : "Refresh review") {
                    Task { await load() }
                }
                .disabled(busy)
                .help("re-read the server's ownership evidence for this job — "
                    + "it reads only, and it clears the tick above because the "
                    + "evidence may have changed")
                if busy {
                    ProgressView()
                        .controlSize(.small)
                }
                CopyButton(
                    "Copy review", systemImage: "doc.on.doc",
                    help: "copy the evidence above, exactly as the server "
                        + "returned it, to the clipboard"
                ) { report.isEmpty ? nil : report }
                .disabled(report.isEmpty)
                Button("Recover this job") { confirmingRecover = true }
                    .buttonStyle(.borderedProminent)
                    .disabled(recoverBlockedReason != nil)
                    .help("record your assertion that the recorded owner has "
                        + "exited — nothing restarts, and the assertion stays "
                        + "on the job for good")
                    .confirmationDialog(
                        "Record recovery of job \(jobID)?",
                        isPresented: $confirmingRecover,
                        titleVisibility: .visible
                    ) {
                        Button("Record recovery", role: .destructive) { recover() }
                        Button("Cancel", role: .cancel) {}
                    } message: {
                        Text("Your assertion and its reason are written to "
                            + "this job's record permanently and cannot be "
                            + "withdrawn from here. If the original controller "
                            + "is in fact still alive, two owners will believe "
                            + "they hold the job.")
                    }
                Spacer()
                Button("Close", role: .cancel) { dismiss() }
                    .keyboardShortcut(.cancelAction)
                    .help("close this sheet — nothing here is submitted until "
                        + "Recover is confirmed")
            }
        }
    }

    /// One reason at a time, in the order the researcher meets them.
    private var recoverBlockedReason: String? {
        if recovered { return "Recovery has already been recorded for this job." }
        if busy { return "Reading the server's evidence…" }
        if review["eligible"] != .bool(true) {
            return "The server does not consider this job eligible for "
                + "recovery — read the evidence above."
        }
        if !confirmed {
            return "Tick the box above once you have established that the "
                + "recorded owner has exited."
        }
        if reason.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "Say what you checked and what it showed."
        }
        return nil
    }

    private func recover() {
        guard case .string(let token) = review["reviewToken"] else {
            failure = "the server's review carries no token — refresh the "
                + "review and try again"
            return
        }
        guard !busy else { return }
        busy = true
        Task {
            defer { busy = false }
            do {
                let result = try await client.recoverJob(
                    jobID, reviewToken: token, reason: reason)
                report = describe(result)
                failure = nil
                recovered = true
            } catch {
                // The evidence on screen is NOT replaced by the failure:
                // losing it was the audit's complaint.
                failure = Self.describe(error)
            }
        }
    }

    private func load() async {
        guard !busy else { return }
        busy = true
        confirmed = false
        defer { busy = false }
        do {
            let result = try await client.recoveryReview(jobID)
            if case .object(let object) = result { review = object }
            report = describe(result)
            failure = nil
        } catch {
            review = [:]
            failure = Self.describe(error)
        }
    }

    private func describe(_ value: JSONValue) -> String {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return (try? String(decoding: encoder.encode(value), as: UTF8.self))
            ?? "Could not display recovery response."
    }

    /// The server's own words for a refusal, never the Swift debug form.
    private static func describe(_ error: any Error) -> String {
        if let client = error as? ClusterClient.ClientError {
            return ClusterClient.unwrappingDetail(client).description
        }
        return error.localizedDescription
    }
}
