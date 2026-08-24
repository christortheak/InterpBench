import AppKit
import ExperimentKit
import SwiftUI

/// The app-side half of the update SIGNPOST (`ExperimentKit/UpdateCheck.swift`
/// holds the policy and both injected seams). Thin by construction: this file
/// owns presentation state and the two affordances — a "Check for Updates…"
/// menu item and a dismissible launch banner — and delegates every decision.
///
/// Nothing here downloads or installs anything. The one action offered is
/// opening the releases page in the person's browser.
@MainActor
@Observable
final class UpdateSignpostModel {
    /// The banner sentence, non-nil only when a newer RELEASE was found and
    /// its version has not been dismissed on this machine.
    private(set) var bannerMessage: String?
    /// Where the banner's button and the manual report send a person.
    private(set) var downloadPage: URL?
    /// Manual-check result, shown as an alert. Set only by the menu item, so
    /// an automatic check that finds nothing (or cannot reach anything) stays
    /// completely silent.
    var manualReport: String?
    private(set) var isChecking = false
    /// The preference behind the automatic launch check (default ON). Mirrored
    /// as a stored property so the menu item and the banner track it; the
    /// durable copy lives in `UpdateCheckPreferences`.
    private(set) var automaticChecksEnabled: Bool

    private var lastAvailability: UpdateAvailability?
    private let preferences: UpdateCheckPreferences
    private let service: UpdateCheckService

    init(
        service: UpdateCheckService = UpdateCheckService(),
        preferences: UpdateCheckPreferences = UpdateCheckPreferences()
    ) {
        self.service = service
        self.preferences = preferences
        self.automaticChecksEnabled = preferences.automaticChecksEnabled
    }

    func setAutomaticChecks(_ enabled: Bool) {
        automaticChecksEnabled = enabled
        preferences.automaticChecksEnabled = enabled
    }

    /// The binding the toggle in the menu and the banner share.
    var automaticChecksBinding: Binding<Bool> {
        Binding(get: { self.automaticChecksEnabled }, set: { self.setAutomaticChecks($0) })
    }

    /// Launch path: at most one check per 24 h, and only while the preference
    /// is on. Called from a `.task`, i.e. after the window is up, and the
    /// whole check runs off the main actor — launch is untouched.
    func runAutomaticCheckIfDue() async {
        guard preferences.shouldRunAutomaticCheck else { return }
        preferences.recordCheck()
        let availability = await service.check()
        apply(availability)
    }

    /// Menu path: always runs, and always reports — including the honest
    /// "could not reach…" an automatic check swallows.
    func checkNow() async {
        guard !isChecking else { return }
        isChecking = true
        defer { isChecking = false }
        preferences.recordCheck()
        let availability = await service.check()
        apply(availability)
        manualReport = UpdateCheckPolicy.manualReport(for: availability)
    }

    /// Hide the banner and remember the version, so this release never
    /// banners again on this machine.
    func dismissBanner() {
        if case .releaseAvailable(let version, _) = lastAvailability {
            preferences.dismiss(version)
        }
        bannerMessage = nil
    }

    func openDownloadPage() {
        guard let downloadPage else { return }
        NSWorkspace.shared.open(downloadPage)
    }

    private func apply(_ availability: UpdateAvailability) {
        lastAvailability = availability
        bannerMessage = preferences.bannerMessage(for: availability)
        if case .releaseAvailable(_, let page) = availability { downloadPage = page }
    }
}

/// Unobtrusive one-line banner above the main content. Present only for a
/// newer packaged release; the weaker "newer source" signal never gets here.
struct UpdateBanner: View {
    let model: UpdateSignpostModel

    var body: some View {
        if let message = model.bannerMessage {
            HStack(spacing: 10) {
                Image(systemName: "arrow.down.circle")
                    .foregroundStyle(.tint)
                Text(message)
                    .font(.callout)
                Spacer(minLength: 8)
                Button("Open Releases…") { model.openDownloadPage() }
                    .help(
                        "open the repository's Releases page in your browser — "
                            + "SteerLab never downloads or installs anything itself")
                Toggle("Check automatically", isOn: model.automaticChecksBinding)
                    .toggleStyle(.checkbox)
                    .font(.caption)
                    .help(
                        "check once a day for a newer release; turning this off "
                            + "leaves the Check for Updates… menu item")
                Button {
                    model.dismissBanner()
                } label: {
                    Image(systemName: "xmark")
                }
                .buttonStyle(.borderless)
                .help("dismiss — this version will not be announced again")
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 7)
            .background(.thinMaterial)
            .overlay(alignment: .bottom) { Divider() }
            .transition(.move(edge: .top).combined(with: .opacity))
        }
    }
}

/// The app-menu affordances, placed after the About item.
struct UpdateCommands: View {
    let model: UpdateSignpostModel

    var body: some View {
        Button("Check for Updates…") {
            Task { await model.checkNow() }
        }
        .disabled(model.isChecking)
        Toggle("Check for Updates Automatically", isOn: model.automaticChecksBinding)
    }
}
