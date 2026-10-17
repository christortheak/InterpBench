import AppKit
import ExperimentKit
import SwiftUI

/// The app-side half of the updater (`ExperimentKit/UpdateCheck.swift` finds
/// a newer release; `ExperimentKit/UpdateInstaller.swift` installs it). This
/// file owns presentation state and the affordances — a "Check for Updates…"
/// menu item, a dismissible launch banner, the Install Update confirmation
/// sheet, and the one-time notice on the first launch after an update — and
/// delegates every decision.
///
/// Nothing is installed without the person's click on Install Update and
/// their confirmation in the sheet. A copy that may not install itself (a
/// developer install, an ad-hoc signature, a translocated or read-only copy,
/// …) keeps the signpost to the Releases page and says why in one line.
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

    // MARK: Installation state

    /// The release this copy may install (the check found it, its files are
    /// there, and this copy is eligible), or nil.
    private(set) var installableRelease: PublishedRelease?
    private(set) var installAssets: UpdateAssetPair?
    private(set) var installTarget: UpdateInstallTarget?
    /// Why a newer release cannot be installed from here, in one plain line.
    private(set) var installRefusal: String?
    var showingInstallSheet = false
    private(set) var installState: InstallState = .reviewing
    /// What is still running, refreshed while the sheet is open.
    private(set) var busy = UpdateBusyState()
    /// The one-time notice on the first launch of a newer version.
    private(set) var afterInstallNotice: UpdateAfterInstallNotice?

    enum InstallState: Equatable {
        /// The confirmation, before the person has said Install.
        case reviewing
        case working(UpdateInstallProgress)
        /// Verified and staged, but something started meanwhile.
        case waitingForIdle
        case installing
        case failed(UpdateInstallError)
    }

    /// Supplied by the app: what is running right now.
    @ObservationIgnored var busyProbe: (@MainActor () async -> UpdateBusyState)?
    /// Supplied by the app: opens the local engine setup (Refresh Engine).
    @ObservationIgnored var openEngineSetup: (@MainActor () -> Void)?

    private var lastAvailability: UpdateAvailability?
    private let preferences: UpdateCheckPreferences
    private let service: UpdateCheckService
    @ObservationIgnored private var installTask: Task<Void, Never>?
    @ObservationIgnored private var busyTask: Task<Void, Never>?
    @ObservationIgnored private var installer: UpdateInstaller?
    @ObservationIgnored private var staged: UpdateStagedApp?

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

    // MARK: Launch

    /// Once per launch, before the automatic check: remove anything an
    /// interrupted update left beside the app, say "updated" once on the
    /// first launch of a newer version, and remember this version.
    func noteLaunch(hasClusterSites: @autoclosure () -> Bool) {
        // A build-script launch check touches nothing beside the app and
        // writes no preference.
        guard !LaunchCheckMode.isActive else { return }
        let app = Bundle.main.bundleURL.standardizedFileURL
        let current = service.version
        if app.pathExtension == "app" {
            UpdateStaging.removeLeftovers(besideApp: app)
            let last = preferences.lastLaunchedVersion
            let previousName = last.flatMap(SemanticVersion.init)
                .map(UpdateStagingLayout.previousCopyName(for:))
            let previousExists = previousName.map {
                FileManager.default.fileExists(
                    atPath: app.deletingLastPathComponent().appending(component: $0).path)
            } ?? false
            afterInstallNotice = UpdateAfterInstall.notice(
                lastLaunched: last, current: current, appName: app.lastPathComponent,
                previousCopyName: previousExists ? previousName : nil,
                engineIsStale: LocalEngineDecisions.materializedEngineIsFromAnotherBuild(),
                hasClusterSites: hasClusterSites())
        }
        preferences.lastLaunchedVersion = current.releaseDescription
    }

    func dismissAfterInstallNotice() { afterInstallNotice = nil }

    // MARK: Checking

    /// Launch path: at most one check per 24 h, and only while the preference
    /// is on. Called from a `.task`, i.e. after the window is up, and the
    /// whole check runs off the main actor — launch is untouched.
    func runAutomaticCheckIfDue() async {
        // A build-script launch check performs no network activity at all,
        // and the release check would otherwise be its one outbound request.
        // Skipped quietly (not counted against the verdict): this task runs
        // on every launch by design, so skipping it is the correct answer,
        // not evidence of a path that should not exist.
        guard !LaunchCheckMode.isActive else { return }
        guard preferences.shouldRunAutomaticCheck else { return }
        preferences.recordCheck()
        let result = await service.checkDetailed()
        apply(result)
    }

    /// Menu path: always runs, and always reports — including the honest
    /// "could not reach…" an automatic check swallows.
    func checkNow() async {
        guard !isChecking else { return }
        isChecking = true
        defer { isChecking = false }
        preferences.recordCheck()
        let result = await service.checkDetailed()
        apply(result)
        var report = UpdateCheckPolicy.manualReport(for: result.availability)
        if case .releaseAvailable = result.availability, let installRefusal {
            report += " " + installRefusal
        }
        if let tag = UpdateCheckOverrides.releaseTag() {
            report += " (This check asked for the release tagged \(tag), as "
                + "\(UpdateCheckOverrides.releaseTagVariable) requests.)"
        }
        manualReport = report
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

    private func apply(_ result: UpdateCheckResult) {
        let availability = result.availability
        lastAvailability = availability
        bannerMessage = preferences.bannerMessage(for: availability)
        if case .releaseAvailable(_, let page) = availability {
            downloadPage = page
            evaluateInstallation(result.release)
        } else {
            installableRelease = nil
            installAssets = nil
            installTarget = nil
            installRefusal = nil
        }
    }

    /// Decide, once per answer, whether Install Update is offered, and if
    /// not, the one line that says why.
    private func evaluateInstallation(_ release: PublishedRelease?) {
        installableRelease = nil
        installAssets = nil
        installTarget = nil
        guard let release else {
            installRefusal =
                "This release's files are not on GitHub's public release page, so "
                + "SteerLab cannot download them itself. Open the Releases page instead."
            return
        }
        switch UpdateInstallEligibility.evaluate(UpdateInstallContextReader.current()) {
        case .failure(let refusal):
            installRefusal = refusal.line
            return
        case .success(let target):
            switch UpdateReleaseAssets.select(tag: release.tag, assets: release.assets) {
            case .failure(let error):
                installRefusal = error.summary + " Open the Releases page instead."
            case .success(let pair):
                installRefusal = nil
                installableRelease = release
                installAssets = pair
                installTarget = target
            }
        }
    }

    // MARK: Installing

    var currentVersion: SemanticVersion { service.version }

    /// Where the running version will be kept.
    var previousCopy: URL? {
        installTarget.map {
            $0.appBundle.deletingLastPathComponent()
                .appending(component: UpdateStagingLayout.previousCopyName(for: currentVersion))
        }
    }

    var releaseNotesPage: URL? {
        installableRelease?.page ?? downloadPage
    }

    /// Install Update: open the confirmation sheet. Nothing is fetched yet.
    func reviewInstall() {
        guard installableRelease != nil else { return }
        installState = .reviewing
        showingInstallSheet = true
        startWatchingBusy()
    }

    private func startWatchingBusy() {
        busyTask?.cancel()
        busyTask = Task { [weak self] in
            while !Task.isCancelled {
                await self?.refreshBusy()
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    func refreshBusy() async {
        busy = await busyProbe?() ?? UpdateBusyState()
    }

    /// The person said Install: download and verify, then swap when idle.
    func startInstall() {
        guard installTask == nil, busy.isIdle, let release = installableRelease,
            let target = installTarget
        else { return }
        let installer = UpdateInstaller(
            target: target, currentVersion: currentVersion,
            environment: UpdateInstallEnvironment(
                transport: GitHubUpdateTransport(),
                codeSigning: SecurityFrameworkCodeSigning()))
        self.installer = installer
        installState = .working(.downloading(received: 0, total: installAssets?.archive.size))
        let onProgress = progressHandler()
        installTask = Task { [weak self] in
            do {
                let staged = try await installer.prepare(release, progress: onProgress)
                guard let self else { return }
                self.staged = staged
                self.installTask = nil
                await self.swapWhenIdle()
            } catch {
                guard let self else { return }
                self.installTask = nil
                let typed = error as? UpdateInstallError
                    ?? .downloadFailed(error.localizedDescription)
                self.installState = .failed(typed)
            }
        }
    }

    private func progressHandler() -> @Sendable (UpdateInstallProgress) -> Void {
        { [weak self] progress in
            guard let model = self else { return }
            Task { @MainActor in model.showProgress(progress) }
        }
    }

    private func showProgress(_ progress: UpdateInstallProgress) {
        guard case .working = installState else { return }
        installState = .working(progress)
    }

    /// The last step, which quits the app: only when nothing is running.
    func swapWhenIdle() async {
        guard let installer, let staged else { return }
        await refreshBusy()
        guard busy.isIdle else {
            installState = .waitingForIdle
            return
        }
        installState = .installing
        do {
            _ = try installer.install(staged)
        } catch {
            self.staged = nil
            installState = .failed(
                error as? UpdateInstallError ?? .installFailedRestored(error.localizedDescription))
            return
        }
        do {
            try UpdateRelaunch.launchWaiter(for: installer.target.appBundle)
        } catch {
            // Installed; only the automatic reopening could not be arranged.
            NSLog("SteerLab update: could not arrange the reopen: \(error)")
        }
        NSApp.terminate(nil)
    }

    /// Cancel at any point before the swap: nothing is changed, and the
    /// downloaded files are removed.
    func cancelInstall() {
        installTask?.cancel()
        installTask = nil
        installer?.discardStaging()
        staged = nil
        installer = nil
        installState = .reviewing
        closeInstallSheet()
    }

    func closeInstallSheet() {
        busyTask?.cancel()
        busyTask = nil
        if case .failed = installState { installState = .reviewing }
        showingInstallSheet = false
    }
}

/// Unobtrusive one-line banner above the main content. Present only for a
/// newer packaged release; the weaker "newer source" signal never gets here.
struct UpdateBanner: View {
    let model: UpdateSignpostModel

    var body: some View {
        if let notice = model.afterInstallNotice {
            AfterInstallBanner(model: model, notice: notice)
        } else if let message = model.bannerMessage {
            HStack(spacing: 10) {
                Image(systemName: "arrow.down.circle")
                    .foregroundStyle(.tint)
                VStack(alignment: .leading, spacing: 1) {
                    Text(message)
                        .font(.callout)
                    if let refusal = model.installRefusal {
                        Text(refusal)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.tail)
                            .help(refusal)
                    }
                }
                Spacer(minLength: 8)
                if model.installableRelease != nil {
                    Button("Install Update…") { model.reviewInstall() }
                        .buttonStyle(.borderedProminent)
                        .help(
                            "review what will happen first: SteerLab downloads the "
                                + "release, checks it, and asks before replacing anything")
                    Button("Release Notes…") { model.openDownloadPage() }
                        .help("open this release's page in your browser")
                } else {
                    Button("Open Releases…") { model.openDownloadPage() }
                        .help("open the repository's Releases page in your browser")
                }
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
                .accessibilityLabel("Dismiss update notice")
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

extension View {
    /// The Install Update sheet, attached once at the window so it survives
    /// the banner being dismissed (and opens from the manual check's alert).
    func updateInstallSheet(_ model: UpdateSignpostModel) -> some View {
        sheet(
            isPresented: Binding(
                get: { model.showingInstallSheet },
                set: { if !$0 { model.closeInstallSheet() } })
        ) {
            UpdateInstallSheet(model: model)
                // Closing by Escape or a click outside must not abandon a
                // download that is still running.
                .interactiveDismissDisabled(model.installState != .reviewing)
        }
    }
}

/// The first launch of a newer version: what changed, how to go back, and
/// what to bring up to date. Shown once.
private struct AfterInstallBanner: View {
    let model: UpdateSignpostModel
    let notice: UpdateAfterInstallNotice

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "checkmark.circle")
                .foregroundStyle(.tint)
            VStack(alignment: .leading, spacing: 3) {
                Text(notice.headline)
                    .font(.callout)
                ForEach(notice.lines, id: \.self) { line in
                    Text(line)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 8)
            if notice.offersEngineSetup, let open = model.openEngineSetup {
                Button("Open Engine Setup…") { open() }
                    .help("the local Python engine's setup, where Refresh Engine brings it up to date")
            }
            Button {
                model.dismissAfterInstallNotice()
            } label: {
                Image(systemName: "xmark")
            }
            .buttonStyle(.borderless)
            .accessibilityLabel("Dismiss")
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 7)
        .background(.thinMaterial)
        .overlay(alignment: .bottom) { Divider() }
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
