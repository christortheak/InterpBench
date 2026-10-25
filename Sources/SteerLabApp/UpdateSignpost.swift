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
/// their confirmation in the sheet. A copy that may not install itself (one
/// built locally, so ad-hoc or development signed; a translocated or
/// read-only copy; …) keeps the signpost to the Releases page and says why
/// in one line.
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
    /// Supplied by the app: opens the local engine setup (Refresh Engine,
    /// Reinstall Packages).
    @ObservationIgnored var openEngineSetup: (@MainActor () -> Void)?

    private var lastAvailability: UpdateAvailability?
    private let preferences: UpdateCheckPreferences
    private let service: UpdateCheckService
    @ObservationIgnored private var installTask: Task<Void, Never>?
    @ObservationIgnored private var busyTask: Task<Void, Never>?
    @ObservationIgnored private var installer: UpdateInstaller?
    @ObservationIgnored private var staged: UpdateStagedApp?
    @ObservationIgnored private var launchNoted = false
    /// Bumped by every Install and every Cancel: work that finishes for an
    /// attempt that is no longer current does nothing.
    @ObservationIgnored private var installGeneration = 0
    /// Set from the first line of `swapWhenIdle` until it returns. A second
    /// click on Install Now during the busy check, or a new Install Update
    /// during the swap, must not start a second, interleaved install (an
    /// independent review, 2026-10-17, traced two swaps undoing each other).
    @ObservationIgnored private var swapping = false

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
        // Once per process, however many windows open; and a build-script
        // launch check touches nothing beside the app and writes no
        // preference.
        guard !launchNoted, !LaunchCheckMode.isActive else { return }
        launchNoted = true
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
                packagesNeedReinstall: LocalEngineDecisions.materializedEnvironmentLockMoved(),
                hasClusterSites: hasClusterSites(),
                sourceCheckout: UpdateInstallContextReader.sourceCheckout())
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
        guard installableRelease != nil, !swapping else { return }
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
        guard installTask == nil, !swapping, busy.isIdle, let release = installableRelease,
            let target = installTarget
        else { return }
        let installer = UpdateInstaller(
            target: target, currentVersion: currentVersion,
            environment: UpdateInstallEnvironment(
                transport: GitHubUpdateTransport(),
                codeSigning: SecurityFrameworkCodeSigning()))
        self.installer = installer
        installGeneration += 1
        let generation = installGeneration
        installState = .working(.downloading(received: 0, total: installAssets?.archive.size))
        let onProgress = progressHandler()
        installTask = Task { [weak self] in
            let outcome: Result<UpdateStagedApp, UpdateInstallError>
            do {
                outcome = .success(try await installer.prepare(release, progress: onProgress))
            } catch {
                outcome = .failure(
                    error as? UpdateInstallError ?? .downloadFailed(error.localizedDescription))
            }
            await self?.prepared(
                outcome, by: installer, cancelled: Task.isCancelled, generation: generation)
        }
    }

    /// Where a download-and-verify ends. A Cancel that arrived meanwhile
    /// wins: the sheet has already closed, so nothing is installed and what
    /// was downloaded is removed (`prepare` removes it on its own failures).
    private func prepared(
        _ outcome: Result<UpdateStagedApp, UpdateInstallError>, by installer: UpdateInstaller,
        cancelled: Bool, generation: Int
    ) async {
        installTask = nil
        if cancelled || generation != installGeneration {
            if case .success = outcome { installer.discardStaging() }
            return
        }
        switch outcome {
        case .failure(let error):
            installState = .failed(error)
        case .success(let staged):
            self.staged = staged
            await swapWhenIdle()
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
        guard !swapping, let installer, let staged else { return }
        swapping = true
        defer { swapping = false }
        let generation = installGeneration
        await refreshBusy()
        // A Cancel (or a new attempt) while the busy check was out wins.
        guard generation == installGeneration, self.staged == staged,
            installState != .reviewing
        else { return }
        guard busy.isIdle else {
            installState = .waitingForIdle
            return
        }
        installState = .installing
        // Verification and the swap run off the main actor; nothing can
        // cancel them from here (the sheet offers no Cancel while
        // installing), and a successful swap always ends in quitting, since
        // the app on disk is already the new version.
        if case .failure(let error) = await installer.installDetached(staged) {
            self.staged = nil
            installState = .failed(error)
            return
        }
        do {
            try UpdateRelaunch.launchWaiter(for: installer.target.appBundle)
        } catch {
            // Installed; only the automatic reopening could not be arranged.
            NSLog("SteerLab update: could not arrange the reopen: \(error)")
        }
        quitForReopen()
    }

    /// Close every sheet, then quit on a later turn of the run loop. AppKit
    /// will not quit while a window has a sheet attached: an end-to-end test
    /// of a real update (2026-10-17) installed the new version and then sat
    /// on "Installing, then reopening SteerLab" with Quit refused, because
    /// `terminate` ran while this sheet was still open.
    private func quitForReopen() {
        showingInstallSheet = false
        for window in NSApp.windows {
            if let sheet = window.attachedSheet { window.endSheet(sheet) }
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            NSApp.terminate(nil)
        }
    }

    /// Cancel at any point before the swap: nothing is changed, and the
    /// downloaded files are removed.
    func cancelInstall() {
        installGeneration += 1
        if let installTask {
            // It removes its own download as it unwinds; a new Install waits
            // until it has (`startInstall` requires no task in flight).
            installTask.cancel()
        } else if staged != nil {
            installer?.discardStaging()
        }
        staged = nil
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
                    .help(
                        "the local Python engine's setup, where Refresh Engine and "
                            + "Reinstall Packages bring it up to date")
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
