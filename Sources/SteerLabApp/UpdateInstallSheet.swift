import ExperimentKit
import SwiftUI

/// Install Update's confirmation, progress, and outcome. Says what will
/// happen before anything is fetched: the new version, the download size,
/// where the running version is kept, that the app quits and reopens, and
/// what to bring up to date afterwards. Install stays disabled while anything
/// is running, and the last step (the one that quits) waits for idle again.
struct UpdateInstallSheet: View {
    let model: UpdateSignpostModel

    private var version: String {
        model.installableRelease?.version.releaseDescription ?? "the new version"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Install SteerLab \(version)?")
                .font(.title3.weight(.semibold))
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    switch model.installState {
                    case .reviewing: review
                    case .working(let progress): working(progress)
                    case .waitingForIdle: waiting
                    case .installing: installing
                    case .failed(let error): failed(error)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(height: 330)
            Divider()
            controls
        }
        .padding(20)
        .frame(width: 560)
        .task { await model.refreshBusy() }
    }

    // MARK: The confirmation

    @ViewBuilder private var review: some View {
        let size = model.installAssets?.archive.size
            .map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) }
        fact(
            "arrow.down.circle",
            "SteerLab downloads \(size.map { "\($0) " } ?? "")from the project's GitHub "
                + "Releases page and checks it: the file against its checksum, and the "
                + "app inside against Apple's notarization and this copy's own developer "
                + "signature. Nothing about this Mac is sent.")
        if let notes = model.releaseNotesPage {
            Link("Read the release notes", destination: notes)
                .font(.callout)
        }
        fact(
            "arrow.triangle.2.circlepath",
            "SteerLab then quits and reopens as \(version). Your workspaces, studies, "
                + "and models are not touched.")
        if let previous = model.previousCopy {
            fact(
                "archivebox",
                "The version you are running now (\(model.currentVersion.releaseDescription)) "
                    + "is kept beside it as \(previous.lastPathComponent), in "
                    + "\(previous.deletingLastPathComponent().path). To go back, quit "
                    + "SteerLab, move the new SteerLab.app to the Trash, and rename the "
                    + "kept copy to SteerLab.app.")
        }
        fact(
            "server.rack",
            "If you use the Python engine on this Mac, open its setup after the update "
                + "and choose Refresh Engine; its installed packages are kept.")
        fact("network", UpdateAfterInstall.clusterLine)
        busyList(heading: "Install becomes available when these finish:")
    }

    // MARK: Progress and outcomes

    @ViewBuilder private func working(_ progress: UpdateInstallProgress) -> some View {
        switch progress {
        case .downloading(let received, let total):
            if let total, total > 0 {
                ProgressView(value: Double(received), total: Double(total)) {
                    Text("Downloading \(bytes(received)) of \(bytes(total))")
                }
            } else {
                ProgressView { Text("Downloading \(bytes(received))") }
            }
        case .verifyingChecksum:
            ProgressView { Text("Checking the download against its checksum") }
        case .unpacking:
            ProgressView { Text("Unpacking") }
        case .checkingSignature:
            ProgressView { Text("Checking the new app's signature and notarization") }
        }
        Text("Nothing has been replaced yet. Cancel stops here and removes the download.")
            .font(.caption)
            .foregroundStyle(.secondary)
    }

    @ViewBuilder private var waiting: some View {
        fact(
            "checkmark.seal",
            "SteerLab \(version) is downloaded and checked. Installing it quits SteerLab, "
                + "so it waits until nothing is running.")
        busyList(heading: "Still running:")
        if model.busy.isIdle {
            Text("Nothing is running now. Choose Install Now to finish.")
                .font(.callout)
        }
    }

    private var installing: some View {
        ProgressView { Text("Installing, then reopening SteerLab") }
    }

    @ViewBuilder private func failed(_ error: UpdateInstallError) -> some View {
        Label(error.summary, systemImage: "exclamationmark.triangle")
            .foregroundStyle(.primary)
            .fixedSize(horizontal: false, vertical: true)
        Text(error.repair)
            .font(.callout)
            .fixedSize(horizontal: false, vertical: true)
    }

    // MARK: Pieces

    @ViewBuilder private func busyList(heading: String) -> some View {
        if !model.busy.isIdle {
            VStack(alignment: .leading, spacing: 4) {
                Text(heading)
                    .font(.callout.weight(.semibold))
                ForEach(model.busy.reasons, id: \.self) { reason in
                    Label(reason.prefix(1).uppercased() + reason.dropFirst(), systemImage: "hourglass")
                        .font(.callout)
                }
            }
            .padding(10)
            .background(.quaternary, in: RoundedRectangle(cornerRadius: 6))
        }
    }

    private func fact(_ symbol: String, _ text: String) -> some View {
        Label {
            Text(text).fixedSize(horizontal: false, vertical: true)
        } icon: {
            Image(systemName: symbol).foregroundStyle(.secondary)
        }
        .font(.callout)
    }

    private func bytes(_ count: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: count, countStyle: .file)
    }

    @ViewBuilder private var controls: some View {
        HStack {
            switch model.installState {
            case .reviewing:
                Spacer()
                Button("Cancel", role: .cancel) { model.closeInstallSheet() }
                    .keyboardShortcut(.cancelAction)
                Button("Install") { model.startInstall() }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!model.busy.isIdle)
                    .help(
                        model.busy.isIdle
                            ? "download, check, and then install \(version)"
                            : "available once nothing is running")
            case .working:
                Spacer()
                Button("Cancel", role: .cancel) { model.cancelInstall() }
                    .keyboardShortcut(.cancelAction)
            case .waitingForIdle:
                Spacer()
                Button("Cancel", role: .cancel) { model.cancelInstall() }
                    .keyboardShortcut(.cancelAction)
                Button("Install Now") { Task { await model.swapWhenIdle() } }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!model.busy.isIdle)
            case .installing:
                Spacer()
            case .failed:
                Button("Open Releases…") { model.openDownloadPage() }
                Spacer()
                Button("Close") { model.closeInstallSheet() }
                    .keyboardShortcut(.defaultAction)
            }
        }
    }
}
