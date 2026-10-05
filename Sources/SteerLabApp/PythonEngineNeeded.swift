import ExperimentKit
import SwiftUI

/// The one notice for a place that needs the Python engine, whatever the app
/// is using: the switch on the quick start (`PythonEngineOffer`), a Connect
/// button when the Python engine is chosen but not connected, and nothing
/// once it is connected. The decision is `PythonEngineNotice`, unit-tested in
/// ExperimentKit; this view only lays it out.
///
/// Without the compute coordinator in the environment it still says what is
/// needed, with no button, rather than going silent.
struct PythonEngineNeeded: View {
    @Environment(ComputeChoiceCoordinator.self) private var coordinator:
        ComputeChoiceCoordinator?
    /// What needs the engine: "Readout traces", "Submitting the batch".
    let subject: String
    var plural = true

    var body: some View {
        if let compute = coordinator {
            switch PythonEngineNotice(
                inUse: compute.inUse, connected: compute.cluster.client != nil)
            {
            case .offerSwitch:
                PythonEngineOffer(subject: subject, plural: plural)
            case .connect:
                VStack(alignment: .leading, spacing: 6) {
                    Label(
                        PythonEngineNotice.notConnected(subject, plural: plural),
                        systemImage: "bolt.horizontal.circle")
                        .font(.callout)
                        .fixedSize(horizontal: false, vertical: true)
                    Button(PythonEngineNotice.connectButton) {
                        Task { await compute.service.connectCluster() }
                    }
                    .controlSize(.small)
                    .disabled(compute.cluster.isConnecting)
                    .help("connect to the Python engine the app is set to use")
                }
                .padding(4)
                .frame(maxWidth: .infinity, alignment: .leading)
            case .none:
                EmptyView()
            }
        } else {
            Label(
                PythonEngineNotice.needsPythonEngineBriefly(subject, plural: plural),
                systemImage: "bolt.horizontal.circle")
                .font(.callout)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// The switch and the guide as two buttons and the cost of switching, for a
/// place that already says what is needed in its own words — an empty-state
/// view, whose description is that sentence. Renders nothing off the quick
/// start.
struct PythonEngineSwitchControls: View {
    @Environment(ComputeChoiceCoordinator.self) private var coordinator:
        ComputeChoiceCoordinator?

    var body: some View {
        if let compute = coordinator,
            PythonEngineNotice(inUse: compute.inUse, connected: true) == .offerSwitch
        {
            VStack(spacing: 6) {
                HStack(spacing: 8) {
                    Button(ComputeGuide.switchButton) {
                        compute.choose(.macFullCapabilities)
                    }
                    .help(
                        "checks what is already set up on this Mac, then "
                            + "either connects to the Python engine or opens "
                            + "its one-time setup")
                    Button(ComputeGuide.guideButton) { compute.showingGuide = true }
                        .help("what each of the three choices can run")
                }
                .controlSize(.small)
                Text(ComputeGuide.switchCaption)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}

/// For a view that is itself a sheet and shows a Python-engine notice: the
/// engine setup and the "what runs where" view the notice can ask for are
/// presented ON this sheet, the way Research Setup presents them.
///
/// Without this, a switch pressed inside a sheet asks the main window to
/// present a second sheet while one is already up. While any such sheet is
/// on screen it is the host (`ComputeChoiceCoordinator.sheetHosts`), the
/// innermost one presents, and the main window's copy stands down.
struct HostsComputeSheets: ViewModifier {
    @Environment(ComputeChoiceCoordinator.self) private var coordinator:
        ComputeChoiceCoordinator?
    @Environment(LocalServerController.self) private var localServer:
        LocalServerController?
    @State private var hostID = UUID()

    func body(content: Content) -> some View {
        if let compute = coordinator, let localServer {
            content
                .modifier(
                    ComputeSheets(
                        compute: compute, service: compute.service,
                        localServer: localServer,
                        isActive: compute.sheetHosts.last == hostID))
                .onAppear { compute.hostSheets(hostID) }
                .onDisappear { compute.releaseSheets(hostID) }
        } else {
            content
        }
    }
}

extension View {
    /// See `HostsComputeSheets`.
    func hostsComputeSheets() -> some View { modifier(HostsComputeSheets()) }
}
