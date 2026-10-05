import ExperimentKit
import SwiftUI

// The three compute choices, as the researcher sees them. Every sentence
// comes from `ComputeChoice` / `ComputeGuide` (ExperimentKit), where the
// wording and the mapping to the persisted binding are unit-tested; these
// views only lay it out.

/// The three choices as a radio list: a title and a plain summary each.
/// Used where there is room to read — Research Setup and the guide.
struct ComputeChoiceList: View {
    let selection: ComputeChoice?
    let choose: (ComputeChoice) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(ComputeChoice.allCases) { choice in
                Button {
                    choose(choice)
                } label: {
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Image(
                            systemName: choice == selection
                                ? "largecircle.fill.circle" : "circle")
                            .foregroundStyle(
                                choice == selection
                                    ? AnyShapeStyle(.tint) : AnyShapeStyle(.secondary))
                        VStack(alignment: .leading, spacing: 2) {
                            Text(choice.title).font(.callout.weight(.medium))
                            Text(choice.summary)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("\(choice.title). \(choice.summary)")
                .accessibilityAddTraits(choice == selection ? .isSelected : [])
                .help(choice.engineNote)
            }
        }
    }
}

/// "What runs where": one compact view, reachable from the Compute menu and
/// from every place that offers a switch.
///
/// macOS 27 layout rule: a plain sheet with a FIXED frame, and the one
/// variable-length region (the table and the limits) inside a `ScrollView`.
struct ComputeGuideSheet: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(ComputeGuide.title).font(.title2.bold())
            Text(ComputeGuide.introduction)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    choices
                    table
                    limits
                }
                .padding(.vertical, 2)
            }
            HStack {
                Spacer()
                Button("Done") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 640, height: 620)
    }

    private var choices: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(ComputeChoice.allCases) { choice in
                VStack(alignment: .leading, spacing: 2) {
                    Text(choice.title).font(.callout.weight(.semibold))
                    Text(choice.summary)
                        .font(.callout)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(choice.engineNote)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    /// One column per choice. The rows and every mark come from the shipped
    /// science catalog (`ComputeGuide.rows`), the same declarations behind
    /// `science list` on both command lines; this view only lays them out.
    private var table: some View {
        GroupBox("What each one can do") {
            VStack(alignment: .leading, spacing: 8) {
                Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 6) {
                    GridRow {
                        Text("").gridColumnAlignment(.leading)
                        ForEach(ComputeChoice.allCases) { choice in
                            Text(ComputeGuide.columnTitle(choice))
                                .font(.caption.weight(.semibold))
                        }
                    }
                    ForEach(ComputeGuide.rows) { row in
                        GridRow {
                            Text(row.activity)
                                .font(.callout)
                                .fixedSize(horizontal: false, vertical: true)
                            ForEach(ComputeChoice.allCases) { choice in
                                mark(row.runs(on: choice))
                            }
                        }
                    }
                }
                Text(ComputeGuide.tableNote)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    /// A word beside the symbol, so the answer never rests on colour alone.
    private func mark(_ runs: Bool) -> some View {
        Label(runs ? "Yes" : "No", systemImage: runs ? "checkmark" : "minus")
            .font(.caption)
            .foregroundStyle(runs ? AnyShapeStyle(.primary) : AnyShapeStyle(.secondary))
    }

    private var limits: some View {
        GroupBox("Before you switch") {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(ComputeGuide.limits, id: \.self) { limit in
                    Text(limit)
                        .font(.callout)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(6)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

/// Shown where a method needs the Python engine and the app is on the quick
/// start: one sentence and a button that offers the switch, in place of a
/// refusal. Renders nothing when the Python engine is already in use.
struct PythonEngineOffer: View {
    /// From the environment (set on the main window's content). Optional so
    /// a view shown somewhere the environment does not reach simply shows no
    /// offer rather than a button that does nothing.
    @Environment(ComputeChoiceCoordinator.self) private var coordinator:
        ComputeChoiceCoordinator?
    /// What needs the engine: "Probes", "Importing an SAE feature".
    let subject: String
    var plural = true

    var body: some View {
        if let compute = coordinator, compute.inUse == .macQuickStart {
            // A plain stack, not a box: this sits inside grouped forms and
            // inside other boxes, and a box in a box reads as an error.
            VStack(alignment: .leading, spacing: 0) {
                VStack(alignment: .leading, spacing: 6) {
                    Label(
                        ComputeGuide.needsPythonEngine(subject, plural: plural),
                        systemImage: "bolt.horizontal.circle")
                        .font(.callout)
                        .fixedSize(horizontal: false, vertical: true)
                    HStack(spacing: 8) {
                        // `choose`, not `use`: the researcher is asking to
                        // move this workspace, and the caption below says
                        // what that costs before they press it.
                        Button(ComputeGuide.switchButton) {
                            compute.choose(.macFullCapabilities)
                        }
                        .help(
                            "checks what is already set up on this Mac, then "
                                + "either connects to the Python engine or "
                                + "opens its one-time setup")
                        Button(ComputeGuide.guideButton) { compute.showingGuide = true }
                            .help("what each of the three choices can run")
                    }
                    .controlSize(.small)
                    Text(ComputeGuide.switchCaption)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .padding(4)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}

/// Presents the two sheets every compute-choice entry point can ask for: the
/// local-engine setup and the "what runs where" view.
///
/// Attached twice — to the main window's content, and inside Research Setup
/// (which is itself a sheet, so a request made from it has to be presented
/// ON it). `isActive` decides which of the two presents, so exactly one
/// does.
struct ComputeSheets: ViewModifier {
    @Bindable var compute: ComputeChoiceCoordinator
    let service: ChatService
    let localServer: LocalServerController
    let isActive: Bool

    func body(content: Content) -> some View {
        content
            .sheet(isPresented: engineSetupPresented) {
                LocalEngineSetupSheet(
                    engine: compute.localEngine, service: service, server: localServer)
            }
            .sheet(isPresented: guidePresented) {
                ComputeGuideSheet()
            }
    }

    private var engineSetupPresented: Binding<Bool> {
        Binding(
            get: { isActive && compute.showingEngineSetup },
            set: { compute.showingEngineSetup = $0 })
    }

    private var guidePresented: Binding<Bool> {
        Binding(
            get: { isActive && compute.showingGuide },
            set: { compute.showingGuide = $0 })
    }
}
