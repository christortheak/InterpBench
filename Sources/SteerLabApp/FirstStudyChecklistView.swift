import ExperimentKit
import SwiftUI

/// The first-study checklist's rows: seven steps, each marked done or not
/// done, each with the button that goes there.
///
/// Whether a step is done, what it says, and which buttons it offers are all
/// `FirstStudyChecklist`'s (ExperimentKit, unit-tested against real workspace
/// folders); this view only lays them out and hands a pressed button to
/// `perform`. Home shows it inside its form, and the no-workspace Home shows
/// it with only the ways into a workspace available.
struct FirstStudyChecklistRows: View {
    let items: [FirstStudyChecklist.Item]
    /// False before any workspace exists: then only the buttons that make
    /// one are offered.
    let hasWorkspace: Bool
    /// True while the workspace is fixed from outside the app, when the
    /// buttons that change it cannot take effect.
    var workspacePinned = false
    /// False for a preview of the steps with no buttons (the no-workspace
    /// Home, whose ways in sit above the preview).
    var showsButtons = true
    let perform: (FirstStudyChecklist.Action) -> Void
    /// Declares one of the three places studies run. Nil offers "Choose…" as
    /// a plain button that `perform` handles instead.
    var chooseCompute: ((ComputeChoice) -> Void)?

    var body: some View {
        let next = FirstStudyChecklist.nextStep(items)
        ForEach(Array(items.enumerated()), id: \.element.id) { index, item in
            row(item, number: index + 1, isNext: item.step == next)
        }
    }

    private func row(_ item: FirstStudyChecklist.Item, number: Int, isNext: Bool) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: item.isDone ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(item.isDone ? Color.green : (isNext ? Color.accentColor : .secondary))
                .accessibilityLabel(item.isDone ? "Done" : "Not done")
            VStack(alignment: .leading, spacing: 4) {
                Text("\(number). \(item.title)")
                    .font(.callout.weight(isNext ? .semibold : .regular))
                Text(item.detail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                let actions = item.actions.filter { hasWorkspace || !$0.needsWorkspace }
                if showsButtons, !actions.isEmpty {
                    // Two buttons at Home's 420 pt floor: side by side where
                    // they fit, stacked where they do not, never clipped.
                    ViewThatFits(in: .horizontal) {
                        HStack(spacing: 8) { buttons(actions, isNext: isNext) }
                        VStack(alignment: .leading, spacing: 6) {
                            buttons(actions, isNext: isNext)
                        }
                    }
                    .controlSize(.small)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.vertical, 2)
        .accessibilityElement(children: .contain)
    }

    @ViewBuilder
    private func buttons(_ actions: [FirstStudyChecklist.Action], isNext: Bool) -> some View {
        ForEach(Array(actions.enumerated()), id: \.element) { index, action in
            let prominent = isNext && index == 0
            if action == .chooseCompute, let chooseCompute {
                Menu(action.title) {
                    ForEach(ComputeChoice.allCases) { choice in
                        Button(choice.title) { chooseCompute(choice) }
                    }
                }
                .fixedSize()
                .help("set where this workspace's studies run: \(choiceList)")
            } else if prominent {
                button(action).buttonStyle(.borderedProminent)
            } else {
                button(action)
            }
        }
    }

    private func button(_ action: FirstStudyChecklist.Action) -> some View {
        let changesWorkspace = !action.needsWorkspace
        return Button(action.title) { perform(action) }
            .disabled(changesWorkspace && workspacePinned)
            .help(
                changesWorkspace && workspacePinned
                    ? DemoWorkspaceCopy.unavailableWhilePinned : help(action))
    }

    private var choiceList: String {
        ComputeChoice.allCases.map(\.title).joined(separator: "; ")
    }

    private func help(_ action: FirstStudyChecklist.Action) -> String {
        switch action {
        case .newWorkspace: "choose a name and a place for a new workspace folder"
        case .openWorkspace: "choose a workspace folder you already have"
        case .chooseCompute: "Research Setup: choose where this workspace's studies run"
        case .openPlayground: "the Playground's Model section, where a model is chosen and downloaded"
        case .openDemoWorkspace: "open a copy of a finished study, in a folder you choose"
        case .openStudies(let name?): "open '\(name)' in Studies"
        case .openStudies(nil): "the Studies section, where a study is drafted, frozen, and run"
        case .openResults: "the Results section, where a run is read and analyzed"
        }
    }
}
