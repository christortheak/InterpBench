import AppKit
import ExperimentKit

/// The Cluster-vs-Local choice, as an `NSSavePanel` accessory.
///
/// A workspace's compute engine is asked at creation because that is the one
/// moment the answer is unambiguous — the researcher is deciding what the
/// folder is FOR. Leaving it to be inferred later is what produced the
/// disagreement this binding replaces: each verb guessed separately from the
/// live server pairing, and a cluster workspace ended up treating its own
/// artifacts as foreign.
///
/// AppKit rather than a SwiftUI sheet because the choice belongs *in* the
/// same dialog as the folder name; a second modal after the save panel is a
/// step the researcher can dismiss, leaving exactly the undeclared state this
/// is meant to prevent.
final class ComputeChoiceAccessory {
    private(set) var selected: WorkspaceCompute
    let view: NSView

    init(selected: WorkspaceCompute) {
        self.selected = selected

        let label = NSTextField(labelWithString: "Computes on:")
        let control = NSSegmentedControl(
            labels: WorkspaceCompute.allCases.map(\.label),
            trackingMode: .selectOne, target: nil, action: nil)
        control.selectedSegment =
            WorkspaceCompute.allCases.firstIndex(of: selected) ?? 0

        let caption = NSTextField(
            wrappingLabelWithString:
                "Cluster: studies run on the Python/PyTorch engine and this "
                + "Mac manages the data. Local: this Mac's MLX engine runs "
                + "everything — toy models and pipeline checks. Changeable "
                + "later from the Workspace menu.")
        caption.font = .preferredFont(forTextStyle: .caption1)
        caption.textColor = .secondaryLabelColor

        let row = NSStackView(views: [label, control])
        row.orientation = .horizontal
        row.spacing = 8

        let stack = NSStackView(views: [row, caption])
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 6
        stack.edgeInsets = NSEdgeInsets(top: 10, left: 16, bottom: 10, right: 16)
        stack.translatesAutoresizingMaskIntoConstraints = false

        let container = NSView()
        container.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.topAnchor.constraint(equalTo: container.topAnchor),
            stack.bottomAnchor.constraint(equalTo: container.bottomAnchor),
            stack.leadingAnchor.constraint(equalTo: container.leadingAnchor),
            stack.trailingAnchor.constraint(equalTo: container.trailingAnchor),
            caption.widthAnchor.constraint(lessThanOrEqualToConstant: 420),
        ])
        self.view = container

        // Retained by the control's target chain for the panel's lifetime,
        // which is the accessory's lifetime.
        let relay = Relay { [weak self] index in
            guard let self, WorkspaceCompute.allCases.indices.contains(index)
            else { return }
            self.selected = WorkspaceCompute.allCases[index]
        }
        control.target = relay
        control.action = #selector(Relay.changed(_:))
        objc_setAssociatedObject(
            control, Unmanaged.passUnretained(self).toOpaque(), relay,
            .OBJC_ASSOCIATION_RETAIN)
    }

    private final class Relay: NSObject {
        private let onChange: (Int) -> Void
        init(_ onChange: @escaping (Int) -> Void) { self.onChange = onChange }
        @objc func changed(_ sender: NSSegmentedControl) {
            onChange(sender.selectedSegment)
        }
    }
}
