import AppKit
import ExperimentKit

/// The three compute choices, as an `NSSavePanel` accessory.
///
/// Where a workspace's studies run is asked at creation because that is the
/// one moment the answer is unambiguous — the researcher is deciding what the
/// folder is FOR. Leaving it to be inferred later is what produced the
/// disagreement the binding replaced: each verb guessed separately from the
/// live server pairing, and a workspace ended up treating its own artifacts
/// as foreign.
///
/// It used to offer two engine names, "Cluster (Python/PyTorch)" and "Local
/// (MLX)", defaulted to the first, and captioned the second "toy models and
/// pipeline checks" — which told a researcher with only a laptop that the
/// real path was not for them. It now offers `ComputeChoice`'s three plainly
/// named choices, each with its own sentence, and starts on the one that
/// needs nothing installed beyond a model.
///
/// AppKit rather than a SwiftUI sheet because the choice belongs *in* the
/// same dialog as the folder name; a second modal after the save panel is a
/// step the researcher can dismiss, leaving exactly the undeclared state this
/// is meant to prevent.
///
/// Main-actor isolation records a fact rather than imposing a rule: the views
/// are built while the panel is being assembled on the main thread, and AppKit
/// delivers a control's action on the main thread as well, so both the
/// accessory and the `Relay` that receives the action already live there.
@MainActor
final class ComputeChoiceAccessory {
    private(set) var selected: ComputeChoice
    let view: NSView
    private let buttons: [NSButton]

    init(selected: ComputeChoice = .newWorkspaceDefault) {
        self.selected = selected

        let heading = NSTextField(
            labelWithString: "Where studies in this workspace run:")
        heading.font = .preferredFont(forTextStyle: .headline)

        let relay = Relay()
        var buttons: [NSButton] = []
        var rows: [NSView] = [heading]
        for (index, choice) in ComputeChoice.allCases.enumerated() {
            let button = NSButton(
                radioButtonWithTitle: choice.title, target: relay,
                action: #selector(Relay.changed(_:)))
            button.tag = index
            button.state = choice == selected ? .on : .off
            // The sentence is also the tooltip and the accessibility help, so
            // VoiceOver reads what the choice means, not just its name.
            button.toolTip = choice.summary
            button.setAccessibilityHelp(choice.summary)
            buttons.append(button)

            let caption = NSTextField(wrappingLabelWithString: choice.summary)
            caption.font = .preferredFont(forTextStyle: .caption1)
            caption.textColor = .secondaryLabelColor
            caption.widthAnchor.constraint(lessThanOrEqualToConstant: 420).isActive = true

            // Indent the sentence under its radio button's title.
            let indent = NSView()
            indent.widthAnchor.constraint(equalToConstant: 16).isActive = true
            let captionRow = NSStackView(views: [indent, caption])
            captionRow.orientation = .horizontal
            captionRow.alignment = .top
            captionRow.spacing = 4

            let row = NSStackView(views: [button, captionRow])
            row.orientation = .vertical
            row.alignment = .leading
            row.spacing = 2
            rows.append(row)
        }

        let footer = NSTextField(
            wrappingLabelWithString:
                "You can change this later from the Workspace menu. Nothing "
                + "is installed when the workspace is created; "
                + "\(ComputeChoice.macFullCapabilities.title) opens its "
                + "one-time setup afterwards, and you approve it there.")
        footer.font = .preferredFont(forTextStyle: .caption1)
        footer.textColor = .secondaryLabelColor
        footer.widthAnchor.constraint(lessThanOrEqualToConstant: 440).isActive = true
        rows.append(footer)

        let stack = NSStackView(views: rows)
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 8
        stack.edgeInsets = NSEdgeInsets(top: 10, left: 16, bottom: 10, right: 16)
        stack.translatesAutoresizingMaskIntoConstraints = false

        let container = NSView()
        container.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.topAnchor.constraint(equalTo: container.topAnchor),
            stack.bottomAnchor.constraint(equalTo: container.bottomAnchor),
            stack.leadingAnchor.constraint(equalTo: container.leadingAnchor),
            stack.trailingAnchor.constraint(equalTo: container.trailingAnchor),
        ])
        self.view = container
        self.buttons = buttons

        // Each radio button sits in its own row, so AppKit's automatic
        // grouping (same superview, same action) does not apply: the relay
        // keeps exactly one of them on.
        relay.onChange = { [weak self] index in
            guard let self, ComputeChoice.allCases.indices.contains(index)
            else { return }
            self.selected = ComputeChoice.allCases[index]
            for button in self.buttons {
                button.state = button.tag == index ? .on : .off
            }
        }
        // Retained by the accessory's own view for the panel's lifetime,
        // which is the accessory's lifetime.
        objc_setAssociatedObject(
            container, Unmanaged.passUnretained(self).toOpaque(), relay,
            .OBJC_ASSOCIATION_RETAIN)
    }

    /// The radio buttons' target. `changed(_:)` is reached only through the
    /// action chain, which AppKit runs on the main thread, so the callback
    /// can read the accessory's state without a hop.
    @MainActor
    private final class Relay: NSObject {
        var onChange: ((Int) -> Void)?
        @objc func changed(_ sender: NSButton) {
            onChange?(sender.tag)
        }
    }
}
