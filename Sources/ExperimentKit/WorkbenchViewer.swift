import Foundation

/// Viewer-state hygiene for the workbench's right-hand DISPLAY PANE.
///
/// The pane follows the selected sidebar section unless the researcher pins
/// it. Reported live: "display panel data from one tab persisting into the
/// display panel of another tab — it feels half-baked". Two different things
/// wear that appearance and only one of them is a defect:
///
/// - The **Activity** feed is one workspace-wide log that several sections
///   share. It is SUPPOSED to survive a section switch; what was missing is
///   any statement that it is workspace-wide rather than that section's own.
/// - A **section-specific** viewer (the selected run, the computed geometry
///   tables, a robustness or multi-agent run, the chat transcript) renders
///   the state of exactly one section. Showing it while a different section
///   is selected is legitimate only under an explicit pin, and only when the
///   pin says whose content it is.
///
/// The decision logic lives here, in the engine, rather than in the SwiftUI
/// layer: the app is an executable target and cannot be imported by the test
/// target, and CLAUDE.md's rule is that views render engine state. The view
/// supplies identities and renders the answer.

/// Whose data a viewer renders — the distinction the hygiene rules turn on.
public enum WorkbenchViewerOwnership: Sendable, Equatable {
    /// The state of exactly ONE section (its selected run, its computed
    /// tables, its transcript). Never legitimate under another section's
    /// header without a pin label.
    case section
    /// A workspace-wide feed that reads identically from every section it
    /// appears in (the Activity log). Persisting across sections is correct,
    /// and claiming it "came from" one section would be a false origin.
    case workspaceWide
}

/// A viewer's identity for hygiene purposes: what it is called, which
/// section owns it, and whether its content is that section's alone.
public struct WorkbenchViewerIdentity: Sendable, Equatable {
    /// The viewer's own title ("Selected Run", "Vector Geometry", …).
    public let title: String
    /// User-facing name of the section this viewer belongs to.
    public let sectionLabel: String
    public let ownership: WorkbenchViewerOwnership

    public init(
        title: String, sectionLabel: String, ownership: WorkbenchViewerOwnership
    ) {
        self.title = title
        self.sectionLabel = sectionLabel
        self.ownership = ownership
    }

    public var isSectionSpecific: Bool { ownership == .section }
}

/// The pin: the ONLY thing that survives a section switch.
///
/// Generic over the app's viewer-mode enum so one value is the single source
/// of truth for both what to RENDER and how to LABEL it — the previous shape
/// stored the mode alone, which is why a pinned viewer could not say which
/// section its content belonged to.
public struct WorkbenchViewerPin<Mode: Equatable & Sendable>: Sendable, Equatable {

    /// A pinned viewer: the mode to render plus the identity it had at the
    /// moment it was pinned (its origin section is a fact about the pin, not
    /// about wherever the researcher navigated afterwards).
    public struct Entry: Sendable, Equatable {
        public let mode: Mode
        public let identity: WorkbenchViewerIdentity

        public init(mode: Mode, identity: WorkbenchViewerIdentity) {
            self.mode = mode
            self.identity = identity
        }
    }

    public private(set) var entry: Entry?

    public init() {}

    public var isPinned: Bool { entry != nil }

    /// Which viewer the pane renders. With no pin the answer is always the
    /// selected section's own viewer — that IS the reset-on-switch rule: the
    /// pane holds no mode of its own to go stale.
    public func resolvedMode(section mode: Mode) -> Mode {
        entry?.mode ?? mode
    }

    /// The identity of whatever `resolvedMode` renders.
    public func resolvedIdentity(
        section identity: WorkbenchViewerIdentity
    ) -> WorkbenchViewerIdentity {
        entry?.identity ?? identity
    }

    /// Pin the viewer currently on screen, or release an existing pin.
    /// `mode`/`identity` are ignored when releasing.
    public mutating func toggle(mode: Mode, identity: WorkbenchViewerIdentity) {
        entry = entry == nil ? Entry(mode: mode, identity: identity) : nil
    }

    public mutating func unpin() {
        entry = nil
    }

    /// The badge beside the viewer title, or nil when nothing is pinned.
    ///
    /// A pinned viewer showing one section's own data while a DIFFERENT
    /// section is selected names its origin ("pinned · Results"); anything
    /// else — same section, or the workspace-wide Activity feed, whose
    /// content does not belong to any one section — reads just "pinned",
    /// because naming an origin there would assert something untrue.
    public func badge(section identity: WorkbenchViewerIdentity) -> String? {
        guard let entry else { return nil }
        guard showsForeignSectionData(section: identity) else { return "pinned" }
        return "pinned · \(entry.identity.sectionLabel)"
    }

    /// True when a section's own data is rendering under another section's
    /// selection. This is exactly the state the label exists for; if it is
    /// ever true with no badge, the hygiene rule is broken.
    public func showsForeignSectionData(
        section identity: WorkbenchViewerIdentity
    ) -> Bool {
        guard let entry else { return false }
        return entry.identity.isSectionSpecific
            && entry.identity.sectionLabel != identity.sectionLabel
    }

    /// Help text for the pin control, phrased for the state it is in.
    public func pinControlHelp(section identity: WorkbenchViewerIdentity) -> String {
        guard let entry else {
            return "pin this viewer so it stays while you navigate to other "
                + "sections; everything else follows the selected section"
        }
        if showsForeignSectionData(section: identity) {
            return "unpin — this pane is showing \(entry.identity.sectionLabel)'s "
                + "\(entry.identity.title.lowercased()), not \(identity.sectionLabel)'s; "
                + "unpinning returns it to the selected section"
        }
        return "unpin — the viewer follows the selected section again"
    }

    /// One-line explanation of what the pane is showing, for the title's
    /// tooltip. Says out loud that Activity is workspace-wide (it is shared
    /// by several sections, and looked like leakage precisely because
    /// nothing said so).
    public func scopeDescription(section identity: WorkbenchViewerIdentity) -> String {
        let shown = resolvedIdentity(section: identity)
        switch shown.ownership {
        case .workspaceWide:
            return "workspace-wide — jobs, builds, training, and runs from "
                + "every section stream into this one feed"
        case .section:
            if showsForeignSectionData(section: identity) {
                return "pinned content from \(shown.sectionLabel) — unpin to "
                    + "follow the selected section"
            }
            return "\(shown.sectionLabel) — this section's own \(shown.title.lowercased())"
        }
    }
}
