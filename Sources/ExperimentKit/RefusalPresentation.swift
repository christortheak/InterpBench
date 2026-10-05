import Foundation
import SteeringKit

/// How the app shows anything that was refused or that failed: the plain
/// reason, then what to do — as an app action where one exists — with the
/// command-line repair kept behind a disclosure for people who use one.
///
/// Before this, panels interpolated `\(error)` into status lines. For a typed
/// refusal that showed the reason and dropped its repair; for anything from
/// Foundation it showed a dump with domains, codes, and pointers. Every
/// refusal the Studies panel speaks now goes through one initializer, which
/// reads what the error already carries (its gate, its reason, its repair)
/// and never invents an action the app does not have.
public struct RefusalPresentation: Codable, Sendable, Equatable, Hashable {

    /// An action the app can take on the researcher's behalf. Only actions a
    /// Studies view really offers are listed; a refusal with no matching
    /// action says what to do in words.
    public enum AppAction: String, Codable, Sendable, CaseIterable, Hashable {
        /// Duplicate the selected study as an editable draft.
        case duplicateStudy
        /// Discard unsaved edits and reload the selected study from disk.
        case reloadStudy

        /// The button title, in the app's own words.
        public var title: String {
            switch self {
            case .duplicateStudy: "Duplicate this study"
            // The same words as the Studies form's own control.
            case .reloadStudy: "Discard edits and reload"
            }
        }
    }

    /// What happened, in plain words: the site's context, then the error's
    /// own reason.
    public var reason: String
    /// What to do next, in app terms.
    public var whatToDo: String
    /// The app action that does it, when one exists.
    public var appAction: AppAction?
    /// The repair as a command line, for people who use one. Shown behind a
    /// disclosure, never as the main instruction.
    public var commandLine: String?
    /// The refusal's machine code (a gate id or a family code), when it had one.
    public var code: String?

    public init(
        reason: String, whatToDo: String, appAction: AppAction? = nil,
        commandLine: String? = nil, code: String? = nil
    ) {
        self.reason = reason
        self.whatToDo = whatToDo
        self.appAction = appAction
        self.commandLine = commandLine
        self.code = code
    }

    /// One line for notices, inline form labels, and status slots: the
    /// reason, then what to do.
    public var summary: String { "\(reason) \(whatToDo)" }

    /// Present an error.
    ///
    /// - Parameters:
    ///   - context: what the researcher was trying to do, as a sentence
    ///     ("Couldn't remove the control."). It leads the reason.
    ///   - advice: what to check when the error itself carries no repair the
    ///     app can use. A known gate's own advice takes precedence, because it
    ///     is about what actually went wrong rather than what usually does.
    public init(_ error: any Error, context: String? = nil, advice: String? = nil) {
        let parts = Self.parts(of: error)
        let detail = Self.sentence(parts.detail)
        if let context, !context.isEmpty {
            reason = Self.sentence(context) + " " + detail
        } else {
            reason = detail
        }
        code = parts.code
        let repair = parts.repair?.trimmingCharacters(in: .whitespacesAndNewlines)
        let repairIsCommand = repair.map(Self.isCommandLine) ?? false
        commandLine = repairIsCommand ? repair : nil
        if let appWords = parts.appWords {
            whatToDo = appWords
            appAction = nil
        } else if let code = parts.code, let known = Self.known[code] {
            whatToDo = known.whatToDo
            appAction = known.action
        } else if let repair, !repair.isEmpty, !repairIsCommand {
            whatToDo = Self.sentence(repair)
            appAction = nil
        } else {
            whatToDo = Self.sentence(advice ?? Self.genericAdvice)
            appAction = nil
        }
    }

    /// Present a refusal that arrived as a code and a repair, with no error
    /// value (an evidence import's `refused` outcome).
    public init(code: String, repair: String, context: String) {
        let trimmed = repair.trimmingCharacters(in: .whitespacesAndNewlines)
        let isCommand = Self.isCommandLine(trimmed)
        reason = Self.sentence(context)
        self.code = code
        commandLine = isCommand ? trimmed : nil
        if let known = Self.known[code] {
            whatToDo = known.whatToDo
            appAction = known.action
        } else if !trimmed.isEmpty, !isCommand {
            whatToDo = Self.sentence(trimmed)
            appAction = nil
        } else {
            whatToDo = Self.sentence(Self.genericAdvice)
            appAction = nil
        }
    }

    /// The plain reason alone, for inline validation text that has its own
    /// layout. Never a raw dump.
    public static func plainReason(_ error: any Error) -> String {
        sentence(parts(of: error).detail)
    }

    static let genericAdvice =
        "Check what the message names, correct it, and try again. The notices "
        + "list keeps this message if you need it later."

    // MARK: - Known refusals

    /// What to do for the refusals whose cause is known from their code. The
    /// app action is set only where a Studies view offers it.
    static let known: [String: (whatToDo: String, action: AppAction?)] = [
        LifecycleGate.statusImmutable.rawValue: (
            "A frozen or complete study does not change. Duplicate it to get an "
                + "editable draft, and make the change there.", .duplicateStudy),
        LifecycleGate.staleManifest.rawValue: (
            "The file changed after you opened it, perhaps from the command line. "
                + "Reload the study, check what changed, and try again.", .reloadStudy),
        LifecycleGate.armsCleared.rawValue: (
            "Nothing was saved. Reload the study to see what it still holds, then "
                + "make the change again.", .reloadStudy),
        LifecycleGate.pinDrift.rawValue: (
            "A file the study pinned has changed or is missing. Restore the file, "
                + "or pin the new version in a draft.", nil),
        LifecycleGate.emptyStudy.rawValue: (
            "Attach a concept or an agent to the study first, then try again.", nil),
        LifecycleGate.studyDeclaration.rawValue: (
            "Correct the setting the message names, then try again.", nil),
        LifecycleGate.missingPrerequisite.rawValue: (
            "Add what the message names, then try again.", nil),
        LifecycleGate.artifactPin.rawValue: (
            "The file changed or could not be read. Reload the list, select it "
                + "again, and check it before trying again.", nil),
        LifecycleGate.conceptInUse.rawValue: (
            "Remove the conditions or settings that use this concept first, then "
                + "try again.", nil),
        LifecycleGate.dataReadiness.rawValue: (
            "Add the study data the message lists, then try again.", nil),
        LifecycleGate.manifestEpoch.rawValue: (
            "That run was made under different study settings. Use a run of the "
                + "current settings.", nil),
        WorkspaceHousekeeping.agentInUseCode: (
            "Remove the agent from the studies named, then delete it.", nil),
        WorkspaceHousekeeping.agentIsRunEvidenceCode: (
            "Nothing needs fixing: an agent no study uses changes nothing.", nil),
        "designChanged": (
            "The template changed after you opened it. Reload it, check what "
                + "changed, and try again.", nil),
        "freezeGateFailed": (
            "Complete the check the message names, then freeze again.", nil),
    ]

    // MARK: - Reading an error

    struct Parts {
        var detail: String
        var repair: String?
        var code: String?
        /// A next step already written for the app, which wins outright.
        var appWords: String?
    }

    static func parts(of error: any Error) -> Parts {
        switch error {
        case let error as ExperimentError:
            if let failure = error.clientIdentityFailure {
                return Parts(detail: failure.appSummary, repair: failure.repair,
                             code: "clientIdentity", appWords: failure.appNextStep)
            }
            if let refusal = error.lifecycleRefusal {
                return Parts(detail: refusal.reason, repair: refusal.repairAction, code: refusal.gateID)
            }
            if let refusal = error.freezeRefusal {
                return Parts(detail: refusal.reason, repair: refusal.repairAction, code: "freezeGateFailed")
            }
            return Parts(detail: error.reason, repair: error.malformedInvocation?.repairAction)
        case let error as StudyDesignAuthoringError:
            return Parts(detail: error.reason, repair: error.repairAction, code: error.code)
        case let error as WorkspaceHousekeeping.Refusal:
            return Parts(detail: error.reason, repair: error.repairAction, code: error.code)
        case let error as ExtractionRendering.DeclarationError:
            return Parts(detail: error.reason, repair: error.repair)
        case let error as ReadingPosition.DeclarationError:
            return Parts(detail: error.reason, repair: error.repair)
        case let error as DemoWorkspace.Refusal:
            return Parts(detail: error.reason, repair: error.repair, code: error.code.rawValue)
        case let error as DecodingError:
            return Parts(detail: describe(error))
        case let error as LocalizedError:
            return Parts(detail: error.errorDescription ?? (error as NSError).localizedDescription)
        default:
            let ns = error as NSError
            // Foundation's own errors describe themselves cleanly through
            // `localizedDescription`; their `description` is the dump.
            if ns.domain == NSCocoaErrorDomain || ns.domain == NSPOSIXErrorDomain
                || ns.domain == NSURLErrorDomain
            {
                return Parts(detail: ns.localizedDescription)
            }
            // The error's own words: its `description` when it has one.
            return Parts(detail: String(describing: error))
        }
    }

    static func describe(_ error: DecodingError) -> String {
        func path(_ context: DecodingError.Context) -> String {
            let keys = context.codingPath.map(\.stringValue).filter { !$0.isEmpty }
            return keys.isEmpty ? "" : " (at \(keys.joined(separator: ".")))"
        }
        switch error {
        case .keyNotFound(let key, let context):
            return "The file is missing the field “\(key.stringValue)”\(path(context))."
        case .typeMismatch(_, let context), .valueNotFound(_, let context):
            return "A field in the file holds the wrong kind of value\(path(context))."
        case .dataCorrupted(let context):
            return "The file is not in the expected format\(path(context))."
        @unknown default:
            return "The file is not in the expected format."
        }
    }

    /// A repair written as a command (the shape every command-line refusal
    /// takes), rather than as advice.
    static func isCommandLine(_ text: String) -> Bool {
        text.contains("steerlab-cli ") || text.hasPrefix("steerlab ")
            || text.contains(" steerlab ") || text.contains(" && ")
    }

    /// Capitalized, with closing punctuation.
    static func sentence(_ text: String) -> String {
        var trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let first = trimmed.first else { return trimmed }
        trimmed = first.uppercased() + trimmed.dropFirst()
        if let last = trimmed.last, !".!?)”\"".contains(last) { trimmed += "." }
        return trimmed
    }
}
