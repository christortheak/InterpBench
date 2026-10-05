import Foundation

/// Why a call to the local Python client could not be confirmed to come from
/// the same source as this Mac build — one value per distinct cause, each with
/// its own wording and its own repair.
///
/// WHAT IS COMPARED. `DiagnosticWorkspace` sends the identity this build was
/// compiled against (`PythonClientIdentity.sourceSHA256`) to a child Python
/// process run against the Python client files this build resolves on this
/// machine (`CodeResources.serverPayload()`: the app bundle's own
/// `ServerPayload` in an installed app, the checkout's `Server` folder in a
/// development build). Python hashes the files it was loaded from and answers
/// with that hash. Nothing else takes part: no request to a server carries
/// the identity, and no server compares it. A controller on an older or newer
/// engine cannot cause this failure, and restarting one cannot clear it.
///
/// WHAT ACTUALLY CAUSES A MISMATCH. The two operating reports of this failure
/// (2026-09-12 and 2026-09-21) were both the same thing: the app was rebuilt
/// and installed while a `remote science-stage` command was still waiting on
/// the controller. That command was the OLD build, in memory; when the
/// controller answered, ten to thirty minutes later, it asked the NEW files
/// on disk for their identity. Every command started after the install
/// matched, with the controller untouched. So the repair that leads is a
/// fresh start, and the stage now makes this check before it asks the
/// controller for anything (`DiagnosticRemote.stage`).
///
/// The command line gets `reason` and `repair` (both sides named, short
/// hashes, the files' path). The app gets `appSummary` and `appNextStep`, in
/// a researcher's words, and may put `details` behind a disclosure.
public struct ClientIdentityFailure: Sendable, Equatable {

    public enum Cause: String, Sendable, Equatable {
        /// Python reported an identity, and it is not this build's.
        case sourcesDiffer
        /// Python answered, but stopped before it could report an identity.
        /// Not a mismatch: the two sides were never compared.
        case identityNotReported
        /// Python produced no answer at all. Not a mismatch either: the
        /// helper did not start, or died before it could say anything.
        case noAnswer
    }

    /// Where the Python client files are, which decides what "use a matching
    /// pair" means in practice.
    public enum Layout: String, Sendable, Equatable {
        /// Inside an installed app (`…/Name.app/Contents/Resources/…`).
        case appBundle
        /// A source checkout's `Server` folder (a development build).
        case developerCheckout
        /// Anywhere else — an explicitly supplied location.
        case other
    }

    public var cause: Cause
    /// The identity this build was compiled against.
    public var expected: String
    /// The identity Python reported for the files it was loaded from.
    public var actual: String?
    /// Where the Python client files are: as Python reported it, else as
    /// this build resolved it.
    public var payloadPath: String
    public var layout: Layout
    /// The files were replaced on disk after this process started — the
    /// running copy is the old build, and what is on disk is the new one.
    public var replacedWhileRunning: Bool
    /// Python's own words, when it answered.
    public var pythonReason: String?
    /// The end of what Python wrote to its error stream.
    public var errorOutput: String?
    public var exitStatus: Int32?
    public var interpreterPath: String?
    /// The controller had already staged an archive when the local step
    /// failed (`DiagnosticRemote.stage`). The wording then says what state
    /// that leaves, and that nothing on the controller needs touching.
    public var afterRemoteStage: Bool

    public init(
        cause: Cause, expected: String, actual: String? = nil,
        payloadPath: String, layout: Layout, replacedWhileRunning: Bool = false,
        pythonReason: String? = nil, errorOutput: String? = nil,
        exitStatus: Int32? = nil, interpreterPath: String? = nil,
        afterRemoteStage: Bool = false
    ) {
        self.cause = cause
        self.expected = expected
        self.actual = actual
        self.payloadPath = payloadPath
        self.layout = layout
        self.replacedWhileRunning = replacedWhileRunning
        self.pythonReason = pythonReason
        self.errorOutput = errorOutput
        self.exitStatus = exitStatus
        self.interpreterPath = interpreterPath
        self.afterRemoteStage = afterRemoteStage
    }

    // MARK: - Command-line wording

    /// Enough of a hash to tell two builds apart at a glance.
    static func short(_ identity: String) -> String { String(identity.prefix(12)) }

    /// Said wherever a reader might otherwise go looking at a server: the
    /// operating notes once recorded a controller restart as the fix.
    static let noServerTakesPart =
        "No server or controller takes part in this check: do not restart "
        + "one, and running jobs are not affected."

    private var stagedPrefix: String {
        afterRemoteStage
            ? "The controller staged the archive, but this Mac could not then "
                + "record it, so there is no local request file to plan from yet. "
            : ""
    }

    /// What is wrong, naming both sides.
    public var reason: String {
        switch cause {
        case .sourcesDiffer:
            let found = actual.map(Self.short) ?? "unknown"
            return stagedPrefix
                + "This Mac build and its Python client files are different "
                + "versions, so nothing was run locally. The build expects "
                + "sources \(Self.short(expected)); the files at \(payloadPath) "
                + "are \(found)."
                + (replacedWhileRunning
                    ? " Those files were replaced after this process started."
                    : "")
        case .identityNotReported:
            return stagedPrefix
                + "The local Python client stopped before it could say which "
                + "version it is, so nothing was run locally. This is not a "
                + "version mismatch: the two sides were never compared. Its "
                + "files are at \(payloadPath)."
                + (pythonReason.map { " It reported: \(Self.sentence($0))" } ?? "")
                + lastOutput
        case .noAnswer:
            let status = exitStatus.map { " (exit status \($0))" } ?? ""
            return stagedPrefix
                + "The local Python client did not answer\(status), so nothing "
                + "was run locally. This is a problem starting the helper, not "
                + "a version mismatch."
                + lastOutput
        }
    }

    private var lastOutput: String {
        guard let errorOutput, !errorOutput.isEmpty else { return "" }
        return " Its last output: \(errorOutput)"
    }

    private static func sentence(_ text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.hasSuffix(".") ? trimmed : trimmed + "."
    }

    /// What to do about it, for the cause this is.
    public var repair: String {
        let again =
            afterRemoteStage
            ? " Then run the same science-stage command again: the controller "
                + "already holds the staged copy and reuses it, so nothing is "
                + "uploaded again."
            : ""
        switch cause {
        case .sourcesDiffer:
            let first: String
            switch (layout, replacedWhileRunning) {
            case (.appBundle, true):
                first =
                    "Run the command again, or quit SteerLab and open it "
                    + "again. The app was updated while this copy was still "
                    + "running; a fresh start loads the new build together "
                    + "with its own files."
            case (.other, true):
                first =
                    "Run the command again. The Python client files were "
                    + "replaced while this process was running; a fresh "
                    + "start reads them together with the build they belong "
                    + "to. If the message returns, use a Mac build and "
                    + "Python client files that come from the same source."
            case (.appBundle, false):
                first =
                    "Start again first: run the command again, or quit "
                    + "SteerLab and open it again. If the message returns "
                    + "from a fresh start, the installed app is incomplete; "
                    + "reinstall the complete app."
            case (.developerCheckout, _):
                first =
                    "The Python sources in this checkout changed after this "
                    + "Mac client was built. From the checkout, run python3 "
                    + "scripts/ci/check-python-client-identity.py --write, "
                    + "then rebuild the Mac client, so both come from the "
                    + "same sources."
            case (.other, false):
                first =
                    "Use a Mac build and Python client files that come from "
                    + "the same source: reinstall the complete app, or "
                    + "rebuild both together."
            }
            return first + again + " " + Self.noServerTakesPart
                + " Setting up the helper again does not change these files."
        case .identityNotReported:
            return "Make the Python client files at \(payloadPath) complete, "
                + "ordinary, readable files: reinstalling the complete app "
                + "restores them (developers: restore the checkout's Server "
                + "folder)." + again
                + " Setting up the helper again does not change these files."
        case .noAnswer:
            // The helper's own environment is what failed, so setting it up
            // again is the right repair here — and only here.
            return ScientificPythonRuntime.setupHint + again
        }
    }

    // MARK: - App wording

    /// One sentence for a researcher: no hashes, paths, or commands.
    public var appSummary: String {
        let staged =
            afterRemoteStage
            ? "The server prepared your archive, but this Mac could not "
                + "record it. "
            : ""
        switch cause {
        case .sourcesDiffer where replacedWhileRunning:
            return staged
                + "SteerLab was updated on this Mac while this copy was "
                + "still open."
        case .sourcesDiffer:
            return staged
                + "This copy of SteerLab and its study-design helper files "
                + "are different versions."
        case .identityNotReported:
            return staged
                + "The study-design helper files in this copy of SteerLab "
                + "could not be read."
        case .noAnswer:
            return staged + "The study-design helper did not start."
        }
    }

    /// The one step a researcher can take.
    public var appNextStep: String {
        let again =
            afterRemoteStage
            ? " Then prepare the same archive again; nothing is uploaded twice."
            : ""
        switch cause {
        case .sourcesDiffer where replacedWhileRunning:
            return "Quit SteerLab and open it again." + again
        case .sourcesDiffer:
            return "Quit SteerLab and open it again. If this message "
                + "returns, reinstall SteerLab." + again
        case .identityNotReported:
            return "Reinstall SteerLab, then open it again." + again
        case .noAnswer:
            return "Open Research Setup from the Workspace menu, choose "
                + "Review Setup Plan, and approve the plan." + again
        }
    }

    /// The technical detail, for a "Details" disclosure: both identities in
    /// full, the files' path, and what Python said.
    public var details: String {
        var lines = ["Identity this build expects: \(expected)"]
        if let actual { lines.append("Identity of the helper files: \(actual)") }
        lines.append("Helper files: \(payloadPath)")
        if let interpreterPath { lines.append("Interpreter: \(interpreterPath)") }
        if let exitStatus { lines.append("Exit status: \(exitStatus)") }
        if let pythonReason, !pythonReason.isEmpty {
            lines.append("The helper reported: \(pythonReason)")
        }
        if let errorOutput, !errorOutput.isEmpty {
            lines.append("Error output: \(errorOutput)")
        }
        return lines.joined(separator: "\n")
    }

    // MARK: - Classifying where the files are

    /// The layout a payload location implies, and the folder whose
    /// replacement would mean "the app was updated": the `.app` for an
    /// installed app, the payload itself otherwise.
    static func classify(
        payload: URL,
        fileExists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) }
    ) -> (layout: Layout, installation: URL) {
        let components = payload.standardizedFileURL.pathComponents
        if let index = components.lastIndex(where: { $0.hasSuffix(".app") }),
            index + 1 < components.count, components[index + 1] == "Contents"
        {
            let app = components[...index].dropFirst()
                .reduce(URL(filePath: "/")) { $0.appending(component: $1) }
            return (.appBundle, app)
        }
        if payload.lastPathComponent == "Server",
            fileExists(
                payload.deletingLastPathComponent()
                    .appending(component: "Package.swift").path)
        {
            return (.developerCheckout, payload)
        }
        return (.other, payload)
    }

    /// Whether the files were put in place after this process started.
    ///
    /// Asked of the installation's own folders, not of every file in them:
    /// installing an app replaces the bundle, which changes those folders,
    /// and that is the event this is looking for. A development checkout is
    /// never claimed — its files change one at a time, which this cannot see
    /// and does not need to: the repair there is the same either way.
    static func replacedSinceLaunch(
        payload: URL, layout: Layout, installation: URL,
        processStart: Date? = ClientIdentityFailure.processStart,
        changeDate: (URL) -> Date? = ClientIdentityFailure.changeDate
    ) -> Bool {
        guard layout != .developerCheckout, let processStart else { return false }
        let watched = [
            installation, payload, payload.appending(component: "steerlab_server"),
        ]
        return watched.contains { url in
            changeDate(url).map { $0 > processStart } ?? false
        }
    }

    /// When this process started, from the kernel's own record.
    static var processStart: Date? {
        var info = kinfo_proc()
        var size = MemoryLayout<kinfo_proc>.stride
        var name: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_PID, getpid()]
        guard sysctl(&name, UInt32(name.count), &info, &size, nil, 0) == 0, size > 0
        else { return nil }
        let started = info.kp_proc.p_starttime
        return Date(
            timeIntervalSince1970: TimeInterval(started.tv_sec)
                + TimeInterval(started.tv_usec) / 1_000_000)
    }

    /// When a folder itself last changed: created, replaced, renamed into
    /// place, or given a new entry.
    static func changeDate(_ url: URL) -> Date? {
        let values = try? url.resourceValues(forKeys: [
            .attributeModificationDateKey, .contentModificationDateKey,
            .creationDateKey,
        ])
        return [
            values?.attributeModificationDate, values?.contentModificationDate,
            values?.creationDate,
        ].compactMap { $0 }.max()
    }
}

extension ExperimentError {
    /// The command-line form of a client identity failure. Classified as it
    /// always was (a malformed invocation: `blocked`, `usage`), so no exit
    /// code or envelope state moves; the failure itself rides along for the
    /// app, which says it in its own words.
    public static func clientIdentity(_ failure: ClientIdentityFailure) -> ExperimentError {
        ExperimentError(clientIdentityFailure: failure)
    }
}
