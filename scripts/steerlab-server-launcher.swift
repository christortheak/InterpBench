// Native arm64 launcher for "SteerLab Server.app" — click to start the web
// server, click again to stop it. State is discovered from the process
// table (pgrep), so it also stops servers started manually from a terminal
// and never double-binds the port.
//
// It builds steerlab-cli when needed with the Xcode scripts/resolve-xcode.sh
// chooses, the same rule the shell scripts use.
//
// Rebuild after editing, with the selected Xcode (the explicit -target
// matters: swiftc can default to a deployment target past the host OS, which
// macOS then refuses to launch):
//   swiftc -O -target arm64-apple-macos26.4 scripts/steerlab-server-launcher.swift \
//     -o "SteerLab Server.app/Contents/MacOS/SteerLabServer"

import Darwin
import Foundation

let port = 8080
let logPath = "/tmp/steerlab-server.log"

let launcherPath = URL(filePath: CommandLine.arguments[0]).resolvingSymlinksInPath()
let projectRoot = launcherPath
    .deletingLastPathComponent()  // MacOS
    .deletingLastPathComponent()  // Contents
    .deletingLastPathComponent()  // SteerLab Server.app
    .deletingLastPathComponent()  // project root
let derivedData = projectRoot.appending(path: ".deriveddata.nosync")
let products = derivedData.appending(path: "Build/Products/Debug")
let cli = products.appending(path: "steerlab-cli")

@discardableResult
func run(
    _ tool: String, _ arguments: [String],
    cwd: URL? = nil, capture: Bool = false
) -> (status: Int32, output: String) {
    let process = Process()
    process.executableURL = URL(filePath: tool)
    process.arguments = arguments
    if let cwd { process.currentDirectoryURL = cwd }
    let pipe = Pipe()
    if capture { process.standardOutput = pipe }
    do {
        try process.run()
    } catch {
        return (-1, "")
    }
    process.waitUntilExit()
    let data = capture ? pipe.fileHandleForReading.readDataToEndOfFile() : Data()
    return (process.terminationStatus, String(decoding: data, as: UTF8.self))
}

func notify(_ message: String) {
    run(
        "/usr/bin/osascript",
        ["-e", "display notification \"\(message)\" with title \"SteerLab Server\""])
}

/// A modal dialog. The message reaches AppleScript as an argument, never
/// spliced into the script, so the resolver's text is shown exactly as written.
func alert(_ message: String, icon: String) {
    run(
        "/usr/bin/osascript",
        [
            "-e", "on run argv",
            "-e", "display dialog (item 1 of argv) buttons {\"OK\"} default button 1 "
                + "with title \"SteerLab Server\" with icon \(icon)",
            "-e", "end run",
            message,
        ])
}

/// The xcodebuild to build with, chosen by scripts/resolve-xcode.sh — the rule
/// the shell scripts share: DEVELOPER_DIR when it is set, otherwise
/// xcode-select's choice, then /Applications/Xcode.app, never a beta by
/// preference, and Xcode 27 or later. `notes` is what the script said: which
/// Xcode and why, or, when `path` is nil, what to install or select.
func resolveXcodebuild() -> (path: String?, notes: String) {
    let process = Process()
    process.executableURL = URL(filePath: "/bin/zsh")
    process.arguments = [projectRoot.appending(path: "scripts/resolve-xcode.sh").path]
    process.currentDirectoryURL = projectRoot
    let output = Pipe()
    let errors = Pipe()
    process.standardOutput = output
    process.standardError = errors
    do {
        try process.run()
    } catch {
        return (nil, "scripts/resolve-xcode.sh could not be run.")
    }
    let developerDir = String(
        decoding: output.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self
    ).trimmingCharacters(in: .whitespacesAndNewlines)
    let notes = String(
        decoding: errors.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self
    ).trimmingCharacters(in: .whitespacesAndNewlines)
    process.waitUntilExit()
    guard process.terminationStatus == 0, !developerDir.isEmpty else { return (nil, notes) }
    return (developerDir + "/usr/bin/xcodebuild", notes)
}

// ── Toggle: stop a running server (however it was started) ────────────────
let running = run("/usr/bin/pgrep", ["-f", "steerlab-cli serve"], capture: true)
if running.status == 0,
    !running.output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
{
    run("/usr/bin/pkill", ["-f", "steerlab-cli serve"])
    notify("Server stopped")
    exit(0)
}

// ── Build the CLI on first use or when sources changed ────────────────────
// The stale-binary trap: building only when the binary is MISSING means a
// Finder launch after any source change serves yesterday's build. Staleness
// is decided by scripts/check-binary-fresh.sh (exit 1 = stale) — the same
// tested rule the SteerLab.app launcher uses.
let freshnessScript = projectRoot.appending(path: "scripts/check-binary-fresh.sh")
var buildReason: String?
if !FileManager.default.isExecutableFile(atPath: cli.path) {
    buildReason = "first launch"
} else if FileManager.default.fileExists(atPath: freshnessScript.path) {
    let freshness = run(
        "/bin/zsh",
        [
            freshnessScript.path, cli.path,
            projectRoot.appending(path: "Sources").path,
            projectRoot.appending(path: "Package.swift").path,
        ],
        cwd: projectRoot)
    if freshness.status == 1 { buildReason = "sources changed since the last build" }
}
if let buildReason {
    notify("Building steerlab-cli (\(buildReason)) — this can take a few minutes…")
    let xcode = resolveXcodebuild()
    var status: Int32 = -1
    if let xcodebuild = xcode.path {
        status = run(
            xcodebuild,
            [
                "build", "-skipMacroValidation", "-scheme", "steerlab-cli",
                "-destination", "platform=macOS,arch=arm64",
                // The auto-generated scheme gathers coverage even for plain
                // builds (the package has test targets); an instrumented CLI
                // drops default.profraw into its cwd.
                "CLANG_COVERAGE_MAPPING=NO",
                "-derivedDataPath", derivedData.path,
            ],
            cwd: projectRoot
        ).status
    }
    if status != 0 || !FileManager.default.isExecutableFile(atPath: cli.path) {
        // With no Xcode to build with, the resolver's explanation is the whole
        // story; after a failed build it still says which Xcode was used.
        let detail =
            xcode.path == nil
            ? xcode.notes
            : "Run scripts/run-app.sh in Terminal to see the build error."
                + (xcode.notes.isEmpty ? "" : "\n\n" + xcode.notes)
        if FileManager.default.isExecutableFile(atPath: cli.path) {
            // A previous build exists: warn loudly, then serve with that
            // stale build rather than dead-ending the Finder user.
            alert(
                "steerlab-cli rebuild failed — starting the PREVIOUS build (your latest "
                    + "source changes are not in it).\n\n" + detail,
                icon: "caution")
        } else {
            alert("SteerLab build failed.\n\n" + detail, icon: "stop")
            exit(1)
        }
    }
}

// ── Start detached, logging to /tmp ────────────────────────────────────────
FileManager.default.createFile(atPath: logPath, contents: nil)
let log = FileHandle(forWritingAtPath: logPath)

let server = Process()
server.executableURL = cli
server.arguments = ["serve", "--port", "\(port)"]
server.currentDirectoryURL = projectRoot
var environment = ProcessInfo.processInfo.environment
environment["DYLD_FRAMEWORK_PATH"] = products.path
server.environment = environment
if let log {
    server.standardOutput = log
    server.standardError = log
}
do {
    try server.run()
} catch {
    notify("Failed to start server — see \(logPath)")
    exit(1)
}

Thread.sleep(forTimeInterval: 1.5)
guard server.isRunning else {
    notify("Server exited immediately — see \(logPath)")
    exit(1)
}
notify("Running at http://localhost:\(port) — click again to stop")
run("/usr/bin/open", ["http://localhost:\(port)"])
exit(0)
