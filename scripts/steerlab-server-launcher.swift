// Native arm64 launcher for "SteerLab Server.app" — click to start the web
// server, click again to stop it. State is discovered from the process
// table (pgrep), so it also stops servers started manually from a terminal
// and never double-binds the port.
//
// Rebuild after editing (the explicit -target matters: this beta's swiftc
// defaults to macosx28.0, which macOS 27 refuses to launch):
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
let betaXcodebuild = URL(filePath: "/Applications/Xcode-beta.app/Contents/Developer/usr/bin/xcodebuild")
let xcodebuild = FileManager.default.isExecutableFile(atPath: betaXcodebuild.path)
    ? betaXcodebuild.path
    : "/usr/bin/xcodebuild"

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
    let build = run(
        xcodebuild,
        [
            "build", "-skipMacroValidation", "-scheme", "steerlab-cli",
            "-destination", "platform=macOS,arch=arm64", "-derivedDataPath", derivedData.path,
        ],
        cwd: projectRoot)
    if build.status != 0 || !FileManager.default.isExecutableFile(atPath: cli.path) {
        if FileManager.default.isExecutableFile(atPath: cli.path) {
            // A previous build exists: warn loudly, then serve with that
            // stale build rather than dead-ending the Finder user.
            run(
                "/usr/bin/osascript",
                [
                    "-e",
                    "display dialog \"steerlab-cli rebuild failed — starting the "
                        + "PREVIOUS build (your latest source changes are not in "
                        + "it). Run scripts/run-app.sh in Terminal to see the "
                        + "build error.\" buttons {\"OK\"} default button 1 "
                        + "with title \"SteerLab Server\" with icon caution",
                ])
        } else {
            run(
                "/usr/bin/osascript",
                [
                    "-e",
                    "display dialog \"SteerLab build failed — run scripts/run-app.sh in "
                        + "Terminal to see the error\" buttons {\"OK\"} default button 1 "
                        + "with title \"SteerLab Server\" with icon stop",
                ])
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
