// Native arm64 launcher for SteerLab.app.
//
// The bundle executable must be a real Mach-O with an arm64 slice —
// LaunchServices classifies script-based bundles as Intel and offers
// Rosetta, which is misleading (and Info.plist now sets
// LSRequiresNativeExecution to forbid emulation outright). This binary does
// exactly what scripts/run-app.sh does: locate the xcodebuild-built
// products (building only if missing), choose the Xcode the same way
// (scripts/resolve-xcode.sh), set DYLD_FRAMEWORK_PATH so mlx-swift's Metal
// shader bundle resolves, and exec the app.
//
// Rebuild after editing, with the selected Xcode (the explicit -target
// matters: swiftc can default to a deployment target past the host OS, which
// macOS then refuses to launch):
//   xcrun swiftc -O -target arm64-apple-macos27.0 scripts/steerlab-launcher.swift \
//     -o "SteerLab.app/Contents/MacOS/SteerLab"

import Darwin
import Foundation

let launcherPath = URL(filePath: CommandLine.arguments[0]).resolvingSymlinksInPath()
let projectRoot = launcherPath
    .deletingLastPathComponent()  // MacOS
    .deletingLastPathComponent()  // Contents
    .deletingLastPathComponent()  // SteerLab.app
    .deletingLastPathComponent()  // project root
let derivedData = projectRoot.appending(path: ".deriveddata.nosync")
let products = derivedData.appending(path: "Build/Products/Debug")
let appBinary = products.appending(path: "SteerLabApp")

@discardableResult
func run(_ tool: String, _ arguments: [String], cwd: URL? = nil) -> Int32 {
    let process = Process()
    process.executableURL = URL(filePath: tool)
    process.arguments = arguments
    if let cwd { process.currentDirectoryURL = cwd }
    do {
        try process.run()
    } catch {
        return -1
    }
    process.waitUntilExit()
    return process.terminationStatus
}

func notify(_ message: String) {
    run(
        "/usr/bin/osascript",
        ["-e", "display notification \"\(message)\" with title \"SteerLab\""])
}

/// A modal dialog. The message reaches AppleScript as an argument, never
/// spliced into the script, so the resolver's text is shown exactly as written.
func alert(_ message: String, icon: String) {
    run(
        "/usr/bin/osascript",
        [
            "-e", "on run argv",
            "-e", "display dialog (item 1 of argv) buttons {\"OK\"} default button 1 "
                + "with title \"SteerLab\" with icon \(icon)",
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

// Build when the binary is missing OR stale (a source file newer than the
// binary — the stale-binary trap: rebuilding only on a missing binary means
// a Finder launch after any source change silently runs yesterday's build).
// Staleness is decided by scripts/check-binary-fresh.sh (exit 1 = stale),
// so the rule is one tested script shared with the server launcher.
let freshnessScript = projectRoot.appending(path: "scripts/check-binary-fresh.sh")
var buildReason: String?
if !FileManager.default.isExecutableFile(atPath: appBinary.path) {
    buildReason = "first launch"
} else if FileManager.default.fileExists(atPath: freshnessScript.path) {
    let status = run(
        "/bin/zsh",
        [
            freshnessScript.path, appBinary.path,
            projectRoot.appending(path: "Sources").path,
            projectRoot.appending(path: "Package.swift").path,
        ],
        cwd: projectRoot)
    if status == 1 { buildReason = "sources changed since the last build" }
}

if let buildReason {
    notify("Building SteerLab (\(buildReason)) — this can take a few minutes…")
    let xcode = resolveXcodebuild()
    var status: Int32 = -1
    if let xcodebuild = xcode.path {
        status = run(
            xcodebuild,
            [
                "build", "-skipMacroValidation", "-scheme", "SteerLabApp",
                "-destination", "platform=macOS,arch=arm64",
                // The auto-generated scheme gathers coverage even for plain
                // builds (the package has test targets); an instrumented app
                // drops default.profraw into its cwd.
                "CLANG_COVERAGE_MAPPING=NO",
                "-derivedDataPath", derivedData.path,
            ],
            cwd: projectRoot)
    }
    if status != 0 || !FileManager.default.isExecutableFile(atPath: appBinary.path) {
        // With no Xcode to build with, the resolver's explanation is the whole
        // story; after a failed build it still says which Xcode was used.
        let detail =
            xcode.path == nil
            ? xcode.notes
            : "Run scripts/run-app.sh in Terminal to see the build error."
                + (xcode.notes.isEmpty ? "" : "\n\n" + xcode.notes)
        if FileManager.default.isExecutableFile(atPath: appBinary.path) {
            // A previous build exists: warn loudly, then launch that stale
            // build rather than dead-ending the Finder user.
            alert(
                "SteerLab rebuild failed — launching the PREVIOUS build (your latest "
                    + "source changes are not in it).\n\n" + detail,
                icon: "caution")
        } else {
            alert("SteerLab build failed.\n\n" + detail, icon: "stop")
            exit(1)
        }
    }
}

FileManager.default.changeCurrentDirectoryPath(projectRoot.path)
setenv("DYLD_FRAMEWORK_PATH", products.path, 1)

// Replace this process with the app so the Dock shows one process.
var argv: [UnsafeMutablePointer<CChar>?] = [strdup(appBinary.path), nil]
execv(appBinary.path, &argv)
perror("execv")  // only reached on failure
exit(1)
