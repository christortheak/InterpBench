// Native arm64 launcher for SteerLab.app.
//
// The bundle executable must be a real Mach-O with an arm64 slice —
// LaunchServices classifies script-based bundles as Intel and offers
// Rosetta, which is misleading (and Info.plist now sets
// LSRequiresNativeExecution to forbid emulation outright). This binary does
// exactly what scripts/run-app.sh does: locate the xcodebuild-built
// products (building only if missing), set DYLD_FRAMEWORK_PATH so
// mlx-swift's Metal shader bundle resolves, and exec the app.
//
// Rebuild after editing (the explicit -target matters: this beta's swiftc
// can default past the host OS, which macOS then refuses to launch):
//   DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer xcrun swiftc \
//     -O -target arm64-apple-macos27.0 scripts/steerlab-launcher.swift \
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
let betaXcodebuild = URL(filePath: "/Applications/Xcode-beta.app/Contents/Developer/usr/bin/xcodebuild")
let xcodebuild = FileManager.default.isExecutableFile(atPath: betaXcodebuild.path)
    ? betaXcodebuild.path
    : "/usr/bin/xcodebuild"

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
    let status = run(
        xcodebuild,
        [
            "build", "-skipMacroValidation", "-scheme", "SteerLabApp",
            "-destination", "platform=macOS,arch=arm64",
            "-derivedDataPath", derivedData.path,
        ],
        cwd: projectRoot)
    if status != 0 || !FileManager.default.isExecutableFile(atPath: appBinary.path) {
        if FileManager.default.isExecutableFile(atPath: appBinary.path) {
            // A previous build exists: warn loudly (naming the terminal
            // command that shows the error), then launch that stale build
            // rather than dead-ending the Finder user.
            run(
                "/usr/bin/osascript",
                [
                    "-e",
                    "display dialog \"SteerLab rebuild failed — launching the "
                        + "PREVIOUS build (your latest source changes are not in "
                        + "it). Run scripts/run-app.sh in Terminal to see the "
                        + "build error.\" buttons {\"OK\"} default button 1 "
                        + "with title \"SteerLab\" with icon caution",
                ])
        } else {
            run(
                "/usr/bin/osascript",
                [
                    "-e",
                    "display dialog \"SteerLab build failed — run scripts/run-app.sh in "
                        + "Terminal to see the error\" buttons {\"OK\"} default button 1 "
                        + "with title \"SteerLab\" with icon stop",
                ])
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
