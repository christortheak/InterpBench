import Foundation

/// Single source of truth for the engine version stamped into frozen
/// manifests (`appVersion`) and per-run `config.json` payloads. Cross-engine
/// key: the Python server stamps its own equivalent.
///
/// Resolution order: a packaged build's resource manifest
/// (`CodeResources.buildManifest()`) carries build-generated metadata —
/// `appVersion` plus an optional `sourceRevision` stamped at packaging time —
/// and wins when present. A packaged `.app` whose manifest is unreadable
/// still carries the same two facts in its Info.plist (`SLFullVersionString`
/// / `SLSourceRevision`, written by `scripts/build-app.sh`), which come next:
/// that revision is the checkout the BUILD was assembled from — the app's own
/// build identity — never an observation of any checkout on this machine now.
/// Otherwise (the developer case): SPM has no
/// reliable way to embed the git SHA at build time (no build-tool step over
/// `.git`, and compile-time path tricks don't survive archiving), so the
/// version is a hand-maintained constant plus a best-effort *runtime* read
/// of the CODE checkout's HEAD via `CodeResources.developerCheckoutRoot` —
/// dev builds running from a checkout get "swift-app <version>+<shortSHA>",
/// git-less machines get "swift-app <version>".
public enum SteerLabVersion {
    /// Hand-maintained; bump on release-worthy changes.
    // One release version across both engines as of the public flip
    // (v0.9.0, 2026-08-20); the server's __version__ matches.
    public static let version = "0.9.3"

    /// "swift-app 0.9.0+1a2b3c4d" (dev checkout) or "swift-app 0.9.0".
    /// Computed once; neither the packaged manifest nor the code repo's HEAD
    /// can change mid-process in any way this stamp should chase.
    public static let current: String = {
        // A packaged build carries build-generated metadata — prefer it.
        if let manifestURL = (try? CodeResources.buildManifest()) ?? nil,
            let manifest = try? ResourceManifest.load(from: manifestURL)
        {
            if let revision = manifest.sourceRevision, !revision.isEmpty {
                return "swift-app \(manifest.appVersion)+\(revision)"
            }
            return "swift-app \(manifest.appVersion)"
        }
        // A packaged .app with no readable manifest: the Info.plist stamp is
        // the same build's identity, and preferring it over a live git read
        // keeps a bundle that happens to sit beside SOMEONE ELSE'S checkout
        // from reporting that checkout's HEAD as its own.
        if let bundleVersion = CodeResources.bundleFullVersion {
            if let revision = CodeResources.bundleSourceRevision {
                return "swift-app \(bundleVersion)+\(revision)"
            }
            return "swift-app \(bundleVersion)"
        }
        if let sha = codeRepoShortSHA() {
            return "swift-app \(version)+\(sha)"
        }
        return "swift-app \(version)"
    }()

    /// Best-effort short SHA of the code repository (never the workspace):
    /// nil when git or the developer checkout is unavailable.
    private static func codeRepoShortSHA() -> String? {
        guard let checkout = CodeResources.developerCheckoutRoot else {
            return nil
        }
        let process = Process()
        process.executableURL = URL(filePath: "/usr/bin/git")
        process.arguments = [
            "-C", checkout.path, "rev-parse", "--short=8", "HEAD",
        ]
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        guard (try? process.run()) != nil else { return nil }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }
        let sha = String(decoding: data, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return sha.isEmpty ? nil : sha
    }
}
