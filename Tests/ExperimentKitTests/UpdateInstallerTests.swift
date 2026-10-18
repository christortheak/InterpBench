import Foundation
import Synchronization
import Testing

@testable import ExperimentKit

/// Click-to-install updates (`Sources/ExperimentKit/UpdateInstaller.swift`).
///
/// No test here opens a socket or needs a signing identity: the network, the
/// code-signature check, the file system, and the clock are seams. `zipinfo`
/// and `ditto` run for real on archives the tests build, and one test signs a
/// bundle ad hoc with `codesign -s -` to prove the real requirement refuses
/// it. The doctrines under test are the ones a person would feel:
///
///   * only a notarized Developer ID copy in a writable folder installs;
///   * nothing is moved until the checksum, the signature, and the version
///     have all passed;
///   * a failed swap puts the original back;
///   * a download is bounded in size and time, and Cancel works.
@Suite(.serialized)
struct UpdateInstallerTests {
    /// The checkout commands paste safely: `$HOME`, `$(…)`, backticks, `"`,
    /// and `'` in the path reach git as written, never expanded or run.
    @Test func theCheckoutCommandsQuoteThePathForAShell() throws {
        let path = "/tmp/a $HOME b/$(echo injected)/`id`/Lab \"v2\"/it's"
        let line = UpdateAfterInstall.checkoutCommands(
            checkout: URL(filePath: path), version: try #require(SemanticVersion("1.0.0")))
        let quoted = UpdateAfterInstall.shellQuoted(path)
        #expect(line == "git -C \(quoted) fetch --tags, then git -C \(quoted) checkout v1.0.0")
        // What a shell makes of the quoted word is the path itself.
        let shell = Process()
        shell.executableURL = URL(filePath: "/bin/sh")
        shell.arguments = ["-c", "printf %s " + quoted]
        let out = Pipe()
        shell.standardOutput = out
        try shell.run()
        shell.waitUntilExit()
        let echoed = String(decoding: out.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        #expect(echoed == path)
    }


    // MARK: - Fixtures

    static let identifier = "org.example.SteerLabTest"
    static let team = "ABCDE12345"

    /// A scratch folder holding an "installed" app, removed by the caller.
    struct Scratch {
        let root: URL
        var apps: URL { root.appending(component: "Apps") }
        var installed: URL { apps.appending(component: "SteerLab.app") }

        init() throws {
            root = FileManager.default.temporaryDirectory
                .appending(component: "steerlab-updater-\(UUID().uuidString)")
            try FileManager.default.createDirectory(
                at: root.appending(component: "Apps"), withIntermediateDirectories: true)
        }

        func remove() { try? FileManager.default.removeItem(at: root) }
    }

    /// A minimal bundle: an Info.plist, an executable-shaped file, and a
    /// helper where the bundled command line lives.
    @discardableResult
    static func makeBundle(
        at app: URL, version: String, identifier: String = identifier, marker: String
    ) throws -> URL {
        let fm = FileManager.default
        try fm.createDirectory(
            at: app.appending(components: "Contents", "MacOS"), withIntermediateDirectories: true)
        try fm.createDirectory(
            at: app.appending(components: "Contents", "Helpers"), withIntermediateDirectories: true)
        let plist: [String: Any] = [
            "CFBundleIdentifier": identifier,
            "CFBundleExecutable": "SteerLabApp",
            "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": version.split(separator: "-").first.map(String.init) ?? version,
            "SLFullVersionString": version,
        ]
        let data = try PropertyListSerialization.data(
            fromPropertyList: plist, format: .xml, options: 0)
        try data.write(to: app.appending(components: "Contents", "Info.plist"))
        try Data(marker.utf8).write(to: app.appending(components: "Contents", "MacOS", "SteerLabApp"))
        try Data(marker.utf8).write(
            to: app.appending(components: "Contents", "Helpers", "steerlab-cli"))
        return app
    }

    /// `ditto -c -k --keepParent`, exactly as `build-app.sh --package` does.
    static func zip(_ app: URL, to archive: URL) throws {
        let process = Process()
        process.executableURL = URL(filePath: "/usr/bin/ditto")
        process.arguments = ["-c", "-k", "--keepParent", app.path, archive.path]
        try process.run()
        process.waitUntilExit()
        #expect(process.terminationStatus == 0)
    }

    static func sha256Line(of file: URL, name: String) throws -> Data {
        Data("\(try UpdateChecksum.sha256Hex(ofFileAt: file))  \(name)\n".utf8)
    }

    static func asset(_ name: String, size: Int64? = nil) -> ReleaseAsset {
        ReleaseAsset(
            name: name,
            downloadURL: URL(
                string: "https://github.com/acme/widget/releases/download/v0.9.9/\(name)")!,
            size: size)
    }

    // MARK: - Seams

    /// Serves canned bytes per URL; can hang a URL (for Cancel and timeout)
    /// or answer with a status. Holds every continuation so a hanging stream
    /// stays open until its task is cancelled.
    final class FakeTransport: UpdateTransport, @unchecked Sendable {
        enum Answer { case bytes(Data, declared: Int64?), status(Int), hang, fail(URLError) }
        let answers: [String: Answer]
        private let held = Mutex<[AsyncThrowingStream<UpdateTransferEvent, Error>.Continuation]>([])
        let opened = Mutex<[String]>([])

        init(_ answers: [String: Answer]) { self.answers = answers }

        func open(_ url: URL) throws -> AsyncThrowingStream<UpdateTransferEvent, Error> {
            opened.withLock { $0.append(url.lastPathComponent) }
            let (stream, continuation) = AsyncThrowingStream<UpdateTransferEvent, Error>.makeStream()
            switch answers[url.lastPathComponent] ?? .status(404) {
            case .bytes(let data, let declared):
                continuation.yield(.response(status: 200, expectedLength: declared ?? Int64(data.count)))
                var offset = 0
                while offset < data.count {
                    let end = min(data.count, offset + 64 * 1024)
                    continuation.yield(.data(data.subdata(in: offset..<end)))
                    offset = end
                }
                continuation.finish()
            case .status(let code):
                continuation.yield(.response(status: code, expectedLength: nil))
                continuation.finish()
            case .hang:
                continuation.yield(.response(status: 200, expectedLength: 1_000))
                held.withLock { $0.append(continuation) }
            case .fail(let error):
                continuation.yield(.response(status: 200, expectedLength: 1_000))
                continuation.finish(throwing: error)
            }
            return stream
        }
    }

    /// Accepts or rejects every bundle, and records the requirement it was
    /// handed — the requirement itself is tested separately.
    final class FakeSigning: UpdateCodeSigning, @unchecked Sendable {
        let accepting: Mutex<Bool>
        let seen = Mutex<[String]>([])
        init(accept: Bool = true) { accepting = Mutex(accept) }
        func runningAppSignature() -> RunningAppSignature {
            .developerID(teamID: UpdateInstallerTests.team, notarized: true)
        }
        func validate(bundle: URL, requirement: String) throws {
            seen.withLock { $0.append(requirement) }
            if !accepting.withLock({ $0 }) {
                throw UpdateInstallError.signatureRejected("refused by the test")
            }
        }
    }

    /// A clock whose deadline passes at once (`expired`) or never.
    struct FakeClock: UpdateClock {
        var expired: Bool
        func now() -> Date { Date(timeIntervalSince1970: 1_000_000) }
        func sleep(until deadline: Date) async throws {
            if expired { return }
            try await Task.sleep(for: .seconds(3_600))
        }
    }

    /// The real file system, failing the moves a test chooses.
    struct FaultyFileSystem: UpdateFileSystem {
        let live = LiveUpdateFileSystem()
        var failMove: @Sendable (URL, URL) -> Bool = { _, _ in false }
        var failTrash = false
        func exists(_ url: URL) -> Bool { live.exists(url) }
        func isDirectory(_ url: URL) -> Bool { live.isDirectory(url) }
        func createDirectory(_ url: URL) throws { try live.createDirectory(url) }
        func move(_ source: URL, to destination: URL) throws {
            if failMove(source, destination) {
                throw CocoaError(.fileWriteNoPermission)
            }
            try live.move(source, to: destination)
        }
        func remove(_ url: URL) throws { try live.remove(url) }
        func trash(_ url: URL) throws {
            if failTrash { throw CocoaError(.fileWriteNoPermission) }
            // Never the real Trash from a test: a removal stands in for it.
            try live.remove(url)
        }
        func contents(of directory: URL) throws -> [URL] { try live.contents(of: directory) }
        func write(_ data: Data, to url: URL) throws { try live.write(data, to: url) }
        func read(_ url: URL) -> Data? { live.read(url) }
        func bundleInfo(at app: URL) -> UpdateBundleInfo { live.bundleInfo(at: app) }
    }

    static func target(_ scratch: Scratch) -> UpdateInstallTarget {
        UpdateInstallTarget(appBundle: scratch.installed, bundleIdentifier: identifier, teamID: team)
    }

    static func installer(
        _ scratch: Scratch, transport: any UpdateTransport, signing: any UpdateCodeSigning = FakeSigning(),
        fileSystem: any UpdateFileSystem = FaultyFileSystem(), clock: any UpdateClock = FakeClock(expired: false),
        current: String = "0.9.8", limits: UpdateInstallLimits = .standard
    ) -> UpdateInstaller {
        UpdateInstaller(
            target: target(scratch), currentVersion: SemanticVersion(current)!,
            environment: UpdateInstallEnvironment(
                transport: transport, codeSigning: signing, fileSystem: fileSystem,
                clock: clock, isProcessAlive: { _ in false }, limits: limits))
    }

    /// A real release: a new bundle zipped the way the release is, with its
    /// checksum, served by the fake transport.
    static func release(
        in scratch: Scratch, version: String = "0.9.9", plistVersion: String? = nil,
        identifier: String = identifier, corruptChecksum: Bool = false
    ) throws -> (PublishedRelease, FakeTransport) {
        let build = scratch.root.appending(component: "build-\(UUID().uuidString)")
        let app = build.appending(component: "SteerLab.app")
        try makeBundle(
            at: app, version: plistVersion ?? version, identifier: identifier, marker: "new \(version)")
        let name = "SteerLab-\(version)+e6bd5258.zip"
        let archive = build.appending(component: name)
        try zip(app, to: archive)
        let bytes = try Data(contentsOf: archive)
        var checksum = try sha256Line(of: archive, name: name)
        if corruptChecksum {
            checksum = Data((String(repeating: "0", count: 64) + "  \(name)\n").utf8)
        }
        let transport = FakeTransport([
            name: .bytes(bytes, declared: nil),
            name + ".sha256": .bytes(checksum, declared: nil),
        ])
        let release = PublishedRelease(
            tag: "v\(version)", version: SemanticVersion(version)!, page: nil,
            assets: [asset(name, size: Int64(bytes.count)), asset(name + ".sha256")])
        return (release, transport)
    }

    // MARK: - 1. Eligibility: each refusal, and the one success

    static func context(
        app: URL? = URL(filePath: "/opt/lab/SteerLab/SteerLab.app"),
        identifier: String? = UpdateInstallerTests.identifier,
        signature: RunningAppSignature = .developerID(teamID: UpdateInstallerTests.team, notarized: true),
        checkout: URL? = nil, translocated: Bool = false, readOnly: Bool = false,
        writable: Bool = true
    ) -> UpdateInstallContext {
        UpdateInstallContext(
            appBundle: app, bundleIdentifier: identifier, signature: signature,
            sourceCheckout: checkout, isTranslocated: translocated,
            isOnReadOnlyVolume: readOnly, canWriteParentFolder: writable)
    }

    @Test func aNotarizedDeveloperIDCopyInAWritableFolderIsEligible() throws {
        let target = try UpdateInstallEligibility.evaluate(Self.context()).get()
        #expect(target.teamID == Self.team)
        #expect(target.bundleIdentifier == Self.identifier)
    }

    @Test func everyRefusalHasItsOwnReasonInOrder() {
        func refusal(_ context: UpdateInstallContext) -> UpdateInstallRefusal? {
            if case .failure(let reason) = UpdateInstallEligibility.evaluate(context) { return reason }
            return nil
        }
        #expect(refusal(Self.context(app: nil)) == .notAnAppBundle)
        #expect(refusal(Self.context(app: URL(filePath: "/build/SteerLabApp"))) == .notAnAppBundle)
        // A checkout beside the app is no reason to refuse; the signature
        // rules still refuse a copy built from it.
        let clone = URL(filePath: "/opt/lab/SteerLab/InterpBench")
        #expect(refusal(Self.context(signature: .adHoc, checkout: clone)) == .adHocSignature)
        #expect(refusal(Self.context(signature: .adHoc)) == .adHocSignature)
        #expect(refusal(Self.context(signature: .otherCertificate(teamID: "X"))) == .otherCertificate)
        #expect(refusal(Self.context(signature: .unsigned)) == .unsigned)
        #expect(refusal(Self.context(signature: .unreadable("no"))) == .signatureUnreadable("no"))
        #expect(
            refusal(Self.context(signature: .developerID(teamID: Self.team, notarized: false)))
                == .notNotarized)
        #expect(refusal(Self.context(identifier: nil)) == .missingBundleIdentifier)
        #expect(refusal(Self.context(translocated: true, readOnly: true)) == .translocated)
        #expect(refusal(Self.context(readOnly: true)) == .readOnlyVolume)
        #expect(refusal(Self.context(writable: false)) == .folderNotWritable("/opt/lab/SteerLab"))
        // A kept copy from an earlier update never updates itself in place.
        #expect(
            refusal(Self.context(app: URL(filePath: "/opt/lab/SteerLab/SteerLab-0.9.9-previous.app")))
                == .runningKeptCopy("0.9.9"))
    }

    @Test func refusalsSayWhatToDoInOneLine() {
        #expect(UpdateInstallRefusal.translocated.line.contains("Move SteerLab.app"))
        #expect(UpdateInstallRefusal.readOnlyVolume.line.contains("Applications"))
        #expect(UpdateInstallRefusal.adHocSignature.line.contains("Releases page"))
        #expect(UpdateInstallRefusal.otherCertificate.line.contains("Releases page"))
        #expect(UpdateInstallRefusal.notAnAppBundle.line.contains("Releases page"))
        #expect(!UpdateInstallRefusal.notNotarized.line.contains("\n"))
    }

    /// The home layout AGENTS.md prescribes: a notarized release beside a
    /// clone of the repository. It installs, and the target carries the
    /// clone so the sheet can say how to bring it along.
    @Test func aNotarizedReleaseBesideACloneIsEligible() throws {
        let clone = URL(filePath: "/opt/lab/SteerLab/InterpBench")
        let target = try UpdateInstallEligibility.evaluate(Self.context(checkout: clone)).get()
        #expect(target.sourceCheckout == clone)
        #expect(try UpdateInstallEligibility.evaluate(Self.context()).get().sourceCheckout == nil)
    }

    @Test func theSheetSaysHowToBringACheckoutAlongAndThatOlderCopiesGoToTheTrash() {
        let clone = URL(filePath: "/opt/lab/My SteerLab/InterpBench")
        let previous = URL(filePath: "/opt/lab/My SteerLab/SteerLab-0.9.9-previous.app")
        let withClone = UpdateInstallReview.facts(
            newVersion: SemanticVersion("1.0.0")!, currentVersion: SemanticVersion("0.9.9")!,
            downloadSize: 50_000_000, previousCopy: previous, sourceCheckout: clone
        ).map(\.text)
        let checkout = withClone.first { $0.contains("code checkout") }
        #expect(checkout?.contains("git -C '/opt/lab/My SteerLab/InterpBench' fetch --tags") == true)
        #expect(checkout?.contains("git -C '/opt/lab/My SteerLab/InterpBench' checkout v1.0.0") == true)
        #expect(!withClone.contains { $0.contains("Refresh Engine") })
        #expect(withClone.contains(UpdateInstallReview.olderCopiesLine))
        #expect(UpdateInstallReview.olderCopiesLine.contains("Trash"))
        #expect(withClone.contains(UpdateAfterInstall.clusterLine))

        let plain = UpdateInstallReview.facts(
            newVersion: SemanticVersion("1.0.0")!, currentVersion: SemanticVersion("0.9.9")!,
            downloadSize: nil, previousCopy: previous, sourceCheckout: nil
        ).map(\.text)
        #expect(plain.contains { $0.contains("Refresh Engine") })
        #expect(!plain.contains { $0.contains("code checkout") })
        #expect(plain.contains(UpdateInstallReview.olderCopiesLine))
    }

    @Test func translocationIsRecognizedByItsMountPath() {
        #expect(
            UpdateInstallContextReader.isTranslocated(
                URL(filePath: "/private/var/folders/xy/T/AppTranslocation/1234/d/SteerLab.app")))
        #expect(
            !UpdateInstallContextReader.isTranslocated(URL(filePath: "/Applications/SteerLab.app")))
    }

    // MARK: - 2. Asset selection

    @Test func theArchiveAndItsChecksumAreChosenFromTheRelease() throws {
        let pair = try UpdateReleaseAssets.select(
            tag: "v0.9.9",
            assets: [
                Self.asset("steerlab-client-0.9.9+e6bd5258.tar.gz"),
                Self.asset("SteerLab-0.9.9+e6bd5258.zip", size: 50_000_000),
                Self.asset("steerlab-client-0.9.9+e6bd5258.tar.gz.sha256"),
                Self.asset("SteerLab-0.9.9+e6bd5258.zip.sha256"),
            ]
        ).get()
        #expect(pair.archive.name == "SteerLab-0.9.9+e6bd5258.zip")
        #expect(pair.checksum.name == "SteerLab-0.9.9+e6bd5258.zip.sha256")
        #expect(pair.version == SemanticVersion("0.9.9")!)
        #expect(pair.revision == "e6bd5258")
    }

    @Test func gitHubsRewrittenSpellingOfThePlusIsAccepted() throws {
        for name in ["SteerLab-0.9.9.e6bd5258.zip", "SteerLab-0.9.9-e6bd5258.zip"] {
            let pair = try UpdateReleaseAssets.select(
                tag: "v0.9.9", assets: [Self.asset(name), Self.asset(name + ".sha256")]
            ).get()
            #expect(pair.version == SemanticVersion("0.9.9")!)
            #expect(pair.revision == "e6bd5258")
        }
        let pre = UpdateReleaseAssets.parseArchiveName("SteerLab-1.0.0-rc.1.abc12345.zip")
        #expect(pre?.0 == SemanticVersion("1.0.0-rc.1")!)
    }

    @Test func wrongNamesAreIgnoredAndAMissingChecksumRefuses() {
        let wrong = UpdateReleaseAssets.select(
            tag: "v0.9.9",
            assets: [
                Self.asset("SteerLab.zip"), Self.asset("SteerLab-0.9.9.zip"),
                Self.asset("SteerLab-0.9.9+nothex!.zip"), Self.asset("Other-0.9.9+e6bd5258.zip"),
                // Right shape, wrong version: never installed for this tag.
                Self.asset("SteerLab-0.9.8+e6bd5258.zip"),
                Self.asset("SteerLab-0.9.8+e6bd5258.zip.sha256"),
            ])
        #expect(wrong == .failure(.noAppArchive(tag: "v0.9.9")))

        let missing = UpdateReleaseAssets.select(
            tag: "v0.9.9", assets: [Self.asset("SteerLab-0.9.9+e6bd5258.zip")])
        #expect(missing == .failure(.missingChecksum(archive: "SteerLab-0.9.9+e6bd5258.zip")))

        let notAVersion = UpdateReleaseAssets.select(tag: "nightly", assets: [])
        #expect(notAVersion == .failure(.noAppArchive(tag: "nightly")))
    }

    @Test func twoCandidateArchivesAreRefusedNotGuessed() {
        let both = UpdateReleaseAssets.select(
            tag: "v0.9.9",
            assets: [
                Self.asset("SteerLab-0.9.9+e6bd5258.zip"), Self.asset("SteerLab-0.9.9+0badf00d.zip"),
                Self.asset("SteerLab-0.9.9+e6bd5258.zip.sha256"),
            ])
        guard case .failure(.ambiguousAppArchive(let names)) = both else {
            Issue.record("expected ambiguity, got \(both)")
            return
        }
        #expect(names.count == 2)
    }

    @Test func onlyGitHubHostsOverHTTPSAreTrusted() {
        let name = "SteerLab-0.9.9+e6bd5258.zip"
        let foreign = ReleaseAsset(
            name: name, downloadURL: URL(string: "https://example.com/acme/widget/releases/download/v0.9.9/\(name)")!)
        let plain = ReleaseAsset(
            name: name + ".sha256",
            downloadURL: URL(string: "http://github.com/acme/widget/releases/download/v0.9.9/\(name).sha256")!)
        #expect(
            UpdateReleaseAssets.select(tag: "v0.9.9", assets: [foreign, Self.asset(name + ".sha256")])
                == .failure(.untrustedDownloadLocation("example.com")))
        #expect(
            UpdateReleaseAssets.select(tag: "v0.9.9", assets: [Self.asset(name), plain])
                == .failure(.untrustedDownloadLocation("github.com")))
        #expect(UpdateReleaseAssets.isTrustedHost(URL(string: "https://release-assets.githubusercontent.com/x")!))
        #expect(UpdateReleaseAssets.isTrustedHost(URL(string: "https://api.github.com/repositories/1/releases/latest")!))
        #expect(!UpdateReleaseAssets.isTrustedHost(URL(string: "https://github.com.example.net/x")!))
        #expect(!UpdateReleaseAssets.isTrustedHost(URL(string: "https://user:pw@github.com/x")!))
        #expect(!UpdateReleaseAssets.isTrustedHost(URL(string: "https://github.com:8443/x")!))
        // A renamed repository's downloads still qualify: the owner and name
        // are not pinned, only the shape of a release download.
        #expect(
            UpdateReleaseAssets.isTrustedAssetURL(
                URL(string: "https://github.com/new-owner/new-name/releases/download/v1.0.0/\(name)")!))
        #expect(
            !UpdateReleaseAssets.isTrustedAssetURL(URL(string: "https://github.com/acme/widget/archive/\(name)")!))
    }

    @Test func theFeedReadsTheAttachedFiles() {
        let json = """
            {"tag_name": "v0.9.9", "html_url": "https://github.com/acme/widget/releases/tag/v0.9.9",
             "draft": false, "assets": [
               {"name": "SteerLab-0.9.9+e6bd5258.zip", "size": 50710542,
                "browser_download_url": "https://github.com/acme/widget/releases/download/v0.9.9/SteerLab-0.9.9+e6bd5258.zip"},
               {"name": "broken"},
               {"name": "SteerLab-0.9.9+e6bd5258.zip.sha256", "size": 94,
                "browser_download_url": "https://github.com/acme/widget/releases/download/v0.9.9/SteerLab-0.9.9+e6bd5258.zip.sha256"}]}
            """
        guard case .release(let tag, _, let assets) = GitHubReleaseFeed.outcome(fromReleaseJSON: Data(json.utf8))
        else {
            Issue.record("expected a release")
            return
        }
        #expect(tag == "v0.9.9")
        #expect(assets.map(\.name) == ["SteerLab-0.9.9+e6bd5258.zip", "SteerLab-0.9.9+e6bd5258.zip.sha256"])
        #expect(assets.first?.size == 50_710_542)
    }

    // MARK: - The signature requirement

    @Test func theRequirementIsBuiltFromTheRunningAppsOwnTeamAndIdentifier() throws {
        let text = try UpdateSignatureRequirement.text(bundleIdentifier: "org.example.App", teamID: "QWERTY1234")
        #expect(text.hasPrefix("identifier \"org.example.App\" and anchor apple generic"))
        #expect(text.contains("certificate leaf[subject.OU] = \"QWERTY1234\""))
        #expect(text.contains("certificate 1[field.1.2.840.113635.100.6.2.6] exists"))
        #expect(text.contains("certificate leaf[field.1.2.840.113635.100.6.1.13] exists"))
        #expect(text.hasSuffix(" and notarized"))
        // The installer hands the seam exactly this, from ITS target.
        let other = try UpdateSignatureRequirement.text(bundleIdentifier: "org.example.App", teamID: "ZZZZZ99999")
        #expect(other != text)
    }

    @Test func nothingThatIsNotAnIdentifierReachesTheRequirementLanguage() {
        #expect(throws: UpdateInstallError.self) {
            try UpdateSignatureRequirement.text(bundleIdentifier: "x\" or anchor apple", teamID: Self.team)
        }
        #expect(throws: UpdateInstallError.self) {
            try UpdateSignatureRequirement.text(bundleIdentifier: Self.identifier, teamID: "abc")
        }
        #expect(throws: UpdateInstallError.self) {
            try UpdateSignatureRequirement.text(bundleIdentifier: Self.identifier, teamID: "ABCDE1234\"")
        }
    }

    /// The one real signature test: a bundle signed ad hoc, in the test,
    /// must fail the Developer ID + notarized requirement, and must read as
    /// ad hoc to the eligibility check.
    @Test func anAdHocSignedBundleFailsTheRealRequirement() throws {
        let scratch = try Scratch()
        defer { scratch.remove() }
        let app = try Self.makeBundle(at: scratch.installed, version: "0.9.9", marker: "adhoc")
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: app.appending(components: "Contents", "MacOS", "SteerLabApp").path)
        // A real Mach-O to sign: copy a system binary in as the executable.
        try FileManager.default.removeItem(at: app.appending(components: "Contents", "MacOS", "SteerLabApp"))
        try FileManager.default.copyItem(
            at: URL(filePath: "/usr/bin/true"), to: app.appending(components: "Contents", "MacOS", "SteerLabApp"))
        try FileManager.default.removeItem(at: app.appending(components: "Contents", "Helpers"))
        let sign = Process()
        sign.executableURL = URL(filePath: "/usr/bin/codesign")
        sign.arguments = ["--force", "--sign", "-", app.path]
        sign.standardError = FileHandle.nullDevice
        try sign.run()
        sign.waitUntilExit()
        #expect(sign.terminationStatus == 0)

        let signing = SecurityFrameworkCodeSigning()
        #expect(signing.signature(ofBundleAt: app) == .adHoc)
        let requirement = try UpdateSignatureRequirement.text(bundleIdentifier: Self.identifier, teamID: Self.team)
        #expect(throws: UpdateInstallError.self) {
            try signing.validate(bundle: app, requirement: requirement)
        }
        // And an unsigned bundle fails it too.
        let unsigned = try Self.makeBundle(
            at: scratch.apps.appending(component: "Unsigned.app"), version: "0.9.9", marker: "u")
        #expect(throws: UpdateInstallError.self) {
            try signing.validate(bundle: unsigned, requirement: requirement)
        }
    }

    /// The positive half needs a notarized Developer ID build, which a test
    /// cannot make. Point `STEERLAB_UPDATE_NOTARIZED_APP` at one (a release
    /// build's SteerLab.app, never the installed copy) to run it; it reads
    /// the bundle's own team and identifier and validates it strictly.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["STEERLAB_UPDATE_NOTARIZED_APP"] != nil))
    func aNotarizedReleaseBuildPassesWhenOneIsProvided() throws {
        let path = try #require(ProcessInfo.processInfo.environment["STEERLAB_UPDATE_NOTARIZED_APP"])
        let app = URL(filePath: path)
        let signing = SecurityFrameworkCodeSigning()
        guard case .developerID(let team, let notarized) = signing.signature(ofBundleAt: app) else {
            Issue.record("\(path) is not a Developer ID build")
            return
        }
        #expect(notarized)
        let identifier = try #require(LiveUpdateFileSystem().bundleInfo(at: app).identifier)
        let requirement = try UpdateSignatureRequirement.text(bundleIdentifier: identifier, teamID: team)
        try signing.validate(bundle: app, requirement: requirement)
        // The same build fails a requirement naming any other team.
        let otherTeam = team == "ZZZZZ99999" ? "YYYYY88888" : "ZZZZZ99999"
        let wrong = try UpdateSignatureRequirement.text(bundleIdentifier: identifier, teamID: otherTeam)
        #expect(throws: UpdateInstallError.self) { try signing.validate(bundle: app, requirement: wrong) }
    }

    /// The whole path on a REAL release archive, for the integrator: point
    /// `STEERLAB_UPDATE_RELEASE_ZIP` at a published `SteerLab-<v>+<r>.zip`
    /// with its `.sha256` beside it. The archive is served through the fake
    /// transport (no network), unpacked with `ditto -x -k`, and validated by
    /// the REAL signature check against the team and identifier of that same
    /// build — which proves the unpacked app (the MLX shader library's
    /// signature lives in an extended attribute) survives unpacking. It then
    /// swaps it over a stand-in "installed" app in a scratch folder.
    @Test(.enabled(if: ProcessInfo.processInfo.environment["STEERLAB_UPDATE_RELEASE_ZIP"] != nil))
    func aRealReleaseArchiveVerifiesAndInstallsInAScratchFolder() async throws {
        let zipPath = try #require(ProcessInfo.processInfo.environment["STEERLAB_UPDATE_RELEASE_ZIP"])
        let zip = URL(filePath: zipPath)
        let name = zip.lastPathComponent
        let (version, _) = try #require(UpdateReleaseAssets.parseArchiveName(name))
        let scratch = try Scratch()
        defer { scratch.remove() }

        // Stand in for "the running app's own signature": read it from the
        // same build, unpacked separately.
        let peek = scratch.root.appending(component: "peek")
        let ditto = Process()
        ditto.executableURL = URL(filePath: "/usr/bin/ditto")
        ditto.arguments = ["-x", "-k", zip.path, peek.path]
        try ditto.run()
        ditto.waitUntilExit()
        let peeked = try #require(
            try FileManager.default.contentsOfDirectory(at: peek, includingPropertiesForKeys: nil)
                .first { $0.pathExtension == "app" })
        let signing = SecurityFrameworkCodeSigning()
        guard case .developerID(let team, true) = signing.signature(ofBundleAt: peeked) else {
            Issue.record("\(name) is not a notarized Developer ID build")
            return
        }
        let identifier = try #require(LiveUpdateFileSystem().bundleInfo(at: peeked).identifier)

        try Self.makeBundle(at: scratch.installed, version: "0.0.1", identifier: identifier, marker: "old")
        let bytes = try Data(contentsOf: zip)
        let checksum = try Data(contentsOf: zip.deletingLastPathComponent().appending(component: name + ".sha256"))
        let transport = FakeTransport([
            name: .bytes(bytes, declared: nil), name + ".sha256": .bytes(checksum, declared: nil),
        ])
        let release = PublishedRelease(
            tag: "v\(version.releaseDescription)", version: version, page: nil,
            assets: [Self.asset(name, size: Int64(bytes.count)), Self.asset(name + ".sha256")])
        let installer = UpdateInstaller(
            target: UpdateInstallTarget(appBundle: scratch.installed, bundleIdentifier: identifier, teamID: team),
            currentVersion: SemanticVersion("0.0.1")!,
            environment: UpdateInstallEnvironment(
                transport: transport, codeSigning: signing, isProcessAlive: { _ in false }))
        let staged = try await installer.prepare(release)
        let receipt = try installer.install(staged)
        #expect(receipt.installedVersion == version)
        // The installed copy still validates in place.
        try signing.validate(
            bundle: scratch.installed,
            requirement: try UpdateSignatureRequirement.text(bundleIdentifier: identifier, teamID: team))
    }

    // MARK: - Checksum and archive listing

    @Test func theChecksumFileIsReadByItsFirstField() {
        let digest = String(repeating: "ab", count: 32)
        #expect(UpdateChecksum.expectedDigest(fromChecksumFile: "\(digest)  SteerLab-0.9.9+e6bd5258.zip\n") == digest)
        #expect(UpdateChecksum.expectedDigest(fromChecksumFile: digest.uppercased()) == digest)
        #expect(UpdateChecksum.expectedDigest(fromChecksumFile: "not a digest  file.zip") == nil)
        #expect(UpdateChecksum.expectedDigest(fromChecksumFile: "") == nil)
    }

    @Test func aDittoArchiveOfOneAppPassesTheListingCheck() async throws {
        let scratch = try Scratch()
        defer { scratch.remove() }
        let app = try Self.makeBundle(at: scratch.apps.appending(component: "SteerLab.app"), version: "0.9.9", marker: "x")
        try FileManager.default.createSymbolicLink(
            at: app.appending(components: "Contents", "Helpers", "link"), withDestinationURL: URL(filePath: "steerlab-cli"))
        let archive = scratch.root.appending(component: "a.zip")
        try Self.zip(app, to: archive)
        let lines = ProvisionLineBuffer()
        let code = try await SystemProvisionRunner().run(["/usr/bin/zipinfo", archive.path]) { lines.append($0) }
        #expect(code == 0)
        let summary = try UpdateArchiveListing.validate(zipinfoOutput: lines.snapshot()).get()
        #expect(summary.appName == "SteerLab.app")
        #expect(summary.entries >= 6)
    }

    static func line(_ mode: String, _ size: Int, _ name: String) -> String {
        "\(mode)  2.1 unx \(size) bX defN 26-Oct-17 15:50 \(name)"
    }

    @Test func archivesThatWouldWriteOutsideTheAppAreRefused() {
        func verdict(_ names: [(String, String)]) -> UpdateInstallError? {
            let lines = ["Archive:  x.zip", "Zip file size: 1 bytes, number of entries: 2"]
                + names.map { Self.line($0.0, 1, $0.1) } + ["2 files, 2 bytes uncompressed, 1 bytes compressed:  50.0%"]
            if case .failure(let error) = UpdateArchiveListing.validate(zipinfoOutput: lines) { return error }
            return nil
        }
        let file = "-rw-r--r--"
        let dir = "drwxr-xr-x"
        let link = "lrwxr-xr-x"
        #expect(verdict([(dir, "SteerLab.app/"), (file, "SteerLab.app/Contents/Info.plist")]) == nil)
        #expect(verdict([(dir, "SteerLab.app/"), (file, "._SteerLab.app")]) == nil)
        #expect(verdict([(dir, "SteerLab.app/"), (file, "/etc/passwd")]) != nil)
        #expect(verdict([(dir, "SteerLab.app/"), (file, "SteerLab.app/../../x")]) != nil)
        #expect(verdict([(dir, "SteerLab.app/"), (file, "Other.app/x")]) != nil)
        #expect(verdict([(dir, "SteerLab.app/"), (file, "README.txt")]) != nil)
        #expect(verdict([(file, "README.txt")]) != nil)
        // Writing through a link the archive itself created.
        #expect(
            verdict([(dir, "SteerLab.app/"), (link, "SteerLab.app/Contents"), (file, "SteerLab.app/Contents/x")])
                != nil)
        // Order does not matter: the link may be listed after the file.
        #expect(
            verdict([(dir, "SteerLab.app/"), (file, "SteerLab.app/Contents/x"), (link, "SteerLab.app/Contents")])
                != nil)
        #expect(verdict([(dir, "SteerLab.app/"), ("garbage", "line")]) != nil)
        // A link named "x " (trailing space) is still the link "x ".
        #expect(
            verdict([(dir, "SteerLab.app/"), (link, "SteerLab.app/x "), (file, "SteerLab.app/x /f")])
                != nil)
        // On a case-insensitive volume `A` and `a` are one folder.
        #expect(
            verdict([(dir, "SteerLab.app/"), (link, "SteerLab.app/Contents/A"), (file, "SteerLab.app/Contents/a/x")])
                != nil)
    }

    @Test func listingBoundsStopAnArchiveBomb() {
        var limits = UpdateInstallLimits.standard
        limits.maximumUnpackedBytes = 100
        let lines = [Self.line("drwxr-xr-x", 0, "SteerLab.app/"), Self.line("-rw-r--r--", 101, "SteerLab.app/big")]
        #expect(
            UpdateArchiveListing.validate(zipinfoOutput: lines, limits: limits)
                == .failure(.archiveRejected("it would unpack to more than 100 bytes")))
        limits = .standard
        limits.maximumArchiveEntries = 1
        guard case .failure(.archiveRejected) = UpdateArchiveListing.validate(zipinfoOutput: lines, limits: limits) else {
            Issue.record("expected the entry bound to refuse")
            return
        }
    }

    // MARK: - What ditto actually wrote

    struct Unpacked {
        let root: URL
        var app: URL { root.appending(components: "unpacked", "SteerLab.app") }
        var unpacked: URL { root.appending(component: "unpacked") }

        init() throws {
            root = FileManager.default.temporaryDirectory
                .appending(component: "steerlab-unpacked-\(UUID().uuidString)")
            try UpdateInstallerTests.makeBundle(
                at: root.appending(components: "unpacked", "SteerLab.app"), version: "0.9.9", marker: "x")
        }

        func link(_ relative: String, to target: String) throws {
            try FileManager.default.createSymbolicLink(
                atPath: app.appending(path: relative).path, withDestinationPath: target)
        }

        func remove() { try? FileManager.default.removeItem(at: root) }
    }

    @Test func oneAppWithLinksInsideItPassesTheWalk() throws {
        let tree = try Unpacked()
        defer { tree.remove() }
        try tree.link("Contents/Helpers/alias", to: "steerlab-cli")
        try tree.link("Contents/Resources", to: "../Contents/MacOS")
        #expect(try UpdateUnpackedTree.inspect(tree.unpacked).get() == tree.app)
    }

    @Test func theWalkRefusesAnythingButOneAppAndLinksThatStayInside() throws {
        func refused(_ build: (Unpacked) throws -> Void, limits: UpdateInstallLimits = .standard) throws -> Bool {
            let tree = try Unpacked()
            defer { tree.remove() }
            try build(tree)
            if case .failure = UpdateUnpackedTree.inspect(tree.unpacked, limits: limits) { return true }
            return false
        }
        // Something beside the app (an entry only the local headers named).
        #expect(try refused { tree in
            try Data("x".utf8).write(to: tree.unpacked.appending(component: "extra.txt"))
        })
        // Absolute, relative, and composed escapes.
        #expect(try refused { try $0.link("Contents/out", to: "/etc") })
        #expect(try refused { try $0.link("Contents/out", to: "../../../outside") })
        #expect(try refused { tree in
            // B points at the app itself (inside); "B/.." reads as inside but
            // resolves to the folder above the app.
            try tree.link("Contents/B", to: "..")
            try tree.link("Contents/A", to: "B/..")
        })
        // Bounds.
        var few = UpdateInstallLimits.standard
        few.maximumArchiveEntries = 3
        #expect(try refused({ _ in }, limits: few))
        var small = UpdateInstallLimits.standard
        small.maximumUnpackedBytes = 4
        #expect(try refused({ _ in }, limits: small))
    }

    /// Runs the real tools, but after `ditto` also writes a file beside the
    /// app: what an entry named only in the archive's local headers does,
    /// which `zipinfo` never lists.
    struct DittoWritesMoreThanListed: ProvisionCommandRunner {
        let real = SystemProvisionRunner()
        func run(_ argv: [String], onLine: @escaping @Sendable (String) -> Void) async throws -> Int32 {
            let code = try await real.run(argv, onLine: onLine)
            if argv.first == "/usr/bin/ditto", let destination = argv.last {
                try Data("unlisted".utf8).write(
                    to: URL(filePath: destination).appending(component: "unlisted.txt"))
            }
            return code
        }
    }

    @Test func whatDittoWroteBeyondTheListingStopsTheUpdate() async throws {
        let scratch = try Scratch()
        defer { scratch.remove() }
        try Self.makeBundle(at: scratch.installed, version: "0.9.8", marker: "old")
        let (release, transport) = try Self.release(in: scratch)
        let signing = FakeSigning()
        let installer = UpdateInstaller(
            target: Self.target(scratch), currentVersion: SemanticVersion("0.9.8")!,
            environment: UpdateInstallEnvironment(
                transport: transport, codeSigning: signing, tools: DittoWritesMoreThanListed(),
                fileSystem: FaultyFileSystem(), clock: FakeClock(expired: false),
                isProcessAlive: { _ in false }))
        await #expect {
            _ = try await installer.prepare(release)
        } throws: { error in
            guard case .archiveRejected = error as? UpdateInstallError else { return false }
            return true
        }
        #expect(signing.seen.withLock { $0.isEmpty })  // stopped before the signature
        #expect(!FileManager.default.fileExists(atPath: installer.stagingRoot.path))
    }

    @Test func theTopItemMustBeAnAppFolderNotALink() throws {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "steerlab-unpacked-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let real = try Self.makeBundle(at: root.appending(components: "elsewhere", "Real.app"), version: "0.9.9", marker: "x")
        try FileManager.default.createDirectory(at: root.appending(component: "unpacked"), withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: root.appending(components: "unpacked", "SteerLab.app"), withDestinationURL: real)
        guard case .failure = UpdateUnpackedTree.inspect(root.appending(component: "unpacked")) else {
            Issue.record("a link at the top level was accepted")
            return
        }
    }

    // MARK: - Download: mismatch, size, cancel, timeout

    @Test func aChecksumMismatchStopsBeforeUnpackingAndCleansUp() async throws {
        let scratch = try Scratch()
        defer { scratch.remove() }
        try Self.makeBundle(at: scratch.installed, version: "0.9.8", marker: "old")
        let (release, transport) = try Self.release(in: scratch, corruptChecksum: true)
        let signing = FakeSigning()
        let installer = Self.installer(scratch, transport: transport, signing: signing)
        await #expect {
            _ = try await installer.prepare(release)
        } throws: { error in
            guard case .checksumMismatch = error as? UpdateInstallError else { return false }
            return true
        }
        #expect(signing.seen.withLock { $0.isEmpty })  // never reached the signature
        #expect(!FileManager.default.fileExists(atPath: installer.stagingRoot.path))
        #expect(LiveUpdateFileSystem().bundleInfo(at: scratch.installed).version == "0.9.8")
    }

    @Test func anOversizedDownloadIsRefusedDeclaredOrNot() async throws {
        let scratch = try Scratch()
        defer { scratch.remove() }
        let clock = FakeClock(expired: false)
        let destination = scratch.root.appending(component: "file")
        let url = URL(string: "https://github.com/acme/widget/releases/download/v0.9.9/big.zip")!
        // Declared too large: refused before reading a byte.
        let declared = FakeTransport(["big.zip": .bytes(Data(count: 10), declared: 5_000)])
        await #expect(throws: UpdateInstallError.archiveTooLarge(bytes: 5_000, limit: 100)) {
            _ = try await UpdateDownload.receive(
                url, into: destination, byteLimit: 100, deadline: .distantFuture,
                transport: declared, clock: clock)
        }
        // Declared small, but the body runs past the limit: stopped.
        let liar = FakeTransport(["big.zip": .bytes(Data(count: 200), declared: 50)])
        await #expect(throws: UpdateInstallError.archiveTooLarge(bytes: 200, limit: 100)) {
            _ = try await UpdateDownload.receive(
                url, into: destination, byteLimit: 100, deadline: .distantFuture,
                transport: liar, clock: clock)
        }
        // The release's own reported size is checked before downloading.
        try Self.makeBundle(at: scratch.installed, version: "0.9.8", marker: "old")
        let (release, transport) = try Self.release(in: scratch)
        var limits = UpdateInstallLimits.standard
        limits.maximumArchiveBytes = 10
        let installer = Self.installer(scratch, transport: transport, limits: limits)
        await #expect {
            _ = try await installer.prepare(release)
        } throws: { error in
            guard case .archiveTooLarge = error as? UpdateInstallError else { return false }
            return true
        }
        #expect(transport.opened.withLock { $0.isEmpty })
    }

    @Test func aShortBodyOrAnErrorStatusIsAFailedDownload() async throws {
        let scratch = try Scratch()
        defer { scratch.remove() }
        let url = URL(string: "https://github.com/acme/widget/releases/download/v0.9.9/f.zip")!
        let destination = scratch.root.appending(component: "f")
        let short = FakeTransport(["f.zip": .bytes(Data(count: 10), declared: 20)])
        await #expect(throws: UpdateInstallError.downloadFailed("it ended after 10 of 20 bytes")) {
            _ = try await UpdateDownload.receive(
                url, into: destination, byteLimit: 100, deadline: .distantFuture, transport: short,
                clock: FakeClock(expired: false))
        }
        let missing = FakeTransport(["f.zip": .status(404)])
        await #expect(throws: UpdateInstallError.downloadFailed("GitHub answered HTTP 404")) {
            _ = try await UpdateDownload.receive(
                url, into: destination, byteLimit: 100, deadline: .distantFuture, transport: missing,
                clock: FakeClock(expired: false))
        }
        await #expect(throws: UpdateInstallError.untrustedDownloadLocation("example.com")) {
            _ = try await UpdateDownload.receive(
                URL(string: "https://example.com/f.zip")!, into: destination, byteLimit: 100,
                deadline: .distantFuture, transport: missing, clock: FakeClock(expired: false))
        }
    }

    @Test func aStallIsReportedAsAStallNotAsTheDeadline() async throws {
        let scratch = try Scratch()
        defer { scratch.remove() }
        let url = URL(string: "https://github.com/acme/widget/releases/download/v0.9.9/stall.zip")!
        let transport = FakeTransport(["stall.zip": .fail(URLError(.timedOut))])
        await #expect(throws: UpdateInstallError.downloadFailed("no data arrived for a minute")) {
            _ = try await UpdateDownload.receive(
                url, into: scratch.root.appending(component: "stall"), byteLimit: 10_000,
                deadline: .distantFuture, transport: transport, clock: FakeClock(expired: false))
        }
    }

    @Test func updateRequestsSendAFixedUserAgent() {
        let lookup = GitHubReleaseFeed.request(
            for: URL(string: "https://api.github.com/repos/acme/widget/releases/latest")!, timeout: 8)
        #expect(lookup.value(forHTTPHeaderField: "User-Agent") == "SteerLab")
        #expect(lookup.httpBody == nil)
        let download = GitHubUpdateTransport.sessionConfiguration(stallTimeout: 60)
        #expect(download.httpAdditionalHeaders?["User-Agent"] as? String == "SteerLab")
        #expect(download.httpCookieAcceptPolicy == .never)
        #expect(download.urlCredentialStorage == nil)
    }

    @Test func aStalledDownloadTimesOutOnTheInjectedClock() async throws {
        let scratch = try Scratch()
        defer { scratch.remove() }
        let url = URL(string: "https://github.com/acme/widget/releases/download/v0.9.9/slow.zip")!
        let transport = FakeTransport(["slow.zip": .hang])
        await #expect {
            _ = try await UpdateDownload.receive(
                url, into: scratch.root.appending(component: "slow"), byteLimit: 10_000,
                deadline: Date(timeIntervalSince1970: 1_000_000 + 1_800), transport: transport,
                clock: FakeClock(expired: true))
        } throws: { error in
            (error as? UpdateInstallError) == .downloadTimedOut(minutes: 30)
        }
    }

    @Test func cancelStopsTheDownloadAndRemovesTheStaging() async throws {
        let scratch = try Scratch()
        defer { scratch.remove() }
        try Self.makeBundle(at: scratch.installed, version: "0.9.8", marker: "old")
        let (release, served) = try Self.release(in: scratch)
        // Serve the checksum, then hang on the archive.
        let checksumName = release.assets[1].name
        guard case .bytes(let checksum, _)? = served.answers[checksumName] else {
            Issue.record("fixture checksum missing")
            return
        }
        let transport = FakeTransport([checksumName: .bytes(checksum, declared: nil), release.assets[0].name: .hang])
        let installer = Self.installer(scratch, transport: transport)
        let task = Task { try await installer.prepare(release) }
        for _ in 0..<500 where transport.opened.withLock({ $0.count }) < 2 {
            try await Task.sleep(for: .milliseconds(10))
        }
        #expect(transport.opened.withLock { $0.count } == 2)
        task.cancel()
        let result = await task.result
        guard case .failure(let error) = result else {
            Issue.record("expected a cancellation")
            return
        }
        #expect((error as? UpdateInstallError) == .cancelled)
        #expect(!FileManager.default.fileExists(atPath: installer.stagingRoot.path))
        #expect(LiveUpdateFileSystem().bundleInfo(at: scratch.installed).version == "0.9.8")
    }

    // MARK: - Version gating

    @Test func neitherTheSameVersionNorADowngradeIsDownloaded() async throws {
        let scratch = try Scratch()
        defer { scratch.remove() }
        try Self.makeBundle(at: scratch.installed, version: "0.9.9", marker: "running")
        for version in ["0.9.9", "0.9.8"] {
            let (release, transport) = try Self.release(in: scratch, version: version)
            let installer = Self.installer(scratch, transport: transport, current: "0.9.9")
            await #expect(
                throws: UpdateInstallError.notNewer(
                    found: SemanticVersion(version)!, running: SemanticVersion("0.9.9")!)
            ) {
                _ = try await installer.prepare(release)
            }
            #expect(transport.opened.withLock { $0.isEmpty })
        }
        // A pre-release of the running version is older than it, not newer.
        #expect(!(SemanticVersion("0.9.9-rc.1")! > SemanticVersion("0.9.9")!))
    }

    @Test func theUnpackedAppMustBeTheVersionTheReleaseNames() async throws {
        let scratch = try Scratch()
        defer { scratch.remove() }
        try Self.makeBundle(at: scratch.installed, version: "0.9.8", marker: "old")
        // The archive is named 0.9.9 but the app inside says 0.9.7.
        let (release, transport) = try Self.release(in: scratch, version: "0.9.9", plistVersion: "0.9.7")
        let installer = Self.installer(scratch, transport: transport)
        await #expect {
            _ = try await installer.prepare(release)
        } throws: { error in
            guard case .versionMismatch = error as? UpdateInstallError else { return false }
            return true
        }
        // And one with another app's identifier is refused too.
        let (foreign, foreignTransport) = try Self.release(in: scratch, identifier: "org.example.Other")
        await #expect {
            _ = try await Self.installer(scratch, transport: foreignTransport).prepare(foreign)
        } throws: { error in
            guard case .versionMismatch = error as? UpdateInstallError else { return false }
            return true
        }
        #expect(!FileManager.default.fileExists(atPath: installer.stagingRoot.path))
    }

    @Test func aSignatureRefusalStopsBeforeAnythingMoves() async throws {
        let scratch = try Scratch()
        defer { scratch.remove() }
        try Self.makeBundle(at: scratch.installed, version: "0.9.8", marker: "old")
        let (release, transport) = try Self.release(in: scratch)
        let signing = FakeSigning(accept: false)
        let installer = Self.installer(scratch, transport: transport, signing: signing)
        await #expect(throws: UpdateInstallError.signatureRejected("refused by the test")) {
            _ = try await installer.prepare(release)
        }
        let expected = try UpdateSignatureRequirement.text(bundleIdentifier: Self.identifier, teamID: Self.team)
        #expect(signing.seen.withLock { $0 } == [expected])
        #expect(!FileManager.default.fileExists(atPath: installer.stagingRoot.path))
    }

    // MARK: - The swap, and its rollback

    @Test func theHappyPathKeepsThePreviousCopyAndTheCommandLineLink() async throws {
        let scratch = try Scratch()
        defer { scratch.remove() }
        try Self.makeBundle(at: scratch.installed, version: "0.9.8", marker: "old")
        // ~/.local/bin/steerlab-cli is a symlink into Contents/Helpers.
        let link = scratch.root.appending(component: "steerlab-cli")
        try FileManager.default.createSymbolicLink(
            at: link, withDestinationURL: scratch.installed.appending(components: "Contents", "Helpers", "steerlab-cli"))
        // An older kept copy of this app, and an unrelated one that must stay.
        try Self.makeBundle(
            at: scratch.apps.appending(component: "SteerLab-0.9.7-previous.app"), version: "0.9.7", marker: "older")
        try Self.makeBundle(
            at: scratch.apps.appending(component: "SteerLab-0.9.6-previous.app"), version: "0.9.6",
            identifier: "org.example.SomethingElse", marker: "foreign")

        let (release, transport) = try Self.release(in: scratch)
        let installer = Self.installer(scratch, transport: transport)
        let staged = try await installer.prepare(release)
        #expect(staged.version == SemanticVersion("0.9.9")!)
        // Prepared, not installed: the running app is untouched.
        #expect(try String(contentsOf: link, encoding: .utf8) == "old")

        let receipt = try installer.install(staged)
        #expect(receipt.previousCopy.lastPathComponent == "SteerLab-0.9.8-previous.app")
        #expect(LiveUpdateFileSystem().bundleInfo(at: scratch.installed).version == "0.9.9")
        #expect(LiveUpdateFileSystem().bundleInfo(at: receipt.previousCopy).version == "0.9.8")
        #expect(try String(contentsOf: link, encoding: .utf8) == "new 0.9.9")
        #expect(receipt.trashedCopies.map(\.lastPathComponent) == ["SteerLab-0.9.7-previous.app"])
        #expect(FileManager.default.fileExists(atPath: scratch.apps.appending(component: "SteerLab-0.9.6-previous.app").path))
        #expect(!FileManager.default.fileExists(atPath: installer.stagingRoot.path))
    }

    /// Each move of the swap fails in turn; every time, the original app is
    /// back at its path, no previous copy is left, and the staging is gone.
    @Test func everyFailedSwapStepPutsTheOriginalBack() async throws {
        enum Failing: CaseIterable { case parkPrevious, keepRunning, placeNew }
        for failing in Failing.allCases {
            let scratch = try Scratch()
            defer { scratch.remove() }
            try Self.makeBundle(at: scratch.installed, version: "0.9.8", marker: "old")
            let samePrevious = scratch.apps.appending(component: "SteerLab-0.9.8-previous.app")
            try Self.makeBundle(at: samePrevious, version: "0.9.8", marker: "kept earlier")
            let (release, transport) = try Self.release(in: scratch)
            let installed = scratch.installed
            var fileSystem = FaultyFileSystem()
            fileSystem.failMove = { source, destination in
                switch failing {
                case .parkPrevious: return destination.lastPathComponent == "replaced-previous.app"
                case .keepRunning: return source == installed
                case .placeNew:
                    return destination == installed
                        && source.path.contains(UpdateStagingLayout.folderName)
                }
            }
            let installer = Self.installer(scratch, transport: transport, fileSystem: fileSystem)
            let staged = try await installer.prepare(release)
            #expect {
                _ = try installer.install(staged)
            } throws: { error in
                guard case .installFailedRestored = error as? UpdateInstallError else { return false }
                return true
            }
            #expect(LiveUpdateFileSystem().bundleInfo(at: scratch.installed).version == "0.9.8", "\(failing)")
            #expect(
                try String(contentsOf: scratch.installed.appending(components: "Contents", "MacOS", "SteerLabApp"), encoding: .utf8)
                    == "old", "\(failing)")
            #expect(
                try String(contentsOf: samePrevious.appending(components: "Contents", "MacOS", "SteerLabApp"), encoding: .utf8)
                    == "kept earlier", "\(failing)")
            #expect(!FileManager.default.fileExists(atPath: installer.stagingRoot.path), "\(failing)")
        }
    }

    @Test func theStagedAppIsVerifiedAgainImmediatelyBeforeTheSwap() async throws {
        let scratch = try Scratch()
        defer { scratch.remove() }
        try Self.makeBundle(at: scratch.installed, version: "0.9.8", marker: "old")
        let (release, transport) = try Self.release(in: scratch)
        let signing = FakeSigning()
        let installer = Self.installer(scratch, transport: transport, signing: signing)
        let staged = try await installer.prepare(release)
        // Something replaced the staged app after it was checked.
        signing.accepting.withLock { $0 = false }
        #expect(throws: UpdateInstallError.signatureRejected("refused by the test")) {
            _ = try installer.install(staged)
        }
        #expect(signing.seen.withLock { $0.count } == 2)
        #expect(LiveUpdateFileSystem().bundleInfo(at: scratch.installed).version == "0.9.8")
        #expect(!FileManager.default.fileExists(atPath: installer.previousCopy.path))
    }

    @Test func aKeptCopyThatCannotGoToTheTrashIsKeptBesideTheApp() async throws {
        let scratch = try Scratch()
        defer { scratch.remove() }
        try Self.makeBundle(at: scratch.installed, version: "0.9.8", marker: "old")
        try Self.makeBundle(
            at: scratch.apps.appending(component: "SteerLab-0.9.8-previous.app"), version: "0.9.8",
            marker: "kept earlier")
        let (release, transport) = try Self.release(in: scratch)
        var fileSystem = FaultyFileSystem()
        fileSystem.failTrash = true
        let installer = Self.installer(scratch, transport: transport, fileSystem: fileSystem)
        let staged = try await installer.prepare(release)
        _ = try installer.install(staged)
        let rescued = scratch.apps.appending(component: "SteerLab-0.9.8-kept-1.app")
        #expect(
            try String(contentsOf: rescued.appending(components: "Contents", "MacOS", "SteerLabApp"), encoding: .utf8)
                == "kept earlier")
        #expect(!FileManager.default.fileExists(atPath: installer.stagingRoot.path))
    }

    @Test func aParkedCopyLeftInTheStagingFolderIsRescuedNotDeleted() throws {
        let scratch = try Scratch()
        defer { scratch.remove() }
        let root = UpdateStagingLayout.stagingRoot(forApp: scratch.installed)
        try Self.makeBundle(
            at: root.appending(component: UpdateStaging.parkedCopyName), version: "0.9.7", marker: "parked")
        let removed = UpdateStaging.removeLeftovers(besideApp: scratch.installed, isProcessAlive: { _ in false })
        #expect(removed)
        #expect(!FileManager.default.fileExists(atPath: root.path))
        #expect(
            FileManager.default.fileExists(
                atPath: scratch.apps.appending(component: "SteerLab-0.9.7-kept-1.app").path))
    }

    @Test func aNewAppThatIsNotTheVerifiedVersionIsRolledBack() async throws {
        let scratch = try Scratch()
        defer { scratch.remove() }
        try Self.makeBundle(at: scratch.installed, version: "0.9.8", marker: "old")
        let (release, transport) = try Self.release(in: scratch)
        let installer = Self.installer(scratch, transport: transport)
        let staged = try await installer.prepare(release)
        // Something rewrites the staged app between verification and the swap.
        try Self.makeBundle(at: staged.stagedApp, version: "0.9.5", marker: "swapped")
        #expect {
            _ = try installer.install(staged)
        } throws: { error in
            guard case .versionMismatch = error as? UpdateInstallError else { return false }
            return true
        }
        #expect(LiveUpdateFileSystem().bundleInfo(at: scratch.installed).version == "0.9.8")
        #expect(!FileManager.default.fileExists(atPath: installer.previousCopy.path))
    }

    @Test func aRollbackThatCannotFinishSaysWhereEachCopyIs() async throws {
        let scratch = try Scratch()
        defer { scratch.remove() }
        try Self.makeBundle(at: scratch.installed, version: "0.9.8", marker: "old")
        let (release, transport) = try Self.release(in: scratch)
        let installed = scratch.installed
        var fileSystem = FaultyFileSystem()
        // Placing the new app fails, and so does renaming the original back.
        fileSystem.failMove = { source, destination in
            destination == installed
        }
        let installer = Self.installer(scratch, transport: transport, fileSystem: fileSystem)
        let staged = try await installer.prepare(release)
        do {
            _ = try installer.install(staged)
            Issue.record("expected a failure")
        } catch let error as UpdateInstallError {
            guard case .installFailedNotRestored(let message) = error else {
                Issue.record("expected an unrestored failure, got \(error)")
                return
            }
            #expect(message.contains(installer.previousCopy.path))
            #expect(message.contains("rename it to SteerLab.app"))
        }
    }

    // MARK: - Staging left over from an earlier launch

    @Test func leftoverStagingIsRemovedOnTheNextLaunchUnlessItsOwnerLives() throws {
        let scratch = try Scratch()
        defer { scratch.remove() }
        try Self.makeBundle(at: scratch.installed, version: "0.9.8", marker: "old")
        let root = UpdateStagingLayout.stagingRoot(forApp: scratch.installed)
        try FileManager.default.createDirectory(at: root.appending(component: "unpacked"), withIntermediateDirectories: true)
        try Data("4242\n".utf8).write(to: root.appending(component: UpdateStagingLayout.ownerFileName))

        // Owner still alive (another copy of the app preparing an update).
        let whileAlive = UpdateStaging.removeLeftovers(
            besideApp: scratch.installed, currentProcess: 1, isProcessAlive: { $0 == 4242 })
        #expect(!whileAlive)
        // Owned by this very process (a second window opening mid-download).
        let ownedHere = UpdateStaging.removeLeftovers(
            besideApp: scratch.installed, currentProcess: 4242, isProcessAlive: { _ in false })
        #expect(!ownedHere)
        #expect(FileManager.default.fileExists(atPath: root.path))
        // Owner gone: removed.
        let afterExit = UpdateStaging.removeLeftovers(
            besideApp: scratch.installed, currentProcess: 1, isProcessAlive: { _ in false })
        #expect(afterExit)
        #expect(!FileManager.default.fileExists(atPath: root.path))
        // Nothing there: nothing to do.
        let nothing = UpdateStaging.removeLeftovers(besideApp: scratch.installed)
        #expect(!nothing)
        // The app itself is never touched.
        #expect(LiveUpdateFileSystem().bundleInfo(at: scratch.installed).version == "0.9.8")
    }

    /// A process number recorded by a copy that crashed may now belong to
    /// another program; only a running SteerLab keeps the staging folder.
    @Test func aRecycledProcessNumberDoesNotCountAsTheOwner() {
        #expect(!UpdateStaging.isProcessAlive(0))
        #expect(!UpdateStaging.isProcessAlive(Int32.max))
        // launchd is alive, and is not SteerLab.
        #expect(!UpdateStaging.isProcessAlive(1))
    }

    @Test func aLeftoverWithNoOwnerFileIsRemovedToo() throws {
        let scratch = try Scratch()
        defer { scratch.remove() }
        let root = UpdateStagingLayout.stagingRoot(forApp: scratch.installed)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let removed = UpdateStaging.removeLeftovers(besideApp: scratch.installed, isProcessAlive: { _ in true })
        #expect(removed)
    }

    // MARK: - Reopening, waiting, and the first launch after

    @Test func theRelaunchStepTakesItsArgumentsPositionally() {
        let app = URL(filePath: "/opt/lab/My Apps/SteerLab.app")
        let argv = UpdateRelaunch.waiterArguments(pid: 4321, app: app)
        #expect(argv.first == "/bin/sh")
        #expect(argv[1] == "-c")
        #expect(argv.suffix(2) == ["4321", "/opt/lab/My Apps/SteerLab.app"])
        #expect(!argv[2].contains("SteerLab.app"))  // never spliced into the script
        #expect(argv[2].contains("/usr/bin/open \"$app\""))
    }

    @Test func modelWorkAndOtherTasksKeepTheUpdateWaiting() {
        let busy = UpdateBusyState.from(
            modelLoading: true, modelDownloading: true, neutralBasisBuilding: true,
            otherTasks: ["an adapter is being trained"])
        #expect(!busy.isIdle)
        #expect(busy.reasons.contains("a model is loading"))
        #expect(busy.reasons.contains("a model is downloading"))
        #expect(busy.reasons.contains("a neutral basis is being built"))
        #expect(busy.reasons.contains("an adapter is being trained"))
    }

    /// Records whether each move ran on the main thread.
    final class ThreadRecordingFileSystem: UpdateFileSystem, @unchecked Sendable {
        let live = LiveUpdateFileSystem()
        let movesOnMain = Mutex<[Bool]>([])
        func exists(_ url: URL) -> Bool { live.exists(url) }
        func isDirectory(_ url: URL) -> Bool { live.isDirectory(url) }
        func createDirectory(_ url: URL) throws { try live.createDirectory(url) }
        func move(_ source: URL, to destination: URL) throws {
            movesOnMain.withLock { $0.append(pthread_main_np() != 0) }
            try live.move(source, to: destination)
        }
        func remove(_ url: URL) throws { try live.remove(url) }
        func trash(_ url: URL) throws { try live.remove(url) }
        func contents(of directory: URL) throws -> [URL] { try live.contents(of: directory) }
        func write(_ data: Data, to url: URL) throws { try live.write(data, to: url) }
        func read(_ url: URL) -> Data? { live.read(url) }
        func bundleInfo(at app: URL) -> UpdateBundleInfo { live.bundleInfo(at: app) }
    }

    /// The swap (and the re-verification before it) must not run on the
    /// main thread, or the sheet freezes while ~4,200 files are hashed.
    @MainActor @Test func theSwapRunsOffTheMainThread() async throws {
        let scratch = try Scratch()
        defer { scratch.remove() }
        try Self.makeBundle(at: scratch.installed, version: "0.9.8", marker: "old")
        let (release, transport) = try Self.release(in: scratch)
        let fileSystem = ThreadRecordingFileSystem()
        let installer = Self.installer(scratch, transport: transport, fileSystem: fileSystem)
        let staged = try await installer.prepare(release)
        let result = await installer.installDetached(staged)
        guard case .success(let receipt) = result else {
            Issue.record("expected an install, got \(result)")
            return
        }
        #expect(receipt.installedVersion == SemanticVersion("0.9.9")!)
        let moves = fileSystem.movesOnMain.withLock { $0 }
        #expect(!moves.isEmpty)
        #expect(!moves.contains(true))
    }

    @Test func busyWorkIsNamedInPlainWords() {
        #expect(UpdateBusyState.from().isIdle)
        let busy = UpdateBusyState.from(
            studyRunning: true, submissions: 1, engineJobs: 2, commandLineProcesses: 1)
        #expect(!busy.isIdle)
        #expect(busy.reasons.contains("a study is running on this Mac"))
        #expect(busy.reasons.contains("a study is being submitted to run"))
        #expect(busy.reasons.contains("the local Python engine is running 2 jobs"))
        #expect(busy.reasons.contains { $0.contains("steerlab-cli") })
    }

    @MainActor @Test func submissionsAreCountedWhileInFlight() {
        let work = UpdateBusyWork()
        work.beginSubmission()
        work.beginSubmission()
        work.endSubmission()
        #expect(work.submissions == 1)
        work.endSubmission()
        work.endSubmission()
        #expect(work.submissions == 0)
    }

    @Test func theFirstLaunchAfterAnUpdateSaysHowToGoBackAndWhatToRefresh() throws {
        let notice = try #require(
            UpdateAfterInstall.notice(
                lastLaunched: "0.9.9", current: SemanticVersion("1.0.0")!, appName: "SteerLab.app",
                previousCopyName: "SteerLab-0.9.9-previous.app", engineIsStale: true, hasClusterSites: true))
        #expect(notice.headline == "SteerLab was updated from 0.9.9 to 1.0.0.")
        #expect(notice.lines.count == 3)
        #expect(notice.lines[0].contains("rename SteerLab-0.9.9-previous.app to SteerLab.app"))
        #expect(notice.lines[1].contains("Refresh Engine"))
        #expect(notice.lines[2].contains("push the engine there again"))
        #expect(notice.offersEngineSetup)

        // With a clone beside the app, the engine runs from it: the notice
        // gives the commands for this release instead of Refresh Engine.
        let withClone = try #require(
            UpdateAfterInstall.notice(
                lastLaunched: "0.9.9", current: SemanticVersion("1.0.0")!, appName: "SteerLab.app",
                previousCopyName: nil, engineIsStale: false, hasClusterSites: false,
                sourceCheckout: URL(filePath: "/opt/lab/SteerLab/InterpBench")))
        #expect(withClone.lines.count == 1)
        #expect(withClone.lines[0].contains("git -C '/opt/lab/SteerLab/InterpBench' checkout v1.0.0"))
        #expect(!withClone.lines[0].contains("Refresh Engine"))

        // Same version, an older one, or no record: nothing to say.
        for last in ["1.0.0", "1.0.1", nil] as [String?] {
            #expect(
                UpdateAfterInstall.notice(
                    lastLaunched: last, current: SemanticVersion("1.0.0")!, appName: "SteerLab.app",
                    previousCopyName: nil, engineIsStale: true, hasClusterSites: true) == nil)
        }
        let quiet = try #require(
            UpdateAfterInstall.notice(
                lastLaunched: "0.9.9", current: SemanticVersion("1.0.0")!, appName: "SteerLab.app",
                previousCopyName: nil, engineIsStale: false, hasClusterSites: false))
        #expect(quiet.lines.isEmpty)
        #expect(!quiet.offersEngineSetup)
    }
}
