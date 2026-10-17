import Foundation
import Testing

@testable import ExperimentKit

/// The install layout and its integrity stamp (WP0-AGENT-SURFACE-AUDIT §6,
/// §7 step 12).
///
/// What these tests are for: `scripts/tests/install-cli-test.sh` proves the
/// end-to-end story on a real install — layout, stripped rpath, a Metal kernel
/// with no `DYLD_FRAMEWORK_PATH` — but it costs a 150 MB copy and a GPU, so it
/// is not something the Swift suite can run. These cover the ENGINE half in
/// milliseconds over a staged directory: what the provenance reports, what the
/// stamp contains, and — the case that matters most — that a modified payload
/// is actually caught. `ResourceManifest.verify(against:)` passing means
/// nothing unless something proves it can fail.
struct InstallProvenanceTests {

    /// A directory shaped like an install root: a stand-in executable, the
    /// colocated shader library, and one resource bundle.
    static func stagedInstall() throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "install-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        try "not really a Mach-O".write(
            to: root.appending(component: "steerlab-cli"),
            atomically: true, encoding: .utf8)
        try "not really a metallib".write(
            to: root.appending(component: InstallProvenance.metallibName),
            atomically: true, encoding: .utf8)
        let bundle = root.appending(
            components: "swift-transformers_Hub.bundle", "Contents", "Resources")
        try FileManager.default.createDirectory(
            at: bundle, withIntermediateDirectories: true)
        try "{}".write(
            to: bundle.appending(component: "gpt2_tokenizer_config.json"),
            atomically: true, encoding: .utf8)
        return root
    }

    static func provenance(of root: URL) -> InstallProvenance {
        InstallProvenance.resolve(
            executable: root.appending(component: "steerlab-cli"),
            environment: [:])
    }

    // MARK: Shape

    @Test func anUnstampedTreeIsABuildProductAndSaysSo() throws {
        let root = try Self.stagedInstall()
        defer { try? FileManager.default.removeItem(at: root) }

        let provenance = Self.provenance(of: root)
        #expect(provenance.shape == .buildProduct)
        #expect(provenance.root == root.standardizedFileURL)
        #expect(provenance.report.contains("build product (unstamped)"))
        #expect(provenance.report.contains("never stamped"))
        // Absence is reported as absence, not as an empty string.
        #expect(provenance.payload["manifestStamped"] == .bool(false))
        #expect(provenance.payload["manifest"] == nil)
    }

    @Test func stampingMakesItAnInstall() throws {
        let root = try Self.stagedInstall()
        defer { try? FileManager.default.removeItem(at: root) }

        let manifest = try InstallProvenance.stamp(
            root: root, sourceRevision: "abcd1234")
        // Three files: the executable, the shader library, and the one
        // resource inside the staged bundle. The manifest never lists ITSELF,
        // which is what makes a re-stamp idempotent.
        #expect(manifest.files.count == 3)
        #expect(manifest.files[InstallProvenance.manifestName] == nil)
        #expect(manifest.serverVersion == InstallProvenance.unbundledServerVersion)

        let provenance = Self.provenance(of: root)
        #expect(provenance.shape == .installed)
        #expect(provenance.manifest?.sourceRevision == "abcd1234")
        #expect(provenance.report.contains("resource-manifest.json: 3 file(s)"))
        // …and it names the check rather than performing it: `--version` must
        // stay instant on a 147 MB executable.
        #expect(provenance.report.contains("steerlab-cli install verify"))
        #expect(try provenance.verifyInstalledTree().isEmpty)
    }

    @Test func reStampingIsIdempotent() throws {
        let root = try Self.stagedInstall()
        defer { try? FileManager.default.removeItem(at: root) }

        let first = try InstallProvenance.stamp(root: root, sourceRevision: "aaaa")
        let second = try InstallProvenance.stamp(root: root, sourceRevision: "aaaa")
        #expect(first.files == second.files)
        #expect(try Self.provenance(of: root).verifyInstalledTree().isEmpty)
    }

    // MARK: The stamp has teeth

    @Test func aModifiedPayloadFileIsAMismatch() throws {
        let root = try Self.stagedInstall()
        defer { try? FileManager.default.removeItem(at: root) }
        try InstallProvenance.stamp(root: root)

        try "tampered".write(
            to: root.appending(component: InstallProvenance.metallibName),
            atomically: true, encoding: .utf8)
        let problems = try Self.provenance(of: root).verifyInstalledTree()
        #expect(problems.count == 1)
        #expect(problems.first?.kind == .mismatch)
        #expect(problems.first?.path == InstallProvenance.metallibName)
    }

    @Test func aRemovedPayloadFileIsMissing() throws {
        let root = try Self.stagedInstall()
        defer { try? FileManager.default.removeItem(at: root) }
        try InstallProvenance.stamp(root: root)

        try FileManager.default.removeItem(
            at: root.appending(component: InstallProvenance.metallibName))
        let problems = try Self.provenance(of: root).verifyInstalledTree()
        #expect(problems.first?.kind == .missing)
    }

    @Test func anExtraSiblingIsNotAProblem() throws {
        let root = try Self.stagedInstall()
        defer { try? FileManager.default.removeItem(at: root) }
        try InstallProvenance.stamp(root: root)

        // A bundle legitimately carries siblings the manifest never listed —
        // a log, a `.DS_Store`, a file a later step added. Flagging those
        // would make the check cry wolf and get ignored.
        try "note".write(
            to: root.appending(component: "NOTES.txt"),
            atomically: true, encoding: .utf8)
        #expect(try Self.provenance(of: root).verifyInstalledTree().isEmpty)
    }

    @Test func verifyingAnUnstampedTreeRefusesRatherThanPassing() throws {
        let root = try Self.stagedInstall()
        defer { try? FileManager.default.removeItem(at: root) }
        #expect(throws: (any Error).self) {
            try Self.provenance(of: root).verifyInstalledTree()
        }
    }

    // MARK: Guards

    @Test func stampRefusesABuildDirectory() throws {
        let root = try Self.stagedInstall()
        defer { try? FileManager.default.removeItem(at: root) }
        // The accident this prevents: pointing `install stamp` at DerivedData,
        // which would hash a gigabyte of object files AND leave a
        // `resource-manifest.json` there that `SteerLabVersion.current` would
        // then believe.
        try "".write(
            to: root.appending(component: "ExperimentKit.o"),
            atomically: true, encoding: .utf8)
        #expect(throws: (any Error).self) {
            try InstallProvenance.stamp(root: root)
        }
    }

    @Test func theShimIsReportedOnlyWhenItPointsHere() throws {
        // Lay out the §6.2 shape: <prefix>/libexec/steerlab + <prefix>/bin.
        let prefix = FileManager.default.temporaryDirectory
            .appending(component: "prefix-\(UUID().uuidString)")
        let libexec = prefix.appending(components: "libexec", "steerlab")
        let bin = prefix.appending(component: "bin")
        try FileManager.default.createDirectory(
            at: libexec, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: bin, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: prefix) }
        try "binary".write(
            to: libexec.appending(component: "steerlab-cli"),
            atomically: true, encoding: .utf8)

        let executable = libexec.appending(component: "steerlab-cli")
        var provenance = InstallProvenance.resolve(
            executable: executable, environment: [:])
        #expect(provenance.shim == nil, "no shim exists yet")

        // A shim for something else must not be claimed: a reported shim that
        // runs a different binary is worse than none.
        try "#!/bin/zsh\nexec /usr/bin/true\n".write(
            to: bin.appending(component: "steerlab"),
            atomically: true, encoding: .utf8)
        provenance = InstallProvenance.resolve(
            executable: executable, environment: [:])
        #expect(provenance.shim == nil)

        try "#!/bin/zsh\nexec \"${0:A:h}/../libexec/steerlab/steerlab-cli\" \"$@\"\n"
            .write(
                to: bin.appending(component: "steerlab"),
                atomically: true, encoding: .utf8)
        provenance = InstallProvenance.resolve(
            executable: executable, environment: [:])
        #expect(provenance.shim?.lastPathComponent == "steerlab")
    }

    // MARK: The claim under test

    @Test func metallibColocationIsNotSatisfiedByTheDeveloperEnvironment() throws {
        let root = try Self.stagedInstall()
        defer { try? FileManager.default.removeItem(at: root) }

        // The audit's §6.1 claim is about the COLOCATED copy. A property that
        // answered "yes" because `DYLD_FRAMEWORK_PATH` happened to be
        // inherited would test the developer's shell, not the install.
        try FileManager.default.removeItem(
            at: root.appending(component: InstallProvenance.metallibName))
        let withDyld = InstallProvenance.resolve(
            executable: root.appending(component: "steerlab-cli"),
            environment: ["DYLD_FRAMEWORK_PATH": "/somewhere/Products/Debug"])
        #expect(withDyld.metallibColocated == false)
        #expect(withDyld.payload["dyldFrameworkPath"] != nil)
        #expect(withDyld.report.contains("relying on DYLD_FRAMEWORK_PATH"))
    }

    @Test func withNeitherShaderSourceTheReportSaysGPUVerbsWillFail() throws {
        let root = try Self.stagedInstall()
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.removeItem(
            at: root.appending(component: InstallProvenance.metallibName))
        let provenance = Self.provenance(of: root)
        #expect(provenance.report.contains("GPU verbs will refuse to load shaders"))
    }

    // MARK: The command line bundled in the app

    /// `<tmp>/SteerLab.app` laid out the way `scripts/build-app.sh` lays it
    /// out: the helper and its shader library in `Contents/Helpers/`, the
    /// build identity in `Contents/Info.plist`, and the app's resource
    /// manifest in `Contents/Resources/`. Nothing is beside the helper.
    static func stagedApp(writeManifest: Bool = true) throws -> (parent: URL, helper: URL) {
        let fm = FileManager.default
        let parent = fm.temporaryDirectory.appending(component: "app-\(UUID().uuidString)")
        let contents = parent.appending(components: "SteerLab.app", "Contents")
        let helpers = contents.appending(component: "Helpers")
        let resources = contents.appending(component: "Resources")
        try fm.createDirectory(at: helpers, withIntermediateDirectories: true)
        try fm.createDirectory(
            at: resources.appending(component: "web"), withIntermediateDirectories: true)
        let helper = helpers.appending(component: "steerlab-cli")
        try "not really a Mach-O".write(to: helper, atomically: true, encoding: .utf8)
        try "not really a metallib".write(
            to: helpers.appending(component: InstallProvenance.metallibName),
            atomically: true, encoding: .utf8)
        let info: [String: Any] = [
            "CFBundleShortVersionString": "0.9.8",
            "SLFullVersionString": "0.9.8",
            "SLSourceRevision": "1a2b3c4d",
        ]
        try PropertyListSerialization.data(fromPropertyList: info, format: .xml, options: 0)
            .write(to: contents.appending(component: "Info.plist"))
        try "<!doctype html>".write(
            to: resources.appending(components: "web", "index.html"),
            atomically: true, encoding: .utf8)
        try "{}".write(
            to: resources.appending(component: "deployment-manifest.json"),
            atomically: true, encoding: .utf8)
        if writeManifest {
            let manifest = try ResourceManifest.generate(
                over: resources, appVersion: "0.9.8", serverVersion: "bundled",
                protocolVersion: 1, sourceRevision: "1a2b3c4d")
            try manifest.write(
                to: resources.appending(component: InstallProvenance.manifestName))
        }
        return (parent, helper)
    }

    @Test func theBundledCommandLineReportsTheAppRatherThanAMissingStamp() throws {
        let (parent, helper) = try Self.stagedApp()
        defer { try? FileManager.default.removeItem(at: parent) }

        let provenance = InstallProvenance.resolve(executable: helper, environment: [:])
        #expect(provenance.shape == .appBundle)
        #expect(provenance.app?.bundle.lastPathComponent == "SteerLab.app")
        #expect(provenance.app?.identity == "0.9.8, build 1a2b3c4d")
        #expect(provenance.app?.manifest?.files.count == 2)

        let report = provenance.report
        #expect(report.contains("layout        the command line bundled in SteerLab.app"))
        #expect(report.contains("app           0.9.8, build 1a2b3c4d — "))
        #expect(report.contains(
            "manifest      the app's resource-manifest.json: 2 file(s), app 0.9.8+1a2b3c4d"))
        #expect(report.contains("codesign --verify --deep --strict"))
        // What made a healthy helper look broken is gone.
        #expect(!report.contains("build product (unstamped)"))
        #expect(!report.contains("never stamped"))
        // The families line is untouched (documents count this phrase).
        #expect(report.contains("resource families resolved"))

        let payload = provenance.payload
        #expect(payload["layout"] == .string("appBundle"))
        // `manifest` still means a stamp beside the binary; the app's is apart.
        #expect(payload["manifestStamped"] == .bool(false))
        #expect(payload["manifest"] == nil)
        guard case .object(let app)? = payload["app"] else {
            Issue.record("the payload has no app record")
            return
        }
        #expect(app["version"] == .string("0.9.8"))
        #expect(app["sourceRevision"] == .string("1a2b3c4d"))
        guard case .object(let appManifest)? = app["manifest"] else {
            Issue.record("the app record has no manifest")
            return
        }
        #expect(appManifest["fileCount"] == .number(2))
        #expect(appManifest["serverVersion"] == .string("bundled"))
    }

    @Test func aBundledCommandLineWithoutTheAppManifestSaysWhatToDo() throws {
        let (parent, helper) = try Self.stagedApp(writeManifest: false)
        defer { try? FileManager.default.removeItem(at: parent) }
        let provenance = InstallProvenance.resolve(executable: helper, environment: [:])
        #expect(provenance.shape == .appBundle)
        #expect(provenance.report.contains("is missing or unreadable"))
        #expect(provenance.report.contains("reinstall the app"))
    }

    @Test func verifyingTheBundledCommandLinePointsAtTheCodeSignature() throws {
        let (parent, helper) = try Self.stagedApp()
        defer { try? FileManager.default.removeItem(at: parent) }
        let provenance = InstallProvenance.resolve(executable: helper, environment: [:])
        do {
            _ = try provenance.verifyInstalledTree()
            Issue.record("verify found a stamp that is not there")
        } catch {
            #expect(String(describing: error).contains("codesign --verify --deep --strict"))
        }
    }

    @Test func onlyAnAppDirectoryMakesTheBundledLayout() throws {
        // `Contents/Helpers/` under a directory that is not an `.app`.
        let fm = FileManager.default
        let parent = fm.temporaryDirectory.appending(component: "notapp-\(UUID().uuidString)")
        let helpers = parent.appending(components: "SteerLab", "Contents", "Helpers")
        try fm.createDirectory(at: helpers, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: parent) }
        let helper = helpers.appending(component: "steerlab-cli")
        try "binary".write(to: helper, atomically: true, encoding: .utf8)
        let provenance = InstallProvenance.resolve(executable: helper, environment: [:])
        #expect(provenance.shape == .buildProduct)
        #expect(provenance.app == nil)
    }

    @Test func aStampBesideTheHelperIsStillTheInstallsOwnRecord() throws {
        let (parent, helper) = try Self.stagedApp()
        defer { try? FileManager.default.removeItem(at: parent) }
        try InstallProvenance.stamp(root: helper.deletingLastPathComponent())
        let provenance = InstallProvenance.resolve(executable: helper, environment: [:])
        #expect(provenance.shape == .installed)
        #expect(provenance.app == nil)
    }

    // MARK: `--version` is a rewrite, not a second implementation

    @Test func theBinaryRewritesTheVersionFlagIntoTheVerb() throws {
        #expect(ExperimentCLIRunner.versionFlag == "--version")
        #expect(ExperimentCLIRunner.namespaces.contains("install"))
        let main = try String(
            contentsOf: CodeResources.compiledCheckoutPath
                .appending(path: "Sources/steerlab-cli/main.swift"),
            encoding: .utf8)
        // The rewrite, not a duplicated report: everything `--version` prints,
        // its envelope, its `--json` mode, and its `--help` page have to come
        // from the one verb, or the two drift.
        #expect(main.contains("ExperimentCLIRunner.versionFlag"))
        #expect(main.contains(#"with: ["install", "version"]"#))
        #expect(!main.contains("SteerLabVersion.current"))
    }

    // MARK: The document's key set

    @Test func thePayloadCarriesTheKeysAnAgentReads() throws {
        let root = try Self.stagedInstall()
        defer { try? FileManager.default.removeItem(at: root) }
        try InstallProvenance.stamp(root: root, sourceRevision: "beefcafe")

        let payload = Self.provenance(of: root).payload
        for key in [
            "version", "layout", "executable", "installRoot",
            "metallibColocated", "resourceMode", "manifestStamped",
            "protocolVersion", "metallib", "manifest",
        ] {
            #expect(payload[key] != nil, "the version payload omits '\(key)'")
        }
        guard case .object(let stamp)? = payload["manifest"] else {
            Issue.record("manifest is not an object")
            return
        }
        #expect(stamp["sourceRevision"] == .string("beefcafe"))
        #expect(stamp["serverVersion"]
            == .string(InstallProvenance.unbundledServerVersion))
    }
}
