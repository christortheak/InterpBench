import Foundation
import Testing

@testable import ExperimentKit

/// `scripts/make-server-payload.sh` (Phase A wave 3): the packaging step
/// that turns a checkout into the immutable cluster deployment payload.
/// Contracts proven here, against a small fixture tree (never the real
/// `Server/`):
///  * the staging filter matches `ClusterProvisioner.pushFilterArguments` —
///    the payload is exactly what clusters already receive from a dev push
///    (Server/ incl. its tests, prompts/fixtures/; no venvs/caches/runs);
///  * the generated `deployment-manifest.json` is the `ResourceManifest`
///    schema — loadable by the Swift verifier, verifies clean, and its file
///    walk agrees with `ResourceManifest.generate`;
///  * two runs over the same tree produce byte-identical manifests;
///  * an existing output is immutable without `--force`.
struct ServerPayloadPackagingTests {

    private static let repoRoot = URL(filePath: #filePath)
        .deletingLastPathComponent()  // ServerPayloadPackagingTests.swift
        .deletingLastPathComponent()  // ExperimentKitTests
        .deletingLastPathComponent()  // Tests
    private static let script = repoRoot
        .appending(components: "scripts", "make-server-payload.sh")

    /// A miniature checkout exercising every filter rule; caller removes it.
    private func stageFixtureCheckout() throws -> URL {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appending(
            component: "steerlab-packaging-fixture-\(UUID().uuidString)")
        func stage(_ relative: String, _ contents: String) throws {
            let url = root.appending(path: relative)
            try fm.createDirectory(
                at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try contents.write(to: url, atomically: true, encoding: .utf8)
        }
        // Ships: the filtered Server/ tree (tests included — they ship from
        // a dev push today) and the parity fixtures.
        try stage("Server/pyproject.toml", "[project]\nname = \"steerlab-server\"\n")
        try stage("Server/steerlab_server/app.py", "print('app')\n")
        try stage("Server/scripts/bootstrap.sh", "#!/bin/sh\necho bootstrap\n")
        try stage("Server/tests/test_parity.py", "def test_parity(): pass\n")
        try stage("prompts/fixtures/golden.json", "{\"prompts\": []}\n")
        // Never ships: local envs, caches, run outputs, everything else.
        try stage("Server/.venv.nosync/lib/junk.txt", "junk\n")
        try stage("Server/steerlab_server/__pycache__/app.cpython-312.pyc", "junk\n")
        try stage("Server/steerlab.egg-info/PKG-INFO", "junk\n")
        try stage("Server/runs/2026-01-01-run/report.json", "{}\n")
        try stage("prompts/concepts/secret/positive.jsonl", "{\"text\": \"secret\"}\n")
        try stage("runs/2026-01-01-run/report.json", "{}\n")
        try stage("docs/notes.md", "not shipped\n")
        try stage("Package.swift", "// swift-tools-version: 6.2\n")
        return root
    }

    @discardableResult
    private func runScript(_ arguments: [String]) throws -> (status: Int32, output: String) {
        let process = Process()
        process.executableURL = URL(filePath: "/bin/bash")
        process.arguments = [Self.script.path] + arguments
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        return (process.terminationStatus, String(decoding: data, as: UTF8.self))
    }

    @Test func packagingMatchesThePushFilterAndVerifiesClean() throws {
        let fm = FileManager.default
        let fixture = try stageFixtureCheckout()
        defer { try? fm.removeItem(at: fixture) }
        let out = fixture.appending(component: "payload-out")
        let versionArguments = [
            "--app-version", "0.0-test", "--server-version", "0.0-test",
            "--protocol-version", "1",
        ]
        let run = try runScript(
            ["--source", fixture.path, "--output", out.path] + versionArguments)
        #expect(run.status == 0, "packaging failed: \(run.output)")

        // Filter parity with the live rsync push.
        let shipped = [
            "Server/pyproject.toml", "Server/steerlab_server/app.py",
            "Server/scripts/bootstrap.sh", "Server/tests/test_parity.py",
            "prompts/fixtures/golden.json",
        ]
        let excluded = [
            "Server/.venv.nosync", "Server/steerlab_server/__pycache__",
            "Server/steerlab.egg-info", "Server/runs",
            "prompts/concepts", "runs", "docs", "Package.swift",
        ]
        for path in shipped {
            #expect(
                fm.fileExists(atPath: out.appending(path: path).path),
                "payload is missing \(path)")
        }
        for path in excluded {
            #expect(
                !fm.fileExists(atPath: out.appending(path: path).path),
                "payload wrongly ships \(path)")
        }

        // The manifest is the ResourceManifest schema and verifies clean…
        let manifestURL = out.appending(
            component: ClusterProvisioner.deploymentManifestFileName)
        let manifest = try ResourceManifest.load(from: manifestURL)
        #expect(manifest.schemaVersion == ResourceManifest.currentSchemaVersion)
        #expect(manifest.appVersion == "0.0-test")
        #expect(manifest.serverVersion == "0.0-test")
        #expect(manifest.protocolVersion == 1)
        #expect(manifest.sourceRevision == nil)  // fixture is not a git repo
        #expect(manifest.verify(against: out).isEmpty)
        #expect(manifest.files.keys.sorted() == shipped.sorted())
        // …and the provisioner's pre-push gate accepts the fresh payload,
        // then names drift plainly.
        #expect(ClusterProvisioner.deploymentManifestFailure(atPayloadRoot: out.path) == nil)

        // The script's walk agrees with the sanctioned Swift generator
        // (which additionally sees the manifest file itself, written last).
        var swiftFiles = try ResourceManifest.generate(
            over: out, serverVersion: "0.0-test", protocolVersion: 1
        ).files
        swiftFiles.removeValue(
            forKey: ClusterProvisioner.deploymentManifestFileName)
        #expect(manifest.files == swiftFiles)

        // Determinism: a second run yields byte-identical manifests.
        let out2 = fixture.appending(component: "payload-out-2")
        let rerun = try runScript(
            ["--source", fixture.path, "--output", out2.path] + versionArguments)
        #expect(rerun.status == 0, "second packaging run failed: \(rerun.output)")
        let firstBytes = try Data(contentsOf: manifestURL)
        let secondBytes = try Data(
            contentsOf: out2.appending(
                component: ClusterProvisioner.deploymentManifestFileName))
        #expect(firstBytes == secondBytes)

        // Immutability: refusing to clobber an existing payload without
        // --force; drift after packaging is caught by the pre-push gate.
        let clobber = try runScript(
            ["--source", fixture.path, "--output", out.path] + versionArguments)
        #expect(clobber.status != 0)
        #expect(clobber.output.contains("--force"))
        try "drifted\n".write(
            to: out.appending(path: "Server/steerlab_server/app.py"),
            atomically: true, encoding: .utf8)
        let failure = ClusterProvisioner.deploymentManifestFailure(atPayloadRoot: out.path)
        #expect(failure?.contains("does not match its manifest") == true)
        #expect(failure?.contains("Server/steerlab_server/app.py") == true)
    }
}
