import Foundation
import SteeringKit

public struct GemmaScopePreparedAnalysis: Codable, Sendable {
    public let directory: String
    public let jobFile: String
    public let vectorFile: String
    public let reportFile: String
    public let command: String
    public let summary: String
}

public struct GemmaScopeRunResult: Codable, Sendable {
    public let prepared: GemmaScopePreparedAnalysis
    public let exitCode: Int32
    public let stdout: String
    public let stderr: String
    public let report: GemmaScopeFeatureReport?

    public var succeeded: Bool { exitCode == 0 && report != nil }

    public var summary: String {
        if succeeded {
            return "Gemma Scope analysis complete for \(report?.vector.concept ?? "vector")."
        }
        return "Gemma Scope analysis failed with exit code \(exitCode)."
    }
}

public enum GemmaScopeAnalysisError: Error, CustomStringConvertible {
    case unsupportedModel(String)
    case layerOutOfRange(Int)
    case pythonLaunchFailed(String)
    /// No code checkout to run the analysis venv from — carries the layout
    /// sentence from `CodeResources.ExecutableCheckoutUnavailable`, so the
    /// user reads a remedy rather than a missing path.
    case engineUnavailable(String)

    public var description: String {
        switch self {
        case .unsupportedModel(let modelID):
            "Gemma Scope 2 is available for Gemma 3 models, not \(modelID)"
        case .layerOutOfRange(let layer):
            "layer \(layer) is outside this vector artifact"
        case .pythonLaunchFailed(let reason):
            "could not start Python Gemma Scope analysis: \(reason)"
        case .engineUnavailable(let reason):
            "Gemma Scope analysis needs the code checkout beside the app — "
                + reason
        }
    }
}

public enum GemmaScopeAnalysis {
    struct VectorExport: Codable {
        let concept: String
        let artifactName: String
        let artifactDirectory: String
        let modelID: String
        let modelRevision: String?
        let layer: Int
        let hiddenSize: Int
        let norm: Float
        let values: [Float]
    }

    struct Job: Codable {
        let createdAt: String
        let vectorFile: String
        let reportFile: String
        let topK: Int
        let gemmaScope: GemmaScopeInfo
        let artifactSidecar: SteeringVectorSidecar
    }

    public static func prepare(
        artifact: VectorArtifact,
        layer: Int,
        info: GemmaScopeInfo,
        topK: Int = 25
    ) throws -> GemmaScopePreparedAnalysis {
        guard artifact.sidecar.modelID.lowercased().contains("gemma-3") else {
            throw GemmaScopeAnalysisError.unsupportedModel(artifact.sidecar.modelID)
        }

        let loaded = try SteeringVectorStore.load(from: artifact.directory, name: artifact.name)
        guard loaded.vectors.perLayer.indices.contains(layer) else {
            throw GemmaScopeAnalysisError.layerOutOfRange(layer)
        }

        let directory = try VectorCatalog.makeUniqueRunDirectory(
            slug: "gemmascope-\(slug(artifact.sidecar.concept))-l\(layer)")
        try RunMetadata.write(
            runType: "gemmascope-analysis", to: directory,
            modelID: artifact.sidecar.modelID, revision: artifact.sidecar.revision)
        let vectorURL = directory.appending(component: "steerlab-vector.json")
        let jobURL = directory.appending(component: "gemmascope-job.json")
        let reportURL = directory.appending(component: "gemmascope-report.json")
        let runnerURL = directory.appending(component: "run-gemmascope.sh")
        let scriptURL = try analysisScriptURL()

        let vector = loaded.vectors.perLayer[layer]
        let export = VectorExport(
            concept: artifact.sidecar.concept,
            artifactName: artifact.name,
            artifactDirectory: artifact.directory.path,
            modelID: artifact.sidecar.modelID,
            modelRevision: artifact.sidecar.revision,
            layer: layer,
            hiddenSize: artifact.sidecar.hiddenSize,
            norm: loaded.vectors.norm(at: layer),
            values: vector)

        let job = Job(
            createdAt: ISO8601DateFormatter().string(from: Date()),
            vectorFile: vectorURL.path,
            reportFile: reportURL.path,
            topK: topK,
            gemmaScope: info,
            artifactSidecar: artifact.sidecar)

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(export).write(to: vectorURL)
        try encoder.encode(job).write(to: jobURL)

        let command =
            "\(shellQuote(try defaultPythonExecutable())) \(shellQuote(scriptURL.path)) --job \(shellQuote(jobURL.path))"
        let runner = """
        #!/bin/sh
        set -eu
        \(command)
        """
        try runner.write(to: runnerURL, atomically: true, encoding: .utf8)

        return GemmaScopePreparedAnalysis(
            directory: directory.path,
            jobFile: jobURL.path,
            vectorFile: vectorURL.path,
            reportFile: reportURL.path,
            command: command,
            summary:
                "Prepared Gemma Scope analysis for \(artifact.sidecar.concept) at layer \(layer).")
    }

    public static func run(
        artifact: VectorArtifact,
        layer: Int,
        info: GemmaScopeInfo,
        pythonExecutable: String? = nil,
        topK: Int = 25
    ) async throws -> GemmaScopeRunResult {
        let prepared = try prepare(artifact: artifact, layer: layer, info: info, topK: topK)
        let scriptURL = try analysisScriptURL()
        let executable: String
        if let pythonExecutable {
            executable = pythonExecutable
        } else {
            executable = try defaultPythonExecutable()
        }
        let output = try await runProcess(
            executable: executable,
            arguments: [scriptURL.path, "--job", prepared.jobFile])

        let reportURL = URL(filePath: prepared.reportFile)
        let report =
            (try? Data(contentsOf: reportURL))
            .flatMap { try? JSONDecoder().decode(GemmaScopeFeatureReport.self, from: $0) }
        return GemmaScopeRunResult(
            prepared: prepared,
            exitCode: output.exitCode,
            stdout: output.stdout,
            stderr: output.stderr,
            report: report)
    }

    private struct ProcessOutput: Sendable {
        let exitCode: Int32
        let stdout: String
        let stderr: String
    }

    private static func runProcess(
        executable: String,
        arguments: [String]
    ) async throws -> ProcessOutput {
        try await Task.detached(priority: .userInitiated) {
            let process = Process()
            process.executableURL = executable.contains("/")
                ? URL(filePath: executable)
                : URL(filePath: "/usr/bin/env")
            process.arguments = executable.contains("/") ? arguments : [executable] + arguments
            process.currentDirectoryURL = VectorCatalog.projectRoot

            let stdoutPipe = Pipe()
            let stderrPipe = Pipe()
            process.standardOutput = stdoutPipe
            process.standardError = stderrPipe

            do {
                try process.run()
            } catch {
                throw GemmaScopeAnalysisError.pythonLaunchFailed("\(error)")
            }
            process.waitUntilExit()

            let stdout = String(
                decoding: stdoutPipe.fileHandleForReading.readDataToEndOfFile(),
                as: UTF8.self)
            let stderr = String(
                decoding: stderrPipe.fileHandleForReading.readDataToEndOfFile(),
                as: UTF8.self)
            return ProcessOutput(
                exitCode: process.terminationStatus,
                stdout: stdout,
                stderr: stderr)
        }.value
    }

    /// The analysis script ships with the CODE (the `analysisTools`
    /// family — `scripts/` in a checkout), never the data workspace.
    private static func analysisScriptURL() throws -> URL {
        try CodeResources.analysisTools()
            .appending(component: "gemmascope_analyze.py")
    }

    /// The dedicated analysis venv lives INSIDE a code checkout
    /// (`.venv.nosync/gemmascope-py312`) — bucket B (WP2): a venv is a
    /// mutable tree, so it resolves through `executableCheckout()` and never
    /// through a shipped family. A checkout that has one uses it; a checkout
    /// without one falls back to `python3`, exactly as before. With no
    /// checkout at all this refuses honestly and names the layout — a bare
    /// `python3` on an end-user machine would not carry the analysis
    /// dependencies.
    private static func defaultPythonExecutable() throws -> String {
        let checkout: URL
        switch CodeResources.executableCheckout() {
        case .success(let resolved): checkout = resolved.root
        case .failure(let absence):
            throw GemmaScopeAnalysisError.engineUnavailable(absence.message)
        }
        let local = checkout.appending(
            components: ".venv.nosync", "gemmascope-py312", "bin", "python")
        if FileManager.default.isExecutableFile(atPath: local.path) {
            return local.path
        }
        return "python3"
    }

    private static func slug(_ value: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        let scalars = value.unicodeScalars.map { scalar in
            allowed.contains(scalar) ? Character(scalar) : "-"
        }
        let collapsed = String(scalars)
            .split(separator: "-", omittingEmptySubsequences: true)
            .joined(separator: "-")
            .lowercased()
        return collapsed.isEmpty ? "vector" : collapsed
    }

    private static func shellQuote(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}
