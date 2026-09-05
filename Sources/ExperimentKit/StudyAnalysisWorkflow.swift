import Foundation

/// Coordinates admitted evidence -> value-only calculation -> report publication.
/// Manifest verification stays at the existing task admission boundary.
enum StudyAnalysisWorkflow {
    static func analyze(
        manifest: ExperimentManifest, repository: StudyAnalysisRepository,
        allowUnverifiedEpoch: Bool
    ) throws -> URL {
        let input = try repository.loadAnalysis(
            manifest: manifest, allowUnverifiedEpoch: allowUnverifiedEpoch)
        let result = try StudyAnalysisCalculator.analyze(input)
        let artifacts = try StudyAnalysisRendering.analyze(input: input, result: result)
        let directory = try StudyAnalysisWriter(workspaceRoot: repository.workspaceRoot)
            .write(artifacts, manifest: manifest, task: .analyze)
        emit(artifacts.diagnostics)
        print("analysis artifacts: \(directory.path)")
        return directory
    }

    static func rescoreStyle(
        manifest: ExperimentManifest, repository: StudyAnalysisRepository,
        runDirectoryName: String?, allowUnverifiedEpoch: Bool
    ) throws -> URL {
        let input = try repository.loadRescore(
            manifest: manifest, runDirectoryName: runDirectoryName,
            allowUnverifiedEpoch: allowUnverifiedEpoch)
        let result = try StudyAnalysisCalculator.rescoreStyle(input)
        let artifacts = try StudyAnalysisRendering.rescoreStyle(input: input, result: result)
        let directory = try StudyAnalysisWriter(workspaceRoot: repository.workspaceRoot)
            .write(artifacts, manifest: manifest, task: .rescoreStyle)
        emit(artifacts.diagnostics)
        print("rescore artifacts: \(directory.path)")
        return directory
    }

    private static func emit(_ diagnostics: [StudyAnalysisDiagnostic]) {
        for diagnostic in diagnostics {
            if diagnostic.standardError {
                FileHandle.standardError.write(Data((diagnostic.text + "\n").utf8))
            } else {
                print(diagnostic.text)
            }
        }
    }
}
