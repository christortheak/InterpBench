import Testing
@testable import SteeringKit

struct NeutralBuildCancellationTests {
    @Test func cancelledBuildStopsBeforeEstimatingComponents() async throws {
        let bank = NeutralActivationBank(layers: [0], rowsByLayer: [[[1, 0], [0, 1], [-1, 0], [0, -1]]],
            residualNormPerLayer: [1],
            screening: .init(readingPosition: .lastToken, sourceCount: 4, includedCount: 4, excludedShortCount: 0),
            tokenRowCount: 4, sourceRowsPerLayer: 4, usedRowsPerLayer: 4, downsampleSeed: nil)
        let operation = Task.detached {
            withUnsafeCurrentTask { $0?.cancel() }
            return try bank.componentsByLayer(selection: .explainedVariance(0.5, maximumCount: 2))
        }
        do {
            _ = try await operation.value
            Issue.record("A cancelled build estimated components")
        } catch is CancellationError {
            // Cancellation precedes numerical work and artifact publication.
        }
    }
}
