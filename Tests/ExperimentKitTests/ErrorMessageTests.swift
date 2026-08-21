import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// Errors must survive `localizedDescription`.
///
/// Swift does not route `Error.localizedDescription` through
/// `CustomStringConvertible`, so a self-describing error still renders as
/// "The operation couldn't be completed. (ExperimentKit.ChatServiceError
/// error 1.)" wherever a caller uses it — SwiftUI, Foundation, and our own
/// catch-all handlers all do. A researcher saw exactly that instead of the
/// reason a cluster submission failed (2026-07-26).
struct ErrorMessageTests {

    @Test func experimentErrorsCarryTheirReasonThroughLocalizedDescription() {
        let error: any Error = ExperimentError(reason: "the pinned corpus moved")
        #expect(error.localizedDescription == "the pinned corpus moved")
        #expect(!error.localizedDescription.contains("couldn’t be completed"))
    }

    @Test func chatServiceErrorsCarryTheirReasonToo() {
        let error: any Error = ChatServiceError(reason: "select a server model first")
        #expect(error.localizedDescription == "select a server model first")
    }

    /// The catch-all that lost the message renders `localizedDescription`
    /// on an arbitrary `any Error`, which is where the conformance has to
    /// take effect — not on the concrete type.
    @Test func theConformanceAppliesThroughTheExistentialAsCallersSeeIt() {
        func rendered(_ body: () throws -> Void) -> String {
            do { try body(); return "" } catch { return error.localizedDescription }
        }
        #expect(
            rendered { throw ExperimentError(reason: "no pinned revision") }
                == "no pinned revision")
        // Across the module boundary too: SteeringKit declares its own.
        let steering = rendered {
            throw ConceptExtractorError.layerMismatch
        }
        #expect(!steering.isEmpty)
        #expect(!steering.contains("couldn’t be completed"))
    }
}
