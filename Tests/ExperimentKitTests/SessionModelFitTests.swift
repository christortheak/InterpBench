import Foundation
import Testing

@testable import ExperimentKit

/// The client-side memory-fit rule (engineer reviews 2026-07-18): one pure
/// arithmetic core shared by every load surface, pinned here against the
/// exact live failure it exists to prevent.
struct SessionModelFitTests {

    /// The numbers from the live incident: gemma-3-12b-it is 22.7 GiB of
    /// bf16 weights; an "L4 24 GB" reports 22.05 GiB of usable CUDA memory.
    private let gemma12BBytes: Int64 = 24_375_986_749   // 22.7 GiB
    private let gemma4BBytes: Int64 = 8_589_934_592     // 8.0 GiB
    private let l4ActualBytes: Int64 = 23_672_599_552   // 22.05 GiB

    @Test func theExact12BOnL4CaseIsRefused() {
        let note = SessionModelFit.tooBigNote(
            modelBytes: gemma12BBytes, capacityBytes: l4ActualBytes)
        #expect(note != nil)
        #expect(note?.contains("22.0 GiB") == true)  // names the REAL capacity
        // The 4B fits the same GPU with room for context.
        #expect(SessionModelFit.tooBigNote(
            modelBytes: gemma4BBytes, capacityBytes: l4ActualBytes) == nil)
    }

    @Test func profileFallbackIsConservativeDecimalGB() {
        // Before the worker's first probe only the profile's marketed "24"
        // is known. Read as GiB it PASSED the 12B (23.7 < 24 — the round-6
        // bug); read conservatively as decimal bytes it refuses.
        let fallback = SessionModelFit.capacityBytes(
            actualBytes: nil, profileVRAMGB: 24)
        #expect(fallback == 24_000_000_000)
        #expect(SessionModelFit.tooBigNote(
            modelBytes: gemma12BBytes, capacityBytes: fallback) != nil)
        // The worker's actual report wins over the profile once present.
        #expect(SessionModelFit.capacityBytes(
            actualBytes: l4ActualBytes, profileVRAMGB: 24) == l4ActualBytes)
    }

    @Test func absentFactsNeverGate() {
        // No size, no capacity, or no session: nothing is refused — a
        // missing fact is not evidence of unfitness (the server preflight
        // stays authoritative at load time).
        #expect(SessionModelFit.tooBigNote(
            modelBytes: nil, capacityBytes: l4ActualBytes) == nil)
        #expect(SessionModelFit.tooBigNote(
            modelBytes: gemma12BBytes, capacityBytes: nil) == nil)
        #expect(SessionModelFit.capacityBytes(
            actualBytes: nil, profileVRAMGB: nil) == nil)
    }

    @Test func headroomMirrorsTheServerPreflight() {
        // Weights alone never run: exactly-capacity is refused, capacity
        // minus (weights + 1 GiB floor) passes at the boundary.
        let capacity: Int64 = 10 << 30
        #expect(SessionModelFit.tooBigNote(
            modelBytes: capacity, capacityBytes: capacity) != nil)
        let fitting = capacity - SessionModelFit.headroomBytes - 1
        #expect(SessionModelFit.tooBigNote(
            modelBytes: fitting, capacityBytes: capacity) == nil)
    }
}
