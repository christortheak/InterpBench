import Foundation

/// Layer restrictions for capture-heavy work.
///
/// The project's standing convention is that the middle third of the network
/// is the steering sweet spot (CLAUDE.md › Steering methods). It is also the
/// cheap default for the neutral token bank: an all-layer capture costs
/// `rows × layers × hidden` float32, and the band cuts that by ~3×.
public enum LayerBand {

    /// The middle third of `blockCount` transformer blocks. Degenerate block
    /// counts (< 3) return every block — there is no meaningful band to take.
    public static func middleThird(ofBlockCount blockCount: Int) -> Set<Int> {
        guard blockCount > 0 else { return [] }
        guard blockCount >= 3 else { return Set(0 ..< blockCount) }
        let lower = blockCount / 3
        let upper = (2 * blockCount) / 3
        return Set(lower ..< max(upper, lower + 1))
    }

    /// Human-readable description of a captured band, for artifact sidecars
    /// and status lines: "middle-third band 16–31 of 48", "all 48 layers".
    public static func description(of layers: Set<Int>, blockCount: Int) -> String {
        guard !layers.isEmpty else { return "no layers" }
        if layers.count == blockCount { return "all \(blockCount) layers" }
        let sorted = layers.sorted()
        let contiguous = sorted.last! - sorted.first! + 1 == sorted.count
        let band = contiguous ? "\(sorted.first!)–\(sorted.last!)" : "\(sorted.count) selected"
        let isMiddleThird = layers == middleThird(ofBlockCount: blockCount)
        let name = isMiddleThird ? "middle-third band" : "band"
        return "\(name) \(band) of \(blockCount)"
    }
}

/// The projected cost of one neutral token bank, computed BEFORE any forward
/// pass runs.
public struct NeutralBankPreflight: Sendable, Equatable {
    /// Token rows the corpus offers per layer, before the cap.
    public let sourceRowsPerLayer: Int
    /// Rows actually retained per layer (`min(source, cap)`).
    public let retainedRowsPerLayer: Int
    /// The per-layer row cap in force.
    public let rowCapPerLayer: Int
    /// Number of layers being captured.
    public let layerCount: Int
    public let hiddenSize: Int
    /// Projected peak CPU float32 working set, in bytes.
    public let estimatedBytes: Int
    /// Budget the estimate is checked against.
    public let budgetBytes: Int

    public var fits: Bool { estimatedBytes <= budgetBytes }

    public var summary: String {
        "\(retainedRowsPerLayer) rows/layer × \(layerCount) layers × \(hiddenSize) hidden ≈ "
            + "\(NeutralBankBudget.humanBytes(estimatedBytes)) CPU float32 "
            + "(budget \(NeutralBankBudget.humanBytes(budgetBytes)))"
    }
}

/// Memory preflight for the neutral token bank.
///
/// The bank is the extraction path's crash-grade hazard: it holds one CPU
/// float32 row per banked token position PER LAYER, and an all-layer capture
/// over a real corpus is multi-GB. Bounded ingestion (see
/// `ActivationBankRecorder`) removes the unbounded transient, but the RETAINED
/// bank is still `rows × layers × hidden` — big enough to be worth refusing
/// up front, with a message that names the two levers that actually shrink it.
public enum NeutralBankBudget {
    /// CPU float32.
    public static let bytesPerValue = 4

    /// Working-set multiplier over the retained bank. The bank itself, plus
    /// the centered copy the per-layer PCA makes of one layer's rows, plus
    /// slack for `[[Float]]` row-array overhead. Deliberately coarse: this
    /// catches orders of magnitude, it does not model the allocator.
    public static let workingSetMultiplier = 2

    /// Fraction of physical memory a bank may project to occupy. A quarter of
    /// a 64 GB machine is 16 GB — comfortably above an honest all-layer 27B
    /// bank (~5.5 GB) and far below the unbounded shapes that killed the
    /// process.
    public static let physicalMemoryFraction = 4

    /// Never refuse below this, so small machines and CI hosts still run
    /// ordinary banks.
    public static let minimumBudgetBytes = 1 << 30  // 1 GiB

    public static func defaultBudgetBytes(
        physicalMemory: UInt64 = ProcessInfo.processInfo.physicalMemory
    ) -> Int {
        let share = Int(min(physicalMemory / UInt64(physicalMemoryFraction), UInt64(Int.max)))
        return max(minimumBudgetBytes, share)
    }

    public static func preflight(
        sourceRowsPerLayer: Int,
        rowCapPerLayer: Int,
        layerCount: Int,
        hiddenSize: Int,
        budgetBytes: Int? = nil
    ) -> NeutralBankPreflight {
        let retained =
            rowCapPerLayer > 0
            ? min(max(0, sourceRowsPerLayer), rowCapPerLayer)
            : max(0, sourceRowsPerLayer)
        let values = retained * max(0, layerCount) * max(0, hiddenSize)
        let estimate = values.multipliedReportingOverflow(by: bytesPerValue * workingSetMultiplier)
        return NeutralBankPreflight(
            sourceRowsPerLayer: sourceRowsPerLayer,
            retainedRowsPerLayer: retained,
            rowCapPerLayer: rowCapPerLayer,
            layerCount: layerCount,
            hiddenSize: hiddenSize,
            estimatedBytes: estimate.overflow ? Int.max : estimate.partialValue,
            budgetBytes: budgetBytes ?? defaultBudgetBytes())
    }

    public static func humanBytes(_ bytes: Int) -> String {
        let gigabyte = 1 << 30
        let megabyte = 1 << 20
        if bytes >= gigabyte {
            return String(format: "%.1f GB", Double(bytes) / Double(gigabyte))
        }
        return String(format: "%.0f MB", Double(bytes) / Double(megabyte))
    }
}
