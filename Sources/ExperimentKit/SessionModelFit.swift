import Foundation

/// Memory-fit gating for server models against the LIVE session's GPU —
/// one eligibility rule shared by every panel that offers a load (Playground
/// and Geometry both do; engineer reviews 2026-07-18: a disabled picker row
/// alone neither clears an already-selected oversized model nor stops a Load
/// button, and a rule that lives in one panel's view file silently misses
/// the next panel).
///
/// The rule is deliberately a mirror of the server's own `_assert_gpu_capacity`
/// preflight — the server stays authoritative; this is the same arithmetic
/// applied before the click instead of after the wait.
public enum SessionModelFit {

    /// Mirror of the server preflight's context floor: weights alone never
    /// run — CUDA context + first KV pages come on top.
    public static let headroomBytes: Int64 = 1 << 30

    // MARK: Pure core (unit-tested)

    /// Human-readable refusal when a model of `modelBytes` cannot fit
    /// `capacityBytes`, else nil. Nil inputs never gate (absent facts are
    /// not evidence of unfitness).
    public static func tooBigNote(
        modelBytes: Int64?, capacityBytes: Int64?
    ) -> String? {
        guard let modelBytes, let capacityBytes,
            Double(modelBytes) + Double(headroomBytes) > Double(capacityBytes)
        else { return nil }
        return String(
            format: "too big for this session's GPU (%.1f GiB)",
            Double(capacityBytes) / Double(1 << 30))
    }

    /// Capacity resolution: the worker's reported ACTUAL CUDA total wins
    /// (an "L4 24 GB" reports 22.05 GiB usable); before the worker's first
    /// probe, the profile's marketed GB is read CONSERVATIVELY as decimal
    /// bytes (24 GB → 24e9 B < 24 GiB) — reading it as GiB passed the exact
    /// 12B-on-L4 case this gate was built for (engineer review 2026-07-18).
    public static func capacityBytes(
        actualBytes: Int64?, profileVRAMGB: Int?
    ) -> Int64? {
        if let actualBytes { return actualBytes }
        guard let profileVRAMGB else { return nil }
        return Int64(profileVRAMGB) * 1_000_000_000
    }

    // MARK: Store adapter

    /// The refusal for `model` against the ACTIVE session's GPU, else nil
    /// (fits, size unknown, or no live session — a future session may be
    /// bigger, so absent facts never gate).
    @MainActor
    public static func tooBigNote(
        cluster: ClusterConnectionStore, model: String?
    ) -> String? {
        guard cluster.computeTarget == .server, let model else { return nil }
        return tooBigNote(
            modelBytes: cluster.remoteState?.modelSizesBytes?[model],
            capacityBytes: sessionGPUCapacityBytes(cluster: cluster))
    }

    /// Capacity of the ACTIVE session's GPU in bytes; nil when no live
    /// (nonterminal) session or nothing is known about its GPU.
    @MainActor
    public static func sessionGPUCapacityBytes(
        cluster: ClusterConnectionStore
    ) -> Int64? {
        let session = cluster.gpuSession
        switch session.displayState {
        case .queued, .starting, .ready, .busy, .idle, .unknownState:
            break  // a live (nonterminal) session pins the GPU
        default:
            return nil
        }
        var profileVRAMGB: Int?
        if let gpuType = session.record?.gpuType,
            let site = cluster.activeSite,
            case .slurm(let slurm) = site.scheduler
        {
            profileVRAMGB = slurm.gpuVRAMGB[gpuType]
        }
        return capacityBytes(
            actualBytes: session.record?.gpuTotalMemoryBytes,
            profileVRAMGB: profileVRAMGB)
    }
}
