import Foundation

@testable import ExperimentKit

/// A complete, FICTIONAL cluster site for tests that need a rich SSH profile.
///
/// Until WP5 Step 12 these tests reached for the shipped institutional preset —
/// the one profile in the tree that had a real host, a partition list, a GPU
/// inventory, and a purge window, which is exactly the shape registry,
/// persistence, editor, and sharding tests need. §4.2 (DECIDED 2026-08-17)
/// moved real site configuration out of the repo, so the SHAPE is declared
/// here instead and the institution is not. Nothing about these tests' meaning
/// changed: they were never about that cluster, only about a fully populated
/// SSH site.
///
/// Values are deliberately the fictional-fixture vocabulary already used by
/// `prompts/fixtures/cluster-site-profile/` (`slurm.example.edu`, `gpu_p`,
/// A100/H100/L4), so a reader moving between the committed cross-engine
/// fixtures and these tests sees one worked example, not two.
extension ClusterSiteProfile {

    /// A fully populated SSH site: daemon-in-a-job topology, four partitions,
    /// a three-type GPU inventory with VRAM, a default gres and partition, a
    /// purge window, and declared compute egress. No storage roots — those are
    /// per-account, and their absence is what several tests are about.
    static var exampleCluster: ClusterSiteProfile {
        ClusterSiteProfile(
            name: "Example HPC",
            notes: """
                FICTIONAL test site. Login/submit nodes are edit-and-submit only, so \
                the controller runs as a 1-core batch job and the tunnel forwards to \
                the node it lands on via serverd.host. accountRequired is left false; \
                flip it (and set the account) if sbatch rejects submissions without \
                --account. Storage roots are per-account and filled after first login.
                """,
            transport: .ssh(
                host: "slurm.example.edu", proxyJump: nil, remotePort: 8080,
                vpnExpected: true),
            topology: .daemonInJob,
            scheduler: .slurm(
                SlurmSiteData(
                    partitions: [
                        SlurmSiteData.PartitionInfo(name: "gpu_p", maxWalltimeHours: 168),
                        SlurmSiteData.PartitionInfo(name: "gpu_long_p", maxWalltimeHours: 720),
                        SlurmSiteData.PartitionInfo(name: "batch", maxWalltimeHours: 168),
                        SlurmSiteData.PartitionInfo(name: "inter_p", maxWalltimeHours: 48),
                    ],
                    gpuTypes: ["A100", "H100", "L4"],
                    gpuVRAMGB: ["A100": 80, "H100": 80, "L4": 24],
                    defaultGres: "gpu:A100:1",
                    defaultPartition: "gpu_p",
                    accountRequired: false,
                    account: nil,
                    billedAllocations: false)),
            constraints: SiteConstraints(
                computeEgress: .yes,
                storageRoots: [:],
                purgeDays: 30,
                purgeWarnDays: 20,
                maintenanceSource: nil))
    }
}
