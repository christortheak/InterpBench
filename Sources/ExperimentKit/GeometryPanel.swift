import Foundation
import Observation
import SteeringKit

/// Shared state for the Analysis section's vector-geometry compute. The
/// CONTROLS (vector selection, layer stepper, Compute) live in the Analysis
/// pane; the computed cosine/RSA tables render in the right-hand viewer —
/// the same controls/viewer split as the robustness check and Results. That
/// split is why this state lives on the service and not in the view.
@Observable @MainActor
public final class GeometryPanel {
    public internal(set) weak var host: ChatService?

    // Local compute (pure math over local vector artifacts).
    public private(set) var result: GeometryAnalysisResult?
    public var selectedLayer = 0
    public private(set) var status: String?

    /// The LOCAL vector selection the Analysis controls edit and `compute`
    /// reads (`VectorArtifact.ID`s).
    ///
    /// It lives here rather than in the view (WP-Data phase 4) for the same
    /// reason `result` does: the pane's controls and the viewer's tables are
    /// two halves of one state, and a cross-section route
    /// (`DerivedArtifactRoute.analysis` — "Open in Analysis" on a vector row)
    /// has to be able to PRESELECT the artifact it names. A view-local
    /// `@State` set is unreachable from the model layer, which is why that
    /// route could only switch sections and ask the researcher to find the
    /// row again.
    public private(set) var selectedIDs: Set<VectorArtifact.ID> = []

    // Server compute (POST /api/geometry over the server catalog; one layer
    // per request, matrix rendered with the same cell views as local).
    public private(set) var serverMatrix: GeometryLayerMatrix?
    public private(set) var serverLayers: [Int] = []
    public private(set) var serverStatus: String?
    public private(set) var isServerComputing = false

    public init() {}

    /// The local matrix at the selected layer (stepper and viewer share it).
    public var activeMatrix: GeometryLayerMatrix? {
        guard let result, !result.matrices.isEmpty else { return nil }
        let index = min(max(0, selectedLayer), result.matrices.count - 1)
        return result.matrices[index]
    }

    public func compute(artifacts: [VectorArtifact]) {
        do {
            let analysis = try GeometryAnalysis.analyze(artifacts: artifacts)
            result = analysis
            selectedLayer = min(selectedLayer, max(0, analysis.matrices.count - 1))
            status = "computed \(analysis.matrices.count) layer matrices — tables in the viewer"
        } catch {
            result = nil
            status = "\(error)"
        }
    }

    // MARK: Local selection

    /// Replace the local selection. Any table computed from a DIFFERENT
    /// selection is retired in the same breath — a cosine matrix beside a
    /// selection it does not describe is the failure this rule exists to
    /// prevent (the same rule the per-row toggle already followed).
    public func select(vectorIDs: Set<VectorArtifact.ID>) {
        guard vectorIDs != selectedIDs else { return }
        selectedIDs = vectorIDs
        clearLocal()
    }

    /// Add or remove one vector — what the pane's per-row toggle calls.
    public func setSelected(_ id: VectorArtifact.ID, _ isSelected: Bool) {
        var next = selectedIDs
        if isSelected { next.insert(id) } else { next.remove(id) }
        select(vectorIDs: next)
    }

    public func clearSelection() { select(vectorIDs: []) }

    /// Selected ids the given offerable list does not contain — a vector a
    /// route preselected that this pane cannot show (the local list is
    /// filtered to the LOADED model, `ChatService.compatibleVectors`).
    /// Pure, so the pane can say so instead of showing an invisible
    /// selection. Sorted for a stable caption.
    public func unlistableSelection(among available: [VectorArtifact]) -> [String] {
        let listed = Set(available.map(\.id))
        return selectedIDs.subtracting(listed).sorted()
    }

    /// Viewer-state hygiene (phase 0): a table whose selection is empty
    /// describes nothing the pane still names, so the empty state wins.
    /// Recompute is one click.
    public func retireOrphanedLocalTable() {
        if selectedIDs.isEmpty { clearLocal() }
    }

    public func clearLocal() {
        result = nil
        status = nil
        selectedLayer = 0
    }

    public func clearServer() {
        serverMatrix = nil
        serverLayers = []
        serverStatus = nil
    }

    public func computeOnServer(records: [RemoteVectorRecord], layer: Int) {
        guard let host, let client = host.cluster.client else {
            serverStatus = "no server connection — check the Compute selector"
            return
        }
        let requested = records.count
        isServerComputing = true
        serverStatus = "computing on \(host.cluster.substrateLabel)…"
        Task { [weak self] in
            do {
                let remote = try await client.geometry(vectors: records, layer: layer)
                guard let self else { return }
                self.isServerComputing = false
                self.serverStatus = remote.statusLine(requested: requested)
                // An empty matrix is NEVER a success caption: a new server
                // answers 400 when every vector was unloadable, but an older
                // server (no skip contract) can still return 0×0 — treat it
                // as the error it is.
                guard !remote.labels.isEmpty else {
                    self.serverMatrix = nil
                    self.serverLayers = []
                    return
                }
                // nil cells (pairs the server could not compare) render as
                // "—", never as a fake 0.
                let values = remote.matrix.map { row in
                    row.map { $0 ?? Float.nan }
                }
                self.serverMatrix = GeometryLayerMatrix(
                    layer: remote.layers.first ?? layer,
                    labels: remote.labels,
                    values: values)
                self.serverLayers = remote.layers
            } catch {
                guard let self else { return }
                self.isServerComputing = false
                self.serverMatrix = nil
                // ClusterClient rethrows the route's 400 with the server's
                // self-naming detail (which vectors failed to load and why)
                // — show it verbatim.
                self.serverStatus = "\(error)"
            }
        }
    }
}
