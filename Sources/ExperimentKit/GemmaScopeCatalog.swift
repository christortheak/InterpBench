import Foundation

public struct GemmaScopeInfo: Codable, Sendable, Equatable {
    public let loadedModelID: String
    public let suiteName: String
    public let modelSize: String
    public let tuning: String
    public let repository: String
    public let repositoryURL: String
    public let landingPageURL: String
    public let recommendedSite: String
    public let recommendedRelease: String
    public let recommendedLayer: Int
    public let recommendedSAEID: String
    public let availableLayers: [Int]
    public let supportedSites: [String]
    public let notes: [String]

    public var saeLensSnippet: String {
        """
        from sae_lens import SAE

        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release="\(recommendedRelease)",
            sae_id="\(recommendedSAEID)",
        )
        """
    }
}

public enum GemmaScopeCatalog {
    public static func info(
        for modelID: String?,
        layerCount: Int? = nil,
        preferredLayer: Int? = nil
    ) -> GemmaScopeInfo? {
        guard let modelID else { return nil }
        let lowercased = modelID.lowercased()
        guard lowercased.contains("gemma-3") else { return nil }

        guard let size = gemma3Size(in: lowercased) else { return nil }
        let tuning = lowercased.contains("-pt") ? "pt" : "it"
        let layers = availableResidualLayers(size: size, tuning: tuning)
        let layer = recommendedLayer(
            layerCount: layerCount, preferredLayer: preferredLayer, availableLayers: layers)
        let repository = "google/gemma-scope-2-\(size)-\(tuning)"
        let releasePrefix = "gemma-scope-2-\(size)-\(tuning)"

        return GemmaScopeInfo(
            loadedModelID: modelID,
            suiteName: "Gemma Scope 2",
            modelSize: size,
            tuning: tuning,
            repository: repository,
            repositoryURL: "https://huggingface.co/\(repository)",
            landingPageURL: "https://huggingface.co/google/gemma-scope-2",
            recommendedSite: "resid_post",
            recommendedRelease: "\(releasePrefix)-res",
            recommendedLayer: layer,
            recommendedSAEID: "layer_\(layer)_width_16k_l0_medium",
            availableLayers: layers,
            supportedSites: [
                "resid_post", "attn_out", "mlp_out", "transcoder",
                "resid_post_all", "attn_out_all", "mlp_out_all", "transcoder_all",
                "crosscoder", "clt",
            ],
            notes: [
                "Start with residual-stream SAEs because SteerLab vectors live in the residual stream.",
                "SAELens exposes Gemma Scope 2 residual SAEs at selected layers; the workbench snaps to the nearest available layer.",
                "Use the suggested SAE id as a fast first pass; sweep wider dictionaries and L0 targets before treating a feature match as evidence.",
                "This panel prepares external SAELens analysis; SteerLab does not yet decode SAE activations inside Swift.",
            ])
    }

    private static func gemma3Size(in modelID: String) -> String? {
        for size in ["270m", "1b", "4b", "12b", "27b"] where modelID.contains(size) {
            return size
        }
        return nil
    }

    private static func availableResidualLayers(size: String, tuning: String) -> [Int] {
        switch (size, tuning) {
        case ("4b", "it"): return [9, 17, 22, 29]
        case ("12b", "it"): return [12, 24, 31, 41]
        case ("27b", "it"): return [16, 31, 40, 53]
        default: return []
        }
    }

    private static func recommendedLayer(
        layerCount: Int?, preferredLayer: Int?, availableLayers: [Int]
    ) -> Int {
        let requested: Int
        if let preferredLayer {
            requested = max(0, preferredLayer)
        } else if let layerCount, layerCount > 0 {
            requested = max(0, layerCount / 2)
        } else {
            requested = 12
        }
        guard !availableLayers.isEmpty else {
            return requested
        }
        return availableLayers.min { lhs, rhs in
            abs(lhs - requested) < abs(rhs - requested)
        } ?? requested
    }
}
