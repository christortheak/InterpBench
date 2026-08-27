import Foundation

/// Everything the sweep surface needs to show, resolved in one place (E2).
///
/// The optimization surface previously spread its facts across three screens:
/// the grid lived in "New Agent", the criterion in Optimizations, and the
/// files the sweep actually opens were named only as raw strings that might or
/// might not exist. A researcher composing a sweep had to hold the layer
/// fractions in one place, guess what they resolved to, and take the file
/// paths on faith.
///
/// Two things this deliberately does NOT do:
///
/// - It does not load a model to resolve depth fractions. Layer counts come
///   from CACHED metadata (a vector sidecar for the study model, local or
///   from the server catalog). Requiring a 27B load to answer "what is 0.66
///   of this network?" is the reason that question went unanswered.
/// - It does not render an absent control margin as an empty field. An empty
///   box and "no control declared" look identical and mean very different
///   things — one of them means the winning cell was never tested against a
///   random direction of the same norm.
public enum SweepPanelModel {

    /// One resolved grid cell — a depth fraction turned into the layer index
    /// the sweep will actually steer at.
    public struct GridCell: Sendable, Equatable {
        public var fraction: Double
        public var layer: Int
        public var alpha: Double
    }

    /// A file the sweep opens, resolved.
    public struct InstrumentFile: Sendable, Equatable {
        public var label: String
        public var declaredPath: String
        public var resolvedPath: String
        public var exists: Bool
        public var rowCount: Int?
        public var sha256: String?
        /// The manifest's pin for this file, when one exists.
        public var pinnedHash: String?

        /// The pin disagrees with the bytes on disk.
        public var drifted: Bool {
            guard let pinnedHash, let sha256 else { return false }
            return pinnedHash != sha256
        }

        public var detail: String {
            guard exists else {
                return "\(declaredPath) — MISSING; the sweep refuses at start"
            }
            var parts: [String] = [declaredPath]
            if let rowCount { parts.append("\(rowCount) rows") }
            if let sha256 { parts.append(String(sha256.prefix(8)) + "…") }
            if drifted { parts.append("DRIFTED from its pin — sweep refuses") }
            else if pinnedHash != nil { parts.append("pinned") }
            return parts.joined(separator: " · ")
        }
    }

    /// The control margin's state, said in words. An absent control is a
    /// FACT about the evidence, not a blank field.
    public enum ControlState: Sendable, Equatable {
        case declared(margin: Double)
        case absent

        public var detail: String {
            switch self {
            case .declared(let margin):
                "matched-norm random control declared: the winner must beat a "
                    + "norm-matched random direction by ≥ "
                    + margin.formatted(.number.precision(.fractionLength(0 ... 4)))
            case .absent:
                "NO matched-norm random control — the winning cell will not be "
                    + "tested against a random direction of the same norm, so "
                    + "its margin over noise is unmeasured"
            }
        }

        public var isDeclared: Bool {
            if case .declared = self { return true }
            return false
        }
    }

    public struct Resolved: Sendable, Equatable {
        public var cells: [GridCell]
        public var layerCount: Int?
        /// Distinct resolved layers, ascending — fractions can collide.
        public var layers: [Int]
        public var alphas: [Double]
        public var files: [InstrumentFile]
        public var objective: String
        public var capabilityTolerance: Double
        /// The ABSOLUTE distinct-2 floor (the backstop, under the
        /// baseline-relative rule) — always a true "no cell passes below
        /// this" number.
        public var coherenceFloor: Double
        /// Non-nil = the baseline-relative rule at this multiple of the α=0
        /// baseline's distinct-2.
        public var coherenceRatioToBaseline: Double?
        public var control: ControlState

        /// Cells actually executed, plus the implied baseline.
        public var cellCount: Int { layers.count * alphas.count }

        /// Two fractions that resolve to the SAME layer are one cell, not
        /// two — worth saying, because a grid of "four depths" that is really
        /// three is a silently smaller sweep.
        public var collapsedFractions: Int {
            max(0, Set(cells.map(\.fraction)).count - layers.count)
        }

        public var gridSummary: String {
            guard let layerCount else {
                return "\(alphas.count) alpha\(alphas.count == 1 ? "" : "s") × "
                    + "\(Set(cells.map(\.fraction)).count) depth fraction"
                    + "\(cells.count == 1 ? "" : "s") — layer indices unknown "
                    + "until a vector for this model is available (no cached "
                    + "layer count)"
            }
            var text = "\(cellCount) cells: layers "
                + layers.map(String.init).joined(separator: ", ")
                + " of \(layerCount) × alphas "
                + alphas.map {
                    $0.formatted(.number.precision(.fractionLength(0 ... 4)))
                }.joined(separator: ", ")
            if collapsedFractions > 0 {
                text += " — \(collapsedFractions) depth fraction"
                    + "\(collapsedFractions == 1 ? "" : "s") collapsed onto a "
                    + "layer already in the grid"
            }
            return text
        }
    }

    /// Resolve a declared sweep spec against a cached layer count.
    ///
    /// `layerCount` nil = no cached vector for this model yet; the grid is
    /// still shown, with the fractions unresolved and that fact stated.
    public static func resolve(
        spec: ExperimentManifest.SweepSpec,
        criterion: SweepSelectionRule.Resolved,
        layerCount: Int?,
        files: [InstrumentFile]
    ) -> Resolved {
        var cells: [GridCell] = []
        var layers: [Int] = []
        if let layerCount, layerCount > 0 {
            layers = spec.resolvedLayers(layerCount: layerCount)
            for fraction in spec.layerFractions {
                // Same truncating, clamped rule `resolvedLayers` applies, so
                // a fraction means one thing across the app.
                let layer = min(max(0, Int(fraction * Double(layerCount))), layerCount - 1)
                for alpha in spec.alphas {
                    cells.append(GridCell(fraction: fraction, layer: layer, alpha: alpha))
                }
            }
        } else {
            for fraction in spec.layerFractions {
                for alpha in spec.alphas {
                    cells.append(GridCell(fraction: fraction, layer: -1, alpha: alpha))
                }
            }
        }
        return Resolved(
            cells: cells,
            layerCount: layerCount,
            layers: layers,
            alphas: spec.alphas,
            files: files,
            objective: criterion.metric,
            capabilityTolerance: criterion.capabilityTolerance,
            coherenceFloor: criterion.coherenceFloor,
            coherenceRatioToBaseline: criterion.coherenceRatioToBaseline,
            control: criterion.matchedNormRandomMargin.map(ControlState.declared)
                ?? .absent)
    }

    /// The three files a sweep opens, in the order it opens them. A
    /// `choicePromptsFile` is listed only when the objective reads it.
    public static func declaredFiles(
        spec: ExperimentManifest.SweepSpec, objective: String
    ) -> [(label: String, path: String, pinnedHash: String?)] {
        var out: [(String, String, String?)] = [
            ("dev prompts", spec.devPromptsFile, spec.devPromptsHash),
            ("capability battery", spec.batteryFile, spec.batteryHash),
        ]
        if objective == "logprobShift" {
            // Choice instruments are freeze-time pins since 2026-08-02
            // (review P1) — display the pin exactly like dev/battery, per
            // declared instrument.
            if let choices = spec.selection?.objective?.choicePromptsFile,
                !choices.isEmpty
            {
                out.append(("choice prompts", choices,
                            spec.selection?.objective?.choicePromptsHash))
            }
            for (concept, rel) in (spec.selection?.objective?
                .choicePromptsFiles ?? [:]).sorted(by: { $0.key < $1.key })
            {
                out.append(("choice prompts '\(concept)'", rel,
                            spec.selection?.objective?
                                .choicePromptsHashes?[concept]))
            }
        }
        return out
    }
}

extension SweepPanelModel {

    /// Read a declared file's real state off disk. Never throws: a missing or
    /// unreadable file is a FACT to display, not an error to swallow.
    public static func inspect(
        label: String, path: String, pinnedHash: String?
    ) -> InstrumentFile {
        let trimmed = path.trimmingCharacters(in: .whitespaces)
        let url = ExperimentStore.resolveProjectPath(trimmed)
        guard !trimmed.isEmpty, let data = try? Data(contentsOf: url) else {
            return InstrumentFile(
                label: label, declaredPath: trimmed, resolvedPath: url.path,
                exists: false, rowCount: nil, sha256: nil,
                pinnedHash: pinnedHash)
        }
        let rows = String(decoding: data, as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: true)
            .count { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        return InstrumentFile(
            label: label, declaredPath: trimmed, resolvedPath: url.path,
            exists: true, rowCount: rows,
            sha256: ExperimentStore.sha256Hex(data), pinnedHash: pinnedHash)
    }

    /// One artifact's claim about a model's depth, for a refusal to name.
    /// Server twin: `experiment_store.DepthWitness`.
    public struct DepthWitness: Sendable, Equatable {
        public let artifact: String
        public let depth: Int

        public init(artifact: String, depth: Int) {
            self.artifact = artifact
            self.depth = depth
        }
    }

    /// What this workspace can say about a pinned model's depth.
    ///
    /// `depth` is the agreed answer, or nil when nothing states it AND when the
    /// witnesses conflict — a caller that only wants a number treats both the
    /// same way, and a caller that must refuse reads `conflict`. Server twin:
    /// `experiment_store.CachedDepth`.
    public struct CachedDepth: Sendable, Equatable {
        public let depth: Int?
        public let conflict: [DepthWitness]

        public init(depth: Int?, conflict: [DepthWitness] = []) {
            self.depth = depth
            self.conflict = conflict
        }
    }

    /// The pinned model's depth, from the vector sidecars already on disk for
    /// it — and the honest answer when they do not agree.
    ///
    /// Deliberately catalog-only: loading a 27B model to answer "what is 0.66
    /// of this network?" is the reason that question went unanswered.
    ///
    /// **Not every artifact may answer (review round 6, finding 2).** The old
    /// rule was "the first sidecar for this model wins", and reader-derived
    /// artifacts are PARTIAL by construction — a reader fitted at block 10
    /// writes `layerCount: 11`. One of those made an entire 42-block model
    /// eleven blocks deep, and absolute sweep layers then converted against a
    /// network that does not exist. Only artifacts that state the model's
    /// DEPTH are witnesses; `SteeringVectorSidecar.statesModelDepth` is the
    /// single definition of which those are.
    ///
    /// **A revision, when the caller knows one, is part of the question.** Two
    /// revisions of a checkpoint can differ in depth, so an artifact stamped
    /// with a different revision is not evidence about this one. An artifact
    /// carrying NO revision is legacy and unattributable: it cannot be shown
    /// to be about a different model, so it still counts.
    ///
    /// **Disagreement is reported, never resolved.** Picking one of two depths
    /// would silently pick a network. Server twin:
    /// `experiment_store.cached_depth`.
    public static func cachedDepth(
        modelID: String, revision: String? = nil
    ) -> CachedDepth {
        var witnesses: [DepthWitness] = []
        var seen: Set<Int> = []
        for artifact in VectorCatalog.scan() {
            let sidecar = artifact.sidecar
            guard sidecar.modelID == modelID, sidecar.layerCount > 0 else { continue }
            if let revision, let stamped = sidecar.revision, stamped != revision {
                continue
            }
            guard sidecar.statesModelDepth else { continue }
            guard seen.insert(sidecar.layerCount).inserted else { continue }
            witnesses.append(
                DepthWitness(
                    artifact: "runs/\(artifact.directory.lastPathComponent)"
                        + "/\(artifact.name)",
                    depth: sidecar.layerCount))
        }
        // Sorted, not scan-ordered: the two engines walk runs/ in opposite
        // directions, and a refusal that names artifacts must read the same on
        // both.
        witnesses.sort { $0.artifact < $1.artifact }
        guard let only = witnesses.first else { return CachedDepth(depth: nil) }
        if witnesses.count > 1 {
            return CachedDepth(depth: nil, conflict: witnesses)
        }
        return CachedDepth(depth: only.depth)
    }

    /// The agreed depth, or nil when nothing states it or the witnesses
    /// conflict. Callers that must refuse on a conflict read `cachedDepth`.
    /// Server twin: `experiment_store.cached_layer_count`.
    public static func cachedLayerCount(
        modelID: String, revision: String? = nil
    ) -> Int? {
        cachedDepth(modelID: modelID, revision: revision).depth
    }

    /// Resolve straight from a manifest — the app's entry point.
    public static func resolve(
        manifest: ExperimentManifest, layerCount: Int? = nil
    ) -> Resolved? {
        guard let spec = manifest.sweep else { return nil }
        let criterion =
            (try? SweepSelectionRule.resolve(spec.selection))
            ?? SweepSelectionRule.Resolved(
                metric: "markerDensity",
                capabilityTolerance: SweepSelectionRule.defaultCapabilityTolerance,
                coherenceFloor: SweepSelectionRule.defaultCoherenceFloor,
                matchedNormRandomMargin: nil)
        let files = declaredFiles(spec: spec, objective: criterion.metric)
            .map { inspect(label: $0.label, path: $0.path, pinnedHash: $0.pinnedHash) }
        return resolve(
            spec: spec, criterion: criterion,
            // The manifest's pinned revision is part of the depth question
            // (review round 7, finding 4): the RUN path already asks it that
            // way (`ExperimentStore.setSweepGrid`), and asking by model id
            // alone here let the editor display absolute layers converted
            // against a different revision's artifacts.
            layerCount: layerCount
                ?? cachedLayerCount(
                    modelID: manifest.modelID, revision: manifest.modelRevision),
            files: files)
    }
}
