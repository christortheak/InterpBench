import Foundation
import SteeringKit

/// Headless Promote: mint an agent (model-variant artifact) from a
/// sweep-selected cell — the lifecycle edge between screening and
/// confirmation.
///
/// A sweep selects `<concept>-recommended` under a data-declared criterion
/// and stamps selection provenance; `promote` turns that cell into a named,
/// reusable variant artifact carrying a *birth certificate* (which sweep run,
/// which resolved criterion, which dev split, which cell, which metrics), so
/// an evidence-grade agent can prove its settings were chosen on dev data by
/// a predeclared rule, before held-out confirmation or panel studies.
///
/// Promoting an arbitrary non-selected cell is possible but LOUD: it requires
/// the explicit `cell:` override and is stamped `promotedBy: "manualOverride"`
/// — never silently, the same pattern as `freeze --force`.
public enum AgentPromotion {

    /// The PINNED promotion contract (B2, 2026-07-26) — cross-engine twin of
    /// the server's `PromotionPins`.
    ///
    /// Promotion used to resolve its own inputs by recency: the newest sweep
    /// run in the workspace, and the newest extraction artifact matching the
    /// recipe. Both are ambient — they change when unrelated work lands.
    /// With several sweeps per concept (the ordinary state once a grid is
    /// being iterated), a promote issued after a re-sweep silently bound to
    /// whichever run was newest, and the birth certificate then recorded a
    /// cell the researcher had not chosen.
    ///
    /// Every field is supplied by the caller and VERIFIED against the
    /// evidence rather than discovered from it.
    public struct Pins: Sendable, Equatable {
        /// The only `recommendations.json` consulted.
        public var sweepRun: String
        /// The manifest epoch the caller believes it is promoting under.
        public var experimentHash: String?
        /// Must AGREE with the sweep's recommendation. A disagreement means
        /// the caller's view is stale and is refused — it is NOT reinterpreted
        /// as a manual override, which is a deliberate, separately stamped
        /// gesture.
        public var winningCell: (layer: Int, alpha: Double)?
        /// The exact extraction artifact to inject. Still checked against the
        /// full recipe identity: pinning selects WHICH artifact, it never
        /// waives WHETHER it matches.
        public var vectorArtifactID: String?
        /// The artifact bytes the caller expects.
        public var vectorArtifactHash: String?

        public init(
            sweepRun: String,
            experimentHash: String? = nil,
            winningCell: (layer: Int, alpha: Double)? = nil,
            vectorArtifactID: String? = nil,
            vectorArtifactHash: String? = nil
        ) {
            self.sweepRun = sweepRun
            self.experimentHash = experimentHash
            self.winningCell = winningCell
            self.vectorArtifactID = vectorArtifactID
            self.vectorArtifactHash = vectorArtifactHash
        }

        public static func == (lhs: Pins, rhs: Pins) -> Bool {
            lhs.sweepRun == rhs.sweepRun
                && lhs.experimentHash == rhs.experimentHash
                && lhs.winningCell?.layer == rhs.winningCell?.layer
                && lhs.winningCell?.alpha == rhs.winningCell?.alpha
                && lhs.vectorArtifactID == rhs.vectorArtifactID
                && lhs.vectorArtifactHash == rhs.vectorArtifactHash
        }
    }

    /// Mint a variant artifact from the concept's sweep-selected cell.
    /// Works on any manifest status: promoting from a frozen experiment is
    /// the expected confirmation-stage gesture. Pure CPU — no model load.
    @discardableResult
    public static func promote(
        experimentName: String,
        concept: String,
        agentName: String? = nil,
        cell: (layer: Int, alpha: Double)? = nil,
        overrideReason: String? = nil,
        pins: Pins? = nil,
        log: (String) -> Void = { print($0) }
    ) throws -> ModelVariantRecord {
        let manifest = try ExperimentStore.load(name: experimentName)
        guard let ref = manifest.concepts.first(where: { $0.name == concept }) else {
            throw ExperimentError.refusing(
                .missingPrerequisite,
                "concept '\(concept)' is not attached to experiment "
                    + "'\(experimentName)'",
                repair: "steerlab-cli experiment attach \(experimentName) "
                    + "\(concept) && "
                    + Self.sweepThenPromoteRepair(experimentName, concept))
        }

        // The PINNED path resolves nothing by recency. The epoch guard runs
        // FIRST: a sweep run from a different manifest epoch selected its
        // cell under different settings, which makes every later check
        // meaningless.
        var pinnedEvidence: SweepRunCatalog.Recommendation?
        if let pins {
            let liveHash = ExperimentStore.manifestHash(manifest)
            if let declared = pins.experimentHash, declared != liveHash {
                throw ExperimentError.refusing(
                    .promotionEpoch,
                    "promote: the pinned contract names experiment "
                        + "hash \(declared), but '\(experimentName)' currently "
                        + "hashes \(liveHash) — the manifest changed since the "
                        + "promotion was planned; re-plan against the current "
                        + "manifest",
                    repair: Self.sweepThenPromoteRepair(experimentName, concept))
            }
            try Self.requirePlainRunName(
                pins.sweepRun, source: "the pinned contract")
            let runDirectory = ExperimentStore.runsDirectory
                .appending(component: pins.sweepRun)
            if let refusal = RunEpoch.refusal(
                verb: "promote", experiment: experimentName,
                liveHash: liveHash, runDirectory: runDirectory,
                liveManifest: manifest)
            {
                throw ExperimentError.refusing(
                    .promotionEpoch,
                    refusal + " — an agent minted from a sweep of a "
                        + "different epoch would carry a birth certificate "
                        + "naming settings selected under a different study",
                    repair: Self.sweepThenPromoteRepair(experimentName, concept))
            }
            guard let run = try? SweepRunCatalog.load(directory: runDirectory)
            else {
                throw ExperimentError.refusing(
                    .promotionEvidence,
                    "promote: sweep run '\(pins.sweepRun)' has no "
                        + "readable sweep.csv — cannot promote from it",
                    repair: Self.sweepThenPromoteRepair(experimentName, concept))
            }
            guard let recommendation = run.recommendations[concept] else {
                throw ExperimentError.refusing(
                    .promotionEvidence,
                    "promote: sweep run '\(pins.sweepRun)' carries no "
                        + "recommendation entry for '\(concept)'",
                    repair: Self.sweepThenPromoteRepair(experimentName, concept))
            }
            if case .selected(let provenance) = recommendation {
                try Self.requireEntryNamesItsRun(
                    provenance, sweepRun: pins.sweepRun, concept: concept)
            }
            pinnedEvidence = recommendation
        }

        let recommended = pins == nil
            ? manifest.conditions.first {
                $0.name == "\(concept)-recommended" && $0.selection != nil
            }
            : nil

        /// The sweep evidence this promotion is entitled to read: the pinned
        /// run when pinned, the newest matching run otherwise. Under pins the
        /// ambient lookup is never reached.
        func sweepEvidence() throws
            -> (runName: String, recommendation: SweepRunCatalog.Recommendation)?
        {
            if let pins {
                return pinnedEvidence.map { (pins.sweepRun, $0) }
            }
            return try newestSweepEvidence(
                experiment: experimentName, concept: concept)
        }

        let layer: Int
        let alpha: Double
        let bandWidth: Int
        let alphaInNormUnits: Bool
        let promotedBy: String
        // The sweep provenance the certificate copies from: the manifest's
        // recommendation when present, else the newest sweep run's
        // recommendations.json entry for this concept (the only place a sweep
        // on a frozen manifest can record its selection).
        var selection = recommended?.selection
        // Set only when the sweep's entry was a FAILURE message: the run that
        // concluded it, and what it concluded — so an override after a failed
        // selection records what the human deviated from.
        var failureSweepRun: String?
        var selectionOutcome: String?
        if let cell {
            layer = cell.layer
            alpha = cell.alpha
            bandWidth = 1
            alphaInNormUnits = true
            promotedBy = "manualOverride"
            log(
                "⚠︎ promote --cell L\(layer) α\(alpha): bypassing the declared "
                    + "selection for '\(concept)' — stamped promotedBy=manualOverride")
            if selection == nil {
                // A manual override documents a DEVIATION from a sweep; with
                // no sweep at all it would be hand-creation wearing a
                // promotion badge, so it must refuse. A failed selection
                // (failure string in recommendations.json) is legitimate
                // evidence — overriding after it is loud, not forbidden.
                guard let evidence = try sweepEvidence() else {
                    throw ExperimentError.refusing(
                        .promotionEvidence,
                        "no sweep has run for '\(concept)' in "
                            + "'\(experimentName)' — run 'experiment sweep' first "
                            + "(manual override documents a deviation from a "
                            + "sweep, it cannot replace one)",
                        repair: Self.sweepThenPromoteRepair(experimentName, concept))
                }
                switch evidence.recommendation {
                case .selected(let provenance):
                    selection = provenance
                case .failure(let message):
                    failureSweepRun = evidence.runName
                    selectionOutcome = message
                }
            }
        } else if let recommended, let slot = recommended.slots.first {
            layer = slot.layer
            alpha = slot.alpha
            bandWidth = recommended.bandWidth
            alphaInNormUnits = recommended.alphaInNormUnits
            // The manifest condition is a PROJECTION of a sweep run, not
            // evidence in itself: require the run it names to have actually
            // completed and to still recommend this exact cell — and take
            // the RUN's entry as the certificate source, so the birth
            // certificate can only carry the evidence's own criterion,
            // metrics, and control outcome (rounds 3–4).
            selection = try Self.conditionRunEvidence(
                concept: concept, selection: recommended.selection,
                layer: slot.layer, alpha: slot.alpha,
                experimentName: experimentName, log: log)
            promotedBy = "criterion"
        } else {
            // No stamped manifest condition. A sweep on a FROZEN manifest
            // cannot stamp `<concept>-recommended` (the manifest is read-only)
            // — it only reports into its run directory's recommendations.json.
            // The declared criterion still selected that cell, so promoting it
            // from the run evidence IS criterion promotion, with the full
            // provenance copied from the run entry.
            guard let evidence = try sweepEvidence() else {
                throw ExperimentError.refusing(
                    .promotionEvidence,
                    "no sweep-selected recommendation for '\(concept)' in "
                        + "'\(experimentName)' — run 'experiment sweep' first",
                    repair: Self.sweepThenPromoteRepair(experimentName, concept))
            }
            switch evidence.recommendation {
            case .selected(let provenance):
                selection = provenance
                layer = provenance.winningCell.layer
                alpha = provenance.winningCell.alpha
                // The sweep writes norm-unit cells; a run-evidence promotion
                // gets the same defaults the sweep stamps on the manifest.
                bandWidth = 1
                alphaInNormUnits = true
                promotedBy = "criterion"
            case .failure(let message):
                throw ExperimentError.refusing(
                    .promotionEvidence,
                    "the sweep selected no cell for '\(concept)': "
                        + "\(message) — a manual override (--cell … --reason …) "
                        + "is the only way to promote from this sweep",
                    repair: "steerlab-cli experiment promote \(experimentName) \(concept) --cell <layer>:<alpha> --reason \"<why this cell, given the sweep selected none>\"")
            }
        }

        // A pinned cell must AGREE with the sweep's recommendation.
        // Disagreement means the caller's plan is stale — refuse rather than
        // silently reinterpret it as a deliberate override.
        if let pins, let wanted = pins.winningCell, cell == nil {
            if wanted.layer != layer || abs(wanted.alpha - alpha) > 1e-12 {
                throw ExperimentError.refusing(
                    .promotionEvidence,
                    "promote: the pinned contract names cell "
                        + "L\(wanted.layer) α\(wanted.alpha) for '\(concept)', "
                        + "but sweep run '\(pins.sweepRun)' selected L\(layer) "
                        + "α\(alpha) — the plan is stale. Re-read the sweep's "
                        + "recommendation, or pass an explicit cell override "
                        + "with a reason to deviate deliberately",
                    repair: Self.sweepThenPromoteRepair(experimentName, concept))
            }
        }

        let match = try matchingVectorArtifact(
            manifest: manifest, ref: ref, experimentName: experimentName,
            pinnedArtifactID: pins?.vectorArtifactID)
        let artifact = match.artifact
        let artifactHash = artifactContentHash(artifact)
        if let pins, let expected = pins.vectorArtifactHash {
            // An expected hash with NO computable actual hash must refuse.
            // Comparing only when both exist fails OPEN: an unreadable or
            // missing .safetensors let the promotion proceed and mint an
            // agent whose claimed vector bytes were never verified — the one
            // thing the pin exists to prevent.
            guard let actual = artifactHash else {
                throw ExperimentError.refusing(
                    .artifactPin,
                    "promote: the pinned contract expects vector "
                        + "artifact bytes \(expected.prefix(12))… for "
                        + "'\(concept)', but the artifact's tensors could not "
                        + "be read at \(artifact.directory.path) — an "
                        + "unverifiable pin cannot pass",
                    repair: Self.extractThenPromoteRepair(experimentName, concept))
            }
            guard expected == actual else {
                throw ExperimentError.refusing(
                    .artifactPin,
                    "promote: the pinned contract expects vector artifact "
                        + "bytes \(expected.prefix(12))… for '\(concept)', but the "
                        + "artifact on disk hashes \(actual.prefix(12))… — it was "
                        + "re-extracted since the promotion was planned; re-plan "
                        + "against the current artifact",
                    repair: Self.extractThenPromoteRepair(experimentName, concept))
            }
        }

        // On a manual override the sweep's criterion/context still travel
        // (they say what the override DEVIATED from), but its metrics
        // describe the selected cell, so they only carry over when the cells
        // coincide.
        let sameCell =
            cell == nil
            || selection?.winningCell
                == ExperimentManifest.SelectionProvenance.Cell(layer: layer, alpha: alpha)
        let variantName = agentName ?? "\(experimentName)-\(concept)-agent"
        let experimentHash = ExperimentStore.manifestHash(manifest)
        // WORKSPACE-RELATIVE, like the Python twin (`os.path.relpath` in
        // `promote.py`) — for the serialized injection below AND the
        // promotion key here, so the two engines mint identical keys for the
        // same promotion. The absolute catalog id (2026-08-04) named this
        // Mac's filesystem inside a portable artifact: every panel/run on
        // the cluster then resolved it literally and died on a path that
        // exists nowhere but here.
        let artifactReference = ArtifactIdentity.workspaceRelative(artifact.id)
        let key = promotionKey(
            experiment: experimentName, experimentHash: experimentHash,
            concept: concept,
            sweepRun: selection?.sweepRun ?? failureSweepRun,
            layer: layer, alpha: alpha, vectorArtifactID: artifactReference,
            promotedBy: promotedBy, agentName: variantName,
            // The BYTES, not only the path — see the Python twin.
            vectorArtifactHash: artifactHash)

        // Idempotent retries: a re-submitted stage must not mint a second
        // agent from the same evidence — the duplicate would then compete
        // with the original in every picker, with nothing to tell them apart.
        if let existing = ModelVariantStore.scan().first(
            where: { $0.artifact.promotion?.promotionKey == key })
        {
            log(
                "promote '\(concept)': an agent for this exact promotion "
                    + "already exists (\(existing.url.path)) — returning it "
                    + "unchanged")
            return existing
        }

        let promotion = ModelVariantArtifact.Promotion(
            experiment: experimentName,
            experimentHash: experimentHash,
            promotedAt: ISO8601DateFormatter().string(from: Date()),
            promotedBy: promotedBy,
            // The reason documents a DEVIATION; a criterion-selected
            // promotion has nothing to explain, so it never carries one.
            overrideReason: cell != nil ? overrideReason : nil,
            sweepRun: selection?.sweepRun ?? failureSweepRun,
            criterion: selection?.criterion,
            devPromptsHash: selection?.devPromptsHash,
            winningCell: .init(layer: layer, alpha: alpha),
            metrics: sameCell ? selection?.metrics : nil,
            control: sameCell ? selection?.control : nil,
            selectionOutcome: selectionOutcome,
            recipeIdentityHash: match.recipeIdentityHash,
            // The substrate these settings APPLY to — the workspace's
            // compute engine, which is where the agent will be injected.
            substrate: ExperimentStore.computeSubstrate,
            appVersion: SteerLabVersion.current,
            // WHICH bytes were injected, not merely which recipe they claim.
            vectorArtifactHash: artifactHash,
            promotionKey: key)

        let variant = ModelVariantArtifact(
            name: variantName,
            baseModelID: manifest.modelID,
            baseRevision: manifest.modelRevision,
            injections: [
                .init(concept: concept, vectorArtifactID: artifactReference,
                      layer: layer, alpha: alpha)
            ],
            bandWidth: bandWidth,
            alphaInNormUnits: alphaInNormUnits,
            promptMode: manifest.promptMode?.rawValue ?? "chatAssistant",
            qwenThinkingEnabled: manifest.qwenThinkingEnabled ?? false,
            temperature: manifest.temperature,
            // A newborn agent's identity is BARE (2026-08-24 ruling). This
            // used to copy `manifest.systemPrompt`, which made the study's
            // deployment frame look like the agent's persona: the copy then
            // travelled with the agent into other studies, and — once
            // effective prompts compose rather than replace — would have been
            // concatenated with the new study's own frame, doubling it. The
            // frame belongs to the study and is applied at run time; a
            // persona is something a researcher gives the agent deliberately.
            // Empty here means the artifact carries neither `systemPrompt`
            // nor `systemPromptHash` (ModelVariantArtifact's initializer).
            // Server twin: `promote.py`'s `system_prompt=None`.
            systemPrompt: "",
            promotion: promotion)
        let record = try ModelVariantStore.save(variant)
        log(
            "promoted '\(concept)' → agent '\(variant.name)' "
                + "(L\(layer), α\(alpha), \(promotedBy)) → \(record.url.path)")
        return record
    }

    /// Deterministic identity of a promotion REQUEST — the idempotency key.
    ///
    /// Built from the same canonical JSON as the server's
    /// `promote.promotion_key`, so the two engines agree on when two
    /// promotions are the same promotion. Deliberately EXCLUDES the
    /// timestamp: a retry is the same promotion even though it happens later.
    public static func promotionKey(
        experiment: String,
        experimentHash: String,
        concept: String,
        sweepRun: String?,
        layer: Int,
        alpha: Double,
        vectorArtifactID: String,
        promotedBy: String,
        agentName: String,
        vectorArtifactHash: String? = nil
    ) -> String {
        // Hand-built canonical form: key order fixed, separators compact, to
        // match Python's json.dumps(sort_keys=True, separators=(",", ":"),
        // ensure_ascii=False) — the recipe-identity house convention.
        // String escaping goes through `RecipeIdentity.jsonString` (raw
        // UTF-8, control characters escaped) rather than a local helper: a
        // local quote() that escaped only backslash and double-quote agreed
        // with Python on ASCII and silently diverged the moment a concept
        // or agent name carried an accent, defeating the "an imported
        // server agent is not re-minted" idempotency guarantee (2026-07-27,
        // the row-hash escaping lesson repeated).
        func quote(_ value: String) -> String {
            RecipeIdentity.jsonString(value)
        }
        let canonical = [
            "\"agentName\":\(quote(agentName))",
            "\"concept\":\(quote(concept))",
            "\"experiment\":\(quote(experiment))",
            "\"experimentHash\":\(quote(experimentHash))",
            "\"promotedBy\":\(quote(promotedBy))",
            "\"sweepRun\":\(sweepRun.map(quote) ?? "null")",
            "\"vectorArtifactHash\":\(vectorArtifactHash.map(quote) ?? "null")",
            "\"vectorArtifactID\":\(quote(vectorArtifactID))",
            "\"winningCell\":{\"alpha\":\(canonicalNumber(alpha)),"
                + "\"layer\":\(layer)}",
        ].joined(separator: ",")
        return ExperimentStore.sha256Hex(Data(("{" + canonical + "}").utf8))
    }

    /// Shortest round-trip decimal, matching Python's `repr`/`json.dumps`.
    ///
    /// Both languages emit the shortest representation that round-trips, so
    /// the two engines agree on the canonical form and therefore on the
    /// promotion key — which matters because a server-minted agent imported
    /// onto the Mac must not be re-minted as a duplicate by a Swift promote
    /// of the same request. `PromotionKeyParityTests` pins representative
    /// alphas against the strings the Python twin produces.
    static func canonicalNumber(_ value: Double) -> String {
        "\(value)"
    }

    /// SHA-256 over the artifact's tensor bytes — WHICH bytes were injected,
    /// not merely which recipe they claim. Nil when unreadable (an artifact
    /// that cannot be read will fail later, loudly, at load).
    static func artifactContentHash(_ artifact: VectorArtifact) -> String? {
        let url = artifact.directory
            .appending(component: "\(artifact.name).safetensors")
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe)
        else { return nil }
        return ExperimentStore.sha256Hex(data)
    }

    /// Evidence that a sweep RAN for this concept — the manual-override gate,
    /// and the criterion path's fallback when the manifest was frozen at
    /// sweep time (no stamped `-recommended` condition exists).
    /// The newest sweep run directory (via `SweepRunCatalog` discovery, so
    /// the matching/ordering rule is the Optimizations surface's rule) whose
    /// `recommendations.json` carries an entry for the concept: either full
    /// selection provenance or the failure message the sweep recorded.
    /// The `<concept>-recommended` condition's named sweep run must carry
    /// the final completion marker (`recommendations.json`) and a
    /// recommendation matching the condition's cell — a condition is a
    /// PROJECTION of a run, never evidence in itself — and the RUN's entry
    /// is returned as the certificate source, so drifted or edited manifest
    /// provenance can never certify inaccurate claims (rounds 3–4).
    /// FAIL-CLOSED: no stamped sweepRun refuses; the run name must be a
    /// plain basename. Server twin: `_condition_run_evidence`; refusal
    /// strings are the cross-engine contract.
    static func conditionRunEvidence(
        concept: String,
        selection: ExperimentManifest.SelectionProvenance?,
        layer: Int, alpha: Double,
        // WP0 step 7: carried only so the typed refusals below can name a
        // RUNNABLE repair. Nothing else reads it, and no refusal prose
        // changed — the cross-engine refusal strings are unaffected.
        experimentName: String = "<name>",
        log: (String) -> Void
    ) throws -> ExperimentManifest.SelectionProvenance {
        guard let stamped = selection, !stamped.sweepRun.isEmpty else {
            throw ExperimentError.refusing(
                .promotionEvidence,
                "promote: condition '\(concept)-recommended' carries "
                    + "no sweepRun stamp — hand-written or legacy provenance "
                    + "is not criterion evidence; re-run the sweep (which "
                    + "stamps it), or promote with an explicit cell override "
                    + "(stamped manualOverride)",
                repair: Self.sweepThenPromoteRepair(experimentName, concept))
        }
        let sweepRun = stamped.sweepRun
        try Self.requirePlainRunName(
            sweepRun, source: "condition '\(concept)-recommended'")
        let runDirectory = ExperimentStore.runsDirectory
            .appending(component: sweepRun)
        var isDirectory: ObjCBool = false
        guard
            FileManager.default.fileExists(
                atPath: runDirectory.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw ExperimentError.refusing(
                .promotionEvidence,
                "promote: condition '\(concept)-recommended' names "
                    + "sweep run '\(sweepRun)' which is not in runs/ — fetch "
                    + "the sweep's results (a bundle-submitted sweep returns "
                    + "them in its results tarball) or re-run the sweep "
                    + "before promoting",
                repair: Self.sweepThenPromoteRepair(experimentName, concept))
        }
        let markerURL = runDirectory.appending(
            component: "recommendations.json")
        guard
            let data = try? Data(contentsOf: markerURL),
            let recommendations = try? SweepRunCatalog.parseRecommendations(data)
        else {
            throw ExperimentError.refusing(
                .promotionEvidence,
                "promote: sweep run '\(sweepRun)' has no readable "
                    + "recommendations.json — that sweep never completed, so "
                    + "condition '\(concept)-recommended' is a projection "
                    + "without evidence; re-run the sweep",
                repair: Self.sweepThenPromoteRepair(experimentName, concept))
        }
        guard case .selected(let provenance)? = recommendations[concept]
        else {
            throw ExperimentError.refusing(
                .promotionEvidence,
                "promote: sweep run '\(sweepRun)' carries no "
                    + "successful recommendation for '\(concept)' — "
                    + "condition '\(concept)-recommended' does not match its "
                    + "own evidence; re-run the sweep",
                repair: Self.sweepThenPromoteRepair(experimentName, concept))
        }
        guard
            provenance.winningCell.layer == layer,
            abs(provenance.winningCell.alpha - alpha) <= 1e-12
        else {
            throw ExperimentError.refusing(
                .promotionEvidence,
                "promote: sweep run '\(sweepRun)' recommends "
                    + "L\(provenance.winningCell.layer) "
                    + "α\(provenance.winningCell.alpha) for '\(concept)' but "
                    + "condition '\(concept)-recommended' pins L\(layer) "
                    + "α\(alpha) — the manifest condition is stale; re-run "
                    + "the sweep or promote with an explicit cell override",
                repair: Self.sweepThenPromoteRepair(experimentName, concept))
        }
        try Self.requireEntryNamesItsRun(
            provenance, sweepRun: sweepRun, concept: concept)
        if stamped != provenance {
            // Same cell, drifted provenance: not a refusal — the identity
            // check passed — but the certificate copies the RUN's block,
            // and the drift deserves a loud line.
            log("⚠︎ condition '\(concept)-recommended' stamped provenance "
                + "differs from sweep run '\(sweepRun)' — the run's "
                + "recommendation is the certificate source")
        }
        return provenance
    }

    /// EVERY path that appends a caller- or manifest-provided sweep-run
    /// name onto runs/ funnels through here (round 5, P1: the condition
    /// path validated, but the pinned contract appended a raw value). A
    /// run name is a plain directory basename — never a path. Server
    /// twin: `_require_plain_run_name`.
    /// THE repair for a promotion with no usable sweep evidence (WP0 step 7).
    ///
    /// `promote` refuses in nine shapes — no recommendation, an unreadable
    /// sweep run, a stale pinned cell, an epoch mismatch — and every one of
    /// them is repaired by the same two commands. Dry run #1 hit this refusal
    /// as `failed/70` with "read the reason and repair the named input", which
    /// is true and useless: the reason names a sweep the agent has no reason to
    /// believe it can run.
    static func sweepThenPromoteRepair(_ experiment: String, _ concept: String) -> String {
        "steerlab-cli experiment sweep \(experiment) && "
            + "steerlab-cli experiment promote \(experiment) \(concept)"
    }

    /// The repair when the vector artifact the promotion would mint FROM is
    /// missing or hashes differently than its pin claims.
    static func extractThenPromoteRepair(
        _ experiment: String, _ concept: String
    ) -> String {
        "steerlab-cli experiment extract \(experiment) && "
            + "steerlab-cli experiment sweep \(experiment) && "
            + "steerlab-cli experiment promote \(experiment) \(concept)"
    }

    static func requirePlainRunName(
        _ sweepRun: String, source: String
    ) throws {
        guard !sweepRun.isEmpty else {
            throw ExperimentError.refusing(
                .promotionEvidence,
                "promote: \(source) carries no sweep run name",
                repair: Self.sweepThenPromoteRepair("<name>", "<concept>"))
        }
        if sweepRun.contains("/") || sweepRun.contains("\\")
            || sweepRun == "." || sweepRun == ".."
        {
            throw ExperimentError.refusing(
                .promotionEvidence,
                "promote: \(source) names sweep run '\(sweepRun)', "
                    + "which is not a plain run name — refusing to read "
                    + "outside runs/",
                repair: Self.sweepThenPromoteRepair("<name>", "<concept>"))
        }
    }

    /// Self-identity (round 5, P2): the recommendation read from directory
    /// `sweepRun` must STAMP that same run — an entry claiming another run
    /// means the directory was copied or its recommendations.json edited,
    /// and a certificate naming run B from bytes read under run A is
    /// exactly the provenance corruption the gate exists to stop. Server
    /// twin: `_require_entry_names_its_run`.
    static func requireEntryNamesItsRun(
        _ provenance: ExperimentManifest.SelectionProvenance,
        sweepRun: String, concept: String
    ) throws {
        guard provenance.sweepRun == sweepRun else {
            throw ExperimentError.refusing(
                .promotionEvidence,
                "promote: sweep run '\(sweepRun)' carries a "
                    + "recommendation for '\(concept)' stamped for run "
                    + "'\(provenance.sweepRun)' — the directory's evidence "
                    + "does not name itself (copied or edited run?); not "
                    + "promotable evidence",
                repair: Self.sweepThenPromoteRepair("<name>", concept))
        }
    }

    static func newestSweepEvidence(
        experiment: String, concept: String
    ) throws -> (runName: String, recommendation: SweepRunCatalog.Recommendation)? {
        let directories = SweepRunCatalog.sweepRunDirectories(experiment: experiment)
        for directory in directories.reversed() {
            guard
                let run = try? SweepRunCatalog.load(directory: directory),
                let recommendation = run.recommendations[concept]
            else { continue }
            // Self-identity applies to the AMBIENT fallback too (review
            // 2026-08-03 round 6, P1): a directory whose entry stamps
            // another run refuses LOUDLY — silently skipping to an older
            // run would promote from different evidence while hiding the
            // corruption. Failure entries carry no stamp to check.
            if case .selected(let provenance) = recommendation {
                try Self.requireEntryNamesItsRun(
                    provenance, sweepRun: run.runName, concept: concept)
            }
            return (run.runName, recommendation)
        }
        return nil
    }

    /// Newest persisted vector artifact matching the experiment's FULL
    /// recipe identity (`RecipeIdentity` — the pinned cross-engine canonical
    /// form covering concept, model, revision, method, stimulus hash,
    /// reading position, neutral projection, norm denominator + corpus hash,
    /// and the complete grand-mean population).
    ///
    /// Match order per candidate: (a) the artifact's stamped
    /// `recipeIdentityHash` when present; else (b) the identity computed
    /// from its sidecar fields when ALL of them are provable; else (c) the
    /// artifact is refused with the exact missing fields named — never a
    /// silent fallback to a partial-field match. Substrate stays OUTSIDE the
    /// identity hash but remains its own criterion, exactly as before: a
    /// foreign-substrate artifact never matches, an unstamped-legacy one may.
    static func matchingVectorArtifact(
        manifest: ExperimentManifest, ref: ExperimentManifest.ConceptRef,
        experimentName: String,
        pinnedArtifactID: String? = nil
    ) throws -> (artifact: VectorArtifact, recipeIdentityHash: String) {
        let required = try RecipeIdentity.required(manifest: manifest, ref: ref)
        let requiredHash = RecipeIdentity.hash(required)

        var matches: [VectorArtifact] = []
        var unprovable: [(id: String, missing: [String])] = []
        // Per-candidate ACTIONABLE refusal detail: (artifact id, the
        // differing canonical fields with both values) — never a bare
        // "different identity" counter (the 2026-07-14 lesson: an opaque
        // refusal hid a one-field revision mismatch).
        var different: [(id: String, fields: String)] = []
        func noteDifferent(
            _ artifact: VectorArtifact, stamped: String?
        ) {
            let candidate = RecipeIdentity.candidate(sidecar: artifact.sidecar)
            let diffs = candidate.components.map {
                RecipeIdentity.diffFields(manifest: required, artifact: $0)
            } ?? []
            if !diffs.isEmpty {
                different.append((artifact.id, diffs.joined(separator: ", ")))
            } else if let stamped, candidate.components != nil {
                // The recorded fields hash to the required identity but the
                // stamp disagrees: a stale/corrupt stamp, and the stamp is
                // authoritative.
                different.append((
                    artifact.id,
                    "stamped recipeIdentityHash \(stamped.prefix(12))… "
                        + "contradicts its own recorded fields — re-extract "
                        + "to restamp"))
            } else {
                different.append((
                    artifact.id,
                    "stamped recipeIdentityHash \(stamped?.prefix(12) ?? "?")… "
                        + "≠ required \(requiredHash.prefix(12))… and the "
                        + "recorded fields cannot be independently read"))
            }
        }
        for artifact in VectorCatalog.scan() {
            let sidecar = artifact.sidecar
            guard sidecar.concept == ref.name,
                sidecar.modelID == manifest.modelID
            else { continue }
            // Keyed on the WORKSPACE's compute substrate, not this engine's
            // constant. A cluster workspace's vectors are extracted on the
            // cluster; treating them as foreign made every artifact in such a
            // workspace invisible to promotion (2026-07-26).
            if let substrate = sidecar.substrate,
                substrate != ExperimentStore.computeSubstrate
            {
                continue
            }
            if let stamped = sidecar.recipeIdentityHash {
                if stamped == requiredHash {
                    matches.append(artifact)
                } else {
                    noteDifferent(artifact, stamped: stamped)
                }
                continue
            }
            let candidate = RecipeIdentity.candidate(sidecar: sidecar)
            if let components = candidate.components {
                if RecipeIdentity.hash(components) == requiredHash {
                    matches.append(artifact)
                } else {
                    noteDifferent(artifact, stamped: nil)
                }
            } else {
                unprovable.append((artifact.id, candidate.missingFields))
            }
        }

        // A PINNED artifact is selected by identity, never by recency.
        // Pinning chooses WHICH artifact; it does not waive WHETHER it
        // matches the recipe — a pinned artifact that failed the identity
        // check above is absent from `matches` and refuses below.
        if let pinnedArtifactID {
            let wanted = ArtifactIdentity.canonical(pinnedArtifactID)
            if let pinned = matches.first(
                where: { ArtifactIdentity.canonical($0.id) == wanted })
            {
                return (pinned, requiredHash)
            }
            throw ExperimentError.refusing(
                .artifactPin,
                "promote: the pinned contract names vector artifact "
                    + "'\(pinnedArtifactID)' for '\(ref.name)', but no "
                    + "artifact at that path matches this experiment's recipe "
                    + "identity (\(requiredHash.prefix(12))…) on this "
                    + "substrate — it was moved, re-extracted under different "
                    + "options, or never existed",
                repair: Self.extractThenPromoteRepair(experimentName, ref.name))
        }

        // Every match above carries the SAME identity hash, i.e. the same
        // recipe — so "newest wins" here is a freshness tie-break among
        // interchangeable artifacts, never a choice between different
        // recipes (those were counted, not ranked). Run directories are
        // timestamp-named: lexicographic order is creation order.
        if let newest = matches
            .sorted(by: { $0.directory.lastPathComponent < $1.directory.lastPathComponent })
            .last
        {
            return (newest, requiredHash)
        }

        var reason =
            "no extraction artifact for '\(ref.name)' matches this "
            + "experiment's full recipe (identity \(requiredHash.prefix(12))…, "
            + "model \(manifest.modelID), stimulus hash "
            + "\(ref.stimulusSetHash.prefix(12))…) on this substrate"
        var details: [String] = []
        let shown = 5  // keep the refusal readable in artifact-heavy workspaces
        for entry in different.prefix(shown) {
            details.append(
                "candidate '\(entry.id)' carries a DIFFERENT recipe identity "
                    + "(never matched by recency) — \(entry.fields)")
        }
        if different.count > shown {
            details.append(
                "… and \(different.count - shown) more candidate(s) with a "
                    + "different recipe identity")
        }
        for entry in unprovable {
            details.append(
                "artifact '\(entry.id)' cannot prove recipe fields "
                    + "[\(entry.missing.joined(separator: ", "))] — re-extract "
                    + "to stamp the full recipe")
        }
        if !details.isEmpty {
            reason += ": " + details.joined(separator: "; ")
        }
        reason += " — run 'experiment extract \(experimentName)' first"
        throw ExperimentError.refusing(
            .artifactPin, reason,
            repair: Self.extractThenPromoteRepair(experimentName, ref.name))
    }
}
