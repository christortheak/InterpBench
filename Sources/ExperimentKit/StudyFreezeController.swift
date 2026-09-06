import Foundation
import Observation

/// Freeze/readiness presentation and coordination. Policies and filesystem admission stay in their stores.
@Observable @MainActor
public final class StudyFreezeController {
    public init() {}
    /// Read-only freeze gate summary for the selected draft (nil for frozen/
    /// completed studies); recomputed on every refresh.
    public private(set) var freezeReadiness: ExperimentStore.FreezeReadiness?
    /// The server's verbatim gate refusal from the last remote-freeze attempt
    /// (the FastAPI `detail` — the server's self-naming "cannot freeze …"
    /// message). Rendered by the view in the same idiom as an unmet local
    /// freeze gate; cleared on selection change and on the next attempt.
    public private(set) var remoteFreezeGateFailure: String?
    /// Non-blocking advisories the server returned WITH a successful remote
    /// freeze (the additive "advisories" response key) — rendered like local
    /// readiness advisories, cross-substrate ones prominently.
    public private(set) var remoteFreezeAdvisories: [String] = []
    /// The manifest-identity BLOCK from the last remote-freeze attempt: the
    /// server's same-named copy is not the document on screen (field-level
    /// mismatch summary + remedy), or the identity was unverifiable on an
    /// unpaired workspace. Rendered prominently; cleared on selection change
    /// and at the next attempt. See `FreezeRouting.remoteFreezePrecheck`.
    public private(set) var remoteFreezeIdentityWarning: String?
    /// Informational identity note when the freeze PROCEEDED anyway: the
    /// server-only-copy statement (name/status/canonical-body hash) or the
    /// paired-workspace could-not-verify note.
    public private(set) var remoteFreezeIdentityNote: String?
    /// The identity BLOCK's one-click remedy is available (2026-07-21
    /// incident, part 3): the mismatch case with a LOCAL DRAFT on screen —
    /// "Update the server's copy" pushes it as the server's draft and
    /// re-runs the identity check. Rule in
    /// `FreezeRouting.canOfferServerDraftSync`; frozen local manifests
    /// never push (duplicate-never-edit).
    public private(set) var remoteFreezeCanSyncDraft = false
    /// True while the push+recheck round-trip runs (disables the button).
    public private(set) var isSyncingServerDraft = false
    /// Decision-time coherence check for a SERVER-routed freeze on a
    /// workspace-PAIRED server: the shared runs/ tree evaluated from the
    /// server's perspective (`WorkspaceScoping.serverSubstrate`), so
    /// validate-locally-then-freeze-on-the-server warns BEFORE the click,
    /// not only in the server's own freeze-time advisory. nil when the
    /// workspace is unpaired (the local tree says nothing about the server's
    /// evidence), the study is not a draft, or the evidence is coherent.
    /// Recomputed on refresh, like `freezeReadiness`.
    public private(set) var serverFreezeCrossSubstrateAdvisory: String?

    public private(set) var isFreezingOnServer = false
    @ObservationIgnored var presentation = StudyFreezePresentation()
    @ObservationIgnored private var activeOperation = UUID()

    @ObservationIgnored private var reviewedServer: (name: String, digest: String, origin: RemoteJobOrigin?)?

    public func resetSelection() {
        reviewedServer = nil
        activeOperation = UUID()
        remoteFreezeGateFailure = nil
        remoteFreezeAdvisories = []
        remoteFreezeIdentityWarning = nil
        remoteFreezeIdentityNote = nil
        remoteFreezeCanSyncDraft = false
        isFreezingOnServer = false
        isSyncingServerDraft = false
    }

    private func beginOperation() -> UUID {
        activeOperation = UUID()
        isFreezingOnServer = false
        isSyncingServerDraft = false
        return activeOperation
    }

    private func note(_ message: String, severity: PanelNotice.Severity = .info) {
        presentation.note(message, severity)
    }
    private func refresh() { presentation.refresh() }

    public func refreshReadiness(
        manifest: ExperimentManifest?, violations: [String], runSubstrate: String,
        serverPaired: Bool
    ) {
        guard let manifest, manifest.status == .draft else {
            freezeReadiness = nil
            serverFreezeCrossSubstrateAdvisory = nil
            return
        }
        freezeReadiness = ExperimentStore.freezeReadiness(
            for: manifest, violations: violations, runSubstrate: runSubstrate)
        serverFreezeCrossSubstrateAdvisory = nil
        if serverPaired,
            !manifest.concepts.isEmpty || !manifest.conditions.isEmpty
                || !manifest.variantConditions.isEmpty
        {
            serverFreezeCrossSubstrateAdvisory = ExperimentStore.crossSubstrateValidationAdvisory(
                for: manifest, perspective: WorkspaceScoping.serverSubstrate)
        }
    }

    public func freeze(name: String?, runSubstrate: String) {
        guard let name else { return }
        do {
            // Mac-authority mode: a local freeze under server Compute
            // matches evidence for the RUN substrate (the server) — an
            // imported server validate run counts; this engine's own
            // evidence does not, because the study will not run here.
            let frozen = try ExperimentStore.freeze(
                name: name, runSubstrate: runSubstrate)
            refresh()
            var line =
                "frozen @ \(frozen.freezeHash?.prefix(12) ?? "?")… "
                + "(git \(frozen.gitCommit?.prefix(8) ?? "uncommitted"))"
            if runSubstrate != ExperimentStore.evidenceSubstrate {
                line +=
                    " — evidence matched for the run substrate "
                    + "\(runSubstrate); submit the frozen study to the server "
                    + "as a bundle to run it"
            }
            note(line, severity: .success)
        } catch {
            note(
                "Freeze did not complete — the study is unchanged and still "
                    + "a draft; the detail names the gate or pin that "
                    + "stopped it. Details: \(error)",
                severity: .error)
        }
    }

    /// Freeze the selected study THROUGH the active server's gated freeze
    /// authority (`POST /api/authoring/{name}/freeze`): the gates are
    /// evaluated server-side against SERVER-substrate evidence and the
    /// manifest is stamped `frozenBy: "server"` — completing the
    /// substrate-coherent order server-extract → server-validate →
    /// server-freeze → server-run without leaving the app. Deliberately no
    /// force parameter (parity with local freeze: forcing requires the CLI).
    /// On a workspace-PAIRED server the frozen manifest lands in the shared
    /// tree and `refresh()` re-reads it — verify() then re-checks it through
    /// the frozenBy:"server" freeze-canonical path; on an unpaired server the
    /// frozen manifest lives in the SERVER's tree and the status line says so.
    func freezeOnServer(
        request: StudyFreezeRequest, transport: StudyFreezeTransport,
        isCurrent: () -> Bool = { true }
    ) async {
        let name = request.name
        let operation = beginOperation()
        isFreezingOnServer = true
        defer { if operation == activeOperation { isFreezingOnServer = false } }
        func current() -> Bool { operation == activeOperation && isCurrent() && !Task.isCancelled }
        guard current() else { return }
        remoteFreezeGateFailure = nil
        remoteFreezeAdvisories = []
        remoteFreezeIdentityWarning = nil
        remoteFreezeIdentityNote = nil
        remoteFreezeCanSyncDraft = false
        let substrate = request.substrate
        // Residency + lifecycle preflight in one read (`GET
        // /api/experiment/{name}`): freeze stamps the SERVER-RESIDENT copy
        // only, and an already-frozen server copy refuses here with a clear
        // message instead of a job-side 400. A transient fetch failure
        // proceeds — the freeze call's own refusal is the backstop.
        var serverReportedStatus: String?
        do {
            let remoteStatus = try await transport.status(name)
            guard current() else { return }
            presentation.residency(name, true)
            serverReportedStatus = remoteStatus
            if let remoteState = remoteStatus, remoteState != "draft" {
                note(
                    "'\(name)' on \(substrate) is already \(remoteState) — "
                        + "duplicate to iterate", severity: .info)
                return
            }
        } catch let error as ClusterClient.ClientError {
            guard current() else { return }
            if case .badResponse(404, _) = error {
                presentation.residency(name, false)
                note(
                    "study '\(name)' is not in \(substrate)'s workspace — "
                        + "freeze stamps the server-resident copy only. Pair the "
                        + "server to this workspace (serve --root <workspace>), or "
                        + "switch Compute to Local (MLX) to freeze the local copy", severity: .info)
                return
            }
        } catch {}
        guard current() else { return }
        // Manifest-identity guard: the freeze call names a study; the server
        // freezes WHICHEVER same-named copy it holds. Verify that copy IS the
        // document on screen (same-engine canonicalization of both documents,
        // volatile freeze stamps excluded — deliberately NOT an invented
        // cross-engine byte-canonical hash) before anything is stamped.
        // Block/proceed rules live in `FreezeRouting.remoteFreezePrecheck`.
        let localData = request.localData
        reviewedServer = nil
        let identity = await remoteFreezeManifestIdentity(
            transport: transport, name: name, substrate: substrate, localData: localData,
            serverReportedStatus: serverReportedStatus, recordRead: { bytes in
                guard current() else { return }
                self.reviewedServer = (name, ManifestFileTransaction.digest(bytes), transport.origin)
            })
        guard current() else { return }
        let precheck = FreezeRouting.remoteFreezePrecheck(
            identity: identity,
            study: name,
            serverLabel: substrate,
            workspacePaired: request.workspacePaired)
        guard precheck.proceed else {
            remoteFreezeIdentityWarning = precheck.message
            // The one-click remedy (2026-07-21 incident, part 3): a
            // mismatched LOCAL DRAFT can be pushed as the server's draft
            // copy — the view offers "Update the server's copy".
            remoteFreezeCanSyncDraft = FreezeRouting.canOfferServerDraftSync(
                identity: identity, localIsDraft: request.localIsDraft)
            note(
                "freeze on \(substrate) refused before submission — "
                    + "its copy of '\(name)' could not be confirmed as the "
                    + "manifest on screen", severity: .error)
            return
        }
        guard (try? ExperimentRepository(workspaceRoot: request.workspaceRoot).snapshot(name: name).data) == localData else {
            let warning =
                "The local manifest changed during the identity check — review it and click Freeze again."
            remoteFreezeIdentityWarning = warning
            note(warning, severity: .warning)
            return
        }
        remoteFreezeIdentityNote = precheck.message
        note("freezing '\(name)' on \(substrate)…", severity: .info)
        do {
            let result = try await transport.freeze(name)
            guard current() else { return }
            remoteFreezeAdvisories = result.advisories
            // Paired server: the frozen manifest was written into the shared
            // tree — re-read it so the panel shows frozen/frozenBy:"server"
            // and local verify() re-checks via freeze-canonical.json.
            refresh()
            var line =
                "frozen on \(substrate) @ "
                + "\(result.manifest.freezeHash?.prefix(12) ?? "?")… "
                + "(frozenBy: server"
                + (result.manifest.gitCommit.map { ", git \($0.prefix(8))" } ?? "")
                + ")"
            if (try? ExperimentRepository(workspaceRoot: request.workspaceRoot).load(name: name))?.status != .frozen {
                line +=
                    " — the frozen manifest is in \(substrate)'s workspace; "
                    + "the local copy is untouched"
            }
            note(line, severity: .success)
        } catch let error as ClusterClient.ClientError {
            guard current() else { return }
            // The server's gate refusal IS the actionable text — keep it
            // verbatim and render it like an unmet local freeze gate.
            let unwrapped = ClusterClient.unwrappingDetail(error)
            if case .badResponse(_, let detail) = unwrapped {
                remoteFreezeGateFailure = detail
            }
            note(
                "\(substrate) declined to freeze — a gate or pin on the "
                    + "server side is unmet; the study is unchanged. "
                    + "Details: \(unwrapped)",
                severity: .error)
        } catch {
            guard current() else { return }
            note(
                "Freeze on \(substrate) did not complete — the study is "
                    + "unchanged; check the server connection and try again. "
                    + "Details: \(error)",
                severity: .error)
        }
    }

    /// Fetch-and-compare step of the remote-freeze identity guard: the
    /// server's raw manifest body vs the LOCAL manifest document (the file
    /// backing what the Studies list displays). Pure comparison lives in
    /// `ExperimentStore.compareManifestDocuments`; pure block/proceed rules
    /// in `FreezeRouting.remoteFreezePrecheck` — this only does the IO.
    private func remoteFreezeManifestIdentity(
        transport: StudyFreezeTransport, name: String, substrate: String, localData: Data?,
        serverReportedStatus: String?, recordRead: (Data) -> Void
    ) async -> FreezeRouting.RemoteManifestIdentity {
        do {
            let serverBody = try await transport.manifestBody(name)
            recordRead(serverBody)
            guard let localData else {
                return .localMissing(
                    serverStatus: serverReportedStatus,
                    canonicalBodyHash: ExperimentStore.canonicalManifestBodyHash(
                        serverBody))
            }
            switch ExperimentStore.compareManifestDocuments(
                local: localData, server: serverBody)
            {
            case .equal:
                return .verifiedEqual
            case .different(let fields):
                return .mismatch(fields)
            case .unparseable:
                return .unverifiable("a manifest body is not a JSON object")
            }
        } catch let error as ClusterClient.ClientError {
            if case .badResponse(404, _) = error {
                return .unverifiable(
                    "\(substrate) does not serve the manifest body — "
                        + "older server without GET /api/experiment/{name}/manifest")
            }
            return .unverifiable(
                "manifest body fetch failed: \(ClusterClient.unwrappingDetail(error))")
        } catch {
            return .unverifiable("manifest body fetch failed: \(error)")
        }
    }

    /// "Update the server's copy" (2026-07-21 incident, part 3): push the
    /// CURRENT local manifest document as the active server's DRAFT copy
    /// (`PUT /api/experiment/{name}/manifest`), then RE-RUN the identity
    /// check and report verified-equal — or the remaining difference,
    /// honestly. Draft manifests only, on both sides: this method refuses a
    /// frozen local document, and the server refuses to overwrite a frozen
    /// copy (freeze firewall — duplicate to iterate). Nothing is
    /// auto-frozen afterwards: the researcher clicks Freeze again with the
    /// identity verified.
    func pushManifest(
        request: StudyFreezeRequest, transport: StudyFreezeTransport,
        isCurrent: () -> Bool = { true }
    ) async {
        let name = request.name
        guard request.localIsDraft else {
            note(
                "only a DRAFT manifest can be pushed as the server's copy — "
                    + "frozen studies are read-only; duplicate to iterate", severity: .info)
            return
        }
        guard let localData = request.localData else {
            note(
                "could not read the local manifest file for '\(name)' — "
                    + "nothing was pushed", severity: .error)
            return
        }
        let substrate = request.substrate
        let operation = beginOperation()
        func current() -> Bool { operation == activeOperation && isCurrent() && !Task.isCancelled }
        guard current() else { return }
        isSyncingServerDraft = true
        defer { if operation == activeOperation { isSyncingServerDraft = false } }
        do {
            note("updating \(substrate)'s copy of '\(name)'…", severity: .info)
            guard let reviewed = reviewedServer, reviewed.name == name,
                reviewed.origin == transport.origin
            else {
                note("Check the server manifest again before syncing; its reviewed file version is unavailable for this connection.", severity: .warning)
                return
            }
            let result = try await transport.replace(name, localData, reviewed.digest)
            guard current() else { return }
            // Merge semantics (2026-08-06): the server KEEPS auto-pins the
            // pushed document omitted (its resolved model revision, sweep
            // projections) and names them in `preserved`. Adopt the
            // revision into the local draft HERE — same rule as
            // EvidenceRevisionAdoption: completing the researcher's own
            // declared state — so the identity re-check below converges
            // instead of re-flagging a difference the researcher never
            // authored. Anything else preserved is reported honestly.
            adoptPreservedServerPins(
                result.preserved, request: request,
                substrate: substrate)
            // Re-run the SAME identity check the freeze precheck uses — the
            // affordance's claim is "now verified equal", never "pushed, so
            // it must match".
            let identity = await remoteFreezeManifestIdentity(
                transport: transport, name: name, substrate: substrate,
                localData: try? ExperimentRepository(workspaceRoot: request.workspaceRoot).snapshot(name: name).data,
                serverReportedStatus: result.status, recordRead: { bytes in
                    guard current() else { return }
                    self.reviewedServer = (name, ManifestFileTransaction.digest(bytes), transport.origin)
                })
            guard current() else { return }
            let outcome = FreezeRouting.serverDraftSyncOutcome(
                recheck: identity,
                study: name,
                serverLabel: substrate,
                canonicalBodyHash: result.canonicalBodyHash)
            if outcome.resolved {
                remoteFreezeIdentityWarning = nil
                remoteFreezeCanSyncDraft = false
                remoteFreezeIdentityNote = outcome.message
                note(outcome.message, severity: .success)
            } else {
                remoteFreezeIdentityWarning = outcome.message
                note(outcome.message, severity: .error)
            }
        } catch let error as ClusterClient.ClientError {
            guard current() else { return }
            // The server's refusal (e.g. its copy is frozen) is the
            // actionable text — verbatim.
            let unwrapped = ClusterClient.unwrappingDetail(error)
            note(
                "\(substrate) declined the manifest push — its copy is "
                    + "unchanged. Details: \(unwrapped)",
                severity: .error)
        } catch {
            guard current() else { return }
            note(
                "Manifest push to \(substrate) did not complete — its copy "
                    + "may be unchanged; check the connection and try again. "
                    + "Details: \(error)",
                severity: .error)
        }
    }

    /// The client half of the push-merge rule (2026-08-06): when the server
    /// KEPT auto-pins the pushed document omitted, adopt the model revision
    /// into the still-unpinned local draft (the same completing-declared-
    /// intent rule as `EvidenceRevisionAdoption`) and name everything else
    /// loudly — preserved sweep projections come home through sweep-run
    /// discovery (`SweepConditionAdoption`), not through this response,
    /// which carries only their names.
    private func adoptPreservedServerPins(
        _ preserved: ClusterClient.RemoteManifestReplaceResult.PreservedPins?,
        request: StudyFreezeRequest,
        substrate: String
    ) {
        guard let preserved else { return }
        let name = request.name
        if let revision = preserved.modelRevision, !revision.isEmpty {
            let reviewed = request.localData.flatMap {
                try? DraftAuthoringSnapshot(workspaceRoot: request.workspaceRoot, name: name,
                                            file: ManifestFileSnapshot(data: $0))
            }
            let local = reviewed?.manifest
            if var local, local.modelRevision == nil, local.status == .draft, let reviewed {
                do {
                    local.modelRevision = revision
                    try DraftAuthoringTransaction.replace(local, reviewed: reviewed)
                    refresh()
                    note(
                        "\(substrate) kept its auto-pinned model revision "
                            + "\(revision) (the push omitted one) — adopted "
                            + "into the local draft so both copies agree",
                        severity: .info)
                } catch {
                    note(
                        "\(substrate) kept its auto-pinned model revision "
                            + "\(revision), but adopting it into the local "
                            + "draft failed: \(error)",
                        severity: .error)
                }
            } else if let localRevision = local?.modelRevision,
                localRevision != revision
            {
                note(
                    "\(substrate) kept model revision \(revision), but the "
                        + "local draft pins \(localRevision) — the copies "
                        + "genuinely disagree; re-validate at one revision "
                        + "or duplicate the study",
                    severity: .warning)
            }
        }
        if let conditions = preserved.conditions, !conditions.isEmpty {
            note(
                "\(substrate) kept its sweep-projected condition(s) "
                    + conditions.joined(separator: ", ")
                    + " — the push omitted them; adopt them locally via the "
                    + "sweep run's Adopt Projections before comparing copies",
                severity: .info)
        }
        if preserved.capabilityBattery != nil {
            note(
                "\(substrate) kept its capability-battery pin — the push "
                    + "omitted one; local freeze re-derives its own pin",
                severity: .info)
        }
    }
}
