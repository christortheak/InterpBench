import ExperimentKit
import SwiftUI
import UniformTypeIdentifiers

/// Captured workspace/client remain authoritative even if the underlying app
/// selection changes while transfer, verification or cleanup is in flight.
///
/// Layout (researcher complaint 2026-09-13: misaligned rows, text running
/// off the sheet): a grouped form whose captions wrap, one control group per
/// row, and a minimum width that fits inside the app's window floor. The
/// common spine — stage inputs, bring evidence home, clean up — is always
/// visible; the two multi-job coordination groups (OptVec campaign cells,
/// J-lens fitting shards) are collapsed until needed. Every action and its
/// gating is unchanged from before the relayout.
struct DiagnosticLifecycleSheet: View {
    let root: URL
    let client: ClusterClient?
    let initialJobID: String?
    var initialRequestFile: String? = nil
    @Environment(\.dismiss) private var dismiss
    @State private var requestFile = ""
    @State private var inputArchiveFile = ""
    @State private var inputArchiveHash = ""
    @State private var archiveFile = ""
    @State private var archiveHash = ""
    @State private var jobID = ""
    @State private var receiptHash = ""
    @State private var sourcePlan: String?
    @State private var stagedRequest: JSONValue?
    @State private var gpuOptions: ScientificGPUPlacement?
    @State private var gpuType = ""
    @State private var gpuRequirement: ScientificGPUPlacement.Requirement = .unknown
    @State private var plannedGPUType: String?
    @State private var placementMessage = "Loading the controller’s GPU choices…"
    @State private var gpuReview: [String] = []
    @State private var roundGPUType = ""
    @State private var shardGPUTypes: [Int: String] = [:]
    @State private var pendingShards: [Int] = []
    @State private var executionPlan: String?
    @State private var campaignPlan: String?
    @State private var campaignConfirmed = false
    @State private var roundReview = FittingRoundReview()
    @State private var cleanupPlan: String?
    @State private var confirmation = false
    @State private var output = ""
    @State private var busy = false
    @State private var choosingRequest = true
    @State private var showingImporter = false
    @State private var scopedURLs: [URL] = []
    @State private var showingCampaign = false
    @State private var showingFittingRounds = false

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Stage inputs, collect evidence, clean up").font(.title2)
                Spacer(); Button("Done") { dismiss() }.disabled(busy)
            }
            Text("Local workspace: " + root.path).font(.caption).textSelection(.enabled)
                .lineLimit(1).truncationMode(.middle).help(root.path)
            if let client { Text("Server: " + client.profile.baseURL.absoluteString).font(.caption).textSelection(.enabled) }
            Text("Each step is a separate explicit action: nothing here runs on its own, "
                + "and a reviewed plan is submitted only by its hash.")
                .font(.caption).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Form {
                prepareInputsSection
                evidenceSection
                if let client {
                    campaignSection(client)
                    fittingRoundsSection(client)
                    cleanupSection(client)
                }
            }
            .formStyle(.grouped)
            .disabled(busy)
            if busy { ProgressView("Working on the request. Reading and verifying large input or evidence files can take several minutes.") }
            ScrollView { Text(output).font(.system(.caption, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
                .frame(minHeight: 120)
        }.padding().frame(minWidth: 860, minHeight: 660)
        .interactiveDismissDisabled(busy)
        .onAppear { jobID = initialJobID ?? ""; requestFile = initialRequestFile ?? "" }
        .task {
            refreshRequestKind()
            if let client {
                do {
                    gpuOptions = try await client.scientificGPUPlacement()
                    placementMessage = gpuOptions == nil ? "This server does not advertise GPU choices; the server default will be used." : ""
                } catch { placementMessage = "GPU choices could not be read. The server default remains available; reopen this sheet to retry." }
            }
        }
        .onChange(of: requestFile) { _, _ in refreshRequestKind(); gpuType = ""; executionPlan = nil; gpuReview = [] }
        .onChange(of: gpuType) { _, _ in executionPlan = nil; gpuReview = [] }
        .onChange(of: roundGPUType) { _, _ in roundReview.invalidate() }
        .onChange(of: shardGPUTypes) { _, _ in roundReview.invalidate() }
        .onChange(of: jobID) { _, _ in pendingShards = []; shardGPUTypes = [:]; roundGPUType = "" }
        .onDisappear { scopedURLs.forEach { $0.stopAccessingSecurityScopedResource() } }
        .fileImporter(isPresented: $showingImporter, allowedContentTypes: choosingRequest ? [.json] : [.data]) { result in
            do {
                let url = try result.get(); if url.startAccessingSecurityScopedResource() { scopedURLs.append(url) }
                if choosingRequest { requestFile = url.path; sourcePlan = nil; inputArchiveFile = ""; inputArchiveHash = ""; stagedRequest = nil; executionPlan = nil }
                else { archiveFile = url.path; archiveHash = "" }
            } catch { output = error.localizedDescription }
        }
    }

    // MARK: Sections

    /// Stage a reviewed request on the server: review → package → upload →
    /// plan → submit, the local steps on one row and the server steps on the
    /// next, so five buttons never fight for one line.
    private var prepareInputsSection: some View {
        Section {
            if gpuRequirement == .gpu, let gpuOptions, gpuOptions.available {
                ScientificGPUSelection(options: gpuOptions, selection: $gpuType)
            } else if client != nil, !placementMessage.isEmpty { caption(placementMessage) }
            if client != nil, !requestFile.isEmpty, gpuRequirement == .unknown {
                caption("GPU requirements are not known locally. Review the server plan to see whether a GPU choice applies.")
            }
            ForEach(gpuReview, id: \.self) { caption($0) }
            LabeledContent("Diagnostic request") {
                HStack(spacing: 8) {
                    Text(requestFile.isEmpty ? "none chosen" : requestFile)
                        .foregroundStyle(requestFile.isEmpty ? .secondary : .primary)
                        .lineLimit(1).truncationMode(.middle)
                        .textSelection(.enabled)
                        .help(requestFile.isEmpty
                            ? "a request JSON prepared with the method guide or your agent"
                            : requestFile)
                    Button("Choose request…") { choosingRequest = true; showingImporter = true }
                        .help("pick the request JSON prepared with the method guide "
                            + "or your agent — choosing one starts the steps over")
                }
            }
            LabeledContent("On this Mac") {
                HStack(spacing: 8) {
                    Button("Review inputs") { perform {
                        let result = try await workspace("input-plan", ["requestFile": .string(requestFile)])
                        sourcePlan = string(result, "planSHA256"); show(result)
                    } }.disabled(requestFile.isEmpty)
                        .help("hash every input the request names and show the plan")
                    Button("Package reviewed inputs") { perform {
                        guard let sourcePlan else { return }
                        let path = root.appending(path: ".steerlab/diagnostic-outgoing/" + UUID().uuidString + ".tar.gz").path
                        let result = try await workspace("package", ["requestFile": .string(requestFile), "planSHA256": .string(sourcePlan), "archivePath": .string(path)])
                        inputArchiveFile = path; inputArchiveHash = string(result, "bundleSha256") ?? ""; stagedRequest = nil; executionPlan = nil; show(result)
                    } }.disabled(sourcePlan == nil)
                        .help("write the reviewed inputs into one archive under "
                            + ".steerlab/diagnostic-outgoing/, pinned to the plan hash")
                }
            }
            if let client {
                LabeledContent("On the server") {
                    HStack(spacing: 8) {
                        Button("Upload and stage inputs") { perform {
                            let uploaded = try await client.uploadBundle(URL(filePath: inputArchiveFile))
                            guard uploaded.sha256 == inputArchiveHash else { throw ExperimentError(reason: "Uploaded input hash differs from the reviewed archive.") }
                            let result = try await DiagnosticRemote.stage(path: uploaded.path, sha256: inputArchiveHash, client: client, root: root)
                            if case .object(let object) = result { stagedRequest = object["request"] }
                            executionPlan = nil; show(result)
                        } }.disabled(inputArchiveFile.isEmpty || inputArchiveHash.isEmpty)
                            .help("upload the archive and have the server verify it into "
                                + "an isolated execution copy — the workspace itself is not edited")
                        Button("Review server plan") { perform {
                            guard let stagedRequest else { return }
                            let selectedGPU = gpuRequirement == .gpu && !gpuType.isEmpty ? gpuType : nil
                            let result = try await client.scientificPlan(stagedRequest, gpuType: selectedGPU)
                            plannedGPUType = selectedGPU
                            gpuRequirement = ScientificGPUPlacement.requirement(stagedRequest, serverPlan: result)
                            executionPlan = string(result, "planSHA256"); gpuReview = ScientificGPUPlacement.reviewLines(result); show(result)
                        } }.disabled(stagedRequest == nil)
                            .help("ask the server what the staged request would do and "
                                + "cost — it plans only, nothing runs")
                        Button("Submit reviewed job") { perform {
                            guard let stagedRequest, let executionPlan else { return }
                            self.executionPlan = nil
                            let result = try await client.scientificSubmit(stagedRequest, planSHA256: executionPlan, gpuType: plannedGPUType)
                            jobID = string(result, "jobId") ?? ""; show(result)
                        } }.disabled(executionPlan == nil)
                            .help("submit exactly the reviewed plan by its hash; the job "
                                + "id lands in the field below")
                    }
                }
            }
            caption("Where HTTP transfer is prohibited, use the site's declared transport and the documented stage/import commands. A reviewed plan must be submitted within the same controller session.")
        } header: {
            Text("Stage inputs on the server")
        }
    }

    /// Fetch a finished job's evidence over HTTP, or import an archive that
    /// travelled another way, then verify custody receipts offline.
    private var evidenceSection: some View {
        Section {
            caption("Large evidence exports can take several minutes to prepare before transfer begins. Keep this window open; the archive is verified before import.")
            TextField("Originating job ID", text: $jobID)
                .font(.system(.body, design: .monospaced))
                .help("the server job whose evidence, cells, shards, or output "
                    + "the rows below act on — pre-filled from the selected job")
                .onChange(of: jobID) { _, _ in cleanupPlan = nil; confirmation = false; campaignPlan = nil; campaignConfirmed = false; roundReview.changeJob() }
            if let client {
                LabeledContent("Over HTTP") {
                    Button("Fetch and verify evidence") { perform {
                        let result = try await DiagnosticRemote.fetch(jobID, client: client, root: root)
                        receiptHash = string(result, "receiptSHA256") ?? ""; show(result)
                    } }.disabled(jobID.isEmpty)
                        .help("have the server package this job's output, download it, "
                            + "verify the bytes, and import it with a custody receipt")
                }
            }
            LabeledContent("Transferred archive") {
                HStack(spacing: 8) {
                    Text(archiveFile.isEmpty ? "none chosen" : archiveFile)
                        .foregroundStyle(archiveFile.isEmpty ? .secondary : .primary)
                        .lineLimit(1).truncationMode(.middle)
                        .textSelection(.enabled)
                        .help(archiveFile.isEmpty
                            ? "an evidence archive that reached this Mac by the site's own transport"
                            : archiveFile)
                    Button("Choose transferred archive…") { choosingRequest = false; showingImporter = true }
                        .help("pick an evidence archive that travelled by the site's "
                            + "own transport instead of HTTP")
                }
            }
            HStack(spacing: 8) {
                TextField("Archive SHA-256 from the exporting job", text: $archiveHash)
                    .font(.system(.body, design: .monospaced))
                    .help("the digest the exporting job reported — the import "
                        + "refuses an archive whose bytes do not match it")
                Button("Import archive") { perform {
                    let result = try await workspace("import", ["archivePath": .string(archiveFile), "archiveSHA256": .string(archiveHash)])
                    receiptHash = string(result, "receiptSHA256") ?? ""; show(result)
                } }.disabled(archiveFile.isEmpty || archiveHash.isEmpty)
                    .help("verify the chosen archive against that digest and "
                        + "import it into this workspace with a custody receipt")
            }
            HStack(spacing: 8) {
                TextField("Custody receipt SHA-256", text: $receiptHash)
                    .font(.system(.body, design: .monospaced))
                    .help("the receipt an import wrote — filled by the fetch and "
                        + "import actions above, or pasted from an earlier one")
                    .onChange(of: receiptHash) { _, _ in cleanupPlan = nil; confirmation = false }
                Button("Verify receipt") { perform { show(try await workspace("verify-custody", ["receiptSHA256": .string(receiptHash)])) } }.disabled(receiptHash.isEmpty)
                    .help("re-read the retained archive and expanded files this "
                        + "receipt covers and check every byte, offline")
                Button("List and verify local receipts") { perform { show(try await workspace("custody", [:])) } }
                    .help("list every custody receipt in this workspace and "
                        + "verify each one, offline")
            }
        } header: {
            Text("Bring evidence home and inspect it offline")
        }
    }

    private func campaignSection(_ client: ClusterClient) -> some View {
        Section {
            DisclosureGroup("Campaign cell coordination (OptVec)", isExpanded: $showingCampaign) {
                caption("For a materialized OptVec campaign job: inspect cells, then explicitly top up its queue or request cancellation. A materialized campaign is not completed training.")
                HStack(spacing: 8) {
                    Button("Review campaign status and plan") { perform {
                        let result = try await campaign("plan", client: client)
                        campaignPlan = string(result, "planSHA256"); campaignConfirmed = false; show(result)
                    } }.disabled(jobID.isEmpty)
                        .help("read the campaign's cells and the plan for what a "
                            + "top-up or cancellation would do — reads only")
                    Toggle("Apply the reviewed campaign action", isOn: $campaignConfirmed)
                        .help("tick to arm the two buttons with the plan just reviewed")
                }
                HStack(spacing: 8) {
                    ForEach(["submit", "cancel"], id: \.self) { action in
                        Button(action == "submit" ? "Top up campaign queue" : "Request cell cancellation") { perform {
                            guard let campaignPlan else { return }
                            self.campaignPlan = nil; campaignConfirmed = false
                            show(try await campaign(action, client: client, hash: campaignPlan))
                        } }.disabled(campaignPlan == nil || !campaignConfirmed)
                    }
                }
            }
        }
    }

    private func fittingRoundsSection(_ client: ClusterClient) -> some View {
        Section {
            DisclosureGroup("J-lens fitting rounds", isExpanded: $showingFittingRounds) {
                roundPlacement
                caption("Use the originating job ID of a materialized fitting round. Review pending shards and capacity before topping up the queue. Each shard remains a separate durable job; failed or uncertain submissions are never retried automatically.")
                HStack(spacing: 8) {
                    Button("Review shard queue") { perform {
                        let result = try await fittingRound("plan", client: client)
                        pendingShards = ScientificGPUPlacement.submitIndices(result)
                        roundReview.record(result, jobID: jobID, kind: .queue); show(result)
                    } }.disabled(jobID.isEmpty)
                        .help("read which shards are pending and what a top-up "
                            + "would submit — reads only")
                    Button("Review merge of completed shards") { perform {
                        let result = try await fittingRound("merge-plan", client: client)
                        roundReview.record(result, jobID: jobID, kind: .merge); show(result)
                    } }.disabled(jobID.isEmpty)
                        .help("read which completed shards a merge would combine "
                            + "and which rows it would still be missing")
                }
                if let snapshot = roundReview.snapshot {
                    caption(snapshot.label)
                    ForEach(snapshot.lines, id: \.self) { line in caption(line) }
                    DisclosureGroup("Full reviewed round plan") {
                        Text(formatted(snapshot.document)).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                    }
                    if roundReview.planSHA256 == nil {
                        caption("Review the queue or merge again before applying another action.")
                    }
                }
                Toggle("Apply the reviewed fitting-round plan", isOn: $roundReview.confirmed)
                    .disabled(roundReview.planSHA256 == nil)
                    .help("tick to arm the buttons below with the plan just reviewed")
                HStack(spacing: 8) {
                    ForEach(["submit", "cancel"], id: \.self) { action in
                        Button(action == "submit" ? "Top up shard queue" : "Request shard cancellation") { perform {
                            guard let hash = roundReview.hash(for: .queue) else { return }
                            roundReview.invalidate()
                            show(try await fittingRound(action, client: client, hash: hash))
                            pendingShards = []; shardGPUTypes = [:]
                        } }.disabled(roundReview.hash(for: .queue) == nil || !roundReview.confirmed)
                    }
                    Button("Merge completed shards") { perform {
                        guard let hash = roundReview.hash(for: .merge) else { return }
                        roundReview.invalidate()
                        show(try await fittingRound("merge-submit", client: client, hash: hash))
                    } }.disabled(roundReview.hash(for: .merge) == nil || !roundReview.confirmed)
                }
                caption("A partial merge records missing rows. After collection, register the merged lens and use the held-out assessment method to compare it with an earlier fit.")
            }
        }
    }

    private func cleanupSection(_ client: ClusterClient) -> some View {
        Section {
            caption("Only isolated, completed diagnostic output copies are eligible. The server rechecks its declared policy and dependencies; this app re-verifies local custody before applying. Inputs, export archives and job records remain.")
            HStack(spacing: 8) {
                Button("Review cleanup plan") { perform {
                    confirmation = false
                    let result = try await DiagnosticRemote.cleanup(jobID, client: client, root: root, receiptSHA256: receiptHash)
                    cleanupPlan = bool(result, "eligible") ? string(result, "planSHA256") : nil; show(result)
                } }.disabled(jobID.isEmpty || receiptHash.isEmpty)
                    .help("verify local custody, then ask the server whether this "
                        + "job's remote output copy may be removed — reads only")
                Toggle("Remove exactly the reviewed remote output", isOn: $confirmation)
                    .help("tick to arm Apply with the plan just reviewed")
                Button("Apply reviewed cleanup") { perform {
                    guard let cleanupPlan else { return }
                    self.cleanupPlan = nil; confirmation = false
                    show(try await DiagnosticRemote.cleanup(jobID, client: client, root: root, receiptSHA256: receiptHash, applyPlanSHA256: cleanupPlan))
                } }.disabled(cleanupPlan == nil || !confirmation)
                    .help("remove the one remote output the plan named — nothing "
                        + "else on the server, and nothing in this workspace")
            }
        } header: {
            Text("Review remote cleanup")
        }
    }

    @ViewBuilder
    private var roundPlacement: some View {
        if let gpuOptions, gpuOptions.available {
            ScientificGPUSelection(options: gpuOptions, selection: $roundGPUType, title: "GPU for this queue top-up")
            if !shardGPUTypes.isEmpty { Button("Clear shard overrides") { shardGPUTypes = [:] } }
            ForEach(pendingShards, id: \.self) { index in
                Picker("Shard \(index) GPU", selection: Binding(get: { shardGPUTypes[index] ?? "" }, set: { shardGPUTypes[index] = $0 })) {
                    Text("Use top-up selection").tag("")
                    ForEach(gpuOptions.gpuTypes, id: \.self) { Text($0).tag($0) }
                }
            }
            caption("Review the queue to see shards eligible for this top-up. Overrides apply only to those shards; submitted jobs keep their placement.")
        }
    }

    /// A wrapping caption row: never truncated, never clipped.
    private func caption(_ text: String) -> some View {
        Text(text).font(.caption).foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
            .textSelection(.enabled)
    }

    // MARK: Actions (unchanged)

    private func refreshRequestKind() {
        guard !requestFile.isEmpty,
              let data = try? Data(contentsOf: URL(filePath: requestFile)),
              let request = try? ScientificRequestDocument.read(data) else { gpuRequirement = .unknown; return }
        gpuRequirement = ScientificGPUPlacement.requirement(request)
    }
    private func fittingRound(_ action: String, client: ClusterClient, hash: String? = nil) async throws -> JSONValue {
        let usesPlacement = ["plan", "submit", "cancel"].contains(action)
        let body = ScientificGPUPlacement.roundBody(hash: hash, gpuType: usesPlacement ? roundGPUType : "",
                                                    overrides: usesPlacement ? shardGPUTypes : [:])
        let document: JSONValue = .object(["path": .object(["job_id": .string(jobID), "action": .string(action)]), "query": .object([:]), "body": .object(body)])
        let result = try await client.callScientificAction(operation: "jlens-fit-round", actionID: "post-fitting-round", document: JSONEncoder().encode(document))
        guard let text = string(result, "responseJSON") else { throw ExperimentError(reason: "Fitting round returned no readable result.") }
        return try JSONDecoder().decode(JSONValue.self, from: Data(text.utf8))
    }
    private func campaign(_ action: String, client: ClusterClient, hash: String? = nil) async throws -> JSONValue {
        let body: [String: JSONValue] = hash.map { ["planSHA256": .string($0), "confirmAction": .bool(true)] } ?? [:]
        let document: JSONValue = .object(["path": .object(["job_id": .string(jobID), "action": .string(action)]), "query": .object([:]), "body": .object(body)])
        let result = try await client.callScientificAction(operation: "optvec-campaign", actionID: "post-science-campaign", document: JSONEncoder().encode(document))
        guard let text = string(result, "responseJSON") else { throw ExperimentError(reason: "Campaign returned no readable result.") }
        return try JSONDecoder().decode(JSONValue.self, from: Data(text.utf8))
    }
    private func workspace(_ action: String, _ payload: [String: JSONValue]) async throws -> JSONValue {
        try await DiagnosticWorkspace.perform(action, payload: payload.merging(["workspaceRoot": .string(root.path)]) { _, new in new })
    }
    private func string(_ value: JSONValue, _ key: String) -> String? {
        guard case .object(let object) = value, case .string(let text) = object[key] else { return nil }; return text
    }
    private func bool(_ value: JSONValue, _ key: String) -> Bool {
        guard case .object(let object) = value else { return false }; return object[key] == .bool(true)
    }
    private func show(_ value: JSONValue) {
        output = formatted(value)
    }
    private func formatted(_ value: JSONValue) -> String {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return (try? String(decoding: encoder.encode(value), as: UTF8.self)) ?? "Could not display result."
    }
    private func perform(_ action: @escaping @MainActor () async throws -> Void) {
        busy = true
        Task {
            defer { busy = false }
            do { try await action() }
            catch { roundReview.invalidate(); cleanupPlan = nil; confirmation = false; executionPlan = nil; output = error.localizedDescription + "\nInspect the originating server's jobs before retrying an uncertain submission." }
        }
    }
}
