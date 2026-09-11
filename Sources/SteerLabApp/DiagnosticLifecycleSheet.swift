import ExperimentKit
import SwiftUI
import UniformTypeIdentifiers

/// Captured workspace/client remain authoritative even if the underlying app
/// selection changes while transfer, verification or cleanup is in flight.
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
    @State private var executionPlan: String?
    @State private var campaignPlan: String?
    @State private var campaignConfirmed = false
    @State private var cleanupPlan: String?
    @State private var confirmation = false
    @State private var output = ""
    @State private var busy = false
    @State private var choosingRequest = true
    @State private var showingImporter = false
    @State private var scopedURLs: [URL] = []

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Scientific inputs, evidence and cleanup").font(.title2)
                Spacer(); Button("Done") { dismiss() }.disabled(busy)
            }
            Text("Local workspace: " + root.path).font(.caption).textSelection(.enabled)
            if let client { Text("Server: " + client.profile.baseURL.absoluteString).font(.caption).textSelection(.enabled) }
            Form {
                Section("Prepare local inputs") {
                    HStack {
                        Text(requestFile.isEmpty ? "Choose a diagnostic request prepared with the method guide or your agent." : requestFile).textSelection(.enabled)
                        Button("Choose request…") { choosingRequest = true; showingImporter = true }
                    }
                    HStack {
                        Button("Review inputs") { perform {
                            let result = try await workspace("input-plan", ["requestFile": .string(requestFile)])
                            sourcePlan = string(result, "planSHA256"); show(result)
                        } }.disabled(requestFile.isEmpty)
                        Button("Package reviewed inputs") { perform {
                            guard let sourcePlan else { return }
                            let path = root.appending(path: ".steerlab/diagnostic-outgoing/" + UUID().uuidString + ".tar.gz").path
                            let result = try await workspace("package", ["requestFile": .string(requestFile), "planSHA256": .string(sourcePlan), "archivePath": .string(path)])
                            inputArchiveFile = path; inputArchiveHash = string(result, "bundleSha256") ?? ""; stagedRequest = nil; executionPlan = nil; show(result)
                        } }.disabled(sourcePlan == nil)
                        if let client {
                            Button("Upload and stage inputs") { perform {
                                let uploaded = try await client.uploadBundle(URL(filePath: inputArchiveFile))
                                guard uploaded.sha256 == inputArchiveHash else { throw ExperimentError(reason: "Uploaded input hash differs from the reviewed archive.") }
                                let result = try await DiagnosticRemote.stage(path: uploaded.path, sha256: inputArchiveHash, client: client, root: root)
                                if case .object(let object) = result { stagedRequest = object["request"] }
                                executionPlan = nil; show(result)
                            } }.disabled(inputArchiveFile.isEmpty || inputArchiveHash.isEmpty)
                            Button("Review server plan") { perform {
                                guard let stagedRequest else { return }
                                let result = try await client.scientificPlan(stagedRequest)
                                executionPlan = string(result, "planSHA256"); show(result)
                            } }.disabled(stagedRequest == nil)
                            Button("Submit reviewed job") { perform {
                                guard let stagedRequest, let executionPlan else { return }
                                self.executionPlan = nil
                                let result = try await client.scientificSubmit(stagedRequest, planSHA256: executionPlan)
                                jobID = string(result, "jobId") ?? ""; show(result)
                            } }.disabled(executionPlan == nil)
                        }
                    }
                    Text("Where HTTP transfer is prohibited, use the site's declared transport and the documented stage/import commands. A reviewed plan must be submitted within the same controller session.").font(.caption)
                }
                Section("Bring evidence home and inspect it offline") {
                    Text("Large evidence exports can take several minutes to prepare before transfer begins. Keep this window open; the archive is verified before import.").font(.caption)
                    TextField("Originating job ID", text: $jobID).onChange(of: jobID) { _, _ in cleanupPlan = nil; confirmation = false; campaignPlan = nil; campaignConfirmed = false }
                    HStack {
                        if let client {
                            Button("Fetch and verify evidence") { perform {
                                let result = try await DiagnosticRemote.fetch(jobID, client: client, root: root)
                                receiptHash = string(result, "receiptSHA256") ?? ""; show(result)
                            } }.disabled(jobID.isEmpty)
                        }
                        Button("Choose transferred archive…") { choosingRequest = false; showingImporter = true }
                        TextField("Archive SHA-256 from the exporting job", text: $archiveHash)
                        Button("Import archive") { perform {
                            let result = try await workspace("import", ["archivePath": .string(archiveFile), "archiveSHA256": .string(archiveHash)])
                            receiptHash = string(result, "receiptSHA256") ?? ""; show(result)
                        } }.disabled(archiveFile.isEmpty || archiveHash.isEmpty)
                    }
                    HStack {
                        TextField("Custody receipt SHA-256", text: $receiptHash).onChange(of: receiptHash) { _, _ in cleanupPlan = nil; confirmation = false }
                        Button("Verify receipt") { perform { show(try await workspace("verify-custody", ["receiptSHA256": .string(receiptHash)])) } }.disabled(receiptHash.isEmpty)
                        Button("List and verify local receipts") { perform { show(try await workspace("custody", [:])) } }
                    }
                }
                if let client {
                    Section("Campaign cell coordination") {
                        Text("For a materialized OptVec campaign job: inspect cells, then explicitly top up its queue or request cancellation. A materialized campaign is not completed training.").font(.caption)
                        HStack {
                            Button("Review campaign status and plan") { perform {
                                let result = try await campaign("plan", client: client)
                                campaignPlan = string(result, "planSHA256"); campaignConfirmed = false; show(result)
                            } }.disabled(jobID.isEmpty)
                            Toggle("Apply the reviewed campaign action", isOn: $campaignConfirmed)
                            ForEach(["submit", "cancel"], id: \.self) { action in
                                Button(action == "submit" ? "Top up campaign queue" : "Request cell cancellation") { perform {
                                    guard let campaignPlan else { return }
                                    self.campaignPlan = nil; campaignConfirmed = false
                                    show(try await campaign(action, client: client, hash: campaignPlan))
                                } }.disabled(campaignPlan == nil || !campaignConfirmed)
                            }
                        }
                    }
                    Section("Review remote cleanup") {
                        Text("Only isolated, completed diagnostic output copies are eligible. The server rechecks its declared policy and dependencies; this app re-verifies local custody before applying. Inputs, export archives and job records remain.").font(.caption)
                        HStack {
                            Button("Review cleanup plan") { perform {
                                confirmation = false
                                let result = try await DiagnosticRemote.cleanup(jobID, client: client, root: root, receiptSHA256: receiptHash)
                                cleanupPlan = bool(result, "eligible") ? string(result, "planSHA256") : nil; show(result)
                            } }.disabled(jobID.isEmpty || receiptHash.isEmpty)
                            Toggle("Remove exactly the reviewed remote output", isOn: $confirmation)
                            Button("Apply reviewed cleanup") { perform {
                                guard let cleanupPlan else { return }
                                self.cleanupPlan = nil; confirmation = false
                                show(try await DiagnosticRemote.cleanup(jobID, client: client, root: root, receiptSHA256: receiptHash, applyPlanSHA256: cleanupPlan))
                            } }.disabled(cleanupPlan == nil || !confirmation)
                        }
                    }
                }
            }.disabled(busy)
            if busy { ProgressView("Working on the request. Reading and verifying large input or evidence files can take several minutes.") }
            ScrollView { Text(output).font(.system(.caption, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
        }.padding().frame(minWidth: 1050, minHeight: 740)
        .interactiveDismissDisabled(busy)
        .onAppear { jobID = initialJobID ?? ""; requestFile = initialRequestFile ?? "" }
        .onDisappear { scopedURLs.forEach { $0.stopAccessingSecurityScopedResource() } }
        .fileImporter(isPresented: $showingImporter, allowedContentTypes: choosingRequest ? [.json] : [.data]) { result in
            do {
                let url = try result.get(); if url.startAccessingSecurityScopedResource() { scopedURLs.append(url) }
                if choosingRequest { requestFile = url.path; sourcePlan = nil; inputArchiveFile = ""; inputArchiveHash = ""; stagedRequest = nil; executionPlan = nil }
                else { archiveFile = url.path; archiveHash = "" }
            } catch { output = error.localizedDescription }
        }
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
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        output = (try? String(decoding: encoder.encode(value), as: UTF8.self)) ?? "Could not display result."
    }
    private func perform(_ action: @escaping @MainActor () async throws -> Void) {
        busy = true
        Task {
            defer { busy = false }
            do { try await action() }
            catch { cleanupPlan = nil; confirmation = false; executionPlan = nil; output = error.localizedDescription + "\nInspect the originating server's jobs before retrying an uncertain submission." }
        }
    }
}
