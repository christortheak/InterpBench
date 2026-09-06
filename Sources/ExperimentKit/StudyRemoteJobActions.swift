import Foundation

extension StudyRemoteJobController {
    private func actionNote(_ text: String, severity: PanelNotice.Severity) {
        presentation.note(text, severity)
    }

    public func resume(_ id: String, client: ClusterClient?) async {
        guard let client else {
            actionNote(
                "remote resume refused: no server connection — connect a "
                    + "server in the substrate selector first",
                severity: .error)
            return
        }
        do {
            let client = try clientForJob(id, connected: client)
            let result = try await client.resubmitJob(id)
            let line = RemoteJobStatusClass.resumedStatusLine(
                jobID: id, slurmJobID: result.slurmJobID,
                continuationJobID: result.jobId)
            remoteStatus = line
            actionNote(line, severity: .success)
            await refreshRecentServerJobs(client: client)
        } catch let error as ClusterClient.ClientError {
            // 409 details are the server's own plain-language refusal
            // (already resubmitted / still running / cancelled) — show its
            // words verbatim.
            let detail = ClusterClient.unwrappingDetail(error)
            remoteStatus = "resume failed: \(detail)"
            actionNote("Couldn't resume job \(id) — \(detail)", severity: .error)
        } catch {
            actionNote(
                "Couldn't resume job \(id) — \(error.localizedDescription)",
                severity: .error)
        }
    }

    public func cancelActiveServerJob(client: ClusterClient?) async {
        guard let job = activeServerJob else { return }
        guard let client else {
            actionNote("invalid server URL", severity: .error)
            return
        }
        do {
            let client = try clientForJob(job.id, connected: client)
            try await client.cancelJob(job.id)
            actionNote(
                "cancel requested for server \(job.verb) job \(job.id) ('\(job.study)')",
                severity: .warning)
        } catch {
            actionNote("cancel failed for job \(job.id): \(error)", severity: .error)
        }
    }

    public func cancelRemoteJob(client: ClusterClient?) async {
        guard let client, let remoteJobID else { return }
        do {
            let client = try clientForJob(remoteJobID, connected: client)
            try await client.cancelJob(remoteJobID)
            remoteStatus = "cancel requested for \(remoteJobID)"
        } catch {
            remoteStatus = "remote cancel failed: \(error)"
        }
    }
}
