import CryptoKit
import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

@Suite(.serialized) struct ClusterClientTests {
    @Test func clientErrorsExposeTheirActionableMessageThroughLocalizedDescription() {
        let error = ClusterClient.ClientError.badResponse(
            401, #"{"detail":"missing or invalid bearer token"}"#)
        #expect(
            error.localizedDescription
                == #"server returned 401: {"detail":"missing or invalid bearer token"}"#)
    }

    @Test func capabilitiesSendsBearerToken() async throws {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            token: "secret",
            session: Self.session { request in
                #expect(request.url?.path == "/api/capabilities")
                #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer secret")
                return (Data(#"{"serverVersion":"test"}"#.utf8), 200)
            })

        let capabilities = try await client.capabilities()

        #expect(capabilities.serverVersion == "test")
    }

    @Test func streamingErrorBodiesSurviveValidation() async throws {
        // Engineer review 2026-07-17: streaming routes validated non-2xx
        // responses with empty Data, so a 409's JSON detail ("no model
        // loaded…") was lost — the UI could only say "server returned 409",
        // and GPUSessionRefusal.hint was starved of the text it matches on.
        // The error body must be drained into badResponse before SSE parsing.
        let stub = Self.session { request in
            #expect(request.url?.path == "/api/load/stream")
            return (
                Data(#"{"detail": "no model loaded; POST /api/load first"}"#.utf8),
                409)
        }
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            token: "secret", session: stub, streamSession: stub)
        do {
            try await client.streamLoadModel("org/m")
            Issue.record("expected badResponse(409)")
        } catch let error as ClusterClient.ClientError {
            guard case .badResponse(let code, let body) = error else {
                Issue.record("unexpected error \(error)")
                return
            }
            #expect(code == 409)
            #expect(body.contains("no model loaded"))
            #expect(
                ClusterClient.unwrappingDetail(error).description
                    .contains("no model loaded"))
        }
    }

    @Test func experimentManifestBodyFetchesRawBytes() async throws {
        // The remote-freeze identity check compares the server's manifest
        // DOCUMENT — the route must be hit verbatim and the body returned as
        // raw bytes, not decoded into a model (decoding would erase the key
        // surface the comparison exists to inspect).
        let body = Data(#"{"name":"case1","modelID":"org/m","unknownKey":42}"#.utf8)
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            token: "secret",
            session: Self.session { request in
                #expect(request.url?.path == "/api/experiment/case1/manifest")
                #expect(request.httpMethod == "GET")
                #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer secret")
                return (body, 200)
            })

        let fetched = try await client.experimentManifestBody(name: "case1")

        #expect(fetched == body)
        // Byte-preserving: the unknown key survives for the comparison.
        let object = try #require(
            JSONSerialization.jsonObject(with: fetched) as? [String: Any])
        #expect(object["unknownKey"] as? Int == 42)
    }

    @Test func experimentManifestBodyMissingIs404Error() async throws {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { _ in
                (Data(#"{"detail":"no experiment 'ghost'"}"#.utf8), 404)
            })
        do {
            _ = try await client.experimentManifestBody(name: "ghost")
            Issue.record("expected a 404 ClientError")
        } catch let error as ClusterClient.ClientError {
            guard case .badResponse(404, _) = error else {
                Issue.record("expected badResponse(404), got \(error)")
                return
            }
        }
    }

    @Test func replaceExperimentManifestPutsTheRawDocument() async throws {
        // One-click server-draft sync (2026-07-21 incident, part 3): the
        // PUT carries the LOCAL manifest document verbatim — the exact
        // bytes the identity check compared — and decodes the server's
        // status + canonical body hash.
        let document = Data(#"{"name":"case1","status":"draft","unknownKey":7}"#.utf8)
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            token: "secret",
            session: Self.session { request in
                #expect(request.url?.path == "/api/experiment/case1/manifest")
                #expect(request.httpMethod == "PUT")
                #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer secret")
                #expect(Self.bodyData(from: request) == document)
                return (
                    Data(#"{"name":"case1","status":"draft","canonicalBodyHash":"abc123"}"#.utf8),
                    200)
            })

        let result = try await client.replaceExperimentManifest(
            name: "case1", manifestBody: document)

        #expect(result.name == "case1")
        #expect(result.status == "draft")
        #expect(result.canonicalBodyHash == "abc123")
    }

    @Test func replaceExperimentManifestSurfacesTheFrozenRefusalVerbatim() async throws {
        // The server's freeze-firewall refusal is the actionable text —
        // unwrapped from the FastAPI detail envelope, never a raw JSON blob.
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { _ in
                (
                    Data(
                        #"{"detail":"refusing to overwrite frozen manifest 'case1' with a pushed draft (freeze firewall) — duplicate to iterate"}"#
                            .utf8),
                    400)
            })
        do {
            _ = try await client.replaceExperimentManifest(
                name: "case1", manifestBody: Data(#"{"name":"case1"}"#.utf8))
            Issue.record("expected the frozen refusal")
        } catch let error as ClusterClient.ClientError {
            guard case .badResponse(400, let detail) = error else {
                Issue.record("expected badResponse(400), got \(error)")
                return
            }
            #expect(detail.contains("refusing to overwrite frozen manifest"))
            #expect(detail.contains("freeze firewall"))
        }
    }

    @Test func precheckPushRecheckFlowEndsVerifiedEqual() async throws {
        // The full sync flow against a scripted server: (1) the identity
        // check sees a MISMATCHED server copy and blocks with the sync
        // affordance; (2) the push replaces the server's draft; (3) the
        // re-check verifies equal and the outcome says so. The scripted
        // server stores what the PUT delivered — GET serves it back, so the
        // verification is a measurement, not an assumption.
        let localDocument = Data(
            #"{"name":"case1","status":"draft","modelID":"org/m","temperature":0}"#.utf8)
        let serverStore = ScriptedManifestStore(
            initial: Data(
                #"{"name":"case1","status":"draft","modelID":"org/m","temperature":0.7}"#.utf8))
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/experiment/case1/manifest")
                switch request.httpMethod {
                case "GET":
                    return (serverStore.body, 200)
                case "PUT":
                    serverStore.body = Self.bodyData(from: request) ?? Data()
                    return (
                        Data(#"{"name":"case1","status":"draft","canonicalBodyHash":"feed"}"#.utf8),
                        200)
                default:
                    Issue.record("unexpected method \(request.httpMethod ?? "?")")
                    return (Data(), 500)
                }
            })

        // (1) Precheck: mismatch → blocked, sync offered for a local draft.
        let before = try await client.experimentManifestBody(name: "case1")
        guard case .different(let fields) = ExperimentStore.compareManifestDocuments(
            local: localDocument, server: before)
        else {
            Issue.record("expected a mismatch before the push")
            return
        }
        let identity = FreezeRouting.RemoteManifestIdentity.mismatch(fields)
        let precheck = FreezeRouting.remoteFreezePrecheck(
            identity: identity, study: "case1", serverLabel: "example-hpc",
            workspacePaired: false)
        #expect(!precheck.proceed)
        #expect(FreezeRouting.canOfferServerDraftSync(
            identity: identity, localIsDraft: true))

        // (2) Push the local document as the server's draft copy.
        let pushed = try await client.replaceExperimentManifest(
            name: "case1", manifestBody: localDocument)
        #expect(pushed.canonicalBodyHash == "feed")

        // (3) Re-check: the server now serves the pushed document.
        let after = try await client.experimentManifestBody(name: "case1")
        guard case .equal = ExperimentStore.compareManifestDocuments(
            local: localDocument, server: after)
        else {
            Issue.record("expected verified-equal after the push")
            return
        }
        let outcome = FreezeRouting.serverDraftSyncOutcome(
            recheck: .verifiedEqual, study: "case1", serverLabel: "example-hpc",
            canonicalBodyHash: pushed.canonicalBodyHash)
        #expect(outcome.resolved)
        #expect(outcome.message.contains("verified equal"))
    }

    @Test func submitBundleEncodesRemoteSubmissionRequest() async throws {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/studies/submit-bundle")
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                #expect(object["bundlePath"] as? String == "/runs/uploaded/study.tar.gz")
                #expect(object["verb"] as? String == "verify")
                #expect(object["executor"] as? String == "slurm")
                #expect(object["dryRun"] as? Bool == true)
                let resources = try #require(object["resources"] as? [String: String])
                #expect(resources["gres"] == "gpu:1")
                return (
                    Data(
                        """
                        {
                          "jobId": "job-1",
                          "experiment": "study",
                          "verb": "verify",
                          "executor": "slurm",
                          "dryRun": true,
                          "runBundle": {},
                          "slurmBundle": null,
                          "slurmJobID": null,
                          "command": ["python"],
                          "recordsDirectory": "/records",
                          "submissionDirectory": "/submit"
                        }
                        """.utf8),
                    200)
            })

        let submission = try await client.submitBundle(
            path: "/runs/uploaded/study.tar.gz",
            verb: "verify",
            executor: "slurm",
            dryRun: true,
            resources: ["gres": "gpu:1"])

        #expect(submission.jobId == "job-1")
        #expect(submission.executor == "slurm")
        #expect(submission.dryRun)
    }

    @Test func submitBundleEncodesResumePolicyAsRealJSONTypes() async throws {
        // 2026-07-22 incident fix: the resume policy must ride resources as
        // JSON bool/number — the server coerces with bool()/int(), and a
        // string "false" would read truthy.
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                let resources = try #require(object["resources"] as? [String: Any])
                #expect(resources["gres"] as? String == "A100")
                #expect(resources["autoResubmit"] as? Bool == true)
                #expect(resources["autoResubmitLimit"] as? Int == 3)
                return (
                    Data(
                        """
                        {
                          "jobId": "job-2", "experiment": "study", "verb": "run",
                          "executor": "slurm", "dryRun": false, "runBundle": {},
                          "command": [], "recordsDirectory": "/r",
                          "submissionDirectory": "/s"
                        }
                        """.utf8),
                    200)
            })

        _ = try await client.submitBundle(
            path: "/b.tar.gz", verb: "run", executor: "slurm", dryRun: false,
            resources: ["gres": "A100"],
            resumePolicy: RemoteResumePolicy(autoResubmit: true, limit: 3))
    }

    @Test func submitBundleOmitsResumeKeysWhenNoPolicyIsPassed() async throws {
        // Older servers (and local-executor submissions) must see no unknown
        // resume keys at all.
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                let resources = try #require(object["resources"] as? [String: Any])
                #expect(resources["autoResubmit"] == nil)
                #expect(resources["autoResubmitLimit"] == nil)
                return (
                    Data(
                        """
                        {
                          "jobId": "job-3", "experiment": "study", "verb": "run",
                          "executor": "local", "dryRun": false, "runBundle": {},
                          "command": [], "recordsDirectory": "/r",
                          "submissionDirectory": "/s"
                        }
                        """.utf8),
                    200)
            })

        _ = try await client.submitBundle(
            path: "/b.tar.gz", verb: "run", executor: "local", dryRun: false,
            resources: ["gres": "A100"])
    }

    @Test func resumePolicyDefaultsOnWithTheServersShippedLimit() {
        // DEFAULT ON: a checkpointed batch run continuing is what the
        // researcher asked for by submitting it — OFF is the surprising
        // choice. The limit mirrors DEFAULT_AUTO_RESUBMIT_LIMIT (executors.py).
        let policy = RemoteResumePolicy()
        #expect(policy.autoResubmit == true)
        #expect(policy.limit == 5)
        #expect(RemoteResumePolicy.serverDefaultLimit == 5)
        #expect(policy.transcriptStamp == "auto-resume on (up to 5 restarts)")
        #expect(
            RemoteResumePolicy(autoResubmit: false).transcriptStamp
                == "auto-resume off")
        #expect(
            RemoteResumePolicy(autoResubmit: true, limit: 1).transcriptStamp
                == "auto-resume on (up to 1 restart)")
    }

    @Test func resubmitJobPostsTheResubmitVerbAndDecodesTheContinuation() async throws {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            token: "secret",
            session: Self.session { request in
                #expect(request.url?.path == "/api/jobs/abc123/resubmit")
                #expect(request.httpMethod == "POST")
                #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer secret")
                return (
                    Data(
                        """
                        {"ok": true, "jobId": "child77", "resubmitOf": "abc123",
                         "slurmJobID": "9002", "resubmitCount": 1,
                         "autoResubmitLimit": 5, "manualResubmit": true}
                        """.utf8),
                    200)
            })

        let result = try await client.resubmitJob("abc123")

        #expect(result.jobId == "child77")
        #expect(result.slurmJobID == "9002")
        #expect(result.manualResubmit == true)
        #expect(result.resubmitCount == 1)
    }

    @Test func resubmitJobSurfacesThe409RefusalDetailVerbatim() async throws {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { _ in
                (
                    Data(
                        #"{"detail": "job abc123 was already resubmitted as child77 — that continuation is carrying the run; follow it instead"}"#
                            .utf8),
                    409)
            })
        do {
            _ = try await client.resubmitJob("abc123")
            Issue.record("expected badResponse(409)")
        } catch let error as ClusterClient.ClientError {
            guard case .badResponse(let code, let body) = error else {
                Issue.record("unexpected error \(error)")
                return
            }
            #expect(code == 409)
            #expect(body.contains("already resubmitted as child77"))
        }
    }

    @Test func resumeButtonVisibilityRuleIsCheckpointedAndNotYetResubmitted() {
        // The pure rule every job row renders from (Compute rows, recent
        // jobs): resumable display class AND no continuation yet.
        #expect(RemoteJobStatusClass.offersResume(status: "checkpointed"))
        #expect(
            !RemoteJobStatusClass.offersResume(
                status: "checkpointed", resubmittedAs: "child77"))
        #expect(!RemoteJobStatusClass.offersResume(status: "running"))
        #expect(!RemoteJobStatusClass.offersResume(status: "cancelling"))
        #expect(!RemoteJobStatusClass.offersResume(status: "succeeded"))
        #expect(!RemoteJobStatusClass.offersResume(status: "failed"))
        #expect(!RemoteJobStatusClass.offersResume(status: "cancelled"))
        #expect(!RemoteJobStatusClass.offersResume(status: "prepared"))
        #expect(!RemoteJobStatusClass.offersResume(status: "submitted"))
    }

    @Test func vectorPickerOrderingGroupsByConceptNewestFirst() {
        // Field report 2026-08-04: the playground vector menu was one long
        // unsorted list. Contract: concept sections A→Z; within a concept
        // newest extraction first, name tie-break; date-less rows sink.
        struct Row {
            let concept: String?
            let name: String
            let date: String?
        }
        let records = [
            Row(concept: "fear", name: "fear-b", date: "2026-08-01T10:00:00Z"),
            Row(concept: "anger", name: "anger-a", date: "2026-07-01T10:00:00Z"),
            Row(concept: "fear", name: "fear-a", date: "2026-08-02T10:00:00Z"),
            Row(concept: "fear", name: "fear-c", date: nil),
        ]
        let groups = VectorPickerOrdering.grouped(
            records,
            concept: { $0.concept },
            name: { $0.name },
            extractionDate: { $0.date })
        #expect(groups.map(\.concept) == ["anger", "fear"])
        #expect(groups[1].items.map(\.name) == ["fear-a", "fear-b", "fear-c"])
    }

    @Test func agentRecipeSyncNormalizesServerAbsoluteVectorReferences() async {
        // The workspace-is-source-of-truth sync (2026-08-03): recipes copied
        // from a server carry references each engine must resolve under its
        // OWN root — server-absolute paths under the serving root become
        // workspace-relative; foreign/relative references pass untouched.
        let artifact = ModelVariantArtifact(
            name: "neuroticism-agent", baseModelID: "google/gemma-3-27b-it",
            injections: [
                .init(
                    concept: "neuroticism",
                    vectorArtifactID:
                        "/scratch/u/ws/runs/20260726T000000000-exp-sweep/neuroticism",
                    layer: 31, alpha: 0.13),
                .init(
                    concept: "fear",
                    vectorArtifactID: "runs/20260720T000000000-exp-extract/fear",
                    layer: 20, alpha: 0.1),
            ],
            promptMode: "chatAssistant", qwenThinkingEnabled: false,
            temperature: 0, systemPrompt: "")
        var withDependencies = artifact
        withDependencies.adapters = [
            .init(
                name: "lora", artifactPath: "/scratch/u/ws/runs/ft/adapter",
                adapterDirectory: "/scratch/u/ws/runs/ft/adapter/weights")
        ]
        withDependencies.neutralPCBasisPath = "/scratch/u/ws/runs/neutral-pcs/b1"
        let normalized = await ClusterConnectionStore.workspaceRelativeArtifact(
            withDependencies, servingRoot: "/scratch/u/ws")
        #expect(
            normalized.injections[0].vectorArtifactID
                == "runs/20260726T000000000-exp-sweep/neuroticism")
        #expect(
            normalized.injections[1].vectorArtifactID
                == "runs/20260720T000000000-exp-extract/fear")
        // Every server-root-absolute reference normalizes, not just vectors
        // (round 2026-08-03, P1): adapters + neutral-PC basis too.
        #expect(normalized.adapters[0].artifactPath == "runs/ft/adapter")
        #expect(
            normalized.adapters[0].adapterDirectory
                == "runs/ft/adapter/weights")
        #expect(normalized.neutralPCBasisPath == "runs/neutral-pcs/b1")
        // No serving root: nothing to strip, artifact untouched.
        let untouched = await ClusterConnectionStore.workspaceRelativeArtifact(
            artifact, servingRoot: nil)
        #expect(untouched == artifact)
    }

    @Test func cancelledResumableClassifiesResumableAndOffersResume() {
        // Review 2026-08-03 round 3, P1: a cancel-parked in-process run must
        // reach the same Resume affordance a checkpointed one does — and the
        // generic "cancel" → failed rule must not swallow it.
        #expect(
            RemoteJobStatusClass.classify(status: "cancelledResumable")
                == .resumable)
        #expect(RemoteJobStatusClass.offersResume(status: "cancelledResumable"))
        #expect(
            !RemoteJobStatusClass.offersResume(
                status: "cancelledResumable", resubmittedAs: "child9"))
        #expect(
            RemoteJobStatusClass.displayText(for: "cancelledResumable")
                == "cancelled (resumable)")
        // The in-process continuation names its job record, never a
        // fictitious Slurm job.
        #expect(
            RemoteJobStatusClass.resumedStatusLine(
                jobID: "abc", slurmJobID: nil, continuationJobID: "def")
                == "job abc resumed as in-process job def — continuing from "
                + "the parked run directory")
    }

    @Test func parkedIsItsOwnStateNeverGreenAndNeverFailed() {
        // 2026-08-06 review round 2, P1. The server's `_park` returns
        // normally, so the generic runner stamped "succeeded" and the app
        // painted a chain that needs a human green. `parked` is now its own
        // terminal status, and it must classify BEFORE the
        // finishedAt-implies-succeeded rule — a parked job IS finished.
        #expect(RemoteJobStatusClass.classify(status: "parked") == .parked)
        #expect(
            RemoteJobStatusClass.classify(status: "parked", finishedAt: 1_700)
                == .parked)
        #expect(RemoteJobStatusClass.classify(status: "parked") != .succeeded)
        #expect(RemoteJobStatusClass.classify(status: "parked") != .failed)
        // Terminal and NOT resumable: a parked chain is resubmitted or
        // imported, never continued from a checkpoint that does not exist.
        #expect(!RemoteJobStatusClass.offersResume(status: "parked"))
        #expect(
            RemoteJobStatusClass.displayText(for: "parked")
                == "parked (needs attention)")
    }

    @Test func aParkedRecordCarriesItsRecoveryAction() throws {
        // The row's guidance is the server's own reason, verbatim: every park
        // site writes the recovery action into it, so the row explains itself
        // without a round trip. The status is a plain String on the wire, so
        // an old client meeting an unknown status still DECODES it.
        let reason = "remaining stage(s) run need the study model — "
            + "resubmit the pipeline to continue, or import the completed "
            + "stage runs (evidence bundle)"
        let record = try JSONDecoder().decode(
            RemoteJobRecord.self,
            from: Data(
                """
                {"id": "abc", "kind": "pipeline-orphan-reconcile",
                 "status": "parked", "createdAt": 1, "finishedAt": 9,
                 "logTail": [], "executor": "local",
                 "cancellationRequested": false,
                 "result": {"parked": true, "verb": "pipeline",
                            "reason": "\(reason)"}}
                """.utf8))
        #expect(record.status == "parked")
        #expect(record.parkedReason?.contains("resubmit the pipeline") == true)
        #expect(
            RemoteJobStatusClass.parkedGuidance(reason: record.parkedReason)
                == record.parkedReason)
        // Not a failure: nothing went wrong, so nothing reads as red.
        #expect(record.failureSummary == nil)
        #expect(record.partialEvidenceBundlePath == nil)
        // A park with no recorded reason still says what to do next.
        let fallback = RemoteJobStatusClass.parkedGuidance(reason: nil)
        #expect(fallback.contains("no recorded reason"))
        #expect(RemoteJobStatusClass.parkedGuidance(reason: "") == fallback)
    }

    @Test func anUnknownStatusDecodesAndStaysNeutral() throws {
        // Forward compatibility in both directions: a status this build has
        // never heard of must not fail the decode, and must not be guessed
        // into a colour that claims something.
        let record = try JSONDecoder().decode(
            RemoteJobRecord.self,
            from: Data(
                """
                {"id": "abc", "kind": "experiment:run", "status": "quiesced",
                 "createdAt": 1, "logTail": [], "executor": "local",
                 "cancellationRequested": false}
                """.utf8))
        #expect(record.status == "quiesced")
        #expect(RemoteJobStatusClass.classify(status: "quiesced") == .neutral)
        #expect(record.parkedReason == nil)
    }

    @Test func resumeStatusAndGuidanceLinesAreActionable() {
        #expect(
            RemoteJobStatusClass.resumedStatusLine(jobID: "abc", slurmJobID: "9002")
                == "job abc resumed as Slurm job 9002 — continuing from the checkpoint")
        #expect(
            RemoteJobStatusClass.resumedStatusLine(jobID: "abc", slurmJobID: nil)
                == "job abc resumed as a new Slurm job — continuing from the checkpoint")
        let guidance = RemoteJobStatusClass.checkpointGuidance(jobID: "abc")
        #expect(guidance.contains("Resume"))
        #expect(guidance.contains("Resume automatically"))
        #expect(guidance.contains("checkpointed (resumable)"))
    }

    @Test func remoteJobRecordExposesTheResubmittedAsStamp() throws {
        let record = try JSONDecoder().decode(
            RemoteJobRecord.self,
            from: Data(
                """
                {"id": "abc", "kind": "study-submit", "status": "checkpointed",
                 "createdAt": 1, "logTail": [], "executor": "slurm",
                 "cancellationRequested": false,
                 "result": {"resubmittedAs": "child77"}}
                """.utf8))
        #expect(record.resubmittedAs == "child77")
        #expect(
            !RemoteJobStatusClass.offersResume(
                status: record.status, resubmittedAs: record.resubmittedAs))
    }

    @Test func requestPreservesBaseURLPathPrefix() async throws {
        // OOD/reverse-proxy base URLs carry a path prefix that must survive.
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://ood.test/node/gpu1/8000")!),
            session: Self.session { request in
                #expect(request.url?.path == "/node/gpu1/8000/api/capabilities")
                return (Data(#"{"serverVersion":"t"}"#.utf8), 200)
            })
        _ = try await client.capabilities()
    }

    @Test func variantArtifactHashPreservesBaseURLPathPrefix() async throws {
        // Regression: the download-path routes used to build their URLs
        // manually and dropped a reverse-proxy/OOD path prefix.
        let payload = Data(#"{"name":"v"}"#.utf8)
        let client = ClusterClient(
            profile: ClusterConnectionProfile(
                baseURL: URL(string: "https://ood.test/node/gpu1/8000")!),
            token: "secret",
            session: Self.session { request in
                #expect(request.url?.path == "/node/gpu1/8000/api/bundles/download")
                #expect(request.url?.query()?.contains("path=") == true)
                let components = URLComponents(
                    url: request.url!, resolvingAgainstBaseURL: false)
                #expect(
                    components?.queryItems?.first(where: { $0.name == "path" })?.value
                        == "/srv/runs/v/variant.json")
                #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer secret")
                return (payload, 200)
            })

        let hash = try await client.variantArtifactHash(path: "/srv/runs/v/variant.json")

        // SHA-256 over the downloaded bytes, hex-encoded (64 chars).
        #expect(hash.count == 64)
    }

    @Test func downloadArtifactPreservesBaseURLPathPrefix() async throws {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(
                baseURL: URL(string: "https://ood.test/node/gpu1/8000")!),
            session: Self.session { request in
                #expect(request.url?.path == "/node/gpu1/8000/api/bundles/download")
                let components = URLComponents(
                    url: request.url!, resolvingAgainstBaseURL: false)
                #expect(
                    components?.queryItems?.first(where: { $0.name == "path" })?.value
                        == "/srv/runs/x/evidence.tar.gz")
                return (Data("bundle-bytes".utf8), 200)
            })
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "steerlab-download-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }

        let destination = try await client.downloadArtifact(
            path: "/srv/runs/x/evidence.tar.gz", to: directory)

        #expect(try Data(contentsOf: destination) == Data("bundle-bytes".utf8))
    }

    @Test func uploadedVariantSummaryDecodesServerPayload() throws {
        // Regression for B1: the server's variant.to_dict() omits createdAt, so
        // the upload response must decode without requiring it.
        let json = Data(
            """
            {
              "path": "/runs/v/variant.json", "hash": "abc", "runDirectory": "/runs/v",
              "variant": {"schemaVersion": 1, "name": "v", "baseModelID": "m",
                          "baseRevision": null, "adapters": [], "injections": [],
                          "bandWidth": 1, "alphaInNormUnits": true,
                          "neutralPCBasisPath": null, "promptMode": "chatAssistant",
                          "qwenThinkingEnabled": false, "temperature": 0.0,
                          "systemPrompt": null},
              "missingArtifacts": [], "compatibleWithServerModels": true
            }
            """.utf8)
        let decoded = try JSONDecoder().decode(UploadedVariant.self, from: json)
        #expect(decoded.variant.baseModelID == "m")
        #expect(decoded.compatibleWithServerModels)
    }

    @Test func vectorArtifactsDecodeTheServerCatalogPayload() async throws {
        // Canned JSON matching the server's /api/vectors field names
        // (asdict(catalog.VectorArtifact) | {"id": ...}).
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/vectors")
                return (
                    Data(
                        """
                        {"vectors": [{
                          "runDirectory": "/data/runs/2026-06-10T000000Z-french",
                          "name": "french",
                          "concept": "french",
                          "modelID": "Qwen/Qwen3-4B",
                          "revision": "abc123",
                          "layerCount": 36,
                          "hiddenSize": 2560,
                          "method": "lat",
                          "reading": "last token",
                          "residualNormSource": "neutral-corpus",
                          "hasResidualNorms": true,
                          "comparisonConcepts": ["calm", "french"],
                          "selectedTopics": ["work"],
                          "selectedSplits": ["build"],
                          "grandMeanPopulation": {
                            "calm": "calm-hash",
                            "french": "french-hash"
                          },
                          "extracted": "2026-06-10",
                          "id": "/data/runs/2026-06-10T000000Z-french/french"
                        }]}
                        """.utf8),
                    200)
            })

        let vectors = try await client.vectorArtifacts()
        let vector = try #require(vectors.first)
        #expect(vector.id == "/data/runs/2026-06-10T000000Z-french/french")
        #expect(vector.concept == "french")
        #expect(vector.modelID == "Qwen/Qwen3-4B")
        #expect(vector.revision == "abc123")
        #expect(vector.layerCount == 36)
        #expect(vector.resolvedMethod == "lat")
        #expect(vector.resolvedReadingPosition == "last token")
        #expect(vector.resolvedExtractionDate == "2026-06-10")
        #expect(vector.comparisonConcepts == ["calm", "french"])
        #expect(vector.selectedTopics == ["work"])
        #expect(vector.selectedSplits == ["build"])
        #expect(vector.grandMeanPopulation == [
            "calm": "calm-hash", "french": "french-hash",
        ])
        #expect(vector.stimulusSetHash == nil)  // not in the current payload
        // Older catalog without the per-layer norm tables: both arrays decode
        // to nil (no preview line, flag-based norm-unit gate).
        #expect(vector.normsPerLayer == nil)
        #expect(vector.residualNormPerLayer == nil)
    }

    @Test func vectorCatalogDecodesPerLayerNormTablesWhenPresent() async throws {
        // Newer servers add the two sidecar-named arrays to each record —
        // EXACTLY "normsPerLayer" / "residualNormPerLayer" (the pinned
        // contract with the server) — backing the slot preview line and the
        // per-layer norm-unit availability gate.
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { _ in
                (
                    Data(
                        """
                        {"vectors": [{
                          "runDirectory": "/data/runs/x", "name": "fear",
                          "concept": "fear", "modelID": "Qwen/Qwen3-4B",
                          "revision": "abc", "layerCount": 3, "hiddenSize": 2560,
                          "method": "meanDifference", "reading": "last token",
                          "residualNormSource": "neutral-corpus",
                          "hasResidualNorms": true,
                          "normsPerLayer": [1.5, 2.5, 4.0],
                          "residualNormPerLayer": [10.0, 20.0, 40.0],
                          "extracted": "2026-07-01", "id": "/data/runs/x/fear"
                        }]}
                        """.utf8),
                    200
                )
            })

        let vector = try #require(try await client.vectorArtifacts().first)
        #expect(vector.normsPerLayer == [1.5, 2.5, 4.0])
        #expect(vector.residualNormPerLayer == [10.0, 20.0, 40.0])
    }

    @Test func installModelPostsAndReturnsTheJobID() async throws {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/models/install")
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                #expect(object["modelID"] as? String == "Qwen/Qwen3-4B")
                #expect(object["revision"] as? String == "abc123")
                return (Data(#"{"jobId":"job-42"}"#.utf8), 200)
            })

        let jobID = try await client.installModel("Qwen/Qwen3-4B", revision: "abc123")
        #expect(jobID == "job-42")
    }

    @Test func installModelSurfacesThe400DetailBody() async throws {
        let detail =
            #"{"detail":"'Qwen/Qwen3-4B-MLX-4bit' is an MLX-quantized repo — install the family twin instead"}"#
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { _ in (Data(detail.utf8), 400) })

        await #expect {
            _ = try await client.installModel("Qwen/Qwen3-4B-MLX-4bit")
        } throws: { error in
            guard case ClusterClient.ClientError.badResponse(400, let body) = error else {
                return false
            }
            return body == detail
        }
    }

    @Test func switchWorkspacePostsRootAndDecodesTheInfoPayload() async throws {
        // POST /api/workspace/switch answers with the server's post-switch
        // /api/info payload, so the store can adopt the new pairing truth
        // (root + source-checkout verdict) without a second round trip.
        let response = #"""
            {"switched": true, "service": "steerlab-server",
             "root": "/srv/new-ws", "rootLooksLikeSourceCheckout": false,
             "device": null, "devices": ["cpu"], "loadedModels": [],
             "capabilities": {"serverVersion": "1.0"}}
            """#
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            token: "secret",
            session: Self.session { request in
                #expect(request.url?.path == "/api/workspace/switch")
                #expect(request.httpMethod == "POST")
                #expect(request.value(forHTTPHeaderField: "Authorization") == "Bearer secret")
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                #expect(object["root"] as? String == "/srv/new-ws")
                return (Data(response.utf8), 200)
            })

        let info = try await client.switchWorkspace(toRoot: "/srv/new-ws")
        #expect(info.root == "/srv/new-ws")
        #expect(info.rootLooksLikeSourceCheckout == false)
        #expect(info.service == "steerlab-server")
    }

    @Test func switchWorkspaceRefusalSurfacesTheServerDetail() async throws {
        // Busy-jobs / containment refusals arrive as badResponse with the
        // server's detail verbatim — the store renders it in `status`.
        let detail =
            #"{"detail":"cannot switch workspace while 1 job(s) are not terminal: j1 (run: running) — wait or cancel them first"}"#
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { _ in (Data(detail.utf8), 409) })

        await #expect {
            _ = try await client.switchWorkspace(toRoot: "/srv/new-ws")
        } throws: { error in
            guard case ClusterClient.ClientError.badResponse(409, let body) = error else {
                return false
            }
            return body == detail
        }
    }

    @Test func capabilitiesDecodeTheWorkspaceSwitchBlock() throws {
        // The "workspace" capability block gates every switch affordance:
        // absent (older server) means unsupported; "switchable" is the live
        // deployment policy verdict; "parent" the allowlist root.
        let gated = try JSONDecoder().decode(
            ClusterCapabilities.self,
            from: Data(#"""
                {"serverVersion": "1.0", "workspace":
                 {"switch": true, "switchable": false, "parent": "/scratch/u"}}
                """#.utf8))
        #expect(gated.supportsWorkspaceSwitch)
        #expect(gated.workspaceSwitchAllowed == false)
        #expect(gated.workspaceSwitchParent == "/scratch/u")

        let open = try JSONDecoder().decode(
            ClusterCapabilities.self,
            from: Data(
                #"{"workspace": {"switch": true, "switchable": true, "parent": null}}"#
                    .utf8))
        #expect(open.supportsWorkspaceSwitch)
        #expect(open.workspaceSwitchAllowed)
        #expect(open.workspaceSwitchParent == nil)

        let older = try JSONDecoder().decode(
            ClusterCapabilities.self, from: Data(#"{"serverVersion": "0.9"}"#.utf8))
        #expect(older.supportsWorkspaceSwitch == false)
        #expect(older.workspaceSwitchAllowed == false)
    }

    @Test func capabilitiesDecodeVariantStudySampling() throws {
        // Study-owned sampling for saved agents (2026-07-21): the
        // "remoteStudy.variantStudySampling" flag gates stochastic
        // saved-agent submissions — absent (older server) means
        // unsupported, and the app refuses instead of submitting a design
        // that would run the agents greedy while the baseline samples.
        let current = try JSONDecoder().decode(
            ClusterCapabilities.self,
            from: Data(
                #"{"remoteStudy": {"submitBundle": true, "variantStudySampling": true}}"#
                    .utf8))
        #expect(current.supportsVariantStudySampling)

        let dotted = try JSONDecoder().decode(
            ClusterCapabilities.self,
            from: Data(
                #"{"features": {"remoteStudy.variantStudySampling": true}}"#.utf8))
        #expect(dotted.supportsVariantStudySampling)

        let older = try JSONDecoder().decode(
            ClusterCapabilities.self,
            from: Data(#"{"remoteStudy": {"submitBundle": true}}"#.utf8))
        #expect(older.supportsVariantStudySampling == false)
        let empty = try JSONDecoder().decode(
            ClusterCapabilities.self, from: Data(#"{"serverVersion": "0.9"}"#.utf8))
        #expect(empty.supportsVariantStudySampling == false)
    }

    @Test func evidenceImporterRejectsUnsafeRunID() {
        #expect(!EvidenceBundleImporter.isSafeComponent("../escape"))
        #expect(!EvidenceBundleImporter.isSafeComponent("a/b"))
        #expect(!EvidenceBundleImporter.isSafeComponent(""))
        #expect(EvidenceBundleImporter.isSafeComponent("2026-07-01T000000Z-exp-demo-run"))
    }

    @Test func pipelineEvidenceBundleImportsEveryDeclaredDirectory() throws {
        // Sixth review round, the essential round-trip: a pipeline bundle's
        // EVIDENCE lives in sibling stage/agent run dirs — the importer must
        // bring every declared one home (verified), place the hash-pinned
        // portable ledger inside the pipeline dir, and skip siblings that
        // already exist locally (immutable, timestamp-named runs).
        try ExperimentRootOverrideLock.withTempRoot(prefix: "evimport") { _ in
            func sha(_ data: Data) -> String {
                SHA256.hash(data: data)
                    .map { String(format: "%02x", $0) }.joined()
            }
            let fm = FileManager.default
            let staging = fm.temporaryDirectory.appending(
                component: "bundle-\(UUID().uuidString)")
            defer { try? fm.removeItem(at: staging) }
            let pipelineID = "20260718T000003000-exp-chain-pipeline"
            let stageID = "20260718T000001000-exp-chain-validate"
            let agentID = "20260718T000002000-variant-chain-fear-agent"
            var entries: [[String: Any]] = []
            func plant(_ runID: String, _ file: String,
                       _ contents: String) throws {
                let dir = staging.appending(components: "runs", runID)
                try fm.createDirectory(at: dir, withIntermediateDirectories: true)
                let data = Data(contents.utf8)
                try data.write(to: dir.appending(component: file))
                entries.append(["path": "runs/\(runID)/\(file)",
                                "sha256": sha(data)])
            }
            try plant(pipelineID, "pipeline.json", #"{"schema": 2}"#)
            try plant(stageID, "validation-report.json", "{}")
            try plant(agentID, "chain-fear-agent.json", "{}")
            let portable = Data(#"""
                {"kind": "pipelinePortable", "experiment": "chain",
                 "ledgerSchema": 2, "disposition": "completed",
                 "updatedAt": "2026-07-18T12:00:00Z",
                 "stages": ["validate"],
                 "stageStatus": {"validate": "completed"},
                 "stageRuns": {"validate": "\#(stageID)"}}
                """#.utf8)
            try portable.write(
                to: staging.appending(component: "steerlab-pipeline.json"))
            let meta: [String: Any] = [
                "runID": pipelineID,
                "entries": entries,
                "pipelineStageDirectories": [stageID, agentID],
                "pipelinePortableSha256": sha(portable),
            ]
            try JSONSerialization.data(withJSONObject: meta).write(
                to: staging.appending(component: "steerlab-evidence.json"))
            let bundle = fm.temporaryDirectory.appending(
                component: "evidence-\(UUID().uuidString).tar.gz")
            defer { try? fm.removeItem(at: bundle) }
            let tar = Process()
            tar.executableURL = URL(filePath: "/usr/bin/tar")
            tar.currentDirectoryURL = staging
            tar.arguments = ["-czf", bundle.path, "."]
            try tar.run()
            tar.waitUntilExit()
            #expect(tar.terminationStatus == 0)

            // A same-ID collision with DIFFERENT content refuses — never a
            // silent substitution of local bytes for the bundle's evidence
            // (seventh round; timestamp naming is not identity).
            let existingAgent = ExperimentStore.runsDirectory.appending(
                component: agentID)
            try fm.createDirectory(
                at: existingAgent, withIntermediateDirectories: true)
            try Data("local".utf8).write(
                to: existingAgent.appending(component: "chain-fear-agent.json"))
            #expect(throws: (any Error).self) {
                _ = try EvidenceBundleImporter.importEvidenceBundle(bundle)
            }
            let runs = ExperimentStore.runsDirectory
            // The refusal rolled everything back — no partial import.
            #expect(!fm.fileExists(
                atPath: runs.appending(component: pipelineID).path))
            #expect(!fm.fileExists(
                atPath: runs.appending(component: stageID).path))

            // A collision PROVEN identical (every declared file matching
            // its hash) is skipped, and the rest imports.
            try Data("{}".utf8).write(
                to: existingAgent.appending(component: "chain-fear-agent.json"))
            let imported = try EvidenceBundleImporter.importEvidenceBundle(bundle)
            #expect(imported.lastPathComponent == pipelineID)
            #expect(fm.fileExists(
                atPath: runs.appending(components: pipelineID, "pipeline.json").path))
            // The declared STAGE directory came home, verified.
            #expect(fm.fileExists(
                atPath: runs.appending(
                    components: stageID, "validation-report.json").path))
            // The portable ledger landed inside the pipeline dir…
            let localPortable = runs.appending(
                components: pipelineID, "pipeline-portable.json")
            #expect(try Data(contentsOf: localPortable) == portable)
            // …and the LOCAL catalog consumes it (retained is not
            // consumed): stage references resolve by run ID.
            let summaries = LocalPipelineCatalog.summaries(experiment: "chain")
            let summary = try #require(
                summaries.first { $0.run == pipelineID })
            #expect(summary.disposition == "completed")
            #expect(summary.stages.first?.stage == "validate")
            #expect(summary.stages.first?.runID == stageID)
        }
    }

    @Test func tokenStoreKeyIsPerHostPort() {
        #expect(ClusterTokenStore.key(forURLString: "http://localhost:8000") == "localhost:8000")
        #expect(ClusterTokenStore.key(forURLString: "https://ood.test") == "ood.test")
    }

    @Test func vectorCatalogDecodesFreshnessPinsWhenPresent() async throws {
        // Newer servers stamp stimulusSetHash + neutralCorpusHash into the
        // catalog payload; both must decode and survive normalization.
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { _ in
                (
                    Data(
                        """
                        {"vectors": [{
                          "runDirectory": "/data/runs/x", "name": "fear",
                          "concept": "fear", "modelID": "Qwen/Qwen3-4B",
                          "revision": "abc", "layerCount": 36, "hiddenSize": 2560,
                          "method": "meanDifference", "reading": "last token",
                          "residualNormSource": null, "hasResidualNorms": false,
                          "extracted": "2026-07-01", "id": "/data/runs/x/fear",
                          "stimulusSetHash": "hash-a",
                          "neutralCorpusHash": "hash-n"
                        }]}
                        """.utf8),
                    200
                )
            })

        let vector = try #require(try await client.vectorArtifacts().first)
        #expect(vector.stimulusSetHash == "hash-a")
        #expect(vector.neutralCorpusHash == "hash-n")
        let normalized = SubstrateVectorRecord(remote: vector)
        #expect(normalized.stimulusSetHash == "hash-a")
        #expect(normalized.neutralCorpusHash == "hash-n")
    }

    @Test func runsListingDecodesServerRunSummaries() async throws {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/runs")
                return (
                    Data(
                        """
                        {"runs": [{
                          "id": "2026-07-03T000000Z-concept-fear",
                          "path": "/srv/runs/2026-07-03T000000Z-concept-fear",
                          "task": null,
                          "hasReport": false,
                          "hasGenerations": true,
                          "hasCosineMatrix": false,
                          "vectorNames": ["fear"],
                          "files": ["fear.safetensors", "fear.json", "generations.jsonl"]
                        }]}
                        """.utf8),
                    200)
            })

        let runs = try await client.runs()

        #expect(runs.count == 1)
        #expect(runs[0].id == "2026-07-03T000000Z-concept-fear")
        #expect(runs[0].hasGenerations)
        #expect(runs[0].vectorNames == ["fear"])
    }

    @Test func conceptExtractPostsMethodAndPoolAndReturnsJobID() async throws {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/concept/fear/extract")
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                #expect(object["method"] as? String == "lat")
                #expect(object["poolFromToken"] as? Int == 50)
                return (Data(#"{"jobId": "job-9"}"#.utf8), 200)
            })

        let jobID = try await client.conceptExtract(
            concept: "fear", method: "lat", poolFromToken: 50)

        #expect(jobID == "job-9")
    }

    // MARK: Gemma Scope (server-side SAE cross-check)

    @Test func gemmaScopeInfoSendsModelQueryAndDecodesRecommendation() async throws {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/gemmascope/info")
                let components = URLComponents(
                    url: request.url!, resolvingAgainstBaseURL: false)
                #expect(
                    components?.queryItems?.first(where: { $0.name == "model" })?.value
                        == "google/gemma-3-4b-it")
                return (
                    Data(
                        """
                        {"available": true, "size": "4b", "tuning": "it",
                         "repository": "google/gemma-scope-2-4b-it",
                         "release": "gemma-scope-2-4b-it-res",
                         "saeID": "layer_17_width_16k_l0_medium",
                         "site": "resid_post", "layer": 17,
                         "availableLayers": [9, 17, 22, 29]}
                        """.utf8),
                    200)
            })

        let info = try await client.gemmaScopeInfo(model: "google/gemma-3-4b-it")

        #expect(info.available)
        #expect(info.release == "gemma-scope-2-4b-it-res")
        #expect(info.saeID == "layer_17_width_16k_l0_medium")
        #expect(info.layer == 17)
        #expect(info.availableLayers == [9, 17, 22, 29])
    }

    @Test func gemmaScopeInfoDecodesUnavailableModels() async throws {
        // Non-Gemma-3 ids answer available:false + reason, not a 4xx.
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { _ in
                (
                    Data(
                        #"{"available": false, "reason": "Gemma Scope SAEs exist only for Gemma 3 (4b/12b/27b)."}"#
                            .utf8),
                    200)
            })

        let info = try await client.gemmaScopeInfo(model: "Qwen/Qwen3-4B")

        #expect(!info.available)
        #expect(info.reason?.contains("Gemma 3") == true)
    }

    @Test func gemmaScopeRunPostsArtifactAddressAndSAEPins() async throws {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/gemmascope/run")
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                #expect(object["vectorPath"] as? String == "/data/runs/x")
                #expect(object["name"] as? String == "fear")
                #expect(object["modelID"] as? String == "google/gemma-3-4b-it")
                #expect(object["layer"] as? Int == 17)
                #expect(object["release"] as? String == "gemma-scope-2-4b-it-res")
                #expect(object["saeID"] as? String == "layer_17_width_16k_l0_medium")
                return (Data(#"{"jobId": "job-77"}"#.utf8), 200)
            })

        let jobID = try await client.gemmaScopeRun(
            vectorRunDirectory: "/data/runs/x",
            name: "fear",
            modelID: "google/gemma-3-4b-it",
            layer: 17,
            release: "gemma-scope-2-4b-it-res",
            saeID: "layer_17_width_16k_l0_medium")

        #expect(jobID == "job-77")
    }

    @Test func gemmaScopeRunUnwrapsTheRefusalDetail() async throws {
        // The server's self-naming 400 must surface verbatim, not as raw JSON.
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { _ in
                (Data(#"{"detail":"not a Gemma 3 model / no SAE available"}"#.utf8), 400)
            })

        await #expect {
            _ = try await client.gemmaScopeRun(
                vectorRunDirectory: "/data/runs/x", name: "fear")
        } throws: { error in
            guard case ClusterClient.ClientError.badResponse(400, let text) = error else {
                return false
            }
            return text == "not a Gemma 3 model / no SAE available"
        }
    }

    @Test func gemmaScopeReportFetchesTheRunFileAndDecodesPinnedRows() async throws {
        // The report is fetched as a run file; its rows are the same pinned
        // shape as local report rows (feature/cosine/sparsity/decoderValues).
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(
                    request.url?.path
                        == "/api/runs/2026-07-07T000000Z-gemmascope-fear/file")
                let components = URLComponents(
                    url: request.url!, resolvingAgainstBaseURL: false)
                #expect(
                    components?.queryItems?.first(where: { $0.name == "name" })?.value
                        == "gemmascope-report.json")
                return (
                    Data(
                        """
                        {"release": "gemma-scope-2-4b-it-res",
                         "saeID": "layer_17_width_16k_l0_medium",
                         "layer": 17, "decoderShape": [16384, 2560],
                         "topPositive": [{"feature": 3, "cosine": 0.42,
                                          "sparsity": null,
                                          "decoderValues": [0.1, 0.2]}],
                         "topNegative": [{"feature": 9, "cosine": -0.31,
                                          "sparsity": 0.5,
                                          "decoderValues": null}],
                         "topAbsolute": [{"feature": 3, "cosine": 0.42,
                                          "sparsity": null,
                                          "decoderValues": [0.1, 0.2]}]}
                        """.utf8),
                    200)
            })

        let report = try await client.gemmaScopeReport(
            runID: "2026-07-07T000000Z-gemmascope-fear")

        #expect(report.saeID == "layer_17_width_16k_l0_medium")
        #expect(report.layer == 17)
        #expect(report.topPositive.first?.feature == 3)
        #expect(report.topPositive.first?.decoderValues == [0.1, 0.2])
        // A row without decoder values decodes (and gates import client-side).
        #expect(report.topNegative.first?.decoderValues == nil)
        #expect(report.topNegative.first?.sparsity == 0.5)
    }

    @Test func gemmaScopeImportPostsReportPathAndFeature() async throws {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/gemmascope/import")
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                #expect(
                    object["reportPath"] as? String
                        == "/data/runs/gemmascope-fear/gemmascope-report.json")
                #expect(object["feature"] as? Int == 3)
                #expect(object.count == 2)
                return (
                    Data(
                        #"{"vectorPath": "/data/runs/sae-feature-3", "name": "sae-feature-3"}"#
                            .utf8),
                    200)
            })

        let imported = try await client.gemmaScopeImport(
            reportPath: "/data/runs/gemmascope-fear/gemmascope-report.json", feature: 3)

        #expect(imported.vectorPath == "/data/runs/sae-feature-3")
        #expect(imported.name == "sae-feature-3")
    }

    @Test func saveConceptPostsThePairedTextsToTheSaveRoute() async throws {
        // The server-workspace vector build persists the drafted dataset on
        // the SERVER before queuing the extract job — same route/body as the
        // server's own browser workbench: POST /api/concept/{name}/save with
        // {"positive": [texts], "negative": [texts]}.
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/concept/fear/save")
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                #expect(object["positive"] as? [String] == ["I am terrified", "dread fills me"])
                #expect(object["negative"] as? [String] == ["I feel calm", "all is well"])
                #expect(object.count == 2)  // exactly the two arrays
                return (
                    Data(
                        #"{"name":"fear","positiveCount":2,"negativeCount":2,"hash":"abc123"}"#
                            .utf8),
                    200)
            })

        let result = try await client.saveConcept(
            name: "fear",
            positive: ["I am terrified", "dread fills me"],
            negative: ["I feel calm", "all is well"])

        #expect(result.name == "fear")
        #expect(result.positiveCount == 2)
        #expect(result.negativeCount == 2)
        #expect(result.hash == "abc123")
    }

    @Test func saveConceptDecodesAHashlessResponse() async throws {
        // authoring.save_concept omits "hash" until both sides have rows.
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { _ in
                (Data(#"{"name":"fear","positiveCount":1,"negativeCount":0}"#.utf8), 200)
            })
        let result = try await client.saveConcept(
            name: "fear", positive: ["x"], negative: [])
        #expect(result.hash == nil)
        #expect(result.contentHash == nil)
    }

    @Test func saveConceptDecodesTheContentHash() async throws {
        // The upload sync verifies the server recomputed the SAME
        // cross-engine content hash — the response must carry it through.
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { _ in
                (
                    Data(
                        (#"{"name":"fear","positiveCount":1,"negativeCount":1,"#
                            + #""hash":"raw123","contentHash":"content456"}"#)
                            .utf8),
                    200
                )
            })
        let result = try await client.saveConcept(
            name: "fear", positive: ["x"], negative: ["y"])
        #expect(result.hash == "raw123")
        #expect(result.contentHash == "content456")
    }

    @Test func remoteConceptsListsTheServerCatalogWithContentHashes() async throws {
        // Concept Lab's server-authoritative browse: GET /api/concepts rows
        // decode with the drift-comparator contentHash; a legacy row without
        // one still decodes (drift stays "unknown", never a failure).
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/concepts")
                #expect(request.httpMethod == "GET")
                return (
                    Data(
                        (#"{"concepts": [{"name": "fear", "positiveCount": 12,"#
                            + #" "negativeCount": 12, "hasValidation": true,"#
                            + #" "hasMarkers": false, "contentHash": "abc123"},"#
                            + #" {"name": "legacy", "positiveCount": 3,"#
                            + #" "negativeCount": 3}]}"#).utf8),
                    200
                )
            })

        let concepts = try await client.remoteConcepts()

        #expect(concepts.count == 2)
        #expect(concepts[0].id == "fear")
        #expect(concepts[0].positiveCount == 12)
        #expect(concepts[0].hasValidation == true)
        #expect(concepts[0].contentHash == "abc123")
        #expect(concepts[1].name == "legacy")
        #expect(concepts[1].hasValidation == nil)
        #expect(concepts[1].contentHash == nil)
    }

    @Test func remoteConceptContentsFetchesTheFullTexts() async throws {
        // "Fetch from server…" reads the full-texts route with the
        // verification hash.
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/concept/fear/full")
                #expect(request.httpMethod == "GET")
                return (
                    Data(
                        (#"{"name": "fear", "positive": ["p1", "p2"],"#
                            + #" "negative": ["n1"], "hasValidation": false,"#
                            + #" "hasMarkers": true, "contentHash": "abc123"}"#)
                            .utf8),
                    200
                )
            })

        let contents = try await client.remoteConceptContents(name: "fear")

        #expect(contents.positive == ["p1", "p2"])
        #expect(contents.negative == ["n1"])
        #expect(contents.contentHash == "abc123")
        // The fetched texts hash exactly as the drift comparator does — the
        // round-trip is verifiable end to end.
        #expect(
            ConceptBuilder.stimulusContentHash(
                positive: contents.positive, negative: contents.negative) != nil)
    }

    @Test func saveProbeItemsPostsRowsWithTextAndExpresses() async throws {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/concept/fear/probe-items")
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                let rows = try #require(object["rows"] as? [[String: Any]])
                #expect(rows.count == 2)
                #expect(rows[0]["text"] as? String == "my heart races")
                #expect(rows[0]["expresses"] as? Bool == true)
                #expect(rows[0]["id"] as? String == "fear-probe-001")
                #expect(rows[0]["split"] as? String == "validation")
                #expect(rows[1]["expresses"] as? Bool == false)
                return (
                    Data(#"{"concept":"fear","items":2,"positive":1,"negative":1}"#.utf8),
                    200)
            })

        try await client.saveProbeItems(
            concept: "fear",
            rows: [
                ConceptBuilder.ProbeExample(
                    id: "fear-probe-001", text: "my heart races",
                    expresses: true, topic: nil, split: "validation", notes: nil),
                ConceptBuilder.ProbeExample(
                    id: "fear-probe-002", text: "the lake is placid",
                    expresses: false, topic: nil, split: "validation", notes: nil),
            ])
    }

    @Test func saveStoriesPostsMultiConceptRowsToTheStoriesRoute() async throws {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/multiconcept/fear/stories")
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                let rows = try #require(object["rows"] as? [[String: Any]])
                #expect(rows.count == 1)
                #expect(rows[0]["concept"] as? String == "fear")
                #expect(rows[0]["text"] as? String == "The floor creaked behind her…")
                #expect(rows[0]["topic"] as? String == "night-house")
                #expect(rows[0]["split"] as? String == "build")
                return (Data(#"{"concept":"fear","stories":1,"hash":"dead"}"#.utf8), 200)
            })

        try await client.saveStories(
            concept: "fear",
            rows: [
                StimulusSet.MultiConceptStimulus(
                    id: "fear-001", concept: "fear", topic: "night-house",
                    text: "The floor creaked behind her…", split: "build")
            ])
    }

    @Test func storyConceptsDecodesTheServerListing() async throws {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/multiconcept/concepts")
                return (
                    Data(
                        """
                        {"concepts": [{"concept": "awe", "stories": 24},
                                      {"concept": "fear", "stories": 36}]}
                        """.utf8),
                    200)
            })

        #expect(try await client.storyConcepts() == ["awe", "fear"])
    }

    @Test func vectorCatalogDecodesTheSubstrateStampWhenPresent() async throws {
        // The server catalog exposes the sidecar's engine stamp under the
        // SAME key as the local sidecar ("substrate") — the picker-filtering
        // contract. Absent (older servers) must decode nil, not fail.
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { _ in
                (
                    Data(
                        """
                        {"vectors": [{
                          "runDirectory": "/data/runs/x", "name": "fear",
                          "concept": "fear", "modelID": "Qwen/Qwen3-4B",
                          "revision": "abc", "layerCount": 3, "hiddenSize": 2560,
                          "method": "meanDifference", "reading": "last token",
                          "residualNormSource": null, "hasResidualNorms": false,
                          "extracted": "2026-07-01", "id": "/data/runs/x/fear",
                          "substrate": "python-hf-transformers"
                        }, {
                          "runDirectory": "/data/runs/y", "name": "awe",
                          "concept": "awe", "modelID": "Qwen/Qwen3-4B",
                          "revision": "abc", "layerCount": 3, "hiddenSize": 2560,
                          "method": "meanDifference", "reading": "last token",
                          "residualNormSource": null, "hasResidualNorms": false,
                          "extracted": "2026-07-01", "id": "/data/runs/y/awe"
                        }]}
                        """.utf8),
                    200
                )
            })

        let vectors = try await client.vectorArtifacts()
        #expect(vectors[0].substrate == "python-hf-transformers")
        #expect(vectors[1].substrate == nil)
        // The filtering rules read exactly this field.
        #expect(
            WorkspaceScoping.offerableForServerSteering(substrate: vectors[0].substrate))
        #expect(
            WorkspaceScoping.offerableForServerSteering(substrate: vectors[1].substrate))
        #expect(
            !WorkspaceScoping.offerableForLocalSteering(substrate: vectors[0].substrate))
    }

    @Test func fitReaderPostsRegistryContractBodyAndReturnsJobID() async throws {
        let pairsJSONL = "{\"concept\":\"fear\"}\n"
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/reader/fit")
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                #expect(object["concept"] as? String == "fear")
                #expect(object["templateID"] as? String == "amount-in-scenario-v1")
                // Exactly one of templateID/templateJSON on the wire: the nil
                // one is ABSENT, not JSON null.
                #expect(object["templateJSON"] == nil)
                #expect(object["pairsJSONL"] as? String == pairsJSONL)
                #expect(object["layers"] as? [Int] == [10, 11])
                #expect(object["outputName"] as? String == "fear-reader")
                return (Data(#"{"jobId": "job-7"}"#.utf8), 200)
            })

        let jobID = try await client.fitReader(
            concept: "fear", templateID: "amount-in-scenario-v1",
            pairsJSONL: pairsJSONL, layers: [10, 11], outputName: "fear-reader")

        #expect(jobID == "job-7")
    }

    @Test func fitReaderSendsCustomTemplateJSONWithoutTemplateID() async throws {
        let templateJSON = #"{"id":"custom-fear-v1","text":"S: {{stimulus}} q"}"#
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/reader/fit")
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                #expect(object["templateID"] == nil)
                #expect(object["templateJSON"] as? String == templateJSON)
                // Optional knobs stay off the wire when unset.
                #expect(object["layers"] == nil)
                #expect(object["outputName"] == nil)
                return (Data(#"{"jobId": "job-8"}"#.utf8), 200)
            })

        let jobID = try await client.fitReader(
            concept: "fear", templateJSON: templateJSON, pairsJSONL: "{}\n")

        #expect(jobID == "job-8")
    }

    @Test func fitReaderOmitsModelIDWhenUnpinned() async throws {
        // Backwards-compatible wire shape: no modelID key at all (not null)
        // when the builder does not pin one — the server then uses whatever
        // model is loaded, the pre-existing behavior.
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                #expect(object["modelID"] == nil)
                #expect(object["revision"] == nil)
                return (Data(#"{"jobId": "job-10"}"#.utf8), 200)
            })

        _ = try await client.fitReader(
            concept: "fear", templateID: "amount-in-scenario-v1", pairsJSONL: "{}\n")
    }

    @Test func fitReaderPinsTheBuilderSelectedModel() async throws {
        // Item A3: the fit request carries the builder's selected server
        // model so the server acquires exactly that model for the fit.
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/reader/fit")
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                #expect(object["modelID"] as? String == "Qwen/Qwen3-4B")
                #expect(object["revision"] as? String == "abc123")
                #expect(object["concept"] as? String == "fear")
                return (Data(#"{"jobId": "job-11"}"#.utf8), 200)
            })

        let jobID = try await client.fitReader(
            concept: "fear", modelID: "Qwen/Qwen3-4B", revision: "abc123",
            templateID: "amount-in-scenario-v1", pairsJSONL: "{}\n")

        #expect(jobID == "job-11")
    }

    @Test func backfillNormsPostsThePinnedBodyShapeAndReturnsJobID() async throws {
        // The norm-backfill route body is a pinned cross-engine contract:
        // {"vectorID", "neutralCorpusPath", "modelID"?, "revision"?,
        //  "outputName"?} — same durable-job pattern as reader fit.
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/vectors/backfill-norms")
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                #expect(object["vectorID"] as? String == "/data/runs/x/fear")
                #expect(
                    object["neutralCorpusPath"] as? String
                        == "prompts/neutral/corpus.jsonl")
                #expect(object["modelID"] as? String == "Qwen/Qwen3-4B")
                #expect(object["revision"] as? String == "abc123")
                #expect(object["outputName"] as? String == "fear-norms")
                return (Data(#"{"jobId": "job-13"}"#.utf8), 200)
            })

        let jobID = try await client.backfillNorms(
            vectorID: "/data/runs/x/fear",
            neutralCorpusPath: "prompts/neutral/corpus.jsonl",
            modelID: "Qwen/Qwen3-4B",
            revision: "abc123",
            outputName: "fear-norms")

        #expect(jobID == "job-13")
    }

    @Test func backfillNormsKeepsOptionalKeysOffTheWireWhenUnset() async throws {
        // Backwards-compatible wire shape: no modelID/revision/outputName keys
        // at all (not JSON null) when unpinned — the server then measures on
        // whatever model matches the artifact.
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                #expect(object["vectorID"] as? String == "/data/runs/x/fear")
                #expect(object["modelID"] == nil)
                #expect(object["revision"] == nil)
                #expect(object["outputName"] == nil)
                return (Data(#"{"jobId": "job-14"}"#.utf8), 200)
            })

        let jobID = try await client.backfillNorms(
            vectorID: "/data/runs/x/fear",
            neutralCorpusPath: "prompts/neutral/corpus.jsonl")

        #expect(jobID == "job-14")
    }

    @Test func variantDetailDecodesSpecAndHash() async throws {
        // Canned JSON matching the server's GET /api/variant/detail response:
        // {"variant": <upload-payload spec dict>, "hash": "<sha256>"}. The
        // server's variant.to_dict() omits createdAt — tolerant decode.
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/variant/detail")
                let components = URLComponents(
                    url: request.url!, resolvingAgainstBaseURL: false)
                #expect(
                    components?.queryItems?.first(where: { $0.name == "path" })?.value
                        == "/srv/runs/v/variant.json")
                return (
                    Data(
                        """
                        {"variant": {"schemaVersion": 1, "name": "fear-mix",
                          "baseModelID": "Qwen/Qwen3-4B", "baseRevision": null,
                          "adapters": [],
                          "injections": [{"concept": "fear",
                            "vectorArtifactID": "/srv/runs/x/fear",
                            "layer": 18, "alpha": 4.5}],
                          "bandWidth": 3, "alphaInNormUnits": true,
                          "neutralPCBasisPath": null,
                          "promptMode": "chatAssistant",
                          "qwenThinkingEnabled": false, "temperature": 0.0,
                          "systemPrompt": "be terse"},
                         "hash": "deadbeef"}
                        """.utf8),
                    200)
            })

        let detail = try await client.variantDetail(path: "/srv/runs/v/variant.json")

        #expect(detail.hash == "deadbeef")
        #expect(detail.variant.name == "fear-mix")
        #expect(detail.variant.baseModelID == "Qwen/Qwen3-4B")
        #expect(detail.variant.bandWidth == 3)
        #expect(detail.variant.alphaInNormUnits)
        #expect(detail.variant.systemPrompt == "be terse")
        let injection = try #require(detail.variant.injections.first)
        #expect(injection.vectorArtifactID == "/srv/runs/x/fear")
        #expect(injection.layer == 18)
        #expect(injection.alpha == 4.5)
    }

    @Test func adaptersListingDecodesTheServerPayload() async throws {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/adapters")
                return (
                    Data(
                        """
                        {"adapters": [{"name": "adapters",
                          "adapterDirectory": "/srv/runs/2026-07-02T000000Z-lora/artifacts/adapters"}]}
                        """.utf8),
                    200)
            })

        let adapters = try await client.adapters()
        #expect(adapters.count == 1)
        #expect(adapters[0].id.hasSuffix("artifacts/adapters"))
    }

    @Test func streamVariantChatStoredSelectionSendsPathAndHashOnly() async throws {
        // Exactly one of variantPath(+variantHash) / variant on the wire:
        // the stored form must carry NO inline "variant" key.
        let sse = "data: {\"chunk\": \"hi\"}\n\ndata: {\"done\": true}\n\n"
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { _ in (Data(), 200) },
            streamSession: Self.session { request in
                #expect(request.url?.path == "/api/variant/generate/stream")
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                #expect(object["variantPath"] as? String == "/srv/runs/v/variant.json")
                #expect(object["variantHash"] as? String == "deadbeef")
                #expect(object["variant"] == nil)
                #expect(object["maxTokens"] as? Int == 128)
                #expect(object["promptMode"] as? String == "chatAssistant")
                #expect(object["stripInterventions"] as? Bool == false)
                return (Data(sse.utf8), 200)
            })

        let chunks = ChunkCollector()
        try await client.streamVariantChat(
            selection: .stored(path: "/srv/runs/v/variant.json", hash: "deadbeef"),
            messages: [ChatWireMessage(role: "user", content: "hello")],
            maxTokens: 128, temperature: nil, promptMode: "chatAssistant",
            systemPrompt: nil, stripInterventions: false
        ) { chunk in
            await chunks.add(chunk)
        }

        #expect(await chunks.chunks == ["hi"])
    }

    @Test func streamVariantChatInlineSelectionSendsTheUploadSchemaSpec() async throws {
        // The inline form: no variantPath/variantHash, and "variant" is the
        // upload-payload spec dict (ModelVariantArtifact's own encoding, so
        // the schemas match by construction). The done event's provenance
        // stamp ({"source": "inline"}) is surfaced through onMetadata.
        let sse = "data: {\"chunk\": \"ok\"}\n\n"
            + "data: {\"done\": true, \"variant\": {\"source\": \"inline\", \"name\": \"chat-inline\"}}\n\n"
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { _ in (Data(), 200) },
            streamSession: Self.session { request in
                #expect(request.url?.path == "/api/variant/generate/stream")
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                #expect(object["variantPath"] == nil)
                #expect(object["variantHash"] == nil)
                let variant = try #require(object["variant"] as? [String: Any])
                #expect(variant["baseModelID"] as? String == "Qwen/Qwen3-4B")
                #expect(variant["name"] as? String == "chat-inline")
                #expect(variant["bandWidth"] as? Int == 3)
                #expect(variant["alphaInNormUnits"] as? Bool == true)
                #expect(variant["promptMode"] as? String == "chatAssistant")
                let injections = try #require(variant["injections"] as? [[String: Any]])
                #expect(
                    injections.first?["vectorArtifactID"] as? String
                        == "/srv/runs/x/fear")
                return (Data(sse.utf8), 200)
            })

        let spec = InlineVariantComposer.compose(
            InlineVariantComposer.ControlState(
                baseModelID: "Qwen/Qwen3-4B",
                slots: [
                    InlineVariantComposer.Slot(
                        concept: "fear", vectorPath: "/srv/runs/x/fear",
                        layer: 18, alpha: 4.5)
                ],
                bandWidth: 3,
                alphaInNormUnits: true,
                promptMode: "chatAssistant",
                temperature: 0.7))

        let chunks = ChunkCollector()
        try await client.streamVariantChat(
            selection: .inline(spec),
            messages: [ChatWireMessage(role: "user", content: "hello")],
            maxTokens: 64, temperature: 0.7, promptMode: "chatAssistant",
            systemPrompt: nil, stripInterventions: false,
            onMetadata: { metadata in
                await chunks.record(metadata)
            }
        ) { chunk in
            await chunks.add(chunk)
        }

        #expect(await chunks.chunks == ["ok"])
        let metadata = try #require(await chunks.metadata.first)
        #expect(metadata["source"] == .string("inline"))
    }

    @Test func listReadersDecodesTheServerCatalogPayload() async throws {
        // Canned JSON matching the server's /api/readers field names
        // (asdict(catalog.ReaderSummary) | {"id": ...}).
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/readers")
                return (
                    Data(
                        """
                        {"readers": [{
                          "runDirectory": "/srv/runs/2026-07-03T000000Z-reader-fear",
                          "name": "reader-fear-layer12",
                          "concept": "fear",
                          "modelID": "Qwen/Qwen3-4B",
                          "revision": "abc123",
                          "substrate": "python-hf-transformers",
                          "layer": 12,
                          "templateID": "amount-in-scenario-v1",
                          "templateHash": "th",
                          "templateDivergence": null,
                          "datasetHash": "dh",
                          "latTokenPosition": "final",
                          "trainAccuracy": 1.0,
                          "heldOutAccuracy": 0.875,
                          "extracted": "2026-07-03",
                          "id": "/srv/runs/2026-07-03T000000Z-reader-fear/reader-fear-layer12.json"
                        }]}
                        """.utf8),
                    200)
            })

        let readers = try await client.listReaders()

        let reader = try #require(readers.first)
        #expect(reader.id.hasSuffix("reader-fear-layer12.json"))
        #expect(reader.concept == "fear")
        #expect(reader.modelID == "Qwen/Qwen3-4B")
        #expect(reader.substrate == "python-hf-transformers")
        #expect(reader.layer == 12)
        #expect(reader.heldOutAccuracy == 0.875)
        #expect(reader.templateDivergence == nil)
    }

    @Test func variantGenerateSendsTheInlineBodyShape() async throws {
        // Fixture for the server robustness path's request body: the variant
        // under test rides as the INLINE `variant` field (the upload-payload
        // schema — exactly what InlineVariantComposer sends), never a stored
        // path/hash, with greedy temperature and the baseline arm's
        // stripInterventions flag.
        let spec = ModelVariantArtifact(
            name: "fear-mix",
            baseModelID: "google/gemma-3-4b-it",
            injections: [
                .init(
                    concept: "fear",
                    vectorArtifactID: "/srv/runs/a/fear",
                    layer: 12,
                    alpha: 1.5)
            ],
            promptMode: "chatAssistant",
            qwenThinkingEnabled: false,
            temperature: 0.7,  // spec temperature — overridden per-request below
            systemPrompt: "")
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/variant/generate")
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                // Exactly the inline selection: no stored identity fields.
                #expect(object["variantPath"] == nil)
                #expect(object["variantHash"] == nil)
                let variant = try #require(object["variant"] as? [String: Any])
                #expect(variant["baseModelID"] as? String == "google/gemma-3-4b-it")
                let injections = try #require(variant["injections"] as? [[String: Any]])
                #expect(injections.first?["vectorArtifactID"] as? String == "/srv/runs/a/fear")
                let messages = try #require(object["messages"] as? [[String: String]])
                #expect(messages == [["role": "user", "content": "What is 2+2?"]])
                #expect(object["maxTokens"] as? Int == 24)
                #expect(object["temperature"] as? Double == 0)  // greedy, always
                #expect(object["promptMode"] as? String == "chatAssistant")
                #expect(object["stripInterventions"] as? Bool == true)  // baseline arm
                #expect(object["systemPrompt"] == nil)  // the spec's own prompt rules
                return (Data(#"{"output": "4", "modelID": "google/gemma-3-4b-it"}"#.utf8), 200)
            })

        let output = try await client.variantGenerate(
            selection: .inline(spec),
            messages: [ChatWireMessage(role: "user", content: "What is 2+2?")],
            maxTokens: 24,
            temperature: 0,
            promptMode: "chatAssistant",
            systemPrompt: nil,
            stripInterventions: true)

        #expect(output == "4")
    }

    @Test func variantBatteryEvaluateSendsThePinAndDecodesTheRecords() async throws {
        // The §23 wire: one request per SIDE, the battery as a PIN (path +
        // digest), and deliberately NO prompt mode / system prompt / token
        // cap — a format-2 battery's arming comes from the file, and the
        // caller having no way to send one is the point of the route.
        let spec = ModelVariantArtifact(
            name: "fear-mix", baseModelID: "google/gemma-3-4b-it",
            injections: [
                .init(concept: "fear", vectorArtifactID: "/srv/runs/a/fear",
                      layer: 12, alpha: 1.5)
            ],
            promptMode: "rawCompletion", qwenThinkingEnabled: true,
            temperature: 0.7, systemPrompt: "Respond only in JSON.")
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/variant/battery")
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                #expect(object["variantPath"] == nil && object["variantHash"] == nil)
                let variant = try #require(object["variant"] as? [String: Any])
                #expect(variant["baseModelID"] as? String == "google/gemma-3-4b-it")
                #expect(object["battery"] as? String == "prompts/batteries/basic.jsonl")
                #expect(object["batteryHash"] as? String == "feedface")
                #expect(object["stripInterventions"] as? Bool == true)
                // Nothing that could re-arm the battery may ride along.
                #expect(object["promptMode"] == nil)
                #expect(object["systemPrompt"] == nil)
                #expect(object["maxTokens"] == nil)
                #expect(object["temperature"] == nil)
                return (
                    Data(
                        #"""
                        {"battery": "prompts/batteries/basic.jsonl",
                         "batteryHash": "feedface", "batteryFormat": 2,
                         "stripInterventions": true, "armingIsolated": true,
                         "armingPromptMode": "chatAssistant",
                         "armingSystemPrompt": false, "armingMaxTokens": 24,
                         "advisory": null,
                         "items": [{"promptIndex": 0, "promptID": "basic-01",
                                    "prompt": "What is 17 + 26?", "answer": "43",
                                    "batteryFormat": 2, "armingIsolated": true,
                                    "armingPromptMode": "chatAssistant",
                                    "armingSystemPrompt": false,
                                    "armingMaxTokens": 24,
                                    "batteryHash": "feedface",
                                    "scoring": "choiceProbability",
                                    "options": ["41", "42", "43"],
                                    "choiceProbability": {"43": 0.8, "41": 0.1, "42": 0.1},
                                    "selected": "43", "output": "43",
                                    "correct": true}],
                         "summary": {"accuracy": 1.0, "itemCount": 1,
                                     "batteryHash": "feedface"}}
                        """#.utf8), 200)
            })

        let evaluation = try await client.variantBatteryEvaluate(
            selection: .inline(spec),
            battery: "prompts/batteries/basic.jsonl",
            batteryHash: "feedface",
            stripInterventions: true)

        #expect(evaluation.batteryFormat == 2)
        #expect(evaluation.armingIsolated && !evaluation.armingSystemPrompt)
        #expect(evaluation.summary.accuracy == 1.0)
        #expect(evaluation.summary.itemCount == 1)
        #expect(evaluation.advisory == nil)
        let item = try #require(evaluation.items.first)
        #expect(item.promptID == "basic-01")
        #expect(item.scoring == "choiceProbability")
        #expect(item.selected == "43" && item.output == "43" && item.correct)
        #expect(item.choiceProbability?["43"] == 0.8)
    }

    @Test func fineTuneTrainSendsInlineCorpusAndHyperparameters() async throws {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/finetune/train")
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                #expect(object["baseModelID"] as? String == "Qwen/Qwen3-4B")
                #expect(object["text"] as? String == "corpus text")
                #expect(object["rank"] as? Int == 16)
                #expect(object["iterations"] as? Int == 300)
                return (Data(#"{"jobId": "job-42"}"#.utf8), 200)
            })

        let jobID = try await client.fineTuneTrain(
            baseModelID: "Qwen/Qwen3-4B", text: "corpus text", name: "judicial",
            rank: 16, alpha: 32, iterations: 300, learningRate: 1e-4)

        #expect(jobID == "job-42")
    }

    @Test func geometryEncodesVectorRefsAndDecodesNullableCells() async throws {
        let record = RemoteVectorRecord(
            id: "runs/x/vectors", runDirectory: "runs/x", name: "french",
            concept: "french", modelID: "Qwen/Qwen3-4B", revision: nil,
            layerCount: 36, hiddenSize: 2560, method: "meanDifference",
            reading: nil, residualNormSource: nil, hasResidualNorms: true,
            extracted: nil, stimulusSetHash: nil, extractionMethod: nil,
            readingPosition: nil, extractionDate: nil)
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/geometry")
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                #expect(object["layer"] as? Int == 12)
                let vectors = try #require(object["vectors"] as? [[String: Any]])
                #expect(vectors.count == 1)
                #expect(vectors[0]["name"] as? String == "french")
                // The run DIRECTORY, never the catalog id (dir/name in one
                // string) — the id form made the route skip every vector and
                // answer a 0×0 matrix (see GeometryServerContractTests).
                #expect(vectors[0]["vectorPath"] as? String == "runs/x")
                #expect(vectors[0]["label"] as? String == "french · french")
                return (
                    Data(
                        """
                        {"labels": ["a", "b"],
                         "matrix": [[1.0, null], [null, 1.0]],
                         "layers": [12, 12]}
                        """.utf8),
                    200)
            })

        let result = try await client.geometry(vectors: [record], layer: 12)

        #expect(result.labels == ["a", "b"])
        #expect(result.layers == [12, 12])
        #expect(result.matrix[0][0] == 1.0)
        // null cells (pairs the server could not compare) decode as nil.
        #expect(result.matrix[0][1] == nil)
    }

    // MARK: Remote freeze (POST /api/authoring/{name}/freeze)

    /// A frozen-manifest response body in the pinned cross-engine
    /// `experiment.json` shape the server's freeze route returns (the raw
    /// manifest dict), optionally with the ADDITIVE `advisories` key.
    private static func frozenManifestJSON(advisories: String? = nil) -> String {
        """
        {
          "name": "pilot-1",
          "experimentDescription": "",
          "createdAt": "2026-07-13T00:00:00Z",
          "modelID": "Qwen/Qwen3-4B",
          "modelRevision": "abc123",
          "status": "frozen",
          "frozenAt": "2026-07-13T01:00:00Z",
          "freezeHash": "feedfacefeedfacefeedface",
          "frozenBy": "server",
          "gitCommit": "deadbeef00"\(advisories.map { ",\n  \"advisories\": \($0)" } ?? "")
        }
        """
    }

    @Test func freezeExperimentDecodesManifestAndAdvisories() async throws {
        let advisory =
            "validation evidence was produced on swift-mlx; runs on "
            + "python-hf-transformers should re-validate on-substrate"
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/authoring/pilot-1/freeze")
                #expect(request.httpMethod == "POST")
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                // force is only encoded when true — older servers must never
                // see an unknown/spurious field on the default path.
                #expect(object["force"] == nil)
                return (Data(Self.frozenManifestJSON(
                    advisories: #"["\#(advisory)"]"#).utf8), 200)
            })

        let result = try await client.freezeExperiment(name: "pilot-1")

        #expect(result.manifest.status == .frozen)
        #expect(result.manifest.frozenBy == "server")
        #expect(result.manifest.freezeHash == "feedfacefeedfacefeedface")
        #expect(result.manifest.gitCommit == "deadbeef00")
        #expect(result.advisories == [advisory])
    }

    @Test func freezeExperimentWithoutAdvisoriesDecodesEmpty() async throws {
        // The server omits the additive key when there is nothing to say
        // (and older servers omit it always) — decodes as empty, never fails.
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { _ in
                (Data(Self.frozenManifestJSON().utf8), 200)
            })

        let result = try await client.freezeExperiment(name: "pilot-1")

        #expect(result.advisories.isEmpty)
        #expect(result.manifest.name == "pilot-1")
    }

    @Test func freezeExperimentEncodesForceOnlyWhenTrue() async throws {
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                let body = try #require(Self.bodyData(from: request))
                let object = try #require(
                    JSONSerialization.jsonObject(with: body) as? [String: Any])
                #expect(object["force"] as? Bool == true)
                return (Data(Self.frozenManifestJSON().utf8), 200)
            })

        _ = try await client.freezeExperiment(name: "pilot-1", force: true)
    }

    @Test func freezeExperimentGateRefusalSurfacesTheServerDetailVerbatim() async throws {
        // The server's gate failures are the actionable text — the FastAPI
        // `{"detail": …}` envelope is unwrapped and the message kept verbatim.
        let detail =
            "cannot freeze 'pilot-1': no validate run matches its exact pins — "
            + "run 'experiment validate pilot-1' first, or force-freeze"
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { _ in
                (Data(#"{"detail": "\#(detail)"}"#.utf8), 400)
            })

        await #expect {
            try await client.freezeExperiment(name: "pilot-1")
        } throws: { error in
            guard let clientError = error as? ClusterClient.ClientError,
                case .badResponse(let code, let text) = clientError
            else { return false }
            return code == 400 && text == detail
        }
    }

    @Test func experimentDetailFetchesTheServerCopy() async throws {
        // The remote lifecycle's read op: residency (404 otherwise) + the
        // server copy's status, from the existing catalog detail route.
        let client = ClusterClient(
            profile: ClusterConnectionProfile(baseURL: URL(string: "http://server.test")!),
            session: Self.session { request in
                #expect(request.url?.path == "/api/experiment/pilot-1")
                #expect(request.httpMethod == "GET")
                return (
                    Data(#"{"name": "pilot-1", "status": "draft", "modelID": "Qwen/Qwen3-4B"}"#.utf8),
                    200)
            })

        let detail = try await client.experimentDetail(name: "pilot-1")

        #expect(detail.name == "pilot-1")
        #expect(detail.status == "draft")
    }

    private static func session(
        handler: @escaping @Sendable (URLRequest) throws -> (Data, Int)
    ) -> URLSession {
        MockClusterURLProtocol.handler = handler
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [MockClusterURLProtocol.self]
        return URLSession(configuration: configuration)
    }

    private static func bodyData(from request: URLRequest) -> Data? {
        if let body = request.httpBody {
            return body
        }
        guard let stream = request.httpBodyStream else {
            return nil
        }
        stream.open()
        defer { stream.close() }
        var data = Data()
        let bufferSize = 4096
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
        defer { buffer.deallocate() }
        while stream.hasBytesAvailable {
            let count = stream.read(buffer, maxLength: bufferSize)
            if count > 0 {
                data.append(buffer, count: count)
            } else {
                break
            }
        }
        return data
    }
}

/// The scripted server's one mutable cell for the precheck→push→recheck
/// flow test (class reference so the @Sendable handler closure can mutate
/// it; the serialized suite keeps access single-threaded).
private final class ScriptedManifestStore: @unchecked Sendable {
    var body: Data
    init(initial: Data) { body = initial }
}

/// Collects streamed chunks + done-event variant metadata from the SSE
/// callbacks (actor-isolated so the @Sendable closures can append safely).
private actor ChunkCollector {
    var chunks: [String] = []
    var metadata: [[String: JSONValue]] = []

    func add(_ chunk: String) { chunks.append(chunk) }
    func record(_ value: [String: JSONValue]) { metadata.append(value) }
}

private final class MockClusterURLProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var handler: (@Sendable (URLRequest) throws -> (Data, Int))?

    override class func canInit(with request: URLRequest) -> Bool { true }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let handler = Self.handler else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }
        do {
            let (data, status) = try handler(request)
            let response = HTTPURLResponse(
                url: request.url!,
                statusCode: status,
                httpVersion: "HTTP/1.1",
                headerFields: ["Content-Type": "application/json"])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: data)
            client?.urlProtocolDidFinishLoading(self)
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }

    override func stopLoading() {}
}
