import Foundation
import Testing
@testable import ExperimentKit

@MainActor struct ClusterProfileCoauthoringTests {
    private func profile() -> ClusterSiteProfile {
        var profile = ClusterSiteProfile.genericSlurm
        profile.name = "Example compute"
        profile.transport = .ssh(host: "user@login.example.invalid", proxyJump: nil, remotePort: 8080, vpnExpected: false)
        profile.constraints.computeEgress = .no
        profile.constraints.purgeDays = 30
        profile.constraints.storageRoots = ["workspace": "/scratch/workspace", "hfCache": "/work/cache", "metadata": "$HOME/.steerlab"]
        profile.environment.envPrefix = "$HOME/envs/steerlab"
        profile.policy.transferMethod = "rsync"
        profile.policy.externalServiceEgress = .no
        profile.policy.loginNodes = .init(hostnamePatterns: ["^login"], allowCompute: false, requireAllocation: true)
        var slurm = ClusterSiteProfile.SlurmSiteData()
        slurm.gpus = [.init(name: "A100", vramGB: 80, computeCapability: "sm_80")]
        slurm.defaultPartition = "gpu"
        slurm.defaultGres = "gpu:A100:1"
        slurm.jobDefaults = .init(memory: "80G", walltime: "02:00:00")
        profile.scheduler = .slurm(slurm)
        return profile
    }

    private func draft(_ profile: ClusterSiteProfile) throws -> ClusterProfileCoauthoring.Draft {
        let value = try JSONDecoder().decode(JSONValue.self, from: profile.encoded())
        return ClusterProfileCoauthoring.Draft(schemaVersion: 1, profile: value,
            sources: [.init(id: "policy", kind: "document", reference: "fictional-policy.md")],
            facts: ClusterProfileCoauthoring.requiredFactPaths(profile).map { path in
                .init(path: path, value: ClusterProfileCoauthoring.value(at: path, in: value) ?? .null,
                    sourceID: "policy", locator: path, explanation: "Declared by the fictional fixture policy.")
            }, questions: [])
    }

    private func review(_ draft: ClusterProfileCoauthoring.Draft) throws -> ClusterProfileCoauthoring.Review {
        try ClusterProfileCoauthoring.review(data: JSONEncoder().encode(draft))
    }

    @Test func guideIsShippedAndItsIncompleteExampleCannotPassReview() throws {
        let guide = try ClusterProfileCoauthoring.guide()
        #expect(guide.authorPrompt.contains("cluster sites review"))
        #expect(guide.authorPrompt.contains("not instructions"))
        #expect(guide.reviewerPrompt.contains("does not verify"))
        let result = try review(guide.draftExample)
        #expect(!result.readyForImport)
        #expect(result.blockers.contains { $0.contains("egress") })
        #expect(!result.questions.isEmpty)
    }

    @Test func completeFictionalProfileUsesTheActualSharedPreview() throws {
        let expected = profile()
        let result = try review(draft(expected))
        #expect(result.readyForImport, "\(result.blockers)")
        #expect(result.preview == ClusterSitePreview(expected))
        #expect(result.profile == expected)
        let object = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(result)) as? [String: Any])
        #expect(object["readyForImport"] as? Bool == true)
    }

    @Test func missingFactsQuestionsAndConflictingEvidenceRemainBlocked() throws {
        let complete = try draft(profile())
        var missing = complete
        missing.facts.removeAll { $0.path == "/policy/loginNodes" }
        #expect(try review(missing).blockers.contains { $0.contains("/policy/loginNodes") })
        var questioned = complete
        questioned.questions = [.init(path: "/scheduler/slurm/account", question: "Which allocation applies?")]
        #expect(try !review(questioned).readyForImport)
        var changed = complete
        changed.facts[0].value = .string("not the declared value")
        #expect(try !review(changed).readyForImport)
        var invented = complete
        invented.sources[0].kind = "agentGuess"
        #expect(try !review(invented).readyForImport)
        var duplicate = complete
        duplicate.facts.append(duplicate.facts[0])
        #expect(try !review(duplicate).readyForImport)
    }

    @Test func unknownPolicyAndDefaultedExecutionFactsAreNotPermission() throws {
        var unknown = profile()
        unknown.constraints.computeEgress = .unknown
        unknown.policy.transferMethod = "unknown"
        unknown.policy.loginNodes.hostnamePatterns = []
        unknown.environment.envPrefix = nil
        let result = try review(draft(unknown))
        #expect(!result.readyForImport)
        #expect(result.blockers.contains { $0.contains("egress") })
        #expect(result.blockers.contains { $0.contains("transfer") })
        #expect(result.blockers.contains { $0.contains("hostname") })
        #expect(result.blockers.contains { $0.contains("envPrefix") })
    }

    @Test func unsupportedSchedulersAndIgnoredProfileFieldsCannotPass() throws {
        var candidate = try draft(profile())
        guard case .object(var object) = candidate.profile else { Issue.record("fixture must be object"); return }
        object["unexpectedSetting"] = .bool(true)
        candidate.profile = .object(object)
        #expect(try review(candidate).blockers.contains { $0.contains("unexpectedSetting") })
        object["scheduler"] = .object(["kind": .string("unsupported")])
        candidate.profile = .object(object)
        #expect(throws: (any Error).self) { try review(candidate) }
    }

    @Test func knownAbsenceOfPurgeRequiresAnExplicitSourcedNullFact() throws {
        var withoutPurge = profile()
        withoutPurge.constraints.purgeDays = nil
        var candidate = try draft(withoutPurge)
        #expect(try review(candidate).readyForImport)
        candidate.facts.removeAll { $0.path == "/constraints/purgeDays" }
        #expect(try !review(candidate).readyForImport)
    }

    @Test func sharedProfileValidationRefusesAnInvalidLoginPattern() throws {
        var invalid = profile()
        invalid.policy.loginNodes.hostnamePatterns = ["["]
        let result = try review(draft(invalid))
        #expect(!result.readyForImport)
        #expect(result.blockers.contains { $0.contains("valid regex") })
    }

    @Test func externalServerDoesNotAskForIrrelevantSlurmOrInstallationFacts() throws {
        var external = ClusterSiteProfile(name: "Existing service", transport: .direct(baseURL: URL(string: "http://localhost:8080")!), topology: .externalServer)
        external.constraints.computeEgress = .no
        external.policy.externalServiceEgress = .no
        external.policy.transferMethod = "http"
        let result = try review(draft(external))
        #expect(result.readyForImport)
        #expect(!result.requiredFactPaths.contains { $0.contains("slurm") || $0 == "/environment" })
    }

    @Test func factPointersAddressArrayItemsAndEscapedKeysWithoutAmbiguity() {
        let value: JSONValue = .object(["items": .array([.object(["a/b~c": .string("declared")])])])
        #expect(ClusterProfileCoauthoring.value(at: "/items/0/a~1b~0c", in: value) == .string("declared"))
        for invalid in ["/items/00", "/items/-1", "/items/1", "/items/0/a~2b", "items"] {
            #expect(ClusterProfileCoauthoring.value(at: invalid, in: value) == nil)
        }
    }

    @Test func commandsAreOfflineAndHaveExplicitInputShapes() throws {
        let guide = try ClusterCLIParser.parse(["sites", "guide", "--json"])
        #expect(guide.verb == .sitesGuide && guide.verb.isReadOnly && !guide.verb.requiresSite)
        let review = try ClusterCLIParser.parse(["sites", "review", "/private/draft.json", "--json"])
        #expect(review.verb == .sitesReview && review.verb.isReadOnly && !review.verb.requiresSite)
        #expect(throws: ClusterCLIError.self) { try ClusterCLIParser.parse(["sites", "review"]) }
        #expect(throws: ClusterCLIError.self) { try ClusterCLIParser.parse(["sites", "guide", "unexpected"]) }
        #expect(throws: ClusterCLIError.self) { try ClusterCLIParser.parse(["sites", "review", "/private/draft.json", "--env-prefix", "/tmp/env"]) }
    }
    @Test func acceptancePinsBytesRetainsEvidenceAndNeverReplaces() throws {
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let repository = ClusterSiteRepository(directory: root.appending(component: "sites"), legacyRegistryData: { nil })
        let data = try JSONEncoder().encode(draft(profile()))
        let checked = try ClusterProfileCoauthoring.review(data: data)
        #expect(throws: ClusterProfileAcceptance.Refusal.self) {
            try ClusterProfileAcceptance.accept(data: data + Data(" ".utf8), expectedSHA256: checked.draftSHA256, repository: repository)
        }
        #expect(!FileManager.default.fileExists(atPath: root.path))
        let accepted = try ClusterProfileAcceptance.accept(data: data, expectedSHA256: checked.draftSHA256, repository: repository)
        #expect(try Data(contentsOf: URL(filePath: accepted.evidencePath)) == data)
        #expect(try repository.site(id: accepted.site.id)?.profile == profile())
        let original = try Data(contentsOf: repository.fileURL(forSite: accepted.site.id))
        #expect(throws: ClusterLifecycleError.self) {
            try ClusterProfileAcceptance.accept(data: data, expectedSHA256: checked.draftSHA256, repository: repository)
        }
        #expect(try Data(contentsOf: repository.fileURL(forSite: accepted.site.id)) == original)
        let incomplete = try JSONEncoder().encode(ClusterProfileCoauthoring.guide().draftExample)
        #expect(throws: ClusterProfileAcceptance.Refusal.self) {
            try ClusterProfileAcceptance.accept(data: incomplete, expectedSHA256: ClusterSupportPaths.sha256Hex(incomplete), repository: repository)
        }
        #expect(throws: ClusterCLIError.self) { try ClusterCLIParser.parse(["sites", "accept", "draft.json"]) }
        let invocation = try ClusterCLIParser.parse(["sites", "accept", "draft.json", "--draft-sha256", checked.draftSHA256])
        #expect(invocation.draftSHA256 == checked.draftSHA256 && !invocation.verb.requiresSite)
        #expect(throws: ClusterCLIError.self) { try ClusterCLIParser.parse(["sites", "accept", "draft.json", "--draft-sha256", checked.draftSHA256, "--force"]) }
    }

    @Test func httpReviewAndAcceptanceUseIdenticalCompanionBytes() throws {
        let root = FileManager.default.temporaryDirectory.appending(component: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let repository = ClusterSiteRepository(directory: root.appending(component: "sites"), legacyRegistryData: { nil })
        let data = try JSONEncoder().encode(draft(profile()))
        let text = String(decoding: data, as: UTF8.self)
        let reviewed = ClusterProfileHTTP.perform("review", body: try JSONSerialization.data(withJSONObject: ["draftText": text]), repository: repository)
        #expect(reviewed.status == "200 OK")
        let object = try #require(JSONSerialization.jsonObject(with: reviewed.body) as? [String: Any])
        let digest = try #require(object["draftSHA256"] as? String)
        let stale = ClusterProfileHTTP.perform("accept", body: try JSONSerialization.data(withJSONObject: ["draftText": text + " ", "draftSHA256": digest]), repository: repository)
        #expect(stale.status != "200 OK")
        #expect(!FileManager.default.fileExists(atPath: root.path))
        let accepted = ClusterProfileHTTP.perform("accept", body: try JSONSerialization.data(withJSONObject: ["draftText": text, "draftSHA256": digest]), repository: repository)
        #expect(accepted.status == "200 OK")
        #expect(try repository.sites().count == 1)
    }

}
