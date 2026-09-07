import Foundation

enum ClusterProfileHTTP {
    @MainActor static func perform(_ operation: String, body: Data,
                                   repository: ClusterSiteRepository = .init()) -> StudyAuthoringHTTP.Response {
        do {
            if operation == "guide" { return .json(try ClusterProfileCoauthoring.guide()) }
            guard let object = try JSONSerialization.jsonObject(with: body) as? [String: Any],
                  Set(object.keys) == (operation == "accept" ? ["draftText", "draftSHA256"] : ["draftText"]),
                  let text = object["draftText"] as? String else {
                throw ExperimentError.malformed("Supply the exact companion bytes as draftText, plus draftSHA256 when accepting.", repair: "Read /api/cluster/sites/guide and review the completed companion before accepting.")
            }
            let data = Data(text.utf8)
            if operation == "review" { return .json(try ClusterProfileCoauthoring.review(data: data)) }
            guard let expected = object["draftSHA256"] as? String else {
                throw ExperimentError(reason: "draftSHA256 must be the reviewed byte digest.")
            }
            let accepted = try ClusterProfileAcceptance.accept(data: data, expectedSHA256: expected, repository: repository)
            return .json(["siteID": accepted.site.id, "evidencePath": accepted.evidencePath,
                          "nextAction": "cluster preview --site " + accepted.site.id])
        } catch let error as ClusterProfileAcceptance.Refusal {
            return .failure(error.code, error.reason, repair: error.repairAction)
        } catch { return StudyAuthoringHTTP.failure(error) }
    }
}
