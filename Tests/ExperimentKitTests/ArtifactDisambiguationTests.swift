import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

/// Telling near-identically-named artifacts apart by what actually differs.
///
/// A timestamp does not help when the work happened on the same day: it says
/// WHICH came later, never WHY there are two.
struct ArtifactDisambiguationTests {

    private func candidate(
        _ id: String, _ name: String, _ identity: [String: String],
        createdAt: String? = nil
    ) -> ArtifactDisambiguation.Candidate {
        .init(id: id, displayName: name, identity: identity, createdAt: createdAt)
    }

    @Test func aUniqueNameGetsNoNoise() {
        let labels = ArtifactDisambiguation.labels([
            candidate("a", "fear", ["method": "meanDifference"]),
            candidate("b", "anger", ["method": "lat"]),
        ])
        #expect(labels.allSatisfy { $0.distinguisher == nil })
        #expect(labels.first?.text == "fear")
    }

    @Test func collidingNamesAreSeparatedByTheFieldThatDiffers() throws {
        let labels = ArtifactDisambiguation.labels([
            candidate("a", "fear", ["method": "meanDifference", "model": "m"]),
            candidate("b", "fear", ["method": "lat", "model": "m"]),
        ])
        // Only the differing field appears — "model" is identical, so naming
        // it would be noise.
        #expect(labels[0].text == "fear — method: meanDifference")
        #expect(labels[1].text == "fear — method: lat")
    }

    @Test func theSmallestSeparatingSetIsUsed() {
        // Three artifacts separable by ONE field; the others must not be
        // dragged in just because they also vary.
        let labels = ArtifactDisambiguation.labels([
            candidate("a", "fear", ["layer": "1", "noise": "x"]),
            candidate("b", "fear", ["layer": "2", "noise": "y"]),
            candidate("c", "fear", ["layer": "3", "noise": "z"]),
        ])
        // Either field alone separates; whichever is chosen, only one appears.
        #expect(labels.allSatisfy { ($0.distinguisher ?? "").split(separator: ",").count == 1 })
    }

    @Test func twoFieldsAreUsedWhenOneWillNotDo() throws {
        let labels = ArtifactDisambiguation.labels([
            candidate("a", "fear", ["method": "caa", "reading": "last"]),
            candidate("b", "fear", ["method": "caa", "reading": "mean50"]),
            candidate("c", "fear", ["method": "lat", "reading": "last"]),
        ])
        let first = try #require(labels.first?.distinguisher)
        #expect(first.contains("method"))
        #expect(first.contains("reading"))
    }

    /// The scientifically important case: identical recipe means
    /// interchangeable, and saying so answers the researcher's real question
    /// ("does it matter which I pick?") better than any hash.
    @Test func identicalRecipesAreCalledInterchangeable() throws {
        let labels = ArtifactDisambiguation.labels([
            candidate("a", "fear", ["method": "caa"], createdAt: "09:15"),
            candidate("b", "fear", ["method": "caa"], createdAt: "16:40"),
        ])
        #expect(labels.allSatisfy { $0.isInterchangeable })
        #expect(labels[0].text.contains("09:15"))
        #expect(labels[1].text.contains("16:40"))
        #expect(labels[0].text.contains("identical to a sibling"))
        let help = try #require(labels.first?.help)
        #expect(help.contains("does not matter which you pick"))
    }

    @Test func labelsAreDeterministic() {
        // Ties broken by field name, so the same catalog labels the same way
        // on every launch and on either engine.
        let group = [
            candidate("a", "fear", ["alpha": "1", "beta": "1"]),
            candidate("b", "fear", ["alpha": "2", "beta": "2"]),
        ]
        let first = ArtifactDisambiguation.labels(group).map(\.text)
        let second = ArtifactDisambiguation.labels(group.reversed()).map(\.text)
        #expect(Set(first) == Set(second))
    }

    // MARK: duplicate families

    /// `x`, `x-2`, `x-2-2` are three DISTINCT names, so the collision rule
    /// never fires for them — yet they are exactly the set a researcher
    /// cannot tell apart.
    @Test func duplicateSuffixesAreStrippedToAStem() {
        #expect(ArtifactDisambiguation.duplicateStem("vignette-with-fear") == "vignette-with-fear")
        #expect(ArtifactDisambiguation.duplicateStem("vignette-with-fear-2") == "vignette-with-fear")
        #expect(ArtifactDisambiguation.duplicateStem("vignette-with-fear-2-2") == "vignette-with-fear")
        // A name that legitimately ends in a number-like word is untouched.
        #expect(ArtifactDisambiguation.duplicateStem("gpt-4o-study") == "gpt-4o-study")
    }

    @Test func aDuplicateFamilyIsLabelledByWhatChanged() throws {
        func manifest(_ name: String, instruments: [String]?) -> ExperimentManifest {
            var m = ExperimentManifest(
                name: name, description: "", modelID: "test/model")
            m.outcomeInstruments = instruments
            return m
        }
        let families = ArtifactDisambiguation.familyLabels([
            manifest("vignette", instruments: ["answerTokenLogprob"]),
            manifest("vignette-2", instruments: ["sampledText"]),
        ])
        let labels = try #require(families["vignette"])
        #expect(labels.count == 2)
        #expect(labels.contains { $0.distinguisher?.contains("answerTokenLogprob") == true })
        #expect(labels.contains { $0.distinguisher?.contains("sampledText") == true })
    }

    @Test func aFamilyWithNoRealDifferenceSaysSo() throws {
        func manifest(_ name: String) -> ExperimentManifest {
            ExperimentManifest(name: name, description: "", modelID: "test/model")
        }
        let families = ArtifactDisambiguation.familyLabels([
            manifest("dup"), manifest("dup-2"),
        ])
        let labels = try #require(families["dup"])
        // Duplicating without changing anything is a real and confusing
        // state; naming it beats inventing a difference.
        #expect(labels.allSatisfy { $0.isInterchangeable })
        #expect(labels.allSatisfy { $0.distinguisher?.contains("identical to a sibling") == true })
    }

    /// The real shape, from the researcher's workspace: the original has an
    /// agent attached and its two duplicates do not — so ONE member is
    /// separated by a field and the other two are identical to each other.
    /// Labelling the family all-or-nothing would leave those two with the
    /// same text, i.e. exactly where they started.
    @Test func aPartlySeparableFamilyLabelsEachMemberHonestly() throws {
        func manifest(_ name: String, agent: String?) -> ExperimentManifest {
            var m = ExperimentManifest(
                name: name, description: "", modelID: "test/model")
            if let agent {
                m.variantConditions = [
                    .init(
                        name: agent, artifactPath: "runs/model-variants/a.json",
                        artifactHash: "h",
                        artifact: .init(
                            name: agent, baseModelID: "test/model",
                            promptMode: "chatAssistant",
                            qwenThinkingEnabled: false, temperature: 0,
                            systemPrompt: ""))
                ]
            }
            return m
        }
        let families = ArtifactDisambiguation.familyLabels([
            manifest("vignette", agent: "fear-agent"),
            manifest("vignette-2", agent: nil),
            manifest("vignette-2-2", agent: nil),
        ])
        let labels = try #require(families["vignette"])
        let byID = Dictionary(uniqueKeysWithValues: labels.map { ($0.id, $0) })

        // The one that differs is named by the field, and is NOT called
        // interchangeable.
        let original = try #require(byID["vignette"])
        #expect(original.distinguisher?.contains("fear-agent") == true)
        #expect(!original.isInterchangeable)

        // The two that do not differ are told so, rather than being given
        // identical labels and left ambiguous.
        for name in ["vignette-2", "vignette-2-2"] {
            let label = try #require(byID[name])
            #expect(label.isInterchangeable)
            #expect(label.distinguisher?.contains("identical to a sibling") == true)
        }
    }

    @Test func unrelatedStudiesAreNotAFamily() {
        func manifest(_ name: String) -> ExperimentManifest {
            ExperimentManifest(name: name, description: "", modelID: "test/model")
        }
        let families = ArtifactDisambiguation.familyLabels([
            manifest("alpha"), manifest("beta"),
        ])
        #expect(families.isEmpty)
    }

    // MARK: vector candidates

    private func vectorArtifact(
        runDirectory: String, extractionDate: String
    ) throws -> VectorArtifact {
        // Decoded, not built with the memberwise init: what reaches
        // `vectorCandidates` in production is always a decoded sidecar, and
        // decoding is where "no date" becomes "" rather than nil.
        let json = """
            {"modelID": "test/model", "concept": "fear",
             "stimulusSetHash": "aaaa1111", "layerCount": 2, "hiddenSize": 2,
             "normsPerLayer": [1.0, 1.0],
             "extractionDate": "\(extractionDate)"}
            """
        let sidecar = try JSONDecoder().decode(
            SteeringVectorSidecar.self, from: Data(json.utf8))
        return VectorArtifact(
            directory: URL(filePath: runDirectory), name: "fear", sidecar: sidecar)
    }

    @Test func aStampedExtractionDateIsTheCreationTime() throws {
        let candidates = ArtifactDisambiguation.vectorCandidates([
            try vectorArtifact(
                runDirectory: "/ws/runs/20260610T093000000-extract-fear",
                extractionDate: "2026-06-10T09:30:00Z")
        ])
        #expect(candidates.first?.createdAt == "2026-06-10T09:30:00Z")
    }

    /// A sidecar cannot LACK the date field (decoding would refuse it long
    /// before the picker), but it can carry the legacy empty string — and
    /// then the run-directory name, which embeds the creation timestamp, is
    /// the ordering fallback rather than a blank "made " label.
    @Test func anEmptyExtractionDateFallsBackToTheRunDirectoryName() throws {
        let candidates = ArtifactDisambiguation.vectorCandidates([
            try vectorArtifact(
                runDirectory: "/ws/runs/20260610T093000000-extract-fear",
                extractionDate: "")
        ])
        #expect(candidates.first?.createdAt == "20260610T093000000-extract-fear")
    }
}

/// The picker wiring — the rule reaching the surfaces that show bare names.
struct ArtifactDisambiguationWiringTests {

    private func agent(
        _ name: String, concept: String, layer: Int, alpha: Double,
        promotedBy: String? = "criterion", sweepRun: String? = "run-1"
    ) -> ModelVariantArtifact {
        ModelVariantArtifact(
            name: name, baseModelID: "test/model", baseRevision: "abc123def",
            injections: [
                .init(concept: concept, vectorArtifactID: "runs/x/\(concept)",
                      layer: layer, alpha: alpha)
            ],
            promptMode: "chatAssistant", qwenThinkingEnabled: false,
            temperature: 0, systemPrompt: "",
            promotion: promotedBy.map {
                .init(
                    experiment: "e", experimentHash: "h", promotedAt: "t",
                    promotedBy: $0, sweepRun: sweepRun,
                    substrate: "swift-mlx", appVersion: "v")
            })
    }

    @Test func theCellIsWhatDistinguishesTwoPromotionsOfOneConcept() throws {
        // The routine case: re-sweep, promote again, two agents with one name.
        let candidates = [
            ArtifactDisambiguation.Candidate(
                id: "a", displayName: "pw-agent",
                identity: ArtifactDisambiguation.agentIdentity(
                    agent("pw-agent", concept: "practicalwisdom", layer: 31, alpha: 0.05))),
            ArtifactDisambiguation.Candidate(
                id: "b", displayName: "pw-agent",
                identity: ArtifactDisambiguation.agentIdentity(
                    agent("pw-agent", concept: "practicalwisdom", layer: 41, alpha: 0.17))),
        ]
        let labels = ArtifactDisambiguation.labels(candidates)
        #expect(labels[0].text.contains("L31"))
        #expect(labels[1].text.contains("L41"))
        // The concept is identical, so naming it would be noise.
        #expect(labels[0].distinguisher?.contains("concepts") != true)
    }

    /// How an agent was chosen is a first-class difference: a
    /// criterion-selected agent and a hand-overridden one at the SAME cell
    /// are not the same evidence, and a picker that hides that invites
    /// citing the wrong one.
    @Test func provenanceSeparatesAgentsAtTheSameCell() throws {
        let candidates = [
            ArtifactDisambiguation.Candidate(
                id: "a", displayName: "pw-agent",
                identity: ArtifactDisambiguation.agentIdentity(
                    agent("pw-agent", concept: "pw", layer: 41, alpha: 0.1))),
            ArtifactDisambiguation.Candidate(
                id: "b", displayName: "pw-agent",
                identity: ArtifactDisambiguation.agentIdentity(
                    agent("pw-agent", concept: "pw", layer: 41, alpha: 0.1,
                          promotedBy: "manualOverride"))),
        ]
        let labels = ArtifactDisambiguation.labels(candidates)
        #expect(labels.contains { $0.distinguisher?.contains("criterion") == true })
        #expect(labels.contains { $0.distinguisher?.contains("manualOverride") == true })
        #expect(labels.allSatisfy { !$0.isInterchangeable })
    }

    @Test func aHandCreatedAgentIsMarkedAsSuch() {
        let identity = ArtifactDisambiguation.agentIdentity(
            agent("x", concept: "c", layer: 1, alpha: 0.1, promotedBy: nil,
                  sweepRun: nil))
        // Absent provenance is a fact, not a blank — an agent with no birth
        // certificate is distinguishable from one that has it.
        #expect(identity["promotedBy"] == "handCreated")
    }

    @Test func theAgentPickerLabelsSameNamedDefinitions() {
        let rows = ChatService.agentPickerRows(
            workspaceIsServer: false,
            serverAgents: [],
            installedServerModels: [],
            localDefinitions: [
                .init(
                    id: "a", name: "pw-agent", baseModelID: "test/model",
                    identity: ["cell": "L31α0.05"]),
                .init(
                    id: "b", name: "pw-agent", baseModelID: "test/model",
                    identity: ["cell": "L41α0.17"]),
                .init(
                    id: "c", name: "fear-agent", baseModelID: "test/model",
                    identity: ["cell": "L20α0.5"]),
            ],
            localLoadedModelID: "test/model")
        #expect(rows.first { $0.id == .localDefinition(id: "a") }?.title
            .contains("L31α0.05") == true)
        #expect(rows.first { $0.id == .localDefinition(id: "b") }?.title
            .contains("L41α0.17") == true)
        // A unique name is left alone — no distinguisher noise.
        #expect(rows.first { $0.id == .localDefinition(id: "c") }?.title == "fear-agent")
    }
}
