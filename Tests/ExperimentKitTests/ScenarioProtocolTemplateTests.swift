import Foundation
import Testing

@testable import ExperimentKit

/// Scenario PROTOCOL templates: reuse the deliberation protocol across cases,
/// and stop losing panel files to a silent decode failure.
///
/// The properties worth testing are the ones that fail QUIETLY. A template
/// that parses as a panel produces a run whose every seat deliberates about an
/// empty record and whose transcript looks exactly like a real one. A checklist
/// that hardens into a block stops a legitimate case whose record has no
/// procedural posture. And a picker that drops what it cannot decode produces
/// the bug this feature was reported alongside: the researcher's panel is on
/// disk, absent from the menu, and nothing anywhere names the failing key.
///
/// Serialized because these move the process-global workspace root.
@Suite(.serialized) struct ScenarioProtocolTemplateTests {

    // MARK: - Fixtures

    private func withTempWorkspace<T>(_ body: (URL) throws -> T) rethrows -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "protocol-template-\(UUID().uuidString)")
        let previous = WorkspaceRoot.programmaticOverride
        WorkspaceRoot.programmaticOverride = temp
        defer {
            WorkspaceRoot.programmaticOverride = previous
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try body(temp)
    }

    /// A protocol with everything a protocol carries: seats, a scripted turn
    /// order, routing, per-turn caps, and a declared endpoint.
    private func panel(named name: String = "appellate-panel") -> MultiAgentScenario {
        PanelComposition.semanticForm(
            MultiAgentScenario(
                name: name,
                description: "Three-judge deliberation.",
                baseModelID: "",
                sharedMaterials: "The case record for THIS case.",
                agents: [
                    .init(
                        id: "judge-1", name: "Judge Whitfield", baseModelID: "",
                        systemPrompt: "You are Judge Whitfield."),
                    .init(
                        id: "judge-2", name: "Judge Marsden", baseModelID: "",
                        systemPrompt: "You are Judge Marsden."),
                ],
                turns: [
                    .init(
                        id: "turn-r1", title: "Round 1 — memo",
                        speakerAgentID: "judge-1",
                        promptTemplate: "Read the record.\n{{scenario.materials}}",
                        outputLabel: "r1_judge-1",
                        routing: .speakerOnly,
                        maxTokens: 600),
                    .init(
                        id: "turn-r3", title: "Round 3 — disposition",
                        speakerAgentID: "judge-2",
                        promptTemplate: "State your disposition.",
                        outputLabel: "r3_judge-2",
                        routing: .all,
                        maxTokens: 250,
                        endpoint: TurnEndpoint(
                            name: "r3ScalePosition", kind: .number,
                            marker: "Scale position:", min: 1, max: 7)),
                ]))
    }

    private let checklist = [
        "the case record",
        "procedural posture",
        "the disposition scale with anchors",
    ]

    // MARK: - Round trip

    @Test func aProtocolIsTheScenarioMinusItsCase() throws {
        let scenario = panel()
        let template = ScenarioProtocolTemplateStore.templateFromScenario(
            scenario, named: "deliberative-appellate-panel-v1",
            materialsChecklist: checklist)

        // The protocol half survives VERBATIM — turn ids, titles, routing,
        // per-turn caps and the endpoint declaration included. Anything less
        // and "reuse the protocol" is really "retype the protocol".
        #expect(template.protocolBody.agents == scenario.agents)
        #expect(template.protocolBody.turns == scenario.turns)
        #expect(template.protocolBody.turns[1].endpoint?.name == "r3ScalePosition")
        #expect(template.protocolBody.turns[0].maxTokens == 600)
        // The case does not.
        #expect(template.protocolBody.sharedMaterials.isEmpty)
        #expect(template.name == "deliberative-appellate-panel-v1")
        #expect(template.materialsChecklist == checklist)
    }

    @Test func instantiatingAProtocolReproducesTheOriginalAroundANewCase() throws {
        try withTempWorkspace { _ in
            let original = panel()
            let template = ScenarioProtocolTemplateStore.templateFromScenario(
                original, named: "deliberative-appellate-panel-v1",
                materialsChecklist: checklist)
            try ScenarioProtocolTemplateStore.save(template)

            let minted = try ScenarioProtocolTemplateStore.instantiate(
                template, name: "study3-hard-case",
                sharedMaterials: "A DIFFERENT case record.",
                confirmedChecklistItems: Set(checklist))

            var expected = original
            expected.name = "study3-hard-case"
            expected.description = template.templateDescription
            expected.sharedMaterials = "A DIFFERENT case record."
            #expect(minted.record.scenario == expected)

            // An ORDINARY panel file, in the ordinary place, readable by the
            // ordinary decoder. No template-aware anything downstream.
            #expect(
                minted.record.url.deletingLastPathComponent().lastPathComponent
                    == "panels")
            let reread = try JSONDecoder().decode(
                MultiAgentScenario.self,
                from: try Data(contentsOf: minted.record.url))
            #expect(reread == expected)
        }
    }

    @Test func aTemplateFileRoundTripsThroughItsOwnCodable() throws {
        let template = ScenarioProtocolTemplateStore.templateFromScenario(
            panel(), named: "protocol-v1", materialsChecklist: checklist)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(template)
        // Flat: the marker keys and the panel body share one JSON object.
        let object = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(object["kind"] as? String == "scenarioProtocolTemplate")
        #expect(object["templateSchemaVersion"] as? Int == 1)
        #expect(object["turns"] != nil)
        #expect(object["agents"] != nil)
        #expect((object["sharedMaterials"] as? String)?.isEmpty == true)
        #expect(
            try JSONDecoder().decode(ScenarioProtocolTemplate.self, from: data)
                == template)
    }

    // MARK: - A protocol is not runnable

    @Test func theScenarioDecoderRefusesAProtocolTemplate() throws {
        let data = try JSONEncoder().encode(
            ScenarioProtocolTemplateStore.templateFromScenario(
                panel(), named: "protocol-v1"))
        #expect(throws: (any Error).self) {
            try JSONDecoder().decode(MultiAgentScenario.self, from: data)
        }
        // And the refusal says WHAT it is, not just that it failed — this
        // string is the whole difference between "fix your file" and "why is
        // my panel gone".
        do {
            _ = try JSONDecoder().decode(MultiAgentScenario.self, from: data)
            Issue.record("a protocol template decoded as a runnable panel")
        } catch {
            let reason = MultiAgentScenarioStore.decodeFailureReason(error)
            #expect(reason.contains("PROTOCOL TEMPLATE"))
        }
    }

    @Test func aPanelFileIsNotAProtocolTemplate() throws {
        let data = try JSONEncoder().encode(panel())
        #expect(!ScenarioProtocolTemplate.isTemplateFile(data))
        #expect(throws: (any Error).self) {
            try JSONDecoder().decode(ScenarioProtocolTemplate.self, from: data)
        }
    }

    // MARK: - The checklist is soft

    @Test func anUnconfirmedChecklistWarnsAndStillWrites() throws {
        try withTempWorkspace { _ in
            let template = ScenarioProtocolTemplateStore.templateFromScenario(
                panel(), named: "protocol-v1", materialsChecklist: checklist)
            let minted = try ScenarioProtocolTemplateStore.instantiate(
                template, name: "case-one",
                sharedMaterials: "Some record.",
                confirmedChecklistItems: ["the case record"])

            #expect(minted.review.unconfirmed == [
                "procedural posture", "the disposition scale with anchors",
            ])
            #expect(minted.warnings.count == 1)
            #expect(minted.warnings[0].contains("procedural posture"))
            // NEVER a block: the file is on disk regardless.
            #expect(
                FileManager.default.fileExists(atPath: minted.record.url.path))
            #expect(minted.record.scenario.sharedMaterials == "Some record.")
        }
    }

    @Test func aFullyConfirmedChecklistWithMaterialsIsClean() throws {
        try withTempWorkspace { _ in
            let template = ScenarioProtocolTemplateStore.templateFromScenario(
                panel(), named: "protocol-v1", materialsChecklist: checklist)
            let minted = try ScenarioProtocolTemplateStore.instantiate(
                template, name: "case-two",
                sharedMaterials: "Some record.",
                confirmedChecklistItems: Set(checklist))
            #expect(minted.review.isClean)
        }
    }

    @Test func emptyMaterialsWarnMechanically() {
        // The one thing that is a FACT about the file rather than a claim
        // about prose — and the exact failure the blank-materials habit
        // produced.
        let review = ScenarioProtocolTemplateStore.review(
            checklist: [], sharedMaterials: "   \n ", confirmed: [])
        #expect(review.unconfirmed.isEmpty)
        #expect(review.warnings.count == 1)
        #expect(review.warnings[0].contains("empty"))
    }

    // MARK: - Library separation

    @Test func templatesListInTheirOwnPickerAndNotTheScenarioPicker() throws {
        try withTempWorkspace { _ in
            try MultiAgentScenarioStore.save(panel(named: "a-real-case"))
            try ScenarioProtocolTemplateStore.save(
                ScenarioProtocolTemplateStore.templateFromScenario(
                    panel(), named: "protocol-v1", materialsChecklist: checklist))

            let panels = MultiAgentScenarioStore.scanAll()
            #expect(panels.records.map(\.scenario.name) == ["a-real-case"])
            #expect(panels.unreadable.isEmpty)

            let templates = ScenarioProtocolTemplateStore.scanAll()
            #expect(templates.records.map(\.template.name) == ["protocol-v1"])
            #expect(templates.unreadable.isEmpty)
            #expect(
                templates.records[0].url.deletingLastPathComponent()
                    .lastPathComponent == "templates")
        }
    }

    @Test func aTemplateSittingAmongThePanelsIsSkippedNotBroken() throws {
        try withTempWorkspace { _ in
            // Dropped in by hand, or written by an older build. It must not be
            // a runnable panel, must not be reported as a broken one, and must
            // still be FINDABLE — "it moved" beats "it vanished".
            let template = ScenarioProtocolTemplateStore.templateFromScenario(
                panel(), named: "stray-protocol", materialsChecklist: checklist)
            let strayed = MultiAgentScenarioStore.directory
                .appending(component: "stray-protocol.json")
            try FileManager.default.createDirectory(
                at: MultiAgentScenarioStore.directory,
                withIntermediateDirectories: true)
            try JSONEncoder().encode(template).write(to: strayed)

            let panels = MultiAgentScenarioStore.scanAll()
            #expect(panels.records.isEmpty)
            #expect(panels.unreadable.isEmpty)

            let templates = ScenarioProtocolTemplateStore.scanAll()
            #expect(templates.records.map(\.template.name) == ["stray-protocol"])
        }
    }

    // MARK: - The loud picker

    @Test func anUndecodablePanelSurfacesWithItsFailingKey() throws {
        try withTempWorkspace { _ in
            try MultiAgentScenarioStore.save(panel(named: "good-panel"))
            // The reported case: a semantic scenario omitting the
            // non-optional `Agent.baseModelID`. Before this it decoded to nil,
            // was `compactMap`ed away, and simply was not in the picker.
            let broken = """
                {
                  "name": "half-authored-panel",
                  "agents": [{ "id": "judge-1", "name": "Judge A" }],
                  "turns": []
                }
                """
            try FileManager.default.createDirectory(
                at: MultiAgentScenarioStore.directory,
                withIntermediateDirectories: true)
            try Data(broken.utf8).write(
                to: MultiAgentScenarioStore.directory
                    .appending(component: "half-authored-panel.json"))

            let scan = MultiAgentScenarioStore.scanAll()
            #expect(scan.records.map(\.scenario.name) == ["good-panel"])
            let issue = try #require(scan.unreadable.first)
            #expect(scan.unreadable.count == 1)
            #expect(issue.fileName == "half-authored-panel.json")
            #expect(issue.reason.contains("baseModelID"))
            #expect(issue.reason.contains("agents[0]"))
            #expect(issue.detail.contains("half-authored-panel.json"))
            // The plain `scan` stays records-only, so every existing caller
            // keeps its contract.
            #expect(MultiAgentScenarioStore.scan().count == 1)
        }
    }

    @Test func aMalformedEndpointDeclarationSurfacesToo() throws {
        try withTempWorkspace { _ in
            let broken = """
                {
                  "name": "bad-endpoint-panel",
                  "agents": [],
                  "turns": [{
                    "id": "t1", "title": "T", "speakerAgentID": "s",
                    "promptTemplate": "", "outputLabel": "",
                    "routing": "all", "routedAgentIDs": [],
                    "includeScenarioMaterials": true,
                    "includeSpeakerContext": true,
                    "endpoint": { "name": "x", "kind": "notAKind" }
                  }]
                }
                """
            try FileManager.default.createDirectory(
                at: MultiAgentScenarioStore.directory,
                withIntermediateDirectories: true)
            try Data(broken.utf8).write(
                to: MultiAgentScenarioStore.directory
                    .appending(component: "bad-endpoint-panel.json"))
            let scan = MultiAgentScenarioStore.scanAll()
            #expect(scan.records.isEmpty)
            #expect(scan.unreadable.count == 1)
            #expect(!(scan.unreadable.first?.reason.isEmpty ?? true))
        }
    }

    @Test func anUnmarkedFileInTheTemplateLibraryIsReportedNotDropped() throws {
        try withTempWorkspace { _ in
            try FileManager.default.createDirectory(
                at: ScenarioProtocolTemplateStore.directory,
                withIntermediateDirectories: true)
            // A plain panel filed as a protocol: it has no marker, so it is
            // not a template — and the researcher who put it there needs to
            // hear that, not to watch it disappear.
            try JSONEncoder().encode(panel()).write(
                to: ScenarioProtocolTemplateStore.directory
                    .appending(component: "not-a-protocol.json"))
            let templates = ScenarioProtocolTemplateStore.scanAll()
            #expect(templates.records.isEmpty)
            #expect(templates.unreadable.count == 1)
            #expect(
                templates.unreadable[0].reason.contains("kind")
                    || templates.unreadable[0].reason.contains("marker"))
        }
    }

    // MARK: - The shipped seed protocol

    private static var repoRoot: URL {
        URL(filePath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
    }

    @Test(
        .enabled(
            if: ResearchTreeFixtures.hasAppellateProtocolTemplate,
            """
            prompts/panels/templates/ is study material and does not ship — \
            the protocol-template machinery above is covered against in-code \
            fixtures, which is all a released tree has to answer for
            """))
    func theShippedAppellateProtocolDecodesAndCarriesItsWholeScript() throws {
        let url = Self.repoRoot.appending(
            components: "prompts", "panels", "templates",
            "deliberative-appellate-panel-v1.json")
        let data = try Data(contentsOf: url)
        let template = try JSONDecoder().decode(
            ScenarioProtocolTemplate.self, from: data)

        #expect(template.name == "deliberative-appellate-panel-v1")
        #expect(template.protocolBody.agents.count == 3)
        #expect(
            template.protocolBody.agents.map(\.name) == [
                "Judge Whitfield", "Judge Marsden", "Judge Calloway",
            ])
        #expect(template.protocolBody.turns.count == 15)
        #expect(template.protocolBody.turns.compactMap(\.endpoint).count == 6)
        #expect(template.protocolBody.sharedMaterials.isEmpty)
        #expect(template.materialsChecklist.count == 4)
        // No seat names a model and no seat is cast: a protocol is an
        // environment, and a study binds it.
        #expect(template.protocolBody.agents.allSatisfy { $0.baseModelID.isEmpty })
        #expect(
            template.protocolBody.agents.allSatisfy { $0.variantArtifactPath == nil })
        // The case it was minted from must not have come along.
        let body = String(data: data, encoding: .utf8) ?? ""
        #expect(!body.contains("Jessica"))
        #expect(!body.contains("Wendy"))

        // It is a protocol, not a panel: the panel decoder refuses it.
        #expect(throws: (any Error).self) {
            try JSONDecoder().decode(MultiAgentScenario.self, from: data)
        }
    }
}
