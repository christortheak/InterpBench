import Foundation
import Testing
@testable import ExperimentKit

/// Real Python operations consume the same workspace as the app's shared owners.
@MainActor @Suite(.serialized)
struct PythonStudyDesignParityTests {
    private var repository: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    private func python(_ script: String, root: URL, arguments: [String] = []) throws -> [String: Any] {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        let local = repository.appending(path: "Server/.venv.nosync/bin/python").path
        let executable = ProcessInfo.processInfo.environment["STEERLAB_TEST_PYTHON"]
            ?? (FileManager.default.isExecutableFile(atPath: local) ? local : "python3")
        process.arguments = [executable, "-c", """
        import json, sys
        from pathlib import Path
        from steerlab_server.client import study_designs, design_files, design_identity, design_panels, study_interviews
        from steerlab_server.experiment import experiment_store as store
        root = Path(sys.argv[1]).resolve()
        """ + "\n" + script, root.path] + arguments
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONPATH"] = repository.appending(component: "Server").path
        environment["HF_HUB_OFFLINE"] = "1"
        process.environment = environment
        let output = Pipe(), errors = Pipe()
        process.standardOutput = output; process.standardError = errors
        try process.run()
        let data = output.fileHandleForReading.readDataToEndOfFile()
        let diagnostics = errors.fileHandleForReading.readDataToEndOfFile()
        process.waitUntilExit()
        guard process.terminationStatus == 0 else {
            throw ExperimentError(reason: String(decoding: diagnostics, as: UTF8.self)
                + "\nSet TEST_RUNNER_STEERLAB_TEST_PYTHON=<client-python> in the shell environment before xcodebuild.")
        }
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    @Test func bothClientsAndTheAppShareExactlyTheSameInterviews() throws {
        let result = try python("print(json.dumps({i: study_interviews.prompt(i) for i in ('conceptStudy','agentComparison','multiAgent')}))", root: repository)
        for intent in StudyIntent.allCases {
            #expect(result[intent.rawValue] as? String == StudyCoauthoring.prompt(for: intent))
        }
    }

    @Test func pythonDesignToAppEditToMacUpdateToPythonSibling() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "python-design-parity") { root in
            let previous = WorkspaceRoot.programmaticOverride
            WorkspaceRoot.programmaticOverride = root
            defer { WorkspaceRoot.programmaticOverride = previous }
            let result = try python("""
            source = store.create('source', model_id='test/model', root=str(root))
            result = study_designs.create('source', root=root, expected=source.source_digest, name='shared-design')
            design = result['design']
            minted = study_designs.instantiate('shared-design', {'agents': []}, root=root,
                expected=design['designFileSHA256'], study_name='python-study')
            print(json.dumps({'design': design, 'study': minted}))
            """, root: root)
            let design = try #require(result["design"] as? [String: Any])
            let reviewed = try StudyDesignSnapshot(workspaceRoot: root, name: "shared-design")
            #expect(design["portableContentHash"] as? String == (try PortableDesignIdentity.hash(reviewed.template)))
            let study = try DraftAuthoringSnapshot(workspaceRoot: root, name: "python-study")
            #expect(study.manifest.status == .draft)
            #expect(StudyTemplateStore.lineage(of: study.manifest).agreement == .matches)
            #expect(!StudyTemplateStore.lineage(of: study.manifest).designRevised)
            let unchangedBytes = try Data(contentsOf: root.appending(path: "experiments/python-study/experiment.json"))
            let reused = try StudyDesignSaving.create(from: StudyDesignSourceReview(study: study))
            #expect(!reused.created)
            // Same scratch-draft operation the app uses to edit a design.
            let edit = try StudyDesignInstantiation.instantiate(reviewed: reviewed, casting: .agents([]), studyName: "app-edit")
            var edited = edit.manifest
            edited.maxTokens = 239
            try ExperimentStore.save(edited, workspaceRoot: root, expectedFile: .sha256(edit.file.sha256))
            let source = try StudyDesignSourceReview(study: DraftAuthoringSnapshot(workspaceRoot: root, name: "app-edit"))
            let updated = try StudyDesignSaving.update(from: source, reviewed: reviewed)
            #expect(updated.changed)
            #expect(try Data(contentsOf: root.appending(path: "experiments/python-study/experiment.json")) == unchangedBytes)
            #expect(StudyTemplateStore.lineage(of: study.manifest).agreement == .matches)
            #expect(StudyTemplateStore.lineage(of: study.manifest).designRevised)
            let returned = try python("""
            review = design_files.read('shared-design', root)
            result = study_designs.instantiate('shared-design', {'agents':[]}, root=root,
                expected=review['designFileSHA256'], study_name='returned-study')
            print(json.dumps(result))
            """, root: root)
            #expect((returned["document"] as? [String: Any])?["maxTokens"] as? Int == 239)
            let returnedStudy = try DraftAuthoringSnapshot(workspaceRoot: root, name: "returned-study")
            #expect(StudyTemplateStore.lineage(of: returnedStudy.manifest).agreement == .matches)
            let oldHash = StudyTemplateStore.hash(reviewed.template)
            #expect(edit.manifest.templateProvenance?.templateHash == oldHash)
            #expect(edit.manifest.templateProvenance?.hashAlgorithm == nil)
        }
    }

    @Test func portableIdentityMatchesAcrossNumericalAndScientificFields() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "portable-design-hash") { root in
            _ = try ExperimentStore.create(name: "source", description: "unicode 雪 / a", modelID: "test/model")
            var study = try ExperimentStore.load(name: "source")
            study.seeds = [UInt64.max, 9_007_199_254_740_993]
            study.temperature = 0.000000000000001
            study.concepts = [.init(name: "synthetic", stimulusSetHash: String(repeating: "a", count: 64), options: .init())]
            study.concepts[0].validationHashPinnedAbsent = true
            study.pipeline = .object(["null-is-data": .null, "values": .array([.number(1.5), .bool(false)])])
            study.judges = [.init(name: "judge", kind: "openrouter", model: "test/judge", provider: "test")]
            let template = StudyTemplate(name: "identity", study: StudyTemplateStore.strippedBody(study))
            let directory = root.appending(path: "templates/identity")
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try JSONEncoder().encode(template).write(to: directory.appending(component: "template.json"))
            let result = try python("print(json.dumps(design_files.read('identity', root)))", root: root)
            #expect(result["portableContentHash"] as? String == (try PortableDesignIdentity.hash(template)))
            // Label and historical stamps are excluded, scientific settings are not.
            var renamed = template
            renamed.name = "other"; renamed.templateDescription = "new note"
            #expect(try PortableDesignIdentity.hash(renamed) == PortableDesignIdentity.hash(template))
            renamed.study.maxTokens += 1
            #expect(try PortableDesignIdentity.hash(renamed) != PortableDesignIdentity.hash(template))
        }
    }

    @Test func pythonPanelCastingCanBeOpenedAndReusedByAppOwners() throws {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "python-panel-design") { root in
            let previous = WorkspaceRoot.programmaticOverride
            WorkspaceRoot.programmaticOverride = root
            defer { WorkspaceRoot.programmaticOverride = previous }
            let result = try python("""
            source = store.create('source', model_id='test/model', root=str(root))
            source.update(studyType='multiAgent', studyKind='multiAgent')
            store.save_raw(source, str(root))
            saved = study_designs.create('source', root=root, expected=source.source_digest, name='panel-design')['design']
            panel = {'schemaVersion':1,'name':'panel','description':'shared roles','sharedMaterials':'shared facts',
                'baseModelID':'','temperature':0,'maxTokens':2048,
                'agents':[{'id':'first','name':'First','baseModelID':'','systemPrompt':'review carefully','role':'reviewer'}],
                'turns':[{'id':'one','title':'Review','speakerAgentID':'first','promptTemplate':'Read the materials.',
                    'outputLabel':'answer','routing':'all','routedAgentIDs':[], 'includeScenarioMaterials':True,'includeSpeakerContext':True}]}
            template = saved['document']
            template['semanticScenario'] = design_panels.pin(panel, root)
            design_files.path(root, 'panel-design').write_bytes(design_files.encode(template))
            review = design_files.read('panel-design', root)
            result = study_designs.instantiate('panel-design', {'seats':{'first':None}}, root=root,
                expected=review['designFileSHA256'], study_name='panel-study')
            print(json.dumps(result))
            """, root: root)
            #expect(result["name"] as? String == "panel-study")
            let read = try DraftAuthoringSnapshot(workspaceRoot: root, name: "panel-study")
            let state = try #require(SeatCasting.state(of: read.manifest, workspaceRoot: root))
            #expect(state.form == .cast)
            #expect(state.seatIDs == ["first"])
            #expect(state.semantic.agents.first?.role == "reviewer")
            #expect(StudyTemplateStore.lineage(of: read.manifest).agreement == .matches)
            let saved = try StudyDesignSaving.create(from: StudyDesignSourceReview(study: read))
            #expect(!saved.created)
            let cast = try StudyDesignInstantiation.instantiate(reviewed: saved.snapshot,
                casting: .seating(state.assignment), studyName: "mac-panel-study")
            let back = try python("""
            source = store.load_raw('mac-panel-study', str(root))
            print(json.dumps(study_designs.create('mac-panel-study', root=root, expected=source.source_digest)))
            """, root: root)
            #expect(back["created"] as? Bool == false)
            #expect(cast.manifest.multiAgentSemanticScenarioHash == read.manifest.multiAgentSemanticScenarioHash)
        }
    }
}
