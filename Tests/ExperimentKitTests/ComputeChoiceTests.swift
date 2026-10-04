import Foundation
import Testing

@testable import ExperimentKit

/// The three plainly named compute choices (release review A3, decision 2).
///
/// A workspace used to be "Cluster (Python/PyTorch)" or "Local (MLX)", the
/// new-workspace panel defaulted to the first and captioned the second "toy
/// models and pipeline checks", and nothing said that the Python engine —
/// every method — runs on a Mac's own GPU. The choices now have three names,
/// and these tests hold the two things that must not move while the names
/// do: the persisted binding and its file format.
struct ComputeChoiceTests {

    private func withRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "cc-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: root.appending(component: "runs"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        return try body(root)
    }

    private func configURL(_ root: URL) -> URL {
        root.appending(components: ".steerlab", "workspace.json")
    }

    /// What a build from before this change wrote: one key, encoded exactly
    /// the way `WorkspaceCompute.declare` always encoded it.
    private struct LegacyConfig: Codable {
        var computeSubstrate: WorkspaceCompute
    }

    private func legacyBytes(_ compute: WorkspaceCompute) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(LegacyConfig(computeSubstrate: compute))
    }

    /// How a build from before this change READS the file.
    private func legacyRead(_ root: URL) throws -> WorkspaceCompute {
        try JSONDecoder().decode(
            LegacyConfig.self, from: Data(contentsOf: configURL(root))
        ).computeSubstrate
    }

    private func writeRun(_ root: URL, name: String, substrate: String) {
        let directory = root.appending(components: "runs", name)
        try? FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try? JSONSerialization.data(
            withJSONObject: ["runType": "sweep", "substrate": substrate]
        ).write(to: directory.appending(component: "config.json"))
    }

    // MARK: - The mapping, both ways

    @Test func theThreeChoicesMapOntoTheTwoPersistedBindings() {
        #expect(ComputeChoice.allCases.count == 3)
        #expect(ComputeChoice.macQuickStart.binding == .localMLX)
        #expect(ComputeChoice.macFullCapabilities.binding == .cluster)
        #expect(ComputeChoice.anotherMachine.binding == .cluster)
        // The two that share `cluster` are told apart by the one extra key.
        #expect(ComputeChoice.macQuickStart.location == nil)
        #expect(ComputeChoice.macFullCapabilities.location == .thisMac)
        #expect(ComputeChoice.anotherMachine.location == .anotherMachine)
        // The persisted binding's own raw values did not move.
        #expect(WorkspaceCompute.cluster.rawValue == "cluster")
        #expect(WorkspaceCompute.localMLX.rawValue == "local-mlx")
        #expect(WorkspaceCompute.Location.thisMac.rawValue == "this-mac")
        #expect(WorkspaceCompute.Location.anotherMachine.rawValue == "another-machine")
    }

    @Test func everyChoiceComesBackFromWhatItWrites() {
        for choice in ComputeChoice.allCases {
            #expect(
                ComputeChoice(binding: choice.binding, location: choice.location)
                    == choice)
            // A recorded location is never overridden by what happens to be
            // connected at the moment.
            for flag in [false, true] {
                if choice.location != nil {
                    #expect(
                        ComputeChoice(
                            binding: choice.binding, location: choice.location,
                            activeEngineIsThisMac: flag) == choice)
                }
            }
        }
        // A `cluster` binding with no recorded location — every workspace
        // declared before the key existed — reads as another machine, unless
        // the engine in use right now is this Mac's own.
        #expect(ComputeChoice(binding: .cluster, location: nil) == .anotherMachine)
        #expect(
            ComputeChoice(binding: .cluster, location: nil, activeEngineIsThisMac: true)
                == .macFullCapabilities)
        // The local binding is the quick start whatever else is true.
        #expect(
            ComputeChoice(binding: .localMLX, location: .thisMac, activeEngineIsThisMac: true)
                == .macQuickStart)
    }

    @Test func aNewWorkspaceStartsOnTheQuickStart() {
        #expect(ComputeChoice.newWorkspaceDefault == .macQuickStart)
        #expect(ComputeChoice.newWorkspaceDefault.binding == .localMLX)
    }

    // MARK: - The file format

    /// The quick start, and a `cluster` declaration made through the older
    /// entry point, write the SAME BYTES this file has always held.
    @Test func theExistingFormIsByteIdentical() throws {
        try withRoot { root in
            try WorkspaceCompute.declare(ComputeChoice.macQuickStart, root: root)
            #expect(try Data(contentsOf: configURL(root)) == legacyBytes(.localMLX))

            try WorkspaceCompute.declare(WorkspaceCompute.cluster, root: root)
            #expect(try Data(contentsOf: configURL(root)) == legacyBytes(.cluster))

            try WorkspaceCompute.declare(WorkspaceCompute.localMLX, root: root)
            #expect(try Data(contentsOf: configURL(root)) == legacyBytes(.localMLX))
        }
    }

    /// The addition is ONE optional key beside the binding, and a build from
    /// before it reads the file exactly as it did.
    @Test func theTwoPythonChoicesAddOneKeyThatOlderBuildsIgnore() throws {
        try withRoot { root in
            for (choice, location) in [
                (ComputeChoice.macFullCapabilities, "this-mac"),
                (ComputeChoice.anotherMachine, "another-machine"),
            ] {
                try WorkspaceCompute.declare(choice, root: root)
                let object = try #require(
                    JSONSerialization.jsonObject(
                        with: Data(contentsOf: configURL(root))) as? [String: String])
                #expect(
                    object == ["computeSubstrate": "cluster", "computeLocation": location])
                // This build reads both facts back…
                #expect(WorkspaceCompute.declared(root: root) == .cluster)
                #expect(WorkspaceCompute.declaredLocation(root: root) == choice.location)
                #expect(WorkspaceCompute.declaredChoice(root: root) == choice)
                #expect(WorkspaceCompute.resolved(root: root) == .cluster)
                // …and an older build reads the binding, untroubled by a key
                // it has never heard of.
                #expect(try legacyRead(root) == .cluster)
            }
        }
    }

    /// Every workspace declared before the key existed still loads.
    @Test func anOldFormFileLoadsAndNamesAChoice() throws {
        try withRoot { root in
            try FileManager.default.createDirectory(
                at: configURL(root).deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try legacyBytes(.cluster).write(to: configURL(root))
            #expect(WorkspaceCompute.declared(root: root) == .cluster)
            #expect(WorkspaceCompute.declaredLocation(root: root) == nil)
            #expect(WorkspaceCompute.declaredChoice(root: root) == .anotherMachine)
            #expect(
                WorkspaceCompute.declaredChoice(root: root, activeEngineIsThisMac: true)
                    == .macFullCapabilities)

            try legacyBytes(.localMLX).write(to: configURL(root))
            #expect(WorkspaceCompute.declaredChoice(root: root) == .macQuickStart)
            #expect(
                WorkspaceCompute.declaredChoice(root: root, activeEngineIsThisMac: true)
                    == .macQuickStart)
        }
    }

    @Test func aLocationThisBuildDoesNotKnowIsIgnoredNotAnError() throws {
        try withRoot { root in
            try FileManager.default.createDirectory(
                at: configURL(root).deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try Data(
                #"{"computeSubstrate": "cluster", "computeLocation": "a-future-place"}"#.utf8
            ).write(to: configURL(root))
            #expect(WorkspaceCompute.declared(root: root) == .cluster)
            #expect(WorkspaceCompute.declaredLocation(root: root) == nil)
            #expect(WorkspaceCompute.declaredChoice(root: root) == .anotherMachine)
        }
    }

    /// A write keeps keys it does not own, keeps the location while the
    /// binding stays `cluster`, and drops it when the binding becomes local —
    /// where it would mean nothing.
    @Test func aWriteKeepsWhatItDoesNotOwnAndDropsAStaleLocation() throws {
        try withRoot { root in
            try FileManager.default.createDirectory(
                at: configURL(root).deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try Data(
                #"{"computeSubstrate": "local-mlx", "somethingLater": "kept"}"#.utf8
            ).write(to: configURL(root))

            try WorkspaceCompute.declare(ComputeChoice.macFullCapabilities, root: root)
            func object() throws -> [String: String] {
                try #require(
                    JSONSerialization.jsonObject(
                        with: Data(contentsOf: configURL(root))) as? [String: String])
            }
            #expect(
                try object() == [
                    "computeSubstrate": "cluster", "computeLocation": "this-mac",
                    "somethingLater": "kept",
                ])

            // Re-declaring the binding alone (the older entry point) keeps
            // which Python choice it was.
            try WorkspaceCompute.declare(WorkspaceCompute.cluster, root: root)
            #expect(WorkspaceCompute.declaredChoice(root: root) == .macFullCapabilities)

            // Becoming local removes the location.
            try WorkspaceCompute.declare(WorkspaceCompute.localMLX, root: root)
            #expect(
                try object() == ["computeSubstrate": "local-mlx", "somethingLater": "kept"])
            #expect(WorkspaceCompute.declaredLocation(root: root) == nil)
            #expect(WorkspaceCompute.declaredChoice(root: root) == .macQuickStart)
        }
    }

    @Test func anUndeclaredWorkspaceNamesNoChoiceButResolvesOne() throws {
        try withRoot { root in
            #expect(WorkspaceCompute.declaredChoice(root: root) == nil)
            // Nothing declared, no runs: the quick start, and nothing written.
            #expect(WorkspaceCompute.resolvedChoice(root: root) == .macQuickStart)
            // Its own runs say the Python engine: a Python choice, which one
            // depending on what the app can see.
            for index in 0..<3 {
                writeRun(root, name: "s\(index)", substrate: "python-hf-transformers")
            }
            #expect(WorkspaceCompute.resolvedChoice(root: root) == .anotherMachine)
            #expect(
                WorkspaceCompute.resolvedChoice(root: root, activeEngineIsThisMac: true)
                    == .macFullCapabilities)
            #expect(
                !FileManager.default.fileExists(atPath: configURL(root).path),
                "a reading must not be written down as a decision")
        }
    }

    /// A workspace made with a choice declares exactly that choice — the
    /// path the new-workspace panel takes from every entry point.
    @Test func aCreatedWorkspaceCarriesItsChoice() throws {
        ExperimentRootOverrideLock.acquire()
        defer { ExperimentRootOverrideLock.release() }
        for choice in ComputeChoice.allCases {
            let root = FileManager.default.temporaryDirectory
                .appending(component: "cc-new-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: root) }
            let created = try WorkspaceStore.create(at: root)
            try WorkspaceCompute.declare(choice, root: created)
            #expect(WorkspaceCompute.declaredChoice(root: created) == choice)
            #expect(WorkspaceCompute.resolved(root: created) == choice.binding)
            #expect(WorkspaceCompute.isDeclared(root: created))
        }
    }

    // MARK: - Declaring versus using

    /// Picking in the Compute menu means "use this now". It records a choice
    /// only for a workspace that has declared none: the declaration decides
    /// whose vectors and evidence count, and must not be rewritten by trying
    /// another engine for an afternoon.
    @Test func usingAChoiceNeverRewritesADeclaration() {
        for choice in ComputeChoice.allCases {
            // Nothing declared and no runs either way: using it records it.
            #expect(
                ComputeChoice.declarationAfterUsing(
                    choice, declared: nil, inferredBinding: nil) == choice)
            // The workspace's own runs already say the same engine: recording
            // it only confirms what they say.
            #expect(
                ComputeChoice.declarationAfterUsing(
                    choice, declared: nil, inferredBinding: choice.binding) == choice)
            // Anything already declared stays exactly as it is.
            for declared in ComputeChoice.allCases {
                for inferred in [nil, WorkspaceCompute.cluster, .localMLX] {
                    #expect(
                        ComputeChoice.declarationAfterUsing(
                            choice, declared: declared, inferredBinding: inferred) == nil)
                }
            }
        }
        // A workspace holding Python runs is not re-labelled by one chat on
        // the quick start, and the reverse.
        #expect(
            ComputeChoice.declarationAfterUsing(
                .macQuickStart, declared: nil, inferredBinding: .cluster) == nil)
        #expect(
            ComputeChoice.declarationAfterUsing(
                .macFullCapabilities, declared: nil, inferredBinding: .localMLX) == nil)
        #expect(
            ComputeChoice.declarationAfterUsing(
                .anotherMachine, declared: nil, inferredBinding: .localMLX) == nil)
    }

    @Test func aMismatchIsAboutTheEngineNotTheMachine() throws {
        // The two Python choices are one engine: no mismatch between them.
        #expect(
            ComputeChoice.mismatchNote(
                workspace: .macFullCapabilities, inUse: .anotherMachine) == nil)
        #expect(
            ComputeChoice.mismatchNote(
                workspace: .anotherMachine, inUse: .macFullCapabilities) == nil)
        for choice in ComputeChoice.allCases {
            #expect(ComputeChoice.mismatchNote(workspace: choice, inUse: choice) == nil)
        }
        // The quick start against either Python choice is, both ways.
        let onQuickStart = try #require(
            ComputeChoice.mismatchNote(workspace: .anotherMachine, inUse: .macQuickStart))
        #expect(onQuickStart.contains("the Python engine"))
        #expect(onQuickStart.contains(ComputeChoice.macQuickStart.title))
        #expect(onQuickStart.contains("do not count"))
        let onPython = try #require(
            ComputeChoice.mismatchNote(
                workspace: .macQuickStart, inUse: .macFullCapabilities))
        #expect(onPython.contains(ComputeChoice.macQuickStart.title))
    }

    // MARK: - The words

    /// Engine names and the old captions, which told a researcher with only
    /// a laptop that the real path was not for them.
    private static let retiredWords = [
        "toy", "pipeline check", "shakedown", "PyTorch", "Cluster (", "Local (MLX)",
        "substrate", "steerlab-cli",
    ]

    @Test func theChoicesAreNamedInPlainWords() {
        #expect(ComputeChoice.macQuickStart.title == "This Mac, quick start")
        #expect(ComputeChoice.macFullCapabilities.title == "This Mac, full capabilities")
        #expect(ComputeChoice.anotherMachine.title == "Another machine")

        let quick = ComputeChoice.macQuickStart.summary
        #expect(quick.contains("small models"))
        #expect(quick.contains("Nothing to install beyond a model"))
        let full = ComputeChoice.macFullCapabilities.summary
        #expect(full.contains("Every method"))
        #expect(full.contains("this Mac"))
        #expect(full.contains("several gigabytes"))
        let other = ComputeChoice.anotherMachine.summary
        #expect(other.contains("workstation or a cluster"))

        #expect(!ComputeChoice.macQuickStart.runsEveryMethod)
        #expect(ComputeChoice.macFullCapabilities.runsEveryMethod)
        #expect(ComputeChoice.anotherMachine.runsEveryMethod)

        var text: [String] = [
            ComputeGuide.title, ComputeGuide.introduction, ComputeGuide.switchButton,
            ComputeGuide.switchCaption, ComputeGuide.guideButton,
            ComputeGuide.needsPythonEngine("Probes", plural: true),
            ComputeChoice.connectAnotherMachine, ComputeChoice.fullCapabilitiesSetup,
            WorkspaceCompute.cluster.label, WorkspaceCompute.localMLX.label,
            ResearchSetupCopy.computeStepTitle, ResearchSetupCopy.computeNeedsWorkspace,
            ResearchSetupCopy.computeCaption,
        ]
        text += ComputeGuide.limits
        text += ComputeGuide.rows.map(\.activity)
        for choice in ComputeChoice.allCases {
            text += [
                choice.title, choice.summary, choice.menuCaption, choice.engineNote,
                ComputeChoice.undeclaredNote(treatingAs: choice),
            ]
        }
        for sentence in text {
            for word in Self.retiredWords {
                #expect(!sentence.contains(word), "'\(word)' in: \(sentence)")
            }
        }
        // The engine IS named once per choice, for the reader who meets the
        // name in an older message.
        #expect(ComputeChoice.macQuickStart.engineNote.contains("MLX"))
        #expect(ComputeChoice.macFullCapabilities.engineNote.contains("Python"))
    }

    // MARK: - What runs where

    @Test func theGuideSaysWhatRunsWhereAndTheLimitsHonestly() throws {
        // Every activity runs on the Python engine — on this Mac or another.
        for row in ComputeGuide.rows {
            #expect(row.pythonEngine, "\(row.id)")
            #expect(row.runs(on: .macFullCapabilities))
            #expect(row.runs(on: .anotherMachine))
            #expect(row.runs(on: .macQuickStart) == row.quickStart)
        }
        #expect(Set(ComputeGuide.rows.map(\.id)).count == ComputeGuide.rows.count)

        // The core of a steering study runs on the quick start…
        let onQuickStart = Set(ComputeGuide.rows.filter(\.quickStart).map(\.id))
        #expect(onQuickStart.isSuperset(of: ["chat", "vector", "study"]))
        // …and the methods that need the Python engine say so.
        let pythonOnly = Set(ComputeGuide.rows.filter { !$0.quickStart }.map(\.id))
        #expect(pythonOnly.isSuperset(of: ["probes", "optvec", "jlens", "sae"]))

        // The three limits, each said plainly.
        #expect(ComputeGuide.limits.count == 3)
        let carry = try #require(ComputeGuide.limits.first { $0.contains("do not carry") })
        #expect(carry.contains("build your vectors again"))
        #expect(ComputeGuide.limits.contains { $0.contains("memory and model size") })
        #expect(
            ComputeGuide.limits.contains { $0.contains("not numerically interchangeable") })
    }

    @Test func aMethodThatNeedsThePythonEngineOffersTheSwitch() {
        let plural = ComputeGuide.needsPythonEngine("Probes", plural: true)
        #expect(plural.hasPrefix("Probes need the Python engine"))
        #expect(plural.contains(ComputeChoice.macQuickStart.title))
        #expect(plural.contains("runs on this Mac too"))
        let singular = ComputeGuide.needsPythonEngine("The Jacobian lens", plural: false)
        #expect(singular.hasPrefix("The Jacobian lens needs the Python engine"))
        #expect(ComputeGuide.switchButton.contains(ComputeChoice.macFullCapabilities.title))
        // The cost of switching is said before the button is pressed.
        #expect(ComputeGuide.switchCaption.contains("nothing is installed until"))
        #expect(ComputeGuide.switchCaption.contains("build them again"))
    }
}

/// Which of the three choices the app is USING — a fact about the
/// connection, read by the Compute menu and the toolbar.
@MainActor
struct ComputeChoiceConnectionTests {

    private func freshDefaults(_ name: String) throws -> UserDefaults {
        let suite = "steerlab.tests.compute-choice.\(name)"
        let defaults = try #require(UserDefaults(suiteName: suite))
        defaults.removePersistentDomain(forName: suite)
        return defaults
    }

    @Test func theConnectionNamesTheChoiceInUse() throws {
        let store = clusterStore(defaults: try freshDefaults("in-use"))
        #expect(store.activeComputeChoice == .macQuickStart)
        #expect(store.activeComputeTitle == "This Mac, quick start")
        #expect(store.substrateLabel == "This Mac, quick start")

        // The Python engine on this Mac: a direct loopback address.
        let engine = store.addServer(
            name: ClusterConnectionStore.thisMacEngineName,
            urlString: "http://127.0.0.1:8080")
        // A workstation reached by its address.
        let workstation = store.addServer(name: "Lab GPU", urlString: "http://gpu-a:8080")

        #expect(store.runsOnThisMac(engine))
        #expect(!store.runsOnThisMac(workstation))
        #expect(store.otherMachines.map(\.id) == [workstation.id])

        store.activeWorkspace = .server(engine.id)
        #expect(store.activeComputeChoice == .macFullCapabilities)
        #expect(store.activeComputeTitle == "This Mac, full capabilities")

        store.activeWorkspace = .server(workstation.id)
        #expect(store.activeComputeChoice == .anotherMachine)
        // Another machine goes by its own name.
        #expect(store.activeComputeTitle == "Lab GPU")

        store.activeWorkspace = .local
        #expect(store.activeComputeChoice == .macQuickStart)
    }

    /// A cluster reached through an SSH tunnel also answers on a loopback
    /// address. It is another machine all the same.
    @Test func aTunnelledClusterIsAnotherMachine() throws {
        let store = clusterStore(defaults: try freshDefaults("tunnel"))
        let entry = store.addPreset(.exampleCluster)
        #expect(entry.resolvedSite.isSSHTransport)
        #expect(!store.runsOnThisMac(entry))
        store.activeWorkspace = .server(entry.id)
        #expect(store.activeComputeChoice == .anotherMachine)
    }
}
