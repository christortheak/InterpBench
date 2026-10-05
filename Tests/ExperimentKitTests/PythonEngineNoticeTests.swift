import Foundation
import Testing

@testable import ExperimentKit

/// Every place in the app that needs the Python engine offers the way to it.
///
/// Wave 1 put "switch to full capabilities" at two sites. A dozen more still
/// refused or only instructed, each in its own words ("Needs a server
/// connection", "Select Python Compute", "switch Compute to …"), and two of
/// them told a researcher on the Python engine to switch to the quick start,
/// which can run none of it. These tests hold the one decision every site now
/// asks (`PythonEngineNotice`), its words, and — because the app target has
/// no unit tests of its own — that each site really asks it.
struct PythonEngineNoticeTests {

    // MARK: - The decision

    @Test func theQuickStartIsOfferedTheSwitchWhateverItsConnection() {
        for connected in [false, true] {
            #expect(
                PythonEngineNotice(inUse: .macQuickStart, connected: connected)
                    == .offerSwitch)
        }
    }

    @Test func thePythonEngineIsNeverOfferedTheSwitch() {
        for choice in [ComputeChoice.macFullCapabilities, .anotherMachine] {
            // Connected: nothing to say, and nothing in the way.
            #expect(PythonEngineNotice(inUse: choice, connected: true) == .none)
            // Not connected: the way out is to connect, not to switch.
            #expect(PythonEngineNotice(inUse: choice, connected: false) == .connect)
        }
    }

    // MARK: - The words

    private static let retiredPhrases = [
        "Python Compute", "server connection", "Compute selector",
        "substrate", "Local (MLX)", "PyTorch",
    ]

    @Test func theSentencesNameTheChoicesInPlainWords() {
        let notConnected = PythonEngineNotice.notConnected("Readout traces", plural: true)
        #expect(notConnected.hasPrefix("Readout traces need the Python engine"))
        #expect(notConnected.contains("Choose Connect here"))
        let statusLine = PythonEngineNotice.notConnected(
            "Copying the vector", plural: false, buttonHere: false)
        #expect(statusLine.hasPrefix("Copying the vector needs the Python engine"))
        // A status line has no button beside it, so it does not point at one.
        #expect(!statusLine.contains("here"))
        #expect(statusLine.contains("connection menu in the toolbar"))

        let whereToRun = PythonEngineNotice.whereToRun("records probe measurements")
        #expect(whereToRun.hasPrefix("The Python engine records probe measurements"))
        #expect(whereToRun.contains("Workspace menu"))

        let switchFirst = PythonEngineNotice.switchFirst("Freezing on the server", plural: false)
        #expect(switchFirst.contains(ComputeChoice.macQuickStart.title))
        #expect(switchFirst.contains("Compute menu"))

        for sentence in [
            notConnected, statusLine, whereToRun, switchFirst,
            PythonEngineNotice.needsPythonEngineBriefly("Jobs", plural: true),
            PythonEngineNotice.notConnectedBriefly,
        ] {
            for phrase in Self.retiredPhrases {
                #expect(!sentence.contains(phrase), "'\(phrase)' in: \(sentence)")
            }
        }
        // Both places the Python engine runs are named wherever the choice
        // is pointed to.
        for sentence in [
            whereToRun, switchFirst,
            PythonEngineNotice.needsPythonEngineBriefly("Jobs", plural: true),
        ] {
            #expect(sentence.contains(ComputeChoice.macFullCapabilities.title))
            #expect(sentence.contains(ComputeChoice.anotherMachine.title))
        }
    }

    // MARK: - The run-start message for probe measurements

    @Test func aStudyWithProbeMeasurementsStopsAndSaysWhereItCanRun() throws {
        var manifest = ExperimentManifest(
            name: "probe-study", description: "", modelID: "test/model")
        #expect(throws: Never.self) { try ExperimentTasks.refuseProbeMeasurements(manifest) }

        manifest.probeMeasurements = .object(["schemaVersion": .number(1)])
        var thrown: ExperimentError?
        do {
            try ExperimentTasks.refuseProbeMeasurements(manifest)
        } catch let error as ExperimentError {
            thrown = error
        }
        let reason = try #require(thrown).reason
        #expect(reason.contains("records probe measurements"))
        #expect(reason.contains("no model was loaded"))
        #expect(reason.contains(ComputeChoice.macFullCapabilities.title))
        #expect(reason.contains(ComputeChoice.anotherMachine.title))
        #expect(!reason.contains("Select Python Compute"))
        #expect(!reason.contains("MLX"))
    }

    // MARK: - Every site asks the one decision (read from the sources)

    private static var repoRoot: URL {
        URL(filePath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().standardizedFileURL
    }

    private static func source(_ relative: String) throws -> String {
        try String(contentsOf: repoRoot.appending(path: relative), encoding: .utf8)
    }

    /// The sites the release review found refusing or only instructing, and
    /// the four wave 1 already fixed. Each shows one of the notice views.
    private static let offerSites = [
        "Sources/SteerLabApp/JLensSupportSection.swift",
        "Sources/SteerLabApp/JLensTraceViews.swift",
        "Sources/SteerLabApp/MethodAuthoringSheet.swift",
        "Sources/SteerLabApp/GeometryPanelView.swift",
        "Sources/SteerLabApp/ServerJobsPanelView.swift",
        "Sources/SteerLabApp/TemplateInstantiationSheet.swift",
        "Sources/SteerLabApp/InterventionPoliciesView.swift",
        "Sources/SteerLabApp/StudyMeasurementsView.swift",
        "Sources/SteerLabApp/ConceptsPanelView.swift",
        "Sources/SteerLabApp/ProbesPanelView.swift",
        "Sources/SteerLabApp/OptVecPanelView.swift",
        "Sources/SteerLabApp/JSpacePanelSection.swift",
        "Sources/SteerLabApp/SAEFeatureImportButton.swift",
    ]

    /// Sheets among them: a switch pressed inside a sheet must present the
    /// engine setup on that sheet, not behind it.
    private static let sheetSites = [
        "Sources/SteerLabApp/MethodAuthoringSheet.swift",
        "Sources/SteerLabApp/TemplateInstantiationSheet.swift",
        "Sources/SteerLabApp/InterventionPoliciesView.swift",
        // The J-lens library sheet, which shows the J-Space section's offer.
        "Sources/SteerLabApp/JSpacePanelSection.swift",
    ]

    @Test func everySiteShowsTheNoticeAndEverySheetHostsItsSetup() throws {
        let views = ["PythonEngineNeeded(", "PythonEngineOffer(", "PythonEngineSwitchControls("]
        for site in Self.offerSites {
            let text = try Self.source(site)
            #expect(views.contains { text.contains($0) }, "no Python-engine notice in \(site)")
        }
        for site in Self.sheetSites {
            #expect(
                try Self.source(site).contains(".hostsComputeSheets()"),
                "\(site) offers the switch but would present its setup behind itself")
        }
        // The notice decides by the one rule, not by a test of its own.
        let view = try Self.source("Sources/SteerLabApp/PythonEngineNeeded.swift")
        #expect(view.contains("PythonEngineNotice(\n                inUse: compute.inUse"))
        // "Create and Submit" follows the same rule, so it is off on the
        // quick start (where a client object always exists) and on once the
        // switch is made.
        let batch = try Self.source("Sources/SteerLabApp/TemplateInstantiationSheet.swift")
        #expect(batch.contains("PythonEngineNotice(\n            inUse: compute.inUse"))
        #expect(batch.contains("|| !canSubmit)"))
        // The main window stands down while a sheet presents the setup.
        let app = try Self.source("Sources/SteerLabApp/SteerLabApp.swift")
        #expect(app.contains("&& compute.sheetHosts.isEmpty"))
        #expect(app.contains(".environment(localServer)"))
    }

    /// The instructions that used to stand where the notice now is. Outside
    /// comments, none is shown to a researcher anywhere in the app.
    @Test func theRetiredInstructionsAreGone() throws {
        let retired = [
            "Select Python Compute", "using Python Compute", "on Python Compute",
            "Needs a server connection", "needs a server connection",
            "no server connection", "Choose a server compute target",
            "Connect to a server workspace", "Select a cluster workspace",
            "switch the substrate selector", "Choose a Python engine in Compute",
        ]
        // ExperimentPanel.swift's refusal text belongs to another stream in
        // this wave (refusal presentation); it is reworded there.
        let excused: Set<String> = ["Sources/ExperimentKit/ExperimentPanel.swift"]
        let enumerator = try #require(
            FileManager.default.enumerator(
                at: Self.repoRoot.appending(component: "Sources"),
                includingPropertiesForKeys: nil))
        var offenders: [String] = []
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            guard let text = try? String(contentsOf: url, encoding: .utf8) else { continue }
            let relative = String(
                url.standardizedFileURL.path.dropFirst(Self.repoRoot.path.count + 1))
            guard !excused.contains(relative) else { continue }
            for line in text.split(separator: "\n") {
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                guard !trimmed.hasPrefix("//") else { continue }
                for phrase in retired where trimmed.contains(phrase) {
                    offenders.append("\(relative): \(phrase)")
                }
            }
        }
        #expect(offenders.isEmpty, "\(offenders.joined(separator: "\n"))")
    }
}
