import Foundation
import Testing

@testable import ExperimentKit

/// A repair is a command the reader can run, on the client in front of them.
///
/// The Mac command line (`steerlab-cli`) and the cross-platform client
/// (`steerlab`) are two products that cannot run each other's commands. The
/// Python side renders its shared repairs for whoever shows them
/// (`Server/steerlab_server/experiment/command_vocabulary.py`), and its test,
/// `Server/tests/test_refusal_vocabulary.py`, holds that no refusal through
/// the cross-platform client names `steerlab-cli` or says "on the Mac".
///
/// This is the other half: no document the Mac command line produces tells
/// its reader to use the cross-platform client's verbs. It checks every
/// committed envelope golden (each produced by this command line) and a set
/// of refusals driven live, including the freeze, frozen-study, detach,
/// sweep-grid, not-found and data-check refusals.
///
/// Serialized, and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct RefusalVocabularyTests {

    /// The cross-platform client's families (`client_cli.FAMILIES`).
    static let clientFamilies = [
        "setup", "workspace", "authoring", "experiment", "concept", "bundle",
        "pack", "design", "agent", "panel", "model", "science", "runner", "run",
    ]

    /// A command spelled for the cross-platform client: its program, then
    /// one of its families. Never preceded by a word character or a hyphen,
    /// so `steerlab-cli …` and `steerlab-server …` do not match.
    static func namesClientCommand(_ text: String) -> Bool {
        let pattern =
            #"(?<![\w-])steerlab (?:"# + clientFamilies.joined(separator: "|") + #")\b"#
        return text.range(of: pattern, options: .regularExpression) != nil
    }

    /// Every sentence a document shows its reader.
    static func texts(of object: [String: Any]) -> [String] {
        var out: [String] = []
        if let message = object["message"] as? String { out.append(message) }
        if let error = object["error"] as? [String: Any] {
            for key in ["reason", "repairAction"] {
                if let text = error[key] as? String { out.append(text) }
            }
        }
        if let action = object["nextAction"] as? [String: Any] {
            for key in ["verb", "detail"] {
                if let text = action[key] as? String { out.append(text) }
            }
        }
        for advisory in object["advisories"] as? [[String: Any]] ?? [] {
            if let detail = advisory["detail"] as? String { out.append(detail) }
        }
        return out
    }

    /// The sentences that would send a Mac reader to the other client.
    static func problems(in object: [String: Any]) -> [String] {
        texts(of: object).filter {
            namesClientCommand($0) || $0.contains("cross-platform client")
                || $0.contains("on your authoring client")
        }
    }

    static func object(_ text: String) throws -> [String: Any] {
        try #require(
            try JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
    }

    func withTempRoot<T>(_ body: (URL) async throws -> T) async throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "refusal-vocabulary-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: temp, withIntermediateDirectories: true)
        ExperimentStore.rootOverride = temp
        defer {
            ExperimentStore.rootOverride = nil
            try? FileManager.default.removeItem(at: temp)
            ExperimentRootOverrideLock.release()
        }
        return try await body(temp)
    }

    @discardableResult
    func invoke(_ namespace: String, _ args: [String]) async -> ExperimentCLIOutcome {
        await ExperimentCLIRunner(sink: .discarding).run(namespace: namespace, args)
    }

    // MARK: -

    @Test func theDetectorTellsTheTwoProgramsApart() {
        #expect(Self.namesClientCommand("then steerlab experiment freeze demo"))
        #expect(Self.namesClientCommand("steerlab run demo --runner <url>"))
        #expect(!Self.namesClientCommand("steerlab-cli experiment freeze demo"))
        #expect(!Self.namesClientCommand("steerlab-server experiment validate demo"))
        #expect(!Self.namesClientCommand("SteerLab experiments"))
    }

    @Test func noCommittedMacDocumentNamesTheOtherClientsVerbs() throws {
        let directory = CodeResources.compiledCheckoutPath.appending(
            components: "Tests", "Fixtures", "cli-envelopes")
        let goldens = ((try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "json" }
        #expect(!goldens.isEmpty, "no committed cli-envelope fixtures")
        for url in goldens {
            let document = try Self.object(String(contentsOf: url, encoding: .utf8))
            #expect(
                Self.problems(in: document).isEmpty,
                "\(url.lastPathComponent): \(Self.problems(in: document))")
        }
    }

    @Test func liveRefusalsNameNoOtherClientsVerbs() async throws {
        try await withTempRoot { _ in
            let model = "mlx-community/gemma-3-4b-it-4bit"
            await invoke("experiment", ["create", "demo", "--model", model])
            await invoke("experiment", ["attach", "demo", "french"])
            await invoke(
                "experiment",
                ["declare-condition", "demo", "arm", "--slots", "french:10:0.1",
                 "--alpha-units", "norm"])
            await invoke("experiment", ["create", "empty", "--model", model])
            await invoke("experiment", ["create", "done", "--model", model])
            await invoke("experiment", ["attach", "done", "french"])
            await invoke(
                "experiment",
                ["declare-condition", "done", "arm", "--slots", "french:10:0.1",
                 "--alpha-units", "norm"])
            let frozen = await invoke("experiment", ["freeze", "done", "--force"])
            #expect(frozen.envelope.state.isSuccess, "the frozen fixture did not freeze")

            let refusals: [(String, [String])] = [
                ("experiment", ["freeze", "demo"]),
                ("experiment", ["freeze", "empty"]),
                ("experiment", ["freeze", "done"]),
                ("experiment", ["attach", "done", "french"]),
                ("experiment", ["detach", "demo", "french"]),
                ("experiment", ["set-sweep-grid", "demo", "--alphas", "0.1,0.05"]),
                ("experiment", ["verify", "nosuch"]),
                ("experiment", ["nonsense"]),
                ("data", ["check", "demo"]),
            ]
            for (namespace, args) in refusals {
                let outcome = await invoke(namespace, args)
                let label = ([namespace] + args).joined(separator: " ")
                #expect(!outcome.envelope.state.isSuccess, "\(label) did not refuse")
                let document = try Self.object(outcome.envelope.jsonText())
                #expect(
                    Self.problems(in: document).isEmpty,
                    "\(label): \(Self.problems(in: document))")
            }
        }
    }
}
