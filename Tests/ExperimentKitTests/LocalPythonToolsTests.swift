import Foundation
import Testing

@testable import ExperimentKit

/// Tests for the local Python tooling: the stimulus-independence verdict
/// phrasing (exit-code contract with
/// `Server/steerlab_server/experiment/stimulus_screen.py`), the local
/// server's exit summaries (exit-code contract with
/// `scripts/start-local-server.sh`), and the pidfile adoption decision with
/// its identity rules (contract with the script's `exec`ed argv and the
/// Python server's `/api/info` self-identification). Pure logic except the
/// two self-probes (own-pid liveness and command line).
@Suite struct LocalPythonToolsTests {

    // MARK: Stimulus-independence verdicts

    @Test func cleanExitReadsClean() {
        let verdict = StimulusIndependenceScreen.verdict(
            exitCode: 0,
            output: [
                "vocabulary: shipped judicial-study default (…)",
                "prompts/concepts/x: clean — no forbidden-vocabulary terms in stimulus text",
            ])
        #expect(verdict.hasPrefix("Clean"))
        #expect(verdict.contains("forbidden-vocabulary"))
    }

    @Test func findingsExitCountsFlaggedLines() {
        let output = [
            "vocabulary: workspace file prompts/screens/forbidden-vocabulary.json — 3 term(s) in list(s): maritime",
            "positive.jsonl:3: court, judge — 'the court found…'",
            "negative.jsonl:7: verdict — 'a verdict was…'",
            "2 flagged line(s); review before pinning this set",
        ]
        let verdict = StimulusIndependenceScreen.verdict(exitCode: 1, output: output)
        #expect(verdict.hasPrefix("2 flagged line(s)"))
        #expect(verdict.contains("forbidden vocabulary"))
        #expect(verdict.contains("review"))
        #expect(StimulusIndependenceScreen.flaggedLineCount(in: output) == 2)
    }

    @Test func findingsExitWithoutSummaryLineStaysHonest() {
        // A truncated/garbled output still yields a review verdict, with
        // "Some" instead of a fabricated count.
        let verdict = StimulusIndependenceScreen.verdict(
            exitCode: 1, output: ["positive.jsonl:1: court — '…'"])
        #expect(verdict.hasPrefix("Some flagged line(s)"))
        #expect(StimulusIndependenceScreen.flaggedLineCount(in: []) == nil)
    }

    @Test func usageExitExplainsPlainly() {
        let verdict = StimulusIndependenceScreen.verdict(exitCode: 2, output: [])
        #expect(verdict.contains("could not run"))
    }

    @Test func unexpectedExitNamesTheCode() {
        let verdict = StimulusIndependenceScreen.verdict(exitCode: 9, output: [])
        #expect(verdict.contains("exit 9"))
    }

    // MARK: Local server exit summaries

    @Test func cleanServerExitIsStopped() {
        #expect(
            LocalServerController.exitSummary(code: 0, wasStopping: false, port: 8080)
                == "stopped")
    }

    @Test func userStopIsStoppedRegardlessOfCode() {
        // SIGTERM usually surfaces as a nonzero code; a user-initiated Stop
        // must still read as a stop, never as a failure.
        #expect(
            LocalServerController.exitSummary(code: 15, wasStopping: true, port: 8080)
                == "stopped")
    }

    @Test func busyPortNamesThePortAndRemedy() {
        let summary = LocalServerController.exitSummary(
            code: 3, wasStopping: false, port: 8080)
        #expect(summary.contains("port 8080"))
        #expect(summary.contains("already in use"))
    }

    @Test func missingPythonNamesTheRemedy() {
        let summary = LocalServerController.exitSummary(
            code: 4, wasStopping: false, port: 8080)
        #expect(summary.contains("python3"))
    }

    @Test func unexpectedServerExitPointsAtActivityPane() {
        let summary = LocalServerController.exitSummary(
            code: 7, wasStopping: false, port: 8080)
        #expect(summary.contains("code 7"))
        #expect(summary.contains("Activity"))
    }

    // MARK: Local-server pidfile (adoption across relaunch)

    @Test func pidfileParsesPidAndPort() {
        let parsed = LocalServerPidfile.parse("4321 8080\n")
        #expect(parsed?.pid == 4321)
        #expect(parsed?.port == 8080)
    }

    @Test func pidfileParseRejectsGarbage() {
        #expect(LocalServerPidfile.parse("") == nil)
        #expect(LocalServerPidfile.parse("not-a-pid") == nil)
        #expect(LocalServerPidfile.parse("4321") == nil)  // pid without port
        #expect(LocalServerPidfile.parse("4321 8080 extra") == nil)
        #expect(LocalServerPidfile.parse("-1 8080") == nil)  // signal-a-group pid
        #expect(LocalServerPidfile.parse("0 8080") == nil)
        #expect(LocalServerPidfile.parse("4321 0") == nil)  // port out of range
        #expect(LocalServerPidfile.parse("4321 70000") == nil)
    }

    @Test func pidfileParseToleratesWhitespaceVariants() {
        let parsed = LocalServerPidfile.parse("  4321\t8080  \n")
        #expect(parsed?.pid == 4321)
        #expect(parsed?.port == 8080)
    }

    /// The full decision, one injected-check matrix. Finding B: liveness
    /// alone used to enable Stop — pid reuse plus an unrelated listener
    /// meant Stop could SIGTERM an innocent process. Now process identity
    /// must be BOUND (argv names steerlab_server AND this pidfile's port)
    /// and the endpoint must identify before `.adopt`.
    @Test func adoptionRequiresLivenessAndBoundIdentityChecks() {
        func decide(
            contents: String = "4321 8080",
            workspaceRoot: String? = "/workspace",
            alive: Bool = true,
            command: String? = Self.serverCommand,
            listening: Bool = true,
            endpoint: Bool = true
        ) -> LocalServerPidfile.Adoption {
            LocalServerPidfile.adoptionDecision(
                contents: contents,
                workspaceRoot: workspaceRoot,
                processIsAlive: { _ in alive },
                commandLine: { _ in command },
                portIsListening: { _ in listening },
                endpointIsSteerLab: { _ in endpoint })
        }
        // Everything checks out → adopt, Stop safe.
        #expect(decide() == .adopt(pid: 4321, port: 8080))
        // Liveness failures are stale exactly as before — even beside a
        // live listener (dead pid) or a live steerlab process (silent port).
        #expect(decide(alive: false) == .stale)
        #expect(decide(listening: false) == .stale)
        #expect(decide(contents: "garbled") == .stale)
        // Identity failures with a live listener: NEVER Stop.
        // Reused pid (argv is some unrelated program):
        #expect(decide(command: "/usr/bin/top") == .portInUse(port: 8080))
        // Command line unavailable (ps failed): unknown is not ours.
        #expect(decide(command: nil) == .portInUse(port: 8080))
        // Right process name, but the port answers as something else
        // entirely (another program grabbed the port first):
        #expect(decide(endpoint: false) == .portInUse(port: 8080))
        // THE BINDING CASES (the P2 fix): a live SteerLab pid whose argv
        // names a DIFFERENT port than the pidfile — with two local servers,
        // the pid identifies server A while port 8080 serves server B; Stop
        // would kill the wrong one. Never adoptable, never stoppable.
        #expect(
            decide(
                command: "/repo/Server/.venv.nosync/bin/python -m "
                    + "steerlab_server.cli serve --port 9090 --root /workspace")
                == .unbound(port: 8080))
        // Legacy pidfile (script version without --port on the argv):
        // degrades to the SAFE unbound state — stop it manually once.
        #expect(
            decide(
                command: "/repo/Server/.venv.nosync/bin/python -m "
                    + "steerlab_server.cli serve")
                == .unbound(port: 8080))
        // Right port, but the argv's --root names a DIFFERENT workspace
        // than the pidfile's: bound-when-present root check demotes safely.
        #expect(decide(workspaceRoot: "/elsewhere") == .unbound(port: 8080))
        // No workspace root supplied (pure callers): port binding alone
        // still gates.
        #expect(decide(workspaceRoot: nil) == .adopt(pid: 4321, port: 8080))
    }

    private static let serverCommand =
        "/repo/Server/.venv.nosync/bin/python -m steerlab_server.cli serve "
        + "--port 8080 --root /workspace"

    @Test func adoptionChecksReceiveTheParsedValues() {
        var probedPID: Int32?
        var commandPID: Int32?
        var probedPort: Int?
        var endpointPort: Int?
        _ = LocalServerPidfile.adoptionDecision(
            contents: "4321 8080",
            processIsAlive: { pid in
                probedPID = pid
                return true
            },
            commandLine: { pid in
                commandPID = pid
                return Self.serverCommand
            },
            portIsListening: { port in
                probedPort = port
                return true
            },
            endpointIsSteerLab: { port in
                endpointPort = port
                return true
            })
        #expect(probedPID == 4321)
        #expect(commandPID == 4321)
        #expect(probedPort == 8080)
        #expect(endpointPort == 8080)
    }

    // MARK: Identity rules (pure matchers)

    @Test func commandLineRuleMatchesTheScriptsArgv() {
        // scripts/start-local-server.sh `exec`s
        // `python -m steerlab_server.cli serve …` — the argv the pidfile's
        // pid must carry.
        #expect(LocalServerPidfile.commandLineIsSteerLabServer(Self.serverCommand))
        #expect(
            LocalServerPidfile.commandLineIsSteerLabServer(
                "env PYTHONUNBUFFERED=1 python -m steerlab_server.cli serve"))
        // Unrelated programs, the SWIFT server, and unknown all fail.
        #expect(!LocalServerPidfile.commandLineIsSteerLabServer("/usr/bin/top"))
        #expect(
            !LocalServerPidfile.commandLineIsSteerLabServer(
                "steerlab-cli serve --port 8080"))
        #expect(!LocalServerPidfile.commandLineIsSteerLabServer(""))
        #expect(!LocalServerPidfile.commandLineIsSteerLabServer(nil))
    }

    /// The BOUND matcher — the P2 fix's core rule: argv must name
    /// steerlab_server AND carry the pidfile's port.
    @Test func boundCommandLineRuleBindsArgvToThePidfilesPort() {
        func matches(
            _ command: String?, port: Int = 8080, root: String? = nil
        ) -> Bool {
            LocalServerPidfile.commandLineMatchesSteerLabServer(
                command, port: port, workspaceRoot: root)
        }
        // The script's exact exec'd argv shape: bound.
        #expect(matches(Self.serverCommand))
        // `--port=8080` form binds too.
        #expect(
            matches("python -m steerlab_server.cli serve --port=8080"))
        // Wrong port: a SteerLab server, but not THIS pidfile's server.
        #expect(!matches(Self.serverCommand, port: 9090))
        // Port token equality, not prefix: 8080 must not match 80801.
        #expect(
            !matches("python -m steerlab_server.cli serve --port 80801"))
        // Legacy argv without --port (older script version): never binds —
        // safe degrade, the user stops that server manually once.
        #expect(!matches("python -m steerlab_server.cli serve"))
        // Right port on a non-SteerLab program: still no.
        #expect(!matches("steerlab-cli serve --port 8080"))
        #expect(!matches(nil))

        // Root binding — enforced only when the argv names a --root.
        // Matching root (incl. trailing-slash spelling): bound.
        #expect(matches(Self.serverCommand, root: "/workspace"))
        #expect(matches(Self.serverCommand, root: "/workspace/"))
        // iCloud-style path with spaces (ps output cannot be tokenized):
        let spaced = "python -m steerlab_server.cli serve --port 8080 "
            + "--root /Users/x/Mobile Documents/ws"
        #expect(matches(spaced, root: "/Users/x/Mobile Documents/ws"))
        // A different workspace, or a sibling that merely shares a prefix:
        // not bound.
        #expect(!matches(Self.serverCommand, root: "/elsewhere"))
        #expect(
            !matches(
                "python -m steerlab_server.cli serve --port 8080 "
                    + "--root /workspace2",
                root: "/workspace"))
        // Argv WITHOUT --root (terminal start relying on STEERLAB_ROOT):
        // root is bound-when-present, never required retroactively.
        #expect(
            matches(
                "python -m steerlab_server.cli serve --port 8080",
                root: "/workspace"))
    }

    @Test func endpointRuleRequiresTheServersSelfIdentification() {
        // /api/info answers {"service": "steerlab-server", …} — the
        // key/value only SteerLab returns.
        let response = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            + "\r\n{\"service\": \"steerlab-server\", \"root\": \"/w\"}"
        #expect(LocalServerPidfile.responseIdentifiesSteerLab(response))
        // Generic health JSON, another program's page, and silence all fail
        // identity.
        #expect(
            !LocalServerPidfile.responseIdentifiesSteerLab(
                "HTTP/1.1 200 OK\r\n\r\n{\"ok\": true}"))
        #expect(
            !LocalServerPidfile.responseIdentifiesSteerLab(
                "HTTP/1.1 200 OK\r\n\r\n<html><body>It works!</body></html>"))
        #expect(!LocalServerPidfile.responseIdentifiesSteerLab(nil))
        #expect(!LocalServerPidfile.responseIdentifiesSteerLab(""))
    }

    /// Token-mode identity WITHOUT authentication: STEERLAB_AUTH_MODE=token
    /// gates every route, so the unauthenticated /api/info probe gets the
    /// server's OWN 401 shape ({"detail": "missing or invalid bearer
    /// token"}, authored in Server/steerlab_server/api/app.py — not a
    /// framework default). That exact shape identifies the endpoint; Stop
    /// still requires the bound-argv process check, so accepting it weakens
    /// nothing, and the probe never sends a stored token.
    @Test func endpointRuleAcceptsTheTokenModeRefusalShapes() {
        // The exact 401 the middleware returns (uvicorn's compact JSON).
        #expect(
            LocalServerPidfile.responseIdentifiesSteerLab(
                "HTTP/1.1 401 Unauthorized\r\nContent-Type: application/json"
                    + "\r\n\r\n{\"detail\":\"missing or invalid bearer token\"}"))
        // Token mode with no token configured → 503 naming
        // STEERLAB_AUTH_TOKEN — unmistakably SteerLab's.
        #expect(
            LocalServerPidfile.responseIdentifiesSteerLab(
                "HTTP/1.1 503 Service Unavailable\r\n\r\n"
                    + "{\"detail\":\"STEERLAB_AUTH_TOKEN is not configured\"}"))
        // Generic 401s (other programs, framework defaults) still fail.
        #expect(
            !LocalServerPidfile.responseIdentifiesSteerLab(
                "HTTP/1.1 401 Unauthorized\r\n\r\n"
                    + "{\"detail\":\"Not authenticated\"}"))
        #expect(
            !LocalServerPidfile.responseIdentifiesSteerLab(
                "HTTP/1.1 401 Unauthorized\r\n\r\nUnauthorized"))
        // The magic phrase in a NON-401 response is not the server's shape.
        #expect(
            !LocalServerPidfile.responseIdentifiesSteerLab(
                "HTTP/1.1 200 OK\r\n\r\n"
                    + "{\"detail\":\"missing or invalid bearer token\"}"))
        // A 401 wrapped in something that is not an HTTP status line fails.
        #expect(
            !LocalServerPidfile.responseIdentifiesSteerLab(
                "401 {\"detail\":\"missing or invalid bearer token\"}"))
    }

    @Test func processCommandLineReadsARealProcess() {
        // Our own pid definitionally has a command line; a nonpositive pid
        // never does.
        #expect(LocalServerPidfile.processCommandLine(getpid()) != nil)
        #expect(LocalServerPidfile.processCommandLine(0) == nil)
        #expect(LocalServerPidfile.processCommandLine(-1) == nil)
    }

    @Test func processLivenessProbeAgreesWithReality() {
        // Our own process is definitionally alive; nonpositive pids are
        // never adoptable (kill(-pid, …) would signal a process GROUP).
        #expect(LocalServerPidfile.processIsAlive(getpid()))
        #expect(!LocalServerPidfile.processIsAlive(0))
        #expect(!LocalServerPidfile.processIsAlive(-1))
    }

    // MARK: Runtime path shape

    @Test func venvAndScriptResolveUnderTheRepoRoot() throws {
        // This test process runs from the developer checkout, so the
        // developer-gated accessors resolve (they are nil only in a build
        // without a checkout — see ReleaseModeResourceTests).
        let root = try #require(LocalPythonRuntime.repoRoot).path
        let venvPython = try #require(LocalPythonRuntime.venvPython)
        #expect(venvPython.path.hasPrefix(root))
        #expect(venvPython.path.hasSuffix("Server/.venv.nosync/bin/python"))
        let script = try #require(LocalPythonRuntime.startServerScript)
        #expect(script.path.hasSuffix("scripts/start-local-server.sh"))
        // The one-click script must actually ship with the checkout.
        #expect(FileManager.default.fileExists(atPath: script.path))
    }
}
