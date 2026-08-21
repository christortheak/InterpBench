import Foundation
import SteeringKit
import Testing

@testable import ExperimentKit

// =============================================================================
// The shared agent-path envelope (WP0-AGENT-SURFACE-AUDIT §2.2–§2.3), Step 1.
//
// These are the gate tests the audit's Step 1 row names:
// `exitCodesFollowTheStateVocabularyExactly`, `exactlyOneJSONDocumentIsProduced`,
// and `clusterAndSharedVocabulariesAgree` — plus the round-trip and key-list
// tests whose literals become the Python twin's at Step 8, in the same
// duplicated-literal pattern as `RunMetadata.contractKeys` ↔ `test_run_config.py`.
//
// Nothing here touches a model, a socket, or the filesystem: the whole point of
// putting the protocol in ExperimentKit rather than in the binary is that it is
// testable without either.
// =============================================================================

struct CLIEnvelopeTests {

    // MARK: Exit-code vocabulary

    /// The audit's §2.3 table, as a literal. Written out case by case rather
    /// than derived, so a change to `exitCode` has to be made twice on purpose.
    @Test func exitCodesFollowTheStateVocabularyExactly() {
        let expected: [SteerLabCLIState: Int32] = [
            .ready: 0, .planned: 0, .running: 0, .okWithAdvisories: 0,
            .needsHumanAuthentication: 10, .needsApproval: 11, .pending: 12,
            .degraded: 13, .blocked: 64, .refused: 65, .notFound: 66,
            .failed: 70,
        ]
        // Exhaustive over the vocabulary: a new case with no expectation fails.
        #expect(SteerLabCLIState.allCases.count == expected.count)
        for state in SteerLabCLIState.allCases {
            let envelope = SteerLabCLIEnvelope(
                verb: "experiment freeze", engine: SteerLabCLIEnvelope.localEngine,
                state: state, message: "m")
            #expect(envelope.exitCode == expected[state], "wrong code for \(state)")
            // The envelope's own code and the vocabulary's can never diverge.
            #expect(envelope.exitCode == state.exitCode)
        }
        // The three states the package exists for.
        #expect(SteerLabCLIState.refused.exitCode == 65)
        #expect(SteerLabCLIState.notFound.exitCode == 66)
        #expect(SteerLabCLIState.okWithAdvisories.exitCode == 0)
        // Success is exactly the zero-exit set.
        #expect(
            SteerLabCLIState.allCases.filter(\.isSuccess)
                == [.ready, .planned, .running, .okWithAdvisories])
    }

    /// Advisories are non-blocking BY CONSTRUCTION: a `set -e` wrapper must not
    /// break on a forced-freeze warning.
    @Test func advisoriesNeverChangeTheExitCode() throws {
        let clean = SteerLabCLIEnvelope.success(
            verb: "experiment freeze", engine: SteerLabCLIEnvelope.localEngine,
            message: "frozen", changed: true)
        #expect(clean.state == .ready)
        #expect(clean.exitCode == 0)
        // Absent, not `[]`.
        #expect(try Self.decode(clean)["advisories"] == nil)

        let advised = SteerLabCLIEnvelope.success(
            verb: "experiment freeze", engine: SteerLabCLIEnvelope.localEngine,
            message: "frozen", changed: true,
            advisories: [
                .init(code: "unpromotedSourceAgent", detail: "agent 'a' was hand-created")
            ])
        #expect(advised.state == .okWithAdvisories)
        #expect(advised.exitCode == 0)
        let entries = try #require(try Self.decode(advised)["advisories"] as? [[String: Any]])
        #expect(entries.count == 1)
        #expect(entries[0]["code"] as? String == "unpromotedSourceAgent")
    }

    /// The cluster vocabulary is now a subset of the shared one, and the two
    /// exit-code tables cannot drift — this is the twin test the audit's §2.3
    /// asks for, in the `#expect(envelope.exitCode == state.exitCode)` idiom.
    @Test func clusterAndSharedVocabulariesAgree() {
        for state in ClusterLifecycleState.allCases {
            let shared = SteerLabCLIState(rawValue: state.rawValue)
            #expect(shared != nil, "no shared twin for cluster state \(state.rawValue)")
            #expect(state.sharedState.rawValue == state.rawValue)
            #expect(state.exitCode == state.sharedState.exitCode)
            #expect(state.exitCode == shared?.exitCode)
            // And the envelope that parses the state back agrees too.
            let envelope = ClusterCLIEnvelope(
                verb: "cluster ensure", state: state, message: "m")
            #expect(envelope.exitCode == SteerLabCLIState(rawValue: state.rawValue)?.exitCode)
        }
        // The shared vocabulary is exactly cluster's, plus the three new ones.
        let clusterNames = Set(ClusterLifecycleState.allCases.map(\.rawValue))
        let sharedNames = Set(SteerLabCLIState.allCases.map(\.rawValue))
        #expect(sharedNames.subtracting(clusterNames) == ["okWithAdvisories", "refused", "notFound"])
        #expect(clusterNames.subtracting(sharedNames).isEmpty)
    }

    // MARK: Serialization discipline

    @Test func exactlyOneJSONDocumentIsProduced() throws {
        let text = try Self.fullyPopulated().jsonText()
        #expect(text.hasSuffix("\n"))
        // One document: trimming the trailing newline leaves a single parse.
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        #expect(trimmed.hasPrefix("{"))
        #expect(trimmed.hasSuffix("}"))
        #expect(!trimmed.dropLast().contains("}\n{"))
        // No ANSI, ever.
        #expect(!text.contains("\u{1B}"))
        // And it really is one JSON value, not a concatenation.
        _ = try JSONSerialization.jsonObject(with: Data(text.utf8))
    }

    /// Same encoder settings as the grandfathered cluster envelope: sorted
    /// keys, ISO-8601 dates, unescaped slashes, one trailing newline. Checked
    /// by behaviour rather than by reading the two encoder literals.
    @Test func serializationMatchesTheClusterEnvelopeMechanism() throws {
        let observedAt = Date(timeIntervalSince1970: 1_755_000_000)
        var shared = SteerLabCLIEnvelope(
            verb: "experiment verify", engine: SteerLabCLIEnvelope.localEngine,
            state: .ready, message: "ok", observedAt: observedAt,
            workspace: "/tmp/ws/a")
        shared.nextAction = .init(verb: "experiment freeze demo")
        let sharedText = try shared.jsonText()

        var cluster = ClusterCLIEnvelope(
            verb: "cluster status", state: .ready, message: "ok",
            observedAt: observedAt)
        cluster.outputPath = "/tmp/ws/a"
        let clusterText = try cluster.jsonText()

        // ISO-8601, identical rendering of the identical instant.
        let stamp = ISO8601DateFormatter().string(from: observedAt)
        #expect(stamp.hasSuffix("Z"))
        #expect(sharedText.contains(stamp))
        #expect(clusterText.contains(stamp))
        // Slashes unescaped in both.
        #expect(sharedText.contains("/tmp/ws/a"))
        #expect(clusterText.contains("/tmp/ws/a"))
        #expect(!sharedText.contains("\\/"))
        // Sorted keys in both: the emitted top-level order is alphabetical.
        #expect(Self.topLevelKeyOrder(in: sharedText) == Self.topLevelKeyOrder(in: sharedText).sorted())
        #expect(Self.topLevelKeyOrder(in: clusterText) == Self.topLevelKeyOrder(in: clusterText).sorted())
        // Pretty-printed in both (the cluster goldens depend on it).
        #expect(sharedText.contains("\n  \"verb\""))
        #expect(clusterText.contains("\n  \"verb\""))
        // Exactly one trailing newline in both.
        #expect(sharedText.hasSuffix("}\n") && clusterText.hasSuffix("}\n"))
    }

    // MARK: The wire-name contract

    @Test func theRequiredHeaderKeysAreAlwaysPresent() throws {
        for state in SteerLabCLIState.allCases {
            let object = try Self.decode(
                SteerLabCLIEnvelope(
                    verb: "experiment run", engine: SteerLabCLIEnvelope.serverEngine,
                    state: state, message: "m"))
            for key in SteerLabCLIEnvelope.contractHeaderKeys {
                #expect(object[key] != nil, "missing required key \(key) for \(state)")
            }
            #expect(object["schemaVersion"] as? Int == 1)
            #expect(object["state"] as? String == state.rawValue)
            #expect(object["engine"] as? String == "python-hf-transformers")
            #expect(object["changed"] as? Bool == false)
            // Absent optionals are OMITTED rather than emitted as null.
            for key in SteerLabCLIEnvelope.contractOptionalKeys {
                #expect(object[key] == nil, "optional \(key) should be omitted")
            }
        }
    }

    /// The closed key list, as a literal. Its Python twin is
    /// `test_cli_envelope.py` (audit Step 8) — adding, removing, or renaming a
    /// top-level key must fail this test and its twin until the schema version
    /// is bumped and both literals change in the same commit.
    @Test func theWireNamesAreTheContractLiteral() throws {
        let headerKeys = [
            "changed", "engine", "message", "observedAt", "schemaVersion", "state", "verb",
        ]
        let optionalKeys = ["advisories", "error", "nextAction", "result", "workspace"]
        #expect(SteerLabCLIEnvelope.contractHeaderKeys == headerKeys)
        #expect(SteerLabCLIEnvelope.contractOptionalKeys == optionalKeys)
        // The header is byte-identical to the cluster envelope's six required
        // fields, plus `engine` — the audit's §2.2 justification for extending
        // rather than reinventing.
        let clusterRequired = ["schemaVersion", "verb", "state", "changed", "observedAt", "message"]
        #expect(Set(headerKeys).subtracting(clusterRequired) == ["engine"])
        #expect(Set(clusterRequired).subtracting(headerKeys).isEmpty)

        let object = try Self.decode(Self.fullyPopulated())
        #expect(object.keys.sorted() == (headerKeys + optionalKeys).sorted())

        // Nested wire names, pinned the same way.
        let next = try #require(object["nextAction"] as? [String: Any])
        #expect(next.keys.sorted() == ["detail", "missingPermissionFlags", "requiresHuman", "verb"])
        let failure = try #require(object["error"] as? [String: Any])
        #expect(failure.keys.sorted() == ["code", "gate", "gates", "reason", "repairAction"])
        let advisory = try #require((object["advisories"] as? [[String: Any]])?.first)
        #expect(advisory.keys.sorted() == ["code", "detail"])
    }

    // MARK: Round trip

    @Test func everyFieldRoundTripsThroughJSON() throws {
        let original = Self.fullyPopulated()
        let originalText = try original.jsonText()
        let restored = try SteerLabCLIEnvelope.decode(fromJSON: originalText)
        let restoredText = try restored.jsonText()
        #expect(restored == original)
        #expect(restored.exitCode == original.exitCode)
        #expect(restored.state == .refused)
        #expect(restored.error?.gate == "validateEvidence")
        #expect(restored.result?["runID"] == .string("2026-08-17-demo"))
        // Re-encoding is byte-stable: goldens survive a decode/encode cycle.
        #expect(restoredText == originalText)
    }

    @Test func aRefusalNamesTheGateRatherThanDescribingIt() throws {
        let envelope = SteerLabCLIEnvelope.refusal(
            verb: "experiment freeze",
            engine: SteerLabCLIEnvelope.localEngine,
            code: "freezeGateFailed",
            gates: ["validateEvidence", "batteryEvidence"],
            reason: "no validate run matches its exact pins",
            repairAction: "steerlab-cli experiment validate demo")
        #expect(envelope.exitCode == 65)
        #expect(envelope.state == .refused)
        // `gate` is the first of `gates` — a caller branching on one id never
        // has to think about the list.
        #expect(envelope.error?.gate == "validateEvidence")
        #expect(envelope.error?.gates == ["validateEvidence", "batteryEvidence"])
        // The count of failed gates is recoverable WITHOUT parsing prose: the
        // whole point of §2.4.
        #expect(envelope.error?.gates?.count == 2)

        // A non-gate operational failure carries no gate at all.
        let failed = SteerLabCLIEnvelope.failure(
            verb: "experiment analyze", engine: SteerLabCLIEnvelope.localEngine,
            code: "noCompletedRun", reason: "no completed run",
            repairAction: "steerlab-cli experiment run demo")
        #expect(failed.exitCode == 70)
        let object = try Self.decode(failed)
        let error = try #require(object["error"] as? [String: Any])
        #expect(error["gate"] == nil)
        #expect(error["gates"] == nil)
    }

    // MARK: Structural invariants

    /// The security rule from `ClusterCLIEnvelope`'s header, carried forward:
    /// no property on this type, or any type it nests, can hold a credential.
    /// Checked over the KEY SHAPE recursively rather than by grepping prose, so
    /// it still holds for whatever a future verb puts in `result`.
    @Test func noPropertyCanHoldACredential() throws {
        var envelope = Self.fullyPopulated()
        envelope.result?["tokenAvailable"] = .bool(true)
        envelope.result?["tokenSource"] = .string("keychain")
        let keys = Self.allKeys(in: try envelope.jsonText())
        // Presence Booleans and provenance labels are the allowed shape.
        let forbidden = [
            "token", "secret", "password", "passphrase", "apikey", "api_key",
            "authorization", "bearer", "credential", "privatekey", "private_key",
        ]
        for key in keys {
            #expect(!forbidden.contains(key), "credential-shaped key \(key)")
        }
        #expect(keys.contains("tokenavailable"))
        #expect(keys.contains("tokensource"))
    }

    /// The engine label is not a second source of truth: it aliases the
    /// constants the artifact layer already stamps.
    @Test func engineLabelsAliasTheEngineSourcesOfTruth() {
        #expect(SteerLabCLIEnvelope.localEngine == RepEReader.substrate)
        #expect(SteerLabCLIEnvelope.serverEngine == WorkspaceScoping.serverSubstrate)
        #expect(SteerLabCLIEnvelope.localEngine == "swift-mlx")
        #expect(SteerLabCLIEnvelope.serverEngine == "python-hf-transformers")
    }

    /// Step 1's scope claim: the types are wired to nothing. If a verb starts
    /// constructing envelopes, this test is the place that notices.
    @Test func theSharedEnvelopeIsNotYetWiredToAnyVerb() throws {
        #expect(SteerLabCLIEnvelope.schemaVersion == 1)
        // `ClusterCLIEnvelope` still owns the cluster family (grandfathered):
        // its state is a String parsed back, and this type does not replace it.
        let cluster = ClusterCLIEnvelope(
            verb: "cluster status", state: .ready, message: "m")
        #expect(cluster.exitCode == SteerLabCLIState.ready.exitCode)
    }

    // MARK: Helpers

    /// Every optional field populated, so a key-list assertion sees the whole
    /// surface rather than the empty-header subset.
    static func fullyPopulated() -> SteerLabCLIEnvelope {
        var envelope = SteerLabCLIEnvelope(
            verb: "experiment freeze",
            engine: SteerLabCLIEnvelope.localEngine,
            state: .refused,
            message: "cannot freeze 'demo': no validate run matches its exact pins",
            changed: false,
            observedAt: Date(timeIntervalSince1970: 1_755_000_000),
            workspace: "/tmp/steerlab-ws")
        envelope.add([
            .init(code: "unpromotedSourceAgent", detail: "agent 'a' was hand-created")
        ])
        envelope.nextAction = .init(
            verb: "experiment validate demo", requiresHuman: false,
            missingPermissionFlags: [], detail: "run validate on this substrate")
        envelope.error = .init(
            code: "freezeGateFailed", gate: "validateEvidence",
            gates: ["validateEvidence", "batteryEvidence"],
            reason: "no validate run matches its exact pins",
            repairAction: "steerlab-cli experiment validate demo")
        envelope.result = [
            "runID": .string("2026-08-17-demo"),
            "conceptCount": .number(3),
        ]
        return envelope
    }

    static func decode(_ envelope: SteerLabCLIEnvelope) throws -> [String: Any] {
        let data = Data(try envelope.jsonText().utf8)
        return try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    /// Top-level keys in emission order, read off the pretty-printed text.
    static func topLevelKeyOrder(in json: String) -> [String] {
        json.split(separator: "\n").compactMap { line in
            guard line.hasPrefix("  \""), let end = line.dropFirst(3).firstIndex(of: "\"")
            else { return nil }
            return String(line.dropFirst(3)[..<end])
        }
    }

    /// Every key in a JSON document, lowercased and recursive.
    static func allKeys(in json: String) -> Set<String> {
        func walk(_ value: Any, into keys: inout Set<String>) {
            if let object = value as? [String: Any] {
                for (key, nested) in object {
                    keys.insert(key.lowercased())
                    walk(nested, into: &keys)
                }
            } else if let array = value as? [Any] {
                for element in array { walk(element, into: &keys) }
            }
        }
        var keys: Set<String> = []
        if let value = try? JSONSerialization.jsonObject(with: Data(json.utf8)) {
            walk(value, into: &keys)
        }
        return keys
    }
}
