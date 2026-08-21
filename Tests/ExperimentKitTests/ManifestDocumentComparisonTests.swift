import Foundation
import Testing

@testable import ExperimentKit

/// The remote-freeze identity check's document comparison: two manifest
/// documents, canonicalized by THIS engine with the same volatile-key list
/// the frozenBy:"server" freeze-canonical verification uses. Pure helpers —
/// no IO except the missing-local read, which uses the store's temp seam.
struct ManifestDocumentComparisonTests {

    private func data(_ json: String) -> Data { Data(json.utf8) }

    // MARK: Equality

    @Test func identicalDocumentsAreEqual() {
        let body = data(#"{"name":"s","modelID":"org/m","conditions":[]}"#)
        #expect(
            ExperimentStore.compareManifestDocuments(local: body, server: body)
                == .equal)
    }

    @Test func keyOrderAndWhitespaceDoNotMatter() {
        let local = data(#"{"name":"s","modelID":"org/m"}"#)
        let server = data("{\n  \"modelID\": \"org/m\",\n  \"name\": \"s\"\n}")
        #expect(
            ExperimentStore.compareManifestDocuments(local: local, server: server)
                == .equal)
    }

    @Test func volatileFreezeStampsAloneDoNotMismatch() {
        // The exact volatile-key list the freeze-canonical verification
        // strips: lifecycle stamps must never make the same design read as
        // a different study.
        let local = data(#"{"name":"s","modelID":"org/m","status":"draft"}"#)
        let server = data(
            #"{"name":"s","modelID":"org/m","status":"frozen","frozenAt":"2026-07-13","#
                + #""freezeHash":"abc","gitCommit":"deadbeef","frozenBy":"server","#
                + #""createdAt":"2026-07-01","appVersion":"1.2","freezeForced":true,"#
                + #""forcedGatesSkipped":["revision"]}"#)
        #expect(
            ExperimentStore.compareManifestDocuments(local: local, server: server)
                == .equal)
    }

    @Test func explicitNullEqualsOmittedKey() {
        // Python writes null where Swift omits the key — semantically the
        // same pin (the freeze-canonical rule, reused here).
        let local = data(#"{"name":"s","modelID":"org/m"}"#)
        let server = data(#"{"name":"s","modelID":"org/m","taskPromptsFile":null}"#)
        #expect(
            ExperimentStore.compareManifestDocuments(local: local, server: server)
                == .equal)
    }

    // MARK: Differences (field-level summary)

    @Test func differingScalarNamesTheKey() {
        let local = data(#"{"name":"s","modelID":"org/m"}"#)
        let server = data(#"{"name":"s","modelID":"org/OTHER"}"#)
        guard
            case .different(let fields) = ExperimentStore.compareManifestDocuments(
                local: local, server: server)
        else {
            Issue.record("expected .different")
            return
        }
        #expect(fields == ["modelID: differs"])
    }

    @Test func differingArraysCarryElementCounts() {
        let local = data(#"{"name":"s","conditions":[{"name":"a"},{"name":"b"},{"name":"c"}]}"#)
        let server = data(#"{"name":"s","conditions":[{"name":"a"}]}"#)
        guard
            case .different(let fields) = ExperimentStore.compareManifestDocuments(
                local: local, server: server)
        else {
            Issue.record("expected .different")
            return
        }
        #expect(fields == ["conditions: differs (local 3, server 1)"])
    }

    @Test func oneSidedKeysNameTheSide() {
        let local = data(#"{"name":"s","taskPromptsHash":"abc"}"#)
        let server = data(#"{"name":"s","concepts":[{"name":"fear"}]}"#)
        guard
            case .different(let fields) = ExperimentStore.compareManifestDocuments(
                local: local, server: server)
        else {
            Issue.record("expected .different")
            return
        }
        #expect(fields.contains("taskPromptsHash: local only"))
        #expect(fields.contains("concepts: server only (1 item)"))
        #expect(fields.count == 2)
    }

    @Test func nestedDifferenceSurfacesAsTheTopLevelKey() {
        // Field-level means TOP-LEVEL keys: a deep difference names its
        // top-level container (with counts when array-valued).
        let local = data(#"{"name":"s","conditions":[{"name":"a","slots":[]}]}"#)
        let server = data(
            #"{"name":"s","conditions":[{"name":"a","slots":[{"concept":"fear","layer":17,"alpha":2}]}]}"#)
        guard
            case .different(let fields) = ExperimentStore.compareManifestDocuments(
                local: local, server: server)
        else {
            Issue.record("expected .different")
            return
        }
        #expect(fields == ["conditions: differs (local 1, server 1)"])
    }

    @Test func nonObjectDocumentsAreUnparseable() {
        #expect(
            ExperimentStore.compareManifestDocuments(
                local: data("[1,2,3]"), server: data(#"{"name":"s"}"#))
                == .unparseable)
        #expect(
            ExperimentStore.compareManifestDocuments(
                local: data(#"{"name":"s"}"#), server: data("not json"))
                == .unparseable)
    }

    // MARK: Canonical body hash (the server-only-copy fingerprint)

    @Test func canonicalBodyHashIgnoresVolatileStampsAndFormatting() {
        let draft = data(#"{"name":"s","modelID":"org/m","status":"draft"}"#)
        let frozen = data(
            "{\n \"modelID\": \"org/m\", \"name\": \"s\", \"status\": \"frozen\","
                + " \"freezeHash\": \"abc\", \"taskPromptsFile\": null }")
        let a = ExperimentStore.canonicalManifestBodyHash(draft)
        let b = ExperimentStore.canonicalManifestBodyHash(frozen)
        #expect(a != nil)
        #expect(a == b)
        // 64 hex chars — a real SHA-256, displayable truncated.
        #expect(a?.count == 64)
        // A design change changes the fingerprint.
        let other = ExperimentStore.canonicalManifestBodyHash(
            data(#"{"name":"s","modelID":"org/OTHER"}"#))
        #expect(other != a)
    }

    @Test func canonicalBodyHashOfNonObjectIsNil() {
        #expect(ExperimentStore.canonicalManifestBodyHash(data("[]")) == nil)
    }

    // MARK: Missing local document

    @Test func manifestDataIsNilForUnknownStudy() {
        // No root override needed: the name cannot exist under any root, and
        // touching the process-global seam from a parallel suite would race
        // the serialized store tests.
        #expect(
            ExperimentStore.manifestData(name: "no-such-study-\(UUID().uuidString)")
                == nil)
    }
}
