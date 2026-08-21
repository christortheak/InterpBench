import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit

/// The declared unit-grammar registry (USABILITY-PLAN Phase-4 item 18).
/// Server twin: `Server/tests/test_parser_registry.py` — the duration and
/// number fixtures are SHARED value-for-value across engines; the
/// byte-identical-to-legacy assertion runs against the SHIPPED default
/// registry entry on both.
@Suite(.serialized) struct ParserRegistryTests {

    func withTempRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        try ExperimentRootOverrideLock.withTempRoot(prefix: "parser", body)
    }

    /// The shipped template registry in the code repo — the exact bytes new
    /// workspaces are seeded with.
    private var shippedRegistryURL: URL {
        VectorCatalog.bundledSeedRoot.appending(
            path: ParserRegistry.registryFile)
    }

    private func sha256Hex(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private func writeShippedRegistry(into root: URL) throws -> String {
        let data = try Data(contentsOf: shippedRegistryURL)
        let destination = root.appending(path: ParserRegistry.registryFile)
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        try data.write(to: destination)
        return sha256Hex(data)
    }

    /// Cross-engine duration fixtures (identical list in the server test):
    /// every historically supported input, expected value or nil-failure.
    static let durationFixtures: [(text: String, expected: Double?)] = [
        ("I sentence the defendant to 18 months.", 18.0),
        ("A term of 2 years is appropriate.", 24.0),
        ("1.5 years of imprisonment", 18.0),
        ("una pena de 1,5 years", 18.0),
        ("a sentence of 18 to 24 months", 21.0),
        ("between 2-4 years in prison", 36.0),
        ("8 years 3 months", 99.0),
        ("8 years and 3 months", 99.0),
        ("8 years, 3 months", 99.0),
        ("8 years, and 3 months", 99.0),
        ("sentenced to 8 years 3 months in prison", 99.0),
        ("18 Monate", 18.0),
        ("2 Jahre 6 Monate", 30.0),
        ("2 Jahre und 6 Monate", 30.0),
        ("1 Jahr und 6 Monate", 18.0),
        ("eine Freiheitsstrafe von 1 Jahr", 12.0),
        ("zu 2 Jahren verurteilt", 24.0),
        ("18 monaten", 18.0),
        ("1 Monat", 1.0),
        ("eine Freiheitsstrafe von 2 jahren und 6 monaten", 30.0),
        ("3 months", 3.0),
        ("2 years", 24.0),
        // Judicial years register (sentencing-anchoring run, 2026-08-10):
        // number words one–twelve, markdown bold, curly apostrophes — exact
        // run phrasings.
        ("I sentence the defendant, A, to **ten years and six months’** imprisonment", 126.0),
        ("I hereby sentence the defendant, A, to **seven years and six months’ imprisonment**", 90.0),
        ("Therefore, I sentence A to **six years and six months** imprisonment", 78.0),
        ("a term of ten years", 120.0),
        ("Ten Years And Six Months", 126.0),
        ("ten years and 6 months", 126.0),
        ("10 years and six months", 126.0),
        // Ranges stay digit-only: no range match, so the single-term rule
        // reads the first word-number followed by a unit.
        ("ten to twelve years", 144.0),
        ("thirteen years", nil),
        ("the defendant was sentenced", nil),
        ("brighten years of effort", nil),
        ("The defendant is guilty.", nil),
        ("Der Angeklagte ist schuldig.", nil),
        ("a lengthy custodial term", nil),
        ("", nil),
    ]

    // MARK: - Byte-identical legacy reproduction (mandatory)

    @Test func shippedDefaultMatchesLegacyParserOnEveryFixture() throws {
        let specs = try ParserRegistry.loadSpecs(at: shippedRegistryURL)
        let spec = try #require(specs["sentencing-months"])
        let parse = try ParserRegistry.buildParser(
            name: "sentencing-months", spec: spec)
        for fixture in Self.durationFixtures {
            let registry = parse(fixture.text)
            let legacy = Judicial.parseMonths(fixture.text)
            #expect(
                registry == legacy,
                "registry vs legacy diverge on '\(fixture.text)': \(String(describing: registry)) vs \(String(describing: legacy))"
            )
            #expect(
                registry == fixture.expected,
                "unexpected value on '\(fixture.text)': \(String(describing: registry))"
            )
        }
    }

    // MARK: - durationMonths declared policies

    @Test func durationRangePoliciesAreData() throws {
        var spec = ParserRegistry.Spec()
        spec.kind = "durationMonths"
        spec.units = ["years": 12, "months": 1]
        spec.range = "refuse"
        let refuse = try ParserRegistry.buildParser(name: "d", spec: spec)
        #expect(refuse("18 to 24 months") == nil)
        #expect(refuse("18 months") == 18.0)
        spec.range = "first"
        let first = try ParserRegistry.buildParser(name: "d", spec: spec)
        #expect(first("18 to 24 months") == 18.0)
    }

    @Test func durationCustomUnitsExtendTheGrammar() throws {
        var spec = ParserRegistry.Spec()
        spec.kind = "durationMonths"
        spec.units = ["ans": 12, "an": 12, "mois": 1]
        spec.joiners = ["et"]
        let parse = try ParserRegistry.buildParser(name: "fr", spec: spec)
        #expect(parse("une peine de 2 ans et 3 mois") == 27.0)
        #expect(parse("1 an") == 12.0)
        #expect(parse("6 mois") == 6.0)
        #expect(parse("aucune peine") == nil)
    }

    // MARK: - number kind (cross-engine fixtures)

    private func numberSpec(
        range: String? = nil, percent: String? = nil
    ) -> ParserRegistry.Spec {
        var spec = ParserRegistry.Spec()
        spec.kind = "number"
        spec.range = range
        spec.percent = percent
        spec.decimalComma = true
        return spec
    }

    @Test func numberKindFixtures() throws {
        let plain = try ParserRegistry.buildParser(name: "n", spec: numberSpec())
        #expect(plain("score: 7.5 overall") == 7.5)
        #expect(plain("6,5 points") == 6.5)
        #expect(plain("about 42% of cases") == 42.0)
        #expect(plain("no digits here") == nil)
        #expect(plain("") == nil)
        // Range default is refuse — "5-7" is a counted parse failure.
        #expect(plain("somewhere in the 5-7 band") == nil)
        #expect(plain("5 to 7") == nil)
        // A number BEFORE the range wins (first-number semantics).
        #expect(plain("rated 4 of 5-7") == 4.0)

        let mean = try ParserRegistry.buildParser(
            name: "n", spec: numberSpec(range: "mean"))
        #expect(mean("somewhere in the 5-7 band") == 6.0)
        let first = try ParserRegistry.buildParser(
            name: "n", spec: numberSpec(range: "first"))
        #expect(first("somewhere in the 5-7 band") == 5.0)

        let fraction = try ParserRegistry.buildParser(
            name: "n", spec: numberSpec(percent: "fraction"))
        #expect(fraction("about 42% of cases") == 0.42)
        #expect(fraction("about 42 % of cases") == 0.42)
        let refuse = try ParserRegistry.buildParser(
            name: "n", spec: numberSpec(percent: "refuse"))
        #expect(refuse("about 42% of cases") == nil)
        #expect(refuse("about 42 cases") == 42.0)
    }

    // MARK: - Registry shape validation (plain-language errors)

    @Test func malformedSpecsRefuseWithPlainMessages() throws {
        var noKind = ParserRegistry.Spec()
        noKind.units = ["years": 12]
        #expect(throws: ExperimentError.self) {
            try ParserRegistry.validate(noKind, name: "x")
        }
        var badKind = ParserRegistry.Spec()
        badKind.kind = "currency"
        do {
            try ParserRegistry.validate(badKind, name: "x")
            Issue.record("unknown kind must refuse")
        } catch {
            let reason = try #require((error as? ExperimentError)?.reason)
            #expect(reason.contains("known kinds"))
        }
        var noUnits = ParserRegistry.Spec()
        noUnits.kind = "durationMonths"
        do {
            try ParserRegistry.validate(noUnits, name: "x")
            Issue.record("missing units must refuse")
        } catch {
            let reason = try #require((error as? ExperimentError)?.reason)
            #expect(reason.contains("units"))
        }
        var badMultiplier = ParserRegistry.Spec()
        badMultiplier.kind = "durationMonths"
        badMultiplier.units = ["years": 0]
        #expect(throws: ExperimentError.self) {
            try ParserRegistry.validate(badMultiplier, name: "x")
        }
        var badRange = ParserRegistry.Spec()
        badRange.kind = "number"
        badRange.range = "average"
        #expect(throws: ExperimentError.self) {
            try ParserRegistry.validate(badRange, name: "x")
        }
        var badPercent = ParserRegistry.Spec()
        badPercent.kind = "number"
        badPercent.percent = "strip"
        #expect(throws: ExperimentError.self) {
            try ParserRegistry.validate(badPercent, name: "x")
        }
    }

    // MARK: - Manifest surface: round trip + verify + freeze pin

    @Test func manifestKeysRoundTripAndLegacyBytesAreUnchanged() throws {
        var manifest = ExperimentManifest(
            name: "p", description: "", modelID: "test/model")
        let legacyHash = ExperimentStore.manifestHash(manifest)
        manifest.numericParser = "sentencing-months"
        manifest.parserRegistryHash = String(repeating: "ab", count: 32)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(manifest)
        let json = String(decoding: data, as: UTF8.self)
        #expect(json.contains("\"numericParser\""))
        #expect(json.contains("\"parserRegistryHash\""))
        let decoded = try JSONDecoder().decode(ExperimentManifest.self, from: data)
        #expect(decoded.numericParser == "sentencing-months")
        #expect(decoded.parserRegistryHash == String(repeating: "ab", count: 32))
        // A legacy manifest (no parser keys) keeps its content hash — the
        // keys are omitted, never encoded as null.
        var legacy = manifest
        legacy.numericParser = nil
        legacy.parserRegistryHash = nil
        #expect(ExperimentStore.manifestHash(legacy) == legacyHash)
        let legacyJSON = String(decoding: try encoder.encode(legacy), as: UTF8.self)
        #expect(!legacyJSON.contains("numericParser"))
        #expect(!legacyJSON.contains("parserRegistryHash"))
    }

    @Test func verifyChecksDeclaredParserAndRegistryDrift() throws {
        try withTempRoot { root in
            var manifest = try ExperimentStore.create(
                name: "pv", description: "", modelID: "test/model",
                modelRevision: "abc")
            // Legacy manifest: no parser named, no registry present — no
            // violations mentioning parsers.
            #expect(
                !ExperimentStore.verify(manifest).contains {
                    $0.contains("parser")
                })
            // Declared parser, no registry on disk: a violation naming the
            // file and the remedy.
            manifest.numericParser = "sentencing-months"
            let missing = ExperimentStore.verify(manifest)
            #expect(
                missing.contains {
                    $0.contains("no parser registry exists")
                        && $0.contains(ParserRegistry.registryFile)
                })
            // Registry present: verify passes.
            let registryHash = try writeShippedRegistry(into: root)
            #expect(
                !ExperimentStore.verify(manifest).contains {
                    $0.contains("parser")
                })
            // Unknown parser name: names the defined entries.
            var unknown = manifest
            unknown.numericParser = "no-such-parser"
            #expect(
                ExperimentStore.verify(unknown).contains {
                    $0.contains("no parser named 'no-such-parser'")
                })
            // Pinned hash + drifted file = violation.
            manifest.parserRegistryHash = registryHash
            #expect(
                !ExperimentStore.verify(manifest).contains {
                    $0.contains("parser")
                })
            let registryURL = root.appending(path: ParserRegistry.registryFile)
            var payload = try JSONSerialization.jsonObject(
                with: Data(contentsOf: registryURL)) as! [String: Any]
            payload["extra"] = "drift"
            try JSONSerialization.data(
                withJSONObject: payload, options: [.sortedKeys]
            ).write(to: registryURL)
            #expect(
                ExperimentStore.verify(manifest).contains {
                    $0.contains("parser registry changed since pinning")
                })
            // A hash with no declared parser is an unusable pin.
            var hashOnly = manifest
            hashOnly.numericParser = nil
            #expect(
                ExperimentStore.verify(hashOnly).contains {
                    $0.contains("no numericParser is declared")
                })
        }
    }

    @Test func freezePinsRegistryHashForDeclaredParser() throws {
        try withTempRoot { root in
            let registryHash = try writeShippedRegistry(into: root)
            var manifest = try ExperimentStore.create(
                name: "pf", description: "", modelID: "test/model",
                modelRevision: "abc")
            // A variant-only study freezes without validate evidence under
            // --force (the FreezeFirewallClosure fixture pattern); the pin
            // itself must land regardless of forced gates.
            let artifact = ModelVariantArtifact(
                name: "agent", baseModelID: "test/model",
                adapters: [], injections: [],
                promptMode: "chatAssistant", qwenThinkingEnabled: false,
                temperature: 0, systemPrompt: "")
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.sortedKeys]
            let artifactData = try encoder.encode(artifact)
            let artifactURL = root.appending(
                path: "runs/model-variants/agent/model-variant.json")
            try FileManager.default.createDirectory(
                at: artifactURL.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try artifactData.write(to: artifactURL)
            manifest.variantConditions = [
                .init(
                    name: "agent",
                    artifactPath: "runs/model-variants/agent/model-variant.json",
                    artifactHash: sha256Hex(artifactData), artifact: artifact)
            ]
            manifest.numericParser = "sentencing-months"
            try ExperimentStore.save(manifest)
            let frozen = try ExperimentStore.freeze(name: "pf", force: true)
            #expect(frozen.parserRegistryHash == registryHash)
            #expect(frozen.numericParser == "sentencing-months")
        }
    }

    @Test func resolveNumericParserHonorsLegacyAbsenceAndParses() throws {
        try withTempRoot { root in
            var manifest = try ExperimentStore.create(
                name: "pr", description: "", modelID: "test/model",
                modelRevision: "abc")
            // No parser named: nil — the legacy caseFamily path.
            #expect(try ParserRegistry.resolveNumericParser(manifest) == nil)
            _ = try writeShippedRegistry(into: root)
            manifest.numericParser = "sentencing-months"
            let resolved = try #require(
                try ParserRegistry.resolveNumericParser(manifest))
            #expect(resolved.parse("8 years and 3 months") == 99.0)
            #expect(resolved.kind == "durationMonths")
            #expect(resolved.provenance.registryFile == ParserRegistry.registryFile)
            #expect(resolved.provenance.registryHash == resolved.registryHash)
            // Drifted pin refuses at resolve (run start), in plain words.
            manifest.parserRegistryHash = String(repeating: "00", count: 32)
            do {
                _ = try ParserRegistry.resolveNumericParser(manifest)
                Issue.record("drifted registry pin must refuse")
            } catch {
                let reason = try #require((error as? ExperimentError)?.reason)
                #expect(reason.contains("drifted"))
            }
        }
    }

    /// The record-level dispatch: a declared parser wins; without one the
    /// historical caseFamily rule is byte-identical to before.
    @Test func judicialParsesDispatchesDeclaredParserOverCaseFamily() throws {
        let specs = try ParserRegistry.loadSpecs(at: shippedRegistryURL)
        let spec = try #require(specs["plain-number"])
        let resolved = ParserRegistry.ResolvedNumericParser(
            name: "plain-number", kind: "number",
            registryHash: "00",
            parseFunction: try ParserRegistry.buildParser(
                name: "plain-number", spec: spec))
        // Declared parser applies regardless of case family.
        let declared = ExperimentTasks.judicialParses(
            output: "I assign a score of 7.", options: nil,
            caseFamily: nil, numericParser: resolved)
        #expect(declared.parsedMonths == .some(.some(7.0)))
        // Legacy: sentencing keys the built-in months parser…
        let sentencing = ExperimentTasks.judicialParses(
            output: "8 years 3 months", options: nil, caseFamily: "sentencing")
        #expect(sentencing.parsedMonths == .some(.some(99.0)))
        // …and other families stamp nothing.
        let none = ExperimentTasks.judicialParses(
            output: "8 years 3 months", options: nil, caseFamily: nil)
        #expect(none.parsedMonths == nil)
    }
}
