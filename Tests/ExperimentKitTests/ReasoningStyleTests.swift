import CryptoKit
import Foundation
import Testing

@testable import ExperimentKit
@testable import SteeringKit

/// Reasoning-style instrument (the third leg of holdings/severity/
/// reasoning-style): taxonomy-as-pinned-data, deterministic per-generation
/// scoring, rs_<featureID> metrics columns + report blocks + effect-size
/// metrics, and the epoch-guarded post-hoc `rescore-style` verb. Declared as
/// an extension of the serialized `ExperimentStoreTests` suite because the
/// pin/rescore tests share its `rootOverride` test seam (a process-global).
/// The scoring math is fixture-tested byte-identically with the Python
/// engine (`Fixtures/reasoning-style/reasoning-style-parity.json`).
extension ExperimentStoreTests {

    private static var reasoningStyleFixtureURL: URL {
        URL(filePath: #filePath)
            .deletingLastPathComponent()
            .appending(
                components: "Fixtures", "reasoning-style",
                "reasoning-style-parity.json")
    }

    private static let taxonomyJSON = """
        {
          "schemaVersion": 1,
          "name": "test-style-v1",
          "features": [
            {"id": "hedge", "title": "Hedging", "kind": "wordList",
             "patterns": ["might", "perhaps"], "normalize": "per1kWords"},
            {"id": "question", "title": "Questions", "kind": "regex",
             "patterns": ["\\\\?"], "normalize": "perSentence"}
          ]
        }
        """

    private func feature(
        id: String = "f", kind: String = "wordList",
        patterns: [String] = ["word"], normalize: String = "rawCount"
    ) -> String {
        let patternList = patterns.map { "\"\($0)\"" }.joined(separator: ", ")
        return """
            {"schemaVersion": 1, "name": "t", "features": [
              {"id": "\(id)", "title": "F", "kind": "\(kind)",
               "patterns": [\(patternList)], "normalize": "\(normalize)"}]}
            """
    }

    private func loadTaxonomy(_ json: String) throws -> ReasoningStyleTaxonomy {
        try ReasoningStyleTaxonomy.load(data: Data(json.utf8))
    }

    // MARK: - Load + validation

    @Test func taxonomyLoadPreservesDeclaredFeatureOrder() throws {
        let taxonomy = try loadTaxonomy(Self.taxonomyJSON)
        #expect(taxonomy.name == "test-style-v1")
        #expect(taxonomy.featureIDs == ["hedge", "question"])
    }

    @Test func taxonomyLoadRejectsBadSchema() throws {
        #expect(throws: ExperimentError.self) {
            try self.loadTaxonomy(#"{"schemaVersion": 2, "name": "t", "features": []}"#)
        }
        #expect(throws: ExperimentError.self) {  // no name
            try self.loadTaxonomy(
                #"{"schemaVersion": 1, "name": "", "features": [{"id": "f"}]}"#)
        }
        #expect(throws: ExperimentError.self) {  // empty features
            try self.loadTaxonomy(#"{"schemaVersion": 1, "name": "t", "features": []}"#)
        }
        #expect(throws: ExperimentError.self) {
            try self.loadTaxonomy("not json")
        }
    }

    @Test func taxonomyLoadRejectsUnknownKindNormalizeAndBadIDs() throws {
        #expect(throws: ExperimentError.self) {
            try self.loadTaxonomy(self.feature(kind: "fancy"))
        }
        #expect(throws: ExperimentError.self) {
            try self.loadTaxonomy(self.feature(normalize: "perParagraph"))
        }
        #expect(throws: ExperimentError.self) {
            try self.loadTaxonomy(self.feature(id: "bad id"))
        }
        // Duplicate ids.
        #expect(throws: ExperimentError.self) {
            try self.loadTaxonomy("""
                {"schemaVersion": 1, "name": "t", "features": [
                  {"id": "f", "kind": "wordList", "patterns": ["a"], "normalize": "rawCount"},
                  {"id": "f", "kind": "wordList", "patterns": ["b"], "normalize": "rawCount"}]}
                """)
        }
    }

    @Test func taxonomyLoadRejectsBadPatterns() throws {
        #expect(throws: ExperimentError.self) {  // empty pattern list
            try self.loadTaxonomy(self.feature(patterns: []))
        }
        #expect(throws: ExperimentError.self) {  // unmatchable word list
            try self.loadTaxonomy(self.feature(patterns: ["!!!"]))
        }
        #expect(throws: ExperimentError.self) {  // caught by the GRAMMAR now
            try self.loadTaxonomy(self.feature(kind: "regex", patterns: ["(unclosed"]))
        }
        // Lookbehind is out of the portable subset even where ICU accepts it.
        #expect(throws: ExperimentError.self) {
            try self.loadTaxonomy(self.feature(kind: "regex", patterns: ["(?<=a)b"]))
        }
        #expect(throws: ExperimentError.self) {
            try self.loadTaxonomy(self.feature(kind: "regex", patterns: ["(?<!a)b"]))
        }
        // …and so are constructs ICU would happily compile.
        #expect(throws: ExperimentError.self) {  // ICU-only named group
            try self.loadTaxonomy(self.feature(kind: "regex", patterns: ["(?<n>a)"]))
        }
        #expect(throws: ExperimentError.self) {  // lookahead (all four rejected)
            try self.loadTaxonomy(self.feature(kind: "regex", patterns: ["a(?=b)"]))
        }
        #expect(throws: ExperimentError.self) {  // \R linebreak (ICU-only)
            try self.loadTaxonomy(self.feature(kind: "regex", patterns: ["a\\R"]))
        }
        #expect(throws: ExperimentError.self) {  // capturing group
            try self.loadTaxonomy(self.feature(kind: "regex", patterns: ["(ab)+"]))
        }
    }

    @Test func portableRegexVectorsAgreeWithTheParser() throws {
        // THE shared cross-engine list: every vector runs verbatim in the
        // Python suite too (test_portable_regex_vectors_agree_with_the_parser).
        struct Vectors: Decodable {
            struct Rejected: Decodable {
                let pattern: String
                let construct: String
            }
            let accepted: [String]
            let rejected: [Rejected]
        }
        let url = Self.reasoningStyleFixtureURL
            .deletingLastPathComponent()
            .appending(component: "portable-regex-vectors.json")
        let vectors = try JSONDecoder().decode(
            Vectors.self, from: Data(contentsOf: url))
        for pattern in vectors.accepted {
            let violation = PortableRegexParser.violation(in: pattern)
            #expect(violation == nil, "\(pattern): \(violation ?? "")")
            // Accepted patterns must also compile under the pinned options.
            #expect(
                (try? NSRegularExpression(
                    pattern: pattern,
                    options: ReasoningStyleTaxonomy.regexOptions)) != nil,
                "\(pattern) does not compile in ICU")
        }
        for entry in vectors.rejected {
            let violation = PortableRegexParser.violation(in: entry.pattern)
            #expect(
                violation?.contains(entry.construct) == true
                    && violation?.contains("at position") == true,
                "\(entry.pattern): got \(violation ?? "accepted"), expected \(entry.construct)")
        }
    }

    // MARK: - Scoring math

    @Test func tokenizerSentencesAndWordDenominators() {
        #expect(
            ReasoningStyleTaxonomy.matchTokens("It's A-B 3rd.")
                == ["it", "s", "a", "b", "3rd"])
        #expect(ReasoningStyleTaxonomy.sentenceCount("One. Two! Three?") == 3)
        #expect(ReasoningStyleTaxonomy.sentenceCount("no terminator") == 1)
        #expect(ReasoningStyleTaxonomy.sentenceCount("") == 1)
        // A mid-token '.' is not a boundary.
        #expect(ReasoningStyleTaxonomy.sentenceCount("v1.2 is out.") == 1)
        #expect(ReasoningStyleTaxonomy.whitespaceWordCount("a  b\tc\nd") == 4)
        #expect(ReasoningStyleTaxonomy.whitespaceWordCount("") == 1)
    }

    private static let decomposed = "de\u{0301}cide\u{0301}"  // décidé, combining acutes
    private static let precomposed = "d\u{00E9}cid\u{00E9}"

    @Test func tokenizerUnicodeRulesMatchThePythonEngine() {
        // NFC first: decomposed and precomposed accents yield the SAME token.
        #expect(ReasoningStyleTaxonomy.matchTokens(Self.decomposed) == [Self.precomposed])
        #expect(ReasoningStyleTaxonomy.matchTokens(Self.precomposed) == [Self.precomposed])
        // Per-scalar unconditional lowercasing: ΛΟΓΟΣ -> λογοσ (NEVER final
        // ς — CPython's whole-string lower() would emit it; String
        // .lowercased() would not — neither whole-string API is the rule).
        #expect(
            ReasoningStyleTaxonomy.matchTokens("\u{039B}\u{039F}\u{0393}\u{039F}\u{03A3}")
                == ["\u{03BB}\u{03BF}\u{03B3}\u{03BF}\u{03C3}"])
        // İ lowercases to i + COMBINING DOT ABOVE; the combining mark (Mn)
        // is not a token scalar, so it SPLITS the run — same on both engines.
        #expect(ReasoningStyleTaxonomy.matchTokens("\u{0130}stanbul") == ["i", "stanbul"])
        // Category rule Nd-only: superscript two is No, not a digit token
        // char (Character.isNumber would have taken it; so would Python's
        // isdigit()).
        #expect(ReasoningStyleTaxonomy.matchTokens("x\u{00B2} y") == ["x", "y"])
    }

    @Test func scoringNormalizesNFCInTextAndPatterns() throws {
        let decomposedText = "Ils ont \(Self.decomposed)."
        let precomposedText = "Ils ont \(Self.precomposed)."
        // Precomposed wordList pattern matches decomposed text…
        var taxonomy = try loadTaxonomy(feature(patterns: [Self.precomposed]))
        #expect(taxonomy.score(decomposedText)["f"] == 1.0)
        // …a DECOMPOSED pattern matches precomposed text…
        taxonomy = try loadTaxonomy(feature(patterns: [Self.decomposed]))
        #expect(taxonomy.score(precomposedText)["f"] == 1.0)
        // …and the same holds for regex \b around the non-ASCII word.
        taxonomy = try loadTaxonomy(
            feature(kind: "regex", patterns: ["\\\\b\(Self.precomposed)\\\\b"]))
        #expect(taxonomy.score(decomposedText)["f"] == 1.0)
    }

    @Test func pinnedRegexSemantics() throws {
        // '.' never matches \n but DOES match \r — .useUnixLineSeparators
        // aligns ICU with Python's default (without it ICU also excludes \r).
        var taxonomy = try loadTaxonomy(feature(kind: "regex", patterns: ["a.b"]))
        #expect(taxonomy.score("a\nb a\rb axb")["f"] == 2.0)
        // '^'/'$' anchor to the whole text; '$' also matches before ONE
        // final \n (both engines).
        taxonomy = try loadTaxonomy(feature(kind: "regex", patterns: ["end$"]))
        #expect(taxonomy.score("the end\nthe end\n")["f"] == 1.0)
        taxonomy = try loadTaxonomy(feature(kind: "regex", patterns: ["^the"]))
        #expect(taxonomy.score("the end\nthe end\n")["f"] == 1.0)
    }

    @Test func wordListMatchesWholeWordsAndPhrases() throws {
        let taxonomy = try loadTaxonomy(
            feature(patterns: ["might", "on the other hand"]))
        // "Mighty" must NOT match "might"; the phrase matches across punctuation.
        #expect(
            taxonomy.score("Mighty things might happen; on the other hand, not.")["f"]
                == 2.0)
        #expect(taxonomy.score("mighty mightier almighty")["f"] == 0.0)
    }

    @Test func normalizeModesAndEmptyTextEdgeCases() throws {
        let text = "Yes sir. No sir. yes YES?"
        // 3 matches; 3 sentences; 6 whitespace words.
        let perSentence = try loadTaxonomy(
            feature(patterns: ["yes"], normalize: "perSentence"))
        let per1k = try loadTaxonomy(
            feature(patterns: ["yes"], normalize: "per1kWords"))
        let raw = try loadTaxonomy(feature(patterns: ["yes"]))
        #expect(perSentence.score(text)["f"] == 1.0)
        #expect(per1k.score(text)["f"] == 3 * 1000.0 / 6)
        #expect(raw.score(text)["f"] == 3.0)
        for taxonomy in [perSentence, per1k, raw] {
            #expect(taxonomy.score("")["f"] == 0.0)
        }
    }

    @Test func regexCountsAreCaseInsensitiveNonOverlapping() throws {
        let taxonomy = try loadTaxonomy(feature(kind: "regex", patterns: ["ab"]))
        #expect(taxonomy.score("AB abab xx")["f"] == 3.0)
    }

    // MARK: - Cross-engine fixture parity

    @Test func scoringMatchesCrossEngineFixtureExactly() throws {
        struct Fixture: Decodable {
            struct Case: Decodable {
                let text: String
                let expected: [String: Double]
            }
            struct RawFeature: Decodable {
                let id: String
                let normalize: String
            }
            struct RawTaxonomy: Decodable {
                let features: [RawFeature]
            }
            let taxonomy: RawTaxonomy
            let cases: [Case]
        }
        let data = try Data(contentsOf: Self.reasoningStyleFixtureURL)
        let fixture = try JSONDecoder().decode(Fixture.self, from: data)
        // The taxonomy is loaded through the REAL loader from the same bytes.
        let object = try #require(
            try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let taxonomyData = try JSONSerialization.data(
            withJSONObject: try #require(object["taxonomy"]))
        let taxonomy = try ReasoningStyleTaxonomy.load(data: taxonomyData)
        let normalizeByID = Dictionary(
            uniqueKeysWithValues: fixture.taxonomy.features.map { ($0.id, $0.normalize) })
        for testCase in fixture.cases {
            let got = taxonomy.score(testCase.text)
            for (id, expected) in testCase.expected {
                let value = try #require(got[id], "missing \(id)")
                if normalizeByID[id] == "rawCount" {
                    #expect(value == expected, "\(id) on \(testCase.text)")
                } else {
                    #expect(
                        abs(value - expected) <= 1e-9,
                        "\(id) on \(testCase.text): \(value) vs \(expected)")
                }
            }
        }
    }

    // MARK: - Manifest pin: set / verify / drift / absent

    private func withStyleTempRoot<T>(_ body: (URL) throws -> T) rethrows -> T {
        // Shared cross-suite lock for the process-global rootOverride.
        try ExperimentRootOverrideLock.withTempRoot(prefix: "style", body)
    }

    @discardableResult
    private func plantTaxonomy(
        at root: URL, json: String = taxonomyJSON,
        name: String = "style.json"
    ) throws -> String {
        let directory = root.appending(components: "prompts", "taxonomies")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        try json.write(
            to: directory.appending(component: name),
            atomically: true, encoding: .utf8)
        return "prompts/taxonomies/\(name)"
    }

    private func taxonomyViolations(_ manifest: ExperimentManifest) -> [String] {
        ExperimentStore.verify(manifest).filter { $0.lowercased().contains("taxonomy") }
    }

    @Test func pinStampsPathAndHashAndVerifyCatchesDriftAndDisappearance() throws {
        try withStyleTempRoot { root in
            let path = try plantTaxonomy(at: root)
            _ = try ExperimentStore.create(
                name: "style-pin", description: "", modelID: "test/model")
            var manifest = try ExperimentStore.pinReasoningStyleTaxonomy(
                experimentName: "style-pin", path: path)
            let bytes = try Data(contentsOf: root.appending(path: path))
            let expected = SHA256.hash(data: bytes)
                .map { String(format: "%02x", $0) }.joined()
            #expect(manifest.reasoningStyleTaxonomyPath == path)
            #expect(manifest.reasoningStyleTaxonomyHash == expected)
            #expect(taxonomyViolations(manifest).isEmpty)

            // Drift after pinning is a violation…
            try Self.taxonomyJSON.replacingOccurrences(of: "test-style-v1", with: "edited")
                .write(
                    to: root.appending(path: path), atomically: true, encoding: .utf8)
            #expect(
                taxonomyViolations(manifest).contains {
                    $0.contains("changed since pinning")
                })
            // …and so is disappearance.
            try FileManager.default.removeItem(at: root.appending(path: path))
            #expect(taxonomyViolations(manifest).contains { $0.contains("missing") })

            // Half-pins certify nothing, in both directions.
            manifest.reasoningStyleTaxonomyHash = nil
            #expect(taxonomyViolations(manifest).contains { $0.contains("incomplete") })
            manifest.reasoningStyleTaxonomyPath = nil
            manifest.reasoningStyleTaxonomyHash = expected
            #expect(
                taxonomyViolations(manifest).contains { $0.contains("without a path") })

            // Absent pin = no reasoning-style scoring, no violation.
            manifest.reasoningStyleTaxonomyHash = nil
            #expect(taxonomyViolations(manifest).isEmpty)
        }
    }

    @Test func pinRefusesMissingOrInvalidTaxonomy() throws {
        try withStyleTempRoot { root in
            _ = try ExperimentStore.create(
                name: "style-bad", description: "", modelID: "test/model")
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.pinReasoningStyleTaxonomy(
                    experimentName: "style-bad", path: "prompts/taxonomies/nope.json")
            }
            let bad = try plantTaxonomy(
                at: root, json: feature(kind: "regex", patterns: ["(?<=a)b"]),
                name: "bad.json")
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.pinReasoningStyleTaxonomy(
                    experimentName: "style-bad", path: bad)
            }
        }
    }

    @Test func loadPinnedEnforcesHashAndCompleteness() throws {
        try withStyleTempRoot { root in
            let path = try plantTaxonomy(at: root)
            _ = try ExperimentStore.create(
                name: "style-load", description: "", modelID: "test/model")
            let manifest = try ExperimentStore.pinReasoningStyleTaxonomy(
                experimentName: "style-load", path: path)
            let pinned = try #require(
                try ExperimentStore.loadPinnedReasoningStyle(manifest))
            #expect(pinned.taxonomy.featureIDs == ["hedge", "question"])
            #expect(pinned.hash == manifest.reasoningStyleTaxonomyHash)

            // Drift refuses at load — scoring never reads what verify rejects.
            try Self.taxonomyJSON.replacingOccurrences(of: "v1", with: "v2").write(
                to: root.appending(path: path), atomically: true, encoding: .utf8)
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.loadPinnedReasoningStyle(manifest)
            }

            // Unpinned = nil (no scoring); half-pin = error.
            var bare = manifest
            bare.reasoningStyleTaxonomyPath = nil
            bare.reasoningStyleTaxonomyHash = nil
            #expect(try ExperimentStore.loadPinnedReasoningStyle(bare) == nil)
            bare.reasoningStyleTaxonomyPath = path
            #expect(throws: ExperimentError.self) {
                try ExperimentStore.loadPinnedReasoningStyle(bare)
            }
        }
    }

    // MARK: - metrics.csv / report.json / effect-size integration

    private func styleRow(
        condition: String, promptID: String, wordCount: Int = 10,
        style: [String: Double]
    ) -> ExperimentTasks.MetricRow {
        ExperimentTasks.MetricRow(
            condition: condition, seed: 1, promptIndex: 0, promptID: promptID,
            wordCount: wordCount, distinct2: 0.9, markerDensity: [:],
            reasoningStyle: style)
    }

    private func testPinnedStyle() throws -> PinnedReasoningStyle {
        PinnedReasoningStyle(
            taxonomy: try loadTaxonomy(Self.taxonomyJSON),
            path: "prompts/taxonomies/style.json", hash: "abc123")
    }

    @Test func metricsCSVGainsStyleColumnsInTaxonomyOrder() throws {
        let style = try testPinnedStyle()
        let rows = [
            styleRow(
                condition: "baseline", promptID: "p1",
                style: ["hedge": 200.0, "question": 0.5])
        ]
        let csv = ExperimentTasks.metricsCSV(
            rows: rows, concepts: [], styleFeatureIDs: style.taxonomy.featureIDs)
        let lines = csv.split(separator: "\n")
        #expect(
            lines[0] == "condition,seed,promptIndex,promptID,wordCount,distinct2,rs_hedge,rs_question")
        #expect(lines[1] == "baseline,1,0,p1,10,0.9,200.0,0.5")
        // Without a taxonomy the CSV is byte-identical to the legacy shape.
        let legacy = ExperimentTasks.metricsCSV(rows: rows, concepts: [])
        #expect(
            legacy.split(separator: "\n")[0]
                == "condition,seed,promptIndex,promptID,wordCount,distinct2")
    }

    @Test func reportCarriesPerConditionReasoningStyleBlockAndEffectSizes() throws {
        let style = try testPinnedStyle()
        var manifest = ExperimentManifest(
            name: "style-report", description: "", modelID: "test/model")
        manifest.reasoningStyleTaxonomyPath = style.path
        manifest.reasoningStyleTaxonomyHash = style.hash
        let rows = [
            styleRow(
                condition: "baseline", promptID: "p1",
                style: ["hedge": 0.0, "question": 0.0]),
            styleRow(
                condition: "baseline", promptID: "p2",
                style: ["hedge": 100.0, "question": 0.5]),
            styleRow(
                condition: "steered", promptID: "p1",
                style: ["hedge": 300.0, "question": 1.0]),
            styleRow(
                condition: "steered", promptID: "p2",
                style: ["hedge": 500.0, "question": 0.5]),
        ]
        let report = ExperimentTasks.report(
            experiment: manifest, experimentHash: "hash",
            taskPrompts: (file: "f", hash: "h", prompts: []),
            rows: rows, conditionCount: 2, concepts: [], style: style)
        let block = try #require(report.conditions["steered"]?.reasoningStyle)
        #expect(block.taxonomy == "test-style-v1")
        #expect(block.taxonomyHash == "abc123")
        // Self-describing + status-stamped: the pinned file is named beside
        // its hash, and the block declares itself a diagnostic/manipulation
        // check so a report reader never cites it as an outcome endpoint.
        #expect(block.taxonomyFile == "prompts/taxonomies/style.json")
        #expect(block.diagnosticOnly == true)
        #expect(block.features["hedge"] == .init(mean: 400.0, n: 2))
        #expect(block.features["question"] == .init(mean: 0.75, n: 2))

        // rs_<id> joins the paired effect-size machinery: per-item diffs are
        // +300/+400 (hedge) and +1.0/0.0 (question).
        let entries = try #require(report.effectSizes)
        let hedge = try #require(
            entries.first { $0.metric == "rs_hedge" && $0.condition == "steered" })
        #expect(hedge.n == 2)
        #expect(hedge.meanDiff == 350.0)
        #expect(entries.contains { $0.metric == "rs_question" })

        // No pinned taxonomy → no block, no rs_ metrics (legacy unchanged).
        let legacy = ExperimentTasks.report(
            experiment: manifest, experimentHash: "hash",
            taskPrompts: (file: "f", hash: "h", prompts: []),
            rows: rows, conditionCount: 2, concepts: [])
        #expect(legacy.conditions["steered"]?.reasoningStyle == nil)
        #expect(legacy.effectSizes?.contains { $0.metric.hasPrefix("rs_") } == false)
    }

    // MARK: - rescore-style verb

    /// Fabricates a completed study run whose generations carry OUTPUT text
    /// (rescore recomputes style from it).
    private func fabricateStyleRun(
        for manifest: ExperimentManifest, stamped: Bool = true
    ) throws -> URL {
        let dir = ExperimentStore.runsDirectory.appending(
            component: "20260713T000000000Z-exp-\(manifest.name)-run")
        try FileManager.default.createDirectory(
            at: dir, withIntermediateDirectories: true)
        try JSONEncoder().encode(manifest).write(
            to: dir.appending(component: "experiment.json"))
        if stamped {
            try ExperimentStore.manifestHash(manifest).write(
                to: dir.appending(component: "experiment-hash.txt"),
                atomically: true, encoding: .utf8)
        }
        var lines = ""
        for (condition, text) in [
            ("baseline", "It is dry. Fine."),
            ("steered", "It might rain? Perhaps it might."),
        ] {
            for promptID in ["p1", "p2"] {
                lines += """
                    {"condition": "\(condition)", "seed": 1, "promptID": "\(promptID)", "promptIndex": 1, "wordCount": 4, "distinct2": 0.9, "output": "\(text)"}

                    """
            }
        }
        // A choice-instrument record must be skipped by rescoring.
        lines += """
            {"condition": "steered", "promptID": "p1", "instrument": "answerTokenLogprob", "selected": "affirm", "margin": 0.5}

            """
        try lines.write(
            to: dir.appending(component: "generations.jsonl"),
            atomically: true, encoding: .utf8)
        try "{}".write(
            to: dir.appending(component: "report.json"),
            atomically: true, encoding: .utf8)
        return dir
    }

    private func makeStyleStudy(name: String, root: URL) throws -> ExperimentManifest {
        let path = try plantTaxonomy(at: root)
        var manifest = try ExperimentStore.create(
            name: name, description: "", modelID: "test/model")
        let french = try StimulusSet(
            directory: VectorCatalog.conceptsDirectory.appending(component: "french"))
        manifest.concepts.append(
            ExperimentStore.makeConceptRef(
                name: "french", stimulusSetHash: french.hash, options: .init()))
        try ExperimentStore.save(manifest)
        return try ExperimentStore.pinReasoningStyleTaxonomy(
            experimentName: name, path: path)
    }

    @Test func rescoreStyleWritesNewFilesAndNeverMutatesTheSource() throws {
        try withStyleTempRoot { root in
            let manifest = try makeStyleStudy(name: "rescore-me", root: root)
            let source = try fabricateStyleRun(for: manifest)
            let fm = FileManager.default
            let before = try fm.contentsOfDirectory(atPath: source.path).sorted()
                .map { ($0, try Data(contentsOf: source.appending(component: $0))) }

            let out = try ExperimentTasks.rescoreStyle(experimentName: "rescore-me")
            #expect(out.lastPathComponent.contains("-exp-rescore-me-rescore-style"))
            // Source run untouched: same file set, same bytes.
            let after = try fm.contentsOfDirectory(atPath: source.path).sorted()
                .map { ($0, try Data(contentsOf: source.appending(component: $0))) }
            #expect(before.map(\.0) == after.map(\.0))
            #expect(zip(before, after).allSatisfy { $0.1 == $1.1 })

            let csv = try String(
                contentsOf: out.appending(component: "reasoning-style.csv"),
                encoding: .utf8)
            let lines = csv.split(separator: "\n")
            #expect(lines[0] == "condition,seed,promptIndex,promptID,rs_hedge,rs_question")
            #expect(lines.count == 5)  // 4 sampled rows; choice record skipped

            let data = try Data(
                contentsOf: out.appending(component: "reasoning-style.json"))
            let object = try #require(
                try JSONSerialization.jsonObject(with: data) as? [String: Any])
            #expect(object["experiment"] as? String == "rescore-me")
            #expect(object["sourceRun"] as? String == source.lastPathComponent)
            #expect(object["taxonomy"] as? String == "test-style-v1")
            #expect(object["taxonomyFile"] as? String != nil)
            #expect(object["diagnosticOnly"] as? Bool == true)
            #expect(object["epochUnverified"] == nil)
            let conditions = try #require(object["conditions"] as? [String: Any])
            let steered = try #require(conditions["steered"] as? [String: Any])
            let features = try #require(steered["features"] as? [String: Any])
            let hedge = try #require(features["hedge"] as? [String: Any])
            // "It might rain? Perhaps it might." — 3 hedges / 6 words → 500/1k.
            #expect(hedge["mean"] as? Double == 500.0)
            #expect(hedge["n"] as? Int == 2)
        }
    }

    @Test func rescoreStyleRequiresAPinnedTaxonomyAndGuardsTheEpoch() throws {
        try withStyleTempRoot { root in
            var manifest = try makeStyleStudy(name: "rescore-guard", root: root)
            _ = try fabricateStyleRun(for: manifest)

            // Manifest drifts AFTER the run → epoch mismatch refuses, and the
            // flag never bypasses a stamped mismatch.
            manifest.maxTokens = 512
            try ExperimentStore.save(manifest)
            #expect(throws: ExperimentError.self) {
                try ExperimentTasks.rescoreStyle(experimentName: "rescore-guard")
            }
            #expect(throws: ExperimentError.self) {
                try ExperimentTasks.rescoreStyle(
                    experimentName: "rescore-guard", allowUnverifiedEpoch: true)
            }

            // A study with no pinned taxonomy refuses with the pin hint.
            var bare = try ExperimentStore.create(
                name: "rescore-bare", description: "", modelID: "test/model")
            let french = try StimulusSet(
                directory: VectorCatalog.conceptsDirectory.appending(
                    component: "french"))
            bare.concepts.append(
                ExperimentStore.makeConceptRef(
                    name: "french", stimulusSetHash: french.hash, options: .init()))
            try ExperimentStore.save(bare)
            _ = try fabricateStyleRun(for: bare)
            #expect(throws: ExperimentError.self) {
                try ExperimentTasks.rescoreStyle(experimentName: "rescore-bare")
            }
        }
    }

    @Test func rescoreStyleAcceptsUnstampedRunsOnlyWithTheFlag() throws {
        try withStyleTempRoot { root in
            let manifest = try makeStyleStudy(name: "rescore-legacy", root: root)
            _ = try fabricateStyleRun(for: manifest, stamped: false)
            #expect(throws: ExperimentError.self) {
                try ExperimentTasks.rescoreStyle(experimentName: "rescore-legacy")
            }
            let out = try ExperimentTasks.rescoreStyle(
                experimentName: "rescore-legacy", allowUnverifiedEpoch: true)
            let object = try #require(
                try JSONSerialization.jsonObject(
                    with: Data(
                        contentsOf: out.appending(
                            component: "reasoning-style.json"))) as? [String: Any])
            #expect(object["epochUnverified"] as? Bool == true)
        }
    }

    // MARK: - analyze integration (rs_ metrics recomputed from output)

    @Test func analyzeRecomputesStyleMetricsFromOutputs() throws {
        try withStyleTempRoot { root in
            let manifest = try makeStyleStudy(name: "style-analyze", root: root)
            _ = try fabricateStyleRun(for: manifest)
            let out = try ExperimentTasks.analyze(experimentName: "style-analyze")
            let object = try #require(
                try JSONSerialization.jsonObject(
                    with: Data(contentsOf: out.appending(component: "analysis.json")))
                    as? [String: Any])
            let entries = try #require(object["effectSizes"] as? [[String: Any]])
            let hedge = try #require(
                entries.first {
                    $0["metric"] as? String == "rs_hedge"
                        && $0["condition"] as? String == "steered"
                })
            // steered 500/1k vs baseline 0 on both items → meanDiff 500.
            #expect(hedge["meanDiff"] as? Double == 500.0)
            #expect(hedge["n"] as? Int == 2)
            let csv = try String(
                contentsOf: out.appending(component: "effect-sizes.csv"),
                encoding: .utf8)
            #expect(csv.contains("rs_question"))
        }
    }

    // MARK: - readiness row + templates

    @Test func readinessRowIsOptionalUntilPinnedThenTracksTheFile() throws {
        try withStyleTempRoot { root in
            var manifest = ExperimentManifest(
                name: "ready", description: "", modelID: "test/model")
            func styleRow() -> DataRequirement? {
                StudyDataReadiness.requirements(for: manifest, workspaceRoot: root)
                    .first { $0.kind == .reasoningStyleTaxonomy }
            }
            // Unpinned → optional, with the template wired for scaffolding.
            let optional = try #require(styleRow())
            #expect(optional.status == .optional)
            #expect(optional.detail.contains("required for reasoning-style"))
            #expect(optional.templateID == DataTemplates.reasoningStyle.id)
            #expect(optional.path.hasPrefix("prompts/taxonomies/"))

            // Pinned + present + loads → present.
            let path = try plantTaxonomy(at: root)
            manifest.reasoningStyleTaxonomyPath = path
            manifest.reasoningStyleTaxonomyHash = "irrelevant-for-readiness"
            #expect(try #require(styleRow()).status == .present)

            // Pinned but broken → partial; pinned but gone → missing.
            try "not json".write(
                to: root.appending(path: path), atomically: true, encoding: .utf8)
            #expect(try #require(styleRow()).status == .partial)
            try FileManager.default.removeItem(at: root.appending(path: path))
            #expect(try #require(styleRow()).status == .missing)
        }
    }

    @Test func reasoningStyleTemplatesAreRegisteredAndLoad() throws {
        // Registered in the template registry (the domain-neutrality test
        // sweeps the whole directory via this registration).
        #expect(DataTemplates.all.contains { $0.id == "reasoning-style" })
        let directory = DataTemplates.seedURL(
            for: DataTemplates.reasoningStyle,
            workspaceRoot: URL(filePath: "/nonexistent")
        ).deletingLastPathComponent()
        // BOTH example taxonomies must load through the real loader — a
        // template a researcher cannot pin is not a starting point.
        for name in [
            "reasoning-style-generic-template.json",
            "reasoning-style-structure-template.json",
        ] {
            let taxonomy = try ReasoningStyleTaxonomy.load(
                url: directory.appending(component: name))
            #expect(!taxonomy.features.isEmpty, Comment(rawValue: name))
        }
    }

    @Test(arguments: ["reasoning-style-parity.json", "portable-regex-vectors.json"])
    func reasoningStyleFixtureCopiesAreByteIdenticalAcrossEngines(
        name: String
    ) throws {
        let serverCopy = URL(filePath: #filePath)
            .deletingLastPathComponent()  // ExperimentKitTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // project root
            .appending(
                components: "Server", "tests", "fixtures", "reasoning-style",
                name)
        guard FileManager.default.fileExists(atPath: serverCopy.path) else {
            return  // server tree not present (Swift-only checkout)
        }
        let swiftCopy = Self.reasoningStyleFixtureURL
            .deletingLastPathComponent()
            .appending(component: name)
        #expect(
            try Data(contentsOf: swiftCopy) == (try Data(contentsOf: serverCopy)),
            Comment(rawValue: name))
    }
}
