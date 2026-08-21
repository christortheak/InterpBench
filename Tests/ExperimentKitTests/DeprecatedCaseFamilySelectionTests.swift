import Foundation
import Testing

@testable import ExperimentKit

/// The `caseFamily: "sentencing"` magic trigger, deprecated 2026-08-18.
///
/// Two claims, and they pull in opposite directions on purpose:
///
/// 1. **It still works.** A manifest that already depends on the trigger keeps
///    parsing `parsedMonths` out of every sampled record, byte for byte. A
///    deprecation that changed a measured number would silently invalidate
///    finished studies — the one thing the firewall exists to prevent.
/// 2. **It says so.** Every site where it fires emits the closed-vocabulary
///    advisory `deprecatedImplicitSelection`, and a study that DECLARES a
///    `numericParser` emits nothing, because nothing was selected implicitly.
///
/// The Python twin is `Server/tests/test_deprecated_case_family.py`.
///
/// Serialized and holding `ExperimentRootOverrideLock`: `rootOverride` is a
/// process-global seam shared with every other lifecycle suite.
@Suite(.serialized) struct DeprecatedCaseFamilySelectionTests {

    private func withTempRoot<T>(_ body: (URL) async throws -> T) async throws -> T {
        ExperimentRootOverrideLock.acquire()
        let temp = FileManager.default.temporaryDirectory
            .appending(component: "deprecated-cf-\(UUID().uuidString)")
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

    private func manifest(
        caseFamily: String?, numericParser: String? = nil,
        studyKind: ExperimentManifest.StudyKind = .modelOutput
    ) -> ExperimentManifest {
        var manifest = ExperimentManifest(
            name: "demo", description: "", modelID: "google/gemma-3-27b-it")
        manifest.caseFamily = caseFamily
        manifest.numericParser = numericParser
        manifest.studyKind = studyKind
        return manifest
    }

    // MARK: - 1. Compatibility: the trigger still selects the endpoint

    /// The behaviour half, pinned. `judicialParses` is the dispatch both the
    /// ordinary and variant condition paths share, so this is the record
    /// contract itself, not a proxy for it.
    @Test func theTriggerStillSelectsTheBuiltInDurationEndpoint() {
        let fired = ExperimentTasks.judicialParses(
            output: "8 years 3 months", options: nil, caseFamily: "sentencing")
        #expect(fired.parsedMonths == .some(.some(99.0)))

        // …and only for that value: any other label parses nothing, which is
        // what makes `caseFamily` a label everywhere else.
        let silent = ExperimentTasks.judicialParses(
            output: "8 years 3 months", options: nil, caseFamily: "katzZamir")
        #expect(silent.parsedMonths == nil)
        let absent = ExperimentTasks.judicialParses(
            output: "8 years 3 months", options: nil, caseFamily: nil)
        #expect(absent.parsedMonths == nil)
    }

    // MARK: - 2. The predicate: exactly when the trigger fires

    @Test func thePredicateFiresOnlyOnTheUndeclaredSentencingLabel() {
        #expect(manifest(caseFamily: "sentencing").usesImplicitCaseFamilyEndpoint)
        #expect(!manifest(caseFamily: nil).usesImplicitCaseFamilyEndpoint)
        #expect(!manifest(caseFamily: "katzZamir").usesImplicitCaseFamilyEndpoint)
        #expect(!manifest(caseFamily: "siliconFormalism").usesImplicitCaseFamilyEndpoint)
    }

    /// A DECLARED parser wins, so nothing was selected implicitly and there is
    /// nothing to advise about. This is the row that keeps the advisory from
    /// becoming noise every migrated study has to ignore.
    @Test func aDeclaredNumericParserSilencesThePredicate() {
        #expect(
            !manifest(caseFamily: "sentencing", numericParser: "sentencing-months")
                .usesImplicitCaseFamilyEndpoint)
        // Whitespace is not a declaration.
        #expect(
            manifest(caseFamily: "sentencing", numericParser: "  ")
                .usesImplicitCaseFamilyEndpoint)
        #expect(
            manifest(caseFamily: "sentencing", numericParser: "")
                .usesImplicitCaseFamilyEndpoint)
    }

    /// The multi-agent row is the one asymmetry, and it is the server's
    /// panel-effects decomposition that creates it: that site reads the label
    /// ALONE, so a declared parser does not silence it. The predicate is
    /// cross-engine, so this engine answers the same even though it runs no
    /// panel-effects decomposition today.
    @Test func multiAgentStudiesFireOnTheLabelAlone() {
        #expect(
            manifest(
                caseFamily: "sentencing", numericParser: "sentencing-months",
                studyKind: .multiAgent
            ).usesImplicitCaseFamilyEndpoint)
        #expect(
            !manifest(caseFamily: nil, studyKind: .multiAgent)
                .usesImplicitCaseFamilyEndpoint)
    }

    // MARK: - 3. The advisory itself

    @Test func theAdvisoryCodeIsInTheClosedVocabulary() {
        #expect(
            CLIAdvisory.vocabulary.contains(
                CLIAdvisory.deprecatedImplicitSelection.rawValue))
    }

    /// The sentence is matched on by agents and by the server twin, so its
    /// load-bearing parts are pinned rather than left to prose drift.
    @Test func theAdvisorySentenceNamesTheTriggerAndTheReplacement() {
        let advisory = ExperimentManifest.implicitCaseFamilyAdvisory
        #expect(advisory.contains("caseFamily 'sentencing'"))
        #expect(advisory.contains("numericParser"))
        #expect(advisory.contains("deprecated"))
        #expect(advisory.contains("sentencing-months"))
    }

    @Test func theEnvelopeAdvisoryFiresExactlyWhenTheTriggerDoes() async throws {
        try await withTempRoot { _ in
            _ = try ExperimentStore.create(
                name: "legacy", description: "", modelID: "google/gemma-3-27b-it")
            _ = try ExperimentStore.updateDraft(name: "legacy") {
                $0.caseFamily = "sentencing"
            }
            _ = try ExperimentStore.create(
                name: "declared", description: "", modelID: "google/gemma-3-27b-it")
            _ = try ExperimentStore.updateDraft(name: "declared") {
                $0.caseFamily = "sentencing"
                $0.numericParser = "sentencing-months"
            }
            _ = try ExperimentStore.create(
                name: "unlabelled", description: "", modelID: "google/gemma-3-27b-it")

            let fired = ExperimentCLIRunner.implicitCaseFamilyAdvisories(
                experimentNamed: "legacy")
            #expect(fired.count == 1)
            #expect(
                fired.first?.code
                    == CLIAdvisory.deprecatedImplicitSelection.rawValue)
            #expect(
                fired.first?.detail
                    == ExperimentManifest.implicitCaseFamilyAdvisory)

            #expect(
                ExperimentCLIRunner.implicitCaseFamilyAdvisories(
                    experimentNamed: "declared"
                ).isEmpty)
            #expect(
                ExperimentCLIRunner.implicitCaseFamilyAdvisories(
                    experimentNamed: "unlabelled"
                ).isEmpty)
            // An unreadable manifest is the verb's problem to report, never
            // the advisory helper's: it must not throw and must not invent an
            // advisory it cannot justify.
            #expect(
                ExperimentCLIRunner.implicitCaseFamilyAdvisories(
                    experimentNamed: "no-such-experiment"
                ).isEmpty)
        }
    }

    /// Advisories never change the exit code — the rule a `set -e` wrapper
    /// depends on. A run that fires this one still succeeds.
    @Test func theAdvisoryPromotesTheStateAndNotTheExitCode() {
        let envelope = SteerLabCLIEnvelope.success(
            verb: "experiment run", engine: SteerLabCLIEnvelope.localEngine,
            message: "ran 'legacy'", changed: true,
            advisories: [
                .init(
                    CLIAdvisory.deprecatedImplicitSelection,
                    ExperimentManifest.implicitCaseFamilyAdvisory),
            ])
        #expect(envelope.state == .okWithAdvisories)
        #expect(envelope.exitCode == 0)
    }

    // MARK: - 4. The durable stamp

    /// `advisories.txt` APPENDS. The cross-substrate advisory and this one can
    /// both be true of the same run, and the truncating write this replaced
    /// would have let the second erase the first.
    @Test func runAdvisoriesAccumulateRatherThanOverwrite() throws {
        let directory = FileManager.default.temporaryDirectory
            .appending(component: "advisory-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        ExperimentTasks.emitRunAdvisory("first advisory", to: directory)
        ExperimentTasks.emitRunAdvisory(
            ExperimentManifest.implicitCaseFamilyAdvisory, to: directory)

        let text = try String(
            contentsOf: directory.appending(component: "advisories.txt"),
            encoding: .utf8)
        #expect(text.hasPrefix("first advisory\n"))
        #expect(text.contains(ExperimentManifest.implicitCaseFamilyAdvisory))
        #expect(text.hasSuffix("\n"))
    }
}
