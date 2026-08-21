import CryptoKit
import Foundation

/// Versioned judge rubrics: plain-text/markdown files under
/// `prompts/rubrics/`, git-versioned and pinned by hash into experiment
/// manifests (`judgeRubricFile` + `judgeRubricHash`) exactly like stimulus
/// sets and task prompts. The default rubric file is owned by the server
/// side of the repo (`prompts/rubrics/default-paired-v1.md` holds the text
/// both engines' paired judges historically inlined); this store only READS
/// rubric files — it never writes them.
public enum JudgeRubricStore {

    /// Rubric directory, relative to the project root.
    public static let relativeDirectory = "prompts/rubrics"

    /// The default paired-judge rubric file (shared cross-engine constant).
    public static let defaultRubricFile = "prompts/rubrics/default-paired-v1.md"

    /// The refusal both engines give an evaluation with no rubric at all.
    /// Byte-identical to the server's `tasks.NO_RUBRIC_REFUSAL`, and it names
    /// `steerlab-cli` on BOTH engines on purpose: authoring is Mac-authority
    /// (WP0 §10.x), and the Python CLI has no `pin-rubric` verb to name.
    public static func noRubricRefusal(_ name: String) -> String {
        "study '\(name)' has no judge rubric — pin one: "
            + "'steerlab-cli experiment pin-rubric \(name) "
            + "\(defaultRubricFile)' (any file under \(relativeDirectory)/; "
            + "inline draft text is draft-only and cannot freeze)"
    }

    public static var directory: URL {
        VectorCatalog.projectRoot.appending(path: relativeDirectory)
    }

    /// Project-relative paths of every rubric file on disk, sorted. Empty
    /// when the directory does not exist yet.
    public static func list() -> [String] {
        let fm = FileManager.default
        guard
            let entries = try? fm.contentsOfDirectory(
                at: directory, includingPropertiesForKeys: [.isRegularFileKey])
        else { return [] }
        return entries
            .filter { !$0.lastPathComponent.hasPrefix(".") }
            .map { "\(relativeDirectory)/\($0.lastPathComponent)" }
            .sorted()
    }

    /// The refusal for a rubric path that is not on disk — the twin of
    /// `ExperimentStore.pinTaskPrompts`' "task prompt file not found", in the
    /// same sentence shape, so the two pin verbs answer the same mistake the
    /// same way.
    ///
    /// Observed live 2026-08-18 (ONBOARDING verification): `pin-rubric`
    /// against a typo'd path — or any workspace with no `prompts/rubrics/` —
    /// let Foundation's `Data(contentsOf:)` out unhandled, so the human line
    /// was an `NSCocoaErrorDomain` dump carrying an `NSUnderlyingError`
    /// POINTER and the envelope was the generic missing-file catch, whose
    /// repair ("no file at /abs/path") names no command and no convention.
    public static func missingRubricRefusal(path: String) -> String {
        "judge rubric file not found: \(path)"
    }

    /// …and the runnable repair: the CONVENTION directory (a rubric lives
    /// under `prompts/rubrics/`, which is also the only form that can freeze),
    /// the shipped default as a working choice, and the verb to retry.
    public static func missingRubricRepair(
        experimentName: String, relativePath: String
    ) -> String {
        // When the ABSENT path is the shipped default, the workspace has no
        // rubric directory at all (a hand-made workspace, or one whose seed
        // was pruned) — pointing at the default as an example would be the
        // repair naming itself.
        let example =
            relativePath == defaultRubricFile
            ? " (a seeded workspace ships \(defaultRubricFile))"
            : " (the shipped \(defaultRubricFile) is one)"
        return "author \(relativePath) under \(relativeDirectory)/\(example), "
            + "then steerlab-cli experiment pin-rubric \(experimentName) "
            + "\(relativePath)"
    }

    /// Loads a rubric by project-relative path (contained to the project
    /// root — rubric paths can arrive from API clients). Returns the text
    /// and the SHA-256 of the file's raw bytes (the pin).
    ///
    /// `experimentName` is used ONLY to make the missing-file repair runnable;
    /// it defaults to a placeholder for the callers that have no manifest.
    public static func load(
        _ relativePath: String, experimentName: String = "<name>"
    ) throws -> (text: String, hash: String) {
        let url = try VectorCatalog.projectFile(relativePath)
        guard let data = try? Data(contentsOf: url) else {
            throw ExperimentError.refusing(
                .missingPrerequisite,
                missingRubricRefusal(path: url.path),
                repair: missingRubricRepair(
                    experimentName: experimentName, relativePath: relativePath))
        }
        let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        return (String(decoding: data, as: UTF8.self), hash)
    }

    /// Pins a rubric file's current hash into a manifest. Returns the hash.
    @discardableResult
    public static func pin(
        _ relativePath: String, into manifest: inout ExperimentManifest
    ) throws -> String {
        let (_, hash) = try load(relativePath, experimentName: manifest.name)
        manifest.judgeRubricFile = relativePath
        manifest.judgeRubricHash = hash
        return hash
    }

    /// The rubric text an evaluation must use: the pinned file (verified
    /// against its pinned hash — drift is an error, not a warning) when the
    /// manifest pins one, else the inline draft text.
    public static func resolveRubric(
        for manifest: ExperimentManifest, inlineRubric: String?
    ) throws -> (text: String, file: String?, hash: String?) {
        if let file = manifest.judgeRubricFile {
            let (text, hash) = try load(file, experimentName: manifest.name)
            if let pinned = manifest.judgeRubricHash, pinned != hash {
                throw ExperimentError(
                    reason: "judge rubric '\(file)' drifted from the pinned hash "
                        + "(have \(hash.prefix(12))…, pinned \(pinned.prefix(12))…)")
            }
            return (text, file, hash)
        }
        let inline = (inlineRubric ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !inline.isEmpty else {
            // WP0 step 5½: "enter draft rubric text" was GUI language — the
            // draft-text field exists only in the Studies panel, so the one
            // remedy this refusal named was unreachable headlessly. It now
            // names the verb that pins a rubric file, which is also the only
            // form that can freeze (inline text is draft-only).
            // WP0 dry run #2's skipped check: this refusal was RIGHT and
            // untyped — `verbFailed`/70 with the boilerplate repair ("read
            // the reason and repair the named input"). It is a
            // missingPrerequisite in the exact sense the vocabulary defines:
            // the verb needs something the study never declared. The reason
            // string is unchanged (human output stays byte-stable); only the
            // gate id and the runnable repair are new. Server twin:
            // `tasks._resolve_rubric`, which raises the SAME sentence.
            throw ExperimentError.refusing(
                .missingPrerequisite, noRubricRefusal(manifest.name),
                repair: "steerlab-cli experiment pin-rubric \(manifest.name) "
                    + "\(defaultRubricFile) && steerlab-cli experiment "
                    + "evaluate \(manifest.name)")
        }
        return (inline, nil, nil)
    }
}
