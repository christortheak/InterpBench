import Foundation
import Synchronization

/// Can a locally cached Hugging Face repo answer a prompt — i.e. act as a
/// paired judge — decided from the snapshot's OWN files, without loading a
/// single weight.
///
/// Why this is separate from `SteeredContainerLoader.localModelIDs`: that
/// scan's only test is "a snapshot exists on disk", deliberately, because it
/// is ALSO the is-installed test every picker's installed badge and every
/// load refusal agree on. Narrowing it would silently un-install artifacts
/// the rest of the app depends on. So the scan stays broad and the judge
/// pickers filter, which is the honest split: the cache holds non-generative
/// artifacts (sparse dictionary repos, lens repos) alongside models, and
/// offering those as judges either garbage-loads them locally or is silently
/// skipped on a server route.
///
/// The discriminator, both halves required:
///
/// 1. **A generative head.** `config.json` exists and its `architectures`
///    names at least one entry ending in `ForCausalLM` or
///    `ForConditionalGeneration`. A repo with no `config.json` at all is not
///    a transformers model; a repo whose only head is, say, an LM-head
///    autoencoder or a classifier cannot answer a prompt.
/// 2. **A chat template.** Either `tokenizer_config.json` carries a
///    non-empty `chat_template`, or a `chat_template.jinja` /
///    `chat_template.json` sits beside it and is itself USABLE. A judge is
///    asked an instruction in a chat turn; a repo with no template has no
///    defined way to be asked one, and rendering it as raw text produces a
///    verdict-shaped hallucination rather than a refusal.
///
///    The sidecar route holds the SAME standard as the tokenizer-config
///    route, which has always required a non-empty template (review round 7,
///    finding 6): existence alone said yes to an empty file, to a
///    *directory* named `chat_template.json`, and to a sidecar full of
///    malformed JSON — three snapshots that pass the picker and then fail at
///    load. So a sidecar must be a regular file with bytes in it, and a
///    `.json` sidecar must parse as JSON. A `.jinja` sidecar is checked no
///    further: non-empty is the whole test, because deciding whether Jinja
///    source is valid means implementing Jinja, which this predicate must
///    never do.
///
/// Verdicts are memoized per repo id: a picker recomputes its options on
/// every render, and re-stating a snapshot each time is a filesystem hit per
/// row per frame.
public enum LocalJudgeCapability {

    /// A repo's judge verdict, with the engine's own words for a refusal —
    /// the picker QUOTES `reason`, it never invents one.
    public struct Verdict: Sendable, Equatable {
        public let isCapable: Bool
        /// Why the repo cannot judge; nil exactly when `isCapable`.
        public let reason: String?

        public init(isCapable: Bool, reason: String?) {
            self.isCapable = isCapable
            self.reason = reason
        }

        public static let capable = Verdict(isCapable: true, reason: nil)

        public static func refused(_ reason: String) -> Verdict {
            Verdict(isCapable: false, reason: reason)
        }
    }

    /// Architecture-name suffixes that mean "this repo generates text from a
    /// prompt". Matched as SUFFIXES, not substrings: the family prefix
    /// varies per model (`Qwen3ForCausalLM`, `Gemma3ForConditionalGeneration`)
    /// but the head is always the tail of the name.
    public static let generativeHeadSuffixes = [
        "ForCausalLM", "ForConditionalGeneration",
    ]

    /// Snapshot files that carry a chat template on their own, beside
    /// `tokenizer_config.json`.
    public static let chatTemplateFileNames = [
        "chat_template.jinja", "chat_template.json",
    ]

    // MARK: - The predicate

    /// Read one cached snapshot directory and say whether it can judge.
    /// Pure over the filesystem: no network, no weights, no tokenizer load.
    public static func inspect(snapshot: URL) -> Verdict {
        guard let data = try? Data(contentsOf: snapshot.appending(component: "config.json"))
        else {
            return .refused(
                "the cached snapshot has no config.json — this repo is not a "
                    + "loadable text model")
        }
        guard
            let config = (try? JSONSerialization.jsonObject(with: data))
                as? [String: Any]
        else {
            return .refused("the cached snapshot's config.json is not readable JSON")
        }
        let architectures = (config["architectures"] as? [Any])?
            .compactMap { $0 as? String } ?? []
        guard architectures.contains(where: isGenerativeHead) else {
            let named = architectures.isEmpty
                ? "no architecture" : "'" + architectures.joined(separator: ", ") + "'"
            return .refused(
                "config.json names \(named) — no text-generation head "
                    + "(…ForCausalLM / …ForConditionalGeneration)")
        }
        guard hasChatTemplate(snapshot: snapshot) else {
            return .refused(
                "the cached snapshot carries no chat template — there is no "
                    + "defined way to put a question to this repo")
        }
        return .capable
    }

    /// Whether one architecture name declares a generative head.
    public static func isGenerativeHead(_ architecture: String) -> Bool {
        let trimmed = architecture.trimmingCharacters(in: .whitespacesAndNewlines)
        return generativeHeadSuffixes.contains { trimmed.hasSuffix($0) }
    }

    /// Whether the snapshot supplies a chat template by either route.
    public static func hasChatTemplate(snapshot: URL) -> Bool {
        for name in chatTemplateFileNames
        where isUsableTemplateSidecar(snapshot.appending(component: name)) {
            return true
        }
        guard
            let data = try? Data(
                contentsOf: snapshot.appending(component: "tokenizer_config.json")),
            let config = (try? JSONSerialization.jsonObject(with: data))
                as? [String: Any]
        else { return false }
        // A string on most repos; a list of NAMED templates on a few. Either
        // is a template.
        if let text = config["chat_template"] as? String {
            return !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
        if let list = config["chat_template"] as? [Any] { return !list.isEmpty }
        return false
    }

    /// Whether ONE sidecar file actually supplies a template. Existence is
    /// not the test (review round 7, finding 6): a directory, an empty file,
    /// or a `.json` sidecar that is not JSON all "exist" and none of them can
    /// be rendered.
    ///
    /// A `.jinja` sidecar stops at non-empty on purpose — the alternative is
    /// a Jinja parser, and a template this predicate cannot parse is still a
    /// template the loader can.
    public static func isUsableTemplateSidecar(_ url: URL) -> Bool {
        var isDirectory: ObjCBool = false
        guard
            FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory),
            !isDirectory.boolValue,
            let data = try? Data(contentsOf: url),
            !data.isEmpty
        else { return false }
        // Whitespace alone is as empty as zero bytes — the same standard the
        // tokenizer_config route holds its `chat_template` string to.
        if let text = String(data: data, encoding: .utf8),
            text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            return false
        }
        guard url.pathExtension.lowercased() == "json" else { return true }
        return (try? JSONSerialization.jsonObject(with: data)) != nil
    }

    // MARK: - Memoized per-repo verdicts

    private static let cache = Mutex<[String: Verdict]>([:])

    /// The verdict for a repo id, resolved through the HF cache's
    /// `refs/main` exactly like a load would, and remembered. A repo with no
    /// cached snapshot is not judge-capable HERE — the picker only ever
    /// offers what this Mac already holds.
    public static func verdict(forModelID modelID: String) -> Verdict {
        if let cached = cache.withLock({ $0[modelID] }) { return cached }
        let verdict: Verdict
        if let snapshot = SteeredContainerLoader.cachedSnapshotDirectory(for: modelID) {
            verdict = inspect(snapshot: snapshot)
        } else {
            verdict = .refused("this Mac holds no cached snapshot for this repo")
        }
        cache.withLock { $0[modelID] = verdict }
        return verdict
    }

    public static func isCapable(_ modelID: String) -> Bool {
        verdict(forModelID: modelID).isCapable
    }

    /// Drop the memo. Called on every installed-models rescan
    /// (`SubstrateCatalog.refreshLocalInstalledModels`), which is what makes
    /// a freshly installed repo judge-capable without a relaunch, and by
    /// tests that inspect the same id twice under different fixtures.
    public static func forgetCachedVerdicts() {
        cache.withLock { $0.removeAll() }
    }

    /// TEST SEAM: plant a verdict so a test can prove the memo is CONSULTED
    /// and then dropped, without needing a real snapshot on disk to change
    /// underneath it.
    public static func rememberForTesting(_ modelID: String, _ verdict: Verdict) {
        cache.withLock { $0[modelID] = verdict }
    }
}
