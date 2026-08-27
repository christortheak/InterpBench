import Foundation
import Testing

@testable import SteeringKit

/// The judge-capability predicate, over synthetic snapshot directories.
///
/// The cache scan `SteeredContainerLoader.localModelIDs` performs is
/// deliberately broad — a snapshot on disk is the whole test, because that
/// scan is ALSO the is-installed answer. So a judge picker cannot narrow the
/// scan; it has to filter, and this is the filter. The fixtures below mirror
/// the three shapes a real cache holds side by side: a chat model, a
/// dictionary-style artifact with no `config.json` at all, and a bare
/// language model with a generative head but no chat template.
struct LocalJudgeCapabilityTests {

    private func withSnapshot<T>(
        files: [String: String], _ body: (URL) throws -> T
    ) rethrows -> T {
        let root = FileManager.default.temporaryDirectory
            .appending(component: "snapshot-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        for (name, contents) in files {
            let url = root.appending(path: name)
            try? FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true)
            try? contents.write(to: url, atomically: true, encoding: .utf8)
        }
        return try body(root)
    }

    // MARK: - Passes

    @Test func aCausalLMWithAChatTemplateCanJudge() {
        withSnapshot(files: [
            "config.json": #"{"architectures": ["FooForCausalLM"], "model_type": "foo"}"#,
            "tokenizer_config.json": #"{"chat_template": "{{ messages }}"}"#,
        ]) { snapshot in
            let verdict = LocalJudgeCapability.inspect(snapshot: snapshot)
            #expect(verdict.isCapable)
            #expect(verdict.reason == nil)
        }
    }

    /// The other generative head this cache actually holds: a multimodal
    /// instruction model whose architecture ends in ConditionalGeneration,
    /// carrying its template in a sidecar file rather than in
    /// `tokenizer_config.json`.
    @Test func aConditionalGenerationHeadWithASidecarTemplateCanJudge() {
        withSnapshot(files: [
            "config.json":
                #"{"architectures": ["BarForConditionalGeneration"], "model_type": "bar"}"#,
            "chat_template.json": #"{"chat_template": "{{ messages }}"}"#,
            "tokenizer_config.json": #"{"bos_token": "<s>"}"#,
        ]) { snapshot in
            #expect(LocalJudgeCapability.inspect(snapshot: snapshot).isCapable)
        }
    }

    @Test func aJinjaSidecarTemplateAlsoCounts() {
        withSnapshot(files: [
            "config.json": #"{"architectures": ["FooForCausalLM"]}"#,
            "chat_template.jinja": "{{ messages }}",
        ]) { snapshot in
            #expect(LocalJudgeCapability.inspect(snapshot: snapshot).isCapable)
        }
    }

    // MARK: - Refusals

    /// A sparse-dictionary-shaped repo: no `config.json` at all, just a
    /// directory of tensors. This is the shape that used to appear in the
    /// judge picker and either garbage-load or be silently skipped.
    @Test func aDictionaryShapedRepoWithNoConfigRefuses() {
        withSnapshot(files: ["layer_post/weights.txt": "not a model"]) { snapshot in
            let verdict = LocalJudgeCapability.inspect(snapshot: snapshot)
            #expect(!verdict.isCapable)
            #expect(verdict.reason?.contains("no config.json") == true)
        }
    }

    /// A repo whose config exists but names no generative head.
    @Test func aNonGenerativeHeadRefusesAndNamesWhatItFound() {
        withSnapshot(files: [
            "config.json": #"{"architectures": ["FooForSequenceClassification"]}"#,
            "tokenizer_config.json": #"{"chat_template": "{{ messages }}"}"#,
        ]) { snapshot in
            let verdict = LocalJudgeCapability.inspect(snapshot: snapshot)
            #expect(!verdict.isCapable)
            #expect(verdict.reason?.contains("FooForSequenceClassification") == true)
            #expect(verdict.reason?.contains("no text-generation head") == true)
        }
    }

    /// A generative head with no way to be ASKED anything: the tokenizer
    /// config carries no template and no sidecar exists.
    @Test func aMissingChatTemplateRefuses() {
        withSnapshot(files: [
            "config.json": #"{"architectures": ["FooLMHeadModel"], "model_type": "foo"}"#,
            "tokenizer_config.json": #"{"bos_token": "<s>"}"#,
        ]) { snapshot in
            // This fixture fails on the head too — an LM-head autoencoder is
            // not a chat model — so assert the template rule on a head that
            // does pass.
            #expect(!LocalJudgeCapability.inspect(snapshot: snapshot).isCapable)
        }
        withSnapshot(files: [
            "config.json": #"{"architectures": ["FooForCausalLM"]}"#,
            "tokenizer_config.json": #"{"bos_token": "<s>"}"#,
        ]) { snapshot in
            let verdict = LocalJudgeCapability.inspect(snapshot: snapshot)
            #expect(!verdict.isCapable)
            #expect(verdict.reason?.contains("no chat template") == true)
        }
    }

    @Test func anEmptyTemplateStringIsNoTemplate() {
        withSnapshot(files: [
            "config.json": #"{"architectures": ["FooForCausalLM"]}"#,
            "tokenizer_config.json": #"{"chat_template": "   "}"#,
        ]) { snapshot in
            #expect(!LocalJudgeCapability.inspect(snapshot: snapshot).isCapable)
        }
    }

    /// Existence was never the test the tokenizer-config route holds itself
    /// to, and the sidecar route now matches it (review round 7, finding 6).
    /// Three snapshots that used to pass the picker and fail at load: an
    /// empty sidecar, a whitespace-only one, and a DIRECTORY wearing the
    /// sidecar's name.
    @Test func anEmptyOrDirectorySidecarIsNoTemplate() {
        withSnapshot(files: [
            "config.json": #"{"architectures": ["FooForCausalLM"]}"#,
            "chat_template.jinja": "",
        ]) { snapshot in
            let verdict = LocalJudgeCapability.inspect(snapshot: snapshot)
            #expect(!verdict.isCapable)
            #expect(verdict.reason?.contains("no chat template") == true)
        }
        withSnapshot(files: [
            "config.json": #"{"architectures": ["FooForCausalLM"]}"#,
            "chat_template.jinja": "   \n  ",
        ]) { snapshot in
            #expect(!LocalJudgeCapability.inspect(snapshot: snapshot).isCapable)
        }
        // A directory named `chat_template.json` "exists" and renders nothing.
        withSnapshot(files: [
            "config.json": #"{"architectures": ["FooForCausalLM"]}"#,
            "chat_template.json/placeholder.txt": "not a template",
        ]) { snapshot in
            #expect(!LocalJudgeCapability.inspect(snapshot: snapshot).isCapable)
        }
    }

    /// A `.json` sidecar must BE json. A `.jinja` one stops at non-empty on
    /// purpose: deciding whether Jinja source is valid means implementing
    /// Jinja, and a template this predicate cannot parse is still a template
    /// the loader can.
    @Test func aJSONSidecarMustParseAndAJinjaOneNeedOnlyHaveBytes() {
        withSnapshot(files: [
            "config.json": #"{"architectures": ["FooForCausalLM"]}"#,
            "chat_template.json": "{ not json at all",
        ]) { snapshot in
            #expect(!LocalJudgeCapability.inspect(snapshot: snapshot).isCapable)
        }
        withSnapshot(files: [
            "config.json": #"{"architectures": ["FooForCausalLM"]}"#,
            "chat_template.json": #"{"chat_template": "{{ messages }}"}"#,
        ]) { snapshot in
            #expect(LocalJudgeCapability.inspect(snapshot: snapshot).isCapable)
        }
        // Jinja source is not JSON and must not be asked to be.
        withSnapshot(files: [
            "config.json": #"{"architectures": ["FooForCausalLM"]}"#,
            "chat_template.jinja": "{% for m in messages %}{{ m }}{% endfor %}",
        ]) { snapshot in
            #expect(LocalJudgeCapability.inspect(snapshot: snapshot).isCapable)
        }
    }

    /// An unusable sidecar does not veto a template the tokenizer config
    /// carries — the two routes are alternatives, and only one need hold.
    @Test func anUnusableSidecarFallsThroughToTheTokenizerConfig() {
        withSnapshot(files: [
            "config.json": #"{"architectures": ["FooForCausalLM"]}"#,
            "chat_template.json": "",
            "tokenizer_config.json": #"{"chat_template": "{{ messages }}"}"#,
        ]) { snapshot in
            #expect(LocalJudgeCapability.inspect(snapshot: snapshot).isCapable)
        }
    }

    @Test func unreadableConfigJSONRefusesRatherThanCrashing() {
        withSnapshot(files: [
            "config.json": "not json at all",
            "tokenizer_config.json": #"{"chat_template": "{{ messages }}"}"#,
        ]) { snapshot in
            let verdict = LocalJudgeCapability.inspect(snapshot: snapshot)
            #expect(!verdict.isCapable)
            #expect(verdict.reason?.contains("not readable JSON") == true)
        }
    }

    // MARK: - Head matching

    @Test func headsAreMatchedAsSuffixesNotSubstrings() {
        #expect(LocalJudgeCapability.isGenerativeHead("Qwen3ForCausalLM"))
        #expect(LocalJudgeCapability.isGenerativeHead("Gemma3ForConditionalGeneration"))
        #expect(!LocalJudgeCapability.isGenerativeHead("GPT2LMHeadModel"))
        #expect(!LocalJudgeCapability.isGenerativeHead("FooForCausalLMEncoder"))
    }

    // MARK: - The memo

    /// A picker recomputes its options on every render; the verdict must be
    /// remembered rather than re-stated per row per frame. A repo this Mac
    /// does not hold is remembered as not-capable, and says so plainly.
    @Test func verdictsAreMemoizedPerRepoID() {
        LocalJudgeCapability.forgetCachedVerdicts()
        let id = "no-such-vendor/no-such-repo-\(UUID().uuidString)"
        let first = LocalJudgeCapability.verdict(forModelID: id)
        let second = LocalJudgeCapability.verdict(forModelID: id)
        #expect(first == second)
        #expect(!first.isCapable)
        #expect(first.reason?.contains("no cached snapshot") == true)
        LocalJudgeCapability.forgetCachedVerdicts()
    }
}
