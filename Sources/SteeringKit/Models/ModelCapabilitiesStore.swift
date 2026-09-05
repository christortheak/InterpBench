import Foundation
import Jinja
import Tokenizers

/// Workspace records of ``ModelCapabilities`` — read, write, look up, ensure
/// from the cached tokenizer — and the human override. Server twin: the
/// workspace half of `experiment/model_capabilities.py`.
public enum ModelCapabilitiesStore {

    // MARK: - Paths

    /// `<owner>--<repo>@<revision>.json`; `unpinned` stands in for an unknown
    /// revision so the name is always well-formed. Server twin:
    /// `model_capabilities.record_filename`.
    public static func recordFilename(modelID: String, revision: String?) -> String {
        var slug = modelID.trimmingCharacters(in: .whitespacesAndNewlines)
        while slug.hasPrefix("/") { slug.removeFirst() }
        while slug.hasSuffix("/") { slug.removeLast() }
        slug = slug.replacingOccurrences(of: "/", with: "--")
        let cleaned = String(slug.map { character in
            (character.isLetter || character.isNumber || "._-".contains(character))
                && character.isASCII ? character : "-"
        })
        return "\(cleaned)@\(revision ?? "unpinned").json"
    }

    public static func recordRelativePath(modelID: String, revision: String?) -> String {
        ModelCapabilities.directory + "/" + recordFilename(modelID: modelID, revision: revision)
    }

    public static func recordURL(modelID: String, revision: String?, root: URL) -> URL {
        root.appending(path: recordRelativePath(modelID: modelID, revision: revision))
    }

    // MARK: - Files

    /// Write (or overwrite) a record; returns the workspace-relative path.
    @discardableResult
    public static func write(_ record: ModelCapabilities.Record, root: URL) throws -> String {
        let url = recordURL(modelID: record.modelID, revision: record.revision, root: root)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        // Pretty, sorted, slashes unescaped — the file the other engine
        // rewrites the same way; the HASH is over the canonical form.
        let data = try JSONSerialization.data(
            withJSONObject: record.jsonObject,
            options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        try (data + Data("\n".utf8)).write(to: url, options: .atomic)
        return recordRelativePath(modelID: record.modelID, revision: record.revision)
    }

    public static func read(_ url: URL) throws -> ModelCapabilities.Record {
        let data = try Data(contentsOf: url)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ModelCapabilities.RecordError("\(url.lastPathComponent): record is not a JSON object")
        }
        return try ModelCapabilities.Record.parse(object)
    }

    /// Every readable record in the workspace as (relative path, record),
    /// sorted by path. Unreadable files are skipped — a listing is a display
    /// surface; the verb that reads a specific record names the problem.
    public static func list(root: URL) -> [(path: String, record: ModelCapabilities.Record)] {
        let directory = root.appending(path: ModelCapabilities.directory)
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: directory.path)
        else { return [] }
        return names.sorted().compactMap { name in
            guard name.hasSuffix(".json"),
                let record = try? read(directory.appending(component: name))
            else { return nil }
            return (ModelCapabilities.directory + "/" + name, record)
        }
    }

    /// The workspace record for (model id, revision), as an effective view:
    /// exact match first, else the model id's most recently probed record at
    /// any revision with an advisory naming the substitution, else nil.
    public static func lookup(modelID: String, revision: String?, root: URL) -> ModelCapabilities? {
        if let revision {
            let exact = recordURL(modelID: modelID, revision: revision, root: root)
            if FileManager.default.fileExists(atPath: exact.path),
                let record = try? read(exact)
            {
                return record.effective(
                    path: recordRelativePath(modelID: modelID, revision: revision))
            }
        }
        let candidates = list(root: root).filter { $0.record.modelID == modelID }
        guard !candidates.isEmpty else { return nil }
        if revision == nil, candidates.count == 1 {
            return candidates[0].record.effective(path: candidates[0].path)
        }
        let best = candidates.max { ($0.record.probedBy?.at ?? "") < ($1.record.probedBy?.at ?? "") }!
        return best.record.effective(path: best.path).withAdvisory(
            "no capability record for \(modelID) at revision \(revision ?? "unpinned"); "
                + "using the record for revision \(best.record.revision ?? "unpinned") "
                + "(\(best.path))")
    }

    /// The record a gate reads: the workspace record when one exists, else
    /// the heuristic (which carries its own advisory).
    public static func resolve(modelID: String, revision: String?, root: URL) -> ModelCapabilities {
        lookup(modelID: modelID, revision: revision, root: root)
            ?? ModelCapabilities.heuristicView(modelID: modelID, revision: revision)
    }

    /// One line per detected fact that changed between two records — what a
    /// re-probe after a template change prints.
    public static func diff(
        old: ModelCapabilities.Record, new: ModelCapabilities.Record
    ) -> [String] {
        var lines: [String] = []
        let before = old.detected.jsonObject
        let after = new.detected.jsonObject
        for key in Set(before.keys).union(after.keys).sorted() {
            let a = CanonicalJSON.encode(before[key] ?? NSNull())
            let b = CanonicalJSON.encode(after[key] ?? NSNull())
            if a != b { lines.append("\(key): \(a) → \(b)") }
        }
        if old.template?.sha256 != new.template?.sha256 {
            lines.insert(
                "template sha256: \(old.template?.sha256 ?? "None") → "
                    + "\(new.template?.sha256 ?? "None")", at: 0)
        }
        return lines
    }

    /// A copy of `record` with one override set (an empty value clears it).
    /// Server twin: `model_capabilities.set_override`.
    public static func setOverride(
        _ record: ModelCapabilities.Record, field: String, value: String, reason: String,
        setAt: String = ModelCapabilitiesProbe.now()
    ) throws -> ModelCapabilities.Record {
        guard let vocabulary = ModelCapabilities.overridableFields[field] else {
            throw ModelCapabilities.RecordError(
                "'\(field)' is not an overridable capability — one of "
                    + ModelCapabilities.overridableFields.keys.sorted().joined(separator: ", "))
        }
        var updated = record
        if value.isEmpty {
            updated.overrides.removeValue(forKey: field)
            return updated
        }
        let parsed: ModelCapabilities.OverrideValue
        if let allowed = vocabulary {
            guard allowed.contains(value) else {
                throw ModelCapabilities.RecordError(
                    "\(field) takes one of " + allowed.joined(separator: ", ") + " — got '\(value)'")
            }
            parsed = .string(value)
        } else {
            switch value.trimmingCharacters(in: .whitespaces).lowercased() {
            case "true": parsed = .boolean(true)
            case "false": parsed = .boolean(false)
            default:
                throw ModelCapabilities.RecordError("\(field) takes true or false — got '\(value)'")
            }
        }
        let trimmedReason = reason.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedReason.isEmpty else {
            throw ModelCapabilities.RecordError(
                "an override needs a --reason: it is displayed beside the detected "
                    + "value and stamped into every run")
        }
        updated.overrides[field] = .init(value: parsed, reason: trimmedReason, setAt: setAt)
        return updated
    }

    // MARK: - The engine probe (a cached snapshot in hand)

    /// The tokenizer bridge cannot render a template to TEXT losslessly for
    /// this model, so the probe cannot classify its renders.
    public struct SnapshotProbeError: Error, CustomStringConvertible {
        public let description: String
    }

    /// Probe the cached snapshot's tokenizer through swift-transformers'
    /// renderer — weights-free. The snapshot is the pinned revision's when
    /// the cache holds it, else the cache's `refs/main`.
    public static func probeSnapshot(
        modelID: String, revision: String?, engineVersion: String
    ) async throws -> ModelCapabilities.Record {
        guard
            let snapshot = SteeredContainerLoader.cachedLoadableSnapshot(
                modelID: modelID, revision: revision)
                ?? SteeredContainerLoader.cachedSnapshotDirectory(for: modelID)
        else {
            throw SnapshotProbeError(
                description: "this Mac holds no cached snapshot for \(modelID)"
                    + (revision.map { " at revision \($0)" } ?? ""))
        }
        let tokenizer = try await AutoTokenizer.from(modelFolder: snapshot)
        return try probe(
            tokenizer: tokenizer, snapshot: snapshot, modelID: modelID,
            revision: snapshot.lastPathComponent, engineVersion: engineVersion)
    }

    /// Probe a loaded swift-transformers tokenizer whose snapshot directory is
    /// known (the template and config hashes come from the files beside it).
    public static func probe(
        tokenizer: any Tokenizers.Tokenizer, snapshot: URL, modelID: String,
        revision: String?, engineVersion: String
    ) throws -> ModelCapabilities.Record {
        let render: ModelCapabilitiesProbe.Render = { messages, context in
            // The probe's context is strings and booleans only.
            var additional: [String: any Sendable] = [:]
            for (key, value) in context {
                if let text = value as? String { additional[key] = text }
                else if let flag = value as? Bool { additional[key] = flag }
            }
            let ids = try tokenizer.applyChatTemplate(
                messages: messages.map { $0 as [String: any Sendable] },
                chatTemplate: nil, addGenerationPrompt: true, truncation: false,
                maxLength: nil, tools: nil,
                additionalContext: additional.isEmpty ? nil : additional)
            let text = tokenizer.decode(tokens: ids, skipSpecialTokens: false)
            guard tokenizer.encode(text: text, addSpecialTokens: false) == ids else {
                throw SnapshotProbeError(
                    description: "the tokenizer's decode/encode round trip is not "
                        + "byte-stable for \(modelID), so its renders cannot be "
                        + "classified on this engine")
            }
            return text
        }
        return try ModelCapabilitiesProbe.probe(
            render: render, modelID: modelID, revision: revision,
            thinkTokenID: { token in
                let ids = tokenizer.encode(text: token, addSpecialTokens: false)
                return ids.count == 1 ? ids[0] : nil
            },
            architecture: architecture(snapshot: snapshot),
            templateSha256: templateSource(snapshot: snapshot).map(ModelCapabilities.templateSha256),
            tokenizerConfigSha256: (try? Data(
                contentsOf: snapshot.appending(component: "tokenizer_config.json")))
                .map(ModelCapabilities.sha256Hex),
            engineVersion: engineVersion)
    }

    /// The template source swift-transformers resolves, in its own order:
    /// the `chat_template.jinja` sidecar, then `chat_template.json`, then
    /// `tokenizer_config.json`'s `chat_template`.
    public static func templateSource(snapshot: URL) -> String? {
        let jinja = snapshot.appending(component: "chat_template.jinja")
        if let text = try? String(contentsOf: jinja, encoding: .utf8), !text.isEmpty {
            return text
        }
        let json = snapshot.appending(component: "chat_template.json")
        if let data = try? Data(contentsOf: json),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let text = object["chat_template"] as? String, !text.isEmpty
        {
            return text
        }
        guard
            let data = try? Data(contentsOf: snapshot.appending(component: "tokenizer_config.json")),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        if let text = object["chat_template"] as? String { return text }
        if let list = object["chat_template"] as? [Any], !list.isEmpty {
            return CanonicalJSON.encode(list)
        }
        return nil
    }

    /// Layer count, hidden size and per-layer attention types from the
    /// snapshot's `config.json`, reading the nested `text_config` of a
    /// multimodal repo when the top level has none. Server twin:
    /// `model_capabilities.architecture_from_config`.
    public static func architecture(snapshot: URL) -> ModelCapabilities.Architecture? {
        guard let data = try? Data(contentsOf: snapshot.appending(component: "config.json")),
            let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return architecture(config: config)
    }

    public static func architecture(config: [String: Any]) -> ModelCapabilities.Architecture? {
        let sources: [[String: Any]] = [config] + [(config["text_config"] as? [String: Any])].compactMap { $0 }
        func int(_ names: [String]) -> Int? {
            for source in sources {
                for name in names {
                    if let value = source[name] as? Int, value > 0 { return value }
                }
            }
            return nil
        }
        var layerTypes: [String]?
        for source in sources {
            if let types = source["layer_types"] as? [String], !types.isEmpty {
                layerTypes = types
                break
            }
        }
        let result = ModelCapabilities.Architecture(
            layerCount: int(["num_hidden_layers", "n_layer", "num_layers"]),
            hiddenSize: int(["hidden_size", "d_model", "n_embd"]),
            layerTypes: layerTypes)
        if result.layerCount == nil, result.hiddenSize == nil, result.layerTypes == nil {
            return nil
        }
        return result
    }

    /// The record a load reads AND leaves behind: probe the cached snapshot;
    /// write the workspace record when none exists; when one exists whose
    /// template hash differs from the live template, re-probe, say what moved
    /// and overwrite (overrides survive). Falls back to the heuristic — saying
    /// so — when this Mac holds no probeable snapshot. Every outcome is
    /// registered for `PromptRendering`, so the renders that follow read what
    /// the load found. Server twin: `model_capabilities.ensure_record`.
    public static func ensure(
        modelID: String, revision: String?, root: URL, engineVersion: String,
        log: (@Sendable (String) -> Void)? = nil
    ) async -> ModelCapabilities {
        let fresh: ModelCapabilities.Record
        do {
            fresh = try await probeSnapshot(
                modelID: modelID, revision: revision, engineVersion: engineVersion)
        } catch {
            let view = resolve(modelID: modelID, revision: revision, root: root)
                .withAdvisory(
                    "could not probe the chat template of \(modelID) on this Mac "
                        + "(\(ModelCapabilitiesProbe.message(of: error))); rendering "
                        + "under the \(lookup(modelID: modelID, revision: revision, root: root) == nil ? "id heuristic" : "workspace record")")
            for line in view.advisories { log?("ADVISORY: \(line)") }
            ModelCapabilities.register(view, for: modelID)
            return view
        }
        let url = recordURL(modelID: modelID, revision: fresh.revision, root: root)
        let relative = recordRelativePath(modelID: modelID, revision: fresh.revision)
        var toWrite = fresh
        if FileManager.default.fileExists(atPath: url.path) {
            if let existing = try? read(url) {
                if existing.template?.sha256 == fresh.template?.sha256 {
                    let view = existing.effective(path: relative)
                    ModelCapabilities.register(view, for: modelID)
                    return view
                }
                toWrite.overrides = existing.overrides
                log?(
                    "ADVISORY: the chat template of \(modelID) changed since \(relative) "
                        + "was probed — re-probed; "
                        + diff(old: existing, new: toWrite).joined(separator: "; "))
            } else {
                log?("ADVISORY: capability record \(relative) is unreadable; rewriting it")
            }
        }
        do {
            let written = try write(toWrite, root: root)
            log?("probed model capabilities for \(modelID) → \(written)")
            let view = toWrite.effective(path: written)
            ModelCapabilities.register(view, for: modelID)
            return view
        } catch {
            let view = toWrite.effective().withAdvisory(
                "could not write \(relative): \(error.localizedDescription)")
            for line in view.advisories { log?("ADVISORY: \(line)") }
            ModelCapabilities.register(view, for: modelID)
            return view
        }
    }

    // MARK: - Template-text rendering (fixtures, tests)

    /// Render a chat-template SOURCE the way swift-transformers does — its
    /// own Jinja engine, the template's special tokens as context — with no
    /// tokenizer. The cross-engine fixture's families render through this
    /// here and through jinja2 on the server, which is how the probe logic is
    /// pinned across engines without vendoring a model. Server twin:
    /// `model_capabilities.render_template_text`.
    public static func renderTemplateText(
        _ source: String, messages: [[String: String]], addGenerationPrompt: Bool = true,
        specialTokens: [String: String] = [:], context: [String: Any] = [:]
    ) throws -> String {
        let template = try Template(source, with: .init(lstripBlocks: true, trimBlocks: true))
        var variables: [String: Value] = [
            "messages": .array(try messages.map { try Value(any: $0 as [String: Any]) }),
            "add_generation_prompt": .boolean(addGenerationPrompt),
        ]
        for (key, value) in specialTokens { variables[key] = .string(value) }
        for (key, value) in context { variables[key] = try Value(any: value) }
        return try template.render(variables)
    }

    public static func templateRenderer(
        _ source: String, specialTokens: [String: String] = [:]
    ) -> ModelCapabilitiesProbe.Render {
        { messages, context in
            try renderTemplateText(
                source, messages: messages, specialTokens: specialTokens, context: context)
        }
    }
}
