import CryptoKit
import Foundation
import SteeringKit

/// `agent create`: an agent a researcher makes by hand (public issue #17).
///
/// A base model at a pinned revision, at most one fine-tuned adapter, any
/// number of steering vectors at the layers and strengths the researcher
/// chooses (or none), a thinking level, a system prompt, and a prompt mode.
/// The verb previews first and writes nothing: it checks every input, shows
/// the agent file it would write, and returns `reviewSHA256`. The same
/// command repeated with `--review-sha256` writes that file into the agent
/// library (`runs/model-variants/`) through `ModelVariantStore.save`, the
/// write the app's agent editor uses, and returns `artifactFileSHA256` for
/// `experiment attach-agent` and panel casting.
///
/// The file carries a `handmade` record: the command, each input and its
/// hash, when it was made, and the stated reason. There is no
/// `--temperature`: under the 2026-11-08 ruling an agent's temperature is
/// provenance only, and the study that casts the agent sets the temperature
/// every arm runs at.
///
/// Same verb, flags, checks, and sentences as the Python client
/// (`steerlab agent create`, `client/agent_creation.py`).
public enum StudyAgentCreation {

    /// The flags that make up one agent, in the order the recorded command
    /// lists them. `--review-sha256` is the reviewed write.
    public static let definingFlags = [
        "--model", "--revision", "--adapter", "--adapter-sha256", "--vector",
        "--vector-sha256", "--layer", "--strength", "--mode", "--reasoning-effort",
        "--system-prompt-file", "--prompt-mode", "--reason",
    ]

    public static let spec = ExperimentCLIVerbSpec(
        namespace: "agent", verb: "create", positional: "<name>",
        purpose: "Preview an agent made by hand: a base model at a pinned revision, at "
            + "most one adapter, steering vectors at the layers and strengths you choose "
            + "(in residual-norm units), a thinking level, a system prompt, and a prompt "
            + "mode. Repeat with --review-sha256 to write it into the agent library with "
            + "a record of how it was made. The study that casts the agent sets its "
            + "temperature.",
        valueFlags: Set(definingFlags + ["--review-sha256"]),
        requiredFlags: ["--model", "--revision"])

    /// What the preview found: the file the write would produce (without its
    /// timestamps), the digest that authorizes the write, and notes.
    public struct Plan: Sendable {
        public var artifact: ModelVariantArtifact
        public var reviewSHA256: String
        public var advisories: [String]
        public var effort: ReasoningEffort
    }

    private static var program: String { ExperimentCLIHelp.program }

    // MARK: - Running the verb

    static func run(
        _ invocation: ExperimentCLIInvocation, workspaceRoot: URL, sink: ExperimentCLISink
    ) throws -> ExperimentCLIResult {
        let (name, flags) = try parse(Array(invocation.args.dropFirst()))
        var command = ["agent", "create", name]
        for flag in definingFlags {
            for value in flags[flag] ?? [] { command += [flag, value] }
        }
        let root = URL(fileURLWithPath: try ManifestFileTransaction.canonicalPath(workspaceRoot))
        let planned = try plan(name: name, flags: flags, workspaceRoot: root,
                               command: [program] + command)
        let advisories = planned.advisories.map {
            SteerLabCLIEnvelope.Advisory(code: "agentCreateNote", detail: $0)
        }
        var payload: [String: JSONValue] = [
            "document": try jsonValue(planned.artifact),
            "reviewSHA256": .string(planned.reviewSHA256),
            "origin": try jsonValue(planned.artifact.origin),
            "reasoningEffort": .string(planned.effort.rawValue),
            "advisories": .array(planned.advisories.map(JSONValue.string)),
        ]
        guard let expected = flags["--review-sha256"]?.last else {
            let confirm = (command + ["--review-sha256", planned.reviewSHA256])
                .map(WorkspaceHousekeepingCLI.shellQuoted).joined(separator: " ")
            payload["written"] = .bool(false)
            payload["confirmCommand"] = .string("\(program) \(confirm)")
            sink.out("Preview — nothing has been written yet.")
            sink.out("  " + planned.artifact.origin.summary)
            for note in planned.advisories { sink.out("  note: " + note) }
            sink.out("To write it, after the researcher agrees: \(program) \(confirm)")
            return ExperimentCLIResult(
                message: "Preview of agent '\(planned.artifact.name)'; nothing was written.",
                changed: false, payload: payload, advisories: advisories,
                nextAction: .init(
                    verb: confirm, missingPermissionFlags: ["--review-sha256"],
                    detail: "Show the researcher the agent file and write it only after "
                        + "they agree."))
        }
        guard expected == planned.reviewSHA256 else {
            throw ExperimentError.refusing(
                .staleManifest,
                "An input changed after the preview: the agent this command would write is "
                    + "not the one you reviewed.",
                repair: "Preview again without --review-sha256, review the agent file, and "
                    + "use the new reviewSHA256.")
        }
        let record = try write(planned, workspaceRoot: root)
        let data = try Data(contentsOf: record.url)
        let prefix = root.path + "/"
        let path = try ManifestFileTransaction.canonicalPath(record.url)
        let relative = path.hasPrefix(prefix) ? String(path.dropFirst(prefix.count)) : path
        payload["document"] = try JSONDecoder().decode(JSONValue.self, from: data)
        payload["origin"] = try jsonValue(record.artifact.origin)
        payload["path"] = .string(relative)
        payload["artifactFileSHA256"] = .string(ManifestFileTransaction.digest(data))
        payload["written"] = .bool(true)
        sink.out("Wrote agent '\(record.artifact.name)' to \(relative).")
        return ExperimentCLIResult(
            message: "Wrote agent '\(record.artifact.name)'. Attach it with experiment "
                + "attach-agent and its artifactFileSHA256.",
            changed: true, payload: payload, advisories: advisories)
    }

    /// The positional name and every flag's values in the order given.
    static func parse(_ arguments: [String]) throws -> (String, [String: [String]]) {
        var positionals: [String] = []
        var flags: [String: [String]] = [:]
        var index = 0
        while index < arguments.count {
            let token = arguments[index]
            if token.hasPrefix("--"), index + 1 < arguments.count {
                flags[token, default: []].append(arguments[index + 1])
                index += 2
            } else {
                positionals.append(token)
                index += 1
            }
        }
        guard positionals.count == 1, flags["--model"] != nil, flags["--revision"] != nil else {
            throw usage("Name the agent and give --model and --revision.",
                "\(program) agent create <name> --model <id> --revision <sha>")
        }
        return (positionals[0], flags)
    }

    // MARK: - The plan

    /// Every check, and the agent file the write would produce (minus its
    /// timestamps). Writes nothing.
    public static func plan(
        name rawName: String, flags: [String: [String]], workspaceRoot root: URL,
        command: [String]
    ) throws -> Plan {
        func one(_ flag: String) -> String? { flags[flag]?.last }
        for flag in ["--model", "--revision", "--reasoning-effort", "--system-prompt-file",
                     "--prompt-mode", "--reason", "--review-sha256"]
        where (flags[flag]?.count ?? 0) > 1 {
            throw usage("Give \(flag) once.", "\(program) agent create --help")
        }
        let name = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, !name.contains("/"), !name.contains("\0"),
            name.unicodeScalars.contains(where: CharacterSet.alphanumerics.contains)
        else {
            throw usage("Give the agent a name with letters or digits and no slash.",
                "\(program) agent create <name> --model <id> --revision <sha>")
        }
        let model = (one("--model") ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let revision = (one("--revision") ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !model.isEmpty else {
            throw usage("Give the base model's id with --model.", "\(program) agent create --help")
        }
        guard isHex(revision, count: 40) else {
            throw ExperimentError.refusing(
                .artifactPin,
                "--revision \(revision) is not a pinned commit: an agent names the exact "
                    + "weights it runs, as a full 40-character commit hash, not a branch or a "
                    + "tag that can move.",
                repair: "Give the full 40-character commit hash of the revision (\(program) "
                    + "model plan \(model) --json shows the revision a prepared model "
                    + "resolves to).")
        }
        var notes: [String] = []

        let adapters = flags["--adapter"] ?? []
        let adapterDigests = flags["--adapter-sha256"] ?? []
        guard adapters.count == adapterDigests.count else {
            throw usage("Give each --adapter its own --adapter-sha256.",
                "\(program) agent create <name> ... --adapter <path> --adapter-sha256 <digest>")
        }
        guard adapters.count <= 1 else {
            throw ExperimentError.refusing(
                .studyDeclaration,
                "An agent runs one adapter: both engines load the first adapter an agent "
                    + "lists and would ignore the rest, so a second adapter would be recorded "
                    + "but never run.",
                repair: "Give one --adapter. To combine adapters, train or merge them into one "
                    + "first.")
        }
        for digest in adapterDigests + (flags["--vector-sha256"] ?? []) where !isHex(digest, count: 64) {
            throw usage("\(digest) is not a lowercase SHA-256 digest.",
                "Give each digest as 64 lowercase hexadecimal characters.")
        }
        var adapterRefs: [ModelVariantArtifact.AdapterRef] = []
        var adapterInputs: [JSONValue] = []
        for (reference, digest) in zip(adapters, adapterDigests) {
            let (ref, recorded) = try adapter(
                root: root, reference: reference, expected: digest, model: model,
                revision: revision, notes: &notes)
            adapterRefs.append(ref)
            adapterInputs.append(recorded)
        }

        var injections: [ModelVariantArtifact.InjectionRef] = []
        var vectorInputs: [JSONValue] = []
        for spec in try grouped(flags) {
            let (injection, recorded) = try vector(
                root: root, spec: spec, model: model, revision: revision, notes: &notes)
            injections.append(injection)
            vectorInputs.append(recorded)
        }

        let promptMode = one("--prompt-mode") ?? "chatAssistant"
        guard ["chatAssistant", "rawCompletion"].contains(promptMode) else {
            throw usage("--prompt-mode \(promptMode) is not chatAssistant or rawCompletion.",
                "Use --prompt-mode chatAssistant (the chat template) or rawCompletion.")
        }

        let spelledEffort = one("--reasoning-effort") ?? ReasoningEffort.off.rawValue
        guard let effort = ReasoningEffort(rawValue: spelledEffort) else {
            throw usage(ReasoningEffort.unknownEffortReason(spelledEffort),
                "\(program) agent create <name> ... --reasoning-effort "
                    + "<off|on|low|medium|high|xhigh>")
        }
        let capabilities = ModelCapabilitiesStore.resolve(
            modelID: model, revision: revision, root: root)
        var probe = ModelVariantArtifact(
            name: name, baseModelID: model, promptMode: promptMode, qwenThinkingEnabled: false,
            temperature: 0, systemPrompt: "", reasoningEffort: effort)
        let problems = probe.effortViolations(capabilities: capabilities)
        guard problems.isEmpty else {
            let accepted = capabilities.acceptedEfforts.joined(separator: ", ")
            throw ExperimentError.refusing(
                .studyDeclaration, problems.joined(separator: "; "),
                repair: capabilities.hasThinkingSwitch
                    ? "Choose a level the model's chat template accepts (accepted: "
                        + (accepted.isEmpty ? "none" : accepted) + ") with --reasoning-effort, "
                        + "or use on (thinking at the template's default) or off."
                    : "Use --reasoning-effort off: this model's chat template has no thinking "
                        + "switch.")
        }
        notes += ReasoningEffort.protocolAdvisories(
            effort: effort, modelID: model, capabilities: capabilities)

        var systemPrompt = ""
        var systemInput: JSONValue = .null
        if let file = one("--system-prompt-file") {
            guard let data = FileManager.default.contents(atPath: file),
                let text = String(data: data, encoding: .utf8)
            else {
                throw ExperimentError.refusing(
                    .missingPrerequisite,
                    "The system prompt file \(file) could not be read as UTF-8 text.",
                    repair: "Check the file, then preview again.")
            }
            guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ExperimentError.refusing(
                    .missingPrerequisite, "The system prompt file \(file) is empty.",
                    repair: "Write the agent's system prompt into the file, or leave out "
                        + "--system-prompt-file.")
            }
            guard promptMode != "rawCompletion" else {
                throw ExperimentError.refusing(
                    .studyDeclaration,
                    "A system prompt reaches the model through the chat template, and "
                        + "--prompt-mode rawCompletion uses none, so it would never be read.",
                    repair: "Use --prompt-mode chatAssistant, or leave out --system-prompt-file.")
            }
            let violations = PromptRendering.systemPromptViolations(
                systemPrompt: text, modelID: model, rawCompletion: false,
                capabilities: capabilities)
            guard violations.isEmpty else {
                throw ExperimentError.refusing(
                    .studyDeclaration, violations.joined(separator: "; "),
                    repair: "Leave out --system-prompt-file for this model.")
            }
            systemPrompt = text
            systemInput = .object(["path": .string(file),
                                   "sha256": .string(ManifestFileTransaction.digest(data))])
        }

        let taken = Set(try StudyAgentAuthoring.list(workspaceRoot: root).agents.map(\.artifact.name))
        guard !taken.contains(name) else {
            throw ExperimentError.refusing(
                .studyDeclaration,
                "An agent named \(name) already exists in this workspace, and a study tells "
                    + "its agents apart by name.",
                repair: "Choose another name (\(program) agent list --json lists the names in "
                    + "use).")
        }

        let reason = one("--reason").flatMap {
            $0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : $0
        }
        let record: JSONValue = .object([
            "schemaVersion": .number(1), "madeWith": .string("agent create"),
            "client": .string(program), "command": .array(command.map(JSONValue.string)),
            "reason": reason.map(JSONValue.string) ?? .null,
            "inputs": .object([
                "model": .object(["id": .string(model), "revision": .string(revision)]),
                "adapters": .array(adapterInputs), "vectors": .array(vectorInputs),
                "reasoningEffort": .string(effort.rawValue), "promptMode": .string(promptMode),
                "systemPromptFile": systemInput,
                "capabilityRecordSource": .string(capabilities.source.rawValue),
            ]),
        ])
        probe = ModelVariantArtifact(
            name: name, baseModelID: model, baseRevision: revision, adapters: adapterRefs,
            injections: injections, bandWidth: 1, alphaInNormUnits: true,
            promptMode: promptMode, qwenThinkingEnabled: effort.isOn,
            // The schema needs a number here; it is the Playground default and
            // is never the temperature a study runs the agent at.
            temperature: 0, systemPrompt: systemPrompt,
            reasoningEffort: effort, handmade: record)
        probe.createdAt = ""
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return Plan(
            artifact: probe,
            reviewSHA256: ManifestFileTransaction.digest(try encoder.encode(probe)),
            advisories: notes, effort: effort)
    }

    /// Publishes the reviewed agent as a new library folder, through the
    /// write the app's editor uses; never replaces one.
    static func write(_ planned: Plan, workspaceRoot root: URL) throws -> ModelVariantRecord {
        var artifact = planned.artifact
        let stamp = ISO8601DateFormatter().string(from: Date())
        artifact.createdAt = stamp
        if case .object(var record)? = artifact.handmade {
            record["createdAt"] = .string(stamp)
            artifact.handmade = .object(record)
        }
        return try ModelVariantStore.save(
            artifact, library: root.appending(components: "runs", "model-variants"))
    }

    // MARK: - The inputs

    struct VectorSpec {
        var reference: String
        var sha256: String
        var layer: Int
        var strength: Double
        var mode: String
    }

    /// The k-th --vector-sha256, --layer, --strength, and --mode belong to
    /// the k-th --vector. --mode may be left out for every vector (each one
    /// steers).
    static func grouped(_ flags: [String: [String]]) throws -> [VectorSpec] {
        let vectors = flags["--vector"] ?? []
        let digests = flags["--vector-sha256"] ?? []
        let layers = flags["--layer"] ?? []
        let strengths = flags["--strength"] ?? []
        let modes = flags["--mode"] ?? []
        guard digests.count == vectors.count, layers.count == vectors.count,
            strengths.count == vectors.count, modes.isEmpty || modes.count == vectors.count
        else {
            throw usage(
                "Give each --vector its own --vector-sha256, --layer, and --strength (and "
                    + "--mode for each vector, or for none).",
                "\(program) agent create <name> --model <id> --revision <sha> --vector "
                    + "<artifact> --vector-sha256 <digest> --layer <n> --strength <alpha> "
                    + "[--mode add|ablate]")
        }
        return try vectors.indices.map { index in
            guard let layer = Int(layers[index]) else {
                throw usage("--layer \(layers[index]) is not a whole number.",
                    "Give the layer as a whole number, such as --layer 14.")
            }
            guard let strength = Double(strengths[index]), strength.isFinite else {
                throw usage("--strength \(strengths[index]) is not a number.",
                    "Give the strength as a number in residual-norm units, such as "
                        + "--strength 0.12.")
            }
            let mode = modes.isEmpty ? "add" : modes[index]
            guard mode == "add" || mode == "ablate" else {
                throw usage("--mode \(mode) is not add or ablate.",
                    "Use --mode add to steer along the vector, or --mode ablate to remove it.")
            }
            return VectorSpec(reference: vectors[index], sha256: digests[index], layer: layer,
                              strength: strength, mode: mode)
        }
    }

    static func vector(
        root: URL, spec: VectorSpec, model: String, revision: String, notes: inout [String]
    ) throws -> (ModelVariantArtifact.InjectionRef, JSONValue) {
        var reference = spec.reference.trimmingCharacters(in: .whitespacesAndNewlines)
        for suffix in [".safetensors", ".json"] where reference.hasSuffix(suffix) {
            reference = String(reference.dropLast(suffix.count))
        }
        let tensor = try ordinary(root, reference + ".safetensors", what: "vector")
        let sidecarURL = try ordinary(root, reference + ".json", what: "vector's sidecar")
        guard isFile(tensor), isFile(sidecarURL) else {
            throw ExperimentError.refusing(
                .missingPrerequisite,
                "The vector \(reference) needs both \(reference).safetensors and "
                    + "\(reference).json.",
                repair: "Check the path with experiment inspect-artifact.")
        }
        let live = ManifestFileTransaction.digest(try Data(contentsOf: tensor))
        guard live == spec.sha256 else {
            throw ExperimentError.refusing(
                .artifactPin,
                "The vector \(reference) has SHA-256 \(live), not the \(spec.sha256) given "
                    + "with --vector-sha256: the file is not the one you reviewed.",
                repair: "Read the vector's current digest with \(program) experiment "
                    + "inspect-artifact \(reference) --json, review it, and give that "
                    + "artifactSHA256.")
        }
        let sidecarData = try Data(contentsOf: sidecarURL)
        let sidecar = try jsonObject(sidecarData, name: sidecarURL.lastPathComponent,
                                     what: "vector sidecar")
        let extractedFrom = sidecar["modelID"] as? String
        guard extractedFrom == model else {
            throw ExperimentError.refusing(
                .artifactPin,
                "The vector \(reference) was extracted from \(extractedFrom ?? "None"), not "
                    + "\(model): steering one model with another model's vector has no meaning.",
                repair: "Choose a vector extracted from the agent's base model.")
        }
        if let recorded = sidecar["revision"] as? String {
            guard recorded == revision else {
                throw ExperimentError.refusing(
                    .artifactPin,
                    "The vector \(reference) was extracted from revision \(recorded) of "
                        + "\(model), not \(revision).",
                    repair: "Use the vector's revision with --revision, or extract the concept "
                        + "again at this revision.")
            }
        } else {
            notes.append("The vector \(reference) does not record which revision of \(model) "
                + "it was extracted from, so SteerLab cannot confirm it matches \(revision).")
        }
        let count = (sidecar["layerCount"] as? NSNumber)?.intValue
        guard let count, (0..<count).contains(spec.layer) else {
            throw ExperimentError.refusing(
                .artifactPin,
                "Layer \(spec.layer) is not in the vector \(reference), which has layers 0 to "
                    + "\(count.map { String($0 - 1) } ?? "?").",
                repair: "Choose a layer the vector has, with --layer.")
        }
        if let norms = sidecar["normsPerLayer"] as? [NSNumber], spec.layer < norms.count,
            norms[spec.layer].doubleValue == 0
        {
            throw ExperimentError.refusing(
                .artifactPin,
                "The vector \(reference) is empty at layer \(spec.layer), so steering there "
                    + "would change nothing.",
                repair: "Choose a layer where the vector has a direction (its sidecar's "
                    + "normsPerLayer is not zero there).")
        }
        let concept = (sidecar["concept"] as? String).flatMap { $0.isEmpty ? nil : $0 }
            ?? URL(fileURLWithPath: reference).lastPathComponent
        if spec.mode == "add" {
            let norms = (sidecar["residualNormPerLayer"] as? [NSNumber])?.map(\.floatValue)
            if ResidualNormConvention.residualNormProblem(
                norms, layer: spec.layer, artifact: concept) != nil
            {
                throw ExperimentError.refusing(
                    .artifactPin,
                    "The vector \(reference) records no residual norm at layer \(spec.layer), so "
                        + "a strength in residual-norm units has nothing to be measured against.",
                    repair: "Measure the vector's residual norms first (steerlab-cli vectors "
                        + "backfill-norms on the Mac, or steerlab-server vectors backfill-norms "
                        + "where the model is installed), then preview again.")
            }
        }
        let covers = sidecar["coversModelDepth"] as? Bool
        let method = sidecar["extractionMethod"] as? String ?? ""
        if !(covers ?? !SteeringVectorSidecar.partialDepthMethods.contains(method)) {
            notes.append("The vector \(reference) covers only some of the model's layers; "
                + "layer \(spec.layer) is one it has.")
        }
        let injection = ModelVariantArtifact.InjectionRef(
            concept: concept, vectorArtifactID: reference, layer: spec.layer,
            alpha: spec.strength, mode: spec.mode == "ablate" ? .ablate : nil)
        return (injection, .object([
            "artifactID": .string(reference), "artifactSHA256": .string(live),
            "sidecarSHA256": .string(ManifestFileTransaction.digest(sidecarData)),
            "layer": .number(Double(spec.layer)), "strength": .number(spec.strength),
            "mode": .string(spec.mode),
        ]))
    }

    private static let weightsByFormat = [
        "hf-peft-lora": "adapter_model.safetensors", "peft": "adapter_model.safetensors",
        "mlx-lora": "adapters.safetensors",
    ]
    private static let weightFiles = ["adapters.safetensors", "adapter_model.safetensors"]

    static func adapter(
        root: URL, reference: String, expected: String, model: String, revision: String,
        notes: inout [String]
    ) throws -> (ModelVariantArtifact.AdapterRef, JSONValue) {
        let path = try ordinary(root, reference, what: "adapter")
        let recordURL: URL
        let record: [String: Any]
        let directory: URL
        let relative: String
        let name: String
        let declaredRevision: String?
        let declaredWeights: String?
        let declaredConfig: String?
        if isFile(path), path.lastPathComponent == "fine-tune.json" {
            recordURL = path
            record = try jsonObject(try Data(contentsOf: path), name: path.lastPathComponent,
                                    what: "adapter record")
            let declared = record["adapterDirectory"] as? String ?? ""
            let resolved = declared.hasPrefix("/")
                ? URL(fileURLWithPath: declared) : root.appending(path: declared)
            let canonical = (try? ManifestFileTransaction.canonicalPath(resolved)) ?? resolved.path
            guard canonical.hasPrefix(root.path + "/") else {
                throw ExperimentError.refusing(
                    .missingPrerequisite,
                    "The adapter record \(reference) names a folder outside this workspace "
                        + "(\(declared)).",
                    repair: "Import the adapter into this workspace, then name its record.")
            }
            relative = String(canonical.dropFirst(root.path.count + 1))
            directory = try ordinary(root, relative, what: "adapter folder")
            name = (record["name"] as? String).flatMap { $0.isEmpty ? nil : $0 }
                ?? directory.lastPathComponent
            declaredRevision = record["baseRevision"] as? String
            declaredWeights = record["adapterHash"] as? String
            declaredConfig = record["configHash"] as? String
        } else if isDirectory(path) {
            directory = path
            relative = reference.trimmingCharacters(in: CharacterSet(charactersIn: "/ "))
            recordURL = path.deletingLastPathComponent()
                .appending(component: path.lastPathComponent + ".json")
            guard isFile(recordURL) else {
                throw ExperimentError.refusing(
                    .missingPrerequisite,
                    "SteerLab cannot tell which base model the adapter \(reference) was "
                        + "trained on: its folder has no training record beside it "
                        + "(\(path.lastPathComponent).json).",
                    repair: "Name the adapter's record instead (the fine-tune.json the app "
                        + "wrote), or restore the training record beside the folder.")
            }
            record = try jsonObject(try Data(contentsOf: recordURL),
                                    name: recordURL.lastPathComponent, what: "adapter record")
            name = (record["name"] as? String).flatMap { $0.isEmpty ? nil : $0 }
                ?? path.lastPathComponent
            declaredRevision = record["revision"] as? String
            declaredWeights = record["adapterBytesHash"] as? String
            declaredConfig = record["adapterConfigHash"] as? String
        } else {
            throw ExperimentError.refusing(
                .missingPrerequisite,
                "The adapter \(reference) is neither an adapter folder nor its fine-tune.json "
                    + "record.",
                repair: "Name the adapter's folder, or the fine-tune.json record the app wrote "
                    + "for it.")
        }
        let trainedOn = record["baseModelID"] as? String
        guard trainedOn == model else {
            throw ExperimentError.refusing(
                .artifactPin,
                "The adapter \(reference) was trained on \(trainedOn ?? "None"), not \(model): "
                    + "an adapter only fits the model it was trained on.",
                repair: "Choose an adapter trained on the agent's base model.")
        }
        if let declaredRevision {
            guard declaredRevision == revision else {
                throw ExperimentError.refusing(
                    .artifactPin,
                    "The adapter \(reference) was trained on revision \(declaredRevision) of "
                        + "\(model), not \(revision).",
                    repair: "Use the adapter's revision with --revision, or train it again at "
                        + "this revision.")
            }
        } else {
            notes.append("The adapter \(reference) does not record which revision of \(model) "
                + "it was trained on, so SteerLab cannot confirm it matches \(revision).")
        }
        let present = weightFiles.filter { isFile(directory.appending(component: $0)) }
        let declaredFile = (record["adapterFormat"] as? String).flatMap { weightsByFormat[$0] }
        let weightsName: String
        if let declaredFile, present.contains(declaredFile) {
            weightsName = declaredFile
        } else if declaredFile == nil, present.count == 1 {
            weightsName = present[0]
        } else {
            throw ExperimentError.refusing(
                .artifactPin,
                "SteerLab cannot tell which weights file in the adapter folder "
                    + "\(directory.lastPathComponent) the model would load (found: "
                    + (present.isEmpty ? "none" : present.joined(separator: ", ")) + ").",
                repair: "Restore the adapter folder from its training run, then preview again.")
        }
        let weights = ManifestFileTransaction.digest(
            try Data(contentsOf: directory.appending(component: weightsName)))
        let configURL = directory.appending(component: "adapter_config.json")
        guard isFile(configURL) else {
            throw ExperimentError.refusing(
                .missingPrerequisite, "The adapter folder \(relative) has no adapter_config.json.",
                repair: "Restore the adapter folder from its training run, then preview again.")
        }
        let config = ManifestFileTransaction.digest(try Data(contentsOf: configURL))
        guard expected == weights else {
            throw ExperimentError.refusing(
                .artifactPin,
                "The adapter \(reference) has weights SHA-256 \(weights) (\(weightsName)), not "
                    + "the \(expected) given with --adapter-sha256: the weights are not the ones "
                    + "you reviewed.",
                repair: "Hash the adapter's \(weightsName) yourself (shasum -a 256), review it, "
                    + "and give that digest.")
        }
        for (claimed, live, what) in [(declaredWeights, weights, weightsName),
                                      (declaredConfig, config, "adapter_config.json")]
        where (claimed ?? "").isEmpty == false && claimed != live {
            throw ExperimentError.refusing(
                .artifactPin,
                "The adapter \(reference)'s \(what) no longer matches the hash its training "
                    + "record wrote: the adapter changed after training.",
                repair: "Restore the adapter folder from its training run, or train it again.")
        }
        let recordPath = try relativePath(recordURL, root: root)
        let ref = ModelVariantArtifact.AdapterRef(
            name: name, artifactPath: recordPath, adapterDirectory: relative,
            adapterHash: weights, configHash: config)
        return (ref, .object([
            "path": .string(reference), "adapterDirectory": .string(relative),
            "weightsFile": .string(weightsName), "adapterSHA256": .string(weights),
            "configSHA256": .string(config),
            "recordSHA256": .string(ManifestFileTransaction.digest(try Data(contentsOf: recordURL))),
        ]))
    }

    // MARK: - Paths and small helpers

    /// A workspace path that is an ordinary file or folder, named relative to
    /// the workspace and reached without a link.
    static func ordinary(_ root: URL, _ relative: String, what: String) throws -> URL {
        let trimmed = relative.trimmingCharacters(in: .whitespaces)
        let components = trimmed.split(separator: "/", omittingEmptySubsequences: true).map(String.init)
        guard !trimmed.isEmpty, !trimmed.hasPrefix("/"), !trimmed.contains("\0"),
            !components.isEmpty, !components.contains("..")
        else {
            throw ExperimentError.refusing(
                .missingPrerequisite,
                "The \(what) must be named by a path inside this workspace, such as "
                    + "runs/<run>/<name>; got \"\(relative)\".",
                repair: "Use the workspace-relative path that agent list, experiment "
                    + "inspect-artifact, or the adapter's folder shows.")
        }
        var url = root
        for (index, component) in components.enumerated() {
            url.append(component: component)
            let type = try? FileManager.default.attributesOfItem(atPath: url.path)[.type]
                as? FileAttributeType
            let last = index == components.count - 1
            guard type == .typeDirectory || (last && type == .typeRegular) else {
                throw ExperimentError.refusing(
                    .missingPrerequisite,
                    "The \(what) \(trimmed) is not an ordinary file or folder in this workspace "
                        + "(it is missing, or it is reached through a link).",
                    repair: "Check the path with agent list or experiment inspect-artifact, "
                        + "then preview again.")
            }
        }
        return url
    }

    private static func relativePath(_ url: URL, root: URL) throws -> String {
        let path = try ManifestFileTransaction.canonicalPath(url)
        return path.hasPrefix(root.path + "/") ? String(path.dropFirst(root.path.count + 1)) : path
    }

    private static func isFile(_ url: URL) -> Bool {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.type] as? FileAttributeType)
            == .typeRegular
    }

    private static func isDirectory(_ url: URL) -> Bool {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[.type] as? FileAttributeType)
            == .typeDirectory
    }

    private static func jsonObject(_ data: Data, name: String, what: String) throws -> [String: Any] {
        guard let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ExperimentError.refusing(
                .artifactPin, "The \(what) \(name) is not a readable JSON object.",
                repair: "Restore the file from its source, then preview again.")
        }
        return object
    }

    private static func isHex(_ value: String, count: Int) -> Bool {
        value.utf8.count == count
            && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
    }

    private static func jsonValue(_ value: some Encodable) throws -> JSONValue {
        try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(value))
    }

    private static func usage(_ reason: String, _ repair: String) -> ExperimentError {
        .malformed(reason, repair: repair)
    }
}
