import CryptoKit
import Foundation
import SteeringKit

/// The pinned cross-engine contract for adapter provenance: adapter sidecars
/// stamp WHICH engine trained them (`substrate`) and WHAT the weights are
/// (`adapterFormat`), because a LoRA directory is engine-specific in a way a
/// steering vector is not — MLX adapter weights do not load through PEFT and
/// vice versa. This engine writes `RepEReader.substrate` ("swift-mlx") +
/// "mlx-lora"; the Python server writes "python-hf-transformers" +
/// "hf-peft-lora". Absent stamps are legacy/unknown and behave as before
/// (loadable); only an EXPLICITLY foreign stamp is refused.
public enum AdapterSubstrateGate {
    /// This engine's adapter weight format stamp.
    public static let localAdapterFormat = "mlx-lora"
    /// This engine's substrate stamp — the same constant vector sidecars use.
    public static let localSubstrate = RepEReader.substrate

    /// True when the sidecar is stamped for ANOTHER engine — either field
    /// present and different refuses (a half-stamped foreign record must not
    /// slip through on the missing half). Unstamped (both nil) is unknown,
    /// not foreign.
    public static func isExplicitlyForeign(
        substrate: String?, adapterFormat: String?
    ) -> Bool {
        if let substrate, substrate != localSubstrate { return true }
        if let adapterFormat, adapterFormat != localAdapterFormat { return true }
        return false
    }

    /// Refusal for applying a foreign-stamped adapter on this engine; nil
    /// when the adapter may load (locally stamped or unstamped-legacy).
    public static func refusalMessage(
        name: String, substrate: String?, adapterFormat: String?
    ) -> String? {
        guard isExplicitlyForeign(substrate: substrate, adapterFormat: adapterFormat) else {
            return nil
        }
        return "adapter '\(name)' was trained as "
            + "'\(adapterFormat ?? "unstamped format")' on "
            + "'\(substrate ?? "unstamped substrate")'; this engine loads "
            + "\(localAdapterFormat) adapters — retrain on this substrate"
    }
}

public struct FineTuneArtifact: Codable, Sendable, Equatable {
    public var schemaVersion: Int
    public var name: String
    public var baseModelID: String
    public var baseRevision: String?
    public var adapterDirectory: String
    public var adapterHash: String?
    public var configHash: String?
    /// Engine that trained/saved this adapter (`AdapterSubstrateGate` — this
    /// engine stamps `RepEReader.substrate`, the server stamps
    /// "python-hf-transformers"). Optional: legacy sidecars carry no stamp.
    public var substrate: String?
    /// Adapter weight format ("mlx-lora" here, "hf-peft-lora" on the server).
    public var adapterFormat: String?
    public var fineTuneType: String
    public var rank: Int
    public var scale: Float
    public var adaptedLayers: Int
    public var trainingWorkspacePath: String?
    public var trainingDataPath: String?
    public var trainingDataHash: String?
    public var validationDataPath: String?
    public var validationDataHash: String?
    public var trainingMode: String?
    public var batchSize: Int
    public var iterations: Int
    public var learningRate: Double
    public var createdAt: String
    public var notes: String

    public init(
        name: String,
        baseModelID: String,
        baseRevision: String? = nil,
        adapterDirectory: String,
        adapterHash: String? = nil,
        configHash: String? = nil,
        substrate: String? = nil,
        adapterFormat: String? = nil,
        fineTuneType: String = "lora",
        rank: Int = 8,
        scale: Float = 10,
        adaptedLayers: Int = 16,
        trainingWorkspacePath: String? = nil,
        trainingDataPath: String? = nil,
        trainingDataHash: String? = nil,
        validationDataPath: String? = nil,
        validationDataHash: String? = nil,
        trainingMode: String? = nil,
        batchSize: Int = 4,
        iterations: Int = 1000,
        learningRate: Double = 1e-5,
        createdAt: Date = Date(),
        notes: String = ""
    ) {
        self.schemaVersion = 1
        self.name = name
        self.baseModelID = baseModelID
        self.baseRevision = baseRevision
        self.adapterDirectory = adapterDirectory
        self.adapterHash = adapterHash
        self.configHash = configHash
        self.substrate = substrate
        self.adapterFormat = adapterFormat
        self.fineTuneType = fineTuneType
        self.rank = rank
        self.scale = scale
        self.adaptedLayers = adaptedLayers
        self.trainingWorkspacePath = trainingWorkspacePath
        self.trainingDataPath = trainingDataPath
        self.trainingDataHash = trainingDataHash
        self.validationDataPath = validationDataPath
        self.validationDataHash = validationDataHash
        self.trainingMode = trainingMode
        self.batchSize = batchSize
        self.iterations = iterations
        self.learningRate = learningRate
        self.createdAt = ISO8601DateFormatter().string(from: createdAt)
        self.notes = notes
    }
}

public struct FineTuneArtifactRecord: Identifiable, Sendable {
    public let url: URL
    public let artifact: FineTuneArtifact

    public var id: String { url.path }

    public var label: String {
        "\(artifact.name) · \(artifact.fineTuneType) r\(artifact.rank) · \(dateLabel(artifact.createdAt))"
    }

    private func dateLabel(_ iso: String) -> String {
        let trimmed = iso.replacingOccurrences(of: "T", with: " ")
            .replacingOccurrences(of: "Z", with: "")
        return String(trimmed.prefix(min(16, trimmed.count)))
    }
}

public enum FineTuneStore {
    public static var directory: URL {
        VectorCatalog.runsDirectory.appending(component: "fine-tunes")
    }

    public static func scan(directory root: URL = directory) -> [FineTuneArtifactRecord] {
        let fm = FileManager.default
        guard
            let enumerator = fm.enumerator(
                at: root,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles])
        else { return [] }

        var records: [FineTuneArtifactRecord] = []
        for case let url as URL in enumerator where url.lastPathComponent == "fine-tune.json" {
            guard
                let data = try? Data(contentsOf: url),
                let artifact = try? JSONDecoder().decode(FineTuneArtifact.self, from: data)
            else { continue }
            records.append(FineTuneArtifactRecord(url: url, artifact: artifact))
        }
        return records.sorted {
            if $0.artifact.createdAt == $1.artifact.createdAt {
                return $0.url.path < $1.url.path
            }
            return $0.artifact.createdAt > $1.artifact.createdAt
        }
    }

    @discardableResult
    public static func save(_ artifact: FineTuneArtifact) throws -> FineTuneArtifactRecord {
        let directory = try VectorCatalog.makeUniqueRunDirectory(
            slug: "fine-tune-\(slugify(artifact.name))",
            under: self.directory)
        try RunMetadata.write(
            runType: "lora-train", to: directory,
            modelID: artifact.baseModelID, revision: artifact.baseRevision)
        let url = directory.appending(component: "fine-tune.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(artifact).write(to: url, options: .atomic)
        return FineTuneArtifactRecord(url: url, artifact: artifact)
    }

    @discardableResult
    public static func update(_ artifact: FineTuneArtifact, at url: URL) throws -> FineTuneArtifactRecord {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(artifact).write(to: url, options: .atomic)
        return FineTuneArtifactRecord(url: url, artifact: artifact)
    }

    public static func hashFile(_ url: URL) -> String? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        let digest = SHA256.hash(data: data)
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    public static func hashFileOrDirectory(_ url: URL) -> String? {
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            return nil
        }
        if !isDirectory.boolValue {
            return hashFile(url)
        }

        guard
            let enumerator = FileManager.default.enumerator(
                at: url,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles])
        else { return nil }

        var entries: [(String, Data)] = []
        // Symlink-resolve both sides of the prefix strip: directory
        // enumerators hand back real paths (/private/var/…) even when the
        // root was given in symlinked form (/var/…) — otherwise nested
        // files silently hash under name-only relative keys.
        let base = url.resolvingSymlinksInPath().path
        for case let fileURL as URL in enumerator {
            guard
                (try? fileURL.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true,
                let data = try? Data(contentsOf: fileURL)
            else { continue }
            let filePath = fileURL.resolvingSymlinksInPath().path
            let relative = filePath.hasPrefix(base + "/")
                ? String(filePath.dropFirst(base.count + 1))
                : fileURL.lastPathComponent
            entries.append((relative, data))
        }
        guard !entries.isEmpty else { return nil }

        var hasher = SHA256()
        for (relative, data) in entries.sorted(by: { $0.0 < $1.0 }) {
            hasher.update(data: Data(relative.utf8))
            hasher.update(data: [0])
            hasher.update(data: data)
            hasher.update(data: [0])
        }
        let digest = hasher.finalize()
        return digest.map { String(format: "%02x", $0) }.joined()
    }

    public static func relativePath(for url: URL) -> String {
        let root = VectorCatalog.projectRoot.path
        let path = url.path
        if path.hasPrefix(root + "/") {
            return String(path.dropFirst(root.count + 1))
        }
        return path
    }

    public static func absoluteURL(_ path: String) -> URL {
        path.hasPrefix("/") ? URL(filePath: path) : VectorCatalog.projectRoot.appending(path: path)
    }

    /// The workspace's top-level `adapters/` folder — the default home for
    /// per-adapter training data and outputs (a sibling of prompts/ and
    /// runs/). New workspaces are seeded with it; for existing workspaces it
    /// is created lazily the first time the adapter UI needs it.
    public static var adaptersDirectory: URL {
        VectorCatalog.projectRoot.appending(component: "adapters")
    }

    /// One adapter's workspace home: `adapters/<slug>/` with `training/`
    /// and `validation/` data folders. The root doubles as the adapter
    /// output directory (adapter_config.json + adapters.safetensors land
    /// beside the data folders).
    public struct AdapterHome: Sendable, Equatable {
        public var root: URL
        public var training: URL
        public var validation: URL
    }

    /// Create (or adopt) `adapters/<slug>/` with its `training/` and
    /// `validation/` subfolders. Idempotent: existing directories are
    /// adopted untouched, and nothing inside is ever overwritten. Pass
    /// `parent` to home the adapter outside the workspace default.
    @discardableResult
    public static func createAdapterHome(
        slug: String, under parent: URL? = nil
    ) throws -> AdapterHome {
        let base = parent ?? adaptersDirectory
        let root = base.appending(component: slug, directoryHint: .isDirectory)
        let home = AdapterHome(
            root: root,
            training: root.appending(component: "training", directoryHint: .isDirectory),
            validation: root.appending(component: "validation", directoryHint: .isDirectory))
        let fm = FileManager.default
        try fm.createDirectory(at: home.training, withIntermediateDirectories: true)
        try fm.createDirectory(at: home.validation, withIntermediateDirectories: true)
        return home
    }

    /// What a drag-and-drop copy into a data folder did, with plain-language
    /// refusals for anything not copied.
    public struct DropCopyResult: Sendable, Equatable {
        /// File names newly written into the folder.
        public var copied: [String] = []
        /// File names already present with identical bytes (quietly fine).
        public var identical: [String] = []
        /// Plain-language messages for files that were not copied.
        public var refusals: [String] = []
    }

    /// Copy dropped files INTO a data folder (never referenced in place),
    /// with the house import rule (`TabularImport.writeRefusingDifferingOverwrite`
    /// semantics): identical bytes already present are idempotent; a
    /// same-named file with DIFFERENT bytes refuses with the remedy —
    /// nothing is ever overwritten; a symlinked destination refuses
    /// outright. The folder is created if missing.
    public static func copyDroppedFiles(
        _ sources: [URL], into folder: URL
    ) -> DropCopyResult {
        var result = DropCopyResult()
        let fm = FileManager.default
        do {
            try fm.createDirectory(at: folder, withIntermediateDirectories: true)
        } catch {
            result.refusals.append(
                "could not create \(folder.path): \(error.localizedDescription)")
            return result
        }
        for source in sources {
            let name = source.lastPathComponent
            var isDirectory = ObjCBool(false)
            guard fm.fileExists(atPath: source.path, isDirectory: &isDirectory) else {
                result.refusals.append(
                    "'\(name)' no longer exists at \(source.path) — it was not copied")
                continue
            }
            if isDirectory.boolValue {
                result.refusals.append(
                    "'\(name)' is a folder — drop its files individually "
                        + "(folders are not copied)")
                continue
            }
            guard let data = try? Data(contentsOf: source) else {
                result.refusals.append(
                    "'\(name)' could not be read from \(source.path)")
                continue
            }
            let destination = folder.appending(component: name)
            if (try? fm.destinationOfSymbolicLink(atPath: destination.path)) != nil {
                result.refusals.append(
                    "'\(name)' is a symlink at \(destination.path) — drops write "
                        + "real files only; remove the link and drop again")
                continue
            }
            if fm.fileExists(atPath: destination.path) {
                if (try? Data(contentsOf: destination)) == data {
                    result.identical.append(name)
                } else {
                    result.refusals.append(
                        "'\(name)' already exists in \(folder.lastPathComponent)/ "
                            + "with different contents — drops never overwrite; "
                            + "rename the file and drop it again")
                }
                continue
            }
            do {
                try data.write(to: destination, options: .atomic)
                result.copied.append(name)
            } catch {
                result.refusals.append(
                    "'\(name)' could not be written to \(destination.path): "
                        + "\(error.localizedDescription)")
            }
        }
        return result
    }

    public static func slugify(_ value: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "-_"))
        let scalars = value.lowercased().unicodeScalars.map { scalar -> Character in
            allowed.contains(scalar) ? Character(scalar) : "-"
        }
        let slug = String(scalars)
            .replacingOccurrences(of: "--+", with: "-", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "-"))
        return slug.isEmpty ? "adapter" : slug
    }
}
