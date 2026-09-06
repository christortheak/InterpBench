import Foundation

/// A study pack is authored input, never a frozen study or an execution bundle.
/// Preview is read-only. Apply rechecks the exact document and file observations
/// before publishing any files, then creates one draft under an absent-file guard.
public enum StudyPackAuthoring {
    public struct File: Codable, Sendable, Equatable {
        public let path: String
        public let sha256: String
        public let disposition: String
        let resolvedPath: String
        let previousSHA256: String?
    }
    public struct Preview: Codable, Sendable {
        public let name: String
        public let workspaceRoot: String
        public let packSHA256: String
        public let reviewSHA256: String
        public let files: [File]
        public let referencedInputs: [File]
        public let advisories: [String]
    }
    public struct Result: Sendable {
        public let study: DraftAuthoringSnapshot
        public let violations: [String]
        public let filesWritten: [String]
    }
    private struct Pack: Codable {
        var study: ExperimentManifest
        var files: [String: String]?
    }
    private struct Observation: Encodable {
        let name: String
        let workspaceRoot: String
        let packSHA256: String
        let files: [File]
        let referencedInputs: [File]
    }

    public static func preview(_ data: Data, workspaceRoot: URL) throws -> Preview {
        let pack: Pack
        do { pack = try decode(data) }
        catch let error as ExperimentError { throw error }
        catch { throw ExperimentError.malformed("study JSON did not decode: \(error)",
            repair: "Supply a complete manifest or a pack with study and files fields; preview it before applying.") }
        let root = URL(fileURLWithPath: try ManifestFileTransaction.canonicalPath(workspaceRoot))
        let storage = ExperimentRepository(workspaceRoot: root)
        guard !FileManager.default.fileExists(atPath: storage.manifestURL(pack.study.name).path),
            !FileManager.default.fileExists(atPath: storage.directory.appending(component: pack.study.name).path) else {
            throw ExperimentError.refusing(.staleManifest, "Study '\(pack.study.name)' already exists; a pack creates a new draft.",
                repair: "Choose a new study name in the pack, preview it again, then apply that review.")
        }
        let files = try (pack.files ?? [:]).sorted { $0.key < $1.key }.map { path, text in
            try observe(path: path, incoming: Data(text.utf8), root: root, packFile: true)
        }
        // Pinning may read named inputs already in the workspace. They are part
        // of the review even when the pack does not supply replacement bytes.
        let references = [pack.study.taskPromptsFile, pack.study.judgeRubricFile,
                          pack.study.capabilityBatteryFile].compactMap { $0 }
        let supplied = Set(files.map(\.path))
        let referenced = try Set(references).subtracting(supplied).sorted().map {
            try observe(path: $0, incoming: nil, root: root, packFile: false)
        }
        let observation = Observation(name: pack.study.name, workspaceRoot: root.path,
            packSHA256: ManifestFileTransaction.digest(data), files: files, referencedInputs: referenced)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return Preview(name: observation.name, workspaceRoot: root.path,
            packSHA256: observation.packSHA256,
            reviewSHA256: ManifestFileTransaction.digest(try encoder.encode(observation)),
            files: files, referencedInputs: referenced,
            advisories: ["Import always creates a draft and removes freeze metadata.",
                "Verification runs after input pinning. An imported draft may still need inputs, artifacts or scientific declarations before it can run."])
    }

    public static func apply(_ data: Data, workspaceRoot: URL, expectedReviewSHA256: String) throws -> Result {
        let read = try preview(data, workspaceRoot: workspaceRoot)
        guard read.reviewSHA256 == expectedReviewSHA256 else { throw stale() }
        let root = URL(fileURLWithPath: read.workspaceRoot)
        // Existing pin/verification adapters are synchronous and active-workspace
        // scoped. Refuse another root; never retarget process-global state here.
        guard try ManifestFileTransaction.canonicalPath(ExperimentStore.workspaceRoot) == root.path else {
            throw ExperimentError.refusing(.staleManifest, "The pack's authoring workspace is not active.",
                repair: "Select the intended workspace, preview the pack there, then apply that review.")
        }
        let manifestURL = ExperimentRepository(workspaceRoot: root).manifestURL(read.name)
        let lockURLs = Set((read.files + read.referencedInputs).map(\.resolvedPath) + [manifestURL.path])
            .sorted().map { URL(fileURLWithPath: $0) }
        return try locked(lockURLs, root: root) {
            let current = try preview(data, workspaceRoot: root)
            guard current.reviewSHA256 == expectedReviewSHA256 else { throw stale() }
            var pack = try decode(data)
            var written: [(String, URL, Data)] = []
            do {
                for file in current.files where file.disposition == "create" {
                    let url = URL(fileURLWithPath: file.resolvedPath)
                    let bytes = Data(pack.files![file.path]!.utf8)
                    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                    try bytes.write(to: url, options: .withoutOverwriting)
                    written.append((file.path, url, bytes))
                }
                autoPinNamedInputs(into: &pack.study)
                try ExperimentStore.save(pack.study, allowCreate: true, workspaceRoot: root, expectedFile: .absent)
            } catch {
                // Remove only files this operation prepared and still owns.
                // Never remove a differing file left by another writer.
                for (_, url, bytes) in written where (try? Data(contentsOf: url)) == bytes {
                    try? FileManager.default.removeItem(at: url)
                }
                throw error
            }
            return Result(study: try DraftAuthoringSnapshot(workspaceRoot: root, name: read.name),
                violations: ExperimentStore.verify(pack.study), filesWritten: written.map { $0.0 })
        }
    }

    /// Export the manifest and its text inputs under prompts/. Other pinned
    /// artifacts are named as dependencies; a study pack never pretends to be
    /// a portable model/vector execution bundle.
    public static func export(reviewed: DraftAuthoringSnapshot) throws -> (data: Data, externalDependencies: [String]) {
        try DraftAuthoringTransaction.perform(reviewed: reviewed, draftOnly: false) { _ in
            var files: [String: String] = [:]
            var external: [String] = []
            for entry in ExperimentStore.pinnedInputEntries(reviewed.manifest) {
                let root = reviewed.workspaceRoot.standardizedFileURL.path + "/"
                let path = entry.url.standardizedFileURL.path
                guard path.hasPrefix(root) else { external.append(entry.label); continue }
                let relative = String(path.dropFirst(root.count))
                guard relative.hasPrefix("prompts/"), let bytes = try? Data(contentsOf: entry.url),
                    let text = String(data: bytes, encoding: .utf8) else {
                    external.append(relative)
                    continue
                }
                files[relative] = text
            }
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            return (try encoder.encode(Pack(study: reviewed.manifest, files: files)), external.sorted())
        }
    }

    private static func decode(_ data: Data) throws -> Pack {
        let raw = try JSONSerialization.jsonObject(with: data)
        var pack: Pack
        if (raw as? [String: Any])?["study"] != nil { pack = try JSONDecoder().decode(Pack.self, from: data) }
        else { pack = Pack(study: try JSONDecoder().decode(ExperimentManifest.self, from: data), files: nil) }
        pack.study.name = ExperimentStore.sanitizedExperimentName(pack.study.name)
        guard !pack.study.name.isEmpty else { throw ExperimentError(reason: "study JSON has no usable name") }
        pack.study.status = .draft
        pack.study.frozenAt = nil
        pack.study.freezeHash = nil
        pack.study.frozenBy = nil
        pack.study.gitCommit = nil
        pack.study.freezeForced = nil
        pack.study.forcedGatesSkipped = nil
        pack.study.preregistrationHash = nil
        pack.study.preregistrationGeneratedHash = nil
        return pack
    }

    private static func observe(path: String, incoming: Data?, root: URL, packFile: Bool) throws -> File {
        guard !path.isEmpty, !(path as NSString).isAbsolutePath, !path.contains(".."),
            !packFile || path.hasPrefix("prompts/") else {
            throw ExperimentError(reason: "study pack file '\(path)' refused — use relative paths under prompts/ without ..")
        }
        let lexical = root.appending(path: path)
        guard (try? FileManager.default.destinationOfSymbolicLink(atPath: lexical.path)) == nil else {
            throw ExperimentError(reason: "study pack file '\(path)' refused — the destination is a symlink")
        }
        let resolved = try ManifestFileTransaction.canonicalPath(lexical)
        let allowed = try ManifestFileTransaction.canonicalPath(packFile ? root.appending(component: "prompts") : root)
        guard resolved.hasPrefix(allowed + "/"), !resolved.hasPrefix(root.appending(component: "runs").path + "/") else {
            throw ExperimentError(reason: "study pack file '\(path)' refused — its directory resolves outside the allowed input tree")
        }
        let previous: Data?
        do { previous = try Data(contentsOf: URL(fileURLWithPath: resolved)) }
        catch let error as CocoaError where error.code == .fileReadNoSuchFile { previous = nil }
        if let incoming, let previous, incoming != previous {
            throw ExperimentError(reason: "study pack file '\(path)' differs from the existing file — packs never overwrite")
        }
        return File(path: path, sha256: ManifestFileTransaction.digest(incoming ?? previous ?? Data()),
            disposition: incoming == nil ? (previous == nil ? "missing" : "read") : (previous == nil ? "create" : "reuse"),
            resolvedPath: resolved, previousSHA256: previous.map(ManifestFileTransaction.digest))
    }

    private static func locked<T>(_ urls: [URL], root: URL, _ body: () throws -> T) throws -> T {
        guard let first = urls.first else { return try body() }
        return try ManifestFileTransaction.withLock(manifestURL: first, workspaceRoot: root) {
            try locked(Array(urls.dropFirst()), root: root, body)
        }
    }
    private static func stale() -> ExperimentError {
        .refusing(.staleManifest, "The pack, workspace or input files changed after preview; nothing was imported.",
            repair: "Preview the exact pack again, inspect the changes, then apply its new reviewSHA256.")
    }
    private static func autoPinNamedInputs(into manifest: inout ExperimentManifest) {
        if let path = manifest.taskPromptsFile, manifest.taskPromptsHash == nil {
            _ = try? ExperimentStore.pinTaskPrompts(path, into: &manifest)
        }
        if let path = manifest.judgeRubricFile, manifest.judgeRubricHash == nil {
            _ = try? JudgeRubricStore.pin(path, into: &manifest)
        }
        if let path = manifest.capabilityBatteryFile, manifest.capabilityBatteryHash == nil {
            _ = try? ExperimentStore.pinCapabilityBattery(path, into: &manifest)
        }
    }
}
