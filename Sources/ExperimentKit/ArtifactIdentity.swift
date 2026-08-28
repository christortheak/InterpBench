import Foundation

/// Resolving an artifact reference (`vectorArtifactID`, `adapterDirectory`,
/// `artifactPath`) to a local file, and comparing two references for
/// identity, without either engine's path conventions leaking through.
///
/// The two engines write the same field in two shapes:
///
/// - Python `promote` stores it **workspace-relative**
///   (`os.path.relpath(artifact.id, root_dir)`, `promote.py:269`) and
///   resolves it through `paths.resolve`, which joins against the project
///   root.
/// - Swift stores the **absolute** `VectorArtifact.id` and, until now,
///   resolved it with a bare `URL(filePath:)` — which joins a relative path
///   against the *process working directory*, not the workspace.
///
/// So a server-promoted agent imported onto the Mac referenced a vector that
/// the Mac looked for beside the app binary, reported as "Missing artifact"
/// however many times the vector was imported. Same defect, string form, in
/// `AgentLibrary`: catalog ids are absolute, so a relative reference never
/// matched any of them.
///
/// This is B3's "path-independent artifact identity": both engines' shapes
/// resolve to one canonical local locator before anything is opened or
/// compared.
public enum ArtifactIdentity {

    /// The workspace-relative or absolute reference, resolved to an absolute
    /// standardized path.
    ///
    /// Delegates to the established workspace resolver rather than adding a
    /// competing rule — `FineTuneStore.absoluteURL` is already documented as
    /// the counterpart of Python's `paths.resolve`. What is new here is
    /// *using* it on the vector path, which took `URL(filePath:)` directly.
    public static func resolve(_ reference: String) -> URL {
        let direct = ModelVariantStore.absoluteURL(reference).standardizedFileURL
        guard let rebased = rebasedToWorkspace(reference, whenMissing: direct)
        else { return direct }
        return rebased
    }

    /// The workspace subtrees an artifact reference can name. A path is
    /// rebased only from one of these — an arbitrary shared prefix is not
    /// evidence that two paths name the same artifact.
    static let rebasableRoots = ["runs", "experiments", "adapters", "prompts"]

    /// A reference recorded on ANOTHER MACHINE's filesystem, rebased onto
    /// this workspace.
    ///
    /// The relative/absolute split this type was built for assumes both
    /// shapes describe the same filesystem. A cluster workspace breaks that:
    /// an agent authored against a vector the researcher was looking at on
    /// the server records `/scratch/<user>/steerlab-workspace/runs/<run>/<leaf>`,
    /// which names nothing on the Mac even though `runs/<run>/<leaf>` is
    /// sitting right there, imported. Observed live 2026-07-26: a
    /// hand-created agent failed to upload with `MissingVariantDependency`
    /// because its dependency was present under a different root.
    ///
    /// Deliberately a FALLBACK, applied only when the direct resolution names
    /// nothing: it can turn a certain failure into a hit and can never
    /// redirect a reference that already resolves. The rebased tail must
    /// itself exist, so an unrelated path with a `runs/` segment is left
    /// alone rather than silently repointed.
    static func rebasedToWorkspace(
        _ reference: String, whenMissing direct: URL
    ) -> URL? {
        guard reference.hasPrefix("/") else { return nil }
        let fm = FileManager.default
        // Vector references are extension-less locators; the sidecar beside
        // them is what proves the artifact is really there.
        guard !exists(direct, fm) else { return nil }
        let components = URL(filePath: reference).standardizedFileURL
            .pathComponents
        // Last occurrence: a cluster root may itself contain "runs".
        for root in rebasableRoots {
            guard let index = components.lastIndex(of: root),
                index + 1 < components.count
            else { continue }
            let tail = components[index...].joined(separator: "/")
            let candidate = ModelVariantStore.absoluteURL(tail).standardizedFileURL
            if exists(candidate, fm) { return candidate }
        }
        return nil
    }

    private static func exists(_ url: URL, _ fm: FileManager) -> Bool {
        if fm.fileExists(atPath: url.path) { return true }
        // Extension-less artifact locator: `<id>.json` + `<id>.safetensors`.
        return fm.fileExists(atPath: url.appendingPathExtension("json").path)
    }

    /// The inverse of `resolve`, for WRITING references: an absolute path
    /// under the current workspace root becomes workspace-relative
    /// (`runs/<run>/<leaf>`) — the shape the Python engine serializes
    /// (`os.path.relpath(artifact.id, root_dir)`, `promote.py`) and the only
    /// shape that survives moving the artifact to another machine. A
    /// reference outside the workspace passes through untouched: it names
    /// something this rule cannot make portable, and rewriting it would only
    /// hide that.
    ///
    /// An ALREADY-relative reference is normalized LEXICALLY (review round 10,
    /// finding 7) rather than passed through: `prompts/x`, `./prompts/x` and
    /// `prompts/a/../x` name one file and used to compare as three, so a
    /// re-declaration that spelled the path the other way read as a different
    /// file and could drop the hash pin already standing beside it. The
    /// normalization strips `./`, collapses empty segments, and resolves `..`
    /// against the components to its left — no filesystem is touched, because
    /// a relative reference has no root to resolve against here and a lexical
    /// answer must not depend on what happens to exist. A reference that
    /// ESCAPES the root (`../outside`, or `a/../../outside`) is returned
    /// verbatim: `..` past the top cannot be resolved lexically without
    /// inventing a root, and the same reasoning that leaves an outside-the-
    /// workspace absolute path alone applies.
    ///
    /// Symlinks are handled the same way `canonical` handles them (macOS
    /// `/var` → `/private/var`): the verbatim prefix is tried first, then
    /// both sides with their containing directories symlink-resolved.
    public static func workspaceRelative(_ reference: String) -> String {
        guard reference.hasPrefix("/") else {
            return lexicallyNormalizedRelative(reference)
        }
        func tail(of path: String, under root: String) -> String? {
            let prefix = root.hasSuffix("/") ? root : root + "/"
            guard path.hasPrefix(prefix), path.count > prefix.count
            else { return nil }
            return String(path.dropFirst(prefix.count))
        }
        let root = VectorCatalog.projectRoot.standardizedFileURL
        let target = URL(filePath: reference).standardizedFileURL
        if let relative = tail(of: target.path, under: root.path) {
            return relative
        }
        // The leaf is an extension-less locator with no file at exactly that
        // path, so resolve the containing directory and re-append (the
        // `canonical` convention).
        let resolvedTarget = target.deletingLastPathComponent()
            .resolvingSymlinksInPath()
            .appending(component: target.lastPathComponent)
        if let relative = tail(
            of: resolvedTarget.path,
            under: root.resolvingSymlinksInPath().path)
        {
            return relative
        }
        return reference
    }

    /// One relative reference reduced to its canonical spelling, purely by
    /// walking components — never by asking the filesystem.
    ///
    /// Returns the input verbatim when it is empty, when it resolves to
    /// nothing at all (`.`, `a/..`), or when a `..` walks off the top: those
    /// are not spellings of a path under the root, and inventing one would be
    /// the silent redirect this whole file refuses.
    static func lexicallyNormalizedRelative(_ reference: String) -> String {
        guard !reference.isEmpty else { return reference }
        var stack: [String] = []
        for segment in reference.split(separator: "/", omittingEmptySubsequences: true) {
            switch segment {
            case ".":
                continue
            case "..":
                // A `..` with nowhere to go escapes the root: leave the whole
                // reference exactly as written.
                guard let last = stack.last, last != ".." else { return reference }
                stack.removeLast()
            default:
                stack.append(String(segment))
            }
        }
        guard !stack.isEmpty else { return reference }
        // A trailing separator is preserved: it is how a directory reference
        // (`adapterDirectory`) is spelled, and this rule normalizes spelling,
        // not shape.
        return stack.joined(separator: "/")
            + (reference.hasSuffix("/") ? "/" : "")
    }

    /// The comparison key for "are these the same artifact?" — resolved, so
    /// an absolute catalog id and a workspace-relative variant reference
    /// compare equal when they name the same file.
    ///
    /// Symlinks are resolved too. On macOS `/var` is a symlink to
    /// `/private/var`, so a catalog that walked the resolved root and a
    /// reference built from the unresolved one name the same file by
    /// different strings — and an artifact-identity comparison that called
    /// those different would reintroduce the "missing artifact" bug this
    /// type exists to fix. The LEAF is not resolved (it is an extension-less
    /// locator with no file at exactly that path, and
    /// `resolvingSymlinksInPath` bails on nonexistent leaves), so resolve the
    /// containing directory and re-append.
    public static func canonical(_ reference: String) -> String {
        let url = resolve(reference)
        return url.deletingLastPathComponent()
            .resolvingSymlinksInPath()
            .appending(component: url.lastPathComponent)
            .path
    }

    /// Canonicalizes a whole set of references (catalog ids) once, for
    /// membership tests against `canonical(_:)`.
    public static func canonical(_ references: Set<String>) -> Set<String> {
        Set(references.map(canonical))
    }

    /// Membership that tolerates either engine's path shape on either side.
    public static func contains(
        _ references: Set<String>, _ reference: String
    ) -> Bool {
        // Fast path first: same-engine artifacts match verbatim, and that is
        // the overwhelmingly common case.
        references.contains(reference)
            || canonical(references).contains(canonical(reference))
    }
}
