import Foundation

/// Which FIELDS differ between two manifests — the readable half of every
/// hash comparison in the lifecycle.
///
/// A hash comparison answers "same or not" and nothing else, which is the
/// right primitive and the wrong message. "Source run X was produced under
/// experiment hash 6eb30c…, but this study currently hashes 9a12f4…" is
/// unactionable: it does not say what changed, so the researcher cannot tell
/// a meaningful edit from a stray one, and the only available move is
/// `--force`. Gates that can only be forced stop being gates.
///
/// Deliberately generic over the encoded JSON rather than enumerating
/// manifest fields: a per-field implementation goes stale the moment the
/// manifest grows a key, and going stale silently is how the epoch guard's
/// diagnosis drifted from its check in the first place.
public enum ManifestDiff {

    /// Dotted paths whose values differ, sorted. Arrays are indexed
    /// (`pipeline.stages[0]`); a key present on one side only is reported
    /// with the absent side rendered as `absent`.
    public static func differingPaths(
        _ left: ExperimentManifest, _ right: ExperimentManifest
    ) -> [String] {
        differences(left, right).map(\.path)
    }

    public struct Difference: Sendable, Equatable {
        public var path: String
        public var left: String
        public var right: String

        /// `path: left → right`, the form the refusals render.
        public var described: String { "\(path): \(left) → \(right)" }
    }

    public static func differences(
        _ left: ExperimentManifest, _ right: ExperimentManifest
    ) -> [Difference] {
        let a = flattened(left)
        let b = flattened(right)
        return Set(a.keys).union(b.keys).sorted().compactMap { path in
            let l = a[path], r = b[path]
            guard l != r else { return nil }
            return Difference(
                path: path, left: l ?? "absent", right: r ?? "absent")
        }
    }

    /// A one-line summary for a refusal: the first `limit` paths, with a
    /// count of the remainder. Empty when nothing differs.
    public static func summary(
        _ left: ExperimentManifest, _ right: ExperimentManifest, limit: Int = 4
    ) -> String {
        let all = differences(left, right)
        guard !all.isEmpty else { return "" }
        let shown = all.prefix(limit).map(\.described).joined(separator: "; ")
        let rest = all.count - min(limit, all.count)
        return rest > 0 ? "\(shown); and \(rest) more" : shown
    }

    // MARK: Flattening

    /// Canonicalized to match `ExperimentStore.manifestHash` EXACTLY: the
    /// volatile freeze stamps it drops must be dropped here too, or the diff
    /// names fields the gate does not check — a message that contradicts its
    /// own verdict is worse than no message.
    static func flattened(_ manifest: ExperimentManifest) -> [String: String] {
        var canonical = manifest
        canonical.status = .draft
        canonical.createdAt = ""
        canonical.frozenAt = nil
        canonical.freezeHash = nil
        canonical.frozenBy = nil
        canonical.gitCommit = nil
        canonical.appVersion = nil
        canonical.freezeForced = nil
        canonical.forcedGatesSkipped = nil
        canonical.preregistrationHash = nil
        canonical.preregistrationGeneratedHash = nil
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(canonical),
            let object = try? JSONSerialization.jsonObject(
                with: data, options: [.fragmentsAllowed])
        else { return [:] }
        var out: [String: String] = [:]
        flatten(object, path: "", into: &out)
        return out
    }

    private static func flatten(
        _ value: Any, path: String, into out: inout [String: String]
    ) {
        switch value {
        case let dictionary as [String: Any]:
            // An EMPTY container is itself a value: without this, `{}` and an
            // absent key flatten identically and a real edit reads as no
            // change.
            if dictionary.isEmpty {
                out[path.isEmpty ? "." : path] = "{}"
                return
            }
            for (key, child) in dictionary {
                flatten(child, path: "\(path).\(key)", into: &out)
            }
        case let array as [Any]:
            if array.isEmpty {
                out[path.isEmpty ? "." : path] = "[]"
                return
            }
            for (index, child) in array.enumerated() {
                flatten(child, path: "\(path)[\(index)]", into: &out)
            }
        case is NSNull:
            out[path] = "null"
        case let number as NSNumber:
            // NSNumber bridges Bool and numerics alike; render booleans as
            // booleans so `true` never prints as `1`.
            if CFGetTypeID(number) == CFBooleanGetTypeID() {
                out[path] = number.boolValue ? "true" : "false"
            } else {
                out[path] = number.stringValue
            }
        default:
            out[path] = "\(value)"
        }
    }
}
