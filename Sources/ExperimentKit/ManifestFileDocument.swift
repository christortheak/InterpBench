import Foundation

/// A manifest as it is written to a file: every modeled field, plus each
/// top-level key this build does not model, with its value exactly as read
/// (`ExactJSONValue`: a 64-bit integer stays that integer).
///
/// The Mac decodes a typed `ExperimentManifest`. Before this, a key the typed
/// structure did not know was dropped, so an older app that loaded and re-saved
/// a draft written by a newer engine or client silently lost the newer fields
/// (a declared primary outcome, a freeze stamp, anything added later). The
/// decoder now keeps those keys in `unknownTopLevelFields`, and every writer
/// of a manifest file encodes through this type so they survive.
///
/// The typed encoding (`JSONEncoder().encode(manifest)`) is unchanged and still
/// excludes them, which is what keeps `ExperimentStore.manifestHash` and every
/// existing Swift-stamped hash byte-identical. Freeze refuses a draft that
/// carries such keys (`ManifestMutationPolicy.admitFreeze`), so no manifest
/// this build freezes holds content its freeze hash does not cover.
public struct ManifestFileDocument: Encodable {
    public let manifest: ExperimentManifest

    public init(_ manifest: ExperimentManifest) {
        self.manifest = manifest
    }

    public func encode(to encoder: Encoder) throws {
        try manifest.encode(to: encoder)
        guard !manifest.unknownTopLevelFields.isEmpty else { return }
        // A second keyed container at the same level writes into the same
        // JSON object the typed encoding just filled.
        var container = encoder.container(keyedBy: Key.self)
        for (name, value) in manifest.unknownTopLevelFields {
            try container.encode(value, forKey: Key(name))
        }
    }

    /// The pretty-printed, key-sorted bytes every manifest file is written as.
    public static func data(_ manifest: ExperimentManifest) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try encoder.encode(ManifestFileDocument(manifest))
    }

    /// The top-level keys of the document being decoded that
    /// `ExperimentManifest.CodingKeys` does not name, with their values.
    static func unknownFields(in decoder: Decoder) throws -> [String: ExactJSONValue] {
        let container = try decoder.container(keyedBy: Key.self)
        var unknown: [String: ExactJSONValue] = [:]
        for key in container.allKeys where ExperimentManifest.CodingKeys(stringValue: key.stringValue) == nil {
            unknown[key.stringValue] = try container.decode(ExactJSONValue.self, forKey: key)
        }
        return unknown
    }

    /// The refusal for freezing a manifest that carries unknown keys, or nil.
    static func freezeRefusal(_ manifest: ExperimentManifest) -> ExperimentError? {
        let names = manifest.unknownTopLevelFields.keys.sorted()
        guard !names.isEmpty else { return nil }
        let listed = names.map { "'\($0)'" }.joined(separator: ", ")
        return .refusing(
            .missingPrerequisite,
            "cannot freeze '\(manifest.name)': it has settings this version of SteerLab "
                + "does not recognise (\(listed)). They were written by a newer version of "
                + "SteerLab or by the Python client, and are kept in the draft, but a freeze "
                + "here would stamp a freeze hash that does not cover them.",
            repair: "Update SteerLab to the current release and freeze again, or freeze with "
                + "the client that wrote them: steerlab experiment freeze \(manifest.name)")
    }

    struct Key: CodingKey {
        let stringValue: String
        var intValue: Int? { nil }
        init(_ string: String) { stringValue = string }
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { nil }
    }
}
