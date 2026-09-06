import Foundation
import CoreFoundation

/// Portable authoring lineage only. Legacy design stamps and all freeze hashes
/// continue to use their existing algorithms. No review precondition is study data.
public enum PortableDesignIdentity {
    public static let algorithm = "portable-v1"

    public static func hash(_ template: StudyTemplate) throws -> String {
        // Refuse invalid encodings instead of hashing an empty fallback.
        var body = try JSONSerialization.jsonObject(with: JSONEncoder().encode(template.study)) as! [String: Any]
        body["createdAt"] = ""
        body["status"] = "draft"
        for key in ["frozenAt", "freezeHash", "frozenBy", "gitCommit", "appVersion", "freezeForced",
                    "forcedGatesSkipped", "preregistrationHash", "preregistrationGeneratedHash"] {
            body.removeValue(forKey: key)
        }
        let scenario: Any
        if let ref = template.semanticScenario { scenario = ["path": ref.path, "hash": ref.hash] }
        else { scenario = NSNull() }
        let material: [String: Any] = ["schemaVersion": template.schemaVersion,
            "study": normalize(body), "semanticScenario": scenario]
        return ManifestFileTransaction.digest(Data((algorithm + "\0").utf8) + frame(material))
    }

    private static func normalize(_ value: Any, opaque: Bool = false) -> Any {
        if opaque { return value }
        if let object = value as? [String: Any] {
            return object.reduce(into: [String: Any]()) { result, entry in
                let (key, value) = entry
                if value is NSNull && key != "validationHash" { return }
                result[key] = normalize(value, opaque: ["pipeline", "jlensReadout", "saeCandidates", "maxSAEMixtureFeatures", "saeLatentConditions"].contains(key))
            }
        }
        if let values = value as? [Any] { return values.map { normalize($0) } }
        return value
    }

    private static func frame(_ value: Any) -> Data {
        func bytes(_ text: String) -> Data { Data(text.utf8) }
        if value is NSNull { return bytes("n") }
        if let number = value as? NSNumber {
            if CFGetTypeID(number) == CFBooleanGetTypeID() { return bytes(number.boolValue ? "t" : "f") }
            // Preserve UInt64 seeds; converting integers through Double loses bits.
            let type = String(cString: number.objCType)
            if !["d", "f"].contains(type) { return bytes("i" + number.stringValue + ";") }
            let double = number.doubleValue
            if double.rounded() == double {
                return bytes("i" + String(format: "%.0f", locale: Locale(identifier: "en_US_POSIX"), double == 0 ? 0 : double) + ";")
            }
            return bytes("d" + String(format: "%016llx", double.bitPattern) + ";")
        }
        if let text = value as? String { return bytes("s\(text.utf8.count):" + text) }
        if let values = value as? [Any] {
            return values.reduce(bytes("a\(values.count):")) { $0 + frame($1) }
        }
        let object = value as! [String: Any]
        return object.keys.sorted { $0.utf8.lexicographicallyPrecedes($1.utf8) }
            .reduce(bytes("o\(object.count):")) { $0 + frame($1) + frame(object[$1]!) }
    }
}
