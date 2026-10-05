import Foundation

/// A study or template whose saved file is there but cannot be read.
///
/// Listed rather than hidden: a study a coding assistant half-wrote, or one
/// edited by hand with a missing comma, used to vanish from the list without
/// a word. The researcher sees its name, the file, and what is wrong with it,
/// and can reveal the file in Finder. Nothing here ever writes to the file.
public struct UnreadableManifest: Identifiable, Sendable, Equatable {
    /// The folder name — the study's or template's canonical name.
    public let name: String
    /// The file that could not be read.
    public let url: URL
    /// What is wrong, in plain words.
    public let reason: String

    public init(name: String, url: URL, reason: String) {
        self.name = name
        self.url = url
        self.reason = reason
    }

    public var id: String { url.path }

    /// The file's path inside the workspace, for display.
    public func relativePath(workspaceRoot: URL) -> String {
        let root = workspaceRoot.standardizedFileURL.path
        let path = url.standardizedFileURL.path
        return path.hasPrefix(root + "/") ? String(path.dropFirst(root.count + 1)) : path
    }

    /// Scan a library folder (`experiments/` or `templates/`) whose entries
    /// are `<name>/<fileName>`. Returns the names that loaded and the files
    /// that are present but unreadable. A hidden folder (a draft moved to
    /// `.trash-…`), a stray file, or a folder with no `fileName` in it is
    /// not a study and is neither.
    public static func scan<T>(
        directory: URL, fileName: String, load: (String) throws -> T
    ) -> (loaded: [T], unreadable: [UnreadableManifest]) {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        var loaded: [T] = []
        var unreadable: [UnreadableManifest] = []
        for name in names.sorted() where !name.hasPrefix(".") {
            let file = directory.appending(components: name, fileName)
            var isDirectory: ObjCBool = false
            guard FileManager.default.fileExists(atPath: file.path, isDirectory: &isDirectory),
                !isDirectory.boolValue
            else { continue }
            do {
                loaded.append(try load(name))
            } catch {
                unreadable.append(
                    UnreadableManifest(name: name, url: file, reason: reason(for: error, folder: name)))
            }
        }
        return (loaded, unreadable)
    }

    /// A plain-language reason for a read failure.
    public static func reason(for error: any Error, folder: String) -> String {
        func field(_ context: DecodingError.Context) -> String {
            var path = ""
            for key in context.codingPath {
                if let index = key.intValue {
                    path += "[\(index)]"
                } else {
                    path += path.isEmpty ? key.stringValue : ".\(key.stringValue)"
                }
            }
            return path.isEmpty ? "the top level" : "“\(path)”"
        }
        switch error {
        case let DecodingError.keyNotFound(key, context):
            let place = context.codingPath.isEmpty ? "" : " inside \(field(context))"
            return "A required setting, “\(key.stringValue)”, is missing\(place)."
        case let DecodingError.typeMismatch(type, context):
            return "The setting \(field(context)) holds the wrong kind of value (expected \(describe(type)))."
        case let DecodingError.valueNotFound(type, context):
            return "The setting \(field(context)) is empty where \(describe(type)) is required."
        case let DecodingError.dataCorrupted(context):
            if context.codingPath.isEmpty {
                return "The file is not valid JSON — it may be incomplete, or have a typing "
                    + "mistake such as a missing comma or quotation mark."
            }
            return "The setting \(field(context)) has a value SteerLab does not recognize "
                + "(\(context.debugDescription))."
        case let error as CocoaError where error.code == .fileReadNoPermission:
            return "SteerLab does not have permission to read this file."
        case let error as CocoaError where error.code == .fileReadNoSuchFile:
            return "The file disappeared while SteerLab was reading it."
        case let error as ExperimentError where error.reason.contains("does not match its authoring destination"):
            return "The file names a different study than its folder, “\(folder)”. "
                + "The \"name\" inside the file must match the folder name."
        case let error as StudyDesignAuthoringError where error.code == "designIdentityMismatch":
            return "The file names a different template than its folder, “\(folder)”. "
                + "The \"name\" inside the file must match the folder name."
        case let error as StudyDesignAuthoringError where error.code == "unsafeDesignPath":
            return "The template is reached through a link or is not an ordinary file; "
                + "SteerLab reads only ordinary folders and files inside the workspace."
        default:
            return error.localizedDescription
        }
    }

    private static func describe(_ type: Any.Type) -> String {
        switch type {
        case is String.Type: return "text"
        case is Int.Type, is Double.Type, is Float.Type: return "a number"
        case is Bool.Type: return "true or false"
        default:
            let name = String(describing: type)
            if name.hasPrefix("Array") || name.hasPrefix("[") { return "a list" }
            if name.hasPrefix("Dictionary") { return "a group of settings" }
            return "a \(name)"
        }
    }
}
