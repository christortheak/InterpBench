import Foundation

/// The scientific body stays raw JSON, preserving UInt64 seeds and opaque
/// records rather than routing them through the UI's Double representation.
enum ScientificActionRequest {
    struct Resolved {
        let method: String
        let path: String
        let query: [URLQueryItem]
        let body: Data
    }
    static func resolve(operation: String, actionID: String, document: Data) throws -> Resolved {
        let operation = try ScienceCatalog.operation(operation)
        guard let action = operation.actions.first(where: { $0.id == actionID }) else {
            throw ExperimentError.malformed("This scientific operation/action has no supported HTTP path.", repair: operation.engineCLI ?? operation.access.restriction)
        }
        guard let object = try JSONSerialization.jsonObject(with: document) as? [String: Any],
              Set(object.keys) == ["path", "query", "body"], let parameters = object["path"] as? [String: String],
              let query = object["query"] as? [String: String], let body = object["body"] as? [String: Any] else {
            throw ExperimentError(reason: "Supply exactly path, query and body objects; path/query values must be strings.")
        }
        let regex = try NSRegularExpression(pattern: #"\{([^}]+)\}"#)
        let range = NSRange(action.path.startIndex..., in: action.path)
        let names = regex.matches(in: action.path, range: range).compactMap { Range($0.range(at: 1), in: action.path).map { String(action.path[$0]) } }
        guard Set(parameters.keys) == Set(names), parameters.values.allSatisfy({ !$0.isEmpty && !$0.contains("/") && $0 != "." && $0 != ".." }), action.method != "GET" || body.isEmpty else {
            throw ExperimentError(reason: "Use exactly this action's path parameters, with single components; GET requires an empty body.")
        }
        var path = action.path
        // ClusterClient's URL builder encodes components while retaining the
        // reverse-proxy prefix; slashes are excluded above.
        for (name, value) in parameters { path = path.replacingOccurrences(of: "{" + name + "}", with: value) }
        return Resolved(method: action.method, path: path,
            query: query.sorted { $0.key < $1.key }.map { URLQueryItem(name: $0.key, value: $0.value) },
            body: try JSONSerialization.data(withJSONObject: body))
    }
}
