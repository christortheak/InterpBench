import Foundation
import SwiftParser
import SwiftSyntax

// Check the shared response extraction independently of the new prompt adapter.
// Parsed syntax trees, not compiler-resolved type equivalence.
guard CommandLine.arguments.count == 3 else {
    fatalError("usage: audit-authoring-http-results <current-checkout> <baseline-checkout>")
}
final class Declarations: SyntaxVisitor {
    var structs: [String: [StructDeclSyntax]] = [:]
    var failures: [FunctionDeclSyntax] = []
    override func visit(_ node: StructDeclSyntax) -> SyntaxVisitorContinueKind {
        structs[node.name.text, default: []].append(node)
        return .visitChildren
    }
    override func visit(_ node: FunctionDeclSyntax) -> SyntaxVisitorContinueKind {
        if node.name.text == "failure" { failures.append(node) }
        return .visitChildren
    }
}
func declarations(_ root: String, _ path: String) throws -> Declarations {
    let url = URL(fileURLWithPath: root).appending(path: path)
    let parsed = Parser.parse(source: try String(contentsOf: url, encoding: .utf8))
    guard !parsed.hasError else { fatalError("source must parse without errors") }
    let visitor = Declarations(viewMode: .sourceAccurate)
    visitor.walk(parsed)
    return visitor
}
func structure(_ node: Syntax) -> String {
    if let token = node.as(TokenSyntax.self) { return "token:\(token.tokenKind)" }
    return "\(node.kind)[" + node.children(viewMode: .sourceAccurate).map(structure).joined(separator: ",") + "]"
}
let old = try declarations(CommandLine.arguments[2], "Sources/ExperimentKit/StudyProtocolHTTP.swift")
let new = try declarations(CommandLine.arguments[1], "Sources/ExperimentKit/StudyAuthoringHTTP.swift")
for name in ["Response", "Document"] {
    guard let before = old.structs[name], before.count == 1,
          let after = new.structs[name], after.count == 1,
          structure(Syntax(before[0].memberBlock)) == structure(Syntax(after[0].memberBlock))
    else { fatalError("\(name) members changed") }
}
// Response.failure is nested; the second declaration is the error classifier.
let before = old.failures.filter { $0.signature.parameterClause.parameters.first?.firstName.text == "_" }
let after = new.failures.filter { $0.signature.parameterClause.parameters.first?.firstName.text == "_" }
guard before.count == 2, after.count == 2,
      let beforeBody = before.last?.body, let afterBody = after.last?.body,
      structure(Syntax(beforeBody)) == structure(Syntax(afterBody))
else { fatalError("error classification body changed") }
print("PASS: shared HTTP response/document members and error body unchanged; only trivia excluded")
