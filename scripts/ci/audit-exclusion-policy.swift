import Foundation
import SwiftParser
import SwiftSyntax

// Compare the old setter's updateDraft closure with the shared pure policy.
// A parsed syntax-tree check, not compiler-resolved type equivalence.
guard CommandLine.arguments.count == 3 else {
    fatalError("usage: audit-exclusion-policy <current-checkout> <baseline-checkout>")
}
let current = URL(fileURLWithPath: CommandLine.arguments[1])
let baseline = URL(fileURLWithPath: CommandLine.arguments[2])

final class Functions: SyntaxVisitor {
    var matches: [FunctionDeclSyntax] = []
    override func visit(_ node: FunctionDeclSyntax) -> SyntaxVisitorContinueKind {
        if node.name.text == "setExclusionRules" { matches.append(node) }
        return .visitChildren
    }
}
final class Closures: SyntaxVisitor {
    var matches: [ClosureExprSyntax] = []
    override func visit(_ node: ClosureExprSyntax) -> SyntaxVisitorContinueKind {
        matches.append(node)
        return .visitChildren
    }
}
func function(_ url: URL) throws -> FunctionDeclSyntax {
    let parsed = Parser.parse(source: try String(contentsOf: url, encoding: .utf8))
    guard !parsed.hasError else { fatalError("source must parse without errors") }
    let visitor = Functions(viewMode: .sourceAccurate)
    visitor.walk(parsed)
    guard visitor.matches.count == 1 else { fatalError("expected exactly one exclusion setter") }
    return visitor.matches[0]
}
func structure(_ node: Syntax) -> String {
    if let token = node.as(TokenSyntax.self) { return "token:\(token.tokenKind)" }
    return "\(node.kind)[" + node.children(viewMode: .sourceAccurate).map(structure).joined(separator: ",") + "]"
}
let old = try function(baseline.appending(path: "Sources/ExperimentKit/ExclusionRules+UI.swift"))
let closures = Closures(viewMode: .sourceAccurate)
closures.walk(old)
guard closures.matches.count == 1 else { fatalError("expected one updateDraft closure") }
let new = try function(current.appending(path: "Sources/ExperimentKit/ManifestDraftEdits.swift"))
guard let body = new.body,
      structure(Syntax(body.statements)) == structure(Syntax(closures.matches[0].statements))
else { fatalError("exclusion policy body changed") }
print("PASS: exclusion policy syntax tree unchanged; only trivia excluded")
