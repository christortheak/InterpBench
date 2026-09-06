import Foundation
import SwiftParser
import SwiftSyntax
import SwiftSyntaxBuilder

// Usage: audit-study-management <after-checkout> <before-checkout>
// The baseline must be the reviewed management bridge (95d14aa). This audit
// verifies its forwarding targets, the context accessor relocation, and every
// Sources/Tests syntax tree after normalizing only those owner accesses and
// the explicit arguments formerly supplied by the bridge. It does not resolve
// Swift types; the compiler, both suites and independent diff review remain gates.
guard CommandLine.arguments.count == 3 else {
    fatalError("usage: audit-study-management <after-checkout> <before-checkout>")
}
let root = URL(fileURLWithPath: CommandLine.arguments[1]).resolvingSymlinksInPath()
let baseline = URL(fileURLWithPath: CommandLine.arguments[2]).resolvingSymlinksInPath()
let bridge = "Sources/ExperimentKit/StudyManagementBindings.swift"
let panel = "Sources/ExperimentKit/ExperimentPanel.swift"
guard !FileManager.default.fileExists(atPath: root.appendingPathComponent(bridge).path) else {
    fatalError("StudyManagementBindings must be removed")
}
let oldBridge = Parser.parse(source: try String(contentsOf: baseline.appendingPathComponent(bridge), encoding: .utf8))
let declaration = oldBridge.statements.compactMap { $0.item.as(ExtensionDeclSyntax.self) }.first!
var mapping: [String: String] = [:]
var contextDeclaration: VariableDeclSyntax?
var properties = 0
var commands = 0
func matches(_ pattern: String, _ source: String) -> [String] {
    let r = try! NSRegularExpression(pattern: pattern)
    return r.matches(in: source, range: NSRange(source.startIndex..., in: source)).compactMap {
        Range($0.range(at: 1), in: source).map { String(source[$0]) }
    }
}
func structure(_ node: Syntax) -> String {
    if let token = node.as(TokenSyntax.self) { return "token:\(token.tokenKind)" }
    return "\(node.kind)[" + node.children(viewMode: .sourceAccurate).map(structure).joined(separator: ",") + "]"
}
for member in declaration.memberBlock.members {
    if let variable = member.decl.as(VariableDeclSyntax.self), let binding = variable.bindings.first,
       let name = binding.pattern.as(IdentifierPatternSyntax.self)?.identifier.text {
        if name == "studyCreationContext" { contextDeclaration = variable; continue }
        let direct = "public var \(name): \(binding.typeAnnotation!.type.trimmedDescription) { management.\(name) }"
        let mutable = "public var \(name): \(binding.typeAnnotation!.type.trimmedDescription) { get { management.\(name) } set { management.\(name) = newValue } }"
        let candidates = [direct, mutable].map { structure(Syntax(DeclSyntax(stringLiteral: $0))) }
        guard candidates.contains(structure(Syntax(variable))) else { fatalError("nontrivial baseline property: \(name)") }
        mapping[name] = "management"; properties += 1
    } else if let function = member.decl.as(FunctionDeclSyntax.self), let body = function.body,
              body.statements.count == 1, let call = body.statements.first?.item.as(FunctionCallExprSyntax.self) {
        let name = function.name.text
        let target = call.calledExpression.trimmedDescription
        let owner = target.hasPrefix("management.designs.") ? "management.designs" : "management"
        guard target == owner + "." + name else { fatalError("unexpected forwarding target: \(name)") }
        let expected: String
        if ["create", "newStudy", "newDesignDraft"].contains(name) {
            expected = "\(target)(context: studyCreationContext)"
        } else if name == "templateLineage" {
            expected = "\(target)(manifest, experiments: management.experiments)"
        } else {
            let arguments = function.signature.parameterClause.parameters.map { parameter in
                let local = parameter.secondName?.text ?? parameter.firstName.text
                return parameter.firstName.text == "_" ? local : parameter.firstName.text + ": " + local
            }.joined(separator: ", ")
            expected = "\(target)(\(arguments))"
        }
        guard structure(Syntax(call)) == structure(Syntax(ExprSyntax(stringLiteral: expected))) else {
            fatalError("nontrivial baseline command: \(name)")
        }
        mapping[name] = owner; commands += 1
    } else { fatalError("unaccounted baseline bridge declaration") }
}
guard properties == 6, commands == 16, mapping.count == 22, contextDeclaration != nil else {
    fatalError("unexpected management bridge census")
}
var publicContext = contextDeclaration!
publicContext.modifiers = DeclModifierListSyntax([DeclModifierSyntax(name: .keyword(.public))])

final class Factories: SyntaxVisitor {
    var names: [String] = []
    override func visit(_ node: FunctionDeclSyntax) -> SyntaxVisitorContinueKind {
        if node.signature.returnClause?.type.trimmedDescription == "ExperimentPanel" { names.append(node.name.text) }
        return .visitChildren
    }
}
final class Normalize: SyntaxRewriter {
    let receivers: Set<String>
    var contextsRemoved = 0
    init(source: String) {
        let factories = Factories(viewMode: .sourceAccurate)
        factories.walk(Parser.parse(source: source))
        let constructed = (["ExperimentPanel"] + factories.names).flatMap {
            matches(#"\b(?:var|let)\s+(\w+)\s*=\s*"# + $0 + #"\s*\("#, source)
        }
        receivers = Set(constructed + matches(#"\b(?:var|let)\s+(\w+)\s*:\s*ExperimentPanel\b"#, source)
            + matches(#"\b(?:var|let)\s+(\w+)\s*=\s*[\w.]+\.experiments\b"#, source))
        super.init(viewMode: .sourceAccurate)
    }
    func inPanel(_ node: some SyntaxProtocol) -> Bool {
        var parent = node.parent
        while let p = parent {
            if let c = p.as(ClassDeclSyntax.self) { return c.name.text == "ExperimentPanel" }
            if p.is(StructDeclSyntax.self) || p.is(EnumDeclSyntax.self) { return false }
            if let e = p.as(ExtensionDeclSyntax.self) { return e.extendedType.trimmedDescription == "ExperimentPanel" }
            parent = p.parent
        }
        return false
    }
    func isReceiver(_ base: ExprSyntax, at node: some SyntaxProtocol) -> Bool {
        if let ref = base.as(DeclReferenceExprSyntax.self) {
            return receivers.contains(ref.baseName.text) || (ref.baseName.text == "self" && inPanel(node))
        }
        if let optional = base.as(OptionalChainingExprSyntax.self) { return isReceiver(optional.expression, at: node) }
        if let access = base.as(MemberAccessExprSyntax.self) { return access.declName.baseName.text == "experiments" }
        return false
    }
    // nil = unrelated expression; empty receiver = an unqualified panel member.
    func originalReceiver(_ access: MemberAccessExprSyntax) -> String? {
        guard let owner = mapping[access.declName.baseName.text], var candidate = access.base else { return nil }
        if candidate.trimmedDescription == owner, inPanel(access) { return "" }
        for component in owner.split(separator: ".").reversed() {
            guard let member = candidate.as(MemberAccessExprSyntax.self),
                  member.declName.baseName.text == component, let base = member.base else { return nil }
            candidate = base
        }
        return isReceiver(candidate, at: access) ? candidate.trimmedDescription : nil
    }
    override func visit(_ node: MemberAccessExprSyntax) -> ExprSyntax {
        if let receiver = originalReceiver(node) {
            let name = node.declName.trimmedDescription
            return ExprSyntax(stringLiteral: receiver.isEmpty ? name : receiver + "." + name)
        }
        return super.visit(node)
    }
    override func visit(_ node: FunctionCallExprSyntax) -> ExprSyntax {
        var result = node
        if let access = node.calledExpression.as(MemberAccessExprSyntax.self), let receiver = originalReceiver(access) {
            let name = access.declName.baseName.text
            let base = receiver.isEmpty ? "self" : receiver
            let label: String, expression: String
            if ["create", "newStudy", "newDesignDraft"].contains(name) {
                label = "context"; expression = base + ".studyCreationContext"
            } else if name == "templateLineage" {
                label = "experiments"; expression = base + ".management.experiments"
            } else { return super.visit(node) }
            guard let last = result.arguments.last, last.label?.text == label,
                  structure(Syntax(last.expression)) == structure(Syntax(ExprSyntax(stringLiteral: expression))) else {
                fatalError("incorrect explicit management context: \(node.trimmedDescription)")
            }
            var args = Array(result.arguments.dropLast())
            if !args.isEmpty { args[args.count - 1].trailingComma = nil }
            result.arguments = LabeledExprListSyntax(args)
        }
        return super.visit(result)
    }
    override func visit(_ node: MemberBlockItemListSyntax) -> MemberBlockItemListSyntax {
        let filtered = node.filter { item in
            guard let variable = item.decl.as(VariableDeclSyntax.self),
                  variable.bindings.first?.pattern.trimmedDescription == "studyCreationContext", inPanel(variable) else { return true }
            guard structure(Syntax(variable)) == structure(Syntax(publicContext)) else { fatalError("creation context accessor changed") }
            contextsRemoved += 1
            return false
        }
        return super.visit(filtered)
    }
}
func sources(_ directory: URL) -> Set<String> {
    var paths: Set<String> = []
    for folder in ["Sources", "Tests"] {
        let entries = FileManager.default.enumerator(at: directory.appendingPathComponent(folder), includingPropertiesForKeys: nil)!
        for case let file as URL in entries where file.pathExtension == "swift" {
            let parts = file.pathComponents
            let relative = parts[parts.firstIndex(of: folder)!...].joined(separator: "/")
            if relative != bridge { paths.insert(relative) }
        }
    }
    return paths
}
let paths = sources(root)
guard paths == sources(baseline) else { fatalError("source census changed beyond bridge retirement") }
var changed = 0
var differences = 0
for relative in paths.sorted() {
    let before = try String(contentsOf: baseline.appendingPathComponent(relative), encoding: .utf8)
    let after = try String(contentsOf: root.appendingPathComponent(relative), encoding: .utf8)
    if before == after { continue }
    changed += 1
    let oldNormalizer = Normalize(source: before), newNormalizer = Normalize(source: after)
    let old = oldNormalizer.rewrite(Parser.parse(source: before))
    let new = newNormalizer.rewrite(Parser.parse(source: after))
    if relative == panel {
        guard oldNormalizer.contextsRemoved == 0, newNormalizer.contextsRemoved == 1 else { fatalError("creation context relocation is missing") }
    }
    if structure(old) != structure(new) { print("AUDIT DIFFERENCE \(relative)"); differences += 1 }
}
print("Audited \(properties) properties, \(commands) commands and \(changed) changed files; syntax-tree differences: \(differences)")
if differences > 0 { exit(1) }
