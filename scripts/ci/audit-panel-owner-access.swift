import Foundation
import SwiftParser
import SwiftSyntax
import SwiftSyntaxBuilder

// Usage: audit-panel-owner-access <current-checkout> <baseline-checkout> [design]
// Compile with SwiftParser/SwiftSyntax from the selected Xcode host libraries.
// This is a parsed syntax-tree audit, paired with the ordinary compiler and
// suites. It does not claim compiler-resolved symbol/type equivalence.
guard (3...4).contains(CommandLine.arguments.count) else {
    fatalError("usage: audit-panel-owner-access <current-checkout> <baseline-checkout> [design]")
}
let root = URL(fileURLWithPath: CommandLine.arguments[1]).resolvingSymlinksInPath()
let beforeRoot = URL(fileURLWithPath: CommandLine.arguments[2]).resolvingSymlinksInPath()
let designMode = CommandLine.arguments.count == 4 && CommandLine.arguments[3] == "design"
guard CommandLine.arguments.count == 3 || designMode else { fatalError("unknown migration mode") }
let bridge = "Sources/ExperimentKit/" + (designMode ? "StudyManagementBindings.swift" : "StudyPanelBindings.swift")
let forwarding = try String(contentsOf: beforeRoot.appendingPathComponent(bridge), encoding: .utf8)
let currentBridge = designMode ? try String(contentsOf: root.appendingPathComponent(bridge), encoding: .utf8) : ""
var mapping: [String: String] = [:]
if designMode {
    let accessor = try NSRegularExpression(pattern: #"public var (\w+):[^\n]+\{\s*(?:get \{ )?management\.designs\.\1\b"#)
    for match in accessor.matches(in: forwarding, range: NSRange(forwarding.startIndex..., in: forwarding)) {
        let name = String(forwarding[Range(match.range(at: 1), in: forwarding)!])
        mapping[name] = "management.designs"
    }
    guard mapping.count == 7 else { fatalError("unexpected baseline design property census") }
    for name in mapping.keys {
        guard !currentBridge.contains("public var " + name + ":") else { fatalError("forwarding property remains: " + name) }
    }
} else {
    guard !FileManager.default.fileExists(atPath: root.appendingPathComponent(bridge).path) else {
        fatalError("the property bridge must be retired before this audit passes")
    }
    let accessor = try NSRegularExpression(pattern: #"var (\w+):[^\n]+\{\s*get \{ (\w+)\.\1 \}\s*set \{ \2\.\1 = newValue \}"#)
    for match in accessor.matches(in: forwarding, range: NSRange(forwarding.startIndex..., in: forwarding)) {
        let name = String(forwarding[Range(match.range(at: 1), in: forwarding)!])
        mapping[name] = String(forwarding[Range(match.range(at: 2), in: forwarding)!])
    }
    guard mapping.count == 109 else { fatalError("unexpected baseline forwarding property census") }
}

func matches(_ pattern: String, _ source: String) -> [String] {
    let r = try! NSRegularExpression(pattern: pattern)
    return r.matches(in: source, range: NSRange(source.startIndex..., in: source)).compactMap {
        Range($0.range(at: 1), in: source).map { String(source[$0]) }
    }
}

final class PanelFactories: SyntaxVisitor {
    var names: [String] = []
    override func visit(_ node: FunctionDeclSyntax) -> SyntaxVisitorContinueKind {
        if node.signature.returnClause?.type.trimmedDescription == "ExperimentPanel" { names.append(node.name.text) }
        return .visitChildren
    }
}

final class FieldAccessRewriter: SyntaxRewriter {
    let receivers: Set<String>
    let bindingOwners: [String: (receiver: String, owner: String)]
    init(source: String) {
        let factories = PanelFactories(viewMode: .sourceAccurate)
        factories.walk(Parser.parse(source: source))
        let constructed = factories.names.flatMap { factory in
            matches(#"\b(?:var|let)\s+(\w+)\s*=\s*"# + NSRegularExpression.escapedPattern(for: factory) + #"\s*\("#, source)
        }
        receivers = Set(constructed + matches(#"\b(?:var|let)\s+(\w+)\s*:\s*ExperimentPanel\b"#, source)
            + matches(#"\b(?:var|let)\s+(\w+)\s*=\s*ExperimentPanel\s*\("#, source)
            + matches(#"\b(?:var|let)\s+(\w+)\s*=\s*[\w.]+\.experiments\b"#, source)
            + matches(#"\b(\w+)\s*:\s*ExperimentPanel\b"#, source))
        let pattern = try! NSRegularExpression(pattern: #"@Bindable\s+var\s+(\w+)\s*=\s*(\w+)\.(draft|localJobs|results|remoteJobs|submission|management\.designs)\b"#)
        var aliases: [String: (receiver: String, owner: String)] = [:]
        for match in pattern.matches(in: source, range: NSRange(source.startIndex..., in: source)) {
            func part(_ index: Int) -> String { String(source[Range(match.range(at: index), in: source)!]) }
            aliases[part(1)] = (part(2), part(3))
        }
        bindingOwners = aliases
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
            let name = ref.baseName.text
            return receivers.contains(name.hasPrefix("$") ? String(name.dropFirst()) : name)
                || (name == "self" && inPanel(node))
        }
        if let optional = base.as(OptionalChainingExprSyntax.self) { return isReceiver(optional.expression, at: node) }
        if let access = base.as(MemberAccessExprSyntax.self) { return access.declName.baseName.text == "experiments" }
        return false
    }
    override func visit(_ node: MemberAccessExprSyntax) -> ExprSyntax {
        let name = node.declName.baseName.text
        if let owner = mapping[name], let base = node.base {
            if  let ref = base.as(DeclReferenceExprSyntax.self),
                ref.baseName.text.hasPrefix("$"),
                let alias = bindingOwners[String(ref.baseName.text.dropFirst())], alias.owner == owner {
                var replacement = node
                replacement.base = ExprSyntax(DeclReferenceExprSyntax(baseName: .identifier("$" + alias.receiver)))
                return ExprSyntax(replacement)
            }
            var candidate = base
            var ownerMatched = true
            for component in owner.split(separator: ".").reversed() {
                guard let member = candidate.as(MemberAccessExprSyntax.self),
                    member.declName.baseName.text == component, let parent = member.base else {
                    ownerMatched = false
                    break
                }
                candidate = parent
            }
            if ownerMatched, isReceiver(candidate, at: node) {
                var replacement = node; replacement.base = candidate
                return ExprSyntax(replacement)
            }
            if  base.trimmedDescription == owner, inPanel(node) {
                var ref = node.declName
                ref.leadingTrivia = node.leadingTrivia
                ref.trailingTrivia = node.trailingTrivia
                return ExprSyntax(ref)
            }

        }
        return super.visit(node)
    }
    override func visit(_ node: CodeBlockItemListSyntax) -> CodeBlockItemListSyntax {
        let visited = super.visit(node)
        return visited.filter { item in
            guard let declaration = item.item.as(VariableDeclSyntax.self),
                declaration.attributes.trimmedDescription == "@Bindable", declaration.bindings.count == 1,
                let binding = declaration.bindings.first,
                let name = binding.pattern.as(IdentifierPatternSyntax.self)?.identifier.text,
                let alias = bindingOwners[name],
                binding.initializer?.value.trimmedDescription == alias.receiver + "." + alias.owner
            else { return true }
            return false
        }
    }
    override func visit(_ node: OptionalBindingConditionSyntax) -> OptionalBindingConditionSyntax {
        var result = super.visit(node)
        if  let pattern = result.pattern.as(IdentifierPatternSyntax.self),
            let value = result.initializer?.value.as(DeclReferenceExprSyntax.self),
            pattern.identifier.text == value.baseName.text {
            result.initializer = nil
        }
        return result
    }

}

func sources(_ directory: URL) -> Set<String> {
    var paths: Set<String> = []
    for folder in ["Sources", "Tests"] {
        let entries = FileManager.default.enumerator(at: directory.appendingPathComponent(folder), includingPropertiesForKeys: nil)!
        for case let file as URL in entries where file.pathExtension == "swift" {
            let components = file.pathComponents
            guard let start = components.firstIndex(of: folder) else { fatalError("missing source folder") }
            let relative = components[start...].joined(separator: "/")
            if relative != bridge { paths.insert(relative) }
        }
    }
    return paths
}
let paths = sources(root)
guard paths == sources(beforeRoot) else {
    print("New: \(paths.subtracting(sources(beforeRoot)).sorted().prefix(5))")
    print("Missing: \(sources(beforeRoot).subtracting(paths).sorted().prefix(5))")
    exit(1)
}
func structure(_ node: Syntax) -> String {
    if let token = node.as(TokenSyntax.self) { return "token:\(token.tokenKind)" }
    return "\(node.kind)[" + node.children(viewMode: .sourceAccurate).map(structure).joined(separator: ",") + "]"
}
// Only the seven forwarding variable declarations may disappear from the bridge.
final class RemoveRetiredProperties: SyntaxRewriter {
    override func visit(_ node: MemberBlockItemListSyntax) -> MemberBlockItemListSyntax {
        super.visit(node).filter { item in
            guard let declaration = item.decl.as(VariableDeclSyntax.self), declaration.bindings.count == 1,
                let name = declaration.bindings.first?.pattern.as(IdentifierPatternSyntax.self)?.identifier.text else { return true }
            return mapping[name] == nil
        }
    }
}
if designMode {
    let oldBridge = RemoveRetiredProperties(viewMode: .sourceAccurate).rewrite(Parser.parse(source: forwarding))
    let newBridge = Parser.parse(source: currentBridge)
    guard structure(oldBridge) == structure(Syntax(newBridge)) else {
        fatalError("the bridge changed beyond retirement of the seven design properties")
    }
}
var changed = 0
var differences = 0
for relative in paths.sorted() {
    let source = try String(contentsOf: root.appendingPathComponent(relative), encoding: .utf8)
    let before = try String(contentsOf: beforeRoot.appendingPathComponent(relative), encoding: .utf8)
    if source == before { continue }
    changed += 1
    let old = FieldAccessRewriter(source: before).rewrite(Parser.parse(source: before))
    let new = FieldAccessRewriter(source: source).rewrite(Parser.parse(source: source))
    if structure(old) != structure(new) { print("AUDIT DIFFERENCE \(relative)"); differences += 1 }
}
print("Audited \(changed) changed source files; normalized syntax-tree differences: \(differences)")
if differences > 0 { exit(1) }
