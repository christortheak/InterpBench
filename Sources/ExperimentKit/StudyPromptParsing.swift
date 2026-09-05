import Foundation
import SteeringKit

/// Value-only rules; no workspace discovery or artifact writes.
enum StudyPromptParsing {
    typealias StudyPrompt = ExperimentTasks.StudyPrompt
    typealias TranscriptTurn = ExperimentTasks.TranscriptTurn
    static let transcriptRoles: Set<String> = ["system", "user", "assistant"]

    static func duplicateTaskPromptIDMessage(
        id: String, firstItem: Int, duplicateItem: Int
    ) -> String {
        "task prompts: duplicate item id '\(id)' (items \(firstItem) and "
            + "\(duplicateItem)) — ids must be unique for pairing and reporting"
    }

    static func duplicateTaskPromptIDs(_ data: Data) -> [(id: String, items: [Int])] {
        struct IDLine: Decodable { let id: String? }
        let decoder = JSONDecoder()
        var positions: [String: [Int]] = [:]
        var order: [String] = []
        var count = 0
        let lines = String(decoding: data, as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: true)
        for raw in lines {
            let trimmed = raw.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty,
                let line = try? decoder.decode(IDLine.self, from: Data(trimmed.utf8))
            else { continue }
            count += 1
            let id = line.id ?? "prompt-\(count)"
            if positions[id] == nil { order.append(id) }
            positions[id, default: []].append(count)
        }
        return order.compactMap { id in
            guard let items = positions[id], items.count > 1 else { return nil }
            return (id: id, items: items)
        }
    }

    static func taskPromptFactorsMessage(itemID: String) -> String {
        "task prompts: item '\(itemID)' has a 'factors' value that is not "
            + "a flat string-to-string object — factor names and level "
            + "names must both be strings"
    }

    static func parseTaskPrompts(_ data: Data) throws -> [StudyPrompt] {
        struct AttentionCheckLine: Decodable {
            let expected: String?
            let grading: String?
        }
        struct Line: Decodable {
            let id: String?
            let prompt: String?
            let text: String?
            let options: [String]?
            let target: String?
            let anchorMonths: Double?
            let severity: Double?
            let arm: String?
            let caseID: String?
            let transcript: [TranscriptTurn]?
            let attentionCheck: AttentionCheckLine?
            let responseFormat: String?
        }
        let decoder = JSONDecoder()
        var prompts: [StudyPrompt] = []
        var seenIDs: [String: Int] = [:]  // id → 1-based item ordinal
        let lines = String(decoding: data, as: UTF8.self)
            .split(separator: "\n", omittingEmptySubsequences: true)
        for (index, raw) in lines.enumerated() {
            let trimmed = raw.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { continue }
            guard
                let line = try? decoder.decode(Line.self, from: Data(trimmed.utf8)),
                line.prompt ?? line.text != nil || line.transcript != nil
            else {
                throw ExperimentError(
                    reason: "malformed task prompt JSONL at line \(index + 1)")
            }
            // Explicit ids must be non-empty; null/absent take the shared
            // prompt-<ordinal> fallback (review 2026-08-03, P2 — message
            // string is the cross-engine contract; server twin:
            // _load_prompts).
            let id: String
            if let declared = line.id {
                guard !declared.trimmingCharacters(in: .whitespaces).isEmpty
                else {
                    throw ExperimentError(
                        reason: "task prompts: item \(prompts.count + 1) "
                            + "declares an empty or non-string 'id' — "
                            + "declare a non-empty string, or omit the key "
                            + "for the prompt-<ordinal> fallback")
                }
                id = declared
            } else {
                id = "prompt-\(prompts.count + 1)"
            }
            // Duplicate ids (explicit or auto-collided) refuse BEFORE the
            // per-item transcript checks — cross-engine ordering contract.
            if let firstItem = seenIDs[id] {
                throw ExperimentError(
                    reason: duplicateTaskPromptIDMessage(
                        id: id, firstItem: firstItem,
                        duplicateItem: prompts.count + 1))
            }
            seenIDs[id] = prompts.count + 1
            if let transcript = line.transcript {
                if let violation = transcriptSchemaViolation(transcript, itemID: id) {
                    throw ExperimentError(reason: violation)
                }
            }
            // Per-item attention check (the exclusion instrument's first
            // user), validated at LOAD with plain-language, cross-engine-
            // identical messages; items without a check are untouched.
            var attentionCheck: AttentionCheck?
            if let check = line.attentionCheck {
                if let violation = ExclusionEngine.attentionCheckViolation(
                    expected: check.expected, grading: check.grading, itemID: id)
                {
                    throw ExperimentError(reason: violation)
                }
                attentionCheck = AttentionCheck(
                    expected: check.expected ?? "",
                    grading: check.grading.flatMap(
                        CapabilityBattery.GradingMode.init(rawValue:)))
            }
            // Factorial cell metadata (the generator's `factors` object):
            // validated as a flat string→string map at LOAD via the raw
            // JSON (Codable can't distinguish wrong-shape from absent) —
            // identical message on the server. Empty ⇒ treated as absent.
            var factors: [String: String]?
            if let object = try? JSONSerialization.jsonObject(
                with: Data(trimmed.utf8)) as? [String: Any],
                let rawFactors = object["factors"]
            {
                guard let typed = rawFactors as? [String: String] else {
                    throw ExperimentError(
                        reason: taskPromptFactorsMessage(itemID: id))
                }
                if !typed.isEmpty { factors = typed }
            }
            // Closed vocabulary, validated at LOAD: an unrecognised value
            // must refuse rather than degrade to "unspecified", which would
            // re-open the hole `ResponseFormat` closes (a typo silently
            // restoring permissive behaviour).
            let responseFormat: ResponseFormat?
            do {
                responseFormat = try ResponseFormat.parse(line.responseFormat)
            } catch {
                throw ExperimentError(
                    reason: "task prompt '\(id)': \(error)")
            }
            let text =
                line.prompt ?? line.text
                ?? line.transcript.map(transcriptDisplayText) ?? ""
            prompts.append(
                StudyPrompt(
                    id: id,
                    text: text,
                    options: line.options,
                    target: line.target,
                    anchorMonths: line.anchorMonths,
                    severity: line.severity,
                    arm: line.arm,
                    caseID: line.caseID,
                    transcript: line.transcript,
                    attentionCheck: attentionCheck,
                    factors: factors,
                    responseFormat: responseFormat))
        }
        return prompts
    }

    static func transcriptSchemaViolation(
        _ turns: [TranscriptTurn], itemID: String
    ) -> String? {
        guard !turns.isEmpty else {
            return "item '\(itemID)': transcript is empty — a scripted "
                + "transcript needs at least a final user turn"
        }
        for (index, turn) in turns.enumerated() {
            guard transcriptRoles.contains(turn.role) else {
                return "item '\(itemID)': transcript turn \(index + 1) has role "
                    + "'\(turn.role)' — allowed roles are system, user, assistant"
            }
            guard !turn.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                return "item '\(itemID)': transcript turn \(index + 1) has "
                    + "empty content"
            }
        }
        guard !turns.dropFirst().contains(where: { $0.role == "system" }) else {
            return "item '\(itemID)': transcript may carry at most one system "
                + "turn, and it must be first"
        }
        switch turns.last?.role {
        case "user":
            return nil
        case "assistant":
            return "item '\(itemID)': transcript ends with an assistant turn — "
                + "generation produces the assistant's reply to a final user "
                + "turn; assistant-prefix continuation is out of scope for "
                + "scripted-transcript studies (v1)"
        default:
            return "item '\(itemID)': transcript must end with a user turn "
                + "(generation produces the assistant's reply to it)"
        }
    }

    static func transcriptDisplayText(_ turns: [TranscriptTurn]) -> String {
        turns.last?.content ?? ""
    }

    static func attentionChecks(of prompts: [StudyPrompt]) -> [String: AttentionCheck] {
        var checks: [String: AttentionCheck] = [:]
        for prompt in prompts {
            if let check = prompt.attentionCheck { checks[prompt.id] = check }
        }
        return checks
    }
}
