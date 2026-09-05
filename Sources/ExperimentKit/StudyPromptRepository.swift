import CryptoKit
import Foundation
import SteeringKit

/// Reads pinned prompt bytes from a supplied project, preserving load-time drift gates.
struct StudyPromptRepository {
    typealias StudyPrompt = ExperimentTasks.StudyPrompt
    let projectRoot: URL

    func load(
        for manifest: ExperimentManifest, override: String? = nil
    ) throws -> (file: String, hash: String, prompts: [StudyPrompt]) {
        let file = override ?? manifest.taskPromptsFile ?? "prompts/dev/dev-prompts.jsonl"
        let url =
            file.hasPrefix("/")
            ? URL(filePath: file)
            : projectRoot.appending(path: file)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ExperimentError.refusing(
                .missingPrerequisite,
                "task prompt file not found: \(url.path)",
                repair: "author \(file) as {\"id\": …, \"prompt\": …} JSONL rows, "
                    + "then steerlab-cli experiment pin-prompts "
                    + "\(manifest.name) \(file)")
        }
        let data = try Data(contentsOf: url)
        // SHA-256 over the raw bytes — identical to `StimulusSet.loadTexts`
        // and the server, so existing pinned hashes stay valid.
        let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        let frozen = manifest.status != .draft
        if override == nil {
            if let pinned = manifest.taskPromptsHash, pinned != hash {
                throw ExperimentError.refusing(
                    .pinDrift,
                    "task prompts '\(file)' drifted from the pinned hash "
                        + "(have \(hash.prefix(12))…, pinned \(pinned.prefix(12))…)",
                    repair: "restore \(file) to its pinned bytes ; then "
                        + "steerlab-cli experiment run \(manifest.name)  (on a "
                        + "DRAFT, re-pin instead: steerlab-cli experiment "
                        + "pin-prompts \(manifest.name) \(file))")
            }
            if frozen, manifest.taskPromptsHash == nil {
                // The prose keeps its shape (it already named the three steps);
                // the machine repair is the same three as one runnable line.
                throw ExperimentError.refusing(
                    .missingPrerequisite,
                    "frozen study has no pinned task prompts — duplicate it "
                        + "('steerlab-cli experiment duplicate \(manifest.name) "
                        + "<new-name>'), pin the prompt set ('steerlab-cli experiment "
                        + "pin-prompts <new-name> prompts/…/file.jsonl'), and re-freeze",
                    repair: "steerlab-cli experiment duplicate \(manifest.name) "
                        + "\(manifest.name)-v2 && steerlab-cli experiment "
                        + "pin-prompts \(manifest.name)-v2 prompts/…/file.jsonl "
                        + "&& steerlab-cli experiment freeze \(manifest.name)-v2 "
                        + "&& steerlab-cli experiment run \(manifest.name)-v2")
            }
        } else if frozen, hash != manifest.taskPromptsHash {
            throw ExperimentError.refusing(
                .pinDrift,
                "prompt override on a FROZEN study must match the pinned prompt "
                    + "set byte-for-byte — duplicate the experiment to iterate",
                repair: "steerlab-cli experiment run \(manifest.name)  (without "
                    + "--prompts: the pin IS the measured task), or steerlab-cli "
                    + "experiment duplicate \(manifest.name) \(manifest.name)-v2 "
                    + "&& steerlab-cli experiment pin-prompts "
                    + "\(manifest.name)-v2 \(file)")
        }
        return (file, hash, try StudyPromptParsing.parseTaskPrompts(data))
    }
}
