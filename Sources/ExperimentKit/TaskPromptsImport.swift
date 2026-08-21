import Foundation

/// The Input Data "Import JSONL…" action: paste-or-file JSONL records go
/// through a parse PREVIEW (record count, how many carry `options`/`target`,
/// first error with its line number), then — only when every line parses —
/// land as the study's task-prompts file at the SAME destination the
/// readiness scaffold uses (`DataTemplates.taskPromptsDestination`), get set
/// as the manifest's prompts file, and are hash-pinned. One action from
/// paste to pinned.
///
/// This exists because the plain prompt-block editor treats pasted JSON as
/// PROMPT TEXT — a record's `options`/`target` would become literal prose
/// and the instrument fields would be lost. Import never coerces: garbage is
/// refused with a line number, and the detector below only OFFERS the import
/// path for JSONL-looking pastes (it never reinterprets silently).
public enum TaskPromptsImport {

    // MARK: - Parse preview

    public struct Preview: Sendable, Equatable {
        /// Parsed records (non-empty lines, each a JSON object with a
        /// `prompt`/`text` string and/or a scripted `transcript`).
        public var recordCount: Int
        /// Records carrying a non-empty `options` array (the categorical
        /// answer-token instrument's option set).
        public var optionsCount: Int
        /// Records carrying a `target` key.
        public var targetCount: Int
        /// Records carrying a non-empty scripted `transcript` array (the
        /// metacognition-study instrument).
        public var transcriptCount: Int

        public init(
            recordCount: Int, optionsCount: Int, targetCount: Int,
            transcriptCount: Int = 0
        ) {
            self.recordCount = recordCount
            self.optionsCount = optionsCount
            self.targetCount = targetCount
            self.transcriptCount = transcriptCount
        }

        public var summaryLine: String {
            var line =
                "\(recordCount) record\(recordCount == 1 ? "" : "s") — "
                + "\(optionsCount) with options, \(targetCount) with target"
            if transcriptCount > 0 {
                line += ", \(transcriptCount) with scripted transcripts"
            }
            return line
        }
    }

    public enum Outcome: Sendable, Equatable {
        /// Nothing but whitespace — nothing to import.
        case empty
        case preview(Preview)
        /// First bad line (1-based, counting EVERY line including blanks —
        /// the number matches what an editor shows) and why.
        case failure(line: Int, message: String)
    }

    /// Line-accurate parse preview. Acceptance matches
    /// `TaskPromptsDocument.load` (JSON object per non-empty line, `prompt`
    /// or `text` string key — or a scripted `transcript`) PLUS the run
    /// loop's transcript schema rules, so a passing preview guarantees both
    /// the import's document load AND the pin's `parseTaskPrompts` succeed.
    public static func preview(_ text: String) -> Outcome {
        var records = 0
        var options = 0
        var targets = 0
        var transcripts = 0
        var seenIDs: [String: Int] = [:]  // id → 1-based item ordinal
        let lines = text.components(separatedBy: "\n")
        for (index, raw) in lines.enumerated() {
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty { continue }
            guard
                let object = (try? JSONSerialization.jsonObject(with: Data(trimmed.utf8)))
                    as? [String: Any]
            else {
                return .failure(line: index + 1, message: "not a JSON object")
            }
            let transcript = object["transcript"] as? [Any]
            guard
                object["prompt"] is String || object["text"] is String
                    || transcript != nil
            else {
                return .failure(
                    line: index + 1,
                    message: "no \"prompt\", \"text\", or \"transcript\" key")
            }
            // Duplicate item ids are refused with the run loop's own message
            // (identical on both engines), and BEFORE the transcript checks —
            // the same ordering as `parseTaskPrompts`, so a passing preview
            // still guarantees the pin's parse succeeds.
            let itemID = object["id"] as? String ?? "prompt-\(records + 1)"
            if let firstItem = seenIDs[itemID] {
                return .failure(
                    line: index + 1,
                    message: ExperimentTasks.duplicateTaskPromptIDMessage(
                        id: itemID, firstItem: firstItem,
                        duplicateItem: records + 1))
            }
            seenIDs[itemID] = records + 1
            if let transcript {
                // The run loop's own schema rules (identical messages on
                // both engines) — a preview that passed but failed to pin
                // would be a lie.
                let turns = transcript.map { any -> ExperimentTasks.TranscriptTurn in
                    let turn = any as? [String: Any]
                    return ExperimentTasks.TranscriptTurn(
                        role: turn?["role"] as? String ?? "",
                        content: turn?["content"] as? String ?? "")
                }
                if let violation = ExperimentTasks.transcriptSchemaViolation(
                    turns, itemID: itemID)
                {
                    return .failure(line: index + 1, message: violation)
                }
                transcripts += 1
            }
            records += 1
            if let optionValues = object["options"] as? [Any], !optionValues.isEmpty {
                options += 1
            }
            if object["target"] != nil { targets += 1 }
        }
        guard records > 0 else { return .empty }
        return .preview(
            Preview(
                recordCount: records, optionsCount: options, targetCount: targets,
                transcriptCount: transcripts))
    }

    // MARK: - JSONL detector (the prompt-block editor's paste guard)

    /// True when content pasted into the PLAIN prompt-block editor looks
    /// like JSONL records: the first non-empty line parses as a JSON object
    /// carrying a `text` (or `prompt`) string. Used only to OFFER the Import
    /// JSONL path with an inline hint — never to reinterpret the paste.
    /// Plain prose that merely starts with "{" does not parse and stays
    /// plain prompt text.
    public static func looksLikeJSONL(_ text: String) -> Bool {
        guard
            let first = text.components(separatedBy: "\n")
                .map({ $0.trimmingCharacters(in: .whitespacesAndNewlines) })
                .first(where: { !$0.isEmpty })
        else { return false }
        guard first.hasPrefix("{"),
            let object = (try? JSONSerialization.jsonObject(with: Data(first.utf8)))
                as? [String: Any]
        else { return false }
        return object["text"] is String || object["prompt"] is String
            || object["transcript"] is [Any]
    }

    // MARK: - Import (write to the scaffold destination + pin)

    public struct ImportError: Error, CustomStringConvertible, Equatable {
        public let message: String
        public var description: String { message }

        public init(message: String) {
            self.message = message
        }
    }

    public struct ImportResult: Sendable, Equatable {
        /// Workspace-relative path the records landed at (also set as the
        /// manifest's `taskPromptsFile`).
        public var file: String
        public var recordCount: Int
        /// The pinned SHA-256 (manifest `taskPromptsHash`).
        public var hash: String
    }

    /// Validates, writes, sets, and pins in one step:
    /// - refuses garbage (the preview's first error, with its line number —
    ///   never a silent coercion),
    /// - writes the records FIELD-PRESERVINGLY through `TaskPromptsDocument`
    ///   to `DataTemplates.taskPromptsDestination(experiment:)` (the
    ///   readiness scaffold's destination rule) under the study-pack write
    ///   rule (`TabularImport.writeRefusingDifferingOverwrite`, shared with
    ///   the tabular importer): identical bytes are idempotent (just
    ///   re-pin), differing bytes REFUSE unless the caller passes
    ///   `replacingExisting: true` — the sheet's explicit "Replace the
    ///   existing file" checkbox. Re-importing an edited set into the same
    ///   draft is a legitimate workflow; a silent overwrite is not,
    /// - sets the file as the manifest's prompts file and pins its hash
    ///   (`ExperimentStore.pinTaskPrompts` — the run loop's own parser
    ///   re-checks the records).
    ///
    /// Transactional (the tabular importer's discipline): a failure after
    /// the write — the pin OR the caller's `persist` step — rolls the
    /// write back (removes a file this import created, restores the bytes
    /// it replaced), so a refused import leaves neither an orphaned file
    /// nor a clobbered one. Callers that save the manifest afterward MUST
    /// pass that save as `persist` so the whole import is atomic. Drafts
    /// only, same rule as every pin edit.
    @discardableResult
    public static func importIntoStudy(
        text: String, manifest: inout ExperimentManifest,
        replacingExisting: Bool = false,
        persist: (ExperimentManifest) throws -> Void = { _ in }
    ) throws -> ImportResult {
        switch preview(text) {
        case .empty:
            throw ImportError(message: "nothing to import — no non-empty lines")
        case .failure(let line, let message):
            throw ImportError(
                message: "refusing to import: line \(line) — \(message). "
                    + "Every line must be a JSON object with a \"prompt\" or "
                    + "\"text\" string (options/target ride along verbatim)")
        case .preview:
            break
        }
        let document = try TaskPromptsDocument.load(Data(text.utf8))
        let file = DataTemplates.taskPromptsDestination(experiment: manifest.name)
        let write: TabularImport.ImportWrite
        do {
            write = try TabularImport.writeRefusingDifferingOverwrite(
                document.serialized(), relativePath: file,
                replacingExisting: replacingExisting,
                differingRemedy: "check \"Replace the existing file\" to "
                    + "overwrite it deliberately, or move it aside and "
                    + "import again")
        } catch let problem as TabularImport.Problem {
            // One error surface for the sheet: refusals arrive as
            // ImportError whichever layer raised them.
            throw ImportError(message: problem.message)
        }
        do {
            let hash = try ExperimentStore.pinTaskPrompts(file, into: &manifest)
            try persist(manifest)
            return ImportResult(
                file: file, recordCount: document.count, hash: hash)
        } catch {
            TabularImport.rollBack(write, relativePath: file)
            throw error
        }
    }
}
