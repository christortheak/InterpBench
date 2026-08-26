import ExperimentKit
import SteeringKit
import SwiftUI

/// The two extraction declarations as controls: WHERE the residual stream is
/// read, and HOW the stimulus reached the model.
///
/// One definition, two thin call sites — the Studies attach row (which pins
/// them into a manifest through `ExperimentStore.attachConcept`) and the
/// Concepts vector builder (which extracts with them locally). Neither control
/// validates anything: a selection becomes the cross-engine declaration text
/// and the ENGINE answers, so the app can never refuse something the CLI
/// accepts, or accept something it refuses.
///
/// Layout note (macOS 27): every conditional field here is a sibling inside an
/// existing row, at a fixed width. Nothing this file draws declares a frame
/// minimum, so no split-view column minimum moves when a parameter field or a
/// chat-template sub-option appears.

// MARK: - Reading position

struct ReadingPositionField: View {
    @Binding var choice: ReadingPositionChoice
    @Binding var parameter: Int
    /// What the "declare nothing" entry says, or nil to leave it out (the
    /// builder always reads at a concrete position).
    var defaultCaption: String?
    var help: String

    var body: some View {
        HStack(spacing: 6) {
            Picker("", selection: $choice) {
                if let defaultCaption {
                    Text(defaultCaption).tag(ReadingPositionChoice.recipeDefault)
                }
                ForEach(ReadingPositionChoice.declarableCases) { entry in
                    Text(entry.menuLabel).tag(entry)
                }
            }
            .labelsHidden()
            .frame(width: 186)
            .help(help)
            // Carrying the number into a new kind's range is the MODEL's job
            // (`ExperimentPanel`/`ConceptBuilder` both do it on the way in),
            // so the two sites cannot drift into two conveniences.
            if let caption = choice.parameterCaption {
                TextField(caption, value: $parameter, format: .number)
                    .frame(width: 52)
                    .help(Self.parameterHelp(choice, caption: caption))
            }
        }
    }

    static func parameterHelp(
        _ choice: ReadingPositionChoice, caption: String
    ) -> String {
        let body: String = switch choice {
        case .meanFromToken:
            "average the residual stream over token positions from K onward"
        case .offsetFromEnd:
            "the single token k back from the end of the SEQUENCE — k = 0 is "
                + "the last token"
        case .postInstruction:
            "Arditi's convention: the i-th token AFTER the instruction "
                + "content, i in 1…5"
        case .contentOffset:
            "the single token k back into what the stimulus itself said — "
                + "k = 0 is the last content token"
        case .meanContentFromToken:
            "average over CONTENT token positions from the n-th on, skipping "
                + "the template's own tokens"
        case .recipeDefault, .lastToken, .lastContentToken, .turnCloseToken:
            "this position takes no parameter"
        }
        return "\(caption): \(body)"
    }
}

// MARK: - Extraction rendering

struct ExtractionRenderingField: View {
    @Binding var choice: ExtractionRenderingChoice
    var help: String

    var body: some View {
        HStack(spacing: 6) {
            Picker("", selection: $choice.mode) {
                Text("raw").tag(ExtractionRendering.Mode.raw)
                Text("chat template").tag(ExtractionRendering.Mode.chatTemplate)
            }
            .labelsHidden()
            .frame(width: 140)
            .help(help)
            if choice.mode == .chatTemplate {
                Picker("", selection: $choice.voice) {
                    Text("as user").tag(ExtractionRendering.Voice.user)
                    Text("as assistant").tag(ExtractionRendering.Voice.assistant)
                }
                .labelsHidden()
                .frame(width: 124)
                .help(
                    "whose turn the stimulus is rendered as. 'as user' is the "
                        + "legacy voice — the model READS the stimulus. 'as "
                        + "assistant' renders it as the model's OWN output; the "
                        + "two engines differ on which can render that, and the "
                        + "declaration itself answers")
                if choice.voice == .user {
                    Toggle("generation prompt", isOn: $choice.addGenerationPrompt)
                        .help(
                            "append the family's generation prompt after the "
                                + "stimulus turn — what a measured generation "
                                + "does, and the default. Turning it off changes "
                                + "the trailing template tokens, and therefore "
                                + "every named reading position that lands on them")
                }
            }
        }
    }
}
