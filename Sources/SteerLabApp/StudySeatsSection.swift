import ExperimentKit
import SteeringKit
import SwiftUI

/// Seat casting presentation; SeatCasting and the coordinator retain the rules and writes.
struct StudySeatsSection: View {
    let manifest: ExperimentManifest
    let panel: ExperimentPanel

    /// WHO sits in each seat of the study's scenario — the casting.
    ///
    /// It exists here because a scenario chosen in Study Setup is SEMANTIC: it
    /// declares roles, turns and materials and binds no model to any seat, so a
    /// study that only picked one refuses at run start (deliberately — see
    /// `PanelComposition`). Casting used to be reachable only through a
    /// design's instantiation table, which meant a directly-authored panel
    /// study had no way to become runnable at all.
    ///
    /// Every rule rendered here lives in `SeatCasting` / `ExperimentPanel`
    /// (ExperimentKit, unit-tested). This view decides nothing: it reads the
    /// state, binds the pickers, and calls the two actions.
    @ViewBuilder
    var body: some View {
        if let casting = panel.seatCasting {
            let refusal = panel.seatCastingRefusal(casting)
            Section("Seats") {
                if casting.seats.isEmpty {
                    Text(
                        "this scenario declares no seats — add roles to it in "
                            + "the Panels editor first"
                    )
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                }
                ForEach(casting.seats) { seat in
                    if casting.isEditable, manifest.status == .draft {
                        Picker(
                            seat.name,
                            selection: seatBinding(seat: seat.id, panel: panel)
                        ) {
                            Text("baseline").tag(String?.none)
                            ForEach(panel.availableAgentsForSeats) { agent in
                                Text(agent.artifact.name)
                                    .tag(String?.some(agent.id))
                            }
                        }
                        .help(StudyControlCopy.seatPickerHelp)
                    } else {
                        LabeledContent(
                            seat.name,
                            value: casting.occupants[seat.id]?.label ?? "baseline"
                        )
                        .font(.caption)
                    }
                }
                if casting.isEditable, panel.availableAgentsForSeats.isEmpty {
                    Text(
                        "no saved agents use this study's base model "
                            + "(\(manifest.modelID)) — every seat can only be "
                            + "baseline until one exists (build one in Agents)"
                    )
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                }
                ForEach(casting.advisories, id: \.self) { advisory in
                    Text(advisory)
                        .font(.caption2)
                        .foregroundStyle(.orange)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
                if casting.isEditable {
                    Text(Self.castingStateLine(casting))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    HStack(spacing: 8) {
                        Button("Save Casting") { panel.saveSeatCasting() }
                            .disabled(refusal != nil || casting.seats.isEmpty)
                            .help(refusal ?? StudyControlCopy.saveCastingHelp)
                        Button("Create permuted siblings…") {
                            panel.startPermutedSiblings()
                        }
                        .disabled(casting.form != .cast)
                        .help(
                            casting.form == .cast
                                ? StudyControlCopy.permutedSiblingsHelp
                                : "save this study's casting first — permuted "
                                    + "siblings re-seat the cast it is running")
                    }
                    if let refusal {
                        Text(refusal)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
            }
        }
    }

    private func seatBinding(
        seat: String, panel: ExperimentPanel
    ) -> Binding<String?> {
        Binding(
            get: { panel.seatAgentID(for: seat) },
            set: { panel.setSeatAgent($0, seat: seat) })
    }

    /// What the study is currently pinning, in one line — the difference
    /// between "this is what will run" and "this is what will run once you
    /// save" is the whole point of the section.
    private static func castingStateLine(_ casting: SeatCasting.State) -> String {
        switch casting.form {
        case .uncast:
            return "not cast yet: this scenario binds no model to any seat, so "
                + "the study refuses at run start until Save Casting compiles "
                + "it. Save Study Setup does the same compile."
        case .cast:
            return "cast: the study pins a compiled copy of this scenario with "
                + "every seat bound. Saving again recompiles it at the study's "
                + "current model and sampling settings."
        case .legacyBound:
            return ""
        }
    }
}
