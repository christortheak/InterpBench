import Foundation
import SteeringKit

/// The user-facing wording, factored out so it is testable and so the three
/// surfaces say the same thing.
public enum InjectionModeCopy {

    public static let pickerHelp =
        "Steer ADDS a fixed amount of the concept (h + α·v) whatever the "
        + "model was already doing. Ablate REMOVES whatever of the concept is "
        + "present (h − λ·(h·v̂)v̂), leaving the rest of the residual stream "
        + "untouched. Steering with a negative α is not the same thing: it "
        + "pushes past zero wherever the concept was weak, asserting the "
        + "opposite concept."

    public static let alphaHelp =
        "α in units of the layer's residual-stream norm, so it is comparable "
        + "across concepts and layers. The layer is widened by the variant's "
        + "band width."

    public static let lambdaHelp =
        "λ = 1 removes the concept completely. Below 1 removes part of it. "
        + "λ = 2 reflects it — the component flips sign while the residual "
        + "stream's length is preserved exactly, which is a cleaner "
        + "'reverse the concept' than a large negative α. λ needs no "
        + "residual-norm units: ablation removes exactly what is present, so "
        + "it already scales itself."

    /// Short label beside the λ field naming what the current value does.
    public static func lambdaLabel(_ lambda: Double) -> String {
        switch lambda {
        case ..<0: return "adds the concept back — did you mean Steer?"
        case 0: return "no effect"
        case 0..<1: return "partial removal"
        case 1: return "full removal"
        case 1..<2: return "removes and overshoots"
        case 2: return "reflection — flips the concept, keeps the norm"
        default: return "beyond reflection — amplifies the opposite"
        }
    }

    /// Values a researcher may not have meant. Flags rather than refuses:
    /// they are all legal, and a study may want them declared deliberately.
    public static func lambdaIsUnusual(_ lambda: Double) -> Bool {
        lambda < 0 || lambda > 2
    }

    /// The plain sentence under the controls.
    public static func explanation(
        mode: InterventionPlan.Mode, concept: String?, strength: Double
    ) -> String {
        let name = (concept?.isEmpty == false) ? concept! : "this concept"
        switch mode {
        case .add:
            return "Adds \(name) to the residual stream at the chosen layer "
                + "and every token the model writes, whether or not it was "
                + "already there."
        case .ablate:
            let amount: String
            switch strength {
            case 1: amount = "Removes"
            case 0..<1: amount = "Removes part of"
            case 2: amount = "Reflects"
            default: amount = "Removes (λ \(strength.formatted())) "
            }
            return "\(amount) \(name) wherever the model represents it — "
                + "every layer, and every token including the whole prompt, "
                + "not just the text it writes. Nothing is added, so the "
                + "residual stream keeps everything unrelated to \(name)."
        }
    }
}
