import Foundation

/// What a source dataset can BECOME, and where pressing the button actually
/// goes (WP-Data phase 3, part B).
///
/// The Data section is being reorganized around "Dataset → Check → Derive".
/// Phase 1 made the dataset visible and phase 2 made it land in the right
/// place; this is the third verb. It is deliberately the SMALLEST possible
/// third verb:
///
/// - **It routes; it never builds.** Every action switches to the tool that
///   already owns that derivation, preconfigured as far as an existing seam
///   allows. No extraction, no probe fit, no PCA is started from the
///   inventory, so no builder's own gates (model loaded, enough rows, recipe
///   valid, freshness advice) can be bypassed by arriving this way.
/// - **The destination sentence is part of the action.** `destination` says
///   what the click does *and what it does not do* — an action that only
///   selects a concept says so rather than implying the recipe is set.
/// - **Gating is by ROLE, not by hope.** A validation set has no derivation
///   at all: it is evidence, not a source. That is a rule with a reason, and
///   `noDerivationReason` carries the reason instead of an empty toolbar.
///
/// Pure and engine-side so the gating is unit-testable without a view.
public enum DatasetDerivation {

    /// The derivations offered for a row, in the order they are shown.
    /// Empty is a legitimate answer (see `noDerivationReason`).
    public static func actions(for entry: DatasetInventoryEntry) -> [DatasetDerivationAction] {
        switch entry.kind {
        case .conceptStimuli, .pairedStimuli:
            guard let concept = entry.conceptName else { return [] }
            return [.buildVector(concept: concept)]
        case .grandMeanCorpus:
            guard let concept = entry.conceptName else { return [] }
            return [.buildGrandMeanVector(concept: concept)]
        case .probeItems:
            guard let concept = entry.conceptName else { return [] }
            return [.trainProbe(concept: concept)]
        case .neutralCorpus:
            return [.buildNeutralPCs]
        case .validationSet, .capabilityBattery, .promptSet:
            // Measurement-side inputs. A study READS them; nothing is
            // derived FROM them.
            return []
        }
    }

    /// Why a row offers nothing — shown in place of the buttons, so an empty
    /// action row is never mistaken for a missing feature.
    public static func noDerivationReason(for entry: DatasetInventoryEntry) -> String? {
        guard actions(for: entry).isEmpty else { return nil }
        switch entry.kind {
        case .validationSet:
            return "A held-out set is EVIDENCE, not a source: nothing is "
                + "derived from it. It is read by the convergent validate "
                + "gate, and its sha256 is pinned as the concept's "
                + "validationHash and re-checked at freeze."
        case .capabilityBattery:
            return "A battery is an instrument a run SCORES with, not a "
                + "source to derive from. Pin it on a study (capability "
                + "battery) or a sweep, and every condition's run reads it."
        case .promptSet:
            return "A prompt set is what a study or sweep RUNS, not a source "
                + "to derive from. Pin it on the study as its task prompts, "
                + "or on the sweep as its dev split."
        case .conceptStimuli, .pairedStimuli, .grandMeanCorpus, .probeItems:
            // Reachable only for a nameless row, which the scan does not
            // produce today — stated rather than crashed on.
            return "This row carries no concept name, so there is nothing to "
                + "preselect in the builder."
        case .neutralCorpus:
            return nil
        }
    }
}

/// One offered derivation. The associated concept is the routing key — the
/// same string `ConceptBuilder.selectedExisting` takes.
public enum DatasetDerivationAction: Sendable, Equatable, Identifiable {
    /// Paired stimuli → the Concept Vector Builder, concept selected. The
    /// recipe is left alone: a paired concept's recipe is already resolved
    /// FROM DISK when the builder loads it
    /// (`ConceptBuilder.pairedRecipeFamilyOnDisk`), and overriding that would
    /// contradict the files.
    case buildVector(concept: String)
    /// Story corpus → the same builder, concept selected AND the grand-mean
    /// recipe chosen.
    case buildGrandMeanVector(concept: String)
    /// Probe items → the same builder; probe training data and the probe fit
    /// live in its Probe Training Data section.
    case trainProbe(concept: String)
    /// Neutral corpus → the same builder's neutral-PC section.
    case buildNeutralPCs

    public var id: String {
        switch self {
        case .buildVector(let concept): "buildVector:\(concept)"
        case .buildGrandMeanVector(let concept): "buildGrandMeanVector:\(concept)"
        case .trainProbe(let concept): "trainProbe:\(concept)"
        case .buildNeutralPCs: "buildNeutralPCs"
        }
    }

    public var title: String {
        switch self {
        case .buildVector: "Build vector…"
        case .buildGrandMeanVector: "Build vector (grand-mean)…"
        case .trainProbe: "Train probe…"
        case .buildNeutralPCs: "Build neutral PCs…"
        }
    }

    /// The concept to preselect, when the action has one.
    public var concept: String? {
        switch self {
        case .buildVector(let concept), .buildGrandMeanVector(let concept),
            .trainProbe(let concept):
            concept
        case .buildNeutralPCs:
            nil
        }
    }

    /// True only where a CLEAN seam exists to set the recipe as well as the
    /// selection. `ConceptBuilder.recipeFamily` is a plain settable property
    /// whose `didSet` performs the whole recipe switch, so grand-mean can be
    /// chosen honestly — but it must be set AFTER the selection, because
    /// loading a concept resolves the recipe from its files.
    public var setsGrandMeanRecipe: Bool {
        if case .buildGrandMeanVector = self { return true }
        return false
    }

    /// What the click does, said plainly — including the part it leaves to
    /// the researcher. Rendered verbatim as the button's help.
    public var destination: String {
        switch self {
        case .buildVector(let concept):
            "opens Concepts & Vectors with '\(concept)' selected. The recipe "
                + "is the one its files already imply; extraction still runs "
                + "from the builder, behind its own gates."
        case .buildGrandMeanVector(let concept):
            "opens Concepts & Vectors with '\(concept)' selected and the "
                + "grand-mean recipe chosen. Nothing is extracted yet — the "
                + "builder's own controls start the build."
        case .trainProbe(let concept):
            "opens Concepts & Vectors with '\(concept)' selected; its Probe "
                + "Training Data section holds the labelled items and the "
                + "probe fit. There is no separate probe tool to route to."
        case .buildNeutralPCs:
            "opens Concepts & Vectors, whose neutral-corpus section builds a "
                + "neutral-PC basis from the pinned corpus (it loads the "
                + "selected model itself if none is loaded)."
        }
    }
}
