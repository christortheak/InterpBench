import ExperimentKit
import SteeringKit
import SwiftUI
import UniformTypeIdentifiers

enum StudyControlCopy {
    static let saveBackHelp =
        "overwrites the design this study was minted from with this study's "
        + "current settings (agents stripped). The design's content hash "
        + "changes; studies already minted from it keep their original lineage "
        + "stamps."

    static let saveAsNewDesignHelp =
        "adds a NEW design to the library from this study's settings, leaving "
        + "any design it came from untouched. An unchanged instance of a live "
        + "design selects that design instead of minting a near-duplicate."

    static let duplicateHelp =
        "copies EVERYTHING, agents included — the only way to iterate on a "
        + "frozen study. (Saving a study as a DESIGN, in Templates, does the "
        + "opposite: it strips the agents and re-derives the derived pins.)"

    static let deleteStudyHelp =
        "moves the draft's directory to a .trash-<timestamp> sibling under "
        + "experiments/ — never a destructive delete. Frozen and completed "
        + "studies cannot be deleted at all: their runs stamp them."

    static let freezeHelp =
        "ONE-WAY: verifies every pinned input, requires a pinned model revision, "
        + "and stamps the content hash and git commit. Agent-comparison "
        + "studies pin each agent (variant artifact) by artifact hash; concept-vector "
        + "studies also require a matching validate run. Settings must be "
        + "frozen before behavior is measured — iterate afterwards by duplicating. "
        + "CLI: freeze --force skips validation gates"

    static let remoteFreezeHelp =
        "ONE-WAY, executed by the ACTIVE SERVER: the server verifies every "
        + "pinned input, evaluates the freeze gates against ITS OWN substrate's "
        + "validation evidence (validation evidence counts on the substrate "
        + "that freezes — validate on the server first), stamps frozenBy: "
        + "\"server\" + the content hash, and exports preregistration.md. "
        + "Freeze stamps the server-resident copy; before submitting, the app "
        + "verifies that copy IS the manifest shown here (field-level "
        + "comparison, volatile freeze stamps excluded) and refuses on a "
        + "mismatch. On a paired workspace the frozen manifest appears here "
        + "immediately. No force option in the app — forcing requires the CLI"

    static let runHelp =
        "executes the study in-app: verifies pins, loads the model at the "
        + "pinned revision, runs the baseline plus pinned agents for every "
        + "task prompt, and writes generations, metrics.csv, report.json, "
        + "and the manifest snapshot into a new immutable runs/ directory. "
        + "Legacy concept-vector studies re-derive their vectors before running. "
        + "Currently requires Temperature = 0 because mlx-swift-lm does not "
        + "expose a per-run seed for reproducible sampling"

    static let validateHelp =
        "creates the validation evidence required by Freeze Study: verifies "
        + "the pinned inputs, loads the model at its pinned revision (or pins "
        + "the local cached revision for drafts), re-derives concept vectors, "
        + "runs never-named validation scenarios when present, writes the "
        + "cross-concept cosine matrix, and stores a manifest snapshot in a "
        + "new runs/ validate directory"

    static let extractHelp =
        "re-derives the pinned concepts' steering vectors from the frozen "
        + "recipe (stimulus hashes + extraction options) into a new immutable "
        + "runs/ extract directory — the CLI 'experiment extract' verb. "
        + "Deterministic re-derivation, so drafts AND frozen studies may run it"

    static let outcomeModeHelp =
        "dispatches real instruments — not a note. Generated choice: the "
        + "run samples text and parses answers from the prose. Answer-token "
        + "probability: no endpoint prose — deterministic logprob records "
        + "over the declared options. Both: the two side by side. Written "
        + "to `outcomeInstruments`; the run's records and Results views "
        + "differ accordingly"

    static let remoteOptionsCaption =
        "the unified Run button packages this study as a hash-pinned bundle "
        + "and submits it with these options — the portable path that works "
        + "whether or not the study exists in the server's own workspace."

    static let unifiedRemoteRunHelp =
        "packages this study as a hash-pinned bundle, uploads it, and submits "
        + "the selected verb as a durable server job. The server preflights "
        + "the submission (memory fit, walltime, quota) — warnings show "
        + "inline; a failing verdict stops the submission unless explicitly "
        + "forced."

    static let importEvidenceCaption =
        "Import Evidence verifies an evidence bundle's hashes and lands it "
        + "under runs/ as an immutable imported run."

    static let recentJobsCaption =
        "jobs persist on the server — reconnect any time from Compute or here"

    static let studyDtypeHelp =
        "the numeric precision the study model runs in on the compute "
        + "substrate. A PIN, not a hint: greedy decoding is not "
        + "precision-proof — at a near-tie between two tokens, bf16 and fp16 "
        + "round differently and the continuation diverges — so two runs at "
        + "different precisions are not the same measurement. Leave it on "
        + "\"device default\" to let the substrate choose, which is what "
        + "every study did before this pin existed. Honored by the server; "
        + "the Mac validates it here so a bad value cannot reach the cluster."

    static let seatPickerHelp =
        "the agent that speaks for this role. 'baseline' is the study's own "
        + "model with no intervention — a real condition (the control "
        + "composition), not an empty seat. Only agents built on this study's "
        + "base model are eligible."

    static let saveCastingHelp =
        "compiles this scenario plus the casting into a bound scenario under "
        + "prompts/panels/compiled/ and pins it as the study's scenario — the "
        + "same write a design's instantiation table performs. The scenario "
        + "itself is not modified."

    static let permutedSiblingsHelp =
        "one study runs ONE casting (a panel's arms are the fixed "
        + "baseline/configured pair), so re-seating the same agents means "
        + "sibling studies. This saves the study as a design and opens the "
        + "new-studies table holding every distinct re-seating of its current "
        + "cast — swapping two identical occupants is the same panel, so those "
        + "are deduped rather than run twice."

    static let canonicalNameHelp =
        "lowercase letters, digits and hyphens; anything else is dropped. This "
        + "IS the experiments/<name>/ directory and the name the CLI takes."

    static func validateCaption(variantsPresent: Bool) -> String {
        "derives vectors from the pinned recipe and writes scope-hashed "
            + "validation evidence (held-out probe accuracy, cross-concept geometry"
            + (variantsPresent ? ", capability battery per agent" : "")
            + "). Optional while drafting — freeze REQUIRES matching evidence."
    }
}
