import Foundation

/// Decisions over manifest values, independent of filesystem or selected workspace.
/// This preserves the store's existing transition order, messages and repair actions.
/// The caller reads the existing manifest, admits the proposed save here, then
/// persists through its repository. This type performs no reads or writes.
enum ManifestMutationPolicy {
    static func admitSave(
        _ manifest: ExperimentManifest, existing: ExperimentManifest?,
        allowCreate: Bool = false, mayClearArms: Bool = false
    ) throws {
        if let existing {
            switch existing.status {
            case .draft:
                if !mayClearArms, holdsArms(existing), !holdsAnySurface(manifest) {
                    throw ExperimentError.refusing(
                        .armsCleared,
                        "refusing to save '\(manifest.name)' with no concepts "
                            + "and no conditions over a draft that has "
                            + "\(existing.concepts.count) concept(s) and "
                            + "\(existing.conditions.count) condition(s) — a "
                            + "manifest does not lose its whole measured "
                            + "surface in one write by accident",
                        repair: clearedArmsRepair(manifest.name))
                }
            case .frozen:
                // Only completion is allowed, and nothing else may change.
                var completed = existing
                completed.status = .complete
                guard manifest == completed else {
                    throw ExperimentError.refusing(
                        .statusImmutable,
                        "experiment '\(manifest.name)' is frozen — duplicate it "
                            + "to iterate",
                        repair: duplicateToIterateRepair(manifest.name))
                }
            case .complete:
                throw ExperimentError.refusing(
                    .statusImmutable,
                    "experiment '\(manifest.name)' is complete and immutable",
                    repair: duplicateToIterateRepair(manifest.name))
            }
        } else if !allowCreate {
            throw ExperimentError(reason: "experiment '\(manifest.name)' does not exist")
        }
    }

    static func holdsArms(_ manifest: ExperimentManifest) -> Bool {
        !manifest.concepts.isEmpty || !manifest.conditions.isEmpty
    }

    static func holdsAnySurface(_ manifest: ExperimentManifest) -> Bool {
        holdsArms(manifest) || !manifest.variantConditions.isEmpty
    }

    static func clearedArmsRepair(_ name: String) -> String {
        "steerlab-cli experiment verify \(name)  "
            + "# the manifest on disk still holds its arms; re-attach what "
            + "the caller dropped (steerlab-cli experiment attach \(name) "
            + "<concept>… ; steerlab-cli experiment declare-condition "
            + "\(name) …), or author the cleared study as its own draft "
            + "with steerlab-cli experiment duplicate \(name) \(name)-v2"
    }

    static func duplicateToIterateRepair(_ name: String) -> String {
        "steerlab-cli experiment duplicate \(name) \(name)-v2 && "
            + "steerlab-cli experiment <the verb you just ran> \(name)-v2 …  "
            + "(frozen studies are immutable; the duplicate is a draft again)"
    }

    static func admitDraftEdit(_ manifest: ExperimentManifest) throws {
        let name = manifest.name
        guard manifest.status == .draft else {
            throw ExperimentError.refusing(
                .statusImmutable,
                "experiment '\(name)' is \(manifest.status.rawValue) — "
                    + "duplicate it to iterate",
                repair: duplicateToIterateRepair(name))
        }
    }

    static func admitFreeze(_ manifest: ExperimentManifest) throws {
        let name = manifest.name
        guard manifest.status == .draft else {
            // Typed since gate-5 dry run #2 (P3): `freeze` was the last
            // manifest-WRITING verb whose immutability refusal arrived as an
            // untyped `verbFailed`/70 — an operational failure, to an agent —
            // while every other writer already answered `statusImmutable`/65
            // with a runnable duplicate-to-iterate repair. Prose unchanged.
            throw ExperimentError.refusing(
                .statusImmutable,
                "'\(name)' is already \(manifest.status.rawValue)",
                repair: duplicateToIterateRepair(name))
        }
    }
}
