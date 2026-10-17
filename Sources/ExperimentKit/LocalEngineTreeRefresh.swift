import Foundation

/// Refresh Engine's file work: replace the materialized engine's SOURCES with
/// the running app's bundled ones, carrying the installed environment across.
///
/// Why this exists: after an app update the engine at `~/SteerLab/Engine` was
/// still copied by the earlier build (its stamp says so), and the only way to
/// re-copy it used to be the first-time materialization, which replaces the
/// whole `Server/` tree — including `Server/.venv.nosync`, the 2 GB of
/// installed packages. The packages do not depend on the sources (they are
/// installed from the lock, and the engine itself is an editable install
/// whose path does not change), so a refresh keeps them.
///
/// Every step is a rename on one volume except the first copy, and a failure
/// at any step puts the original tree back as it was.
public enum LocalEngineTreeRefresh {
    /// Kept from the old tree: the installed environment, and the metadata
    /// folder an editable install leaves beside the sources.
    public static func isCarriedOver(_ name: String) -> Bool {
        name == ".venv.nosync" || name.hasSuffix(".egg-info")
    }

    /// Replace the contents of `serverDirectory` with a copy of
    /// `packageRoot`, moving the carried-over entries into the new tree.
    /// Returns the number of regular files copied from `packageRoot`.
    @discardableResult
    public static func replaceSources(
        in serverDirectory: URL, from packageRoot: URL,
        fileManager fm: FileManager = .default
    ) throws -> Int {
        let parent = serverDirectory.deletingLastPathComponent()
        let token = UUID().uuidString.prefix(8)
        let incoming = parent.appending(component: "Server.incoming-\(token)")
        let replaced = parent.appending(component: "Server.replaced-\(token)")

        try fm.copyItem(at: packageRoot, to: incoming)
        let count = regularFileCount(under: incoming, fileManager: fm)

        let shipped = Set((try? fm.contentsOfDirectory(atPath: incoming.path)) ?? [])
        let carried = ((try? fm.contentsOfDirectory(atPath: serverDirectory.path)) ?? [])
            .filter { isCarriedOver($0) && !shipped.contains($0) }
            .sorted()

        var movedIn: [String] = []
        func putCarriedBack(into directory: URL) {
            for name in movedIn.reversed() {
                try? fm.moveItem(
                    at: incoming.appending(component: name),
                    to: directory.appending(component: name))
            }
        }
        do {
            for name in carried {
                try fm.moveItem(
                    at: serverDirectory.appending(component: name),
                    to: incoming.appending(component: name))
                movedIn.append(name)
            }
        } catch {
            putCarriedBack(into: serverDirectory)
            try? fm.removeItem(at: incoming)
            throw error
        }

        do {
            try fm.moveItem(at: serverDirectory, to: replaced)
        } catch {
            putCarriedBack(into: serverDirectory)
            try? fm.removeItem(at: incoming)
            throw error
        }

        do {
            try fm.moveItem(at: incoming, to: serverDirectory)
        } catch {
            // Original tree back first, then the environment back into it.
            try? fm.moveItem(at: replaced, to: serverDirectory)
            putCarriedBack(into: serverDirectory)
            try? fm.removeItem(at: incoming)
            throw error
        }
        try? fm.removeItem(at: replaced)
        return count
    }

    static func regularFileCount(under root: URL, fileManager fm: FileManager) -> Int {
        guard let enumerator = fm.enumerator(
            at: root, includingPropertiesForKeys: [.isRegularFileKey])
        else { return 0 }
        var count = 0
        for case let url as URL in enumerator
        where (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true {
            count += 1
        }
        return count
    }
}
