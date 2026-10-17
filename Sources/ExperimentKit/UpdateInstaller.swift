import CryptoKit
import Foundation
import Observation

// MARK: - Click-to-install updates: the policy and the steps
//
// `UpdateCheck.swift` finds a newer release. This file installs it, and only
// after the person clicks Install Update and confirms the sheet that says what
// will happen. Every decision here is pure over injected seams (the network,
// the file system, the code-signature check, the tools, and the clock), so
// the whole path is unit-tested without a socket or a signing identity;
// `UpdateInstallerLive.swift` holds the real seams.
//
// The order is fixed, and nothing is moved until everything has passed:
//
//   1. WHO GETS IT (`UpdateInstallEligibility`). Only a Developer ID–signed,
//      notarized copy of the packaged app, in a folder the person can write
//      to, with no code checkout beside it. Anything else keeps the
//      signpost, with one plain line saying why.
//   2. DOWNLOAD. The release's app archive and its `.sha256`, chosen from the
//      release the check found (`UpdateReleaseAssets`), over HTTPS from
//      GitHub's hosts only, into a staging folder beside the app (so the final
//      move is a rename on one volume). Size, time, and Cancel are bounded.
//   3. VERIFY. The archive matches its checksum (integrity only: both come
//      from the same place); its file list is safe to unpack; `ditto -x -k`
//      unpacks it keeping extended attributes (the MLX shader library's
//      signature lives in one); the unpacked app passes strict validation of
//      all nested code against a requirement built from the RUNNING app's own
//      bundle identifier and Developer ID team, plus "notarized"; and its
//      version is newer than the running one.
//   4. INSTALL. The running app is renamed to
//      `SteerLab-<current version>-previous.app` beside it and the new app is
//      moved into the original path, so `~/.local/bin/steerlab-cli` (a
//      symlink into `Contents/Helpers`) stays valid. Any failure puts the
//      original back. A small detached step then waits for this process to
//      exit and opens the new app.
//
// What leaves the machine: the two downloads above, and nothing else. The
// notarization check reads the ticket stapled inside the downloaded app and
// is told not to use the network.

// MARK: - Limits and names

/// Bounds on what an update may cost. The release archive is about 50 MB and
/// 4,200 files today; each bound leaves ample room and still stops a runaway.
public struct UpdateInstallLimits: Sendable, Equatable {
    public var maximumArchiveBytes: Int64
    public var maximumChecksumBytes: Int64
    public var maximumUnpackedBytes: Int64
    public var maximumArchiveEntries: Int
    /// The whole download, start to finish.
    public var downloadDeadline: TimeInterval

    public init(
        maximumArchiveBytes: Int64 = 1_000_000_000,
        maximumChecksumBytes: Int64 = 4_096,
        maximumUnpackedBytes: Int64 = 4_000_000_000,
        maximumArchiveEntries: Int = 100_000,
        downloadDeadline: TimeInterval = 30 * 60
    ) {
        self.maximumArchiveBytes = maximumArchiveBytes
        self.maximumChecksumBytes = maximumChecksumBytes
        self.maximumUnpackedBytes = maximumUnpackedBytes
        self.maximumArchiveEntries = maximumArchiveEntries
        self.downloadDeadline = downloadDeadline
    }

    public static let standard = UpdateInstallLimits()
}

/// Where an update is staged, and what the kept copy is called.
public enum UpdateStagingLayout {
    /// Hidden folder BESIDE the installed app: the same volume, so the final
    /// move is a rename. Removed after an install, a failure, or a Cancel,
    /// and at the next launch if anything is left over.
    public static let folderName = ".SteerLab-update"
    /// Inside the staging folder: the process that owns it, so a second copy
    /// of the app never deletes an update the first is still preparing.
    public static let ownerFileName = "owner.pid"

    public static func stagingRoot(forApp app: URL) -> URL {
        app.deletingLastPathComponent().appending(component: folderName)
    }

    /// `SteerLab-0.9.9-previous.app`.
    public static func previousCopyName(for version: SemanticVersion) -> String {
        "SteerLab-\(version.releaseDescription)-previous.app"
    }

    /// The version a kept copy's name records, when the name is one this
    /// updater writes.
    public static func previousCopyVersion(fromName name: String) -> SemanticVersion? {
        let prefix = "SteerLab-"
        let suffix = "-previous.app"
        guard name.hasPrefix(prefix), name.hasSuffix(suffix),
            name.count > prefix.count + suffix.count
        else { return nil }
        let middle = name.dropFirst(prefix.count).dropLast(suffix.count)
        return SemanticVersion(String(middle))
    }
}

// MARK: - What can go wrong, in plain words

/// Every way an update can stop, each with what happened and what to do.
/// Whatever the case, nothing has been replaced unless the case says so.
public enum UpdateInstallError: Error, Sendable, Equatable {
    case noAppArchive(tag: String)
    case ambiguousAppArchive(names: [String])
    case missingChecksum(archive: String)
    case untrustedDownloadLocation(String)
    case notNewer(found: SemanticVersion, running: SemanticVersion)
    case archiveTooLarge(bytes: Int64, limit: Int64)
    case downloadFailed(String)
    case downloadTimedOut(minutes: Int)
    case cancelled
    case checksumUnreadable
    case checksumMismatch(expected: String, actual: String)
    case archiveRejected(String)
    case unpackFailed(String)
    case signatureRejected(String)
    case versionMismatch(String)
    case stagingUnavailable(String)
    case busy([String])
    /// The swap failed and the original app is back where it was.
    case installFailedRestored(String)
    /// The swap failed AND putting the original back failed. Says exactly
    /// where each copy is, so the person can finish by hand.
    case installFailedNotRestored(String)

    /// What happened, in one or two sentences.
    public var summary: String {
        switch self {
        case .noAppArchive(let tag):
            return "The release \(tag) has no SteerLab app file "
                + "(SteerLab-<version>+<revision>.zip), so there is nothing to install."
        case .ambiguousAppArchive(let names):
            return "The release has more than one SteerLab app file ("
                + names.joined(separator: ", ") + "), so SteerLab cannot tell which to install."
        case .missingChecksum(let archive):
            return "The release has no checksum file for \(archive) "
                + "(\(archive).sha256), so the download could not be checked."
        case .untrustedDownloadLocation(let place):
            return "The download would have come from \(place), which is not one of "
                + "GitHub's addresses, so it was not fetched."
        case .notNewer(let found, let running):
            return "The release is SteerLab \(found.releaseDescription), which is not newer "
                + "than the version you are running (\(running.releaseDescription))."
        case .archiveTooLarge(let bytes, let limit):
            return "The download is \(Self.describe(bytes)), more than the "
                + "\(Self.describe(limit)) an update may be, so it was stopped."
        case .downloadFailed(let reason):
            return "The download did not complete: \(reason)."
        case .downloadTimedOut(let minutes):
            return "The download took longer than \(minutes) minutes, so it was stopped."
        case .cancelled:
            return "The update was cancelled."
        case .checksumUnreadable:
            return "The release's checksum file could not be read."
        case .checksumMismatch(let expected, let actual):
            return "The downloaded file does not match its checksum (expected "
                + "\(expected.prefix(12))…, got \(actual.prefix(12))…). It may have been "
                + "damaged on the way."
        case .archiveRejected(let reason):
            return "The downloaded archive was not unpacked: \(reason)."
        case .unpackFailed(let reason):
            return "The downloaded archive could not be unpacked: \(reason)."
        case .signatureRejected(let reason):
            return "The downloaded app did not pass the signature check, so it was not "
                + "installed: \(reason)."
        case .versionMismatch(let reason):
            return "The downloaded app is not the version the release names: \(reason)."
        case .stagingUnavailable(let reason):
            return "SteerLab could not prepare the update beside the app: \(reason)."
        case .busy(let reasons):
            return "SteerLab is still working: " + reasons.joined(separator: "; ") + "."
        case .installFailedRestored(let reason):
            return "The new version could not be put in place (\(reason)). Your current "
                + "version is back where it was."
        case .installFailedNotRestored(let reason):
            return "The new version could not be put in place, and SteerLab could not put "
                + "everything back on its own. \(reason)"
        }
    }

    /// The step the person can take.
    public var repair: String {
        switch self {
        case .cancelled:
            return "Nothing was changed. Choose Install Update again when you are ready."
        case .busy:
            return "Wait for it to finish, or stop it, then choose Install Now."
        case .downloadFailed, .downloadTimedOut, .checksumMismatch:
            return "Nothing was changed. Try again later, or download the release from "
                + "the Releases page."
        case .installFailedNotRestored:
            return "Do not delete either copy. Rename the copy named in the message back "
                + "to its original name, or ask for help on the Releases page."
        case .stagingUnavailable:
            return "Nothing was changed. Download the release from the Releases page, "
                + "unzip it, and replace SteerLab.app with it."
        default:
            return "Nothing was changed. Download the release from the Releases page "
                + "instead."
        }
    }

    static func describe(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}

// MARK: - 1. Who gets it

/// The running app's own code signature, read offline.
public enum RunningAppSignature: Sendable, Equatable {
    /// Developer ID Application, from this team. `notarized` is true only
    /// when a notarization ticket stapled inside the app satisfies Apple's
    /// "notarized" requirement without the network.
    case developerID(teamID: String, notarized: Bool)
    /// Signed for this Mac only (`codesign -s -`, or the linker's default).
    case adHoc
    /// Signed with some other certificate, such as Apple Development.
    case otherCertificate(teamID: String?)
    case unsigned
    case unreadable(String)
}

/// Everything eligibility looks at, gathered by the live reader or handed
/// over whole by a test.
public struct UpdateInstallContext: Sendable, Equatable {
    /// The running `.app`, or nil when this process is not one (a SwiftPM
    /// executable started from a build folder).
    public var appBundle: URL?
    public var bundleIdentifier: String?
    public var signature: RunningAppSignature
    /// A SteerLab code checkout beside the app: a developer install.
    public var hasSourceCheckout: Bool
    /// macOS is running the app from a randomized read-only mount because it
    /// was opened straight from a quarantined download.
    public var isTranslocated: Bool
    public var isOnReadOnlyVolume: Bool
    public var canWriteParentFolder: Bool

    public init(
        appBundle: URL?, bundleIdentifier: String?, signature: RunningAppSignature,
        hasSourceCheckout: Bool, isTranslocated: Bool, isOnReadOnlyVolume: Bool,
        canWriteParentFolder: Bool
    ) {
        self.appBundle = appBundle
        self.bundleIdentifier = bundleIdentifier
        self.signature = signature
        self.hasSourceCheckout = hasSourceCheckout
        self.isTranslocated = isTranslocated
        self.isOnReadOnlyVolume = isOnReadOnlyVolume
        self.canWriteParentFolder = canWriteParentFolder
    }
}

/// Why this copy keeps the signpost instead of installing. Each says it in
/// one plain line and, where there is one, names the move that changes it.
public enum UpdateInstallRefusal: Error, Sendable, Equatable {
    case notAnAppBundle
    case developerInstall
    case adHocSignature
    case otherCertificate
    case unsigned
    case notNotarized
    case signatureUnreadable(String)
    case missingBundleIdentifier
    case translocated
    case readOnlyVolume
    case folderNotWritable(String)

    public var line: String {
        switch self {
        case .notAnAppBundle:
            return "This copy of SteerLab runs from a build folder, not the packaged app, "
                + "so it is updated by rebuilding."
        case .developerInstall:
            return "A SteerLab code checkout sits beside this app, so this is a developer "
                + "install: update it by pulling the code and rebuilding."
        case .adHocSignature:
            return "This copy is signed only for this Mac, so SteerLab cannot confirm that "
                + "a download comes from the same developer. Download the new release "
                + "from the Releases page."
        case .otherCertificate:
            return "This copy carries a development signature rather than a Developer ID "
                + "one, so it is updated by rebuilding or from the Releases page."
        case .unsigned:
            return "This copy is not signed, so SteerLab cannot confirm that a download "
                + "comes from the same developer. Download the new release from the "
                + "Releases page."
        case .notNotarized:
            return "This copy is not notarized by Apple, so SteerLab will not replace it "
                + "automatically. Download the new release from the Releases page."
        case .signatureUnreadable(let reason):
            return "SteerLab could not read this copy's own signature (\(reason)). "
                + "Download the new release from the Releases page."
        case .missingBundleIdentifier:
            return "This copy has no bundle identifier, so a download cannot be matched to "
                + "it. Download the new release from the Releases page."
        case .translocated:
            return "macOS is running this copy from a temporary, read-only place because "
                + "it was opened straight from the download. Move SteerLab.app to your "
                + "SteerLab folder or to Applications, open it from there, and check again."
        case .readOnlyVolume:
            return "This copy is on a disk SteerLab cannot write to. Move SteerLab.app to "
                + "your SteerLab folder or to Applications, open it from there, and check "
                + "again."
        case .folderNotWritable(let folder):
            return "SteerLab cannot write to the folder that holds it (\(folder)), so it "
                + "cannot put the new version there. Move SteerLab.app to your SteerLab "
                + "folder or to Applications, or download the new release from the "
                + "Releases page."
        }
    }
}

/// An app this updater may replace, and whose signature a download must match.
public struct UpdateInstallTarget: Sendable, Equatable {
    public let appBundle: URL
    public let bundleIdentifier: String
    /// Read from the RUNNING app's signature, never compiled in: a build
    /// signed by someone else can never replace this one, and a fork signed
    /// by its own team updates from its own releases.
    public let teamID: String

    public init(appBundle: URL, bundleIdentifier: String, teamID: String) {
        self.appBundle = appBundle
        self.bundleIdentifier = bundleIdentifier
        self.teamID = teamID
    }
}

public enum UpdateInstallEligibility {
    /// The first reason this copy may not install, in the order a person
    /// would want to hear them, or the target when there is none.
    public static func evaluate(_ context: UpdateInstallContext)
        -> Result<UpdateInstallTarget, UpdateInstallRefusal>
    {
        guard let app = context.appBundle, app.pathExtension == "app" else {
            return .failure(.notAnAppBundle)
        }
        if context.hasSourceCheckout { return .failure(.developerInstall) }
        let team: String
        switch context.signature {
        case .adHoc: return .failure(.adHocSignature)
        case .otherCertificate: return .failure(.otherCertificate)
        case .unsigned: return .failure(.unsigned)
        case .unreadable(let reason): return .failure(.signatureUnreadable(reason))
        case .developerID(let teamID, let notarized):
            guard notarized else { return .failure(.notNotarized) }
            team = teamID
        }
        guard let identifier = context.bundleIdentifier, !identifier.isEmpty else {
            return .failure(.missingBundleIdentifier)
        }
        if context.isTranslocated { return .failure(.translocated) }
        if context.isOnReadOnlyVolume { return .failure(.readOnlyVolume) }
        guard context.canWriteParentFolder else {
            return .failure(.folderNotWritable(app.deletingLastPathComponent().path))
        }
        return .success(
            UpdateInstallTarget(appBundle: app, bundleIdentifier: identifier, teamID: team))
    }
}

// MARK: - The signature a download must carry

public enum UpdateSignatureRequirement {
    /// A Developer ID team identifier is ten upper-case letters and digits.
    public static func isPlausibleTeamID(_ text: String) -> Bool {
        text.count == 10 && text.allSatisfy { $0.isASCII && ($0.isUppercase || $0.isNumber) }
    }

    /// Reverse-DNS characters only, so nothing can be smuggled into the
    /// requirement language.
    public static func isPlausibleBundleIdentifier(_ text: String) -> Bool {
        guard let first = text.first, first.isASCII, first.isLetter || first.isNumber,
            text.count <= 255
        else { return false }
        return text.allSatisfy { $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "." || $0 == "-") }
    }

    /// Apple's designated requirement for a Developer ID application, bound to
    /// the running app's own identifier and team, plus "notarized". Throws
    /// rather than quote anything that does not look like an identifier.
    public static func text(bundleIdentifier: String, teamID: String) throws -> String {
        guard isPlausibleBundleIdentifier(bundleIdentifier) else {
            throw UpdateInstallError.signatureRejected(
                "this copy's bundle identifier cannot be used in a signature requirement")
        }
        guard isPlausibleTeamID(teamID) else {
            throw UpdateInstallError.signatureRejected(
                "this copy's Developer ID team cannot be used in a signature requirement")
        }
        return "identifier \"\(bundleIdentifier)\" and anchor apple generic"
            // Developer ID certificate authority (intermediate) ...
            + " and certificate 1[field.1.2.840.113635.100.6.2.6] exists"
            // ... and a Developer ID Application leaf ...
            + " and certificate leaf[field.1.2.840.113635.100.6.1.13] exists"
            // ... issued to the running app's own team ...
            + " and certificate leaf[subject.OU] = \"\(teamID)\""
            // ... with Apple's notarization ticket.
            + " and notarized"
    }
}

/// The code-signature seam.
public protocol UpdateCodeSigning: Sendable {
    /// The running app's own signature, read offline.
    func runningAppSignature() -> RunningAppSignature
    /// Strict validation of `bundle` and all the code nested in it against
    /// `requirement`, offline. Throws `UpdateInstallError.signatureRejected`.
    func validate(bundle: URL, requirement: String) throws
}

// MARK: - 2. Which files to download

/// The app archive and its checksum, chosen from one release.
public struct UpdateAssetPair: Sendable, Equatable {
    public let archive: ReleaseAsset
    public let checksum: ReleaseAsset
    public let version: SemanticVersion
    public let revision: String
}

public enum UpdateReleaseAssets {
    /// `SteerLab-<version>+<revision>.zip`, or the same name with the `+`
    /// spelled `.` or `-` (GitHub rewrites some characters of uploaded file
    /// names). Returns the version and revision, or nil for any other name.
    public static func parseArchiveName(_ name: String) -> (SemanticVersion, String)? {
        let prefix = "SteerLab-"
        let suffix = ".zip"
        guard name.hasPrefix(prefix), name.hasSuffix(suffix) else { return nil }
        let middle = String(name.dropFirst(prefix.count).dropLast(suffix.count))
        for mark: Character in ["+", ".", "-"] {
            guard let separator = middle.lastIndex(of: mark) else { continue }
            let versionText = String(middle[..<separator])
            let revision = String(middle[middle.index(after: separator)...])
            guard (7...40).contains(revision.count), revision.allSatisfy(\.isHexDigit),
                !versionText.contains("+"), let version = SemanticVersion(versionText)
            else { continue }
            return (version, revision.lowercased())
        }
        return nil
    }

    /// The one app archive whose version is the release tag's version, and
    /// the `.sha256` beside it. Archives for other versions, and names that
    /// are not app archives, are ignored; two candidates are refused.
    public static func select(tag: String, assets: [ReleaseAsset])
        -> Result<UpdateAssetPair, UpdateInstallError>
    {
        guard let tagVersion = SemanticVersion(tag) else {
            return .failure(.noAppArchive(tag: tag))
        }
        let archives = assets.compactMap { asset -> (ReleaseAsset, SemanticVersion, String)? in
            guard let (version, revision) = parseArchiveName(asset.name),
                version == tagVersion
            else { return nil }
            return (asset, version, revision)
        }
        guard let first = archives.first else { return .failure(.noAppArchive(tag: tag)) }
        guard archives.count == 1 else {
            return .failure(.ambiguousAppArchive(names: archives.map(\.0.name).sorted()))
        }
        let (archive, version, revision) = first
        guard let checksum = assets.first(where: { $0.name == archive.name + ".sha256" }) else {
            return .failure(.missingChecksum(archive: archive.name))
        }
        for url in [archive.downloadURL, checksum.downloadURL]
        where !isTrustedAssetURL(url) {
            return .failure(.untrustedDownloadLocation(url.host() ?? url.absoluteString))
        }
        return .success(
            UpdateAssetPair(
                archive: archive, checksum: checksum, version: version, revision: revision))
    }

    /// GitHub's own hosts, over HTTPS, with no credentials in the address.
    /// A release download starts on github.com and is redirected to a
    /// githubusercontent.com storage host; a renamed repository's API
    /// answers are redirected within api.github.com.
    public static func isTrustedHost(_ url: URL) -> Bool {
        guard url.scheme?.lowercased() == "https", url.user() == nil,
            url.password() == nil, url.port == nil || url.port == 443,
            let host = url.host()?.lowercased()
        else { return false }
        return host == "github.com" || host == "api.github.com"
            || host.hasSuffix(".githubusercontent.com")
    }

    /// A release file's starting address: a trusted host and the shape of a
    /// release download (`/<owner>/<repository>/releases/download/<tag>/<file>`).
    /// The owner and repository are deliberately not pinned, so a renamed
    /// repository's releases still install; the signature check is what binds
    /// the download to this app's developer.
    public static func isTrustedAssetURL(_ url: URL) -> Bool {
        guard isTrustedHost(url) else { return false }
        let parts = url.path().split(separator: "/")
        return parts.count >= 6 && parts[2] == "releases" && parts[3] == "download"
    }
}

// MARK: - 3. Verify: the checksum

public enum UpdateChecksum {
    /// The digest a `shasum -a 256` line records: its first field, when that
    /// is 64 hexadecimal characters. The file name after it is ignored,
    /// because GitHub may have rewritten the archive's name.
    public static func expectedDigest(fromChecksumFile text: String) -> String? {
        guard let first = text.split(whereSeparator: { $0 == " " || $0 == "\t" || $0 == "\n" || $0 == "\r" }).first
        else { return nil }
        let digest = String(first).lowercased()
        guard digest.count == 64, digest.allSatisfy(\.isHexDigit) else { return nil }
        return digest
    }

    /// SHA-256 of a file, read in chunks.
    public static func sha256Hex(ofFileAt url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - 3. Verify: what the archive would write

/// Reads `zipinfo`'s listing of the archive BEFORE anything is unpacked, and
/// refuses an archive that would write outside one `.app` folder: absolute
/// names, `..`, an entry under a symbolic link, more than one app, or more
/// files or bytes than an update may hold.
public enum UpdateArchiveListing {
    public struct Summary: Sendable, Equatable {
        /// The archive's one top-level app folder, such as `SteerLab.app`.
        public let appName: String
        public let entries: Int
        public let unpackedBytes: Int64
    }

    struct Entry {
        let isSymlink: Bool
        let size: Int64
        let path: String
    }

    /// One `zipinfo` entry line: permissions, version, system, size, flags,
    /// method, date, time, then the name (which may contain spaces).
    static func parseEntry(_ line: String) -> Entry? {
        let fields = line.split(separator: " ", maxSplits: 8, omittingEmptySubsequences: true)
        guard fields.count == 9, let mode = fields.first, mode.count == 10,
            let kind = mode.first, "-dl".contains(kind),
            let size = Int64(fields[3])
        else { return nil }
        return Entry(isSymlink: kind == "l", size: size, path: String(fields[8]))
    }

    public static func validate(
        zipinfoOutput lines: [String], limits: UpdateInstallLimits = .standard
    ) -> Result<Summary, UpdateInstallError> {
        var entries: [Entry] = []
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("Archive:")
                || trimmed.hasPrefix("Zip file size:")
                || trimmed.range(of: #"^\d+ files?, "#, options: .regularExpression) != nil
            {
                continue
            }
            guard let entry = parseEntry(trimmed) else {
                return .failure(.archiveRejected("its list of files could not be read"))
            }
            entries.append(entry)
            if entries.count > limits.maximumArchiveEntries {
                return .failure(
                    .archiveRejected(
                        "it holds more than \(limits.maximumArchiveEntries) files"))
            }
        }
        guard !entries.isEmpty else { return .failure(.archiveRejected("it is empty")) }

        func components(_ path: String) -> [Substring] {
            path.split(separator: "/", omittingEmptySubsequences: true)
        }
        var symlinks: Set<String> = []
        var apps: Set<String> = []
        var total: Int64 = 0
        for entry in entries {
            if entry.path.hasPrefix("/") {
                return .failure(.archiveRejected("it names a file by an absolute path"))
            }
            let parts = components(entry.path)
            guard let top = parts.first else {
                return .failure(.archiveRejected("it holds a file with no name"))
            }
            if parts.contains(where: { $0 == ".." || $0 == "." }) {
                return .failure(.archiveRejected("it names a file outside its own folder"))
            }
            let topName = String(top)
            if topName.hasSuffix(".app"), !topName.hasPrefix("._") {
                apps.insert(topName)
            } else if topName == "__MACOSX" || (topName.hasPrefix("._") && parts.count == 1) {
                // Extended-attribute companions ditto writes; restored, never run.
            } else {
                return .failure(.archiveRejected("it holds \(topName) beside the app"))
            }
            if entry.isSymlink { symlinks.insert(parts.joined(separator: "/")) }
            total += max(0, entry.size)
            if total > limits.maximumUnpackedBytes {
                return .failure(
                    .archiveRejected(
                        "it would unpack to more than "
                            + UpdateInstallError.describe(limits.maximumUnpackedBytes)))
            }
        }
        guard apps.count == 1, let app = apps.first else {
            return .failure(
                .archiveRejected(
                    apps.isEmpty ? "it holds no app" : "it holds more than one app"))
        }
        // An entry beneath a symbolic link would be written wherever the link
        // points — the one way an ordinary-looking archive escapes its folder.
        for entry in entries {
            let parts = components(entry.path)
            guard parts.count > 1 else { continue }
            for depth in 1..<parts.count
            where symlinks.contains(parts[0..<depth].joined(separator: "/")) {
                return .failure(.archiveRejected("it writes a file through a symbolic link"))
            }
        }
        return .success(Summary(appName: app, entries: entries.count, unpackedBytes: total))
    }
}

// MARK: - The network seam, and the bounded download over it

/// One thing a transfer reports: the response (status and declared length)
/// first, then the body in chunks.
public enum UpdateTransferEvent: Sendable, Equatable {
    case response(status: Int, expectedLength: Int64?)
    case data(Data)
}

/// The network seam. The live transport refuses any redirect off GitHub's
/// hosts and sends no credentials; a test hands back a scripted stream.
public protocol UpdateTransport: Sendable {
    func open(_ url: URL) throws -> AsyncThrowingStream<UpdateTransferEvent, Error>
}

/// The clock seam: the download's deadline is measured and awaited through
/// it, so a timeout is tested without waiting.
public protocol UpdateClock: Sendable {
    func now() -> Date
    /// Returns at `deadline`; throws when the task is cancelled first.
    func sleep(until deadline: Date) async throws
}

public struct SystemUpdateClock: UpdateClock {
    public init() {}
    public func now() -> Date { Date() }
    public func sleep(until deadline: Date) async throws {
        let seconds = max(0, deadline.timeIntervalSinceNow)
        try await Task.sleep(for: .seconds(seconds))
    }
}

/// How far the update has got, for the sheet.
public enum UpdateInstallProgress: Sendable, Equatable {
    case downloading(received: Int64, total: Int64?)
    case verifyingChecksum
    case unpacking
    case checkingSignature
}

public enum UpdateDownload {
    /// Streams `url` into a new file at `destination`, refusing an untrusted
    /// address, a non-200 answer, more than `byteLimit` bytes (declared or
    /// actual), a body shorter than it declared, and anything still running
    /// at `deadline`. Cancelling the task stops it. Returns the bytes written.
    public static func receive(
        _ url: URL, into destination: URL, byteLimit: Int64, deadline: Date,
        transport: any UpdateTransport, clock: any UpdateClock,
        progress: (@Sendable (Int64, Int64?) -> Void)? = nil
    ) async throws -> Int64 {
        guard UpdateReleaseAssets.isTrustedHost(url) else {
            throw UpdateInstallError.untrustedDownloadLocation(url.host() ?? url.absoluteString)
        }
        let minutes = max(1, Int((deadline.timeIntervalSince(clock.now()) / 60).rounded()))
        do {
            return try await withThrowingTaskGroup(of: Int64?.self) { group in
                group.addTask {
                    try await consume(
                        url, into: destination, byteLimit: byteLimit, deadline: deadline,
                        minutes: minutes, transport: transport, clock: clock,
                        progress: progress)
                }
                group.addTask {
                    try await clock.sleep(until: deadline)
                    return nil
                }
                defer { group.cancelAll() }
                guard let first = try await group.next(), let total = first else {
                    throw UpdateInstallError.downloadTimedOut(minutes: minutes)
                }
                return total
            }
        } catch let error as UpdateInstallError {
            throw error
        } catch is CancellationError {
            throw UpdateInstallError.cancelled
        } catch let error as URLError where error.code == .cancelled {
            throw UpdateInstallError.cancelled
        } catch let error as URLError where error.code == .timedOut {
            throw UpdateInstallError.downloadTimedOut(minutes: minutes)
        } catch {
            throw UpdateInstallError.downloadFailed(error.localizedDescription)
        }
    }

    private static func consume(
        _ url: URL, into destination: URL, byteLimit: Int64, deadline: Date, minutes: Int,
        transport: any UpdateTransport, clock: any UpdateClock,
        progress: (@Sendable (Int64, Int64?) -> Void)?
    ) async throws -> Int64 {
        let fm = FileManager.default
        if fm.fileExists(atPath: destination.path) { try fm.removeItem(at: destination) }
        guard fm.createFile(atPath: destination.path, contents: nil) else {
            throw UpdateInstallError.stagingUnavailable(
                "could not create \(destination.lastPathComponent)")
        }
        let handle = try FileHandle(forWritingTo: destination)
        defer { try? handle.close() }
        var total: Int64 = 0
        var expected: Int64?
        var answered = false
        for try await event in try transport.open(url) {
            try Task.checkCancellation()
            switch event {
            case .response(let status, let length):
                guard status == 200 else {
                    throw UpdateInstallError.downloadFailed("GitHub answered HTTP \(status)")
                }
                if let length, length > byteLimit {
                    throw UpdateInstallError.archiveTooLarge(bytes: length, limit: byteLimit)
                }
                expected = length
                answered = true
            case .data(let chunk):
                guard answered else {
                    throw UpdateInstallError.downloadFailed("the answer came without a status")
                }
                total += Int64(chunk.count)
                if total > byteLimit {
                    throw UpdateInstallError.archiveTooLarge(bytes: total, limit: byteLimit)
                }
                try handle.write(contentsOf: chunk)
                progress?(total, expected)
                if clock.now() > deadline {
                    throw UpdateInstallError.downloadTimedOut(minutes: minutes)
                }
            }
        }
        try Task.checkCancellation()
        guard answered else { throw UpdateInstallError.downloadFailed("there was no answer") }
        if let expected, total != expected {
            throw UpdateInstallError.downloadFailed(
                "it ended after \(total) of \(expected) bytes")
        }
        return total
    }
}

// MARK: - The file-system seam

/// What a bundle's Info.plist says about it.
public struct UpdateBundleInfo: Sendable, Equatable {
    public let identifier: String?
    /// `SLFullVersionString`, else `CFBundleShortVersionString`.
    public let version: String?

    public init(identifier: String?, version: String?) {
        self.identifier = identifier
        self.version = version
    }
}

/// Every file operation the updater performs. The live one is `FileManager`;
/// tests wrap it to fail chosen steps.
public protocol UpdateFileSystem: Sendable {
    func exists(_ url: URL) -> Bool
    func isDirectory(_ url: URL) -> Bool
    func createDirectory(_ url: URL) throws
    /// A rename when both are on one volume, which they always are here.
    func move(_ source: URL, to destination: URL) throws
    func remove(_ url: URL) throws
    func trash(_ url: URL) throws
    func contents(of directory: URL) throws -> [URL]
    func write(_ data: Data, to url: URL) throws
    func read(_ url: URL) -> Data?
    func bundleInfo(at app: URL) -> UpdateBundleInfo
}

public struct LiveUpdateFileSystem: UpdateFileSystem {
    public init() {}

    public func exists(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.path)
    }

    public func isDirectory(_ url: URL) -> Bool {
        var isDirectory: ObjCBool = false
        return FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory)
            && isDirectory.boolValue
    }

    public func createDirectory(_ url: URL) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    }

    public func move(_ source: URL, to destination: URL) throws {
        try FileManager.default.moveItem(at: source, to: destination)
    }

    public func remove(_ url: URL) throws {
        try FileManager.default.removeItem(at: url)
    }

    public func trash(_ url: URL) throws {
        try FileManager.default.trashItem(at: url, resultingItemURL: nil)
    }

    public func contents(of directory: URL) throws -> [URL] {
        try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil)
    }

    public func write(_ data: Data, to url: URL) throws {
        try data.write(to: url, options: .atomic)
    }

    public func read(_ url: URL) -> Data? {
        try? Data(contentsOf: url)
    }

    public func bundleInfo(at app: URL) -> UpdateBundleInfo {
        let plist = app.appending(components: "Contents", "Info.plist")
        guard let data = try? Data(contentsOf: plist),
            let dictionary = try? PropertyListSerialization.propertyList(
                from: data, format: nil) as? [String: Any]
        else { return UpdateBundleInfo(identifier: nil, version: nil) }
        let full = (dictionary["SLFullVersionString"] as? String)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let short = dictionary["CFBundleShortVersionString"] as? String
        return UpdateBundleInfo(
            identifier: dictionary["CFBundleIdentifier"] as? String,
            version: (full?.isEmpty == false ? full : nil) ?? short)
    }
}

// MARK: - The installer

/// A downloaded, verified app, staged beside the installed one and ready to
/// be swapped in.
public struct UpdateStagedApp: Sendable, Equatable {
    public let stagedApp: URL
    public let version: SemanticVersion
    public let stagingRoot: URL
}

/// What an install did.
public struct UpdateInstallReceipt: Sendable, Equatable {
    public let installedVersion: SemanticVersion
    /// Where the version that was running is kept.
    public let previousCopy: URL
    /// Older kept copies this install moved to the Trash.
    public let trashedCopies: [URL]
}

/// The seams one installer runs over.
public struct UpdateInstallEnvironment: Sendable {
    public var transport: any UpdateTransport
    public var codeSigning: any UpdateCodeSigning
    /// Runs `/usr/bin/zipinfo` and `/usr/bin/ditto`.
    public var tools: any ProvisionCommandRunner
    public var fileSystem: any UpdateFileSystem
    public var clock: any UpdateClock
    public var processID: Int32
    public var isProcessAlive: @Sendable (Int32) -> Bool
    public var limits: UpdateInstallLimits

    public init(
        transport: any UpdateTransport, codeSigning: any UpdateCodeSigning,
        tools: any ProvisionCommandRunner = SystemProvisionRunner(),
        fileSystem: any UpdateFileSystem = LiveUpdateFileSystem(),
        clock: any UpdateClock = SystemUpdateClock(),
        processID: Int32 = ProcessInfo.processInfo.processIdentifier,
        isProcessAlive: @escaping @Sendable (Int32) -> Bool = UpdateStaging.isProcessAlive,
        limits: UpdateInstallLimits = .standard
    ) {
        self.transport = transport
        self.codeSigning = codeSigning
        self.tools = tools
        self.fileSystem = fileSystem
        self.clock = clock
        self.processID = processID
        self.isProcessAlive = isProcessAlive
        self.limits = limits
    }
}

/// Owning, clearing, and cleaning the staging folder.
public enum UpdateStaging {
    public static func isProcessAlive(_ pid: Int32) -> Bool {
        guard pid > 0 else { return false }
        return kill(pid, 0) == 0 || errno == EPERM
    }

    /// The process recorded as preparing an update in `root`, if any.
    static func owner(of root: URL, fileSystem: any UpdateFileSystem) -> Int32? {
        guard let data = fileSystem.read(root.appending(component: UpdateStagingLayout.ownerFileName)),
            let text = String(data: data, encoding: .utf8)
        else { return nil }
        return Int32(text.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    /// Next-launch cleanup: removes what an update left beside the app,
    /// unless a live SteerLab process still owns it. Returns true when
    /// something was removed.
    @discardableResult
    public static func removeLeftovers(
        besideApp app: URL, fileSystem: any UpdateFileSystem = LiveUpdateFileSystem(),
        currentProcess: Int32 = ProcessInfo.processInfo.processIdentifier,
        isProcessAlive: @Sendable (Int32) -> Bool = UpdateStaging.isProcessAlive
    ) -> Bool {
        let root = UpdateStagingLayout.stagingRoot(forApp: app)
        guard fileSystem.exists(root) else { return false }
        if let owner = owner(of: root, fileSystem: fileSystem), owner != currentProcess,
            isProcessAlive(owner)
        {
            return false
        }
        return (try? fileSystem.remove(root)) != nil
    }
}

public struct UpdateInstaller: Sendable {
    public let target: UpdateInstallTarget
    public let currentVersion: SemanticVersion
    public let environment: UpdateInstallEnvironment

    public init(
        target: UpdateInstallTarget, currentVersion: SemanticVersion,
        environment: UpdateInstallEnvironment
    ) {
        self.target = target
        self.currentVersion = currentVersion
        self.environment = environment
    }

    private var fs: any UpdateFileSystem { environment.fileSystem }
    public var stagingRoot: URL { UpdateStagingLayout.stagingRoot(forApp: target.appBundle) }
    public var previousCopy: URL {
        target.appBundle.deletingLastPathComponent()
            .appending(component: UpdateStagingLayout.previousCopyName(for: currentVersion))
    }

    // MARK: Download and verify

    /// Download, check, unpack, and verify `release` into the staging folder.
    /// Nothing outside the staging folder is touched, and the staging folder
    /// is removed again on any failure or Cancel.
    public func prepare(
        _ release: PublishedRelease,
        progress: @escaping @Sendable (UpdateInstallProgress) -> Void = { _ in }
    ) async throws -> UpdateStagedApp {
        let pair = try UpdateReleaseAssets.select(tag: release.tag, assets: release.assets).get()
        // Version gating before a byte is fetched (and again on the app).
        guard pair.version > currentVersion else {
            throw UpdateInstallError.notNewer(found: pair.version, running: currentVersion)
        }
        if let size = pair.archive.size, size > environment.limits.maximumArchiveBytes {
            throw UpdateInstallError.archiveTooLarge(
                bytes: size, limit: environment.limits.maximumArchiveBytes)
        }
        try openStaging()
        do {
            let staged = try await downloadAndVerify(pair, progress: progress)
            return staged
        } catch {
            discardStaging()
            if error is CancellationError { throw UpdateInstallError.cancelled }
            if let typed = error as? UpdateInstallError { throw typed }
            throw UpdateInstallError.downloadFailed(error.localizedDescription)
        }
    }

    private func openStaging() throws {
        let root = stagingRoot
        if fs.exists(root) {
            if let owner = UpdateStaging.owner(of: root, fileSystem: fs),
                owner != environment.processID, environment.isProcessAlive(owner)
            {
                throw UpdateInstallError.stagingUnavailable(
                    "another copy of SteerLab is preparing an update")
            }
            do { try fs.remove(root) } catch {
                throw UpdateInstallError.stagingUnavailable(
                    "an earlier update's files at \(root.path) could not be removed")
            }
        }
        do {
            try fs.createDirectory(root)
            try fs.write(
                Data("\(environment.processID)\n".utf8),
                to: root.appending(component: UpdateStagingLayout.ownerFileName))
        } catch {
            throw UpdateInstallError.stagingUnavailable(
                "could not create \(root.path): \(error.localizedDescription)")
        }
    }

    /// Remove the staging folder, if this process owns it.
    public func discardStaging() {
        let root = stagingRoot
        guard fs.exists(root) else { return }
        if let owner = UpdateStaging.owner(of: root, fileSystem: fs),
            owner != environment.processID
        {
            return
        }
        try? fs.remove(root)
    }

    private func downloadAndVerify(
        _ pair: UpdateAssetPair, progress: @escaping @Sendable (UpdateInstallProgress) -> Void
    ) async throws -> UpdateStagedApp {
        let root = stagingRoot
        let limits = environment.limits
        let deadline = environment.clock.now().addingTimeInterval(limits.downloadDeadline)

        // The checksum first: it is tiny, and without it nothing is checked.
        let checksumFile = root.appending(component: "download.sha256")
        _ = try await UpdateDownload.receive(
            pair.checksum.downloadURL, into: checksumFile,
            byteLimit: limits.maximumChecksumBytes, deadline: deadline,
            transport: environment.transport, clock: environment.clock)
        guard let checksumData = fs.read(checksumFile),
            let expected = UpdateChecksum.expectedDigest(
                fromChecksumFile: String(decoding: checksumData, as: UTF8.self))
        else { throw UpdateInstallError.checksumUnreadable }

        let archiveFile = root.appending(component: "download.zip")
        let limit = min(limits.maximumArchiveBytes, pair.archive.size ?? limits.maximumArchiveBytes)
        progress(.downloading(received: 0, total: pair.archive.size))
        _ = try await UpdateDownload.receive(
            pair.archive.downloadURL, into: archiveFile, byteLimit: limit, deadline: deadline,
            transport: environment.transport, clock: environment.clock,
            progress: { received, total in
                progress(.downloading(received: received, total: total ?? pair.archive.size))
            })
        try Task.checkCancellation()

        // 1. Integrity.
        progress(.verifyingChecksum)
        let actual: String
        do { actual = try UpdateChecksum.sha256Hex(ofFileAt: archiveFile) } catch {
            throw UpdateInstallError.downloadFailed("the downloaded file could not be read")
        }
        guard actual == expected else {
            throw UpdateInstallError.checksumMismatch(expected: expected, actual: actual)
        }
        try Task.checkCancellation()

        // 2. Unpack — after reading what the archive would write.
        progress(.unpacking)
        let listing = try await runTool(
            ["/usr/bin/zipinfo", archiveFile.path], failure: { code in
                .archiveRejected("its list of files could not be read (zipinfo exited \(code))")
            })
        let summary = try UpdateArchiveListing.validate(zipinfoOutput: listing, limits: limits)
            .get()
        let unpacked = root.appending(component: "unpacked")
        _ = try await runTool(
            ["/usr/bin/ditto", "-x", "-k", archiveFile.path, unpacked.path],
            failure: { code in .unpackFailed("ditto exited \(code)") })
        try? fs.remove(archiveFile)
        let app = unpacked.appending(component: summary.appName)
        guard fs.isDirectory(app) else {
            throw UpdateInstallError.unpackFailed("\(summary.appName) was not in the archive")
        }
        try Task.checkCancellation()

        // 3. The signature: this app's own identifier and team, notarized.
        progress(.checkingSignature)
        let requirement = try UpdateSignatureRequirement.text(
            bundleIdentifier: target.bundleIdentifier, teamID: target.teamID)
        try environment.codeSigning.validate(bundle: app, requirement: requirement)

        // 4. The version, read from the now-verified bundle.
        let info = fs.bundleInfo(at: app)
        guard info.identifier == target.bundleIdentifier else {
            throw UpdateInstallError.versionMismatch(
                "it identifies itself as \(info.identifier ?? "nothing")")
        }
        guard let raw = info.version, let version = SemanticVersion(raw) else {
            throw UpdateInstallError.versionMismatch("it does not say which version it is")
        }
        guard version == pair.version else {
            throw UpdateInstallError.versionMismatch(
                "it is \(version.releaseDescription), but the release file names "
                    + pair.version.releaseDescription)
        }
        guard version > currentVersion else {
            throw UpdateInstallError.notNewer(found: version, running: currentVersion)
        }
        return UpdateStagedApp(stagedApp: app, version: version, stagingRoot: root)
    }

    private func runTool(
        _ argv: [String], failure: (Int32) -> UpdateInstallError
    ) async throws -> [String] {
        let lines = ProvisionLineBuffer()
        let code: Int32
        do {
            code = try await environment.tools.run(argv) { lines.append($0) }
        } catch is CancellationError {
            throw UpdateInstallError.cancelled
        } catch {
            throw failure(-1)
        }
        try Task.checkCancellation()
        guard code == 0 else { throw failure(code) }
        return lines.snapshot()
    }

    // MARK: Install

    /// Swap the staged app in: keep the running app beside it as the
    /// previous copy, move the new app into the original path, and check it
    /// is there. Any failure undoes what was done, in reverse.
    public func install(_ staged: UpdateStagedApp) throws -> UpdateInstallReceipt {
        let original = target.appBundle
        let previous = previousCopy
        let parked = staged.stagingRoot.appending(component: "replaced-previous.app")
        guard fs.isDirectory(staged.stagedApp) else {
            throw UpdateInstallError.installFailedRestored(
                "the downloaded app is no longer in \(staged.stagingRoot.path)")
        }
        guard fs.isDirectory(original) else {
            throw UpdateInstallError.installFailedRestored(
                "the running app is no longer at \(original.path)")
        }

        enum Done { case parkedPrevious, keptRunning, placedNew }
        var done: [Done] = []

        func rollback(_ reason: String) -> UpdateInstallError {
            var problems: [String] = []
            let fileSystem = fs
            for step in done.reversed() {
                let undone: Bool
                switch step {
                case .placedNew:
                    undone = (try? fileSystem.move(original, to: staged.stagedApp)) != nil
                case .keptRunning:
                    undone = (try? fileSystem.move(previous, to: original)) != nil
                case .parkedPrevious:
                    undone = (try? fileSystem.move(parked, to: previous)) != nil
                }
                if !undone {
                    switch step {
                    case .placedNew:
                        problems.append("the new version is still at \(original.path)")
                    case .keptRunning:
                        problems.append(
                            "your current version is at \(previous.path); rename it to "
                                + original.lastPathComponent)
                    case .parkedPrevious:
                        problems.append(
                            "the earlier kept copy is at \(parked.path); move it back to "
                                + previous.path)
                    }
                }
            }
            guard problems.isEmpty else {
                return .installFailedNotRestored(
                    "It stopped because \(reason). " + problems.joined(separator: "; ")
                        + ".")
            }
            discardStaging()
            return .installFailedRestored(reason)
        }

        if fs.exists(previous) {
            do { try fs.move(previous, to: parked); done.append(.parkedPrevious) } catch {
                throw rollback("the earlier kept copy at \(previous.path) could not be moved aside")
            }
        }
        do { try fs.move(original, to: previous); done.append(.keptRunning) } catch {
            throw rollback("the running app could not be renamed to \(previous.lastPathComponent)")
        }
        do { try fs.move(staged.stagedApp, to: original); done.append(.placedNew) } catch {
            throw rollback("the new version could not be moved to \(original.path)")
        }
        let info = fs.bundleInfo(at: original)
        guard info.identifier == target.bundleIdentifier,
            let raw = info.version, SemanticVersion(raw) == staged.version
        else {
            throw rollback("the app now at \(original.path) is not the version just verified")
        }

        // Committed. Everything after this is tidying, and never undoes it.
        // A kept copy that already had this name (the same version installed
        // twice) goes to the Trash, recoverable, like the older ones.
        var trashed = trashOlderPreviousCopies(keeping: previous)
        if done.contains(.parkedPrevious), (try? fs.trash(parked)) != nil {
            trashed.insert(previous, at: 0)
        }
        discardStaging()
        return UpdateInstallReceipt(
            installedVersion: staged.version, previousCopy: previous, trashedCopies: trashed)
    }

    /// Keep one previous copy: move older ones this updater named, of this
    /// same app, to the Trash (recoverable). Anything else is left alone.
    private func trashOlderPreviousCopies(keeping kept: URL) -> [URL] {
        let folder = target.appBundle.deletingLastPathComponent()
        guard let entries = try? fs.contents(of: folder) else { return [] }
        var trashed: [URL] = []
        for entry in entries where entry.lastPathComponent != kept.lastPathComponent {
            guard let version = UpdateStagingLayout.previousCopyVersion(
                fromName: entry.lastPathComponent),
                version < currentVersion,
                fs.bundleInfo(at: entry).identifier == target.bundleIdentifier
            else { continue }
            if (try? fs.trash(entry)) != nil { trashed.append(entry) }
        }
        return trashed
    }
}

// MARK: - 4. Reopening the new app

public enum UpdateRelaunch {
    /// A detached `/bin/sh` that waits (up to five minutes) for `pid` to
    /// exit and then opens `app`. The process ID and the path are positional
    /// arguments, never spliced into the script.
    public static func waiterArguments(pid: Int32, app: URL) -> [String] {
        let script = """
            trap '' HUP
            pid="$1"; app="$2"; n=0
            while /bin/kill -0 "$pid" 2>/dev/null; do
              n=$((n + 1)); [ "$n" -gt 600 ] && exit 0
              /bin/sleep 0.5
            done
            exec /usr/bin/open "$app"
            """
        return ["/bin/sh", "-c", script, "steerlab-relaunch", "\(pid)", app.path]
    }
}

// MARK: - Waiting until SteerLab is idle

/// What is still running, in a researcher's words. The update does not
/// proceed while any of it is.
public struct UpdateBusyState: Sendable, Equatable {
    public var reasons: [String]

    public init(reasons: [String] = []) { self.reasons = reasons }

    public var isIdle: Bool { reasons.isEmpty }

    /// Builds the list from what the app can see.
    public static func from(
        studyRunning: Bool = false, validating: Bool = false, extracting: Bool = false,
        sweeping: Bool = false, evaluating: Bool = false, generating: Bool = false,
        submissions: Int = 0, engineSettingUp: Bool = false,
        engineStartingOrStopping: Bool = false, engineJobs: Int = 0,
        commandLineProcesses: Int = 0
    ) -> UpdateBusyState {
        var reasons: [String] = []
        if studyRunning { reasons.append("a study is running on this Mac") }
        if validating { reasons.append("a study is being validated on this Mac") }
        if extracting { reasons.append("vectors are being extracted on this Mac") }
        if sweeping { reasons.append("a sweep is running on this Mac") }
        if evaluating { reasons.append("responses are being judged on this Mac") }
        if generating { reasons.append("the Playground is generating a response") }
        if submissions > 0 { reasons.append("a study is being submitted to run") }
        if engineSettingUp { reasons.append("the local Python engine is being set up") }
        if engineStartingOrStopping {
            reasons.append("the local Python engine is starting or stopping")
        }
        if engineJobs > 0 {
            reasons.append(
                "the local Python engine is running \(engineJobs) job"
                    + (engineJobs == 1 ? "" : "s"))
        }
        if commandLineProcesses > 0 {
            reasons.append(
                "this app's command line (steerlab-cli) is running"
                    + (commandLineProcesses == 1 ? "" : " in \(commandLineProcesses) places")
                    + ", perhaps for your coding assistant")
        }
        return UpdateBusyState(reasons: reasons)
    }
}

/// Work in flight that has no other place the updater can see it: today, a
/// study submission (`UnifiedStudyRunner`). Counted, never inspected.
@MainActor
@Observable
public final class UpdateBusyWork {
    public static let shared = UpdateBusyWork()
    public private(set) var submissions = 0

    public init() {}

    public func beginSubmission() { submissions += 1 }
    public func endSubmission() { submissions = max(0, submissions - 1) }
}

// MARK: - The first launch after an update

/// What the first launch of a newer version says, once.
public struct UpdateAfterInstallNotice: Sendable, Equatable {
    public let headline: String
    public let lines: [String]
    /// The local engine was copied from the earlier version; the notice
    /// offers to open the engine setup, where Refresh Engine brings it up
    /// to date.
    public let offersEngineSetup: Bool
}

public enum UpdateAfterInstall {
    /// The version the app last launched as, on this Mac.
    public static let lastLaunchedVersionKey = "UpdateLastLaunchedVersion"

    /// Shared with the notice and the confirmation sheet so the two never
    /// disagree. Worded from the cluster status's own advisory ("server-side
    /// changes since that push are NOT running; push a fresh payload").
    public static let clusterLine =
        "If you run studies on a cluster, push the engine there again. The cluster "
        + "keeps the engine you pushed from the earlier version: until you push this "
        + "version's engine (the cluster setup's Push step, or steerlab-cli cluster push "
        + "--site <site>, then restart its controller), changes on the engine side are "
        + "not running there, and submissions say so."

    public static func goingBackLine(previousCopyName name: String, appName: String) -> String {
        "The earlier version is kept beside it as \(name). To go back, quit SteerLab, "
            + "move \(appName) to the Trash, rename \(name) to \(appName), and open it."
    }

    public static let engineLine =
        "The Python engine on this Mac still holds the earlier version's code. Open the "
        + "engine setup and choose Refresh Engine: it copies this version's engine and "
        + "keeps the packages already installed."

    /// - Parameters:
    ///   - lastLaunched: the recorded version of the previous launch (nil on
    ///     a first launch, or after a version that did not record one).
    ///   - previousCopyName: the kept copy beside the app, when one exists.
    ///   - engineIsStale: the materialized engine's stamp names a different
    ///     build from this app's.
    public static func notice(
        lastLaunched: String?, current: SemanticVersion, appName: String,
        previousCopyName: String?, engineIsStale: Bool, hasClusterSites: Bool
    ) -> UpdateAfterInstallNotice? {
        guard let lastLaunched, let earlier = SemanticVersion(lastLaunched),
            current > earlier
        else { return nil }
        var lines: [String] = []
        if let previousCopyName {
            lines.append(goingBackLine(previousCopyName: previousCopyName, appName: appName))
        }
        if engineIsStale { lines.append(engineLine) }
        if hasClusterSites { lines.append(clusterLine) }
        return UpdateAfterInstallNotice(
            headline: "SteerLab was updated from \(earlier.releaseDescription) to "
                + "\(current.releaseDescription).",
            lines: lines, offersEngineSetup: engineIsStale)
    }
}
