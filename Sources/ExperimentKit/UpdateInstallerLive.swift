import Darwin
import Foundation
import Security

// MARK: - The updater's real seams
//
// `UpdateInstaller.swift` decides; this file touches the world: the network
// (URLSession, GitHub's hosts only, no credentials), Apple's code-signature
// checks (offline), the running process's own location, the processes
// running out of this app, and the detached step that reopens it.

// MARK: - The network: GitHub only, unauthenticated, streamed

/// Streams one download through an ephemeral session: no cookies, no cache,
/// no stored credentials, and no header beyond what URLSession always sends.
/// A redirect is followed only to another GitHub host over HTTPS; any other
/// is refused and ends the transfer. An authentication challenge other than
/// the server's own certificate is cancelled, never answered.
public struct GitHubUpdateTransport: UpdateTransport {
    /// No bytes for this long ends the transfer.
    public let stallTimeout: TimeInterval

    public init(stallTimeout: TimeInterval = 60) {
        self.stallTimeout = stallTimeout
    }

    public func open(_ url: URL) throws -> AsyncThrowingStream<UpdateTransferEvent, Error> {
        if LaunchCheckMode.isActive {
            let description = "GET \(url.absoluteString)"
            LaunchCheckMode.recordBlockedAttempt(description)
            throw LaunchCheckMode.refusalError(description)
        }
        guard UpdateReleaseAssets.isTrustedHost(url) else {
            throw UpdateInstallError.untrustedDownloadLocation(url.host() ?? url.absoluteString)
        }
        let (stream, continuation) = AsyncThrowingStream<UpdateTransferEvent, Error>.makeStream()
        let configuration = Self.sessionConfiguration(stallTimeout: stallTimeout)
        let delegate = UpdateStreamingDelegate(continuation: continuation)
        let session = URLSession(
            configuration: configuration, delegate: delegate, delegateQueue: nil)
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        let task = session.dataTask(with: request)
        continuation.onTermination = { _ in
            task.cancel()
            session.invalidateAndCancel()
        }
        task.resume()
        return stream
    }

    /// No cookies, no cache, no stored credentials, and a fixed User-Agent:
    /// the default one names this app's version and the macOS build, which
    /// the privacy promise says are never sent.
    static func sessionConfiguration(stallTimeout: TimeInterval) -> URLSessionConfiguration {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = stallTimeout
        configuration.httpShouldSetCookies = false
        configuration.httpCookieAcceptPolicy = .never
        configuration.urlCache = nil
        configuration.urlCredentialStorage = nil
        configuration.httpAdditionalHeaders = ["User-Agent": UpdateCheckOverrides.userAgent]
        return configuration
    }
}

/// `@unchecked Sendable`: the only stored state is the stream continuation
/// (itself Sendable); URLSession calls the delegate on its own serial queue.
final class UpdateStreamingDelegate: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let continuation: AsyncThrowingStream<UpdateTransferEvent, Error>.Continuation

    init(continuation: AsyncThrowingStream<UpdateTransferEvent, Error>.Continuation) {
        self.continuation = continuation
    }

    func urlSession(
        _ session: URLSession, task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
        completionHandler: @escaping @Sendable (URLRequest?) -> Void
    ) {
        guard let target = request.url, UpdateReleaseAssets.isTrustedHost(target) else {
            completionHandler(nil)
            continuation.finish(
                throwing: UpdateInstallError.untrustedDownloadLocation(
                    request.url?.host() ?? "an unknown address"))
            task.cancel()
            return
        }
        completionHandler(request)
    }

    func urlSession(
        _ session: URLSession, task: URLSessionTask,
        didReceive challenge: URLAuthenticationChallenge,
        completionHandler: @escaping @Sendable (URLSession.AuthChallengeDisposition, URLCredential?) -> Void
    ) {
        if challenge.protectionSpace.authenticationMethod == NSURLAuthenticationMethodServerTrust {
            completionHandler(.performDefaultHandling, nil)
        } else {
            completionHandler(.cancelAuthenticationChallenge, nil)
        }
    }

    func urlSession(
        _ session: URLSession, dataTask: URLSessionDataTask,
        didReceive response: URLResponse,
        completionHandler: @escaping @Sendable (URLSession.ResponseDisposition) -> Void
    ) {
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        let length = response.expectedContentLength
        continuation.yield(.response(status: status, expectedLength: length >= 0 ? length : nil))
        completionHandler(.allow)
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        continuation.yield(.data(data))
    }

    func urlSession(
        _ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?
    ) {
        if let error {
            continuation.finish(throwing: error)
        } else {
            continuation.finish()
        }
        session.finishTasksAndInvalidate()
    }
}

// MARK: - Code signatures: Apple's checks, offline

/// The Security framework, told never to use the network: the notarization
/// requirement is then satisfied only by a ticket stapled inside the app.
public struct SecurityFrameworkCodeSigning: UpdateCodeSigning {
    public init() {}

    static let developerIDRequirement =
        "anchor apple generic and certificate 1[field.1.2.840.113635.100.6.2.6] exists"
        + " and certificate leaf[field.1.2.840.113635.100.6.1.13] exists"

    private static let offline = SecCSFlags.noNetworkAccess

    public func runningAppSignature() -> RunningAppSignature {
        var code: SecCode?
        guard SecCodeCopySelf([], &code) == errSecSuccess, let code else {
            return .unreadable("the running process has no code object")
        }
        var staticCode: SecStaticCode?
        guard SecCodeCopyStaticCode(code, [], &staticCode) == errSecSuccess,
            let staticCode
        else { return .unreadable("the running app's code could not be located") }
        return Self.signature(of: staticCode)
    }

    /// The signature of a bundle on disk, read the same way (tests use it on
    /// bundles they sign themselves).
    public func signature(ofBundleAt url: URL) -> RunningAppSignature {
        var staticCode: SecStaticCode?
        guard SecStaticCodeCreateWithPath(url as CFURL, [], &staticCode) == errSecSuccess,
            let staticCode
        else { return .unreadable("the app at \(url.path) could not be read") }
        return Self.signature(of: staticCode)
    }

    static func signature(of staticCode: SecStaticCode) -> RunningAppSignature {
        var information: CFDictionary?
        let status = SecCodeCopySigningInformation(
            staticCode, SecCSFlags(rawValue: kSecCSSigningInformation), &information)
        if status == errSecCSUnsigned { return .unsigned }
        guard status == errSecSuccess, let dictionary = information as? [String: Any] else {
            return .unreadable(message(for: status))
        }
        if dictionary[kSecCodeInfoIdentifier as String] == nil { return .unsigned }
        let flags = (dictionary[kSecCodeInfoFlags as String] as? NSNumber)?.uint32Value ?? 0
        if flags & SecCodeSignatureFlags.adhoc.rawValue != 0 { return .adHoc }
        let team = dictionary[kSecCodeInfoTeamIdentifier as String] as? String
        guard let team, satisfies(staticCode, requirement: developerIDRequirement) else {
            return .otherCertificate(teamID: team)
        }
        let notarized = satisfies(staticCode, requirement: "notarized")
        return .developerID(teamID: team, notarized: notarized)
    }

    /// Basic validation only (the running app is not re-hashed at every
    /// check) against one requirement, offline.
    static func satisfies(_ staticCode: SecStaticCode, requirement text: String) -> Bool {
        var requirement: SecRequirement?
        guard SecRequirementCreateWithString(text as CFString, [], &requirement) == errSecSuccess,
            let requirement
        else { return false }
        let flags = SecCSFlags(rawValue: UInt32(kSecCSBasicValidateOnly)).union(offline)
        return SecStaticCodeCheckValidity(staticCode, flags, requirement) == errSecSuccess
    }

    public func validate(bundle: URL, requirement text: String) throws {
        var staticCode: SecStaticCode?
        let created = SecStaticCodeCreateWithPath(bundle as CFURL, [], &staticCode)
        guard created == errSecSuccess, let staticCode else {
            throw UpdateInstallError.signatureRejected(Self.message(for: created))
        }
        var requirement: SecRequirement?
        let parsed = SecRequirementCreateWithString(text as CFString, [], &requirement)
        guard parsed == errSecSuccess, let requirement else {
            throw UpdateInstallError.signatureRejected(
                "the signature requirement could not be read (" + Self.message(for: parsed) + ")")
        }
        // Strict validation of every architecture and every piece of nested
        // code (the bundled command line, the shader library), offline.
        let flags = SecCSFlags(
            rawValue: UInt32(kSecCSCheckAllArchitectures) | UInt32(kSecCSCheckNestedCode)
                | UInt32(kSecCSStrictValidate) | UInt32(kSecCSRestrictSymlinks)
        ).union(Self.offline)
        var error: Unmanaged<CFError>?
        let status = SecStaticCodeCheckValidityWithErrors(staticCode, flags, requirement, &error)
        guard status == errSecSuccess else {
            let detail = error.map { CFErrorCopyDescription($0.takeRetainedValue()) as String }
            throw UpdateInstallError.signatureRejected(
                Self.explain(status) + (detail.map { " (\($0))" } ?? ""))
        }
    }

    static func explain(_ status: OSStatus) -> String {
        switch status {
        case errSecCSReqFailed:
            return "it is not a notarized Developer ID build of this app from the same "
                + "developer as the copy you are running"
        case errSecCSUnsigned:
            return "it is not signed"
        case errSecCSSignatureFailed, errSecCSBadResource, errSecCSResourcesInvalid,
            errSecCSResourcesNotSealed, errSecCSResourcesNotFound:
            return "its contents do not match its signature"
        default:
            return message(for: status)
        }
    }

    static func message(for status: OSStatus) -> String {
        (SecCopyErrorMessageString(status, nil) as String?) ?? "error \(status)"
    }
}

// MARK: - The running app, as eligibility sees it

public enum UpdateInstallContextReader {
    /// The context for THIS process. Only reads: the bundle's location and
    /// folder permissions, its own signature (offline), and whether a code
    /// checkout sits beside it.
    public static func current(
        codeSigning: any UpdateCodeSigning = SecurityFrameworkCodeSigning()
    ) -> UpdateInstallContext {
        let bundle = Bundle.main.bundleURL.standardizedFileURL
        let isApp = bundle.pathExtension == "app"
        // A developer install: the same resolver the local engine uses finds
        // a code checkout beside the app (or in the default home). Read-only.
        let hasCheckout = (try? CodeResources.executableCheckout().get()) != nil
        let readOnly =
            (try? bundle.resourceValues(forKeys: [.volumeIsReadOnlyKey]))?.volumeIsReadOnly
            ?? false
        return UpdateInstallContext(
            appBundle: isApp ? bundle : nil,
            bundleIdentifier: Bundle.main.bundleIdentifier,
            signature: isApp ? codeSigning.runningAppSignature() : .unsigned,
            hasSourceCheckout: hasCheckout,
            isTranslocated: isTranslocated(bundle),
            isOnReadOnlyVolume: readOnly,
            canWriteParentFolder: FileManager.default.isWritableFile(
                atPath: bundle.deletingLastPathComponent().path))
    }

    /// Gatekeeper's App Translocation runs a quarantined app opened in place
    /// from a randomized read-only mount under `…/AppTranslocation/…`.
    public static func isTranslocated(_ bundle: URL) -> Bool {
        bundle.path.contains("/AppTranslocation/")
    }
}

// MARK: - Processes running out of this app

public enum UpdateBundleProcesses {
    /// How many OTHER processes run an executable inside `app` — the bundled
    /// command line a coding assistant started, most often. Replacing the app
    /// under them would hand them the new version's files mid-command.
    public static func count(insideApp app: URL, excluding own: Int32 = getpid()) -> Int {
        let prefix = app.resolvingSymlinksInPath().standardizedFileURL.path + "/"
        let capacity = proc_listallpids(nil, 0)
        guard capacity > 0 else { return 0 }
        var pids = [pid_t](repeating: 0, count: Int(capacity) + 64)
        let listed = pids.withUnsafeMutableBufferPointer {
            proc_listallpids($0.baseAddress, Int32($0.count * MemoryLayout<pid_t>.size))
        }
        guard listed > 0 else { return 0 }
        var buffer = [CChar](repeating: 0, count: Int(MAXPATHLEN) * 4)
        var found = 0
        for pid in pids.prefix(Int(listed)) where pid > 0 && pid != own {
            let length = proc_pidpath(pid, &buffer, UInt32(buffer.count))
            guard length > 0 else { continue }
            let path = String(
                decoding: buffer.prefix(Int(length)).map { UInt8(bitPattern: $0) },
                as: UTF8.self)
            if path.hasPrefix(prefix) { found += 1 }
        }
        return found
    }
}

// MARK: - Reopening

extension UpdateRelaunch {
    /// Start the detached waiter (see `waiterArguments`). The app quits right
    /// after; the waiter opens the new app once this process has exited.
    public static func launchWaiter(for app: URL, pid: Int32 = getpid()) throws {
        let argv = waiterArguments(pid: pid, app: app)
        let process = Process()
        process.executableURL = URL(filePath: argv[0])
        process.arguments = Array(argv.dropFirst())
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
    }
}
