import Foundation

// =============================================================================
// `remote … --site <id>` (CLUSTER-CLI-LIFECYCLE-PLAN §5.4/§6) — Phase C.
//
// Once a site is connected, an agent should not need to know the ephemeral
// local port or handle the bearer token at all:
//
//     steerlab-cli remote capabilities --site lab-cluster
//
// This resolves the endpoint from the shared site registry and reads the token
// from the Keychain INTERNALLY. The token is returned so the HTTP client can
// use it, and it is never rendered: no caller in this repository puts a
// `ClusterRemoteSiteResolution.token` into stdout, stderr, JSON, argv, or a
// record (asserted by test).
//
// `--url`/`--token` survive as the compatibility path for unmanaged servers.
// =============================================================================

/// A resolved site connection, ready to hand to `ClusterClient`.
public struct ClusterRemoteSiteResolution: Sendable {
    public var siteID: String
    public var siteName: String
    public var baseURL: URL
    /// The bearer token. Present so the transport can authenticate; NEVER for
    /// display. `tokenSource` is what a report gets to say about it.
    public var token: String?
    public var tokenSource: String?

    public var tokenAvailable: Bool { token != nil }

    public init(
        siteID: String, siteName: String, baseURL: URL, token: String?,
        tokenSource: String?
    ) {
        self.siteID = siteID
        self.siteName = siteName
        self.baseURL = baseURL
        self.token = token
        self.tokenSource = tokenSource
    }

    /// What a log line or envelope may say — presence and provenance only.
    public var redactedSummary: String {
        "\(siteName) (\(siteID)) at \(baseURL.absoluteString), token "
            + (tokenAvailable ? "from \(tokenSource ?? "keychain")" : "absent")
    }
}

/// Which endpoint source a `remote` invocation selected.
public enum ClusterRemoteEndpointChoice: Sendable, Equatable {
    /// A saved cluster site: endpoint from the registry, token from Keychain.
    case site(String)
    /// The compatibility path: an explicit URL (nil = the historical default).
    case url(String?)
}

public enum ClusterRemoteSiteResolver {

    /// Decide where a `remote` verb should talk to.
    ///
    /// The two forms are mutually exclusive rather than ordered by precedence:
    /// silently preferring one would let a typo look like a working connection
    /// to the WRONG server, which is the failure mode a study cannot afford.
    public static func choose(
        site: String?, url: String?
    ) throws -> ClusterRemoteEndpointChoice {
        switch (site, url) {
        case (.some, .some):
            throw ClusterCLIError.siteAndURLAreMutuallyExclusive
        case (.some(let reference), .none):
            return .site(reference)
        case (.none, let url):
            return .url(url)
        }
    }
}

extension ClusterRemoteSiteResolver {

    /// Resolve a site reference to an endpoint + token, or refuse with a repair
    /// action naming `cluster ensure`.
    ///
    /// The staleness check is real rather than optimistic: a registered
    /// endpoint whose forward is no longer up is a site to repair, not one to
    /// send a request at and time out on.
    public static func resolve(
        reference: String,
        repository: ClusterSiteRepository = ClusterSiteRepository(),
        secrets: any ClusterSecretStore = KeychainClusterSecretStore(),
        tunnel: any ClusterTunnelControlling = SSHClusterTunnelController()
    ) async throws -> ClusterRemoteSiteResolution {
        guard let site = try repository.resolve(reference: reference) else {
            throw ClusterLifecycleError.unknownSite(reference)
        }
        let token = secrets.token(forKey: site.tokenKey)

        // Direct transport needs no forward: the site IS a URL.
        if let direct = site.profile.directURLString,
            let url = ClusterConnectionStore.endpointURL(from: direct)
        {
            return ClusterRemoteSiteResolution(
                siteID: site.id, siteName: site.displayName, baseURL: url,
                token: token, tokenSource: token == nil ? nil : "keychain")
        }

        guard let endpoint = site.lastEndpoint,
            let url = ClusterConnectionStore.endpointURL(from: endpoint)
        else {
            throw ClusterCLIError.siteNotConnected(
                siteID: site.id,
                reason: "no endpoint has been registered for it yet")
        }
        let observation = await tunnel.observe(
            site: site.profile,
            persistedPort: ClusterLifecycleInspector.persistedPort(for: site),
            targetHost: nil)
        switch observation {
        case .up, .notApplicable:
            break
        case .absent:
            throw ClusterCLIError.siteNotConnected(
                siteID: site.id,
                reason: "the registered endpoint \(url.absoluteString) has no forward")
        case .stale(let port, let reason):
            throw ClusterCLIError.siteNotConnected(
                siteID: site.id,
                reason: "the forward on 127.0.0.1:\(port) is stale (\(reason))")
        case .conflicted(let port, let reason):
            throw ClusterCLIError.siteNotConnected(
                siteID: site.id,
                reason: "127.0.0.1:\(port) is held by something else (\(reason))")
        }
        guard let token else {
            throw ClusterCLIError.siteNotConnected(
                siteID: site.id,
                reason: "no bearer token for it is in the Keychain")
        }
        return ClusterRemoteSiteResolution(
            siteID: site.id, siteName: site.displayName, baseURL: url,
            token: token, tokenSource: "keychain")
    }
}
