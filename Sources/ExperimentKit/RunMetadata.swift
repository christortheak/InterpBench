import Foundation
import SteeringKit

/// Canonical per-run `config.json` — one uniform provenance stamp for EVERY
/// kind of run directory (extraction, study, validate, sweep, evaluate,
/// multi-agent, LoRA train, reader fit, norm backfill, variant save, …).
/// The JSON shape is a pinned cross-engine contract (the Python server's
/// `experiment/run_config.py` writes byte-compatible keys):
///
/// ```json
/// {
///   "schemaVersion": 4,
///   "runId": "<run-directory basename>",
///   "runType": "<extract|validate|sweep|run|evaluate|analyze|reader-fit|
///                norm-backfill|lora-train|variant-save|multi-agent|...>",
///   "createdAt": "<ISO8601, UTC Z>",
///   "substrate": "swift-mlx",
///   "appVersion": "<engine version stamp, e.g. swift-app 0.9.0-dev+abc12345>",
///   "platform": "<os-arch, e.g. macOS-arm64 — never a hostname>",
///   "modelID": null | "<model id>",
///   "revision": null | "<commit>",
///   "experiment": null | "<experiment name>",
///   "experimentHash": null | "<manifest content hash>",
///   "temperature": null | <number>,      // sampling policy of generation-
///   "samplesPerItem": null | <int>,      // bearing runs; null elsewhere
///   "seedPolicy": null | "manifestSeeds" | "derivedSHA256",
///   "dtype": null | "<bfloat16|float16|float32>",  // schema 3; see below
///   "pythonEnvironment": null,           // schema 4; null on THIS engine
///   "jobId": null | "<slurm job id>",
///   "notes": {}                          // engine-specific extras ONLY here
/// }
/// ```
///
/// Schema rules: the top-level key set is CLOSED — adding, removing, or
/// renaming a key requires bumping `schemaVersion` and updating
/// `contractKeys` (both engines' closed-key tests compare the same literal
/// list). Every key is always present; absent knowledge encodes as JSON
/// `null`, never key omission. No user paths, hostnames, or secrets.
///
/// Additive only: it never replaces a run type's richer artifacts (manifest
/// snapshots, sidecars, reports) and is never written where a legacy
/// `config.json` already exists with different semantics (`ToyConceptRun`
/// keeps its historical config file untouched).
///
/// Schema 3 (2026-07-24) added `dtype`: the numeric precision the model
/// ACTUALLY ran in. Greedy decoding is not precision-proof — at a near-tie
/// between two tokens, bf16 and fp16 round differently, the argmax flips,
/// and the continuation diverges — so a measured run that does not record
/// its precision cannot be reproduced from its own record.
///
/// **This engine writes null.** MLX study models are quantized repos
/// (`-4bit`, `-8bit`) with no single parameter dtype to report, and the Mac
/// is a testing substrate rather than a measurement one. The key exists here
/// so the cross-engine shape stays identical and a reader never has to
/// wonder whether a missing key means "float32" or "not recorded" — null
/// means the latter, explicitly.
///
/// Schema 4 (2026-08-18, WP6 R1) added `pythonEnvironment`: the resolved
/// versions of the science-relevant Python packages a run actually imported
/// (`{"python": …, "implementation": …, "packages": {"torch": …, …}}`).
/// `Server/pyproject.toml` declares dependency FLOORS, so `appVersion` pinned
/// the engine's own code while the stack underneath it was free to differ
/// between two sites running the same frozen manifest; committed per-platform
/// locks state the intended resolution and this key records the achieved one.
///
/// **This engine writes null**, following the same engine-conditional pattern
/// `dtype` established in schema 3: there is no Python environment under
/// Swift/MLX, and inventing a `swiftEnvironment` twin would be a second,
/// differently-shaped key for a substrate that is a *testing* tier anyway
/// (the Swift engine's own build identity already rides in `appVersion`).
/// Null here reads as "this engine has no Python environment", never as
/// "nobody recorded it".
public enum RunMetadata {
    public static let schemaVersion = 4
    public static let fileName = "config.json"

    /// The CLOSED top-level key set (sorted). Byte-identical across engines —
    /// the Python server pins the same list in `run_config.RUN_CONFIG_KEYS`.
    public static let contractKeys: [String] = [
        "appVersion",
        "createdAt",
        "dtype",
        "experiment",
        "experimentHash",
        "jobId",
        "modelID",
        "notes",
        "platform",
        "pythonEnvironment",
        "revision",
        "runId",
        "runType",
        "samplesPerItem",
        "schemaVersion",
        "seedPolicy",
        "substrate",
        "temperature",
    ]

    /// OS + architecture, e.g. "macOS-arm64" / "linux-x86_64". Never a
    /// hostname (the stamp must carry no machine identity).
    public static var platform: String {
        #if os(macOS)
            let system = "macOS"
        #elseif os(Linux)
            let system = "linux"
        #else
            let system = "unknown"
        #endif
        #if arch(arm64)
            return system + "-arm64"
        #elseif arch(x86_64)
            return system + "-x86_64"
        #else
            return system + "-unknown"
        #endif
    }

    /// The scheduler job id this process runs under, if any (Slurm sets
    /// `SLURM_JOB_ID`; a local Mac run stamps null).
    public static var environmentJobID: String? {
        let value = ProcessInfo.processInfo.environment["SLURM_JOB_ID"] ?? ""
        return value.isEmpty ? nil : value
    }

    /// Pure payload assembly (unit-testable without touching disk). Every
    /// key is ALWAYS present; absent values encode as JSON `null` — the
    /// pinned contract is `null|value`, not key omission.
    public static func payload(
        runID: String,
        runType: String,
        createdAt: Date = Date(),
        modelID: String? = nil,
        revision: String? = nil,
        experiment: String? = nil,
        experimentHash: String? = nil,
        temperature: Double? = nil,
        samplesPerItem: Int? = nil,
        seedPolicy: String? = nil,
        dtype: String? = nil,
        jobID: String? = environmentJobID,
        notes: [String: String] = [:],
        /// Notes whose value is a JSON OBJECT rather than a string — merged
        /// into the same `notes` block. `notes` stayed string-valued because
        /// every note before 2026-08-29 was one sentence; the evaluate
        /// subsample's stamp is five fields a reader has to be able to read
        /// individually, and flattening it to prose would have made the run's
        /// own record less machine-readable than the report beside it.
        structuredNotes: [String: Any] = [:]
    ) -> [String: Any] {
        var mergedNotes: [String: Any] = notes
        for (key, value) in structuredNotes { mergedNotes[key] = value }
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        formatter.timeZone = TimeZone(identifier: "UTC")
        return [
            "schemaVersion": schemaVersion,
            "runId": runID,
            "runType": runType,
            "createdAt": formatter.string(from: createdAt),
            "substrate": RepEReader.substrate,
            "appVersion": SteerLabVersion.current,
            "platform": platform,
            "modelID": modelID ?? NSNull(),
            "revision": revision ?? NSNull(),
            "experiment": experiment ?? NSNull(),
            "experimentHash": experimentHash ?? NSNull(),
            "temperature": temperature ?? NSNull(),
            "samplesPerItem": samplesPerItem ?? NSNull(),
            "seedPolicy": seedPolicy ?? NSNull(),
            "dtype": dtype ?? NSNull(),
            // Schema 4: always null on this engine — there is no Python
            // environment under Swift/MLX. Present so the cross-engine shape
            // is identical and a reader of a Mac run sees an explicit "not
            // applicable" rather than a missing key.
            "pythonEnvironment": NSNull(),
            "jobId": jobID ?? NSNull(),
            "notes": mergedNotes,
        ]
    }

    /// Writes `config.json` into a run directory (`runId` is the directory's
    /// basename). Refuses to overwrite an existing file (run directories are
    /// immutable, and some legacy run types own a differently-shaped
    /// config.json).
    @discardableResult
    public static func write(
        runType: String,
        to runDirectory: URL,
        createdAt: Date = Date(),
        modelID: String? = nil,
        revision: String? = nil,
        experiment: String? = nil,
        experimentHash: String? = nil,
        temperature: Double? = nil,
        samplesPerItem: Int? = nil,
        seedPolicy: String? = nil,
        notes: [String: String] = [:],
        structuredNotes: [String: Any] = [:]
    ) throws -> URL {
        let url = runDirectory.appending(component: fileName)
        guard !FileManager.default.fileExists(atPath: url.path) else { return url }
        let data = try JSONSerialization.data(
            withJSONObject: payload(
                runID: runDirectory.lastPathComponent,
                runType: runType, createdAt: createdAt, modelID: modelID,
                revision: revision, experiment: experiment,
                experimentHash: experimentHash, temperature: temperature,
                samplesPerItem: samplesPerItem, seedPolicy: seedPolicy,
                notes: notes, structuredNotes: structuredNotes),
            options: [.prettyPrinted, .sortedKeys])
        try data.write(to: url, options: .atomic)
        return url
    }
}
