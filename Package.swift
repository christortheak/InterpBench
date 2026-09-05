// swift-tools-version: 6.2
// SteerLab — activation-steering experiments on open-weight LLMs (see README.md).
import PackageDescription

let package = Package(
    name: "SteerLab",
    platforms: [
        // macOS 26.2 is the runtime floor for MLX M5 Neural Accelerator support;
        // we pin 26.4 because Xcode 27 itself requires it. See CLAUDE.md › Environment.
        .macOS("26.4")
    ],
    products: [
        .library(name: "SteeringKit", targets: ["SteeringKit"]),
        .library(name: "ExperimentKit", targets: ["ExperimentKit"]),
        .executable(name: "steerlab-cli", targets: ["SteerLabCLI"]),
        .executable(name: "SteerLabApp", targets: ["SteerLabApp"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.31.3")),
        // mlx-swift-lm 3.x decoupled tokenizers/downloaders; we pin them ourselves
        // and wire them via the MLXHuggingFace macros. See CLAUDE.md › Dependencies.
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        // The Jinja engine swift-transformers renders chat templates with —
        // pinned directly so the chat-template capability probe can render a
        // template SOURCE (the cross-engine fixture families) through the
        // same engine a loaded tokenizer renders through.
        .package(url: "https://github.com/huggingface/swift-jinja.git", from: "2.3.6"),
    ],
    targets: [
        // Concept-agnostic steering core — no UI, no experiment logic.
        .target(
            name: "SteeringKit",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Jinja", package: "swift-jinja"),
            ]
        ),
        // Experiment definitions, run configs, controls, metrics.
        .target(
            name: "ExperimentKit",
            dependencies: [
                "SteeringKit",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                // Direct tokenizer access for the assistant-prefix
                // continuation render (applyChatTemplate with
                // addGenerationPrompt: false — the MLXLMCommon tokenizer
                // bridge only exposes the generation-prompt form).
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        // Headless runner — the GUI calls into this layer, not the reverse.
        .executableTarget(
            name: "SteerLabCLI",
            dependencies: ["ExperimentKit"],
            path: "Sources/steerlab-cli"
        ),
        // Thin SwiftUI shell over ExperimentKit state.
        .executableTarget(
            name: "SteerLabApp",
            dependencies: ["ExperimentKit"]
        ),
        .testTarget(name: "SteeringKitTests", dependencies: ["SteeringKit"]),
        .testTarget(
            name: "ExperimentKitTests",
            dependencies: ["ExperimentKit"],
            // Committed cross-engine parity fixtures (WS7.3), read via
            // #filePath — byte-identical twins of Server/tests/fixtures/parity.
            exclude: ["Fixtures"]
        ),
    ]
)
