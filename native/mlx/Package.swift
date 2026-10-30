// swift-tools-version: 6.2
import PackageDescription
let package = Package(
    name: "LubanMLXWorker",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "luban-mlx-worker", targets: ["LubanMLXWorker"])],
    dependencies: [
        .package(name: "mlx-swift-lm", path: "../../build/native-mlx-deps/mlx-swift-lm-3.32.3"),
        .package(name: "mlx-swift", path: "../../build/native-mlx-deps/mlx-swift-0.32.3"),
    ],
    targets: [.executableTarget(name: "LubanMLXWorker", dependencies: [
        .product(name: "MLXLLM", package: "mlx-swift-lm"),
        .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
        .product(name: "MLXVLM", package: "mlx-swift-lm"),
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
    ])]
)
