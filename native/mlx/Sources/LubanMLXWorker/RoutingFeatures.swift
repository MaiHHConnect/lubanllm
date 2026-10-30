import Foundation
import CoreFoundation
import MLX
import MLXLLM
import MLXLMCommon

/// Quantized MLX router features.  The legacy BF16 model is deliberately not
/// touched here: these features come from the same Q4 backbone used to answer.
func routeFeatures(_ message: [String: Any], model: any LanguageModel) throws {
    guard let id = message["id"] as? String, !id.isEmpty, id.utf8.count <= 256 else {
        throw WorkerFailure("route_features requires id")
    }
    guard let raw = message["tokens"] as? [NSNumber] else {
        throw WorkerFailure("route_features requires tokens")
    }
    try require(!raw.isEmpty && raw.count <= 32768, "Invalid route feature context")
    try require(raw.allSatisfy {
        CFGetTypeID($0) != CFBooleanGetTypeID() && $0.doubleValue.isFinite
            && $0.doubleValue >= 0 && $0.doubleValue < 248320
            && $0.doubleValue.rounded(.towardZero) == $0.doubleValue
    }, "Invalid route feature token")
    let tokens = raw.map { $0.intValue }
    let layers = message["layers"] ?? 4
    guard let count = layers as? NSNumber, CFGetTypeID(count) != CFBooleanGetTypeID(),
          count.doubleValue.isFinite, (1...4).contains(count.doubleValue),
          count.doubleValue.rounded(.towardZero) == count.doubleValue else {
        throw WorkerFailure("Invalid route feature prefix depth")
    }
    let layerCount = count.intValue
    try require((1...4).contains(layerCount), "Router prefix must use one to four layers")
    guard let qwen = model as? Qwen35Model else {
        throw WorkerFailure("route_features requires Qwen3.5 MLX model")
    }
    let input = MLXArray(tokens, [1, tokens.count])
    let hidden = qwen.prefixForward(input, layerCount: layerCount)
    let pooled = hidden.mean(axis: 1).asType(.float32)
    eval(pooled)
    let features = pooled.asArray(Float.self)
    try require(features.count == 2048 && features.allSatisfy { $0.isFinite }, "Invalid route features")
    try emit([
        "id": id,
        "event": "route_features",
        "features": features,
        "feature_dim": features.count,
        "layers": layerCount,
        "source": "mlx-q4-prefix",
        "legacy_bf16_loaded": false,
    ])
}
