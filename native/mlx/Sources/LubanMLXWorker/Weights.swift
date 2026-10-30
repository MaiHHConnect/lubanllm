import Foundation
import CryptoKit
import MLX
import MLXNN
import MLXLMCommon

struct WorkerFailure: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}

func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw WorkerFailure(message) }
}

struct FileBinding: Equatable {
    let device: UInt64
    let inode: UInt64
    let bytes: UInt64
    let modified: Date
    let created: Date
    init(_ url: URL) throws {
        let a = try FileManager.default.attributesOfItem(atPath: url.path)
        try require(a[.type] as? FileAttributeType == .typeRegular, "Weights must be regular files")
        device = (a[.systemNumber] as! NSNumber).uint64Value
        inode = (a[.systemFileNumber] as! NSNumber).uint64Value
        bytes = (a[.size] as! NSNumber).uint64Value
        modified = a[.modificationDate] as! Date
        created = a[.creationDate] as! Date
    }
}

// Core/expert files are small. Validate their immutable lifecycle once before MLX loads them.
func verifiedBF16(_ path: String, sha: String, shapes: [String: [Int]]) throws -> [String: MLXArray] {
    let url = URL(fileURLWithPath: path)
    let before = try FileBinding(url)
    let data = try Data(contentsOf: url, options: .mappedIfSafe)
    try require(data.count >= 8 && data.count <= 64 * 1024 * 1024, "Invalid adapter size")
    let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    try require(digest == sha, "Adapter SHA256 mismatch")
    let count = data.withUnsafeBytes { Int(UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))) }
    try require(count > 1 && count <= data.count - 8, "Invalid safetensors header")
    let header = try JSONSerialization.jsonObject(with: data.subdata(in: 8..<(8 + count))) as? [String: Any]
    guard let header else { throw WorkerFailure("Invalid adapter header") }
    try require(Set(header.keys.filter { $0 != "__metadata__" }) == Set(shapes.keys), "Adapter keys mismatch")
    var spans = [(Int, Int)]()
    for (key, shape) in shapes {
        guard let entry = header[key] as? [String: Any], let offsets = entry["data_offsets"] as? [Int] else {
            throw WorkerFailure("Missing adapter tensor: \(key)")
        }
        try require(entry["dtype"] as? String == "BF16" && entry["shape"] as? [Int] == shape, "Adapter dtype/shape mismatch: \(key)")
        let expected = shape.reduce(1, *) * 2
        try require(offsets.count == 2 && offsets[0] >= 0 && offsets[1] - offsets[0] == expected && offsets[1] <= data.count - 8 - count, "Adapter offsets invalid")
        let start = 8 + count + offsets[0], end = 8 + count + offsets[1]
        try data.withUnsafeBytes { raw in
            for p in stride(from: start, to: end, by: 2) {
                let v = UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: p, as: UInt16.self))
                try require(v & 0x7f80 != 0x7f80, "Nonfinite adapter value")
            }
        }
        spans.append((offsets[0], offsets[1]))
    }
    spans.sort { $0.0 < $1.0 }
    var end = 0
    for span in spans { try require(span.0 == end, "Adapter gaps/overlap"); end = span.1 }
    try require(end == data.count - 8 - count, "Adapter trailing payload")
    let arrays = try MLX.loadArrays(url: url)
    try require(try FileBinding(url) == before, "Adapter changed while loading")
    return arrays
}

func expertWidth(_ path: String) throws -> Int {
    let f = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
    defer { try? f.close() }
    guard let prefix = try f.read(upToCount: 8), prefix.count == 8 else { throw WorkerFailure("Short expert header") }
    let rawCount = prefix.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) }
    try require(rawCount > 0 && rawCount <= 65536, "Invalid expert header size")
    let n = Int(rawCount)
    guard let bytes = try f.read(upToCount: n), bytes.count == n,
          let header = try JSONSerialization.jsonObject(with: bytes) as? [String: Any],
          let gate = header["gate.weight"] as? [String: Any], let shape = gate["shape"] as? [Int], shape.count == 2 else {
        throw WorkerFailure("Invalid expert gate shape")
    }
    try require([32,64,128,256,512].contains(shape[0]), "Unsupported expert width")
    return shape[0]
}

final class ExpertMLP: Module, UnaryLayer {
    @ModuleInfo var base: Module
    @ModuleInfo var gate: Linear
    @ModuleInfo var up: Linear
    @ModuleInfo var down: Linear
    var detachFrozenInput = false
    init(base: Module, weights: [String: MLXArray]) {
        _base.wrappedValue = base
        _gate.wrappedValue = Linear(weight: weights["gate.weight"]!)
        _up.wrappedValue = Linear(weight: weights["up.weight"]!)
        _down.wrappedValue = Linear(weight: weights["down.weight"]!)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Explicit BF16 boundaries match the existing NeuralExpert / native expert graph.
        let input = (detachFrozenInput ? stopGradient(x) : x).asType(.bfloat16)
        let g = gate(input).asType(.bfloat16)
        let u = up(input).asType(.bfloat16)
        let activated = (g.asType(.float32) * sigmoid(g.asType(.float32))).asType(.bfloat16)
        let hidden = (activated * u).asType(.bfloat16)
        let residual = down(hidden).asType(.bfloat16)
        let original = (base as! any UnaryLayer)(x).asType(.bfloat16)
        return ((detachFrozenInput ? stopGradient(original) : original) + residual).asType(.bfloat16)
    }
}

func installAdapterModules(_ model: any LanguageModel, core: MLXArray?, expert: [String: MLXArray]?) throws {
    var layer: Module?, base: Module?
    model.visit { key, module in
        if key == "language_model.model.layers.23" { layer = module }
        if key == "language_model.model.layers.23.mlp" { base = module }
    }
    guard let layer, let base, base is any UnaryLayer else { throw WorkerFailure("Missing final Qwen MLP") }
    // Numeric unflattened paths cannot represent a sparse update to the model's layer array.
    // Update the actual parent modules, retaining strict key/type validation.
    if let core {
        try base.update(modules: ModuleChildren.unflattened(["down_proj": Linear(weight: core)]), verify: .noUnusedKeys)
    }
    if let expert {
        try layer.update(modules: ModuleChildren.unflattened(["mlp": ExpertMLP(base: base, weights: expert)]), verify: .noUnusedKeys)
    }
}

func attachAdapters(_ model: any LanguageModel, options: Options) throws -> Int {
    var coreWeight: MLXArray?
    if let core = options.coreFile {
        let weights = try verifiedBF16(core, sha: options.coreSHA!, shapes: [options.coreKey!: [2048,6144]])
        coreWeight = weights[options.coreKey!]!
    }
    var expertWeights: [String: MLXArray]?
    var width = 0
    if let expert = options.expertFile {
        width = try expertWidth(expert)
        expertWeights = try verifiedBF16(expert, sha: options.expertSHA!, shapes: ["gate.weight": [width,2048], "up.weight": [width,2048], "down.weight": [2048,width]])
    }
    if coreWeight != nil || expertWeights != nil {
        try installAdapterModules(model, core: coreWeight, expert: expertWeights)
    }
    return width
}
