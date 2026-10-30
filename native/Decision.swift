import Foundation
import CryptoKit

enum NativeDecisionError: Error {
    case invalidWeights, invalidMetadata, invalidInput, hashMismatch
}

// The shared shallow graph returns raw layer-4 BF16 output, before final RMSNorm.
// Runtime.shallow(...).float() precedes FP32 masked-mean pooling and this head.
struct NativeDecision {
    static let actions = ["local_core", "neural_expert", "teacher"]
    let width: Int
    let weight: [Float]
    let bias: [Float]

    struct Policy: Codable {
        let calibrated: Bool
        let supportedActions: [String]
        let temperature: Float
        let threshold: Double

        init(calibrated: Bool, supportedActions: [String] = ["local_core", "teacher"],
             temperature: Float = 1, threshold: Double = 0.8) throws {
            guard temperature.isFinite, temperature > 0, threshold.isFinite,
                  threshold >= 0, threshold <= Double(1).nextUp, !supportedActions.isEmpty,
                  Set(supportedActions).count == supportedActions.count,
                  supportedActions.allSatisfy({ NativeDecision.actions.contains($0) }) else {
                throw NativeDecisionError.invalidMetadata
            }
            self.calibrated = calibrated
            self.supportedActions = supportedActions
            self.temperature = temperature
            self.threshold = threshold
        }

        enum CodingKeys: String, CodingKey {
            case calibrated, temperature, threshold
            case supportedActions = "supported_actions"
        }

        init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: CodingKeys.self)
            try self.init(calibrated: values.decode(Bool.self, forKey: .calibrated),
                          supportedActions: values.decodeIfPresent([String].self, forKey: .supportedActions)
                            ?? ["local_core", "teacher"],
                          temperature: values.decodeIfPresent(Float.self, forKey: .temperature) ?? 1,
                          threshold: values.decodeIfPresent(Double.self, forKey: .threshold) ?? 0.8)
        }
    }

    struct Result: Codable {
        let route: String
        let action: String
        let source: String
        let confidence: Float?
    }

    // Caller must independently verify model/core identity, active pointer and expert binding.
    init(file: URL, approvedSHA256: String) throws {
        let attributes = try FileManager.default.attributesOfItem(atPath: file.path)
        guard attributes[.type] as? FileAttributeType == .typeRegular,
              let size = attributes[.size] as? NSNumber, size.intValue <= 1 << 20 else {
            throw NativeDecisionError.invalidWeights
        }
        let raw = try Data(contentsOf: file)
        let actual = SHA256.hash(data: raw).map { String(format: "%02x", $0) }.joined()
        guard approvedSHA256.count == 64, actual == approvedSHA256 else {
            throw NativeDecisionError.hashMismatch
        }
        guard raw.count >= 8 else { throw NativeDecisionError.invalidWeights }
        let headerSize = raw.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) }
        guard headerSize <= UInt64(raw.count - 8) else { throw NativeDecisionError.invalidWeights }
        let headerCount = Int(headerSize)
        guard headerCount >= 2, headerCount <= raw.count - 8 else { throw NativeDecisionError.invalidWeights }
        let decoded = try JSONSerialization.jsonObject(with: raw.subdata(in: 8..<(8 + headerCount)))
        guard let header = decoded as? [String: Any],
              Set(header.keys).subtracting(["__metadata__"]) == Set(["projection.weight", "projection.bias"]) else {
            throw NativeDecisionError.invalidWeights
        }
        var intervals: [Range<Int>] = []
        func tensor(_ name: String) throws -> (shape: [Int], data: [Float]) {
            guard let entry = header[name] as? [String: Any], entry["dtype"] as? String == "F32",
                  let dims = entry["shape"] as? [Int], !dims.isEmpty, dims.count <= 2,
                  dims.allSatisfy({ $0 > 0 && $0 <= 65536 }),
                  let offsets = entry["data_offsets"] as? [Int], offsets.count == 2,
                  offsets[0] >= 0, offsets[1] >= offsets[0], offsets[1] <= raw.count - 8 - headerCount else {
                throw NativeDecisionError.invalidWeights
            }
            let count = dims.reduce(1, *)
            guard count <= 196608, offsets[1] - offsets[0] == count * 4 else {
                throw NativeDecisionError.invalidWeights
            }
            intervals.append(offsets[0]..<offsets[1])
            let start = 8 + headerCount + offsets[0]
            let numbers = raw.withUnsafeBytes { bytes in
                (0..<count).map { Float(bitPattern: UInt32(littleEndian:
                    bytes.loadUnaligned(fromByteOffset: start + $0 * 4, as: UInt32.self))) }
            }
            guard numbers.allSatisfy(\.isFinite) else { throw NativeDecisionError.invalidWeights }
            return (dims, numbers)
        }
        let w = try tensor("projection.weight"), b = try tensor("projection.bias")
        guard w.shape.count == 2, w.shape[0] == 3, b.shape == [3] else { throw NativeDecisionError.invalidWeights }
        let ordered = intervals.sorted { $0.lowerBound < $1.lowerBound }
        guard ordered[0].lowerBound == 0, ordered[0].upperBound == ordered[1].lowerBound,
              ordered[1].upperBound == raw.count - 8 - headerCount else { throw NativeDecisionError.invalidWeights }
        width = w.shape[1]
        weight = w.data
        bias = b.data
    }

    static func fp32(_ bf16: UInt16) -> Float { Float(bitPattern: UInt32(bf16) << 16) }

    static func bf16(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        // Preserve NaN instead of turning a tiny NaN payload into infinity.
        if bits & 0x7fffffff > 0x7f800000 { return UInt16(bits >> 16) | 0x40 }
        return UInt16(truncatingIfNeeded: (bits &+ 0x7fff &+ ((bits >> 16) & 1)) >> 16)
    }

    func pool(hidden: [Float], mask: [Float]) throws -> [Float] {
        guard !mask.isEmpty, mask.count <= Int.max / width, hidden.count == mask.count * width,
              hidden.allSatisfy(\.isFinite), mask.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
            throw NativeDecisionError.invalidInput
        }
        let count = mask.reduce(Float(0), +)
        guard count.isFinite, count > 0 else { throw NativeDecisionError.invalidInput }
        var pooled = [Float](repeating: 0, count: width)
        for token in mask.indices {
            for column in 0..<width { pooled[column] += hidden[token * width + column] * mask[token] }
        }
        for column in 0..<width { pooled[column] /= count }
        guard pooled.allSatisfy(\.isFinite) else { throw NativeDecisionError.invalidInput }
        return pooled
    }

    func logits(pooled: [Float]) throws -> [Float] {
        guard pooled.count == width, pooled.allSatisfy(\.isFinite) else { throw NativeDecisionError.invalidInput }
        var output = [Float](repeating: 0, count: 3)
        for action in 0..<3 {
            var sum: Float = 0
            for column in 0..<width { sum += pooled[column] * weight[action * width + column] }
            output[action] = sum + bias[action]
        }
        guard output.allSatisfy(\.isFinite) else { throw NativeDecisionError.invalidInput }
        return output
    }

    func choose(logits: [Float], policy: Policy) throws -> Result {
        let fallback = Result(route: "local", action: "local_core", source: "local_default", confidence: nil)
        guard policy.calibrated else { return fallback }
        guard logits.count == 3, logits.allSatisfy(\.isFinite) else { throw NativeDecisionError.invalidInput }
        // Mask before softmax; an unsupported expert must not steal probability mass.
        let scaled = zip(Self.actions, logits).map { policy.supportedActions.contains($0) ? $1 / policy.temperature : -Float.infinity }
        let largest = scaled.max()!
        guard largest.isFinite else { throw NativeDecisionError.invalidInput }
        let unnormalized = scaled.map { exp($0 - largest) }
        let denominator = unnormalized.reduce(Float(0), +)
        let probabilities = unnormalized.map { $0 / denominator }
        let index = probabilities.indices.max { probabilities[$0] < probabilities[$1] }!
        // PyTorch max returns the first index on a tie.
        let first = probabilities.firstIndex(of: probabilities[index])!
        let confidence = probabilities[first]
        guard Double(confidence) >= policy.threshold else { return fallback }
        let action = Self.actions[first]
        return Result(route: action == "teacher" ? "teacher" : "local", action: action,
                      source: "calibrated_head", confidence: confidence)
    }
}
