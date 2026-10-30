import Foundation
import CryptoKit

@main
struct DecisionChecks {
    struct Fixture: Decodable {
        let head: String
        let sha256: String
        let cases: [Case]
    }
    struct Case: Decodable {
        let id: String
        let hidden: [Float]
        let mask: [Float]
        let pooled: [Float]
        let logits: [Float]
        let policy: NativeDecision.Policy
        let expected: NativeDecision.Result
    }
    static func check(_ condition: Bool, _ message: String) throws {
        if !condition { throw NSError(domain: "NativeDecisionTest", code: 1,
                                      userInfo: [NSLocalizedDescriptionKey: message]) }
    }
    static func rejects(_ message: String, _ operation: () throws -> Void) throws {
        do { try operation() } catch { return }
        try check(false, "accepted invalid " + message)
    }
    static func main() throws {
        guard CommandLine.arguments.count == 2 else { fatalError("pass fixture.json") }
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf:
            URL(fileURLWithPath: CommandLine.arguments[1])))
        let head = try NativeDecision(file: URL(fileURLWithPath: fixture.head), approvedSHA256: fixture.sha256)
        var maxPoolError: Float = 0, maxLogitError: Float = 0, maxProbabilityError: Float = 0
        for row in fixture.cases {
            let pooled = try head.pool(hidden: row.hidden, mask: row.mask)
            maxPoolError = max(maxPoolError, zip(pooled, row.pooled).map { abs($0 - $1) }.max()!)
            let logits = try head.logits(pooled: pooled)
            let logitError = zip(logits, row.logits).map { abs($0 - $1) }.max()!
            maxLogitError = max(maxLogitError, logitError)
            try check(logitError < 0.0002, "CPU head logits differ: " + row.id)
            let result = try head.choose(logits: logits, policy: row.policy)
            try check(result.action == row.expected.action && result.route == row.expected.route && result.source == row.expected.source,
                      "route differs: " + row.id)
            if let expected = row.expected.confidence {
                try check(result.confidence != nil, "missing confidence")
                let error = abs(result.confidence! - expected)
                maxProbabilityError = max(maxProbabilityError, error)
                try check(error < 0.00002, "probability differs: " + row.id)
            } else { try check(result.confidence == nil, "fallback exposes wrong confidence") }
        }
        // These cases detect last-token pooling, BF16 pooling, unmasked softmax,
        // strict > threshold, unstable softmax and a non-first tie implementation.
        let supported = try NativeDecision.Policy(calibrated: true, supportedActions: ["local_core", "teacher"], threshold: 0)
        let masked = try head.choose(logits: [1, 100, 0], policy: supported)
        try check(masked.action == "local_core", "unsupported expert wins")
        let tied = try head.choose(logits: [10000, 10000, 10000], policy:
            NativeDecision.Policy(calibrated: true, supportedActions: NativeDecision.actions, threshold: 0))
        try check(tied.action == "local_core", "tie or large-logit softmax failed")
        let boundary = try head.choose(logits: [0, 100, 0], policy:
            NativeDecision.Policy(calibrated: true, threshold: 0.5))
        try check(boundary.source == "calibrated_head" && boundary.confidence == 0.5, "threshold equality failed")
        let sentinel = try head.choose(logits: [100, 0, 0], policy:
            NativeDecision.Policy(calibrated: true, threshold: Double(1).nextUp))
        try check(sentinel.source == "local_default", "Float64 rejection threshold rounded to 1")
        let low = try head.choose(logits: [0, 0, 0], policy: NativeDecision.Policy(calibrated: true))
        try check(low.source == "local_default" && low.confidence == nil, "low-confidence fallback failed")
        let untrained = try head.choose(logits: [.nan], policy: NativeDecision.Policy(calibrated: false))
        try check(untrained.source == "local_default", "untrained policy evaluated logits")
        let one = [Float](repeating: 1, count: head.width)
        let three = [Float](repeating: 3, count: head.width)
        let huge = [Float](repeating: 999, count: head.width)
        try check(try head.pool(hidden: one + three + huge, mask: [1, 1, 0]) == [Float](repeating: 2, count: head.width),
                  "mean pooling became last-token or includes padding")
        try check(try head.pool(hidden: one + three, mask: [1, 3]) == [Float](repeating: 2.5, count: head.width),
                  "mask weights or denominator incorrect")
        let bf = [UInt16](repeating: 0x3f80, count: head.width) + [UInt16](repeating: 0x3f81, count: head.width)
        let bfPool = try head.pool(hidden: bf.map(NativeDecision.fp32), mask: [1, 1])
        try check(bfPool[0] == 1.00390625 && bfPool[0] != NativeDecision.fp32(NativeDecision.bf16(bfPool[0])),
                  "pooled features rounded back to BF16")
        try check(NativeDecision.bf16(1.00390625) == 0x3f80 && NativeDecision.bf16(1.01171875) == 0x3f82,
                  "BF16 ties-to-even failed")
        try check(NativeDecision.fp32(NativeDecision.bf16(.nan)).isNaN, "NaN converted to infinity")
        try rejects("all-masked input") { _ = try head.pool(hidden: one, mask: [0]) }
        try rejects("shape mismatch") { _ = try head.pool(hidden: [], mask: [1]) }
        try rejects("nonfinite input") { _ = try head.pool(hidden: [.nan] + one.dropFirst(), mask: [1]) }
        try rejects("zero temperature") { _ = try NativeDecision.Policy(calibrated: true, temperature: 0) }
        try rejects("unknown action") { _ = try NativeDecision.Policy(calibrated: true, supportedActions: ["future"]) }
        try rejects("wrong hash") { _ = try NativeDecision(file: URL(fileURLWithPath: fixture.head), approvedSHA256: String(repeating: "0", count: 64)) }
        func invalidFile(_ label: String, _ raw: Data) throws {
            let path = FileManager.default.temporaryDirectory.appendingPathComponent("luban-decision-" + UUID().uuidString)
            try raw.write(to: path)
            defer { try? FileManager.default.removeItem(at: path) }
            let sha = SHA256.hash(data: raw).map { String(format: "%02x", $0) }.joined()
            try rejects(label) { _ = try NativeDecision(file: path, approvedSHA256: sha) }
        }
        let raw = try Data(contentsOf: URL(fileURLWithPath: fixture.head))
        try invalidFile("oversized header", Data(repeating: 0xff, count: 8))
        try invalidFile("truncated tensor", raw.dropLast())
        let headerCount = raw.withUnsafeBytes { Int(UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self))) }
        var nonfinite = raw
        nonfinite.replaceSubrange((8 + headerCount)..<(12 + headerCount), with: [0, 0, 0x80, 0x7f])
        try invalidFile("nonfinite weights", nonfinite)
        var header = try JSONSerialization.jsonObject(with: raw.subdata(in: 8..<(8 + headerCount))) as! [String: Any]
        func changedHeader(_ entry: [String: Any]) throws -> Data {
            var altered = header
            altered["projection.weight"] = entry
            let bytes = try JSONSerialization.data(withJSONObject: altered, options: [.sortedKeys])
            var size = UInt64(bytes.count).littleEndian
            var out = withUnsafeBytes(of: &size) { Data($0) }
            out.append(bytes)
            out.append(raw[(8 + headerCount)...])
            return out
        }
        var entry = header["projection.weight"] as! [String: Any]
        entry["dtype"] = "BF16"
        try invalidFile("BF16 head weights in FP32 head", changedHeader(entry))
        entry = header["projection.weight"] as! [String: Any]
        entry["shape"] = [1, 3, head.width]
        try invalidFile("unexpected rank", changedHeader(entry))
        header["unexpected.tensor"] = header["projection.bias"]
        try invalidFile("extra tensor", changedHeader(header["projection.weight"] as! [String: Any]))
        let output: [String: Any] = ["passed": true, "fixture_cases": fixture.cases.count, "counterexample_checks": 23,
            "max_pool_error": maxPoolError, "max_logit_error": maxLogitError,
            "max_probability_error": maxProbabilityError, "width": head.width]
        let resultJSON = try JSONSerialization.data(withJSONObject: output, options: [.prettyPrinted, .sortedKeys])
        print(String(decoding: resultJSON, as: UTF8.self))
    }
}
