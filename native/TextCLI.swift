import Foundation
import Darwin

// Isolated candidate harness, never an installed replacement for the full application.
@main struct NativeTextCLI {
    static func output(_ value: [String: Any]) throws {
        let data = try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
        FileHandle.standardOutput.write(data + Data([10]))
    }
    static func integer(_ value: Any?, default fallback: Int, maximum: Int) throws -> Int {
        guard let value else { return fallback }
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
              number.doubleValue.isFinite, number.doubleValue == Double(number.intValue),
              (1...maximum).contains(number.intValue) else { throw WorkerFailure("Invalid integer") }
        return number.intValue
    }
    static func main() throws {
        guard CommandLine.arguments.count == 2 else { throw WorkerFailure("Expected candidate config JSON path") }
        let configURL = URL(fileURLWithPath: CommandLine.arguments[1])
        guard let config = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any],
              let binaryPath = config["binary"] as? String, let modelPath = config["model"] as? String,
              let manifest = config["manifest_sha256"] as? String, let identity = config["identity"] as? String,
              let logPath = config["log"] as? String else { throw WorkerFailure("Invalid candidate config") }
        var inferencePrecision = "bf16"
        if let value = config["inference_precision"] {
            guard let precision = value as? String, ["bf16", "mixed-q4", "mixed-q4-g16", "mixed-q4-g32-middle", "mixed-q4-g32-clip97", "mixed-q4-g32-calibrated", "mixed-q4-g32", "mixed-q4-g32-skip0-early8", "mixed-q4-g32-skip0", "mixed-q4-g32-early12", "mixed-q4-mlp", "mixed-q4-gateup", "mixed-q4-down", "mixed-q4-first1", "mixed-q4-first2", "mixed-q4-first4", "mixed-q4-first6", "mixed-q4-early", "mixed-q4-early8", "mixed-q4-early12", "mixed-q4-calibrated", "mixed-q8"].contains(precision) else {
                throw WorkerFailure("Invalid inference precision")
            }
            inferencePrecision = precision
        }
        var calibrationProfile: URL?
        if inferencePrecision == "mixed-q4-calibrated" || inferencePrecision == "mixed-q4-g32-calibrated" {
            guard let path = config["calibration_profile"] as? String else { throw WorkerFailure("Calibrated Q4 requires a profile path") }
            calibrationProfile = URL(fileURLWithPath: path)
        } else if config["calibration_profile"] != nil { throw WorkerFailure("Calibration profile is only valid for calibrated Q4") }
        var core: NativeWorker.Core?
        if config["core"] != nil {
            guard let bound = config["core"] as? [String: String] else { throw WorkerFailure("Invalid candidate core") }
            guard let file = bound["file"], let key = bound["tensor_key"], let sha = bound["sha256"] else {
                throw WorkerFailure("Invalid candidate core")
            }
            core = NativeWorker.Core(file: URL(fileURLWithPath: file), tensorKey: key, sha256: sha)
        }
        let startup = DispatchTime.now().uptimeNanoseconds
        let tokenizer = try QwenTokenizer(modelDirectory: URL(fileURLWithPath: modelPath))
        let worker = try NativeWorker(binary: URL(fileURLWithPath: binaryPath), model: URL(fileURLWithPath: modelPath),
                                      manifestSHA: manifest, identity: identity, core: core, inferencePrecision: inferencePrecision, calibrationProfile: calibrationProfile)
        defer { worker.close() }
        let loaded = try worker.start(log: URL(fileURLWithPath: logPath))
        try output(["event": "ready", "pid": getpid(), "worker_pid": worker.pid, "loaded": loaded,
                    "startup_ns": DispatchTime.now().uptimeNanoseconds - startup, "python_invocations": 0])
        while let line = readLine() {
            do {
                guard line.utf8.count <= 4 << 20,
                      let request = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                      let rawMessages = request["messages"] as? [[String: Any]], (1...100).contains(rawMessages.count),
                      rawMessages.allSatisfy({ ["system", "user", "assistant"].contains($0["role"] as? String ?? "")
                          && ($0["content"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false }) else {
                    throw WorkerFailure("Invalid product text messages")
                }
                let maximum = try integer(request["max_tokens"], default: 128, maximum: 2048)
                // Match validated_messages: extra message fields are not part of the product prompt.
                let messages = rawMessages.map { ["role": $0["role"]!, "content": $0["content"]!] }
                guard messages.reduce(0, { $0 + ($1["content"] as! String).unicodeScalars.count }) <= 1048576,
                      request["route"] == nil || ["auto", "local"].contains(request["route"] as? String ?? "") else {
                    throw WorkerFailure("Candidate supports bounded local text requests only")
                }
                var temperature: Double = 0
                if let value = request["temperature"] {
                    guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
                          number.doubleValue.isFinite, (0...2).contains(number.doubleValue) else {
                        throw WorkerFailure("Invalid temperature")
                    }
                    temperature = number.doubleValue
                }
                guard request["thinking"] == nil || request["thinking"] is String else { throw WorkerFailure("Invalid thinking setting") }
                let requestedThinking = request["thinking"] as? String ?? "off"
                guard ["off", "low", "medium", "high"].contains(requestedThinking) else { throw WorkerFailure("Invalid thinking setting") }
                // Existing adopted core was qualified only with the direct-answer template.
                let thinking = core == nil && requestedThinking != "off"
                let template = try tokenizer.chatTemplate(messages, enableThinking: thinking)
                let ids = try tokenizer.encode(template)
                var streamed = ""
                let decoder = QwenTokenizer.StreamDecoder(tokenizer)
                let began = DispatchTime.now().uptimeNanoseconds
                var first: UInt64?
                let result = try worker.generate(tokens: ids.map(UInt32.init), maximum: maximum,
                                                  eos: [248044], temperature: temperature, seed: 42) { token in
                    let delta = try decoder.append(Int(token))
                    streamed += delta
                    if !delta.isEmpty && first == nil { first = DispatchTime.now().uptimeNanoseconds - began }
                }
                streamed += decoder.finish()
                let tokens = result["tokens"] as! [Int]
                let text = try tokenizer.decode(tokens)
                guard text == streamed else { throw WorkerFailure("Streaming and full text decode mismatch") }
                var completion = result
                completion["event"] = "completion"
                completion["input_tokens"] = ids
                completion["text"] = text
                completion["streamed_text"] = streamed
                completion["wall_ns"] = DispatchTime.now().uptimeNanoseconds - began
                completion["visible_first_ns"] = first.map { $0 as Any } ?? NSNull()
                completion["thinking"] = thinking ? requestedThinking : "off"
                try output(completion)
            } catch {
                try output(["event": "candidate_error", "message": String(describing: error)])
                // A failed native worker is not silently replaced by Python.
                if worker.pid == 0 { throw error }
            }
        }
    }
}
