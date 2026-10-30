import Foundation
import Darwin

// Closed-book ordinary-Q4 observation; reuse the product tokenizer and worker.
@main struct EvolutionObserve {
    static func emit(_ value: [String: Any]) throws {
        FileHandle.standardOutput.write(try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]) + Data([10]))
    }
    static func main() throws {
        guard CommandLine.arguments.count == 2,
              let config = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))) as? [String: Any],
              let binary = config["binary"] as? String, let model = config["model"] as? String,
              let manifest = config["manifest_sha256"] as? String, let identity = config["identity"] as? String,
              let log = config["log"] as? String else { throw WorkerFailure("Invalid Q4 observation configuration") }
        var core: NativeWorker.Core?
        if let value = config["core"] {
            guard let bound = value as? [String: String], let file = bound["file"],
                  let key = bound["tensor_key"], let sha = bound["sha256"] else { throw WorkerFailure("Invalid Q4 core binding") }
            core = NativeWorker.Core(file: URL(fileURLWithPath: file), tensorKey: key, sha256: sha)
        }
        var expert: NativeWorker.Expert?
        if let value = config["expert"] {
            guard let bound = value as? [String: Any], let file = bound["file"] as? String,
                  let sha = bound["sha256"] as? String, let width = bound["width"] as? NSNumber,
                  CFGetTypeID(width) != CFBooleanGetTypeID(), width.doubleValue == Double(width.intValue),
                  let coreSHA = bound["core_file_sha256"] as? String, let layer = bound["layer"] as? NSNumber,
                  CFGetTypeID(layer) != CFBooleanGetTypeID(), layer.doubleValue == 23,
                  let version = bound["core_version"] as? String, !version.isEmpty else { throw WorkerFailure("Invalid Q4 expert binding") }
            expert = NativeWorker.Expert(file: URL(fileURLWithPath: file), sha256: sha, width: width.intValue, coreFileSHA256: coreSHA)
        }
        let directory = URL(fileURLWithPath: model)
        let modelConfig = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("config.json"))) as? [String: Any]
        let textConfig = modelConfig?["text_config"] as? [String: Any]
        // This worker/tokenizer pair is qualified only for this pinned Qwen vocabulary and EOS.
        guard let eos = textConfig?["eos_token_id"] as? NSNumber,
              CFGetTypeID(eos) != CFBooleanGetTypeID(), eos.intValue == 248044 else { throw WorkerFailure("Unsupported Q4 model EOS") }
        let tokenizer = try QwenTokenizer(modelDirectory: directory)
        let worker = try NativeWorker(binary: URL(fileURLWithPath: binary), model: directory,
                                      manifestSHA: manifest, identity: identity, core: core, inferencePrecision: "mixed-q4-g16", traceDirectory: (config["trace_directory"] as? String).map { URL(fileURLWithPath: $0) }, expert: expert)
        defer { worker.close() }
        signal(SIGUSR1, SIG_IGN); signal(SIGTERM, SIG_IGN)
        let cancellation = DispatchSource.makeSignalSource(signal: SIGUSR1, queue: .global())
        cancellation.setEventHandler { worker.cancel() }; cancellation.resume()
        let termination = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .global())
        termination.setEventHandler { worker.close(); exit(0) }; termination.resume()
        defer { cancellation.cancel(); termination.cancel() }
        let loaded = try worker.start(log: URL(fileURLWithPath: log))
        try emit(["event": "ready", "pid": getpid(), "worker_pid": worker.pid, "loaded": loaded, "eos": eos.intValue])
        while let line = readLine() {
            guard line.utf8.count <= 4 << 20 else { throw WorkerFailure("Q4 request too large") }
            if let trace = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any], trace["operation"] as? String == "trace" {
                guard let id = trace["id"] as? String, let ids = trace["input_tokens"] as? [UInt32], let target = trace["tokens"] as? [UInt32], target.last == UInt32(eos.intValue), (1...256).contains(target.count) else { throw WorkerFailure("Invalid Q4 calibration history") }
                let result = try worker.generate(tokens: ids, maximum: target.count, eos: [UInt32(eos.intValue)], forcedTokens: Array(target.dropLast()), onTrace: { event in
                    var frame = event; frame["request_id"] = id; try emit(frame)
                })
                guard result["tokens"] as? [UInt32] == target else { throw WorkerFailure("Q4 calibration history changed") }
                try emit(["event": "trace_complete", "id": id, "worker_pid": worker.pid, "positions": target.count]); continue
            }
            guard
                  let request = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                  let id = request["id"] as? String, !id.isEmpty,
                  let rows = request["messages"] as? [[String: Any]], (1...100).contains(rows.count),
                  rows.allSatisfy({ ["system", "user", "assistant"].contains($0["role"] as? String ?? "")
                      && ($0["content"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false }),
                  let maximum = request["max_tokens"] as? NSNumber, CFGetTypeID(maximum) != CFBooleanGetTypeID(),
                  maximum.doubleValue == Double(maximum.intValue), (1...2048).contains(maximum.intValue)
            else { throw WorkerFailure("Invalid closed-book Q4 observation") }
            let messages = rows.map { ["role": $0["role"]!, "content": $0["content"]!] }
            guard messages.reduce(0, { $0 + ($1["content"] as! String).unicodeScalars.count }) <= 1048576 else { throw WorkerFailure("Q4 context exceeds limit") }
            let ids = try tokenizer.encode(tokenizer.chatTemplate(messages, enableThinking: false))
            let result = try worker.generate(tokens: ids.map(UInt32.init), maximum: maximum.intValue,
                                              eos: [UInt32(eos.intValue)], temperature: 0, seed: 42, onToken: { _ in })
            guard let tokens = result["tokens"] as? [Int] else { throw WorkerFailure("Q4 output tokens missing") }
            try emit(["event": "observation", "id": id, "input_tokens": ids, "tokens": tokens,
                      "answer": try tokenizer.decode(tokens), "finish_reason": result["finish_reason"]!,
                      "worker_pid": worker.pid, "seed": 42, "temperature": 0, "thinking": "off",
                      "branch_enabled": expert != nil, "reference_used": false])
        }
        worker.close()
        try emit(["event": "closed", "pid": getpid()])
    }
}
