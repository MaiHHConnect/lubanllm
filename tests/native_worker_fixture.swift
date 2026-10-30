import Foundation
import Darwin

// Protocol fixture only: this is not a model or an inference quality test.
@main struct FixtureWorker {
    static func emit(_ value: [String: Any]) {
        let data = try! JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
        FileHandle.standardOutput.write(data + Data([10]))
    }
    static func main() throws {
        let args = CommandLine.arguments
        let mode = URL(fileURLWithPath: args[1]).lastPathComponent
        let identity = String(repeating: "b", count: 64)
        var lockInfo = stat()
        guard fstat(4, &lockInfo) == 0, lockInfo.st_mode & S_IFMT == S_IFREG else { exit(4) }
        guard ProcessInfo.processInfo.environment.keys.allSatisfy({ !$0.hasPrefix("PYTHON") && $0 != "VIRTUAL_ENV" }) else { exit(5) }
        let workerArgs = Array(args.dropFirst(4))
        let plainArgs = workerArgs.filter { !$0.hasPrefix("--expert-") }
        let flag = plainArgs.last ?? ""
        let precision = ["--mixed-q4", "--mixed-q4-g16", "--mixed-q4-mlp", "--mixed-q4-early", "--mixed-q8"].contains(flag) ? String(flag.dropFirst(2)) : "bf16"
        let coreArgs = precision != "bf16" ? Array(plainArgs.dropLast()) : plainArgs
        var loaded: [String: Any] = ["event": "loaded", "protocol": "luban-worker-jsonl-v1", "manifest_sha256": args[2],
                                    "base_identity": identity, "core_loaded": coreArgs.count == 3, "worker_args": workerArgs]
        if coreArgs.count == 3 { loaded["core_file_sha256"] = coreArgs[2] }
        if !mode.hasPrefix("legacy-") {
            if mode.hasPrefix("force-bf16-") { loaded["inference_precision"] = "bf16" }
            else if mode.hasPrefix("force-mlp-") { loaded["inference_precision"] = "mixed-q4-mlp" }
            else if mode.hasPrefix("force-q4-early-") { loaded["inference_precision"] = "mixed-q4-early" }
            else if mode.hasPrefix("force-q8-") { loaded["inference_precision"] = "mixed-q8" }
            else if mode.hasPrefix("force-mixed-") { loaded["inference_precision"] = "mixed-q4" }
            else { loaded["inference_precision"] = precision }
            if mode.hasPrefix("bad-precision-") { loaded["inference_precision"] = "fp32" }
            if mode.hasPrefix("null-precision-") { loaded["inference_precision"] = NSNull() }
        }
        if let expertSHA = workerArgs.first(where: { $0.hasPrefix("--expert-sha=") }) {
            loaded["expert_loaded"] = true
            loaded["expert_file_sha256"] = String(expertSHA.dropFirst("--expert-sha=".count))
            loaded["expert_width"] = 128
            loaded["expert_layer"] = 23
            if mode == "expert-wrong-sha" { loaded["expert_file_sha256"] = identity }
            if mode == "expert-wrong-width" { loaded["expert_width"] = 64 }
            if mode == "expert-wrong-layer" { loaded["expert_layer"] = 22 }
            if mode == "expert-missing-ack" { loaded.removeValue(forKey: "expert_loaded") }
        } else if mode == "expert-unexpected" { loaded["expert_loaded"] = true }
        emit(loaded)
        while let line = readLine() {
            let request = try JSONSerialization.jsonObject(with: Data(line.utf8)) as! [String: Any]
            let id = request["id"] as! String
            if mode == "wrong-id" { emit(["event": "token", "id": "wrong", "token": 42]); continue }
            if mode == "boolean-token" { emit(["event": "token", "id": id, "token": true]); continue }
            if mode == "oversize" { FileHandle.standardOutput.write(Data(repeating: 65, count: (4 << 20) + 1) + Data([10])); continue }
            if mode == "timeout" { continue }
            if mode == "crash" { exit(7) }
            if mode == "boolean-count" {
                emit(["event": "token", "id": id, "token": 42])
                emit(["event": "result", "id": id, "tokens": [42], "generated_tokens": true, "finish_reason": "length"])
                continue
            }
            var cancelled = false
            for token in [42, 43] {
                emit(["event": "token", "id": id, "token": token])
                if mode == "cancel" {
                    var fd = pollfd(fd: 3, events: Int16(POLLIN), revents: 0)
                    if poll(&fd, 1, 1000) > 0 {
                        var byte: UInt8 = 0
                        if Darwin.read(3, &byte, 1) > 0 { cancelled = true; break }
                    }
                }
            }
            let tokens = cancelled ? [42] : mode == "wrong-final" ? [42, 44] : [42, 43]
            emit(["event": "result", "id": id, "tokens": tokens, "generated_tokens": mode == "fractional-count" ? 2.5 : Double(tokens.count),
                  "finish_reason": cancelled ? "cancelled" : "stop"])
        }
    }
}
