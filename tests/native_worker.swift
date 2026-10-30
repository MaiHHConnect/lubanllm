import Foundation

@main struct WorkerTests {
    static func main() throws {
        let root = URL(fileURLWithPath: CommandLine.arguments[1])
        let binary = URL(fileURLWithPath: CommandLine.arguments[2])
        let fm = FileManager.default
        let identity = String(repeating: "b", count: 64)
        let manifest = Data("fixture-manifest".utf8)
        var results: [[String: Any]] = []
        func make(_ mode: String, precision: String = "bf16", core: NativeWorker.Core? = nil, expert: NativeWorker.Expert? = nil) throws -> NativeWorker {
            let folder = root.appendingPathComponent(mode)
            try fm.createDirectory(at: folder, withIntermediateDirectories: false)
            try manifest.write(to: folder.appendingPathComponent("upstream-file-manifest.json"), options: .withoutOverwriting)
            return try NativeWorker(binary: binary, model: folder,
                manifestSHA: nativeSHA(folder.appendingPathComponent("upstream-file-manifest.json")), identity: identity,
                core: core, inferencePrecision: precision, expert: expert)
        }
        func start(_ worker: NativeWorker, _ name: String) throws {
            _ = try worker.start(log: root.appendingPathComponent(name + ".stderr"), timeout: 3)
        }
        func record(_ name: String, _ passed: Bool) throws {
            results.append(["check": name, "passed": passed])
            if !passed { throw WorkerFailure("Failed: " + name) }
        }
        let receipt: [String: Any] = ["binary_sha256": try nativeSHA(binary), "protocol": "luban-worker-jsonl-v1"]
        try JSONSerialization.data(withJSONObject: receipt).write(to: URL(fileURLWithPath: binary.path + ".json"), options: .withoutOverwriting)
        let normal = try make("normal")
        try start(normal, "normal")
        try record("default bf16 has no additional flag", normal.inferencePrecision == "bf16"
                   && normal.loaded?["inference_precision"] as? String == "bf16" && normal.loaded?["worker_args"] as? [String] == [])
        let firstPID = normal.pid
        let other = try make("duplicate")
        var lockRejected = false
        do { try start(other, "duplicate") } catch { lockRejected = true }
        try record("shared OS model lock rejects second worker", lockRejected && normal.pid == firstPID)
        let result = try normal.generate(tokens: [1, 2], maximum: 2, eos: [43])
        try record("stream matches complete result", result["tokens"] as? [Int] == [42, 43])
        _ = try normal.generate(tokens: [2], maximum: 2, eos: [43], seed: UInt64.max)
        try record("persistent worker reused", normal.pid == firstPID)
        var rejected = false
        do { _ = try normal.generate(tokens: [248320], maximum: 2, eos: [43]) } catch { rejected = true }
        try record("out of vocabulary request rejected before submission", rejected && normal.pid == firstPID)
        normal.close()
        try record("close releases child and model lock", normal.pid == 0)
        let legacy = try make("legacy-bf16")
        try start(legacy, "legacy-bf16")
        try record("legacy absent precision accepted only for bf16", legacy.loaded?["inference_precision"] == nil)
        legacy.close()
        for precision in ["q4", "BF16", ""] {
            rejected = false
            do { _ = try NativeWorker(binary: binary, model: root, manifestSHA: identity, identity: identity, inferencePrecision: precision) }
            catch { rejected = true }
            try record("invalid precision rejected before spawn: " + precision, rejected)
        }
        for (precision, modes) in [("bf16", ["force-mixed-bf16", "force-mlp-bf16", "force-q4-early-bf16", "force-q8-bf16", "bad-precision-bf16", "null-precision-bf16"]),
                                   ("mixed-q4", ["force-bf16-mixed", "force-mlp-mixed", "force-q8-mixed", "legacy-mixed", "bad-precision-mixed", "null-precision-mixed"]),
                                   ("mixed-q4-mlp", ["force-bf16-mlp", "force-mixed-mlp", "force-q8-mlp", "legacy-mlp", "bad-precision-mlp", "null-precision-mlp"]),
                                   ("mixed-q4-early", ["force-bf16-q4-early", "force-mixed-q4-early", "force-mlp-q4-early", "legacy-q4-early", "bad-precision-q4-early", "null-precision-q4-early"]),
                                   ("mixed-q8", ["force-bf16-q8", "force-mixed-q8", "force-mlp-q8", "legacy-q8", "bad-precision-q8", "null-precision-q8"])] {
            for mode in modes {
                let worker = try make(mode, precision: precision)
                var failure: String?
                do { try start(worker, mode) } catch { failure = String(describing: error) }
                try record(mode + " rejected and child reaped", failure?.contains("inference precision mismatch") == true && worker.pid == 0)
            }
        }
        let coreFile = root.appendingPathComponent("fixture-core.safetensors")
        try Data("fixture core".utf8).write(to: coreFile, options: .withoutOverwriting)
        let core = NativeWorker.Core(file: coreFile, tensorKey: "down.weight", sha256: try nativeSHA(coreFile))
        for precision in ["mixed-q4", "mixed-q4-g16", "mixed-q4-mlp", "mixed-q4-early", "mixed-q8"] {
            for boundCore in [nil, core] as [NativeWorker.Core?] {
                let mode = precision + (boundCore == nil ? "" : "-core")
                let worker = try make(mode, precision: precision, core: boundCore)
                try start(worker, mode)
                let expected = boundCore.map { [$0.file.path, $0.tensorKey, $0.sha256, "--" + precision] } ?? ["--" + precision]
                try record(mode + " flag follows all binding arguments", worker.loaded?["worker_args"] as? [String] == expected
                           && worker.loaded?["inference_precision"] as? String == precision)
                let completion = try worker.generate(tokens: [1], maximum: 2, eos: [43])
                try record(mode + " IPC completes without Python", completion["tokens"] as? [Int] == [42, 43])
                worker.close()
            }
        }
        let expertFile = root.appendingPathComponent("fixture-expert.safetensors")
        try Data("fixture expert".utf8).write(to: expertFile, options: .withoutOverwriting)
        let expert = NativeWorker.Expert(file: expertFile, sha256: try nativeSHA(expertFile), width: 128, coreFileSHA256: core.sha256)
        for mode in ["expert-normal", "expert-wrong-sha", "expert-wrong-width", "expert-wrong-layer", "expert-missing-ack"] {
            let worker = try make(mode, precision: "mixed-q4-g16", core: core, expert: expert)
            var failure: String?
            do { try start(worker, mode) } catch { failure = String(describing: error) }
            if mode == "expert-normal" {
                let expected = [core.file.path, core.tensorKey, core.sha256, "--mixed-q4-g16", "--expert-file=" + expert.file.path, "--expert-sha=" + expert.sha256]
                try record("immutable expert arguments and exact ACK", failure == nil && worker.loaded?["worker_args"] as? [String] == expected)
                _ = try worker.generate(tokens: [1], maximum: 2, eos: [43])
                worker.close()
            } else { try record(mode + " rejected and reaped", failure?.contains("Loaded expert changed") == true && worker.pid == 0) }
        }
        let unexpected = try make("expert-unexpected", precision: "mixed-q4-g16", core: core)
        rejected = false
        do { try start(unexpected, "expert-unexpected") } catch { rejected = true }
        try record("unrequested expert refused", rejected && unexpected.pid == 0)
        for (label, candidateCore, precision, width, boundSHA) in [
            ("no-core", nil, "mixed-q4-g16", 128, core.sha256),
            ("wrong-core", core, "mixed-q4-g16", 128, identity),
            ("wrong-mode", core, "mixed-q8", 128, core.sha256),
            ("bad-width", core, "mixed-q4-g16", 513, core.sha256)
        ] as [(String, NativeWorker.Core?, String, Int, String)] {
            rejected = false
            do { _ = try make("expert-init-" + label, precision: precision, core: candidateCore,
                expert: .init(file: expertFile, sha256: expert.sha256, width: width, coreFileSHA256: boundSHA)) } catch { rejected = true }
            try record("expert rejected before spawn: " + label, rejected)
        }
        let changedExpert = try make("expert-changed-file", precision: "mixed-q4-g16", core: core, expert: expert)
        try Data("changed expert".utf8).write(to: expertFile)
        rejected = false
        do { try start(changedExpert, "expert-changed-file") } catch { rejected = true }
        try record("changed expert source refused before spawn", rejected && changedExpert.pid == 0)
        let cancellation = try make("cancel")
        try start(cancellation, "cancel")
        let cancelled = try cancellation.generate(tokens: [1], maximum: 2, eos: [43]) { _ in cancellation.cancel() }
        try record("independent native cancellation pipe", cancelled["finish_reason"] as? String == "cancelled")
        cancellation.close()
        for mode in ["wrong-id", "boolean-token", "boolean-count", "fractional-count", "wrong-final", "oversize", "timeout", "crash"] {
            let worker = try make(mode)
            try start(worker, mode)
            var failure: String?
            do { _ = try worker.generate(tokens: [1], maximum: 2, eos: [43], timeout: mode == "timeout" ? 0.1 : 3) }
            catch { failure = String(describing: error) }
            try record(mode + " rejected and child reaped", failure != nil && worker.pid == 0)
        }
        let changed = try make("changed-manifest")
        try Data("tampered".utf8).write(to: changed.model.appendingPathComponent("upstream-file-manifest.json"))
        rejected = false
        do { try start(changed, "changed-manifest") } catch { rejected = true }
        try record("changed model manifest rejected before spawning", rejected && changed.pid == 0)
        let report: [String: Any] = ["scope": "Swift IPC protocol fixtures; no real model loaded", "passed": true,
                                   "python_child_invocations": 0, "checks": results]
        let bytes = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try bytes.write(to: root.appendingPathComponent("result.json"), options: .withoutOverwriting)
        FileHandle.standardOutput.write(bytes + Data([10]))
    }
}
