import Foundation
import CryptoKit
import Darwin

// Native candidate bridge. Production is switched only after the full parity gate.
struct WorkerFailure: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}

func nativeSHA(_ url: URL) throws -> String {
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    var hash = SHA256()
    while let bytes = try file.read(upToCount: 1 << 20), !bytes.isEmpty { hash.update(data: bytes) }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
}

private func isDigest(_ value: String) -> Bool {
    value.utf8.count == 64 && value.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
}

private func monotonic() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

final class NativeWorker {
    struct Core {
        let file: URL
        let tensorKey: String
        let sha256: String
    }
    // Construct only after the caller has checked independent route scope and
    // expert metadata against this accepted core. Immutable for this worker.
    struct Expert {
        let file: URL
        let sha256: String
        let width: Int
        let coreFileSHA256: String
    }
    let expert: Expert?
    let binary: URL
    let model: URL
    let manifestSHA: String
    let identity: String
    let core: Core?
    let inferencePrecision: String
    let calibrationProfile: URL?
    let traceDirectory: URL?
    private(set) var pid: pid_t = 0
    private(set) var loaded: [String: Any]?
    private let serial = NSRecursiveLock()
    private let lifecycle = NSLock()
    private var input: Int32 = -1
    private var output: Int32 = -1
    private var cancelInput: Int32 = -1
    private var cancelOutput: Int32 = -1
    private var modelLock: Int32 = -1
    private var pending = Data()
    private var verifiedBinarySHA: String?
    private var ordinal = 0
    private var requestActive = false
    private var cancellationDeadline: Double?

    init(binary: URL, model: URL, manifestSHA: String, identity: String, core: Core? = nil,
         inferencePrecision: String = "bf16", calibrationProfile: URL? = nil, traceDirectory: URL? = nil, expert: Expert? = nil) throws {
        guard isDigest(manifestSHA), isDigest(identity) else { throw WorkerFailure("Invalid model binding") }
        guard ["bf16", "mixed-q4", "mixed-q4-g16", "mixed-q4-g32-middle", "mixed-q4-g32-clip97", "mixed-q4-g32-calibrated", "mixed-q4-g32", "mixed-q4-g32-skip0-early8", "mixed-q4-g32-skip0", "mixed-q4-g32-early12", "mixed-q4-mlp", "mixed-q4-gateup", "mixed-q4-down", "mixed-q4-first1", "mixed-q4-first2", "mixed-q4-first4", "mixed-q4-first6", "mixed-q4-early", "mixed-q4-early8", "mixed-q4-early12", "mixed-q4-calibrated", "mixed-q8"].contains(inferencePrecision) else { throw WorkerFailure("Invalid inference precision") }
        if inferencePrecision == "mixed-q4-calibrated" || inferencePrecision == "mixed-q4-g32-calibrated" {
            guard let calibrationProfile, calibrationProfile.isFileURL, calibrationProfile.path.hasPrefix("/"), FileManager.default.isReadableFile(atPath: calibrationProfile.path) else { throw WorkerFailure("Calibrated Q4 requires a readable absolute profile") }
        } else if calibrationProfile != nil {
            throw WorkerFailure("Calibration profile is only valid for calibrated Q4")
        }
        if let traceDirectory {
            guard inferencePrecision == "mixed-q4-g16", traceDirectory.isFileURL, traceDirectory.path.hasPrefix("/"), FileManager.default.fileExists(atPath: traceDirectory.path) else { throw WorkerFailure("Invalid Q4 trace directory") }
        }
        if let expert {
            guard let core, core.sha256 == expert.coreFileSHA256, isDigest(expert.sha256),
                  (32...512).contains(expert.width), expert.file.isFileURL, expert.file.path.hasPrefix("/"),
                  inferencePrecision == "mixed-q4-g16", traceDirectory == nil else {
                throw WorkerFailure("Expert requires qualified ordinary Q4 and the bound accepted core")
            }
        }
        self.expert = expert
        self.traceDirectory = traceDirectory
        self.binary = binary; self.model = model; self.manifestSHA = manifestSHA
        self.identity = identity; self.core = core
        self.inferencePrecision = inferencePrecision; self.calibrationProfile = calibrationProfile
    }
    deinit { close() }

    private func verify() throws {
        guard try nativeSHA(model.appendingPathComponent("upstream-file-manifest.json")) == manifestSHA else {
            throw WorkerFailure("Model manifest changed")
        }
        let receipt = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: binary.path + ".json"))) as? [String: Any]
        guard let sha = receipt?["binary_sha256"] as? String, isDigest(sha), try nativeSHA(binary) == sha,
              receipt?["protocol"] as? String == "luban-worker-jsonl-v1" else {
            throw WorkerFailure("Native binary receipt mismatch")
        }
        verifiedBinarySHA = sha
        if let expert, try nativeSHA(expert.file) != expert.sha256 { throw WorkerFailure("Approved expert binding changed") }
        if let core {
            guard !core.tensorKey.isEmpty, isDigest(core.sha256), try nativeSHA(core.file) == core.sha256 else {
                throw WorkerFailure("Approved core binding changed")
            }
        }
    }

    private func acquireModelLock() throws {
        let directory = "/tmp/luban-model-locks-\(getuid())"
        if mkdir(directory, 0o700) != 0 && errno != EEXIST { throw WorkerFailure("Model lock directory failed") }
        var info = stat()
        guard lstat(directory, &info) == 0, info.st_mode & S_IFMT == S_IFDIR,
              info.st_uid == getuid(), info.st_mode & 0o077 == 0 else { throw WorkerFailure("Unsafe model lock directory") }
        let fd = Darwin.open(directory + "/" + identity + "-mps.lock", O_RDWR | O_CREAT | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw WorkerFailure("Model lock open failed") }
        guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFREG, info.st_uid == getuid(),
              info.st_nlink == 1, info.st_mode & 0o077 == 0, flock(fd, LOCK_EX | LOCK_NB) == 0 else {
            Darwin.close(fd); throw WorkerFailure("Model is already running or lock is unsafe")
        }
        modelLock = fd
    }

    // Allocate above our child descriptor targets so dup2 cannot clobber a source.
    private func highPipe() throws -> [Int32] {
        var fds: [Int32] = [-1, -1]
        guard pipe(&fds) == 0 else { throw WorkerFailure("Pipe failed") }
        let read = fcntl(fds[0], F_DUPFD_CLOEXEC, 10)
        let write = fcntl(fds[1], F_DUPFD_CLOEXEC, 10)
        Darwin.close(fds[0]); Darwin.close(fds[1])
        guard read >= 0, write >= 0 else {
            if read >= 0 { Darwin.close(read) }; if write >= 0 { Darwin.close(write) }
            throw WorkerFailure("Pipe duplication failed")
        }
        return [read, write]
    }

    func start(log: URL, timeout: Double = 180) throws -> [String: Any] {
        serial.lock(); defer { serial.unlock() }
        guard timeout.isFinite, timeout > 0 else { throw WorkerFailure("Invalid timeout") }
        if let loaded { return loaded }
        do {
            try verify()
            try acquireModelLock()
            let readLock = fcntl(modelLock, F_DUPFD_CLOEXEC, 10)
            guard readLock >= 0 else { throw WorkerFailure("Lock duplication failed") }
            defer { Darwin.close(readLock) }
            var temporary: [Int32] = []
            defer { temporary.forEach { Darwin.close($0) } }
            let stdinPipe = try highPipe(); temporary += stdinPipe
            let stdoutPipe = try highPipe(); temporary += stdoutPipe
            let cancelPipe = try highPipe(); temporary += cancelPipe
            let errorFD = Darwin.open(log.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
            guard errorFD >= 0 else { throw WorkerFailure("Fresh private worker log required") }
            defer { Darwin.close(errorFD) }
            let errorHigh = fcntl(errorFD, F_DUPFD_CLOEXEC, 10)
            guard errorHigh >= 0 else { throw WorkerFailure("Log duplication failed") }
            defer { Darwin.close(errorHigh) }
            var actions: posix_spawn_file_actions_t?
            guard posix_spawn_file_actions_init(&actions) == 0 else { throw WorkerFailure("Spawn actions failed") }
            defer { posix_spawn_file_actions_destroy(&actions) }
            for (source, target) in [(stdinPipe[0], Int32(0)), (stdoutPipe[1], 1), (errorHigh, 2), (cancelPipe[0], 3), (readLock, 4)] {
                guard posix_spawn_file_actions_adddup2(&actions, source, target) == 0 else { throw WorkerFailure("Spawn fd binding failed") }
            }
            for fd in temporary + [readLock, errorHigh] {
                guard posix_spawn_file_actions_addclose(&actions, fd) == 0 else { throw WorkerFailure("Spawn fd close failed") }
            }
            var attrs: posix_spawnattr_t?
            guard posix_spawnattr_init(&attrs) == 0 else { throw WorkerFailure("Spawn attributes failed") }
            defer { posix_spawnattr_destroy(&attrs) }
            guard posix_spawnattr_setflags(&attrs, Int16(POSIX_SPAWN_CLOEXEC_DEFAULT)) == 0 else { throw WorkerFailure("Spawn inheritance failed") }
            var args = [binary.path, model.path, manifestSHA, "3"]
            if let core { args += [core.file.path, core.tensorKey, core.sha256] }
            if inferencePrecision == "mixed-q4-calibrated" || inferencePrecision == "mixed-q4-g32-calibrated" {
                args.append((inferencePrecision == "mixed-q4-g32-calibrated" ? "--mixed-q4-g32-calibrated=" : "--mixed-q4-calibrated=") + calibrationProfile!.path)
            } else if inferencePrecision != "bf16" { args.append("--" + inferencePrecision) }
            if let traceDirectory { args.append("--research-trace=" + traceDirectory.path) }
            if let expert { args += ["--expert-file=" + expert.file.path, "--expert-sha=" + expert.sha256] }
            var cArgs = args.map { strdup($0) } + [nil]
            let environment = ProcessInfo.processInfo.environment.filter {
                !$0.key.hasPrefix("PYTHON") && !["VIRTUAL_ENV", "CONDA_PREFIX", "DYLD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES"].contains($0.key)
            }
            var cEnv = environment.sorted { $0.key < $1.key }.map { strdup($0.key + "=" + $0.value) } + [nil]
            defer { cArgs.compactMap { $0 }.forEach { free($0) }; cEnv.compactMap { $0 }.forEach { free($0) } }
            var child: pid_t = 0
            let result = posix_spawn(&child, binary.path, &actions, &attrs, &cArgs, &cEnv)
            guard result == 0 else { throw WorkerFailure("Native spawn failed: \(result)") }
            lifecycle.lock()
            pid = child; input = stdinPipe[1]; output = stdoutPipe[0]
            cancelInput = cancelPipe[0]; cancelOutput = cancelPipe[1]
            lifecycle.unlock()
            temporary.removeAll { [input, output, cancelInput, cancelOutput].contains($0) }
            for fd in [input, output, cancelInput, cancelOutput] { _ = fcntl(fd, F_SETFL, O_NONBLOCK) }
            for fd in [input, cancelOutput] { _ = fcntl(fd, F_SETNOSIGPIPE, 1) }
            var event = try nextEvent(deadline: monotonic() + timeout)
            guard event["event"] as? String == "loaded", event["protocol"] as? String == "luban-worker-jsonl-v1",
                  event["manifest_sha256"] as? String == manifestSHA, event["base_identity"] as? String == identity,
                  event["core_loaded"] as? Bool == (core != nil) else { throw WorkerFailure("Native load acknowledgment mismatch") }
            if let core, event["core_file_sha256"] as? String != core.sha256 { throw WorkerFailure("Loaded core changed") }
            if let expert {
                guard event["expert_loaded"] as? Bool == true, event["expert_file_sha256"] as? String == expert.sha256,
                      event["expert_width"] as? Int == expert.width, event["expert_layer"] as? Int == 23 else {
                    throw WorkerFailure("Loaded expert changed")
                }
            } else if event["expert_loaded"] as? Bool == true { throw WorkerFailure("Unexpected native expert") }
            let precision: String? = event["inference_precision"] == nil ? "bf16" : event["inference_precision"] as? String
            guard precision == inferencePrecision else { throw WorkerFailure("Native worker inference precision mismatch") }
            event["worker_binary_sha256"] = verifiedBinarySHA
            loaded = event
            return event
        } catch { close(); throw error }
    }

    private func nextEvent(deadline: Double) throws -> [String: Any] {
        while monotonic() < effectiveDeadline(deadline) {
            if let end = pending.firstIndex(of: 10) {
                guard end <= 4 << 20 else { throw WorkerFailure("Worker frame too large") }
                let line = pending.prefix(upTo: end)
                pending.removeSubrange(...end)
                guard let value = try JSONSerialization.jsonObject(with: line) as? [String: Any], value["event"] is String else {
                    throw WorkerFailure("Invalid worker event")
                }
                if value["event"] as? String == "error" { throw WorkerFailure("Worker error: \(value["error"] ?? value["message"] ?? "unknown")") }
                return value
            }
            guard pending.count <= 4 << 20 else { throw WorkerFailure("Worker frame too large") }
            var pollFD = pollfd(fd: output, events: Int16(POLLIN), revents: 0)
            let status = poll(&pollFD, 1, 50)
            if status < 0 { if errno == EINTR { continue }; throw WorkerFailure("Worker poll failed") }
            if status == 0 { continue }
            var bytes = [UInt8](repeating: 0, count: 65536)
            let length = Darwin.read(output, &bytes, bytes.count)
            if length < 0 && [EINTR, EAGAIN].contains(errno) { continue }
            guard length > 0 else { throw WorkerFailure("Worker exited without completion") }
            pending.append(contentsOf: bytes.prefix(length))
        }
        throw WorkerFailure("Native response timed out")
    }

    private func effectiveDeadline(_ deadline: Double) -> Double {
        lifecycle.lock(); defer { lifecycle.unlock() }
        return min(deadline, cancellationDeadline ?? deadline)
    }

    func generate(tokens: [UInt32], maximum: Int, eos: [UInt32], temperature: Double = 0, seed: UInt64 = 42,
                  timeout: Double = 180, forcedTokens: [UInt32]? = nil, onTrace: (([String: Any]) throws -> Void)? = nil, onToken: (UInt32) throws -> Void = { _ in }) throws -> [String: Any] {
        serial.lock(); defer { serial.unlock() }
        guard loaded != nil, !tokens.isEmpty, !eos.isEmpty, eos.count <= 64, (1...2048).contains(maximum),
              tokens.count + maximum <= 32768, temperature.isFinite, (0...2).contains(temperature),
              timeout.isFinite, timeout > 0, (tokens + eos).allSatisfy({ $0 < 248320 }) else { throw WorkerFailure("Invalid native generation request") }
        if let forcedTokens {
            guard traceDirectory != nil, onTrace != nil, forcedTokens.count + 1 == maximum, maximum <= 256, forcedTokens.allSatisfy({ $0 < 248320 && !eos.contains($0) }), temperature == 0, seed == 42 else { throw WorkerFailure("Invalid fixed Q4 trace history") }
        } else if onTrace != nil { throw WorkerFailure("Trace callback requires fixed history") }
        do {
            var drain = [UInt8](repeating: 0, count: 256)
            while Darwin.read(cancelInput, &drain, drain.count) > 0 {}
            lifecycle.lock(); requestActive = true; cancellationDeadline = nil; lifecycle.unlock()
            defer {
                lifecycle.lock(); requestActive = false; cancellationDeadline = nil; lifecycle.unlock()
            }
            ordinal += 1
            let id = String(ordinal)
            var request: [String: Any] = ["id": id, "tokens": tokens, "max_tokens": maximum,
                                          "eos_ids": eos, "temperature": temperature, "seed": seed]
            if let forcedTokens { request["forced_tokens"] = forcedTokens; request["tail_only"] = true; request["cache_reuse"] = false }
            var payload = try JSONSerialization.data(withJSONObject: request)
            payload.append(10)
            let deadline = monotonic() + timeout
            try payload.withUnsafeBytes { raw in
                var at = 0
                while at < raw.count {
                    guard monotonic() < effectiveDeadline(deadline) else { throw WorkerFailure("Native request write timed out") }
                    let count = Darwin.write(input, raw.baseAddress!.advanced(by: at), raw.count - at)
                    if count < 0 && [EINTR, EAGAIN].contains(errno) {
                        var fd = pollfd(fd: input, events: Int16(POLLOUT), revents: 0)
                        _ = poll(&fd, 1, 50); continue
                    }
                    guard count > 0 else { throw WorkerFailure("Worker request pipe failed") }
                    at += count
                }
            }
            var emitted: [UInt32] = []
            var traceCount = 0
            while true {
                let event = try nextEvent(deadline: deadline)
                guard event["id"] as? String == id else { throw WorkerFailure("Worker request identity mismatch") }
                switch event["event"] as? String {
                case "trace":
                    guard forcedTokens != nil, let onTrace, event["step"] as? Int == traceCount, traceCount < maximum, event["tail_only"] as? Bool == true else { throw WorkerFailure("Unexpected Q4 trace event") }
                    try onTrace(event); traceCount += 1
                case "token":
                    guard let number = event["token"] as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
                          number.doubleValue == Double(number.uint32Value), number.uint32Value < 248320,
                          emitted.count < maximum else { throw WorkerFailure("Invalid worker output token") }
                    emitted.append(number.uint32Value); try onToken(number.uint32Value)
                case "result":
                    if let forcedTokens {
                        guard traceCount == maximum, emitted.isEmpty, event["trace_only"] as? Bool == true, event["complete"] as? Bool == true, event["finish_reason"] as? String == "trace", event["generated_tokens"] as? Int == maximum, let predicted = event["tokens"] as? [UInt32], predicted.count == maximum, event["forced_tokens"] as? [UInt32] == forcedTokens else { throw WorkerFailure("Incomplete Q4 trace result") }
                        return event
                    }
                    guard let values = event["tokens"] as? [NSNumber], values.count == emitted.count,
                          zip(values, emitted).allSatisfy({ CFGetTypeID($0.0) != CFBooleanGetTypeID() && $0.0.doubleValue == Double($0.1) }),
                          ["stop", "length", "cancelled"].contains(event["finish_reason"] as? String ?? ""),
                          let count = event["generated_tokens"] as? NSNumber,
                          CFGetTypeID(count) != CFBooleanGetTypeID(), count.doubleValue == Double(emitted.count) else {
                        throw WorkerFailure("Worker final result does not match stream")
                    }
                    return event
                default: throw WorkerFailure("Unexpected worker event")
                }
            }
        } catch { close(); throw error }
    }

    func cancel() {
        lifecycle.lock(); defer { lifecycle.unlock() }
        if cancelOutput >= 0 {
            if requestActive { cancellationDeadline = min(cancellationDeadline ?? .infinity, monotonic() + 5) }
            var byte: UInt8 = 1; _ = Darwin.write(cancelOutput, &byte, 1)
        }
    }

    func close() {
        cancel()
        serial.lock(); defer { serial.unlock() }
        lifecycle.lock(); defer { lifecycle.unlock() }
        for fd in [input, output, cancelInput, cancelOutput] where fd >= 0 { Darwin.close(fd) }
        input = -1; output = -1; cancelInput = -1; cancelOutput = -1
        if pid > 0 {
            _ = kill(pid, SIGTERM)
            let deadline = monotonic() + 2
            var status: Int32 = 0
            while waitpid(pid, &status, WNOHANG) == 0 && monotonic() < deadline { usleep(10000) }
            if waitpid(pid, &status, WNOHANG) == 0 { _ = kill(pid, SIGKILL); _ = waitpid(pid, &status, 0) }
        }
        if modelLock >= 0 { Darwin.close(modelLock) }
        modelLock = -1; pid = 0; loaded = nil; pending.removeAll()
    }
}
