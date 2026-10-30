// JSONL driver for real ZIP differential fixtures; no Python in the candidate executable.
import Foundation

@main struct NativeUpdateArchiveTests {
    static func main() throws {
        while let line = readLine() {
            do {
                guard let request = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                      let path = request["archive"] as? String else { throw NativeUpdateArchiveError("Missing archive") }
                let archive = URL(fileURLWithPath: path), hash = request["sha256"] as? String
                let inspected: NativeUpdateArchive.Inspection
                if let stage = request["stage"] as? String {
                    if let base = request["base"] as? String, let hash {
                        inspected = try NativeUpdateArchive.reconstruct(archive, base: URL(fileURLWithPath: base),
                            to: URL(fileURLWithPath: stage), expectedSHA256: hash, removed: request["removed"] as? [String] ?? [])
                    } else {
                        inspected = try NativeUpdateArchive.extract(archive, to: URL(fileURLWithPath: stage), expectedSHA256: hash)
                    }
                    if let identity = request["identity"] as? [String: Any] {
                        guard let id = identity["bundle_identifier"] as? String, let version = identity["version"] as? String,
                              let build = identity["build"] as? Int, let project = identity["project"] as? String else { throw NativeUpdateArchiveError("Missing expected identity") }
                        try NativeUpdateArchive.validateApp(URL(fileURLWithPath: stage).appendingPathComponent("Luban.app"), bundleIdentifier: id, version: version, build: build, currentProject: project)
                    }
                } else { inspected = try NativeUpdateArchive.inspect(archive, expectedSHA256: hash) }
                let result: [String: Any] = ["accepted": true, "count": inspected.entries.count, "expanded_size": inspected.expandedSize,
                    "sha256": inspected.sha256, "entries": inspected.entries.map { ["name": $0.name, "size": $0.expandedSize, "sha256": $0.sha256] as [String: Any] }]
                print(String(decoding: try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys]), as: UTF8.self))
            } catch {
                print(String(decoding: try JSONSerialization.data(withJSONObject: ["accepted": false, "error": String(describing: error)], options: [.sortedKeys]), as: UTF8.self))
            }
        }
    }
}
