// Build-time adapter: reuse the shipped archive parser and extraction verification.
import Foundation

@main struct ReleaseInspection {
    static func main() {
        do {
            if CommandLine.arguments.count == 3, CommandLine.arguments[1] == "--internal-only" {
                let inspection = try NativeUpdateArchive.inspect(URL(fileURLWithPath: CommandLine.arguments[2]), internalOnly: true)
                print("{\"schema\":\"luban-native-internal-inspection-v1\",\"sha256\":\"\(inspection.sha256)\"}")
                return
            }
            guard CommandLine.arguments.count == 4 else { throw NativeUpdateArchiveError("Archive, new private stage and approved SHA required") }
            let inspection = try NativeUpdateArchive.extract(URL(fileURLWithPath: CommandLine.arguments[1]),
                to: URL(fileURLWithPath: CommandLine.arguments[2]), expectedSHA256: CommandLine.arguments[3])
            let result: [String: Any] = ["schema": "luban-native-release-inspection-v1", "sha256": inspection.sha256,
                "expanded_size": inspection.expandedSize, "entries": inspection.entries.map {
                    ["name": $0.name, "size": $0.expandedSize, "sha256": $0.sha256, "directory": $0.isDirectory] as [String: Any]
                }]
            print(String(decoding: try JSONSerialization.data(withJSONObject: result, options: [.sortedKeys]), as: UTF8.self))
        } catch { FileHandle.standardError.write(Data(("发行归档核查失败：\(error)\n").utf8)); exit(1) }
    }
}
