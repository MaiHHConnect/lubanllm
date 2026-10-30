// SPDX-License-Identifier: Apache-2.0
import Foundation
import CryptoKit
import Darwin
import zlib

struct NativeUpdateArchiveError: Error, CustomStringConvertible {
    let description: String
    init(_ message: String) { description = message }
}

/// Isolated candidate only. Call validateApp before an extracted app can be admitted.
enum NativeUpdateArchive {
    // Only the build-time payload inspector defines this flag. App updates keep
    // their original limits and never admit full model installers.
#if COMPLETE_RELEASE_INSPECTION
    static let maximumArchiveSize = 4 * 1024 * 1024 * 1024 - 1
    static let maximumExpandedSize = 8 * 1024 * 1024 * 1024
#else
    static let maximumArchiveSize = 512 * 1024 * 1024
    static let maximumExpandedSize = 2 * 1024 * 1024 * 1024
#endif
    static let maximumEntries = 30_000
    // The guard is part of the shipped executable: keep its own bytes from
    // looking like the prohibited document while comparing the same full marker.
    private static let marker = Data(["LUBAN_INTERNAL", "ONLY_MINDMAP"].joined(separator: "_").utf8)

    struct Entry {
        let name: String
        let compressedSize: Int
        let expandedSize: Int
        let method: Int
        let flags: Int
        let crc: UInt32
        let mode: UInt32
        let localOffset: Int
        let payloadOffset: Int
        let endOffset: Int
        var sha256: String = ""
        var isDirectory: Bool { name.hasSuffix("/") }
    }
    struct Inspection {
        let entries: [Entry]
        let expandedSize: Int
        let sha256: String
    }

    private final class Reader {
        let handle: FileHandle
        let size: Int
        init(_ path: URL) throws {
            let fd = open(path.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
            guard fd >= 0 else { throw NativeUpdateArchiveError("Cannot open regular archive without following a link") }
            var info = stat()
            guard fstat(fd, &info) == 0, info.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG),
                  info.st_size > 0, info.st_size <= NativeUpdateArchive.maximumArchiveSize else {
                close(fd); throw NativeUpdateArchiveError("Archive size/type exceeds update limits")
            }
            size = Int(info.st_size); handle = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
        }
        func read(_ offset: Int, _ count: Int) throws -> Data {
            guard offset >= 0, count >= 0, offset <= size, count <= size - offset else { throw NativeUpdateArchiveError("ZIP read outside archive") }
            try handle.seek(toOffset: UInt64(offset))
            guard let data = try handle.read(upToCount: count), data.count == count else { throw NativeUpdateArchiveError("Truncated ZIP") }
            return data
        }
    }

    private static func u16(_ data: Data, _ offset: Int) -> Int {
        Int(data[offset]) | Int(data[offset + 1]) << 8
    }
    private static func u32(_ data: Data, _ offset: Int) -> UInt32 {
        UInt32(data[offset]) | UInt32(data[offset + 1]) << 8 | UInt32(data[offset + 2]) << 16 | UInt32(data[offset + 3]) << 24
    }
    private static func digest(_ hash: SHA256) -> String { hash.finalize().map { String(format: "%02x", $0) }.joined() }

    private static func checkExtra(_ extra: Data) throws {
        var at = 0
        // ditto's timestamp/UID/GID metadata is safe; alternate names and Unix link payloads are not.
        let allowed = Set([0x5455, 0x5855, 0x7855, 0x7875, 0x6375])
        while at < extra.count {
            guard extra.count - at >= 4 else { throw NativeUpdateArchiveError("Malformed ZIP extra field") }
            let type = u16(extra, at), size = u16(extra, at + 2)
            guard size <= extra.count - at - 4 else { throw NativeUpdateArchiveError("Truncated ZIP extra field") }
            guard allowed.contains(type) else { throw NativeUpdateArchiveError("Unsupported ZIP extra field (ZIP64, alternate path, links and unknown metadata are rejected)") }
            at += 4 + size
        }
    }

    private static func canonicalPath(_ name: String, internalOnly: Bool = false) throws -> String {
        guard !name.isEmpty, name.utf8.count <= 4096,
              !name.hasPrefix("/"), !name.contains("\\"), !name.contains(":"),
              !name.unicodeScalars.contains(where: { $0.value < 32 || (127...159).contains($0.value) || $0.value == 0x2028 || $0.value == 0x2029 }) else {
            throw NativeUpdateArchiveError("Unsafe/control-character archive path")
        }
        let normalized = name.precomposedStringWithCompatibilityMapping
        let parts = name.split(separator: "/", omittingEmptySubsequences: false)
        let normalizedParts = normalized.split(separator: "/", omittingEmptySubsequences: false)
        guard (internalOnly || parts.first == "Luban.app"), normalizedParts.count == parts.count,
              !normalized.contains("\\"), !normalized.contains(":"),
              !normalizedParts.contains(where: { $0 == "." || $0 == ".." }),
              !parts.enumerated().contains(where: { index, part in
                  (part.isEmpty && index != parts.count - 1) || part == "." || part == ".." || part.hasSuffix(" ") || part.hasSuffix(".")
              }), !normalizedParts.contains(where: { $0.lowercased() == "internal" || $0.hasPrefix("._") }),
              !(normalizedParts.last ?? "").contains("模型流程脑图") else {
            throw NativeUpdateArchiveError("Unsafe/outside-app/internal archive path")
        }
        return normalized.folding(options: [.caseInsensitive], locale: Locale(identifier: "en_US_POSIX")).trimmingCharacters(in: CharacterSet(charactersIn: "/"))
    }

    /// Parse both ZIP headers and the actual streams, not newline-delimited `unzip -Z1` output.
    static func inspect(_ archive: URL, expectedSHA256: String? = nil, internalOnly: Bool = false) throws -> Inspection {
        let reader = try Reader(archive)
        var archiveHash = SHA256()
        var hashAt = 0
        while hashAt < reader.size {
            let data = try reader.read(hashAt, min(65536, reader.size - hashAt)); archiveHash.update(data: data); hashAt += data.count
        }
        let archiveDigest = digest(archiveHash)
        if let expectedSHA256, expectedSHA256 != archiveDigest { throw NativeUpdateArchiveError("Update archive SHA256 mismatch") }
        let tailAt = max(0, reader.size - 65557), tail = try reader.read(tailAt, reader.size - tailAt)
        guard tail.count >= 22 else { throw NativeUpdateArchiveError("Missing ZIP end record") }
        var end: Int?
        for at in stride(from: tail.count - 22, through: 0, by: -1) {
            if u32(tail, at) == 0x06054b50, at + 22 + u16(tail, at + 20) == tail.count { end = at; break }
        }
        guard let end else { throw NativeUpdateArchiveError("Missing ZIP end record/trailing bytes") }
        guard tail.subdata(in: end + 22..<tail.count).range(of: Data([0x50, 0x4b, 0x05, 0x06])) == nil else {
            throw NativeUpdateArchiveError("Ambiguous ZIP end signature inside archive comment")
        }
        let count = u16(tail, end + 10), centralSize = Int(u32(tail, end + 12)), centralAt = Int(u32(tail, end + 16))
        guard u16(tail, end + 4) == 0, u16(tail, end + 6) == 0, u16(tail, end + 8) == count,
              count > 0, count <= maximumEntries, count != 0xffff,
              centralSize > 0, centralSize <= 32 * 1024 * 1024,
              centralAt != Int(UInt32.max), centralSize != Int(UInt32.max),
              centralAt <= reader.size, centralSize <= reader.size - centralAt,
              centralAt + centralSize == tailAt + end else {
            throw NativeUpdateArchiveError("Unsupported ZIP64/multi-disk or oversized/invalid central directory")
        }
        let central = try reader.read(centralAt, centralSize)
        var entries: [Entry] = [], at = 0, total = 0
        var names: Set<String> = [], directories: Set<String> = [], files: Set<String> = []
        for _ in 0..<count {
            guard central.count - at >= 46, u32(central, at) == 0x02014b50 else { throw NativeUpdateArchiveError("Malformed ZIP central entry") }
            let version = u16(central, at + 6), flags = u16(central, at + 8), method = u16(central, at + 10)
            let crc = u32(central, at + 16), compressed = Int(u32(central, at + 20)), expanded = Int(u32(central, at + 24))
            let nameSize = u16(central, at + 28), extraSize = u16(central, at + 30), commentSize = u16(central, at + 32)
            let mode = u32(central, at + 38) >> 16, localAt = Int(u32(central, at + 42))
            guard version <= 20, flags & ~0x080e == 0, method == 0 || method == 8,
                  (method == 8 || flags & 6 == 0), u16(central, at + 34) == 0,
                  compressed != Int(UInt32.max), expanded != Int(UInt32.max), localAt != Int(UInt32.max),
                  nameSize > 0, nameSize + extraSize + commentSize <= central.count - at - 46,
                  [UInt32(0), UInt32(S_IFREG), UInt32(S_IFDIR)].contains(mode & UInt32(S_IFMT)) else {
                throw NativeUpdateArchiveError("Encrypted/unsupported compression, ZIP64, links or invalid ZIP metadata")
            }
            let rawName = central.subdata(in: at + 46..<at + 46 + nameSize)
            guard let name = String(data: rawName, encoding: .utf8) else { throw NativeUpdateArchiveError("Only UTF-8/ASCII ZIP names are supported (including ditto's unflagged UTF-8)") }
            let key = try canonicalPath(name, internalOnly: internalOnly), isDirectory = name.hasSuffix("/")
            guard names.insert(key).inserted, !(mode & UInt32(S_IFMT) == UInt32(S_IFDIR) && !isDirectory),
                  !(mode & UInt32(S_IFMT) == UInt32(S_IFREG) && isDirectory),
                  (internalOnly || name != "Luban.app"), expanded <= maximumExpandedSize - total,
                  expanded <= max(4096, compressed * 1000), !isDirectory || (expanded == 0 && compressed == 0 && crc == 0) else {
                throw NativeUpdateArchiveError("Duplicate/alias, mismatched type or expansion limit")
            }
            total += expanded
            try checkExtra(central.subdata(in: at + 46 + nameSize..<at + 46 + nameSize + extraSize))
            let components = key.split(separator: "/")
            if components.count > 1 {
                for length in 1..<components.count { directories.insert(components.prefix(length).joined(separator: "/")) }
            }
            if isDirectory { directories.insert(key) } else { files.insert(key) }
            guard localAt < centralAt, centralAt - localAt >= 30 else { throw NativeUpdateArchiveError("Invalid local-header offset") }
            let local = try reader.read(localAt, 30)
            let localNameSize = u16(local, 26), localExtraSize = u16(local, 28)
            guard u32(local, 0) == 0x04034b50, u16(local, 4) == version, u16(local, 6) == flags,
                  u16(local, 8) == method, localNameSize == nameSize,
                  localNameSize + localExtraSize <= centralAt - localAt - 30 else { throw NativeUpdateArchiveError("Local/central ZIP header mismatch") }
            let localMetadata = try reader.read(localAt + 30, localNameSize + localExtraSize)
            guard localMetadata.prefix(localNameSize) == rawName else { throw NativeUpdateArchiveError("Local/central ZIP name mismatch") }
            try checkExtra(localMetadata.subdata(in: localNameSize..<localMetadata.count))
            let payloadAt = localAt + 30 + localNameSize + localExtraSize
            guard compressed <= centralAt - payloadAt else { throw NativeUpdateArchiveError("ZIP data crosses central directory") }
            var entryEnd = payloadAt + compressed
            if flags & 8 == 0 {
                guard u32(local, 14) == crc, Int(u32(local, 18)) == compressed, Int(u32(local, 22)) == expanded else { throw NativeUpdateArchiveError("Local/central ZIP sizes or CRC mismatch") }
            } else {
                guard [UInt32(0), crc].contains(u32(local, 14)), [UInt32(0), UInt32(compressed)].contains(u32(local, 18)),
                      [UInt32(0), UInt32(expanded)].contains(u32(local, 22)), centralAt - entryEnd >= 12 else { throw NativeUpdateArchiveError("Invalid ZIP data descriptor") }
                let signed = u32(try reader.read(entryEnd, 4), 0) == 0x08074b50
                let descriptor = try reader.read(entryEnd + (signed ? 4 : 0), 12)
                guard u32(descriptor, 0) == crc, Int(u32(descriptor, 4)) == compressed, Int(u32(descriptor, 8)) == expanded else { throw NativeUpdateArchiveError("ZIP data descriptor mismatch") }
                entryEnd += signed ? 16 : 12
            }
            guard entryEnd <= centralAt else { throw NativeUpdateArchiveError("ZIP descriptor crosses central directory") }
            entries.append(Entry(name: name, compressedSize: compressed, expandedSize: expanded, method: method, flags: flags, crc: crc, mode: mode, localOffset: localAt, payloadOffset: payloadAt, endOffset: entryEnd))
            at += 46 + nameSize + extraSize + commentSize
        }
        guard at == central.count, files.isDisjoint(with: directories) else { throw NativeUpdateArchiveError("ZIP central padding or file/directory collision") }
        var localEnd = 0
        for entry in entries.sorted(by: { $0.localOffset < $1.localOffset }) {
            guard entry.localOffset == localEnd else { throw NativeUpdateArchiveError("ZIP overlapping/gapped/prepended local records") }
            localEnd = entry.endOffset
        }
        guard localEnd == centralAt else { throw NativeUpdateArchiveError("Unexpected data before central directory") }
        for index in entries.indices { entries[index].sha256 = try inspectPayload(reader, entries[index], internalOnly: internalOnly) }
        return Inspection(entries: entries, expandedSize: total, sha256: archiveDigest)
    }

    private static func textPrefix(_ bytes: Data) -> Bool {
        guard !bytes.isEmpty, !bytes.contains(where: { $0 < 32 && ![9, 10, 13].contains($0) }) else { return false }
        if String(data: bytes, encoding: .utf8) != nil { return true }
        // The sample may end in the middle of one UTF-8 scalar, as in an incremental decoder.
        for count in 1...min(3, bytes.count) {
            let suffix = bytes.suffix(count), first = suffix.first!
            let width = (0xc2...0xdf).contains(first) ? 2 : (0xe0...0xef).contains(first) ? 3 : (0xf0...0xf4).contains(first) ? 4 : 0
            if width > count, suffix.dropFirst().allSatisfy({ (0x80...0xbf).contains($0) }),
               String(data: bytes.dropLast(count), encoding: .utf8) != nil { return true }
        }
        return false
    }

    private static func inspectPayload(_ reader: Reader, _ entry: Entry, internalOnly: Bool) throws -> String {
        var stream = z_stream(), hash = SHA256(), crc: uLong = 0, written = 0, markerTail = Data()
        let textExtensions = Set(["html", "htm", "txt", "md", "json", "js", "ts", "css", "xml", "svg", "py", "plist", "yaml", "yml", "csv", "conf", "ini", "map", "toml", "sh", "license"])
        var scanMarker: Bool? = !internalOnly || entry.expandedSize <= 1024 * 1024 || textExtensions.contains((entry.name as NSString).pathExtension.lowercased()) ? true : nil
        var pending = Data()
        func checkMarker(_ data: Data) throws {
            let checked = markerTail + data
            guard checked.range(of: marker) == nil else { throw NativeUpdateArchiveError("Internal-only content marker is forbidden") }
            markerTail = Data(checked.suffix(marker.count - 1))
        }
        if entry.method == 8 {
            guard inflateInit2_(&stream, -MAX_WBITS, ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size)) == Z_OK else { throw NativeUpdateArchiveError("Cannot initialize system raw DEFLATE decoder") }
        }
        defer { if entry.method == 8 { inflateEnd(&stream) } }
        func consume(_ data: Data) throws {
            guard data.count <= entry.expandedSize - written else { throw NativeUpdateArchiveError("Actual ZIP stream exceeds declared size") }
            written += data.count; hash.update(data: data)
            crc = data.withUnsafeBytes { crc32(crc, $0.bindMemory(to: Bytef.self).baseAddress, uInt(data.count)) }
            if scanMarker == nil {
                pending.append(data)
                if pending.count >= 4096 || written == entry.expandedSize {
                    scanMarker = textPrefix(Data(pending.prefix(4096)))
                    if scanMarker == true { try checkMarker(pending) }
                    pending.removeAll(keepingCapacity: false)
                }
            } else if scanMarker == true { try checkMarker(data) }
        }
        var offset = 0, ended = false
        if entry.method == 0 {
            guard entry.compressedSize == entry.expandedSize else { throw NativeUpdateArchiveError("Stored ZIP entry size mismatch") }
        }
        while offset < entry.compressedSize {
            let input = try reader.read(entry.payloadOffset + offset, min(65536, entry.compressedSize - offset))
            offset += input.count
            if entry.method == 0 { try consume(input); continue }
            try input.withUnsafeBytes { bytes in
                stream.next_in = UnsafeMutablePointer(mutating: bytes.bindMemory(to: Bytef.self).baseAddress!); stream.avail_in = uInt(input.count)
                var output = [UInt8](repeating: 0, count: 65536)
                repeat {
                    let before = stream.avail_in
                    let status = output.withUnsafeMutableBufferPointer { buffer -> Int32 in
                        stream.next_out = buffer.baseAddress!; stream.avail_out = uInt(buffer.count)
                        return inflate(&stream, Z_NO_FLUSH)
                    }
                    let count = output.count - Int(stream.avail_out)
                    try consume(Data(output.prefix(count)))
                    if status == Z_STREAM_END {
                        guard stream.avail_in == 0, offset == entry.compressedSize else { throw NativeUpdateArchiveError("Trailing bytes inside DEFLATE stream") }
                        ended = true; break
                    }
                    guard status == Z_OK, count > 0 || stream.avail_in < before else { throw NativeUpdateArchiveError("Invalid/non-progressing DEFLATE stream") }
                } while stream.avail_in > 0 || stream.avail_out == 0
            }
        }
        guard (entry.method == 0 || ended), written == entry.expandedSize, UInt32(crc) == entry.crc else { throw NativeUpdateArchiveError("Actual ZIP stream size/CRC mismatch") }
        return digest(hash)
    }

    private static func run(_ executable: String, _ arguments: [String]) throws {
        let process = Process(), pipe = Pipe()
        process.executableURL = URL(fileURLWithPath: executable); process.arguments = arguments
        process.standardOutput = pipe; process.standardError = pipe
        try process.run()
        let output = pipe.fileHandleForReading.readDataToEndOfFile(); process.waitUntilExit()
        guard process.terminationStatus == 0 else { throw NativeUpdateArchiveError("System archive/app validator failed: " + String(decoding: output.prefix(2048), as: UTF8.self)) }
    }

    /// Snapshot into an exclusively-created private stage; ditto never sees unvalidated source bytes.
    static func extract(_ archive: URL, to stage: URL, expectedSHA256: String? = nil) throws -> Inspection {
        let parent = stage.deletingLastPathComponent().standardizedFileURL
        guard parent.resolvingSymlinksInPath().path == parent.path, stage.lastPathComponent != ".", stage.lastPathComponent != "..",
              mkdir(stage.path, 0o700) == 0 else { throw NativeUpdateArchiveError("Extraction requires a new stage under an existing real directory") }
        do {
#if COMPLETE_RELEASE_INSPECTION
            // A newly produced, immutable release archive is inspected directly;
            // validation checks every extracted file before payload admission.
            let inspected = try inspect(archive, expectedSHA256: expectedSHA256)
            try run("/usr/bin/ditto", ["-x", "-k", "--norsrc", "--noextattr", archive.path, stage.path])
            try validateExtraction(stage, inspected)
            return inspected
#else
            let snapshot = stage.appendingPathComponent(".validated-update.zip")
            let source = try Reader(archive)
            let fd = open(snapshot.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
            guard fd >= 0 else { throw NativeUpdateArchiveError("Cannot create private update snapshot") }
            let destination = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
            var offset = 0
            while offset < source.size {
                let data = try source.read(offset, min(65536, source.size - offset)); try destination.write(contentsOf: data); offset += data.count
            }
            try destination.close()
            let inspected = try inspect(snapshot, expectedSHA256: expectedSHA256)
            try FileManager.default.setAttributes([.posixPermissions: 0o400], ofItemAtPath: snapshot.path)
            try run("/usr/bin/ditto", ["-x", "-k", "--norsrc", "--noextattr", snapshot.path, stage.path])
            try FileManager.default.removeItem(at: snapshot)
            try validateExtraction(stage, inspected)
            return inspected
#endif
        } catch {
            try? FileManager.default.removeItem(at: stage)
            throw error
        }
    }

    private static func validateExtraction(_ stage: URL, _ inspection: Inspection) throws {
        var expectedFiles: [String: Entry] = [:], expectedDirectories: Set<String> = []
        for entry in inspection.entries {
            let key = try canonicalPath(entry.name)
            if entry.isDirectory { expectedDirectories.insert(key) } else { expectedFiles[key] = entry }
            let components = key.split(separator: "/")
            if components.count > 1 {
                for length in 1..<components.count { expectedDirectories.insert(components.prefix(length).joined(separator: "/")) }
            }
        }
        var files: Set<String> = [], directories: Set<String> = []
        for url in try allEntries(stage) {
            var info = stat()
            guard lstat(url.path, &info) == 0 else { throw NativeUpdateArchiveError("Missing extracted entry") }
            let relative = String(url.path.dropFirst(stage.path.count + 1)), key = try canonicalPath(relative)
            if info.st_mode & mode_t(S_IFMT) == mode_t(S_IFDIR) {
                guard expectedDirectories.contains(key) else { throw NativeUpdateArchiveError("Unexpected extracted directory") }
                directories.insert(key)
                guard chmod(url.path, 0o755) == 0 else { throw NativeUpdateArchiveError("Cannot set directory permissions") }
            } else {
                guard info.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG), info.st_nlink == 1,
                      let expected = expectedFiles[key], Int(info.st_size) == expected.expandedSize else { throw NativeUpdateArchiveError("Unexpected/link/type/size in extracted update") }
                let handle = try FileHandle(forReadingFrom: url); defer { try? handle.close() }
                var hash = SHA256()
                while let data = try handle.read(upToCount: 65536), !data.isEmpty { hash.update(data: data) }
                guard digest(hash) == expected.sha256 else { throw NativeUpdateArchiveError("Extracted content differs from validated ZIP stream") }
                files.insert(key)
                guard chmod(url.path, expected.mode & 0o111 != 0 ? 0o755 : 0o644) == 0 else { throw NativeUpdateArchiveError("Cannot set executable permissions") }
            }
        }
        guard files == Set(expectedFiles.keys), directories == expectedDirectories else { throw NativeUpdateArchiveError("Missing extracted update entry") }
        // Permission writes can create platform metadata (notably AppleDouble on ExFAT).
        // Verify the final tree after all writes; never approve entries created after the first enumeration.
        var finalFiles: Set<String> = [], finalDirectories: Set<String> = []
        for url in try allEntries(stage) {
            var info = stat()
            guard lstat(url.path, &info) == 0 else { throw NativeUpdateArchiveError("Missing final update entry") }
            let key = try canonicalPath(String(url.path.dropFirst(stage.path.count + 1)))
            if info.st_mode & mode_t(S_IFMT) == mode_t(S_IFDIR), expectedDirectories.contains(key) { finalDirectories.insert(key) }
            else if info.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG), info.st_nlink == 1,
                    let expected = expectedFiles[key], Int(info.st_size) == expected.expandedSize { finalFiles.insert(key) }
            else { throw NativeUpdateArchiveError("Unexpected final extracted entry") }
        }
        guard finalFiles == files, finalDirectories == directories else { throw NativeUpdateArchiveError("Final update tree changed during permission normalization") }
    }

    /// Foundation hides AppleDouble records on some volumes; POSIX reads the actual inventory.
    private static func allEntries(_ root: URL) throws -> [URL] {
        var pending = [root], result: [URL] = []
        while let directory = pending.popLast() {
            guard let stream = opendir(directory.path) else { throw NativeUpdateArchiveError("Cannot read actual extracted directory") }
            defer { closedir(stream) }
            while true {
                errno = 0
                guard let record = readdir(stream) else {
                    guard errno == 0 else { throw NativeUpdateArchiveError("Directory enumeration failed") }
                    break
                }
                let name = withUnsafeBytes(of: &record.pointee.d_name) { raw in
                    String(bytes: raw.prefix(while: { $0 != 0 }), encoding: .utf8)
                }
                guard let name else { throw NativeUpdateArchiveError("Non-UTF8 extracted filesystem name") }
                if name == "." || name == ".." { continue }
                let child = directory.appendingPathComponent(name)
                var info = stat()
                guard lstat(child.path, &info) == 0 else { throw NativeUpdateArchiveError("Cannot inspect actual extracted file") }
                result.append(child)
                if info.st_mode & mode_t(S_IFMT) == mode_t(S_IFDIR) { pending.append(child) }
            }
        }
        return result
    }

    static func deltaPath(_ path: String) throws -> String {
        guard path.hasPrefix("Contents/"), !path.hasSuffix("/"),
              path.precomposedStringWithCompatibilityMapping == path else {
            throw NativeUpdateArchiveError("Unsafe module update path")
        }
        return try canonicalPath("Luban.app/" + path)
    }

    /// Build a new candidate from an unchanged installed app and a validated partial ZIP.
    /// The caller must verify both the baseline and the reconstructed full app signature.
    static func reconstruct(_ archive: URL, base: URL, to stage: URL,
                            expectedSHA256: String, removed: [String]) throws -> Inspection {
        let fm = FileManager.default
        func checkTree(_ root: URL) throws {
            guard root.standardizedFileURL == root.resolvingSymlinksInPath() else {
                throw NativeUpdateArchiveError("Linked module baseline")
            }
            let entries = try allEntries(root)
            guard entries.count <= maximumEntries else { throw NativeUpdateArchiveError("Module tree is too large") }
            var bytes: Int64 = 0, names: Set<String> = []
            for file in entries {
                let relative = String(file.path.dropFirst(root.path.count + 1))
                let key = try canonicalPath("Luban.app/" + relative)
                var info = stat()
                guard names.insert(key).inserted, lstat(file.path, &info) == 0,
                      (info.st_mode & mode_t(S_IFMT) == mode_t(S_IFDIR)
                        || info.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG) && info.st_nlink == 1) else {
                    throw NativeUpdateArchiveError("Linked, duplicate or special module entry")
                }
                if info.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG) { bytes += info.st_size }
                guard bytes <= maximumExpandedSize else { throw NativeUpdateArchiveError("Module app size exceeds limits") }
            }
        }
        try checkTree(base)
        let inspection = try extract(archive, to: stage, expectedSHA256: expectedSHA256)
        do {
            let app = stage.appendingPathComponent("Luban.app", isDirectory: true)
            let patch = stage.appendingPathComponent(".delta-source", isDirectory: true)
            try fm.moveItem(at: app, to: patch)
            try fm.copyItem(at: base, to: app)
            try checkTree(app)
            let changed = try Set(inspection.entries.filter { !$0.isDirectory }.map { try canonicalPath($0.name) })
            var deleted: Set<String> = []
            for path in removed {
                let key = try deltaPath(path), target = app.appendingPathComponent(path)
                var info = stat()
                guard deleted.insert(key).inserted, !changed.contains(key),
                      lstat(target.path, &info) == 0, info.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG) else {
                    throw NativeUpdateArchiveError("Missing, conflicting or non-file module deletion")
                }
                try fm.removeItem(at: target)
            }
            for entry in inspection.entries where !entry.isDirectory {
                let path = String(entry.name.dropFirst("Luban.app/".count))
                _ = try deltaPath(path)
                let target = app.appendingPathComponent(path)
                try fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                if fm.fileExists(atPath: target.path) { try fm.removeItem(at: target) }
                try fm.copyItem(at: patch.appendingPathComponent(path), to: target)
            }
            try fm.removeItem(at: patch)
            try checkTree(app)
            return inspection
        } catch {
            try? fm.removeItem(at: stage)
            throw error
        }
    }

    /// The existing updater's app identity, local-environment, architecture and signature checks.
    static func validateApp(_ app: URL, bundleIdentifier: String, version: String, build: Int, currentProject: String) throws {
        let contents = app.appendingPathComponent("Contents"), resources = contents.appendingPathComponent("Resources")
        let plist = try PropertyListSerialization.propertyList(from: Data(contentsOf: contents.appendingPathComponent("Info.plist")), format: nil) as? [String: Any]
        guard plist?["CFBundleIdentifier"] as? String == bundleIdentifier,
              plist?["CFBundleShortVersionString"] as? String == version,
              plist?["CFBundleVersion"] as? String == String(build), plist?["CFBundleExecutable"] as? String == "Luban",
              FileManager.default.isExecutableFile(atPath: contents.appendingPathComponent("MacOS/Luban").path),
              FileManager.default.fileExists(atPath: resources.appendingPathComponent("backend/src/luban/server.py").path) else { throw NativeUpdateArchiveError("Update app identity/version/payload mismatch") }
        func json(_ path: String) throws -> [String: Any] {
            guard let value = try JSONSerialization.jsonObject(with: Data(contentsOf: resources.appendingPathComponent(path))) as? [String: Any] else { throw NativeUpdateArchiveError("Invalid app version metadata") }
            return value
        }
        for path in ["release.json", "backend/release.json"] {
            let release = try json(path)
            guard release["version"] as? String == version, release["build"] as? Int == build else { throw NativeUpdateArchiveError("Update release metadata mismatch") }
        }
        guard try json("launcher.json")["project"] as? String == currentProject else { throw NativeUpdateArchiveError("Update local runtime environment mismatch") }
        #if arch(arm64)
        let architecture = "arm64"
        #else
        let architecture = "x86_64"
        #endif
        try run("/usr/bin/lipo", [contents.appendingPathComponent("MacOS/Luban").path, "-verify_arch", architecture])
        try run("/usr/bin/codesign", ["--verify", "--deep", "--strict", app.path])
    }
}
