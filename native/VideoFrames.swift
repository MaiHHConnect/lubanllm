import Foundation
import AVFoundation
import ImageIO
import UniformTypeIdentifiers
import Darwin

// A one-shot local decoder. The caller owns the private upload and output directory.
struct VideoFailure: Error { let message: String }
func require(_ value: Bool, _ message: String) throws {
    if !value { throw VideoFailure(message: message) }
}
func emit(_ value: [String: Any], _ status: Int32) -> Never {
    let data = try! JSONSerialization.data(withJSONObject: value, options: [.sortedKeys])
    FileHandle.standardOutput.write(data + Data([10]))
    exit(status)
}
func metadata(_ fd: Int32) throws -> stat {
    var value = stat()
    try require(fstat(fd, &value) == 0, "无法读取视频文件信息")
    return value
}
func sameFile(_ a: stat, _ b: stat) -> Bool {
    a.st_dev == b.st_dev && a.st_ino == b.st_ino && a.st_size == b.st_size &&
    a.st_mtimespec.tv_sec == b.st_mtimespec.tv_sec && a.st_mtimespec.tv_nsec == b.st_mtimespec.tv_nsec &&
    a.st_ctimespec.tv_sec == b.st_ctimespec.tv_sec && a.st_ctimespec.tv_nsec == b.st_ctimespec.tv_nsec
}
func checkedPath(_ path: String) throws {
    try require(path.hasPrefix("/") && !path.utf8.contains(0), "视频路径必须是本机绝对路径")
    let parts = path.split(separator: "/")
    try require(!parts.contains(".") && !parts.contains("..") && path == "/" + parts.joined(separator: "/"),
                "视频路径不能包含相对目录")
    var current = ""
    for part in path.split(separator: "/") {
        current += "/" + part
        var value = stat()
        try require(lstat(current, &value) == 0 && (value.st_mode & S_IFMT) != S_IFLNK,
                    "视频路径不能经过符号链接")
    }
}
func decode(_ input: String, _ output: String) throws -> [String: Any] {
    try checkedPath(input)
    try checkedPath(output)
    let inputFD = open(input, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
    try require(inputFD >= 0, "无法打开视频文件")
    defer { close(inputFD) }
    let before = try metadata(inputFD)
    try require((before.st_mode & S_IFMT) == S_IFREG && before.st_uid == getuid() && before.st_nlink == 1 &&
                before.st_size > 0 && before.st_size <= 64 * 1024 * 1024, "视频必须是本人拥有且不超过64MiB的普通文件")
    let outputFD = open(output, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
    try require(outputFD >= 0, "无法打开视频帧目录")
    defer { close(outputFD) }
    let directory = try metadata(outputFD)
    try require(directory.st_uid == getuid() && directory.st_mode & 0o777 == 0o700,
                "视频帧目录必须是本人的0700私有目录")
    var created: [String] = []
    var completed = false
    defer { if !completed { for name in created { unlinkat(outputFD, name, 0) } } }
    let asset = AVURLAsset(url: URL(fileURLWithPath: input))
    let seconds = CMTimeGetSeconds(asset.duration)
    try require(seconds.isFinite && seconds > 0 && seconds <= 30, "视频必须在30秒以内")
    guard let track = asset.tracks(withMediaType: .video).first else {
        throw VideoFailure(message: "视频没有可解码的画面")
    }
    let size = track.naturalSize
    try require(size.width.isFinite && size.height.isFinite && size.width > 0 && size.height > 0 &&
                size.width <= 16_384 && size.height <= 16_384, "视频画面尺寸不受支持")
    let generator = AVAssetImageGenerator(asset: asset)
    generator.appliesPreferredTrackTransform = true
    generator.maximumSize = CGSize(width: 768, height: 768)
    generator.requestedTimeToleranceBefore = .zero
    generator.requestedTimeToleranceAfter = .zero
    var frames: [[String: Any]] = []
    var previous: Double = -1
    for index in 0..<4 {
        let requested = CMTime(seconds: seconds * (Double(index) + 0.5) / 4, preferredTimescale: 60_000)
        var actual = CMTime.zero
        let image = try generator.copyCGImage(at: requested, actualTime: &actual)
        let timestamp = CMTimeGetSeconds(actual)
        try require(timestamp.isFinite && timestamp >= 0 && timestamp <= seconds && image.width <= 768 && image.height <= 768,
                    "视频帧或真实时间戳无效")
        if timestamp == previous { continue }
        try require(timestamp > previous, "视频帧时间顺序无效")
        previous = timestamp
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil) else {
            throw VideoFailure(message: "无法创建视频帧")
        }
        CGImageDestinationAddImage(destination, image, [kCGImageDestinationLossyCompressionQuality: 0.8] as CFDictionary)
        try require(CGImageDestinationFinalize(destination) && data.length > 0 && data.length <= 2 * 1024 * 1024,
                    "无法编码视频帧")
        let name = String(format: "frame-%03d.jpg", index)
        let fd = openat(outputFD, name, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        try require(fd >= 0, "视频帧已存在或无法保存")
        created.append(name)
        do {
            try require(fchmod(fd, 0o600) == 0, "无法设置视频帧私有权限")
            try Data(referencing: data).withUnsafeBytes { raw in
                var offset = 0
                while offset < raw.count {
                    let count = write(fd, raw.baseAddress!.advanced(by: offset), raw.count - offset)
                    if count < 0 && errno == EINTR { continue }
                    try require(count > 0, "视频帧写入失败")
                    offset += count
                }
            }
            try require(fsync(fd) == 0, "视频帧保存失败")
            close(fd)
        } catch { close(fd); throw error }
        frames.append(["path": output + "/" + name, "timestamp_seconds": timestamp])
    }
    var afterPath = stat()
    try require(!frames.isEmpty && lstat(input, &afterPath) == 0 && sameFile(before, afterPath) &&
                sameFile(before, try metadata(inputFD)), "解码期间视频文件发生变化")
    var outputPath = stat()
    try require(lstat(output, &outputPath) == 0 && outputPath.st_dev == directory.st_dev &&
                outputPath.st_ino == directory.st_ino && outputPath.st_mode & 0o777 == 0o700,
                "解码期间视频帧目录发生变化")
    completed = true
    return ["ok": true, "duration_seconds": seconds, "frames": frames, "audio_processed": false]
}

DispatchQueue.global().asyncAfter(deadline: .now() + 30) {
    emit(["ok": false, "error": "视频解码超时"], 1)
}
do {
    try require(CommandLine.arguments.count == 3, "用法：luban-video-frames 输入视频绝对路径 私有输出目录")
    emit(try decode(CommandLine.arguments[1], CommandLine.arguments[2]), 0)
} catch let error as VideoFailure {
    emit(["ok": false, "error": error.message], 1)
} catch {
    emit(["ok": false, "error": "视频画面解码失败"], 1)
}
