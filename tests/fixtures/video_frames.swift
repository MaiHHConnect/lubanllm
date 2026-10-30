import AVFoundation
import CoreVideo
import Foundation

let output = URL(fileURLWithPath: CommandLine.arguments[1])
let duration = Int(CommandLine.arguments[2])!
let writer = try AVAssetWriter(outputURL: output, fileType: .mov)
let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
    AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: 320, AVVideoHeightKey: 180,
])
let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
    kCVPixelBufferWidthKey as String: 320, kCVPixelBufferHeightKey as String: 180,
])
writer.add(input)
guard writer.startWriting() else { fatalError("fixture start") }
writer.startSession(atSourceTime: .zero)
for frame in 0..<(duration * 10) {
    while !input.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.001) }
    var buffer: CVPixelBuffer?
    guard CVPixelBufferCreate(kCFAllocatorDefault, 320, 180, kCVPixelFormatType_32ARGB, nil, &buffer) == kCVReturnSuccess,
          let pixels = buffer else { fatalError("fixture pixels") }
    CVPixelBufferLockBaseAddress(pixels, [])
    let pointer = CVPixelBufferGetBaseAddress(pixels)!.assumingMemoryBound(to: UInt8.self)
    for row in 0..<180 {
        for column in 0..<320 {
            let offset = row * CVPixelBufferGetBytesPerRow(pixels) + column * 4
            pointer[offset] = 255
            pointer[offset + 1] = UInt8(frame % 256)
            pointer[offset + 2] = UInt8(column % 256)
            pointer[offset + 3] = UInt8(row % 256)
        }
    }
    CVPixelBufferUnlockBaseAddress(pixels, [])
    guard adaptor.append(pixels, withPresentationTime: CMTime(value: Int64(frame), timescale: 10)) else {
        fatalError("fixture append")
    }
}
writer.endSession(atSourceTime: CMTime(value: Int64(duration), timescale: 1))
input.markAsFinished()
let done = DispatchSemaphore(value: 0)
writer.finishWriting { done.signal() }
guard done.wait(timeout: .now() + 20) == .success, writer.status == .completed else { fatalError("fixture finish") }
