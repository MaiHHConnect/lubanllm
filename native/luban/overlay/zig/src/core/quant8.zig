// SPDX-License-Identifier: Apache-2.0
//! Symmetric signed-int8/group64 copies with BF16 scales, for inference only.
const std = @import("std");
const bf = @import("affine4.zig");
const Failure = error{NonfiniteQuant8};

fn groups(result: *Failure!void, source: []const u16, codes: []u8, scales: []u16, begin: usize, end: usize) void {
    result.* = convert(source, codes, scales, begin, end);
}

fn convert(source: []const u16, codes: []u8, scales: []u16, begin: usize, end: usize) Failure!void {
    for (begin..end) |group| {
        const values = source[group * 64 ..][0..64];
        var peak: f32 = 0;
        for (values) |bits| {
            const value = bf.asFloat(bits);
            if (!std.math.isFinite(value)) return error.NonfiniteQuant8;
            peak = @max(peak, @abs(value));
        }
        const bits = bf.asBf16(if (peak == 0) @as(f32, 1) else @max(peak / 127, 1e-30));
        const scale = bf.asFloat(bits);
        if (!std.math.isFinite(scale) or scale <= 0) return error.NonfiniteQuant8;
        scales[group] = bits;
        for (values, 0..) |value, i| {
            const code: i8 = @intFromFloat(@min(@as(f32, 127), @max(@as(f32, -127), @round(bf.asFloat(value) / scale))));
            codes[group * 64 + i] = @bitCast(code);
        }
    }
}

pub fn quantize(source: []const u16, codes: []u8, scales: []u16) !void {
    if (source.len == 0 or source.len % 64 != 0 or codes.len != source.len or scales.len != source.len / 64)
        return error.BadQuant8Shape;
    const workers = @min(@as(usize, 16), std.Thread.getCpuCount() catch 1, @max(1, scales.len / 4096));
    const per = (scales.len + workers - 1) / workers;
    var results: [16]Failure!void = @splat({});
    var threads: [16]?std.Thread = @splat(null);
    for (1..workers) |w| threads[w] = std.Thread.spawn(.{}, groups, .{
        &results[w], source, codes, scales, w * per, @min(scales.len, (w + 1) * per),
    }) catch null;
    groups(&results[0], source, codes, scales, 0, @min(scales.len, per));
    for (1..workers) |w| if (threads[w]) |t| t.join() else
        groups(&results[w], source, codes, scales, w * per, @min(scales.len, (w + 1) * per));
    for (results[0..workers]) |r| try r;
}

test "signed int8 reconstructs exact dyadic values and zero groups" {
    var source: [128]u16 = @splat(0);
    for (source[0..64], 0..) |*v, i| v.* = bf.asBf16(@as(f32, @floatFromInt(@as(i32, @intCast(i)) * 4 - 127)) / 64);
    var codes: [128]u8 = undefined;
    var scales: [2]u16 = undefined;
    try quantize(&source, &codes, &scales);
    for (source, codes, 0..) |value, code, i| try std.testing.expectEqual(bf.asFloat(value),
        @as(f32, @floatFromInt(@as(i8, @bitCast(code)))) * bf.asFloat(scales[i / 64]));
    source[12] = 0x7fc0;
    try std.testing.expectError(error.NonfiniteQuant8, quantize(&source, &codes, &scales));
    try std.testing.expectError(error.BadQuant8Shape, quantize(source[0..63], codes[0..63], &scales));
}

test "threaded quantization equals serial and error is bounded by one rounded scale" {
    const a = std.testing.allocator;
    const n = 64 * 9000;
    const source = try a.alloc(u16, n);
    defer a.free(source);
    const codes = try a.alloc(u8, n);
    defer a.free(codes);
    const expected = try a.alloc(u8, n);
    defer a.free(expected);
    const scales = try a.alloc(u16, n / 64);
    defer a.free(scales);
    const expected_scales = try a.alloc(u16, n / 64);
    defer a.free(expected_scales);
    var rng = std.Random.DefaultPrng.init(42);
    for (source) |*v| v.* = bf.asBf16(rng.random().floatNorm(f32));
    try quantize(source, codes, scales);
    try convert(source, expected, expected_scales, 0, scales.len);
    try std.testing.expectEqualSlices(u8, expected, codes);
    try std.testing.expectEqualSlices(u16, expected_scales, scales);
    for (source, codes, 0..) |value, code, i| {
        const scale = bf.asFloat(scales[i / 64]);
        const reconstructed = @as(f32, @floatFromInt(@as(i8, @bitCast(code)))) * scale;
        try std.testing.expect(@abs(reconstructed - bf.asFloat(value)) <= scale);
    }
}
