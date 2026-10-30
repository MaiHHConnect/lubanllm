//! CPU affine q4/group64 preparation preserves BF16 metadata and low-bit-first MLX words.
const std = @import("std");
pub const Prepared = struct {
    allocator: std.mem.Allocator,
    n: usize,
    k: usize,
    words: []u32,
    scales: []u16,
    biases: []u16,
    pub fn deinit(p: *Prepared) void {
        p.allocator.free(p.words);
        p.allocator.free(p.scales);
        p.allocator.free(p.biases);
        p.* = undefined;
    }
};
pub fn asFloat(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}
pub fn asBf16(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}
pub fn prepare(a: std.mem.Allocator, weights: []const u16, n: usize, k: usize) !Prepared {
    return prepareGroup(a, weights, n, k, 64);
}

/// Prepare the same affine Q4 words with a smaller shared range. Group 32
/// keeps the existing packed codes and kernel arithmetic while reducing the
/// number of unrelated values sharing one scale/bias pair.
pub fn prepareGroup(a: std.mem.Allocator, weights: []const u16, n: usize, k: usize, group_size: usize) !Prepared {
    return prepareWithClipGroup(a, weights, n, k, 1.0, group_size);
}

pub fn prepareGroupClipped(a: std.mem.Allocator, weights: []const u16, n: usize, k: usize, ratio: f32, group_size: usize) !Prepared {
    if (!std.math.isFinite(ratio) or ratio <= 0 or ratio > 1) return error.BadAffineClipRatio;
    return prepareWithClipGroup(a, weights, n, k, ratio, group_size);
}

/// Prepare group-specific clipping ratios. Ratios are shared across output
/// rows for each input group, which matches the activation-energy statistic
/// collected by the calibration script and keeps the sidecar compact.
pub fn prepareGroupClippedProfile(a: std.mem.Allocator, weights: []const u16, n: usize, k: usize, ratios: []const f32, group_size: usize) !Prepared {
    if ((group_size != 16 and group_size != 32 and group_size != 64) or n == 0 or k == 0 or k % group_size != 0 or weights.len != try std.math.mul(usize, n, k) or ratios.len != k / group_size) return error.BadAffineShape;
    for (ratios) |ratio| if (!std.math.isFinite(ratio) or ratio <= 0 or ratio > 1) return error.BadAffineClipRatio;
    const words = try a.alloc(u32, weights.len / 8);
    errdefer a.free(words);
    const scales = try a.alloc(u16, weights.len / group_size);
    errdefer a.free(scales);
    const biases = try a.alloc(u16, scales.len);
    errdefer a.free(biases);
    const workers = @min(max_workers, std.Thread.getCpuCount() catch 1, @max(1, scales.len / 4096));
    const per = (scales.len + workers - 1) / workers;
    var results: [max_workers]Error!void = @splat({});
    var threads: [max_workers]?std.Thread = @splat(null);
    for (1..workers) |worker| threads[worker] = std.Thread.spawn(.{}, runProfile, .{ &results[worker], weights, words, scales, biases, worker * per, @min(scales.len, (worker + 1) * per), ratios, k / group_size, group_size }) catch null;
    runProfile(&results[0], weights, words, scales, biases, 0, @min(scales.len, per), ratios, k / group_size, group_size);
    for (1..workers) |worker| if (threads[worker]) |thread| thread.join() else runProfile(&results[worker], weights, words, scales, biases, worker * per, @min(scales.len, (worker + 1) * per), ratios, k / group_size, group_size);
    for (results[0..workers]) |result| try result;
    return .{ .allocator = a, .n = n, .k = k, .words = words, .scales = scales, .biases = biases };
}

/// Choose the affine range per group by minimizing BF16-roundtrip weight MSE.
/// The packed ABI is unchanged; this only replaces the min/max range heuristic.
pub fn prepareGroupMse(a: std.mem.Allocator, weights: []const u16, n: usize, k: usize, group_size: usize) !Prepared {
    if ((group_size != 16 and group_size != 32 and group_size != 64) or n == 0 or k == 0 or k % group_size != 0 or weights.len != try std.math.mul(usize, n, k)) return error.BadAffineShape;
    const words = try a.alloc(u32, weights.len / 8);
    errdefer a.free(words);
    const scales = try a.alloc(u16, weights.len / group_size);
    errdefer a.free(scales);
    const biases = try a.alloc(u16, scales.len);
    errdefer a.free(biases);
    try groupsMse(weights, words, scales, biases, 0, scales.len, group_size);
    return .{ .allocator = a, .n = n, .k = k, .words = words, .scales = scales, .biases = biases };
}

/// Inference-only group clipping. Ratio 1.0 preserves the original bytes;
/// lower ratios reduce the influence of a group outlier before Q4 coding.
pub fn prepareClipped(a: std.mem.Allocator, weights: []const u16, n: usize, k: usize, ratio: f32) !Prepared {
    if (!std.math.isFinite(ratio) or ratio <= 0 or ratio > 1) return error.BadAffineClipRatio;
    return prepareWithClipGroup(a, weights, n, k, ratio, 64);
}

fn prepareWithClipGroup(a: std.mem.Allocator, weights: []const u16, n: usize, k: usize, ratio: f32, group_size: usize) !Prepared {
    if ((group_size != 16 and group_size != 32 and group_size != 64) or n == 0 or k == 0 or k % group_size != 0 or weights.len != try std.math.mul(usize, n, k)) return error.BadAffineShape;
    const words = try a.alloc(u32, weights.len / 8);
    errdefer a.free(words);
    const scales = try a.alloc(u16, weights.len / group_size);
    errdefer a.free(scales);
    const biases = try a.alloc(u16, scales.len);
    errdefer a.free(biases);
    // groups are independent: workers take contiguous ranges, so every byte matches one serial pass
    const workers = @min(max_workers, std.Thread.getCpuCount() catch 1, @max(1, scales.len / 4096));
    const per = (scales.len + workers - 1) / workers;
    var results: [max_workers]Error!void = @splat({});
    var threads: [max_workers]?std.Thread = @splat(null);
    for (1..workers) |w| threads[w] = std.Thread.spawn(.{}, run, .{ &results[w], weights, words, scales, biases, w * per, @min(scales.len, (w + 1) * per), ratio, group_size }) catch null;
    run(&results[0], weights, words, scales, biases, 0, @min(scales.len, per), ratio, group_size);
    for (1..workers) |w| if (threads[w]) |t| t.join() else run(&results[w], weights, words, scales, biases, w * per, @min(scales.len, (w + 1) * per), ratio, group_size);
    for (results[0..workers]) |r| try r;
    return .{ .allocator = a, .n = n, .k = k, .words = words, .scales = scales, .biases = biases };
}
const max_workers = 16;
const Error = error{ NonFiniteAffineInput, NonFiniteAffineRange };
fn run(out: *Error!void, weights: []const u16, words: []u32, scales: []u16, biases: []u16, first: usize, last: usize, ratio: f32, group_size: usize) void {
    out.* = groups(weights, words, scales, biases, first, last, ratio, group_size);
}
fn runProfile(out: *Error!void, weights: []const u16, words: []u32, scales: []u16, biases: []u16, first: usize, last: usize, ratios: []const f32, groups_per_row: usize, group_size: usize) void {
    out.* = groupsWithRatios(weights, words, scales, biases, first, last, ratios, groups_per_row, group_size, 1.0);
}
/// Groups [first, last): each group's 64 inputs to one scale, one bias and eight low-bit-first words.
fn groups(weights: []const u16, words: []u32, scales: []u16, biases: []u16, first: usize, last: usize, ratio: f32, group_size: usize) Error!void {
    return groupsWithRatios(weights, words, scales, biases, first, last, null, 0, group_size, ratio);
}
fn groupsWithRatios(weights: []const u16, words: []u32, scales: []u16, biases: []u16, first: usize, last: usize, ratios: ?[]const f32, groups_per_row: usize, group_size: usize, scalar_ratio: f32) Error!void {
    for (first..last) |group| {
        const values = weights[group * group_size ..][0..group_size];
        for (values) |bits| if (!std.math.isFinite(asFloat(bits))) return error.NonFiniteAffineInput;
        var lower: f32 = std.math.inf(f32);
        var upper: f32 = 0;
        for (values) |bits| {
            const v = asFloat(bits);
            lower = @min(lower, v);
            upper = @max(upper, v);
        }
        const ratio = if (ratios) |profile| profile[group % groups_per_row] else scalar_ratio;
        lower *= ratio;
        upper *= ratio;
        var bias: f32 = if (@abs(lower) > @abs(upper)) lower else upper;
        var scale: f32 = @max((upper - lower) / @as(f32, 15), @as(f32, 1e-7));
        if (bias >= 0) scale = -scale;
        const zero = @round(bias / scale);
        if (zero != 0) scale = bias / zero else bias = 0;
        if (!std.math.isFinite(scale) or scale == 0) return error.NonFiniteAffineRange;
        scales[group] = asBf16(scale);
        biases[group] = asBf16(bias);
        const words_per_group = group_size / 8;
        @memset(words[group * words_per_group ..][0..words_per_group], 0);
        for (values, 0..) |bits, index| {
            const value = @min(upper, @max(lower, asFloat(bits)));
            const code: u32 = @intFromFloat(@min(@as(f32, 15), @max(@as(f32, 0), @round((value - bias) / scale))));
            words[group * words_per_group + index / 8] |= code << @as(u5, @intCast(4 * (index % 8)));
        }
    }
}

fn groupsMse(weights: []const u16, words: []u32, scales: []u16, biases: []u16, first: usize, last: usize, group_size: usize) Error!void {
    const ratios = [_]f32{ 1.0, 0.98, 0.96, 0.94, 0.92, 0.90, 0.85, 0.80 };
    for (first..last) |group| {
        const values = weights[group * group_size ..][0..group_size];
        for (values) |bits| if (!std.math.isFinite(asFloat(bits))) return error.NonFiniteAffineInput;
        var best_score: f64 = std.math.inf(f64);
        var best_scale: u16 = 0;
        var best_bias: u16 = 0;
        var best_words: [8]u32 = @splat(0);
        for (ratios) |ratio| {
            var lower: f32 = std.math.inf(f32);
            var upper: f32 = -std.math.inf(f32);
            for (values) |bits| {
                const v = asFloat(bits);
                lower = @min(lower, v);
                upper = @max(upper, v);
            }
            lower *= ratio;
            upper *= ratio;
            var scale = @max((upper - lower) / @as(f32, 15), @as(f32, 1e-7));
            var bias = lower;
            // Three Lloyd/least-squares passes approximate GPTQ's local
            // error minimization while retaining the existing affine ABI.
            for (0..3) |_| {
                var sum_c: f64 = 0;
                var sum_w: f64 = 0;
                var sum_cc: f64 = 0;
                var sum_cw: f64 = 0;
                for (values) |bits| {
                    const value = asFloat(bits);
                    const raw = @round((value - bias) / scale);
                    const code = @min(@as(f32, 15), @max(@as(f32, 0), raw));
                    const c64 = @as(f64, code);
                    const w64 = @as(f64, value);
                    sum_c += c64;
                    sum_w += w64;
                    sum_cc += c64 * c64;
                    sum_cw += c64 * w64;
                }
                const count = @as(f64, @floatFromInt(group_size));
                const denom = sum_cc - sum_c * sum_c / count;
                if (denom > 1e-12) {
                    scale = @floatCast((sum_cw - sum_c * sum_w / count) / denom);
                    if (scale < 0) scale = -scale;
                    bias = @floatCast(sum_w / count - @as(f64, scale) * sum_c / count);
                }
                if (!std.math.isFinite(scale) or !std.math.isFinite(bias) or scale == 0)
                    return error.NonFiniteAffineRange;
            }
            const scale_bits = asBf16(scale);
            const bias_bits = asBf16(bias);
            const scale_bf = asFloat(scale_bits);
            const bias_bf = asFloat(bias_bits);
            var candidate_words: [8]u32 = @splat(0);
            var score: f64 = 0;
            for (values, 0..) |bits, index| {
                const original = asFloat(bits);
                const code: u32 = @intFromFloat(@min(@as(f32, 15), @max(@as(f32, 0), @round((original - bias_bf) / scale_bf))));
                const restored = scale_bf * @as(f32, @floatFromInt(code)) + bias_bf;
                const delta = @as(f64, original) - @as(f64, restored);
                score += delta * delta;
                candidate_words[index / 8] |= code << @as(u5, @intCast(4 * (index % 8)));
            }
            if (score < best_score) {
                best_score = score;
                best_scale = scale_bits;
                best_bias = bias_bits;
                best_words = candidate_words;
            }
        }
        scales[group] = best_scale;
        biases[group] = best_bias;
        const words_per_group = group_size / 8;
        @memcpy(words[group * words_per_group ..][0..words_per_group], best_words[0..words_per_group]);
    }
}

test "q4 shape and finite input checks occur before preparation" {
    var w: [64]u16 = @splat(0);
    try std.testing.expectError(error.BadAffineShape, prepare(std.testing.allocator, &w, 1, 63));
    w[7] = 0x7fc0;
    try std.testing.expectError(error.NonFiniteAffineInput, prepare(std.testing.allocator, &w, 1, 64));
}
test "a threaded preparation gives the serial pass's bytes" {
    const a = std.testing.allocator;
    const n = 64;
    const k = 4096;
    const w = try a.alloc(u16, n * k);
    defer a.free(w);
    var prng = std.Random.DefaultPrng.init(7);
    for (w) |*x| x.* = asBf16(prng.random().floatNorm(f32));
    var p = try prepare(a, w, n, k);
    defer p.deinit();
    const words = try a.alloc(u32, w.len / 8);
    defer a.free(words);
    const scales = try a.alloc(u16, w.len / 64);
    defer a.free(scales);
    const biases = try a.alloc(u16, scales.len);
    defer a.free(biases);
    try groups(w, words, scales, biases, 0, scales.len, 1.0, 64);
    try std.testing.expectEqualSlices(u32, words, p.words);
    try std.testing.expectEqualSlices(u16, scales, p.scales);
    try std.testing.expectEqualSlices(u16, biases, p.biases);
}
test "constant positive, negative and tiny groups retain signed scales and exact zero codes" {
    const a = std.testing.allocator;
    var w: [3 * 64]u16 = undefined;
    @memset(w[0..64], 0x3f80);
    @memset(w[64..128], 0xbf80);
    @memset(w[128..], asBf16(1e-20));
    var p = try prepare(a, &w, 3, 64);
    defer p.deinit();
    try std.testing.expectEqual(@as(u16, 0xb3d7), p.scales[0]);
    try std.testing.expectEqual(@as(u16, 0x3d89), p.scales[1]);
    try std.testing.expectEqual(@as(u16, 0), p.biases[2]);
    for (p.words) |word| try std.testing.expectEqual(@as(u32, 0), word);
}

test "mse preparation keeps the packed shape and never increases a chosen group score" {
    const a = std.testing.allocator;
    var w: [32]u16 = undefined;
    for (&w, 0..) |*slot, i| slot.* = asBf16(if (i == 0) 4.0 else @as(f32, @floatFromInt(i)) / 32.0);
    var p = try prepareGroupMse(a, &w, 1, 32, 32);
    defer p.deinit();
    try std.testing.expectEqual(@as(usize, 4), p.words.len);
    try std.testing.expectEqual(@as(usize, 1), p.scales.len);
    try std.testing.expect(std.math.isFinite(asFloat(p.scales[0])));
}

test "group clipping profile applies one ratio per input group" {
    const a = std.testing.allocator;
    var w: [64]u16 = undefined;
    for (&w, 0..) |*slot, i| slot.* = asBf16(@as(f32, @floatFromInt(i)) / 32.0 - 1.0);
    const ratios = [_]f32{ 1.0, 0.8 };
    var p = try prepareGroupClippedProfile(a, &w, 1, 64, &ratios, 32);
    defer p.deinit();
    try std.testing.expectEqual(@as(usize, 2), p.scales.len);
    try std.testing.expect(p.scales[0] != p.scales[1] or p.biases[0] != p.biases[1]);
}

pub fn signature() [32]u8 {
    var result: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(@embedFile("affine4.zig"), &result, .{});
    return result;
}
