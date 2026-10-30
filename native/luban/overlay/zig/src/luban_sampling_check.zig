// SPDX-License-Identifier: Apache-2.0
//! CPU-only original-vs-bucket keyed sampling equivalence and 248320-token timing.
const std = @import("std");
const current = @import("sampling");
const reference = @import("sampling_reference");
const Time = extern struct { sec: i64, nsec: i64 };
extern "c" fn clock_gettime(clock: i32, time: *Time) c_int;

fn nanos() !i64 {
    var time: Time = undefined;
    if (clock_gettime(6, &time) != 0) return error.ClockFailed;
    return time.sec * 1000000000 + time.nsec;
}

fn value(word: u16) f64 {
    return @as(f32, @bitCast(@as(u32, word) << 16));
}

fn setting(s: current.Sampling) reference.Sampling {
    return .{ .seed = s.seed, .temperature = s.temperature, .top_k = s.top_k, .top_p = s.top_p, .min_p = s.min_p };
}

// The full-vocabulary branch of the original backend.draw, including validation,
// BF16 expansion and original arrays, calls the independently imported old choose.
fn original(gpa: std.mem.Allocator, words: []const u16, position: u64, s: current.Sampling) !u64 {
    var best: usize = 0;
    for (words, 0..) |word, id| {
        const x = value(word);
        if (!std.math.isFinite(x)) return error.NonfiniteQwenLogits;
        if (x > value(words[best])) best = id;
    }
    if (s.temperature <= 0) return best;
    const values = try gpa.alloc(f64, words.len);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, words.len);
    defer gpa.free(ids);
    for (words, values, ids, 0..) |word, *v, *id, i| {
        v.* = value(word);
        id.* = i;
    }
    return reference.choose(gpa, values, ids, position, setting(s));
}

const Mode = enum { random, ties, zeros, encodings };
fn fill(words: []u16, seed: u64, mode: Mode) void {
    var rng = std.Random.DefaultPrng.init(seed);
    const random = rng.random();
    for (words, 0..) |*word, i| {
        switch (mode) {
            .random => {
                const f: f32 = @as(f32, @floatFromInt(random.intRangeAtMost(i32, -16384, 16384))) / 1024.0;
                const bits: u32 = @bitCast(f);
                word.* = @truncate((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
            },
            .ties => word.* = if (i % 3 == 0) 0x3f80 else 0x3f00,
            .zeros => word.* = if (i % 2 == 0) 0x8000 else 0,
            .encodings => {
                var raw: u16 = @truncate(i);
                if (raw & 0x7f80 == 0x7f80) raw ^= 0x80;
                word.* = raw;
            },
        }
    }
}

fn check(gpa: std.mem.Allocator) !usize {
    var count: usize = 0;
    const short = try gpa.alloc(u16, 513);
    defer gpa.free(short);
    for ([_]u64{ 0, 42, std.math.maxInt(u64) }) |seed| {
        for ([_]Mode{ .random, .ties, .zeros }) |mode| {
            fill(short, seed, mode);
            for ([_]f64{ 0.1, 0.6, 1, 2 }) |temperature| for ([_]f64{ 0, 0.000001, 0.5, 0.95, 1 }) |top_p| for ([_]f64{ 0, 0.1, 1 }) |min_p| for ([_]u64{ 0, 17, 65536 }) |position| {
                const s = current.Sampling{ .seed = seed, .top_k = 0, .temperature = temperature, .top_p = top_p, .min_p = min_p };
                try std.testing.expectEqual(try original(gpa, short, position, s), try current.chooseBf16(gpa, short, position, s));
                count += 1;
            };
        }
    }
    const words = try gpa.alloc(u16, 248320);
    defer gpa.free(words);
    for ([_]u64{ 42, 20261015 }) |seed| {
        fill(words, seed, .random);
        for ([_]f64{ 0.1, 0.6, 1, 2 }) |temperature| for ([_]f64{ 0, 0.5, 0.95, 1 }) |top_p| {
            const s = current.Sampling{ .seed = seed, .top_k = 0, .temperature = temperature, .top_p = top_p };
            try std.testing.expectEqual(try original(gpa, words, 1024, s), try current.chooseBf16(gpa, words, 1024, s));
            count += 1;
        };
    }
    fill(words, 42, .encodings);
    for ([_]f64{ 0.1, 0.6, 1, 2 }) |temperature| {
        const s = current.Sampling{ .seed = 42, .top_k = 0, .temperature = temperature };
        try std.testing.expectEqual(try original(gpa, words, 7, s), try current.chooseBf16(gpa, words, 7, s));
        count += 1;
    }
    // Nonfinite validation includes both signs of infinities and NaN payloads.
    for ([_]u16{ 0x7f80, 0xff80, 0x7f81, 0x7fc0, 0xffc1 }) |nonfinite| {
        short[256] = nonfinite;
        const s = current.Sampling{ .seed = 42, .top_k = 0 };
        try std.testing.expectError(error.NonfiniteQwenLogits, current.chooseBf16(gpa, short, 0, s));
        try std.testing.expectError(error.NonfiniteQwenLogits, original(gpa, short, 0, s));
        count += 1;
    }
    return count;
}

test "BF16 stable bucket sampler exactly preserves independent original keyed draws" {
    _ = try check(std.testing.allocator);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const cases = try check(gpa);
    std.debug.print("{{\"equivalent_cases\":{d},\"vocab\":248320,\"cpu_only\":true}}\n", .{cases});
    const words = try gpa.alloc(u16, 248320);
    defer gpa.free(words);
    var old: [100]i64 = undefined;
    var newer: [100]i64 = undefined;
    var total_old: f64 = 0;
    var total_new: f64 = 0;
    for (0..100) |round| {
        fill(words, round + 42, .random);
        const s = current.Sampling{ .seed = round + 7, .top_k = 0, .temperature = 0.6, .top_p = 0.95 };
        var want: u64 = undefined;
        var actual: u64 = undefined;
        if (round % 2 == 0) {
            var start = try nanos();
            want = try original(gpa, words, round, s);
            old[round] = (try nanos()) - start;
            start = try nanos();
            actual = try current.chooseBf16(gpa, words, round, s);
            newer[round] = (try nanos()) - start;
        } else {
            var start = try nanos();
            actual = try current.chooseBf16(gpa, words, round, s);
            newer[round] = (try nanos()) - start;
            start = try nanos();
            want = try original(gpa, words, round, s);
            old[round] = (try nanos()) - start;
        }
        try std.testing.expectEqual(want, actual);
        total_old += @floatFromInt(old[round]);
        total_new += @floatFromInt(newer[round]);
        std.debug.print("{{\"round\":{d},\"reference_ns\":{d},\"candidate_ns\":{d},\"token\":{d},\"equal\":true}}\n", .{ round, old[round], newer[round], actual });
    }
    std.mem.sort(i64, &old, {}, std.sort.asc(i64));
    std.mem.sort(i64, &newer, {}, std.sort.asc(i64));
    std.debug.print("{{\"rounds\":100,\"reference_mean_ms\":{d:.6},\"candidate_mean_ms\":{d:.6},\"mean_speedup\":{d:.6},\"reference_median_ns\":{d},\"candidate_median_ns\":{d}}}\n", .{ total_old / 1e8, total_new / 1e8, total_old / total_new, old[50], newer[50] });
}
