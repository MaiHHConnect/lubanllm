// SPDX-License-Identifier: Apache-2.0
//! One immutable, parent-qualified last-layer NeuralExpert per worker lifetime.
const std = @import("std");
const mtl = @import("metal");
const st = @import("state.zig");
const c = @import("config.zig");
const w = @import("weights.zig");
const safe = @import("../../core/safetensors.zig");

pub const Payload = struct {
    bytes: []const u8,
    width: usize,
    offsets: [3]usize,
};

// Validate the exact bytes that will enter Metal, with no second file read.
pub fn parse(gpa: std.mem.Allocator, raw: []const u8, approved: [32]u8) !Payload {
    var actual: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(raw, &actual, .{});
    if (!std.mem.eql(u8, &actual, &approved)) return error.ExpertSHAMismatch;
    if (raw.len < 8) return error.BadSafetensors;
    const size = std.mem.readInt(u64, raw[0..8], .little);
    if (size > raw.len - 8) return error.BadSafetensors;
    const start = 8 + @as(usize, @intCast(size));
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const entries = try safe.parseHeader(arena.allocator(), raw[8..start], raw.len - start);
    if (entries.count() != 3) return error.InvalidExpertTensors;
    const gate = entries.get("gate.weight") orelse return error.InvalidExpertTensors;
    const width = gate.shape[0];
    if (width < 32 or width > 512) return error.InvalidExpertWidth;
    var offsets: [3]usize = undefined;
    var covered: usize = 0;
    for ([_][]const u8{ "gate.weight", "up.weight", "down.weight" }, 0..) |name, index| {
        const entry = entries.get(name) orelse return error.InvalidExpertTensors;
        const outputs = if (index == 2) c.hidden else width;
        const inputs = if (index == 2) width else c.hidden;
        if (entry.dtype != .bf16 or entry.rank != 2 or entry.shape[0] != outputs or entry.shape[1] != inputs or
            entry.begin % 2 != 0 or entry.end - entry.begin != outputs * inputs * 2) return error.InvalidExpertTensors;
        offsets[index] = entry.begin;
        for (offsets[0..index]) |other| if (entry.begin < other + width * c.hidden * 2 and other < entry.end)
            return error.OverlappingExpertTensors;
        covered += entry.end - entry.begin;
    }
    if (covered != raw.len - start) return error.InvalidExpertPayload;
    const bytes = raw[start..];
    var at: usize = 0;
    while (at < bytes.len) : (at += 2) {
        if (std.mem.readInt(u16, bytes[at..][0..2], .little) & 0x7f80 == 0x7f80) return error.NonfiniteExpert;
    }
    return .{ .bytes = bytes, .width = width, .offsets = offsets };
}

pub const Expert = struct {
    buffer: mtl.Buffer,
    gate: w.Linear,
    up: w.Linear,
    down: w.Linear,
    act: mtl.Pipeline,
    add: mtl.Pipeline,
    width: usize,
    bytes: usize,
    sha256: [64]u8,

    pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, path: []const u8, approved: [32]u8) !Expert {
        if (!std.fs.path.isAbsolute(path)) return error.InvalidExpertPath;
        const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(8 << 20));
        defer gpa.free(raw);
        const payload = try parse(gpa, raw, approved);
        const buffer = try device.buffer(payload.bytes.len, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer buffer.deinit();
        @memcpy(buffer.contents()[0..payload.bytes.len], payload.bytes);
        var linear: [3]w.Linear = undefined;
        for (&linear, 0..) |*slot, i| {
            const outputs = if (i == 2) c.hidden else payload.width;
            const inputs = if (i == 2) payload.width else c.hidden;
            const tensor = w.Tensor{ .buffer = buffer, .offset = payload.offsets[i], .bytes = outputs * inputs * 2,
                .dtype = .bf16, .rank = 2, .shape = .{ outputs, inputs, 1, 1 } };
            slot.* = try w.denseProjection(tensor, outputs, inputs);
        }
        // Ordinary core-only startup does not compile these optional kernels.
        const library = try mtl.Library.fromSource(device, @import("kernel_sources").qwen35.expert_source, mtl.CompileOptions.mlx());
        defer library.deinit();
        const act = try mtl.Pipeline.init(device, library, "luban_expert_act", false);
        errdefer act.deinit();
        const add = try mtl.Pipeline.init(device, library, "luban_expert_add", false);
        return .{ .buffer = buffer, .gate = linear[0], .up = linear[1], .down = linear[2], .act = act, .add = add,
            .width = payload.width, .bytes = payload.bytes.len, .sha256 = std.fmt.bytesToHex(approved, .lower) };
    }

    pub fn deinit(self: *Expert) void {
        self.act.deinit();
        self.add.deinit();
        self.buffer.deinit();
    }

    pub fn encode(self: *const Expert, enc: anytype, scratch: *st.Scratch, rows: usize) void {
        // Last-layer x remains the post-attention normalized input. The main
        // MLP has finished, so its gate/up/act and attention mix are reusable.
        enc.projection(self.gate, scratch.x, 0, scratch.gate, 0, rows);
        enc.projection(self.up, scratch.x, 0, scratch.up, 0, rows);
        enc.e.setPipeline(self.act);
        enc.e.setBuffer(scratch.gate, 0, 0);
        enc.e.setBuffer(scratch.up, 0, 1);
        enc.e.setBuffer(scratch.act, 0, 2);
        enc.e.setValue(@as(u32, @intCast(self.width)), 3);
        enc.e.dispatchThreads(mtl.Size.of(self.width, rows, 1), mtl.Size.of(256, 1, 1));
        enc.projection(self.down, scratch.act, 0, scratch.mix, 0, rows);
        enc.e.setPipeline(self.add);
        enc.e.setBuffer(scratch.mix, 0, 0);
        enc.e.setBuffer(scratch.r, 0, 1);
        enc.e.setValue(@as(u32, c.hidden), 2);
        enc.e.dispatchThreads(mtl.Size.of(c.hidden, rows, 1), mtl.Size.of(256, 1, 1));
    }
};

fn fixture(gpa: std.mem.Allocator, width: usize) ![]u8 {
    const n = width * c.hidden * 2;
    const header = try std.json.Stringify.valueAlloc(gpa, .{
        .@"gate.weight" = .{ .dtype = "BF16", .shape = [2]usize{ width, c.hidden }, .data_offsets = [2]usize{ 0, n } },
        .@"up.weight" = .{ .dtype = "BF16", .shape = [2]usize{ width, c.hidden }, .data_offsets = [2]usize{ n, 2 * n } },
        .@"down.weight" = .{ .dtype = "BF16", .shape = [2]usize{ c.hidden, width }, .data_offsets = [2]usize{ 2 * n, 3 * n } },
    }, .{});
    defer gpa.free(header);
    const raw = try gpa.alloc(u8, 8 + header.len + 3 * n);
    std.mem.writeInt(u64, raw[0..8], header.len, .little);
    @memcpy(raw[8..][0..header.len], header);
    @memset(raw[8 + header.len ..], 0);
    return raw;
}

fn testHash(raw: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(raw, &out, .{});
    return out;
}

test "expert exact three tensor BF16 widths and finite source binding" {
    const gpa = std.testing.allocator;
    for ([_]usize{ 32, 65, 128, 512 }) |width| {
        const raw = try fixture(gpa, width);
        defer gpa.free(raw);
        const payload = try parse(gpa, raw, testHash(raw));
        try std.testing.expectEqual(width, payload.width);
        try std.testing.expectEqual(3 * width * c.hidden * 2, payload.bytes.len);
        var bad = testHash(raw);
        bad[0] ^= 1;
        try std.testing.expectError(error.ExpertSHAMismatch, parse(gpa, raw, bad));
        raw[raw.len - 1] = 0x7f;
        raw[raw.len - 2] = 0x80;
        try std.testing.expectError(error.NonfiniteExpert, parse(gpa, raw, testHash(raw)));
    }
    for ([_]usize{ 0, 31, 513 }) |width| {
        const raw = try fixture(gpa, width);
        defer gpa.free(raw);
        try std.testing.expectError(error.InvalidExpertWidth, parse(gpa, raw, testHash(raw)));
    }
    const short = [_]u8{ 1, 2, 3 };
    try std.testing.expectError(error.BadSafetensors, parse(gpa, &short, testHash(&short)));
}
