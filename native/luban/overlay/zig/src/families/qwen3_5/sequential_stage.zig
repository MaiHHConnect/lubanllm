//! Small research-stage carrier; final deployment uses the separate packed v2 format.
const std = @import("std");
const mtl = @import("metal");
const st = @import("state.zig");
const w = @import("weights.zig");
const packed_q4 = @import("packed.zig");
pub const header_bytes = 160;
pub const payload_bytes = 2048 * 6144 * 3 / 4;

pub fn validateHeader(raw: []const u8, manifest: []const u8, layer: usize, mask: u32, revision: usize) !void {
    if (raw.len != 160 or manifest.len != 64 or layer >= 24 or revision != layer or
        (mask != 3 and mask != 4) or (layer == 23 and mask == 4)) return error.InvalidSequentialStage;
    const zeros: [64]u8 = @splat('0');
    if (!std.mem.eql(u8, raw[0..8], "LQ4ST01\n") or
        std.mem.readInt(u32, raw[8..12], .little) != layer or
        std.mem.readInt(u32, raw[12..16], .little) != mask or
        std.mem.readInt(u32, raw[16..20], .little) != @popCount(mask) or
        std.mem.readInt(u32, raw[20..24], .little) != 16 or
        !std.mem.eql(u8, raw[24..88], manifest) or !std.mem.eql(u8, raw[88..152], &zeros) or
        std.mem.readInt(u64, raw[152..160], .little) != revision) return error.InvalidSequentialStage;
}

pub const Installed = struct {
    memory: st.Storage,
    projections: [3]?w.Linear = @splat(null),
    pub fn deinit(self: *Installed) void {
        self.memory.deinit();
        self.* = undefined;
    }
    pub fn read(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, path: []const u8, manifest: []const u8, layer: usize, mask: u32, revision: usize) !Installed {
        if (!std.fs.path.isAbsolute(path)) return error.InvalidSequentialStage;
        const file = try std.Io.Dir.cwd().openFile(io, path, .{ .follow_symlinks = false });
        defer file.close(io);
        const stat = try file.stat(io);
        const expected = header_bytes + @as(usize, @popCount(mask)) * (8 + payload_bytes);
        if (stat.kind != .file or stat.size != expected) return error.InvalidSequentialStage;
        var raw: [160]u8 = undefined;
        if (try file.readPositionalAll(io, &raw, 0) != raw.len) return error.InvalidSequentialStage;
        try validateHeader(&raw, manifest, layer, mask, revision);
        var result: Installed = .{ .memory = .{ .gpa = gpa, .device = device } };
        errdefer result.deinit();
        var at: u64 = 160;
        for (0..3) |index| {
            if (mask & (@as(u32, 1) << @intCast(index)) == 0) continue;
            const n: usize = if (index == 2) 2048 else 6144;
            const k: usize = if (index == 2) 6144 else 2048;
            var dims: [8]u8 = undefined;
            if (try file.readPositionalAll(io, &dims, at) != 8 or std.mem.readInt(u32, dims[0..4], .little) != n or std.mem.readInt(u32, dims[4..8], .little) != k) return error.InvalidSequentialStage;
            at += 8;
            const b = try result.memory.alloc(payload_bytes);
            if (try file.readPositionalAll(io, b.contents()[0..payload_bytes], at) != payload_bytes) return error.InvalidSequentialStage;
            at += payload_bytes;
            const words = n * k / 2;
            const meta = n * k / 16 * 2;
            try packed_q4.validateMetadata(b.contents()[words..][0..meta], b.contents()[words + meta ..][0..meta]);
            result.projections[index] = .{
                .weight = .{ .buffer = b, .offset = 0, .bytes = words, .dtype = .u32, .rank = 2, .shape = .{ n, k / 8, 1, 1 } },
                .scales = .{ .buffer = b, .offset = words, .bytes = meta, .dtype = .bf16, .rank = 2, .shape = .{ n, k / 16, 1, 1 } },
                .biases = .{ .buffer = b, .offset = words + meta, .bytes = meta, .dtype = .bf16, .rank = 2, .shape = .{ n, k / 16, 1, 1 } },
                .inputs = k,
                .outputs = n,
                .bits = 16,
            };
        }
        var extra: [1]u8 = undefined;
        if (at != expected or try file.readPositionalAll(io, &extra, at) != 0 or (try file.stat(io)).size != expected) return error.InvalidSequentialStage;
        return result;
    }
};

pub fn activationHeader(manifest: []const u8, layer: usize, kind: usize, prompt: usize, forced: usize) ![160]u8 {
    if (manifest.len != 64 or layer >= 24 or kind < 1 or kind > 3 or prompt == 0 or forced > 255 or prompt + forced > 2048) return error.InvalidSequentialActivation;
    for (manifest) |ch| if (!std.ascii.isDigit(ch) and !(ch >= 'a' and ch <= 'f')) return error.InvalidSequentialActivation;
    var raw: [160]u8 = undefined;
    @memcpy(raw[0..8], "LNQLA01\n");
    for ([_]usize{ layer, kind, prompt + forced, if (kind == 2) 6144 else 4096, prompt, forced }, 0..) |value, i| std.mem.writeInt(u32, raw[8 + 4 * i ..][0..4], @intCast(value), .little);
    @memcpy(raw[32..96], manifest);
    @memset(raw[96..160], '0');
    return raw;
}

pub fn finite(values: []const u16) !void {
    for (values) |v| if (v & 0x7f80 == 0x7f80) return error.NonfiniteSequentialActivation;
}

test "stage header rejects phase layer binding revision count and down23" {
    const manifest: [64]u8 = @splat('a');
    var raw: [160]u8 = @splat(0);
    @memcpy(raw[0..8], "LQ4ST01\n");
    std.mem.writeInt(u32, raw[12..16], 3, .little);
    std.mem.writeInt(u32, raw[16..20], 2, .little);
    std.mem.writeInt(u32, raw[20..24], 16, .little);
    @memcpy(raw[24..88], &manifest);
    @memset(raw[88..152], '0');
    try validateHeader(&raw, &manifest, 0, 3, 0);
    try std.testing.expectError(error.InvalidSequentialStage, validateHeader(&raw, &manifest, 0, 4, 0));
    try std.testing.expectError(error.InvalidSequentialStage, validateHeader(&raw, &manifest, 0, 3, 1));
    try std.testing.expectError(error.InvalidSequentialStage, validateHeader(&raw, &manifest, 23, 4, 23));
    raw[88] = '1';
    try std.testing.expectError(error.InvalidSequentialStage, validateHeader(&raw, &manifest, 0, 3, 0));
}
test "activation carrier dimensions and finite values" {
    const manifest: [64]u8 = @splat('b');
    const h = try activationHeader(&manifest, 2, 1, 33, 5);
    try std.testing.expectEqual(@as(u32, 38), std.mem.readInt(u32, h[16..20], .little));
    try std.testing.expectEqual(@as(u32, 4096), std.mem.readInt(u32, h[20..24], .little));
    try finite(&.{ 0, 0xbf80, 0x7f7f });
    try std.testing.expectError(error.NonfiniteSequentialActivation, finite(&.{0x7f80}));
    try std.testing.expectError(error.InvalidSequentialActivation, activationHeader(&manifest, 0, 1, 1, 256));
}
