// SPDX-License-Identifier: Apache-2.0
//! One disk-backed layer store and one reusable Metal slot. No requantization.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const w = @import("weights.zig");
const c = @import("config.zig");

const Range = struct { offset: usize, bytes: usize };

pub const Pager = struct {
    fd: std.c.fd_t,
    slot: mtl.Buffer,
    ranges: [c.layers]Range,
    store_bytes: usize,
    slot_bytes: usize,
    read_bytes: u64 = 0,
    read_calls: u64 = 0,
    loaded_layer: ?usize = null,

    /// `next` initially aliases the original weights. Only the caller commits it
    /// after this succeeds. The exclusive file is unlinked immediately: its live
    /// descriptor remains disk-backed, and crashes cannot leave cache garbage.
    pub fn init(device: mtl.Device, path: [:0]const u8, next: *w.Weights, globals: *ck.Checkpoint) !Pager {
        const fd = std.c.open(path, .{ .ACCMODE = .RDWR, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) return error.LayerStoreCreateFailed;
        errdefer _ = std.c.close(fd);
        if (std.c.unlink(path) != 0) return error.LayerStoreUnlinkFailed;
        var ranges: [c.layers]Range = undefined;
        var offset: usize = 0;
        var maximum: usize = 0;
        for (&next.blocks, 0..) |*layer, i| {
            var exporter = Export{ .fd = fd, .base = offset };
            try w.visitLayer(layer, &exporter);
            ranges[i] = .{ .offset = offset, .bytes = exporter.offset };
            maximum = @max(maximum, exporter.offset);
            offset = try std.math.add(usize, offset, exporter.offset);
        }
        if (maximum == 0 or std.c.lseek(fd, 0, std.c.SEEK.END) != @as(std.c.off_t, @intCast(offset)))
            return error.InvalidLayerStoreSize;
        const slot = try device.buffer(maximum, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer slot.deinit();
        var binder = Bind{ .buffer = slot };
        for (&next.blocks) |*layer| try w.visitLayer(layer, &binder);
        var copier = Copy{ .device = device, .checkpoint = globals };
        try w.visitLinear(&next.embedding, &copier);
        try copier.tensor(&next.norm);
        return .{ .fd = fd, .slot = slot, .ranges = ranges, .store_bytes = offset, .slot_bytes = maximum };
    }

    /// The caller must wait for all GPU use of the previous layer before this.
    /// Failed/partial reads never mark the slot ready; the request must abort.
    pub fn acquire(self: *Pager, layer: usize) !void {
        if (layer >= c.layers) return error.InvalidPagedLayer;
        if (self.loaded_layer == layer) return;
        self.loaded_layer = null;
        const range = self.ranges[layer];
        try readAll(self.fd, self.slot.contents()[0..range.bytes], range.offset);
        self.read_bytes += range.bytes;
        self.read_calls += 1;
        self.loaded_layer = layer;
    }

    pub fn deinit(self: *Pager) void {
        self.slot.deinit();
        _ = std.c.close(self.fd);
    }
};

const Export = struct {
    fd: std.c.fd_t,
    base: usize,
    offset: usize = 0,

    pub fn tensor(self: *Export, t: *w.Tensor) !void {
        // Metal tensor offsets retain at least the original scalar alignment.
        self.offset = std.mem.alignForward(usize, self.offset, 16);
        try writeAll(self.fd, t.host(u8), try std.math.add(usize, self.base, self.offset));
        t.offset = self.offset;
        self.offset = try std.math.add(usize, self.offset, t.bytes);
    }
};

const Bind = struct {
    buffer: mtl.Buffer,
    pub fn tensor(self: *Bind, t: *w.Tensor) !void {
        if (t.offset > self.buffer.length() or t.bytes > self.buffer.length() - t.offset)
            return error.InvalidPagedTensor;
        t.buffer = self.buffer;
    }
};

const Copy = struct {
    device: mtl.Device,
    checkpoint: *ck.Checkpoint,
    pub fn tensor(self: *Copy, t: *w.Tensor) !void {
        const buffer = try self.device.buffer(t.bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer buffer.deinit();
        @memcpy(buffer.contents()[0..t.bytes], t.host(u8));
        try self.checkpoint.own(buffer, t.bytes);
        t.buffer = buffer;
        t.offset = 0;
    }
};

fn readAll(fd: std.c.fd_t, dest: []u8, offset: usize) !void {
    var done: usize = 0;
    while (done < dest.len) {
        const n = std.c.pread(fd, dest.ptr + done, dest.len - done, @intCast(offset + done));
        if (n < 0 and std.c._errno().* == @intFromEnum(std.c.E.INTR)) continue;
        if (n <= 0) return error.LayerStoreShortRead;
        done += @intCast(n);
    }
}

fn writeAll(fd: std.c.fd_t, source: []const u8, offset: usize) !void {
    var done: usize = 0;
    while (done < source.len) {
        const n = std.c.pwrite(fd, source.ptr + done, source.len - done, @intCast(offset + done));
        if (n < 0 and std.c._errno().* == @intFromEnum(std.c.E.INTR)) continue;
        if (n <= 0) return error.LayerStoreShortWrite;
        done += @intCast(n);
    }
}

test "disk slot reads exact byte ranges and rejects truncated storage" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrintSentinel(std.testing.allocator, ".zig-cache/tmp/{s}/layers.bin", .{tmp.sub_path}, 0);
    defer std.testing.allocator.free(path);
    const fd = std.c.open(path, .{ .ACCMODE = .RDWR, .CREAT = true, .EXCL = true }, @as(std.c.mode_t, 0o600));
    try std.testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    // Existing paths cannot be overwritten, even by an accidental second init.
    const existing = std.c.open(path, .{ .ACCMODE = .RDWR, .CREAT = true, .EXCL = true }, @as(std.c.mode_t, 0o600));
    if (existing >= 0) _ = std.c.close(existing);
    try std.testing.expect(existing < 0);
    try std.testing.expectEqual(@as(c_int, 0), std.c.unlink(path));
    try writeAll(fd, &.{ 1, 2, 3, 4, 5, 6 }, 32);
    var output: [4]u8 = undefined;
    try readAll(fd, &output, 33);
    try std.testing.expectEqualSlices(u8, &.{ 2, 3, 4, 5 }, &output);
    try std.testing.expectError(error.LayerStoreShortRead, readAll(fd, &output, 36));
}
