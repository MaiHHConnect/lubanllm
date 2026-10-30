// SPDX-License-Identifier: Apache-2.0
//! Strict, source-bound research sidecar; no quantized/base/core file is modified.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const w = @import("weights.zig");
const c = @import("config.zig");

pub fn attach(checkpoint: *ck.Checkpoint, device: mtl.Device, weights: *w.Weights, io: std.Io, path: []const u8, manifest: []const u8, core: []const u8) !usize {
    if (path.len == 0 or path[0] != '/' or manifest.len != 64 or core.len != 64) return error.InvalidResidualSidecar;
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, checkpoint.allocator, .limited(40 << 20));
    defer checkpoint.allocator.free(raw);
    if (raw.len < 140 or !std.mem.eql(u8, raw[0..8], "LQ4RES1\n")) return error.InvalidResidualSidecar;
    const rank: usize = std.mem.readInt(u32, raw[8..12], .little);
    if (rank != 8 and rank != 32) return error.InvalidResidualRank;
    if (!std.mem.eql(u8, raw[12..76], manifest) or !std.mem.eql(u8, raw[76..140], core)) return error.ResidualSourceMismatch;
    const size = (c.layers * 3 - 1) * rank * (c.hidden + c.intermediate) * 2;
    if (raw.len != 140 + size) return error.InvalidResidualSidecar;
    var at: usize = 140;
    while (at < raw.len) : (at += 2) {
        const value = std.mem.readInt(u16, raw[at..][0..2], .little);
        if (value & 0x7f80 == 0x7f80) return error.NonfiniteResidual;
    }
    const buffer = try device.buffer(size, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer buffer.deinit();
    @memcpy(buffer.contents()[0..size], raw[140..]);
    var next = weights.*;
    at = 0;
    for (&next.blocks, 0..) |*block, layer| {
        for ([_]*w.Linear{ &block.gate, &block.up, &block.down }, 0..) |linear, projection| {
            if (layer == c.layers - 1 and projection == 2) continue;
            if (linear.dense or linear.bits != 32 or linear.residual != null) return error.ResidualRequiresQ4Group32;
            const b_size = rank * linear.inputs * 2;
            const u_size = linear.outputs * rank * 2;
            linear.residual = .{
                .basis = .{ .buffer = buffer, .offset = at, .bytes = b_size, .dtype = .bf16, .rank = 2, .shape = .{ rank, linear.inputs, 1, 1 } },
                .update = .{ .buffer = buffer, .offset = at + b_size, .bytes = u_size, .dtype = .bf16, .rank = 2, .shape = .{ linear.outputs, rank, 1, 1 } },
                .rank = rank,
            };
            at += b_size + u_size;
        }
    }
    if (at != size) return error.InvalidResidualSidecar;
    try checkpoint.own(buffer, size);
    weights.* = next;
    return rank;
}
