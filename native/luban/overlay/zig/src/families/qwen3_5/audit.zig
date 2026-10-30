// SPDX-License-Identifier: Apache-2.0
//! Isolated research audit of live quantized Q4 and its original retained source.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const w = @import("weights.zig");
const c = @import("config.zig");
pub const Projection = enum { gate, up, down };
pub const Result = struct { layer: usize, projection: Projection, inputs: usize, outputs: usize, group_size: usize, channel_indices: [16]usize, rows: [2]usize = .{ 1, 9 }, directory: []const u8, stem: []const u8 };

fn finite(values: []const u16) !void {
    for (values) |v| if (v & 0x7f80 == 0x7f80) return error.NonfiniteAudit;
}
fn create(io: std.Io, gpa: std.mem.Allocator, directory: []const u8, stem: []const u8, suffix: []const u8) !std.Io.File {
    const path = try std.fmt.allocPrint(gpa, "{s}/{s}.{s}", .{ directory, stem, suffix });
    defer gpa.free(path);
    return std.Io.Dir.cwd().createFile(io, path, .{ .exclusive = true });
}
fn sample(io: std.Io, gpa: std.mem.Allocator, directory: []const u8, stem: []const u8, suffix: []const u8, tensor: w.Tensor, row_bytes: usize, channels: [16]usize) !void {
    const file = try create(io, gpa, directory, stem, suffix);
    defer file.close(io);
    const bytes = tensor.host(u8);
    for (channels) |channel| try file.writeStreamingAll(io, bytes[channel * row_bytes ..][0..row_bytes]);
}

pub fn run(m: *Model, io: std.Io, directory: []const u8, stem: []const u8, layer: usize, projection: Projection, input_path: []const u8) !Result {
    if (layer >= c.layers or !std.fs.path.isAbsolute(input_path) or stem.len == 0 or stem.len > 128) return error.InvalidAuditRequest;
    for (stem) |ch| if (!std.ascii.isAlphanumeric(ch) and ch != '-' and ch != '_') return error.InvalidAuditRequest;
    const original = &(m.original orelse return error.OriginalWeightsNotRetained).weights.blocks[layer];
    const live = &m.weights.blocks[layer];
    const source = switch (projection) { .gate => original.gate, .up => original.up, .down => original.down };
    const quantized = switch (projection) { .gate => live.gate, .up => live.up, .down => live.down };
    if (!source.dense or quantized.dense or (quantized.bits != 32 and quantized.bits != 16) or quantized.residual != null or m.residual_rank != 0) return error.AuditRequiresPlainQ4Group16Or32;
    const group_size: usize = quantized.bits;
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, input_path, m.gpa, .limited(9 * c.intermediate * 2 + 1));
    defer m.gpa.free(raw);
    if (raw.len != 9 * quantized.inputs * 2) return error.InvalidAuditInputLength;
    var at: usize = 0;
    while (at < raw.len) : (at += 2) if (std.mem.readInt(u16, raw[at..][0..2], .little) & 0x7f80 == 0x7f80) return error.NonfiniteAudit;
    const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const x = try m.device.buffer(raw.len, options);
    defer x.deinit();
    @memcpy(x.contents()[0..raw.len], raw);
    const y = try m.device.buffer(9 * quantized.outputs * 2, options);
    defer y.deinit();
    var channels: [16]usize = undefined;
    for (&channels, 0..) |*channel, i| channel.* = i * (quantized.outputs - 1) / 15;
    try sample(io, m.gpa, directory, stem, "original.bf16", source.weight, quantized.inputs * 2, channels);
    try sample(io, m.gpa, directory, stem, "words.u32", quantized.weight, quantized.inputs / 2, channels);
    try sample(io, m.gpa, directory, stem, "scales.bf16", quantized.scales, quantized.inputs / group_size * 2, channels);
    try sample(io, m.gpa, directory, stem, "biases.bf16", quantized.biases, quantized.inputs / group_size * 2, channels);
    for ([_]usize{ 1, 9 }, [_][]const u8{ "y1.bf16", "y9.bf16" }) |rows, suffix| {
        try @import("forward.zig").auditProjection(m, quantized, x, y, rows);
        const values = y.slice(u16, rows * quantized.outputs);
        try finite(values);
        const file = try create(io, m.gpa, directory, stem, suffix);
        defer file.close(io);
        try file.writeStreamingAll(io, std.mem.sliceAsBytes(values));
    }
    return .{ .layer = layer, .projection = projection, .inputs = quantized.inputs, .outputs = quantized.outputs, .group_size = group_size, .channel_indices = channels, .directory = directory, .stem = stem };
}
