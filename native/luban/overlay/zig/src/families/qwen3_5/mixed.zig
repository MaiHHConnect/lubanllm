// SPDX-License-Identifier: Apache-2.0
//! Inference-only q4/group64 copies; the accepted down23 and scalar/state tensors stay exact.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("../../core/checkpoint_metal.zig");
const affine = @import("../../core/affine4.zig");
const w = @import("weights.zig");
const c = @import("config.zig");

const Copy = struct {
    checkpoint: *ck.Checkpoint,
    device: mtl.Device,
    bits: u8,
    packed_reader: ?*@import("packed.zig").Reader = null,
    quantized: usize = 0,

    fn buffer(self: *Copy, bytes: usize) !mtl.Buffer {
        const b = try self.device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer b.deinit();
        try self.checkpoint.own(b, bytes);
        return b;
    }

    fn tensor(self: *Copy, old: w.Tensor) !w.Tensor {
        var t = old;
        t.buffer = try self.buffer(old.bytes);
        t.offset = 0;
        @memcpy(t.buffer.contents()[0..t.bytes], old.host(u8));
        return t;
    }

    fn linear(self: *Copy, old: w.Linear, quantize: bool, clip_ratio: ?f32, group_size: usize, group_clip_ratio: ?f32, mse_quant: bool, group_clip_profile: ?[]const f32) !w.Linear {
        if (!old.dense) return error.MixedRequiresOriginalDenseWeights;
        if (!quantize) return w.denseProjection(try self.tensor(old.weight), old.outputs, old.inputs);
        if (self.packed_reader) |reader| {
            if (self.bits != 4 or group_size != 16) return error.PackedRequiresQ4Group16;
            const count = old.outputs * old.inputs;
            const words_bytes = count / 2;
            const metadata_bytes = count / 16 * 2;
            const q4_bytes = words_bytes + metadata_bytes * 2;
            const rank = reader.header.residual_rank;
            const b_bytes = rank * old.inputs * 2;
            const u_bytes = old.outputs * rank * 2;
            const b = try self.buffer(q4_bytes + b_bytes + u_bytes);
            const previous_residuals = reader.residual_projections;
            try reader.projection(b.contents()[0 .. q4_bytes + b_bytes + u_bytes], old.outputs, old.inputs);
            self.quantized += 1;
            return .{
                .residual = if (reader.residual_projections > previous_residuals) .{
                    .basis = .{ .buffer = b, .offset = q4_bytes, .bytes = b_bytes, .dtype = .bf16,
                        .rank = 2, .shape = .{ rank, old.inputs, 1, 1 } },
                    .update = .{ .buffer = b, .offset = q4_bytes + b_bytes, .bytes = u_bytes, .dtype = .bf16,
                        .rank = 2, .shape = .{ old.outputs, rank, 1, 1 } },
                    .rank = rank,
                } else null,
                .weight = .{ .buffer = b, .offset = 0, .bytes = words_bytes, .dtype = .u32,
                    .rank = 2, .shape = .{ old.outputs, old.inputs / 8, 1, 1 } },
                .scales = .{ .buffer = b, .offset = words_bytes, .bytes = metadata_bytes, .dtype = .bf16,
                    .rank = 2, .shape = .{ old.outputs, old.inputs / 16, 1, 1 } },
                .biases = .{ .buffer = b, .offset = words_bytes + metadata_bytes, .bytes = metadata_bytes, .dtype = .bf16,
                    .rank = 2, .shape = .{ old.outputs, old.inputs / 16, 1, 1 } },
                .inputs = old.inputs, .outputs = old.outputs, .bits = 16,
            };
        }
        if (self.bits == 8) {
            const count = old.outputs * old.inputs;
            const b = try self.buffer(count + count / c.group * 2);
            const scales = @as([*]u16, @ptrCast(@alignCast(b.contents() + count)))[0 .. count / c.group];
            try @import("../../core/quant8.zig").quantize(old.weight.host(u16), b.contents()[0..count], scales);
            const metadata = w.Tensor{ .buffer = b, .offset = count, .bytes = scales.len * 2, .dtype = .bf16,
                .rank = 2, .shape = .{ old.outputs, old.inputs / c.group, 1, 1 } };
            self.quantized += 1;
            return .{
                .weight = .{ .buffer = b, .offset = 0, .bytes = count, .dtype = .u8,
                    .rank = 2, .shape = .{ old.outputs, old.inputs, 1, 1 } },
                .scales = metadata, .biases = metadata, .bits = 8,
                .inputs = old.inputs, .outputs = old.outputs,
            };
        }
        var prepared = if (group_clip_profile) |ratios| blk: {
            break :blk try affine.prepareGroupClippedProfile(self.checkpoint.allocator, old.weight.host(u16), old.outputs, old.inputs, ratios, group_size);
        } else if (clip_ratio) |ratio| blk: {
            if (group_size == 16 or group_size == 32)
                break :blk try affine.prepareGroupClipped(self.checkpoint.allocator, old.weight.host(u16), old.outputs, old.inputs, ratio, group_size);
            break :blk try affine.prepareClipped(self.checkpoint.allocator, old.weight.host(u16), old.outputs, old.inputs, ratio);
        } else if (mse_quant and group_size == 32) blk: {
            break :blk try affine.prepareGroupMse(self.checkpoint.allocator, old.weight.host(u16), old.outputs, old.inputs, group_size);
        } else if (group_size == 16 or group_size == 32) blk: {
            if (group_clip_ratio) |ratio|
                break :blk try affine.prepareGroupClipped(self.checkpoint.allocator, old.weight.host(u16), old.outputs, old.inputs, ratio, group_size)
            else
                break :blk try affine.prepareGroup(self.checkpoint.allocator, old.weight.host(u16), old.outputs, old.inputs, group_size);
        }
        else if (group_size == 32)
            try affine.prepareGroup(self.checkpoint.allocator, old.weight.host(u16), old.outputs, old.inputs, 32)
        else
            try affine.prepare(self.checkpoint.allocator, old.weight.host(u16), old.outputs, old.inputs);
        defer prepared.deinit();
        const wb = std.mem.sliceAsBytes(prepared.words);
        const sb = std.mem.sliceAsBytes(prepared.scales);
        const bb = std.mem.sliceAsBytes(prepared.biases);
        // Reject metadata overflow/underflow too, not only nonfinite source weights.
        for (prepared.scales) |v| if (!std.math.isFinite(affine.asFloat(v)) or affine.asFloat(v) == 0)
            return error.InvalidMixedScale;
        for (prepared.biases) |v| if (!std.math.isFinite(affine.asFloat(v))) return error.InvalidMixedBias;
        const b = try self.buffer(wb.len + sb.len + bb.len);
        @memcpy(b.contents()[0..wb.len], wb);
        @memcpy((b.contents() + wb.len)[0..sb.len], sb);
        @memcpy((b.contents() + wb.len + sb.len)[0..bb.len], bb);
        self.quantized += 1;
        return .{
            .weight = .{ .buffer = b, .offset = 0, .bytes = wb.len, .dtype = .u32, .rank = 2,
                .shape = .{ old.outputs, old.inputs / 8, 1, 1 } },
            .scales = .{ .buffer = b, .offset = wb.len, .bytes = sb.len, .dtype = .bf16, .rank = 2,
                .shape = .{ old.outputs, old.inputs / group_size, 1, 1 } },
            .biases = .{ .buffer = b, .offset = wb.len + sb.len, .bytes = bb.len, .dtype = .bf16, .rank = 2,
                .shape = .{ old.outputs, old.inputs / group_size, 1, 1 } },
            .inputs = old.inputs, .outputs = old.outputs, .bits = if (group_size == 16) 16 else if (group_size == 32) 32 else 4,
        };
    }
};

fn profileLinear(copy: *Copy, old: w.Linear, quantize: bool, profile: []const f32, index: *usize, group_size: usize, group_clip_ratio: ?f32, mse_quant: bool, group_clip_profile: bool) !w.Linear {
    const group_ratios: ?[]const f32 = if (quantize and group_clip_profile and profile.len != 0) blk: {
        const count = old.inputs / group_size;
        if (old.inputs % group_size != 0 or index.* + count > profile.len) return error.InvalidMixedClipProfile;
        const values = profile[index.* .. index.* + count];
        index.* += count;
        break :blk values;
    } else null;
    const ratio = if (quantize and !group_clip_profile and profile.len != 0) blk: {
        if (index.* >= profile.len) return error.InvalidMixedClipProfile;
        const value = profile[index.*];
        index.* += 1;
        break :blk @as(?f32, value);
    } else null;
    return copy.linear(old, quantize, ratio, group_size, group_clip_ratio, mse_quant, group_ratios);
}

/// Build all owning storage before changing the live graph. On failure the
/// original graph/storage remains intact. No original file is written.
pub fn prepare(checkpoint: *ck.Checkpoint, device: mtl.Device, original: w.Weights, mlp_only: bool, bits: u8, preserve_last_mlp_layers: usize, clip_profile: ?[]const f32, gate_up_only: bool, down_only: bool, group_size: usize, preserve_first_mlp_layers: usize, group_clip_ratio: ?f32, mse_quant: bool, group_clip_profile: bool, packed_reader: ?*@import("packed.zig").Reader) !struct { weights: w.Weights, quantized: usize } {
    if (packed_reader != null and (bits != 4 or group_size != 16 or !mlp_only or preserve_first_mlp_layers != 0 or
        preserve_last_mlp_layers != 0 or gate_up_only or down_only or clip_profile != null or group_clip_ratio != null or
        mse_quant or group_clip_profile)) return error.PackedRequiresAllMLPGroup16;
    if (bits != 4 and bits != 8) return error.UnsupportedMixedBits;
    if (clip_profile) |profile| {
        if (bits != 4 or !mlp_only or preserve_last_mlp_layers > c.layers) return error.InvalidMixedClipProfile;
        if (group_clip_profile and group_size != 32) return error.InvalidMixedClipProfile;
        if (group_clip_profile) {
            if (profile.len == 0) return error.InvalidMixedClipProfile;
        } else {
        // The final layer's down projection is the bound core tensor and is
        // intentionally preserved, so a full MLP profile has 71 entries.
        const expected: usize = if (preserve_last_mlp_layers == 0)
            (c.layers - preserve_last_mlp_layers) * 3 - 1
        else
            (c.layers - preserve_last_mlp_layers) * 3;
            if (profile.len != expected) return error.InvalidMixedClipProfile;
        }
    }
    var copy = Copy{ .checkpoint = checkpoint, .device = device, .bits = bits, .packed_reader = packed_reader };
    var next = original;
    var clip_index: usize = 0;
    const clip = if (clip_profile) |profile| profile else &[_]f32{};
    next.embedding = try profileLinear(&copy, original.embedding, !mlp_only, clip, &clip_index, group_size, group_clip_ratio, mse_quant, group_clip_profile);
    next.norm = try copy.tensor(original.norm);
    for (original.blocks, 0..) |old, i| {
        const block = &next.blocks[i];
        block.input_norm = try copy.tensor(old.input_norm);
        block.post_norm = try copy.tensor(old.post_norm);
        const quantize_mlp = if (packed_reader) |reader| reader.header.containsLayer(i) else
            !mlp_only or (i >= preserve_first_mlp_layers and i + preserve_last_mlp_layers < c.layers);
        block.gate = try profileLinear(&copy, old.gate, quantize_mlp and !down_only, clip, &clip_index, group_size, group_clip_ratio, mse_quant, group_clip_profile);
        block.up = try profileLinear(&copy, old.up, quantize_mlp and !down_only, clip, &clip_index, group_size, group_clip_ratio, mse_quant, group_clip_profile);
        block.down = try profileLinear(&copy, old.down, quantize_mlp and !gate_up_only and i != c.layers - 1, clip, &clip_index, group_size, group_clip_ratio, mse_quant, group_clip_profile);
        block.mixer = switch (old.mixer) {
            .delta => |d| .{ .delta = .{
                .qkv = try profileLinear(&copy, d.qkv, !mlp_only, clip, &clip_index, group_size, group_clip_ratio, mse_quant, group_clip_profile), .z = try profileLinear(&copy, d.z, !mlp_only, clip, &clip_index, group_size, group_clip_ratio, mse_quant, group_clip_profile),
                .a = try profileLinear(&copy, d.a, false, clip, &clip_index, group_size, group_clip_ratio, mse_quant, group_clip_profile), .b = try profileLinear(&copy, d.b, false, clip, &clip_index, group_size, group_clip_ratio, mse_quant, group_clip_profile),
                .out = try profileLinear(&copy, d.out, !mlp_only, clip, &clip_index, group_size, group_clip_ratio, mse_quant, group_clip_profile), .conv = try copy.tensor(d.conv),
                .norm = try copy.tensor(d.norm), .a_log = try copy.tensor(d.a_log),
                .dt_bias = try copy.tensor(d.dt_bias),
            } },
            .attention => |a| .{ .attention = .{
                .q = try profileLinear(&copy, a.q, !mlp_only, clip, &clip_index, group_size, group_clip_ratio, mse_quant, group_clip_profile), .k = try profileLinear(&copy, a.k, !mlp_only, clip, &clip_index, group_size, group_clip_ratio, mse_quant, group_clip_profile),
                .v = try profileLinear(&copy, a.v, !mlp_only, clip, &clip_index, group_size, group_clip_ratio, mse_quant, group_clip_profile), .out = try profileLinear(&copy, a.out, !mlp_only, clip, &clip_index, group_size, group_clip_ratio, mse_quant, group_clip_profile),
                .q_norm = try copy.tensor(a.q_norm), .k_norm = try copy.tensor(a.k_norm),
            } },
        };
    }
    if (packed_reader) |reader| try reader.finish();
    // The learned core must survive byte for byte, including after old storage is released.
    if (!std.mem.eql(u8, original.blocks[c.layers - 1].down.weight.host(u8),
        next.blocks[c.layers - 1].down.weight.host(u8))) return error.MixedCoreCopyMismatch;
    if (clip.len != 0 and clip_index != clip.len) return error.InvalidMixedClipProfile;
    return .{ .weights = next, .quantized = copy.quantized };
}
