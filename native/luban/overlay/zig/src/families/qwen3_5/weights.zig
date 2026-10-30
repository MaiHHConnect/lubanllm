// SPDX-License-Identifier: Apache-2.0
//! TensorFold-derived views: tied original HF BF16 or packed MLX affine4 weights.
const std = @import("std");
const ck = @import("../../core/checkpoint_metal.zig");
const c = @import("config.zig");

pub const Tensor = ck.Tensor;
// Dense placeholders alias weight and must never be read by the quantized ABI.
pub const Residual = struct { basis: Tensor, update: Tensor, rank: usize };
pub const Linear = struct { residual: ?Residual = null, weight: Tensor, scales: Tensor, biases: Tensor, inputs: usize, outputs: usize, dense: bool = false, bits: u8 = 4 };
pub const Delta = struct { qkv: Linear, z: Linear, a: Linear, b: Linear, out: Linear, conv: Tensor, norm: Tensor, a_log: Tensor, dt_bias: Tensor };
pub const Attention = struct { q: Linear, k: Linear, v: Linear, out: Linear, q_norm: Tensor, k_norm: Tensor };
pub const Layer = struct {
    input_norm: Tensor,
    post_norm: Tensor,
    gate: Linear,
    up: Linear,
    down: Linear,
    mixer: union(enum) { delta: Delta, attention: Attention },
};

pub const Weights = struct {
    embedding: Linear,
    norm: Tensor,
    blocks: [c.layers]Layer,

    pub fn head(self: *const Weights) Linear {
        return self.embedding;
    }
};

/// Visit each stored tensor once; dense/Q8 aliases are restored after rebinding.
/// The pager uses the same order to size, export and bind a layer's byte range.
pub fn visitLayer(layer: *Layer, visitor: anytype) !void {
    try visitor.tensor(&layer.input_norm);
    try visitor.tensor(&layer.post_norm);
    try visitLinear(&layer.gate, visitor);
    try visitLinear(&layer.up, visitor);
    try visitLinear(&layer.down, visitor);
    switch (layer.mixer) {
        .delta => |*d| {
            try visitLinear(&d.qkv, visitor);
            try visitLinear(&d.z, visitor);
            try visitLinear(&d.a, visitor);
            try visitLinear(&d.b, visitor);
            try visitLinear(&d.out, visitor);
            try visitor.tensor(&d.conv);
            try visitor.tensor(&d.norm);
            try visitor.tensor(&d.a_log);
            try visitor.tensor(&d.dt_bias);
        },
        .attention => |*a| {
            try visitLinear(&a.q, visitor);
            try visitLinear(&a.k, visitor);
            try visitLinear(&a.v, visitor);
            try visitLinear(&a.out, visitor);
            try visitor.tensor(&a.q_norm);
            try visitor.tensor(&a.k_norm);
        },
    }
}

pub fn visitLinear(linear_value: *Linear, visitor: anytype) !void {
    try visitor.tensor(&linear_value.weight);
    if (linear_value.dense) {
        linear_value.scales = linear_value.weight;
        linear_value.biases = linear_value.weight;
    } else {
        try visitor.tensor(&linear_value.scales);
        if (linear_value.bits == 8) linear_value.biases = linear_value.scales else try visitor.tensor(&linear_value.biases);
    }
    if (linear_value.residual) |*r| {
        try visitor.tensor(&r.basis);
        try visitor.tensor(&r.update);
    }
}

test "paging visits stored projection payloads once and restores dense/Q8 aliases" {
    const Counter = struct {
        count: usize = 0,
        pub fn tensor(self: *@This(), t: *Tensor) !void {
            self.count += 1;
            t.offset = self.count * 16;
        }
    };
    const t = Tensor{ .buffer = undefined, .offset = 0, .bytes = 16, .dtype = .bf16 };
    var dense = Linear{ .weight = t, .scales = t, .biases = t, .inputs = 4, .outputs = 2, .dense = true };
    var count = Counter{};
    try visitLinear(&dense, &count);
    try std.testing.expectEqual(@as(usize, 1), count.count);
    try std.testing.expectEqual(dense.weight.offset, dense.scales.offset);
    try std.testing.expectEqual(dense.weight.offset, dense.biases.offset);
    var q8 = dense;
    q8.dense = false;
    q8.bits = 8;
    try visitLinear(&q8, &count);
    try std.testing.expectEqual(@as(usize, 3), count.count);
    try std.testing.expectEqual(q8.scales.offset, q8.biases.offset);
    var q4 = q8;
    q4.bits = 16;
    q4.residual = .{ .basis = t, .update = t, .rank = 1 };
    try visitLinear(&q4, &count);
    try std.testing.expectEqual(@as(usize, 8), count.count);
    try std.testing.expect(q4.scales.offset != q4.biases.offset);
    var layer = Layer{ .input_norm = t, .post_norm = t, .gate = dense, .up = dense, .down = dense,
        .mixer = .{ .attention = .{ .q = dense, .k = dense, .v = dense, .out = dense, .q_norm = t, .k_norm = t } } };
    count.count = 0;
    try visitLayer(&layer, &count);
    try std.testing.expectEqual(@as(usize, 11), count.count);
    layer.mixer = .{ .delta = .{ .qkv = dense, .z = dense, .a = dense, .b = dense, .out = dense,
        .conv = t, .norm = t, .a_log = t, .dt_bias = t } };
    count.count = 0;
    try visitLayer(&layer, &count);
    try std.testing.expectEqual(@as(usize, 14), count.count);
}

fn expect(ckpt: *const ck.Checkpoint, name: []const u8, dtype: ck.DType, shape: []const usize) !Tensor {
    const t = try ckpt.get(name);
    if (t.dtype != dtype or t.rank != shape.len or !std.mem.eql(usize, t.shape[0..t.rank], shape)) {
        std.log.err("{s}: unexpected Qwen tensor dtype or shape", .{name});
        return error.UnexpectedQwenTensor;
    }
    return t;
}

fn tensor(ckpt: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, dtype: ck.DType, shape: []const usize) !Tensor {
    var buf: [256]u8 = undefined;
    return expect(ckpt, try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix }), dtype, shape);
}

fn projection(ckpt: *const ck.Checkpoint, prefix: []const u8, suffix: []const u8, outputs: usize, inputs: usize, group_size: usize) !Linear {
    var buf: [256]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, suffix });
    return .{
        .weight = try tensor(ckpt, name, ".weight", .u32, &.{ outputs, inputs / 8 }),
        .scales = try tensor(ckpt, name, ".scales", .bf16, &.{ outputs, inputs / group_size }),
        .biases = try tensor(ckpt, name, ".biases", .bf16, &.{ outputs, inputs / group_size }),
        .inputs = inputs,
        .outputs = outputs,
        .bits = if (group_size == 16) 16 else 4,
    };
}

pub fn denseProjection(weight: Tensor, outputs: usize, inputs: usize) !Linear {
    if (weight.dtype != .bf16 or weight.rank != 2 or
        weight.shape[0] != outputs or weight.shape[1] != inputs or weight.bytes != outputs * inputs * 2)
        return error.UnexpectedDenseQwenTensor;
    return .{ .weight = weight, .scales = weight, .biases = weight, .inputs = inputs, .outputs = outputs, .dense = true };
}

fn linear(ckpt: *const ck.Checkpoint, format: c.Format, prefix: []const u8, suffix: []const u8, outputs: usize, inputs: usize) !Linear {
    if (format == .affine4) return projection(ckpt, prefix, suffix, outputs, inputs, c.group);
    var buf: [256]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "{s}{s}.weight", .{ prefix, suffix });
    return denseProjection(try expect(ckpt, name, .bf16, &.{ outputs, inputs }), outputs, inputs);
}

pub fn load(ckpt: *const ck.Checkpoint) !Weights {
    return loadFormat(ckpt, .affine4);
}

pub fn loadFormat(ckpt: *const ck.Checkpoint, format: c.Format) !Weights {
    return loadLayout(ckpt, format, false);
}

pub fn loadDirectQ4(ckpt: *const ck.Checkpoint) !Weights {
    return loadLayout(ckpt, .bf16, true);
}

fn loadLayout(ckpt: *const ck.Checkpoint, format: c.Format, direct_q4: bool) !Weights {
    const prefix = if (format == .bf16) "model.language_model." else "language_model.model.";
    var w = Weights{
        .embedding = try linear(ckpt, format, prefix, "embed_tokens", c.vocab, c.hidden),
        .norm = try tensor(ckpt, prefix, "norm.weight", .bf16, &.{c.hidden}),
        .blocks = undefined,
    };
    if (ckpt.has("language_model.lm_head.weight") or ckpt.has("lm_head.weight")) return error.UnexpectedUntiedQwenHead;
    for (&w.blocks, 0..) |*block, i| {
        var buf: [128]u8 = undefined;
        const p = try std.fmt.bufPrint(&buf, "{s}layers.{d}.", .{ prefix, i });
        block.* = .{
            .input_norm = try tensor(ckpt, p, "input_layernorm.weight", .bf16, &.{c.hidden}),
            .post_norm = try tensor(ckpt, p, "post_attention_layernorm.weight", .bf16, &.{c.hidden}),
            .gate = if (direct_q4) try projection(ckpt, p, "mlp.gate_proj", c.intermediate, c.hidden, 16) else try linear(ckpt, format, p, "mlp.gate_proj", c.intermediate, c.hidden),
            .up = if (direct_q4) try projection(ckpt, p, "mlp.up_proj", c.intermediate, c.hidden, 16) else try linear(ckpt, format, p, "mlp.up_proj", c.intermediate, c.hidden),
            .down = if (direct_q4 and i != c.layers - 1) try projection(ckpt, p, "mlp.down_proj", c.hidden, c.intermediate, 16) else try linear(ckpt, format, p, "mlp.down_proj", c.hidden, c.intermediate),
            .mixer = undefined,
        };
        if (c.linear(i)) {
            block.mixer = .{ .delta = .{
                .qkv = try linear(ckpt, format, p, "linear_attn.in_proj_qkv", c.conv_dim, c.hidden),
                .z = try linear(ckpt, format, p, "linear_attn.in_proj_z", c.hidden, c.hidden),
                .a = try linear(ckpt, format, p, "linear_attn.in_proj_a", c.linear_heads, c.hidden),
                .b = try linear(ckpt, format, p, "linear_attn.in_proj_b", c.linear_heads, c.hidden),
                .out = try linear(ckpt, format, p, "linear_attn.out_proj", c.hidden, c.hidden),
                // Singleton axes differ, but the original contiguous bytes do not.
                .conv = try tensor(ckpt, p, "linear_attn.conv1d.weight", .bf16,
                    if (format == .bf16) &.{ c.conv_dim, 1, c.conv_taps } else &.{ c.conv_dim, c.conv_taps, 1 }),
                .norm = try tensor(ckpt, p, "linear_attn.norm.weight", if (format == .bf16) .f32 else .bf16, &.{c.linear_dim}),
                .a_log = try tensor(ckpt, p, "linear_attn.A_log", .f32, &.{c.linear_heads}),
                .dt_bias = try tensor(ckpt, p, "linear_attn.dt_bias", .bf16, &.{c.linear_heads}),
            } };
        } else {
            block.mixer = .{ .attention = .{
                .q = try linear(ckpt, format, p, "self_attn.q_proj", 2 * c.hidden, c.hidden),
                .k = try linear(ckpt, format, p, "self_attn.k_proj", c.kv_heads * c.head_dim, c.hidden),
                .v = try linear(ckpt, format, p, "self_attn.v_proj", c.kv_heads * c.head_dim, c.hidden),
                .out = try linear(ckpt, format, p, "self_attn.o_proj", c.hidden, c.hidden),
                .q_norm = try tensor(ckpt, p, "self_attn.q_norm.weight", .bf16, &.{c.head_dim}),
                .k_norm = try tensor(ckpt, p, "self_attn.k_norm.weight", .bf16, &.{c.head_dim}),
            } };
        }
    }
    return w;
}
