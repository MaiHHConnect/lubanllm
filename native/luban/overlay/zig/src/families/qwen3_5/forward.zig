//! BF16/affine row projections, segment-local recurrence and causal attention on the native Metal queue.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const st = @import("state.zig");
const wts = @import("weights.zig");
const c = @import("config.zig");
const execution = @import("execution.zig");
const Buffer = mtl.Buffer;
pub const Segment = struct { cache: *st.Cache, rows: usize };
pub const Head = enum { none, last, all, greedy };

pub fn greedyToken(s: *const st.Scratch) !u32 {
    const token = s.ids.slice(u32, 1)[0];
    if (token >= c.vocab) return error.NonfiniteQwenLogits;
    return token;
}

const Encoder = struct {
    e: mtl.ComputeEncoder,
    m: *Model,
    residual: ?Buffer = null,

    fn pipe(self: Encoder, comptime key: []const u8) void {
        self.e.setPipeline(self.m.kernels.get(key));
    }
    fn tensor(self: Encoder, t: wts.Tensor, slot: usize) void {
        self.e.setBuffer(t.buffer, t.offset, slot);
    }
    fn run(self: Encoder, grid: [3]usize, group: [3]usize) void {
        self.e.dispatchThreads(mtl.Size.of(grid[0], grid[1], grid[2]), mtl.Size.of(group[0], group[1], group[2]));
    }
    fn traceCopy(self: Encoder, destination: Buffer, source: Buffer, row: usize, width: usize, offset: usize) void {
        self.pipe("qwen35_research_trace_copy");
        self.e.setBuffer(source, row * width * 2, 0);
        self.e.setBuffer(destination, offset * 2, 1);
        self.e.setValue(@as(u32, @intCast(width)), 2);
        self.run(.{ width, 1, 1 }, .{ 256, 1, 1 });
    }
    pub fn projection(self: Encoder, linear: wts.Linear, x: Buffer, offset: usize, y: Buffer, y_offset: usize, rows: usize) void {
        if (linear.dense) {
            const matrix = rows >= 8 and linear.outputs >= 512;
            if (matrix) self.pipe("qwen35_bf16_prefill_matrix") else self.pipe("qwen35_bf16_projection");
            self.e.setBuffer(x, offset, 0);
            self.e.setValue([3]u32{ @intCast(rows), @intCast(linear.inputs), @intCast(linear.outputs) }, 1);
            self.tensor(linear.weight, 2);
            self.e.setBuffer(y, y_offset, 3);
            const columns: usize = if (matrix) 64 else 8;
            const tile_rows: usize = if (matrix) 16 else 8;
            self.run(.{ 256 * ((linear.outputs + columns - 1) / columns), (rows + tile_rows - 1) / tile_rows, 1 }, .{ 256, 1, 1 });
            return;
        }
        if (linear.bits == 8) {
            const matrix = rows >= 8 and linear.outputs >= 512;
            if (matrix) self.pipe("qwen35_q8_prefill_matrix") else self.pipe("qwen35_q8_projection");
            self.e.setBuffer(x, offset, 0);
            self.e.setValue([3]u32{ @intCast(rows), @intCast(linear.inputs), @intCast(linear.outputs) }, 1);
            self.tensor(linear.weight, 2);
            self.tensor(linear.scales, 3);
            self.e.setBuffer(y, y_offset, 4);
            const columns: usize = if (matrix) 64 else 8;
            const tile_rows: usize = if (matrix) 16 else 8;
            self.run(.{ 256 * ((linear.outputs + columns - 1) / columns), (rows + tile_rows - 1) / tile_rows, 1 }, .{ 256, 1, 1 });
            return;
        }
        const p = self.m.kernels.projection(linear.outputs, linear.inputs, if (linear.bits == 16) 16 else if (linear.bits == 32) 32 else 64).?;
        self.e.setPipeline(p.pipeline);
        self.e.setBuffer(x, offset, 0);
        self.e.setValue([2]i32{ @intCast(rows), @intCast(linear.inputs) }, 1);
        self.tensor(linear.weight, 2);
        self.tensor(linear.scales, 3);
        self.tensor(linear.biases, 4);
        self.e.setValue([1]f32{1}, 5);
        self.e.setBuffer(y, y_offset, 6);
        self.run(.{ p.threads * ((linear.outputs + p.columns - 1) / p.columns), (rows + 7) / 8, 1 }, .{ p.threads, 1, 1 });
        if (linear.residual) |r| {
            const dims = [4]u32{ @intCast(rows), @intCast(linear.inputs), @intCast(linear.outputs), @intCast(r.rank) };
            self.pipe("qwen35_residual_project");
            self.e.setBuffer(x, offset, 0);
            self.tensor(r.basis, 1);
            self.e.setValue(dims, 2);
            self.e.setBuffer(self.residual.?, 0, 3);
            self.run(.{ r.rank * 32, rows, 1 }, .{ 32, 1, 1 });
            self.pipe("qwen35_residual_add");
            self.e.setBuffer(self.residual.?, 0, 0);
            self.tensor(r.update, 1);
            self.e.setValue(dims, 2);
            self.e.setBuffer(y, y_offset, 3);
            self.run(.{ linear.outputs, rows, 1 }, .{ 256, 1, 1 });
        }
    }
    fn norm(self: Encoder, h: Buffer, residual: ?Buffer, weight: wts.Tensor, x: Buffer, rows: usize) void {
        if (self.m.config.format == .bf16) {
            self.pipe("qwen35_bf16_norm");
            self.e.setBuffer(h, 0, 0);
            self.e.setBuffer(residual orelse h, 0, 1);
            self.tensor(weight, 2);
            self.e.setValue(c.eps, 3);
            self.e.setBuffer(h, 0, 4);
            self.e.setBuffer(x, 0, 5);
            self.e.setValue(@as(u32, @intFromBool(residual != null)), 6);
            self.run(.{ 128, rows, 1 }, .{ 128, 1, 1 });
            return;
        }
        if (residual) |r| {
            self.pipe("norm");
            self.e.setBuffer(h, 0, 0);
            self.e.setBuffer(r, 0, 1);
            self.tensor(weight, 2);
            self.e.setValue(c.eps, 3);
            self.e.setBuffer(h, 0, 4);
            self.e.setBuffer(x, 0, 5);
        } else {
            self.pipe("norm_nores");
            self.e.setBuffer(h, 0, 0);
            self.tensor(weight, 1);
            self.e.setValue(c.eps, 2);
            self.e.setBuffer(x, 0, 3);
        }
        self.run(.{ 128, rows, 1 }, .{ 128, 1, 1 });
    }
};

/// Same live projection launch, isolated from cache, layer state and sampling.
pub fn auditProjection(m: *Model, linear: wts.Linear, x: Buffer, y: Buffer, rows: usize) !void {
    if (rows != 1 and rows != 9) return error.InvalidAuditRows;
    if (linear.dense or (linear.bits != 32 and linear.bits != 16) or linear.residual != null) return error.AuditRequiresPlainQ4Group16Or32;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const enc = Encoder{ .e = cb.compute(.serial), .m = m };
    enc.projection(linear, x, 0, y, 0, rows);
    enc.e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
}

pub fn run(m: *Model, s: *st.Scratch, segments: []const Segment, tokens: []const u32, record: bool, head: Head) !void {
    const execution_mask = try m.execution_policy.mask();
    if (execution_mask != execution.all_layers and (s.capture_enabled or s.trace_enabled)) return error.ReducedGraphCannotCaptureFullTrace;
    if (tokens.len == 0 or tokens.len > s.rows or (record and tokens.len > s.snapshot_rows)) return error.QwenRowsExceeded;
    if (head == .greedy and s.rows * c.hidden * 2 < ((c.vocab + 255) / 256) * 8) return error.QwenRowsExceeded;
    var total: usize = 0;
    for (segments) |seg| {
        try execution.validateCache(execution_mask, seg.cache.execution_mask, seg.cache.len);
        if (seg.rows == 0 or seg.cache.len + seg.rows > seg.cache.capacity) return error.QwenContextFull;
        total += seg.rows;
    }
    if (total != tokens.len or (head == .all and total > st.batch_rows) or (!record and segments.len != 1)) return error.InvalidQwenSegments;
    for (tokens) |id| if (id >= c.vocab) return error.InvalidQwenToken;
    @memcpy(s.ids.slice(u32, total), tokens);
    var base: usize = 0;
    const windows = s.windows.slice(i32, total * c.conv_taps);
    for (segments) |seg| {
        for (0..seg.rows) |r| for (0..c.conv_taps) |tap| {
            windows[(base + r) * c.conv_taps + tap] = @intCast(r + tap);
        };
        base += seg.rows;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var cb = m.queue.commandBuffer();
    if (m.residual_rank != 0 and s.residual == null)
        s.residual = try s.memory.alloc(s.rows * 32 * 4);
    var enc = Encoder{ .e = cb.compute(.serial), .m = m, .residual = s.residual };
    embed(enc, s, total);
    for (0..c.layers) |i| {
        if (i >= m.execution_policy.exit_after) break;
        // Leave h and the pending MLP residual untouched. The next executed
        // layer (or final norm) consumes that residual exactly once.
        if (!execution.executes(execution_mask, i)) continue;
        if (m.layer_pager != null) {
            // The previous command buffer must finish before a one-layer
            // paging slot can be overwritten. Also covers the embedding pass.
            enc.e.end();
            cb.commit();
            cb.wait();
            if (cb.failure() != null) return error.GpuFailed;
        }
        const block = try m.acquireLayer(i);
        if (m.layer_pager != null) {
            cb = m.queue.commandBuffer();
            enc = .{ .e = cb.compute(.serial), .m = m, .residual = s.residual };
        }
        layerInputs(enc, s, block, i, segments, record, total);
        if (s.capture_enabled)
            enc.traceCopy(s.capture.?, s.x, 0, total * c.hidden, s.rows * st.capture_inputs.offset(i, false));
        if (s.trace_enabled) {
            const offset = i * st.trace_layer_values;
            enc.traceCopy(s.trace.?, s.h, total - 1, c.hidden, offset);
            enc.traceCopy(s.trace.?, s.r, total - 1, c.hidden, offset + c.hidden);
            enc.traceCopy(s.trace.?, s.x, total - 1, c.hidden, offset + 2 * c.hidden);
        }
        gateUp(enc, s, block, total);
        if (s.capture_enabled and i != c.layers - 1)
            enc.traceCopy(s.capture.?, s.act, 0, total * c.intermediate, s.rows * st.capture_inputs.offset(i, true));
        enc.projection(block.down, s.act, 0, s.r, 0, total);
        if (s.trace_enabled) {
            const offset = i * st.trace_layer_values + 3 * c.hidden;
            enc.traceCopy(s.trace.?, s.gate, total - 1, c.intermediate, offset);
            enc.traceCopy(s.trace.?, s.up, total - 1, c.intermediate, offset + c.intermediate);
            enc.traceCopy(s.trace.?, s.act, total - 1, c.intermediate, offset + 2 * c.intermediate);
            enc.traceCopy(s.trace.?, s.r, total - 1, c.hidden, offset + 3 * c.intermediate);
        }
    }
    if (m.expert) |*expert| expert.encode(enc, s, total);
    projectHead(enc, s, total, head);
    enc.e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("Qwen forward: {s}\n", .{msg});
        return error.GpuFailed;
    }
    base = 0;
    for (segments) |seg| {
        const logit_row = if (head == .all) base + seg.rows - 1 else 0;
        if (head != .none) @memcpy(seg.cache.logits.contents()[0 .. c.vocab * 2], (s.logits.contents() + logit_row * c.vocab * 2)[0 .. c.vocab * 2]);
        seg.cache.commit(s, base, seg.rows, record, execution_mask);
        base += seg.rows;
    }
}

fn delta(enc: Encoder, s: *st.Scratch, d: wts.Delta, layer: usize, segments: []const Segment, record: bool, total: usize) void {
    const e = enc.e;
    enc.projection(d.qkv, s.x, 0, s.qkv, 0, total);
    enc.projection(d.z, s.x, 0, s.z, 0, total);
    enc.projection(d.a, s.x, 0, s.a, 0, total);
    enc.projection(d.b, s.x, 0, s.b, 0, total);
    var base: usize = 0;
    const snapshot = s.snapshots[layer].?;
    for (segments) |seg| {
        const cache = seg.cache.blocks[layer].delta;
        if (enc.m.config.format == .bf16) enc.pipe("qwen35_bf16_gdn_pre") else enc.pipe("gdn_pre");
        e.setBuffer(s.qkv, base * c.conv_dim * 2, 0);
        e.setBuffer(cache.conv, 0, 1);
        enc.tensor(d.conv, 2);
        e.setBuffer(s.windows, base * c.conv_taps * 4, 3);
        e.setBuffer(s.a, base * c.linear_heads * 2, 4);
        e.setBuffer(s.b, base * c.linear_heads * 2, 5);
        enc.tensor(d.a_log, 6);
        enc.tensor(d.dt_bias, 7);
        for ([_]Buffer{ s.q, s.k, s.v }, 8..) |b, slot| e.setBuffer(b, base * c.hidden * 2, slot);
        e.setBuffer(s.g, base * c.linear_heads * 4, 11);
        e.setBuffer(s.beta, base * c.linear_heads * 2, 12);
        e.setBuffer(snapshot.conv, base * st.conv_bytes, 13);
        enc.run(.{ 32, 3 * c.linear_heads, seg.rows }, .{ 32, 1, 1 });
        if (enc.m.config.format == .bf16) enc.pipe("qwen35_bf16_gdn_chain") else enc.pipe("gdn_chain");
        for ([_]Buffer{ s.q, s.k, s.v }, 0..) |b, slot| e.setBuffer(b, base * c.hidden * 2, slot);
        e.setBuffer(s.g, base * c.linear_heads * 4, 3);
        e.setBuffer(s.beta, base * c.linear_heads * 2, 4);
        e.setBuffer(cache.recurrence, 0, 5);
        e.setValue([2]i32{ @intCast(seg.rows), @intFromBool(record) }, 6);
        e.setBuffer(s.y, base * c.hidden * 2, 7);
        e.setBuffer(if (record) snapshot.recurrence else cache.recurrence, if (record) base * st.delta_bytes else 0, 8);
        enc.run(.{ 32, c.linear_dim, c.linear_heads }, .{ 32, 4, 1 });
        base += seg.rows;
    }
    if (enc.m.config.format == .bf16) enc.pipe("qwen35_bf16_gdn_post") else enc.pipe("gdn_post");
    e.setBuffer(s.y, 0, 0);
    e.setBuffer(s.z, 0, 1);
    enc.tensor(d.norm, 2);
    e.setValue(c.eps, 3);
    e.setBuffer(s.mix, 0, 4);
    enc.run(.{ 32, c.linear_heads, total }, .{ 32, 1, 1 });
    enc.projection(d.out, s.mix, 0, s.r, 0, total);
}

fn attention(enc: Encoder, s: *st.Scratch, a: wts.Attention, layer: usize, segments: []const Segment, total: usize) void {
    const e = enc.e;
    enc.projection(a.q, s.x, 0, s.qraw, 0, total);
    enc.projection(a.k, s.x, 0, s.k, 0, total);
    enc.projection(a.v, s.x, 0, s.v, 0, total);
    if (enc.m.config.format == .bf16) enc.pipe("qwen35_bf16_head_norm") else enc.pipe("qwen35_head_norm");
    e.setBuffer(s.qraw, 0, 0);
    enc.tensor(a.q_norm, 1);
    e.setBuffer(s.q, 0, 2);
    e.setValue([4]u32{ @intCast(total), c.query_heads, 2 * c.hidden, 2 * c.head_dim }, 3);
    enc.run(.{ 64 * total * c.query_heads, 1, 1 }, .{ 64, 1, 1 });
    if (enc.m.config.format == .bf16) enc.pipe("qwen35_bf16_head_norm") else enc.pipe("qwen35_head_norm");
    e.setBuffer(s.k, 0, 0);
    enc.tensor(a.k_norm, 1);
    e.setBuffer(s.knorm, 0, 2);
    e.setValue([4]u32{ @intCast(total), c.kv_heads, c.kv_heads * c.head_dim, c.head_dim }, 3);
    enc.run(.{ 64 * total * c.kv_heads, 1, 1 }, .{ 64, 1, 1 });
    var base: usize = 0;
    for (segments) |seg| {
        const cache = seg.cache.blocks[layer].attention;
        if (enc.m.config.format == .bf16) enc.pipe("qwen35_bf16_queries") else enc.pipe("qwen35_queries");
        e.setBuffer(s.q, 0, 0);
        e.setBuffer(s.queries, 0, 1);
        e.setValue([4]u32{ @intCast(seg.cache.len), @intCast(total), @intCast(base), 0 }, 2);
        enc.run(.{ c.head_dim, c.query_heads, seg.rows }, .{ 256, 1, 1 });
        if (enc.m.config.format == .bf16) enc.pipe("qwen35_bf16_keys") else enc.pipe("qwen35_keys");
        e.setBuffer(s.knorm, 0, 0);
        e.setBuffer(s.v, 0, 1);
        e.setBuffer(cache.keys, 0, 2);
        e.setBuffer(cache.values, 0, 3);
        e.setValue([4]u32{ @intCast(seg.cache.len), @intCast(seg.cache.capacity), @intCast(base), 0 }, 4);
        enc.run(.{ c.head_dim, c.kv_heads, seg.rows }, .{ 256, 1, 1 });
        const nch = (seg.cache.len + seg.rows + 127) / 128;
        const dims = [8]i32{ @intCast(seg.cache.len), @intCast(seg.rows), @intCast(seg.cache.capacity), @intCast(nch), 0, @intCast(total), @intCast(base), 0 };
        enc.pipe("attn_partial");
        e.setBuffer(s.queries, 0, 0);
        e.setBuffer(cache.keys, 0, 1);
        e.setBuffer(cache.values, 0, 2);
        e.setValue(@as(f32, 1.0 / 16.0), 3);
        e.setBytes(std.mem.asBytes(&dims), 4);
        e.setBuffer(s.pm, 0, 5);
        e.setBuffer(s.pl, 0, 6);
        e.setBuffer(s.po, 0, 7);
        enc.run(.{ 512, nch, c.kv_heads * seg.rows }, .{ 512, 1, 1 });
        enc.pipe("attn_merge");
        e.setBuffer(s.pm, 0, 0);
        e.setBuffer(s.pl, 0, 1);
        e.setBuffer(s.po, 0, 2);
        e.setBytes(std.mem.asBytes(&dims), 3);
        e.setBuffer(s.y, base * c.hidden * 2, 4);
        enc.run(.{ 32, c.query_heads, seg.rows }, .{ 32, 1, 1 });
        base += seg.rows;
    }
    enc.pipe("qwen35_attention_gate");
    e.setBuffer(s.y, 0, 0);
    e.setBuffer(s.qraw, 0, 1);
    e.setBuffer(s.mix, 0, 2);
    enc.run(.{ c.hidden, total, 1 }, .{ 256, 1, 1 });
    enc.projection(a.out, s.mix, 0, s.r, 0, total);
}

pub const sequential = @import("sequential.zig");
fn embed(enc: Encoder, s: *st.Scratch, total: usize) void {
    const e = enc.e;
    if (enc.m.weights.embedding.dense) {
        enc.pipe("qwen35_bf16_embed");
        e.setBuffer(s.ids, 0, 0);
        enc.tensor(enc.m.weights.embedding.weight, 1);
        e.setBuffer(s.h, 0, 2);
        e.setValue([2]u32{ @intCast(total), c.hidden }, 3);
        enc.run(.{ c.hidden, total, 1 }, .{ 256, 1, 1 });
    } else if (enc.m.weights.embedding.bits == 8) {
        enc.pipe("qwen35_q8_embed");
        e.setBuffer(s.ids, 0, 0);
        enc.tensor(enc.m.weights.embedding.weight, 1);
        enc.tensor(enc.m.weights.embedding.scales, 2);
        e.setBuffer(s.h, 0, 3);
        e.setValue([2]u32{ @intCast(total), c.hidden }, 4);
        enc.run(.{ c.hidden, total, 1 }, .{ 256, 1, 1 });
    } else {
        enc.pipe("qwen35_embed");
        e.setBuffer(s.ids, 0, 0);
        enc.tensor(enc.m.weights.embedding.weight, 1);
        enc.tensor(enc.m.weights.embedding.scales, 2);
        enc.tensor(enc.m.weights.embedding.biases, 3);
        e.setBuffer(s.h, 0, 4);
        enc.run(.{ 1024, total, 1 }, .{ 256, 1, 1 });
    }
}
fn layerInputs(enc: Encoder, s: *st.Scratch, block: wts.Layer, i: usize, segments: []const Segment, record: bool, total: usize) void {
    enc.norm(s.h, if (i == 0) null else s.r, block.input_norm, s.x, total);
    switch (block.mixer) {
        .delta => |d| delta(enc, s, d, i, segments, record, total),
        .attention => |a| attention(enc, s, a, i, segments, total),
    }
    enc.norm(s.h, s.r, block.post_norm, s.x, total);
}
fn gateUp(enc: Encoder, s: *st.Scratch, block: wts.Layer, total: usize) void {
    const e = enc.e;
    enc.projection(block.gate, s.x, 0, s.gate, 0, total);
    enc.projection(block.up, s.x, 0, s.up, 0, total);
    if (enc.m.config.format == .bf16) enc.pipe("qwen35_bf16_mlp_act") else enc.pipe("mlp_act");
    e.setBuffer(s.gate, 0, 0);
    e.setBuffer(s.up, 0, 1);
    e.setBuffer(s.act, 0, 2);
    enc.run(.{ c.intermediate, total, 1 }, .{ 256, 1, 1 });
}
fn projectHead(enc: Encoder, s: *st.Scratch, total: usize, head: Head) void {
    const e = enc.e;
    if (head != .none) {
        enc.norm(s.h, s.r, enc.m.weights.norm, s.x, total);
        if (head == .all) enc.projection(enc.m.weights.head(), s.x, 0, s.logits, 0, total) else {
            enc.projection(enc.m.weights.head(), s.x, (total - 1) * c.hidden * 2, s.logits, 0, 1);
        }
        if (head == .greedy) {
            const groups = (c.vocab + 255) / 256;
            enc.pipe("qwen35_greedy_partial");
            e.setBuffer(s.logits, 0, 0);
            e.setValue(@as(u32, c.vocab), 1);
            e.setBuffer(s.r, 0, 2);
            enc.run(.{ groups * 256, 1, 1 }, .{ 256, 1, 1 });
            enc.pipe("qwen35_greedy_finish");
            e.setBuffer(s.r, 0, 0);
            e.setValue(@as(u32, groups), 1);
            e.setBuffer(s.ids, 0, 2);
            enc.run(.{ 256, 1, 1 }, .{ 256, 1, 1 });
        }
    }
}
/// Research layer pass. Input buffers come from the preceding exact BF16 boundary.
pub const LayerPhase = enum { inputs, act, output, head };
pub fn layerPass(m: *Model, s: *st.Scratch, cache: *st.Cache, layer: usize, tokens: []const u32, phase: LayerPhase, recompute_act: bool) !void {
    if (try m.execution_policy.mask() != execution.all_layers or m.layer_pager != null) return error.LayerPassRequiresResidentFullGraph;
    if (layer >= c.layers or tokens.len == 0 or tokens.len > s.rows or s.snapshot_rows != 1) return error.InvalidLayerPass;
    const total = tokens.len;
    if (phase == .inputs and cache.len + total > cache.capacity) return error.QwenContextFull;
    for (tokens) |id| if (id >= c.vocab) return error.InvalidQwenToken;
    @memcpy(s.ids.slice(u32, total), tokens);
    if (phase == .inputs) {
        const windows = s.windows.slice(i32, total * c.conv_taps);
        for (0..total) |r| for (0..c.conv_taps) |tap| {
            windows[r * c.conv_taps + tap] = @intCast(r + tap);
        };
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const enc = Encoder{ .e = cb.compute(.serial), .m = m, .residual = s.residual };
    const block = m.weights.blocks[layer];
    switch (phase) {
        .inputs => {
            if (layer == 0) embed(enc, s, total);
            layerInputs(enc, s, block, layer, &.{.{ .cache = cache, .rows = total }}, false, total);
        },
        .act => gateUp(enc, s, block, total),
        .output => {
            if (recompute_act) gateUp(enc, s, block, total);
            enc.projection(block.down, s.act, 0, s.r, 0, total);
        },
        .head => projectHead(enc, s, total, .greedy),
    }
    enc.e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
    if (phase == .inputs) try cache.commitLayer(s, layer, 0, total);
}
