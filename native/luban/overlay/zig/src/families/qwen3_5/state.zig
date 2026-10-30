//! Per-stream committed recurrence and KV state, with a shared pool of window snapshots for partial acceptance.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const execution = @import("execution.zig");
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
pub const delta_bytes = c.linear_heads * c.linear_dim * c.linear_dim * 4;
pub const conv_bytes = (c.conv_taps - 1) * c.conv_dim * 2;
pub const window_rows = 16;
pub const batch_rows = 32;
// Per layer: post-norm h, mixer output, MLP input, gate, up, act, down.
pub const trace_layer_values = 4 * c.hidden + 3 * c.intermediate;
pub const trace_bytes = c.layers * trace_layer_values * 2;
pub const capture_inputs = @import("capture_inputs.zig");

pub const Storage = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    buffers: std.ArrayList(mtl.Buffer) = .empty,

    pub fn alloc(self: *Storage, bytes: usize) !mtl.Buffer {
        const b = try self.device.buffer(@max(bytes, 16), opts);
        errdefer b.deinit();
        try self.buffers.append(self.gpa, b);
        return b;
    }

    pub fn deinit(self: *Storage) void {
        for (self.buffers.items) |b| b.deinit();
        self.buffers.deinit(self.gpa);
    }
};

pub const Delta = struct { recurrence: mtl.Buffer, conv: mtl.Buffer };
pub const Attention = struct { keys: mtl.Buffer, values: mtl.Buffer };
pub const Mixer = union(enum) { delta: Delta, attention: Attention };

pub const Cache = struct {
    memory: Storage,
    capacity: usize,
    len: usize = 0,
    execution_mask: u32 = execution.all_layers,
    last: ?struct { start: usize, base: usize, rows: usize } = null,
    blocks: [c.layers]Mixer,
    logits: mtl.Buffer,

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, capacity: usize) !Cache {
        var out: Cache = .{ .memory = .{ .gpa = gpa, .device = device }, .capacity = capacity, .blocks = undefined, .logits = undefined };
        errdefer out.memory.deinit();
        out.logits = try out.memory.alloc(c.vocab * 2);
        for (&out.blocks, 0..) |*b, i| {
            if (c.linear(i)) {
                const r = try out.memory.alloc(delta_bytes);
                const conv = try out.memory.alloc(conv_bytes);
                @memset(r.contents()[0..delta_bytes], 0);
                @memset(conv.contents()[0..conv_bytes], 0);
                b.* = .{ .delta = .{ .recurrence = r, .conv = conv } };
            } else {
                const bytes = capacity * c.kv_heads * c.head_dim * 2;
                b.* = .{ .attention = .{ .keys = try out.memory.alloc(bytes), .values = try out.memory.alloc(bytes) } };
            }
        }
        return out;
    }

    pub fn deinit(self: *Cache) void {
        self.memory.deinit();
    }

    /// Restart a stream without reallocating its buffers. Old attention rows
    /// are unreachable after len is reset; recurrence and convolution are not.
    pub fn reset(self: *Cache) void {
        self.len = 0;
        self.execution_mask = execution.all_layers;
        self.last = null;
        for (self.blocks) |block| if (block == .delta) {
            @memset(block.delta.recurrence.contents()[0..delta_bytes], 0);
            @memset(block.delta.conv.contents()[0..conv_bytes], 0);
        };
    }

    pub fn commit(self: *Cache, scratch: *const Scratch, base: usize, rows: usize, record: bool, mask: u32) void {
        // Unexecuted GDN snapshots are stale/uninitialized, not cache state.
        for (0..c.layers) |layer| if (execution.executes(mask, layer)) self.commitMixer(scratch, layer, base, rows, record);
        self.execution_mask = mask;
        if (record) self.last = .{ .start = self.len, .base = base, .rows = rows } else self.last = null;
        self.len += rows;
    }

    fn commitMixer(self: *Cache, scratch: *const Scratch, layer: usize, base: usize, rows: usize, record: bool) void {
        const block = self.blocks[layer];
        if (block == .delta) {
            const snapshot = scratch.snapshots[layer].?;
            const saved = if (record) base + rows - 1 else 0;
            if (record) @memcpy(block.delta.recurrence.contents()[0..delta_bytes], (snapshot.recurrence.contents() + saved * delta_bytes)[0..delta_bytes]);
            @memcpy(block.delta.conv.contents()[0..conv_bytes], (snapshot.conv.contents() + (base + rows - 1) * conv_bytes)[0..conv_bytes]);
        }
    }

    /// A research layer pass must never commit unexecuted layer snapshots.
    pub fn commitLayer(self: *Cache, scratch: *const Scratch, layer: usize, base: usize, rows: usize) !void {
        if (layer >= c.layers or rows == 0 or base + rows > scratch.rows or self.len + rows > self.capacity) return error.InvalidLayerCommit;
        self.commitMixer(scratch, layer, base, rows, false);
        self.last = null;
        self.len += rows;
    }

    pub fn keep(self: *Cache, scratch: *const Scratch, path: []const u32) !void {
        const last = self.last orelse return error.NothingToKeep;
        if (path.len == 0 or path.len > last.rows) return error.NothingToKeep;
        for (path, 0..) |row, i| if (row != i) return error.UnsupportedQwenTree;
        for (&self.blocks, scratch.snapshots, 0..) |*b, snapshot, layer| {
            if (!execution.executes(self.execution_mask, layer)) continue;
            if (b.* == .delta) {
                const src = last.base + path.len - 1;
                @memcpy(b.delta.recurrence.contents()[0..delta_bytes], (snapshot.?.recurrence.contents() + src * delta_bytes)[0..delta_bytes]);
                @memcpy(b.delta.conv.contents()[0..conv_bytes], (snapshot.?.conv.contents() + src * conv_bytes)[0..conv_bytes]);
            }
        }
        self.len = last.start + path.len;
        const row = last.base + path.len - 1;
        @memcpy(self.logits.contents()[0 .. c.vocab * 2], (scratch.logits.contents() + row * c.vocab * 2)[0 .. c.vocab * 2]);
        self.last = null;
    }
};

pub const Scratch = struct {
    memory: Storage,
    rows: usize,
    snapshot_rows: usize,
    residual: ?mtl.Buffer,
    trace: ?mtl.Buffer,
    trace_enabled: bool,
    capture: ?mtl.Buffer,
    capture_enabled: bool,
    ids: mtl.Buffer,
    windows: mtl.Buffer,
    dims: mtl.Buffer,
    h: mtl.Buffer,
    x: mtl.Buffer,
    r: mtl.Buffer,
    qkv: mtl.Buffer,
    z: mtl.Buffer,
    a: mtl.Buffer,
    b: mtl.Buffer,
    q: mtl.Buffer,
    k: mtl.Buffer,
    v: mtl.Buffer,
    g: mtl.Buffer,
    beta: mtl.Buffer,
    y: mtl.Buffer,
    mix: mtl.Buffer,
    qraw: mtl.Buffer,
    knorm: mtl.Buffer,
    queries: mtl.Buffer,
    gate: mtl.Buffer,
    up: mtl.Buffer,
    act: mtl.Buffer,
    pm: mtl.Buffer,
    pl: mtl.Buffer,
    po: mtl.Buffer,
    logits: mtl.Buffer,
    snapshots: [c.layers]?Delta = @splat(null),

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, rows: usize, capacity: usize) !Scratch {
        return initMode(gpa, device, rows, capacity, batch_rows);
    }

    pub fn initInference(gpa: std.mem.Allocator, device: mtl.Device, rows: usize, capacity: usize) !Scratch {
        return initMode(gpa, device, rows, capacity, 1);
    }

    fn initMode(gpa: std.mem.Allocator, device: mtl.Device, rows: usize, capacity: usize, snapshot_rows: usize) !Scratch {
        var out: Scratch = undefined;
        out.memory = .{ .gpa = gpa, .device = device };
        errdefer out.memory.deinit();
        out.rows = rows;
        out.snapshot_rows = snapshot_rows;
        out.residual = null;
        out.trace = null;
        out.trace_enabled = false;
        out.capture = null;
        out.capture_enabled = false;
        out.snapshots = @splat(null);
        const mem = &out.memory;
        out.ids = try mem.alloc(rows * 4);
        out.windows = try mem.alloc(rows * c.conv_taps * 4);
        out.dims = try mem.alloc(8 * 4);
        inline for (.{ "h", "x", "r", "z", "q", "k", "v", "y", "mix", "queries" }) |name| @field(out, name) = try mem.alloc(rows * c.hidden * 2);
        inline for (.{ "qkv", "gate", "up", "act" }) |name| @field(out, name) = try mem.alloc(rows * c.intermediate * 2);
        out.qraw = try mem.alloc(rows * 2 * c.hidden * 2);
        out.knorm = try mem.alloc(rows * c.kv_heads * c.head_dim * 2);
        inline for (.{ "a", "b", "beta" }) |name| @field(out, name) = try mem.alloc(rows * c.linear_heads * 2);
        out.g = try mem.alloc(rows * c.linear_heads * 4);
        const chunks = (capacity + 127) / 128;
        out.pm = try mem.alloc(rows * c.query_heads * chunks * 4);
        out.pl = try mem.alloc(rows * c.query_heads * chunks * 4);
        out.po = try mem.alloc(rows * c.query_heads * chunks * c.head_dim * 4);
        out.logits = try mem.alloc(batch_rows * c.vocab * 2);
        for (&out.snapshots, 0..) |*s, i| if (c.linear(i)) {
            s.* = .{ .recurrence = try mem.alloc(snapshot_rows * delta_bytes), .conv = try mem.alloc(rows * conv_bytes) };
        };
        return out;
    }

    pub fn deinit(self: *Scratch) void {
        self.memory.deinit();
    }
};
