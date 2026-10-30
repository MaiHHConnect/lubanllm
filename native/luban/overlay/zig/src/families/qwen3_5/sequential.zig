//! Layerwise research execution with one resident original model and bounded activation files.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("model.zig").Model;
const w = @import("weights.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const c = @import("config.zig");
pub const stage = @import("sequential_stage.zig");
pub const Op = enum { begin, inputs, gateup, down, decide, verify };
pub const Phase = enum { ready, inputs, act, candidate, complete, failed };
pub const Command = struct {
    op: Op,
    revision: usize,
    layer: usize,
    plan: ?[]const u8 = null,
    payload: ?[]const u8 = null,
    dry_run: bool = false,
    accept: ?bool = null,
};
pub const Case = struct { id: []const u8, split: []const u8, tokens: []const u32, forced_tokens: []const u32, positions: []const u32 };
pub const Plan = struct { schema: []const u8, manifest: []const u8, core: ?[]const u8, eos_ids: []const u32, cases: []const Case };
pub const Record = struct { id: []const u8, split: []const u8, path: []const u8, rows: usize, prompt_tokens: usize, forced_tokens: usize, row_values: usize, kind: usize };
pub const Prediction = struct { id: []const u8, tokens: []const u32 };
pub const Response = struct {
    op: Op,
    revision: usize,
    layer: usize,
    phase: Phase,
    accepted_layer_mask: u32,
    records: []const Record = &.{},
    predictions: []const Prediction = &.{},
    verified_values: usize = 0,
};

pub fn validatePlan(plan: Plan, manifest: []const u8) !void {
    if (!std.mem.eql(u8, plan.schema, "q4-sequential-plan-v1") or !std.mem.eql(u8, plan.manifest, manifest) or plan.core != null or (plan.cases.len != 20 and plan.cases.len != 36) or plan.eos_ids.len == 0) return error.InvalidSequentialPlan;
    for (plan.eos_ids) |token| if (token >= c.vocab) return error.InvalidSequentialPlan;
    var cal: usize = 0;
    var obs: usize = 0;
    for (plan.cases, 0..) |case, i| {
        if (case.id.len == 0 or case.id.len > 128) return error.InvalidSequentialPlan;
        for (case.id) |ch| if (!std.ascii.isAlphanumeric(ch) and ch != '-' and ch != '_') return error.InvalidSequentialPlan;
        for (plan.cases[0..i]) |previous| if (std.mem.eql(u8, case.id, previous.id)) return error.InvalidSequentialPlan;
        if (std.mem.eql(u8, case.split, "calibration")) cal += 1 else if (std.mem.eql(u8, case.split, "observation")) obs += 1 else return error.InvalidSequentialPlan;
        try st.capture_inputs.validate(case.tokens.len, case.forced_tokens.len, case.positions);
        for (case.tokens) |token| if (token >= c.vocab) return error.InvalidSequentialPlan;
        for (case.forced_tokens) |token| {
            if (token >= c.vocab) return error.InvalidSequentialPlan;
            for (plan.eos_ids) |eos| if (token == eos) return error.InvalidSequentialPlan;
        }
    }
    if (cal != plan.cases.len - 4 or obs != 4) return error.InvalidSequentialPlan;
}

pub fn requirePhase(phase: Phase, revision: usize, command: Command) !void {
    if (command.layer >= 24 or command.layer != revision or command.revision != revision or command.op == .begin) return error.InvalidSequentialOrder;
    const expected: Phase = switch (command.op) {
        .inputs => .ready,
        .gateup => .inputs,
        .down => .act,
        .decide, .verify => .candidate,
        .begin => unreachable,
    };
    if (phase != expected or (command.op == .decide) != (command.accept != null) or command.plan != null or command.dry_run) return error.InvalidSequentialOrder;
    if (command.op != .gateup and command.op != .down and command.payload != null) return error.InvalidSequentialOrder;
}

pub const Session = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    model: *Model,
    directory: []const u8,
    parsed: std.json.Parsed(Plan),
    original: w.Weights,
    dry_run: bool,
    revision: usize = 0,
    phase: Phase = .ready,
    mask: u32 = 0,
    gateups: [24]?stage.Installed = @splat(null),
    downs: [24]?stage.Installed = @splat(null),
    fd: c_int,
    cancelled: *const fn (c_int) anyerror!bool,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, model: *Model, directory: []const u8, command: Command, fd: c_int, cancel: *const fn (c_int) anyerror!bool) !Session {
        if (command.op != .begin or command.revision != 0 or command.layer != 0 or command.payload != null or command.accept != null) return error.InvalidSequentialOrder;
        const plan_path = command.plan orelse return error.InvalidSequentialPlan;
        if (!std.fs.path.isAbsolute(plan_path) or model.mixed or model.config.format != .bf16 or model.core_binding != null or model.original != null or model.residual_rank != 0) return error.SequentialRequiresOriginal;
        const binding = model.base_binding orelse return error.SequentialRequiresOriginal;
        const file = try std.Io.Dir.cwd().openFile(io, plan_path, .{ .follow_symlinks = false });
        defer file.close(io);
        const size = (try file.stat(io)).size;
        if (size > 4 << 20) return error.InvalidSequentialPlan;
        const bytes = try gpa.alloc(u8, @intCast(size));
        defer gpa.free(bytes);
        if (try file.readPositionalAll(io, bytes, 0) != size) return error.InvalidSequentialPlan;
        const parsed = try std.json.parseFromSlice(Plan, gpa, bytes, .{ .allocate = .alloc_always });
        errdefer parsed.deinit();
        try validatePlan(parsed.value, &binding.manifest_sha256);
        return .{ .gpa = gpa, .io = io, .model = model, .directory = directory, .parsed = parsed, .original = model.weights, .dry_run = command.dry_run, .fd = fd, .cancelled = cancel };
    }
    pub fn deinit(self: *Session) void {
        self.model.weights = self.original;
        self.model.mixed = false;
        for (&self.gateups) |*owner| if (owner.*) |*value| value.deinit();
        for (&self.downs) |*owner| if (owner.*) |*value| value.deinit();
        self.parsed.deinit();
        self.* = undefined;
    }
    pub fn allowsGenerate(self: *const Session) bool {
        return self.phase == .ready or self.phase == .candidate or self.phase == .complete;
    }
    fn checkCancel(self: *Session) !void {
        if (try self.cancelled(self.fd)) return error.SequentialCancelled;
    }
    fn restoreLayer(self: *Session, layer: usize) void {
        self.model.weights.blocks[layer].gate = self.original.blocks[layer].gate;
        self.model.weights.blocks[layer].up = self.original.blocks[layer].up;
        self.model.weights.blocks[layer].down = self.original.blocks[layer].down;
    }
    fn freeLayer(self: *Session, layer: usize) void {
        if (self.gateups[layer]) |*owner| owner.deinit();
        self.gateups[layer] = null;
        if (self.downs[layer]) |*owner| owner.deinit();
        self.downs[layer] = null;
    }
    /// A partial phase failure is terminal. Keep files as evidence, restore the pending views.
    pub fn fail(self: *Session) void {
        if (self.revision < 24) {
            self.restoreLayer(self.revision);
            self.freeLayer(self.revision);
        }
        self.model.mixed = self.mask != 0;
        self.phase = .failed;
    }
    pub fn response(self: *const Session, op: Op, layer: usize) Response {
        return .{ .op = op, .layer = layer, .revision = self.revision, .phase = self.phase, .accepted_layer_mask = self.mask };
    }
    fn path(self: *Session, a: std.mem.Allocator, layer: usize, kind: usize, id: []const u8) ![]const u8 {
        const name = switch (kind) {
            1 => "inputs",
            2 => "act",
            3 => "output",
            else => unreachable,
        };
        return std.fmt.allocPrint(a, "{s}/layer-{d:0>2}/{s}/{s}.bin", .{ self.directory, layer, name, id });
    }
    fn folder(self: *Session, a: std.mem.Allocator, layer: usize, kind: usize) !void {
        const example = try self.path(a, layer, kind, "unused");
        if (kind == 1) {
            const parent = std.fs.path.dirname(std.fs.path.dirname(example).?).?;
            try std.Io.Dir.cwd().createDir(self.io, parent, .default_dir);
        }
        try std.Io.Dir.cwd().createDir(self.io, std.fs.path.dirname(example).?, .default_dir);
    }
    fn read(self: *Session, a: std.mem.Allocator, layer: usize, kind: usize, case: Case) ![]u16 {
        const path_ = try self.path(a, layer, kind, case.id);
        const file = try std.Io.Dir.cwd().openFile(self.io, path_, .{ .follow_symlinks = false });
        defer file.close(self.io);
        const header = try stage.activationHeader(self.parsed.value.manifest, layer, kind, case.tokens.len, case.forced_tokens.len);
        const width: usize = if (kind == 2) 6144 else 4096;
        const rows = case.tokens.len + case.forced_tokens.len;
        const stat = try file.stat(self.io);
        if (stat.kind != .file or stat.size != 160 + rows * width * 2) return error.InvalidSequentialActivation;
        var raw: [160]u8 = undefined;
        if (try file.readPositionalAll(self.io, &raw, 0) != 160 or !std.mem.eql(u8, &raw, &header)) return error.InvalidSequentialActivation;
        const values = try a.alloc(u16, rows * width);
        if (try file.readPositionalAll(self.io, std.mem.sliceAsBytes(values), 160) != rows * width * 2) return error.InvalidSequentialActivation;
        try stage.finite(values);
        return values;
    }
    fn create(self: *Session, a: std.mem.Allocator, layer: usize, kind: usize, case: Case) !struct { file: std.Io.File, record: Record } {
        const path_ = try self.path(a, layer, kind, case.id);
        const file = try std.Io.Dir.cwd().createFile(self.io, path_, .{ .exclusive = true });
        errdefer file.close(self.io);
        const header = try stage.activationHeader(self.parsed.value.manifest, layer, kind, case.tokens.len, case.forced_tokens.len);
        try file.writeStreamingAll(self.io, &header);
        return .{ .file = file, .record = .{ .id = case.id, .split = case.split, .path = path_, .rows = case.tokens.len + case.forced_tokens.len, .prompt_tokens = case.tokens.len, .forced_tokens = case.forced_tokens.len, .row_values = if (kind == 2) 6144 else 4096, .kind = kind } };
    }
    fn remove(self: *Session, a: std.mem.Allocator, layer: usize, kind: usize) !void {
        for (self.parsed.value.cases) |case| try std.Io.Dir.cwd().deleteFile(self.io, try self.path(a, layer, kind, case.id));
    }
    fn install(self: *Session, command: Command, mask: u32) !void {
        if (self.dry_run or (command.layer == 23 and mask == 4)) {
            if (command.payload != null) return error.InvalidSequentialStage;
            return;
        }
        const file = command.payload orelse return error.InvalidSequentialStage;
        const installed = try stage.Installed.read(self.gpa, self.io, self.model.device, file, self.parsed.value.manifest, command.layer, mask, command.revision);
        const block = &self.model.weights.blocks[command.layer];
        if (mask == 3) {
            self.gateups[command.layer] = installed;
            block.gate = installed.projections[0].?;
            block.up = installed.projections[1].?;
        } else {
            self.downs[command.layer] = installed;
            block.down = installed.projections[2].?;
        }
        self.model.mixed = true;
    }
    fn chunk(prompt: usize, at: usize) usize {
        return if (at < prompt) @min(@as(usize, 32), prompt - at) else 1;
    }
    fn inputRows(destination: mtl.Buffer, source: []const u16, at: usize, n: usize, width: usize, offset: usize) void {
        const target = destination.slice(u16, n * 2048);
        for (0..n) |row| @memcpy(target[row * 2048 ..][0..2048], source[(at + row) * width + offset ..][0..2048]);
    }
    fn append(self: *Session, file: std.Io.File, s: *st.Scratch, kind: usize, n: usize) !void {
        if (kind == 2) {
            const values = s.act.slice(u16, n * 6144);
            try stage.finite(values);
            try file.writeStreamingAll(self.io, std.mem.sliceAsBytes(values));
            return;
        }
        var row: [4096]u16 = undefined;
        for (0..n) |i| {
            @memcpy(row[0..2048], s.h.slice(u16, n * 2048)[i * 2048 ..][0..2048]);
            const other = if (kind == 1) s.x else s.r;
            @memcpy(row[2048..4096], other.slice(u16, n * 2048)[i * 2048 ..][0..2048]);
            try stage.finite(&row);
            try file.writeStreamingAll(self.io, std.mem.sliceAsBytes(&row));
        }
    }
    fn pass(self: *Session, a: std.mem.Allocator, layer: usize, kind: usize, rejected: bool) !struct { records: []Record, predictions: []Prediction } {
        try self.folder(a, layer, kind);
        const records = try a.alloc(Record, self.parsed.value.cases.len);
        const predictions = try a.alloc(Prediction, if (layer == 23 and kind == 3) self.parsed.value.cases.len else 0);
        // Reuse one bounded stream cache/scratch across all cases of this layer.
        var cache = try st.Cache.init(self.gpa, self.model.device, 2048);
        defer cache.deinit();
        var scratch = try st.Scratch.initInference(self.gpa, self.model.device, 32, 2048);
        defer scratch.deinit();
        for (self.parsed.value.cases, 0..) |case, index| {
            try self.checkCancel();
            cache.reset();
            var arena = std.heap.ArenaAllocator.init(self.gpa);
            defer arena.deinit();
            const ca = arena.allocator();
            const tokens = try ca.alloc(u32, case.tokens.len + case.forced_tokens.len);
            @memcpy(tokens[0..case.tokens.len], case.tokens);
            @memcpy(tokens[case.tokens.len..], case.forced_tokens);
            const hx: []const u16 = if (kind == 1 and layer != 0) try self.read(ca, layer - 1, 3, case) else if (kind != 1) try self.read(ca, layer, 1, case) else &.{};
            const act: []const u16 = if (kind == 3 and !rejected) try self.read(ca, layer, 2, case) else &.{};
            const output = try self.create(a, layer, kind, case);
            defer output.file.close(self.io);
            records[index] = output.record;
            const predicted = try a.alloc(u32, if (predictions.len != 0) case.forced_tokens.len + 1 else 0);
            var at: usize = 0;
            var pi: usize = 0;
            while (at < tokens.len) {
                try self.checkCancel();
                const n = chunk(case.tokens.len, at);
                if (kind == 1) {
                    if (layer != 0) {
                        inputRows(scratch.h, hx, at, n, 4096, 0);
                        inputRows(scratch.r, hx, at, n, 4096, 2048);
                    }
                    try fwd.layerPass(self.model, &scratch, &cache, layer, tokens[at..][0..n], .inputs, false);
                } else {
                    inputRows(scratch.h, hx, at, n, 4096, 0);
                    inputRows(scratch.x, hx, at, n, 4096, 2048);
                    if (kind == 3 and !rejected) @memcpy(scratch.act.slice(u16, n * 6144), act[at * 6144 ..][0 .. n * 6144]);
                    try fwd.layerPass(self.model, &scratch, &cache, layer, tokens[at..][0..n], if (kind == 2) .act else .output, rejected);
                }
                try self.append(output.file, &scratch, kind, n);
                if (predictions.len != 0 and at + n >= case.tokens.len) {
                    try fwd.layerPass(self.model, &scratch, &cache, layer, tokens[at..][0..n], .head, false);
                    predicted[pi] = try fwd.greedyToken(&scratch);
                    pi += 1;
                }
                at += n;
            }
            if ((try output.file.stat(self.io)).size != 160 + output.record.rows * output.record.row_values * 2) return error.InvalidSequentialActivation;
            if (predictions.len != 0) {
                if (pi != predicted.len) return error.InvalidSequentialPrediction;
                predictions[index] = .{ .id = case.id, .tokens = predicted };
            }
        }
        return .{ .records = records, .predictions = predictions };
    }
    pub fn execute(self: *Session, a: std.mem.Allocator, command: Command) !Response {
        try requirePhase(self.phase, self.revision, command);
        // A validation error before mutation leaves the session usable. Execution errors are terminal.
        try self.checkCancel();
        switch (command.op) {
            .inputs => {
                const output = self.pass(a, command.layer, 1, false) catch |err| {
                    self.fail();
                    return err;
                };
                if (command.layer != 0) self.remove(a, command.layer - 1, 3) catch |err| {
                    self.fail();
                    return err;
                };
                self.phase = .inputs;
                var result = self.response(.inputs, command.layer);
                result.records = output.records;
                return result;
            },
            .gateup => {
                try self.install(command, 3);
                const output = self.pass(a, command.layer, 2, false) catch |err| {
                    self.fail();
                    return err;
                };
                self.phase = .act;
                var result = self.response(.gateup, command.layer);
                result.records = output.records;
                return result;
            },
            .down => {
                try self.install(command, 4);
                self.phase = .candidate;
                return self.response(.down, command.layer);
            },
            .decide => {
                const accept = command.accept.?;
                if (!accept) self.restoreLayer(command.layer);
                const output = self.pass(a, command.layer, 3, !accept) catch |err| {
                    self.fail();
                    return err;
                };
                self.remove(a, command.layer, 1) catch |err| {
                    self.fail();
                    return err;
                };
                self.remove(a, command.layer, 2) catch |err| {
                    self.fail();
                    return err;
                };
                if (accept and !self.dry_run) self.mask |= @as(u32, 1) << @intCast(command.layer);
                if (!accept) self.freeLayer(command.layer);
                self.model.mixed = self.mask != 0;
                self.revision += 1;
                self.phase = if (self.revision == 24) .complete else .ready;
                var result = self.response(.decide, command.layer);
                result.records = output.records;
                result.predictions = output.predictions;
                return result;
            },
            .verify => return self.verify(a, command.layer),
            .begin => unreachable,
        }
    }
    fn verify(self: *Session, a: std.mem.Allocator, layer: usize) !Response {
        // This independent full-forward control executes the current mixed prefix.
        var cache = try st.Cache.init(self.gpa, self.model.device, 2048);
        defer cache.deinit();
        var scratch = try st.Scratch.initInference(self.gpa, self.model.device, 32, 2048);
        defer scratch.deinit();
        scratch.capture = try scratch.memory.alloc(32 * st.capture_inputs.row_values * 2);
        scratch.capture_enabled = true;
        scratch.trace = try scratch.memory.alloc(st.trace_bytes);
        scratch.trace_enabled = true;
        var isolated = try st.Scratch.initInference(self.gpa, self.model.device, 32, 2048);
        defer isolated.deinit();
        var checked: usize = 0;
        for (self.parsed.value.cases) |case| {
            var arena = std.heap.ArenaAllocator.init(self.gpa);
            defer arena.deinit();
            const ca = arena.allocator();
            cache.reset();
            const hx = try self.read(ca, layer, 1, case);
            const act = try self.read(ca, layer, 2, case);
            const tokens = try ca.alloc(u32, case.tokens.len + case.forced_tokens.len);
            @memcpy(tokens[0..case.tokens.len], case.tokens);
            @memcpy(tokens[case.tokens.len..], case.forced_tokens);
            var at: usize = 0;
            while (at < tokens.len) {
                try self.checkCancel();
                const n = chunk(case.tokens.len, at);
                try fwd.run(self.model, &scratch, &.{.{ .cache = &cache, .rows = n }}, tokens[at..][0..n], false, .none);
                const raw = scratch.capture.?.slice(u16, 32 * st.capture_inputs.row_values);
                for (0..n) |row| {
                    const x = raw[32 * st.capture_inputs.offset(layer, false) + row * 2048 ..][0..2048];
                    if (!std.mem.eql(u16, x, hx[(at + row) * 4096 + 2048 ..][0..2048])) return error.SequentialFullForwardMismatch;
                    checked += 2048;
                    if (layer != 23) {
                        const v = raw[32 * st.capture_inputs.offset(layer, true) + row * 6144 ..][0..6144];
                        if (!std.mem.eql(u16, v, act[(at + row) * 6144 ..][0..6144])) return error.SequentialFullForwardMismatch;
                        checked += 6144;
                    }
                }
                inputRows(isolated.h, hx, at, n, 4096, 0);
                @memcpy(isolated.act.slice(u16, n * 6144), act[at * 6144 ..][0 .. n * 6144]);
                try fwd.layerPass(self.model, &isolated, &cache, layer, tokens[at..][0..n], .output, false);
                const trace = scratch.trace.?.slice(u16, st.trace_bytes / 2);
                const base = layer * st.trace_layer_values;
                if (!std.mem.eql(u16, trace[base..][0..2048], isolated.h.slice(u16, n * 2048)[(n - 1) * 2048 ..][0..2048]) or
                    !std.mem.eql(u16, trace[base + 3 * 2048 + 3 * 6144 ..][0..2048], isolated.r.slice(u16, n * 2048)[(n - 1) * 2048 ..][0..2048])) return error.SequentialFullForwardMismatch;
                if (layer == 23) {
                    if (!std.mem.eql(u16, trace[base + 3 * 2048 + 2 * 6144 ..][0..6144], act[(at + n - 1) * 6144 ..][0..6144])) return error.SequentialFullForwardMismatch;
                    checked += 6144;
                }
                checked += 4096;
                at += n;
            }
        }
        _ = a;
        var result = self.response(.verify, layer);
        result.verified_values = checked;
        return result;
    }
};

test "sequential state rejects skipped duplicate stale and invalid decisions" {
    try requirePhase(.ready, 0, .{ .op = .inputs, .revision = 0, .layer = 0 });
    try requirePhase(.candidate, 2, .{ .op = .decide, .revision = 2, .layer = 2, .accept = false });
    try std.testing.expectError(error.InvalidSequentialOrder, requirePhase(.act, 0, .{ .op = .inputs, .revision = 0, .layer = 0 }));
    try std.testing.expectError(error.InvalidSequentialOrder, requirePhase(.ready, 1, .{ .op = .inputs, .revision = 0, .layer = 1 }));
    try std.testing.expectError(error.InvalidSequentialOrder, requirePhase(.candidate, 0, .{ .op = .decide, .revision = 0, .layer = 0 }));
    try std.testing.expectError(error.InvalidSequentialOrder, requirePhase(.failed, 0, .{ .op = .inputs, .revision = 0, .layer = 0 }));
    try std.testing.expectError(error.InvalidSequentialOrder, requirePhase(.ready, 24, .{ .op = .inputs, .revision = 24, .layer = 24 }));
}
test "chunking preserves prefill tails and never batches decode" {
    try std.testing.expectEqual(@as(usize, 32), Session.chunk(33, 0));
    try std.testing.expectEqual(@as(usize, 1), Session.chunk(33, 32));
    try std.testing.expectEqual(@as(usize, 1), Session.chunk(33, 33));
}

test "plan rejects duplicate IDs split leakage invalid source and unforwarded positions" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var cases: [20]Case = undefined;
    for (&cases, 0..) |*case, i| case.* = .{ .id = try std.fmt.allocPrint(arena.allocator(), "case-{d}", .{i}), .split = if (i < 16) "calibration" else "observation", .tokens = &.{ 1, 2 }, .forced_tokens = &.{3}, .positions = &.{ 0, 2 } };
    const manifest: [64]u8 = @splat('a');
    var plan = Plan{ .schema = "q4-sequential-plan-v1", .manifest = &manifest, .core = null, .eos_ids = &.{4}, .cases = &cases };
    try validatePlan(plan, &manifest);
    cases[1].id = cases[0].id;
    try std.testing.expectError(error.InvalidSequentialPlan, validatePlan(plan, &manifest));
    cases[1].id = "restored";
    cases[19].split = "calibration";
    try std.testing.expectError(error.InvalidSequentialPlan, validatePlan(plan, &manifest));
    cases[19].split = "observation";
    cases[0].positions = &.{3};
    try std.testing.expectError(error.InvalidCapturePosition, validatePlan(plan, &manifest));
    cases[0].positions = &.{ 0, 2 };
    cases[0].forced_tokens = &.{4};
    try std.testing.expectError(error.InvalidSequentialPlan, validatePlan(plan, &manifest));
    cases[0].forced_tokens = &.{3};
    plan.core = "unexpected";
    try std.testing.expectError(error.InvalidSequentialPlan, validatePlan(plan, &manifest));
}

test "augmented plan requires exactly 32 calibration and 4 observation cases" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cases: [37]Case = undefined;
    for (&cases, 0..) |*case, i| case.* = .{ .id = try std.fmt.allocPrint(arena.allocator(), "augmented-{d}", .{i}), .split = if (i < 32) "calibration" else "observation", .tokens = &.{ 1, 2 }, .forced_tokens = &.{3}, .positions = &.{ 0, 2 } };
    const manifest: [64]u8 = @splat('a');
    var plan = Plan{ .schema = "q4-sequential-plan-v1", .manifest = &manifest, .core = null, .eos_ids = &.{4}, .cases = cases[0..36] };
    try validatePlan(plan, &manifest);
    plan.cases = cases[0..35];
    try std.testing.expectError(error.InvalidSequentialPlan, validatePlan(plan, &manifest));
    plan.cases = &cases;
    try std.testing.expectError(error.InvalidSequentialPlan, validatePlan(plan, &manifest));
    plan.cases = cases[0..36];
    cases[31].split = "observation";
    try std.testing.expectError(error.InvalidSequentialPlan, validatePlan(plan, &manifest));
    cases[31].split = "calibration";
    cases[35].split = "calibration";
    try std.testing.expectError(error.InvalidSequentialPlan, validatePlan(plan, &manifest));
    cases[35].split = "other";
    try std.testing.expectError(error.InvalidSequentialPlan, validatePlan(plan, &manifest));
    cases[35].split = "observation";
    const original = cases[35];
    cases[35].id = cases[0].id;
    try std.testing.expectError(error.InvalidSequentialPlan, validatePlan(plan, &manifest));
    cases[35] = original;
    cases[35].tokens = &.{ c.vocab, 2 };
    try std.testing.expectError(error.InvalidSequentialPlan, validatePlan(plan, &manifest));
    cases[35] = original;
    cases[35].forced_tokens = &.{4};
    try std.testing.expectError(error.InvalidSequentialPlan, validatePlan(plan, &manifest));
    cases[35] = original;
    cases[35].positions = &.{ 2, 0 };
    try std.testing.expectError(error.InvalidCapturePosition, validatePlan(plan, &manifest));
    cases[35] = original;
    const too_long: [256]u32 = @splat(3);
    cases[35].forced_tokens = &too_long;
    try std.testing.expectError(error.InvalidCaptureShape, validatePlan(plan, &manifest));
    cases[35] = original;
    plan.eos_ids = &.{c.vocab};
    try std.testing.expectError(error.InvalidSequentialPlan, validatePlan(plan, &manifest));
}

fn fixtureLinear(marker: usize) w.Linear {
    const tensor = w.Tensor{ .buffer = undefined, .offset = marker, .bytes = 2, .dtype = .bf16, .rank = 2, .shape = .{ 1, 1, 1, 1 } };
    return .{ .weight = tensor, .scales = tensor, .biases = tensor, .inputs = 1, .outputs = 1, .dense = true };
}

test "failed pending layer restores all original views without undoing accepted prefix" {
    var model: Model = undefined;
    model.mixed = true;
    model.weights.blocks[0].gate = fixtureLinear(999);
    model.weights.blocks[2].gate = fixtureLinear(100);
    model.weights.blocks[2].up = fixtureLinear(101);
    model.weights.blocks[2].down = fixtureLinear(102);
    var session: Session = undefined;
    session.model = &model;
    session.revision = 2;
    session.mask = 1;
    session.phase = .candidate;
    session.original = undefined;
    session.original.blocks[2].gate = fixtureLinear(1);
    session.original.blocks[2].up = fixtureLinear(2);
    session.original.blocks[2].down = fixtureLinear(3);
    session.gateups = @splat(null);
    session.downs = @splat(null);
    session.gateups[2] = .{ .memory = .{ .gpa = std.testing.allocator, .device = undefined } };
    session.downs[2] = .{ .memory = .{ .gpa = std.testing.allocator, .device = undefined } };
    session.fail();
    try std.testing.expectEqual(@as(usize, 1), model.weights.blocks[2].gate.weight.offset);
    try std.testing.expectEqual(@as(usize, 2), model.weights.blocks[2].up.weight.offset);
    try std.testing.expectEqual(@as(usize, 3), model.weights.blocks[2].down.weight.offset);
    try std.testing.expectEqual(@as(usize, 999), model.weights.blocks[0].gate.weight.offset);
    try std.testing.expectEqual(@as(u32, 1), session.mask);
    try std.testing.expectEqual(@as(usize, 2), session.revision);
    try std.testing.expectEqual(Phase.failed, session.phase);
    try std.testing.expect(session.gateups[2] == null and session.downs[2] == null and model.mixed);
}
