// SPDX-License-Identifier: Apache-2.0
//! Luban's isolated original-HF BF16 loader and deterministic inference probe.
//! This executable borrows TensorFold's Metal queue and Qwen state plumbing.
const std = @import("std");
const mtl = @import("metal");
const q = @import("luban_core").qwen35;

extern "c" fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *anyopaque) c_int;
extern "c" fn getpid() c_int;

const Case = struct {
    id: []const u8,
    tokens: []const u32,
    max_tokens: usize = 32,
    eos_ids: []const u32 = &.{248044},
    dump_logits: bool = false,
};

fn emit(io: std.Io, a: std.mem.Allocator, value: anytype) !void {
    const bytes = try std.json.Stringify.valueAlloc(a, value, .{});
    defer a.free(bytes);
    try std.Io.File.stdout().writeStreamingAll(io, bytes);
    try std.Io.File.stdout().writeStreamingAll(io, "\n");
}

fn memory() !struct { physical_footprint_bytes: u64, lifetime_peak_physical_footprint_bytes: u64 } {
    var usage: [296]u8 align(8) = @splat(0);
    if (proc_pid_rusage(getpid(), 4, &usage) != 0) return error.MemoryMeasurementFailed;
    return .{ .physical_footprint_bytes = std.mem.readInt(u64, usage[72..80], .little),
        .lifetime_peak_physical_footprint_bytes = std.mem.readInt(u64, usage[240..248], .little) };
}

fn now(io: std.Io) i128 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 2 and std.mem.eql(u8, args[1], "--version")) {
        try emit(init.io, a, .{ .name = "luban-native", .format = "original-hf-bf16",
            .backend = "metal", .training = false, .product_adopted = false });
        return;
    }
    if (args.len != 5 and args.len != 8) return error.ExpectedModelManifestSHA256CasesOutputAndOptionalCoreProvenanceSHA;
    var approved: [32]u8 = undefined;
    if (args[2].len != 64) return error.InvalidManifestSHA256;
    if ((try std.fmt.hexToBytes(&approved, args[2])).len != 32) return error.InvalidManifestSHA256;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[3], a, .limited(4 << 20));
    const cases = try std.json.parseFromSlice([]Case, a, bytes, .{ .allocate = .alloc_always });
    if (cases.value.len == 0) return error.EmptyCases;
    for (cases.value) |case| {
        if (case.tokens.len == 0 or case.tokens.len > 1024 or case.max_tokens == 0 or case.max_tokens > 512 or case.eos_ids.len == 0)
            return error.InvalidCase;
        for (case.tokens) |id| if (id >= q.config.vocab) return error.InvalidQwenToken;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const load_start = now(init.io);
    var core: ?q.Model.CoreOverride = null;
    if (args.len == 8) {
        var provenance_sha: [32]u8 = undefined;
        if (args[7].len != 64) return error.InvalidProvenanceSHA256;
        if ((try std.fmt.hexToBytes(&provenance_sha, args[7])).len != 32) return error.InvalidProvenanceSHA256;
        core = .{ .path = args[5], .provenance_path = args[6], .approved_provenance_sha256 = provenance_sha };
    }
    const model = try q.Model.loadNative(init.gpa, init.io, args[1], approved, core);
    defer model.deinit();
    const load_ns = now(init.io) - load_start;
    const binding = model.base_binding orelse return error.UnboundNativeManifest;
    try emit(init.io, a, .{ .event = "loaded", .name = "luban-native", .load_ns = load_ns,
        .weight_format = "BF16 with original FP32 A_log and gated norm", .memory = try memory(),
        .candidate_loaded = core != null, .training = false, .product_adopted = false,
        .base_identity = @as([]const u8, &binding.identity),
        .manifest_sha256 = @as([]const u8, &binding.manifest_sha256),
        .checkpoint_sha256 = @as([]const u8, &binding.checkpoint_sha256),
        .original_down_sha256 = @as([]const u8, &binding.original_down_sha256) });
    var scratch = try q.state.Scratch.initInference(init.gpa, model.device, 32, 1600);
    defer scratch.deinit();
    for (cases.value, 0..) |case, case_index| {
        var cache = try q.state.Cache.init(init.gpa, model.device, case.tokens.len + case.max_tokens + 32);
        defer cache.deinit();
        try emit(init.io, a, .{ .event = "request_started", .case_index = case_index, .id = case.id });
        const started = now(init.io);
        var at: usize = 0;
        var logits: std.ArrayList(u8) = .empty;
        defer logits.deinit(init.gpa);
        while (at < case.tokens.len) {
            const n: usize = @min(@as(usize, 32), case.tokens.len - at);
            try q.forward.run(model, &scratch, &.{.{ .cache = &cache, .rows = n }}, case.tokens[at..][0..n], false, if (case.dump_logits) .all else .last);
            if (case.dump_logits) try logits.appendSlice(init.gpa, scratch.logits.contents()[0 .. n * q.config.vocab * 2]);
            at += n;
        }
        const prefill_ns = now(init.io) - started;
        var generated: std.ArrayList(u32) = .empty;
        defer generated.deinit(init.gpa);
        var timestamps: std.ArrayList(i128) = .empty;
        defer timestamps.deinit(init.gpa);
        var stop = false;
        for (0..case.max_tokens) |_| {
            const token = try q.backend.draw(init.gpa, cache.logits.slice(u16, q.config.vocab), null, cache.len);
            try generated.append(init.gpa, token);
            try timestamps.append(init.gpa, now(init.io) - started);
            // The measured timestamp is before log formatting and output delivery.
            try emit(init.io, a, .{ .event = "token", .case_index = case_index, .id = case.id,
                .token = token, .elapsed_ns = timestamps.getLast() });
            for (case.eos_ids) |eos| if (token == eos) { stop = true; break; };
            if (stop or generated.items.len == case.max_tokens) break;
            try q.forward.run(model, &scratch, &.{.{ .cache = &cache, .rows = 1 }}, &.{token}, false, .last);
        }
        if (case.dump_logits) {
            const path = try std.fmt.allocPrint(a, "{s}/{d}.logits.bf16", .{ args[4], case_index });
            try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = path, .data = logits.items });
        }
        const ttft_ns: ?i128 = if (timestamps.items.len > 0) timestamps.items[0] else null;
        const decode_ns: ?i128 = if (timestamps.items.len > 1) timestamps.getLast() - timestamps.items[0] else null;
        try emit(init.io, a, .{ .event = "result", .case_index = case_index, .id = case.id,
            .tokens = generated.items, .prompt_tokens = case.tokens.len, .finish_reason = if (stop) "stop" else "length",
            .complete = stop, .prefill_ns = prefill_ns, .ttft_ns = ttft_ns, .decode_ns = decode_ns,
            .generated_tokens = generated.items.len, .token_timestamps_ns = timestamps.items,
            .memory = try memory(), .logits_dumped = case.dump_logits });
    }
    try emit(init.io, a, .{ .event = "done", .memory = try memory() });
}
