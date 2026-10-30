// SPDX-License-Identifier: Apache-2.0
//! Resident original-HF worker with explicit mixed-precision candidates; stdout is JSON lines only.
const std = @import("std");
const mtl = @import("metal");
const q = @import("luban_core").qwen35;
const seq = q.forward.sequential;

extern "c" fn proc_pid_rusage(pid: c_int, flavor: c_int, buffer: *anyopaque) c_int;
extern "c" fn getpid() c_int;

const max_context = 32768;
const max_line = 4 << 20;
const retained_context = 8192;
const RestoreProjection = enum { all, gate, up, down };
const Request = struct {
    id: []const u8,
    tokens: []const u32 = &.{},
    max_tokens: usize = 0,
    eos_ids: []const u32 = &.{},
    seq_op: ?seq.Op = null,
    seq_revision: usize = 0,
    seq_layer: usize = 0,
    seq_plan: ?[]const u8 = null,
    seq_payload: ?[]const u8 = null,
    seq_dry_run: bool = false,
    seq_accept: ?bool = null,
    temperature: f64 = 0,
    seed: u64 = 0,
    cache_reuse: bool = false,
    forced_tokens: ?[]const u32 = null,
    tail_only: bool = false,
    capture_positions: ?[]const u32 = null,
    restore_layer: ?usize = null,
    restore_layers: ?[]const usize = null,
    audit_quantization: bool = false,
    audit_layer: ?usize = null,
    audit_projection: ?q.Model.audit.Projection = null,
    audit_input_path: ?[]const u8 = null,
    restore_projection: ?RestoreProjection = null,
    probe_tokens: ?[]const u32 = null,
};

/// The model and applied core are fixed for this worker's entire lifetime.
/// Only exact token prefixes may reuse its one complete neural-state slot.
const Prefix = struct {
    tokens: std.ArrayList(u32) = .empty,

    fn matches(self: *const Prefix, request: Request) bool {
        // Neural state is fixed by the committed token history, not by the
        // next request's seed, temperature or stopping tokens.
        return request.cache_reuse and self.tokens.items.len > 0 and
            self.tokens.items.len <= request.tokens.len and
            std.mem.eql(u32, self.tokens.items, request.tokens[0..self.tokens.items.len]);
    }

    fn remember(self: *Prefix, a: std.mem.Allocator, request: Request, generated: []const u32, committed: usize) !void {
        // The last emitted token has not been forwarded. Never claim its
        // recurrence/KV state exists or perform an extra forward to cache it.
        if (committed < request.tokens.len or committed - request.tokens.len > generated.len)
            return error.InvalidPrefixState;
        self.tokens.clearRetainingCapacity();
        try self.tokens.appendSlice(a, request.tokens);
        try self.tokens.appendSlice(a, generated[0 .. committed - request.tokens.len]);
    }
};

/// Rows per ordinary prefill pass; scratch is sized to match so long prompts
/// need fewer GPU submissions. Decode and diagnostic traces are unchanged.
const prefill_rows: usize = 128;

const Resident = struct {
    cache: ?q.state.Cache = null,
    scratch: ?q.state.Scratch = null,
    prefix: Prefix = .{},

    fn clear(self: *Resident, a: std.mem.Allocator) void {
        if (self.cache) |*cache| cache.deinit();
        if (self.scratch) |*scratch| scratch.deinit();
        self.prefix.tokens.deinit(a);
        self.* = .{};
    }

    fn prepare(self: *Resident, a: std.mem.Allocator, model: *q.Model, request: Request) !usize {
        const need = request.tokens.len + request.max_tokens;
        if (self.cache) |*cache| {
            if (cache.capacity >= need) {
                if (self.prefix.matches(request) and self.prefix.tokens.items.len == cache.len)
                    return cache.len;
                cache.reset();
                self.prefix.tokens.clearRetainingCapacity();
                return 0;
            }
        }
        self.clear(a);
        // Grow only to the next required bucket; never reserve the full context
        // for an ordinary chat. Capacity growth deliberately recomputes cold.
        var capacity: usize = 1024;
        while (capacity < need) capacity *= 2;
        self.cache = try q.state.Cache.init(a, model.device, capacity);
        errdefer self.clear(a);
        self.scratch = try q.state.Scratch.initInference(a, model.device, prefill_rows, capacity);
        return 0;
    }
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
    return .{ .physical_footprint_bytes = std.mem.readInt(u64, usage[72..80], .little), .lifetime_peak_physical_footprint_bytes = std.mem.readInt(u64, usage[240..248], .little) };
}

fn pagingStats(model: *const q.Model) struct { enabled: bool, store_bytes: usize, slot_bytes: usize, read_bytes: u64, read_calls: u64 } {
    if (model.layer_pager) |pager| return .{ .enabled = true, .store_bytes = pager.store_bytes, .slot_bytes = pager.slot_bytes, .read_bytes = pager.read_bytes, .read_calls = pager.read_calls };
    return .{ .enabled = false, .store_bytes = 0, .slot_bytes = 0, .read_bytes = 0, .read_calls = 0 };
}

fn now(io: std.Io) i128 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

fn digest(hex: []const u8) ![32]u8 {
    if (hex.len != 64) return error.InvalidSHA256;
    var value: [32]u8 = undefined;
    if ((try std.fmt.hexToBytes(&value, hex)).len != 32) return error.InvalidSHA256;
    return value;
}

/// The parent uses an independent pipe so cancellation never needs stdin parsing.
/// Drain all currently available bytes after cancellation to keep the next
/// request independent. A broken cancel channel is a real request error.
fn cancelled(fd: c_int) !bool {
    var signalled = false;
    while (true) {
        var fds = [_]std.c.pollfd{.{ .fd = fd, .events = std.c.POLL.IN, .revents = 0 }};
        const ready = std.c.poll(&fds, 1, 0);
        if (ready < 0) {
            if (std.c.errno(ready) == .INTR) continue;
            return error.CancelPollFailed;
        }
        if (ready == 0) return signalled;
        if (fds[0].revents & (std.c.POLL.NVAL | std.c.POLL.ERR) != 0) return error.CancelChannelFailed;
        if (fds[0].revents & std.c.POLL.IN != 0) {
            var bytes: [256]u8 = undefined;
            const n = std.c.read(fd, &bytes, bytes.len);
            if (n < 0 and std.c.errno(n) == .INTR) continue;
            if (n <= 0) return error.CancelChannelClosed;
            signalled = true;
            continue;
        }
        if (fds[0].revents & std.c.POLL.HUP != 0) return error.CancelChannelClosed;
        return signalled;
    }
}

const Input = struct {
    buffer: [4096]u8 = undefined,
    at: usize = 0,
    end: usize = 0,

    /// Preserve buffered future lines; an oversized line is drained through its
    /// newline and reported once, allowing subsequent requests to recover.
    fn line(self: *Input, a: std.mem.Allocator) !?[]const u8 {
        var out: std.ArrayList(u8) = .empty;
        var oversized = false;
        while (true) {
            if (self.at == self.end) {
                const n = std.c.read(0, &self.buffer, self.buffer.len);
                if (n < 0) {
                    if (std.c.errno(n) == .INTR) continue;
                    return error.StdinReadFailed;
                }
                if (n == 0) {
                    if (oversized) return error.RequestLineTooLong;
                    return if (out.items.len == 0) null else out.items;
                }
                self.at = 0;
                self.end = @intCast(n);
            }
            const rest = self.buffer[self.at..self.end];
            const newline = std.mem.indexOfScalar(u8, rest, '\n');
            const length = newline orelse rest.len;
            if (!oversized) {
                if (length > max_line - out.items.len) oversized = true else try out.appendSlice(a, rest[0..length]);
            }
            self.at += length + @as(usize, if (newline != null) 1 else 0);
            if (newline != null) {
                if (oversized) return error.RequestLineTooLong;
                return out.items;
            }
        }
    }
};

fn validate(request: Request) !void {
    if (request.id.len == 0 or request.id.len > 128) return error.InvalidRequestID;
    if (request.tokens.len == 0 or request.tokens.len > max_context or request.max_tokens == 0 or request.max_tokens > 2048 or
        request.max_tokens > max_context - request.tokens.len or request.eos_ids.len == 0 or request.eos_ids.len > 64)
        return error.InvalidRequestLength;
    if (!std.math.isFinite(request.temperature) or request.temperature < 0 or request.temperature > 2) return error.InvalidTemperature;
    for (request.tokens) |id| if (id >= q.config.vocab) return error.InvalidQwenToken;
    for (request.eos_ids) |id| if (id >= q.config.vocab) return error.InvalidQwenEOS;
}

fn generate(init: std.process.Init, model: *q.Model, fd: c_int, resident: *Resident, request: Request) !void {
    errdefer resident.clear(init.gpa);
    try validate(request);
    const cached_tokens = try resident.prepare(init.gpa, model, request);
    const cache = &resident.cache.?;
    const scratch = &resident.scratch.?;
    const output_head: q.forward.Head = if (request.temperature <= 0) .greedy else .last;
    var generated: std.ArrayList(u32) = .empty;
    defer generated.deinit(init.gpa);
    try generated.ensureTotalCapacity(init.gpa, request.max_tokens);
    const started = now(init.io);
    var at: usize = cached_tokens;
    var greedy_ready = false;
    var abort = try cancelled(fd);
    while (at < request.tokens.len and !abort) {
        const n = @min(prefill_rows, request.tokens.len - at);
        const head: q.forward.Head = if (at + n == request.tokens.len) output_head else .none;
        try q.forward.run(model, scratch, &.{.{ .cache = cache, .rows = n }}, request.tokens[at..][0..n], false, head);
        greedy_ready = head == .greedy;
        at += n;
        abort = try cancelled(fd);
    }
    const prefill_ns = now(init.io) - started;
    var stop = false;
    var ttft_ns: ?i128 = null;
    var last_token_ns: ?i128 = null;
    while (!abort and generated.items.len < request.max_tokens) {
        abort = try cancelled(fd);
        if (abort) break;
        // A fully cached prompt has not run this request's GPU head. Draw its
        // first token from the retained logits rather than scratch token IDs.
        const token = if (greedy_ready) try q.forward.greedyToken(scratch) else try q.backend.draw(init.gpa, cache.logits.slice(u16, q.config.vocab), .{ .seed = request.seed, .temperature = request.temperature, .top_k = 0, .top_p = 0.95, .min_p = 0 }, cache.len);
        // Sampling can also take time; honor a cancel before publishing a token.
        abort = try cancelled(fd);
        if (abort) break;
        generated.appendAssumeCapacity(token);
        const elapsed_ns = now(init.io) - started;
        if (ttft_ns == null) ttft_ns = elapsed_ns;
        last_token_ns = elapsed_ns;
        try emit(init.io, init.gpa, .{ .event = "token", .id = request.id, .token = token, .elapsed_ns = elapsed_ns });
        for (request.eos_ids) |eos| if (token == eos) {
            stop = true;
            break;
        };
        if (stop or generated.items.len == request.max_tokens) break;
        abort = try cancelled(fd);
        if (abort) break;
        try q.forward.run(model, scratch, &.{.{ .cache = cache, .rows = 1 }}, &.{token}, false, output_head);
        greedy_ready = output_head == .greedy;
    }
    const capacity = cache.capacity;
    if (abort or !request.cache_reuse or capacity > retained_context) {
        resident.clear(init.gpa);
    } else {
        try resident.prefix.remember(init.gpa, request, generated.items, cache.len);
    }
    try emit(init.io, init.gpa, .{ .event = "result", .id = request.id, .tokens = generated.items, .prompt_tokens = request.tokens.len, .generated_tokens = generated.items.len, .finish_reason = if (abort) "cancelled" else if (stop) "stop" else "length", .complete = stop, .cached_prompt_tokens = cached_tokens, .cache_capacity = if (resident.cache != null) capacity else @as(usize, 0), .execution_skip_mask = model.execution_policy.skip_mask, .execution_exit_after = model.execution_policy.exit_after, .layer_paging = pagingStats(model), .prefill_ns = prefill_ns, .ttft_ns = ttft_ns, .decode_ns = if (generated.items.len > 1) last_token_ns.? - ttft_ns.? else @as(?i128, null), .memory = try memory() });
}

fn saveTrace(init: std.process.Init, directory: []const u8, id: []const u8, step: usize, kind: []const u8, values: []const u16) ![]const u8 {
    for (values) |v| if (v & 0x7f80 == 0x7f80) return error.NonfiniteTrace;
    const path = try std.fmt.allocPrint(init.arena.allocator(), "{s}/{s}-step{d:0>4}.{s}.bf16", .{ directory, id, step, kind });
    const file = try std.Io.Dir.cwd().createFile(init.io, path, .{ .exclusive = true });
    defer file.close(init.io);
    try file.writeStreamingAll(init.io, std.mem.sliceAsBytes(values));
    return path;
}

const tail_values = 2 * q.config.hidden + q.config.intermediate;

fn copyTail(trace: []const u16, tail: *[tail_values]u16) !void {
    if (trace.len != q.state.trace_bytes / 2) return error.InvalidTailTraceShape;
    const last = trace[(q.config.layers - 1) * q.state.trace_layer_values ..];
    @memcpy(tail[0..q.config.hidden], last[0..q.config.hidden]);
    const act = 3 * q.config.hidden + 2 * q.config.intermediate;
    @memcpy(tail[q.config.hidden..], last[act..]);
}

fn validateTrace(request: Request) !void {
    const forced = request.forced_tokens orelse return error.InvalidTraceRequest;
    const limit: usize = if (request.tail_only) 255 else 64;
    if (forced.len > limit or request.max_tokens != forced.len + 1 or request.cache_reuse or request.temperature != 0)
        return error.InvalidTraceRequest;
    for (request.id) |ch| if (!std.ascii.isAlphanumeric(ch) and ch != '-' and ch != '_') return error.InvalidTraceID;
    for (forced) |token| {
        if (token >= q.config.vocab) return error.InvalidQwenToken;
        if (request.tail_only) for (request.eos_ids) |eos| {
            if (token == eos) return error.UnforwardedCaptureEOS;
        };
    }
}

/// Prompt chunks and single-token decode are identical to ordinary greedy
/// generation. Forced history replaces only the token chosen for the next step.
fn diagnose(init: std.process.Init, model: *q.Model, fd: c_int, resident: *Resident, request: Request, directory: ?[]const u8) !void {
    errdefer resident.clear(init.gpa);
    try validate(request);
    const path = directory orelse return error.ResearchTraceNotEnabled;
    try validateTrace(request);
    const forced = request.forced_tokens.?;
    _ = try resident.prepare(init.gpa, model, request);
    defer resident.clear(init.gpa);
    const cache = &resident.cache.?;
    const scratch = &resident.scratch.?;
    if (request.probe_tokens == null) scratch.trace = try scratch.memory.alloc(q.state.trace_bytes);
    var at: usize = 0;
    while (at < request.tokens.len) {
        if (try cancelled(fd)) return error.TraceCancelled;
        const n = @min(@as(usize, 32), request.tokens.len - at);
        const final = at + n == request.tokens.len;
        scratch.trace_enabled = final and request.probe_tokens == null;
        try q.forward.run(model, scratch, &.{.{ .cache = cache, .rows = n }}, request.tokens[at..][0..n], false, if (final) .greedy else .none);
        at += n;
    }
    var greedy: [256]u32 = undefined;
    for (0..forced.len + 1) |step| {
        if (try cancelled(fd)) return error.TraceCancelled;
        greedy[step] = try q.forward.greedyToken(scratch);
        if (request.probe_tokens) |tokens| {
            const logits = cache.logits.slice(u16, q.config.vocab);
            for (logits) |value| if (value & 0x7f80 == 0x7f80) return error.NonfiniteTrace;
            var values: [16]f32 = undefined;
            for (tokens, 0..) |token, i| values[i] = @bitCast(@as(u32, logits[token]) << 16);
            try emit(init.io, init.gpa, .{ .event = "trace", .id = request.id, .step = step, .top1 = greedy[step], .probe_tokens = tokens, .probe_logits = values[0..tokens.len], .committed_tokens = cache.len });
        } else {
            const logits_path = try saveTrace(init, path, request.id, step, "logits", cache.logits.slice(u16, q.config.vocab));
            const trace = scratch.trace.?.slice(u16, q.state.trace_bytes / 2);
            if (request.tail_only) {
                var tail: [tail_values]u16 = undefined;
                try copyTail(trace, &tail);
                const tail_path = try saveTrace(init, path, request.id, step, "tail", &tail);
                try emit(init.io, init.gpa, .{ .event = "trace", .id = request.id, .step = step, .top1 = greedy[step], .logits_path = logits_path, .tail_path = tail_path, .tail_only = true, .layer = q.config.layers - 1, .tail_values = tail_values, .vocab = q.config.vocab, .committed_tokens = cache.len });
            } else {
                const layers_path = try saveTrace(init, path, request.id, step, "layers", trace);
                try emit(init.io, init.gpa, .{ .event = "trace", .id = request.id, .step = step, .top1 = greedy[step], .logits_path = logits_path, .layers_path = layers_path, .vocab = q.config.vocab, .layers = q.config.layers, .layer_values = q.state.trace_layer_values, .committed_tokens = cache.len });
            }
        }
        if (step < forced.len) {
            try q.forward.run(model, scratch, &.{.{ .cache = cache, .rows = 1 }}, forced[step..][0..1], false, .greedy);
        }
    }
    try emit(init.io, init.gpa, .{ .event = "result", .id = request.id, .tokens = greedy[0 .. forced.len + 1], .forced_tokens = forced, .prompt_tokens = request.tokens.len, .generated_tokens = forced.len + 1, .finish_reason = "trace", .complete = true, .trace_only = true, .cached_prompt_tokens = @as(usize, 0), .memory = try memory() });
}

fn validateCapture(request: Request) !void {
    const forced = request.forced_tokens orelse return error.InvalidCaptureRequest;
    const positions = request.capture_positions orelse return error.InvalidCaptureRequest;
    if (request.max_tokens != forced.len + 1 or request.cache_reuse or request.temperature != 0 or
        request.restore_layer != null or request.restore_layers != null or request.restore_projection != null or
        request.audit_quantization or request.audit_layer != null or request.audit_projection != null or
        request.audit_input_path != null or request.probe_tokens != null) return error.InvalidCaptureRequest;
    try q.state.capture_inputs.validate(request.tokens.len, forced.len, positions);
    for (forced) |token| {
        if (token >= q.config.vocab) return error.InvalidQwenToken;
        for (request.eos_ids) |eos| if (token == eos) return error.UnforwardedCaptureEOS;
    }
}

test "native capture rejects unforwarded EOS invalid positions and mixed research requests" {
    const initial: Request = .{ .id = "capture", .tokens = &.{ 1, 2, 3 }, .max_tokens = 3, .eos_ids = &.{9}, .forced_tokens = &.{ 4, 5 }, .capture_positions = &.{ 0, 2, 3, 4 } };
    try validateCapture(initial);
    var r = initial;
    r.forced_tokens = &.{ 4, 9 };
    try std.testing.expectError(error.UnforwardedCaptureEOS, validateCapture(r));
    r = initial;
    r.capture_positions = &.{ 0, 5 };
    try std.testing.expectError(error.InvalidCapturePosition, validateCapture(r));
    r = initial;
    r.capture_positions = &.{ 2, 2 };
    try std.testing.expectError(error.InvalidCapturePosition, validateCapture(r));
    r = initial;
    r.cache_reuse = true;
    try std.testing.expectError(error.InvalidCaptureRequest, validateCapture(r));
    r = initial;
    r.temperature = 0.6;
    try std.testing.expectError(error.InvalidCaptureRequest, validateCapture(r));
    r = initial;
    r.max_tokens = 4;
    try std.testing.expectError(error.InvalidCaptureRequest, validateCapture(r));
    r = initial;
    r.probe_tokens = &.{1};
    try std.testing.expectError(error.InvalidCaptureRequest, validateCapture(r));
    r = initial;
    r.restore_layers = &.{0};
    try std.testing.expectError(error.InvalidCaptureRequest, validateCapture(r));
    r = initial;
    r.audit_quantization = true;
    try std.testing.expectError(error.InvalidCaptureRequest, validateCapture(r));
}

/// Fixed native token history, with bounded copies of only MLP input tensors.
/// Ordinary prefill/decode projection shapes, state updates and heads are unchanged.
fn captureInputs(init: std.process.Init, model: *q.Model, fd: c_int, resident: *Resident, request: Request, directory: ?[]const u8) !void {
    const capture = q.state.capture_inputs;
    const path = directory orelse return error.ResearchCaptureNotEnabled;
    try validateCapture(request);
    if (model.config.format != .bf16 or model.base_binding == null or model.core_binding != null or model.original != null or
        (model.residual_rank != 0 and (!model.mixed or model.residual_rank != 8))) return error.InvalidCaptureSource;
    const forced = request.forced_tokens.?;
    _ = try resident.prepare(init.gpa, model, request);
    defer resident.clear(init.gpa);
    const cache = &resident.cache.?;
    const scratch = &resident.scratch.?;
    scratch.capture = try scratch.memory.alloc(scratch.rows * capture.row_values * 2);
    var writer = try capture.Writer.initForSource(init.io, init.gpa, path, request.id, if (model.mixed) .packed_q4 else .original_bf16, &model.base_binding.?.manifest_sha256, request.tokens.len, forced.len, request.capture_positions.?);
    defer writer.deinit();
    var at: usize = 0;
    while (at < request.tokens.len) {
        if (try cancelled(fd)) return error.TraceCancelled;
        const n = @min(@as(usize, 32), request.tokens.len - at);
        scratch.capture_enabled = writer.needs(at, n);
        try q.forward.run(model, scratch, &.{.{ .cache = cache, .rows = n }}, request.tokens[at..][0..n], false, if (at + n == request.tokens.len) .greedy else .none);
        if (scratch.capture_enabled) try writer.append(scratch.capture.?.slice(u16, scratch.rows * capture.row_values), scratch.rows, at, n);
        at += n;
    }
    var greedy: [256]u32 = undefined;
    greedy[0] = try q.forward.greedyToken(scratch);
    for (forced, 0..) |_, step| {
        if (try cancelled(fd)) return error.TraceCancelled;
        const position = request.tokens.len + step;
        scratch.capture_enabled = writer.needs(position, 1);
        try q.forward.run(model, scratch, &.{.{ .cache = cache, .rows = 1 }}, forced[step..][0..1], false, .greedy);
        if (scratch.capture_enabled) try writer.append(scratch.capture.?.slice(u16, scratch.rows * capture.row_values), scratch.rows, position, 1);
        greedy[step + 1] = try q.forward.greedyToken(scratch);
    }
    if (try cancelled(fd)) return error.TraceCancelled;
    try writer.finish();
    try emit(init.io, init.gpa, .{ .event = "capture", .id = request.id, .path = writer.path, .rows = writer.count, .row_values = capture.row_values, .positions = request.capture_positions.?, .prompt_tokens = request.tokens.len, .forced_tokens = forced.len });
    try emit(init.io, init.gpa, .{ .event = "result", .id = request.id, .tokens = greedy[0 .. forced.len + 1], .prompt_tokens = request.tokens.len, .generated_tokens = forced.len + 1, .finish_reason = "capture", .complete = true, .capture_only = true, .cached_prompt_tokens = @as(usize, 0), .memory = try memory() });
}

fn validateResearch(request: Request, enabled: bool) !void {
    if (request.tail_only and (enabled or request.forced_tokens == null or request.seq_op != null or
        request.capture_positions != null or request.probe_tokens != null or request.restore_layer != null or
        request.restore_layers != null or request.restore_projection != null or request.audit_quantization or
        request.audit_layer != null or request.audit_projection != null or request.audit_input_path != null))
        return error.InvalidTailTraceMode;
    if (request.restore_layer != null and request.restore_layers != null) return error.InvalidRestoreRequest;
    const restore = request.restore_layer != null or request.restore_layers != null;
    if (restore != (request.restore_projection != null)) return error.InvalidRestoreRequest;
    if (restore and !enabled) return error.ResearchInterventionNotEnabled;
    if (restore and request.cache_reuse) return error.InvalidRestoreRequest;
    if (request.restore_layer) |layer| if (layer >= q.config.layers) return error.InvalidRestoreRequest;
    if (request.restore_layers) |layers| {
        if (layers.len == 0 or layers.len > q.config.layers) return error.InvalidRestoreRequest;
        var seen: [q.config.layers]bool = @splat(false);
        for (layers) |layer| {
            if (layer >= q.config.layers or seen[layer]) return error.InvalidRestoreRequest;
            seen[layer] = true;
        }
    }
    if (request.audit_quantization) {
        if (!enabled or restore or request.forced_tokens != null or request.probe_tokens != null or request.cache_reuse or
            request.audit_layer == null or request.audit_projection == null or request.audit_input_path == null) return error.InvalidAuditRequest;
        if (request.audit_layer.? >= q.config.layers or !std.fs.path.isAbsolute(request.audit_input_path.?)) return error.InvalidAuditRequest;
    } else if (request.audit_layer != null or request.audit_projection != null or request.audit_input_path != null) return error.InvalidAuditRequest;
    if (request.probe_tokens) |tokens| {
        if (request.forced_tokens == null or tokens.len == 0 or tokens.len > 16) return error.InvalidProbeTokens;
        for (tokens) |token| if (token >= q.config.vocab) return error.InvalidProbeTokens;
    }
}

fn dispatch(init: std.process.Init, model: *q.Model, fd: c_int, resident: *Resident, request: Request, trace_directory: ?[]const u8, intervention: bool, audit_directory: ?[]const u8, capture_directory: ?[]const u8) !void {
    errdefer resident.clear(init.gpa);
    try validate(request);
    try validateResearch(request, intervention);
    if (request.capture_positions != null) {
        try captureInputs(init, model, fd, resident, request, capture_directory);
        return;
    }
    if (request.audit_quantization) {
        if (try cancelled(fd)) return error.TraceCancelled;
        const result = try q.Model.audit.run(model, init.io, audit_directory orelse return error.ResearchAuditNotEnabled, request.id, request.audit_layer.?, request.audit_projection.?, request.audit_input_path.?);
        try emit(init.io, init.gpa, .{ .event = "audit", .id = request.id, .audit = result });
        try emit(init.io, init.gpa, .{ .event = "result", .id = request.id, .finish_reason = "audit", .complete = true, .audit_only = true });
        return;
    }
    // Views only: restore every changed layer even if any subsequent step fails.
    const Saved = struct { layer: usize, gate: q.weights.Linear, up: q.weights.Linear, down: q.weights.Linear };
    var saved: [q.config.layers]Saved = undefined;
    var saved_count: usize = 0;
    defer for (saved[0..saved_count]) |old| {
        const block = &model.weights.blocks[old.layer];
        block.gate = old.gate;
        block.up = old.up;
        block.down = old.down;
    };
    const single = [1]usize{request.restore_layer orelse 0};
    const layers = request.restore_layers orelse if (request.restore_layer != null) &single else &.{};
    for (layers) |layer| {
        const original = &(model.original orelse return error.OriginalWeightsNotRetained).weights.blocks[layer];
        const block = &model.weights.blocks[layer];
        saved[saved_count] = .{ .layer = layer, .gate = block.gate, .up = block.up, .down = block.down };
        saved_count += 1;
        const projection = request.restore_projection.?;
        if (projection == .all or projection == .gate) block.gate = original.gate;
        if (projection == .all or projection == .up) block.up = original.up;
        if (projection == .all or projection == .down) block.down = original.down;
    }
    if (request.forced_tokens != null)
        try diagnose(init, model, fd, resident, request, trace_directory)
    else
        try generate(init, model, fd, resident, request);
}

pub fn main(init: std.process.Init) !void {
    run(init) catch |err| {
        try emit(init.io, init.gpa, .{ .event = "error", .id = @as(?[]const u8, null), .@"error" = @errorName(err) });
        return err;
    };
}

const ClipProfile = struct { values: []f32, preserve_last_mlp_layers: usize, group_profile: bool = false };

fn loadClipProfile(io: std.Io, gpa: std.mem.Allocator, path: []const u8) !ClipProfile {
    if (path.len == 0 or path[0] != '/') return error.InvalidMixedClipProfile;
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(raw);
    var lines = std.mem.splitScalar(u8, raw, '\n');
    const header = std.mem.trim(u8, lines.next() orelse return error.InvalidMixedClipProfile, " \t\r");
    const mode = std.mem.trim(u8, lines.next() orelse return error.InvalidMixedClipProfile, " \t\r");
    const count_text = std.mem.trim(u8, lines.next() orelse return error.InvalidMixedClipProfile, " \t\r");
    const group_profile = std.mem.eql(u8, header, "LUBAN_Q4_GROUP_CLIP_V1");
    if (!group_profile and !std.mem.eql(u8, header, "LUBAN_Q4_CLIP_V1"))
        return error.InvalidMixedClipProfile;
    const count = std.fmt.parseInt(usize, count_text, 10) catch return error.InvalidMixedClipProfile;
    const preserve_last_mlp_layers: usize = if (group_profile and std.mem.eql(u8, mode, "mixed-q4-g32-group")) 0 else if (group_profile and std.mem.eql(u8, mode, "mixed-q4-g32-group-early4")) 4 else if (group_profile and std.mem.eql(u8, mode, "mixed-q4-g32-group-early8")) 8 else if ((std.mem.eql(u8, mode, "mixed-q4-mlp") or std.mem.eql(u8, mode, "mixed-q4-g32-calibrated")) and count == 71) 0 else if (std.mem.eql(u8, mode, "mixed-q4-early") and count == 60) 4 else return error.InvalidMixedClipProfile;
    const profile = try gpa.alloc(f32, count);
    errdefer gpa.free(profile);
    for (profile) |*slot| {
        const text = std.mem.trim(u8, lines.next() orelse return error.InvalidMixedClipProfile, " \t\r");
        slot.* = std.fmt.parseFloat(f32, text) catch return error.InvalidMixedClipProfile;
        if (!std.math.isFinite(slot.*) or slot.* <= 0 or slot.* > 1) return error.InvalidMixedClipProfile;
    }
    while (lines.next()) |line| if (std.mem.trim(u8, line, " \t\r").len != 0) return error.InvalidMixedClipProfile;
    return .{ .values = profile, .preserve_last_mlp_layers = preserve_last_mlp_layers, .group_profile = group_profile };
}

fn validateExpertMode(file: bool, sha: bool, core: bool, ordinary_q4: bool, skip: u32, depth: usize, paging: bool, research: bool) !void {
    if (file != sha) return error.IncompleteExpertBinding;
    if (file and (!core or !ordinary_q4 or skip != 0 or depth != q.config.layers or paging or research))
        return error.ExpertRequiresOrdinaryFullGraph;
}

test "expert requires paired binding accepted core and full ordinary graph" {
    try validateExpertMode(false, false, false, false, 0, 24, false, false);
    try validateExpertMode(true, true, true, true, 0, 24, false, false);
    try std.testing.expectError(error.IncompleteExpertBinding, validateExpertMode(true, false, true, true, 0, 24, false, false));
    try std.testing.expectError(error.IncompleteExpertBinding, validateExpertMode(false, true, true, true, 0, 24, false, false));
    try std.testing.expectError(error.ExpertRequiresOrdinaryFullGraph, validateExpertMode(true, true, false, true, 0, 24, false, false));
    try std.testing.expectError(error.ExpertRequiresOrdinaryFullGraph, validateExpertMode(true, true, true, false, 0, 24, false, false));
    try std.testing.expectError(error.ExpertRequiresOrdinaryFullGraph, validateExpertMode(true, true, true, true, 2, 24, false, false));
    try std.testing.expectError(error.ExpertRequiresOrdinaryFullGraph, validateExpertMode(true, true, true, true, 0, 12, false, false));
    try std.testing.expectError(error.ExpertRequiresOrdinaryFullGraph, validateExpertMode(true, true, true, true, 0, 24, true, false));
    try std.testing.expectError(error.ExpertRequiresOrdinaryFullGraph, validateExpertMode(true, true, true, true, 0, 24, false, true));
}

fn validateDirectMode(path: bool, sha: bool, quantizer: bool, ordinary: bool, skip: u32, exit: usize, paging: bool, research: bool) !void {
    if (path != sha or path != quantizer) return error.IncompleteDirectQ4Binding;
    if (path and (!ordinary or skip != 0 or exit != 24 or paging or research)) return error.DirectQ4RequiresOrdinaryFullGraph;
}

test "direct q4 requires paired file approval quantizer and ordinary full graph" {
    try validateDirectMode(false, false, false, false, 0, 24, false, false);
    try validateDirectMode(true, true, true, true, 0, 24, false, false);
    try std.testing.expectError(error.IncompleteDirectQ4Binding, validateDirectMode(true, false, true, true, 0, 24, false, false));
    try std.testing.expectError(error.IncompleteDirectQ4Binding, validateDirectMode(true, true, false, true, 0, 24, false, false));
    try std.testing.expectError(error.IncompleteDirectQ4Binding, validateDirectMode(false, true, false, true, 0, 24, false, false));
    try std.testing.expectError(error.DirectQ4RequiresOrdinaryFullGraph, validateDirectMode(true, true, true, false, 0, 24, false, false));
    try std.testing.expectError(error.DirectQ4RequiresOrdinaryFullGraph, validateDirectMode(true, true, true, true, 1, 24, false, false));
    try std.testing.expectError(error.DirectQ4RequiresOrdinaryFullGraph, validateDirectMode(true, true, true, true, 0, 23, false, false));
    try std.testing.expectError(error.DirectQ4RequiresOrdinaryFullGraph, validateDirectMode(true, true, true, true, 0, 24, true, false));
    try std.testing.expectError(error.DirectQ4RequiresOrdinaryFullGraph, validateDirectMode(true, true, true, true, 0, 24, false, true));
}

fn run(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const supplied_args = try init.minimal.args.toSlice(a);
    if (supplied_args.len >= 2 and std.mem.eql(u8, supplied_args[1], "--prepare-direct-q4")) {
        if (supplied_args.len != 7 or !std.mem.startsWith(u8, supplied_args[6], "--quantizer-sha256="))
            return error.ExpectedDirectQ4PreparationArguments;
        const manifest = try digest(supplied_args[3]);
        const quantizer = try digest(supplied_args[6]["--quantizer-sha256=".len..]);
        const cancel_fd = try std.fmt.parseInt(c_int, supplied_args[5], 10);
        if (cancel_fd < 3) return error.InvalidCancelFD;
        const start = now(init.io);
        const prepared = try q.Model.prepareDirectQ4(init.gpa, init.io, supplied_args[2], manifest, supplied_args[4], quantizer, cancel_fd, cancelled);
        const quantizer_sha = std.fmt.bytesToHex(quantizer, .lower);
        const manifest_sha = std.fmt.bytesToHex(manifest, .lower);
        try emit(init.io, init.gpa, .{ .event = "prepared", .format = "luban-direct-q4-g16-v1",
            .manifest_sha256 = @as([]const u8, &manifest_sha), .direct_q4_sha256 = @as([]const u8, &prepared.file_sha256),
            .quantizer_sha256 = @as([]const u8, &quantizer_sha), .direct_q4_weight_bytes = prepared.bytes, .file_bytes = prepared.file_bytes,
            .max_source_tensor_bytes = prepared.max_source_tensor_bytes, .prepare_workspace_bound_bytes = @as(usize, 128 << 20),
            .quantized_projections = prepared.quantized_projections, .prepare_ns = now(init.io) - start, .memory = try memory() });
        return;
    }
    var direct_q4_path: ?[]const u8 = null;
    var direct_q4_sha: ?[32]u8 = null;
    var quantizer_sha: ?[32]u8 = null;
    var filtered: std.ArrayList([]const u8) = .empty;
    var trace_directory: ?[]const u8 = null;
    var expert_file: ?[]const u8 = null;
    var expert_sha: ?[32]u8 = null;
    var intervention = false;
    var audit_directory: ?[]const u8 = null;
    var capture_directory: ?[]const u8 = null;
    var sequential_directory: ?[]const u8 = null;
    var skip_layer_mask: u32 = 0;
    var exit_after: usize = q.config.layers;
    var skip_supplied = false;
    var exit_supplied = false;
    var layer_paging_path: ?[]const u8 = null;
    const sequential_prefix = "--research-sequential=";
    const capture_prefix = "--research-capture-inputs=";
    const audit_prefix = "--research-audit-quantization=";
    const trace_prefix = "--research-trace=";
    for (supplied_args) |arg| {
        if (std.mem.startsWith(u8, arg, "--direct-q4=")) {
            if (direct_q4_path != null) return error.DuplicateDirectQ4;
            const path = arg["--direct-q4=".len..];
            if (!std.fs.path.isAbsolute(path)) return error.InvalidDirectQ4Path;
            direct_q4_path = path;
        } else if (std.mem.startsWith(u8, arg, "--direct-q4-sha256=")) {
            if (direct_q4_sha != null) return error.DuplicateDirectQ4;
            direct_q4_sha = try digest(arg["--direct-q4-sha256=".len..]);
        } else if (std.mem.startsWith(u8, arg, "--quantizer-sha256=")) {
            if (quantizer_sha != null) return error.DuplicateDirectQ4;
            quantizer_sha = try digest(arg["--quantizer-sha256=".len..]);
        } else if (std.mem.startsWith(u8, arg, "--expert-file=")) {
            if (expert_file != null) return error.DuplicateExpert;
            const path = arg["--expert-file=".len..];
            if (!std.fs.path.isAbsolute(path)) return error.InvalidExpertPath;
            expert_file = path;
        } else if (std.mem.startsWith(u8, arg, "--expert-sha=")) {
            if (expert_sha != null) return error.DuplicateExpert;
            expert_sha = try digest(arg["--expert-sha=".len..]);
        } else if (std.mem.startsWith(u8, arg, "--skip-layer-mask=")) {
            if (skip_supplied) return error.DuplicateLayerPolicy;
            skip_supplied = true;
            skip_layer_mask = try std.fmt.parseInt(u32, arg["--skip-layer-mask=".len..], 10);
            if (skip_layer_mask >= (1 << q.config.layers) or skip_layer_mask & 1 != 0) return error.InvalidLayerPolicy;
        } else if (std.mem.startsWith(u8, arg, "--exit-after=")) {
            if (exit_supplied) return error.DuplicateLayerPolicy;
            exit_supplied = true;
            exit_after = try std.fmt.parseInt(usize, arg["--exit-after=".len..], 10);
            if (exit_after == 0 or exit_after > q.config.layers) return error.InvalidLayerPolicy;
        } else if (std.mem.startsWith(u8, arg, "--layer-paging=")) {
            if (layer_paging_path != null) return error.DuplicateLayerPaging;
            const path = arg["--layer-paging=".len..];
            if (!std.fs.path.isAbsolute(path)) return error.InvalidLayerPagingPath;
            layer_paging_path = path;
        } else if (std.mem.eql(u8, arg, "--research-intervention")) {
            if (intervention) return error.InvalidResearchIntervention;
            intervention = true;
        } else if (std.mem.startsWith(u8, arg, sequential_prefix)) {
            if (sequential_directory != null) return error.InvalidSequentialDirectory;
            const directory = arg[sequential_prefix.len..];
            if (!std.fs.path.isAbsolute(directory)) return error.InvalidSequentialDirectory;
            var opened = try std.Io.Dir.cwd().openDir(init.io, directory, .{ .follow_symlinks = false });
            opened.close(init.io);
            sequential_directory = directory;
        } else if (std.mem.startsWith(u8, arg, capture_prefix)) {
            if (capture_directory != null) return error.InvalidCapturePath;
            const directory = arg[capture_prefix.len..];
            if (!std.fs.path.isAbsolute(directory)) return error.InvalidCapturePath;
            var opened = try std.Io.Dir.cwd().openDir(init.io, directory, .{ .follow_symlinks = false });
            opened.close(init.io);
            capture_directory = directory;
        } else if (std.mem.startsWith(u8, arg, audit_prefix)) {
            if (audit_directory != null) return error.InvalidAuditDirectory;
            const directory = arg[audit_prefix.len..];
            if (!std.fs.path.isAbsolute(directory)) return error.InvalidAuditDirectory;
            var opened = try std.Io.Dir.cwd().openDir(init.io, directory, .{ .follow_symlinks = false });
            opened.close(init.io);
            audit_directory = directory;
        } else if (std.mem.startsWith(u8, arg, trace_prefix)) {
            if (trace_directory != null) return error.InvalidTraceDirectory;
            const directory = arg[trace_prefix.len..];
            if (directory.len == 0 or !std.fs.path.isAbsolute(directory)) return error.InvalidTraceDirectory;
            var opened = try std.Io.Dir.cwd().openDir(init.io, directory, .{ .follow_symlinks = false });
            opened.close(init.io);
            trace_directory = directory;
        } else try filtered.append(a, arg);
    }
    if (audit_directory != null and !intervention) return error.ResearchInterventionNotEnabled;
    if (intervention and trace_directory == null) return error.ResearchTraceNotEnabled;
    if ((skip_layer_mask != 0 or exit_after != q.config.layers) and
        (intervention or trace_directory != null or audit_directory != null or capture_directory != null or sequential_directory != null))
        return error.LayerPolicyRequiresOrdinaryInference;
    // A full-depth trace may compare paged and resident bytes. Mutation and
    // layer-at-a-time research modes still require resident original weights.
    if (layer_paging_path != null and (intervention or audit_directory != null or capture_directory != null or sequential_directory != null))
        return error.LayerPagingRequiresImmutableInference;
    const all_args = filtered.items;
    const last = if (all_args.len > 1) all_args[all_args.len - 1] else "";
    const mlp_only = std.mem.eql(u8, last, "--mixed-q4-mlp");
    const gate_up_only = std.mem.eql(u8, last, "--mixed-q4-gateup");
    const down_only = std.mem.eql(u8, last, "--mixed-q4-down");
    const q4_early = std.mem.eql(u8, last, "--mixed-q4-early");
    const q4_early8 = std.mem.eql(u8, last, "--mixed-q4-early8");
    const q4_early12 = std.mem.eql(u8, last, "--mixed-q4-early12");
    const q4_first1 = std.mem.eql(u8, last, "--mixed-q4-first1");
    const q4_first2 = std.mem.eql(u8, last, "--mixed-q4-first2");
    const q4_first4 = std.mem.eql(u8, last, "--mixed-q4-first4");
    const q4_first6 = std.mem.eql(u8, last, "--mixed-q4-first6");
    const packed_prefix = "--mixed-q4-g16-packed=";
    const q4_packed = std.mem.startsWith(u8, last, packed_prefix);
    if (q4_packed and !std.fs.path.isAbsolute(last[packed_prefix.len..])) return error.InvalidPackedPath;
    const q4_g16 = q4_packed or std.mem.eql(u8, last, "--mixed-q4-g16");
    const residual_prefix = "--mixed-q4-g32-residual=";
    const q4_residual = std.mem.startsWith(u8, last, residual_prefix);
    const q4_g32 = q4_residual or std.mem.eql(u8, last, "--mixed-q4-g32");
    const q4_g32_mse = std.mem.eql(u8, last, "--mixed-q4-g32-mse");
    const q4_g32_clip97 = std.mem.eql(u8, last, "--mixed-q4-g32-clip97");
    const q4_g32_middle = std.mem.eql(u8, last, "--mixed-q4-g32-middle");
    const q4_g32_early12 = std.mem.eql(u8, last, "--mixed-q4-g32-early12");
    const q4_g32_early4 = std.mem.eql(u8, last, "--mixed-q4-g32-early4");
    const q4_g32_skip0 = std.mem.eql(u8, last, "--mixed-q4-g32-skip0");
    const q4_g32_skip0_early8 = std.mem.eql(u8, last, "--mixed-q4-g32-skip0-early8");
    const q4_g32_gateup = std.mem.eql(u8, last, "--mixed-q4-g32-gateup");
    const q4_g32_down = std.mem.eql(u8, last, "--mixed-q4-g32-down");
    const q8_mlp = std.mem.eql(u8, last, "--mixed-q8-mlp");
    const q8_gateup = std.mem.eql(u8, last, "--mixed-q8-gateup");
    const q8 = q8_mlp or q8_gateup or std.mem.eql(u8, last, "--mixed-q8");
    const calibrated_prefix = "--mixed-q4-calibrated=";
    const g32_calibrated_prefix = "--mixed-q4-g32-calibrated=";
    const g32_group_calibrated_prefix = "--mixed-q4-g32-group-calibrated=";
    const g32_group_early4_calibrated_prefix = "--mixed-q4-g32-group-early4-calibrated=";
    const g32_group_early8_calibrated_prefix = "--mixed-q4-g32-group-early8-calibrated=";
    const q4_calibrated = std.mem.startsWith(u8, last, calibrated_prefix);
    const q4_g32_calibrated = std.mem.startsWith(u8, last, g32_calibrated_prefix);
    const q4_g32_group_calibrated = std.mem.startsWith(u8, last, g32_group_calibrated_prefix);
    const q4_g32_group_early4_calibrated = std.mem.startsWith(u8, last, g32_group_early4_calibrated_prefix);
    const q4_g32_group_early8_calibrated = std.mem.startsWith(u8, last, g32_group_early8_calibrated_prefix);
    const mixed_quant = q8 or mlp_only or gate_up_only or down_only or q4_early or q4_early8 or q4_early12 or q4_first1 or q4_first2 or q4_first4 or q4_first6 or q4_g16 or q4_g32 or q4_g32_mse or q4_g32_clip97 or q4_g32_middle or q4_g32_early12 or q4_g32_early4 or q4_g32_skip0 or q4_g32_skip0_early8 or q4_g32_gateup or q4_g32_down or q4_calibrated or q4_g32_calibrated or q4_g32_group_calibrated or q4_g32_group_early4_calibrated or q4_g32_group_early8_calibrated or std.mem.eql(u8, last, "--mixed-q4");
    if (audit_directory != null and ((!q4_g32 and !q4_g16) or q4_residual)) return error.AuditRequiresPlainQ4Group16Or32;
    const args = if (mixed_quant) all_args[0 .. all_args.len - 1] else all_args;
    if (args.len == 2 and std.mem.eql(u8, args[1], "--version")) {
        try emit(init.io, init.gpa, .{ .name = "luban-worker", .protocol = "luban-worker-jsonl-v1", .format = "original-hf-bf16", .backend = "metal" });
        return;
    }
    if (args.len != 4 and args.len != 7) return error.ExpectedModelManifestSHACancelFDAndOptionalCoreFileKeySHA;
    try validateExpertMode(expert_file != null, expert_sha != null, args.len == 7,
        q4_g16 and !q4_packed, skip_layer_mask, exit_after, layer_paging_path != null,
        intervention or trace_directory != null or audit_directory != null or capture_directory != null or sequential_directory != null);
    if (args.len == 7 and (exit_after != q.config.layers or skip_layer_mask & (1 << (q.config.layers - 1)) != 0))
        return error.LayerPolicyWouldSkipAcceptedCore;
    if (capture_directory != null and (args.len != 4 or (mixed_quant and !q4_packed) or intervention or trace_directory != null or audit_directory != null))
        return error.InvalidCaptureSource;
    if (sequential_directory != null and (args.len != 4 or mixed_quant or intervention or trace_directory != null or audit_directory != null or capture_directory != null))
        return error.SequentialRequiresOriginal;
    try validateDirectMode(direct_q4_path != null, direct_q4_sha != null, quantizer_sha != null,
        q4_g16 and !q4_packed, skip_layer_mask, exit_after, layer_paging_path != null,
        intervention or audit_directory != null or capture_directory != null or sequential_directory != null);
    const manifest = try digest(args[2]);
    const fd = try std.fmt.parseInt(c_int, args[3], 10);
    if (fd < 3) return error.InvalidCancelFD;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const load_start = now(init.io);
    const model = if (direct_q4_path) |path|
        try q.Model.loadDirectQ4(init.gpa, init.io, args[1], manifest, path, direct_q4_sha.?, quantizer_sha.?, fd, cancelled)
    else try q.Model.loadNative(init.gpa, init.io, args[1], manifest, null);
    defer model.deinit();
    model.execution_policy = .{ .skip_mask = skip_layer_mask, .exit_after = exit_after };
    _ = try model.execution_policy.mask();
    var core_sha: ?[64]u8 = null;
    var core_tensor_sha: ?[64]u8 = null;
    if (args.len == 7) {
        const binding = try model.applyProductCore(init.io, args[4], args[5], try digest(args[6]));
        core_sha = binding.file_sha256;
        core_tensor_sha = binding.tensor_sha256;
    }
    var clip_profile: ?ClipProfile = null;
    defer if (clip_profile) |profile| init.gpa.free(profile.values);
    if (q4_calibrated) clip_profile = try loadClipProfile(init.io, init.gpa, last[calibrated_prefix.len..]);
    if (q4_g32_calibrated) clip_profile = try loadClipProfile(init.io, init.gpa, last[g32_calibrated_prefix.len..]);
    if (q4_g32_group_calibrated) clip_profile = try loadClipProfile(init.io, init.gpa, last[g32_group_calibrated_prefix.len..]);
    if (q4_g32_group_early4_calibrated) clip_profile = try loadClipProfile(init.io, init.gpa, last[g32_group_early4_calibrated_prefix.len..]);
    if (q4_g32_group_early8_calibrated) clip_profile = try loadClipProfile(init.io, init.gpa, last[g32_group_early8_calibrated_prefix.len..]);
    const binding = model.base_binding orelse return error.UnboundNativeManifest;
    const packed_core = core_sha orelse @as([64]u8, @splat('0'));
    var packed_reader: ?q.Model.packed_q4.Reader = if (q4_packed)
        try q.Model.packed_q4.Reader.open(init.io, last[packed_prefix.len..], &binding.manifest_sha256, &packed_core)
    else
        null;
    defer if (packed_reader) |*reader| reader.close();
    model.retain_original_for_research = intervention;
    const quantized_projections = if (direct_q4_path != null) @as(usize, 71) else if (mixed_quant) try model.enableMixed(q8_mlp or q8_gateup or mlp_only or q4_g16 or q4_g32 or q4_g32_mse or q4_g32_clip97 or q4_g32_middle or q4_g32_early12 or q4_g32_early4 or q4_g32_skip0 or q4_g32_skip0_early8 or q4_g32_gateup or q4_g32_down or q4_g32_calibrated or q4_g32_group_calibrated or q4_g32_group_early4_calibrated or q4_g32_group_early8_calibrated or gate_up_only or down_only or q4_early or q4_early8 or q4_early12 or q4_first1 or q4_first2 or q4_first4 or q4_first6 or q4_calibrated, if (q8) 8 else 4, if (q4_g32_middle) 8 else if (q4_g32_skip0_early8) 16 else if (q4_g32_early12) 12 else if (q4_g32_early4) 4 else if (q4_first6) 18 else if (q4_first4) 20 else if (q4_first1) 23 else if (q4_first2) 22 else if (q4_early12) 12 else if (q4_early8) 8 else if (q4_early) 4 else if (clip_profile) |profile| profile.preserve_last_mlp_layers else 0, if (clip_profile) |profile| @as(?[]const f32, profile.values) else null, gate_up_only or q4_g32_gateup or q8_gateup, down_only or q4_g32_down, if (q4_g16) 16 else if (q4_g32 or q4_g32_mse or q4_g32_clip97 or q4_g32_middle or q4_g32_early12 or q4_g32_early4 or q4_g32_skip0 or q4_g32_skip0_early8 or q4_g32_gateup or q4_g32_down or q4_g32_calibrated or q4_g32_group_calibrated or q4_g32_group_early4_calibrated or q4_g32_group_early8_calibrated) 32 else 64, if (q4_g32_middle) 8 else if (q4_g32_skip0 or q4_g32_skip0_early8) 1 else 0, if (q4_g32_clip97) @as(?f32, 0.97) else null, q4_g32_mse, q4_g32_group_calibrated or q4_g32_group_early4_calibrated or q4_g32_group_early8_calibrated, if (packed_reader) |*reader| reader else null) else 0;
    if (expert_file) |path| try model.attachExpert(init.io, path, expert_sha.?);
    const residual_rank = if (q4_residual) try model.enableResidual(init.io, last[residual_prefix.len..], &binding.manifest_sha256, &(core_sha orelse return error.ResidualRequiresCore)) else model.residual_rank;
    if (capture_directory != null) try q.state.capture_inputs.validateScope(model.mixed, if (packed_reader) |reader| reader.header.version else 0, quantized_projections, residual_rank);
    if (layer_paging_path) |path| try model.enableLayerPaging(try a.dupeSentinel(u8, path, 0));
    const retained_original_bytes = if (model.original) |*original| original.checkpoint.residentBytes() else 0;
    const direct_sha_hex = if (direct_q4_sha) |value| std.fmt.bytesToHex(value, .lower) else @as([64]u8, @splat('0'));
    const quantizer_sha_hex = if (quantizer_sha) |value| std.fmt.bytesToHex(value, .lower) else @as([64]u8, @splat('0'));
    try emit(init.io, init.gpa, .{ .event = "loaded", .direct_q4_loaded = direct_q4_path != null, .direct_q4_sha256 = if (direct_q4_path != null) @as(?[]const u8, &direct_sha_hex) else null, .quantizer_sha256 = if (direct_q4_path != null) @as(?[]const u8, &quantizer_sha_hex) else null, .direct_q4_weight_bytes = if (direct_q4_path != null) model.checkpoint.shards.items[0].bytes else @as(usize, 0), .expert_loaded = model.expert != null, .expert_file_sha256 = if (model.expert) |expert| @as(?[]const u8, &expert.sha256) else null, .expert_width = if (model.expert) |expert| expert.width else @as(usize, 0), .expert_layer = if (model.expert != null) @as(?usize, 23) else null, .research_sequential = sequential_directory != null, .research_capture_inputs = capture_directory != null, .research_intervention = intervention, .retained_original_weight_bytes = retained_original_bytes, .name = "luban-worker", .protocol = "luban-worker-jsonl-v1", .load_ns = now(init.io) - load_start, .residual_rank = residual_rank, .packed_version = if (packed_reader) |reader| reader.header.version else @as(u8, 0), .active_residual_projections = if (packed_reader) |reader| reader.residual_projections else if (q4_residual) @as(usize, 71) else 0, .inference_precision = if (q4_packed) @as([]const u8, "mixed-q4-g16-packed") else if (q4_residual) @as([]const u8, "mixed-q4-g32-residual") else if (q8_mlp) @as([]const u8, "mixed-q8-mlp") else if (q8_gateup) @as([]const u8, "mixed-q8-gateup") else if (q8) @as([]const u8, "mixed-q8") else if (q4_g16) @as([]const u8, "mixed-q4-g16") else if (q4_g32_group_early8_calibrated) @as([]const u8, "mixed-q4-g32-group-early8-calibrated") else if (q4_g32_group_early4_calibrated) @as([]const u8, "mixed-q4-g32-group-early4-calibrated") else if (q4_g32_group_calibrated) @as([]const u8, "mixed-q4-g32-group-calibrated") else if (q4_g32_mse) @as([]const u8, "mixed-q4-g32-mse") else if (q4_g32_middle) @as([]const u8, "mixed-q4-g32-middle") else if (q4_g32_clip97) @as([]const u8, "mixed-q4-g32-clip97") else if (q4_g32_calibrated) @as([]const u8, "mixed-q4-g32-calibrated") else if (q4_g32_gateup) @as([]const u8, "mixed-q4-g32-gateup") else if (q4_g32_down) @as([]const u8, "mixed-q4-g32-down") else if (q4_g32_skip0_early8) @as([]const u8, "mixed-q4-g32-skip0-early8") else if (q4_g32_skip0) @as([]const u8, "mixed-q4-g32-skip0") else if (q4_g32_early12) @as([]const u8, "mixed-q4-g32-early12") else if (q4_g32_early4) @as([]const u8, "mixed-q4-g32-early4") else if (q4_g32) @as([]const u8, "mixed-q4-g32") else if (q4_calibrated) @as([]const u8, "mixed-q4-calibrated") else if (q4_first6) @as([]const u8, "mixed-q4-first6") else if (q4_first4) @as([]const u8, "mixed-q4-first4") else if (q4_first1) @as([]const u8, "mixed-q4-first1") else if (q4_first2) @as([]const u8, "mixed-q4-first2") else if (q4_early12) @as([]const u8, "mixed-q4-early12") else if (q4_early8) @as([]const u8, "mixed-q4-early8") else if (q4_early) @as([]const u8, "mixed-q4-early") else if (down_only) @as([]const u8, "mixed-q4-down") else if (gate_up_only) @as([]const u8, "mixed-q4-gateup") else if (mlp_only) @as([]const u8, "mixed-q4-mlp") else if (mixed_quant) @as([]const u8, "mixed-q4") else "bf16", .quantized_projections = quantized_projections, .resident_weight_bytes = model.residentWeightBytes() + retained_original_bytes, .execution_skip_mask = skip_layer_mask, .execution_exit_after = exit_after, .layer_paging_enabled = model.layer_pager != null, .layer_paging = pagingStats(model), .weight_format = if (q4_packed) (if (residual_rank != 0) @as([]const u8, "source-bound packed affine Q4/group16 MLP plus rank8 BF16 column correction; BF16 embedding/attention/delta/core; original FP32 state") else @as([]const u8, "source-bound packed affine Q4/group16 MLP; BF16 embedding/attention/delta/core; original FP32 state")) else if (q4_residual) @as([]const u8, "affine Q4/group32 MLP plus BF16 low-rank error correction; FP32 residual accumulation; BF16 output; bound core unchanged") else if (q8_mlp) @as([]const u8, "symmetric I8/group64 MLP; BF16 embedding/attention/delta/core; original FP32 state") else if (q8_gateup) @as([]const u8, "symmetric I8/group64 MLP gate/up; BF16 down/embedding/attention/delta/core; original FP32 state") else if (q8) @as([]const u8, "symmetric I8/group64 projections; BF16 scale/core/norm/conv/delta-a-b; original FP32 A_log/gated norm/state") else if (q4_g16) @as([]const u8, "affine Q4/group16 MLP; BF16 embedding/attention/delta/core; original FP32 state") else if (q4_g32_group_early8_calibrated) @as([]const u8, "per-input-group activation-calibrated affine Q4/group32 MLP with last 8 layers protected; BF16 embedding/attention/delta/core; original FP32 state") else if (q4_g32_group_early4_calibrated) @as([]const u8, "per-input-group activation-calibrated affine Q4/group32 MLP with last 4 layers protected; BF16 embedding/attention/delta/core; original FP32 state") else if (q4_g32_group_calibrated) @as([]const u8, "per-input-group activation-calibrated affine Q4/group32 MLP; BF16 embedding/attention/delta/core; original FP32 state") else if (q4_g32_mse) @as([]const u8, "MSE-optimized affine Q4/group32 MLP; BF16 embedding/attention/delta/core; original FP32 state") else if (q4_g32_middle) @as([]const u8, "affine Q4/group32 MLP layers 8-15; first/last 8 MLP layers and attention remain BF16") else if (q4_g32_clip97) @as([]const u8, "affine Q4/group32 MLP with 0.97 clipping; embedding/attention remain BF16") else if (q4_g32_calibrated) @as([]const u8, "activation-calibrated affine Q4/group32 MLP; embedding/attention remain BF16") else if (q4_g32_gateup) @as([]const u8, "affine Q4/group32 MLP gate/up projections; down and attention remain BF16") else if (q4_g32_down) @as([]const u8, "affine Q4/group32 MLP down projections; gate/up and attention remain BF16") else if (q4_g32_skip0_early8) @as([]const u8, "affine Q4/group32 MLP layers 1-7; layer 0, later MLP and attention remain BF16") else if (q4_g32_skip0) @as([]const u8, "affine Q4/group32 MLP with first layer preserved; embedding/attention remain BF16") else if (q4_g32_early12) @as([]const u8, "affine Q4/group32 first 12 MLP layers; last 12 MLP layers and attention remain BF16") else if (q4_g32_early4) @as([]const u8, "affine Q4/group32 first 20 MLP layers; last 4 MLP layers and attention remain BF16") else if (q4_g32) @as([]const u8, "affine Q4/group32 MLP; BF16 embedding/attention/delta/core; original FP32 state") else if (q4_calibrated) @as([]const u8, "calibrated affine Q4/group64 MLP; BF16 embedding/attention/delta/core; original FP32 state") else if (q4_first6) @as([]const u8, "affine Q4/group64 first 6 MLP layers; remaining MLP and attention remain BF16") else if (q4_first4) @as([]const u8, "affine Q4/group64 first 4 MLP layers; remaining MLP and attention remain BF16") else if (q4_first1) @as([]const u8, "affine Q4/group64 first MLP layer; remaining MLP and attention remain BF16") else if (q4_first2) @as([]const u8, "affine Q4/group64 first 2 MLP layers; remaining MLP and attention remain BF16") else if (q4_early12) @as([]const u8, "affine Q4/group64 first 12 MLP layers; last 12 MLP layers plus attention remain BF16") else if (q4_early8) @as([]const u8, "affine Q4/group64 first 16 MLP layers; last 8 MLP layers plus attention remain BF16") else if (q4_early) @as([]const u8, "affine Q4/group64 early MLP only; last 4 MLP layers plus attention remain BF16") else if (down_only) @as([]const u8, "affine Q4/group64 MLP down projections; gate/up and attention remain BF16") else if (gate_up_only) @as([]const u8, "affine Q4/group64 MLP gate/up projections; MLP down and attention remain BF16") else if (mlp_only) @as([]const u8, "affine Q4/group64 frozen MLP; original BF16 embedding/attention/delta/core; original FP32 state") else if (mixed_quant) @as([]const u8, "affine Q4/group64 projections; BF16 accepted core and norms; original FP32 A_log/gated norm/state") else "BF16 with original FP32 A_log and gated norm", .memory = try memory(), .core_loaded = core_sha != null, .core_file_sha256 = if (core_sha) |*sha| @as([]const u8, sha) else null, .core_tensor_sha256 = if (core_tensor_sha) |*sha| @as([]const u8, sha) else null, .base_identity = @as([]const u8, &binding.identity), .manifest_sha256 = @as([]const u8, &binding.manifest_sha256), .checkpoint_sha256 = @as([]const u8, &binding.checkpoint_sha256), .original_down_sha256 = @as([]const u8, &binding.original_down_sha256) });
    var sequential: ?seq.Session = null;
    defer if (sequential) |*session| session.deinit();
    var input: Input = .{};
    var resident: Resident = .{};
    defer resident.clear(init.gpa);
    while (true) {
        var arena = std.heap.ArenaAllocator.init(init.gpa);
        defer arena.deinit();
        const line = input.line(arena.allocator()) catch |err| {
            resident.clear(init.gpa);
            try emit(init.io, init.gpa, .{ .event = "error", .id = @as(?[]const u8, null), .@"error" = @errorName(err) });
            if (err == error.RequestLineTooLong) continue;
            return err;
        } orelse break;
        const parsed = std.json.parseFromSlice(Request, arena.allocator(), line, .{ .allocate = .alloc_always }) catch |err| {
            resident.clear(init.gpa);
            try emit(init.io, init.gpa, .{ .event = "error", .id = @as(?[]const u8, null), .@"error" = @errorName(err) });
            continue;
        };
        if (parsed.value.seq_op != null) {
            resident.clear(init.gpa);
            sequentialControl(init, model, fd, &sequential, sequential_directory, parsed.value, arena.allocator()) catch |err| {
                try emit(init.io, init.gpa, .{ .event = "error", .id = parsed.value.id, .@"error" = @errorName(err) });
            };
            continue;
        }
        if (sequential) |*session| {
            if (!session.allowsGenerate() or parsed.value.tail_only or parsed.value.cache_reuse or parsed.value.forced_tokens != null or parsed.value.capture_positions != null or parsed.value.audit_quantization or parsed.value.restore_layer != null or parsed.value.restore_layers != null) {
                try emit(init.io, init.gpa, .{ .event = "error", .id = parsed.value.id, .@"error" = "InvalidSequentialGeneration" });
                continue;
            }
        }
        dispatch(init, model, fd, &resident, parsed.value, trace_directory, intervention, audit_directory, capture_directory) catch |err| {
            try emit(init.io, init.gpa, .{ .event = "error", .id = parsed.value.id, .@"error" = @errorName(err) });
        };
    }
}

fn sequentialControl(init: std.process.Init, model: *q.Model, fd: c_int, session: *?seq.Session, directory: ?[]const u8, request: Request, a: std.mem.Allocator) !void {
    if (request.tail_only) return error.InvalidTailTraceMode;
    const root = directory orelse return error.SequentialNotEnabled;
    if (request.tokens.len != 0 or request.max_tokens != 0 or request.eos_ids.len != 0 or request.cache_reuse or request.temperature != 0 or request.seed != 0 or request.forced_tokens != null or request.capture_positions != null or request.restore_layer != null or request.restore_layers != null or request.restore_projection != null or request.audit_quantization or request.audit_layer != null or request.audit_projection != null or request.audit_input_path != null or request.probe_tokens != null) return error.InvalidSequentialRequest;
    const command = seq.Command{ .op = request.seq_op.?, .revision = request.seq_revision, .layer = request.seq_layer, .plan = request.seq_plan, .payload = request.seq_payload, .dry_run = request.seq_dry_run, .accept = request.seq_accept };
    var result: seq.Response = undefined;
    if (command.op == .begin) {
        if (session.* != null) return error.SequentialAlreadyStarted;
        session.* = try seq.Session.init(init.gpa, init.io, model, root, command, fd, cancelled);
        result = session.*.?.response(.begin, 0);
    } else {
        if (session.*) |*active| {
            result = try active.execute(a, command);
        } else return error.SequentialNotStarted;
    }
    try emit(init.io, init.gpa, .{ .event = "sequential", .id = request.id, .op = result.op, .revision = result.revision, .layer = result.layer, .phase = result.phase, .accepted_layer_mask = result.accepted_layer_mask, .records = result.records, .predictions = result.predictions, .verified_values = result.verified_values });
}

test "prefix cache keeps only forwarded tokens and requires exact extension" {
    const a = std.testing.allocator;
    var prefix: Prefix = .{};
    defer prefix.tokens.deinit(a);
    const request: Request = .{ .id = "a", .tokens = &.{ 1, 2 }, .max_tokens = 3, .eos_ids = &.{5}, .seed = 42, .cache_reuse = true };
    try prefix.remember(a, request, &.{ 3, 4, 5 }, 4);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4 }, prefix.tokens.items);
    var next = request;
    next.tokens = &.{ 1, 2, 3, 4, 5, 6 };
    try std.testing.expect(prefix.matches(next));
    next.tokens = prefix.tokens.items;
    try std.testing.expect(prefix.matches(next));
    next.tokens = &.{ 1, 2, 3 };
    try std.testing.expect(!prefix.matches(next));
    next.tokens = &.{ 1, 2, 9, 4, 5, 6 };
    try std.testing.expect(!prefix.matches(next));
    try std.testing.expectError(error.InvalidPrefixState, prefix.remember(a, request, &.{3}, 1));
    try std.testing.expectError(error.InvalidPrefixState, prefix.remember(a, request, &.{3}, 4));
}

test "prefix cache permits fresh sampling settings and is disabled by default" {
    const a = std.testing.allocator;
    var prefix: Prefix = .{};
    defer prefix.tokens.deinit(a);
    const request: Request = .{ .id = "a", .tokens = &.{ 1, 2 }, .max_tokens = 1, .eos_ids = &.{5}, .seed = 42 };
    try prefix.remember(a, request, &.{3}, 2);
    try std.testing.expect(!prefix.matches(request));
    var next = request;
    next.cache_reuse = true;
    try std.testing.expect(prefix.matches(next));
    next.temperature = 0.5;
    try std.testing.expect(prefix.matches(next));
    next.seed += 1;
    try std.testing.expect(prefix.matches(next));
    next.eos_ids = &.{ 5, 6 };
    try std.testing.expect(prefix.matches(next));
    next.eos_ids = &.{6};
    try std.testing.expect(prefix.matches(next));
}

test "research intervention and probes reject unsafe request combinations" {
    var r: Request = .{ .id = "test", .tokens = &.{1}, .max_tokens = 1, .eos_ids = &.{2} };
    try validateResearch(r, false);
    r.restore_layer = 0;
    try std.testing.expectError(error.InvalidRestoreRequest, validateResearch(r, true));
    r.restore_projection = .all;
    try std.testing.expectError(error.ResearchInterventionNotEnabled, validateResearch(r, false));
    try validateResearch(r, true);
    r.restore_layer = 24;
    try std.testing.expectError(error.InvalidRestoreRequest, validateResearch(r, true));
    r.restore_layer = 23;
    r.cache_reuse = true;
    try std.testing.expectError(error.InvalidRestoreRequest, validateResearch(r, true));
    r.cache_reuse = false;
    r.probe_tokens = &.{1};
    try std.testing.expectError(error.InvalidProbeTokens, validateResearch(r, true));
    r.forced_tokens = &.{};
    try validateResearch(r, true);
    r.probe_tokens = &.{};
    try std.testing.expectError(error.InvalidProbeTokens, validateResearch(r, true));
    r.probe_tokens = &.{q.config.vocab};
    try std.testing.expectError(error.InvalidProbeTokens, validateResearch(r, true));
    r.probe_tokens = &(@as([17]u32, @splat(1)));
    try std.testing.expectError(error.InvalidProbeTokens, validateResearch(r, true));
}

test "joint restore validates uniqueness and audit remains isolated" {
    var r: Request = .{ .id = "test", .tokens = &.{1}, .max_tokens = 1, .eos_ids = &.{2}, .restore_layers = &.{ 0, 23 }, .restore_projection = .all };
    try validateResearch(r, true);
    r.restore_layer = 1;
    try std.testing.expectError(error.InvalidRestoreRequest, validateResearch(r, true));
    r.restore_layer = null;
    r.restore_layers = &.{ 0, 0 };
    try std.testing.expectError(error.InvalidRestoreRequest, validateResearch(r, true));
    r.restore_layers = &.{};
    try std.testing.expectError(error.InvalidRestoreRequest, validateResearch(r, true));
    r.restore_layers = &.{24};
    try std.testing.expectError(error.InvalidRestoreRequest, validateResearch(r, true));
    var all: [24]usize = undefined;
    for (&all, 0..) |*layer, i| layer.* = i;
    r.restore_layers = &all;
    try validateResearch(r, true);
    r.audit_quantization = true;
    r.audit_layer = 0;
    r.audit_projection = .gate;
    r.audit_input_path = "/absolute/input.bf16";
    try std.testing.expectError(error.InvalidAuditRequest, validateResearch(r, true));
    r.restore_layers = null;
    r.restore_projection = null;
    try validateResearch(r, true);
    r.audit_input_path = "relative/input";
    try std.testing.expectError(error.InvalidAuditRequest, validateResearch(r, true));
}

test "sequential controls parse without ordinary generation fields" {
    const parsed = try std.json.parseFromSlice(Request, std.testing.allocator, "{\"id\":\"control\",\"seq_op\":\"decide\",\"seq_layer\":2,\"seq_revision\":2,\"seq_accept\":false}", .{});
    defer parsed.deinit();
    try std.testing.expect(parsed.value.tokens.len == 0 and parsed.value.max_tokens == 0 and parsed.value.eos_ids.len == 0);
    try std.testing.expectEqual(seq.Op.decide, parsed.value.seq_op.?);
    try std.testing.expectEqual(false, parsed.value.seq_accept.?);
}

test "tail trace rejects ordinary capture probe intervention audit and sequential modes" {
    const initial: Request = .{ .id = "tail", .tokens = &.{1}, .max_tokens = 1, .eos_ids = &.{9}, .forced_tokens = &.{}, .tail_only = true };
    try validateResearch(initial, false);
    try std.testing.expectError(error.InvalidTailTraceMode, validateResearch(initial, true));
    var r = initial;
    r.forced_tokens = null;
    try std.testing.expectError(error.InvalidTailTraceMode, validateResearch(r, false));
    r = initial;
    r.capture_positions = &.{0};
    try std.testing.expectError(error.InvalidTailTraceMode, validateResearch(r, false));
    r = initial;
    r.probe_tokens = &.{1};
    try std.testing.expectError(error.InvalidTailTraceMode, validateResearch(r, false));
    r = initial;
    r.restore_layer = 23;
    try std.testing.expectError(error.InvalidTailTraceMode, validateResearch(r, false));
    r = initial;
    r.audit_quantization = true;
    try std.testing.expectError(error.InvalidTailTraceMode, validateResearch(r, false));
    r = initial;
    r.seq_op = .begin;
    try std.testing.expectError(error.InvalidTailTraceMode, validateResearch(r, false));
}

test "trace retains old limit and tail permits exactly 255 forced non EOS tokens" {
    const forced: [256]u32 = @splat(1);
    var r: Request = .{ .id = "tail", .tokens = &.{1}, .max_tokens = 65, .eos_ids = &.{9}, .forced_tokens = forced[0..64] };
    try validateTrace(r);
    r.forced_tokens = forced[0..65];
    r.max_tokens = 66;
    try std.testing.expectError(error.InvalidTraceRequest, validateTrace(r));
    r.tail_only = true;
    try validateTrace(r);
    r.forced_tokens = forced[0..255];
    r.max_tokens = 256;
    try validateTrace(r);
    r.forced_tokens = &forced;
    r.max_tokens = 257;
    try std.testing.expectError(error.InvalidTraceRequest, validateTrace(r));
    r.forced_tokens = &.{9};
    r.max_tokens = 2;
    try std.testing.expectError(error.UnforwardedCaptureEOS, validateTrace(r));
    r.forced_tokens = &.{q.config.vocab};
    try std.testing.expectError(error.InvalidQwenToken, validateTrace(r));
    r.forced_tokens = &.{1};
    r.cache_reuse = true;
    try std.testing.expectError(error.InvalidTraceRequest, validateTrace(r));
    r.cache_reuse = false;
    r.temperature = 0.1;
    try std.testing.expectError(error.InvalidTraceRequest, validateTrace(r));
    r.temperature = 0;
    r.max_tokens = 1;
    try std.testing.expectError(error.InvalidTraceRequest, validateTrace(r));
}

test "tail trace copies only last layer h act and down in that order" {
    const trace = try std.testing.allocator.alloc(u16, q.state.trace_bytes / 2);
    defer std.testing.allocator.free(trace);
    for (trace, 0..) |*v, i| v.* = @intCast(i % 65536);
    var tail: [tail_values]u16 = undefined;
    try copyTail(trace, &tail);
    const last = trace[23 * 26624 ..];
    try std.testing.expectEqual(@as(usize, 10240), tail.len);
    try std.testing.expectEqualSlices(u16, last[0..2048], tail[0..2048]);
    try std.testing.expectEqualSlices(u16, last[18432..24576], tail[2048..8192]);
    try std.testing.expectEqualSlices(u16, last[24576..26624], tail[8192..10240]);
    try std.testing.expectError(error.InvalidTailTraceShape, copyTail(trace[0 .. trace.len - 1], &tail));
}
