// SPDX-License-Identifier: Apache-2.0
//! Luban native loader derived from TensorFold: original HF BF16, with explicit inference-only mixed precision.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const ckpt = @import("../../core/checkpoint_metal.zig");
const shards = @import("../../core/checkpoint.zig");
const weights = @import("weights.zig");
const kernels = @import("kernels.zig");
const direct = @import("direct_q4.zig");
const safetensors = @import("../../core/safetensors.zig");

const CoreOptions = struct {
    path: []const u8,
    provenance_path: []const u8,
    approved_provenance_sha256: [32]u8,
};

pub const BaseBinding = struct {
    identity: [64]u8,
    manifest_sha256: [64]u8,
    config_sha256: [64]u8,
    checkpoint_sha256: [64]u8,
    original_down_sha256: [64]u8 = undefined,
};

pub const CoreBinding = struct {
    provenance_sha256: [64]u8,
    file_sha256: [64]u8,
    tensor_sha256: [64]u8,
    closed_validation_sha256: [64]u8,
};

const down_name = "model.language_model.layers.23.mlp.down_proj.weight";

pub const Model = struct {
    pub const CoreOverride = CoreOptions;
    pub const audit = @import("audit.zig");
    pub const packed_q4 = @import("packed.zig");
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    config: cfg.Config,
    checkpoint: ckpt.Checkpoint,
    weights: weights.Weights,
    kernels: kernels.Kernels,
    base_binding: ?BaseBinding = null,
    core_binding: ?CoreBinding = null,
    mixed: bool = false,
    expert: ?@import("expert.zig").Expert = null,
    residual_rank: usize = 0,
    retain_original_for_research: bool = false,
    original: ?struct { checkpoint: ckpt.Checkpoint, weights: weights.Weights } = null,
    execution_policy: @import("execution.zig").Policy = .{},
    layer_pager: ?@import("layer_paging.zig").Pager = null,

    pub fn enableLayerPaging(m: *Model, path: [:0]const u8) !void {
        if (m.layer_pager != null or m.original != null or m.retain_original_for_research)
            return error.LayerPagingRequiresInference;
        var globals = ckpt.Checkpoint.init(m.gpa);
        errdefer globals.deinit();
        var next = m.weights;
        const pager = try @import("layer_paging.zig").Pager.init(m.device, path, &next, &globals);
        m.checkpoint.deinit();
        m.checkpoint = globals;
        m.weights = next;
        m.layer_pager = pager;
    }

    pub fn acquireLayer(m: *Model, index: usize) !weights.Layer {
        if (index >= cfg.layers) return error.InvalidPagedLayer;
        if (m.layer_pager) |*pager| try pager.acquire(index);
        return m.weights.blocks[index];
    }

    pub fn residentWeightBytes(m: *const Model) usize {
        return m.checkpoint.residentBytes() + (if (m.layer_pager) |pager| pager.slot_bytes else @as(usize, 0)) +
            (if (m.expert) |expert| expert.bytes else @as(usize, 0));
    }

    /// Only the inference worker opts in, after accepted-core installation.
    /// Keep BF16 graph semantics for norms, activations and FP32 recurrence.
    pub fn enableMixed(m: *Model, mlp_only: bool, bits: u8, preserve_last_mlp_layers: usize, clip_profile: ?[]const f32, gate_up_only: bool, down_only: bool, group_size: usize, preserve_first_mlp_layers: usize, group_clip_ratio: ?f32, mse_quant: bool, group_clip_profile: bool, packed_reader: ?*packed_q4.Reader) !usize {
        if (m.config.format != .bf16 or m.base_binding == null or m.mixed or m.layer_pager != null)
            return error.MixedRequiresBoundBF16;
        var next = ckpt.Checkpoint.init(m.gpa);
        errdefer next.deinit();
        const converted = try @import("mixed.zig").prepare(&next, m.device, m.weights, mlp_only, bits, preserve_last_mlp_layers, clip_profile, gate_up_only, down_only, group_size, preserve_first_mlp_layers, group_clip_ratio, mse_quant, group_clip_profile, packed_reader);
        if (m.retain_original_for_research) {
            m.original = .{ .checkpoint = m.checkpoint, .weights = m.weights };
        } else m.checkpoint.deinit();
        m.checkpoint = next;
        m.weights = converted.weights;
        m.mixed = true;
        m.residual_rank = if (packed_reader) |reader| reader.header.residual_rank else 0;
        return converted.quantized;
    }

    pub fn enableResidual(m: *Model, io: std.Io, path: []const u8, manifest: []const u8, core: []const u8) !usize {
        if (!m.mixed or m.layer_pager != null) return error.ResidualRequiresQ4Group32;
        m.residual_rank = try @import("residual.zig").attach(&m.checkpoint, m.device, &m.weights, io, path, manifest, core);
        return m.residual_rank;
    }

    pub fn attachExpert(m: *Model, io: std.Io, path: []const u8, approved: [32]u8) !void {
        if (m.expert != null or !m.mixed or m.layer_pager != null or m.execution_policy.exit_after != cfg.layers or m.execution_policy.skip_mask != 0)
            return error.ExpertRequiresOrdinaryFullGraph;
        m.expert = try @import("expert.zig").Expert.load(m.gpa, io, m.device, path, approved);
    }

    pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !*Model {
        return loadBound(gpa, io, dir, null, null);
    }

    /// The caller approves the actual official manifest SHA before calling.
    /// This loader then independently hashes the config and complete checkpoint.
    pub fn loadNative(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, approved_manifest_sha256: [32]u8, core: ?CoreOverride) !*Model {
        const approved = std.fmt.bytesToHex(approved_manifest_sha256, .lower);
        return loadBound(gpa, io, dir, &approved, core);
    }

    pub fn prepareDirectQ4(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, approved_manifest: [32]u8, output_path: []const u8, quantizer: [32]u8, cancel_fd: c_int, cancel: *const fn (c_int) anyerror!bool) !direct.Prepared {
        const approved = std.fmt.bytesToHex(approved_manifest, .lower);
        const official = try verifyOfficial(gpa, io, dir, &approved, true, cancel_fd, cancel);
        defer gpa.free(official.path);
        const config = try cfg.Config.read(gpa, io, dir);
        if (config.format != .bf16) return error.NativeBF16Required;
        var source = try direct.Source.open(gpa, io, official.path);
        defer source.close();
        const entry = source.entries.get(down_name) orelse return error.MissingTensor;
        if (entry.dtype != .bf16 or entry.rank != 2 or entry.shape[0] != cfg.hidden or entry.shape[1] != cfg.intermediate)
            return error.UnexpectedNativeCoreTensor;
        const original = blk: {
            const raw = try gpa.alloc(u8, entry.end - entry.begin);
            defer gpa.free(raw);
            try source.read(raw, source.start + entry.begin);
            break :blk shaBytes(raw);
        };
        const quantizer_sha = std.fmt.bytesToHex(quantizer, .lower);
        return direct.prepare(gpa, io, official.path, output_path, .{ .format = direct.format,
            .quantizer_sha256 = &quantizer_sha, .base_identity = &official.binding.identity, .manifest_sha256 = &approved,
            .config_sha256 = &official.binding.config_sha256, .checkpoint_sha256 = &official.binding.checkpoint_sha256,
            .original_down_sha256 = &original }, cancel_fd, cancel);
    }

    /// No BF16 shard is opened here. The caller approves this complete derived
    /// checkpoint; its manifest/config binding and every payload byte are verified.
    pub fn loadDirectQ4(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, approved_manifest: [32]u8, path: []const u8, approved_file: [32]u8, quantizer: [32]u8, cancel_fd: c_int, cancel: *const fn (c_int) anyerror!bool) !*Model {
        const approved = std.fmt.bytesToHex(approved_manifest, .lower);
        const official = try verifyOfficial(gpa, io, dir, &approved, false, cancel_fd, cancel);
        defer gpa.free(official.path);
        const config = try cfg.Config.read(gpa, io, dir);
        if (config.format != .bf16) return error.NativeBF16Required;
        var source = try direct.Source.open(gpa, io, path);
        defer source.close();
        const meta = try source.metadata();
        const quantizer_sha = std.fmt.bytesToHex(quantizer, .lower);
        try direct.validateMetadata(meta, .{ .format = direct.format, .quantizer_sha256 = &quantizer_sha,
            .base_identity = &official.binding.identity, .manifest_sha256 = &approved,
            .config_sha256 = &official.binding.config_sha256, .checkpoint_sha256 = &official.binding.checkpoint_sha256,
            .original_down_sha256 = "" });
        try direct.validateLayout(source.entries, source.size - source.start);
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.* = .{ .gpa = gpa, .device = undefined, .queue = undefined, .config = config,
            .checkpoint = ckpt.Checkpoint.init(gpa), .weights = undefined, .kernels = undefined,
            .base_binding = official.binding, .mixed = true };
        m.device = try mtl.Device.init(); errdefer m.device.deinit();
        m.queue = try m.device.queue(); errdefer m.queue.deinit();
        errdefer m.checkpoint.deinit();
        const bytes = source.size - source.start;
        const buffer = try m.device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        m.checkpoint.own(buffer, bytes) catch |err| { buffer.deinit(); return err; };
        try source.readVerified(buffer.contents()[0..bytes], approved_file, cancel_fd, cancel);
        try m.checkpoint.index(buffer, source.raw, bytes, "");
        m.weights = try weights.loadDirectQ4(&m.checkpoint);
        const original = shaBytes(m.weights.blocks[cfg.layers - 1].down.weight.host(u8));
        if (!std.mem.eql(u8, &original, meta.original_down_sha256)) return error.DirectQ4BindingMismatch;
        m.base_binding.?.original_down_sha256 = original;
        for (m.weights.blocks, 0..) |block, i| {
            for ([_]weights.Linear{ block.gate, block.up, block.down }, 0..) |linear, j| {
                if (i == cfg.layers - 1 and j == 2) continue;
                try packed_q4.validateMetadata(linear.scales.host(u8), linear.biases.host(u8));
            }
        }
        if (try cancel(cancel_fd)) return error.DirectQ4Cancelled;
        m.kernels = try kernels.load(m.device);
        return m;
    }

    /// Product adoption/identity is checked by the parent. Independently verify
    /// the exact accepted file and tensor here, without research provenance.
    pub fn applyProductCore(m: *Model, io: std.Io, path: []const u8, key: []const u8, approved_file_sha256: [32]u8) !struct { file_sha256: [64]u8, tensor_sha256: [64]u8 } {
        if (m.config.format != .bf16 or m.base_binding == null or m.layer_pager != null) return error.NativeBF16Required;
        if (!std.mem.eql(u8, key, down_name) and !std.mem.eql(u8, key, "model.language_model.layers.23.mlp.base.down_proj.weight"))
            return error.UnexpectedNativeCoreTensor;
        const raw = try std.Io.Dir.cwd().readFileAlloc(io, path, m.gpa, .limited(32 << 20));
        defer m.gpa.free(raw);
        const actual = shaBytes(raw);
        const approved = std.fmt.bytesToHex(approved_file_sha256, .lower);
        if (!std.mem.eql(u8, &actual, &approved)) return error.NativeCoreFileSHAMismatch;
        const bytes = try coreTensorBytes(m.gpa, raw, key);
        try installCoreTensor(m, bytes);
        return .{ .file_sha256 = actual, .tensor_sha256 = shaBytes(bytes) };
    }

    fn loadBound(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, approved_manifest_sha256: ?[]const u8, core: ?CoreOverride) !*Model {
        const config = try cfg.Config.read(gpa, io, dir);
        if (config.format == .affine4 and (approved_manifest_sha256 != null or core != null)) return error.NativeBF16Required;
        var official: ?Official = null;
        defer if (official) |o| gpa.free(o.path);
        if (config.format == .bf16) official = try verifyOfficial(gpa, io, dir, approved_manifest_sha256 orelse return error.UnboundNativeManifest, true, 0, null);
        const m = try gpa.create(Model);
        errdefer gpa.destroy(m);
        m.gpa = gpa;
        m.config = config;
        m.base_binding = null;
        m.core_binding = null;
        m.mixed = false;
        m.expert = null;
        m.retain_original_for_research = false;
        m.residual_rank = 0;
        m.original = null;
        m.execution_policy = .{};
        m.layer_pager = null;
        m.device = try mtl.Device.init();
        errdefer m.device.deinit();
        m.queue = try m.device.queue();
        errdefer m.queue.deinit();
        m.checkpoint = ckpt.Checkpoint.init(gpa);
        errdefer m.checkpoint.deinit();
        if (official) |o| {
            // Only the language tensors enter shared Metal buffers. No vision.
            try m.checkpoint.addFileSelected(m.device, o.path, "", "model.language_model.");
            m.base_binding = o.binding;
        } else {
            const files = try shards.shardFiles(gpa, io, dir);
            defer shards.freeShardFiles(gpa, files);
            for (files) |path| try m.checkpoint.addFileSelected(m.device, path, "", "language_model.");
        }
        m.weights = try weights.loadFormat(&m.checkpoint, config.format);
        if (m.base_binding) |*binding| {
            const down = m.weights.blocks[cfg.layers - 1].down.weight;
            binding.original_down_sha256 = shaBytes(down.host(u8));
        }
        if (core) |override| m.core_binding = try applyCore(m, io, override);
        m.kernels = try kernels.load(m.device);
        return m;
    }

    pub fn deinit(m: *Model) void {
        if (m.expert) |*expert| expert.deinit();
        m.kernels.deinit();
        if (m.layer_pager) |*pager| pager.deinit();
        m.checkpoint.deinit();
        if (m.original) |*original| original.checkpoint.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.gpa.destroy(m);
    }
};

const Official = struct { path: [:0]u8, binding: BaseBinding };

fn shaBytes(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn validSha(value: []const u8) bool {
    if (value.len != 64) return false;
    for (value) |ch| if (!((ch >= '0' and ch <= '9') or (ch >= 'a' and ch <= 'f'))) return false;
    return true;
}

fn copySha(value: []const u8) ![64]u8 {
    if (!validSha(value)) return error.BadNativeSHA256;
    var result: [64]u8 = undefined;
    @memcpy(&result, value);
    return result;
}

fn obj(value: std.json.Value) !std.json.ObjectMap {
    return if (value == .object) value.object else error.BadNativeProvenance;
}

fn text(o: std.json.ObjectMap, key: []const u8) ![]const u8 {
    const value = o.get(key) orelse return error.BadNativeProvenance;
    return if (value == .string) value.string else error.BadNativeProvenance;
}

fn assertText(o: std.json.ObjectMap, key: []const u8, expected: []const u8) !void {
    if (!std.mem.eql(u8, try text(o, key), expected)) return error.NativeProvenanceMismatch;
}

fn assertBool(o: std.json.ObjectMap, key: []const u8, expected: bool) !void {
    const value = o.get(key) orelse return error.BadNativeProvenance;
    if (value != .bool or value.bool != expected) return error.NativeProvenanceMismatch;
}

fn size(o: std.json.ObjectMap, key: []const u8) !usize {
    const value = o.get(key) orelse return error.BadNativeProvenance;
    if (value != .integer or value.integer < 0) return error.BadNativeProvenance;
    return std.math.cast(usize, value.integer) orelse error.BadNativeProvenance;
}

fn basename(name: []const u8) bool {
    if (name.len == 0 or name[0] == '.') return false;
    for (name) |ch| if (!(std.ascii.isAlphanumeric(ch) or ch == '.' or ch == '-' or ch == '_')) return false;
    return true;
}

/// Stream the original file, rather than trusting a receipt's claimed digest.
fn verifyFile(path: [:0]const u8, expected_sha: []const u8, expected_size: usize, cancel_fd: c_int, cancel: ?*const fn (c_int) anyerror!bool) !void {
    if (!validSha(expected_sha)) return error.BadNativeSHA256;
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true });
    if (fd < 0) return error.OpenFailed;
    defer _ = std.c.close(fd);
    const end = std.c.lseek(fd, 0, std.c.SEEK.END);
    if (end < 0 or @as(u64, @intCast(end)) != expected_size) return error.NativeFileSizeMismatch;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var chunk: [1 << 20]u8 = undefined;
    var offset: usize = 0;
    while (offset < expected_size) {
        if (cancel) |check| if (try check(cancel_fd)) return error.DirectQ4Cancelled;
        const wanted = @min(chunk.len, expected_size - offset);
        const n = std.c.pread(fd, &chunk, wanted, @intCast(offset));
        if (n <= 0) return error.ShortRead;
        const count: usize = @intCast(n);
        hash.update(chunk[0..count]);
        offset += count;
    }
    var digest: [32]u8 = undefined;
    hash.final(&digest);
    const actual = std.fmt.bytesToHex(digest, .lower);
    if (!std.mem.eql(u8, &actual, expected_sha)) return error.NativeFileSHAMismatch;
    if (std.c.lseek(fd, 0, std.c.SEEK.END) != end) return error.NativeFileSizeMismatch;
}

fn verifyOfficial(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, approved_sha: []const u8, verify_checkpoint: bool, cancel_fd: c_int, cancel: ?*const fn (c_int) anyerror!bool) !Official {
    const approved = try copySha(approved_sha);
    const manifest_path = try std.fs.path.join(gpa, &.{ dir, "upstream-file-manifest.json" });
    defer gpa.free(manifest_path);
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, manifest_path, gpa, .limited(1 << 22));
    defer gpa.free(raw);
    const manifest_sha = shaBytes(raw);
    if (!std.mem.eql(u8, &manifest_sha, &approved)) return error.NativeManifestSHAMismatch;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, raw, .{});
    defer parsed.deinit();
    const root = try obj(parsed.value);
    try assertText(root, "model", "Qwen/Qwen3.5-2B");
    const revision = try text(root, "revision");
    if (revision.len != 40) return error.BadNativeProvenance;
    const data = try obj(root.get("Data") orelse return error.BadNativeProvenance);
    const files = data.get("Files") orelse return error.BadNativeProvenance;
    if (files != .array) return error.BadNativeProvenance;
    var config: ?std.json.ObjectMap = null;
    var checkpoint: ?std.json.ObjectMap = null;
    for (files.array.items) |entry| {
        const record = try obj(entry);
        const name = try text(record, "Path");
        if (!basename(name)) return error.UnsafeNativeFileName;
        if (std.mem.eql(u8, name, "config.json")) {
            if (config != null) return error.DuplicateNativeFile;
            config = record;
        } else if (std.mem.endsWith(u8, name, ".safetensors")) {
            // The fixed HF 2B checkpoint is one shard; do not silently ignore more.
            if (checkpoint != null or !std.mem.startsWith(u8, name, "model")) return error.UnsupportedNativeShards;
            checkpoint = record;
        }
    }
    const conf = config orelse return error.BadNativeProvenance;
    const shard = checkpoint orelse return error.BadNativeProvenance;
    const config_sha = try copySha(try text(conf, "Sha256"));
    const checkpoint_sha = try copySha(try text(shard, "Sha256"));
    const config_path = try std.fmt.allocPrintSentinel(gpa, "{s}/config.json", .{dir}, 0);
    defer gpa.free(config_path);
    try verifyFile(config_path, &config_sha, try size(conf, "Size"), cancel_fd, cancel);
    const shard_name = try text(shard, "Path");
    const checkpoint_path = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ dir, shard_name }, 0);
    errdefer gpa.free(checkpoint_path);
    if (verify_checkpoint) try verifyFile(checkpoint_path, &checkpoint_sha, try size(shard, "Size"), cancel_fd, cancel);
    // Runtime.load uses Python's default json.dumps on these sorted two records.
    const identity_records = try std.fmt.allocPrint(gpa, "[[\"config.json\", \"{s}\"], [\"{s}\", \"{s}\"]]", .{ config_sha, shard_name, checkpoint_sha });
    defer gpa.free(identity_records);
    return .{ .path = checkpoint_path, .binding = .{
        .identity = shaBytes(identity_records),
        .manifest_sha256 = approved,
        .config_sha256 = config_sha,
        .checkpoint_sha256 = checkpoint_sha,
    } };
}

fn applyCore(m: *Model, io: std.Io, override: CoreOptions) !CoreBinding {
    if (m.config.format != .bf16) return error.NativeBF16Required;
    const base = m.base_binding orelse return error.UnboundNativeManifest;
    const approved = std.fmt.bytesToHex(override.approved_provenance_sha256, .lower);
    const provenance = try std.Io.Dir.cwd().readFileAlloc(io, override.provenance_path, m.gpa, .limited(1 << 20));
    defer m.gpa.free(provenance);
    const provenance_sha = shaBytes(provenance);
    if (!std.mem.eql(u8, &provenance_sha, &approved)) return error.NativeCoreProvenanceSHAMismatch;
    const parsed = try std.json.parseFromSlice(std.json.Value, m.gpa, provenance, .{});
    defer parsed.deinit();
    const record = try obj(parsed.value);
    try assertText(record, "format", "luban-native-bf16-core-v1");
    try assertText(record, "base_identity", &base.identity);
    try assertText(record, "official_manifest_sha256", &base.manifest_sha256);
    try assertText(record, "original_tensor_sha256", &base.original_down_sha256);
    try assertText(record, "parameter", down_name);
    try assertText(record, "source_parameter", "model.language_model.layers.23.mlp.base.down_proj.weight");
    try assertText(record, "dtype", "torch.bfloat16");
    const shape = record.get("shape") orelse return error.BadNativeProvenance;
    if (shape != .array or shape.array.items.len != 2) return error.BadNativeProvenance;
    for (shape.array.items, [_]usize{ cfg.hidden, cfg.intermediate }) |d, expected| {
        if (d != .integer or d.integer != @as(i64, @intCast(expected))) return error.BadNativeProvenance;
    }
    try assertBool(record, "safety_passed", true);
    try assertBool(record, "quality_passed", true);
    try assertBool(record, "branch_enabled", false);
    try assertBool(record, "reference_used", false);
    const file_sha = try copySha(try text(record, "file_sha256"));
    const tensor_sha = try copySha(try text(record, "native_tensor_sha256"));
    const closed_sha = try copySha(try text(record, "closed_validation_sha256"));
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, override.path, m.gpa, .limited(32 << 20));
    defer m.gpa.free(raw);
    const actual_file_sha = shaBytes(raw);
    if (!std.mem.eql(u8, &actual_file_sha, &file_sha)) return error.NativeCoreFileSHAMismatch;
    const bytes = try coreTensorBytes(m.gpa, raw, down_name);
    const actual_tensor_sha = shaBytes(bytes);
    if (!std.mem.eql(u8, &actual_tensor_sha, &tensor_sha)) return error.NativeCoreTensorSHAMismatch;
    try installCoreTensor(m, bytes);
    return .{ .provenance_sha256 = approved, .file_sha256 = file_sha, .tensor_sha256 = tensor_sha, .closed_validation_sha256 = closed_sha };
}

fn coreTensorBytes(gpa: std.mem.Allocator, raw: []const u8, key: []const u8) ![]const u8 {
    if (raw.len < 8) return error.BadSafetensors;
    const header_len = std.mem.readInt(u64, raw[0..8], .little);
    if (header_len > raw.len - 8) return error.BadSafetensors;
    const begin: usize = 8 + @as(usize, @intCast(header_len));
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const header = try safetensors.parseHeader(arena.allocator(), raw[8..begin], raw.len - begin);
    if (header.count() != 1) return error.UnexpectedNativeCoreTensor;
    const entry = header.get(key) orelse return error.UnexpectedNativeCoreTensor;
    if (entry.dtype != .bf16 or entry.rank != 2 or entry.shape[0] != cfg.hidden or entry.shape[1] != cfg.intermediate or
        entry.begin != 0 or entry.end != raw.len - begin) return error.UnexpectedNativeCoreTensor;
    const bytes = raw[begin..];
    var at: usize = 0;
    while (at < bytes.len) : (at += 2) {
        const bits = std.mem.readInt(u16, bytes[at..][0..2], .little);
        if (bits & 0x7f80 == 0x7f80) return error.NonfiniteNativeCoreTensor;
    }
    return bytes;
}

fn installCoreTensor(m: *Model, bytes: []const u8) !void {
    const buffer = try m.device.buffer(bytes.len, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    var owned = false;
    errdefer if (!owned) buffer.deinit();
    @memcpy(buffer.contents()[0..bytes.len], bytes);
    try m.checkpoint.own(buffer, bytes.len);
    owned = true;
    var weight = ckpt.Tensor{ .buffer = buffer, .offset = 0, .bytes = bytes.len, .dtype = .bf16, .rank = 2 };
    weight.shape[0] = cfg.hidden;
    weight.shape[1] = cfg.intermediate;
    m.weights.blocks[cfg.layers - 1].down = try weights.denseProjection(weight, cfg.hidden, cfg.intermediate);
}
