// SPDX-License-Identifier: Apache-2.0
//! Complete inference checkpoint, prepared once with bounded CPU storage.
//! The approved file digest covers every protected and quantized byte.
const std = @import("std");
const st = @import("../../core/safetensors.zig");
const affine = @import("../../core/affine4.zig");
const packed_q4 = @import("packed.zig");
pub const format = "luban-direct-q4-g16-v1";
pub const prefix = "model.language_model.";
pub const down_name = prefix ++ "layers.23.mlp.down_proj.weight";

pub const Metadata = struct {
    format: []const u8,
    quantizer_sha256: []const u8,
    base_identity: []const u8,
    manifest_sha256: []const u8,
    config_sha256: []const u8,
    checkpoint_sha256: []const u8,
    original_down_sha256: []const u8,
};

pub fn sha(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn validSha(value: []const u8) bool {
    if (value.len != 64) return false;
    for (value) |ch| if (!(ch >= '0' and ch <= '9') and !(ch >= 'a' and ch <= 'f')) return false;
    return true;
}

pub fn validateMetadata(metadata: Metadata, expected: Metadata) !void {
    if (!std.mem.eql(u8, metadata.format, format)) return error.InvalidDirectQ4Format;
    inline for (.{ "quantizer_sha256", "base_identity", "manifest_sha256", "config_sha256", "checkpoint_sha256", "original_down_sha256" }) |key| {
        const actual = @field(metadata, key);
        if (!validSha(actual) or ( @field(expected, key).len != 0 and !std.mem.eql(u8, actual, @field(expected, key))))
            return error.DirectQ4BindingMismatch;
    }
}

/// Exact fixed ordinary profile: all 71 MLP matrices except protected down23.
pub fn quantizedName(name: []const u8) bool {
    if (!std.mem.startsWith(u8, name, prefix ++ "layers.") or std.mem.eql(u8, name, down_name)) return false;
    return std.mem.endsWith(u8, name, ".mlp.gate_proj.weight") or
        std.mem.endsWith(u8, name, ".mlp.up_proj.weight") or std.mem.endsWith(u8, name, ".mlp.down_proj.weight");
}

pub const Source = struct {
    file: std.Io.File,
    io: std.Io,
    arena: std.heap.ArenaAllocator,
    raw: []const u8,
    entries: st.Header,
    start: usize,
    size: usize,

    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !Source {
        if (!std.fs.path.isAbsolute(path)) return error.InvalidDirectQ4Path;
        const file = try std.Io.Dir.cwd().openFile(io, path, .{ .follow_symlinks = false });
        errdefer file.close(io);
        const stat = try file.stat(io);
        if (stat.kind != .file or stat.size < 10) return error.InvalidDirectQ4Length;
        var head: [8]u8 = undefined;
        if (try file.readPositionalAll(io, &head, 0) != head.len) return error.TruncatedDirectQ4;
        const len = std.mem.readInt(u64, &head, .little);
        if (len > 4 << 20 or len > stat.size - 8) return error.InvalidDirectQ4Length;
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const raw = try arena.allocator().alloc(u8, @intCast(len));
        if (try file.readPositionalAll(io, raw, 8) != raw.len) return error.TruncatedDirectQ4;
        const entries = try st.parseHeaderPrefix(arena.allocator(), raw, @intCast(stat.size - len - 8), prefix);
        if (entries.count() == 0) return error.InvalidDirectQ4Length;
        return .{ .file = file, .io = io, .arena = arena, .raw = raw, .entries = entries, .start = @intCast(len + 8), .size = @intCast(stat.size) };
    }
    pub fn close(self: *Source) void {
        self.file.close(self.io);
        self.arena.deinit();
    }
    pub fn read(self: *const Source, target: []u8, offset: usize) !void {
        if (try self.file.readPositionalAll(self.io, target, offset) != target.len) return error.TruncatedDirectQ4;
    }
    pub fn readVerified(self: *Source, target: []u8, approved: [32]u8, cancel_fd: c_int, cancel: *const fn (c_int) anyerror!bool) !void {
        if (target.len != self.size - self.start) return error.InvalidDirectQ4Length;
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        var head: [8]u8 = undefined;
        std.mem.writeInt(u64, &head, @intCast(self.raw.len), .little);
        hash.update(&head); hash.update(self.raw);
        var at: usize = 0;
        while (at < target.len) {
            if (try cancel(cancel_fd)) return error.DirectQ4Cancelled;
            const chunk = target[at..@min(at + (4 << 20), target.len)];
            try self.read(chunk, self.start + at); hash.update(chunk); at += chunk.len;
        }
        var digest: [32]u8 = undefined; hash.final(&digest);
        if (!std.mem.eql(u8, &digest, &approved)) return error.DirectQ4FileSHAMismatch;
        if ((try self.file.stat(self.io)).size != self.size) return error.InvalidDirectQ4Length;
    }
    pub fn metadata(self: *Source) !Metadata {
        const root = try std.json.parseFromSliceLeaky(std.json.Value, self.arena.allocator(), self.raw, .{});
        if (root != .object) return error.InvalidDirectQ4Format;
        const value = root.object.get("__metadata__") orelse return error.InvalidDirectQ4Format;
        const raw = try std.json.Stringify.valueAlloc(self.arena.allocator(), value, .{});
        return try std.json.parseFromSliceLeaky(Metadata, self.arena.allocator(), raw, .{});
    }
};

const Output = struct { dtype: []const u8, shape: []const usize, data_offsets: [2]usize };
fn addEntry(a: std.mem.Allocator, root: *std.json.ObjectMap, name: []const u8, dtype: []const u8, shape: []const usize, bytes: usize, offset: *usize) !void {
    const raw = try std.json.Stringify.valueAlloc(a, Output{ .dtype = dtype, .shape = shape, .data_offsets = .{ offset.*, offset.* + bytes } }, .{});
    const value = try std.json.parseFromSliceLeaky(std.json.Value, a, raw, .{});
    try root.put(a, name, value);
    offset.* += bytes;
}
fn dtypeName(dtype: st.DType) ![]const u8 {
    return switch (dtype) { .bf16 => "BF16", .f32 => "F32", else => error.InvalidDirectQ4SourceDType };
}

/// Contiguous/nonoverlapping complete payload required, not merely individually valid offsets.
pub fn validateLayout(entries: st.Header, bytes: usize) !void {
    var next: usize = 0;
    for (entries.values()) |entry| {
        if (entry.begin != next or entry.end <= entry.begin) return error.InvalidDirectQ4Layout;
        next = entry.end;
    }
    if (next != bytes) return error.InvalidDirectQ4Layout;
}

pub const Prepared = struct { file_sha256: [64]u8, bytes: usize, file_bytes: usize, quantized_projections: usize, max_source_tensor_bytes: usize };

/// Stream protected tensors in 1MiB chunks; at most one 25MiB MLP matrix and
/// its Q4 words/scales/biases are live. No mmap, Metal or complete BF16 model.
pub fn prepare(gpa: std.mem.Allocator, io: std.Io, source_path: []const u8, output_path: []const u8, metadata: Metadata, cancel_fd: c_int, cancel: *const fn (c_int) anyerror!bool) !Prepared {
    try validateMetadata(metadata, metadata);
    if (!std.fs.path.isAbsolute(output_path) or std.mem.eql(u8, source_path, output_path)) return error.InvalidDirectQ4Path;
    var source = try Source.open(gpa, io, source_path);
    defer source.close();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var root: std.json.ObjectMap = .empty;
    const meta_raw = try std.json.Stringify.valueAlloc(a, metadata, .{});
    try root.put(a, "__metadata__", try std.json.parseFromSliceLeaky(std.json.Value, a, meta_raw, .{}));
    var offset: usize = 0;
    var count: usize = 0;
    var max_source: usize = 0;
    var it = source.entries.iterator();
    while (it.next()) |kv| {
        const name = kv.key_ptr.*;
        const entry = kv.value_ptr.*;
        if (quantizedName(name)) {
            if (entry.dtype != .bf16 or entry.rank != 2 or entry.shape[1] % 16 != 0) return error.InvalidDirectQ4Source;
            const n = entry.shape[0]; const k = entry.shape[1];
            const base = name[0 .. name.len - ".weight".len];
            try addEntry(a, &root, name, "U32", &.{ n, k / 8 }, n * k / 2, &offset);
            try addEntry(a, &root, try std.fmt.allocPrint(a, "{s}.scales", .{base}), "BF16", &.{ n, k / 16 }, n * k / 8, &offset);
            try addEntry(a, &root, try std.fmt.allocPrint(a, "{s}.biases", .{base}), "BF16", &.{ n, k / 16 }, n * k / 8, &offset);
            max_source = @max(max_source, entry.end - entry.begin);
            count += 1;
        } else try addEntry(a, &root, name, try dtypeName(entry.dtype), entry.shape[0..entry.rank], entry.end - entry.begin, &offset);
    }
    if (count != packed_q4.projection_count) return error.InvalidDirectQ4Source;
    const header = try std.json.Stringify.valueAlloc(a, std.json.Value{ .object = root }, .{});
    const header_len = std.mem.alignForward(usize, header.len, 8);
    const tmp = try std.fmt.allocPrint(a, "{s}.preparing", .{output_path});
    // Existing final output is never overwritten, even by a competing prepare.
    if (std.Io.Dir.cwd().openFile(io, output_path, .{ .follow_symlinks = false })) |existing| {
        existing.close(io); return error.DirectQ4OutputExists;
    } else |err| if (err != error.FileNotFound) return err;
    const output = try std.Io.Dir.cwd().createFile(io, tmp, .{ .exclusive = true });
    defer output.close(io);
    defer std.Io.Dir.cwd().deleteFile(io, tmp) catch {};
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var head: [8]u8 = undefined;
    std.mem.writeInt(u64, &head, @intCast(header_len), .little);
    try write(output, io, &hash, &head);
    try write(output, io, &hash, header);
    const padding: [8]u8 = @splat(' ');
    try write(output, io, &hash, padding[0 .. header_len - header.len]);
    const chunk = try gpa.alloc(u8, 1 << 20);
    defer gpa.free(chunk);
    it = source.entries.iterator();
    while (it.next()) |kv| {
        if (try cancel(cancel_fd)) return error.DirectQ4Cancelled;
        const e = kv.value_ptr.*;
        if (quantizedName(kv.key_ptr.*)) {
            const values = try gpa.alloc(u16, (e.end - e.begin) / 2);
            defer gpa.free(values);
            try source.read(std.mem.sliceAsBytes(values), source.start + e.begin);
            var quantized = try affine.prepareGroup(gpa, values, e.shape[0], e.shape[1], 16);
            defer quantized.deinit();
            try packed_q4.validateMetadata(std.mem.sliceAsBytes(quantized.scales), std.mem.sliceAsBytes(quantized.biases));
            if (try cancel(cancel_fd)) return error.DirectQ4Cancelled;
            try write(output, io, &hash, std.mem.sliceAsBytes(quantized.words));
            try write(output, io, &hash, std.mem.sliceAsBytes(quantized.scales));
            try write(output, io, &hash, std.mem.sliceAsBytes(quantized.biases));
        } else {
            var at = e.begin;
            var down_hash = std.crypto.hash.sha2.Sha256.init(.{});
            while (at < e.end) {
                if (try cancel(cancel_fd)) return error.DirectQ4Cancelled;
                const bytes = chunk[0..@min(chunk.len, e.end - at)];
                try source.read(bytes, source.start + at);
                if (std.mem.eql(u8, kv.key_ptr.*, down_name)) down_hash.update(bytes);
                try write(output, io, &hash, bytes);
                at += bytes.len;
            }
            if (std.mem.eql(u8, kv.key_ptr.*, down_name)) {
                var digest: [32]u8 = undefined; down_hash.final(&digest);
                if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), metadata.original_down_sha256)) return error.DirectQ4BindingMismatch;
            }
        }
    }
    if ((try source.file.stat(io)).size != source.size) return error.InvalidDirectQ4Length;
    if (try cancel(cancel_fd)) return error.DirectQ4Cancelled;
    try output.sync(io);
    // EXCL rename is unavailable on some external ExFAT volumes. The exclusive
    // .preparing file already serializes producers for this destination; in that
    // case recheck absence under that lock, then atomically rename the complete file.
    std.Io.Dir.cwd().renamePreserve(tmp, std.Io.Dir.cwd(), output_path, io) catch |err| {
        if (err != error.OperationUnsupported) return err;
        if (std.Io.Dir.cwd().openFile(io, output_path, .{ .follow_symlinks = false })) |existing| {
            existing.close(io); return error.DirectQ4OutputExists;
        } else |check_err| if (check_err != error.FileNotFound) return check_err;
        try std.Io.Dir.cwd().rename(tmp, std.Io.Dir.cwd(), output_path, io);
    };
    var digest: [32]u8 = undefined; hash.final(&digest);
    return .{ .file_sha256 = std.fmt.bytesToHex(digest, .lower), .bytes = offset, .file_bytes = 8 + header_len + offset,
        .quantized_projections = count, .max_source_tensor_bytes = max_source };
}

fn write(file: std.Io.File, io: std.Io, hash: *std.crypto.hash.sha2.Sha256, bytes: []const u8) !void {
    try file.writeStreamingAll(io, bytes);
    hash.update(bytes);
}

test "direct q4 rejects source format and every stale binding" {
    const a: [64]u8 = @splat('a'); const b: [64]u8 = @splat('b');
    const expected = Metadata{ .format = format, .quantizer_sha256 = &a, .base_identity = &a, .manifest_sha256 = &a, .config_sha256 = &a,
        .checkpoint_sha256 = &a, .original_down_sha256 = &a };
    try validateMetadata(expected, expected);
    var changed = expected; changed.format = "other";
    try std.testing.expectError(error.InvalidDirectQ4Format, validateMetadata(changed, expected));
    inline for (.{ "quantizer_sha256", "base_identity", "manifest_sha256", "config_sha256", "checkpoint_sha256", "original_down_sha256" }) |key| {
        changed = expected; @field(changed, key) = &b;
        try std.testing.expectError(error.DirectQ4BindingMismatch, validateMetadata(changed, expected));
        @field(changed, key) = "bad";
        try std.testing.expectError(error.DirectQ4BindingMismatch, validateMetadata(changed, expected));
    }
}

test "direct q4 protects exact down23 and rejects holes overlap and trailing bytes" {
    try std.testing.expect(quantizedName(prefix ++ "layers.23.mlp.gate_proj.weight"));
    try std.testing.expect(!quantizedName(down_name));
    try std.testing.expect(!quantizedName(prefix ++ "layers.23.self_attn.q_proj.weight"));
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator); defer arena.deinit();
    const entries = try st.parseHeader(arena.allocator(),
        "{\"a\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[0,2]},\"b\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[2,4]}}", 4);
    try validateLayout(entries, 4);
    try std.testing.expectError(error.InvalidDirectQ4Layout, validateLayout(entries, 5));
    var invalid = entries; invalid.values()[1].begin = 1;
    try std.testing.expectError(error.InvalidDirectQ4Layout, validateLayout(invalid, 4));
    invalid.values()[1].begin = 3;
    try std.testing.expectError(error.InvalidDirectQ4Layout, validateLayout(invalid, 4));
}

const TestCancel = struct {
    var calls: usize = 0;
    var stop_at: usize = std.math.maxInt(usize);
    fn check(_: c_int) anyerror!bool { calls += 1; return calls >= stop_at; }
};

test "direct q4 CPU roundtrip uses original affine bytes protects BF16 FP32 and rejects corruption cancellation partial publication" {
    const a = std.testing.allocator; const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{}); defer tmp.cleanup();
    var arena = std.heap.ArenaAllocator.init(a); defer arena.deinit();
    const scratch = arena.allocator();
    var root: std.json.ObjectMap = .empty;
    var offset: usize = 0;
    var values: [32]u16 = undefined;
    for (&values, 0..) |*v, i| v.* = affine.asBf16(@as(f32, @floatFromInt(i)) / 17 - 0.5);
    for (0..24) |layer| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |projection| {
        const name = try std.fmt.allocPrint(scratch, prefix ++ "layers.{d}.mlp.{s}.weight", .{ layer, projection });
        try addEntry(scratch, &root, name, "BF16", &.{ 2, 16 }, 64, &offset);
    };
    try addEntry(scratch, &root, prefix ++ "norm.weight", "F32", &.{16}, 64, &offset);
    const raw_header = try std.json.Stringify.valueAlloc(scratch, std.json.Value{ .object = root }, .{});
    const original_file = try tmp.dir.createFile(io, "source.safetensors", .{ .read = true });
    defer original_file.close(io);
    var head: [8]u8 = undefined; std.mem.writeInt(u64, &head, raw_header.len, .little);
    try original_file.writeStreamingAll(io, &head); try original_file.writeStreamingAll(io, raw_header);
    for (0..73) |_| try original_file.writeStreamingAll(io, std.mem.sliceAsBytes(&values));
    const source_path = try tmp.dir.realPathFileAlloc(io, "source.safetensors", scratch);
    const output_path = try std.fmt.allocPrint(scratch, "{s}.direct", .{source_path});
    const binding: [64]u8 = @splat('a'); const original_sha = sha(std.mem.sliceAsBytes(&values));
    const meta = Metadata{ .format = format, .quantizer_sha256 = &binding, .base_identity = &binding,
        .manifest_sha256 = &binding, .config_sha256 = &binding, .checkpoint_sha256 = &binding, .original_down_sha256 = &original_sha };
    TestCancel.calls = 0; TestCancel.stop_at = 3;
    try std.testing.expectError(error.DirectQ4Cancelled, prepare(a, io, source_path, output_path, meta, 3, TestCancel.check));
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().openFile(io, output_path, .{}));
    const preparing = try std.fmt.allocPrint(scratch, "{s}.preparing", .{output_path});
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().openFile(io, preparing, .{}));
    TestCancel.calls = 0; TestCancel.stop_at = std.math.maxInt(usize);
    const result = try prepare(a, io, source_path, output_path, meta, 3, TestCancel.check);
    try std.testing.expectEqual(@as(usize, 71), result.quantized_projections);
    try std.testing.expectError(error.DirectQ4OutputExists, prepare(a, io, source_path, output_path, meta, 3, TestCancel.check));
    var loaded = try Source.open(a, io, output_path); defer loaded.close();
    try validateMetadata(try loaded.metadata(), meta);
    try validateLayout(loaded.entries, loaded.size - loaded.start);
    const data = try a.alloc(u8, loaded.size - loaded.start); defer a.free(data);
    var approved: [32]u8 = undefined; _ = try std.fmt.hexToBytes(&approved, &result.file_sha256);
    try loaded.readVerified(data, approved, 3, TestCancel.check);
    var quant = try affine.prepareGroup(a, &values, 2, 16, 16); defer quant.deinit();
    const weight = loaded.entries.get(prefix ++ "layers.0.mlp.gate_proj.weight").?;
    const scale = loaded.entries.get(prefix ++ "layers.0.mlp.gate_proj.scales").?;
    const bias = loaded.entries.get(prefix ++ "layers.0.mlp.gate_proj.biases").?;
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(quant.words), data[weight.begin..weight.end]);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(quant.scales), data[scale.begin..scale.end]);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(quant.biases), data[bias.begin..bias.end]);
    for ([_][]const u8{ down_name, prefix ++ "norm.weight" }) |name| {
        const e = loaded.entries.get(name).?;
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&values), data[e.begin..e.end]);
    }
    try std.testing.expectEqual(st.DType.f32, loaded.entries.get(prefix ++ "norm.weight").?.dtype);
    approved[0] ^= 1;
    try std.testing.expectError(error.DirectQ4FileSHAMismatch, loaded.readVerified(data, approved, 3, TestCancel.check));
    approved[0] ^= 1;
    const editable = try std.Io.Dir.cwd().openFile(io, output_path, .{ .mode = .read_write }); defer editable.close(io);
    const corrupt: [1]u8 = .{data[0] ^ 1}; try editable.writePositionalAll(io, &corrupt, loaded.start);
    try std.testing.expectError(error.DirectQ4FileSHAMismatch, loaded.readVerified(data, approved, 3, TestCancel.check));
    try editable.setLength(io, loaded.size - 1);
    try std.testing.expectError(error.TruncatedDirectQ4, loaded.readVerified(data, approved, 3, TestCancel.check));
    try std.testing.expectError(error.BadSafetensors, Source.open(a, io, output_path));
}
