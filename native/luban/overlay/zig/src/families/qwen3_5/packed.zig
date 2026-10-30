// SPDX-License-Identifier: Apache-2.0
//! Source-bound, streaming affine Q4/group16 payload. No Metal or quantizer dependency.
const std = @import("std");

pub const projection_count = 71;
pub const header_bytes = 144;
pub const sparse_header_bytes = 148;
pub const residual_header_bytes = 152;
pub const residual_bytes = 8 * (hidden + intermediate) * 2;
pub const all_layers: u32 = 0x00ffffff;
const hidden: usize = 2048;
const intermediate: usize = 6144;
pub const payload_bytes = hidden * intermediate * 3 / 4;
pub const file_bytes = header_bytes + projection_count * (8 + payload_bytes);

fn validSha(value: []const u8) bool {
    if (value.len != 64) return false;
    for (value) |ch| if (!(ch >= '0' and ch <= '9') and !(ch >= 'a' and ch <= 'f')) return false;
    return true;
}

pub const Header = struct {
    version: u8 = 1,
    residual_rank: usize = 0,
    layer_mask: u32 = all_layers,
    projection_count: usize = 71,
    header_bytes: usize = 144,
    file_bytes: u64 = 144 + 71 * (8 + payload_bytes),

    pub fn containsLayer(self: Header, layer: usize) bool {
        return layer < 24 and self.layer_mask & (@as(u32, 1) << @intCast(layer)) != 0;
    }

    fn projectionIndex(self: Header, ordinal: usize) !usize {
        var at: usize = 0;
        for (0..24) |layer| {
            if (!self.containsLayer(layer)) continue;
            const count: usize = if (layer == 23) 2 else 3;
            if (ordinal >= at and ordinal - at < count) return layer * 3 + ordinal - at;
            at += count;
        }
        return error.InvalidPackedProjection;
    }
};

pub fn validateHeader(raw: []const u8, manifest: []const u8, core: []const u8) !Header {
    if ((raw.len != header_bytes and raw.len != sparse_header_bytes and raw.len != residual_header_bytes) or !validSha(manifest) or !validSha(core))
        return error.InvalidPackedHeader;
    var result: Header = .{};
    const residual = std.mem.eql(u8, raw[0..8], "LQ4PK03\n") and raw.len == residual_header_bytes;
    if (residual or (std.mem.eql(u8, raw[0..8], "LQ4PK02\n") and raw.len == sparse_header_bytes)) {
        result.version = if (residual) 3 else 2;
        if (residual) {
            result.residual_rank = std.mem.readInt(u32, raw[148..152], .little);
            if (result.residual_rank != 8) return error.InvalidPackedHeader;
        }
        result.layer_mask = std.mem.readInt(u32, raw[144..148], .little);
        if (result.layer_mask == 0 or result.layer_mask & ~all_layers != 0) return error.InvalidPackedHeader;
        result.projection_count = 3 * @as(usize, @popCount(result.layer_mask)) - @as(usize, @intFromBool(result.containsLayer(23)));
        result.header_bytes = raw.len;
        result.file_bytes = raw.len + result.projection_count * (8 + payload_bytes + (if (residual) residual_bytes else @as(usize, 0)));
    } else if (!std.mem.eql(u8, raw[0..8], "LQ4PK01\n") or raw.len != header_bytes) return error.InvalidPackedHeader;
    if (std.mem.readInt(u32, raw[8..12], .little) != 16 or
        std.mem.readInt(u32, raw[12..16], .little) != result.projection_count) return error.InvalidPackedHeader;
    if (!std.mem.eql(u8, raw[16..80], manifest) or !std.mem.eql(u8, raw[80..144], core)) return error.PackedSourceMismatch;
    return result;
}

/// Fixed rank-eight column protection: one-hot or unused basis rows, BF16 updates.
/// Returns whether a nonzero update needs the existing residual GPU operators.
pub fn validateResidual(raw: []const u8, n: usize, k: usize) !bool {
    if (!((n == hidden and k == intermediate) or (n == intermediate and k == hidden)) or raw.len != residual_bytes)
        return error.InvalidPackedResidual;
    var columns: [8]?usize = @splat(null);
    for (0..8) |r| {
        for (0..k) |j| {
            const value = std.mem.readInt(u16, raw[(r * k + j) * 2 ..][0..2], .little);
            if (value & 0x7fff == 0) continue;
            if (value != 0x3f80 or columns[r] != null) return error.InvalidPackedResidual;
            for (columns[0..r]) |previous| if (previous == j) return error.InvalidPackedResidual;
            columns[r] = j;
        }
    }
    var active = false;
    for (0..n) |i| for (0..8) |r| {
        const value = std.mem.readInt(u16, raw[(8 * k + i * 8 + r) * 2 ..][0..2], .little);
        if (value & 0x7f80 == 0x7f80 or (columns[r] == null and value & 0x7fff != 0)) return error.InvalidPackedResidual;
        active = active or value & 0x7fff != 0;
    };
    return active;
}

fn validateDimensions(raw: []const u8, index: usize, n: usize, k: usize) !void {
    if (raw.len != 8 or index >= projection_count) return error.InvalidPackedProjection;
    const down = index % 3 == 2;
    if (n != (if (down) hidden else intermediate) or k != (if (down) intermediate else hidden) or
        std.mem.readInt(u32, raw[0..4], .little) != n or
        std.mem.readInt(u32, raw[4..8], .little) != k) return error.InvalidPackedProjection;
}

pub fn validateMetadata(scales: []const u8, biases: []const u8) !void {
    if (scales.len == 0 or scales.len % 2 != 0 or scales.len != biases.len) return error.InvalidPackedMetadata;
    var at: usize = 0;
    while (at < scales.len) : (at += 2) {
        const scale = std.mem.readInt(u16, scales[at..][0..2], .little);
        const bias = std.mem.readInt(u16, biases[at..][0..2], .little);
        // Signed scales are intentional in the original endpoint quantizer.
        if (scale & 0x7fff == 0 or scale & 0x7f80 == 0x7f80 or bias & 0x7f80 == 0x7f80)
            return error.InvalidPackedMetadata;
        const s: f32 = @bitCast(@as(u32, scale) << 16);
        const b: f32 = @bitCast(@as(u32, bias) << 16);
        if (!std.math.isFinite(s * 15 + b)) return error.InvalidPackedMetadata;
    }
}

pub const Reader = struct {
    file: std.Io.File,
    io: std.Io,
    offset: u64 = 0,
    count: usize = 0,
    residual_projections: usize = 0,
    header: Header = .{},

    pub fn open(io: std.Io, path: []const u8, manifest: []const u8, core: []const u8) !Reader {
        if (!std.fs.path.isAbsolute(path)) return error.InvalidPackedPath;
        const file = try std.Io.Dir.cwd().openFile(io, path, .{ .follow_symlinks = false });
        errdefer file.close(io);
        const stat = try file.stat(io);
        if (stat.kind != .file or stat.size < header_bytes) return error.InvalidPackedLength;
        var reader = Reader{ .file = file, .io = io };
        var raw: [residual_header_bytes]u8 = undefined;
        try reader.readExact(raw[0..header_bytes]);
        const size: usize = if (std.mem.eql(u8, raw[0..8], "LQ4PK03\n")) residual_header_bytes else
            if (std.mem.eql(u8, raw[0..8], "LQ4PK02\n")) sparse_header_bytes else header_bytes;
        if (size > header_bytes) try reader.readExact(raw[header_bytes..size]);
        reader.header = try validateHeader(raw[0..size], manifest, core);
        if (stat.size != reader.header.file_bytes) return error.InvalidPackedLength;
        return reader;
    }

    pub fn close(reader: *Reader) void {
        reader.file.close(reader.io);
        reader.* = undefined;
    }

    fn readExact(reader: *Reader, target: []u8) !void {
        const count = try reader.file.readPositionalAll(reader.io, target, reader.offset);
        if (count != target.len) return error.TruncatedPackedPayload;
        reader.offset += count;
    }

    /// Read directly into the destination's owning Metal buffer, one projection at a time.
    pub fn projection(reader: *Reader, target: []u8, n: usize, k: usize) !void {
        const extra: usize = if (reader.header.residual_rank != 0) residual_bytes else 0;
        if (target.len != payload_bytes + extra) return error.InvalidPackedLength;
        var dimensions: [8]u8 = undefined;
        try reader.readExact(&dimensions);
        try validateDimensions(&dimensions, try reader.header.projectionIndex(reader.count), n, k);
        try reader.readExact(target);
        const words_bytes = n * k / 2;
        const metadata_bytes = n * k / 16 * 2;
        try validateMetadata(target[words_bytes..][0..metadata_bytes], target[words_bytes + metadata_bytes .. payload_bytes]);
        if (extra != 0 and try validateResidual(target[payload_bytes..], n, k)) reader.residual_projections += 1;
        reader.count += 1;
    }

    pub fn finish(reader: *Reader) !void {
        if (reader.count != reader.header.projection_count or reader.offset != reader.header.file_bytes) return error.IncompletePackedPayload;
        var extra: [1]u8 = undefined;
        if (try reader.file.readPositionalAll(reader.io, &extra, reader.offset) != 0 or
            (try reader.file.stat(reader.io)).size != reader.header.file_bytes) return error.InvalidPackedLength;
    }
};

fn testHeader() [header_bytes]u8 {
    var header: [header_bytes]u8 = @splat('0');
    @memcpy(header[0..8], "LQ4PK01\n");
    std.mem.writeInt(u32, header[8..12], 16, .little);
    std.mem.writeInt(u32, header[12..16], projection_count, .little);
    @memset(header[16..80], 'a');
    return header;
}

test "packed header rejects wrong format, source, groups, count and short input" {
    const manifest: [64]u8 = @splat('a');
    const core: [64]u8 = @splat('0');
    var header = testHeader();
    _ = try validateHeader(&header, &manifest, &core);
    try std.testing.expectError(error.InvalidPackedHeader, validateHeader(header[0..143], &manifest, &core));
    for ([_]usize{ 0, 8, 12 }) |at| {
        header = testHeader(); header[at] +%= 1;
        try std.testing.expectError(error.InvalidPackedHeader, validateHeader(&header, &manifest, &core));
    }
    for ([_]usize{ 16, 80 }) |at| {
        header = testHeader(); header[at] = 'b';
        try std.testing.expectError(error.PackedSourceMismatch, validateHeader(&header, &manifest, &core));
    }
}

test "packed dimensions enforce all71 order including final gate and up" {
    var dims: [8]u8 = undefined;
    for (0..projection_count) |index| {
        const n: usize = if (index % 3 == 2) hidden else intermediate;
        const k: usize = if (index % 3 == 2) intermediate else hidden;
        std.mem.writeInt(u32, dims[0..4], @intCast(n), .little);
        std.mem.writeInt(u32, dims[4..8], @intCast(k), .little);
        try validateDimensions(&dims, index, n, k);
        try std.testing.expectError(error.InvalidPackedProjection, validateDimensions(&dims, index, k, n));
    }
    try std.testing.expectError(error.InvalidPackedProjection, validateDimensions(&dims, 71, hidden, intermediate));
}

test "packed BF16 metadata accepts signed scales and rejects zero nonfinite and overflow" {
    var scale: [2]u8 = undefined;
    var bias: [2]u8 = .{ 0, 0 };
    for ([_]u16{ 0x3f80, 0xbf80 }) |value| {
        std.mem.writeInt(u16, &scale, value, .little);
        try validateMetadata(&scale, &bias);
    }
    for ([_]u16{ 0, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0x7f7f }) |value| {
        std.mem.writeInt(u16, &scale, value, .little);
        try std.testing.expectError(error.InvalidPackedMetadata, validateMetadata(&scale, &bias));
    }
    std.mem.writeInt(u16, &scale, 0x3f80, .little);
    std.mem.writeInt(u16, &bias, 0x7fc0, .little);
    try std.testing.expectError(error.InvalidPackedMetadata, validateMetadata(&scale, &bias));
    try std.testing.expectError(error.InvalidPackedMetadata, validateMetadata(scale[0..1], &bias));
}

test "packed streaming detects truncated reads and trailing data without GPU allocation" {
    const io = std.testing.io;
    var dir = std.testing.tmpDir(.{});
    defer dir.cleanup();
    const file = try dir.dir.createFile(io, "payload", .{ .read = true });
    defer file.close(io);
    try file.writePositionalAll(io, "abc", 0);
    var reader = Reader{ .file = file, .io = io };
    var bytes: [4]u8 = undefined;
    try std.testing.expectError(error.TruncatedPackedPayload, reader.readExact(&bytes));
    try std.testing.expectError(error.IncompletePackedPayload, reader.finish());
    // Sparse file: validate exact EOF without allocating or writing the 639MiB payload.
    try file.setLength(io, file_bytes);
    reader.offset = file_bytes;
    reader.count = projection_count;
    try reader.finish();
    try file.writePositionalAll(io, "x", file_bytes);
    try std.testing.expectError(error.InvalidPackedLength, reader.finish());
}

test "packed file reader binds source and reads the real projection layout directly" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var dir = std.testing.tmpDir(.{});
    defer dir.cleanup();
    const file = try dir.dir.createFile(io, "payload", .{ .read = true });
    defer file.close(io);
    const path = try dir.dir.realPathFileAlloc(io, "payload", allocator);
    defer allocator.free(path);
    const manifest: [64]u8 = @splat('a');
    const core: [64]u8 = @splat('0');
    try std.testing.expectError(error.InvalidPackedPath, Reader.open(io, "relative.bin", &manifest, &core));
    try std.testing.expectError(error.InvalidPackedLength, Reader.open(io, path, &manifest, &core));
    try file.setLength(io, file_bytes);
    const header = testHeader();
    try file.writePositionalAll(io, &header, 0);
    const wrong: [64]u8 = @splat('b');
    try std.testing.expectError(error.PackedSourceMismatch, Reader.open(io, path, &wrong, &core));
    var dimensions: [8]u8 = undefined;
    std.mem.writeInt(u32, dimensions[0..4], intermediate, .little);
    std.mem.writeInt(u32, dimensions[4..8], hidden, .little);
    try file.writePositionalAll(io, &dimensions, header_bytes);
    const original = try allocator.alloc(u8, payload_bytes);
    defer allocator.free(original);
    @memset(original, 0);
    const words_bytes = hidden * intermediate / 2;
    const metadata_bytes = hidden * intermediate / 16 * 2;
    var at: usize = words_bytes;
    while (at < words_bytes + metadata_bytes) : (at += 2)
        std.mem.writeInt(u16, original[at..][0..2], if (at % 4 == 0) 0x3f80 else 0xbf80, .little);
    original[0] = 0x32; original[1] = 0x10;
    try file.writePositionalAll(io, original, header_bytes + 8);
    var reader = try Reader.open(io, path, &manifest, &core);
    defer reader.close();
    const actual = try allocator.alloc(u8, payload_bytes);
    defer allocator.free(actual);
    try reader.projection(actual, intermediate, hidden);
    try std.testing.expectEqualSlices(u8, original, actual);
    try std.testing.expectEqual(@as(usize, 1), reader.count);
    try std.testing.expectError(error.IncompletePackedPayload, reader.finish());
}

fn testSparseHeader(mask: u32) [sparse_header_bytes]u8 {
    var raw: [sparse_header_bytes]u8 = undefined;
    @memcpy(raw[0..header_bytes], &testHeader());
    @memcpy(raw[0..8], "LQ4PK02\n");
    const count = 3 * @as(u32, @popCount(mask)) - @as(u32, @intFromBool(mask & (1 << 23) != 0));
    std.mem.writeInt(u32, raw[12..16], count, .little);
    std.mem.writeInt(u32, raw[144..148], mask, .little);
    return raw;
}

fn testResidualHeader(mask: u32) [residual_header_bytes]u8 {
    var raw: [residual_header_bytes]u8 = undefined;
    @memcpy(raw[0..sparse_header_bytes], &testSparseHeader(mask));
    @memcpy(raw[0..8], "LQ4PK03\n");
    std.mem.writeInt(u32, raw[148..152], 8, .little);
    return raw;
}

test "v3 binds declared rank and sparse layout without changing v1 or v2" {
    const manifest: [64]u8 = @splat('a');
    const core: [64]u8 = @splat('0');
    for (0..24) |layer| {
        const raw = testResidualHeader(@as(u32, 1) << @intCast(layer));
        const h = try validateHeader(&raw, &manifest, &core);
        const count: usize = if (layer == 23) 2 else 3;
        try std.testing.expectEqual(@as(u8, 3), h.version);
        try std.testing.expectEqual(@as(usize, 8), h.residual_rank);
        try std.testing.expectEqual(@as(u64, 152 + count * (8 + payload_bytes + residual_bytes)), h.file_bytes);
        try std.testing.expectError(error.InvalidPackedHeader, validateHeader(raw[0..148], &manifest, &core));
    }
    var raw = testResidualHeader(1 << 23);
    for ([_]u32{ 0, 7, 32, 0xffffffff }) |rank| {
        std.mem.writeInt(u32, raw[148..152], rank, .little);
        try std.testing.expectError(error.InvalidPackedHeader, validateHeader(&raw, &manifest, &core));
    }
    try std.testing.expectEqual(@as(u8, 1), (try validateHeader(&testHeader(), &manifest, &core)).version);
    const old = try validateHeader(&testSparseHeader(1), &manifest, &core);
    try std.testing.expectEqual(@as(u8, 2), old.version);
    try std.testing.expectEqual(@as(usize, 0), old.residual_rank);
}

test "rank8 onehot factors reject duplicate columns inactive updates nonfinite and bad lengths" {
    const a = std.testing.allocator;
    const raw = try a.alloc(u8, residual_bytes);
    defer a.free(raw);
    for ([_][2]usize{ .{ intermediate, hidden }, .{ hidden, intermediate } }) |shape| {
        const n = shape[0]; const k = shape[1];
        @memset(raw, 0);
        try std.testing.expect(!try validateResidual(raw, n, k));
        std.mem.writeInt(u16, raw[(k - 1) * 2 ..][0..2], 0x3f80, .little);
        std.mem.writeInt(u16, raw[(8 * k + (n - 1) * 8) * 2 ..][0..2], 0xbe80, .little);
        try std.testing.expect(try validateResidual(raw, n, k));
        std.mem.writeInt(u16, raw[(k + k - 1) * 2 ..][0..2], 0x3f80, .little);
        try std.testing.expectError(error.InvalidPackedResidual, validateResidual(raw, n, k));
        @memset(raw, 0);
        std.mem.writeInt(u16, raw[8 * k * 2 ..][0..2], 0x3f80, .little);
        try std.testing.expectError(error.InvalidPackedResidual, validateResidual(raw, n, k));
        for ([_]u16{ 0x3f00, 0xbf80, 0x7fc0 }) |value| {
            @memset(raw, 0); std.mem.writeInt(u16, raw[0..2], value, .little);
            try std.testing.expectError(error.InvalidPackedResidual, validateResidual(raw, n, k));
        }
        @memset(raw, 0); std.mem.writeInt(u16, raw[0..2], 0x3f80, .little);
        std.mem.writeInt(u16, raw[8 * k * 2 ..][0..2], 0x7f80, .little);
        try std.testing.expectError(error.InvalidPackedResidual, validateResidual(raw, n, k));
        try std.testing.expectError(error.InvalidPackedResidual, validateResidual(raw[0 .. raw.len - 2], n, k));
        try std.testing.expectError(error.InvalidPackedResidual, validateResidual(raw, n, k - 1));
    }
}

test "v3 streams integrated final-layer factors counts active projections and rejects malformed tail" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var dir = std.testing.tmpDir(.{}); defer dir.cleanup();
    const file = try dir.dir.createFile(io, "v3", .{ .read = true }); defer file.close(io);
    const path = try dir.dir.realPathFileAlloc(io, "v3", allocator); defer allocator.free(path);
    const manifest: [64]u8 = @splat('a'); const core: [64]u8 = @splat('0');
    const raw = testResidualHeader(1 << 23);
    const h = try validateHeader(&raw, &manifest, &core);
    try file.setLength(io, h.file_bytes);
    try file.writePositionalAll(io, &raw, 0);
    var dims: [8]u8 = undefined;
    std.mem.writeInt(u32, dims[0..4], intermediate, .little);
    std.mem.writeInt(u32, dims[4..8], hidden, .little);
    const payload = try allocator.alloc(u8, payload_bytes + residual_bytes); defer allocator.free(payload);
    @memset(payload, 0);
    const words = hidden * intermediate / 2;
    const scales = hidden * intermediate / 16 * 2;
    var at: usize = words;
    while (at < words + scales) : (at += 2) std.mem.writeInt(u16, payload[at..][0..2], 0x3f80, .little);
    for (0..2) |index| {
        if (index == 1) {
            std.mem.writeInt(u16, payload[payload_bytes + 2 * (hidden - 1) ..][0..2], 0x3f80, .little);
            std.mem.writeInt(u16, payload[payload_bytes + 8 * hidden * 2 ..][0..2], 0xbe80, .little);
        }
        const start = residual_header_bytes + index * (8 + payload.len);
        try file.writePositionalAll(io, &dims, start);
        try file.writePositionalAll(io, payload, start + 8);
    }
    var reader = try Reader.open(io, path, &manifest, &core); defer reader.close();
    try std.testing.expectError(error.InvalidPackedLength, reader.projection(payload[0..payload_bytes], intermediate, hidden));
    try reader.projection(payload, intermediate, hidden);
    try std.testing.expectEqual(@as(usize, 0), reader.residual_projections);
    try reader.projection(payload, intermediate, hidden);
    try std.testing.expectEqual(@as(usize, 1), reader.residual_projections);
    try reader.finish();
    try file.writePositionalAll(io, "x", h.file_bytes);
    try std.testing.expectError(error.InvalidPackedLength, reader.finish());
    try file.setLength(io, h.file_bytes - 2);
    try std.testing.expectError(error.InvalidPackedLength, Reader.open(io, path, &manifest, &core));
    try file.setLength(io, h.file_bytes);
    // Correct length, but infinity in the last update must fail before adoption.
    try file.writePositionalAll(io, &.{ 0x80, 0x7f }, h.file_bytes - 2);
    var invalid = try Reader.open(io, path, &manifest, &core); defer invalid.close();
    try invalid.projection(payload, intermediate, hidden);
    try std.testing.expectError(error.InvalidPackedResidual, invalid.projection(payload, intermediate, hidden));
}

test "sparse header counts selected layers preserves v1 and rejects invalid masks or lengths" {
    const manifest: [64]u8 = @splat('a');
    const core: [64]u8 = @splat('0');
    const old = try validateHeader(&testHeader(), &manifest, &core);
    try std.testing.expectEqual(@as(u32, all_layers), old.layer_mask);
    try std.testing.expectEqual(@as(u64, file_bytes), old.file_bytes);
    for (0..24) |layer| {
        const mask = @as(u32, 1) << @intCast(layer);
        const value = try validateHeader(&testSparseHeader(mask), &manifest, &core);
        const count: usize = if (layer == 23) 2 else 3;
        try std.testing.expectEqual(count, value.projection_count);
        try std.testing.expectEqual(@as(u64, sparse_header_bytes + count * (8 + payload_bytes)), value.file_bytes);
        for (0..24) |other| try std.testing.expectEqual(layer == other, value.containsLayer(other));
        for (0..count) |i| try std.testing.expectEqual(layer * 3 + i, try value.projectionIndex(i));
        try std.testing.expectError(error.InvalidPackedProjection, value.projectionIndex(count));
    }
    const mask = (1 << 0) | (1 << 7) | (1 << 23);
    const value = try validateHeader(&testSparseHeader(mask), &manifest, &core);
    for ([_]usize{ 0, 1, 2, 21, 22, 23, 69, 70 }, 0..) |index, ordinal|
        try std.testing.expectEqual(index, try value.projectionIndex(ordinal));
    try std.testing.expect(!value.containsLayer(24));
    for ([_]u32{ 0, 1 << 24, 0xffffffff }) |invalid|
        try std.testing.expectError(error.InvalidPackedHeader, validateHeader(&testSparseHeader(invalid), &manifest, &core));
    var raw = testSparseHeader(mask);
    raw[12] += 1;
    try std.testing.expectError(error.InvalidPackedHeader, validateHeader(&raw, &manifest, &core));
    raw = testSparseHeader(mask);
    try std.testing.expectError(error.InvalidPackedHeader, validateHeader(raw[0..144], &manifest, &core));
    raw[6] = '1';
    try std.testing.expectError(error.InvalidPackedHeader, validateHeader(&raw, &manifest, &core));
}

test "sparse reader streams final gate up and enforces dynamic EOF and metadata" {
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    var dir = std.testing.tmpDir(.{});
    defer dir.cleanup();
    const file = try dir.dir.createFile(io, "sparse", .{ .read = true });
    defer file.close(io);
    const path = try dir.dir.realPathFileAlloc(io, "sparse", allocator);
    defer allocator.free(path);
    const manifest: [64]u8 = @splat('a');
    const core: [64]u8 = @splat('0');
    const raw = testSparseHeader(1 << 23);
    const metadata = try validateHeader(&raw, &manifest, &core);
    try file.setLength(io, metadata.file_bytes);
    try file.writePositionalAll(io, &raw, 0);
    var dims: [8]u8 = undefined;
    std.mem.writeInt(u32, dims[0..4], intermediate, .little);
    std.mem.writeInt(u32, dims[4..8], hidden, .little);
    const payload = try allocator.alloc(u8, payload_bytes);
    defer allocator.free(payload);
    @memset(payload, 0);
    const words = hidden * intermediate / 2;
    const scale_bytes = hidden * intermediate / 16 * 2;
    var at: usize = words;
    while (at < words + scale_bytes) : (at += 2) std.mem.writeInt(u16, payload[at..][0..2], 0xbf80, .little);
    for (0..2) |index| {
        const start = sparse_header_bytes + index * (8 + payload_bytes);
        try file.writePositionalAll(io, &dims, start);
        try file.writePositionalAll(io, payload, start + 8);
    }
    var reader = try Reader.open(io, path, &manifest, &core);
    defer reader.close();
    try std.testing.expectEqual(@as(usize, 2), reader.header.projection_count);
    try std.testing.expectError(error.IncompletePackedPayload, reader.finish());
    try reader.projection(payload, intermediate, hidden);
    try reader.projection(payload, intermediate, hidden);
    try reader.finish();
    try file.writePositionalAll(io, "x", metadata.file_bytes);
    try std.testing.expectError(error.InvalidPackedLength, reader.finish());
    try std.testing.expectError(error.InvalidPackedLength, Reader.open(io, path, &manifest, &core));
    try file.setLength(io, metadata.file_bytes);
    // A bad scale in a correctly sized sparse carrier must fail before adoption.
    try file.writePositionalAll(io, &.{ 0, 0 }, sparse_header_bytes + 8 + words);
    var invalid = try Reader.open(io, path, &manifest, &core);
    defer invalid.close();
    try std.testing.expectError(error.InvalidPackedMetadata, invalid.projection(payload, intermediate, hidden));
    try std.testing.expectEqual(@as(usize, 0), invalid.count);
}
