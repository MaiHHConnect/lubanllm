// SPDX-License-Identifier: Apache-2.0
//! Bounded, source-bound raw native MLP input carrier. No Metal dependency.
const std = @import("std");

pub const hidden = 2048;
pub const intermediate = 6144;
pub const layers = 24;
pub const streams = 47;
pub const row_values = layers * hidden + (layers - 1) * intermediate;
pub const header_bytes = 160;
pub const buffer_rows = 64;
pub const max_positions = 288;
pub const SourceKind = enum { original_bf16, packed_q4 };

/// Only the already supported core-less packed32 graphs may produce Q4 captures.
pub fn validateScope(mixed: bool, packed_version: u8, quantized: usize, residual_rank: usize) !void {
    if (!mixed and packed_version == 0 and quantized == 0 and residual_rank == 0) return;
    if (mixed and quantized == 32 and
        ((packed_version == 2 and residual_rank == 0) or (packed_version == 3 and residual_rank == 8))) return;
    return error.InvalidCaptureSource;
}

pub fn offset(layer: usize, down: bool) usize {
    return layer * (hidden + intermediate) + @as(usize, if (down) hidden else 0);
}

pub fn validate(prompt: usize, forced: usize, positions: []const u32) !void {
    if (prompt == 0 or prompt > 2048 or forced > 255 or prompt + forced > 2048 or
        positions.len == 0 or positions.len > max_positions) return error.InvalidCaptureShape;
    for (positions, 0..) |p, i| {
        if (p >= prompt + forced or (i != 0 and p <= positions[i - 1])) return error.InvalidCapturePosition;
    }
}

pub fn header(manifest: []const u8, prompt: usize, forced: usize, positions: []const u32) ![header_bytes]u8 {
    return headerForSource(.original_bf16, manifest, prompt, forced, positions);
}

pub fn headerForSource(source: SourceKind, manifest: []const u8, prompt: usize, forced: usize, positions: []const u32) ![header_bytes]u8 {
    try validate(prompt, forced, positions);
    if (manifest.len != 64) return error.InvalidCaptureSource;
    for (manifest) |ch| if (!(ch >= '0' and ch <= '9') and !(ch >= 'a' and ch <= 'f')) return error.InvalidCaptureSource;
    var raw: [header_bytes]u8 = undefined;
    @memcpy(raw[0..8], if (source == .original_bf16) "LNAC001\n" else "LNACQ01\n");
    for ([_]usize{ positions.len, row_values, prompt, forced, layers, streams }, 0..) |value, i|
        std.mem.writeInt(u32, raw[8 + 4 * i ..][0..4], @intCast(value), .little);
    @memcpy(raw[32..96], manifest);
    @memset(raw[96..160], '0');
    return raw;
}

/// GPU buffer is stream-major, with a fixed row stride. Carrier is row-major.
pub fn gather(destination: []u16, source: []const u16, stride: usize, row: usize) !void {
    if (stride == 0 or stride > 32 or row >= stride or source.len != row_values * stride or destination.len != row_values)
        return error.InvalidCaptureShape;
    for (0..layers) |layer| {
        for ([_]bool{ false, true }) |down| {
            if (down and layer == layers - 1) continue;
            const width: usize = if (down) intermediate else hidden;
            const begin = offset(layer, down);
            const values = source[begin * stride + row * width ..][0..width];
            for (values) |v| if (v & 0x7f80 == 0x7f80) return error.NonfiniteCapture;
            @memcpy(destination[begin..][0..width], values);
        }
    }
}

pub const Writer = struct {
    file: std.Io.File,
    io: std.Io,
    gpa: std.mem.Allocator,
    path: []u8,
    positions: []const u32,
    buffer: []u16,
    count: usize = 0,
    buffered: usize = 0,

    pub fn init(io: std.Io, gpa: std.mem.Allocator, directory: []const u8, id: []const u8, manifest: []const u8, prompt: usize, forced: usize, positions: []const u32) !Writer {
        return initForSource(io, gpa, directory, id, .original_bf16, manifest, prompt, forced, positions);
    }

    pub fn initForSource(io: std.Io, gpa: std.mem.Allocator, directory: []const u8, id: []const u8, source: SourceKind, manifest: []const u8, prompt: usize, forced: usize, positions: []const u32) !Writer {
        const raw = try headerForSource(source, manifest, prompt, forced, positions);
        if (!std.fs.path.isAbsolute(directory) or id.len == 0 or id.len > 128) return error.InvalidCapturePath;
        for (id) |ch| if (!std.ascii.isAlphanumeric(ch) and ch != '-' and ch != '_') return error.InvalidCapturePath;
        const path = try std.fmt.allocPrint(gpa, "{s}/{s}.inputs.bin", .{ directory, id });
        errdefer gpa.free(path);
        const buffer = try gpa.alloc(u16, buffer_rows * row_values);
        errdefer gpa.free(buffer);
        const file = try std.Io.Dir.cwd().createFile(io, path, .{ .exclusive = true });
        errdefer file.close(io);
        try file.writeStreamingAll(io, &raw);
        var encoded: [max_positions * 4]u8 = undefined;
        for (positions, 0..) |position, i| std.mem.writeInt(u32, encoded[i * 4 ..][0..4], position, .little);
        try file.writeStreamingAll(io, encoded[0 .. positions.len * 4]);
        return .{ .file = file, .io = io, .gpa = gpa, .path = path, .positions = positions, .buffer = buffer };
    }

    pub fn deinit(self: *Writer) void {
        self.file.close(self.io);
        self.gpa.free(self.path);
        self.gpa.free(self.buffer);
    }

    pub fn needs(self: *const Writer, start: usize, rows: usize) bool {
        return self.count < self.positions.len and self.positions[self.count] < start + rows;
    }

    fn flush(self: *Writer) !void {
        try self.file.writeStreamingAll(self.io, std.mem.sliceAsBytes(self.buffer[0 .. self.buffered * row_values]));
        self.buffered = 0;
    }

    pub fn append(self: *Writer, source: []const u16, stride: usize, start: usize, rows: usize) !void {
        if (rows == 0 or rows > stride) return error.InvalidCaptureShape;
        while (self.needs(start, rows)) {
            const position = self.positions[self.count];
            if (position < start) return error.MissingCapturePosition;
            try gather(self.buffer[self.buffered * row_values ..][0..row_values], source, stride, position - start);
            self.count += 1;
            self.buffered += 1;
            if (self.buffered == buffer_rows) try self.flush();
        }
    }

    pub fn finish(self: *Writer) !void {
        if (self.count != self.positions.len) return error.MissingCapturePosition;
        try self.flush();
        if ((try self.file.stat(self.io)).size != header_bytes + 4 * self.positions.len + 2 * row_values * self.positions.len)
            return error.InvalidCaptureLength;
    }
};

test "capture positions reject duplicates ordering and unforwarded final token" {
    try validate(3, 2, &.{ 0, 2, 3, 4 });
    try std.testing.expectError(error.InvalidCapturePosition, validate(3, 2, &.{ 0, 5 }));
    try std.testing.expectError(error.InvalidCapturePosition, validate(3, 2, &.{ 2, 2 }));
    try std.testing.expectError(error.InvalidCapturePosition, validate(3, 2, &.{ 3, 1 }));
    try std.testing.expectError(error.InvalidCaptureShape, validate(3, 256, &.{0}));
    try std.testing.expectError(error.InvalidCaptureShape, validate(2048, 1, &.{0}));
    try std.testing.expectError(error.InvalidCaptureShape, validate(3, 0, &.{}));
    try validate(3, 0, &.{ 0, 2 });
}

test "capture source header carries exact layout and original core zeros" {
    const manifest: [64]u8 = @splat('a');
    const raw = try header(&manifest, 3, 2, &.{ 0, 4 });
    try std.testing.expectEqualStrings("LNAC001\n", raw[0..8]);
    try std.testing.expectEqual(@as(u32, 2), std.mem.readInt(u32, raw[8..12], .little));
    try std.testing.expectEqual(@as(u32, 190464), std.mem.readInt(u32, raw[12..16], .little));
    try std.testing.expectEqualStrings(&manifest, raw[32..96]);
    const zeros: [64]u8 = @splat('0');
    try std.testing.expectEqualStrings(&zeros, raw[96..160]);
    try std.testing.expectError(error.InvalidCaptureSource, header("bad", 3, 0, &.{0}));
}

test "Q4 capture has distinct magic with otherwise identical original carrier layout" {
    const manifest: [64]u8 = @splat('a');
    const original = try header(&manifest, 3, 2, &.{ 0, 4 });
    const quantized = try headerForSource(.packed_q4, &manifest, 3, 2, &.{ 0, 4 });
    try std.testing.expectEqualStrings("LNACQ01\n", quantized[0..8]);
    try std.testing.expect(!std.mem.eql(u8, original[0..8], quantized[0..8]));
    try std.testing.expectEqualSlices(u8, original[8..], quantized[8..]);
    try std.testing.expectError(error.InvalidCaptureSource, headerForSource(.packed_q4, "bad", 3, 0, &.{0}));
    try std.testing.expectError(error.InvalidCapturePosition, headerForSource(.packed_q4, &manifest, 3, 0, &.{3}));
}

test "capture scope permits original BF16 and only matching packed32 versions and ranks" {
    try validateScope(false, 0, 0, 0);
    try validateScope(true, 2, 32, 0);
    try validateScope(true, 3, 32, 8);
    try std.testing.expectError(error.InvalidCaptureSource, validateScope(false, 2, 32, 0));
    try std.testing.expectError(error.InvalidCaptureSource, validateScope(false, 0, 0, 8));
    try std.testing.expectError(error.InvalidCaptureSource, validateScope(true, 0, 32, 0));
    try std.testing.expectError(error.InvalidCaptureSource, validateScope(true, 1, 32, 0));
    try std.testing.expectError(error.InvalidCaptureSource, validateScope(true, 2, 71, 0));
    try std.testing.expectError(error.InvalidCaptureSource, validateScope(true, 2, 32, 8));
    try std.testing.expectError(error.InvalidCaptureSource, validateScope(true, 3, 32, 0));
    try std.testing.expectError(error.InvalidCaptureSource, validateScope(true, 3, 32, 32));
    try std.testing.expectError(error.InvalidCaptureSource, validateScope(true, 3, 31, 8));
    try std.testing.expectError(error.InvalidCaptureSource, validateScope(true, 4, 32, 8));
}

test "all47 stream-major inputs gather exact selected rows including last gate" {
    const a = std.testing.allocator;
    const source = try a.alloc(u16, row_values * 2);
    defer a.free(source);
    const dest = try a.alloc(u16, row_values);
    defer a.free(dest);
    @memset(source, 0);
    for (0..layers) |layer| for ([_]bool{ false, true }) |down| {
        if (down and layer == layers - 1) continue;
        const begin = offset(layer, down);
        const width: usize = if (down) intermediate else hidden;
        for (0..2) |row| @memset(source[begin * 2 + row * width ..][0..width], @intCast(100 + layer * 4 + @as(usize, @intFromBool(down)) * 2 + row));
    };
    for (0..2) |row| {
        try gather(dest, source, 2, row);
        for (0..layers) |layer| for ([_]bool{ false, true }) |down| {
            if (down and layer == layers - 1) continue;
            const width: usize = if (down) intermediate else hidden;
            for (dest[offset(layer, down)..][0..width]) |v| try std.testing.expectEqual(@as(u16, @intCast(100 + layer * 4 + @as(usize, @intFromBool(down)) * 2 + row)), v);
        };
    }
    source[0] = 0x7f80;
    try std.testing.expectError(error.NonfiniteCapture, gather(dest, source, 2, 0));
    try std.testing.expectError(error.InvalidCaptureShape, gather(dest, source, 2, 2));
    try std.testing.expectEqual(row_values, offset(23, false) + hidden);
}

test "bounded writer flushes64 rows preserves header and rejects missing positions or overwrite" {
    const io = std.testing.io;
    const a = std.testing.allocator;
    var dir = std.testing.tmpDir(.{});
    defer dir.cleanup();
    const marker = try dir.dir.createFile(io, "marker", .{});
    marker.close(io);
    const marker_path = try dir.dir.realPathFileAlloc(io, "marker", a);
    defer a.free(marker_path);
    const manifest: [64]u8 = @splat('a');
    var positions: [65]u32 = undefined;
    for (&positions, 0..) |*position, i| position.* = @intCast(i);
    var writer = try Writer.init(io, a, std.fs.path.dirname(marker_path).?, "fixture", &manifest, 1, 64, &positions);
    defer writer.deinit();
    const values = try a.alloc(u16, row_values);
    defer a.free(values);
    @memset(values, 0x3f80);
    try std.testing.expectError(error.MissingCapturePosition, writer.append(values, 1, 1, 1));
    for (0..64) |start| try writer.append(values, 1, start, 1);
    try std.testing.expectEqual(@as(usize, 0), writer.buffered);
    try std.testing.expectEqual(@as(usize, 64), writer.count);
    try std.testing.expectError(error.MissingCapturePosition, writer.finish());
    try writer.append(values, 1, 64, 1);
    try writer.finish();
    try std.testing.expectError(error.PathAlreadyExists, Writer.init(io, a, std.fs.path.dirname(marker_path).?, "fixture", &manifest, 1, 64, &positions));
    const file = try std.Io.Dir.cwd().openFile(io, writer.path, .{});
    defer file.close(io);
    var raw: [header_bytes]u8 = undefined;
    try std.testing.expectEqual(header_bytes, try file.readPositionalAll(io, &raw, 0));
    try std.testing.expectEqualSlices(u8, &(try header(&manifest, 1, 64, &positions)), &raw);
    var tail: [2]u8 = undefined;
    const size = header_bytes + 4 * positions.len + 2 * row_values * positions.len;
    try std.testing.expectEqual(@as(usize, 2), try file.readPositionalAll(io, &tail, size - 2));
    try std.testing.expectEqual(@as(u16, 0x3f80), std.mem.readInt(u16, &tail, .little));
}
