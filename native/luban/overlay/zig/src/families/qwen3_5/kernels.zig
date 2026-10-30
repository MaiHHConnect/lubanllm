//! Qwen pipelines retain fixed affine arithmetic and add full BF16 projections without tensor-unit requirements.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources").qwen35;

const layout_names = [_][:0]const u8{ "qwen35_embed", "qwen35_head_norm", "qwen35_queries", "qwen35_keys", "qwen35_attention_gate" };
const layout = @import("kernel_sources").qwen35_layout;
pub const total = sources.all.len + layout_names.len;

pub const Kernels = struct {
    pipelines: [total]mtl.Pipeline,

    pub fn get(self: *const Kernels, comptime key: []const u8) mtl.Pipeline {
        return self.pipelines[comptime index(key)];
    }

    pub fn byKey(self: *const Kernels, key: []const u8) ?mtl.Pipeline {
        inline for (sources.all, 0..) |s, i| if (std.mem.eql(u8, key, s.key)) return self.pipelines[i];
        for (layout_names, 0..) |s, i| if (std.mem.eql(u8, key, s)) return self.pipelines[sources.all.len + i];
        return null;
    }

    pub fn projection(self: *const Kernels, n: usize, k: usize, group_size: usize) ?Projection {
        @setEvalBranchQuota(10000);
        inline for (sources.all, 0..) |s, i| {
            if (s.columns > 0) {
                const group_match = if (group_size == 16) std.mem.endsWith(u8, s.key, "_g16") else if (group_size == 32) std.mem.endsWith(u8, s.key, "_g32") else (!std.mem.endsWith(u8, s.key, "_g32") and !std.mem.endsWith(u8, s.key, "_g16"));
                if (group_match) {
                    const nk = comptime dimensions(s.key);
                    if (n == nk[0] and k == nk[1]) return .{ .pipeline = self.pipelines[i], .columns = s.columns, .threads = s.threads };
                }
            }
        }
        return null;
    }

    pub fn deinit(self: *Kernels) void {
        for (self.pipelines) |p| p.deinit();
    }
};

pub const Projection = struct { pipeline: mtl.Pipeline, columns: usize, threads: usize };

fn dimensions(comptime key: []const u8) [2]usize {
    const tail = key[4..];
    const at = std.mem.indexOfScalar(u8, tail, '_').?;
    const rest = tail[at + 1 ..];
    const end = std.mem.indexOfScalar(u8, rest, '_') orelse rest.len;
    return .{ std.fmt.parseInt(usize, tail[0..at], 10) catch unreachable, std.fmt.parseInt(usize, rest[0..end], 10) catch unreachable };
}

fn index(comptime key: []const u8) usize {
    inline for (sources.all, 0..) |s, i| if (comptime std.mem.eql(u8, key, s.key)) return i;
    inline for (layout_names, 0..) |s, i| if (comptime std.mem.eql(u8, key, s)) return sources.all.len + i;
    @compileError("no Qwen3.5 kernel " ++ key);
}

pub fn load(device: mtl.Device) !Kernels {
    var out: Kernels = undefined;
    var loaded: usize = 0;
    errdefer for (out.pipelines[0..loaded]) |p| p.deinit();
    inline for (sources.all, 0..) |s, i| {
        const lib = try mtl.Library.fromSource(device, s.source, mtl.CompileOptions.mlx());
        defer lib.deinit();
        out.pipelines[i] = try mtl.Pipeline.init(device, lib, s.function, false);
        loaded += 1;
        if (out.pipelines[i].simdWidth() != 32) return error.UnsupportedQwenSIMDWidth;
    }
    const lib = try mtl.Library.fromSource(device, layout, mtl.CompileOptions.mlx());
    defer lib.deinit();
    for (layout_names, 0..) |name, i| {
        out.pipelines[sources.all.len + i] = try mtl.Pipeline.init(device, lib, name, false);
        loaded += 1;
    }
    return out;
}

pub const projection_reference = sources.projection_reference;
