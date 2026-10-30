// SPDX-License-Identifier: Apache-2.0
//! Test BF16 projection at real input dimensions using exactly representable data.
const std = @import("std");
const mtl = @import("metal");
const q = @import("luban_core").qwen35;

fn bf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @intCast((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

fn project(columns: u32, queue: mtl.Queue, pipeline: mtl.Pipeline, x: mtl.Buffer, w: mtl.Buffer, out: mtl.Buffer, rows: u32, n: u32, k: u32, offset: usize) !void {
    const command = queue.commandBuffer();
    const encoder = command.compute(.serial);
    encoder.setPipeline(pipeline);
    encoder.setBuffer(x, offset, 0);
    encoder.setValue([3]u32{ rows, k, n }, 1);
    encoder.setBuffer(w, 0, 2);
    encoder.setBuffer(out, 0, 3);
    const tile_rows: u32 = if (columns == 64) 16 else 8;
    encoder.dispatchThreads(mtl.Size.of(256 * ((n + columns - 1) / columns), (rows + tile_rows - 1) / tile_rows, 1), mtl.Size.of(256, 1, 1));
    encoder.end();
    command.commit();
    command.wait();
    if (command.failure() != null) return error.GpuFailed;
}

pub fn main(_: std.process.Init) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var kernels = try q.kernels.load(device);
    defer kernels.deinit();
    const pipeline = kernels.byKey("qwen35_bf16_projection").?;
    const matrix_pipeline = kernels.byKey("qwen35_bf16_prefill_matrix").?;
    const reference_lib = try mtl.Library.fromSource(device, q.kernels.projection_reference, mtl.CompileOptions.mlx());
    defer reference_lib.deinit();
    const reference_pipeline = try mtl.Pipeline.init(device, reference_lib, "qwen35_bf16_projection", false);
    defer reference_pipeline.deinit();
    var checked: usize = 0;
    for ([_]usize{ 2048, 6144 }) |k| {
        for ([_]usize{ 16, 17, 512, 2048 }) |n| {
            const x = try device.buffer(17 * k * 2, mtl.ResourceOptions.shared);
            defer x.deinit();
            const w = try device.buffer(n * k * 2, mtl.ResourceOptions.shared);
            defer w.deinit();
            const out = try device.buffer(17 * n * 2, mtl.ResourceOptions.shared);
            defer out.deinit();
            const single = try device.buffer(n * 2, mtl.ResourceOptions.shared);
            defer single.deinit();
            for (x.slice(u16, 17 * k), 0..) |*value, index| value.* = bf16(@as(f32, @floatFromInt(@as(i32, @intCast(index % 7)) - 3)) / 32);
            for (w.slice(u16, n * k), 0..) |*value, index| value.* = bf16(@as(f32, @floatFromInt(@as(i32, @intCast((index / k + index % k) % 5)) - 2)) / 16);
            try project(64, queue, matrix_pipeline, x, w, out, 17, @intCast(n), @intCast(k), 0);
            for (0..17) |row| {
                try project(8, queue, pipeline, x, w, single, 1, @intCast(n), @intCast(k), row * k * 2);
                try std.testing.expectEqualSlices(u16, out.slice(u16, 17 * n)[row * n ..][0..n], single.slice(u16, n));
                for (0..n) |column| {
                    var exact: f32 = 0;
                    for (0..k) |at| {
                        const xv = @as(f32, @floatFromInt(@as(i32, @intCast((row * k + at) % 7)) - 3)) / 32;
                        const wv = @as(f32, @floatFromInt(@as(i32, @intCast((column + at) % 5)) - 2)) / 16;
                        exact += xv * wv;
                    }
                    try std.testing.expectEqual(bf16(exact), out.slice(u16, 17 * n)[row * n + column]);
                }
            }
            // Nontrivial BF16 patterns verify the optimized single-row kernel
            // against the actual original shader, including a partial output group.
            for (x.slice(u16, k), 0..) |*value, index| {
                const f = @as(f32, @floatFromInt(@as(i32, @intCast((index * 491 + 73) % 10007)) - 5003));
                value.* = bf16(f / 1573.0);
            }
            for (w.slice(u16, n * k), 0..) |*value, index| {
                const f = @as(f32, @floatFromInt(@as(i32, @intCast((index * 257 + 19) % 8191)) - 4095));
                value.* = bf16(f / 39217.0);
            }
            try project(8, queue, pipeline, x, w, single, 1, @intCast(n), @intCast(k), 0);
            try project(8, queue, reference_pipeline, x, w, out, 1, @intCast(n), @intCast(k), 0);
            try std.testing.expectEqualSlices(u16, out.slice(u16, n), single.slice(u16, n));
            checked += 1;
        }
    }
    std.debug.print("Luban BF16 projection: {d} real-dimension cases, exact matrix CPU sums, serial/batch bits and original-shader nontrivial BF16 decode bits passed\n", .{checked});
}
