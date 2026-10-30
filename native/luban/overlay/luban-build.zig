// SPDX-License-Identifier: Apache-2.0
//! Build only Luban's BF16 language loader, using Metal's runtime compiler.
const std = @import("std");
const builtin = @import("builtin");

comptime {
    if (!std.mem.eql(u8, builtin.zig_version_string, "0.17.0"))
        @compileError("luban-native requires verified Zig 0.17.0");
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    if (target.result.os.tag != .macos) @panic("BF16 native prototype is qualified for Metal only");
    const optimize = std.builtin.OptimizeMode.ReleaseSafe;
    const metal = b.createModule(.{ .root_source_file = b.path("zig/src/metal/metal.zig"), .target = target, .optimize = optimize, .link_libc = true });
    metal.linkFramework("Metal", .{});
    metal.linkFramework("Foundation", .{});
    metal.linkFramework("IOSurface", .{});
    metal.linkSystemLibrary("objc", .{});
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const sources = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig") });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/luban_core.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "lanes", .module = lanes }, .{ .name = "kernel_sources", .module = sources } } });
    const executable = b.addExecutable(.{ .name = "luban-native", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/luban_native.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "luban_core", .module = core } },
    }) });
    b.installArtifact(executable);
    const worker = b.addExecutable(.{ .name = "luban-worker", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/luban_worker.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "luban_core", .module = core } },
    }) });
    b.step("worker", "Build the resident product inference worker").dependOn(&b.addInstallArtifact(worker, .{}).step);
    const worker_tests = b.addTest(.{ .root_module = worker.root_module });
    b.step("worker-tests", "Run request and cache validation without loading the GPU").dependOn(&b.addRunArtifact(worker_tests).step);
    const layer_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/luban_layer_tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal } },
    }), .filters = &.{ "paging", "disk slot", "default runs every layer", "early exit", "reject zero depth", "live stream" } });
    b.step("layer-tests", "Validate fixed layer policy and disk paging without loading the GPU").dependOn(&b.addRunArtifact(layer_tests).step);
    const packed_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/qwen3_5/packed.zig"),
        .target = target,
        .optimize = optimize,
    }) });
    b.step("packed-tests", "Validate source-bound Q4 payloads without loading the GPU").dependOn(&b.addRunArtifact(packed_tests).step);
    const direct_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/luban_direct_tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    }) });
    b.step("direct-tests", "Validate complete streaming Q4 files on CPU without loading the GPU").dependOn(&b.addRunArtifact(direct_tests).step);
    const capture_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/families/qwen3_5/capture_inputs.zig"),
        .target = target,
        .optimize = optimize,
    }) });
    b.step("capture-tests", "Validate native activation layout and positions without loading the GPU").dependOn(&b.addRunArtifact(capture_tests).step);
    const sequential_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/luban_sequential_tests.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "kernel_sources", .module = sources }, .{ .name = "lanes", .module = lanes } },
    }) });
    b.step("sequential-tests", "Validate sequential phases and stage carriers without GPU").dependOn(&b.addRunArtifact(sequential_tests).step);
    const check = b.addExecutable(.{ .name = "luban-kernel-check", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/luban_kernel_check.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = metal }, .{ .name = "luban_core", .module = core } },
    }) });
    b.step("kernel-check", "Build BF16 real-dimension checks without running the GPU").dependOn(&b.addInstallArtifact(check, .{}).step);
}
