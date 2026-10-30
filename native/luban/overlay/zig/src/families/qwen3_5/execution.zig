//! A fixed reduced-depth graph for a whole stream, never a token-local exit.
//! This preserves cache consistency, not the original model's answer quality.
const std = @import("std");
const c = @import("config.zig");

pub const all_layers: u32 = (1 << c.layers) - 1;

pub const Policy = struct {
    skip_mask: u32 = 0,
    exit_after: usize = c.layers,

    pub fn mask(self: Policy) !u32 {
        if (self.exit_after == 0 or self.exit_after > c.layers) return error.InvalidExitDepth;
        const prefix = (@as(u32, 1) << @intCast(self.exit_after)) - 1;
        // The first layer initializes the delayed residual. Skips beyond the
        // chosen exit are rejected instead of silently ignoring configuration.
        if (self.skip_mask & 1 != 0 or self.skip_mask & ~prefix != 0) return error.InvalidLayerSkipMask;
        return prefix & ~self.skip_mask;
    }
};

pub fn executes(mask: u32, layer: usize) bool {
    return layer < c.layers and mask & (@as(u32, 1) << @intCast(layer)) != 0;
}

/// A skipped attention/GDN layer has no history for those tokens. Re-enabling
/// it therefore requires replay from an empty cache, not just a new policy.
pub fn validateCache(mask: u32, previous: u32, rows: usize) !void {
    if (rows != 0 and mask != previous) return error.LayerPolicyChangedWithLiveCache;
}

test "default runs every layer and reduced graph keeps original indices" {
    try std.testing.expectEqual(all_layers, try (Policy{}).mask());
    const p = try (Policy{ .skip_mask = 1 << 22 }).mask();
    try std.testing.expect(executes(p, 0));
    try std.testing.expect(!executes(p, 22));
    try std.testing.expect(executes(p, 23));
    try std.testing.expectEqual(@as(u32, 23), @popCount(p));
}

test "early exit and intermediate skips compose without executing tail" {
    const p = try (Policy{ .skip_mask = (1 << 4) | (1 << 5), .exit_after = 23 }).mask();
    try std.testing.expect(!executes(p, 4));
    try std.testing.expect(!executes(p, 5));
    try std.testing.expect(executes(p, 22));
    try std.testing.expect(!executes(p, 23));
    try std.testing.expect(!executes(p, 32));
    try std.testing.expectEqual(@as(u32, 21), @popCount(p));
}

test "reject zero depth first-layer skip and bits outside chosen depth" {
    try std.testing.expectError(error.InvalidExitDepth, (Policy{ .exit_after = 0 }).mask());
    try std.testing.expectError(error.InvalidExitDepth, (Policy{ .exit_after = 25 }).mask());
    try std.testing.expectError(error.InvalidLayerSkipMask, (Policy{ .skip_mask = 1 }).mask());
    try std.testing.expectError(error.InvalidLayerSkipMask, (Policy{ .skip_mask = 1 << 24 }).mask());
    try std.testing.expectError(error.InvalidLayerSkipMask, (Policy{ .skip_mask = 1 << 23, .exit_after = 23 }).mask());
}

test "live stream cannot change skip graph or exit depth before replay" {
    const skip = try (Policy{ .skip_mask = 1 << 22 }).mask();
    const exit = try (Policy{ .exit_after = 23 }).mask();
    try validateCache(skip, skip, 128);
    try validateCache(skip, all_layers, 0);
    try std.testing.expectError(error.LayerPolicyChangedWithLiveCache, validateCache(all_layers, skip, 1));
    try std.testing.expectError(error.LayerPolicyChangedWithLiveCache, validateCache(skip, all_layers, 1));
    try std.testing.expectError(error.LayerPolicyChangedWithLiveCache, validateCache(exit, skip, 1));
}
