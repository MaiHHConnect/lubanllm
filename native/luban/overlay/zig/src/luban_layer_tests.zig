// SPDX-License-Identifier: Apache-2.0
test {
    _ = @import("families/qwen3_5/execution.zig");
    _ = @import("families/qwen3_5/layer_paging.zig");
    _ = @import("families/qwen3_5/weights.zig");
}
