// SPDX-License-Identifier: Apache-2.0
//! CPU only: no Metal device, model checkpoint or trained weights loaded.
test {
    _ = @import("families/qwen3_5/direct_q4.zig");
}
