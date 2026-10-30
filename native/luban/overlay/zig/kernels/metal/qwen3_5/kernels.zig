//! Generated affine kernels plus Luban BF16 kernels; fixed affine launch geometries.
pub const Kernel = struct { key: []const u8, function: [:0]const u8, source: []const u8, columns: usize = 0, threads: usize = 0 };
pub const all = [_]Kernel{
    .{ .key = "qwen35_research_trace_copy", .function = "qwen35_research_trace_copy", .source = @embedFile("trace.metal") },
    .{ .key = "qwen35_residual_project", .function = "qwen35_residual_project", .source = @embedFile("residual.metal") },
    .{ .key = "qwen35_residual_add", .function = "qwen35_residual_add", .source = @embedFile("residual.metal") },
    .{ .key = "qwen35_q8_projection", .function = "qwen35_q8_projection", .source = @embedFile("q8.metal") },
    .{ .key = "qwen35_q8_prefill_matrix", .function = "qwen35_q8_prefill_matrix", .source = @embedFile("q8.metal") },
    .{ .key = "qwen35_q8_embed", .function = "qwen35_q8_embed", .source = @embedFile("q8.metal") },
    .{ .key = "qwen35_greedy_partial", .function = "qwen35_greedy_partial", .source = @embedFile("greedy.metal") },
    .{ .key = "qwen35_greedy_finish", .function = "qwen35_greedy_finish", .source = @embedFile("greedy.metal") },
    .{ .key = "qwen35_bf16_prefill_matrix", .function = "qwen35_bf16_prefill_matrix", .source = @embedFile("bf16.metal") },
    .{ .key = "qwen35_bf16_projection", .function = "qwen35_bf16_projection", .source = @embedFile("bf16.metal") },
    .{ .key = "qwen35_bf16_embed", .function = "qwen35_bf16_embed", .source = @embedFile("bf16.metal") },
    .{ .key = "qwen35_bf16_norm", .function = "qwen35_bf16_norm", .source = @embedFile("bf16.metal") },
    .{ .key = "qwen35_bf16_head_norm", .function = "qwen35_bf16_head_norm", .source = @embedFile("bf16.metal") },
    .{ .key = "qwen35_bf16_mlp_act", .function = "qwen35_bf16_mlp_act", .source = @embedFile("bf16.metal") },
    .{ .key = "qwen35_bf16_queries", .function = "qwen35_bf16_queries", .source = @embedFile("bf16.metal") },
    .{ .key = "qwen35_bf16_keys", .function = "qwen35_bf16_keys", .source = @embedFile("bf16.metal") },
    .{ .key = "qwen35_bf16_gdn_pre", .function = "qwen35_bf16_gdn_pre", .source = @embedFile("bf16_delta.metal") },
    .{ .key = "qwen35_bf16_gdn_chain", .function = "qwen35_bf16_gdn_chain", .source = @embedFile("bf16_delta.metal") },
    .{ .key = "qwen35_bf16_gdn_post", .function = "qwen35_bf16_gdn_post", .source = @embedFile("bf16_delta.metal") },
    .{ .key = "qmm_6144_2048_g32", .function = "custom_kernel_qwen35_qmm_g32_6144_2048_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t", .source = @embedFile("qmm_g32_6144_2048.metal"), .columns = 32, .threads = 256 },
    .{ .key = "qmm_2048_6144_g32", .function = "custom_kernel_qwen35_qmm_g32_2048_6144_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t", .source = @embedFile("qmm_g32_2048_6144.metal"), .columns = 32, .threads = 256 },
    .{ .key = "qmm_6144_2048_g16", .function = "custom_kernel_qwen35_qmm_g16_6144_2048_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t", .source = @embedFile("qmm_g16_6144_2048.metal"), .columns = 32, .threads = 256 },
    .{ .key = "qmm_2048_6144_g16", .function = "custom_kernel_qwen35_qmm_g16_2048_6144_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t", .source = @embedFile("qmm_g16_2048_6144.metal"), .columns = 32, .threads = 256 },
    .{ .key = "qmm_6144_2048", .function = "custom_kernel_qwen35_qmm_6144_2048_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t", .source = @embedFile("qmm_6144_2048.metal"), .columns = 32, .threads = 256 },
    .{ .key = "qmm_2048_2048", .function = "custom_kernel_qwen35_qmm_2048_2048_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t", .source = @embedFile("qmm_2048_2048.metal"), .columns = 32, .threads = 256 },
    .{ .key = "qmm_16_2048", .function = "custom_kernel_qwen35_qmm_16_2048_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t", .source = @embedFile("qmm_16_2048.metal"), .columns = 16, .threads = 256 },
    .{ .key = "qmm_4096_2048", .function = "custom_kernel_qwen35_qmm_4096_2048_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t", .source = @embedFile("qmm_4096_2048.metal"), .columns = 32, .threads = 256 },
    .{ .key = "qmm_512_2048", .function = "custom_kernel_qwen35_qmm_512_2048_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t", .source = @embedFile("qmm_512_2048.metal"), .columns = 32, .threads = 256 },
    .{ .key = "qmm_2048_6144", .function = "custom_kernel_qwen35_qmm_2048_6144_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t", .source = @embedFile("qmm_2048_6144.metal"), .columns = 32, .threads = 256 },
    .{ .key = "qmm_248320_2048", .function = "custom_kernel_qwen35_qmm_248320_2048_bfloat16_t_uint32_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t", .source = @embedFile("qmm_248320_2048.metal"), .columns = 32, .threads = 256 },
    .{ .key = "norm", .function = "custom_kernel_qwen35_norm_bfloat16_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t_bfloat16_t", .source = @embedFile("norm.metal") },
    .{ .key = "norm_nores", .function = "custom_kernel_qwen35_norm_nores_bfloat16_t_bfloat16_t_floatc_bfloat16_t", .source = @embedFile("norm_nores.metal") },
    .{ .key = "gdn_pre", .function = "custom_kernel_qwen35_gdn_pre_bfloat16_t_bfloat16_t_bfloat16_t_int32_t_bfloat16_t_bfloat16_t_float_bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_float_bfloat16_t_bfloat16_t", .source = @embedFile("gdn_pre.metal") },
    .{ .key = "gdn_chain", .function = "custom_kernel_qwen35_gdn_chain__bfloat16_t_bfloat16_t_bfloat16_t_bfloat16_t_float_bfloat16_t_float_int32_tc_bfloat16_t_float", .source = @embedFile("gdn_chain.metal") },
    .{ .key = "gdn_post", .function = "custom_kernel_qwen35_gdn_post_bfloat16_t_bfloat16_t_bfloat16_t_floatc_bfloat16_t", .source = @embedFile("gdn_post.metal") },
    .{ .key = "mlp_act", .function = "custom_kernel_qwen35_mlp_act_bfloat16_t_bfloat16_t_bfloat16_t", .source = @embedFile("mlp_act.metal") },
    .{ .key = "attn_partial", .function = "custom_kernel_qwen35_attn_partial_bfloat16_t_bfloat16_t_bfloat16_t_floatc_int32_t_float_float_float", .source = @embedFile("attn_partial.metal") },
    .{ .key = "attn_merge", .function = "custom_kernel_qwen35_attn_merge_float_float_float_int32_t_bfloat16_t", .source = @embedFile("attn_merge.metal") },
};

// Test-only original projection source; inference does not compile this library.
pub const projection_reference = @embedFile("bf16_projection_reference.metal");

// Optional expert kernels are compiled only when an approved expert is loaded.
pub const expert_source = @embedFile("expert.metal");
