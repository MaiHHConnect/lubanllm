import Foundation
import MLX
import MLXNN
import MLXLLM

// Independent qualification entry only; never called by ordinary inference.
func checkExpert(_ args: [String]) throws {
    try require(args.count == 5, "Usage: --check-expert expert.safetensors expert-sha256 oracle.safetensors")
    let width = try expertWidth(args[2])
    let weights = try verifiedBF16(args[2], sha: args[3], shapes: ["gate.weight":[width,2048],"up.weight":[width,2048],"down.weight":[2048,width]])
    let oracle = try MLX.loadArrays(url: URL(fileURLWithPath: args[4]))
    guard let input = oracle["input"] else { throw WorkerFailure("Oracle input missing") }
    var reports = [[String:Any]]()
    var fp32Pass = true
    for (label, type) in [("fp32", DType.float32),("bf16", DType.bfloat16)] {
        let x = input.asType(type)
        let gate = matmul(x, weights["gate.weight"]!.asType(type).T).asType(type)
        let up = matmul(x, weights["up.weight"]!.asType(type).T).asType(type)
        let silu = (gate.asType(.float32) * sigmoid(gate.asType(.float32))).asType(type)
        let activation = (silu * up).asType(type)
        let output = matmul(activation, weights["down.weight"]!.asType(type).T).asType(type)
        for (stage, actual) in [("gate",gate),("up",up),("activation",activation),("output",output)] {
            guard let reference = oracle[label + "_" + stage] else { throw WorkerFailure("Oracle stage missing") }
            try require(actual.shape == reference.shape, "Oracle shape mismatch")
            let a = actual.asType(.float32), b = reference.asType(.float32)
            let difference = abs(a - b)
            let finite = all(isFinite(a)).item(Bool.self)
            let exact = all(a .== b).item(Bool.self)
            let maxError = max(difference).item(Float.self)
            let l2 = sqrt(sum(square(a - b)) / maximum(sum(square(b)), MLXArray(Float(1e-30)))).item(Float.self)
            let close = all(difference .<= (MLXArray(Float(1e-5)) + abs(b) * Float(1e-4))).item(Bool.self)
            if label == "fp32" { fp32Pass = fp32Pass && close && finite }
            reports.append(["dtype":label,"stage":stage,"finite":finite,"exact":exact,"max_abs_error":maxError,"relative_l2":l2,"fp32_tolerance_pass":close,"nonzero":any(a .!= MLXArray(Float(0))).item(Bool.self)])
        }
    }
    Stream.defaultStream.synchronize()
    try emit(["event":"expert_math", "schema":"luban-mlx-expert-math-v1", "expert_file_sha256":args[3],"width":width,"fp32_pass":fp32Pass,"stages":reports,"bf16_pass_assumed":false])
}

// Exercise the real upstream Qwen module-array structure on CPU, without loading model files.
func checkAdapterStructure() throws {
    try Device.withDefaultDevice(.cpu) {
        let text: [String:Any] = ["hidden_size":8,"num_hidden_layers":24,"intermediate_size":16,
            "num_attention_heads":1,"num_key_value_heads":1,"head_dim":8,"vocab_size":16,
            "linear_num_value_heads":1,"linear_num_key_heads":1,"linear_key_head_dim":4,
            "linear_value_head_dim":4,"linear_conv_kernel_dim":4,"tie_word_embeddings":true]
        let data = try JSONSerialization.data(withJSONObject: ["model_type":"qwen3_5","text_config":text])
        let model = Qwen35Model(try JSONDecoder().decode(Qwen35Configuration.self, from: data))
        var previous = [String: Module]()
        model.visit { key, module in if key.hasSuffix(".mlp") { previous[key] = module } }
        try require(previous.count == 24, "CPU Qwen fixture must have 24 MLP modules")
        let core = MLXArray.zeros([8,16], dtype: .bfloat16)
        var oldPathRejected = false
        do {
            try model.update(modules: ModuleChildren.unflattened(["language_model.model.layers.23.mlp.down_proj": Linear(weight:core)]), verify:.noUnusedKeys)
        } catch { oldPathRejected = true }
        try require(oldPathRejected, "Expected sparse-array update regression was not reproduced")
        let expert = ["gate.weight":MLXArray.zeros([2,8],dtype:.bfloat16), "up.weight":MLXArray.zeros([2,8],dtype:.bfloat16), "down.weight":MLXArray.zeros([8,2],dtype:.bfloat16)]
        try installAdapterModules(model, core:core, expert:expert)
        try require(model.modules().contains { $0.training }, "Training-mode regression fixture missing")
        model.train(false)
        try require(model.modules().allSatisfy { !$0.training }, "Inference mode did not cover the full model and new expert")
        var after = [String: Module]()
        model.visit { key, module in if key.hasSuffix(".mlp") { after[key] = module } }
        for index in 0..<23 {
            let key = "language_model.model.layers.\(index).mlp"
            try require(after[key] === previous[key], "Adapter changed another layer")
        }
        guard let attached = after["language_model.model.layers.23.mlp"] as? ExpertMLP else { throw WorkerFailure("Expert not installed in real Qwen layer") }
        try require(attached.base === previous["language_model.model.layers.23.mlp"], "Expert lost original MLP")
        var down: Linear?
        attached.base.visit { key, module in if key == "down_proj" { down = module as? Linear } }
        try require(down?.weight.shape == [8,16] && down?.weight.dtype == .bfloat16, "Core not installed")
        var strictRejected = false
        do { try attached.base.update(modules: ModuleChildren.unflattened(["missing_proj":Linear(weight:core)]), verify:.noUnusedKeys) }
        catch { strictRejected = true }
        try require(strictRejected, "Strict unknown-module validation was bypassed")
        try emit(["event":"adapter_structure", "passed":true,"device":"cpu","layers":24,"unchanged_other_layers":23,"old_path_rejected":true,"strict_rejection":true,"all_modules_eval":true])
    }
}
