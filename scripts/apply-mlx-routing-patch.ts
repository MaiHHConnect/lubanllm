import { readFile, writeFile } from "node:fs/promises";

/** Apply the small public prefix API needed by the router to the pinned MLXLMCommon source. */
const path = process.argv[2];
if (!path) throw new Error("Qwen35.swift path is required");
let source = await readFile(path, "utf8");
if (!source.includes("public func prefixForward(_ inputs: MLXArray, layerCount: Int = 4)")) {
  const innerNeedle = "        return applyFinalNorm ? norm(hiddenStates) : hiddenStates\n    }\n\n    // MARK: - Whole-step decode schedule";
  const innerInsert = `        return applyFinalNorm ? norm(hiddenStates) : hiddenStates
    }

    /// Run a fixed prefix of the quantized text backbone for routing features.
    public func prefixForward(_ inputs: MLXArray, layerCount: Int = 4) -> MLXArray {
        precondition(layerCount > 0 && layerCount <= layers.count)
        var hiddenStates = embedTokens(inputs)
        let cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
        let faMask = createAttentionMask(h: hiddenStates, cache: nil)
        let ssmMask = createSSMMask(h: hiddenStates, cache: nil)
        for (i, layer) in layers.prefix(layerCount).enumerated() {
            let mask = layer.isLinear ? ssmMask : nil
            let attnMask = layer.isLinear
                ? MLXFast.ScaledDotProductAttentionMaskMode.none : faMask
            hiddenStates = layer(
                hiddenStates, attentionMask: attnMask, ssmMask: mask, cache: cacheArray[i])
        }
        return hiddenStates
    }

    // MARK: - Whole-step decode schedule`;
  if (!source.includes(innerNeedle)) throw new Error("Qwen35 inner forward anchor not found");
  source = source.replace(innerNeedle, innerInsert);
  const textNeedle = "    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {\n        var out = model(inputs, cache: cache)";
  const textInsert = "    public func prefixForward(_ inputs: MLXArray, layerCount: Int = 4) -> MLXArray {\n        model.prefixForward(inputs, layerCount: layerCount)\n    }\n\n" + textNeedle;
  if (!source.includes(textNeedle)) throw new Error("Qwen35 text model anchor not found");
  source = source.replace(textNeedle, textInsert);
  const modelNeedle = "    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {\n        languageModel(inputs, cache: cache)";
  const modelInsert = "    /// Hidden states after a quantized backbone prefix for routing features.\n    public func prefixForward(_ inputs: MLXArray, layerCount: Int = 4) -> MLXArray {\n        languageModel.prefixForward(inputs, layerCount: layerCount)\n    }\n\n" + modelNeedle;
  if (!source.includes(modelNeedle)) throw new Error("Qwen35 model anchor not found");
  source = source.replace(modelNeedle, modelInsert);
  await writeFile(path, source);
}
