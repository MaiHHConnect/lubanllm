# 第三方声明

当前基座：Qwen/Qwen3.5-2B，Qwen 团队，Apache License 2.0；固定版本 `15852e8c16360a2fea060d615a32b45270f8a8fc`。[官方模型](https://huggingface.co/Qwen/Qwen3.5-2B)。

当前语音：Qwen/Qwen3-ASR-0.6B-hf，Qwen 团队，Apache License 2.0；原生支持版本参考 `7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c`，官方 ModelScope 固定运输记录及文件 SHA 随模型保存。[官方模型](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf)。

调优改造：保留原始 Qwen3.5 权重/许可不覆写，在语言模型末层插入零初始化并联神经分支，并保存实际已有末层 down_proj 核心张量的版本化替换。训练文案“麦好火调优”不意味着麦好火从零预训练了上游基座。ASR 转写后送入语言核心，没有声明联合训练音频权重。完整来源/参数计数/失败范围见 docs/LUBAN0_V1.md。

历史基座：Qwen/Qwen3-1.7B，Qwen 团队，Apache License 2.0。用户要求已删除原始旧权重；以下保留历史溯源。

- 官方模型与许可：https://huggingface.co/Qwen/Qwen3-1.7B
- 实际下载的官方仓库：https://modelscope.cn/models/Qwen/Qwen3-1.7B
- 原始 LICENSE 随模型保留，项目 LICENSE 使用同一 Apache-2.0 标准文本。
- 当前改造：保留 Qwen3 embedding、attention、RMSNorm、原 FFN、输出头；在指定 FFN 内加并联神经分支。改动原代码/权重时，继续在此记录。
- 下载校验以模型目录的 `upstream-file-manifest.json` 为准。ModelScope revision 与 Hugging Face revision 属于不同仓库，不能混称同一个提交。

Strata：仅研读 Niko1221/Strata 官方公开文档；没有复制其实现代码、模型权重或实验向量。该项目引擎 MIT 许可不等于它加载的每个模型也具备相同许可。

PyTorch、Transformers、safetensors 等依赖保留各自许可证。自有数据、公开语料、教师蒸馏输出的使用权分别记录，Apache-2.0 基座不能替它们授权。

MLX 原生本地试用（2026-10-18）：复用 Apple/ml-explore 的 mlx-swift 0.32.3、mlx-swift-lm 3.32.3，以及其固定 MLX C/C++ 依赖，原版权和 MIT 许可保存在 `native/mlx/licenses`；同时保留 swift-numerics、fmt、nlohmann-json、metal-cpp 和上游 ACKNOWLEDGMENTS。构建只选择所需官方库目标并使用固定本地依赖，不将这些实现称为自研。独立 GPU 资源来自官方 mlx-metal 0.32.2 分发，不打包 Python 扩展；同版本号不等于证明源码提交完全相同，实际兼容性按本轮原生数学与生成证据核验。构建回执记录确切源码依赖和资源绑定。

MLX 社区 Q4 模型：`mlx-community/Qwen3.5-2B-MLX-4bit`，固定版本 `93760be4f1f69842a46bc13dbdc0f19e291392a3`，沿用 Qwen3.5 Apache-2.0 来源许可。模型保留在外置盘，不混入应用包；Luban 身份核心与专家为独立版本文件，原始社区载体不覆写。试用不代表量化无损或正式质量采纳，来源与对照见 `research/COMMUNITY_Q4_20261017.md`。

用户名词库：yuanjie-ai/ChineseSensitiveVocabulary，固定提交c2547155fdbec7a2519a3f1eb4901be8f0c36041，MIT，Copyright (c) 2018 .尐；只使用words.txt首列词条并作规范化、长度筛选、去重和自有词补充。原许可保存在src/luban/name_words.LICENSE与website/name_words.LICENSE，完整来源和哈希见research/name_filter_sources.json。词库仅用于排行用户名遮罩，不用于模型训练。


排行榜评测仅使用TIGER-AI-Lab/MMLU-Pro固定官方缓存的24道原始题（不使用缓存模型答案或推理），保留MIT原MMLU署名和官方缓存仓库Apache-2.0许可于benchmarks/LICENSE；固定版本、SHA与选题规则见benchmarks/provenance.json。另24道变式为本项目自编，评测资料不进入训练；24题零样本子集不可称为完整官方MMLU-Pro成绩。


Luban原生BF16研究原型借鉴ashhart/TensorFold v1.0.4，固定提交d4fc1964d0637c33e09ddac18aec936f962bb646。独立加载、BF16扩展与构建补丁在native/luban；复用Metal桥接、Qwen状态和其他上游实现。原Apache-2.0 LICENSE/NOTICE、第三方MIT与MiaAI-Lab MIT署名原文保存在native/luban/licenses，上游原许可不变。底层复用不能称全部从零自研。


## 首页应用识别图标（2026-10-11）

Pi、OpenClaw、腾讯 CodeBuddy IDE、字节 TRAE IDE、Cherry Studio、Chatbox 的名称与图标归各自权利人。官网 logo/favicon 按用户要求下载，仅用于接入入口的产品识别，不表示合作或背书；下载点没有单独声明品牌素材再分发许可，不将软件开源许可冒充品牌图标授权。逐项来源、字节与 SHA256 见 [接入调研](docs/HOME_CONNECTIONS.md)。
