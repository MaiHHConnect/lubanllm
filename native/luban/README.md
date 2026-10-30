# Luban 原生 BF16 文本引擎

独立 macOS Metal 文本推理原型，基于 TensorFold `v1.0.4`，固定上游提交 `d4fc1964d0637c33e09ddac18aec936f962bb646`。直接读取原始 HF BF16 文本权重，保留原始 F32 `A_log` 和 gated norm 参数原始字节与 dtype；可显式加载带已关闭合格证据绑定的 BF16 down23 sidecar。常驻 `luban-worker` 接入正式文本路由，旧 `luban-native` 保留为离线研究入口；训练继续使用 PyTorch，已学专家走兼容图，视觉入口仍未接入。源码交付和构建成功不代表模型等价或性能验收通过。

`overlay/` 保存相对上游目录的最终源码覆盖文件。构建脚本只依赖 Python 标准库、git 和调用者已经安装的 Zig **0.17.0**，不安装依赖、不下载模型、不运行 GPU 或推理。`kernel-check` 在此命令中仅编译并安装检查程序。

```sh
.venv/bin/python native/luban/build.py \
  --checkout '/Volumes/移动数据库/lubanllm/experiments/luban-native-build-new' \
  --zig '/Volumes/移动数据库/lubanllm/toolchains/zig-0.17.0/zig'
```

替换为实际 Zig 可执行文件路径及全新构建目录。两个路径必须位于已挂载的外置卷；脚本拒绝任何已有 checkout（包括干净目录），不会复用原始 TensorFold checkout。它校验工具链及固定上游提交，复制 overlay 后执行：

```sh
zig build --build-file luban-build.zig install kernel-check worker -j1 \
  --cache-dir .luban-zig-cache
```

已有外置缓存时可追加 `--source-cache '/Volumes/移动数据库/lubanllm/experiments/已有缓存/tensorfold-source'`。脚本只读校验缓存的 HTTPS 上游 origin 和 `v1.0.4` 标签提交，再以 `git clone --no-hardlinks --branch v1.0.4` 复制独立 Git 对象及官方提交；支持浅克隆，不复制缓存工作区改动、不联网 fetch、不修改缓存。新 checkout 的 origin 设置为固定上游，确认 HEAD 正确、工作区干净且无 alternates 依赖后才复制 overlay。

全局缓存也限定在新 checkout 的 `.luban-zig-global-cache`。输出包括 `zig-out/bin/luban-native`、`zig-out/bin/luban-kernel-check` 和常驻 `zig-out/bin/luban-worker`；运行、模型来源批准、官方权重及候选证据校验另由实验 runner 显式执行。

TensorFold 原始 Apache-2.0 许可及第三方 MIT 归属保留在`native/luban/licenses/` 与构建 checkout 的 `LICENSE`、`NOTICE`、`LICENSES/`，覆盖源码继续保留各自的 SPDX 与原作者归属；不改变第三方许可。

当前结果见 [性能研究](../../docs/NATIVE_BF16_PERFORMANCE.md)：16行矩阵预填充五轮CN/EN首段132/138ms，48+112题有限能力保持通过；连续生成相对旧原生没有明确大幅提升，这份历史结果当时未替换正式引擎；后续产品接入与独立验收见 [正式接入](../../docs/NATIVE_PRODUCT_ENGINE.md)。[首版失败](../../docs/NATIVE_BF16_LOADER.md)及原始数据保留。使用 `run_native.py` 运行独立入口和继承现有模型锁，`prepare_cases.py` 对齐官方文本模板，`repeat.py` 保留五轮逐条数据。训练后 sidecar 的来源审核与导出由 `export_core.py` 执行，当前固定为闭合首轮450候选。

正式本机打包：构建命令加 `--install-product`，将worker与绑定全部overlay SHA的构建回执复制到项目 `build/native/`，然后 `bun run app:build`。应用只打包明确的worker与回执，不依赖实验路径。worker采用有界JSONL、独立取消pipe、每请求独立神经状态，支持0–2温度与top_p0.95；当前工程上下文上限32768，输出2048。正式已采纳核心经现有journal/文件/tensor校验后直接加载，不将研究sidecar当正式采纳。
