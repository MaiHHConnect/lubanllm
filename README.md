# 鲁班零号 / Luban Zero

**应用内测中 · 技术报告 Research Preview**

[官网](https://ai.maihh.net/) · [English](README_EN.md) · [内测状态](STATUS.md)

鲁班零号（luban0）是面向个人设备的持续学习AI原型，以约2B语言核心起步，探索本地推理、路由选择、用户授权资料学习和遗忘控制。

## 本地生成速度实测

2026-10-23，Apple M6、16GB 内存 Mac 上的一次本地回答，界面记录生成速度 **94.1 token/秒**，输入 **96 tokens**、输出 **30 tokens**，使用本地核心。

![M6、16GB Mac：一次本地回答的界面记录，输入96、输出30 tokens，生成速度94.1 token/秒](docs/images/m6-16gb-generation-94.1-20261023.png)

此数值为单次短回答的生成阶段（decode）速度，不包含输入预处理和首 token 等待；不是多轮平均或端到端速度，不能据此推断训练提速或所有任务的表现。

## Research Paper — Luban0

**Luban: Budget-Constrained Continual Learning with Disk-Resident Neural Experts**

v0.2 DOI: [10.5281/zenodo.23585073](https://doi.org/10.5281/zenodo.23585073)

[全部版本（自动指向最新版）](https://doi.org/10.5281/zenodo.23552844) · [原v0.1归档](https://doi.org/10.5281/zenodo.23552845)；旧版独立保留。

状态：**Technical Report / Preprint（未经同行评审）**。

[📄 Read Paper](paper/Luban_Technical_Report.md) · [PDF](paper/Luban_Technical_Report.pdf) · [中文补充](paper/Luban_Technical_Report_ZH.md) · [📊 Experimental Results](docs/EXPERIMENTS.md)

技术报告v0.2（2026-10-23）以英文为主，介绍当前MLX 4bit架构、专家学习、实际路由验收、净增采纳、失败记录及研究限制。状态为 **Technical Report — Research Preview / Work in Progress**，尚未投稿或经过同行评审；作者为vinson，所属机构为麦好火电商研究；终稿检查尚待确认，未经独立同行评审。报告及脱敏汇总采用[CC BY-NC-ND 4.0](LICENSE-REPORT.txt)。

当前日常学习冻结基座和已采纳核心，训练专家候选；“进化”入口同样使用该专家学习链。新增掌握多于抽查遗忘且通过质量检查，才保存采纳。自动分化多个独立专家和长期持续增长仍是研究方向。

小规模实测中，两个身份事实的开发验证题从2/6提高到6/6，但该轮历史保护池为空，不能据此宣称全面无遗忘或广泛泛化。详细条件和失败见[实验记录](docs/EXPERIMENTS.md)。

## 项目方向

- 本地客服和企业知识问答
- 个人知识与工作习惯适配
- 简单自动化任务
- 在有限资源下验证长期学习与专家扩展

以上为产品方向，不是已完成的能力承诺。完整当前架构见[ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 公开范围

本仓库公开项目介绍、技术报告和脱敏实验汇总。源码、个人数据、完整原始研究证据和内部流程资料不在本仓库公开；未授予项目源码或模型新的开源许可，上游组件保留各自许可。

完整安装包含基础模型、语音模型、元数据和初始专家，约3.05GB，建议下载前预留16GB。应用 **0.3.248 内测版**：[完整包（含模型）](https://ai.maihh.net/downloads/Luban-0.3.248-macos-arm64-full.zip) · [GitHub 小包与更新说明](https://github.com/MaiHHConnect/lubanllm/releases/tag/v0.3.248-beta.1)。完整包解压后移入“应用程序”，打开并同意使用协议，内置模型会自动安装并启动；GitHub 小包不含模型。需要 macOS 26 或更新版本、Apple Silicon；尚未经过 Apple 公证。

10月22日连续FAQ研究因现有合格独立验证题已覆盖而暂停，实际18/20轮、2800次Adam、4轮采纳、14轮质量拒绝；跨208/218/私有219阶段分别报告，不声称20轮完成或18项新能力，确定性失败与教师准备费用保留。详[实验汇总](docs/EXPERIMENTS.md)和[复现协议](docs/REPRODUCIBILITY.md)。

完整包包含基座、元数据、语音和初始专家；GitHub应用小包是单独的进阶选项。


另于2026-10-23完成20/20轮隔离推理与训练恢复测试：2960次一致性比较全部一致，80次实际更新含20次断点重放，0采纳、0教师；这不补齐此前18/20轮学习，也不证明新能力或提速。结果见[实验汇总](docs/EXPERIMENTS.md)。

## 0.3.248 磁盘前缀缓存（2026-10-28）

长提示的相同开头再次出现时，可从磁盘恢复，重启或卸载模型后也有效。合成提示单次实测：56,000 词元冷算 24.3 秒，重启后磁盘命中约 0.9 秒；只比较了 32 个贪心词元，不是质量评测。缓存只认精确相同的开头，开头有一个字符不同则整段失效；缓存位于 `~/Library/Application Support/Luban/prefix-cache`，仅本人可读写、不参与备份，上限 8 GiB，可手动删除，设置环境变量 `LUBAN_PREFIX_CACHE_DIR=off` 可关闭。2B 模型在带大量工具定义的 Agent 场景下仍可能复述上下文或误发工具调用。

## 0.3.246 API 兼容更新（2026-10-28）

增加 Claude Messages 协议与文本工具调用兼容，改善长对话内存准入、精确前缀缓存和生成中的状态读取。长历史主动发起新工具调用仍有失败，不代表完整 256K 上下文或所有 Agent 任务均已通过。官网现提供 0.3.248 含模型完整包（0.3.246 及更早的完整包已下线），GitHub 提供不含模型的小包；旧夸克镜像未更新，官网入口暂时停用。
