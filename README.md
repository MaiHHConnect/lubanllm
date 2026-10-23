# 鲁班零号 / Luban Zero

**应用内测中 · 技术报告 Research Preview**

[官网](https://ai.maihh.net/) · [English](README_EN.md) · [内测状态](STATUS.md)

鲁班零号（luban0）是面向个人设备的持续学习AI原型，以约2B语言核心起步，探索本地推理、路由选择、用户授权资料学习和遗忘控制。

## Research Paper — Luban0

**Luban: Budget-Constrained Continual Learning with Disk-Resident Neural Experts**

DOI: [10.5281/zenodo.23552845](https://doi.org/10.5281/zenodo.23552845)

DOI 对应归档预印本；本仓库文件包含后续实验更新。

状态：**Technical Report / Preprint（未经同行评审）**。

DOI: [10.5281/zenodo.23552845](https://doi.org/10.5281/zenodo.23552845)

DOI 对应归档预印本；本仓库文件包含后续实验更新。

状态：**Technical Report / Preprint（未经同行评审）**。

[📄 Read Paper](paper/Luban_Technical_Report.md) · [PDF](paper/Luban_Technical_Report.pdf) · [中文补充](paper/Luban_Technical_Report_ZH.md) · [📊 Experimental Results](docs/EXPERIMENTS.md)

技术报告v0.1（2026-10-22）以英文为主，介绍当前MLX 4bit架构、专家学习、实际路由验收、净增采纳、失败记录及研究限制。状态为 **Technical Report — Research Preview / Work in Progress**，尚未投稿或经过同行评审；作者为vinson，所属机构为麦好火电商研究；终稿检查尚待确认，未经独立同行评审。报告及脱敏汇总采用[CC BY-NC-ND 4.0](LICENSE-REPORT.txt)。

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

完整安装包含基础模型、语音模型、元数据和初始专家，约3.62GB，建议下载前预留16GB。应用0.3.218内测下载：[完整包（含模型）](https://ai.maihh.net/downloads/Luban-0.3.218-macos-arm64-full.zip) · [下载 macOS 内测版](https://github.com/MaiHHConnect/lubanllm/releases/tag/v0.3.218-beta.1) · [夸克网盘](https://pan.quark.cn/s/0c64deea0cd9)。需要 macOS 26 或更新版本、Apple Silicon；尚未经过 Apple 公证。

10月22日连续FAQ研究因现有合格独立验证题已覆盖而暂停，实际18/20轮、2800次Adam、4轮采纳、14轮质量拒绝；跨208/218/私有219阶段分别报告，不声称20轮完成或18项新能力，确定性失败与教师准备费用保留。详[实验汇总](docs/EXPERIMENTS.md)和[复现协议](docs/REPRODUCIBILITY.md)。

完整包包含基座、元数据、语音和初始专家；GitHub应用小包是单独的进阶选项。


另于2026-10-23完成20/20轮隔离推理与训练恢复测试：2960次一致性比较全部一致，80次实际更新含20次断点重放，0采纳、0教师；这不补齐此前18/20轮学习，也不证明新能力或提速。结果见[实验汇总](docs/EXPERIMENTS.md)。
