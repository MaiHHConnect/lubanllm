# 鲁班零号 / Luban Zero

**应用内测中 · 技术报告 Research Preview**

[官网](https://ai.maihh.net/) · [English](README_EN.md) · [内测状态](STATUS.md)

鲁班零号（luban0）是面向个人设备的持续学习AI原型，以约2B语言核心起步，探索本地推理、路由选择、用户授权资料学习和遗忘控制。

## Research Paper — Luban0

**Luban: Budget-Constrained Continual Learning with Disk-Resident Neural Experts**

DOI: [10.5281/zenodo.23552845](https://doi.org/10.5281/zenodo.23552845)

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

应用 0.3.217 已开放内测下载：[下载 macOS 内测版](https://github.com/MaiHHConnect/lubanllm/releases/tag/v0.3.217-beta.1) · [夸克网盘](https://pan.quark.cn/s/0e9c14c97c78)。需要 macOS 26 或更新版本、Apple Silicon；尚未经过 Apple 公证。
