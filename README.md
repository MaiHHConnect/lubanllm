# 鲁班零号 / Luban Zero

> **内测中 / Private beta**
>
> 本页面公开项目介绍，不公开源代码、模型权重或内部研究资料。> This page introduces the project publicly; source code, model weights, and internal research materials remain restricted.

官网 / Website: [ai.maihh.net](https://ai.maihh.net/)

## 中文简介

鲁班零号（luban0）是一个以约 2B 语言核心起步的本地 AI 原型，面向个人设备上的低成本推理、快速路由和受控学习。

回答前，前置路由头会根据任务选择本地模型或远程模型进行推理处理。对于本地模型回答错误或需要改进的内容，系统会记录为学习材料并尝试临时专家训练。临时学习采用“盈亏比”思路：根据预期收益、训练成本和遗忘风险决定是否采纳，不强制每一次临时训练都立即学会；在有新材料、资源和授权时持续积累。

当学习材料积累到足够规模并通过独立资格检查后，可以启动 MoE 混合专家训练，为模型增加独立专家，让模型在持续使用和验证中逐步扩展可用能力。每次增长都需要经过质量、保留和失败回滚检查，实际效果以测试结果为准。

当前内测中，在 Mac mini M4 上约 2B 模型的实测生成速度达到 **58 词元/秒**；速度会随模型版本、上下文长度、量化方式和设备状态变化。

## 适合的场景

- 低成本客服模型建立
- 企业内部知识库问答
- 不复杂的自动化任务
- 陪伴个人成长的本地模型
- 根据长期使用逐步增加模型参数和专长

这些是产品方向和内测目标，不代表所有能力已经完成或经过公开评测。“越用越有料”描述的是受控学习与专家扩展的设计方向，不是无限能力增长的保证。

## English

Luban Zero (luban0) is a local AI prototype starting from an approximately 2B language core. It focuses on low-cost inference, fast routing, and controlled learning on personal devices.

Before an answer is generated, a front routing head selects either a local model or a remote model for inference. When the local model gives an incorrect or improvable answer, the system records it as learning material and may run temporary expert training. This process follows a return-on-investment idea: expected benefit, training cost, and forgetting risk are considered before adoption. A temporary update is not forced to succeed every time; material can continue to accumulate when new data, resources, and authorization are available.

After enough learning material has accumulated and passed independent qualification checks, MoE mixture-of-experts training can be started to add independent experts. The model is intended to expand its usable capabilities gradually through continued use and validation. Each update remains subject to quality, retention, and failure rollback checks; actual results are determined by tests.

In the current beta, an approximately 2B model measured about **58 tokens per second** on a Mac mini M4. Speed varies with the model version, context length, quantization, and device state.

## Suitable scenarios

- Building a low-cost customer-service model
- Enterprise knowledge-base question answering
- Non-complex automation tasks
- A local model for personal growth and companionship
- Gradually adding parameters and specialties through long-term use

These are product directions and beta goals, not claims that every capability is complete or publicly benchmarked. “More useful with use” describes the controlled learning and expert-expansion direction; it is not a guarantee of unlimited capability growth.

## 公开范围 / Public scope

公开仓库只提供项目介绍、内测状态和官网入口。源代码、模型权重、内部研究证据、流程脑图和实验资料保留在受限项目环境中；本仓库不授予开源许可。

This public repository contains only the project introduction, beta status, and website link. Source code, model weights, internal research evidence, workflow maps, and experiment materials remain in a restricted project environment; this repository grants no open-source license.

更多状态说明见 [STATUS.md](STATUS.md)。
