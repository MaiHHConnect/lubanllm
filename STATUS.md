# 内测状态 / Beta status

**状态：内测中 / Status: Private beta**

鲁班零号当前探索一条受控的本地学习路径：前置路由头在本地模型和远程模型之间选择推理路径；本地回答出现错误或需要改进时，系统记录学习材料并尝试临时专家训练；材料累积并通过资格检查后，可以启动 MoE 混合专家训练，增加独立专家。

Luban Zero is exploring a controlled local-learning path: a front routing head selects between local and remote inference; incorrect or improvable local answers are recorded as learning material for possible temporary expert training; after material accumulates and passes qualification checks, MoE training can add independent experts.

临时学习按预期收益、训练成本和遗忘风险进行取舍，不要求每一次尝试都成功。持续积累、参数增长和“越用越有料”是产品方向，实际采纳仍取决于质量、保留和失败回滚检查。

Temporary learning weighs expected benefit, training cost, and forgetting risk; every attempt is not required to succeed. Continued accumulation, parameter growth, and becoming more useful with use are product directions, while adoption still depends on quality, retention, and failure rollback checks.

## 当前内测实测 / Current beta measurement

- 约 2B 模型在 Mac mini M4 上约 **58 词元/秒 / 58 tokens per second**
- 实际速度随模型版本、上下文、量化方式和设备状态变化
- Actual speed varies with model version, context, quantization, and device state

## 适合场景 / Suitable scenarios

- 低成本客服模型建立 / Low-cost customer-service model building
- 企业知识库 / Enterprise knowledge bases
- 不复杂的自动化 / Non-complex automation
- 陪伴个人成长的模型 / A model for personal growth and companionship
- 按长期使用逐步增加参数和专长 / Gradually adding parameters and specialties through long-term use

## 目前可以了解 / What is currently public

- [项目官网 / Project website](https://ai.maihh.net/)
- [双语项目介绍 / Bilingual introduction](README.md)

## 目前不公开 / What remains restricted

- 源代码和内部构建文件 / Source code and internal build files
- 模型权重和个人数据 / Model weights and personal data
- 内部研究证据、原始实验资料和流程脑图 / Internal research evidence, raw experiment materials, and workflow maps

如需参与内测，请通过项目维护者提供的受限渠道联系。For beta access, contact the project maintainer through the restricted channel provided to invited testers.
