# Reproduction protocol / 复现协议

Technical Report v0.1, 22 October 2026. Research preview; exact independent reproduction is not yet available because the implementation and complete raw material are restricted. This document records measured conditions and the missing evidence; it is not a completed scientific evaluation.

研究预览，未完成独立完整复现：实现及完整原始材料受限。本文补充真实条件和证据缺口，不代表科学评估完成。

[Main report](../paper/Luban_Technical_Report.md) · [中文补充](../paper/Luban_Technical_Report_ZH.md) · [Aggregates](experiments.json)

## E4 fixed prefix / E4固定前缀

1. Use application 0.3.208/build 216, the existing Qwen3.5-2B-derived MLX Q4 installation, its accepted frozen core, expert and router. The three reported rounds precede the 0.3.218 retry change. A new factory installation is a different starting state.
2. Conditions: Mac16,10, 16 GiB, macOS 26.4; seed 42, learning rate 0.0002, preservation weight 5. The expert uses BF16 weights and FP32 Adam state/update arithmetic. Preserve the original source binding and runtime qualification; do not transfer acceptance evidence between binaries or starting experts.
3. Materials: four assistant-authored, user-authorized facts (minimum macOS 26, Apple Silicon, application filename Luban.app and brand 麦好火). Initially add 16 TRAIN/4 VALID per batch, keeping roles fixed. Normal selection produced actual VALID counts 4/6/10 and protection 6/8/8. Shared fact families mean paraphrase adaptation, not transfer or an untouched test.
4. Keep teacher address/model empty and teaching modalities disabled. Record the session's teacher-call counter before and after. Preparation had 2 unintended incomplete reviews / 1958 provider-reported tokens, monetary charge unknown; later controlled learning added 0. A configured limit of 0 means unlimited, not disabled.
5. Trigger one normal explicit learning request and wait for its terminal state **and** released model lease. Each candidate allows 64 Adam updates with checks every 8; subsequent candidates can make a round 192 updates. Do not replace a pending checkpoint, force a route, insert a reference into student input, or accept a candidate solely from loss reduction.
6. Observe question-only generation twice under the actual route: thinking off, temperature 0, seed 42, fresh per-request cache. New study records allow 192 output tokens; some earlier protection records allow 64. The 24-token preparation probes are a different protocol. Score both passes, exact output/input-token agreement, validation loss and sampled historical retention.
7. Count only rounds with actual Adam updates toward the 20-round target. Keep committed, quality-rejected and zero-update admission attempts separate. Record submission-to-terminal/lease-release wall time and engine report time separately. Preserve every rejected result; last-candidate gains are not saved abilities.

中文：固定原0.3.208及原采纳来源；每批16训练/4验证，实际验证会累积，不能混写。教师0上限是不限，必须验证不可调用并记录计数。每轮正常请求、等待租约释放；64步是候选上限，192是三候选总量。双遍真实路由、输入与确定性、质量、防遗忘、保存重载门均保留，失败成绩不计采纳。资料生成和路由准备不包含在逐轮提交计时内，峰值内存未测。

### Admission interpretation / 采纳口径

G counts previously incorrect development-VALID records mastered by the candidate. F counts sampled historical knowledge not mastered by that candidate, even if the immediate baseline already missed it. Acceptance requires G−F > 0 plus every quality/persistence check. Thus historical forgetting and immediate before-to-after regressions are different denominators.

Round 2 baseline repeated observations differed on 2/30 records (core); its last candidate differed on 4/30 (1 core, 3 expert). Round 3 baseline differed on 2/34 (accepted expert); its last candidate differed on 7/34 (2 core, 5 expert). Exact inputs, greedy decoding and fresh caches do not establish stability. The underlying cause remains undiagnosed; this is an open reliability issue, not evidence that the frozen weights were optimized.

中文：F不是仅统计本轮从对变错；第2、3轮涉及原核心及专家路径的双遍不一致。底层原因仍待独立诊断，不能称已经修复，也不能把所有变化归为训练导致遗忘。

### Existing-data continuation / 现有资料续训

From 0.3.218, explicit learning may retry a fully quality-checked rejected expert run that performed positive actual updates. Automatic scheduling retains deduplication; zero-update admission rejections, successful commits and interrupted jobs do not qualify. The 20-round controller continued using existing approved materials without adding questions, then stopped at18 actual rounds as documented below. Version, material selection and retry-rule changes are a separate protocol phase. Exact repetition with the same seed is not an independent seed replication. The first continuation performed 192 updates but was rejected (G/F 1/2); the next preparation encountered a zero-update database-lock failure. Preserve failed submissions separately and count no Adam that did not run. A normal retry is an operational observation, not proof that the locking cause is fixed.

0.3.218只开放明确手动重试已实际训练的质量拒绝轮；不降低采纳规则。按用户要求复用已有资料，不再为了改变签名新增问法；后继另记版本及协议，重复同种子不算独立重复。首个续训192步后G/F为1/2而拒绝，后继0步数据库锁失败单独保留，不计学习轮；正常重试进入训练不代表根因已修复。

### Stopped at18/20 / 停在18/20

The controller stopped after18 actual rounds/2800 updates/4 commits. Its unchanged136TRAIN/37VALID pool had zero uncovered eligible daily VALID after the final commit; an additional submission was rejected before job creation. Do not reuse already-covered questions as new evidence, reclassify excluded material or add data merely to meet a round count. Private219 database coordination is a distinct phase; the full phase ledger is in [experiments.json](experiments.json).

最后采纳后未覆盖合格VALID为0，原门要求至少2条，故停在18/20并暂停跟进；不重标已掌握/排除材料，不为凑轮数加料。原37条库存不等于37条当前可用新验证题；协议与工作量另见汇总。

## Evidence required before scientific completion / 科学结论仍需的证据

| Item / 项目 | Present evidence / 当前证据 | Required completion / 待补 |
| --- | --- | --- |
| Learning and retention / 学习与保持 | Small same-fact development sets; bounded sampled history / 小型同事实开发集与抽查 | Untouched final test, task-family split, task-by-time matrix / 最终盲测、事实族隔离、任务时间矩阵 |
| Replication / 重复 | One seed and one physical Mac / 单种子单机 | Multiple seeds, uncertainty, independent device/environment / 多种子、不确定性、独立环境 |
| Comparison / 对照 | No matched-budget system comparison / 无匹配预算系统对照 | Frozen base, fixed expert and growing system at equal compute/teacher budget / 冻结基座、固定专家、增长系统等预算对照 |
| Reliability / 可靠性 | Repeated-output instability detected; candidates rejected / 检出不一致且拒绝 | Controlled diagnosis on the exact installed runtime / 对准确安装运行时独立受控诊断 |
| Resources / 资源 | Defined partial timers; no peak sampling / 明确计时口径、未采峰值 | All-process physical peak, device allocation, throughput, storage and teacher charge separately / 分别测峰值、设备内存、速度、存储和教师费用 |
| Reproduction / 复现 | Public method and sanitized aggregates; private implementation / 方法及脱敏汇总公开、实现受限 | Releasable authorized data, exact revisions, executable full protocol / 可发布授权数据、准确版本及完整可执行协议 |
| Review / 审阅 | Author vinson; review pending / 作者vinson、待审阅 | Actual final manuscript check and independent scientific review / 实际终稿检查与独立科学审阅 |

No status flag, public beta release or DOI substitutes for these experiments and review. No claim of an optimal material threshold, universal zero forgetting, automatic many-expert specialization or unlimited improvement is supported.

内测开放、DOI或状态字段都不能代替实测和审阅；不把实现门槛称最优数量，不宣称全面无遗忘、自动多专家长期效果或无限成长。


## E6 fixed-source reliability protocol / E6固定来源可靠性协议

Twenty independent candidate repetitions on 23 October used the exposed 34-record observation set and a separate two-TRAIN/one-protection lifecycle set. Each repetition ran 64 ABBA + 16 continuous + 68 full-pair checked generations, two warmups, and eight lifecycle generations; all observations were compared against the first repetition's corresponding full-record outputs. The lifecycle executed four Adam updates including a checkpoint replay, ending at step 3, and checked seven gates plus a fresh default-product reload before selecting learning mode. Observation width 32 and lifecycle width 64 are distinct sources. Fixed seed42, temperature0, fresh per-request caches and non-purged filesystem caches are part of the protocol. The candidate disables compilation through training ACKs rather than an environment override. The immutable worker SHA, exact hardware/runtime and counts are in E6 of [experiments.json](experiments.json).

20个独立会话仅复测同一候选，不采纳；每轮148次比较＋2预热＋8资格生成，4次实际Adam含1次重放、step3。32宽观察和64宽生命周期来源分开，保留七门及默认产品新进程重载。固定seed42/贪心/新请求新缓存、文件缓存不清空，通过训练ACK切换非编译模式，未使用环境强制开关；原18/20学习实验仍暂停。

The private raw records, implementation and checkpoints remain restricted. This protocol describes the measurement but does not make the full experiment independently executable. Multiple seeds, long sequences, exact installed-product qualification, a matched compiled baseline and broad semantic/retention tests remain outstanding. No new source-code license or DOI-archive revision is implied.

原始资料、实现及检查点保持受限；公开协议不等于已可独立完整执行。多种子、长序列、准确安装产品资格、匹配编译基线和广泛语义/遗忘评估仍待完成，不授予新源码许可、不替换DOI归档。
