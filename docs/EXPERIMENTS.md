# Experimental results / 实验结果

**Technical Report v0.1 - Research Preview.** Reported local observations, not independently reproduced or peer reviewed. / 本地实测汇总，未经独立复现或同行评审。

[Full method and limitations](../paper/Luban_Technical_Report.md) · [中文补充](../paper/Luban_Technical_Report_ZH.md) · [Machine-readable summary](experiments.json)

| ID / 编号 | Version / 应用版本 | Observation / 实测 | Limit / 边界 |
| --- | --- | --- | --- |
| E1 | 0.3.208 / build 216 | TRAIN 3/16→16/16; development VALID 2/6→6/6; 48 Adam, seed 42, width 32 | Two identity facts; same-family prompts; no historical protection examples / 两个身份事实、同类问法、保护池为空 |
| E2 | 0.3.208 / build 216 | Actual-route scored cases 25/34→28/34; validation gain 1, recorded forgetting 0 | Reuses an existing candidate; two of the three improvements were TRAIN / 复用已有候选；3条新增中2条属于训练题 |
| E3 | 0.3.217 / build 225 | Isolated installation: 3/3 short text probes; empty personal history; learning reports 0/16 TRAIN and 0/2 VALID | Same physical Mac; zero new Adam updates / 同一台Mac，未进行新训练 |

## E1 conditions and failures / E1条件与失败

Preparation produced 32 TRAIN and 12 VALID candidates. Selection used routing coverage without answer-correctness filtering, resulting in 16 TRAIN plus 4 new VALID; the application selected 2 previously correct basic-task VALID checks (case conversion and simple logic), yielding 6 total. VALID was consulted during early stopping/admission, so it is development validation. Recorded learning duration was 64.5786 seconds, measured up to report construction, excluding final publication/reload and not click-to-completion latency. Exact chip/RAM and peak memory were not established in this summary.

准备32条TRAIN及12条VALID候选，按路由覆盖筛选而非答案正确率筛题；最终16条TRAIN、4条身份新问法VALID和2条原本正确的基础题VALID（大小写转换、简单逻辑）。反复用于早停和采纳的VALID属于开发验证。64.5786秒为截至报告构造的记录时长，排除最终采纳重载，不是点击到完成总时间；本汇总不补猜具体芯片、内存容量或峰值。

Two preceding launches failed with database-lock errors; an automatic retry succeeded. A separate earlier 256-update attempt with no net gain was rejected. These failures are retained in the report. No new teacher API call was used to prepare this fact-learning run.

前两次启动发生数据库锁错误，后继自动重试成功；另一个更早的256步尝试净增0被拒绝。失败结果保留，本次事实学习准备没有新增教师API调用。

## E2 interpretation / E2解读

The historical forced-expert 23/24 result uses a different protocol and denominator from the actual-route 34 scored cases. Of 40 observed cases, 6 had no scoring target. Two routed validation questions remained unresolved. These previously exposed examples are not a new blinded test; the adopted candidate was not optimized further, although separate runtime training-qualification checks were rerun.

历史强制专家23/24与实际路由的34条评分题协议和分母不同。40条观察中6条无评分目标，仍有2条路由验证题未解决；材料此前已曝光，不属于新盲测，本次未进一步训练该候选，但另行重跑了含真实更新的运行时训练资格检查。

## E3 interpretation / E3解读

The installed application was denied access to development directories and existing user data. After explicit model startup, arithmetic and two identity probes succeeded. Inherited factory records were excluded from personal counters. This verifies inference and learning admission, not a new user's first successful training round. The private migration text roundtrip was a separate local check; its private archive is not published.

隔离启动禁止读取开发目录及旧用户数据；手动启动后算术和两个身份探针通过。出厂记录不计个人成长；该结果验证推理及学习入口，不代表新用户首轮训练成功。完整私有迁移的文本往返另有本地检查，私有归档不公开。

## E4 completed prefix / E4已完成前缀

A 20-round study was initiated on 22 October; the fixed 0.3.208/build 216 prefix contains 3 actual rounds, 408 Adam updates, 1 commit and 2 quality rejections. A preceding 0-update route rejection is excluded. These are paraphrases of 4 authorized product facts, using 16 TRAIN per round and actual development VALID counts 4/6/10, with historical protection 6/8/8. It is not 20 completed rounds or a transfer benchmark.

10月22日启动的20轮研究，固定0.3.208/build 216前缀实际完成3轮、408Adam、采纳1轮、质量拒绝2轮；早先0步路由拒绝不计。资料是4项授权产品事实的改写，各轮16TRAIN，实际开发VALID4/6/10、历史保护6/8/8；不是已完成20轮或迁移评测。

| Round / 轮 | Adam; wall seconds / 更新;墙钟秒 | Last candidate G/F / 末候选新增遗忘 | Outcome / 结果 |
| --- | --- | --- | --- |
| 1 | 24;53.1968 | 1/0; VALID 1/4→2/4 | Committed / 采纳 |
| 2 | 192;350.1305 | 2/1; VALID 2/6→4/6 | Rejected / 拒绝 |
| 3 | 192;358.8743 | 4/4; VALID 4/10→8/10 | Rejected / 拒绝 |

Wall time runs from submission through terminal lease release, excluding prior preparation. Engine timers 47.9278/347.1417/357.4094 seconds have a narrower scope. Rejected candidate scores do not represent persisted abilities. Historical forgetting includes sampled known records not mastered by the candidate, even if the immediate baseline already missed them.

墙钟从提交到终态且租约释放，不含准备；引擎47.9278/347.1417/357.4094秒口径更窄。拒绝候选成绩不计已保存能力；遗忘统计也包含本轮前测已经答错的历史已学题。

Baseline pair mismatches 0/26, 2/30, 2/34 and last-candidate mismatches 0/26, 4/30, 7/34 include core and expert routes. Exact-input/greedy/fresh-cache settings did not eliminate observed instability; the underlying cause remains unresolved. Mac16,10 / 16 GiB / macOS 26.4, seed 42, BF16 expert/FP32 Adam, learning rate 0.0002, protection weight 5; no peak-memory sampling. Preparation caused 2 unexpected incomplete teacher reviews / 1958 tokens, charge unknown; controlled training added 0 calls after persistent blocking. Continuing under 0.3.218 without adding data is a separate protocol phase.

双遍不一致涉及核心与专家，具体根因未明，不降低门槛。设备Mac16,10 / 16 GiB / macOS 26.4，seed 42、专家BF16/AdamFP32、学习率0.0002、保护权重5；未采峰值内存。准备阶段意外2次不完整教师批改/1958词元，费用未知；持久关闭后受控训练新增0。0.3.218复用已有资料续训另列阶段。

[Protocol and remaining evidence / 复现协议与待补证据](REPRODUCIBILITY.md)

## 0.3.218 continuation prefix / 218续训前缀

The existing pool remains 136 TRAIN/37 VALID, with no new questions. The first continuation round selected 16 TRAIN/10 VALID/8 protection records: 192 Adam, 344.3273 wall seconds, 342.4822 engine seconds, last-candidate G/F 1/2, rejected. The following preparation failed with a database lock after 25.1759 seconds and zero Adam; it is excluded from the round count. A normal retry entered training. The suspected transaction-arbitration mechanism is not a captured lock-holder diagnosis or a proven fix. This bounded observation brings the cumulative count to 4 actual rounds/600 updates/1 commit at its cutoff, with the 208 and 218 phases kept separate.

218现有资料136TRAIN/37VALID不新增，首轮16TRAIN/10VALID/8保护，192Adam后拒绝；344.3273秒墙钟与342.4822秒引擎时间分开。后继25.1759秒数据库锁失败为0步、不计轮，正常重试随后进入训练；仲裁机制仅疑似，未捕获当时持锁者，也未证明修复。此固定截点累计实际4轮/600Adam/1采纳，208与218阶段不混合成同条件实验。

## Availability / 可用性

The JSON contains sanitized aggregate observations only. No raw prompts, personal records, local filesystem paths, credentials, or internal workflow documents are included. Full independent reproduction remains unavailable because the complete implementation and raw evidence are restricted.

JSON仅包含脱敏汇总，不含原始题目、个人记录、本机路径、凭据或内部流程文档；完整实现和原始证据受限，尚不能独立完整复现。
