# Experimental results / 实验结果

**Technical Report v0.1 - Research Preview.** Reported local observations, not independently reproduced or peer reviewed. / 本地实测汇总，未经独立复现或同行评审。

[Full method and limitations](../paper/Luban_Technical_Report.md) · [中文补充](../paper/Luban_Technical_Report_ZH.md) · [Machine-readable summary](experiments.json)

| ID / 编号 | Version / 应用版本 | Observation / 实测 | Limit / 边界 |
| --- | --- | --- | --- |
| E1 | 0.3.208 / build216 | TRAIN 3/16→16/16; development VALID 2/6→6/6; 48 Adam, seed42, width32 | Two identity facts; same-family prompts; no historical protection examples / 两个身份事实、同类问法、保护池为空 |
| E2 | 0.3.208 / build216 | Actual-route scored cases 25/34→28/34; validation gain1, recorded forgetting0 | Reuses an existing candidate; two of the three improvements were TRAIN / 复用已有候选；3条新增中2条属于训练题 |
| E3 | 0.3.217 / build225 | Isolated installation: 3/3 short text probes; empty personal history; learning reports0/16 TRAIN and0/2 VALID | Same physical Mac; zero new Adam updates / 同一台Mac，未进行新训练 |

## E1 conditions and failures / E1条件与失败

Preparation produced32 TRAIN and12 VALID candidates. Selection used routing coverage without answer-correctness filtering, resulting in16 TRAIN plus4 new VALID; the application selected2 previously correct basic-task VALID checks (case conversion and simple logic), yielding6 total. VALID was consulted during early stopping/admission, so it is development validation. Recorded learning duration was64.5786 seconds, measured up to report construction, excluding final publication/reload and not click-to-completion latency. Exact chip/RAM and peak memory were not established in this summary.

准备32条TRAIN及12条VALID候选，按路由覆盖筛选而非答案正确率筛题；最终16条TRAIN、4条身份新问法VALID和2条原本正确的基础题VALID（大小写转换、简单逻辑）。反复用于早停和采纳的VALID属于开发验证。64.5786秒为截至报告构造的记录时长，排除最终采纳重载，不是点击到完成总时间；本汇总不补猜具体芯片、内存容量或峰值。

Two preceding launches failed with database-lock errors; an automatic retry succeeded. A separate earlier256-update attempt with no net gain was rejected. These failures are retained in the report. No new teacher API call was used to prepare this fact-learning run.

前两次启动发生数据库锁错误，后继自动重试成功；另一个更早的256步尝试净增0被拒绝。失败结果保留，本次事实学习准备没有新增教师API调用。

## E2 interpretation / E2解读

The historical forced-expert23/24 result uses a different protocol and denominator from the actual-route34 scored cases. Of40 observed cases,6 had no scoring target. Two routed validation questions remained unresolved. These previously exposed examples are not a new blinded test; the adopted candidate was not optimized further, although separate runtime training-qualification checks were rerun.

历史强制专家23/24与实际路由的34条评分题协议和分母不同。40条观察中6条无评分目标，仍有2条路由验证题未解决；材料此前已曝光，不属于新盲测，本次未进一步训练该候选，但另行重跑了含真实更新的运行时训练资格检查。

## E3 interpretation / E3解读

The installed application was denied access to development directories and existing user data. After explicit model startup, arithmetic and two identity probes succeeded. Inherited factory records were excluded from personal counters. This verifies inference and learning admission, not a new user's first successful training round. The private migration text roundtrip was a separate local check; its private archive is not published.

隔离启动禁止读取开发目录及旧用户数据；手动启动后算术和两个身份探针通过。出厂记录不计个人成长；该结果验证推理及学习入口，不代表新用户首轮训练成功。完整私有迁移的文本往返另有本地检查，私有归档不公开。

## Availability / 可用性

The JSON contains sanitized aggregate observations only. No raw prompts, personal records, local filesystem paths, credentials, or internal workflow documents are included. Full independent reproduction remains unavailable because the complete implementation and raw evidence are restricted.

JSON仅包含脱敏汇总，不含原始题目、个人记录、本机路径、凭据或内部流程文档；完整实现和原始证据受限，尚不能独立完整复现。
