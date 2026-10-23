# Luban: Budget-Constrained Continual Learning with Disk-Resident Neural Experts

**Technical Report v0.2 - Research Preview / Work in Progress**

**Author: vinson | Affiliation: 麦好火电商研究**

23 October 2026; original v0.1 dated 22 October 2026 | Implementation reference: application 0.3.218 (build 226); experiment versions are recorded separately

The author is responsible for the report. Completion of the final manuscript review has not been confirmed; no independent review is claimed. This report has not been submitted to a venue or peer reviewed. AI assistance was used to organize the report and implement parts of the prototype.

[PDF reading edition](Luban_Technical_Report.pdf) | [中文补充](Luban_Technical_Report_ZH.md) | [Experimental results](../docs/EXPERIMENTS.md)

### Abstract

Luban Zero investigates controlled learning on personal Apple Silicon devices using an approximately 2B language core, a shallow routing head, optional neural experts, and an external teacher. The current application uses a Qwen3.5-2B-derived MLX 4-bit model with an accepted core modification and a learned expert. Daily learning freezes the base/core, trains an expert candidate, evaluates generated answers through the actual local route, and accepts an update only when measured gains exceed sampled forgetting and additional quality checks pass. A small identity-fact experiment improved development validation answers from 2/6 to 6/6 after 48 Adam updates; its four newly correct validation prompts concerned those facts, while two existing basic-task checks were already correct. Its empty historical protection set prevents a claim of preserved general capability. A subsequent three-round FAQ prefix performed 408 Adam updates: one round was committed and two were rejected, including repeated-output instability despite improved loss. This report describes the implemented system, bounded observations, failures, and outstanding evaluation. It does not establish broad transfer, an optimal training budget, or unlimited self-improvement.

## Problem

Personal assistants need to adapt to user-approved knowledge while working within finite memory, compute time, storage, and teacher-call budgets. We study whether small neural updates can improve a local model's answers without unacceptable losses on previously learned material.

The engineering goal is a complete loop: gather authorized material, train a candidate, test it in the same routing context used by the application, and either persist or reject it. The longer-term research question is whether this loop outperforms fixed-capacity alternatives across many tasks under matched resource budgets. The latter question remains open.

A stored checkpoint, an active expert, and a resident tensor are different quantities. Keeping inactive weights on disk does not by itself demonstrate effective continual learning or remove finite-resource limits.

## Method

### Current architecture

The current language/vision foundation is derived from Qwen3.5-2B [1], replacing the Qwen3-1.7B configuration used in early prototypes. Product name `luban0` does not change upstream provenance. MLX [2] provides the current 4-bit inference path on Apple Silicon; quantization describes the base representation, not every learned tensor or optimizer state. An earlier accepted core modification is loaded with the base and remains frozen during the daily expert updates discussed here.

A shallow head uses features from the first four decoder layers to choose a supported local path. The local generation path can use the core alone or the currently accepted expert, subject to routing and source checks. Teacher participation is governed by the selected answer mode, configured provider, and authorization. Routing uncertainty or unavailable teacher eligibility must not be interpreted as evidence that a teacher was called.

The application supports visual inputs through its model/processor integration. Speech recognition uses a separate Qwen3-ASR-0.6B model [3]. Neither capability was measured in the text-learning experiment reported below. Removing the original full-precision weight shard did not remove the separately retained tokenizer, configuration, quantized weights, or speech model.

The current product manages accepted core, expert, and router artifacts on disk and loads compatible components for execution. The daily MLX path updates an active expert candidate; it does not automatically split all learned knowledge into an expanding set of independent specialists. The user-facing **Evolution** action currently enters this same MLX expert-learning path. Historical full-precision core-training experiments are separate and are not evidence that this action rewrites the current core.

### Learning and adoption

Material has separate training and validation roles. The present operational minimum is 16 training records and at least 2 eligible validation records, with at most 16 validation records selected for a round. These are implementation settings, not empirically optimal quantities. An eligible historical pool supplies up to 16 rotating protection examples. Absence of such a pool is reported explicitly.

The current candidate search permits up to 64 Adam updates per candidate and reevaluates every 8 updates. Candidate attempts are bounded; a round can exceed 64 updates across candidates. The first candidate passing admission can be retained. Frozen base/core weights and trainable expert parameters must not be conflated.

Closed-book generation gives the student the question without supplying its reference answer or a teacher answer at evaluation time. Candidate and baseline are compared under controlled inputs with the real local router, and repeated deterministic observations must agree. Matching uses the declared scoring rule for each record. A correct answer in a forced-expert test is insufficient if ordinary routing selects another path and fails.

Let **G** be previously incorrect development-validation records answered correctly after the update, and **F** be failed records from the sampled previously learned validation pool. The current net-gain rule requires **G - F > 0**. F counts sampled historical records not mastered by the candidate, including a historical record already incorrect in the immediate baseline; it is not exclusively a before-correct/after-incorrect transition count. Training-set successes do not contribute to G. Additional checks include validation-loss improvement greater than 0.5%, deterministic output agreement, comparable inputs, finite values, and supported routing. This is a count-based acceptance rule, not a calibrated economic return-on-investment model.

The rule can accept partial learning and some sampled forgetting when gains exceed losses. Unlearned material remains available for later consideration; validation records are not silently converted into training records. A failed check retains the previous accepted state. From application 0.3.218, an explicit learning request may retry an actually trained, quality-rejected expert cohort without creating new examples; automatic scheduling remains deduplicated. Zero-update admission failures, successful commits and interrupted jobs are not reopened through this exception. Successful candidates must survive save/reload checks before their accepted state is committed. Source and engine bindings prevent an old validation record from automatically qualifying a different runtime.

Validation is consulted repeatedly during candidate selection. It is therefore **development validation**, not an untouched final test. Repeated answer agreement detects some instability but is not independent statistical replication or proof of semantic understanding.

### Lifecycle and first installation

Opening the application leaves the model stopped. Automatic learning requires a started, available model, automatic learning enabled, eligible material, and a valid source. Training shares the model resource with inference; requests may report that the model is busy.

A fresh installation imports a small factory package containing the accepted core modification, expert, router, and required acceptance records. These are starting capabilities, not a new user's personal learning achievements. Personal learning counts and last-learning timestamps start empty. Existing user data is not overwritten by first-install setup. Private full-model migration is separate from the public factory package.

## Experiments

We report three observations from 21 October and a fixed three-round prefix from 22 October 2026. They are local engineering studies, not a preregistered benchmark. Raw internal records support the summaries but are not included in this public repository. Sanitized aggregates are available in [the experiment summary](../docs/EXPERIMENTS.md) and [machine-readable data](../docs/experiments.json).

**E1 - Expert learning on two identity facts.** Candidate prompts concerned the project's model name and brand. The preparation stage generated 32 training and 12 validation candidates. Routing coverage, without examining correctness, selected 16 training and 4 validation prompts; the application added 2 existing, previously correct basic-task checks (case conversion and simple logic). The four new validation prompts were paraphrases of the trained identity facts, not disjoint task families. Questions were assistant-prepared from user-authorized facts; no new paid teacher calls were made for this run.

The installed application was 0.3.208 (build 216), using the MLX 4-bit path on the project's Apple Silicon machine. The run used seed 42 and a width-32 expert candidate. It performed 48 Adam updates. Recorded duration was 64.58 seconds from baseline preparation through training and intermediate verification up to report construction. It excludes final publication/reload and is not the full interval from the user's click. The record does not establish peak memory. The historical protection set was empty.

**E2 - Route-aware adoption of a previously trained candidate.** An existing 64-update candidate was reevaluated without further optimization of the adopted candidate. Separate runtime-qualification checks, including actual training checks, were rerun. Forty cases were observed, of which 34 had scoring targets. Actual routing was used for before/after comparison; six unscored cases were diagnostic only. Direct expert output and routed output were recorded separately. This was a narrow migration/admission exercise, not a fresh blinded evaluation or a new candidate-training run.

**E3 - Factory installation and private migration.** Application 0.3.217 was launched in an isolated home with access to the development environment and existing application data denied. A fresh factory installation was explicitly started and tested on three short text prompts. A separate private migration roundtrip verified imported artifacts and two identity answers. Both studies were conducted on the existing machine, not on a second physical Mac. Neither performed new learning updates.

**E4 - Sequential FAQ learning, completed prefix.** A 20-round local study was initiated with four assistant-authored, user-authorized product facts: minimum macOS version, chip family, application filename and brand. This report freezes the first three actual rounds under application 0.3.208/build 216: 16 training records per round, with 4, 6 and 10 development-validation records selected from the accumulating pool, and 6, 8 and 8 historical protection records. Different question forms share the same fact families; this is neither task-family transfer nor 20 new abilities. One earlier route-admission rejection performed zero Adam updates and is excluded.

The machine identifier was Mac16,10 with 16 GiB RAM, macOS 26.4. The base/core remained frozen; the expert used BF16 weights with FP32 Adam moments and update arithmetic, learning rate 0.0002, seed 42 and preservation weight 5. Decoding used temperature 0, thinking off, seed 42 and a new cache per request. New study records allowed 192 output tokens; some old protection records used 64. Preparation route probes used a different 24-token limit. No dedicated all-process physical-memory or Metal peak sampler was used.

Round 1 committed after 24 updates, with development validation 1/4 to 2/4, all 6 historical checks retained and G/F = 1/0. Rounds 2 and 3 each used 192 updates across three candidates (width-32 reuse, width-64 reuse and width-32 fresh, each up to 64), and were rejected. Their last-candidate G/F counts were 2/1 and 4/4; these candidate answers were not committed capabilities. The submission-to-terminal-and-lease-release times were 53.1968, 350.1305 and 358.8743 seconds; engine report times were 47.9278, 347.1417 and 357.4094 seconds. Both exclude earlier material preparation; the engine timer can exclude final publication/reload.

Preparation probes unexpectedly triggered two existing background teacher reviews (1,958 provider-reported tokens, both incomplete; monetary charge unknown). The configuration was then persistently disabled; the controlled learning rounds added zero teacher calls. The whole preparation-plus-training study cannot be described as having zero teacher cost.

The user subsequently requested reuse of the existing data. Continuation on 0.3.218 changes the application and material-retry protocol and is recorded as a separate phase; it is not merged into this fixed-runtime prefix. The 20-round target was not complete at this report's observation cutoff. In the first continuation round on 0.3.218, the unchanged existing pool contained 136 TRAIN and 37 VALID records; normal selection again used 16 TRAIN, 10 VALID and 8 historical checks. It performed 192 updates in 344.3273 wall seconds (342.4822 engine seconds), but was rejected: last-candidate G/F was 1/2. The next submission failed during database preparation after 25.1759 seconds with zero updates and is not counted. A normal retry subsequently entered training. A same-process connection/transaction-arbitration issue is suspected; the original lock holder was not captured, and no fix is claimed by this observation.

### E5: existing-data continuation and stopping condition

The finite study stopped at **18 of 20 actual rounds**, with **2,800 Adam updates, 4 commits and 14 quality rejections**. Counts below include the fixed E4 prefix; changes of application and protocol prevent treating the total as a single unchanged-condition experiment.

| Phase | Actual rounds | Adam updates | Committed / rejected |
| --- | --- | --- | --- |
| 0.3.208 / build 216 | 3 | 408 | 1 / 2 |
| 0.3.218 / build 226 | 3 | 576 | 0 / 3 |
| Private 0.3.219 / build 227 | 12 | 1816 | 3 / 9 |
| Total work | 18 | 2800 | 4 / 14 |

After the first three rounds, the stored pool stayed at 136 TRAIN and 37 VALID; no new questions were added. Existing material rules admit 80 TRAIN and 18 daily VALID records. Following the final 56-update commit, all 18 eligible VALID records were covered by the accepted report, leaving 29 uncovered eligible TRAIN and zero uncovered eligible VALID. The next request was rejected before job creation because at least two independent VALID records are required. The other 19 stored VALID records are excluded by existing daily-material rules; they were not relabelled. Two earlier 218 database-preparation failures also performed zero updates. These three continuation failures do not count as training rounds; an earlier zero-update route rejection is retained separately.

Private 219 reused the main database connection and awaited existing RPC leases asynchronously for short ledger stages. Relevant CPU tests passed 98/98; its 12 real rounds completed without another observed database-lock failure. The original failed-round lock holder was not captured, and repeated-output instability remains unresolved. This coordination repair does not establish improved model mathematics or remove quality checks. Public installer 218 remains a separate frozen artifact.

The controller exited, no native job or model lease remained, automatic learning stayed off and controlled teacher calls remained zero. Follow-up was paused rather than inventing two extra rounds or adding material against the existing-data instruction. The 18-record knowledge pool includes prior and previously correct records; it is not 18 newly learned abilities or a simultaneous fresh retest of every retained record. Rotating historical checks and repeated same-family development validation do not establish broad transfer, universal retention or scientific completion. This v0.2 edition adds these results while retaining the original v0.1 archive as a separate historical version.

### E6: isolated inference and training-lifecycle reliability (23 October 2026)

This follow-up tests an inference-engine candidate, separately from E5's stopped learning study. Source-switch diagnostics twice reproduced greedy-output differences on the compiled path; disabling compilation in the same original worker made the shared observations agree. The candidate disables MLX graph compilation only after a training operation in a learning-owned process and checks the source/mode acknowledgement. A fresh ordinary product process retains its default mode. This narrows the investigated execution path; it does not identify a unique kernel-level cause or retroactively validate rejected E4/E5 candidates.

Twenty independent repetitions used the same candidate worker, fixed sources and seed 42 on Apple M6 / Mac18,5, 16 GiB RAM, macOS 27.0.1, MLX Swift 0.32.3 and Swift LM 3.32.3. The base was Qwen3.5-2B-derived affine Q4 group64, with BF16 core/expert tensors and FP32 Adam. Decoding was greedy with fresh per-request caches; filesystem caches were not purged. This hardware differs from E4/E5 and does not provide a matched performance comparison.

Each repetition checked 64 interleaved/grouped ABBA generations, 16 continuous repetitions and two full passes over 34 previously exposed records: 148 checked generations plus 2 warmups. Grouping was diagnostic only and was not adopted as the production observation order. All 2,960 checked observations matched their corresponding first-repetition full-record tokens, inputs, completion states and selected sources. There were also 40 warmups and 160 lifecycle generations, for 3,160 actual generation requests. The eight lifecycle generation stages agreed across repetitions.

Each lifecycle session used two approved TRAIN records and one frozen-teacher protection record, executing four Adam updates including one checkpoint replay and ending at step 3. All twenty sessions passed preparation equivalence, real updates, exact checkpoint replay, atomic cancellation, continuation after cancellation, saved-expert reload equality and frozen-source checks. Reload verification first generated in a fresh default product process, then checked the learning mode. Observation used the current width-32 expert; the independent lifecycle test used a historical width-64 parent. Thus the total of 80 executed updates, including 20 replays, is not one candidate trained continuously for 80 steps or twenty product-learning rounds.

All twenty repetitions completed in 386.70 seconds, including both subprocess drivers through exit and the explicit reload checks; the median repetition was 19.3892 seconds (range 18.6460-20.0591). No teacher calls or adoptions occurred. Frozen source-file states and production pointers remained unchanged, all workers closed and model locks were released. The installed application and current qualification proofs were not replaced. Child RSS samples and lifecycle current/peak metrics were retained privately; they are not an all-process or GPU peak benchmark.

These are fixed-seed reliability repetitions on exposed material, not semantic-accuracy tests, new knowledge gains, independent seed replications, universal determinism or a speedup measurement. Exact full-product-route qualification and longer-term evaluation remain necessary before this candidate becomes an installed release. This v0.2 edition includes the reliability results; the original v0.1 archive remains a separate historical version. Aggregate settings and results are in [the experiment ledger](../docs/experiments.json).

## Results

| Observation | Before | After | Interpretation |
| --- | --- | --- | --- |
| E1 training answers | 3/16 | 16/16 | Training fit; excluded from independent gain count |
| E1 development validation | 2/6 | 6/6 | Four new correct answers within the same two facts |
| E1 historical protection | 0 eligible records | 0 tested | No basis for a broad retention claim |
| E2 routed scored cases | 25/34 | 28/34 | Includes two newly correct training cases |
| E2 validation gain | - | +1, recorded forgetting 0 | Limited route-aware admission result |
| E3 first-install text probes | - | 3/3 expected answers | Inference/lifecycle smoke test, not a capability score |
| E4 completed FAQ prefix | 3 scheduled rounds reached | 408 Adam; 1 committed, 2 rejected | Fixed three-round prefix of an unfinished 20-round study |
| E4 committed round | VALID 1/4 | VALID 2/4; 6/6 history retained | One new validation record; other candidate gains were rejected |

E1 passed the implemented admission checks and was saved, reloaded, and retained. An earlier separate attempt used 256 updates across candidates but produced no validation net gain and was rejected. Two initial launch attempts for E1 failed with database-lock errors; a later automatic retry succeeded. This does not establish that concurrency failures were fixed or that the manual launch was fault-free.

In E2, a previous forced-expert result of 23/24 did not imply equivalent performance through the application router. Two validation questions still failed through the real route and remained unresolved. Reporting the 34-case improvement as three newly learned validation examples would be incorrect: two were training examples.

In E3, the factory installation answered a short arithmetic question and the project brand/model-name questions as expected. The public learning endpoint recognized the inherited source and correctly reported insufficient new material: 0/16 training and 0/2 validation. Personal counters were zero and no prior learning timestamp was assigned. This verifies startup and learning admission, not newly acquired knowledge. Intermediate builds failed on material inheritance and SQL column naming; those defects were corrected before the final smoke test.

In E4, baseline repeated observations differed on 0/26, 2/30 and 2/34 records; last-candidate observations differed on 0/26, 4/30 and 7/34. Round 2 baseline differences were on the core route, while round 3 baseline differences were on the accepted-expert route. Candidate differences occurred on both routes. Determinism and unchanged-core-output checks rejected all candidates in rounds 2 and 3. Inputs were identical, generation was greedy and caches were fresh; the underlying numerical/model-state cause remains unresolved. These observations do not isolate training-induced forgetting or prove that instability is exclusively in the core. Validation loss reductions alone did not qualify adoption.

The factory expert archive is **25,763,569 bytes** (about 25.8 decimal MB), including its required core/router artifacts and records. The application ZIP is **288,527,153 bytes**; this is the historical 0.3.217 application-only ZIP, with separate base, metadata and speech archives. A later 0.3.218 complete package bundles these archives for offline first installation; package availability and size are recorded in the download release, separately from E3. These file sizes are storage measurements, not resident-memory requirements. No controlled throughput or peak-memory benchmark is reported here.

## Limitations

E6 provides twenty same-seed reliability repetitions on a separate candidate and device, not a new learning-quality or transfer result. Its observed stability does not establish that the installed E4/E5 runtime is repaired.

E1 covers four validation paraphrases of two identity facts, alongside two already-correct basic-task checks. E4 adds a committed one-record gain on a four-fact FAQ set, but also two rejected rounds and unresolved repeated-output instability. Shared fact families, routing-based selection, repeated validation access, one seed, and missing multi-run uncertainty estimates limit interpretation. Generalization to unseen tasks, long-term stability, teacher-call savings, and superiority over fixed-capacity baselines remain unmeasured.

An empty protection pool cannot prove absence of forgetting. Even a nonempty rotating sample can miss regressions, and the net-gain rule explicitly permits some losses. Record-level answer matching also cannot establish robust knowledge or safety outside the tested prompts.

Dynamic creation and specialization of many independent experts, cold-expert scheduling benefits, automatic core consolidation, and sustained gains under matched budgets remain research directions. The accepted core artifact has historical provenance; it is not a newly demonstrated daily core-evolution result.

The public repository exposes this report and sanitized summaries, not the implementation, private learned material, full raw records, or internal workflow maps. Independent end-to-end reproduction is therefore currently unavailable. No claim of external audit or peer review is made. The application remains a beta; its isolated startup test does not substitute for testing on another machine, and its current macOS package is not notarized.

## Reproducibility

The [reproduction protocol and remaining evaluation](../docs/REPRODUCIBILITY.md) distinguishes repeatable functional steps from unavailable exact independent reproduction. The public [experiment ledger](../docs/experiments.json) records report/run identifiers, application versions, sample counts, known settings, observed outcomes, and missing measurements. It intentionally excludes local absolute paths, credentials, personal conversations, raw teacher responses, and private training records. Unknown hardware details or timings are left unknown rather than inferred from another experiment.

For a future independently reproducible evaluation, the project needs a releasable dataset with provenance and training rights, exact model/runtime revisions, complete hardware and decoding settings, a held-out final test, matched baseline budgets, multiple seeds, and task-by-time retention results. Teacher costs, storage, active parameters, resident tensors, process RSS, and device allocation should be measured and reported separately.

The planned comparisons are a frozen backbone, a fixed-capacity expert/adapter, and a growing expert system with matched training and teacher budgets. Ablations should isolate replay/protection, routing, and disk residency. These experiments are proposed, not completed results. This v0.2 report is a dated research preview; publishing it does not mark the application's scientific release gate as complete.

## References

1. Qwen team. [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B). Upstream language/vision foundation; upstream results are not Luban measurements.
2. MLX contributors. [MLX](https://github.com/ml-explore/mlx). Array framework for Apple Silicon.
3. Qwen team. [Qwen3-ASR-0.6B model card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B). Separate speech-recognition foundation.

### Version and license

Report v0.2, 23 October 2026; original v0.1 dated 22 October 2026. English is primary; Chinese is supplementary. Review remains pending. This report does not announce an application release.

Copyright 2026 vinson (麦好火电商研究). The report, Chinese supplement, architecture description and sanitized experiment summaries are licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). Application source code, model weights, private data and upstream works are excluded. Attribution: vinson, 麦好火电商研究, *Luban Technical Report v0.2*, 2026.
