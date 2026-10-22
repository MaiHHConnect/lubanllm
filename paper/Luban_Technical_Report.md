# Luban: Budget-Constrained Continual Learning with Disk-Resident Neural Experts

**Technical Report v0.1 - Research Preview / Work in Progress**

**Author: vinson | Affiliation: 麦好火电商研究**

22 October 2026 | Implementation reference: application 0.3.217 (build 225)

The author is responsible for the report. Completion of the final manuscript review has not been confirmed; no independent review is claimed. This report has not been submitted to a venue or peer reviewed. AI assistance was used to organize the report and implement parts of the prototype.

[PDF reading edition](Luban_Technical_Report.pdf) | [中文补充](Luban_Technical_Report_ZH.md) | [Experimental results](../docs/EXPERIMENTS.md)

### Abstract

Luban Zero investigates controlled learning on personal Apple Silicon devices using an approximately 2B language core, a shallow routing head, optional neural experts, and an external teacher. The current application uses a Qwen3.5-2B-derived MLX 4-bit model with an accepted core modification and a learned expert. Daily learning freezes the base/core, trains an expert candidate, evaluates generated answers through the actual local route, and accepts an update only when measured gains exceed sampled forgetting and additional quality checks pass. A small identity-fact experiment improved development validation answers from 2/6 to 6/6 after 48 Adam updates; its four newly correct validation prompts concerned those facts, while two existing basic-task checks were already correct. Its empty historical protection set prevents a claim of preserved general capability. This report describes the implemented system, bounded observations, failures, and outstanding evaluation. It does not establish broad transfer, an optimal training budget, or unlimited self-improvement.

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

Let **G** be previously incorrect development-validation records answered correctly after the update, and **F** be failed records from the sampled previously learned validation pool. The current net-gain rule requires **G - F > 0**. Training-set successes do not contribute to G. Additional checks include validation-loss improvement greater than 0.5%, deterministic output agreement, comparable inputs, finite values, and supported routing. This is a count-based acceptance rule, not a calibrated economic return-on-investment model.

The rule can accept partial learning and some sampled forgetting when gains exceed losses. Unlearned material remains available for later consideration; validation records are not silently converted into training records. A failed check retains the previous accepted state. Successful candidates must survive save/reload checks before their accepted state is committed. Source and engine bindings prevent an old validation record from automatically qualifying a different runtime.

Validation is consulted repeatedly during candidate selection. It is therefore **development validation**, not an untouched final test. Repeated answer agreement detects some instability but is not independent statistical replication or proof of semantic understanding.

### Lifecycle and first installation

Opening the application leaves the model stopped. Automatic learning requires a started, available model, automatic learning enabled, eligible material, and a valid source. Training shares the model resource with inference; requests may report that the model is busy.

A fresh installation imports a small factory package containing the accepted core modification, expert, router, and required acceptance records. These are starting capabilities, not a new user's personal learning achievements. Personal learning counts and last-learning timestamps start empty. Existing user data is not overwritten by first-install setup. Private full-model migration is separate from the public factory package.

## Experiments

We report three observations from 21 October 2026. They are local engineering studies, not a preregistered benchmark. Raw internal records support the summaries but are not included in this public repository. Sanitized aggregates are available in [the experiment summary](../docs/EXPERIMENTS.md) and [machine-readable data](../docs/experiments.json).

**E1 - Expert learning on two identity facts.** Candidate prompts concerned the project's model name and brand. The preparation stage generated 32 training and 12 validation candidates. Routing coverage, without examining correctness, selected 16 training and 4 validation prompts; the application added 2 existing, previously correct basic-task checks (case conversion and simple logic). The four new validation prompts were paraphrases of the trained identity facts, not disjoint task families. Questions were assistant-prepared from user-authorized facts; no new paid teacher calls were made for this run.

The installed application was 0.3.208 (build 216), using the MLX 4-bit path on the project's Apple Silicon machine. The run used seed 42 and a width-32 expert candidate. It performed 48 Adam updates. Recorded duration was 64.58 seconds from baseline preparation through training and intermediate verification up to report construction. It excludes final publication/reload and is not the full interval from the user's click. The record does not establish peak memory. The historical protection set was empty.

**E2 - Route-aware adoption of a previously trained candidate.** An existing 64-update candidate was reevaluated without further optimization of the adopted candidate. Separate runtime-qualification checks, including actual training checks, were rerun. Forty cases were observed, of which 34 had scoring targets. Actual routing was used for before/after comparison; six unscored cases were diagnostic only. Direct expert output and routed output were recorded separately. This was a narrow migration/admission exercise, not a fresh blinded evaluation or a new candidate-training run.

**E3 - Factory installation and private migration.** Application 0.3.217 was launched in an isolated home with access to the development environment and existing application data denied. A fresh factory installation was explicitly started and tested on three short text prompts. A separate private migration roundtrip verified imported artifacts and two identity answers. Both studies were conducted on the existing machine, not on a second physical Mac. Neither performed new learning updates.

## Results

| Observation | Before | After | Interpretation |
| --- | --- | --- | --- |
| E1 training answers | 3/16 | 16/16 | Training fit; excluded from independent gain count |
| E1 development validation | 2/6 | 6/6 | Four new correct answers within the same two facts |
| E1 historical protection | 0 eligible records | 0 tested | No basis for a broad retention claim |
| E2 routed scored cases | 25/34 | 28/34 | Includes two newly correct training cases |
| E2 validation gain | - | +1, recorded forgetting 0 | Limited route-aware admission result |
| E3 first-install text probes | - | 3/3 expected answers | Inference/lifecycle smoke test, not a capability score |

E1 passed the implemented admission checks and was saved, reloaded, and retained. An earlier separate attempt used 256 updates across candidates but produced no validation net gain and was rejected. Two initial launch attempts for E1 failed with database-lock errors; a later automatic retry succeeded. This does not establish that concurrency failures were fixed or that the manual launch was fault-free.

In E2, a previous forced-expert result of 23/24 did not imply equivalent performance through the application router. Two validation questions still failed through the real route and remained unresolved. Reporting the 34-case improvement as three newly learned validation examples would be incorrect: two were training examples.

In E3, the factory installation answered a short arithmetic question and the project brand/model-name questions as expected. The public learning endpoint recognized the inherited source and correctly reported insufficient new material: 0/16 training and 0/2 validation. Personal counters were zero and no prior learning timestamp was assigned. This verifies startup and learning admission, not newly acquired knowledge. Intermediate builds failed on material inheritance and SQL column naming; those defects were corrected before the final smoke test.

The factory expert archive is **25,763,569 bytes** (about 25.8 decimal MB), including its required core/router artifacts and records. The application ZIP is **288,527,153 bytes**; the MLX base, metadata, and speech archives remain separate downloads. These file sizes are storage measurements, not resident-memory requirements. No controlled throughput or peak-memory benchmark is reported here.

## Limitations

The positive learning result covers four validation paraphrases of two identity facts, alongside two already-correct basic-task checks (six development-validation prompts in total). Shared fact families, routing-based selection, repeated validation access, one seed, and missing multi-run uncertainty estimates limit interpretation. Generalization to unseen tasks, long-term stability, teacher-call savings, and superiority over fixed-capacity baselines remain unmeasured.

An empty protection pool cannot prove absence of forgetting. Even a nonempty rotating sample can miss regressions, and the net-gain rule explicitly permits some losses. Record-level answer matching also cannot establish robust knowledge or safety outside the tested prompts.

Dynamic creation and specialization of many independent experts, cold-expert scheduling benefits, automatic core consolidation, and sustained gains under matched budgets remain research directions. The accepted core artifact has historical provenance; it is not a newly demonstrated daily core-evolution result.

The public repository exposes this report and sanitized summaries, not the implementation, private learned material, full raw records, or internal workflow maps. Independent end-to-end reproduction is therefore currently unavailable. No claim of external audit or peer review is made. The application remains a beta; its isolated startup test does not substitute for testing on another machine, and its current macOS package is not notarized.

## Reproducibility

The public [experiment ledger](../docs/experiments.json) records report/run identifiers, application versions, sample counts, known settings, observed outcomes, and missing measurements. It intentionally excludes local absolute paths, credentials, personal conversations, raw teacher responses, and private training records. Unknown hardware details or timings are left unknown rather than inferred from another experiment.

For a future independently reproducible evaluation, the project needs a releasable dataset with provenance and training rights, exact model/runtime revisions, complete hardware and decoding settings, a held-out final test, matched baseline budgets, multiple seeds, and task-by-time retention results. Teacher costs, storage, active parameters, resident tensors, process RSS, and device allocation should be measured and reported separately.

The planned comparisons are a frozen backbone, a fixed-capacity expert/adapter, and a growing expert system with matched training and teacher budgets. Ablations should isolate replay/protection, routing, and disk residency. These experiments are proposed, not completed results. This v0.1 report is a dated research preview; publishing it does not mark the application's scientific release gate as complete.

## References

1. Qwen team. [Qwen3.5-2B model card](https://huggingface.co/Qwen/Qwen3.5-2B). Upstream language/vision foundation; upstream results are not Luban measurements.
2. MLX contributors. [MLX](https://github.com/ml-explore/mlx). Array framework for Apple Silicon.
3. Qwen team. [Qwen3-ASR-0.6B model card](https://huggingface.co/Qwen/Qwen3-ASR-0.6B). Separate speech-recognition foundation.

### Version and license

Report v0.1, 22 October 2026. English is primary; Chinese is supplementary. Review remains pending. This report does not announce an application release.

Copyright 2026 vinson (麦好火电商研究). The report, Chinese supplement, architecture description and sanitized experiment summaries are licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). Application source code, model weights, private data and upstream works are excluded. Attribution: vinson, 麦好火电商研究, *Luban Technical Report v0.1*, 2026.
