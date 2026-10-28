# Luban Zero

**Application: public beta · Technical report: Research Preview**

[Website](https://ai.maihh.net/?lang=en) · [中文](README.md) · [Beta status](STATUS.md)

Luban Zero (luban0) is a continual-learning AI prototype for personal devices, starting from an approximately 2B language core. It explores local inference, routing, learning from authorized material, and forgetting control.

## Local generation speed observation

On 23 October 2026, a local response on an **Apple M6 Mac with 16 GB memory** displayed **94.1 tokens/s** generation speed: **96 input tokens**, **30 output tokens**, using the local core.

![UI observation on an M6 Mac with 16 GB memory: 96 input tokens, 30 output tokens, and 94.1 tokens/s generation speed](docs/images/m6-16gb-generation-94.1-20261023.png)

This is the decode speed of one short response, excluding input processing and the wait for the first token. It is not a multi-run average or end-to-end rate and does not establish training speedup or performance across tasks.

## Research Paper — Luban0

**Luban: Budget-Constrained Continual Learning with Disk-Resident Neural Experts**

v0.2 DOI: [10.5281/zenodo.23585073](https://doi.org/10.5281/zenodo.23585073)

[All versions (resolves to latest)](https://doi.org/10.5281/zenodo.23552844) · [Original v0.1 archive](https://doi.org/10.5281/zenodo.23552845); the earlier version remains separately available.

Status: **Technical Report / Preprint (not peer reviewed)**.

[📄 Read Paper](paper/Luban_Technical_Report.md) · [PDF](paper/Luban_Technical_Report.pdf) · [Chinese supplement](paper/Luban_Technical_Report_ZH.md) · [📊 Experimental Results](docs/EXPERIMENTS.md)

Report v0.2 (23 October 2026) describes the current MLX 4-bit architecture, expert learning, route-aware evaluation, net-gain admission, observed failures, and limitations. It is a **Technical Report — Research Preview / Work in Progress**, not submitted or peer reviewed. Author: vinson. Affiliation: 麦好火电商研究. Final manuscript review remains pending; no independent peer review is claimed. The report and sanitized summaries use [CC BY-NC-ND 4.0](LICENSE-REPORT.txt).

Current daily learning freezes the base and accepted core while training an expert candidate. The Evolution action uses this same expert-learning path. Adoption requires measured validation gains to exceed sampled forgetting, plus quality and persistence checks. Automatic specialization into many independent experts and sustained long-term improvement remain research directions.

A small study improved development-validation answers from 2/6 to 6/6 for two identity facts. Its historical protection pool was empty, so it does not demonstrate broad retention or generalization. The later FAQ study paused at18/20 actual rounds,2800 Adam updates,4 commits and14 quality rejections because no uncovered eligible independent validation records remained. The208/218/private219 phases are reported separately; this is neither20 completed rounds nor18 novel abilities. Read the [experiment summary](docs/EXPERIMENTS.md) and [reproduction protocol](docs/REPRODUCIBILITY.md) for conditions, instability and missing evaluation.

On 23 October, a separate 20/20-repetition isolated inference/lifecycle test passed: all 2,960 checked observations agreed; 80 actual updates included 20 checkpoint replays, with no adoption or teacher calls. This does not complete the earlier 18/20 learning study or establish new abilities or speedup. See the [experiment summary](docs/EXPERIMENTS.md).

## Product directions

- Local customer service and enterprise knowledge questions
- Adaptation to personal knowledge and work habits
- Simple automation tasks
- Studying continual learning and expert growth within finite resources

These are directions rather than completed capability promises. See [the current architecture](docs/ARCHITECTURE.md).

## Public scope

This repository publishes project information, the technical report, and sanitized aggregate results. Source code, personal data, complete raw research evidence, and internal workflow documents remain restricted. Publication grants no new source-code or model license; upstream components retain their own licenses.

Application **0.3.248 beta**: [complete installer with models](https://ai.maihh.net/downloads/Luban-0.3.248-macos-arm64-full.zip) · [GitHub small app and release notes](https://github.com/MaiHHConnect/lubanllm/releases/tag/v0.3.248-beta.1). The complete installer includes the base, metadata, speech and factory expert and is about 3.05 GB; keep 16 GB free before downloading. Extract it into Applications, open it and accept the terms; included models install and start automatically. The small GitHub archive excludes models. Requires macOS 26 or later and Apple Silicon; not notarized by Apple.

## 0.3.248 disk prefix cache (2026-10-28)

When the same long prompt prefix recurs, its state is restored from disk, including after a restart or model unload. Synthetic single-run check: a 56,000-token prompt took 24.3 s cold and about 0.9 s on a disk hit after restart; only 32 greedy tokens were compared, so this is not a quality evaluation. The cache matches an exact prefix only; changing one character at the start invalidates all of it. It lives in `~/Library/Application Support/Luban/prefix-cache`, readable only by you and excluded from backups, capped at 8 GiB; you can delete it, or set `LUBAN_PREFIX_CACHE_DIR=off` to disable it. In agent setups with many tool definitions the 2B model may still echo context or issue spurious tool calls.

## 0.3.246 API compatibility update (2026-10-28)

Adds Claude Messages and text tool-call compatibility, and improves memory admission for long conversations, exact prefix caching, and status reads during generation. New tool decisions in long histories can still fail; full 256K context and general Agent task completion have not been validated. The website now provides the 0.3.248 complete installer (0.3.246 and earlier complete installers have been taken down), while GitHub provides the application-only archive without models. The old Quark mirror has not been updated and its website button is temporarily disabled.
